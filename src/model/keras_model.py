from keras.callbacks import CSVLogger, ReduceLROnPlateau
from keras.callbacks import LearningRateScheduler
import pdb
from src.utils.package_utils import OnAllValDataEarlyStopping, MTOnAllValDataEarlyStopping
from src.utils.package_utils import get_imbalance_data, LossAdaptation
from keras.models import Model
import numpy as np
from src.utils.DB_utils import LIST_MULTIOUT_DATASETS, LIST_CLF_DATASETS
from src.utils.DB_utils import LIST_AA_DATASETS, LIST_MTDTI_DATASETS
from tqdm import tqdm
import tensorflow as tf
import keras.backend as K


###################################
# TensorFlow wizardry
config = tf.ConfigProto()

# Don't pre-allocate memory; allocate as-needed
config.gpu_options.allow_growth = True

# Only allow a total of half the GPU memory to be allocated
config.gpu_options.per_process_gpu_memory_fraction = 0.6

# Create a session with the above options specified.
K.tensorflow_backend.set_session(tf.Session(config=config))
###################################


dict_shortcut = {'batch_size': 'bs', 'n_epochs': 'ne', 'init_lr': 'lri',
                 'patience_early_stopping': 'pa', 'lr_scheduler': 'lrs', 'predBN': 'prB',
                 'pred_layers': 'prl', 'pred_dropout': 'prd', 'pred_reg': 'prr',
                 'n_steps': 'nconv', 'agg_nei': 'aggn', 'agg_all': 'agga', 'mol_BN': 'mB',
                 'mol_dropout': 'md', 'mol_reg': 'mr', 'prot_BN': 'pB',
                 'prot_dropout': 'pd', 'prot_reg': 'pr'}


class STModel():

    def __init__(self, batch_size, n_epochs, init_lr, patience_early_stopping, lr_scheduler,
                 pred_layers, pred_BN, pred_dropout, pred_reg,
                 cpu_workers_for_generator, queue_size_for_generator,
                 # train_gen, val_gen, test_gen,
                 dataset,
                 enc_dict_param):
        # learning parameters
        self.batch_size = batch_size  # int
        self.n_epochs = n_epochs  # int
        self.init_lr = init_lr  # float
        self.patience_early_stopping = patience_early_stopping  # int
        self.lr_scheduler = lr_scheduler  # dict: contain name and other params

        # pred architecture parameters
        self.pred_layers = pred_layers  # list of nb of units per layer (except the last one)
        self.pred_BN = pred_BN  # True or False : batch normalization or not
        self.pred_dropout = pred_dropout  # None, or float : fraction of the input units to drop
        self.pred_reg = pred_reg  # float, weight in l2 reg

        # data generator capacities
        self.cpu_workers_for_generator = cpu_workers_for_generator  # int
        self.queue_size_for_generator = queue_size_for_generator  # int

        # data generators associated with the exp.
        # self.train_gen = train_gen
        # self.val_gen = val_gen
        # self.test_gen = test_gen

        # dataset parameters
        self.dataset = dataset

        # encoder parameters
        self._init_encoder(enc_dict_param)

    def __str__(self):
        s = []
        for cle, val in self.__dict__.items():
            if not cle.startswith('__') and \
                    cle not in ['build', '_init_encoder', 'lr_schedule', 'fit',
                                'predict', 'evaluate', 'model', 'output', 'embedding', 'dataset',
                                'cpu_workers_for_generator', 'queue_size_for_generator',
                                'n_att_atom', 'emb_size', 'mol_embedding', 'prot_embedding']:
                if cle in dict_shortcut.keys():
                    cle = dict_shortcut[cle]
                s.append(str(cle) + ':' + str(val).replace(' ', ''))
        return ';'.join(s)

    def _init_encoder(self, enc_dict_param):
        pass

    def build(self,):
        pass

    def unfreeze_layers(self, n_epoch):
        print('BEFORE trainable weights')
        for layer in self.model.trainable_weights:
            print(layer.name, layer.trainable)

        list_del = []
        for iu, unfreeze_type in enumerate(self.list_type_unfreeze):
            if n_epoch == self.list_nepoch_unfreeze[iu]:
                for layer in self.model.layers:
                    name = layer.name
                    if (unfreeze_type == 'prot' and ('pconv' in name or 'biLSTM' in name)) or \
                            (unfreeze_type == 'mol' and ('emb_conv' in name or 'emb_nei' in name or
                                                         'agg' in name)) or \
                            (unfreeze_type == 'pred' and ('pred_layer' in name or
                                                          'BNpred' in name)):
                        layer.trainable = True
                        print(layer.name, 'becomes trainable')
                list_del.append(iu)
        for iu in np.sort(list_del)[::-1].tolist():
            del self.list_type_unfreeze[iu]
            del self.list_nepoch_unfreeze[iu]
        self.model.compile(optimizer=self.optimizer, loss={'output': self.dataset.loss},
                           loss_weights={'output': 1.}, metrics={'output': self.output_metrics})

        print('AFTER trainable weights')
        for layer in self.model.trainable_weights:
            print(layer.name, layer.trainable)

    def get_callbacks(self, val_gen, val_y, train_gen, train_val):
        # callbacks when fitting
        early_stopping = OnAllValDataEarlyStopping(
            self.dataset.name, 'gen', val_gen, val_y, train_gen, train_val,
            self.queue_size_for_generator, self.cpu_workers_for_generator,
            monitor=self.dataset.ea_metric, mode=self.dataset.ea_mode,
            min_delta=0,  # minimum change to qualify as an improvement
            patience=self.patience_early_stopping, verbose=1, restore_best_weights=True)
        csv_logger = CSVLogger('.tmp/temp.log')  # streams epoch results to a csv
        callbacks = [early_stopping, csv_logger]

        if self.lr_scheduler['name'] == 'ReduceLROnPlateau':
            lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2,
                                             verbose=1, mode='auto',
                                             min_delta=0.0001, min_lr=0.001)
            callbacks += [lr_scheduler]
        elif self.lr_scheduler['name'] == 'LearningRateScheduler':
            rate = self.lr_scheduler['rate']
            # reducing the learning rate by "rate" every 2 epochs
            lr_scheduler = LearningRateScheduler(
                lambda epoch: self.init_lr * rate ** (epoch // 2), verbose=0)
            callbacks += [lr_scheduler]

        return callbacks

    def fit(self, train_gen, val_gen, train_val, val_y, ratio_tr, debug=False):
        # from keras.utils import plot_model
        # plot_model(self.model, to_file='model.png')
        # exit(1)
        # import pdb; pdb.Pdb().set_trace()

        callbacks = self.get_callbacks(val_gen, val_y, train_gen, train_val)

        # balance predictor in case of uneven class distribution in train data
        # inverse_class_proportion = get_imbalance_data(train_val, self.dataset.name)
        inverse_class_proportion = None
        if inverse_class_proportion is not None:
            print('WARNING: imbalance class proportion ', inverse_class_proportion)

        # fit model on train data
        if self.dataset.name in LIST_AA_DATASETS:
            early_stopping = callbacks[0]
            early_stopping.model = self.model
            early_stopping.on_train_begin()
            for n_epoch in range(self.n_epochs):
                for i_tr in tqdm(range(len(train_gen))):
                    bx, by = train_gen.__getitem__(i_tr)
                    bx = [np.swapaxes(bx[0], 0, 2), np.repeat(bx[1], bx[1][0, 0], 0)]
                    # if i_tr == 53:
                    #     pdb.Pdb().set_trace()
                    temp_res = self.model.train_on_batch(bx, by,
                                                         class_weight=inverse_class_proportion)
                early_stopping.on_epoch_end(n_epoch)
            early_stopping.on_train_end()
        elif debug is False:
            # on train begin
            init_e = 0
            while len(self.list_nepoch_unfreeze) > 0:
                print('FIT WITH UNFREEZING ', self.list_nepoch_unfreeze)
                ne_curr = min(self.list_nepoch_unfreeze)
                self.model.fit_generator(train_gen, steps_per_epoch=None, epochs=ne_curr,
                                         verbose=1,
                                         callbacks=callbacks, validation_data=None,
                                         validation_steps=None,
                                         class_weight=inverse_class_proportion,
                                         max_queue_size=self.queue_size_for_generator,
                                         workers=self.cpu_workers_for_generator,
                                         use_multiprocessing=True,
                                         shuffle=True, initial_epoch=init_e)
                self.unfreeze_layers(ne_curr)
                init_e = ne_curr

            self.model.fit_generator(train_gen, steps_per_epoch=None, epochs=self.n_epochs,
                                     verbose=1,
                                     callbacks=callbacks, validation_data=None,
                                     validation_steps=None,
                                     class_weight=inverse_class_proportion,
                                     max_queue_size=self.queue_size_for_generator,
                                     workers=self.cpu_workers_for_generator,
                                     use_multiprocessing=True,
                                     shuffle=True, initial_epoch=init_e)

        else:
            for e in range(self.n_epochs):
                vizualised_layers = [layer.output for layer in self.model.layers
                                     if 'in_' not in layer.name]
                vizualised_layers = [layer.output for layer in self.model.layers
                                     if ("0x2" in layer.name)]
                vizualised_layers = [layer.output for layer in self.model.layers
                                     if ("output" in layer.name or "embedding" in layer.name)]
                self.model.metrics_tensors += vizualised_layers
                for i_tr in range(len(train_gen)):
                    # if self.dataset.name in LIST_MULTIOUT_DATASETS and \
                    #         self.dataset.name in LIST_CLF_DATASETS:
                    #     bx, by, bsw = train_gen.__getitem__(i_tr)

                    # else:
                    bx, by = train_gen.__getitem__(i_tr)
                    bsw = None
                    temp_res = self.model.train_on_batch(bx, by,
                                                         class_weight=inverse_class_proportion,
                                                         sample_weight=bsw)
                    if i_tr < 10:
                        print(i_tr)
                        pdb.Pdb().set_trace()

        # print(hist.history)
        # import pdb; pdb.Pdb().set_trace()

    def predict(self, test_gen):
        print('PREDICT')
        pred_model = Model(inputs=self.model.input,
                           outputs=[self.model.get_layer('output').output,
                                    self.model.get_layer('embedding').output])

        if self.dataset.name in LIST_AA_DATASETS:
            y_pred, embedding = [], []
            for i_te in tqdm(range(len(test_gen))):
                bx, by = test_gen.__getitem__(i_te)
                bx = [np.swapaxes(bx[0], 0, 2), np.repeat(bx[1], bx[1][0, 0], 0)]
                # pdb.Pdb().set_trace()
                temp_pred, temp_emb = pred_model.predict_on_batch(bx)
                y_pred.append(temp_pred)
                embedding.append(temp_emb)
            y_pred, embedding = np.concatenate(y_pred, axis=0), np.concatenate(embedding, axis=0)
        else:
            y_pred, embedding = pred_model.predict_generator(
                test_gen, steps=None, max_queue_size=self.queue_size_for_generator,
                workers=self.cpu_workers_for_generator, use_multiprocessing=True, verbose=1)
        return y_pred, embedding

    # def evaluate(self, test_gen):
    #     visualized_layers = ['embedding', 'output']
    #     self.model.metrics_tensors += [layer.output for layer in self.model.layers
    #                                    if layer.name in visualized_layers]
    #     # perf, emb, pred =
    #     perf = self.model.evaluate_generator(
    #         test_gen, steps=None, max_queue_size=self.queue_size_for_generator,
    #         workers=self.cpu_workers_for_generator, use_multiprocessing=True, verbose=1)
    #     import pdb; pdb.Pdb().set_trace()
    #     # return perf, emb, pred

    #     # self.model.evaluate_generator(test_gen, steps=None, max_queue_size=self.queue_size_for_generator,workers=self.cpu_workers_for_generator, use_multiprocessing=True, verbose=1)


class MTModel(STModel):

    def get_callbacks(self, val_gen, val_y, train_gen, train_val):
        # callbacks when fitting
        early_stopping = MTOnAllValDataEarlyStopping(
            self.dataset.name, 'gen', val_gen, val_y, train_gen, train_val,
            monitor=self.dataset.ea_metric, mode=self.dataset.ea_mode,
            min_delta=0,  # minimum change to qualify as an improvement
            patience=self.patience_early_stopping, verbose=1, restore_best_weights=True)
        csv_logger = CSVLogger('.tmp/temp.log')  # streams epoch results to a csv

        loss_adaptation = LossAdaptation(self.MT_lossw, verbose=1)

        callbacks = [early_stopping, csv_logger, loss_adaptation]

        if self.lr_scheduler['name'] == 'ReduceLROnPlateau':
            lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2,
                                             verbose=1, mode='auto',
                                             min_delta=0.0001, min_lr=0.001)
            callbacks += [lr_scheduler]
        elif self.lr_scheduler['name'] == 'LearningRateScheduler':
            rate = self.lr_scheduler['rate']
            # reducing the learning rate by "rate" every 2 epochs
            lr_scheduler = LearningRateScheduler(
                lambda epoch: self.init_lr * rate ** (epoch // 2), verbose=0)
            callbacks += [lr_scheduler]

        return callbacks

    def fit(self, train_gen, val_gen, train_val, val_y, ratio_tr, debug=False):
        # from keras.utils import plot_model
        # plot_model(self.model, to_file='model.png')
        # exit(1)
        # import pdb; pdb.Pdb().set_trace()

        callbacks = self.get_callbacks(val_gen, val_y, train_gen, train_val)

        # balance predictor in case of uneven class distribution in train data
        # inverse_class_proportion = get_imbalance_data(train_val, self.dataset.name)
        inverse_class_proportion = None
        if inverse_class_proportion is not None:
            print('WARNING: imbalance class proportion ', inverse_class_proportion)

        # fit model on train data
        if debug is False:
            self.model.fit_generator(train_gen, steps_per_epoch=None, epochs=self.n_epochs,
                                     verbose=1,
                                     callbacks=callbacks, validation_data=val_gen,
                                     validation_steps=None,
                                     class_weight=inverse_class_proportion,
                                     max_queue_size=self.queue_size_for_generator,
                                     workers=self.cpu_workers_for_generator,
                                     use_multiprocessing=True,
                                     shuffle=True, initial_epoch=0)
        else:
            for e in range(self.n_epochs):
                vizualised_layers = [layer.output for layer in self.model.layers
                                     if ("output" in layer.name or "embedding" in layer.name)]
                self.model.metrics_tensors += vizualised_layers
                for i_tr in range(len(train_gen)):
                    bx, by = train_gen.__getitem__(i_tr)
                    temp_res = \
                        self.model.train_on_batch(bx, by, class_weight=inverse_class_proportion)
                    if i_tr < 10:
                        print(i_tr)
                        pdb.Pdb().set_trace()

        # print(hist.history)
        # import pdb; pdb.Pdb().set_trace()

    def predict(self, test_gen):
        print('PREDICT')
        pred_model = Model(inputs=self.model.input,
                           outputs=[self.model.get_layer('dti_output').output,
                                    self.model.get_layer('dti_embedding').output])

        y_pred, embedding = pred_model.predict_generator(
            test_gen, steps=None, max_queue_size=self.queue_size_for_generator,
            workers=self.cpu_workers_for_generator, use_multiprocessing=True, verbose=1)
        return y_pred, embedding
