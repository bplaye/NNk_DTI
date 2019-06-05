from sklearn import metrics
import keras.utils
import numpy as np
# from keras.callbacks import Callback
from tqdm import tqdm
from src.utils.DB_utils import LIST_AA_DATASETS, MISSING_DATA_MASK_VALUE
from src.utils.DB_utils import LIST_MULTICLF_DATASETS, LIST_REGR_DATASETS
from src.utils.DB_utils import LIST_CLF_DATASETS, LIST_MULTIOUT_DATASETS
import collections
import os
import sys
import keras.backend as K
from keras.models import Model


def get_lr_metric(optimizer):
    def lr(y_true, y_pred):
        return optimizer.lr
    return lr


def get_lossw_metric(dti_lossw, pcba_lossw):
    def lossw(y_true, y_pred):
        return (K.get_value(dti_lossw), K.get_value(pcba_lossw))
    return lossw


def my_to_categorical(list_temp, dataname):
    if dataname == "SecondaryStructure":
        n_classes = 8
    elif dataname == "TransmembraneRegions":
        n_classes = 3
    elif dataname == "CellLoc":
        n_classes = 11
    elif dataname == "SCOPe":
        n_classes = 7
    temp = np.zeros((len(list_temp), n_classes))
    for iel, el in enumerate(list_temp):
        temp[iel, el] = 1
    return temp


def check_multiclass_y(y_):
    set_batch = set(np.array(y_).flatten().tolist())
    if None in set_batch:
        set_batch.remove(None)
    if MISSING_DATA_MASK_VALUE in set_batch:
        set_batch.remove(MISSING_DATA_MASK_VALUE)
    return True if len(set_batch) > 2 else False


def normalise_class_proportion(class_proportion):
    maxx = float(max(class_proportion.values()))
    for cle in class_proportion.keys():
        class_proportion[cle] = maxx / float(class_proportion[cle])
    return class_proportion


def get_imbalance_data(y_, dataname):
    # import pdb; pdb.Pdb().set_trace()
    if dataname in LIST_MULTICLF_DATASETS:
        if dataname in LIST_AA_DATASETS:
            y_ = [el for ll in y_ for el in ll]
        class_proportion = normalise_class_proportion(dict(collections.Counter(y_)))
        return class_proportion
    elif dataname in LIST_MULTIOUT_DATASETS and dataname in LIST_CLF_DATASETS:
        # multi output pb
        # class_proportion = {0: 0, 1: 0}
        # for iout in range(y_.shape[1]):
        #     c = collections.Counter(y_[:, iout])
        #     class_proportion[0] += c[0]
        #     class_proportion[1] += c[1]
        # if class_proportion[0] != class_proportion[1]:
        #     class_proportion = normalise_class_proportion(class_proportion)
        #     print('WARNING: imbalance class proportion ', class_proportion)
        #     return class_proportion
        # else:
            return None
    elif dataname in LIST_CLF_DATASETS:
        class_proportion = normalise_class_proportion(dict(collections.Counter(y_)))
        return class_proportion


def sround(score):
    return round(score * 100, 2)


def get_clf_perf(y_pred, y_true,
                 perf_names=['AUPR', 'ROCAUC', 'F1', 'Recall', "Precision", "ACC"]):
    dict_perf_fct = {'AUPR': aupr, 'ROCAUC': rocauc, 'F1': f1,
                     'Recall': recall, "Precision": precision, "ACC": acc}
    dict_perf = {}
    # import pdb; pdb.Pdb().set_trace()
    for perf_name in perf_names:
        if perf_name in ['F1', 'Recall', "Precision", "ACC"]:
            y_pred_ = y_pred.copy()
            if len(y_pred_.shape) > 1 and y_pred_.shape[1] != 1:
                y_pred_ = (y_pred_ == y_pred_.max(axis=1)[:, None]).astype(int)
            else:
                y_pred_ = np.round(y_pred_)
            # import pdb; pdb.Pdb().set_trace()
        else:
            y_pred_ = y_pred
        dict_perf[perf_name] = dict_perf_fct[perf_name](y_true, y_pred_)
    return dict_perf


def get_regr_perf(y_pred, y_true,
                  perf_names=['MSE']):
    dict_perf_fct = {'MSE': mse}
    dict_perf = {}
    for perf_name in perf_names:
        dict_perf[perf_name] = dict_perf_fct[perf_name](y_true, y_pred)

    return dict_perf


def display_perf(earlystopping_object, dict_perf):
    if earlystopping_object.pb is 'clf':
        earlystopping_object.val_auprs.append(dict_perf['AUPR'])
        earlystopping_object.val_rocaucs.append(dict_perf['ROCAUC'])
        earlystopping_object.val_accs.append(dict_perf['ACC'])
        earlystopping_object.val_f1s.append(dict_perf['F1'])
        earlystopping_object.val_recalls.append(dict_perf['Recall'])
        earlystopping_object.val_precisions.append(dict_perf['Precision'])
        # import pdb; pdb.Pdb().set_trace()
        print("##### — AUPR: " + str(dict_perf['AUPR'][0]) +
              " — ROCAUC: " + str(dict_perf['ROCAUC'][0]) +
              " — ACC: " + str(dict_perf['ACC'][0]) + " — f1: " + str(dict_perf['F1'][0]) +
              " — Precision: " + str(dict_perf['Precision'][0]) +
              " — Recall " + str(dict_perf['Recall'][0]) + " #####")
        if len(dict_perf['AUPR'][1]) > 1:
            print("— AUPR: " + str(dict_perf['AUPR'][1]))
            print("— ROCAUC: " + str(dict_perf['ROCAUC'][1]))
            print("— ACC: " + str(dict_perf['ACC'][1]))
            print("— f1: " + str(dict_perf['F1'][1]))
            print("— Precision: " + str(dict_perf['Precision'][1]))
            print("— Recall " + str(dict_perf['Recall'][1]))
    elif earlystopping_object.pb is 'regr':
        earlystopping_object.val_mses.append(dict_perf['MSE'])
        print("— val_mse: " + str(dict_perf['MSE'][0]))
        if len(dict_perf['MSE'][1]) > 1:
            print("— val_mse: " + str(dict_perf['MSE'][1]))


class LossAdaptation(keras.callbacks.Callback):
    def __init__(self, adapt_type, dti_lossw, pcba_lossw, verbose):
        super(keras.callbacks.Callback, self).__init__()
        self.adapt_type = adapt_type
        self.verbose = verbose
        self.dti_lossw, self.pcba_lossw = dti_lossw, pcba_lossw

    def on_batch_end(self, batch, logs={}):
        if self.adapt_type == 'None':
            K.set_value(self.dti_lossw, K.get_value(self.dti_lossw))
            K.set_value(self.pcba_lossw, K.get_value(self.pcba_lossw))
        elif self.adapt_type == 'GradNorm':
            K.set_value(self.alpha, K.get_value(self.alpha) / 1.5)
            K.set_value(self.beta, K.get_value(self.beta) * 1.5)


class unfreeze_callback(keras.callbacks.Callback):
    def __init__(self, _model, n_epochs_unfreeze, unfreeze_part, optimizer, dataset, output,
                 verbose=1):
        super(keras.callbacks.Callback, self).__init__()
        self._model = _model
        self.n_epochs_unfreeze = n_epochs_unfreeze
        self.unfreeze_part = unfreeze_part
        self.optimizer = optimizer
        self.dataset = dataset
        self.output = output
        self.verbose = verbose

    def unfreeze_layers(self):
        for layer in self._model.layers:
            name = layer.name
            if (self.unfreeze_part == 'prot' and ('pconv' in name or 'biLSTM' in name)) or \
                    (self.unfreeze_part == 'mol' and ('emb_conv' in name or 'emb_nei' in name or
                                                      'agg' in name)) or \
                    (self.unfreeze_part == 'pred' and ('pred_layer' in name or 'BNpred' in name)):
                layer.trainable = True
                if self.verbose:
                    print(layer.name, 'becomes trainable')
        self._model.compile(optimizer=self.optimizer, loss={'output': self.dataset.loss},
                            loss_weights={'output': 1.}, metrics={'output': self.output})

    def on_train_begin(self, logs=None):
        if self.n_epochs_unfreeze == 1:
            self.unfreeze_layers()
            self.delayed = False
        else:
            self.delayed = True

    def on_epoch_end(self, epoch, logs=None):
        for layer in self._model.trainable_weights:
            print(layer.name, layer.trainable)
        print(self._model._collected_trainable_weights)
        if epoch == self.n_epochs_unfreeze - 1 and self.delayed is True:
            self.unfreeze_layers()


class OnAllValDataEarlyStopping(keras.callbacks.Callback):
    def __init__(self, dataname, gen_bool, val_gen, val_y, train_gen, train_y,
                 qsize, workers,
                 monitor, mode, patience, min_delta=0,
                 restore_best_weights=False, verbose=0):
        super(keras.callbacks.Callback, self).__init__()

        self.dataname = dataname
        self.gen_bool = True if gen_bool == 'gen' else False
        self.validation_data_generator = val_gen
        self.validation_target = val_y
        self.train_data_generator = train_gen
        self.train_target = train_y

        self.qsize, self.workers = qsize, workers

        self.monitor = monitor
        self.mode = mode
        self.min_delta = min_delta
        self.verbose = verbose
        self.patience = patience
        self.restore_best_weights = restore_best_weights  # if true, save best_model at the end
        if restore_best_weights:
            self.best_weights = None

        self.best = 0 if mode == 'max' else 10000000 if mode == 'min' else None
        self.best_dict_perf = None
        self.counter = 0
        self.stopped_epoch = 0

        if mode == 'min':
            self.monitor_op = np.less
        elif mode == 'max':
            self.monitor_op = np.greater

        if self.monitor in ['aupr', 'rocauc', 'acc', 'f1']:
            self.pb = 'clf'
            self.val_auprs = []
            self.val_rocaucs = []
            self.val_accs = []
            self.val_f1s = []
            self.val_recalls = []
            self.val_precisions = []
        elif self.monitor in ['mse']:
            self.pb = 'regr'
            self.val_mses = []

    def get_score(self):
        if self.dataname in LIST_AA_DATASETS:
            if self.gen_bool:
                y_pred = []
                for i_val in tqdm(range(len(self.validation_data_generator))):
                    bx, by = self.validation_data_generator.__getitem__(i_val)
                    bx = [np.swapaxes(bx[0], 0, 2), np.repeat(bx[1], bx[1][0, 0], 0)]
                    # pdb.Pdb().set_trace()
                    temp_pred = self.model.predict_on_batch(bx)
                    y_pred.append(temp_pred)
                prediction = np.concatenate(y_pred, axis=0)
            else:
                prediction = self.model.predict(self.validation_data_generator)
            target = np.concatenate([my_to_categorical(ll, self.dataname)
                                     for ll in self.validation_target], axis=0)
            # import pdb; pdb.Pdb().set_trace()
        else:
            if self.gen_bool:
                prediction = \
                    np.asarray(self.model.predict_generator(
                        self.validation_data_generator, max_queue_size=round(self.qsize),
                        workers=round(self.workers), use_multiprocessing=True, verbose=0))
                # prediction = \
                #     np.asarray(self.model.predict_generator(
                #         self.validation_data_generator, use_multiprocessing=True, verbose=0))
            else:
                prediction = \
                    np.asarray(self.model.predict(self.validation_data_generator))
            target = self.validation_target

        print('######## get perf')
        if self.dataname in LIST_REGR_DATASETS:
            dict_perf = get_regr_perf(prediction, target)
        elif self.dataname in LIST_CLF_DATASETS:
            dict_perf = get_clf_perf(prediction, target)

        print('######## print perf')
        display_perf(self, dict_perf)

        if self.monitor == 'aupr':
            return dict_perf['AUPR'][0], dict_perf
        elif self.monitor == 'rocauc':
            return dict_perf['ROCAUC'][0], dict_perf
        elif self.monitor == 'acc':
            return dict_perf['ACC'][0], dict_perf
        elif self.monitor == 'f1':
            return dict_perf['F1'][0], dict_perf
        elif self.monitor == 'mse':
            return dict_perf['MSE'][0], dict_perf

    def on_train_begin(self, logs=None):
        # Allow instances to be re-used
        self.counter = 0
        self.stopped_epoch = 0

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0 and self.verbose > 0:
            print('(ES) Epoch %d: early stopping' % (self.stopped_epoch + 1))
            self.model.set_weights(self.best_weights)
            save_model_weights(self.model, '.tmp/' + str(self.model))

    def on_epoch_end(self, epoch, logs=None):
        # plot and record score
        print('######## get score')
        mean_score, dict_perf = self.get_score()

        # prediction = np.asarray(self.model.predict_generator(self.train_data_generator))
        # dict_perf = get_clf_perf(prediction, self.train_target)
        # print("TRAIN ##### — AUPR: " + str(dict_perf['AUPR'][0]) +
        #       " — ROCAUC: " + str(dict_perf['ROCAUC'][0]) +
        #       " — MCC: " + str(dict_perf['MCC'][0]) + " — f1: " + str(dict_perf['F1'][0]) +
        #       " — Precision: " + str(dict_perf['Precision'][0]) +
        #       " — Recall " + str(dict_perf['Recall'][0]) + " #####")

        # do early stopping
        if self.monitor_op(mean_score - self.min_delta, self.best):
            if self.verbose > 0:
                print("(ES) Epoch %d: validation perf. increasing: %f" %
                      (epoch, mean_score - self.best))
            self.counter = 0
            self.best = mean_score
            self.best_dict_perf = dict_perf
            self.best_weights = self.model.get_weights()
        else:
            self.counter += 1
            if self.verbose > 0:
                print("(ES) Epoch %d: validation perf. decreasing: %f ; counter: %d" %
                      (epoch, mean_score - self.best, self.counter))
        print('##############################################')
        if self.counter == self.patience:
            self.model.stop_training = True
            self.stopped_epoch = epoch
            if self.verbose > 0:
                print("(ES) Epoch %d: early stopping ; end perf.: %f ; counter: %d" %
                      (epoch, mean_score, self.counter))
                print("BEST DICT PERF")
                display_perf(self, self.best_dict_perf)
            if self.restore_best_weights:
                if self.verbose > 0:
                    print("Restoring model weights from the end of the best epoch")
                    self.model.set_weights(self.best_weights)


class MTOnAllValDataEarlyStopping(OnAllValDataEarlyStopping):
    def __init__(self, dataname, gen_bool, val_gen, val_y, train_gen, train_y,
                 qsize, workers,
                 monitor, mode, patience, min_delta=0,
                 restore_best_weights=False, verbose=0):
        super(OnAllValDataEarlyStopping, self).__init__()
        self.dti_val_gen, self.pcba_val_gen = val_gen
        self.dti_val_target, self.pcba_val_target = val_y

    def get_score(self):
        # PCBA
        pred_model = Model(inputs=self.model.input,
                           outputs=[self.model.get_layer('pcba_output').output])
        if self.gen_bool:
            prediction = \
                np.asarray(pred_model.predict_generator(
                    self.pcba_val_gen, max_queue_size=self.qsize,
                    workers=self.workers, use_multiprocessing=True, verbose=0))
        else:
            prediction = \
                np.asarray(pred_model.predict(self.pcba_val_gen))
        target = self.pcba_val_target
        dict_perf = get_clf_perf(prediction, target)
        print('### PCBA')
        print("##### — AUPR: " + str(dict_perf['AUPR'][0]) +
              " — ROCAUC: " + str(dict_perf['ROCAUC'][0]) +
              " — ACC: " + str(dict_perf['ACC'][0]) + " — f1: " + str(dict_perf['F1'][0]) +
              " — Precision: " + str(dict_perf['Precision'][0]) +
              " — Recall " + str(dict_perf['Recall'][0]) + " #####")
        if len(dict_perf['AUPR'][1]) > 1:
            print("— AUPR: " + str(dict_perf['AUPR'][1]))
            print("— ROCAUC: " + str(dict_perf['ROCAUC'][1]))
            print("— ACC: " + str(dict_perf['ACC'][1]))
            print("— f1: " + str(dict_perf['F1'][1]))
            print("— Precision: " + str(dict_perf['Precision'][1]))
            print("— Recall " + str(dict_perf['Recall'][1]))
        print('### end PCBA')

        # DTI
        pred_model = Model(inputs=self.model.input,
                           outputs=[self.model.get_layer('dti_output').output])
        if self.gen_bool:
            prediction = \
                np.asarray(pred_model.predict_generator(
                    self.dti_val_gen, max_queue_size=self.qsize,
                    workers=self.workers, use_multiprocessing=True, verbose=0))
        else:
            prediction = \
                np.asarray(pred_model.predict(self.dti_val_gen))
        target = self.dti_val_target

        dict_perf = get_clf_perf(prediction, target)

        if self.monitor == 'aupr':
            return dict_perf['AUPR'][0], dict_perf
        elif self.monitor == 'rocauc':
            return dict_perf['ROCAUC'][0], dict_perf
        elif self.monitor == 'f1':
            return dict_perf['F1'][0], dict_perf


def del_missing_data(y_true, y_pred):
    list_y_true, list_y_pred, missing_ratio = [], [], []
    # print(y_true)
    # print(y_true.shape)
    if len(y_pred.shape) > 1:
        if y_pred.shape[1] == 1:
            y_pred = y_pred[:, 0]
    if len(y_pred.shape) > 1:
        if len(y_true.shape) == 1:
            y_true = keras.utils.to_categorical(y_true)
        for i_out in range(y_pred.shape[1]):
            ind_missing = np.where(y_true[:, i_out] == MISSING_DATA_MASK_VALUE)[0]
            list_y_true.append(np.delete(y_true[:, i_out], ind_missing))
            list_y_pred.append(np.delete(y_pred[:, i_out], ind_missing))
            missing_ratio.append(float(y_true.shape[0] - len(ind_missing)))
        missing_ratio = np.array(missing_ratio) / np.sum(missing_ratio)
    else:
        ind_missing = np.where(y_true[:] == MISSING_DATA_MASK_VALUE)[0]
        list_y_true.append(np.delete(y_true[:], ind_missing))
        list_y_pred.append(np.delete(y_pred[:], ind_missing))
        missing_ratio = np.array([1.])
    # import pdb; pdb.Pdb().set_trace()
    return list_y_true, list_y_pred, missing_ratio


def mcc(y_true, y_pred):
    if len(y_pred.shape) > 1:
        list_y_true, list_y_pred, ratio_per_output = del_missing_data(y_true, y_pred)
        mcc = []
        for i_out in range(len(list_y_pred)):
            # import pdb; pdb.Pdb().set_trace()
            mcc.append(round(metrics.matthews_corrcoef(list_y_true[i_out], list_y_pred[i_out]), 4))
        return (round(np.dot(mcc, ratio_per_output), 4), mcc)
    else:
        mcc = round(metrics.matthews_corrcoef(y_true, y_pred), 4)
        return (mcc, [mcc])


def acc(y_true, y_pred):
    if len(y_pred.shape) > 1:
        list_y_true, list_y_pred, ratio_per_output = del_missing_data(y_true, y_pred)
        acc = []
        for i_out in range(len(list_y_pred)):
            # import pdb; pdb.Pdb().set_trace()
            acc.append(round(metrics.accuracy_score(list_y_true[i_out], list_y_pred[i_out]), 4))
        return (round(np.dot(acc, ratio_per_output), 4), acc)
    else:
        acc = round(metrics.accuracy_score(y_true, y_pred), 4)
        return (acc, [acc])


def aupr(y_true, y_pred):
    if len(y_pred.shape) > 1:
        list_y_true, list_y_pred, ratio_per_output = del_missing_data(y_true, y_pred)
        # import pdb; pdb.Pdb().set_trace()
        aupr = []
        for i_out in range(len(list_y_pred)):
            aupr.append(sround(
                metrics.average_precision_score(list_y_true[i_out], list_y_pred[i_out])))
        return (round(np.dot(aupr, ratio_per_output), 2), aupr)
    else:
        aupr = sround(metrics.average_precision_score(y_true, y_pred))
        return (aupr, [aupr])


def rocauc(y_true, y_pred):
    if len(y_pred.shape) > 1:
        list_y_true, list_y_pred, ratio_per_output = del_missing_data(y_true, y_pred)
        auc = []
        for i_out in range(len(list_y_pred)):
            auc.append(sround(
                metrics.roc_auc_score(list_y_true[i_out], list_y_pred[i_out])))
        return (round(np.dot(auc, ratio_per_output), 2), auc)
    else:
        rocauc = sround(metrics.roc_auc_score(y_true, y_pred))
        return (rocauc, [rocauc])


def f1(y_true, y_pred):
    if len(y_pred.shape) > 1:
        list_y_true, list_y_pred, ratio_per_output = del_missing_data(y_true, y_pred)
        f1 = []
        for i_out in range(len(list_y_pred)):
            f1.append(sround(
                metrics.f1_score(list_y_true[i_out], list_y_pred[i_out])))
        return (round(np.dot(f1, ratio_per_output), 2), f1)
    else:
        f1 = sround(metrics.f1_score(y_true, y_pred))
        return (f1, [f1])


def recall(y_true, y_pred):
    if len(y_pred.shape) > 1:
        list_y_true, list_y_pred, ratio_per_output = del_missing_data(y_true, y_pred)
        recall = []
        for i_out in range(len(list_y_pred)):
            recall.append(sround(
                metrics.recall_score(list_y_true[i_out], list_y_pred[i_out])))
        return (round(np.dot(recall, ratio_per_output), 2), recall)
    else:
        recall = sround(metrics.recall_score(y_true, y_pred))
        return (recall, [recall])


def precision(y_true, y_pred):
    if len(y_pred.shape) > 1:
        list_y_true, list_y_pred, ratio_per_output = del_missing_data(y_true, y_pred)
        recall = []
        for i_out in range(len(list_y_pred)):
            recall.append(sround(
                metrics.precision_score(list_y_true[i_out], list_y_pred[i_out])))
        return (round(np.dot(recall, ratio_per_output), 2), recall)
    else:
        precision = sround(metrics.precision_score(y_true, y_pred))
        return (precision, [precision])


def mse(y_true, y_pred):
    if len(y_pred.shape) > 1:
        list_y_true, list_y_pred, ratio_per_output = del_missing_data(y_true, y_pred)
        mse = []
        for i_out in range(len(list_y_pred)):
            mse.append(round(
                metrics.mean_squared_error(list_y_true[i_out], list_y_pred[i_out]), 5))
        return (round(np.dot(mse, ratio_per_output), 5), mse)
    else:
        mse = round(metrics.mean_squared_error(y_true, y_pred), 5)
        return (mse, [mse])


def plot_model(model_, filename):
    keras.utils.plot_model(model_, to_file=filename)


def save_model_weights(model, filename):
    layer_dict = dict([(layer.name, layer) for layer in model.trainable_weights])

    for name, w in layer_dict.items():
        if 'pconv' in name or 'biLSTM' in name or 'emb_conv' in name or 'emb_nei' in name or \
                'agg' in name or 'emb_conv' in name:
            np.save(filename + name.replace('/', '_'), K.get_value(w))


def save_FFN_weights(model, filename):
    layer_dict = dict([(layer.name, layer) for layer in model.trainable_weights])

    for name, w in layer_dict.items():
        if 'fea_layer' in name:
            np.save(filename + name.replace('/', '_'), K.get_value(w))


def save_model_pred_weights(model, filename):
    layer_dict = dict([(layer.name, layer) for layer in model.trainable_weights])

    for name, w in layer_dict.items():
        if 'pred_layer' in name or 'BNpred' in name or 'output' in name:
            np.save(filename + name.replace('/', '_'), K.get_value(w))


def load_FFN_weights(model, filename):
    layer_dict = dict([(layer.name, layer) for layer in model.trainable_weights])
    for name, w in layer_dict.items():
        if 'fea_layer' in name:
            if os.path.isfile(filename + name.replace('/', '_') + '.npy'):
                print('FOUND', filename + name.replace('/', '_') + '.npy')
                value = np.load(filename + name.replace('/', '_') + '.npy')
                keras.backend.set_value(w, value)
            else:
                print('NOT FOUND', filename + name.replace('/', '_') + '.npy')
            # layer_dict['some_name'].set_weights(w)
    print('done load_FFN_weights')


def load_model_pred_weights(model, filename):
    layer_dict = dict([(layer.name, layer) for layer in model.trainable_weights])
    for name, w in layer_dict.items():
        if 'pred_layer' in name or 'BNpred' in name or 'output' in name:
            if os.path.isfile(filename + name.replace('/', '_') + '.npy'):
                print('FOUND', filename + name.replace('/', '_') + '.npy')
                value = np.load(filename + name.replace('/', '_') + '.npy')
                keras.backend.set_value(w, value)
            else:
                print('NOT FOUND', filename + name.replace('/', '_') + '.npy')
            # layer_dict['some_name'].set_weights(w)
    print('done load_model_pred_weights')


def load_model_weights(model, filename):
    # YOU MUST NAMED ANY LAYER !!!!
    # If you need to load the weights into a different architecture (with some layers in common),
    # for instance for fine-tuning or transfer-learning, you can load them by layer name
    # model.load_weights(filename, by_name=True)

    # layer_dict = dict([(layer.name, layer) for layer in model.layers])
    layer_dict = dict([(layer.name, layer) for layer in model.trainable_weights])
    for name, w in layer_dict.items():
        if (('pconv' in name or 'biLSTM' in name) and
                ('CellLoc' in filename or 'DrugBankH' in filename or
                 'DrugBankEC' in filename)) or \
                (('emb_conv' in name or 'emb_nei' in name or 'agg' in name or 'emb_conv' in name)
                 and ('PCBA' in filename or 'DrugBankH' in filename or 'DrugBankEC' in filename)):
            if os.path.isfile(filename + name.replace('/', '_') + '.npy'):
                print('FOUND', filename + name.replace('/', '_') + '.npy')
                value = np.load(filename + name.replace('/', '_') + '.npy')
                keras.backend.set_value(w, value)
            else:
                print('NOT FOUND', filename + name.replace('/', '_') + '.npy')
            # layer_dict['some_name'].set_weights(w)
    print('done load_model_weights')

    sys.stdout.flush()
