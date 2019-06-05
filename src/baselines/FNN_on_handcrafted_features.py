import pickle
import argparse
import numpy as np
from keras.utils import Sequence
from keras import optimizers
import os
import collections
from keras.layers import Dense, Dropout, BatchNormalization, Input
from keras import regularizers
from src.utils.package_utils import OnAllValDataEarlyStopping
from keras.callbacks import CSVLogger, ReduceLROnPlateau
from keras.callbacks import LearningRateScheduler
from src.utils.package_utils import get_imbalance_data
from src.utils.DB_utils import LIST_CLF_DATASETS, LIST_REGR_DATASETS, LIST_MOL_DATASETS
from src.utils.DB_utils import LIST_AA_DATASETS, LIST_PROT_DATASETS, data_file, y_file
from src.utils.DB_utils import LIST_DTI_DATASETS, load_dataset_options, LIST_MULTIOUT_DATASETS
from src.utils.DB_utils import MISSING_DATA_MASK_VALUE
from src.utils.package_utils import get_clf_perf, get_regr_perf
from src.utils.mol_utils import get_mol_fea, NB_MOL_FEATURES
from src.utils.prot_utils import NB_PROT_FEATURES
from keras.models import Model
import json
from sklearn import preprocessing
from src.utils.package_utils import normalise_class_proportion, get_lr_metric


LIST_C = [0.001, 0.01, 0.1, 1., 10., 100.]


def get_DB(DB):

    dict_ligand = pickle.load(open('data/' + DB + '/' + DB + '_dict_ID2SMILES.data', 'rb'))
    dict_target = pickle.load(open('data/' + DB + '/' + DB + '_dict_ID2FASTA.data', 'rb'))
    intMat = np.load('data/' + DB + '/' + DB + '_intMat.npy')
    dict_ind2prot = pickle.load(open('data/' + DB + '/' + DB + '_dict_ind2prot.data', 'rb'))
    dict_ind2mol = pickle.load(open('data/' + DB + '/' + DB + '_dict_ind2mol.data', 'rb'))
    dict_prot2ind = pickle.load(open('data/' + DB + '/' + DB + '_dict_prot2ind.data', 'rb'))
    dict_mol2ind = pickle.load(open('data/' + DB + '/' + DB + '_dict_mol2ind.data', 'rb'))
    return dict_ligand, dict_target, intMat, dict_ind2prot, dict_ind2mol, dict_prot2ind, \
        dict_mol2ind


class MolGenerator(Sequence):
    def __init__(self, dataname, x_set, y_set, batch_size, dict_id2features, training):
        self.dataname = dataname
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.indices = np.arange(len(self.x))
        if training is True:
            np.random.shuffle(self.indices)
        self.dict_id2features = dict_id2features
        self.training = training
        if dataname in LIST_MULTIOUT_DATASETS and dataname in LIST_CLF_DATASETS:
            y_set = np.array(y_set)
            class_proportion = {0: 0, 1: 0}
            for iout in range(y_set.shape[1]):
                c = collections.Counter(y_set[:, iout])
                class_proportion[0] += c[0]
                class_proportion[1] += c[1]
            if class_proportion[0] != class_proportion[1]:
                class_proportion = normalise_class_proportion(class_proportion)
            self.sw = []
            for row in y_set:
                if np.max(row) == 1:
                    self.sw.append(class_proportion[1])
                elif np.max(row) == 0:
                    self.sw.append(class_proportion[0])

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def get_batch_x(self, batch_x):
        list_mol_fea = []
        for ID in batch_x:
            fea = self.dict_id2fea[ID]
            list_mol_fea.append(fea)
        return np.stack(list_mol_fea, axis=0)

    def get_batch_y(self, batch_y):
        return np.array(batch_y)

    def __getitem__(self, idx):
        inds = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = np.array(self.x)[inds].tolist()
        batch_y = [self.y[i] for i in range(len(self.y)) if i in inds]

        batch_x = self.get_batch_x(batch_x)
        batch_y = self.get_batch_y(batch_y)

        if self.dataname in LIST_MULTIOUT_DATASETS and self.dataname in LIST_CLF_DATASETS:
            batch_sw = np.array(self.sw)[inds]
            # batch_sw = self.get_batch_sample_weight(batch_sw)
            return batch_x, batch_y, batch_sw
        else:
            return batch_x, batch_y

    def on_epoch_end(self):
        if self.training is True:
            np.random.shuffle(self.indices)


class DTIGenerator(Sequence):
    def __init__(self, dataname, x_set, y_set, batch_size, dict_, training):
        self.dataname = dataname
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.indices = np.arange(len(self.x))
        if training is True:
            np.random.shuffle(self.indices)
        self.dict_id2protfeatures, self.dict_id2molfeatures = dict_
        self.training = training

    def get_batch_x(self, batch_x):
        list_molfea, list_protfea = [], []
        for (prot_id, mol_id) in batch_x:
            mol_fea = self.dict_id2molfeatures[mol_id]
            list_molfea.append(mol_fea)
            prot_fea = self.dict_id2protfeatures[prot_id]
            list_protfea.append(prot_fea)
        prot_fea = preprocessing.scale(np.stack(list_protfea, axis=0))
        mol_fea = np.stack(list_molfea, axis=0)
        return np.concatenate((prot_fea, mol_fea), axis=1)

    def get_batch_y(self, batch_y):
        return np.expand_dims(np.array(batch_y, dtype=np.float32), -1)

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        inds = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = np.array(self.x)[inds].tolist()
        batch_y = np.array(self.y)[inds].tolist()

        batch_x = self.get_batch_x(batch_x)
        batch_y = self.get_batch_y(batch_y)

        if self.dataname in LIST_MULTIOUT_DATASETS and self.dataname in LIST_CLF_DATASETS:
            batch_sw = np.array(self.sw)[inds]
            # batch_sw = self.get_batch_sample_weight(batch_sw)
            return batch_x, batch_y, batch_sw
        else:
            return batch_x, batch_y

    def on_epoch_end(self):
        if self.training is True:
            np.random.shuffle(self.indices)


def get_generator(dataname, x_tr, y_tr, x_te, y_te, x_val, y_val, batch_size):
    if dataname in LIST_MOL_DATASETS:
        dict_id2molfeatures = pickle.load(open('data/' + dataname + '/' + dataname +
                                               '_dict_ID2features.data', 'rb'))
        return (MolGenerator(dataname, x_tr, y_tr, batch_size, dict_id2molfeatures, True),
                MolGenerator(dataname, x_val, y_val, batch_size, dict_id2molfeatures, False),
                MolGenerator(dataname, x_te, y_te, batch_size, dict_id2molfeatures, False))
    elif dataname in LIST_DTI_DATASETS:
        dict_id2molfeatures = pickle.load(open('data/' + dataname + '/' + dataname +
                                               '_dict_ID2molfeatures.data', 'rb'))
        dict_id2protfeatures = pickle.load(open('data/' + dataname + '/' + dataname +
                                                '_dict_ID2protfeatures.data', 'rb'))
        dict_ = (dict_id2protfeatures, dict_id2molfeatures)
        return (DTIGenerator(dataname, x_tr, y_tr, batch_size, dict_, True),
                DTIGenerator(dataname, x_val, y_val, batch_size, dict_, False),
                DTIGenerator(dataname, x_te, y_te, batch_size, dict_, False))


class FNN_model():
    def __init__(self, param, DB):
        self.DB = DB
        if DB in LIST_MOL_DATASETS:
            self.nb_features = NB_MOL_FEATURES
        elif DB in LIST_PROT_DATASETS or DB in LIST_AA_DATASETS:
            self.nb_features = NB_PROT_FEATURES
        elif DB in LIST_DTI_DATASETS:
            self.nb_features = NB_MOL_FEATURES + NB_PROT_FEATURES
        self.dataset = load_dataset_options(DB)
        self.n_epochs = param['n_epochs']
        self.init_lr = param['init_lr']
        self.layers_units = param['layers_units']
        self.n_layers = len(self.layers_units)
        self.BN = param['BN']
        if param['reg'] == 0:
            self.kreg, self.breg = None, None
        else:
            self.kreg, self.breg = regularizers.l2(param['reg']), regularizers.l2(param['reg'])
        self.drop = param['drop']
        self.patience_early_stopping = param['patience']
        self.lr_scheduler = param['lr_scheduler']
        self.queue_size_for_generator = 1000
        self.cpu_workers_for_generator = 5

    def build(self):
        batch = Input(shape=(self.nb_features,), name='input')
        x = batch  # BatchNormalization(axis=-1)(batch)
        for nl in range(self.n_layers):
            x = Dense(self.layers_units[nl], activation='relu',
                      kernel_initializer='glorot_uniform', bias_initializer='zeros',
                      kernel_regularizer=self.kreg, bias_regularizer=self.breg,
                      activity_regularizer=None, kernel_constraint=None, bias_constraint=None,
                      name='layer_' + str(nl))(x)
            if self.drop != 0.:
                x = Dropout(self.drop, noise_shape=None, seed=None)(x)
            if self.BN is True:
                x = BatchNormalization(axis=-1)(x)  # emb must be (batch_size, emb_size)
        predictions = Dense(self.dataset.n_outputs,
                            activation=self.dataset.final_activation, name='output')(x)
        optimizer = optimizers.Adam(lr=self.init_lr)
        lr_metric = get_lr_metric(optimizer)
        model = Model(inputs=[batch], outputs=[predictions])
        model.compile(optimizer=optimizer,
                      loss={'output': self.dataset.loss},
                      loss_weights={'output': 1.},
                      metrics={'output': self.dataset.metrics + [lr_metric]})
        self.model = model

    def fit(self, X_tr, y_tr, X_val, y_val):
        callbacks = self.get_callbacks('fea', X_val, y_val)
        # balance predictor in case of uneven class distribution in train data
        inverse_class_proportion = get_imbalance_data(y_tr, self.dataset.name)
        if inverse_class_proportion is not None:
            print('WARNING: imbalance class proportion ', inverse_class_proportion)
        # import pdb; pdb.Pdb().set_trace()

        self.model.fit(X_tr, np.array(y_tr), steps_per_epoch=None, epochs=self.n_epochs,
                       verbose=1, callbacks=callbacks, validation_data=(X_val, y_val),
                       validation_steps=None, class_weight=inverse_class_proportion,
                       shuffle=True, initial_epoch=0)

    def get_callbacks(self, gen_bool, val_gen, val_y, train_gen, train_y):
        # callbacks when fitting
        early_stopping = OnAllValDataEarlyStopping(
            self.dataset.name, gen_bool, val_gen, val_y, train_gen, train_y,
            qsize=1000, workers=5,
            monitor=self.dataset.ea_metric, mode=self.dataset.ea_mode,
            min_delta=0,  # minimum change to qualify as an improvement
            patience=self.patience_early_stopping, verbose=1, restore_best_weights=True)
        csv_logger = CSVLogger('.tmp/temp.log')  # streams epoch results to a csv
        if self.lr_scheduler['name'] == 'ReduceLROnPlateau':
            lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2,
                                             verbose=1, mode='auto',
                                             min_delta=0.0001, min_lr=0.001)
        elif self.lr_scheduler['name'] == 'LearningRateScheduler':
            rate = self.lr_scheduler['rate']
            # reducing the learning rate by "rate" every 2 epochs
            lr_scheduler = LearningRateScheduler(
                lambda epoch: self.init_lr * rate ** (epoch // 2), verbose=0)
        callbacks = [early_stopping, lr_scheduler] + [csv_logger]
        return callbacks

    def fit_generator(self, train_gen, train_y, val_gen, val_y):
        callbacks = self.get_callbacks('gen', val_gen, val_y, train_gen, train_y)
        # balance predictor in case of uneven class distribution in train data
        inverse_class_proportion = None
        # inverse_class_proportion = get_imbalance_data(train_y, self.dataset.name)
        # if inverse_class_proportion is not None:
        #     print('WARNING: imbalance class proportion ', inverse_class_proportion)
        print(val_gen)

        # for e in range(self.n_epochs):
        #     vizualised_layers = [layer.output for layer in self.model.layers
        #                          if 'input' not in layer.name]
        #     self.model.metrics_tensors += vizualised_layers
        #     for i_tr in range(len(train_gen)):
        #         bx, by = train_gen.__getitem__(i_tr)
        #         res = self.model.train_on_batch(bx, by,
        #                                              class_weight=inverse_class_proportion)
        #         print(i_tr)
        #         import pdb ; pdb.Pdb().set_trace()
        self.model.fit_generator(train_gen, steps_per_epoch=None, epochs=self.n_epochs,
                                 verbose=1,
                                 callbacks=callbacks, validation_data=val_gen,
                                 validation_steps=None,
                                 class_weight=inverse_class_proportion,
                                 max_queue_size=self.queue_size_for_generator,
                                 workers=self.cpu_workers_for_generator,
                                 use_multiprocessing=True,
                                 shuffle=True, initial_epoch=0)

    def def_pred_model(self):
        return Model(inputs=self.model.input, outputs=self.model.get_layer('output').output)

    def predict(self, X_te):
        pred_model = self.def_pred_model()
        return pred_model.predict(X_te)

    def predict_generator(self, gen):
        pred_model = self.def_pred_model()
        return pred_model.predict_generator(
            gen, steps=None, max_queue_size=1000, workers=5, use_multiprocessing=True, verbose=1)


def get_perf(param, DB, X_tr, y_tr, X_val, y_val, X_te, y_te):
    ml = FNN_model(param, DB)
    ml.fit(X_tr, y_tr, X_val, y_val)
    pred_temp = ml.predict(X_te)
    # import pdb; pdb.Pdb().set_trace()
    if DB in LIST_CLF_DATASETS:
        dict_perf = get_clf_perf(pred_temp, y_te)
        return dict_perf['AUPR'], dict_perf, pred_temp
    elif DB in LIST_REGR_DATASETS:
        dict_perf = get_regr_perf(pred_temp, y_te)
        return dict_perf['AUPR'], dict_perf, pred_temp


def get_perf_generator(param, DB, tr_gen, tr_y, val_gen, val_y, te_gen, te_y):
    ml = FNN_model(param, DB)
    ml.build()
    ml.fit_generator(tr_gen, tr_y, val_gen, val_y)
    pred_temp = ml.predict_generator(te_gen)
    if DB in LIST_CLF_DATASETS:
        dict_perf = get_clf_perf(np.array(pred_temp), te_y)
        return dict_perf['AUPR'], dict_perf, pred_temp
    elif DB in LIST_REGR_DATASETS:
        dict_perf = get_regr_perf(pred_temp, te_y)
        return dict_perf['MSE'], dict_perf, pred_temp


def get_Xcouple(x, X_mol, X_prot, dict_mol2ind, dict_prot2ind):
    X = np.zeros((len(x), Xmol.shape[1] + Xprot.shape[1]))
    for i in range(len(x)):
        prot, mol = x[i]
        X[i, :] = np.concatenate([X_mol[dict_mol2ind[mol], :],
                                  X_prot[dict_prot2ind[prot], :]])
    return X


def get_Xsingle(x, X, dict_el2):
    X_temp = np.zeros((len(x), X.shape[1]))
    for i in range(len(x)):
        el = x[i]
        X_temp[i, :] = X[dict_el2[el], :]
    return X_temp


def get_design_matrices(DB, x_tr, x_te, x_val, Xmol, Xprot, dict_prot2ind, dict_mol2ind):
    if DB in LIST_DTI_DATASETS:
        X_tr = get_Xcouple(x_tr, Xmol, Xprot, dict_mol2ind, dict_prot2ind)
        X_val = get_Xcouple(x_val, Xmol, Xprot, dict_mol2ind, dict_prot2ind)
        X_te = get_Xcouple(x_te, Xmol, Xprot, dict_mol2ind, dict_prot2ind)
    elif DB in LIST_PROT_DATASETS + LIST_AA_DATASETS:
        X_tr = get_Xsingle(x_tr, Xprot, dict_prot2ind)
        X_val = get_Xsingle(x_val, Xprot, dict_prot2ind)
        X_te = get_Xsingle(x_te, Xprot, dict_prot2ind)
    elif DB in LIST_MOL_DATASETS:
        X_tr = get_Xsingle(x_tr, Xmol, dict_mol2ind)
        X_val = get_Xsingle(x_val, Xmol, dict_mol2ind)
        X_te = get_Xsingle(x_te, Xmol, dict_mol2ind)
    return X_tr, X_val, X_te


def get_fold_data(DB, nfolds, fold_val, fold_te, setting, ratio_tr, ratio_te):
    if not(setting == 4 or DB in ["DrugBankHEC-ECstand", "DrugBankHEC-Hstand"]):
        x_tr = [ind for sub in [pickle.load(open(data_file(DB, i, setting, ratio_tr), 'rb'))
                                for i in range(nfolds) if i != fold_val and i != fold_te]
                for ind in sub]
        y_tr = [ind for sub in [pickle.load(open(y_file(DB, i, setting, ratio_tr), 'rb'))
                                for i in range(nfolds) if i != fold_val and i != fold_te]
                for ind in sub]
        x_val = pickle.load(open(data_file(DB, fold_val, setting, ratio_te), 'rb'))
        y_val = pickle.load(open(y_file(DB, fold_val, setting, ratio_te), 'rb'))
        x_te = pickle.load(open(data_file(DB, fold_te, setting, ratio_te), 'rb'))
        y_te = pickle.load(open(y_file(DB, fold_te, setting, ratio_te), 'rb'))
    else:
        ifold = (fold_te, fold_val)
        x_val = pickle.load(open(data_file(DB, ifold, setting, ratio_te, 'val'), 'rb'))
        y_val = pickle.load(open(y_file(DB, ifold, setting, ratio_te, 'val'), 'rb'))
        x_te = pickle.load(open(data_file(DB, ifold, setting, ratio_te, 'test'), 'rb'))
        y_te = pickle.load(open(y_file(DB, ifold, setting, ratio_te, 'test'), 'rb'))
        x_tr = pickle.load(open(data_file(DB, ifold, setting, ratio_tr, 'train'), 'rb'))
        y_tr = pickle.load(open(y_file(DB, ifold, setting, ratio_tr, 'train'), 'rb'))
    # if DB in LIST_MULTIOUT_DATASETS:
    #     for i in range(len(y_tr)):
    #         y_tr[i][y_tr[i] == None] = MISSING_DATA_MASK_VALUE
    #     for i in range(len(y_te)):
    #         y_te[i][y_te[i] == None] = MISSING_DATA_MASK_VALUE
    #     for i in range(len(y_val)):
    #         y_val[i][y_val[i] == None] = MISSING_DATA_MASK_VALUE
    #     y_tr = np.array(np.stack(y_tr, axis=0), dtype=np.float32)
    #     y_val = np.array(np.stack(y_val, axis=0), dtype=np.float32)
    #     y_te = np.array(np.stack(y_te, axis=0), dtype=np.float32)
    return x_tr, y_tr, x_val, y_val, x_te, y_te


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--save', action='store_true', default=False,
                        help='true if in command line, else false')
    parser.add_argument('-db', '--dataname', type=str, help='name of dataset')
    parser.add_argument('-nf', '--n_folds', type=int, help='nb of folds')
    parser.add_argument('-tef', '--test_fold', type=int, help='which fold to test on')
    parser.add_argument('-valf', '--val_fold', type=int,
                        help='which fold to use a validation')
    parser.add_argument('-set', '--setting', type=int,
                        help='setting of CV either 1,2,3,4')
    parser.add_argument('-rtr', '--ratio_tr', type=int,
                        help='ratio pos/neg in train either 1,2,5')
    parser.add_argument('-rte', '--ratio_te', type=int,
                        help='ratio pos/neg in test either 1,2,5')

    parser.add_argument('-gen', '--generator', action='store_true', default=False,
                        help='true if in command line, else false')

    parser.add_argument('-ne', '--n_epochs', type=int, help='')
    parser.add_argument('-bs', '--batch_size', type=int, help='')
    parser.add_argument('-ilr', '--init_lr', type=float, help='')
    parser.add_argument('-eap', '--patience_early_stopping', type=int,
                        help='patience_early_stopping')
    parser.add_argument('-lr', '--lr_scheduler',
                        type=json.loads, help='dict with at least "name" as key')
    parser.add_argument('-l', '--layers', nargs="+",
                        type=int, help='nb of units per layer (except the last one)')
    parser.add_argument('-bn', '--batchnorm', action='store_true', default=False,
                        help='true if in command line, else false')
    parser.add_argument('-d', '--dropout',
                        type=float, help='dropout remove rate, if 0,ignored')
    parser.add_argument('-r', '--reg', default=0., type=float, help='l2 reg coef. if 0,ignored')

    args = parser.parse_args()
    param = {'layers_units': args.layers, 'BN': args.batchnorm, 'reg': args.reg,
             'drop': args.dropout, 'n_epochs': args.n_epochs, 'init_lr': args.init_lr,
             'patience': args.patience_early_stopping, 'lr_scheduler': args.lr_scheduler,
             'batch_size': args.batch_size}
    DB, fold_val, fold_te, setting, ratio_tr, ratio_te = \
        args.dataname, args.val_fold, args.test_fold, args.setting, args.ratio_tr, args.ratio_te
    nfolds, seed = 5, 92
    x_tr, y_tr, x_val, y_val, x_te, y_te = \
        get_fold_data(DB, nfolds, fold_val, fold_te, setting, ratio_tr, ratio_te)

    if not args.generator:
        if DB in LIST_DTI_DATASETS:
            dict_ligand, dict_target, intMat, dict_ind2prot, dict_ind2mol, dict_prot2ind, \
                dict_mol2ind = get_DB(DB)
            Xmol = pickle.load(open('data/' + DB + '/' + DB + '_Xmol.data', 'rb'))
            Xprot = pickle.load(open('data/' + DB + '/' + DB + '_Xprot.data', 'rb'))
        elif DB in LIST_PROT_DATASETS + LIST_AA_DATASETS:
            list_ID = pickle.load(open('data/' + DB + '/' + DB + '_list_ID.data', 'rb'))
            dict_prot2ind = {prot: ind for ind, prot in enumerate(list_ID)}
            Xprot = pickle.load(open('data/' + DB + '/' + DB + '_Xprot.data', 'rb'))
            Xmol, dict_mol2ind = None, None
        elif DB in LIST_MOL_DATASETS:
            Xmol = pickle.load(open('data/' + DB + '/' + DB + '_Xmol.data', 'rb'))
            Xprot, dict_prot2ind = None, None
            list_ID = pickle.load(open('data/' + DB + '/' + DB + '_list_ID.data', 'rb'))
            dict_mol2ind = {mol: ind for ind, mol in enumerate(list_ID)}

        X_tr, X_val, X_te = \
            get_design_matrices(DB, x_tr, x_te, x_val, Xmol, Xprot, dict_prot2ind, dict_mol2ind)

        perf, dict_perf, pred = get_perf(param, DB, X_tr, y_tr, X_val, y_val, X_te, y_te)

    else:
        if DB not in LIST_MOL_DATASETS + LIST_DTI_DATASETS:
            print('generator only available for MOLECULAR datasets')
            exit(1)
        tr_gen, val_gen, te_gen = \
            get_generator(DB, x_tr, y_tr, x_te, y_te, x_val, y_val, batch_size=args.batch_size)

        perf, dict_perf, pred = get_perf_generator(param, DB, tr_gen, y_tr, val_gen, y_val,
                                                   te_gen, y_te)

    foldname = 'results/pred/' + DB + '_' + str(fold_te) + ',' + str(fold_val) + '_' + \
        str(setting) + '_' + str(ratio_tr) + '_' + str(ratio_te)
    if not os.path.exists(foldname):
        os.makedirs(foldname)
    paramstr = ';'.join([cle + ':' + str(param[cle]).replace(' ', '')
                         for cle in np.sort(list(param.keys()))])
    print(paramstr)
    if args.save:
        pickle.dump((pred, y_te, dict_perf),
                    open(foldname + '/handFNN_' + paramstr + '.data', 'wb'))
    else:
        print('NOT SAVE')
    print('FINISHED', dict_perf)
