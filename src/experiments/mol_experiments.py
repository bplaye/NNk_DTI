import tempfile
import argparse
import sys
import os
import pickle
import json
import numpy as np
# from src.utils.package_utils import load_model_weights
from src.utils.DB_utils import load_dataset_options, LIST_MOL_DATASETS, LIST_PROT_DATASETS
from src.utils.DB_utils import LIST_AA_DATASETS, LIST_DTI_DATASETS, LIST_CLF_DATASETS
from src.utils.DB_utils import LIST_MULTICLF_DATASETS, LIST_MTDTI_DATASETS
from src.utils.generators_utils import load_generator, y_file
from src.model.mol_keras_model import MolModel
from src.model.prot_keras_model import ProtModel
from src.model.dti_keras_model import DTIModel, MTDTIModel
from src.utils.package_utils import get_clf_perf, get_regr_perf, my_to_categorical
from src.utils.package_utils import save_model_weights, save_model_pred_weights, save_FFN_weights
from src.utils.mol_utils import NB_MAX_ATOMS
from keras import backend as K


class Experiment():

    def __init__(self, filename):
        self.parser = argparse.ArgumentParser()
        self.filename = filename

    def run(self):
        self._get_parameters()
        dataname, batch_size, n_folds, val_fold, test_fold, setting, ratio_tr, ratio_te = \
            self.args.dataname, self.args.batch_size, self.args.n_folds, self.args.val_fold, \
            self.args.test_fold, self.args.setting, self.args.ratio_tr, self.args.ratio_te

        if self.args.dataname in LIST_MULTICLF_DATASETS:
            option = {'padding': self.args.padd, 'n_classes': self.dataset.n_outputs,
                      'hand_crafted_features': self.hand_crafted_features}
        else:
            option = {'padding': self.args.padd,
                      'hand_crafted_features': self.hand_crafted_features}

        if self.args.dataname in LIST_MOL_DATASETS + LIST_DTI_DATASETS:
            option['aug'] = 2 if 'bond' in self.args.m_agg_nei['name'] else \
                1 if 'GAT' in self.args.m_agg_nei['name'] else 0
            option['bonds'] = True if 'bond' in self.args.m_agg_nei['name'] else False

        foldname = dataname + '_' + str(test_fold) + ',' + str(val_fold) + '_' + \
            str(setting) + '_' + str(ratio_tr) + '_' + str(ratio_te)
        if not os.path.exists('results/pred/' + foldname):
            os.makedirs('results/pred/' + foldname)
        if not os.path.exists('results/model/' + foldname):
            os.makedirs('results/model/' + foldname)
        if not os.path.exists('results/perf/' + foldname):
            os.makedirs('results/perf/' + foldname)
        while True:
            filename = tempfile.NamedTemporaryFile(prefix='NN_').name.split('/')[-1]
            if not os.path.isfile('results/pred/' + foldname + '/' + filename + '.data'):
                break

        if self.args.save_model is True:
            test_fold = None

            tr_gen, val_gen, test_gen, y_data = load_generator(
                dataname, batch_size, n_folds, val_fold, test_fold, setting, ratio_tr, ratio_te,
                option)

            max_score = 0 if dataname in LIST_CLF_DATASETS else 10000000
            for trys in range(self.args.times):
                self._build_model()

                if self.args.times != 1:
                    strmodel = str(self.args.times) + '_' + str(self.model)
                else:
                    strmodel = str(self.model)

                self.model.fit(tr_gen, val_gen, y_data[0], y_data[1], exp.args.ratio_tr,
                               debug=self.args.debug)

                print("PRED ON VALIDATION DATA")
                test_pred_, test_emb_ = self.model.predict(val_gen)

                if self.dataset.name in LIST_CLF_DATASETS:
                    dict_perf_ = get_clf_perf(test_pred_, y_data[1])
                    if dict_perf_['AUPR'][0] > max_score:
                        dict_perf = dict_perf_
                        max_score = dict_perf_['AUPR'][0]
                        save_model_weights(self.model.model,
                                           'results/model/' + foldname + '/' + filename)
                        if self.dataset.name in LIST_DTI_DATASETS:
                            save_model_pred_weights(self.model.model,
                                                    'results/model/' + foldname + '/' + filename)
                else:
                    dict_perf_ = get_regr_perf(test_pred_, y_data[1])
                    if dict_perf_['MSE'][0] < max_score:
                        dict_perf = dict_perf_
                        max_score = dict_perf_['MSE'][0]
                        save_model_weights(self.model.model,
                                           'results/model/' + foldname + '/' + filename)
                K.clear_session()
                del self.model
        else:

            tr_gen, val_gen, test_gen, y_data = load_generator(
                dataname, batch_size, n_folds, val_fold, test_fold, setting, ratio_tr, ratio_te,
                option)

            train_pred, train_emb, val_pred, val_emb, test_pred, test_emb = \
                None, None, None, None, None, None
            max_score = 0 if dataname in LIST_CLF_DATASETS else 10000000
            dict_perf_all = []
            for trys in range(self.args.times):
                self._build_model()

                if self.args.times != 1:
                    strmodel = str(self.args.times) + '_' + str(self.model)
                else:
                    strmodel = str(self.model)

                self.model.fit(tr_gen, val_gen, y_data[0], y_data[1], exp.args.ratio_tr,
                               debug=self.args.debug)

                print("PRED ON TRAIN DATA")
                train_pred_, train_emb_ = self.model.predict(tr_gen)
                print("PRED ON VALIDATION DATA")
                val_pred_, val_emb_ = self.model.predict(val_gen)
                print("PRED ON TEST DATA")
                test_pred_, test_emb_ = self.model.predict(test_gen)
                # perf, emb, pred = self.model.evaluate(test_gen)
                # if self.args.dataname in LIST_AA_DATASETS:
                #     y_data[2] = np.concatenate(y_data[2], axis=0)
                print('PERF ON VAL', get_clf_perf(val_pred_, y_data[1]))

                if self.dataset.name in LIST_CLF_DATASETS:
                    dict_perf_ = get_clf_perf(test_pred_, y_data[2])
                    if dict_perf_['AUPR'][0] > max_score:
                        train_pred, train_emb, val_pred, val_emb, test_pred, test_emb = \
                            train_pred_, train_emb_, val_pred_, val_emb_, test_pred_, test_emb_
                        dict_perf = dict_perf_
                        max_score = dict_perf_['AUPR'][0]
                else:
                    dict_perf_ = get_regr_perf(test_pred_, y_data[2])
                    if dict_perf_['MSE'][0] < max_score:
                        train_pred, train_emb, val_pred, val_emb, test_pred, test_emb = \
                            train_pred_, train_emb_, val_pred_, val_emb_, test_pred_, test_emb_
                        dict_perf = dict_perf_
                        max_score = dict_perf_['MSE'][0]
                dict_perf_all.append(dict_perf_)

                if not (self.args.save_perf_try_save_FFN_weigths or
                        self.args.save_perf_try_save_model_weigths or
                        self.args.save_perf_try_save_pred_weigths):
                    K.clear_session()
                    del self.model

            if self.args.save_pred is True:
                pickle.dump((strmodel, test_pred, train_emb, y_data[0],
                             val_emb, y_data[1], test_emb, y_data[2]),
                            open('results/pred/' + foldname + '/' + filename + '.data', 'wb'))
            if self.args.save_perf is True:
                pickle.dump((strmodel, dict_perf),
                            open('results/perf/' + foldname + '/' + filename + '.data', 'wb'))
            if self.args.save_perf_try is True:
                list_aupr = [dict_perf_['AUPR'][0] for dict_perf_ in dict_perf_all]
                print("MEAN AUPR", np.mean(list_aupr), np.std(list_aupr))
                pickle.dump((strmodel, dict_perf_all),
                            open('results/perf/' + foldname + '/2' + filename + '_all.data', 'wb'))
                if self.args.save_perf_try_save_FFN_weigths is True:
                    save_FFN_weights(self.model.model,
                                     'results/model/' + foldname + '/ffn2' + filename)
                if self.args.save_perf_try_save_model_weigths is True:
                    save_model_weights(self.model.model,
                                       'results/model/' + foldname + '/mod2' + filename)
                if self.args.save_perf_try_save_pred_weigths is True:
                    save_model_pred_weights(self.model.model,
                                            'results/model/' + foldname + '/pred2' + filename)

        print('FINAL DICT PERF', dict_perf)
        print('results/model/' + foldname + '/' + filename)

    def _parse_parameters(self):
        pass

    def _make_enc_dict_param(self):
        pass

    def _get_parameters(self):
        self._parse_parameters()
        self.args = self.parser.parse_args()
        self.dataset = load_dataset_options(self.args.dataname)
        self._make_enc_dict_param()

    def _build_model(self):
        model_args = self.args.batch_size, self.args.n_epochs, self.args.init_lr, \
            self.args.patience_early_stopping, \
            self.args.lr_scheduler, self.args.pred_layers, self.args.pred_batchnorm, \
            self.args.pred_dropout, self.args.pred_reg, self.args.cpu_workers_for_generator, \
            self.args.queue_size_for_generator, \
            self.dataset, \
            self.enc_dict_param

        print(self.dataset.name)

        if self.dataset.name in LIST_MOL_DATASETS:
            exp.model = MolModel(*model_args)
        elif self.dataset.name in LIST_PROT_DATASETS or self.dataset.name in LIST_AA_DATASETS:
            exp.model = ProtModel(*model_args)
        elif self.dataset.name in LIST_DTI_DATASETS:
            exp.model = DTIModel(*model_args)
        elif self.dataset.name in LIST_DTI_DATASETS:
            exp.model = MTDTIModel(*model_args)

        exp.model.build()

    def _get_exp_parameters(self):
        # exp parameters
        # dataname, n_folds, fold_val, fold_te, setting, ratio_tr, ratio_te = 'tox21',
        #    5, 8, 9, 0, 1
        self.parser.add_argument('-D', '--debug', action='store_true',
                                 help='true if in command line, else false')
        self.parser.add_argument('-spr', '--save_pred', action='store_true',
                                 help='true if in command line, else false')
        self.parser.add_argument('-spe', '--save_perf', action='store_true',
                                 help='true if in command line, else false')
        self.parser.add_argument('-spet', '--save_perf_try', action='store_true',
                                 help='true if in command line, else false')
        self.parser.add_argument('-spetsf', '--save_perf_try_save_FFN_weigths',
                                 action='store_true', help='true if in command line, else false')
        self.parser.add_argument('-spetsm', '--save_perf_try_save_model_weigths',
                                 action='store_true', help='true if in command line, else false')
        self.parser.add_argument('-spetsp', '--save_perf_try_save_pred_weigths',
                                 action='store_true', help='true if in command line, else false')
        self.parser.add_argument('-sm', '--save_model', action='store_true',
                                 help='true if in command line, else false')
        self.parser.add_argument('-t', '--times', type=int, default=1,
                                 help='nb of time to repeat the exp')
        self.parser.add_argument('-db', '--dataname', type=str, help='name of dataset')
        self.parser.add_argument('-nf', '--n_folds', type=int, help='nb of folds')
        self.parser.add_argument('-tef', '--test_fold', type=int, help='which fold to test on')
        self.parser.add_argument('-valf', '--val_fold', type=int,
                                 help='which fold to use a validation')
        self.parser.add_argument('-set', '--setting', type=int,
                                 help='setting of CV either 1,2,3,4')
        self.parser.add_argument('-rtr', '--ratio_tr', type=int,
                                 help='ratio pos/neg in train either 1,2,5')
        self.parser.add_argument('-rte', '--ratio_te', type=int,
                                 help='ratio pos/neg in test either 1,2,5')

    def _get_generator_parameters(self):
        # generator parameters
        self.parser.add_argument('-wgen', '--cpu_workers_for_generator', type=int, default=10,
                                 help='If 0, execute generator on the main thread')
        self.parser.add_argument('-qgen', '--queue_size_for_generator',
                                 type=int, default=5000, help='nb samples in queue')

    def _get_learning_parameters(self):
        # learning parameters
        # batch_size, n_epochs, patience_early_stopping = 10, 50, 10
        self.parser.add_argument('-bs', '--batch_size', type=int, help='batch size')
        self.parser.add_argument('-ne', '--n_epochs', type=int, help='nb of epochs')
        self.parser.add_argument('-ilr', '--init_lr', type=float, help='initial learnig rate')
        self.parser.add_argument('-eap', '--patience_early_stopping', type=int,
                                 help='patience_early_stopping')
        self.parser.add_argument('-lr', '--lr_scheduler',
                                 type=json.loads, help='dict with at least "name" as key')

    def _get_predictor_parameters(self):
        # pred architectre parameters
        self.parser.add_argument('-l', '--pred_layers', nargs="+",
                                 type=int, help='nb of units per layer (except the last one)')
        self.parser.add_argument('-bn', '--pred_batchnorm', action='store_true', default=False,
                                 help='true if in command line, else false')
        self.parser.add_argument('-pred_curr', '--pred_curriculum', type=str, default='None',
                                 help='None or str')
        self.parser.add_argument('-prednecurr', '--pred_epochs_curriculum', type=int, default=0,
                                 help='nb of epochs before unfreezing')
        self.parser.add_argument('-d', '--pred_dropout',
                                 type=float, help='dropout remove rate, if 0,ignored')
        self.parser.add_argument('-r', '--pred_reg', type=float, help='l2 reg coef. if 0,ignored')


class MolExperiment(Experiment):
    def _parse_parameters(self):
        self._get_exp_parameters()
        self._get_learning_parameters()
        self._get_generator_parameters()
        self._get_predictor_parameters()
        self._get_mol_encoder_parameters()

    def _get_mol_encoder_parameters(self):
        # mol architecture parameters
        self.parser.add_argument('-nat', '--m_n_att_atom', type=int, help='None or int')
        self.parser.add_argument('-mcurr', '--m_curriculum', type=str, default='None',
                                 help='None or str')
        self.parser.add_argument('-mnecurr', '--m_epochs_curriculum', type=int, default=0,
                                 help='nb of epochs before unfreezing')
        self.parser.add_argument('-ns', '--m_n_steps', type=int, help='nb of conv')
        self.parser.add_argument('-maggn', '--m_agg_nei',
                                 type=json.loads, help='dict with at least "name" as key')
        self.parser.add_argument('-magga', '--m_agg_all',
                                 type=json.loads, help='dict with at least "name" as key')
        self.parser.add_argument('-mc', '--m_combine',
                                 type=json.loads, help='dict with at least "name" as key')
        self.parser.add_argument('-mbn', '--m_batchnorm',
                                 action='store_true',
                                 help='true or false, any time there is a conv')
        self.parser.add_argument('-md', '--m_dropout',
                                 type=float, help='dropout remove rate')
        self.parser.add_argument('-mr', '--m_reg',
                                 type=float, help='regularisation coef')

    def _make_enc_dict_param(self):
        self.enc_dict_param = {'n_att_atom': self.args.m_n_att_atom,
                               'n_steps': self.args.m_n_steps,
                               'm_curriculum': self.args.m_curriculum,
                               'm_ne_curriculum': self.args.m_epochs_curriculum,
                               'agg_nei': self.args.m_agg_nei, 'agg_all': self.args.m_agg_all,
                               'combine': self.args.m_combine, 'BN': self.args.m_batchnorm,
                               'dropout': self.args.m_dropout, 'reg': self.args.m_reg}
        if self.enc_dict_param['agg_all']['name'] == 'pool':
            self.args.m_n_att_atom = NB_MAX_ATOMS[self.args.dataname]
            self.enc_dict_param['n_att_atoms'] = NB_MAX_ATOMS[self.args.dataname]
            self.enc_dict_param['agg_all']['n_atoms'] = NB_MAX_ATOMS[self.args.dataname]
        self.hand_crafted_features = self.args.m_combine['hand_crafted_features']
        self.args.padd = True if self.args.m_n_att_atom != 0 else False


class ProtExperiment(Experiment):
    def _parse_parameters(self):
        self._get_exp_parameters()
        self._get_learning_parameters()
        self._get_generator_parameters()
        self._get_predictor_parameters()
        self._get_prot_encoder_parameters()

    def _get_prot_encoder_parameters(self):
        # mol architecture parameters
        self.parser.add_argument('-sl', '--seq_len', type=int, help='None or int')
        self.parser.add_argument('-pcurr', '--p_curriculum', type=str, default='None',
                                 help='None or str')
        self.parser.add_argument('-pnecurr', '--p_epochs_curriculum', type=int, default=0,
                                 help='nb of epochs before unfreezing')
        self.parser.add_argument('-penc', '--p_encoder', type=json.loads,
                                 help='dict with at least "name" as key')
        self.parser.add_argument('-pbn', '--p_batchnorm',
                                 action='store_true',
                                 help='true or false, any time there is a conv')
        self.parser.add_argument('-pd', '--p_dropout',
                                 type=float, help='dropout remove rate')
        self.parser.add_argument('-pr', '--p_reg',
                                 type=float, help='regularisation coef')

    def _make_enc_dict_param(self):
        enc_dict_param = {'p_encoder': self.args.p_encoder, 'seq_len': self.args.seq_len,
                          'BN': self.args.p_batchnorm, 'p_curriculum': self.args.p_curriculum,
                          'p_ne_curriculum': self.args.p_epochs_curriculum,
                          'dropout': self.args.p_dropout, 'reg': self.args.p_reg}
        self.enc_dict_param = enc_dict_param
        self.hand_crafted_features = self.args.p_encoder['hand_crafted_features']
        self.args.padd = True if self.args.seq_len != 0 else False


class DTIExperiment(ProtExperiment, MolExperiment):
    def _parse_parameters(self):
        self._get_exp_parameters()
        self._get_learning_parameters()
        self._get_generator_parameters()
        self._get_predictor_parameters()
        self._get_mol_encoder_parameters()
        self._get_prot_encoder_parameters()
        self._get_dti_encoder_parameters()

    def _get_dti_encoder_parameters(self):
        # mol architecture parameters
        self.parser.add_argument('-dtij', '--dti_joint', type=json.loads,
                                 help='dict with at least "name" as key')
        self.parser.add_argument('-mtt', '--MT_type', type=json.loads,
                                 help='dict with at least "name" as key')
        self.parser.add_argument('-mtl', '--MT_lossw', type=json.loads,
                                 help='dict with at least "name" as key')

    def _make_enc_dict_param(self):
        enc_dict_param = {'p_encoder': self.args.p_encoder, 'seq_len': self.args.seq_len,
                          'p_BN': self.args.p_batchnorm, 'p_drop': self.args.p_dropout,
                          'p_reg': self.args.p_reg, 'm_BN': self.args.m_batchnorm,
                          'm_drop': self.args.m_dropout, 'm_reg': self.args.m_reg,
                          'n_att_atom': self.args.m_n_att_atom, 'agg_nei': self.args.m_agg_nei,
                          'agg_all': self.args.m_agg_all, 'combine': self.args.m_combine,
                          'joint': self.args.dti_joint, 'm_n_steps': self.args.m_n_steps,
                          'p_curriculum': self.args.p_curriculum,
                          'p_ne_curriculum': self.args.p_epochs_curriculum,
                          'm_curriculum': self.args.m_curriculum,
                          'm_ne_curriculum': self.args.m_epochs_curriculum,
                          'pred_curriculum': self.args.pred_curriculum,
                          'pred_ne_curriculum': self.args.pred_epochs_curriculum}
        self.enc_dict_param = enc_dict_param
        self.hand_crafted_features = self.args.dti_joint['hand_crafted_features']
        if self.enc_dict_param['agg_all']['name'] == 'pool':
            self.args.m_n_att_atom = NB_MAX_ATOMS[self.args.dataname]
            self.enc_dict_param['n_att_atoms'] = NB_MAX_ATOMS[self.args.dataname]
            self.enc_dict_param['agg_all']['n_atoms'] = NB_MAX_ATOMS[self.args.dataname]
        self.args.padd = True if (self.args.seq_len != 0 or self.args.m_n_att_atom != 0) else \
            False


class MTDTIExperiment(DTIExperiment):
    def _parse_parameters(self):
        self._get_exp_parameters()
        self._get_learning_parameters()
        self._get_generator_parameters()
        self._get_predictor_parameters()
        self._get_mol_encoder_parameters()
        self._get_prot_encoder_parameters()
        self._get_dti_encoder_parameters()
        self._get_mtdti_encoder_parameters()

    def _get_mtdti_encoder_parameters(self):
        # mol architecture parameters
        self.parser.add_argument('-mtt', '--MT_type', type=json.loads,
                                 help='dict with at least "name" as key')
        self.parser.add_argument('-mtl', '--MT_lossw', type=json.loads,
                                 help='dict with at least "name" as key')
        self.parser.add_argument('-dtij', '--dti_joint', type=json.loads,
                                 help='dict with at least "name" as key')

    def _make_enc_dict_param(self):
        enc_dict_param = {'p_encoder': self.args.p_encoder, 'seq_len': self.args.seq_len,
                          'p_BN': self.args.p_batchnorm, 'p_drop': self.args.p_dropout,
                          'p_reg': self.args.p_reg, 'm_BN': self.args.m_batchnorm,
                          'm_drop': self.args.m_dropout, 'm_reg': self.args.m_reg,
                          'n_att_atom': self.args.m_n_att_atom, 'agg_nei': self.args.m_agg_nei,
                          'agg_all': self.args.m_agg_all, 'combine': self.args.m_combine,
                          'joint': self.args.dti_joint, 'm_n_steps': self.args.m_n_steps,
                          'p_curriculum': self.args.p_curriculum,
                          'm_curriculum': self.args.m_curriculum,
                          'MT_type': self.args.MT_type, 'MT_lossw': self.args.MT_lossw
                          }
        self.enc_dict_param = enc_dict_param
        self.hand_crafted_features = self.args.dti_joint['hand_crafted_features']
        self.args.padd = True if (self.args.seq_len != 0 and self.args.m_n_att_atom != 0) else \
            False


if __name__ == "__main__":
    if sys.argv[2] in LIST_MOL_DATASETS:
        exp = MolExperiment('essai')
    elif sys.argv[2] in LIST_PROT_DATASETS or sys.argv[2] in LIST_AA_DATASETS:
        exp = ProtExperiment('essai')
    elif sys.argv[2] in LIST_DTI_DATASETS:
        exp = DTIExperiment('essai')
    elif sys.argv[2] in LIST_MTDTI_DATASETS:
        exp = MTDTIExperiment('essai')
    exp.run()
