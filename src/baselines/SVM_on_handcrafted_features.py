import pickle
import argparse
import numpy as np
import os
from sklearn.svm import SVC, SVR
from src.utils.DB_utils import LIST_BINARYCLF_DATASETS, LIST_REGR_DATASETS, LIST_MOL_DATASETS
from src.utils.DB_utils import LIST_AA_DATASETS, LIST_PROT_DATASETS, data_file, y_file
from src.utils.DB_utils import LIST_DTI_DATASETS, LIST_MULTICLF_DATASETS
from src.utils.package_utils import get_clf_perf, get_regr_perf
from src.process_datasets.process_DrugBank import get_DB
from src.utils.mol_utils import mol_build_K


LIST_C = [0.0001, 0.001, 0.01, 0.1, 1., 10., 100., 1000.]


def get_perf(DB, C, K_tr, y_tr, K_val, y_val):
    if DB in LIST_BINARYCLF_DATASETS:
        ml = SVC(kernel='precomputed', probability=True, class_weight='balanced', C=C)
        ml.fit(K_tr, y_tr)
        pred_temp = ml.predict_proba(K_val)[:, ml.classes_.tolist().index(1)]
        dict_perf = get_clf_perf(np.array(pred_temp), np.array(y_val))
        return dict_perf['AUPR'][0], dict_perf, pred_temp
    elif DB in LIST_MULTICLF_DATASETS:
        ml = SVC(kernel='precomputed', probability=True, C=C, class_weight='balanced',
                 decision_function_shape='ovr')
        ml.fit(K_tr, y_tr)
        pred_temp = ml.predict_proba(K_val)
        # import pdb; pdb.Pdb().set_trace()
        dict_perf = get_clf_perf(np.array(pred_temp), np.array(y_val))
        return dict_perf['AUPR'][0], dict_perf, pred_temp
    elif DB in LIST_REGR_DATASETS:
        ml = SVR(kernel='precomputed', C=C)
        ml.fit(K_tr, y_tr)
        pred_temp = ml.predict(K_val)
        dict_perf = get_regr_perf(pred_temp, y_val)
        return dict_perf['MSE'][0], dict_perf, pred_temp


def cv(DB, K_tr, y_tr, K_val, y_val):
    list_perf = []
    for iC, C in enumerate(LIST_C):
        perf, _, _ = get_perf(DB, C, K_tr, y_tr, K_val, y_val)
        list_perf.append(perf)
    return list_perf


def get_kernels(DB, x_tr, x_te, x_val, Kmol, Kprot, dict_prot2ind, dict_mol2ind):
    if DB in LIST_DTI_DATASETS:
        K_tr = make_Kcouple(x_tr, None, Kmol, Kprot, dict_prot2ind, dict_mol2ind)
        K_val = make_Kcouple(x_val, x_tr, Kmol, Kprot, dict_prot2ind, dict_mol2ind)
        K_te = make_Kcouple(x_te, x_tr, Kmol, Kprot, dict_prot2ind, dict_mol2ind)
    elif DB in LIST_PROT_DATASETS + LIST_AA_DATASETS:
        tr_indices = [dict_prot2ind[prot] for prot in x_tr]
        val_indices = [dict_prot2ind[prot] for prot in x_val]
        te_indices = [dict_prot2ind[prot] for prot in x_te]
        K_tr, K_val, K_te = Kprot[tr_indices, :], Kprot[val_indices, :], Kprot[te_indices, :]
        K_tr, K_val, K_te = K_tr[:, tr_indices], K_val[:, tr_indices], K_te[:, tr_indices]
    elif DB in LIST_MOL_DATASETS:
        tr_indices = [dict_mol2ind[mol] for mol in x_tr]
        val_indices = [dict_mol2ind[mol] for mol in x_val]
        te_indices = [dict_mol2ind[mol] for mol in x_te]
        # import pdb; pdb.Pdb().set_trace()
        K_tr, K_val, K_te = Kmol[tr_indices, :], Kmol[val_indices, :], Kmol[te_indices, :]
        K_tr, K_val, K_te = K_tr[:, tr_indices], K_val[:, tr_indices], K_te[:, tr_indices]
    return K_tr, K_val, K_te


def get_fold_data(DB, nfolds, fold_val, fold_te, setting, ratio_tr, ratio_te):
    if setting != 4:
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
    return x_tr, y_tr, x_val, y_val, x_te, y_te


def make_Kcouple(x1, x2, Kmol, Kprot, dict_prot2ind, dict_mol2ind):
    if x2 is None:
        K_temp = np.zeros((len(x1), len(x1)))
    else:
        K_temp = np.zeros((len(x1), len(x2)))
    for i in range(len(x1)):
        prot1, mol1 = x1[i]
        if x2 is None:
            for j in range(i, len(x1)):
                prot2, mol2 = x1[j]
                K_temp[i, j] = Kmol[dict_mol2ind[mol1], dict_mol2ind[mol2]] * \
                    Kprot[dict_prot2ind[prot1], dict_prot2ind[prot2]]
                K_temp[j, i] = K_temp[i, j]
        else:
            for j in range(len(x2)):
                prot2, mol2 = x2[j]
                K_temp[i, j] = Kmol[dict_mol2ind[mol1], dict_mol2ind[mol2]] * \
                    Kprot[dict_prot2ind[prot1], dict_prot2ind[prot2]]
    return K_temp


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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
    parser.add_argument('-cvv', '--cv_val', action='store_true', default=False,
                        help='true if in command line, else false')
    parser.add_argument('-f', '--force', action='store_true', default=False,
                        help='true if in command line, else false')
    args = parser.parse_args()
    DB, fold_val, fold_te, setting, ratio_tr, ratio_te = \
        args.dataname, args.val_fold, args.test_fold, args.setting, args.ratio_tr, args.ratio_te
    force = args.force

    foldname = 'results/pred/' + DB + '_' + str(fold_te) + ',' + str(fold_val) + '_' + \
        str(setting) + '_' + str(ratio_tr) + '_' + str(ratio_te)
    if not os.path.exists(foldname):
        os.makedirs(foldname)

    if not os.path.isfile(foldname + '/handSVM.data') or force:
        if DB in LIST_DTI_DATASETS:
            dict_ligand, dict_target, intMat, dict_ind2prot, dict_ind2mol, dict_prot2ind, \
                dict_mol2ind = get_DB(DB)
            Kmol = pickle.load(open('data/' + DB + '/' + DB + '_Kmol.data', 'rb'))
            Kprot = pickle.load(open('data/' + DB + '/' + DB + '_Kprot.data', 'rb'))
        elif DB in LIST_PROT_DATASETS + LIST_AA_DATASETS:
            list_ID = pickle.load(open('data/' + DB + '/' + DB + '_list_ID.data', 'rb'))
            dict_prot2ind = {prot: ind for ind, prot in enumerate(list_ID)}
            Kprot = pickle.load(open('data/' + DB + '/' + DB + '_Kprot.data', 'rb'))
            Kmol, dict_mol2ind = None, None
        elif DB in LIST_MOL_DATASETS:
            if DB not in ['HIV']:
                Kmol = pickle.load(open('data/' + DB + '/' + DB + '_Kmol.data', 'rb'))
            else:
                list_SMILES = pickle.load(
                    open('data/' + DB + '/' + DB + '_list_SMILES.data', 'rb'))
                Kmol = mol_build_K(list_SMILES)
            Kprot, dict_prot2ind = None, None
            list_ID = pickle.load(open('data/' + DB + '/' + DB + '_list_ID.data', 'rb'))
            dict_mol2ind = {mol: ind for ind, mol in enumerate(list_ID)}

        nfolds, seed = 5, 92

        x_tr, y_tr, x_val, y_val, x_te, y_te = \
            get_fold_data(DB, nfolds, fold_val, fold_te, setting, ratio_tr, ratio_te)
        # import pdb; pdb.Pdb().set_trace()
        K_tr, K_val, K_te = \
            get_kernels(DB, x_tr, x_te, x_val, Kmol, Kprot, dict_prot2ind, dict_mol2ind)

        if args.cv_val:
            list_perf = cv(DB, K_tr, y_tr, K_val, y_val)
            print(list_perf)
            C = LIST_C[np.argmax(np.array(list_perf))]
        else:
            C = 10.

        perf, dict_perf, pred = \
            get_perf(DB, C, K_tr, y_tr, K_te, y_te)
        print(C, dict_perf)

        pickle.dump((pred, y_te, dict_perf), open(foldname + '/handSVM.data', 'wb'))

