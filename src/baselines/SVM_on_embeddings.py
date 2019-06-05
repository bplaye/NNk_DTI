import pickle
import argparse
import numpy as np
import os
from sklearn.svm import SVC, SVR
from src.utils.DB_utils import LIST_CLF_DATASETS, LIST_REGR_DATASETS
from src.utils.package_utils import get_clf_perf, get_regr_perf
LIST_C = [0.001, 0.01, 0.1, 1., 10., 100.]


def get_perf_and_pred(DB, C, train_emb, train_label, test_emb, test_label):
    if DB in LIST_CLF_DATASETS:
        ml = SVC(kernel='rbf', gamma='scale', probability=True, C=C)
        ml.fit(train_emb, train_label)
        pred_temp = ml.predict_proba(test_emb)[:, 1]
        dict_perf = get_clf_perf(test_label, pred_temp)
        perf = dict_perf['AUPR'][0]
    elif DB in LIST_REGR_DATASETS:
        ml = SVR(kernel='rbf', gamma='scale', C=C)
        ml.fit(train_emb, train_label)
        pred_temp = ml.predict(test_emb)
        dict_perf = get_regr_perf(test_label, pred_temp)
        perf = dict_perf['MSE'][0]
    return perf, dict_perf, pred_temp


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
    parser.add_argument('-m', '--NN_model', type=str, help='str(model) from which to take emb')
    args = parser.parse_args()
    DB, fold_val, fold_te, setting, ratio_tr, ratio_te = \
        args.dataname, args.val_fold, args.test_fold, args.setting, args.ratio_tr, args.ratio_te

    foldname = DB + '_' + str(fold_te) + ',' + str(fold_val) + '_' + \
        str(setting) + '_' + str(ratio_tr) + '_' + str(ratio_te)
    filename = foldname + '/' + args.NN_model
    pred, train_emb, train_label, val_emb, val_label, test_emb, test_label = \
        pickle.load(open('results/pred/' + filename + '.data', 'rb'))

    if args.cv_val:
        list_perf = []
        for iC, C in enumerate(LIST_C):
            perf, _, _ = \
                get_perf_and_pred(DB, C, train_emb, train_label, val_emb, val_label)
            list_perf.append(perf)
        C = LIST_C[np.argmax(list_perf)]
    else:
        C = 10.

    perf, dict_perf, pred_temp = \
        get_perf_and_pred(DB, C, train_emb, train_label, test_emb, test_label)
    print(C, dict_perf)

    foldname = 'results/pred/' + DB + '_' + str(fold_te) + ',' + str(fold_val) + '_' + \
        str(setting) + '_' + str(ratio_tr) + '_' + str(ratio_te)
    if not os.path.exists(foldname):
        os.makedirs(foldname)
    pickle.dump((pred_temp, test_label, dict_perf),
                open(foldname + '/embSVM_' + str(args.cv_val) + '.data', 'wb'))
