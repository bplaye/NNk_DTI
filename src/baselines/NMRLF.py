'''
[1] Yong Liu, Min Wu, Chunyan Miao, Peilin Zhao, Xiao-Li Li, "Neighborhood Regularized Logistic Matrix Factorization for Drug-target Interaction Prediction", under review.
'''
import numpy as np
import argparse
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.metrics import auc
from src.utils.DB_utils import y_file, data_file
from src.utils.package_utils import get_clf_perf
from src.process_datasets.process_DrugBank import get_DB
import pickle
import os


class NRLMF:

    def __init__(self, cfix=5, K1=5, K2=5, num_factors=10, theta=1.0, lambda_d=0.625,
                 lambda_t=0.625, alpha=0.1, beta=0.1, max_iter=100):
        self.cfix = int(cfix)  # importance level for positive observations
        self.K1 = int(K1)
        self.K2 = int(K2)
        self.num_factors = int(num_factors)
        self.theta = float(theta)
        self.lambda_d = float(lambda_d)
        self.lambda_t = float(lambda_t)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.max_iter = int(max_iter)

    def AGD_optimization(self, seed=None):
        if seed is None:
            self.U = np.sqrt(1 / float(self.num_factors)) * \
                np.random.normal(size=(self.num_drugs, self.num_factors))
            self.V = np.sqrt(1 / float(self.num_factors)) * \
                np.random.normal(size=(self.num_targets, self.num_factors))
        else:
            prng = np.random.RandomState(seed)
            self.U = np.sqrt(1 / float(self.num_factors)) * \
                prng.normal(size=(self.num_drugs, self.num_factors))
            self.V = np.sqrt(1 / float(self.num_factors)) * \
                prng.normal(size=(self.num_targets, self.num_factors))
        dg_sum = np.zeros((self.num_drugs, self.U.shape[1]))
        tg_sum = np.zeros((self.num_targets, self.V.shape[1]))
        # import pdb; pdb.Pdb().set_trace()
        last_log = self.log_likelihood()
        for t in range(self.max_iter):
            dg = self.deriv(True)
            dg_sum += np.square(dg)
            vec_step_size = self.theta / np.sqrt(dg_sum)
            self.U += vec_step_size * dg
            tg = self.deriv(False)
            tg_sum += np.square(tg)
            vec_step_size = self.theta / np.sqrt(tg_sum)
            self.V += vec_step_size * tg
            curr_log = self.log_likelihood()
            delta_log = (curr_log - last_log) / abs(last_log)
            if abs(delta_log) < 1e-5:
                break
            last_log = curr_log

    def deriv(self, drug):
        if drug:
            vec_deriv = np.dot(self.intMat, self.V)
        else:
            vec_deriv = np.dot(self.intMat.T, self.U)
        A = np.dot(self.U, self.V.T)
        A = np.exp(A)
        A /= (A + self.ones)
        A = self.intMat1 * A
        if drug:
            vec_deriv -= np.dot(A, self.V)
            vec_deriv -= self.lambda_d * self.U + self.alpha * np.dot(self.DL, self.U)
        else:
            vec_deriv -= np.dot(A.T, self.U)
            vec_deriv -= self.lambda_t * self.V + self.beta * np.dot(self.TL, self.V)
        return vec_deriv

    def log_likelihood(self):
        loglik = 0
        A = np.dot(self.U, self.V.T)
        B = A * self.intMat
        loglik += np.sum(B)
        A = np.exp(A)
        A += self.ones
        A = np.log(A)
        A = self.intMat1 * A
        loglik -= np.sum(A)
        loglik -= 0.5 * self.lambda_d * np.sum(np.square(self.U)) + \
            0.5 * self.lambda_t * np.sum(np.square(self.V))
        # import pdb; pdb.Pdb().set_trace()
        loglik -= 0.5 * self.alpha * np.sum(np.diag((np.dot(self.U.T, self.DL)).dot(self.U)))
        loglik -= 0.5 * self.beta * np.sum(np.diag((np.dot(self.V.T, self.TL)).dot(self.V)))
        return loglik

    def construct_neighborhood(self, drugMat, targetMat):
        self.dsMat = drugMat - np.diag(np.diag(drugMat))
        self.tsMat = targetMat - np.diag(np.diag(targetMat))
        if self.K1 > 0:
            S1 = self.get_nearest_neighbors(self.dsMat, self.K1)
            self.DL = self.laplacian_matrix(S1)
            S2 = self.get_nearest_neighbors(self.tsMat, self.K1)
            self.TL = self.laplacian_matrix(S2)
        else:
            self.DL = self.laplacian_matrix(self.dsMat)
            self.TL = self.laplacian_matrix(self.tsMat)

    def laplacian_matrix(self, S):
        x = np.sum(S, axis=0)
        y = np.sum(S, axis=1)
        L = 0.5 * (np.diag(x + y) - (S + S.T))  # neighborhood regularization matrix
        return L

    def get_nearest_neighbors(self, S, size=5):
        m, n = S.shape
        X = np.zeros((m, n))
        for i in range(m):
            ii = np.argsort(S[i, :])[::-1][:min(size, n)]
            X[i, ii] = S[i, ii]
        return X

    def fix_model(self, W, intMat, drugMat, targetMat, seed=None):
        self.num_drugs, self.num_targets = intMat.shape
        self.ones = np.ones((self.num_drugs, self.num_targets))
        self.intMat = self.cfix * intMat * W
        self.intMat1 = (self.cfix - 1) * intMat * W + self.ones
        x, y = np.where(self.intMat > 0)
        self.train_drugs, self.train_targets = set(x.tolist()), set(y.tolist())
        self.construct_neighborhood(drugMat, targetMat)
        self.AGD_optimization(seed)

    def predict_scores(self, test_data, N):
        dinx = np.array(list(self.train_drugs))
        DS = self.dsMat[:, dinx]
        tinx = np.array(list(self.train_targets))
        TS = self.tsMat[:, tinx]
        scores = []
        for d, t in test_data:
            if d in self.train_drugs:
                if t in self.train_targets:
                    val = np.sum(self.U[d, :] * self.V[t, :])
                else:
                    jj = np.argsort(TS[t, :])[::-1][:N]
                    val = np.sum(self.U[d, :] * np.dot(TS[t, jj], self.V[tinx[jj], :])) / \
                        np.sum(TS[t, jj])
            else:
                if t in self.train_targets:
                    ii = np.argsort(DS[d, :])[::-1][:N]
                    val = np.sum(np.dot(DS[d, ii], self.U[dinx[ii], :]) * self.V[t, :]) / \
                        np.sum(DS[d, ii])
                else:
                    ii = np.argsort(DS[d, :])[::-1][:N]
                    jj = np.argsort(TS[t, :])[::-1][:N]
                    v1 = DS[d, ii].dot(self.U[dinx[ii], :]) / np.sum(DS[d, ii])
                    v2 = TS[t, jj].dot(self.V[tinx[jj], :]) / np.sum(TS[t, jj])
                    val = np.sum(v1 * v2)
            scores.append(np.exp(val) / (1 + np.exp(val)))
        return np.array(scores)

    def evaluation(self, test_data, test_label, intMat, R):
        dinx = np.array(list(self.train_drugs))
        DS = self.dsMat[:, dinx]
        tinx = np.array(list(self.train_targets))
        TS = self.tsMat[:, tinx]
        scores = []
        if self.K2 > 0:
            for d, t in test_data:
                if d in self.train_drugs:
                    if t in self.train_targets:
                        val = np.sum(self.U[d, :] * self.V[t, :])
                    else:
                        jj = np.argsort(TS[t, :])[::-1][:self.K2]
                        val = np.sum(self.U[d, :] * np.dot(TS[t, jj], self.V[tinx[jj], :])) / \
                            np.sum(TS[t, jj])
                else:
                    if t in self.train_targets:
                        ii = np.argsort(DS[d, :])[::-1][:self.K2]
                        val = np.sum(np.dot(DS[d, ii], self.U[dinx[ii], :]) * self.V[t, :]) / \
                            np.sum(DS[d, ii])
                    else:
                        ii = np.argsort(DS[d, :])[::-1][:self.K2]
                        jj = np.argsort(TS[t, :])[::-1][:self.K2]
                        v1 = DS[d, ii].dot(self.U[dinx[ii], :]) / np.sum(DS[d, ii])
                        v2 = TS[t, jj].dot(self.V[tinx[jj], :]) / np.sum(TS[t, jj])
                        val = np.sum(v1 * v2)
                scores.append(np.exp(val) / (1 + np.exp(val)))
        elif self.K2 == 0:
            for d, t in test_data:
                val = np.sum(self.U[d, :] * self.V[t, :])
                scores.append(np.exp(val) / (1 + np.exp(val)))
        prec, rec, thr = precision_recall_curve(test_label, np.array(scores))
        aupr_val = auc(rec, prec)
        fpr, tpr, thr = roc_curve(test_label, np.array(scores))
        auc_val = auc(fpr, tpr)
        return aupr_val, auc_val, np.array(scores)

    def predict(self, test_data, R, intMat_for_verbose, true_test_data=None):
        iii, jjj = test_data[:, 0], test_data[:, 1]

        dinx = np.array(list(self.train_drugs))
        DS = self.dsMat[:, dinx]
        tinx = np.array(list(self.train_targets))
        TS = self.tsMat[:, tinx]
        scores = []
        if self.K2 > 0:
            for dd, tt in test_data:
                casse = False
                # print('d', dd, 'label', intMat_for_verbose[dd, tt])
                # if dd == 33:
                #     import pdb; pdb.Pdb().set_trace()
                if dd in self.train_drugs:
                    # print('in selftrain drugs')
                    if tt in self.train_targets:
                        # print('in selftrain targets')
                        val = np.sum(self.U[dd, :] * self.V[tt, :])
                    else:
                        jj = np.argsort(TS[tt, :])[::-1][:self.K2]
                        val = np.sum(self.U[dd, :] * np.dot(TS[tt, jj], self.V[tinx[jj], :])) / \
                            np.sum(TS[tt, jj])
                else:
                    if tt in self.train_targets:
                        # print('in selftrain targets')
                        ii = np.argsort(DS[dd, :])[::-1][:self.K2]
                        val = np.sum(np.dot(DS[dd, ii], self.U[dinx[ii], :]) * self.V[tt, :]) / \
                            np.sum(DS[dd, ii])
                    else:
                        ii = np.argsort(DS[dd, :])[::-1][:self.K2]
                        jj = np.argsort(TS[tt, :])[::-1][:self.K2]
                        # if d == 413:
                        #     import pdb; pdb.Pdb().set_trace()
                        if np.sum(DS[dd, ii]) == 0 or np.sum(TS[tt, jj]) == 0:
                            val = -100
                            casse = True
                        else:
                            v1 = DS[dd, ii].dot(self.U[dinx[ii], :]) / np.sum(DS[dd, ii])
                            v2 = TS[tt, jj].dot(self.V[tinx[jj], :]) / np.sum(TS[tt, jj])
                            # print(v1, self.U[dinx[ii], :], DS[dd, ii],
                            #       DS[dd, ii].dot(self.U[dinx[ii], :]), np.sum(DS[dd, ii]))
                            # print(v2, self.V[tinx[jj], :], TS[tt, jj],
                            #       TS[tt, jj].dot(self.V[tinx[jj], :]), np.sum(TS[tt, jj]))
                            val = np.sum(v1 * v2)
                ss = np.exp(val) / (1 + np.exp(val))
                if not (np.isinf(ss) or np.isnan(ss)) and not casse:
                    scores.append(ss)
                else:
                    print('casse')
                    scores.append(np.random.random())
                # print(casse, val, scores[-1], round(scores[-1]))
                # print('')
        elif self.K2 == 0:
            for d, t in test_data:
                val = np.sum(self.U[d, :] * self.V[t, :])
                scores.append(np.exp(val) / (1 + np.exp(val)))
        if true_test_data is not None:
            iii, jjj = true_test_data[:, 0], true_test_data[:, 1]
            self.pred[iii, jjj] = scores
        else:
            self.pred[iii, jjj] = scores

    def get_perf(self, intMat):
        pred_ind = np.where(self.pred != np.inf)
        pred_local = self.pred[pred_ind[0], pred_ind[1]]
        test_local = intMat[pred_ind[0], pred_ind[1]]
        prec, rec, thr = precision_recall_curve(test_local, pred_local)
        aupr_val = auc(rec, prec)
        fpr, tpr, thr = roc_curve(test_local, pred_local)
        auc_val = auc(fpr, tpr)
        return aupr_val, auc_val

    def __str__(self):
        return "Model: NRLMF, c:%s, K1:%s, K2:%s, r:%s, lambda_d:%s, lambda_t:%s, alpha:%s," + \
            "beta:%s, theta:%s, max_iter:%s" % \
            (self.cfix, self.K1, self.K2, self.num_factors, self.lambda_d, self.lambda_t,
             self.alpha, self.beta, self.theta, self.max_iter)


if __name__ == "__main__":
    # get command line options
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
    args = parser.parse_args()
    DB, fold_val, fold_te, setting, ratio_tr, ratio_te = \
        args.dataname, args.val_fold, args.test_fold, args.setting, args.ratio_tr, args.ratio_te

    # get dataset specific kernels and items
    dict_ligand, dict_target, intMat, dict_ind2prot, dict_ind2mol, dict_prot2ind, \
        dict_mol2ind = get_DB(DB)
    drugMat = pickle.load(open('data/' + DB + '/' + DB + '_Kmol.data', 'rb'))
    targetMat = pickle.load(open('data/' + DB + '/' + DB + '_Kprot.data', 'rb'))
    n_folds, seed = 5, 92

    # get folds of data
    # x_tr, x_val, x_te are list of (protein, molecule) pairs (of IDs)
    #       for training, validation, test
    # y_tr, y_val, y_te are the true labels associated with the pairs
    if setting != 4:
        x_tr = [ind for sub in [pickle.load(open(data_file(DB, i, setting, ratio_tr), 'rb'))
                                for i in range(n_folds) if i != fold_val and i != fold_te]
                for ind in sub]
        y_tr = [ind for sub in [pickle.load(open(y_file(DB, i, setting, ratio_tr), 'rb'))
                                for i in range(n_folds) if i != fold_val and i != fold_te]
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

    intMat = intMat.T
    # get ids of mols and prots for each pairs in validation and test data
    val_label, test_label = y_val, y_te
    val_data, test_data = [], []
    for prot_id, mol_id in x_val:
        val_data.append([dict_mol2ind[mol_id], dict_prot2ind[prot_id]])
    val_data = np.stack(val_data, axis=0)
    # val_data is a "nb_samples_in_validation_data * 2" matrix, giving mol_id and prot_id for
    # each pair in the validation data
    for prot_id, mol_id in x_te:
        test_data.append([dict_mol2ind[mol_id], dict_prot2ind[prot_id]])
    test_data = np.stack(test_data, axis=0)
    # test_data is a "nb_samples_in_test_data * 2" matrix, giving mol_id and prot_id for
    # each pair in the test data
    W = np.zeros(intMat.shape)
    for prot_id, mol_id in x_tr:
        W[dict_mol2ind[mol_id], dict_prot2ind[prot_id]] = 1
    # W is a binary matrix to indicate what are the train data (pairs that can be used to train)
    R = W * intMat

    # if cross validation, find the best parameters on vaidation data
    # else get best parameters in original paper
    if args.cv_val:
        list_param = []
        for r in [50, 100]:
            for x in np.arange(-5, 2):
                for y in np.arange(-5, 3):
                    for z in np.arange(-5, 1):
                        for t in np.arange(-3, 1):
                            list_param.append((r, x, y, z, t))
        list_perf = []
        for par in list_param:
            param = {'c': 5, 'K1': 5, 'K2': 5, 'r': par[0], 'lambda_d': 2**(par[1]),
                     'lambda_t': 2**(par[1]), 'alpha': 2**(par[2]), 'beta': 2**(par[3]),
                     'theta': 2**(par[4]), 'max_iter': 100}
            model = NRLMF(cfix=param['c'], K1=param['K1'], K2=param['K2'],
                          num_factors=param['r'], lambda_d=param['lambda_d'],
                          lambda_t=param['lambda_t'], alpha=param['alpha'],
                          beta=param['beta'], theta=param['theta'],
                          max_iter=param['max_iter'])
            model.pred = np.full(intMat.shape, np.inf)
            model.fix_model(W, intMat, drugMat, targetMat, seed)
            aupr_val, auc_val, _ = model.evaluation(test_data, test_label, intMat, R)
            list_perf.append(aupr_val)

        par = list_param[np.argmax(list_perf)]
        best_param = {'c': 5, 'K1': 5, 'K2': 5, 'r': par[0], 'lambda_d': 2**(par[1]),
                      'lambda_t': 2**(par[1]), 'alpha': 2**(par[2]), 'beta': 2**(par[3]),
                      'theta': 2**(par[4]), 'max_iter': 100}
    else:
        best_param = {'c': 5, 'K1': 5, 'K2': 5, 'r': 50, 'lambda_d': 0.125, 'lambda_t': 0.125,
                      'alpha': 0.25, 'beta': 0.125, 'theta': 0.5, 'max_iter': 100}

    # define model with best parameters, fit and predict
    model = NRLMF(cfix=best_param['c'], K1=best_param['K1'], K2=best_param['K2'],
                  num_factors=best_param['r'], lambda_d=best_param['lambda_d'],
                  lambda_t=best_param['lambda_t'], alpha=best_param['alpha'],
                  beta=best_param['beta'], theta=best_param['theta'],
                  max_iter=best_param['max_iter'])
    model.pred = np.full(intMat.shape, np.inf)
    R = W * intMat
    # W is a binary matrix to indicate what are the train data (pairs that can be used to train)
    # intMat is the binary interaction matrix
    model.fix_model(W, intMat, drugMat, targetMat, seed)
    model.predict(test_data, R, intMat)

    # get list of true labels and associated predicted labels for the current test folds
    pred, truth = [], test_label
    for prot_id, mol_id in x_te:
        pred.append(model.pred[dict_mol2ind[mol_id], dict_prot2ind[prot_id]])
    pred, truth = np.array(pred), np.array(truth)

    # get performance based on the predictions
    dict_perf = get_clf_perf(pred, truth)
    print(dict_perf)

    # save prediction and performance
    foldname = 'results/pred/' + DB + '_' + str(fold_te) + ',' + str(fold_val) + '_' + \
        str(setting) + '_' + str(ratio_tr) + '_' + str(ratio_te)
    if not os.path.exists(foldname):
        os.makedirs(foldname)
    pickle.dump((pred, truth, dict_perf), open(foldname + '/NMRLF_' + str(args.cv_val) + '.data',
                                               'wb'))

