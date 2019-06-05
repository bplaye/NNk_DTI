import copy
import csv
import pickle
import sys
from rdkit import Chem
import collections
import os
import numpy as np
import sklearn.model_selection as model_selection
from src.utils.mol_utils import mol_build_K, mol_build_X
from src.process_datasets.cross_validation import mol_make_CV, Kmedoid_cluster
from src.process_datasets.cross_validation import Khierarchical_cluster, Xkmeans_cluster


list_classes = ['PCBA-1030', 'PCBA-1379', 'PCBA-1452', 'PCBA-1454', 'PCBA-1457',
                'PCBA-1458', 'PCBA-1460', 'PCBA-1461', 'PCBA-1468', 'PCBA-1469',
                'PCBA-1471', 'PCBA-1479', 'PCBA-1631', 'PCBA-1634', 'PCBA-1688',
                'PCBA-1721', 'PCBA-2100', 'PCBA-2101', 'PCBA-2147', 'PCBA-2242',
                'PCBA-2326', 'PCBA-2451', 'PCBA-2517', 'PCBA-2528', 'PCBA-2546',
                'PCBA-2549', 'PCBA-2551', 'PCBA-2662', 'PCBA-2675', 'PCBA-2676',
                'PCBA-411', 'PCBA-463254', 'PCBA-485281', 'PCBA-485290',
                'PCBA-485294', 'PCBA-485297', 'PCBA-485313', 'PCBA-485314',
                'PCBA-485341', 'PCBA-485349', 'PCBA-485353', 'PCBA-485360',
                'PCBA-485364', 'PCBA-485367', 'PCBA-492947', 'PCBA-493208',
                'PCBA-504327', 'PCBA-504332', 'PCBA-504333', 'PCBA-504339',
                'PCBA-504444', 'PCBA-504466', 'PCBA-504467', 'PCBA-504706',
                'PCBA-504842', 'PCBA-504845', 'PCBA-504847', 'PCBA-504891',
                'PCBA-540276', 'PCBA-540317', 'PCBA-588342', 'PCBA-588453',
                'PCBA-588456', 'PCBA-588579', 'PCBA-588590', 'PCBA-588591',
                'PCBA-588795', 'PCBA-588855', 'PCBA-602179', 'PCBA-602233',
                'PCBA-602310', 'PCBA-602313', 'PCBA-602332', 'PCBA-624170',
                'PCBA-624171', 'PCBA-624173', 'PCBA-624202', 'PCBA-624246',
                'PCBA-624287', 'PCBA-624288', 'PCBA-624291', 'PCBA-624296',
                'PCBA-624297', 'PCBA-624417', 'PCBA-651635', 'PCBA-651644',
                'PCBA-651768', 'PCBA-651965', 'PCBA-652025', 'PCBA-652104']


def get_multi_label(labels, n_None):
    y = []
    for il, lab in enumerate(labels):
        if lab != '':
            y.append(int(lab))
        else:
            y.append(None)
            n_None[il] += 1
    return y, n_None


def process_DB(DB):
    # df = pd.read_csv(data_folder + 'tox21.csv', sep=',')
    list_ID, list_SMILES, list_y, dict_id2smile = [], [], [], {}
    dict_per_class = {i: {'ind': [], 'y': []} for i in range(len(list_classes))}
    reader = csv.reader(open('data/PCBA/pcba.csv'), delimiter=',')
    # if DB == 'PCBA':
    n_None = [0 for _ in range(len(list_classes))]
    ii = 0
    for row in reader:
        if ii > 0:
            if ii % 100000 == 0:
                print(ii)
            # import pdb; pdb.Pdb().set_trace()
            smile = row[-1]
            # print(smile)
            m = Chem.MolFromSmiles(smile)
            if m is not None and smile != '':
                # if DB == 'PCBA':
                list_ID.append(row[-2])
                list_SMILES.append(row[-1])
                y_temp, n_None = get_multi_label(np.array(row)[list_col], n_None)
                list_y.append(y_temp)
                dict_id2smile[row[-2]] = row[-1]
                # elif 'PCBA' in DB:
                    # icl = int(DB.split('_')[1])
                for icl in range(len(list_classes)):
                    icl_true = list_all_classes.index(list_classes[icl])
                    if row[icl_true] != '':
                        dict_per_class[icl]['ind'].append(len(list_ID) - 1)
                        dict_per_class[icl]['y'].append(int(row[icl_true]))
        else:
            list_all_classes = row[:-2]
            list_col = [list_all_classes.index(cl) for cl in list_classes]
        ii += 1

    np.random.seed(92)
    # for DB in ['PCBA'] + ['PCBA_' + str(i) for i in range(len(list_classes))]:
    for DB in ['PCBA10', 'PCBA100']:
        foldname = 'data/' + DB
        if not os.path.exists(foldname):
            os.makedirs(foldname)
        if DB not in ['PCBA', 'PCBA10', 'PCBA100']:
            ind = dict_per_class[int(DB.split('_')[1])]['ind']
            list_y_temp = dict_per_class[int(DB.split('_')[1])]['y']
            list_SMILES_temp = np.array(list_SMILES)[ind].tolist()
            list_ID_temp = np.array(list_ID)[ind].tolist()
            dict_id2smile_temp = {ID: dict_id2smile[ID] for ID in list_ID_temp}
        elif DB == 'PCBA10':
            list_y_temp = np.array(list_y)
            for icl in range(list_y_temp.shape[1]):
                dict_cl = collections.Counter(list_y_temp[:, icl])
                nbpos, nbneg = dict_cl[1], dict_cl[0]
                indneg = np.where(list_y_temp[:, icl] == 0)[0]
                indchange = np.random.choice(indneg, max(nbneg - (10 * nbpos), 0), replace=False)
                list_y_temp[:, icl][indchange] = None
                print(icl, collections.Counter(list_y_temp[:, icl]))
            list_y_temp = list_y_temp.tolist()
            list_SMILES_temp, list_ID_temp, dict_id2smile_temp = \
                list_SMILES, list_ID, dict_id2smile
        elif DB == 'PCBA100':
            list_y_temp = np.array(list_y)
            for icl in range(list_y_temp.shape[1]):
                dict_cl = collections.Counter(list_y_temp[:, icl])
                nbpos, nbneg = dict_cl[1], dict_cl[0]
                indneg = np.where(list_y_temp[:, icl] == 0)[0]
                indchange = np.random.choice(indneg, max(nbneg - (100 * nbpos), 0), replace=False)
                list_y_temp[:, icl][indchange] = None
                print(icl, collections.Counter(list_y_temp[:, icl]))
            list_y_temp = list_y_temp.tolist()
            list_SMILES_temp, list_ID_temp, dict_id2smile_temp = \
                list_SMILES, list_ID, dict_id2smile
        else:
            list_y_temp, list_SMILES_temp, list_ID_temp, dict_id2smile_temp = \
                list_y, list_SMILES, list_ID, dict_id2smile

        pickle.dump(dict_id2smile_temp,
                    open('data/' + DB + '/' + DB + '_dict_ID2SMILES.data', 'wb'))
        # pickle.dump(dict_uniprot2fasta,
        #             open(root + 'data/' + DB + '/' + DB + '_dict_ID2FASTA.data', 'wb'))
        pickle.dump(list_SMILES_temp,
                    open('data/' + DB + '/' + DB + '_list_SMILES.data', 'wb'))
        pickle.dump(list_y_temp,
                    open('data/' + DB + '/' + DB + '_list_y.data', 'wb'))
        pickle.dump(list_ID_temp,
                    open('data/' + DB + '/' + DB + '_list_ID.data', 'wb'))

        f = open('data/' + DB + '/' + DB + '_dict_ID2SMILES.tsv', 'w')
        for cle, valeur in dict_id2smile_temp.items():
            f.write(cle + '\t' + valeur + '\n')
        f.close()
        f = open('data/' + DB + '/' + DB + '_list_SMILES.tsv', 'w')
        for s in list_SMILES_temp:
            f.write(s + '\n')
        f.close()
        f = open('data/' + DB + '/' + DB + '_list_y.tsv', 'w')
        for s in list_y_temp:
            if type(s) is list:
                for ll in s:
                    f.write(str(ll) + '\t')
                f.write('\n')
            else:
                f.write(str(s) + '\n')
        f.close()
        f = open('data/' + DB + '/' + DB + '_list_ID.tsv', 'w')
        for s in list_ID_temp:
            f.write(s + '\n')
        f.close()

        print(len(list_SMILES_temp))
        if DB in ['PCBA', 'PCBA10', 'PCBA100']:
            print([len(list_SMILES_temp) - n_None[i] for i in range(len(n_None))])
        elif 'PCBA' in DB:
            print(collections.Counter(list_y_temp))


def PCBA_CV(DB, data_type, list_ID, list_y, list_SMILES, dict_id2smile, n_folds):

    if data_type == 'kernel':
        if not os.path.isfile('data/' + DB + '/' + DB + '_K.npy'):
            K = mol_build_K(list_SMILES)
            np.save('data/' + DB + '/' + DB + '_K', K)
        else:
            K = np.load('data/' + DB + '/' + DB + '_K.npy')

        if DB == 'PCBA':
            # if Kmedoid
            # list_assignment, medoids = Kmedoid_cluster(K, n_folds)

            # if agglomerative clustering
            list_assignment = Khierarchical_cluster(K, n_folds)
        else:
            list_assignment = np.zeros(K.shape[0])
            for y in [0, 1]:
                indices = np.where(list_y == y)[0]
                K_local = K[indices, :]
                K_local = K_local[:, indices]
                local_assignment = Khierarchical_cluster(K_local, n_folds)
                list_assignment[indices] = local_assignment

    elif data_type == 'features':
        if not os.path.isfile('data/' + DB + '/' + DB + '_X.npy'):
            X = mol_build_X(list_SMILES)
            np.save('data/' + DB + '/' + DB + '_X', X)
        else:
            X = np.load('data/' + DB + '/' + DB + '_X.npy')

        if DB == 'PCBA':
            list_assignment = Xkmeans_cluster(X, n_folds)
        else:
            list_assignment = np.zeros(X.shape[0])
            for y in [0, 1]:
                indices = np.where(list_y == y)[0]
                X_local = X[indices, :]
                local_assignment = Xkmeans_cluster(X_local, n_folds)
                list_assignment[indices] = local_assignment

    elif data_type == 'standard':
        # if not os.path.isfile('data/' + DB + '/' + DB + '_X.npy'):
        #     X = mol_build_X(list_SMILES)
        #     np.save('data/' + DB + '/' + DB + '_X', X)
        # else:
        #     X = np.load('data/' + DB + '/' + DB + '_X.npy')
        list_ID = pickle.load(open('data/' + DB + '/' + DB + '_list_ID.data', 'rb'))
        list_y = np.array(pickle.load(open('data/' + DB + '/' + DB + '_list_y.data', 'rb')))
        X = np.zeros((len(list_ID), 1))
        list_assignment = np.zeros(X.shape[0])
        if DB not in ['PCBA', 'PCBA10', 'PCBA100']:
            skf = model_selection.StratifiedKFold(n_folds, shuffle=True, random_state=92)
            skf.get_n_splits(X, list_y)
            ifold = 0
            for train_index, test_index in skf.split(X, list_y):
                list_assignment[test_index] = ifold
                ifold += 1
        else:
            skf = model_selection.KFold(n_folds, shuffle=True, random_state=92)
            skf.get_n_splits(X)
            ifold = 0
            for train_index, test_index in skf.split(X):
                list_assignment[test_index] = ifold
                ifold += 1

    # import pdb; pdb.Pdb().set_trace()
    c = collections.Counter(list_assignment)
    print(c)
    folds = [np.where(list_assignment == cle)[0] for cle in list(c.keys())]

    fo = open('data/' + DB + '/' + DB + '_folds.txt', 'w')
    for ifold in range(n_folds):
        fo.write("ifold" + str(ifold) + '\n')
        if DB in ['PCBA', 'PCBA10', 'PCBA100']:
            for iclass in range(list_y.shape[1]):
                fo.write("iclass " + str(iclass) + ' ' +
                         str(collections.Counter(list_y[folds[ifold], iclass])) + '\n')
                print("iclass " + str(iclass) + ' ' +
                      str(collections.Counter(list_y[folds[ifold], iclass])))
        else:
            fo.write(str(collections.Counter(list_y[folds[ifold]])) + '\n')
            print(ifold, collections.Counter(list_y[folds[ifold]]))
        fo.write('\n')

    return folds


if __name__ == "__main__":
    # process_DB(sys.argv[1])

    # for DB in ['PCBA'] + ['PCBA_' + str(i) for i in range(len(list_classes))]:
    for DB in ['PCBA100']:
        mol_make_CV(DB, PCBA_CV, 'standard')

