import csv
import pickle
import sys
from rdkit import Chem
import collections
from src.utils.mol_utils import mol_build_K, mol_build_X
import os
import sklearn.model_selection as model_selection
import numpy as np
from src.process_datasets.cross_validation import mol_make_CV, Kmedoid_cluster
from src.process_datasets.cross_validation import Khierarchical_cluster, Xkmeans_cluster


def get_label_tox21(labels, n_None):
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
    reader = csv.reader(open('data/Tox21/tox21.csv'), delimiter=',')
    if DB == 'Tox21':
        n_None = [0 for _ in range(12)]
    i = 0
    for row in reader:
        if i > 0:
            smile = row[13]
            m = Chem.MolFromSmiles(smile)
            if m is not None and smile != '':
                if DB == 'Tox21':
                    list_ID.append(row[12])
                    list_SMILES.append(row[13])
                    y_temp, n_None = get_label_tox21(row[:12], n_None)
                    list_y.append(y_temp)
                    dict_id2smile[row[12]] = row[13]
                elif 'Tox21' in DB:
                    if row[int(DB.split('_')[1])] != '':
                        list_ID.append(row[12])
                        list_SMILES.append(row[13])
                        dict_id2smile[row[12]] = row[13]
                        list_y.append(int(row[int(DB.split('_')[1])]))
        i += 1
    pickle.dump(dict_id2smile,
                open('data/' + DB + '/' + DB + '_dict_ID2SMILES.data', 'wb'))
    # pickle.dump(dict_uniprot2fasta,
    #             open(root + 'data/' + DB + '/' + DB + '_dict_ID2FASTA.data', 'wb'))
    pickle.dump(list_SMILES,
                open('data/' + DB + '/' + DB + '_list_SMILES.data', 'wb'))
    pickle.dump(list_y,
                open('data/' + DB + '/' + DB + '_list_y.data', 'wb'))
    pickle.dump(list_ID,
                open('data/' + DB + '/' + DB + '_list_ID.data', 'wb'))

    f = open('data/' + DB + '/' + DB + '_dict_ID2SMILES.tsv', 'w')
    for cle, valeur in dict_id2smile.items():
        f.write(cle + '\t' + valeur + '\n')
    f.close()
    f = open('data/' + DB + '/' + DB + '_list_SMILES.tsv', 'w')
    for s in list_SMILES:
        f.write(s + '\n')
    f.close()
    f = open('data/' + DB + '/' + DB + '_list_y.tsv', 'w')
    for s in list_y:
        if type(s) is list:
            for ll in s:
                f.write(str(ll) + '\t')
            f.write('\n')
        else:
            f.write(str(s) + '\n')
    f.close()
    f = open('data/' + DB + '/' + DB + '_list_ID.tsv', 'w')
    for s in list_ID:
        f.write(s + '\n')
    f.close()

    print(len(list_SMILES))
    if DB == 'Tox21':
        print([len(list_SMILES) - n_None[i] for i in range(len(n_None))])
    elif 'Tox21' in DB:
        print(collections.Counter(list_y))


def tox21_CV(DB, data_type, list_ID, list_y, list_SMILES, dict_id2smile, n_folds):
    if data_type == 'kernel':
        if not os.path.isfile('data/' + DB + '/' + DB + '_K.npy'):
            K = mol_build_K(list_SMILES)
            np.save('data/' + DB + '/' + DB + '_K', K)
        else:
            K = np.load('data/' + DB + '/' + DB + '_K.npy')

        if DB == 'Tox21':
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

        if DB == 'Tox21':
            list_assignment = Xkmeans_cluster(X, n_folds)
        else:
            list_assignment = np.zeros(X.shape[0])
            for y in [0, 1]:
                indices = np.where(list_y == y)[0]
                X_local = X[indices, :]
                local_assignment = Xkmeans_cluster(X_local, n_folds)
                list_assignment[indices] = local_assignment

    elif data_type == 'standard':
        if not os.path.isfile('data/' + DB + '/' + DB + '_X.npy'):
            X = mol_build_X(list_SMILES)
            np.save('data/' + DB + '/' + DB + '_X', X)
        else:
            X = np.load('data/' + DB + '/' + DB + '_X.npy')

        list_assignment = np.zeros(X.shape[0])
        if DB == 'Tox21':
            skf = model_selection.KFold(n_folds, shuffle=True, random_state=92)
            skf.get_n_splits(X)
            ifold = 0
            for train_index, test_index in skf.split(X):
                list_assignment[test_index] = ifold
                ifold += 1
        else:
            skf = model_selection.StratifiedKFold(n_folds, shuffle=True, random_state=92)
            skf.get_n_splits(X, list_y)
            ifold = 0
            for train_index, test_index in skf.split(X, list_y):
                list_assignment[test_index] = ifold
                ifold += 1

    # import pdb; pdb.Pdb().set_trace()
    c = collections.Counter(list_assignment)
    print(c)
    folds = [np.where(list_assignment == cle)[0] for cle in list(c.keys())]

    fo = open('data/' + DB + '/' + DB + '_folds.txt', 'w')
    for ifold in range(n_folds):
        fo.write("ifold" + str(ifold) + '\n')
        if DB == 'Tox21':
            for iclass in range(12):
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
    mol_make_CV(sys.argv[1], tox21_CV, 'standard')
