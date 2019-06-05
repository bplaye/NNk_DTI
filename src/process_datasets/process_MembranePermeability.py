import csv
import pickle
import sys
from rdkit import Chem
import collections
import os
import numpy as np
from src.utils.mol_utils import mol_build_K, mol_build_X
import sklearn.model_selection as model_selection
from src.process_datasets.cross_validation import mol_make_CV, Kmedoid_cluster
from src.process_datasets.cross_validation import Khierarchical_cluster, Xkmeans_cluster


def process_DB():
    DB = 'MembranePermeability'

    list_ID, list_SMILES, list_y, dict_id2smile = [], [], [], {}
    reader = csv.reader(open('data/MembranePermeability/membrane_permeability.sdf.csv'),
                        delimiter=',')
    i = 0
    for row in reader:
        if i > 0:
            smile = row[1]
            m = Chem.MolFromSmiles(smile)
            if m is not None and smile != '':
                list_ID.append(str(i - 1))
                list_SMILES.append(row[1])
                list_y.append(float(row[0]))
                dict_id2smile[str(i - 1)] = row[1]
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
        f.write(str(s) + '\n')
    f.close()
    f = open('data/' + DB + '/' + DB + '_list_ID.tsv', 'w')
    for s in list_ID:
        f.write(s + '\n')
    f.close()

    print(len(list_SMILES))
    print(collections.Counter(list_y))


def MembranePermeability_CV(DB, data_type, list_ID, list_y, list_SMILES, dict_id2smile, n_folds):
    if data_type == 'kernel':
        if not os.path.isfile('data/' + DB + '/' + DB + '_K.npy'):
            K = mol_build_K(list_SMILES)
            np.save('data/' + DB + '/' + DB + '_K', K)
        else:
            K = np.load('data/' + DB + '/' + DB + '_K.npy')

        list_assignment = Khierarchical_cluster(K, n_folds)

    elif data_type == 'features':
        if not os.path.isfile('data/' + DB + '/' + DB + '_X.npy'):
            X = mol_build_X(list_SMILES)
            np.save('data/' + DB + '/' + DB + '_X', X)
        else:
            X = np.load('data/' + DB + '/' + DB + '_X.npy')

        list_assignment = Xkmeans_cluster(X, n_folds)

    elif data_type == 'standard':
        if not os.path.isfile('data/' + DB + '/' + DB + '_X.npy'):
            X = mol_build_X(list_SMILES)
            np.save('data/' + DB + '/' + DB + '_X', X)
        else:
            X = np.load('data/' + DB + '/' + DB + '_X.npy')

        list_assignment = np.zeros(X.shape[0])
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
    fo.write(str(c) + '\n')
    fo.close()

    return folds


if __name__ == "__main__":
    # process_DB()
    mol_make_CV('MembranePermeability', MembranePermeability_CV, 'standard')

