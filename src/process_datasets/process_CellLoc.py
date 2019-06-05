import pickle
import collections
from src.utils.prot_utils import LIST_AA
import os
import sys
import numpy as np
import sklearn.model_selection as model_selection
from src.process_datasets.cross_validation import prot_make_CV, Kmedoid_cluster
from src.process_datasets.cross_validation import Khierarchical_cluster, Xkmeans_cluster


def process_DB():
    DB = 'CellLoc'

    list_ID, list_FASTA, list_y, dict_id2fasta = [], [], [], {}
    file_list = ['chloroplast', 'cytoplasmic', 'ER',
                 'extracellular', 'Golgi', 'vacuolar',
                 'lysosomal', 'mitochondrial', 'nuclear',
                 'peroxisomal', 'plasma_membrane']
    aa_rejected_prot, max_seq_length = [], 0
    for ifile, file in enumerate(file_list):
        f = open('data/CellLoc/' + file + '.fasta', 'r')
        for line in f:
            line = line.rstrip()
            if line[0] == '>':
                prot_id = line[1:7]  # line.strip(' ')[0][1:]
            else:
                issue = False
                for i_aa, aa in enumerate(line):
                    if aa not in LIST_AA:
                        aa_rejected_prot.append(prot_id)
                        issue = True
                        break
                if issue is False:
                    list_ID.append(prot_id)
                    list_FASTA.append(line)
                    list_y.append(ifile)
                    dict_id2fasta[prot_id] = line
                    if len(line) > max_seq_length:
                        max_seq_length = len(line)
        f.close()
    print('number of proteins in the dataset:', len(list_ID))
    print('number of rejected proteins because unknown aa:', len(aa_rejected_prot))
    print('max_seq_length', max_seq_length)

    pickle.dump(dict_id2fasta,
                open('data/' + DB + '/' + DB + '_dict_ID2FASTA.data', 'wb'))
    # pickle.dump(dict_uniprot2fasta,
    #             open(root + 'data/' + DB + '/' + DB + '_dict_ID2FASTA.data', 'wb'))
    pickle.dump(list_FASTA,
                open('data/' + DB + '/' + DB + '_list_FASTA.data', 'wb'))
    pickle.dump(list_y,
                open('data/' + DB + '/' + DB + '_list_y.data', 'wb'))
    pickle.dump(list_ID,
                open('data/' + DB + '/' + DB + '_list_ID.data', 'wb'))

    f = open('data/' + DB + '/' + DB + '_dict_ID2FASTA.tsv', 'w')
    for cle, valeur in dict_id2fasta.items():
        f.write(cle + '\t' + valeur + '\n')
    f.close()
    f = open('data/' + DB + '/' + DB + '_list_FASTA.tsv', 'w')
    for s in list_FASTA:
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

    print(len(list_FASTA))
    print(collections.Counter(list_y))


def CellLoc_CV(DB, data_type, list_ID, list_y, list_FASTA, dict_id2fasta, n_folds):
    if data_type == 'kernel':
        if not os.path.isfile('data/' + DB + '/' + DB + '_K.npy'):
            print('data/' + DB + '/' + DB + '_K.npy', 'does not exist')
        else:
            K = np.load('data/' + DB + '/' + DB + '_K.npy')

        list_assignment = Khierarchical_cluster(K, n_folds)

    elif data_type == 'features':
        if not os.path.isfile('data/' + DB + '/' + DB + '_X.npy'):
            print('data/' + DB + '/' + DB + '_X.npy', 'does not exist')
        else:
            X = np.load('data/' + DB + '/' + DB + '_X.npy')

        list_assignment = Xkmeans_cluster(X, n_folds)

    elif data_type == 'standard':
        X = np.zeros((len(list_ID), 1))

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
    for ifold in range(n_folds):
        fo.write("ifold " + str(ifold) + '\t' + str(collections.Counter(list_y[folds[ifold]])))
        fo.write('\n')
        print("ifold " + str(ifold), collections.Counter(list_y[folds[ifold]]))
    fo.close()

    return folds


if __name__ == "__main__":
    # process_DB()
    prot_make_CV('CellLoc', CellLoc_CV, 'standard')
