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
    DB = 'SCOPe'

    path = 'data/SCOPe/'

    labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g']

    f_in =  open(path + "astral-scopedom-seqres-gd-sel-gs-bib-40-2.07.fa", "r")
    # fout = open(path + 'SCOPe.fasta', 'w')
    list_prot, list_X, list_label, list_lab = [], [], [], []
    freq_aa = {aa: 0 for aa in LIST_AA}  # regarder fréquence d'aa dans la LIST_AA
    freq_label_along_seq = {l: [] for l in labels}
    list_seq_length, length_rejected_prot, aa_rejected_prot = [], [], []
    # regarder distribution de taille de séquence.
    max_seq_length, list_fasta = 0, []
    found = False
    issue = True
    for line in f_in:
        line = line.rstrip()
        if line[0] == ">":
            if issue is False:
                # if len(fasta) < max_seq_length:
                        # X_local = np.zeros((1, max_seq_length, len(LIST_AA), 1), dtype=np.int32)
                        # label_local = np.zeros((1, max_seq_length, len(labels)), dtype=np.int32)

                        # iaa=0
                        # for aa in fasta:
                        #     freq_aa[aa]+=1
                        #     X_local[0, iaa, dict_aa[aa], 0] = 1
                        #     iaa+=1

                # list_X.append(X_local)
                list_label.append(labels.index(prot_family_id))
                list_seq_length.append(len(fasta))
                if len(fasta) > max_seq_length:
                    max_seq_length = len(fasta)
                list_fasta.append(fasta)
                list_prot.append(prot_id)
                # fout.write('>'+prot_id+'\n'+fasta+'\n')
                # else:
                #     length_rejected_prot.append(prot_id)
                #     # n_ = int(np.ceil(float(len(fasta))/1000.))
                #     # m_ = int(np.ceil(n_ * 1000 - len(fasta)))
                #     # fastas = [fasta[i:i+1000] for i in range(0, len(fasta), 1000-m_)]
                #     # for ifasta, fasta in enumerate(fastas):
                #     #     X_local = np.zeros((1, max_seq_length, len(LIST_AA), 1), dtype=np.int32)
                #     #     label_local = np.zeros((1, max_seq_length, len(labels)), dtype=np.int32)

                #     #     iaa=0
                #     #     for aa in fasta:
                #     #         freq_aa[aa]+=1
                #     #         X_local[0, iaa, dict_aa[aa], 0] = 1
                #     #         iaa+=1

                #     #     list_X.append(X_local)
                #     #     list_seq_length.append(len(fasta))
                #     #     list_label.append(labels.index(prot_family_id))
                #     #     list_prot.append(prot_id+'_'+str(ifasta))
            line = line.split(' ')
            prot_id = line[0][1:]
            prot_family_id = line[1].split('.')[0]
            found = False
            issue = False
        else:
            if found is False:
                fasta = ''
                found = True
                issue = False
            line = line.upper()
            for l in line:
                if l not in LIST_AA:
                    issue = True
                    break
            fasta += line

    f_in.close()

    print('number of proteins in the dataset:', len(list_prot))
    print('number of rejected proteins because unknown aa:', len(aa_rejected_prot))
    print('number of rejected proteins because too long:', len(length_rejected_prot))

    list_ID, list_FASTA, list_y, dict_id2fasta = list_prot, list_fasta, list_label, {}
    for ip in range(len(list_prot)):
        dict_id2fasta[list_prot[ip]] = list_FASTA[ip]
        # if len(list_FASTA[ip]) != len(list_y[ip]):
        #     print(ip, len(list_FASTA[ip]), len(list_y[ip]))

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
        print(cle)
        print(valeur)
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
    print(collections.Counter(list_y))  # [el for y in list_y for el in y]))
    print('max_seq_length', max_seq_length)


def SCOPe_CV(DB, data_type, list_ID, list_y, list_FASTA, dict_id2fasta, n_folds):
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
    prot_make_CV('SCOPe', SCOPe_CV, 'standard')

