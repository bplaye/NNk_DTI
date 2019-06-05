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
    DB = 'TransmembraneRegions'

    path = 'data/TransmembraneRegions/'

    labels = ['T', 'S', '.', 't']

    f_in = open(path + 'euk.2.how', 'r')

    f, prot_id, fout = False, None, open(path + 'transmembrane_regions.fasta', 'w')
    # list_prot, list_X, list_label, list_lab = [], [], [], []
    list_label, list_prot, list_fasta, aa_rejected_prot = [], [], [], []
    max_seq_length = 0
    # freq_aa = {aa: 0 for aa in LIST_AA}  # regarder fréquence d'aa dans la list_aa
    # freq_label_along_seq = {l: [] for l in labels}
    # list_seq_length, aa_rejected_prot, length_rejected_prot = [], [], []
    # regarder distribution de taille de séquence.
    for line in f_in:
        line = line.rstrip()
        if line[0] == ' ':
            if prot_id is not None:
                if issue is False:
                    # if len(fasta) < max_seq_length:
                    #     X_local = np.zeros((1, max_seq_length, len(list_aa), 1), dtype=np.int32)
                    #     label_local = np.zeros((1, max_seq_length), dtype=np.int32)

                        # iaa=0
                        # for aa in fasta:
                        #     freq_aa[aa]+=1
                        #     X_local[0, iaa, dict_aa[aa], 0] = 1
                        #     iaa+=1

                    # il=0
                    # for l in label:
                    #     list_lab.append(l)
                    #     freq_label_along_seq[l].append(il)
                    #     l = 'T' if l == 't' else l
                    #     label_local[0, il] = labels.index(l) + 1
                    #     il +=1

                    label = list(label.replace('t', 'T'))
                    label = [labels.index(el) for el in label]
                    list_label.append(label)
                    # list_seq_length.append(len(fasta))
                    if len(fasta) > max_seq_length:
                        max_seq_length = len(fasta)
                    list_prot.append(prot_id)
                    list_fasta.append(fasta)
                    fout.write('>' + prot_id + '\n' + fasta + '\n')

            prot_id = line.strip(' ').split(' ')[1]  # .split('_')[0]
            fasta = ''
            found = False
            issue = False
        elif '.' not in line != "SSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS":
            fasta += line
            for aa in line:
                # if iaa==max_seq_length:
                #    length_rejected_prot.append(prot_id)
                #    issue = True
                #    break
                if aa not in LIST_AA:
                    aa_rejected_prot.append(prot_id)
                    issue = True
                    break
        else:
            if found is False:
                label = ''
                found = True
                il = 0
            if issue is False:
                label += line
    f_in.close()
    print('number of proteins in the dataset:', len(list_prot))
    print('number of rejected proteins because unknown aa:', len(aa_rejected_prot))
    # print('number of rejected proteins because too long:', len(length_rejected_prot))

    # X = np.concatenate(list_X, axis=0)
    # np.save(path+'TransmembraneSeq_X', X)

    list_ID, list_FASTA, list_y, dict_id2fasta = list_prot, list_fasta, list_label, {}
    for ip in range(len(list_prot)):
        dict_id2fasta[list_prot[ip]] = list_FASTA[ip]
        if len(list_FASTA[ip]) != len(list_y[ip]):
            print(ip, len(list_FASTA[ip]), len(list_y[ip]))

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

    print(len(list_FASTA))
    print(collections.Counter([el for y in list_y for el in y]))
    print('max_seq_length', max_seq_length)


def TransmembraneRegions_CV(DB, data_type, list_ID, list_y, list_FASTA, dict_id2fasta, n_folds):
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
        fo.write("ifold " + str(ifold) + '\t' +
                 str(collections.Counter([el for ll in list_y[folds[ifold]]
                                          for el in ll])))
        fo.write('\n')
        print("ifold " + str(ifold), str(collections.Counter([el for ll in list_y[folds[ifold]]
                                                              for el in ll])))
    fo.close()

    return folds


if __name__ == "__main__":
    # process_DB()
    prot_make_CV('TransmembraneRegions', TransmembraneRegions_CV, 'standard')
