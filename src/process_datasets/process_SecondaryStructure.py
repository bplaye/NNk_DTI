import numpy as np
import gzip
import os
import csv
import pickle
import collections
from src.utils.prot_utils import LIST_AA
import sklearn.model_selection as model_selection
from src.process_datasets.cross_validation import prot_make_CV, Kmedoid_cluster
from src.process_datasets.cross_validation import Khierarchical_cluster, Xkmeans_cluster


def load_gz(path):  # load a .npy.gz file
    if path.endswith(".gz"):
        f = gzip.open(path, 'rb')
        return np.load(f)
    else:
        return np.load(path)


def process_DB():
    DB = 'SecondaryStructure'
    path = 'data/SecondaryStructure/'

    if not os.path.isfile(path + 'secondary_structure_X_aa' + '.npy'):
        # list_file = ['data/cullpdb+profile_6133_filtered.npy.gz',
        #              'data/cb513+profile_split1.npy.gz']
        # ces 2 fichiers h+'cullpdb+profile_6133_filtered.npy.gz')
        X1 = load_gz(path + 'cullpdb+profile_6133_filtered.npy.gz')
        X1 = np.reshape(X1, (-1, 700, 57))
        X2 = load_gz(path + 'cb513+profile_split1.npy.gz')
        X2 = np.reshape(X2, (-1, 700, 57))
        X = np.concatenate((X1, X2), axis=0)
        del X1
        del X2

        X_aa = X[:, :, :22]  # get aa
        X_label = X[:, :, 22:31]  # get label (1 class per aa)
        np.save(path + 'secondary_structure_X_aa', X_aa)
        np.save(path + 'secondary_structure_X_label', X_label)
        del X
    else:
        X_aa = np.load(path + 'secondary_structure_X_aa.npy')
        X_label = np.load(path + 'secondary_structure_X_label.npy')

    # print('nb samples:', X_aa.shape[0])

    labels = ['L', 'B', 'E', 'G', 'I', 'H', 'S', 'T']

    if not os.path.isfile(path + 'secondary_structure.fasta'):
        f_X, f_l = open(path + 'secondary_structure.fasta', 'w'), open(path + '/labels.csv', 'w')
        list_prot, list_X, list_label, list_lab = [], [], [], []
        aa_rejected_prot, max_seq_length = [], 700

        freq_aa = {aa: 0 for aa in LIST_AA}  # regarder fréquence d'aa dans la list_aa
        freq_label_along_seq = {l: [] for l in labels}
        list_seq_length = []  # regarder distribution de taille de séquence.
        for i_sample in range(X_aa.shape[0]):
            list_prot.append(i_sample)
            X_local = np.zeros((1, max_seq_length, len(LIST_AA), 1), dtype=np.int32)
            label_local = np.zeros((1, max_seq_length), dtype=np.int32)
            fasta, lab = '', ''
            found = False
            for i_aa in range(X_aa.shape[1]):
                if X_aa[i_sample, i_aa, -1] == 0:
                    X_local[0, i_aa, :, 0] = X_aa[i_sample, i_aa, :-1]
                    label_local[0, i_aa] = np.where(X_label[i_sample, i_aa, :] == 1)[0][0] + 1
                    if len(np.where(X_label[i_sample, i_aa, :] == 1)[0]) != 1:
                        # regarder qu'il y a bien une class par aa et que cette classe soit unique.
                        print('problem of labeling')
                    ind_aa = np.where(X_aa[i_sample, i_aa, :] == 1)[0][0]
                    freq_aa[LIST_AA[ind_aa]] += 1
                    fasta += LIST_AA[ind_aa]
                    ind_lab = np.where(X_label[i_sample, i_aa, :] == 1)[0][0]
                    lab += labels[ind_lab]
                    list_lab.append(labels[ind_lab])
                    freq_label_along_seq[labels[ind_lab]].append(i_aa)
                else:
                    found = True
                    list_seq_length.append(i_aa)
                    f_X.write('>' + str(i_sample) + '\n' + fasta + '\n')
                    f_l.write(str(i_sample) + ',' + lab + '\n')
                    list_X.append(X_local)
                    list_label.append(label_local)
                    break
            if found is False:
                list_seq_length.append(i_aa)
                f_X.write('>' + str(i_sample) + '\n' + fasta + '\n')
                f_l.write(str(i_sample) + ',' + lab + '\n')
                list_X.append(X_local)
                list_label.append(label_local)
        f_X.close()
        f_l.close()

    list_ID, list_FASTA, list_y, dict_id2fasta = [], [], [], {}
    aa_rejected_prot, max_seq_length = [], 0

    f_X, f_l = open(path + '/secondary_structure.fasta', 'r'), open(path + '/labels.csv', 'r')
    ind = 0
    for line in f_X:
        line = line.rstrip()
        if line[0] != '>':
            prot_id = ind
            issue = False
            for i_aa, aa in enumerate(line):
                if aa not in LIST_AA:
                    aa_rejected_prot.append(prot_id)
                    issue = True
                    break
            if issue is False:
                list_ID.append(str(prot_id))
                list_FASTA.append(line)
                dict_id2fasta[str(prot_id)] = line
                if len(line) > max_seq_length:
                    max_seq_length = len(line)
            ind += 1
    reader = csv.reader(f_l, delimiter=',')
    ir, i = 0, 0
    for row in reader:
        if ir not in aa_rejected_prot:
            list_y.append([labels.index(el) for el in list(row[1])])
            if len(row[1]) != len(list_FASTA[i]):
                print(str(ir) + ' not same size', len(row[1]), len(list_FASTA[i]))
            i += 1
        ir += 1
    f_X.close()
    f_l.close()
    print(len(list_y), len(list_y))
    print(len(list_FASTA))

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
    print('max_seq_length', max_seq_length)
    print(collections.Counter([el for y in list_y for el in y]))


def SecondaryStructure_CV(DB, data_type, list_ID, list_y, list_FASTA, dict_id2fasta, n_folds):
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

    import pdb; pdb.Pdb().set_trace()
    c = collections.Counter(list_assignment)
    folds = [np.where(list_assignment == cle)[0] for cle in list(c.keys())]

    fo = open('data/' + DB + '/' + DB + '_folds.txt', 'w')
    for ifold in range(n_folds):
        # import pdb; pdb.Pdb().set_trace()
        fo.write("ifold " + str(ifold) + '\t' +
                 str(collections.Counter([el for ll in list_y[folds[ifold]]
                                          for el in ll])))
        fo.write('\n')
        print("ifold " + str(ifold) + '\t' +
              str(collections.Counter([el for ll in list_y[folds[ifold]]
                                       for el in ll])))
    fo.close()

    return folds


if __name__ == "__main__":
    # process_DB()
    prot_make_CV('SecondaryStructure', SecondaryStructure_CV, 'standard')
