import csv
import numpy as np
import pickle
import sys
import os
from sklearn.decomposition import KernelPCA
from src.utils.DB_utils import LIST_DTI_DATASETS


LIST_AA = ['A', 'C', 'E', 'D', 'G', 'F', 'I', 'H', 'K', 'M', 'L', 'N', 'Q', 'P', 'S', 'R', 'T',
           'W', 'V', 'Y', 'X']
dict_aa2ind = {aa: ind for ind, aa in enumerate(LIST_AA)}
MAX_SEQ_LENGTH = 5000
NB_PROT_FEATURES = {'DrugBankEC': 275, 'DrugBankECstand': 275,
                    'DrugBankH': 1876, 'DrugBankHstand': 1876, 'CellLoc': 3915, 'SCOPe': 8876}
NB_PROT_FEATURES = 1920
NB_AA_ATTRIBUTES = len(LIST_AA)


def featurise_prot(fasta, padding):
    if padding:
        X_prot = np.zeros((len(LIST_AA), MAX_SEQ_LENGTH), dtype=np.float32)
    else:
        X_prot = np.zeros((len(LIST_AA), len(fasta)), dtype=np.float32)

    for iaa, aa in enumerate(fasta):
        X_prot[dict_aa2ind[aa], iaa] = 1.

    prot_length = np.asarray([len(fasta)], dtype=np.int32)
    return X_prot, prot_length


def padSeq(seq, n_aa):
    if seq.shape[1] != n_aa:
        seq = np.concatenate((seq, np.zeros((seq.shape[0], n_aa - seq.shape[1]))), axis=1)
    return seq


def write_fasta_file(DB):
    list_ID = np.array(pickle.load(open('data/' + DB + '/' + DB + '_list_ID.data', 'rb')))
    list_FASTA = pickle.load(open('data/' + DB + '/' + DB + '_list_FASTA.data', 'rb'))

    fo = open('data/' + DB + '/' + DB + '_fasta.fasta', 'w')
    for isample in range(len(list_ID)):
        fo.write('>' + list_ID[isample] + '\n' + list_FASTA[isample] + '\n')
    fo.close()


def make_temp_Kprot(DB, index):
    list_FASTA = pickle.load(open('data/' + DB + '/' + DB + '_list_FASTA.data', 'rb'))
    nb_prot = len(list(list_FASTA))

    path = 'data/' + DB + '/' + 'res'
    if not os.path.isdir(path):
        os.mkdir(path)

    FASTA1 = list_FASTA[index]
    outf = path + '/LA_' + str(index) + '.txt'
    if not os.path.isfile(outf):
        for j in range(index, nb_prot):
            FASTA2 = list_FASTA[j]
            com = '$HOME/LAkernel-0.2/LAkernel_direct ' + FASTA1 + ' ' + FASTA2 + ' >> ' + outf
            cmd = os.popen(com)
            cmd.read()
    else:
        j = index
        for line in open(outf, 'r'):
            j += 1
        if j != nb_prot:
            os.remove(outf)
            for j in range(index, nb_prot):
                FASTA2 = list_FASTA[j]
                com = '$HOME/LAkernel-0.2/LAkernel_direct ' + FASTA1 + ' ' + FASTA2 + ' >> ' + outf
                cmd = os.popen(com)
                cmd.read()
        else:
            print('file exist and good size')


def check_temp_Kprot(DB, redo):
    path = 'data/' + DB + '/'
    list_FASTA = pickle.load(open(path + DB + '_list_FASTA.data', 'rb'))
    nb_prot = len(list(list_FASTA))

    list_ = []
    for index in range(nb_prot):
        outf = path + 'res/LA_' + str(index) + '.txt'
        if not os.path.isfile(outf):
            list_.append(index)
        else:
            j = index
            for line in open(path + 'res/LA_' + str(index) + '.txt', 'r'):
                j += 1
            if j != nb_prot:
                list_.append(index)
    print(list_)

    if redo:
        for index in list_:
            print(index)
            FASTA1 = list_FASTA[index]
            outf = path + 'res/LA_' + str(index) + '.txt'
            if os.path.isfile(outf):
                os.remove(outf)
            for j in range(index, nb_prot):
                FASTA2 = list_FASTA[j]
                com = '$HOME/LAkernel-0.2/LAkernel_direct ' + FASTA1 + ' ' + FASTA2 + ' >> ' + outf
                cmd = os.popen(com)
                cmd.read()


def make_group_Kprot(DB):
    import math
    from sklearn.preprocessing import KernelCenterer
    path = 'data/' + DB + '/'
    list_FASTA = pickle.load(open(path + DB + '_list_FASTA.data', 'rb'))
    nb_prot = len(list(list_FASTA))

    X = np.zeros((nb_prot, nb_prot))
    for i in range(nb_prot):
        # print(i)
        j = i
        for line in open(path + 'res/LA_' + str(i) + '.txt', 'r'):
            r = float(line.rstrip())
            X[i, j] = r
            X[j, i] = X[i, j]
            j += 1
        if j != nb_prot:
            print(i, 'not total')
            exit(1)

    X = KernelCenterer().fit_transform(X)
    K = np.zeros((nb_prot, nb_prot))
    for i in range(nb_prot):
        for j in range(i, nb_prot):
            K[i, j] = X[i, j] / math.sqrt(X[i, i] * X[j, j])
            K[j, i] = K[i, j]
    pickle.dump(K, open(path + DB + '_Kprot.data', 'wb'), protocol=2)


def prot_build_X(DB, ratio=0.9):
    path = 'data/' + DB + '/'
    K = pickle.load(open(path + DB + '_Kprot.data', 'rb'))
    kpca = KernelPCA(kernel='precomputed')
    X = kpca.fit_transform(K)
    print(X.shape)
    s = np.sum(kpca.lambdas_)
    for k in range(1, len(kpca.lambdas_) + 1):
        if np.sum(kpca.lambdas_[:k]) / s > ratio:
            print(k, 'components explain ' + str(round(ratio * 100)) + '% of the data')
            X = X[:, :k]
            print(X.shape)
            return X


if __name__ == "__main__":
    # write_fasta_file(sys.argv[1])

    if 'kernel' in sys.argv[2]:
        # make_temp_Kprot(sys.argv[1], int(sys.argv[2].split('_')[1]))
        # check_temp_Kprot(sys.argv[1], redo=False)
        make_group_Kprot(sys.argv[1])

    elif 'fasta' in sys.argv[2]:
        DB = sys.argv[1]
        dict_id2fasta = pickle.load(open('data/' + DB + '/' + DB + '_dict_ID2FASTA.data', 'rb'))
        fo = open('data/' + DB + '/' + DB + '_allprots.fasta', 'w')
        for cle, val in dict_id2fasta.items():
            fo.write('>' + cle + '\n' + val + '\n')
        fo.close()

    elif sys.argv[2] == 'feature':
        dict_nb_descriptors_per_feature = {'AAC': 20, 'DC': 400, 'MoreauBroto': 240, 'Moran': 240,
                                           'Geary': 240, 'CTDC': 21, 'CTDT': 21, 'CTDD': 105,
                                           'CTriad': 343, 'SOCN': 60, 'QSO': 100, 'PAAC': 50,
                                           'APAAC': 80}
        list_descriptors = ['AAC', 'DC', 'MoreauBroto', 'Moran', 'Geary', 'CTDC', 'CTDT', 'CTDD',
                            'CTriad', 'SOCN', 'QSO', 'PAAC', 'APAAC']

        DB = sys.argv[1]
        if DB not in LIST_DTI_DATASETS:
            list_ID = pickle.load(open('data/' + DB + '/' + DB + '_list_ID.data', 'rb'))
        else:
            list_ID = pickle.load(open('data/' + DB + '/' + DB + '_list_ID_prot.data', 'rb'))
        X = np.zeros((len(list_ID), 1920))
        dict_id2features = {}
        if "DrugBankH" not in DB:
            reader = csv.reader(open('data/' + DB + '/' + DB + '_standardfeatures.csv', 'r'),
                                delimiter=',')
            irow = 0
            for row in reader:
                if irow != 0:
                    dict_id2features[row[0]] = np.array([float(el) for el in row[1:]])
                irow += 1
            del reader
        else:
            reader = csv.reader(
                open('data/' + DB + '/' + DB + '_standardfeatures_more.tsv', 'r'),
                delimiter='\t')
            irow = 0
            for row in reader:
                if irow != 0:
                    dict_id2features[row[0]] = np.array([float(el) for el in row[1:]])
                else:
                    list_all_prot_features = row[1:]
                irow += 1
            del reader

            for indr in range(3):
                reader = csv.reader(
                    open('data/' + DB + '/' + DB + '_standardfeatures_less_' + str(indr) + '.tsv',
                         'r'), delimiter='\t')
                irow = 0
                nbf = 0
                for row in reader:
                    if irow == 0:
                        list_all_prot_features_temp = row[1:]
                    else:
                        dict_id2features[row[0]] = np.zeros(len(list_all_prot_features))
                        for ip, protf in enumerate(list_all_prot_features_temp):
                            if row[1 + ip] != "NA":
                                dict_id2features[row[0]]\
                                    [list_all_prot_features.index(protf)] = float(row[1 + ip])
                                nbf += 1
                    irow += 1
                print(indr, len(list_all_prot_features), nbf)
                del reader

        for iid, uniprotid in enumerate(list_ID):
            X[iid, :] = dict_id2features[uniprotid]

        # if DB not in LIST_DTI_DATASETS:
        #     list_ID = pickle.load(open('data/' + DB + '/' + DB + '_list_ID.data', 'rb'))
        # else:
        #     list_ID = pickle.load(open('data/' + DB + '/' + DB + '_list_ID_prot.data', 'rb'))
        # X = prot_build_X(DB)
        # if DB not in LIST_DTI_DATASETS:
        #     pickle.dump({list_ID[i]: X[i, :] for i in range(len(list_ID))},
        #                 open('data/' + DB + '/' + DB + '_dict_ID2features.data', 'wb'))
        # else:
        #     pickle.dump({list_ID[i]: X[i, :] for i in range(len(list_ID))},
        #                 open('data/' + DB + '/' + DB + '_dict_ID2protfeatures.data', 'wb'))
        if DB not in LIST_DTI_DATASETS:
            pickle.dump(dict_id2features,
                        open('data/' + DB + '/' + DB + '_dict_ID2features.data', 'wb'))
        else:
            pickle.dump(dict_id2features,
                        open('data/' + DB + '/' + DB + '_dict_ID2protfeatures.data', 'wb'))
        pickle.dump(X, open('data/' + DB + '/' + DB + '_Xprot.data', 'wb'))

    elif sys.argv[2] == 'DTI_dict':
        DB = sys.argv[1]
        dict_ID2molfeatures = \
            pickle.load(open('data/' + DB + '/' + DB + '_dict_ID2molfeatures.data', 'rb'))
        dict_ID2protfeatures = \
            pickle.load(open('data/' + DB + '/' + DB + '_dict_ID2protfeatures.data', 'rb'))
        dict_ID2features = dict_ID2molfeatures.update(dict_ID2protfeatures)
        pickle.dump(dict_ID2features,
                    open('data/' + DB + '/' + DB + '_dict_ID2features.data', 'wb'))

