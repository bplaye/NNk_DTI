import pickle
import numpy as np
import sys
from rdkit import Chem
from src.utils.prot_utils import LIST_AA
import collections
from src.process_datasets.cross_validation import Khierarchical_cluster
import sklearn.model_selection as model_selection
from src.utils.DB_utils import data_file, y_file


def block_diag(*arrs):
    """Create a block diagonal matrix from the provided arrays.

    Given the inputs `A`, `B` and `C`, the output will have these
    arrays arranged on the diagonal::

        [[A, 0, 0],
         [0, B, 0],
         [0, 0, C]]

    If all the input arrays are square, the output is known as a
    block diagonal matrix.

    Parameters
    ----------
    A, B, C, ... : array-like, up to 2D
        Input arrays.  A 1D array or array-like sequence with length n is
        treated as a 2D array with shape (1,n).

    Returns
    -------
    D : ndarray
        Array with `A`, `B`, `C`, ... on the diagonal.  `D` has the
        same dtype as `A`.

    References
    ----------
    .. [1] Wikipedia, "Block matrix",
           http://en.wikipedia.org/wiki/Block_diagonal_matrix

    Examples
    --------
    >>> A = [[1, 0],
    ...      [0, 1]]
    >>> B = [[3, 4, 5],
    ...      [6, 7, 8]]
    >>> C = [[7]]
    >>> print(block_diag(A, B, C))
    [[1 0 0 0 0 0]
     [0 1 0 0 0 0]
     [0 0 3 4 5 0]
     [0 0 6 7 8 0]
     [0 0 0 0 0 7]]
    >>> block_diag(1.0, [2, 3], [[4, 5], [6, 7]])
    array([[ 1.,  0.,  0.,  0.,  0.],
           [ 0.,  2.,  3.,  0.,  0.],
           [ 0.,  0.,  0.,  4.,  5.],
           [ 0.,  0.,  0.,  6.,  7.]])

    """
    if arrs == ():
        arrs = ([],)
    arrs = [np.atleast_2d(a) for a in arrs]

    bad_args = [k for k in range(len(arrs)) if arrs[k].ndim > 2]
    if bad_args:
        raise ValueError("arguments in the following positions have dimension " +
                         "greater than 2: %s" % bad_args)

    shapes = np.array([a.shape for a in arrs])
    out = np.zeros(np.sum(shapes, axis=0), dtype=arrs[0].dtype)

    r, c = 0, 0
    for i, (rr, cc) in enumerate(shapes):
        out[r:r + rr, c:c + cc] = arrs[i]
        r += rr
        c += cc
    return out


def check_mol_weight(DB_type, m, dict_id2smile, weight, smile, dbid):
    # if DB_type == 'H':
    #     if weight > 100 and weight < 800 and m is not None and smile != '':
    #         dict_id2smile[dbid] = smile
    # elif DB_type == 'S0' or DB_type == 'S0h':
    if m is not None and smile != '':
        dict_id2smile[dbid] = smile
    return dict_id2smile


def get_specie_per_uniprot():
    import csv
    dict_specie_per_prot = {}
    reader = \
        csv.reader(open('data/DrugBank/drugbank_small_molecule_target_polypeptide_ids.csv/all.csv',
                        'r'), delimiter=',')
    i = 0
    for row in reader:
        if i > 0:
            dict_specie_per_prot[row[0]] = row[11]
        i += 1
    return dict_specie_per_prot


def get_all_DrugBank_smiles(DB_type, root='./'):
    dict_id2smile = {}
    dbid, smile = None, None
    found_id, found_smile, found_weight = False, False, False
    f = open(root + 'data/DrugBank/structures.sdf', 'r')
    for line in f:
        line = line.rstrip()
        if found_id:
            if smile is not None:
                m = Chem.MolFromSmiles(smile)
                dict_id2smile = check_mol_weight(DB_type, m, dict_id2smile, weight, smile, dbid)
            dbid = line
            found_id = False
        if found_smile:
            smile = line
            found_smile = False
        if found_weight:
            weight = float(line)
            found_weight = False

        if line == "> <DATABASE_ID>":
            found_id = True
        elif line == "> <SMILES>":
            found_smile = True
        elif line == '> <MOLECULAR_WEIGHT>':
            found_weight = True
    f.close()

    m = Chem.MolFromSmiles(smile)
    dict_id2smile = check_mol_weight(DB_type, m, dict_id2smile, weight, smile, dbid)

    return dict_id2smile


def check_prot_length(DB_type, fasta, dict_uniprot2seq, dbid, list_ligand, list_inter):
    # if DB_type == 'S':
    #     aa_bool = True
    #     for aa in fasta:
    #         if aa not in LIST_AA:
    #             aa_bool = False
    #             break
    #     if len(fasta) < 1000 and aa_bool is True:
    #         dict_uniprot2seq[dbid] = fasta
    #         for ligand in list_ligand:
    #             list_inter.append((dbid, ligand))
    # elif DB_type == 'S0':
    #     dict_uniprot2seq[dbid] = fasta
    #     for ligand in list_ligand:
    #         list_inter.append((dbid, ligand))
    if 'DrugBankH' in DB_type:
        aa_bool = True
        for aa in fasta:
            if aa not in LIST_AA:
                aa_bool = False
                break

        if aa_bool is True:
            dict_specie_per_prot = get_specie_per_uniprot()
            if dict_specie_per_prot[dbid] == 'Human':
                dict_uniprot2seq[dbid] = fasta
                for ligand in list_ligand:
                    list_inter.append((dbid, ligand))
    if 'DrugBankEC' in DB_type:
        aa_bool = True
        for aa in fasta:
            if aa not in LIST_AA:
                aa_bool = False
                break

        if aa_bool is True:
            dict_specie_per_prot = get_specie_per_uniprot()
            if dict_specie_per_prot[dbid] == 'Escherichia coli (strain K12)':
                dict_uniprot2seq[dbid] = fasta
                for ligand in list_ligand:
                    list_inter.append((dbid, ligand))
    return list_inter, dict_uniprot2seq


def get_all_DrugBank_fasta(DB_type, root='./'):
    f = open(root +
             'data/DrugBank/drugbank_small_molecule_target_polypeptide_sequences.fasta/' +
             'protein.fasta', 'r')
    dict_uniprot2seq, list_inter = {}, []
    dbid, fasta = None, None
    for line in f:
        line = line.rstrip()
        if line[0] != '>':
            fasta += line
        else:
            if fasta is not None:
                list_inter, dict_uniprot2seq = check_prot_length(DB_type, fasta, dict_uniprot2seq,
                                                                 dbid, list_ligand, list_inter)
            dbid = line.split('|')[1].split(' ')[0]
            list_ligand = line.split('(')[1].split(')')[0].split('; ')
            if type(list_ligand) is not list:
                list_ligand = [list_ligand]
            fasta = ''
    f.close()

    # for the final protein
    list_inter, dict_uniprot2seq = check_prot_length(DB_type, fasta, dict_uniprot2seq,
                                                     dbid, list_ligand, list_inter)
    return dict_uniprot2seq, list_inter


def process_DB(DB, root='./'):
    # drugbank v5.1.1
    # interactions from these filters: Small Molecules targets;
    # mol with known SMILES, loadable with Chem, µM between 100 and 800
    # prot with all aa in list (no weird aa), fasta connu, length < 1000
    dict_id2smile = get_all_DrugBank_smiles(DB)
    dict_uniprot2fasta, list_inter = get_all_DrugBank_fasta(DB)

    dict_id2smile_inter, dict_uniprot2fasta_inter, list_interactions = {}, {}, []
    print('len(list_inter)', len(list_inter))
    max_seq_length = 0
    for couple in list_inter:
        if couple[1] in list(dict_id2smile.keys()) and \
                couple[0] in list(dict_uniprot2fasta.keys()):
            list_interactions.append(couple)
            dict_id2smile_inter[couple[1]] = dict_id2smile[couple[1]]
            dict_uniprot2fasta_inter[couple[0]] = dict_uniprot2fasta[couple[0]]
            if len(dict_uniprot2fasta_inter[couple[0]]) > max_seq_length:
                max_seq_length = len(dict_uniprot2fasta_inter[couple[0]])
    print('nb interactions', len(list_interactions))
    print('max_seq_length', max_seq_length)

    pickle.dump(dict_id2smile_inter,
                open(root + 'data/' + DB + '/' + DB + '_dict_ID2SMILES.data', 'wb'))
    pickle.dump(dict_uniprot2fasta_inter,
                open(root + 'data/' + DB + '/' + DB + '_dict_ID2FASTA.data', 'wb'))
    pickle.dump(list_interactions,
                open(root + 'data/' + DB + '/' + DB + '_list_interactions.data', 'wb'))

    dict_ind2prot, dict_prot2ind, dict_ind2mol, dict_mol2ind = {}, {}, {}, {}
    f = open(root + 'data/' + DB + '/' + DB + '_interactions.tsv', 'w')
    for couple in list_interactions:
        f.write(couple[0] + '\t' + couple[1] + '\n')
    f.close()
    f = open(root + 'data/' + DB + '/' + DB + '_ID2FASTA.tsv', 'w')
    for ip, prot in enumerate(list(dict_uniprot2fasta_inter.keys())):
        f.write(prot + '\t' + dict_uniprot2fasta_inter[prot] + '\n')
        dict_ind2prot[ip] = prot
        dict_prot2ind[prot] = ip
    f.close()
    f = open(root + 'data/' + DB + '/' + DB + '_ID2SMILES.tsv', 'w')
    for im, mol in enumerate(list(dict_id2smile_inter.keys())):
        f.write(mol + '\t' + dict_id2smile_inter[mol] + '\n')
        dict_ind2mol[im] = mol
        dict_mol2ind[mol] = im
    f.close()

    intMat = np.zeros((len(list(dict_uniprot2fasta_inter.keys())),
                       len(list(dict_id2smile_inter.keys()))),
                      dtype=np.int32)
    for couple in list_interactions:
        intMat[dict_prot2ind[couple[0]], dict_mol2ind[couple[1]]] = 1
    np.save(root + 'data/' + DB + '/' + DB + '_intMat', intMat)
    pickle.dump(dict_ind2prot,
                open(root + 'data/' + DB + '/' + DB + '_dict_ind2prot.data', 'wb'), protocol=2)
    pickle.dump(dict_prot2ind,
                open(root + 'data/' + DB + '/' + DB + '_dict_prot2ind.data', 'wb'), protocol=2)
    pickle.dump(dict_ind2mol,
                open(root + 'data/' + DB + '/' + DB + '_dict_ind2mol.data', 'wb'), protocol=2)
    pickle.dump(dict_mol2ind,
                open(root + 'data/' + DB + '/' + DB + '_dict_mol2ind.data', 'wb'), protocol=2)

    list_SMILES = [dict_id2smile_inter[dict_ind2mol[i]] for i in range(len(dict_ind2mol.keys()))]
    list_FASTA = [dict_uniprot2fasta_inter[dict_ind2prot[i]]
                  for i in range(len(dict_ind2prot.keys()))]
    list_ID_prot = [dict_ind2prot[i] for i in range(len(dict_ind2prot.keys()))]
    list_ID_mol = [dict_ind2mol[i] for i in range(len(dict_ind2mol.keys()))]
    pickle.dump(list_SMILES,
                open(root + 'data/' + DB + '/' + DB + '_list_SMILES.data', 'wb'), protocol=2)
    pickle.dump(list_FASTA,
                open(root + 'data/' + DB + '/' + DB + '_list_FASTA.data', 'wb'), protocol=2)
    pickle.dump(list_ID_prot,
                open(root + 'data/' + DB + '/' + DB + '_list_ID_prot.data', 'wb'), protocol=2)
    pickle.dump(list_ID_mol,
                open(root + 'data/' + DB + '/' + DB + '_list_ID_mol.data', 'wb'), protocol=2)


def make_Kcouple(DB, list_couple):
    import math
    from sklearn.preprocessing import KernelCenterer
    dict_ligand, dict_target, intMat, dict_ind2prot, dict_ind2mol, dict_prot2ind, \
        dict_mol2ind = get_DB(DB)
    Kmol = pickle.load(open('data/' + DB + '/' + DB + '_Kmol.data', 'rb'))
    Kprot = pickle.load(open('data/' + DB + '/' + DB + '_Kprot.data', 'rb'))
    nb_couple = len(list_couple)
    print(nb_couple)

    X = np.zeros((nb_couple, nb_couple))
    for i in range(nb_couple):
        ind1_prot = dict_prot2ind[list_couple[i][0]]
        ind1_mol = dict_mol2ind[list_couple[i][1]]
        for j in range(i, nb_couple):
            ind2_prot = dict_prot2ind[list_couple[j][0]]
            ind2_mol = dict_mol2ind[list_couple[j][1]]

            X[i, j] = Kmol[ind1_mol, ind2_mol] * Kprot[ind1_prot, ind2_prot]
            X[j, i] = X[i, j]

    X = KernelCenterer().fit_transform(X)
    K = np.zeros((nb_couple, nb_couple))
    for i in range(nb_couple):
        for j in range(i, nb_couple):
            K[i, j] = X[i, j] / math.sqrt(X[i, i] * X[j, j])
            K[j, i] = K[i, j]
    return K


def setting_makefold(setting, test_index, intMat, ratio, dict_ind2prot, dict_ind2mol, seed):
    ind_inter, ind_non_inter = np.where(intMat == 1), np.where(intMat == 0)
    if setting == 2:
        index = np.isin(ind_inter[0], test_index)
        ind_inter = (ind_inter[0][index], ind_inter[1][index])
        index = np.isin(ind_non_inter[0], test_index)
        ind_non_inter = (ind_non_inter[0][index], ind_non_inter[1][index])
    elif setting == 3:
        index = np.isin(ind_inter[1], test_index)
        ind_inter = (ind_inter[0][index], ind_inter[1][index])
        index = np.isin(ind_non_inter[1], test_index)
        ind_non_inter = (ind_non_inter[0][index], ind_non_inter[1][index])
    elif setting == 4:
        val_index, test_index = test_index
        test_index_mol, test_index_prot = test_index
        val_index_mol, val_index_prot = val_index

        te_index_mol = np.isin(ind_inter[1], test_index_mol)
        te_index_prot = np.isin(ind_inter[0], test_index_prot)
        te_index = te_index_mol * te_index_prot
        te_ind_inter = (ind_inter[0][te_index], ind_inter[1][te_index])

        te_index_mol = np.isin(ind_non_inter[1], test_index_mol)
        te_index_prot = np.isin(ind_non_inter[0], test_index_prot)
        te_index = te_index_mol * te_index_prot
        te_ind_non_inter = (ind_non_inter[0][te_index], ind_non_inter[1][te_index])

        v_index_mol = np.isin(ind_inter[1], val_index_mol)
        v_index_prot = np.isin(ind_inter[0], val_index_prot)
        v_index = v_index_mol * v_index_prot
        val_ind_inter = (ind_inter[0][v_index], ind_inter[1][v_index])

        v_index_mol = np.isin(ind_non_inter[1], val_index_mol)
        v_index_prot = np.isin(ind_non_inter[0], val_index_prot)
        v_index = v_index_mol * v_index_prot
        val_ind_non_inter = (ind_non_inter[0][v_index], ind_non_inter[1][v_index])
        # import pdb; pdb.Pdb().set_trace()

        tr_index_mol = np.logical_not(
            np.isin(ind_inter[1], np.concatenate((val_index_mol, test_index_mol))))
        tr_index_prot = np.logical_not(
            np.isin(ind_inter[0], np.concatenate((val_index_prot, test_index_prot))))
        tr_index = tr_index_mol * tr_index_prot
        tr_ind_inter = (ind_inter[0][tr_index], ind_inter[1][tr_index])

        tr_index_mol = np.logical_not(
            np.isin(ind_non_inter[1], np.concatenate((val_index_mol, test_index_mol))))
        tr_index_prot = np.logical_not(
            np.isin(ind_non_inter[0], np.concatenate((val_index_prot, test_index_prot))))
        tr_index = tr_index_mol * tr_index_prot
        tr_ind_non_inter = (ind_non_inter[0][tr_index], ind_non_inter[1][tr_index])

        # import pdb; pdb.Pdb().set_trace()
    return te_ind_inter, te_ind_non_inter, val_ind_inter, val_ind_non_inter, \
        tr_ind_inter, tr_ind_non_inter


def get_DB(DB):

    dict_ligand = pickle.load(open('data/' + DB + '/' + DB + '_dict_ID2SMILES.data', 'rb'))
    dict_target = pickle.load(open('data/' + DB + '/' + DB + '_dict_ID2FASTA.data', 'rb'))
    intMat = np.load('data/' + DB + '/' + DB + '_intMat.npy')
    dict_ind2prot = pickle.load(open('data/' + DB + '/' + DB + '_dict_ind2prot.data', 'rb'))
    dict_ind2mol = pickle.load(open('data/' + DB + '/' + DB + '_dict_ind2mol.data', 'rb'))
    dict_prot2ind = pickle.load(open('data/' + DB + '/' + DB + '_dict_prot2ind.data', 'rb'))
    dict_mol2ind = pickle.load(open('data/' + DB + '/' + DB + '_dict_mol2ind.data', 'rb'))
    return dict_ligand, dict_target, intMat, dict_ind2prot, dict_ind2mol, dict_prot2ind, \
        dict_mol2ind


# def DrugBank_CV(DB, list_ratio, setting, cluster_cv, n_folds=5, seed=324):
#     dict_ligand, dict_target, intMat, dict_ind2prot, dict_ind2mol, dict_prot2ind, \
#         dict_mol2ind = get_DB(DB)
#     mratio = max(list_ratio)
#     list_ratio = np.sort(list_ratio)[::-1].tolist()
#     print('DB got')
#     np.random.seed(seed)

#     if setting == 1:
#         ind_inter, ind_non_inter = np.where(intMat == 1), np.where(intMat == 0)
#         np.random.seed(seed)
#         mmask = np.random.choice(np.arange(len(ind_non_inter[0])), len(ind_inter[0]) * mratio,
#                                  replace=False)
#         ind_non_inter = (ind_non_inter[0][mmask], ind_non_inter[1][mmask])
#         previous_nb_non_inter = len(ind_inter[0]) * mratio
#         print("list_on_inter made")
#         for ir, ratio in enumerate(list_ratio):
#             print("ratio", ratio)
#             nb_non_inter = previous_nb_non_inter if ratio == mratio else \
#                 round((float(ratio) / float(list_ratio[ir - 1])) * previous_nb_non_inter)
#             ind_non_inter = (np.random.choice(ind_non_inter[0], nb_non_inter),
#                              np.random.choice(ind_non_inter[1], nb_non_inter))
#             previous_nb_non_inter = nb_non_inter
#             list_couple, y = [], []
#             for i in range(len(ind_inter[0])):
#                 list_couple.append((dict_ind2prot[ind_inter[0][i]], dict_ind2mol[ind_inter[1][i]]))
#                 y.append(1)
#             for i in range(len(ind_non_inter[0])):
#                 list_couple.append((dict_ind2prot[ind_non_inter[0][i]],
#                                     dict_ind2mol[ind_non_inter[1][i]]))
#                 y.append(0)
#             list_couple, y = np.array(list_couple), np.array(y)
#             print('list couple get')

#             if cluster_cv:
#                 Kcouple = make_Kcouple(DB, list_couple)
#                 list_assignment = Khierarchical_cluster(Kcouple, n_folds)
#             else:
#                 X = np.zeros((len(list_couple), 1))
#                 list_assignment = np.zeros(X.shape[0])
#                 skf = model_selection.KFold(n_folds, shuffle=True, random_state=92)
#                 skf.get_n_splits(X)
#                 ifold = 0
#                 for train_index, test_index in skf.split(X):
#                     list_assignment[test_index] = ifold
#                     ifold += 1

#             c = collections.Counter(list_assignment)
#             print(c)
#             folds = [np.where(list_assignment == cle)[0] for cle in list(c.keys())]
#             fo = open('data/' + DB + '/' + DB + '_' + str(setting) + '_' + str(ratio) +
#                       '_folds.txt', 'w')
#             for ifold in range(n_folds):
#                 # import pdb; pdb.Pdb().set_trace()
#                 fo.write("ifold " + str(ifold) + '\t' +
#                          str(collections.Counter(y[folds[ifold]])))
#                 fo.write('\n')
#                 print("ifold " + str(ifold), str(collections.Counter(y[folds[ifold]])))
#             fo.close()

#             for ifold in range(n_folds):
#                 pickle.dump(list_couple[folds[ifold]],
#                             open(data_file(DB, ifold, setting, ratio), 'wb'))
#                 pickle.dump(y[folds[ifold]],
#                             open(y_file(DB, ifold, setting, ratio), 'wb'))

#     elif setting == 2:
#         if cluster_cv:
#             Kprot = pickle.load(open('data/' + DB + '/' + DB + '_Kprot.data', 'rb'))
#             list_assignment = Khierarchical_cluster(Kprot, n_folds)
#             row_c = collections.Counter(list_assignment)
#             row_folds = [np.where(list_assignment == cle)[0] for cle in list(row_c.keys())]

#             ifold = 0
#             fo = [open('data/' + DB + '/' + DB + '_' + str(setting) + '_' + str(ratio) +
#                        '_folds.txt', 'w') for ratio in list_ratio]
#             for test_index in row_folds:
#                 ind_inter, ind_non_inter = setting_makefold(setting, test_index, intMat, mratio,
#                                                             dict_ind2prot, dict_ind2mol, seed)
#                 previous_nb_non_inter = len(ind_inter[0]) * mratio
#                 for ir, ratio in enumerate(list_ratio):
#                     nb_non_inter = previous_nb_non_inter if ratio == mratio else \
#                         round((float(ratio) / float(list_ratio[ir - 1])) * previous_nb_non_inter)
#                     ind_non_inter = (np.random.choice(ind_non_inter[0], nb_non_inter),
#                                      np.random.choice(ind_non_inter[1], nb_non_inter))
#                     previous_nb_non_inter = nb_non_inter
#                     list_couple, y = [], []
#                     for i in range(len(ind_inter[0])):
#                         list_couple.append((dict_ind2prot[ind_inter[0][i]],
#                                             dict_ind2mol[ind_inter[1][i]]))
#                         y.append(1)
#                     for i in range(len(ind_non_inter[0])):
#                         list_couple.append((dict_ind2prot[ind_non_inter[0][i]],
#                                             dict_ind2mol[ind_non_inter[1][i]]))
#                         y.append(0)
#                     list_couple, y = np.array(list_couple), np.array(y)

#                     pickle.dump(list_couple, open(data_file(DB, ifold, setting, ratio), 'wb'))
#                     pickle.dump(y, open(y_file(DB, ifold, setting, ratio), 'wb'))
#                     fo[ir].write("ifold " + str(ifold) + '\t' + str(collections.Counter(y)) + '\n')
#                     print("ratio ", ratio, "ifold " + str(ifold), str(collections.Counter(y)))
#                 ifold += 1
#             for ifo in fo:
#                 ifo.close()
#         else:
#             X = np.zeros((intMat.shape[0], 1))
#             skf = model_selection.KFold(n_folds, shuffle=True, random_state=92)
#             skf.get_n_splits(X)
#             ifold = 0
#             fo = [open('data/' + DB + '/' + DB + '_' + str(setting) + '_' + str(ratio) +
#                        '_folds.txt', 'w') for ratio in list_ratio]
#             for train_index, test_index in skf.split(X):
#                 ind_inter, ind_non_inter = setting_makefold(setting, test_index, intMat, mratio,
#                                                             dict_ind2prot, dict_ind2mol, seed)
#                 previous_nb_non_inter = len(ind_inter[0]) * mratio
#                 for ir, ratio in enumerate(list_ratio):
#                     nb_non_inter = previous_nb_non_inter if ratio == mratio else \
#                         round((float(ratio) / float(list_ratio[ir - 1])) * previous_nb_non_inter)
#                     ind_non_inter = (np.random.choice(ind_non_inter[0], nb_non_inter),
#                                      np.random.choice(ind_non_inter[1], nb_non_inter))
#                     previous_nb_non_inter = nb_non_inter
#                     list_couple, y = [], []
#                     for i in range(len(ind_inter[0])):
#                         list_couple.append((dict_ind2prot[ind_inter[0][i]],
#                                             dict_ind2mol[ind_inter[1][i]]))
#                         y.append(1)
#                     for i in range(len(ind_non_inter[0])):
#                         list_couple.append((dict_ind2prot[ind_non_inter[0][i]],
#                                             dict_ind2mol[ind_non_inter[1][i]]))
#                         y.append(0)
#                     list_couple, y = np.array(list_couple), np.array(y)

#                     pickle.dump(list_couple, open(data_file(DB, ifold, setting, ratio), 'wb'))
#                     pickle.dump(y, open(y_file(DB, ifold, setting, ratio), 'wb'))
#                     fo[ir].write("ifold " + str(ifold) + '\t' + str(collections.Counter(y)) + '\n')
#                     print("ratio ", ratio, "ifold " + str(ifold), str(collections.Counter(y)))
#                 ifold += 1
#             for ifo in fo:
#                 ifo.close()

#     elif setting == 3:
#         if cluster_cv:
#             Kmol = pickle.load(open('data/' + DB + '/' + DB + '_Kmol.data', 'rb'))
#             list_assignment = Khierarchical_cluster(Kmol, n_folds)
#             col_c = collections.Counter(list_assignment)
#             col_folds = [np.where(list_assignment == cle)[0] for cle in list(col_c.keys())]

#             ifold = 0
#             fo = [open('data/' + DB + '/' + DB + '_' + str(setting) + '_' + str(ratio) +
#                        '_folds.txt', 'w') for ratio in list_ratio]
#             for test_index in row_folds:
#                 ind_inter, ind_non_inter = setting_makefold(setting, test_index, intMat, mratio,
#                                                             dict_ind2prot, dict_ind2mol, seed)
#                 previous_nb_non_inter = len(ind_inter[0]) * mratio
#                 for ir, ratio in enumerate(list_ratio):
#                     nb_non_inter = previous_nb_non_inter if ratio == mratio else \
#                         round((float(ratio) / float(list_ratio[ir - 1])) * previous_nb_non_inter)
#                     ind_non_inter = (np.random.choice(ind_non_inter[0], nb_non_inter),
#                                      np.random.choice(ind_non_inter[1], nb_non_inter))
#                     previous_nb_non_inter = nb_non_inter
#                     list_couple, y = [], []
#                     for i in range(len(ind_inter[0])):
#                         list_couple.append((dict_ind2prot[ind_inter[0][i]],
#                                             dict_ind2mol[ind_inter[1][i]]))
#                         y.append(1)
#                     for i in range(len(ind_non_inter[0])):
#                         list_couple.append((dict_ind2prot[ind_non_inter[0][i]],
#                                             dict_ind2mol[ind_non_inter[1][i]]))
#                         y.append(0)
#                     list_couple, y = np.array(list_couple), np.array(y)

#                     pickle.dump(list_couple, open(data_file(DB, ifold, setting, ratio), 'wb'))
#                     pickle.dump(y, open(y_file(DB, ifold, setting, ratio), 'wb'))
#                     fo[ir].write("ifold " + str(ifold) + '\t' + str(collections.Counter(y)) + '\n')
#                     print("ratio ", ratio, "ifold " + str(ifold), str(collections.Counter(y)))
#                 ifold += 1
#             for ifo in fo:
#                 ifo.close()

#         else:
#             X = np.zeros((intMat.shape[1], 1))
#             skf = model_selection.KFold(n_folds, shuffle=True, random_state=92)
#             skf.get_n_splits(X)
#             ifold = 0
#             fo = [open('data/' + DB + '/' + DB + '_' + str(setting) + '_' + str(ratio) +
#                        '_folds.txt', 'w') for ratio in list_ratio]
#             for train_index, test_index in skf.split(X):
#                 ind_inter, ind_non_inter = setting_makefold(setting, test_index, intMat, mratio,
#                                                             dict_ind2prot, dict_ind2mol, seed)
#                 previous_nb_non_inter = len(ind_inter[0]) * mratio
#                 for ir, ratio in enumerate(list_ratio):
#                     nb_non_inter = previous_nb_non_inter if ratio == mratio else \
#                         round((float(ratio) / float(list_ratio[ir - 1])) * previous_nb_non_inter)
#                     ind_non_inter = (np.random.choice(ind_non_inter[0], nb_non_inter),
#                                      np.random.choice(ind_non_inter[1], nb_non_inter))
#                     previous_nb_non_inter = nb_non_inter
#                     list_couple, y = [], []
#                     for i in range(len(ind_inter[0])):
#                         list_couple.append((dict_ind2prot[ind_inter[0][i]],
#                                             dict_ind2mol[ind_inter[1][i]]))
#                         y.append(1)
#                     for i in range(len(ind_non_inter[0])):
#                         list_couple.append((dict_ind2prot[ind_non_inter[0][i]],
#                                             dict_ind2mol[ind_non_inter[1][i]]))
#                         y.append(0)
#                     list_couple, y = np.array(list_couple), np.array(y)

#                     pickle.dump(list_couple, open(data_file(DB, ifold, setting, ratio), 'wb'))
#                     pickle.dump(y, open(y_file(DB, ifold, setting, ratio), 'wb'))
#                     fo[ir].write("ifold " + str(ifold) + '\t' + str(collections.Counter(y)) + '\n')
#                     print("ratio ", ratio, "ifold " + str(ifold), str(collections.Counter(y)))
#                 ifold += 1
#             for ifo in fo:
#                 ifo.close()

#     elif setting == 4:
#         n_folds = 9
#         if cluster_cv:
#             Kmol = pickle.load(open('data/' + DB + '/' + DB + '_Kmol.data', 'rb'))
#             list_assignment = Khierarchical_cluster(Kmol, 3)
#             col_c = collections.Counter(list_assignment)
#             col_folds = [np.where(list_assignment == cle)[0] for cle in list(col_c.keys())]

#             Kprot = pickle.load(open('data/' + DB + '/' + DB + '_Kprot.data', 'rb'))
#             list_assignment = Khierarchical_cluster(Kprot, n_folds)
#             row_c = collections.Counter(list_assignment)
#             row_folds = [np.where(list_assignment == cle)[0] for cle in list(row_c.keys())]

#             ifold = 0
#             fo = [open('data/' + DB + '/' + DB + '_' + str(setting) + '_' + str(ratio) +
#                        '_folds.txt', 'w') for ratio in list_ratio]
#             for prot_test_index in row_folds:
#                 for mol_test_index in col_folds:
#                     test_index = (mol_test_index, prot_test_index)
#                     ind_inter, ind_non_inter = setting_makefold(setting, test_index, intMat,
#                                                                 mratio,
#                                                                 dict_ind2prot, dict_ind2mol, seed)
#                     previous_nb_non_inter = len(ind_inter[0]) * mratio
#                     for ir, ratio in enumerate(list_ratio):
#                         nb_non_inter = previous_nb_non_inter if ratio == mratio else \
#                             round((float(ratio) / float(list_ratio[ir - 1])) * previous_nb_non_inter)
#                         ind_non_inter = (np.random.choice(ind_non_inter[0], nb_non_inter),
#                                          np.random.choice(ind_non_inter[1], nb_non_inter))
#                         previous_nb_non_inter = nb_non_inter
#                         list_couple, y = [], []
#                         for i in range(len(ind_inter[0])):
#                             list_couple.append((dict_ind2prot[ind_inter[0][i]],
#                                                 dict_ind2mol[ind_inter[1][i]]))
#                             y.append(1)
#                         for i in range(len(ind_non_inter[0])):
#                             list_couple.append((dict_ind2prot[ind_non_inter[0][i]],
#                                                 dict_ind2mol[ind_non_inter[1][i]]))
#                             y.append(0)
#                         list_couple, y = np.array(list_couple), np.array(y)

#                         pickle.dump(list_couple, open(data_file(DB, ifold, setting, ratio), 'wb'))
#                         pickle.dump(y, open(y_file(DB, ifold, setting, ratio), 'wb'))
#                         fo[ir].write("ifold " + str(ifold) + '\t' + str(collections.Counter(y)) + '\n')
#                         print("ratio ", ratio, "ifold " + str(ifold), str(collections.Counter(y)))
#                     ifold += 1
#             for ifo in fo:
#                 ifo.close()
#         else:
#             X = np.zeros((intMat.shape[1], 1))
#             skf = model_selection.KFold(3, shuffle=True, random_state=92)
#             skf.get_n_splits(X)
#             mol_folds = []
#             for train_index, test_index in skf.split(X):
#                 mol_folds.append(test_index)

#             X = np.zeros((intMat.shape[0], 1))
#             skf = model_selection.KFold(3, shuffle=True, random_state=92)
#             skf.get_n_splits(X)
#             prot_folds = []
#             for train_index, test_index in skf.split(X):
#                 prot_folds.append(test_index)

#             ifold = 0
#             fo = [open('data/' + DB + '/' + DB + '_' + str(setting) + '_' + str(ratio) +
#                        '_folds.txt', 'w') for ratio in list_ratio]
#             for prot_test_index in prot_folds:
#                 for mol_test_index in mol_folds:
#                     test_index = (mol_test_index, prot_test_index)
#                     ind_inter, ind_non_inter = setting_makefold(setting, test_index, intMat,
#                                                                 mratio,
#                                                                 dict_ind2prot, dict_ind2mol, seed)
#                     previous_nb_non_inter = len(ind_inter[0]) * mratio
#                     for ir, ratio in enumerate(list_ratio):
#                         nb_non_inter = previous_nb_non_inter if ratio == mratio else \
#                             round((float(ratio) / float(list_ratio[ir - 1])) * previous_nb_non_inter)
#                         ind_non_inter = (np.random.choice(ind_non_inter[0], nb_non_inter),
#                                          np.random.choice(ind_non_inter[1], nb_non_inter))
#                         previous_nb_non_inter = nb_non_inter
#                         list_couple, y = [], []
#                         for i in range(len(ind_inter[0])):
#                             list_couple.append((dict_ind2prot[ind_inter[0][i]],
#                                                 dict_ind2mol[ind_inter[1][i]]))
#                             y.append(1)
#                         for i in range(len(ind_non_inter[0])):
#                             list_couple.append((dict_ind2prot[ind_non_inter[0][i]],
#                                                 dict_ind2mol[ind_non_inter[1][i]]))
#                             y.append(0)
#                         list_couple, y = np.array(list_couple), np.array(y)

#                         pickle.dump(list_couple, open(data_file(DB, ifold, setting, ratio), 'wb'))
#                         pickle.dump(y, open(y_file(DB, ifold, setting, ratio), 'wb'))
#                         fo[ir].write("ifold " + str(ifold) + '\t' + str(collections.Counter(y)) + '\n')
#                         print("ratio ", ratio, "ifold " + str(ifold), str(collections.Counter(y)))
#                     ifold += 1
#             for ifo in fo:
#                 ifo.close()


def DrugBank_CV(DB, list_ratio, setting, cluster_cv, n_folds=5, seed=324):
    dict_ligand, dict_target, intMat, dict_ind2prot, dict_ind2mol, dict_prot2ind, \
        dict_mol2ind = get_DB(DB)
    mratio = max(list_ratio)
    list_ratio = np.sort(list_ratio)[::-1].tolist()
    print('DB got')
    np.random.seed(seed)

    if setting == 1:
        ind_inter, ind_non_inter = np.where(intMat == 1), np.where(intMat == 0)
        n_folds = 5
        np.random.seed(seed)

        # pos folds
        pos_folds_data, pos_folds_y = [], []
        list_couple, y = [], []
        for i in range(len(ind_inter[0])):
            list_couple.append((dict_ind2prot[ind_inter[0][i]], dict_ind2mol[ind_inter[1][i]]))
            y.append(1)
        y, list_couple = np.array(y), np.array(list_couple)
        X = np.zeros((len(list_couple), 1))
        skf = model_selection.KFold(n_folds, shuffle=True, random_state=92)
        skf.get_n_splits(X)
        ifold = 0
        for train_index, test_index in skf.split(X):
            print(len(train_index), len(test_index))
            pos_folds_data.append(list_couple[test_index].tolist())
            pos_folds_y.append(y[test_index].tolist())
            ifold += 1
        for n in range(n_folds):
            for n2 in range(n_folds):
                if n2 != n:
                    for c in pos_folds_data[n2]:
                        if c in pos_folds_data[n]:
                            print(c)
                            exit(1)

        # neg folds
        neg_folds_data, neg_folds_y = {r: [] for r in list_ratio}, {r: [] for r in list_ratio}
        mmask = np.random.choice(np.arange(len(ind_non_inter[0])), len(ind_inter[0]) * mratio,
                                 replace=False)
        ind_non_inter = (ind_non_inter[0][mmask], ind_non_inter[1][mmask])

        list_couple, y = [], []
        for i in range(len(ind_non_inter[0])):
            list_couple.append((dict_ind2prot[ind_non_inter[0][i]],
                                dict_ind2mol[ind_non_inter[1][i]]))
            y.append(0)
        list_couple, y = np.array(list_couple), np.array(y)
        X = np.zeros((len(list_couple), 1))
        skf = model_selection.KFold(n_folds, shuffle=True, random_state=92)
        skf.get_n_splits(X)
        ifold = 0
        for train_index, test_index in skf.split(X):
            neg_folds_data[mratio].append(np.array(list_couple)[test_index].tolist())
            neg_folds_y[mratio].append(np.array(y)[test_index].tolist())
            ifold += 1

        previous_nb_non_inter = len(ind_inter[0]) * mratio
        for ir, ratio in enumerate(list_ratio):
            print(ratio)
            if ratio != mratio:
                nb_non_inter = \
                    round((float(ratio) / (float(list_ratio[ir - 1]))) *
                          previous_nb_non_inter)
                previous_nb_non_inter = nb_non_inter
                nb_non_inter = round(float(nb_non_inter) / float(n_folds))
                print('nb_non_inter', previous_nb_non_inter)
                print(len(neg_folds_data[list_ratio[ir - 1]][0]))
                for ifold in range(n_folds):
                    mask = np.random.choice(
                        np.arange(len(neg_folds_data[list_ratio[ir - 1]][ifold])),
                        nb_non_inter, replace=False)
                    neg_folds_data[ratio].append(
                        np.array(neg_folds_data[list_ratio[ir - 1]][ifold])[mask].tolist())
                    neg_folds_y[ratio].append(
                        np.array(neg_folds_y[list_ratio[ir - 1]][ifold])[mask].tolist())
                    ifold += 1
            print('nb_non_inter', previous_nb_non_inter)

        # save folds
        for ir, ratio in enumerate(list_ratio):
            print("ratio", ratio)
            fo = open('data/' + DB + '/' + DB + '_' + str(setting) + '_' + str(ratio) +
                      '_folds.txt', 'w')
            folds_data = []
            for ifold in range(n_folds):
                datatemp = pos_folds_data[ifold] + neg_folds_data[ratio][ifold]
                folds_data.append([c[0] + c[1] for c in datatemp])
                ytemp = pos_folds_y[ifold] + neg_folds_y[ratio][ifold]
                # import pdb; pdb.Pdb().set_trace()
                fo.write("ifold " + str(ifold) + '\t' + str(collections.Counter(ytemp)) + '\n')
                pickle.dump(datatemp, open(data_file(DB, ifold, setting, ratio), 'wb'))
                pickle.dump(ytemp, open(y_file(DB, ifold, setting, ratio), 'wb'))
                print("ifold " + str(ifold), str(collections.Counter(ytemp)))
            fo.close()
            for n in range(n_folds):
                for n2 in range(n_folds):
                    if n2 != n:
                        for c in folds_data[n2]:
                            if c in folds_data[n]:
                                print('alerte', c)
                                exit(1)

    elif setting == 2 or setting == 3:
        if setting == 2:
            X = np.zeros((intMat.shape[0], 1))
        elif setting == 3:
            X = np.zeros((intMat.shape[1], 1))
        skf = model_selection.KFold(n_folds, shuffle=True, random_state=92)
        skf.get_n_splits(X)
        ifold = 0
        fo = [open('data/' + DB + '/' + DB + '_' + str(setting) + '_' + str(ratio) +
                   '_folds.txt', 'w') for ratio in list_ratio]
        for train_index, test_index in skf.split(X):
            ind_inter, ind_non_inter = setting_makefold(setting, test_index, intMat, mratio,
                                                        dict_ind2prot, dict_ind2mol, seed)
            previous_nb_non_inter = len(ind_inter[0]) * mratio
            for ir, ratio in enumerate(list_ratio):
                nb_non_inter = previous_nb_non_inter if ratio == mratio else \
                    round((float(ratio) / float(list_ratio[ir - 1])) * previous_nb_non_inter)
                mask = np.random.choice(np.arange(len(ind_non_inter[0])), nb_non_inter,
                                        replace=False)
                ind_non_inter = (ind_non_inter[0][mask], ind_non_inter[1][mask])
                previous_nb_non_inter = nb_non_inter
                print('nb_non_inter', previous_nb_non_inter)
                list_couple, y = [], []
                for i in range(len(ind_inter[0])):
                    list_couple.append((dict_ind2prot[ind_inter[0][i]],
                                        dict_ind2mol[ind_inter[1][i]]))
                    y.append(1)
                for i in range(len(ind_non_inter[0])):
                    list_couple.append((dict_ind2prot[ind_non_inter[0][i]],
                                        dict_ind2mol[ind_non_inter[1][i]]))
                    y.append(0)
                list_couple, y = np.array(list_couple), np.array(y)

                pickle.dump(list_couple, open(data_file(DB, ifold, setting, ratio), 'wb'))
                pickle.dump(y, open(y_file(DB, ifold, setting, ratio), 'wb'))
                fo[ir].write("ifold " + str(ifold) + '\t' + str(collections.Counter(y)) + '\n')
                print("ratio ", ratio, "ifold " + str(ifold), str(collections.Counter(y)))
            ifold += 1
        for ifo in fo:
            ifo.close()

    # elif setting == 3:
    #     X = np.zeros((intMat.shape[1], 1))
    #     skf = model_selection.KFold(n_folds, shuffle=True, random_state=92)
    #     skf.get_n_splits(X)
    #     ifold = 0
    #     fo = [open('data/' + DB + '/' + DB + '_' + str(setting) + '_' + str(ratio) +
    #                '_folds.txt', 'w') for ratio in list_ratio]
    #     for train_index, test_index in skf.split(X):
    #         ind_inter, ind_non_inter = setting_makefold(setting, test_index, intMat, mratio,
    #                                                     dict_ind2prot, dict_ind2mol, seed)
    #         previous_nb_non_inter = len(ind_inter[0]) * mratio
    #         for ir, ratio in enumerate(list_ratio):
    #             nb_non_inter = previous_nb_non_inter if ratio == mratio else \
    #                 round((float(ratio) / float(list_ratio[ir - 1])) * previous_nb_non_inter)
    #             mask = np.random.choice(np.arange(len(ind_non_inter[0])), nb_non_inter,
    #                                     replace=False)
    #             ind_non_inter = (ind_non_inter[0][mask], ind_non_inter[1][mask])
    #             previous_nb_non_inter = nb_non_inter
    #             print('nb_non_inter', previous_nb_non_inter)
    #             list_couple, y = [], []
    #             for i in range(len(ind_inter[0])):
    #                 list_couple.append((dict_ind2prot[ind_inter[0][i]],
    #                                     dict_ind2mol[ind_inter[1][i]]))
    #                 y.append(1)
    #             for i in range(len(ind_non_inter[0])):
    #                 list_couple.append((dict_ind2prot[ind_non_inter[0][i]],
    #                                     dict_ind2mol[ind_non_inter[1][i]]))
    #                 y.append(0)
    #             list_couple, y = np.array(list_couple), np.array(y)

    #             pickle.dump(list_couple, open(data_file(DB, ifold, setting, ratio), 'wb'))
    #             pickle.dump(y, open(y_file(DB, ifold, setting, ratio), 'wb'))
    #             fo[ir].write("ifold " + str(ifold) + '\t' + str(collections.Counter(y)) + '\n')
    #             print("ratio ", ratio, "ifold " + str(ifold), str(collections.Counter(y)))
    #         ifold += 1
    #     for ifo in fo:
    #         ifo.close()

    elif setting == 4:
        n_folds = 25
        X = np.zeros((intMat.shape[1], 1))
        skf = model_selection.KFold(5, shuffle=True, random_state=92)
        skf.get_n_splits(X)
        mol_folds = []
        for train_index, test_index in skf.split(X):
            mol_folds.append(test_index)

        X = np.zeros((intMat.shape[0], 1))
        skf = model_selection.KFold(5, shuffle=True, random_state=92)
        skf.get_n_splits(X)
        prot_folds = []
        for train_index, test_index in skf.split(X):
            prot_folds.append(test_index)

        ifold = 0
        fo = [open('data/' + DB + '/' + DB + '_' + str(setting) + '_' + str(ratio) +
                   '_folds.txt', 'w') for ratio in list_ratio]
        ite = 0
        for tepi, te_prot_test_index in enumerate(prot_folds):
            for temi, te_mol_test_index in enumerate(mol_folds):
                test_index = (te_mol_test_index, te_prot_test_index)

                ival = 0
                for valpi, val_prot_test_index in enumerate(prot_folds):
                    for valmi, val_mol_test_index in enumerate(mol_folds):
                        val_index = (val_mol_test_index, val_prot_test_index)
                        if tepi != valpi and temi != valmi:
                            ifold = (ite, ival)
                            print('ifold', ifold)

                            te_ind_inter, te_ind_non_inter, val_ind_inter, val_ind_non_inter,\
                                tr_ind_inter, tr_ind_non_inter = \
                                setting_makefold(setting, (val_index, test_index), intMat, mratio,
                                                 dict_ind2prot, dict_ind2mol, seed)

                            te_previous_nb_non_inter = len(te_ind_inter[0]) * mratio
                            val_previous_nb_non_inter = len(val_ind_inter[0]) * mratio
                            tr_previous_nb_non_inter = len(tr_ind_inter[0]) * mratio
                            for ir, ratio in enumerate(list_ratio):
                                te_nb_non_inter = te_previous_nb_non_inter if ratio == mratio else \
                                    round((float(ratio) / float(list_ratio[ir - 1])) *
                                          te_previous_nb_non_inter)
                                te_mask = np.random.choice(np.arange(len(te_ind_non_inter[0])),
                                                           te_nb_non_inter, replace=False)
                                te_ind_non_inter = (te_ind_non_inter[0][te_mask],
                                                    te_ind_non_inter[1][te_mask])
                                te_previous_nb_non_inter = te_nb_non_inter
                                print('nb_non_inter', te_previous_nb_non_inter)
                                list_couple, y = [], []
                                for i in range(len(te_ind_inter[0])):
                                    list_couple.append((dict_ind2prot[te_ind_inter[0][i]],
                                                        dict_ind2mol[te_ind_inter[1][i]]))
                                    y.append(1)
                                for i in range(len(te_ind_non_inter[0])):
                                    list_couple.append((dict_ind2prot[te_ind_non_inter[0][i]],
                                                        dict_ind2mol[te_ind_non_inter[1][i]]))
                                    y.append(0)
                                list_couple, y = np.array(list_couple), np.array(y)
                                pickle.dump(list_couple,
                                            open(data_file(DB, ifold, setting, ratio, 'test'), 'wb'))
                                pickle.dump(y, open(y_file(DB, ifold, setting, ratio, 'test'), 'wb'))
                                fo[ir].write("TEST ifold : " + str(ifold) + '\t' +
                                             str(collections.Counter(y)) + '\n')
                                print("TEST ratio ", ratio, "ifold " + str(ifold),
                                      str(collections.Counter(y)))

                                val_nb_non_inter = val_previous_nb_non_inter if ratio == mratio else \
                                    round((float(ratio) / float(list_ratio[ir - 1])) *
                                          val_previous_nb_non_inter)
                                val_mask = np.random.choice(np.arange(len(val_ind_non_inter[0])),
                                                            val_nb_non_inter, replace=False)
                                val_ind_non_inter = (val_ind_non_inter[0][val_mask],
                                                     val_ind_non_inter[1][val_mask])
                                val_previous_nb_non_inter = val_nb_non_inter
                                print('nb_non_inter', val_previous_nb_non_inter)
                                list_couple, y = [], []
                                for i in range(len(val_ind_inter[0])):
                                    list_couple.append((dict_ind2prot[val_ind_inter[0][i]],
                                                        dict_ind2mol[val_ind_inter[1][i]]))
                                    y.append(1)
                                for i in range(len(val_ind_non_inter[0])):
                                    list_couple.append((dict_ind2prot[val_ind_non_inter[0][i]],
                                                        dict_ind2mol[val_ind_non_inter[1][i]]))
                                    y.append(0)
                                list_couple, y = np.array(list_couple), np.array(y)
                                pickle.dump(list_couple,
                                            open(data_file(DB, ifold, setting, ratio, 'val'), 'wb'))
                                pickle.dump(y, open(y_file(DB, ifold, setting, ratio, 'val'), 'wb'))
                                fo[ir].write("VALIDATION ifold : " + str(ifold) + '\t' +
                                             str(collections.Counter(y)) + '\n')
                                print("VALIDATION ratio ", ratio, "ifold " + str(ifold),
                                      str(collections.Counter(y)))

                                tr_nb_non_inter = tr_previous_nb_non_inter if ratio == mratio else \
                                    round((float(ratio) / float(list_ratio[ir - 1])) *
                                          tr_previous_nb_non_inter)
                                tr_mask = np.random.choice(np.arange(len(tr_ind_non_inter[0])),
                                                           tr_nb_non_inter, replace=False)
                                tr_ind_non_inter = (tr_ind_non_inter[0][tr_mask],
                                                    tr_ind_non_inter[1][tr_mask])
                                tr_previous_nb_non_inter = tr_nb_non_inter
                                print('nb_non_inter', tr_previous_nb_non_inter)
                                list_couple, y = [], []
                                for i in range(len(tr_ind_inter[0])):
                                    list_couple.append((dict_ind2prot[tr_ind_inter[0][i]],
                                                        dict_ind2mol[tr_ind_inter[1][i]]))
                                    y.append(1)
                                for i in range(len(tr_ind_non_inter[0])):
                                    list_couple.append((dict_ind2prot[tr_ind_non_inter[0][i]],
                                                        dict_ind2mol[tr_ind_non_inter[1][i]]))
                                    y.append(0)
                                list_couple, y = np.array(list_couple), np.array(y)
                                pickle.dump(list_couple,
                                            open(data_file(DB, ifold, setting, ratio, 'train'), 'wb'))
                                pickle.dump(y, open(y_file(DB, ifold, setting, ratio, 'train'), 'wb'))
                                fo[ir].write("TRAIN ifold : " + str(ifold) + '\t' +
                                             str(collections.Counter(y)) + '\n')
                                print("TRAIN ratio ", ratio, "ifold " + str(ifold),
                                      str(collections.Counter(y)))

                            ival += 1

                ite += 1

        for ifo in fo:
            ifo.close()


if __name__ == "__main__":
    ##### process
    # process_DB(sys.argv[1])

    ##### make fasta
    # DB = sys.argv[1]
    # nb_prot_per_file = 500
    # dict_id2fasta = pickle.load(open('data/' + DB + '/' + DB + '_dict_ID2FASTA.data', 'rb'))
    # fo1 = open('data/' + DB + '/' + DB + '_ID2FASTA_more30.tsv', 'w')
    # fo2 = open('data/' + DB + '/' + DB + '_ID2FASTA_less30.tsv', 'w')
    # for cle, val in dict_id2fasta.items():
    #     if len(val) > 30:
    #         fo1.write(cle + '\t' + val + '\n')
    #     else:
    #         fo2.write(cle + '\t' + val + '\n')
    # fo1.close()
    # fo2.close()

    # fo = open('data/' + DB + '/' + DB + '_allprots.fasta', 'w')
    # for cle, val in dict_id2fasta.items():
    #     fo.write('> ' + cle + '\n' + val + '\n')
    # fo.close()

    # for cle, val in dict_id2fasta.items():
    #     fo = open('data/' + DB + '/allfastas/' + cle + '.fasta', 'w')
    #     fo.write('> ' + cle + '\n' + val + '\n')
    #     fo.close()

    # nb_files = int(np.ceil(float(len(dict_id2fasta.values())) / nb_prot_per_file))
    # fo = [open('data/' + DB + '/' + DB + '_allprots_' + str(nf) + '.fasta', 'w')
    #       for nf in range(nb_files)]
    # ifo = 0
    # for cle, val in dict_id2fasta.items():
    #     i = ifo // nb_prot_per_file
    #     fo[i].write('>' + cle + '\n' + val + '\n')
    #     ifo += 1
    # for nf in range(nb_files):
    #     fo[nf].close()

    ##### make CV
    # for ratio in [1, 2, 5]:
    #     print('ratio', ratio)
    list_ratio = [10, 5, 2, 1]
    # for setting in [4]:
    #     print('setting', setting)
    #     # if not ('DrugBankH' in sys.argv[1] and ratio == 5 and setting == 1):
    #     DrugBank_CV(sys.argv[1], list_ratio, setting, cluster_cv=False)

    # # ##### check CV
    # DB = sys.argv[1]
    # # for setting in [1,2,3]:
    # #     print("setting", setting)
    # #     for ratio_te in [5, 2, 1]:
    # #         print("ratio_te", ratio_te)
    # #         nfolds = 25 if setting == 4 else 5
    # #         for ite in range(nfolds):
    # #             print("ite", ite)
    # #             list_couple_te = pickle.load(open(data_file(DB, ite, setting, ratio_te), 'rb'))
    # #             list_prot_te = [c[0] for c in list_couple_te]
    # #             list_mol_te = [c[1] for c in list_couple_te]
    # #             list_couple_te = [c[0] + c[1] for c in list_couple_te]
    # #             for ratio_tr in [5, 2, 1]:
    # #                 print("ratio_tr", ratio_tr)
    # #                 for itr in range(nfolds):
    # #                     if itr != ite:
    # #                         print('itr', itr)
    # #                         list_couple_tr = \
    # #                             pickle.load(open(data_file(DB, itr, setting, ratio_tr), 'rb'))
    # #                         # list_couple_tr = [c[0] + c[1] for c in list_couple_tr]
    # #                         list_y = pickle.load(open(y_file(DB, itr, setting, ratio_tr), 'rb'))
    # #                         print(len(list_couple_tr))
    # #                         for ic, couple in enumerate(list_couple_tr):
    # #                             # import pdb; pdb.Pdb().set_trace()
    # #                             if couple[0] + couple[1] in list_couple_te:
    # #                                 print('ALERTE ', couple, list_y[ic])
    # #                                 exit(1)
    # #                             if setting in [2, 4] and couple[0] in list_prot_te:
    # #                                 print('alerte mol', couple[1])
    # #                                 exit(1)
    # #                             elif setting in [3, 4] and couple[1] in list_mol_te:
    # #                                 print('alerte mol', couple[0])
    # #                                 exit(1)

    # for setting in [4]:
    #     print("setting", setting)
    #     for ratio_te in [5, 2, 1]:
    #         print("ratio_te", ratio_te)
    #         nfolds = 25
    #         for ite in range(nfolds):
    #             print("ite", ite)
    #             for ival in range(16):
    #                 print("ival", ival)
    #                 ifold = (ite, ival)

    #                 list_couple_te = pickle.load(
    #                     open(data_file(DB, ifold, setting, ratio_te, 'test'), 'rb'))
    #                 list_prot_te = [c[0] for c in list_couple_te]
    #                 list_mol_te = [c[1] for c in list_couple_te]
    #                 list_couple_te = [c[0] + c[1] for c in list_couple_te]

    #                 list_couple_val = pickle.load(
    #                     open(data_file(DB, ifold, setting, ratio_te, 'val'), 'rb'))
    #                 list_prot_val = [c[0] for c in list_couple_val]
    #                 list_mol_val = [c[1] for c in list_couple_val]
    #                 list_couple_val = [c[0] + c[1] for c in list_couple_val]
    #                 for ratio_tr in [5, 2, 1]:
    #                     print("ratio_tr", ratio_tr)
    #                     list_couple_tr = pickle.load(
    #                         open(data_file(DB, ifold, setting, ratio_tr, 'train'), 'rb'))
    #                     # list_couple_tr = [c[0] + c[1] for c in list_couple_tr]
    #                     list_y = pickle.load(
    #                         open(y_file(DB, ifold, setting, ratio_tr, 'train'), 'rb'))
    #                     print(len(list_couple_tr), list_couple_tr)
    #                     for ic, couple in enumerate(list_couple_tr):
    #                         import pdb; pdb.Pdb().set_trace()
    #                         if couple[0] + couple[1] in list_couple_te or \
    #                                 couple[0] + couple[1] in list_couple_val:
    #                             print('ALERTE ', couple, list_y[ic])
    #                             exit(1)
    #                         if couple[0] in list_prot_te or couple[0] in list_prot_val:
    #                             print('alerte mol', couple[0])
    #                             exit(1)
    #                         elif couple[1] in list_mol_te or couple[1] in list_mol_val:
    #                             print('alerte mol', couple[1])
    #                             exit(1)

    DB = 'DrugBankHstand'
    DBH_list_SMILES = pickle.load(open('data/' + DB + '/' + DB + '_list_SMILES.data', 'rb'))
    DBH_list_FASTA = pickle.load(open('data/' + DB + '/' + DB + '_list_FASTA.data', 'rb'))
    DBH_list_ID_prot = pickle.load(open('data/' + DB + '/' + DB + '_list_ID_prot.data', 'rb'))
    DBH_list_ID_mol = pickle.load(open('data/' + DB + '/' + DB + '_list_ID_mol.data', 'rb'))
    DBH_dict_id2smile_inter = \
        pickle.load(open('data/' + DB + '/' + DB + '_dict_ID2SMILES.data', 'rb'))
    DBH_dict_uniprot2fasta_inter = \
        pickle.load(open('data/' + DB + '/' + DB + '_dict_ID2FASTA.data', 'rb'))
    DBH_list_interactions = \
        pickle.load(open('data/' + DB + '/' + DB + '_list_interactions.data', 'rb'))
    DBH_dict_ligand, DBH_dict_target, DBH_intMat, DBH_dict_ind2prot, DBH_dict_ind2mol, \
        DBH_dict_prot2ind, DBH_dict_mol2ind = get_DB(DB)
    DBH_dict_ID2molfeatures = pickle.load(open('data/' + DB + '/' + DB +
                                               '_dict_ID2molfeatures.data', 'rb'))
    DBH_dict_ID2protfeatures = pickle.load(open('data/' + DB + '/' + DB +
                                                '_dict_ID2protfeatures.data', 'rb'))

    DB = 'DrugBankECstand'
    DBEC_list_SMILES = pickle.load(open('data/' + DB + '/' + DB + '_list_SMILES.data', 'rb'))
    DBEC_list_FASTA = pickle.load(open('data/' + DB + '/' + DB + '_list_FASTA.data', 'rb'))
    DBEC_list_ID_prot = pickle.load(open('data/' + DB + '/' + DB + '_list_ID_prot.data', 'rb'))
    DBEC_list_ID_mol = pickle.load(open('data/' + DB + '/' + DB + '_list_ID_mol.data', 'rb'))
    DBEC_dict_id2smile_inter = \
        pickle.load(open('data/' + DB + '/' + DB + '_dict_ID2SMILES.data', 'rb'))
    DBEC_dict_uniprot2fasta_inter = \
        pickle.load(open('data/' + DB + '/' + DB + '_dict_ID2FASTA.data', 'rb'))
    DBEC_list_interactions = \
        pickle.load(open('data/' + DB + '/' + DB + '_list_interactions.data', 'rb'))
    DBEC_dict_ligand, DBEC_dict_target, DBEC_intMat, DBEC_dict_ind2prot, DBEC_dict_ind2mol, \
        DBEC_dict_prot2ind, DBEC_dict_mol2ind = get_DB(DB)
    DBEC_dict_ID2molfeatures = pickle.load(open('data/' + DB + '/' + DB +
                                                '_dict_ID2molfeatures.data', 'rb'))
    DBEC_dict_ID2protfeatures = pickle.load(open('data/' + DB + '/' + DB +
                                                 '_dict_ID2protfeatures.data', 'rb'))

    DB = "DrugBankHEC-ECstand"
    DBEC_dict_id2smile_inter.update(DBH_dict_id2smile_inter)
    pickle.dump(DBEC_dict_id2smile_inter,
                open('data/' + DB + '/' + DB + '_dict_ID2SMILES.data', 'wb'))
    DBEC_dict_uniprot2fasta_inter.update(DBH_dict_uniprot2fasta_inter)
    pickle.dump(DBEC_dict_uniprot2fasta_inter,
                open('data/' + DB + '/' + DB + '_dict_ID2FASTA.data', 'wb'))
    pickle.dump(DBEC_list_interactions + DBH_list_interactions,
                open('data/' + DB + '/' + DB + '_list_interactions.data', 'wb'))
    np.save('data/' + DB + '/' + DB + '_intMat', block_diag(DBH_intMat, DBEC_intMat))
    DBEC_dict_ind2prot.update(DBH_dict_ind2prot)
    pickle.dump(DBEC_dict_ind2prot,
                open('data/' + DB + '/' + DB + '_dict_ind2prot.data', 'wb'))
    DBEC_dict_prot2ind.update(DBH_dict_prot2ind)
    pickle.dump(DBEC_dict_prot2ind,
                open('data/' + DB + '/' + DB + '_dict_prot2ind.data', 'wb'))
    DBEC_dict_ind2mol.update(DBH_dict_ind2mol)
    pickle.dump(DBEC_dict_prot2ind,
                open('data/' + DB + '/' + DB + '_dict_ind2mol.data', 'wb'))
    DBEC_dict_mol2ind.update(DBH_dict_mol2ind)
    pickle.dump(DBEC_dict_mol2ind,
                open('data/' + DB + '/' + DB + '_dict_mol2ind.data', 'wb'))
    pickle.dump(DBEC_list_SMILES + DBH_list_SMILES,
                open('data/' + DB + '/' + DB + '_list_SMILES.data', 'wb'))
    pickle.dump(DBEC_list_FASTA + DBH_list_FASTA,
                open('data/' + DB + '/' + DB + '_list_FASTA.data', 'wb'))
    pickle.dump(DBEC_list_ID_prot + DBH_list_ID_prot,
                open('data/' + DB + '/' + DB + '_list_ID_prot.data', 'wb'))
    pickle.dump(DBEC_list_ID_mol + DBH_list_ID_mol,
                open('data/' + DB + '/' + DB + '_list_ID_mol.data', 'wb'))
    DBEC_dict_ID2molfeatures.update(DBH_dict_ID2molfeatures)
    pickle.dump(DBEC_dict_ID2molfeatures,
                open('data/' + DB + '/' + DB + '_dict_ID2molfeatures.data', 'wb'))
    DBEC_dict_ID2protfeatures.update(DBH_dict_ID2protfeatures)
    pickle.dump(DBEC_dict_ID2protfeatures,
                open('data/' + DB + '/' + DB + '_dict_ID2protfeatures.data', 'wb'))

    for ratio_tr in [5, 2, 1]:
        # get all DBH
        list_couple_DBH, y_DBH = [], []
        for iDBHte in range(5):
            list_couple_te = pickle.load(open(data_file("DrugBankHstand", iDBHte, 1,
                                                        ratio_tr), 'rb'))
            y_te = pickle.load(open(y_file("DrugBankHstand", iDBHte, 1, ratio_tr), 'rb'))
            list_couple_DBH += list_couple_te
            y_DBH += y_te

        #for setting in [1, 2, 3, 4]:
        for setting in [1,2,3,4]:
          print("setting", setting)
          for ratio_te in [5, 2, 1]:
            print("ratio_te", ratio_te)
            nfolds = 5 if setting != 4 else 25
            if setting != 4:
              for ite in range(nfolds):
                print("ite", ite)
                for ival in [ival for ival in range(nfolds) if ival != ite]:
                    print("ival", ival)
                    ifold = (ite, ival)

                    # get test DBEC
                    list_couple_te = pickle.load(open(data_file("DrugBankECstand", ite, setting,
                                                                ratio_te), 'rb'))
                    y_te = pickle.load(open(y_file("DrugBankECstand", ite, setting, ratio_te),
                                            'rb'))

                    # get val DBEC
                    list_couple_val = pickle.load(open(data_file("DrugBankECstand", ival, setting,
                                                                 ratio_te), 'rb'))
                    y_val = pickle.load(open(y_file("DrugBankECstand", ival, setting, ratio_te),
                                             'rb'))

                    list_couple_tr, y_tr = list_couple_DBH.copy(), y_DBH.copy()
                    for itr in [itr for itr in range(nfolds) if itr not in [ite, ival]]:
                        print("itr", itr)
                        if setting >= 2:
                            list_couple = pickle.load(
                                open(data_file("DrugBankECstand", itr, setting, ratio_tr),
                                     'rb')).tolist()
                            y_ = pickle.load(open(y_file("DrugBankECstand", itr, setting,
                                                         ratio_tr), 'rb')).tolist()
                        else:
                            list_couple = pickle.load(open(data_file("DrugBankECstand", itr,
                                                                     setting, ratio_tr), 'rb'))
                            y_ = pickle.load(open(y_file("DrugBankECstand", itr, setting,
                                                         ratio_tr), 'rb'))
                        list_couple_tr += list_couple
                        y_tr += y_

                    pickle.dump(y_tr, open(y_file(DB, ifold, setting, ratio_tr, 'train'), 'wb'))
                    pickle.dump(list_couple_tr,
                                open(data_file(DB, ifold, setting, ratio_tr, 'train'), 'wb'))
                    pickle.dump(y_te, open(y_file(DB, ifold, setting, ratio_te, 'test'), 'wb'))
                    pickle.dump(list_couple_te,
                                open(data_file(DB, ifold, setting, ratio_te, 'test'), 'wb'))
                    pickle.dump(y_val, open(y_file(DB, ifold, setting, ratio_te, 'val'), 'wb'))
                    pickle.dump(list_couple_val,
                                open(data_file(DB, ifold, setting, ratio_te, 'val'), 'wb'))
            else:
              for ite in range(nfolds):
                for ival in range(16):
                    ifold = (ite, ival)
                    # import pdb; pdb.Pdb().set_trace()

                    # get test DBEC
                    list_couple_te = pickle.load(open(data_file("DrugBankECstand", ifold, setting,
                                                                ratio_te, 'test'), 'rb')).tolist()
                    y_te = pickle.load(open(y_file("DrugBankECstand", ifold, setting, ratio_te,
                                                   'test'), 'rb')).tolist()

                    # get val DBEC
                    list_couple_val = pickle.load(open(data_file("DrugBankECstand", ifold, setting,
                                                                 ratio_te, 'val'), 'rb')).tolist()
                    y_val = pickle.load(open(y_file("DrugBankECstand", ifold, setting, ratio_te,
                                                    'val'), 'rb')).tolist()

                    # get tr DBEC
                    list_couple_tr = pickle.load(open(data_file("DrugBankECstand", ifold, setting,
                                                                ratio_tr, 'train'), 'rb')).tolist()
                    y_tr = pickle.load(open(y_file("DrugBankECstand", ifold, setting, ratio_tr,
                                                   'train'), 'rb')).tolist()
                    list_couple_tr += list_couple_DBH.copy()
                    y_tr += y_DBH.copy()

                    pickle.dump(y_tr, open(y_file(DB, ifold, setting, ratio_tr, 'train'), 'wb'))
                    pickle.dump(list_couple_tr,
                                open(data_file(DB, ifold, setting, ratio_tr, 'train'), 'wb'))
                    pickle.dump(y_te, open(y_file(DB, ifold, setting, ratio_te, 'test'), 'wb'))
                    pickle.dump(list_couple_te,
                                open(data_file(DB, ifold, setting, ratio_te, 'test'), 'wb'))
                    pickle.dump(y_val, open(y_file(DB, ifold, setting, ratio_te, 'val'), 'wb'))
                    pickle.dump(list_couple_val,
                                open(data_file(DB, ifold, setting, ratio_te, 'val'), 'wb'))



