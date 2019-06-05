import pickle
import math
import random
import sklearn.cluster
from sklearn.preprocessing import KernelCenterer
import numpy as np
from src.utils.DB_utils import LIST_MOL_DATASETS, LIST_AA_DATASETS, LIST_PROT_DATASETS
from src.utils.DB_utils import LIST_DTI_DATASETS, data_file, y_file
from src.utils.mol_utils import mol_build_X


def get_DB(DB):
    if DB in LIST_DTI_DATASETS:
        dict_ligand = pickle.load(open('data/' + DB + '/' + DB + '_dict_DBid2smiles.data', 'rb'))
        dict_target = pickle.load(open('data/' + DB + '/' + DB + '_dict_uniprot2fasta.data', 'rb'))
        intMat = np.load('data/' + DB + '/' + DB + '_intMat.npy')
        dict_ind2prot = pickle.load(open('data/' + DB + '/' + DB + '_dict_ind2prot.data', 'rb'))
        dict_ind2mol = pickle.load(open('data/' + DB + '/' + DB + '_dict_ind2mol.data', 'rb'))
        dict_prot2ind = pickle.load(open('data/' + DB + '/' + DB + '_dict_prot2ind.data', 'rb'))
        dict_mol2ind = pickle.load(open('data/' + DB + '/' + DB + '_dict_mol2ind.data', 'rb'))

        return dict_ligand, dict_target, intMat, dict_ind2prot, dict_ind2mol, dict_prot2ind, \
            dict_mol2ind


def assign_points_to_clusters(medoids, distances):
    distances_to_medoids = distances[:,medoids]
    clusters = medoids[np.argmin(distances_to_medoids, axis=1)]
    clusters[medoids] = medoids
    return clusters


def compute_new_medoid(cluster, distances):
    mask = np.ones(distances.shape)
    mask[np.ix_(cluster,cluster)] = 0.
    cluster_distances = np.ma.masked_array(data=distances, mask=mask, fill_value=10e9)
    costs = cluster_distances.sum(axis=1)
    return costs.argmin(axis=0, fill_value=10e9)


def Kmedoid_cluster(distances, k):

    m = distances.shape[0] # number of points

    # Pick k random medoids.
    curr_medoids = np.array([-1]*k)
    while not len(np.unique(curr_medoids)) == k:
        curr_medoids = np.array([random.randint(0, m - 1) for _ in range(k)])
    old_medoids = np.array([-1]*k) # Doesn't matter what we initialize these to.
    new_medoids = np.array([-1]*k)

    # Until the medoids stop updating, do the following:
    while not ((old_medoids == curr_medoids).all()):
        # Assign each point to cluster with closest medoid.
        clusters = assign_points_to_clusters(curr_medoids, distances)

        # Update cluster medoids to be lowest cost point.
        for curr_medoid in curr_medoids:
            cluster = np.where(clusters == curr_medoid)[0]
            new_medoids[curr_medoids == curr_medoid] = compute_new_medoid(cluster, distances)

        old_medoids[:] = curr_medoids[:]
        curr_medoids[:] = new_medoids[:]

    return clusters, curr_medoids


def Khierarchical_cluster(distances, n_folds):
    clustering = sklearn.cluster.AgglomerativeClustering(n_folds, affinity='precomputed',
                                                         linkage='complete')
    clusters_labels = clustering.fit_predict(distances)
    return clusters_labels


def Xkmeans_cluster(X, n_folds):
    clustering = sklearn.cluster.KMeans(n_folds)
    clusters_labels = clustering.fit_predict(X)
    return clusters_labels


def mol_make_CV(DB, fold_fct, data_type):
    n_folds, setting, ratio = 5, 1, 1

    dict_id2smile = pickle.load(open('data/' + DB + '/' + DB + '_dict_ID2SMILES.data', 'rb'))
    # pickle.dump(dict_uniprot2fasta,
    #             open(root + 'data/' + DB + '/' + DB + '_dict_ID2FASTA.data', 'wb'))
    list_SMILES = pickle.load(open('data/' + DB + '/' + DB + '_list_SMILES.data', 'rb'))
    list_y = np.array(pickle.load(open('data/' + DB + '/' + DB + '_list_y.data', 'rb')))
    list_ID = np.array(pickle.load(open('data/' + DB + '/' + DB + '_list_ID.data', 'rb')))

    folds = fold_fct(DB, data_type, list_ID, list_y, list_SMILES, dict_id2smile, n_folds)

    for ifold in range(n_folds):
        pickle.dump(list_ID[folds[ifold]],
                    open(data_file(DB, ifold, setting, ratio), 'wb'))
        pickle.dump(list_y[folds[ifold]],
                    open(y_file(DB, ifold, setting, ratio), 'wb'))


def prot_make_CV(DB, fold_fct, data_type):
    n_folds, setting, ratio = 5, 1, 1

    dict_id2fasta = pickle.load(open('data/' + DB + '/' + DB + '_dict_ID2FASTA.data', 'rb'))
    # pickle.dump(dict_uniprot2fasta,
    #             open(root + 'data/' + DB + '/' + DB + '_dict_ID2FASTA.data', 'wb'))
    list_FASTA = pickle.load(open('data/' + DB + '/' + DB + '_list_FASTA.data', 'rb'))
    list_y = np.array(pickle.load(open('data/' + DB + '/' + DB + '_list_y.data', 'rb')))
    list_ID = np.array(pickle.load(open('data/' + DB + '/' + DB + '_list_ID.data', 'rb')))

    folds = fold_fct(DB, data_type, list_ID, list_y, list_FASTA, dict_id2fasta, n_folds)

    for ifold in range(n_folds):
        pickle.dump(list_ID[folds[ifold]],
                    open(data_file(DB, ifold, setting, ratio), 'wb'))
        pickle.dump(list_y[folds[ifold]],
                    open(y_file(DB, ifold, setting, ratio), 'wb'))


def make_folds(nb_fold=5, seed=324):
    dict_ligand, dict_target, intMat, dict_ind2prot, dict_ind2mol, dict_prot2ind, \
        dict_mol2ind = get_DB()
    print('DB got')

    ind_inter, ind_non_inter = np.where(intMat == 1), np.where(intMat == 0)
    np.random.seed(seed)
    mask = np.random.choice(np.arange(len(ind_non_inter[0])), len(ind_inter[0]),
                            replace=False)
    ind_non_inter = (ind_non_inter[0][mask], ind_non_inter[1][mask])
    print("list_on_inter made")

    list_couple, y = [], []
    for i in range(len(ind_inter[0])):
        list_couple.append((dict_ind2prot[ind_inter[0][i]], dict_ind2mol[ind_inter[1][i]]))
        y.append(1)
    for i in range(len(ind_non_inter[0])):
        list_couple.append((dict_ind2prot[ind_non_inter[0][i]], dict_ind2mol[ind_non_inter[1][i]]))
        y.append(0)
    print('list couple get')

    # np.save('data/DB_S/DB_S_y', np.array(y))
    # pickle.dump(list_couple, open('data/DB_S/DB_S_list_couple.data', 'wb'), protocol=2)

    # Kcouple = make_Kcouple(list_couple)
    # np.save('data/DB_S/DB_S_Kcouple', Kcouple)
    # del Kcouple
    # print('Kcouple made')

    # Xcouple = make_Xcouple(list_couple)
    # np.save('data/DB_S/DB_S_Xcouple', Xcouple)
    # del Xcouple
    # print('Xcouple made')

    # Xcouple_kron = make_Xcouple_kron(list_couple)
    # np.save('data/DB_S/DB_S_Xcouple_kron', Xcouple_kron)
    # del Xcouple_kron
    # print('Xcouple_kron made')

    # make_TFdataset('data/DB_S/', 'DB_S', list_couple, y, dict_ligand, dict_target, nb_fold, seed)
    # print('TFdataset made')
