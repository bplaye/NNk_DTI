import numpy as np
# import keras
from keras.utils import to_categorical
import pickle
from keras.utils import Sequence
import collections
from sklearn import preprocessing
from src.utils.DB_utils import data_file, y_file, LIST_MOL_DATASETS, LIST_PROT_DATASETS
from src.utils.DB_utils import LIST_AA_DATASETS, LIST_DTI_DATASETS, LIST_MULTIOUT_DATASETS
from src.utils.DB_utils import MISSING_DATA_MASK_VALUE, LIST_MULTICLF_DATASETS, LIST_CLF_DATASETS
from src.utils.DB_utils import LIST_MTDTI_DATASETS, MT_data_file, MT_y_file
from src.utils.DB_utils import MTextra_data_file, MTextra_y_file
from src.utils.mol_utils import featurise_mol, padGraphTensor, NB_MAX_ATOMS
from src.utils.prot_utils import featurise_prot, padSeq
from src.utils.package_utils import check_multiclass_y, my_to_categorical
from src.utils.package_utils import normalise_class_proportion


# Here, `x_set` is list of SMILES string and `y_set` are the associated classes.
# The method __getitem__ should return a complete batch : [list_of_input, output_array]
# list_of_inuput is [atoms attributes, bonds_attributes, adjacency_matric]
class Generator(Sequence):

    def __init__(self, dataname, x_set, y_set, batch_size, dict_, training, option):

        # import pdb; pdb.Pdb().set_trace()
        self.dataname = dataname
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.indices = np.arange(len(self.x))
        if training is True:
            np.random.shuffle(self.indices)
        self._get_dict_id2item(dict_[0])
        self.training = training

        padding, fea = option['padding'], option['hand_crafted_features']
        self.padd = padding
        self.aug = option['aug'] if 'aug' in list(option.keys()) else 0
        self.fea = fea
        if self.fea:
            self.dict_id2fea = dict_[1]
        if dataname in LIST_MULTIOUT_DATASETS and dataname in LIST_CLF_DATASETS and False:
            y_set = np.array(y_set)
            class_proportion = {0: 0, 1: 0}
            for iout in range(y_set.shape[1]):
                c = collections.Counter(y_set[:, iout])
                class_proportion[0] += c[0]
                class_proportion[1] += c[1]
            if class_proportion[0] != class_proportion[1]:
                class_proportion = normalise_class_proportion(class_proportion)
            self.sw = []
            for row in y_set:
                if np.max(row) == 1:
                    self.sw.append(class_proportion[1])
                elif np.max(row) == 0:
                    self.sw.append(class_proportion[0])

    def _get_dict_id2item(self, dict_id2item):
        pass

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def get_batch_x(self, batch_x):
        print('Should depend on each generator')
        pass

    def get_batch_y(self, batch_y):
        if self.dataname in LIST_AA_DATASETS:
            batch_y = np.array(batch_y[0])
        #     batch_y = keras.utils.to_categorical(batch_y)
        # # elif self.dataname in LIST_MULTICLF_DATASETS:
        # #     batch_y = keras.utils.to_categorical(np.array(batch_y))
        # elif self.dataname not in LIST_MULTIOUT_DATASETS:
        #     # batch_y = np.array(np.expand_dims(batch_y, axis=0))
        #     batch_y = np.array(batch_y)
        return np.array(batch_y)

    def get_batch_sample_weight(self, batch_sw):
        print('Should depend on each generator')
        pass

    def __getitem__(self, idx):
        inds = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = np.array(self.x)[inds].tolist()
        # batch_y = [self.y[i] for i in range(len(self.y)) if i in inds]
        batch_y = np.array(self.y)[inds].tolist()


        batch_x = self.get_batch_x(batch_x)
        batch_y = self.get_batch_y(batch_y)

        if self.dataname in LIST_MULTIOUT_DATASETS and self.dataname in LIST_CLF_DATASETS and \
                False:
            batch_sw = np.array(self.sw)[inds]
            # batch_sw = self.get_batch_sample_weight(batch_sw)
            return batch_x, batch_y, batch_sw
        else:
            return batch_x, batch_y

    def on_epoch_end(self):
        if self.training is True:
            np.random.shuffle(self.indices)


class MolGenerator(Generator):

    def __init__(self, dataname, x_set, y_set, batch_size, dict_, training, option):
        super().__init__(dataname, x_set, y_set, batch_size, dict_, training, option)
        self.bonds_bool = option['bonds']

    def _get_dict_id2item(self, dict_id2item):
        self.dict_id2smiles = dict_id2item
        self.n_atoms = NB_MAX_ATOMS[self.dataname]

    def get_batch_x(self, batch_x):
        atom_att_matrices, bond_att_matrices, ad_matrices, nmax_atoms, feas = [], [], [], 0, []
        for ID in batch_x:
            mol_smiles = self.dict_id2smiles[ID]
            att_matrix, bond_matrix, ad_matrix, n_atoms = \
                featurise_mol(mol_smiles, aug=self.aug, padd=self.padd,
                              n_atoms=self.n_atoms)
            atom_att_matrices.append(att_matrix)
            bond_att_matrices.append(bond_matrix)
            ad_matrices.append(ad_matrix)
            if n_atoms > nmax_atoms:
                nmax_atoms = n_atoms
            if self.fea:
                feas.append(self.dict_id2fea[ID])
        if not self.padd:
            for im in range(len(atom_att_matrices)):
                atom_att_matrices[im] = padGraphTensor(atom_att_matrices[im], nmax_atoms, 'atoms')
                bond_att_matrices[im] = padGraphTensor(bond_att_matrices[im], nmax_atoms, 'bonds')
                if self.aug == 2:
                    ad_matrices[im] = padGraphTensor(ad_matrices[im], nmax_atoms, 'adjacency_aug')
                else:
                    ad_matrices[im] = padGraphTensor(ad_matrices[im], nmax_atoms, 'adjacency')
        if self.fea:
            if self.bonds_bool:
                return [np.array(atom_att_matrices), np.array(bond_att_matrices),
                        np.array(ad_matrices), np.array(feas)]
            else:
                return [np.array(atom_att_matrices),
                        np.array(ad_matrices), np.array(feas)]
        else:
            if self.bonds_bool:
                return [np.array(atom_att_matrices), np.array(bond_att_matrices),
                        np.array(ad_matrices)]
            else:
                return [np.array(atom_att_matrices), np.array(ad_matrices)]


class ProtGenerator(Generator):

    def _get_dict_id2item(self, dict_id2item):
        self.dict_id2fasta = dict_id2item

    def get_batch_x(self, batch_x):
        seqs, len_seq, feas = [], [], []
        for ID in batch_x:
            fasta = self.dict_id2fasta[ID]
            # seq.append(keras.preprocessing.text.one_hot(fasta, len(LIST_AA), filters=LIST_AA,
            #                                             lower=False, split=' '))
            seq, n_aa = featurise_prot(fasta, padding=self.padd)
            seqs.append(seq)
            len_seq.append(n_aa)
            if self.fea:
                feas.append(self.dict_id2fea[ID])
        if not self.padd:
            n_aa_max = int(max(len_seq))
            seqs = [padSeq(seq, n_aa_max) for seq in seqs]
        if self.fea:
            return [np.array(seqs), np.array(len_seq), np.array(feas)]
        else:
            return [np.array(seqs), np.array(len_seq)]


class DTIGenerator(Generator):

    def __init__(self, dataname, x_set, y_set, batch_size, dict_, training, option):
        super().__init__(dataname, x_set, y_set, batch_size, dict_, training, option)
        self.bonds_bool = option['bonds']

    def _get_dict_id2item(self, dict_id2item):
        self.dict_id2smiles = dict_id2item[1]
        self.dict_id2fasta = dict_id2item[0]
        self.n_atoms = NB_MAX_ATOMS[self.dataname]

    def get_batch_x(self, batch_x):
        seqs, len_seq, prot_feas = [], [], []
        atom_att_matrices, bond_att_matrices, ad_matrices, nmax_atoms, mol_feas = [], [], [], 0, []
        for (prot_id, mol_id) in batch_x:
            # smiles
            mol_smiles = self.dict_id2smiles[mol_id]
            att_matrix, bond_matrix, ad_matrix, n_atoms = \
                featurise_mol(mol_smiles, aug=self.aug, padd=self.padd, n_atoms=self.n_atoms)
            atom_att_matrices.append(att_matrix)
            bond_att_matrices.append(bond_matrix)
            ad_matrices.append(ad_matrix)
            if n_atoms > nmax_atoms:
                nmax_atoms = n_atoms
            if self.fea:
                mol_feas.append(self.dict_id2fea[1][mol_id])

            # fasta
            fasta = self.dict_id2fasta[prot_id]
            # seq.append(keras.preprocessing.text.one_hot(fasta, len(LIST_AA), filters=LIST_AA,
            #                                             lower=False, split=' '))
            seq, n_aa = featurise_prot(fasta, padding=self.padd)
            seqs.append(seq)
            len_seq.append(n_aa)
            if self.fea:
                prot_feas.append(self.dict_id2fea[0][prot_id])

        if not self.padd:
            for im in range(len(atom_att_matrices)):
                atom_att_matrices[im] = padGraphTensor(atom_att_matrices[im], nmax_atoms, 'atoms')
                bond_att_matrices[im] = padGraphTensor(bond_att_matrices[im], nmax_atoms, 'bonds')
                if self.aug == 2:
                    ad_matrices[im] = padGraphTensor(ad_matrices[im], nmax_atoms, 'adjacency_aug')
                else:
                    ad_matrices[im] = padGraphTensor(ad_matrices[im], nmax_atoms, 'adjacency')

        if not self.padd:
            n_aa_max = int(max(len_seq))
            seqs = [padSeq(seq, n_aa_max) for seq in seqs]

        if self.fea:
            prot_feas = preprocessing.scale(np.stack(prot_feas, axis=0))
            mol_feas = np.stack(mol_feas, axis=0)
            if self.bonds_bool:
                return [np.array(atom_att_matrices), np.array(bond_att_matrices),
                        np.array(ad_matrices), mol_feas,
                        np.array(seqs), np.array(len_seq), prot_feas]
            else:
                return [np.array(atom_att_matrices),
                        np.array(ad_matrices), mol_feas,
                        np.array(seqs), np.array(len_seq), prot_feas]
        else:
            if self.bonds_bool:
                return [np.array(atom_att_matrices), np.array(bond_att_matrices),
                        np.array(ad_matrices), np.array(seqs), np.array(len_seq)]
            else:
                return [np.array(atom_att_matrices),
                        np.array(ad_matrices), np.array(seqs), np.array(len_seq)]


class MTDTIGenerator(DTIGenerator):

    def get_batch_x(self, batch_x):
        seqs, len_seq, prot_feas = [], [], []
        dti_atom_att_matrices, dti_bond_att_matrices, dti_ad_matrices, nmax_atoms = [], [], [], 0
        pcba_atom_att_matrices, pcba_bond_att_matrices, pcba_ad_matrices = [], [], []
        for (prot_id, dti_mol_id, pcba_mol_id) in batch_x:
            # smiles
            dti_mol_smiles = self.dict_id2smiles[dti_mol_id]
            att_matrix, bond_matrix, ad_matrix, n_atoms = \
                featurise_mol(dti_mol_smiles, aug=self.aug, padd=self.padd)
            dti_atom_att_matrices.append(att_matrix)
            dti_bond_att_matrices.append(bond_matrix)
            dti_ad_matrices.append(ad_matrix)
            if n_atoms > nmax_atoms:
                nmax_atoms = n_atoms

            # smiles
            pcba_mol_smiles = self.dict_id2smiles[dti_mol_id]
            att_matrix, bond_matrix, ad_matrix, n_atoms = \
                featurise_mol(pcba_mol_smiles, aug=self.aug, padd=self.padd)
            pcba_atom_att_matrices.append(att_matrix)
            pcba_bond_att_matrices.append(bond_matrix)
            pcba_ad_matrices.append(ad_matrix)
            if n_atoms > nmax_atoms:
                nmax_atoms = n_atoms

            # fasta
            fasta = self.dict_id2fasta[prot_id]
            # seq.append(keras.preprocessing.text.one_hot(fasta, len(LIST_AA), filters=LIST_AA,
            #                                             lower=False, split=' '))
            seq, n_aa = featurise_prot(fasta, padding=self.padd)
            seqs.append(seq)
            len_seq.append(n_aa)
            if self.fea:
                prot_feas.append(self.dict_id2fea[0][prot_id])

        if not self.padd:
            for atom_att_matrices, bond_att_matrices, ad_matrices in \
                    zip([dti_atom_att_matrices, pcba_atom_att_matrices],
                        [dti_bond_att_matrices, pcba_bond_att_matrices],
                        [dti_ad_matrices, pcba_ad_matrices]):
                for im in range(len(atom_att_matrices)):
                    atom_att_matrices[im] = padGraphTensor(atom_att_matrices[im],
                                                           nmax_atoms, 'atoms')
                    bond_att_matrices[im] = padGraphTensor(bond_att_matrices[im],
                                                           nmax_atoms, 'bonds')
                    if self.aug == 2:
                        ad_matrices[im] = padGraphTensor(ad_matrices[im],
                                                         nmax_atoms, 'adjacency_aug')
                    else:
                        ad_matrices[im] = padGraphTensor(ad_matrices[im],
                                                         nmax_atoms, 'adjacency')

        if not self.padd:
            n_aa_max = int(max(len_seq))
            seqs = [padSeq(seq, n_aa_max) for seq in seqs]

        return [np.array(dti_atom_att_matrices), np.array(dti_bond_att_matrices),
                np.array(dti_ad_matrices), np.array(pcba_atom_att_matrices),
                np.array(pcba_bond_att_matrices), np.array(pcba_ad_matrices),
                np.array(seqs), np.array(len_seq)]


def load_generator(dataname, batch_size, n_folds, fold_val, fold_te, setting, ratio_tr, ratio_te,
                   option):
    if dataname not in LIST_MTDTI_DATASETS:
        if not(setting == 4 or dataname in ["DrugBankHEC-ECstand", "DrugBankHEC-Hstand"]):
            x_tr = [ind
                    for sub in [pickle.load(open(data_file(dataname, i, setting, ratio_tr), 'rb'))
                                for i in range(n_folds) if i != fold_val and i != fold_te]
                    for ind in sub]
            y_tr = [ind for sub in [pickle.load(open(y_file(dataname, i, setting, ratio_tr), 'rb'))
                                    for i in range(n_folds) if i != fold_val and i != fold_te]
                    for ind in sub]
            x_val = pickle.load(open(data_file(dataname, fold_val, setting, ratio_te), 'rb'))
            y_val = pickle.load(open(y_file(dataname, fold_val, setting, ratio_te), 'rb'))
            if fold_te is not None:
                x_te = pickle.load(open(data_file(dataname, fold_te, setting, ratio_te), 'rb'))
                y_te = pickle.load(open(y_file(dataname, fold_te, setting, ratio_te), 'rb'))
            else:
                x_te = np.zeros((1, 1))
                y_te = np.zeros((1, 1))
        else:
            ifold = (fold_te, fold_val)
            x_val = pickle.load(open(data_file(dataname, ifold, setting, ratio_te, 'val'), 'rb'))
            y_val = pickle.load(open(y_file(dataname, ifold, setting, ratio_te, 'val'), 'rb'))
            x_te = pickle.load(
                open(data_file(dataname, ifold, setting, ratio_te, 'test'), 'rb'))
            y_te = pickle.load(open(y_file(dataname, ifold, setting, ratio_te, 'test'), 'rb'))
            x_tr = pickle.load(open(data_file(dataname, ifold, setting, ratio_tr, 'train'), 'rb'))
            y_tr = pickle.load(open(y_file(dataname, ifold, setting, ratio_tr, 'train'), 'rb'))
    else:
        x_tr = pickle.load(open(
            MT_data_file(dataname, fold_val, fold_te, setting, ratio_tr), 'rb'))
        y_tr = pickle.load(open(MT_y_file(dataname, fold_val, fold_te, setting, ratio_tr), 'rb'))
        x_dti_val = pickle.load(open(data_file(dataname, fold_val, setting, ratio_te), 'rb'))
        y_dti_val = pickle.load(open(y_file(dataname, fold_val, setting, ratio_te), 'rb'))
        x_pcba_val = pickle.load(open(MTextra_data_file(dataname), 'rb'))
        y_pcba_val = pickle.load(open(MTextra_y_file(dataname), 'rb'))
        x_dti_te = pickle.load(open(data_file(dataname, fold_te, setting, ratio_te), 'rb'))
        y_dti_te = pickle.load(open(y_file(dataname, fold_te, setting, ratio_te), 'rb'))

    if dataname in LIST_MULTIOUT_DATASETS:
        for i in range(len(y_tr)):
            y_tr[i][y_tr[i] == None] = MISSING_DATA_MASK_VALUE
        for i in range(len(y_te)):
            y_te[i][y_te[i] == None] = MISSING_DATA_MASK_VALUE
        for i in range(len(y_val)):
            y_val[i][y_val[i] == None] = MISSING_DATA_MASK_VALUE
        y_data = (np.array(np.stack(y_tr, axis=0), dtype=np.float32),
                  np.array(np.stack(y_val, axis=0), dtype=np.float32),
                  np.array(np.stack(y_te, axis=0), dtype=np.float32))
    else:
        # if dataname in LIST_MULTICLF_DATASETS:
        #     y_tr = np.array(y_tr, dtype=np.int32)
        #     y_val = np.array(y_val, dtype=np.int32)
        #     y_te = np.array(y_te, dtype=np.int32)
            # n_classes = option['n_classes']
            # a, b, c = np.zeros((len(y_tr), n_classes)), np.zeros((len(y_val), n_classes)), \
            #     np.zeros((len(y_te), n_classes))
            # a[np.arange(len(y_tr)), y_tr] = 1
            # b[np.arange(len(y_val)), y_val] = 1
            # c[np.arange(len(y_te)), y_te] = 1
            # y_tr, y_val, y_te = a, b, c
        y_data = (np.array(y_tr), np.array(y_val), np.array(y_te))

    if dataname in LIST_MULTICLF_DATASETS:
        if dataname in LIST_AA_DATASETS:
            y_tr = [my_to_categorical(y_tr[i], dataname) for i in range(len(y_tr))]
            y_val = [my_to_categorical(y_val[i], dataname) for i in range(len(y_val))]
            y_te = [my_to_categorical(y_te[i], dataname) for i in range(len(y_te))]
        else:
            y_tr, y_val, y_te = \
                to_categorical(y_tr), to_categorical(y_val), to_categorical(y_te)

    fea = option['hand_crafted_features']
    if dataname in LIST_MOL_DATASETS:
        dict_id2smiles = pickle.load(open('data/' + dataname + '/' + dataname +
                                          '_dict_ID2SMILES.data', 'rb'))
        if fea:
            dict_id2fea = pickle.load(open('data/' + dataname + '/' + dataname +
                                           '_dict_ID2features.data', 'rb'))
            dict_ = (dict_id2smiles, dict_id2fea)
        else:
            dict_ = (dict_id2smiles, None)
        return (MolGenerator(dataname, x_tr, y_tr, batch_size, dict_, True, option),
                MolGenerator(dataname, x_val, y_val, batch_size, dict_, False, option),
                MolGenerator(dataname, x_te, y_te, batch_size, dict_, False, option),
                y_data)
    elif dataname in LIST_PROT_DATASETS or dataname in LIST_AA_DATASETS:
        dict_id2fasta = pickle.load(open('data/' + dataname + '/' + dataname +
                                         '_dict_ID2FASTA.data', 'rb'))
        if fea:
            dict_id2fea = pickle.load(open('data/' + dataname + '/' + dataname +
                                           '_dict_ID2features.data', 'rb'))
            dict_ = (dict_id2fasta, dict_id2fea)
        else:
            dict_ = (dict_id2fasta, None)
        return (ProtGenerator(dataname, x_tr, y_tr, batch_size, dict_, True, option),
                ProtGenerator(dataname, x_val, y_val, batch_size, dict_, False, option),
                ProtGenerator(dataname, x_te, y_te, batch_size, dict_, False, option),
                y_data)
    elif dataname in LIST_DTI_DATASETS:
        dict_id2smiles = pickle.load(open('data/' + dataname + '/' + dataname +
                                          '_dict_ID2SMILES.data', 'rb'))
        dict_id2fasta = pickle.load(open('data/' + dataname + '/' + dataname +
                                         '_dict_ID2FASTA.data', 'rb'))
        if fea:
            dict_id2molfea = pickle.load(open('data/' + dataname + '/' + dataname +
                                              '_dict_ID2molfeatures.data', 'rb'))
            dict_id2protfea = pickle.load(open('data/' + dataname + '/' + dataname +
                                               '_dict_ID2protfeatures.data', 'rb'))
            dict_ = ((dict_id2fasta, dict_id2smiles), (dict_id2protfea, dict_id2molfea))
        else:
            dict_ = ((dict_id2fasta, dict_id2smiles), None)
        return (DTIGenerator(dataname, x_tr, y_tr, batch_size, dict_, True, option),
                DTIGenerator(dataname, x_val, y_val, batch_size, dict_, False, option),
                DTIGenerator(dataname, x_te, y_te, batch_size, dict_, False, option),
                y_data)
    elif dataname in LIST_MTDTI_DATASETS:
        dict_id2smiles = pickle.load(open('data/' + dataname + '/' + dataname +
                                          '_dict_ID2SMILES.data', 'rb'))
        dict_id2fasta = pickle.load(open('data/' + dataname + '/' + dataname +
                                         '_dict_ID2FASTA.data', 'rb'))
        dict_ = ((dict_id2fasta, dict_id2smiles), None)
        return (MTDTIGenerator(dataname, x_tr, y_tr, batch_size, dict_, True, option),
                (MTDTIGenerator(dataname, x_dti_val, y_dti_val, batch_size, dict_, False, option),
                 MTDTIGenerator(dataname, x_pcba_val, y_pcba_val, batch_size, dict_,
                                False, option)),
                MTDTIGenerator(dataname, x_dti_te, y_dti_te, batch_size, dict_, False, option),
                y_data)
