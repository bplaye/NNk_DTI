from __future__ import print_function
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
import rdkit.Chem.rdMolDescriptors as rdMolDescriptors
import rdkit.Chem.EState as EState
import rdkit.Chem.rdPartialCharges as rdPartialCharges
import copy
import numpy as np
import sys
import pickle
from src.utils.DB_utils import TOX21_DATASETS, LIST_DTI_DATASETS
att_dtype = np.float32


NB_MAX_ATOMS = {'Tox21': 140, 'HIV': 222, 'AtomizationEnergy': 23, 'MembranePermeability': 59,
                'PCBA': 332, 'PCBA10': 332, 'PCBA100': 332, 'DrugBankH': 457, 'DrugBankEC': 112,
                'DrugBankHstand': 457, 'DrugBankECstand': 112, "DrugBankHEC-ECstand": 457,
                "DrugBankHEC-Hstand": 457}
for db in TOX21_DATASETS:
    NB_MAX_ATOMS[db] = 140
MAX_SEQ_LENGTH = 1000
NB_ATOM_ATTRIBUTES = 32
NB_BOND_ATTRIBUTES = 8
NB_MOL_FEATURES = 1024


class Graph():
    '''Describes an undirected graph class'''

    def __init__(self):
        self.nodes = []
        self.num_nodes = 0
        self.edges = []
        self.num_edges = 0
        self.N_features = 0
        return

    def clone(self):
        '''clone() method to trick Theano'''
        return copy.deepcopy(self)

    def dump_as_matrices(self, aug):
        # Bad input handling
        if not self.nodes:
            raise(ValueError, 'Error generating tensor for graph with no nodes')
        # if not self.edges:
        #   raise(ValueError, 'Need at least one bond!')

        # get number of nodes
        N_nodes = len(self.nodes)
        # get size of vertex and edge attribute vectors
        F_a, F_b = sizeAttributeVectors(molecular_attributes=self.molecular_attributes)

        # initialize output
        # X_a = np.zeros((N_nodes, F_a, 1), dtype=np.float32)
        # A = np.zeros((N_nodes, N_nodes, 1), dtype=np.float32)
        # Aaug = np.zeros((N_nodes, N_nodes, 1), dtype=np.float32)
        # X_b = np.zeros((N_nodes, N_nodes, F_b, 1), dtype=np.float32)
        X_a = np.zeros((N_nodes, F_a), dtype=np.float32)
        if aug == 2:
            A = np.zeros((N_nodes, N_nodes * N_nodes), dtype=np.float32)
        else:
            A = np.zeros((N_nodes, N_nodes), dtype=np.float32)
        Afull = np.zeros((N_nodes, N_nodes), dtype=np.float32)
        X_b = np.zeros((N_nodes * N_nodes, F_b), dtype=np.float32)

        # X_b = np.zeros((N_nodes, N_nodes, F_b), dtype=np.float32)

        if len(self.edges) != 0:
            edgeAttributes = np.vstack([x.attributes for x in self.edges])
        else:
            edgeAttributes = np.zeros(F_b)
        nodeAttributes = np.vstack([x.attributes for x in self.nodes])

        for i, node in enumerate(self.nodes):
            # X_a[i, :, 0] = nodeAttributes[i]
            # Aaug[i, i, 0] = 1.0  # include self terms
            X_a[i, :] = nodeAttributes[i]
            Afull[i, i] = 1.0  # include self terms

        for e, edge in enumerate(self.edges):
            (i, j) = edge.connects
            # A[i, j, 0] = 1.0
            # A[j, i, 0] = 1.0
            # Aaug[i, j, 0] = 1.0
            # Aaug[j, i, 0] = 1.0
            A[i, j] = 1.0
            A[j, i] = 1.0
            Afull[i, j] = 1.0
            Afull[j, i] = 1.0

            # Keep track of extra special bond types - which are nothing more than
            # bias terms specific to the bond type because they are all one-hot encoded
            # X_b[i, j, :] += edgeAttributes[e]
            # X_b[j, i, :] += edgeAttributes[e]
            X_b[i * N_nodes + j, :] += edgeAttributes[e]
            X_b[j * N_nodes + i, :] += edgeAttributes[e]

        return (X_a, X_b, A, Afull)


class Node():
    '''Describes an attributed node in an undirected graph'''

    def __init__(self, i=None, attributes=np.array([], dtype=att_dtype)):
        self.i = i
        self.attributes = attributes  # 1D array
        self.neighbors = []  # (atom index, bond index)
        return


class Edge():
    '''Describes an attributed edge in an undirected graph'''

    def __init__(self, connects=(), i=None, attributes=np.array([], dtype=att_dtype)):
        self.i = i
        self.attributes = attributes  # 1D array
        self.connects = connects  # (atom index, atom index)
        return


# -------------------------------------------------------------------------------
## @brief      convert a molecule from RDKit class to Graph class
##
## @param      rdmol                 The mol of RDKit class
## @param      molecular_attributes  boolean, add of not extra attribute to atom
##                                   attributes
##
## @return     the same mol, of class Graph
##
def molToGraph(rdmol, molecular_attributes=False):

    # Initialize
    graph = Graph()
    graph.molecular_attributes = molecular_attributes

    # Calculate atom-level molecule descriptors
    attributes = [[] for i in rdmol.GetAtoms()]
    if molecular_attributes:
        labels = []
        [attributes[i].append(x[0])
            for (i, x) in enumerate(rdMolDescriptors._CalcCrippenContribs(rdmol))]
        labels.append('Crippen contribution to logp')

        [attributes[i].append(x[1])
            for (i, x) in enumerate(rdMolDescriptors._CalcCrippenContribs(rdmol))]
        labels.append('Crippen contribution to mr')

        [attributes[i].append(x)
            for (i, x) in enumerate(rdMolDescriptors._CalcTPSAContribs(rdmol))]
        labels.append('TPSA contribution')

        [attributes[i].append(x)
            for (i, x) in enumerate(rdMolDescriptors._CalcLabuteASAContribs(rdmol)[0])]
        labels.append('Labute ASA contribution')

        [attributes[i].append(x)
            for (i, x) in enumerate(EState.EStateIndices(rdmol))]
        labels.append('EState Index')

        rdPartialCharges.ComputeGasteigerCharges(rdmol)
        [attributes[i].append(float(a.GetProp('_GasteigerCharge')))
            for (i, a) in enumerate(rdmol.GetAtoms())]
        labels.append('Gasteiger partial charge')

        # Gasteiger partial charges sometimes gives NaN
        for i in range(len(attributes)):
            if np.isnan(attributes[i][-1]) or np.isinf(attributes[i][-1]):
                attributes[i][-1] = 0.0

        [attributes[i].append(float(a.GetProp('_GasteigerHCharge')))
            for (i, a) in enumerate(rdmol.GetAtoms())]
        labels.append('Gasteiger hydrogen partial charge')

        # Gasteiger partial charges sometimes gives NaN
        for i in range(len(attributes)):
            if np.isnan(attributes[i][-1]) or np.isinf(attributes[i][-1]):
                attributes[i][-1] = 0.0

    # Add bonds
    for bond in rdmol.GetBonds():
        edge = Edge()
        edge.i = bond.GetIdx()
        edge.attributes = bondAttributes(bond)
        edge.connects = (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
        graph.edges.append(edge)
    # Add atoms
    for k, atom in enumerate(rdmol.GetAtoms()):
        node = Node()
        node.i = atom.GetIdx()
        node.attributes = atomAttributes(atom, extra_attributes=attributes[k])
        for neighbor in atom.GetNeighbors():
            node.neighbors.append((
                neighbor.GetIdx(),
                rdmol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx()).GetIdx()
            ))
        graph.nodes.append(node)
    # Add counts, for convenience
    graph.num_edges = len(graph.edges)
    graph.num_nodes = len(graph.nodes)
    return graph


def padGraphTensor(old_tensor, new_dsize, tensor_type):
    '''This function takes an input tensor of dsize x dsize x Nfeatures and pads
    up the first two dimensions to new_dsize with zeros as needed'''

    # old_shape = old_tensor.shape
    # if tensor_type == 'adjacency':
    #     new_tensor = np.zeros((new_dsize, new_dsize, old_shape[2]), dtype=np.float32)
    #     for i in range(old_shape[0]):
    #         for j in range(old_shape[1]):
    #             for k in range(old_shape[2]):
    #                 new_tensor[i, j, k] = old_tensor[i, j, k]
    # elif tensor_type == 'bonds':
    #     new_tensor = np.zeros((new_dsize, new_dsize, old_shape[2], old_shape[3]), dtype=np.float32)
    #     for i in range(old_shape[0]):
    #         for j in range(old_shape[1]):
    #             for k in range(old_shape[2]):
    #                 for l in range(old_shape[3]):
    #                     new_tensor[i, j, k, l] = old_tensor[i, j, k, l]
    # elif tensor_type == 'atoms':
    #     print(new_dsize, old_shape[1], old_shape[2])
    #     new_tensor = np.zeros((new_dsize, old_shape[1], old_shape[2]), dtype=np.float32)
    #     for i in range(old_shape[0]):
    #         for j in range(old_shape[1]):
    #             for k in range(old_shape[2]):
    #                 new_tensor[i, j, k] = old_tensor[i, j, k]

    old_shape = old_tensor.shape
    if tensor_type == 'adjacency':
        new_tensor = np.zeros((new_dsize, new_dsize), dtype=np.float32)
        for i in range(old_shape[0]):
            for j in range(old_shape[1]):
                    new_tensor[i, j] = old_tensor[i, j]
    elif tensor_type == 'adjacency_aug':
        new_tensor = np.zeros((new_dsize, new_dsize * new_dsize), dtype=np.float32)
        old_dsize = int(np.sqrt(old_shape[0]))
        for i in range(old_dsize):
            for j in range(old_dsize):
                    new_tensor[i, i * new_dsize + j] = old_tensor[i, i * old_dsize + j]
    # elif tensor_type == 'bonds':
    #     new_tensor = np.zeros((new_dsize, new_dsize, old_shape[2]), dtype=np.float32)
    #     for i in range(old_shape[0]):
    #         for j in range(old_shape[1]):
    #             for k in range(old_shape[2]):
    #                     new_tensor[i, j, k] = old_tensor[i, j, k]
    elif tensor_type == 'bonds':
        new_tensor = np.zeros((new_dsize * new_dsize, old_shape[1]), dtype=np.float32)
        old_dsize = int(np.sqrt(old_shape[0]))
        for i in range(old_dsize):
            for j in range(old_dsize):
                for k in range(old_shape[1]):
                    new_tensor[i * new_dsize + j, k] = old_tensor[i * old_dsize + j, k]
                    new_tensor[j * new_dsize + i, k] = old_tensor[j * old_dsize + i, k]
    elif tensor_type == 'atoms':
        new_tensor = np.zeros((new_dsize, old_shape[1]), dtype=np.float32)
        for i in range(old_shape[0]):
            for j in range(old_shape[1]):
                    new_tensor[i, j] = old_tensor[i, j]

    return new_tensor


def bondAttributes(bond):
    '''Returns a numpy array of attributes for an RDKit bond
    From Neural FP defaults:
    The bond features were a concatenation of whether the bond type was single, double, triple,
    or aromatic, whether the bond was conjugated, and whether the bond was part of a ring.
    '''
    # Initialize
    attributes = []
    # Add bond type
    attributes += oneHotVector(bond.GetBondTypeAsDouble(), [1.0, 1.5, 2.0, 3.0])
    # Add if is aromatic
    attributes.append(bond.GetIsAromatic())
    # Add if bond is conjugated
    attributes.append(bond.GetIsConjugated())
    # Add if bond is part of ring
    attributes.append(bond.IsInRing())

    # NEED THIS FOR TENSOR REPRESENTATION - 1 IF THERE IS A BOND
    attributes.append(1)

    return np.array(attributes, dtype=att_dtype)


def atomAttributes(atom, extra_attributes=[]):
    '''Returns a numpy array of attributes for an RDKit atom
    From ECFP defaults:
    <IdentifierConfiguration>
          <Property Name="AtomicNumber" Value="1"/>
          <Property Name="HeavyNeighborCount" Value="1"/>
          <Property Name="HCount" Value="1"/>
          <Property Name="FormalCharge" Value="1"/>
          <Property Name="IsRingAtom" Value="1"/>
      </IdentifierConfiguration>
      '''
    # Initialize
    attributes = []
    # Add atomic number
    attributes += oneHotVector(atom.GetAtomicNum(), [5, 6, 7, 8, 9, 15, 16, 17, 35, 53, 999])
    # Add heavy neighbor count
    attributes += oneHotVector(len(atom.GetNeighbors()), [0, 1, 2, 3, 4, 5])
    # Add hydrogen count
    attributes += oneHotVector(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])
    # Add formal charge
    attributes.append(atom.GetFormalCharge())
    # Add boolean if in ring
    attributes.append(atom.IsInRing())
    # Add boolean if aromatic atom
    attributes.append(atom.GetIsAromatic())

    attributes += extra_attributes

    return np.array(attributes, dtype=att_dtype)


def oneHotVector(val, lst):
    '''Converts a value to a one-hot vector based on options in lst'''
    if val not in lst:
        val = lst[-1]  # all atoms other than B C N O F P S Cl Br I are consider the same
    return list(map(lambda x: x == val, lst))


def sizeAttributeVector(molecular_attributes=False):
    m = AllChem.MolFromSmiles('CC')
    g = molToGraph(m, molecular_attributes=molecular_attributes)
    a = g.nodes[0]
    b = g.edges[0]
    return len(a.attributes) + len(b.attributes)


def sizeAttributeVectors(molecular_attributes=False):
    m = AllChem.MolFromSmiles('CC')
    g = molToGraph(m, molecular_attributes=molecular_attributes)
    a = g.nodes[0]
    b = g.edges[0]
    return len(a.attributes), len(b.attributes)


def featurise_mol(smile, aug, padd, n_atoms):
    m = Chem.MolFromSmiles(smile, sanitize=False)
    Chem.SanitizeMol(m)
    # print('MOL', smile, m)

    (X_a, X_b, A, Afull) = \
        molToGraph(m, molecular_attributes=True).dump_as_matrices(aug)
    if aug == 1:
        A = Afull

    if padd is True:
        X_a = padGraphTensor(X_a, n_atoms, 'atoms')
        X_b = padGraphTensor(X_b, n_atoms, 'bonds')
        if aug == 2:
            A = padGraphTensor(A, n_atoms, 'adjacency_aug')
        else:
            A = padGraphTensor(A, n_atoms, 'adjacency')
        # Aaug = padGraphTensor(Aaug, NB_MAX_ATOMS, 'adjacency')

    n_atoms = X_a.shape[0]
    return X_a, X_b, A, n_atoms


class MolLoadError(Exception):
    # value is either a SMILES or a file name
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


def LoadFromSmiles(mol_smiles):
    # ex: Chem.MolFromSmiles('Cc1ccccc1')
    m = Chem.MolFromSmiles(mol_smiles, sanitize=False)
    Chem.SanitizeMol(m)
    if m is None:
        raise MolLoadError(mol_smiles)
    return m


def get_mol_fea(smile):
    m = LoadFromSmiles(smile)
    arr = np.zeros((1,))
    DataStructs.ConvertToNumpyArray(
        AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=1024), arr)
    return arr


def mol_build_X(list_SMILES):
    X_fingerprint = np.zeros((len(list_SMILES), 1024), dtype=np.int32)
    for i in range(len(list_SMILES)):
        arr = get_mol_fea(list_SMILES[i])
        X_fingerprint[i, :] = arr
    return X_fingerprint


def mol_build_K(list_SMILES):
    list_fingerprint = []
    for i in range(len(list_SMILES)):
        m = LoadFromSmiles(list_SMILES[i])
        list_fingerprint.append(AllChem.GetMorganFingerprint(m, 2))
    X = np.zeros((len(list_fingerprint), len(list_fingerprint)))
    for i in range(len(list_fingerprint)):
        if i % 5000 == 0:
            print(i)
        for j in range(i, len(list_fingerprint)):
            X[i, j] = DataStructs.TanimotoSimilarity(list_fingerprint[i], list_fingerprint[j])
            X[j, i] = X[i, j]
    return X


if __name__ == "__main__":
    DB = sys.argv[1]
    list_SMILES = pickle.load(open('data/' + DB + '/' + DB + '_list_SMILES.data', 'rb'))
    if DB not in LIST_DTI_DATASETS:
        list_ID = pickle.load(open('data/' + DB + '/' + DB + '_list_ID.data', 'rb'))
    else:
        list_ID = pickle.load(open('data/' + DB + '/' + DB + '_list_ID_mol.data', 'rb'))
    if sys.argv[2] == 'kernel':
        K = mol_build_K(list_SMILES)
        pickle.dump(K, open('data/' + DB + '/' + DB + '_Kmol.data', 'wb'))
    elif sys.argv[2] == 'max_atoms':
        natoms = 0
        for s in list_SMILES:
            m = Chem.MolFromSmiles(s, sanitize=False)
            Chem.SanitizeMol(m)
            n = m.GetNumAtoms()
            if n > natoms:
                natoms = n
        print(natoms)
    elif sys.argv[2] == 'feature':
        X = mol_build_X(list_SMILES)
        if DB not in LIST_DTI_DATASETS:
            pickle.dump({list_ID[i]: X[i, :] for i in range(len(list_ID))},
                        open('data/' + DB + '/' + DB + '_dict_ID2features.data', 'wb'))
        else:
            pickle.dump({list_ID[i]: X[i, :] for i in range(len(list_ID))},
                        open('data/' + DB + '/' + DB + '_dict_ID2molfeatures.data', 'wb'))
        pickle.dump(X, open('data/' + DB + '/' + DB + '_Xmol.data', 'wb'))
