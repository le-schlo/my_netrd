"""
netsimile.py
--------------

Graph distance based on:
Berlingerio, M., Koutra, D., Eliassi-Rad, T. & Faloutsos, C. NetSimile: A Scalable Approach to Size-Independent Network Similarity. arXiv (2012)

author: Alex Gates
email: ajgates42@gmail.com (optional)
Submitted as part of the 2019 NetSI Collabathon.

"""
import networkx as nx
import numpy as np
from scipy.spatial.distance import canberra
from scipy.stats import skew, kurtosis

import base

#from ..utilities import undirected, unweighted

# import packages

# general tools
import numpy as np

# RDkit
from rdkit import Chem
from rdkit.Chem.rdmolops import GetAdjacencyMatrix

# Pytorch and Pytorch Geometric
def one_hot_encoding(x, permitted_list):
    """
    Maps input elements x which are not in the permitted list to the last element
    of the permitted list.
    """

    if x not in permitted_list:
        x = permitted_list[-1]

    binary_encoding = [int(boolean_value) for boolean_value in list(map(lambda s: x == s, permitted_list))]

    return binary_encoding


def get_atom_features(atom,
                      use_chirality=False,
                      hydrogens_implicit=True):
    """
    Takes an RDKit atom object as input and gives a 1d-numpy array of atom features as output.
    """

    # define list of permitted atoms

    permitted_list_of_atoms = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As', 'Al', 'I',
                               'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'Li', 'Ge',
                               'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'Unknown']

    if hydrogens_implicit == False:
        permitted_list_of_atoms = ['H'] + permitted_list_of_atoms

    # compute atom features

    atom_type_enc = one_hot_encoding(str(atom.GetSymbol()), permitted_list_of_atoms)

    n_heavy_neighbors_enc = one_hot_encoding(int(atom.GetDegree()), [0, 1, 2, 3, 4, "MoreThanFour"])

    formal_charge_enc = one_hot_encoding(int(atom.GetFormalCharge()), [-3, -2, -1, 0, 1, 2, 3, "Extreme"])

    hybridisation_type_enc = one_hot_encoding(str(atom.GetHybridization()),
                                              ["S", "SP", "SP2", "SP3", "SP3D", "SP3D2", "OTHER"])

    is_in_a_ring_enc = [int(atom.IsInRing())]

    is_aromatic_enc = [int(atom.GetIsAromatic())]

    atomic_mass_scaled = [float((atom.GetMass() - 10.812) / 116.092)]

    vdw_radius_scaled = [float((Chem.GetPeriodicTable().GetRvdw(atom.GetAtomicNum()) - 1.5) / 0.6)]

    covalent_radius_scaled = [float((Chem.GetPeriodicTable().GetRcovalent(atom.GetAtomicNum()) - 0.64) / 0.76)]

    atom_feature_vector = atom_type_enc + n_heavy_neighbors_enc + formal_charge_enc + hybridisation_type_enc + is_in_a_ring_enc + is_aromatic_enc + atomic_mass_scaled + vdw_radius_scaled + covalent_radius_scaled

    if use_chirality == True:
        chirality_type_enc = one_hot_encoding(str(atom.GetChiralTag()),
                                              ["CHI_UNSPECIFIED", "CHI_TETRAHEDRAL_CW", "CHI_TETRAHEDRAL_CCW",
                                               "CHI_OTHER"])
        atom_feature_vector += chirality_type_enc

    if hydrogens_implicit == True:
        n_hydrogens_enc = one_hot_encoding(int(atom.GetTotalNumHs()), [0, 1, 2, 3, 4, "MoreThanFour"])
        atom_feature_vector += n_hydrogens_enc

    return np.array(atom_feature_vector)


def get_bond_features(bond,
                      use_stereochemistry=False):
    """
    Takes an RDKit bond object as input and gives a 1d-numpy array of bond features as output.
    """

    permitted_list_of_bond_types = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
                                    Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]

    bond_type_enc = one_hot_encoding(bond.GetBondType(), permitted_list_of_bond_types)

    bond_is_conj_enc = [int(bond.GetIsConjugated())]

    bond_is_in_ring_enc = [int(bond.IsInRing())]

    bond_feature_vector = bond_type_enc + bond_is_conj_enc + bond_is_in_ring_enc

    if use_stereochemistry == True:
        stereo_type_enc = one_hot_encoding(str(bond.GetStereo()), ["STEREOZ", "STEREOE", "STEREOANY", "STEREONONE"])
        bond_feature_vector += stereo_type_enc

    return np.array(bond_feature_vector)

class NetSimile(base.BaseDistance):
    """Compares node signature distributions."""

    #@undirected
    #@unweighted
    def dist(self, G1, G1SD, G1AF, G2, G2SD, G2AF):
        """A scalable approach to network similarity.

        A network similarity measure based on node signature distributions.

        The results dictionary includes the underlying feature matrices in
        `'feature_matrices'` and the underlying signature vectors in
        `'signature_vectors'`.

        Parameters
        ----------

        G1, G2 (nx.Graph)
            two undirected networkx graphs to be compared.

        Returns
        -------

        dist (float)
            the distance between `G1` and `G2`.

        References
        ----------

        .. [1] Michele Berlingerio, Danai Koutra, Tina Eliassi-Rad,
               Christos Faloutsos: NetSimile: A Scalable Approach to
               Size-Independent Network Similarity. CoRR abs/1209.2684
               (2012)

        """

        # find the graph node feature matrices
        G1_node_features = feature_extraction(G1, G1SD, G1AF)
        G2_node_features = feature_extraction(G2, G2SD, G2AF)

        # get the graph signature vectors
        G1_signature = graph_signature(G1_node_features)
        G2_signature = graph_signature(G2_node_features)
        if len(G1SD) == len(G2SD):
            G1_signature = np.append(G1_signature, G1SD)
            G2_signature = np.append(G2_signature, G2SD)

        # the final distance is the absolute canberra distance
        dist = abs(canberra(G1_signature, G2_signature))

        self.results['feature_matrices'] = G1_node_features, G2_node_features
        self.results['signature_vectors'] = G1_signature, G2_signature
        self.results['dist'] = dist

        return dist


def feature_extraction(G, G_SD, AtomFeatures):
    """Node feature extraction.

    Parameters
    ----------

    G (nx.Graph): a networkx graph.

    Returns
    -------

    node_features (float): the Nx7 matrix of node features."""

    # necessary data structures
    node_features = np.zeros(shape=(G.number_of_nodes(), 83))
    node_list = sorted(G.nodes())
    node_degree_dict = dict(G.degree())
    node_clustering_dict = dict(nx.clustering(G))
    egonets = {n: nx.ego_graph(G, n) for n in node_list}

    # node degrees
    degs = [node_degree_dict[n] for n in node_list]

    # clustering coefficient
    clusts = [node_clustering_dict[n] for n in node_list]

    # average degree of neighborhood
    neighbor_degs = [
        np.mean([node_degree_dict[m] for m in egonets[n].nodes if m != n])
        if node_degree_dict[n] > 0
        else 0
        for n in node_list
    ]

    # average clustering coefficient of neighborhood
    neighbor_clusts = [
        np.mean([node_clustering_dict[m] for m in egonets[n].nodes if m != n])
        if node_degree_dict[n] > 0
        else 0
        for n in node_list
    ]

    # number of edges in the neighborhood
    neighbor_edges = [
        egonets[n].number_of_edges() if node_degree_dict[n] > 0 else 0
        for n in node_list
    ]

    # number of outgoing edges from the neighborhood
    # the sum of neighborhood degrees = 2*(internal edges) + external edges
    # node_features[:,5] = node_features[:,0] * node_features[:,2] - 2*node_features[:,4]
    neighbor_outgoing_edges = [
        len(
            [
                edge
                for edge in set.union(*[set(G.edges(j)) for j in egonets[i].nodes])
                if not egonets[i].has_edge(*edge)
            ]
        )
        for i in node_list
    ]

    # number of neighbors of neighbors (not in neighborhood)
    neighbors_of_neighbors = [
        len(
            set([p for m in G.neighbors(n) for p in G.neighbors(m)])
            - set(G.neighbors(n))
            - set([n])
        )
        if node_degree_dict[n] > 0
        else 0
        for n in node_list
    ]



    # assembling the features
    node_features[:, 0] = degs
    node_features[:, 1] = clusts
    node_features[:, 2] = neighbor_degs
    node_features[:, 3] = neighbor_clusts
    node_features[:, 4] = neighbor_edges
    node_features[:, 5] = neighbor_outgoing_edges
    node_features[:, 6] = neighbors_of_neighbors
    node_features[:, 7] = G_SD
    node_features[:, 8:83] = AtomFeatures

    return np.nan_to_num(node_features)


def graph_signature(node_features):
    signature_vec = np.zeros(83 * 5)

    # for each of the 8 features
    for k in range(83):
        # find the mean
        signature_vec[k * 5] = node_features[:, k].mean()
        # find the median
        signature_vec[k * 5 + 1] = np.median(node_features[:, k])
        # find the std
        signature_vec[k * 5 + 2] = node_features[:, k].std()
        # find the skew
        signature_vec[k * 5 + 3] = skew(node_features[:, k])
        # find the kurtosis
        signature_vec[k * 5 + 4] = kurtosis(node_features[:, k])

    return signature_vec

def AtomFeatures(smi1):
    rdkitmol = Chem.MolFromSmiles(smi1)
    if rdkitmol is None:
        print(smi1)
        return np.nan
    n_nodes = rdkitmol.GetNumAtoms()
    #n_edges = 2*rdkitmol.GetNumBonds()
    unrelated_smiles = "O=O"
    unrelated_mol = Chem.MolFromSmiles(unrelated_smiles)
    n_node_features = len(get_atom_features(unrelated_mol.GetAtomWithIdx(0)))
    n_edge_features = len(get_bond_features(unrelated_mol.GetBondBetweenAtoms(0,1)))
    X = np.zeros((n_nodes, n_node_features))
    for atom in rdkitmol.GetAtoms():
        X[atom.GetIdx(), :] = get_atom_features(atom)
    return X
# # sample usage
# #def GraphComparison(pred_smi: str, true_smi: str, DensityArray: np.ndarray, true_DensityArray: np.ndarray) -> float:
# from pysmiles import read_smiles
# smi1 = 'NC1CCN(S(=O)(=O)c2ccc3[nH]c(=O)oc32)C1'
# smi2 = 'NC1CCN(S(=O)(=O)c2ccc3[nH]c(=O)oc3c2)C1'
# mol = read_smiles(smi1)
# mol1 = read_smiles(smi2)
#
# AF1 = AtomFeatures(smi1)
# AF2 = AtomFeatures(smi2)
# DensityArray = np.array([10, 10, 11, 11, 10, 10, 17, 18, 12, 10, 10, 10, 10, 10, 10, 10, 10, 10])
# true_DensityArray = np.array([10, 10, 11, 11, 10, 10, 17, 18, 12, 10, 10, 10, 10, 10, 10, 10, 10, 10,11])
# G1, G1_SD = mol, DensityArray
# G2, G2_SD = mol1, true_DensityArray
#
# test = NetSimile()
# distance = test.dist(G1, G1_SD, AF1, G2, G2_SD, AF2)
#
#
#
#
# print('ENDE')
