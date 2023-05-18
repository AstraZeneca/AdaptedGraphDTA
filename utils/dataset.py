import torch
from rdkit import Chem
import networkx as nx
import pandas as pd
import json
import numpy as np
from torch_geometric.data import InMemoryDataset, Batch
from torch_geometric import data as DATA
from torch_geometric.loader import DataLoader
import torch
from utils.utils import *


class TestbedDataset(InMemoryDataset):
    def __init__(self, data_list=None):
        super(TestbedDataset, self).__init__()
        self.data_list=data_list
    def __len__(self):
        return len(self.data_list)
    def __getitem__(self, idx):
        dict_item=self.data_list[idx]
        pro_name = dict_item.get('uniprot_accession')
        ligand = dict_item.get('compound_iso_smiles')
        label = dict_item.get('pIC50')
        seq_coding = get_pro_seq(pro_name)
        c_size, features, edge_index=smile_to_graph(ligand)
        GCNData_mol = DATA.Data(x=torch.Tensor(np.array(features)),
                                edge_index=torch.LongTensor(np.array(edge_index)).transpose(1, 0),
                                y=torch.FloatTensor([label]))
        GCNData_mol.__setitem__('c_size', torch.LongTensor([c_size]))
        GCNData_mol.__setitem__('smiles', ligand)
        GCNData_mol.__setitem__('pro', pro_name)

        seq_coding_features= torch.LongTensor(seq_coding)

        return GCNData_mol, seq_coding_features

def seq_cat(prot):
    seq_voc = "ABCDEFGHIKLMNOPQRSTUVWXYZ-"
    seq_dict = {v:i for i,v in enumerate(seq_voc)}
    seq_dict_len = len(seq_dict)
    max_seq_len = 85
    x = np.zeros(max_seq_len)
    for i, ch in enumerate(prot[:max_seq_len]): 
        x[i] = seq_dict[ch]
    return x 

def get_pro_seq(name):
    protein_dic = json.load(open("proteins.txt"))
    seq = protein_dic[name]
    seq_coding = seq_cat(seq)
    return seq_coding

def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)
    c_size = mol.GetNumAtoms()
    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append( feature / sum(feature) )
    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    edge_index = []
    mol_adj = np.zeros((c_size, c_size))
    for e1, e2 in g.edges:
        mol_adj[e1, e2] = 1
    mol_adj += np.matrix(np.eye(mol_adj.shape[0]))
    index_row, index_col = np.where(mol_adj >= 0.5)
    for i, j in zip(index_row, index_col):
        edge_index.append([i, j])   
    return c_size, features, edge_index

def atom_features(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na','Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb','Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H','Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr','Cr', 'Pt', 'Hg', 'Pb', 'Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    [atom.GetIsAromatic()])

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def load_data(params, dataset, test=False):
    if test == True: 
        data = pd.read_csv('Data/' + dataset + '/test.csv').to_dict('index')
        shuffle = False
    else:
        data = pd.read_csv('Data/' + dataset + '/train.csv').to_dict('index')
        shuffle = True
    data = TestbedDataset( data_list=data)
    loader = DataLoader(data, batch_size=params.get("batch_size", 128),  num_workers=params.get("num_workers", 1), shuffle=shuffle) 
    return loader

def predict(model, device, loader, name_study):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    total_smiles = []
    total_pro_1D = []
    logging('Make prediction for {} samples...'.format(len(loader.dataset)), name_study)
    with torch.no_grad():
        for data in loader:
            data_mol = data[0].to(device)
            pro_1D = data[1].to(device)
            output = model(data_mol, pro_1D)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data_mol.y.view(-1, 1).cpu()), 0)
            total_smiles = total_smiles + data_mol.smiles
            total_pro_1D = total_pro_1D + data_mol.pro

    return total_labels.numpy().flatten(),total_preds.numpy().flatten(),  total_smiles, total_pro_1D

