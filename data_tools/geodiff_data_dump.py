import os
import pickle
import copy
import json
from collections import defaultdict

import numpy as np
import random
import time
import torch
from torch_geometric.data import Data, Dataset, Batch

from torch_geometric.utils import to_networkx
from torch_scatter import scatter
#from torch.utils.data import Dataset

import rdkit
from rdkit import Chem
from rdkit.Chem.rdchem import Mol, HybridizationType, BondType
from rdkit import RDLogger
import networkx as nx
from tqdm import tqdm

from copy import deepcopy
import torch
from torchvision.transforms.functional import to_tensor
import rdkit
import rdkit.Chem.Draw
from rdkit import Chem
from rdkit.Chem import rdDepictor as DP
from rdkit.Chem import PeriodicTable as PT
from rdkit.Chem import rdMolAlign as MA
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem.rdchem import Mol,GetPeriodicTable
from rdkit.Chem.Draw import rdMolDraw2D as MD2
from rdkit.Chem.rdmolops import RemoveHs
from typing import List, Tuple

BOND_TYPES = {t: i for i, t in enumerate(BT.names.values())}
BOND_NAMES = {i: t for i, t in enumerate(BT.names.keys())}

import threading
import time

def rdmol_to_data(mol_set, conf_ids,smiles=None, data_cls=Data):
    mol = mol_set.get('conformers')[conf_ids[0]].get('rd_mol')
    assert mol.GetNumConformers() == 1
    N = mol.GetNumAtoms()
    atomic_number = []
    aromatic = []
    sp = []
    sp2 = []
    sp3 = []
    num_hs = []
    for atom in mol.GetAtoms():
        atomic_number.append(atom.GetAtomicNum())
        aromatic.append(1 if atom.GetIsAromatic() else 0)
        hybridization = atom.GetHybridization()
        sp.append(1 if hybridization == HybridizationType.SP else 0)
        sp2.append(1 if hybridization == HybridizationType.SP2 else 0)
        sp3.append(1 if hybridization == HybridizationType.SP3 else 0)

    z = torch.tensor(atomic_number, dtype=torch.long)

    row, col, edge_type = [], [], []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row += [start, end]
        col += [end, start]
        edge_type += 2 * [BOND_TYPES[bond.GetBondType()]]

    edge_index = torch.tensor([row, col], dtype=torch.long)
    edge_type = torch.tensor(edge_type)

    # perm = (edge_index[0] * N + edge_index[1]).argsort()
    # edge_index = edge_index[:, perm]
    # edge_type = edge_type[perm]

    row, col = edge_index
    hs = (z == 1).to(torch.float32)

    num_hs = scatter(hs[row], col, dim_size=N, reduce='sum').tolist()

    if smiles is None:
        smiles = Chem.MolToSmiles(mol)

    pos_list = []
    t_energy = []
    b_weight = []
    for conf_id in conf_ids:
        conf_meta = mol_set.get('conformers')[conf_id]
        mol = conf_meta.get('rd_mol')
        pos = torch.tensor(mol.GetConformer(0).GetPositions(), dtype=torch.float32)
        pos_list.append(pos)
        t_energy.append(conf_meta['totalenergy'])
        b_weight.append(conf_meta['boltzmannweight'])
    
    data = data_cls(atom_type=z, pos=pos_list, edge_index=edge_index, edge_type=edge_type,
                rdmol=copy.deepcopy(mol), totalenergy=t_energy, boltzmannweight=b_weight, smiles=smiles)
    #data.nx = to_networkx(data, to_undirected=True)

    return data

base_path = '/path/to/'
out_dir = base_path + 'data/diff/DRUGS/fragdiff_dump4geodiff_nostd/'
os.path.join(base_path, 'data', 'diff', 'DRUGS', 'drugs')
path_list = os.listdir(os.path.join(base_path, 'data', 'diff', 'DRUGS', 'drugs'))

import glob
root = os.path.join(base_path, 'data', 'diff', 'DRUGS', 'drugs')+'/'
files = np.array(sorted(glob.glob(f'{root}*.pickle')))
split_path = base_path + 'data/diff/DRUGS/split.npy'
max_conf=30
# train part
# train_list = []
# split_idx = 0
# split = sorted(np.load(split_path, allow_pickle=True)[split_idx])
# bad_case = 0
# for i in tqdm(range(len(split))):
#     try:
#         with open(files[split][i], 'rb') as fin:
#             mol = pickle.load(fin)
        
#         if mol.get('uniqueconfs') > len(mol.get('conformers')):
#             bad_case += 1
#             continue
#         if mol.get('uniqueconfs') <= 0:
#             bad_case += 1
#             continue
        
#         datas = []
#         smiles = mol.get('smiles')
        
#         if mol.get('uniqueconfs') <= max_conf:
#             # use all confs
#             conf_ids = np.arange(mol.get('uniqueconfs'))
#         else:
#             # filter the most probable 'max_conf' confs
#             all_weights = np.array([_.get('boltzmannweight', -1.) for _ in mol.get('conformers')])
#             descend_conf_id = (-all_weights).argsort()
#             conf_ids = descend_conf_id[:max_conf]
        
#         train_data_sample = rdmol_to_data(mol, conf_ids, smiles)
#         train_list.append(train_data_sample)
#     except:
#         continue
# with open(out_dir + 'train_data.pkl','wb') as f:
#     pickle.dump(train_list , f)
# print('train files dump finished!')
# print('bad case', bad_case)
# val_list
val_list = []
split_idx = 1
split = sorted(np.load(split_path, allow_pickle=True)[split_idx])
bad_case = 0
for i in tqdm(range(len(split))):
    try:
        with open(files[split][i], 'rb') as fin:
            mol = pickle.load(fin)
        
        if mol.get('uniqueconfs') > len(mol.get('conformers')):
            bad_case += 1
            continue
        if mol.get('uniqueconfs') <= 0:
            bad_case += 1
            continue
        
        datas = []
        smiles = mol.get('smiles')
        
        if mol.get('uniqueconfs') <= max_conf:
            # use all confs
            conf_ids = np.arange(mol.get('uniqueconfs'))
        else:
            # filter the most probable 'max_conf' confs
            all_weights = np.array([_.get('boltzmannweight', -1.) for _ in mol.get('conformers')])
            descend_conf_id = (-all_weights).argsort()
            conf_ids = descend_conf_id[:max_conf]
        
        val_data_sample = rdmol_to_data(mol, conf_ids, smiles)
        val_list.append(val_data_sample)
    except:
        continue
with open(out_dir + 'val_data.pkl','wb') as f:
    pickle.dump(val_list , f)
print('dump finished!')
print('bad case', bad_case)
# test_list
test_list = []
split_idx = 2
split = sorted(np.load(split_path, allow_pickle=True)[split_idx])
bad_case = 0
for i in tqdm(range(len(split))):
    try:
        with open(files[split][i], 'rb') as fin:
            mol = pickle.load(fin)
        
        if mol.get('uniqueconfs') > len(mol.get('conformers')):
            bad_case += 1
            continue
        if mol.get('uniqueconfs') <= 0:
            bad_case += 1
            continue
        
        datas = []
        smiles = mol.get('smiles')
        
        if mol.get('uniqueconfs') <= max_conf:
            # use all confs
            conf_ids = np.arange(mol.get('uniqueconfs'))
        else:
            # filter the most probable 'max_conf' confs
            all_weights = np.array([_.get('boltzmannweight', -1.) for _ in mol.get('conformers')])
            descend_conf_id = (-all_weights).argsort()
            conf_ids = descend_conf_id[:max_conf]
        
        test_data_sample = rdmol_to_data(mol, conf_ids, smiles)
        test_list.append(test_data_sample)
    except:
        continue
with open(out_dir + 'test_data.pkl','wb') as f:
    pickle.dump(test_list , f)
print('dump finished!')
print('bad case', bad_case)