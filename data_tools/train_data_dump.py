import numpy as np
import pandas as pd
import pickle

import numpy as np
from rdkit import Chem
from rdkit.Chem.rdchem import HybridizationType
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem.rdchem import ChiralType
import networkx as nx
import numpy as np
import torch, copy
from scipy.spatial.transform import Rotation as R
from torch_geometric.utils import to_networkx
from torch_geometric.data import Data

import torch
import torch.nn.functional as F
from torch_scatter import scatter
from torch_geometric.data import Dataset, Data, DataLoader
from utils.featurization import dihedral_pattern, featurize_mol, qm9_types, qm9_types
from multiprocessing import Pool
from utils.torsion import get_transformation_mask_old

def featurize_mol_(mol_dic):
    confs = mol_dic['conformers']
    name = mol_dic["smiles"]

    mol_ = Chem.MolFromSmiles(name)
    canonical_smi = Chem.MolToSmiles(mol_, isomericSmiles=False)

    pos = []
    weights = []
    for conf in confs:
        mol = conf['rd_mol']

        # filter for conformers that may have reacted
        try:
            conf_canonical_smi = Chem.MolToSmiles(Chem.RemoveHs(mol, sanitize=False), isomericSmiles=False)
        except Exception as e:
            print(e)
            continue

        if conf_canonical_smi != canonical_smi:
            continue

        pos.append(torch.tensor(mol.GetConformer().GetPositions(), dtype=torch.float))
        weights.append(conf['boltzmannweight'])
        correct_mol = mol

    # return None if no non-reactive conformers were found
    if len(pos) == 0:
        return None

    data = featurize_mol(correct_mol, types)
    normalized_weights = list(np.array(weights) / np.sum(weights))
    if np.isnan(normalized_weights).sum() != 0:
        print(name, len(confs), len(pos), weights)
        normalized_weights = [1 / len(weights)] * len(weights)
    data.canonical_smi, data.mol, data.pos, data.weights = canonical_smi, correct_mol, pos, normalized_weights
    return data

# refactor for qm9
file_path = '/path/to/QM9/split.npy'
split = np.load(file_path, allow_pickle=True)

import glob
import os.path as osp
from tqdm import tqdm
mols_per_pickle = 1000
root = '/path/to/QM9/qm9/'
split_path = '/path/to/QM9/split.npy'
# mode = 'train'
import os
mode='val'
split_idx = 0 if mode == 'train' else 1 if mode == 'val' else 2
split = sorted(np.load(split_path, allow_pickle=True)[split_idx])
limit_molecules = 0
if limit_molecules:
    split = split[:limit_molecules]
smiles = np.array(sorted(glob.glob(osp.join(root, '*.pickle'))))
smiles = smiles[split]
pickle_dir = '/path/to/QM9/standardized_pickles'
smiles = [(i // mols_per_pickle, smi[len(root):-7]) for i, smi in zip(split, smiles)]
if limit_molecules:
    smiles = smiles[:limit_molecules]
current_pickle = (None, None)
pickle_dir = pickle_dir

# for smile in tqdm(smiles):
def filter_smiles(smile):
    try:
        if type(smile) is tuple:
            pickle_id, smile = smile
        else:
            return False
        path = osp.join(pickle_dir, str(pickle_id).zfill(3) + '.pickle')
        try:
            with open(path, 'rb') as f:
                current_pickle = current_id, current_pickle = pickle_id, pickle.load(f)
        except:
            return False
        if smile not in current_pickle:
            return False
        mol_dic = current_pickle[smile]
        smile = mol_dic['smiles']
        if '.' in smile:
            return False
        mol = Chem.MolFromSmiles(smile)  #print(smile)
        if not mol:
            return False
        mol = mol_dic['conformers'][0]['rd_mol']
        N = mol.GetNumAtoms()
        if not mol.HasSubstructMatch(dihedral_pattern):
            return False
        if N < 4:
            return False
        data = featurize_mol_(mol_dic)
        edge_mask, mask_rotate, conn_rotate_idx = get_transformation_mask_old(data)
        data.edge_mask = torch.tensor(edge_mask)
        data.mask_rotate = mask_rotate
        # data_list.append(data)
        return data
    except:
        return None

output_dir = '/path/to/QM9/fragdiff_dump/'
os.makedirs(output_dir, exist_ok=True)
output_file_pattern = osp.join(output_dir, 'val_samples_{}.pkl')

def process_batch(batch_smiles, batch_index):
    """Process a single batch of SMILES and save the results."""
    datapoints = []
    with Pool(processes=32) as p:
        with tqdm(total=len(batch_smiles)) as pbar:
            map_fn = p.imap if len(batch_smiles) > 1 else map
            for t in map_fn(filter_smiles, batch_smiles):
                if t:
                    datapoints.append(t)
                pbar.update()
    
    # Save processed batch to a new file
    output_file = output_file_pattern.format(batch_index + 1)
    with open(output_file, 'wb') as f:
        pickle.dump(datapoints, f)

types = qm9_types
batch_size = 20000

for i in range(0, len(smiles), batch_size):
    print('processing smiles of batch', i)
    batch_smiles = smiles[i:i + batch_size]
    process_batch(batch_smiles, i // batch_size)
# types = qm9_types
# num_workers = 32
# p = Pool(num_workers)
# p.__enter__()
# datapoints=[]
# with tqdm(total=len(smiles)) as pbar:
#     map_fn = p.imap if num_workers > 1 else map
#     for t in map_fn(filter_smiles, smiles):
#         if t:
#             datapoints.append(t)
#         pbar.update()

# with open('/path/to/QM9/fragdiff_dump/train_samples.pkl', 'wb') as f:
#     pickle.dump(datapoints, f)

# p.__exit__(None, None, None)