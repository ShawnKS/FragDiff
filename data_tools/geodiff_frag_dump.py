import os
import pickle
import copy
import json
from collections import defaultdict

import numpy as np
import random

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
from torch_geometric.utils import to_dense_adj, dense_to_sparse

from utils.brics_rec import FindBRICSBonds
from utils.recap_rec import FindRECAPBonds, FindBRBonds
from utils.junc_dec import FindJuncBonds, get_non_ring_edges_and_atoms
from utils.connect import ConnectBonds
from torch_geometric.data import Data, Batch
from utils.frag_aug import frag_aug, frag_aug_conn,fragmentation, chemfrag_bondidx
from utils.featurization import dihedral_pattern, featurize_mol, qm9_types, drugs_types
from utils.torsion import *

BOND_TYPES = {t: i for i, t in enumerate(BT.names.values())}
BOND_NAMES = {i: t for i, t in enumerate(BT.names.keys())}
def get_transformation_mask_old(pyg_data):
    G = to_networkx(pyg_data, to_undirected=False)
    to_rotate = []
    edges = pyg_data.edge_index.T.numpy()
    rotate_idx = []
    for i in range(0, edges.shape[0], 2):
        assert edges[i, 0] == edges[i+1, 1]

        G2 = G.to_undirected()
        G2.remove_edge(*edges[i])
        if not nx.is_connected(G2):
            l = list(sorted(nx.connected_components(G2), key=len)[0])
            if len(l) > 1:
                if edges[i, 0] in l:
                    to_rotate.append([])
                    to_rotate.append(l)
                    rotate_idx.append(i+1)
                else:
                    to_rotate.append(l)
                    to_rotate.append([])
                    rotate_idx.append(i)
                continue
        to_rotate.append([])
        to_rotate.append([])

    mask_edges = np.asarray([0 if len(l) == 0 else 1 for l in to_rotate], dtype=bool)
    mask_rotate = np.zeros((np.sum(mask_edges), len(G.nodes())), dtype=bool)
    idx = 0
    for i in range(len(G.edges())):
        if mask_edges[i]:
            mask_rotate[idx][np.asarray(to_rotate[i], dtype=int)] = True
            idx += 1

    return mask_edges, mask_rotate, rotate_idx

def rdmol_edge_reorder(mol:Mol, smiles=None, data_cls=Data):
    row, col, edge_type = [], [], []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row += [start, end]
        col += [end, start]
        edge_type += 2 * [BOND_TYPES[bond.GetBondType()]]

    edge_index = torch.tensor([row, col], dtype=torch.long)
    edge_type = torch.tensor(edge_type)
    return edge_index, edge_type

def fragmentation(data, rotate_idx, rotate_idx_sel, z=20):
    augmented_data = copy.deepcopy(data)
    G = to_networkx(augmented_data, to_undirected=False)
    G2 = G.to_undirected()
    if len(rotate_idx_sel) < 2:
        return None
        # TODO Fix attr mismatching 
        data_batch = Batch.from_data_list([data])
        # print(data_batch)
        # data_batch.mask_rotate = combine_mask_rotate(data_batch)
        return [data_batch]
    # 如果rotate_idx的长度大于3，随机选择K在1到len(rotate_idx)-1之间
    if len(rotate_idx_sel) > 1:
        K = np.random.randint(1, min(len(rotate_idx_sel),5))
    remove_edges_idx = np.random.choice(rotate_idx_sel, K, replace=False)
    rest_edges_idx = np.setdiff1d(rotate_idx, remove_edges_idx)
    iedge_list = augmented_data.edge_index.T.numpy()
    iedge_type = augmented_data.edge_type
    remove_edges = copy.deepcopy(iedge_list)[remove_edges_idx]
    rest_edges = copy.deepcopy(iedge_list)[rest_edges_idx]
    for i in range(len(remove_edges)):
        G2.remove_edge(*remove_edges[i])
    modified_array = np.where(remove_edges_idx % 2 == 0, remove_edges_idx + 1, remove_edges_idx - 1)
    remove_edges_idx = np.concatenate((remove_edges_idx, modified_array))
    edge_list_after_removal = np.delete(iedge_list, remove_edges_idx, axis=0)
    # edge_type_after_removal = np.delete(iedge_type, remove_edges_idx, axis=0)
    mask = torch.ones(iedge_type.size(0), dtype=torch.bool)
    mask[remove_edges_idx] = False
    edge_type_after_removal = iedge_type[mask]
    iedge_list = edge_list_after_removal.tolist()
    mask_edges = np.zeros( len(iedge_list), dtype=bool)
    G3 = copy.deepcopy(G2)
    rotate_idx = []
    to_rotate = []
    for i in range(len(rest_edges)):
        G3 = copy.deepcopy(G2)
        G3.remove_edge(*rest_edges[i])
        n_u = rest_edges[i][0]
        n_v = rest_edges[i][1]
        connected_components = nx.connected_components(G3)
        n_u_component_size = None
        n_u_component_index = None
        n_v_component_size = None
        n_v_component_index = None
        
        for index, component in enumerate(connected_components):
            if n_u in component:
                n_u_component_size = len(component)
                n_u_component_index = index
            elif n_v in component:
                n_v_component_size = len(component)
                n_v_component_index = index
            
            if n_u_component_size is not None and n_v_component_size is not None:
                break
        if n_u_component_size is None or n_v_component_size is None:
            continue
                # large part append the node index in smaller part
        if n_u_component_size > n_v_component_size:
            rotate_index = iedge_list.index([n_u, n_v])
            rotate_idx.append(rotate_index)
            mask_edges[rotate_index] = True
            l = list(list(nx.connected_components(G3))[n_v_component_index])
            to_rotate.append(l)
            # mask_edges.append(rotate_index)
        else:
            rotate_index = iedge_list.index([n_v, n_u])
            rotate_idx.append(rotate_index)
            mask_edges[rotate_index] = True
            l = list(list(nx.connected_components(G3))[n_u_component_index])
            to_rotate.append(l)
    mask_rotate = np.zeros((np.sum(mask_edges), len(G2.nodes())), dtype=bool)
    for i in range( np.sum(mask_edges) ):
        mask_rotate[i][np.asarray(to_rotate[i], dtype=int)] = True
        # print(mask_rotate[idx][np.asarray(to_rotate[idx], dtype=int)])
    sorted_indices = sorted(range(len(rotate_idx)), key=lambda x: rotate_idx[x])
    mask_rotate = np.array([mask_rotate[i] for i in sorted_indices], dtype=bool)
    connected_components = list(nx.connected_components(G2))
# 存储新建的 data 对象
    new_data_list = []
    # 获取原始图中需要旋转的边的索引
    original_rotate_indices = [rotate_idx[i] for i in sorted_indices]
    for component in connected_components:
        # 获取当前连通分量的节点列表
        component_nodes = list(component)
        
        # 创建节点索引映射：原始索引 -> 新索引
        node_index_map = {node: idx for idx, node in enumerate(component_nodes)}
        
        # 提取连通分量中的边
        # component_edges = [(u, v) for u, v in G2.edges(component) if u in component and v in component]
        component_edges = []
        for u, v in G2.edges(component):
            if u in component and v in component:
                component_edges.append([u,v])
                component_edges.append([v,u])
        
        # 重新映射边索引
        new_edge_index = np.array([[node_index_map[u], node_index_map[v]] for u, v in component_edges]).T
        
        # 提取边属性和掩码
        edge_indices = [iedge_list.index([u, v]) for u, v in component_edges]
        # new_edge_type = iedge_type[edge_indices]
        new_edge_type = edge_type_after_removal[edge_indices]
        new_edge_mask = mask_edges[edge_indices]
        new_edge_list = [iedge_list[i] for i in edge_indices]
    # 提取旋转轴和影响节点
        new_mask_rotate = []
        new_rotate_order = []
        for original_index in original_rotate_indices:
            # 检查该旋转边是否在当前连通分量中
            u, v = iedge_list[original_index]
            if u in component_nodes and v in component_nodes:
                # 获取原始旋转轴影响的节点掩码
                original_mask = mask_rotate[original_rotate_indices.index(original_index)]
                # 映射到新节点索引
                new_mask = np.zeros(len(component_nodes), dtype=bool)
                for original_node_index in np.where(original_mask)[0]:
                    if original_node_index in node_index_map:
                        new_node_index = node_index_map[original_node_index]
                        new_mask[new_node_index] = True
                new_rotate_order.append(new_edge_list.index([u,v]) )
                new_mask_rotate.append(new_mask)
        if len(new_mask_rotate)<1:
            continue
        else:
            # print(new_rotate_order)
            sorted_indices = sorted(range(len(new_rotate_order)), key=lambda x: new_rotate_order[x])
            new_mask_rotate = np.array([new_mask_rotate[i] for i in sorted_indices], dtype=bool)
        new_mask_rotate = np.array(new_mask_rotate, dtype=bool)
        # max_weight_index = torch.argmax(torch.tensor(augmented_data.weights))
        # pos = data.pos[max_weight_index]
        # weight = random.choice(data.weights)
        # pos = data.pos[component_nodes]
        pos = [pos[component_nodes] for pos in data.pos]
        # w = [data.weights[max_weight_index]]
        # w = data.weights
        # z = [data.z[i] for i in component_nodes]
        
        # 创建新的 data 对象
        new_data = Data(
            # x= data.x[component_nodes],  # 提取连通分量中的节点特征
            edge_index=torch.tensor(new_edge_index, dtype=torch.long),  # 设置新的边索引
            edge_type=new_edge_type,  # 设置新的边属性
            atom_type = data.atom_type[component_nodes],  # 提取连通分量中的z属性
            pos = pos,
            smiles = data.smiles,
            rdmol = data.rdmol,
            boltzmannweight = data.boltzmannweight,
            # weights = w,
            # canonical_smi = data.canonical_smi,  
            # mol = data.mol,
            # edge_mask=torch.tensor(new_edge_mask),  # 设置新的边掩码
            # mask_rotate=new_mask_rotate  # 设置新的旋转掩码
        )
        
        # 将新创建的 data 对象添加到列表中
        new_data_list = []
        if len(new_data.atom_type) > z and not isinstance(new_data, Batch):
            # print(new_data)
            # new_data_list.append(Batch.from_data_list([new_data]))
            new_data_list.append(new_data)
    # if len(new_data_list)>0:
    #     data_batch = Batch.from_data_list([i for i in new_data_list])
            return new_data_list
        else:
            return None
    # if len(new_data_list)>0:
    #     data_batch = Batch.from_data_list([i for i in new_data_list] + [data])
    #     data_batch.mask_rotate = combine_mask_rotate(data_batch)
    #     return data_batch
    # else:
    #     data_batch = Batch.from_data_list([data])
    #     data_batch.mask_rotate = combine_mask_rotate(data_batch)
    #     return data_batch
    # return new_data_list

def is_tuple(obj):
    if hasattr(obj, '__getitem__') and hasattr(obj, '__iter__'):
        return all(hasattr(obj, attr) for attr in ('count', 'index'))
    return False

directories = [
    '/path/to/DRUGS/fragdiff_dump4geodiff_nostd/',
]
noaug = None
if noaug:
    output_dir = f'/path/to/DRUGS/fragdiff_dump4geodiff_nostd/_frag_{noaug}/'
else:
    output_dir = f'/path/to/DRUGS/fragdiff_dump4geodiff_nostd/_frag/'
 # + '_converted' 
for dirs in directories:
    # print(dirs)
    for file in os.listdir(dirs):
        pkl_path = os.path.join(dirs, file)
        if os.path.isdir(pkl_path):
            continue
        output_file_path = output_dir + file
        print(output_file_path)
        print(pkl_path)
        with open(pkl_path, 'rb') as f:
            train_data = pickle.load(f)
# with open('/path/to/data/new_geodiff/Drugs/_converted/val_data_5k.pkl', 'rb') as f:
#     train_data = pickle.load(f)

        frag_list = []
        for data in tqdm(train_data):
            data.mol = data.rdmol
            edge_mask, mask_rotate, conn_rotate_idx = get_transformation_mask_old(data)
            if noaug:
                rotate_idx = chemfrag_bondidx(data, edge_mask, noaug)
                rotate_idx = [item for item in conn_rotate_idx if item not in rotate_idx]
            else:
                rotate_idx = conn_rotate_idx
            data_list = fragmentation(data, conn_rotate_idx, rotate_idx, 5)
            if data_list != None and not is_tuple(data_list[0]):
                # print(data_list)
                frag_list += data_list
        # new_save_path = pkl_path.replace('_converted', '_frag')
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
        with open(output_file_path, 'wb') as f:
            pickle.dump(frag_list , f)



























