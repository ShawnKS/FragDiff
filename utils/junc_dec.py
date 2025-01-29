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
from collections import defaultdict
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from rdkit import Chem

MST_MAX_WEIGHT = 100  # Assuming a constant value for MST_MAX_WEIGHT
from collections import defaultdict
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from rdkit import Chem

MST_MAX_WEIGHT = 100  # Assuming a constant value for MST_MAX_WEIGHT

def tree_decomp_with_shared_edges(mol):
    n_atoms = mol.GetNumAtoms()
    if n_atoms == 1:  # special case
        return [[0]], [], []

    cliques = []
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom().GetIdx()
        a2 = bond.GetEndAtom().GetIdx()
        if not bond.IsInRing():
            cliques.append([a1, a2])

    ssr = [list(x) for x in Chem.GetSymmSSSR(mol)]
    cliques.extend(ssr)

    nei_list = [[] for _ in range(n_atoms)]
    for i in range(len(cliques)):
        for atom in cliques[i]:
            nei_list[atom].append(i)

    # Merge Rings with intersection > 2 atoms
    for i in range(len(cliques)):
        if len(cliques[i]) <= 2:
            continue
        for atom in cliques[i]:
            for j in nei_list[atom]:
                if i >= j or len(cliques[j]) <= 2:
                    continue
                inter = set(cliques[i]) & set(cliques[j])
                if len(inter) > 2:
                    cliques[i].extend(cliques[j])
                    cliques[i] = list(set(cliques[i]))
                    cliques[j] = []

    cliques = [c for c in cliques if len(c) > 0]
    nei_list = [[] for _ in range(n_atoms)]
    for i in range(len(cliques)):
        for atom in cliques[i]:
            nei_list[atom].append(i)

    # Build edges and add singleton cliques
    edges = defaultdict(int)
    shared_edges = defaultdict(list)
    for atom in range(n_atoms):
        if len(nei_list[atom]) <= 1:
            continue
        cnei = nei_list[atom]
        bonds = [c for c in cnei if len(cliques[c]) == 2]
        rings = [c for c in cnei if len(cliques[c]) > 4]
        if len(bonds) > 2 or (len(bonds) == 2 and len(cnei) > 2):
            cliques.append([atom])
            c2 = len(cliques) - 1
            for c1 in cnei:
                edges[(c1, c2)] = 1
        elif len(rings) > 2:
            cliques.append([atom])
            c2 = len(cliques) - 1
            for c1 in cnei:
                edges[(c1, c2)] = MST_MAX_WEIGHT - 1
        else:
            for i in range(len(cnei)):
                for j in range(i + 1, len(cnei)):
                    c1, c2 = cnei[i], cnei[j]
                    inter = set(cliques[c1]) & set(cliques[c2])
                    if edges[(c1, c2)] < len(inter):
                        edges[(c1, c2)] = len(inter)
                        shared_edges[(c1, c2)] = list(inter)

    edges = [u + (MST_MAX_WEIGHT - v,) for u, v in edges.items()]
    if len(edges) == 0:
        return cliques, edges, shared_edges

    # Compute Maximum Spanning Tree
    row, col, data = zip(*edges)
    junc = zip(*edges)
    n_clique = len(cliques)
    clique_graph = csr_matrix((data, (row, col)), shape=(n_clique, n_clique))
    junc_tree = minimum_spanning_tree(clique_graph)
    row, col = junc_tree.nonzero()
    edges = [(row[i], col[i]) for i in range(len(row))]

    # Get the actual shared edges for the MST edges
    actual_shared_edges = [(shared_edges[edge]) for edge in edges if edge in shared_edges]
    actual_shared_edges = [edge for edge in actual_shared_edges if len(edge)==2]

    return cliques, edges, actual_shared_edges

from rdkit import Chem
import random

def find_subring_atoms(mol, shared_edge):
    # 创建一个图的邻接表
    adjacency_list = {i: [] for i in range(mol.GetNumAtoms())}
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtomIdx()
        a2 = bond.GetEndAtomIdx()
        adjacency_list[a1].append(a2)
        adjacency_list[a2].append(a1)

    def dfs(current, target, visited):
        visited.add(current)
        for neighbor in adjacency_list[current]:
            if neighbor not in visited:
                if neighbor is not target:
                    dfs(neighbor, target, visited)
                else:
                    visited.add(target)
    # 从共享边的一个原子开始，随机选择一个不是终点的邻居节点作为起始方向
    start, end = shared_edge
    neighbors = [n for n in adjacency_list[start] if n != end]
    if not neighbors:
        return set()  # 如果没有其他邻居，返回空集

    # 随机选择一个邻居节点作为初始方向
    initial_neighbor = random.choice(neighbors)

    # 从起始节点进行DFS遍历，确保访问到目标节点
    visited = set()
    visited.add(start)
    dfs(initial_neighbor, end, visited)

    return visited


def get_ring_edge_indices(mol, ring_edges):
    # 创建一个空列表来存储环边的索引
    ring_edge_indices = []

    # 遍历分子中的所有化学键
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtomIdx()
        a2 = bond.GetEndAtomIdx()

        # 检查化学键是否在环边集合中
        if [a1, a2] in ring_edges or [a2, a1] in ring_edges:
            # 如果是，记录该化学键的索引
            ring_edge_indices.append(bond.GetIdx())

    return ring_edge_indices

def get_non_ring_edges_and_atoms(mol):
    # 获取分子的环信息
    ring_info = mol.GetRingInfo()

    # 创建两个列表来存储不在环上的边的索引和对应的原子索引对
    non_ring_edge_indices = []
    non_ring_atom_pairs = []

    # 遍历分子中的所有化学键
    for bond in mol.GetBonds():
        bond_idx = bond.GetIdx()
        a1 = bond.GetBeginAtomIdx()
        a2 = bond.GetEndAtomIdx()

        # 检查化学键是否在任何环中
        if not ring_info.NumBondRings(bond_idx):
            # 如果不在环中，记录该化学键的索引和对应的原子索引对
            non_ring_edge_indices.append(bond_idx)
            non_ring_atom_pairs.append([a1, a2])
            non_ring_atom_pairs.append([a2, a1])

    return non_ring_edge_indices, np.array(non_ring_atom_pairs)

def get_other_atoms(mol, subgraph_atoms):
    # 获取所有原子索引
    all_atoms = set(range(mol.GetNumAtoms()))
    # 计算其他原子的索引
    other_atoms = set(all_atoms) - set(subgraph_atoms)

    return list(other_atoms)

def find_neighbors(mol, atom_idx, exclude_idx):
    atom = mol.GetAtomWithIdx(atom_idx)
    neighbors = [neighbor.GetIdx() for neighbor in atom.GetNeighbors() if neighbor.GetIdx() != exclude_idx]
    return neighbors

def FindJuncBonds(pyg_data, edge_list, G):
    mol = pyg_data.mol
    # iedge_list = pyg_data.edge_index.T.tolist()
    _ , _, ring_edges = tree_decomp_with_shared_edges(mol)
    rotate_index = []
    if len(ring_edges) > 0:
        for edge in ring_edges:
        # side_atoms = get_side_of_edge(mol, edge)
            side_atoms = find_subring_atoms(mol, edge)
            rotate_index.append(edge_list.index([edge[0], edge[1]]))
        rotate_index = list(set(rotate_index))
        # num_edges = mol.GetNumBonds()
        # 有概率出现重复ring_edges
        mask_edges = np.zeros( len(edge_list), dtype=bool)
        mask_edges[np.array( rotate_index )] = True
        mask_rotate = np.zeros(( np.sum(mask_edges), len(G.nodes()) ), dtype=bool)
        idx = 0
        for i in range(np.sum(mask_edges)):
            # side_atoms = get_side_of_edge(mol, edge)
            # img = highlight_atoms(mol, list(edge))
            # img.show()
            edge = ring_edges[i]
            side_atoms = find_subring_atoms(mol, edge)
            mask_rotate[i][np.asarray(np.asarray(list(side_atoms)), dtype=int)] = True
        sorted_indices = sorted(range(len(rotate_index)), key=lambda x: rotate_index[x])
        # decompose的边顺序(按照每条规则match而不是纯按edge顺序的match)需要和edge_idex里面的顺序对齐
        mask_rotate = np.array([mask_rotate[i] for i in sorted_indices], dtype=bool)
        return mask_edges, mask_rotate, rotate_index
    else:
        return None, None, None

