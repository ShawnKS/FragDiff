import glob
import os.path as osp
import pickle
import numpy as np
from rdkit import Chem
from rdkit.Chem.rdchem import HybridizationType
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem.rdchem import ChiralType

import torch
import torch.nn.functional as F
from torch_scatter import scatter
from torch_geometric.data import Dataset, Data, DataLoader

import networkx as nx
import numpy as np
import torch, copy
from scipy.spatial.transform import Rotation as R
from torch_geometric.utils import to_networkx
from torch_geometric.data import Data
from rdkit.Chem import Recap,BRICS
from rdkit.Chem import Draw
from utils.brics_rec import FindBRICSBonds
# RECAP
def FindRECAPBonds(mol, randomizeOrder=False, silent=True):
    reactionDefs = (
    "[#7;+0;D2,D3:1]!@C(!@=O)!@[#7;+0;D2,D3:2]>>*[#7:1].[#7:2]*",  # urea
    "[C;!$(C([#7])[#7]):1](=!@[O:2])!@[#7;+0;!D1:3]>>*[C:1]=[O:2].*[#7:3]",  # amide
    "[C:1](=!@[O:2])!@[O;+0:3]>>*[C:1]=[O:2].[O:3]*",  # ester
    "[N;!D1;+0;!$(N-C=[#7,#8,#15,#16])](-!@[*:1])-!@[*:2]>>*[*:1].[*:2]*",  # amines
    # "[N;!D1](!@[*:1])!@[*:2]>>*[*:1].[*:2]*", # amines

    # again: what about aromatics?
    "[#7;R;D3;+0:1]-!@[*:2]>>*[#7:1].[*:2]*",  # cyclic amines
    "[#6:1]-!@[O;+0]-!@[#6:2]>>[#6:1]*.*[#6:2]",  # ether
    "[C:1]=!@[C:2]>>[C:1]*.*[C:2]",  # olefin
    "[n;+0:1]-!@[C:2]>>[n:1]*.[C:2]*",  # aromatic nitrogen - aliphatic carbon
    "[O:3]=[C:4]-@[N;+0:1]-!@[C:2]>>[O:3]=[C:4]-[N:1]*.[C:2]*",  # lactam nitrogen - aliphatic carbon
    "[c:1]-!@[c:2]>>[c:1]*.*[c:2]",  # aromatic carbon - aromatic carbon
    # aromatic nitrogen - aromatic carbon *NOTE* this is not part of the standard recap set.
    "[n;+0:1]-!@[c:2]>>[n:1]*.*[c:2]",
    "[#7;+0;D2,D3:1]-!@[S:2](=[O:3])=[O:4]>>[#7:1]*.*[S:2](=[O:3])=[O:4]",  # sulphonamide
    )
    bondMatchers = []
    for reaction in reactionDefs:
        reactants, products = reaction.split('>>')
        patt = Chem.MolFromSmarts(reactants)
        bondMatchers.append(patt)

    indices = list(range(len(bondMatchers)))
    if randomizeOrder:
        rng = random.Random()
        rng.shuffle(indices)
    edges_list = []
    for indice in indices:
        patt = bondMatchers[indice]
        edge_idx = mol.GetSubstructMatches(patt)
        for match in edge_idx:
            # 将多元组转换为二元组
            # 这里假设您希望从多元组中提取连续的两两元素
            two_pairs = [(match[i], match[i + 1]) for i in range(len(match) - 1)]
            edges_list.extend(two_pairs)
    return list(set(edges_list))

def FindBRBonds(mol, randomizeOrder=False, silent=True):
    reactionDefs = (
    "[#7;+0;D2,D3:1]!@C(!@=O)!@[#7;+0;D2,D3:2]>>*[#7:1].[#7:2]*",  # urea
    "[C;!$(C([#7])[#7]):1](=!@[O:2])!@[#7;+0;!D1:3]>>*[C:1]=[O:2].*[#7:3]",  # amide
    "[C:1](=!@[O:2])!@[O;+0:3]>>*[C:1]=[O:2].[O:3]*",  # ester
    "[N;!D1;+0;!$(N-C=[#7,#8,#15,#16])](-!@[*:1])-!@[*:2]>>*[*:1].[*:2]*",  # amines
    # "[N;!D1](!@[*:1])!@[*:2]>>*[*:1].[*:2]*", # amines

    # again: what about aromatics?
    "[#7;R;D3;+0:1]-!@[*:2]>>*[#7:1].[*:2]*",  # cyclic amines
    "[#6:1]-!@[O;+0]-!@[#6:2]>>[#6:1]*.*[#6:2]",  # ether
    "[C:1]=!@[C:2]>>[C:1]*.*[C:2]",  # olefin
    "[n;+0:1]-!@[C:2]>>[n:1]*.[C:2]*",  # aromatic nitrogen - aliphatic carbon
    "[O:3]=[C:4]-@[N;+0:1]-!@[C:2]>>[O:3]=[C:4]-[N:1]*.[C:2]*",  # lactam nitrogen - aliphatic carbon
    "[c:1]-!@[c:2]>>[c:1]*.*[c:2]",  # aromatic carbon - aromatic carbon
    # aromatic nitrogen - aromatic carbon *NOTE* this is not part of the standard recap set.
    "[n;+0:1]-!@[c:2]>>[n:1]*.*[c:2]",
    "[#7;+0;D2,D3:1]-!@[S:2](=[O:3])=[O:4]>>[#7:1]*.*[S:2](=[O:3])=[O:4]",  # sulphonamide
    )
    bondMatchers = []
    for reaction in reactionDefs:
        reactants, products = reaction.split('>>')
        patt = Chem.MolFromSmarts(reactants)
        bondMatchers.append(patt)

    indices = list(range(len(bondMatchers)))
    if randomizeOrder:
        rng = random.Random()
        rng.shuffle(indices)
    edges_list = []
    for indice in indices:
        patt = bondMatchers[indice]
        edge_idx = mol.GetSubstructMatches(patt)
        for match in edge_idx:
            # 将多元组转换为二元组
            # 这里假设您希望从多元组中提取连续的两两元素
            two_pairs = [(match[i], match[i + 1]) for i in range(len(match) - 1)]
            edges_list.extend(two_pairs)
    edges_list = edges_list + FindBRICSBonds(mol)
    return list(set(edges_list))