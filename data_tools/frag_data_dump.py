import networkx as nx
import numpy as np
import os
import re
import torch, copy
from scipy.spatial.transform import Rotation as R
from torch_geometric.utils import to_networkx
from torch_geometric.data import Data
from utils.brics_rec import FindBRICSBonds
from utils.recap_rec import FindRECAPBonds, FindBRBonds
from utils.junc_dec import FindJuncBonds, get_non_ring_edges_and_atoms
from utils.connect import ConnectBonds
from torch_geometric.data import Data, Batch
from utils.frag_aug import frag_aug, frag_aug_conn,fragmentation, chemfrag_bondidx
from utils.featurization import dihedral_pattern, featurize_mol, qm9_types, drugs_types
from utils.torsion import *

def combine_mask_rotate(data_batch):
    # 获取总列数
    total_cols = data_batch.x.size(0)  # 假设x的行数为总列数
    # 初始化一个空的列表来存储所有的mask_rotate
    # 遍历每个子图的mask_rotate
    mask_rotate = data_batch.mask_rotate
    # 计算总行数
    total_rows = sum(mask.shape[0] for mask in mask_rotate)

    # 初始化最终的mask_rotate矩阵
    full_mask_rotate = np.zeros((total_rows, total_cols), dtype=bool)

    # 填充full_mask_rotate
    current_row = 0
    current_col = 0
    for mask in data_batch.mask_rotate:
        rows, cols = mask.shape
        full_mask_rotate[current_row:current_row+rows, current_col:current_col+cols] = mask
        current_row += rows
        current_col += cols
    return full_mask_rotate

def fragmentation(data, rotate_idx, rotate_idx_sel, z=20):
    augmented_data = copy.deepcopy(data)
    G = to_networkx(augmented_data, to_undirected=False)
    G2 = G.to_undirected()
    if len(rotate_idx_sel) < 2:
        data_batch = Batch.from_data_list([data])
        data_batch.mask_rotate = combine_mask_rotate(data_batch)
        return None
    # 如果rotate_idx的长度大于3，随机选择K在1到len(rotate_idx)-1之间
    if len(rotate_idx_sel) > 1:
        K = np.random.randint(1, min(len(rotate_idx_sel),5))
    remove_edges_idx = np.random.choice(rotate_idx_sel, K, replace=False)
    rest_edges_idx = np.setdiff1d(rotate_idx, remove_edges_idx)
    iedge_list = augmented_data.edge_index.T.numpy()
    iedge_attr = augmented_data.edge_attr
    remove_edges = copy.deepcopy(iedge_list)[remove_edges_idx]
    rest_edges = copy.deepcopy(iedge_list)[rest_edges_idx]
    for i in range(len(remove_edges)):
        G2.remove_edge(*remove_edges[i])
    modified_array = np.where(remove_edges_idx % 2 == 0, remove_edges_idx + 1, remove_edges_idx - 1)
    remove_edges_idx = np.concatenate((remove_edges_idx, modified_array))
    edge_list_after_removal = np.delete(iedge_list, remove_edges_idx, axis=0)
    # edge_attr_after_removal = np.delete(iedge_attr, remove_edges_idx, axis=0)
    mask = torch.ones(iedge_attr.size(0), dtype=torch.bool)
    mask[remove_edges_idx] = False
    edge_attr_after_removal = iedge_attr[mask]
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
        # new_edge_attr = iedge_attr[edge_indices]
        new_edge_attr = edge_attr_after_removal[edge_indices]
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
        pos = [pos[component_nodes] for pos in data.pos]
        # w = [data.weights[max_weight_index]]
        w = data.weights
        # z = [data.z[i] for i in component_nodes]
        
        # 创建新的 data 对象
        new_data = Data(
            x= data.x[component_nodes],  # 提取连通分量中的节点特征
            edge_index=torch.tensor(new_edge_index, dtype=torch.long),  # 设置新的边索引
            edge_attr=new_edge_attr,  # 设置新的边属性
            z= data.z[component_nodes],  # 提取连通分量中的z属性
            pos = pos,
            weights = w,
            canonical_smi = data.canonical_smi,
            mol = data.mol,
            edge_mask=torch.tensor(new_edge_mask),  # 设置新的边掩码
            mask_rotate=new_mask_rotate  # 设置新的旋转掩码
        )
        
        # 将新创建的 data 对象添加到列表中
        if(len(new_data.z)>z):
            new_data_list.append(new_data)
    if len(new_data_list)>0:
        return new_data_list
        # data_batch = Batch.from_data_list([i for i in new_data_list])
        # data_batch.mask_rotate = combine_mask_rotate(data_batch)
        # return data_batch
    else:
        return None

import pickle
from tqdm import tqdm

# input_dir = '/path/to/DRUGS/fragdiff_dump'
# output_dir_base = os.path.join(input_dir, '_frag')
# os.makedirs(output_dir_base, exist_ok=True)  # 确保输出目录存在

# # 正则表达式用于匹配文件名
# pattern = re.compile(r'^train_samples_(\d+)\.pkl$')

# # 遍历目录中的所有文件
# for filename in os.listdir(input_dir):
#     match = pattern.match(filename)
#     if match:
#         file_index = match.group(1)
#         input_file_path = os.path.join(input_dir, filename)
#         output_file_path = os.path.join(output_dir_base, filename)

#         with open(input_file_path, 'rb') as f:
#             datapoints = pickle.load(f)

#         frag_list = []
#         for data in tqdm(datapoints, desc=f'Processing {filename}'):
#             edge_mask, mask_rotate, conn_rotate_idx = get_transformation_mask_old(data)
#             rotate_idx = conn_rotate_idx
#             data.edge_mask = torch.tensor(edge_mask)
#             data.mask_rotate = mask_rotate
#             data_frag = fragmentation(data, conn_rotate_idx, rotate_idx, 10)
#             if data_frag:
#                 # frag_list.append(data_frag)
#                 frag_list.extend(data_frag)

#         # 将处理后的数据写入新的pkl文件
#         with open(output_file_path, 'wb') as f:
#             pickle.dump(frag_list, f)

# print('Fragmentation processing completed.')
# --------------------------------- all
# TODO no brics/recap br aug frag
import os.path as osp
noaug = 'braug'
output_dir = f'/path/to/DRUGS/fragdiff_dump/_frag_{noaug}/'
os.makedirs(output_dir, exist_ok=True)
output_file_path = output_dir + 'frag_train_data.pkl'

merge_file_path = '/path/to/DRUGS/fragdiff_dump/merged_train_data.pkl'
with open(merge_file_path, 'rb') as f:
    datapoints = pickle.load(f)

frag_list = []
for data in tqdm(datapoints):
    edge_mask, mask_rotate, conn_rotate_idx = get_transformation_mask_old(data)
    if noaug:
        rotate_idx = chemfrag_bondidx(data, edge_mask, noaug)
        rotate_idx = [item for item in conn_rotate_idx if item not in rotate_idx]
    else:
        rotate_idx = conn_rotate_idx
    data.edge_mask = torch.tensor(edge_mask)
    data.mask_rotate = mask_rotate
    data_frag = fragmentation(data, conn_rotate_idx, rotate_idx, 10)
    if data_frag:
        # frag_list.append(data_frag)
        frag_list.extend(data_frag)

with open(output_file_path, 'wb') as f:
    pickle.dump(frag_list, f)
