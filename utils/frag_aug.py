import networkx as nx
import numpy as np
import torch, copy
from scipy.spatial.transform import Rotation as R
from torch_geometric.utils import to_networkx
from torch_geometric.data import Data
from utils.brics_rec import FindBRICSBonds
from utils.recap_rec import FindRECAPBonds, FindBRBonds
from utils.junc_dec import FindJuncBonds, get_non_ring_edges_and_atoms
from utils.connect import ConnectBonds
from torch_geometric.data import Data, Batch

def chemfrag_bondidx_2(data, edge_mask , dec):
    bond_idx_list = []
    if 'raug' in dec:
        chem_bonds_list = FindRECAPBonds(data.mol)
    if 'baug' in dec:
        chem_bonds_list = FindBRICSBonds(data.mol)
    # if dec == 'bra':
    if 'braug' in dec:
        chem_bonds_list = FindBRBonds(data.mol)
    edges_list = data.edge_index.T.tolist()
    for bond in chem_bonds_list:
        try:
            bond_idx = edges_list.index(list(bond))
            if edge_mask[bond_idx]:
                bond_idx_list.append(bond_idx)
        except ValueError:
            # bond may not in edges list(for example RECAP)
            # print(f"Bond {bond} not found in edges_list.")
            continue

def chemfrag_bondidx(data, edge_mask , dec):
    bond_idx_list = []
    if 'recap' in dec:
        chem_bonds_list = FindRECAPBonds(data.mol)
    if 'brics' in dec:
        chem_bonds_list = FindBRICSBonds(data.mol)
    if dec == 'braug':
        chem_bonds_list = FindBRBonds(data.mol)
    edges_list = data.edge_index.T.tolist()
    for bond in chem_bonds_list:
        try:
            bond_idx = edges_list.index(list(bond))
            if edge_mask[bond_idx]:
                bond_idx_list.append(bond_idx)
        except ValueError:
            # bond may not in edges list(for example RECAP)
            # print(f"Bond {bond} not found in edges_list.")
            continue
    
    return bond_idx_list
    # if self.dec == 'recap':
    #         edge_mask, mask_rotate, rotate_idx = get_transformation_mask_recap(data)
    #     if self.dec == 'brics':
    #         edge_mask, mask_rotate, rotate_idx = get_transformation_mask_brics(data)
    #     if self.dec == 'br':
    #         edge_mask, mask_rotate, rotate_idx = get_transformation_mask_bnr(data)


def frag_aug(data, rotate_idx):
    augmented_data = copy.deepcopy(data)
    G = to_networkx(augmented_data, to_undirected=False)
    G2 = G.to_undirected()
    if len(rotate_idx) < 2:
        return data, data
    # 如果rotate_idx的长度大于3，随机选择K在1到len(rotate_idx)-1之间
    if len(rotate_idx) > 1:
        K = np.random.randint(1, len(rotate_idx))
    remove_edges_idx = np.random.choice(rotate_idx, K, replace=False)
    edge_list = augmented_data.edge_index.T.numpy()
    remove_edges = copy.deepcopy(augmented_data.edge_index.T.numpy())[remove_edges_idx]
    for i in range(len(remove_edges)):
        G2.remove_edge(*remove_edges[i])
    modified_array = np.where(remove_edges_idx % 2 == 0, remove_edges_idx + 1, remove_edges_idx - 1)
# 扩展数组
    remove_edges_idx = np.concatenate((remove_edges_idx, modified_array))
    edge_list_after_removal = np.delete(edge_list, remove_edges_idx, axis=0)
    iedge_list = edge_list_after_removal.tolist()
    mask_edges = np.zeros( len(iedge_list), dtype=bool)
    rotate_idx = []
    to_rotate = []
    for i in range(len(iedge_list)):
        if G2.has_edge(*iedge_list[i]):
            G2.remove_edge(*iedge_list[i])
            n_u = iedge_list[i][0]
            n_v = iedge_list[i][1]
            connected_components = nx.connected_components(G2)
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
                l = list(list(nx.connected_components(G2))[n_v_component_index])
                to_rotate.append(l)
                # mask_edges.append(rotate_index)
            else:
                rotate_index = iedge_list.index([n_v, n_u])
                rotate_idx.append(rotate_index)
                mask_edges[rotate_index] = True
                l = list(list(nx.connected_components(G2))[n_u_component_index])
                to_rotate.append(l)
    mask_rotate = np.zeros((np.sum(mask_edges), len(G.nodes())), dtype=bool)
    idx = 0
    # for e in mask_edges:
    #     mask_rotate[idx][np.asarray(to_rotate[idx], dtype=int)] = True
    #     idx += 1 # reorder_
    for i in range( len(iedge_list) ):
        if mask_edges[i]:
            mask_rotate[idx][np.asarray(to_rotate[idx], dtype=int)] = True
            # print(mask_rotate[idx][np.asarray(to_rotate[idx], dtype=int)])
            idx += 1
    sorted_indices = sorted(range(len(rotate_idx)), key=lambda x: rotate_idx[x])
    # decompose的边顺序(按照每条规则match而不是纯按edge顺序的match)需要和edge_idex里面的顺序对齐
    mask_rotate = np.array([mask_rotate[i] for i in sorted_indices], dtype=bool)
    augmented_data.edge_mask = torch.tensor(mask_edges)
    augmented_data.mask_rotate = mask_rotate
    edge_index = augmented_data.edge_index
    edge_attr = augmented_data.edge_attr

    # 移除指定索引的边
    mask = torch.ones(edge_index.size(1), dtype=torch.bool)
    mask[remove_edges_idx] = False

    # 更新edge_index和edge_attr
    new_edge_index = edge_index[:, mask]
    new_edge_attr = edge_attr[mask]
    augmented_data.edge_index = new_edge_index
    augmented_data.edge_attr = new_edge_attr
    data1 = data    # 第二个数据对象
    data2 = augmented_data  # 第一个数据对象
#     atoms = data1.mask_rotate.shape[1]
#     mask_rotate2_expanded = np.zeros((data2.mask_rotate.shape[0], atoms * 2), dtype=bool)
#     mask_rotate2_expanded[:, atoms:] = data2.mask_rotate
#     # 扩展第一个图的mask_rotate
#     mask_rotate1_expanded = np.hstack([data1.mask_rotate, np.zeros((data1.mask_rotate.shape[0], atoms), dtype=bool)])

#     # 合并两个mask_rotate
#     mask_rotate = np.vstack([mask_rotate1_expanded, mask_rotate2_expanded])
#     x = torch.cat([data1.x, data2.x], dim=0)
# # 合并边索引
#     edge_index2 = data2.edge_index + data1.x.size(0)
#     edge_index = torch.cat([data1.edge_index, edge_index2], dim=1)
#     # 合并边特征
#     edge_attr = torch.cat([data1.edge_attr, data2.edge_attr], dim=0)
#     pos = [torch.cat([data1.pos[0], data2.pos[0]], dim=0)]
#     z =  torch.cat([data1.z, data1.z], dim=0)
#     edge_mask = torch.cat([data1.edge_mask, data2.edge_mask], dim=0)
#     atoms = data1.mask_rotate.shape[1]
#     mask_rotate2_expanded = np.zeros((data2.mask_rotate.shape[0], atoms * 2), dtype=bool)
#     mask_rotate2_expanded[:, atoms:] = data2.mask_rotate
#     # 扩展第一个图的mask_rotate
#     mask_rotate1_expanded = np.hstack([data1.mask_rotate, np.zeros((data1.mask_rotate.shape[0], atoms), dtype=bool)])
#     # 合并两个mask_rotate
#     mask_rotate = np.vstack([mask_rotate1_expanded, mask_rotate2_expanded])
#     # 创建新的Data对象
#     merged_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, z = z,canonical_smi=data1.canonical_smi, mol=data1.mol ,pos= pos, weights= data1.weights, edge_mask=edge_mask, mask_rotate = mask_rotate)
    return data, augmented_data


def frag_aug_conn(data, rotate_idx):
    augmented_data = copy.deepcopy(data)
    G = to_networkx(augmented_data, to_undirected=False)
    G2 = G.to_undirected()
    if len(rotate_idx) < 2:
        return data, data
    # 如果rotate_idx的长度大于3，随机选择K在1到len(rotate_idx)-1之间
    if len(rotate_idx) > 1:
        K = np.random.randint(1, len(rotate_idx))
    remove_edges_idx = np.random.choice(rotate_idx, K, replace=False)
    edge_list = augmented_data.edge_index.T.numpy()
    remove_edges = copy.deepcopy(augmented_data.edge_index.T.numpy())[remove_edges_idx]
    for i in range(len(remove_edges)):
        G2.remove_edge(*remove_edges[i])
        # 有向图两条边douyaoremove
    modified_array = np.where(remove_edges_idx % 2 == 0, remove_edges_idx + 1, remove_edges_idx - 1)
# 扩展数组
    remove_edges_idx = np.concatenate((remove_edges_idx, modified_array))
    edge_list_after_removal = np.delete(edge_list, remove_edges_idx, axis=0)
    iedge_list = edge_list_after_removal.tolist()
    mask_edges = np.zeros( len(iedge_list), dtype=bool)
    rotate_idx = []
    to_rotate = []
    G3 = copy.deepcopy(G2)
    for i in range(len(iedge_list)):
        if G3.has_edge(*iedge_list[i]):
            G3.remove_edge(*iedge_list[i])
            n_u = iedge_list[i][0]
            n_v = iedge_list[i][1]
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
    mask_rotate = np.zeros((np.sum(mask_edges), len(G.nodes())), dtype=bool)
    idx = 0
    # for e in mask_edges:
    #     mask_rotate[idx][np.asarray(to_rotate[idx], dtype=int)] = True
    #     idx += 1 # reorder_
    for i in range( len(iedge_list) ):
        if mask_edges[i]:
            mask_rotate[idx][np.asarray(to_rotate[idx], dtype=int)] = True
            # print(mask_rotate[idx][np.asarray(to_rotate[idx], dtype=int)])
            idx += 1
    sorted_indices = sorted(range(len(rotate_idx)), key=lambda x: rotate_idx[x])
    # decompose的边顺序(按照每条规则match而不是纯按edge顺序的match)需要和edge_idex里面的顺序对齐
    mask_rotate = np.array([mask_rotate[i] for i in sorted_indices], dtype=bool)
    augmented_data.edge_mask = torch.tensor(mask_edges)
    augmented_data.mask_rotate = mask_rotate
    edge_index = augmented_data.edge_index
    edge_attr = augmented_data.edge_attr

    # 移除指定索引的边
    mask = torch.ones(edge_index.size(1), dtype=torch.bool)
    mask[remove_edges_idx] = False

    # 更新edge_index和edge_attr
    new_edge_index = edge_index[:, mask]
    new_edge_attr = edge_attr[mask]
    augmented_data.edge_index = new_edge_index
    augmented_data.edge_attr = new_edge_attr
    data1 = data    # 第二个数据对象
    data2 = augmented_data  # 第一个数据对象
    # data merge
#     atoms = data1.mask_rotate.shape[1]
#     mask_rotate2_expanded = np.zeros((data2.mask_rotate.shape[0], atoms * 2), dtype=bool)
#     mask_rotate2_expanded[:, atoms:] = data2.mask_rotate
#     # 扩展第一个图的mask_rotate
#     mask_rotate1_expanded = np.hstack([data1.mask_rotate, np.zeros((data1.mask_rotate.shape[0], atoms), dtype=bool)])

#     # 合并两个mask_rotate
#     mask_rotate = np.vstack([mask_rotate1_expanded, mask_rotate2_expanded])
#     x = torch.cat([data1.x, data2.x], dim=0)
# # 合并边索引
#     edge_index2 = data2.edge_index + data1.x.size(0)
#     edge_index = torch.cat([data1.edge_index, edge_index2], dim=1)
#     # 合并边特征
#     edge_attr = torch.cat([data1.edge_attr, data2.edge_attr], dim=0)
#     pos = [torch.cat([data1.pos[0], data2.pos[0]], dim=0)]
#     z =  torch.cat([data1.z, data1.z], dim=0)
#     edge_mask = torch.cat([data1.edge_mask, data2.edge_mask], dim=0)
#     atoms = data1.mask_rotate.shape[1]
#     mask_rotate2_expanded = np.zeros((data2.mask_rotate.shape[0], atoms * 2), dtype=bool)
#     mask_rotate2_expanded[:, atoms:] = data2.mask_rotate
#     # 扩展第一个图的mask_rotate
#     mask_rotate1_expanded = np.hstack([data1.mask_rotate, np.zeros((data1.mask_rotate.shape[0], atoms), dtype=bool)])
#     # 合并两个mask_rotate
#     mask_rotate = np.vstack([mask_rotate1_expanded, mask_rotate2_expanded])
#     # 创建新的Data对象
#     merged_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, z = z,canonical_smi=data1.canonical_smi, mol=data1.mol ,pos= pos, weights= data1.weights, edge_mask=edge_mask, mask_rotate = mask_rotate)
    return data, augmented_data

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
# 假设data_batch是已经合并的DataBatch对象
# combined_mask_rotate = combine_mask_rotate(data_batch)

# TODO generate pretrain dataset
def fragmentation(data, rotate_idx, rotate_idx_sel, z=20):
    augmented_data = copy.deepcopy(data)
    G = to_networkx(augmented_data, to_undirected=False)
    G2 = G.to_undirected()
    if len(rotate_idx_sel) < 2:
        data_batch = Batch.from_data_list([data])
        data_batch.mask_rotate = combine_mask_rotate(data_batch)
        return data_batch
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
        data_batch = Batch.from_data_list([i for i in new_data_list] + [data])
        data_batch.mask_rotate = combine_mask_rotate(data_batch)
        return data_batch
    else:
        data_batch = Batch.from_data_list([data])
        data_batch.mask_rotate = combine_mask_rotate(data_batch)
        return data_batch
    return new_data_list



    # data merge
#     atoms = data1.mask_rotate.shape[1]
#     mask_rotate2_expanded = np.zeros((data2.mask_rotate.shape[0], atoms * 2), dtype=bool)
#     mask_rotate2_expanded[:, atoms:] = data2.mask_rotate
#     # 扩展第一个图的mask_rotate
#     mask_rotate1_expanded = np.hstack([data1.mask_rotate, np.zeros((data1.mask_rotate.shape[0], atoms), dtype=bool)])

#     # 合并两个mask_rotate
#     mask_rotate = np.vstack([mask_rotate1_expanded, mask_rotate2_expanded])
#     x = torch.cat([data1.x, data2.x], dim=0)
# # 合并边索引
#     edge_index2 = data2.edge_index + data1.x.size(0)
#     edge_index = torch.cat([data1.edge_index, edge_index2], dim=1)
#     # 合并边特征
#     edge_attr = torch.cat([data1.edge_attr, data2.edge_attr], dim=0)
#     pos = [torch.cat([data1.pos[0], data2.pos[0]], dim=0)]
#     z =  torch.cat([data1.z, data1.z], dim=0)
#     edge_mask = torch.cat([data1.edge_mask, data2.edge_mask], dim=0)
#     atoms = data1.mask_rotate.shape[1]
#     mask_rotate2_expanded = np.zeros((data2.mask_rotate.shape[0], atoms * 2), dtype=bool)
#     mask_rotate2_expanded[:, atoms:] = data2.mask_rotate
#     # 扩展第一个图的mask_rotate
#     mask_rotate1_expanded = np.hstack([data1.mask_rotate, np.zeros((data1.mask_rotate.shape[0], atoms), dtype=bool)])
#     # 合并两个mask_rotate
#     mask_rotate = np.vstack([mask_rotate1_expanded, mask_rotate2_expanded])
#     # 创建新的Data对象
#     merged_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, z = z,canonical_smi=data1.canonical_smi, mol=data1.mol ,pos= pos, weights= data1.weights, edge_mask=edge_mask, mask_rotate = mask_rotate)
    return data, augmented_data