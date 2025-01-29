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


def get_transformation_mask_bnr(pyg_data):
    mol = pyg_data.mol
    G = to_networkx(pyg_data, to_undirected=False)
    G2 = G.to_undirected()
    iedge_list = pyg_data.edge_index.T.tolist()
    mask_edges = np.zeros( pyg_data.edge_index.T.numpy().shape[0], dtype=bool)
    # mask_edges = []
    edges_list = FindBRBonds(mol)
    rotate_idx = []
    to_rotate = []
    # print(edges_list,len(edges_list))
    # print(*edges_list[0])
    for i in range(len(edges_list)):
        if G2.has_edge(*edges_list[i]):
            G2.remove_edge(*edges_list[i])
            n_u = edges_list[i][0]
            n_v = edges_list[i][1]
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
                # mask_edges.append(rotate_index)
    mask_rotate = np.zeros((np.sum(mask_edges), len(G.nodes())), dtype=bool)
    idx = 0
    # for e in mask_edges:
    #     mask_rotate[idx][np.asarray(to_rotate[idx], dtype=int)] = True
    #     idx += 1 # reorder_
    for i in range( len(G.edges()) ):
        if mask_edges[i]:
            mask_rotate[idx][np.asarray(to_rotate[idx], dtype=int)] = True
            idx += 1
    sorted_indices = sorted(range(len(rotate_idx)), key=lambda x: rotate_idx[x])
    # decompose的边顺序(按照每条规则match而不是纯按edge顺序的match)需要和edge_idex里面的顺序对齐
    mask_rotate = np.array([mask_rotate[i] for i in sorted_indices], dtype=bool)
    return mask_edges, mask_rotate, rotate_idx

def get_transformation_mask_recap(pyg_data):
    mol = pyg_data.mol
    G = to_networkx(pyg_data, to_undirected=False)
    G2 = G.to_undirected()
    iedge_list = pyg_data.edge_index.T.tolist()
    mask_edges = np.zeros( pyg_data.edge_index.T.numpy().shape[0], dtype=bool)
    # mask_edges = []
    edges_list = FindRECAPBonds(mol)
    rotate_idx = []
    to_rotate = []
    # print(edges_list,len(edges_list))
    # print(*edges_list[0])
    for i in range(len(edges_list)):
        if G2.has_edge(*edges_list[i]):
            G2.remove_edge(*edges_list[i])
            n_u = edges_list[i][0]
            n_v = edges_list[i][1]
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
                # mask_edges.append(rotate_index)
    mask_rotate = np.zeros((np.sum(mask_edges), len(G.nodes())), dtype=bool)
    idx = 0
    # for e in mask_edges:
    #     mask_rotate[idx][np.asarray(to_rotate[idx], dtype=int)] = True
    #     idx += 1 # reorder_
    for i in range( len(G.edges()) ):
        if mask_edges[i]:
            mask_rotate[idx][np.asarray(to_rotate[idx], dtype=int)] = True
            idx += 1
    sorted_indices = sorted(range(len(rotate_idx)), key=lambda x: rotate_idx[x])
    # decompose的边顺序(按照每条规则match而不是纯按edge顺序的match)需要和edge_idex里面的顺序对齐
    mask_rotate = np.array([mask_rotate[i] for i in sorted_indices], dtype=bool)
    return mask_edges, mask_rotate, rotate_idx

def get_transformation_mask_brics(pyg_data):
    mol = pyg_data.mol
    G = to_networkx(pyg_data, to_undirected=False)
    G2 = G.to_undirected()
    iedge_list = pyg_data.edge_index.T.tolist()
    mask_edges = np.zeros( pyg_data.edge_index.T.numpy().shape[0], dtype=bool)
    # mask_edges = []
    edges_list = FindBRICSBonds(mol)
    rotate_idx = []
    to_rotate = []
    
    for i in range(len(edges_list)):
        G2.remove_edge(*edges_list[i])
        n_u = edges_list[i][0]
        n_v = edges_list[i][1]
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
            # mask_edges.append(rotate_index)
    mask_rotate = np.zeros((np.sum(mask_edges), len(G.nodes())), dtype=bool)
    idx = 0
    # for e in mask_edges:
    #     mask_rotate[idx][np.asarray(to_rotate[idx], dtype=int)] = True
    #     idx += 1 # reorder_
    for i in range( len(G.edges()) ):
        if mask_edges[i]:
            mask_rotate[idx][np.asarray(to_rotate[idx], dtype=int)] = True
            idx += 1
    sorted_indices = sorted(range(len(rotate_idx)), key=lambda x: rotate_idx[x])
    # decompose的边顺序(按照每条规则match而不是纯按edge顺序的match)需要和edge_idex里面的顺序对齐
    mask_rotate = np.array([mask_rotate[i] for i in sorted_indices], dtype=bool)
    return mask_edges, mask_rotate, rotate_idx

def get_transformation_mask(pyg_data):
    G = to_networkx(pyg_data, to_undirected=False)
    mask_edges = []
    to_rotate = []
    edges = pyg_data.edge_index.T.numpy()
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
                else:
                    to_rotate.append(l)
                    to_rotate.append([])
                continue
        to_rotate.append([])
        to_rotate.append([])
    for i in range(len(to_rotate)):
        if len(to_rotate[i]) == 0:
            continue
        else:
            mask_edges.append(i)
    # mask_edges = np.asarray([0 if len(l) == 0 else 1 for l in to_rotate], dtype=bool)
    mask_rotate = np.zeros((len(mask_edges), len(G.nodes())), dtype=bool)
    idx = 0
    for e in mask_edges:
        mask_rotate[idx][np.asarray(to_rotate[e], dtype=int)] = True
        idx += 1

    return mask_edges, mask_rotate, rotate_idx

def get_transformation_mask_frag(pyg_data):
    mol = pyg_data.mol
    G = to_networkx(pyg_data, to_undirected=False)
    G2 = G.to_undirected()
    iedge_list = pyg_data.edge_index.T.tolist()
    mask_edges = np.zeros( pyg_data.edge_index.T.numpy().shape[0], dtype=bool)
    # mask_edges = []
    edges_list = ConnectBonds(pyg_data)
    rotate_idx = []
    to_rotate = []
    
    for i in range(len(edges_list)):
        if G2.has_edge(*edges_list[i]):
            G2.remove_edge(*edges_list[i])
            n_u = edges_list[i][0]
            n_v = edges_list[i][1]
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
                # mask_edges.append(rotate_index)
    mask_rotate = np.zeros((np.sum(mask_edges), len(G.nodes())), dtype=bool)
    idx = 0
    # for e in mask_edges:
    #     mask_rotate[idx][np.asarray(to_rotate[idx], dtype=int)] = True
    #     idx += 1 # reorder_
    for i in range( len(G.edges()) ):
        if mask_edges[i]:
            mask_rotate[idx][np.asarray(to_rotate[idx], dtype=int)] = True
            idx += 1
    sorted_indices = sorted(range(len(rotate_idx)), key=lambda x: rotate_idx[x])
    # decompose的边顺序(按照每条规则match而不是纯按edge顺序的match)需要和edge_idex里面的顺序对齐
    mask_rotate = np.array([mask_rotate[i] for i in sorted_indices], dtype=bool)
    return mask_edges, mask_rotate, rotate_idx


def get_transformation_mask_junc(pyg_data):
    mol = pyg_data.mol
    G = to_networkx(pyg_data, to_undirected=False)
    G2 = G.to_undirected()
    iedge_list = pyg_data.edge_index.T.tolist()
    ring_mask_edges, ring_mask_rotate, ring_rotate_index = FindJuncBonds(pyg_data, iedge_list, G2)
    mask_edges = np.zeros( pyg_data.edge_index.T.numpy().shape[0], dtype=bool)
    # mask_edges = []
    # edges_list = copy.deepcopy(pyg_data.edge_index.T.numpy())
    _, edges_list = get_non_ring_edges_and_atoms(mol)
    # random or not?
    edges_list = edges_list.reshape(-1,2,2)
    np.random.shuffle(edges_list)
    edges_list = edges_list.reshape(-1,2)
    rotate_idx = []
    to_rotate = []
    
    for i in range(len(edges_list)):
        if G2.has_edge(*edges_list[i]):
            G2.remove_edge(*edges_list[i])
            n_u = edges_list[i][0]
            n_v = edges_list[i][1]
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
                # mask_edges.append(rotate_index)
    mask_rotate = np.zeros((np.sum(mask_edges), len(G.nodes())), dtype=bool)
    idx = 0
    # for e in mask_edges:
    #     mask_rotate[idx][np.asarray(to_rotate[idx], dtype=int)] = True
    #     idx += 1 # reorder_
    for i in range( len(G.edges()) ):
        if mask_edges[i]:
            mask_rotate[idx][np.asarray(to_rotate[idx], dtype=int)] = True
            idx += 1

    if ring_rotate_index is not None:
        t_rotate_idx = rotate_idx + ring_rotate_index
        mask_rotate = np.concatenate((mask_rotate, ring_mask_rotate))
    else:
        t_rotate_idx = rotate_idx
    # print(mask_rotate)
    # print(ring_mask_rotate)
    # sys.exit(0)
    sorted_indices = sorted(range(len(t_rotate_idx)), key=lambda x: t_rotate_idx[x])
    # decompose的边顺序(按照每条规则match而不是纯按edge顺序的match)需要和edge_idex里面的顺序对齐
    mask_rotate = np.array([mask_rotate[i] for i in sorted_indices], dtype=bool)
    mask_edges = mask_edges + ring_mask_edges
    # print(mask_rotate)
    return mask_edges, mask_rotate, rotate_idx

def get_transformation_mask_junc_brics(pyg_data):
    mol = pyg_data.mol
    G = to_networkx(pyg_data, to_undirected=False)
    G2 = G.to_undirected()
    iedge_list = pyg_data.edge_index.T.tolist()
    ring_mask_edges, ring_mask_rotate, ring_rotate_index = FindJuncBonds(pyg_data, iedge_list, G2)
    mask_edges = np.zeros( pyg_data.edge_index.T.numpy().shape[0], dtype=bool)
    # mask_edges = []
    edges_list = FindBRICSBonds(mol)
    if len(edges_list) == 0:
        edges_list = ConnectBonds(pyg_data)
    rotate_idx = []
    to_rotate = []
    
    for i in range(len(edges_list)):
        G2.remove_edge(*edges_list[i])
        n_u = edges_list[i][0]
        n_v = edges_list[i][1]
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
            # mask_edges.append(rotate_index)
    mask_rotate = np.zeros((np.sum(mask_edges), len(G.nodes())), dtype=bool)
    idx = 0
    # for e in mask_edges:
    #     mask_rotate[idx][np.asarray(to_rotate[idx], dtype=int)] = True
    #     idx += 1 # reorder_
    for i in range( len(G.edges()) ):
        if mask_edges[i]:
            mask_rotate[idx][np.asarray(to_rotate[idx], dtype=int)] = True
            idx += 1
    if ring_rotate_index is not None:
        t_rotate_idx = rotate_idx + ring_rotate_index
        mask_rotate = np.concatenate((mask_rotate, ring_mask_rotate))
    else:
        t_rotate_idx = rotate_idx
    sorted_indices = sorted(range(len(t_rotate_idx)), key=lambda x: t_rotate_idx[x])
    # decompose的边顺序(按照每条规则match而不是纯按edge顺序的match)需要和edge_idex里面的顺序对齐
    mask_edges = mask_edges + ring_mask_edges
    mask_rotate = np.array([mask_rotate[i] for i in sorted_indices], dtype=bool)
    return mask_edges, mask_rotate, rotate_idx

def get_transformation_mask_junc_recap(pyg_data):
    mol = pyg_data.mol
    G = to_networkx(pyg_data, to_undirected=False)
    G2 = G.to_undirected()
    iedge_list = pyg_data.edge_index.T.tolist()
    ring_mask_edges, ring_mask_rotate, ring_rotate_index = FindJuncBonds(pyg_data, iedge_list, G2)
    # print(ring_mask_rotate, ring_rotate_index)
    mask_edges = np.zeros( pyg_data.edge_index.T.numpy().shape[0], dtype=bool)
    # mask_edges = []
    edges_list = FindRECAPBonds(mol)
    if len(edges_list) == 0:
        edges_list = ConnectBonds(pyg_data)
    rotate_idx = []
    to_rotate = []
    
    for i in range(len(edges_list)):
        if G2.has_edge(*edges_list[i]):
            G2.remove_edge(*edges_list[i])
            n_u = edges_list[i][0]
            n_v = edges_list[i][1]
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
                # mask_edges.append(rotate_index)
    mask_rotate = np.zeros((np.sum(mask_edges), len(G.nodes())), dtype=bool)
    idx = 0
    # for e in mask_edges:
    #     mask_rotate[idx][np.asarray(to_rotate[idx], dtype=int)] = True
    #     idx += 1 # reorder_
    for i in range( len(G.edges()) ):
        if mask_edges[i]:
            mask_rotate[idx][np.asarray(to_rotate[idx], dtype=int)] = True
            idx += 1
    if ring_rotate_index is not None:
        t_rotate_idx = rotate_idx + ring_rotate_index
        mask_rotate = np.concatenate((mask_rotate, ring_mask_rotate))
    else:
        t_rotate_idx = rotate_idx
    # rotate_idx = rotate_idx + ring_rotate_index
    # mask_rotate = np.concatenate((mask_rotate, ring_mask_rotate))
    sorted_indices = sorted(range(len(t_rotate_idx)), key=lambda x: t_rotate_idx[x])
    # decompose的边顺序(按照每条规则match而不是纯按edge顺序的match)需要和edge_idex里面的顺序对齐
    mask_edges = mask_edges + ring_mask_edges
    mask_rotate = np.array([mask_rotate[i] for i in sorted_indices], dtype=bool)
    return mask_edges, mask_rotate, rotate_idx

def get_transformation_mask_all(pyg_data):
    mol = pyg_data.mol
    G = to_networkx(pyg_data, to_undirected=False)
    G2 = G.to_undirected()
    iedge_list = pyg_data.edge_index.T.tolist()
    mask_edges = np.zeros( pyg_data.edge_index.T.numpy().shape[0], dtype=bool)
    # mask_edges = []
    edges_list = copy.deepcopy(pyg_data.edge_index.T.numpy())
    # random or not?
    edges_list = edges_list.reshape(-1,2,2)
    np.random.shuffle(edges_list)
    edges_list = edges_list.reshape(-1,2)
    rotate_idx = []
    to_rotate = []
    
    for i in range(len(edges_list)):
        if G2.has_edge(*edges_list[i]):
            G2.remove_edge(*edges_list[i])
            n_u = edges_list[i][0]
            n_v = edges_list[i][1]
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
                # mask_edges.append(rotate_index)
    mask_rotate = np.zeros((np.sum(mask_edges), len(G.nodes())), dtype=bool)
    idx = 0
    # for e in mask_edges:
    #     mask_rotate[idx][np.asarray(to_rotate[idx], dtype=int)] = True
    #     idx += 1 # reorder_
    for i in range( len(G.edges()) ):
        if mask_edges[i]:
            mask_rotate[idx][np.asarray(to_rotate[idx], dtype=int)] = True
            idx += 1
    sorted_indices = sorted(range(len(rotate_idx)), key=lambda x: rotate_idx[x])
    # decompose的边顺序(按照每条规则match而不是纯按edge顺序的match)需要和edge_idex里面的顺序对齐
    mask_rotate = np.array([mask_rotate[i] for i in sorted_indices], dtype=bool)
    return mask_edges, mask_rotate, rotate_idx

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

def get_transformation_mask_junc_old(pyg_data):
    G = to_networkx(pyg_data, to_undirected=False)
    to_rotate = []
    rotate_idx = []
    edges = pyg_data.edge_index.T.numpy()
    iedge_list = pyg_data.edge_index.T.tolist()
    G2 = G.to_undirected()
    ring_mask_edges, ring_mask_rotate, ring_rotate_index = FindJuncBonds(pyg_data, iedge_list, G2)
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
                    # rotate_idx.append([edges[i, 1], edges[i, 0]])
                    rotate_idx.append(i+1)
                else:
                    to_rotate.append(l)
                    to_rotate.append([])
                    # rotate_idx.append([edges[i, 0], edges[i, 1]])
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
    if ring_rotate_index is not None:
        t_rotate_idx = rotate_idx + ring_rotate_index
        mask_rotate = np.concatenate((mask_rotate, ring_mask_rotate))
    else:
        t_rotate_idx = rotate_idx
    sorted_indices = sorted(range(len(t_rotate_idx)), key=lambda x: t_rotate_idx[x])
    # decompose的边顺序(按照每条规则match而不是纯按edge顺序的match)需要和edge_idex里面的顺序对齐
    mask_edges = mask_edges + ring_mask_edges
    mask_rotate = np.array([mask_rotate[i] for i in sorted_indices], dtype=bool)
    return mask_edges, mask_rotate, rotate_idx


def get_distance_matrix(pyg_data, mask_edges, mask_rotate):
    G = to_networkx(pyg_data, to_undirected=False)
    N = G.number_of_nodes()
    edge_distances = []
    for i, e in enumerate(pyg_data.edge_index.T.numpy()[mask_edges]):
        v = e[1]
        d = nx.shortest_path_length(G, source=v)
        d = np.asarray([d[j] for j in range(N)])
        d = d - 1 + mask_rotate[i]
        edge_distances.append(d)

    edge_distances = np.asarray(edge_distances)
    return edge_distances


def modify_conformer(pos, edge_index, mask_rotate, torsion_updates, as_numpy=False, no_junc = True):
    if type(pos) != np.ndarray: pos = pos.cpu().numpy()
    for idx_edge, e in enumerate(edge_index.cpu().numpy()):
        if torsion_updates[idx_edge] == 0:
            continue
        u, v = e[0], e[1]

        # check if need to reverse the edge, v should be connected to the part that gets rotated
        if no_junc:
            assert not mask_rotate[idx_edge, u]
            # print(edge_index.cpu().numpy(),idx_edge, v)
            # print(mask_rotate[idx_edge, v])
            assert mask_rotate[idx_edge, v]

        rot_vec = pos[u] - pos[v] # convention: positive rotation if pointing inwards. NOTE: DIFFERENT FROM THE PAPER!
        # print(pos)
        # print(u, v)
        # print(rot_vec, torsion_updates[idx_edge])
        rot_vec = rot_vec * torsion_updates[idx_edge] / np.linalg.norm(rot_vec) # idx_edge!
        rot_mat = R.from_rotvec(rot_vec).as_matrix()

        pos[mask_rotate[idx_edge]] = (pos[mask_rotate[idx_edge]] - pos[v]) @ rot_mat.T + pos[v]

    if not as_numpy: pos = torch.from_numpy(pos.astype(np.float32))
    return pos


def perturb_batch(data, torsion_updates, split=False, return_updates=False):
    if type(data) is Data:
        # 用edge_mask做有个问题 edge_index一定按小到大排列
        return modify_conformer(data.pos, 
            data.edge_index.T[data.edge_mask], 
            data.mask_rotate, torsion_updates)
    pos_new = [] if split else copy.deepcopy(data.pos)
    edges_of_interest = data.edge_index.T[data.edge_mask]
    idx_node = 0
    idx_edges = 0
    torsion_update_list = []
    for i, mask_rotate in enumerate(data.mask_rotate):
        pos = data.pos[idx_node:idx_node + mask_rotate.shape[1]]
        edges = edges_of_interest[idx_edges:idx_edges + mask_rotate.shape[0]] - idx_node
        torsion_update = torsion_updates[idx_edges:idx_edges + mask_rotate.shape[0]]
        torsion_update_list.append(torsion_update)
        pos_new_ = modify_conformer(pos, edges, mask_rotate, torsion_update)
        if split:
            pos_new.append(pos_new_)
        else:
            pos_new[idx_node:idx_node + mask_rotate.shape[1]] = pos_new_

        idx_node += mask_rotate.shape[1]
        idx_edges += mask_rotate.shape[0]
    if return_updates:
        return pos_new, torsion_update_list
    return pos_new


def bdot(a, b):
    return torch.sum(a*b, dim=-1, keepdim=True)


def get_torsion_angles(dihedral, batch_pos, batch_size):
    batch_pos = batch_pos.reshape(batch_size, -1, 3)

    c, a, b, d = dihedral[:, 0], dihedral[:, 1], dihedral[:, 2], dihedral[:, 3]
    c_project_ab = batch_pos[:,a] + bdot(batch_pos[:,c] - batch_pos[:,a], batch_pos[:,b] - batch_pos[:,a]) / bdot(batch_pos[:,b] - batch_pos[:,a], batch_pos[:,b] - batch_pos[:,a]) * (batch_pos[:,b] - batch_pos[:,a])
    d_project_ab = batch_pos[:,a] + bdot(batch_pos[:,d] - batch_pos[:,a], batch_pos[:,b] - batch_pos[:,a]) / bdot(batch_pos[:,b] - batch_pos[:,a], batch_pos[:,b] - batch_pos[:,a]) * (batch_pos[:,b] - batch_pos[:,a])
    dshifted = batch_pos[:,d] - d_project_ab + c_project_ab
    cos = bdot(dshifted - c_project_ab, batch_pos[:,c] - c_project_ab) / (
                torch.norm(dshifted - c_project_ab, dim=-1, keepdim=True) * torch.norm(batch_pos[:,c] - c_project_ab, dim=-1,
                                                                                       keepdim=True))
    cos = torch.clamp(cos, -1 + 1e-5, 1 - 1e-5)
    angle = torch.acos(cos)
    sign = torch.sign(bdot(torch.cross(dshifted - c_project_ab, batch_pos[:,c] - c_project_ab), batch_pos[:,b] - batch_pos[:,a]))
    torsion_angles = (angle * sign).squeeze(-1)
    return torsion_angles
