import os.path
from multiprocessing import Pool

from rdkit import Chem
import numpy as np
import glob, pickle, random
import os.path as osp
import torch, tqdm
import networkx as nx
import copy
from torch_geometric.data import Dataset, DataLoader
from torch_geometric.transforms import BaseTransform
from collections import defaultdict
from utils.frag_aug import frag_aug, frag_aug_conn,fragmentation, chemfrag_bondidx

from utils.featurization import dihedral_pattern, featurize_mol, qm9_types, drugs_types
from utils.torsion import *


class TorsionNoiseTransform(BaseTransform):
    def __init__(self, sigma_min=0.01 * np.pi, sigma_max=np.pi, boltzmann_weight=False, dec=None):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.boltzmann_weight = boltzmann_weight
        if 'junc' in dec:
            self.junc = False
        else:
            self.junc = True

    def __call__(self, data):
        # select conformer
        if self.boltzmann_weight:
            data.pos = random.choices(data.pos, data.weights, k=1)[0]
        else:
            data.pos = random.choice(data.pos)

        try:
            edge_mask, mask_rotate = data.edge_mask, data.mask_rotate
        except:
            edge_mask, mask_rotate = data.mask_edges, data.mask_rotate
            data.edge_mask = torch.tensor(data.mask_edges)

        sigma = np.exp(np.random.uniform(low=np.log(self.sigma_min), high=np.log(self.sigma_max)))
        data.node_sigma = sigma * torch.ones(data.num_nodes)

        torsion_updates = np.random.normal(loc=0.0, scale=sigma, size=edge_mask.sum())
        # torsion_updates = np.random.normal(loc=0.0, scale=sigma, size=len(edge_mask) )
        data.pos = modify_conformer(data.pos, data.edge_index.T[edge_mask], mask_rotate, torsion_updates, no_junc=self.junc)
        data.edge_rotate = torch.tensor(torsion_updates)
        return data

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(sigma_min={self.sigma_min}, '
                f'sigma_max={self.sigma_max})')


class ConformerDataset(Dataset):
    def __init__(self, root, split_path, mode, types, dataset, dec='none', frag_aug=False , transform=None, num_workers=1, limit_molecules=None,cache=None, pickle_dir=None, boltzmann_resampler=None, rigid=False, z=20):
        # part of the featurisation and filtering code taken from GeoMol https://github.com/PattanaikL/GeoMol

        super(ConformerDataset, self).__init__(root, transform)
        self.dec = dec
        self.root = root
        self.types = types
        self.rigid = rigid
        self.z = z
        self.frag_aug = frag_aug
        self.failures = defaultdict(int)
        self.dataset = dataset
        self.boltzmann_resampler = boltzmann_resampler

        if cache: cache += "_" + dec + "." +mode
        self.cache = cache
        if cache and os.path.exists(cache):
            print('Reusing preprocessing from cache', cache)
            with open(cache, "rb") as f:
                self.datapoints = pickle.load(f)
        else:
            print("Preprocessing")
            self.datapoints = self.preprocess_datapoints(root, split_path, pickle_dir, mode, num_workers, limit_molecules)
            if cache:
                print("Caching at", cache)
                with open(cache, "wb") as f:
                    pickle.dump(self.datapoints, f)

        if limit_molecules:
            self.datapoints = self.datapoints[:limit_molecules]


    def preprocess_datapoints(self, root, split_path, pickle_dir, mode, num_workers, limit_molecules):
        mols_per_pickle = 1000
        split_idx = 0 if mode == 'train' else 1 if mode == 'val' else 2
        split = sorted(np.load(split_path, allow_pickle=True)[split_idx])
        if limit_molecules:
            split = split[:limit_molecules]
        smiles = np.array(sorted(glob.glob(osp.join(self.root, '*.pickle'))))
        smiles = smiles[split]

        self.open_pickles = {}
        if pickle_dir:
            smiles = [(i // mols_per_pickle, smi[len(root):-7]) for i, smi in zip(split, smiles)]
            if limit_molecules:
                smiles = smiles[:limit_molecules]
            self.current_pickle = (None, None)
            self.pickle_dir = pickle_dir
        else:
            smiles = [smi[len(root):-7] for smi in smiles]
        # print(smiles)
        # sys.exit(0)
        print('Preparing to process', len(smiles), 'smiles')
        # print(smiles)
        datapoints = []
        if num_workers > 1:
            p = Pool(num_workers)
            p.__enter__()
        with tqdm.tqdm(total=len(smiles)) as pbar:
            map_fn = p.imap if num_workers > 1 else map
            for t in map_fn(self.filter_smiles, smiles):
                if t:
                    datapoints.append(t)
                pbar.update()
        if num_workers > 1: p.__exit__(None, None, None)
        print('Fetched', len(datapoints), 'mols successfully')
        print(self.failures)
        if pickle_dir: del self.current_pickle
        return datapoints

    def filter_smiles(self, smile):
        if type(smile) is tuple:
            pickle_id, smile = smile
            current_id, current_pickle = self.current_pickle
            if current_id != pickle_id:
                path = osp.join(self.pickle_dir, str(pickle_id).zfill(3) + '.pickle')
                if not osp.exists(path):
                    self.failures[f'std_pickle{pickle_id}_not_found'] += 1
                    return False
                with open(path, 'rb') as f:
                    self.current_pickle = current_id, current_pickle = pickle_id, pickle.load(f)
            if smile not in current_pickle:
                self.failures['smile_not_in_std_pickle'] += 1
                return False
            mol_dic = current_pickle[smile]

        else:
            if not os.path.exists(os.path.join(self.root, smile + '.pickle')):
                self.failures['raw_pickle_not_found'] += 1
                return False
            pickle_file = osp.join(self.root, smile + '.pickle')
            mol_dic = self.open_pickle(pickle_file)

        smile = mol_dic['smiles']

        if '.' in smile:
            self.failures['dot_in_smile'] += 1
            return False

        # filter mols rdkit can't intrinsically handle
        mol = Chem.MolFromSmiles(smile)
        if not mol:
            self.failures['mol_from_smiles_failed'] += 1
            return False

        mol = mol_dic['conformers'][0]['rd_mol']
        N = mol.GetNumAtoms()
        if not mol.HasSubstructMatch(dihedral_pattern):
            self.failures['no_substruct_match'] += 1
            return False

        if N < 4:
            self.failures['mol_too_small'] += 1
            return False

        data = self.featurize_mol(mol_dic)
        if not data:
            self.failures['featurize_mol_failed'] += 1
            return False
# get transformation mask is important
        # edge_mask, mask_rotate, rotate_idx = get_transformation_mask(data)
        # edge_mask, mask_rotate, rotate_idx = get_transformation_mask_old(data)
        # edge_mask, mask_rotate, rotate_idx = get_transformation_mask_recap(data)
        # if self.dec == 'none':
        if self.rigid:
            edge_mask, mask_rotate, conn_rotate_idx = get_transformation_mask_old(data)
            # if self.dec == 'none':
            if 'none' in self.dec:
                rotate_idx = conn_rotate_idx
            else:
                rotate_idx = chemfrag_bondidx(data, edge_mask, self.dec)
                if 'aug' in self.dec:
                    rotate_idx = [item for item in conn_rotate_idx if item not in rotate_idx]
            if 'junc' in self.dec:
                G = to_networkx(data, to_undirected=False).to_undirected()
                iedge_list = data.edge_index.T.tolist()
                ring_mask_edges, ring_mask_rotate, ring_rotate_index = FindJuncBonds(data, iedge_list, G)
                if ring_rotate_index is not None:
                    t_rotate_idx = conn_rotate_idx + ring_rotate_index
                    mask_rotate = np.concatenate((mask_rotate, ring_mask_rotate))
                    sorted_indices = sorted(range(len(t_rotate_idx)), key=lambda x: t_rotate_idx[x])
                    edge_mask = edge_mask + ring_mask_edges
                    mask_rotate = np.array([mask_rotate[i] for i in sorted_indices], dtype=bool)
                # else:
                #     t_rotate_idx = rotate_idx
        # else:
        #     if self.dec == 'frag_conn':
        #         edge_mask, mask_rotate, rotate_idx = get_transformation_mask_frag(data)
        #     if self.dec == 'frag_all':
        #         edge_mask, mask_rotate, rotate_idx = get_transformation_mask_all(data)
        #     if self.dec == 'recap':
        #         edge_mask, mask_rotate, rotate_idx = get_transformation_mask_recap(data)
        #     if self.dec == 'brics':
        #         edge_mask, mask_rotate, rotate_idx = get_transformation_mask_brics(data)
        #     if self.dec == 'br':
        #         edge_mask, mask_rotate, rotate_idx = get_transformation_mask_bnr(data)
        #     if self.dec == 'junc':
        #         edge_mask, mask_rotate, rotate_idx = get_transformation_mask_junc(data)
        #     if self.dec == 'junc_none':
        #         edge_mask, mask_rotate, rotate_idx = get_transformation_mask_junc_old(data)
        #     if self.dec == 'junc_brics':
        #         edge_mask, mask_rotate, rotate_idx = get_transformation_mask_junc_brics(data)
        #     if self.dec == 'junc_recap':
        #         edge_mask, mask_rotate, rotate_idx = get_transformation_mask_junc_recap(data)
        #     if self.dec == 'none':
        #         edge_mask, mask_rotate, rotate_idx = get_transformation_mask_old(data)
            # edge_mask, mask_rotate = conn_edge_mask, conn_mask_rotate
        if np.sum(edge_mask) < 0.5:
            self.failures['no_rotable_bonds'] += 1
            return False

        data.edge_mask = torch.tensor(edge_mask)
        data.mask_rotate = mask_rotate
        if self.frag_aug:
            if self.rigid:
                # data, augmented_data = frag_aug_conn(data, rotate_idx)
                data = fragmentation(data, conn_rotate_idx, rotate_idx, self.z)
            else:
                data, _ = frag_aug(data, rotate_idx)
            return data
        return data

    def len(self):
        return len(self.datapoints)

    def get(self, idx):
        data = self.datapoints[idx]
        if self.boltzmann_resampler:
            self.boltzmann_resampler.try_resample(data)
        return copy.deepcopy(data)

    def open_pickle(self, mol_path):
        with open(mol_path, "rb") as f:
            dic = pickle.load(f)
        return dic

    def featurize_mol(self, mol_dic):
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

            if self.boltzmann_resampler is not None:
                # torsional Boltzmann generator uses only the local structure of the first conformer
                break

        # return None if no non-reactive conformers were found
        if len(pos) == 0:
            return None

        data = featurize_mol(correct_mol, self.types)
        normalized_weights = list(np.array(weights) / np.sum(weights))
        if np.isnan(normalized_weights).sum() != 0:
            print(name, len(confs), len(pos), weights)
            normalized_weights = [1 / len(weights)] * len(weights)
        data.canonical_smi, data.mol, data.pos, data.weights = canonical_smi, correct_mol, pos, normalized_weights

        return data

    def resample_all(self, resampler, temperature=None):
        ess = []
        for data in tqdm.tqdm(self.datapoints):
            ess.append(resampler.resample(data, temperature=temperature))
        return ess


def construct_loader(args, modes=('train', 'val'), boltzmann_resampler=None):
    if isinstance(modes, str):
        modes = [modes]

    loaders = []
    transform = TorsionNoiseTransform(sigma_min=args.sigma_min, sigma_max=args.sigma_max,
                                      boltzmann_weight=args.boltzmann_weight, dec = args.dec)
    types = qm9_types if args.dataset == 'qm9' else drugs_types

    for mode in modes:
        dataset = ConformerDataset(args.data_dir, args.split_path, mode,dec=args.dec, frag_aug=args.aug, 
                                   rigid=args.rigid , z = args.z, dataset=args.dataset,
                                   types=types, transform=transform,
                                   num_workers=1,
                                   limit_molecules=args.limit_train_mols,
                                   cache=args.cache,
                                   pickle_dir=args.std_pickles,
                                   boltzmann_resampler=boltzmann_resampler)
        loader = DataLoader(dataset=dataset,
                            batch_size=args.batch_size,
                            num_workers=args.num_workers,
                            shuffle=False if mode == 'test' else True)
        loaders.append(loader)

    if len(loaders) == 1:
        return loaders[0]
    else:
        return loaders



class ConformerDataset_cache(Dataset):
    def __init__(self, root, cache, transform=None):
        # part of the featurisation and filtering code taken from GeoMol https://github.com/PattanaikL/GeoMol

        super(ConformerDataset_cache, self).__init__(root, transform)
        self.root = root
        self.cache = cache
        print(cache)
        print(os.path.exists(cache))
        if cache and os.path.exists(cache):
            print('Reusing preprocessing from cache', cache)
            with open(cache, "rb") as f:
                self.datapoints = pickle.load(f)
    def len(self):
        return len(self.datapoints)
    
    def get(self, idx):
        data = self.datapoints[idx]
        return copy.deepcopy(data)
def construct_loader_new(args, train_data_path, val_data_path, modes=('train', 'val')):
    if isinstance(modes, str):
        modes = [modes]

    loaders = []
    transform = TorsionNoiseTransform(sigma_min=args.sigma_min, sigma_max=args.sigma_max,
                                      boltzmann_weight=args.boltzmann_weight, dec = args.dec)
    types = qm9_types if args.dataset == 'qm9' else drugs_types

    data_paths = {
        'train': train_data_path,
        'val': val_data_path
    }
    print(data_paths)
    for mode in modes:
        print('-----------------',mode)
        print(data_paths[mode])
        print('----------------',mode)
        dataset = ConformerDataset_cache(args.data_dir, data_paths[mode], transform)
        loader = DataLoader(dataset=dataset,
                            batch_size=args.batch_size,
                            num_workers=args.num_workers,
                            shuffle=False if mode == 'test' else True)
        loaders.append(loader)

    if len(loaders) == 1:
        return loaders[0]
    else:
        return loaders