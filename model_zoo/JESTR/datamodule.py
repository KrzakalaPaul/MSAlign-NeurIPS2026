from torch.utils.data import Dataset
from collections import OrderedDict
import torch
import pandas as pd
import numpy as np
import lightning.pytorch as pl
import json
from transforms.molecules_transforms import MoleculeToGraph
from transforms.spectra_transforms import BIN_Transform
import dgl


class MoleculeGraphStore:
    """Build each valid molecular graph once and reuse it across epochs.

    The store is constructed before dataloader workers are started. On the
    Linux training nodes the workers inherit these read-only graphs, avoiding
    repeated RDKit and DGL featurization in ``__getitem__``.
    """

    def __init__(self, smiles_iterable):
        self.graphs = {}
        transform = MoleculeToGraph()
        unique_smiles = sorted(set(smiles_iterable))
        print(f"Precomputing {len(unique_smiles):,} molecular graphs for JESTR...", flush=True)
        for index, smiles in enumerate(unique_smiles, start=1):
            try:
                graph = transform(smiles)
                if graph.ndata['h'].shape[1] != 78:
                    raise ValueError(
                        f"Expected 78 node features, got {graph.ndata['h'].shape[1]}"
                    )
                self.graphs[smiles] = graph
            except Exception:
                # Invalid molecules are excluded before candidate sampling.
                pass
            if index % 10_000 == 0:
                print(f"  prepared {index:,}/{len(unique_smiles):,}", flush=True)
        n_invalid = len(unique_smiles) - len(self.graphs)
        print(
            f"JESTR graph store ready: {len(self.graphs):,} valid, "
            f"{n_invalid:,} invalid.",
            flush=True,
        )

    def __contains__(self, smiles):
        return smiles in self.graphs

    def __getitem__(self, smiles):
        return self.graphs[smiles]


def load_fold_data(labelled_dataset_name, split_method, fold):
    '''
    In JESTR each epoch = one pass on the list of unique smiles.
    (this way, spectra with the same smiles will be seen in different epochs which prevents false negatives in the in-batch contrastive loss)
    
    This functions loads:
    - the list of spectra for the fold
    - a mapping whose keys are the unique smiles in the fold, and the values are lists of indices of spectra with that smiles in the fold spectra list
    '''
    metadata = pd.read_csv(f'data/{labelled_dataset_name}/metadata.csv')
    split = pd.read_csv(f'data/{labelled_dataset_name}/splits/{split_method}.csv')['fold']
    split_mask = (split == fold).values
    spectra = np.load(f'data/{labelled_dataset_name}/spectra.npy')
    spectra_to_smiles = metadata['unique_smiles_idx'].values
    unique_smiles = pd.read_csv(f'data/{labelled_dataset_name}/unique_smiles.csv')['smiles'].values

    print("Preparing safe iteration for JESTR...")
    spectra_fold = spectra[split_mask]
    spectra_to_smiles_fold = spectra_to_smiles[split_mask]
    map_smiles_to_spectra_fold = {}
    for idx, smiles_idx in enumerate(spectra_to_smiles_fold):
        smiles = unique_smiles[smiles_idx]
        if smiles not in map_smiles_to_spectra_fold:
            map_smiles_to_spectra_fold[smiles] = []
        map_smiles_to_spectra_fold[smiles].append(idx)
        
    return map_smiles_to_spectra_fold, spectra_fold

def keep_only_k_candidates(candidates, k):
    """Keep the target followed by at most ``k`` sampled negatives."""

    n_valid_candidates = len(candidates) - 1
    if k is not None and n_valid_candidates > k:
        indices = np.random.choice(range(1, n_valid_candidates + 1), size=k, replace=False)
        candidates = [candidates[0]] + [candidates[i] for i in indices]
    return candidates

class PairDataset(Dataset):
    '''
    Used for pretraining JESTR, no negative loading, just return pairs of (spectrum, positive candidate)
    '''
    def __init__(self,
                 labelled_dataset_name,
                 split_method,
                 fold,
                 graph_store=None,
                 bin_width=1.0,
                 max_mz=1005
                 ):
        
        # Load fold data
        self.map_smiles_to_spectra_fold, self.spectra_fold = load_fold_data(labelled_dataset_name, split_method, fold)
        self.unique_smiles_fold = sorted(self.map_smiles_to_spectra_fold.keys()) # use sorted to ensure deterministic order of unique smiles for reproducibility
        
        # Load transforms
        self.graph_store = graph_store or MoleculeGraphStore(self.unique_smiles_fold)
        self.spectra_transform = BIN_Transform(max_mz=max_mz, bin_width=bin_width)
        
    def __len__(self):
        return len(self.unique_smiles_fold)
    
    def __getitem__(self, idx):
        smiles = self.unique_smiles_fold[idx]
        mol = self.graph_store[smiles]
        spectra_indices = self.map_smiles_to_spectra_fold[smiles]
        idx = np.random.choice(spectra_indices) # shuffle the spectra indices for this smiles to ensure different spectra are seen in different epochs
        spectra = self.spectra_fold[idx] # randomly choose one spectrum for this smiles
        ms = torch.from_numpy(self.spectra_transform(spectra)).to(torch.float32)
        return ms, mol
    
    def collate_fn(self, batch):
        ms, mol = zip(*batch)
        ms = torch.stack(ms)
        mol = dgl.batch(mol)
        return ms, mol
    
    
class CandidateDataset(Dataset):
    '''
    Used for loading candidate molecules for each spectrum in the fold (evaluation and finetuning JESTR)
    '''
    def __init__(self,
                 labelled_dataset_name,
                 candidate_map_name,
                 split_method,
                 fold,
                 k_candidates,
                 bin_width=1.0,
                 max_mz=1005,
                 all_spectra=False,
                 candidate_map=None,
                 graph_store=None,
                 ):
        
        self.k_candidates = k_candidates
        
        # Load fold data
        self.map_smiles_to_spectra_fold, self.spectra_fold = load_fold_data(labelled_dataset_name, split_method, fold)
        self.unique_smiles_fold = sorted(self.map_smiles_to_spectra_fold.keys()) # use sorted to ensure deterministic order of unique smiles for reproducibility
        self.all_spectra = all_spectra
        self.samples = [
            (smiles, spectrum_idx)
            for smiles in self.unique_smiles_fold
            for spectrum_idx in self.map_smiles_to_spectra_fold[smiles]
        ] if all_spectra else None
        if candidate_map is None:
            candidate_map_path = f'data/{labelled_dataset_name}/candidates/{candidate_map_name}/map.json'
            with open(candidate_map_path, 'r') as f:
                candidate_map = json.load(f)
        self.candidate_map = candidate_map
        # Cache labelled targets only. Candidate maps contain millions of
        # mostly one-off negatives, so materializing every candidate graph is
        # prohibitively expensive. Negatives are transformed lazily below.
        self.graph_store = graph_store or MoleculeGraphStore(self.unique_smiles_fold)
        self.mol_transform = MoleculeToGraph()
        self._candidate_graph_cache = OrderedDict()
        self._candidate_graph_cache_size = 8
        self.spectra_transform = BIN_Transform(max_mz=max_mz, bin_width=bin_width)
        
    def __len__(self):
        return len(self.samples) if self.all_spectra else len(self.unique_smiles_fold)
    
    def __getitem__(self, idx):
        if self.all_spectra:
            smiles, spectrum_idx = self.samples[idx]
        else:
            smiles = self.unique_smiles_fold[idx]
            spectrum_idx = np.random.choice(self.map_smiles_to_spectra_fold[smiles])
        spectra = self.spectra_fold[spectrum_idx]
        ms = torch.from_numpy(self.spectra_transform(spectra)).to(torch.float32)
        
        if self.all_spectra and smiles in self._candidate_graph_cache:
            candidates_graphs = self._candidate_graph_cache.pop(smiles)
            self._candidate_graph_cache[smiles] = candidates_graphs
            return ms, dgl.batch(candidates_graphs)

        candidates = self.candidate_map[smiles]
        if not candidates:
            raise ValueError(f"No valid candidates are available for {smiles!r}.")
        # Sample identifiers before graph construction. This changes an
        # O(total candidate universe) initialization into O(K) work per
        # training query while preserving fresh negative samples each epoch.
        candidates = keep_only_k_candidates(candidates, self.k_candidates)
        candidates_graphs = []
        for position, candidate in enumerate(candidates):
            try:
                graph = (
                    self.graph_store[candidate]
                    if candidate in self.graph_store
                    else self.mol_transform(candidate)
                )
                if graph.ndata['h'].shape[1] != 78:
                    raise ValueError("unexpected atom feature width")
                candidates_graphs.append(graph)
            except Exception as error:
                if position == 0:
                    raise ValueError(
                        f"The target candidate for {smiles!r} could not be converted to a graph."
                    ) from error
                continue

        if self.all_spectra:
            self._candidate_graph_cache[smiles] = candidates_graphs
            if len(self._candidate_graph_cache) > self._candidate_graph_cache_size:
                self._candidate_graph_cache.popitem(last=False)
            
        return ms, dgl.batch(candidates_graphs)
    
    def collate_fn(self, batch):
        ms, candidates_graphs = zip(*batch)
        ms = torch.stack(ms)
        return ms, candidates_graphs # candidates_graphs is a list of batched graphs
    

    
class JESTER_Datamodule(pl.LightningDataModule):

    def __init__(self, 
                 labelled_dataset_name,
                 candidate_map_name,
                 split_method,
                 k_candidates,
                 mode,
                 bin_width=0.1,
                 max_mz=1005,
                 batch_size=128, 
                 batch_size_test=16, # ALL candidates are loaded during validation/test, so we need to reduce the batch size to fit in memory
                 n_workers=8, 
                 prefetch_factor=2,
                 ):
        super().__init__()
        
        ### Load datasets
        if mode == 'pretrain':
            self.train_dataset = PairDataset(labelled_dataset_name=labelled_dataset_name,
                                            split_method=split_method,
                                            fold='train',
                                            bin_width=bin_width,
                                            max_mz=max_mz)
            
            self.val_dataset = PairDataset(labelled_dataset_name=labelled_dataset_name,
                                            split_method=split_method,
                                            fold='val',
                                            bin_width=bin_width,
                                            max_mz=max_mz)
            self.test_dataset = None
        elif mode == 'finetune':
            candidate_map_path = f'data/{labelled_dataset_name}/candidates/{candidate_map_name}/map.json'
            with open(candidate_map_path, 'r') as f:
                candidate_map = json.load(f)
            # Candidate-map keys are labelled targets. This is tens of
            # thousands of reusable graphs, rather than 5--6.5 million
            # mostly one-off negative graphs.
            graph_store = MoleculeGraphStore(candidate_map.keys())
            self.train_dataset = CandidateDataset(labelled_dataset_name=labelled_dataset_name,
                                                candidate_map_name=candidate_map_name,
                                                split_method=split_method,
                                                fold='train',
                                                k_candidates=k_candidates,
                                                bin_width=bin_width,
                                                max_mz=max_mz,
                                                all_spectra=False,
                                                candidate_map=candidate_map,
                                                graph_store=graph_store)
            
            self.val_dataset = CandidateDataset(labelled_dataset_name=labelled_dataset_name,
                                                candidate_map_name=candidate_map_name,
                                                split_method=split_method,
                                                fold='val',
                                                k_candidates=None,
                                                bin_width=bin_width,
                                                max_mz=max_mz,
                                                all_spectra=True,
                                                candidate_map=candidate_map,
                                                graph_store=graph_store)

            self.test_dataset = CandidateDataset(labelled_dataset_name=labelled_dataset_name,
                                                  candidate_map_name=candidate_map_name,
                                                  split_method=split_method,
                                                  fold='test',
                                                  k_candidates=None,
                                                  bin_width=bin_width,
                                                  max_mz=max_mz,
                                                  all_spectra=True,
                                                  candidate_map=candidate_map,
                                                  graph_store=graph_store)
        else:
            raise ValueError(f"Invalid mode {mode}. Expected 'pretrain' or 'finetune'.")
        
        self.mode = mode
        
        ### Save parameters
        self.batch_size = batch_size
        self.batch_size_test = batch_size_test
        self.n_workers = n_workers
        self.prefetch_factor = prefetch_factor
        
    def train_dataloader(self):
        options = self._loader_options(persistent_workers=True)
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=self.train_dataset.collate_fn, **options)
    
    def val_dataloader(self):
        generator = torch.Generator()
        generator.manual_seed(42)
        batch_size= self.batch_size if self.mode == 'pretrain' else self.batch_size_test
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=batch_size, shuffle=self.mode == 'pretrain', collate_fn=self.val_dataset.collate_fn, generator=generator, **self._loader_options())
    
    def test_dataloader(self):
        if self.test_dataset is None:
            raise RuntimeError("The pretraining datamodule has no test dataset.")
        generator = torch.Generator()
        generator.manual_seed(42)
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size_test, shuffle=False, collate_fn=self.test_dataset.collate_fn, generator=generator, **self._loader_options())

    def _loader_options(self, *, persistent_workers: bool = False):
        """Only pass multiprocessing-only options when workers are enabled."""

        options = {"num_workers": self.n_workers}
        if self.n_workers > 0:
            options["prefetch_factor"] = self.prefetch_factor
            options["persistent_workers"] = persistent_workers
        return options
