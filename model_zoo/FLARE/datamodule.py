from torch.utils.data import Dataset
from collections import OrderedDict
import torch
import pandas as pd
import numpy as np
import lightning.pytorch as pl
import json
from model_zoo.JESTR.datamodule import keep_only_k_candidates
from transforms.spectra_transforms import Subformula_Transform
from transforms.molecules_transforms import MoleculeToGraph


def spectrum_to_tokens(spectrum, transform):
    """Materialize one annotated spectrum once, outside the epoch loop."""
    if (
        spectrum['mz'] is None
        or spectrum['intensities'] is None
        or spectrum['subformulas'] is None
    ):
        return torch.zeros((1, transform.get_dim()), dtype=torch.float32)
    return torch.from_numpy(transform(
        mz_list=spectrum['mz'],
        intensities_list=spectrum['intensities'],
        formulas_list=spectrum['subformulas'],
    )).to(torch.float32)


class MoleculeGraphStore:
    """Build each molecular graph once and reuse it for every epoch/query."""

    def __init__(self, smiles_values):
        self.transform = MoleculeToGraph()
        self.graphs = {}
        self.invalid = set()
        self.add(smiles_values)

    def add(self, smiles_values):
        unique_smiles = sorted(set(smiles_values).difference(self.graphs, self.invalid))
        if not unique_smiles:
            return
        print(f"Precomputing {len(unique_smiles):,} new molecular graphs for FLARE...", flush=True)
        n_before = len(self.graphs)
        for smiles in unique_smiles:
            try:
                graph = self.transform(smiles)
                if graph.ndata['h'].shape[1] != self.transform.get_dim():
                    raise ValueError("unexpected atom feature width")
                self.graphs[smiles] = graph
            except Exception:
                # Invalid molecules are removed once, rather than retried for
                # every occurrence in every epoch.
                self.invalid.add(smiles)
        print(
            f"FLARE graph store: {len(self.graphs):,} total valid; "
            f"{len(unique_smiles) - (len(self.graphs) - n_before):,} new invalid.",
            flush=True,
        )

    def __contains__(self, smiles):
        return smiles in self.graphs

    def __getitem__(self, smiles):
        return self.graphs[smiles]


def pack_graphs(graphs):
    """Pack DGL graphs as ordinary tensors safe for DataLoader IPC."""
    # DGL requires batch metadata to use the graph's ID dtype. Molecular
    # graphs from dgllife use int32, whereas hand-built graphs often use int64.
    idtype = graphs[0].idtype
    node_counts = torch.tensor([graph.num_nodes() for graph in graphs], dtype=idtype)
    edge_counts = torch.tensor([graph.num_edges() for graph in graphs], dtype=idtype)
    offsets = torch.cat((torch.zeros(1, dtype=idtype), node_counts.cumsum(0)[:-1]))
    sources, destinations = [], []
    for graph, offset in zip(graphs, offsets):
        source, destination = graph.edges()
        sources.append(source + offset)
        destinations.append(destination + offset)
    return {
        'node_counts': node_counts,
        'edge_counts': edge_counts,
        'src': torch.cat(sources),
        'dst': torch.cat(destinations),
        'h': torch.cat([graph.ndata['h'] for graph in graphs]),
    }

def collate_tokens(batch: list[torch.Tensor], padding_value: float = 0.0) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        batch: list of N tensors of shape (L_i, d)
        padding_value: value to pad with (default 0.0)
    Returns:
        tokens: (N, L_max, d) padded token tensor
        mask:   (N, L_max) boolean mask, True where valid (not padding)
    """
    max_len = max(tokens.shape[0] for tokens in batch)
    N = len(batch)
    d = batch[0].shape[1]

    padded = torch.full((N, max_len, d), fill_value=padding_value, dtype=torch.float)
    mask = torch.ones(N, max_len, dtype=torch.bool)

    for i, tokens in enumerate(batch):
        L = tokens.shape[0]
        padded[i, :L] = tokens
        mask[i, :L] = False

    return padded, mask

def load_fold_data(labelled_dataset_name, split_method, fold):
    '''
    In FLARE each epoch = one pass on the list of unique smiles.
    (this way, spectra with the same smiles will be seen in different epochs which prevents false negatives in the in-batch contrastive loss)
    
    This functions loads:
    - the list of spectra for the fold (with their mz, intensities and subformulas)
    - a mapping whose keys are the unique smiles in the fold, and the values are lists of indices of spectra with that smiles in the fold spectra list
    '''
    metadata = pd.read_csv(f'data/{labelled_dataset_name}/metadata.csv')
    split = pd.read_csv(f'data/{labelled_dataset_name}/splits/{split_method}.csv')['fold']
    split_mask = (split == fold).values
    with open(f"data/{labelled_dataset_name}/annotated_peaks.json", "r") as f:
        spectra = json.load(f)  # spectra['mz'][i] = list of float, spectra['intensities'][i] list of float, spectra['subformulas'][i] list of string
    spectra_to_smiles = metadata['unique_smiles_idx'].values
    unique_smiles = pd.read_csv(f'data/{labelled_dataset_name}/unique_smiles.csv')['smiles'].values

    print("Preparing safe iteration for FLARE...")
    spectra_fold = []    
    for i in range(len(spectra_to_smiles)):
        if split_mask[i]:
            spectra_fold.append({'mz': spectra['mz'][i], 'intensities': spectra['intensities'][i], 'subformulas': spectra['subformulas'][i]})
    spectra_to_smiles_fold = spectra_to_smiles[split_mask]
    map_smiles_to_spectra_fold = {}
    for idx, smiles_idx in enumerate(spectra_to_smiles_fold):
        smiles = unique_smiles[smiles_idx]
        if smiles not in map_smiles_to_spectra_fold:
            map_smiles_to_spectra_fold[smiles] = []
        map_smiles_to_spectra_fold[smiles].append(idx)
        
    return map_smiles_to_spectra_fold, spectra_fold

class PairDataset(Dataset):
    '''
    Used for pretraining JESTR, no negative loading, just return pairs of (spectrum, positive candidate)
    '''
    def __init__(self,
                 labelled_dataset_name,
                 split_method,
                 fold,
                 graph_store=None,
                 ):
        
        # Load fold data
        self.map_smiles_to_spectra_fold, self.spectra_fold = load_fold_data(labelled_dataset_name, split_method, fold)
        self.graph_store = graph_store or MoleculeGraphStore(())
        self.graph_store.add(self.map_smiles_to_spectra_fold)
        self.unique_smiles_fold = sorted(
            smiles
            for smiles in self.map_smiles_to_spectra_fold
            if smiles in self.graph_store
        )
        
        # Load transforms
        self.spectra_transform = Subformula_Transform()
        print(f"Precomputing {len(self.spectra_fold):,} FLARE spectrum token tensors...", flush=True)
        self.spectra_fold = [
            spectrum_to_tokens(spectrum, self.spectra_transform)
            for spectrum in self.spectra_fold
        ]
        
    def __len__(self):
        return len(self.unique_smiles_fold)
    
    def __getitem__(self, idx):
        smiles = self.unique_smiles_fold[idx]
        mol = self.graph_store[smiles]
        spectra_indices = self.map_smiles_to_spectra_fold[smiles]
        idx = np.random.choice(spectra_indices) # shuffle the spectra indices for this smiles to ensure different spectra are seen in different epochs
        tokens = self.spectra_fold[idx]
        return tokens, mol
    
    def collate_fn(self, batch):
        tokens, mol = zip(*batch)
        tokens, mask = collate_tokens(tokens)
        ms = {'tokens': tokens, 'mask': mask}
        mol = pack_graphs(mol)
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
                 graph_store=None,
                 ):
        
        self.k_candidates = k_candidates
        
        # Load fold data
        self.map_smiles_to_spectra_fold, self.spectra_fold = load_fold_data(labelled_dataset_name, split_method, fold)
        # Evaluation is spectrum-level: retain every spectrum instead of sampling
        # one spectrum per unique molecule as done by the training dataset.
        self.samples = [
            (smiles, spectrum_idx)
            for smiles in sorted(self.map_smiles_to_spectra_fold)
            for spectrum_idx in self.map_smiles_to_spectra_fold[smiles]
        ]
        candidate_map_path = f'data/{labelled_dataset_name}/candidates/{candidate_map_name}/map.json'
        with open(candidate_map_path, 'r') as f:
            self.candidate_map = json.load(f)
        self.graph_store = graph_store or MoleculeGraphStore(())
        # Cache only labelled molecules.  The SpectraVerse candidate map has
        # millions of distinct negatives, most used only once during final
        # evaluation; eagerly materializing that entire universe is both slow
        # and prohibitively memory hungry.
        self.graph_store.add(self.map_smiles_to_spectra_fold)
        self.mol_transform = MoleculeToGraph()
        self.spectra_transform = Subformula_Transform()
        self._candidate_graph_cache = OrderedDict()
        self._candidate_graph_cache_size = 8
        print(f"Precomputing {len(self.spectra_fold):,} FLARE spectrum token tensors...", flush=True)
        self.spectra_fold = [
            spectrum_to_tokens(spectrum, self.spectra_transform)
            for spectrum in self.spectra_fold
        ]
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        smiles, spectrum_idx = self.samples[idx]
        tokens = self.spectra_fold[spectrum_idx]

        if smiles in self._candidate_graph_cache:
            candidates_graphs = self._candidate_graph_cache.pop(smiles)
            self._candidate_graph_cache[smiles] = candidates_graphs
            return tokens, candidates_graphs

        candidates = self.candidate_map[smiles]
        candidates_graphs = []
        for position, candidate in enumerate(candidates):
            try:
                graph = (
                    self.graph_store[candidate]
                    if candidate in self.graph_store
                    else self.mol_transform(candidate)
                )
                if graph.ndata['h'].shape[1] != self.mol_transform.get_dim():
                    raise ValueError("unexpected atom feature width")
                candidates_graphs.append(graph)
            except Exception as error:
                if position == 0:
                    raise ValueError(
                        f"Target molecule {smiles!r} has no valid FLARE graph."
                    ) from error
                # An invalid negative can be omitted without changing the
                # invariant that candidate zero is the target.
                continue

        candidates_graphs = keep_only_k_candidates(candidates_graphs, self.k_candidates)
        self._candidate_graph_cache[smiles] = candidates_graphs
        if len(self._candidate_graph_cache) > self._candidate_graph_cache_size:
            self._candidate_graph_cache.popitem(last=False)
            
        return tokens, candidates_graphs
    
    def collate_fn(self, batch):
        tokens, candidate_lists = zip(*batch)
        tokens, mask = collate_tokens(tokens)
        ms = {'tokens': tokens, 'mask': mask}
        query_counts = torch.tensor([len(graphs) for graphs in candidate_lists], dtype=torch.long)
        candidates = pack_graphs([
            graph for graphs in candidate_lists for graph in graphs
        ])
        candidates['query_counts'] = query_counts
        return ms, candidates
    

    
class FLARE_Datamodule(pl.LightningDataModule):

    def __init__(self, 
                 labelled_dataset_name,
                 candidate_map_name,
                 split_method,
                 batch_size=128, 
                 batch_size_test=16, # ALL candidates are loaded during validation/test, so we need to reduce the batch size to fit in memory
                 n_workers=8, 
                 prefetch_factor=2
                 ):
        super().__init__()
        
        graph_store = MoleculeGraphStore(())
        self.train_dataset = PairDataset(labelled_dataset_name=labelled_dataset_name,
                                        split_method=split_method,
                                        fold='train',
                                        graph_store=graph_store)
        
        self.val_dataset = PairDataset(labelled_dataset_name=labelled_dataset_name,
                                        split_method=split_method,
                                        fold='val',
                                        graph_store=graph_store)

        self.test_dataset = CandidateDataset(labelled_dataset_name=labelled_dataset_name,
                                              candidate_map_name=candidate_map_name,
                                              split_method=split_method,
                                              fold='test',
                                              k_candidates=None,
                                              graph_store=graph_store)
        
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
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=self.val_dataset.collate_fn, generator=generator, **self._loader_options(persistent_workers=True))
    
    def test_dataloader(self):
        generator = torch.Generator()
        generator.manual_seed(42)
        # Samples are grouped by target molecule. Keeping this order lets each
        # worker reuse the bounded candidate-graph cache for replicate spectra.
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size_test, shuffle=False, collate_fn=self.test_dataset.collate_fn, generator=generator, **self._loader_options(persistent_workers=True))

    def _loader_options(self, *, persistent_workers: bool = False):
        """Only pass multiprocessing-only options when workers are enabled."""

        options = {"num_workers": self.n_workers}
        if self.n_workers > 0:
            options["prefetch_factor"] = self.prefetch_factor
            options["persistent_workers"] = persistent_workers
        return options
