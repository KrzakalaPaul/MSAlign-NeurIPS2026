from torch.utils.data import Dataset
import torch
import pandas as pd
import numpy as np
import h5py
import lightning.pytorch as pl
import json
from transforms.molecules_transforms import MorganFingerprintTransform
from transforms.spectra_transforms import BIN_Transform
from model_zoo.common import keep_only_k_candidates, collate_candidates

class CandidateDataset(Dataset):
    def __init__(self,
                 labelled_dataset_name,
                 candidate_map_name,
                 split_method,
                 fold,
                 k_candidates,
                 fingerprint_size=4096,
                 bin_width=0.1,
                 max_mz=1005
                 ):
        
        self.k_candidates = k_candidates
        
        # Load All
        candidate_map_path = f'data/{labelled_dataset_name}/candidates/{candidate_map_name}/map.json'
        with open(candidate_map_path, 'r') as f:
            self.candidate_map = json.load(f)
        metadata = pd.read_csv(f'data/{labelled_dataset_name}/metadata.csv')
        split = pd.read_csv(f'data/{labelled_dataset_name}/splits/{split_method}.csv')['fold']
        split_mask = (split == fold).values
        spectra = np.load(f'data/{labelled_dataset_name}/spectra.npy')
        spectra_to_smiles = metadata['unique_smiles_idx'].values
        self.unique_smiles = pd.read_csv(f'data/{labelled_dataset_name}/unique_smiles.csv')['smiles'].values
        
        # Keep only fold data
        self.spectra = spectra[split_mask]
        self.spectra_to_smiles = spectra_to_smiles[split_mask]
        
        # Load transforms
        self.mol_transform = MorganFingerprintTransform(fp_size=fingerprint_size)
        self.spectra_transform = BIN_Transform(max_mz=max_mz, bin_width=bin_width)


    def __len__(self):
        return len(self.spectra)

    def collate_fn(self, batch):
        spectra_embeddings, candidates_embeddings, candidates_masks = zip(*batch)
        spectra_embeddings = torch.stack(spectra_embeddings)
        if self.k_candidates is None:
            # If k_candidates is None, the number of candidates can vary across samples, so we need to collate them with padding
            candidates_embeddings, candidates_masks = collate_candidates(candidates_embeddings, candidates_masks)
        else:
            candidates_embeddings, candidates_masks = torch.stack(candidates_embeddings), torch.stack(candidates_masks)

        return spectra_embeddings, candidates_embeddings, candidates_masks

    def __getitem__(self, idx):
        
        spectra = self.spectra[idx]
        ms = torch.from_numpy(self.spectra_transform(spectra)).to(torch.float32)
        
        smiles_idx = self.spectra_to_smiles[idx]
        smiles = self.unique_smiles[smiles_idx]
        candidates = self.candidate_map[smiles]
        
        candidates_embedding = torch.stack([torch.from_numpy(self.mol_transform(cand_smiles)).to(torch.float32) for cand_smiles in candidates])  # (n_candidates, dim)
        candidates_mask = torch.ones(candidates_embedding.shape[0], dtype=torch.bool) # (n_candidates,)
        
        if self.k_candidates is not None:
            candidates_embedding, candidates_mask = keep_only_k_candidates(
                candidates_embedding, candidates_mask, self.k_candidates
            )

        return ms, candidates_embedding, candidates_mask
    
    
class EmbCos_Datamodule(pl.LightningDataModule):

    def __init__(self, 
                 labelled_dataset_name,
                 candidate_map_name,
                 split_method,
                 k_candidates,
                 fingerprint_size=4096,
                 bin_width=0.1,
                 max_mz=1005,
                 batch_size=128, 
                 batch_size_test=16, # ALL candidates are loaded during validation/test, so we need to reduce the batch size to fit in memory
                 n_workers=8, 
                 prefetch_factor=2
                 ):
        super().__init__()
        
        ### Load datasets
        self.train_dataset = CandidateDataset(labelled_dataset_name=labelled_dataset_name,
                                              candidate_map_name=candidate_map_name,
                                              split_method=split_method,
                                              fold='train',
                                              k_candidates=k_candidates,
                                              fingerprint_size=fingerprint_size,
                                              bin_width=bin_width,
                                              max_mz=max_mz)
        
        self.val_dataset = CandidateDataset(labelled_dataset_name=labelled_dataset_name,
                                              candidate_map_name=candidate_map_name,
                                              split_method=split_method,
                                              fold='val',
                                              k_candidates=None,
                                              fingerprint_size=fingerprint_size,
                                              bin_width=bin_width,
                                              max_mz=max_mz)
        
        self.test_dataset = CandidateDataset(labelled_dataset_name=labelled_dataset_name,
                                              candidate_map_name=candidate_map_name,
                                              split_method=split_method,
                                              fold='test',
                                              k_candidates=None,
                                              fingerprint_size=fingerprint_size,
                                              bin_width=bin_width,
                                              max_mz=max_mz)
        
        ### Save parameters
        self.batch_size = batch_size
        self.batch_size_test = batch_size_test
        self.n_workers = n_workers
        self.prefetch_factor = prefetch_factor
        
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=self.train_dataset.collate_fn, num_workers=self.n_workers, prefetch_factor=self.prefetch_factor, persistent_workers=True)
    
    def val_dataloader(self):
        generator = torch.Generator()
        generator.manual_seed(42)
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size_test, shuffle=True, collate_fn=self.val_dataset.collate_fn, num_workers=self.n_workers, prefetch_factor=self.prefetch_factor, generator=generator)
    
    def test_dataloader(self):
        generator = torch.Generator()
        generator.manual_seed(42)
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size_test, shuffle=True, collate_fn=self.test_dataset.collate_fn, num_workers=self.n_workers, prefetch_factor=self.prefetch_factor, generator=generator)
