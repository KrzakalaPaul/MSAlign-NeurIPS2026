import pandas as pd
import numpy as np
import os
from preprocessing.utils.rdkit_mp import normalize_smiles_list

def process_smiles(dataset_name, n_threads=16, chunk_size=4096, overwrite=False):

    
    if os.path.exists(f"data/{dataset_name}/unique_smiles.csv") and not overwrite:
        print(f"Unique SMILES file already exists for {dataset_name}. Skipping processing.")
    else:
        df_old = pd.read_csv(f"data/{dataset_name}/metadata.csv")
        spectra_old = np.load(f"data/{dataset_name}/spectra.npy")

        # Normalize SMILES
        df_old["smiles"] = normalize_smiles_list(
            df_old["smiles"].tolist(),
            n_threads=n_threads, chunk_size=chunk_size,
            description=f"Normalizing SMILES for {dataset_name}"
        )

        print(f"Grouping and reordering {len(df_old):,} spectra by molecule...")

        # Group row indices by SMILES, preserving sorted order
        grouped = df_old.groupby("smiles", sort=True).indices  # dict: smiles -> array of row indices

        unique_smiles = sorted(grouped.keys())
        row_order = np.concatenate([grouped[smi] for smi in unique_smiles])

        # Reorder both spectra and all metadata columns consistently
        spectra_grouped = np.concatenate([spectra_old[grouped[smi]] for smi in unique_smiles])
        df_reordered = df_old.iloc[row_order].reset_index(drop=True)
        
        # Save the mapping from original row indices to unique SMILES index
        unique_smiles_idx = np.empty(len(df_old), dtype=np.int64)
        for idx, smi in enumerate(unique_smiles):
            unique_smiles_idx[grouped[smi]] = idx
        df_reordered["unique_smiles_idx"] = unique_smiles_idx[row_order]
                
        # Build index pointers
        counts = [len(grouped[smi]) for smi in unique_smiles]
        spectra_end_idx = np.cumsum(counts)
        spectra_start_idx = spectra_end_idx - counts

        df_unique = pd.DataFrame({
            "smiles": unique_smiles,
            "spectra_start_idx": spectra_start_idx,
            "spectra_end_idx": spectra_end_idx,
        })

        df_reordered.to_csv(f"data/{dataset_name}/metadata.csv", index=False)
        df_unique.to_csv(f"data/{dataset_name}/unique_smiles.csv", index=False)
        np.save(f"data/{dataset_name}/spectra.npy", spectra_grouped)
        print(f"Processed SMILES for {dataset_name}. Found {len(unique_smiles)} unique SMILES.")
        
