from preprocessing.download import download_candidate_pool
import os
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
from preprocessing.utils.rdkit_mp import compute_formulas, compute_masses
from .definitions import CANDIDATE_SOURCES
import numpy as np  
import json

def prepare_candidates(dataset_name,
                       n_candidates,
                       kind="mass",
                       overwrite=False,
                       seed=42,
                       n_threads=16):
    if kind not in {"mass", "formula"}:
        raise ValueError("kind must be 'mass' or 'formula'")
    
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    # ----------- Build candidate source name -----------
    candidate_map_name = f"{n_candidates}_candidates"
    candidate_map_name += f"_by_{kind}"
    candidate_map_path = f"data/{dataset_name}/candidates/{candidate_map_name}/map.json"
    if os.path.exists(candidate_map_path) and not overwrite:
        print(f"Candidate map {candidate_map_name} already exists. Skipping candidate preparation.")
        return
    
    print(f"Preparing candidate map {candidate_map_name}...")
    print(f"Candidate source (by order of priority): {CANDIDATE_SOURCES}")
    print(f"Number of candidates per spectrum: {n_candidates}")
    print(f"Candidate selection method: {kind}")
    print('-'*50)
    
    # ----------- Download candidate pools -----------
    for source in CANDIDATE_SOURCES:
        assert source in ['1M', '4M', '118M'], f"Source {source} is not valid. Valid sources are: '1M', '4M', '118M'."
        download_candidate_pool(
            subset=source,
            overwrite=False,
            n_threads=n_threads,
            chunk_size=4096,
        )
    
    # ----------- Load unlabeled datasets -----------
    value_column = "formula" if kind == "formula" else "mass"
    unlabeled_smiles = []
    unlabeled_values = []
    for source in CANDIDATE_SOURCES:
        path = f"data/candidates_pools/{source}.csv"
        print(f"Loading candidate pool {source} from {path}...")
        frame = pd.read_csv(path, usecols=["smiles", value_column])
        unlabeled_smiles.append(frame["smiles"].tolist())
        values = frame[value_column]
        if kind == "mass":
            values = values.round(4)
        unlabeled_values.append(values.tolist())
        print(f"Loaded {len(frame):,} molecules from candidate pool {source}.")
        del frame

     # ----------- Build lookup indices for fast candidate retrieval -----------
    if kind == "formula":
        print("Building formula to SMILES mapping for candidate retrieval...")
        formula_to_smiles_list = []
        for source, formula_series, smiles in zip(
            CANDIDATE_SOURCES, unlabeled_values, unlabeled_smiles
        ):
            # For each source, build a formula -> list of SMILES mapping
            formula_to_smiles = defaultdict(list)
            for formula, smiles_value in tqdm(
                zip(formula_series, smiles),
                total=len(smiles),
                desc=f"Indexing formulas ({source})",
                unit="molecule",
                dynamic_ncols=True,
            ):
                formula_to_smiles[formula].append(smiles_value)
            formula_to_smiles_list.append(formula_to_smiles)

    elif kind == "mass":
        print("Building mass to SMILES mapping for candidate retrieval...")
        mass_coarse_to_fine_list = []
        mass_to_smiles_list = []
        for source, mass_series, smiles in zip(
            CANDIDATE_SOURCES, unlabeled_values, unlabeled_smiles
        ):
            # For each source, build:
            # coarse mass (rounded to 1 decimal) -> set of fine masses, 
            # fine mass -> list of smiles 
            mass_coarse_to_fine = defaultdict(set)
            for mass in tqdm(
                mass_series,
                desc=f"Indexing coarse masses ({source})",
                unit="molecule",
                dynamic_ncols=True,
            ):
                mass_coarse_to_fine[round(mass, 1)].add(mass)
            mass_coarse_to_fine_list.append(mass_coarse_to_fine)

            mass_to_smiles = defaultdict(list)
            for mass, smiles_value in tqdm(
                zip(mass_series, smiles),
                total=len(smiles),
                desc=f"Indexing exact masses ({source})",
                unit="molecule",
                dynamic_ncols=True,
            ):
                mass_to_smiles[mass].append(smiles_value)
            mass_to_smiles_list.append(mass_to_smiles)

    
    # ----------- Load labeled dataset and compute formulas/masses -----------
    smiles = pd.read_csv(f"data/{dataset_name}/unique_smiles.csv")['smiles'].tolist()
    formulas_labeled = (
        compute_formulas(smiles, n_threads=n_threads)
        if kind == "formula"
        else None
    )
    mass_labeled = (
        compute_masses(smiles, n_threads=n_threads)
        if kind == "mass"
        else None
    )
    
    candidate_map = {}
    
    print("Building candidate map...")
    for idx, query_smiles in tqdm(
        enumerate(smiles),
        total=len(smiles),
        desc=f"Candidates: {dataset_name}/{candidate_map_name}",
        unit="target",
        dynamic_ncols=True,
    ):
        
        candidates = [query_smiles] # candidate 0 is always the correct one
        
        query_formula = formulas_labeled[idx] if kind == "formula" else None
        query_mass = mass_labeled[idx] if kind == "mass" else None
        query_mass_coarse = round(query_mass, 1) if kind == "mass" else None
        
        for source_idx in range(len(CANDIDATE_SOURCES)): 
            
            if kind == "formula":
                formula_to_smiles = formula_to_smiles_list[source_idx] 
                candidates_this_source = formula_to_smiles.get(query_formula, [])
            elif kind == "mass":
                mass_coarse_to_fine = mass_coarse_to_fine_list[source_idx]
                mass_to_smiles = mass_to_smiles_list[source_idx]        
                epsilon = query_mass * 10 * 10**(-6)  # 10 ppm tolerance
                candidate_masses_coarse = mass_coarse_to_fine.get(query_mass_coarse, set())
                candidate_masses = [m for m in candidate_masses_coarse if abs(m - query_mass) <= epsilon]
                candidates_this_source = []
                for mass in candidate_masses:
                    candidates_this_source.extend(mass_to_smiles[mass])
                    
            # remove duplicates and already retrieved candidates
            candidates_this_source = sorted(list(set(candidates_this_source) - set(candidates)))
                
            # Add candidates from this source until we have enough candidates
            # If excess of candidates, randomly sample without replacement
            n_needed = n_candidates - len(candidates)
            if n_needed <= len(candidates_this_source):
                candidates.extend(np.random.choice(candidates_this_source, size=n_needed, replace=False).tolist())
                break # stop looking at other sources, we have enough candidates
            else:
                candidates.extend(candidates_this_source)   
        
        candidate_map[query_smiles] = candidates[:n_candidates] 
    
    os.makedirs(f"data/{dataset_name}/candidates/{candidate_map_name}", exist_ok=True)
    with open(candidate_map_path, 'w') as f:
        json.dump(candidate_map, f)
    row_sizes = [len(row) for row in candidate_map.values()]
    print(
        f"Wrote candidate map for {len(candidate_map):,} targets to "
        f"{candidate_map_path} (candidates per target: "
        f"min={min(row_sizes)}, max={max(row_sizes)})."
    )
