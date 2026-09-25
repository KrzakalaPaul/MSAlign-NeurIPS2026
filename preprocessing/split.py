import numpy as np
import pandas as pd
import os
from preprocessing.utils.rdkit_mp import compute_formulas, compute_inchi, compute_murcko_histograms, compute_murcko_scaffolds
from preprocessing.definitions import TRAIN_RATIO, TEST_RATIO, VAL_RATIO
import random

def split_by_key(keys: list[str], train_ratio, test_ratio, val_ratio, seed=42):

    rng = random.Random(seed)

    n = len(keys)

    # shuffle indices to remove ordering bias
    indices = list(range(n))
    rng.shuffle(indices)

    shuffled_keys = [keys[i] for i in indices]

    ratios_sum = train_ratio + test_ratio + val_ratio
    train_ratio /= ratios_sum
    test_ratio /= ratios_sum
    val_ratio /= ratios_sum

    unique_keys = sorted(set(shuffled_keys)) # sort to ensure deterministic order !
    print(f"Unique keys: {len(unique_keys)}/{n}")
    rng.shuffle(unique_keys)

    n_unique = len(unique_keys)

    train_cut = int(train_ratio * n_unique)
    test_cut = train_cut + int(test_ratio * n_unique)

    train_set = set(unique_keys[:train_cut])
    test_set = set(unique_keys[train_cut:test_cut])
    val_set = set(unique_keys[test_cut:])

    shuffled_splits = []
    for k in shuffled_keys:
        if k in train_set:
            shuffled_splits.append("train")
        elif k in test_set:
            shuffled_splits.append("test")
        else:
            shuffled_splits.append("val")

    # restore original order
    splits = [None] * n
    for idx, split in zip(indices, shuffled_splits):
        splits[idx] = split

    return splits

def split(dataset_name, split_method, overwrite=False, seed=42, add_seed_name=False):
    
    if add_seed_name:
        path = f"data/{dataset_name}/splits/{split_method}_seed{seed}.csv"
    else:
        path = f"data/{dataset_name}/splits/{split_method}.csv"
    if os.path.exists(path) and not overwrite:
        print(f"Split {split_method} already exists for dataset {dataset_name}. Skipping splitting.")
        return
    
    df = pd.read_csv(f"data/{dataset_name}/metadata.csv")
    all_smiles = df['smiles'].tolist()

    print(f"Splitting dataset {dataset_name} using method {split_method}...")
    if split_method == "as_provided":
        all_fold = df['fold'].tolist()
    elif split_method == "formula":
        formulas = compute_formulas(all_smiles)
        print('Splitting by formula...')
        all_fold = split_by_key(formulas, train_ratio=TRAIN_RATIO, test_ratio=TEST_RATIO, val_ratio=VAL_RATIO, seed=seed)
    elif split_method == "inchi":
        print('Computing InChI keys...')
        all_inchi = compute_inchi(all_smiles)
        print('Splitting by InChI key...')
        all_fold = split_by_key(all_inchi, train_ratio=TRAIN_RATIO, test_ratio=TEST_RATIO, val_ratio=VAL_RATIO, seed=seed)
    elif split_method == "murcko_hist":
        print('Computing Murcko histograms...')
        all_murcko_hist = compute_murcko_histograms(all_smiles)
        print('Splitting by Murcko histogram...')
        all_fold = split_by_key(all_murcko_hist, train_ratio=TRAIN_RATIO, test_ratio=TEST_RATIO, val_ratio=VAL_RATIO, seed=seed)
    elif split_method in ("murcko", "murcko_scaffold"):
        print('Computing Murcko scaffolds...')
        all_murcko_scaffolds = compute_murcko_scaffolds(all_smiles)
        print('Splitting by Murcko scaffold...')
        all_fold = split_by_key(all_murcko_scaffolds, train_ratio=TRAIN_RATIO, test_ratio=TEST_RATIO, val_ratio=VAL_RATIO, seed=seed)
    elif split_method == "random":
        print('Splitting randomly...')
        all_fold = split_by_key(np.arange(len(all_smiles)), train_ratio=TRAIN_RATIO, test_ratio=TEST_RATIO, val_ratio=VAL_RATIO, seed=seed)
    else:
        raise ValueError(f"Unknown splitting method: {split_method}")
    os.makedirs(f"data/{dataset_name}/splits", exist_ok=True)
    df_fold = pd.DataFrame({'fold': all_fold})
    df_fold.to_csv(path, index=False)
    print(f"Done. Split counts: {pd.Series(all_fold).value_counts().to_dict()}")
