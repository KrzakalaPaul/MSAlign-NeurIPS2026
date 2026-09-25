"""Oracle molecular-formula filtering for evaluation only."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors


@lru_cache(maxsize=None)
def molecular_formula(smiles: str) -> str:
    """Compute the neutral candidate formula, including implicit hydrogens."""

    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"RDKit could not parse candidate SMILES: {smiles!r}.")
    return rdMolDescriptors.CalcMolFormula(Chem.AddHs(molecule))


def candidate_formula_mask(
    datasets_params: dict,
    fold: str,
    candidate_mask: torch.Tensor,
) -> torch.Tensor:
    """Return candidates sharing candidate zero's oracle molecular formula.

    Rows follow the same metadata/split order as every MSAlign scorer. The
    returned mask is always a subset of ``candidate_mask`` and always retains
    candidate zero.
    """

    if candidate_mask.ndim != 2 or candidate_mask.dtype != torch.bool:
        raise ValueError("candidate_mask must be boolean with shape [N, K].")
    dataset = datasets_params["labelled_dataset_name"]
    candidate_map_name = datasets_params["candidate_map_name"]
    split_method = datasets_params["split_method"]
    root = Path(datasets_params.get("data_root", "data")) / dataset

    with (root / "candidates" / candidate_map_name / "map.json").open() as handle:
        candidate_map = json.load(handle)
    unique_smiles = pd.read_csv(root / "unique_smiles.csv")["smiles"].tolist()
    metadata = pd.read_csv(root / "metadata.csv")
    folds = pd.read_csv(root / "splits" / f"{split_method}.csv")["fold"]
    if len(metadata) != len(folds):
        raise ValueError("Metadata and split row counts differ.")
    row_indices = np.flatnonzero((folds == fold).to_numpy())
    if len(row_indices) != candidate_mask.shape[0]:
        raise ValueError(
            "Formula-filter rows do not match the scorer output: "
            f"{len(row_indices)} != {candidate_mask.shape[0]}."
        )
    spectra_to_smiles = metadata["unique_smiles_idx"].to_numpy()

    output = torch.zeros_like(candidate_mask)
    for output_row, metadata_row in enumerate(row_indices):
        target = unique_smiles[int(spectra_to_smiles[metadata_row])]
        candidates = candidate_map.get(target)
        if not candidates or candidates[0] != target:
            raise ValueError("Candidate zero must be the target SMILES.")
        valid_count = int(candidate_mask[output_row].sum())
        if valid_count != len(candidates):
            raise ValueError(
                f"Candidate mask/map mismatch for {target!r}: "
                f"{valid_count} != {len(candidates)}."
            )
        target_formula = molecular_formula(target)
        keep = [molecular_formula(candidate) == target_formula for candidate in candidates]
        output[output_row, : len(keep)] = torch.tensor(keep, dtype=torch.bool)

    output &= candidate_mask
    if not torch.all(output[:, 0]):
        raise RuntimeError("The oracle formula filter removed a target candidate.")
    return output
