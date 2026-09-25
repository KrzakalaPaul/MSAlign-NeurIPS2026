"""Lightweight candidate-mass loading for model-independent scorers."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem.Descriptors import ExactMolWt
from torch.utils.data import Dataset

from models.MSAlign.encoders import normalize_adduct
from preprocessing.subformula_annotation.chem_utils import ion_to_mass, standardize_adduct


@lru_cache(maxsize=None)
def exact_mass(smiles: str) -> float:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"RDKit could not parse candidate SMILES: {smiles!r}.")
    return float(ExactMolWt(molecule))


def adduct_mass(adduct: object) -> float:
    """Return the signed ion-to-neutral mass offset, or NaN if unavailable."""

    if not isinstance(adduct, str) or not adduct.strip():
        return float("nan")
    try:
        canonical = standardize_adduct(normalize_adduct(adduct))
        return float(ion_to_mass[canonical])
    except (KeyError, TypeError, ValueError):
        return float("nan")


class CandidateMassDataset(Dataset):
    """Load candidate exact masses and metadata-derived target mass estimates."""

    def __init__(self, datasets_params: dict, fold: str) -> None:
        dataset = datasets_params["labelled_dataset_name"]
        candidate_map = datasets_params["candidate_map_name"]
        split = datasets_params["split_method"]
        root = Path(datasets_params.get("data_root", "data")) / dataset

        with (root / "candidates" / candidate_map / "map.json").open() as handle:
            self.candidate_map = json.load(handle)
        self.unique_smiles = pd.read_csv(root / "unique_smiles.csv")["smiles"].tolist()
        metadata = pd.read_csv(root / "metadata.csv")
        folds = pd.read_csv(root / "splits" / f"{split}.csv")["fold"]
        if len(metadata) != len(folds):
            raise ValueError("Metadata and split row counts differ.")
        self.row_indices = np.flatnonzero((folds == fold).to_numpy())
        self.spectra_to_smiles = metadata["unique_smiles_idx"].to_numpy()
        precursor_mz = (
            pd.to_numeric(metadata["precursor_mz"], errors="coerce").to_numpy(float)
            if "precursor_mz" in metadata
            else np.full(len(metadata), np.nan)
        )
        raw_adducts = (
            metadata["adduct"]
            if "adduct" in metadata
            else pd.Series([None] * len(metadata))
        )
        self.estimated_mass = precursor_mz - np.asarray(
            [adduct_mass(value) for value in raw_adducts], dtype=float
        )

    def __len__(self) -> int:
        return len(self.row_indices)

    def __getitem__(self, index: int) -> dict:
        row = int(self.row_indices[index])
        target = self.unique_smiles[int(self.spectra_to_smiles[row])]
        candidates = self.candidate_map.get(target)
        if not candidates or candidates[0] != target:
            raise ValueError("Candidate zero must be the target SMILES.")
        return {
            "candidate_mass": torch.tensor(
                [exact_mass(smiles) for smiles in candidates], dtype=torch.float32
            ),
            "candidate_mask": torch.ones(len(candidates), dtype=torch.bool),
            "estimated_mass": torch.tensor(
                self.estimated_mass[row], dtype=torch.float32
            ),
        }


def collate_mass_candidates(batch: list[dict]) -> dict:
    """Pad variable-length candidate mass vectors."""

    width = max(len(sample["candidate_mask"]) for sample in batch)
    candidate_mask = torch.zeros(len(batch), width, dtype=torch.bool)
    candidate_mass = torch.zeros(len(batch), width, dtype=torch.float32)
    for row, sample in enumerate(batch):
        size = len(sample["candidate_mask"])
        candidate_mask[row, :size] = sample["candidate_mask"]
        candidate_mass[row, :size] = sample["candidate_mass"]
    return {
        "candidate_mass": candidate_mass,
        "candidate_mask": candidate_mask,
        "estimated_mass": torch.stack([sample["estimated_mass"] for sample in batch]),
    }
