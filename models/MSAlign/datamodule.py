"""Data loading for one spectrum and one molecular representation."""

from __future__ import annotations

import json
import math
from pathlib import Path

import h5py
import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from transforms.spectra_transforms import BIN_Transform

from .encoders import normalize_adduct


FINGERPRINT_WIDTHS = {"morgan_2_4096": 4096}


def _representation_name(value: object, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} must be a non-empty string.")
    return value.strip()


class SpectrumMoleculeDataset(Dataset):
    """Load labelled spectrum–molecule pairs without candidate sets.

    Each spectrum row is paired with the molecular representation indexed by
    ``metadata.csv::unique_smiles_idx``. Dense molecular embeddings are read
    from ``mol_embeddings`` and packed Morgan fingerprints from
    ``mol_fingerprints``.
    """

    def __init__(
        self,
        *,
        labelled_dataset_name: str,
        split_method: str,
        fold: str,
        ms_representation: str,
        mol_representation: str,
        spectra_bins: dict,
        data_root: str | Path = "data",
    ) -> None:
        super().__init__()
        self.ms_representation = _representation_name(
            ms_representation, "ms_representation"
        )
        self.mol_representation = _representation_name(
            mol_representation, "mol_representation"
        )

        root = Path(data_root) / labelled_dataset_name
        metadata = pd.read_csv(root / "metadata.csv")
        folds = pd.read_csv(root / "splits" / f"{split_method}.csv")["fold"]
        if len(metadata) != len(folds):
            raise ValueError("Metadata and split row counts differ.")
        if "unique_smiles_idx" not in metadata:
            raise ValueError("metadata.csv lacks unique_smiles_idx.")
        self.row_indices = np.flatnonzero((folds == fold).to_numpy())
        self.spectra_to_molecules = metadata["unique_smiles_idx"].to_numpy(
            dtype=np.int64
        )

        self.raw_spectra = None
        self.spectra_embeddings = None
        self.spectra_bin_transform = None
        if self.ms_representation == "bins":
            max_mz = float(spectra_bins["max_mz"])
            bin_width = float(spectra_bins["bin_width"])
            if max_mz <= 0 or bin_width <= 0:
                raise ValueError("Spectrum binning values must be positive.")
            self.raw_spectra = np.load(root / "spectra.npy", mmap_mode="r")
            self.spectra_bin_transform = BIN_Transform(
                max_mz=max_mz, bin_width=bin_width
            )
            self.d_ms_in = int(math.ceil(max_mz / bin_width))
        else:
            path = root / "spectra_embeddings" / f"{self.ms_representation}.npy"
            self.spectra_embeddings = np.load(path, mmap_mode="r")
            self.d_ms_in = int(self.spectra_embeddings.shape[-1])
        n_spectra = (
            len(self.raw_spectra)
            if self.raw_spectra is not None
            else len(self.spectra_embeddings)
        )
        if n_spectra != len(metadata):
            raise ValueError("Spectrum representation and metadata row counts differ.")

        self.is_fingerprint = self.mol_representation in FINGERPRINT_WIDTHS
        directory = "mol_fingerprints" if self.is_fingerprint else "mol_embeddings"
        path = root / directory / f"{self.mol_representation}.npy"
        self.molecule_representations = np.load(path, mmap_mode="r")
        n_molecules = len(pd.read_csv(root / "unique_smiles.csv"))
        if len(self.molecule_representations) != n_molecules:
            raise ValueError(
                "Molecular representation and unique_smiles.csv row counts differ."
            )
        if len(metadata) and (
            self.spectra_to_molecules.min() < 0
            or self.spectra_to_molecules.max() >= n_molecules
        ):
            raise ValueError("metadata.csv contains an invalid unique_smiles_idx.")
        if self.is_fingerprint:
            self.d_mol_in = FINGERPRINT_WIDTHS[self.mol_representation]
            packed_width = (self.d_mol_in + 7) // 8
            if self.molecule_representations.shape[-1] != packed_width:
                raise ValueError("Packed fingerprint width does not match its name.")
        else:
            self.d_mol_in = int(self.molecule_representations.shape[-1])

    def __len__(self) -> int:
        return len(self.row_indices)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = int(self.row_indices[index])
        molecule_index = int(self.spectra_to_molecules[row])
        if self.ms_representation == "bins":
            spectrum = np.asarray(self.spectra_bin_transform(self.raw_spectra[row]))
        else:
            spectrum = np.array(self.spectra_embeddings[row], copy=True)

        molecule = np.array(
            self.molecule_representations[molecule_index], copy=True
        )
        if self.is_fingerprint:
            molecule = np.unpackbits(
                molecule,
                count=self.d_mol_in,
                bitorder="little",
            )
        return {
            "spectrum": torch.from_numpy(spectrum).float(),
            "molecule": torch.from_numpy(molecule).float(),
        }


class CandidateDataset(Dataset):
    """Load a single representation for each modality.

    Representation-specific storage details end here: every returned spectrum
    and molecule is a dense floating-point vector.
    """

    def __init__(
        self,
        *,
        labelled_dataset_name: str,
        candidate_map_name: str,
        split_method: str,
        fold: str,
        k_candidates: int | None,
        ms_representation: str,
        mol_representation: str,
        use_energy: bool,
        use_adduct: bool,
        spectra_bins: dict,
        adduct_vocabulary: list[str] | None,
        data_root: str | Path = "data",
    ) -> None:
        super().__init__()
        self.ms_representation = _representation_name(
            ms_representation, "ms_representation"
        )
        self.mol_representation = _representation_name(
            mol_representation, "mol_representation"
        )
        if k_candidates is not None and k_candidates <= 0:
            raise ValueError("k_candidates must be positive or null.")
        self.k_candidates = k_candidates
        self.uses_energy = bool(use_energy)
        self.uses_adduct = bool(use_adduct)

        root = Path(data_root) / labelled_dataset_name
        candidate_root = root / "candidates" / candidate_map_name
        with (candidate_root / "map.json").open() as handle:
            self.candidate_map = json.load(handle)
        self.unique_smiles = pd.read_csv(root / "unique_smiles.csv")["smiles"].tolist()
        metadata = pd.read_csv(root / "metadata.csv")
        folds = pd.read_csv(root / "splits" / f"{split_method}.csv")["fold"]
        if len(metadata) != len(folds):
            raise ValueError("Metadata and split row counts differ.")
        self.row_indices = np.flatnonzero((folds == fold).to_numpy())
        self.spectra_to_smiles = metadata["unique_smiles_idx"].to_numpy()

        self.raw_spectra = None
        self.spectra_embeddings = None
        self.spectra_bin_transform = None
        if self.ms_representation == "bins":
            max_mz = float(spectra_bins["max_mz"])
            bin_width = float(spectra_bins["bin_width"])
            if max_mz <= 0 or bin_width <= 0:
                raise ValueError("Spectrum binning values must be positive.")
            self.raw_spectra = np.load(root / "spectra.npy", mmap_mode="r")
            self.spectra_bin_transform = BIN_Transform(
                max_mz=max_mz, bin_width=bin_width
            )
            self.d_ms_in = int(math.ceil(max_mz / bin_width))
        else:
            path = root / "spectra_embeddings" / f"{self.ms_representation}.npy"
            self.spectra_embeddings = np.load(path, mmap_mode="r")
            self.d_ms_in = int(self.spectra_embeddings.shape[-1])
        source_rows = (
            len(self.raw_spectra)
            if self.raw_spectra is not None
            else len(self.spectra_embeddings)
        )
        if source_rows != len(metadata):
            raise ValueError("Spectrum representation and metadata row counts differ.")

        self.mol_path = candidate_root / f"{self.mol_representation}.h5"
        self.is_fingerprint = self.mol_representation in FINGERPRINT_WIDTHS
        self.mol_key = (
            "candidate_fingerprints" if self.is_fingerprint else "candidates_embeddings"
        )
        with h5py.File(self.mol_path, "r") as handle:
            if "candidate_mask" not in handle or self.mol_key not in handle:
                raise ValueError(f"Invalid molecular representation cache: {self.mol_path}")
            if handle[self.mol_key].shape[:2] != handle["candidate_mask"].shape:
                raise ValueError("Molecular representation and candidate mask shapes differ.")
            if self.is_fingerprint:
                self.d_mol_in = FINGERPRINT_WIDTHS[self.mol_representation]
                packed_width = (self.d_mol_in + 7) // 8
                if handle[self.mol_key].shape[-1] != packed_width:
                    raise ValueError("Packed fingerprint width does not match its name.")
            else:
                self.d_mol_in = int(handle[self.mol_key].shape[-1])
        self._mol_handle = None

        self.collision_energy = None
        if self.uses_energy:
            self.collision_energy = (
                pd.to_numeric(metadata["collision_energy"], errors="coerce").to_numpy(
                    dtype=np.float32
                )
                if "collision_energy" in metadata
                else np.full(len(metadata), np.nan, dtype=np.float32)
            )

        self.adducts = None
        self.unknown_adduct_index = None
        if self.uses_adduct:
            if adduct_vocabulary is None:
                raise ValueError("adduct_vocabulary is required when adduct is enabled.")
            if "adduct" not in metadata:
                raise ValueError("Spectrum metadata must contain an adduct column.")
            lookup = {value: index for index, value in enumerate(adduct_vocabulary)}
            self.unknown_adduct_index = len(adduct_vocabulary)
            normalized = metadata["adduct"].map(normalize_adduct)
            self.adducts = np.asarray(
                [lookup.get(value, self.unknown_adduct_index) for value in normalized],
                dtype=np.int64,
            )

    def __len__(self) -> int:
        return len(self.row_indices)

    def _mol_h5(self):
        if self._mol_handle is None:
            self._mol_handle = h5py.File(self.mol_path, "r")
        return self._mol_handle

    def __getitem__(self, index: int) -> dict:
        row = int(self.row_indices[index])
        smiles_index = int(self.spectra_to_smiles[row])
        target = self.unique_smiles[smiles_index]
        candidates = self.candidate_map.get(target)
        if not candidates or candidates[0] != target:
            raise ValueError("Candidate zero must be the target SMILES.")

        handle = self._mol_h5()
        mask = np.asarray(handle["candidate_mask"][smiles_index], dtype=bool)
        valid = np.flatnonzero(mask)
        if len(valid) != len(candidates) or not mask[0]:
            raise ValueError(f"Candidate map/cache mismatch for target {target!r}.")

        selected = torch.from_numpy(valid.astype(np.int64))
        if self.k_candidates is not None:
            negatives = selected[selected != 0]
            selected = torch.cat(
                (
                    torch.zeros(1, dtype=torch.long),
                    negatives[torch.randperm(len(negatives))[: self.k_candidates - 1]],
                )
            )
        selected_numpy = selected.numpy()

        values = np.asarray(handle[self.mol_key][smiles_index])[selected_numpy]
        if self.is_fingerprint:
            values = np.unpackbits(
                values,
                axis=-1,
                count=self.d_mol_in,
                bitorder="little",
            )
        mol = torch.from_numpy(np.array(values, copy=True)).float()
        selected_mask = torch.ones(len(selected), dtype=torch.bool)
        if self.k_candidates is not None and len(selected) < self.k_candidates:
            padding = self.k_candidates - len(selected)
            selected_mask = torch.cat(
                (selected_mask, torch.zeros(padding, dtype=torch.bool))
            )
            mol = torch.cat((mol, mol.new_zeros(padding, self.d_mol_in)))

        if self.ms_representation == "bins":
            ms_values = np.asarray(self.spectra_bin_transform(self.raw_spectra[row]))
        else:
            ms_values = np.array(self.spectra_embeddings[row], copy=True)
        ms = torch.from_numpy(ms_values).float()

        metadata = {}
        if self.uses_energy:
            metadata["collision_energy"] = torch.tensor(
                self.collision_energy[row], dtype=torch.float32
            )
        if self.uses_adduct:
            metadata["adduct"] = torch.tensor(self.adducts[row], dtype=torch.long)
        return {
            "spectrum": ms,
            "candidates": mol,
            "metadata": metadata,
            "candidate_mask": selected_mask,
        }


def collate_candidates(batch: list[dict]) -> dict:
    """Stack spectra and pad the candidate dimension of molecule vectors."""

    max_candidates = max(len(sample["candidate_mask"]) for sample in batch)
    d_mol = batch[0]["candidates"].shape[-1]
    candidate_mask = torch.zeros(len(batch), max_candidates, dtype=torch.bool)
    mol = batch[0]["candidates"].new_zeros(len(batch), max_candidates, d_mol)
    for row, sample in enumerate(batch):
        size = len(sample["candidate_mask"])
        candidate_mask[row, :size] = sample["candidate_mask"]
        mol[row, :size] = sample["candidates"]

    metadata = {
        key: torch.stack([sample["metadata"][key] for sample in batch])
        for key in batch[0]["metadata"]
    }
    return {
        "spectrum": torch.stack([sample["spectrum"] for sample in batch]),
        "candidates": mol,
        "metadata": metadata,
        "candidate_mask": candidate_mask,
    }


class MSAlignDataModule(pl.LightningDataModule):
    """Build train/validation/test datasets and infer model input widths."""

    def __init__(
        self,
        *,
        labelled_dataset_name: str,
        candidate_map_name: str,
        split_method: str,
        k_candidates: int | None,
        ms_representation: str,
        mol_representation: str,
        use_energy: bool,
        use_adduct: bool,
        spectra_bins: dict,
        batch_size: int = 128,
        batch_size_test: int = 16,
        n_workers: int = 8,
        prefetch_factor: int = 2,
        data_root: str | Path = "data",
    ) -> None:
        super().__init__()
        metadata = pd.read_csv(Path(data_root) / labelled_dataset_name / "metadata.csv")
        if use_adduct:
            if "adduct" not in metadata:
                raise ValueError("Spectrum metadata must contain an adduct column.")
            self.adduct_vocabulary = sorted(
                set(metadata["adduct"].dropna().map(normalize_adduct))
            )
            if not self.adduct_vocabulary:
                raise ValueError("Adduct vocabulary is empty.")
        else:
            self.adduct_vocabulary = None

        common = dict(
            labelled_dataset_name=labelled_dataset_name,
            candidate_map_name=candidate_map_name,
            split_method=split_method,
            ms_representation=ms_representation,
            mol_representation=mol_representation,
            use_energy=use_energy,
            use_adduct=use_adduct,
            spectra_bins=spectra_bins,
            adduct_vocabulary=self.adduct_vocabulary,
            data_root=data_root,
        )
        self.train_dataset = CandidateDataset(
            fold="train", k_candidates=k_candidates, **common
        )
        self.val_dataset = CandidateDataset(fold="val", k_candidates=None, **common)
        self.test_dataset = CandidateDataset(fold="test", k_candidates=None, **common)
        self.batch_size = int(batch_size)
        self.batch_size_test = int(batch_size_test)
        self.n_workers = int(n_workers)
        self.prefetch_factor = int(prefetch_factor)

    @property
    def input_dimensions(self) -> dict[str, object]:
        inputs: dict[str, object] = {
            "d_ms": self.train_dataset.d_ms_in,
            "d_mol": self.train_dataset.d_mol_in,
        }
        if self.adduct_vocabulary is not None:
            signs = []
            for adduct in self.adduct_vocabulary:
                if adduct.endswith("+"):
                    signs.append(1)
                elif adduct.endswith("-"):
                    signs.append(-1)
                else:
                    raise ValueError(f"Cannot infer charge sign from adduct {adduct!r}.")
            inputs.update(
                n_adducts=len(self.adduct_vocabulary) + 1,
                unknown_adduct_index=len(self.adduct_vocabulary),
                adduct_signs=signs + [0],
                # Standalone inference needs the exact training-time mapping.
                adduct_vocabulary=list(self.adduct_vocabulary),
            )
        return inputs

    def _loader(self, dataset, batch_size: int, shuffle: bool) -> DataLoader:
        options = dict(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_candidates,
            num_workers=self.n_workers,
        )
        if self.n_workers:
            options.update(
                prefetch_factor=self.prefetch_factor,
                persistent_workers=True,
            )
        return DataLoader(**options)

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train_dataset, self.batch_size, True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.val_dataset, self.batch_size_test, False)

    def test_dataloader(self) -> DataLoader:
        return self._loader(self.test_dataset, self.batch_size_test, False)
