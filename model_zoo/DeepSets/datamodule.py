"""Data shared by the DeepSet and fingerprint-FFN retrieval baselines."""

from __future__ import annotations

from pathlib import Path

import h5py
import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


FINGERPRINT_NAME = "morgan_2_4096"
FINGERPRINT_WIDTH = 4096


def _normalized_peaks(
    spectrum: np.ndarray,
    *,
    mz_min: float,
    mz_max: float,
    n_peaks: int | None,
) -> np.ndarray:
    """Apply the peak filtering used by the MassSpecGym baselines."""

    spectrum = np.asarray(spectrum, dtype=np.float32)
    valid = (
        np.isfinite(spectrum[:, 0])
        & np.isfinite(spectrum[:, 1])
        & (spectrum[:, 0] >= mz_min)
        & (spectrum[:, 0] <= mz_max)
        & (spectrum[:, 1] > 0)
    )
    peaks = np.array(spectrum[valid], copy=True)
    if n_peaks is not None and len(peaks) > n_peaks:
        strongest = np.argpartition(peaks[:, 1], -n_peaks)[-n_peaks:]
        peaks = peaks[strongest]
    if len(peaks):
        peaks = peaks[np.argsort(peaks[:, 0])]
        maximum = float(peaks[:, 1].max())
        if maximum > 0:
            peaks[:, 1] /= maximum
    return peaks


def tokenize_peaks(
    spectrum: np.ndarray,
    precursor_mz: float,
    *,
    n_peaks: int,
    mz_min: float,
    mz_max: float,
    precursor_intensity: float,
) -> np.ndarray:
    """Return ``n_peaks`` normalized peaks plus one precursor token."""

    peaks = _normalized_peaks(
        spectrum, mz_min=mz_min, mz_max=mz_max, n_peaks=n_peaks
    )
    tokens = np.zeros((n_peaks + 1, 2), dtype=np.float32)
    tokens[0] = (
        float(precursor_mz) if np.isfinite(precursor_mz) else 0.0,
        float(precursor_intensity),
    )
    tokens[1 : len(peaks) + 1] = peaks
    return tokens


def bin_spectrum(
    spectrum: np.ndarray,
    *,
    bin_width: float,
    max_mz: float,
    mz_min: float,
) -> np.ndarray:
    """Reproduce MassSpecGym's relative-intensity spectrum binner."""

    peaks = _normalized_peaks(
        spectrum, mz_min=mz_min, mz_max=max_mz, n_peaks=None
    )
    n_bins = int(round(max_mz / bin_width))
    output = np.zeros(n_bins, dtype=np.float32)
    if len(peaks):
        indices = np.floor(peaks[:, 0] / bin_width).astype(np.int64)
        indices = np.clip(indices, 0, n_bins - 1)
        np.add.at(output, indices, peaks[:, 1])
        maximum = float(output.max())
        if maximum > 0:
            output /= maximum
    return output


class FingerprintRegressionDataset(Dataset):
    """Load spectra, target fingerprints, and optional candidate fingerprints."""

    def __init__(
        self,
        *,
        labelled_dataset_name: str,
        candidate_map_name: str,
        split_method: str,
        fold: str,
        spectrum_transform: str,
        spectrum_options: dict,
        include_candidates: bool,
        data_root: str | Path = "data",
    ) -> None:
        root = Path(data_root) / labelled_dataset_name
        metadata = pd.read_csv(root / "metadata.csv")
        folds = pd.read_csv(root / "splits" / f"{split_method}.csv")["fold"]
        if len(metadata) != len(folds):
            raise ValueError("Metadata and split row counts differ.")
        self.row_indices = np.flatnonzero((folds == fold).to_numpy())
        if not len(self.row_indices):
            raise ValueError(f"Fold {fold!r} is empty.")

        self.spectra = np.load(root / "spectra.npy", mmap_mode="r")
        if len(self.spectra) != len(metadata):
            raise ValueError("Spectra and metadata row counts differ.")
        self.spectra_to_smiles = metadata["unique_smiles_idx"].to_numpy(np.int64)
        if "precursor_mz" not in metadata:
            raise ValueError("Metadata must contain precursor_mz.")
        self.precursor_mz = pd.to_numeric(
            metadata["precursor_mz"], errors="coerce"
        ).to_numpy(np.float32)

        self.spectrum_transform = spectrum_transform
        self.spectrum_options = dict(spectrum_options)
        if spectrum_transform not in {"peaks", "bins"}:
            raise ValueError("spectrum_transform must be 'peaks' or 'bins'.")
        self.include_candidates = bool(include_candidates)

        self.fingerprint_path = (
            root / "candidates" / candidate_map_name / f"{FINGERPRINT_NAME}.h5"
        )
        with h5py.File(self.fingerprint_path, "r") as handle:
            required = {"candidate_fingerprints", "candidate_mask"}
            if not required.issubset(handle):
                raise ValueError(f"Invalid fingerprint cache: {self.fingerprint_path}")
            fingerprints = handle["candidate_fingerprints"]
            masks = handle["candidate_mask"]
            if fingerprints.shape[:2] != masks.shape:
                raise ValueError("Fingerprint values and masks have different shapes.")
            if fingerprints.shape[-1] != FINGERPRINT_WIDTH // 8:
                raise ValueError("Expected packed 4096-bit Morgan fingerprints.")
            if int(self.spectra_to_smiles.max()) >= fingerprints.shape[0]:
                raise ValueError("Fingerprint cache has too few molecule rows.")
        self._fingerprint_handle = None

    def __len__(self) -> int:
        return len(self.row_indices)

    def _fingerprints(self):
        if self._fingerprint_handle is None:
            self._fingerprint_handle = h5py.File(self.fingerprint_path, "r")
        return self._fingerprint_handle

    @staticmethod
    def _unpack(values: np.ndarray) -> torch.Tensor:
        unpacked = np.unpackbits(
            np.asarray(values, dtype=np.uint8),
            axis=-1,
            count=FINGERPRINT_WIDTH,
            bitorder="little",
        )
        return torch.from_numpy(np.array(unpacked, copy=True)).float()

    def _transform_spectrum(self, row: int) -> torch.Tensor:
        if self.spectrum_transform == "peaks":
            values = tokenize_peaks(
                self.spectra[row], self.precursor_mz[row], **self.spectrum_options
            )
        else:
            values = bin_spectrum(self.spectra[row], **self.spectrum_options)
        return torch.from_numpy(values).float()

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = int(self.row_indices[index])
        molecule = int(self.spectra_to_smiles[row])
        handle = self._fingerprints()
        mask = np.asarray(handle["candidate_mask"][molecule], dtype=bool)
        if not mask[0]:
            raise ValueError("Candidate zero must be the target molecule.")

        sample = {
            "spectrum": self._transform_spectrum(row),
            "target": self._unpack(
                np.asarray(handle["candidate_fingerprints"][molecule, 0])
            ),
        }
        if self.include_candidates:
            valid = np.flatnonzero(mask)
            sample["candidates"] = self._unpack(
                np.asarray(handle["candidate_fingerprints"][molecule, valid])
            )
            sample["candidate_mask"] = torch.ones(len(valid), dtype=torch.bool)
        return sample


def collate_fingerprint_regression(batch: list[dict]) -> dict[str, torch.Tensor]:
    output = {
        "spectrum": torch.stack([sample["spectrum"] for sample in batch]),
        "target": torch.stack([sample["target"] for sample in batch]),
    }
    if "candidates" not in batch[0]:
        return output

    max_candidates = max(len(sample["candidates"]) for sample in batch)
    candidates = output["target"].new_zeros(
        len(batch), max_candidates, FINGERPRINT_WIDTH
    )
    candidate_mask = torch.zeros(len(batch), max_candidates, dtype=torch.bool)
    for index, sample in enumerate(batch):
        count = len(sample["candidates"])
        candidates[index, :count] = sample["candidates"]
        candidate_mask[index, :count] = True
    output.update(candidates=candidates, candidate_mask=candidate_mask)
    return output


class FingerprintRegressionDataModule(pl.LightningDataModule):
    """Positive-pair training and full candidate evaluation."""

    def __init__(
        self,
        *,
        labelled_dataset_name: str,
        candidate_map_name: str,
        split_method: str,
        spectrum_transform: str,
        spectrum_options: dict,
        batch_size: int,
        batch_size_test: int,
        n_workers: int,
        prefetch_factor: int = 2,
        data_root: str | Path = "data",
    ) -> None:
        super().__init__()
        common = dict(
            labelled_dataset_name=labelled_dataset_name,
            candidate_map_name=candidate_map_name,
            split_method=split_method,
            spectrum_transform=spectrum_transform,
            spectrum_options=spectrum_options,
            data_root=data_root,
        )
        self.train_dataset = FingerprintRegressionDataset(
            fold="train", include_candidates=False, **common
        )
        self.val_dataset = FingerprintRegressionDataset(
            fold="val", include_candidates=True, **common
        )
        self.test_dataset = FingerprintRegressionDataset(
            fold="test", include_candidates=True, **common
        )
        self.batch_size = int(batch_size)
        self.batch_size_test = int(batch_size_test)
        self.n_workers = int(n_workers)
        self.prefetch_factor = int(prefetch_factor)

    def _loader(self, dataset, *, batch_size: int, shuffle: bool) -> DataLoader:
        options = dict(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_fingerprint_regression,
            num_workers=self.n_workers,
        )
        if self.n_workers:
            options.update(
                prefetch_factor=self.prefetch_factor,
                persistent_workers=True,
            )
        return DataLoader(**options)

    def train_dataloader(self):
        return self._loader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True
        )

    def val_dataloader(self):
        return self._loader(
            self.val_dataset, batch_size=self.batch_size_test, shuffle=False
        )

    def test_dataloader(self):
        return self._loader(
            self.test_dataset, batch_size=self.batch_size_test, shuffle=False
        )
