"""Molecule-centred four-view data loading for MVP."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import re

import dgl
import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from preprocessing.definitions import CHEM_ELEMS_MS, CHEM_ELEMS_MS_ABUNDANCE
from transforms.molecules_transforms import MoleculeToGraph, MorganFingerprintTransform


def _formula_counts(formula: str) -> np.ndarray:
    counts = {element: 0 for element in CHEM_ELEMS_MS}
    for element, number in re.findall(r"([A-Z][a-z]?)(\d*)", formula or ""):
        if element in counts:
            counts[element] += int(number or 1)
    return np.asarray([counts[element] for element in CHEM_ELEMS_MS], dtype=np.float32)


class SpectrumTransform:
    """Create the published FormSpec tokens or the BinnedSpec vector."""

    def __init__(self, mode: str, max_mz: float, bin_width: float, max_peaks: int = 60):
        if mode not in {"formula", "bins"}:
            raise ValueError("spectrum_mode must be 'formula' or 'bins'.")
        self.mode, self.max_mz, self.bin_width, self.max_peaks = mode, max_mz, bin_width, max_peaks
        self.d_bins = int(np.ceil(max_mz / bin_width))

    def __call__(self, mz, intensity, formulas=None) -> torch.Tensor:
        mz = np.asarray([] if mz is None else mz, dtype=np.float32)
        intensity = np.asarray([] if intensity is None else intensity, dtype=np.float32)
        if len(mz):
            keep = np.argsort(intensity)[-self.max_peaks:]
            mz, intensity = mz[keep], intensity[keep]
            formulas = None if formulas is None else [formulas[index] for index in keep]
        if self.mode == "bins":
            result = np.zeros(self.d_bins, dtype=np.float32)
            valid = (mz >= 0) & (mz <= self.max_mz)
            bins = np.minimum((mz[valid] / self.bin_width).astype(int), self.d_bins - 1)
            np.add.at(result, bins, intensity[valid])
            maximum = result.max(initial=0.0)
            if maximum > 0:
                result = np.log10(result / maximum * 999.0 + 1.0) / 3.0
            return torch.from_numpy(result)
        if not formulas or not len(mz):
            return torch.zeros((1, len(CHEM_ELEMS_MS) + 1), dtype=torch.float32)
        abundance = np.asarray(CHEM_ELEMS_MS_ABUNDANCE, dtype=np.float32)
        tokens = np.stack([np.r_[_formula_counts(formula) / abundance, value] for formula, value in zip(formulas, intensity)])
        return torch.from_numpy(tokens.astype(np.float32))


class MVPDataStore:
    """Raw arrays shared by the train, validation, and test datasets.

    In particular, the expanded Python representation of ``annotated_peaks``
    is much larger than its JSON file. Loading it once instead of once per fold
    is essential for SpectraVerse.
    """

    def __init__(self, root: Path, dataset: str, split_method: str, need_annotations: bool):
        self.dataset_root = root / dataset
        self.metadata = pd.read_csv(self.dataset_root / "metadata.csv")
        self.split = pd.read_csv(
            self.dataset_root / "splits" / f"{split_method}.csv"
        )["fold"].to_numpy()
        if len(self.split) != len(self.metadata):
            raise ValueError("Split and metadata lengths differ.")
        self.spectra = np.load(self.dataset_root / "spectra.npy", mmap_mode="r")
        self.annotations = None
        if need_annotations:
            with (self.dataset_root / "annotated_peaks.json").open() as handle:
                self.annotations = json.load(handle)

    def indices(self, fold: str) -> np.ndarray:
        return np.flatnonzero(self.split == fold)


def _shared_data(store: MVPDataStore, fold: str):
    if len(store.split) != len(store.metadata):
        raise ValueError("Split and metadata lengths differ.")
    return store.metadata, store.indices(fold), store.spectra, store.annotations


class MVPPairDataset(Dataset):
    """One molecule per item, with one rotating individual spectrum."""

    def __init__(self, store, fold, spectrum_mode, max_mz, bin_width):
        self.metadata, indices, self.spectra, self.annotations = _shared_data(store, fold)
        self.groups: dict[str, list[int]] = defaultdict(list)
        for index in indices:
            self.groups[str(self.metadata.iloc[index]["smiles"])].append(int(index))
        self.smiles = sorted(self.groups)
        self.counters = defaultdict(int)
        self.spectrum_transform = SpectrumTransform(spectrum_mode, max_mz, bin_width)
        self.graph_transform = MoleculeToGraph()
        self.fp_transform = MorganFingerprintTransform(fp_size=1024, radius=5)
        self.consensus = {smiles: self._consensus(rows) for smiles, rows in self.groups.items()}

    def _raw(self, index):
        if self.spectrum_transform.mode == "formula":
            return self.annotations["mz"][index], self.annotations["intensities"][index], self.annotations["subformulas"][index]
        spectrum = np.asarray(self.spectra[index]); valid = spectrum[:, 1] > 0
        return spectrum[valid, 0], spectrum[valid, 1], None

    def _consensus(self, indices):
        if self.spectrum_transform.mode == "formula":
            merged = {}
            for index in indices:
                mz, intensity, formulas = self._raw(index)
                if formulas is None: continue
                for mass, value, formula in zip(mz, intensity, formulas):
                    previous = merged.get(formula)
                    if previous is None or value > previous[1]: merged[formula] = (mass, value)
            formulas = list(merged)
            return self.spectrum_transform([merged[f][0] for f in formulas], [merged[f][1] for f in formulas], formulas)
        mzs, intensities = [], []
        for index in indices:
            mz, intensity, _ = self._raw(index); mzs.extend(mz); intensities.extend(intensity)
        return self.spectrum_transform(mzs, intensities)

    def __len__(self): return len(self.smiles)

    def __getitem__(self, item):
        smiles = self.smiles[item]
        choices = self.groups[smiles]
        index = choices[self.counters[smiles] % len(choices)]
        self.counters[smiles] += 1
        return {
            "spectrum": self.spectrum_transform(*self._raw(index)),
            "consensus": self.consensus[smiles],
            "fingerprint": torch.from_numpy(self.fp_transform(smiles)).float(),
            "molecule": self.graph_transform(smiles),
        }


class MVPCandidateDataset(Dataset):
    def __init__(self, store, candidate_map, fold, spectrum_mode, max_mz, bin_width):
        self.metadata, indices, self.spectra, self.annotations = _shared_data(store, fold)
        self.indices = indices.tolist(); self.transform = SpectrumTransform(spectrum_mode, max_mz, bin_width)
        self.graph_transform = MoleculeToGraph()
        with (store.dataset_root / "candidates" / candidate_map / "map.json").open() as handle:
            self.candidates = json.load(handle)

    def __len__(self): return len(self.indices)

    def __getitem__(self, item):
        index = self.indices[item]; smiles = str(self.metadata.iloc[index]["smiles"])
        if self.transform.mode == "formula":
            spectrum = self.transform(self.annotations["mz"][index], self.annotations["intensities"][index], self.annotations["subformulas"][index])
        else:
            raw = np.asarray(self.spectra[index]); valid = raw[:, 1] > 0
            spectrum = self.transform(raw[valid, 0], raw[valid, 1])
        candidates = self.candidates[smiles]
        if not candidates or candidates[0] != smiles:
            raise ValueError("Candidate zero must be the target.")
        return spectrum, dgl.batch([self.graph_transform(candidate) for candidate in candidates])


def _collate_spectra(values):
    if values[0].ndim == 1:
        return torch.stack(values)
    length = max(value.shape[0] for value in values); width = values[0].shape[1]
    tokens = torch.zeros((len(values), length, width)); mask = torch.ones((len(values), length), dtype=torch.bool)
    for row, value in enumerate(values): tokens[row, :len(value)] = value; mask[row, :len(value)] = False
    return {"tokens": tokens, "padding_mask": mask}


def collate_pairs(batch):
    return {
        "spectrum": _collate_spectra([x["spectrum"] for x in batch]),
        "consensus": _collate_spectra([x["consensus"] for x in batch]),
        "fingerprint": torch.stack([x["fingerprint"] for x in batch]),
        "molecule": dgl.batch([x["molecule"] for x in batch]),
    }


def collate_candidates(batch):
    return _collate_spectra([x[0] for x in batch]), [x[1] for x in batch]


class MVPDataModule(pl.LightningDataModule):
    def __init__(self, *, labelled_dataset_name, candidate_map_name, split_method, config, n_workers=8, batch_size_test=16, evaluation_only=False):
        super().__init__(); data = config["data"]; spectrum = config["spectrum"]
        self.store = MVPDataStore(
            Path(data["root"]), labelled_dataset_name, split_method,
            need_annotations=config["spectrum_mode"] == "formula",
        )
        common = dict(store=self.store, spectrum_mode=config["spectrum_mode"],
                      max_mz=spectrum["max_mz"], bin_width=spectrum["bin_width"])
        self.train_dataset = None if evaluation_only else MVPPairDataset(fold="train", **common)
        self.val_dataset = None if evaluation_only else MVPPairDataset(fold="val", **common)
        self.test_dataset = MVPCandidateDataset(candidate_map=candidate_map_name, fold="test", **common)
        self.batch_size=config["training"]["batch_size"]; self.batch_size_test=batch_size_test; self.n_workers=n_workers

    def _options(self, persistent=False):
        result={"num_workers":self.n_workers};
        if self.n_workers: result.update(prefetch_factor=2, persistent_workers=persistent)
        return result
    def train_dataloader(self):
        if self.train_dataset is None: raise RuntimeError("Training data is disabled in evaluation-only mode.")
        return DataLoader(self.train_dataset,batch_size=self.batch_size,shuffle=True,collate_fn=collate_pairs,**self._options(True))
    def val_dataloader(self):
        if self.val_dataset is None: raise RuntimeError("Validation data is disabled in evaluation-only mode.")
        return DataLoader(self.val_dataset,batch_size=self.batch_size,shuffle=False,collate_fn=collate_pairs,**self._options())
    def test_dataloader(self): return DataLoader(self.test_dataset,batch_size=self.batch_size_test,shuffle=False,collate_fn=collate_candidates,**self._options())
