"""Sliced Wasserstein distance between paired train and test samples."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from scipy.stats import wasserstein_distance
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from models.MSAlign import SpectrumMoleculeDataset


SPECTRA_BINS = {"max_mz": 1005.0, "bin_width": 0.1}
N_RANDOM_PROJECTIONS = 100
RANDOM_PROJECTION_SEED = 42


def random_directions(dimension: int, count: int, seed: int) -> torch.Tensor:
    """Draw reproducible directions uniformly from the unit sphere."""

    if dimension <= 0 or count <= 0:
        raise ValueError("dimension and count must be positive")
    generator = torch.Generator().manual_seed(seed)
    directions = torch.randn(dimension, count, generator=generator)
    return directions / torch.linalg.vector_norm(directions, dim=0, keepdim=True)


def project_dataset(
    dataset: SpectrumMoleculeDataset,
    directions: torch.Tensor,
    *,
    batch_size: int,
    n_workers: int,
    description: str,
) -> np.ndarray:
    """Compute all one-dimensional projections without storing feature vectors."""

    if batch_size <= 0 or n_workers < 0:
        raise ValueError("batch_size must be positive and n_workers non-negative")
    expected_dimension = dataset.d_ms_in + dataset.d_mol_in
    if directions.ndim != 2 or directions.shape[0] != expected_dimension:
        raise ValueError(
            f"Expected directions of shape [{expected_dimension}, L], "
            f"got {tuple(directions.shape)}."
        )

    loader_options = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": n_workers,
    }
    if n_workers:
        loader_options.update(persistent_workers=True, prefetch_factor=2)
    loader = DataLoader(**loader_options)
    projections = np.empty((len(dataset), directions.shape[1]), dtype=np.float32)

    start = 0
    for batch in tqdm(loader, desc=description, unit="batch", dynamic_ncols=True):
        spectrum = batch["spectrum"].float()
        molecule = batch["molecule"].float()
        spectrum_norm = torch.linalg.vector_norm(spectrum, dim=1, keepdim=True)
        molecule_norm = torch.linalg.vector_norm(molecule, dim=1, keepdim=True)
        if torch.any(spectrum_norm == 0):
            raise ValueError("A spectrum representation has zero L2 norm.")
        if torch.any(molecule_norm == 0):
            raise ValueError("A molecular representation has zero L2 norm.")
        features = torch.cat(
            (spectrum / spectrum_norm, molecule / molecule_norm), dim=1
        )
        values = (features @ directions).numpy()
        end = start + len(values)
        projections[start:end] = values
        start = end
    return projections


def projection_distances(
    train_projections: np.ndarray, test_projections: np.ndarray
) -> np.ndarray:
    """Return the empirical Wasserstein-1 distance for every projection."""

    if train_projections.ndim != 2 or test_projections.ndim != 2:
        raise ValueError("Projection arrays must be two-dimensional.")
    if train_projections.shape[1] != test_projections.shape[1]:
        raise ValueError("Train and test arrays have different projection counts.")
    return np.asarray(
        [
            wasserstein_distance(train_projections[:, index], test_projections[:, index])
            for index in tqdm(
                range(train_projections.shape[1]),
                desc="Wasserstein distances",
                unit="projection",
                dynamic_ncols=True,
            )
        ],
        dtype=np.float64,
    )


def compute_swd(
    *,
    dataset: str,
    split: str,
    ms_representation: str,
    mol_representation: str,
    batch_size: int = 256,
    n_workers: int = 8,
    data_root: str | Path = "data",
) -> float:
    """Compute the mean projected Wasserstein-1 distance for one split."""

    common = {
        "labelled_dataset_name": dataset,
        "split_method": split,
        "ms_representation": ms_representation,
        "mol_representation": mol_representation,
        "spectra_bins": SPECTRA_BINS,
        "data_root": data_root,
    }
    train = SpectrumMoleculeDataset(fold="train", **common)
    test = SpectrumMoleculeDataset(fold="test", **common)
    if (train.d_ms_in, train.d_mol_in) != (test.d_ms_in, test.d_mol_in):
        raise ValueError("Train and test representation dimensions differ.")

    dimension = train.d_ms_in + train.d_mol_in
    print(
        f"Projecting {len(train):,} train and {len(test):,} test pairs from "
        f"R^{dimension} onto {N_RANDOM_PROJECTIONS} directions "
        f"(seed={RANDOM_PROJECTION_SEED}).",
        flush=True,
    )
    directions = random_directions(
        dimension, N_RANDOM_PROJECTIONS, RANDOM_PROJECTION_SEED
    )
    train_projections = project_dataset(
        train,
        directions,
        batch_size=batch_size,
        n_workers=n_workers,
        description=f"Train projections ({split})",
    )
    test_projections = project_dataset(
        test,
        directions,
        batch_size=batch_size,
        n_workers=n_workers,
        description=f"Test projections ({split})",
    )
    return float(projection_distances(train_projections, test_projections).mean())
