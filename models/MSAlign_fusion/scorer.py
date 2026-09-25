"""Modular candidate scorers used by MSAlign_fusion."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from models.MSAlign import CandidateDataset, MSAlign, collate_candidates
from models.MSAlign.encoders import normalize_adduct

from .data import CandidateMassDataset, collate_mass_candidates
from .mass import MassScore, score_mass_candidates


DATASET_KEYS = {"labelled_dataset_name", "candidate_map_name", "split_method", "data_root"}
RUNTIME_KEYS = {"batch_size", "n_workers", "device"}


def _validate_datasets_params(params: Mapping[str, object]) -> dict:
    required = {"labelled_dataset_name", "candidate_map_name", "split_method"}
    missing = required - set(params)
    unknown = set(params) - DATASET_KEYS
    if missing:
        raise ValueError(f"datasets_params is missing: {sorted(missing)}.")
    if unknown:
        raise ValueError(f"Unknown datasets_params: {sorted(unknown)}.")
    return dict(params)


def _device(value: str | torch.device) -> torch.device:
    if str(value) == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return device


def resolve_checkpoint(path: str | Path, datasets_params: Mapping[str, object]) -> Path:
    """Resolve one checkpoint file, optionally from a six-checkpoint run."""

    path = Path(path)
    if path.is_dir():
        name = "__".join(
            str(datasets_params[key])
            for key in ("labelled_dataset_name", "candidate_map_name", "split_method")
        )
        path = path / f"{name}.ckpt"
    if not path.is_file():
        raise FileNotFoundError(f"MSAlign checkpoint not found: {path}")
    return path


def _torch_load(path: Path) -> dict:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid checkpoint payload: {path}")
    return payload


def load_msalign_checkpoint(
    checkpoint: str | Path,
    datasets_params: Mapping[str, object],
    device: str | torch.device = "auto",
) -> tuple[MSAlign, dict, Path]:
    """Load and freeze one dataset-compatible MSAlign checkpoint."""

    path = resolve_checkpoint(checkpoint, datasets_params)
    payload = _torch_load(path)
    if "state_dict" not in payload:
        raise ValueError(f"Checkpoint {path} does not contain a state_dict.")
    hyperparameters = payload.get("hyper_parameters", {})
    config = hyperparameters.get("config")
    input_dimensions = hyperparameters.get("input_dimensions")
    # Standalone released checkpoints may store these fields at the top level.
    config = payload.get("config", config)
    input_dimensions = payload.get("input_dimensions", input_dimensions)
    if config is None or input_dimensions is None:
        raise ValueError(f"Checkpoint {path} lacks config or input dimensions.")
    model = MSAlign(config, input_dimensions)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.requires_grad_(False)
    model.eval()
    model.to(_device(device))
    return model, config, path


def _loader(dataset, batch_size: int, n_workers: int, collate_fn) -> DataLoader:
    if batch_size <= 0 or n_workers < 0:
        raise ValueError("batch_size must be positive and n_workers non-negative.")
    options = dict(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_workers,
        collate_fn=collate_fn,
    )
    if n_workers:
        options.update(persistent_workers=True, prefetch_factor=2)
    return DataLoader(**options)


def _batches(loader: DataLoader, *, verbose: bool, description: str):
    return (
        tqdm(loader, desc=description, unit="batch", dynamic_ncols=True)
        if verbose
        else loader
    )


def _to_device(value: Any, device: torch.device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _to_device(item, device) for key, item in value.items()}
    return value


def _pad_and_concatenate(
    score_batches: list[torch.Tensor], mask_batches: list[torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor]:
    if not score_batches:
        raise ValueError("The requested fold contains no samples.")
    width = max(batch.shape[1] for batch in score_batches)
    n_rows = sum(batch.shape[0] for batch in score_batches)
    scores = score_batches[0].new_zeros((n_rows, width))
    masks = torch.zeros((n_rows, width), dtype=torch.bool)
    start = 0
    for batch_scores, batch_mask in zip(score_batches, mask_batches):
        if batch_scores.shape != batch_mask.shape:
            raise ValueError("A scorer produced mismatched score and mask shapes.")
        end = start + batch_scores.shape[0]
        scores[start:end, : batch_scores.shape[1]] = batch_scores
        masks[start:end, : batch_mask.shape[1]] = batch_mask
        start = end
    return scores, masks


def _adduct_vocabulary(config: dict, datasets_params: Mapping[str, object]):
    if not config["model"].get("use_adduct", True):
        return None
    root = Path(
        datasets_params.get("data_root", config.get("data_root", "data"))
    )
    metadata = pd.read_csv(
        root / str(datasets_params["labelled_dataset_name"]) / "metadata.csv"
    )
    normalize = normalize_adduct
    vocabulary = sorted(set(metadata["adduct"].dropna().map(normalize)))
    return vocabulary


class Scorer:
    """Minimal interface implemented by every score source."""

    cache_params: dict[str, object]

    def cache_identity(self, datasets_params: Mapping[str, object]) -> dict:
        return dict(self.cache_params)

    def get_score(
        self,
        datasets_params: Mapping[str, object],
        fold: str,
        *,
        verbose: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError


class MSAlignScorer(Scorer):
    """Use the shared-space score from a frozen MSAlign checkpoint."""

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        checkpoint_split_method: str | None = None,
        batch_size: int = 16,
        n_workers: int = 0,
        device: str = "auto",
    ) -> None:
        self.checkpoint = Path(checkpoint)
        if checkpoint_split_method is not None and (
            not isinstance(checkpoint_split_method, str) or not checkpoint_split_method
        ):
            raise ValueError("checkpoint_split_method must be a non-empty string or null.")
        self.checkpoint_split_method = checkpoint_split_method
        self.batch_size = int(batch_size)
        self.n_workers = int(n_workers)
        self.device = device
        self._loaded_checkpoint = None
        self._loaded_model = None
        self._loaded_config = None

    def cache_identity(self, datasets_params: Mapping[str, object]) -> dict:
        path = resolve_checkpoint(self.checkpoint, self._checkpoint_params(datasets_params))
        stat = path.stat()
        return {
            **self.cache_params,
            "resolved_checkpoint": {
                "name": path.name,
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            },
        }

    def _model(self, datasets_params):
        checkpoint_params = self._checkpoint_params(datasets_params)
        path = resolve_checkpoint(self.checkpoint, checkpoint_params)
        if path != self._loaded_checkpoint:
            model, config, path = load_msalign_checkpoint(
                path, checkpoint_params, self.device
            )
            self._loaded_checkpoint = path
            self._loaded_model = model
            self._loaded_config = config
        return self._loaded_model, self._loaded_config

    def _checkpoint_params(self, datasets_params: Mapping[str, object]) -> dict:
        """Use a separately named training split when resolving a checkpoint."""

        params = dict(datasets_params)
        if self.checkpoint_split_method is not None:
            params["split_method"] = self.checkpoint_split_method
        return params

    @torch.inference_mode()
    def get_score(self, datasets_params, fold, *, verbose=False):
        params = _validate_datasets_params(datasets_params)
        model, config = self._model(params)
        model_config = config["model"]
        dataset = CandidateDataset(
            labelled_dataset_name=params["labelled_dataset_name"],
            candidate_map_name=params["candidate_map_name"],
            split_method=params["split_method"], fold=fold, k_candidates=None,
            ms_representation=config["representations"]["spectrum"],
            mol_representation=config["representations"]["molecule"],
            use_energy=model_config.get("use_collision_energy", True),
            use_adduct=model_config.get("use_adduct", True),
            spectra_bins=config.get("spectrum_bins", {"max_mz": 1005.0, "bin_width": 0.1}),
            adduct_vocabulary=_adduct_vocabulary(config, params),
            data_root=params.get("data_root", config.get("data_root", "data")),
        )
        scorer_collate = collate_candidates
        score_batches, mask_batches = [], []
        device = next(model.parameters()).device
        loader = _loader(dataset, self.batch_size, self.n_workers, scorer_collate)
        for batch in _batches(
            loader,
            verbose=verbose,
            description=f"{fold}: {getattr(self, 'name', type(self).__name__)}",
        ):
            batch = _to_device(batch, device)
            score_batches.append(model.forward_batch(batch).cpu())
            mask_batches.append(batch["candidate_mask"].cpu())
        return _pad_and_concatenate(score_batches, mask_batches)


class EstimatedMassScorer(Scorer):
    """Compare adduct-corrected precursor mass with candidate masses."""

    def __init__(self, mass_score, *, batch_size: int = 256, n_workers: int = 0) -> None:
        self.mass_score = MassScore(mass_score)
        self.batch_size = int(batch_size)
        self.n_workers = int(n_workers)

    @torch.inference_mode()
    def get_score(self, datasets_params, fold, *, verbose=False):
        params = _validate_datasets_params(datasets_params)
        dataset = CandidateMassDataset(params, fold)
        score_batches, mask_batches = [], []
        loader = _loader(dataset, self.batch_size, self.n_workers, collate_mass_candidates)
        for batch in _batches(
            loader,
            verbose=verbose,
            description=f"{fold}: {getattr(self, 'name', type(self).__name__)}",
        ):
            scores = score_mass_candidates(
                batch["estimated_mass"],
                batch["candidate_mass"],
                batch["candidate_mask"],
                self.mass_score,
            )
            score_batches.append(scores)
            mask_batches.append(batch["candidate_mask"])
        return _pad_and_concatenate(score_batches, mask_batches)


def get_scorer(params: Mapping[str, object]) -> Scorer:
    """Construct a scorer while keeping the interface open to new types."""

    params = dict(params)
    cache_params = {
        key: value for key, value in params.items() if key not in RUNTIME_KEYS
    }
    scorer_type = params.pop("type", None)
    name = params.pop("name", scorer_type)
    if not isinstance(name, str) or not name.strip():
        raise ValueError("A scorer name must be a non-empty string.")
    if scorer_type == "msalign":
        allowed = {"checkpoint", "checkpoint_split_method", *RUNTIME_KEYS}
        unknown = set(params) - allowed
        if unknown:
            raise ValueError(f"Unknown msalign scorer parameters: {sorted(unknown)}.")
        scorer = MSAlignScorer(**params)
    elif scorer_type == "estimated_mass":
        allowed = {"mass_score", "batch_size", "n_workers"}
        unknown = set(params) - allowed
        if unknown:
            raise ValueError(f"Unknown estimated_mass parameters: {sorted(unknown)}.")
        scorer = EstimatedMassScorer(**params)
    else:
        raise ValueError("type must be 'msalign' or 'estimated_mass'.")
    scorer.name = name
    scorer.scorer_type = scorer_type
    scorer.cache_params = cache_params
    return scorer
