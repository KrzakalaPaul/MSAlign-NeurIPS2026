"""Persistent per-scorer score cache for MSAlign fusion."""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Callable, Mapping
from pathlib import Path

import torch


CACHE_FORMAT_VERSION = 1


def _jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _digest(value: object) -> str:
    encoded = json.dumps(
        _jsonable(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _safe_name(value: object) -> str:
    result = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value)).strip("-.")
    return result or "unnamed"


def _dataset_identity(datasets_params: Mapping[str, object]) -> dict:
    """Fingerprint score-relevant dataset files without reading their contents."""

    root = Path(datasets_params.get("data_root", "data")) / str(
        datasets_params["labelled_dataset_name"]
    )
    paths = [
        root / "metadata.csv",
        root / "unique_smiles.csv",
        root / "spectra.npy",
        root / "splits" / f"{datasets_params['split_method']}.csv",
    ]
    candidate_root = (
        root / "candidates" / str(datasets_params["candidate_map_name"])
    )
    if candidate_root.is_dir():
        paths.extend(path for path in candidate_root.rglob("*") if path.is_file())
    spectra_embeddings = root / "spectra_embeddings"
    if spectra_embeddings.is_dir():
        paths.extend(path for path in spectra_embeddings.glob("*.npy") if path.is_file())

    identity = {}
    for path in sorted(set(paths)):
        if path.is_file():
            stat = path.stat()
            identity[str(path.relative_to(root))] = {
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
    return identity


def _validate(scores: torch.Tensor, mask: torch.Tensor) -> None:
    if scores.ndim != 2 or not scores.is_floating_point():
        raise ValueError("Cached scores must be floating point with shape [N, K].")
    if mask.shape != scores.shape or mask.dtype != torch.bool:
        raise ValueError("Cached mask must be boolean and match the score shape.")
    if scores.shape[0] == 0 or scores.shape[1] == 0 or not torch.all(mask[:, 0]):
        raise ValueError("Cached scores must contain rows with valid candidate zero.")
    if not torch.all(torch.isfinite(scores[mask])):
        raise ValueError("Cached valid scores must be finite.")


class ScoreCache:
    """Load or atomically create one score tensor for one scorer and fold."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def _identity(self, scorer, datasets_params, fold: str) -> dict:
        return {
            "format_version": CACHE_FORMAT_VERSION,
            "scorer": scorer.cache_identity(datasets_params),
            "dataset": _jsonable(dict(datasets_params)),
            "dataset_files": _dataset_identity(datasets_params),
            "fold": fold,
        }

    def path(self, scorer, datasets_params, fold: str) -> Path:
        scorer_key = _digest(scorer.cache_params)[:16]
        target = "__".join(
            _safe_name(datasets_params[key])
            for key in (
                "labelled_dataset_name",
                "candidate_map_name",
                "split_method",
            )
        )
        return self.root / f"{_safe_name(scorer.name)}__{scorer_key}" / f"{target}__{fold}.pt"

    def get_or_compute(
        self,
        scorer,
        datasets_params: Mapping[str, object],
        fold: str,
        compute: Callable[[], tuple[torch.Tensor, torch.Tensor]],
        *,
        verbose: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        identity = self._identity(scorer, datasets_params, fold)
        path = self.path(scorer, datasets_params, fold)
        if path.is_file():
            try:
                payload = torch.load(path, map_location="cpu", weights_only=True)
                if payload.get("identity") == identity:
                    scores = payload["scores"]
                    mask = payload["mask"]
                    _validate(scores, mask)
                    if verbose:
                        print(f"    cache hit: {path}", flush=True)
                    return scores, mask
            except (KeyError, TypeError, ValueError, RuntimeError, EOFError):
                pass
            if verbose:
                print(f"    stale/invalid cache: {path}", flush=True)
        elif verbose:
            print(f"    cache miss: {path}", flush=True)

        scores, mask = compute()
        scores = scores.detach().cpu()
        mask = mask.detach().cpu()
        _validate(scores, mask)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(f"{path.suffix}.{os.getpid()}.tmp")
        torch.save(
            {"identity": identity, "scores": scores, "mask": mask}, temporary
        )
        os.replace(temporary, path)
        if verbose:
            print(f"    cached: {path}", flush=True)
        return scores, mask
