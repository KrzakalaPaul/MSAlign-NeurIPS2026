"""Read and verify portable MSAlign inference bundles."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import yaml


FORMAT_VERSION = 1
MODEL_TYPES = {"raw", "fusion"}


def sha256_file(path: str | Path) -> str:
    """Return the hexadecimal SHA-256 digest of one file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(bundle: str | Path, *, verify: bool = True) -> tuple[Path, dict]:
    """Load a bundle manifest and optionally verify its checkpoint files."""

    bundle = Path(bundle)
    manifest_path = bundle / "manifest.yaml"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"MSAlign bundle manifest not found: {manifest_path}")
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = yaml.safe_load(handle)
    if not isinstance(manifest, dict):
        raise ValueError("manifest.yaml must contain a mapping")
    if manifest.get("format_version") != FORMAT_VERSION:
        raise ValueError(
            f"Unsupported bundle format {manifest.get('format_version')!r}; "
            f"expected {FORMAT_VERSION}"
        )
    if manifest.get("model_type") not in MODEL_TYPES:
        raise ValueError("model_type must be 'raw' or 'fusion'")
    external = manifest.get("external_models")
    if not isinstance(external, dict):
        raise ValueError("manifest requires external_models")
    moldeberta = external.get("moldeberta")
    if not isinstance(moldeberta, dict) or not isinstance(
        moldeberta.get("revision"), str
    ):
        raise ValueError("manifest requires a pinned MolDeBERTa revision")
    scorers = manifest.get("scorers")
    if not isinstance(scorers, list) or not scorers:
        raise ValueError("manifest scorers must be a non-empty list")

    names = [scorer.get("name") for scorer in scorers if isinstance(scorer, dict)]
    if len(names) != len(scorers) or any(not isinstance(name, str) for name in names):
        raise ValueError("every scorer must have a string name")
    if len(names) != len(set(names)):
        raise ValueError("scorer names must be unique")

    for scorer in scorers:
        scorer_type = scorer.get("type")
        if scorer_type == "msalign":
            relative = scorer.get("checkpoint")
            expected = scorer.get("sha256")
            config_relative = scorer.get("config")
            config_expected = scorer.get("config_sha256")
            if not all(
                isinstance(value, str)
                for value in (relative, expected, config_relative, config_expected)
            ):
                raise ValueError(
                    "MSAlign scorers require checkpoint/config paths and checksums"
                )
            relative_path = Path(relative)
            config_relative_path = Path(config_relative)
            if (
                relative_path.is_absolute()
                or ".." in relative_path.parts
                or config_relative_path.is_absolute()
                or ".." in config_relative_path.parts
            ):
                raise ValueError(f"Invalid checkpoint path: {relative!r}")
            checkpoint = bundle / relative
            config = bundle / config_relative
            if not checkpoint.is_file():
                raise FileNotFoundError(f"Bundle checkpoint not found: {checkpoint}")
            if not config.is_file():
                raise FileNotFoundError(f"Bundle config not found: {config}")
            if verify and sha256_file(checkpoint) != expected:
                raise ValueError(f"Checkpoint checksum mismatch: {checkpoint}")
            if verify and sha256_file(config) != config_expected:
                raise ValueError(f"Config checksum mismatch: {config}")
        elif scorer_type == "estimated_mass":
            scale = scorer.get("scale_ppm")
            if not isinstance(scale, (int, float)) or scale <= 0:
                raise ValueError("Mass scorers require a positive scale_ppm")
        else:
            raise ValueError(f"Unknown scorer type: {scorer_type!r}")

    if manifest["model_type"] == "raw":
        if len(scorers) != 1 or scorers[0]["type"] != "msalign":
            raise ValueError("A raw bundle must contain one MSAlign scorer")
    else:
        weights = manifest.get("weights")
        if not isinstance(weights, dict) or set(weights) != set(names):
            raise ValueError("Fusion weights must match the scorer names exactly")
        values = [weights[name] for name in names]
        if any(
            not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
            for value in values
        ):
            raise ValueError("Fusion weights must be finite and non-negative")
        if not math.isclose(sum(values), 1.0, abs_tol=1e-5):
            raise ValueError("Fusion weights must sum to one")

    return bundle, manifest
