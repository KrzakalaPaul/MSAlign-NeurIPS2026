"""Export compact raw and fused MSAlign checkpoint bundles.

The command consumes the checkpoints and calibration CSV produced by the
standard training pipeline. One invocation writes a raw MSAlign ZIP and an
MSAlign+ ZIP for either ``formula_seed0`` or ``mces_1``.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import tempfile
import zipfile
from pathlib import Path

import torch
import yaml

from playground.bundle import FORMAT_VERSION, sha256_file
from preprocessing.encode_mol import MODEL_ID, MODEL_REVISION


DATASET = "massspecgym"
CANDIDATE_MAP = "official_candidates_by_mass"
DREAMS_REVISION = "dbec3a0b514a99e5056cfccde4559fda8cfe8129"
ALIGNMENT_SCORERS = (
    "dreams_moldeberta",
    "dreams_fingerprint",
    "bins_moldeberta",
    "bins_fingerprint",
)
MASS_SCALES = (0.01, 0.1, 1.0, 10.0)


def _load_checkpoint(path: Path) -> dict:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    hyperparameters = payload.get("hyper_parameters", {})
    config = payload.get("config", hyperparameters.get("config"))
    dimensions = payload.get(
        "input_dimensions", hyperparameters.get("input_dimensions")
    )
    if config is None or dimensions is None or "state_dict" not in payload:
        raise ValueError(f"Checkpoint lacks config, input dimensions, or weights: {path}")
    if "adduct_vocabulary" not in dimensions:
        raise ValueError(f"Checkpoint lacks its adduct vocabulary: {path}")
    return {
        "format_version": FORMAT_VERSION,
        "config": dict(config),
        "input_dimensions": dict(dimensions),
        "state_dict": payload["state_dict"],
    }


def _alignment_files(
    source: Path, checkpoint: Path, config: Path
) -> dict:
    """Write compact weights and their matching human-readable configuration."""

    payload = _load_checkpoint(source)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    config.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, checkpoint)
    config.write_text(
        yaml.safe_dump(payload["config"], sort_keys=False), encoding="utf-8"
    )
    return payload["config"]


def _scorer_entry(name: str, checkpoint: Path, config: Path, root: Path) -> dict:
    return {
        "name": name,
        "type": "msalign",
        "checkpoint": str(checkpoint.relative_to(root)),
        "sha256": sha256_file(checkpoint),
        "config": str(config.relative_to(root)),
        "config_sha256": sha256_file(config),
    }


def _result_rows(path: Path, split: str) -> dict[str, dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [
            row for row in csv.DictReader(handle)
            if row["labelled_dataset_name"] == DATASET
            and row["candidate_map_name"] == CANDIDATE_MAP
            and row["split_method"] == split
        ]
    indexed = {row["name"]: row for row in rows}
    required = {"MSAlign_fusion", *ALIGNMENT_SCORERS}
    missing = required - set(indexed)
    if missing:
        raise ValueError(f"Calibration CSV is missing rows: {sorted(missing)}")
    return indexed


def _metrics(row: dict) -> dict[str, float]:
    return {key: float(row[key]) for key in ("R@1", "R@5", "R@20")}


def _git_commit() -> str:
    return subprocess.check_output(
        ("git", "rev-parse", "HEAD"), text=True
    ).strip()


def _base_manifest(name: str, split: str, model_type: str, metrics: dict) -> dict:
    return {
        "format_version": FORMAT_VERSION,
        "name": name,
        "model_type": model_type,
        "external_models": {
            "dreams": {
                "name": "DreaMS_embedding",
                "source_revision": DREAMS_REVISION,
            },
            "moldeberta": {"name": MODEL_ID, "revision": MODEL_REVISION},
        },
        "provenance": {
            "dataset": DATASET,
            "candidate_map": CANDIDATE_MAP,
            "split": split,
            "source_commit": _git_commit(),
            "test_metrics": metrics,
        },
    }


def _mass_name(scale: float) -> str:
    return f"mass_ppm_{scale:g}"


def _write_manifest(path: Path, manifest: dict) -> None:
    path.write_text(
        yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8"
    )


def _zip_directory(directory: Path, output: Path) -> None:
    """Create a stable ZIP with the bundle directory as its top-level entry."""

    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(directory.rglob("*")):
            if not path.is_file():
                continue
            relative = Path(directory.name) / path.relative_to(directory)
            info = zipfile.ZipInfo(str(relative), date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, path.read_bytes())


def _bundle_names(split: str) -> tuple[str, str]:
    suffix = "formula_seed0" if split == "formula_seed0" else "mces_1"
    return f"msalign_massspecgym_{suffix}", f"msalign_star_massspecgym_{suffix}"


def export(args) -> list[dict]:
    rows = _result_rows(args.fusion_results, args.split)
    combined = rows["MSAlign_fusion"]
    weights = json.loads(combined["alpha"])
    expected_names = set(ALIGNMENT_SCORERS) | {
        _mass_name(scale) for scale in MASS_SCALES
    }
    if set(weights) != expected_names:
        raise ValueError(
            "Fusion weights do not match the released eight scorers: "
            f"{sorted(weights)}"
        )

    filename = f"{DATASET}__{CANDIDATE_MAP}__{args.split}.ckpt"
    sources = {
        name: args.checkpoints / name / filename for name in ALIGNMENT_SCORERS
    }
    missing = [path for path in sources.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing checkpoints:\n" + "\n".join(map(str, missing)))

    raw_name, fused_name = _bundle_names(args.split)
    args.output.mkdir(parents=True, exist_ok=True)
    inventory = []
    with tempfile.TemporaryDirectory(dir=args.output) as temporary:
        temporary = Path(temporary)

        raw_directory = temporary / raw_name
        raw_checkpoint = raw_directory / "checkpoints" / "dreams_moldeberta.ckpt"
        raw_config = raw_directory / "configs" / "dreams_moldeberta.yaml"
        _alignment_files(
            sources["dreams_moldeberta"], raw_checkpoint, raw_config
        )
        raw_manifest = _base_manifest(
            raw_name,
            args.split,
            "raw",
            _metrics(rows["dreams_moldeberta"]),
        )
        raw_manifest["scorers"] = [
            _scorer_entry(
                "dreams_moldeberta", raw_checkpoint, raw_config, raw_directory
            )
        ]
        _write_manifest(raw_directory / "manifest.yaml", raw_manifest)

        fused_directory = temporary / fused_name
        fused_scorers = []
        for name in ALIGNMENT_SCORERS:
            relative = Path("checkpoints") / f"{name}.ckpt"
            destination = fused_directory / relative
            config = fused_directory / "configs" / f"{name}.yaml"
            _alignment_files(sources[name], destination, config)
            fused_scorers.append(
                _scorer_entry(name, destination, config, fused_directory)
            )
        fused_scorers.extend(
            {
                "name": _mass_name(scale),
                "type": "estimated_mass",
                "scale_ppm": scale,
            }
            for scale in MASS_SCALES
        )
        fused_manifest = _base_manifest(
            fused_name, args.split, "fusion", _metrics(combined)
        )
        fused_manifest.update(
            scorers=fused_scorers,
            weights={name: float(weights[name]) for name in weights},
            selected_temperature=float(combined["selected_temperature"]),
        )
        _write_manifest(fused_directory / "manifest.yaml", fused_manifest)

        for name, directory in ((raw_name, raw_directory), (fused_name, fused_directory)):
            manifest = raw_manifest if name == raw_name else fused_manifest
            archive = args.output / f"{name}.zip"
            if archive.exists() and not args.overwrite:
                raise FileExistsError(f"Archive already exists: {archive}")
            temporary_archive = archive.with_suffix(".zip.tmp")
            _zip_directory(directory, temporary_archive)
            temporary_archive.replace(archive)
            inventory.append(
                {
                    "name": name,
                    "file": archive.name,
                    "size_bytes": archive.stat().st_size,
                    "sha256": sha256_file(archive),
                    "dataset": DATASET,
                    "candidate_map": CANDIDATE_MAP,
                    "split": args.split,
                    "source_commit": _git_commit(),
                    "configurations": {
                        scorer["name"]: yaml.safe_load(
                            (directory / scorer["config"]).read_text(encoding="utf-8")
                        )
                        for scorer in manifest["scorers"]
                        if scorer["type"] == "msalign"
                    },
                    "fusion_weights": manifest.get("weights"),
                    "selected_temperature": manifest.get("selected_temperature"),
                    "test_metrics": manifest["provenance"]["test_metrics"],
                }
            )
            print(f"Wrote {archive}")
    return inventory


def _update_inventory(path: Path, new_entries: list[dict]) -> None:
    if path.is_file():
        existing = yaml.safe_load(path.read_text(encoding="utf-8")) or []
    else:
        existing = []
    by_name = {entry["name"]: entry for entry in existing}
    by_name.update({entry["name"]: entry for entry in new_entries})
    path.write_text(
        yaml.safe_dump([by_name[name] for name in sorted(by_name)], sort_keys=False),
        encoding="utf-8",
    )
    print(f"Wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("formula_seed0", "mces_1"), required=True)
    parser.add_argument("--fusion-results", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, default=Path("checkpoints"))
    parser.add_argument("--output", type=Path, default=Path("releases"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    entries = export(args)
    _update_inventory(args.output / "inventory.yaml", entries)


if __name__ == "__main__":
    main()
