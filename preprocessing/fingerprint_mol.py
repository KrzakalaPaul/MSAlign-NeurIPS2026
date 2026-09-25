"""Precompute compact molecular fingerprints on CPU.

The public helpers mirror the molecular-embedding preprocessing API:

* :func:`get_molecule_fingerprint` encodes ``unique_smiles.csv``;
* :func:`get_molecule_fingerprint_for_candidates` encodes ``map.json`` into
  the same padded row layout and mask used by candidate molecular embeddings.

Fingerprints are bit-packed before storage. ``morgan_2_4096`` therefore uses
512 uint8 values per molecule instead of 4096 float32 values.
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import h5py
import numpy as np
import pandas as pd
import rdkit
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from tqdm import tqdm


RDLogger.DisableLog("rdApp.*")


@dataclass(frozen=True)
class FingerprintSpec:
    """Complete definition of a supported fingerprint representation."""

    name: str
    radius: int
    n_bits: int
    include_chirality: bool = False
    use_features: bool = False
    storage_dtype: str = "uint8"
    packed: bool = True
    bitorder: str = "little"

    @property
    def packed_size(self) -> int:
        return (self.n_bits + 7) // 8


FINGERPRINT_SPECS = {
    "morgan_2_4096": FingerprintSpec(
        name="morgan_2_4096",
        radius=2,
        n_bits=4096,
    ),
}


_WORKER_GENERATOR = None
_WORKER_SPEC: FingerprintSpec | None = None


def resolve_fingerprint(fingerprint: str) -> FingerprintSpec:
    """Resolve a clear public fingerprint name to its immutable definition."""

    try:
        return FINGERPRINT_SPECS[fingerprint]
    except KeyError as exc:
        raise ValueError(
            f"Unknown fingerprint {fingerprint!r}; choose one of "
            f"{sorted(FINGERPRINT_SPECS)}."
        ) from exc


def _make_generator(spec: FingerprintSpec):
    atom_invariants = (
        rdFingerprintGenerator.GetMorganFeatureAtomInvGen()
        if spec.use_features
        else None
    )
    return rdFingerprintGenerator.GetMorganGenerator(
        radius=spec.radius,
        fpSize=spec.n_bits,
        includeChirality=spec.include_chirality,
        atomInvariantsGenerator=atom_invariants,
    )


def _init_worker(fingerprint: str) -> None:
    global _WORKER_GENERATOR, _WORKER_SPEC
    _WORKER_SPEC = resolve_fingerprint(fingerprint)
    _WORKER_GENERATOR = _make_generator(_WORKER_SPEC)


def _encode_one(smiles: str) -> np.ndarray:
    if _WORKER_GENERATOR is None or _WORKER_SPEC is None:
        raise RuntimeError("Fingerprint worker was not initialized.")
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles!r}.")
    bit_vector = _WORKER_GENERATOR.GetFingerprint(molecule)
    unpacked = np.zeros(_WORKER_SPEC.n_bits, dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(bit_vector, unpacked)
    return np.packbits(unpacked, bitorder=_WORKER_SPEC.bitorder)


class _FingerprintPool:
    """Persistent CPU fingerprint workers with deterministic output ordering."""

    def __init__(self, fingerprint: str, n_workers: int, batch_size: int) -> None:
        if n_workers < 1:
            raise ValueError("n_workers must be positive.")
        if batch_size < 1:
            raise ValueError("batch_size must be positive.")
        self.fingerprint = fingerprint
        self.n_workers = int(n_workers)
        self.batch_size = int(batch_size)
        self.pool = None

    def __enter__(self) -> "_FingerprintPool":
        if self.n_workers == 1:
            _init_worker(self.fingerprint)
        else:
            start_method = "fork" if "fork" in mp.get_all_start_methods() else "spawn"
            self.pool = mp.get_context(start_method).Pool(
                processes=self.n_workers,
                initializer=_init_worker,
                initargs=(self.fingerprint,),
            )
        return self

    def encode(
        self,
        smiles: Sequence[str],
        *,
        description: str | None = None,
    ) -> np.ndarray:
        """Encode SMILES in order while avoiding duplicate work within a call."""

        smiles = list(smiles)
        spec = resolve_fingerprint(self.fingerprint)
        if not smiles:
            return np.empty((0, spec.packed_size), dtype=np.uint8)
        unique_smiles = list(dict.fromkeys(smiles))
        if self.pool is None:
            values = unique_smiles
            if description is not None:
                values = tqdm(
                    values,
                    desc=description,
                    unit="molecule",
                    dynamic_ncols=True,
                )
            unique_encoded = [_encode_one(value) for value in values]
        else:
            values = self.pool.imap(
                _encode_one,
                unique_smiles,
                chunksize=self.batch_size,
            )
            if description is not None:
                values = tqdm(
                    values,
                    total=len(unique_smiles),
                    desc=description,
                    unit="molecule",
                    dynamic_ncols=True,
                )
            unique_encoded = list(
                values
            )
        encoded = np.stack(unique_encoded).astype(np.uint8, copy=False)
        indices = {value: index for index, value in enumerate(unique_smiles)}
        return encoded[
            np.fromiter(
                (indices[value] for value in smiles),
                dtype=np.int64,
                count=len(smiles),
            )
        ]

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self.pool is not None:
            self.pool.close()
            self.pool.join()


def unpack_fingerprints(
    packed: np.ndarray,
    *,
    fingerprint: str = "morgan_2_4096",
) -> np.ndarray:
    """Expand stored fingerprints to a final uint8 feature dimension."""

    spec = resolve_fingerprint(fingerprint)
    packed = np.asarray(packed, dtype=np.uint8)
    if packed.shape[-1] != spec.packed_size:
        raise ValueError(
            f"Expected packed width {spec.packed_size}, got {packed.shape[-1]}."
        )
    return np.unpackbits(
        packed,
        axis=-1,
        count=spec.n_bits,
        bitorder=spec.bitorder,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _metadata(spec: FingerprintSpec, *, source_path: Path) -> dict:
    metadata = {
        "fingerprint": asdict(spec),
        "source_path": str(source_path),
        "source_sha256": _sha256_file(source_path),
        "rdkit_version": getattr(rdkit, "__version__", None),
        "storage_shape_per_molecule": [spec.packed_size],
    }
    metadata["cache_key"] = hashlib.sha256(
        json.dumps(metadata, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return metadata


def _metadata_matches(path: Path, expected: dict) -> bool:
    try:
        cached = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return cached.get("cache_key") == expected.get("cache_key")


def get_molecule_fingerprint(
    dataset_name: str,
    fingerprint: str = "morgan_2_4096",
    batch_size: int = 256,
    n_workers: int = 8,
    overwrite: bool = False,
    data_root: str | Path = "data",
) -> None:
    """Precompute packed fingerprints for the labeled molecule vocabulary."""

    spec = resolve_fingerprint(fingerprint)
    root = Path(data_root) / dataset_name
    source_path = root / "unique_smiles.csv"
    output_path = root / "mol_fingerprints" / f"{spec.name}.npy"
    metadata_path = output_path.with_suffix(".metadata.json")
    expected = _metadata(spec, source_path=source_path)

    if output_path.exists() and not overwrite:
        if _metadata_matches(metadata_path, expected):
            print(
                f"Molecule fingerprints already exist for {dataset_name} "
                f"with fingerprint {spec.name}. Skipping."
            )
            return
        raise RuntimeError(
            f"Existing cache {output_path} is incompatible. "
            "Use --overwrite to regenerate it."
        )

    smiles = pd.read_csv(source_path)["smiles"].tolist()
    print(
        f"Computing {spec.name} fingerprints for {len(smiles):,} labeled "
        f"molecules in {dataset_name} using {n_workers} CPU worker(s)."
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with _FingerprintPool(spec.name, n_workers, batch_size) as pool:
        packed = pool.encode(
            smiles,
            description=f"Morgan fingerprints ({dataset_name})",
        )
    if packed.shape != (len(smiles), spec.packed_size):
        raise RuntimeError(f"Unexpected fingerprint shape: {packed.shape}.")

    temporary_path = output_path.with_suffix(".tmp.npy")
    np.save(temporary_path, packed)
    os.replace(temporary_path, output_path)
    metadata_path.write_text(
        json.dumps(expected, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(smiles):,} packed fingerprints to {output_path}")


def get_molecule_fingerprint_for_candidates(
    dataset_name: str,
    candidate_map_name: str,
    fingerprint: str = "morgan_2_4096",
    batch_size: int = 256,
    n_workers: int = 8,
    overwrite: bool = False,
    chunk_size: int = 32,
    data_root: str | Path = "data",
) -> None:
    """Precompute packed fingerprints in candidate-map row order."""

    if chunk_size < 1:
        raise ValueError("chunk_size must be positive.")
    spec = resolve_fingerprint(fingerprint)
    root = Path(data_root) / dataset_name
    candidate_root = root / "candidates" / candidate_map_name
    map_path = candidate_root / "map.json"
    output_path = candidate_root / f"{spec.name}.h5"
    expected = _metadata(spec, source_path=map_path)

    if output_path.exists() and not overwrite:
        try:
            with h5py.File(output_path, "r") as existing:
                cached = json.loads(existing.attrs["metadata_json"])
                complete = bool(existing.attrs.get("encoding_complete", False))
        except (OSError, KeyError, TypeError, json.JSONDecodeError):
            cached, complete = None, False
        if cached and cached.get("cache_key") == expected["cache_key"] and complete:
            print(
                f"Candidate fingerprints already exist for {dataset_name} "
                f"with fingerprint {spec.name}. Skipping."
            )
            return
        if complete:
            raise RuntimeError(
                f"Existing cache {output_path} is incompatible. "
                "Use --overwrite to regenerate it."
            )
        print(f"Incomplete candidate cache {output_path} found. Regenerating it.")

    with map_path.open(encoding="utf-8") as handle:
        candidate_map = json.load(handle)
    smiles = pd.read_csv(root / "unique_smiles.csv")["smiles"].tolist()
    missing = [value for value in smiles if value not in candidate_map]
    if missing:
        raise KeyError(f"Candidate map is missing {len(missing)} labeled molecules.")
    n_candidates = max((len(candidate_map[value]) for value in smiles), default=0)
    if not smiles or n_candidates == 0:
        raise ValueError("Candidate map must contain at least one non-empty row.")

    print(
        f"Computing {spec.name} candidate fingerprints for {len(smiles):,} "
        f"targets in {dataset_name}/{candidate_map_name} "
        f"(maximum width={n_candidates}, workers={n_workers})."
    )

    candidate_root.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(".tmp.h5")
    if temporary_path.exists():
        temporary_path.unlink()
    try:
        with _FingerprintPool(spec.name, n_workers, batch_size) as pool, h5py.File(
            temporary_path, "w"
        ) as file:
            fingerprints = file.create_dataset(
                "candidate_fingerprints",
                shape=(len(smiles), n_candidates, spec.packed_size),
                dtype=np.uint8,
                chunks=(
                    min(chunk_size, len(smiles)),
                    n_candidates,
                    spec.packed_size,
                ),
            )
            masks = file.create_dataset(
                "candidate_mask",
                shape=(len(smiles), n_candidates),
                dtype=bool,
                chunks=(min(chunk_size, len(smiles)), n_candidates),
            )
            file.attrs["metadata_json"] = json.dumps(expected, sort_keys=True)
            file.attrs["encoding_complete"] = False

            for start in tqdm(
                range(0, len(smiles), chunk_size),
                desc=(
                    f"Fingerprinting candidates, dataset={dataset_name}, "
                    f"candidate_map={candidate_map_name}"
                ),
                unit="chunk",
                dynamic_ncols=True,
            ):
                source_smiles = smiles[start : start + chunk_size]
                rows: list[list[str]] = []
                flattened: list[str] = []
                mask_buffer = np.zeros((len(source_smiles), n_candidates), dtype=bool)
                for row, source in enumerate(source_smiles):
                    candidates = candidate_map[source]
                    if not candidates or candidates[0] != source:
                        raise ValueError(
                            "Candidate map must put the original molecule at index 0 "
                            f"for {source!r}."
                        )
                    if source in candidates[1:]:
                        raise ValueError(
                            f"Original molecule {source!r} is duplicated in candidates."
                        )
                    rows.append(candidates)
                    flattened.extend(candidates)
                    mask_buffer[row, : len(candidates)] = True

                encoded = pool.encode(flattened)
                packed_buffer = np.zeros(
                    (len(source_smiles), n_candidates, spec.packed_size),
                    dtype=np.uint8,
                )
                offset = 0
                for row, candidates in enumerate(rows):
                    count = len(candidates)
                    packed_buffer[row, :count] = encoded[offset : offset + count]
                    offset += count

                stop = start + len(source_smiles)
                fingerprints[start:stop] = packed_buffer
                masks[start:stop] = mask_buffer
            file.attrs["encoding_complete"] = True
        os.replace(temporary_path, output_path)
        print(f"Wrote packed candidate fingerprints to {output_path}")
    except BaseException:
        if temporary_path.exists():
            temporary_path.unlink()
        raise


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Precompute compact CPU molecular fingerprints."
    )
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--candidate_map_name")
    parser.add_argument(
        "--fingerprint",
        choices=tuple(FINGERPRINT_SPECS),
        default="morgan_2_4096",
    )
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--chunk_size", type=int, default=32)
    parser.add_argument("--n_workers", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    get_molecule_fingerprint(
        dataset_name=args.dataset_name,
        fingerprint=args.fingerprint,
        batch_size=args.batch_size,
        n_workers=args.n_workers,
        overwrite=args.overwrite,
    )
    if args.candidate_map_name:
        get_molecule_fingerprint_for_candidates(
            dataset_name=args.dataset_name,
            candidate_map_name=args.candidate_map_name,
            fingerprint=args.fingerprint,
            batch_size=args.batch_size,
            n_workers=args.n_workers,
            overwrite=args.overwrite,
            chunk_size=args.chunk_size,
        )
