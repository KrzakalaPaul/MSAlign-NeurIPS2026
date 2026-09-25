"""Precompute frozen MolDeBERTa representations.

The model is evaluated once and never fine-tuned by MSAlign. Candidate caches
share the same padded row order as ``map.json``; candidate zero is checked to be
the target before anything is written.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Sequence

import h5py
import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


MODEL_ID = "SaeedLab/MolDeBERTa-base-123M-mtr"
REPRESENTATION_NAME = "moldeberta_base_123m_mtr"
OUTPUT_DIMENSION = 768


def sequence_length_limit(tokenizer, model) -> int:
    """Return the shortest finite sequence limit declared by model or tokenizer."""

    limits = []
    for value in (
        getattr(model.config, "max_position_embeddings", None),
        getattr(tokenizer, "model_max_length", None),
    ):
        if isinstance(value, int) and 0 < value < 1_000_000:
            limits.append(value)
    if not limits:
        raise RuntimeError("MolDeBERTa does not declare a finite sequence limit")
    return min(limits)


def canonical_smiles(smiles: str) -> str:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"Invalid SMILES: {smiles!r}")
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


class MolDeBERTaEncoder:
    """Frozen MolDeBERTa using its first-token representation."""

    def __init__(self, device: str = "cuda", revision: str | None = None):
        token = os.environ.get("HF_TOKEN")
        if not token:
            raise RuntimeError("Set HF_TOKEN after accepting the MolDeBERTa model terms")
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("MolDeBERTa preprocessing requires an available GPU")
        self.device = torch.device(device)
        self.revision = revision
        print(f"Loading {MODEL_ID} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID, token=token, revision=revision
        )
        self.model = AutoModel.from_pretrained(
            MODEL_ID, token=token, revision=revision, use_safetensors=True
        ).eval().to(self.device)
        if int(self.model.config.hidden_size) != OUTPUT_DIMENSION:
            raise RuntimeError("Unexpected MolDeBERTa hidden dimension")
        self.max_length = sequence_length_limit(self.tokenizer, self.model)
        print(
            f"Loaded {MODEL_ID} ({OUTPUT_DIMENSION}-dimensional output, "
            f"maximum sequence length={self.max_length})."
        )

    @torch.inference_mode()
    def encode(
        self,
        smiles: Sequence[str],
        batch_size: int = 32,
        *,
        description: str | None = None,
    ) -> np.ndarray:
        """Encode SMILES in order, avoiding duplicate model evaluations."""

        processed = [canonical_smiles(value) for value in smiles]
        unique = list(dict.fromkeys(processed))
        encoded = []
        batches = range(0, len(unique), batch_size)
        if description is not None:
            batches = tqdm(
                batches,
                total=(len(unique) + batch_size - 1) // batch_size,
                desc=description,
                unit="batch",
                dynamic_ncols=True,
            )
        for start in batches:
            batch = self.tokenizer(
                unique[start : start + batch_size],
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            batch = {key: value.to(self.device) for key, value in batch.items()}
            vectors = self.model(**batch).last_hidden_state[:, 0]
            encoded.append(vectors.float().cpu().numpy())
        unique_vectors = np.concatenate(encoded).astype(np.float32, copy=False)
        lookup = {value: index for index, value in enumerate(unique)}
        return unique_vectors[[lookup[value] for value in processed]]


def _metadata(encoder: MolDeBERTaEncoder, smiles: Sequence[str]) -> dict:
    processed = [canonical_smiles(value) for value in smiles]
    metadata = {
        "model_id": MODEL_ID,
        "revision": encoder.revision,
        "representation": REPRESENTATION_NAME,
        "pooling": "last_hidden_state[:, 0]",
        "max_sequence_length": encoder.max_length,
        "output_dimension": OUTPUT_DIMENSION,
        "canonical_isomeric_smiles": True,
        "l2_normalized": False,
        "dtype": "float32",
        "smiles_sha256": hashlib.sha256(
            json.dumps(processed, ensure_ascii=False).encode()
        ).hexdigest(),
    }
    metadata["cache_key"] = hashlib.sha256(
        json.dumps(metadata, sort_keys=True).encode()
    ).hexdigest()
    return metadata


def get_molecule_embeddings(
    dataset_name: str,
    batch_size: int = 32,
    n_workers: int = 0,
    mol_encoder: str = REPRESENTATION_NAME,
    overwrite: bool = False,
    encoder_kwargs: dict | None = None,
    n_gpus: int = 1,
    gpu_ids: Sequence[int] | None = None,
) -> None:
    """Encode the labeled molecule vocabulary to a dense NumPy matrix."""

    if mol_encoder != REPRESENTATION_NAME:
        raise ValueError(f"Only {REPRESENTATION_NAME!r} is released")
    if n_gpus != 1 or (gpu_ids is not None and len(gpu_ids) != 1):
        raise ValueError("The public encoder uses one GPU")
    root = Path("data") / dataset_name
    output = root / "mol_embeddings" / f"{REPRESENTATION_NAME}.npy"
    if output.exists() and not overwrite:
        print(f"Using existing {output}")
        return
    smiles = pd.read_csv(root / "unique_smiles.csv")["smiles"].tolist()
    device = f"cuda:{gpu_ids[0]}" if gpu_ids else "cuda"
    print(
        f"Encoding {len(smiles):,} labeled molecules for {dataset_name} "
        f"with MolDeBERTa (batch_size={batch_size})."
    )
    encoder = MolDeBERTaEncoder(device=device, **(encoder_kwargs or {}))
    vectors = encoder.encode(
        smiles,
        batch_size,
        description=f"MolDeBERTa molecules ({dataset_name})",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp.npy")
    np.save(temporary, vectors)
    os.replace(temporary, output)
    output.with_suffix(".metadata.json").write_text(
        json.dumps(_metadata(encoder, smiles), indent=2, sort_keys=True) + "\n"
    )
    print(f"Wrote {len(smiles):,} MolDeBERTa embeddings to {output}")


def get_molecule_embeddings_for_candidates(
    dataset_name: str,
    candidate_map_name: str,
    batch_size: int = 32,
    n_workers: int = 0,
    mol_encoder: str = REPRESENTATION_NAME,
    overwrite: bool = False,
    chunk_size: int = 32,
    encoder_kwargs: dict | None = None,
    n_gpus: int = 1,
    gpu_ids: Sequence[int] | None = None,
) -> None:
    """Encode every candidate row to a padded HDF5 tensor."""

    if mol_encoder != REPRESENTATION_NAME:
        raise ValueError(f"Only {REPRESENTATION_NAME!r} is released")
    if n_gpus != 1 or (gpu_ids is not None and len(gpu_ids) != 1):
        raise ValueError("The public encoder uses one GPU")
    root = Path("data") / dataset_name
    candidate_root = root / "candidates" / candidate_map_name
    output = candidate_root / f"{REPRESENTATION_NAME}.h5"
    if output.exists() and not overwrite:
        with h5py.File(output) as handle:
            if bool(handle.attrs.get("encoding_complete", False)):
                print(f"Using existing {output}")
                return
        print(f"Regenerating incomplete cache {output}")

    with (candidate_root / "map.json").open() as handle:
        candidate_map = json.load(handle)
    targets = pd.read_csv(root / "unique_smiles.csv")["smiles"].tolist()
    width = max(len(candidate_map[target]) for target in targets)
    device = f"cuda:{gpu_ids[0]}" if gpu_ids else "cuda"
    print(
        f"Encoding MolDeBERTa candidate rows for {len(targets):,} targets in "
        f"{dataset_name}/{candidate_map_name} (maximum width={width})."
    )
    encoder = MolDeBERTaEncoder(device=device, **(encoder_kwargs or {}))

    output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output, "w") as handle:
        vectors = handle.create_dataset(
            "candidates_embeddings",
            shape=(len(targets), width, OUTPUT_DIMENSION), dtype="float32",
            chunks=(min(chunk_size, len(targets)), width, OUTPUT_DIMENSION),
        )
        masks = handle.create_dataset(
            "candidate_mask", shape=(len(targets), width), dtype="bool",
            chunks=(min(chunk_size, len(targets)), width),
        )
        handle.attrs["encoding_complete"] = False
        handle.attrs["metadata_json"] = json.dumps(_metadata(encoder, targets), sort_keys=True)

        for start in tqdm(
            range(0, len(targets), chunk_size),
            desc=f"MolDeBERTa candidates ({dataset_name})",
            unit="chunk",
            dynamic_ncols=True,
        ):
            chunk_targets = targets[start : start + chunk_size]
            rows = [candidate_map[target] for target in chunk_targets]
            for target, row in zip(chunk_targets, rows):
                if not row or row[0] != target:
                    raise ValueError(f"Candidate zero is not the target for {target!r}")
            flat = [smiles for row in rows for smiles in row]
            flat_vectors = encoder.encode(flat, batch_size)
            buffer = np.zeros((len(rows), width, OUTPUT_DIMENSION), dtype=np.float32)
            mask = np.zeros((len(rows), width), dtype=bool)
            offset = 0
            for row_index, row in enumerate(rows):
                buffer[row_index, : len(row)] = flat_vectors[offset : offset + len(row)]
                mask[row_index, : len(row)] = True
                offset += len(row)
            vectors[start : start + len(rows)] = buffer
            masks[start : start + len(rows)] = mask
        handle.attrs["encoding_complete"] = True
    print(f"Wrote MolDeBERTa candidate embeddings to {output}")
