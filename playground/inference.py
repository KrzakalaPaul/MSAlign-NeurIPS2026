"""Standalone raw-input inference for released MSAlign bundles.

This module deliberately keeps deployment concerns out of the training code.
It computes the frozen input representations on demand, loads either one
alignment model or a calibrated collection of scorers, and exposes the small
``model(spectrum, candidate)`` interface used in the playground notebook.
"""

from __future__ import annotations

import math
import os
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, rdFingerprintGenerator

from models.MSAlign.encoders import normalize_adduct
from models.MSAlign.model import MSAlign as AlignmentModel

from .bundle import load_manifest


MOLDEBERTA_ID = "SaeedLab/MolDeBERTa-base-123M-mtr"
MORGAN_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=4096)
ELECTRON_MASS = 0.00054858


def _torch_load(path: Path) -> dict:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid checkpoint: {path}")
    return payload


def _alignment_checkpoint(path: Path, device: torch.device):
    payload = _torch_load(path)
    config = payload.get("config")
    dimensions = payload.get("input_dimensions")
    if config is None or dimensions is None or "state_dict" not in payload:
        raise ValueError(f"Checkpoint is not a portable MSAlign checkpoint: {path}")
    model = AlignmentModel(config, dimensions)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.requires_grad_(False).eval().to(device)
    return model, config, dimensions


def _canonical_smiles(values: Sequence[str]) -> tuple[list[str], list[Chem.Mol]]:
    canonical, molecules = [], []
    for value in values:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("candidate SMILES must be non-empty strings")
        molecule = Chem.MolFromSmiles(value)
        if molecule is None:
            raise ValueError(f"invalid candidate SMILES: {value!r}")
        normalized = Chem.MolToSmiles(
            molecule, canonical=True, isomericSmiles=True
        )
        canonical.append(normalized)
        molecules.append(Chem.MolFromSmiles(normalized))
    return canonical, molecules


def _validate_spectrum(spectrum: dict) -> tuple[np.ndarray, float, object, float]:
    if not isinstance(spectrum, dict):
        raise ValueError("spectrum must be a dictionary")
    if "peaks" not in spectrum or "precursor_mz" not in spectrum:
        raise ValueError("spectrum requires 'peaks' and 'precursor_mz'")

    peaks = np.asarray(spectrum["peaks"], dtype=np.float32)
    if peaks.ndim != 2 or peaks.shape[1] != 2 or len(peaks) == 0:
        raise ValueError("spectrum['peaks'] must have shape [n_peaks, 2]")
    if not np.isfinite(peaks).all():
        raise ValueError("spectrum peaks must be finite")
    if np.any(peaks[:, 0] < 0) or np.any(peaks[:, 1] < 0):
        raise ValueError("peak m/z and intensity values must be non-negative")
    if not np.any(peaks[:, 1] > 0):
        raise ValueError("at least one peak intensity must be positive")

    precursor_mz = float(spectrum["precursor_mz"])
    if not math.isfinite(precursor_mz) or precursor_mz <= 0:
        raise ValueError("spectrum['precursor_mz'] must be positive and finite")

    collision_energy = spectrum.get("collision_energy")
    if collision_energy is None:
        collision_energy = float("nan")
    else:
        collision_energy = float(collision_energy)
        if math.isinf(collision_energy):
            raise ValueError("collision_energy must be finite, NaN, or None")
    return peaks, precursor_mz, spectrum.get("adduct"), collision_energy


def _binned_spectrum(peaks: np.ndarray, *, max_mz: float, bin_width: float) -> np.ndarray:
    """Apply the same fixed histogram transform used during training."""

    n_bins = int(np.ceil(max_mz / bin_width))
    valid = peaks[:, 0] <= max_mz
    indices = np.floor(peaks[valid, 0] / bin_width).astype(int)
    indices = np.clip(indices, 0, n_bins - 1)
    vector = np.zeros(n_bins, dtype=np.float32)
    np.add.at(vector, indices, peaks[valid, 1])
    if vector.max() <= 0:
        raise ValueError(f"spectrum has no positive-intensity peak below {max_mz} m/z")
    vector = vector / vector.max() * 999.0
    return (np.log10(vector + 1.0) / 3.0).astype(np.float32)


def _morgan_fingerprints(molecules: Sequence[Chem.Mol]) -> np.ndarray:
    vectors = np.zeros((len(molecules), 4096), dtype=np.float32)
    for index, molecule in enumerate(molecules):
        fingerprint = MORGAN_GENERATOR.GetFingerprint(molecule)
        DataStructs.ConvertToNumpyArray(fingerprint, vectors[index])
    return vectors


def _adduct_offset(adduct: object) -> float:
    """Return the signed ion-to-neutral mass offset used by mass scorers."""

    adduct = normalize_adduct(adduct)
    table = Chem.GetPeriodicTable()
    mass = lambda symbol: table.GetMostCommonIsotopeMass(
        Chem.Atom(symbol).GetAtomicNum()
    )
    offsets = {
        "[M+H]+": mass("H") - ELECTRON_MASS,
        "[M+Na]+": mass("Na") - ELECTRON_MASS,
        "[M+K]+": mass("K") - ELECTRON_MASS,
        "[M-H2O+H]+": -mass("O") - mass("H") - ELECTRON_MASS,
        "[M+H3N+H]+": mass("N") + 4 * mass("H") - ELECTRON_MASS,
        "[M+NH4]+": mass("N") + 4 * mass("H") - ELECTRON_MASS,
        "[M]+": -ELECTRON_MASS,
        "[M-H4O2+H]+": -2 * mass("O") - 3 * mass("H") - ELECTRON_MASS,
        "[M-H]-": -mass("H") + ELECTRON_MASS,
        "[M+HCOOH-H]-": mass("H") + mass("C") + 2 * mass("O") + ELECTRON_MASS,
        "[M+CH3COOH-H]-": 3 * mass("H") + 2 * mass("C") + 2 * mass("O") + ELECTRON_MASS,
        "[M+Cl]-": mass("Cl") + ELECTRON_MASS,
    }
    return offsets.get(adduct, float("nan"))


def _combine_channels(manifest: dict, channels: dict[str, torch.Tensor]) -> torch.Tensor:
    """Apply the calibrated convex combination stored in a bundle."""

    weights = manifest.get("weights")
    if weights is None:
        return channels[manifest["scorers"][0]["name"]]
    scores = torch.zeros_like(next(iter(channels.values())))
    for name, weight in weights.items():
        scores.add_(channels[name], alpha=float(weight))
    return scores


class MSAlign:
    """Load one portable bundle and score raw spectrum–molecule pairs."""

    def __init__(self, checkpoint: str | Path) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("The MSAlign playground requires a CUDA GPU")
        self.device = torch.device("cuda")
        self.bundle, self.manifest = load_manifest(checkpoint)

        self.alignment_scorers = []
        self.mass_scorers = []
        for scorer in self.manifest["scorers"]:
            if scorer["type"] == "msalign":
                model, config, dimensions = _alignment_checkpoint(
                    self.bundle / scorer["checkpoint"], self.device
                )
                self.alignment_scorers.append(
                    {
                        "name": scorer["name"],
                        "model": model,
                        "config": config,
                        "dimensions": dimensions,
                    }
                )
            else:
                self.mass_scorers.append(scorer)

        self._dreams = None
        self._dreams_preprocessor = None
        self._moldeberta = None
        self._moldeberta_tokenizer = None

    def _load_dreams(self) -> None:
        if self._dreams is not None:
            return
        from dreams.api import PreTrainedModel
        from dreams.utils.data import SpectrumPreprocessor
        from dreams.utils.dformats import DataFormatA

        self._dreams_preprocessor = SpectrumPreprocessor(
            DataFormatA(), n_highest_peaks=100
        )
        self._dreams = (
            PreTrainedModel.from_name("DreaMS_embedding")
            .model.requires_grad_(False)
            .eval()
            .to(self.device)
        )

    def _load_moldeberta(self) -> None:
        if self._moldeberta is not None:
            return
        token = os.environ.get("HF_TOKEN")
        if not token:
            raise RuntimeError(
                "Set HF_TOKEN after accepting the MolDeBERTa terms on Hugging Face"
            )
        from transformers import AutoModel, AutoTokenizer

        revision = self.manifest["external_models"]["moldeberta"]["revision"]
        self._moldeberta_tokenizer = AutoTokenizer.from_pretrained(
            MOLDEBERTA_ID, revision=revision, token=token
        )
        self._moldeberta = (
            AutoModel.from_pretrained(
                MOLDEBERTA_ID,
                revision=revision,
                token=token,
                use_safetensors=True,
            )
            .requires_grad_(False)
            .eval()
            .to(self.device)
        )

    @torch.inference_mode()
    def _spectrum_representations(
        self, peaks: np.ndarray, precursor_mz: float
    ) -> dict[str, torch.Tensor]:
        needed = {
            scorer["config"]["representations"]["spectrum"]
            for scorer in self.alignment_scorers
        }
        representations = {}
        if "dreams" in needed:
            self._load_dreams()
            processed = self._dreams_preprocessor(
                peaks, precursor_mz, high_form=True
            )
            tensor = torch.as_tensor(
                processed, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            representations["dreams"] = self._dreams(tensor).float()
        if "bins" in needed:
            config = next(
                scorer["config"] for scorer in self.alignment_scorers
                if scorer["config"]["representations"]["spectrum"] == "bins"
            )
            bins = _binned_spectrum(peaks, **config["spectrum_bins"])
            representations["bins"] = torch.from_numpy(bins).to(self.device)[None]
        return representations

    @torch.inference_mode()
    def _molecule_representations(
        self, canonical: Sequence[str], molecules: Sequence[Chem.Mol]
    ) -> dict[str, torch.Tensor]:
        needed = {
            scorer["config"]["representations"]["molecule"]
            for scorer in self.alignment_scorers
        }
        representations = {}
        if "moldeberta_base_123m_mtr" in needed:
            self._load_moldeberta()
            maximum = min(
                int(self._moldeberta.config.max_position_embeddings),
                int(self._moldeberta_tokenizer.model_max_length),
            )
            encoded = []
            for start in range(0, len(canonical), 32):
                batch = self._moldeberta_tokenizer(
                    list(canonical[start : start + 32]),
                    padding=True,
                    truncation=True,
                    max_length=maximum,
                    return_tensors="pt",
                )
                batch = {
                    name: value.to(self.device) for name, value in batch.items()
                }
                encoded.append(
                    self._moldeberta(**batch).last_hidden_state[:, 0].float()
                )
            representations["moldeberta_base_123m_mtr"] = torch.cat(encoded)
        if "morgan_2_4096" in needed:
            fingerprints = _morgan_fingerprints(molecules)
            representations["morgan_2_4096"] = torch.from_numpy(
                fingerprints
            ).to(self.device)
        return representations

    @staticmethod
    def _metadata(dimensions: dict, adduct: object, energy: float, device):
        vocabulary = dimensions.get("adduct_vocabulary")
        if vocabulary is None:
            adduct_index = 0
        else:
            lookup = {value: index for index, value in enumerate(vocabulary)}
            adduct_index = lookup.get(
                normalize_adduct(adduct), int(dimensions["unknown_adduct_index"])
            )
        return {
            "adduct": torch.tensor([adduct_index], dtype=torch.long, device=device),
            "collision_energy": torch.tensor(
                [energy], dtype=torch.float32, device=device
            ),
        }

    @torch.inference_mode()
    def _score_channels(
        self, spectrum: dict, candidates: Sequence[str]
    ) -> dict[str, torch.Tensor]:
        """Compute every scorer channel before calibrated fusion."""

        peaks, precursor_mz, adduct, energy = _validate_spectrum(spectrum)
        canonical, molecules = _canonical_smiles(candidates)
        spectra = self._spectrum_representations(peaks, precursor_mz)
        molecules_by_representation = self._molecule_representations(
            canonical, molecules
        )

        channels = {}
        mask = torch.ones((1, len(candidates)), dtype=torch.bool, device=self.device)
        for scorer in self.alignment_scorers:
            config = scorer["config"]
            ms_name = config["representations"]["spectrum"]
            mol_name = config["representations"]["molecule"]
            channels[scorer["name"]] = scorer["model"](
                spectra[ms_name],
                molecules_by_representation[mol_name][None],
                self._metadata(scorer["dimensions"], adduct, energy, self.device),
                mask,
            )[0]

        offset = _adduct_offset(adduct)
        estimated_mass = precursor_mz - offset
        candidate_masses = torch.tensor(
            [Descriptors.ExactMolWt(molecule) for molecule in molecules],
            dtype=torch.float32,
            device=self.device,
        )
        for scorer in self.mass_scorers:
            if math.isfinite(estimated_mass) and estimated_mass > 0:
                ppm = 1e6 * abs(estimated_mass - candidate_masses) / candidate_masses
                channel = torch.exp(
                    -0.5 * (ppm / float(scorer["scale_ppm"])).square()
                )
            else:
                channel = torch.zeros_like(candidate_masses)
            channels[scorer["name"]] = channel
        return channels

    @torch.inference_mode()
    def __call__(self, spectrum: dict, candidate: str | Sequence[str]):
        """Return one score or a score list preserving candidate order."""

        single = isinstance(candidate, str)
        if not single and not isinstance(candidate, Sequence):
            raise ValueError("candidate must be a SMILES string or a list of SMILES")
        candidates = [candidate] if single else list(candidate)
        if not candidates:
            raise ValueError("candidate list must not be empty")

        channels = self._score_channels(spectrum, candidates)
        scores = _combine_channels(self.manifest, channels)
        values = scores.float().cpu().tolist()
        return values[0] if single else values
