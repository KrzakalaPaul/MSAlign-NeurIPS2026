"""Raw-spectrum inference used by the MSAlign playground notebook."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from dreams.api import PreTrainedModel
from dreams.utils.data import SpectrumPreprocessor
from dreams.utils.dformats import DataFormatA

from models.MSAlign.encoders import normalize_adduct
from models.MSAlign.model import MSAlign as AlignmentModel
from preprocessing.encode_mol import MolDeBERTaEncoder


class MSAlign:
    """Inference-only facade accepting raw peaks and a SMILES string."""

    def __init__(self, checkpoint: str | Path, device: str = "cuda") -> None:
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("The playground requires an available CUDA GPU.")

        checkpoint = Path(checkpoint)
        print(f"Loading MSAlign checkpoint from {checkpoint}...")
        try:
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        except TypeError:  # torch 2.2 does not expose weights_only everywhere.
            payload = torch.load(checkpoint, map_location="cpu")
        hyperparameters = payload.get("hyper_parameters", {})
        config = payload.get("config", hyperparameters.get("config"))
        dimensions = payload.get(
            "input_dimensions", hyperparameters.get("input_dimensions")
        )
        if config is None or dimensions is None or "state_dict" not in payload:
            raise ValueError("Checkpoint lacks model configuration or weights.")
        if config["representations"] != {
            "spectrum": "dreams",
            "molecule": "moldeberta_base_123m_mtr",
        }:
            raise ValueError(
                "The playground supports DreaMS–MolDeBERTa checkpoints only."
            )
        if "adduct_vocabulary" not in dimensions:
            raise ValueError(
                "Checkpoint lacks its ordered adduct vocabulary. Retrain it with "
                "the current MSAlign code before using raw-input inference."
            )

        self.adduct_vocabulary = list(dimensions["adduct_vocabulary"])
        self._adduct_to_index = {
            value: index for index, value in enumerate(self.adduct_vocabulary)
        }
        self._unknown_adduct_index = int(dimensions["unknown_adduct_index"])

        self.alignment = AlignmentModel(config, dimensions)
        self.alignment.load_state_dict(payload["state_dict"], strict=True)
        self.alignment.requires_grad_(False).eval().to(self.device)

        print("Loading frozen DreaMS and MolDeBERTa encoders...")
        self.spectrum_preprocessor = SpectrumPreprocessor(
            DataFormatA(), n_highest_peaks=100
        )
        self.dreams = (
            PreTrainedModel.from_name("DreaMS_embedding")
            .model.requires_grad_(False)
            .eval()
            .to(self.device)
        )
        self.moldeberta = MolDeBERTaEncoder(device=str(self.device))
        print("MSAlign is ready for raw-input inference.")

    @torch.inference_mode()
    def __call__(
        self,
        spectrum,
        precursor_mz: float,
        adduct: str | None,
        collision_energy: float | None,
        candidate: str,
    ) -> float:
        """Score one raw spectrum–candidate pair."""

        peaks = np.asarray(spectrum, dtype=np.float32)
        if peaks.ndim != 2 or peaks.shape[1] != 2 or len(peaks) == 0:
            raise ValueError(
                "spectrum must be a non-empty [n_peaks, 2] m/z-intensity matrix"
            )
        if not np.isfinite(peaks).all():
            raise ValueError("spectrum contains a non-finite value")
        if not math.isfinite(float(precursor_mz)) or precursor_mz <= 0:
            raise ValueError("precursor_mz must be a positive finite number")
        if not isinstance(candidate, str) or not candidate.strip():
            raise ValueError("candidate must be a non-empty SMILES string")

        processed = self.spectrum_preprocessor(peaks, float(precursor_mz))
        processed = torch.as_tensor(
            processed, dtype=torch.float32, device=self.device
        )
        spectrum_embedding = self.dreams(processed.unsqueeze(0)).float()

        molecule_embedding = torch.from_numpy(
            self.moldeberta.encode([candidate], batch_size=1)
        ).to(self.device)
        candidates = molecule_embedding.unsqueeze(1)
        candidate_mask = torch.ones((1, 1), dtype=torch.bool, device=self.device)

        normalized_adduct = normalize_adduct(adduct)
        adduct_index = self._adduct_to_index.get(
            normalized_adduct, self._unknown_adduct_index
        )
        energy = float("nan") if collision_energy is None else float(collision_energy)
        metadata = {
            "adduct": torch.tensor([adduct_index], dtype=torch.long, device=self.device),
            "collision_energy": torch.tensor(
                [energy], dtype=torch.float32, device=self.device
            ),
        }
        score = self.alignment(
            spectrum_embedding, candidates, metadata, candidate_mask
        )
        return float(score[0, 0].cpu())
