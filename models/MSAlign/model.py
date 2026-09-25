"""The alignment model described in the MSAlign paper.

The implementation intentionally mirrors the equations in the paper. A spectrum
representation is projected, concatenated with collision-energy and adduct
embeddings, and projected again. A molecular representation is projected by an
independent MLP. Both outputs live in the same space and are compared by cosine
similarity.
"""

from __future__ import annotations

from collections.abc import Mapping

import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim.lr_scheduler import LambdaLR

from .encoders import AdductEncoder, CollisionEnergyEncoder
from .losses import candidate_cross_entropy, candidate_retrieval_metrics, mask_logits
from .modules import MLP


class MSAlign(pl.LightningModule):
    """Align one spectrum representation with one molecular representation.

    ``input_dimensions`` is inferred by the data module. Keeping it separate from
    YAML lets the same architecture consume DreaMS or bins and MolDeBERTa or
    Morgan fingerprints without representation-specific model code.
    """

    def __init__(self, config: Mapping, input_dimensions: Mapping) -> None:
        super().__init__()
        self.config = dict(config)
        model = config["model"]
        training = config["training"]

        d_shared = int(model["d_shared"])
        d_hidden = int(model["d_hidden"])
        d_metadata = int(model["d_metadata"])
        dropout = float(model["dropout"])
        use_energy = bool(model.get("use_collision_energy", True))
        use_adduct = bool(model.get("use_adduct", True))
        n_metadata = int(use_energy) + int(use_adduct)
        if n_metadata and d_metadata % n_metadata:
            raise ValueError("d_metadata must be divisible by the number of metadata fields")
        d_metadata_part = d_metadata // n_metadata if n_metadata else 0

        self.ms_pre = MLP(
            int(input_dimensions["d_ms"]), d_hidden, d_shared,
            int(model["n_hidden_layers_ms_pre"]), dropout,
        )
        self.energy_encoder = CollisionEnergyEncoder(d_metadata_part) if use_energy else None
        self.adduct_encoder = (
            AdductEncoder(
                int(input_dimensions["n_adducts"]), d_metadata_part,
                int(input_dimensions["unknown_adduct_index"]),
                input_dimensions["adduct_signs"],
            )
            if use_adduct else None
        )
        self.ms_post = MLP(
            d_shared + (d_metadata if n_metadata else 0), d_hidden, d_shared,
            int(model["n_hidden_layers_ms_post"]), dropout,
        )
        self.mol = MLP(
            int(input_dimensions["d_mol"]), d_hidden, d_shared,
            int(model["n_hidden_layers_mol"]), dropout,
        )

        # Optimizing log(tau) keeps the temperature positive without clipping.
        self.log_temperature = nn.Parameter(
            torch.tensor(float(model.get("temperature_init", 0.07))).log()
        )
        self.lr = float(training["learning_rate"])
        self.weight_decay = float(training["weight_decay"])
        self.n_warmup_steps = int(training["n_warmup_steps"])
        self.n_max_steps = int(training["n_max_steps"])
        self.save_hyperparameters(
            {"config": dict(config), "input_dimensions": dict(input_dimensions)}
        )

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_temperature.exp()

    def encode_spectrum(
        self, spectrum: torch.Tensor, metadata: Mapping[str, torch.Tensor]
    ) -> torch.Tensor:
        """Return the unnormalized spectrum vector in the shared space."""

        spectrum = self.ms_pre(spectrum.float())
        metadata_vectors = []
        if self.energy_encoder is not None:
            metadata_vectors.append(self.energy_encoder(metadata["collision_energy"]))
        if self.adduct_encoder is not None:
            metadata_vectors.append(self.adduct_encoder(metadata["adduct"]))
        if metadata_vectors:
            spectrum = torch.cat((spectrum, *metadata_vectors), dim=-1)
        return self.ms_post(spectrum)

    def encode_molecule(self, molecule: torch.Tensor) -> torch.Tensor:
        """Return the unnormalized molecule vector in the shared space."""

        return self.mol(molecule.float())

    def forward(
        self,
        spectrum: torch.Tensor,
        candidates: torch.Tensor,
        metadata: Mapping[str, torch.Tensor],
        candidate_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Score candidates; candidate zero is the target by construction."""

        spectrum = F.normalize(self.encode_spectrum(spectrum, metadata), dim=-1)
        valid_indices = candidate_mask.nonzero(as_tuple=False)
        molecules = F.normalize(self.encode_molecule(candidates[candidate_mask]), dim=-1)
        scores = spectrum.new_zeros(candidate_mask.shape)
        scores[candidate_mask] = (
            spectrum[valid_indices[:, 0]] * molecules
        ).sum(dim=-1) / self.temperature
        return mask_logits(scores, candidate_mask)

    def forward_batch(self, batch: Mapping[str, object]) -> torch.Tensor:
        return self(
            batch["spectrum"], batch["candidates"], batch["metadata"],
            batch["candidate_mask"],
        )

    def _step(self, batch: Mapping[str, object], stage: str):
        scores = self.forward_batch(batch)
        mask = batch["candidate_mask"]
        if stage == "train":
            loss = candidate_cross_entropy(scores, mask)
            self.log("loss/train", loss, on_step=False, on_epoch=True, prog_bar=True)
            return loss
        metrics = candidate_retrieval_metrics(scores, mask)
        self.log_dict(
            {f"{name}/{stage}": value for name, value in metrics.items()},
            on_step=False, on_epoch=True, prog_bar=(stage == "val"),
        )
        return metrics["R@1"]

    def training_step(self, batch, batch_idx=0):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx=0):
        return self._step(batch, "val")

    def test_step(self, batch, batch_idx=0):
        return self._step(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        def cosine_with_warmup(step: int) -> float:
            if step < self.n_warmup_steps:
                return step / max(1, self.n_warmup_steps)
            progress = (step - self.n_warmup_steps) / max(
                1, self.n_max_steps - self.n_warmup_steps
            )
            return 0.5 * (1.0 + np.cos(np.pi * min(progress, 1.0)))

        scheduler = LambdaLR(optimizer, cosine_with_warmup)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}
