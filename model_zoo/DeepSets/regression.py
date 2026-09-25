"""Shared model logic for fingerprint-regression retrieval baselines."""

from __future__ import annotations

import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.MSAlign.losses import candidate_retrieval_metrics, mask_logits


class MLP(nn.Module):
    """A plain ReLU MLP matching the layer semantics of PyG's MLP."""

    def __init__(
        self,
        d_in: int,
        d_hidden: int,
        d_out: int,
        n_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if n_layers < 1:
            raise ValueError("n_layers must be positive.")
        layers = []
        for index in range(n_layers):
            input_width = d_in if index == 0 else d_hidden
            output_width = d_out if index == n_layers - 1 else d_hidden
            layers.append(nn.Linear(input_width, output_width))
            if index != n_layers - 1:
                layers.extend((nn.ReLU(), nn.Dropout(dropout)))
        self.layers = nn.Sequential(*layers)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.layers(values)


class FingerprintRegressionModel(pl.LightningModule):
    """Predict fingerprints and rank candidates by cosine similarity."""

    def __init__(self, backbone: nn.Module, *, lr: float, model_name: str) -> None:
        super().__init__()
        self.backbone = backbone
        self.lr = float(lr)
        self.model_name = model_name
        self.save_hyperparameters(ignore=["backbone"])

    def forward(self, spectrum: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.backbone(spectrum))

    def _loss(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return 1.0 - F.cosine_similarity(prediction, target, dim=-1).mean()

    def training_step(self, batch, batch_idx=0):
        prediction = self(batch["spectrum"])
        loss = self._loss(prediction, batch["target"])
        self.log(
            "loss (train)", loss, on_step=False, on_epoch=True,
            prog_bar=True, batch_size=prediction.shape[0],
        )
        return loss

    def _evaluation_step(self, batch, stage: str):
        prediction = self(batch["spectrum"])
        loss = self._loss(prediction, batch["target"])
        candidates = batch["candidates"]
        candidate_mask = batch["candidate_mask"]
        scores = F.cosine_similarity(prediction[:, None, :], candidates, dim=-1)
        scores = mask_logits(scores, candidate_mask)
        metrics = {
            f"{name} ({stage})": value
            for name, value in candidate_retrieval_metrics(
                scores, candidate_mask
            ).items()
        }
        metrics[f"loss ({stage})"] = loss
        self.log_dict(
            metrics, on_step=False, on_epoch=True, prog_bar=True,
            batch_size=prediction.shape[0],
        )
        return loss

    def validation_step(self, batch, batch_idx=0):
        return self._evaluation_step(batch, "val")

    def test_step(self, batch, batch_idx=0):
        return self._evaluation_step(batch, "test")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def num_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
