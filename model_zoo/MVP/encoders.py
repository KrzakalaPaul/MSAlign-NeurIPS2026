"""Encoders used by the MVP four-view contrastive model."""

from __future__ import annotations

import torch
from torch import nn

from model_zoo.FLARE.utils import MolEncoder


class FingerprintEncoder(nn.Module):
    """Published 1024 -> 512 -> 1024 -> 512 fingerprint projection."""

    def __init__(self, d_in: int = 1024, d_out: int = 512) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(d_in, d_out), nn.ReLU(),
            nn.Linear(d_out, 2 * d_out), nn.ReLU(),
            nn.Linear(2 * d_out, d_out),
        )

    def forward(self, fingerprint: torch.Tensor) -> torch.Tensor:
        return self.network(fingerprint.float())


class BinnedSpectrumEncoder(nn.Module):
    """Three-linear-layer encoder used by the MVP BinnedSpec ablation."""

    def __init__(self, d_in: int, d_out: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(d_in, 2 * d_out), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(2 * d_out, 2 * d_out), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(2 * d_out, d_out), nn.Dropout(dropout),
        )

    def forward(self, spectrum: torch.Tensor) -> torch.Tensor:
        return self.network(spectrum.float())


class FormulaSpectrumEncoder(nn.Module):
    """Encode 14 element counts plus intensity with an MLP and Transformer."""

    def __init__(
        self,
        d_in: int,
        d_out: int,
        formula_dimensions: list[int],
        formula_dropout: float,
        n_transformer_layers: int,
        n_heads: int,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for index, width in enumerate(formula_dimensions):
            layers.append(nn.Linear(d_in if index == 0 else formula_dimensions[index - 1], width))
            if index + 1 < len(formula_dimensions):
                layers.extend((nn.ReLU(), nn.Dropout(formula_dropout)))
        self.peak_mlp = nn.Sequential(*layers)
        width = formula_dimensions[-1]
        layer = nn.TransformerEncoderLayer(d_model=width, nhead=n_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_transformer_layers)
        self.output = nn.Linear(width, d_out)

    def forward(self, tokens: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        hidden = self.transformer(
            self.peak_mlp(tokens.float()), src_key_padding_mask=padding_mask
        )
        valid = ~padding_mask
        pooled = (hidden * valid.unsqueeze(-1)).sum(1) / valid.sum(1, keepdim=True).clamp_min(1)
        return self.output(pooled)


def molecular_graph_encoder(config: dict) -> MolEncoder:
    """Build the JESTR/MVP graph encoder already shared through FLARE."""
    return MolEncoder(
        in_dim=78,
        out_dim=config["d_shared"],
        gnn_type=config["gnn_type"],
        gnn_dropout=config["gnn_dropout"],
        mlp_dropout=config["fc_dropout"],
        gnn_channels=config["gnn_channels"],
        gnn_hidden_dim=config["gnn_hidden_dim"],
        pool="max",
    )
