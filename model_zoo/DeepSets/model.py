"""DeepSet fingerprint-regression model."""

from __future__ import annotations

import torch
import torch.nn as nn
from math import ceil

from .regression import FingerprintRegressionModel, MLP


class DeepSetBackbone(nn.Module):
    """Permutation-invariant ``rho(sum(phi(peak)))`` spectrum encoder."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        output_dim: int,
        num_layers_per_mlp: int,
        dropout: float,
        fourier_features: bool,
    ) -> None:
        super().__init__()
        self.fourier_features = fourier_features
        if fourier_features:
            frequencies = [1 / (1e-4 * i) for i in range(2, ceil(1 / 1e-4), 2)]
            frequencies += [1 / i for i in range(2, ceil(1000), 1)]
            self.register_buffer("frequencies", torch.tensor(frequencies).view(1, 1, -1))
            mz_width = int(0.8 * hidden_dim)
            self.mz_projection = nn.Linear(2 * len(frequencies), mz_width)
            self.intensity_projection = nn.Linear(1, hidden_dim - mz_width)
            d_peak = hidden_dim
        else:
            d_peak = 2
        self.phi = MLP(d_peak, hidden_dim, hidden_dim, num_layers_per_mlp, dropout)
        self.rho = MLP(
            hidden_dim, hidden_dim, output_dim, num_layers_per_mlp, dropout
        )

    def forward(self, peaks: torch.Tensor) -> torch.Tensor:
        if self.fourier_features:
            angles = 2 * torch.pi * peaks[..., :1] * self.frequencies
            mz = self.mz_projection(torch.cat((angles.cos(), angles.sin()), dim=-1))
            intensity = self.intensity_projection(peaks[..., 1:2])
            peaks = torch.cat((mz, intensity), dim=-1)
        return self.rho(self.phi(peaks).sum(dim=-2))


class DeepSet(FingerprintRegressionModel):
    """Predict a Morgan fingerprint directly from a set of spectral peaks."""

    def __init__(self, config: dict) -> None:
        architecture = config["architecture"]
        if int(architecture["output_dim"]) != 4096:
            raise ValueError("DeepSet must predict the 4096-bit cached fingerprint.")
        backbone = DeepSetBackbone(
            hidden_dim=int(architecture["hidden_dim"]),
            output_dim=int(architecture["output_dim"]),
            num_layers_per_mlp=int(architecture["num_layers_per_mlp"]),
            dropout=float(architecture["dropout"]),
            fourier_features=bool(architecture["fourier_features"]),
        )
        super().__init__(
            backbone,
            lr=float(config["training"]["lr"]),
            model_name="DeepSet",
        )
