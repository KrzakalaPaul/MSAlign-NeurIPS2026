"""Neural-network layers used by MSAlign."""

from __future__ import annotations

import torch
from torch import nn


class MLP(nn.Module):
    """An MLP with ``n_hidden_layers`` hidden blocks and one output projection.

    A hidden block is ``Linear -> LayerNorm -> GELU -> Dropout``. Consequently,
    ``n_hidden_layers=0`` is exactly one linear projection.
    """

    def __init__(self, d_in, d_hidden, d_out, n_hidden_layers, dropout):
        super().__init__()
        layers: list[nn.Module] = []
        width = d_in
        for _ in range(n_hidden_layers):
            layers.extend(
                (nn.Linear(width, d_hidden), nn.LayerNorm(d_hidden), nn.GELU(), nn.Dropout(dropout))
            )
            width = d_hidden
        layers.append(nn.Linear(width, d_out))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
