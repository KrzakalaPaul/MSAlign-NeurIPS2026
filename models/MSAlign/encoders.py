"""Metadata encoders used in the released MSAlign architecture."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


ADDUCT_ALIASES = {
    "[M+CH3COO]-": "[M+CH3COOH-H]-",
    "[M+FA-H]-": "[M+HCOOH-H]-",
    "[M+HCOO]-": "[M+HCOOH-H]-",
}


def normalize_adduct(adduct):
    """Map equivalent adduct spellings to the same category."""

    if not isinstance(adduct, str):
        return adduct
    compact = adduct.replace(" ", "")
    return ADDUCT_ALIASES.get(compact, compact)


class CollisionEnergyEncoder(nn.Module):
    r"""Sinusoidal encoding of collision energy.

    For energy :math:`e`, dimensions ``2i`` and ``2i+1`` contain
    :math:`\sin(e/(100\,10000^{2i/d}))` and the corresponding cosine. Missing
    energies use a learned vector.
    """

    def __init__(self, d_out: int) -> None:
        super().__init__()
        if d_out % 2:
            raise ValueError("The collision-energy dimension must be even")
        positions = torch.arange(0, d_out, 2, dtype=torch.float32)
        self.register_buffer("frequencies", 10_000.0 ** (positions / d_out))
        self.missing = nn.Parameter(torch.empty(d_out))
        nn.init.normal_(self.missing, std=0.02)

    def forward(self, energy: torch.Tensor) -> torch.Tensor:
        missing = torch.isnan(energy)
        angles = energy.nan_to_num().unsqueeze(-1) / 100.0 / self.frequencies
        encoded = torch.stack((angles.sin(), angles.cos()), dim=-1).flatten(-2)
        return torch.where(missing.unsqueeze(-1), self.missing, encoded)


class AdductEncoder(nn.Module):
    """Learn adduct identity and charge-sign embeddings.

    Half the output represents the category and half represents its sign.
    Unknown or missing adducts have their own category and sign entries.
    """

    def __init__(self, n_adducts: int, d_out: int, unknown_index: int, adduct_signs: Sequence[int]):
        super().__init__()
        if d_out % 2:
            raise ValueError("The adduct dimension must be even")
        self.category = nn.Embedding(n_adducts, d_out // 2)
        self.sign = nn.Embedding(3, d_out // 2)  # negative, positive, unknown
        nn.init.normal_(self.category.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.sign.weight, mean=0.0, std=0.02)
        signs = torch.as_tensor(adduct_signs)
        sign_indices = torch.where(signs == 0, 2, (signs + 1) // 2)
        self.register_buffer("sign_indices", sign_indices.long())
        self.unknown_index = int(unknown_index)

    def forward(self, adduct: torch.Tensor) -> torch.Tensor:
        return torch.cat((self.category(adduct), self.sign(self.sign_indices[adduct])), dim=-1)
