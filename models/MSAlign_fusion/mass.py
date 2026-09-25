"""Estimated-mass score used by MSAlign^M and MSAlign^4M."""

from __future__ import annotations

from collections.abc import Mapping

import torch


class MassScore:
    r"""Gaussian similarity between estimated and candidate neutral masses.

    For scale :math:`\sigma` in ppm,

    .. math:: \rho_\sigma(\hat m,m)=\exp[-\tfrac12(
              10^6|\hat m-m|/(\sigma m))^2].
    """

    def __init__(self, params: Mapping[str, object]) -> None:
        if params.get("mass_error_unit") != "ppm" or params.get("transform") != "gaussian":
            raise ValueError("The released mass score is Gaussian in ppm")
        self.scale = float(params["scale"])
        if self.scale <= 0:
            raise ValueError("The ppm scale must be positive")

    def __call__(self, estimated_mass: torch.Tensor, candidate_mass: torch.Tensor) -> torch.Tensor:
        ppm = 1e6 * (estimated_mass - candidate_mass).abs() / candidate_mass
        return torch.exp(-0.5 * (ppm / self.scale).square())


def score_mass_candidates(
    estimated_mass: torch.Tensor,
    candidate_mass: torch.Tensor,
    candidate_mask: torch.Tensor,
    definition: MassScore,
) -> torch.Tensor:
    """Score valid candidates; missing estimates yield an all-zero channel."""

    scores = candidate_mass.new_zeros(candidate_mass.shape)
    rows = torch.isfinite(estimated_mass) & (estimated_mass > 0)
    valid = candidate_mask & rows[:, None]
    row_indices = valid.nonzero(as_tuple=False)[:, 0]
    scores[valid] = definition(estimated_mass[row_indices], candidate_mass[valid])
    return scores
