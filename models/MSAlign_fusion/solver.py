"""Convex listwise calibration used by MSAlign score fusion."""

from __future__ import annotations

from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from scipy.optimize import minimize
from scipy.special import logsumexp


def _arrays(delta: torch.Tensor, mask: torch.Tensor):
    """Move valid target-minus-negative margins to NumPy for SciPy."""

    if delta.ndim != 3 or mask.shape != delta.shape[:2] or mask.dtype != torch.bool:
        raise ValueError("delta [N,K,S] and boolean mask [N,K] have incompatible shapes")
    values = delta.detach().cpu().double().numpy()
    valid = mask.detach().cpu().numpy()
    if not np.isfinite(values[valid]).all():
        raise ValueError("valid score margins must be finite")
    return np.where(valid[..., None], values, 0.0), valid


def fit_listwise(delta: np.ndarray, mask: np.ndarray, temperature: float) -> np.ndarray:
    r"""Minimize the convex listwise loss over the simplex.

    With margin :math:`d_{ik}^T\alpha`, the per-query loss is

    .. math:: \tau\log(1 + \sum_k \exp(-d_{ik}^T\alpha/\tau)).
    """

    n_samples, _, n_scores = delta.shape

    def objective(alpha):
        logits = -np.einsum("nks,s->nk", delta, alpha, optimize=True) / temperature
        logits = np.where(mask, logits, -np.inf)
        normalizer = np.logaddexp(0.0, logsumexp(logits, axis=1))
        weights = np.exp(logits - normalizer[:, None])
        gradient = -np.einsum("nk,nks->s", weights, delta, optimize=True)
        return float(temperature * normalizer.mean()), gradient / n_samples

    result = minimize(
        objective,
        np.full(n_scores, 1.0 / n_scores),
        method="SLSQP",
        jac=True,
        bounds=[(0.0, 1.0)] * n_scores,
        constraints={"type": "eq", "fun": lambda a: a.sum() - 1.0, "jac": lambda a: np.ones_like(a)},
        options={"ftol": 1e-12, "maxiter": 1000},
    )
    if not result.success:
        raise RuntimeError(f"Listwise calibration failed: {result.message}")
    alpha = np.maximum(result.x, 0.0)
    return alpha / alpha.sum()


class ListwiseTemperatureGrid:
    """Fit each temperature and select the coefficients with best validation R@1."""

    def __init__(self, temperature_min, temperature_max, n_temperature_values, n_workers=1):
        self.temperatures = np.geomspace(
            float(temperature_min), float(temperature_max), int(n_temperature_values)
        )
        self.n_workers = int(n_workers)
        self.selected_temperature = None

    @torch.no_grad()
    def __call__(self, delta: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        values, valid = _arrays(delta, mask)

        def fit(temperature):
            alpha = fit_listwise(values, valid, float(temperature))
            margins = np.einsum("nks,s->nk", values, alpha, optimize=True)
            # Strict > 0 makes target/negative ties failures.
            r1 = int(np.all((margins > 0.0) | ~valid, axis=1).sum())
            return float(temperature), alpha, r1

        workers = min(self.n_workers, len(self.temperatures))
        if workers == 1:
            fitted = list(map(fit, self.temperatures))
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                fitted = list(pool.map(fit, self.temperatures))
        self.selected_temperature, alpha, _ = max(fitted, key=lambda item: item[2])
        return torch.as_tensor(alpha, dtype=delta.dtype, device=delta.device)


def get_solver(params: Mapping[str, object]) -> ListwiseTemperatureGrid:
    return ListwiseTemperatureGrid(
        params["temperature_min"], params["temperature_max"],
        params["n_temperature_values"], params.get("n_workers", 1),
    )
