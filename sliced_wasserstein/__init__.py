"""Utilities for train–test sliced Wasserstein analysis."""

from .distance import compute_swd
from .results import get_result, normalized_swd, write_result

__all__ = ["compute_swd", "get_result", "normalized_swd", "write_result"]
