"""MultiView Projection (MVP) baseline."""

from .datamodule import MVPDataModule
from .model import MVP

__all__ = ["MVP", "MVPDataModule"]
