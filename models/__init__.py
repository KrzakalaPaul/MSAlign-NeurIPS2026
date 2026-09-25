"""Public MSAlign model entry points."""

__all__ = ["train_MSAlign"]


def __getattr__(name):
    if name == "train_MSAlign":
        from .MSAlign import train_MSAlign

        return train_MSAlign
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
