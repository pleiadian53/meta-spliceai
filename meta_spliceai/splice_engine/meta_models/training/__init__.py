"""Meta-model training subpackage.

Provides convenient public API:

>>> from meta_spliceai.splice_engine.meta_models.training import Trainer
>>> trainer = Trainer().fit("dataset_dir")

Modules are lazily imported to keep startup lightweight.
"""
from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import TYPE_CHECKING

__all__ = [
    "datasets",
    "models",
    "trainer",
    "Trainer",
]


def __getattr__(name: str) -> ModuleType:  # pragma: no cover
    if name in ("datasets", "models", "trainer"):
        return import_module(f"{__name__}.{name}")
    if name == "Trainer":
        submod = import_module(f"{__name__}.trainer")
        return getattr(submod, "Trainer")
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def __dir__() -> list[str]:  # pragma: no cover
    return sorted(__all__)


if TYPE_CHECKING:  # pragma: no cover
    from .trainer import Trainer  # noqa: F401
