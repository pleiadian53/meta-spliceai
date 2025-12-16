"""
Data loading and processing for the meta-layer.

- dataset.py: PyTorch Dataset and DataLoader utilities
- splicevardb_loader.py: SpliceVarDB data loading for evaluation
"""

from .dataset import (
    MetaLayerDataset,
    create_dataloaders,
    prepare_training_data
)
from .splicevardb_loader import (
    SpliceVarDBLoader,
    VariantRecord,
    load_splicevardb
)

__all__ = [
    "MetaLayerDataset",
    "create_dataloaders",
    "prepare_training_data",
    "SpliceVarDBLoader",
    "VariantRecord",
    "load_splicevardb",
]

