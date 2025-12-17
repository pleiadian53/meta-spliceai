"""
Data loading and processing for the meta-layer.

Modules
-------
- dataset.py: PyTorch Dataset and DataLoader utilities
- splicevardb_loader.py: SpliceVarDB data loading and HGVS parsing
- position_labels.py: Position label derivation for Multi-Step Step 3
- variant_dataset.py: Variant-aware dataset with delta computation
"""

from .dataset import (
    MetaLayerDataset,
    create_dataloaders,
    prepare_training_data
)
from .splicevardb_loader import (
    SpliceVarDBLoader,
    VariantRecord,
    HGVSPositionHint,
    load_splicevardb,
    parse_hgvs_position_hint
)
from .position_labels import (
    AffectedPosition,
    PositionLabelResult,
    derive_position_labels_from_delta,
    derive_position_labels_from_hgvs,
    derive_position_labels,
    create_position_attention_target,
    create_binary_position_mask,
    analyze_position_label_distribution
)

__all__ = [
    # Dataset utilities
    "MetaLayerDataset",
    "create_dataloaders",
    "prepare_training_data",
    # SpliceVarDB
    "SpliceVarDBLoader",
    "VariantRecord",
    "HGVSPositionHint",
    "load_splicevardb",
    "parse_hgvs_position_hint",
    # Position labels (Multi-Step Step 3)
    "AffectedPosition",
    "PositionLabelResult",
    "derive_position_labels_from_delta",
    "derive_position_labels_from_hgvs",
    "derive_position_labels",
    "create_position_attention_target",
    "create_binary_position_mask",
    "analyze_position_label_distribution",
]

