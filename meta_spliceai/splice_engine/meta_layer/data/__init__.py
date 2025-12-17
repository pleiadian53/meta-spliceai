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
    derive_position_labels_from_delta,
    derive_position_labels_per_channel,
    create_position_attention_target,
    create_binary_position_mask,
    create_offset_target,
    derive_position_from_hgvs,
    effect_type_to_channel,
    channel_to_effect_type,
    summarize_affected_positions
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
    "derive_position_labels_from_delta",
    "derive_position_labels_per_channel",
    "create_position_attention_target",
    "create_binary_position_mask",
    "create_offset_target",
    "derive_position_from_hgvs",
    "effect_type_to_channel",
    "channel_to_effect_type",
    "summarize_affected_positions",
]

