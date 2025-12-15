"""
Meta-Layer: Base-Model-Agnostic Multimodal Meta-Learning for Splice Site Prediction
====================================================================================

This package implements a multimodal deep learning meta-layer that recalibrates
base model splice site predictions using:
1. Contextual DNA sequences (via HyenaDNA or other DNA language models)
2. Base model score features (donor, acceptor, neither probabilities)
3. Derived features (entropy, peak patterns, signal strength)

Key Features:
- Base-model-agnostic: Works with any base model (SpliceAI, OpenSpliceAI, etc.)
- Multimodal: Combines sequence and score information
- Scalable: Supports CPU (M1) and GPU (RunPods) training
- Variant-aware: Integrates SpliceVarDB for context-dependent splice sites
- Integrated: Uses genomic_resources for consistent path resolution

Quick Start:
-----------
>>> from meta_spliceai.splice_engine.meta_layer import MetaLayerConfig, MetaSpliceModel
>>> from meta_spliceai.splice_engine.meta_layer.data import MetaLayerDataset
>>> 
>>> # Configure for OpenSpliceAI
>>> config = MetaLayerConfig(base_model='openspliceai')
>>> print(config.artifacts_dir)  # Uses genomic_resources
>>> 
>>> # Create model
>>> model = MetaSpliceModel(
...     sequence_encoder='cnn',
...     num_score_features=50,
...     hidden_dim=256
... )

Documentation:
-------------
See `meta_spliceai/splice_engine/meta_layer/docs/` for detailed documentation:
- ARCHITECTURE.md: System architecture and design principles
- LABELING_STRATEGY.md: How labels are created from base layer + variants
- ALTERNATIVE_SPLICING_PIPELINE.md: From scores to exon-intron predictions
- GENOMIC_RESOURCES_INTEGRATION.md: Integration with genomic_resources system
- TRAINING_GUIDE.md: Step-by-step training instructions
"""

__version__ = "0.1.0"
__author__ = "MetaSpliceAI Team"

# Core configuration
from .core.config import MetaLayerConfig
from .core.artifact_loader import ArtifactLoader
from .core.feature_schema import FeatureSchema, LABEL_ENCODING, LABEL_DECODING

# Models
from .models import (
    MetaSpliceModel,
    ScoreOnlyModel,
    SequenceEncoderFactory,
    CNNEncoder,
    ScoreEncoder
)

# Data
from .data import (
    MetaLayerDataset,
    create_dataloaders,
    prepare_training_data
)

__all__ = [
    # Version
    "__version__",
    # Core
    "MetaLayerConfig",
    "ArtifactLoader",
    "FeatureSchema",
    "LABEL_ENCODING",
    "LABEL_DECODING",
    # Models
    "MetaSpliceModel",
    "ScoreOnlyModel",
    "SequenceEncoderFactory",
    "CNNEncoder",
    "ScoreEncoder",
    # Data
    "MetaLayerDataset",
    "create_dataloaders",
    "prepare_training_data",
]

