"""
Model components for the meta-layer.

Classification Models:
- MetaSpliceModel: Per-window classification (501nt → [1, 3])
- MetaSpliceModelV2: Sequence-to-sequence (L nt → [L, 3]) - MATCHES BASE MODEL FORMAT
- SpliceInducingClassifier: Binary "Is this variant splice-altering?" classifier
- EffectTypeClassifier: Multi-class effect type classifier

Delta Prediction Models:
- DeltaPredictor: Siamese network for paired prediction
- DeltaPredictorV2: Per-position delta output [L, 2]
- ValidatedDeltaPredictor: Single-pass with SpliceVarDB-validated targets (BEST)
- SimpleCNNDeltaPredictor: Lightweight CNN for delta prediction

Encoders:
- sequence_encoder.py: DNA language model wrappers (HyenaDNA, CNN, etc.)
- score_encoder.py: MLP for base model score features

Calibration:
- delta_predictor_calibrated.py: Scaling, temperature, quantile, hybrid options
"""

from .sequence_encoder import (
    SequenceEncoderFactory,
    CNNEncoder,
    HyenaDNAEncoder,
    IdentityEncoder
)
from .score_encoder import ScoreEncoder, AttentiveScoreEncoder
from .meta_splice_model import MetaSpliceModel, ScoreOnlyModel, CrossAttentionFusion
from .meta_splice_model_v2 import MetaSpliceModelV2, create_meta_model_v2
from .delta_predictor import (
    DeltaPredictor,
    DeltaPredictorWithClassifier,
    WeightedMSELoss,
    create_delta_predictor
)
from .delta_predictor_v2 import DeltaPredictorV2, create_delta_predictor_v2
from .hyenadna_delta_predictor import SimpleCNNDeltaPredictor, HyenaDNADeltaPredictor
from .delta_predictor_calibrated import (
    ScaledDeltaPredictor,
    TemperatureScaledPredictor,
    QuantileDeltaPredictor,
    HybridDeltaPredictor,
    create_calibrated_predictor
)
from .validated_delta_predictor import (
    ValidatedDeltaPredictor,
    ValidatedDeltaPredictorWithAttention,
    create_validated_delta_predictor
)
from .splice_classifier import (
    SpliceInducingClassifier,
    EffectTypeClassifier,
    UnifiedSpliceClassifier,
    create_splice_classifier
)

__all__ = [
    # Sequence encoders
    "SequenceEncoderFactory",
    "CNNEncoder",
    "HyenaDNAEncoder",
    "IdentityEncoder",
    # Score encoders
    "ScoreEncoder",
    "AttentiveScoreEncoder",
    # Classification models
    "MetaSpliceModel",           # V1: Per-window [1, 3] output
    "MetaSpliceModelV2",         # V2: Per-nucleotide [L, 3] output
    "create_meta_model_v2",
    "ScoreOnlyModel",
    "CrossAttentionFusion",
    "SpliceInducingClassifier",  # Binary: splice-altering?
    "EffectTypeClassifier",      # Multi-class: effect type
    "UnifiedSpliceClassifier",   # Multi-task: binary + effect + attention
    "create_splice_classifier",
    # Delta prediction models (paired/Siamese)
    "DeltaPredictor",
    "DeltaPredictorV2",
    "DeltaPredictorWithClassifier",
    "SimpleCNNDeltaPredictor",
    "HyenaDNADeltaPredictor",
    "WeightedMSELoss",
    "create_delta_predictor",
    "create_delta_predictor_v2",
    # Validated delta predictor (single-pass with ground truth targets)
    "ValidatedDeltaPredictor",
    "ValidatedDeltaPredictorWithAttention",
    "create_validated_delta_predictor",
    # Calibrated predictors
    "ScaledDeltaPredictor",
    "TemperatureScaledPredictor",
    "QuantileDeltaPredictor",
    "HybridDeltaPredictor",
    "create_calibrated_predictor",
]

