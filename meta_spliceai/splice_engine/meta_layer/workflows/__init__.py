"""
High-level workflows for the meta-layer.

Workflows:
- canonical_splice_training.py: Train on canonical splice sites from artifacts
- prepare_training_data.py: Dataset preparation from artifacts (TODO)
- evaluate_meta_model.py: Comprehensive evaluation (TODO)
- predict_alternative_splicing.py: Inference for alternative splicing (TODO)

NOTE: Class names retain "Phase1" prefix for backward compatibility but the
module has been renamed for better self-documentation. Consider the following
semantic mapping:
    "Phase1" → "Canonical Splice Training" (trains on known splice sites)
    "Phase2" → "Delta Prediction" (predicts variant effects)
"""

from .canonical_splice_training import (
    Phase1Workflow,           # Backward compatible name
    Phase1Config,             # Backward compatible name
    Phase1Result,             # Backward compatible name
    run_phase1                # Backward compatible name
)

# Aliases with descriptive names (preferred for new code)
CanonicalSpliceWorkflow = Phase1Workflow
CanonicalSpliceConfig = Phase1Config
CanonicalSpliceResult = Phase1Result
run_canonical_splice_training = run_phase1

__all__ = [
    # Backward compatible exports
    "Phase1Workflow",
    "Phase1Config",
    "Phase1Result",
    "run_phase1",
    # Descriptive aliases (preferred)
    "CanonicalSpliceWorkflow",
    "CanonicalSpliceConfig",
    "CanonicalSpliceResult",
    "run_canonical_splice_training",
]





