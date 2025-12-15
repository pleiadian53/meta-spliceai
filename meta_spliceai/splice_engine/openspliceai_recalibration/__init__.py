"""
OpenSpliceAI Recalibration Package
===================================

Experimental package for direct recalibration of OpenSpliceAI predictions
using external validation datasets (SpliceVarDB, ClinVar).

This package is separate from the meta_models package to:
1. Use OpenSpliceAI as the base model (vs SpliceAI in meta_models)
2. Test direct recalibration approaches independently
3. Leverage SpliceVarDB's comprehensive validated variants

Architecture:
------------
- core/: Base prediction and recalibration models
- data/: SpliceVarDB and variant data handling
- training/: Recalibration model training pipelines
- workflows/: End-to-end training and inference workflows

Quick Start:
-----------
from meta_spliceai.splice_engine.openspliceai_recalibration import (
    SpliceVarDBLoader,
    OpenSpliceAIPredictor,
    SpliceVarDBTrainingPipeline
)

# Load SpliceVarDB data
loader = SpliceVarDBLoader(output_dir="./data/splicevardb")
variants = loader.load_validated_variants()

# Generate predictions
predictor = OpenSpliceAIPredictor()
predictions = predictor.predict_batch(variants.to_dict('records'))

# Train recalibration model
pipeline = SpliceVarDBTrainingPipeline(
    data_dir="./data/splicevardb",
    output_dir="./models/recalibration"
)
results = pipeline.train()

Documentation:
-------------
- README.md: Package overview and usage
- INTEGRATION_GUIDE.md: Integration with existing infrastructure
- IMPLEMENTATION_SUMMARY.md: Implementation details and status
- examples/train_with_splicevardb.py: Complete example
"""

__version__ = "0.1.0"
__author__ = "MetaSpliceAI Team"

# Import main components (only those that are implemented)
from .data.splicevardb_loader import SpliceVarDBLoader, SpliceVarDBRecord
from .core.base_predictor import OpenSpliceAIPredictor
from .workflows.splicevardb_pipeline import (
    SpliceVarDBTrainingPipeline,
    PipelineConfig
)

__all__ = [
    "__version__",
    "__author__",
    # Data loading
    "SpliceVarDBLoader",
    "SpliceVarDBRecord",
    # Prediction
    "OpenSpliceAIPredictor",
    # Training
    "SpliceVarDBTrainingPipeline",
    "PipelineConfig",
]

