"""
Meta Models for MetaSpliceAI.

This package provides tools for developing and evaluating meta models that 
enhance base model predictions by correcting systematic errors.
"""

from meta_spliceai.splice_engine.meta_models.workflows.data_generation import (
    run_training_data_generation, 
    make_kmer_featurized_dataset
)
