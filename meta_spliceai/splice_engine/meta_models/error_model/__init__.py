"""
Deep Error Model Subpackage for Meta-Learning

This subpackage implements deep learning models for analyzing and correcting
splice site prediction errors using transformer-based architectures and 
Integrated Gradients analysis.

The package is designed to work with position-centric data representation
from the meta_models artifacts (analysis_sequences_*, splice_positions_enhanced_*).
"""

from .config import ErrorModelConfig
from .modeling.transformer_trainer import TransformerTrainer
from .modeling.ig_analyzer import IGAnalyzer
from .visualization.alignment_plot import AlignmentPlotter
from .visualization.frequency_plot import FrequencyPlotter
from .dataset.dataset_preparer import ErrorDatasetPreparer
from .dataset.data_utils import DataUtils, DNASequenceDataset

# Analysis configurations
from .modeling.ig_analyzer import IGAnalysisConfig

# Data consolidation utilities
from .dataset.consolidate_analysis_data import (
    AnalysisDataConsolidator,
    consolidate_and_filter_analysis_sequences
)

__all__ = [
    'ErrorModelConfig',
    'TransformerTrainer', 
    'IGAnalyzer',
    'IGAnalysisConfig',
    'AlignmentPlotter',
    'FrequencyPlotter',
    'ErrorDatasetPreparer',
    'DataUtils',
    'DNASequenceDataset',
    'AnalysisDataConsolidator',
    'consolidate_and_filter_analysis_sequences'
]
