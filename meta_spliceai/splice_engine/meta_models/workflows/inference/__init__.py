"""
Inference workflow modules for selective meta-model inference.

This package contains modular components extracted from selective_meta_inference.py
to improve maintainability and reduce complexity.
"""

from .config import (
    SelectiveInferenceConfig,
    SelectiveInferenceResults,
    create_selective_config
)

from .verification import (
    verify_selective_featurization,
    verify_no_label_leakage
)

from .io_utils import (
    setup_inference_directories,
    track_processed_genes,
    load_processed_genes,
    create_gene_manifest,
    get_test_data_directory
)

from .prediction_combiner import (
    combine_predictions_for_complete_coverage
)

from .feature_processor import (
    generate_selective_meta_predictions,
    generate_chunked_meta_predictions
)

__all__ = [
    # Config
    'SelectiveInferenceConfig',
    'SelectiveInferenceResults',
    'create_selective_config',
    # Verification
    'verify_selective_featurization',
    'verify_no_label_leakage',
    # I/O
    'setup_inference_directories',
    'track_processed_genes',
    'load_processed_genes',
    'create_gene_manifest',
    'get_test_data_directory',
    # Processing
    'combine_predictions_for_complete_coverage',
    'generate_selective_meta_predictions',
    'generate_chunked_meta_predictions',
]