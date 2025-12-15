"""
Error Model package for splice site error analysis and modeling.

This package contains modules for analyzing and modeling errors in splice site prediction,
including both traditional ML models and sequence-based neural models.

Commonly imported directly from the package level:
- ErrorClassifier: Main class for error classification models.
- process_error_model(), process_all_error_models(): Core functions for workflow automation.
- verify_error_model_outputs(): Utility for checking output integrity.

Modules provided:
- classifier: Classification-related classes and methods.
- workflow: Error modeling workflow utilities.
- sequence: Sequence data utilities for error analysis.
- utils: Miscellaneous utility functions for error modeling tasks.
"""


from . import classifier
from . import workflow
from . import sequence
from . import utils
from . import training

# Expose commonly used functions at the package level
from .workflow import process_error_model, process_all_error_models, apply_stratified_sampling, load_and_subsample_dataset
from .utils import (
    verify_error_model_outputs, 
    get_model_status, 
    get_model_file_paths, 
    safely_save_figure,
    select_samples_for_analysis,
    get_sample_shap_files,
    get_analysis_summary
)
from .classifier import ErrorClassifier

# List of all public symbols in the public API
__all__ = [
    'classifier', 
    'workflow', 
    'sequence', 
    'utils',
    'training',
    'process_error_model',
    'process_all_error_models',
    'apply_stratified_sampling',
    'load_and_subsample_dataset',
    'verify_error_model_outputs',
    'get_model_status',
    'get_model_file_paths',
    'safely_save_figure',
    'select_samples_for_analysis',
    'get_sample_shap_files',
    'get_analysis_summary',
    'ErrorClassifier'
]
# NOTE: 
#  - When someone uses from module import *, Python will only import names listed in __all__
#  - Without __all__, Python imports all names that don't start with an underscore (_)
