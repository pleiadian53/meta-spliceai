"""
Feature Importance Module for MetaSpliceAI.

This module provides various methods for quantifying feature importance in splice site
prediction models, including:
- Mutual information-based importance
- Effect size analysis
- Hypothesis testing
- SHAP-based importance analysis
- XGBoost native importance scores

Each submodule implements one approach to feature importance quantification while
maintaining a consistent API for ease of use.
"""

# Import public APIs from submodules
from .mutual_info import quantify_feature_importance_via_mutual_info
from .effect_sizes import quantify_feature_importance_via_measuring_effect_sizes
from .hypothesis_testing import quantify_feature_importance_via_hypothesis_testing
from .shap_analysis import (
    compute_shap_values,
    compute_global_feature_importance_from_shap,
    analyze_feature_group_importance,
    identify_important_motifs_from_shap,
    quantify_feature_importance_via_shap
)
from .xgboost_importance import get_xgboost_feature_importance

# Import standardized wrapper functions with consistent naming
from .wrappers import (
    calculate_xgboost_importance,
    calculate_shap_importance,
    calculate_hypothesis_testing,
    calculate_effect_sizes,
    calculate_mutual_information
)

# Version
__version__ = "0.1.0"

__all__ = [
    # Original detailed API
    'quantify_feature_importance_via_mutual_info',
    'quantify_feature_importance_via_measuring_effect_sizes',
    'quantify_feature_importance_via_hypothesis_testing',
    'compute_shap_values',
    'compute_global_feature_importance_from_shap',
    'analyze_feature_group_importance',
    'identify_important_motifs_from_shap',
    'quantify_feature_importance_via_shap',
    'get_xgboost_feature_importance',
    
    # Standardized API
    'calculate_xgboost_importance',
    'calculate_shap_importance',
    'calculate_hypothesis_testing',
    'calculate_effect_sizes',
    'calculate_mutual_information'
]
