"""
Case study workflows.

This module provides high-level workflows for executing disease-specific
case studies and validation analyses using the meta-learning model.
"""

# Import only when explicitly needed to avoid circular dependencies
# from .disease_validation import DiseaseValidationWorkflow
# from .mutation_analysis import MutationAnalysisWorkflow

__all__ = [
    "DiseaseValidationWorkflow",
    "MutationAnalysisWorkflow",
    "AlternativeSplicingPipeline",
    "OpenSpliceAIDeltaBridge",
] 