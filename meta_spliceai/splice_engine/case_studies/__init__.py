"""
MetaSpliceAI Case Studies Package

This package provides data ingestion, processing, and analysis workflows
for validating the meta-learning model against disease-specific datasets
and well-characterized splice-altering mutations.

Modules:
--------
data_sources/     - Database-specific ingestion pipelines
workflows/        - Case study execution workflows  
validation/       - Validation and benchmarking utilities
formats/          - Data format converters and standardizers
diseases/         - Disease-specific analysis modules
"""

from pathlib import Path

# Package metadata
__version__ = "0.1.0"
__author__ = "MetaSpliceAI Team"

# Define standard case study data directory
CASE_STUDIES_DATA_DIR = Path(__file__).parent.parent.parent.parent / "data" / "case_studies"

# Export key classes and functions
from .data_sources import (
    SpliceVarDBIngester,
    MutSpliceDBIngester, 
    DBASSIngester,
    ClinVarIngester
)

from .workflows.disease_validation import (
    DiseaseValidationWorkflow,
    run_met_exon14_case_study
)
from .workflows.mutation_analysis import (
    MutationAnalysisWorkflow,
    analyze_cftr_cryptic_exon,
    analyze_met_exon14_cohort
)
from .workflows.alternative_splicing_pipeline import (
    AlternativeSplicingPipeline,
    AlternativeSpliceSite
)
from .workflows.regulatory_features import (
    RegulatoryFeatureExtractor,
    RegulatoryContext
)
from .workflows.disease_specific_alternative_splicing import (
    DiseaseSpecificAlternativeSplicingAnalyzer,
    DiseaseAlternativeSplicingResult,
    AlternativeSplicingPattern
)

from .formats import (
    HGVSParser,
    HGVSVariant,
    VariantStandardizer
)

__all__ = [
    'SpliceVarDBIngester', 'MutSpliceDBIngester', 'DBASSIngester', 'ClinVarIngester',
    'DiseaseValidationWorkflow', 'MutationAnalysisWorkflow',
    'AlternativeSplicingPipeline', 'AlternativeSpliceSite',
    'RegulatoryFeatureExtractor', 'RegulatoryContext',
    'DiseaseSpecificAlternativeSplicingAnalyzer', 'DiseaseAlternativeSplicingResult', 'AlternativeSplicingPattern',
    'run_met_exon14_case_study', 'analyze_cftr_cryptic_exon', 'analyze_met_exon14_cohort',
    'HGVSParser', 'HGVSVariant',
    'VariantStandardizer'
]