"""
MetaSpliceAI Case Studies Entry Points

This module provides command-line entry points for MetaSpliceAI case study tools.
These scripts are designed to be run directly from the command line and provide
easy access to the main functionality of the case studies package.

Available Entry Points:
- run_complete_clinvar_pipeline.py: Complete ClinVar processing pipeline
- run_vcf_column_documenter.py: VCF column analysis and documentation tool

Usage:
    # Run from project root
    python meta_spliceai/splice_engine/case_studies/entry_points/run_complete_clinvar_pipeline.py --help
    
    # Or add to PATH and run directly
    export PATH=$PATH:$(pwd)/meta_spliceai/splice_engine/case_studies/entry_points
    run_complete_clinvar_pipeline.py --help
"""

__version__ = "1.0.0"
__author__ = "MetaSpliceAI Team"
