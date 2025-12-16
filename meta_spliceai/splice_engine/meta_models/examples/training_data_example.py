"""
Example: Generating training data for meta models using the refactored codebase.

This script demonstrates how to use the refactored meta_models package to
generate training data for meta models that correct base model predictions.
"""

import os
import sys
import pandas as pd
import polars as pl

# Import the refactored meta model components
from meta_spliceai.splice_engine.meta_models.core.data_types import MetaModelConfig
from meta_spliceai.splice_engine.meta_models.workflows.data_generation import run_training_data_generation
from meta_spliceai.splice_engine.extract_genomic_features import FeatureAnalyzer
from meta_spliceai.splice_engine.splice_error_analyzer import ErrorAnalyzer


def main():
    """Run a complete meta model training data generation example."""
    
    # Get the GTF file path from ErrorAnalyzer
    gtf_file = ErrorAnalyzer.gtf_file
    
    # Print information about the example
    print("=" * 80)
    print(" Meta Model Training Data Generation Example ")
    print("=" * 80)
    print(f"GTF file: {gtf_file}")
    
    # Create a configuration for FP vs TP meta model
    fp_tp_config = MetaModelConfig(
        gtf_file=gtf_file,
        pred_type='FP',
        error_label='FP',
        correct_label='TP',
        kmer_sizes=[3, 6],  # Use both 3-mers and 6-mers
        subset_genes=True,
        subset_policy='hard',  # Focus on "hard" genes where the base model struggles
        n_genes=50,  # Use a small number for the example
        overwrite=False  # Set to True to regenerate data
    )
    
    # Run the training data generation workflow for FP vs TP
    print("\nGenerating training data for FP vs TP meta model...")
    fp_tp_dataset = run_training_data_generation(fp_tp_config)
    
    # Display information about the dataset
    print(f"\nFP vs TP dataset shape: {fp_tp_dataset.shape}")
    print(f"Number of unique genes: {fp_tp_dataset['gene_id'].nunique()}")
    print(f"Label distribution: {fp_tp_dataset['label'].value_counts()}")
    print("\nFeature preview:")
    print(fp_tp_dataset.head(2))

    # Create a configuration for FN vs TP meta model
    fn_tp_config = MetaModelConfig(
        gtf_file=gtf_file,
        pred_type='FN',
        error_label='FN',
        correct_label='TP',
        kmer_sizes=[3, 6],
        subset_genes=True,
        subset_policy='hard',
        n_genes=50,
        overwrite=False
    )
    
    # Run the training data generation workflow for FN vs TP
    print("\nGenerating training data for FN vs TP meta model...")
    fn_tp_dataset = run_training_data_generation(fn_tp_config)
    
    # Display information about the dataset
    print(f"\nFN vs TP dataset shape: {fn_tp_dataset.shape}")
    print(f"Number of unique genes: {fn_tp_dataset['gene_id'].nunique()}")
    print(f"Label distribution: {fn_tp_dataset['label'].value_counts()}")
    
    # Create a configuration for FN vs TN meta model
    fn_tn_config = MetaModelConfig(
        gtf_file=gtf_file,
        pred_type='FN',
        error_label='FN',
        correct_label='TN',
        kmer_sizes=[3, 6],
        subset_genes=True,
        subset_policy='hard',
        n_genes=50,
        overwrite=False
    )
    
    # Run the training data generation workflow for FN vs TN
    print("\nGenerating training data for FN vs TN meta model...")
    fn_tn_dataset = run_training_data_generation(fn_tn_config)
    
    # Display information about the dataset
    print(f"\nFN vs TN dataset shape: {fn_tn_dataset.shape}")
    print(f"Number of unique genes: {fn_tn_dataset['gene_id'].nunique()}")
    print(f"Label distribution: {fn_tn_dataset['label'].value_counts()}")
    
    print("\nAll datasets generated successfully!")


if __name__ == "__main__":
    main()
