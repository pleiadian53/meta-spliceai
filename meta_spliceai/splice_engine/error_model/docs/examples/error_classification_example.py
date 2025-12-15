#!/usr/bin/env python
"""
Example script demonstrating the error analysis workflow in MetaSpliceAI.

This example shows how to:
1. Load a dataset with splice site prediction errors
2. Train an error classification model
3. Generate feature importance analysis
4. Visualize results

Usage:
    conda run -n surveyor python error_classification_example.py --input path/to/labeled_data.tsv --output path/to/output_dir

The input file should contain labeled splice sites (TP, FP, FN, TN) with feature columns.
"""

import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from meta_spliceai.splice_engine.error_model import (
    process_error_model,
    apply_stratified_sampling,
    verify_error_model_outputs,
    get_model_status
)

from meta_spliceai.splice_engine.error_model.utils import (
    safely_save_figure,
    select_samples_for_analysis
)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Error classification example")
    parser.add_argument("--input", required=True, help="Path to labeled splice sites file")
    parser.add_argument("--output", required=True, help="Path to output directory")
    parser.add_argument("--sample-size", type=int, default=100000, 
                        help="Number of samples to use (default: 100000)")
    parser.add_argument("--cv-folds", type=int, default=5, 
                        help="Number of cross-validation folds (default: 5)")
    parser.add_argument("--error-type", choices=["all", "fp", "fn"], default="all", 
                        help="Type of error to focus on (default: all)")
    return parser.parse_args()


def prepare_data(input_file, sample_size, error_type="all"):
    """Load and prepare data for error analysis."""
    print(f"Loading data from {input_file}")
    data = pd.read_csv(input_file, sep="\t")
    
    # Create binary target for error classification
    if error_type == "all":
        # 1 for any error (FP or FN), 0 for correct predictions (TP or TN)
        data["is_error"] = data["prediction_type"].isin(["FP", "FN"]).astype(int)
    elif error_type == "fp":
        # Focus on false positives only (against true negatives)
        data = data[data["prediction_type"].isin(["FP", "TN"])]
        data["is_error"] = (data["prediction_type"] == "FP").astype(int)
    elif error_type == "fn":
        # Focus on false negatives only (against true positives)
        data = data[data["prediction_type"].isin(["FN", "TP"])]
        data["is_error"] = (data["prediction_type"] == "FN").astype(int)
    
    # Apply stratified sampling to maintain class balance
    if sample_size and sample_size < len(data):
        print(f"Applying stratified sampling to reduce data to {sample_size} samples")
        data = apply_stratified_sampling(
            data, 
            target_column="is_error",
            sample_size=sample_size
        )
    
    # Identify feature columns (example - adjust to your dataset)
    sequence_cols = [col for col in data.columns if col.startswith("seq_")]
    gene_cols = [col for col in data.columns if col.startswith("gene_")]
    transcript_cols = [col for col in data.columns if col.startswith("transcript_")]
    
    # Combine all feature columns
    feature_columns = sequence_cols + gene_cols + transcript_cols
    
    print(f"Dataset shape: {data.shape}")
    print(f"Number of features: {len(feature_columns)}")
    print(f"Class distribution: {data['is_error'].value_counts(normalize=True)}")
    
    return data, "is_error", feature_columns


def main():
    """Run the error analysis example workflow."""
    args = parse_arguments()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    # Prepare data
    data, target_column, feature_columns = prepare_data(
        args.input, 
        args.sample_size,
        args.error_type
    )
    
    # Run the error model workflow
    print(f"Training error classification model with {args.cv_folds}-fold cross-validation")
    results = process_error_model(
        data=data,
        output_dir=args.output,
        feature_columns=feature_columns,
        target_column=target_column,
        model_type="xgboost",
        cv_folds=args.cv_folds,
        generate_shap=True,
        verbose=True
    )
    
    # Verify outputs were generated
    print("\nVerifying model outputs...")
    verification = verify_error_model_outputs(args.output)
    print(f"Output verification: {'Success' if verification else 'Failed'}")
    
    # Get model performance
    model_status = get_model_status(args.output)
    print("\nModel Performance:")
    for metric, value in model_status.items():
        print(f"  {metric}: {value}")
    
    # Create custom visualization
    create_custom_visualization(data, args.output)
    
    print(f"\nAnalysis complete. Results saved to {args.output}")


def create_custom_visualization(data, output_dir):
    """Create a custom visualization of error distribution."""
    if "gene_type" not in data.columns or "prediction_type" not in data.columns:
        print("Skipping custom visualization - required columns not found")
        return
    
    print("\nCreating custom error distribution visualization...")
    
    # Plot error distribution by gene type
    plt.figure(figsize=(12, 8))
    
    # Calculate error rates by gene type
    gene_error_rates = data.groupby("gene_type")["is_error"].mean().reset_index()
    gene_error_rates = gene_error_rates.sort_values("is_error", ascending=False)
    
    # Create bar plot
    sns.barplot(data=gene_error_rates, x="gene_type", y="is_error")
    plt.title("Error Rate by Gene Type")
    plt.xlabel("Gene Type")
    plt.ylabel("Error Rate")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    
    # Save figure safely
    output_path = os.path.join(output_dir, "error_rate_by_gene_type.pdf")
    safely_save_figure(output_path)
    print(f"Custom visualization saved to {output_path}")


if __name__ == "__main__":
    main()
