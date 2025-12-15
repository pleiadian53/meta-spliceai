#!/usr/bin/env python
"""
Demonstration script for the refactored error classifier components.

This script provides examples of how to use the new model_training package
to train error classifiers, while preserving the original implementations.

Usage:
    python -m meta_spliceai.splice_engine.demo_error_classifier
"""

import os
import matplotlib.pyplot as plt
from pathlib import Path

from .model_training.error_classifier import train_error_classifier
from .model_training.xgboost_trainer import xgboost_pipeline
from .splice_error_analyzer import ErrorAnalyzer
from .utils_doc import print_emphasized, print_section_separator, print_with_indent


def run_basic_demo():
    """Run a basic demonstration of the error classifier."""
    print_emphasized("Running basic error classifier demo")
    print_section_separator()
    
    # Define output directory
    output_dir = os.path.join(ErrorAnalyzer.analysis_dir, "error_classifier_demo")
    os.makedirs(output_dir, exist_ok=True)
    
    # Train a simple FP vs TP classifier
    model, results = train_error_classifier(
        error_label="FP",
        correct_label="TP",
        splice_type="donor",  # Optional: focus on just donor sites
        remove_strong_predictors=True,  # Remove 'score' to focus on other features
        output_dir=output_dir,
        top_k=15,
        n_splits=5,
        model_type="xgboost",
        verbose=1
    )
    
    print_section_separator()
    print_emphasized("Demo completed - Results saved to:")
    print_with_indent(output_dir, indent_level=1)
    
    return model, results


def compare_donor_acceptor():
    """Compare feature importance between donor and acceptor site errors."""
    print_emphasized("Comparing donor vs acceptor error patterns")
    print_section_separator()
    
    # Define output directory
    output_dir = os.path.join(ErrorAnalyzer.analysis_dir, "donor_acceptor_comparison")
    os.makedirs(output_dir, exist_ok=True)
    
    # Train models for each splice type
    donor_model, donor_results = train_error_classifier(
        error_label="FP",
        correct_label="TP",
        splice_type="donor",
        remove_strong_predictors=True,
        output_dir=output_dir,
        subject="donor_fp_vs_tp",
        top_k=10,
        model_type="xgboost",
        verbose=1
    )
    
    acceptor_model, acceptor_results = train_error_classifier(
        error_label="FP",
        correct_label="TP",
        splice_type="acceptor",
        remove_strong_predictors=True,
        output_dir=output_dir,
        subject="acceptor_fp_vs_tp",
        top_k=10,
        model_type="xgboost",
        verbose=1
    )
    
    # Compare SHAP importance between donor and acceptor
    donor_shap = donor_results['importance_df_shap'].set_index('feature')
    acceptor_shap = acceptor_results['importance_df_shap'].set_index('feature')
    
    # Find common features
    common_features = list(set(donor_shap.index) & set(acceptor_shap.index))
    
    if common_features:
        print_emphasized("Comparing feature importance between donor and acceptor sites")
        plt.figure(figsize=(12, 8))
        
        # Create comparison plot
        donor_vals = donor_shap.loc[common_features].head(10)
        acceptor_vals = acceptor_shap.loc[common_features].head(10)
        
        # Sort by combined importance
        combined = donor_vals['importance_score'] + acceptor_vals['importance_score']
        sorted_features = combined.sort_values(ascending=True).index
        
        # Plot
        y_pos = range(len(sorted_features))
        width = 0.35
        
        plt.barh([p + width/2 for p in y_pos], 
                donor_vals.loc[sorted_features]['importance_score'], 
                height=width, label='Donor Sites')
        plt.barh([p - width/2 for p in y_pos], 
                acceptor_vals.loc[sorted_features]['importance_score'], 
                height=width, label='Acceptor Sites')
        
        plt.yticks(y_pos, sorted_features)
        plt.xlabel('SHAP Importance')
        plt.title('Feature Importance Comparison: Donor vs Acceptor')
        plt.legend()
        plt.tight_layout()
        
        # Save plot
        output_path = os.path.join(output_dir, "donor_acceptor_comparison.pdf")
        plt.savefig(output_path, dpi=300)
        print_with_indent(f"Saved comparison plot to: {output_path}", indent_level=1)
    
    print_section_separator()
    print_emphasized("Comparison completed")
    
    return {
        'donor': {'model': donor_model, 'results': donor_results},
        'acceptor': {'model': acceptor_model, 'results': acceptor_results}
    }


def run_xgboost_pipeline_directly(X, y):
    """
    Example of using xgboost_pipeline directly with a DataFrame.
    
    Parameters
    ----------
    X : pd.DataFrame
        Feature DataFrame
    y : pd.Series
        Target labels
    """
    print_emphasized("Running XGBoost pipeline directly")
    
    output_dir = os.path.join(ErrorAnalyzer.analysis_dir, "direct_xgboost_demo")
    os.makedirs(output_dir, exist_ok=True)
    
    # Run pipeline
    model, results = xgboost_pipeline(
        X, y,
        output_dir=output_dir,
        subject="custom_analysis",
        top_k=15,
        n_splits=5,
        verbose=1
    )
    
    print_section_separator()
    print_emphasized("XGBoost pipeline completed")
    
    return model, results


def main():
    """Main function to run demonstrations."""
    print_emphasized("ERROR CLASSIFIER DEMONSTRATION")
    print_section_separator()
    print("This script demonstrates using the refactored model_training components.")
    print("Choose an example to run:")
    print("1. Basic error classifier demo (FP vs TP, donor sites)")
    print("2. Compare donor and acceptor error patterns")
    print("3. Run all demonstrations")
    print("4. Exit")
    
    choice = input("\nEnter your choice (1-4): ")
    
    if choice == '1':
        model, results = run_basic_demo()
    elif choice == '2':
        compare_results = compare_donor_acceptor()
    elif choice == '3':
        # Run all demos
        print_emphasized("Running all demonstrations")
        model, results = run_basic_demo()
        compare_results = compare_donor_acceptor()
        print_emphasized("All demonstrations completed")
    elif choice == '4':
        print("Exiting...")
        return 0
    else:
        print("Invalid choice. Exiting...")
        return 1
    
    return 0


if __name__ == "__main__":
    main()
