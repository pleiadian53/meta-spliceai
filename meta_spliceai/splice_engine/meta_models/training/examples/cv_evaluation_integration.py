#!/usr/bin/env python3
"""
Example of integrating CV evaluation extension with existing CV workflows.

This script demonstrates how to incorporate the enhanced evaluation features
into an existing CV pipeline without modifying the original scripts.
"""
import os
import argparse
from pathlib import Path
from typing import Dict, List, Any

# Import the CV evaluation utilities
from meta_spliceai.splice_engine.meta_models.evaluation.cv_evaluation import (
    run_full_cv_evaluation
)
from meta_spliceai.splice_engine.meta_models.evaluation.transcript_level_cv import (
    run_transcript_level_cv_evaluation
)


def collect_fold_results(cv_output_dir: Path, fold_pattern: str = "fold_*") -> List[Dict[str, Dict[str, Any]]]:
    """
    Collect performance metrics from fold directories.
    
    Parameters
    ----------
    cv_output_dir : Path
        Directory containing cross-validation results
    fold_pattern : str, default="fold_*"
        Pattern to match fold directories
        
    Returns
    -------
    List[Dict[str, Dict[str, Any]]]
        List of dicts with base and meta model metrics for each fold
    """
    # Import here to avoid circular imports
    from meta_spliceai.splice_engine.meta_models.training.run_cv_evaluation import (
        find_fold_directories,
        extract_metrics_from_fold
    )
    
    # Find fold directories
    fold_dirs = find_fold_directories(cv_output_dir, fold_pattern)
    
    # Extract metrics from each fold
    fold_results = []
    for fold_dir in fold_dirs:
        fold_metrics = extract_metrics_from_fold(fold_dir)
        fold_results.append(fold_metrics)
    
    return fold_results


def integrate_with_gene_cv(
    cv_output_dir: str,
    annotations_path: str,
    output_dir: str = None,
    base_model_name: str = "SpliceAI",
    meta_model_name: str = "Meta-Model",
    transcript_k_values: List[int] = None
):
    """
    Integrate CV evaluation with an existing gene CV output directory.
    
    Parameters
    ----------
    cv_output_dir : str
        Directory with existing cross-validation results
    annotations_path : str
        Path to annotations file for transcript mapping
    output_dir : str, optional
        Directory to save evaluation outputs (defaults to cv_output_dir/evaluation)
    base_model_name : str
        Name of the base model
    meta_model_name : str
        Name of the meta model
    transcript_k_values : List[int], optional
        K values for transcript-level evaluation (default: [1, 3, 5, 10])
    """
    # Setup paths
    cv_output_dir = Path(cv_output_dir)
    if output_dir is None:
        output_dir = cv_output_dir / "evaluation"
    else:
        output_dir = Path(output_dir)
    
    # Default k values if not provided
    if transcript_k_values is None:
        transcript_k_values = [1, 3, 5, 10]
    
    print("="*80)
    print(f"INTEGRATING CV EVALUATION WITH: {cv_output_dir}")
    print("="*80)
    
    # Collect fold results
    print("Collecting fold results...")
    fold_results = collect_fold_results(cv_output_dir)
    
    if not fold_results:
        print("No fold results found. Please check your CV output directory.")
        return
    
    # Run CV evaluation
    print("\nRunning cross-validation evaluation...")
    cv_results = run_full_cv_evaluation(
        fold_results=fold_results,
        output_dir=output_dir,
        base_model_name=base_model_name,
        meta_model_name=meta_model_name
    )
    
    # Print headline metrics
    print("\nHeadline Metrics:")
    print(f"  F1 Improvement: {cv_results['headline_metrics']['f1_improvement']:.2f}%")
    print(f"  FP Reduction: {cv_results['headline_metrics']['fp_reduction']:.2f}%") 
    print(f"  FN Reduction: {cv_results['headline_metrics']['fn_reduction']:.2f}%")
    
    # Collect fold predictions for transcript-level evaluation
    print("\nCollecting fold predictions for transcript-level evaluation...")
    fold_predictions = []
    for i, fold_dir in enumerate(find_fold_directories(cv_output_dir)):
        # Look for predictions file in common locations
        predictions_file = None
        candidates = [
            fold_dir / "meta_predictions.csv",
            fold_dir / "meta_predictions.tsv",
            fold_dir / "predictions.csv",
            fold_dir / "predictions.tsv"
        ]
        
        for candidate in candidates:
            if candidate.exists():
                predictions_file = candidate
                break
        
        if predictions_file:
            fold_predictions.append((predictions_file, i+1))
    
    # Run transcript-level evaluation if predictions are available
    if fold_predictions:
        print(f"Running transcript-level evaluation with k={transcript_k_values}...")
        transcript_results = run_transcript_level_cv_evaluation(
            fold_predictions=fold_predictions,
            annotations_file=annotations_path,
            output_dir=output_dir,
            k_values=transcript_k_values
        )
        
        # Print transcript-level summary
        print("\nTranscript-Level Top-K Accuracy:")
        for k in transcript_k_values:
            mean_key = f"top_{k}_mean"
            std_key = f"top_{k}_std"
            if mean_key in transcript_results["summary"] and std_key in transcript_results["summary"]:
                mean = transcript_results["summary"][mean_key]
                std = transcript_results["summary"][std_key]
                print(f"  Top-{k}: {mean:.4f} ± {std:.4f}")
    else:
        print("\nNo prediction files found for transcript-level evaluation")
    
    print("\nEvaluation complete!")
    print(f"Results saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Integrate CV evaluation extension with existing CV workflows"
    )
    parser.add_argument(
        "--cv-output-dir", 
        required=True, 
        help="Directory containing cross-validation results"
    )
    parser.add_argument(
        "--annotations-path",
        required=True,
        help="Path to annotations file for transcript mapping"
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for evaluation results (default: cv-output-dir/evaluation)"
    )
    parser.add_argument(
        "--base-model-name",
        default="SpliceAI",
        help="Name of the base model"
    )
    parser.add_argument(
        "--meta-model-name",
        default="Meta-Model",
        help="Name of the meta model"
    )
    parser.add_argument(
        "--transcript-k-values",
        nargs="+",
        type=int,
        default=[1, 3, 5, 10],
        help="K values for transcript-level evaluation"
    )
    
    args = parser.parse_args()
    
    integrate_with_gene_cv(
        cv_output_dir=args.cv_output_dir,
        annotations_path=args.annotations_path,
        output_dir=args.output_dir,
        base_model_name=args.base_model_name,
        meta_model_name=args.meta_model_name,
        transcript_k_values=args.transcript_k_values
    )


if __name__ == "__main__":
    main()
