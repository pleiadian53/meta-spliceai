#!/usr/bin/env python3
"""
Cross-Validation Evaluation Runner for MetaSpliceAI Meta-Models (Extension Module)

IMPORTANT: This is an extension to the existing CV evaluation pipeline and will not
interfere with or modify the behavior of the standard gene_cv and chromosome_cv scripts.

Cross-Validation Evaluation Runner for MetaSpliceAI Meta-Models

This script analyzes fold-by-fold results from gene-aware or chromosome-aware 
cross-validation runs and generates comprehensive performance reports and visualizations.

Example usage:
python -m meta_spliceai.splice_engine.meta_models.training.run_cv_evaluation \
    --cv-output-dir models/meta_model_per_class_calibrated \
    --base-model-name "SpliceAI" \
    --meta-model-name "Meta-Model" \
    --fold-pattern "fold_*" \
    --out-dir evaluation_reports
"""

import os
import re
import argparse
import json
import glob
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union

# Script version
__version__ = "0.1.0"

import pandas as pd
import numpy as np

from meta_spliceai.splice_engine.meta_models.evaluation.cv_evaluation import (
    calculate_fold_metrics,
    save_fold_metrics,
    aggregate_fold_metrics,
    plot_fold_distributions,
    create_cv_evaluation_report,
    run_full_cv_evaluation
)


def find_fold_directories(cv_output_dir: Union[str, Path], fold_pattern: str = "fold_*") -> List[Path]:
    """
    Find fold directories matching the given pattern.
    
    Parameters
    ----------
    cv_output_dir : str or Path
        The main directory containing cross-validation results
    fold_pattern : str
        Pattern to match fold directories
        
    Returns
    -------
    List[Path]
        List of fold directories
    """
    cv_output_dir = Path(cv_output_dir)
    
    # Find all directories matching the pattern
    fold_dirs = list(cv_output_dir.glob(fold_pattern))
    
    # Sort by fold number if possible
    def extract_fold_number(path):
        match = re.search(r'fold_(\d+)', str(path))
        return int(match.group(1)) if match else float('inf')
    
    fold_dirs.sort(key=extract_fold_number)
    
    return fold_dirs


def extract_metrics_from_fold(fold_dir: Path) -> Dict[str, Dict[str, Any]]:
    """
    Extract base and meta model metrics from a fold directory.
    
    Parameters
    ----------
    fold_dir : Path
        Directory containing fold results
        
    Returns
    -------
    Dict
        Dictionary with 'base' and 'meta' keys containing respective metrics
    """
    # Look for performance files
    base_perf_file = fold_dir / "base_performance.json"
    meta_perf_file = fold_dir / "meta_performance.json"
    
    # Alternative locations
    if not base_perf_file.exists():
        base_perf_file = fold_dir / "base_metrics.json"
    if not meta_perf_file.exists():
        meta_perf_file = fold_dir / "meta_metrics.json"
    
    # If still not found, check for TSV files and parse them
    if not base_perf_file.exists():
        base_tsv = fold_dir / "full_splice_performance.tsv"
        if base_tsv.exists():
            base_metrics = parse_performance_tsv(base_tsv)
        else:
            print(f"Warning: No base performance found for {fold_dir}")
            base_metrics = {"TP": 0, "FP": 0, "FN": 0, "TN": 0}
    else:
        with open(base_perf_file, "r") as f:
            base_metrics = json.load(f)
    
    if not meta_perf_file.exists():
        meta_tsv = fold_dir / "full_splice_performance_meta.tsv"
        if meta_tsv.exists():
            meta_metrics = parse_performance_tsv(meta_tsv)
        else:
            print(f"Warning: No meta performance found for {fold_dir}")
            meta_metrics = {"TP": 0, "FP": 0, "FN": 0, "TN": 0}
    else:
        with open(meta_perf_file, "r") as f:
            meta_metrics = json.load(f)
    
    return {
        "base": base_metrics,
        "meta": meta_metrics
    }


def parse_performance_tsv(tsv_path: Path) -> Dict[str, Any]:
    """
    Parse a performance TSV file into a metrics dictionary.
    
    Parameters
    ----------
    tsv_path : Path
        Path to the performance TSV file
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing TP, FP, FN, TN counts
    """
    try:
        df = pd.read_csv(tsv_path, sep='\t')
        
        # Try to extract metrics based on known column structures
        metrics = {}
        
        # Try to get TP, FP, FN, TN from direct columns if they exist
        for metric in ["TP", "FP", "FN", "TN"]:
            if metric in df.columns:
                metrics[metric] = df[metric].sum()
            
        # If metrics are still empty, try to calculate from other columns
        if not metrics:
            # Look for standard columns that might be present
            if all(col in df.columns for col in ["prediction", "truth"]):
                # Convert truth/prediction to TP, FP, FN, TN
                metrics["TP"] = ((df["prediction"] == 1) & (df["truth"] == 1)).sum()
                metrics["FP"] = ((df["prediction"] == 1) & (df["truth"] == 0)).sum()
                metrics["FN"] = ((df["prediction"] == 0) & (df["truth"] == 1)).sum()
                metrics["TN"] = ((df["prediction"] == 0) & (df["truth"] == 0)).sum()
        
        # Ensure we have all required metrics
        required_metrics = ["TP", "FP", "FN", "TN"]
        missing = [m for m in required_metrics if m not in metrics]
        if missing:
            print(f"Warning: Missing metrics in {tsv_path}: {missing}")
            for m in missing:
                metrics[m] = 0
                
        return metrics
    
    except Exception as e:
        print(f"Error parsing {tsv_path}: {e}")
        return {"TP": 0, "FP": 0, "FN": 0, "TN": 0}


def get_version() -> str:
    """Return the version of this script."""
    return __version__


def main():
    parser = argparse.ArgumentParser(description="Generate cross-validation evaluation reports and visualizations")
    parser.add_argument("--cv-output-dir", required=True, help="Directory containing cross-validation results")
    parser.add_argument("--base-model-name", default="SpliceAI", help="Name of the base model")
    parser.add_argument("--meta-model-name", default="Meta-Model", help="Name of the meta model")
    parser.add_argument("--fold-pattern", default="fold_*", help="Pattern to match fold directories")
    parser.add_argument("--confidence-level", type=float, default=0.95, help="Confidence level for intervals")
    parser.add_argument("--out-dir", default=None, help="Output directory for reports (defaults to cv-output-dir/evaluation)")
    
    args = parser.parse_args()
    
    # Set up output directory
    if args.out_dir:
        output_dir = Path(args.out_dir)
    else:
        output_dir = Path(args.cv_output_dir) / "evaluation"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"CV Evaluation Extension v{get_version()}")
    print(f"Searching for fold directories in {args.cv_output_dir}...")
    fold_dirs = find_fold_directories(args.cv_output_dir, args.fold_pattern)
    print(f"Found {len(fold_dirs)} fold directories")
    
    if not fold_dirs:
        print(f"Error: No fold directories matching '{args.fold_pattern}' found in {args.cv_output_dir}")
        return
    
    print("Extracting metrics from each fold...")
    fold_results = []
    for fold_dir in fold_dirs:
        print(f"  Processing {fold_dir.name}...")
        fold_metrics = extract_metrics_from_fold(fold_dir)
        fold_results.append(fold_metrics)
    
    print("Running cross-validation evaluation...")
    results = run_full_cv_evaluation(
        fold_results=fold_results,
        output_dir=output_dir,
        base_model_name=args.base_model_name,
        meta_model_name=args.meta_model_name,
        confidence_level=args.confidence_level
    )
    
    print("\nEvaluation complete!")
    print(f"Report saved to: {results['report_path']}")
    print(f"Headline metrics:")
    print(f"  - F1 improvement: {results['headline_metrics']['f1_improvement']:.2f}%")
    print(f"  - FP reduction: {results['headline_metrics']['fp_reduction']:.2f}%")
    print(f"  - FN reduction: {results['headline_metrics']['fn_reduction']:.2f}%")
    print(f"\nVisualizations saved to: {output_dir}")


if __name__ == "__main__":
    main()
