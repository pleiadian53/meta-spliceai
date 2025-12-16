#!/usr/bin/env python3
"""
Run post-training diagnostics on pre-trained meta-models.

This script loads pre-trained meta-models and runs only the post-training
diagnostic routines, allowing for easier debugging and testing without
requiring the full training process.
"""

import argparse
import json
import logging
import sys
import traceback
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
import pickle

from meta_spliceai.splice_engine.meta_models.training import classifier_utils as _cutils
from meta_spliceai.splice_engine.meta_models.training.label_utils import load_dataset_sample
from meta_spliceai.splice_engine.meta_models.training.run_gene_cv_sigmoid import (
    _get_dataset_info,  # Reuse helper function for dataset info
)

logger = logging.getLogger(__name__)


def _setup_logging(verbose: bool = False):
    """Configure logging based on verbosity level."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )


def _parse_args(argv=None):
    """Parse command line arguments."""
    p = argparse.ArgumentParser(description="Run post-training diagnostics on pre-trained meta-models")
    p.add_argument("--model-dir", required=True, type=str,
                  help="Directory containing trained meta-models and metadata")
    p.add_argument("--dataset", required=True,
                  help="Path to dataset for testing (same as used during training)")
    p.add_argument("--diag-sample", type=int, default=5000,
                  help="Number of samples to use for diagnostics")
    p.add_argument("--neigh-sample", type=int, default=3,
                  help="Sample size for neighborhood diagnostics")
    p.add_argument("--neigh-window", type=int, default=12,
                  help="Window size for neighborhood diagnostics")
    p.add_argument("--plot-curves", action="store_true",
                  help="Generate ROC and PR curves")
    p.add_argument("--n-roc-points", type=int, default=201,
                  help="Number of points in ROC curve")
    p.add_argument("--plot-format", type=str, default="pdf",
                  choices=["pdf", "png", "svg"],
                  help="Format for output plots")
    p.add_argument("--include-tns", action="store_true",
                  help="Include true negatives in performance metrics")
    p.add_argument("--check-leakage", action="store_true",
                  help="Run leakage checks on features")
    p.add_argument("--leakage-probe", action="store_true",
                  help="Run detailed leakage probe analysis")
    p.add_argument("--leakage-threshold", type=float, default=0.95,
                  help="Correlation threshold for leakage detection")
    p.add_argument("--force-canonical-labels", action="store_true",
                  help="Force canonical label encoding for consistency")
    p.add_argument("--verbose", action="store_true",
                  help="Enable verbose output")
    p.add_argument("--skip-module", nargs="*", choices=[
                  "richer_metrics", "gene_score_delta", "shap_importance",
                  "base_vs_meta_diagnostic", "probability_diagnostics",
                  "meta_splice_performance", "base_vs_meta", "leakage_probe", 
                  "neighbour_diagnostics"],
                  help="Skip specific diagnostic modules")
    
    return p.parse_args(argv)


def load_model(model_dir: Path) -> Dict[str, Any]:
    """Load trained model and metadata from directory."""
    model_path = model_dir / "model_multiclass.pkl"
    metadata_path = model_dir / "metadata.json"
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Load the model
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    
    # Load metadata if available
    metadata = {}
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
    
    # Try to determine threshold from various sources
    threshold = 0.5  # Default
    
    # Check in model attributes
    if hasattr(model, "optimal_threshold"):
        threshold = model.optimal_threshold
    elif hasattr(model, "threshold"):
        threshold = model.threshold
    
    # Check in metadata
    elif "threshold" in metadata:
        threshold = metadata["threshold"]
    
    # Try to find in probability diagnostics file
    prob_diag_path = model_dir / "probability_diagnostics.json"
    if prob_diag_path.exists():
        try:
            with open(prob_diag_path, "r") as f:
                prob_diag = json.load(f)
                if "suggested_threshold" in prob_diag:
                    threshold = prob_diag["suggested_threshold"]
        except Exception as e:
            logger.warning(f"Error loading probability diagnostics: {e}")
    
    return {
        "model": model,
        "metadata": metadata,
        "threshold": threshold
    }


def run_meta_splice_performance(
    model_dir: Path,
    dataset_path: str,
    diag_sample: int,
    include_tns: bool = False,
    verbose: bool = False
) -> bool:
    """Run meta splice performance evaluation."""
    logger.info("Running meta splice performance evaluation")
    
    try:
        # Get model threshold
        model_info = load_model(model_dir)
        threshold = model_info["threshold"]
        
        if verbose:
            logger.debug(f"Using threshold: {threshold}")
        
        # Run performance evaluation
        _cutils.meta_splice_performance(
            dataset_path=dataset_path,
            run_dir=model_dir,
            threshold=threshold,
            include_tns=include_tns,
            sample=diag_sample,
            verbose=verbose
        )
        
        logger.info("Meta splice performance evaluation completed successfully")
        return True
    
    except ValueError as e:
        # Handle non-numeric data error
        if "could not convert string to float" in str(e):
            logger.warning("Detected non-numeric values in features, attempting fallback approach")
            
            try:
                # Manual implementation using our own dataset loading
                sample_df = load_dataset_sample(dataset_path, diag_sample)
                
                if verbose:
                    logger.debug(f"Loaded {len(sample_df)} samples for performance evaluation")
                
                # Create output files
                meta_tsv = model_dir / "full_splice_performance_meta.tsv"
                base_tsv = model_dir / "full_splice_performance.tsv"
                
                # Generate files directly
                if not meta_tsv.exists():
                    _cutils.evaluate_predictions(
                        df=sample_df,
                        model_dir=model_dir,
                        out_file=meta_tsv,
                        is_meta=True,
                        include_tns=include_tns
                    )
                
                if not base_tsv.exists():
                    _cutils.evaluate_predictions(
                        df=sample_df,
                        model_dir=model_dir, 
                        out_file=base_tsv,
                        is_meta=False,
                        include_tns=include_tns
                    )
                
                logger.info("Performance files generated using fallback approach")
                return True
                
            except Exception as inner_e:
                logger.error(f"Fallback approach failed: {inner_e}")
                if verbose:
                    traceback.print_exc()
                return False
        else:
            logger.error(f"Meta splice performance evaluation failed: {e}")
            if verbose:
                traceback.print_exc()
            return False
            
    except Exception as e:
        logger.error(f"Meta splice performance evaluation failed: {e}")
        if verbose:
            traceback.print_exc()
        return False


def run_base_vs_meta_comparison(
    model_dir: Path,
    verbose: bool = False
) -> bool:
    """Run base vs meta model performance comparison."""
    logger.info("Running base vs meta comparison")
    
    try:
        # Check for required files
        meta_tsv = model_dir / "full_splice_performance_meta.tsv"
        base_tsv = model_dir / "full_splice_performance.tsv"
        
        has_meta_tsv = meta_tsv.exists()
        has_base_tsv = base_tsv.exists()
        
        if verbose:
            logger.debug(f"Meta TSV exists: {has_meta_tsv}, path: {meta_tsv}")
            logger.debug(f"Base TSV exists: {has_base_tsv}, path: {base_tsv}")
        
        if not (has_meta_tsv and has_base_tsv):
            logger.warning("Missing performance files for comparison")
            return False
        
        # Run comparison
        base_vs_meta.compare_performance(
            meta_tsv=meta_tsv,
            base_tsv=base_tsv,
            out_dir=model_dir
        )
        
        logger.info("Base vs meta comparison completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Base vs meta comparison failed: {e}")
        if verbose:
            traceback.print_exc()
        return False


def run_leakage_probe(
    model_dir: Path,
    dataset_path: str,
    diag_sample: int,
    leakage_threshold: float = 0.95,
    verbose: bool = False
) -> bool:
    """Run leakage probe analysis."""
    logger.info("Running leakage probe analysis")
    
    try:
        # Run leakage probe
        _cutils.leakage_probe(
            dataset_path=dataset_path,
            run_dir=model_dir,
            threshold=leakage_threshold,
            sample=diag_sample
        )
        
        logger.info("Leakage probe completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Leakage probe failed: {e}")
        if verbose:
            traceback.print_exc()
        return False


def run_neighbour_diagnostics(
    model_dir: Path,
    dataset_path: str,
    neigh_sample: int,
    neigh_window: int,
    verbose: bool = False
) -> bool:
    """Run neighbour window diagnostics."""
    logger.info("Running neighbour window diagnostics")
    
    try:
        # Make sure output directory exists
        neighbor_dir = model_dir / "neighbor_diagnostics"
        neighbor_dir.mkdir(exist_ok=True, parents=True)
        
        # Run diagnostics
        _cutils.neighbour_window_diagnostics(
            dataset_path=dataset_path,
            run_dir=model_dir,
            window=neigh_window,
            sample=neigh_sample if neigh_sample > 0 else None
        )
        
        logger.info("Neighbour window diagnostics completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Neighbour window diagnostics failed: {e}")
        if verbose:
            traceback.print_exc()
        return False


def run_probability_diagnostics(
    model_dir: Path,
    dataset_path: str,
    diag_sample: int,
    verbose: bool = False
) -> bool:
    """Run probability diagnostics."""
    logger.info("Running probability diagnostics")
    
    try:
        # Run diagnostics
        _cutils.probability_diagnostics(
            dataset_path=dataset_path,
            run_dir=model_dir,
            sample=diag_sample
        )
        
        logger.info("Probability diagnostics completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Probability diagnostics failed: {e}")
        if verbose:
            traceback.print_exc()
        return False


def run_richer_metrics(
    model_dir: Path,
    dataset_path: str,
    diag_sample: int,
    verbose: bool = False
) -> bool:
    """Run richer metrics evaluation."""
    logger.info("Running richer metrics evaluation")
    
    try:
        # Run metrics evaluation
        _cutils.richer_metrics(
            dataset_path=dataset_path,
            run_dir=model_dir,
            sample=diag_sample
        )
        
        logger.info("Richer metrics evaluation completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Richer metrics evaluation failed: {e}")
        if verbose:
            traceback.print_exc()
        return False


def run_gene_score_delta(
    model_dir: Path,
    dataset_path: str,
    diag_sample: int,
    verbose: bool = False
) -> bool:
    """Run gene score delta analysis."""
    logger.info("Running gene score delta analysis")
    
    try:
        # Run gene score delta analysis
        _cutils.gene_score_delta(
            dataset_path=dataset_path,
            run_dir=model_dir,
            sample=diag_sample
        )
        
        logger.info("Gene score delta analysis completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Gene score delta analysis failed: {e}")
        if verbose:
            traceback.print_exc()
        return False


def run_shap_importance(
    model_dir: Path,
    dataset_path: str,
    diag_sample: int,
    verbose: bool = False
) -> bool:
    """Run SHAP importance analysis."""
    logger.info("Running SHAP importance analysis")
    
    try:
        # Run SHAP importance analysis
        _cutils.shap_importance(
            dataset_path=dataset_path,
            run_dir=model_dir,
            sample=diag_sample
        )
        
        logger.info("SHAP importance analysis completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"SHAP importance analysis failed: {e}")
        if verbose:
            traceback.print_exc()
        return False


def run_base_vs_meta_diagnostic(
    model_dir: Path,
    dataset_path: str,
    diag_sample: int,
    verbose: bool = False
):
    """Run base vs meta diagnostic (initial phase, not the comparison).

    Args:
        model_dir: Path to model directory
        dataset_path: Path to dataset
        diag_sample: Number of samples to use
        verbose: If True, print verbose output
    """
    try:
        logger.info("Running base vs meta diagnostic")
        # Call the base_vs_meta function from classifier_utils
        _cutils.base_vs_meta(dataset_path, model_dir, sample=diag_sample)
        logger.info("Base vs meta diagnostic completed")
    except Exception as e:
        if verbose:
            logger.error(f"Error in base vs meta diagnostic: {e}")
            logger.error(traceback.format_exc())
        else:
            logger.error(f"Error in base vs meta diagnostic: {e}")
        return False
    
    return True


def run_base_vs_meta_comparison(
    model_dir: Path,
    verbose: bool = False
) -> bool:
    """Run base vs meta comparison."""
    logger.info("Running base vs meta comparison")
    
    try:
        # Get paths to meta and base TSVs
        meta_tsv = model_dir / "meta_performance.tsv"
        base_tsv = model_dir / "base_performance.tsv"
        
        # Check if files exist
        has_meta_tsv = meta_tsv.exists()
        has_base_tsv = base_tsv.exists()
        
        if verbose:
            logger.debug(f"Meta TSV exists: {has_meta_tsv}, path: {meta_tsv}")
            logger.debug(f"Base TSV exists: {has_base_tsv}, path: {base_tsv}")
        
        if not (has_meta_tsv and has_base_tsv):
            logger.warning("Missing performance files for comparison")
            return False
        
        # Run comparison
        _cutils.base_vs_meta_comparison(
            meta_tsv=meta_tsv,
            base_tsv=base_tsv,
            out_dir=model_dir
        )
        
        logger.info("Base vs meta comparison completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Base vs meta comparison failed: {e}")
        if verbose:
            traceback.print_exc()
        return False


def main():
    """Main entry point."""
    args = _parse_args()
    _setup_logging(args.verbose)
    
    # Convert to Path object
    model_dir = Path(args.model_dir)
    
    # Check if model directory exists
    if not model_dir.exists():
        logger.error(f"Model directory not found: {model_dir}")
        return 1
    
    # Track which diagnostics succeeded/failed
    results = {}
    
    # Create skip set
    skip_modules = set(args.skip_module or [])
    
    # Run richer metrics
    if "richer_metrics" not in skip_modules:
        results["richer_metrics"] = run_richer_metrics(
            model_dir=model_dir,
            dataset_path=args.dataset,
            diag_sample=args.diag_sample,
            verbose=args.verbose
        )
    
    # Run gene score delta
    if "gene_score_delta" not in skip_modules:
        results["gene_score_delta"] = run_gene_score_delta(
            model_dir=model_dir,
            dataset_path=args.dataset,
            diag_sample=args.diag_sample,
            verbose=args.verbose
        )
    
    # Run SHAP importance
    if "shap_importance" not in skip_modules:
        results["shap_importance"] = run_shap_importance(
            model_dir=model_dir,
            dataset_path=args.dataset,
            diag_sample=args.diag_sample,
            verbose=args.verbose
        )
    
    # Run base vs meta diagnostic (initial)
    if "base_vs_meta_diagnostic" not in skip_modules:
        results["base_vs_meta_diagnostic"] = run_base_vs_meta_diagnostic(
            model_dir=model_dir,
            dataset_path=args.dataset,
            diag_sample=args.diag_sample,
            verbose=args.verbose
        )
        
    # Run probability diagnostics
    if "probability_diagnostics" not in skip_modules:
        results["probability_diagnostics"] = run_probability_diagnostics(
            model_dir=model_dir,
            dataset_path=args.dataset,
            diag_sample=args.diag_sample,
            verbose=args.verbose
        )
    
    # Run meta splice performance evaluation
    if "meta_splice_performance" not in skip_modules:
        results["meta_splice_performance"] = run_meta_splice_performance(
            model_dir=model_dir,
            dataset_path=args.dataset,
            diag_sample=args.diag_sample,
            include_tns=args.include_tns,
            verbose=args.verbose
        )
    
    # Run base vs meta comparison
    if "base_vs_meta" not in skip_modules:
        results["base_vs_meta"] = run_base_vs_meta_comparison(
            model_dir=model_dir,
            verbose=args.verbose
        )
    
    # Run leakage probe
    if args.leakage_probe and "leakage_probe" not in skip_modules:
        results["leakage_probe"] = run_leakage_probe(
            model_dir=model_dir,
            dataset_path=args.dataset,
            diag_sample=args.diag_sample,
            leakage_threshold=args.leakage_threshold,
            verbose=args.verbose
        )
    
    # Run neighbour diagnostics
    if args.neigh_sample > 0 and "neighbour_diagnostics" not in skip_modules:
        results["neighbour_diagnostics"] = run_neighbour_diagnostics(
            model_dir=model_dir,
            dataset_path=args.dataset,
            neigh_sample=args.neigh_sample,
            neigh_window=args.neigh_window,
            verbose=args.verbose
        )
    
    # Run probability diagnostics
    if "probability_diagnostics" not in skip_modules:
        results["probability_diagnostics"] = run_probability_diagnostics(
            model_dir=model_dir,
            dataset_path=args.dataset,
            diag_sample=args.diag_sample,
            verbose=args.verbose
        )
    
    # Print summary
    logger.info("Diagnostics Summary:")
    for module, success in results.items():
        status = "SUCCESS" if success else "FAILED"
        logger.info(f"  {module}: {status}")
    
    # Return failure if any module failed
    return 0 if all(results.values()) else 1


def run_all_diagnostics(args, model_dir: Path):
    """Run all diagnostic functions in the proper sequence."""
    # Create the correct command for each diagnostic module
    cmd_parts = [
        "python", "-m", "meta_spliceai.splice_engine.meta_models.training.run_diagnostics",
        f"--model-dir {model_dir}",
        f"--dataset {args.dataset}",
        f"--diag-sample {args.diag_sample}",
    ]
    
    # Add optional flags
    if args.verbose:
        cmd_parts.append("--verbose")
    if args.include_tns:
        cmd_parts.append("--include-tns")
    if args.leakage_probe:
        cmd_parts.append("--leakage-probe")
    if args.check_leakage:
        cmd_parts.append("--check-leakage")
    if args.plot_curves:
        cmd_parts.append("--plot-curves")
        cmd_parts.append(f"--n-roc-points {args.n_roc_points}")
        cmd_parts.append(f"--plot-format {args.plot_format}")
    if args.neigh_sample > 0:
        cmd_parts.append(f"--neigh-sample {args.neigh_sample}")
        cmd_parts.append(f"--neigh-window {args.neigh_window}")
    
    # Print the command that would run all diagnostics
    if args.verbose:
        logger.info("Complete diagnostic command:\n" + " \
  ".join(cmd_parts))


if __name__ == "__main__":
    sys.exit(main())
