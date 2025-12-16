#!/usr/bin/env python3
"""Comprehensive diagnostic runner for the splice meta-model training pipeline.

This script orchestrates multiple diagnostic analyses to ensure the robustness
and reliability of the meta-model training pipeline. It runs:

1. **Shuffle Label Sanity Checks**: Binary and multiclass versions to detect label leakage
2. **Ablation Analysis**: Feature subset analysis with both CV strategies  
3. **Main Model Training**: Gene-aware and/or chromosome-aware CV with full diagnostics
4. **Comparative Analysis**: Cross-diagnostic comparison and summary reports

Example Usage
-------------

Basic diagnostic suite:
```bash
python -m meta_spliceai.splice_engine.meta_models.training.run_comprehensive_diagnostics \
  --dataset train_pc_1000/master \
  --out-dir results/comprehensive_diagnostics \
  --run-shuffle-checks \
  --run-ablation \
  --run-main-cv \
  --verbose
```

Minimal testing suite:
```bash
python -m meta_spliceai.splice_engine.meta_models.training.run_comprehensive_diagnostics \
  --dataset train_pc_1000/master \
  --out-dir results/quick_diagnostics \
  --sample-genes 100 \
  --run-shuffle-checks \
  --quick-mode \
  --verbose
```

Full production suite:
```bash
python -m meta_spliceai.splice_engine.meta_models.training.run_comprehensive_diagnostics \
  --dataset train_pc_1000/master \
  --out-dir results/production_diagnostics \
  --run-all \
  --cv-strategies gene,chromosome \
  --ablation-modes full,raw_scores,no_probs,no_kmer,only_kmer \
  --n-folds 5 \
  --n-estimators 500 \
  --run-full-diagnostics \
  --verbose
```
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

# Import submodules for validation
try:
    from meta_spliceai.splice_engine.meta_models.training import (
        shuffle_label_sanity,
        shuffle_label_sanity_multiclass,
        run_ablation_multiclass,
        run_gene_cv_sigmoid,
        run_loco_cv_multiclass_scalable
    )
    MODULES_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Some diagnostic modules not available: {e}")
    MODULES_AVAILABLE = False


def _parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Comprehensive diagnostic runner for splice meta-model training pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Dataset and output
    p.add_argument("--dataset", required=True, help="Dataset directory or Parquet file")
    p.add_argument("--out-dir", required=True, help="Where to save all diagnostic results")
    
    # Diagnostic module selection
    p.add_argument("--run-shuffle-checks", action="store_true", default=False,
                   help="Run shuffle label sanity checks (binary and multiclass)")
    p.add_argument("--run-ablation", action="store_true", default=False,
                   help="Run ablation analysis with feature subset modes")
    p.add_argument("--run-main-cv", action="store_true", default=False,
                   help="Run main CV scripts (gene-aware and/or chromosome-aware)")
    p.add_argument("--run-all", action="store_true", default=False,
                   help="Run all diagnostic modules (overrides individual flags)")
    
    # CV strategy configuration
    p.add_argument("--cv-strategies", default="gene", 
                   help="Comma-separated list of CV strategies: gene, chromosome")
    p.add_argument("--ablation-modes", default="full,raw_scores,no_probs,no_kmer,only_kmer",
                   help="Comma-separated list of ablation modes to test")
    
    # Data and model parameters
    p.add_argument("--gene-col", default="gene_id")
    p.add_argument("--n-folds", type=int, default=5, help="Number of CV folds")
    p.add_argument("--valid-size", type=float, default=0.1)
    p.add_argument("--row-cap", type=int, default=100_000)
    p.add_argument("--sample-genes", type=int, default=None,
                   help="Sample genes for faster testing (applies to all modules)")
    
    # Model parameters
    p.add_argument("--n-estimators", type=int, default=500)
    p.add_argument("--tree-method", default="hist", choices=["hist", "gpu_hist", "approx"])
    p.add_argument("--max-bin", type=int, default=256)
    p.add_argument("--device", default="auto")
    
    # Diagnostic configuration
    p.add_argument("--run-full-diagnostics", action="store_true", default=False,
                   help="Run full post-training diagnostics in each module")
    p.add_argument("--diag-sample", type=int, default=15000,
                   help="Sample size for diagnostic analyses")
    p.add_argument("--quick-mode", action="store_true", default=False,
                   help="Use smaller samples and fewer modes for faster testing")
    
    # Feature handling
    p.add_argument("--exclude-features", default="configs/exclude_features.txt",
                   help="Features to exclude (file path or comma-separated list)")
    p.add_argument("--check-leakage", action="store_true", default=True)
    p.add_argument("--leakage-threshold", type=float, default=0.95)
    p.add_argument("--auto-exclude-leaky", action="store_true", default=False)
    
    # Output and visualization
    p.add_argument("--plot-format", default="pdf", choices=["pdf", "png", "svg"])
    p.add_argument("--generate-report", action="store_true", default=True,
                   help="Generate comprehensive diagnostic report")
    
    # Execution control
    p.add_argument("--parallel", action="store_true", default=False,
                   help="Run independent analyses in parallel (experimental)")
    p.add_argument("--continue-on-error", action="store_true", default=True,
                   help="Continue with other diagnostics if one fails")
    p.add_argument("--dry-run", action="store_true", default=False,
                   help="Show what would be run without executing")
    
    # General
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--verbose", action="store_true")
    
    return p.parse_args(argv)


class DiagnosticRunner:
    """Orchestrates comprehensive diagnostic analyses for the meta-model pipeline."""
    
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.out_dir = Path(args.out_dir)
        self.results = {}
        self.start_time = datetime.now()
        
        # Apply quick mode adjustments
        if args.quick_mode:
            self._apply_quick_mode()
        
        # Apply run-all flag
        if args.run_all:
            args.run_shuffle_checks = True
            args.run_ablation = True
            args.run_main_cv = True
        
        # Setup logging
        self._setup_logging()
        
        # Create output directory structure
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.shuffle_dir = self.out_dir / "shuffle_sanity_checks"
        self.ablation_dir = self.out_dir / "ablation_analysis"
        self.main_cv_dir = self.out_dir / "main_cv_training"
        self.reports_dir = self.out_dir / "reports"
        
        for subdir in [self.shuffle_dir, self.ablation_dir, self.main_cv_dir, self.reports_dir]:
            subdir.mkdir(exist_ok=True)
    
    def _apply_quick_mode(self):
        """Apply quick mode settings for faster testing."""
        self.args.sample_genes = self.args.sample_genes or 50
        self.args.n_folds = min(self.args.n_folds, 3)
        self.args.n_estimators = min(self.args.n_estimators, 200)
        self.args.diag_sample = min(self.args.diag_sample, 5000)
        self.args.row_cap = min(self.args.row_cap, 50000)
        
        # Reduce ablation modes for quick testing
        if self.args.ablation_modes == "full,raw_scores,no_probs,no_kmer,only_kmer":
            self.args.ablation_modes = "full,raw_scores,only_kmer"
        
        logging.info("[Quick Mode] Applied reduced settings for faster testing")
    
    def _setup_logging(self):
        """Setup logging configuration."""
        log_level = logging.INFO if self.args.verbose else logging.WARNING
        log_file = self.out_dir / "diagnostic_runner.log"
        
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
    
    def _run_command(self, cmd: List[str], name: str, capture_output: bool = False) -> Tuple[bool, str]:
        """Run a subprocess command with error handling."""
        if self.args.dry_run:
            logging.info(f"[DRY RUN] Would run {name}: {' '.join(cmd)}")
            return True, "Dry run - not executed"
        
        try:
            logging.info(f"[{name}] Starting: {' '.join(cmd)}")
            start_time = time.time()
            
            if capture_output:
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                output = result.stdout
            else:
                result = subprocess.run(cmd, check=True)
                output = "Command completed successfully"
            
            duration = time.time() - start_time
            logging.info(f"[{name}] Completed in {duration:.1f}s")
            return True, output
            
        except subprocess.CalledProcessError as e:
            error_msg = f"Command failed with return code {e.returncode}"
            if capture_output and e.stderr:
                error_msg += f": {e.stderr}"
            
            logging.error(f"[{name}] {error_msg}")
            
            if not self.args.continue_on_error:
                raise
            
            return False, error_msg
        except Exception as e:
            error_msg = f"Unexpected error: {e}"
            logging.error(f"[{name}] {error_msg}")
            
            if not self.args.continue_on_error:
                raise
            
            return False, error_msg
    
    def run_shuffle_sanity_checks(self) -> Dict[str, bool]:
        """Run shuffle label sanity checks for label leakage detection."""
        if not self.args.run_shuffle_checks:
            return {}
        
        logging.info("=" * 60)
        logging.info("RUNNING SHUFFLE LABEL SANITY CHECKS")
        logging.info("=" * 60)
        
        results = {}
        
        # Common arguments for shuffle checks
        common_args = [
            "--dataset", self.args.dataset,
            "--n-folds", str(self.args.n_folds),
            "--n-estimators", str(self.args.n_estimators),
            "--tree-method", self.args.tree_method,
            "--device", self.args.device,
            "--exclude-features", self.args.exclude_features,
            "--plot-format", self.args.plot_format,
            "--seed", str(self.args.seed),
        ]
        
        if self.args.sample_genes:
            common_args.extend(["--sample-genes", str(self.args.sample_genes)])
        
        if self.args.verbose:
            common_args.append("--verbose")
        
        if self.args.check_leakage:
            common_args.extend([
                "--check-leakage",
                "--leakage-threshold", str(self.args.leakage_threshold)
            ])
            if self.args.auto_exclude_leaky:
                common_args.append("--auto-exclude-leaky")
        
        # Binary shuffle sanity check
        binary_dir = self.shuffle_dir / "binary"
        binary_cmd = [
            sys.executable, "-m",
            "meta_spliceai.splice_engine.meta_models.training.shuffle_label_sanity",
            "--out-dir", str(binary_dir),
            *common_args
        ]
        
        success, output = self._run_command(binary_cmd, "Binary Shuffle Sanity")
        results["binary_shuffle"] = success
        
        # Multiclass shuffle sanity check
        multiclass_dir = self.shuffle_dir / "multiclass"
        multiclass_cmd = [
            sys.executable, "-m",
            "meta_spliceai.splice_engine.meta_models.training.shuffle_label_sanity_multiclass",
            "--out-dir", str(multiclass_dir),
            *common_args
        ]
        
        success, output = self._run_command(multiclass_cmd, "Multiclass Shuffle Sanity")
        results["multiclass_shuffle"] = success
        
        # Generate shuffle sanity summary
        self._generate_shuffle_summary(results)
        
        return results
    
    def run_ablation_analysis(self) -> Dict[str, bool]:
        """Run ablation analysis with different feature subset modes."""
        if not self.args.run_ablation:
            return {}
        
        logging.info("=" * 60)
        logging.info("RUNNING ABLATION ANALYSIS")
        logging.info("=" * 60)
        
        results = {}
        cv_strategies = [s.strip() for s in self.args.cv_strategies.split(",")]
        
        # Common arguments for ablation
        common_args = [
            "--dataset", self.args.dataset,
            "--modes", self.args.ablation_modes,
            "--n-estimators", str(self.args.n_estimators),
            "--tree-method", self.args.tree_method,
            "--device", self.args.device,
            "--exclude-features", self.args.exclude_features,
            "--plot-format", self.args.plot_format,
            "--diag-sample", str(self.args.diag_sample),
            "--seed", str(self.args.seed),
        ]
        
        if self.args.sample_genes:
            common_args.extend(["--sample-genes", str(self.args.sample_genes)])
        
        if self.args.verbose:
            common_args.append("--verbose")
        
        if self.args.check_leakage:
            common_args.extend([
                "--check-leakage",
                "--leakage-threshold", str(self.args.leakage_threshold)
            ])
            if self.args.auto_exclude_leaky:
                common_args.append("--auto-exclude-leaky")
        
        if self.args.run_full_diagnostics:
            common_args.append("--run-full-diagnostics")
        
        # Run ablation for each CV strategy
        for strategy in cv_strategies:
            strategy_dir = self.ablation_dir / f"{strategy}_aware"
            ablation_cmd = [
                sys.executable, "-m",
                "meta_spliceai.splice_engine.meta_models.training.run_ablation_multiclass",
                "--out-dir", str(strategy_dir),
                "--cv-strategy", strategy,
                "--n-folds", str(self.args.n_folds),
                *common_args
            ]
            
            success, output = self._run_command(ablation_cmd, f"Ablation ({strategy}-aware)")
            results[f"ablation_{strategy}"] = success
        
        # Generate ablation summary
        self._generate_ablation_summary(results, cv_strategies)
        
        return results
    
    def run_main_cv_training(self) -> Dict[str, bool]:
        """Run main CV training scripts with full diagnostics."""
        if not self.args.run_main_cv:
            return {}
        
        logging.info("=" * 60)
        logging.info("RUNNING MAIN CV TRAINING")
        logging.info("=" * 60)
        
        results = {}
        cv_strategies = [s.strip() for s in self.args.cv_strategies.split(",")]
        
        for strategy in cv_strategies:
            if strategy == "gene":
                success = self._run_gene_aware_cv()
                results["gene_cv"] = success
            elif strategy == "chromosome":
                success = self._run_chromosome_aware_cv()
                results["chromosome_cv"] = success
            else:
                logging.warning(f"Unknown CV strategy: {strategy}")
                results[f"{strategy}_cv"] = False
        
        return results
    
    def _run_gene_aware_cv(self) -> bool:
        """Run gene-aware CV with full diagnostics."""
        gene_dir = self.main_cv_dir / "gene_aware"
        
        cmd = [
            sys.executable, "-m",
            "meta_spliceai.splice_engine.meta_models.training.run_gene_cv_sigmoid",
            "--dataset", self.args.dataset,
            "--out-dir", str(gene_dir),
            "--n-folds", str(self.args.n_folds),
            "--n-estimators", str(self.args.n_estimators),
            "--tree-method", self.args.tree_method,
            "--device", self.args.device,
            "--diag-sample", str(self.args.diag_sample),
            "--top-k", "5",
            "--plot-curves",
            "--plot-format", self.args.plot_format,
            "--check-leakage",
            "--leakage-threshold", str(self.args.leakage_threshold),
            "--calibrate",
            "--calib-method", "platt",
            "--exclude-features", self.args.exclude_features,
            "--seed", str(self.args.seed),
        ]
        
        if self.args.sample_genes:
            cmd.extend(["--sample-genes", str(self.args.sample_genes)])
        
        if self.args.verbose:
            cmd.append("--verbose")
        
        if self.args.auto_exclude_leaky:
            cmd.append("--auto-exclude-leaky")
        
        success, output = self._run_command(cmd, "Gene-Aware CV")
        return success
    
    def _run_chromosome_aware_cv(self) -> bool:
        """Run chromosome-aware CV with full diagnostics."""
        chrom_dir = self.main_cv_dir / "chromosome_aware"
        
        cmd = [
            sys.executable, "-m",
            "meta_spliceai.splice_engine.meta_models.training.run_loco_cv_multiclass_scalable",
            "--dataset", self.args.dataset,
            "--out-dir", str(chrom_dir),
            "--n-estimators", str(self.args.n_estimators),
            "--tree-method", self.args.tree_method,
            "--device", self.args.device,
            "--diag-sample", str(self.args.diag_sample),
            "--top-k", "5",
            "--plot-curves",
            "--plot-format", self.args.plot_format,
            "--check-leakage",
            "--leakage-threshold", str(self.args.leakage_threshold),
            "--calibrate",
            "--calib-method", "platt",
            "--exclude-features", self.args.exclude_features,
            "--seed", str(self.args.seed),
        ]
        
        if self.args.sample_genes:
            cmd.extend(["--sample-genes", str(self.args.sample_genes)])
        
        if self.args.verbose:
            cmd.append("--verbose")
        
        if self.args.auto_exclude_leaky:
            cmd.append("--auto-exclude-leaky")
        
        success, output = self._run_command(cmd, "Chromosome-Aware CV")
        return success
    
    def _generate_shuffle_summary(self, results: Dict[str, bool]):
        """Generate summary report for shuffle sanity checks."""
        summary = {
            "timestamp": datetime.now().isoformat(),
            "results": results,
            "interpretation": {}
        }
        
        # Load and analyze results if available
        for check_type in ["binary", "multiclass"]:
            check_dir = self.shuffle_dir / check_type
            summary_file = check_dir / f"shuffle_sanity_{check_type}_summary.json" if check_type == "multiclass" else check_dir / "shuffle_sanity_summary.json"
            
            if summary_file.exists():
                try:
                    with open(summary_file, 'r') as f:
                        check_data = json.load(f)
                        summary["interpretation"][check_type] = check_data.get("sanity_check", {})
                except Exception as e:
                    logging.warning(f"Could not load {check_type} shuffle results: {e}")
        
        # Save summary
        with open(self.shuffle_dir / "shuffle_checks_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
    
    def _generate_ablation_summary(self, results: Dict[str, bool], strategies: List[str]):
        """Generate summary report for ablation analysis."""
        summary = {
            "timestamp": datetime.now().isoformat(),
            "results": results,
            "strategies": strategies,
            "comparison": {}
        }
        
        # Load and compare results across strategies
        for strategy in strategies:
            strategy_dir = self.ablation_dir / f"{strategy}_aware"
            summary_file = strategy_dir / "ablation_summary.csv"
            
            if summary_file.exists():
                try:
                    df = pd.read_csv(summary_file)
                    summary["comparison"][strategy] = {
                        "best_mode_accuracy": df.loc[df['accuracy'].idxmax(), 'mode'] if not df['accuracy'].isna().all() else None,
                        "best_mode_f1": df.loc[df['macro_f1'].idxmax(), 'mode'] if not df['macro_f1'].isna().all() else None,
                        "mean_accuracy": float(df['accuracy'].mean()),
                        "mean_f1": float(df['macro_f1'].mean()),
                        "modes_tested": len(df)
                    }
                except Exception as e:
                    logging.warning(f"Could not load {strategy} ablation results: {e}")
        
        # Save summary
        with open(self.ablation_dir / "ablation_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
    
    def generate_comprehensive_report(self):
        """Generate comprehensive diagnostic report."""
        if not self.args.generate_report:
            return
        
        logging.info("=" * 60)
        logging.info("GENERATING COMPREHENSIVE DIAGNOSTIC REPORT")
        logging.info("=" * 60)
        
        # Collect all results
        all_results = {}
        
        # Load shuffle check results
        shuffle_summary_file = self.shuffle_dir / "shuffle_checks_summary.json"
        if shuffle_summary_file.exists():
            with open(shuffle_summary_file, 'r') as f:
                all_results["shuffle_checks"] = json.load(f)
        
        # Load ablation results
        ablation_summary_file = self.ablation_dir / "ablation_summary.json"
        if ablation_summary_file.exists():
            with open(ablation_summary_file, 'r') as f:
                all_results["ablation_analysis"] = json.load(f)
        
        # Load main CV results
        main_cv_results = {}
        for strategy in ["gene_aware", "chromosome_aware"]:
            cv_dir = self.main_cv_dir / strategy
            
            # Look for various result files
            possible_files = [
                "gene_cv_metrics.csv",
                "loco_metrics.csv", 
                "metrics_aggregate.json",
                "cv_metrics_summary.txt"
            ]
            
            for filename in possible_files:
                filepath = cv_dir / filename
                if filepath.exists():
                    if filename.endswith('.json'):
                        with open(filepath, 'r') as f:
                            main_cv_results[f"{strategy}_{filename}"] = json.load(f)
                    elif filename.endswith('.csv'):
                        df = pd.read_csv(filepath)
                        main_cv_results[f"{strategy}_{filename}"] = {
                            "shape": df.shape,
                            "columns": df.columns.tolist(),
                            "summary": df.describe().to_dict() if df.select_dtypes(include=[np.number]).shape[1] > 0 else {}
                        }
        
        if main_cv_results:
            all_results["main_cv_training"] = main_cv_results
        
        # Create comprehensive report
        report = {
            "diagnostic_run_info": {
                "timestamp": datetime.now().isoformat(),
                "start_time": self.start_time.isoformat(),
                "duration_minutes": (datetime.now() - self.start_time).total_seconds() / 60,
                "dataset": self.args.dataset,
                "configuration": {
                    "cv_strategies": self.args.cv_strategies,
                    "ablation_modes": self.args.ablation_modes,
                    "n_folds": self.args.n_folds,
                    "n_estimators": self.args.n_estimators,
                    "sample_genes": self.args.sample_genes,
                    "quick_mode": self.args.quick_mode,
                    "run_full_diagnostics": self.args.run_full_diagnostics
                },
                "modules_run": {
                    "shuffle_checks": self.args.run_shuffle_checks,
                    "ablation_analysis": self.args.run_ablation,
                    "main_cv_training": self.args.run_main_cv
                }
            },
            "results": all_results,
            "overall_status": self._assess_overall_status(all_results)
        }
        
        # Save comprehensive report
        report_file = self.reports_dir / "comprehensive_diagnostic_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Create human-readable summary
        self._create_readable_summary(report)
        
        logging.info(f"Comprehensive diagnostic report saved to: {report_file}")
    
    def _assess_overall_status(self, results: Dict) -> Dict[str, str]:
        """Assess overall status of diagnostic run."""
        status = {
            "label_leakage_risk": "unknown",
            "feature_importance": "unknown",
            "model_performance": "unknown",
            "pipeline_health": "unknown"
        }
        
        # Assess label leakage risk
        if "shuffle_checks" in results:
            shuffle_results = results["shuffle_checks"].get("interpretation", {})
            healthy_checks = []
            
            for check_type in ["binary", "multiclass"]:
                if check_type in shuffle_results:
                    healthy_checks.append(shuffle_results[check_type].get("appears_healthy", False))
            
            if healthy_checks:
                status["label_leakage_risk"] = "low" if all(healthy_checks) else "high"
        
        # Assess feature importance from ablation
        if "ablation_analysis" in results:
            ablation_results = results["ablation_analysis"].get("comparison", {})
            if ablation_results:
                # Check if different feature sets show meaningful differences
                accuracies = [v.get("mean_accuracy", 0) for v in ablation_results.values()]
                if len(accuracies) > 1 and max(accuracies) - min(accuracies) > 0.05:
                    status["feature_importance"] = "meaningful_differences"
                else:
                    status["feature_importance"] = "minimal_differences"
        
        # Assess overall pipeline health
        failed_modules = []
        if "shuffle_checks" in results and not all(results["shuffle_checks"]["results"].values()):
            failed_modules.append("shuffle_checks")
        if "ablation_analysis" in results and not all(results["ablation_analysis"]["results"].values()):
            failed_modules.append("ablation_analysis")
        
        if not failed_modules:
            status["pipeline_health"] = "healthy"
        elif len(failed_modules) == 1:
            status["pipeline_health"] = "mostly_healthy"
        else:
            status["pipeline_health"] = "issues_detected"
        
        return status
    
    def _create_readable_summary(self, report: Dict):
        """Create human-readable summary report."""
        summary_lines = []
        
        summary_lines.append("SPLICE META-MODEL COMPREHENSIVE DIAGNOSTIC REPORT")
        summary_lines.append("=" * 60)
        summary_lines.append("")
        
        # Run info
        run_info = report["diagnostic_run_info"]
        summary_lines.append(f"Dataset: {run_info['dataset']}")
        summary_lines.append(f"Duration: {run_info['duration_minutes']:.1f} minutes")
        summary_lines.append(f"Configuration: {run_info['configuration']}")
        summary_lines.append("")
        
        # Overall status
        status = report["overall_status"]
        summary_lines.append("OVERALL ASSESSMENT:")
        summary_lines.append(f"  Label Leakage Risk: {status['label_leakage_risk'].upper()}")
        summary_lines.append(f"  Feature Importance: {status['feature_importance']}")
        summary_lines.append(f"  Pipeline Health: {status['pipeline_health']}")
        summary_lines.append("")
        
        # Module-specific results
        if "shuffle_checks" in report["results"]:
            summary_lines.append("SHUFFLE LABEL SANITY CHECKS:")
            shuffle_results = report["results"]["shuffle_checks"]["results"]
            for check_type, success in shuffle_results.items():
                status_str = "✓ PASSED" if success else "✗ FAILED"
                summary_lines.append(f"  {check_type}: {status_str}")
            summary_lines.append("")
        
        if "ablation_analysis" in report["results"]:
            summary_lines.append("ABLATION ANALYSIS:")
            ablation_comparison = report["results"]["ablation_analysis"].get("comparison", {})
            for strategy, results in ablation_comparison.items():
                summary_lines.append(f"  {strategy.upper()}-aware CV:")
                summary_lines.append(f"    Best accuracy mode: {results.get('best_mode_accuracy', 'N/A')}")
                summary_lines.append(f"    Best F1 mode: {results.get('best_mode_f1', 'N/A')}")
                summary_lines.append(f"    Mean accuracy: {results.get('mean_accuracy', 0):.3f}")
            summary_lines.append("")
        
        if "main_cv_training" in report["results"]:
            summary_lines.append("MAIN CV TRAINING:")
            summary_lines.append("  See detailed results in main_cv_training/ subdirectories")
            summary_lines.append("")
        
        # Recommendations
        summary_lines.append("RECOMMENDATIONS:")
        if status["label_leakage_risk"] == "high":
            summary_lines.append("  ⚠️  HIGH LABEL LEAKAGE RISK DETECTED - Review feature engineering pipeline")
        else:
            summary_lines.append("  ✓ No significant label leakage detected")
        
        if status["feature_importance"] == "meaningful_differences":
            summary_lines.append("  ✓ Feature sets show meaningful performance differences")
        else:
            summary_lines.append("  ⚠️  Minimal performance differences between feature sets")
        
        if status["pipeline_health"] != "healthy":
            summary_lines.append("  ⚠️  Some diagnostic modules encountered issues - Check logs")
        else:
            summary_lines.append("  ✓ All diagnostic modules completed successfully")
        
        # Save readable summary
        summary_file = self.reports_dir / "diagnostic_summary.txt"
        with open(summary_file, 'w') as f:
            f.write('\n'.join(summary_lines))
        
        # Also print to console
        print('\n'.join(summary_lines))
    
    def run(self) -> Dict[str, bool]:
        """Run the comprehensive diagnostic suite."""
        logging.info("Starting comprehensive diagnostic run")
        logging.info(f"Output directory: {self.out_dir}")
        logging.info(f"Configuration: {vars(self.args)}")
        
        all_results = {}
        
        try:
            # Run shuffle sanity checks
            shuffle_results = self.run_shuffle_sanity_checks()
            all_results.update(shuffle_results)
            
            # Run ablation analysis
            ablation_results = self.run_ablation_analysis()
            all_results.update(ablation_results)
            
            # Run main CV training
            main_cv_results = self.run_main_cv_training()
            all_results.update(main_cv_results)
            
            # Generate comprehensive report
            self.generate_comprehensive_report()
            
        except Exception as e:
            logging.error(f"Fatal error in diagnostic runner: {e}")
            if not self.args.continue_on_error:
                raise
        
        # Final summary
        total_modules = len(all_results)
        successful_modules = sum(all_results.values())
        
        logging.info("=" * 60)
        logging.info("DIAGNOSTIC RUN COMPLETED")
        logging.info("=" * 60)
        logging.info(f"Modules run: {total_modules}")
        logging.info(f"Successful: {successful_modules}")
        logging.info(f"Failed: {total_modules - successful_modules}")
        logging.info(f"Duration: {(datetime.now() - self.start_time).total_seconds() / 60:.1f} minutes")
        logging.info(f"Results saved to: {self.out_dir}")
        
        return all_results


def main(argv: List[str] | None = None) -> None:
    """Main entry point for comprehensive diagnostic runner."""
    if not MODULES_AVAILABLE:
        print("Error: Required diagnostic modules are not available.")
        sys.exit(1)
    
    args = _parse_args(argv)
    
    # Validate arguments
    if not any([args.run_shuffle_checks, args.run_ablation, args.run_main_cv, args.run_all]):
        print("Error: Must specify at least one diagnostic module to run.")
        print("Use --run-shuffle-checks, --run-ablation, --run-main-cv, or --run-all")
        sys.exit(1)
    
    # Create and run diagnostic suite
    runner = DiagnosticRunner(args)
    results = runner.run()
    
    # Exit with appropriate code
    if all(results.values()):
        sys.exit(0)  # All diagnostics passed
    else:
        sys.exit(1)  # Some diagnostics failed


if __name__ == "__main__":
    main() 