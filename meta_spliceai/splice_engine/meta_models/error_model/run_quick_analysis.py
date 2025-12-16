#!/usr/bin/env python3
"""
Quick analysis script with sensible defaults.

This script provides a simplified interface for common error model analysis tasks.
It uses the main workflow script with optimized parameters for different scenarios.

Usage:
    # Quick FP analysis
    python run_quick_analysis.py --data_dir data/ensembl/spliceai_eval/meta_models

    # Full analysis with both error types
    python run_quick_analysis.py --data_dir data/ensembl/spliceai_eval/meta_models --full

    # Fast test run
    python run_quick_analysis.py --data_dir data/ensembl/spliceai_eval/meta_models --test
"""

import argparse
import subprocess
import sys
from pathlib import Path
from datetime import datetime


def run_workflow(args_list, description):
    """Run the main workflow with given arguments."""
    print(f"\n🚀 {description}")
    print("=" * 60)
    
    cmd = [sys.executable, "run_error_model_workflow.py"] + args_list
    print(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"✅ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed with exit code {e.returncode}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Quick error model analysis")
    parser.add_argument("--data_dir", type=Path, required=True, 
                       help="Meta-model artifacts directory")
    parser.add_argument("--output_base", type=Path, 
                       default=Path("output/error_analysis"),
                       help="Base output directory")
    
    # Analysis modes
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--full", action="store_true",
                           help="Run full analysis (both FP and FN)")
    mode_group.add_argument("--test", action="store_true", 
                           help="Quick test run with minimal parameters")
    mode_group.add_argument("--ig_only", action="store_true",
                           help="IG analysis only (skip training)")
    
    # Optional parameters
    parser.add_argument("--error_type", choices=["FP_vs_TP", "FN_vs_TP"], 
                       default="FP_vs_TP", help="Error type for single analysis")
    parser.add_argument("--splice_type", choices=["donor", "acceptor", "any"], 
                       default="any", help="Splice site type")
    
    args = parser.parse_args()
    
    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    success_count = 0
    total_runs = 0
    
    if args.test:
        # Quick test run
        output_dir = args.output_base / f"test_run_{timestamp}"
        workflow_args = [
            "--data_dir", str(args.data_dir),
            "--output_dir", str(output_dir),
            "--error_type", args.error_type,
            "--splice_type", args.splice_type,
            "--num_epochs", "1",
            "--max_ig_samples", "50",
            "--batch_size", "8"
        ]
        
        total_runs = 1
        if run_workflow(workflow_args, "Quick Test Analysis"):
            success_count += 1
            
    elif args.full:
        # Full analysis with both error types
        for error_type in ["FP_vs_TP", "FN_vs_TP"]:
            output_dir = args.output_base / f"{error_type.lower()}_{timestamp}"
            workflow_args = [
                "--data_dir", str(args.data_dir),
                "--output_dir", str(output_dir),
                "--error_type", error_type,
                "--splice_type", args.splice_type,
                "--num_epochs", "10",
                "--max_ig_samples", "500",
                "--batch_size", "16"
            ]
            
            total_runs += 1
            if run_workflow(workflow_args, f"Full {error_type} Analysis"):
                success_count += 1
                
    elif args.ig_only:
        # IG analysis only
        output_dir = args.output_base / f"ig_only_{args.error_type.lower()}_{timestamp}"
        workflow_args = [
            "--data_dir", str(args.data_dir),
            "--output_dir", str(output_dir),
            "--error_type", args.error_type,
            "--splice_type", args.splice_type,
            "--skip_training",
            "--max_ig_samples", "1000"
        ]
        
        total_runs = 1
        if run_workflow(workflow_args, "IG Analysis Only"):
            success_count += 1
            
    else:
        # Standard single analysis
        output_dir = args.output_base / f"{args.error_type.lower()}_{timestamp}"
        workflow_args = [
            "--data_dir", str(args.data_dir),
            "--output_dir", str(output_dir),
            "--error_type", args.error_type,
            "--splice_type", args.splice_type,
            "--num_epochs", "10",
            "--max_ig_samples", "500",
            "--batch_size", "16"
        ]
        
        total_runs = 1
        if run_workflow(workflow_args, f"Standard {args.error_type} Analysis"):
            success_count += 1
    
    # Summary
    print("\n" + "=" * 60)
    print("ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"Completed: {success_count}/{total_runs} analyses")
    
    if success_count == total_runs:
        print("🎉 All analyses completed successfully!")
    elif success_count > 0:
        print("⚠️  Some analyses completed with issues")
    else:
        print("❌ All analyses failed")
    
    print(f"Results saved under: {args.output_base}")
    print("=" * 60)


if __name__ == "__main__":
    main()
