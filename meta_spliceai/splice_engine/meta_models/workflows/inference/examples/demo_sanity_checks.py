#!/usr/bin/env python3
"""Demo: Meta-Model Inference Sanity Checks
==========================================

This script demonstrates basic sanity checks for meta-model inference workflows,
ensuring input-output consistency, complete coverage, and prediction reliability.

EXAMPLE USAGE:
    # RECOMMENDED: Run as module from project root (paths relative to project root):
    python -m meta_spliceai.splice_engine.meta_models.workflows.inference.demo_sanity_checks \
        --model results/gene_cv_pc_1000_3mers_run_4/model_multiclass.pkl \
        --training-dataset train_pc_1000_3mers \
        --genes ENSG00000104435 \
        --positions 10000

    # ALTERNATIVE: Run script directly from project root:
    cd /home/bchiu/work/splice-surveyor  # Change to your project root
    python meta_spliceai/splice_engine/meta_models/workflows/inference/demo_sanity_checks.py \
        --model results/gene_cv_pc_1000_3mers_run_4/model_multiclass.pkl \
        --training-dataset train_pc_1000_3mers \
        --genes ENSG00000104435 \
        --positions 10000

    # COMPREHENSIVE CHECKS with multiple genes:
    python -m meta_spliceai.splice_engine.meta_models.workflows.inference.demo_sanity_checks \
        --model results/gene_cv_pc_1000_3mers_run_4/model_multiclass.pkl \
        --training-dataset train_pc_1000_3mers \
        --genes-file test_genes.txt \
        --positions 25000 \
        --check-consistency \
        --verbose

FEATURES:
- Input-output length consistency verification
- Complete positional coverage validation
- Prediction tensor shape and range verification
- Known position consistency checks
- Splice site distribution analysis
- Meta-model vs base model comparison

REQUIREMENTS:
- Pre-trained meta-model (.pkl file)
- Training dataset directory (for coverage analysis)
- Target genes for testing
- Sufficient memory for position processing
"""

import argparse
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import tempfile

import numpy as np
import polars as pl
import pandas as pd

# Add project root to Python path
import os
project_root = os.path.join(os.path.expanduser("~"), "work/splice-surveyor")
sys.path.insert(0, project_root)

from meta_spliceai.splice_engine.meta_models.workflows.selective_meta_inference import (
    run_selective_meta_inference,
    SelectiveInferenceConfig
)


class SanityCheckRunner:
    """Run comprehensive sanity checks for inference workflows."""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.checks_passed = 0
        self.checks_failed = 0
        self.check_details = []
    
    def log_check(self, name: str, passed: bool, details: str = "", expected: Any = None, actual: Any = None):
        """Log a sanity check result."""
        status = "✅ PASS" if passed else "❌ FAIL"
        
        check_record = {
            'name': name,
            'passed': passed,
            'details': details,
            'expected': expected,
            'actual': actual,
            'timestamp': time.strftime("%H:%M:%S")
        }
        
        self.check_details.append(check_record)
        
        if passed:
            self.checks_passed += 1
        else:
            self.checks_failed += 1
        
        if self.verbose:
            print(f"      {status} {name}")
            if details:
                print(f"           {details}")
            if not passed and expected is not None and actual is not None:
                print(f"           Expected: {expected}")
                print(f"           Actual: {actual}")
    
    def check_input_output_length_consistency(
        self, 
        gene_id: str, 
        expected_positions: int,
        predictions: pl.DataFrame
    ) -> bool:
        """Test 1: Input-Output Length Test - ensure complete positional coverage."""
        
        if self.verbose:
            print(f"   🧪 Check 1: Input-Output Length Consistency")
        
        # Filter predictions to this gene
        gene_predictions = predictions.filter(pl.col('gene_id') == gene_id)
        actual_positions = len(gene_predictions)
        
        # Check basic length
        length_ok = actual_positions == expected_positions
        self.log_check(
            "Input-Output Length Match",
            length_ok,
            f"Gene {gene_id}: Expected {expected_positions}, got {actual_positions}",
            expected_positions,
            actual_positions
        )
        
        # Check for duplicate positions
        unique_positions = gene_predictions['position'].n_unique()
        duplicates_ok = unique_positions == actual_positions
        self.log_check(
            "No Duplicate Positions",
            duplicates_ok,
            f"Gene {gene_id}: {actual_positions} total, {unique_positions} unique",
            actual_positions,
            unique_positions
        )
        
        # Check prediction columns exist and have correct shape
        required_cols = ['donor_meta', 'acceptor_meta', 'neither_meta']
        cols_ok = all(col in gene_predictions.columns for col in required_cols)
        self.log_check(
            "Required Prediction Columns Present",
            cols_ok,
            f"Missing columns: {[col for col in required_cols if col not in gene_predictions.columns]}",
            required_cols,
            list(gene_predictions.columns)
        )
        
        if cols_ok:
            # Check prediction ranges [0,1]
            for col in required_cols:
                values = gene_predictions[col].to_numpy()
                min_val, max_val = np.min(values), np.max(values)
                range_ok = (min_val >= 0.0) and (max_val <= 1.0)
                self.log_check(
                    f"{col} Values in [0,1] Range",
                    range_ok,
                    f"Range: [{min_val:.3f}, {max_val:.3f}]",
                    "[0.0, 1.0]",
                    f"[{min_val:.3f}, {max_val:.3f}]"
                )
        
        return length_ok and duplicates_ok and cols_ok
    
    def check_prediction_consistency(
        self,
        gene_id: str,
        predictions: pl.DataFrame,
        training_dataset_path: Optional[Path] = None
    ) -> bool:
        """Test 2: Consistency Test for Known Positions."""
        
        if self.verbose:
            print(f"   🧪 Check 2: Prediction Consistency")
        
        gene_predictions = predictions.filter(pl.col('gene_id') == gene_id)
        
        if len(gene_predictions) == 0:
            self.log_check(
                "Gene Predictions Available",
                False,
                f"No predictions found for gene {gene_id}"
            )
            return False
        
        # Check probability sum consistency (should sum to ~1 for each position)
        donor_vals = gene_predictions['donor_meta'].to_numpy()
        acceptor_vals = gene_predictions['acceptor_meta'].to_numpy()
        neither_vals = gene_predictions['neither_meta'].to_numpy()
        
        prob_sums = donor_vals + acceptor_vals + neither_vals
        sum_consistency = np.allclose(prob_sums, 1.0, atol=0.01)
        
        self.log_check(
            "Probability Sums to 1.0",
            sum_consistency,
            f"Sum range: [{np.min(prob_sums):.3f}, {np.max(prob_sums):.3f}]",
            "~1.0",
            f"[{np.min(prob_sums):.3f}, {np.max(prob_sums):.3f}]"
        )
        
        # Check for reasonable splice site distribution
        # Expect majority "neither", small fraction donors/acceptors
        max_indices = np.argmax(np.column_stack([donor_vals, acceptor_vals, neither_vals]), axis=1)
        predicted_types = ['donor', 'acceptor', 'neither']
        type_counts = {ptype: np.sum(max_indices == i) for i, ptype in enumerate(predicted_types)}
        
        total_positions = len(gene_predictions)
        neither_fraction = type_counts['neither'] / total_positions
        splice_fraction = (type_counts['donor'] + type_counts['acceptor']) / total_positions
        
        # Expect majority neither (>90%), minority splice sites (<10%)
        distribution_ok = neither_fraction > 0.90 and splice_fraction < 0.10
        
        self.log_check(
            "Realistic Splice Site Distribution",
            distribution_ok,
            f"Neither: {neither_fraction:.1%}, Splice: {splice_fraction:.1%}",
            "Neither >90%, Splice <10%",
            f"Neither: {neither_fraction:.1%}, Splice: {splice_fraction:.1%}"
        )
        
        # Check for base model consistency if available
        base_cols = ['donor_score', 'acceptor_score', 'neither_score']
        if all(col in gene_predictions.columns for col in base_cols):
            
            # Compare base vs meta predictions for high-confidence base predictions
            base_donor = gene_predictions['donor_score'].to_numpy()
            base_acceptor = gene_predictions['acceptor_score'].to_numpy()
            base_neither = gene_predictions['neither_score'].to_numpy()
            
            # Find high-confidence base predictions (max score > 0.9)
            base_max_scores = np.maximum.reduce([base_donor, base_acceptor, base_neither])
            high_conf_mask = base_max_scores > 0.9
            
            if np.sum(high_conf_mask) > 0:
                # For high-confidence positions, meta and base should largely agree
                base_argmax = np.argmax(np.column_stack([base_donor, base_acceptor, base_neither]), axis=1)
                meta_argmax = np.argmax(np.column_stack([donor_vals, acceptor_vals, neither_vals]), axis=1)
                
                high_conf_agreement = np.mean(base_argmax[high_conf_mask] == meta_argmax[high_conf_mask])
                agreement_ok = high_conf_agreement > 0.8  # 80% agreement for high-confidence predictions
                
                self.log_check(
                    "Base-Meta Agreement (High Confidence)",
                    agreement_ok,
                    f"Agreement: {high_conf_agreement:.1%} on {np.sum(high_conf_mask)} positions",
                    ">80%",
                    f"{high_conf_agreement:.1%}"
                )
        
        return sum_consistency and distribution_ok
    
    def check_coverage_completeness(
        self,
        gene_id: str,
        predictions: pl.DataFrame,
        expected_positions: int
    ) -> bool:
        """Test 3: Complete Coverage Verification."""
        
        if self.verbose:
            print(f"   🧪 Check 3: Coverage Completeness")
        
        gene_predictions = predictions.filter(pl.col('gene_id') == gene_id)
        
        if len(gene_predictions) == 0:
            self.log_check(
                "Gene Coverage Available",
                False,
                f"No coverage found for gene {gene_id}"
            )
            return False
        
        # Check positional continuity (no gaps)
        positions = sorted(gene_predictions['position'].to_list())
        
        if len(positions) >= 2:
            # Check for reasonable position spacing
            position_diffs = np.diff(positions)
            median_diff = np.median(position_diffs)
            max_gap = np.max(position_diffs)
            
            # Positions should be mostly consecutive (diff=1) or have small gaps
            continuity_ok = median_diff <= 1 and max_gap <= 10
            
            self.log_check(
                "Positional Continuity",
                continuity_ok,
                f"Median gap: {median_diff}, Max gap: {max_gap}",
                "Median ≤1, Max ≤10",
                f"Median: {median_diff}, Max: {max_gap}"
            )
        else:
            continuity_ok = True
            self.log_check(
                "Positional Continuity",
                True,
                "Too few positions to check continuity"
            )
        
        # Check position range coverage
        if len(positions) > 0:
            pos_range = positions[-1] - positions[0] + 1
            coverage_ratio = len(positions) / pos_range
            
            coverage_ok = coverage_ratio > 0.8  # At least 80% of range covered
            
            self.log_check(
                "Position Range Coverage",
                coverage_ok,
                f"Coverage: {len(positions)}/{pos_range} positions ({coverage_ratio:.1%})",
                ">80%",
                f"{coverage_ratio:.1%}"
            )
        else:
            coverage_ok = False
            self.log_check(
                "Position Range Coverage",
                False,
                "No positions found"
            )
        
        # Check for prediction completeness (no missing values)
        required_cols = ['donor_meta', 'acceptor_meta', 'neither_meta']
        missing_values = {col: gene_predictions[col].null_count() for col in required_cols}
        completeness_ok = all(count == 0 for count in missing_values.values())
        
        self.log_check(
            "Prediction Completeness (No NaNs)",
            completeness_ok,
            f"Missing values: {missing_values}",
            "All zeros",
            str(missing_values)
        )
        
        return continuity_ok and coverage_ok and completeness_ok
    
    def check_meta_model_vs_base_differences(
        self,
        gene_id: str,
        predictions: pl.DataFrame
    ) -> bool:
        """Test 4: Meta-model provides meaningful adjustments."""
        
        if self.verbose:
            print(f"   🧪 Check 4: Meta-Model vs Base Model Differences")
        
        gene_predictions = predictions.filter(pl.col('gene_id') == gene_id)
        
        # Check if base model scores are available
        base_cols = ['donor_score', 'acceptor_score', 'neither_score']
        meta_cols = ['donor_meta', 'acceptor_meta', 'neither_meta']
        
        if not all(col in gene_predictions.columns for col in base_cols + meta_cols):
            self.log_check(
                "Base and Meta Predictions Available",
                False,
                f"Missing required columns for comparison"
            )
            return False
        
        # Calculate differences between meta and base predictions
        differences = {}
        for base_col, meta_col in zip(base_cols, meta_cols):
            base_vals = gene_predictions[base_col].to_numpy()
            meta_vals = gene_predictions[meta_col].to_numpy()
            
            abs_diff = np.abs(meta_vals - base_vals)
            mean_diff = np.mean(abs_diff)
            max_diff = np.max(abs_diff)
            
            differences[meta_col] = {
                'mean_abs_diff': mean_diff,
                'max_abs_diff': max_diff,
                'positions_changed': np.sum(abs_diff > 0.01)  # Changed by >1%
            }
        
        # Meta-model should make some meaningful adjustments but not be wildly different
        total_positions = len(gene_predictions)
        positions_adjusted = sum(d['positions_changed'] for d in differences.values())
        adjustment_rate = positions_adjusted / (total_positions * 3)  # 3 prediction types
        
        # Expect 5-50% of predictions to be meaningfully adjusted
        meaningful_adjustment = 0.05 <= adjustment_rate <= 0.50
        
        self.log_check(
            "Meaningful Meta-Model Adjustments",
            meaningful_adjustment,
            f"Adjustment rate: {adjustment_rate:.1%} of predictions",
            "5-50%",
            f"{adjustment_rate:.1%}"
        )
        
        # Check that differences are reasonable (not too extreme)
        overall_mean_diff = np.mean([d['mean_abs_diff'] for d in differences.values()])
        overall_max_diff = np.max([d['max_abs_diff'] for d in differences.values()])
        
        reasonable_differences = overall_mean_diff <= 0.1 and overall_max_diff <= 0.5
        
        self.log_check(
            "Reasonable Difference Magnitude",
            reasonable_differences,
            f"Mean diff: {overall_mean_diff:.3f}, Max diff: {overall_max_diff:.3f}",
            "Mean ≤0.1, Max ≤0.5",
            f"Mean: {overall_mean_diff:.3f}, Max: {overall_max_diff:.3f}"
        )
        
        if self.verbose:
            print(f"      📊 Adjustment details:")
            for pred_type, stats in differences.items():
                print(f"         {pred_type}: {stats['positions_changed']} positions changed")
        
        return meaningful_adjustment and reasonable_differences
    
    def run_sanity_checks(
        self,
        model_path: Path,
        training_dataset_path: Path,
        target_genes: List[str],
        expected_positions: int,
        output_dir: Path,
        check_consistency: bool = True,
        max_positions_per_gene: int = None
    ) -> Dict[str, Any]:
        """Run comprehensive sanity checks."""
        
        print("🧪 Meta-Model Inference Sanity Checks")
        print("=" * 45)
        print(f"   📁 Model: {model_path}")
        print(f"   🧬 Target genes: {len(target_genes)}")
        print(f"   📏 Expected positions per gene: {expected_positions:,}")
        
        # Reset counters
        self.checks_passed = 0
        self.checks_failed = 0
        self.check_details = []
        
        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Use limited positions if specified
        effective_positions = min(expected_positions, max_positions_per_gene) if max_positions_per_gene else expected_positions
        
        try:
            # Step 1: Run inference to get predictions
            if self.verbose:
                print(f"\n🔬 STEP 1: Running inference for sanity testing")
            
            selective_config = SelectiveInferenceConfig(
                model_path=str(model_path),
                target_genes=target_genes,
                training_dataset_path=str(training_dataset_path) if training_dataset_path else None,
                uncertainty_threshold_low=0.01,  # Lower threshold to get more positions
                uncertainty_threshold_high=0.99,  # Higher threshold to get more positions  
                max_positions_per_gene=effective_positions,
                inference_base_dir=output_dir / "sanity_artifacts",
                verbose=max(0, 1 if self.verbose else 0),
                cleanup_intermediates=False
            )
            
            workflow_results = run_selective_meta_inference(selective_config)
            
            if not workflow_results.get('success', False):
                raise RuntimeError("Inference workflow failed for sanity checks")
            
            predictions = workflow_results['predictions']
            
            if self.verbose:
                print(f"   ✅ Inference completed: {len(predictions)} total predictions")
            
            # Step 2: Run checks for each gene
            gene_results = {}
            
            for gene_id in target_genes:
                if self.verbose:
                    print(f"\n📊 STEP 2: Running checks for gene {gene_id}")
                
                # Test 1: Input-Output Length Consistency
                test1_passed = self.check_input_output_length_consistency(
                    gene_id, effective_positions, predictions
                )
                
                # Test 2: Prediction Consistency
                test2_passed = self.check_prediction_consistency(
                    gene_id, predictions, training_dataset_path
                ) if check_consistency else True
                
                # Test 3: Coverage Completeness  
                test3_passed = self.check_coverage_completeness(
                    gene_id, predictions, effective_positions
                )
                
                # Test 4: Meta vs Base Differences
                test4_passed = self.check_meta_model_vs_base_differences(
                    gene_id, predictions
                )
                
                gene_passed = test1_passed and test2_passed and test3_passed and test4_passed
                
                gene_results[gene_id] = {
                    'all_tests_passed': gene_passed,
                    'input_output_consistency': test1_passed,
                    'prediction_consistency': test2_passed,
                    'coverage_completeness': test3_passed,
                    'meta_base_differences': test4_passed
                }
            
            # Step 3: Summary
            if self.verbose:
                print(f"\n📈 STEP 3: Sanity Check Summary")
                print(f"   ✅ Checks passed: {self.checks_passed}")
                print(f"   ❌ Checks failed: {self.checks_failed}")
                print(f"   📊 Success rate: {self.checks_passed/(self.checks_passed + self.checks_failed):.1%}")
            
            # Compile results
            sanity_results = {
                'success': True,
                'config': {
                    'model_path': str(model_path),
                    'target_genes': target_genes,
                    'expected_positions': expected_positions,
                    'effective_positions': effective_positions,
                    'check_consistency': check_consistency
                },
                'summary': {
                    'total_checks': self.checks_passed + self.checks_failed,
                    'checks_passed': self.checks_passed,
                    'checks_failed': self.checks_failed,
                    'success_rate': self.checks_passed/(self.checks_passed + self.checks_failed),
                    'genes_tested': len(target_genes)
                },
                'gene_results': gene_results,
                'check_details': self.check_details,
                'predictions_generated': len(predictions),
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Save results
            results_file = output_dir / "sanity_check_results.json"
            with open(results_file, 'w') as f:
                json.dump(sanity_results, f, indent=2, default=str)
            
            if self.verbose:
                print(f"   💾 Results saved to: {results_file}")
            
            return sanity_results
            
        except Exception as e:
            if self.verbose:
                print(f"   ❌ Sanity checks failed: {e}")
            
            return {
                'success': False,
                'error': str(e),
                'summary': {
                    'checks_passed': self.checks_passed,
                    'checks_failed': self.checks_failed
                }
            }


def parse_genes_input(genes_arg: str, genes_file: Optional[str]) -> List[str]:
    """Parse gene input from command line or file."""
    if genes_file:
        genes_path = Path(genes_file)
        if not genes_path.exists():
            raise FileNotFoundError(f"Gene file not found: {genes_file}")
        
        with open(genes_path, 'r') as f:
            genes = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        
        if not genes:
            raise ValueError(f"No genes found in file: {genes_file}")
        
        return genes
    
    elif genes_arg:
        genes = [g.strip() for g in genes_arg.split(',') if g.strip()]
        if not genes:
            raise ValueError("No valid genes provided")
        return genes
    
    else:
        raise ValueError("Must provide either --genes or --genes-file")


def main():
    parser = argparse.ArgumentParser(
        description="Demo: Meta-Model Inference Sanity Checks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLE USAGE:
    # RECOMMENDED: Run as module from project root:
    python -m meta_spliceai.splice_engine.meta_models.workflows.inference.demo_sanity_checks \\
        --model results/gene_cv_pc_1000_3mers_run_4/model_multiclass.pkl \\
        --training-dataset train_pc_1000_3mers \\
        --genes ENSG00000104435 \\
        --positions 10000

    # ALTERNATIVE: Run script from project root:
    cd /home/bchiu/work/splice-surveyor  # Change to your project root
    python meta_spliceai/splice_engine/meta_models/workflows/inference/demo_sanity_checks.py \\
        --model results/gene_cv_pc_1000_3mers_run_4/model_multiclass.pkl \\
        --training-dataset train_pc_1000_3mers \\
        --genes-file test_genes.txt \\
        --positions 25000 \\
        --check-consistency \\
        --verbose

NOTE: Paths like "results/..." are relative to project root.
      Use absolute paths if running from different directories.
        """
    )
    
    # Required arguments
    parser.add_argument('--model', '-m', required=True, type=Path,
                       help='Path to pre-trained meta-model (.pkl file)')
    parser.add_argument('--training-dataset', '-t', required=True, type=Path,
                       help='Path to training dataset directory')
    parser.add_argument('--positions', '-p', required=True, type=int,
                       help='Expected number of positions per gene')
    
    # Gene specification (mutually exclusive)
    gene_group = parser.add_mutually_exclusive_group(required=True)
    gene_group.add_argument('--genes', '-g', type=str,
                           help='Comma-separated list of gene IDs')
    gene_group.add_argument('--genes-file', '-gf', type=str,
                           help='Path to file containing gene IDs (one per line)')
    
    # Optional arguments
    parser.add_argument('--output-dir', '-o', type=Path,
                       default=Path(tempfile.gettempdir()) / "sanity_checks",
                       help='Output directory for results')
    parser.add_argument('--max-positions', type=int,
                       help='Maximum positions per gene to test (limits memory usage)')
    parser.add_argument('--check-consistency', action='store_true',
                       help='Enable detailed consistency checking')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Suppress all output except errors')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.model.exists():
        print(f"❌ Model file not found: {args.model}")
        return 1
    
    if not args.training_dataset.exists():
        print(f"❌ Training dataset not found: {args.training_dataset}")
        return 1
    
    if args.positions <= 0:
        print(f"❌ Positions must be positive: {args.positions}")
        return 1
    
    # Parse genes
    try:
        target_genes = parse_genes_input(args.genes, args.genes_file)
    except (FileNotFoundError, ValueError) as e:
        print(f"❌ Gene input error: {e}")
        return 1
    
    # Set verbosity
    verbose = args.verbose and not args.quiet
    
    try:
        # Create checker and run
        checker = SanityCheckRunner(verbose=verbose)
        
        results = checker.run_sanity_checks(
            model_path=args.model,
            training_dataset_path=args.training_dataset,
            target_genes=target_genes,
            expected_positions=args.positions,
            output_dir=args.output_dir,
            check_consistency=args.check_consistency,
            max_positions_per_gene=args.max_positions
        )
        
        if results['success']:
            if not args.quiet:
                summary = results['summary']
                print(f"\n✅ Sanity checks completed!")
                print(f"📊 Success rate: {summary['success_rate']:.1%} ({summary['checks_passed']}/{summary['total_checks']})")
            
            # Return exit code based on success rate
            success_rate = results['summary']['success_rate']
            return 0 if success_rate >= 0.8 else 1  # 80% success threshold
        else:
            print(f"❌ Sanity checks failed: {results.get('error', 'Unknown error')}")
            return 1
            
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())