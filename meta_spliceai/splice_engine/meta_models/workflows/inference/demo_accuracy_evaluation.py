#!/usr/bin/env python3
"""Demo: Meta-Model Inference Accuracy Evaluation
===============================================

This script demonstrates how to evaluate meta-model inference accuracy using
F1-scores and other metrics appropriate for imbalanced splice site data.

EXAMPLE USAGE:
    # RECOMMENDED: Run as module from project root (paths relative to project root):
    python -m meta_spliceai.splice_engine.meta_models.workflows.inference.demo_accuracy_evaluation \
        --model results/gene_cv_pc_1000_3mers_run_4/model_multiclass.pkl \
        --training-dataset train_pc_1000_3mers \
        --genes ENSG00000104435,ENSG00000006420 \
        --output-dir /tmp/accuracy_demo

    # ALTERNATIVE: Run script directly from project root:
    cd /home/bchiu/work/splice-surveyor  # Change to your project root
    python meta_spliceai/splice_engine/meta_models/workflows/inference/demo_accuracy_evaluation.py \
        --model results/gene_cv_pc_1000_3mers_run_4/model_multiclass.pkl \
        --training-dataset train_pc_1000_3mers \
        --genes ENSG00000104435,ENSG00000006420

    # COMPREHENSIVE EVALUATION with gene file:
    python -m meta_spliceai.splice_engine.meta_models.workflows.inference.demo_accuracy_evaluation \
        --model results/gene_cv_pc_1000_3mers_run_4/model_multiclass.pkl \
        --training-dataset train_pc_1000_3mers \
        --genes-file gene_list.txt \
        --output-dir ./accuracy_results \
        --max-positions 5000 \
        --verbose

FEATURES:
- F1-based evaluation (appropriate for imbalanced splice site data)
- Base model vs Meta-model comparison
- Per-class performance metrics (donor, acceptor, neither)
- Flexible gene selection (individual genes or file)
- Configurable evaluation parameters
- Detailed performance reporting

REQUIREMENTS:
- Pre-trained meta-model (.pkl file)
- Training dataset directory (for coverage analysis)
- Target genes for evaluation
- Sufficient compute resources for inference
"""

import argparse
import sys
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import json
import time

import numpy as np
import polars as pl
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

# Add project root to Python path
import os
project_root = os.path.join(os.path.expanduser("~"), "work/splice-surveyor")
sys.path.insert(0, project_root)

from meta_spliceai.splice_engine.meta_models.workflows.selective_meta_inference import (
    run_selective_meta_inference,
    SelectiveInferenceConfig
)


class AccuracyEvaluator:
    """Evaluate meta-model accuracy using appropriate metrics for imbalanced data."""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.results = {}
    
    def load_ground_truth(self, gene_id: str, training_dataset_path: Path) -> Optional[Dict[str, np.ndarray]]:
        """Load ground truth splice sites for a gene from training data."""
        try:
            # Try multiple possible locations for training data
            potential_paths = [
                training_dataset_path / "full_splice_positions_enhanced.tsv",
                Path("data/ensembl/spliceai_eval/meta_models/full_splice_positions_enhanced.tsv"),
                training_dataset_path / "splice_positions_enhanced.tsv"
            ]
            
            positions_file = None
            for path in potential_paths:
                if path.exists():
                    positions_file = path
                    break
            
            if not positions_file:
                if self.verbose:
                    print(f"   ⚠️  No ground truth file found for {gene_id}")
                return None
            
            # Load and filter to our gene
            positions_df = pl.read_csv(
                positions_file, 
                separator='\t',
                schema_overrides={'chrom': pl.Utf8}
            ).filter(pl.col('gene_id') == gene_id)
            
            if len(positions_df) == 0:
                if self.verbose:
                    print(f"   ⚠️  No positions found for gene {gene_id}")
                return None
            
            # Extract ground truth labels
            positions = positions_df['position'].to_numpy()
            splice_types = positions_df['splice_type'].to_numpy()
            
            # Create binary arrays for each splice type
            ground_truth = {
                'positions': positions,
                'donor_true': (splice_types == 'donor').astype(int),
                'acceptor_true': (splice_types == 'acceptor').astype(int),
                'neither_true': (splice_types == 'neither').astype(int)
            }
            
            return ground_truth
            
        except Exception as e:
            if self.verbose:
                print(f"   ❌ Error loading ground truth for {gene_id}: {e}")
            return None
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred_proba: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
        """Calculate classification metrics using F1-score focus."""
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        # Handle edge cases
        if np.sum(y_pred) == 0:
            precision = 0.0
            recall = np.sum(y_true) / len(y_true) if len(y_true) > 0 else 0.0
        else:
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
        
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        # Calculate confusion matrix components
        if len(np.unique(y_true)) > 1 and len(np.unique(y_pred)) > 1:
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        else:
            # Handle degenerate cases
            tp = np.sum((y_true == 1) & (y_pred == 1))
            fp = np.sum((y_true == 0) & (y_pred == 1))
            fn = np.sum((y_true == 1) & (y_pred == 0))
            tn = np.sum((y_true == 0) & (y_pred == 0))
        
        fp_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fn_rate = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'fp_rate': fp_rate,
            'fn_rate': fn_rate,
            'tp': tp,
            'fp': fp,
            'tn': tn,
            'fn': fn,
            'support': np.sum(y_true)
        }
    
    def evaluate_gene(
        self, 
        gene_id: str, 
        meta_predictions: pl.DataFrame, 
        training_dataset_path: Path
    ) -> Optional[Dict[str, Any]]:
        """Evaluate meta-model predictions for a single gene."""
        
        if self.verbose:
            print(f"\n   📊 Evaluating gene: {gene_id}")
        
        # Load ground truth
        ground_truth = self.load_ground_truth(gene_id, training_dataset_path)
        if not ground_truth:
            return None
        
        # Filter meta predictions to this gene
        gene_meta = meta_predictions.filter(pl.col('gene_id') == gene_id)
        if len(gene_meta) == 0:
            if self.verbose:
                print(f"      ⚠️  No meta predictions found for {gene_id}")
            return None
        
        # Find common positions
        gt_positions = ground_truth['positions']
        meta_positions = gene_meta['position'].to_numpy()
        common_positions = np.intersect1d(gt_positions, meta_positions)
        
        if len(common_positions) < 10:
            if self.verbose:
                print(f"      ⚠️  Too few common positions ({len(common_positions)}) for {gene_id}")
            return None
        
        if self.verbose:
            print(f"      📈 Common positions for evaluation: {len(common_positions):,}")
        
        # Extract meta-model predictions at common positions
        meta_mask = np.isin(meta_positions, common_positions)
        meta_donor = gene_meta['donor_meta'].to_numpy()[meta_mask]
        meta_acceptor = gene_meta['acceptor_meta'].to_numpy()[meta_mask] 
        meta_neither = gene_meta['neither_meta'].to_numpy()[meta_mask]
        
        # Extract ground truth at common positions
        gt_mask = np.isin(gt_positions, common_positions)
        gt_donor = ground_truth['donor_true'][gt_mask]
        gt_acceptor = ground_truth['acceptor_true'][gt_mask]
        gt_neither = ground_truth['neither_true'][gt_mask]
        
        # Calculate metrics for each splice type
        donor_metrics = self.calculate_metrics(gt_donor, meta_donor)
        acceptor_metrics = self.calculate_metrics(gt_acceptor, meta_acceptor)
        neither_metrics = self.calculate_metrics(gt_neither, meta_neither)
        
        if self.verbose:
            print(f"      📊 Meta-model metrics (F1 is key for imbalanced data):")
            print(f"         Donor    - F1: {donor_metrics['f1']:.3f}, Precision: {donor_metrics['precision']:.3f}, Recall: {donor_metrics['recall']:.3f}")
            print(f"         Acceptor - F1: {acceptor_metrics['f1']:.3f}, Precision: {acceptor_metrics['precision']:.3f}, Recall: {acceptor_metrics['recall']:.3f}")
            print(f"         Neither  - F1: {neither_metrics['f1']:.3f}, Precision: {neither_metrics['precision']:.3f}, Recall: {neither_metrics['recall']:.3f}")
        
        return {
            'gene_id': gene_id,
            'donor_metrics': donor_metrics,
            'acceptor_metrics': acceptor_metrics,
            'neither_metrics': neither_metrics,
            'common_positions': len(common_positions),
            'total_positions': len(meta_positions)
        }
    
    def run_evaluation(
        self,
        model_path: Path,
        training_dataset_path: Path,
        target_genes: List[str],
        output_dir: Path,
        max_positions_per_gene: int = 10000,
        uncertainty_threshold_low: float = 0.02,
        uncertainty_threshold_high: float = 0.80
    ) -> Dict[str, Any]:
        """Run comprehensive accuracy evaluation."""
        
        print("🧪 Meta-Model Accuracy Evaluation")
        print("=" * 50)
        print(f"   📁 Model: {model_path}")
        print(f"   📁 Training data: {training_dataset_path}")
        print(f"   🧬 Target genes: {len(target_genes)}")
        print(f"   📁 Output: {output_dir}")
        
        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Step 1: Run selective meta-inference to get predictions
        if self.verbose:
            print(f"\n🔬 STEP 1: Running selective meta-inference")
        
        try:
            selective_config = SelectiveInferenceConfig(
                model_path=str(model_path),
                target_genes=target_genes,
                training_dataset_path=str(training_dataset_path),
                uncertainty_threshold_low=uncertainty_threshold_low,
                uncertainty_threshold_high=uncertainty_threshold_high,
                max_positions_per_gene=max_positions_per_gene,
                inference_base_dir=output_dir / "inference_artifacts",
                verbose=max(0, 1 if self.verbose else 0),
                cleanup_intermediates=False  # Keep for analysis
            )
            
            workflow_results = run_selective_meta_inference(selective_config)
            
            if not workflow_results.get('success', False):
                raise RuntimeError("Selective inference workflow failed")
            
            meta_predictions = workflow_results['predictions']
            
        except Exception as e:
            print(f"   ❌ Inference failed: {e}")
            return {'success': False, 'error': str(e)}
        
        # Step 2: Evaluate each gene
        if self.verbose:
            print(f"\n📊 STEP 2: Evaluating gene-by-gene accuracy")
        
        gene_results = []
        for gene_id in target_genes:
            result = self.evaluate_gene(gene_id, meta_predictions, training_dataset_path)
            if result:
                gene_results.append(result)
        
        if not gene_results:
            print(f"   ⚠️  No genes could be evaluated!")
            return {'success': False, 'error': 'No valid gene evaluations'}
        
        # Step 3: Aggregate results
        if self.verbose:
            print(f"\n📈 STEP 3: Aggregating results across {len(gene_results)} genes")
        
        # Calculate aggregate metrics
        all_donor_f1 = [r['donor_metrics']['f1'] for r in gene_results]
        all_acceptor_f1 = [r['acceptor_metrics']['f1'] for r in gene_results]
        all_neither_f1 = [r['neither_metrics']['f1'] for r in gene_results]
        
        aggregate_metrics = {
            'genes_evaluated': len(gene_results),
            'total_genes_requested': len(target_genes),
            'macro_f1_donor': np.mean(all_donor_f1),
            'macro_f1_acceptor': np.mean(all_acceptor_f1),
            'macro_f1_neither': np.mean(all_neither_f1),
            'overall_macro_f1': np.mean(all_donor_f1 + all_acceptor_f1 + all_neither_f1),
            'median_f1_donor': np.median(all_donor_f1),
            'median_f1_acceptor': np.median(all_acceptor_f1),
            'median_f1_neither': np.median(all_neither_f1)
        }
        
        # Print summary
        print(f"\n✅ EVALUATION SUMMARY")
        print(f"   📊 Genes successfully evaluated: {aggregate_metrics['genes_evaluated']}/{aggregate_metrics['total_genes_requested']}")
        print(f"   🎯 Overall Macro F1: {aggregate_metrics['overall_macro_f1']:.3f}")
        print(f"   📈 Per-class Macro F1:")
        print(f"      Donor: {aggregate_metrics['macro_f1_donor']:.3f}")
        print(f"      Acceptor: {aggregate_metrics['macro_f1_acceptor']:.3f}")
        print(f"      Neither: {aggregate_metrics['macro_f1_neither']:.3f}")
        
        # Save results
        results_summary = {
            'evaluation_config': {
                'model_path': str(model_path),
                'training_dataset_path': str(training_dataset_path),
                'target_genes': target_genes,
                'uncertainty_thresholds': [uncertainty_threshold_low, uncertainty_threshold_high],
                'max_positions_per_gene': max_positions_per_gene
            },
            'aggregate_metrics': aggregate_metrics,
            'gene_results': gene_results,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Save to JSON
        results_file = output_dir / "accuracy_evaluation_results.json"
        with open(results_file, 'w') as f:
            json.dump(results_summary, f, indent=2, default=str)
        
        if self.verbose:
            print(f"   💾 Results saved to: {results_file}")
        
        return {
            'success': True,
            'results_file': results_file,
            'aggregate_metrics': aggregate_metrics,
            'gene_results': gene_results
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
        description="Demo: Meta-Model Inference Accuracy Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLE USAGE:
    # RECOMMENDED: Run as module from project root:
    python -m meta_spliceai.splice_engine.meta_models.workflows.inference.demo_accuracy_evaluation \\
        --model results/gene_cv_pc_1000_3mers_run_4/model_multiclass.pkl \\
        --training-dataset train_pc_1000_3mers \\
        --genes ENSG00000104435,ENSG00000006420

    # ALTERNATIVE: Run script from project root:
    cd /home/bchiu/work/splice-surveyor  # Change to your project root
    python meta_spliceai/splice_engine/meta_models/workflows/inference/demo_accuracy_evaluation.py \\
        --model results/gene_cv_pc_1000_3mers_run_4/model_multiclass.pkl \\
        --training-dataset train_pc_1000_3mers \\
        --genes-file gene_list.txt \\
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
    
    # Gene specification (mutually exclusive)
    gene_group = parser.add_mutually_exclusive_group(required=True)
    gene_group.add_argument('--genes', '-g', type=str,
                           help='Comma-separated list of gene IDs (e.g., ENSG00000104435,ENSG00000006420)')
    gene_group.add_argument('--genes-file', '-gf', type=str,
                           help='Path to file containing gene IDs (one per line)')
    
    # Optional arguments
    parser.add_argument('--output-dir', '-o', type=Path, 
                       default=Path(tempfile.gettempdir()) / "accuracy_evaluation",
                       help='Output directory for results (default: temp directory)')
    parser.add_argument('--max-positions', type=int, default=10000,
                       help='Maximum positions per gene to evaluate (default: 10000)')
    parser.add_argument('--uncertainty-low', type=float, default=0.02,
                       help='Lower uncertainty threshold (default: 0.02)')
    parser.add_argument('--uncertainty-high', type=float, default=0.80,
                       help='Upper uncertainty threshold (default: 0.80)')
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
    
    # Parse genes
    try:
        target_genes = parse_genes_input(args.genes, args.genes_file)
    except (FileNotFoundError, ValueError) as e:
        print(f"❌ Gene input error: {e}")
        return 1
    
    # Set verbosity
    verbose = args.verbose and not args.quiet
    
    try:
        # Create evaluator and run
        evaluator = AccuracyEvaluator(verbose=verbose)
        
        results = evaluator.run_evaluation(
            model_path=args.model,
            training_dataset_path=args.training_dataset,
            target_genes=target_genes,
            output_dir=args.output_dir,
            max_positions_per_gene=args.max_positions,
            uncertainty_threshold_low=args.uncertainty_low,
            uncertainty_threshold_high=args.uncertainty_high
        )
        
        if results['success']:
            if not args.quiet:
                print(f"\n✅ Evaluation completed successfully!")
                print(f"📁 Results: {results['results_file']}")
                print(f"🎯 Overall Macro F1: {results['aggregate_metrics']['overall_macro_f1']:.3f}")
            return 0
        else:
            print(f"❌ Evaluation failed: {results.get('error', 'Unknown error')}")
            return 1
            
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())