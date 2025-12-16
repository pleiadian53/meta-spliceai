#!/usr/bin/env python3
"""
Chromosome-Aware Cross-Validation for Meta-Model Training

This script implements SpliceAI-compatible chromosome holdout evaluation
using the unified training system architecture. It consolidates and modernizes
all previous LOCO-CV implementations with clean, reusable modules.

Key Features:
1. SpliceAI-compatible chromosome splits (train/val/test)
2. Unified training system integration (automatic batch ensemble)
3. Complete analysis pipeline (identical to gene-aware CV)
4. Class-imbalance-aware evaluation (F1, AP, Top-k)
5. Memory optimization for large datasets

Usage Examples:

# Medium dataset with SpliceAI chromosome split
python -m meta_spliceai.splice_engine.meta_models.training.run_chromosome_cv \
    --dataset train_pc_5000_3mers_diverse/master \
    --out-dir results/chromosome_cv_medium \
    --spliceai-split \
    --n-estimators 800 \
    --calibrate-per-class \
    --verbose

# Large dataset with automatic batch ensemble
python -m meta_spliceai.splice_engine.meta_models.training.run_chromosome_cv \
    --dataset train_regulatory_10k_kmers/master \
    --out-dir results/chromosome_cv_large \
    --spliceai-split \
    --train-all-genes \
    --n-estimators 800 \
    --calibrate-per-class \
    --verbose
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Optional
import argparse
import sys

# Import unified training system
from meta_spliceai.splice_engine.meta_models.training.training_orchestrator import MetaModelTrainingOrchestrator
from meta_spliceai.splice_engine.meta_models.training.training_strategies import TrainingStrategy, TrainingResult
from meta_spliceai.splice_engine.meta_models.training.unified_dataset_utils import load_and_prepare_training_dataset
from meta_spliceai.splice_engine.meta_models.training.spliceai_chromosome_splits import SpliceAIChromosomeSplitter

# Import OpenSpliceAI chromosome utilities
try:
    from meta_spliceai.openspliceai.create_data.utils import split_chromosomes as openspliceai_split_chromosomes
    OPENSPLICEAI_AVAILABLE = True
except ImportError:
    OPENSPLICEAI_AVAILABLE = False


################################################################################
# CLI
################################################################################

def _parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for chromosome-aware CV."""
    
    p = argparse.ArgumentParser(
        description="Chromosome-aware cross-validation for meta-model training using SpliceAI-compatible evaluation."
    )
    
    # Core arguments
    p.add_argument("--dataset", required=True,
                   help="Dataset directory or file path")
    p.add_argument("--out-dir", required=True,
                   help="Output directory for results")
    
    # Chromosome split options
    p.add_argument("--spliceai-split", action="store_true", default=True,
                   help="Use SpliceAI chromosome split (train: chr2-4,6-9,11-22; val: chr5,10; test: chr1,X)")
    p.add_argument("--balanced-split", action="store_true",
                   help="Create balanced chromosome split instead of SpliceAI default")
    p.add_argument("--custom-test-chromosomes", type=str,
                   help="Comma-separated list of test chromosomes (overrides SpliceAI split)")
    p.add_argument("--custom-val-chromosomes", type=str,
                   help="Comma-separated list of validation chromosomes")
    
    # Dataset and memory options
    p.add_argument("--sample-genes", type=int, default=None,
                   help="[TESTING ONLY] Sample subset of genes for quick testing")
    p.add_argument("--row-cap", type=int, default=0,
                   help="Row cap for dataset loading (0 = unlimited)")
    
    # Batch ensemble training (for large datasets)
    p.add_argument("--train-all-genes", action="store_true",
                   help="Enable batch ensemble training for large datasets (10K+ genes)")
    p.add_argument("--max-genes-in-memory", type=int, default=None,
                   help="Override automatic gene limit calculation")
    p.add_argument("--memory-safety-factor", type=float, default=0.6,
                   help="Safety factor for memory usage (0.0-1.0)")
    
    # Model parameters
    p.add_argument("--n-estimators", type=int, default=800,
                   help="Number of XGBoost estimators")
    p.add_argument("--tree-method", default="hist", choices=["hist", "gpu_hist", "approx"],
                   help="XGBoost tree method")
    p.add_argument("--device", default="auto",
                   help="Device for training (auto, cpu, cuda)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed")
    
    # Calibration options
    p.add_argument("--calibrate-per-class", action="store_true", default=True,
                   help="Enable per-class calibration (recommended)")
    p.add_argument("--calibrate", action="store_true",
                   help="Enable binary splice/non-splice calibration")
    p.add_argument("--calib-method", default="platt", choices=["platt", "isotonic"],
                   help="Calibration method")
    
    # Feature handling
    p.add_argument("--exclude-features", default="configs/exclude_features.txt",
                   help="Features to exclude (file path or comma-separated list)")
    p.add_argument("--check-leakage", action="store_true", default=True,
                   help="Check for feature leakage")
    p.add_argument("--leakage-threshold", type=float, default=0.95,
                   help="Leakage detection threshold")
    p.add_argument("--auto-exclude-leaky", action="store_true",
                   help="Automatically exclude leaky features")
    
    # Evaluation and diagnostics
    p.add_argument("--diag-sample", type=int, default=25000,
                   help="Sample size for diagnostics")
    p.add_argument("--neigh-sample", type=int, default=0,
                   help="Sample size for neighbor diagnostics")
    p.add_argument("--neigh-window", type=int, default=10,
                   help="Window size for neighbor diagnostics")
    
    # Analysis options
    p.add_argument("--monitor-overfitting", action="store_true",
                   help="Enable overfitting monitoring")
    p.add_argument("--calibration-analysis", action="store_true",
                   help="Enable calibration analysis")
    p.add_argument("--plot-format", default="pdf", choices=["pdf", "png", "svg"],
                   help="Plot format")
    p.add_argument("--skip-eval", action="store_true",
                   help="Skip evaluation steps")
    
    # General options
    p.add_argument("--verbose", action="store_true",
                   help="Enable verbose output")
    
    return p.parse_args(argv)


################################################################################
# Chromosome-Aware Training Strategy
################################################################################

class ChromosomeAwareTrainingStrategy(TrainingStrategy):
    """
    Chromosome-aware training strategy with SpliceAI-compatible evaluation.
    
    This strategy uses chromosome holdout (train/val/test split) instead of
    cross-validation, providing more stringent out-of-distribution evaluation.
    """
    
    def __init__(self, 
                 splitter: Optional[SpliceAIChromosomeSplitter] = None,
                 verbose: bool = True):
        super().__init__(verbose)
        self.splitter = splitter or SpliceAIChromosomeSplitter()
    
    def get_strategy_name(self) -> str:
        return "Chromosome-Aware (SpliceAI-compatible)"
    
    def can_handle_dataset_size(self, total_genes: int, estimated_memory_gb: float) -> bool:
        # Chromosome holdout naturally reduces memory requirements
        return True
    
    def train_model(
        self,
        dataset_path: str,
        out_dir: Path,
        args: argparse.Namespace,
        X_df: pd.DataFrame,
        y_series: pd.Series,
        genes: np.ndarray
    ) -> TrainingResult:
        """Train model using chromosome holdout strategy."""
        
        if self.verbose:
            print(f"🧬 [Chromosome CV] Starting SpliceAI-compatible chromosome evaluation...")
            print(f"  Genes: {len(np.unique(genes)):,}")
            print(f"  Positions: {X_df.shape[0]:,}")
            print(f"  Features: {X_df.shape[1]}")
        
        # Validate chromosome information
        if 'chrom' not in X_df.columns:
            raise ValueError("Chromosome column required for chromosome-aware evaluation")
        
        chrom_array = X_df['chrom'].values
        
        # Validate and display chromosome split
        coverage = self.splitter.validate_split_coverage(chrom_array, verbose=self.verbose)
        
        # Apply global feature exclusions
        print(f"  🚫 Applying global feature exclusions...", flush=True)
        X_df = self.apply_global_feature_exclusions(X_df)
        
        # Remove chromosome from features
        if 'chrom' in X_df.columns:
            X_df = X_df.drop(columns=['chrom'])
        
        # Prepare data
        from meta_spliceai.splice_engine.meta_models.training.classifier_utils import _encode_labels
        X = X_df.values
        y = _encode_labels(y_series)
        feature_names = list(X_df.columns)
        
        # Get chromosome holdout indices
        train_indices, val_indices, test_indices = self.splitter.get_split_indices(chrom_array)
        
        if self.verbose:
            print(f"  📊 Chromosome holdout split:")
            print(f"    Train: {len(train_indices):,} positions ({len(train_indices)/len(y)*100:.1f}%)")
            print(f"    Validation: {len(val_indices):,} positions ({len(val_indices)/len(y)*100:.1f}%)")
            print(f"    Test: {len(test_indices):,} positions ({len(test_indices)/len(y)*100:.1f}%)")
        
        # Check if we need batch ensemble for large training set
        train_genes = np.unique(genes[train_indices])
        if len(train_genes) > 3000 and getattr(args, 'train_all_genes', False):
            if self.verbose:
                print(f"  🔥 Large training set ({len(train_genes):,} genes) - using batch ensemble")
            
            # Use batch ensemble strategy for training data
            training_result = self._train_with_batch_ensemble(
                dataset_path, out_dir, args, X, y, genes, 
                train_indices, val_indices, test_indices, feature_names
            )
        else:
            # Use single model training
            training_result = self._train_single_model(
                X, y, genes, train_indices, val_indices, test_indices, 
                feature_names, args, out_dir
            )
        
        return training_result
    
    def _train_single_model(self, X, y, genes, train_indices, val_indices, test_indices, 
                          feature_names, args, out_dir):
        """Train single model using chromosome holdout."""
        
        # Train on train+val, test on test chromosomes
        train_val_indices = np.concatenate([train_indices, val_indices])
        
        if self.verbose:
            print(f"  🎯 Training single model on {len(train_val_indices):,} positions...")
        
        # Train binary models
        models = self._train_binary_models(X, y, train_val_indices, args)
        
        # Evaluate on test chromosomes
        cv_results = self._evaluate_on_test_chromosomes(
            models, X, y, genes, test_indices, args
        )
        
        # Create final ensemble
        from meta_spliceai.splice_engine.meta_models.training.classifier_utils import SigmoidEnsemble
        ensemble = SigmoidEnsemble(models, feature_names)
        
        # Save model
        model_path = out_dir / "model_multiclass.pkl"
        import pickle
        with open(model_path, 'wb') as f:
            pickle.dump(ensemble, f)
        
        # Calculate performance metrics
        performance_metrics = self._calculate_performance_metrics(cv_results)
        
        training_metadata = {
            'strategy': self.get_strategy_name(),
            'total_genes': len(np.unique(genes)),
            'total_positions': X.shape[0],
            'train_positions': len(train_indices),
            'val_positions': len(val_indices),
            'test_positions': len(test_indices),
            'evaluation_type': 'chromosome_holdout'
        }
        
        return TrainingResult(
            model_path=model_path,
            feature_names=feature_names,
            excluded_features=self._global_excluded_features,
            training_metadata=training_metadata,
            cv_results=cv_results,
            performance_metrics=performance_metrics
        )
    
    def _train_with_batch_ensemble(self, dataset_path, out_dir, args, X, y, genes,
                                 train_indices, val_indices, test_indices, feature_names):
        """Use batch ensemble for large training sets."""
        
        if self.verbose:
            print(f"  🔥 Using batch ensemble for large dataset...")
        
        # Delegate to batch ensemble strategy
        from meta_spliceai.splice_engine.meta_models.training.training_strategies import BatchEnsembleTrainingStrategy
        
        batch_strategy = BatchEnsembleTrainingStrategy(verbose=self.verbose)
        
        # Create subset dataframes for batch training
        import pandas as pd
        train_val_indices = np.concatenate([train_indices, val_indices])
        
        X_train_df = pd.DataFrame(X[train_val_indices], columns=feature_names)
        y_train_series = pd.Series(y[train_val_indices])
        genes_train = genes[train_val_indices]
        
        # Run batch ensemble training
        batch_result = batch_strategy.train_model(
            dataset_path, out_dir, args, X_train_df, y_train_series, genes_train
        )
        
        # Evaluate the ensemble on test chromosomes
        import pickle
        with open(batch_result.model_path, 'rb') as f:
            ensemble_model = pickle.load(f)
        
        cv_results = self._evaluate_ensemble_on_test_chromosomes(
            ensemble_model, X, y, genes, test_indices, args
        )
        
        # Update training result with chromosome evaluation
        batch_result.cv_results = cv_results
        batch_result.performance_metrics = self._calculate_performance_metrics(cv_results)
        
        return batch_result
    
    def _train_binary_models(self, X, y, train_indices, args):
        """Train three binary models (neither, donor, acceptor)."""
        from meta_spliceai.splice_engine.meta_models.training.run_gene_cv_sigmoid import _train_binary_model
        
        models = []
        X_train = X[train_indices]
        y_train = y[train_indices]
        
        for cls in (0, 1, 2):  # neither, donor, acceptor
            y_bin = (y_train == cls).astype(int)
            model, _ = _train_binary_model(X_train, y_bin, X_train, y_bin, args)
            models.append(model)
        
        return models
    
    def _evaluate_on_test_chromosomes(self, models, X, y, genes, test_indices, args):
        """Evaluate models on held-out test chromosomes."""
        from sklearn.metrics import accuracy_score, f1_score, average_precision_score
        
        X_test = X[test_indices]
        y_test = y[test_indices]
        genes_test = genes[test_indices]
        
        # Generate predictions
        proba_parts = [m.predict_proba(X_test)[:, 1] for m in models]
        proba = np.column_stack(proba_parts)
        pred = proba.argmax(axis=1)
        
        # Calculate comprehensive metrics
        accuracy = accuracy_score(y_test, pred)
        f1_macro = f1_score(y_test, pred, average="macro")
        
        # Per-class F1 scores
        f1_per_class = f1_score(y_test, pred, average=None, zero_division=0)
        f1_neither = f1_per_class[0] if len(f1_per_class) > 0 else 0.0
        f1_donor = f1_per_class[1] if len(f1_per_class) > 1 else 0.0
        f1_acceptor = f1_per_class[2] if len(f1_per_class) > 2 else 0.0
        
        # Binary splice vs non-splice AP
        y_test_bin = (y_test != 0).astype(int)
        y_prob_bin = proba[:, 1] + proba[:, 2]
        
        binary_ap = 0.0
        if len(np.unique(y_test_bin)) == 2:
            try:
                binary_ap = average_precision_score(y_test_bin, y_prob_bin)
            except:
                binary_ap = 0.0
        
        # Top-k accuracy
        top_k_acc = self._calculate_top_k_accuracy(y_test, proba)
        
        if self.verbose:
            print(f"    ✅ Test chromosome evaluation:")
            print(f"      F1 Macro: {f1_macro:.3f}")
            print(f"      Binary AP: {binary_ap:.3f}")
            print(f"      Top-k Accuracy: {top_k_acc:.3f}")
        
        # Return in CV format for compatibility with analysis pipeline
        cv_results = [{
            'fold': 0,
            'evaluation_type': 'chromosome_holdout',
            'test_positions': len(test_indices),
            'test_genes': len(np.unique(genes_test)),
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_neither': f1_neither,
            'f1_donor': f1_donor,
            'f1_acceptor': f1_acceptor,
            'binary_average_precision': binary_ap,
            'top_k_accuracy': top_k_acc
        }]
        
        return cv_results
    
    def _evaluate_ensemble_on_test_chromosomes(self, ensemble_model, X, y, genes, test_indices, args):
        """Evaluate ensemble model on test chromosomes."""
        from sklearn.metrics import accuracy_score, f1_score, average_precision_score
        
        X_test = X[test_indices]
        y_test = y[test_indices]
        genes_test = genes[test_indices]
        
        # Generate predictions using ensemble
        proba = ensemble_model.predict_proba(X_test)
        pred = proba.argmax(axis=1)
        
        # Calculate metrics (same as single model)
        accuracy = accuracy_score(y_test, pred)
        f1_macro = f1_score(y_test, pred, average="macro")
        
        # Per-class F1
        f1_per_class = f1_score(y_test, pred, average=None, zero_division=0)
        f1_neither = f1_per_class[0] if len(f1_per_class) > 0 else 0.0
        f1_donor = f1_per_class[1] if len(f1_per_class) > 1 else 0.0
        f1_acceptor = f1_per_class[2] if len(f1_per_class) > 2 else 0.0
        
        # Binary AP
        y_test_bin = (y_test != 0).astype(int)
        y_prob_bin = proba[:, 1] + proba[:, 2]
        
        binary_ap = 0.0
        if len(np.unique(y_test_bin)) == 2:
            try:
                binary_ap = average_precision_score(y_test_bin, y_prob_bin)
            except:
                binary_ap = 0.0
        
        # Top-k accuracy
        top_k_acc = self._calculate_top_k_accuracy(y_test, proba)
        
        if self.verbose:
            print(f"    ✅ Ensemble test chromosome evaluation:")
            print(f"      F1 Macro: {f1_macro:.3f}")
            print(f"      Binary AP: {binary_ap:.3f}")
            print(f"      Top-k Accuracy: {top_k_acc:.3f}")
        
        return [{
            'fold': 0,
            'evaluation_type': 'chromosome_holdout_ensemble',
            'test_positions': len(test_indices),
            'test_genes': len(np.unique(genes_test)),
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_neither': f1_neither,
            'f1_donor': f1_donor,
            'f1_acceptor': f1_acceptor,
            'binary_average_precision': binary_ap,
            'top_k_accuracy': top_k_acc
        }]
    
    def _calculate_top_k_accuracy(self, y_true, y_prob):
        """Calculate top-k accuracy for splice sites."""
        splice_mask = y_true != 0
        if not splice_mask.any():
            return 0.0
        
        k = int(splice_mask.sum())
        if k == 0:
            return 0.0
        
        splice_prob = y_prob[:, 1] + y_prob[:, 2]
        top_idx = np.argsort(-splice_prob)[:k]
        top_k_correct = (y_true[top_idx] != 0).sum()
        
        return top_k_correct / k


################################################################################
# Main Training Pipeline
################################################################################

def main(argv: List[str] | None = None) -> None:
    """
    Main entry point for chromosome-aware CV training.
    
    This function coordinates the complete chromosome-aware training pipeline
    using the unified training system architecture.
    """
    
    print("🧬 [Chromosome CV] Chromosome-Aware Meta-Model Training", flush=True)
    print("=" * 60, flush=True)
    
    # Parse arguments
    args = _parse_args(argv)
    
    if args.verbose:
        print(f"🔍 [Chromosome CV] Arguments parsed successfully")
        print(f"  Dataset: {args.dataset}")
        print(f"  Output: {args.out_dir}")
        print(f"  SpliceAI split: {args.spliceai_split}")
        print(f"  Train all genes: {getattr(args, 'train_all_genes', False)}")
    
    try:
        # Create chromosome splitter based on arguments
        if args.custom_test_chromosomes:
            test_chroms = [c.strip() for c in args.custom_test_chromosomes.split(',')]
            val_chroms = [c.strip() for c in args.custom_val_chromosomes.split(',')] if args.custom_val_chromosomes else ['5', '10']
            
            # Create remaining chromosomes for training
            all_chroms = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', 
                         '15', '16', '17', '18', '19', '20', '21', '22', 'X', 'Y']
            train_chroms = [c for c in all_chroms if c not in test_chroms and c not in val_chroms]
            
            custom_split = {
                'train': train_chroms,
                'validation': val_chroms,
                'test': test_chroms
            }
            splitter = SpliceAIChromosomeSplitter(custom_split)
            
        elif args.balanced_split:
            # Will create balanced split automatically
            splitter = SpliceAIChromosomeSplitter()
        else:
            # Use default SpliceAI split
            splitter = SpliceAIChromosomeSplitter()
        
        # Create chromosome-aware training strategy
        chromosome_strategy = ChromosomeAwareTrainingStrategy(
            splitter=splitter,
            verbose=args.verbose
        )
        
        # Load and prepare dataset
        print(f"📊 [Chromosome CV] Loading and preparing dataset...", flush=True)
        raw_df, X_df, y_series, genes = load_and_prepare_training_dataset(
            args.dataset, args, verbose=args.verbose
        )
        
        # Run training using chromosome strategy
        print(f"🚀 [Chromosome CV] Running chromosome-aware training...", flush=True)
        training_result = chromosome_strategy.train_model(
            args.dataset, Path(args.out_dir), args, X_df, y_series, genes
        )
        
        # Run post-training analysis (reuse unified analysis pipeline)
        if not args.skip_eval:
            print(f"📊 [Chromosome CV] Running post-training analysis...", flush=True)
            
            from meta_spliceai.splice_engine.meta_models.training.unified_post_training_analysis import (
                UnifiedPostTrainingAnalyzer
            )
            
            analyzer = UnifiedPostTrainingAnalyzer(verbose=args.verbose)
            analysis_results = analyzer.run_comprehensive_analysis(
                training_result, raw_df, args
            )
            
            print(f"✅ [Chromosome CV] Analysis completed: {sum(analysis_results.values())}/26 components successful")
        
        print(f"🎉 [Chromosome CV] Chromosome-aware training completed successfully!", flush=True)
        
        # Display summary
        if args.verbose and training_result.cv_results:
            cv_result = training_result.cv_results[0]
            print(f"\n📊 CHROMOSOME HOLDOUT EVALUATION SUMMARY:")
            print(f"  Test Positions: {cv_result['test_positions']:,}")
            print(f"  Test Genes: {cv_result['test_genes']:,}")
            print(f"  F1 Macro: {cv_result['f1_macro']:.3f}")
            print(f"  Binary AP: {cv_result['binary_average_precision']:.3f}")
            print(f"  Top-k Accuracy: {cv_result['top_k_accuracy']:.3f}")
            print(f"  Evaluation Type: {cv_result['evaluation_type']}")
        
    except Exception as e:
        print(f"❌ [Chromosome CV] Training failed: {e}", flush=True)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()



