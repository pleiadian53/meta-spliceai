#!/usr/bin/env python3
"""
Gene-aware Cross-Validation for Deep Learning Models

This module provides gene-aware cross-validation specifically designed for deep learning
models that use multi-class classification instead of the 3-binary-classifier approach
used in run_gene_cv_sigmoid.py.

Supported Models:
- TabNet (multi-class attention-based)
- TensorFlow MLP (multi-class neural networks)
- Multi-modal transformer models (sequence + features)
- Any sklearn-compatible multi-class classifier

Key Features:
- Gene-aware cross-validation (no data leakage between genes)
- Multi-class classification (neither/donor/acceptor)
- Support for both tabular and sequence-based models
- Comprehensive evaluation metrics
- Model-agnostic design

Usage:
    python run_gene_cv_deep_learning.py \
        --dataset data/ensembl/spliceai_analysis \
        --out-dir results/tabnet_cv \
        --algorithm tabnet \
        --n-folds 5
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, precision_score, recall_score,
    roc_curve, precision_recall_curve, auc, average_precision_score,
    classification_report
)

# Import MetaSpliceAI modules
from meta_spliceai.splice_engine.meta_models.training import datasets
from meta_spliceai.splice_engine.meta_models.builder import preprocessing
from meta_spliceai.splice_engine.meta_models.training.models import get_model, AVAILABLE_MODELS
from meta_spliceai.splice_engine.meta_models.training.classifier_utils import _encode_labels
from meta_spliceai.splice_engine.meta_models.evaluation.viz_utils import (
    plot_roc_pr_curves, plot_combined_roc_pr_curves_meta
)
from meta_spliceai.splice_engine.meta_models.evaluation.multiclass_roc_pr import (
    plot_multiclass_roc_pr_curves
)
from meta_spliceai.splice_engine.utils_doc import print_emphasized, print_with_indent

# Import CV utilities
from meta_spliceai.splice_engine.meta_models.training import cv_utils

logger = logging.getLogger(__name__)


@dataclass
class DeepLearningCVConfig:
    """Configuration for deep learning gene-aware CV."""
    
    # Dataset configuration
    dataset_path: Path
    output_dir: Path
    gene_col: str = "gene_id"
    
    # Model configuration
    algorithm: str = "tabnet"
    algorithm_params: Optional[Dict[str, Any]] = None
    
    # CV configuration
    n_folds: int = 5
    valid_size: float = 0.1
    random_seed: int = 42
    
    # Data configuration
    max_variants: Optional[int] = None
    row_cap: int = 100_000
    
    # Evaluation configuration
    plot_curves: bool = True
    plot_format: str = "pdf"
    n_roc_points: int = 101
    
    # Model-specific configuration
    use_sequence_data: bool = False  # For multi-modal models
    sequence_length: int = 1000      # For sequence-based models
    include_additional_features: bool = True  # For multi-modal models


class DeepLearningGeneCV:
    """
    Gene-aware cross-validation for deep learning models.
    
    This class handles the complete CV pipeline for deep learning models that use
    multi-class classification instead of the 3-binary-classifier approach.
    """
    
    def __init__(self, config: DeepLearningCVConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize model
        self.model = self._create_model()
        
        # Results storage
        self.cv_results = []
        self.fold_metrics = []
        
    def _create_model(self):
        """Create the specified model."""
        model_params = self.config.algorithm_params or {}
        
        # Handle special cases for deep learning models
        if self.config.algorithm == "tf_mlp_multiclass":
            # Create multi-class TensorFlow model
            model_params.update({
                "n_classes": 3,  # neither, donor, acceptor
                "loss": "categorical_crossentropy",
                "output_activation": "softmax"
            })
        elif self.config.algorithm == "tabnet":
            # TabNet is already multi-class by default
            model_params.update({
                "n_d": model_params.get("n_d", 64),
                "n_a": model_params.get("n_a", 64),
                "n_steps": model_params.get("n_steps", 5),
            })
        
        # Create model using the registry
        model_spec = {"name": self.config.algorithm, **model_params}
        return get_model(model_spec)
    
    def run_cross_validation(self) -> Dict[str, Any]:
        """
        Run complete gene-aware cross-validation.
        
        Returns:
            Dict containing CV results, metrics, and artifacts
        """
        print_emphasized(f"🧬 Starting Gene-Aware CV for {self.config.algorithm.upper()}")
        print_emphasized(f"📊 Dataset: {self.config.dataset_path}")
        print_emphasized(f"📁 Output: {self.config.output_dir}")
        
        # Setup output directory
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load and prepare data
        print_emphasized("📥 Loading and preparing data...")
        X, y, genes, feature_names = self._load_and_prepare_data()
        
        # Run CV
        print_emphasized(f"🔄 Running {self.config.n_folds}-fold gene-aware CV...")
        self._run_cv_loop(X, y, genes, feature_names)
        
        # Generate results
        print_emphasized("📈 Generating results and visualizations...")
        results = self._generate_results()
        
        print_emphasized("✅ Gene-aware CV completed successfully!")
        return results
    
    def _load_and_prepare_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """Load and prepare data for CV."""
        
        # Load dataset
        df = datasets.load_dataset(self.config.dataset_path)
        
        if self.config.gene_col not in df.columns:
            raise KeyError(f"Gene column '{self.config.gene_col}' not found in dataset")
        
        # Apply row cap if specified
        if self.config.max_variants:
            df = df.head(self.config.max_variants)
        elif self.config.row_cap > 0:
            df = df.head(self.config.row_cap)
        
        # Prepare features and labels
        X_df, y_series = preprocessing.prepare_training_data(
            df,
            label_col="splice_type",
            return_type="pandas",
            verbose=1,
            preserve_transcript_columns=True,
            encode_chrom=True
        )
        
        # Convert to numpy arrays
        X = X_df.values
        y = _encode_labels(y_series)
        genes = df[self.config.gene_col].to_numpy()
        feature_names = list(X_df.columns)
        
        print_with_indent(f"📊 Data shape: {X.shape}")
        print_with_indent(f"🧬 Unique genes: {len(np.unique(genes))}")
        print_with_indent(f"🏷️  Label distribution: {np.bincount(y)}")
        
        return X, y, genes, feature_names
    
    def _run_cv_loop(self, X: np.ndarray, y: np.ndarray, genes: np.ndarray, 
                    feature_names: List[str]) -> None:
        """Run the main CV loop."""
        
        # Initialize CV
        gkf = GroupKFold(n_splits=self.config.n_folds)
        
        # Storage for results
        y_true_all = []
        y_pred_all = []
        y_prob_all = []
        
        for fold_idx, (train_val_idx, test_idx) in enumerate(gkf.split(X, y, groups=genes)):
            print_with_indent(f"🔄 Fold {fold_idx + 1}/{self.config.n_folds}")
            print_with_indent(f"   Test samples: {len(test_idx)}")
            
            # Train/validation split preserving gene groups
            rel_valid = self.config.valid_size / (1.0 - 1.0 / self.config.n_folds)
            gss = GroupShuffleSplit(n_splits=1, test_size=rel_valid, random_state=self.config.random_seed)
            train_idx, valid_idx = next(gss.split(train_val_idx, y[train_val_idx], groups=genes[train_val_idx]))
            train_idx = train_val_idx[train_idx]
            valid_idx = train_val_idx[valid_idx]
            
            # Train model
            print_with_indent("   🤖 Training model...")
            self._train_model(X[train_idx], y[train_idx], X[valid_idx], y[valid_idx])
            
            # Evaluate on test set
            print_with_indent("   📊 Evaluating on test set...")
            y_pred, y_prob = self._predict(X[test_idx])
            
            # Store results
            y_true_all.extend(y[test_idx])
            y_pred_all.extend(y_pred)
            y_prob_all.extend(y_prob)
            
            # Calculate fold metrics
            fold_metrics = self._calculate_fold_metrics(
                y[test_idx], y_pred, y_prob, fold_idx, len(test_idx)
            )
            self.fold_metrics.append(fold_metrics)
            
            print_with_indent(f"   ✅ Fold {fold_idx + 1} completed")
        
        # Store all results
        self.cv_results = {
            'y_true': np.array(y_true_all),
            'y_pred': np.array(y_pred_all),
            'y_prob': np.array(y_prob_all)
        }
    
    def _train_model(self, X_train: np.ndarray, y_train: np.ndarray, 
                    X_val: np.ndarray, y_val: np.ndarray) -> None:
        """Train the model with appropriate method based on algorithm."""
        
        algorithm = self.config.algorithm
        
        if algorithm in ['tabnet']:
            # TabNet with custom training parameters
            fit_params = {
                'eval_set': [(X_val, y_val)],
                'eval_metric': ['logloss'],
                'patience': getattr(self.model, '_tabnet_patience', 15),
                'max_epochs': getattr(self.model, '_tabnet_max_epochs', 100),
                'batch_size': getattr(self.model, '_tabnet_batch_size', 1024),
                'virtual_batch_size': getattr(self.model, '_tabnet_virtual_batch_size', 128),
                'num_workers': getattr(self.model, '_tabnet_num_workers', 0),
            }
            self.model.fit(X_train, y_train, **fit_params)
            
        elif algorithm in ['tf_mlp_multiclass']:
            # TensorFlow model with evaluation set
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=0
            )
            
        else:
            # Standard sklearn-compatible models
            self.model.fit(X_train, y_train)
    
    def _predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions and return both class predictions and probabilities."""
        
        # Get class predictions
        y_pred = self.model.predict(X)
        
        # Get probabilities
        if hasattr(self.model, 'predict_proba'):
            y_prob = self.model.predict_proba(X)
        else:
            # Fallback: create one-hot encoding from predictions
            y_prob = np.eye(3)[y_pred]
        
        return y_pred, y_prob
    
    def _calculate_fold_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                               y_prob: np.ndarray, fold_idx: int, n_test: int) -> Dict[str, Any]:
        """Calculate comprehensive metrics for a single fold."""
        
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        macro_f1 = f1_score(y_true, y_pred, average='macro')
        micro_f1 = f1_score(y_true, y_pred, average='micro')
        
        # Per-class metrics
        precision = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
        
        # ROC/AUC metrics (one-vs-rest)
        roc_auc = {}
        avg_precision = {}
        
        for class_idx in range(3):
            class_name = ['neither', 'donor', 'acceptor'][class_idx]
            y_binary = (y_true == class_idx).astype(int)
            y_prob_class = y_prob[:, class_idx]
            
            if len(np.unique(y_binary)) > 1:  # Both classes present
                fpr, tpr, _ = roc_curve(y_binary, y_prob_class)
                roc_auc[class_name] = auc(fpr, tpr)
                
                precision_curve, recall_curve, _ = precision_recall_curve(y_binary, y_prob_class)
                avg_precision[class_name] = average_precision_score(y_binary, y_prob_class)
            else:
                roc_auc[class_name] = 0.0
                avg_precision[class_name] = 0.0
        
        return {
            'fold': fold_idx,
            'n_test': n_test,
            'accuracy': accuracy,
            'macro_f1': macro_f1,
            'micro_f1': micro_f1,
            'precision_neither': precision[0],
            'precision_donor': precision[1],
            'precision_acceptor': precision[2],
            'recall_neither': recall[0],
            'recall_donor': recall[1],
            'recall_acceptor': recall[2],
            'f1_neither': f1_per_class[0],
            'f1_donor': f1_per_class[1],
            'f1_acceptor': f1_per_class[2],
            'auc_neither': roc_auc['neither'],
            'auc_donor': roc_auc['donor'],
            'auc_acceptor': roc_auc['acceptor'],
            'ap_neither': avg_precision['neither'],
            'ap_donor': avg_precision['donor'],
            'ap_acceptor': avg_precision['acceptor'],
            'confusion_matrix': cm.tolist()
        }
    
    def _generate_results(self) -> Dict[str, Any]:
        """Generate comprehensive results and visualizations."""
        
        # Calculate aggregate metrics
        df_metrics = pd.DataFrame(self.fold_metrics)
        aggregate_metrics = self._calculate_aggregate_metrics(df_metrics)
        
        # Save metrics
        self._save_metrics(df_metrics, aggregate_metrics)
        
        # Generate visualizations
        if self.config.plot_curves:
            self._generate_visualizations()
        
        # Save model
        self._save_model()
        
        # Generate report
        self._generate_report(aggregate_metrics)
        
        return {
            'fold_metrics': self.fold_metrics,
            'aggregate_metrics': aggregate_metrics,
            'cv_results': self.cv_results
        }
    
    def _calculate_aggregate_metrics(self, df_metrics: pd.DataFrame) -> Dict[str, Any]:
        """Calculate aggregate metrics across all folds."""
        
        metrics = {}
        
        # Overall metrics
        for metric in ['accuracy', 'macro_f1', 'micro_f1']:
            if metric in df_metrics.columns:
                metrics[f'{metric}_mean'] = df_metrics[metric].mean()
                metrics[f'{metric}_std'] = df_metrics[metric].std()
        
        # Per-class metrics
        for class_name in ['neither', 'donor', 'acceptor']:
            for metric in ['precision', 'recall', 'f1', 'auc', 'ap']:
                col_name = f'{metric}_{class_name}'
                if col_name in df_metrics.columns:
                    metrics[f'{col_name}_mean'] = df_metrics[col_name].mean()
                    metrics[f'{col_name}_std'] = df_metrics[col_name].std()
        
        return metrics
    
    def _save_metrics(self, df_metrics: pd.DataFrame, aggregate_metrics: Dict[str, Any]) -> None:
        """Save metrics to files."""
        
        # Save fold-level metrics
        df_metrics.to_csv(self.config.output_dir / "fold_metrics.csv", index=False)
        
        # Save aggregate metrics
        with open(self.config.output_dir / "aggregate_metrics.json", 'w') as f:
            json.dump(aggregate_metrics, f, indent=2)
        
        print_with_indent(f"📊 Metrics saved to {self.config.output_dir}")
    
    def _generate_visualizations(self) -> None:
        """Generate ROC/PR curves and other visualizations."""
        
        if not self.cv_results:
            return
        
        y_true = self.cv_results['y_true']
        y_prob = self.cv_results['y_prob']
        
        # Generate multiclass ROC/PR curves
        try:
            plot_multiclass_roc_pr_curves(
                y_true=[y_true],
                y_pred_base=[y_prob],  # Using same data for base comparison
                y_pred_meta=[y_prob],  # This is the actual model
                out_dir=self.config.output_dir,
                plot_format=self.config.plot_format,
                base_name='Reference',
                meta_name=self.config.algorithm.upper()
            )
            print_with_indent("📈 Multiclass ROC/PR curves generated")
        except Exception as e:
            print_with_indent(f"⚠️  Error generating visualizations: {e}")
    
    def _save_model(self) -> None:
        """Save the trained model."""
        
        import pickle
        
        model_path = self.config.output_dir / "trained_model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        
        print_with_indent(f"💾 Model saved to {model_path}")
    
    def _generate_report(self, aggregate_metrics: Dict[str, Any]) -> None:
        """Generate comprehensive text report."""
        
        report_path = self.config.output_dir / "cv_report.txt"
        
        with open(report_path, 'w') as f:
            f.write("🧬 GENE-AWARE CROSS-VALIDATION REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Algorithm: {self.config.algorithm.upper()}\n")
            f.write(f"CV Folds: {self.config.n_folds}\n")
            f.write(f"Dataset: {self.config.dataset_path}\n\n")
            
            f.write("📊 AGGREGATE PERFORMANCE METRICS\n")
            f.write("-" * 30 + "\n")
            
            # Overall metrics
            for metric in ['accuracy', 'macro_f1', 'micro_f1']:
                mean_key = f'{metric}_mean'
                std_key = f'{metric}_std'
                if mean_key in aggregate_metrics:
                    f.write(f"{metric.replace('_', ' ').title()}: "
                           f"{aggregate_metrics[mean_key]:.3f} ± {aggregate_metrics[std_key]:.3f}\n")
            
            f.write("\n📈 PER-CLASS PERFORMANCE\n")
            f.write("-" * 30 + "\n")
            
            for class_name in ['neither', 'donor', 'acceptor']:
                f.write(f"\n{class_name.title()}:\n")
                for metric in ['precision', 'recall', 'f1', 'auc', 'ap']:
                    mean_key = f'{metric}_{class_name}_mean'
                    std_key = f'{metric}_{class_name}_std'
                    if mean_key in aggregate_metrics:
                        f.write(f"  {metric.title()}: "
                               f"{aggregate_metrics[mean_key]:.3f} ± {aggregate_metrics[std_key]:.3f}\n")
        
        print_with_indent(f"📋 Report saved to {report_path}")


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""
    
    parser = argparse.ArgumentParser(
        description="Gene-aware cross-validation for deep learning models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument("--dataset", required=True, help="Path to dataset directory")
    parser.add_argument("--out-dir", required=True, help="Output directory for results")
    parser.add_argument("--algorithm", required=True, 
                       choices=list(AVAILABLE_MODELS.keys()),
                       help="Algorithm to use for training")
    
    # CV configuration
    parser.add_argument("--n-folds", type=int, default=5, help="Number of CV folds")
    parser.add_argument("--valid-size", type=float, default=0.1, help="Validation set size")
    parser.add_argument("--gene-col", default="gene_id", help="Gene column name")
    
    # Data configuration
    parser.add_argument("--max-variants", type=int, help="Maximum variants to process")
    parser.add_argument("--row-cap", type=int, default=100_000, help="Row cap for dataset")
    
    # Model configuration
    parser.add_argument("--algorithm-params", type=str, 
                       help="JSON string of algorithm-specific parameters")
    
    # Evaluation configuration
    parser.add_argument("--plot-curves", action="store_true", default=True,
                       help="Generate ROC/PR curves")
    parser.add_argument("--no-plot-curves", dest="plot_curves", action="store_false",
                       help="Disable curve plotting")
    parser.add_argument("--plot-format", default="pdf", choices=["pdf", "png", "svg"],
                       help="Plot file format")
    
    # Other
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> None:
    """Main entry point."""
    
    # Parse arguments
    args = parse_args(argv)
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Parse algorithm parameters
    algorithm_params = {}
    if args.algorithm_params:
        try:
            algorithm_params = json.loads(args.algorithm_params)
        except json.JSONDecodeError as e:
            print(f"Error parsing algorithm parameters: {e}")
            sys.exit(1)
    
    # Create configuration
    config = DeepLearningCVConfig(
        dataset_path=Path(args.dataset),
        output_dir=Path(args.out_dir),
        gene_col=args.gene_col,
        algorithm=args.algorithm,
        algorithm_params=algorithm_params,
        n_folds=args.n_folds,
        valid_size=args.valid_size,
        random_seed=args.seed,
        max_variants=args.max_variants,
        row_cap=args.row_cap,
        plot_curves=args.plot_curves,
        plot_format=args.plot_format
    )
    
    # Run CV
    cv_runner = DeepLearningGeneCV(config)
    results = cv_runner.run_cross_validation()
    
    print_emphasized("🎉 Gene-aware CV completed successfully!")
    print_emphasized(f"📁 Results saved to: {config.output_dir}")


if __name__ == "__main__":
    main()
