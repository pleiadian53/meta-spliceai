"""
Overfitting monitoring utilities for XGBoost CV training.

This module provides comprehensive tools to detect and visualize overfitting
during cross-validation training, including learning curves, performance gaps,
and early stopping analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import json
from dataclasses import dataclass
from sklearn.metrics import log_loss, roc_auc_score, f1_score, accuracy_score
import warnings


@dataclass
class OverfittingMetrics:
    """Container for overfitting detection metrics."""
    fold_idx: int
    training_history: Dict[str, List[float]]
    validation_history: Dict[str, List[float]]
    best_iteration: int
    early_stopped: bool
    final_train_score: float
    final_val_score: float
    performance_gap: float
    convergence_iteration: int
    overfitting_score: float


class OverfittingMonitor:
    """Monitor and detect overfitting during XGBoost training."""
    
    def __init__(self, 
                 primary_metric: str = "logloss",
                 gap_threshold: float = 0.05,
                 patience: int = 10,
                 min_improvement: float = 0.001):
        """
        Initialize overfitting monitor.
        
        Parameters
        ----------
        primary_metric : str
            Primary metric to monitor ("logloss", "auc", "f1", "accuracy")
        gap_threshold : float
            Threshold for performance gap to flag overfitting
        patience : int
            Number of iterations to wait for improvement
        min_improvement : float
            Minimum improvement threshold for convergence detection
        """
        self.primary_metric = primary_metric
        self.gap_threshold = gap_threshold
        self.patience = patience
        self.min_improvement = min_improvement
        self.fold_metrics: List[OverfittingMetrics] = []
        self.training_callbacks = []
        
    def create_xgb_callback(self, fold_idx: int) -> callable:
        """Create XGBoost callback function for monitoring."""
        
        def monitor_callback(env):
            """XGBoost callback function to track training progress."""
            # Get training and validation metrics
            train_scores = env.evaluation_result_list
            
            # Store the scores (this is a simplified version)
            # In practice, you'd extract specific metrics from the evaluation results
            iteration = env.iteration
            
            # This callback would be called after each boosting iteration
            # We can access training and validation metrics here
            
            return False  # Continue training
        
        return monitor_callback
    
    def analyze_training_history(self, 
                                evals_result: Dict[str, Dict[str, List[float]]],
                                fold_idx: int) -> OverfittingMetrics:
        """
        Analyze training history for overfitting patterns.
        
        Parameters
        ----------
        evals_result : dict
            XGBoost evaluation results containing training history
        fold_idx : int
            Current fold index
            
        Returns
        -------
        OverfittingMetrics
            Comprehensive overfitting analysis results
        """
        # Extract training and validation histories
        train_key = 'train' if 'train' in evals_result else list(evals_result.keys())[0]
        val_key = 'eval' if 'eval' in evals_result else 'validation'
        
        if val_key not in evals_result:
            # Find validation set key
            val_key = next((k for k in evals_result.keys() if k != train_key), None)
        
        if val_key is None:
            raise ValueError("No validation set found in evaluation results")
        
        train_hist = evals_result[train_key]
        val_hist = evals_result[val_key]
        
        # Get primary metric history
        metric_key = self.primary_metric
        if metric_key not in train_hist:
            # Try common alternatives
            metric_alternatives = {
                'logloss': ['logloss', 'log_loss', 'mlogloss'],
                'auc': ['auc', 'roc_auc'],
                'f1': ['f1', 'f1_score'],
                'accuracy': ['accuracy', 'acc']
            }
            
            for alt in metric_alternatives.get(metric_key, [metric_key]):
                if alt in train_hist:
                    metric_key = alt
                    break
        
        if metric_key not in train_hist:
            raise ValueError(f"Metric '{self.primary_metric}' not found in training history")
        
        train_scores = train_hist[metric_key]
        val_scores = val_hist[metric_key]
        
        # Find best iteration (lowest for loss metrics, highest for performance metrics)
        is_loss_metric = metric_key in ['logloss', 'log_loss', 'mlogloss']
        if is_loss_metric:
            best_iteration = np.argmin(val_scores)
        else:
            best_iteration = np.argmax(val_scores)
        
        # Calculate performance gap
        final_train = train_scores[-1]
        final_val = val_scores[-1]
        
        if is_loss_metric:
            performance_gap = final_val - final_train  # Positive gap indicates overfitting
        else:
            performance_gap = final_train - final_val  # Positive gap indicates overfitting
        
        # Detect convergence point
        convergence_iter = self._detect_convergence(val_scores, is_loss_metric)
        
        # Calculate overfitting score (composite metric)
        overfitting_score = self._calculate_overfitting_score(
            train_scores, val_scores, is_loss_metric
        )
        
        # Check if early stopping would have triggered
        early_stopped = self._would_early_stop(val_scores, is_loss_metric)
        
        return OverfittingMetrics(
            fold_idx=fold_idx,
            training_history=train_hist,
            validation_history=val_hist,
            best_iteration=best_iteration,
            early_stopped=early_stopped,
            final_train_score=final_train,
            final_val_score=final_val,
            performance_gap=performance_gap,
            convergence_iteration=convergence_iter,
            overfitting_score=overfitting_score
        )
    
    def _detect_convergence(self, scores: List[float], is_loss_metric: bool) -> int:
        """Detect convergence point in validation scores."""
        if len(scores) < self.patience:
            return len(scores) - 1
        
        for i in range(self.patience, len(scores)):
            recent_window = scores[i-self.patience:i]
            current_score = scores[i]
            
            if is_loss_metric:
                # For loss metrics, check if we're not improving (decreasing)
                if min(recent_window) - current_score < self.min_improvement:
                    return i
            else:
                # For performance metrics, check if we're not improving (increasing)
                if current_score - max(recent_window) < self.min_improvement:
                    return i
        
        return len(scores) - 1
    
    def _would_early_stop(self, scores: List[float], is_loss_metric: bool) -> bool:
        """Check if early stopping would have triggered."""
        if len(scores) < self.patience:
            return False
        
        # Find the best score up to each point
        best_scores = []
        for i in range(len(scores)):
            if is_loss_metric:
                best_scores.append(min(scores[:i+1]))
            else:
                best_scores.append(max(scores[:i+1]))
        
        # Check if we went patience iterations without improvement
        for i in range(self.patience, len(scores)):
            if is_loss_metric:
                if all(scores[j] >= best_scores[i-self.patience] for j in range(i-self.patience+1, i+1)):
                    return True
            else:
                if all(scores[j] <= best_scores[i-self.patience] for j in range(i-self.patience+1, i+1)):
                    return True
        
        return False
    
    def _calculate_overfitting_score(self, 
                                   train_scores: List[float], 
                                   val_scores: List[float], 
                                   is_loss_metric: bool) -> float:
        """Calculate composite overfitting score."""
        if len(train_scores) != len(val_scores):
            return 0.0
        
        # Calculate area between curves (proxy for overfitting)
        if is_loss_metric:
            gaps = [val - train for train, val in zip(train_scores, val_scores)]
        else:
            gaps = [train - val for train, val in zip(train_scores, val_scores)]
        
        # Normalize by number of iterations
        overfitting_score = np.mean([max(0, gap) for gap in gaps])
        
        # Add penalty for large final gap
        final_gap = gaps[-1] if gaps else 0
        if final_gap > self.gap_threshold:
            overfitting_score += final_gap * 2  # Penalty multiplier
        
        return overfitting_score
    
    def add_fold_metrics(self, evals_result: Dict[str, Dict[str, List[float]]], fold_idx: int):
        """Add metrics for a completed fold."""
        metrics = self.analyze_training_history(evals_result, fold_idx)
        self.fold_metrics.append(metrics)
    
    def _convert_numpy_types(self, obj):
        """Convert numpy types to JSON-serializable Python types."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        else:
            return obj
    
    def generate_overfitting_report(self, out_dir: Union[str, Path]) -> Dict[str, Any]:
        """Generate comprehensive overfitting analysis report."""
        out_dir = Path(out_dir)
        
        if not self.fold_metrics:
            raise ValueError("No fold metrics available. Run CV training first.")
        
        # Aggregate statistics
        performance_gaps = [m.performance_gap for m in self.fold_metrics]
        overfitting_scores = [m.overfitting_score for m in self.fold_metrics]
        best_iterations = [m.best_iteration for m in self.fold_metrics]
        convergence_iters = [m.convergence_iteration for m in self.fold_metrics]
        early_stopped_count = sum(1 for m in self.fold_metrics if m.early_stopped)
        
        # Summary statistics
        report = {
            'summary': {
                'total_folds': len(self.fold_metrics),
                'folds_with_overfitting': sum(1 for gap in performance_gaps if gap > self.gap_threshold),
                'early_stopped_folds': early_stopped_count,
                'mean_performance_gap': np.mean(performance_gaps),
                'std_performance_gap': np.std(performance_gaps),
                'mean_overfitting_score': np.mean(overfitting_scores),
                'mean_best_iteration': np.mean(best_iterations),
                'std_best_iteration': np.std(best_iterations),
                'mean_convergence_iteration': np.mean(convergence_iters),
                'recommended_n_estimators': int(np.mean(best_iterations) + np.std(best_iterations))
            },
            'fold_details': []
        }
        
        # Individual fold details
        for metrics in self.fold_metrics:
            fold_detail = {
                'fold_idx': metrics.fold_idx,
                'performance_gap': metrics.performance_gap,
                'overfitting_score': metrics.overfitting_score,
                'best_iteration': metrics.best_iteration,
                'convergence_iteration': metrics.convergence_iteration,
                'early_stopped': metrics.early_stopped,
                'final_train_score': metrics.final_train_score,
                'final_val_score': metrics.final_val_score,
                'overfitting_detected': metrics.performance_gap > self.gap_threshold
            }
            report['fold_details'].append(fold_detail)
        
        # Convert numpy types to JSON-serializable types
        report = self._convert_numpy_types(report)
        
        # Save report
        report_path = out_dir / "overfitting_analysis.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        return report
    
    def plot_learning_curves(self, out_dir: Union[str, Path], plot_format: str = "pdf"):
        """Create comprehensive learning curves visualization."""
        out_dir = Path(out_dir)
        
        if not self.fold_metrics:
            raise ValueError("No fold metrics available.")
        
        # Create subplot grid
        n_folds = len(self.fold_metrics)
        n_cols = min(3, n_folds)
        n_rows = (n_folds + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        if n_folds == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        
        # Plot individual fold learning curves
        for i, metrics in enumerate(self.fold_metrics):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col] if n_rows > 1 else axes[col]
            
            # Get training and validation scores
            train_scores = metrics.training_history[self.primary_metric]
            val_scores = metrics.validation_history[self.primary_metric]
            # Convert to numeric range for proper plotting
            iterations = list(range(1, len(train_scores) + 1))
            
            # Plot curves
            ax.plot(iterations, train_scores, 'b-', label='Training', linewidth=2)
            ax.plot(iterations, val_scores, 'r-', label='Validation', linewidth=2)
            
            # Mark best iteration (convert to numeric)
            best_iter_numeric = int(metrics.best_iteration) + 1
            ax.axvline(x=best_iter_numeric, color='g', linestyle='--', 
                      label=f'Best iter: {best_iter_numeric}')
            
            # Mark convergence point (convert to numeric)
            conv_iter_numeric = int(metrics.convergence_iteration) + 1
            ax.axvline(x=conv_iter_numeric, color='orange', linestyle=':', 
                      label=f'Convergence: {conv_iter_numeric}')
            
            # Styling
            ax.set_xlabel('Boosting Iterations')
            ax.set_ylabel(self.primary_metric.upper())
            
            # Handle fold_idx for display (keep as string for readability)
            fold_label = str(metrics.fold_idx)
            # For Gene CV, we have string IDs like "0_0" representing fold_class
            # Keep them as-is for better understanding
            
            ax.set_title(f'Model {fold_label}\n'
                        f'Gap: {metrics.performance_gap:.4f}, '
                        f'Overfitting: {metrics.overfitting_score:.4f}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Remove empty subplots
        for i in range(n_folds, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col] if n_rows > 1 else axes[col]
            ax.remove()
        
        plt.tight_layout()
        plt.savefig(out_dir / f"learning_curves_by_fold.{plot_format}", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create aggregated learning curves
        self._plot_aggregated_curves(out_dir, plot_format)
        
        # Create overfitting summary plots
        self._plot_overfitting_summary(out_dir, plot_format)
    
    def _plot_aggregated_curves(self, out_dir: Path, plot_format: str):
        """Plot aggregated learning curves across all folds."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Collect all training and validation histories
        all_train_histories = []
        all_val_histories = []
        max_iterations = 0
        
        for metrics in self.fold_metrics:
            train_scores = metrics.training_history[self.primary_metric]
            val_scores = metrics.validation_history[self.primary_metric]
            all_train_histories.append(train_scores)
            all_val_histories.append(val_scores)
            max_iterations = max(max_iterations, len(train_scores))
        
        # Interpolate to common length (ensure numeric iteration range)
        common_iterations = list(range(1, max_iterations + 1))
        train_matrix = np.full((len(self.fold_metrics), max_iterations), np.nan)
        val_matrix = np.full((len(self.fold_metrics), max_iterations), np.nan)
        
        for i, (train_hist, val_hist) in enumerate(zip(all_train_histories, all_val_histories)):
            train_matrix[i, :len(train_hist)] = train_hist
            val_matrix[i, :len(val_hist)] = val_hist
        
        # Calculate mean and std
        train_mean = np.nanmean(train_matrix, axis=0)
        train_std = np.nanstd(train_matrix, axis=0)
        val_mean = np.nanmean(val_matrix, axis=0)
        val_std = np.nanstd(val_matrix, axis=0)
        
        # Plot mean curves with confidence intervals
        ax1.plot(common_iterations, train_mean, 'b-', label='Training (mean)', linewidth=2)
        ax1.fill_between(common_iterations, train_mean - train_std, train_mean + train_std, 
                        alpha=0.3, color='blue', label='Training (±1 std)')
        
        ax1.plot(common_iterations, val_mean, 'r-', label='Validation (mean)', linewidth=2)
        ax1.fill_between(common_iterations, val_mean - val_std, val_mean + val_std, 
                        alpha=0.3, color='red', label='Validation (±1 std)')
        
        # Mark average best iteration (ensure numeric)
        avg_best_iter = float(np.mean([int(m.best_iteration) for m in self.fold_metrics]))
        ax1.axvline(x=avg_best_iter + 1, color='g', linestyle='--', 
                   label=f'Avg best iter: {avg_best_iter + 1:.0f}')
        
        ax1.set_xlabel('Boosting Iterations')
        ax1.set_ylabel(self.primary_metric.upper())
        ax1.set_title('Aggregated Learning Curves')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot individual fold curves (lighter)
        for i, metrics in enumerate(self.fold_metrics):
            train_scores = metrics.training_history[self.primary_metric]
            val_scores = metrics.validation_history[self.primary_metric]
            # Convert to list for proper matplotlib handling
            iterations = list(range(1, len(train_scores) + 1))
            
            ax2.plot(iterations, train_scores, 'b-', alpha=0.3, linewidth=1)
            ax2.plot(iterations, val_scores, 'r-', alpha=0.3, linewidth=1)
        
        # Overlay mean curves
        ax2.plot(common_iterations, train_mean, 'b-', label='Training (mean)', linewidth=2)
        ax2.plot(common_iterations, val_mean, 'r-', label='Validation (mean)', linewidth=2)
        
        ax2.set_xlabel('Boosting Iterations')
        ax2.set_ylabel(self.primary_metric.upper())
        ax2.set_title('All Folds + Mean Curves')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(out_dir / f"aggregated_learning_curves.{plot_format}", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_overfitting_summary(self, out_dir: Path, plot_format: str):
        """Create overfitting summary visualizations."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Extract metrics
        performance_gaps = [m.performance_gap for m in self.fold_metrics]
        overfitting_scores = [m.overfitting_score for m in self.fold_metrics]
        best_iterations = [m.best_iteration for m in self.fold_metrics]
        convergence_iters = [m.convergence_iteration for m in self.fold_metrics]
        fold_indices_raw = [m.fold_idx for m in self.fold_metrics]
        
        # Convert fold indices to numeric for proper plotting
        # Handle cases where fold_idx might be strings like "0_0", "0_1", etc.
        fold_indices_numeric = []
        fold_labels = []
        for i, fold_idx in enumerate(fold_indices_raw):
            if isinstance(fold_idx, str):
                # Use sequential numbering for mixed string identifiers
                fold_indices_numeric.append(i)
                fold_labels.append(fold_idx)
            else:
                # Use the numeric value directly
                fold_indices_numeric.append(int(fold_idx))
                fold_labels.append(str(fold_idx))
        
        # 1. Performance gap across folds
        bars = ax1.bar(fold_indices_numeric, performance_gaps, alpha=0.7, color='red')
        ax1.axhline(y=self.gap_threshold, color='orange', linestyle='--', 
                   label=f'Threshold: {self.gap_threshold}')
        ax1.set_xlabel('Model Instance')
        ax1.set_ylabel('Performance Gap')
        ax1.set_title('Training-Validation Performance Gap')
        ax1.set_xticks(fold_indices_numeric)
        ax1.set_xticklabels(fold_labels, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Overfitting score distribution
        ax2.hist(overfitting_scores, bins=min(10, len(overfitting_scores)), 
                alpha=0.7, color='purple')
        ax2.axvline(x=np.mean(overfitting_scores), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(overfitting_scores):.4f}')
        ax2.set_xlabel('Overfitting Score')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Overfitting Score Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Best iteration consistency
        ax3.scatter(fold_indices_numeric, best_iterations, s=100, alpha=0.7, color='green')
        ax3.axhline(y=np.mean(best_iterations), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(best_iterations):.1f}')
        ax3.set_xlabel('Model Instance')
        ax3.set_ylabel('Best Iteration')
        ax3.set_title('Best Iteration Across Folds')
        ax3.set_xticks(fold_indices_numeric)
        ax3.set_xticklabels(fold_labels, rotation=45)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Convergence analysis
        ax4.plot(fold_indices_numeric, best_iterations, 'o-', label='Best Iteration', linewidth=2)
        ax4.plot(fold_indices_numeric, convergence_iters, 's-', label='Convergence Point', linewidth=2)
        ax4.set_xlabel('Model Instance')
        ax4.set_ylabel('Iteration Number')
        ax4.set_title('Convergence Analysis')
        ax4.set_xticks(fold_indices_numeric)
        ax4.set_xticklabels(fold_labels, rotation=45)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(out_dir / f"overfitting_summary.{plot_format}", dpi=300, bbox_inches='tight')
        plt.close()

def create_enhanced_xgb_model(args, monitor: OverfittingMonitor = None):
    """Create XGBoost model with enhanced overfitting monitoring."""
    from xgboost import XGBClassifier
    
    # Standard model configuration
    model = XGBClassifier(
        n_estimators=args.n_estimators,
        tree_method=args.tree_method,
        max_bin=args.max_bin if args.tree_method in {"hist", "gpu_hist"} else None,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=args.seed,
        n_jobs=-1,
        device=args.device if args.device != "auto" else None,
    )
    
    # Add monitoring callback if monitor is provided
    if monitor is not None:
        # Note: This would require custom integration with XGBoost callbacks
        # For now, we'll rely on evals_result parameter in fit()
        pass
    
    return model

def enhanced_model_training(X_train, y_train, X_val, y_val, args, monitor: OverfittingMonitor, fold_idx: int):
    """Enhanced model training with comprehensive overfitting monitoring."""
    model = create_enhanced_xgb_model(args, monitor)
    
    # Fit model with evaluation tracking
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        eval_names=['train', 'eval'],
        verbose=False
    )
    
    # Get evaluation results
    if hasattr(model, 'evals_result'):
        evals_result = model.evals_result()
    else:
        # Fallback: create dummy evaluation results
        evals_result = {
            'train': {'logloss': []},
            'eval': {'logloss': []}
        }
    
    # Add fold metrics to monitor
    monitor.add_fold_metrics(evals_result, fold_idx)
    
    return model 