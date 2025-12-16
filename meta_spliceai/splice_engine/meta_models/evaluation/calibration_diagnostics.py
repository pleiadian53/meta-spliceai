"""
Calibration diagnostics and correction methods for meta-models.

This module provides tools to detect and correct problematic calibration in meta-models,
particularly addressing overconfidence issues that arise from biased training data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import json
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import brier_score_loss, log_loss
import warnings

def diagnose_calibration_issues(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    y_pred_class: np.ndarray,
    out_dir: Path,
    model_name: str = "meta",
    plot_format: str = "png",
    verbose: bool = True
) -> Dict[str, float]:
    """
    Comprehensive calibration diagnostics for meta-model predictions.
    
    Parameters
    ----------
    y_true : np.ndarray
        True binary labels (0/1)
    y_prob : np.ndarray  
        Predicted probabilities
    y_pred_class : np.ndarray
        Predicted class labels
    out_dir : Path
        Output directory for plots and reports
    model_name : str
        Name of the model for labeling
    plot_format : str
        Format for plots ('png', 'pdf', 'svg')
    verbose : bool
        Print diagnostic information
        
    Returns
    -------
    Dict[str, float]
        Dictionary of calibration metrics
    """
    
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Basic calibration metrics
    brier_score = brier_score_loss(y_true, y_prob)
    log_loss_score = log_loss(y_true, y_prob)
    
    # Reliability diagram (calibration curve)
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=20, strategy='uniform')
    
    # Expected Calibration Error (ECE)
    ece = np.mean(np.abs(prob_true - prob_pred))
    
    # Maximum Calibration Error (MCE)  
    mce = np.max(np.abs(prob_true - prob_pred))
    
    # Overconfidence detection
    # Check for high-confidence wrong predictions
    high_conf_mask = y_prob > 0.9
    high_conf_errors = np.sum((y_pred_class != y_true) & high_conf_mask) if high_conf_mask.any() else 0
    high_conf_total = np.sum(high_conf_mask)
    overconf_rate = high_conf_errors / high_conf_total if high_conf_total > 0 else 0.0
    
    # Probability distribution analysis
    prob_entropy = -np.mean(y_prob * np.log(y_prob + 1e-15) + (1-y_prob) * np.log(1-y_prob + 1e-15))
    prob_mean = np.mean(y_prob)
    prob_std = np.std(y_prob)
    
    # Detect probability concentration near extremes
    extreme_low = np.sum(y_prob < 0.01) / len(y_prob)
    extreme_high = np.sum(y_prob > 0.99) / len(y_prob)
    extreme_total = extreme_low + extreme_high
    
    metrics = {
        'brier_score': float(brier_score),
        'log_loss': float(log_loss_score),
        'expected_calibration_error': float(ece),
        'max_calibration_error': float(mce),
        'overconfidence_rate': float(overconf_rate),
        'high_conf_errors': int(high_conf_errors),
        'high_conf_total': int(high_conf_total),
        'prob_entropy': float(prob_entropy),
        'prob_mean': float(prob_mean),
        'prob_std': float(prob_std),
        'extreme_low_rate': float(extreme_low),
        'extreme_high_rate': float(extreme_high),
        'extreme_total_rate': float(extreme_total),
        'n_samples': len(y_true)
    }
    
    if verbose:
        print(f"\n📊 CALIBRATION DIAGNOSTICS FOR {model_name.upper()} MODEL:")
        print("-" * 50)
        print(f"  Brier Score: {brier_score:.4f}")
        print(f"  Log Loss: {log_loss_score:.4f}")
        print(f"  Expected Calibration Error: {ece:.4f}")
        print(f"  Max Calibration Error: {mce:.4f}")
        print(f"  Overconfidence Rate: {overconf_rate:.1%} ({high_conf_errors}/{high_conf_total} high-conf errors)")
        print(f"  Probability Entropy: {prob_entropy:.4f}")
        print(f"  Extreme Probabilities: {extreme_total:.1%} (low: {extreme_low:.1%}, high: {extreme_high:.1%})")
        
        # Calibration quality assessment
        if ece > 0.1:
            print("  ⚠️  HIGH CALIBRATION ERROR detected!")
        elif ece > 0.05:
            print("  ⚠️  Moderate calibration error detected")
        else:
            print("  ✅ Good calibration quality")
            
        if extreme_total > 0.5:
            print("  ⚠️  EXTREME OVERCONFIDENCE detected! (>50% predictions near 0 or 1)")
        elif extreme_total > 0.3:
            print("  ⚠️  High overconfidence detected (>30% predictions near extremes)")
            
        if overconf_rate > 0.1:
            print("  ⚠️  HIGH OVERCONFIDENCE RATE detected! Many high-confidence errors")
    
    # Create calibration plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Reliability diagram
    ax = axes[0, 0]
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect calibration')
    ax.plot(prob_pred, prob_true, 'o-', alpha=0.8, label=f'{model_name} model')
    ax.set_xlabel('Mean Predicted Probability')
    ax.set_ylabel('Fraction of Positives')
    ax.set_title(f'Reliability Diagram\nECE = {ece:.4f}, MCE = {mce:.4f}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Probability histogram
    ax = axes[0, 1]
    ax.hist(y_prob, bins=50, alpha=0.7, density=True, color='skyblue', edgecolor='black')
    ax.axvline(prob_mean, color='red', linestyle='--', label=f'Mean = {prob_mean:.3f}')
    ax.set_xlabel('Predicted Probability')
    ax.set_ylabel('Density')
    ax.set_title(f'Probability Distribution\nEntropy = {prob_entropy:.3f}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Confidence vs Accuracy
    ax = axes[1, 0]
    # Bin predictions by confidence
    conf_bins = np.linspace(0, 1, 11)
    bin_accs = []
    bin_confs = []
    bin_counts = []
    
    for i in range(len(conf_bins)-1):
        mask = (y_prob >= conf_bins[i]) & (y_prob < conf_bins[i+1])
        if mask.sum() > 0:
            bin_acc = np.mean(y_true[mask])
            bin_conf = np.mean(y_prob[mask])
            bin_accs.append(bin_acc)
            bin_confs.append(bin_conf)
            bin_counts.append(mask.sum())
        else:
            bin_accs.append(np.nan)
            bin_confs.append(np.nan)
            bin_counts.append(0)
    
    # Plot with point sizes proportional to sample count
    sizes = np.array(bin_counts) * 100 / max(bin_counts) if max(bin_counts) > 0 else np.ones(len(bin_counts))
    scatter = ax.scatter(bin_confs, bin_accs, s=sizes, alpha=0.7, c=bin_counts, cmap='viridis')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax.set_xlabel('Mean Confidence (Predicted Probability)')
    ax.set_ylabel('Accuracy (Fraction of Positives)')
    ax.set_title('Confidence vs Accuracy\n(point size ∝ sample count)')
    plt.colorbar(scatter, ax=ax, label='Sample Count')
    ax.grid(True, alpha=0.3)
    
    # Error analysis by confidence
    ax = axes[1, 1]
    conf_ranges = ['0.0-0.5', '0.5-0.7', '0.7-0.9', '0.9-0.95', '0.95-0.99', '0.99-1.0']
    conf_masks = [
        y_prob < 0.5,
        (y_prob >= 0.5) & (y_prob < 0.7),
        (y_prob >= 0.7) & (y_prob < 0.9),
        (y_prob >= 0.9) & (y_prob < 0.95),
        (y_prob >= 0.95) & (y_prob < 0.99),
        y_prob >= 0.99
    ]
    
    error_rates = []
    sample_counts = []
    for mask in conf_masks:
        if mask.sum() > 0:
            error_rate = np.mean(y_pred_class[mask] != y_true[mask])
            error_rates.append(error_rate)
            sample_counts.append(mask.sum())
        else:
            error_rates.append(0)
            sample_counts.append(0)
    
    bars = ax.bar(conf_ranges, error_rates, alpha=0.7, color='orange')
    ax.set_xlabel('Confidence Range')
    ax.set_ylabel('Error Rate')
    ax.set_title('Error Rate by Confidence Level')
    ax.tick_params(axis='x', rotation=45)
    
    # Add sample counts as text on bars
    for bar, count in zip(bars, sample_counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'n={count}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plot_path = out_dir / f"{model_name}_calibration_diagnostics.{plot_format}"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save metrics to JSON
    metrics_path = out_dir / f"{model_name}_calibration_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    if verbose:
        print(f"  📁 Calibration diagnostics saved:")
        print(f"    Plot: {plot_path}")
        print(f"    Metrics: {metrics_path}")
    
    return metrics


def temperature_scaling_calibration(
    y_true: np.ndarray,
    logits: np.ndarray,
    validation_split: float = 0.2,
    random_state: int = 42
) -> Tuple[float, np.ndarray]:
    """
    Apply temperature scaling for calibration.
    
    Temperature scaling is simpler and often more robust than Platt scaling.
    
    Parameters
    ----------
    y_true : np.ndarray
        True binary labels
    logits : np.ndarray
        Raw logits (before softmax)
    validation_split : float
        Fraction of data to use for temperature fitting
    random_state : int
        Random seed
        
    Returns
    -------
    Tuple[float, np.ndarray]
        Optimal temperature and calibrated probabilities
    """
    from scipy.optimize import minimize_scalar
    from sklearn.model_selection import train_test_split
    
    # Split data for temperature fitting
    logits_train, logits_val, y_train, y_val = train_test_split(
        logits, y_true, test_size=validation_split, random_state=random_state, stratify=y_true
    )
    
    def temperature_loss(T):
        """Negative log-likelihood with temperature scaling."""
        scaled_logits = logits_val / T
        probs = 1 / (1 + np.exp(-scaled_logits))  # sigmoid
        probs = np.clip(probs, 1e-15, 1-1e-15)  # avoid log(0)
        return -np.mean(y_val * np.log(probs) + (1-y_val) * np.log(1-probs))
    
    # Find optimal temperature
    result = minimize_scalar(temperature_loss, bounds=(0.1, 10.0), method='bounded')
    optimal_temp = result.x
    
    # Apply temperature scaling to all data
    calibrated_probs = 1 / (1 + np.exp(-logits / optimal_temp))
    
    return optimal_temp, calibrated_probs


def entropy_based_confidence_correction(
    y_prob: np.ndarray,
    correction_strength: float = 0.1,
    min_entropy_threshold: float = 0.1
) -> np.ndarray:
    """
    Apply entropy-based confidence correction to reduce overconfidence.
    
    Parameters
    ----------
    y_prob : np.ndarray
        Original probabilities
    correction_strength : float
        Strength of correction (0 = no correction, 1 = maximum correction)
    min_entropy_threshold : float
        Minimum entropy threshold below which correction is applied
        
    Returns
    -------
    np.ndarray
        Corrected probabilities
    """
    
    # Calculate entropy for each prediction
    entropy = -(y_prob * np.log(y_prob + 1e-15) + (1-y_prob) * np.log(1-y_prob + 1e-15))
    
    # Apply correction to low-entropy (overconfident) predictions
    correction_mask = entropy < min_entropy_threshold
    
    corrected_probs = y_prob.copy()
    
    if correction_mask.any():
        # Move probabilities toward 0.5 (maximum entropy)
        correction = correction_strength * (0.5 - y_prob[correction_mask])
        corrected_probs[correction_mask] += correction
        
        # Ensure probabilities remain in [0, 1]
        corrected_probs = np.clip(corrected_probs, 1e-15, 1-1e-15)
    
    return corrected_probs


def bias_aware_calibration(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    bias_indicators: Optional[np.ndarray] = None,
    method: str = "stratified_isotonic"
) -> np.ndarray:
    """
    Calibration method that accounts for selection bias in training data.
    
    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_prob : np.ndarray
        Predicted probabilities
    bias_indicators : np.ndarray, optional
        Indicators of bias (e.g., gene difficulty, error propensity)
    method : str
        Calibration method ('stratified_isotonic', 'weighted_platt')
        
    Returns
    -------
    np.ndarray
        Bias-aware calibrated probabilities
    """
    
    if bias_indicators is None:
        # Fallback to standard isotonic regression
        calibrator = IsotonicRegression(out_of_bounds='clip')
        return calibrator.fit_transform(y_prob, y_true)
    
    if method == "stratified_isotonic":
        # Stratify by bias level and calibrate separately
        bias_quantiles = np.quantile(bias_indicators, [0.33, 0.67])
        low_bias = bias_indicators <= bias_quantiles[0]
        mid_bias = (bias_indicators > bias_quantiles[0]) & (bias_indicators <= bias_quantiles[1])
        high_bias = bias_indicators > bias_quantiles[1]
        
        calibrated_probs = y_prob.copy()
        
        for mask, name in [(low_bias, "low"), (mid_bias, "mid"), (high_bias, "high")]:
            if mask.sum() > 10:  # Ensure sufficient data
                calibrator = IsotonicRegression(out_of_bounds='clip')
                calibrated_probs[mask] = calibrator.fit_transform(y_prob[mask], y_true[mask])
        
        return calibrated_probs
    
    elif method == "weighted_platt":
        # Weight samples inversely to their bias level
        weights = 1.0 / (bias_indicators + 1e-6)  # Inverse weighting
        weights = weights / weights.sum() * len(weights)  # Normalize
        
        calibrator = LogisticRegression()
        calibrator.fit(y_prob.reshape(-1, 1), y_true, sample_weight=weights)
        return calibrator.predict_proba(y_prob.reshape(-1, 1))[:, 1]
    
    else:
        raise ValueError(f"Unknown bias-aware calibration method: {method}")


def evaluate_calibration_methods(
    y_true: np.ndarray,
    y_prob_original: np.ndarray,
    logits: Optional[np.ndarray] = None,
    out_dir: Path,
    plot_format: str = "png",
    verbose: bool = True
) -> Dict[str, Dict[str, float]]:
    """
    Compare multiple calibration methods and recommend the best one.
    
    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_prob_original : np.ndarray
        Original (uncalibrated) probabilities
    logits : np.ndarray, optional
        Raw logits for temperature scaling
    out_dir : Path
        Output directory
    plot_format : str
        Plot format
    verbose : bool
        Verbose output
        
    Returns
    -------
    Dict[str, Dict[str, float]]
        Calibration metrics for each method
    """
    
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    methods = {}
    
    # Original (uncalibrated)
    methods['original'] = y_prob_original
    
    # Isotonic regression
    iso_reg = IsotonicRegression(out_of_bounds='clip')
    methods['isotonic'] = iso_reg.fit_transform(y_prob_original, y_true)
    
    # Platt scaling
    platt = LogisticRegression()
    platt.fit(y_prob_original.reshape(-1, 1), y_true)
    methods['platt'] = platt.predict_proba(y_prob_original.reshape(-1, 1))[:, 1]
    
    # Temperature scaling (if logits available)
    if logits is not None:
        try:
            temp, temp_probs = temperature_scaling_calibration(y_true, logits)
            methods['temperature'] = temp_probs
            if verbose:
                print(f"Optimal temperature: {temp:.3f}")
        except Exception as e:
            if verbose:
                print(f"Temperature scaling failed: {e}")
    
    # Entropy-based correction
    methods['entropy_corrected'] = entropy_based_confidence_correction(y_prob_original)
    
    # Evaluate each method
    results = {}
    for name, probs in methods.items():
        metrics = diagnose_calibration_issues(
            y_true, probs, (probs > 0.5).astype(int),
            out_dir, model_name=f"calibrated_{name}", 
            plot_format=plot_format, verbose=False
        )
        results[name] = metrics
    
    # Create comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Reliability diagrams
    ax = axes[0, 0]
    for name, probs in methods.items():
        if name == 'original':
            continue
        prob_true, prob_pred = calibration_curve(y_true, probs, n_bins=10)
        ax.plot(prob_pred, prob_true, 'o-', alpha=0.8, label=name)
    
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect calibration')
    ax.set_xlabel('Mean Predicted Probability')
    ax.set_ylabel('Fraction of Positives')
    ax.set_title('Calibration Methods Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # ECE comparison
    ax = axes[0, 1]
    eces = [results[name]['expected_calibration_error'] for name in methods.keys()]
    bars = ax.bar(list(methods.keys()), eces, alpha=0.7)
    ax.set_ylabel('Expected Calibration Error')
    ax.set_title('ECE Comparison (lower is better)')
    ax.tick_params(axis='x', rotation=45)
    
    # Color bars by performance
    for bar, ece in zip(bars, eces):
        if ece < 0.05:
            bar.set_color('green')
        elif ece < 0.1:
            bar.set_color('orange')
        else:
            bar.set_color('red')
    
    # Brier score comparison
    ax = axes[1, 0]
    brier_scores = [results[name]['brier_score'] for name in methods.keys()]
    bars = ax.bar(list(methods.keys()), brier_scores, alpha=0.7)
    ax.set_ylabel('Brier Score')
    ax.set_title('Brier Score Comparison (lower is better)')
    ax.tick_params(axis='x', rotation=45)
    
    # Overconfidence rate comparison
    ax = axes[1, 1]
    overconf_rates = [results[name]['overconfidence_rate'] for name in methods.keys()]
    bars = ax.bar(list(methods.keys()), overconf_rates, alpha=0.7)
    ax.set_ylabel('Overconfidence Rate')
    ax.set_title('Overconfidence Rate (lower is better)')
    ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plot_path = out_dir / f"calibration_methods_comparison.{plot_format}"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Recommend best method
    best_method = min(methods.keys(), key=lambda x: results[x]['expected_calibration_error'])
    
    if verbose:
        print(f"\n📊 CALIBRATION METHODS COMPARISON:")
        print("-" * 50)
        for name in methods.keys():
            metrics = results[name]
            marker = "🏆" if name == best_method else "  "
            print(f"{marker} {name:15} ECE={metrics['expected_calibration_error']:.4f} "
                  f"Brier={metrics['brier_score']:.4f} "
                  f"OverConf={metrics['overconfidence_rate']:.1%}")
        
        print(f"\n🏆 RECOMMENDED METHOD: {best_method}")
        print(f"  Comparison plot: {plot_path}")
    
    # Save detailed results
    results_path = out_dir / "calibration_methods_comparison.json"
    with open(results_path, 'w') as f:
        json.dump({
            'results': results,
            'recommended_method': best_method,
            'summary': {
                'best_ece': results[best_method]['expected_calibration_error'],
                'best_brier': results[best_method]['brier_score'],
                'best_overconf_rate': results[best_method]['overconfidence_rate']
            }
        }, f, indent=2)
    
    return results


def detect_selection_bias_effects(
    dataset_path: Union[str, Path],
    model_predictions: np.ndarray,
    true_labels: np.ndarray,
    out_dir: Path,
    verbose: bool = True
) -> Dict[str, float]:
    """
    Detect effects of selection bias in the training dataset.
    
    Parameters
    ----------
    dataset_path : str or Path
        Path to the original dataset
    model_predictions : np.ndarray
        Model probability predictions
    true_labels : np.ndarray
        True labels
    out_dir : Path
        Output directory
    verbose : bool
        Verbose output
        
    Returns
    -------
    Dict[str, float]
        Selection bias metrics
    """
    
    try:
        # Load dataset metadata if available
        import polars as pl
        if Path(dataset_path).is_dir():
            df = pl.scan_parquet(str(Path(dataset_path) / "*.parquet")).collect().to_pandas()
        else:
            df = pd.read_parquet(dataset_path)
        
        # Analyze gene-level statistics
        if 'gene_id' in df.columns:
            gene_stats = df.groupby('gene_id').agg({
                'donor_score': ['mean', 'std', 'min', 'max'],
                'acceptor_score': ['mean', 'std', 'min', 'max'],
                'splice_type': lambda x: (x != 0).sum() / len(x)  # splice site density
            }).round(4)
            
            # Flatten column names
            gene_stats.columns = ['_'.join(col).strip() for col in gene_stats.columns.values]
            
            # Calculate bias indicators
            bias_metrics = {
                'gene_count': len(gene_stats),
                'mean_splice_density': gene_stats['splice_type_<lambda>'].mean(),
                'std_splice_density': gene_stats['splice_type_<lambda>'].std(),
                'high_density_genes': (gene_stats['splice_type_<lambda>'] > 0.1).sum(),
                'low_density_genes': (gene_stats['splice_type_<lambda>'] < 0.01).sum(),
            }
            
            # Check for systematic patterns
            if 'donor_score_mean' in gene_stats.columns:
                bias_metrics.update({
                    'mean_donor_score': gene_stats['donor_score_mean'].mean(),
                    'mean_acceptor_score': gene_stats['acceptor_score_mean'].mean(),
                    'high_score_genes': ((gene_stats['donor_score_mean'] + gene_stats['acceptor_score_mean']) > 0.5).sum(),
                })
            
            if verbose:
                print(f"\n🔍 SELECTION BIAS ANALYSIS:")
                print("-" * 50)
                print(f"  Dataset: {dataset_path}")
                print(f"  Genes analyzed: {bias_metrics['gene_count']}")
                print(f"  Mean splice density: {bias_metrics['mean_splice_density']:.3f}")
                print(f"  High-density genes (>10% splice sites): {bias_metrics['high_density_genes']}")
                print(f"  Low-density genes (<1% splice sites): {bias_metrics['low_density_genes']}")
                
                if bias_metrics['mean_splice_density'] > 0.05:
                    print("  ⚠️  HIGH SPLICE DENSITY detected - possible bias toward error-prone genes")
                if bias_metrics['high_density_genes'] / bias_metrics['gene_count'] > 0.3:
                    print("  ⚠️  Many high-density genes - training set may be biased")
            
            # Save bias analysis
            bias_path = out_dir / "selection_bias_analysis.json"
            with open(bias_path, 'w') as f:
                json.dump(bias_metrics, f, indent=2)
            
            return bias_metrics
        
    except Exception as e:
        if verbose:
            print(f"Selection bias analysis failed: {e}")
        return {}
    
    return {}


def generate_calibration_report(
    dataset_path: Union[str, Path],
    model_path: Union[str, Path],
    out_dir: Union[str, Path],
    sample_size: Optional[int] = None,
    plot_format: str = "png",
    verbose: bool = True
) -> Dict[str, any]:
    """
    Generate comprehensive calibration analysis report.
    
    Parameters
    ----------
    dataset_path : str or Path
        Path to dataset
    model_path : str or Path
        Path to trained model
    out_dir : str or Path
        Output directory
    sample_size : int, optional
        Sample size for analysis
    plot_format : str
        Plot format
    verbose : bool
        Verbose output
        
    Returns
    -------
    Dict[str, any]
        Comprehensive calibration report
    """
    
    from pathlib import Path
    import pickle
    
    out_dir = Path(out_dir)
    calib_dir = out_dir / "calibration_analysis"
    calib_dir.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        print(f"\n🔧 GENERATING COMPREHENSIVE CALIBRATION REPORT")
        print(f"Dataset: {dataset_path}")
        print(f"Model: {model_path}")
        print(f"Output: {calib_dir}")
    
    try:
        # Load model and data
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        # Load dataset
        if Path(dataset_path).is_dir():
            import polars as pl
            df = pl.scan_parquet(str(Path(dataset_path) / "*.parquet")).collect().to_pandas()
        else:
            df = pd.read_parquet(dataset_path)
        
        if sample_size and len(df) > sample_size:
            df = df.sample(sample_size, random_state=42)
        
        # Prepare features and labels
        from meta_spliceai.splice_engine.meta_models.builder import preprocessing
        from meta_spliceai.splice_engine.meta_models.training.classifier_utils import _encode_labels
        
        X_df, y_series = preprocessing.prepare_training_data(
            df, label_col="splice_type", return_type="pandas", verbose=0
        )
        
        # Get model predictions
        y_true = _encode_labels(y_series)
        X = X_df.values
        
        # Get model probabilities
        if hasattr(model, 'predict_proba'):
            proba_multiclass = model.predict_proba(X)
            # Convert to binary splice/non-splice
            y_prob = proba_multiclass[:, 1] + proba_multiclass[:, 2]  # donor + acceptor
            y_true_binary = (y_true != 0).astype(int)
        else:
            raise ValueError("Model does not support probability prediction")
        
        # 1. Basic calibration diagnostics
        original_metrics = diagnose_calibration_issues(
            y_true_binary, y_prob, (y_prob > 0.5).astype(int),
            calib_dir, "original", plot_format, verbose
        )
        
        # 2. Selection bias analysis
        bias_metrics = detect_selection_bias_effects(
            dataset_path, y_prob, y_true_binary, calib_dir, verbose
        )
        
        # 3. Calibration methods comparison
        # Try to get logits for temperature scaling
        logits = None
        if hasattr(model, 'models') and len(model.models) >= 2:
            # For ensemble models, use the sum of class 1 and 2 logits
            try:
                import numpy as np
                donor_logits = model.models[1].predict(X, output_margin=True) if hasattr(model.models[1], 'predict') else None
                acceptor_logits = model.models[2].predict(X, output_margin=True) if hasattr(model.models[2], 'predict') else None
                if donor_logits is not None and acceptor_logits is not None:
                    logits = donor_logits + acceptor_logits
            except Exception:
                pass
        
        calibration_comparison = evaluate_calibration_methods(
            y_true_binary, y_prob, logits, calib_dir, plot_format, verbose
        )
        
        # 4. Generate summary report
        report = {
            'dataset_info': {
                'path': str(dataset_path),
                'n_samples': len(df),
                'n_features': X.shape[1] if X.ndim > 1 else 1
            },
            'original_calibration': original_metrics,
            'selection_bias': bias_metrics,
            'calibration_methods': calibration_comparison,
            'recommendations': []
        }
        
        # Generate recommendations
        if original_metrics['expected_calibration_error'] > 0.1:
            report['recommendations'].append("HIGH calibration error detected - recalibration strongly recommended")
        
        if original_metrics['extreme_total_rate'] > 0.5:
            report['recommendations'].append("EXTREME overconfidence detected - consider entropy-based correction")
        
        if bias_metrics.get('mean_splice_density', 0) > 0.05:
            report['recommendations'].append("Selection bias detected - consider bias-aware calibration or dataset rebalancing")
        
        best_method = min(calibration_comparison.keys(), 
                         key=lambda x: calibration_comparison[x]['expected_calibration_error'])
        report['recommendations'].append(f"Best calibration method: {best_method}")
        
        # Save comprehensive report
        report_path = calib_dir / "calibration_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Create summary text report
        summary_path = calib_dir / "calibration_summary.txt"
        with open(summary_path, 'w') as f:
            f.write("COMPREHENSIVE CALIBRATION ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Dataset: {dataset_path}\n")
            f.write(f"Model: {model_path}\n")
            f.write(f"Analysis Date: {pd.Timestamp.now()}\n\n")
            
            f.write("CALIBRATION QUALITY:\n")
            f.write(f"  Expected Calibration Error: {original_metrics['expected_calibration_error']:.4f}\n")
            f.write(f"  Brier Score: {original_metrics['brier_score']:.4f}\n")
            f.write(f"  Overconfidence Rate: {original_metrics['overconfidence_rate']:.1%}\n")
            f.write(f"  Extreme Predictions: {original_metrics['extreme_total_rate']:.1%}\n\n")
            
            if bias_metrics:
                f.write("SELECTION BIAS INDICATORS:\n")
                f.write(f"  Mean Splice Density: {bias_metrics.get('mean_splice_density', 0):.3f}\n")
                f.write(f"  High-Density Genes: {bias_metrics.get('high_density_genes', 0)}\n\n")
            
            f.write("RECOMMENDATIONS:\n")
            for i, rec in enumerate(report['recommendations'], 1):
                f.write(f"  {i}. {rec}\n")
        
        if verbose:
            print(f"✅ Calibration analysis completed")
            print(f"📁 Reports saved to: {calib_dir}")
            print(f"📄 Summary: {summary_path}")
            print(f"📊 Detailed: {report_path}")
        
        return report
        
    except Exception as e:
        if verbose:
            print(f"❌ Calibration analysis failed: {e}")
        return {'error': str(e)} 