"""
Integration module for optional calibration analysis in CV training.

This module provides safe, optional integration of advanced calibration analysis
into the main CV training pipeline without breaking existing functionality.
"""

from pathlib import Path
from typing import Optional, Dict, Any, Union
import warnings

def run_calibration_analysis(
    dataset_path: Union[str, Path],
    model_path: Union[str, Path], 
    out_dir: Union[str, Path],
    *,
    sample_size: Optional[int] = None,
    plot_format: str = "png",
    verbose: bool = True,
    enable_analysis: bool = True
) -> Optional[Dict[str, Any]]:
    """
    Run comprehensive calibration analysis if requested and possible.
    
    This function provides a safe interface for calibration analysis that:
    1. Can be disabled without affecting the main pipeline
    2. Handles import errors gracefully
    3. Returns None if analysis fails or is disabled
    
    Parameters
    ----------
    dataset_path : str or Path
        Path to dataset
    model_path : str or Path
        Path to trained model
    out_dir : str or Path
        Output directory
    sample_size : int, optional
        Sample size for analysis (None = use all data)
    plot_format : str
        Plot format ('png', 'pdf', 'svg')
    verbose : bool
        Enable verbose output
    enable_analysis : bool
        Whether to run calibration analysis
        
    Returns
    -------
    Optional[Dict[str, Any]]
        Calibration analysis results, or None if disabled/failed
    """
    
    if not enable_analysis:
        if verbose:
            print("[Calibration Analysis] Disabled - skipping")
        return None
    
    try:
        from .calibration_diagnostics import generate_calibration_report
        
        if verbose:
            print("\n[Calibration Analysis] Running comprehensive calibration diagnostics...")
        
        results = generate_calibration_report(
            dataset_path=dataset_path,
            model_path=model_path,
            out_dir=out_dir,
            sample_size=sample_size,
            plot_format=plot_format,
            verbose=verbose
        )
        
        if 'error' in results:
            if verbose:
                print(f"[Calibration Analysis] ❌ Analysis failed: {results['error']}")
            return None
        
        if verbose:
            print("[Calibration Analysis] ✅ Analysis completed successfully")
            
            # Print key findings
            original = results.get('original_calibration', {})
            ece = original.get('expected_calibration_error', 0)
            overconf = original.get('overconfidence_rate', 0)
            extreme = original.get('extreme_total_rate', 0)
            
            print(f"[Calibration Analysis] Key findings:")
            print(f"  Expected Calibration Error: {ece:.4f} {'(HIGH)' if ece > 0.1 else '(OK)' if ece < 0.05 else '(MODERATE)'}")
            print(f"  Overconfidence Rate: {overconf:.1%} {'(HIGH)' if overconf > 0.1 else '(OK)'}")
            print(f"  Extreme Predictions: {extreme:.1%} {'(HIGH)' if extreme > 0.5 else '(MODERATE)' if extreme > 0.3 else '(OK)'}")
            
            # Show recommendations
            recommendations = results.get('recommendations', [])
            if recommendations:
                print(f"[Calibration Analysis] Recommendations:")
                for i, rec in enumerate(recommendations[:3], 1):  # Show top 3
                    print(f"  {i}. {rec}")
        
        return results
        
    except ImportError as e:
        if verbose:
            print(f"[Calibration Analysis] ⚠️  Module not available: {e}")
        return None
    except Exception as e:
        if verbose:
            print(f"[Calibration Analysis] ❌ Analysis failed: {e}")
        return None


def detect_overconfidence_issues(
    y_true, 
    y_prob, 
    y_splice_type=None,
    verbose: bool = True
) -> Dict[str, float]:
    """
    Quick overconfidence detection for inline use.
    
    Parameters
    ----------
    y_true : array-like
        True labels
    y_prob : array-like
        Predicted probabilities
    y_splice_type : array-like, optional
        Splice type labels (e.g., "donor", "acceptor", "neither", "0")
        If provided, statistics will be broken down by splice type
    verbose : bool
        Print findings
        
    Returns
    -------
    Dict[str, float]
        Basic overconfidence metrics
    """
    
    try:
        import numpy as np
        
        y_true = np.asarray(y_true)
        y_prob = np.asarray(y_prob)
        
        # Basic overconfidence metrics
        extreme_high = np.sum(y_prob > 0.99) / len(y_prob)
        extreme_low = np.sum(y_prob < 0.01) / len(y_prob)
        extreme_total = extreme_high + extreme_low
        
        # High-confidence errors
        high_conf_mask = y_prob > 0.9
        high_conf_errors = np.sum(((y_prob > 0.5).astype(int) != y_true) & high_conf_mask) if high_conf_mask.any() else 0
        high_conf_total = np.sum(high_conf_mask)
        overconf_rate = high_conf_errors / high_conf_total if high_conf_total > 0 else 0.0
        
        # Probability statistics
        prob_mean = np.mean(y_prob)
        prob_std = np.std(y_prob)
        prob_entropy = -np.mean(y_prob * np.log(y_prob + 1e-15) + (1-y_prob) * np.log(1-y_prob + 1e-15))
        
        metrics = {
            'extreme_total_rate': float(extreme_total),
            'extreme_high_rate': float(extreme_high),
            'extreme_low_rate': float(extreme_low),
            'overconfidence_rate': float(overconf_rate),
            'prob_mean': float(prob_mean),
            'prob_std': float(prob_std),
            'prob_entropy': float(prob_entropy),
            'n_samples': len(y_true)
        }
        
        if verbose:
            print(f"\n[Quick Overconfidence Check]")
            print(f"  Extreme predictions (near 0 or 1): {extreme_total:.1%}")
            print(f"  High-confidence errors: {overconf_rate:.1%} ({high_conf_errors}/{high_conf_total})")
            
            # If splice type information is provided, break down by type
            if y_splice_type is not None:
                y_splice_type = np.asarray(y_splice_type)
                
                # Map splice types to consistent labels
                def normalize_splice_type(splice_type):
                    if isinstance(splice_type, (int, np.integer)):
                        if splice_type == 0:
                            return "neither"
                        elif splice_type == 1:
                            return "donor"
                        elif splice_type == 2:
                            return "acceptor"
                        else:
                            return "neither"
                    else:
                        splice_str = str(splice_type).lower()
                        if splice_str in ["0", "neither", "non-splice"]:
                            return "neither"
                        elif splice_str in ["1", "donor"]:
                            return "donor"
                        elif splice_str in ["2", "acceptor"]:
                            return "acceptor"
                        else:
                            return "neither"
                
                # Normalize splice types
                normalized_types = np.array([normalize_splice_type(st) for st in y_splice_type])
                
                # Calculate statistics for each type
                for splice_type in ["donor", "acceptor", "neither"]:
                    type_mask = normalized_types == splice_type
                    if np.sum(type_mask) > 0:
                        type_prob = y_prob[type_mask]
                        type_mean = np.mean(type_prob)
                        type_std = np.std(type_prob)
                        type_count = np.sum(type_mask)
                        
                        # Store type-specific metrics
                        metrics[f'{splice_type}_prob_mean'] = float(type_mean)
                        metrics[f'{splice_type}_prob_std'] = float(type_std)
                        metrics[f'{splice_type}_n_samples'] = int(type_count)
                        
                        print(f"  {splice_type.capitalize():8} probability mean: {type_mean:.3f}, std: {type_std:.3f} (n={type_count})")
                
                print(f"  Overall probability mean: {prob_mean:.3f}, std: {prob_std:.3f}")
            else:
                print(f"  Probability mean: {prob_mean:.3f}, std: {prob_std:.3f}")
            
            if extreme_total > 0.5:
                print(f"  ⚠️  EXTREME overconfidence detected! Consider calibration correction.")
            elif extreme_total > 0.3:
                print(f"  ⚠️  High overconfidence detected.")
            else:
                print(f"  ✅ Reasonable confidence distribution.")
                
        return metrics
        
    except Exception as e:
        if verbose:
            print(f"[Quick Overconfidence Check] Failed: {e}")
        return {}


def suggest_evaluation_approach(
    calibration_results: Optional[Dict[str, Any]] = None,
    overconf_metrics: Optional[Dict[str, float]] = None,
    verbose: bool = True
) -> str:
    """
    Suggest the best evaluation approach based on calibration analysis.
    
    Parameters
    ----------
    calibration_results : Dict, optional
        Results from full calibration analysis
    overconf_metrics : Dict, optional
        Results from quick overconfidence check
    verbose : bool
        Print suggestions
        
    Returns
    -------
    str
        Recommended evaluation approach
    """
    
    # Extract metrics from either source
    extreme_rate = 0.0
    overconf_rate = 0.0
    ece = 0.0
    
    if calibration_results:
        original = calibration_results.get('original_calibration', {})
        extreme_rate = original.get('extreme_total_rate', 0)
        overconf_rate = original.get('overconfidence_rate', 0)
        ece = original.get('expected_calibration_error', 0)
    elif overconf_metrics:
        extreme_rate = overconf_metrics.get('extreme_total_rate', 0)
        overconf_rate = overconf_metrics.get('overconfidence_rate', 0)
    
    # Decision logic
    if extreme_rate > 0.5 or ece > 0.1:
        approach = "argmax_only"
        reason = "Severe overconfidence detected - threshold-based evaluation unreliable"
    elif extreme_rate > 0.3 or ece > 0.05:
        approach = "argmax_primary"
        reason = "Moderate overconfidence - use argmax as primary, thresholds as secondary"
    else:
        approach = "threshold_safe"
        reason = "Good calibration - threshold-based evaluation should work"
    
    if verbose:
        print(f"\n[Evaluation Approach Recommendation]")
        print(f"  Analysis: Extreme rate={extreme_rate:.1%}, Overconf rate={overconf_rate:.1%}, ECE={ece:.3f}")
        print(f"  Recommendation: {approach}")
        print(f"  Reason: {reason}")
        
        if approach == "argmax_only":
            print(f"  💡 Use only argmax-based evaluation methods")
            print(f"     Avoid threshold-based metrics entirely")
        elif approach == "argmax_primary":
            print(f"  💡 Use argmax as primary evaluation method")
            print(f"     Include threshold metrics but interpret cautiously")
        else:
            print(f"  💡 Threshold-based evaluation should work reliably")
    
    return approach


# Safe import function for the main CV script
def get_calibration_functions():
    """
    Get calibration analysis functions if available.
    
    Returns
    -------
    Dict[str, callable] or None
        Dictionary of available functions, or None if not available
    """
    try:
        return {
            'run_calibration_analysis': run_calibration_analysis,
            'detect_overconfidence_issues': detect_overconfidence_issues,
            'suggest_evaluation_approach': suggest_evaluation_approach
        }
    except Exception:
        return None 