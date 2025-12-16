"""
Integration utilities for memory checking in training scripts.

This module provides easy integration of memory checking into
existing training workflows.
"""

from pathlib import Path
from typing import Optional
import warnings

from .memory_checker import MemoryChecker


def check_memory_before_training(dataset_path: str | Path, 
                                cv_folds: int = 5,
                                n_estimators: int = 500,
                                auto_adjust_row_cap: bool = True,
                                verbose: bool = True) -> dict:
    """Check memory before training and optionally suggest row cap.
    
    Parameters
    ----------
    dataset_path : str | Path
        Path to the dataset
    cv_folds : int, default=5
        Number of CV folds
    n_estimators : int, default=500
        Number of XGBoost estimators
    auto_adjust_row_cap : bool, default=True
        Whether to suggest row cap adjustments
    verbose : bool, default=True
        Whether to print assessment
        
    Returns
    -------
    dict
        Assessment results with suggested row_cap if needed
    """
    checker = MemoryChecker(dataset_path)
    assessment = checker.assess_oom_risk(cv_folds, n_estimators)
    
    if verbose:
        print("\n" + "="*50)
        print("Pre-Training Memory Assessment")
        print("="*50)
        
        if 'error' in assessment:
            print(f"❌ Error: {assessment['error']}")
            return assessment
        
        print(f"Dataset: {dataset_path}")
        print(f"Size: {assessment['dataset_memory_gb']:.1f} GB in memory")
        print(f"Estimated peak usage: {assessment['peak_memory_gb']:.1f} GB")
        print(f"Available memory: {assessment['available_memory_gb']:.1f} GB")
        print(f"\n{assessment['color']} Risk Level: {assessment['risk_level']}")
        print(f"Recommendation: {assessment['recommendation']}")
    
    # Add suggested row cap if high risk and auto-adjust enabled
    if auto_adjust_row_cap and assessment.get('risk_level') == 'HIGH':
        # Calculate suggested row cap based on safe memory usage
        safe_memory = assessment['available_memory_gb'] * 0.7  # 70% of available
        current_peak = assessment['peak_memory_gb']
        reduction_factor = safe_memory / current_peak
        suggested_row_cap = int(500_000 * reduction_factor)
        
        assessment['suggested_row_cap'] = max(50_000, suggested_row_cap)  # Minimum 50k rows
        
        if verbose:
            print(f"Suggested --row-cap: {assessment['suggested_row_cap']}")
    
    if verbose:
        print("="*50 + "\n")
    
    return assessment


def memory_check_decorator(cv_folds: int = 5, n_estimators: int = 500):
    """Decorator to add memory checking to training functions.
    
    Parameters
    ----------
    cv_folds : int, default=5
        Number of CV folds
    n_estimators : int, default=500
        Number of XGBoost estimators
        
    Returns
    -------
    function
        Decorated function with memory checking
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Try to extract dataset path from args
            dataset_path = None
            if args and len(args) > 0:
                # Assume first argument might be dataset path
                potential_path = Path(args[0])
                if potential_path.exists():
                    dataset_path = potential_path
            
            # Try to extract from kwargs
            if not dataset_path:
                for key in ['dataset', 'dataset_path', 'data_path']:
                    if key in kwargs:
                        potential_path = Path(kwargs[key])
                        if potential_path.exists():
                            dataset_path = potential_path
                            break
            
            if dataset_path:
                assessment = check_memory_before_training(
                    dataset_path, cv_folds, n_estimators, verbose=True
                )
                
                if assessment.get('risk_level') == 'HIGH':
                    response = input("High OOM risk detected. Continue anyway? (y/N): ")
                    if response.lower() not in ['y', 'yes']:
                        print("Training cancelled by user.")
                        return None
            else:
                warnings.warn("Could not determine dataset path for memory checking")
            
            return func(*args, **kwargs)
        return wrapper
    return decorator


# Example usage in training scripts:
def integrate_memory_check_into_cv_script(args):
    """Example integration into existing CV scripts.
    
    Parameters
    ----------
    args : argparse.Namespace
        Parsed command line arguments
        
    Returns
    -------
    bool
        True if safe to proceed, False otherwise
    """
    if hasattr(args, 'dataset') and hasattr(args, 'n_folds'):
        assessment = check_memory_before_training(
            args.dataset,
            cv_folds=getattr(args, 'n_folds', 5),
            n_estimators=getattr(args, 'n_estimators', 500),
            verbose=True
        )
        
        # Auto-adjust row cap if not set and high risk
        if (assessment.get('risk_level') == 'HIGH' and 
            getattr(args, 'row_cap', 0) == 0 and
            'suggested_row_cap' in assessment):
            
            print(f"\n⚠️  Auto-adjusting row cap to {assessment['suggested_row_cap']} due to memory constraints")
            args.row_cap = assessment['suggested_row_cap']
            return True
        
        return assessment.get('risk_level') != 'HIGH'
    
    return True  # Default to safe if can't assess
