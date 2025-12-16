#!/usr/bin/env python3
"""
Memory usage assessment tool for meta-model training.

This module provides utilities to assess OOM risk before running
gene-aware or chromosome-aware cross-validation experiments.

Usage:
    python -m meta_spliceai.splice_engine.meta_models.diagnosis.memory_checker <dataset_path> [options]

Example:
    python -m meta_spliceai.splice_engine.meta_models.diagnosis.memory_checker train_pc_1000_3mers/master --cv-folds 5
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Tuple, Optional
import warnings

try:
    import psutil
except ImportError:
    psutil = None
    warnings.warn("psutil not available - system memory detection will be limited")


class MemoryChecker:
    """Memory usage assessment for meta-model training."""
    
    # Memory multipliers for different stages
    PARQUET_MEMORY_MULTIPLIER = 3.0  # Parquet in memory vs on disk
    XGBOOST_TRAINING_MULTIPLIER = 3.0  # XGBoost training vs dataset size
    CV_FOLD_MULTIPLIER = 1.2  # Additional memory per CV fold
    SAFETY_BUFFER = 1.5  # Safety buffer multiplier
    
    def __init__(self, dataset_path: str | Path):
        """Initialize memory checker for a dataset.
        
        Parameters
        ----------
        dataset_path : str | Path
            Path to the dataset directory or file
        """
        self.dataset_path = Path(dataset_path)
        self.dataset_info = self._analyze_dataset()
        self.system_info = self._get_system_info()
    
    def _analyze_dataset(self) -> Dict:
        """Analyze dataset size and structure."""
        info = {
            'path': str(self.dataset_path),
            'exists': self.dataset_path.exists(),
            'total_size_mb': 0,
            'total_size_gb': 0,
            'file_count': 0,
            'files': []
        }
        
        if not info['exists']:
            return info
        
        if self.dataset_path.is_file():
            # Single file
            size = self.dataset_path.stat().st_size
            info['total_size_mb'] = size / 1024 / 1024
            info['file_count'] = 1
            info['files'] = [{'name': self.dataset_path.name, 'size_mb': info['total_size_mb']}]
        elif self.dataset_path.is_dir():
            # Directory with multiple files
            for file_path in self.dataset_path.glob('*.parquet'):
                size = file_path.stat().st_size
                size_mb = size / 1024 / 1024
                info['files'].append({'name': file_path.name, 'size_mb': size_mb})
                info['total_size_mb'] += size_mb
                info['file_count'] += 1
        
        info['total_size_gb'] = info['total_size_mb'] / 1024
        return info
    
    def _get_system_info(self) -> Dict:
        """Get system resource information."""
        info = {
            'total_memory_gb': 0,
            'available_memory_gb': 0,
            'cpu_cores_physical': 0,
            'cpu_cores_logical': 0,
            'psutil_available': psutil is not None
        }
        
        if psutil:
            memory = psutil.virtual_memory()
            info['total_memory_gb'] = memory.total / 1024**3
            info['available_memory_gb'] = memory.available / 1024**3
            info['cpu_cores_physical'] = psutil.cpu_count(logical=False)
            info['cpu_cores_logical'] = psutil.cpu_count(logical=True)
        else:
            # Fallback to /proc/meminfo on Linux
            try:
                with open('/proc/meminfo', 'r') as f:
                    for line in f:
                        if 'MemTotal' in line:
                            total_kb = int(line.split()[1])
                            info['total_memory_gb'] = total_kb / 1024 / 1024
                        elif 'MemAvailable' in line:
                            available_kb = int(line.split()[1])
                            info['available_memory_gb'] = available_kb / 1024 / 1024
            except (FileNotFoundError, ValueError):
                pass
            
            # Try to get CPU info
            try:
                info['cpu_cores_logical'] = os.cpu_count() or 0
            except:
                pass
        
        return info
    
    def estimate_memory_usage(self, cv_folds: int = 5, n_estimators: int = 500) -> Dict:
        """Estimate memory usage for training.
        
        Parameters
        ----------
        cv_folds : int, default=5
            Number of CV folds
        n_estimators : int, default=500
            Number of XGBoost estimators
            
        Returns
        -------
        Dict
            Memory usage estimates in GB
        """
        if not self.dataset_info['exists']:
            return {'error': 'Dataset not found'}
        
        # Base dataset memory usage
        dataset_memory_gb = self.dataset_info['total_size_gb'] * self.PARQUET_MEMORY_MULTIPLIER
        
        # XGBoost training memory (per model)
        training_memory_gb = dataset_memory_gb * self.XGBOOST_TRAINING_MULTIPLIER
        
        # CV memory (multiple models in memory)
        cv_memory_gb = training_memory_gb * cv_folds * self.CV_FOLD_MULTIPLIER
        
        # Peak memory with safety buffer
        peak_memory_gb = cv_memory_gb * self.SAFETY_BUFFER
        
        # Estimator impact (rough approximation)
        estimator_factor = min(2.0, 1.0 + (n_estimators - 100) / 1000)
        peak_memory_gb *= estimator_factor
        
        return {
            'dataset_memory_gb': dataset_memory_gb,
            'training_memory_gb': training_memory_gb,
            'cv_memory_gb': cv_memory_gb,
            'peak_memory_gb': peak_memory_gb,
            'estimator_factor': estimator_factor,
            'cv_folds': cv_folds,
            'n_estimators': n_estimators
        }
    
    def assess_oom_risk(self, cv_folds: int = 5, n_estimators: int = 500, 
                       safety_threshold: float = 0.8) -> Dict:
        """Assess OOM risk for training.
        
        Parameters
        ----------
        cv_folds : int, default=5
            Number of CV folds
        n_estimators : int, default=500
            Number of XGBoost estimators
        safety_threshold : float, default=0.8
            Fraction of available memory to use as threshold
            
        Returns
        -------
        Dict
            Risk assessment results
        """
        memory_est = self.estimate_memory_usage(cv_folds, n_estimators)
        
        if 'error' in memory_est:
            return memory_est
        
        available_gb = self.system_info['available_memory_gb']
        peak_gb = memory_est['peak_memory_gb']
        safe_threshold_gb = available_gb * safety_threshold
        
        # Determine risk level
        if peak_gb <= safe_threshold_gb:
            risk_level = 'SAFE'
            recommendation = f'Use --row-cap 0 for full dataset'
            color = '✅'
        elif peak_gb <= available_gb:
            risk_level = 'MODERATE'
            recommendation = f'Try --row-cap 0 but monitor memory usage'
            color = '⚠️'
        else:
            risk_level = 'HIGH'
            # Calculate suggested row cap
            target_memory = safe_threshold_gb / self.SAFETY_BUFFER
            row_cap = int(500_000 * target_memory / memory_est['cv_memory_gb'])
            recommendation = f'Use --row-cap {row_cap} or smaller dataset'
            color = '❌'
        
        return {
            'risk_level': risk_level,
            'recommendation': recommendation,
            'color': color,
            'available_memory_gb': available_gb,
            'estimated_peak_gb': peak_gb,
            'safe_threshold_gb': safe_threshold_gb,
            'memory_usage_pct': (peak_gb / available_gb * 100) if available_gb > 0 else 0,
            **memory_est
        }
    
    def print_assessment(self, cv_folds: int = 5, n_estimators: int = 500, 
                        verbose: bool = True) -> None:
        """Print comprehensive memory assessment.
        
        Parameters
        ----------
        cv_folds : int, default=5
            Number of CV folds
        n_estimators : int, default=500
            Number of XGBoost estimators
        verbose : bool, default=True
            Whether to print detailed information
        """
        print("=" * 60)
        print("Memory Usage Assessment for Meta-Model Training")
        print("=" * 60)
        
        # Dataset information
        if verbose:
            print(f"\nDataset Analysis:")
            print(f"  Path: {self.dataset_info['path']}")
            print(f"  Exists: {self.dataset_info['exists']}")
            if self.dataset_info['exists']:
                print(f"  Files: {self.dataset_info['file_count']}")
                print(f"  Total size: {self.dataset_info['total_size_mb']:.1f} MB ({self.dataset_info['total_size_gb']:.2f} GB)")
                
                if len(self.dataset_info['files']) <= 10:  # Don't spam for large datasets
                    for file_info in self.dataset_info['files']:
                        print(f"    {file_info['name']}: {file_info['size_mb']:.1f} MB")
        
        # System information
        if verbose:
            print(f"\nSystem Resources:")
            if self.system_info['psutil_available']:
                print(f"  Total RAM: {self.system_info['total_memory_gb']:.1f} GB")
                print(f"  Available RAM: {self.system_info['available_memory_gb']:.1f} GB")
                print(f"  CPU cores: {self.system_info['cpu_cores_physical']} physical, {self.system_info['cpu_cores_logical']} logical")
            else:
                print(f"  Total RAM: {self.system_info['total_memory_gb']:.1f} GB")
                print(f"  Available RAM: {self.system_info['available_memory_gb']:.1f} GB")
                print(f"  CPU cores: {self.system_info['cpu_cores_logical']} logical")
        
        # Memory estimates
        assessment = self.assess_oom_risk(cv_folds, n_estimators)
        
        if 'error' in assessment:
            print(f"\n❌ Error: {assessment['error']}")
            return
        
        if verbose:
            print(f"\nMemory Usage Estimates:")
            print(f"  Dataset in memory: ~{assessment['dataset_memory_gb']:.1f} GB")
            print(f"  XGBoost training: ~{assessment['training_memory_gb']:.1f} GB per model")
            print(f"  {cv_folds}-fold CV peak: ~{assessment['cv_memory_gb']:.1f} GB")
            print(f"  With safety buffer: ~{assessment['peak_memory_gb']:.1f} GB")
            print(f"  Estimator factor: {assessment['estimator_factor']:.1f}x (n_estimators={n_estimators})")
        
        # Risk assessment
        print(f"\n{assessment['color']} OOM Risk Assessment: {assessment['risk_level']}")
        print(f"Your system has {assessment['available_memory_gb']:.1f} GB available, estimated peak usage is {assessment['peak_memory_gb']:.1f} GB ({assessment['memory_usage_pct']:.1f}% of available).")
        print(f"Recommendation: {assessment['recommendation']}")
        
        print("=" * 60)


def assess_oom_risk(dataset_path: str | Path, cv_folds: int = 5, 
                   n_estimators: int = 500, verbose: bool = True) -> Dict:
    """Quick OOM risk assessment function.
    
    Parameters
    ----------
    dataset_path : str | Path
        Path to the dataset
    cv_folds : int, default=5
        Number of CV folds
    n_estimators : int, default=500
        Number of XGBoost estimators
    verbose : bool, default=True
        Whether to print assessment
        
    Returns
    -------
    Dict
        Assessment results
    """
    checker = MemoryChecker(dataset_path)
    assessment = checker.assess_oom_risk(cv_folds, n_estimators)
    
    if verbose:
        checker.print_assessment(cv_folds, n_estimators, verbose=True)
    
    return assessment


def main():
    """Command-line interface for memory assessment."""
    parser = argparse.ArgumentParser(
        description="Assess OOM risk for meta-model training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m meta_spliceai.splice_engine.meta_models.diagnosis.memory_checker train_pc_1000_3mers/master
  python -m meta_spliceai.splice_engine.meta_models.diagnosis.memory_checker train_pc_1000/master --cv-folds 5 --n-estimators 500
  python -m meta_spliceai.splice_engine.meta_models.diagnosis.memory_checker data/my_dataset.parquet --quiet
        """
    )
    
    parser.add_argument('dataset_path', help='Path to dataset directory or file')
    parser.add_argument('--cv-folds', type=int, default=5, help='Number of CV folds (default: 5)')
    parser.add_argument('--n-estimators', type=int, default=500, help='Number of XGBoost estimators (default: 500)')
    parser.add_argument('--safety-threshold', type=float, default=0.8, 
                       help='Memory safety threshold (default: 0.8)')
    parser.add_argument('--quiet', action='store_true', help='Only show risk assessment')
    
    args = parser.parse_args()
    
    try:
        assessment = assess_oom_risk(
            args.dataset_path, 
            cv_folds=args.cv_folds,
            n_estimators=args.n_estimators,
            verbose=not args.quiet
        )
        
        # Exit with appropriate code
        if assessment.get('risk_level') == 'HIGH':
            sys.exit(1)  # High risk
        elif assessment.get('risk_level') == 'MODERATE':
            sys.exit(2)  # Moderate risk
        else:
            sys.exit(0)  # Safe
            
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(3)


if __name__ == "__main__":
    main()
