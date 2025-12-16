#!/usr/bin/env python3
"""
Cross-dataset validation utility for meta-models.

This script validates trained meta-models on different datasets to assess
generalization performance across various gene sets, k-mer configurations,
and splice site densities.
"""

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from meta_spliceai.splice_engine.meta_models.training import datasets
from meta_spliceai.splice_engine.meta_models.builder import preprocessing
from meta_spliceai.splice_engine.meta_models.training.classifier_utils import _encode_labels


def load_trained_model(model_path: Path) -> Tuple[object, bool]:
    """
    Load a trained meta-model from pickle file.
    
    Args:
        model_path: Path to the pickled model file
        
    Returns:
        Tuple of (model, success_flag)
    """
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model, True
    except Exception as e:
        print(f"❌ Failed to load model from {model_path}: {e}")
        return None, False


def load_and_preprocess_dataset(dataset_path: str, sample_size: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, List[str], bool]:
    """
    Load and preprocess a dataset for validation.
    
    Args:
        dataset_path: Path to the dataset
        sample_size: Optional sample size for large datasets
        
    Returns:
        Tuple of (X, y, feature_names, success_flag)
    """
    try:
        print(f"  Loading dataset: {dataset_path}")
        
        # Load dataset
        df = datasets.load_dataset(dataset_path)
        
        if sample_size and len(df) > sample_size:
            print(f"  Sampling {sample_size} rows from {len(df)} total rows")
            df = df.sample(n=sample_size, random_state=42)
        
        print(f"  Dataset shape: {df.shape}")
        
        # Preprocess data
        X_df, y_series = preprocessing.prepare_training_data(
            df, 
            label_col="splice_type", 
            return_type="pandas", 
            verbose=0,
            encode_chrom=True
        )
        
        # Convert to numpy
        X = X_df.values
        y = _encode_labels(y_series)
        feature_names = list(X_df.columns)
        
        print(f"  Preprocessed shape: X={X.shape}, y={y.shape}")
        print(f"  Class distribution: {np.bincount(y)}")
        
        return X, y, feature_names, True
        
    except Exception as e:
        print(f"  ❌ Failed to load dataset {dataset_path}: {e}")
        return None, None, None, False


def align_features(X: np.ndarray, dataset_features: List[str], model_features: List[str]) -> Tuple[np.ndarray, List[str]]:
    """
    Align dataset features with model features.
    
    Args:
        X: Feature matrix
        dataset_features: List of feature names in dataset
        model_features: List of feature names expected by model
        
    Returns:
        Tuple of (aligned_X, missing_features)
    """
    # Find common features and their indices
    common_features = []
    common_indices = []
    missing_features = []
    
    for i, model_feat in enumerate(model_features):
        if model_feat in dataset_features:
            dataset_idx = dataset_features.index(model_feat)
            common_features.append(model_feat)
            common_indices.append(dataset_idx)
        else:
            missing_features.append(model_feat)
    
    # Extract common features
    if common_indices:
        X_aligned = X[:, common_indices]
    else:
        X_aligned = np.empty((X.shape[0], 0))
    
    print(f"  Feature alignment: {len(model_features)} model features, {len(common_features)} common, {len(missing_features)} missing")
    
    if missing_features:
        print(f"  Missing features: {missing_features[:5]}{'...' if len(missing_features) > 5 else ''}")
    
    return X_aligned, missing_features


def evaluate_model_on_dataset(model, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """
    Evaluate a model on a dataset and return comprehensive metrics.
    
    Args:
        model: Trained model
        X: Feature matrix
        y: Target labels
        
    Returns:
        Dictionary of evaluation metrics
    """
    try:
        # Make predictions
        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X)
            y_pred = np.argmax(y_prob, axis=1)
        else:
            y_pred = model.predict(X)
            y_prob = None
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'macro_f1': f1_score(y, y_pred, average='macro'),
            'weighted_f1': f1_score(y, y_pred, average='weighted'),
            'macro_precision': precision_score(y, y_pred, average='macro', zero_division=0),
            'macro_recall': recall_score(y, y_pred, average='macro', zero_division=0),
            'n_samples': len(y),
            'n_features': X.shape[1]
        }
        
        # Class-specific metrics
        for class_idx in range(len(np.unique(y))):
            class_name = ['neither', 'donor', 'acceptor'][class_idx] if class_idx < 3 else f'class_{class_idx}'
            class_mask = (y == class_idx)
            if class_mask.any():
                class_f1 = f1_score(y, y_pred, labels=[class_idx], average='macro')
                metrics[f'{class_name}_f1'] = class_f1
                metrics[f'{class_name}_support'] = class_mask.sum()
        
        return metrics
        
    except Exception as e:
        print(f"  ❌ Evaluation failed: {e}")
        return {'error': str(e)}


def run_cross_dataset_validation(model_path: str, test_datasets: List[str], 
                                sample_size: Optional[int] = None, 
                                output_file: Optional[str] = None) -> Dict:
    """
    Run cross-dataset validation on multiple datasets.
    
    Args:
        model_path: Path to trained model
        test_datasets: List of dataset paths to test on
        sample_size: Optional sample size for large datasets
        output_file: Optional output file for results
        
    Returns:
        Dictionary containing all validation results
    """
    print("🔍 CROSS-DATASET VALIDATION")
    print("=" * 50)
    
    # Load trained model
    model_path = Path(model_path)
    print(f"Loading model: {model_path}")
    
    model, success = load_trained_model(model_path)
    if not success:
        return {'error': 'Failed to load model'}
    
    # Get model feature names if available
    model_features = None
    if hasattr(model, 'feature_names'):
        model_features = model.feature_names
    elif hasattr(model, 'feature_names_'):
        model_features = model.feature_names_
    elif hasattr(model, 'models') and hasattr(model.models[0], 'feature_names_in_'):
        model_features = model.models[0].feature_names_in_.tolist()
    
    print(f"Model loaded successfully")
    if model_features:
        print(f"Model expects {len(model_features)} features")
    
    # Results storage
    results = {
        'model_path': str(model_path),
        'model_features_count': len(model_features) if model_features else None,
        'validation_results': {},
        'summary': {}
    }
    
    # Test on each dataset
    all_metrics = []
    
    for dataset_path in test_datasets:
        print(f"\n📊 Testing on dataset: {dataset_path}")
        print("-" * 40)
        
        # Check if dataset exists
        if not Path(dataset_path).exists():
            print(f"  ⚠️ Dataset not found: {dataset_path}")
            results['validation_results'][dataset_path] = {'error': 'Dataset not found'}
            continue
        
        # Load and preprocess dataset
        X, y, dataset_features, success = load_and_preprocess_dataset(dataset_path, sample_size)
        if not success:
            results['validation_results'][dataset_path] = {'error': 'Failed to load dataset'}
            continue
        
        # Align features if model features are known
        if model_features:
            X_aligned, missing_features = align_features(X, dataset_features, model_features)
            if len(missing_features) > len(model_features) * 0.5:  # More than 50% features missing
                print(f"  ⚠️ Too many missing features ({len(missing_features)}/{len(model_features)}), skipping")
                results['validation_results'][dataset_path] = {'error': 'Too many missing features'}
                continue
            X = X_aligned
        
        # Evaluate model
        print(f"  Evaluating model...")
        metrics = evaluate_model_on_dataset(model, X, y)
        
        if 'error' not in metrics:
            print(f"  ✅ Accuracy: {metrics['accuracy']:.3f}")
            print(f"  ✅ Macro F1: {metrics['macro_f1']:.3f}")
            print(f"  ✅ Weighted F1: {metrics['weighted_f1']:.3f}")
            all_metrics.append(metrics)
        
        results['validation_results'][dataset_path] = metrics
    
    # Generate summary statistics
    if all_metrics:
        summary_metrics = {}
        for metric_name in ['accuracy', 'macro_f1', 'weighted_f1', 'macro_precision', 'macro_recall']:
            values = [m[metric_name] for m in all_metrics if metric_name in m]
            if values:
                summary_metrics[f'mean_{metric_name}'] = np.mean(values)
                summary_metrics[f'std_{metric_name}'] = np.std(values)
                summary_metrics[f'min_{metric_name}'] = np.min(values)
                summary_metrics[f'max_{metric_name}'] = np.max(values)
        
        summary_metrics['n_datasets_tested'] = len(all_metrics)
        summary_metrics['total_samples'] = sum(m['n_samples'] for m in all_metrics)
        
        results['summary'] = summary_metrics
        
        print(f"\n📋 SUMMARY ACROSS {len(all_metrics)} DATASETS:")
        print("-" * 40)
        print(f"Mean Accuracy: {summary_metrics.get('mean_accuracy', 0):.3f} ± {summary_metrics.get('std_accuracy', 0):.3f}")
        print(f"Mean Macro F1: {summary_metrics.get('mean_macro_f1', 0):.3f} ± {summary_metrics.get('std_macro_f1', 0):.3f}")
        print(f"Mean Weighted F1: {summary_metrics.get('mean_weighted_f1', 0):.3f} ± {summary_metrics.get('std_weighted_f1', 0):.3f}")
        print(f"Total samples tested: {summary_metrics.get('total_samples', 0):,}")
    
    # Save results if output file specified
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n💾 Results saved to: {output_path}")
    
    return results


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(
        description="Cross-dataset validation utility for meta-models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test model on multiple datasets
  python cross_dataset_validator.py \\
    --model results/gene_cv_pc_5000_3mers_diverse_run1/model_multiclass.pkl \\
    --datasets train_pc_7000_3mers_opt/master train_pc_1000_3mers/master \\
    --output cross_validation_results.json

  # Test with sampling for large datasets
  python cross_dataset_validator.py \\
    --model results/gene_cv_pc_5000_3mers_diverse_run1/model_multiclass.pkl \\
    --datasets train_pc_7000_3mers_opt/master \\
    --sample-size 10000 \\
    --verbose
        """
    )
    
    parser.add_argument("--model", required=True,
                       help="Path to trained model (.pkl file)")
    parser.add_argument("--datasets", nargs="+", required=True,
                       help="List of dataset paths to test on")
    parser.add_argument("--sample-size", type=int, default=None,
                       help="Sample size for large datasets (default: use all data)")
    parser.add_argument("--output", default=None,
                       help="Output file for results (JSON format)")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose output")
    
    args = parser.parse_args()
    
    # Run cross-dataset validation
    results = run_cross_dataset_validation(
        model_path=args.model,
        test_datasets=args.datasets,
        sample_size=args.sample_size,
        output_file=args.output
    )
    
    # Check for errors
    if 'error' in results:
        print(f"❌ Validation failed: {results['error']}")
        sys.exit(1)
    
    # Print final status
    n_successful = len([r for r in results['validation_results'].values() if 'error' not in r])
    n_total = len(results['validation_results'])
    
    print(f"\n🎉 Cross-dataset validation completed!")
    print(f"Successfully tested on {n_successful}/{n_total} datasets")
    
    if n_successful == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()




