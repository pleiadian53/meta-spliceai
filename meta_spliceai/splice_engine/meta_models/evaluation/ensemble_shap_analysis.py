#!/usr/bin/env python3
"""
Comprehensive SHAP Analysis for Multi-Instance Ensemble Models

This module provides specialized SHAP analysis for ConsolidatedMetaModel instances,
computing feature importance across all instances and binary models.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, List
import json
import pickle

def run_comprehensive_ensemble_shap_analysis(
    dataset_path: str,
    model_path: str,
    out_dir: str,
    sample_size: int = 1000,
    background_size: int = 100,
    save_detailed_results: bool = True,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run comprehensive SHAP analysis on a ConsolidatedMetaModel.
    
    This function:
    1. Loads the ensemble model and dataset
    2. Computes SHAP values for all binary models across all instances
    3. Aggregates results with performance weighting
    4. Saves detailed analysis and visualizations
    
    Parameters
    ----------
    dataset_path : str
        Path to the dataset used for training
    model_path : str
        Path to the saved ConsolidatedMetaModel
    out_dir : str
        Output directory for results
    sample_size : int
        Number of samples to use for SHAP analysis
    background_size : int
        Number of background samples for SHAP explainer
    save_detailed_results : bool
        Whether to save detailed per-instance results
    verbose : bool
        Whether to print progress messages
        
    Returns
    -------
    dict
        Dictionary containing comprehensive SHAP results
    """
    
    if verbose:
        print("🔍 [Comprehensive SHAP] Starting ensemble SHAP analysis...")
        print(f"  Model: {model_path}")
        print(f"  Dataset: {dataset_path}")
        print(f"  Output: {out_dir}")
    
    out_path = Path(out_dir)
    shap_dir = out_path / "comprehensive_shap_analysis"
    shap_dir.mkdir(exist_ok=True, parents=True)
    
    try:
        # Load the consolidated model
        if verbose:
            print("  📦 Loading consolidated model...")
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        # Verify this is a ConsolidatedMetaModel
        if not hasattr(model, 'compute_ensemble_shap_importance'):
            raise ValueError("Model does not support comprehensive SHAP analysis")
        
        if verbose:
            print(f"  ✓ Loaded {type(model).__name__} with {model.n_instances} instances")
        
        # Load dataset sample for analysis
        if verbose:
            print(f"  📊 Loading dataset sample ({sample_size} samples)...")
        
        X_sample, feature_names = _load_dataset_sample(dataset_path, sample_size, model.feature_names)
        
        if verbose:
            print(f"  ✓ Loaded {X_sample.shape[0]} samples with {X_sample.shape[1]} features")
        
        # Run comprehensive SHAP analysis
        if verbose:
            print("  🧠 Computing comprehensive SHAP importance...")
        
        shap_results = model.compute_ensemble_shap_importance(
            X_sample, 
            background_data=X_sample[:background_size],
            max_evals=1000
        )
        
        if shap_results is None:
            raise RuntimeError("SHAP analysis failed - no results returned")
        
        # Save main results
        if verbose:
            print("  💾 Saving SHAP results...")
        
        # Save ensemble importance (main result)
        ensemble_importance_path = shap_dir / "ensemble_shap_importance.csv"
        shap_results['ensemble_importance'].to_csv(ensemble_importance_path)
        
        # Save per-class importance
        for class_name in ['neither', 'donor', 'acceptor']:
            class_importance_path = shap_dir / f"{class_name}_shap_importance.csv"
            shap_results[f'{class_name}_importance'].to_csv(class_importance_path)
        
        # Create summary report
        summary_report = _create_shap_summary_report(shap_results, model)
        summary_path = shap_dir / "shap_analysis_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary_report, f, indent=2)
        
        # Save detailed per-instance results if requested
        if save_detailed_results:
            if verbose:
                print("  📋 Saving detailed per-instance results...")
            
            instance_dir = shap_dir / "per_instance_results"
            instance_dir.mkdir(exist_ok=True)
            
            for instance_key, instance_data in shap_results['instance_contributions'].items():
                instance_path = instance_dir / f"{instance_key}_shap_importance.json"
                
                # Convert numpy arrays to lists for JSON serialization
                serializable_data = {}
                for class_name, importance_array in instance_data.items():
                    if isinstance(importance_array, np.ndarray):
                        serializable_data[class_name] = importance_array.tolist()
                    else:
                        serializable_data[class_name] = importance_array
                
                with open(instance_path, 'w') as f:
                    json.dump(serializable_data, f, indent=2)
        
        # Create visualizations
        if verbose:
            print("  📈 Creating SHAP visualizations...")
        
        _create_ensemble_shap_visualizations(shap_results, shap_dir, model)
        
        # Create compatibility file for existing SHAP infrastructure
        _create_legacy_shap_compatibility_file(shap_results, shap_dir)
        
        if verbose:
            print("✓ [Comprehensive SHAP] Analysis completed successfully!")
            print(f"  📁 Results saved to: {shap_dir}")
            print(f"  🎯 Top 5 features: {shap_results['ensemble_importance'].nlargest(5).index.tolist()}")
        
        return {
            'success': True,
            'output_directory': str(shap_dir),
            'ensemble_importance': shap_results['ensemble_importance'],
            'summary': summary_report,
            'top_features': shap_results['ensemble_importance'].nlargest(10).to_dict()
        }
        
    except Exception as e:
        error_msg = f"Comprehensive SHAP analysis failed: {str(e)}"
        if verbose:
            print(f"✗ [Comprehensive SHAP] {error_msg}")
        
        # Create error report
        error_path = shap_dir / "comprehensive_shap_failed.txt"
        with open(error_path, 'w') as f:
            f.write("Comprehensive SHAP Analysis Failed\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"Error: {error_msg}\n\n")
            
            import traceback
            f.write("Full traceback:\n")
            f.write(traceback.format_exc())
        
        return {
            'success': False,
            'error': error_msg,
            'output_directory': str(shap_dir)
        }


def _load_dataset_sample(dataset_path: str, sample_size: int, feature_names: List[str]) -> tuple:
    """Load a sample of the dataset for SHAP analysis."""
    try:
        # Use memory-efficient sampling approach
        import os
        
        # Set row cap to prevent loading full dataset
        original_row_cap = os.environ.get("SS_MAX_ROWS")
        os.environ["SS_MAX_ROWS"] = str(min(sample_size * 10, 50000))  # Load at most 50k rows
        
        try:
            from meta_spliceai.splice_engine.meta_models.training import datasets
            from meta_spliceai.splice_engine.meta_models.builder import preprocessing
            
            # Load dataset with row cap
            df = datasets.load_dataset(dataset_path)
            
            # Sample if needed
            if len(df) > sample_size:
                # Handle both pandas and polars DataFrames
                if hasattr(df, 'sample'):
                    try:
                        # Try pandas-style sampling first
                        df = df.sample(n=sample_size, random_state=42)
                    except TypeError:
                        # Fall back to polars-style sampling
                        df = df.sample(n=sample_size, seed=42)
                else:
                    # If no sample method, just take first N rows
                    df = df.head(sample_size)
            
            # Prepare features
            X_df, _ = preprocessing.prepare_training_data(
                df, 
                label_col="splice_type", 
                return_type="pandas",
                verbose=0
            )
            
            # Ensure we have the same features as the model
            available_features = [f for f in feature_names if f in X_df.columns]
            X_df = X_df[available_features]
            
            return X_df.values, available_features
            
        finally:
            # Restore original row cap
            if original_row_cap is not None:
                os.environ["SS_MAX_ROWS"] = original_row_cap
            else:
                os.environ.pop("SS_MAX_ROWS", None)
        
    except Exception as e:
        raise RuntimeError(f"Failed to load dataset sample: {e}")


def _create_shap_summary_report(shap_results: Dict[str, Any], model) -> Dict[str, Any]:
    """Create a comprehensive summary report of SHAP analysis."""
    
    ensemble_importance = shap_results['ensemble_importance']
    
    summary = {
        'analysis_type': 'comprehensive_ensemble_shap',
        'model_type': type(model).__name__,
        'n_instances': model.n_instances,
        'n_features': len(ensemble_importance),
        'instance_weights': model.instance_weights,
        
        # Top features analysis
        'top_10_features': ensemble_importance.nlargest(10).to_dict(),
        'top_features_names': ensemble_importance.nlargest(10).index.tolist(),
        
        # Feature type analysis
        'feature_type_analysis': _analyze_feature_types(ensemble_importance),
        
        # Per-class summaries
        'class_summaries': {
            'neither': {
                'top_5_features': shap_results['neither_importance'].nlargest(5).to_dict(),
                'mean_importance': float(shap_results['neither_importance'].mean())
            },
            'donor': {
                'top_5_features': shap_results['donor_importance'].nlargest(5).to_dict(),
                'mean_importance': float(shap_results['donor_importance'].mean())
            },
            'acceptor': {
                'top_5_features': shap_results['acceptor_importance'].nlargest(5).to_dict(),
                'mean_importance': float(shap_results['acceptor_importance'].mean())
            }
        },
        
        # Statistical summary
        'importance_statistics': {
            'mean': float(ensemble_importance.mean()),
            'std': float(ensemble_importance.std()),
            'max': float(ensemble_importance.max()),
            'min': float(ensemble_importance.min()),
            'median': float(ensemble_importance.median())
        }
    }
    
    return summary


def _analyze_feature_types(importance_series: pd.Series) -> Dict[str, Any]:
    """Analyze importance by feature types (k-mers vs non-k-mers)."""
    
    # Categorize features
    kmer_features = []
    non_kmer_features = []
    
    for feature in importance_series.index:
        if any(feature.startswith(prefix) for prefix in ['3mer_', '5mer_', '7mer_']):
            kmer_features.append(feature)
        else:
            non_kmer_features.append(feature)
    
    analysis = {
        'n_kmer_features': len(kmer_features),
        'n_non_kmer_features': len(non_kmer_features),
        'kmer_importance_mean': float(importance_series[kmer_features].mean()) if kmer_features else 0.0,
        'non_kmer_importance_mean': float(importance_series[non_kmer_features].mean()) if non_kmer_features else 0.0,
        'top_kmer_features': importance_series[kmer_features].nlargest(5).to_dict() if kmer_features else {},
        'top_non_kmer_features': importance_series[non_kmer_features].nlargest(5).to_dict() if non_kmer_features else {}
    }
    
    return analysis


def _create_ensemble_shap_visualizations(shap_results: Dict[str, Any], output_dir: Path, model):
    """Create visualizations for ensemble SHAP results."""
    
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Top features importance plot
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Comprehensive Ensemble SHAP Analysis', fontsize=16, fontweight='bold')
        
        # Top 20 ensemble features
        top_features = shap_results['ensemble_importance'].nlargest(20)
        axes[0, 0].barh(range(len(top_features)), top_features.values)
        axes[0, 0].set_yticks(range(len(top_features)))
        axes[0, 0].set_yticklabels(top_features.index, fontsize=8)
        axes[0, 0].set_xlabel('SHAP Importance')
        axes[0, 0].set_title('Top 20 Ensemble Features')
        axes[0, 0].invert_yaxis()
        
        # Per-class comparison for top 10 features
        top_10_features = shap_results['ensemble_importance'].nlargest(10).index
        class_data = pd.DataFrame({
            'Neither': shap_results['neither_importance'][top_10_features],
            'Donor': shap_results['donor_importance'][top_10_features],
            'Acceptor': shap_results['acceptor_importance'][top_10_features]
        })
        
        class_data.plot(kind='bar', ax=axes[0, 1])
        axes[0, 1].set_title('Top 10 Features by Class')
        axes[0, 1].set_xlabel('Features')
        axes[0, 1].set_ylabel('SHAP Importance')
        axes[0, 1].tick_params(axis='x', rotation=45, labelsize=8)
        axes[0, 1].legend()
        
        # Feature type analysis
        kmer_features = [f for f in shap_results['ensemble_importance'].index 
                        if any(f.startswith(p) for p in ['3mer_', '5mer_', '7mer_'])]
        non_kmer_features = [f for f in shap_results['ensemble_importance'].index 
                           if f not in kmer_features]
        
        feature_type_importance = [
            shap_results['ensemble_importance'][kmer_features].mean() if kmer_features else 0,
            shap_results['ensemble_importance'][non_kmer_features].mean() if non_kmer_features else 0
        ]
        
        axes[1, 0].bar(['K-mer Features', 'Non K-mer Features'], feature_type_importance)
        axes[1, 0].set_title('Average Importance by Feature Type')
        axes[1, 0].set_ylabel('Mean SHAP Importance')
        
        # Instance contribution comparison
        if len(model.instance_weights) > 1:
            axes[1, 1].bar(range(len(model.instance_weights)), model.instance_weights)
            axes[1, 1].set_title('Instance Performance Weights')
            axes[1, 1].set_xlabel('Instance ID')
            axes[1, 1].set_ylabel('Weight')
            axes[1, 1].set_xticks(range(len(model.instance_weights)))
        
        plt.tight_layout()
        plt.savefig(output_dir / 'ensemble_shap_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Detailed per-class feature importance
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        for i, class_name in enumerate(['neither', 'donor', 'acceptor']):
            top_class_features = shap_results[f'{class_name}_importance'].nlargest(15)
            axes[i].barh(range(len(top_class_features)), top_class_features.values)
            axes[i].set_yticks(range(len(top_class_features)))
            axes[i].set_yticklabels(top_class_features.index, fontsize=8)
            axes[i].set_xlabel('SHAP Importance')
            axes[i].set_title(f'Top 15 {class_name.title()} Features')
            axes[i].invert_yaxis()
        
        plt.tight_layout()
        plt.savefig(output_dir / 'per_class_shap_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("  ✓ SHAP visualizations created")
        
    except Exception as e:
        print(f"  ⚠️  Failed to create visualizations: {e}")


def _create_legacy_shap_compatibility_file(shap_results: Dict[str, Any], output_dir: Path):
    """Create a compatibility file for existing SHAP infrastructure."""
    
    # Create the expected shap_importance_incremental.csv file
    legacy_path = output_dir / "shap_importance_incremental.csv"
    
    # Format as expected by existing visualization code
    legacy_df = pd.DataFrame({
        'feature': shap_results['ensemble_importance'].index,
        'importance_neither': shap_results['neither_importance'].values,
        'importance_donor': shap_results['donor_importance'].values,
        'importance_acceptor': shap_results['acceptor_importance'].values,
        'importance_mean': shap_results['ensemble_importance'].values,
        'shap_importance': shap_results['ensemble_importance'].values  # Backward compatibility
    })
    
    legacy_df.to_csv(legacy_path, index=False)
    print("  ✓ Legacy compatibility file created")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run comprehensive SHAP analysis on ensemble model")
    parser.add_argument("--dataset", required=True, help="Dataset path")
    parser.add_argument("--model", required=True, help="Model path")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    parser.add_argument("--sample-size", type=int, default=1000, help="Sample size for analysis")
    parser.add_argument("--background-size", type=int, default=100, help="Background sample size")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    results = run_comprehensive_ensemble_shap_analysis(
        dataset_path=args.dataset,
        model_path=args.model,
        out_dir=args.out_dir,
        sample_size=args.sample_size,
        background_size=args.background_size,
        verbose=args.verbose
    )
    
    if results['success']:
        print(f"\n✅ Analysis completed successfully!")
        print(f"📁 Results: {results['output_directory']}")
        print(f"🎯 Top features: {list(results['top_features'].keys())[:5]}")
    else:
        print(f"\n❌ Analysis failed: {results['error']}")
