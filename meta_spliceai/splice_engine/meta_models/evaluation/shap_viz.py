#!/usr/bin/env python3
"""
SHAP Feature Importance Visualization Module.

This module provides comprehensive visualization tools for SHAP-based feature importance
analysis in splice site classification, including bar charts and beeswarm plots.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union, Any
import warnings


def create_feature_importance_barcharts(
    importance_csv: Union[str, Path],
    out_dir: Union[str, Path],
    top_n: int = 20,
    plot_format: str = "png",
    figsize: Tuple[int, int] = (12, 8),
    color_palette: Optional[List[str]] = None
) -> Dict[str, Path]:
    """
    Create bar charts for feature importance from SHAP incremental analysis.
    
    Parameters
    ----------
    importance_csv : str or Path
        Path to the shap_importance_incremental.csv file
    out_dir : str or Path
        Output directory for plots
    top_n : int
        Number of top features to display
    plot_format : str
        File format for plots (png, pdf, svg)
    figsize : Tuple[int, int]
        Figure size (width, height)
    color_palette : List[str], optional
        Custom color palette for the plots
        
    Returns
    -------
    Dict[str, Path]
        Dictionary mapping plot names to file paths
    """
    # Load importance data
    df = pd.read_csv(importance_csv)
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    
    if color_palette is None:
        color_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Blue, Orange, Green, Red
    
    plot_paths = {}
    
    # CRITICAL FIX: Check if we have per-class columns or simple format
    has_per_class_columns = all(col in df.columns for col in ['importance_neither', 'importance_donor', 'importance_acceptor'])
    
    if has_per_class_columns:
        # 1. Individual class importance bar charts
        class_columns = ['importance_neither', 'importance_donor', 'importance_acceptor']
        class_names = ['Neither', 'Donor', 'Acceptor']
        
        for i, (col, name) in enumerate(zip(class_columns, class_names)):
            fig, ax = plt.subplots(figsize=figsize)
            
            # Get top N features for this class
            top_features = df.nlargest(top_n, col)
            
            # Create horizontal bar chart
            bars = ax.barh(range(len(top_features)), top_features[col], 
                          color=color_palette[i], alpha=0.8)
            
            # Customize plot
            ax.set_yticks(range(len(top_features)))
            ax.set_yticklabels(top_features['feature'], fontsize=10)
            ax.set_xlabel('SHAP Importance', fontsize=12)
            ax.set_title(f'Top {top_n} Features - {name} Classification', fontsize=14, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
            
            # Add value labels on bars
            for j, bar in enumerate(bars):
                width = bar.get_width()
                ax.text(width + 0.001, bar.get_y() + bar.get_height()/2, 
                       f'{width:.4f}', ha='left', va='center', fontsize=9)
            
            # Invert y-axis to show highest importance at top
            ax.invert_yaxis()
            
            plt.tight_layout()
            
            # Save plot
            plot_path = out_dir / f"feature_importance_barchart_{name.lower()}.{plot_format}"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            plot_paths[f'{name.lower()}_barchart'] = plot_path
            print(f"✓ Created {name} feature importance bar chart: {plot_path}")
        
        # 2. Mean importance bar chart
        fig, ax = plt.subplots(figsize=figsize)
        
        top_features_mean = df.nlargest(top_n, 'importance_mean')
        
        bars = ax.barh(range(len(top_features_mean)), top_features_mean['importance_mean'], 
                      color=color_palette[3], alpha=0.8)
        
        ax.set_yticks(range(len(top_features_mean)))
        ax.set_yticklabels(top_features_mean['feature'], fontsize=10)
        ax.set_xlabel('Mean SHAP Importance', fontsize=12)
        ax.set_title(f'Top {top_n} Features - Mean Importance Across All Classes', fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for j, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width + 0.001, bar.get_y() + bar.get_height()/2, 
                   f'{width:.4f}', ha='left', va='center', fontsize=9)
        
        ax.invert_yaxis()
        plt.tight_layout()
        
        # Save plot
        plot_path = out_dir / f"feature_importance_barchart_mean.{plot_format}"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        plot_paths['mean_barchart'] = plot_path
        print(f"✓ Created mean feature importance bar chart: {plot_path}")
        
        # 3. Comparison bar chart (all classes side by side)
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Get top features by mean importance
        top_features_comparison = df.nlargest(top_n, 'importance_mean')
        
        # Prepare data for grouped bar chart
        x = np.arange(len(top_features_comparison))
        width = 0.25
        
        bars1 = ax.bar(x - width, top_features_comparison['importance_neither'], width, 
                       label='Neither', color=color_palette[0], alpha=0.8)
        bars2 = ax.bar(x, top_features_comparison['importance_donor'], width, 
                       label='Donor', color=color_palette[1], alpha=0.8)
        bars3 = ax.bar(x + width, top_features_comparison['importance_acceptor'], width, 
                       label='Acceptor', color=color_palette[2], alpha=0.8)
        
        ax.set_xlabel('Features', fontsize=12)
        ax.set_ylabel('SHAP Importance', fontsize=12)
        ax.set_title(f'Feature Importance Comparison - Top {top_n} Features', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(top_features_comparison['feature'], rotation=45, ha='right', fontsize=10)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = out_dir / f"feature_importance_comparison.{plot_format}"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        plot_paths['comparison_barchart'] = plot_path
        print(f"✓ Created feature importance comparison chart: {plot_path}")
    else:
        # SIMPLE FORMAT: Only 'feature' and 'shap_importance' columns available
        print(f"⚠️  Simple SHAP format detected - creating basic importance bar chart")
        
        # Check if we have the expected simple columns
        if 'shap_importance' not in df.columns:
            # Try alternative column names
            importance_cols = [col for col in df.columns if 'importance' in col.lower() or 'shap' in col.lower()]
            if importance_cols:
                importance_col = importance_cols[0]
                print(f"Using column '{importance_col}' for importance values")
            else:
                print(f"✗ Error: No recognizable importance column found in {df.columns.tolist()}")
                return plot_paths
        else:
            importance_col = 'shap_importance'
        
        # Create single importance bar chart
        fig, ax = plt.subplots(figsize=figsize)
        
        top_features = df.nlargest(top_n, importance_col)
        
        bars = ax.barh(range(len(top_features)), top_features[importance_col], 
                      color=color_palette[0], alpha=0.8)
        
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['feature'], fontsize=10)
        ax.set_xlabel('SHAP Importance', fontsize=12)
        ax.set_title(f'Top {top_n} Features - SHAP Importance', fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for j, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width + 0.001, bar.get_y() + bar.get_height()/2, 
                   f'{width:.4f}', ha='left', va='center', fontsize=9)
        
        ax.invert_yaxis()
        plt.tight_layout()
        
        # Save plot
        plot_path = out_dir / f"feature_importance_barchart_simple.{plot_format}"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        plot_paths['simple_barchart'] = plot_path
        print(f"✓ Created simple feature importance bar chart: {plot_path}")
    
    return plot_paths


def create_shap_beeswarm_plots(
    model_path: Union[str, Path],
    dataset_path: Union[str, Path],
    out_dir: Union[str, Path],
    sample_size: int = 1000,
    top_n_features: int = 20,
    plot_format: str = "png",
    figsize: Tuple[int, int] = (12, 8)
) -> Dict[str, Path]:
    """
    Create SHAP beeswarm summary plots for each class in the sigmoid ensemble.
    
    Parameters
    ----------
    model_path : str or Path
        Path to the trained model (.pkl file)
    dataset_path : str or Path
        Path to the dataset for SHAP analysis
    out_dir : str or Path
        Output directory for plots
    sample_size : int
        Number of samples to use for SHAP analysis
    top_n_features : int
        Number of top features to display
    plot_format : str
        File format for plots
    figsize : Tuple[int, int]
        Figure size
        
    Returns
    -------
    Dict[str, Path]
        Dictionary mapping plot names to file paths
    """
    import pickle
    import os
    import sys
    import warnings
    
    # CRITICAL: Set environment variables BEFORE importing SHAP to prevent dependency conflicts
    os.environ.update({
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1", 
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_CACHE": "/tmp/transformers_cache_disabled",
        "HF_HOME": "/tmp/hf_home_disabled",
        "CUDA_VISIBLE_DEVICES": "",
        "TF_CPP_MIN_LOG_LEVEL": "3",
        "PYTHONWARNINGS": "ignore",
        "MLFLOW_TRACKING_URI": "",
        "WANDB_DISABLED": "true",
        "COMET_DISABLE": "true"
    })
    
    # Comprehensive warning suppression
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning) 
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore")
    
    # Redirect stderr temporarily to suppress TensorFlow messages
    old_stderr = sys.stderr
    try:
        import io
        sys.stderr = io.StringIO()
    finally:
        # Restore stderr
        sys.stderr = old_stderr
    
    # Load model
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # Load and prepare data
    from meta_spliceai.splice_engine.meta_models.builder.preprocessing import prepare_training_data
    from meta_spliceai.splice_engine.meta_models.training.datasets import load_dataset
    
    print(f"Loading dataset from {dataset_path}...")
    df = load_dataset(dataset_path)
    
    # Sample data for SHAP analysis
    if len(df) > sample_size:
        try:
            df = df.sample(n=sample_size, random_state=42)
        except TypeError:
            # Fallback for older pandas versions
            df = df.sample(n=sample_size)
        print(f"Sampled {sample_size} rows for SHAP analysis")
    
    # Prepare features with chromosome encoding
    X_df, y_series = prepare_training_data(
        df, 
        label_col="splice_type", 
        return_type="pandas", 
        verbose=0,
        encode_chrom=True  # Enable chromosome encoding
    )
    
    # Ensure all data is numeric
    for col in X_df.columns:
        if not pd.api.types.is_numeric_dtype(X_df[col]):
            X_df[col] = pd.to_numeric(X_df[col], errors='coerce').fillna(0)
        elif X_df[col].dtype == object:
            # Handle object columns that might contain strings
            unique_values = X_df[col].dropna().unique()
            value_map = {val: idx for idx, val in enumerate(unique_values)}
            X_df[col] = X_df[col].map(value_map).fillna(0)
    
    X = X_df.values.astype(np.float32)  # Use float32 for memory efficiency
    feature_names = list(X_df.columns)
    
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    plot_paths = {}
    
    # Check if model is a sigmoid ensemble with binary models
    if hasattr(model, 'models') and len(model.models) == 3:
        print("Detected sigmoid ensemble - creating beeswarm plots for each binary classifier")
        
        class_names = ['Neither', 'Donor', 'Acceptor']
        
        for i, (binary_model, class_name) in enumerate(zip(model.models, class_names)):
            print(f"Creating beeswarm plot for {class_name} classifier...")
            
            try:
                # Create SHAP explainer for this binary model with enhanced error handling
                background_size = min(100, len(X))  # Small background for memory efficiency
                background_idx = np.random.choice(len(X), size=background_size, replace=False)
                background = X[background_idx]
                
                # Additional environment isolation before TreeExplainer creation
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    
                    # Try to create explainer with minimal configuration first
                    try:
                        explainer = shap.TreeExplainer(
                            binary_model,
                            data=background,
                            feature_perturbation="interventional",
                            model_output="probability",
                            approximate=True
                        )
                    except Exception as explainer_error:
                        print(f"✗ Error creating SHAP explainer for {class_name}: {explainer_error}")
                        # Try simpler explainer configuration
                        try:
                            explainer = shap.TreeExplainer(binary_model, data=background)
                        except Exception as simple_explainer_error:
                            print(f"✗ Error creating simple SHAP explainer for {class_name}: {simple_explainer_error}")
                            continue
                
                # Calculate SHAP values in batches to manage memory
                batch_size = min(200, len(X))
                
                # CRITICAL FIX: Ensure batch_size is never 0
                if batch_size == 0 or len(X) == 0:
                    print(f"✗ Error: No data available for {class_name} SHAP analysis (len(X)={len(X)})")
                    continue
                
                all_shap_values = []
                
                for start_idx in range(0, len(X), batch_size):
                    end_idx = min(start_idx + batch_size, len(X))
                    batch_X = X[start_idx:end_idx]
                    
                    batch_shap_values = explainer.shap_values(batch_X)
                    
                    # For binary classification, get the positive class SHAP values
                    if isinstance(batch_shap_values, list):
                        batch_shap_values = batch_shap_values[1]  # Positive class
                    
                    all_shap_values.append(batch_shap_values)
                
                # Combine all batches
                shap_values = np.vstack(all_shap_values)
                
                # Get top features by importance
                feature_importance = np.abs(shap_values).mean(axis=0)
                top_features_idx = np.argsort(feature_importance)[-top_n_features:][::-1]
                
                # Create beeswarm plot with only top features
                plt.figure(figsize=figsize)
                shap.summary_plot(
                    shap_values[:, top_features_idx], 
                    X[:, top_features_idx], 
                    feature_names=[feature_names[idx] for idx in top_features_idx],
                    max_display=top_n_features,
                    show=False,
                    plot_type="dot"  # This creates the beeswarm plot
                )
                
                plt.title(f'SHAP Summary - {class_name} Classification', fontsize=14, fontweight='bold')
                plt.tight_layout()
                
                # Save plot
                plot_path = out_dir / f"shap_beeswarm_{class_name.lower()}.{plot_format}"
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                plot_paths[f'{class_name.lower()}_beeswarm'] = plot_path
                print(f"✓ Created {class_name} beeswarm plot: {plot_path}")
                
            except Exception as e:
                print(f"✗ Error creating beeswarm plot for {class_name}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    elif hasattr(model, 'get_base_models'):
        # Handle SigmoidEnsemble with get_base_models method
        print("Detected SigmoidEnsemble - creating beeswarm plots for each binary classifier")
        
        binary_models = model.get_base_models()
        class_names = ['Neither', 'Donor', 'Acceptor']
        
        for i, (binary_model, class_name) in enumerate(zip(binary_models, class_names)):
            print(f"Creating beeswarm plot for {class_name} classifier...")
            
            try:
                # Create SHAP explainer for this binary model with enhanced error handling
                background_size = min(100, len(X))
                background_idx = np.random.choice(len(X), size=background_size, replace=False)
                background = X[background_idx]
                
                # Additional environment isolation before TreeExplainer creation
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    
                    # Try to create explainer with minimal configuration first
                    try:
                        explainer = shap.TreeExplainer(
                            binary_model,
                            data=background,
                            feature_perturbation="interventional",
                            model_output="probability",
                            approximate=True
                        )
                    except Exception as explainer_error:
                        print(f"✗ Error creating SHAP explainer for {class_name}: {explainer_error}")
                        # Try simpler explainer configuration
                        try:
                            explainer = shap.TreeExplainer(binary_model, data=background)
                        except Exception as simple_explainer_error:
                            print(f"✗ Error creating simple SHAP explainer for {class_name}: {simple_explainer_error}")
                            continue
                
                # Calculate SHAP values in batches
                batch_size = min(200, len(X))
                
                # CRITICAL FIX: Ensure batch_size is never 0
                if batch_size == 0 or len(X) == 0:
                    print(f"✗ Error: No data available for {class_name} SHAP analysis (len(X)={len(X)})")
                    continue
                
                all_shap_values = []
                
                for start_idx in range(0, len(X), batch_size):
                    end_idx = min(start_idx + batch_size, len(X))
                    batch_X = X[start_idx:end_idx]
                    
                    batch_shap_values = explainer.shap_values(batch_X)
                    
                    # For binary classification, get the positive class SHAP values
                    if isinstance(batch_shap_values, list):
                        batch_shap_values = batch_shap_values[1]  # Positive class
                    
                    all_shap_values.append(batch_shap_values)
                
                # Combine all batches
                shap_values = np.vstack(all_shap_values)
                
                # Get top features by importance
                feature_importance = np.abs(shap_values).mean(axis=0)
                top_features_idx = np.argsort(feature_importance)[-top_n_features:][::-1]
                
                # Create beeswarm plot
                plt.figure(figsize=figsize)
                shap.summary_plot(
                    shap_values[:, top_features_idx], 
                    X[:, top_features_idx], 
                    feature_names=[feature_names[idx] for idx in top_features_idx],
                    max_display=top_n_features,
                    show=False,
                    plot_type="dot"
                )
                
                plt.title(f'SHAP Summary - {class_name} Classification', fontsize=14, fontweight='bold')
                plt.tight_layout()
                
                # Save plot
                plot_path = out_dir / f"shap_beeswarm_{class_name.lower()}.{plot_format}"
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                plot_paths[f'{class_name.lower()}_beeswarm'] = plot_path
                print(f"✓ Created {class_name} beeswarm plot: {plot_path}")
                
            except Exception as e:
                print(f"✗ Error creating beeswarm plot for {class_name}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    else:
        print("Model is not a sigmoid ensemble - creating single beeswarm plot")
        
        try:
            # Create explainer for the full model with enhanced error handling
            background_size = min(100, len(X))
            background_idx = np.random.choice(len(X), size=background_size, replace=False)
            background = X[background_idx]
            
            # Additional environment isolation before TreeExplainer creation
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                # Try to create explainer with minimal configuration first
                try:
                    explainer = shap.TreeExplainer(
                        model,
                        data=background,
                        feature_perturbation="interventional",
                        model_output="probability",
                        approximate=True
                    )
                except Exception as explainer_error:
                    print(f"✗ Error creating SHAP explainer: {explainer_error}")
                    # Try simpler explainer configuration
                    try:
                        explainer = shap.TreeExplainer(model, data=background)
                    except Exception as simple_explainer_error:
                        print(f"✗ Error creating simple SHAP explainer: {simple_explainer_error}")
                        # Create a fallback error plot
                        fig, ax = plt.subplots(figsize=figsize)
                        ax.text(0.5, 0.5, 'SHAP beeswarm plot failed\ndue to dependency conflicts.\n\nSee shap_analysis_failed.txt for details.', 
                               ha='center', va='center', fontsize=12, 
                               bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.5))
                        ax.set_xlim(0, 1)
                        ax.set_ylim(0, 1)
                        ax.axis('off')
                        ax.set_title('SHAP Summary - Multi-class Classification (Failed)', fontsize=14, fontweight='bold')
                        
                        plot_path = out_dir / f"shap_beeswarm_multiclass_failed.{plot_format}"
                        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                        plt.close()
                        
                        plot_paths['multiclass_beeswarm'] = plot_path
                        print(f"✓ Created fallback plot: {plot_path}")
                        return plot_paths
            
            # Calculate SHAP values in batches
            batch_size = min(200, len(X))
            
            # CRITICAL FIX: Ensure batch_size is never 0
            if batch_size == 0 or len(X) == 0:
                print(f"✗ Error: No data available for multiclass SHAP analysis (len(X)={len(X)})")
                # Create a fallback error plot
                fig, ax = plt.subplots(figsize=figsize)
                ax.text(0.5, 0.5, 'SHAP beeswarm plot failed\ndue to insufficient data.\n\nTry increasing sample_size.', 
                       ha='center', va='center', fontsize=12, 
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.5))
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.axis('off')
                ax.set_title('SHAP Summary - Multi-class Classification (No Data)', fontsize=14, fontweight='bold')
                
                plot_path = out_dir / f"shap_beeswarm_multiclass_nodata.{plot_format}"
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                plot_paths['multiclass_beeswarm'] = plot_path
                print(f"✓ Created no-data fallback plot: {plot_path}")
                return plot_paths
            
            all_shap_values = []
            
            for start_idx in range(0, len(X), batch_size):
                end_idx = min(start_idx + batch_size, len(X))
                batch_X = X[start_idx:end_idx]
                
                batch_shap_values = explainer.shap_values(batch_X)
                all_shap_values.append(batch_shap_values)
            
            # Combine all batches
            if isinstance(all_shap_values[0], list):
                # Multi-class case
                shap_values = [np.vstack([batch[i] for batch in all_shap_values]) for i in range(len(all_shap_values[0]))]
            else:
                shap_values = np.vstack(all_shap_values)
            
            # Create beeswarm plot
            plt.figure(figsize=figsize)
            shap.summary_plot(
                shap_values, 
                X, 
                feature_names=feature_names,
                max_display=top_n_features,
                show=False,
                plot_type="dot"
            )
            
            plt.title('SHAP Summary - Multi-class Classification', fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            # Save plot
            plot_path = out_dir / f"shap_beeswarm_multiclass.{plot_format}"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            plot_paths['multiclass_beeswarm'] = plot_path
            print(f"✓ Created multiclass beeswarm plot: {plot_path}")
            
        except Exception as e:
            print(f"✗ Error creating beeswarm plot: {e}")
            import traceback
            traceback.print_exc()
    
    return plot_paths


def create_feature_importance_heatmap(
    importance_csv: Union[str, Path],
    out_dir: Union[str, Path],
    top_n: int = 30,
    plot_format: str = "png",
    figsize: Tuple[int, int] = (10, 12)
) -> Path:
    """
    Create a heatmap showing feature importance across all three classes.
    
    Parameters
    ----------
    importance_csv : str or Path
        Path to the shap_importance_incremental.csv file
    out_dir : str or Path
        Output directory for plots
    top_n : int
        Number of top features to display
    plot_format : str
        File format for plots
    figsize : Tuple[int, int]
        Figure size
        
    Returns
    -------
    Path
        Path to the saved heatmap
    """
    # Load importance data
    df = pd.read_csv(importance_csv)
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    
    # CRITICAL FIX: Check if we have per-class columns or simple format
    has_per_class_columns = all(col in df.columns for col in ['importance_neither', 'importance_donor', 'importance_acceptor'])
    
    if has_per_class_columns:
        # Get top features by mean importance
        top_features = df.nlargest(top_n, 'importance_mean')
        
        # Prepare data for heatmap
        heatmap_data = top_features[['importance_neither', 'importance_donor', 'importance_acceptor']].values
        feature_labels = top_features['feature'].values
        class_labels = ['Neither', 'Donor', 'Acceptor']
    else:
        # SIMPLE FORMAT: Create a simple heatmap with single importance column
        print(f"⚠️  Simple SHAP format detected - creating basic feature importance plot instead of heatmap")
        
        # Check for importance column
        if 'shap_importance' not in df.columns:
            importance_cols = [col for col in df.columns if 'importance' in col.lower() or 'shap' in col.lower()]
            if importance_cols:
                importance_col = importance_cols[0]
            else:
                print(f"✗ Error: No recognizable importance column found in {df.columns.tolist()}")
                return None
        else:
            importance_col = 'shap_importance'
        
        # Get top features
        top_features = df.nlargest(top_n, importance_col)
        
        # Create a single-column heatmap (actually just a bar chart in horizontal format)
        heatmap_data = top_features[importance_col].values.reshape(-1, 1)
        feature_labels = top_features['feature'].values
        class_labels = ['SHAP Importance']
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap with custom colormap
    sns.heatmap(
        heatmap_data, 
        xticklabels=class_labels,
        yticklabels=feature_labels,
        annot=True, 
        fmt='.4f',
        cmap='YlOrRd',
        cbar_kws={'label': 'SHAP Importance'},
        ax=ax
    )
    
    ax.set_title(f'Feature Importance Heatmap - Top {top_n} Features', fontsize=14, fontweight='bold')
    ax.set_xlabel('Classification Task', fontsize=12)
    ax.set_ylabel('Features', fontsize=12)
    
    # Rotate feature names for better readability
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=10)
    plt.tight_layout()
    
    # Save plot
    plot_path = out_dir / f"feature_importance_heatmap.{plot_format}"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Created feature importance heatmap: {plot_path}")
    return plot_path


def generate_comprehensive_shap_report(
    importance_csv: Union[str, Path],
    model_path: Union[str, Path],
    dataset_path: Union[str, Path],
    out_dir: Union[str, Path],
    top_n: int = 20,
    sample_size: int = 1000,
    plot_format: str = "png"
) -> Dict[str, Any]:
    """
    Generate a comprehensive SHAP analysis report with all visualization types.
    
    Parameters
    ----------
    importance_csv : str or Path
        Path to the shap_importance_incremental.csv file
    model_path : str or Path
        Path to the trained model
    dataset_path : str or Path
        Path to the dataset
    out_dir : str or Path
        Output directory for all plots
    top_n : int
        Number of top features to display
    sample_size : int
        Sample size for beeswarm plots
    plot_format : str
        File format for plots
        
    Returns
    -------
    Dict[str, Any]
        Summary of generated plots and statistics
    """
    out_dir = Path(out_dir)
    
    # Organize SHAP outputs under feature_importance_analysis for consistency
    feature_importance_dir = out_dir / "feature_importance_analysis"
    feature_importance_dir.mkdir(exist_ok=True, parents=True)
    
    shap_analysis_dir = feature_importance_dir / "shap_analysis"
    shap_analysis_dir.mkdir(exist_ok=True, parents=True)
    
    shap_viz_dir = shap_analysis_dir / "comprehensive_visualizations"
    shap_viz_dir.mkdir(exist_ok=True, parents=True)
    
    print("=" * 60)
    print("COMPREHENSIVE SHAP ANALYSIS REPORT")
    print("=" * 60)
    print(f"Saving outputs to: {shap_analysis_dir}")
    
    results = {
        'bar_charts': {},
        'beeswarm_plots': {},
        'heatmap': None,
        'summary_stats': {}
    }
    
    # 1. Create bar charts
    print("\n1. Creating feature importance bar charts...")
    try:
        bar_chart_paths = create_feature_importance_barcharts(
            importance_csv, shap_viz_dir, top_n=top_n, plot_format=plot_format
        )
        results['bar_charts'] = bar_chart_paths
    except Exception as e:
        print(f"✗ Error creating bar charts: {e}")
    
    # 2. Create beeswarm plots
    print("\n2. Creating SHAP beeswarm plots...")
    try:
        beeswarm_paths = create_shap_beeswarm_plots(
            model_path, dataset_path, shap_viz_dir, 
            sample_size=sample_size, top_n_features=top_n, plot_format=plot_format
        )
        results['beeswarm_plots'] = beeswarm_paths
    except Exception as e:
        print(f"✗ Error creating beeswarm plots: {e}")
        import traceback
        traceback.print_exc()
    
    # 3. Create heatmap
    print("\n3. Creating feature importance heatmap...")
    try:
        heatmap_path = create_feature_importance_heatmap(
            importance_csv, shap_viz_dir, top_n=top_n*2, plot_format=plot_format  # More features for heatmap
        )
        results['heatmap'] = heatmap_path
    except Exception as e:
        print(f"✗ Error creating heatmap: {e}")
    
    # 4. Generate summary statistics
    print("\n4. Generating summary statistics...")
    try:
        df = pd.read_csv(importance_csv)
        
        # CRITICAL FIX: Check format and generate appropriate stats
        has_per_class_columns = all(col in df.columns for col in ['importance_neither', 'importance_donor', 'importance_acceptor'])
        
        if has_per_class_columns:
            results['summary_stats'] = {
                'total_features': len(df),
                'top_feature_overall': df.loc[df['importance_mean'].idxmax(), 'feature'],
                'top_feature_neither': df.loc[df['importance_neither'].idxmax(), 'feature'],
                'top_feature_donor': df.loc[df['importance_donor'].idxmax(), 'feature'],
                'top_feature_acceptor': df.loc[df['importance_acceptor'].idxmax(), 'feature'],
                'mean_importance_range': f"{df['importance_mean'].min():.6f} - {df['importance_mean'].max():.6f}",
                'features_with_zero_importance': len(df[df['importance_mean'] == 0])
            }
        else:
            # SIMPLE FORMAT: Generate basic stats
            importance_col = 'shap_importance' if 'shap_importance' in df.columns else df.columns[1]
            
            results['summary_stats'] = {
                'total_features': len(df),
                'top_feature_overall': df.loc[df[importance_col].idxmax(), 'feature'],
                'top_feature_neither': 'N/A (simple format)',
                'top_feature_donor': 'N/A (simple format)', 
                'top_feature_acceptor': 'N/A (simple format)',
                'importance_range': f"{df[importance_col].min():.6f} - {df[importance_col].max():.6f}",
                'features_with_zero_importance': len(df[df[importance_col] == 0]),
                'format': 'simple'
            }
        
        print("✓ Summary statistics generated")
        
    except Exception as e:
        print(f"✗ Error generating summary statistics: {e}")
    
    print("\n" + "=" * 60)
    print("SHAP ANALYSIS COMPLETE!")
    print(f"All visualizations saved to: {shap_viz_dir}")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 4:
        print("Usage: python shap_viz.py <importance_csv> <model_path> <dataset_path> [out_dir]")
        sys.exit(1)
    
    importance_csv = sys.argv[1]
    model_path = sys.argv[2] 
    dataset_path = sys.argv[3]
    out_dir = sys.argv[4] if len(sys.argv) > 4 else "shap_analysis_output"
    
    # Generate comprehensive report
    results = generate_comprehensive_shap_report(
        importance_csv=importance_csv,
        model_path=model_path,
        dataset_path=dataset_path,
        out_dir=out_dir,
        top_n=20,
        sample_size=1000,
        plot_format="png"
    ) 