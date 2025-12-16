"""
Feature Importance Integration Module

This module integrates the comprehensive feature importance analysis
with the gene-wise cross-validation workflow, providing seamless
integration of multiple importance analysis methods.
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
import joblib
from pathlib import Path
import json

from .feature_importance import FeatureImportanceAnalyzer

# Optional imports - handle gracefully if not available
try:
    from .shap_incremental import incremental_shap_importance, run_incremental_shap_analysis
    HAS_INCREMENTAL_SHAP = True
except ImportError:
    HAS_INCREMENTAL_SHAP = False
    incremental_shap_importance = None
    run_incremental_shap_analysis = None

try:
    from .shap_viz import generate_comprehensive_shap_report
    HAS_SHAP_VIZ = True
except ImportError:
    HAS_SHAP_VIZ = False
    generate_comprehensive_shap_report = None


def run_comprehensive_feature_importance_analysis(
    dataset_path: str | Path,
    out_dir: str | Path,
    sample: int = None,
    top_k: int = 25,
    methods: List[str] = None,
    include_shap: bool = True,
    verbose: int = 1
) -> Dict:
    """
    Run comprehensive feature importance analysis following the same pattern as SHAP analysis.
    
    This function mirrors the approach used in run_incremental_shap_analysis to ensure
    consistency with the existing CV workflow, particularly for categorical variable encoding.
    
    Parameters
    ----------
    dataset_path : str | Path
        Path to the dataset file or directory (same as used in CV)
    out_dir : str | Path
        Output directory from CV analysis
    sample : int, optional
        Number of samples to use for analysis
    top_k : int, optional
        Number of top features to analyze
    methods : List[str], optional
        Feature importance methods to use
    include_shap : bool, optional
        Whether to include SHAP analysis (uses existing results if available)
    verbose : int, optional
        Verbosity level
        
    Returns
    -------
    Dict
        Comprehensive results dictionary
    """
    import pickle
    import polars as pl
    
    run_dir = Path(out_dir)
    
    if verbose:
        print(f"[Feature Importance Analysis] Starting comprehensive analysis")
        print(f"[Feature Importance Analysis] Dataset: {dataset_path}")
        print(f"[Feature Importance Analysis] Output: {run_dir}")
    
    # Load the model (same approach as SHAP analysis)
    model_path_pkl = run_dir / "model_multiclass.pkl"
    model_path_json = run_dir / "model_multiclass.json"
    
    if model_path_pkl.exists():
        with open(model_path_pkl, "rb") as fh:
            model = pickle.load(fh)
    elif model_path_json.exists():
        import xgboost as xgb
        model = xgb.Booster()
        model.load_model(str(model_path_json))
    else:
        raise FileNotFoundError("No model file found in output directory")
    
    # Get feature names (same approach as SHAP analysis)
    csv_path = run_dir / "feature_manifest.csv"
    json_path_manifest = run_dir / "train.features.json"
    
    if csv_path.exists():
        feature_names = pd.read_csv(csv_path)["feature"].tolist()
    elif json_path_manifest.exists():
        feature_names = json.loads(json_path_manifest.read_text())["feature_names"]
    else:
        raise FileNotFoundError("Feature manifest not found in run_dir")
    
    # Load and process dataset using the same approach as SHAP analysis
    if verbose:
        print(f"[Feature Importance Analysis] Loading dataset...")
    
    # Load dataset using the same pattern as SHAP analysis
    if Path(dataset_path).is_dir():
        df_lf = pl.scan_parquet(str(Path(dataset_path) / "*.parquet"), missing_columns="insert")
    else:
        df_lf = pl.scan_parquet(str(dataset_path), missing_columns="insert")
    
    # Select only relevant feature columns plus gene_id and splice_type for hierarchical sampling
    required_columns = feature_names.copy()
    if 'gene_id' not in required_columns:
        required_columns.append('gene_id')
    if 'splice_type' not in required_columns:
        required_columns.append('splice_type')
    
    # Use helper function for robust column selection
    from meta_spliceai.splice_engine.meta_models.training.classifier_utils import select_available_columns
    df_lf, missing_cols, existing_cols = select_available_columns(
        df_lf, required_columns, context_name="Feature Importance Analysis", verbose=verbose
    )
    
    # Implement hierarchical sampling if sample is requested
    if sample is not None:
        if verbose:
            print(f"[Feature Importance Analysis] Applying hierarchical sampling to get {sample} rows...")
            print(f"[Feature Importance Analysis] Strategy: Sample genes, keep all splice sites, subsample non-splice sites")
        
        # First collect basic info to design sampling strategy
        try:
            # Import the LazyFrame sampling utility
            from meta_spliceai.splice_engine.meta_models.training.classifier_utils import _lazyframe_sample
            
            # Get class distribution and gene information
            class_info = df_lf.group_by("splice_type").agg(pl.count().alias("count")).collect()
            if verbose:
                print(f"[Feature Importance Analysis] Original class distribution:")
                for row in class_info.iter_rows(named=True):
                    print(f"  {row['splice_type']}: {row['count']:,} samples")
            
            # Count unique genes
            gene_count = df_lf.select(pl.col("gene_id").n_unique()).collect().item()
            if verbose:
                print(f"[Feature Importance Analysis] Total genes in dataset: {gene_count:,}")
            
            # Estimate target number of genes to sample
            # Strategy: Keep enough genes to ensure good representation while hitting sample target
            avg_samples_per_gene = df_lf.select(pl.count()).collect().item() / gene_count
            target_genes = max(50, min(gene_count, int(sample / (avg_samples_per_gene * 0.3))))  # Conservative estimate
            
            if verbose:
                print(f"[Feature Importance Analysis] Estimated samples per gene: {avg_samples_per_gene:.1f}")
                print(f"[Feature Importance Analysis] Target genes to sample: {target_genes}")
            
            # Sample genes using the utility function
            unique_genes_lf = df_lf.select("gene_id").unique()
            sampled_genes_lf = _lazyframe_sample(unique_genes_lf, target_genes, seed=42)
            sampled_genes = sampled_genes_lf.collect()["gene_id"].to_list()
            
            if verbose:
                print(f"[Feature Importance Analysis] Sampled {len(sampled_genes)} genes")
            
            # Filter to sampled genes
            df_genes_sampled = df_lf.filter(pl.col("gene_id").is_in(sampled_genes))
            
            # Now apply class-aware sampling within these genes
            # Keep ALL splice sites (donors and acceptors), subsample non-splice sites
            
            # Identify splice sites vs non-splice sites
            splice_sites = df_genes_sampled.filter(
                pl.col("splice_type").is_in(["donor", "acceptor"])
            )
            
            non_splice_sites = df_genes_sampled.filter(
                pl.col("splice_type").is_in(["neither", "0"])
            )
            
            # Count what we have
            splice_count = splice_sites.select(pl.count()).collect().item()
            non_splice_count = non_splice_sites.select(pl.count()).collect().item()
            
            if verbose:
                print(f"[Feature Importance Analysis] In sampled genes:")
                print(f"  Splice sites (keep all): {splice_count:,}")
                print(f"  Non-splice sites (will subsample): {non_splice_count:,}")
            
            # Determine how many non-splice sites to sample
            remaining_budget = sample - splice_count
            if remaining_budget <= 0:
                if verbose:
                    print(f"[Feature Importance Analysis] Warning: {splice_count} splice sites exceed sample budget of {sample}")
                    print(f"[Feature Importance Analysis] Keeping all splice sites anyway to preserve minority class")
                sampled_non_splice_lf = non_splice_sites.limit(0)  # Keep none
            else:
                # Sample non-splice sites using the utility function
                if verbose:
                    print(f"[Feature Importance Analysis] Subsampling {remaining_budget} non-splice sites from {non_splice_count}")
                sampled_non_splice_lf = _lazyframe_sample(non_splice_sites, remaining_budget, seed=42)
            
            # Combine splice sites and sampled non-splice sites
            splice_sites_df = splice_sites.collect()
            sampled_non_splice_df = sampled_non_splice_lf.collect()
            
            if sampled_non_splice_df.height > 0:
                df_sampled = pl.concat([splice_sites_df, sampled_non_splice_df])
            else:
                df_sampled = splice_sites_df
            
            # Final sampling if we still exceed the budget (using utility function)
            if df_sampled.height > sample:
                if verbose:
                    print(f"[Feature Importance Analysis] Final sampling: {df_sampled.height} -> {sample} rows")
                df_sampled_lf = _lazyframe_sample(pl.LazyFrame(df_sampled), sample, seed=42)
                df_sampled = df_sampled_lf.collect()
            
            final_class_dist = df_sampled.group_by("splice_type").agg(pl.count().alias("count"))
            if verbose:
                print(f"[Feature Importance Analysis] Final sampled distribution:")
                for row in final_class_dist.iter_rows(named=True):
                    print(f"  {row['splice_type']}: {row['count']:,} samples")
                print(f"[Feature Importance Analysis] Total sampled: {df_sampled.height:,} rows")
            
            # Convert back to lazy frame for downstream processing
            df_lf = pl.LazyFrame(df_sampled)
            
        except Exception as e:
            if verbose:
                print(f"[Feature Importance Analysis] Error in hierarchical sampling: {e}")
                print(f"[Feature Importance Analysis] Falling back to naive sampling...")
                import traceback
                traceback.print_exc()
            # Fallback to naive sampling using the utility function
            df_lf = _lazyframe_sample(df_lf, sample, seed=42)
    
    # Select only the feature columns for downstream processing
    # BUT preserve splice_type for label processing!
    columns_to_select = feature_names.copy()
    if 'splice_type' not in columns_to_select:
        columns_to_select.append('splice_type')
    
    # Use helper function for robust final column selection
    df_lf, missing_final_cols, existing_final_cols = select_available_columns(
        df_lf, columns_to_select, 
        context_name="Feature Importance Analysis - Final Selection", 
        verbose=verbose
    )
    
    # Convert to polars DataFrame first for preprocessing
    if verbose:
        print("[Feature Importance Analysis] Converting to Polars DataFrame...")
    X_pl = df_lf.collect(streaming=True)
    
    # Apply the same preprocessing pipeline used during training and SHAP analysis
    if verbose:
        print("[Feature Importance Analysis] Preprocessing features using training pipeline (includes chromosome encoding)...")
    
    from meta_spliceai.splice_engine.meta_models.builder.preprocessing import prepare_training_data
    
    # Use the same preprocessing as training (ensures consistent categorical encoding)
    X_processed, y_processed = prepare_training_data(
        X_pl,
        label_col="splice_type",
        encode_chrom=True,  # This handles chromosome string->numeric conversion
        return_type="pandas",
        verbose=0
    )
    
    # Now select only the features we need, adding zeros for missing ones
    from meta_spliceai.splice_engine.meta_models.training.classifier_utils import add_missing_features_with_zeros
    X_processed = add_missing_features_with_zeros(
        X_processed, feature_names, 
        context_name="Feature Importance Analysis", 
        verbose=verbose
    )
    
    # Ensure column order matches feature_names (and exclude splice_type from features)
    X = X_processed[feature_names]
    y = y_processed
    
    if verbose:
        print(f"[Feature Importance Analysis] Data shape: {X.shape}")
        print(f"[Feature Importance Analysis] Target distribution: {y.value_counts().to_dict()}")
    
    # Check for feature classification issues that could cause empty plots
    if verbose:
        print(f"[Feature Importance Analysis] Diagnosing data for statistical tests...")
        
        # Check data types
        print(f"[Feature Importance Analysis] Data types:")
        for col in X.columns[:10]:  # Show first 10 columns
            unique_vals = X[col].nunique()
            dtype = X[col].dtype
            has_strings = X[col].astype(str).str.contains(r'[A-Za-z]').any()
            print(f"  {col}: {dtype}, {unique_vals} unique values, contains_strings={has_strings}")
        
        # Check target encoding
        print(f"[Feature Importance Analysis] Target encoding: {y.dtype}, unique values: {y.unique()}")
        
    # Ensure target is properly encoded as binary (0/1) for statistical tests
    unique_labels = set(y.unique())
    if verbose:
        print(f"[Feature Importance Analysis] Unique target labels: {unique_labels}")
    
    if unique_labels != {0, 1}:
        if verbose:
            print(f"[Feature Importance Analysis] Converting multi-class target to binary (splice vs non-splice)")
        
        # Handle mixed data types: "0"/"neither" vs "donor"/"acceptor"
        # Convert to binary: splice sites (1) vs non-splice sites (0)
        y_binary = pd.Series(index=y.index, dtype=int)
        
        for i, label in enumerate(y):
            if label in ["0", 0, "neither"]:
                y_binary.iloc[i] = 0  # Non-splice site
            elif label in ["donor", "acceptor", 1, 2]:
                y_binary.iloc[i] = 1  # Splice site
            else:
                if verbose:
                    print(f"[Feature Importance Analysis] Warning: Unknown label '{label}' at index {i}, treating as non-splice")
                y_binary.iloc[i] = 0
    else:
        y_binary = y.copy()
    
    if verbose:
        print(f"[Feature Importance Analysis] Binary target distribution: {y_binary.value_counts().to_dict()}")
        print(f"[Feature Importance Analysis] Features to analyze: {len(X.columns)}")
        
        # Ensure we have valid data for analysis
        if X.empty:
            raise ValueError("Feature matrix X is empty!")
        if y_binary.empty:
            raise ValueError("Target vector y_binary is empty!")
        if len(X) != len(y_binary):
            raise ValueError(f"Mismatched lengths: X has {len(X)} samples, y_binary has {len(y_binary)} samples")
    
    # Set default methods if not provided
    if methods is None:
        methods = ['xgboost', 'hypothesis_testing', 'effect_sizes', 'mutual_info']
    
    if verbose:
        print(f"[Feature Importance Analysis] Methods to run: {methods}")
    
    # Create output directory for feature importance analysis
    feature_importance_dir = run_dir / "feature_importance_analysis"
    feature_importance_dir.mkdir(exist_ok=True, parents=True)
    
    # Initialize comprehensive analyzer
    analyzer = FeatureImportanceAnalyzer(
        output_dir=feature_importance_dir,
        subject="gene_cv_analysis"
    )
    
    # Enhanced feature categories for better statistical test classification
    feature_categories = analyzer._classify_features(X, y_binary)
    
    if verbose:
        print(f"[Feature Importance Analysis] Feature classification:")
        print(f"  Numerical features: {len(feature_categories.get('numerical_features', []))}")
        print(f"  Categorical features: {len(feature_categories.get('categorical_features', []))}")
        print(f"  K-mer features: {len(feature_categories.get('kmer_features', []))}")
        print(f"  Sample data shape: {X.shape}")
    
    # Initialize results dictionary
    analysis_results = {}
    
    # Run standard feature importance analysis methods
    if verbose:
        print("[Feature Importance Analysis] Running feature importance methods...")
    
    # Handle different model types - but ALWAYS run non-XGBoost methods
    ensemble_model = None
    regular_model = None
    
    if hasattr(model, 'models') and hasattr(model, 'feature_names'):
        # This is an ensemble model
        ensemble_model = model
        if verbose:
            print(f"[Feature Importance Analysis] Detected ensemble model ({type(model).__name__})")
    elif hasattr(model, 'get_booster'):
        # Regular XGBoost model
        regular_model = model
        if verbose:
            print("[Feature Importance Analysis] Detected standard XGBoost model")
    else:
        if verbose:
            print(f"[Feature Importance Analysis] Unknown model type ({type(model).__name__}), skipping XGBoost-specific analysis")
    
    # ALWAYS run non-XGBoost methods first (these are independent of model type)
    non_xgb_methods = [m for m in methods if m != 'xgboost']
    
    for method in non_xgb_methods:
        if verbose:
            print(f"[Feature Importance Analysis] Running {method} analysis...")
        
        try:
            if method == 'hypothesis_testing':
                method_result = analyzer.analyze_hypothesis_testing(
                    X=X,
                    y=y_binary,
                    top_k=top_k,
                    feature_categories=feature_categories,
                    verbose=verbose
                )
                analysis_results['hypothesis_testing'] = method_result
                if verbose:
                    print(f"[Feature Importance Analysis] ✓ Hypothesis testing completed - {method_result.get('num_significant', 0)} significant features")
            
            elif method == 'effect_sizes':
                method_result = analyzer.analyze_effect_sizes(
                    X=X,
                    y=y_binary,
                    top_k=top_k,
                    feature_categories=feature_categories,
                    verbose=verbose
                )
                analysis_results['effect_sizes'] = method_result
                if verbose:
                    print(f"[Feature Importance Analysis] ✓ Effect sizes completed - {method_result.get('num_large_effects', 0)} large effects")
            
            elif method == 'mutual_info':
                method_result = analyzer.analyze_mutual_information(
                    X=X,
                    y=y_binary,
                    top_k=top_k,
                    feature_categories=feature_categories,
                    verbose=verbose
                )
                analysis_results['mutual_info'] = method_result
                if verbose:
                    print(f"[Feature Importance Analysis] ✓ Mutual information completed")
            
        except Exception as e:
            if verbose:
                print(f"[Feature Importance Analysis] ⚠️  Error in {method} analysis: {e}")
                import traceback
                traceback.print_exc()
            # Continue with other methods even if one fails
            continue
    
    # Handle XGBoost importance analysis
    if 'xgboost' in methods:
        try:
            if ensemble_model is not None:
                if verbose:
                    print(f"[Feature Importance Analysis] Processing ensemble XGBoost importance...")
                
                # Aggregate feature importance from all models in the ensemble
                importance_types = ['weight', 'gain', 'cover', 'total_gain', 'total_cover']
                ensemble_importance_by_type = {}
                
                for importance_type in importance_types:
                    ensemble_importance_by_type[importance_type] = {}
                    
                    for i, xgb_model in enumerate(ensemble_model.models):
                        if hasattr(xgb_model, 'get_booster'):
                            booster = xgb_model.get_booster()
                            try:
                                model_importance = booster.get_score(importance_type=importance_type)
                                
                                # Aggregate importance scores
                                for feature, importance in model_importance.items():
                                    if feature not in ensemble_importance_by_type[importance_type]:
                                        ensemble_importance_by_type[importance_type][feature] = []
                                    ensemble_importance_by_type[importance_type][feature].append(importance)
                            except Exception as e:
                                if verbose:
                                    print(f"[Feature Importance Analysis] Warning: Could not extract {importance_type} importance from model {i}: {e}")
                
                # Average importance across models for each type
                ensemble_results = {}
                
                for importance_type, type_importance in ensemble_importance_by_type.items():
                    if not type_importance:
                        continue
                        
                    averaged_importance = {}
                    for feature, importance_list in type_importance.items():
                        averaged_importance[feature] = np.mean(importance_list)
                    
                    # Map feature indices back to names
                    feature_index_map = {f"f{i}": name for i, name in enumerate(ensemble_model.feature_names)}
                    named_importance = {}
                    for feature_key, importance in averaged_importance.items():
                        feature_name = feature_index_map.get(feature_key, feature_key)
                        named_importance[feature_name] = importance
                    
                    # Add missing features with zero importance
                    for feature_name in feature_names:
                        if feature_name not in named_importance:
                            named_importance[feature_name] = 0.0
                    
                    # Create DataFrame
                    importance_df = pd.DataFrame([
                        {"feature": feature, "importance_score": importance}
                        for feature, importance in named_importance.items()
                    ]).sort_values("importance_score", ascending=False)
                    
                    ensemble_results[importance_type] = {
                        'full_df': importance_df,
                        'top_k': importance_df.head(top_k)
                    }
                    
                    # Create XGBoost-specific plots for each importance type
                    if verbose:
                        print(f"[Feature Importance Analysis] Creating XGBoost {importance_type} importance plot...")
                    analyzer._plot_xgboost_importance(
                        importance_df.head(top_k), 
                        f"ensemble_{importance_type}", 
                        top_k
                    )
                
                # Add our aggregated XGBoost results
                analysis_results['xgboost'] = ensemble_results
                
                if verbose:
                    print(f"[Feature Importance Analysis] ✓ Successfully created XGBoost importance plots for {len(ensemble_results)} importance types from {len(ensemble_model.models)} ensemble models")
                
            elif regular_model is not None:
                if verbose:
                    print("[Feature Importance Analysis] Processing standard XGBoost importance...")
                
                xgb_result = analyzer.analyze_xgboost_importance(
                    model=regular_model,
                    X=X,
                    top_k=top_k,
                    verbose=verbose
                )
                analysis_results['xgboost'] = xgb_result
                
                if verbose:
                    print(f"[Feature Importance Analysis] ✓ XGBoost importance analysis completed")
            
            else:
                if verbose:
                    print("[Feature Importance Analysis] No compatible XGBoost model found, skipping XGBoost importance")
                
        except Exception as e:
            if verbose:
                print(f"[Feature Importance Analysis] ⚠️  Error in XGBoost analysis: {e}")
                import traceback
                traceback.print_exc()
    
    # Create comparison plots if we have multiple methods
    if len(analysis_results) > 1:
        if verbose:
            print("[Feature Importance Analysis] Creating comparison plots...")
        try:
            analyzer.create_comparison_plots(analysis_results, top_k=top_k)
            if verbose:
                print("[Feature Importance Analysis] ✓ Comparison plots created")
        except Exception as e:
            if verbose:
                print(f"[Feature Importance Analysis] ⚠️  Error creating comparison plots: {e}")
    
    if verbose:
        print(f"[Feature Importance Analysis] Completed analysis for {len(analysis_results)} methods")
        for method_name, method_result in analysis_results.items():
            if method_name == 'xgboost':
                print(f"  - {method_name}: {len(method_result)} importance types")
            else:
                num_features = len(method_result.get('full_df', [])) if 'full_df' in method_result else 0
                print(f"  - {method_name}: {num_features} features analyzed")

    # Include existing SHAP results if available and requested
    shap_results = None
    if include_shap:
        if verbose:
            print("[Feature Importance Analysis] Checking for existing SHAP results...")
        
        # Look for existing SHAP results from the CV analysis
        shap_csv_path = run_dir / "shap_importance_incremental.csv"
        
        if shap_csv_path.exists():
            if verbose:
                print("[Feature Importance Analysis] Found existing SHAP results, integrating...")
            
            try:
                shap_df = pd.read_csv(shap_csv_path)
                
                # Handle different SHAP result formats
                if 'importance_mean' in shap_df.columns:
                    # Multi-class SHAP results
                    shap_results = {
                        'feature_importance': shap_df[['feature', 'importance_mean']].rename(
                            columns={'importance_mean': 'importance_score'}
                        ).sort_values('importance_score', ascending=False),
                        'multi_class_importance': shap_df,
                        'source': 'existing_incremental_shap'
                    }
                elif 'importance' in shap_df.columns:
                    # Single importance column
                    shap_results = {
                        'feature_importance': shap_df[['feature', 'importance']].rename(
                            columns={'importance': 'importance_score'}
                        ).sort_values('importance_score', ascending=False),
                        'source': 'existing_shap'
                    }
                else:
                    if verbose:
                        print("[Feature Importance Analysis] Warning: SHAP CSV format not recognized")
                    shap_results = None
            except Exception as e:
                if verbose:
                    print(f"[Feature Importance Analysis] Error reading SHAP results: {e}")
                shap_results = None
        else:
            if verbose:
                print("[Feature Importance Analysis] No existing SHAP results found")
    
    # Save comprehensive results
    analyzer.save_results("gene_cv_comprehensive_results.xlsx")
    
    # Create integrated summary
    summary_results = create_integrated_summary(
        standard_results=analysis_results,
        shap_results=shap_results,
        output_dir=feature_importance_dir,
        subject="gene_cv_analysis",
        top_k=top_k,
        verbose=verbose
    )
    
    if verbose:
        print(f"[Feature Importance Analysis] Analysis complete! Results saved to: {feature_importance_dir}")
        
        # Log summary of top features
        if 'consensus_features' in summary_results and summary_results['consensus_features']:
            print(f"[Feature Importance Analysis] Top consensus features: {summary_results['consensus_features'][:5]}")
    
    return {
        'standard_results': analysis_results,
        'shap_results': shap_results,
        'summary': summary_results,
        'output_dir': feature_importance_dir
    }


def create_integrated_summary(
    standard_results: Dict,
    shap_results: Dict,
    output_dir: str,
    subject: str,
    top_k: int = 25,
    verbose: int = 1
) -> Dict:
    """
    Create integrated summary combining all analysis methods.
    """
    summary = {
        'top_features_by_method': {},
        'consensus_features': [],
        'method_agreement': {},
        'feature_rankings': {}
    }
    
    # Extract top features from each method
    all_methods = {}
    
    # Standard methods
    for method_name, method_results in standard_results.items():
        if method_name == 'xgboost':
            # Use gain as primary XGBoost method
            if 'gain' in method_results:
                all_methods['xgboost_gain'] = set(
                    method_results['gain']['top_k']['feature'].head(top_k).tolist()
                )
        else:
            if 'top_k' in method_results:
                all_methods[method_name] = set(
                    method_results['top_k']['feature'].head(top_k).tolist()
                )
    
    # SHAP results
    if shap_results and 'feature_importance' in shap_results:
        all_methods['shap'] = set(
            shap_results['feature_importance']['feature'].head(top_k).tolist()
        )
    
    summary['top_features_by_method'] = {
        method: list(features) for method, features in all_methods.items()
    }
    
    # Find consensus features (appearing in multiple methods)
    if len(all_methods) > 1:
        feature_counts = {}
        for method, features in all_methods.items():
            for feature in features:
                feature_counts[feature] = feature_counts.get(feature, 0) + 1
        
        # Features appearing in at least 2 methods
        consensus_features = [
            feature for feature, count in feature_counts.items()
            if count >= 2
        ]
        
        summary['consensus_features'] = sorted(
            consensus_features, 
            key=lambda x: feature_counts[x], 
            reverse=True
        )
    
    # Calculate method agreement
    if len(all_methods) > 1:
        method_pairs = []
        method_names = list(all_methods.keys())
        
        for i in range(len(method_names)):
            for j in range(i + 1, len(method_names)):
                method1, method2 = method_names[i], method_names[j]
                overlap = len(all_methods[method1] & all_methods[method2])
                union = len(all_methods[method1] | all_methods[method2])
                
                jaccard = overlap / union if union > 0 else 0
                
                method_pairs.append({
                    'method1': method1,
                    'method2': method2,
                    'overlap': overlap,
                    'jaccard_similarity': jaccard
                })
        
        summary['method_agreement'] = method_pairs
    
    # Create unified feature ranking
    unified_ranking = create_unified_ranking(
        standard_results, shap_results, top_k
    )
    summary['feature_rankings'] = unified_ranking
    
    # Save summary
    summary_path = os.path.join(output_dir, "integrated_summary.json")
    with open(summary_path, 'w') as f:
        # Convert sets to lists for JSON serialization
        json_summary = {}
        for key, value in summary.items():
            if key == 'top_features_by_method':
                json_summary[key] = {
                    method: list(features) if isinstance(features, set) else features
                    for method, features in value.items()
                }
            elif key == 'feature_rankings' and hasattr(value, 'to_dict'):
                json_summary[key] = value.to_dict('records')
            else:
                json_summary[key] = value
        json.dump(json_summary, f, indent=2)
    
    if verbose:
        print(f"[Feature Importance Analysis] Integrated summary saved to: {summary_path}")
    
    return summary


def create_unified_ranking(
    standard_results: Dict,
    shap_results: Dict,
    top_k: int
) -> pd.DataFrame:
    """
    Create unified feature ranking combining all methods.
    """
    # Collect all features and their ranks from different methods
    feature_ranks = {}
    
    # Process standard results
    for method_name, method_results in standard_results.items():
        if method_name == 'xgboost':
            # Use gain as primary
            if 'gain' in method_results:
                features_df = method_results['gain']['top_k']
                for idx, row in features_df.iterrows():
                    feature = row['feature']
                    if feature not in feature_ranks:
                        feature_ranks[feature] = {}
                    feature_ranks[feature]['xgboost_gain'] = idx + 1
        else:
            if 'top_k' in method_results:
                features_df = method_results['top_k']
                for idx, row in features_df.iterrows():
                    feature = row['feature']
                    if feature not in feature_ranks:
                        feature_ranks[feature] = {}
                    feature_ranks[feature][method_name] = idx + 1
    
    # Process SHAP results
    if shap_results and 'feature_importance' in shap_results:
        features_df = shap_results['feature_importance'].head(top_k)
        for idx, row in features_df.iterrows():
            feature = row['feature']
            if feature not in feature_ranks:
                feature_ranks[feature] = {}
            feature_ranks[feature]['shap'] = idx + 1
    
    # Create DataFrame
    ranking_data = []
    for feature, ranks in feature_ranks.items():
        row = {'feature': feature}
        row.update(ranks)
        
        # Calculate average rank (lower is better)
        rank_values = [rank for rank in ranks.values() if rank <= top_k]
        if rank_values:
            row['average_rank'] = np.mean(rank_values)
            row['num_methods'] = len(rank_values)
        else:
            row['average_rank'] = float('inf')
            row['num_methods'] = 0
        
        ranking_data.append(row)
    
    # Sort by average rank
    unified_df = pd.DataFrame(ranking_data).sort_values(
        ['num_methods', 'average_rank'], 
        ascending=[False, True]
    ).reset_index(drop=True)
    
    return unified_df


# Main function for integration into CV workflow
def run_gene_cv_feature_importance_analysis(
    dataset_path: str | Path,
    out_dir: str | Path,
    sample: int = None,
    **kwargs
) -> Path:
    """
    Main function to integrate feature importance analysis into gene CV workflow.
    
    This function follows the same signature pattern as run_incremental_shap_analysis
    to ensure seamless integration into the existing CV workflow.
    
    Parameters
    ----------
    dataset_path : str | Path
        Path to dataset file or directory
    out_dir : str | Path
        Path to model directory
    sample : int, optional
        Number of samples to use for analysis
    **kwargs
        Additional arguments passed to the analysis
        
    Returns
    -------
    Path
        Path to the feature importance analysis directory
    """
    try:
        results = run_comprehensive_feature_importance_analysis(
            dataset_path=dataset_path,
            out_dir=out_dir,
            sample=sample,
            **kwargs
        )
        
        output_dir = results['output_dir']
        print(f"[Feature Importance Analysis] ✓ Comprehensive analysis completed successfully")
        print(f"[Feature Importance Analysis] Results saved to: {output_dir}")
        
        return Path(output_dir)
        
    except Exception as e:
        print(f"[Feature Importance Analysis] ✗ Analysis failed: {e}")
        # Don't raise the exception to avoid breaking the CV workflow
        return None


# Convenience function for command-line usage
def main():
    """
    Main function for command-line usage.
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run comprehensive feature importance analysis"
    )
    
    parser.add_argument(
        "dataset_path",
        help="Path to the dataset file or directory"
    )
    
    parser.add_argument(
        "out_dir",
        help="Path to the CV output directory"
    )
    
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Number of samples to use for analysis"
    )
    
    parser.add_argument(
        "--top-k", "-k",
        type=int,
        default=25,
        help="Number of top features to analyze"
    )
    
    parser.add_argument(
        "--methods", "-m",
        nargs="+",
        default=None,
        help="Feature importance methods to use"
    )
    
    parser.add_argument(
        "--no-shap",
        action="store_true",
        help="Skip SHAP analysis integration"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    # Run analysis
    output_dir = run_gene_cv_feature_importance_analysis(
        dataset_path=args.dataset_path,
        out_dir=args.out_dir,
        sample=args.sample,
        top_k=args.top_k,
        methods=args.methods,
        include_shap=not args.no_shap,
        verbose=int(args.verbose)
    )
    
    if output_dir:
        print(f"Analysis complete! Results saved to: {output_dir}")
    else:
        print("Analysis failed!")


if __name__ == "__main__":
    main() 