#!/usr/bin/env python3
"""
Training Strategy Abstraction for Meta-Model Training

This module provides a pluggable abstraction for different training methodologies
while ensuring consistent outputs and post-training analysis regardless of the
underlying training approach (single XGBoost, batch ensemble, future models, etc.).

Key Design Principles:
1. Training methodology is encapsulated as a strategy pattern
2. All strategies produce the same interface and outputs
3. Post-training analysis is unified and methodology-agnostic
4. Feature screening is consistent across all approaches
5. Easy to swap training backends without changing downstream code
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd
import polars as pl
import pickle
import json
from datetime import datetime
from sklearn.metrics import accuracy_score, f1_score

from meta_spliceai.splice_engine.meta_models.training.classifier_utils import (
    SigmoidEnsemble, PerClassCalibratedSigmoidEnsemble
)


@dataclass
class TrainingResult:
    """Standardized result from any training strategy."""
    model_path: Path
    feature_names: List[str]
    excluded_features: List[str]
    training_metadata: Dict[str, Any]
    cv_results: Optional[List[Dict]] = None
    performance_metrics: Optional[Dict] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'model_path': str(self.model_path),
            'feature_names': self.feature_names,
            'excluded_features': self.excluded_features,
            'training_metadata': self.training_metadata,
            'cv_results': self.cv_results,
            'performance_metrics': self.performance_metrics
        }


@dataclass
class GlobalFeatureScreeningResult:
    """Result from global feature screening process."""
    excluded_features: List[str]
    leaky_features: List[str]
    correlation_report_path: Path
    screening_metadata: Dict[str, Any]


class TrainingStrategy(ABC):
    """Abstract base class for all training strategies."""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self._global_excluded_features: List[str] = []
        
    @abstractmethod
    def get_strategy_name(self) -> str:
        """Return human-readable name for this training strategy."""
        pass
    
    @abstractmethod
    def can_handle_dataset_size(self, total_genes: int, estimated_memory_gb: float) -> bool:
        """Check if this strategy can handle the given dataset size."""
        pass
    
    @abstractmethod
    def train_model(
        self,
        dataset_path: str,
        out_dir: Path,
        args,
        X_df: pd.DataFrame,
        y_series: pd.Series,
        genes: np.ndarray
    ) -> TrainingResult:
        """Train the model using this strategy."""
        pass
    
    def run_global_feature_screening(
        self,
        dataset_path: str,
        out_dir: Path,
        args,
        sample_fraction: float = 0.1
    ) -> GlobalFeatureScreeningResult:
        """
        Run global feature screening before training.
        
        This ensures consistent feature sets across all training approaches.
        """
        print(f"🔍 [Global Feature Screening] Running unified feature screening...", flush=True)
        print(f"  🎯 Strategy: {self.get_strategy_name()}", flush=True)
        print(f"  📊 Sample fraction: {sample_fraction}", flush=True)
        
        from meta_spliceai.splice_engine.meta_models.training import datasets
        from meta_spliceai.splice_engine.meta_models.builder import preprocessing
        from meta_spliceai.splice_engine.meta_models.evaluation.viz_utils import check_feature_correlations
        from meta_spliceai.splice_engine.meta_models.training.classifier_utils import _encode_labels
        
        # Load representative sample
        if sample_fraction < 1.0:
            from meta_spliceai.splice_engine.meta_models.training.label_utils import load_dataset_sample
            # Calculate sample size based on estimated total genes
            try:
                # Quick estimate of total genes
                lf = pl.scan_parquet(f"{dataset_path}/*.parquet", extra_columns='ignore')
                total_genes = lf.select(pl.col("gene_id").n_unique()).collect().item()
                
                if self.verbose:
                    print(f"  🔍 [Feature Screening] Analyzing sample data to identify leaky features")
                    print(f"  📊 Full dataset contains: {total_genes:,} total genes")
                    print(f"  🎯 Loading representative sample for feature screening only")
                
                # CRITICAL FIX: Respect user's sample_genes parameter for memory efficiency
                if hasattr(args, 'sample_genes') and args.sample_genes is not None:
                    # Use user's sample size, but ensure it's reasonable for feature screening
                    sample_genes = max(args.sample_genes, 50)  # Minimum 50 genes for screening
                    if self.verbose:
                        print(f"  Estimated total genes: {total_genes:,}")
                        print(f"  Using user-specified sample: {sample_genes:,} genes (respecting --sample-genes {args.sample_genes})")
                        if sample_genes > args.sample_genes:
                            print(f"  Note: Increased to {sample_genes} genes minimum for reliable feature screening")
                else:
                    # Use percentage-based sampling only when user didn't specify sample size
                    sample_genes = max(100, int(total_genes * sample_fraction))
                    if self.verbose:
                        print(f"  📊 Using sample: {sample_genes:,} genes ({sample_fraction:.1%}) for feature screening")
                
                print(f"  ⚠️  This sample is ONLY for identifying leaky features - NOT for model training", flush=True)
                df = load_dataset_sample(dataset_path, sample_genes=sample_genes, random_seed=42)
                
                # Add clarifying message about actual genes found vs requested
                actual_genes = df['gene_id'].nunique() if 'gene_id' in df.columns else 0
                if self.verbose and actual_genes != sample_genes:
                    print(f"  📊 Requested {sample_genes} genes, found {actual_genes} genes in sampled data")
                    print(f"  ✅ Using all {actual_genes} available genes for feature screening")
            except Exception as e:
                if self.verbose:
                    print(f"  Warning: Could not sample dataset: {e}")
                    print(f"  Falling back to full dataset load")
                df = datasets.load_dataset(dataset_path)
        else:
            df = datasets.load_dataset(dataset_path)
        
        # Prepare features
        X_df, y_series = preprocessing.prepare_training_data(
            df, 
            label_col="splice_type", 
            return_type="pandas", 
            verbose=1 if self.verbose else 0,
            encode_chrom=True
        )
        
        if self.verbose:
            print(f"  Feature screening dataset: {X_df.shape[0]:,} positions, {X_df.shape[1]} features")
        
        # Run comprehensive leakage analysis
        excluded_features = []
        leaky_features = []
        correlation_report_path = out_dir / "global_feature_correlations.csv"
        
        if args.check_leakage:
            try:
                from meta_spliceai.splice_engine.meta_models.evaluation.leakage_analysis import LeakageAnalyzer
                
                leakage_analysis_dir = out_dir / "leakage_analysis"
                leakage_analysis_dir.mkdir(exist_ok=True, parents=True)
                
                analyzer = LeakageAnalyzer(
                    output_dir=leakage_analysis_dir,
                    subject="global_feature_screening"
                )
                
                leakage_results = analyzer.run_comprehensive_analysis(
                    X=X_df,
                    y=y_series,
                    threshold=args.leakage_threshold,
                    methods=['pearson', 'spearman'],
                    top_n=50,
                    verbose=1 if self.verbose else 0
                )
                
                # Extract leaky features
                for method_results in leakage_results['correlation_results'].values():
                    leaky_features.extend(method_results['leaky_features']['feature'].tolist())
                
                leaky_features = list(set(leaky_features))  # Remove duplicates
                
                if args.auto_exclude_leaky:
                    excluded_features.extend(leaky_features)
                
                if self.verbose:
                    print(f"  Found {len(leaky_features)} potentially leaky features")
                    if args.auto_exclude_leaky:
                        print(f"  Auto-excluding {len(leaky_features)} leaky features")
                
            except Exception as e:
                if self.verbose:
                    print(f"  Warning: Advanced leakage analysis failed: {e}")
                    print(f"  Falling back to basic correlation analysis")
                
                # Fallback to basic analysis
                X_np = X_df.values
                y_np = _encode_labels(y_series)
                curr_features = X_df.columns.tolist()
                
                basic_leaky_features, corr_df = check_feature_correlations(
                    X_np, y_np, curr_features, args.leakage_threshold, correlation_report_path
                )
                
                leaky_features = basic_leaky_features
                if args.auto_exclude_leaky:
                    excluded_features.extend(leaky_features)
        
        # Handle manual exclusions
        if hasattr(args, 'exclude_features') and args.exclude_features:
            from meta_spliceai.splice_engine.meta_models.evaluation.feature_utils import load_excluded_features
            
            exclude_path = Path(args.exclude_features)
            if exclude_path.exists() and exclude_path.is_file():
                manual_exclusions = load_excluded_features(exclude_path)
                excluded_features.extend(manual_exclusions)
            else:
                # Comma-separated list
                manual_exclusions = [f.strip() for f in args.exclude_features.split(',') if f.strip()]
                excluded_features.extend(manual_exclusions)
        
        # Remove duplicates while preserving order
        excluded_features = list(dict.fromkeys(excluded_features))
        
        # Save global exclusion list
        global_exclusions_path = out_dir / "global_excluded_features.txt"
        with open(global_exclusions_path, 'w') as f:
            f.write("# Global feature exclusions applied to all training strategies\n")
            f.write(f"# Generated: {datetime.now().isoformat()}\n")
            f.write(f"# Strategy: {self.get_strategy_name()}\n")
            f.write(f"# Total excluded: {len(excluded_features)}\n\n")
            for feature in excluded_features:
                f.write(f"{feature}\n")
        
        # Store for use by training strategies
        self._global_excluded_features = excluded_features
        
        if self.verbose:
            print(f"  ✅ Global feature screening completed")
            print(f"  Total exclusions: {len(excluded_features)}")
            print(f"  Saved to: {global_exclusions_path}")
        
        return GlobalFeatureScreeningResult(
            excluded_features=excluded_features,
            leaky_features=leaky_features,
            correlation_report_path=correlation_report_path,
            screening_metadata={
                'strategy': self.get_strategy_name(),
                'sample_fraction': sample_fraction,
                'total_features_before': X_df.shape[1],
                'total_features_after': X_df.shape[1] - len(excluded_features),
                'screening_date': datetime.now().isoformat(),
                'leakage_threshold': getattr(args, 'leakage_threshold', 0.95),
                'auto_exclude_leaky': getattr(args, 'auto_exclude_leaky', False)
            }
        )
    
    def apply_global_feature_exclusions(self, X_df: pd.DataFrame) -> pd.DataFrame:
        """Apply globally determined feature exclusions to feature matrix."""
        if not self._global_excluded_features:
            return X_df
        
        excluded_count = 0
        for feature in self._global_excluded_features:
            if feature in X_df.columns:
                X_df = X_df.drop(columns=[feature])
                excluded_count += 1
        
        if self.verbose and excluded_count > 0:
            print(f"  Applied {excluded_count} global feature exclusions")
        
        return X_df


class SingleModelTrainingStrategy(TrainingStrategy):
    """Traditional single XGBoost model training strategy."""
    
    def get_strategy_name(self) -> str:
        return "Single XGBoost Model"
    
    def can_handle_dataset_size(self, total_genes: int, estimated_memory_gb: float) -> bool:
        """Check if single model training can handle the dataset size."""
        # Get system memory to make intelligent decisions
        try:
            import psutil
            system_memory_gb = psutil.virtual_memory().total / (1024**3)
            available_memory_gb = psutil.virtual_memory().available / (1024**3)
            
            # More conservative estimates for large datasets with many features
            # The regulatory dataset has ~1150 features vs ~130 for the PC dataset
            max_genes_safe = min(2000, int(available_memory_gb * 100))  # More conservative: ~100 genes per GB
            max_memory_safe = min(16.0, available_memory_gb * 0.6)  # More conservative: 60% of available memory
            
            if self.verbose:
                print(f"    System memory: {system_memory_gb:.1f} GB total, {available_memory_gb:.1f} GB available")
                print(f"    Single model limits: {max_genes_safe:,} genes, {max_memory_safe:.1f} GB memory")
            
        except ImportError:
            # Fallback if psutil not available
            max_genes_safe = 2000
            max_memory_safe = 16.0
            
            if self.verbose:
                print(f"    Using default limits: {max_genes_safe:,} genes, {max_memory_safe:.1f} GB memory")
        
        return total_genes <= max_genes_safe and estimated_memory_gb <= max_memory_safe
    
    def train_model(
        self,
        dataset_path: str,
        out_dir: Path,
        args,
        X_df: pd.DataFrame,
        y_series: pd.Series,
        genes: np.ndarray
    ) -> TrainingResult:
        """Train single XGBoost model using existing pipeline."""
        
        print(f"🚀 [Single Model Training] Training unified XGBoost model...", flush=True)
        print(f"  🧬 Genes: {len(np.unique(genes)):,}", flush=True)
        print(f"  📊 Positions: {X_df.shape[0]:,}", flush=True)
        print(f"  🔧 Features: {X_df.shape[1]}", flush=True)
        
        # For large datasets, we already have the data loaded - don't reload
        # Load original dataframe for transcript information access during CV only if needed
        try:
            # CRITICAL FIX: Don't reload full dataset when user specified sample_genes
            if (hasattr(args, 'sample_genes') and args.sample_genes is not None):
                # User specified sample_genes - don't reload full dataset to avoid OOM
                self._original_df = None
                if self.verbose:
                    print(f"  🧬 Gene sampling mode: Using provided sample data to avoid memory issues")
                    print(f"  💡 Full dataset loading skipped due to --sample-genes {args.sample_genes}")
            elif len(np.unique(genes)) < 1000:  # Only for smaller datasets without explicit sampling
                from meta_spliceai.splice_engine.meta_models.training import datasets
                self._original_df = datasets.load_dataset(dataset_path)
                if self.verbose:
                    has_transcript = 'transcript_id' in self._original_df.columns
                    print(f"  🧬 Transcript info available: {has_transcript}")
            else:
                # For large datasets, use the passed dataframe to avoid reloading
                self._original_df = None
                if self.verbose:
                    print(f"  🧬 Large dataset: Using provided data to avoid reloading")
        except Exception as e:
            if self.verbose:
                print(f"  ⚠️  Could not load original dataframe for transcript info: {e}")
            self._original_df = None
        
        # Extract transcript data from original dataframe if available
        self._transcript_data = None
        
        # First check if transcript data is in X_df
        if 'transcript_id' in X_df.columns:
            transcript_cols = ['transcript_id']
            if 'position' in X_df.columns:
                transcript_cols.append('position')
            self._transcript_data = X_df[transcript_cols].copy()
            if self.verbose:
                transcript_count = self._transcript_data['transcript_id'].nunique()
                print(f"  🧬 Extracted transcript data from features: {transcript_count:,} unique transcripts")
        # Otherwise check the original dataframe
        elif hasattr(self, '_original_df') and self._original_df is not None and 'transcript_id' in self._original_df.columns:
            transcript_cols = ['transcript_id']
            if 'position' in self._original_df.columns:
                transcript_cols.append('position')
            self._transcript_data = self._original_df[transcript_cols].reset_index(drop=True)
            if self.verbose:
                transcript_count = self._transcript_data['transcript_id'].nunique()
                print(f"  🧬 Extracted transcript data from original df: {transcript_count:,} unique transcripts")
        else:
            if self.verbose:
                print(f"  ⚠️  No transcript data available (missing transcript_id column)")
        
        # Apply global feature exclusions
        print(f"  🚫 Applying global feature exclusions...", flush=True)
        original_feature_count = X_df.shape[1]
        X_df = self.apply_global_feature_exclusions(X_df)
        
        # Remove transcript_id and position from features (they're metadata, not features)
        metadata_cols_to_remove = ['transcript_id', 'position']
        for col in metadata_cols_to_remove:
            if col in X_df.columns:
                X_df = X_df.drop(columns=[col])
                if self.verbose:
                    print(f"    Removed metadata column: {col}")
        
        if X_df.shape[1] < original_feature_count:
            excluded_count = original_feature_count - X_df.shape[1]
            print(f"  ✅ Features after exclusions: {X_df.shape[1]} (removed {excluded_count})", flush=True)
        
        # Import training utilities
        from meta_spliceai.splice_engine.meta_models.training.classifier_utils import _encode_labels
        from sklearn.model_selection import GroupKFold, GroupShuffleSplit
        
        # Prepare data
        print(f"  🔄 Preparing data arrays...", flush=True)
        X = X_df.values
        y = _encode_labels(y_series)
        feature_names = list(X_df.columns)
        
        # Store CV data for final model training
        self._X_cv = X
        self._y_cv = y
        self._genes_cv = genes
        
        # Run gene-aware cross-validation (this is the existing CV loop from run_gene_cv_sigmoid.py)
        print(f"  🔀 Running gene-aware cross-validation ({args.n_folds} folds)...", flush=True)
        cv_results = self._run_gene_aware_cv(X, y, genes, feature_names, out_dir, args)
        
        # Train final model on available data
        print(f"  🎯 Training final model on available data...", flush=True)
        
        # For large datasets, use the current data instead of reloading everything
        total_unique_genes = len(np.unique(genes))
        if total_unique_genes >= 1000:
            print(f"  ℹ️  Large dataset ({total_unique_genes:,} genes): Training on current sample to avoid memory issues")
            final_model = self._train_final_model_from_cv_data(feature_names, args)
        else:
            final_model = self._train_final_model_on_all_genes(dataset_path, out_dir, feature_names, args)
        
        # Save model
        print(f"  💾 Saving trained model...", flush=True)
        model_path = out_dir / "model_multiclass.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(final_model, f)
        
        # Save feature manifest for post-training analysis
        print(f"  📄 Saving feature manifest...", flush=True)
        feature_manifest = pd.DataFrame({'feature': feature_names})
        feature_manifest.to_csv(out_dir / "feature_manifest.csv", index=False)
        
        # Also save as JSON for compatibility
        features_json = {"feature_names": feature_names}
        with open(out_dir / "train.features.json", 'w') as f:
            json.dump(features_json, f)
        
        # Save CV results
        print(f"  📊 Saving CV results...", flush=True)
        cv_df = pd.DataFrame(cv_results)
        cv_df.to_csv(out_dir / "gene_cv_metrics.csv", index=False)
        cv_df.to_csv(out_dir / "metrics_folds.tsv", sep="\t", index=False)
        
        # Save individual fold JSON files for compatibility with downstream analysis
        for fold_result in cv_results:
            fold_idx = fold_result['fold']
            # Ensure JSON serialisability of NumPy scalars
            row_serial = {
                k: (int(v) if isinstance(v, np.integer) else float(v) if isinstance(v, np.floating) else v)
                for k, v in fold_result.items()
            }
            with open(out_dir / f"metrics_fold{fold_idx}.json", "w") as fh:
                json.dump(row_serial, fh, indent=2)
        
        # Save aggregate metrics (excluding accuracy, focusing on meaningful metrics)
        mean_metrics = {
            "test_macro_f1": cv_df["test_macro_f1"].mean(),
            "test_macro_avg_precision": cv_df["test_macro_avg_precision"].mean() if "test_macro_avg_precision" in cv_df else 0.0,
            "top_k_accuracy": cv_df["top_k_accuracy"].mean(),
            "donor_f1": cv_df["donor_f1"].mean() if "donor_f1" in cv_df else 0.0,
            "acceptor_f1": cv_df["acceptor_f1"].mean() if "acceptor_f1" in cv_df else 0.0,
            "donor_ap": cv_df["donor_ap"].mean() if "donor_ap" in cv_df else 0.0,
            "acceptor_ap": cv_df["acceptor_ap"].mean() if "acceptor_ap" in cv_df else 0.0
        }
        with open(out_dir / "metrics_aggregate.json", 'w') as f:
            json.dump(mean_metrics, f, indent=2)
        
        # Run proper holdout evaluation for realistic metrics
        holdout_metrics = self._run_proper_holdout_evaluation(dataset_path, out_dir, feature_names, args)
        
        # Calculate performance metrics
        print(f"  📊 Calculating performance metrics...", flush=True)
        performance_metrics = self._calculate_performance_metrics(cv_results)
        
        # Add holdout metrics to performance summary
        if holdout_metrics:
            performance_metrics.update({
                'holdout_f1_macro': holdout_metrics.get('mean_f1_macro', 0.0),
                'holdout_ap_macro': holdout_metrics.get('mean_macro_ap', 0.0),
                'holdout_binary_auc': holdout_metrics.get('mean_binary_auc', 0.0),
                'holdout_binary_ap': holdout_metrics.get('mean_binary_ap', 0.0)
            })
        
        training_metadata = {
            'strategy': self.get_strategy_name(),
            'total_genes': len(np.unique(genes)),
            'total_positions': X.shape[0],
            'features_used': len(feature_names),
            'training_date': datetime.now().isoformat(),
            'cv_folds': args.n_folds,
            'n_estimators': args.n_estimators,
            'calibration': 'per_class' if args.calibrate_per_class else ('binary' if args.calibrate else 'none'),
            'training_genes': list(np.unique(genes))  # Track genes used for training
        }
        
        return TrainingResult(
            model_path=model_path,
            feature_names=feature_names,
            excluded_features=self._global_excluded_features,
            training_metadata=training_metadata,
            cv_results=cv_results,
            performance_metrics=performance_metrics
        )
    
    def _run_gene_aware_cv(self, X: np.ndarray, y: np.ndarray, genes: np.ndarray, feature_names: List[str], out_dir: Path, args) -> List[Dict]:
        """Run gene-aware cross-validation - extracted from existing code."""
        from sklearn.model_selection import GroupKFold, GroupShuffleSplit
        from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc, average_precision_score
        from meta_spliceai.splice_engine.meta_models.training.run_gene_cv_sigmoid import _train_binary_model, _binary_metrics
        
        gkf = GroupKFold(n_splits=args.n_folds)
        cv_results = []
        
        # Initialize calibration data collection
        self.per_class_calib_scores = [[] for _ in range(3)]  # For each class
        self.per_class_calib_labels = [[] for _ in range(3)]
        self.calib_scores = []  # For binary calibration
        self.calib_labels = []
        
        # Initialize data collection for comprehensive plotting (CRITICAL - was missing!)
        self.y_true_bins = []      # Binary truth labels for plotting
        self.y_prob_bases = []     # Base model probabilities for plotting  
        self.y_prob_metas = []     # Meta model probabilities for plotting
        self.y_true_multiclass = []        # Multiclass truth for enhanced plots
        self.y_prob_bases_multiclass = []  # Base multiclass probs for enhanced plots
        self.y_prob_metas_multiclass = []  # Meta multiclass probs for enhanced plots
        
        # Locate base probability columns (needed for base vs meta comparison)
        splice_prob_idx = None
        donor_idx = None
        acceptor_idx = None
        
        donor_present = getattr(args, 'donor_score_col', 'donor_score') in feature_names
        accept_present = getattr(args, 'acceptor_score_col', 'acceptor_score') in feature_names
        
        if donor_present and accept_present:
            donor_idx = feature_names.index(getattr(args, 'donor_score_col', 'donor_score'))
            acceptor_idx = feature_names.index(getattr(args, 'acceptor_score_col', 'acceptor_score'))
        else:
            splice_prob_col = getattr(args, 'splice_prob_col', 'score')
            if splice_prob_col in feature_names:
                splice_prob_idx = feature_names.index(splice_prob_col)
        
        for fold_idx, (train_val_idx, test_idx) in enumerate(gkf.split(X, y, groups=genes)):
            print(f"    🔀 Fold {fold_idx+1}/{args.n_folds}: {len(test_idx):,} test positions", flush=True)
            
            # Train/validation split
            rel_valid = args.valid_size / (1.0 - 1.0 / args.n_folds)
            gss = GroupShuffleSplit(n_splits=1, test_size=rel_valid, random_state=args.seed)
            train_idx, valid_idx = next(gss.split(train_val_idx, y[train_val_idx], groups=genes[train_val_idx]))
            train_idx = train_val_idx[train_idx]
            valid_idx = train_val_idx[valid_idx]
            
            # Train 3 binary models
            print(f"      🎯 Training 3 binary classifiers for fold {fold_idx+1}...", flush=True)
            fold_models = []
            for cls in (0, 1, 2):
                class_name = ['neither', 'donor', 'acceptor'][cls]
                print(f"        🔧 Training {class_name} classifier...", flush=True)
                y_train_bin = (y[train_idx] == cls).astype(int)
                y_val_bin = (y[valid_idx] == cls).astype(int)
                model_c, _ = _train_binary_model(X[train_idx], y_train_bin, X[valid_idx], y_val_bin, args)
                fold_models.append(model_c)
            
            # Evaluate on test set and collect calibration data
            print(f"      📊 Evaluating fold {fold_idx+1} performance...", flush=True)
            proba_parts = [m.predict_proba(X[test_idx])[:, 1] for m in fold_models]
            proba = np.column_stack(proba_parts)
            pred = proba.argmax(axis=1)
            
            # Collect calibration data from validation set
            if args.calibrate_per_class or args.calibrate:
                val_proba_parts = [m.predict_proba(X[valid_idx])[:, 1] for m in fold_models]
                val_proba = np.column_stack(val_proba_parts)
                
                # Per-class calibration data
                if args.calibrate_per_class:
                    for cls_idx in range(3):
                        self.per_class_calib_scores[cls_idx].append(val_proba[:, cls_idx])
                        self.per_class_calib_labels[cls_idx].append((y[valid_idx] == cls_idx).astype(int))
                
                # Binary calibration data (splice vs non-splice)
                if args.calibrate:
                    splice_scores = val_proba[:, 1] + val_proba[:, 2]  # donor + acceptor
                    splice_labels = (y[valid_idx] != 0).astype(int)  # any splice site
                    self.calib_scores.append(splice_scores)
                    self.calib_labels.append(splice_labels)
            
            accuracy = accuracy_score(y[test_idx], pred)
            f1_macro = f1_score(y[test_idx], pred, average="macro")
            test_genes = len(np.unique(genes[test_idx]))
            
            # Calculate per-class AP for immediate display
            try:
                from sklearn.metrics import average_precision_score
                y_test = y[test_idx]
                donor_ap = average_precision_score(y_test == 1, proba[:, 1])
                acceptor_ap = average_precision_score(y_test == 2, proba[:, 2])
                test_macro_avg_precision = (donor_ap + acceptor_ap) / 2
            except Exception as e:
                if self.verbose:
                    print(f"        ⚠️  Error calculating AP metrics: {e}")
                test_macro_avg_precision = 0.0
            
            # Display confusion matrix for tracking training progress
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(y[test_idx], pred, labels=[0, 1, 2])
            print(f"      📋 Confusion Matrix (Fold {fold_idx+1}):", flush=True)
            print(f"           Pred:  Neither  Donor  Acceptor", flush=True)
            print(f"        Neither: {cm[0,0]:8d} {cm[0,1]:6d} {cm[0,2]:9d}", flush=True)
            print(f"          Donor: {cm[1,0]:8d} {cm[1,1]:6d} {cm[1,2]:9d}", flush=True)
            print(f"       Acceptor: {cm[2,0]:8d} {cm[2,1]:6d} {cm[2,2]:9d}", flush=True)
            
            # Calculate top-k accuracy (prefer transcript-level when available, fallback to gene-level)
            try:
                # First, try transcript-level top-k (SpliceAI standard)
                if hasattr(self, '_transcript_data') and self._transcript_data is not None and len(self._transcript_data) >= len(test_idx):
                    from meta_spliceai.splice_engine.meta_models.training.transcript_topk_accuracy import (
                        calculate_transcript_top_k_for_cv_fold
                    )
                    
                    # Get transcript IDs for test samples
                    transcript_ids = self._transcript_data['transcript_id'].iloc[test_idx].values
                    
                    # Get positions if available, otherwise use test indices
                    if 'position' in self._transcript_data.columns:
                        positions = self._transcript_data['position'].iloc[test_idx].values
                    else:
                        positions = test_idx  # Use indices as fallback positions
                    
                    topk_metrics = calculate_transcript_top_k_for_cv_fold(
                        X[test_idx], y[test_idx], proba, transcript_ids, positions,
                        donor_label=1, acceptor_label=2, neither_label=0, verbose=False
                    )
                    
                    top_k_combined = topk_metrics["combined_top_k"]
                    top_k_donor = topk_metrics["donor_top_k"] 
                    top_k_acceptor = topk_metrics["acceptor_top_k"]
                    topk_method = "transcript-level"
                    
                else:
                    # Fallback to gene-level top-k
                    from meta_spliceai.splice_engine.meta_models.evaluation.top_k_metrics import (
                        calculate_cv_fold_top_k
                    )
                    
                    gene_ids_test = genes[test_idx]
                    gene_top_k_metrics = calculate_cv_fold_top_k(
                        X=X[test_idx], 
                        y=y[test_idx], 
                        probs=proba, 
                        gene_ids=gene_ids_test,
                        donor_label=1, acceptor_label=2, neither_label=0,
                    )
                    
                    top_k_combined = gene_top_k_metrics["combined_top_k"]
                    top_k_donor = gene_top_k_metrics["donor_top_k"]
                    top_k_acceptor = gene_top_k_metrics["acceptor_top_k"]
                    topk_method = "gene-level"
                
            except Exception as e:
                if self.verbose:
                    print(f"        ⚠️  Error calculating top-k metrics: {e}")
                top_k_combined = 0.0
                top_k_donor = 0.0
                top_k_acceptor = 0.0
                topk_method = "failed"
            
            print(f"      ✅ Fold {fold_idx+1} results: F1={f1_macro:.3f}, AP={test_macro_avg_precision:.3f}, Top-k={top_k_combined:.3f} ({topk_method}, {test_genes} genes)", flush=True)
            
            # Calculate base vs meta comparison metrics for CV visualization
            y_true_bin = (y[test_idx] != 0).astype(int)  # Binary: splice vs non-splice
            y_prob_meta = proba[:, 1] + proba[:, 2]  # Meta model splice probability
            
            # Base model splice probability
            if splice_prob_idx is not None:
                y_prob_base = X[test_idx, splice_prob_idx]
            elif donor_idx is not None and acceptor_idx is not None:
                y_prob_base = X[test_idx, donor_idx] + X[test_idx, acceptor_idx]
            else:
                # Fallback: use meta probabilities (no comparison possible)
                y_prob_base = y_prob_meta
            
            # Calculate binary metrics (simplified - remove problematic top-k)
            base_thresh = getattr(args, 'base_thresh', 0.5)
            meta_thresh = getattr(args, 'threshold', 0.5) or 0.5
            
            # Use simplified binary metrics without top-k for base/meta comparison
            y_pred_base = (y_prob_base >= base_thresh).astype(int)
            y_pred_meta = (y_prob_meta >= meta_thresh).astype(int)
            
            from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
            
            # Base metrics
            tn_b, fp_b, fn_b, tp_b = confusion_matrix(y_true_bin, y_pred_base, labels=[0, 1]).ravel()
            base_metrics = {
                "tn": tn_b, "fp": fp_b, "fn": fn_b, "tp": tp_b,
                "precision": precision_score(y_true_bin, y_pred_base, zero_division=0),
                "recall": recall_score(y_true_bin, y_pred_base, zero_division=0),
                "f1": f1_score(y_true_bin, y_pred_base, zero_division=0),
                "topk_acc": f1_score(y_true_bin, y_pred_base, zero_division=0)  # Use F1 as proxy
            }
            
            # Meta metrics
            tn_m, fp_m, fn_m, tp_m = confusion_matrix(y_true_bin, y_pred_meta, labels=[0, 1]).ravel()
            meta_metrics = {
                "tn": tn_m, "fp": fp_m, "fn": fn_m, "tp": tp_m,
                "precision": precision_score(y_true_bin, y_pred_meta, zero_division=0),
                "recall": recall_score(y_true_bin, y_pred_meta, zero_division=0),
                "f1": f1_score(y_true_bin, y_pred_meta, zero_division=0),
                "topk_acc": f1_score(y_true_bin, y_pred_meta, zero_division=0)  # Use F1 as proxy
            }
            
            # Calculate AUC and AP
            try:
                from sklearn.metrics import roc_curve, auc, average_precision_score
                # Base model ROC/PR
                fpr_b, tpr_b, _ = roc_curve(y_true_bin, y_prob_base)
                auc_base = auc(fpr_b, tpr_b)
                ap_base = average_precision_score(y_true_bin, y_prob_base)
                
                # Meta model ROC/PR
                fpr_m, tpr_m, _ = roc_curve(y_true_bin, y_prob_meta)
                auc_meta = auc(fpr_m, tpr_m)
                ap_meta = average_precision_score(y_true_bin, y_prob_meta)
            except Exception:
                auc_base = auc_meta = ap_base = ap_meta = 0.0
            
            # Calculate PROPER transcript-level top-k accuracy
            transcript_topk_metrics = {'combined_top_k': 0.0}  # Default fallback
            try:
                # Check if we have transcript information in the feature matrix
                # (should be preserved by preprocessing with preserve_transcript_columns=True)
                if hasattr(self, '_transcript_data') and self._transcript_data is not None and len(self._transcript_data) >= len(test_idx):
                    from meta_spliceai.splice_engine.meta_models.training.transcript_topk_accuracy import (
                        calculate_transcript_top_k_for_cv_fold
                    )
                    
                    # Get transcript IDs for test samples
                    transcript_ids = self._transcript_data['transcript_id'].iloc[test_idx].values
                    
                    # Get positions if available, otherwise use test indices
                    if 'position' in self._transcript_data.columns:
                        positions = self._transcript_data['position'].iloc[test_idx].values
                    else:
                        positions = test_idx  # Use indices as fallback positions
                    
                    transcript_topk_metrics = calculate_transcript_top_k_for_cv_fold(
                        X[test_idx], y[test_idx], proba, transcript_ids, positions,
                        donor_label=1, acceptor_label=2, neither_label=0, verbose=False
                    )
                    
                    if self.verbose:
                        print(f"        📊 Transcript top-k: Donor={transcript_topk_metrics['donor_top_k']:.3f}, "
                              f"Acceptor={transcript_topk_metrics['acceptor_top_k']:.3f}, "
                              f"Combined={transcript_topk_metrics['combined_top_k']:.3f}")
                        print(f"        📊 Per-class AP: Donor={donor_ap:.3f}, Acceptor={acceptor_ap:.3f}, Macro={test_macro_avg_precision:.3f}")
                else:
                    if self.verbose:
                        print(f"        ⚠️  No transcript data available for top-k calculation")
                        
            except Exception as e:
                if self.verbose:
                    print(f"        ⚠️  Transcript top-k calculation failed: {e}")
                transcript_topk_metrics = {'combined_top_k': 0.0}
            
            # Calculate per-class F1 scores (AP already calculated above)
            try:
                from sklearn.metrics import f1_score
                y_test = y[test_idx]
                
                # Per-class F1 scores
                donor_f1 = f1_score(y_test == 1, pred == 1, average='binary')
                acceptor_f1 = f1_score(y_test == 2, pred == 2, average='binary')
                
                # AP values already calculated above for immediate display
                # donor_ap and acceptor_ap are already available from earlier calculation
            except Exception as e:
                if self.verbose:
                    print(f"        ⚠️  Error calculating per-class F1 metrics: {e}")
                donor_f1 = acceptor_f1 = 0.0
            
            cv_results.append({
                'fold': fold_idx,
                'test_rows': len(test_idx),  # Match original column name
                'test_macro_f1': f1_macro,  # Main macro F1 score
                'test_macro_avg_precision': test_macro_avg_precision,  # Macro average precision
                'splice_macro_f1': f1_macro,  # For compatibility (splice-only F1 is same as macro F1 for 3-class)
                'top_k_accuracy': top_k_combined,  # Use gene-level top-k (calculated above)
                'top_k_donor': top_k_donor,
                'top_k_acceptor': top_k_acceptor,
                'top_k_n_genes': topk_metrics.get('n_groups', 0) if 'topk_metrics' in locals() else 0,  # Number of transcripts/genes evaluated
                'donor_f1': donor_f1,
                'acceptor_f1': acceptor_f1,
                'donor_ap': donor_ap,
                'acceptor_ap': acceptor_ap,
                # Base vs meta binary metrics
                'base_f1': base_metrics['f1'],
                'meta_f1': meta_metrics['f1'],
                'base_topk': base_metrics['topk_acc'],
                'meta_topk': meta_metrics['topk_acc'],
                'delta_fp': base_metrics['fp'] - meta_metrics['fp'],
                'delta_fn': base_metrics['fn'] - meta_metrics['fn'],
                'auc_base': auc_base,
                'auc_meta': auc_meta,
                'ap_base': ap_base,
                'ap_meta': ap_meta
            })
            
            # CRITICAL: Collect data for comprehensive plotting (was missing!)
            self.y_true_bins.append(y_true_bin)
            self.y_prob_bases.append(y_prob_base)
            self.y_prob_metas.append(y_prob_meta)
            self.y_true_multiclass.append(y[test_idx])
            
            # Reconstruct base multiclass probabilities for enhanced plotting
            if splice_prob_idx is not None:
                base_prob_splice = X[test_idx, splice_prob_idx]
                base_prob_neither = 1 - base_prob_splice
                base_prob_donor = base_prob_splice * 0.5
                base_prob_acceptor = base_prob_splice * 0.5
                y_prob_base_multiclass = np.column_stack([base_prob_neither, base_prob_donor, base_prob_acceptor])
            elif donor_idx is not None and acceptor_idx is not None:
                base_prob_donor = X[test_idx, donor_idx]
                base_prob_acceptor = X[test_idx, acceptor_idx]
                base_prob_neither = 1 - (base_prob_donor + base_prob_acceptor)
                base_prob_neither = np.clip(base_prob_neither, 0, 1)
                y_prob_base_multiclass = np.column_stack([base_prob_neither, base_prob_donor, base_prob_acceptor])
            else:
                # Fallback: approximate from meta probabilities
                y_prob_base_multiclass = proba.copy()
            
            self.y_prob_bases_multiclass.append(y_prob_base_multiclass)
            self.y_prob_metas_multiclass.append(proba)
        
        # Print CV summary statistics
        if cv_results:
            f1_scores = [r['test_macro_f1'] for r in cv_results]
            ap_scores = [r.get('test_macro_avg_precision', 0.0) for r in cv_results]
            total_genes = sum(r.get('test_genes', 0) for r in cv_results)
            
            print(f"  📊 Cross-Validation Summary:", flush=True)
            print(f"    🎯 Mean F1 Macro: {np.mean(f1_scores):.3f} ± {np.std(f1_scores):.3f}", flush=True)
            print(f"    🎯 Mean AP Macro: {np.mean(ap_scores):.3f} ± {np.std(ap_scores):.3f}", flush=True)
            print(f"    🧬 Total genes evaluated: {total_genes} across {len(cv_results)} folds", flush=True)
        
        # Generate comprehensive ROC/PR plots after CV (CRITICAL - was missing!)
        self._generate_comprehensive_plots(out_dir, args)
        
        return cv_results
    
    def _generate_comprehensive_plots(self, out_dir: Path, args):
        """Generate comprehensive ROC/PR plots using collected CV data."""
        if not hasattr(self, 'y_true_bins') or not self.y_true_bins:
            print(f"  ⚠️  No plotting data collected during CV, skipping plot generation")
            return
            
        print(f"\n📊 [Enhanced Plotting] Generating comprehensive ROC/PR visualizations...", flush=True)
        
        try:
            # 1. Generate base vs meta ROC/PR comparison plots
            from meta_spliceai.splice_engine.meta_models.evaluation.viz_utils import plot_roc_pr_curves
            
            print(f"  📈 Creating base vs meta ROC/PR curves...", flush=True)
            curve_metrics = plot_roc_pr_curves(
                y_true=self.y_true_bins,
                y_pred_base=self.y_prob_bases,
                y_pred_meta=self.y_prob_metas,
                out_dir=out_dir,
                n_roc_points=getattr(args, 'n_roc_points', 101),
                plot_format=getattr(args, 'plot_format', 'pdf'),
                base_name='Base',
                meta_name='Meta',
                fold_ids=list(range(len(self.y_true_bins)))
            )
            print(f"  ✅ Base vs meta curves generated: roc_base_vs_meta.pdf, pr_base_vs_meta.pdf")
            
        except Exception as e:
            print(f"  ❌ Error generating base vs meta curves: {e}")
        
        try:
            # 2. Generate enhanced binary PR plots
            from meta_spliceai.splice_engine.meta_models.evaluation.multiclass_roc_pr import create_improved_binary_pr_plot
            
            print(f"  📈 Creating improved binary PR plots...", flush=True)
            create_improved_binary_pr_plot(
                y_true=self.y_true_bins,
                y_pred_base=self.y_prob_bases,
                y_pred_meta=self.y_prob_metas,
                out_dir=out_dir,
                plot_format=getattr(args, 'plot_format', 'pdf'),
                base_name='Base',
                meta_name='Meta'
            )
            print(f"  ✅ Enhanced binary PR plots generated: pr_binary_improved.pdf")
            
        except Exception as e:
            print(f"  ❌ Error generating enhanced binary PR plots: {e}")
        
        try:
            # 3. Generate multiclass ROC/PR curves
            from meta_spliceai.splice_engine.meta_models.evaluation.multiclass_roc_pr import plot_multiclass_roc_pr_curves
            
            print(f"  📈 Creating multiclass ROC/PR curves...", flush=True)
            multiclass_metrics = plot_multiclass_roc_pr_curves(
                y_true=self.y_true_multiclass,
                y_pred_base=self.y_prob_bases_multiclass,
                y_pred_meta=self.y_prob_metas_multiclass,
                out_dir=out_dir,
                plot_format=getattr(args, 'plot_format', 'pdf'),
                base_name='Base',
                meta_name='Meta'
            )
            print(f"  ✅ Multiclass curves generated: pr_donor_class.pdf, roc_acceptor_class.pdf, etc.")
            
        except Exception as e:
            print(f"  ❌ Error generating multiclass curves: {e}")
        
        try:
            # 4. Generate combined meta curves
            from meta_spliceai.splice_engine.meta_models.evaluation.viz_utils import plot_combined_roc_pr_curves_meta
            
            print(f"  📈 Creating combined meta ROC/PR curves...", flush=True)
            plot_combined_roc_pr_curves_meta(
                y_true=self.y_true_bins,
                y_pred_base=self.y_prob_bases,
                y_pred_meta=self.y_prob_metas,
                out_dir=out_dir,
                plot_format=getattr(args, 'plot_format', 'pdf'),
                base_name='Base',
                meta_name='Meta',
                n_roc_points=getattr(args, 'n_roc_points', 101)
            )
            print(f"  ✅ Combined meta curves generated: pr_curves_meta.pdf, roc_curves_meta.pdf")
            
        except Exception as e:
            print(f"  ❌ Error generating combined meta curves: {e}")
            
        print(f"📊 [Enhanced Plotting] Comprehensive visualization generation completed!", flush=True)
    
    def _train_final_model_on_all_genes(self, dataset_path: str, out_dir: Path, feature_names: List[str], args):
        """Train final model on available genes for maximum pattern learning."""
        from meta_spliceai.splice_engine.meta_models.training.run_gene_cv_sigmoid import _train_binary_model
        from meta_spliceai.splice_engine.meta_models.training.classifier_utils import _encode_labels
        
        # CRITICAL FIX: Respect sample_genes parameter for memory efficiency
        if hasattr(args, 'sample_genes') and args.sample_genes is not None:
            print(f"    🧬 Gene sampling mode: Training final model on sampled data to avoid memory issues", flush=True)
            print(f"    💡 Using existing sample data instead of loading full dataset (--sample-genes {args.sample_genes})", flush=True)
            
            # Use the CV data we already have - it's the same sample
            if hasattr(self, '_X_cv') and hasattr(self, '_y_cv'):
                return self._train_final_model_from_cv_data(feature_names, args)
            else:
                # Load the same sample as used in CV
                from meta_spliceai.splice_engine.meta_models.training.label_utils import load_dataset_sample
                print(f"    📊 Loading sample dataset for final training ({args.sample_genes} genes)...", flush=True)
                full_df = load_dataset_sample(dataset_path, sample_genes=args.sample_genes, random_seed=args.seed)
                unique_genes = full_df['gene_id'].unique()
                print(f"    ✅ Loaded {len(full_df):,} positions from {len(unique_genes):,} genes (sample)", flush=True)
        else:
            print(f"    🌍 Loading ALL available genes for final model training...", flush=True)
            
            # Load ALL genes (not just CV subset) with memory optimization
            try:
                # Use the same loading approach as CV but without gene sampling
                from meta_spliceai.splice_engine.meta_models.training import datasets
                
                # Load full dataset
                print(f"    📊 Loading full dataset for final training...", flush=True)
                full_df = datasets.load_dataset(dataset_path)
                
                unique_genes = full_df['gene_id'].unique()
                print(f"    ✅ Loaded {len(full_df):,} positions from {len(unique_genes):,} genes", flush=True)
            
            except Exception as e:
                print(f"    ⚠️  Failed to load full dataset ({e}), using CV subset...", flush=True)
                # Fallback: use the data already loaded
                if hasattr(self, '_X_cv') and hasattr(self, '_y_cv'):
                    return self._train_final_model_from_cv_data(feature_names, args)
                else:
                    raise RuntimeError("No data available for final model training")
        
        # Prepare full dataset for training
        from meta_spliceai.splice_engine.meta_models.builder import preprocessing
        
        print(f"    🔧 Preprocessing full dataset...", flush=True)
        X_full_df, y_full_series = preprocessing.prepare_training_data(
            full_df, 
            label_col="splice_type", 
            return_type="pandas", 
            verbose=0,
            encode_chrom=True
        )
        
        # Apply the same feature exclusions as CV
        X_full_df = self.apply_global_feature_exclusions(X_full_df)
        
        # Ensure feature consistency with CV
        X_full_df = X_full_df[feature_names]  # Use same features as CV
        
        X_full = X_full_df.values
        y_full = _encode_labels(y_full_series)
        
        print(f"    🎯 Training final model on {X_full.shape[0]:,} positions from {full_df['gene_id'].n_unique():,} genes...", flush=True)
        
        # Train 3 binary models on full data
        models_full = []
        for cls in (0, 1, 2):
            class_name = ['neither', 'donor', 'acceptor'][cls]
            print(f"      🔧 Training final {class_name} classifier...", flush=True)
            y_bin = (y_full == cls).astype(int)
            model_c, _ = _train_binary_model(X_full, y_bin, X_full, y_bin, args)
            models_full.append(model_c)
        
        # Create ensemble based on calibration settings (using CV calibration data)
        print(f"      🔧 Creating ensemble model...", flush=True)
        if args.calibrate_per_class:
            print(f"        📊 Using per-class calibration", flush=True)
            # Train calibrators using collected CV data
            calibrators = self._train_per_class_calibrators(args)
            ensemble = PerClassCalibratedSigmoidEnsemble(models_full, feature_names, calibrators)
        elif args.calibrate:
            print(f"        📊 Using binary calibration", flush=True)
            from meta_spliceai.splice_engine.meta_models.training import classifier_utils as _cutils
            # Train binary calibrator using collected CV data
            calibrator = self._train_binary_calibrator(args)
            ensemble = _cutils.CalibratedSigmoidEnsemble(models_full, feature_names, calibrator)
        else:
            print(f"        📊 Using uncalibrated ensemble", flush=True)
            ensemble = SigmoidEnsemble(models_full, feature_names)
        
        print(f"      ✅ Final ensemble model trained on {X_full.shape[0]:,} positions!", flush=True)
        return ensemble

    def _train_final_model_from_cv_data(self, feature_names: List[str], args):
        """Train final model using CV data when full dataset cannot be loaded."""
        from meta_spliceai.splice_engine.meta_models.training.run_gene_cv_sigmoid import _train_binary_model
        
        print(f"    🔄 Training final model using CV data...", flush=True)
        
        if not hasattr(self, '_X_cv') or not hasattr(self, '_y_cv'):
            raise RuntimeError("No CV data available for final model training")
        
        X_full = self._X_cv
        y_full = self._y_cv
        
        print(f"    🎯 Training final model on {X_full.shape[0]:,} CV positions...", flush=True)
        
        # Train 3 binary models on CV data
        models_full = []
        for cls in (0, 1, 2):
            class_name = ['neither', 'donor', 'acceptor'][cls]
            print(f"      🔧 Training final {class_name} classifier...", flush=True)
            y_bin = (y_full == cls).astype(int)
            model_c, _ = _train_binary_model(X_full, y_bin, X_full, y_bin, args)
            models_full.append(model_c)
        
        # Create ensemble based on calibration settings
        print(f"      🔧 Creating ensemble model...", flush=True)
        if args.calibrate_per_class:
            print(f"        📊 Using per-class calibration", flush=True)
            calibrators = self._train_per_class_calibrators(args)
            ensemble = PerClassCalibratedSigmoidEnsemble(models_full, feature_names, calibrators)
        elif args.calibrate:
            print(f"        📊 Using binary calibration", flush=True)
            from meta_spliceai.splice_engine.meta_models.training import classifier_utils as _cutils
            calibrator = self._train_binary_calibrator(args)
            ensemble = _cutils.CalibratedSigmoidEnsemble(models_full, feature_names, calibrator)
        else:
            print(f"        📊 Using uncalibrated ensemble", flush=True)
            ensemble = SigmoidEnsemble(models_full, feature_names)
        
        print(f"      ✅ Final ensemble model trained on CV data!", flush=True)
        return ensemble
    
    def _train_per_class_calibrators(self, args):
        """Train per-class calibrators using collected CV data."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.isotonic import IsotonicRegression
        
        calibrators = []
        for cls_idx in range(3):
            if not self.per_class_calib_scores[cls_idx]:
                print(f"        ⚠️  No calibration data for class {cls_idx}, using identity calibrator")
                calibrators.append(None)
                continue
                
            # Concatenate scores and labels from all CV folds for this class
            cls_scores = np.concatenate(self.per_class_calib_scores[cls_idx])
            cls_labels = np.concatenate(self.per_class_calib_labels[cls_idx])
            
            print(f"        📊 Class {cls_idx}: {cls_scores.shape[0]} samples, {cls_labels.sum()} positives")
            
            # Create and fit the calibrator
            if args.calib_method == "platt":
                calibrator = LogisticRegression(class_weight="balanced", max_iter=1000, solver="lbfgs")
                calibrator.fit(cls_scores.reshape(-1, 1), cls_labels)
            elif args.calib_method == "isotonic":
                calibrator = IsotonicRegression(out_of_bounds="clip")
                calibrator.fit(cls_scores, cls_labels)
            else:
                raise ValueError(f"Unsupported calibration method: {args.calib_method}")
            
            calibrators.append(calibrator)
        
        return calibrators
    
    def _train_binary_calibrator(self, args):
        """Train binary calibrator using collected CV data."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.isotonic import IsotonicRegression
        
        if not self.calib_scores:
            print(f"        ⚠️  No calibration data for binary calibration")
            return None
            
        # Concatenate scores and labels from all CV folds
        s_train = np.concatenate(self.calib_scores)
        y_bin = np.concatenate(self.calib_labels)
        
        print(f"        📊 Binary calibration: {s_train.shape[0]} samples, {y_bin.sum()} splice sites")
        
        # Create and fit the calibrator
        if args.calib_method == "platt":
            calibrator = LogisticRegression(class_weight="balanced", max_iter=1000, solver="lbfgs")
            calibrator.fit(s_train.reshape(-1, 1), y_bin)
        elif args.calib_method == "isotonic":
            calibrator = IsotonicRegression(out_of_bounds="clip")
            calibrator.fit(s_train, y_bin)
        else:
            raise ValueError(f"Unsupported calibration method: {args.calib_method}")
        
        return calibrator

    def _run_proper_holdout_evaluation(self, dataset_path: str, out_dir: Path, feature_names: List[str], args) -> Dict[str, Any]:
        """
        Run proper gene-aware holdout evaluation to get realistic performance metrics.
        
        This addresses the evaluation optimism issue (PR-AUC = 1.0) by ensuring
        true train/test separation at the gene level.
        """
        print(f"\n🎯 [Holdout Evaluation] Running gene-aware holdout evaluation for realistic metrics...", flush=True)
        print(f"  ℹ️  This trains a SEPARATE evaluation model (80% genes) to evaluate on unseen genes (20%)")
        print(f"  ℹ️  This is DIFFERENT from the final production model (trained on ALL genes)")
        print(f"  ℹ️  Purpose: Get realistic performance estimates without data leakage")
        
        try:
            from meta_spliceai.splice_engine.meta_models.training import datasets
            from meta_spliceai.splice_engine.meta_models.builder import preprocessing
            from meta_spliceai.splice_engine.meta_models.training.classifier_utils import _encode_labels
            from sklearn.model_selection import GroupShuffleSplit
            from sklearn.metrics import f1_score, average_precision_score, roc_auc_score
            
            # CRITICAL FIX: Respect sample_genes parameter for holdout evaluation
            if hasattr(args, 'sample_genes') and args.sample_genes is not None:
                print(f"  📊 Gene sampling mode: Loading sample for holdout evaluation ({args.sample_genes} genes)...", flush=True)
                from meta_spliceai.splice_engine.meta_models.training.label_utils import load_dataset_sample
                df = load_dataset_sample(dataset_path, sample_genes=args.sample_genes, random_seed=args.seed)
                print(f"  💡 Using sample data instead of full dataset for memory efficiency", flush=True)
            else:
                print(f"  📊 Loading full dataset for holdout evaluation...", flush=True)
                df = datasets.load_dataset(dataset_path)
            
            X_df, y_series = preprocessing.prepare_training_data(
                df, label_col="splice_type", return_type="pandas", verbose=0, encode_chrom=True
            )
            
            # Apply same feature exclusions as global screening
            X_df = self.apply_global_feature_exclusions(X_df)
            
            # CRITICAL: Remove metadata columns that shouldn't be used for training
            metadata_cols_to_remove = ['transcript_id', 'position']
            for col in metadata_cols_to_remove:
                if col in X_df.columns:
                    X_df = X_df.drop(columns=[col])
            
            # Use the same feature set as the final model for consistency
            # But this evaluation model will be trained only on 80% of genes
            available_features = [f for f in feature_names if f in X_df.columns]
            X_df = X_df[available_features]
            
            print(f"  🔧 Holdout evaluation features: {len(available_features)} (after exclusions)")
            if len(available_features) != len(feature_names):
                missing_features = set(feature_names) - set(available_features)
                print(f"  ⚠️  Missing features in holdout data: {missing_features}")
            
            X = X_df.values
            y = _encode_labels(y_series)
            genes = df['gene_id'].to_numpy()
            
            # Gene-aware 80/20 split for realistic evaluation
            gss = GroupShuffleSplit(n_splits=3, test_size=0.2, random_state=42)  # Multiple splits for robustness
            
            holdout_results = []
            
            for split_idx, (train_idx, test_idx) in enumerate(gss.split(X, y, groups=genes)):
                train_genes = np.unique(genes[train_idx])
                test_genes = np.unique(genes[test_idx])
                
                print(f"  🔀 Holdout split {split_idx+1}/3: {len(train_genes)} train genes, {len(test_genes)} test genes", flush=True)
                print(f"    📊 Train: {len(train_idx):,} positions, Test: {len(test_idx):,} positions", flush=True)
                
                # Verify no gene overlap between train and test
                gene_overlap = set(train_genes) & set(test_genes)
                if gene_overlap:
                    print(f"  ❌ ERROR: Gene overlap detected in split {split_idx+1}: {len(gene_overlap)} genes")
                    continue
                else:
                    print(f"  ✅ No gene overlap - proper holdout split")
                
                # Train SEPARATE evaluation model on training genes only (NOT the final model)
                print(f"    🔧 Training evaluation model on {len(train_genes)} genes (excludes leaky features)")
                holdout_models = []
                
                # For simplicity and to avoid indexing issues, train on all training data without validation split
                # This is acceptable for holdout evaluation since we're testing on completely unseen genes
                for cls in (0, 1, 2):
                    class_name = ['neither', 'donor', 'acceptor'][cls]
                    
                    # Create binary labels for this class using the training data
                    y_train_bin = (y[train_idx] == cls).astype(int)
                    
                    print(f"      🎯 Training {class_name} evaluation classifier...")
                    from meta_spliceai.splice_engine.meta_models.training.run_gene_cv_sigmoid import _train_binary_model
                    
                    # Use the training data as both train and validation to avoid indexing issues
                    # Since we're evaluating on completely unseen genes, this is acceptable
                    model_c, _ = _train_binary_model(
                        X[train_idx], y_train_bin, 
                        X[train_idx], y_train_bin,  # Use same data for validation to avoid indexing issues
                        args
                    )
                    holdout_models.append(model_c)
                
                # Save evaluation model for reference (separate from final production model)
                eval_model_path = out_dir / f"evaluation_model_split_{split_idx}.pkl"
                from meta_spliceai.splice_engine.meta_models.training.classifier_utils import SigmoidEnsemble
                eval_ensemble = SigmoidEnsemble(holdout_models, available_features)
                with open(eval_model_path, 'wb') as f:
                    pickle.dump(eval_ensemble, f)
                print(f"    💾 Evaluation model saved: {eval_model_path.name}")
                
                # Evaluate on completely unseen test genes
                proba_parts = [m.predict_proba(X[test_idx])[:, 1] for m in holdout_models]
                proba = np.column_stack(proba_parts)
                pred = proba.argmax(axis=1)
                
                # Calculate realistic metrics on unseen genes
                f1_macro = f1_score(y[test_idx], pred, average="macro")
                
                # Per-class metrics
                y_test = y[test_idx]
                donor_f1 = f1_score(y_test == 1, pred == 1, average='binary')
                acceptor_f1 = f1_score(y_test == 2, pred == 2, average='binary')
                donor_ap = average_precision_score(y_test == 1, proba[:, 1])
                acceptor_ap = average_precision_score(y_test == 2, proba[:, 2])
                macro_ap = (donor_ap + acceptor_ap) / 2
                
                # Binary splice vs non-splice metrics
                y_true_bin = (y[test_idx] != 0).astype(int)
                y_prob_meta_bin = proba[:, 1] + proba[:, 2]
                
                try:
                    binary_auc = roc_auc_score(y_true_bin, y_prob_meta_bin)
                    binary_ap = average_precision_score(y_true_bin, y_prob_meta_bin)
                except Exception:
                    binary_auc = binary_ap = 0.0
                
                holdout_result = {
                    'split': split_idx,
                    'train_genes': len(train_genes),
                    'test_genes': len(test_genes),
                    'train_positions': len(train_idx),
                    'test_positions': len(test_idx),
                    'f1_macro': f1_macro,
                    'donor_f1': donor_f1,
                    'acceptor_f1': acceptor_f1,
                    'donor_ap': donor_ap,
                    'acceptor_ap': acceptor_ap,
                    'macro_ap': macro_ap,
                    'binary_auc': binary_auc,
                    'binary_ap': binary_ap
                }
                
                holdout_results.append(holdout_result)
                
                print(f"    📈 Holdout metrics: F1={f1_macro:.3f}, AP={macro_ap:.3f}, AUC={binary_auc:.3f}")
                print(f"    📊 Per-class: Donor F1={donor_f1:.3f}/AP={donor_ap:.3f}, Acceptor F1={acceptor_f1:.3f}/AP={acceptor_ap:.3f}")
            
            # Save holdout evaluation results
            if holdout_results:
                import pandas as pd
                holdout_df = pd.DataFrame(holdout_results)
                holdout_path = out_dir / "holdout_evaluation_results.csv"
                holdout_df.to_csv(holdout_path, index=False)
                
                # Calculate summary statistics
                summary_stats = {
                    'mean_f1_macro': holdout_df['f1_macro'].mean(),
                    'std_f1_macro': holdout_df['f1_macro'].std(),
                    'mean_macro_ap': holdout_df['macro_ap'].mean(),
                    'std_macro_ap': holdout_df['macro_ap'].std(),
                    'mean_binary_auc': holdout_df['binary_auc'].mean(),
                    'std_binary_auc': holdout_df['binary_auc'].std(),
                    'mean_binary_ap': holdout_df['binary_ap'].mean(),
                    'std_binary_ap': holdout_df['binary_ap'].std(),
                    'n_splits': len(holdout_results)
                }
                
                # Save summary
                with open(out_dir / "holdout_evaluation_summary.json", 'w') as f:
                    json.dump(summary_stats, f, indent=2)
                
                print(f"\n🎯 [Holdout Evaluation] REALISTIC PERFORMANCE METRICS:")
                print(f"  📊 F1 Macro: {summary_stats['mean_f1_macro']:.3f} ± {summary_stats['std_f1_macro']:.3f}")
                print(f"  📊 AP Macro: {summary_stats['mean_macro_ap']:.3f} ± {summary_stats['std_macro_ap']:.3f}")
                print(f"  📊 Binary AUC: {summary_stats['mean_binary_auc']:.3f} ± {summary_stats['std_binary_auc']:.3f}")
                print(f"  📊 Binary AP: {summary_stats['mean_binary_ap']:.3f} ± {summary_stats['std_binary_ap']:.3f}")
                print(f"  ✅ Results saved: {holdout_path}")
                print(f"\n🔍 [Model Architecture Summary]:")
                print(f"  🎯 Final Production Model: Trained on ALL {len(np.unique(genes)):,} genes → model_multiclass.pkl")
                print(f"  🔬 Evaluation Models: Trained on ~80% genes → evaluation_model_split_*.pkl")
                print(f"  ✅ No data leakage: Evaluation models tested on completely unseen genes")
                
                return summary_stats
            else:
                print(f"  ❌ No valid holdout splits generated")
                return {}
                
        except Exception as e:
            print(f"  ❌ Holdout evaluation failed: {e}")
            if self.verbose:
                import traceback
                traceback.print_exc()
            return {}

    def _calculate_performance_metrics(self, cv_results: List[Dict]) -> Dict[str, float]:
        """Calculate aggregate performance metrics from CV results."""
        if not cv_results:
            return {}
        
        # Focus on meaningful metrics for imbalanced data
        f1_scores = [r.get('f1_macro', r.get('test_macro_f1', 0)) for r in cv_results]
        avg_precision_scores = [r.get('avg_precision_macro', r.get('test_macro_avg_precision', 0)) for r in cv_results]
        top_k_scores = [r.get('top_k_accuracy', 0) for r in cv_results]
        
        metrics = {
            'mean_f1_macro': np.mean(f1_scores),
            'std_f1_macro': np.std(f1_scores),
            'mean_avg_precision_macro': np.mean(avg_precision_scores) if avg_precision_scores else 0.0,
            'std_avg_precision_macro': np.std(avg_precision_scores) if avg_precision_scores else 0.0,
            'mean_top_k_accuracy': np.mean(top_k_scores) if top_k_scores else 0.0,
            'std_top_k_accuracy': np.std(top_k_scores) if top_k_scores else 0.0,
            'total_cv_folds': len(cv_results)
        }
        
        # Add per-class metrics if available
        for class_name in ['donor', 'acceptor']:
            f1_key = f'{class_name}_f1'
            ap_key = f'{class_name}_ap'
            
            if any(f1_key in r for r in cv_results):
                f1_values = [r.get(f1_key, 0) for r in cv_results]
                metrics[f'mean_{f1_key}'] = np.mean(f1_values)
                metrics[f'std_{f1_key}'] = np.std(f1_values)
                
            if any(ap_key in r for r in cv_results):
                ap_values = [r.get(ap_key, 0) for r in cv_results]
                metrics[f'mean_{ap_key}'] = np.mean(ap_values)
                metrics[f'std_{ap_key}'] = np.std(ap_values)
        
        return metrics


class BatchEnsembleTrainingStrategy(TrainingStrategy):
    """Batch ensemble training strategy for large datasets."""
    
    def __init__(self, max_genes_per_batch: int = 1200, max_memory_gb: float = 12.0, verbose: bool = True):
        super().__init__(verbose)
        self.max_genes_per_batch = max_genes_per_batch
        self.max_memory_gb = max_memory_gb
    
    def get_strategy_name(self) -> str:
        return f"Batch Ensemble ({self.max_genes_per_batch} genes/batch)"
    
    def can_handle_dataset_size(self, total_genes: int, estimated_memory_gb: float) -> bool:
        """Batch ensemble can handle any dataset size."""
        return True  # Always capable due to batching
    
    def train_model(
        self,
        dataset_path: str,
        out_dir: Path,
        args,
        X_df: pd.DataFrame,
        y_series: pd.Series,
        genes: np.ndarray
    ) -> TrainingResult:
        """Train batch ensemble model."""
        
        if self.verbose:
            print(f"\n🔥 [Batch Ensemble Training] Training ensemble from multiple batches...")
            print(f"  Total genes: {len(np.unique(genes)):,}")
            print(f"  Max genes per batch: {self.max_genes_per_batch}")
            print(f"  Max memory per batch: {self.max_memory_gb} GB")
        
        # Apply global feature exclusions
        original_feature_count = X_df.shape[1]
        X_df = self.apply_global_feature_exclusions(X_df)
        feature_names = list(X_df.columns)
        
        if self.verbose and X_df.shape[1] < original_feature_count:
            print(f"  Features after global exclusions: {X_df.shape[1]} (removed {original_feature_count - X_df.shape[1]})")
        
        # Use direct batch trainer to avoid subprocess issues
        from meta_spliceai.splice_engine.meta_models.training.direct_batch_trainer import DirectBatchTrainer
        
        if self.verbose:
            print(f"  🔧 Using direct in-process batch training...")
        
        # Create direct batch trainer
        trainer = DirectBatchTrainer(
            dataset_path=dataset_path,
            max_genes_per_batch=self.max_genes_per_batch,
            verbose=self.verbose
        )
        
        # Run direct batch training
        results = trainer.train_all_genes_direct(out_dir, args)
        
        # Extract model path
        model_path = Path(results['model_path'])
        
        # Run ensemble-level cross-validation for generalization assessment
        ensemble_cv_results = self._run_ensemble_cross_validation(
            dataset_path, model_path, out_dir, args
        )
        
        # Calculate performance metrics
        performance_metrics = {
            'batch_count': results['successful_batch_count'],
            'total_genes_trained': results['total_genes_trained'],
            'ensemble_cv_f1_mean': ensemble_cv_results.get('mean_f1', 0),
            'ensemble_cv_f1_std': ensemble_cv_results.get('std_f1', 0)
        }
        
        training_metadata = {
            'strategy': self.get_strategy_name(),
            'total_genes': results['total_genes_trained'],
            'total_positions': results.get('total_positions', 0),
            'features_used': len(feature_names),
            'training_date': datetime.now().isoformat(),
            'batch_count': results['successful_batch_count'],
            'max_genes_per_batch': self.max_genes_per_batch,
            'max_memory_gb': self.max_memory_gb,
            'ensemble_combination_method': 'voting'
        }
        
        return TrainingResult(
            model_path=model_path,
            feature_names=feature_names,
            excluded_features=self._global_excluded_features,
            training_metadata=training_metadata,
            cv_results=ensemble_cv_results.get('fold_results', []),
            performance_metrics=performance_metrics
        )
    
    def _run_ensemble_cross_validation(
        self,
        dataset_path: str,
        ensemble_model_path: Path,
        out_dir: Path,
        args
    ) -> Dict[str, Any]:
        """Run cross-validation evaluation on the final ensemble model."""
        
        if self.verbose:
            print(f"\n🎯 [Ensemble CV] Evaluating ensemble generalization...")
        
        try:
            from meta_spliceai.splice_engine.meta_models.training import datasets
            from meta_spliceai.splice_engine.meta_models.builder import preprocessing
            from meta_spliceai.splice_engine.meta_models.training.classifier_utils import _encode_labels
            from sklearn.model_selection import GroupKFold
            from sklearn.metrics import accuracy_score, f1_score
            import pickle
            
            # Load dataset
            df = datasets.load_dataset(dataset_path)
            X_df, y_series = preprocessing.prepare_training_data(
                df, label_col="splice_type", return_type="pandas", verbose=0, encode_chrom=True
            )
            
            # Apply same global exclusions
            X_df = self.apply_global_feature_exclusions(X_df)
            
            X = X_df.values
            y = _encode_labels(y_series)
            genes = df['gene_id'].to_numpy()
            
            # Load ensemble model
            with open(ensemble_model_path, 'rb') as f:
                ensemble_model = pickle.load(f)
            
            # Gene-aware cross-validation
            gkf = GroupKFold(n_splits=args.n_folds)
            ensemble_cv_results = []
            
            for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=genes)):
                test_genes = np.unique(genes[test_idx])
                
                # Evaluate ensemble on test set
                if hasattr(ensemble_model, 'predict_proba'):
                    # Single model
                    proba = ensemble_model.predict_proba(X[test_idx])
                else:
                    # Ensemble model - need to handle differently
                    from meta_spliceai.splice_engine.meta_models.workflows.inference.ensemble_model_loader import EnsembleModelWrapper
                    if isinstance(ensemble_model, dict) and ensemble_model.get('type') == 'AllGenesBatchEnsemble':
                        wrapper = EnsembleModelWrapper(ensemble_model)
                        proba = wrapper.predict_proba(X[test_idx])
                    else:
                        raise ValueError(f"Unknown ensemble model type: {type(ensemble_model)}")
                
                pred = proba.argmax(axis=1)
                accuracy = accuracy_score(y[test_idx], pred)
                f1_macro = f1_score(y[test_idx], pred, average="macro")
                
                ensemble_cv_results.append({
                    'fold': fold_idx,
                    'test_positions': len(test_idx),
                    'test_genes': len(test_genes),
                    'accuracy': accuracy,
                    'f1_macro': f1_macro
                })
                
                if self.verbose:
                    print(f"    Fold {fold_idx+1}/{args.n_folds}: Acc={accuracy:.3f}, F1={f1_macro:.3f} ({len(test_genes)} genes)")
            
            # Save ensemble CV results
            ensemble_cv_df = pd.DataFrame(ensemble_cv_results)
            ensemble_cv_path = out_dir / "ensemble_cv_results.csv"
            ensemble_cv_df.to_csv(ensemble_cv_path, index=False)
            
            # Calculate summary statistics
            f1_scores = [r['f1_macro'] for r in ensemble_cv_results]
            
            summary_stats = {
                'mean_f1': np.mean(f1_scores),
                'std_f1': np.std(f1_scores),
                'fold_results': ensemble_cv_results
            }
            
            if self.verbose:
                print(f"  ✅ Ensemble CV completed: F1={summary_stats['mean_f1']:.3f}±{summary_stats['std_f1']:.3f}")
            
            return summary_stats
            
        except Exception as e:
            if self.verbose:
                print(f"  ❌ Ensemble CV failed: {e}")
            return {'fold_results': []}


def select_optimal_training_strategy(
    dataset_path: str,
    args,
    verbose: bool = True
) -> TrainingStrategy:
    """
    Automatically select the optimal training strategy based on dataset characteristics.
    """
    
    if verbose:
        print(f"\n🤖 [Strategy Selection] Analyzing dataset for optimal training approach...")
    
    # Handle --train-all-genes flag intelligently
    if hasattr(args, 'train_all_genes') and args.train_all_genes:
        # Check if we have gene sampling that makes single model viable
        if hasattr(args, 'sample_genes') and args.sample_genes and args.sample_genes <= 2000:
            if verbose:
                print(f"  🎯 --train-all-genes with --sample-genes {args.sample_genes}: Using single model with sampling")
            strategy = SingleModelTrainingStrategy(verbose=verbose)
            return strategy
        else:
            strategy = BatchEnsembleTrainingStrategy(verbose=verbose)
            if verbose:
                print(f"  🔥 Selected: {strategy.get_strategy_name()} (forced by --train-all-genes)")
            return strategy
    
    try:
        # Quick dataset analysis
        import polars as pl
        lf = pl.scan_parquet(f"{dataset_path}/*.parquet", extra_columns='ignore')
        
        # Get basic statistics
        stats = lf.select([
            pl.col("gene_id").n_unique().alias("total_genes"),
            pl.count().alias("total_positions")
        ]).collect()
        
        total_genes = stats["total_genes"].item()
        total_positions = stats["total_positions"].item()
        
        # Estimate memory requirements (rough)
        estimated_memory_gb = (total_positions * 150 * 8) / (1024**3)  # 150 features * 8 bytes
        
        if verbose:
            print(f"  Dataset characteristics:")
            print(f"    Genes: {total_genes:,}")
            print(f"    Positions: {total_positions:,}")
            print(f"    Estimated memory: {estimated_memory_gb:.1f} GB")
        
        # Evaluate both strategies to see which can handle the dataset
        single_model_strategy = SingleModelTrainingStrategy(verbose=verbose)
        batch_ensemble_strategy = BatchEnsembleTrainingStrategy(verbose=verbose)
        
        # Check if single model strategy can handle this dataset size
        can_handle_single = single_model_strategy.can_handle_dataset_size(total_genes, estimated_memory_gb)
        can_handle_batch = batch_ensemble_strategy.can_handle_dataset_size(total_genes, estimated_memory_gb)
        
        if verbose:
            print(f"  Strategy evaluation:")
            print(f"    Single Model can handle: {can_handle_single}")
            print(f"    Batch Ensemble can handle: {can_handle_batch}")
        
        # Prefer single model for smaller datasets (faster, simpler)
        if can_handle_single:
            strategy = single_model_strategy
            if verbose:
                print(f"  ✅ Selected: {strategy.get_strategy_name()}")
                if total_genes > 1000:
                    print(f"  📊 Medium dataset: Will use memory-safe gene sampling within single model")
        elif can_handle_batch:
            strategy = batch_ensemble_strategy
            if verbose:
                print(f"  ✅ Selected: {strategy.get_strategy_name()}")
                print(f"  🔥 Large dataset: Using batch ensemble for scalability")
        else:
            # Fallback: batch ensemble should always be able to handle any size
            strategy = batch_ensemble_strategy
            if verbose:
                print(f"  ⚠️  Dataset exceeds normal limits, forcing batch ensemble")
                print(f"  ✅ Selected: {strategy.get_strategy_name()}")
        
        return strategy
        
    except Exception as e:
        if verbose:
            print(f"  ⚠️  Strategy selection failed: {e}")
            print(f"  🔄 Defaulting to single model training")
        
        return SingleModelTrainingStrategy(verbose=verbose)
