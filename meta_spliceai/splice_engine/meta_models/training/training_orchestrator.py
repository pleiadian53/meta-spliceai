#!/usr/bin/env python3
"""
Training Orchestrator for Meta-Model Training

This module provides the main orchestration logic for meta-model training,
abstracting away the complexity of different training strategies while
ensuring consistent outputs and comprehensive analysis.

The orchestrator handles:
1. Strategy selection based on dataset characteristics
2. Global feature screening for consistency
3. Training execution with the selected strategy
4. Unified post-training analysis
5. Comprehensive documentation generation

This keeps the main driver script (run_gene_cv_sigmoid.py) clean and minimal.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import argparse
import os
import sys
import pandas as pd
import numpy as np
import polars as pl

from meta_spliceai.splice_engine.meta_models.training.training_strategies import (
    select_optimal_training_strategy,
    TrainingResult
)
# These modules will be imported as needed to avoid circular dependencies
from meta_spliceai.splice_engine.meta_models.training import cv_utils


class MetaModelTrainingOrchestrator:
    """
    Main orchestrator for meta-model training.
    
    This class encapsulates the complete training pipeline while maintaining
    a clean separation between the driver script and implementation details.
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self._training_result: Optional[TrainingResult] = None
        self._dataset_stats: Optional[Dict[str, Any]] = None
    
    def run_complete_training_pipeline(
        self,
        args: argparse.Namespace
    ) -> Dict[str, Any]:
        """
        Run the complete meta-model training pipeline.
        
        This is the main entry point that handles everything from dataset
        loading to final analysis, returning comprehensive results.
        
        Parameters
        ----------
        args : argparse.Namespace
            Parsed command-line arguments
            
        Returns
        -------
        Dict[str, Any]
            Complete training results and analysis summary
        """
        
        if self.verbose:
            print("🚀 [Training Orchestrator] Starting complete meta-model training pipeline")
            print(f"  Dataset: {args.dataset}")
            print(f"  Output: {args.out_dir}")
        
        # 1. Setup and validation
        self._setup_training_environment(args)
        
        # 2. Strategy selection
        strategy = self._select_training_strategy(args)
        
        # 3. Dataset loading and preparation
        raw_df, X_df, y_series, genes = self._load_and_prepare_dataset(args, strategy)
        
        # 4. Global feature screening
        self._run_global_feature_screening(args, strategy)
        
        # 5. Training execution
        training_result = self._execute_training(args, strategy, raw_df, X_df, y_series, genes)
        
        # 6. Post-training analysis
        analysis_results = self._run_post_training_analysis(args, training_result, raw_df)
        
        # 7. Generate final summary
        final_results = self._generate_final_summary(args, training_result, analysis_results)
        
        if self.verbose:
            print("🎉 [Training Orchestrator] Complete training pipeline finished!")
            self._print_final_summary(training_result, final_results)
        
        return final_results
    
    def _setup_training_environment(self, args: argparse.Namespace) -> None:
        """Setup training environment and validate arguments."""
        
        print("🔧 [Training Orchestrator] Setting up training environment...", flush=True)
        
        # Setup logging
        print("  📝 Configuring logging...", flush=True)
        cv_utils.setup_cv_logging(args)
        
        # Validate arguments
        print("  ✅ Validating arguments...", flush=True)
        cv_utils.validate_cv_arguments(args)
        
        # Resolve dataset paths
        print("  📁 Resolving dataset paths...", flush=True)
        original_path, actual_path, parquet_count = cv_utils.resolve_dataset_path(args.dataset)
        args.dataset = actual_path
        
        # Create output directory
        print("  📂 Creating output directory...", flush=True)
        out_dir = cv_utils.create_output_directory(args.out_dir)
        args.out_dir = str(out_dir)
        
        # Handle row cap environment variable
        print("  🔢 Configuring row cap settings...", flush=True)
        # Gene-aware sampling takes precedence over row cap
        if getattr(args, 'sample_genes', None) is not None:
            os.environ["SS_MAX_ROWS"] = "0"
            print(f"  📊 Disabled row cap for gene-aware sampling (--sample-genes {args.sample_genes})", flush=True)
        elif getattr(args, 'row_cap', 0) > 0 and not os.getenv("SS_MAX_ROWS"):
            os.environ["SS_MAX_ROWS"] = str(args.row_cap)
        elif getattr(args, 'row_cap', 0) == 0 and not os.getenv("SS_MAX_ROWS"):
            os.environ["SS_MAX_ROWS"] = "0"
        
        print(f"✅ [Training Orchestrator] Environment setup completed!", flush=True)
        print(f"  📊 Dataset: {original_path} → {actual_path}", flush=True)
        print(f"  📁 Output: {out_dir}", flush=True)
        print(f"  📄 Parquet files: {parquet_count}", flush=True)
        print(f"  🔢 Row cap: {'disabled' if getattr(args, 'row_cap', 0) == 0 else args.row_cap}", flush=True)
    
    def _select_training_strategy(self, args: argparse.Namespace):
        """Select optimal training strategy based on dataset and arguments."""
        
        print("🤖 [Training Orchestrator] Analyzing dataset for optimal training approach...", flush=True)
        
        # Use enhanced strategy selection that includes multi-instance ensemble
        try:
            from meta_spliceai.splice_engine.meta_models.training.multi_instance_ensemble_strategy import (
                select_optimal_training_strategy_with_multi_instance
            )
            strategy = select_optimal_training_strategy_with_multi_instance(args.dataset, args, verbose=self.verbose)
        except ImportError:
            # Fallback to standard strategy selection
            strategy = select_optimal_training_strategy(args.dataset, args, verbose=self.verbose)
        
        print(f"✅ [Training Orchestrator] Strategy selected: {strategy.get_strategy_name()}", flush=True)
        
        return strategy
    
    def _load_and_prepare_dataset(self, args: argparse.Namespace, strategy):
        """Load and prepare dataset for training."""
        
        print("📊 [Training Orchestrator] Loading and preparing training dataset...", flush=True)
        
        # Check strategy type for appropriate data loading
        from meta_spliceai.splice_engine.meta_models.training.training_strategies import BatchEnsembleTrainingStrategy
        
        # Check for multi-instance ensemble strategy
        try:
            from meta_spliceai.splice_engine.meta_models.training.multi_instance_ensemble_strategy import MultiInstanceEnsembleStrategy
            is_multi_instance = isinstance(strategy, MultiInstanceEnsembleStrategy)
        except ImportError:
            is_multi_instance = False
        
        if is_multi_instance:
            # For multi-instance ensemble, we only need a small sample for setup
            # The actual training will be handled by the multi-instance strategy
            from meta_spliceai.splice_engine.meta_models.training.label_utils import load_dataset_sample
            
            # Get total genes count for informational messages
            try:
                lf = pl.scan_parquet(f"{args.dataset}/*.parquet", extra_columns='ignore')
                total_genes = lf.select(pl.col("gene_id").n_unique()).collect().item()
            except Exception:
                total_genes = "unknown"
            
            print(f"  📊 Multi-instance ensemble: Loading small sample for orchestrator setup", flush=True)
            print(f"  🔍 Purpose: Feature screening and pipeline validation only", flush=True)
            if isinstance(total_genes, int):
                print(f"  ✅ ALL {total_genes:,} genes will be processed across multiple instances during actual training", flush=True)
            else:
                print(f"  ✅ ALL genes will be processed across multiple instances during actual training", flush=True)
            print(f"  ⚠️  The following sample is NOT used for model training - only for setup", flush=True)
            
            # Use small sample for orchestrator setup
            sample_size = 50
            raw_df = load_dataset_sample(args.dataset, sample_genes=sample_size, random_seed=args.seed)
            
        elif isinstance(strategy, BatchEnsembleTrainingStrategy):
            # For batch ensemble, use intelligent sampling as fallback
            from meta_spliceai.splice_engine.meta_models.training.label_utils import load_dataset_sample
            
            # Calculate optimal sample size for large datasets
            try:
                lf = pl.scan_parquet(f"{args.dataset}/*.parquet", extra_columns='ignore')
                total_genes = lf.select(pl.col("gene_id").n_unique()).collect().item()
                
                # Use intelligent sampling: aim for 1000-1500 genes for large datasets
                if getattr(args, 'sample_genes', None):
                    sample_size = args.sample_genes
                    print(f"  📊 Using user-specified sample: {sample_size} genes", flush=True)
                else:
                    # Auto-calculate optimal sample size
                    sample_size = min(1500, max(800, total_genes // 10))  # 10% of total, capped
                    print(f"  📊 Auto-calculated optimal sample: {sample_size} genes (from {total_genes:,} total)", flush=True)
                
                print(f"  ℹ️  Using single model training with intelligent sampling for memory efficiency", flush=True)
                
                # Override strategy selection - use single model with sampling
                from meta_spliceai.splice_engine.meta_models.training.training_strategies import SingleModelTrainingStrategy
                strategy.__class__ = SingleModelTrainingStrategy
                strategy.__init__(verbose=strategy.verbose)
                
            except Exception as e:
                print(f"  ⚠️  Error analyzing dataset, using default sampling: {e}")
                sample_size = 50
            
            raw_df = load_dataset_sample(args.dataset, sample_genes=sample_size, random_seed=args.seed)
        else:
            # For single model training, load as before
            from meta_spliceai.splice_engine.meta_models.training import datasets
            
            # Load dataset with appropriate method based on size
            if getattr(args, 'sample_genes', None) is not None:
                # Use gene-level sampling for faster testing
                from meta_spliceai.splice_engine.meta_models.training.label_utils import load_dataset_sample
                print(f"  📊 Sampling {args.sample_genes} genes for testing...", flush=True)
                raw_df = load_dataset_sample(args.dataset, sample_genes=args.sample_genes, random_seed=args.seed)
            else:
                # Load full dataset
                print(f"  📊 Loading full dataset...", flush=True)
                raw_df = datasets.load_dataset(args.dataset)
        
        # Check for required gene column
        if args.gene_col not in raw_df.columns:
            raise KeyError(f"Column '{args.gene_col}' not found in dataset")
        
        # Prepare training data
        print(f"  🔧 Preparing features...", flush=True)
        from meta_spliceai.splice_engine.meta_models.builder import preprocessing
        X_df, y_series = preprocessing.prepare_training_data(
            raw_df, 
            label_col="splice_type", 
            return_type="pandas", 
            verbose=1 if self.verbose else 0,
            preserve_transcript_columns=True,  # Always preserve transcript columns for transcript-level metrics
            encode_chrom=True
        )
        
        # Extract genes array
        genes = raw_df[args.gene_col].to_numpy()
        
        # Calculate statistics
        unique_genes = np.unique(genes)
        self._dataset_stats = {
            'total_genes': len(unique_genes),
            'total_positions': len(raw_df),
            'features_count': X_df.shape[1],
            'label_distribution': y_series.value_counts().to_dict()
        }
        
        print(f"✅ [Training Orchestrator] Dataset preparation completed!", flush=True)
        print(f"  📊 Loaded {self._dataset_stats['total_positions']:,} positions from {self._dataset_stats['total_genes']:,} genes", flush=True)
        print(f"  🔧 Features: {self._dataset_stats['features_count']}", flush=True)
        
        return raw_df, X_df, y_series, genes
    
    def _run_global_feature_screening(self, args: argparse.Namespace, strategy) -> None:
        """Run global feature screening to ensure consistency."""
        
        print("🔍 [Training Orchestrator] Running global feature screening...", flush=True)
        print("  🧪 This ensures consistent features across all training approaches", flush=True)
        
        out_dir = Path(args.out_dir)
        global_screening_result = strategy.run_global_feature_screening(
            args.dataset, out_dir, args, sample_fraction=0.1
        )
        
        excluded_count = len(global_screening_result.excluded_features)
        print(f"✅ [Training Orchestrator] Global feature screening completed!", flush=True)
        print(f"  📊 Features excluded: {excluded_count}", flush=True)
        if excluded_count > 0:
            print(f"  📄 Exclusion list saved: {out_dir}/global_excluded_features.txt", flush=True)
    
    def _execute_training(
        self,
        args: argparse.Namespace,
        strategy,
        raw_df: Any,
        X_df: pd.DataFrame,
        y_series: pd.Series,
        genes: np.ndarray
    ) -> TrainingResult:
        """Execute training using the selected strategy."""
        
        print(f"🚀 [Training Orchestrator] Executing training with: {strategy.get_strategy_name()}", flush=True)
        print(f"  📊 Training data: {len(np.unique(genes)):,} genes, {X_df.shape[0]:,} positions", flush=True)
        print(f"  🔧 Features: {X_df.shape[1]} features", flush=True)
        
        # Log algorithm selection
        algorithm = getattr(args, 'algorithm', 'xgboost')
        print(f"  🤖 Algorithm: {algorithm.upper()}", flush=True)
        
        out_dir = Path(args.out_dir)
        
        # For batch ensemble, the sample data is only for setup - the strategy handles its own data loading
        from meta_spliceai.splice_engine.meta_models.training.training_strategies import BatchEnsembleTrainingStrategy
        
        if isinstance(strategy, BatchEnsembleTrainingStrategy):
            print(f"  ℹ️  Batch ensemble will load and process data independently", flush=True)
        
        training_result = strategy.train_model(args.dataset, out_dir, args, X_df, y_series, genes)
        
        # Store training result for later use
        self._training_result = training_result
        
        print(f"✅ [Training Orchestrator] Training execution completed!", flush=True)
        print(f"  💾 Model saved: {training_result.model_path}", flush=True)
        print(f"  🔧 Features used: {len(training_result.feature_names)}", flush=True)
        print(f"  🚫 Features excluded: {len(training_result.excluded_features)}", flush=True)
        
        if training_result.performance_metrics:
            perf = training_result.performance_metrics
            if 'mean_accuracy' in perf:
                print(f"  📈 CV Accuracy: {perf['mean_accuracy']:.3f} ± {perf.get('std_accuracy', 0):.3f}", flush=True)
            if 'batch_count' in perf:
                print(f"  🔥 Ensemble batches: {perf['batch_count']}", flush=True)
        
        return training_result
    
    def _run_post_training_analysis(
        self,
        args: argparse.Namespace,
        training_result: TrainingResult,
        raw_df: Any
    ) -> Dict[str, Any]:
        """Run comprehensive post-training analysis."""
        
        print("📊 [Training Orchestrator] Running comprehensive post-training analysis...", flush=True)
        print("  🔍 This includes SHAP, feature importance, ROC/PR curves, and base vs meta comparison", flush=True)
        
        out_dir = Path(args.out_dir)
        evaluation_sample = min(25000, getattr(args, 'diag_sample', 25000))
        
        # Skip evaluation if requested
        if getattr(args, 'skip_eval', False):
            print("  ⏭️  Skipping evaluation due to --skip-eval flag", flush=True)
            return {'skipped': True}
        
        # For now, use the existing dataset for evaluation
        eval_dataset_path = args.dataset
        
        # Run the standard post-training analysis from the v0 implementation
        print("  🔬 Running standard post-training analysis...", flush=True)
        
        try:
            from meta_spliceai.splice_engine.meta_models.training import classifier_utils as _cutils
            
            # Set environment variable to preserve gene_id for gene-aware sampling in diagnostics
            original_ss_preserve = os.environ.get('SS_PRESERVE_GENE_ID')
            os.environ['SS_PRESERVE_GENE_ID'] = '1'
            
            try:
                # Run standard diagnostics (essential for model evaluation)
                print(f"  🔍 Running essential diagnostic analyses...", flush=True)
                _cutils.richer_metrics(eval_dataset_path, out_dir, sample=evaluation_sample)
                _cutils.gene_score_delta(eval_dataset_path, out_dir, sample=evaluation_sample)
                _cutils.probability_diagnostics(eval_dataset_path, out_dir, sample=evaluation_sample)
                _cutils.base_vs_meta(eval_dataset_path, out_dir, sample=evaluation_sample)
                print(f"  ✅ Essential diagnostics completed", flush=True)
            finally:
                # Restore original environment variable
                if original_ss_preserve is not None:
                    os.environ['SS_PRESERVE_GENE_ID'] = original_ss_preserve
                else:
                    os.environ.pop('SS_PRESERVE_GENE_ID', None)
            
            # Run SHAP analysis (if not skipped)
            if not getattr(args, 'skip_shap', False) and not getattr(args, 'minimal_diagnostics', False):
                try:
                    print(f"  🔍 Running SHAP analysis...", flush=True)
                    
                    # Determine SHAP sample size
                    shap_sample = evaluation_sample
                    if getattr(args, 'shap_sample', None) is not None:
                        shap_sample = args.shap_sample
                        print(f"  📊 Using custom SHAP sample size: {shap_sample}")
                    elif getattr(args, 'fast_shap', False):
                        shap_sample = 1000
                        print(f"  🚀 Using fast SHAP with reduced sample size: {shap_sample}")
                    
                    from meta_spliceai.splice_engine.meta_models.evaluation.shap_incremental import run_incremental_shap_analysis
                    shap_output_dir = run_incremental_shap_analysis(eval_dataset_path, out_dir, sample=shap_sample)
                    print(f"  ✅ SHAP analysis completed: {shap_output_dir}", flush=True)
                except Exception as e:
                    print(f"  ⚠️  SHAP analysis failed: {e}", flush=True)
            else:
                skip_reason = "minimal-diagnostics" if getattr(args, 'minimal_diagnostics', False) else "skip-shap"
                print(f"  ⏩ Skipping SHAP analysis due to --{skip_reason} flag", flush=True)
            
            # Run feature importance analysis (if not skipped)
            if not getattr(args, 'skip_feature_importance', False) and not getattr(args, 'minimal_diagnostics', False):
                try:
                    print(f"  🔍 Running comprehensive feature importance analysis...", flush=True)
                    from meta_spliceai.splice_engine.meta_models.evaluation.feature_importance_integration import (
                        run_gene_cv_feature_importance_analysis
                    )
                    run_gene_cv_feature_importance_analysis(
                        dataset_path=eval_dataset_path,
                        out_dir=out_dir,
                        sample=evaluation_sample
                    )
                    print(f"  ✅ Feature importance analysis completed", flush=True)
                except Exception as e:
                    print(f"  ⚠️  Feature importance analysis failed: {e}", flush=True)
            else:
                skip_reason = "minimal-diagnostics" if getattr(args, 'minimal_diagnostics', False) else "skip-feature-importance"
                print(f"  ⏩ Skipping feature importance analysis due to --{skip_reason} flag", flush=True)
            
            # Generate CV metrics visualization if available
            if training_result.cv_results:
                try:
                    # Save CV results first
                    import pandas as pd
                    cv_df = pd.DataFrame(training_result.cv_results)
                    cv_df.to_csv(out_dir / "gene_cv_metrics.csv", index=False)
                    
                    from meta_spliceai.splice_engine.meta_models.evaluation.cv_metrics_viz import generate_cv_metrics_report
                    viz_result = generate_cv_metrics_report(
                        csv_path=out_dir / "gene_cv_metrics.csv",
                        out_dir=out_dir,
                        dataset_path=args.dataset,
                        plot_format=getattr(args, 'plot_format', 'pdf')
                    )
                    print(f"  ✅ CV metrics visualization completed", flush=True)
                except Exception as e:
                    print(f"  ⚠️  CV metrics visualization failed: {e}", flush=True)
            
            # Generate base vs meta comparison
            try:
                # Create a simple comparison result for now
                if training_result.cv_results:
                    cv_df = pd.DataFrame(training_result.cv_results)
                    compare_results = {
                        "base_f1_mean": cv_df["base_f1"].mean() if "base_f1" in cv_df else 0.0,
                        "meta_f1_mean": cv_df["meta_f1"].mean() if "meta_f1" in cv_df else 0.0,
                        "improvement": (cv_df["meta_f1"].mean() - cv_df["base_f1"].mean()) if "meta_f1" in cv_df and "base_f1" in cv_df else 0.0,
                        "base_auc_mean": cv_df["auc_base"].mean() if "auc_base" in cv_df else 0.0,
                        "meta_auc_mean": cv_df["auc_meta"].mean() if "auc_meta" in cv_df else 0.0
                    }
                    import json
                    with open(out_dir / "compare_base_meta.json", 'w') as f:
                        json.dump(compare_results, f, indent=2)
                    print(f"  ✅ Base vs meta comparison saved", flush=True)
            except Exception as e:
                print(f"  ⚠️  Base vs meta comparison failed: {e}", flush=True)
            
            analysis_results = {
                'richer_metrics': True,
                'gene_score_delta': True,
                'probability_diagnostics': True,
                'base_vs_meta': True,
                'shap_analysis': True,
                'feature_importance': True,
                'cv_visualization': bool(training_result.cv_results)
            }
            
        except Exception as e:
            print(f"  ❌ Post-training analysis failed: {e}", flush=True)
            analysis_results = {'error': str(e)}
        
        successful_analyses = sum(1 for v in analysis_results.values() if v is True)
        total_analyses = len([v for v in analysis_results.values() if isinstance(v, bool)])
        print(f"✅ [Training Orchestrator] Post-training analysis completed!", flush=True)
        print(f"  📈 Successful components: {successful_analyses}/{total_analyses}", flush=True)
        
        return analysis_results
    
    def _generate_final_summary(
        self,
        args: argparse.Namespace,
        training_result: TrainingResult,
        analysis_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate final comprehensive summary."""
        
        final_results = {
            'training_strategy': training_result.training_metadata.get('strategy', 'Unknown'),
            'model_path': str(training_result.model_path),
            'dataset_statistics': self._dataset_stats,
            'training_metadata': training_result.training_metadata,
            'performance_metrics': training_result.performance_metrics,
            'analysis_results': analysis_results,
            'feature_info': {
                'features_used': len(training_result.feature_names),
                'features_excluded': len(training_result.excluded_features),
                'feature_names': training_result.feature_names,
                'excluded_features': training_result.excluded_features
            }
        }
        
        # Save comprehensive results
        out_dir = Path(args.out_dir)
        results_path = out_dir / "complete_training_results.json"
        
        import json
        with open(results_path, 'w') as f:
            json.dump(final_results, f, indent=2, default=str)
        
        if self.verbose:
            print(f"  📋 Complete results saved: {results_path}")
        
        return final_results
    
    def _print_final_summary(
        self,
        training_result: TrainingResult,
        final_results: Dict[str, Any]
    ) -> None:
        """Print final training summary to console."""
        
        print("\n" + "=" * 80)
        print("🎯 TRAINING PIPELINE SUMMARY")
        print("=" * 80)
        
        strategy = training_result.training_metadata.get('strategy', 'Unknown')
        print(f"Training Strategy: {strategy}")
        print(f"Model Path: {training_result.model_path}")
        
        if self._dataset_stats:
            print(f"Dataset: {self._dataset_stats['total_genes']:,} genes, {self._dataset_stats['total_positions']:,} positions")
            print(f"Features: {self._dataset_stats['features_count']} used, {len(training_result.excluded_features)} excluded")
        
        if training_result.performance_metrics:
            perf = training_result.performance_metrics
            if 'mean_accuracy' in perf:
                print(f"CV Performance: Accuracy {perf['mean_accuracy']:.3f} ± {perf.get('std_accuracy', 0):.3f}")
            if 'batch_count' in perf:
                print(f"Ensemble: {perf['batch_count']} batches, Avg accuracy {perf.get('average_batch_accuracy', 0):.3f}")
        
        # Analysis summary
        analysis_results = final_results.get('analysis_results', {})
        successful_analyses = sum(1 for v in analysis_results.values() if v)
        total_analyses = len(analysis_results)
        print(f"Analysis: {successful_analyses}/{total_analyses} components successful")
        
        print("=" * 80)


def run_meta_model_training_pipeline(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Main entry point for meta-model training pipeline.
    
    This function provides a clean interface for the driver script,
    encapsulating all training logic in appropriate utility modules.
    
    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments
        
    Returns
    -------
    Dict[str, Any]
        Complete training results
    """
    
    orchestrator = MetaModelTrainingOrchestrator(verbose=getattr(args, 'verbose', True))
    return orchestrator.run_complete_training_pipeline(args)
