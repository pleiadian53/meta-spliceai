#!/usr/bin/env python3
"""
Multi-Instance Ensemble Training Strategy

This module implements a training strategy that:
1. Trains multiple meta-models on random subsets of genes
2. Consolidates them into a single unified meta-model
3. Maintains interface compatibility with downstream workflows

Key Benefits:
- Memory efficient for large datasets
- Better generalization through diverse gene sampling
- Interface consistency (looks like single model to inference)
- Easy to extend to other algorithms
"""

import numpy as np
import pandas as pd
import polars as pl
import pickle
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass

from meta_spliceai.splice_engine.meta_models.training.training_strategies import (
    TrainingStrategy, TrainingResult, SingleModelTrainingStrategy
)
from meta_spliceai.splice_engine.meta_models.training.classifier_utils import (
    SigmoidEnsemble, PerClassCalibratedSigmoidEnsemble
)


@dataclass
class MultiInstanceResult:
    """Result from training a single instance in the multi-instance ensemble."""
    instance_id: int
    genes_used: List[str]
    model_path: Path
    feature_names: List[str]
    performance_metrics: Dict[str, float]
    cv_results: List[Dict]
    training_metadata: Dict[str, Any]


class ConsolidatedMetaModel:
    """
    Consolidated meta-model that combines multiple instance models.
    
    This class provides the same interface as a single model but internally
    uses multiple models for prediction, making it transparent to downstream code.
    """
    
    def __init__(
        self, 
        instance_models: List[SigmoidEnsemble], 
        feature_names: List[str],
        consolidation_method: str = "voting",
        instance_weights: Optional[List[float]] = None
    ):
        self.instance_models = instance_models
        self.feature_names = feature_names
        self.consolidation_method = consolidation_method
        self.instance_weights = instance_weights or [1.0] * len(instance_models)
        
        # Normalize weights
        total_weight = sum(self.instance_weights)
        self.instance_weights = [w / total_weight for w in self.instance_weights]
        
        # Store metadata for compatibility
        self.n_instances = len(instance_models)
        self.model_type = "ConsolidatedMetaModel"
        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities using consolidated model."""
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
            
        # Collect predictions from all instances
        all_predictions = []
        
        for model, weight in zip(self.instance_models, self.instance_weights):
            # Ensure feature compatibility
            if hasattr(model, 'feature_names') and len(model.feature_names) == X.shape[1]:
                pred = model.predict_proba(X) * weight
            else:
                # Handle potential feature mismatch (use available features)
                available_features = min(len(model.feature_names), X.shape[1])
                pred = model.predict_proba(X[:, :available_features]) * weight
            
            all_predictions.append(pred)
        
        # Consolidate predictions
        if self.consolidation_method == "voting":
            # Average weighted predictions
            consolidated_pred = np.mean(all_predictions, axis=0)
        elif self.consolidation_method == "max":
            # Take maximum confidence prediction
            consolidated_pred = np.max(all_predictions, axis=0)
        else:
            raise ValueError(f"Unknown consolidation method: {self.consolidation_method}")
        
        # Ensure probabilities sum to 1
        row_sums = consolidated_pred.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        consolidated_pred = consolidated_pred / row_sums
        
        return consolidated_pred
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels using consolidated model."""
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)
    
    def get_instance_info(self) -> Dict[str, Any]:
        """Get information about the constituent instances."""
        return {
            'n_instances': self.n_instances,
            'consolidation_method': self.consolidation_method,
            'instance_weights': self.instance_weights,
            'total_features': len(self.feature_names),
            'instance_models': [type(model).__name__ for model in self.instance_models]
        }
    
    def get_base_models(self):
        """Get underlying binary models for SHAP analysis compatibility."""
        # Return the first instance's binary models for SHAP analysis
        # This allows SHAP TreeExplainer to work with the ensemble
        if self.instance_models and hasattr(self.instance_models[0], 'get_base_models'):
            return self.instance_models[0].get_base_models()
        elif self.instance_models and hasattr(self.instance_models[0], 'models'):
            return self.instance_models[0].models
        return []
    
    @property
    def models(self):
        """Alternative property for SHAP compatibility."""
        return self.get_base_models()
    
    def get_comprehensive_shap_models(self):
        """Get all binary models from all instances for comprehensive SHAP analysis.
        
        Returns
        -------
        dict
            Dictionary with structure:
            {
                'instance_0': {
                    'neither_model': XGBClassifier,
                    'donor_model': XGBClassifier, 
                    'acceptor_model': XGBClassifier,
                    'weight': float
                },
                ...
            }
        """
        comprehensive_models = {}
        
        for i, (instance, weight) in enumerate(zip(self.instance_models, self.instance_weights)):
            instance_key = f'instance_{i}'
            
            if hasattr(instance, 'get_base_models'):
                base_models = instance.get_base_models()
                if len(base_models) >= 3:
                    comprehensive_models[instance_key] = {
                        'neither_model': base_models[0],  # Neither vs Rest
                        'donor_model': base_models[1],    # Donor vs Rest  
                        'acceptor_model': base_models[2], # Acceptor vs Rest
                        'weight': weight,
                        'instance_id': i
                    }
            elif hasattr(instance, 'models'):
                models = instance.models
                if len(models) >= 3:
                    comprehensive_models[instance_key] = {
                        'neither_model': models[0],
                        'donor_model': models[1],
                        'acceptor_model': models[2], 
                        'weight': weight,
                        'instance_id': i
                    }
        
        return comprehensive_models
    
    def compute_ensemble_shap_importance(self, X, background_data=None, max_evals=1000):
        """Compute SHAP importance averaged across all instances and binary models.
        
        Parameters
        ----------
        X : np.ndarray
            Data to compute SHAP values for
        background_data : np.ndarray, optional
            Background data for SHAP explainer. If None, uses subset of X
        max_evals : int
            Maximum evaluations for SHAP explainer
            
        Returns
        -------
        dict
            Dictionary with SHAP importance for each class and overall ensemble:
            {
                'neither_importance': pd.Series,
                'donor_importance': pd.Series,
                'acceptor_importance': pd.Series,
                'ensemble_importance': pd.Series,  # Weighted average
                'instance_contributions': dict     # Per-instance breakdown
            }
        """
        try:
            import shap
        except ImportError:
            print("SHAP not available for ensemble importance computation")
            return None
        
        # Prepare background data
        if background_data is None:
            # Use a sample of X as background
            n_background = min(100, len(X))
            background_data = X[:n_background]
        
        comprehensive_models = self.get_comprehensive_shap_models()
        
        if not comprehensive_models:
            print("No models available for comprehensive SHAP analysis")
            return None
        
        # Initialize importance accumulators
        class_importances = {
            'neither': np.zeros(X.shape[1]),
            'donor': np.zeros(X.shape[1]),
            'acceptor': np.zeros(X.shape[1])
        }
        
        instance_contributions = {}
        
        print(f"Computing SHAP values for {len(comprehensive_models)} instances...")
        
        for instance_key, instance_data in comprehensive_models.items():
            instance_contributions[instance_key] = {}
            weight = instance_data['weight']
            
            print(f"  Processing {instance_key} (weight: {weight:.3f})")
            
            # Compute SHAP for each binary model in this instance
            for class_name in ['neither', 'donor', 'acceptor']:
                model_key = f'{class_name}_model'
                model = instance_data[model_key]
                
                try:
                    # Create SHAP explainer for this binary model
                    explainer = shap.TreeExplainer(
                        model,
                        data=background_data,
                        feature_perturbation="interventional",
                        model_output="probability"
                    )
                    
                    # Compute SHAP values
                    shap_values = explainer.shap_values(X)
                    
                    # Handle different SHAP output formats
                    if isinstance(shap_values, list):
                        # Binary classification returns [class_0_shap, class_1_shap]
                        # We want class_1 (positive class) SHAP values
                        shap_vals = shap_values[1] if len(shap_values) > 1 else shap_values[0]
                    else:
                        shap_vals = shap_values
                    
                    # Compute mean absolute SHAP importance
                    importance = np.mean(np.abs(shap_vals), axis=0)
                    
                    # Weight by instance performance and accumulate
                    weighted_importance = importance * weight
                    class_importances[class_name] += weighted_importance
                    
                    # Store instance contribution
                    instance_contributions[instance_key][class_name] = importance
                    
                    print(f"    ✓ {class_name} model SHAP computed")
                    
                except Exception as e:
                    print(f"    ✗ Failed to compute SHAP for {class_name} model: {e}")
                    # Fill with zeros for this model
                    instance_contributions[instance_key][class_name] = np.zeros(X.shape[1])
        
        # Create ensemble importance (weighted average across all models)
        total_models = len(comprehensive_models) * 3  # 3 binary models per instance
        ensemble_importance = (
            class_importances['neither'] + 
            class_importances['donor'] + 
            class_importances['acceptor']
        ) / total_models
        
        # Handle potential feature dimension mismatch
        actual_n_features = len(class_importances['neither'])
        feature_names_to_use = self.feature_names[:actual_n_features]
        
        if len(feature_names_to_use) != actual_n_features:
            print(f"  ⚠️  Feature dimension mismatch: {len(self.feature_names)} expected, {actual_n_features} actual")
            print(f"  📝 Using first {actual_n_features} feature names")
        
        # Convert to pandas Series with feature names
        results = {
            'neither_importance': pd.Series(
                class_importances['neither'] / len(comprehensive_models), 
                index=feature_names_to_use
            ),
            'donor_importance': pd.Series(
                class_importances['donor'] / len(comprehensive_models), 
                index=feature_names_to_use
            ),
            'acceptor_importance': pd.Series(
                class_importances['acceptor'] / len(comprehensive_models), 
                index=feature_names_to_use
            ),
            'ensemble_importance': pd.Series(
                ensemble_importance, 
                index=feature_names_to_use
            ),
            'instance_contributions': instance_contributions
        }
        
        print(f"✓ Comprehensive SHAP analysis completed")
        print(f"  Top ensemble features: {results['ensemble_importance'].nlargest(5).index.tolist()}")
        
        return results


class MultiInstanceEnsembleStrategy(TrainingStrategy):
    """
    Multi-instance ensemble training strategy for large datasets.
    
    This strategy trains multiple meta-models on random subsets of genes
    and consolidates them into a single unified model that maintains
    interface compatibility with downstream workflows.
    """
    
    def __init__(
        self, 
        n_instances: int = 5,
        genes_per_instance: int = 1500,
        overlap_ratio: float = 0.1,
        consolidation_method: str = "voting",
        verbose: bool = True
    ):
        super().__init__(verbose)
        self.n_instances = n_instances
        self.genes_per_instance = genes_per_instance
        self.overlap_ratio = overlap_ratio
        self.consolidation_method = consolidation_method
        
    def get_strategy_name(self) -> str:
        return f"Multi-Instance Ensemble ({self.n_instances} instances, {self.genes_per_instance} genes each)"
    
    def can_handle_dataset_size(self, total_genes: int, estimated_memory_gb: float) -> bool:
        """Multi-instance ensemble can handle any dataset size."""
        return True  # Always capable through gene sampling
    
    def train_model(
        self,
        dataset_path: str,
        out_dir: Path,
        args,
        X_df: pd.DataFrame,
        y_series: pd.Series,
        genes: np.ndarray
    ) -> TrainingResult:
        """Train multi-instance ensemble model."""
        
        print(f"🔥 [Multi-Instance Ensemble] Training {self.n_instances} instances...", flush=True)
        print(f"  📊 Setup data: {len(np.unique(genes)):,} genes (orchestrator sample only)", flush=True)
        print(f"  🎯 Actual training: {self.genes_per_instance:,} genes per instance from full dataset", flush=True)
        print(f"  🔀 Gene overlap ratio: {self.overlap_ratio:.1%}", flush=True)
        print(f"  ⚠️  Each instance will load fresh data from the complete {dataset_path} dataset", flush=True)
        
        # Create instance-specific directories
        instances_dir = out_dir / "multi_instance_training"
        instances_dir.mkdir(exist_ok=True, parents=True)
        
        # Generate gene subsets for each instance
        gene_subsets = self._generate_gene_subsets(dataset_path, args)
        
        # Train each instance with checkpointing support
        instance_results = []
        successful_instances = []
        
        # Check for existing instances (checkpointing)
        existing_instances = []
        if getattr(args, 'resume_from_checkpoint', True) and not getattr(args, 'force_retrain_all', False):
            existing_instances = self._check_existing_instances(instances_dir)
        elif getattr(args, 'force_retrain_all', False):
            print(f"  🔄 Force retrain enabled - ignoring existing instances", flush=True)
        
        for i, gene_subset in enumerate(gene_subsets):
            instance_id = i
            instance_dir = instances_dir / f"instance_{instance_id:02d}"
            
            # Check if this instance is already complete
            if instance_id in existing_instances:
                print(f"\n♻️  [Instance {i+1}/{self.n_instances}] Found existing completed instance - reusing", flush=True)
                try:
                    # Load existing instance result
                    instance_result = self._load_existing_instance(instance_dir, instance_id, gene_subset)
                    instance_results.append(instance_result)
                    successful_instances.append(i)
                    # Use the correct metric name from the saved metrics
                    f1_score = instance_result.performance_metrics.get('test_macro_f1', 
                               instance_result.performance_metrics.get('mean_f1_macro', 0))
                    print(f"  ✅ Instance {i+1} loaded: F1={f1_score:.3f}")
                    continue
                except Exception as e:
                    print(f"  ⚠️  Failed to load existing instance {i+1}: {e}")
                    print(f"  🔄 Will retrain this instance...")
            
            print(f"\n🔧 [Instance {i+1}/{self.n_instances}] Training on {len(gene_subset)} genes...", flush=True)
            print(f"    📊 Loading fresh data from full dataset for this gene subset", flush=True)
            
            try:
                instance_result = self._train_single_instance(
                    instance_id=i,
                    gene_subset=gene_subset,
                    dataset_path=dataset_path,
                    instances_dir=instances_dir,
                    args=args
                )
                
                instance_results.append(instance_result)
                successful_instances.append(i)
                
                print(f"  ✅ Instance {i+1} completed: F1={instance_result.performance_metrics.get('f1_macro', 0):.3f}")
                
            except Exception as e:
                print(f"  ❌ Instance {i+1} failed: {e}")
                if self.verbose:
                    import traceback
                    traceback.print_exc()
        
        if not instance_results:
            raise RuntimeError("No instances trained successfully")
        
        print(f"\n🎯 [Multi-Instance Ensemble] {len(successful_instances)}/{self.n_instances} instances successful", flush=True)
        
        # Consolidate instances into single model
        consolidated_model = self._consolidate_instances(instance_results, out_dir)
        
        # Save consolidated model
        model_path = out_dir / "model_multiclass.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(consolidated_model, f)
        
        print(f"  💾 Consolidated model saved: {model_path}", flush=True)
        
        # Save feature manifest
        feature_manifest = pd.DataFrame({'feature': consolidated_model.feature_names})
        feature_manifest.to_csv(out_dir / "feature_manifest.csv", index=False)
        
        # Save features as JSON for compatibility
        features_json = {"feature_names": consolidated_model.feature_names}
        with open(out_dir / "train.features.json", 'w') as f:
            json.dump(features_json, f)
        
        # Aggregate CV results from all instances
        all_cv_results = []
        for instance_result in instance_results:
            for cv_result in instance_result.cv_results:
                cv_result['instance_id'] = instance_result.instance_id
                all_cv_results.append(cv_result)
        
        # Calculate aggregate performance metrics
        performance_metrics = self._calculate_aggregate_performance(instance_results)
        
        # Create training metadata
        training_metadata = {
            'strategy': self.get_strategy_name(),
            'n_instances': len(successful_instances),
            'total_genes_trained': sum(len(r.genes_used) for r in instance_results),
            'unique_genes_trained': len(set().union(*[r.genes_used for r in instance_results])),
            'genes_per_instance': self.genes_per_instance,
            'overlap_ratio': self.overlap_ratio,
            'consolidation_method': self.consolidation_method,
            'training_date': datetime.now().isoformat(),
            'successful_instances': successful_instances,
            'features_used': len(consolidated_model.feature_names)
        }
        
        return TrainingResult(
            model_path=model_path,
            feature_names=consolidated_model.feature_names,
            excluded_features=self._global_excluded_features,
            training_metadata=training_metadata,
            cv_results=all_cv_results,
            performance_metrics=performance_metrics
        )
    
    def _generate_gene_subsets(self, dataset_path: str, args) -> List[List[str]]:
        """Generate gene subsets ensuring ALL genes are covered across instances."""
        
        # Get all available genes from the dataset
        lf = pl.scan_parquet(f"{dataset_path}/*.parquet", extra_columns='ignore')
        all_genes = lf.select("gene_id").unique().collect()["gene_id"].to_list()
        
        print(f"  🧬 Discovered {len(all_genes):,} total genes in dataset", flush=True)
        
        # Calculate number of instances needed to cover all genes
        genes_per_instance_effective = int(self.genes_per_instance * (1 - self.overlap_ratio))
        min_instances_needed = max(self.n_instances, (len(all_genes) + genes_per_instance_effective - 1) // genes_per_instance_effective)
        
        if min_instances_needed > self.n_instances:
            print(f"  🔄 Adjusting instances: {self.n_instances} → {min_instances_needed} to cover all {len(all_genes):,} genes")
            self.n_instances = min_instances_needed
        
        # Shuffle genes for random distribution across instances
        np.random.seed(getattr(args, 'seed', 42))
        shuffled_genes = all_genes.copy()
        np.random.shuffle(shuffled_genes)
        
        gene_subsets = []
        genes_covered = set()
        
        # Calculate overlap size
        overlap_size = int(self.genes_per_instance * self.overlap_ratio)
        step_size = self.genes_per_instance - overlap_size
        
        # Distribute genes ensuring complete coverage
        for i in range(self.n_instances):
            start_idx = i * step_size
            end_idx = start_idx + self.genes_per_instance
            
            # Handle end-of-list wrapping to ensure all genes are covered
            if start_idx >= len(shuffled_genes):
                # We've covered all genes, but add some overlap for robustness
                subset = shuffled_genes[-self.genes_per_instance:]
            elif end_idx > len(shuffled_genes):
                # Take remaining genes and add overlap from beginning
                subset = shuffled_genes[start_idx:]
                remaining_needed = self.genes_per_instance - len(subset)
                if remaining_needed > 0:
                    subset.extend(shuffled_genes[:remaining_needed])
            else:
                subset = shuffled_genes[start_idx:end_idx]
            
            gene_subsets.append(subset)
            genes_covered.update(subset)
            
            if self.verbose:
                actual_overlap = 0
                if i > 0:
                    actual_overlap = len(set(subset) & set(gene_subsets[i-1]))
                print(f"    Instance {i+1}: {len(subset)} genes (overlap: {actual_overlap}, unique: {len(set(subset) - genes_covered) + len(subset)})")
        
        # Verify complete coverage
        uncovered_genes = set(all_genes) - genes_covered
        if uncovered_genes:
            print(f"  ⚠️  {len(uncovered_genes)} genes not covered, adding to last instance")
            gene_subsets[-1].extend(list(uncovered_genes))
            genes_covered.update(uncovered_genes)
        
        coverage_ratio = len(genes_covered) / len(all_genes)
        print(f"  ✅ Gene coverage: {len(genes_covered):,}/{len(all_genes):,} ({coverage_ratio:.1%})")
        
        if coverage_ratio < 1.0:
            raise RuntimeError(f"Incomplete gene coverage: {coverage_ratio:.1%}. All genes must be included in training.")
        
        return gene_subsets
    
    def _train_single_instance(
        self,
        instance_id: int,
        gene_subset: List[str],
        dataset_path: str,
        instances_dir: Path,
        args
    ) -> MultiInstanceResult:
        """Train a single instance on a gene subset."""
        
        instance_dir = instances_dir / f"instance_{instance_id:02d}"
        instance_dir.mkdir(exist_ok=True, parents=True)
        
        # Load data for this gene subset
        from meta_spliceai.splice_engine.meta_models.training.label_utils import load_genes_subset
        
        try:
            # Load specific genes (return polars format for preprocessing compatibility)
            instance_df = load_genes_subset(dataset_path, gene_subset, return_polars=True)
            
            print(f"    📊 Loaded {len(instance_df):,} positions from {len(gene_subset)} genes")
            
        except Exception as e:
            # Fallback: use gene sampling
            from meta_spliceai.splice_engine.meta_models.training.label_utils import load_dataset_sample
            instance_df = load_dataset_sample(dataset_path, sample_genes=len(gene_subset), random_seed=args.seed + instance_id)
            print(f"    📊 Fallback sampling: {len(instance_df):,} positions")
        
        # Prepare data for this instance
        from meta_spliceai.splice_engine.meta_models.builder import preprocessing
        
        X_instance_df, y_instance_series = preprocessing.prepare_training_data(
            instance_df,
            label_col="splice_type",
            return_type="pandas",
            verbose=0,
            preserve_transcript_columns=True,
            encode_chrom=True
        )
        
        # Apply global feature exclusions
        X_instance_df = self.apply_global_feature_exclusions(X_instance_df)
        
        # Remove metadata columns
        metadata_cols_to_remove = ['transcript_id', 'position']
        for col in metadata_cols_to_remove:
            if col in X_instance_df.columns:
                X_instance_df = X_instance_df.drop(columns=[col])
        
        # Extract gene array (handle polars DataFrame)
        if hasattr(instance_df, 'to_pandas'):
            genes_instance = instance_df[args.gene_col].to_pandas().values
        else:
            genes_instance = instance_df[args.gene_col].values
        
        # Create a single model strategy for this instance
        instance_strategy = SingleModelTrainingStrategy(verbose=False)
        instance_strategy._global_excluded_features = self._global_excluded_features
        
        # Create instance-specific args
        instance_args = type(args)(**vars(args))
        instance_args.n_folds = min(3, args.n_folds)  # Reduce folds for instances
        instance_args.verbose = False
        
        # Train the instance
        training_result = instance_strategy.train_model(
            dataset_path=dataset_path,
            out_dir=instance_dir,
            args=instance_args,
            X_df=X_instance_df,
            y_series=y_instance_series,
            genes=genes_instance
        )
        
        # Load the trained model
        with open(training_result.model_path, 'rb') as f:
            instance_model = pickle.load(f)
        
        return MultiInstanceResult(
            instance_id=instance_id,
            genes_used=gene_subset,
            model_path=training_result.model_path,
            feature_names=training_result.feature_names,
            performance_metrics=training_result.performance_metrics or {},
            cv_results=training_result.cv_results or [],
            training_metadata=training_result.training_metadata
        )
    
    def _consolidate_instances(
        self, 
        instance_results: List[MultiInstanceResult], 
        out_dir: Path
    ) -> ConsolidatedMetaModel:
        """Consolidate multiple instance models into a single unified model."""
        
        print(f"\n🔄 [Multi-Instance Ensemble] Consolidating {len(instance_results)} instances...", flush=True)
        
        # Load all instance models
        instance_models = []
        instance_weights = []
        
        # Use the feature names from the first successful instance
        feature_names = instance_results[0].feature_names
        
        for result in instance_results:
            try:
                with open(result.model_path, 'rb') as f:
                    model = pickle.load(f)
                    instance_models.append(model)
                    
                    # Weight by performance (F1 score)
                    f1_score = result.performance_metrics.get('mean_f1_macro', 1.0)
                    instance_weights.append(max(0.1, f1_score))  # Minimum weight of 0.1
                    
                    print(f"    Instance {result.instance_id}: F1={f1_score:.3f}, weight={instance_weights[-1]:.3f}")
                    
            except Exception as e:
                print(f"    ❌ Failed to load instance {result.instance_id}: {e}")
        
        # Create consolidated model
        consolidated = ConsolidatedMetaModel(
            instance_models=instance_models,
            feature_names=feature_names,
            consolidation_method=self.consolidation_method,
            instance_weights=instance_weights
        )
        
        # Save consolidation metadata
        consolidation_info = {
            'n_instances': len(instance_models),
            'consolidation_method': self.consolidation_method,
            'instance_weights': instance_weights,
            'feature_names': feature_names,
            'genes_per_instance': self.genes_per_instance,
            'total_unique_genes': len(set().union(*[r.genes_used for r in instance_results])),
            'consolidation_date': datetime.now().isoformat()
        }
        
        with open(out_dir / "consolidation_info.json", 'w') as f:
            json.dump(consolidation_info, f, indent=2)
        
        print(f"  ✅ Consolidated model created with {len(instance_models)} instances", flush=True)
        print(f"  🎯 Total unique genes covered: {consolidation_info['total_unique_genes']:,}", flush=True)
        
        # Copy key visualization files from the best-performing instance to main directory
        self._consolidate_visualization_files(instance_results, out_dir)
        
        return consolidated
    
    def _calculate_aggregate_performance(self, instance_results: List[MultiInstanceResult]) -> Dict[str, float]:
        """Calculate aggregate performance metrics across all instances."""
        
        if not instance_results:
            return {}
        
        # Collect metrics from all instances
        f1_scores = []
        ap_scores = []
        
        for result in instance_results:
            perf = result.performance_metrics
            if 'mean_f1_macro' in perf:
                f1_scores.append(perf['mean_f1_macro'])
            if 'mean_avg_precision_macro' in perf:
                ap_scores.append(perf['mean_avg_precision_macro'])
        
        aggregate_metrics = {
            'mean_f1_macro': np.mean(f1_scores) if f1_scores else 0.0,
            'std_f1_macro': np.std(f1_scores) if f1_scores else 0.0,
            'mean_avg_precision_macro': np.mean(ap_scores) if ap_scores else 0.0,
            'std_avg_precision_macro': np.std(ap_scores) if ap_scores else 0.0,
            'n_instances': len(instance_results),
            'total_genes_covered': len(set().union(*[r.genes_used for r in instance_results]))
        }
        
        return aggregate_metrics
    
    def _consolidate_visualization_files(
        self, 
        instance_results: List[MultiInstanceResult], 
        out_dir: Path
    ) -> None:
        """
        Copy key visualization files from the best-performing instance to main directory.
        
        This ensures that multi-instance training produces the same output structure
        as single-instance training, maintaining compatibility with downstream tools.
        """
        
        if not instance_results:
            print("  ⚠️  No instances available for visualization consolidation")
            return
        
        # Find the best-performing instance (highest F1 score)
        best_instance = max(
            instance_results, 
            key=lambda r: r.performance_metrics.get('mean_f1_macro', 0.0)
        )
        
        # Get the instance directory
        instance_dir = Path(best_instance.model_path).parent
        
        print(f"  📊 Consolidating visualization files from best instance {best_instance.instance_id} (F1={best_instance.performance_metrics.get('mean_f1_macro', 0.0):.3f})")
        
        # Define key files to copy from instance to main directory
        files_to_copy = [
            # ROC/PR curve visualizations
            "pr_base_vs_meta.pdf",
            "roc_base_vs_meta.pdf", 
            "pr_curves_meta.pdf",
            "roc_curves_meta.pdf",
            "pr_binary_improved.pdf",
            
            # Multiclass visualizations
            "multiclass_summary.pdf",
            "pr_acceptor_class.pdf",
            "pr_donor_class.pdf", 
            "pr_neither_class.pdf",
            "roc_acceptor_class.pdf",
            "roc_donor_class.pdf",
            "roc_neither_class.pdf",
            
            # Fold-level metrics (for compatibility)
            "metrics_fold0.json",
            "metrics_fold1.json", 
            "metrics_fold2.json",
            "metrics_fold3.json",
            "metrics_fold4.json",
            
            # Holdout evaluation results
            "holdout_evaluation_results.csv",
            "holdout_evaluation_summary.json",
            
            # Aggregate metrics
            "metrics_aggregate.json"
        ]
        
        copied_files = []
        missing_files = []
        
        for filename in files_to_copy:
            source_file = instance_dir / filename
            target_file = out_dir / filename
            
            if source_file.exists():
                try:
                    import shutil
                    shutil.copy2(source_file, target_file)
                    copied_files.append(filename)
                except Exception as e:
                    print(f"    ⚠️  Failed to copy {filename}: {e}")
                    missing_files.append(filename)
            else:
                missing_files.append(filename)
        
        # Report consolidation results
        if copied_files:
            print(f"    ✅ Copied {len(copied_files)} visualization files to main directory")
            if len(copied_files) <= 10:  # Show details for small lists
                for filename in copied_files[:5]:  # Show first 5
                    print(f"      - {filename}")
                if len(copied_files) > 5:
                    print(f"      - ... and {len(copied_files) - 5} more files")
        
        if missing_files:
            print(f"    ⚠️  {len(missing_files)} files not found in best instance:")
            for filename in missing_files[:3]:  # Show first 3 missing
                print(f"      - {filename}")
            if len(missing_files) > 3:
                print(f"      - ... and {len(missing_files) - 3} more missing files")
    
    def _check_existing_instances(self, instances_dir: Path) -> List[int]:
        """Check which instances are already completed and can be reused."""
        existing_instances = []
        
        if not instances_dir.exists():
            return existing_instances
        
        for i in range(self.n_instances):
            instance_dir = instances_dir / f"instance_{i:02d}"
            
            # Check if instance is complete by verifying key files exist
            required_files = [
                "model_multiclass.pkl",
                "metrics_aggregate.json", 
                "gene_cv_metrics.csv",
                "holdout_evaluation_summary.json"
            ]
            
            if instance_dir.exists():
                missing_files = []
                for file_name in required_files:
                    file_path = instance_dir / file_name
                    if not file_path.exists():
                        missing_files.append(file_name)
                
                if not missing_files:
                    existing_instances.append(i)
                    if self.verbose:
                        print(f"  ♻️  Found complete instance {i}: {instance_dir}")
                elif len(missing_files) < len(required_files):
                    if self.verbose:
                        print(f"  ⚠️  Incomplete instance {i}: missing {missing_files}")
        
        if existing_instances and self.verbose:
            print(f"  🎯 Checkpointing: Found {len(existing_instances)} existing instances to reuse")
            print(f"  📊 Will skip: {existing_instances}")
        
        return existing_instances
    
    def _load_existing_instance(self, instance_dir: Path, instance_id: int, gene_subset: List[str]) -> MultiInstanceResult:
        """Load an existing completed instance."""
        
        # Load performance metrics
        metrics_file = instance_dir / "metrics_aggregate.json"
        with open(metrics_file, 'r') as f:
            performance_metrics = json.load(f)
        
        # Load CV results if available
        cv_results = []
        for fold_idx in range(5):  # Assume max 5 folds
            fold_file = instance_dir / f"metrics_fold{fold_idx}.json"
            if fold_file.exists():
                with open(fold_file, 'r') as f:
                    cv_results.append(json.load(f))
        
        # Load feature names
        feature_file = instance_dir / "feature_manifest.csv"
        if feature_file.exists():
            feature_df = pd.read_csv(feature_file)
            feature_names = feature_df['feature'].tolist()
        else:
            # Fallback to train.features.json
            features_json_file = instance_dir / "train.features.json"
            with open(features_json_file, 'r') as f:
                features_data = json.load(f)
                feature_names = features_data['feature_names']
        
        # Create training metadata
        training_metadata = {
            'instance_id': instance_id,
            'genes_used': len(gene_subset),
            'reused_from_checkpoint': True,
            'checkpoint_timestamp': datetime.now().isoformat()
        }
        
        return MultiInstanceResult(
            instance_id=instance_id,
            genes_used=gene_subset,
            model_path=instance_dir / "model_multiclass.pkl",
            feature_names=feature_names,
            performance_metrics=performance_metrics,
            cv_results=cv_results,
            training_metadata=training_metadata
        )


def load_genes_subset(dataset_path: str, gene_list: List[str]) -> pd.DataFrame:
    """
    Load data for a specific subset of genes.
    
    This function efficiently loads only the data for specified genes,
    avoiding memory issues with large datasets.
    """
    
    # Load only the specified genes
    lf = pl.scan_parquet(f"{dataset_path}/*.parquet", extra_columns='ignore')
    df = lf.filter(pl.col("gene_id").is_in(gene_list)).collect()
    
    # Convert to pandas for compatibility
    return df.to_pandas()


# Update the strategy selection to include multi-instance ensemble
def select_optimal_training_strategy_with_multi_instance(
    dataset_path: str,
    args,
    verbose: bool = True
) -> TrainingStrategy:
    """
    Enhanced strategy selection that includes multi-instance ensemble option.
    """
    from meta_spliceai.splice_engine.meta_models.training.training_strategies import (
        select_optimal_training_strategy, SingleModelTrainingStrategy, BatchEnsembleTrainingStrategy
    )
    
    # First try the standard selection
    try:
        lf = pl.scan_parquet(f"{dataset_path}/*.parquet", extra_columns='ignore')
        
        stats = lf.select([
            pl.col("gene_id").n_unique().alias("total_genes"),
            pl.count().alias("total_positions")
        ]).collect()
        
        total_genes = stats["total_genes"].item()
        total_positions = stats["total_positions"].item()
        estimated_memory_gb = (total_positions * 150 * 8) / (1024**3)
        
        if verbose:
            print(f"🤖 [Enhanced Strategy Selection] Analyzing dataset...")
            print(f"  📊 Total genes: {total_genes:,}")
            print(f"  📊 Total positions: {total_positions:,}")
            print(f"  📊 Estimated memory: {estimated_memory_gb:.1f} GB")
        
        # Check if standard strategies can handle the dataset
        single_strategy = SingleModelTrainingStrategy(verbose=False)
        can_handle_single = single_strategy.can_handle_dataset_size(total_genes, estimated_memory_gb)
        
        # Decision logic for complete gene coverage
        if hasattr(args, 'train_all_genes') and args.train_all_genes:
            # Always use multi-instance ensemble when --train-all-genes is specified
            # This ensures ALL genes are included in training
            
            # Get configurable parameters from args
            base_genes_per_instance = getattr(args, 'genes_per_instance', 1500)
            max_instances = getattr(args, 'max_instances', 10)
            instance_overlap = getattr(args, 'instance_overlap', 0.1)
            memory_per_gene_mb = getattr(args, 'memory_per_gene_mb', 8.0)
            max_memory_per_instance_gb = getattr(args, 'max_memory_per_instance_gb', 15.0)
            auto_adjust = getattr(args, 'auto_adjust_instance_size', True)
            
            # Hardware-adaptive configuration
            optimal_genes_per_instance = base_genes_per_instance
            
            if auto_adjust:
                # Adjust based on available memory
                import psutil
                available_memory_gb = psutil.virtual_memory().available / (1024**3)
                
                # Calculate max genes based on memory constraints
                max_genes_by_memory = int((max_memory_per_instance_gb * 1024) / memory_per_gene_mb)
                optimal_genes_per_instance = min(base_genes_per_instance, max_genes_by_memory)
                
                # Ensure reasonable bounds
                optimal_genes_per_instance = max(500, min(optimal_genes_per_instance, 5000))
            
            if total_genes > optimal_genes_per_instance:
                # Calculate instances needed for complete coverage
                genes_per_instance_effective = int(optimal_genes_per_instance * (1 - instance_overlap))
                optimal_instances = max(3, (total_genes + genes_per_instance_effective - 1) // genes_per_instance_effective)
                optimal_instances = min(optimal_instances, max_instances)
                
                if verbose:
                    print(f"  🎯 --train-all-genes: Using Multi-Instance Ensemble for complete gene coverage")
                    if auto_adjust:
                        print(f"  🔧 Hardware-adaptive configuration:")
                        print(f"     💾 Available memory: {available_memory_gb:.1f}GB")
                        print(f"     📊 Max genes by memory: {max_genes_by_memory:,}")
                        print(f"     🎯 Adjusted genes per instance: {optimal_genes_per_instance:,}")
                    else:
                        print(f"  📊 Using user-specified configuration:")
                    print(f"  📊 Will train {optimal_instances} instances with {optimal_genes_per_instance} genes each")
                    print(f"  🔀 Instance overlap: {instance_overlap:.1%}")
                    print(f"  💾 Estimated memory per instance: {optimal_genes_per_instance * memory_per_gene_mb / 1024:.1f}GB")
                    print(f"  ✅ This ensures ALL {total_genes:,} genes contribute to the final model")
                
                return MultiInstanceEnsembleStrategy(
                    n_instances=optimal_instances,
                    genes_per_instance=optimal_genes_per_instance,
                    overlap_ratio=instance_overlap,
                    verbose=verbose
                )
            else:
                if verbose:
                    print(f"  🎯 Small-medium dataset with --train-all-genes: Using standard single model")
        
        elif not can_handle_single and total_genes > 2000:
            if verbose:
                print(f"  🎯 Large dataset detected: {total_genes:,} genes")
                print(f"  💡 Recommendation: Use --train-all-genes for complete gene coverage via multi-instance ensemble")
                print(f"  🔄 Current approach: Will use intelligent sampling (subset of genes)")
        
        # Fall back to standard selection
        return select_optimal_training_strategy(dataset_path, args, verbose)
        
    except Exception as e:
        if verbose:
            print(f"  ⚠️  Enhanced strategy selection failed: {e}")
        return select_optimal_training_strategy(dataset_path, args, verbose)
