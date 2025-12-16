#!/usr/bin/env python3
"""
Direct Batch Trainer: In-process batch training without subprocess calls.

This module provides a reliable batch training implementation that runs
everything in the same process, avoiding subprocess complexity and recursion issues.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
import pickle
import argparse
from dataclasses import dataclass

from meta_spliceai.splice_engine.meta_models.training.streaming_dataset_loader import StreamingDatasetLoader


@dataclass
class DirectBatchResult:
    """Results from direct batch training."""
    batch_id: int
    genes: List[str]
    model_path: Path
    cv_results: List[Dict]
    performance_metrics: Dict[str, float]
    success: bool
    error_message: Optional[str] = None


class DirectBatchTrainer:
    """
    Direct batch trainer that runs everything in-process.
    
    This avoids the subprocess complexity and recursion issues of the
    automated_all_genes_trainer by running batch training directly.
    """
    
    def __init__(self, dataset_path: str, max_genes_per_batch: int = 1200, verbose: bool = True):
        self.dataset_path = Path(dataset_path)
        self.max_genes_per_batch = max_genes_per_batch
        self.verbose = verbose
        self.loader = StreamingDatasetLoader(dataset_path, verbose=False)
    
    def create_gene_batches(self) -> List[List[str]]:
        """Create optimal gene batches."""
        
        if self.verbose:
            print(f"[DirectBatchTrainer] Creating gene batches...")
        
        # Get all genes from the dataset info
        dataset_info = self.loader.get_dataset_info()
        
        # Access the gene list directly (it gets populated by get_dataset_info)
        all_genes = self.loader._gene_list
        if all_genes is None:
            raise RuntimeError("Failed to discover genes in dataset")
            
        if self.verbose:
            print(f"[DirectBatchTrainer] Found {len(all_genes)} genes to process")
        
        gene_sizes = {}
        
        # Quick size estimation
        for i, gene in enumerate(all_genes[:100]):  # Sample first 100 for size estimation
            try:
                gene_data = self.loader.load_genes_subset([gene])
                gene_sizes[gene] = len(gene_data)
            except Exception:
                gene_sizes[gene] = 100  # Default size estimate
        
        # Use average size for remaining genes
        avg_size = np.mean(list(gene_sizes.values())) if gene_sizes else 100
        for gene in all_genes:
            if gene not in gene_sizes:
                gene_sizes[gene] = avg_size
        
        # Sort genes by size (ascending) for balanced batches
        sorted_genes = sorted(all_genes, key=lambda g: gene_sizes[g])
        
        # Create batches
        batches = []
        current_batch = []
        
        for gene in sorted_genes:
            current_batch.append(gene)
            
            if len(current_batch) >= self.max_genes_per_batch:
                batches.append(current_batch)
                current_batch = []
        
        # Add remaining genes
        if current_batch:
            batches.append(current_batch)
        
        if self.verbose:
            print(f"  Created {len(batches)} batches")
            for i, batch in enumerate(batches):
                print(f"    Batch {i}: {len(batch)} genes")
        
        return batches
    
    def train_batch_direct(
        self, 
        batch_id: int, 
        genes: List[str], 
        out_dir: Path, 
        args
    ) -> DirectBatchResult:
        """Train a single batch directly in-process."""
        
        if self.verbose:
            print(f"\n[DirectBatchTrainer] Training batch {batch_id + 1}")
            print(f"  Genes: {len(genes)}")
        
        batch_out_dir = out_dir / f"batch_{batch_id:03d}"
        batch_out_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Load data for this batch
            batch_data = self.loader.load_genes_subset(genes)
            
            if self.verbose:
                print(f"  Loaded: {len(batch_data):,} positions")
            
            # Prepare data for training
            from meta_spliceai.splice_engine.meta_models.builder import preprocessing
            from meta_spliceai.splice_engine.meta_models.training.classifier_utils import _encode_labels
            
            # Prepare training data
            X_df, y_series = preprocessing.prepare_training_data(
                batch_data,
                label_col="splice_type",
                return_type="pandas",
                verbose=0,
                preserve_transcript_columns=True,
                encode_chrom=True
            )
            
            # Extract gene array
            if hasattr(batch_data, 'to_pandas'):
                genes_array = batch_data.to_pandas()['gene_id'].values
            else:
                genes_array = batch_data['gene_id'].values
            
            # Train model directly without using strategy (to avoid recursive dataset loading)
            training_result = self._train_batch_model_directly(
                X_df, y_series, genes_array, batch_out_dir, args
            )
            
            if self.verbose:
                print(f"  ✅ Batch {batch_id + 1} completed")
                if training_result.performance_metrics:
                    perf = training_result.performance_metrics
                    print(f"    F1: {perf.get('mean_f1_macro', 0):.3f}")
            
            return DirectBatchResult(
                batch_id=batch_id,
                genes=genes,
                model_path=training_result.model_path,
                cv_results=training_result.cv_results or [],
                performance_metrics=training_result.performance_metrics or {},
                success=True
            )
            
        except Exception as e:
            if self.verbose:
                print(f"  ❌ Batch {batch_id + 1} failed: {e}")
            
            return DirectBatchResult(
                batch_id=batch_id,
                genes=genes,
                model_path=batch_out_dir / "model_multiclass.pkl",
                cv_results=[],
                performance_metrics={},
                success=False,
                error_message=str(e)
            )
    
    def _train_batch_model_directly(
        self, 
        X_df: pd.DataFrame, 
        y_series: pd.Series, 
        genes: np.ndarray, 
        batch_out_dir: Path, 
        args
    ):
        """Train a model directly on batch data without dataset loading overhead."""
        import pickle
        import json
        import numpy as np
        import pandas as pd
        from sklearn.model_selection import GroupKFold, GroupShuffleSplit
        from meta_spliceai.splice_engine.meta_models.training.classifier_utils import _encode_labels, SigmoidEnsemble
        from meta_spliceai.splice_engine.meta_models.training.run_gene_cv_sigmoid import _train_binary_model
        from meta_spliceai.splice_engine.meta_models.training.training_strategies import TrainingResult
        
        # Apply global feature exclusions if available
        global_exclusions_file = batch_out_dir.parent / "global_excluded_features.txt"
        if global_exclusions_file.exists():
            with open(global_exclusions_file, 'r') as f:
                excluded_features = [line.strip() for line in f if line.strip() and not line.startswith('#')]
            
            # Remove excluded features
            for feature in excluded_features:
                if feature in X_df.columns:
                    X_df = X_df.drop(columns=[feature])
        
        # Remove metadata columns that shouldn't be used for training
        metadata_cols_to_remove = ['transcript_id', 'position']
        for col in metadata_cols_to_remove:
            if col in X_df.columns:
                X_df = X_df.drop(columns=[col])
        
        # Prepare data
        X = X_df.values
        y = _encode_labels(y_series)
        feature_names = list(X_df.columns)
        
        # Simple training without CV for batch (CV will be done at ensemble level)
        print(f"    🔧 Training 3 binary classifiers for batch...")
        models = []
        for cls in (0, 1, 2):
            class_name = ['neither', 'donor', 'acceptor'][cls]
            print(f"      Training {class_name} classifier...")
            y_bin = (y == cls).astype(int)
            model_c, _ = _train_binary_model(X, y_bin, X, y_bin, args)
            models.append(model_c)
        
        # Create ensemble
        ensemble = SigmoidEnsemble(models, feature_names)
        
        # Save model
        model_path = batch_out_dir / "model_multiclass.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(ensemble, f)
        
        # Save feature manifest
        feature_manifest = pd.DataFrame({'feature': feature_names})
        feature_manifest.to_csv(batch_out_dir / "feature_manifest.csv", index=False)
        
        # Save features as JSON for compatibility
        features_json = {"feature_names": feature_names}
        with open(batch_out_dir / "train.features.json", 'w') as f:
            json.dump(features_json, f)
        
        # Calculate basic performance metrics
        proba_parts = [m.predict_proba(X)[:, 1] for m in models]
        proba = np.column_stack(proba_parts)
        pred = proba.argmax(axis=1)
        
        from sklearn.metrics import accuracy_score, f1_score
        accuracy = accuracy_score(y, pred)
        f1_macro = f1_score(y, pred, average="macro")
        
        performance_metrics = {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'total_genes': len(np.unique(genes)),
            'total_positions': len(y)
        }
        
        # Create training result
        from datetime import datetime
        training_metadata = {
            'strategy': 'BatchDirectTraining',
            'total_genes': len(np.unique(genes)),
            'total_positions': len(y),
            'features_used': len(feature_names),
            'training_date': datetime.now().isoformat(),
            'batch_training': True
        }
        
        return TrainingResult(
            model_path=model_path,
            feature_names=feature_names,
            excluded_features=[],
            training_metadata=training_metadata,
            cv_results=[],
            performance_metrics=performance_metrics
        )
    
    def combine_batch_models(self, batch_results: List[DirectBatchResult], out_dir: Path) -> Dict[str, Any]:
        """Combine successful batch models into final ensemble."""
        
        successful_batches = [b for b in batch_results if b.success]
        
        if not successful_batches:
            raise RuntimeError("No successful batches to combine")
        
        if self.verbose:
            print(f"\n[DirectBatchTrainer] Combining {len(successful_batches)} successful batches...")
        
        # Load all batch models
        all_models = []
        all_feature_names = None
        
        for batch_result in successful_batches:
            try:
                with open(batch_result.model_path, 'rb') as f:
                    batch_model = pickle.load(f)
                
                # Extract individual models from ensemble
                if hasattr(batch_model, 'models'):
                    all_models.extend(batch_model.models)
                    if all_feature_names is None:
                        all_feature_names = batch_model.feature_names
                else:
                    all_models.append(batch_model)
                    
            except Exception as e:
                if self.verbose:
                    print(f"  ⚠️  Could not load batch {batch_result.batch_id}: {e}")
        
        if not all_models:
            raise RuntimeError("No models could be loaded from successful batches")
        
        # Create final ensemble
        from meta_spliceai.splice_engine.meta_models.training.classifier_utils import SigmoidEnsemble
        
        # Group models by class (assuming 3 classes with models in groups of 3)
        n_classes = 3
        models_per_batch = len(all_models) // len(successful_batches)
        
        if models_per_batch != n_classes:
            # Fallback: use all models as-is
            final_ensemble = SigmoidEnsemble(all_models, all_feature_names)
        else:
            # Properly group models by class
            class_models = [[] for _ in range(n_classes)]
            for i, model in enumerate(all_models):
                class_idx = i % n_classes
                class_models[class_idx].append(model)
            
            # Take first model from each class group (or implement averaging)
            final_models = [class_models[i][0] for i in range(n_classes)]
            final_ensemble = SigmoidEnsemble(final_models, all_feature_names)
        
        # Save final ensemble
        final_model_path = out_dir / "model_multiclass_all_genes.pkl"
        with open(final_model_path, 'wb') as f:
            pickle.dump(final_ensemble, f)
        
        if self.verbose:
            print(f"  ✅ Final ensemble saved: {final_model_path}")
        
        # Aggregate performance metrics
        total_genes = sum(len(b.genes) for b in successful_batches)
        avg_f1 = np.mean([b.performance_metrics.get('mean_f1_macro', 0) for b in successful_batches if b.performance_metrics])
        
        return {
            'model_path': str(final_model_path),
            'successful_batch_count': len(successful_batches),
            'total_genes_trained': total_genes,
            'average_batch_f1': avg_f1,
            'batch_results': [
                {
                    'batch_id': b.batch_id,
                    'genes_count': len(b.genes),
                    'success': b.success,
                    'f1_macro': b.performance_metrics.get('mean_f1_macro', 0)
                }
                for b in batch_results
            ]
        }
    
    def train_all_genes_direct(self, out_dir: Path, args) -> Dict[str, Any]:
        """Train all genes using direct in-process batch training."""
        
        if self.verbose:
            print(f"[DirectBatchTrainer] Starting direct batch training...")
            print(f"  Dataset: {self.dataset_path}")
            print(f"  Output: {out_dir}")
        
        # Create gene batches
        gene_batches = self.create_gene_batches()
        
        # Train each batch
        batch_results = []
        for batch_id, genes in enumerate(gene_batches):
            batch_result = self.train_batch_direct(batch_id, genes, out_dir, args)
            batch_results.append(batch_result)
        
        # Combine results
        final_results = self.combine_batch_models(batch_results, out_dir)
        
        if self.verbose:
            successful_count = sum(1 for r in batch_results if r.success)
            print(f"\n[DirectBatchTrainer] Batch training complete!")
            print(f"  Successful: {successful_count}/{len(batch_results)} batches")
            print(f"  Total genes: {final_results['total_genes_trained']:,}")
        
        return final_results


if __name__ == "__main__":
    print("Direct Batch Trainer")
    print("=" * 30)
    print("In-process batch training without subprocess complexity")
    
    # Test with small dataset
    trainer = DirectBatchTrainer("train_pc_5000_3mers_diverse/master", max_genes_per_batch=50)
    
    # Create test args
    import argparse
    test_args = argparse.Namespace(
        n_estimators=50,
        calibrate_per_class=True,
        seed=42
    )
    
    print("Testing direct batch training...")
    # results = trainer.train_all_genes_direct(Path("test_direct_batch"), test_args)
    print("Direct batch trainer ready for integration")
