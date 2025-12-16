#!/usr/bin/env python3
"""
Incremental Gene Trainer: Train meta-models on all genes without loading full dataset into memory.

This module implements a streaming approach that:
1. Processes genes in batches (e.g., 1000 genes at a time)
2. Loads only the data for those genes into memory
3. Trains incrementally on all genes without OOM
4. Preserves gene structure and cross-validation integrity
"""

import polars as pl
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging
from sklearn.model_selection import GroupKFold
import gc


class IncrementalGeneTrainer:
    """
    Trains meta-models incrementally on all genes without loading full dataset.
    
    Key features:
    - Processes genes in memory-efficient batches
    - Maintains gene-aware cross-validation
    - Supports all existing training parameters
    - Preserves full dataset diversity for meta-learning
    """
    
    def __init__(
        self,
        dataset_path: str | Path,
        batch_size_genes: int = 1000,
        max_memory_gb: float = 8.0,
        verbose: bool = True
    ):
        self.dataset_path = Path(dataset_path)
        self.batch_size_genes = batch_size_genes
        self.max_memory_gb = max_memory_gb
        self.verbose = verbose
        
        # Cache for dataset metadata
        self._gene_list = None
        self._total_genes = None
        self._estimated_rows = None
        
    def get_gene_list(self) -> List[str]:
        """Get list of all genes in dataset without loading full data."""
        if self._gene_list is not None:
            return self._gene_list
            
        if self.verbose:
            print("[IncrementalTrainer] Discovering genes in dataset...")
            
        # Scan dataset to get unique genes
        lf = pl.scan_parquet(
            str(self.dataset_path / "*.parquet"), 
            extra_columns='ignore'
        )
        
        # Get unique genes efficiently
        genes_df = lf.select("gene_id").unique().collect()
        self._gene_list = genes_df["gene_id"].to_list()
        self._total_genes = len(self._gene_list)
        
        if self.verbose:
            print(f"[IncrementalTrainer] Found {self._total_genes} unique genes")
            
        return self._gene_list
    
    def load_gene_batch(self, gene_batch: List[str]) -> pl.DataFrame:
        """Load data for a specific batch of genes."""
        if self.verbose and len(gene_batch) <= 10:
            print(f"[IncrementalTrainer] Loading genes: {gene_batch}")
        elif self.verbose:
            print(f"[IncrementalTrainer] Loading {len(gene_batch)} genes...")
            
        # Load data for specific genes
        lf = pl.scan_parquet(
            str(self.dataset_path / "*.parquet"), 
            extra_columns='ignore'
        )
        
        # Filter to specific genes
        batch_df = lf.filter(pl.col("gene_id").is_in(gene_batch)).collect()
        
        if self.verbose:
            print(f"[IncrementalTrainer] Loaded {batch_df.shape[0]:,} positions for {len(gene_batch)} genes")
            
        return batch_df
    
    def create_gene_batches(self, genes: List[str]) -> List[List[str]]:
        """Split genes into memory-efficient batches."""
        batches = []
        for i in range(0, len(genes), self.batch_size_genes):
            batch = genes[i:i + self.batch_size_genes]
            batches.append(batch)
            
        if self.verbose:
            print(f"[IncrementalTrainer] Created {len(batches)} gene batches")
            print(f"[IncrementalTrainer] Batch sizes: {[len(b) for b in batches]}")
            
        return batches
    
    def incremental_cross_validation(
        self,
        n_folds: int = 5,
        valid_size: float = 0.1,
        random_seed: int = 42
    ) -> Tuple[List[Dict], np.ndarray, np.ndarray, np.ndarray]:
        """
        Perform gene-aware cross-validation incrementally.
        
        Returns:
            fold_results: List of fold metrics
            X_full: Feature matrix for all data (built incrementally)
            y_full: Target vector for all data
            genes_full: Gene IDs for all data
        """
        genes = self.get_gene_list()
        
        if self.verbose:
            print(f"[IncrementalTrainer] Starting incremental CV with {len(genes)} genes")
            print(f"[IncrementalTrainer] Folds: {n_folds}, Validation size: {valid_size}")
        
        # Create gene-level CV splits
        gene_array = np.array(genes)
        dummy_y = np.zeros(len(genes))  # Dummy target for gene-level splitting
        
        gkf = GroupKFold(n_splits=n_folds)
        gene_splits = list(gkf.split(gene_array, dummy_y, gene_array))
        
        if self.verbose:
            print(f"[IncrementalTrainer] Created {len(gene_splits)} CV splits")
            
        # Process each fold incrementally
        fold_results = []
        all_data_batches = []  # Store processed batches for final concatenation
        
        for fold_idx, (train_gene_idx, test_gene_idx) in enumerate(gene_splits):
            if self.verbose:
                print(f"\n[IncrementalTrainer] Processing fold {fold_idx + 1}/{n_folds}")
                print(f"  Train genes: {len(train_gene_idx)}")
                print(f"  Test genes: {len(test_gene_idx)}")
            
            train_genes = gene_array[train_gene_idx].tolist()
            test_genes = gene_array[test_gene_idx].tolist()
            
            # Process training genes in batches
            fold_result = self._process_fold_incremental(
                fold_idx, train_genes, test_genes, valid_size, random_seed
            )
            fold_results.append(fold_result)
            
            # Collect data for this fold (for final model training)
            if fold_idx == 0:  # Only collect data once
                fold_data = self._collect_fold_data(train_genes + test_genes)
                all_data_batches.append(fold_data)
        
        # Combine all data for final model
        if self.verbose:
            print(f"\n[IncrementalTrainer] Combining all data for final model...")
            
        combined_data = pl.concat(all_data_batches)
        X_full, y_full, genes_full = self._prepare_features_and_targets(combined_data)
        
        return fold_results, X_full, y_full, genes_full
    
    def _process_fold_incremental(
        self,
        fold_idx: int,
        train_genes: List[str],
        test_genes: List[str],
        valid_size: float,
        random_seed: int
    ) -> Dict:
        """Process a single fold incrementally."""
        
        # Create gene batches for memory efficiency
        train_batches = self.create_gene_batches(train_genes)
        test_batches = self.create_gene_batches(test_genes)
        
        # Process training data in batches
        train_data_parts = []
        for batch_idx, gene_batch in enumerate(train_batches):
            if self.verbose:
                print(f"  Loading training batch {batch_idx + 1}/{len(train_batches)}")
            batch_data = self.load_gene_batch(gene_batch)
            train_data_parts.append(batch_data)
            
        # Combine training data
        train_data = pl.concat(train_data_parts)
        if self.verbose:
            print(f"  Combined training data: {train_data.shape}")
            
        # Process test data in batches
        test_data_parts = []
        for batch_idx, gene_batch in enumerate(test_batches):
            if self.verbose:
                print(f"  Loading test batch {batch_idx + 1}/{len(test_batches)}")
            batch_data = self.load_gene_batch(gene_batch)
            test_data_parts.append(batch_data)
            
        # Combine test data
        test_data = pl.concat(test_data_parts)
        if self.verbose:
            print(f"  Combined test data: {test_data.shape}")
        
        # Clean up batch data to free memory
        del train_data_parts, test_data_parts
        gc.collect()
        
        # Prepare features and targets
        X_train, y_train, genes_train = self._prepare_features_and_targets(train_data)
        X_test, y_test, genes_test = self._prepare_features_and_targets(test_data)
        
        # TODO: Implement actual training logic here
        # This would include:
        # 1. Train/validation split within training genes
        # 2. Train XGBoost models
        # 3. Evaluate on test set
        # 4. Return fold metrics
        
        fold_result = {
            'fold': fold_idx,
            'train_genes': len(train_genes),
            'test_genes': len(test_genes),
            'train_positions': X_train.shape[0],
            'test_positions': X_test.shape[0],
            # Add actual metrics here after implementing training
        }
        
        return fold_result
    
    def _collect_fold_data(self, genes: List[str]) -> pl.DataFrame:
        """Collect data for specified genes in batches."""
        gene_batches = self.create_gene_batches(genes)
        data_parts = []
        
        for batch_idx, gene_batch in enumerate(gene_batches):
            batch_data = self.load_gene_batch(gene_batch)
            data_parts.append(batch_data)
            
        return pl.concat(data_parts)
    
    def _prepare_features_and_targets(self, df: pl.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract features, targets, and gene IDs from DataFrame."""
        # TODO: Implement feature extraction logic
        # This should match the existing feature preparation in run_gene_cv_sigmoid.py
        
        # Placeholder implementation
        feature_cols = [col for col in df.columns if col not in ['gene_id', 'splice_type']]
        
        X = df.select(feature_cols).to_numpy()
        y = df['splice_type'].to_numpy()
        genes = df['gene_id'].to_numpy()
        
        return X, y, genes


def run_incremental_training(
    dataset_path: str | Path,
    out_dir: str | Path,
    n_folds: int = 5,
    batch_size_genes: int = 1000,
    max_memory_gb: float = 8.0,
    verbose: bool = True
) -> Dict:
    """
    Main entry point for incremental training on all genes.
    
    Args:
        dataset_path: Path to dataset directory with parquet files
        out_dir: Output directory for results
        n_folds: Number of CV folds
        batch_size_genes: Number of genes to process per batch
        max_memory_gb: Maximum memory to use
        verbose: Whether to print progress
        
    Returns:
        Dictionary with training results and metrics
    """
    trainer = IncrementalGeneTrainer(
        dataset_path=dataset_path,
        batch_size_genes=batch_size_genes,
        max_memory_gb=max_memory_gb,
        verbose=verbose
    )
    
    # Run incremental cross-validation
    fold_results, X_full, y_full, genes_full = trainer.incremental_cross_validation(
        n_folds=n_folds
    )
    
    if verbose:
        print(f"\n[IncrementalTrainer] Training completed!")
        print(f"  Total positions processed: {X_full.shape[0]:,}")
        print(f"  Total features: {X_full.shape[1]:,}")
        print(f"  Total genes: {len(np.unique(genes_full)):,}")
    
    results = {
        'fold_results': fold_results,
        'total_positions': X_full.shape[0],
        'total_features': X_full.shape[1],
        'total_genes': len(np.unique(genes_full)),
        'dataset_path': str(dataset_path),
        'output_dir': str(out_dir)
    }
    
    return results


if __name__ == "__main__":
    # Test the incremental trainer
    import argparse
    
    parser = argparse.ArgumentParser(description="Test incremental gene trainer")
    parser.add_argument("--dataset", required=True, help="Dataset path")
    parser.add_argument("--out-dir", default="results/incremental_test", help="Output directory")
    parser.add_argument("--batch-size", type=int, default=100, help="Genes per batch")
    parser.add_argument("--max-memory", type=float, default=8.0, help="Max memory GB")
    
    args = parser.parse_args()
    
    results = run_incremental_training(
        dataset_path=args.dataset,
        out_dir=args.out_dir,
        batch_size_genes=args.batch_size,
        max_memory_gb=args.max_memory,
        verbose=True
    )
    
    print(f"\nResults: {results}")
