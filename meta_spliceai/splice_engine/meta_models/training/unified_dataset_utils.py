#!/usr/bin/env python3
"""
Unified Dataset Loading Utilities

This module provides a clean, unified interface for dataset loading that handles:
- Memory optimization for large datasets
- Gene sampling for testing
- Schema handling and error recovery
- Feature preprocessing and encoding
- Consistent data preparation across all training strategies

This replaces the scattered dataset loading logic that was previously embedded
in the main training scripts, providing a clean abstraction layer.
"""

from pathlib import Path
from typing import Tuple, Optional, Any
import numpy as np
import pandas as pd
import argparse

from meta_spliceai.splice_engine.meta_models.training import datasets
from meta_spliceai.splice_engine.meta_models.builder import preprocessing


def load_and_prepare_training_dataset(
    dataset_path: str,
    args: argparse.Namespace,
    verbose: bool = True
) -> Tuple[Any, pd.DataFrame, pd.Series, np.ndarray]:
    """
    Load and prepare dataset for training with unified handling.
    
    This function encapsulates all dataset loading logic including:
    - Gene sampling vs full dataset loading
    - Memory optimization for large datasets
    - Schema error handling
    - Feature preprocessing
    - Gene extraction
    
    Parameters
    ----------
    dataset_path : str
        Path to dataset directory or file
    args : argparse.Namespace
        Parsed command-line arguments
    verbose : bool, default=True
        Whether to print loading information
        
    Returns
    -------
    Tuple[Any, pd.DataFrame, pd.Series, np.ndarray]
        (raw_df, X_df, y_series, genes)
        - raw_df: Original dataframe with all columns
        - X_df: Feature matrix as pandas DataFrame
        - y_series: Target labels as pandas Series
        - genes: Gene IDs as numpy array
    """
    
    if verbose:
        print(f"🔍 [Dataset Loading] Loading dataset: {dataset_path}", flush=True)
    
    # 1. Load raw dataset using appropriate strategy
    print(f"  📂 Loading raw dataset...", flush=True)
    raw_df = _load_raw_dataset(dataset_path, args, verbose)
    
    # 2. Validate required columns
    print(f"  ✅ Validating dataset columns...", flush=True)
    _validate_dataset_columns(raw_df, args)
    
    # 3. Prepare training features and labels
    print(f"  🔧 Preparing training features and labels...", flush=True)
    X_df, y_series = _prepare_training_features(raw_df, args, verbose)
    
    # 4. Extract gene information
    print(f"  🧬 Extracting gene information...", flush=True)
    genes = _extract_gene_array(raw_df, args)
    
    print(f"✅ [Dataset Loading] Dataset preparation completed!", flush=True)
    print(f"  📊 Raw data: {raw_df.shape[0]:,} positions", flush=True)
    print(f"  🔧 Features: {X_df.shape[1]} features", flush=True)
    print(f"  🧬 Genes: {len(np.unique(genes)):,} unique genes", flush=True)
    print(f"  🏷️  Labels: {len(y_series)} positions", flush=True)
    
    return raw_df, X_df, y_series, genes


def _load_raw_dataset(
    dataset_path: str,
    args: argparse.Namespace,
    verbose: bool = True
) -> Any:
    """Load raw dataset using appropriate loading strategy."""
    
    # Handle gene sampling
    if getattr(args, 'sample_genes', None) is not None:
        if verbose:
            print(f"  📊 Gene sampling: {args.sample_genes} genes")
        
        from meta_spliceai.splice_engine.meta_models.training.label_utils import load_dataset_sample
        return load_dataset_sample(
            dataset_path, 
            sample_genes=args.sample_genes, 
            random_seed=getattr(args, 'seed', 42)
        )
    
    # Handle full dataset loading with memory optimization
    return _load_full_dataset_with_optimization(dataset_path, args, verbose)


def _load_full_dataset_with_optimization(
    dataset_path: str,
    args: argparse.Namespace,
    verbose: bool = True
) -> Any:
    """Load full dataset with intelligent memory optimization."""
    
    try:
        from meta_spliceai.splice_engine.meta_models.training.streaming_dataset_loader import (
            create_memory_efficient_dataset,
            StreamingDatasetLoader
        )
        
        if verbose:
            print(f"  🧠 Using memory-optimized loading")
        
        # Analyze dataset characteristics
        loader = StreamingDatasetLoader(dataset_path, verbose=verbose)
        info = loader.get_dataset_info()
        total_genes = info['total_genes']
        
        if verbose:
            print(f"  📈 Dataset analysis: {total_genes:,} genes")
        
        # Calculate optimal gene limit
        from meta_spliceai.splice_engine.meta_models.training.run_gene_cv_sigmoid import _calculate_optimal_gene_limit
        max_genes_safe = _calculate_optimal_gene_limit(
            total_genes=total_genes,
            max_genes_override=getattr(args, 'max_genes_in_memory', None),
            safety_factor=getattr(args, 'memory_safety_factor', 0.6),
            verbose=verbose
        )
        
        # Load dataset based on memory constraints
        if total_genes <= max_genes_safe:
            if verbose:
                print(f"  ✅ Loading all {total_genes} genes (fits safely in memory)")
            
            return create_memory_efficient_dataset(
                dataset_path,
                max_genes_in_memory=total_genes,
                max_memory_gb=25.0,
                gene_start_idx=getattr(args, 'gene_start_idx', 0),
                gene_end_idx=getattr(args, 'gene_end_idx', None),
                verbose=verbose
            )
        else:
            if verbose:
                print(f"  ⚠️  Large dataset: {total_genes} genes > {max_genes_safe} memory limit")
                print(f"  📊 Using representative sample of {max_genes_safe} genes")
            
            return create_memory_efficient_dataset(
                dataset_path,
                max_genes_in_memory=max_genes_safe,
                max_memory_gb=15.0,
                gene_start_idx=getattr(args, 'gene_start_idx', 0),
                gene_end_idx=getattr(args, 'gene_end_idx', None),
                verbose=verbose
            )
            
    except ImportError as e:
        if verbose:
            print(f"  ⚠️  Memory optimization not available: {e}")
            print(f"  🔄 Falling back to standard dataset loading")
        
        return datasets.load_dataset(dataset_path)
        
    except Exception as e:
        if verbose:
            print(f"  ❌ Memory-optimized loading failed: {e}")
        
        # Handle schema errors specifically
        if "3mer_NNN" in str(e) or "SchemaError" in str(e):
            print(f"  🔧 Schema mismatch detected - run schema validation:")
            print(f"     python -m meta_spliceai.splice_engine.meta_models.builder.validate_dataset_schema {dataset_path} --fix")
            raise RuntimeError(f"Dataset schema issues prevent loading: {e}")
        else:
            raise RuntimeError(f"Dataset loading failed: {e}")


def _validate_dataset_columns(df: Any, args: argparse.Namespace) -> None:
    """Validate that dataset contains required columns."""
    
    required_columns = [
        getattr(args, 'gene_col', 'gene_id'),
        'splice_type'
    ]
    
    missing_columns = []
    for col in required_columns:
        if col not in df.columns:
            missing_columns.append(col)
    
    if missing_columns:
        raise KeyError(f"Dataset missing required columns: {missing_columns}")


def _prepare_training_features(
    df: Any,
    args: argparse.Namespace,
    verbose: bool = True
) -> Tuple[pd.DataFrame, pd.Series]:
    """Prepare training features and labels."""
    
    if verbose:
        print(f"  🔧 Preparing training features...")
    
    return preprocessing.prepare_training_data(
        df,
        label_col="splice_type",
        return_type="pandas",
        verbose=1 if verbose else 0,
        preserve_transcript_columns=True,  # Always preserve for top-k accuracy
        encode_chrom=True
    )


def _extract_gene_array(df: Any, args: argparse.Namespace) -> np.ndarray:
    """Extract gene IDs as numpy array."""
    gene_col = getattr(args, 'gene_col', 'gene_id')
    return df[gene_col].to_numpy()


def get_dataset_statistics(
    raw_df: Any,
    X_df: pd.DataFrame,
    genes: np.ndarray,
    verbose: bool = True
) -> dict:
    """Get comprehensive dataset statistics for documentation."""
    
    unique_genes = np.unique(genes)
    
    stats = {
        'total_positions': len(raw_df),
        'total_genes': len(unique_genes),
        'features_count': X_df.shape[1],
        'feature_names': list(X_df.columns),
        'genes_list': unique_genes.tolist()
    }
    
    if verbose:
        print(f"📊 [Dataset Statistics]:")
        print(f"  Total positions: {stats['total_positions']:,}")
        print(f"  Total genes: {stats['total_genes']:,}")
        print(f"  Features: {stats['features_count']}")
    
    return stats


def create_evaluation_dataset(
    raw_df: Any,
    out_dir: Path,
    sample_size: Optional[int] = None,
    verbose: bool = True
) -> Path:
    """
    Create temporary evaluation dataset for post-training analysis.
    
    This avoids reloading the full dataset during evaluation phases.
    
    ⚠️  WARNING: This function currently uses the same data as training,
    which leads to data leakage and artificially high performance scores.
    For proper evaluation, use create_holdout_evaluation_dataset() instead.
    """
    
    temp_dataset_path = out_dir / "temp_evaluation_dataset.parquet"
    
    if not temp_dataset_path.exists():
        if verbose:
            print(f"  💾 Creating evaluation dataset...")
            print(f"  ⚠️  WARNING: Using same data as training (data leakage!)")
        
        eval_df = raw_df
        
        # Apply sampling if needed
        if sample_size is not None and len(eval_df) > sample_size:
            # Polars syntax: use seed instead of random_state
            eval_df = eval_df.sample(n=sample_size, seed=42)
            if verbose:
                print(f"    📊 Sampled to {len(eval_df):,} positions")
        
        # Save as parquet
        eval_df.write_parquet(temp_dataset_path)
        
        if verbose:
            print(f"    ✅ Evaluation dataset saved: {temp_dataset_path}")
    else:
        if verbose:
            print(f"  ♻️  Using existing evaluation dataset: {temp_dataset_path}")
    
    return temp_dataset_path


def create_holdout_evaluation_dataset(
    dataset_path: str,
    training_genes: set,
    out_dir: Path,
    sample_size: Optional[int] = None,
    verbose: bool = True
) -> Path:
    """
    Create a proper evaluation dataset using UNSEEN GENES for honest evaluation.
    
    This function creates a holdout test set that excludes all genes used during training,
    preventing data leakage and providing realistic performance estimates.
    
    Parameters
    ----------
    dataset_path : str
        Path to the full dataset
    training_genes : set
        Set of gene IDs used during training (to exclude)
    out_dir : Path
        Output directory for the evaluation dataset
    sample_size : Optional[int]
        Maximum number of positions to include
    verbose : bool
        Whether to print progress information
        
    Returns
    -------
    Path
        Path to the created holdout evaluation dataset
    """
    import polars as pl
    from pathlib import Path
    
    holdout_dataset_path = out_dir / "holdout_evaluation_dataset.parquet"
    
    if not holdout_dataset_path.exists():
        if verbose:
            print(f"  🎯 Creating holdout evaluation dataset with unseen genes...")
        
        # Load full dataset
        if Path(dataset_path).is_dir():
            lf = pl.scan_parquet(str(Path(dataset_path) / "*.parquet"), 
                               missing_columns="insert", extra_columns="ignore")
        else:
            lf = pl.scan_parquet(str(dataset_path), 
                               missing_columns="insert", extra_columns="ignore")
        
        # Get all unique genes
        all_genes_df = lf.select("gene_id").unique().collect()
        all_genes = set(all_genes_df["gene_id"].to_list())
        
        # Find holdout genes (not used in training)
        holdout_genes = all_genes - training_genes
        
        if verbose:
            print(f"    📊 Total genes in dataset: {len(all_genes):,}")
            print(f"    🚫 Training genes (excluded): {len(training_genes):,}")
            print(f"    ✅ Holdout genes (for evaluation): {len(holdout_genes):,}")
        
        if len(holdout_genes) == 0:
            raise ValueError("No holdout genes available! All genes were used in training.")
        
        # Filter to holdout genes only
        holdout_lf = lf.filter(pl.col("gene_id").is_in(list(holdout_genes)))
        
        # Collect the holdout data
        eval_df = holdout_lf.collect(streaming=True)
        
        if verbose:
            print(f"    📈 Holdout dataset size: {len(eval_df):,} positions")
        
        # Apply sampling if needed (gene-aware sampling)
        if sample_size is not None and len(eval_df) > sample_size:
            # Use gene-aware sampling to preserve gene structure
            from meta_spliceai.splice_engine.meta_models.training.meta_evaluation_utils import gene_wise_sample_dataframe
            eval_df = gene_wise_sample_dataframe(
                eval_df, target_positions=sample_size, verbose=verbose
            )
            if verbose:
                print(f"    📊 Sampled to {len(eval_df):,} positions (gene-aware)")
        
        # Save as parquet
        eval_df.write_parquet(holdout_dataset_path)
        
        if verbose:
            print(f"    ✅ Holdout evaluation dataset saved: {holdout_dataset_path}")
            print(f"    🎯 This dataset contains UNSEEN GENES for honest evaluation")
    else:
        if verbose:
            print(f"  ♻️  Using existing holdout evaluation dataset: {holdout_dataset_path}")
    
    return holdout_dataset_path
