"""Meta-model evaluation utilities.

This module contains utility functions for evaluating meta-model performance
that are used by various driver scripts and analysis tools.
"""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
import polars as pl
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from . import classifier_utils as _cutils


def _incremental_gene_evaluation(
    df: pl.DataFrame,
    dataset_path: Path,
    run_dir: Path,
    predict_fn,
    feature_names: list,
    available_feature_cols: list,
    target_sample_size: int,
    verbose: bool,
    out_tsv: Path | None = None
) -> Path:
    """
    Perform incremental gene-based evaluation to avoid memory issues.
    
    Processes genes in chunks, preserving gene structure while staying within memory limits.
    """
    import numpy as np
    import pandas as pd
    from pathlib import Path
    import json
    
    if verbose:
        print(f"[incremental_eval] Starting incremental gene-based evaluation")
        print(f"[incremental_eval] Target sample size: {target_sample_size:,}")
    
    # Get unique genes and estimate chunk size
    unique_genes = df.select("gene_id").unique().to_pandas()["gene_id"].tolist()
    total_genes = len(unique_genes)
    
    # Calculate chunk size to stay within memory limits
    avg_positions_per_gene = df.height // total_genes
    target_genes_per_chunk = max(1, target_sample_size // (avg_positions_per_gene * 2))  # Conservative estimate
    
    if verbose:
        print(f"[incremental_eval] Total genes: {total_genes:,}")
        print(f"[incremental_eval] Avg positions per gene: {avg_positions_per_gene:.0f}")
        print(f"[incremental_eval] Target genes per chunk: {target_genes_per_chunk}")
    
    # Process genes in chunks
    all_results = []
    processed_positions = 0
    processed_genes = 0
    
    for chunk_start in range(0, total_genes, target_genes_per_chunk):
        chunk_end = min(chunk_start + target_genes_per_chunk, total_genes)
        chunk_genes = unique_genes[chunk_start:chunk_end]
        
        if verbose:
            print(f"[incremental_eval] Processing chunk {chunk_start//target_genes_per_chunk + 1}/{(total_genes + target_genes_per_chunk - 1)//target_genes_per_chunk}")
            print(f"[incremental_eval] Genes {chunk_start+1}-{chunk_end} ({len(chunk_genes)} genes)")
        
        # Filter to this chunk of genes
        chunk_df = df.filter(pl.col("gene_id").is_in(chunk_genes))
        chunk_positions = chunk_df.height
        
        if chunk_positions == 0:
            continue
            
        # Convert chunk to pandas for processing
        chunk_pd = chunk_df.to_pandas()
        
        # Extract features for this chunk
        from meta_spliceai.splice_engine.meta_models.builder.preprocessing import (
            LEAKAGE_COLUMNS, METADATA_COLUMNS
        )
        
        exclude_cols = set(LEAKAGE_COLUMNS + METADATA_COLUMNS + ["splice_type"])
        chunk_feature_cols = [col for col in available_feature_cols if col not in exclude_cols]
        
        # Create feature matrix for this chunk with robust data type handling
        X_chunk = np.zeros((len(chunk_pd), len(feature_names)), dtype=np.float32)
        
        # Fill in available features at their correct positions with data type conversion
        for i, feat_name in enumerate(feature_names):
            if feat_name in chunk_feature_cols:
                try:
                    # Handle numeric columns
                    col_values = chunk_pd[feat_name].fillna(0)
                    if col_values.dtype == 'object':
                        # Convert categorical/string columns to numeric
                        col_values = pd.to_numeric(col_values, errors='coerce').fillna(0)
                    X_chunk[:, i] = col_values.values.astype(np.float32)
                except (ValueError, TypeError) as e:
                    if verbose > 1:
                        print(f"[incremental_eval] Warning: Could not convert feature '{feat_name}' to numeric, using zeros: {e}")
                    # Leave as zeros for problematic columns
        
        # Get predictions for this chunk
        meta_proba = predict_fn(X_chunk)
        meta_pred = np.argmax(meta_proba, axis=1)
        
        # Get base predictions
        base_donor = chunk_pd["donor_score"].values
        base_acceptor = chunk_pd["acceptor_score"].values
        base_proba = np.column_stack([
            1 - (base_donor + base_acceptor),  # neither
            base_donor,                        # donor  
            base_acceptor                      # acceptor
        ])
        base_proba = np.clip(base_proba, 0, 1)  # Ensure valid probabilities
        base_pred = np.argmax(base_proba, axis=1)
        
        # Get true labels
        true_labels = chunk_pd["splice_type"].map({"neither": 0, "donor": 1, "acceptor": 2}).values
        
        # Calculate metrics for this chunk
        chunk_results = {
            "chunk_id": chunk_start // target_genes_per_chunk,
            "genes": chunk_genes,
            "positions": chunk_positions,
            "base_accuracy": (base_pred == true_labels).mean(),
            "meta_accuracy": (meta_pred == true_labels).mean(),
            "base_pred": base_pred,
            "meta_pred": meta_pred,
            "true_labels": true_labels,
            "gene_ids": chunk_pd["gene_id"].values
        }
        
        all_results.append(chunk_results)
        processed_positions += chunk_positions
        processed_genes += len(chunk_genes)
        
        if verbose:
            print(f"[incremental_eval] Chunk {chunk_results['chunk_id']+1} complete: {chunk_positions:,} positions")
            print(f"[incremental_eval] Chunk accuracy: Base={chunk_results['base_accuracy']:.3f}, Meta={chunk_results['meta_accuracy']:.3f}")
        
        # Stop if we've processed enough positions
        if processed_positions >= target_sample_size:
            if verbose:
                print(f"[incremental_eval] Reached target sample size, stopping early")
            break
    
    # Aggregate results across all chunks
    if verbose:
        print(f"[incremental_eval] Aggregating results from {len(all_results)} chunks")
        print(f"[incremental_eval] Total processed: {processed_positions:,} positions, {processed_genes} genes")
    
    # Combine predictions and labels
    all_base_pred = np.concatenate([r["base_pred"] for r in all_results])
    all_meta_pred = np.concatenate([r["meta_pred"] for r in all_results])
    all_true_labels = np.concatenate([r["true_labels"] for r in all_results])
    all_gene_ids = np.concatenate([r["gene_ids"] for r in all_results])
    
    # Calculate overall metrics
    base_accuracy = (all_base_pred == all_true_labels).mean()
    meta_accuracy = (all_meta_pred == all_true_labels).mean()
    
    # Calculate F1 scores
    from sklearn.metrics import f1_score, classification_report
    base_f1_macro = f1_score(all_true_labels, all_base_pred, average='macro')
    meta_f1_macro = f1_score(all_true_labels, all_meta_pred, average='macro')
    
    # Count corrections and regressions
    base_correct = all_base_pred == all_true_labels
    meta_correct = all_meta_pred == all_true_labels
    corrections = np.sum(~base_correct & meta_correct)  # meta fixed base errors
    regressions = np.sum(base_correct & ~meta_correct)  # meta introduced errors
    
    # Calculate per-class F1 improvements
    base_report = classification_report(all_true_labels, all_base_pred, output_dict=True, zero_division=0)
    meta_report = classification_report(all_true_labels, all_meta_pred, output_dict=True, zero_division=0)
    
    class_names = ["neither", "donor", "acceptor"]
    per_class_f1 = {}
    for i, class_name in enumerate(class_names):
        base_f1 = base_report[str(i)]["f1-score"]
        meta_f1 = meta_report[str(i)]["f1-score"]
        per_class_f1[class_name] = {
            "base_f1": base_f1,
            "meta_f1": meta_f1,
            "improvement": meta_f1 - base_f1
        }
    
    # Create evaluation summary
    evaluation_summary = {
        "total_positions": processed_positions,
        "total_genes": processed_genes,
        "base_accuracy": float(base_accuracy),
        "meta_accuracy": float(meta_accuracy),
        "accuracy_improvement": float(meta_accuracy - base_accuracy),
        "base_f1_macro": float(base_f1_macro),
        "meta_f1_macro": float(meta_f1_macro),
        "f1_improvement": float(meta_f1_macro - base_f1_macro),
        "corrections": int(corrections),
        "regressions": int(regressions),
        "net_improvement": int(corrections - regressions),
        "improvement_rate": float((corrections - regressions) / processed_positions),
        "per_class_f1": per_class_f1
    }
    
    # Add per-class F1 scores to summary for backward compatibility
    for class_name, metrics in per_class_f1.items():
        evaluation_summary[f"base_f1_{class_name}"] = metrics["base_f1"]
        evaluation_summary[f"meta_f1_{class_name}"] = metrics["meta_f1"] 
        evaluation_summary[f"f1_improvement_{class_name}"] = metrics["improvement"]
    
    # Save evaluation summary
    summary_path = run_dir / "meta_evaluation_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(evaluation_summary, f, indent=2)
    
    # Save detailed results if requested
    if out_tsv:
        detailed_results = pd.DataFrame({
            "gene_id": all_gene_ids,
            "true_label": all_true_labels,
            "base_pred": all_base_pred,
            "meta_pred": all_meta_pred,
            "base_correct": base_correct,
            "meta_correct": meta_correct,
            "correction": ~base_correct & meta_correct,
            "regression": base_correct & ~meta_correct
        })
        detailed_results.to_csv(out_tsv, sep='\t', index=False)
        
        if verbose:
            print(f"[incremental_eval] Detailed results saved to: {out_tsv}")
    
    if verbose:
        print(f"[incremental_eval] 📊 INCREMENTAL EVALUATION COMPLETE:")
        print(f"  Processed: {processed_positions:,} positions from {processed_genes} genes")
        print(f"  Base accuracy: {base_accuracy:.3f}")
        print(f"  Meta accuracy: {meta_accuracy:.3f}")
        print(f"  Accuracy improvement: {meta_accuracy - base_accuracy:+.3f}")
        print(f"  Base F1 (macro): {base_f1_macro:.3f}")
        print(f"  Meta F1 (macro): {meta_f1_macro:.3f}")
        print(f"  F1 improvement: {meta_f1_macro - base_f1_macro:+.3f}")
        print(f"  Corrections: {corrections:,}, Regressions: {regressions:,}")
        print(f"  Net improvement: {corrections - regressions:+,} ({(corrections - regressions) / processed_positions:.1%})")
    
    return summary_path


def gene_wise_sample_dataframe(
    df: Union[pd.DataFrame, pl.DataFrame],
    target_positions: int,
    gene_col: str = "gene_id",
    random_state: int = 42,
    verbose: bool = True
) -> Union[pd.DataFrame, pl.DataFrame]:
    """
    Sample genes from a DataFrame to approximate target position count while preserving gene structure.
    
    This utility implements the gene-wise sampling logic that's reused across evaluation functions
    to ensure meaningful splice site distribution and avoid class imbalance issues.
    
    Parameters
    ----------
    df : pd.DataFrame or pl.DataFrame
        Input dataframe to sample from
    target_positions : int
        Approximate number of positions/rows desired in output
    gene_col : str, default="gene_id"
        Column name containing gene identifiers
    random_state : int, default=42
        Random seed for reproducibility
    verbose : bool, default=True
        Whether to print sampling information
        
    Returns
    -------
    pd.DataFrame or pl.DataFrame
        Sampled dataframe containing complete genes (same type as input)
        
    Examples
    --------
    >>> # Sample approximately 1000 positions by gene
    >>> sampled_df = gene_wise_sample_dataframe(df, target_positions=1000, verbose=True)
    """
    is_polars_input = isinstance(df, pl.DataFrame)
    
    # Convert to pandas for easier manipulation
    if is_polars_input:
        df_pd = df.to_pandas()
    else:
        df_pd = df.copy()
    
    # Calculate gene statistics
    positions_per_gene = df_pd.groupby(gene_col).size()
    mean_positions_per_gene = positions_per_gene.mean()
    total_genes = len(positions_per_gene)
    target_genes = max(1, int(target_positions / mean_positions_per_gene))
    
    if total_genes <= target_genes:
        # Use all genes if we don't have enough
        if verbose:
            print(f"[gene_wise_sample] Using all {total_genes} genes ({len(df_pd):,} positions)")
        return df if is_polars_input else df_pd
    
    # Sample genes
    sampled_genes = positions_per_gene.sample(n=target_genes, random_state=random_state).index
    df_sampled = df_pd[df_pd[gene_col].isin(sampled_genes)]
    
    if verbose:
        # Show splice site distribution if available
        if 'splice_type' in df_sampled.columns:
            splice_counts = df_sampled['splice_type'].value_counts()
            print(f"[gene_wise_sample] {target_genes} genes → {len(df_sampled):,} positions")
            print(f"  Splice site distribution: {dict(splice_counts)}")
        else:
            print(f"[gene_wise_sample] {target_genes} genes → {len(df_sampled):,} positions")
    
    # Convert back to original format
    if is_polars_input:
        return pl.from_pandas(df_sampled)
    else:
        return df_sampled


def meta_splice_performance_correct(
    dataset_path: str | Path,
    run_dir: str | Path,
    *,
    sample: int | None = None,
    out_tsv: str | Path | None = None,
    verbose: int = 1,
) -> Path:
    """
    CORRECT meta-model evaluation that compares meta vs base model performance
    at the SAME positions that were included in training data.
    
    This evaluation approach:
    1. Loads the training dataset positions (NOT full gene sequences)
    2. Applies meta-model to those exact positions 
    3. Compares meta vs base predictions at each position
    4. Reports improvement in classification accuracy (NOT position detection)
    
    Parameters
    ----------
    dataset_path : str | Path
        Path to training dataset (Parquet file or directory with *.parquet)
    run_dir : str | Path  
        Directory containing trained meta-model artifacts
    sample : int | None
        Optional: sample this many positions for faster evaluation
    out_tsv : str | Path | None
        Output file for detailed per-position results
    verbose : int
        Verbosity level
        
    Returns
    -------
    Path
        Path to generated evaluation summary file
    """
    dataset_path = Path(dataset_path)
    run_dir = Path(run_dir)
    
    if verbose:
        print(f"\n[meta_splice_eval] CORRECT EVALUATION: Comparing meta vs base at training positions")
        print(f"[meta_splice_eval] Dataset: {dataset_path}")
        print(f"[meta_splice_eval] Model: {run_dir}")
    
    # Load trained meta-model
    predict_fn, feature_names = _cutils._load_model_generic(run_dir)
    
    # Load training dataset (contains the positions we should evaluate)
    if dataset_path.is_dir():
        # Scan multiple parquet files with IGNORING extra columns to avoid schema errors
        lf = pl.scan_parquet(str(dataset_path / "*.parquet"), missing_columns="insert", 
                           extra_columns="ignore")
    else:
        # Single parquet file with IGNORING extra columns to avoid schema errors
        lf = pl.scan_parquet(str(dataset_path), missing_columns="insert", 
                           extra_columns="ignore")
    
    # First, get available columns from the dataset
    try:
        sample_df = lf.select(pl.all()).limit(1).collect()
        available_dataset_cols = set(sample_df.columns)
    except Exception:
        # Fallback: try to read schema
        available_dataset_cols = set(lf.schema.keys())
    
    # Select required columns for evaluation
    base_required_cols = [
        "gene_id", "splice_type", "donor_score", "acceptor_score", "neither_score",
        "position", "chrom"  # strand is optional
    ]
    
    # Add strand if available
    if "strand" in available_dataset_cols:
        base_required_cols.append("strand")
    
    # Add positional mapping columns (for enhanced position-aware evaluation)
    positional_cols = [
        "true_position", "predicted_position", "gene_start", "gene_end", 
        "absolute_position", "distance_to_start", "distance_to_end"
    ]
    
    # Filter feature names to only include those actually available in the dataset
    available_feature_cols = [col for col in feature_names if col in available_dataset_cols]
    available_positional_cols = [col for col in positional_cols if col in available_dataset_cols]
    
    # Combine base required + available features + available positional columns
    required_cols = base_required_cols + available_feature_cols + available_positional_cols
    
    # Remove duplicates and filter to only available columns
    available_cols = []
    for col in required_cols:
        if col in available_dataset_cols and col not in available_cols:
            available_cols.append(col)
    
    if verbose:
        missing_features = set(feature_names) - set(available_feature_cols)
        if missing_features:
            print(f"[meta_splice_eval] Warning: {len(missing_features)} features missing from dataset")
            if verbose > 1:
                print(f"[meta_splice_eval] Missing features: {sorted(list(missing_features))[:10]}{'...' if len(missing_features) > 10 else ''}")
        
        # Report positional capabilities
        if available_positional_cols:
            print(f"[meta_splice_eval] ✓ Position-aware evaluation enabled with {len(available_positional_cols)} positional columns")
    
    # Load data using robust column selection
    from meta_spliceai.splice_engine.meta_models.training.classifier_utils import select_available_columns
    lf, missing_cols, existing_cols = select_available_columns(
        lf, available_cols, context_name="Meta Splice Performance Evaluation"
    )
    
    # Use memory-optimized loading with early sampling for large datasets
    try:
        from meta_spliceai.splice_engine.meta_models.training.memory_optimized_datasets import (
            estimate_dataset_size_efficiently,
            load_dataset_with_memory_management
        )
        
        # Estimate size to decide loading strategy
        estimated_rows, file_count = estimate_dataset_size_efficiently(dataset_path)
        
        if estimated_rows > 50_000:
            # For large datasets, use the memory-optimized loader directly
            effective_sample = sample if sample is not None else 25_000
            effective_sample = min(effective_sample, 25_000)  # Cap at 25k for memory safety
            
            if verbose:
                print(f"[meta_splice_eval] Large dataset detected ({estimated_rows:,} rows)")
                print(f"[meta_splice_eval] Using memory-optimized loader with {effective_sample:,} target samples")
            
            # For very large datasets, skip full loading and go directly to incremental evaluation
            # Create a dummy DataFrame to trigger incremental evaluation
            df = pl.DataFrame({
                "gene_id": ["dummy"], 
                "splice_type": [0],
                "dummy_flag": [True]  # Flag to indicate this is a placeholder
            })
            
            if verbose:
                print(f"[meta_splice_eval] Memory-optimized loading complete: {df.height:,} rows")
            
            # For large datasets with dummy DataFrame, go directly to incremental evaluation
            if "dummy_flag" in df.columns or df.height > effective_sample:
                if verbose:
                    print(f"[meta_splice_eval] Large dataset detected - using gene-preserving evaluation")
                    print(f"[meta_splice_eval] Will process genes incrementally to stay within memory limits")
                
                # Use incremental gene-based evaluation instead of sampling
                return _incremental_gene_evaluation(
                    df=df, 
                    dataset_path=dataset_path,
                    run_dir=run_dir,
                    predict_fn=predict_fn,
                    feature_names=feature_names,
                    available_feature_cols=available_feature_cols,
                    target_sample_size=effective_sample,
                    verbose=verbose,
                    out_tsv=out_tsv
                )
        else:
            # Standard loading for smaller datasets
            df = lf.collect()
            
            # Apply normal sampling if requested
            if sample is not None and df.height > sample:
                df = gene_wise_sample_dataframe(
                    df, target_positions=sample, gene_col="gene_id", 
                    random_state=42, verbose=verbose
                )
            
    except ImportError:
        if verbose:
            print(f"[meta_splice_eval] Memory optimization not available, using standard loading")
        df = lf.collect()
        
        # Apply sampling if needed
        if sample is not None and df.height > sample:
            df = gene_wise_sample_dataframe(
                df, target_positions=sample, gene_col="gene_id", 
                random_state=42, verbose=verbose
            )
            
    except Exception as e:
        if verbose:
            print(f"[meta_splice_eval] Memory optimization failed ({e}), using aggressive fallback sampling")
        
        # Last resort: try to collect a small sample directly
        try:
            # Use a very small sample to avoid OOM
            emergency_sample = min(10_000, sample if sample else 10_000)
            if verbose:
                print(f"[meta_splice_eval] Emergency fallback: sampling {emergency_sample:,} rows directly")
            
            # Try to use LazyFrame limit for emergency sampling
            df = lf.limit(emergency_sample * 10).collect(streaming=True)
            if df.height > emergency_sample:
                df = df.sample(n=emergency_sample, seed=42)
        except Exception as e2:
            if verbose:
                print(f"[meta_splice_eval] Emergency fallback failed ({e2}), evaluation will be skipped")
            raise RuntimeError(f"Cannot load dataset for evaluation due to memory constraints: {e}")
            
    
    if verbose:
        print(f"[meta_splice_eval] Processing {df.height:,} positions for evaluation")
    
    # Convert to pandas for easier processing
    df_pd = df.to_pandas()
    
    # Extract features (excluding leaky columns)
    from meta_spliceai.splice_engine.meta_models.builder.preprocessing import (
        LEAKAGE_COLUMNS, METADATA_COLUMNS
    )
    
    exclude_cols = set(LEAKAGE_COLUMNS + METADATA_COLUMNS + ["splice_type"])
    available_feature_cols = [col for col in available_feature_cols if col not in exclude_cols]
    
    if verbose:
        print(f"[meta_splice_eval] Using {len(available_feature_cols)} available features for meta-model")
        print(f"[meta_splice_eval] Meta-model expects {len(feature_names)} features total")
        excluded = set(feature_names) - set(available_feature_cols) 
        if excluded and verbose > 1:
            print(f"[meta_splice_eval] Excluded/missing features: {sorted(excluded)[:10]}{'...' if len(excluded) > 10 else ''}")
    
    # Create feature matrix with proper shape for meta-model
    # Start with available features
    X_available = df_pd[available_feature_cols].fillna(0).to_numpy(dtype=np.float32)
    
    # Create full feature matrix with zeros for missing features
    X = np.zeros((len(df_pd), len(feature_names)), dtype=np.float32)
    
    # Fill in available features at their correct positions
    for i, feat_name in enumerate(feature_names):
        if feat_name in available_feature_cols:
            col_idx = available_feature_cols.index(feat_name)
            X[:, i] = X_available[:, col_idx]
        # Missing features remain as zeros
    
    if verbose and X.shape[1] != len(feature_names):
        print(f"[meta_splice_eval] Warning: Feature matrix shape mismatch")
        print(f"  Expected: ({len(df_pd)}, {len(feature_names)})")
        print(f"  Got: {X.shape}")

    # Get base model predictions (from raw SpliceAI scores)
    donor_scores = df_pd["donor_score"].to_numpy()
    acceptor_scores = df_pd["acceptor_score"].to_numpy() 
    neither_scores = df_pd["neither_score"].to_numpy()
    
    # Base prediction: argmax of [neither, donor, acceptor]
    base_probs = np.column_stack([neither_scores, donor_scores, acceptor_scores])
    y_base = np.array(["neither", "donor", "acceptor"])[np.argmax(base_probs, axis=1)]
    
    # Get meta-model predictions
    meta_probs = predict_fn(X)
    y_meta = np.array(["neither", "donor", "acceptor"])[np.argmax(meta_probs, axis=1)]
    
    # ENHANCED: Generate per-nucleotide probability tensors
    # Create donor_meta, acceptor_meta, neither_meta scores
    donor_meta = meta_probs[:, 1]  # Meta-model donor probabilities
    acceptor_meta = meta_probs[:, 2]  # Meta-model acceptor probabilities  
    neither_meta = meta_probs[:, 0]  # Meta-model neither probabilities
    
    # Add these to the dataframe for positional mapping
    df_pd["donor_meta"] = donor_meta
    df_pd["acceptor_meta"] = acceptor_meta  
    df_pd["neither_meta"] = neither_meta
    df_pd["y_base_pred"] = y_base
    df_pd["y_meta_pred"] = y_meta
    
    # ENHANCED: Positional mapping capability
    positional_mapping_available = all(col in df_pd.columns for col in 
                                     ["gene_id", "position", "gene_start", "gene_end"])
    
    if positional_mapping_available and verbose:
        print(f"[meta_splice_eval] ✓ Full positional mapping enabled")
        print(f"  - Can map predictions back to genomic coordinates")
        print(f"  - Can generate per-nucleotide score tensors")
        print(f"  - Available columns: gene_id, position, gene_start, gene_end")
        
        # Optional: Check for additional positional columns
        extra_pos_cols = [col for col in ["true_position", "predicted_position", 
                         "absolute_position", "distance_to_start", "distance_to_end"] 
                         if col in df_pd.columns]
        if extra_pos_cols:
            print(f"  - Additional positional data: {', '.join(extra_pos_cols)}")
    
    # Calculate performance metrics
    from sklearn.metrics import classification_report, confusion_matrix, f1_score
    
    # Overall accuracy
    base_acc = (y_base == df_pd["splice_type"].to_numpy()).mean()
    meta_acc = (y_meta == df_pd["splice_type"].to_numpy()).mean()
    
    # F1 scores
    base_f1 = f1_score(df_pd["splice_type"].to_numpy(), y_base, average='macro', zero_division=0)
    meta_f1 = f1_score(df_pd["splice_type"].to_numpy(), y_meta, average='macro', zero_division=0)
    
    # Per-class F1 scores
    from sklearn.metrics import f1_score
    classes = ["neither", "donor", "acceptor"]
    base_f1_per_class = {}
    meta_f1_per_class = {}
    
    for cls in classes:
        y_true_binary = (df_pd["splice_type"].to_numpy() == cls).astype(int)
        y_base_binary = (y_base == cls).astype(int) 
        y_meta_binary = (y_meta == cls).astype(int)
        
        base_f1_per_class[cls] = f1_score(y_true_binary, y_base_binary, zero_division=0)
        meta_f1_per_class[cls] = f1_score(y_true_binary, y_meta_binary, zero_division=0)
    
    # Calculate corrections and regressions
    base_correct = (y_base == df_pd["splice_type"].to_numpy())
    meta_correct = (y_meta == df_pd["splice_type"].to_numpy())
    
    corrections = ((~base_correct) & meta_correct).sum()  # Meta fixed base error
    regressions = (base_correct & (~meta_correct)).sum()  # Meta introduced error
    maintained = (base_correct & meta_correct).sum()     # Both correct
    both_wrong = ((~base_correct) & (~meta_correct)).sum() # Both wrong
    
    # Summary statistics
    results = {
        "total_positions": len(df_pd["splice_type"].to_numpy()),
        "base_accuracy": float(base_acc),
        "meta_accuracy": float(meta_acc),
        "accuracy_improvement": float(meta_acc - base_acc),
        "base_f1_macro": float(base_f1),
        "meta_f1_macro": float(meta_f1), 
        "f1_improvement": float(meta_f1 - base_f1),
        "corrections": int(corrections),
        "regressions": int(regressions),
        "maintained_correct": int(maintained),
        "both_wrong": int(both_wrong),
        "net_improvement": int(corrections - regressions),
        "improvement_rate": float(corrections / len(df_pd["splice_type"].to_numpy())) if len(df_pd["splice_type"].to_numpy()) > 0 else 0.0,
        "regression_rate": float(regressions / len(df_pd["splice_type"].to_numpy())) if len(df_pd["splice_type"].to_numpy()) > 0 else 0.0,
    }
    
    # Add per-class results
    for cls in classes:
        results[f"base_f1_{cls}"] = float(base_f1_per_class[cls])
        results[f"meta_f1_{cls}"] = float(meta_f1_per_class[cls])
        results[f"f1_improvement_{cls}"] = float(meta_f1_per_class[cls] - base_f1_per_class[cls])
    
    if verbose:
        print(f"\n[meta_splice_eval] 📊 EVALUATION RESULTS:")
        print(f"  Total positions evaluated: {results['total_positions']:,}")
        print(f"  Base model accuracy: {results['base_accuracy']:.3f}")
        print(f"  Meta model accuracy: {results['meta_accuracy']:.3f}")
        print(f"  Accuracy improvement: {results['accuracy_improvement']:+.3f}")
        print(f"  Base F1 (macro): {results['base_f1_macro']:.3f}")
        print(f"  Meta F1 (macro): {results['meta_f1_macro']:.3f}")
        print(f"  F1 improvement: {results['f1_improvement']:+.3f}")
        print(f"  Corrections (meta fixed base errors): {results['corrections']:,}")
        print(f"  Regressions (meta introduced errors): {results['regressions']:,}")
        print(f"  Net improvement: {results['net_improvement']:+,}")
        print(f"  Improvement rate: {results['improvement_rate']:.1%}")
        
        print(f"\n  Per-class F1 improvements:")
        for cls in classes:
            base_f1 = results[f"base_f1_{cls}"]
            meta_f1 = results[f"meta_f1_{cls}"]
            improvement = results[f"f1_improvement_{cls}"]
            print(f"    {cls}: {base_f1:.3f} → {meta_f1:.3f} ({improvement:+.3f})")
    
    # Save detailed results if requested
    if out_tsv is not None:
        # Normalize splice_type to ensure consistent string encoding before creating true_label
        # Apply fail-safe logic: if not donor or acceptor, then it's neither
        # This covers None (becomes "None"), "0", 0, "nan", etc.
        normalized_splice_type = df_pd["splice_type"].astype(str).apply(
            lambda x: x if x in ["donor", "acceptor"] else "neither"
        )
        
        # Create detailed results dataframe
        detailed_data = {
            "gene_id": df_pd["gene_id"],
            "position": df_pd["position"],
            "chrom": df_pd["chrom"],
            "true_label": normalized_splice_type.to_numpy(),
            "base_prediction": y_base,
        }
        
        # Add strand if available
        if "strand" in df_pd.columns:
            detailed_data["strand"] = df_pd["strand"]
        
        detailed_data.update({
            "meta_prediction": y_meta,
            "base_correct": base_correct,
            "meta_correct": meta_correct,
            "base_donor_score": donor_scores,
            "base_acceptor_score": acceptor_scores,
            "base_neither_score": neither_scores,
            "meta_donor_prob": meta_probs[:, 1],
            "meta_acceptor_prob": meta_probs[:, 2],
            "meta_neither_prob": meta_probs[:, 0],
        })
        
        detailed_df = pd.DataFrame(detailed_data)
        
        detailed_df.to_csv(out_tsv, sep='\t', index=False)
        if verbose:
            print(f"[meta_splice_eval] Saved detailed results to: {out_tsv}")
    
    # Save summary results
    summary_path = run_dir / "meta_evaluation_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    if verbose:
        print(f"[meta_splice_eval] Saved summary to: {summary_path}")
    
    return summary_path


def _incremental_gene_evaluation_argmax(
    df: pl.DataFrame,
    dataset_path: Path,
    run_dir: Path,
    predict_fn,
    feature_names: list,
    target_sample_size: int,
    verbose: bool,
    out_tsv: Path | None = None,
    donor_score_col: str = "donor_score",
    acceptor_score_col: str = "acceptor_score", 
    gene_col: str = "gene_id",
    label_col: str = "splice_type"
) -> Path:
    """
    Incremental argmax-based evaluation that processes genes in chunks.
    """
    import numpy as np
    import pandas as pd
    from pathlib import Path
    import json
    from sklearn.metrics import f1_score, classification_report
    
    if verbose:
        print(f"[incremental_argmax] Starting incremental argmax evaluation")
        print(f"[incremental_argmax] Target sample size: {target_sample_size:,}")
    
    # Get unique genes and estimate chunk size
    unique_genes = df.select(gene_col).unique().to_pandas()[gene_col].tolist()
    total_genes = len(unique_genes)
    
    # Calculate chunk size to stay within memory limits
    avg_positions_per_gene = df.height // total_genes
    target_genes_per_chunk = max(1, target_sample_size // (avg_positions_per_gene * 2))
    
    if verbose:
        print(f"[incremental_argmax] Total genes: {total_genes:,}")
        print(f"[incremental_argmax] Avg positions per gene: {avg_positions_per_gene:.0f}")
        print(f"[incremental_argmax] Target genes per chunk: {target_genes_per_chunk}")
    
    # Process genes in chunks
    all_results = []
    processed_positions = 0
    processed_genes = 0
    
    for chunk_start in range(0, total_genes, target_genes_per_chunk):
        chunk_end = min(chunk_start + target_genes_per_chunk, total_genes)
        chunk_genes = unique_genes[chunk_start:chunk_end]
        
        if verbose:
            print(f"[incremental_argmax] Processing chunk {chunk_start//target_genes_per_chunk + 1}/{(total_genes + target_genes_per_chunk - 1)//target_genes_per_chunk}")
            print(f"[incremental_argmax] Genes {chunk_start+1}-{chunk_end} ({len(chunk_genes)} genes)")
        
        # Filter to this chunk of genes
        chunk_df = df.filter(pl.col(gene_col).is_in(chunk_genes))
        chunk_positions = chunk_df.height
        
        if chunk_positions == 0:
            continue
            
        # Convert chunk to pandas for processing
        chunk_pd = chunk_df.to_pandas()
        
        # Verify required columns
        required_cols = [gene_col, label_col, donor_score_col, acceptor_score_col]
        missing = [col for col in required_cols if col not in chunk_pd.columns]
        if missing:
            if verbose:
                print(f"[incremental_argmax] Warning: Missing columns in chunk: {missing}")
            continue
        
        # Extract features for meta-model prediction
        available_features = [col for col in feature_names if col in chunk_pd.columns]
        missing_features = set(feature_names) - set(available_features)
        
        # Create feature matrix with proper data type handling
        X = np.zeros((len(chunk_pd), len(feature_names)), dtype=np.float32)
        for i, feat_name in enumerate(feature_names):
            if feat_name in available_features:
                try:
                    # Handle numeric columns
                    col_values = chunk_pd[feat_name].fillna(0)
                    if col_values.dtype == 'object':
                        # Convert categorical/string columns to numeric
                        col_values = pd.to_numeric(col_values, errors='coerce').fillna(0)
                    X[:, i] = col_values.values.astype(np.float32)
                except (ValueError, TypeError) as e:
                    if verbose > 1:
                        print(f"[incremental_argmax] Warning: Could not convert feature '{feat_name}' to numeric, using zeros: {e}")
                    # Leave as zeros for problematic columns
        
        # Get base model predictions using argmax
        base_donor = chunk_pd[donor_score_col].values
        base_acceptor = chunk_pd[acceptor_score_col].values
        base_neither = 1 - (base_donor + base_acceptor)
        base_probs = np.column_stack([base_neither, base_donor, base_acceptor])
        base_probs = np.clip(base_probs, 0, 1)
        base_pred = np.argmax(base_probs, axis=1)
        
        # Get meta-model predictions using argmax
        meta_probs = predict_fn(X)
        meta_pred = np.argmax(meta_probs, axis=1)
        
        # Get true labels
        label_map = {"neither": 0, "donor": 1, "acceptor": 2}
        true_labels = chunk_pd[label_col].map(label_map).values
        
        # Store chunk results
        chunk_results = {
            "chunk_id": chunk_start // target_genes_per_chunk,
            "genes": chunk_genes,
            "positions": chunk_positions,
            "base_pred": base_pred,
            "meta_pred": meta_pred,
            "true_labels": true_labels,
            "gene_ids": chunk_pd[gene_col].values
        }
        
        all_results.append(chunk_results)
        processed_positions += chunk_positions
        processed_genes += len(chunk_genes)
        
        if verbose:
            chunk_base_acc = (base_pred == true_labels).mean()
            chunk_meta_acc = (meta_pred == true_labels).mean()
            print(f"[incremental_argmax] Chunk {chunk_results['chunk_id']+1} complete: {chunk_positions:,} positions")
            print(f"[incremental_argmax] Chunk accuracy: Base={chunk_base_acc:.3f}, Meta={chunk_meta_acc:.3f}")
        
        # Stop if we've processed enough positions
        if processed_positions >= target_sample_size:
            if verbose:
                print(f"[incremental_argmax] Reached target sample size, stopping early")
            break
    
    # Aggregate results
    if verbose:
        print(f"[incremental_argmax] Aggregating results from {len(all_results)} chunks")
        print(f"[incremental_argmax] Total processed: {processed_positions:,} positions, {processed_genes} genes")
    
    # Combine all predictions and labels
    all_base_pred = np.concatenate([r["base_pred"] for r in all_results])
    all_meta_pred = np.concatenate([r["meta_pred"] for r in all_results])
    all_true_labels = np.concatenate([r["true_labels"] for r in all_results])
    all_gene_ids = np.concatenate([r["gene_ids"] for r in all_results])
    
    # Calculate overall metrics
    base_accuracy = (all_base_pred == all_true_labels).mean()
    meta_accuracy = (all_meta_pred == all_true_labels).mean()
    
    # Calculate F1 scores
    base_f1_macro = f1_score(all_true_labels, all_base_pred, average='macro', zero_division=0)
    meta_f1_macro = f1_score(all_true_labels, all_meta_pred, average='macro', zero_division=0)
    
    # Per-class F1 scores
    base_report = classification_report(all_true_labels, all_base_pred, output_dict=True, zero_division=0)
    meta_report = classification_report(all_true_labels, all_meta_pred, output_dict=True, zero_division=0)
    
    # Count corrections and regressions
    base_correct = all_base_pred == all_true_labels
    meta_correct = all_meta_pred == all_true_labels
    corrections = np.sum(~base_correct & meta_correct)
    regressions = np.sum(base_correct & ~meta_correct)
    
    # Create evaluation summary
    evaluation_summary = {
        "total_positions": processed_positions,
        "total_genes": processed_genes,
        "base_accuracy": float(base_accuracy),
        "meta_accuracy": float(meta_accuracy),
        "accuracy_improvement": float(meta_accuracy - base_accuracy),
        "base_f1_macro": float(base_f1_macro),
        "meta_f1_macro": float(meta_f1_macro),
        "f1_improvement": float(meta_f1_macro - base_f1_macro),
        "corrections": int(corrections),
        "regressions": int(regressions),
        "net_improvement": int(corrections - regressions),
        "improvement_rate": float((corrections - regressions) / processed_positions),
        "evaluation_method": "argmax_incremental"
    }
    
    # Save evaluation summary
    summary_path = run_dir / "meta_evaluation_argmax_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(evaluation_summary, f, indent=2)
    
    # Save detailed results if requested
    if out_tsv:
        detailed_results = pd.DataFrame({
            gene_col: all_gene_ids,
            "true_label": all_true_labels,
            "base_pred": all_base_pred,
            "meta_pred": all_meta_pred,
            "base_correct": base_correct,
            "meta_correct": meta_correct
        })
        detailed_results.to_csv(out_tsv, sep='\t', index=False)
        
        if verbose:
            print(f"[incremental_argmax] Detailed results saved to: {out_tsv}")
    
    if verbose:
        print(f"[incremental_argmax] 📊 ARGMAX EVALUATION COMPLETE:")
        print(f"  Processed: {processed_positions:,} positions from {processed_genes} genes")
        print(f"  Base accuracy: {base_accuracy:.3f}")
        print(f"  Meta accuracy: {meta_accuracy:.3f}")
        print(f"  Accuracy improvement: {meta_accuracy - base_accuracy:+.3f}")
        print(f"  Base F1 (macro): {base_f1_macro:.3f}")
        print(f"  Meta F1 (macro): {meta_f1_macro:.3f}")
        print(f"  F1 improvement: {meta_f1_macro - base_f1_macro:+.3f}")
        print(f"  Corrections: {corrections:,}, Regressions: {regressions:,}")
        print(f"  Net improvement: {corrections - regressions:+,} ({(corrections - regressions) / processed_positions:.1%})")
    
    return summary_path


def meta_splice_performance_argmax(
    dataset_path: str | Path,
    run_dir: str | Path,
    *,
    sample: int | None = None,
    out_tsv: str | Path | None = None,
    verbose: int = 1,
    donor_score_col: str = "donor_score",
    acceptor_score_col: str = "acceptor_score",
    gene_col: str = "gene_id",
    label_col: str = "splice_type",
) -> Path:
    """
    Argmax-based meta-model evaluation that aligns with CV loop methodology.
    
    Uses argmax on probability distributions instead of thresholds, which works
    better with overconfident/poorly-calibrated meta-model outputs.
    
    Parameters
    ----------
    dataset_path : str | Path
        Path to training dataset  
    run_dir : str | Path
        Directory containing trained meta-model
    sample : int | None
        Optional sample size for faster evaluation
    out_tsv : str | Path | None
        Output path for detailed results
    verbose : int
        Verbosity level
    donor_score_col : str
        Column name for base donor scores
    acceptor_score_col : str  
        Column name for base acceptor scores
    gene_col : str
        Column name for gene IDs
    label_col : str
        Column name for true labels
        
    Returns
    -------
    Path
        Path to evaluation summary file
    """
    
    import pandas as pd
    import numpy as np
    import json
    from pathlib import Path
    
    dataset_path = Path(dataset_path)
    run_dir = Path(run_dir)
    
    if verbose:
        print(f"\n[meta_splice_eval] ARGMAX EVALUATION: Using argmax instead of thresholds")
        print(f"[meta_splice_eval] Dataset: {dataset_path}")
        print(f"[meta_splice_eval] Model: {run_dir}")
    
    # Load dataset with robust column handling
    if dataset_path.is_dir():
        # Handle directory of parquet files with proper schema handling
        import polars as pl
        lf = pl.scan_parquet(str(dataset_path / "*.parquet"), missing_columns="insert", 
                           extra_columns="ignore")
        
        # Use robust column selection to avoid schema errors
        from meta_spliceai.splice_engine.meta_models.training.classifier_utils import select_available_columns
        
        # Get all columns first to check what's available
        try:
            sample_df = lf.select(pl.all()).limit(1).collect()
            available_cols = list(sample_df.columns)
        except Exception:
            available_cols = list(lf.schema.keys())
        
        lf, missing_cols, existing_cols = select_available_columns(
            lf, available_cols, context_name="Meta Splice Performance Argmax"
        )
        
        # Use memory-optimized loading for large datasets with incremental evaluation
        try:
            from meta_spliceai.splice_engine.meta_models.training.memory_optimized_datasets import (
                estimate_dataset_size_efficiently,
                load_dataset_with_memory_management
            )
            
            # Estimate size to decide loading strategy
            estimated_rows, file_count = estimate_dataset_size_efficiently(dataset_path)
            
            if estimated_rows > 50_000:
                # For very large datasets, use incremental evaluation approach
                effective_sample = sample if sample is not None else 25_000
                effective_sample = min(effective_sample, 25_000)  # Cap at 25k for memory safety
                
                if verbose:
                    print(f"[meta_splice_eval] Large dataset detected ({estimated_rows:,} rows)")
                    print(f"[meta_splice_eval] Using incremental gene-based evaluation with {effective_sample:,} target samples")
                
                # Load trained meta-model
                predict_fn, feature_names = _cutils._load_model_generic(run_dir)
                
                # For very large datasets, skip full loading and go directly to incremental evaluation
                # Create a dummy DataFrame to trigger incremental evaluation
                df_pl = pl.DataFrame({
                    "gene_id": ["dummy"], 
                    "splice_type": [0],
                    "dummy_flag": [True]  # Flag to indicate this is a placeholder
                })
                
                # Use incremental evaluation instead of loading full dataset into pandas
                return _incremental_gene_evaluation_argmax(
                    df=df_pl,
                    dataset_path=dataset_path,
                    run_dir=run_dir,
                    predict_fn=predict_fn,
                    feature_names=feature_names,
                    target_sample_size=effective_sample,
                    verbose=verbose,
                    out_tsv=out_tsv,
                    donor_score_col=donor_score_col,
                    acceptor_score_col=acceptor_score_col,
                    gene_col=gene_col,
                    label_col=label_col
                )
            else:
                # Standard loading for smaller datasets
                df_pl = lf.collect()
                df = df_pl.to_pandas()
                
        except ImportError as e:
            print(f"[ERROR] Memory optimization module not available: {e}")
            print(f"[ERROR] Cannot evaluate large dataset without memory optimization")
            raise ImportError(f"Memory optimization required for large datasets: {e}")
        except Exception as e:
            print(f"[ERROR] Memory optimization failed: {e}")
            print(f"[ERROR] Cannot evaluate large dataset - memory optimization required")
            raise RuntimeError(f"Memory-optimized evaluation failed for large dataset: {e}")
    else:
        # Single parquet file
        df = pd.read_parquet(dataset_path)
    
    # Apply aggressive sampling for large datasets to prevent OOM
    if len(df) > 100_000:
        # For very large datasets, always sample to prevent OOM
        effective_sample = sample if sample is not None else 50_000
        effective_sample = min(effective_sample, 100_000)  # Cap at 100k for memory safety
        
        if verbose:
            print(f"[meta_splice_eval] Large dataset detected ({len(df):,} rows)")
            print(f"[meta_splice_eval] Applying sampling to {effective_sample:,} positions for memory safety")
        
        df = gene_wise_sample_dataframe(
            df, target_positions=effective_sample, gene_col="gene_id", 
            random_state=42, verbose=verbose
        )
    elif sample is not None and len(df) > sample:
        # Normal sampling for smaller datasets
        df = gene_wise_sample_dataframe(
            df, target_positions=sample, gene_col="gene_id", 
            random_state=42, verbose=verbose
        )
    
    # CRITICAL: Reset index after sampling to avoid IndexError
    df = df.reset_index(drop=True)
    
    if verbose:
        print(f"[meta_splice_eval] Processing {len(df):,} positions for evaluation")
    
    # Verify required columns
    required_cols = [gene_col, label_col, donor_score_col, acceptor_score_col]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")
    
    # Load meta-model and get predictions
    predict_fn, feature_names = _cutils._load_model_generic(run_dir)
    
    # Check feature availability
    available_features = [col for col in feature_names if col in df.columns]
    missing_features = set(feature_names) - set(available_features)
    
    if verbose:
        print(f"[meta_splice_eval] Using {len(available_features)}/{len(feature_names)} features")
        if missing_features and verbose > 1:
            print(f"[meta_splice_eval] Missing features (filled with 0): {len(missing_features)}")
    
    # Create feature matrix
    X = np.zeros((len(df), len(feature_names)), dtype=np.float32)
    for i, feat_name in enumerate(feature_names):
        if feat_name in available_features:
            col_data = df[feat_name]
            # Handle non-numeric columns (like chromosome)
            if not pd.api.types.is_numeric_dtype(col_data):
                if verbose and 'chrom' in feat_name.lower():
                    print(f"[meta_splice_eval] Encoding non-numeric feature: {feat_name}")
                # Map unique values to integers
                unique_vals = col_data.dropna().unique()
                val_map = {val: float(idx) for idx, val in enumerate(sorted(unique_vals))}
                X[:, i] = col_data.map(val_map).fillna(-1.0).astype(np.float32)
            else:
                X[:, i] = col_data.fillna(0.0).astype(np.float32)
    
    # Get base model predictions (argmax of raw SpliceAI scores)
    if "neither_score" in df.columns:
        base_probs = np.column_stack([
            df["neither_score"].values,
            df[donor_score_col].values, 
            df[acceptor_score_col].values
        ])
    else:
        # If neither_score missing, derive it
        donor_scores = df[donor_score_col].values
        acceptor_scores = df[acceptor_score_col].values
        neither_scores = np.clip(1.0 - donor_scores - acceptor_scores, 0.0, 1.0)
        base_probs = np.column_stack([neither_scores, donor_scores, acceptor_scores])
    
    # Base predictions using argmax
    base_pred_idx = np.argmax(base_probs, axis=1)
    base_pred_labels = np.array(["neither", "donor", "acceptor"])[base_pred_idx]
    
    # Get meta-model predictions
    meta_probs = predict_fn(X)
    meta_pred_idx = np.argmax(meta_probs, axis=1) 
    meta_pred_labels = np.array(["neither", "donor", "acceptor"])[meta_pred_idx]
    
    # True labels
    true_labels = df[label_col].astype(str).values
    
    # Handle label normalization (convert "0" to "neither" if needed)
    true_labels = np.where(true_labels == "0", "neither", true_labels)
    
    if verbose:
        print(f"\n[meta_splice_eval] 📊 Prediction distribution summary:")
        print("True labels:")
        unique_true, counts_true = np.unique(true_labels, return_counts=True)
        for label, count in zip(unique_true, counts_true):
            print(f"  {label}: {count:,} ({count/len(true_labels)*100:.1f}%)")
        
        print("Base predictions:")
        unique_base, counts_base = np.unique(base_pred_labels, return_counts=True)
        for label, count in zip(unique_base, counts_base):
            print(f"  {label}: {count:,} ({count/len(base_pred_labels)*100:.1f}%)")
        
        print("Meta predictions:")
        unique_meta, counts_meta = np.unique(meta_pred_labels, return_counts=True)
        for label, count in zip(unique_meta, counts_meta):
            print(f"  {label}: {count:,} ({count/len(meta_pred_labels)*100:.1f}%)")
    
    # Calculate overall metrics
    from sklearn.metrics import accuracy_score, f1_score, classification_report
    
    base_accuracy = accuracy_score(true_labels, base_pred_labels)
    meta_accuracy = accuracy_score(true_labels, meta_pred_labels)
    
    base_f1_macro = f1_score(true_labels, base_pred_labels, average='macro', zero_division=0)
    meta_f1_macro = f1_score(true_labels, meta_pred_labels, average='macro', zero_division=0)
    
    # Per-class F1 scores
    classes = ["neither", "donor", "acceptor"]
    base_f1_per_class = {}
    meta_f1_per_class = {}
    
    for cls in classes:
        # Create binary labels for this class
        y_true_binary = (true_labels == cls).astype(int)
        y_base_binary = (base_pred_labels == cls).astype(int)
        y_meta_binary = (meta_pred_labels == cls).astype(int)
        
        base_f1_per_class[cls] = f1_score(y_true_binary, y_base_binary, zero_division=0)
        meta_f1_per_class[cls] = f1_score(y_true_binary, y_meta_binary, zero_division=0)
    
    # Calculate error analysis
    base_correct = (base_pred_labels == true_labels)
    meta_correct = (meta_pred_labels == true_labels)
    
    corrections = ((~base_correct) & meta_correct).sum()  # Meta fixed base error
    regressions = (base_correct & (~meta_correct)).sum()  # Meta introduced error
    maintained = (base_correct & meta_correct).sum()     # Both correct
    both_wrong = ((~base_correct) & (~meta_correct)).sum() # Both wrong
    
    # Per-gene analysis
    gene_results = []
    for gene_id, gene_df in df.groupby(gene_col):
        gene_idx = gene_df.index
        
        gene_true = true_labels[gene_idx]
        gene_base = base_pred_labels[gene_idx]
        gene_meta = meta_pred_labels[gene_idx]
        
        # Per-gene metrics for each site type
        for site_type in ["donor", "acceptor"]:
            # True positives, false positives, false negatives for base and meta
            true_mask = (gene_true == site_type)
            
            base_pred_mask = (gene_base == site_type)
            meta_pred_mask = (gene_meta == site_type)
            
            # Base model counts
            tp_base = (base_pred_mask & true_mask).sum()
            fp_base = (base_pred_mask & ~true_mask).sum()
            fn_base = (~base_pred_mask & true_mask).sum()
            
            # Meta model counts  
            tp_meta = (meta_pred_mask & true_mask).sum()
            fp_meta = (meta_pred_mask & ~true_mask).sum()
            fn_meta = (~meta_pred_mask & true_mask).sum()
            
            # Calculate metrics
            base_prec = tp_base / (tp_base + fp_base) if (tp_base + fp_base) > 0 else 0.0
            base_rec = tp_base / (tp_base + fn_base) if (tp_base + fn_base) > 0 else 0.0
            base_f1 = 2 * base_prec * base_rec / (base_prec + base_rec) if (base_prec + base_rec) > 0 else 0.0
            
            meta_prec = tp_meta / (tp_meta + fp_meta) if (tp_meta + fp_meta) > 0 else 0.0
            meta_rec = tp_meta / (tp_meta + fn_meta) if (tp_meta + fn_meta) > 0 else 0.0
            meta_f1 = 2 * meta_prec * meta_rec / (meta_prec + meta_rec) if (meta_prec + meta_rec) > 0 else 0.0
            
            gene_results.append({
                gene_col: gene_id,
                "site_type": site_type,
                "TP_base": tp_base,
                "FP_base": fp_base, 
                "FN_base": fn_base,
                "TP_meta": tp_meta,
                "FP_meta": fp_meta,
                "FN_meta": fn_meta,
                "precision_base": round(base_prec, 3),
                "recall_base": round(base_rec, 3),
                "f1_base": round(base_f1, 3),
                "precision_meta": round(meta_prec, 3),
                "recall_meta": round(meta_rec, 3),
                "f1_meta": round(meta_f1, 3),
                "f1_delta": round(meta_f1 - base_f1, 3),
                "tp_delta": tp_meta - tp_base,
                "fp_delta": fp_base - fp_meta,  # Positive = FP reduction
                "fn_delta": fn_base - fn_meta,  # Positive = FN reduction
            })
    
    # Create results DataFrame
    results_df = pd.DataFrame(gene_results)
    
    # Summary statistics
    summary = {
        "total_positions": len(true_labels),
        "base_accuracy": float(base_accuracy),
        "meta_accuracy": float(meta_accuracy),
        "accuracy_improvement": float(meta_accuracy - base_accuracy),
        "base_f1_macro": float(base_f1_macro),
        "meta_f1_macro": float(meta_f1_macro),
        "f1_improvement": float(meta_f1_macro - base_f1_macro),
        "corrections": int(corrections),
        "regressions": int(regressions),
        "maintained_correct": int(maintained),
        "both_wrong": int(both_wrong),
        "net_improvement": int(corrections - regressions),
        "improvement_rate": float(corrections / len(true_labels)),
        "regression_rate": float(regressions / len(true_labels)),
    }
    
    # Add per-class F1 scores
    for cls in classes:
        summary[f"base_f1_{cls}"] = float(base_f1_per_class[cls])
        summary[f"meta_f1_{cls}"] = float(meta_f1_per_class[cls])
        summary[f"f1_improvement_{cls}"] = float(meta_f1_per_class[cls] - base_f1_per_class[cls])
    
    # Add aggregate gene-level statistics
    if len(results_df) > 0:
        summary["total_genes"] = results_df[gene_col].nunique()
        
        # TP/FP/FN deltas
        summary["total_tp_gained"] = int(results_df["tp_delta"].sum())
        summary["total_fp_reduced"] = int(results_df["fp_delta"].sum())
        summary["total_fn_reduced"] = int(results_df["fn_delta"].sum())
        
        # Gene improvement counts
        summary["genes_f1_improved"] = int((results_df["f1_delta"] > 0).sum())
        summary["genes_f1_worsened"] = int((results_df["f1_delta"] < 0).sum())
        summary["genes_f1_unchanged"] = int((results_df["f1_delta"] == 0).sum())
        
        # Mean F1 improvements
        summary["mean_f1_improvement"] = float(results_df["f1_delta"].mean())
        summary["median_f1_improvement"] = float(results_df["f1_delta"].median())
    
    if verbose:
        print(f"\n[meta_splice_eval] 📊 EVALUATION RESULTS (ARGMAX-BASED):")
        print(f"  Total positions: {summary['total_positions']:,}")
        print(f"  Base accuracy: {summary['base_accuracy']:.3f}")
        print(f"  Meta accuracy: {summary['meta_accuracy']:.3f}")
        print(f"  Accuracy improvement: {summary['accuracy_improvement']:+.3f}")
        print(f"  Base F1 (macro): {summary['base_f1_macro']:.3f}")
        print(f"  Meta F1 (macro): {summary['meta_f1_macro']:.3f}")
        print(f"  F1 improvement: {summary['f1_improvement']:+.3f}")
        
        print(f"\n  Error correction analysis:")
        print(f"  Corrections (meta fixed base errors): {summary['corrections']:,}")
        print(f"  Regressions (meta introduced errors): {summary['regressions']:,}")
        print(f"  Net improvement: {summary['net_improvement']:+,}")
        print(f"  Improvement rate: {summary['improvement_rate']:.1%}")
        
        print(f"\n  Per-class F1 improvements:")
        for cls in classes:
            base_f1 = summary[f"base_f1_{cls}"]
            meta_f1 = summary[f"meta_f1_{cls}"]
            improvement = summary[f"f1_improvement_{cls}"]
            print(f"    {cls}: {base_f1:.3f} → {meta_f1:.3f} ({improvement:+.3f})")
        
        if "total_genes" in summary:
            print(f"\n  Gene-level analysis:")
            print(f"  Total genes: {summary['total_genes']:,}")
            print(f"  TP gained: {summary['total_tp_gained']:+,}")
            print(f"  FP reduced: {summary['total_fp_reduced']:+,}")
            print(f"  FN reduced: {summary['total_fn_reduced']:+,}")
            print(f"  Genes improved: {summary['genes_f1_improved']:,}")
            print(f"  Genes worsened: {summary['genes_f1_worsened']:,}")
            print(f"  Mean F1 improvement: {summary['mean_f1_improvement']:+.3f}")
    
    # Save detailed results
    if out_tsv is not None:
        results_df.to_csv(out_tsv, sep='\t', index=False)
        if verbose:
            print(f"[meta_splice_eval] Saved gene-level results to: {out_tsv}")
    
    # Save summary
    summary_path = run_dir / "meta_evaluation_argmax_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    if verbose:
        print(f"[meta_splice_eval] Saved summary to: {summary_path}")
        
        # Show sample results
        if len(results_df) > 0:
            print(f"\n[meta_splice_eval] Sample gene-level results (first 10 rows):")
            display_cols = [gene_col, "site_type", "f1_base", "f1_meta", "f1_delta", "tp_delta", "fp_delta", "fn_delta"]
            sample_df = results_df[display_cols].head(10)
            print(sample_df.to_string(index=False, float_format=lambda x: f"{x:+.3f}" if abs(x) < 10 else f"{x:+.0f}"))
    
    return summary_path


def generate_per_nucleotide_meta_scores(
    dataset_path: str | Path,
    run_dir: str | Path,
    gene_manifest_path: Optional[str | Path] = None,
    *,
    sample: int | None = None,
    output_format: str = "parquet",
    verbose: int = 1,
) -> Path:
    """
    Generate per-nucleotide meta-model probability tensors for full genomic evaluation.
    
    This function creates donor_meta, acceptor_meta, neither_meta scores for all positions
    in the training dataset, enabling direct comparison with base model scores and 
    position-aware splice site prediction.
    
    Key capabilities:
    - Maps meta-model predictions back to genomic coordinates
    - Generates score tensors compatible with splice_inference_workflow.py
    - Enables per-nucleotide evaluation beyond training positions
    - Supports gene-level aggregation via gene manifest
    
    Parameters
    ----------
    dataset_path : str | Path
        Path to training dataset (Parquet files)
    run_dir : str | Path  
        Directory containing trained meta-model artifacts
    gene_manifest_path : Optional[str | Path]
        Path to gene manifest CSV for gene-level mapping
        If None, infers from dataset_path/gene_manifest.csv
    sample : int | None
        Optional: sample this many positions for faster processing
    output_format : str
        Output format: "parquet" or "tsv"
    verbose : int
        Verbosity level
        
    Returns
    -------
    Path
        Path to generated score tensor file
        
    Examples
    --------
    >>> # Generate meta-scores for full evaluation
    >>> score_path = generate_per_nucleotide_meta_scores(
    ...     dataset_path="train_pc_1000_3mers/master",
    ...     run_dir="results/gene_cv_run_3",
    ...     verbose=2
    ... )
    
    >>> # Load and use the scores
    >>> import pandas as pd
    >>> scores_df = pd.read_parquet(score_path)
    >>> # scores_df contains: gene_id, position, absolute_position, 
    >>> #                     donor_score, acceptor_score, neither_score (base),
    >>> #                     donor_meta, acceptor_meta, neither_meta (meta)
    """
    
    dataset_path = Path(dataset_path)
    run_dir = Path(run_dir)
    
    if gene_manifest_path is None:
        gene_manifest_path = dataset_path / "gene_manifest.csv"
    else:
        gene_manifest_path = Path(gene_manifest_path)
    
    if verbose:
        print(f"\n[per_nucleotide_meta] Generating per-nucleotide meta-model scores")
        print(f"[per_nucleotide_meta] Dataset: {dataset_path}")
        print(f"[per_nucleotide_meta] Model: {run_dir}")
        print(f"[per_nucleotide_meta] Gene manifest: {gene_manifest_path}")
    
    # Load trained meta-model
    predict_fn, feature_names = _cutils._load_model_generic(run_dir)
    
    # Load dataset with robust column handling
    if dataset_path.is_dir():
        lf = pl.scan_parquet(str(dataset_path / "*.parquet"), missing_columns="insert", 
                           extra_columns="ignore")
    else:
        lf = pl.scan_parquet(str(dataset_path), missing_columns="insert", 
                           extra_columns="ignore")
    
    # Get all required columns for scoring and mapping
    required_cols = [
        # Essential for scoring
        "gene_id", "splice_type", "donor_score", "acceptor_score", "neither_score",
        # Essential for positional mapping 
        "position", "strand", "chrom",
        # Optional but valuable for enhanced mapping
        "true_position", "predicted_position", "gene_start", "gene_end", 
        "absolute_position", "distance_to_start", "distance_to_end"
    ]
    
    # Add feature columns (avoid duplicates)
    for feat_name in feature_names:
        if feat_name not in required_cols:
            required_cols.append(feat_name)
    
    # Use robust column selection
    from meta_spliceai.splice_engine.meta_models.training.classifier_utils import select_available_columns
    lf, missing_cols, existing_cols = select_available_columns(
        lf, required_cols, context_name="Per-Nucleotide Meta Score Generation"
    )
    
    if verbose:
        # Debug: Check for potential duplicate columns
        duplicate_cols = [col for col in required_cols if required_cols.count(col) > 1]
        if duplicate_cols:
            print(f"[per_nucleotide_meta] Warning: Duplicate columns detected: {duplicate_cols}")
            # Remove duplicates while preserving order
            seen = set()
            unique_required_cols = []
            for col in required_cols:
                if col not in seen:
                    seen.add(col)
                    unique_required_cols.append(col)
            required_cols = unique_required_cols
            print(f"[per_nucleotide_meta] Removed duplicates, using {len(required_cols)} unique columns")
    
    # Load data with error handling
    try:
        df = lf.collect()
    except Exception as e:
        if "duplicate" in str(e).lower():
            print(f"[per_nucleotide_meta] Duplicate column error: {e}")
            print(f"[per_nucleotide_meta] Attempting to resolve by selecting only essential columns...")
            
            # Fallback: select only essential columns
            essential_cols = ["gene_id", "splice_type", "donor_score", "acceptor_score", "neither_score",
                             "position", "chrom"]
            
            # Add strand if available in schema
            try:
                schema_cols = set(pl.scan_parquet(str(dataset_path / "*.parquet")).schema.keys())
                if "strand" in schema_cols:
                    essential_cols.append("strand")
            except Exception:
                pass
            lf_fallback = pl.scan_parquet(str(dataset_path / "*.parquet"), missing_columns="insert", 
                                        extra_columns="ignore").select(essential_cols)
            df = lf_fallback.collect()
            print(f"[per_nucleotide_meta] Fallback successful with {len(essential_cols)} essential columns")
        else:
            raise e
    
    if verbose:
        print(f"[per_nucleotide_meta] Loaded {df.height:,} positions from training dataset")
        
        # Check positional mapping capabilities
        pos_mapping_cols = ["gene_id", "position", "gene_start", "gene_end", "absolute_position"]
        available_pos_cols = [col for col in pos_mapping_cols if col in df.columns]
        print(f"[per_nucleotide_meta] Positional mapping columns: {len(available_pos_cols)}/{len(pos_mapping_cols)}")
    
    # Sample if requested - use reusable gene-wise sampling utility
    if sample is not None and df.height > sample:
        df = gene_wise_sample_dataframe(
            df, target_positions=sample, gene_col="gene_id", 
            random_state=42, verbose=verbose
        )
    
    # Convert to pandas and prepare features
    df_pd = df.to_pandas()
    
    # Prepare feature matrix (same logic as meta_splice_performance_correct)
    from meta_spliceai.splice_engine.meta_models.builder.preprocessing import (
        LEAKAGE_COLUMNS, METADATA_COLUMNS
    )
    
    exclude_cols = set(LEAKAGE_COLUMNS + METADATA_COLUMNS + ["splice_type"])
    available_feature_cols = [col for col in feature_names if col in df_pd.columns and col not in exclude_cols]
    
    # Create feature matrix
    X_available = df_pd[available_feature_cols].fillna(0).to_numpy(dtype=np.float32)
    X = np.zeros((len(df_pd), len(feature_names)), dtype=np.float32)
    
    # Fill in available features at correct positions
    for i, feat_name in enumerate(feature_names):
        if feat_name in available_feature_cols:
            col_idx = available_feature_cols.index(feat_name)
            X[:, i] = X_available[:, col_idx]
    
    if verbose:
        print(f"[per_nucleotide_meta] Feature matrix: {X.shape} (using {len(available_feature_cols)}/{len(feature_names)} features)")
    
    # Generate meta-model predictions
    meta_probs = predict_fn(X)
    
    # Create output dataframe with essential columns
    output_cols = [
        "gene_id", "position", "splice_type", "chrom",
        # Base model scores  
        "donor_score", "acceptor_score", "neither_score",
        # Meta model scores (NEW!)
        "donor_meta", "acceptor_meta", "neither_meta"
    ]
    
    # Add strand to output columns if available
    if "strand" in df_pd.columns:
        output_cols.insert(4, "strand")  # Insert after chrom
    
    # Select base columns for result dataframe
    base_cols = ["gene_id", "position", "splice_type", "chrom", "donor_score", "acceptor_score", "neither_score"]
    if "strand" in df_pd.columns:
        base_cols.insert(4, "strand")  # Insert after chrom
    
    result_df = df_pd[base_cols].copy()
    
    # Add meta-model scores
    result_df["donor_meta"] = meta_probs[:, 1]
    result_df["acceptor_meta"] = meta_probs[:, 2] 
    result_df["neither_meta"] = meta_probs[:, 0]
    
    # Add positional mapping columns if available
    pos_cols = ["gene_start", "gene_end", "absolute_position", "true_position", 
                "predicted_position", "distance_to_start", "distance_to_end"]
    for col in pos_cols:
        if col in df_pd.columns:
            result_df[col] = df_pd[col]
            if col not in output_cols:
                output_cols.append(col)
    
    # Reorder columns
    result_df = result_df[output_cols]
    
    # Save results
    if output_format.lower() == "parquet":
        output_path = run_dir / "per_nucleotide_meta_scores.parquet"
        result_df.to_parquet(output_path, index=False)
    else:
        output_path = run_dir / "per_nucleotide_meta_scores.tsv"
        result_df.to_csv(output_path, sep='\t', index=False)
    
    if verbose:
        print(f"[per_nucleotide_meta] Saved {len(result_df):,} scored positions to: {output_path}")
        print(f"[per_nucleotide_meta] Output columns: {list(result_df.columns)}")
        
        # Show score distributions
        print(f"\n[per_nucleotide_meta] Score distributions:")
        for score_type in ["donor", "acceptor", "neither"]:
            base_col = f"{score_type}_score"
            meta_col = f"{score_type}_meta"
            if base_col in result_df.columns and meta_col in result_df.columns:
                base_mean = result_df[base_col].mean()
                meta_mean = result_df[meta_col].mean()
                print(f"  {score_type}: base={base_mean:.3f}, meta={meta_mean:.3f}")
    
    return output_path


def enhanced_generate_per_nucleotide_meta_scores(
    model_path: str | Path,
    target_genes: List[str],
    *,
    training_dataset_path: Optional[str | Path] = None,
    output_dir: str | Path = "enhanced_meta_scores",
    flanking_size: int = 2000,
    use_training_positions: bool = True,
    chunk_size: int = 1000,
    cleanup_intermediates: bool = True,
    verbose: int = 1,
    **inference_kwargs
) -> Dict[str, Path]:
    """
    Generate COMPLETE per-nucleotide meta-model scores for full genomic evaluation.
    
        This enhanced version addresses the fundamental limitation of the original function:
    it can generate meta-model predictions with COMPLETE COVERAGE CAPABILITY while using
    SELECTIVE FEATURIZATION for computational efficiency.

    Key Improvements:
    -----------------
    1. **Complete Coverage Capability**: Can predict at every nucleotide in principle
    2. **Selective Featurization**: Only generates features for uncertain positions
    3. **Base Model Reuse**: Directly uses confident base model predictions 
    4. **Hybrid Prediction System**: Seamlessly combines base + meta predictions
    5. **Memory Efficient**: Avoids unnecessarily large feature matrices

    How It Works:
    -------------
    1. **Selective Analysis**: Generate base model predictions for all positions
    2. **Uncertainty Detection**: Identify positions where base model is uncertain
    3. **Targeted Featurization**: Create feature matrices only for uncertain positions  
    4. **Meta-Model Recalibration**: Apply meta-model to uncertain positions only
    5. **Hybrid Assembly**: Combine confident base predictions with meta recalibrations
    
    Parameters
    ----------
    model_path : str | Path
        Path to trained meta-model (.json, .pkl, or directory with model artifacts)
    target_genes : List[str]
        List of gene IDs for which to generate complete meta-scores
    training_dataset_path : Optional[str | Path]
        Path to training dataset for reusing existing predictions (optional)
    output_dir : str | Path
        Directory for output files and intermediate results
    flanking_size : int, default=2000
        Context window size for sequence extraction (should match training)
    use_training_positions : bool, default=True
        Whether to reuse predictions from training dataset where available
    chunk_size : int, default=1000
        Number of genes to process in each chunk (memory management)
    cleanup_intermediates : bool, default=True
        Whether to remove intermediate files after completion
    **inference_kwargs
        Additional arguments passed to splice_inference_workflow
        
    Returns
    -------
    Dict[str, Path]
        Dictionary containing paths to generated files:
        - 'complete_scores': Per-nucleotide scores for all positions
        - 'coverage_report': Report on training vs. inferred positions
        - 'gene_summaries': Per-gene score summaries
        
        Examples
    --------
    >>> # Generate complete coverage with selective efficiency
    >>> results = enhanced_generate_per_nucleotide_meta_scores(
    ...     model_path="results/gene_cv_run_3/best_model.json",
    ...     target_genes=["ENSG00000104435", "ENSG00000130477"],  # STMN2, UNC13A
    ...     training_dataset_path="train_pc_1000_3mers/master",
    ...     output_dir="selective_meta_scores",
    ...     verbose=2
    ... )
    >>>
    >>> # Load hybrid predictions (complete coverage)
    >>> import pandas as pd
    >>> scores_df = pd.read_parquet(results['complete_scores'])
    >>> # Contains: donor_meta, acceptor_meta, neither_meta for ALL positions
    >>> # Uses meta-model recalibration for uncertain positions
    >>> # Uses base model predictions for confident positions
    """
    
    import tempfile
    import shutil
    from pathlib import Path
    from collections import defaultdict
    import datetime
    import json as _json
    import pandas as pd
    import numpy as np
    
    model_path = Path(model_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        print(f"\n🚀 ENHANCED META-SCORE GENERATION")
        print(f"=" * 70)
        print(f"📊 Target genes: {len(target_genes)}")
        print(f"🤖 Model: {model_path}")
        print(f"📁 Output: {output_dir}")
        print(f"🧬 Flanking size: {flanking_size}")
        print(f"♻️  Use training positions: {use_training_positions}")
    
    # Initialize results tracking
    results = {
        'complete_scores': None,
        'coverage_report': None, 
        'gene_summaries': None
    }
    
    # Step 1: Identify training coverage (if available)
    training_positions = defaultdict(set)
    if use_training_positions and training_dataset_path:
        training_dataset_path = Path(training_dataset_path)
        if training_dataset_path.exists():
            if verbose:
                print(f"\n📚 STEP 1: Loading training dataset coverage")
                print(f"   Dataset: {training_dataset_path}")
            
            try:
                # Load training positions for coverage analysis
                if training_dataset_path.is_dir():
                    train_lf = pl.scan_parquet(str(training_dataset_path / "*.parquet"))
                else:
                    train_lf = pl.scan_parquet(str(training_dataset_path))
                
                # Get gene-position mapping for target genes only
                coverage_df = (train_lf
                    .filter(pl.col("gene_id").is_in(target_genes))
                    .select(["gene_id", "position"])
                    .unique()
                    .collect())
                
                # Build coverage map
                for row in coverage_df.iter_rows():
                    gene_id, position = row
                    training_positions[gene_id].add(position)
                
                if verbose:
                    total_training_pos = sum(len(positions) for positions in training_positions.values())
                    print(f"   ✅ Found {total_training_pos:,} training positions across {len(training_positions)} genes")
                    
            except Exception as e:
                if verbose:
                    print(f"   ⚠️  Could not load training coverage: {e}")
                    print(f"   ➡️  Will generate predictions for all positions")
                training_positions = defaultdict(set)
    
    # Step 2: Run inference workflow for complete feature generation
    if verbose:
        print(f"\n🔬 STEP 2: Running splice inference workflow")
        print(f"   Generating base model predictions + features for all positions")
    
    # Create temporary directory for inference workflow
    with tempfile.TemporaryDirectory(prefix="enhanced_meta_inference_") as temp_dir:
        temp_dir = Path(temp_dir)
        
        # Configure inference workflow
        inference_config = {
            'target_genes': target_genes,
            'output_dir': temp_dir / "inference_output",
            'do_prepare_sequences': False,      # Use existing gene_sequence_*.parquet
            'do_prepare_annotations': False,    # Use existing annotations.db
            'cleanup': False,  # Keep files for meta-model inference
            'verbosity': max(0, verbose - 1),
            **inference_kwargs
        }
        
        # Import and run selective meta-inference workflow
        from meta_spliceai.splice_engine.meta_models.workflows.selective_meta_inference import (
            run_selective_meta_inference,
            SelectiveInferenceConfig
        )
        
        # Create selective inference config (aligned with clarified requirements)
        selective_config = SelectiveInferenceConfig(
            model_path=str(model_path),
            target_genes=target_genes,
            training_dataset_path=training_dataset_path if training_dataset_path else None,
            uncertainty_threshold_low=0.02,   # Below this: confident non-splice
            uncertainty_threshold_high=0.80,  # Above this: confident splice  
            max_positions_per_gene=10000,     # Prevent huge feature matrices
            inference_base_dir=temp_dir / "selective_inference",
            verbose=max(0, verbose - 1),
            cleanup_intermediates=cleanup_intermediates
        )
        
        workflow_results = run_selective_meta_inference(selective_config)
        
        if not workflow_results.success:
            error_msg = "; ".join(workflow_results.error_messages) if workflow_results.error_messages else "Unknown error"
            raise RuntimeError(f"Selective inference workflow failed: {error_msg}")
        
        if verbose:
            print(f"   ✅ Selective inference workflow completed successfully")
            print(f"   📊 Total positions: {workflow_results.total_positions:,}")
            print(f"   🤖 Recalibrated: {workflow_results.positions_recalibrated:,}")
            print(f"   🔄 Reused: {workflow_results.positions_reused:,}")
            print(f"   🧬 Genes: {workflow_results.genes_processed}")
        
        # Step 3: Load results from selective workflow
        if verbose:
            print(f"\n📊 STEP 3: Loading results from selective workflow")
        
        # The selective workflow provides hybrid predictions (complete coverage)!
        if not workflow_results.hybrid_predictions_path or not workflow_results.hybrid_predictions_path.exists():
            raise RuntimeError("Selective workflow did not generate hybrid predictions file")
        
        # Load the hybrid predictions (base + meta combined)
        final_results = pd.read_parquet(workflow_results.hybrid_predictions_path)
        
        # Map hybrid prediction columns to expected output format
        final_results = final_results.rename(columns={
            'donor_hybrid': 'donor_meta',
            'acceptor_hybrid': 'acceptor_meta', 
            'neither_hybrid': 'neither_meta'
        })
        
        # Add training information if not present
        if 'in_training' not in final_results.columns:
            final_results['in_training'] = final_results['prediction_source'] == 'meta_model'
        
        # Ensure we have the expected columns
        expected_cols = ["gene_id", "position", "chrom", 
                        "donor_score", "acceptor_score", "neither_score",
                        "donor_meta", "acceptor_meta", "neither_meta"]
        
        # Add strand to expected columns if available
        if "strand" in final_results.columns:
            expected_cols.insert(3, "strand")  # Insert after chrom
        
        missing_cols = [col for col in expected_cols if col not in final_results.columns]
        if missing_cols:
            raise RuntimeError(f"Missing expected columns in results: {missing_cols}")
        
        # Sort by gene and position for clean output
        final_results = final_results.sort_values(['gene_id', 'position'])
        
        if verbose:
            total_positions = len(final_results)
            recalibrated_positions = workflow_results.positions_recalibrated
            reused_positions = workflow_results.positions_reused
            print(f"   ✅ Final hybrid dataset: {total_positions:,} total positions")
            print(f"      🤖 Meta-model recalibrated: {recalibrated_positions:,} ({recalibrated_positions/total_positions:.1%})")
            print(f"      🔄 Base model reused: {reused_positions:,} ({reused_positions/total_positions:.1%})")
            print(f"      🧬 Genes covered: {final_results['gene_id'].nunique()}")
        
        # Save complete hybrid scores
        complete_scores_path = output_dir / "complete_per_nucleotide_meta_scores.parquet"
        final_results.to_parquet(complete_scores_path, index=False)
        results['complete_scores'] = complete_scores_path
        
        # Copy other outputs from selective workflow if they exist
        if workflow_results.meta_predictions_path and workflow_results.meta_predictions_path.exists():
            meta_only_path = output_dir / "meta_model_predictions_only.parquet"
            shutil.copy2(workflow_results.meta_predictions_path, meta_only_path)
            results['meta_predictions'] = meta_only_path
        
        if workflow_results.base_predictions_path and workflow_results.base_predictions_path.exists():
            base_only_path = output_dir / "base_model_predictions.parquet"
            shutil.copy2(workflow_results.base_predictions_path, base_only_path)
            results['base_predictions'] = base_only_path
        
        # Generate coverage report
        if workflow_results.per_gene_stats:
            coverage_report = pd.DataFrame(workflow_results.per_gene_stats).T
            coverage_report['gene_id'] = coverage_report.index
            coverage_report['coverage_efficiency'] = (
                coverage_report['recalibrated_positions'] / coverage_report['total_positions']
            )
            
            coverage_report_path = output_dir / "coverage_report.csv"
            coverage_report.to_csv(coverage_report_path, index=False)
            results['coverage_report'] = coverage_report_path
        
        # Gene-level summaries
        gene_summaries = final_results.groupby('gene_id').agg({
            'donor_meta': ['mean', 'std', 'min', 'max'],
            'acceptor_meta': ['mean', 'std', 'min', 'max'], 
            'neither_meta': ['mean', 'std', 'min', 'max'],
            'position': 'count',
            'is_uncertain': 'sum'
        })
        gene_summaries.columns = ['_'.join(col).strip() for col in gene_summaries.columns]
        
        gene_summaries_path = output_dir / "gene_summaries.csv"
        gene_summaries.to_csv(gene_summaries_path)
        results['gene_summaries'] = gene_summaries_path
        
        if verbose:
            print(f"\n🎉 SELECTIVE META-SCORE GENERATION COMPLETE!")
            print(f"=" * 70)
            print(f"📁 Complete hybrid scores: {complete_scores_path}")
            print(f"🎯 Strategy: Selective featurization + base model reuse")
            print(f"💾 Feature matrix efficiency: {workflow_results.feature_matrix_size_mb:.1f} MB")
            if 'coverage_report' in results:
                print(f"📊 Coverage report: {results['coverage_report']}")
            if 'gene_summaries' in results:
                print(f"📈 Gene summaries: {results['gene_summaries']}")
            print(f"\n✨ Complete coverage achieved with computational efficiency!")
    
    return results 