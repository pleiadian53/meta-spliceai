#!/usr/bin/env python3
"""
Incremental Per-Nucleotide Score Generator

This module provides memory-optimized generation of per-nucleotide meta-model scores
for large datasets, processing genes in chunks while preserving gene structure.

Key Features:
- Memory-optimized loading using incremental batching
- Gene-preserving chunked processing 
- Robust data type handling
- Compatible with existing meta-model workflows
- Configurable memory limits and chunk sizes

Author: AI Assistant
Created: 2025-09-02
"""

from __future__ import annotations

import json
import numpy as np
import pandas as pd
import polars as pl
from pathlib import Path
from typing import Optional, Union, Dict, List, Any
from sklearn.metrics import f1_score, classification_report

from . import classifier_utils as _cutils


class IncrementalScoreGenerator:
    """
    Memory-optimized generator for per-nucleotide meta-model scores.
    
    Processes large datasets in gene-preserving chunks to avoid OOM issues
    while maintaining biological validity.
    """
    
    def __init__(
        self,
        dataset_path: Union[str, Path],
        run_dir: Union[str, Path],
        *,
        max_memory_gb: float = 8.0,
        target_chunk_size: int = 25_000,
        verbose: int = 1
    ):
        """
        Initialize the incremental score generator.
        
        Parameters
        ----------
        dataset_path : str | Path
            Path to training dataset
        run_dir : str | Path
            Directory containing trained meta-model
        max_memory_gb : float
            Maximum memory to use for loading (default: 8.0GB)
        target_chunk_size : int
            Target number of positions per chunk (default: 25,000)
        verbose : int
            Verbosity level (default: 1)
        """
        self.dataset_path = Path(dataset_path)
        self.run_dir = Path(run_dir)
        self.max_memory_gb = max_memory_gb
        self.target_chunk_size = target_chunk_size
        self.verbose = verbose
        
        # Load meta-model once during initialization
        if self.verbose:
            print(f"[IncrementalScoreGen] Initializing with dataset: {self.dataset_path}")
            print(f"[IncrementalScoreGen] Model directory: {self.run_dir}")
            print(f"[IncrementalScoreGen] Memory limit: {self.max_memory_gb}GB")
            print(f"[IncrementalScoreGen] Target chunk size: {self.target_chunk_size:,}")
        
        self.predict_fn, self.feature_names = _cutils._load_model_generic(self.run_dir)
        
        if self.verbose:
            print(f"[IncrementalScoreGen] ✓ Meta-model loaded successfully")
            print(f"[IncrementalScoreGen] Expected features: {len(self.feature_names)}")
    
    def generate_scores(
        self,
        sample: Optional[int] = None,
        output_format: str = "parquet",
        gene_manifest_path: Optional[Union[str, Path]] = None
    ) -> Path:
        """
        Generate per-nucleotide meta-model scores using incremental processing.
        
        Parameters
        ----------
        sample : int | None
            Optional sample size for faster processing
        output_format : str
            Output format: "parquet" or "tsv" (default: "parquet")
        gene_manifest_path : str | Path | None
            Optional path to gene manifest
            
        Returns
        -------
        Path
            Path to generated score file
        """
        if self.verbose:
            print(f"\n[IncrementalScoreGen] 🚀 Starting incremental score generation")
            print(f"[IncrementalScoreGen] Sample size: {sample or 'full dataset'}")
            print(f"[IncrementalScoreGen] Output format: {output_format}")
        
        # Load dataset using memory optimization
        df = self._load_dataset_optimized()
        
        # Apply sampling if requested (gene-preserving)
        if sample is not None and df.height > sample:
            df = self._apply_gene_preserving_sampling(df, sample)
        
        # Process genes incrementally
        results = self._process_genes_incrementally(df)
        
        # Save results
        output_path = self._save_results(results, output_format)
        
        if self.verbose:
            print(f"[IncrementalScoreGen] ✅ Score generation complete!")
            print(f"[IncrementalScoreGen] Output saved to: {output_path}")
        
        return output_path
    
    def _load_dataset_optimized(self) -> pl.DataFrame:
        """Load dataset using memory optimization."""
        try:
            from meta_spliceai.splice_engine.meta_models.training.memory_optimized_datasets import (
                load_dataset_with_memory_management,
                estimate_dataset_size_efficiently
            )
            
            # Estimate size
            estimated_rows, file_count = estimate_dataset_size_efficiently(self.dataset_path)
            
            if estimated_rows > 100_000:
                if self.verbose:
                    print(f"[IncrementalScoreGen] Large dataset detected ({estimated_rows:,} rows)")
                    print(f"[IncrementalScoreGen] Using memory-optimized loading...")
                
                df = load_dataset_with_memory_management(
                    self.dataset_path,
                    max_memory_gb=self.max_memory_gb,
                    fallback_to_standard=False
                )
            else:
                if self.verbose:
                    print(f"[IncrementalScoreGen] Standard dataset size, using direct loading")
                
                # Standard loading for smaller datasets
                if self.dataset_path.is_dir():
                    lf = pl.scan_parquet(str(self.dataset_path / "*.parquet"), 
                                       missing_columns="insert", extra_columns="ignore")
                else:
                    lf = pl.scan_parquet(str(self.dataset_path), 
                                       missing_columns="insert", extra_columns="ignore")
                df = lf.collect()
            
            if self.verbose:
                print(f"[IncrementalScoreGen] Dataset loaded: {df.shape}")
            
            return df
            
        except Exception as e:
            print(f"[ERROR] Failed to load dataset: {e}")
            raise RuntimeError(f"Cannot load dataset for score generation: {e}")
    
    def _apply_gene_preserving_sampling(self, df: pl.DataFrame, sample_size: int) -> pl.DataFrame:
        """Apply gene-preserving sampling to reduce dataset size."""
        if self.verbose:
            print(f"[IncrementalScoreGen] Applying gene-preserving sampling: {sample_size:,} positions")
        
        # Get unique genes
        unique_genes = df.select("gene_id").unique().to_pandas()["gene_id"].tolist()
        total_genes = len(unique_genes)
        avg_positions_per_gene = df.height // total_genes
        
        # Calculate how many genes to sample
        target_genes = min(total_genes, max(1, sample_size // avg_positions_per_gene))
        
        if self.verbose:
            print(f"[IncrementalScoreGen] Sampling {target_genes} genes from {total_genes} total")
        
        # Sample genes randomly
        import random
        random.seed(42)
        sampled_genes = random.sample(unique_genes, target_genes)
        
        # Filter to sampled genes
        df_sampled = df.filter(pl.col("gene_id").is_in(sampled_genes))
        
        if self.verbose:
            print(f"[IncrementalScoreGen] Sampled dataset: {df_sampled.shape}")
        
        return df_sampled
    
    def _process_genes_incrementally(self, df: pl.DataFrame) -> List[Dict[str, Any]]:
        """Process genes in chunks to generate scores incrementally."""
        if self.verbose:
            print(f"[IncrementalScoreGen] Processing genes incrementally...")
        
        # Get unique genes and calculate chunk size
        unique_genes = df.select("gene_id").unique().to_pandas()["gene_id"].tolist()
        total_genes = len(unique_genes)
        avg_positions_per_gene = df.height // total_genes
        
        # Calculate genes per chunk to stay within target chunk size
        target_genes_per_chunk = max(1, self.target_chunk_size // (avg_positions_per_gene * 2))
        
        if self.verbose:
            print(f"[IncrementalScoreGen] Total genes: {total_genes:,}")
            print(f"[IncrementalScoreGen] Avg positions per gene: {avg_positions_per_gene:.0f}")
            print(f"[IncrementalScoreGen] Genes per chunk: {target_genes_per_chunk}")
        
        all_results = []
        processed_positions = 0
        processed_genes = 0
        
        # Process genes in chunks
        for chunk_start in range(0, total_genes, target_genes_per_chunk):
            chunk_end = min(chunk_start + target_genes_per_chunk, total_genes)
            chunk_genes = unique_genes[chunk_start:chunk_end]
            
            chunk_num = chunk_start // target_genes_per_chunk + 1
            total_chunks = (total_genes + target_genes_per_chunk - 1) // target_genes_per_chunk
            
            if self.verbose:
                print(f"[IncrementalScoreGen] Processing chunk {chunk_num}/{total_chunks}")
                print(f"[IncrementalScoreGen] Genes {chunk_start+1}-{chunk_end} ({len(chunk_genes)} genes)")
            
            # Process this chunk
            chunk_results = self._process_gene_chunk(df, chunk_genes, chunk_num)
            
            if chunk_results:
                all_results.append(chunk_results)
                processed_positions += chunk_results["positions"]
                processed_genes += len(chunk_genes)
                
                if self.verbose:
                    print(f"[IncrementalScoreGen] Chunk {chunk_num} complete: {chunk_results['positions']:,} positions")
        
        if self.verbose:
            print(f"[IncrementalScoreGen] Processing complete: {processed_positions:,} positions, {processed_genes} genes")
        
        return all_results
    
    def _process_gene_chunk(self, df: pl.DataFrame, chunk_genes: List[str], chunk_id: int) -> Optional[Dict[str, Any]]:
        """Process a single chunk of genes."""
        # Filter to this chunk of genes
        chunk_df = df.filter(pl.col("gene_id").is_in(chunk_genes))
        
        if chunk_df.height == 0:
            return None
        
        # Convert to pandas for processing
        chunk_pd = chunk_df.to_pandas()
        
        # Verify required columns
        required_cols = ["gene_id", "splice_type", "donor_score", "acceptor_score", "neither_score"]
        missing = [col for col in required_cols if col not in chunk_pd.columns]
        if missing:
            if self.verbose > 1:
                print(f"[IncrementalScoreGen] Warning: Missing required columns in chunk {chunk_id}: {missing}")
            return None
        
        # Extract features with robust data type handling
        X = self._extract_features_robust(chunk_pd)
        
        # Generate meta-model predictions
        meta_probs = self.predict_fn(X)
        
        # Extract base model scores
        base_scores = {
            "donor_score": chunk_pd["donor_score"].values,
            "acceptor_score": chunk_pd["acceptor_score"].values,
            "neither_score": chunk_pd["neither_score"].values
        }
        
        # Create result structure
        chunk_results = {
            "chunk_id": chunk_id,
            "genes": chunk_genes,
            "positions": len(chunk_pd),
            "gene_ids": chunk_pd["gene_id"].values,
            "splice_types": chunk_pd["splice_type"].values,
            "base_scores": base_scores,
            "meta_scores": {
                "donor_meta": meta_probs[:, 1],
                "acceptor_meta": meta_probs[:, 2], 
                "neither_meta": meta_probs[:, 0]
            }
        }
        
        # Add positional information if available
        positional_cols = ["position", "chrom", "strand", "gene_start", "gene_end"]
        for col in positional_cols:
            if col in chunk_pd.columns:
                chunk_results[col] = chunk_pd[col].values
        
        return chunk_results
    
    def _extract_features_robust(self, chunk_pd: pd.DataFrame) -> np.ndarray:
        """Extract features with robust data type handling."""
        # Create feature matrix
        X = np.zeros((len(chunk_pd), len(self.feature_names)), dtype=np.float32)
        
        # Fill features with robust conversion
        for i, feat_name in enumerate(self.feature_names):
            if feat_name in chunk_pd.columns:
                try:
                    # Handle numeric columns
                    col_values = chunk_pd[feat_name].fillna(0)
                    if col_values.dtype == 'object':
                        # Convert categorical/string columns to numeric
                        col_values = pd.to_numeric(col_values, errors='coerce').fillna(0)
                    X[:, i] = col_values.values.astype(np.float32)
                except (ValueError, TypeError) as e:
                    if self.verbose > 1:
                        print(f"[IncrementalScoreGen] Warning: Could not convert feature '{feat_name}' to numeric: {e}")
                    # Leave as zeros for problematic columns
        
        return X
    
    def _save_results(self, all_results: List[Dict[str, Any]], output_format: str) -> Path:
        """Save aggregated results to file."""
        if not all_results:
            raise ValueError("No results to save")
        
        if self.verbose:
            print(f"[IncrementalScoreGen] Aggregating results from {len(all_results)} chunks...")
        
        # Combine all chunk results
        combined_data = {
            "gene_id": np.concatenate([r["gene_ids"] for r in all_results]),
            "splice_type": np.concatenate([r["splice_types"] for r in all_results]),
            "donor_score": np.concatenate([r["base_scores"]["donor_score"] for r in all_results]),
            "acceptor_score": np.concatenate([r["base_scores"]["acceptor_score"] for r in all_results]),
            "neither_score": np.concatenate([r["base_scores"]["neither_score"] for r in all_results]),
            "donor_meta": np.concatenate([r["meta_scores"]["donor_meta"] for r in all_results]),
            "acceptor_meta": np.concatenate([r["meta_scores"]["acceptor_meta"] for r in all_results]),
            "neither_meta": np.concatenate([r["meta_scores"]["neither_meta"] for r in all_results])
        }
        
        # Add positional columns if available
        positional_cols = ["position", "chrom", "strand", "gene_start", "gene_end"]
        for col in positional_cols:
            if col in all_results[0]:
                combined_data[col] = np.concatenate([r[col] for r in all_results])
        
        # Create DataFrame
        results_df = pd.DataFrame(combined_data)
        
        # Calculate some summary statistics
        total_positions = len(results_df)
        total_genes = len(results_df["gene_id"].unique())
        
        if self.verbose:
            print(f"[IncrementalScoreGen] Final results: {total_positions:,} positions, {total_genes} genes")
            print(f"[IncrementalScoreGen] Splice type distribution:")
            splice_counts = results_df["splice_type"].value_counts()
            for splice_type, count in splice_counts.items():
                print(f"  {splice_type}: {count:,} ({count/total_positions:.1%})")
        
        # Save to file
        if output_format.lower() == "parquet":
            output_path = self.run_dir / "per_nucleotide_meta_scores.parquet"
            results_df.to_parquet(output_path, index=False)
        else:
            output_path = self.run_dir / "per_nucleotide_meta_scores.tsv"
            results_df.to_csv(output_path, sep='\t', index=False)
        
        # Save metadata
        metadata = {
            "total_positions": total_positions,
            "total_genes": total_genes,
            "chunk_count": len(all_results),
            "target_chunk_size": self.target_chunk_size,
            "max_memory_gb": self.max_memory_gb,
            "output_format": output_format,
            "feature_count": len(self.feature_names),
            "generation_timestamp": pd.Timestamp.now().isoformat(),
            "splice_type_distribution": splice_counts.to_dict()
        }
        
        metadata_path = self.run_dir / "per_nucleotide_meta_scores_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        if self.verbose:
            print(f"[IncrementalScoreGen] Metadata saved to: {metadata_path}")
        
        return output_path


def generate_per_nucleotide_meta_scores_incremental(
    dataset_path: Union[str, Path],
    run_dir: Union[str, Path],
    *,
    sample: Optional[int] = None,
    output_format: str = "parquet",
    max_memory_gb: float = 8.0,
    target_chunk_size: int = 25_000,
    verbose: int = 1,
    **kwargs  # Accept additional kwargs for compatibility
) -> Path:
    """
    Generate per-nucleotide meta-model scores using incremental processing.
    
    This function provides a memory-optimized alternative to the original
    generate_per_nucleotide_meta_scores function, processing genes in chunks
    to avoid OOM issues while preserving gene structure.
    
    Parameters
    ----------
    dataset_path : str | Path
        Path to training dataset
    run_dir : str | Path
        Directory containing trained meta-model
    sample : int | None
        Optional sample size for faster processing
    output_format : str
        Output format: "parquet" or "tsv" (default: "parquet")
    max_memory_gb : float
        Maximum memory to use for loading (default: 8.0GB)
    target_chunk_size : int
        Target positions per chunk (default: 25,000)
    verbose : int
        Verbosity level (default: 1)
    **kwargs
        Additional arguments for compatibility (ignored)
        
    Returns
    -------
    Path
        Path to generated score file
        
    Examples
    --------
    >>> # Generate scores for large dataset
    >>> score_path = generate_per_nucleotide_meta_scores_incremental(
    ...     dataset_path="train_regulatory_10k_kmers/master",
    ...     run_dir="results/gene_cv_run",
    ...     sample=50000,
    ...     verbose=2
    ... )
    
    >>> # Load the results
    >>> import pandas as pd
    >>> scores_df = pd.read_parquet(score_path)
    >>> # Contains: gene_id, splice_type, donor_score, acceptor_score, neither_score,
    >>> #          donor_meta, acceptor_meta, neither_meta, position, chrom, etc.
    """
    if verbose:
        print(f"\n🧬 INCREMENTAL PER-NUCLEOTIDE SCORE GENERATION")
        print(f"=" * 60)
    
    # Create generator and process
    generator = IncrementalScoreGenerator(
        dataset_path=dataset_path,
        run_dir=run_dir,
        max_memory_gb=max_memory_gb,
        target_chunk_size=target_chunk_size,
        verbose=verbose
    )
    
    return generator.generate_scores(
        sample=sample,
        output_format=output_format
    )


# Backward compatibility alias
generate_per_nucleotide_meta_scores = generate_per_nucleotide_meta_scores_incremental

