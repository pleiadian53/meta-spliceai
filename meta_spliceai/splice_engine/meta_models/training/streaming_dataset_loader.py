#!/usr/bin/env python3
"""
Streaming Dataset Loader: Load large datasets in chunks without OOM.

This module provides a memory-efficient way to work with large datasets by:
1. Loading data in gene-based chunks
2. Processing chunks sequentially
3. Never loading the full dataset into memory at once
4. Supporting all existing training workflows
"""

import polars as pl
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Iterator, Union
import logging
import gc
from dataclasses import dataclass


@dataclass
class DatasetChunk:
    """Represents a chunk of the dataset."""
    data: pl.DataFrame
    genes: List[str]
    chunk_id: int
    total_chunks: int
    
    @property
    def shape(self) -> Tuple[int, int]:
        return self.data.shape
    
    @property
    def memory_usage_gb(self) -> float:
        """Estimate memory usage in GB."""
        return self.data.estimated_size() / (1024**3)


class StreamingDatasetLoader:
    """
    Loads large datasets in memory-efficient chunks.
    
    Key features:
    - Processes genes in batches to avoid OOM
    - Maintains gene structure integrity
    - Compatible with existing training pipelines
    - Provides iterator interface for streaming processing
    """
    
    def __init__(
        self,
        dataset_path: str | Path,
        chunk_size_genes: int = 500,
        max_memory_gb: float = 8.0,
        verbose: bool = True
    ):
        self.dataset_path = Path(dataset_path)
        self.chunk_size_genes = chunk_size_genes
        self.max_memory_gb = max_memory_gb
        self.verbose = verbose
        
        # Cache for metadata
        self._gene_list = None
        self._total_genes = None
        self._schema = None
        
    def get_dataset_info(self) -> Dict:
        """Get dataset metadata without loading full data."""
        if self._gene_list is None:
            self._discover_genes()
            
        return {
            'total_genes': self._total_genes,
            'estimated_chunks': len(self.get_gene_chunks()),
            'chunk_size_genes': self.chunk_size_genes,
            'dataset_path': str(self.dataset_path)
        }
    
    def _discover_genes(self) -> None:
        """Discover unique genes in dataset."""
        if self.verbose:
            print("[StreamingLoader] Discovering genes in dataset...")
            
        lf = pl.scan_parquet(
            str(self.dataset_path / "*.parquet"), 
            extra_columns='ignore'
        )
        
        # Get unique genes efficiently
        genes_df = lf.select("gene_id").unique().collect()
        self._gene_list = sorted(genes_df["gene_id"].to_list())
        self._total_genes = len(self._gene_list)
        
        if self.verbose:
            print(f"[StreamingLoader] Found {self._total_genes} unique genes")
    
    def get_gene_chunks(self) -> List[List[str]]:
        """Split genes into chunks for streaming processing."""
        if self._gene_list is None:
            self._discover_genes()
            
        chunks = []
        for i in range(0, len(self._gene_list), self.chunk_size_genes):
            chunk = self._gene_list[i:i + self.chunk_size_genes]
            chunks.append(chunk)
            
        return chunks
    
    def load_chunk(self, gene_chunk: List[str], chunk_id: int, total_chunks: int) -> DatasetChunk:
        """Load a specific chunk of genes."""
        if self.verbose:
            print(f"[StreamingLoader] Loading chunk {chunk_id + 1}/{total_chunks} ({len(gene_chunk)} genes)")
            
        # Load data for specific genes
        lf = pl.scan_parquet(
            str(self.dataset_path / "*.parquet"), 
            extra_columns='ignore'
        )
        
        # Filter to specific genes
        chunk_data = lf.filter(pl.col("gene_id").is_in(gene_chunk)).collect()
        
        chunk = DatasetChunk(
            data=chunk_data,
            genes=gene_chunk,
            chunk_id=chunk_id,
            total_chunks=total_chunks
        )
        
        if self.verbose:
            print(f"[StreamingLoader] Loaded {chunk.shape[0]:,} positions, {chunk.memory_usage_gb:.2f} GB")
            
        return chunk
    
    def stream_chunks(self) -> Iterator[DatasetChunk]:
        """Stream dataset chunks one at a time."""
        gene_chunks = self.get_gene_chunks()
        total_chunks = len(gene_chunks)
        
        if self.verbose:
            print(f"[StreamingLoader] Streaming {total_chunks} chunks...")
            
        for chunk_id, gene_chunk in enumerate(gene_chunks):
            chunk = self.load_chunk(gene_chunk, chunk_id, total_chunks)
            yield chunk
            
            # Force garbage collection after each chunk
            gc.collect()
    
    def load_genes_subset(self, genes: List[str]) -> pl.DataFrame:
        """Load data for a specific subset of genes."""
        if self.verbose:
            print(f"[StreamingLoader] Loading {len(genes)} specific genes...")
            
        lf = pl.scan_parquet(
            str(self.dataset_path / "*.parquet"), 
            missing_columns='insert',
            extra_columns='ignore'
        )
        
        # Filter to specific genes
        data = lf.filter(pl.col("gene_id").is_in(genes)).collect()
        
        if self.verbose:
            print(f"[StreamingLoader] Loaded {data.shape[0]:,} positions for gene subset")
            
        return data
    
    def get_schema(self) -> Dict[str, str]:
        """Get dataset schema without loading data."""
        if self._schema is None:
            lf = pl.scan_parquet(
                str(self.dataset_path / "*.parquet"), 
                extra_columns='ignore'
            )
            self._schema = dict(lf.schema)
            
        return self._schema
    
    def estimate_total_rows(self) -> int:
        """Estimate total rows in dataset."""
        lf = pl.scan_parquet(
            str(self.dataset_path / "*.parquet"), 
            extra_columns='ignore'
        )
        
        count = lf.select(pl.len()).collect().item()
        return count
    
    def create_streaming_cv_splits(
        self, 
        n_folds: int = 5, 
        random_seed: int = 42
    ) -> List[Tuple[List[str], List[str]]]:
        """
        Create gene-aware CV splits for streaming processing.
        
        Returns:
            List of (train_genes, test_genes) tuples for each fold
        """
        if self._gene_list is None:
            self._discover_genes()
            
        from sklearn.model_selection import GroupKFold
        
        # Create gene-level splits
        genes_array = np.array(self._gene_list)
        dummy_y = np.zeros(len(self._gene_list))
        
        gkf = GroupKFold(n_splits=n_folds)
        splits = []
        
        for train_idx, test_idx in gkf.split(genes_array, dummy_y, genes_array):
            train_genes = genes_array[train_idx].tolist()
            test_genes = genes_array[test_idx].tolist()
            splits.append((train_genes, test_genes))
            
        if self.verbose:
            print(f"[StreamingLoader] Created {len(splits)} CV splits")
            for i, (train, test) in enumerate(splits):
                print(f"  Fold {i+1}: {len(train)} train genes, {len(test)} test genes")
                
        return splits


def create_memory_efficient_dataset(
    dataset_path: str | Path,
    max_genes_in_memory: int = 2000,
    max_memory_gb: float = 8.0,
    gene_start_idx: int = 0,
    gene_end_idx: Optional[int] = None,
    verbose: bool = True
) -> pl.DataFrame:
    """
    Create a memory-efficient dataset by sampling genes intelligently.
    
    This function determines the maximum number of genes that can fit in memory
    and samples them to create a representative dataset for training.
    
    Args:
        dataset_path: Path to dataset
        max_genes_in_memory: Maximum genes to keep in memory
        max_memory_gb: Maximum memory to use
        verbose: Whether to print progress
        
    Returns:
        DataFrame with sampled genes that fits in memory
    """
    loader = StreamingDatasetLoader(
        dataset_path=dataset_path,
        chunk_size_genes=500,
        max_memory_gb=max_memory_gb,
        verbose=verbose
    )
    
    info = loader.get_dataset_info()
    total_genes = info['total_genes']
    
    # Apply gene range selection if specified
    if gene_end_idx is not None:
        selected_genes = loader._gene_list[gene_start_idx:gene_end_idx]
        if verbose:
            print(f"[MemoryEfficientDataset] Using gene range {gene_start_idx}:{gene_end_idx} ({len(selected_genes)} genes)")
    elif gene_start_idx > 0:
        selected_genes = loader._gene_list[gene_start_idx:gene_start_idx + max_genes_in_memory]
        if verbose:
            print(f"[MemoryEfficientDataset] Using genes starting at index {gene_start_idx} ({len(selected_genes)} genes)")
    else:
        selected_genes = loader._gene_list
    
    # Determine final gene set based on memory constraints
    if len(selected_genes) <= max_genes_in_memory:
        # Selected genes fit in memory
        if verbose:
            print(f"[MemoryEfficientDataset] Loading {len(selected_genes)} genes (fits in memory)")
        genes_to_load = selected_genes
    else:
        # Sample from selected genes to fit in memory
        if verbose:
            print(f"[MemoryEfficientDataset] Sampling {max_genes_in_memory} from {len(selected_genes)} selected genes")
        
        # Use systematic sampling to ensure good coverage
        import random
        random.seed(42)
        genes_to_load = random.sample(selected_genes, max_genes_in_memory)
        genes_to_load.sort()  # Keep sorted for reproducibility
    
    # Load the selected genes
    dataset = loader.load_genes_subset(genes_to_load)
    
    if verbose:
        print(f"[MemoryEfficientDataset] Final dataset: {dataset.shape[0]:,} positions, {len(genes_to_load)} genes")
        estimated_memory = dataset.estimated_size() / (1024**3)
        print(f"[MemoryEfficientDataset] Estimated memory usage: {estimated_memory:.2f} GB")
    
    return dataset


if __name__ == "__main__":
    # Test the streaming loader
    import argparse
    
    parser = argparse.ArgumentParser(description="Test streaming dataset loader")
    parser.add_argument("--dataset", required=True, help="Dataset path")
    parser.add_argument("--chunk-size", type=int, default=500, help="Genes per chunk")
    parser.add_argument("--max-memory", type=float, default=8.0, help="Max memory GB")
    parser.add_argument("--test-streaming", action="store_true", help="Test streaming chunks")
    parser.add_argument("--test-efficient", action="store_true", help="Test memory-efficient dataset")
    
    args = parser.parse_args()
    
    loader = StreamingDatasetLoader(
        dataset_path=args.dataset,
        chunk_size_genes=args.chunk_size,
        max_memory_gb=args.max_memory,
        verbose=True
    )
    
    # Print dataset info
    info = loader.get_dataset_info()
    print(f"\nDataset Info: {info}")
    
    if args.test_streaming:
        print("\nTesting streaming chunks...")
        chunk_count = 0
        total_positions = 0
        
        for chunk in loader.stream_chunks():
            chunk_count += 1
            total_positions += chunk.shape[0]
            
            if chunk_count >= 3:  # Test first 3 chunks only
                print(f"Stopping after {chunk_count} chunks for testing")
                break
                
        print(f"Processed {chunk_count} chunks with {total_positions:,} total positions")
    
    if args.test_efficient:
        print("\nTesting memory-efficient dataset creation...")
        dataset = create_memory_efficient_dataset(
            dataset_path=args.dataset,
            max_genes_in_memory=2000,
            verbose=True
        )
        print(f"Created efficient dataset with shape: {dataset.shape}")
