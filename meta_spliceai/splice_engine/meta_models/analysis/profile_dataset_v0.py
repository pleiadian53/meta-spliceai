#!/usr/bin/env python3
"""Comprehensive EDA module for splice site prediction datasets.

This script provides detailed profiling of training datasets specifically for
splice site prediction, including:
- Dataset size and composition
- Splice type distributions  
- Base model error analysis (FP, FN, TP, TN)
- Feature categorization and quality
- Data quality metrics
- Gene manifest validation
- Artifact analysis (analysis_sequences, splice_positions_enhanced)
- Unseen position analysis
- Comprehensive visualizations

Supports batch-by-batch processing for large datasets to avoid OOM issues.

Usage
-----
python -m meta_spliceai.splice_engine.meta_models.analysis.profile_dataset \
    --dataset train_pc_1000/master \
    --output-dir results/dataset_profile \
    --generate-plots \
    --batch-size 100000 \
    --verbose

Example outputs:
- Dataset size: 1,234,567 samples across 3,456 features
- Splice sites: 12.3% (donor: 6.1%, acceptor: 6.2%, neither: 87.7%)
- Base model accuracy: 94.2% (FP: 2.1%, FN: 3.7%)
- Feature types: SpliceAI (43), K-mers (4096), Genomic (234)
- Gene manifest validation: 1,001 genes in manifest, 1,000 in training data
- Unseen positions: 45.2% of gene positions are unseen (not in training data)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union, Iterator
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import gc
from itertools import islice

# Data loading
try:
    import polars as pl
    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False
    pl = None

from meta_spliceai.splice_engine.meta_models.training import datasets
from meta_spliceai.splice_engine.meta_models.training.classifier_utils import _encode_labels
from meta_spliceai.splice_engine.meta_models.training.label_utils import LABEL_MAP_STR, INT_TO_LABEL

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class StreamingStats:
    """Accumulate statistics in a streaming fashion to avoid OOM."""
    
    def __init__(self, track_values_for_median=False, max_values_for_median=10000):
        self.count = 0
        self.sum = 0.0
        self.sum_sq = 0.0
        self.min_val = float('inf')
        self.max_val = float('-inf')
        self.value_counts = Counter()
        
        # For median calculation (optional, memory-intensive)
        self.track_values_for_median = track_values_for_median
        self.max_values_for_median = max_values_for_median
        self.values_sample = [] if track_values_for_median else None
        
    def update(self, values):
        """Update stats with new values (can be single value or array)."""
        if np.isscalar(values):
            values = [values]
        
        for val in values:
            if pd.isna(val):
                continue
                
            self.count += 1
            self.sum += val
            self.sum_sq += val ** 2
            self.min_val = min(self.min_val, val)
            self.max_val = max(self.max_val, val)
            
            # Store values for median calculation (with sampling to avoid OOM)
            if self.track_values_for_median and self.values_sample is not None:
                if len(self.values_sample) < self.max_values_for_median:
                    self.values_sample.append(val)
                else:
                    # Reservoir sampling to maintain representative sample
                    import random
                    k = random.randint(0, self.count - 1)
                    if k < self.max_values_for_median:
                        self.values_sample[k] = val
    
    def update_categorical(self, values):
        """Update categorical value counts."""
        if np.isscalar(values):
            values = [values]
        
        for val in values:
            if not pd.isna(val):
                self.value_counts[val] += 1
    
    def get_stats(self):
        """Get computed statistics."""
        if self.count == 0:
            return {
                'count': 0, 'mean': 0, 'std': 0, 'min': 0, 'max': 0,
                'var': 0, 'sum': 0, 'median': 0
            }
        
        mean = self.sum / self.count
        var = (self.sum_sq / self.count) - (mean ** 2)
        std = np.sqrt(max(0, var))  # Avoid negative variance due to floating point errors
        
        # Calculate median if values are tracked
        median = 0
        if self.track_values_for_median and self.values_sample:
            median = np.median(self.values_sample)
        else:
            # Approximate median as mean for streaming case
            median = mean
        
        return {
            'count': self.count,
            'mean': mean,
            'std': std,
            'min': self.min_val,
            'max': self.max_val,
            'var': var,
            'sum': self.sum,
            'median': median
        }


class SpliceDatasetProfiler:
    """Comprehensive profiler for splice site prediction datasets with batch processing support."""
    
    def __init__(self, verbose: bool = True, batch_size: int = 100000):
        self.verbose = verbose
        self.batch_size = batch_size
        self.profile = {}
        self.gene_name_mapping = {}  # For mapping gene IDs to gene names
        self.gene_features = None  # Gene features DataFrame
        self.gene_manifest = None  # Gene manifest DataFrame
        
        # Streaming statistics accumulators
        self.streaming_stats = {}
        self.categorical_stats = {}
        self.splice_type_counts = Counter()
        self.gene_counts = Counter()
        self.total_samples = 0
        self.duplicate_count = 0
        self.null_counts = Counter()
        
    def log(self, message: str, level: str = "INFO"):
        """Log messages with appropriate level."""
        if self.verbose:
            if level == "INFO":
                logger.info(message)
            elif level == "WARNING":
                logger.warning(message)
            elif level == "ERROR":
                logger.error(message)
    
    def load_dataset_batches(self, dataset_path: str, gene_filter: Optional[List[str]] = None, 
                           max_files: Optional[int] = None, sample_rows: Optional[int] = None) -> Iterator[pd.DataFrame]:
        """Load dataset in batches to avoid OOM issues."""
        self.log(f"Loading dataset in batches from: {dataset_path}")
        
        # Get list of parquet files
        dataset_dir = Path(dataset_path)
        if dataset_dir.is_file():
            parquet_files = [dataset_dir]
        else:
            parquet_files = sorted(list(dataset_dir.glob("*.parquet")))
        
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found in {dataset_path}")
        
        # Limit number of files for testing
        if max_files:
            parquet_files = parquet_files[:max_files]
            self.log(f"Limited to first {max_files} files for testing")
        
        self.log(f"Found {len(parquet_files)} parquet files")
        
        if gene_filter:
            self.log(f"Filtering for genes: {gene_filter}")
        if sample_rows:
            self.log(f"Sampling {sample_rows} rows per file for testing")
        
        total_rows = 0
        for file_path in parquet_files:
            self.log(f"Processing file: {file_path.name}")
            
            try:
                # Read file info first to estimate memory usage
                file_info = file_path.stat()
                file_size_mb = file_info.st_size / (1024**2)
                self.log(f"  File size: {file_size_mb:.1f} MB")
                
                # Use polars for more memory-efficient loading if available
                if POLARS_AVAILABLE:
                    # Read in chunks using polars
                    df_lazy = pl.scan_parquet(file_path)
                    
                    # Get total rows for this file
                    try:
                        file_rows = df_lazy.select(pl.len()).collect().item()
                        self.log(f"  Rows in file: {file_rows:,}")
                        total_rows += file_rows
                    except Exception:
                        self.log(f"  Could not get row count for {file_path.name}")
                    
                    # Apply gene filtering if specified
                    if gene_filter:
                        gene_col = None
                        # Try to identify gene column
                        for col in df_lazy.columns:
                            if 'gene_id' in col.lower():
                                gene_col = col
                                break
                        
                        if gene_col:
                            df_lazy = df_lazy.filter(pl.col(gene_col).is_in(gene_filter))
                            self.log(f"  Applied gene filter on column: {gene_col}")
                        else:
                            self.log("  Warning: Could not find gene_id column for filtering", "WARNING")
                    
                    # Apply row sampling if specified
                    if sample_rows:
                        df_lazy = df_lazy.head(sample_rows)
                    
                    # Process in batches
                    offset = 0
                    while True:
                        try:
                            batch_df = df_lazy.slice(offset, self.batch_size).collect().to_pandas()
                            if len(batch_df) == 0:
                                break
                            
                            self.log(f"  Processing batch: rows {offset:,} to {offset + len(batch_df):,}")
                            yield batch_df
                            
                            offset += self.batch_size
                            
                            # Force garbage collection after each batch
                            del batch_df
                            gc.collect()
                            
                        except Exception as e:
                            self.log(f"  Error reading batch at offset {offset}: {e}", "WARNING")
                            break
                
                else:
                    # Fallback to pandas with chunking
                    try:
                        # Try to read in chunks
                        chunk_iter = pd.read_parquet(file_path, chunksize=self.batch_size)
                        for batch_df in chunk_iter:
                            # Apply gene filtering if specified
                            if gene_filter:
                                gene_col = self._find_gene_column(batch_df)
                                if gene_col:
                                    batch_df = batch_df[batch_df[gene_col].isin(gene_filter)]
                                    if len(batch_df) == 0:
                                        continue  # Skip empty batch after filtering
                            
                            # Apply row sampling if specified
                            if sample_rows and len(batch_df) > sample_rows:
                                batch_df = batch_df.head(sample_rows)
                            
                            self.log(f"  Processing batch: {len(batch_df):,} rows")
                            yield batch_df
                            
                            # Force garbage collection
                            del batch_df
                            gc.collect()
                    except Exception:
                        # If chunking fails, read entire file (risk OOM)
                        self.log(f"  Warning: Reading entire file {file_path.name} at once", "WARNING")
                        batch_df = pd.read_parquet(file_path)
                        
                        # Apply filtering and sampling to full file
                        if gene_filter:
                            gene_col = self._find_gene_column(batch_df)
                            if gene_col:
                                batch_df = batch_df[batch_df[gene_col].isin(gene_filter)]
                        
                        if sample_rows and len(batch_df) > sample_rows:
                            batch_df = batch_df.head(sample_rows)
                        
                        if len(batch_df) > 0:
                            yield batch_df
                        del batch_df
                        gc.collect()
                        
            except Exception as e:
                self.log(f"Error processing {file_path}: {e}", "ERROR")
                continue
        
        self.log(f"Total estimated rows across all files: {total_rows:,}")
    
    def load_dataset(self, dataset_path: str) -> pd.DataFrame:
        """Load full dataset (kept for backward compatibility, but not recommended for large datasets)."""
        self.log(f"WARNING: Loading full dataset - may cause OOM for large datasets", "WARNING")
        self.log(f"Consider using batch processing instead")
        
        try:
            # Use the standard dataset loader
            df = datasets.load_dataset(dataset_path)
            
            # Convert to pandas if it's a polars DataFrame
            if hasattr(df, 'to_pandas'):
                self.log("Converting Polars DataFrame to Pandas...")
                df = df.to_pandas()
            
            self.log(f"Successfully loaded {len(df):,} samples with {len(df.columns)} columns")
            return df
        except Exception as e:
            self.log(f"Error loading dataset: {e}", "ERROR")
            raise
    
    def load_gene_features(self, gene_features_path: str = "data/ensembl/spliceai_analysis/gene_features.tsv") -> pd.DataFrame:
        """Load gene features and create gene name mapping."""
        try:
            if os.path.exists(gene_features_path):
                self.log(f"Loading gene features from: {gene_features_path}")
                self.gene_features = pd.read_csv(gene_features_path, sep='\t')
                
                # Create mapping from gene_id to gene_name
                if 'gene_id' in self.gene_features.columns and 'gene_name' in self.gene_features.columns:
                    # Handle missing gene names gracefully
                    self.gene_features['gene_name'] = self.gene_features['gene_name'].fillna(self.gene_features['gene_id'])
                    self.gene_name_mapping = dict(zip(self.gene_features['gene_id'], self.gene_features['gene_name']))
                    self.log(f"Created gene name mapping for {len(self.gene_name_mapping)} genes")
                else:
                    self.log("Gene features file missing 'gene_id' or 'gene_name' columns", "WARNING")
                
                return self.gene_features
            else:
                self.log(f"Gene features file not found at: {gene_features_path}", "WARNING")
                return None
        except Exception as e:
            self.log(f"Error loading gene features: {e}", "WARNING")
            return None
    
    def load_gene_manifest(self, dataset_path: str) -> pd.DataFrame:
        """Load gene manifest from dataset directory."""
        try:
            # Look for gene_manifest.csv in the dataset directory or parent directory
            dataset_dir = Path(dataset_path).parent if Path(dataset_path).is_file() else Path(dataset_path)
            manifest_path = dataset_dir / "gene_manifest.csv"
            
            # If not found in dataset directory, try parent directory
            if not manifest_path.exists():
                parent_dir = dataset_dir.parent
                manifest_path = parent_dir / "gene_manifest.csv"
            
            if manifest_path.exists():
                self.log(f"Loading gene manifest from: {manifest_path}")
                self.gene_manifest = pd.read_csv(manifest_path)
                self.log(f"Loaded gene manifest with {len(self.gene_manifest)} entries")
                return self.gene_manifest
            else:
                self.log(f"Gene manifest not found at: {manifest_path}", "WARNING")
                return None
        except Exception as e:
            self.log(f"Error loading gene manifest: {e}", "WARNING")
            return None
    
    def get_gene_display_name(self, gene_id: str) -> str:
        """Get display name for a gene (gene name if available, otherwise gene ID)."""
        if gene_id in self.gene_name_mapping:
            gene_name = self.gene_name_mapping[gene_id]
            # If gene name is not empty/null and different from gene ID, use it
            if gene_name and gene_name != gene_id:
                return f"{gene_name} ({gene_id})"
            else:
                return gene_id
        else:
            return gene_id
    
    def update_streaming_stats(self, df: pd.DataFrame):
        """Update streaming statistics with a new batch."""
        self.total_samples += len(df)
        
        # Update null counts
        null_counts_batch = df.isnull().sum()
        for col, count in null_counts_batch.items():
            self.null_counts[col] += count
        
        # Update duplicate count (approximate - only within batch)
        self.duplicate_count += df.duplicated().sum()
        
        # Initialize streaming stats for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col not in self.streaming_stats:
                # Track median for SpliceAI scores (important for interpretation)
                track_median = any(score_type in col.lower() for score_type in ['donor_score', 'acceptor_score', 'neither_score'])
                self.streaming_stats[col] = StreamingStats(track_values_for_median=track_median)
            
            # Update with non-null values
            values = df[col].dropna()
            if len(values) > 0:
                self.streaming_stats[col].update(values)
        
        # Update categorical stats
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            if col not in self.categorical_stats:
                self.categorical_stats[col] = Counter()
            
            # Update value counts
            values = df[col].dropna()
            for val in values:
                self.categorical_stats[col][val] += 1
        
        # Update splice type counts
        splice_col = self._find_splice_column(df)
        if splice_col:
            splice_values = df[splice_col].dropna()
            for val in splice_values:
                self.splice_type_counts[val] += 1
        
        # Update gene counts
        gene_col = self._find_gene_column(df)
        if gene_col:
            gene_values = df[gene_col].dropna()
            for val in gene_values:
                self.gene_counts[val] += 1
    
    def _find_splice_column(self, df: pd.DataFrame) -> Optional[str]:
        """Find splice type column in DataFrame."""
        for col in df.columns:
            if 'splice_type' in col.lower() or col in ['label', 'target', 'class']:
                return col
        return None
    
    def _find_gene_column(self, df: pd.DataFrame) -> Optional[str]:
        """Find gene ID column in DataFrame."""
        for col in df.columns:
            if 'gene_id' in col.lower() or col == 'gene':
                return col
        return None
    
    def analyze_basic_info_streaming(self, dataset_path: str, gene_filter: Optional[List[str]] = None, 
                                   max_files: Optional[int] = None, sample_rows: Optional[int] = None) -> Dict[str, Any]:
        """Analyze basic dataset information using streaming approach."""
        self.log("Analyzing basic dataset information (streaming)...")
        
        # Process batches and accumulate statistics
        columns = None
        memory_usage_mb = 0
        
        for batch_df in self.load_dataset_batches(dataset_path, gene_filter, max_files, sample_rows):
            if columns is None:
                columns = list(batch_df.columns)
            
            self.update_streaming_stats(batch_df)
            
            # Estimate memory usage
            try:
                memory_usage_mb += batch_df.memory_usage(deep=True).sum() / (1024**2)
            except Exception:
                # Rough estimate
                memory_usage_mb += batch_df.size * 8 / (1024**2)
            
            # Force cleanup
            del batch_df
            gc.collect()
        
        # Compile basic info
        basic_info = {
            'dataset_shape': (self.total_samples, len(columns) if columns else 0),
            'sample_size': self.total_samples,
            'num_features': len(columns) if columns else 0,
            'has_duplicates': self.duplicate_count > 0,
            'duplicate_count': self.duplicate_count,
            'duplicate_percentage': (self.duplicate_count / self.total_samples * 100) if self.total_samples > 0 else 0,
            'estimated_memory_usage_mb': memory_usage_mb
        }
        
        # Identify key columns
        if columns:
            key_columns = {
                'splice_type_col': None,
                'gene_col': None,
                'transcript_col': None,
                'chrom_col': None,
                'position_col': None
            }
            
            for col in columns:
                col_lower = col.lower()
                if 'splice_type' in col_lower or col in ['label', 'target', 'class']:
                    key_columns['splice_type_col'] = col
                elif 'gene_id' in col_lower or col == 'gene':
                    key_columns['gene_col'] = col
                elif 'transcript_id' in col_lower or col == 'transcript':
                    key_columns['transcript_col'] = col
                elif col_lower in ['chrom', 'chromosome', 'chr']:
                    key_columns['chrom_col'] = col
                elif col_lower in ['position', 'pos', 'start']:
                    key_columns['position_col'] = col
            
            basic_info['key_columns'] = key_columns
            
            # Count unique values for key identifiers
            unique_counts = {}
            if key_columns['splice_type_col'] and key_columns['splice_type_col'] in self.categorical_stats:
                unique_counts['splice_type_col'] = len(self.categorical_stats[key_columns['splice_type_col']])
            if key_columns['gene_col'] and key_columns['gene_col'] in self.categorical_stats:
                unique_counts['gene_col'] = len(self.categorical_stats[key_columns['gene_col']])
            
            basic_info['unique_counts'] = unique_counts
        
        return basic_info
    
    def analyze_splice_types_streaming(self) -> Dict[str, Any]:
        """Analyze splice type distributions using streaming data."""
        self.log("Analyzing splice type distributions (streaming)...")
        
        if not self.splice_type_counts:
            return {'error': 'No splice_type data collected during streaming'}
        
        # Get value counts from accumulated data
        total_samples = sum(self.splice_type_counts.values())
        
        # Normalize labels (handle both string and numeric)
        normalized_counts = {}
        splice_sites = 0
        
        for label, count in self.splice_type_counts.items():
            # Convert to string for comparison
            label_str = str(label).lower()
            
            if label_str in ['donor', '1']:
                normalized_counts['donor'] = count
                splice_sites += count
            elif label_str in ['acceptor', '2']:
                normalized_counts['acceptor'] = count  
                splice_sites += count
            elif label_str in ['neither', '0']:
                normalized_counts['neither'] = count
            else:
                # Handle other labels
                normalized_counts[str(label)] = count
        
        # Calculate percentages
        percentages = {
            label: (count / total_samples) * 100 
            for label, count in normalized_counts.items()
        }
        
        # Overall splice site statistics
        splice_percentage = (splice_sites / total_samples) * 100
        neither_percentage = percentages.get('neither', 0)
        
        splice_analysis = {
            'total_samples': total_samples,
            'raw_distribution': dict(self.splice_type_counts),
            'normalized_distribution': normalized_counts,
            'percentage_distribution': percentages,
            'splice_sites_total': splice_sites,
            'splice_sites_percentage': splice_percentage,
            'neither_percentage': neither_percentage,
            'class_balance': {
                'is_balanced': min(percentages.values()) / max(percentages.values()) > 0.3,
                'balance_ratio': min(percentages.values()) / max(percentages.values()),
                'most_common': max(normalized_counts, key=normalized_counts.get),
                'least_common': min(normalized_counts, key=normalized_counts.get)
            }
        }
        
        return splice_analysis
    
    def analyze_gene_splice_sites_streaming(self, top_k: int = 20) -> Dict[str, Any]:
        """Analyze splice sites distribution by gene using streaming data."""
        self.log("Analyzing splice sites by gene (streaming)...")
        
        if not self.gene_counts:
            return {'error': 'No gene data collected during streaming'}
        
        # For streaming analysis, we'll use the gene counts directly
        # This is a simplified version that counts total occurrences per gene
        # rather than splice sites per gene (would require more complex tracking)
        
        total_genes = len(self.gene_counts)
        gene_counts_series = pd.Series(dict(self.gene_counts))
        
        # Sort by count
        gene_counts_sorted = gene_counts_series.sort_values(ascending=False)
        
        # Distribution statistics (approximate)
        splice_counts_stats = {
            'mean': float(gene_counts_series.mean()),
            'median': float(gene_counts_series.median()),
            'std': float(gene_counts_series.std()),
            'min': int(gene_counts_series.min()),
            'max': int(gene_counts_series.max()),
            'total_occurrences': int(gene_counts_series.sum())
        }
        
        # Top K genes
        top_genes = gene_counts_sorted.head(top_k)
        top_genes_dict = dict(zip(top_genes.index, top_genes.values))
        
        # Distribution by count ranges (approximate)
        count_ranges = {
            '0': 0,  # Can't detect zero counts in streaming
            '1-100': ((gene_counts_series >= 1) & (gene_counts_series <= 100)).sum(),
            '101-500': ((gene_counts_series >= 101) & (gene_counts_series <= 500)).sum(),
            '501-1000': ((gene_counts_series >= 501) & (gene_counts_series <= 1000)).sum(),
            '1001-5000': ((gene_counts_series >= 1001) & (gene_counts_series <= 5000)).sum(),
            '5000+': (gene_counts_series > 5000).sum()
        }
        
        gene_analysis = {
            'total_genes': total_genes,
            'genes_with_data': total_genes,  # All genes in our counts have data
            'genes_without_data': 0,  # Can't detect in streaming
            'percentage_genes_with_data': 100.0,
            'occurrence_count_stats': splice_counts_stats,
            'top_genes': top_genes_dict,
            'distribution_by_ranges': count_ranges,
            'note': 'Streaming analysis shows gene occurrence counts, not splice site counts per se'
        }
        
        return gene_analysis
    
    def analyze_base_model_performance_streaming(self, dataset_path: str, gene_filter: Optional[List[str]] = None, 
                                               max_files: Optional[int] = None, sample_rows: Optional[int] = None) -> Dict[str, Any]:
        """Analyze base model performance using streaming approach."""
        self.log("Analyzing base model performance (streaming)...")
        
        # We need to process the data again to compute confusion matrix
        # But we'll do it in a memory-efficient way
        
        total_samples = 0
        confusion_matrix = np.zeros((3, 3), dtype=int)  # 3x3 for neither, donor, acceptor
        
        for batch_df in self.load_dataset_batches(dataset_path, gene_filter, max_files, sample_rows):
            # Look for SpliceAI probability columns
            prob_cols = {
                'donor_score': None,
                'acceptor_score': None,
                'neither_score': None
            }
            
            for col in batch_df.columns:
                col_lower = col.lower()
                if 'donor_score' in col_lower or 'donor_prob' in col_lower:
                    prob_cols['donor_score'] = col
                elif 'acceptor_score' in col_lower or 'acceptor_prob' in col_lower:
                    prob_cols['acceptor_score'] = col
                elif 'neither_score' in col_lower or 'neither_prob' in col_lower:
                    prob_cols['neither_score'] = col
            
            # Check if we have the necessary columns
            missing_cols = [name for name, col in prob_cols.items() if col is None]
            if missing_cols:
                del batch_df
                gc.collect()
                continue
            
            # Find splice_type column
            splice_col = self._find_splice_column(batch_df)
            if not splice_col:
                del batch_df
                gc.collect()
                continue
            
            # Get base model predictions (argmax of probabilities)
            prob_matrix = batch_df[[prob_cols['neither_score'], prob_cols['donor_score'], prob_cols['acceptor_score']]].values
            base_predictions = prob_matrix.argmax(axis=1)
            
            # Encode true labels
            true_labels = _encode_labels(batch_df[splice_col])
            
            # Update confusion matrix
            for true_label, pred_label in zip(true_labels, base_predictions):
                confusion_matrix[true_label, pred_label] += 1
            
            total_samples += len(batch_df)
            
            # Cleanup
            del batch_df, prob_matrix, base_predictions, true_labels
            gc.collect()
        
        if total_samples == 0:
            return {'error': 'No valid data found for performance analysis'}
        
        # Calculate metrics from confusion matrix
        cm = confusion_matrix
        accuracy = np.trace(cm) / np.sum(cm)
        
        # Per-class metrics
        class_names = ['neither', 'donor', 'acceptor']
        error_analysis = {}
        
        # Calculate total FPs and FNs across all classes
        total_fps = 0
        total_fns = 0
        
        precision_list = []
        recall_list = []
        f1_list = []
        
        for i, class_name in enumerate(class_names):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            tn = cm.sum() - tp - fp - fn
            
            # Add to totals
            total_fps += fp
            total_fns += fn
            
            # Calculate metrics
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            precision_list.append(precision)
            recall_list.append(recall)
            f1_list.append(f1)
            
            error_analysis[class_name] = {
                'true_positives': int(tp),
                'false_positives': int(fp),
                'false_negatives': int(fn),
                'true_negatives': int(tn),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
                'support': int(tp + fn)
            }
        
        # Overall error counts
        total_errors = total_samples - np.trace(cm)
        
        # Calculate donor and acceptor specific error counts
        donor_fps = cm[0, 1] + cm[2, 1]  # Neither->Donor + Acceptor->Donor
        donor_fns = cm[1, 0] + cm[1, 2]  # Donor->Neither + Donor->Acceptor
        
        acceptor_fps = cm[0, 2] + cm[1, 2]  # Neither->Acceptor + Donor->Acceptor
        acceptor_fns = cm[2, 0] + cm[2, 1]  # Acceptor->Neither + Acceptor->Donor
        
        # Splice site errors
        splice_site_fps = donor_fps + acceptor_fps
        splice_site_fns = donor_fns + acceptor_fns
        
        performance_analysis = {
            'confusion_matrix': cm.tolist(),
            'class_names': class_names,
            'overall_accuracy': float(accuracy),
            'total_errors': int(total_errors),
            'error_rate': float(total_errors / total_samples),
            'per_class_metrics': error_analysis,
            'macro_avg': {
                'precision': float(np.mean(precision_list)),
                'recall': float(np.mean(recall_list)),
                'f1_score': float(np.mean(f1_list))
            },
            'total_error_counts': {
                'total_false_positives': int(total_fps),
                'total_false_negatives': int(total_fns),
                'total_errors': int(total_errors),
                'fp_rate': float(total_fps / total_samples),
                'fn_rate': float(total_fns / total_samples),
                'error_breakdown': {
                    'donor_false_positives': int(donor_fps),
                    'donor_false_negatives': int(donor_fns),
                    'acceptor_false_positives': int(acceptor_fps),
                    'acceptor_false_negatives': int(acceptor_fns),
                    'splice_site_false_positives': int(splice_site_fps),
                    'splice_site_false_negatives': int(splice_site_fns)
                }
            }
        }
        
        return performance_analysis
    
    def categorize_features_streaming(self, columns: List[str]) -> Dict[str, Any]:
        """Categorize features by type using column names only."""
        self.log("Categorizing features by type (streaming)...")
        
        feature_categories = {
            'splice_ai_scores': [],
            'probability_context_features': [],
            'kmer_features': [],
            'positional_features': [],
            'structural_features': [],
            'sequence_context': [],
            'genomic_annotations': [],
            'identifiers': [],
            'other': []
        }
        
        # Define probability and context-derived feature patterns
        prob_context_patterns = [
            'acceptor_context_diff_ratio', 'acceptor_diff_m1', 'acceptor_diff_m2', 'acceptor_diff_p1', 'acceptor_diff_p2',
            'acceptor_is_local_peak', 'acceptor_peak_height_ratio', 'acceptor_second_derivative', 'acceptor_signal_strength',
            'acceptor_surge_ratio', 'acceptor_weighted_context', 'context_asymmetry', 'context_max', 'context_neighbor_mean',
            'context_score_m1', 'context_score_m2', 'context_score_p1', 'context_score_p2', 'donor_acceptor_diff',
            'donor_acceptor_logodds', 'donor_acceptor_peak_ratio', 'donor_context_diff_ratio', 'donor_diff_m1',
            'donor_diff_m2', 'donor_diff_p1', 'donor_diff_p2', 'donor_is_local_peak', 'donor_peak_height_ratio',
            'donor_second_derivative', 'donor_signal_strength', 'donor_surge_ratio', 'donor_weighted_context',
            'probability_entropy', 'relative_donor_probability', 'score_difference_ratio', 'signal_strength_ratio',
            'splice_neither_diff', 'splice_neither_logodds', 'splice_probability', 'type_signal_difference', 'score'
        ]
        
        # Categorize each column
        for col in columns:
            col_lower = col.lower()
            
            # SpliceAI raw probability scores
            if any(score in col_lower for score in ['donor_score', 'acceptor_score', 'neither_score']):
                feature_categories['splice_ai_scores'].append(col)
            # Probability and context-derived features from enhanced workflow
            elif any(pattern.lower() in col_lower for pattern in prob_context_patterns):
                feature_categories['probability_context_features'].append(col)
            # K-mer features (common patterns)
            elif any(pattern in col_lower for pattern in ['6mer_', 'kmer_', '_mer_']):
                feature_categories['kmer_features'].append(col)
            # Positional features
            elif any(pos in col_lower for pos in ['position', 'distance', 'start', 'end', 'offset']):
                feature_categories['positional_features'].append(col)
            # Structural features
            elif any(struct in col_lower for struct in ['exon', 'intron', 'length', 'size', 'overlap']):
                feature_categories['structural_features'].append(col)
            # Sequence context
            elif any(seq in col_lower for seq in ['gc_content', 'complexity', 'context', 'motif']):
                feature_categories['sequence_context'].append(col)
            # Genomic annotations
            elif any(annot in col_lower for annot in ['gene_type', 'biotype', 'strand', 'chrom']):
                feature_categories['genomic_annotations'].append(col)
            # Identifiers
            elif any(id_col in col_lower for id_col in ['id', 'gene_id', 'transcript_id']):
                feature_categories['identifiers'].append(col)
            else:
                feature_categories['other'].append(col)
        
        # Count features by category
        feature_counts = {
            category: len(features) for category, features in feature_categories.items()
        }
        
        # Basic analysis for key categories
        category_analysis = {}
        
        # SpliceAI scores analysis
        if feature_categories['splice_ai_scores']:
            splice_cols = feature_categories['splice_ai_scores']
            splice_stats = {}
            for col in splice_cols:
                if col in self.streaming_stats:
                    stats = self.streaming_stats[col].get_stats()
                    splice_stats[col] = {
                        'min': stats['min'],
                        'max': stats['max'],
                        'mean': stats['mean'],
                        'std': stats['std'],
                        'count': stats['count']
                    }
            category_analysis['splice_ai_stats'] = splice_stats
        
        # K-mer features analysis
        if feature_categories['kmer_features']:
            kmer_cols = feature_categories['kmer_features']
            category_analysis['kmer_stats'] = {
                'total_kmers': len(kmer_cols),
                'sample_features': kmer_cols[:5],  # Show first 5 as examples
                'note': 'Detailed sparsity analysis requires full data loading'
            }
        
        return {
            'feature_categories': feature_categories,
            'feature_counts': feature_counts,
            'category_analysis': category_analysis,
            'total_features': len(columns)
        }
    
    def analyze_data_quality_streaming(self) -> Dict[str, Any]:
        """Analyze data quality using streaming statistics."""
        self.log("Analyzing data quality (streaming)...")
        
        quality_metrics = {}
        
        # Missing values analysis
        total_cells = self.total_samples * len(self.null_counts) if self.null_counts else 0
        total_nulls = sum(self.null_counts.values())
        
        quality_metrics['missing_values'] = {
            'total_missing': int(total_nulls),
            'percentage_missing': float(total_nulls / total_cells * 100) if total_cells > 0 else 0,
            'columns_with_missing': dict(self.null_counts),
            'complete_rows': 'Cannot determine with streaming analysis'
        }
        
        # Numeric features analysis
        if self.streaming_stats:
            # Check for extreme outliers and other quality issues
            outlier_counts = {}
            zero_variance_features = []
            
            for col, stats_obj in self.streaming_stats.items():
                stats = stats_obj.get_stats()
                
                # Check for zero variance
                if stats['std'] == 0:
                    zero_variance_features.append(col)
                
                # Note: Can't easily detect outliers without seeing all data
                # Would need to implement streaming quantile estimation
            
            quality_metrics['numeric_quality'] = {
                'infinite_values': 'Cannot detect with streaming analysis',
                'extreme_outliers': 'Cannot detect with streaming analysis - would need quantile estimation',
                'zero_variance_features': zero_variance_features
            }
        
        # Categorical features analysis
        if self.categorical_stats:
            cat_analysis = {}
            for col, value_counts in self.categorical_stats.items():
                unique_count = len(value_counts)
                most_common = max(value_counts, key=value_counts.get) if value_counts else None
                
                cat_analysis[col] = {
                    'unique_values': unique_count,
                    'is_high_cardinality': unique_count > self.total_samples * 0.1,
                    'most_common': most_common
                }
            
            quality_metrics['categorical_quality'] = cat_analysis
        
        return quality_metrics
    
    def generate_summary_statistics_streaming(self) -> Dict[str, Any]:
        """Generate summary statistics from streaming data."""
        self.log("Generating summary statistics (streaming)...")
        
        # Numeric feature summary
        numeric_summary = {}
        if self.streaming_stats:
            for col, stats_obj in self.streaming_stats.items():
                stats = stats_obj.get_stats()
                numeric_summary[col] = {
                    'count': stats['count'],
                    'mean': stats['mean'],
                    'std': stats['std'],
                    'min': stats['min'],
                    'max': stats['max'],
                    '25%': 'Not available in streaming mode',
                    '50%': 'Not available in streaming mode',
                    '75%': 'Not available in streaming mode'
                }
        
        # Key features summary (same as numeric summary for streaming)
        key_features_summary = {}
        key_patterns = ['score', 'probability', 'distance', 'length', 'gc_content']
        
        for pattern in key_patterns:
            matching_cols = [col for col in self.streaming_stats.keys() if pattern in col.lower()]
            for col in matching_cols[:5]:  # Limit to first 5 matches
                if col in self.streaming_stats:
                    stats = self.streaming_stats[col].get_stats()
                    key_features_summary[col] = {
                        'min': stats['min'],
                        'max': stats['max'],
                        'mean': stats['mean'],
                        'median': 'Not available in streaming mode',
                        'std': stats['std']
                    }
        
        return {
            'numeric_summary': numeric_summary,
            'key_features_summary': key_features_summary,
            'total_samples_processed': self.total_samples,
            'note': 'Streaming analysis - percentiles not available'
        }
    
    def analyze_gene_manifest_consistency(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze consistency between gene manifest and training data."""
        self.log("Analyzing gene manifest consistency...")
        
        if self.gene_manifest is None:
            return {'error': 'No gene manifest loaded'}
        
        # Find gene column in training data
        gene_col = None
        for col in df.columns:
            if 'gene_id' in col.lower():
                gene_col = col
                break
        
        if not gene_col:
            return {'error': 'No gene_id column found in training data'}
        
        # Get unique genes from both sources
        manifest_genes = set(self.gene_manifest['gene_id'].unique())
        training_genes = set(df[gene_col].unique())
        
        # Analyze overlap
        common_genes = manifest_genes.intersection(training_genes)
        manifest_only = manifest_genes - training_genes
        training_only = training_genes - manifest_genes
        
        consistency_analysis = {
            'manifest_genes_count': len(manifest_genes),
            'training_genes_count': len(training_genes),
            'common_genes_count': len(common_genes),
            'manifest_only_count': len(manifest_only),
            'training_only_count': len(training_only),
            'consistency_percentage': len(common_genes) / len(manifest_genes) * 100 if manifest_genes else 0,
            'manifest_only_genes': list(manifest_only)[:10],  # First 10 for display
            'training_only_genes': list(training_only)[:10],  # First 10 for display
            'total_manifest_genes': len(manifest_genes),
            'total_training_genes': len(training_genes)
        }
        
        # Analyze file distribution in manifest
        if 'file_name' in self.gene_manifest.columns:
            file_distribution = self.gene_manifest['file_name'].value_counts()
            consistency_analysis['file_distribution'] = file_distribution.to_dict()
            consistency_analysis['total_files'] = len(file_distribution)
        
        return consistency_analysis
    
    def analyze_artifacts(self, dataset_path: str) -> Dict[str, Any]:
        """Analyze training artifacts (analysis_sequences, splice_positions_enhanced)."""
        self.log("Analyzing training artifacts...")
        
        # Find artifact directory
        artifact_dir = Path("data/ensembl/spliceai_eval/meta_models")
        if not artifact_dir.exists():
            return {'error': f'Artifact directory not found: {artifact_dir}'}
        
        artifact_analysis = {
            'artifact_directory': str(artifact_dir),
            'analysis_sequences_files': [],
            'splice_positions_files': [],
            'splice_errors_files': [],
            'total_files': 0,
            'file_sizes': {},
            'chunk_analysis': {}
        }
        
        # Find all artifact files
        analysis_files = list(artifact_dir.glob("analysis_sequences_*.tsv"))
        position_files = list(artifact_dir.glob("splice_positions_enhanced_*.tsv"))
        error_files = list(artifact_dir.glob("splice_errors_*.tsv"))
        
        artifact_analysis['analysis_sequences_files'] = [f.name for f in analysis_files]
        artifact_analysis['splice_positions_files'] = [f.name for f in position_files]
        artifact_analysis['splice_errors_files'] = [f.name for f in error_files]
        artifact_analysis['total_files'] = len(analysis_files) + len(position_files) + len(error_files)
        
        # Analyze file sizes
        all_files = analysis_files + position_files + error_files
        for file_path in all_files:
            size_mb = file_path.stat().st_size / (1024 * 1024)
            artifact_analysis['file_sizes'][file_path.name] = round(size_mb, 2)
        
        # Analyze chunks
        chunk_patterns = {}
        for file_path in all_files:
            filename = file_path.name
            # Extract chunk information (e.g., "1_chunk_1_500" -> chunk 1, range 1-500)
            if '_chunk_' in filename:
                parts = filename.split('_chunk_')
                if len(parts) == 2:
                    chunk_id = parts[0].split('_')[-1]  # Extract chunk number
                    range_part = parts[1].split('.')[0]  # Remove extension
                    if '_' in range_part:
                        start, end = range_part.split('_')
                        chunk_key = f"chunk_{chunk_id}"
                        if chunk_key not in chunk_patterns:
                            chunk_patterns[chunk_key] = {
                                'files': [],
                                'ranges': [],
                                'total_size_mb': 0
                            }
                        chunk_patterns[chunk_key]['files'].append(filename)
                        chunk_patterns[chunk_key]['ranges'].append(f"{start}-{end}")
                        chunk_patterns[chunk_key]['total_size_mb'] += artifact_analysis['file_sizes'].get(filename, 0)
        
        artifact_analysis['chunk_analysis'] = chunk_patterns
        
        # Sample analysis of one file to understand structure
        if analysis_files:
            sample_file = analysis_files[0]
            try:
                sample_df = pd.read_csv(sample_file, sep='\t', nrows=1000)  # Sample first 1000 rows
                artifact_analysis['sample_analysis'] = {
                    'columns': list(sample_df.columns),
                    'sample_rows': len(sample_df),
                    'sample_genes': sample_df['gene_id'].nunique() if 'gene_id' in sample_df.columns else 0,
                    'sample_transcripts': sample_df['transcript_id'].nunique() if 'transcript_id' in sample_df.columns else 0
                }
            except Exception as e:
                artifact_analysis['sample_analysis'] = {'error': str(e)}
        
        return artifact_analysis
    
    def analyze_unseen_positions(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze unseen positions using gene boundaries and training data positions."""
        self.log("Analyzing unseen positions...")
        
        if self.gene_features is None:
            return {'error': 'No gene features loaded for unseen position analysis'}
        
        # Find required columns
        gene_col = None
        position_col = None
        strand_col = None
        
        for col in df.columns:
            col_lower = col.lower()
            if 'gene_id' in col_lower:
                gene_col = col
            elif col_lower in ['position', 'pos']:
                position_col = col
            elif col_lower == 'strand':
                strand_col = col
        
        if not gene_col or not position_col:
            return {'error': f'Missing required columns. gene_col: {gene_col}, position_col: {position_col}'}
        
        unseen_analysis = {
            'total_genes_analyzed': 0,
            'genes_with_unseen_positions': 0,
            'total_positions_analyzed': 0,
            'total_unseen_positions': 0,
            'unseen_percentage': 0.0,
            'gene_level_analysis': {},
            'position_coverage_stats': {}
        }
        
        # Analyze each gene
        gene_groups = df.groupby(gene_col)
        gene_analysis = {}
        
        for gene_id, gene_data in gene_groups:
            if gene_id not in self.gene_features['gene_id'].values:
                continue  # Skip genes not in gene features
            
            gene_feature = self.gene_features[self.gene_features['gene_id'] == gene_id].iloc[0]
            gene_length = gene_feature.get('gene_len', 0)
            gene_start = gene_feature.get('start', 0)
            gene_end = gene_feature.get('end', 0)
            gene_strand = gene_feature.get('strand', '+')
            
            if gene_length == 0:
                continue
            
            # Get positions in training data for this gene
            training_positions = set(gene_data[position_col].unique())
            
            # Calculate all possible positions in the gene
            all_positions = set(range(1, gene_length + 1))  # 1-based positions
            unseen_positions = all_positions - training_positions
            
            # Calculate coverage statistics
            coverage_percentage = len(training_positions) / len(all_positions) * 100
            unseen_percentage = len(unseen_positions) / len(all_positions) * 100
            
            gene_analysis[gene_id] = {
                'gene_length': gene_length,
                'training_positions_count': len(training_positions),
                'unseen_positions_count': len(unseen_positions),
                'coverage_percentage': coverage_percentage,
                'unseen_percentage': unseen_percentage,
                'gene_start': gene_start,
                'gene_end': gene_end,
                'gene_strand': gene_strand
            }
            
            unseen_analysis['total_genes_analyzed'] += 1
            unseen_analysis['total_positions_analyzed'] += gene_length
            unseen_analysis['total_unseen_positions'] += len(unseen_positions)
            
            if len(unseen_positions) > 0:
                unseen_analysis['genes_with_unseen_positions'] += 1
        
        # Calculate overall statistics
        if unseen_analysis['total_positions_analyzed'] > 0:
            unseen_analysis['unseen_percentage'] = (
                unseen_analysis['total_unseen_positions'] / 
                unseen_analysis['total_positions_analyzed'] * 100
            )
        
        # Position coverage statistics
        coverage_percentages = [g['coverage_percentage'] for g in gene_analysis.values()]
        unseen_percentages = [g['unseen_percentage'] for g in gene_analysis.values()]
        
        if coverage_percentages:
            unseen_analysis['position_coverage_stats'] = {
                'mean_coverage': np.mean(coverage_percentages),
                'median_coverage': np.median(coverage_percentages),
                'std_coverage': np.std(coverage_percentages),
                'mean_unseen': np.mean(unseen_percentages),
                'median_unseen': np.median(unseen_percentages),
                'std_unseen': np.std(unseen_percentages),
                'min_coverage': np.min(coverage_percentages),
                'max_coverage': np.max(coverage_percentages)
            }
        
        # Top genes with most unseen positions
        genes_by_unseen = sorted(
            gene_analysis.items(), 
            key=lambda x: x[1]['unseen_positions_count'], 
            reverse=True
        )
        
        unseen_analysis['top_genes_unseen'] = {
            gene_id: {
                'unseen_count': data['unseen_positions_count'],
                'unseen_percentage': data['unseen_percentage'],
                'gene_length': data['gene_length'],
                'display_name': self.get_gene_display_name(gene_id)
            }
            for gene_id, data in genes_by_unseen[:10]  # Top 10
        }
        
        # Top genes with least coverage
        genes_by_coverage = sorted(
            gene_analysis.items(), 
            key=lambda x: x[1]['coverage_percentage']
        )
        
        unseen_analysis['top_genes_low_coverage'] = {
            gene_id: {
                'coverage_percentage': data['coverage_percentage'],
                'unseen_count': data['unseen_positions_count'],
                'gene_length': data['gene_length'],
                'display_name': self.get_gene_display_name(gene_id)
            }
            for gene_id, data in genes_by_coverage[:10]  # Top 10
        }
        
        unseen_analysis['gene_level_analysis'] = gene_analysis
        
        return unseen_analysis
    
    def profile_dataset(self, dataset_path: str, output_dir: Optional[str] = None, 
                       generate_plots: bool = False, gene_filter: Optional[List[str]] = None,
                       max_files: Optional[int] = None, sample_rows: Optional[int] = None) -> Dict[str, Any]:
        """Generate comprehensive dataset profile using streaming approach for large datasets."""
        self.log("Starting comprehensive dataset profiling (streaming mode)...")
        
        # Load gene features for name mapping
        self.load_gene_features()
        
        # Load gene manifest
        self.load_gene_manifest(dataset_path)
        
        # Start with basic info which will populate streaming stats
        profile = {
            'dataset_path': dataset_path,
            'timestamp': pd.Timestamp.now().isoformat(),
            'basic_info': self.analyze_basic_info_streaming(dataset_path, gene_filter, max_files, sample_rows),
        }
        
        # Get columns from the first batch for feature analysis
        columns = None
        first_batch = next(iter(self.load_dataset_batches(dataset_path, gene_filter, max_files, sample_rows)))
        if first_batch is not None:
            columns = list(first_batch.columns)
            del first_batch
            gc.collect()
        
        # Splice type analysis
        profile['splice_analysis'] = self.analyze_splice_types_streaming()
        
        # Gene-level analysis
        profile['gene_splice_analysis'] = self.analyze_gene_splice_sites_streaming(top_k=20)
        
        # Base model performance analysis
        profile['base_model_performance'] = self.analyze_base_model_performance_streaming(dataset_path, gene_filter, max_files, sample_rows)
        
        # Feature analysis
        if columns:
            profile['feature_analysis'] = self.categorize_features_streaming(columns)
        
        # Data quality analysis
        profile['quality_analysis'] = self.analyze_data_quality_streaming()
        
        # Summary statistics
        profile['summary_statistics'] = self.generate_summary_statistics_streaming()
        
        # Gene manifest validation (still requires loading gene data but much smaller)
        if self.gene_manifest is not None:
            # Create a minimal DataFrame with just gene_id for manifest comparison
            gene_ids_df = pd.DataFrame({'gene_id': list(self.gene_counts.keys())})
            profile['gene_manifest_analysis'] = self.analyze_gene_manifest_consistency(gene_ids_df)
            del gene_ids_df
            gc.collect()
        
        # Artifact analysis (filesystem-based, no data loading)
        profile['artifact_analysis'] = self.analyze_artifacts(dataset_path)
        
        # Unseen position analysis (requires gene features, skip if too memory intensive)
        if self.gene_features is not None and len(self.gene_counts) < 5000:  # Only for smaller gene sets
            try:
                # Create minimal DataFrame for unseen analysis
                positions_data = []
                gene_col = None
                position_col = None
                
                # Sample some data for unseen analysis
                sample_count = 0
                max_samples = 50000  # Limit sample size
                
                for batch_df in self.load_dataset_batches(dataset_path, gene_filter, max_files, sample_rows):
                    if gene_col is None:
                        gene_col = self._find_gene_column(batch_df)
                    if position_col is None:
                        for col in batch_df.columns:
                            if col.lower() in ['position', 'pos']:
                                position_col = col
                                break
                    
                    if gene_col and position_col:
                        batch_sample = batch_df[[gene_col, position_col]].copy()
                        positions_data.append(batch_sample)
                        sample_count += len(batch_sample)
                        
                        if sample_count >= max_samples:
                            break
                    
                    del batch_df
                    gc.collect()
                
                if positions_data:
                    positions_df = pd.concat(positions_data, ignore_index=True)
                    profile['unseen_position_analysis'] = self.analyze_unseen_positions(positions_df)
                    del positions_df, positions_data
                    gc.collect()
                else:
                    profile['unseen_position_analysis'] = {'note': 'Could not perform unseen position analysis - missing required columns'}
            except Exception as e:
                self.log(f"Error in unseen position analysis: {e}", "WARNING")
                profile['unseen_position_analysis'] = {'error': f'Unseen position analysis failed: {e}'}
        else:
            profile['unseen_position_analysis'] = {'note': 'Skipped unseen position analysis for large datasets or missing gene features'}
        
        # Generate recommendations
        profile['recommendations'] = self.generate_recommendations(profile)
        
        # Create visualizations if requested (using streaming-friendly approach)
        if generate_plots and output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            profile['visualizations'] = self.create_visualizations_streaming(output_path, profile)
        
        self.log("Dataset profiling completed successfully!")
        
        return profile
    
    def create_visualizations_streaming(self, output_dir: Path, profile: Dict[str, Any] = None) -> Dict[str, str]:
        """Create visualizations using streaming data and profile statistics."""
        self.log("Creating visualizations (streaming mode)...")
        
        vis_dir = output_dir / "visualizations"
        vis_dir.mkdir(exist_ok=True)
        
        plot_files = {}
        
        # Set style for better-looking plots
        plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
        
        try:
            # 1. Splice type distribution from streaming stats
            splice_analysis = profile.get('splice_analysis', {})
            if 'normalized_distribution' in splice_analysis:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                
                labels = list(splice_analysis['normalized_distribution'].keys())
                values = list(splice_analysis['normalized_distribution'].values())
                
                # Pie chart
                ax1.pie(values, labels=labels, autopct='%1.1f%%', startangle=90)
                ax1.set_title('Splice Type Distribution')
                
                # Bar chart
                ax2.bar(labels, values, color=['skyblue', 'lightcoral', 'lightgreen'])
                ax2.set_title('Splice Type Counts')
                ax2.set_xlabel('Splice Type')
                ax2.set_ylabel('Count')
                ax2.tick_params(axis='x', rotation=45)
                
                plt.tight_layout()
                plot_file = vis_dir / "splice_type_distribution.png"
                plt.savefig(plot_file, dpi=300, bbox_inches='tight')
                plt.close()
                plot_files['splice_distribution'] = str(plot_file)
            
            # 2. SpliceAI scores statistics - improved interpretation
            feature_analysis = profile.get('feature_analysis', {})
            splice_stats = feature_analysis.get('category_analysis', {}).get('splice_ai_stats', {})
            
            if splice_stats:
                fig, axes = plt.subplots(2, 2, figsize=(15, 10))
                axes = axes.flatten()
                
                # Get splice type distribution for context
                splice_analysis = profile.get('splice_analysis', {})
                splice_dist = splice_analysis.get('normalized_distribution', {})
                
                # 1. Mean scores by splice type (what matters)
                if splice_dist:
                    donor_count = splice_dist.get('donor', 0)
                    acceptor_count = splice_dist.get('acceptor', 0)
                    neither_count = splice_dist.get('neither', 0)
                    
                    # Approximate type-specific means (simplified visualization)
                    type_specific_means = {
                        'Donor Sites\n(donor_score)': splice_stats.get('donor_score', {}).get('mean', 0),
                        'Acceptor Sites\n(acceptor_score)': splice_stats.get('acceptor_score', {}).get('mean', 0),
                        'Non-splice Sites\n(neither_score)': splice_stats.get('neither_score', {}).get('mean', 0)
                    }
                    
                    axes[0].bar(type_specific_means.keys(), type_specific_means.values(), 
                               alpha=0.7, color=['skyblue', 'lightcoral', 'lightgreen'])
                    axes[0].set_title('Mean Scores by Site Type\n(Relevant Score for Each Type)')
                    axes[0].set_ylabel('Mean Score')
                    axes[0].tick_params(axis='x', rotation=0)
                    
                    for i, (k, v) in enumerate(type_specific_means.items()):
                        axes[0].text(i, v + max(type_specific_means.values())*0.01, f'{v:.3f}', ha='center', va='bottom')
                
                # 2. Median scores
                score_types = ['donor_score', 'acceptor_score', 'neither_score']
                medians = [splice_stats.get(score_type, {}).get('median', 0) for score_type in score_types]
                axes[1].bar(score_types, medians, alpha=0.7, color=['skyblue', 'lightcoral', 'lightgreen'])
                axes[1].set_title('Median Scores by Type')
                axes[1].set_ylabel('Median Score')
                axes[1].tick_params(axis='x', rotation=45)
                
                for i, v in enumerate(medians):
                    axes[1].text(i, v + max(medians)*0.01, f'{v:.3f}', ha='center', va='bottom')
                
                # 3. Standard deviation (variability)
                stds = [splice_stats.get(score_type, {}).get('std', 0) for score_type in score_types]
                axes[2].bar(score_types, stds, alpha=0.7, color=['skyblue', 'lightcoral', 'lightgreen'])
                axes[2].set_title('Score Variability (Std Dev)')
                axes[2].set_ylabel('Standard Deviation')
                axes[2].tick_params(axis='x', rotation=45)
                
                for i, v in enumerate(stds):
                    axes[2].text(i, v + max(stds)*0.01, f'{v:.3f}', ha='center', va='bottom')
                
                # 4. Score ranges (max - min)
                ranges = [splice_stats.get(score_type, {}).get('max', 0) - splice_stats.get(score_type, {}).get('min', 0) 
                         for score_type in score_types]
                axes[3].bar(score_types, ranges, alpha=0.7, color=['skyblue', 'lightcoral', 'lightgreen'])
                axes[3].set_title('Score Ranges (Max - Min)')
                axes[3].set_ylabel('Range')
                axes[3].tick_params(axis='x', rotation=45)
                
                for i, v in enumerate(ranges):
                    axes[3].text(i, v + max(ranges)*0.01, f'{v:.3f}', ha='center', va='bottom')
                
                plt.tight_layout()
                plot_file = vis_dir / "spliceai_scores_stats.png"
                plt.savefig(plot_file, dpi=300, bbox_inches='tight')
                plt.close()
                plot_files['scores_stats'] = str(plot_file)
            
            # 3. Missing values analysis - improved interpretation
            quality_analysis = profile.get('quality_analysis', {})
            missing_data = quality_analysis.get('missing_values', {}).get('columns_with_missing', {})
            
            if missing_data:
                # Sort by missing count and limit to top problematic columns
                sorted_missing = sorted(missing_data.items(), key=lambda x: x[1], reverse=True)
                top_missing = sorted_missing[:20]  # Show top 20 columns with missing values
                
                if top_missing:
                    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
                    
                    # Top plot: Missing value counts
                    cols, counts = zip(*top_missing)
                    total_samples = profile.get('basic_info', {}).get('sample_size', 1)
                    percentages = [count/total_samples*100 for count in counts]
                    
                    bars = ax1.bar(range(len(cols)), counts, alpha=0.7, color='red')
                    ax1.set_xticks(range(len(cols)))
                    ax1.set_xticklabels(cols, rotation=45, ha='right')
                    ax1.set_ylabel('Missing Value Count')
                    ax1.set_title(f'Top {len(top_missing)} Columns with Missing Values')
                    
                    # Add percentage labels
                    for i, (count, pct) in enumerate(zip(counts, percentages)):
                        ax1.text(i, count + max(counts)*0.01, f'{pct:.1f}%', ha='center', va='bottom', fontsize=8)
                    
                    # Bottom plot: Percentage of missing values
                    bars2 = ax2.bar(range(len(cols)), percentages, alpha=0.7, color='orange')
                    ax2.set_xticks(range(len(cols)))
                    ax2.set_xticklabels(cols, rotation=45, ha='right')
                    ax2.set_ylabel('Missing Percentage (%)')
                    ax2.set_title('Percentage of Missing Values by Column')
                    ax2.axhline(y=5, color='red', linestyle='--', alpha=0.7, label='5% threshold')
                    ax2.axhline(y=10, color='darkred', linestyle='--', alpha=0.7, label='10% threshold')
                    ax2.legend()
                    
                    plt.tight_layout()
                    plot_file = vis_dir / "missing_values_by_column.png"
                    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
                    plt.close()
                    plot_files['missing_values'] = str(plot_file)
                    
                    # Generate missing values summary
                    missing_summary = {
                        'total_columns_with_missing': len(missing_data),
                        'total_missing_values': sum(missing_data.values()),
                        'columns_over_5_percent': sum(1 for count in missing_data.values() if count/total_samples > 0.05),
                        'columns_over_10_percent': sum(1 for count in missing_data.values() if count/total_samples > 0.10),
                        'worst_column': max(missing_data.items(), key=lambda x: x[1]) if missing_data else None,
                        'average_missing_percentage': sum(count/total_samples*100 for count in missing_data.values()) / len(missing_data)
                    }
                    
                    # Save summary to profile for reporting
                    if 'quality_analysis' not in profile:
                        profile['quality_analysis'] = {}
                    profile['quality_analysis']['missing_values_summary'] = missing_summary
            
            # 4. Top genes chart - clarified interpretation
            gene_analysis = profile.get('gene_splice_analysis', {})
            if 'top_genes' in gene_analysis and gene_analysis['top_genes']:
                plt.figure(figsize=(14, 8))
                
                top_genes = gene_analysis['top_genes']
                gene_ids = list(top_genes.keys())[:15]  # Limit to top 15 for readability
                counts = [top_genes[gid] for gid in gene_ids]
                
                # Get display names for genes
                gene_display_names = [self.get_gene_display_name(gene_id) for gene_id in gene_ids]
                
                # Create horizontal bar chart
                y_pos = np.arange(len(gene_ids))
                plt.barh(y_pos, counts, color='steelblue', alpha=0.7)
                plt.yticks(y_pos, gene_display_names)
                plt.xlabel('Number of Training Samples')
                plt.title(f'Top {len(gene_ids)} Genes by Training Sample Count\n(Genes with most splice site positions in training data)')
                plt.gca().invert_yaxis()
                
                # Add value labels and percentage of total
                total_samples = sum(counts)
                for i, v in enumerate(counts):
                    pct = v/total_samples*100 if total_samples > 0 else 0
                    plt.text(v + max(counts)*0.01, i, f'{v:,} ({pct:.1f}%)', va='center', fontsize=9)
                
                plt.tight_layout()
                plot_file = vis_dir / "top_genes_occurrence.png"
                plt.savefig(plot_file, dpi=300, bbox_inches='tight')
                plt.close()
                plot_files['top_genes'] = str(plot_file)
            
            # 5. Feature category distribution - improved labels
            feature_counts = feature_analysis.get('feature_counts', {})
            if feature_counts:
                plt.figure(figsize=(14, 8))
                
                # Better category display names
                category_display_names = {
                    'splice_ai_scores': 'SpliceAI\nRaw Scores',
                    'probability_context_features': 'Probability &\nContext Features',
                    'kmer_features': 'K-mer\nFeatures',
                    'positional_features': 'Positional\nFeatures',
                    'structural_features': 'Structural\nFeatures',
                    'sequence_context': 'Sequence\nContext',
                    'genomic_annotations': 'Genomic\nAnnotations',
                    'identifiers': 'Identifiers',
                    'other': 'Other\nFeatures'
                }
                
                # Only show categories with > 0 features
                non_zero_categories = [(cat, count) for cat, count in feature_counts.items() if count > 0]
                if non_zero_categories:
                    categories, counts = zip(*non_zero_categories)
                    display_names = [category_display_names.get(cat, cat.replace('_', ' ').title()) for cat in categories]
                    
                    # Use different colors for different types
                    colors = ['red', 'orange', 'gold', 'lightgreen', 'lightblue', 'purple', 'pink', 'gray', 'brown']
                    bar_colors = [colors[i % len(colors)] for i in range(len(categories))]
                    
                    plt.bar(display_names, counts, alpha=0.7, color=bar_colors)
                    plt.xticks(rotation=45, ha='right')
                    plt.ylabel('Number of Features')
                    plt.title('Feature Distribution by Category\n(Enhanced workflow features now properly categorized)')
                    
                    # Add value labels
                    for i, v in enumerate(counts):
                        plt.text(i, v + max(counts)*0.01, str(v), ha='center', va='bottom', fontweight='bold')
                    
                    plt.tight_layout()
                    plot_file = vis_dir / "feature_categories.png"
                    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
                    plt.close()
                    plot_files['feature_categories'] = str(plot_file)
            
            # 6. Gene manifest consistency (if available)
            manifest_analysis = profile.get('gene_manifest_analysis', {})
            if manifest_analysis and 'total_manifest_genes' in manifest_analysis:
                plt.figure(figsize=(10, 6))
                
                manifest_count = manifest_analysis.get('total_manifest_genes', 0)
                training_count = manifest_analysis.get('total_training_genes', 0)
                common_count = manifest_analysis.get('common_genes_count', 0)
                
                categories = ['Manifest Only', 'Common', 'Training Only']
                counts = [
                    manifest_count - common_count,
                    common_count,
                    training_count - common_count
                ]
                
                colors = ['lightcoral', 'lightgreen', 'lightblue']
                plt.bar(categories, counts, color=colors, alpha=0.7)
                plt.title('Gene Manifest vs Training Data Consistency')
                plt.ylabel('Number of Genes')
                
                # Add value labels
                for i, v in enumerate(counts):
                    plt.text(i, v + max(counts)*0.01, str(v), ha='center', va='bottom')
                
                plt.tight_layout()
                plot_file = vis_dir / "gene_manifest_consistency.png"
                plt.savefig(plot_file, dpi=300, bbox_inches='tight')
                plt.close()
                plot_files['manifest_consistency'] = str(plot_file)
            
            # 7. Performance metrics visualization
            performance = profile.get('base_model_performance', {})
            if 'per_class_metrics' in performance:
                metrics_data = performance['per_class_metrics']
                classes = list(metrics_data.keys())
                
                # Create subplot for precision, recall, f1
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                
                metrics = ['precision', 'recall', 'f1_score']
                metric_names = ['Precision', 'Recall', 'F1-Score']
                
                for i, (metric, name) in enumerate(zip(metrics, metric_names)):
                    values = [metrics_data[cls][metric] for cls in classes]
                    axes[i].bar(classes, values, alpha=0.7, color=['skyblue', 'lightcoral', 'lightgreen'])
                    axes[i].set_title(f'{name} by Class')
                    axes[i].set_ylabel(name)
                    axes[i].set_ylim(0, 1)
                    
                    # Add value labels
                    for j, v in enumerate(values):
                        axes[i].text(j, v + 0.02, f'{v:.3f}', ha='center', va='bottom')
                
                plt.tight_layout()
                plot_file = vis_dir / "performance_metrics.png"
                plt.savefig(plot_file, dpi=300, bbox_inches='tight')
                plt.close()
                plot_files['performance_metrics'] = str(plot_file)
            
            self.log(f"Created {len(plot_files)} visualization plots (streaming mode)")
            
        except Exception as e:
            self.log(f"Error creating visualizations: {e}", "WARNING")
        
        return plot_files
    
    def print_summary(self, profile: Dict[str, Any]):
        """Print a concise summary of the dataset profile to console."""
        print("\n" + "="*80)
        print("🔬 SPLICE SITE DATASET PROFILE SUMMARY")
        print("="*80)
        
        # Basic info
        basic_info = profile.get('basic_info', {})
        dataset_shape = basic_info.get('dataset_shape', (0, 0))
        print(f"\n📊 Dataset Overview:")
        print(f"   • Path: {profile.get('dataset_path', 'Unknown')}")
        print(f"   • Size: {dataset_shape[0]:,} samples × {dataset_shape[1]:,} features")
        
        memory_mb = basic_info.get('estimated_memory_usage_mb', 0)
        if memory_mb > 0:
            print(f"   • Memory usage: {memory_mb:.1f} MB")
        
        duplicate_count = basic_info.get('duplicate_count', 0)
        if duplicate_count > 0:
            duplicate_pct = basic_info.get('duplicate_percentage', 0)
            print(f"   • Duplicates: {duplicate_count:,} ({duplicate_pct:.2f}%)")
        
        # Splice type distribution
        splice_analysis = profile.get('splice_analysis', {})
        if 'normalized_distribution' in splice_analysis:
            print(f"\n🧬 Splice Site Distribution:")
            dist = splice_analysis['normalized_distribution']
            pct_dist = splice_analysis.get('percentage_distribution', {})
            
            for splice_type, count in dist.items():
                pct = pct_dist.get(splice_type, 0)
                print(f"   • {splice_type.capitalize()}: {count:,} ({pct:.1f}%)")
            
            # Class balance
            balance_info = splice_analysis.get('class_balance', {})
            if balance_info:
                balance_ratio = balance_info.get('balance_ratio', 1.0)
                is_balanced = balance_info.get('is_balanced', True)
                status = "✅ Balanced" if is_balanced else "⚠️  Imbalanced"
                print(f"   • Class balance: {status} (ratio: {balance_ratio:.3f})")
        
        # Gene analysis
        gene_analysis = profile.get('gene_splice_analysis', {})
        if 'total_genes' in gene_analysis:
            total_genes = gene_analysis['total_genes']
            print(f"\n🧬 Gene Coverage:")
            print(f"   • Total genes: {total_genes:,}")
            
            stats = gene_analysis.get('occurrence_count_stats', {})
            if stats:
                mean_count = stats.get('mean', 0)
                median_count = stats.get('median', 0)
                max_count = stats.get('max', 0)
                print(f"   • Avg occurrences per gene: {mean_count:.1f}")
                print(f"   • Median occurrences: {median_count:.0f}")
                print(f"   • Max occurrences: {max_count:,}")
        
        # Performance metrics
        performance = profile.get('base_model_performance', {})
        if 'overall_accuracy' in performance:
            accuracy = performance['overall_accuracy']
            error_rate = performance.get('error_rate', 0)
            print(f"\n📈 Base Model Performance:")
            print(f"   • Overall accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
            print(f"   • Error rate: {error_rate:.3f} ({error_rate*100:.1f}%)")
            
            # Error breakdown
            error_counts = performance.get('total_error_counts', {})
            if error_counts:
                total_fps = error_counts.get('total_false_positives', 0)
                total_fns = error_counts.get('total_false_negatives', 0)
                fp_rate = error_counts.get('fp_rate', 0)
                fn_rate = error_counts.get('fn_rate', 0)
                print(f"   • False positives: {total_fps:,} ({fp_rate*100:.2f}%)")
                print(f"   • False negatives: {total_fns:,} ({fn_rate*100:.2f}%)")
        
        # Feature analysis
        feature_analysis = profile.get('feature_analysis', {})
        if 'feature_counts' in feature_analysis:
            feature_counts = feature_analysis['feature_counts']
            print(f"\n🔧 Feature Categories:")
            
            # Show non-zero categories with better names
            category_display_names = {
                'splice_ai_scores': 'SpliceAI Raw Scores',
                'probability_context_features': 'Probability & Context Features',
                'kmer_features': 'K-mer Features',
                'positional_features': 'Positional Features',
                'structural_features': 'Structural Features',
                'sequence_context': 'Sequence Context',
                'genomic_annotations': 'Genomic Annotations',
                'identifiers': 'Identifiers',
                'other': 'Other Features'
            }
            
            for category, count in feature_counts.items():
                if count > 0:
                    display_name = category_display_names.get(category, category.replace('_', ' ').title())
                    print(f"   • {display_name}: {count:,}")
        
        # Data quality
        quality_analysis = profile.get('quality_analysis', {})
        missing_data = quality_analysis.get('missing_values', {})
        missing_summary = quality_analysis.get('missing_values_summary', {})
        
        if missing_data or missing_summary:
            total_missing = missing_data.get('total_missing', 0)
            pct_missing = missing_data.get('percentage_missing', 0)
            if total_missing > 0 or missing_summary:
                print(f"\n⚠️  Data Quality Issues:")
                if total_missing > 0:
                    print(f"   • Missing values: {total_missing:,} ({pct_missing:.2f}%)")
                
                # Enhanced missing values summary
                if missing_summary:
                    total_cols_missing = missing_summary.get('total_columns_with_missing', 0)
                    cols_over_5pct = missing_summary.get('columns_over_5_percent', 0)
                    cols_over_10pct = missing_summary.get('columns_over_10_percent', 0)
                    worst_column = missing_summary.get('worst_column', None)
                    avg_missing_pct = missing_summary.get('average_missing_percentage', 0)
                    
                    print(f"   • Columns with missing values: {total_cols_missing:,}")
                    if cols_over_5pct > 0:
                        print(f"   • Columns >5% missing: {cols_over_5pct:,}")
                    if cols_over_10pct > 0:
                        print(f"   • Columns >10% missing: {cols_over_10pct:,}")
                    if worst_column:
                        worst_col_name, worst_col_count = worst_column
                        dataset_size = profile.get('basic_info', {}).get('sample_size', 1)
                        worst_pct = worst_col_count/dataset_size*100 if dataset_size > 0 else 0
                        print(f"   • Worst column: '{worst_col_name}' ({worst_col_count:,} missing, {worst_pct:.1f}%)")
                    if avg_missing_pct > 0:
                        print(f"   • Average missing %: {avg_missing_pct:.2f}%")
        
        # Gene manifest consistency
        manifest_analysis = profile.get('gene_manifest_analysis', {})
        if manifest_analysis and 'consistency_percentage' in manifest_analysis:
            consistency_pct = manifest_analysis['consistency_percentage']
            manifest_genes = manifest_analysis.get('total_manifest_genes', 0)
            training_genes = manifest_analysis.get('total_training_genes', 0)
            common_genes = manifest_analysis.get('common_genes_count', 0)
            
            print(f"\n📋 Gene Manifest Consistency:")
            print(f"   • Manifest genes: {manifest_genes:,}")
            print(f"   • Training genes: {training_genes:,}")
            print(f"   • Common genes: {common_genes:,}")
            print(f"   • Consistency: {consistency_pct:.1f}%")
        
        # Unseen positions
        unseen_analysis = profile.get('unseen_position_analysis', {})
        if unseen_analysis and 'unseen_percentage' in unseen_analysis:
            unseen_pct = unseen_analysis['unseen_percentage']
            total_positions = unseen_analysis.get('total_positions_analyzed', 0)
            unseen_positions = unseen_analysis.get('total_unseen_positions', 0)
            
            print(f"\n🔍 Position Coverage:")
            print(f"   • Total positions analyzed: {total_positions:,}")
            print(f"   • Unseen positions: {unseen_positions:,} ({unseen_pct:.1f}%)")
            
            coverage_stats = unseen_analysis.get('position_coverage_stats', {})
            if coverage_stats:
                mean_coverage = coverage_stats.get('mean_coverage', 0)
                print(f"   • Average gene coverage: {mean_coverage:.1f}%")
        
        # Visualizations
        visualizations = profile.get('visualizations', {})
        if visualizations:
            print(f"\n🎨 Generated Visualizations:")
            for plot_name, plot_path in visualizations.items():
                plot_display_name = plot_name.replace('_', ' ').title()
                print(f"   • {plot_display_name}: {plot_path}")
        
        # Recommendations
        recommendations = profile.get('recommendations', [])
        if recommendations:
            print(f"\n💡 Recommendations:")
            for i, rec in enumerate(recommendations[:8], 1):  # Show first 8
                print(f"   {i}. {rec}")
            
            if len(recommendations) > 8:
                print(f"   ... and {len(recommendations) - 8} more recommendations")
        
        print("\n" + "="*80)
        print("✅ Profile completed successfully!")
        print("="*80 + "\n")

    def generate_recommendations(self, profile: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on the dataset profile."""
        recommendations = []
        
        # Data size recommendations
        sample_size = profile.get('basic_info', {}).get('sample_size', 0)
        if sample_size < 10000:
            recommendations.append("⚠️  Small dataset detected (<10K samples) - consider data augmentation or more conservative validation strategies")
        elif sample_size > 5000000:
            recommendations.append("💾 Large dataset detected (>5M samples) - consider chunked processing and memory optimization")
        
        # Class imbalance recommendations
        splice_analysis = profile.get('splice_analysis', {})
        if 'class_balance' in splice_analysis:
            balance_ratio = splice_analysis['class_balance'].get('balance_ratio', 1.0)
            if balance_ratio < 0.1:
                recommendations.append("⚖️  Severe class imbalance detected - use appropriate metrics (F1, AUC, AP) and consider resampling")
            elif balance_ratio < 0.3:
                recommendations.append("⚖️  Moderate class imbalance - monitor minority class performance closely")
        
        # Feature recommendations
        feature_analysis = profile.get('feature_analysis', {})
        feature_counts = feature_analysis.get('feature_counts', {})
        
        total_features = sum(feature_counts.values())
        if total_features > 5000:
            recommendations.append("🔍 High-dimensional dataset - consider feature selection or dimensionality reduction")
        
        kmer_count = feature_counts.get('kmer_features', 0)
        if kmer_count > 1000:
            recommendations.append("🧬 Large number of k-mer features - sparse matrices may improve memory efficiency")
        
        # Data quality recommendations
        quality_analysis = profile.get('quality_analysis', {})
        missing_pct = quality_analysis.get('missing_values', {}).get('percentage_missing', 0)
        if missing_pct > 5:
            recommendations.append("❗ Significant missing values detected - implement robust imputation strategies")
        
        duplicate_pct = profile.get('basic_info', {}).get('duplicate_percentage', 0)
        if duplicate_pct > 1:
            recommendations.append("🔄 Duplicate rows detected - verify data integrity and remove if appropriate")
        
        # Performance recommendations
        performance = profile.get('base_model_performance', {})
        if 'overall_accuracy' in performance:
            accuracy = performance['overall_accuracy']
            if accuracy < 0.8:
                recommendations.append("📊 Low base model accuracy - verify data quality and feature engineering")
            elif accuracy > 0.99:
                recommendations.append("🎯 Very high base model accuracy - check for data leakage")
        
        # Memory recommendations
        memory_mb = profile.get('basic_info', {}).get('estimated_memory_usage_mb', 0)
        if memory_mb > 8000:  # > 8GB
            recommendations.append("💾 High memory usage - consider data type optimization and chunked processing")
        
        # Gene manifest consistency recommendations
        manifest_analysis = profile.get('gene_manifest_analysis', {})
        if manifest_analysis:
            consistency_pct = manifest_analysis.get('consistency_percentage', 100)
            if consistency_pct < 95:
                recommendations.append("🧬 Gene manifest inconsistency detected - verify data pipeline integrity")
            
            manifest_only = manifest_analysis.get('manifest_only_count', 0)
            training_only = manifest_analysis.get('training_only_count', 0)
            if manifest_only > 0:
                recommendations.append(f"📋 {manifest_only} genes in manifest but not in training data - check data filtering")
            if training_only > 0:
                recommendations.append(f"📋 {training_only} genes in training data but not in manifest - check manifest generation")
        
        # Unseen position recommendations
        unseen_analysis = profile.get('unseen_position_analysis', {})
        if unseen_analysis and 'unseen_percentage' in unseen_analysis:
            unseen_pct = unseen_analysis.get('unseen_percentage', 0)
            if unseen_pct > 50:
                recommendations.append(f"🔍 High percentage of unseen positions ({unseen_pct:.1f}%) - consider expanding training data coverage")
            elif unseen_pct < 10:
                recommendations.append(f"🔍 Low percentage of unseen positions ({unseen_pct:.1f}%) - training data may be comprehensive")
            
            mean_coverage = unseen_analysis.get('position_coverage_stats', {}).get('mean_coverage', 100)
            if mean_coverage < 50:
                recommendations.append("📊 Low average position coverage - consider more comprehensive gene sampling")
        
        # Streaming-specific recommendations
        if sample_size > 100000:
            recommendations.append("🔧 Large dataset processed in streaming mode - some statistics may be approximate")
        
        return recommendations


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Comprehensive EDA profiler for splice site prediction datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --dataset train_pc_1000/master --verbose
  %(prog)s --dataset data/training --output-dir results/profile --generate-plots
  %(prog)s --dataset /path/to/data --output-file profile.json --no-summary
  
  # Quick testing with gene subset and single file:
  %(prog)s --dataset train_pc_1000/master --genes ENSG00000131018,ENSG00000114270,ENSG00000183878 --max-files 1 --sample-rows 10000 --verbose
  
  # Test with first file only:
  %(prog)s --dataset train_pc_1000/master --max-files 1 --generate-plots --verbose
        """
    )
    
    parser.add_argument(
        '--dataset', 
        required=True,
        help='Path to dataset directory or file'
    )
    
    parser.add_argument(
        '--output-dir',
        help='Output directory for results and plots'
    )
    
    parser.add_argument(
        '--output-file',
        help='Output JSON file for detailed profile (default: dataset_profile.json)'
    )
    
    parser.add_argument(
        '--generate-plots',
        action='store_true',
        help='Generate visualization plots'
    )
    
    parser.add_argument(
        '--no-summary',
        action='store_true',
        help='Skip printing summary to console'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=100000,
        help='Batch size for processing large datasets (default: 100000)'
    )
    
    parser.add_argument(
        '--genes',
        help='Comma-separated list of gene IDs to analyze (for testing/subsetting)'
    )
    
    parser.add_argument(
        '--max-files',
        type=int,
        help='Maximum number of parquet files to process (for testing)'
    )
    
    parser.add_argument(
        '--sample-rows',
        type=int,
        help='Sample only this many rows from each file (for testing)'
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Create profiler
    profiler = SpliceDatasetProfiler(verbose=args.verbose, batch_size=args.batch_size)
    
    try:
        # Parse gene filter if provided
        gene_filter = None
        if args.genes:
            gene_filter = [gene.strip() for gene in args.genes.split(',')]
            profiler.log(f"Filtering for {len(gene_filter)} genes: {gene_filter[:5]}{'...' if len(gene_filter) > 5 else ''}")
        
        # Generate profile
        profile = profiler.profile_dataset(
            dataset_path=args.dataset,
            output_dir=args.output_dir,
            generate_plots=args.generate_plots,
            gene_filter=gene_filter,
            max_files=args.max_files,
            sample_rows=args.sample_rows
        )
        
        # Print summary unless disabled
        if not args.no_summary:
            profiler.print_summary(profile)
        
        # Save detailed profile to JSON
        if args.output_file or args.output_dir:
            if args.output_file:
                output_file = Path(args.output_file)
            else:
                output_dir = Path(args.output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                output_file = output_dir / "dataset_profile.json"
            
            with open(output_file, 'w') as f:
                json.dump(profile, f, indent=2, default=str)
            
            profiler.log(f"Detailed profile saved to: {output_file}")
        
    except Exception as e:
        profiler.log(f"Error during profiling: {e}", "ERROR")
        raise


if __name__ == "__main__":
    main() 