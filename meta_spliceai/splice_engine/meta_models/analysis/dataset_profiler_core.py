"""
Core dataset profiling logic for splice site prediction datasets.

This module contains the main SpliceDatasetProfiler class and core analysis
functions, separated from visualization and statistics utilities.
"""
import logging
import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union, Iterator
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
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

from meta_spliceai.splice_engine.meta_models.analysis.streaming_stats import (
    StreamingStats, categorize_features, find_splice_column, find_gene_column
)
from meta_spliceai.splice_engine.meta_models.analysis.dataset_visualizer import (
    create_splice_distribution_plot, create_scores_stats_plot, create_missing_values_plot,
    create_top_genes_plot, create_feature_categories_plot, create_gene_manifest_consistency_plot,
    create_performance_metrics_plot
)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


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
            else:
                print(f"[{level}] {message}")

    def load_dataset_batches(self, dataset_path: str, gene_filter: Optional[List[str]] = None, 
                           max_files: Optional[int] = None, sample_rows: Optional[int] = None) -> Iterator[pd.DataFrame]:
        """Load dataset in batches to avoid OOM issues."""
        dataset_path = Path(dataset_path)
        
        # Find parquet files
        if dataset_path.is_file():
            parquet_files = [dataset_path]
        else:
            parquet_files = list(dataset_path.glob("*.parquet"))
        
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
                        df_lazy = df_lazy.limit(sample_rows)
                        self.log(f"  Limited to {sample_rows} rows")
                    
                    # Convert to pandas for compatibility
                    df = df_lazy.collect().to_pandas()
                    
                else:
                    # Fallback to pandas
                    df = pd.read_parquet(file_path)
                    
                    # Apply gene filtering
                    if gene_filter:
                        gene_col = find_gene_column(df)
                        if gene_col:
                            df = df[df[gene_col].isin(gene_filter)]
                            self.log(f"  Applied gene filter on column: {gene_col}")
                        else:
                            self.log("  Warning: Could not find gene_id column for filtering", "WARNING")
                    
                    # Apply row sampling
                    if sample_rows:
                        df = df.head(sample_rows)
                        self.log(f"  Limited to {sample_rows} rows")
                
                if len(df) > 0:
                    self.log(f"  Yielding batch with {len(df):,} rows, {len(df.columns)} columns")
                    yield df
                else:
                    self.log(f"  No data after filtering")
                
                # Clean up memory
                del df
                gc.collect()
                
            except Exception as e:
                self.log(f"Error processing file {file_path.name}: {e}", "ERROR")
                continue
        
        self.log(f"Total estimated rows processed: {total_rows:,}")

    def load_gene_features(self, gene_features_path: str = "data/ensembl/spliceai_analysis/gene_features.tsv"):
        """Load gene features and create gene name mapping."""
        try:
            if Path(gene_features_path).exists():
                self.gene_features = pd.read_csv(gene_features_path, sep='\t')
                
                # Create gene name mapping
                if 'gene_id' in self.gene_features.columns and 'gene_name' in self.gene_features.columns:
                    self.gene_name_mapping = dict(zip(
                        self.gene_features['gene_id'], 
                        self.gene_features['gene_name']
                    ))
                    self.log(f"Loaded gene features with {len(self.gene_name_mapping)} gene name mappings")
                else:
                    self.log("Gene features file missing required columns", "WARNING")
            else:
                self.log(f"Gene features file not found: {gene_features_path}", "WARNING")
        except Exception as e:
            self.log(f"Error loading gene features: {e}", "WARNING")

    def get_gene_display_name(self, gene_id: str) -> str:
        """Get display name for a gene (gene name if available, otherwise gene ID)."""
        return self.gene_name_mapping.get(gene_id, gene_id)

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
        splice_col = find_splice_column(df)
        if splice_col:
            splice_values = df[splice_col].dropna()
            for val in splice_values:
                self.splice_type_counts[val] += 1
        
        # Update gene counts
        gene_col = find_gene_column(df)
        if gene_col:
            gene_values = df[gene_col].dropna()
            for val in gene_values:
                self.gene_counts[val] += 1

    def analyze_basic_info_streaming(self, dataset_path: str, gene_filter: Optional[List[str]] = None, 
                                   max_files: Optional[int] = None, sample_rows: Optional[int] = None) -> Dict[str, Any]:
        """Analyze basic dataset information using streaming approach."""
        self.log("Analyzing basic dataset information (streaming)...")
        
        # Store parameters for later use in type-specific analysis
        self.current_dataset_path = dataset_path
        self.current_gene_filter = gene_filter
        self.current_max_files = max_files
        self.current_sample_rows = sample_rows
        
        columns = None
        for batch_df in self.load_dataset_batches(dataset_path, gene_filter, max_files, sample_rows):
            if columns is None:
                columns = list(batch_df.columns)
            
            self.update_streaming_stats(batch_df)
        
        if columns is None:
            return {'error': 'No data found'}
        
        # Calculate basic statistics
        basic_info = {
            'dataset_path': str(dataset_path),
            'dataset_shape': (self.total_samples, len(columns)),
            'sample_size': self.total_samples,
            'feature_count': len(columns),
            'duplicate_count': self.duplicate_count,
            'duplicate_percentage': (self.duplicate_count / self.total_samples * 100) if self.total_samples > 0 else 0,
            'estimated_memory_usage_mb': self.total_samples * len(columns) * 8 / (1024**2),  # Rough estimate
            'columns': columns
        }
        
        return basic_info

    def analyze_spliceai_scores_by_type_streaming(self) -> Dict[str, Any]:
        """Analyze SpliceAI scores by predicted type (argmax-based)."""
        self.log("Analyzing SpliceAI scores by predicted type...")
        
        # We need to re-process the data to compute type-specific statistics
        # This is necessary because we need to match scores to predicted types
        type_specific_stats = {
            'donor': {'scores': [], 'count': 0},
            'acceptor': {'scores': [], 'count': 0}, 
            'neither': {'scores': [], 'count': 0}
        }
        
        # Store original dataset path for reprocessing
        if hasattr(self, 'current_dataset_path'):
            try:
                # Reprocess data to get type-specific statistics
                for batch_df in self.load_dataset_batches(self.current_dataset_path, 
                                                        self.current_gene_filter, 
                                                        self.current_max_files, 
                                                        self.current_sample_rows):
                    # Find score columns
                    donor_col = None
                    acceptor_col = None
                    neither_col = None
                    
                    for col in batch_df.columns:
                        col_lower = col.lower()
                        if 'donor_score' in col_lower and 'meta' not in col_lower:
                            donor_col = col
                        elif 'acceptor_score' in col_lower and 'meta' not in col_lower:
                            acceptor_col = col
                        elif 'neither_score' in col_lower and 'meta' not in col_lower:
                            neither_col = col
                    
                    if donor_col and acceptor_col and neither_col:
                        # Get scores
                        donor_scores = batch_df[donor_col].fillna(0)
                        acceptor_scores = batch_df[acceptor_col].fillna(0)
                        neither_scores = batch_df[neither_col].fillna(0)
                        
                        # Find predicted type by argmax
                        scores_array = np.column_stack([donor_scores, acceptor_scores, neither_scores])
                        predicted_types = np.argmax(scores_array, axis=1)
                        
                        # Collect type-specific scores
                        for i, pred_type in enumerate(predicted_types):
                            if pred_type == 0:  # Donor
                                type_specific_stats['donor']['scores'].append(donor_scores.iloc[i])
                                type_specific_stats['donor']['count'] += 1
                            elif pred_type == 1:  # Acceptor
                                type_specific_stats['acceptor']['scores'].append(acceptor_scores.iloc[i])
                                type_specific_stats['acceptor']['count'] += 1
                            elif pred_type == 2:  # Neither
                                type_specific_stats['neither']['scores'].append(neither_scores.iloc[i])
                                type_specific_stats['neither']['count'] += 1
                
                # Compute statistics for each type
                computed_stats = {}
                for splice_type, data in type_specific_stats.items():
                    if data['scores']:
                        scores = np.array(data['scores'])
                        min_val = float(np.min(scores))
                        max_val = float(np.max(scores))
                        
                        computed_stats[f'{splice_type}_score'] = {
                            'mean': float(np.mean(scores)),
                            'median': float(np.median(scores)),
                            'std': float(np.std(scores)),
                            'min': min_val,
                            'max': max_val,
                            'count': data['count']
                        }
                        
                        # Debug logging
                        self.log(f"Type-specific stats for {splice_type}: min={min_val:.4f}, max={max_val:.4f}, count={data['count']}")
                    else:
                        computed_stats[f'{splice_type}_score'] = {
                            'mean': 0.0, 'median': 0.0, 'std': 0.0,
                            'min': 0.0, 'max': 0.0, 'count': 0
                        }
                
                return computed_stats
                
            except Exception as e:
                self.log(f"Error computing type-specific SpliceAI stats: {e}", "ERROR")
                # Fallback to original streaming stats
                return self.get_spliceai_streaming_stats()
        
        # Fallback if no dataset path stored
        return self.get_spliceai_streaming_stats()
    
    def get_spliceai_streaming_stats(self) -> Dict[str, Any]:
        """Get SpliceAI statistics from streaming stats (fallback method)."""
        spliceai_stats = {}
        for col, stats in self.streaming_stats.items():
            if any(score_type in col.lower() for score_type in ['donor_score', 'acceptor_score', 'neither_score']):
                spliceai_stats[col] = stats.get_stats()
        return spliceai_stats

    def analyze_splice_types_streaming(self) -> Dict[str, Any]:
        """Analyze splice type distributions using streaming data."""
        self.log("Analyzing splice type distribution (streaming)...")
        
        if not self.splice_type_counts:
            return {'error': 'No splice type information found'}
        
        total_splice_sites = sum(self.splice_type_counts.values())
        
        # Normalize and calculate percentages
        normalized_distribution = dict(self.splice_type_counts)
        percentage_distribution = {
            splice_type: (count / total_splice_sites * 100) if total_splice_sites > 0 else 0
            for splice_type, count in self.splice_type_counts.items()
        }
        
        # Calculate class balance metrics
        counts = list(self.splice_type_counts.values())
        if len(counts) >= 2:
            min_count = min(counts)
            max_count = max(counts)
            balance_ratio = min_count / max_count if max_count > 0 else 0
            is_balanced = balance_ratio >= 0.1  # Consider balanced if ratio >= 0.1
        else:
            balance_ratio = 1.0
            is_balanced = True
        
        return {
            'normalized_distribution': normalized_distribution,
            'percentage_distribution': percentage_distribution,
            'total_splice_sites': total_splice_sites,
            'class_balance': {
                'balance_ratio': balance_ratio,
                'is_balanced': is_balanced,
                'min_class_count': min(counts) if counts else 0,
                'max_class_count': max(counts) if counts else 0
            }
        }

    def analyze_gene_splice_sites_streaming(self, top_k: int = 20) -> Dict[str, Any]:
        """Analyze splice sites distribution by gene using streaming data."""
        self.log("Analyzing gene-level splice site distribution (streaming)...")
        
        if not self.gene_counts:
            return {'error': 'No gene information found'}
        
        # Get top genes by count
        top_genes = dict(self.gene_counts.most_common(top_k))
        
        # Calculate statistics
        gene_counts_list = list(self.gene_counts.values())
        total_genes = len(self.gene_counts)
        
        occurrence_count_stats = {
            'mean': np.mean(gene_counts_list) if gene_counts_list else 0,
            'median': np.median(gene_counts_list) if gene_counts_list else 0,
            'std': np.std(gene_counts_list) if gene_counts_list else 0,
            'min': min(gene_counts_list) if gene_counts_list else 0,
            'max': max(gene_counts_list) if gene_counts_list else 0
        }
        
        return {
            'top_genes': top_genes,
            'total_genes': total_genes,
            'occurrence_count_stats': occurrence_count_stats,
            'total_gene_occurrences': sum(gene_counts_list)
        }

    def generate_summary_statistics_streaming(self) -> Dict[str, Any]:
        """Generate summary statistics from streaming data."""
        self.log("Generating summary statistics (streaming)...")
        
        # Feature analysis using categorization
        columns = list(self.streaming_stats.keys()) + list(self.categorical_stats.keys())
        feature_analysis = categorize_features(columns)
        
        # Add splice AI stats to feature analysis
        splice_ai_stats = {}
        for col in feature_analysis['feature_categories']['splice_ai_scores']:
            if col in self.streaming_stats:
                splice_ai_stats[col] = self.streaming_stats[col].get_stats()
        
        if splice_ai_stats:
            feature_analysis['category_analysis'] = {'splice_ai_stats': splice_ai_stats}
        
        # Quality analysis
        total_missing = sum(self.null_counts.values())
        columns_with_missing = {col: count for col, count in self.null_counts.items() if count > 0}
        
        quality_analysis = {
            'missing_values': {
                'total_missing': total_missing,
                'percentage_missing': (total_missing / (self.total_samples * len(columns)) * 100) if self.total_samples > 0 and columns else 0,
                'columns_with_missing': columns_with_missing,
                'columns_with_missing_count': len(columns_with_missing)
            }
        }
        
        return {
            'feature_analysis': feature_analysis,
            'quality_analysis': quality_analysis,
            'streaming_stats_summary': {
                'total_numeric_columns': len(self.streaming_stats),
                'total_categorical_columns': len(self.categorical_stats),
                'total_samples_processed': self.total_samples
            }
        }

    def create_visualizations_streaming(self, output_dir: Path, profile: Dict[str, Any] = None) -> Dict[str, str]:
        """Create visualizations using streaming data and profile statistics."""
        if profile is None:
            profile = self.profile
        
        self.log("Creating visualizations (streaming mode)...")
        
        # Create visualization directory
        vis_dir = output_dir / "visualizations"
        vis_dir.mkdir(parents=True, exist_ok=True)
        
        plot_files = {}
        
        try:
            # 1. Splice distribution plot
            splice_analysis = profile.get('splice_analysis', {})
            if splice_analysis:
                plot_path = create_splice_distribution_plot(splice_analysis, vis_dir)
                if plot_path:
                    plot_files['splice_distribution'] = plot_path
            
            # 2. SpliceAI scores statistics (type-specific)
            spliceai_type_stats = profile.get('spliceai_type_stats', {})
            if spliceai_type_stats:
                plot_path = create_scores_stats_plot(spliceai_type_stats, splice_analysis, vis_dir)
                if plot_path:
                    plot_files['scores_stats'] = plot_path
            else:
                # Fallback to original method if type-specific stats not available
                feature_analysis = profile.get('feature_analysis', {})
                splice_stats = feature_analysis.get('category_analysis', {}).get('splice_ai_stats', {})
                if splice_stats:
                    plot_path = create_scores_stats_plot(splice_stats, splice_analysis, vis_dir)
                    if plot_path:
                        plot_files['scores_stats'] = plot_path
            
            # 3. Missing values analysis (always generate, even if no missing values)
            quality_analysis = profile.get('quality_analysis', {})
            missing_data = quality_analysis.get('missing_values', {}).get('columns_with_missing', {})
            total_samples = profile.get('basic_info', {}).get('sample_size', 1)
            plot_path, missing_summary = create_missing_values_plot(missing_data, total_samples, vis_dir)
            if plot_path:
                plot_files['missing_values'] = plot_path
                # Save summary to profile
                if missing_summary:
                    if 'quality_analysis' not in profile:
                        profile['quality_analysis'] = {}
                    profile['quality_analysis']['missing_values_summary'] = missing_summary
            
            # 4. Top genes chart
            gene_analysis = profile.get('gene_splice_analysis', {})
            if 'top_genes' in gene_analysis and gene_analysis['top_genes']:
                plot_path = create_top_genes_plot(gene_analysis['top_genes'], self.gene_name_mapping, vis_dir)
                if plot_path:
                    plot_files['top_genes'] = plot_path
            
            # 5. Feature categories
            feature_analysis = profile.get('feature_analysis', {})
            feature_counts = feature_analysis.get('feature_counts', {})
            if feature_counts:
                plot_path = create_feature_categories_plot(feature_counts, vis_dir)
                if plot_path:
                    plot_files['feature_categories'] = plot_path
        
            # 6. Performance metrics (if available)
            if 'performance_metrics' in profile:
                plot_path = create_performance_metrics_plot(profile['performance_metrics'], vis_dir)
                if plot_path:
                    plot_files['performance_metrics'] = plot_path
            
            self.log(f"Created {len(plot_files)} visualization plots (streaming mode)")
            
        except Exception as e:
            self.log(f"Error creating visualizations: {e}", "WARNING")
        
        return plot_files

    def profile_dataset(self, dataset_path: str, output_dir: Optional[str] = None, 
                       generate_plots: bool = False, gene_filter: Optional[List[str]] = None,
                       max_files: Optional[int] = None, sample_rows: Optional[int] = None) -> Dict[str, Any]:
        """Generate comprehensive dataset profile using streaming approach for large datasets."""
        self.log("Starting comprehensive dataset profiling (streaming mode)...")
        
        # Load gene features for better reporting
        self.load_gene_features()
        
        # Basic dataset information
        basic_info = self.analyze_basic_info_streaming(dataset_path, gene_filter, max_files, sample_rows)
        if 'error' in basic_info:
            return basic_info
        
        # Splice type analysis
        splice_analysis = self.analyze_splice_types_streaming()
        
        # Type-specific SpliceAI score analysis (argmax-based)
        spliceai_type_stats = self.analyze_spliceai_scores_by_type_streaming()
        
        # Gene-level analysis
        gene_splice_analysis = self.analyze_gene_splice_sites_streaming()
        
        # Summary statistics and feature analysis
        summary_stats = self.generate_summary_statistics_streaming()
        
        # Compile profile
        profile = {
            'dataset_path': dataset_path,
            'basic_info': basic_info,
            'splice_analysis': splice_analysis,
            'spliceai_type_stats': spliceai_type_stats,
            'gene_splice_analysis': gene_splice_analysis,
            **summary_stats
        }
        
        # Generate visualizations if requested
        if generate_plots and output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            plot_files = self.create_visualizations_streaming(output_path, profile)
            profile['visualizations'] = plot_files
        
        # Generate recommendations
        recommendations = self.generate_recommendations(profile)
        profile['recommendations'] = recommendations
        
        self.profile = profile
        return profile

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
        
        # Memory recommendations
        memory_mb = profile.get('basic_info', {}).get('estimated_memory_usage_mb', 0)
        if memory_mb > 8000:  # > 8GB
            recommendations.append("💾 High memory usage - consider data type optimization and chunked processing")
        
        return recommendations

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
        
        # Splice type distribution
        splice_analysis = profile.get('splice_analysis', {})
        if 'normalized_distribution' in splice_analysis:
            print(f"\n🧬 Splice Site Distribution:")
            dist = splice_analysis['normalized_distribution']
            pct_dist = splice_analysis.get('percentage_distribution', {})
            
            for splice_type, count in dist.items():
                pct = pct_dist.get(splice_type, 0)
                # Handle numeric keys (convert to string labels)
                if isinstance(splice_type, (int, float)) or str(splice_type).isdigit():
                    if splice_type == 0:
                        display_name = "Neither"
                    else:
                        display_name = f"Type {splice_type}"
                else:
                    display_name = str(splice_type).capitalize()
                print(f"   • {display_name}: {count:,} ({pct:.1f}%)")
            
            # Class balance
            balance_info = splice_analysis.get('class_balance', {})
            if balance_info:
                balance_ratio = balance_info.get('balance_ratio', 1.0)
                is_balanced = balance_info.get('is_balanced', True)
                status = "✅ Balanced" if is_balanced else "⚠️  Imbalanced"
                print(f"   • Class balance: {status} (ratio: {balance_ratio:.3f})")
        
        # Feature analysis
        feature_analysis = profile.get('feature_analysis', {})
        if 'feature_counts' in feature_analysis:
            feature_counts = feature_analysis['feature_counts']
            print(f"\n🔧 Feature Categories:")
            
            # Show non-zero categories with better names
            category_display_names = {
                'splice_ai_scores': 'SpliceAI Raw Scores',
                'probability_context_features': 'Probability & Context',
                'kmer_features': 'K-mers',
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
        if missing_data.get('total_missing', 0) > 0:
            total_missing = missing_data['total_missing']
            pct_missing = missing_data.get('percentage_missing', 0)
            print(f"\n⚠️  Data Quality Issues:")
            print(f"   • Missing values: {total_missing:,} ({pct_missing:.2f}%)")
        
        # Recommendations
        recommendations = profile.get('recommendations', [])
        if recommendations:
            print(f"\n💡 Recommendations:")
            for rec in recommendations[:5]:  # Show first 5
                print(f"   {rec}")
        
        print("\n" + "="*80)
