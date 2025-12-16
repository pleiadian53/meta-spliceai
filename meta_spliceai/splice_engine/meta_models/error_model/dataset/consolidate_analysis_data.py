#!/usr/bin/env python3
"""
Consolidate chunked analysis sequence files for error model training.

This module provides functionality to find all chunked analysis sequence files 
in a directory, filter them by prediction type, and consolidate them into a 
single file suitable for error model training.
"""

import argparse
import logging
from pathlib import Path
from typing import List, Set, Optional, Dict, Any, Tuple
import polars as pl
from tqdm import tqdm
import numpy as np


class AnalysisDataConsolidator:
    """Consolidates chunked analysis sequence files for error model training."""
    
    def __init__(self, verbose: bool = False):
        """Initialize the consolidator.
        
        Parameters
        ----------
        verbose : bool
            Enable verbose logging
        """
        self.verbose = verbose
        self.logger = self._setup_logging()
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            level = logging.INFO if self.verbose else logging.WARNING
            handler = logging.StreamHandler()
            handler.setLevel(level)
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(level)
        return logger
    
    def extract_chromosome_from_filename(self, filepath: Path) -> Optional[str]:
        """Extract chromosome from filename pattern: analysis_sequences_CHR_chunk_*.tsv"""
        import re
        pattern = r'analysis_sequences_([^_]+)_chunk_'
        match = re.search(pattern, filepath.name)
        if match:
            return match.group(1)
        return None
    
    def find_chunk_files(self, data_dir: Path, target_chromosomes: Optional[Set[str]] = None) -> List[Path]:
        """Find all analysis sequence chunk files, optionally filtering by chromosome."""
        chunk_files = []
        
        # Look for patterns like: analysis_sequences_*_chunk_*.tsv
        for pattern in ["analysis_sequences_*_chunk_*.tsv", "analysis_sequences_*_chunk_*.tsv.gz"]:
            for filepath in data_dir.glob(pattern):
                # Filter by chromosome if specified
                if target_chromosomes:
                    chrom = self.extract_chromosome_from_filename(filepath)
                    if chrom and chrom not in target_chromosomes:
                        self.logger.debug(f"Skipping {filepath.name} (chr {chrom} not in target)")
                        continue
                chunk_files.append(filepath)
        
        return sorted(chunk_files)
    
    def get_prediction_types_for_error_analysis(self, error_type: str) -> Tuple[Set[str], Dict[str, int]]:
        """Get the required prediction types and label mapping for error analysis.
        
        Returns:
        - Set of prediction types to include
        - Label mapping (FP/FN -> 1 for errors, TP/TN -> 0 for correct)
        """
        error_type_mapping = {
            'fp_vs_tp': ({'FP', 'TP'}, {'FP': 1, 'TP': 0}),  # FP is error (positive)
            'fn_vs_tn': ({'FN', 'TN'}, {'FN': 1, 'TN': 0}),  # FN is error (positive)
            'fn_vs_tp': ({'FN', 'TP'}, {'FN': 1, 'TP': 0}),  # FN is error (positive)
            'fp_vs_tn': ({'FP', 'TN'}, {'FP': 1, 'TN': 0}),  # FP is error (positive)
            'error_vs_correct': ({'FP', 'FN', 'TP'}, {'FP': 1, 'FN': 1, 'TP': 0}),
            'all': ({'FP', 'FN', 'TP', 'TN'}, {'FP': 1, 'FN': 1, 'TP': 0, 'TN': 0})
        }
        
        if error_type not in error_type_mapping:
            raise ValueError(f"Unknown error type: {error_type}. Available: {list(error_type_mapping.keys())}")
        
        return error_type_mapping[error_type]


    def identify_gene_column(self, df: pl.DataFrame) -> Optional[str]:
        """Identify which gene column to use (gene_name or gene_id as fallback)."""
        if 'gene_name' in df.columns:
            # Check if gene_name has valid values
            non_null_count = df.filter(pl.col('gene_name').is_not_null()).height
            if non_null_count > 0:
                return 'gene_name'
        
        if 'gene_id' in df.columns:
            non_null_count = df.filter(pl.col('gene_id').is_not_null()).height
            if non_null_count > 0:
                self.logger.info("Using 'gene_id' column as gene_name is not available")
                return 'gene_id'
        
        self.logger.warning("No 'gene_name' or 'gene_id' column found. Using row-level sampling.")
        return None

    def sample_by_genes(self, df: pl.DataFrame, gene_col: str, max_samples: int, seed: int = 42) -> pl.DataFrame:
        """Sample at gene level to preserve genomic structure."""
        np.random.seed(seed)
        
        # Get unique genes and their sample counts
        gene_counts = df.group_by(gene_col).agg(pl.count().alias('count'))
        gene_counts = gene_counts.filter(pl.col(gene_col).is_not_null())
        
        # Shuffle genes
        genes = gene_counts[gene_col].to_list()
        counts = gene_counts['count'].to_list()
        
        indices = np.arange(len(genes))
        np.random.shuffle(indices)
        
        # Select genes until we reach approximate sample size
        selected_genes = []
        current_samples = 0
        
        for idx in indices:
            gene = genes[idx]
            count = counts[idx]
            
            if current_samples + count <= max_samples * 1.1:  # Allow 10% overshoot
                selected_genes.append(gene)
                current_samples += count
            elif current_samples < max_samples * 0.9:  # If we're under 90%, include anyway
                selected_genes.append(gene)
                current_samples += count
                break
            else:
                break
        
        # Filter to selected genes
        if selected_genes:
            sampled_df = df.filter(pl.col(gene_col).is_in(selected_genes))
            self.logger.info(f"Sampled {sampled_df.height} rows from {len(selected_genes)} genes")
            return sampled_df
        else:
            return df

    def consolidate_and_filter_analysis_sequences(
        self,
        data_dir: Path,
        output_file: Path,
        error_type: str,
        max_rows_per_type: Optional[int] = None,
        chromosomes: Optional[List[str]] = None,
        genes: Optional[List[str]] = None,
        sample_seed: int = 42
    ) -> Dict[str, Any]:
        """
        Consolidate chunked analysis sequence files and filter by prediction type.
        
        Parameters
        ----------
        data_dir : Path
            Directory containing chunked analysis sequence files
        output_file : Path
            Output file path for consolidated data
        error_type : str
            Error analysis type (fp_vs_tp, fn_vs_tp, etc.)
        max_rows_per_type : int, optional
            Maximum rows to keep per prediction type (for memory management)
        sample_seed : int
            Random seed for sampling
            
        Returns
        -------
        Dict[str, Any]
            Consolidation results with statistics
        """
        
        # Convert chromosomes to set for efficient lookup
        target_chromosomes = set(chromosomes) if chromosomes else None
        target_genes = set(genes) if genes else None
        
        # Find chunk files, filtering by chromosome if specified
        chunk_files = self.find_chunk_files(data_dir, target_chromosomes)
        if not chunk_files:
            raise FileNotFoundError(f"No analysis sequence chunk files found in {data_dir}")
        
        self.logger.info(f"Found {len(chunk_files)} chunk files to process")
        
        # Get required prediction types and label mapping
        required_pred_types, label_map = self.get_prediction_types_for_error_analysis(error_type)
        self.logger.info(f"Filtering for prediction types: {required_pred_types}")
        self.logger.info(f"Label mapping: {label_map}")
    
        # Process files in batches to manage memory
        consolidated_dfs = []
        total_rows_by_type = {pred_type: 0 for pred_type in required_pred_types}
        common_schema = None
        
        for chunk_file in tqdm(chunk_files, desc="Processing chunks"):
            try:
                # Read chunk directly (not lazy) for better control
                df = pl.read_csv(chunk_file, separator='\t')
                
                # Filter by prediction type first (most important filter)
                df = df.filter(pl.col('pred_type').is_in(list(required_pred_types)))
                
                if df.height == 0:
                    continue
                
                # Filter by genes if specified
                if target_genes:
                    gene_col = self.identify_gene_column(df)
                    if gene_col:
                        df = df.filter(pl.col(gene_col).is_in(list(target_genes)))
                        if df.height == 0:
                            continue
                
                # Add gene_id as gene_name if using gene_id
                gene_col = self.identify_gene_column(df)
                if gene_col == 'gene_id' and 'gene_name' not in df.columns:
                    df = df.with_columns(pl.col('gene_id').alias('gene_name'))
                
                df_collected = df
            
                if df_collected.height > 0:
                    # Establish common schema from first non-empty dataframe
                    if common_schema is None:
                        common_schema = df_collected.schema
                        self.logger.info(f"Established schema from {chunk_file.name}")
                    else:
                        # Cast to common schema to avoid type mismatches
                        try:
                            df_collected = df_collected.cast(common_schema)
                        except Exception as cast_error:
                            self.logger.warning(f"Schema cast failed for {chunk_file.name}: {cast_error}")
                            # Try to align schemas manually
                            for col_name, expected_dtype in common_schema.items():
                                if col_name in df_collected.columns:
                                    current_dtype = df_collected[col_name].dtype
                                    if current_dtype != expected_dtype:
                                        try:
                                            if expected_dtype == pl.Int64 and current_dtype == pl.String:
                                                # Handle string to int conversion
                                                df_collected = df_collected.with_columns(
                                                    pl.col(col_name).cast(pl.Int64, strict=False)
                                                )
                                            elif expected_dtype == pl.Float64 and current_dtype in [pl.String, pl.Int64]:
                                                # Handle to float conversion
                                                df_collected = df_collected.with_columns(
                                                    pl.col(col_name).cast(pl.Float64, strict=False)
                                                )
                                            elif expected_dtype == pl.String:
                                                # Convert anything to string
                                                df_collected = df_collected.with_columns(
                                                    pl.col(col_name).cast(pl.String)
                                                )
                                        except Exception as col_cast_error:
                                            self.logger.warning(f"Failed to cast column {col_name}: {col_cast_error}")
                    
                    # Track rows by type
                    for pred_type in required_pred_types:
                        type_count = df_collected.filter(pl.col('pred_type') == pred_type).height
                        total_rows_by_type[pred_type] += type_count
                    
                    consolidated_dfs.append(df_collected)
                    self.logger.info(f"Processed {chunk_file.name}: {df_collected.height} rows")
                
            except Exception as e:
                self.logger.warning(f"Error processing {chunk_file}: {e}")
                continue
    
        if not consolidated_dfs:
            raise ValueError("No valid data found after filtering")
        
        # Concatenate all chunks with diagonal strategy to handle schema differences
        self.logger.info("Concatenating all chunks...")
        try:
            consolidated_df = pl.concat(consolidated_dfs, how='vertical')
        except Exception as concat_error:
            self.logger.warning(f"Vertical concat failed: {concat_error}")
            self.logger.info("Trying diagonal concat to handle schema differences...")
            consolidated_df = pl.concat(consolidated_dfs, how='diagonal')
        
        # Log statistics
        self.logger.info(f"Total rows before sampling: {consolidated_df.height}")
        for pred_type, count in total_rows_by_type.items():
            self.logger.info(f"  {pred_type}: {count:,} rows")
    
        # Apply gene-level sampling if requested
        if max_rows_per_type is not None:
            self.logger.info(f"Applying gene-level sampling (approx {max_rows_per_type} samples per type)...")
            
            # Identify gene column
            gene_col = self.identify_gene_column(consolidated_df)
            
            if gene_col:
                sampled_dfs = []
                
                for pred_type in required_pred_types:
                    type_df = consolidated_df.filter(pl.col('pred_type') == pred_type)
                    
                    if type_df.height > max_rows_per_type:
                        # Gene-level sampling
                        type_df = self.sample_by_genes(type_df, gene_col, max_rows_per_type, sample_seed)
                    else:
                        self.logger.info(f"Kept all {pred_type}: {type_df.height:,} rows")
                    
                    sampled_dfs.append(type_df)
                
                consolidated_df = pl.concat(sampled_dfs, how='vertical')
            else:
                # Fallback to simple sampling if no gene column
                sampled_dfs = []
                for pred_type in required_pred_types:
                    type_df = consolidated_df.filter(pl.col('pred_type') == pred_type)
                    if type_df.height > max_rows_per_type:
                        type_df = type_df.sample(n=max_rows_per_type, seed=sample_seed)
                    sampled_dfs.append(type_df)
                consolidated_df = pl.concat(sampled_dfs, how='vertical')
        
        # Add binary labels based on prediction type
        consolidated_df = consolidated_df.with_columns(
            pl.col('pred_type').replace(label_map).alias('label')
        )
        
        # Shuffle the final dataset
        consolidated_df = consolidated_df.sample(fraction=1.0, seed=sample_seed)
        
        # Save consolidated data
        self.logger.info(f"Saving consolidated data to {output_file}")
        consolidated_df.write_csv(output_file, separator='\t')
        
        # Final statistics
        self.logger.info(f"Final consolidated dataset: {consolidated_df.height:,} rows")
        final_counts = consolidated_df.group_by('pred_type').agg(pl.len().alias('count'))
        for row in final_counts.iter_rows(named=True):
            self.logger.info(f"  {row['pred_type']}: {row['count']:,} rows")
        
        # Return consolidation results
        return {
            'output_file': output_file,
            'total_rows': consolidated_df.height,
            'rows_by_type': {row['pred_type']: row['count'] for row in final_counts.iter_rows(named=True)},
            'chunk_files_processed': len(chunk_files),
            'error_type': error_type
        }


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration for standalone script usage."""
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def consolidate_and_filter_analysis_sequences(
    data_dir: Path,
    output_file: Path,
    error_type: str,
    max_rows_per_type: Optional[int] = None,
    chromosomes: Optional[List[str]] = None,
    genes: Optional[List[str]] = None,
    sample_seed: int = 42,
    verbose: bool = True
) -> Dict[str, Any]:
    """Standalone function wrapper for backward compatibility."""
    consolidator = AnalysisDataConsolidator(verbose=verbose)
    return consolidator.consolidate_and_filter_analysis_sequences(
        data_dir=data_dir,
        output_file=output_file,
        error_type=error_type,
        max_rows_per_type=max_rows_per_type,
        chromosomes=chromosomes,
        genes=genes,
        sample_seed=sample_seed
    )


def main():
    parser = argparse.ArgumentParser(description='Consolidate and filter analysis sequence data')
    parser.add_argument('--data-dir', type=Path, required=True,
                       help='Directory containing chunked analysis sequence files')
    parser.add_argument('--output-file', type=Path, required=True,
                       help='Output file for consolidated data')
    parser.add_argument('--error-type', choices=['fp_vs_tp', 'fn_vs_tp', 'fp_vs_tn', 'fn_vs_tn', 'error_vs_correct', 'all'],
                       default='fp_vs_tp', help='Error analysis type')
    parser.add_argument('--max-rows-per-type', type=int, default=None,
                       help='Maximum rows per prediction type (for memory management)')
    parser.add_argument('--chromosomes', nargs='+', default=None,
                       help='Specific chromosomes to include')
    parser.add_argument('--genes', nargs='+', default=None,
                       help='Specific genes to include')
    parser.add_argument('--genes-file', type=Path, default=None,
                       help='File containing gene names (one per line)')
    parser.add_argument('--sample-seed', type=int, default=42,
                       help='Random seed for sampling')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    # Ensure output directory exists
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Load genes from file if provided
    genes = args.genes
    if args.genes_file:
        with open(args.genes_file) as f:
            file_genes = [line.strip() for line in f if line.strip()]
        genes = list(set(genes or []) | set(file_genes))
    
    try:
        result = consolidate_and_filter_analysis_sequences(
            data_dir=args.data_dir,
            output_file=args.output_file,
            error_type=args.error_type,
            max_rows_per_type=args.max_rows_per_type,
            chromosomes=args.chromosomes,
            genes=genes,
            sample_seed=args.sample_seed,
            verbose=args.verbose
        )
        logger.info("✅ Data consolidation completed successfully!")
        logger.info(f"Output: {result['output_file']}")
        logger.info(f"Total rows: {result['total_rows']:,}")
        
    except Exception as e:
        logger.error(f"❌ Consolidation failed: {e}")
        raise


if __name__ == "__main__":
    main()
