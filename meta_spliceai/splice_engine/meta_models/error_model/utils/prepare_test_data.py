#!/usr/bin/env python3
"""
Utility to prepare minimal test datasets for error model workflow validation.

This tool creates small, balanced datasets from full analysis_sequences artifacts,
useful for:
- Quick workflow testing and validation
- CI/CD pipeline smoke tests  
- Development and debugging
- Memory-constrained environments

Example usage:
    # Create a minimal 1000-sample test dataset with 50nt context
    python prepare_test_data.py \
        --input data/analysis_sequences_full.tsv \
        --output data/test_minimal.tsv \
        --context-length 50 \
        --sample-size 1000
        
    # Use only essential features for faster testing
    python prepare_test_data.py \
        --input data/analysis_sequences_full.tsv \
        --output data/test_minimal_features.tsv \
        --features score donor_score acceptor_score
"""

import pandas as pd
import polars as pl
from pathlib import Path
import argparse
import logging
from typing import List, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Essential columns that must always be preserved
ESSENTIAL_COLUMNS = [
    'gene_id', 'transcript_id', 'sequence', 
    'prediction_type', 'predicted_position', 
    'position', 'strand', 'splice_type'
]

# Default minimal feature set for quick testing
DEFAULT_TEST_FEATURES = [
    'score', 'donor_score', 'acceptor_score', 'neither_score',
    'context_score_m1', 'context_score_p1'
]

def trim_sequence_to_context(sequence: str, context_length: int) -> str:
    """
    Trim sequence to specified context length around the center.
    
    Args:
        sequence: DNA sequence string
        context_length: Target context length in nucleotides
        
    Returns:
        Trimmed sequence of specified length
    """
    if len(sequence) <= context_length:
        return sequence
    
    # Extract symmetric context around center
    center = len(sequence) // 2
    half_context = context_length // 2
    
    start = max(0, center - half_context)
    end = min(len(sequence), center + half_context)
    
    return sequence[start:end]

def prepare_minimal_test_data(
    input_file: Path,
    output_file: Path,
    context_length: int = 50,
    feature_columns: Optional[List[str]] = None,
    sample_size: int = 1000,
    use_polars: bool = False,
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Prepare minimal test data with reduced context and features.
    
    Args:
        input_file: Path to input TSV file (analysis_sequences_* artifact)
        output_file: Path for output TSV file
        context_length: Context length for sequences (default 50nt for fast testing)
        feature_columns: List of feature columns to include (None = all features)
        sample_size: Number of samples to include (balanced across classes)
        use_polars: Use Polars for faster processing of large files
        random_seed: Random seed for reproducible sampling
        
    Returns:
        Prepared DataFrame with minimal test data
    """
    
    logger.info(f"Loading data from {input_file}...")
    
    if use_polars:
        # Use Polars for large files
        df_pl = pl.read_csv(input_file, separator='\t')
        total_rows = df_pl.height
        logger.info(f"Loaded {total_rows} rows using Polars")
        
        # Convert to pandas for processing (after initial filtering if needed)
        if sample_size and total_rows > sample_size:
            df_pl = df_pl.sample(n=sample_size, seed=random_seed)
        df = df_pl.to_pandas()
    else:
        # Use pandas directly for smaller files
        df = pd.read_csv(input_file, sep='\t')
        logger.info(f"Loaded {len(df)} rows")
    
    # Balanced sampling across prediction classes
    if sample_size and len(df) > sample_size:
        if 'prediction_type' in df.columns:
            # Sample equally from each prediction type (FP, TP, FN, TN)
            unique_types = df['prediction_type'].unique()
            samples_per_type = sample_size // len(unique_types)
            remainder = sample_size % len(unique_types)
            
            sampled_dfs = []
            for i, pred_type in enumerate(unique_types):
                type_df = df[df['prediction_type'] == pred_type]
                # Add remainder samples to first class to reach exact sample_size
                n_samples = samples_per_type + (1 if i < remainder else 0)
                n_samples = min(n_samples, len(type_df))
                sampled = type_df.sample(n=n_samples, random_state=random_seed)
                sampled_dfs.append(sampled)
            
            df = pd.concat(sampled_dfs, ignore_index=True)
            logger.info(f"Sampled {len(df)} rows (balanced across {len(unique_types)} classes)")
            
            # Log class distribution
            for pred_type in unique_types:
                count = len(df[df['prediction_type'] == pred_type])
                logger.info(f"  {pred_type}: {count} samples ({count/len(df)*100:.1f}%)")
        else:
            # Random sampling if no prediction_type column
            df = df.sample(n=sample_size, random_state=random_seed)
            logger.info(f"Randomly sampled {sample_size} rows")
    
    # Trim sequences to specified context length
    logger.info(f"Trimming sequences to {context_length}nt...")
    original_lengths = df['sequence'].str.len()
    df['sequence'] = df['sequence'].apply(lambda x: trim_sequence_to_context(x, context_length))
    trimmed_lengths = df['sequence'].str.len()
    
    logger.info(f"  Original: {original_lengths.mean():.1f} ± {original_lengths.std():.1f} nt")
    logger.info(f"  Trimmed:  {trimmed_lengths.mean():.1f} ± {trimmed_lengths.std():.1f} nt")
    
    # Select feature columns
    if feature_columns:
        # Combine essential columns with requested features
        all_cols = list(set(ESSENTIAL_COLUMNS + feature_columns))
        
        # Keep only columns that exist in the data
        available_cols = [col for col in all_cols if col in df.columns]
        missing_cols = [col for col in all_cols if col not in df.columns]
        
        if missing_cols:
            logger.warning(f"Missing columns (will be skipped): {missing_cols}")
        
        df = df[available_cols]
        logger.info(f"Selected {len(available_cols)} columns")
        logger.info(f"  Essential: {[c for c in ESSENTIAL_COLUMNS if c in available_cols]}")
        logger.info(f"  Features: {[c for c in feature_columns if c in available_cols]}")
    
    # Save prepared data
    output_file.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving to {output_file}...")
    df.to_csv(output_file, sep='\t', index=False)
    
    # Print summary statistics
    logger.info("=" * 60)
    logger.info("TEST DATA SUMMARY:")
    logger.info(f"  Output file: {output_file}")
    logger.info(f"  Total samples: {len(df)}")
    logger.info(f"  Context length: {context_length} nt")
    logger.info(f"  Sequence stats: {df['sequence'].str.len().mean():.1f} ± {df['sequence'].str.len().std():.1f} nt")
    logger.info(f"  Features: {len(df.columns)} columns")
    
    if 'prediction_type' in df.columns:
        logger.info("  Class distribution:")
        pred_counts = df['prediction_type'].value_counts()
        for pred_type, count in pred_counts.items():
            logger.info(f"    {pred_type}: {count} samples ({count/len(df)*100:.1f}%)")
    
    # Estimate memory usage
    memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
    logger.info(f"  Memory usage: {memory_mb:.2f} MB")
    logger.info("=" * 60)
    
    return df

def main():
    parser = argparse.ArgumentParser(
        description="Prepare minimal test datasets for error model workflow validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with defaults
  %(prog)s --input data/analysis_sequences.tsv --output data/test.tsv
  
  # Custom context and sample size
  %(prog)s --input data/full.tsv --output data/minimal.tsv \
           --context-length 100 --sample-size 500
  
  # Select specific features for faster testing
  %(prog)s --input data/full.tsv --output data/quick_test.tsv \
           --features score donor_score acceptor_score
        """
    )
    
    parser.add_argument(
        "--input", 
        type=Path, 
        required=True, 
        help="Input TSV file (analysis_sequences_* artifact)"
    )
    parser.add_argument(
        "--output", 
        type=Path, 
        required=True, 
        help="Output TSV file for test data"
    )
    parser.add_argument(
        "--context-length", 
        type=int, 
        default=50, 
        help="Context length in nucleotides (default: 50 for fast testing)"
    )
    parser.add_argument(
        "--sample-size", 
        type=int, 
        default=1000, 
        help="Number of samples to use (balanced across classes)"
    )
    parser.add_argument(
        "--features", 
        nargs='+', 
        default=None,
        help="Feature columns to include (default: all features)"
    )
    parser.add_argument(
        "--use-polars",
        action="store_true",
        help="Use Polars for faster processing of large files"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling"
    )
    
    args = parser.parse_args()
    
    # Validate input file exists
    if not args.input.exists():
        parser.error(f"Input file does not exist: {args.input}")
    
    # Use default test features if none specified
    features = args.features if args.features else DEFAULT_TEST_FEATURES
    
    prepare_minimal_test_data(
        input_file=args.input,
        output_file=args.output,
        context_length=args.context_length,
        feature_columns=features,
        sample_size=args.sample_size,
        use_polars=args.use_polars,
        random_seed=args.seed
    )

if __name__ == "__main__":
    main()
