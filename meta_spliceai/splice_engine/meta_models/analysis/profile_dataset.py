#!/usr/bin/env python3
"""
Dataset profiling module for splice site prediction datasets.

This module provides comprehensive profiling capabilities for splice site prediction
datasets, including statistical analysis, visualization, and quality assessment.
Refactored to use helper modules for better maintainability.

Usage:
    python profile_dataset.py <dataset_path> [options]

Example:
    python profile_dataset.py data/datasets/train_dataset.parquet --output results/ --plots
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Optional, List

# Import the core profiler class
from meta_spliceai.splice_engine.meta_models.analysis.dataset_profiler_core import (
    SpliceDatasetProfiler
)

def save_profile_to_json(profile: dict, output_path: Path):
    """Save profile results to JSON file."""
    json_path = output_path / "dataset_profile.json"
    
    # Convert numpy types to native Python types for JSON serialization
    def convert_numpy_types(obj):
        if hasattr(obj, 'item'):
            return obj.item()
        elif hasattr(obj, 'tolist'):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(v) for v in obj]
        else:
            return obj
    
    serializable_profile = convert_numpy_types(profile)
    
    with open(json_path, 'w') as f:
        json.dump(serializable_profile, f, indent=2, default=str)
    
    print(f"Profile saved to: {json_path}")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Profile splice site prediction datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s data/train_dataset.parquet
  %(prog)s data/datasets/ --output results/ --plots
  %(prog)s data/train_dataset.parquet --genes BRCA1,TP53 --max-files 5
  %(prog)s data/train_dataset.parquet --sample-rows 10000 --quiet
        """
    )
    
    # Required arguments
    parser.add_argument(
        'dataset_path',
        help='Path to dataset file (.parquet) or directory containing parquet files'
    )
    
    # Optional arguments
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='dataset_profile_results',
        help='Output directory for results (default: dataset_profile_results)'
    )
    
    parser.add_argument(
        '--plots', '-p',
        action='store_true',
        help='Generate visualization plots'
    )
    
    parser.add_argument(
        '--genes', '-g',
        type=str,
        help='Comma-separated list of gene IDs to filter for (e.g., ENSG00000139618,ENSG00000141510)'
    )
    
    parser.add_argument(
        '--max-files',
        type=int,
        help='Maximum number of parquet files to process (for testing)'
    )
    
    parser.add_argument(
        '--sample-rows',
        type=int,
        help='Sample only N rows per file (for testing)'
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress verbose output'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=100000,
        help='Batch size for streaming processing (default: 100000)'
    )
    
    return parser.parse_args()


def main():
    """Main entry point for dataset profiling."""
    args = parse_args()
    
    # Validate dataset path
    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        print(f"Error: Dataset path does not exist: {dataset_path}")
        sys.exit(1)
    
    # Parse gene filter
    gene_filter = None
    if args.genes:
        gene_filter = [gene.strip() for gene in args.genes.split(',')]
        print(f"Gene filter: {gene_filter}")
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize profiler
    verbose = not args.quiet
    profiler = SpliceDatasetProfiler(verbose=verbose, batch_size=args.batch_size)
    
    try:
        print(f"\n🔬 Starting dataset profiling...")
        print(f"Dataset: {dataset_path}")
        print(f"Output: {output_dir}")
        if args.plots:
            print("Visualizations: Enabled")
        
        # Profile the dataset
        profile = profiler.profile_dataset(
            dataset_path=str(dataset_path),
            output_dir=str(output_dir) if args.plots else None,
            generate_plots=args.plots,
            gene_filter=gene_filter,
            max_files=args.max_files,
            sample_rows=args.sample_rows
        )
        
        # Check for errors
        if 'error' in profile:
            print(f"Error during profiling: {profile['error']}")
            sys.exit(1)
        
        # Print summary to console
        profiler.print_summary(profile)
        
        # Save results to JSON
        save_profile_to_json(profile, output_dir)
        
        # Print visualization info
        if args.plots and 'visualizations' in profile:
            print(f"\n📊 Generated {len(profile['visualizations'])} visualization plots:")
            for plot_name, plot_path in profile['visualizations'].items():
                print(f"   • {plot_name}: {plot_path}")
        
        print(f"\n✅ Profiling completed successfully!")
        print(f"Results saved to: {output_dir}")
        
    except KeyboardInterrupt:
        print("\n⚠️  Profiling interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during profiling: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
