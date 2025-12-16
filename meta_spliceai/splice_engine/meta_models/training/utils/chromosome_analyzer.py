#!/usr/bin/env python3
"""
Chromosome-aware analysis utility.

This script provides comprehensive analysis for chromosome-aware cross-validation
results, including performance analysis, feature importance, and visualization.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import numpy as np


def analyze_chromosome_distribution(dataset_path: str) -> Dict:
    """
    Analyze chromosome distribution in a dataset.
    
    Args:
        dataset_path: Path to the dataset directory
        
    Returns:
        Dictionary containing chromosome distribution analysis
    """
    print("📊 CHROMOSOME DISTRIBUTION ANALYSIS")
    print("-" * 40)
    
    try:
        import glob
        
        # Find batch files
        batch_files = glob.glob(f'{dataset_path}/batch_*.parquet')
        if not batch_files:
            batch_files = glob.glob(f'{dataset_path}/master/batch_*.parquet')
        
        if not batch_files:
            print(f"❌ No batch files found in {dataset_path}")
            return {'error': 'No batch files found'}
        
        print(f"Found {len(batch_files)} batch files")
        
        # Analyze first few batches for chromosome distribution
        chrom_counts = {}
        total_rows = 0
        
        for i, batch_file in enumerate(batch_files[:5]):  # Sample first 5 batches
            df = pd.read_parquet(batch_file)
            batch_chrom_counts = df['chrom'].value_counts()
            
            for chrom, count in batch_chrom_counts.items():
                chrom_counts[chrom] = chrom_counts.get(chrom, 0) + count
            
            total_rows += len(df)
            print(f"  Batch {i+1}: {len(df):,} rows")
        
        # Sort chromosomes
        sorted_chroms = sorted(chrom_counts.items(), key=lambda x: (
            int(x[0]) if str(x[0]).isdigit() else (25 if x[0] == 'X' else (26 if x[0] == 'Y' else 27))
        ))
        
        print(f"\nChromosome distribution (sample of {total_rows:,} rows):")
        for chrom, count in sorted_chroms:
            percentage = (count / total_rows) * 100
            print(f"  Chr {chrom}: {count:,} ({percentage:.1f}%)")
        
        return {
            'total_rows_sampled': total_rows,
            'chromosome_counts': dict(sorted_chroms),
            'n_chromosomes': len(chrom_counts)
        }
        
    except Exception as e:
        print(f"❌ Failed to analyze chromosome distribution: {e}")
        return {'error': str(e)}


def analyze_chromosome_cv_performance(results_dir: str) -> Dict:
    """
    Analyze chromosome-specific CV performance.
    
    Args:
        results_dir: Directory containing chromosome CV results
        
    Returns:
        Dictionary containing performance analysis
    """
    print("\n🧬 CHROMOSOME CV PERFORMANCE ANALYSIS")
    print("-" * 45)
    
    results_dir = Path(results_dir)
    
    # Look for LOCO CV metrics
    cv_metrics_file = results_dir / 'loco_cv_metrics.csv'
    if not cv_metrics_file.exists():
        print(f"❌ CV metrics file not found: {cv_metrics_file}")
        return {'error': 'CV metrics file not found'}
    
    try:
        cv_metrics = pd.read_csv(cv_metrics_file)
        print(f"Loaded CV results for {len(cv_metrics)} chromosome folds")
        
        # Analyze performance by chromosome
        if 'test_chromosome' in cv_metrics.columns and 'test_macro_f1' in cv_metrics.columns:
            chrom_performance = cv_metrics[['test_chromosome', 'test_macro_f1', 'test_accuracy']].copy()
            chrom_performance = chrom_performance.sort_values('test_macro_f1', ascending=False)
            
            print(f"\nPerformance by test chromosome (sorted by F1):")
            for _, row in chrom_performance.iterrows():
                chrom = row['test_chromosome']
                f1 = row['test_macro_f1']
                acc = row['test_accuracy']
                print(f"  Chr {chrom}: F1={f1:.3f}, Acc={acc:.3f}")
            
            # Summary statistics
            f1_mean = cv_metrics['test_macro_f1'].mean()
            f1_std = cv_metrics['test_macro_f1'].std()
            f1_min = cv_metrics['test_macro_f1'].min()
            f1_max = cv_metrics['test_macro_f1'].max()
            
            print(f"\nOverall performance statistics:")
            print(f"  Mean F1: {f1_mean:.3f} ± {f1_std:.3f}")
            print(f"  Range: {f1_min:.3f} - {f1_max:.3f}")
            print(f"  Coefficient of variation: {(f1_std/f1_mean)*100:.1f}%")
            
            return {
                'chromosome_performance': chrom_performance.to_dict('records'),
                'summary_stats': {
                    'mean_f1': f1_mean,
                    'std_f1': f1_std,
                    'min_f1': f1_min,
                    'max_f1': f1_max,
                    'cv_f1': (f1_std/f1_mean)*100
                }
            }
        else:
            print("❌ Required columns not found in CV metrics")
            return {'error': 'Required columns not found'}
            
    except Exception as e:
        print(f"❌ Failed to analyze CV performance: {e}")
        return {'error': str(e)}


def quick_chromosome_check(dataset_path: str) -> None:
    """Quick chromosome distribution check for a single batch."""
    try:
        batch_file = f"{dataset_path}/batch_00001.parquet"
        if not Path(batch_file).exists():
            batch_file = f"{dataset_path}/master/batch_00001.parquet"
        
        if Path(batch_file).exists():
            df = pd.read_parquet(batch_file)
            chrom_dist = df['chrom'].value_counts().sort_index()
            print("Quick chromosome check (first batch):")
            for chrom, count in chrom_dist.items():
                print(f"  Chr {chrom}: {count}")
        else:
            print(f"❌ No batch file found for quick check")
    except Exception as e:
        print(f"❌ Quick check failed: {e}")


def quick_cv_results_check(results_dir: str) -> None:
    """Quick CV results check."""
    try:
        cv_file = f"{results_dir}/loco_cv_metrics.csv"
        if Path(cv_file).exists():
            cv = pd.read_csv(cv_file)
            if 'test_chromosome' in cv.columns and 'test_macro_f1' in cv.columns:
                top_chroms = cv[['test_chromosome', 'test_macro_f1']].sort_values('test_macro_f1', ascending=False)
                print("Quick CV results check (top 5 chromosomes by F1):")
                for _, row in top_chroms.head().iterrows():
                    print(f"  Chr {row['test_chromosome']}: F1={row['test_macro_f1']:.3f}")
            else:
                print("❌ Required columns not found in CV results")
        else:
            print(f"❌ CV results file not found: {cv_file}")
    except Exception as e:
        print(f"❌ Quick CV check failed: {e}")


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(
        description="Chromosome-aware analysis utility",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze chromosome distribution
  python chromosome_analyzer.py \\
    --dataset train_pc_5000_3mers_diverse/master \\
    --analysis distribution

  # Analyze CV performance
  python chromosome_analyzer.py \\
    --results-dir results/chromosome_cv_pc_5000_3mers_diverse \\
    --analysis performance

  # Quick checks
  python chromosome_analyzer.py \\
    --dataset train_pc_5000_3mers_diverse/master \\
    --analysis quick-distribution
    
  python chromosome_analyzer.py \\
    --results-dir results/chromosome_cv \\
    --analysis quick-cv
        """
    )
    
    parser.add_argument("--dataset", 
                       help="Path to dataset directory")
    parser.add_argument("--results-dir", 
                       help="Path to results directory")
    parser.add_argument("--analysis", required=True,
                       choices=['distribution', 'performance', 'quick-distribution', 'quick-cv'],
                       help="Type of analysis to perform")
    parser.add_argument("--output", 
                       help="Output file for analysis results (JSON format)")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose output")
    
    args = parser.parse_args()
    
    print("🧬 CHROMOSOME-AWARE ANALYSIS")
    print("=" * 50)
    
    results = {}
    
    if args.analysis == 'distribution':
        if not args.dataset:
            print("❌ --dataset is required for distribution analysis")
            sys.exit(1)
        results = analyze_chromosome_distribution(args.dataset)
        
    elif args.analysis == 'performance':
        if not args.results_dir:
            print("❌ --results-dir is required for performance analysis")
            sys.exit(1)
        results = analyze_chromosome_cv_performance(args.results_dir)
        
    elif args.analysis == 'quick-distribution':
        if not args.dataset:
            print("❌ --dataset is required for quick distribution check")
            sys.exit(1)
        quick_chromosome_check(args.dataset)
        results = {'analysis': 'quick-distribution', 'completed': True}
        
    elif args.analysis == 'quick-cv':
        if not args.results_dir:
            print("❌ --results-dir is required for quick CV check")
            sys.exit(1)
        quick_cv_results_check(args.results_dir)
        results = {'analysis': 'quick-cv', 'completed': True}
    
    # Save results if requested
    if args.output and results:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n💾 Analysis results saved to: {output_path}")
    
    if 'error' in results:
        print(f"\n❌ Analysis failed: {results['error']}")
        sys.exit(1)
    else:
        print(f"\n🎉 Chromosome analysis completed successfully!")
        sys.exit(0)


if __name__ == "__main__":
    main()




