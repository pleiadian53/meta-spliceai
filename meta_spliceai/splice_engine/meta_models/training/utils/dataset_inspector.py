#!/usr/bin/env python3
"""
Dataset quality assessment utility.

This script provides comprehensive quality assessment for training datasets,
including schema validation, memory usage analysis, and data completeness checks.
"""

import argparse
import glob
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


def inspect_dataset_quality(dataset_path: str, output_file: Optional[str] = None) -> Dict:
    """
    Perform comprehensive quality assessment of a training dataset.
    
    Args:
        dataset_path: Path to the dataset directory
        output_file: Optional output file for detailed results
        
    Returns:
        Dictionary containing quality assessment results
    """
    dataset_path = Path(dataset_path)
    
    print("🔍 DATASET QUALITY ASSESSMENT")
    print("=" * 50)
    print(f"Analyzing dataset: {dataset_path}")
    
    # Find batch files
    if dataset_path.is_dir():
        batch_files = list(dataset_path.glob("batch_*.parquet"))
        if not batch_files:
            batch_files = list(dataset_path.glob("*.parquet"))
    else:
        print(f"❌ Dataset path not found: {dataset_path}")
        return {'error': 'Dataset path not found'}
    
    if not batch_files:
        print(f"❌ No parquet files found in: {dataset_path}")
        return {'error': 'No parquet files found'}
    
    print(f"📊 Found {len(batch_files)} batch files")
    
    # Initialize results
    results = {
        'dataset_path': str(dataset_path),
        'n_batch_files': len(batch_files),
        'batch_analysis': [],
        'summary': {},
        'quality_issues': []
    }
    
    # Analyze each batch
    total_rows = 0
    total_memory_mb = 0
    all_columns = set()
    column_counts = {}
    
    print(f"\n📋 BATCH-BY-BATCH ANALYSIS:")
    print("-" * 40)
    
    for i, batch_file in enumerate(sorted(batch_files)[:10]):  # Analyze first 10 batches
        try:
            df = pd.read_parquet(batch_file)
            
            batch_info = {
                'file': batch_file.name,
                'shape': df.shape,
                'columns': len(df.columns),
                'memory_mb': df.memory_usage(deep=True).sum() / 1024**2,
                'dtypes': df.dtypes.value_counts().to_dict()
            }
            
            results['batch_analysis'].append(batch_info)
            
            total_rows += df.shape[0]
            total_memory_mb += batch_info['memory_mb']
            all_columns.update(df.columns)
            column_counts[batch_file.name] = len(df.columns)
            
            print(f"  {batch_file.name}: {df.shape[0]:,} rows, {df.shape[1]} cols, {batch_info['memory_mb']:.1f} MB")
            
        except Exception as e:
            error_info = {'file': batch_file.name, 'error': str(e)}
            results['batch_analysis'].append(error_info)
            results['quality_issues'].append(f"Failed to read {batch_file.name}: {e}")
            print(f"  ❌ {batch_file.name}: Error - {e}")
    
    if len(batch_files) > 10:
        print(f"  ... and {len(batch_files) - 10} more files")
    
    # Check for schema consistency
    unique_column_counts = set(column_counts.values())
    if len(unique_column_counts) > 1:
        results['quality_issues'].append(f"Inconsistent column counts: {unique_column_counts}")
        print(f"⚠️ Schema inconsistency detected: {unique_column_counts} different column counts")
    else:
        print(f"✅ Consistent schema: {list(unique_column_counts)[0]} columns per batch")
    
    # Sample first batch for detailed analysis
    if batch_files and not results['batch_analysis'][0].get('error'):
        sample_batch = sorted(batch_files)[0]
        try:
            df_sample = pd.read_parquet(sample_batch)
            
            print(f"\n🔬 DETAILED ANALYSIS (Sample: {sample_batch.name}):")
            print("-" * 50)
            
            # Check for required columns
            required_cols = ['gene_id', 'position', 'chrom', 'splice_type']
            missing_cols = [col for col in required_cols if col not in df_sample.columns]
            
            if missing_cols:
                results['quality_issues'].append(f"Missing required columns: {missing_cols}")
                print(f"⚠️ Missing required columns: {missing_cols}")
            else:
                print(f"✅ All required columns present: {required_cols}")
            
            # Data type analysis
            print(f"\n📊 Data Types:")
            for dtype, count in df_sample.dtypes.value_counts().items():
                print(f"  {dtype}: {count} columns")
            
            # Memory usage breakdown
            print(f"\n💾 Memory Usage:")
            memory_by_dtype = df_sample.select_dtypes(include=['object']).memory_usage(deep=True).sum() / 1024**2
            print(f"  Object columns: {memory_by_dtype:.1f} MB")
            memory_numeric = df_sample.select_dtypes(exclude=['object']).memory_usage(deep=True).sum() / 1024**2
            print(f"  Numeric columns: {memory_numeric:.1f} MB")
            print(f"  Total: {(memory_by_dtype + memory_numeric):.1f} MB")
            
            # Check for null values
            null_counts = df_sample.isnull().sum()
            null_columns = null_counts[null_counts > 0]
            if len(null_columns) > 0:
                results['quality_issues'].append(f"Null values found in {len(null_columns)} columns")
                print(f"⚠️ Null values found in {len(null_columns)} columns:")
                for col, count in null_columns.head().items():
                    print(f"    {col}: {count} nulls ({count/len(df_sample)*100:.1f}%)")
            else:
                print(f"✅ No null values detected")
            
            # Check splice_type distribution if present
            if 'splice_type' in df_sample.columns:
                splice_dist = df_sample['splice_type'].value_counts()
                print(f"\n🧬 Splice Type Distribution:")
                for splice_type, count in splice_dist.items():
                    print(f"  {splice_type}: {count:,} ({count/len(df_sample)*100:.1f}%)")
                
                results['summary']['splice_type_distribution'] = splice_dist.to_dict()
            
        except Exception as e:
            results['quality_issues'].append(f"Failed detailed analysis: {e}")
            print(f"❌ Detailed analysis failed: {e}")
    
    # Generate summary
    results['summary'].update({
        'total_estimated_rows': total_rows * len(batch_files) // min(len(batch_files), 10),
        'total_estimated_memory_gb': total_memory_mb * len(batch_files) / min(len(batch_files), 10) / 1024,
        'unique_columns': len(all_columns),
        'schema_consistent': len(unique_column_counts) == 1,
        'quality_score': max(0, 100 - len(results['quality_issues']) * 10)
    })
    
    print(f"\n📈 SUMMARY:")
    print("-" * 20)
    print(f"  Estimated total rows: {results['summary']['total_estimated_rows']:,}")
    print(f"  Estimated memory usage: {results['summary']['total_estimated_memory_gb']:.1f} GB")
    print(f"  Quality score: {results['summary']['quality_score']}/100")
    print(f"  Issues found: {len(results['quality_issues'])}")
    
    if results['quality_issues']:
        print(f"\n⚠️ QUALITY ISSUES:")
        for issue in results['quality_issues']:
            print(f"  - {issue}")
    else:
        print(f"\n✅ NO QUALITY ISSUES DETECTED")
    
    # Save results if requested
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n💾 Quality assessment saved to: {output_path}")
    
    return results


def check_environment():
    """Check if required packages are available."""
    try:
        import polars
        import xgboost
        import sklearn
        print('✅ Environment ready')
        return True
    except ImportError as e:
        print(f'❌ Missing dependency: {e}')
        return False


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(
        description="Dataset quality assessment utility",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check environment
  python dataset_inspector.py --check-env

  # Assess dataset quality
  python dataset_inspector.py \\
    --dataset train_pc_5000_3mers_diverse/master \\
    --output quality_report.json

  # Quick assessment without output file
  python dataset_inspector.py --dataset train_pc_5000_3mers_diverse/master
        """
    )
    
    parser.add_argument("--dataset", 
                       help="Path to dataset directory")
    parser.add_argument("--output", 
                       help="Output file for quality report (JSON format)")
    parser.add_argument("--check-env", action="store_true",
                       help="Check if required packages are available")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose output")
    
    args = parser.parse_args()
    
    if args.check_env:
        success = check_environment()
        sys.exit(0 if success else 1)
    
    if not args.dataset:
        print("❌ Dataset path is required. Use --dataset to specify.")
        parser.print_help()
        sys.exit(1)
    
    # Run quality assessment
    results = inspect_dataset_quality(args.dataset, args.output)
    
    if 'error' in results:
        print(f"❌ Quality assessment failed: {results['error']}")
        sys.exit(1)
    
    # Exit with appropriate code based on quality
    quality_score = results['summary'].get('quality_score', 0)
    if quality_score < 70:
        print(f"\n⚠️ Low quality score ({quality_score}/100). Review issues before training.")
        sys.exit(1)
    else:
        print(f"\n🎉 Dataset quality assessment completed successfully!")
        sys.exit(0)


if __name__ == "__main__":
    main()




