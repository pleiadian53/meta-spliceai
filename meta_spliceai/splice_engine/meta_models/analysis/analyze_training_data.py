#!/usr/bin/env python3
"""
Training Data Analysis Script for Splice-Surveyor

This script provides comprehensive analysis of ML training data in parquet format,
specifically designed for splice site prediction datasets.

Usage:
    python analyze_training_data.py <path_to_parquet_file_or_directory> [options]

Examples:
    python analyze_training_data.py /path/to/data.parquet
    python analyze_training_data.py /path/to/data_directory/ --output report.json
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import warnings

import pandas as pd
import numpy as np
from glob import glob


class TrainingDataAnalyzer:
    """Comprehensive analyzer for ML training data in parquet format."""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.results = {}
        
    def log(self, message: str, level: str = "INFO"):
        """Log messages with timestamp."""
        if self.verbose:
            print(f"[{level}] {message}")
    
    def find_parquet_files(self, path: str) -> List[str]:
        """Find all parquet files in given path."""
        path_obj = Path(path)
        
        if path_obj.is_file() and path_obj.suffix in ['.parquet', '.pq']:
            return [str(path_obj)]
        elif path_obj.is_dir():
            parquet_files = []
            for pattern in ['*.parquet', '*.pq']:
                parquet_files.extend(glob(str(path_obj / pattern)))
            return sorted(parquet_files)
        else:
            raise ValueError(f"Path {path} is neither a parquet file nor a directory")
    
    def analyze_basic_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze basic dataset information."""
        self.log("Analyzing basic dataset information...")
        
        basic_info = {
            'sample_size': len(df),
            'num_features': len(df.columns),
            'dataset_shape': df.shape,
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2
        }
        
        # Column information
        column_info = {}
        for col in df.columns:
            column_info[col] = {
                'dtype': str(df[col].dtype),
                'unique_values': df[col].nunique() if df[col].dtype != 'object' or df[col].nunique() < 1000 else 'high_cardinality'
            }
        
        basic_info['columns'] = column_info
        return basic_info
    
    def analyze_null_values(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze null values in the dataset."""
        self.log("Analyzing null values...")
        
        null_counts = df.isnull().sum()
        total_nulls = null_counts.sum()
        
        null_analysis = {
            'total_null_values': int(total_nulls),
            'has_nulls': total_nulls > 0,
            'null_percentage': float(total_nulls / (len(df) * len(df.columns)) * 100)
        }
        
        if total_nulls > 0:
            null_by_column = {}
            for col, null_count in null_counts.items():
                if null_count > 0:
                    null_by_column[col] = {
                        'count': int(null_count),
                        'percentage': float(null_count / len(df) * 100)
                    }
            null_analysis['null_by_column'] = null_by_column
        
        return null_analysis
    
    def analyze_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze data quality issues."""
        self.log("Analyzing data quality...")
        
        quality_analysis = {
            'duplicate_rows': int(df.duplicated().sum()),
            'duplicate_percentage': float(df.duplicated().sum() / len(df) * 100)
        }
        
        # Check for infinite values in numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        inf_counts = {}
        
        for col in numeric_cols:
            inf_count = np.isinf(df[col]).sum()
            if inf_count > 0:
                inf_counts[col] = int(inf_count)
        
        quality_analysis['infinite_values'] = inf_counts
        quality_analysis['has_infinite_values'] = len(inf_counts) > 0
        
        return quality_analysis
    
    def analyze_feature_types(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Categorize and analyze different types of features."""
        self.log("Analyzing feature types...")
        
        feature_categories = {
            'identifiers': [],
            'scores': [],
            'positions': [],
            'sequence_features': [],
            'structural_features': [],
            'derived_features': [],
            'metadata': [],
            'other': []
        }
        
        for col in df.columns:
            col_lower = col.lower()
            
            # Categorize columns based on naming patterns
            if any(x in col_lower for x in ['id', 'gene_id', 'transcript_id']):
                feature_categories['identifiers'].append(col)
            elif any(x in col_lower for x in ['score', 'probability', 'odds']):
                feature_categories['scores'].append(col)
            elif any(x in col_lower for x in ['position', 'start', 'end', 'distance']):
                feature_categories['positions'].append(col)
            elif any(x in col_lower for x in ['6mer_', 'gc_content', 'sequence', 'complexity']):
                feature_categories['sequence_features'].append(col)
            elif any(x in col_lower for x in ['length', 'exon', 'intron', 'overlap']):
                feature_categories['structural_features'].append(col)
            elif any(x in col_lower for x in ['context', 'diff', 'ratio', 'peak', 'signal']):
                feature_categories['derived_features'].append(col)
            elif any(x in col_lower for x in ['type', 'strand', 'chrom', 'has_', 'missing']):
                feature_categories['metadata'].append(col)
            else:
                feature_categories['other'].append(col)
        
        # Count features by category
        feature_counts = {cat: len(features) for cat, features in feature_categories.items()}
        
        return {
            'feature_categories': feature_categories,
            'feature_counts': feature_counts
        }
    
    def analyze_target_distribution(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze target variable distributions."""
        self.log("Analyzing target distributions...")
        
        target_analysis = {}
        
        # Common target columns for splice site prediction
        potential_targets = ['splice_type', 'pred_type', 'label', 'target', 'class']
        
        for col in potential_targets:
            if col in df.columns:
                if df[col].dtype == 'object' or df[col].nunique() < 20:
                    value_counts = df[col].value_counts()
                    target_analysis[col] = {
                        'unique_values': int(df[col].nunique()),
                        'distribution': value_counts.to_dict(),
                        'balance_ratio': float(value_counts.min() / value_counts.max()) if len(value_counts) > 1 else 1.0
                    }
        
        return target_analysis
    
    def analyze_numeric_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze numeric feature statistics."""
        self.log("Analyzing numeric features...")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            return {'message': 'No numeric columns found'}
        
        # Basic statistics
        stats_df = df[numeric_cols].describe()
        
        # Additional statistics
        numeric_analysis = {
            'num_numeric_features': len(numeric_cols),
            'statistics': stats_df.to_dict(),
            'zero_variance_features': [],
            'high_correlation_pairs': [],
            'outlier_features': []
        }
        
        # Find zero variance features
        for col in numeric_cols:
            if df[col].var() == 0:
                numeric_analysis['zero_variance_features'].append(col)
        
        # Sample correlation analysis (limit to avoid memory issues)
        sample_cols = numeric_cols[:50] if len(numeric_cols) > 50 else numeric_cols
        if len(sample_cols) > 1:
            corr_matrix = df[sample_cols].corr()
            high_corr_pairs = []
            
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) > 0.9:
                        high_corr_pairs.append({
                            'feature1': corr_matrix.columns[i],
                            'feature2': corr_matrix.columns[j],
                            'correlation': float(corr_val)
                        })
            
            numeric_analysis['high_correlation_pairs'] = high_corr_pairs[:10]  # Limit output
        
        return numeric_analysis
    
    def generate_sample_data(self, df: pd.DataFrame, n_samples: int = 5) -> Dict[str, Any]:
        """Generate sample data for inspection."""
        self.log("Generating sample data...")
        
        sample_data = {
            'first_rows': df.head(n_samples).to_dict('records'),
            'random_rows': df.sample(n=min(n_samples, len(df)), random_state=42).to_dict('records')
        }
        
        return sample_data
    
    def analyze_parquet_file(self, file_path: str) -> Dict[str, Any]:
        """Analyze a single parquet file."""
        self.log(f"Analyzing parquet file: {file_path}")
        
        try:
            # Read parquet file
            df = pd.read_parquet(file_path)
            
            analysis_results = {
                'file_path': file_path,
                'file_size_mb': os.path.getsize(file_path) / 1024**2,
                'basic_info': self.analyze_basic_info(df),
                'null_analysis': self.analyze_null_values(df),
                'quality_analysis': self.analyze_data_quality(df),
                'feature_analysis': self.analyze_feature_types(df),
                'target_analysis': self.analyze_target_distribution(df),
                'numeric_analysis': self.analyze_numeric_features(df),
                'sample_data': self.generate_sample_data(df)
            }
            
            return analysis_results
            
        except Exception as e:
            self.log(f"Error analyzing {file_path}: {str(e)}", "ERROR")
            return {'file_path': file_path, 'error': str(e)}
    
    def analyze_multiple_files(self, file_paths: List[str]) -> Dict[str, Any]:
        """Analyze multiple parquet files."""
        self.log(f"Analyzing {len(file_paths)} parquet files...")
        
        all_results = {}
        summary_stats = {
            'total_files': len(file_paths),
            'successful_analyses': 0,
            'failed_analyses': 0,
            'total_samples': 0,
            'total_features': set(),
            'file_sizes_mb': []
        }
        
        for file_path in file_paths:
            result = self.analyze_parquet_file(file_path)
            file_name = os.path.basename(file_path)
            all_results[file_name] = result
            
            if 'error' not in result:
                summary_stats['successful_analyses'] += 1
                summary_stats['total_samples'] += result['basic_info']['sample_size']
                summary_stats['total_features'].update(result['basic_info']['columns'].keys())
                summary_stats['file_sizes_mb'].append(result['file_size_mb'])
            else:
                summary_stats['failed_analyses'] += 1
        
        summary_stats['total_features'] = len(summary_stats['total_features'])
        summary_stats['avg_file_size_mb'] = np.mean(summary_stats['file_sizes_mb']) if summary_stats['file_sizes_mb'] else 0
        
        return {
            'summary': summary_stats,
            'individual_results': all_results
        }
    
    def generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on analysis."""
        recommendations = []
        
        if 'individual_results' in results:
            # Multi-file analysis
            for file_name, file_result in results['individual_results'].items():
                if 'error' in file_result:
                    continue
                recommendations.extend(self._get_file_recommendations(file_result))
        else:
            # Single file analysis
            recommendations.extend(self._get_file_recommendations(results))
        
        return list(set(recommendations))  # Remove duplicates
    
    def _get_file_recommendations(self, result: Dict[str, Any]) -> List[str]:
        """Get recommendations for a single file analysis."""
        recommendations = []
        
        # Null value recommendations
        if result['null_analysis']['has_nulls']:
            recommendations.append("⚠️  Address missing values - consider imputation or removal strategies")
        
        # Duplicate recommendations
        if result['quality_analysis']['duplicate_percentage'] > 10:
            recommendations.append("⚠️  High percentage of duplicate rows detected - investigate and potentially remove")
        
        # Infinite value recommendations
        if result['quality_analysis']['has_infinite_values']:
            recommendations.append("⚠️  Infinite values found - replace or cap extreme values")
        
        # Feature recommendations
        feature_count = result['basic_info']['num_features']
        if feature_count > 1000:
            recommendations.append("📊 Consider dimensionality reduction or feature selection with >1000 features")
        
        # Target balance recommendations
        if 'splice_type' in result['target_analysis']:
            balance_ratio = result['target_analysis']['splice_type'].get('balance_ratio', 1.0)
            if balance_ratio < 0.3:
                recommendations.append("⚖️  Class imbalance detected - consider resampling or class weighting")
        
        # Memory recommendations
        memory_mb = result['basic_info']['memory_usage_mb']
        if memory_mb > 5000:  # > 5GB
            recommendations.append("💾 Large dataset detected - consider batch processing or data chunking")
        
        # Zero variance recommendations
        zero_var_features = result['numeric_analysis'].get('zero_variance_features', [])
        if zero_var_features:
            recommendations.append(f"🔍 Remove {len(zero_var_features)} zero-variance features")
        
        return recommendations


def main():
    """Main function to run the training data analysis."""
    parser = argparse.ArgumentParser(
        description="Analyze ML training data in parquet format for splice site prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s /path/to/data.parquet
  %(prog)s /path/to/data_directory/ --output report.json --verbose
  %(prog)s /mnt/nfs1/splice-surveyor/data/ensembl/spliceai_eval/meta_models/train_pc_1000/master
        """
    )
    
    parser.add_argument(
        'path',
        help='Path to parquet file or directory containing parquet files'
    )
    
    parser.add_argument(
        '--output', '-o',
        help='Output file path for JSON report (default: print to stdout)',
        default=None
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--no-samples',
        action='store_true',
        help='Skip sample data generation to reduce output size'
    )
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = TrainingDataAnalyzer(verbose=args.verbose)
    
    try:
        # Find parquet files
        parquet_files = analyzer.find_parquet_files(args.path)
        
        if not parquet_files:
            print(f"No parquet files found in {args.path}")
            sys.exit(1)
        
        analyzer.log(f"Found {len(parquet_files)} parquet file(s)")
        
        # Analyze files
        if len(parquet_files) == 1:
            results = analyzer.analyze_parquet_file(parquet_files[0])
        else:
            results = analyzer.analyze_multiple_files(parquet_files)
        
        # Remove sample data if requested
        if args.no_samples:
            if 'sample_data' in results:
                del results['sample_data']
            elif 'individual_results' in results:
                for file_result in results['individual_results'].values():
                    if 'sample_data' in file_result:
                        del file_result['sample_data']
        
        # Generate recommendations
        recommendations = analyzer.generate_recommendations(results)
        results['recommendations'] = recommendations
        
        # Output results
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            analyzer.log(f"Analysis report saved to {args.output}")
        else:
            print(json.dumps(results, indent=2, default=str))
        
        # Print summary
        if args.verbose:
            print("\n" + "="*60)
            print("ANALYSIS SUMMARY")
            print("="*60)
            
            if 'summary' in results:
                summary = results['summary']
                print(f"Files analyzed: {summary['successful_analyses']}/{summary['total_files']}")
                print(f"Total samples: {summary['total_samples']:,}")
                print(f"Unique features: {summary['total_features']}")
            else:
                print(f"Sample size: {results['basic_info']['sample_size']:,}")
                print(f"Features: {results['basic_info']['num_features']}")
                print(f"Memory usage: {results['basic_info']['memory_usage_mb']:.1f} MB")
            
            print(f"\nRecommendations ({len(recommendations)}):")
            for rec in recommendations:
                print(f"  {rec}")
    
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()