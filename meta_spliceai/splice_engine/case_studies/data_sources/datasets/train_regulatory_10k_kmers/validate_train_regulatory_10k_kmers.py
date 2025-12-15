#!/usr/bin/env python3
"""
Validation script for train_regulatory_10k_kmers dataset

This script performs comprehensive validation of the train_regulatory_10k_kmers dataset,
including schema validation, data quality checks, and statistical analysis.

Usage:
    python validate_train_regulatory_10k_kmers.py [--fix] [--verbose] [--output-dir DIR]

Options:
    --fix           Attempt to fix detected issues automatically
    --verbose       Enable detailed logging output
    --output-dir    Directory to save validation reports (default: current directory)
"""

import argparse
import sys
import os
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import polars as pl
import numpy as np
from datetime import datetime

# Add the meta_spliceai package to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from meta_spliceai.splice_engine.meta_models.training import datasets
from meta_spliceai.splice_engine.meta_models.builder.validate_dataset_schema import validate_dataset_schema

class DatasetValidator:
    """Comprehensive validator for train_regulatory_10k_kmers dataset"""
    
    def __init__(self, dataset_path: str, verbose: bool = False, fix_issues: bool = False):
        self.dataset_path = dataset_path
        self.verbose = verbose
        self.fix_issues = fix_issues
        self.validation_results = {}
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging configuration"""
        level = logging.DEBUG if self.verbose else logging.INFO
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(f'validation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def validate_schema(self) -> bool:
        """Validate dataset schema consistency"""
        self.logger.info("🔍 Validating dataset schema...")
        
        try:
            # Use existing schema validation utility
            result = validate_dataset_schema(self.dataset_path, fix_issues=self.fix_issues, verbose=self.verbose)
            
            if result['issues_found'] == 0:
                self.logger.info("✅ Schema validation passed")
                self.validation_results['schema'] = {'status': 'PASS', 'issues': []}
                return True
            else:
                self.logger.error(f"❌ Schema validation failed: {result['issues_found']} issues found")
                self.validation_results['schema'] = {'status': 'FAIL', 'issues': [f"{result['issues_found']} schema issues detected"]}
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Schema validation error: {e}")
            self.validation_results['schema'] = {'status': 'ERROR', 'issues': [str(e)]}
            return False
    
    def validate_data_types(self, df: pl.DataFrame) -> bool:
        """Validate expected data types"""
        self.logger.info("🔍 Validating data types...")
        
        expected_types = {
            'gene_id': pl.Utf8,
            'chr': pl.Utf8,
            'position': pl.Int64,
            'strand': pl.Utf8,
            'delta_score_donor': pl.Float64,
            'delta_score_acceptor': pl.Float64,
            'prob_donor': pl.Float64,
            'prob_acceptor': pl.Float64,
            'is_donor': pl.Boolean,
            'is_acceptor': pl.Boolean,
        }
        
        issues = []
        for col, expected_type in expected_types.items():
            if col in df.columns:
                actual_type = df[col].dtype
                if actual_type != expected_type:
                    issues.append(f"Column {col}: expected {expected_type}, got {actual_type}")
            else:
                issues.append(f"Missing required column: {col}")
        
        if issues:
            self.logger.error(f"❌ Data type validation failed: {len(issues)} issues")
            for issue in issues:
                self.logger.error(f"  - {issue}")
            self.validation_results['data_types'] = {'status': 'FAIL', 'issues': issues}
            return False
        else:
            self.logger.info("✅ Data type validation passed")
            self.validation_results['data_types'] = {'status': 'PASS', 'issues': []}
            return True
    
    def validate_value_ranges(self, df: pl.DataFrame) -> bool:
        """Validate value ranges for numerical columns"""
        self.logger.info("🔍 Validating value ranges...")
        
        issues = []
        
        # Check probability columns (should be 0-1)
        prob_columns = [col for col in df.columns if col.startswith('prob_')]
        for col in prob_columns:
            if col in df.columns:
                min_val = df[col].min()
                max_val = df[col].max()
                if min_val < 0 or max_val > 1:
                    issues.append(f"Column {col}: values outside [0,1] range (min={min_val}, max={max_val})")
        
        # Check chromosome values
        if 'chr' in df.columns:
            valid_chrs = set(map(str, range(1, 23))) | {'X', 'Y'}
            actual_chrs = set(df['chr'].unique().to_list())
            invalid_chrs = actual_chrs - valid_chrs
            if invalid_chrs:
                issues.append(f"Invalid chromosome values: {invalid_chrs}")
        
        # Check strand values
        if 'strand' in df.columns:
            valid_strands = {'+', '-'}
            actual_strands = set(df['strand'].unique().to_list())
            invalid_strands = actual_strands - valid_strands
            if invalid_strands:
                issues.append(f"Invalid strand values: {invalid_strands}")
        
        # Check k-mer columns for reasonable ranges
        kmer_3_cols = [col for col in df.columns if col.startswith('3mer_')]
        kmer_5_cols = [col for col in df.columns if col.startswith('5mer_')]
        
        for col in kmer_3_cols[:5]:  # Check first 5 to avoid too much output
            if col in df.columns:
                max_val = df[col].max()
                if max_val > 100:  # Reasonable upper bound for 3-mer counts
                    issues.append(f"Column {col}: suspiciously high max value ({max_val})")
        
        if issues:
            self.logger.error(f"❌ Value range validation failed: {len(issues)} issues")
            for issue in issues:
                self.logger.error(f"  - {issue}")
            self.validation_results['value_ranges'] = {'status': 'FAIL', 'issues': issues}
            return False
        else:
            self.logger.info("✅ Value range validation passed")
            self.validation_results['value_ranges'] = {'status': 'PASS', 'issues': []}
            return True
    
    def validate_kmer_sequences(self, df: pl.DataFrame) -> bool:
        """Validate k-mer column names contain only valid nucleotides"""
        self.logger.info("🔍 Validating k-mer sequences...")
        
        valid_nucleotides = set('ACGT')
        issues = []
        
        # Check 3-mer columns
        kmer_3_cols = [col for col in df.columns if col.startswith('3mer_')]
        for col in kmer_3_cols:
            kmer_seq = col.replace('3mer_', '')
            if len(kmer_seq) != 3:
                issues.append(f"Invalid 3-mer column name: {col} (sequence length != 3)")
            elif not set(kmer_seq).issubset(valid_nucleotides):
                issues.append(f"Invalid 3-mer sequence: {col} (contains invalid nucleotides)")
        
        # Check 5-mer columns
        kmer_5_cols = [col for col in df.columns if col.startswith('5mer_')]
        for col in kmer_5_cols:
            kmer_seq = col.replace('5mer_', '')
            if len(kmer_seq) != 5:
                issues.append(f"Invalid 5-mer column name: {col} (sequence length != 5)")
            elif not set(kmer_seq).issubset(valid_nucleotides):
                issues.append(f"Invalid 5-mer sequence: {col} (contains invalid nucleotides)")
        
        # Check expected counts
        expected_3mer_count = 4**3  # 64
        expected_5mer_count = 4**5  # 1024
        
        if len(kmer_3_cols) != expected_3mer_count:
            issues.append(f"Expected {expected_3mer_count} 3-mer columns, found {len(kmer_3_cols)}")
        
        if len(kmer_5_cols) != expected_5mer_count:
            issues.append(f"Expected {expected_5mer_count} 5-mer columns, found {len(kmer_5_cols)}")
        
        if issues:
            self.logger.error(f"❌ K-mer validation failed: {len(issues)} issues")
            for issue in issues[:10]:  # Show first 10 issues to avoid spam
                self.logger.error(f"  - {issue}")
            if len(issues) > 10:
                self.logger.error(f"  ... and {len(issues) - 10} more issues")
            self.validation_results['kmer_sequences'] = {'status': 'FAIL', 'issues': issues}
            return False
        else:
            self.logger.info("✅ K-mer validation passed")
            self.logger.info(f"  - Found {len(kmer_3_cols)} valid 3-mer columns")
            self.logger.info(f"  - Found {len(kmer_5_cols)} valid 5-mer columns")
            self.validation_results['kmer_sequences'] = {'status': 'PASS', 'issues': []}
            return True
    
    def validate_completeness(self, df: pl.DataFrame) -> bool:
        """Validate data completeness (no unexpected nulls)"""
        self.logger.info("🔍 Validating data completeness...")
        
        # Core columns that should never be null
        critical_columns = ['gene_id', 'chr', 'position', 'strand']
        
        issues = []
        for col in critical_columns:
            if col in df.columns:
                null_count = df[col].null_count()
                if null_count > 0:
                    issues.append(f"Column {col}: {null_count} null values found")
        
        if issues:
            self.logger.error(f"❌ Completeness validation failed: {len(issues)} issues")
            for issue in issues:
                self.logger.error(f"  - {issue}")
            self.validation_results['completeness'] = {'status': 'FAIL', 'issues': issues}
            return False
        else:
            self.logger.info("✅ Completeness validation passed")
            self.validation_results['completeness'] = {'status': 'PASS', 'issues': []}
            return True
    
    def analyze_statistics(self, df: pl.DataFrame) -> Dict:
        """Generate statistical summary of the dataset"""
        self.logger.info("📊 Analyzing dataset statistics...")
        
        stats = {
            'total_records': len(df),
            'total_features': len(df.columns),
            'memory_usage_mb': df.estimated_size('mb'),
        }
        
        # Gene statistics
        if 'gene_id' in df.columns:
            stats['unique_genes'] = df['gene_id'].n_unique()
        
        # Chromosome distribution
        if 'chr' in df.columns:
            chr_dist = df['chr'].value_counts().sort('chr')
            stats['chromosome_distribution'] = dict(zip(chr_dist['chr'].to_list(), chr_dist['count'].to_list()))
        
        # Splice site statistics
        if 'is_donor' in df.columns and 'is_acceptor' in df.columns:
            stats['donor_sites'] = df['is_donor'].sum()
            stats['acceptor_sites'] = df['is_acceptor'].sum()
            stats['total_splice_sites'] = (df['is_donor'] | df['is_acceptor']).sum()
        
        # Feature type counts
        kmer_3_cols = [col for col in df.columns if col.startswith('3mer_')]
        kmer_5_cols = [col for col in df.columns if col.startswith('5mer_')]
        
        stats['feature_breakdown'] = {
            '3mer_features': len(kmer_3_cols),
            '5mer_features': len(kmer_5_cols),
            'other_features': len(df.columns) - len(kmer_3_cols) - len(kmer_5_cols)
        }
        
        self.logger.info(f"📊 Dataset Statistics:")
        self.logger.info(f"  - Total Records: {stats['total_records']:,}")
        self.logger.info(f"  - Total Features: {stats['total_features']:,}")
        self.logger.info(f"  - Unique Genes: {stats.get('unique_genes', 'N/A'):,}")
        self.logger.info(f"  - Memory Usage: {stats['memory_usage_mb']:.1f} MB")
        
        return stats
    
    def generate_report(self, output_dir: str, stats: Dict) -> str:
        """Generate comprehensive validation report"""
        report_path = os.path.join(output_dir, f"validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md")
        
        with open(report_path, 'w') as f:
            f.write("# train_regulatory_10k_kmers Dataset Validation Report\n\n")
            f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Dataset Path**: {self.dataset_path}\n\n")
            
            # Overall status
            all_passed = all(result['status'] == 'PASS' for result in self.validation_results.values())
            status_emoji = "✅" if all_passed else "❌"
            f.write(f"## Overall Status: {status_emoji} {'PASS' if all_passed else 'FAIL'}\n\n")
            
            # Validation results
            f.write("## Validation Results\n\n")
            for test_name, result in self.validation_results.items():
                status_emoji = "✅" if result['status'] == 'PASS' else "❌"
                f.write(f"### {test_name.replace('_', ' ').title()}: {status_emoji} {result['status']}\n")
                if result['issues']:
                    f.write("**Issues:**\n")
                    for issue in result['issues']:
                        f.write(f"- {issue}\n")
                f.write("\n")
            
            # Statistics
            f.write("## Dataset Statistics\n\n")
            f.write(f"- **Total Records**: {stats['total_records']:,}\n")
            f.write(f"- **Total Features**: {stats['total_features']:,}\n")
            f.write(f"- **Unique Genes**: {stats.get('unique_genes', 'N/A'):,}\n")
            f.write(f"- **Memory Usage**: {stats['memory_usage_mb']:.1f} MB\n\n")
            
            if 'feature_breakdown' in stats:
                f.write("### Feature Breakdown\n")
                for feature_type, count in stats['feature_breakdown'].items():
                    f.write(f"- **{feature_type.replace('_', ' ').title()}**: {count:,}\n")
                f.write("\n")
            
            if 'chromosome_distribution' in stats:
                f.write("### Chromosome Distribution\n")
                for chr_name, count in sorted(stats['chromosome_distribution'].items()):
                    f.write(f"- **Chr {chr_name}**: {count:,} records\n")
                f.write("\n")
        
        return report_path
    
    def run_validation(self, output_dir: str = ".") -> bool:
        """Run complete validation pipeline"""
        self.logger.info("🚀 Starting train_regulatory_10k_kmers dataset validation")
        
        # Step 1: Schema validation
        schema_valid = self.validate_schema()
        
        # Step 2: Load dataset for detailed validation
        try:
            self.logger.info("📂 Loading dataset for detailed validation...")
            df = datasets.load_dataset(self.dataset_path)
            self.logger.info(f"✅ Successfully loaded {len(df):,} records")
        except Exception as e:
            self.logger.error(f"❌ Failed to load dataset: {e}")
            return False
        
        # Step 3: Run detailed validations
        validations = [
            self.validate_data_types(df),
            self.validate_value_ranges(df),
            self.validate_kmer_sequences(df),
            self.validate_completeness(df),
        ]
        
        # Step 4: Generate statistics
        stats = self.analyze_statistics(df)
        
        # Step 5: Generate report
        report_path = self.generate_report(output_dir, stats)
        self.logger.info(f"📄 Validation report saved to: {report_path}")
        
        # Overall result
        all_passed = schema_valid and all(validations)
        
        if all_passed:
            self.logger.info("🎉 All validations passed! Dataset is ready for training.")
        else:
            self.logger.error("⚠️  Some validations failed. Please review the issues above.")
        
        return all_passed


def main():
    parser = argparse.ArgumentParser(description="Validate train_regulatory_10k_kmers dataset")
    parser.add_argument('--dataset-path', default='train_regulatory_10k_kmers/master',
                       help='Path to dataset directory')
    parser.add_argument('--fix', action='store_true',
                       help='Attempt to fix detected issues automatically')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable detailed logging output')
    parser.add_argument('--output-dir', default='.',
                       help='Directory to save validation reports')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run validation
    validator = DatasetValidator(
        dataset_path=args.dataset_path,
        verbose=args.verbose,
        fix_issues=args.fix
    )
    
    success = validator.run_validation(args.output_dir)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
