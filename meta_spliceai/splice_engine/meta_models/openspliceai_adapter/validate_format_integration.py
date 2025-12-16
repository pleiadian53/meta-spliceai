#!/usr/bin/env python3
"""
Format Integration Validation Script

This script validates that your MetaSpliceAI splice site annotation format
is properly handled by the OpenSpliceAI adapter, ensuring seamless integration.

Your format example:
chrom	start	end	position	strand	site_type	gene_id	transcript_id
1	2581649	2581653	2581651	+	donor	ENSG00000228037	ENST00000424215
1	2583367	2583371	2583369	+	acceptor	ENSG00000228037	ENST00000424215
"""

import os
import sys
import pandas as pd
import polars as pl
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging

# Add the openspliceai_adapter to path
sys.path.insert(0, str(Path(__file__).parent))

from format_compatibility import SpliceFormatConverter, ensure_format_compatibility

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MetaSpliceAIFormatValidator:
    """
    Validates your specific MetaSpliceAI format and ensures OpenSpliceAI compatibility.
    """
    
    def __init__(self, verbose: int = 1):
        self.verbose = verbose
        self.converter = SpliceFormatConverter(verbose=verbose)
        
        # Your exact format specification
        self.expected_format = {
            'columns': ['chrom', 'start', 'end', 'position', 'strand', 'site_type', 'gene_id', 'transcript_id'],
            'column_types': {
                'chrom': 'string',  # chromosome (1, 2, ..., X, Y)
                'start': 'int64',   # start coordinate (1-based)
                'end': 'int64',     # end coordinate (1-based)
                'position': 'int64', # splice site position (1-based)
                'strand': 'string',  # '+' or '-'
                'site_type': 'string', # 'donor' or 'acceptor'
                'gene_id': 'string',   # ENSG00000228037
                'transcript_id': 'string' # ENST00000424215
            },
            'site_type_values': ['donor', 'acceptor'],
            'strand_values': ['+', '-'],
            'file_format': 'tsv',
            'separator': '\t'
        }
    
    def create_test_data(self) -> pd.DataFrame:
        """Create test data matching your exact format."""
        
        test_data = pd.DataFrame([
            {
                'chrom': '1',
                'start': 2581649,
                'end': 2581653,
                'position': 2581651,
                'strand': '+',
                'site_type': 'donor',
                'gene_id': 'ENSG00000228037',
                'transcript_id': 'ENST00000424215'
            },
            {
                'chrom': '1',
                'start': 2583367,
                'end': 2583371,
                'position': 2583369,
                'strand': '+',
                'site_type': 'acceptor',
                'gene_id': 'ENSG00000228037',
                'transcript_id': 'ENST00000424215'
            },
            {
                'chrom': '1',
                'start': 2583494,
                'end': 2583498,
                'position': 2583496,
                'strand': '+',
                'site_type': 'donor',
                'gene_id': 'ENSG00000228037',
                'transcript_id': 'ENST00000424215'
            },
            {
                'chrom': '1',
                'start': 2584122,
                'end': 2584126,
                'position': 2584124,
                'strand': '+',
                'site_type': 'acceptor',
                'gene_id': 'ENSG00000228037',
                'transcript_id': 'ENST00000424215'
            },
            # Add some negative strand examples
            {
                'chrom': '2',
                'start': 1000000,
                'end': 1000004,
                'position': 1000002,
                'strand': '-',
                'site_type': 'donor',
                'gene_id': 'ENSG00000123456',
                'transcript_id': 'ENST00000654321'
            },
            {
                'chrom': '2',
                'start': 1001000,
                'end': 1001004,
                'position': 1001002,
                'strand': '-',
                'site_type': 'acceptor',
                'gene_id': 'ENSG00000123456',
                'transcript_id': 'ENST00000654321'
            }
        ])
        
        return test_data
    
    def validate_format_structure(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate that DataFrame matches your expected format structure."""
        
        validation = {
            'success': True,
            'issues': [],
            'warnings': [],
            'format_compliance': True
        }
        
        # Check columns
        expected_cols = set(self.expected_format['columns'])
        actual_cols = set(df.columns)
        
        missing_cols = expected_cols - actual_cols
        extra_cols = actual_cols - expected_cols
        
        if missing_cols:
            validation['success'] = False
            validation['issues'].append(f"Missing required columns: {missing_cols}")
        
        if extra_cols:
            validation['warnings'].append(f"Extra columns found: {extra_cols}")
        
        # Check data types and values
        for col, expected_type in self.expected_format['column_types'].items():
            if col in df.columns:
                # Check site_type values
                if col == 'site_type':
                    unique_values = set(df[col].unique())
                    expected_values = set(self.expected_format['site_type_values'])
                    invalid_values = unique_values - expected_values
                    
                    if invalid_values:
                        validation['issues'].append(f"Invalid site_type values: {invalid_values}")
                        validation['success'] = False
                
                # Check strand values
                elif col == 'strand':
                    unique_values = set(df[col].unique())
                    expected_values = set(self.expected_format['strand_values'])
                    invalid_values = unique_values - expected_values
                    
                    if invalid_values:
                        validation['issues'].append(f"Invalid strand values: {invalid_values}")
                        validation['success'] = False
                
                # Check numeric columns
                elif col in ['start', 'end', 'position']:
                    if not pd.api.types.is_numeric_dtype(df[col]):
                        validation['issues'].append(f"Column '{col}' should be numeric")
                        validation['success'] = False
                    elif df[col].isna().any():
                        validation['warnings'].append(f"Column '{col}' contains NaN values")
        
        # Check coordinate consistency
        if all(col in df.columns for col in ['start', 'end', 'position']):
            # Position should be within start-end range for most cases
            position_in_range = ((df['position'] >= df['start']) & (df['position'] <= df['end']))
            out_of_range_count = (~position_in_range).sum()
            
            if out_of_range_count > 0:
                validation['warnings'].append(f"{out_of_range_count} positions outside start-end range")
        
        return validation
    
    def test_openspliceai_conversion(self, df: pd.DataFrame, output_dir: str = "test_conversion") -> Dict[str, Any]:
        """Test conversion to OpenSpliceAI format."""
        
        if self.verbose >= 1:
            print(f"[validator] Testing OpenSpliceAI format conversion...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            # Test the conversion
            conversion_results = self.converter.create_openspliceai_workflow_compatible_output(
                splice_sites_df=df,
                output_dir=output_dir,
                preserve_original_format=True
            )
            
            # Validate the conversion
            if 'splice_sites_file' in conversion_results:
                original_file = conversion_results['splice_sites_file']
                converted_file = conversion_results.get('openspliceai_tsv')
                
                if converted_file and os.path.exists(converted_file):
                    verification = self.converter.verify_format_compatibility(
                        original_file=original_file,
                        converted_file=converted_file
                    )
                    
                    conversion_results['verification'] = verification
                    conversion_results['conversion_success'] = verification['success']
                else:
                    conversion_results['conversion_success'] = False
                    conversion_results['error'] = "Converted file not created"
            
            return conversion_results
            
        except Exception as e:
            return {
                'conversion_success': False,
                'error': str(e),
                'traceback': str(e.__traceback__)
            }
    
    def run_comprehensive_validation(
        self,
        splice_sites_file: Optional[str] = None,
        output_dir: str = "validation_output"
    ) -> Dict[str, Any]:
        """Run comprehensive validation of format integration."""
        
        print("="*60)
        print("MetaSpliceAI ↔ OpenSpliceAI Format Integration Validation")
        print("="*60)
        
        results = {
            'validation_success': True,
            'tests_passed': 0,
            'tests_total': 0,
            'issues': [],
            'warnings': []
        }
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Test 1: Load your actual data or use test data
        print("\n1. Loading and validating data format...")
        results['tests_total'] += 1
        
        if splice_sites_file and os.path.exists(splice_sites_file):
            print(f"   Loading actual data: {splice_sites_file}")
            df = pd.read_csv(splice_sites_file, sep='\t')
            results['data_source'] = 'actual_file'
            results['data_file'] = splice_sites_file
        else:
            print("   Using test data (matching your format)")
            df = self.create_test_data()
            results['data_source'] = 'test_data'
            
            # Save test data for reference
            test_file = os.path.join(output_dir, "test_splice_sites.tsv")
            df.to_csv(test_file, sep='\t', index=False)
            results['test_data_file'] = test_file
        
        print(f"   Data loaded: {len(df)} rows, {len(df.columns)} columns")
        results['data_rows'] = len(df)
        results['data_columns'] = list(df.columns)
        results['tests_passed'] += 1
        
        # Test 2: Validate format structure
        print("\n2. Validating format structure...")
        results['tests_total'] += 1
        
        format_validation = self.validate_format_structure(df)
        results['format_validation'] = format_validation
        
        if format_validation['success']:
            print("   ✓ Format structure validation passed")
            results['tests_passed'] += 1
        else:
            print("   ✗ Format structure validation failed")
            results['validation_success'] = False
            results['issues'].extend(format_validation['issues'])
        
        if format_validation['warnings']:
            print(f"   ⚠ Warnings: {format_validation['warnings']}")
            results['warnings'].extend(format_validation['warnings'])
        
        # Test 3: Test OpenSpliceAI conversion
        print("\n3. Testing OpenSpliceAI format conversion...")
        results['tests_total'] += 1
        
        conversion_results = self.test_openspliceai_conversion(df, output_dir)
        results['conversion_results'] = conversion_results
        
        if conversion_results.get('conversion_success', False):
            print("   ✓ OpenSpliceAI conversion successful")
            results['tests_passed'] += 1
        else:
            print("   ✗ OpenSpliceAI conversion failed")
            results['validation_success'] = False
            error_msg = conversion_results.get('error', 'Unknown conversion error')
            results['issues'].append(f"Conversion error: {error_msg}")
        
        # Test 4: Validate data integrity
        print("\n4. Validating data integrity...")
        results['tests_total'] += 1
        
        if 'verification' in conversion_results:
            verification = conversion_results['verification']
            if verification['success'] and verification['data_integrity']:
                print("   ✓ Data integrity preserved")
                results['tests_passed'] += 1
            else:
                print("   ✗ Data integrity issues detected")
                results['validation_success'] = False
                results['issues'].extend(verification.get('issues', []))
        else:
            print("   ⚠ Data integrity verification skipped (conversion failed)")
        
        # Summary
        print("\n" + "="*60)
        print("VALIDATION SUMMARY")
        print("="*60)
        print(f"Tests passed: {results['tests_passed']}/{results['tests_total']}")
        print(f"Overall success: {'✓' if results['validation_success'] else '✗'}")
        
        if results['issues']:
            print(f"\nIssues found ({len(results['issues'])}):")
            for i, issue in enumerate(results['issues'], 1):
                print(f"  {i}. {issue}")
        
        if results['warnings']:
            print(f"\nWarnings ({len(results['warnings'])}):")
            for i, warning in enumerate(results['warnings'], 1):
                print(f"  {i}. {warning}")
        
        print(f"\nOutput directory: {output_dir}")
        
        # Save results
        results_file = os.path.join(output_dir, "validation_results.json")
        import json
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"Detailed results saved: {results_file}")
        
        return results


def main():
    """Main validation script."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate MetaSpliceAI ↔ OpenSpliceAI format integration")
    parser.add_argument(
        '--splice-sites-file',
        type=str,
        help='Path to your splice_sites.tsv file (optional, will use test data if not provided)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='validation_output',
        help='Output directory for validation results'
    )
    parser.add_argument(
        '--verbose',
        type=int,
        default=1,
        help='Verbosity level (0=quiet, 1=normal, 2=verbose)'
    )
    
    args = parser.parse_args()
    
    # Run validation
    validator = MetaSpliceAIFormatValidator(verbose=args.verbose)
    results = validator.run_comprehensive_validation(
        splice_sites_file=args.splice_sites_file,
        output_dir=args.output_dir
    )
    
    # Exit with appropriate code
    exit_code = 0 if results['validation_success'] else 1
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
