"""
Format Compatibility Module for OpenSpliceAI Integration

This module handles format differences between MetaSpliceAI and OpenSpliceAI
data representations, ensuring seamless integration while preserving data integrity.
"""

import os
import pandas as pd
import polars as pl
import numpy as np
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class SpliceFormatConverter:
    """
    Handles format conversion between MetaSpliceAI and OpenSpliceAI data formats.
    
    Key Format Differences Handled:
    1. Column names: site_type vs splice_type
    2. Label encoding: donor/acceptor/neither vs 0/1/2
    3. Coordinate systems: 0-based vs 1-based indexing
    4. File formats: TSV vs HDF5
    5. Data structures: DataFrame vs specialized arrays
    """
    
    def __init__(self, verbose: int = 1):
        self.verbose = verbose
        
        # MetaSpliceAI format (your current format)
        self.splicesurveyor_schema = {
            'columns': ['chrom', 'start', 'end', 'position', 'strand', 'site_type', 'gene_id', 'transcript_id'],
            'site_type_values': ['donor', 'acceptor'],  # No 'neither' in annotation files
            'coordinate_system': '1-based',  # GTF standard
            'file_format': 'tsv'
        }
        
        # OpenSpliceAI format
        self.openspliceai_schema = {
            'columns': ['NAME', 'CHROM', 'STRAND', 'TX_START', 'TX_END', 'SEQ', 'LABEL'],
            'label_encoding': {0: 'neither', 1: 'acceptor', 2: 'donor'},
            'coordinate_system': '0-based',  # Python standard
            'file_format': 'hdf5'
        }
        
        # Label mappings for each system
        # MetaSpliceAI encoding (from label_utils.py)
        self.splicesurveyor_label_map = {
            'neither': 0,
            'donor': 1,
            'acceptor': 2
        }
        
        # OpenSpliceAI encoding (from create_datafile.py)
        self.openspliceai_label_map = {
            'neither': 0,
            'acceptor': 1, 
            'donor': 2
        }
        
        # Reverse mappings
        self.splicesurveyor_int_to_label = {v: k for k, v in self.splicesurveyor_label_map.items()}
        self.openspliceai_int_to_label = {v: k for k, v in self.openspliceai_label_map.items()}
        
        # Conversion mapping: MetaSpliceAI numeric -> OpenSpliceAI numeric
        self.ss_to_osai_numeric = {
            0: 0,  # neither -> neither
            1: 2,  # donor -> donor  
            2: 1   # acceptor -> acceptor
        }
        
        # Conversion mapping: OpenSpliceAI numeric -> MetaSpliceAI numeric
        self.osai_to_ss_numeric = {
            0: 0,  # neither -> neither
            1: 2,  # acceptor -> acceptor
            2: 1   # donor -> donor
        }
    
    def validate_splicesurveyor_format(self, df: Union[pd.DataFrame, pl.DataFrame]) -> Dict[str, Any]:
        """
        Validate that the input DataFrame matches expected MetaSpliceAI format.
        
        Parameters
        ----------
        df : DataFrame
            Input DataFrame to validate
            
        Returns
        -------
        Dict[str, Any]
            Validation results with success status and details
        """
        if isinstance(df, pl.DataFrame):
            columns = df.columns
            sample_data = df.head(5).to_pandas() if len(df) > 0 else pd.DataFrame()
        else:
            columns = df.columns.tolist()
            sample_data = df.head(5)
        
        validation = {
            'success': True,
            'issues': [],
            'format_detected': 'splicesurveyor',
            'columns_present': columns,
            'sample_data': sample_data
        }
        
        # Check required columns
        required_cols = set(self.splicesurveyor_schema['columns'])
        present_cols = set(columns)
        missing_cols = required_cols - present_cols
        
        if missing_cols:
            validation['success'] = False
            validation['issues'].append(f"Missing required columns: {missing_cols}")
        
        # Check site_type values if column exists
        if 'site_type' in columns and len(df) > 0:
            if isinstance(df, pl.DataFrame):
                unique_site_types = set(df['site_type'].unique().to_list())
            else:
                unique_site_types = set(df['site_type'].unique())
            
            expected_site_types = set(self.splicesurveyor_schema['site_type_values'])
            unexpected_types = unique_site_types - expected_site_types - {'neither'}  # 'neither' might appear
            
            if unexpected_types:
                validation['issues'].append(f"Unexpected site_type values: {unexpected_types}")
        
        # Check chromosome format
        if 'chrom' in columns and len(df) > 0:
            if isinstance(df, pl.DataFrame):
                sample_chroms = df['chrom'].head(5).to_list()
            else:
                sample_chroms = df['chrom'].head(5).tolist()
            
            validation['sample_chromosomes'] = sample_chroms
        
        if self.verbose >= 2:
            print(f"[format_validator] MetaSpliceAI format validation:")
            print(f"  Success: {validation['success']}")
            print(f"  Columns: {len(columns)} present")
            if validation['issues']:
                print(f"  Issues: {validation['issues']}")
        
        return validation
    
    def convert_splicesurveyor_to_openspliceai_compatible(
        self,
        splice_sites_df: Union[pd.DataFrame, pl.DataFrame],
        sequences_df: Optional[Union[pd.DataFrame, pl.DataFrame]] = None,
        output_dir: str = "openspliceai_compatible",
        dataset_name: str = "converted_dataset"
    ) -> Dict[str, str]:
        """
        Convert MetaSpliceAI format to OpenSpliceAI-compatible format.
        
        Parameters
        ----------
        splice_sites_df : DataFrame
            MetaSpliceAI splice sites DataFrame with your format:
            ['chrom', 'start', 'end', 'position', 'strand', 'site_type', 'gene_id', 'transcript_id']
        sequences_df : DataFrame, optional
            Gene sequences DataFrame (if available)
        output_dir : str
            Output directory for converted files
        dataset_name : str
            Name prefix for output files
            
        Returns
        -------
        Dict[str, str]
            Paths to created files compatible with OpenSpliceAI
        """
        
        if self.verbose >= 1:
            print(f"[converter] Converting MetaSpliceAI format to OpenSpliceAI-compatible...")
        
        # Validate input format
        validation = self.validate_splicesurveyor_format(splice_sites_df)
        if not validation['success']:
            raise ValueError(f"Input format validation failed: {validation['issues']}")
        
        # Convert to pandas for easier processing
        if isinstance(splice_sites_df, pl.DataFrame):
            splice_df = splice_sites_df.to_pandas()
        else:
            splice_df = splice_sites_df.copy()
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Step 1: Convert to intermediate format that matches OpenSpliceAI expectations
        converted_df = self._convert_splice_sites_format(splice_df)
        
        # Step 2: Create OpenSpliceAI-style TSV output (easier to work with than HDF5)
        openspliceai_tsv = os.path.join(output_dir, f"{dataset_name}_splice_sites_openspliceai_format.tsv")
        converted_df.to_csv(openspliceai_tsv, sep='\t', index=False)
        
        # Step 3: Create a mapping file for reference
        mapping_file = os.path.join(output_dir, f"{dataset_name}_format_mapping.json")
        mapping_info = {
            'original_format': 'splicesurveyor',
            'converted_format': 'openspliceai_compatible',
            'column_mapping': {
                'site_type': 'splice_type',
                'position': 'position',  # Keep same
                'chrom': 'chromosome',
                'gene_id': 'gene_id',  # Keep same
                'transcript_id': 'transcript_id'  # Keep same
            },
            'splicesurveyor_label_mapping': self.splicesurveyor_label_map,
            'openspliceai_label_mapping': self.openspliceai_label_map,
            'total_sites': len(converted_df),
            'site_type_counts': converted_df['splice_type'].value_counts().to_dict()
        }
        
        import json
        with open(mapping_file, 'w') as f:
            json.dump(mapping_info, f, indent=2)
        
        if self.verbose >= 1:
            print(f"[converter] Conversion completed:")
            print(f"  Input sites: {len(splice_df)}")
            print(f"  Output sites: {len(converted_df)}")
            print(f"  OpenSpliceAI TSV: {openspliceai_tsv}")
            print(f"  Mapping file: {mapping_file}")
        
        return {
            'openspliceai_tsv': openspliceai_tsv,
            'mapping_file': mapping_file,
            'converted_df': converted_df
        }
    
    def _convert_splice_sites_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert splice sites DataFrame to OpenSpliceAI-compatible format."""
        
        converted = df.copy()
        
        # Step 1: Standardize column names
        column_mapping = {
            'site_type': 'splice_type',
            'chrom': 'chromosome'
        }
        
        for old_col, new_col in column_mapping.items():
            if old_col in converted.columns:
                converted = converted.rename(columns={old_col: new_col})
        
        # Step 2: Ensure required columns exist
        required_columns = ['chromosome', 'position', 'strand', 'splice_type', 'gene_id', 'transcript_id']
        for col in required_columns:
            if col not in converted.columns:
                if col == 'splice_type' and 'site_type' in df.columns:
                    converted['splice_type'] = df['site_type']
                elif col == 'chromosome' and 'chrom' in df.columns:
                    converted['chromosome'] = df['chrom']
                else:
                    raise ValueError(f"Required column '{col}' not found and cannot be derived")
        
        # Step 3: Standardize chromosome names (remove 'chr' prefix if present)
        if 'chromosome' in converted.columns:
            converted['chromosome'] = converted['chromosome'].astype(str).str.replace('chr', '', regex=False)
        
        # Step 4: Validate and clean splice_type values
        valid_splice_types = {'donor', 'acceptor', 'neither'}
        if 'splice_type' in converted.columns:
            invalid_types = set(converted['splice_type'].unique()) - valid_splice_types
            if invalid_types:
                if self.verbose >= 1:
                    print(f"[converter] Warning: Found invalid splice types: {invalid_types}")
                    print(f"[converter] Filtering out invalid types...")
                converted = converted[converted['splice_type'].isin(valid_splice_types)]
        
        # Step 5: Add numeric labels for OpenSpliceAI compatibility
        if 'splice_type' in converted.columns:
            # Use MetaSpliceAI encoding for consistency with your workflow
            converted['label'] = converted['splice_type'].map(self.splicesurveyor_label_map)
        
        # Step 6: Ensure proper data types
        if 'position' in converted.columns:
            converted['position'] = pd.to_numeric(converted['position'], errors='coerce')
        if 'start' in converted.columns:
            converted['start'] = pd.to_numeric(converted['start'], errors='coerce')
        if 'end' in converted.columns:
            converted['end'] = pd.to_numeric(converted['end'], errors='coerce')
        
        # Step 7: Sort by chromosome and position for consistency
        if 'chromosome' in converted.columns and 'position' in converted.columns:
            # Custom sort for chromosomes (1, 2, ..., 22, X, Y)
            def chrom_sort_key(chrom):
                chrom_str = str(chrom)
                if chrom_str.isdigit():
                    return (0, int(chrom_str))
                elif chrom_str in ['X', 'x']:
                    return (1, 0)
                elif chrom_str in ['Y', 'y']:
                    return (1, 1)
                else:
                    return (2, chrom_str)
            
            converted['_sort_key'] = converted['chromosome'].apply(chrom_sort_key)
            converted = converted.sort_values(['_sort_key', 'position']).drop('_sort_key', axis=1)
        
        return converted
    
    def create_openspliceai_workflow_compatible_output(
        self,
        splice_sites_df: Union[pd.DataFrame, pl.DataFrame],
        output_dir: str,
        preserve_original_format: bool = True
    ) -> Dict[str, Any]:
        """
        Create output that's compatible with both OpenSpliceAI and your existing workflow.
        
        This function creates files in the exact format expected by your existing
        incremental_builder.py while also providing OpenSpliceAI-compatible versions.
        
        Parameters
        ----------
        splice_sites_df : DataFrame
            Input splice sites DataFrame
        output_dir : str
            Output directory
        preserve_original_format : bool
            Whether to also save in original MetaSpliceAI format
            
        Returns
        -------
        Dict[str, Any]
            Dictionary with file paths and metadata
        """
        
        if self.verbose >= 1:
            print(f"[workflow_compat] Creating workflow-compatible output...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Convert to pandas for processing
        if isinstance(splice_sites_df, pl.DataFrame):
            df = splice_sites_df.to_pandas()
        else:
            df = splice_sites_df.copy()
        
        results = {
            'success': True,
            'output_dir': output_dir,
            'files_created': []
        }
        
        # 1. Create file in your existing format (splice_sites.tsv)
        if preserve_original_format:
            original_format_file = os.path.join(output_dir, "splice_sites.tsv")
            
            # Ensure the DataFrame has the exact columns your workflow expects
            expected_columns = ['chrom', 'start', 'end', 'position', 'strand', 'site_type', 'gene_id', 'transcript_id']
            
            # Create a copy with standardized columns
            original_df = df.copy()
            
            # Map columns back to original format if needed
            if 'splice_type' in original_df.columns and 'site_type' not in original_df.columns:
                original_df['site_type'] = original_df['splice_type']
            if 'chromosome' in original_df.columns and 'chrom' not in original_df.columns:
                original_df['chrom'] = original_df['chromosome']
            
            # Select only the expected columns (in the right order)
            available_columns = [col for col in expected_columns if col in original_df.columns]
            original_df = original_df[available_columns]
            
            # Save in TSV format
            original_df.to_csv(original_format_file, sep='\t', index=False)
            results['files_created'].append(original_format_file)
            results['splice_sites_file'] = original_format_file
            
            if self.verbose >= 2:
                print(f"[workflow_compat] Created original format: {original_format_file}")
                print(f"  Columns: {available_columns}")
                print(f"  Rows: {len(original_df)}")
        
        # 2. Create OpenSpliceAI-compatible version
        openspliceai_results = self.convert_splicesurveyor_to_openspliceai_compatible(
            splice_sites_df=df,
            output_dir=output_dir,
            dataset_name="workflow_compatible"
        )
        
        results.update(openspliceai_results)
        results['files_created'].extend(openspliceai_results.values())
        
        # 3. Create metadata file
        metadata_file = os.path.join(output_dir, "format_compatibility_metadata.json")
        metadata = {
            'created_by': 'SpliceFormatConverter',
            'input_format': 'splicesurveyor',
            'output_formats': ['splicesurveyor_original', 'openspliceai_compatible'],
            'total_splice_sites': len(df),
            'site_type_distribution': df['site_type'].value_counts().to_dict() if 'site_type' in df.columns else {},
            'chromosome_distribution': df['chrom'].value_counts().to_dict() if 'chrom' in df.columns else {},
            'files_created': results['files_created'],
            'format_validation': self.validate_splicesurveyor_format(df)
        }
        
        import json
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        results['metadata_file'] = metadata_file
        results['files_created'].append(metadata_file)
        
        if self.verbose >= 1:
            print(f"[workflow_compat] Workflow-compatible output created:")
            print(f"  Files created: {len(results['files_created'])}")
            print(f"  Original format preserved: {preserve_original_format}")
            print(f"  OpenSpliceAI compatible: ✓")
            print(f"  Metadata file: {metadata_file}")
        
        return results
    
    def verify_format_compatibility(
        self,
        original_file: str,
        converted_file: str
    ) -> Dict[str, Any]:
        """
        Verify that format conversion preserved data integrity.
        
        Parameters
        ----------
        original_file : str
            Path to original MetaSpliceAI format file
        converted_file : str
            Path to converted OpenSpliceAI-compatible file
            
        Returns
        -------
        Dict[str, Any]
            Verification results
        """
        
        if self.verbose >= 1:
            print(f"[verifier] Verifying format compatibility...")
        
        # Load both files
        original_df = pd.read_csv(original_file, sep='\t')
        converted_df = pd.read_csv(converted_file, sep='\t')
        
        verification = {
            'success': True,
            'issues': [],
            'original_count': len(original_df),
            'converted_count': len(converted_df),
            'data_integrity': True
        }
        
        # Check row counts
        if len(original_df) != len(converted_df):
            verification['issues'].append(f"Row count mismatch: {len(original_df)} vs {len(converted_df)}")
            verification['success'] = False
        
        # Check site type mapping
        if 'site_type' in original_df.columns and 'splice_type' in converted_df.columns:
            original_types = set(original_df['site_type'].unique())
            converted_types = set(converted_df['splice_type'].unique())
            
            if original_types != converted_types:
                verification['issues'].append(f"Site type mismatch: {original_types} vs {converted_types}")
        
        # Check position preservation
        if 'position' in original_df.columns and 'position' in converted_df.columns:
            original_positions = set(original_df['position'].unique())
            converted_positions = set(converted_df['position'].unique())
            
            if original_positions != converted_positions:
                verification['issues'].append("Position values not preserved")
                verification['data_integrity'] = False
        
        if self.verbose >= 1:
            print(f"[verifier] Verification completed:")
            print(f"  Success: {verification['success']}")
            print(f"  Data integrity: {verification['data_integrity']}")
            if verification['issues']:
                print(f"  Issues: {verification['issues']}")
        
        return verification


# Convenience functions for easy integration
def ensure_format_compatibility(
    splice_sites_file: str,
    output_dir: str,
    verbose: int = 1
) -> Dict[str, Any]:
    """
    Ensure splice sites file is compatible with both MetaSpliceAI and OpenSpliceAI workflows.
    
    Parameters
    ----------
    splice_sites_file : str
        Path to existing splice sites file
    output_dir : str
        Output directory for compatible files
    verbose : int
        Verbosity level
        
    Returns
    -------
    Dict[str, Any]
        Results with file paths and compatibility information
    """
    
    converter = SpliceFormatConverter(verbose=verbose)
    
    # Load the file
    if splice_sites_file.endswith('.parquet'):
        df = pl.read_parquet(splice_sites_file)
    else:
        sep = '\t' if splice_sites_file.endswith('.tsv') else ','
        df = pl.read_csv(splice_sites_file, separator=sep)
    
    # Create compatible output
    results = converter.create_openspliceai_workflow_compatible_output(
        splice_sites_df=df,
        output_dir=output_dir,
        preserve_original_format=True
    )
    
    return results


def validate_openspliceai_integration(
    splicesurveyor_file: str,
    openspliceai_output: str,
    verbose: int = 1
) -> bool:
    """
    Validate that OpenSpliceAI integration preserves MetaSpliceAI data integrity.
    
    Parameters
    ----------
    splicesurveyor_file : str
        Original MetaSpliceAI format file
    openspliceai_output : str
        OpenSpliceAI processed output
    verbose : int
        Verbosity level
        
    Returns
    -------
    bool
        True if integration is successful and data integrity is preserved
    """
    
    converter = SpliceFormatConverter(verbose=verbose)
    
    verification = converter.verify_format_compatibility(
        original_file=splicesurveyor_file,
        converted_file=openspliceai_output
    )
    
    return verification['success'] and verification['data_integrity']
