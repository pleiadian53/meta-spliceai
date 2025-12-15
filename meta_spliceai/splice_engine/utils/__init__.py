"""Splice Engine Utilities Package
==============================
Utility modules for the splice engine.
"""

import sys
from pathlib import Path

# Import from output_enhancement module
from .output_enhancement import create_output_enhancer, OutputEnhancer

# Import from the main utils.py file (in parent directory)
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

try:
    from utils import (
        setup_logger,
        calculate_duration,
        format_time,
        print_emphasized,
        pandas_to_polars,
        polars_to_pandas,
        convert_csv_to_tsv,
        calculate_and_format_duration,
        format_duration_as_string_or_tuple,
        extract_gene_features_from_gtf_by_awk_command,
    )
except ImportError:
    # Fallback - define stub functions to avoid import errors
    def setup_logger(log_file_path):
        import logging
        return logging.getLogger('fallback_logger')
    
    def calculate_duration(start_time):
        import time
        return time.time() - start_time
    
    def format_time(duration, return_tuple=False):
        if return_tuple:
            return (f"{duration:.2f}s", duration)
        return f"{duration:.2f}s"
    
    def print_emphasized(text, style='bold', edge_effect=True, symbol='='):
        print(f"{symbol * 50}")
        print(text)
        print(f"{symbol * 50}")
    
    def pandas_to_polars(pandas_df):
        import polars as pl
        return pl.from_pandas(pandas_df)
    
    def polars_to_pandas(polars_df):
        return polars_df.to_pandas()
    
    def convert_csv_to_tsv(input_csv_path, output_tsv_path=None, schema_overrides=None):
        raise NotImplementedError("Function not available in fallback mode")
    
    def calculate_and_format_duration(start_time):
        duration = calculate_duration(start_time)
        return format_time(duration)
    
    def format_duration_as_string_or_tuple(duration, return_tuple=False):
        return format_time(duration, return_tuple)
    
    def extract_gene_features_from_gtf_by_awk_command(gtf_file, output_file):
        raise NotImplementedError("Function not available in fallback mode")

__all__ = [
    'create_output_enhancer',
    'OutputEnhancer',
    'setup_logger',
    'calculate_duration', 
    'format_time',
    'print_emphasized',
    'pandas_to_polars',
    'polars_to_pandas',
    'convert_csv_to_tsv',
    'calculate_and_format_duration',
    'format_duration_as_string_or_tuple',
    'extract_gene_features_from_gtf_by_awk_command',
]