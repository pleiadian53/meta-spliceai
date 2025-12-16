"""
Utilities for enhancing splice site annotations with gene and transcript features.

This module provides easy access to the splice site enhancement functionality
that's used in the standalone scripts/enhance_splice_sites.py script. It's 
particularly useful when the enhanced splice site data doesn't exist yet or 
needs to be regenerated.
"""

import os
import sys
import polars as pl
import pandas as pd
from pathlib import Path
from typing import Optional, List, Union, Dict, Any

# Import the core functionality from the utils directory
from meta_spliceai.splice_engine.meta_models.utils import (
    enhance_splice_sites_with_features,
    analyze_splicing_patterns
)


def find_project_root() -> str:
    """
    Find the project root directory of splice-surveyor.
    
    Returns
    -------
    str
        Absolute path to the project root directory
    """
    # Start from current directory and move up until we find the project root
    current_dir = Path.cwd()
    while current_dir.name:
        if (current_dir / '.git').exists() or (current_dir / 'setup.py').exists() or current_dir.name == 'splice-surveyor':
            return str(current_dir)
        current_dir = current_dir.parent
        
    # Fallback if we can't detect automatically
    # First try to use system config if available
    try:
        # Dynamic import only when needed to avoid package coupling
        from meta_spliceai.system.config import Config
        return Config.PROJ_DIR
    except (ImportError, AttributeError):
        # If system config is unavailable, fall back to generic path
        home_dir = str(Path.home())
        return os.path.join(home_dir, "work", "splice-surveyor")


def get_default_file_paths(project_dir: Optional[str] = None) -> Dict[str, str]:
    """
    Get default file paths for splice site annotation and feature files.
    
    Parameters
    ----------
    project_dir : Optional[str], optional
        Path to project directory, by default None (auto-detected)
        
    Returns
    -------
    Dict[str, str]
        Dictionary of default file paths for:
        - splice_sites_path: Path to the base splice sites TSV
        - enhanced_output_path: Path for saving enhanced splice sites
        - gtf_path: Path to the GTF annotation file
        - gene_features_path: Path to gene features
        - transcript_features_path: Path to transcript features
    """
    if project_dir is None:
        project_dir = find_project_root()
        
    paths = {
        'splice_sites_path': os.path.join(project_dir, "data", "ensembl", "splice_sites.tsv"),
        'enhanced_output_path': os.path.join(project_dir, "data", "ensembl", "splice_sites_enhanced.tsv"),
        'gtf_path': os.path.join(project_dir, "data", "ensembl", "Homo_sapiens.GRCh38.112.gtf"),
        'gene_features_path': os.path.join(project_dir, "data", "ensembl", "spliceai_analysis", "gene_features.tsv"),
        'transcript_features_path': os.path.join(project_dir, "data", "ensembl", "spliceai_analysis", "transcript_features.tsv")
    }
    
    return paths


def generate_enhanced_splice_sites(
    output_path: Optional[str] = None,
    input_path: Optional[str] = None, 
    gene_types: Optional[List[str]] = None,
    transcript_types: Optional[List[str]] = None,
    analyze_results: bool = False,
    verbose: int = 1
) -> pl.DataFrame:
    """
    Generate enhanced splice site annotations with gene and transcript features.
    
    This function provides a convenient interface to the enhancement functionality,
    automatically finding input files and generating the enhanced output.
    
    Parameters
    ----------
    output_path : Optional[str], optional
        Path to save the enhanced splice sites, by default None (uses default path)
    input_path : Optional[str], optional
        Path to the input splice sites file, by default None (uses default path)
    gene_types : Optional[List[str]], optional
        List of gene types to include, by default None (all types)
    transcript_types : Optional[List[str]], optional
        List of transcript types to include, by default None (all types)
    analyze_results : bool, optional
        Whether to run analysis on the enhanced data, by default False
    verbose : int, optional
        Verbosity level (0=quiet, 1=normal, 2=verbose), by default 1
        
    Returns
    -------
    pl.DataFrame
        Enhanced splice sites DataFrame
        
    Notes
    -----
    This function is useful when load_full_positions_data() fails because
    enhanced splice site data doesn't exist or is outdated.
    
    Examples
    --------
    >>> # Generate enhanced splice sites with all gene types
    >>> enhanced_df = generate_enhanced_splice_sites()
    >>> 
    >>> # Generate for specific gene types
    >>> protein_coding_df = generate_enhanced_splice_sites(
    ...     gene_types=["protein_coding"], 
    ...     analyze_results=True
    ... )
    """
    # Get default paths
    paths = get_default_file_paths()
    
    # Override with provided paths if any
    if input_path:
        paths['splice_sites_path'] = input_path
    if output_path:
        paths['enhanced_output_path'] = output_path
        
    # Check if input file exists
    if not os.path.exists(paths['splice_sites_path']):
        raise FileNotFoundError(f"Splice sites file not found: {paths['splice_sites_path']}")
        
    if verbose > 0:
        print(f"Reading splice sites from: {paths['splice_sites_path']}")
        print(f"Output will be written to: {paths['enhanced_output_path']}")
    
    # Enhance splice sites with gene and transcript information
    enhanced_df = enhance_splice_sites_with_features(
        splice_sites_path=paths['splice_sites_path'],
        gene_features_path=paths['gene_features_path'],
        transcript_features_path=paths['transcript_features_path'],
        gene_types_to_keep=gene_types,
        transcript_types_to_keep=transcript_types,
        verbose=verbose
    )
    
    # Save enhanced dataframe
    if paths['enhanced_output_path']:
        output_dir = os.path.dirname(paths['enhanced_output_path'])
        os.makedirs(output_dir, exist_ok=True)
        enhanced_df.write_csv(paths['enhanced_output_path'], separator='\t')
        if verbose > 0:
            print(f"Enhanced splice sites saved to: {paths['enhanced_output_path']}")
            
    if verbose > 0:
        feature_cols = [col for col in enhanced_df.columns 
                      if col not in ['chrom', 'start', 'end', 'strand', 'site_type', 'gene_id', 'transcript_id']]
        print(f"Added feature columns: {feature_cols}")
    
    # Analyze splicing patterns if requested
    if analyze_results:
        gene_type_str = ", ".join(gene_types) if gene_types else "All Types"
        analyze_splicing_patterns(enhanced_df, gene_types=gene_types, title=f"Splicing Analysis - {gene_type_str}")
        
    return enhanced_df


def verify_splice_sites_enhancement():
    """
    Verify that splice site enhancement is working correctly.
    
    This function tests the enhancement pipeline with a small subset of data
    and prints diagnostics about the enhanced features.
    
    Returns
    -------
    bool
        True if verification succeeds, False otherwise
    """
    try:
        # Get default paths
        paths = get_default_file_paths()
        
        # Check if files exist
        files_exist = all(os.path.exists(path) for path in [
            paths['splice_sites_path'],
            paths['gene_features_path'],
            paths['transcript_features_path']
        ])
        
        if not files_exist:
            print("Verification failed: Required input files don't exist")
            return False
            
        print("Testing splice site enhancement...")
        
        # Load just a small sample for verification
        splice_sites = pl.read_csv(paths['splice_sites_path'], separator='\t').limit(100)
        
        # Save to a temporary file
        temp_path = os.path.join(os.path.dirname(paths['splice_sites_path']), "temp_splice_sites.tsv")
        splice_sites.write_csv(temp_path, separator='\t')
        
        # Run enhancement on the sample
        enhanced_df = generate_enhanced_splice_sites(
            input_path=temp_path,
            output_path=None,  # Don't save the output
            verbose=1
        )
        
        # Cleanup
        os.remove(temp_path)
        
        # Verify enhancement
        original_cols = set(splice_sites.columns)
        enhanced_cols = set(enhanced_df.columns)
        new_cols = enhanced_cols - original_cols
        
        print(f"Original columns: {len(original_cols)}")
        print(f"Enhanced columns: {len(enhanced_cols)}")
        print(f"New feature columns added: {len(new_cols)}")
        print(f"New columns: {new_cols}")
        
        if len(new_cols) > 0:
            print("Verification successful: Enhancement added new feature columns")
            return True
        else:
            print("Verification failed: No new columns were added")
            return False
            
    except Exception as e:
        print(f"Verification failed with error: {e}")
        return False


if __name__ == "__main__":
    """
    When run as a script, this module verifies the enhancement functionality.
    """
    verify_splice_sites_enhancement()
