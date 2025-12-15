"""Schema standardization for genomic datasets.

This module provides utilities for standardizing column names and schemas
across different genomic datasets to ensure consistency throughout the system.

Key Functions:
- standardize_splice_sites_schema: Standardize splice site annotation column names
- standardize_gene_features_schema: Standardize gene feature column names
- standardize_transcript_features_schema: Standardize transcript feature column names
- get_standard_column_mapping: Get the standard column name mapping

Design Principles:
1. **Non-destructive**: Original columns are renamed, not replaced
2. **Idempotent**: Can be called multiple times safely
3. **Flexible**: Handles both Polars and Pandas DataFrames
4. **Documented**: Clear mapping between synonymous column names

Documentation:
See docs/SCHEMA_STANDARDIZATION.md for comprehensive usage guide.
"""

from typing import Dict, Union, Optional
import polars as pl
import pandas as pd


# ============================================================================
# Standard Column Mappings
# ============================================================================

# Splice site column synonyms
SPLICE_SITE_COLUMN_MAPPING = {
    'site_type': 'splice_type',  # GTF convention → biological terminology
    'type': 'splice_type',        # Generic → specific
}

# Gene feature column synonyms
GENE_FEATURE_COLUMN_MAPPING = {
    'seqname': 'chrom',          # GTF convention → genomics convention
    'gene_type': 'gene_biotype',  # Alternative naming
    'biotype': 'gene_biotype',    # Short form → full form
}

# Transcript feature column synonyms
TRANSCRIPT_FEATURE_COLUMN_MAPPING = {
    'seqname': 'chrom',
    'transcript_type': 'transcript_biotype',
    'biotype': 'transcript_biotype',
}

# Exon feature column synonyms
EXON_FEATURE_COLUMN_MAPPING = {
    'seqname': 'chrom',
}


# ============================================================================
# Schema Standardization Functions
# ============================================================================

def standardize_splice_sites_schema(
    df: Union[pl.DataFrame, pd.DataFrame],
    inplace: bool = False,
    verbose: bool = False
) -> Union[pl.DataFrame, pd.DataFrame]:
    """Standardize splice site annotation column names.
    
    Ensures consistent naming across the system by renaming synonym columns
    to their standard equivalents.
    
    Standard Mappings:
    - site_type → splice_type (GTF convention → biological terminology)
    - type → splice_type (generic → specific)
    
    Parameters
    ----------
    df : pl.DataFrame or pd.DataFrame
        Splice site annotations DataFrame
    inplace : bool, default=False
        If True, modify DataFrame in place (Pandas only)
    verbose : bool, default=False
        If True, print renaming operations
        
    Returns
    -------
    pl.DataFrame or pd.DataFrame
        DataFrame with standardized column names
        
    Examples
    --------
    >>> import polars as pl
    >>> df = pl.DataFrame({
    ...     'chrom': ['1'],
    ...     'position': [1000],
    ...     'site_type': ['donor'],  # Non-standard name
    ...     'gene_id': ['ENSG00000000001']
    ... })
    >>> standardized = standardize_splice_sites_schema(df)
    >>> 'splice_type' in standardized.columns
    True
    >>> 'site_type' in standardized.columns
    False
    
    Notes
    -----
    - This function is idempotent - calling it multiple times is safe
    - Only renames columns that exist and don't conflict with targets
    - Preserves all data and column order
    """
    return _standardize_schema(
        df=df,
        column_mapping=SPLICE_SITE_COLUMN_MAPPING,
        inplace=inplace,
        verbose=verbose,
        schema_name="splice_sites"
    )


def standardize_gene_features_schema(
    df: Union[pl.DataFrame, pd.DataFrame],
    inplace: bool = False,
    verbose: bool = False
) -> Union[pl.DataFrame, pd.DataFrame]:
    """Standardize gene feature column names.
    
    Standard Mappings:
    - seqname → chrom (GTF convention → genomics convention)
    - gene_type → gene_biotype (alternative naming)
    - biotype → gene_biotype (short form → full form)
    
    Parameters
    ----------
    df : pl.DataFrame or pd.DataFrame
        Gene features DataFrame
    inplace : bool, default=False
        If True, modify DataFrame in place (Pandas only)
    verbose : bool, default=False
        If True, print renaming operations
        
    Returns
    -------
    pl.DataFrame or pd.DataFrame
        DataFrame with standardized column names
    """
    return _standardize_schema(
        df=df,
        column_mapping=GENE_FEATURE_COLUMN_MAPPING,
        inplace=inplace,
        verbose=verbose,
        schema_name="gene_features"
    )


def standardize_transcript_features_schema(
    df: Union[pl.DataFrame, pd.DataFrame],
    inplace: bool = False,
    verbose: bool = False
) -> Union[pl.DataFrame, pd.DataFrame]:
    """Standardize transcript feature column names.
    
    Standard Mappings:
    - seqname → chrom
    - transcript_type → transcript_biotype
    - biotype → transcript_biotype
    
    Parameters
    ----------
    df : pl.DataFrame or pd.DataFrame
        Transcript features DataFrame
    inplace : bool, default=False
        If True, modify DataFrame in place (Pandas only)
    verbose : bool, default=False
        If True, print renaming operations
        
    Returns
    -------
    pl.DataFrame or pd.DataFrame
        DataFrame with standardized column names
    """
    return _standardize_schema(
        df=df,
        column_mapping=TRANSCRIPT_FEATURE_COLUMN_MAPPING,
        inplace=inplace,
        verbose=verbose,
        schema_name="transcript_features"
    )


def standardize_exon_features_schema(
    df: Union[pl.DataFrame, pd.DataFrame],
    inplace: bool = False,
    verbose: bool = False
) -> Union[pl.DataFrame, pd.DataFrame]:
    """Standardize exon feature column names.
    
    Standard Mappings:
    - seqname → chrom
    
    Parameters
    ----------
    df : pl.DataFrame or pd.DataFrame
        Exon features DataFrame
    inplace : bool, default=False
        If True, modify DataFrame in place (Pandas only)
    verbose : bool, default=False
        If True, print renaming operations
        
    Returns
    -------
    pl.DataFrame or pd.DataFrame
        DataFrame with standardized column names
    """
    return _standardize_schema(
        df=df,
        column_mapping=EXON_FEATURE_COLUMN_MAPPING,
        inplace=inplace,
        verbose=verbose,
        schema_name="exon_features"
    )


def get_standard_column_mapping(schema_type: str) -> Dict[str, str]:
    """Get the standard column mapping for a schema type.
    
    Parameters
    ----------
    schema_type : str
        Type of schema: 'splice_sites', 'gene_features', 
        'transcript_features', or 'exon_features'
        
    Returns
    -------
    Dict[str, str]
        Mapping from non-standard to standard column names
        
    Raises
    ------
    ValueError
        If schema_type is not recognized
        
    Examples
    --------
    >>> mapping = get_standard_column_mapping('splice_sites')
    >>> mapping['site_type']
    'splice_type'
    """
    mappings = {
        'splice_sites': SPLICE_SITE_COLUMN_MAPPING,
        'gene_features': GENE_FEATURE_COLUMN_MAPPING,
        'transcript_features': TRANSCRIPT_FEATURE_COLUMN_MAPPING,
        'exon_features': EXON_FEATURE_COLUMN_MAPPING,
    }
    
    if schema_type not in mappings:
        raise ValueError(
            f"Unknown schema type: {schema_type}. "
            f"Must be one of: {list(mappings.keys())}"
        )
    
    return mappings[schema_type]


# ============================================================================
# Internal Helper Functions
# ============================================================================

def _standardize_schema(
    df: Union[pl.DataFrame, pd.DataFrame],
    column_mapping: Dict[str, str],
    inplace: bool = False,
    verbose: bool = False,
    schema_name: str = "dataset"
) -> Union[pl.DataFrame, pd.DataFrame]:
    """Internal function to standardize schema.
    
    Parameters
    ----------
    df : pl.DataFrame or pd.DataFrame
        Input DataFrame
    column_mapping : Dict[str, str]
        Mapping from non-standard to standard column names
    inplace : bool
        Whether to modify in place (Pandas only)
    verbose : bool
        Whether to print operations
    schema_name : str
        Name of schema for logging
        
    Returns
    -------
    pl.DataFrame or pd.DataFrame
        Standardized DataFrame
    """
    is_polars = isinstance(df, pl.DataFrame)
    
    # Determine which columns need renaming
    rename_dict = {}
    for old_col, new_col in column_mapping.items():
        # Only rename if:
        # 1. Old column exists
        # 2. New column doesn't exist (avoid conflicts)
        if old_col in df.columns and new_col not in df.columns:
            rename_dict[old_col] = new_col
    
    # If nothing to rename, return as-is
    if not rename_dict:
        if verbose:
            print(f"[schema] No column renaming needed for {schema_name}")
        return df
    
    # Log renaming operations
    if verbose:
        print(f"[schema] Standardizing {schema_name} columns:")
        for old_col, new_col in rename_dict.items():
            print(f"  {old_col} → {new_col}")
    
    # Perform renaming based on DataFrame type
    if is_polars:
        # Polars: Always returns a new DataFrame
        return df.rename(rename_dict)
    else:
        # Pandas: Can modify in place or return copy
        if inplace:
            df.rename(columns=rename_dict, inplace=True)
            return df
        else:
            return df.rename(columns=rename_dict)


def standardize_all_schemas(
    splice_sites: Optional[Union[pl.DataFrame, pd.DataFrame]] = None,
    gene_features: Optional[Union[pl.DataFrame, pd.DataFrame]] = None,
    transcript_features: Optional[Union[pl.DataFrame, pd.DataFrame]] = None,
    exon_features: Optional[Union[pl.DataFrame, pd.DataFrame]] = None,
    verbose: bool = False
) -> Dict[str, Union[pl.DataFrame, pd.DataFrame]]:
    """Standardize all provided genomic datasets.
    
    Convenience function to standardize multiple datasets at once.
    
    Parameters
    ----------
    splice_sites : pl.DataFrame or pd.DataFrame, optional
        Splice site annotations
    gene_features : pl.DataFrame or pd.DataFrame, optional
        Gene features
    transcript_features : pl.DataFrame or pd.DataFrame, optional
        Transcript features
    exon_features : pl.DataFrame or pd.DataFrame, optional
        Exon features
    verbose : bool, default=False
        If True, print renaming operations
        
    Returns
    -------
    Dict[str, pl.DataFrame or pd.DataFrame]
        Dictionary with standardized DataFrames (only for provided inputs)
        
    Examples
    --------
    >>> result = standardize_all_schemas(
    ...     splice_sites=ss_df,
    ...     gene_features=gf_df,
    ...     verbose=True
    ... )
    >>> standardized_ss = result['splice_sites']
    >>> standardized_gf = result['gene_features']
    """
    result = {}
    
    if splice_sites is not None:
        result['splice_sites'] = standardize_splice_sites_schema(
            splice_sites, verbose=verbose
        )
    
    if gene_features is not None:
        result['gene_features'] = standardize_gene_features_schema(
            gene_features, verbose=verbose
        )
    
    if transcript_features is not None:
        result['transcript_features'] = standardize_transcript_features_schema(
            transcript_features, verbose=verbose
        )
    
    if exon_features is not None:
        result['exon_features'] = standardize_exon_features_schema(
            exon_features, verbose=verbose
        )
    
    return result


# ============================================================================
# Documentation and Examples
# ============================================================================

def print_standard_schemas():
    """Print all standard schema mappings for reference.
    
    Useful for understanding what column names are considered standard
    and what synonyms are automatically converted.
    """
    print("=" * 80)
    print("STANDARD GENOMIC DATASET SCHEMAS")
    print("=" * 80)
    print()
    
    schemas = {
        'Splice Sites': SPLICE_SITE_COLUMN_MAPPING,
        'Gene Features': GENE_FEATURE_COLUMN_MAPPING,
        'Transcript Features': TRANSCRIPT_FEATURE_COLUMN_MAPPING,
        'Exon Features': EXON_FEATURE_COLUMN_MAPPING,
    }
    
    for schema_name, mapping in schemas.items():
        print(f"{schema_name}:")
        print("-" * 40)
        if mapping:
            for old_col, new_col in mapping.items():
                print(f"  {old_col:20s} → {new_col}")
        else:
            print("  (No mappings defined)")
        print()


if __name__ == '__main__':
    # Print standard schemas when run as script
    print_standard_schemas()

