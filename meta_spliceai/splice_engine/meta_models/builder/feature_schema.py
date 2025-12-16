"""
Feature schema and categorical encoding configuration.

This module provides a centralized specification for:
1. Which features are categorical vs. numerical
2. How categorical features should be encoded
3. Feature type validation

This ensures consistency between training and inference pipelines.
"""
from __future__ import annotations

from typing import Dict, List, Literal, Optional
from dataclasses import dataclass
import polars as pl


# ==============================================================================
# Categorical Feature Specifications
# ==============================================================================

@dataclass
class CategoricalFeatureSpec:
    """Specification for a categorical feature and its encoding strategy.
    
    Parameters
    ----------
    name : str
        Feature column name
    encoding_type : str
        Encoding strategy: 'ordinal', 'onehot', 'custom'
    custom_mapping : dict, optional
        Custom mapping for 'custom' encoding type
    handle_unknown : str
        How to handle unknown categories: 'error', 'use_default', 'skip'
    default_value : int, optional
        Default value for unknown categories (when handle_unknown='use_default')
    description : str, optional
        Human-readable description
    """
    name: str
    encoding_type: Literal['ordinal', 'onehot', 'custom']
    custom_mapping: Optional[Dict[str, int]] = None
    handle_unknown: Literal['error', 'use_default', 'skip'] = 'use_default'
    default_value: Optional[int] = None
    description: str = ""


# ==============================================================================
# Categorical Feature Registry
# ==============================================================================

# Chromosome encoding: Custom mapping with biological ordering
CHROM_SPEC = CategoricalFeatureSpec(
    name='chrom',
    encoding_type='custom',
    custom_mapping={
        # Standard autosomes
        **{str(i): i for i in range(1, 23)},  # 1-22
        **{f'chr{i}': i for i in range(1, 23)},  # chr1-chr22
        # Sex chromosomes
        'X': 23, 'chrX': 23,
        'Y': 24, 'chrY': 24,
        # Mitochondrial
        'MT': 25, 'chrMT': 25, 'M': 25, 'chrM': 25,
    },
    handle_unknown='use_default',
    default_value=100,  # Unknown chromosomes/scaffolds start at 100
    description="Chromosome identifier with biological ordering"
)

# Strand encoding: Simple binary encoding
STRAND_SPEC = CategoricalFeatureSpec(
    name='strand',
    encoding_type='custom',
    custom_mapping={
        '+': 1,
        '-': -1,
        '.': 0,  # unstranded/unknown
    },
    handle_unknown='use_default',
    default_value=0,
    description="Strand orientation: +1 (forward), -1 (reverse), 0 (unknown)"
)

# Gene type encoding: Common biotypes
GENE_TYPE_SPEC = CategoricalFeatureSpec(
    name='gene_type',
    encoding_type='custom',
    custom_mapping={
        'protein_coding': 1,
        'lncRNA': 2,
        'miRNA': 3,
        'snRNA': 4,
        'snoRNA': 5,
        'rRNA': 6,
        'pseudogene': 7,
        'processed_pseudogene': 8,
        'unprocessed_pseudogene': 9,
        'transcribed_pseudogene': 10,
        'IG_V_gene': 11,
        'IG_C_gene': 12,
        'IG_J_gene': 13,
        'IG_D_gene': 14,
        'TR_V_gene': 15,
        'TR_C_gene': 16,
        'TR_J_gene': 17,
        'TR_D_gene': 18,
    },
    handle_unknown='use_default',
    default_value=99,  # Unknown gene types
    description="Gene biotype classification"
)


# Registry of all categorical features
CATEGORICAL_FEATURES: Dict[str, CategoricalFeatureSpec] = {
    'chrom': CHROM_SPEC,
    # 'strand': STRAND_SPEC,  # Currently excluded as metadata
    # 'gene_type': GENE_TYPE_SPEC,  # Currently excluded as metadata
}


# ==============================================================================
# Numerical Feature Specifications (for validation)
# ==============================================================================

# Features that should ALWAYS be treated as numerical, even if they have
# few distinct values or look categorical
ALWAYS_NUMERICAL_FEATURES: List[str] = [
    # K-mer counts (even if some rare k-mers have few distinct values)
    # Pattern: Any feature matching /^[ACGT]{k}$/ for k=1,2,3,...
    # Note: These are dynamically detected by regex, not listed explicitly
    
    # Splice site scores
    'donor_score',
    'acceptor_score',
    'neither_score',
    
    # Probability-derived features
    'relative_donor_probability',
    'splice_probability',
    'donor_acceptor_diff',
    'splice_neither_diff',
    'donor_acceptor_logodds',
    'splice_neither_logodds',
    'probability_entropy',
    
    # Context scores
    'context_score_p1',
    'context_score_m1',
    'context_score_p2',
    'context_score_m2',
    'context_neighbor_mean',
    'context_asymmetry',
    'context_max',
    
    # Genomic coordinates (even though they're integers)
    'gene_start',
    'gene_end',
    'transcript_start',
    'transcript_end',
    'exon_start',
    'exon_end',
    'tx_start',
    'tx_end',
    
    # Derived genomic features
    'gene_length',
    'transcript_length',
    'exon_length',
    'exon_rank',
    'num_overlaps',
    
    # Sequence features
    'gc_content',
    'sequence_length',
    'sequence_complexity',
]


# ==============================================================================
# Encoding Functions
# ==============================================================================

def encode_categorical_features(
    df: pl.DataFrame,
    features_to_encode: Optional[List[str]] = None,
    verbose: bool = False
) -> pl.DataFrame:
    """
    Encode categorical features according to their specifications.
    
    Parameters
    ----------
    df : pl.DataFrame
        Input dataframe with categorical features
    features_to_encode : list of str, optional
        Specific features to encode. If None, encode all registered categorical features.
    verbose : bool
        Print encoding information
        
    Returns
    -------
    pl.DataFrame
        Dataframe with encoded categorical features
        
    Notes
    -----
    This function ensures consistency between training and inference by using
    the same encoding specifications from CATEGORICAL_FEATURES registry.
    """
    result_df = df.clone()
    
    if features_to_encode is None:
        features_to_encode = list(CATEGORICAL_FEATURES.keys())
    
    for feature_name in features_to_encode:
        if feature_name not in result_df.columns:
            if verbose:
                print(f"  ⚠️  Feature '{feature_name}' not in dataframe, skipping")
            continue
            
        if feature_name not in CATEGORICAL_FEATURES:
            if verbose:
                print(f"  ⚠️  No encoding spec for '{feature_name}', skipping")
            continue
        
        spec = CATEGORICAL_FEATURES[feature_name]
        
        if verbose:
            print(f"  Encoding '{feature_name}' using {spec.encoding_type} encoding...")
        
        # Get unique values
        unique_vals = result_df.select(pl.col(feature_name).unique())[feature_name].to_list()
        
        # Check if already encoded (numeric type)
        dtype = result_df.select(feature_name).dtypes[0]
        if dtype in [pl.Int64, pl.Int32, pl.Int16, pl.Int8, pl.UInt64, pl.UInt32, pl.UInt16, pl.UInt8]:
            if verbose:
                print(f"    Feature '{feature_name}' is already numeric, skipping encoding")
            continue
        
        if spec.encoding_type == 'custom':
            # Apply custom mapping
            mapping = spec.custom_mapping.copy()
            
            # Handle unknown categories
            if spec.handle_unknown == 'use_default':
                next_code = spec.default_value
                for val in unique_vals:
                    if val not in mapping:
                        mapping[val] = next_code
                        if next_code >= spec.default_value:
                            next_code += 1
            elif spec.handle_unknown == 'error':
                unknown = [v for v in unique_vals if v not in mapping]
                if unknown:
                    raise ValueError(
                        f"Unknown categories for '{feature_name}': {unknown}\n"
                        f"Known categories: {list(mapping.keys())}"
                    )
            
            # Apply mapping
            col_list = result_df[feature_name].to_list()
            encoded = [mapping.get(x, spec.default_value or 0) for x in col_list]
            
            result_df = result_df.with_columns(
                pl.Series(feature_name, encoded)
            )
            
            if verbose:
                if len(mapping) > 20:
                    print(f"    Encoded {len(mapping)} unique values")
                else:
                    print(f"    Mapping: {mapping}")
        
        elif spec.encoding_type == 'ordinal':
            # Simple ordinal encoding (alphabetical order)
            unique_sorted = sorted(unique_vals)
            mapping = {val: idx for idx, val in enumerate(unique_sorted)}
            
            col_list = result_df[feature_name].to_list()
            encoded = [mapping.get(x, spec.default_value or -1) for x in col_list]
            
            result_df = result_df.with_columns(
                pl.Series(feature_name, encoded)
            )
            
            if verbose:
                print(f"    Ordinal encoding: {mapping}")
        
        elif spec.encoding_type == 'onehot':
            # One-hot encoding (creates multiple columns)
            # TODO: Implement if needed
            raise NotImplementedError("One-hot encoding not yet implemented")
    
    if verbose:
        print(f"  ✅ Categorical encoding completed")
    
    return result_df


def is_kmer_feature(feature_name: str) -> bool:
    """
    Check if a feature is a k-mer count feature.
    
    Parameters
    ----------
    feature_name : str
        Feature name to check
        
    Returns
    -------
    bool
        True if the feature is a k-mer count
        
    Examples
    --------
    >>> is_kmer_feature('AAA')
    True
    >>> is_kmer_feature('donor_score')
    False
    """
    import re
    kmer_pattern = re.compile(r'^[ACGT]+$')
    return bool(kmer_pattern.match(feature_name))


def validate_feature_types(df: pl.DataFrame, verbose: bool = False) -> Dict[str, List[str]]:
    """
    Validate that features have appropriate data types.
    
    Parameters
    ----------
    df : pl.DataFrame
        Input dataframe
    verbose : bool
        Print validation information
        
    Returns
    -------
    dict
        Dictionary with keys: 'categorical', 'numerical', 'kmer', 'invalid'
        
    Raises
    ------
    ValueError
        If any features have invalid types (e.g., string k-mers)
    """
    validation_result = {
        'categorical': [],
        'numerical': [],
        'kmer': [],
        'invalid': []
    }
    
    for col in df.columns:
        dtype = df.select(col).dtypes[0]
        
        # Check k-mer features
        if is_kmer_feature(col):
            if dtype in [pl.Int64, pl.Int32, pl.Float64, pl.Float32]:
                validation_result['kmer'].append(col)
            else:
                validation_result['invalid'].append((col, 'kmer should be numerical'))
        
        # Check categorical features
        elif col in CATEGORICAL_FEATURES:
            validation_result['categorical'].append(col)
        
        # Check always-numerical features
        elif col in ALWAYS_NUMERICAL_FEATURES:
            if dtype in [pl.Int64, pl.Int32, pl.Float64, pl.Float32]:
                validation_result['numerical'].append(col)
            else:
                validation_result['invalid'].append((col, 'should be numerical'))
        
        # Other features
        else:
            if dtype in [pl.Int64, pl.Int32, pl.Float64, pl.Float32]:
                validation_result['numerical'].append(col)
            elif dtype in [pl.Utf8, pl.Categorical]:
                validation_result['categorical'].append(col)
    
    if verbose:
        print(f"Feature validation:")
        print(f"  Categorical: {len(validation_result['categorical'])}")
        print(f"  Numerical: {len(validation_result['numerical'])}")
        print(f"  K-mers: {len(validation_result['kmer'])}")
        if validation_result['invalid']:
            print(f"  ⚠️  Invalid: {validation_result['invalid']}")
    
    if validation_result['invalid']:
        raise ValueError(f"Invalid feature types found: {validation_result['invalid']}")
    
    return validation_result


# ==============================================================================
# Convenience Functions
# ==============================================================================

def get_categorical_feature_names() -> List[str]:
    """Return list of all registered categorical feature names."""
    return list(CATEGORICAL_FEATURES.keys())


def get_numerical_feature_patterns() -> List[str]:
    """Return list of feature names that should always be numerical."""
    return ALWAYS_NUMERICAL_FEATURES.copy()


def is_categorical_feature(feature_name: str) -> bool:
    """Check if a feature is registered as categorical."""
    return feature_name in CATEGORICAL_FEATURES


def get_encoding_spec(feature_name: str) -> Optional[CategoricalFeatureSpec]:
    """Get the encoding specification for a categorical feature."""
    return CATEGORICAL_FEATURES.get(feature_name)

