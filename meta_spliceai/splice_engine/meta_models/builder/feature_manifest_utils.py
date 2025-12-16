"""
Feature Manifest Utilities

This module provides utilities for creating and managing feature manifests
that document both raw and encoded feature schemas, including:
1. Feature names and types (numerical, categorical, k-mer, etc.)
2. Categorical encoding specifications
3. Feature statistics and metadata
4. Version information for reproducibility

The manifest design supports both human readability and programmatic use.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import pandas as pd
import polars as pl
from datetime import datetime

from meta_spliceai.splice_engine.meta_models.builder.feature_schema import (
    CATEGORICAL_FEATURES,
    ALWAYS_NUMERICAL_FEATURES,
    is_kmer_feature,
    get_encoding_spec
)


@dataclass
class FeatureInfo:
    """Information about a single feature in the manifest.
    
    Attributes
    ----------
    name : str
        Feature name (as it appears in the model)
    original_name : str, optional
        Original name before encoding (if different)
    dtype : str
        Data type (int64, float64, etc.)
    feature_type : str
        Category: 'numerical', 'categorical', 'kmer', 'probability', 'genomic', etc.
    is_encoded : bool
        Whether the feature was categorically encoded
    encoding_method : str, optional
        Encoding method if is_encoded=True ('ordinal', 'custom', etc.)
    encoding_mapping : dict, optional
        Encoding mapping (original_value → encoded_value) if applicable
    statistics : dict, optional
        Feature statistics (min, max, mean, unique_count, etc.)
    description : str, optional
        Human-readable description
    """
    name: str
    original_name: Optional[str] = None
    dtype: str = "unknown"
    feature_type: str = "numerical"
    is_encoded: bool = False
    encoding_method: Optional[str] = None
    encoding_mapping: Optional[Dict[str, int]] = None
    statistics: Optional[Dict[str, Any]] = None
    description: str = ""


@dataclass
class FeatureManifest:
    """Complete feature manifest with metadata.
    
    Attributes
    ----------
    features : list of FeatureInfo
        List of all features in the dataset
    metadata : dict
        Manifest metadata (creation_date, version, etc.)
    encoding_specs : dict
        Categorical encoding specifications used
    excluded_features : list
        Features that were excluded (leakage, metadata, etc.)
    """
    features: List[FeatureInfo]
    metadata: Dict[str, Any]
    encoding_specs: Dict[str, Dict[str, Any]]
    excluded_features: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'features': [asdict(f) for f in self.features],
            'metadata': self.metadata,
            'encoding_specs': self.encoding_specs,
            'excluded_features': self.excluded_features
        }
    
    def to_json(self, filepath: Union[str, Path], indent: int = 2):
        """Save as JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=indent)
    
    def to_csv(self, filepath: Union[str, Path], include_encoding: bool = True):
        """
        Save as CSV file (simplified view).
        
        Parameters
        ----------
        filepath : str or Path
            Output CSV file path
        include_encoding : bool
            Include encoding information columns
        """
        rows = []
        for feature in self.features:
            row = {
                'feature': feature.name,
                'dtype': feature.dtype,
                'feature_type': feature.feature_type,
                'is_encoded': feature.is_encoded,
                'description': feature.description
            }
            
            if include_encoding and feature.is_encoded:
                row['original_name'] = feature.original_name or feature.name
                row['encoding_method'] = feature.encoding_method
                
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(filepath, index=False)
    
    @classmethod
    def from_json(cls, filepath: Union[str, Path]) -> 'FeatureManifest':
        """Load from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        features = [FeatureInfo(**f) for f in data['features']]
        
        return cls(
            features=features,
            metadata=data['metadata'],
            encoding_specs=data['encoding_specs'],
            excluded_features=data['excluded_features']
        )
    
    def get_feature_names(self) -> List[str]:
        """Get list of all feature names."""
        return [f.name for f in self.features]
    
    def get_categorical_features(self) -> List[FeatureInfo]:
        """Get list of categorical features."""
        return [f for f in self.features if f.feature_type == 'categorical']
    
    def get_encoded_features(self) -> List[FeatureInfo]:
        """Get list of encoded features."""
        return [f for f in self.features if f.is_encoded]


def infer_feature_type(feature_name: str, dtype: Any) -> str:
    """
    Infer the feature type based on name and dtype.
    
    Parameters
    ----------
    feature_name : str
        Name of the feature
    dtype : Any
        Data type (polars dtype or string)
        
    Returns
    -------
    str
        Feature type: 'numerical', 'categorical', 'kmer', 'probability', etc.
    """
    # K-mer features
    if is_kmer_feature(feature_name):
        return 'kmer'
    
    # Probability-related features
    if any(x in feature_name.lower() for x in ['score', 'probability', 'prob', 'entropy']):
        return 'probability'
    
    # Context features
    if 'context' in feature_name.lower():
        return 'context'
    
    # Genomic coordinate features
    if any(x in feature_name.lower() for x in ['start', 'end', 'length', 'position']):
        return 'genomic'
    
    # Categorical features
    if feature_name in CATEGORICAL_FEATURES:
        return 'categorical'
    
    # Default to numerical
    return 'numerical'


def compute_feature_statistics(df: pl.DataFrame, feature_name: str) -> Dict[str, Any]:
    """
    Compute basic statistics for a feature.
    
    Parameters
    ----------
    df : pl.DataFrame
        Input dataframe
    feature_name : str
        Feature to analyze
        
    Returns
    -------
    dict
        Statistics dictionary
    """
    stats = {}
    
    try:
        col = df[feature_name]
        
        # Basic stats
        stats['count'] = len(col)
        stats['null_count'] = col.null_count()
        stats['unique_count'] = col.n_unique()
        
        # Numeric stats (if applicable)
        dtype = df.select(feature_name).dtypes[0]
        if dtype in [pl.Int64, pl.Int32, pl.Float64, pl.Float32]:
            stats['min'] = float(col.min())
            stats['max'] = float(col.max())
            stats['mean'] = float(col.mean())
            stats['std'] = float(col.std())
            
            # Percentiles
            stats['p25'] = float(col.quantile(0.25))
            stats['p50'] = float(col.quantile(0.50))
            stats['p75'] = float(col.quantile(0.75))
    
    except Exception as e:
        stats['error'] = str(e)
    
    return stats


def create_feature_manifest(
    df: pl.DataFrame,
    excluded_features: Optional[List[str]] = None,
    include_statistics: bool = True,
    categorical_encodings: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> FeatureManifest:
    """
    Create a comprehensive feature manifest from a dataframe.
    
    Parameters
    ----------
    df : pl.DataFrame
        Input dataframe (after preprocessing and encoding)
    excluded_features : list of str, optional
        List of features that were excluded
    include_statistics : bool
        Whether to compute feature statistics
    categorical_encodings : dict, optional
        Record of categorical encodings applied
    metadata : dict, optional
        Additional metadata to include
        
    Returns
    -------
    FeatureManifest
        Complete feature manifest
    """
    features = []
    
    for col_name in df.columns:
        # Get dtype
        dtype_obj = df.select(col_name).dtypes[0]
        dtype_str = str(dtype_obj)
        
        # Infer feature type
        feature_type = infer_feature_type(col_name, dtype_obj)
        
        # Check if this is an encoded feature
        is_encoded = False
        encoding_method = None
        encoding_mapping = None
        original_name = None
        
        if col_name in CATEGORICAL_FEATURES:
            is_encoded = True
            spec = get_encoding_spec(col_name)
            if spec:
                encoding_method = spec.encoding_type
                if spec.custom_mapping:
                    # Convert to serializable format
                    encoding_mapping = {str(k): int(v) for k, v in spec.custom_mapping.items()}
                original_name = col_name  # Same name, but different type
        
        # Compute statistics
        stats = None
        if include_statistics:
            stats = compute_feature_statistics(df, col_name)
        
        # Create feature info
        feature_info = FeatureInfo(
            name=col_name,
            original_name=original_name,
            dtype=dtype_str,
            feature_type=feature_type,
            is_encoded=is_encoded,
            encoding_method=encoding_method,
            encoding_mapping=encoding_mapping,
            statistics=stats,
            description=""  # Can be filled in later
        )
        
        features.append(feature_info)
    
    # Create metadata
    if metadata is None:
        metadata = {}
    
    metadata.update({
        'creation_date': datetime.now().isoformat(),
        'total_features': len(features),
        'encoded_features': sum(1 for f in features if f.is_encoded),
        'kmer_features': sum(1 for f in features if f.feature_type == 'kmer'),
        'numerical_features': sum(1 for f in features if f.feature_type == 'numerical'),
        'probability_features': sum(1 for f in features if f.feature_type == 'probability'),
    })
    
    # Get encoding specs from registry
    encoding_specs = {}
    for feature_name in CATEGORICAL_FEATURES:
        spec = get_encoding_spec(feature_name)
        if spec:
            encoding_specs[feature_name] = {
                'encoding_type': spec.encoding_type,
                'handle_unknown': spec.handle_unknown,
                'default_value': spec.default_value,
                'description': spec.description
            }
    
    return FeatureManifest(
        features=features,
        metadata=metadata,
        encoding_specs=encoding_specs,
        excluded_features=excluded_features or []
    )


def save_feature_manifests(
    df: pl.DataFrame,
    output_dir: Union[str, Path],
    excluded_features: Optional[List[str]] = None,
    include_statistics: bool = True,
    save_legacy_csv: bool = True
):
    """
    Save feature manifests in multiple formats.
    
    This function saves:
    1. feature_manifest.json - Complete manifest with all metadata
    2. feature_manifest.csv - Simplified CSV for quick reference
    3. feature_manifest_legacy.csv - Legacy format (feature names only) for compatibility
    
    Parameters
    ----------
    df : pl.DataFrame
        Input dataframe (after preprocessing and encoding)
    output_dir : str or Path
        Output directory
    excluded_features : list of str, optional
        List of features that were excluded
    include_statistics : bool
        Whether to compute feature statistics
    save_legacy_csv : bool
        Whether to save legacy CSV format
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create manifest
    manifest = create_feature_manifest(
        df=df,
        excluded_features=excluded_features,
        include_statistics=include_statistics
    )
    
    # Save JSON (complete manifest)
    manifest.to_json(output_dir / "feature_manifest.json")
    
    # Save CSV (enriched manifest)
    manifest.to_csv(output_dir / "feature_manifest.csv", include_encoding=True)
    
    # Save legacy CSV (just feature names) for backward compatibility
    if save_legacy_csv:
        pd.DataFrame({
            "feature": manifest.get_feature_names()
        }).to_csv(output_dir / "feature_manifest_legacy.csv", index=False)
    
    print(f"  📄 Feature manifests saved to {output_dir}/")
    print(f"     - feature_manifest.json (complete metadata)")
    print(f"     - feature_manifest.csv (enriched view)")
    if save_legacy_csv:
        print(f"     - feature_manifest_legacy.csv (legacy format)")


def compare_manifests(
    manifest1: FeatureManifest,
    manifest2: FeatureManifest
) -> Dict[str, Any]:
    """
    Compare two feature manifests and report differences.
    
    Parameters
    ----------
    manifest1 : FeatureManifest
        First manifest (e.g., training)
    manifest2 : FeatureManifest
        Second manifest (e.g., inference)
        
    Returns
    -------
    dict
        Comparison results
    """
    names1 = set(manifest1.get_feature_names())
    names2 = set(manifest2.get_feature_names())
    
    comparison = {
        'matching_features': list(names1 & names2),
        'only_in_first': list(names1 - names2),
        'only_in_second': list(names2 - names1),
        'feature_count': {
            'first': len(names1),
            'second': len(names2)
        },
        'type_mismatches': []
    }
    
    # Check for type mismatches in common features
    feat_dict1 = {f.name: f for f in manifest1.features}
    feat_dict2 = {f.name: f for f in manifest2.features}
    
    for name in comparison['matching_features']:
        f1 = feat_dict1[name]
        f2 = feat_dict2[name]
        
        if f1.dtype != f2.dtype or f1.feature_type != f2.feature_type:
            comparison['type_mismatches'].append({
                'feature': name,
                'first': {'dtype': f1.dtype, 'type': f1.feature_type},
                'second': {'dtype': f2.dtype, 'type': f2.feature_type}
            })
    
    return comparison

