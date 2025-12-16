"""
Sequence featurization for meta models.

This module handles the featurization of sequence data for different prediction types
(TP, TN, FP, FN), providing a unified interface for the meta-model training pipeline.
"""

import pandas as pd
import polars as pl
from typing import Dict, List, Tuple, Optional, Union, Set, Any, Literal

# Import original functionality without modifying it
from meta_spliceai.splice_engine.splice_error_analyzer import (
    featurize_gene_sequences,
    downsample_dataframe
)

# Import moved to function level to avoid circular imports with kmer_features.py
from meta_spliceai.splice_engine.sequence_featurizer import harmonize_features
from meta_spliceai.splice_engine.utils_doc import (
    print_emphasized, print_with_indent
)


def featurize_analysis_sequences(
    sequence_dfs: Dict[str, Union[pd.DataFrame, pl.DataFrame]],
    *,
    kmer_sizes: List[int] = [6],
    downsample_options: Dict[str, Dict[str, float]] = None,
    verbose: int = 1
) -> Tuple[Dict[str, Union[pd.DataFrame, pl.DataFrame]], Dict[str, List[str]]]:
    """
    Featurize analysis sequences for different prediction types.
    
    This function takes a dictionary of sequence DataFrames, where the keys are
    prediction types (e.g., 'TP', 'TN', 'FP', 'FN') and the values are the
    corresponding sequence DataFrames. It featurizes each DataFrame and returns
    dictionaries of featurized DataFrames and feature sets.
    
    Parameters
    ----------
    sequence_dfs : Dict[str, Union[pd.DataFrame, pl.DataFrame]]
        Dictionary of sequence DataFrames, keyed by prediction type
    kmer_sizes : List[int], optional
        List of k-mer sizes to extract, by default [6]
    downsample_options : Dict[str, Dict[str, float]], optional
        Dictionary of downsampling options for each prediction type, by default None.
        Format: {'TN': {'fraction': 0.5, 'max_size': 50000}, 'FN': {...}}
    verbose : int, optional
        Verbosity level, by default 1
        
    Returns
    -------
    Tuple[Dict[str, Union[pd.DataFrame, pl.DataFrame]], Dict[str, List[str]]]
        Dictionary of featurized DataFrames and dictionary of feature sets
    """
    if downsample_options is None:
        downsample_options = {
            'TN': {'fraction': 0.5, 'max_size': 50000},
            'FN': {'fraction': 1.0, 'max_size': 50000}
        }
    
    # Initialize dictionaries to store results
    featurized_dfs = {}
    feature_sets = {}
    
    # Process each sequence type
    for pred_type, sequence_df in sequence_dfs.items():
        if sequence_df is None or sequence_df.shape[0] == 0:
            continue
            
        # Apply downsampling if options exist for this prediction type
        if pred_type in downsample_options:
            options = downsample_options[pred_type]
            fraction = options.get('fraction', 1.0)
            max_size = options.get('max_size', None)
            
            if fraction < 1.0 or max_size is not None:
                print_with_indent(f"[info] Downsampling {pred_type} sequences...", indent_level=1)
                sequence_df = downsample_dataframe(
                    sequence_df,
                    sample_fraction=fraction,
                    max_sample_size=max_size,
                    verbose=verbose
                )
                if verbose >= 1:
                    print(f"[info] Shape of (downsampled) {pred_type} sequence_df: {sequence_df.shape}")
        
        # Featurize the sequence data
        print_emphasized(f"[action] Featurizing analysis sequences of type {pred_type} ...")
        
        # Import locally to avoid circular imports
        from meta_spliceai.splice_engine.meta_models.features.kmer_features import make_kmer_features
        
        featurized_df, features = make_kmer_features(
            sequence_df, 
            kmer_sizes=kmer_sizes, 
            return_feature_set=True, 
            verbose=verbose
        )
        
        # Store results
        featurized_dfs[pred_type] = featurized_df
        feature_sets[pred_type] = features
    
    # Check for feature set consistency between different prediction types
    _check_feature_consistency(feature_sets, verbose=verbose)
    
    return featurized_dfs, feature_sets


def _check_feature_consistency(
    feature_sets: Dict[str, List[str]],
    verbose: int = 1
) -> bool:
    """
    Check for consistency between feature sets from different prediction types.
    
    Parameters
    ----------
    feature_sets : Dict[str, List[str]]
        Dictionary of feature sets, keyed by prediction type
    verbose : int, optional
        Verbosity level, by default 1
        
    Returns
    -------
    bool
        True if all feature sets are consistent, False otherwise
    """
    # Get all prediction types with features
    pred_types = list(feature_sets.keys())
    if len(pred_types) <= 1:
        return True
    
    # Use TP as the reference if available, otherwise use the first type
    reference_type = 'TP' if 'TP' in pred_types else pred_types[0]
    reference_features = set(feature_sets[reference_type])
    
    # Compare each feature set to the reference
    all_consistent = True
    for pred_type in pred_types:
        if pred_type == reference_type:
            continue
            
        features = set(feature_sets[pred_type])
        missing_in_reference = features - reference_features
        missing_in_current = reference_features - features
        
        if missing_in_reference or missing_in_current:
            all_consistent = False
            if verbose >= 1:
                print(f"[diagnostics] Feature set mismatch detected between {reference_type} and {pred_type}!")
                print(f"Features missing in {reference_type}: {missing_in_reference}")
                print(f"Features missing in {pred_type}: {missing_in_current}")
        elif verbose >= 1:
            print_emphasized(f"[info] Feature sets are consistent between {reference_type} and {pred_type}.")
    
    return all_consistent


def harmonize_all_feature_sets(
    featurized_dfs: Dict[str, Union[pd.DataFrame, pl.DataFrame]],
    feature_sets: Dict[str, List[str]],
    default_value: float = 0.0,
    verbose: int = 1
) -> Dict[str, Union[pd.DataFrame, pl.DataFrame]]:
    """
    Harmonize feature sets across all prediction types (TP, TN, FP, FN).
    
    This function ensures that all DataFrames have the same feature columns,
    which is crucial for training consistent meta-models across different
    prediction types.
    
    Parameters
    ----------
    featurized_dfs : Dict[str, Union[pd.DataFrame, pl.DataFrame]]
        Dictionary of featurized DataFrames, keyed by prediction type
    feature_sets : Dict[str, List[str]]
        Dictionary of feature sets, keyed by prediction type
    default_value : float, optional
        Default value for missing features, by default 0.0
    verbose : int, optional
        Verbosity level, by default 1
        
    Returns
    -------
    Dict[str, Union[pd.DataFrame, pl.DataFrame]]
        Dictionary of harmonized DataFrames, keyed by prediction type
    """
    # Skip empty inputs
    if not featurized_dfs or not feature_sets:
        if verbose >= 1:
            print("[warning] Empty inputs provided to harmonize_all_feature_sets")
        return featurized_dfs
    
    # Get all prediction types with data
    pred_types = list(featurized_dfs.keys())
    if len(pred_types) <= 1:
        if verbose >= 1:
            print(f"[info] Only one prediction type ({pred_types[0]}) provided, skipping harmonization")
        return featurized_dfs
    
    # Determine the union of all features across all prediction types
    all_features = set()
    for feature_set in feature_sets.values():
        all_features.update(feature_set)
    
    if verbose >= 1:
        print(f"[info] Harmonizing feature sets across {len(pred_types)} prediction types")
        print(f"[info] Total number of unique features: {len(all_features)}")
    
    # Process each prediction type
    harmonized_dfs = {}
    for pred_type, df in featurized_dfs.items():
        if df is None or df.shape[0] == 0:
            harmonized_dfs[pred_type] = df
            continue
        
        # Get features for this prediction type
        features = set(feature_sets.get(pred_type, []))
        missing_features = all_features - features
        
        if verbose >= 1 and missing_features:
            print(f"[info] Adding {len(missing_features)} missing features to {pred_type} data")
        
        # Check if the dataframe is Polars
        if isinstance(df, pl.DataFrame):
            # Add missing features to Polars dataframe
            for feature in missing_features:
                df = df.with_columns(pl.lit(default_value).alias(feature))
        else:
            # Add missing features to pandas dataframe
            for feature in missing_features:
                df[feature] = default_value
        
        harmonized_dfs[pred_type] = df
    
    return harmonized_dfs


def create_feature_matrix(
    featurized_dfs: Dict[str, Union[pd.DataFrame, pl.DataFrame]],
    feature_sets: Dict[str, List[str]],
    label_mode: Literal['error_analysis', 'meta_model'] = 'meta_model',
    balance_classes: bool = True,
    drop_columns: List[str] = ['gene_id', 'position', 'sequence'],
    verbose: int = 1
) -> Tuple[Union[pd.DataFrame, pl.DataFrame], Union[pd.Series, pl.Series]]:
    """
    Create a feature matrix for model training from featurized DataFrames.
    
    Parameters
    ----------
    featurized_dfs : Dict[str, Union[pd.DataFrame, pl.DataFrame]]
        Dictionary of featurized DataFrames, keyed by prediction type
    feature_sets : Dict[str, List[str]]
        Dictionary of feature sets, keyed by prediction type
    label_mode : Literal['error_analysis', 'meta_model'], optional
        The labeling mode to use, by default 'meta_model'.
        - 'error_analysis': Binary classification (1=errors, 0=correct)
        - 'meta_model': Multi-class classification (0=donor, 1=acceptor, 2=neither)
    balance_classes : bool, optional
        Whether to balance the classes in the feature matrix, by default True
    drop_columns : List[str], optional
        Columns to drop from the feature matrix, by default ['gene_id', 'position', 'sequence']
    verbose : int, optional
        Verbosity level, by default 1
        
    Returns
    -------
    Tuple[Union[pd.DataFrame, pl.DataFrame], Union[pd.Series, pl.Series]]
        Feature matrix and label vector
    """
    # Validate inputs
    pred_types = list(featurized_dfs.keys())
    if not pred_types:
        raise ValueError("No prediction types provided in featurized_dfs")
    
    # Convert all DataFrames to pandas for consistent processing
    pandas_dfs = {}
    for pred_type, df in featurized_dfs.items():
        if df is None or df.shape[0] == 0:
            continue
            
        if isinstance(df, pl.DataFrame):
            pandas_dfs[pred_type] = df.to_pandas()
        else:
            pandas_dfs[pred_type] = df
    
    # Add labels based on the selected mode
    labeled_dfs = {}
    
    if label_mode == 'error_analysis':
        # Binary classification: 1=errors (FP/FN), 0=correct (TP/TN)
        for pred_type, df in pandas_dfs.items():
            df = df.copy()  # Create a copy to avoid modifying the original
            
            # Label as error (1) or correct (0)
            if pred_type in ['FP', 'FN']:
                df['label'] = 1  # Errors
            else:  # TP or TN
                df['label'] = 0  # Correct predictions
            
            labeled_dfs[pred_type] = df
            
    else:  # meta_model mode
        # Multi-class classification based on biological annotation
        for pred_type, df in pandas_dfs.items():
            df = df.copy()  # Create a copy to avoid modifying the original
            
            if 'splice_type' in df.columns:
                # Use the biological annotation directly
                df['label'] = df['splice_type'].map({'donor': 0, 'acceptor': 1, None: 2})
            else:
                # For datasets without splice_type column, infer from pred_type
                if pred_type in ['TP', 'FN']:
                    # Check if we can determine donor vs acceptor
                    if 'is_donor' in df.columns:
                        df['label'] = df['is_donor'].map({True: 0, False: 1})
                    elif 'is_acceptor' in df.columns:
                        df['label'] = df['is_acceptor'].map({False: 0, True: 1})
                    else:
                        # Default to donor if can't determine (should be fixed in dataset)
                        print(f"[warning] Cannot determine splice type for {pred_type}, defaulting to donor (0)")
                        df['label'] = 0
                else:  # FP or TN
                    # These are neither donor nor acceptor
                    df['label'] = 2
            
            labeled_dfs[pred_type] = df
    
    # Combine all DataFrames
    combined_df = pd.concat(list(labeled_dfs.values()), axis=0)
    
    # Balance classes if requested
    if balance_classes:
        class_counts = combined_df['label'].value_counts()
        min_class_count = class_counts.min()
        
        if verbose >= 1:
            print(f"[info] Balancing classes to {min_class_count} samples per class")
            print(f"[info] Original class distribution: {class_counts.to_dict()}")
        
        balanced_dfs = []
        for label, count in class_counts.items():
            class_df = combined_df[combined_df['label'] == label]
            if count > min_class_count:
                class_df = class_df.sample(n=min_class_count, random_state=42)
            balanced_dfs.append(class_df)
        
        combined_df = pd.concat(balanced_dfs, axis=0)
    
    # Extract labels and drop unwanted columns
    y = combined_df['label']
    X = combined_df.drop(['label'] + [col for col in drop_columns if col in combined_df.columns], 
                         axis=1, errors='ignore')
    
    if verbose >= 1:
        print(f"[info] Feature matrix shape: {X.shape}")
        print(f"[info] Final class distribution: {y.value_counts().to_dict()}")
    
    return X, y
