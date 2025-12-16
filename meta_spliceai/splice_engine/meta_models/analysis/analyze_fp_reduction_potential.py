"""
FP Reduction Analysis

This script demonstrates the potential of using context-aware probability features
to filter out False Positive splice sites incorrectly predicted by the base SpliceAI model.

It analyzes the patterns in FP sites compared to TP sites and identifies 
which FPs could potentially be filtered out based on their probability feature patterns.
"""

import os
import pandas as pd
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
import argparse

# Import shared analysis utilities
from meta_spliceai.splice_engine.meta_models.analysis.shared_analysis_utils import (
    analyze_feature_distributions as shared_analyze_feature_distributions
)

# Local imports
from meta_spliceai.splice_engine.meta_models.io.handlers import MetaModelDataHandler
from meta_spliceai.splice_engine.meta_models.core.enhanced_evaluation import is_dataframe_empty

from meta_spliceai.splice_engine.meta_models.utils.workflow_utils import (
    print_emphasized, 
    print_with_indent
)

# Import analysis functions from fp_analysis_utils
from meta_spliceai.splice_engine.meta_models.analysis.fp_analysis_utils import (
    analyze_genes_with_most_fps,
    analyze_genes_with_most_fps_by_type
)

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['figure.dpi'] = 100


def load_positions_data(data_path: Optional[str] = None) -> pl.DataFrame:
    """
    Load the enhanced positions data with all probability features.
    
    Parameters
    ----------
    data_path : str, optional
        Path to the enhanced positions TSV file, by default uses the standard location
        
    Returns
    -------
    pl.DataFrame
        DataFrame containing all positions with probability features
    """
    if data_path is None:
        # Use default path - directly loading from the handler
        data_handler = MetaModelDataHandler()
        positions_df = data_handler.load_splice_positions(
            aggregated=True, 
            enhanced=True,
            convert_to_pandas=False,
            verbose=1
        )
        
        print(f"Loaded positions data using MetaModelDataHandler")
    else:
        # Load from specific path
        print(f"Loading enhanced positions data from: {data_path}")
        positions_df = pl.read_csv(data_path, separator='\t')
    
    print(f"Loaded {positions_df.height} position records with {len(positions_df.columns)} features")
    
    # Extract column names to identify available features
    column_names = positions_df.columns
    print(f"\nAvailable columns in the dataset:")
    for i, col in enumerate(sorted(column_names)):
        print(f"{i+1:3d}. {col}")
    
    # Find probability-related features
    prob_features = [col for col in column_names if 'probability' in col or 'score' in col]
    peak_features = [col for col in column_names if 'peak' in col]
    context_features = [col for col in column_names if 'context' in col]
    
    print(f"\nFound {len(prob_features)} probability features")
    print(f"Found {len(peak_features)} peak-related features")
    print(f"Found {len(context_features)} context features")
    
    return positions_df


def load_full_positions_data(verbose: int = 1) -> pl.DataFrame:
    """
    Load the full splice positions dataset across all genes.
    
    This function loads the complete aggregated dataset of all splice positions,
    which contains information for TPs, TNs, FPs, and FNs across all evaluated genes.
    It first tries to load the enhanced version with context features, and if that
    doesn't contain a full gene set, falls back to the basic positions dataset.
    
    Parameters
    ----------
    verbose : int, optional
        Verbosity level, by default 1
        
    Returns
    -------
    pl.DataFrame
        DataFrame containing all splice positions across all genes
    """
    from meta_spliceai.splice_engine.model_evaluator import ModelEvaluationFileHandler
    from meta_spliceai.splice_engine.meta_models.workflows.data_preparation import prepare_splice_site_annotations
    
    # Set minimum number of genes to consider a "full" dataset
    MIN_GENES_THRESHOLD = 15000
    enhanced_positions_df = None
    fallback_positions_df = None
    
    # First try the standard MetaModelDataHandler approach for enhanced data
    data_handler = MetaModelDataHandler()
    try:
        # Attempt to load using enhanced=True to get context features
        enhanced_positions_df = data_handler.load_splice_positions(
            aggregated=True, 
            enhanced=True,
            convert_to_pandas=False,
            verbose=verbose
        )
        
        # Check if we have a full set of genes
        unique_genes = enhanced_positions_df.select(pl.col("gene_id")).unique().height
        
        if verbose > 0:
            print(f"Loaded enhanced positions dataset with {enhanced_positions_df.height} records")
            print(f"Found {unique_genes} unique genes in enhanced dataset")
        
        # If we have enough genes, use this dataset
        if unique_genes >= MIN_GENES_THRESHOLD:
            if verbose > 0:
                print(f"Using enhanced positions dataset with {len(enhanced_positions_df.columns)} features")
            return enhanced_positions_df
        else:
            if verbose > 0:
                print_with_indent(f"Enhanced dataset only contains {unique_genes} genes, which is below the threshold of {MIN_GENES_THRESHOLD}", indent_level=1)
                print_with_indent("Will attempt to load the full non-enhanced dataset as fallback", indent_level=1)
    except Exception as e:
        if verbose > 0:
            print(f"Could not load enhanced dataset via MetaModelDataHandler: {e}")
    
    # If we couldn't load enhanced data or it doesn't have enough genes, try the direct file approach
    # at the shared evaluation directory level
    # NOTE: This is effectively the same as using the base data from the evaluation directory
    #       that contains base datasets shared across various analysis modules
    #       This is also equivalent to using: 
    #           data_handler = ModelEvaluationFileHandler(eval_dir, separator='\t')
    #           data_handler.load_splice_positions(...)
    try:
        # Try using the same data_handler but with use_shared_dir=True to access shared evaluation data
        if verbose > 0:
            print_with_indent("Attempting to load non-enhanced positions from shared evaluation directory...", indent_level=1)
        
        # Use the same data_handler but look in shared evaluation directory
        fallback_positions_df = data_handler.load_splice_positions(
            aggregated=True, 
            enhanced=False,
            convert_to_pandas=False,
            use_shared_dir=True,  # Look in shared evaluation directory instead of subject-specific meta_models dir
            verbose=verbose
        )
        
        # Check gene count
        unique_genes = fallback_positions_df.select(pl.col("gene_id")).unique().height
        
        if verbose > 0:
            print_with_indent(f"Loaded non-enhanced positions dataset with {fallback_positions_df.height} records", indent_level=1)
            print_with_indent(f"Found {unique_genes} unique genes in non-enhanced dataset", indent_level=1)
        
        if unique_genes >= MIN_GENES_THRESHOLD:
            if verbose > 0:
                print_with_indent(f"Using non-enhanced positions dataset with {len(fallback_positions_df.columns)} features", indent_level=1)
            
            # Print distribution of prediction types
            try:
                pred_counts = fallback_positions_df.select(pl.col("pred_type")).to_pandas().value_counts()
                print("\nPrediction type distribution across all genes:")
                print(pred_counts)
            except:
                print("Could not analyze prediction type distribution")
                
            return fallback_positions_df
        else:
            print_with_indent(f"Warning: Non-enhanced dataset only contains {unique_genes} genes, which is below the threshold", indent_level=1)
    except Exception as e:
        # If we get here, we've exhausted all options
        if enhanced_positions_df is not None:
            print(f"Warning: Using incomplete enhanced dataset as fallback. Error loading full dataset: {e}")
            return enhanced_positions_df
        elif fallback_positions_df is not None:
            print(f"Warning: Using incomplete non-enhanced dataset as fallback. Error loading full dataset: {e}")
            return fallback_positions_df
        else:
            raise RuntimeError(f"Failed to load any positions dataset after trying all fallback methods: {e}")
    
    # Return the fallback dataset from direct file loading
    if fallback_positions_df is not None:
        # Print distribution of prediction types
        try:
            pred_counts = fallback_positions_df.select(pl.col("pred_type")).to_pandas().value_counts()
            print("\nPrediction type distribution across all genes:")
            print(pred_counts)
        except:
            print("Could not analyze prediction type distribution")
            
        return fallback_positions_df
    else:
        # Should never get here, but just in case
        raise FileNotFoundError("Failed to load any splice positions dataset")


def analyze_feature_distributions(positions_df: pl.DataFrame, output_dir: str):
    """
    Analyze and visualize feature distributions for different prediction types,
    focusing on comparing FPs to TPs.
    
    This is a wrapper around the shared implementation that focuses on FP vs TP comparison.
    
    Parameters
    ----------
    positions_df : pl.DataFrame
        DataFrame containing all positions with prediction types
    output_dir : str
        Directory to save output visualizations
    """
    # Call the shared implementation with FP/TP comparison types
    return shared_analyze_feature_distributions(positions_df, output_dir, comparison_types=('FP', 'TP'))


def identify_fp_filter_rules(positions_df: pl.DataFrame, splice_type: str = 'donor') -> Dict:
    """
    Identify rules that can filter out FPs based on context features
    
    Parameters
    ----------
    positions_df : pl.DataFrame
        DataFrame with all position records
    splice_type : str, optional
        Type of splice site to analyze ('donor' or 'acceptor'), by default 'donor'
    
    Returns
    -------
    Dict
        Dictionary of rules and filtered FP counts
    """
    # Filter to get only the FPs of the specified type
    type_col = 'splice_type'  # Using splice_type as the column name is available
    fp_df = positions_df.filter(
        (pl.col('pred_type') == 'FP') 
        # (pl.col(type_col) == splice_type)  # FP is not a splice site
    ).to_pandas()
    
    # If no FPs found, return empty dict
    if fp_df.empty:
        print(f"No {splice_type} FPs found in dataset")
        return {}
    
    # Filter to get only TPs for the same type (for threshold comparison)
    tp_df = positions_df.filter(
        (pl.col('pred_type') == 'TP') &
        (pl.col(type_col) == splice_type)
    ).to_pandas()
    
    # Construct feature names based on available columns
    prob_feature = f"{splice_type}_score"  # Primary probability score
    peak_feature = f"{splice_type}_is_local_peak"
    height_ratio_feature = f"{splice_type}_peak_height_ratio"
    signal_feature = f"{splice_type}_signal_strength"
    context_diff_ratio = f"{splice_type}_context_diff_ratio"  # Context diff ratio
    diff_m1_feature = f"{splice_type}_diff_m1"  # Differential feature
    diff_p1_feature = f"{splice_type}_diff_p1"  # Differential feature
    
    # Context-agnostic features to consider
    context_neighbor_mean = "context_neighbor_mean"
    context_max = "context_max"
    
    # Cross-type features if both donor and acceptor are being analyzed
    if splice_type == 'donor':
        cross_feature = "score_difference_ratio"  # Added cross-type feature
    else:
        cross_feature = "score_difference_ratio"  # Same feature but different threshold
    
    # List of columns that should NOT be converted to numeric
    non_numeric_columns = ['chrom', 'gene_id', 'transcript_id', 'strand', 
                           'splice_type', 'type', 'pred_type', 'error_type', 'sequence']
    
    # Treat every column that is not explicitly non-numeric as numeric candidate
    candidate_numeric_cols = [col for col in fp_df.columns if col not in non_numeric_columns]

    for feature in candidate_numeric_cols:
        if feature in fp_df.columns and fp_df[feature].dtype == 'object':
            try:
                fp_df[feature] = pd.to_numeric(fp_df[feature], errors='coerce')
            except Exception:
                pass  # silently ignore – non-numeric strings will become NaN
        if feature in tp_df.columns and tp_df[feature].dtype == 'object':
            try:
                tp_df[feature] = pd.to_numeric(tp_df[feature], errors='coerce')
            except Exception:
                pass

    # Check if features exist in the dataset
    feature_list = [prob_feature, peak_feature, height_ratio_feature, signal_feature, diff_m1_feature, diff_p1_feature,
                   context_diff_ratio, context_neighbor_mean, context_max, cross_feature]
    
    if not all(f in fp_df.columns for f in feature_list if f is not None):
        missing_features = [f for f in feature_list if f is not None and f not in fp_df.columns]
        print(f"Warning: Some required {splice_type} features are missing in the dataset: {missing_features}")
    
    # Initialize filter rules dictionary
    filter_rules = {}
    
    # Calculate distribution-based thresholds
    try:
        # Use stricter (20th / 80th percentile) thresholds compared with FN rescue
        quantile_cut = 0.20
        
        # For each feature, we check if it exists in both dataframes
        # Using the 20th percentile of TPs as threshold for features where higher values are better
        # Using the 80th percentile of TPs as threshold for features where lower values are better
        
        if prob_feature in tp_df.columns and prob_feature in fp_df.columns:
            prob_threshold = tp_df[prob_feature].quantile(quantile_cut)
        else:
            prob_threshold = fp_df[prob_feature].quantile(quantile_cut) if prob_feature in fp_df.columns else None
        
        if height_ratio_feature in tp_df.columns and height_ratio_feature in fp_df.columns:
            height_ratio_threshold = tp_df[height_ratio_feature].quantile(quantile_cut)
        else:
            height_ratio_threshold = 1.5  # Default
            
        if signal_feature in tp_df.columns and signal_feature in fp_df.columns:
            signal_threshold = tp_df[signal_feature].quantile(quantile_cut)
        else:
            signal_threshold = 0.3  # Default
            
        if diff_m1_feature in tp_df.columns and diff_m1_feature in fp_df.columns:
            diff_m1_threshold = tp_df[diff_m1_feature].quantile(quantile_cut)
        else:
            diff_m1_threshold = 0.15  # Default
            
        if diff_p1_feature in tp_df.columns and diff_p1_feature in fp_df.columns:
            diff_p1_threshold = tp_df[diff_p1_feature].quantile(quantile_cut)
        else:
            diff_p1_threshold = 0.15  # Default
            
        if context_diff_ratio in tp_df.columns and context_diff_ratio in fp_df.columns:
            context_diff_threshold = tp_df[context_diff_ratio].quantile(quantile_cut)
        else:
            context_diff_threshold = 1.1  # Default
            
        # For cross-type features, the logic depends on splice type
        if cross_feature in tp_df.columns and cross_feature in fp_df.columns:
            if splice_type == 'donor':
                # For donor sites, the score_difference_ratio should be positive
                cross_threshold = tp_df[cross_feature].quantile(quantile_cut)
            else:
                # For acceptor sites, the score_difference_ratio should be negative
                cross_threshold = tp_df[cross_feature].quantile(1 - quantile_cut)
        else:
            cross_threshold = 0.0  # Default (neutral)
            
        # Print thresholds for debugging
        if prob_threshold is not None:
            print(f"Probability: < {prob_threshold}")
        print(f"Height Ratio: < {height_ratio_threshold}")
        print(f"Signal Strength: < {signal_threshold}")
        print(f"Diff -1: < {diff_m1_threshold}")
        print(f"Diff +1: < {diff_p1_threshold}")
        print(f"Context Diff Ratio: < {context_diff_threshold}")
        
        if splice_type == 'donor':
            print(f"Cross-type: < {cross_threshold} (should be positive for true donors)")
        else:
            print(f"Cross-type: > {cross_threshold} (should be negative for true acceptors)")
            
    except Exception as e:
        print(f"Error calculating thresholds: {e}")
        # Set default thresholds as fallback
        prob_threshold = None
        height_ratio_threshold = 1.5
        signal_threshold = 0.3
        diff_m1_threshold = 0.15
        diff_p1_threshold = 0.15
        context_diff_threshold = 1.1
        cross_threshold = 0.0
    
    # Apply filter rules and count filtered FPs
    # Rule 0: Low probability (model not confident)
    if prob_feature in fp_df.columns and prob_threshold is not None:
        try:
            low_prob_mask = fp_df[prob_feature] < prob_threshold
            
            filter_rules['low_probability'] = {
                'count': low_prob_mask.sum(),
                'percentage': (low_prob_mask.sum() / len(fp_df)) * 100 if len(fp_df) > 0 else 0,
                'threshold': prob_threshold,
                'feature': prob_feature,
                'examples': fp_df[low_prob_mask].head(5) if low_prob_mask.sum() > 0 else None
            }
        except Exception as e:
            print(f"Error applying probability rule: {e}")
    
    # Rule 1: Low height ratio (not standing out from background)
    if height_ratio_feature in fp_df.columns:
        try:
            low_height_ratio_mask = fp_df[height_ratio_feature] < height_ratio_threshold
            
            filter_rules['low_height_ratio'] = {
                'count': low_height_ratio_mask.sum(),
                'percentage': (low_height_ratio_mask.sum() / len(fp_df)) * 100 if len(fp_df) > 0 else 0,
                'threshold': height_ratio_threshold,
                'feature': height_ratio_feature,
                'examples': fp_df[low_height_ratio_mask].head(5) if low_height_ratio_mask.sum() > 0 else None
            }
        except Exception as e:
            print(f"Error applying height ratio rule: {e}")
    
    # Rule 2: Not a local peak
    if peak_feature in fp_df.columns:
        try:
            not_peak_mask = fp_df[peak_feature] == False
            
            filter_rules['not_local_peak'] = {
                'count': not_peak_mask.sum(),
                'percentage': (not_peak_mask.sum() / len(fp_df)) * 100 if len(fp_df) > 0 else 0,
                'feature': peak_feature,
                'examples': fp_df[not_peak_mask].head(5) if not_peak_mask.sum() > 0 else None
            }
        except Exception as e:
            print(f"Error applying local peak rule: {e}")
    
    # Rule 3: Weak signal strength
    if signal_feature in fp_df.columns:
        try:
            weak_signal_mask = fp_df[signal_feature] < signal_threshold
            
            filter_rules['weak_signal'] = {
                'count': weak_signal_mask.sum(),
                'percentage': (weak_signal_mask.sum() / len(fp_df)) * 100 if len(fp_df) > 0 else 0,
                'threshold': signal_threshold,
                'feature': signal_feature,
                'examples': fp_df[weak_signal_mask].head(5) if weak_signal_mask.sum() > 0 else None
            }
        except Exception as e:
            print(f"Error applying signal strength rule: {e}")
            
    # Rule 4: Poor peak shape (weak differential features)
    if diff_m1_feature in fp_df.columns and diff_p1_feature in fp_df.columns:
        try:
            # Either weak rise before site OR weak fall after site indicates a poor peak
            weak_diff_m1_mask = fp_df[diff_m1_feature] < diff_m1_threshold
            weak_diff_p1_mask = fp_df[diff_p1_feature] < diff_p1_threshold
            
            poor_peak_shape_mask = weak_diff_m1_mask | weak_diff_p1_mask
            
            filter_rules['poor_peak_shape'] = {
                'count': poor_peak_shape_mask.sum(),
                'percentage': (poor_peak_shape_mask.sum() / len(fp_df)) * 100 if len(fp_df) > 0 else 0,
                'threshold_m1': diff_m1_threshold,
                'threshold_p1': diff_p1_threshold,
                'feature_m1': diff_m1_feature,
                'feature_p1': diff_p1_feature,
                'examples': fp_df[poor_peak_shape_mask].head(5) if poor_peak_shape_mask.sum() > 0 else None
            }
        except Exception as e:
            print(f"Error applying peak shape rule: {e}")
    
    # Rule 5: Low context diff ratio (center not standing out from neighbors)
    if context_diff_ratio in fp_df.columns:
        try:
            low_context_diff_mask = fp_df[context_diff_ratio] < context_diff_threshold
            
            filter_rules['low_context_diff'] = {
                'count': low_context_diff_mask.sum(),
                'percentage': (low_context_diff_mask.sum() / len(fp_df)) * 100 if len(fp_df) > 0 else 0,
                'threshold': context_diff_threshold,
                'feature': context_diff_ratio,
                'examples': fp_df[low_context_diff_mask].head(5) if low_context_diff_mask.sum() > 0 else None
            }
        except Exception as e:
            print(f"Error applying context diff ratio rule: {e}")
            
    # Rule 6: Contradictory cross-type evidence
    if cross_feature in fp_df.columns:
        try:
            if splice_type == 'donor':
                # For donor sites, a low or negative score_difference_ratio contradicts donor identity
                contradictory_cross_mask = fp_df[cross_feature] < cross_threshold
            else:
                # For acceptor sites, a high or positive score_difference_ratio contradicts acceptor identity
                contradictory_cross_mask = fp_df[cross_feature] > cross_threshold
            
            filter_rules['contradictory_cross_type'] = {
                'count': contradictory_cross_mask.sum(),
                'percentage': (contradictory_cross_mask.sum() / len(fp_df)) * 100 if len(fp_df) > 0 else 0,
                'threshold': cross_threshold,
                'feature': cross_feature,
                'examples': fp_df[contradictory_cross_mask].head(5) if contradictory_cross_mask.sum() > 0 else None
            }
        except Exception as e:
            print(f"Error applying cross-type rule: {e}")
            
    # Rule 7: Combined rules (any of the above)
    try:
        combined_mask = pd.Series([False] * len(fp_df))
        
        # Apply all rules we've defined
        for rule_name, rule in filter_rules.items():
            if rule_name == 'combined' or rule_name == 'rule_importance':
                continue  # Skip combined/summary rules
                
            # Handle different rule types based on their structure
            if 'feature' in rule and 'threshold' in rule:
                # Standard threshold rules - different logic depending on the rule
                if rule_name in ['low_height_ratio', 'weak_signal', 'low_context_diff', 'low_probability']:
                    # For these rules, we want values BELOW threshold
                    rule_mask = fp_df[rule['feature']] < rule['threshold']
                elif rule_name == 'contradictory_cross_type':
                    # Cross-type logic depends on splice type
                    if splice_type == 'donor':
                        rule_mask = fp_df[rule['feature']] < rule['threshold']
                    else:
                        rule_mask = fp_df[rule['feature']] > rule['threshold']
                else:
                    # Default case
                    rule_mask = fp_df[rule['feature']] < rule['threshold']
                    
                combined_mask = combined_mask | rule_mask
                
            elif 'feature' in rule and 'threshold' not in rule:
                # Boolean feature
                if rule_name == 'not_local_peak':
                    rule_mask = fp_df[rule['feature']] == False
                    combined_mask = combined_mask | rule_mask
                    
            elif 'feature_m1' in rule and 'feature_p1' in rule:
                # Dual feature rule
                rule_mask = (fp_df[rule['feature_m1']] < rule['threshold_m1']) | \
                            (fp_df[rule['feature_p1']] < rule['threshold_p1'])
                combined_mask = combined_mask | rule_mask
        
        filter_rules['combined'] = {
            'count': combined_mask.sum(),
            'percentage': (combined_mask.sum() / len(fp_df)) * 100 if len(fp_df) > 0 else 0,
            'examples': fp_df[combined_mask].head(5) if combined_mask.sum() > 0 else None
        }
        
        # Add rule importance analysis
        if combined_mask.sum() > 0:
            print(f"\nRule importance analysis for {splice_type} sites:")
            # For each FP filtered by combined rules, which individual rules caught it?
            filtered_fp_df = fp_df[combined_mask]
            rule_overlap = {}
            
            for rule_name, rule in filter_rules.items():
                if rule_name == 'combined' or rule_name == 'rule_importance':
                    continue
                    
                # Count how many of the filtered FPs each rule catches
                rule_count = 0
                
                if 'feature' in rule and 'threshold' in rule:
                    if rule_name in ['low_height_ratio', 'weak_signal', 'low_context_diff', 'low_probability']:
                        rule_count = (filtered_fp_df[rule['feature']] < rule['threshold']).sum()
                    elif rule_name == 'contradictory_cross_type':
                        if splice_type == 'donor':
                            rule_count = (filtered_fp_df[rule['feature']] < rule['threshold']).sum()
                        else:
                            rule_count = (filtered_fp_df[rule['feature']] > rule['threshold']).sum()
                    else:
                        rule_count = (filtered_fp_df[rule['feature']] < rule['threshold']).sum()
                        
                elif 'feature' in rule and 'threshold' not in rule:
                    if rule_name == 'not_local_peak':
                        rule_count = (filtered_fp_df[rule['feature']] == False).sum()
                        
                elif 'feature_m1' in rule and 'feature_p1' in rule:
                    rule_count = ((filtered_fp_df[rule['feature_m1']] < rule['threshold_m1']) | \
                                  (filtered_fp_df[rule['feature_p1']] < rule['threshold_p1'])).sum()
                
                rule_overlap[rule_name] = {
                    'count': rule_count,
                    'percentage': (rule_count / len(filtered_fp_df)) * 100 if len(filtered_fp_df) > 0 else 0
                }
                print(f"  - {rule_name}: Filters {rule_count} ({rule_overlap[rule_name]['percentage']:.1f}%) of combined filtered FPs")
                
            filter_rules['rule_importance'] = rule_overlap

        unfiltered_fps = fp_df[~combined_mask]
        print(f"Number of FPs not matched by any rule: {len(unfiltered_fps)}")
        if len(unfiltered_fps) > 0:
            print("Sample unfiltered FPs:")
            print(unfiltered_fps[['position', 'gene_id', prob_feature]].head())
    except Exception as e:
        print(f"Error applying combined rules: {e}")
        import traceback
        traceback.print_exc()
    
    return filter_rules


def analyze_fp_reduction_potential_by_gene(positions_df: pl.DataFrame, gene_ids: List[str], 
                                       output_dir: str) -> pd.DataFrame:
    """
    Analyze FP reduction potential for a list of specific genes.
    
    Parameters
    ----------
    positions_df : pl.DataFrame
        DataFrame containing all positions with prediction types
    gene_ids : List[str]
        List of gene IDs to analyze
    output_dir : str
        Directory to save output visualizations and statistics
        
    Returns
    -------
    pd.DataFrame
        DataFrame with gene-level FP reduction statistics
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize summary data
    summary_data = {
        "gene_id": [],
        "total_fps": [],
        "donor_fps": [],
        "acceptor_fps": [],
        "donor_filtered": [],
        "acceptor_filtered": [],
        "total_filtered": [],
        "reduction_percentage": []
    }

    # Ensure splice_type is populated for FP rows …
    if 'splice_type' in positions_df.columns and 'donor_score' in positions_df.columns and 'acceptor_score' in positions_df.columns:
        positions_df = positions_df.with_columns(
            pl.when(pl.col('splice_type').is_null())
            .then(
                pl.when(pl.col('donor_score') > pl.col('acceptor_score'))
                    .then(pl.lit('donor'))
                    .otherwise(pl.lit('acceptor'))
            )
            .otherwise(pl.col('splice_type'))
            .alias('splice_type')
        )
    
    # Analyze each gene
    for gene_id in gene_ids:
        print_emphasized(f"\nAnalyzing FP reduction potential for gene: {gene_id}")
        
        # Filter to the specific gene
        gene_positions = positions_df.filter(pl.col('gene_id') == gene_id)
        
        if is_dataframe_empty(gene_positions):
            print(f"No data found for gene {gene_id}, skipping...")
            continue
        
        # Create output directory for this gene
        gene_dir = os.path.join(output_dir, f"gene_{gene_id}")
        os.makedirs(gene_dir, exist_ok=True)
        
        # Get FP counts by splice type
        fp_counts = gene_positions.filter(pl.col('pred_type') == 'FP')
        
        total_fps = fp_counts.shape[0]
        donor_fps = fp_counts.filter(pl.col('splice_type') == 'donor').shape[0]
        acceptor_fps = fp_counts.filter(pl.col('splice_type') == 'acceptor').shape[0]
        
        print(f"Total FPs: {total_fps} ({donor_fps} donor, {acceptor_fps} acceptor)")
        
        if total_fps == 0:
            print(f"No FPs found for gene {gene_id}, skipping...")
            continue
        
        # Analyze feature distributions for this gene
        feature_dir = os.path.join(gene_dir, "features")
        analyze_feature_distributions(gene_positions, feature_dir)
        
        # Identify filter rules for donor and acceptor sites
        print("\nIdentifying FP filter rules for donor sites...")
        donor_rules = identify_fp_filter_rules(gene_positions, splice_type='donor')
        
        print("\nIdentifying FP filter rules for acceptor sites...")
        acceptor_rules = identify_fp_filter_rules(gene_positions, splice_type='acceptor')
        
        # Extract counts from rules
        donor_filtered = donor_rules.get('combined', {}).get('count', 0)
        acceptor_filtered = acceptor_rules.get('combined', {}).get('count', 0)
        
        # Calculate total reduction (properly handling donor and acceptor FPs)
        # Each FP is either a donor or acceptor, so we need to ensure we don't exceed the total FP count
        total_filtered = min(donor_filtered + acceptor_filtered, total_fps)
        reduction_pct = (total_filtered / total_fps) * 100 if total_fps > 0 else 0
        
        # Save summary for this gene
        summary_data["gene_id"].append(gene_id)
        summary_data["total_fps"].append(total_fps)
        summary_data["donor_fps"].append(donor_fps)
        summary_data["acceptor_fps"].append(acceptor_fps)
        summary_data["donor_filtered"].append(donor_filtered)
        summary_data["acceptor_filtered"].append(acceptor_filtered)
        summary_data["total_filtered"].append(total_filtered)
        summary_data["reduction_percentage"].append(reduction_pct)
        
        # Save detailed rules to file
        with open(os.path.join(gene_dir, "fp_filter_summary.txt"), "w") as f:
            f.write(f"=== FP Reduction Analysis for Gene {gene_id} ===\n")
            f.write(f"Total FPs: {total_fps} ({donor_fps} donor, {acceptor_fps} acceptor)\n")
            f.write(f"Potentially filtered: {total_filtered} ({reduction_pct:.1f}%)\n\n")
            
            f.write("Donor Rules:\n")
            for rule_name, rule in donor_rules.items():
                if rule_name not in ['combined', 'rule_importance']:
                    f.write(f"  Rule: {rule_name}\n")
                    f.write(f"    - Could filter {rule.get('count', 0)} FPs ({rule.get('percentage', 0):.1f}%)\n")
                    if 'threshold' in rule:
                        f.write(f"    - Threshold: {rule['threshold']}\n")
                    if 'feature' in rule:
                        f.write(f"    - Feature: {rule['feature']}\n")
                    if 'feature_m1' in rule and 'feature_p1' in rule:
                        f.write(f"    - Features: {rule['feature_m1']}, {rule['feature_p1']}\n")
                        f.write(f"    - Thresholds: {rule.get('threshold_m1')}, {rule.get('threshold_p1')}\n")
            
            f.write("\nAcceptor Rules:\n")
            for rule_name, rule in acceptor_rules.items():
                if rule_name not in ['combined', 'rule_importance']:
                    f.write(f"  Rule: {rule_name}\n")
                    f.write(f"    - Could filter {rule.get('count', 0)} FPs ({rule.get('percentage', 0):.1f}%)\n")
                    if 'threshold' in rule:
                        f.write(f"    - Threshold: {rule['threshold']}\n")
                    if 'feature' in rule:
                        f.write(f"    - Feature: {rule['feature']}\n")
                    if 'feature_m1' in rule and 'feature_p1' in rule:
                        f.write(f"    - Features: {rule['feature_m1']}, {rule['feature_p1']}\n")
                        f.write(f"    - Thresholds: {rule.get('threshold_m1')}, {rule.get('threshold_p1')}\n")
            
            # Rule importance section for gene-level summary
            if donor_rules.get('rule_importance'):
                f.write("\nDonor Rule Importance:\n")
                for rname, rstat in donor_rules['rule_importance'].items():
                    f.write(f"  {rname}: {rstat['count']} ({rstat['percentage']:.1f}%) of combined filtered FPs\n")
            if acceptor_rules.get('rule_importance'):
                f.write("\nAcceptor Rule Importance:\n")
                for rname, rstat in acceptor_rules['rule_importance'].items():
                    f.write(f"  {rname}: {rstat['count']} ({rstat['percentage']:.1f}%) of combined filtered FPs\n")
    
    # Create summary DataFrame
    summary_df = pd.DataFrame(summary_data)
    
    # Calculate overall effectiveness
    total_fps_all = summary_df["total_fps"].sum()
    # Fix: The sum of filtered FPs across genes may exceed total FPs due to double-counting
    # both across rules and across splice types (donor/acceptor)
    total_filtered_all = summary_df["total_filtered"].sum()
    # Cap at total FP count to prevent percentages over 100%
    total_filtered_all = min(total_filtered_all, total_fps_all)
    overall_reduction_pct = (total_filtered_all / total_fps_all) * 100 if total_fps_all > 0 else 0
    
    print("\n=== Overall FP Reduction Potential Summary ===")
    print(f"Total FPs across analyzed genes: {total_fps_all}")
    print(f"Total potential FP reductions: {total_filtered_all}")
    print(f"Overall reduction percentage: {overall_reduction_pct:.1f}%")
    
    # Save summary to file
    summary_path = os.path.join(output_dir, "multi_gene_filter_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved multi-gene analysis summary to: {summary_path}")
    
    return summary_df


def explore_promising_fp_examples(positions_df: pl.DataFrame, 
                                 filtered_ids: List[Tuple[str, str, int]],
                                 output_dir: str):
    """
    Explore and visualize promising FP examples that could be filtered out.
    
    Parameters
    ----------
    positions_df : pl.DataFrame
        DataFrame containing all positions
    filtered_ids : List[Tuple[str, str, int]]
        List of (gene_id, transcript_id, position) tuples that could be filtered
    output_dir : str
        Directory to save output visualizations
    """
    if not filtered_ids:
        print("No filtered IDs provided for exploration.")
        return
    
    # Convert to pandas for easier analysis
    pdf = positions_df.to_pandas()
    
    # Prepare columns for visualization based on what's available
    context_prefix = 'context_'
    probability_cols = [col for col in pdf.columns if col.startswith(context_prefix)]
    
    num_examples = min(5, len(filtered_ids))
    print(f"\nExploring {num_examples} promising FP examples that could be filtered:")
    
    for i, (gene_id, transcript_id, position) in enumerate(filtered_ids[:num_examples]):
        # Get the example row
        example = pdf[(pdf['gene_id'] == gene_id) & 
                     (pdf['transcript_id'] == transcript_id) & 
                     (pdf['position'] == position)]
        
        if len(example) == 0:
            print(f"Example {gene_id}:{transcript_id}:{position} not found, skipping...")
            continue
        
        example = example.iloc[0]
        
        # Print key information
        print(f"\nExample {i+1}: {gene_id}:{transcript_id} at position {position}")
        print(f"  Splice type: {example['splice_type']}")
        print(f"  Base probabilities: donor={example['donor_score']:.4f}, "
              f"acceptor={example['acceptor_score']:.4f}, "
              f"neither={example['neither_score']:.4f}")
        
        # Display key context-based features
        for feature in ['is_local_peak', 'peak_height_ratio', 'signal_strength', 
                      'surge_ratio', 'second_derivative']:
            donor_feature = f"donor_{feature}"
            acceptor_feature = f"acceptor_{feature}"
            
            if donor_feature in example:
                print(f"  {donor_feature}: {example[donor_feature]}")
            if acceptor_feature in example:
                print(f"  {acceptor_feature}: {example[acceptor_feature]}")
        
        # Plot context probabilities if available
        if any(col.startswith(context_prefix) for col in example.index):
            plt.figure(figsize=(10, 6))
            
            # Determine which splice type to focus on
            splice_type = example['splice_type']
            positions = [-2, -1, 0, 1, 2]
            
            # Get donor context if available
            donor_context = []
            if all(f'context_donor_score_m{i}' in example for i in [1, 2]) and \
               all(f'context_donor_score_p{i}' in example for i in [1, 2]):
                donor_context = [
                    example['context_donor_score_m2'],
                    example['context_donor_score_m1'],
                    example['donor_score'],
                    example['context_donor_score_p1'],
                    example['context_donor_score_p2']
                ]
                
            # Get acceptor context if available
            acceptor_context = []
            if all(f'context_acceptor_score_m{i}' in example for i in [1, 2]) and \
               all(f'context_acceptor_score_p{i}' in example for i in [1, 2]):
                acceptor_context = [
                    example['context_acceptor_score_m2'],
                    example['context_acceptor_score_m1'],
                    example['acceptor_score'],
                    example['context_acceptor_score_p1'],
                    example['context_acceptor_score_p2']
                ]
            
            # Plot available contexts
            if donor_context:
                plt.plot(positions, donor_context, 'b-o', label='Donor Probability')
            if acceptor_context:
                plt.plot(positions, acceptor_context, 'r-o', label='Acceptor Probability')
                
            plt.axvline(x=0, color='gray', linestyle='--', alpha=0.7)
            plt.axhline(y=0.5, color='green', linestyle='--', alpha=0.5, label='0.5 Threshold')
            
            plt.title(f"Probability Context for {splice_type.capitalize()} FP at position {position}")
            plt.xlabel("Relative Position")
            plt.ylabel("Probability")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            plt.savefig(os.path.join(output_dir, f"fp_example_{i+1}_context.png"))
            plt.close()


def evaluate_fp_reduction_potential(positions_df: pl.DataFrame, output_dir: str) -> None:
    """
    Evaluate the potential for reducing FPs using context features.
    
    This function performs a standard FP reduction analysis focused on the provided dataset,
    exploring feature distributions and identifying potential filter rules for both donor
    and acceptor sites.
    
    Parameters
    ----------
    positions_df : pl.DataFrame
        DataFrame containing all positions with probability scores, context scores, and 
        their derived features
    output_dir : str
        Directory to save analysis results and visualizations
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # First explore feature distributions
    print_emphasized("\nAnalyzing feature distributions across all positions...")
    analyze_feature_distributions(positions_df, output_dir)
    
    # Identify potential rules for donor sites
    print_emphasized("\nAnalyzing donor FPs across all genes...")
    donor_rules = identify_fp_filter_rules(positions_df, splice_type='donor')
    
    # Identify potential rules for acceptor sites
    print_emphasized("\nAnalyzing acceptor FPs across all genes...")
    acceptor_rules = identify_fp_filter_rules(positions_df, splice_type='acceptor')
    
    # Print the rules summary
    print("\n=== Rule Summary ===")
    
    if donor_rules:
        print(f"\nDonor Site Rules:")
        for rule_name, rule in donor_rules.items():
            # Skip rule_importance and other metadata entries
            if rule_name in ['combined', 'rule_importance'] or not isinstance(rule, dict) or 'count' not in rule:
                continue
            print(f"  Rule: {rule_name}")
            print(f"    - Could filter {rule.get('count', 0)} FPs ({rule.get('percentage', 0.0):.1f}%)")
            if 'threshold' in rule:
                print(f"    - Threshold: {rule['threshold']}")
            if 'feature' in rule:
                print(f"    - Feature: {rule['feature']}")
            if 'feature_m1' in rule and 'feature_p1' in rule:
                print(f"    - Features: {rule['feature_m1']}, {rule['feature_p1']}")
                print(f"    - Thresholds: {rule.get('threshold_m1')}, {rule.get('threshold_p1')}")
    
    if acceptor_rules:
        print(f"\nAcceptor Site Rules:")
        for rule_name, rule in acceptor_rules.items():
            # Skip rule_importance and other metadata entries
            if rule_name in ['combined', 'rule_importance'] or not isinstance(rule, dict) or 'count' not in rule:
                continue
            print(f"  Rule: {rule_name}")
            print(f"    - Could filter {rule.get('count', 0)} FPs ({rule.get('percentage', 0.0):.1f}%)")
            if 'threshold' in rule:
                print(f"    - Threshold: {rule['threshold']}")
            if 'feature' in rule:
                print(f"    - Feature: {rule['feature']}")
            if 'feature_m1' in rule and 'feature_p1' in rule:
                print(f"    - Features: {rule['feature_m1']}, {rule['feature_p1']}")
                print(f"    - Thresholds: {rule.get('threshold_m1')}, {rule.get('threshold_p1')}")
    
    # Write rule importance breakdowns if available
    if donor_rules.get('rule_importance'):
        print("\nDonor Rule Importance:")
        for rname, rstat in donor_rules['rule_importance'].items():
            print(f"  {rname}: {rstat['count']} ({rstat['percentage']:.1f}%) of combined filtered FPs")
    if acceptor_rules.get('rule_importance'):
        print("\nAcceptor Rule Importance:")
        for rname, rstat in acceptor_rules['rule_importance'].items():
            print(f"  {rname}: {rstat['count']} ({rstat['percentage']:.1f}%) of combined filtered FPs")
    
    # Memory-efficient approach - avoid full pandas conversion
    # Get total FP counts for summary
    total_fp = positions_df.filter(pl.col("pred_type") == "FP").height
    donor_fp = positions_df.filter((pl.col("pred_type") == "FP") & (pl.col("splice_type") == "donor")).height
    acceptor_fp = positions_df.filter((pl.col("pred_type") == "FP") & (pl.col("splice_type") == "acceptor")).height
    
    # Calculate filtered counts from rules directly
    donor_filtered = donor_rules.get('combined', {}).get('count', 0) if donor_rules else 0
    acceptor_filtered = acceptor_rules.get('combined', {}).get('count', 0) if acceptor_rules else 0
    max_filtered = donor_filtered + acceptor_filtered
    
    filtered_fp_examples = []
    max_examples_per_type = 5  # Limit examples to prevent memory issues
    
    # Use the most promising donor rule if available - process in polars as much as possible
    if donor_rules and 'combined' in donor_rules and donor_rules['combined'].get('count', 0) > 0:
        # Get indices of filterable FPs from the rules
        donor_mask = donor_rules['combined'].get('mask_fp', [])
        
        if len(donor_mask) > 0:
            # Take just a small sample of indices to prevent memory issues
            sample_indices = donor_mask[:min(max_examples_per_type, len(donor_mask))]
            
            # Filter the donor FPs
            fp_donors = positions_df.filter(
                (pl.col("pred_type") == "FP") & 
                (pl.col("splice_type") == "donor")
            )
            
            # Only if we have a reasonably small dataset, extract examples
            if fp_donors.height <= 10000:  # Safety limit
                # Convert only this subset to pandas for row access
                donors_pdf = fp_donors.to_pandas()
                if not donors_pdf.empty and len(sample_indices) > 0 and max(sample_indices) < len(donors_pdf):
                    filterable_donors = donors_pdf.iloc[sample_indices]
                    for _, row in filterable_donors.iterrows():
                        filtered_fp_examples.append(
                            (row['gene_id'], row['transcript_id'], row['position'])
                        )
    
    # Use the most promising acceptor rule if available - process in polars as much as possible
    if acceptor_rules and 'combined' in acceptor_rules and acceptor_rules['combined'].get('count', 0) > 0:
        # Get indices of filterable FPs from the rules
        acceptor_mask = acceptor_rules['combined'].get('mask_fp', [])
        
        if len(acceptor_mask) > 0:
            # Take just a small sample of indices to prevent memory issues
            sample_indices = acceptor_mask[:min(max_examples_per_type, len(acceptor_mask))]
            
            # Filter the acceptor FPs
            fp_acceptors = positions_df.filter(
                (pl.col("pred_type") == "FP") & 
                (pl.col("splice_type") == "acceptor")
            )
            
            # Only if we have a reasonably small dataset, extract examples
            if fp_acceptors.height <= 10000:  # Safety limit
                # Convert only this subset to pandas for row access
                acceptors_pdf = fp_acceptors.to_pandas()
                if not acceptors_pdf.empty and len(sample_indices) > 0 and max(sample_indices) < len(acceptors_pdf):
                    filterable_acceptors = acceptors_pdf.iloc[sample_indices]
                    for _, row in filterable_acceptors.iterrows():
                        filtered_fp_examples.append(
                            (row['gene_id'], row['transcript_id'], row['position'])
                        )
    
    # Explore promising examples (only if we found any)
    if filtered_fp_examples:
        explore_promising_fp_examples(positions_df, filtered_fp_examples, output_dir)
    
    # Calculate overall potential improvement
    
    # Remove potential double-counting (simplified approach)
    max_filtered = max(donor_filtered, acceptor_filtered) if donor_filtered > 0 or acceptor_filtered > 0 else 0
    
    print("\n=== FP Reduction Potential Summary ===")
    print(f"Total FPs in dataset: {total_fp}")
    print(f"Estimated filterable donor FPs: {donor_filtered}")
    print(f"Estimated filterable acceptor FPs: {acceptor_filtered}")
    print(f"Total potential FP reduction: {max_filtered} ({(max_filtered/total_fp)*100:.1f}% of all FPs)")
    
    # Write summary to file
    summary_path = os.path.join(output_dir, "fp_reduction_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("=== FP Reduction Potential Summary ===\n")
        f.write(f"Total FPs in dataset: {total_fp}\n")
        f.write(f"Estimated filterable donor FPs: {donor_filtered}\n")
        f.write(f"Estimated filterable acceptor FPs: {acceptor_filtered}\n")
        f.write(f"Total potential FP reduction: {max_filtered} ({(max_filtered/total_fp)*100:.1f}% of all FPs)\n\n")
        
        f.write(f"Donor Site Rules:\n")
        for rule_name, rule in donor_rules.items():
            if rule_name not in ['combined', 'rule_importance']:
                f.write(f"  Rule: {rule_name}\n")
                f.write(f"    - Could filter {rule.get('count', 0)} FPs ({rule.get('percentage', 0):.1f}%)\n")
                if 'threshold' in rule:
                    f.write(f"    - Threshold: {rule['threshold']}\n")
                if 'feature' in rule:
                    f.write(f"    - Feature: {rule['feature']}\n")
                if 'feature_m1' in rule and 'feature_p1' in rule:
                    f.write(f"    - Features: {rule['feature_m1']}, {rule['feature_p1']}\n")
                    f.write(f"    - Thresholds: {rule.get('threshold_m1')}, {rule.get('threshold_p1')}\n")
        
        f.write(f"\nAcceptor Site Rules:\n")
        for rule_name, rule in acceptor_rules.items():
            if rule_name not in ['combined', 'rule_importance']:
                f.write(f"  Rule: {rule_name}\n")
                f.write(f"    - Could filter {rule.get('count', 0)} FPs ({rule.get('percentage', 0):.1f}%)\n")
                if 'threshold' in rule:
                    f.write(f"    - Threshold: {rule['threshold']}\n")
                if 'feature' in rule:
                    f.write(f"    - Feature: {rule['feature']}\n")
                if 'feature_m1' in rule and 'feature_p1' in rule:
                    f.write(f"    - Features: {rule['feature_m1']}, {rule['feature_p1']}\n")
                    f.write(f"    - Thresholds: {rule.get('threshold_m1')}, {rule.get('threshold_p1')}\n")
        
        # Write rule importance breakdowns if available
        if donor_rules.get('rule_importance'):
            f.write("\nDonor Rule Importance:\n")
            for rname, rstat in donor_rules['rule_importance'].items():
                f.write(f"  {rname}: {rstat['count']} ({rstat['percentage']:.1f}%) of combined filtered FPs\n")
        if acceptor_rules.get('rule_importance'):
            f.write("\nAcceptor Rule Importance:\n")
            for rname, rstat in acceptor_rules['rule_importance'].items():
                f.write(f"  {rname}: {rstat['count']} ({rstat['percentage']:.1f}%) of combined filtered FPs\n")


def main():
    """Main entry point for the FP reduction analysis script."""
    print("MetaSpliceAI Context Feature FP Reduction Analysis")
    print("====================================================")
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Analyze FP reduction potential using context features")
    parser.add_argument("--data-path", type=str, help="Path to enhanced splice positions TSV file (optional)")
    parser.add_argument("--output-dir", type=str, default="splice-surveyor/data/ensembl/spliceai_eval/meta_models/fp_reduction_analysis", 
                        help="Directory to save analysis results")
    parser.add_argument("--full-dataset", action="store_true", 
                        help="Analyze the full aggregated dataset across all genes")
    parser.add_argument("--top-genes", type=int, default=10,
                        help="Number of top genes to return")
    parser.add_argument("--gene-ids", type=str, nargs="+",
                        help="Specific gene IDs to analyze (e.g. ENSG00000141510)")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the data - either full dataset or single gene
    if args.full_dataset:
        print("Loading full positions dataset across all genes...")
        positions_df = load_full_positions_data(verbose=1)
        
        # Find genes with most FPs
        top_genes_df = analyze_genes_with_most_fps(positions_df, os.path.join(output_dir, "top_genes"), top_n=args.top_genes)
        top_genes_path = os.path.join(output_dir, "top_fp_genes.csv")
        top_genes_df.to_csv(top_genes_path)
        print(f"Saved top {args.top_genes} genes with most FPs to: {top_genes_path}")
        
        # If specific genes are provided, analyze those
        if args.gene_ids:
            gene_list = args.gene_ids
            print(f"Analyzing {len(gene_list)} specified genes")
        else:
            # Otherwise use the top genes we just found
            gene_list = top_genes_df.index[:args.top_genes].tolist()
            print(f"Analyzing top {len(gene_list)} genes with most FPs")
        
        # Analyze reduction potential across the selected genes
        analyze_fp_reduction_potential_by_gene(positions_df, gene_list, os.path.join(output_dir, "gene_analysis"))
        
    else:
        # Original single-gene analysis
        print("Performing standard FP reduction analysis...")
        positions_df = load_positions_data(args.data_path)
        
        # Print prediction type counts
        print("Prediction type counts:")
        pred_counts = positions_df.select(pl.col("pred_type")).to_pandas().value_counts()
        print(pred_counts)
        
        # Run standard analysis
        evaluate_fp_reduction_potential(positions_df, output_dir)
    
    print(f"\nAnalysis complete. Results saved to {output_dir}")


if __name__ == "__main__":
    main()
