#!/usr/bin/env python
"""
FN Rescue Analysis

This script demonstrates the potential of using context-aware probability features
to rescue False Negative splice sites missed by the base SpliceAI model.

It analyzes the patterns in FN sites compared to TN sites and identifies 
which FNs could potentially be rescued based on their probability feature patterns.
"""

import os
import pandas as pd
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tqdm
from typing import Dict, List, Tuple, Optional, Union

# Import shared analysis utilities
from meta_spliceai.splice_engine.meta_models.analysis.shared_analysis_utils import (
    get_detailed_splice_site_counts,
    analyze_feature_distributions
)
import argparse

# Local imports
from meta_spliceai.splice_engine.meta_models.io.handlers import MetaModelDataHandler
from meta_spliceai.splice_engine.meta_models.core.enhanced_evaluation import is_dataframe_empty

from meta_spliceai.splice_engine.meta_models.utils.workflow_utils import (
    print_emphasized, 
    print_with_indent
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


# def get_detailed_splice_site_counts(positions_df: pl.DataFrame) -> pd.DataFrame:
#     pass
# This function has been moved to shared_analysis_utils.py


def analyze_genes_with_most_fns(positions_df: pl.DataFrame, top_n: int = 10, 
                               use_detailed_counts: bool = False) -> pd.DataFrame:
    """
    Identify genes with the highest number of False Negatives.
    
    Parameters
    ----------
    positions_df : pl.DataFrame
        DataFrame containing all splice positions
    top_n : int, optional
        Number of top genes to return, by default 10
    use_detailed_counts : bool, optional
        Whether to use detailed counts for additional metrics, by default False
        
    Returns
    -------
    pd.DataFrame
        DataFrame with gene_id and FN counts, sorted by count

    Memo
    ----
    The total_positions calculation counts all positions for each gene in the dataset, which includes:

    - All donor sites from all transcripts/isoforms of the gene
    - All acceptor sites from all transcripts/isoforms of the gene
    - All prediction categories (TPs, TNs, FPs, and FNs)
    """
    if use_detailed_counts:
        # Get detailed counts with all metrics
        counts_df = get_detailed_splice_site_counts(positions_df)
        
        # Sort by FN count and take top N
        result = counts_df.sort_values("fn_count", ascending=False).head(top_n)
    else:
        # Use simpler counting approach
        # Count FNs by gene
        gene_fn_counts = (
            positions_df
            .filter(pl.col("pred_type") == "FN")
            .group_by("gene_id")
            .agg(pl.len().alias("fn_count"))
            .sort("fn_count", descending=True)
            .limit(top_n)
            .to_pandas()
        )
        
        # Get total positions per gene for context
        gene_total_counts = (
            positions_df
            .group_by("gene_id")
            .agg(pl.len().alias("total_positions"))
            .to_pandas()
        )
        
        # Merge to get percentage
        result = gene_fn_counts.merge(gene_total_counts, on="gene_id", how="left")
        result["fn_percentage"] = (result["fn_count"] / result["total_positions"]) * 100
    
    print(f"\nTop {top_n} genes with most False Negatives:")
    print(result)
    
    return result


def analyze_genes_with_most_fns_by_type(
    positions_df: pl.DataFrame, 
    top_n: int = 10, 
    gene_types: Optional[List[str]] = None,
    gene_features_path: Optional[str] = None,
    project_dir: Optional[str] = None,
    use_detailed_counts: bool = False,
    verbose: int = 1
) -> pd.DataFrame:
    """
    Identify genes with the highest number of False Negatives, with optional filtering by gene type.
    
    Parameters
    ----------
    positions_df : pl.DataFrame
        DataFrame containing all splice positions
    top_n : int, optional
        Number of top genes to return, by default 10
    gene_types : Optional[List[str]], optional
        List of gene types to include (e.g., ["protein_coding", "lncRNA"]). 
        If None, all gene types are considered.
    gene_features_path : Optional[str], optional
        Path to the gene features TSV file. If None, will try to find in default location.
    project_dir : Optional[str], optional
        Project directory root, used to find default file locations if paths not provided.
    use_detailed_counts : bool, optional
        Whether to use detailed counts for additional metrics, by default False
    verbose : int, optional
        Verbosity level (0=silent, 1=basic info, 2=detailed), by default 1
        
    Returns
    -------
    pd.DataFrame
        DataFrame with gene_id, gene_type, and FN counts, sorted by count
    """
    # If no gene type filtering is needed, use the original function
    if gene_types is None:
        return analyze_genes_with_most_fns(positions_df, top_n, use_detailed_counts)
        
    # Resolve project directory if needed
    if project_dir is None:
        # Try to infer project directory from system config
        try:
            from meta_spliceai.system.config import Config
            project_dir = Config.PROJ_DIR
        except ImportError:
            # Fallback to using HOME environment variable instead of hardcoded path
            project_dir = os.path.join(os.environ.get('HOME', ''), 'work', 'splice-surveyor')
            if verbose >= 1:
                print(f"[warning] Could not import Config, using fallback project dir: {project_dir}")
    
    # Derive default gene features path if not provided
    if gene_features_path is None:
        gene_features_path = os.path.join(
            project_dir, "data", "ensembl", "spliceai_analysis", "gene_features.tsv"
        )
        if verbose >= 1:
            print(f"[info] Using default gene features path: {gene_features_path}")
    
    # Load gene features
    if verbose >= 1:
        print(f"[i/o] Loading gene features from: {gene_features_path}")
    
    if not os.path.exists(gene_features_path):
        raise FileNotFoundError(f"Gene features file not found: {gene_features_path}")
        
    gene_features_df = pl.read_csv(
        gene_features_path,
        separator='\t',
        schema_overrides={'chrom': pl.Utf8}
    )
    
    # Display gene type distribution if verbose
    if verbose >= 2 and 'gene_type' in gene_features_df.columns:
        gene_type_counts = gene_features_df.group_by('gene_type').agg(
            pl.count('gene_id').alias('count')
        ).sort('count', descending=True)
        print("[info] Gene type distribution:")
        print(gene_type_counts.head(10))
    
    # Filter gene features by requested gene types
    if gene_types:
        filtered_gene_features = gene_features_df.filter(
            pl.col('gene_type').is_in(gene_types)
        )
        filtered_gene_ids = filtered_gene_features.select('gene_id').to_series().to_list()
        
        if verbose >= 1:
            print(f"[info] Filtered to {len(filtered_gene_ids)} genes of types: {', '.join(gene_types)}")
        
        # Filter positions dataframe to only include genes of the requested types
        filtered_positions_df = positions_df.filter(
            pl.col('gene_id').is_in(filtered_gene_ids)
        )
        
        if verbose >= 1:
            original_count = positions_df.height
            filtered_count = filtered_positions_df.height
            print(f"[info] Filtered positions from {original_count:,} to {filtered_count:,} rows ({filtered_count/original_count*100:.1f}%)")
    else:
        filtered_positions_df = positions_df
    
    # Now analyze the filtered dataset
    if use_detailed_counts:
        # Get detailed counts with all metrics
        counts_df = get_detailed_splice_site_counts(filtered_positions_df)
        
        # Sort by FN count and take top N
        initial_result = counts_df.sort_values("fn_count", ascending=False).head(top_n)
    else:
        # Use simpler counting approach
        # Count FNs by gene
        gene_fn_counts = (
            filtered_positions_df
            .filter(pl.col("pred_type") == "FN")
            .group_by("gene_id")
            .agg(pl.len().alias("fn_count"))
            .sort("fn_count", descending=True)
            .limit(top_n)
            .to_pandas()
        )
        
        # Get total positions per gene for context
        gene_total_counts = (
            filtered_positions_df
            .group_by("gene_id")
            .agg(pl.len().alias("total_positions"))
            .to_pandas()
        )
        
        # Merge to get percentage
        initial_result = gene_fn_counts.merge(gene_total_counts, on="gene_id", how="left")
        initial_result["fn_percentage"] = (initial_result["fn_count"] / initial_result["total_positions"]) * 100
    
    # Add gene type information to the result
    gene_types_df = gene_features_df.select(['gene_id', 'gene_type', 'gene_name']).to_pandas()
    result = initial_result.merge(gene_types_df, on="gene_id", how="left")
    
    # Print results
    if gene_types:
        print(f"\nTop {top_n} {', '.join(gene_types)} genes with most False Negatives:")
    else:
        print(f"\nTop {top_n} genes with most False Negatives:")
    print(result)
    
    return result


def analyze_fn_rescue_potential_by_gene(positions_df: pl.DataFrame, gene_ids: List[str], output_dir: str):
    """
    Analyze FN rescue potential for a list of specific genes.
    
    Parameters
    ----------
    positions_df : pl.DataFrame
        DataFrame containing all splice positions
    gene_ids : List[str]
        List of gene_ids to analyze
    output_dir : str
        Directory to save output results
    """
    # Create output directory for per-gene analysis
    gene_output_dir = os.path.join(output_dir, "gene_analysis")
    os.makedirs(gene_output_dir, exist_ok=True)
    
    # Overall summary of rescue potential across all genes
    summary_data = {
        "gene_id": [],
        "total_fns": [],
        "donor_fns": [],
        "acceptor_fns": [],
        "donor_rescued": [],
        "acceptor_rescued": [],
        "total_rescued": [],
        "rescue_percentage": []
    }
    
    # Analyze each gene individually
    for gene_id in gene_ids:
        print(f"\n\n=== Analyzing FN rescue potential for gene: {gene_id} ===")
        
        # Filter to this gene only
        gene_positions = positions_df.filter(pl.col("gene_id") == gene_id)
        
        # Count FNs for this gene
        gene_fn_df = gene_positions.filter(pl.col("pred_type") == "FN")
        gene_tn_df = gene_positions.filter(pl.col("pred_type") == "TN")
        total_fns = gene_fn_df.height
        total_tns = gene_tn_df.height
        
        if total_fns == 0:
            print(f"No FNs found for gene {gene_id}, skipping")
            continue
        
        # Count FNs by splice type
        donor_fns = gene_fn_df.filter(pl.col("splice_type") == "donor").height
        acceptor_fns = gene_fn_df.filter(pl.col("splice_type") == "acceptor").height
        
        print(f"Found {total_fns} FNs: {donor_fns} donor, {acceptor_fns} acceptor")
        print(f"Found {total_tns} TNs")
        
        # Create gene-specific output directory
        gene_dir = os.path.join(gene_output_dir, gene_id)
        os.makedirs(gene_dir, exist_ok=True)
        
        # Evaluate rescue potential
        donor_rules = identify_fn_rescue_rules(gene_positions, splice_type='donor')
        acceptor_rules = identify_fn_rescue_rules(gene_positions, splice_type='acceptor')
        
        # Extract counts from rules
        donor_rescued = donor_rules.get('combined', {}).get('count', 0)
        acceptor_rescued = acceptor_rules.get('combined', {}).get('count', 0)
        
        # Calculate total rescue (taking higher of the two, since some could be double-counted)
        total_rescued = max(donor_rescued, acceptor_rescued)
        rescue_pct = (total_rescued / total_fns) * 100 if total_fns > 0 else 0
        
        # Save summary for this gene
        summary_data["gene_id"].append(gene_id)
        summary_data["total_fns"].append(total_fns)
        summary_data["donor_fns"].append(donor_fns)
        summary_data["acceptor_fns"].append(acceptor_fns)
        summary_data["donor_rescued"].append(donor_rescued)
        summary_data["acceptor_rescued"].append(acceptor_rescued)
        summary_data["total_rescued"].append(total_rescued)
        summary_data["rescue_percentage"].append(rescue_pct)
        
        # Save detailed rules to file
        with open(os.path.join(gene_dir, "fn_rescue_summary.txt"), "w") as f:
            f.write(f"=== FN Rescue Analysis for Gene {gene_id} ===\n")
            f.write(f"Total FNs: {total_fns} ({donor_fns} donor, {acceptor_fns} acceptor)\n")
            f.write(f"Potentially rescued: {total_rescued} ({rescue_pct:.1f}%)\n\n")
            
            f.write("Donor Rules:\n")
            for rule_name, rule in donor_rules.items():
                f.write(f"  Rule: {rule_name}\n")
                f.write(f"    - Could rescue {rule.get('count', 0)} FNs ({rule.get('percentage', 0):.1f}%)\n")
            
            f.write("\nAcceptor Rules:\n")
            for rule_name, rule in acceptor_rules.items():
                f.write(f"  Rule: {rule_name}\n")
                f.write(f"    - Could rescue {rule.get('count', 0)} FNs ({rule.get('percentage', 0):.1f}%)\n")
    
    # Create summary DataFrame
    summary_df = pd.DataFrame(summary_data)
    
    # Calculate overall effectiveness
    total_fns_all = summary_df["total_fns"].sum()
    total_rescued_all = summary_df["total_rescued"].sum()
    overall_rescue_pct = (total_rescued_all / total_fns_all) * 100 if total_fns_all > 0 else 0
    
    print("\n=== Overall FN Rescue Potential Summary ===")
    print(f"Total FNs across analyzed genes: {total_fns_all}")
    print(f"Total potential FN reductions: {total_rescued_all}")
    print(f"Overall rescue percentage: {overall_rescue_pct:.1f}%")
    
    # Save summary to file
    summary_path = os.path.join(output_dir, "multi_gene_rescue_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved multi-gene analysis summary to: {summary_path}")
    
    return summary_df


def analyze_feature_distributions(positions_df: pl.DataFrame, output_dir: str):
    """
    Analyze and visualize feature distributions for different prediction types.
    This is a wrapper around the shared implementation that focuses on FN vs TN comparison.
    
    Parameters
    ----------
    positions_df : pl.DataFrame
        DataFrame containing all positions with prediction types
    output_dir : str
        Directory to save output visualizations
    """
    # Call the shared implementation with FN/TN comparison types
    from meta_spliceai.splice_engine.meta_models.analysis.shared_analysis_utils import analyze_feature_distributions as shared_analyze_feature_distributions
    return shared_analyze_feature_distributions(positions_df, output_dir, comparison_types=('FN', 'TN'))
    
    # Note: The shared implementation includes correlation heatmap functionality
    # and handles all the feature filtering and visualization that was in the original


def identify_fn_rescue_rules(positions_df: pl.DataFrame, splice_type: str = 'donor') -> Dict:
    """
    Identify rules that can rescue FNs based on context features
    
    Parameters
    ----------
    positions_df : pl.DataFrame
        DataFrame with all position records
    splice_type : str, optional
        Type of splice site to analyze ('donor' or 'acceptor'), by default 'donor'
    
    Returns
    -------
    Dict
        Dictionary of rules and rescued FN counts
    """
    # Filter to get only the FNs of the specified type
    type_col = 'splice_type'  # Using splice_type as the column name is available
    fn_df = positions_df.filter(
        (pl.col('pred_type') == 'FN') & 
        (pl.col(type_col) == splice_type)
    ).to_pandas()
    
    # If no FNs found, return empty dict
    if fn_df.empty:
        print(f"No {splice_type} FNs found in dataset")
        return {}
    
    # Filter to get only TNs for the same type (for threshold comparison)
    tn_df = positions_df.filter(
        pl.col('pred_type') == 'TN'
    ).to_pandas()
    
    # Construct feature names based on available columns from the enhanced workflow
    prob_feature = f"{splice_type}_score"  # Primary probability score
    peak_feature = f"{splice_type}_is_local_peak"
    height_ratio_feature = f"{splice_type}_peak_height_ratio"
    signal_feature = f"{splice_type}_signal_strength"
    context_diff_ratio = f"{splice_type}_context_diff_ratio"  # Added new feature
    diff_m1_feature = f"{splice_type}_diff_m1"  # Added differential feature
    diff_p1_feature = f"{splice_type}_diff_p1"  # Added differential feature
    
    # Context-agnostic features to consider
    context_neighbor_mean = "context_neighbor_mean"
    context_max = "context_max"
    
    # Cross-type features if both donor and acceptor are being analyzed
    if splice_type == 'donor':
        alt_prob_feature = "relative_donor_probability"  
        cross_feature = "score_difference_ratio"  # Added new cross-type feature
    else:
        alt_prob_feature = None
        cross_feature = None
    
    # List of columns that should NOT be converted to numeric
    non_numeric_columns = ['chrom', 'gene_id', 'transcript_id', 'strand', 
                          'splice_type', 'type', 'pred_type', 'error_type', 'sequence']
    
    # Convert potential numeric columns to numeric, excluding explicit metadata
    candidate_numeric_cols = [col for col in fn_df.columns if col not in non_numeric_columns]
    
    for feature in candidate_numeric_cols:
        # Convert in FN dataframe
        if feature in fn_df.columns and fn_df[feature].dtype == 'object':
            try:
                fn_df[feature] = pd.to_numeric(fn_df[feature], errors='coerce')
            except Exception:
                pass  # leave non-convertible values as NaN
        
        # Convert in TN dataframe if present
        if feature in tn_df.columns and tn_df[feature].dtype == 'object':
            try:
                tn_df[feature] = pd.to_numeric(tn_df[feature], errors='coerce')
            except Exception:
                pass
    
    # Check if features exist in the dataset
    feature_list = [prob_feature, peak_feature, height_ratio_feature, signal_feature]
    if alt_prob_feature:
        feature_list.append(alt_prob_feature)
    
    if not all(f in fn_df.columns for f in feature_list if f is not None):
        missing_features = [f for f in feature_list if f is not None and f not in fn_df.columns]
        print(f"Warning: Some required {splice_type} features are missing in the dataset: {missing_features}")
    
    # Initialize rescue rules dictionary
    rescue_rules = {}
    
    # Basic threshold approach
    try:
        # Get thresholds based on percentiles of the FN distribution
        # Use alt_prob_feature if available, otherwise use the primary prob_feature
        if alt_prob_feature and alt_prob_feature in fn_df.columns:
            prob_value_threshold = fn_df[alt_prob_feature].quantile(0.75)
        else:
            prob_value_threshold = fn_df[prob_feature].quantile(0.75) if prob_feature in fn_df.columns else None
        
        ratio_threshold = fn_df[height_ratio_feature].quantile(0.75) if height_ratio_feature in fn_df.columns else None
        signal_threshold = fn_df[signal_feature].quantile(0.75) if signal_feature in fn_df.columns else None
        
        # Calculate TN control group thresholds to avoid excessive FPs
        if not tn_df.empty:
            if alt_prob_feature and alt_prob_feature in tn_df.columns:
                tn_prob_value_threshold = tn_df[alt_prob_feature].quantile(0.25)
            else:
                tn_prob_value_threshold = tn_df[prob_feature].quantile(0.25) if prob_feature in tn_df.columns else None
                
            tn_ratio_threshold = tn_df[height_ratio_feature].quantile(0.25) if height_ratio_feature in tn_df.columns else None
            tn_signal_threshold = tn_df[signal_feature].quantile(0.25) if signal_feature in tn_df.columns else None
        else:
            tn_prob_value_threshold = tn_ratio_threshold = tn_signal_threshold = None
            
        # Print the thresholds for debugging
        print(f"\n{splice_type.capitalize()} FN Rescue Thresholds:")
        print(f"Probability: {prob_value_threshold} (TN control: {tn_prob_value_threshold})")
        print(f"Height Ratio: {ratio_threshold} (TN control: {tn_ratio_threshold})")
        print(f"Signal Strength: {signal_threshold} (TN control: {tn_signal_threshold})")
    except Exception as e:
        print(f"Error calculating thresholds: {e}")
        # Provide default thresholds as fallback
        prob_value_threshold = 0.3
        ratio_threshold = 1.5
        signal_threshold = 0.5
        tn_prob_value_threshold = tn_ratio_threshold = tn_signal_threshold = None
    
    # Apply rules and count rescued FNs
    # Rule 1: High probability sites
    prob_feature_to_use = alt_prob_feature if alt_prob_feature and alt_prob_feature in fn_df.columns else prob_feature
    
    if prob_feature_to_use in fn_df.columns:
        try:
            high_prob_mask = fn_df[prob_feature_to_use] > prob_value_threshold
            if tn_prob_value_threshold is not None:
                high_prob_mask = high_prob_mask & (fn_df[prob_feature_to_use] > tn_prob_value_threshold)
            
            rescue_rules['high_probability'] = {
                'count': high_prob_mask.sum(),
                'percentage': (high_prob_mask.sum() / len(fn_df)) * 100 if len(fn_df) > 0 else 0,
                'threshold': prob_value_threshold,
                'feature': prob_feature_to_use,
                'examples': fn_df[high_prob_mask].head(5) if high_prob_mask.sum() > 0 else None
            }
        except Exception as e:
            print(f"Error applying high probability rule: {e}")
    
    # Rule 2: Local peaks
    if peak_feature in fn_df.columns:
        try:
            local_peak_mask = fn_df[peak_feature] == True
            
            rescue_rules['local_peak'] = {
                'count': local_peak_mask.sum(),
                'percentage': (local_peak_mask.sum() / len(fn_df)) * 100 if len(fn_df) > 0 else 0,
                'feature': peak_feature,
                'examples': fn_df[local_peak_mask].head(5) if local_peak_mask.sum() > 0 else None
            }
        except Exception as e:
            print(f"Error applying local peak rule: {e}")
    
    # Rule 3: Good height ratio
    if height_ratio_feature in fn_df.columns:
        try:
            good_ratio_mask = fn_df[height_ratio_feature] > ratio_threshold
            if tn_ratio_threshold is not None:
                good_ratio_mask = good_ratio_mask & (fn_df[height_ratio_feature] > tn_ratio_threshold)
                
            rescue_rules['good_height_ratio'] = {
                'count': good_ratio_mask.sum(),
                'percentage': (good_ratio_mask.sum() / len(fn_df)) * 100 if len(fn_df) > 0 else 0,
                'threshold': ratio_threshold,
                'feature': height_ratio_feature,
                'examples': fn_df[good_ratio_mask].head(5) if good_ratio_mask.sum() > 0 else None
            }
        except Exception as e:
            print(f"Error applying height ratio rule: {e}")
    
    # Rule 4: Strong signal strength
    if signal_feature in fn_df.columns:
        try:
            strong_signal_mask = fn_df[signal_feature] > signal_threshold
            if tn_signal_threshold is not None:
                strong_signal_mask = strong_signal_mask & (fn_df[signal_feature] > tn_signal_threshold)
                
            rescue_rules['strong_signal'] = {
                'count': strong_signal_mask.sum(),
                'percentage': (strong_signal_mask.sum() / len(fn_df)) * 100 if len(fn_df) > 0 else 0,
                'threshold': signal_threshold,
                'feature': signal_feature,
                'examples': fn_df[strong_signal_mask].head(5) if strong_signal_mask.sum() > 0 else None
            }
        except Exception as e:
            print(f"Error applying signal strength rule: {e}")
            
    # NEW Rule 5: Strong differential features (characterizes sharp peak)
    if diff_m1_feature in fn_df.columns and diff_p1_feature in fn_df.columns:
        try:
            # Calculate thresholds for differential features
            diff_m1_threshold = fn_df[diff_m1_feature].quantile(0.75) if not fn_df[diff_m1_feature].isna().all() else 0.2
            diff_p1_threshold = fn_df[diff_p1_feature].quantile(0.75) if not fn_df[diff_p1_feature].isna().all() else 0.2
            
            # Sharp rise before site AND sharp fall after site indicates good peak
            sharp_peak_mask = (fn_df[diff_m1_feature] > diff_m1_threshold) & (fn_df[diff_p1_feature] > diff_p1_threshold)
            
            rescue_rules['sharp_peak_pattern'] = {
                'count': sharp_peak_mask.sum(),
                'percentage': (sharp_peak_mask.sum() / len(fn_df)) * 100 if len(fn_df) > 0 else 0,
                'threshold_m1': diff_m1_threshold,
                'threshold_p1': diff_p1_threshold,
                'feature_m1': diff_m1_feature,
                'feature_p1': diff_p1_feature,
                'examples': fn_df[sharp_peak_mask].head(5) if sharp_peak_mask.sum() > 0 else None
            }
        except Exception as e:
            print(f"Error applying differential features rule: {e}")
    
    # NEW Rule 6: Context-based rules
    if context_diff_ratio in fn_df.columns:
        try:
            # Good context diff ratio indicates the center score is much higher than any neighbor
            context_diff_ratio_threshold = fn_df[context_diff_ratio].quantile(0.75) if not fn_df[context_diff_ratio].isna().all() else 1.2
            
            good_context_diff_mask = fn_df[context_diff_ratio] > context_diff_ratio_threshold
            
            rescue_rules['context_diff_pattern'] = {
                'count': good_context_diff_mask.sum(),
                'percentage': (good_context_diff_mask.sum() / len(fn_df)) * 100 if len(fn_df) > 0 else 0,
                'threshold': context_diff_ratio_threshold,
                'feature': context_diff_ratio,
                'examples': fn_df[good_context_diff_mask].head(5) if good_context_diff_mask.sum() > 0 else None
            }
        except Exception as e:
            print(f"Error applying context diff ratio rule: {e}")
            
    # NEW Rule 7: Cross-type feature for distinguishing between donor and acceptor
    if cross_feature in fn_df.columns:
        try:
            # For donor sites, score_difference_ratio should be positive
            # For acceptor sites, it should be negative
            if splice_type == 'donor':
                cross_threshold = fn_df[cross_feature].quantile(0.75) if not fn_df[cross_feature].isna().all() else 0.1
                cross_mask = fn_df[cross_feature] > cross_threshold
            else:
                cross_threshold = fn_df[cross_feature].quantile(0.25) if not fn_df[cross_feature].isna().all() else -0.1
                cross_mask = fn_df[cross_feature] < cross_threshold
            
            rescue_rules['cross_type_pattern'] = {
                'count': cross_mask.sum(),
                'percentage': (cross_mask.sum() / len(fn_df)) * 100 if len(fn_df) > 0 else 0,
                'threshold': cross_threshold,
                'feature': cross_feature,
                'examples': fn_df[cross_mask].head(5) if cross_mask.sum() > 0 else None
            }
        except Exception as e:
            print(f"Error applying cross-type feature rule: {e}")
    
    # Rule 8: Combined rules (any of the above)
    try:
        combined_mask = pd.Series([False] * len(fn_df))
        
        # Apply all rules we've defined
        for rule_name, rule in rescue_rules.items():
            if rule_name == 'combined':
                continue  # Skip combined rule to avoid circular reference
                
            # Handle different rule types based on their structure
            if 'feature' in rule and 'threshold' in rule:
                # Standard threshold rule
                rule_mask = fn_df[rule['feature']] > rule['threshold']
                combined_mask = combined_mask | rule_mask
                
            elif 'feature' in rule and 'threshold' not in rule:
                # Boolean feature
                if rule['feature'] == peak_feature:
                    rule_mask = fn_df[rule['feature']] == True
                    combined_mask = combined_mask | rule_mask
                    
            elif 'feature_m1' in rule and 'feature_p1' in rule:
                # Dual feature rule
                rule_mask = (fn_df[rule['feature_m1']] > rule['threshold_m1']) & \
                            (fn_df[rule['feature_p1']] > rule['threshold_p1'])
                combined_mask = combined_mask | rule_mask
        
        rescue_rules['combined'] = {
            'count': combined_mask.sum(),
            'percentage': (combined_mask.sum() / len(fn_df)) * 100 if len(fn_df) > 0 else 0,
            'examples': fn_df[combined_mask].head(5) if combined_mask.sum() > 0 else None
        }
        
        # Add rule importance analysis
        if combined_mask.sum() > 0:
            print(f"\nRule importance analysis for {splice_type} sites:")
            # For each FN rescued by combined rules, which individual rules caught it?
            rescued_fn_df = fn_df[combined_mask]
            rule_overlap = {}
            
            for rule_name, rule in rescue_rules.items():
                if rule_name == 'combined':
                    continue
                    
                # Count how many of the rescued FNs each rule catches
                rule_count = 0
                if 'feature' in rule and 'threshold' in rule:
                    rule_count = (rescued_fn_df[rule['feature']] > rule['threshold']).sum()
                elif 'feature' in rule and 'threshold' not in rule:
                    if rule['feature'] == peak_feature:
                        rule_count = (rescued_fn_df[rule['feature']] == True).sum()
                elif 'feature_m1' in rule and 'feature_p1' in rule:
                    rule_count = ((rescued_fn_df[rule['feature_m1']] > rule['threshold_m1']) & \
                                  (rescued_fn_df[rule['feature_p1']] > rule['threshold_p1'])).sum()
                
                rule_overlap[rule_name] = {
                    'count': rule_count,
                    'percentage': (rule_count / len(rescued_fn_df)) * 100 if len(rescued_fn_df) > 0 else 0
                }
                print(f"  - {rule_name}: Rescues {rule_count} ({rule_overlap[rule_name]['percentage']:.1f}%) of combined rescued FNs")
                
            rescue_rules['rule_importance'] = rule_overlap
    except Exception as e:
        print(f"Error applying combined rules: {e}")
        import traceback
        traceback.print_exc()
    
    return rescue_rules


def explore_promising_fn_examples(positions_df: pl.DataFrame, 
                                 rescued_ids: List[Tuple[str, str, int]],
                                 output_dir: str):
    """
    Explore and visualize promising FN examples that could be rescued.
    
    Parameters
    ----------
    positions_df : pl.DataFrame
        DataFrame containing all positions
    rescued_ids : List[Tuple[str, str, int]]
        List of (gene_id, transcript_id, position) tuples that could be rescued
    output_dir : str
        Directory to save output visualizations
    """
    if not rescued_ids:
        print("No rescued IDs provided for exploration.")
        return
    
    # Convert to pandas for easier analysis
    pdf = positions_df.to_pandas()
    
    # Prepare columns for visualization based on what's available
    context_prefix = 'context_'
    probability_cols = [col for col in pdf.columns if col.startswith(context_prefix)]
    
    num_examples = min(5, len(rescued_ids))
    print(f"\nExploring {num_examples} promising FN examples that could be rescued:")
    
    for i, (gene_id, transcript_id, position) in enumerate(rescued_ids[:num_examples]):
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
            
            plt.title(f"Probability Context for {splice_type.capitalize()} FN at position {position}")
            plt.xlabel("Relative Position")
            plt.ylabel("Probability")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            plt.savefig(os.path.join(output_dir, f"fn_example_{i+1}_context.png"))
            plt.close()


def evaluate_fn_rescue_potential(positions_df: pl.DataFrame, output_dir: str):
    """
    Evaluate the potential for rescuing False Negatives using probability features.
    
    Parameters
    ----------
    positions_df : pl.DataFrame
        DataFrame containing all positions with probability features
    output_dir : str
        Directory to save analysis results and visualizations
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # First explore feature distributions
    analyze_feature_distributions(positions_df, output_dir)
    
    # Identify potential rules for donor sites
    donor_rules = identify_fn_rescue_rules(positions_df, splice_type='donor')
    
    # Identify potential rules for acceptor sites
    acceptor_rules = identify_fn_rescue_rules(positions_df, splice_type='acceptor')
    
    # Print the rules summary
    print("\n=== Rule Summary ===")
    
    if donor_rules:
        print(f"\nDonor Site Rules:")
        for rule_name, rule in donor_rules.items():
            # Skip rule_importance and other metadata entries that don't have count/percentage
            if rule_name in ['rule_importance'] or not isinstance(rule, dict) or 'count' not in rule:
                continue
            print(f"  Rule: {rule_name}")
            print(f"    - Could rescue {rule.get('count', 0)} FNs ({rule.get('percentage', 0.0):.1f}%)")
    
    if acceptor_rules:
        print(f"\nAcceptor Site Rules:")
        for rule_name, rule in acceptor_rules.items():
            # Skip rule_importance and other metadata entries that don't have count/percentage
            if rule_name in ['rule_importance'] or not isinstance(rule, dict) or 'count' not in rule:
                continue
            print(f"  Rule: {rule_name}")
            print(f"    - Could rescue {rule.get('count', 0)} FNs ({rule.get('percentage', 0.0):.1f}%)")
            
    # Memory-efficient approach - avoid full pandas conversion
    # Get total FN counts for summary
    total_fn = positions_df.filter(pl.col("pred_type") == "FN").height
    donor_fn = positions_df.filter((pl.col("pred_type") == "FN") & (pl.col("splice_type") == "donor")).height
    acceptor_fn = positions_df.filter((pl.col("pred_type") == "FN") & (pl.col("splice_type") == "acceptor")).height
    
    # Calculate rescued counts from rules directly
    donor_rescued = donor_rules.get('combined', {}).get('count', 0) if donor_rules else 0
    acceptor_rescued = acceptor_rules.get('combined', {}).get('count', 0) if acceptor_rules else 0
    max_rescued = donor_rescued + acceptor_rescued
    
    rescued_fn_examples = []
    max_examples_per_type = 5  # Limit examples to prevent memory issues
    
    # Use the most promising donor rule if available - process in polars as much as possible
    if donor_rules and 'combined' in donor_rules and donor_rules['combined'].get('count', 0) > 0:
        # Get indices of rescuable FNs from the rules
        donor_mask = donor_rules['combined'].get('mask_fn', [])
        
        if len(donor_mask) > 0:
            # Take just a small sample of indices to prevent memory issues
            sample_indices = donor_mask[:min(max_examples_per_type, len(donor_mask))]
            
            # Filter the donor FNs
            fn_donors = positions_df.filter(
                (pl.col("pred_type") == "FN") & 
                (pl.col("splice_type") == "donor")
            )
            
            # Only if we have a reasonably small dataset, extract examples
            if fn_donors.height <= 10000:  # Safety limit
                # Convert only this subset to pandas for row access
                donors_pdf = fn_donors.to_pandas()
                if not donors_pdf.empty and len(sample_indices) > 0 and max(sample_indices) < len(donors_pdf):
                    rescuable_donors = donors_pdf.iloc[sample_indices]
                    for _, row in rescuable_donors.iterrows():
                        rescued_fn_examples.append(
                            (row['gene_id'], row['transcript_id'], row['position'])
                        )
    
    # Use the most promising acceptor rule if available - process in polars as much as possible
    if acceptor_rules and 'combined' in acceptor_rules and acceptor_rules['combined'].get('count', 0) > 0:
        # Get indices of rescuable FNs from the rules
        acceptor_mask = acceptor_rules['combined'].get('mask_fn', [])
        
        if len(acceptor_mask) > 0:
            # Take just a small sample of indices to prevent memory issues
            sample_indices = acceptor_mask[:min(max_examples_per_type, len(acceptor_mask))]
            
            # Filter the acceptor FNs
            fn_acceptors = positions_df.filter(
                (pl.col("pred_type") == "FN") & 
                (pl.col("splice_type") == "acceptor")
            )
            
            # Only if we have a reasonably small dataset, extract examples
            if fn_acceptors.height <= 10000:  # Safety limit
                # Convert only this subset to pandas for row access
                acceptors_pdf = fn_acceptors.to_pandas()
                if not acceptors_pdf.empty and len(sample_indices) > 0 and max(sample_indices) < len(acceptors_pdf):
                    rescuable_acceptors = acceptors_pdf.iloc[sample_indices]
                    for _, row in rescuable_acceptors.iterrows():
                        rescued_fn_examples.append(
                            (row['gene_id'], row['transcript_id'], row['position'])
                        )
    
    # Explore promising examples (only if we found any)
    if rescued_fn_examples:
        explore_promising_fn_examples(positions_df, rescued_fn_examples, output_dir)
    
    # Calculate overall potential improvement
    
    # Remove potential double-counting (simplified approach)
    max_rescued = max(donor_rescued, acceptor_rescued) if donor_rescued > 0 or acceptor_rescued > 0 else 0
    
    print("\n=== FN Rescue Potential Summary ===")
    print(f"Total FNs in dataset: {total_fn}")
    print(f"Estimated rescuable donor FNs: {donor_rescued}")
    print(f"Estimated rescuable acceptor FNs: {acceptor_rescued}")
    print(f"Total potential FN reduction: {max_rescued} ({(max_rescued/total_fn)*100:.1f}% of all FNs)")
    
    # Write summary to file
    summary_path = os.path.join(output_dir, "fn_rescue_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("=== FN Rescue Potential Summary ===\n")
        f.write(f"Total FNs in dataset: {total_fn}\n")
        f.write(f"Estimated rescuable donor FNs: {donor_rescued}\n")
        f.write(f"Estimated rescuable acceptor FNs: {acceptor_rescued}\n")
        f.write(f"Total potential FN reduction: {max_rescued} ({(max_rescued/total_fn)*100:.1f}% of all FNs)\n\n")
        
        f.write(f"Donor Site Rules:\n")
        for rule_name, rule in donor_rules.items():
            f.write(f"  Rule: {rule_name}\n")
            f.write(f"    - Could rescue {rule.get('count', 0)} FNs ({rule.get('percentage', 0):.1f}%)\n")
        
        f.write(f"\nAcceptor Site Rules:\n")
        for rule_name, rule in acceptor_rules.items():
            f.write(f"  Rule: {rule_name}\n")
            f.write(f"    - Could rescue {rule.get('count', 0)} FNs ({rule.get('percentage', 0):.1f}%)\n")


def main():
    """Main entry point for the FN rescue analysis script."""
    print("MetaSpliceAI Context Feature FN Rescue Analysis")
    print("=================================================")
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Analyze FN rescue potential using context features")
    parser.add_argument("--data-path", type=str, help="Path to enhanced splice positions TSV file (optional)")
    parser.add_argument("--output-dir", type=str, default="splice-surveyor/data/ensembl/spliceai_eval/meta_models/fn_rescue_analysis", 
                        help="Directory to save analysis results")
    parser.add_argument("--full-dataset", action="store_true", 
                        help="Analyze the full aggregated dataset across all genes")
    parser.add_argument("--top-genes", type=int, default=10,
                        help="Number of top genes to return")
    parser.add_argument("--gene-ids", type=str, nargs="+",
                        help="Specific gene IDs to analyze (e.g. ENSG00000141510)")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load the data - either full dataset or single gene
    if args.full_dataset:
        print("Loading full positions dataset across all genes...")
        positions_df = load_full_positions_data(verbose=1)
        
        # Find genes with most FNs
        top_genes_df = analyze_genes_with_most_fns(positions_df, top_n=args.top_genes, use_detailed_counts=True)
        top_genes_path = os.path.join(output_dir, "top_fn_genes.csv")
        top_genes_df.to_csv(top_genes_path, index=False)
        print(f"Saved top {args.top_genes} genes with most FNs to: {top_genes_path}")
        
        # If specific genes are provided, analyze those
        if args.gene_ids:
            gene_list = args.gene_ids
            print(f"Analyzing {len(gene_list)} specified genes")
        else:
            # Otherwise use the top genes we just found
            gene_list = top_genes_df["gene_id"].tolist()
            print(f"Analyzing top {len(gene_list)} genes with most FNs")
        
        # Analyze rescue potential across the selected genes
        analyze_fn_rescue_potential_by_gene(positions_df, gene_list, str(output_dir))
        
    else:
        # Original single-gene analysis
        print("Performing single gene FN rescue analysis...")
        positions_df = load_positions_data(args.data_path)
        
        # Print prediction type counts
        print("Prediction type counts:")
        pred_counts = positions_df.select(pl.col("pred_type")).to_pandas().value_counts()
        print(pred_counts)
        
        # Run standard analysis
        evaluate_fn_rescue_potential(positions_df, str(output_dir))
    
    print(f"\nAnalysis complete. Results saved to {output_dir}")


if __name__ == "__main__":
    main()
