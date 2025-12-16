"""
Enhanced workflow functions for SpliceAI prediction processing.

This module provides enhanced versions of the workflow functions that use
the enhanced evaluation functions to directly incorporate all three probability
scores (donor, acceptor, neither) without relying on complex position matching.
"""
import copy
import polars as pl
from typing import Dict, List, Tuple, Union, Any, Optional, Callable

from .enhanced_evaluation import enhanced_evaluate_splice_site_errors
from .position_analysis import display_position_samples, format_diagnostic_summary
from meta_spliceai.system.genomic_resources import standardize_splice_sites_schema


def generate_context_agnostic_features(epsilon: float = 1e-10) -> List[pl.Expr]:
    """
    Generate context-agnostic features that apply to all positions.
    
    Args:
        epsilon: Small value to prevent division by zero
        
    Returns:
        List of Polars expressions for context-agnostic features
    """
    return [
        # Mean of neighbor values
        ((pl.col("context_score_m2") + pl.col("context_score_m1") + 
          pl.col("context_score_p1") + pl.col("context_score_p2")) / 4.0).alias("context_neighbor_mean"),
        
        # Context asymmetry (upstream vs downstream)
        ((pl.col("context_score_m1") + pl.col("context_score_m2")) - 
         (pl.col("context_score_p1") + pl.col("context_score_p2"))).alias("context_asymmetry"),
        
        # Maximum context score
        (pl.max_horizontal([
            pl.col("context_score_m2"), pl.col("context_score_m1"),
            pl.col("context_score_p1"), pl.col("context_score_p2")
        ])).alias("context_max")
    ]


def generate_donor_features(epsilon: float = 1e-10) -> List[pl.Expr]:
    """
    Generate donor-specific context features.
    
    Args:
        epsilon: Small value to prevent division by zero
        
    Returns:
        List of Polars expressions for donor features
    """
    # Basic differential features
    basic_features = [
        # Differential from previous positions (rise)
        (pl.col("donor_score") - pl.col("context_score_m1")).alias("donor_diff_m1"),
        (pl.col("donor_score") - pl.col("context_score_m2")).alias("donor_diff_m2"),
        
        # Differential to next positions (fall)
        (pl.col("donor_score") - pl.col("context_score_p1")).alias("donor_diff_p1"),
        (pl.col("donor_score") - pl.col("context_score_p2")).alias("donor_diff_p2"),
        
        # Surge ratio (how much higher is the probability compared to neighbors)
        (pl.col("donor_score") / (pl.col("context_score_m1") + pl.col("context_score_p1") + epsilon)).alias("donor_surge_ratio"),
        
        # Local peak detection (is this position a local maximum?)  
        # Include minimum threshold to avoid false peaks with near-zero probabilities
        ((pl.col("donor_score") > pl.col("context_score_m1")) & 
         (pl.col("donor_score") > pl.col("context_score_p1")) &
         (pl.col("donor_score") > 1e-3)).cast(pl.Int8).alias("donor_is_local_peak"),
         
        # Weighted context score (higher weight for central position)
        ((0.1 * pl.col("context_score_m2") + 
          0.2 * pl.col("context_score_m1") + 
          0.4 * pl.col("donor_score") + 
          0.2 * pl.col("context_score_p1") + 
          0.1 * pl.col("context_score_p2"))).alias("donor_weighted_context")
    ]
    
    # Advanced statistical features
    advanced_features = [
        # Normalized peak height (how many times higher than average neighbors)
        (pl.col("donor_score") / 
         ((pl.col("context_score_m2") + pl.col("context_score_m1") + 
           pl.col("context_score_p1") + pl.col("context_score_p2")) / 4.0 + epsilon)
        ).alias("donor_peak_height_ratio"),
        
        # Second derivative approximation (rate of change of the rate of change)
        ((pl.col("donor_score") - pl.col("context_score_m1")) - 
         (pl.col("context_score_p1") - pl.col("donor_score"))
        ).alias("donor_second_derivative"),
        
        # Signal strength compared to background (useful for detecting weak but true signals)
        (pl.col("donor_score") - 
         ((pl.col("context_score_m2") + pl.col("context_score_m1") + 
           pl.col("context_score_p1") + pl.col("context_score_p2")) / 4.0)
        ).alias("donor_signal_strength"),
        
        # Context difference ratio (how much higher is the center than the highest surrounding position)
        (pl.col("donor_score") / 
         (pl.max_horizontal([
             pl.col("context_score_m2"), pl.col("context_score_m1"),
             pl.col("context_score_p1"), pl.col("context_score_p2")
         ]) + epsilon)
        ).alias("donor_context_diff_ratio")
    ]
    
    return basic_features + advanced_features


def generate_acceptor_features(epsilon: float = 1e-10) -> List[pl.Expr]:
    """
    Generate acceptor-specific context features.
    
    Args:
        epsilon: Small value to prevent division by zero
        
    Returns:
        List of Polars expressions for acceptor features
    """
    # Basic differential features
    basic_features = [
        # Differential from previous positions
        (pl.col("acceptor_score") - pl.col("context_score_m1")).alias("acceptor_diff_m1"),
        (pl.col("acceptor_score") - pl.col("context_score_m2")).alias("acceptor_diff_m2"),
        
        # Differential to next positions
        (pl.col("acceptor_score") - pl.col("context_score_p1")).alias("acceptor_diff_p1"),
        (pl.col("acceptor_score") - pl.col("context_score_p2")).alias("acceptor_diff_p2"),
        
        # Surge ratio
        (pl.col("acceptor_score") / (pl.col("context_score_m1") + pl.col("context_score_p1") + epsilon)).alias("acceptor_surge_ratio"),
        
        # Local peak detection
        # Include minimum threshold to avoid false peaks with near-zero probabilities
        ((pl.col("acceptor_score") > pl.col("context_score_m1")) & 
         (pl.col("acceptor_score") > pl.col("context_score_p1")) &
         (pl.col("acceptor_score") > 1e-3)).cast(pl.Int8).alias("acceptor_is_local_peak"),
         
        # Weighted context score
        ((0.1 * pl.col("context_score_m2") + 
          0.2 * pl.col("context_score_m1") + 
          0.4 * pl.col("acceptor_score") + 
          0.2 * pl.col("context_score_p1") + 
          0.1 * pl.col("context_score_p2"))).alias("acceptor_weighted_context")
    ]
    
    # Advanced statistical features
    advanced_features = [
        # Normalized peak height (how many times higher than average neighbors)
        (pl.col("acceptor_score") / 
         ((pl.col("context_score_m2") + pl.col("context_score_m1") + 
           pl.col("context_score_p1") + pl.col("context_score_p2")) / 4.0 + epsilon)
        ).alias("acceptor_peak_height_ratio"),
        
        # Second derivative approximation (rate of change of the rate of change)
        ((pl.col("acceptor_score") - pl.col("context_score_m1")) - 
         (pl.col("context_score_p1") - pl.col("acceptor_score"))
        ).alias("acceptor_second_derivative"),
        
        # Signal strength compared to background
        (pl.col("acceptor_score") - 
         ((pl.col("context_score_m2") + pl.col("context_score_m1") + 
           pl.col("context_score_p1") + pl.col("context_score_p2")) / 4.0)
        ).alias("acceptor_signal_strength"),
        
        # Context difference ratio (how much higher is the center than the highest surrounding position)
        (pl.col("acceptor_score") / 
         (pl.max_horizontal([
             pl.col("context_score_m2"), pl.col("context_score_m1"),
             pl.col("context_score_p1"), pl.col("context_score_p2")
         ]) + epsilon)
        ).alias("acceptor_context_diff_ratio")
    ]
    
    return basic_features + advanced_features


def generate_cross_type_features(epsilon: float = 1e-10) -> List[pl.Expr]:
    """
    Generate cross-type features comparing donor and acceptor patterns.
    
    Args:
        epsilon: Small value to prevent division by zero
        
    Returns:
        List of Polars expressions for cross-type features
    """
    return [
        # Ratio of donor to acceptor peak heights
        ((pl.col("donor_peak_height_ratio") / (pl.col("acceptor_peak_height_ratio") + epsilon))
        ).alias("donor_acceptor_peak_ratio"),
        
        # Type dominance (which type has stronger signal)
        ((pl.col("donor_signal_strength") - pl.col("acceptor_signal_strength"))
        ).alias("type_signal_difference"),

        # Additional cross-type feature: probability difference ratio
        ((pl.col("donor_score") - pl.col("acceptor_score")) / 
         (pl.col("donor_score") + pl.col("acceptor_score") + epsilon)
        ).alias("score_difference_ratio"),
         
        # Signal strength ratio
        (pl.col("donor_signal_strength") / (pl.col("acceptor_signal_strength") + epsilon)
        ).alias("signal_strength_ratio")
    ]

def enhanced_process_predictions_with_all_scores(
    predictions: Dict[str, Any],
    ss_annotations_df: pl.DataFrame = None,
    threshold: float = 0.5,
    consensus_window: int = 2,
    error_window: int = 500,
    analyze_position_offsets: bool = False,
    collect_tn: bool = True,
    no_tn_sampling: bool = False,
    predicted_delta_correction: bool = False,
    splice_site_adjustments: Dict[str, Dict[str, int]] = None,
    add_derived_features: bool = True,
    fill_missing_values: bool = False,
    fill_value: float = 0.0,
    compute_all_context_features: bool = False,
    verbose: int = 1,
    **kwargs
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """
    Enhanced processing of SpliceAI predictions that directly incorporates
    all three probability scores (donor, acceptor, neither).
    
    This function replaces the original process_predictions_with_all_scores
    by using enhanced_evaluate_splice_site_errors, which preserves all three
    probability scores for each position.

    Parameters
    ----------
    predictions : Dict[str, Any]
        Output from predict_splice_sites_for_genes(), containing per-nucleotide probabilities
        Must include 'donor_prob', 'acceptor_prob', and 'neither_prob' for each gene
    ss_annotations_df : pl.DataFrame
        Splice site annotations with columns: chrom, start, end, strand, site_type, gene_id, transcript_id
    threshold : float, optional
        Threshold for classifying a prediction as a splice site, by default 0.5
    consensus_window : int, optional
        Size of window to use for consensus at true splice sites, by default 2
    error_window : int, optional
        Size of error window for FP/FN sites, by default 500
    analyze_position_offsets : bool, optional
        Whether to analyze positional offsets in predictions, by default False
    collect_tn : bool, optional
        Whether to collect true negative positions, by default True
    no_tn_sampling : bool, optional
        If True, preserve all TN positions without sampling. If False, apply sampling based on tn_sample_factor, by default False
    tn_sampling_mode : str, optional
        Mode for sampling true negatives, by default "random".
        Options: "random", "proximity", "window" (collects TNs adjacent to splice sites)
    predicted_delta_correction : bool, optional
        Whether to apply systematic prediction adjustments, by default False
    splice_site_adjustments : Dict[str, Dict[str, int]], optional
        Dictionary with custom adjustments to apply to predictions.
        Format: {'donor': {'plus': offset, 'minus': offset}, 
                'acceptor': {'plus': offset, 'minus': offset}}
    add_derived_features : bool, optional
        Whether to add derived features based on probability scores that may be useful 
        for meta-modeling, by default True
    verbose : int, optional
        Verbosity level, by default 1
        
    Returns
    -------
    Tuple[pl.DataFrame, pl.DataFrame]
        Tuple containing (error_df, positions_df)

    - error_df: DataFrame with error analysis
    - positions_df: DataFrame with all positions and their three probabilities
    """
    # For consistent debugging, ensure predictions are in deterministic order
    gene_ids = sorted(predictions.keys())
    
    # Check if we have splice site annotations
    if ss_annotations_df is None or ss_annotations_df.shape[0] == 0:
        return pl.DataFrame(), pl.DataFrame()
    
    # Standardize splice site schema (handles site_type → splice_type conversion)
    ss_annotations_df = standardize_splice_sites_schema(
        ss_annotations_df,
        verbose=(verbose >= 2)
    )
    
    # Verify 'splice_type' column exists in annotations after standardization
    if 'splice_type' not in ss_annotations_df.columns:
        raise ValueError("Splice site annotations must have a 'splice_type' column after standardization")
    
    # Verify that predictions have all three probability scores
    for gene_id, gene_data in predictions.items():
        if "donor_prob" not in gene_data:
            raise ValueError(f"Missing 'donor_prob' for gene {gene_id}")
        if "acceptor_prob" not in gene_data:
            raise ValueError(f"Missing 'acceptor_prob' for gene {gene_id}")
        if "neither_prob" not in gene_data:
            raise ValueError(f"Missing 'neither_prob' for gene {gene_id}")

    tn_sample_factor = kwargs.get("tn_sample_factor", 1.2)
    tn_sampling_mode = kwargs.get("tn_sampling_mode", "random")
    tn_proximity_radius = kwargs.get("tn_proximity_radius", 50)
    
    # Make a deep copy of predictions if adjustments will be applied
    # This prevents modifications to the original predictions dictionary
    # import copy
    if predicted_delta_correction:
        if verbose >= 1:
            print("[info] Making deep copy of predictions for adjustments")
        pred_results = copy.deepcopy(predictions)
    else:
        pred_results = predictions
    
    # Call enhanced evaluation function that directly incorporates all three probabilities
    error_df, positions_df = enhanced_evaluate_splice_site_errors(
        ss_annotations_df,
        pred_results,
        threshold=threshold,
        consensus_window=consensus_window,
        error_window=error_window,
        analyze_position_offsets=analyze_position_offsets,
        collect_tn=collect_tn,
        no_tn_sampling=no_tn_sampling,
        tn_sample_factor=tn_sample_factor,
        tn_sampling_mode=tn_sampling_mode,
        tn_proximity_radius=tn_proximity_radius,
        predicted_delta_correction=predicted_delta_correction,
        splice_site_adjustments=splice_site_adjustments,
        return_positions_df=True,  # Always return positions DataFrame
        verbose=verbose
    )
    
    # Add derived features if requested
    if add_derived_features and positions_df.height > 0:
        if verbose >= 1:
            print("[info] Adding derived probability features for meta-modeling")
            
        # Avoid division by zero by adding a small epsilon
        epsilon = 1e-10
        
        # Create a list to hold all the feature expressions
        derived_features = []
        
        # Basic probability ratios
        derived_features.extend([
            
            # Normalized probabilities (0-1 scale)
            (pl.col("donor_score") / (pl.col("donor_score") + pl.col("acceptor_score") + epsilon)).alias("relative_donor_probability"),
            ((pl.col("donor_score") + pl.col("acceptor_score")) / (pl.col("donor_score") + pl.col("acceptor_score") + pl.col("neither_score") + epsilon)).alias("splice_probability"),
            
            # Relative differences between probabilities
            ((pl.col("donor_score") - pl.col("acceptor_score")) / (pl.max_horizontal([pl.col("donor_score"), pl.col("acceptor_score")]) + epsilon)).alias("donor_acceptor_diff"),
            ((pl.max_horizontal([pl.col("donor_score"), pl.col("acceptor_score")]) - pl.col("neither_score")) / 
             (pl.max_horizontal([pl.col("donor_score"), pl.col("acceptor_score"), pl.col("neither_score")]) + epsilon)).alias("splice_neither_diff"),
            
            # Log-odds like transformations (handling zeros appropriately)
            (pl.col("donor_score").add(epsilon).log() - pl.col("acceptor_score").add(epsilon).log()).alias("donor_acceptor_logodds"),
            (pl.col("donor_score").add(pl.col("acceptor_score")).add(epsilon).log() - pl.col("neither_score").add(epsilon).log()).alias("splice_neither_logodds")
        ])
        
        # Add entropy-like feature measuring uncertainty in the probability distribution
        entropy_expr = (
            -pl.col("donor_score") * pl.col("donor_score").add(epsilon).log()
            -pl.col("acceptor_score") * pl.col("acceptor_score").add(epsilon).log()
            -pl.col("neither_score") * pl.col("neither_score").add(epsilon).log()
        ).alias("probability_entropy")
        derived_features.append(entropy_expr)
        
        # Apply the basic derived features first, before attempting to create context-based features
        # This ensures columns like donor_score, acceptor_score, etc. are available for further calculations
        positions_df = positions_df.with_columns(derived_features)
        
        # Reset derived_features list for context-based features
        context_derived_features = []
        
        # Add context-based features if available
        print("[DEBUG] Checking for context columns: ", list(positions_df.columns))
        context_columns = [col for col in positions_df.columns if col.startswith('context_')]
        if context_columns:
            if verbose >= 1:
                print("[info] Adding context-based meta-features")

            # Check if we have all needed context columns
            context_available = all(col in positions_df.columns for col in 
                ['context_score_m2', 'context_score_m1', 'context_score_p1', 'context_score_p2'])

            if verbose >= 1:
                print(f"[info] Context available: {context_available}")
            
            # Generate and add features if context columns are available
            if context_available:
                ###############################################
                ### SECTION: Context Feature Generation    ###
                ###############################################

                # 1. Generate and add site-agnostic context features
                context_agnostic_features = generate_context_agnostic_features(epsilon=epsilon)
                context_derived_features.extend(context_agnostic_features)

                # 2. Generate and add donor-specific features
                donor_features = generate_donor_features(epsilon=epsilon)
                context_derived_features.extend(donor_features)

                # 3. Generate and add acceptor-specific features
                acceptor_features = generate_acceptor_features(epsilon=epsilon)
                context_derived_features.extend(acceptor_features)

            # Apply the donor and acceptor context features before attempting to create cross-type features
            # This ensures columns like donor_peak_height_ratio and acceptor_signal_strength are available
            if context_derived_features:
                # DEBUG: Before adding the features, check what types of positions we have
                if verbose >= 1:
                    print(f"[debug] Position types distribution:")
                    if "splice_type" in positions_df.columns:
                        type_counts = positions_df.group_by("splice_type").count()
                        print(type_counts.to_pandas())
                    if "pred_type" in positions_df.columns:
                        pred_counts = positions_df.group_by("pred_type").count()
                        print(pred_counts.to_pandas())
                
                positions_df = positions_df.with_columns(context_derived_features)
                
                # DEBUG: After adding context features, check for null values
                if verbose >= 1:
                    print("[debug] Checking for null values in new context features:")
                    
                    # Check context-agnostic features
                    context_agnostic_cols = [col for col in positions_df.columns if col.startswith("context_") and 
                                          col not in {"context_score_m2", "context_score_m1", "context_score_p1", "context_score_p2"}]
                    
                    if context_agnostic_cols:
                        context_null_counts = positions_df.select(
                            [pl.col(col).is_null().sum().alias(col) for col in context_agnostic_cols]
                        )
                        print(f"Context agnostic feature null counts: {context_null_counts.to_pandas().iloc[0].to_dict()}")
                    
                    # Check donor features
                    donor_feature_cols = [col for col in positions_df.columns if col.startswith("donor_") and 
                                        col not in {"donor_score", "donor_prob"}]
                    if donor_feature_cols:
                        donor_null_counts = positions_df.select(
                            [pl.col(col).is_null().sum().alias(col) for col in donor_feature_cols]
                        )
                        print(f"Donor feature null counts: {donor_null_counts.to_pandas().iloc[0].to_dict()}")
                        
                        # Check where nulls are occurring by splice type
                        if "splice_type" in positions_df.columns and any(donor_null_counts.to_pandas().iloc[0] > 0):
                            print("[debug] Analyzing donor nulls by splice type:")
                            donor_sample = positions_df.filter(pl.col(donor_feature_cols[0]).is_null())
                            if donor_sample.height > 0:
                                donor_null_types = donor_sample.group_by("splice_type").count()
                                print(donor_null_types.to_pandas())
                
                    # Check acceptor features
                    acceptor_feature_cols = [col for col in positions_df.columns if col.startswith("acceptor_") and 
                                            col not in {"acceptor_score", "acceptor_prob"}]
                    if acceptor_feature_cols:
                        acceptor_null_counts = positions_df.select(
                            [pl.col(col).is_null().sum().alias(col) for col in acceptor_feature_cols]
                        )
                        print(f"Acceptor feature null counts: {acceptor_null_counts.to_pandas().iloc[0].to_dict()}")
                        
                        # Check where nulls are occurring by splice type
                        if "splice_type" in positions_df.columns and any(acceptor_null_counts.to_pandas().iloc[0] > 0):
                            print("[debug] Analyzing acceptor nulls by splice type:")
                            acceptor_sample = positions_df.filter(pl.col(acceptor_feature_cols[0]).is_null())
                            if acceptor_sample.height > 0:
                                acceptor_null_types = acceptor_sample.group_by("splice_type").count()
                                print(acceptor_null_types.to_pandas())
                
            # Apply the context features first to make them available for cross-type features
            positions_df = positions_df.with_columns(context_derived_features)
            
            ###############################################
            ### SECTION: Cross-Type Features          ###
            ###############################################
            if context_available: 
                # Generate cross-type features (comparing donor and acceptor patterns)
                cross_type_features = generate_cross_type_features(epsilon=epsilon)
                
                # Apply cross-type features
                positions_df = positions_df.with_columns(cross_type_features)
                
            ###############################################
            ### SECTION: Handle Missing Values        ###
            ###############################################
            if fill_missing_values and context_available:
                if verbose >= 1:
                    print(f"[info] Filling missing values with {fill_value}")
                
                # Identify all derived feature columns (excluding basic scores)
                feature_cols = []
                
                # Context agnostic features
                context_agnostic_cols = [col for col in positions_df.columns if col.startswith("context_") and 
                                        col not in {"context_score_m2", "context_score_m1", "context_score_p1", "context_score_p2"}]
                feature_cols.extend(context_agnostic_cols)
                
                # Donor context features
                donor_context_cols = [col for col in positions_df.columns 
                                    if col.startswith("donor_") and 
                                    col not in {"donor_score", "donor_prob"}]
                feature_cols.extend(donor_context_cols)
            
                # Acceptor context features
                acceptor_context_cols = [col for col in positions_df.columns 
                                        if col.startswith("acceptor_") and 
                                        col not in {"acceptor_score", "acceptor_prob"}]
                feature_cols.extend(acceptor_context_cols)
                
                # Cross-type features
                cross_type_cols = [
                    "donor_acceptor_peak_ratio", "type_signal_difference",
                    "score_difference_ratio", "signal_strength_ratio"
                ]
                feature_cols.extend([col for col in cross_type_cols if col in positions_df.columns])
            
                # Create a dictionary mapping column names to fill values
                fill_dict = {col: fill_value for col in feature_cols}
                
                # Fill null values
                if fill_dict:
                    positions_df = positions_df.fill_null(fill_dict)
                    
                    if verbose >= 1:
                        print(f"[info] Filled null values in {len(fill_dict)} columns")
                        
                    # Verify no nulls remain in derived features
                    if verbose >= 2:
                        print("[debug] Verifying no null values remain in derived features...")
                        for col in feature_cols:
                            null_count = positions_df.select(pl.col(col).is_null().sum()).item()
                            if null_count > 0:
                                print(f"WARNING: {col} still has {null_count} null values after filling!")
                        print("[debug] Null value verification complete.")
        
        # Display sample position information using the new utility function
        if verbose >= 1:
            # Display a more detailed summary
            summary = format_diagnostic_summary(positions_df)
            print(f"\n{summary}")
            
            # Sample and display the position information
            sample_tables = display_position_samples(
                positions_df,
                max_samples_per_type=10,  # Show up to 10 samples per prediction type
                verbose=1 if verbose == 1 else 2  # Show more details if higher verbosity
            )
    
    # Print summary information
    if verbose >= 1:
        print(f"[info] Enhanced evaluation complete")
        print(f"[info] Error DataFrame shape: {error_df.shape}")
        print(f"[info] Positions DataFrame shape: {positions_df.shape}")
        
        # Original statistics on prediction results
        if error_df.height > 0:
            fp_count = error_df.filter(pl.col("error_type") == "FP").height
            fn_count = error_df.filter(pl.col("error_type") == "FN").height
            print(f"[info] Found {fp_count} false positives and {fn_count} false negatives")
            
        if positions_df.height > 0:
            tp_count = positions_df.filter(pl.col("pred_type") == "TP").height
            fp_count = positions_df.filter(pl.col("pred_type") == "FP").height
            fn_count = positions_df.filter(pl.col("pred_type") == "FN").height
            tn_count = positions_df.filter(pl.col("pred_type") == "TN").height
            print(f"[info] Positions by type: TP={tp_count}, FP={fp_count}, FN={fn_count}, TN={tn_count}")
        
        # Verify final column count to ensure all features were preserved
        feature_count = len(positions_df.columns)
        context_cols = [col for col in positions_df.columns if col.startswith('context_')]
        derived_cols = [col for col in positions_df.columns 
                       if any(col.startswith(prefix) for prefix in 
                             ['donor_diff', 'donor_surge', 'donor_peak', 'donor_is_local', 
                              'acceptor_diff', 'acceptor_surge', 'acceptor_peak'])]
        
        # Report on feature engineering results
        if add_derived_features:
            base_column_count = 12  # Approximate count of original columns
            print(f"[info] Added ~{feature_count - base_column_count} derived features for meta-modeling")
            print(f"[info] Context features: {len(context_cols)}/{feature_count}")
            print(f"[info] Derived features: {len(derived_cols)}/{feature_count}")
            
            if len(context_cols) == 0:
                print("[warning] No context features found - derived features may not be calculated correctly")
    
    # Ensure all new columns are preserved when returning the dataframe
    # This is critical so they don't get lost in downstream processing
    return error_df, positions_df
