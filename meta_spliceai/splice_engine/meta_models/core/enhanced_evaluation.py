"""
Enhanced evaluation functions for splice site prediction.

This module provides enhanced versions of the splice site evaluation functions
that directly incorporate all three probability scores (donor, acceptor, neither)
to avoid positional matching issues in the original implementation.

These functions build on the original evaluation logic from evaluate_models.py
but add direct support for multi-score processing.
"""

from turtle import pos
import numpy as np
import polars as pl
import random
from collections import defaultdict
from typing import Dict, List, Tuple, Union, Any, Optional

from meta_spliceai.splice_engine.utils_df import is_dataframe_empty
from meta_spliceai.splice_engine.splice_error_analyzer import SpliceAnalyzer
from meta_spliceai.splice_engine.evaluate_models import is_within_overlapping_splice_site

from meta_spliceai.splice_engine.meta_models.utils.verify_splice_adjustment import (
    verify_probability_sums,
    verify_adjustment_effect
)

# Helper function to normalize strand notation
from meta_spliceai.splice_engine.utils_bio import normalize_strand

# Import utilities for adjusting predicted splice site probabilities
from meta_spliceai.splice_engine.meta_models.utils.infer_splice_site_adjustments import (
    adjust_scores,
    apply_custom_splice_site_adjustments,
    auto_detect_splice_site_adjustments
)

from meta_spliceai.splice_engine.meta_models.core.schema_utils import (
    ensure_schema, 
    extend_schema, 
    DEFAULT_POSITION_SCHEMA, 
    DEFAULT_ERROR_SCHEMA, 
    prepare_dataframes_for_stacking
)
from meta_spliceai.splice_engine.meta_models.core.position_analysis import (
    analyze_transcript_position_stats,
    analyze_gene_level_positions
)

from meta_spliceai.splice_engine.meta_models.utils.workflow_utils import (
    print_emphasized,
    print_with_indent,
    print_section_separator
)


def get_context_scores(probabilities: np.ndarray, position: int, window_size: int = 2) -> List[float]:
    """
    Extract probability scores from a window around the specified position,
    with proper handling of boundary conditions and symmetric padding.
    
    Parameters
    ----------
    probabilities : np.ndarray
        Array of probability scores
    position : int
        Central position to extract context around
    window_size : int, default=2
        Size of the window on each side of the position
        
    Returns
    -------
    List[float]
        List of context scores with consistent length (2*window_size + 1)
        centered on the specified position, with zero padding as needed
    """
    # Calculate window boundaries
    start = max(0, position - window_size)
    end = min(len(probabilities), position + window_size + 1)
    
    # Extract available scores
    context_scores = list(probabilities[start:end])
    
    # Calculate padding needed on each side
    left_pad = max(0, window_size - (position - start))
    right_pad = max(0, window_size - (end - position - 1))
    
    # Apply symmetric padding
    context_scores = [0.0] * left_pad + context_scores + [0.0] * right_pad
    
    # Ensure we have exactly the expected number of elements
    expected_length = 2 * window_size + 1
    if len(context_scores) != expected_length:
        # This is a safety check - in theory our padding logic should guarantee the correct length
        if len(context_scores) < expected_length:
            # Pad to the right if somehow we're still short
            context_scores = context_scores + [0.0] * (expected_length - len(context_scores))
        else:
            # Trim if somehow we have too many elements
            context_scores = context_scores[:expected_length]
    
    return context_scores


def enhanced_evaluate_donor_site_errors(
    annotations_df, 
    pred_results, 
    threshold=0.5, 
    consensus_window=2, 
    error_window=500, 
    collect_tn=True,
    tn_sample_factor=1.2,
    tn_sampling_mode="random",
    tn_proximity_radius=50,
    no_tn_sampling=False,
    predicted_delta_correction=False,  # Add as explicit parameter with default False
    splice_site_adjustments=None,
    return_positions_df=True,
    **kargs
):
    """
    Evaluate SpliceAI predictions for donor splice sites and identify FPs and FNs, including windowed regions.

    Parameters:
    - annotations_df (pl.DataFrame): DataFrame containing true splice site annotations.
      Columns: ['chrom', 'start', 'end', 'position', 'strand', 'site_type', 'gene_id', 'transcript_id']
      Also see more details in run_spliceai_workflow.py -> retrieve_splice_sites()

    - pred_results (dict): The output of predict_splice_sites_for_genes(), containing per-nucleotide probabilities.
    - threshold (float): Threshold for classifying a prediction as a donor site (default is 0.9).
    - consensus_window (int): Tolerance window around true splice sites.
    - error_window (int or tuple): Window size (or sizes) for tracking the surrounding region of FPs and FNs.
    - collect_tn : bool
        Whether to gather TN data points. If False, no TN rows are appended to positions_df.
    - tn_sample_factor : float
        Ratio controlling how many TN rows we keep relative to the total of {TP,FP,FN}.
        e.g., if we have 100 TPs+FPs+FNs total, sample up to tn_sample_factor * 100 = 200 TN.
    - tn_sampling_mode : str
        "random" => random subset of TN
        "proximity" => prefer TN near TPs/FNs (within tn_proximity_radius).
        "window" => collect TNs adjacent to true splice sites within error_window
    - tn_proximity_radius : int
        Radius for measuring closeness if tn_sampling_mode="proximity".
    - no_tn_sampling : bool
        If True, preserve all TN positions without sampling. If False, apply sampling based on tn_sample_factor.
    - predicted_delta_correction : bool
        Whether to apply adjustment to SpliceAI predictions.
    - splice_site_adjustments : dict, optional
        Dictionary with custom adjustments to apply to SpliceAI predictions. If provided, 
        this is used instead of the standard SpliceAI adjustments when predicted_delta_correction is True.
        Format: {'donor': {'plus': offset, 'minus': offset}, 'acceptor': {'plus': offset, 'minus': offset}}
    - return_positions_df : bool, optional
        Whether to return the positions DataFrame, by default False.
    - verbose (int): Level of verbosity.

    Returns:
    - error_df (pl.DataFrame): A DataFrame containing positions of FPs and FNs, with window coordinates.


    NOTE: 
    - evaluate_donor_site_errors_at_gene_level() can be used to evaluate predictions at the gene level.
    - .index: The .index(gene_id) method is used to find the index of the gene_id in the grouped_annotations['gene_id'] list. 
            This index is then used to access the corresponding donor sites in the grouped_annotations['donor_sites'] list.
    """
    # from .extract_genomic_features import get_overlapping_gene_metadata

    verbose = kargs.get('verbose', 1)
    chromosome = kargs.get('chromosome', kargs.get('chr', '?'))
    overlapping_genes_metadata = kargs.get('overlapping_genes_metadata', None)
    adjust_for_overlapping_genes = kargs.get('adjust_for_overlapping_genes', True)

    if overlapping_genes_metadata is None:
        sa = SpliceAnalyzer()
        overlapping_genes_metadata = sa.retrieve_overlapping_gene_metadata()

    # Data structures for storing errors and positions
    error_list = []
    positions_list = []

    # Group true donor annotations by gene
    grouped_annotations = annotations_df.filter(pl.col('splice_type') == 'donor').group_by('gene_id').agg(
        pl.struct(['start', 'end', 'position', 'transcript_id']).alias('donor_sites')
    ).to_dict(as_series=False)  # 'transcript_id'

    # Process each gene's predictions in pred_results
    for gene_id, gene_data in pred_results.items():

        strand = normalize_strand(gene_data['strand'])

        if gene_id not in grouped_annotations['gene_id']:
            if verbose: 
                print(f"No donor annotations for gene: {gene_id}")

            # Optional: Append an entry for these genes with no donor annotations
            # error_list.append({
            #     'gene_id': gene_id,
            #     'transcript_id': None,
            #     'error_type': None,  # No error type, since there are no donor annotations
            #     'position': None,
            #     'window_start': None,
            #     'window_end': None, 
            #     'strand': strand
            # })

            # positions_list.append({
            #     'gene_id': gene_id,
            #     'transcript_id': None,
            #     'position': None,
            #     'predicted_position': None,
            #     'pred_type': None,
            #     'score': 0.0,
            #     'strand': strand,
            #     'donor_score': 0.0,
            #     'acceptor_score': 0.0,
            #     'neither_score': 0.0,
            #     'splice_type': None
            # })

            continue
        
        # Extract relevant info
        # strand = normalize_strand(gene_data['strand'])
        donor_probabilities = np.array(gene_data['donor_prob'])  # Probabilities for all positions
        acceptor_probabilities = np.array(gene_data['acceptor_prob'])  # Probabilities for all positions
        neither_probabilities = np.array(gene_data['neither_prob'])  # Probabilities for all positions
        gene_len = len(donor_probabilities)
        # gene_start = gene_data['gene_start']
        # gene_end = gene_data['gene_end']

        # True donor positions
        # - Extract donor sites from grouped_annotations
        # - Directly extract donor positions using the 'position' column, and adjust donor positions based on strand
        # - Calculate relative positions and map them to transcript_id
        donor_sites = grouped_annotations['donor_sites'][grouped_annotations['gene_id'].index(gene_id)]

        true_donor_positions = []
        position_to_transcript = defaultdict(set)  # Dictionary to map relative positions to transcript IDs

        for site in donor_sites:
            if strand == '+':
                relative_position = site['position'] - gene_data['gene_start']
            elif strand == '-':
                relative_position = gene_data['gene_end'] - site['position']
            else:
                raise ValueError(f"Invalid strand value: {strand}")

            # Append the relative position
            true_donor_positions.append(relative_position)

            # Map position to transcript_id, accounting for shared positions
            position_to_transcript[relative_position].add(site['transcript_id'])

        # Sort positions and convert transcript ID sets to lists
        true_donor_positions = np.array(sorted(set(true_donor_positions)))
        # This retains only "unique" donor sites

        position_to_transcript = {pos: list(transcripts) for pos, transcripts in position_to_transcript.items()}

        # ------------------------------------------
        # Align probabilities with donor positions
        if predicted_delta_correction:
            # Apply custom adjustments if provided, otherwise use standard adjustments
            if splice_site_adjustments is not None:
                # Apply the same position adjustments to all three probability arrays
                donor_probabilities = apply_custom_splice_site_adjustments(
                    donor_probabilities, strand, 'donor', splice_site_adjustments
                )
                # Apply identical position shifts to acceptor and neither probabilities
                acceptor_probabilities = apply_custom_splice_site_adjustments(
                    acceptor_probabilities, strand, 'donor', splice_site_adjustments
                )
                neither_probabilities = apply_custom_splice_site_adjustments(
                    neither_probabilities, strand, 'donor', splice_site_adjustments, is_neither_prob=True
                )
            else:
                # Apply the same adjustments to all three probability arrays
                donor_probabilities = adjust_scores(donor_probabilities, strand, 'donor')
                # Use 'donor' as site_type for all arrays to maintain consistent shifting
                acceptor_probabilities = adjust_scores(acceptor_probabilities, strand, 'donor')
                neither_probabilities = adjust_scores(neither_probabilities, strand, 'donor', is_neither_prob=True)
            
            # Verify probability sums after adjustment
            verify_probability_sums(donor_probabilities, acceptor_probabilities, neither_probabilities, 
                                    strand, 'donor', splice_site_adjustments, gene_id=gene_id, verbose=verbose)
            
        # Validate positions within range
        assert true_donor_positions.max() < len(donor_probabilities), \
            "true_donor_positions contain indices out of range for donor_probabilities"

        # Scores corresponding to the positions
        true_donor_scores = donor_probabilities[true_donor_positions]  # for debugging only
        true_donor_scores = np.round(true_donor_scores, 3)  # Round for readability

        # Binarize predictions at each position
        # - Initialize label prediction vector based on the threshold
        # label_predictions = (donor_probabilities >= threshold).astype(int)
        label_predictions = np.array([1 if prob >= threshold else 0 for prob in donor_probabilities])

        # if strand == '-':
        #     label_predictions = label_predictions[::-1]

        # Initialize counts for the current gene
        gene_results = {'gene_id': gene_id, 'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}
        # Initialize a dictionary to track if a true donor site is missed (for FN counting)
        true_donor_status = {i: False for i in range(len(true_donor_positions))}
        
        fp_positions, fn_positions, tp_positions = [], [], []
        tn_positions_all = []  # We'll store them here, then sample if collect_tn=True

        # Loop over each position in the label sequence
        for i, pred_label in enumerate(label_predictions):
            found_in_window = False

            # Check if this position is a true splice site for the current gene
            for idx, true_pos in enumerate(true_donor_positions):

                if true_donor_status[idx]:  # Skip already processed true donor sites
                    continue

                window_start = true_pos - consensus_window
                window_end = true_pos + consensus_window

                if window_start <= i <= window_end:  # i is the predicted position
                    found_in_window = True  # i is within the window (close enough to true position)

                    # Get the associated transcript IDs for this true position
                    associated_transcripts = position_to_transcript[true_pos]   
                    
                    # Count TP or FN based on the prediction
                    if pred_label == 1:
                        gene_results['TP'] += 1
                        true_donor_status[idx] = True  # Mark this true donor site as found

                        # Use the maximum probability within the window instead of just at position i
                        # --- robustly find max prob within window (avoid empty slice) ---
                        slice_start = max(0, window_start)
                        slice_end   = min(len(donor_probabilities), window_end + 1)
                        window_probs = donor_probabilities[slice_start:slice_end]
                        if window_probs.size == 0:
                            # Fallback: use the current index if window is empty (should not happen but safeguards against
                            # out-of-range windows due to annotation issues near sequence boundaries).
                            max_prob_idx = i
                        else:
                            max_prob_idx = slice_start + np.argmax(window_probs)
                        score = donor_probabilities[max_prob_idx]

                        assert score >= threshold, \
                            f"Unexpected score for TP: {score} < {threshold}"

                        # Get context scores (surrounding positions)
                        context_scores = get_context_scores(donor_probabilities, i, consensus_window)
                        # Pad with zeros if needed to ensure we always have 5 elements (center ± 2)
                        # while len(context_scores) < 5:
                        #     context_scores.append(0.0)

                        # Append all associated transcript IDs for this position
                        for transcript_id in associated_transcripts:
                            tp_positions.append({
                                'gene_id': gene_id,
                                'transcript_id': transcript_id,
                                'position': i,  # Current position (+: relative to gene_start, -: relative to gene_end)
                                'predicted_position': i,  # i should be within a small window of true_pos
                                'true_position': true_pos,  # True splice site position
                                'pred_type': 'TP',  # Made flexiby by allowing i to be within a small window of true_pos
                                'score': score,
                                'strand': strand,
                                'donor_score': score,  # Use the max score here too
                                'acceptor_score': acceptor_probabilities[i],
                                'neither_score': neither_probabilities[i],
                                # Add surrounding context scores for meta-modeling
                                'context_score_m2': context_scores[0] if len(context_scores) > 0 else 0.0,
                                'context_score_m1': context_scores[1] if len(context_scores) > 1 else 0.0,
                                'context_score_p1': context_scores[3] if len(context_scores) > 3 else 0.0,
                                'context_score_p2': context_scores[4] if len(context_scores) > 4 else 0.0,
                                'splice_type': 'donor'
                            })
                            
                    break
            # End for each position in the label sequence

            # Count non-splice sites (negative examples)
            if not found_in_window:  # Position i is not within a small window of a true_pos
                is_fp = False
                if pred_label == 1:
                    # Predicted positive but not a splice site -> FP
                    is_fp = True

                    # Adjust FP count for overlapping genes
                    if adjust_for_overlapping_genes and is_within_overlapping_splice_site(
                        i, consensus_window, gene_id, gene_data, overlapping_genes_metadata, grouped_annotations
                    ):
                        is_fp = False
                    
                    if is_fp:
                        gene_results['FP'] += 1  

                        # Get context scores (surrounding positions)
                        context_scores = get_context_scores(donor_probabilities, i, consensus_window)
                        # Pad with zeros if needed to ensure we always have 5 elements (center ± 2)
                        # while len(context_scores) < 5:
                        #     context_scores.append(0.0)

                        # FP positions are not associated with transcripts since they're not true splice sites
                        fp_positions.append({
                            'gene_id': gene_id,
                            'transcript_id': None,  # Not associated with a transcript since it's not a splice site
                            'position': i,  # Current position (+: relative to gene_start, -: relative to gene_end)
                            'predicted_position': i,  # i should be within a small window of true_pos
                            'true_position': None,  # Not a splice site by ground truth
                            'pred_type': 'FP',
                            'score': donor_probabilities[i],
                            'strand': strand,  
                            'donor_score': donor_probabilities[i],
                            'acceptor_score': acceptor_probabilities[i],
                            'neither_score': neither_probabilities[i],
                            # Add surrounding context scores for meta-modeling
                            'context_score_m2': context_scores[0] if len(context_scores) > 0 else 0.0,
                            'context_score_m1': context_scores[1] if len(context_scores) > 1 else 0.0,
                            'context_score_p1': context_scores[3] if len(context_scores) > 3 else 0.0,
                            'context_score_p2': context_scores[4] if len(context_scores) > 4 else 0.0,
                            'splice_type': None  # Not a splice site 
                        })

                else:
                    # This is a negative prediction => candidate TN
                    # We'll store it and sample later
                    # Get context scores (surrounding positions)
                    context_scores = get_context_scores(donor_probabilities, i, consensus_window)
                    # Pad with zeros if needed to ensure we always have 5 elements (center ± 2)
                    # while len(context_scores) < 5:
                    #     context_scores.append(0.0)
                    
                    tn_positions_all.append({
                        'gene_id': gene_id,
                        'transcript_id': None,  # Not associated with a transcript since it's not a splice site
                        'position': i,  # Current position 
                        'predicted_position': None,  # predicted to be non-splice site (TN)
                        'true_position': None,  # Not a splice site by ground truth
                        'pred_type': 'TN',
                        'score': donor_probabilities[i],
                        'strand': strand,
                        'donor_score': donor_probabilities[i],
                        'acceptor_score': acceptor_probabilities[i],
                        'neither_score': neither_probabilities[i],
                        # Add surrounding context scores for meta-modeling
                        'context_score_m2': context_scores[0] if len(context_scores) > 0 else 0.0,
                        'context_score_m1': context_scores[1] if len(context_scores) > 1 else 0.0,
                        'context_score_p1': context_scores[3] if len(context_scores) > 3 else 0.0,
                        'context_score_p2': context_scores[4] if len(context_scores) > 4 else 0.0,
                        'splice_type': None  # Not a splice site
                    })

                    # Predicted negative correctly => TN
                    gene_results['TN'] += 1
        # End for each position in the label sequence

        # Count false negatives
        d = 0
        for idx, status in true_donor_status.items():
            if not status:  # Unmarked true donor site
                # => FN
                position = true_donor_positions[idx]  # true relative position
                score = donor_probabilities[position]
                label = label_predictions[position]

                assert 0 <= position < len(label_predictions), \
                    f"Position {position} out of bounds for label_predictions array of length {len(label_predictions)}"

                # Get the associated transcript IDs for this true position
                associated_transcripts = position_to_transcript[position]

                # Peek the surrounding scores
                context_scores = get_context_scores(donor_probabilities, position, consensus_window)

                if verbose > 0: 
                    if d < 2: 
                        print(f"[DEBUG] FN: gene_id={gene_id}, position={position}, score={score:.4f}, label={label}  threshold={threshold}")
                    # Format array elements with 4 decimal places
                    formatted_scores = [f"{score:.4f}" for score in context_scores]
                    print_with_indent(f"Surrounding donor scores for FN at {position}: {formatted_scores}", indent_level=1)
                    d += 1
                
                # Should we use the max score within the window instead of just at position i?
                score = max(context_scores)

                # assert label == 0, f"Unexpected label for FN: {label} != 0"  # Todo
                # assert score <= threshold, f"Unexpected score for FN: {score} > {threshold}, label={label}"  # Todo
                
                gene_results['FN'] += 1
                
                # Extract context scores (surrounding positions)
                # Ensure we have proper padding if we're at gene boundaries
                # context_scores = list(surrounding_scores)
                # # Pad with zeros if needed to ensure we always have 5 elements (center ± 2)
                # while len(context_scores) < 5:
                #     context_scores.append(0.0)
                
                # Append FN positions with all associated transcript IDs
                for transcript_id in associated_transcripts:
                    fn_positions.append({
                        'gene_id': gene_id,
                        'transcript_id': transcript_id,  # Transcript that uses the splice site
                        'position': position,  # Current position (+: relative to gene_start, -: relative to gene_end)
                        'predicted_position': None,  # predicted to be non-splice site (FN)
                        'true_position': position,  # True splice site position
                        'pred_type': 'FN',
                        'score': score,
                        'strand': strand,
                        'donor_score': score, # donor_probabilities[position],
                        'acceptor_score': acceptor_probabilities[position],
                        'neither_score': neither_probabilities[position],
                        # Add surrounding context scores for meta-modeling
                        'context_score_m2': context_scores[0] if len(context_scores) > 0 else 0.0,
                        'context_score_m1': context_scores[1] if len(context_scores) > 1 else 0.0,
                        'context_score_p1': context_scores[3] if len(context_scores) > 3 else 0.0,
                        'context_score_p2': context_scores[4] if len(context_scores) > 4 else 0.0,
                        'splice_type': 'donor'
                    })
                    
        # End for each true donor site and the counting of FNs

        # Sample TN positions if needed
        tn_collection = []  # Initialize for this gene
        
        if collect_tn and len(tn_positions_all) > 0:
            if no_tn_sampling:
                # No sampling mode - preserve all TN positions
                tn_collection = tn_positions_all
                if verbose >= 1:
                    print(f"[info] No TN sampling: preserving all {len(tn_positions_all)} TN positions for gene {gene_id}")
            else:
                # Apply sampling based on tn_sample_factor
                # Calculate how many TN samples we should collect
                num_tp_fp_fn = len(tp_positions) + len(fp_positions) + len(fn_positions)
                num_tn_to_sample = min(int(num_tp_fp_fn * tn_sample_factor), len(tn_positions_all))
                
                # Sample TNs with specified mode/strategy
                if num_tn_to_sample > 0:
                    # We need to sample TNs
                    if len(tn_positions_all) <= num_tn_to_sample:
                        # If we have fewer TN positions than desired, keep all of them
                        tn_collection = tn_positions_all
                    else:
                        # We have more than we want to sample
                        if tn_sampling_mode == "random":
                            # Select a random subset of TN positions
                            tn_collection = random.sample(tn_positions_all, num_tn_to_sample)
                        elif tn_sampling_mode == "proximity":
                            # Prefer positions near PREDICTED splice sites (TP + FP)
                            # Use predicted positions to avoid data leakage
                            predicted_positions = [p['predicted_position'] for p in (tp_positions + fp_positions) 
                                                 if p.get('predicted_position') is not None]
                            
                            if predicted_positions:
                                # Calculate minimum distance to a predicted position for each TN
                                tn_with_dists = []
                                for tn_pos in tn_positions_all:
                                    pos = tn_pos['position']
                                    min_dist = min(abs(pos - pred_pos) for pred_pos in predicted_positions)
                                    tn_with_dists.append((tn_pos, min_dist))
                                
                                # Sort by distance and take the top num_tn_to_sample
                                tn_with_dists.sort(key=lambda x: x[1])
                                tn_collection = [x[0] for x in tn_with_dists[:num_tn_to_sample]]
                            else:
                                # fallback to random if no predicted positions
                                tn_collection = random.sample(tn_positions_all, num_tn_to_sample)
                        elif tn_sampling_mode == "window":
                            # Collect TNs adjacent to PREDICTED donor sites within proximity radius
                            # Use predicted positions (TP + FP) to avoid data leakage
                            if verbose >= 1:
                                print(f"[info] Window-based TN sampling: collecting TNs adjacent to predicted donor sites")
                            
                            # Get PREDICTED donor positions (TP + FP) - no data leakage
                            predicted_donor_positions = [p['predicted_position'] for p in (tp_positions + fp_positions) 
                                                        if p.get('predicted_position') is not None]
                            
                            if predicted_donor_positions:
                                # Create windows around each PREDICTED donor site
                                window_tn_positions = set()  # Use set to avoid duplicates
                                
                                # Use tn_proximity_radius as the window size (typically 50)
                                window_size = tn_proximity_radius
                                
                                for pred_pos in predicted_donor_positions:
                                    # Define window boundaries
                                    window_start = max(0, pred_pos - window_size)
                                    window_end = min(gene_len, pred_pos + window_size + 1)
                                    
                                    # Collect all TN positions within this window
                                    for tn_pos in tn_positions_all:
                                        pos = tn_pos['position']
                                        if window_start <= pos <= window_end:
                                            window_tn_positions.add(pos)
                                
                                # Convert back to position objects
                                window_tn_list = [tn for tn in tn_positions_all if tn['position'] in window_tn_positions]
                                
                                # Apply sampling if we have more than desired
                                if len(window_tn_list) <= num_tn_to_sample:
                                    tn_collection = window_tn_list
                                else:
                                    # If we have too many window TNs, sample randomly within the windows
                                    tn_collection = random.sample(window_tn_list, num_tn_to_sample)
                                    
                                if verbose >= 1:
                                    print(f"[info] Collected {len(tn_collection)} TN positions from windows around {len(predicted_donor_positions)} predicted donor sites")
                            else:
                                # Fallback to random if no predicted positions found
                                if verbose >= 1:
                                    print(f"[warning] No predicted donor positions found for window sampling, falling back to random")
                                tn_collection = random.sample(tn_positions_all, num_tn_to_sample)
                        else:
                            # default fallback => random
                            tn_collection = random.sample(tn_positions_all, num_tn_to_sample)

        # Run position statistics analysis if verbose
        if verbose >= 1:
            analyze_transcript_position_stats(
                tp_positions, 
                fp_positions, 
                fn_positions, 
                tn_collection,
                prefix=f"Gene {gene_id} (Strand {strand})",
                verbose=(verbose > 1)  # Only show detailed stats if verbose is high enough
            )
        
        # ==========================================================
        # CRITICAL: Add all position types to the main positions_list
        # This step ensures all collected data points are included in the final output
        # NEVER modify this section without thorough testing
        # ==========================================================
        positions_list.extend(tp_positions)
        positions_list.extend(fp_positions)
        positions_list.extend(fn_positions)

        # Add per-gene TNs if requested
        if collect_tn and tn_collection:
            positions_list.extend(tn_collection)

        # Debugging: Display true donor positions and their surrounding probabilities
        if verbose:
            # Extend donor probabilities with neighbors
            neighbor_range = 2  # Extend by ±2 nucleotides
            true_donor_scores_with_neighbors = []
            
            # Track correctly predicted sites
            correctly_predicted_count = 0
            high_score_threshold = threshold  # Use same threshold as classification

            for pos in true_donor_positions:
                neighbor_scores = donor_probabilities[max(0, pos - neighbor_range):min(len(donor_probabilities), pos + neighbor_range + 1)]
                true_donor_scores_with_neighbors.append((pos, list(neighbor_scores)))
                
                # Count sites with score above threshold as correctly predicted
                if pos < len(donor_probabilities) and donor_probabilities[pos] >= high_score_threshold:
                    correctly_predicted_count += 1

            # Calculate percentage of correctly predicted sites
            total_true_sites = len(true_donor_positions)
            capture_ratio = (correctly_predicted_count / total_true_sites * 100) if total_true_sites > 0 else 0
            
            print(f"[analysis] Gene {gene_id} @ strand={strand}, chr={chromosome}:")
            print(f"  True donor positions (relative): {true_donor_positions}")
            print(f"  Donor probabilities with neighbors (±{neighbor_range} nts):")
            print(f"     Donor score adjusted? {predicted_delta_correction}")
            print(f"  Correctly captured: {correctly_predicted_count}/{total_true_sites} sites ({capture_ratio:.2f}%)")
            for pos, scores in true_donor_scores_with_neighbors:
                # Format each score in the array to 4 decimal places
                formatted_scores = [f"{score:.4f}" for score in scores]
                score_status = "✓" if pos < len(donor_probabilities) and donor_probabilities[pos] >= high_score_threshold else "✗"
                print(f"    Position {pos}: {formatted_scores} {score_status}")

        # Step 4: Calculate windowed regions for FPs and FNs
        if isinstance(error_window, int):
            error_window = (error_window, error_window)  # Symmetric window

        # Also build error_list with windows for FP/FN
        for pos in fp_positions + fn_positions:
            pos_type = pos['pred_type']
            strand = pos['strand']
            tx_id = pos['transcript_id']
            pos_window_start = max(0, pos['position'] - error_window[0])
            pos_window_end = min(gene_len, pos['position'] + error_window[1])
            error_list.append({
                'gene_id': pos['gene_id'],
                'transcript_id': tx_id,
                'error_type': pos_type,
                'position': pos['position'],
                'window_start': pos_window_start,
                'window_end': pos_window_end, 
                'strand': strand
            })

    # End for each gene    

    # Use provided schema or fall back to default
    position_schema = kargs.get("position_schema", DEFAULT_POSITION_SCHEMA)
    error_schema = kargs.get("error_schema", DEFAULT_ERROR_SCHEMA)
    
    # Ensure all entries have the same schema and handle None values consistently
    for entry in positions_list:
        # Ensure all fields exist
        for field in position_schema:
            if field not in entry:
                entry[field] = None
                
        # Ensure consistent types for transcript_id (None vs empty string)
        if entry['transcript_id'] is None:
            entry['transcript_id'] = ""  # Use empty string instead of None

    # DEBUG: Check if ENSG00000104435 has donor records in positions_list
    # debug_gene = "ENSG00000104435"
    # debug_records = [p for p in positions_list if p.get('gene_id') == debug_gene]
    # debug_donor_records = [p for p in debug_records if p.get('splice_type') == 'donor']
    # if debug_records:
    #     print(f"\n[DEBUG] Found {len(debug_records)} records for gene {debug_gene}")
    #     print(f"[DEBUG] Found {len(debug_donor_records)} donor records for gene {debug_gene}")
    #     if debug_donor_records:
    #         print(f"[DEBUG] Sample donor record: {debug_donor_records[0]}")
    #     # Check complete set of splice_types for this gene
    #     splice_types = set(p.get('splice_type') for p in debug_records)
    #     print(f"[DEBUG] Splice types for {debug_gene}: {splice_types}")
    # else:
    #     print(f"\n[DEBUG] No records found for gene {debug_gene} in positions_list")
    
    # DEBUG: Check length of positions_list
    # print(f"[DEBUG] Total records in positions_list: {len(positions_list)}")
    # print(f"[DEBUG] Position schema: {position_schema}")

    # Create a DataFrame from the position data
    if positions_list:  # If we found any positions
        # print(f"[DEBUG] Total records in positions_list: {len(positions_list)}")
        # print(f"[DEBUG] Position schema: {position_schema}")
        
        # Debug - check if context columns exist in the positions_list
        sample_position = positions_list[0] if positions_list else {}
        context_keys = [k for k in sample_position.keys() if k.startswith('context_')]
        # print(f"[DEBUG] Context keys in first position: {context_keys}")
        if context_keys:
            # print(f"[DEBUG] Example context values: {[(k, sample_position[k]) for k in context_keys[:3]]}")
            pass
        
        # Extract all column names that appear in the positions_list
        all_columns = set()
        for pos in positions_list:
            all_columns.update(pos.keys())
            
        # Create a more flexible schema that preserves context columns
        extended_schema = position_schema.copy()
        for col in all_columns:
            if col not in extended_schema and col.startswith('context_'):
                # Add context columns to the schema as Float64
                extended_schema[col] = pl.Float64
                
        # print(f"[DEBUG] Extended schema with {len(extended_schema) - len(position_schema)} additional columns")
        # print(f"[DEBUG] Added columns: {[col for col in extended_schema if col not in position_schema]}")
        
        # Use the extended schema instead of the default
        positions_df = pl.DataFrame(positions_list, schema=extended_schema)
        
        # Verify the resulting DataFrame has the context columns
        df_context_cols = [col for col in positions_df.columns if col.startswith('context_')]
        print(f"[DEBUG] Context columns in DataFrame: {df_context_cols}")
    else:
        positions_df = pl.DataFrame(schema=position_schema)

    # Convert error list to DataFrame
    # Use infer_schema_length=None to scan all rows for consistent schema inference
    if error_list:
        error_df = pl.DataFrame(error_list, infer_schema_length=None)
        error_df = ensure_schema(error_df, error_schema)
    else:
        error_df = pl.DataFrame(schema=error_schema)
    
    if return_positions_df:
        return error_df, positions_df
    
    return error_df


def enhanced_evaluate_acceptor_site_errors(
    annotations_df, 
    pred_results, 
    threshold=0.5, 
    consensus_window=2, 
    error_window=500, 
    collect_tn=True,
    tn_sample_factor=1.2,
    tn_sampling_mode="random",
    tn_proximity_radius=50,
    no_tn_sampling=False,
    predicted_delta_correction=False,  # Add as explicit parameter with default False
    splice_site_adjustments=None,
    return_positions_df=True,
    **kargs
):
    """
    Evaluate SpliceAI predictions for acceptor splice sites and identify TPs, FPs, FNs,
    optionally also collecting a subset of TNs.

    Similar to the donor-site logic, but adapted for acceptor sites.

    Parameters
    ----------
    annotations_df : pl.DataFrame
        Contains true splice site annotations.
        Columns typically: ['chrom', 'start', 'end', 'strand', 'site_type', 'gene_id', 'transcript_id']
    pred_results : dict
        Output of predict_splice_sites_for_genes(), containing per-nucleotide acceptor probabilities
        plus gene metadata (e.g. gene_start, gene_end, strand).
    threshold : float
        Threshold for classifying a position as an acceptor site (default 0.5).
    consensus_window : int
        Tolerance window around true splice sites for matching (TP or FN).
    error_window : int or tuple
        Window size for collecting the local sequence region around FPs or FNs.
        If int, we treat it as (error_window, error_window).
    collect_tn : bool
        Whether to sample a subset of TNs. Defaults to False.
    tn_sample_factor : float
        Sampling ratio for true negatives, by default 1.2.
    tn_sampling_mode : str
        "random" => random sampling of TNs
        "proximity" => prefer TN near TPs/FNs (within tn_proximity_radius).
        "window" => collect TNs adjacent to true splice sites within error_window
    tn_proximity_radius : int
        Radius for proximity mode.
    no_tn_sampling : bool
        If True, preserve all TN positions without sampling. If False, apply sampling based on tn_sample_factor.
    predicted_delta_correction : bool, optional
        Whether to apply adjustment to SpliceAI predictions. Defaults to False.
    splice_site_adjustments : dict, optional
        Dictionary with custom adjustments to apply to SpliceAI predictions. If provided, 
        this is used instead of the standard SpliceAI adjustments when predicted_delta_correction is True.
        Format: {'donor': {'plus': offset, 'minus': offset}, 'acceptor': {'plus': offset, 'minus': offset}}
    return_positions_df : bool, optional
        Whether to return the positions DataFrame, by default False.

    Returns:
    -------
    
    Tuple[pl.DataFrame, pl.DataFrame]
        Tuple of (error_df, positions_df).

    - error_df (pl.DataFrame): A DataFrame containing positions of FPs and FNs, each with a window around the error.
    
      Columns: ['gene_id', 'transcript_id', 'error_type', 'position', 'window_start',
                  'window_end', 'strand']
    
    - positions_df (pl.DataFrame): A DataFrame with all positions and their three probability scores.

      Columns: ['gene_id', 'transcript_id', 'position', 'pred_type', 'score', 'strand']
    """
    # from .extract_genomic_features import get_overlapping_gene_metadata

    verbose = kargs.get('verbose', 1)
    chromosome = kargs.get('chromosome', kargs.get('chr', '?'))
    overlapping_genes_metadata = kargs.get('overlapping_genes_metadata', None)
    adjust_for_overlapping_genes = kargs.get('adjust_for_overlapping_genes', True)
    # REMOVE THIS LINE - it's overriding the parameter with a default of True
    # predicted_delta_correction = kargs.get('predicted_delta_correction', True)

    if overlapping_genes_metadata is None:
        sa = SpliceAnalyzer()
        overlapping_genes_metadata = sa.retrieve_overlapping_gene_metadata()

    error_list = []
    positions_list = []

    # Group true acceptor annotations by gene
    grouped_annotations = (
        annotations_df
        .filter(pl.col('splice_type') == 'acceptor')
        .group_by('gene_id')
        .agg(pl.struct(['start', 'end', 'position', 'transcript_id']).alias('acceptor_sites'))
        .to_dict(as_series=False)  # => { 'gene_id': [...], 'acceptor_sites': [...]}
    )

    # Loop over each gene in pred_results
    for gene_id, gene_data in pred_results.items():

        strand = normalize_strand(gene_data['strand'])

        if gene_id not in grouped_annotations['gene_id']:
            # No acceptor annotations for this gene => store placeholders
            if verbose:
                print(f"No acceptor annotations for gene: {gene_id}")

            # error_list.append({
            #     'gene_id': gene_id,
            #     'transcript_id': None,
            #     'error_type': None,  # no acceptor annotation
            #     'position': None,
            #     'window_start': None,
            #     'window_end': None,
            #     'strand': strand
            # })
            # positions_list.append({
            #     'gene_id': gene_id,
            #     'transcript_id': None,
            #     'position': None,
            #     'pred_type': None,
            #     'score': 0.0,
            #     'strand': strand,
            #     'donor_score': 0.0,
            #     'acceptor_score': 0.0,
            #     'neither_score': 0.0,
            #     'splice_type': None
            # })
            continue

        # Extract acceptor annotations
        # strand = normalize_strand(gene_data['strand'])
        acceptor_sites = grouped_annotations['acceptor_sites'][grouped_annotations['gene_id'].index(gene_id)]

        # Build up true acceptor positions (relative coords) + transcript IDs
        true_acceptor_positions = []
        position_to_transcript = defaultdict(set)

        for site in acceptor_sites:
            if strand == '+':
                relative_position = site['position'] - gene_data['gene_start']
            elif strand == '-':
                relative_position = gene_data['gene_end'] - site['position']
            else:
                raise ValueError(f"Invalid strand value: {strand}")

            true_acceptor_positions.append(relative_position)
            position_to_transcript[relative_position].add(site['transcript_id'])

        true_acceptor_positions = np.array(sorted(set(true_acceptor_positions)))
        position_to_transcript = {
            pos: list(tids) for pos, tids in position_to_transcript.items()
        }

        # Align probabilities
        acceptor_probabilities = np.array(gene_data['acceptor_prob'])
        donor_probabilities = np.array(gene_data['donor_prob'])
        neither_probabilities = np.array(gene_data['neither_prob'])

        if predicted_delta_correction:
            if splice_site_adjustments is not None:
                # Apply the same position adjustments to all three probability arrays
                acceptor_probabilities = apply_custom_splice_site_adjustments(
                    acceptor_probabilities, strand, 'acceptor', splice_site_adjustments
                )
                # Apply identical position shifts to donor and neither probabilities
                donor_probabilities = apply_custom_splice_site_adjustments(
                    donor_probabilities, strand, 'acceptor', splice_site_adjustments
                )
                neither_probabilities = apply_custom_splice_site_adjustments(
                    neither_probabilities, strand, 'acceptor', splice_site_adjustments, is_neither_prob=True
                )
            else:
                # Apply the same adjustments to all three probability arrays
                acceptor_probabilities = adjust_scores(acceptor_probabilities, strand, 'acceptor')
                # Use 'acceptor' as site_type for all arrays to maintain consistent shifting
                donor_probabilities = adjust_scores(donor_probabilities, strand, 'acceptor')
                neither_probabilities = adjust_scores(neither_probabilities, strand, 'acceptor', is_neither_prob=True)
            
            # Verify probability sums after adjustment
            verify_probability_sums(acceptor_probabilities, donor_probabilities, neither_probabilities, 
                                    strand, 'acceptor', splice_site_adjustments, gene_id=gene_id, verbose=verbose)
            
        gene_len = len(acceptor_probabilities)
        if len(true_acceptor_positions) > 0:
            assert true_acceptor_positions.max() < gene_len, (
                "true_acceptor_positions contain indices out of range."
            )

        # For debugging only, if you want:
        true_acceptor_scores = acceptor_probabilities[true_acceptor_positions] if len(true_acceptor_positions) > 0 else []
        # Round or do something if you want to print them
        # true_acceptor_scores = np.round(true_acceptor_scores, 3)

        # Binarize predictions at each position
        label_predictions = np.array(
            [1 if p >= threshold else 0 for p in acceptor_probabilities]
        )

        # For debugging or local reference
        gene_results = {'gene_id': gene_id, 'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}
        # Track which acceptor sites got matched => so we know which are FN
        true_acceptor_status = {idx: False for idx in range(len(true_acceptor_positions))}

        # local containers
        fp_positions, fn_positions, tp_positions = [], [], []
        tn_positions_all = []  # All candidate TN => we will sample some subset if collect_tn=True

        # Evaluate each position
        for i, pred_label in enumerate(label_predictions):
            found_in_window = False

            # Check if this is near a real acceptor site
            for idx, true_pos in enumerate(true_acceptor_positions):
                if true_acceptor_status[idx]:  # Already matched
                    continue

                wstart = true_pos - consensus_window
                wend   = true_pos + consensus_window

                if wstart <= i <= wend:
                    found_in_window = True
                    
                    # If pred_label=1 => TP
                    if pred_label == 1:
                        gene_results['TP'] += 1
                        true_acceptor_status[idx] = True  # Mark as matched

                        # Use the maximum probability within the window instead of just at position i
                        # --- robustly find max prob within window (avoid empty slice) ---
                        slice_start = max(0, wstart)
                        slice_end   = min(len(acceptor_probabilities), wend + 1)
                        window_probs = acceptor_probabilities[slice_start:slice_end]
                        if window_probs.size == 0:
                            max_prob_idx = i  # fallback to current position
                        else:
                            max_prob_idx = slice_start + np.argmax(window_probs)
                        score = acceptor_probabilities[max_prob_idx]

                        assert score >= threshold, \
                            f"Unexpected score for TP: {score} < {threshold}"

                        associated_tids = position_to_transcript[true_pos]

                        # Get context scores (surrounding positions)
                        context_scores = get_context_scores(acceptor_probabilities, i, consensus_window)

                        # store them
                        for t_id in associated_tids:
                            tp_positions.append({
                                'gene_id': gene_id,
                                'transcript_id': t_id,  # Transcript that uses the splice site
                                'position': i,  # Current position
                                'predicted_position': i,  # i should be within the window of true_pos
                                'true_position': true_pos,  # True position of the acceptor site
                                'pred_type': 'TP',
                                'score': score,
                                'strand': strand,
                                'donor_score': donor_probabilities[i],
                                'acceptor_score': score,  # acceptor_probabilities[i],
                                'neither_score': neither_probabilities[i],
                                # Add surrounding context scores for meta-modeling
                                'context_score_m2': context_scores[0] if len(context_scores) > 0 else 0.0,
                                'context_score_m1': context_scores[1] if len(context_scores) > 1 else 0.0,
                                'context_score_p1': context_scores[3] if len(context_scores) > 3 else 0.0,
                                'context_score_p2': context_scores[4] if len(context_scores) > 4 else 0.0,
                                'splice_type': 'acceptor'
                            })

                    break

            # If not in any real acceptor window => either FP or TN
            if not found_in_window:
                if pred_label == 1:
                    # => predicted acceptor but not a real site => FP
                    is_fp = True
                    if (adjust_for_overlapping_genes and
                        is_within_overlapping_splice_site(
                            i, consensus_window, gene_id, gene_data,
                            overlapping_genes_metadata, grouped_annotations
                        )):
                        # Overlapping gene logic can "un-flag" it as FP
                        is_fp = False

                    if is_fp:
                        gene_results['FP'] += 1
                        # Get context scores (surrounding positions)
                        context_scores = get_context_scores(acceptor_probabilities, i, consensus_window)

                        fp_positions.append({
                            'gene_id': gene_id,
                            'transcript_id': None,  # Irrelevant since it's not a splice site
                            'position': i,  # Current position
                            'predicted_position': i,
                            'true_position': None,  # Not a splice site
                            'pred_type': 'FP',
                            'score': acceptor_probabilities[i],
                            'strand': strand,
                            'donor_score': donor_probabilities[i],
                            'acceptor_score': acceptor_probabilities[i],
                            'neither_score': neither_probabilities[i],
                            # Add surrounding context scores for meta-modeling
                            'context_score_m2': context_scores[0] if len(context_scores) > 0 else 0.0,
                            'context_score_m1': context_scores[1] if len(context_scores) > 1 else 0.0,
                            'context_score_p1': context_scores[3] if len(context_scores) > 3 else 0.0,
                            'context_score_p2': context_scores[4] if len(context_scores) > 4 else 0.0,
                            'splice_type': None  # Not a splice site
                        })

                else:
                    # This is a negative prediction => candidate TN
                    # We'll store it and sample later
                    # Get context scores (surrounding positions)
                    context_scores = get_context_scores(acceptor_probabilities, i, consensus_window)
                    
                    tn_positions_all.append({
                        'gene_id': gene_id,
                        'transcript_id': None,  # Irrelevant since it's not a splice site
                        'position': i,  # Current position
                        'predicted_position': None,  # predicted to be non-splice site (TN)
                        'true_position': None,  # Not a splice site by ground truth
                        'pred_type': 'TN',
                        'score': acceptor_probabilities[i],
                        'strand': strand,
                        'donor_score': donor_probabilities[i],
                        'acceptor_score': acceptor_probabilities[i],
                        'neither_score': neither_probabilities[i],
                        # Add surrounding context scores for meta-modeling
                        'context_score_m2': context_scores[0] if len(context_scores) > 0 else 0.0,
                        'context_score_m1': context_scores[1] if len(context_scores) > 1 else 0.0,
                        'context_score_p1': context_scores[3] if len(context_scores) > 3 else 0.0,
                        'context_score_p2': context_scores[4] if len(context_scores) > 4 else 0.0,
                        'splice_type': None  # Not a splice site
                    })

                    # Predicted negative correctly => TN
                    gene_results['TN'] += 1
        # End for each position

        # Now check for FN (i.e. real acceptor that never got labeled=1)
        d = 0
        for idx, matched in true_acceptor_status.items():
            if not matched:
                # => FN
                position = true_acceptor_positions[idx]
                score = acceptor_probabilities[position]
                label_val = label_predictions[position]
                gene_results['FN'] += 1

                associated_tids = position_to_transcript[position]

                rng_start = max(0, position - consensus_window)
                rng_end   = min(gene_len, position + consensus_window + 1)
                neighbors = acceptor_probabilities[rng_start:rng_end]
                if verbose:
                    if d < 2: 
                        print(f"[DEBUG] FN: gene_id={gene_id}, pos={position}, "
                          f"score={score:.4f}, label={label_val}, threshold={threshold}")
                    # Format array elements with 4 decimal places
                    formatted_scores = [f"{score:.4f}" for score in neighbors]
                    print_with_indent(f"Surrounding acceptor scores at pos={position}: {formatted_scores}", indent_level=1)
                    d += 1

                # Should we use the max score within the window instead of just at position i?
                score = max(neighbors)

                # Extract context scores (surrounding positions)
                context_scores = get_context_scores(acceptor_probabilities, position, consensus_window)

                for t_id in associated_tids:
                    fn_positions.append({
                        'gene_id': gene_id,
                        'transcript_id': t_id,
                        'position': position,
                        'predicted_position': None,  # Falsely predicted negative
                        'true_position': position,  # True position of the acceptor site
                        'pred_type': 'FN',
                        'score': score,
                        'strand': strand,
                        'donor_score': donor_probabilities[position],
                        'acceptor_score': score,  # cceptor_probabilities[position]
                        'neither_score': neither_probabilities[position],
                        # Add surrounding context scores for meta-modeling
                        'context_score_m2': context_scores[0] if len(context_scores) > 0 else 0.0,
                        'context_score_m1': context_scores[1] if len(context_scores) > 1 else 0.0,
                        'context_score_p1': context_scores[3] if len(context_scores) > 3 else 0.0,
                        'context_score_p2': context_scores[4] if len(context_scores) > 4 else 0.0,
                        'splice_type': 'acceptor'
                    })

        # Sample TN positions if needed
        tn_collection = []  # Initialize for this gene
        
        if collect_tn and len(tn_positions_all) > 0:
            if no_tn_sampling:
                # No sampling mode - preserve all TN positions
                tn_collection = tn_positions_all
                if verbose >= 1:
                    print(f"[info] No TN sampling: preserving all {len(tn_positions_all)} TN positions for gene {gene_id}")
            else:
                # Apply sampling based on tn_sample_factor
                # Count how many TPs+FPs+FNs we got for this gene
                num_tp_fp_fn = len(tp_positions) + len(fp_positions) + len(fn_positions)
                num_tn_to_sample = min(int(num_tp_fp_fn * tn_sample_factor), len(tn_positions_all))

                if num_tn_to_sample > 0:
                    if len(tn_positions_all) <= num_tn_to_sample:
                        tn_collection = tn_positions_all
                    else:
                        # sampling
                        if tn_sampling_mode == "random":
                            tn_collection = random.sample(tn_positions_all, num_tn_to_sample)

                        elif tn_sampling_mode == "proximity":
                            # Prefer positions near PREDICTED splice sites (TP + FP)
                            # Use predicted positions to avoid data leakage
                            predicted_positions = [p['predicted_position'] for p in (tp_positions + fp_positions) 
                                                 if p.get('predicted_position') is not None]
                            
                            if predicted_positions:
                                tn_with_dists = []
                                for tn_row in tn_positions_all:
                                    min_dist = min(abs(tn_row['position'] - pred_pos) for pred_pos in predicted_positions)
                                    tn_with_dists.append((tn_row, min_dist))
                                # sort ascending
                                tn_with_dists.sort(key=lambda x: x[1])
                                tn_collection = [x[0] for x in tn_with_dists[:num_tn_to_sample]]
                            else:
                                # fallback random
                                tn_collection = random.sample(tn_positions_all, num_tn_to_sample)

                        elif tn_sampling_mode == "window":
                            # Collect TNs adjacent to PREDICTED acceptor sites within proximity radius
                            # Use predicted positions (TP + FP) to avoid data leakage
                            if verbose >= 1:
                                print(f"[info] Window-based TN sampling: collecting TNs adjacent to predicted acceptor sites")
                            
                            # Get PREDICTED acceptor positions (TP + FP) - no data leakage
                            predicted_acceptor_positions = [p['predicted_position'] for p in (tp_positions + fp_positions) 
                                                           if p.get('predicted_position') is not None]
                            
                            if predicted_acceptor_positions:
                                # Create windows around each PREDICTED acceptor site
                                window_tn_positions = set()  # Use set to avoid duplicates
                                
                                # Use tn_proximity_radius as the window size (typically 50)
                                window_size = tn_proximity_radius
                                
                                for pred_pos in predicted_acceptor_positions:
                                    # Define window boundaries
                                    window_start = max(0, pred_pos - window_size)
                                    window_end = min(gene_len, pred_pos + window_size + 1)
                                    
                                    # Collect all TN positions within this window
                                    for tn_pos in tn_positions_all:
                                        pos = tn_pos['position']
                                        if window_start <= pos <= window_end:
                                            window_tn_positions.add(pos)
                                
                                # Convert back to position objects
                                window_tn_list = [tn for tn in tn_positions_all if tn['position'] in window_tn_positions]
                                
                                # Apply sampling if we have more than desired
                                if len(window_tn_list) <= num_tn_to_sample:
                                    tn_collection = window_tn_list
                                else:
                                    # If we have too many window TNs, sample randomly within the windows
                                    tn_collection = random.sample(window_tn_list, num_tn_to_sample)
                                    
                                if verbose >= 1:
                                    print(f"[info] Collected {len(tn_collection)} TN positions from windows around {len(predicted_acceptor_positions)} predicted acceptor sites")
                            else:
                                # Fallback to random if no predicted positions found
                                if verbose >= 1:
                                    print(f"[warning] No predicted acceptor positions found for window sampling, falling back to random")
                                tn_collection = random.sample(tn_positions_all, num_tn_to_sample)

                        else:
                            # default fallback => random
                            tn_collection = random.sample(tn_positions_all, num_tn_to_sample)

        # Run position statistics analysis if verbose
        if verbose >= 1:
            analyze_transcript_position_stats(
                tp_positions, 
                fp_positions, 
                fn_positions, 
                tn_collection,
                prefix=f"Gene {gene_id} (Strand {strand})",
                verbose=(verbose > 1)  # Only show detailed stats if verbose is high enough
            )
        
        # ==========================================================
        # CRITICAL: Add all position types to the main positions_list
        # This step ensures all collected data points are included in the final output
        # NEVER modify this section without thorough testing
        # ==========================================================
        positions_list.extend(tp_positions)
        positions_list.extend(fp_positions)
        positions_list.extend(fn_positions)

        # Add per-gene TNs if requested
        if collect_tn and tn_collection:
            positions_list.extend(tn_collection)

        # Debugging info for the neighbor distribution
        if verbose:
            neighbor_range = 2  # or consensus_window
            
            # Track correctly predicted sites
            correctly_predicted_count = 0
            high_score_threshold = threshold  # Use same threshold as classification
            
            print(f"[analysis] Gene {gene_id} @ strand={strand}, chr={chromosome}:")
            print(f"  True acceptor positions (relative): {true_acceptor_positions}")
            
            if len(true_acceptor_positions) > 0:
                acceptor_scores_with_neighbors = []
                for pos in true_acceptor_positions:
                    n_start = max(0, pos - neighbor_range)
                    n_end   = min(gene_len, pos + neighbor_range + 1)
                    neighbor_scores = list(acceptor_probabilities[n_start:n_end])
                    acceptor_scores_with_neighbors.append((pos, neighbor_scores))
                    
                    # Count sites with score above threshold as correctly predicted
                    if pos < len(acceptor_probabilities) and acceptor_probabilities[pos] >= high_score_threshold:
                        correctly_predicted_count += 1
                
                # Calculate percentage of correctly predicted sites
                total_true_sites = len(true_acceptor_positions)
                capture_ratio = (correctly_predicted_count / total_true_sites * 100) if total_true_sites > 0 else 0
                
                print(f"  Acceptor probabilities ±{neighbor_range} nt around each real site:")
                print(f"     Acceptor score adjusted? {predicted_delta_correction}")
                print(f"  Correctly captured: {correctly_predicted_count}/{total_true_sites} sites ({capture_ratio:.2f}%)")
                
                for ppos, scarr in acceptor_scores_with_neighbors:
                    # Format each score in the array to 4 decimal places
                    formatted_scores = [f"{score:.4f}" for score in scarr]
                    score_status = "✓" if ppos < len(acceptor_probabilities) and acceptor_probabilities[ppos] >= high_score_threshold else "✗"
                    print(f"    Position {ppos}: {formatted_scores} {score_status}")
        
        # Build error_list with windowed FPs and FNs
        if isinstance(error_window, int):
            error_window = (error_window, error_window)

        for pos_row in (fp_positions + fn_positions):
            pos_type = pos_row['pred_type']
            strand_ = pos_row['strand']
            tx_ = pos_row['transcript_id']
            wstart = max(0, pos_row['position'] - error_window[0])
            wend   = min(gene_len, pos_row['position'] + error_window[1])
            error_list.append({
                'gene_id': pos_row['gene_id'],
                'transcript_id': tx_,
                'error_type': pos_type,
                'position': pos_row['position'],
                'window_start': wstart,
                'window_end': wend,
                'strand': strand_
            })

    # End for each gene

    # Use provided schema or fall back to default
    position_schema = kargs.get("position_schema", DEFAULT_POSITION_SCHEMA)
    error_schema = kargs.get("error_schema", DEFAULT_ERROR_SCHEMA)
    
    # Ensure all entries have the same schema and handle None values consistently
    for entry in positions_list:
        # Ensure all fields exist
        for field in position_schema:
            if field not in entry:
                entry[field] = None
                
        # Ensure consistent types for transcript_id (None vs empty string)
        if entry['transcript_id'] is None:
            entry['transcript_id'] = ""  # Use empty string instead of None
    
    # DEBUG: Check length of positions_list
    # print(f"[DEBUG] Total records in positions_list: {len(positions_list)}")
    # print(f"[DEBUG] Position schema: {position_schema}")

    # Create a DataFrame from the position data
    if positions_list:  # If we found any positions
        # print(f"[DEBUG] Total records in positions_list: {len(positions_list)}")
        # print(f"[DEBUG] Position schema: {position_schema}")
        
        # Debug - check if context columns exist in the positions_list
        sample_position = positions_list[0] if positions_list else {}
        context_keys = [k for k in sample_position.keys() if k.startswith('context_')]
        # print(f"[DEBUG] Context keys in first position: {context_keys}")
        if context_keys:
            # print(f"[DEBUG] Example context values: {[(k, sample_position[k]) for k in context_keys[:3]]}")
            pass
        
        # Extract all column names that appear in the positions_list
        all_columns = set()
        for pos in positions_list:
            all_columns.update(pos.keys())
            
        # Create a more flexible schema that preserves context columns
        extended_schema = position_schema.copy()
        for col in all_columns:
            if col not in extended_schema and col.startswith('context_'):
                # Add context columns to the schema as Float64
                extended_schema[col] = pl.Float64
                
        # print(f"[DEBUG] Extended schema with {len(extended_schema) - len(position_schema)} additional columns")
        # print(f"[DEBUG] Added columns: {[col for col in extended_schema if col not in position_schema]}")
        
        # Use the extended schema instead of the default
        positions_df = pl.DataFrame(positions_list, schema=extended_schema)
        
        # Verify the resulting DataFrame has the context columns
        df_context_cols = [col for col in positions_df.columns if col.startswith('context_')]
        print(f"[DEBUG] Context columns in DataFrame: {df_context_cols}")
    else:
        positions_df = pl.DataFrame(schema=position_schema)

    # Convert error list to DataFrame
    # Use infer_schema_length=None to scan all rows for consistent schema inference
    if error_list:
        error_df = pl.DataFrame(error_list, infer_schema_length=None)
        error_df = ensure_schema(error_df, error_schema)
    else:
        error_df = pl.DataFrame(schema=error_schema)
    
    if return_positions_df:
        return error_df, positions_df

    return error_df

########################################################################################


def enhanced_evaluate_splice_site_errors(
    annotations_df, 
    pred_results, 
    threshold=0.5, 
    consensus_window=2, 
    error_window=500, 
    collect_tn=True,
    tn_sample_factor=1.2,
    tn_sampling_mode="random",
    tn_proximity_radius=50,
    no_tn_sampling=False,
    predicted_delta_correction=False,
    splice_site_adjustments=None,
    return_positions_df=True,
    verbose=1,
    **kargs
):
    """
    Enhanced evaluation of SpliceAI predictions for both donor and acceptor splice sites.
    This function combines the donor and acceptor site evaluations, preserving all three
    probability scores (donor, acceptor, neither) for each position.
    
    Parameters:
    ----------
    annotations_df : pl.DataFrame
        DataFrame with columns 'transcript_id', 'gene_id', 'site_type', 'position'.
        Used for evaluating errors against ground truth.
    pred_results : dict
        Dictionary with prediction results.
    threshold : float, optional
        Threshold for classifying a position as a splice site, by default 0.5.
    consensus_window : int, optional
        Size of window to search for nearest prediction, by default 2.
    error_window : int or tuple, optional
        Size of window to search for nearest prediction, by default 500.
    collect_tn : bool, optional
        Whether to collect true negative positions, by default False.
    tn_sample_factor : float, optional
        Sampling ratio for true negatives, by default 1.2.
    tn_sampling_mode : str, optional
        Mode for sampling true negatives, by default "random".
        Options: "random", "proximity", "window"
    tn_proximity_radius : int, optional
        Radius for proximity-based TN sampling, by default 50.
    no_tn_sampling : bool, optional
        If True, preserve all TN positions without sampling. If False, apply sampling based on tn_sample_factor, by default False.
    predicted_delta_correction : bool, optional
        Whether to apply adjustment to SpliceAI predictions, by default False.
    splice_site_adjustments : dict, optional
        Dictionary with custom adjustments to apply to SpliceAI predictions. If provided, 
        this is used instead of the standard SpliceAI adjustments when predicted_delta_correction is True.
        Format: {'donor': {'plus': offset, 'minus': offset}, 'acceptor': {'plus': offset, 'minus': offset}}
    return_positions_df : bool, optional
        Whether to return the positions DataFrame, by default True.
    verbose : int, optional
        Level of verbosity, by default 1.
    **kargs : dict
        Additional arguments.
    
    Returns:
    -------
    
    Tuple[pl.DataFrame, pl.DataFrame]
        Tuple of (error_df, positions_df).

    - error_df (pl.DataFrame): A DataFrame containing positions of FPs and FNs, with window coordinates.
    - positions_df (pl.DataFrame): A DataFrame with all positions and their three probability scores.
    """
    # Validate threshold
    if threshold <= 0 or threshold >= 1:
        raise ValueError(f"Threshold must be between 0 and 1, got {threshold}")
    
    # Handle error window
    if isinstance(error_window, (list, tuple)):
        if len(error_window) != 2:
            raise ValueError(f"error_window must be an int or a tuple of 2 ints, got {error_window}")
        donor_error_window, acceptor_error_window = error_window
    else:
        donor_error_window = acceptor_error_window = error_window

    # Define the expected schema for combined DataFrames
    error_schema = DEFAULT_ERROR_SCHEMA

    # Define a consistent schema
    position_schema = DEFAULT_POSITION_SCHEMA
    
    # Evaluate donor site errors with all three probability scores
    donor_error_df, donor_positions_df = enhanced_evaluate_donor_site_errors(
        annotations_df, 
        pred_results, 
        threshold=threshold,
        consensus_window=consensus_window, 
        error_window=error_window, 
        collect_tn=collect_tn,
        tn_sample_factor=tn_sample_factor,
        tn_sampling_mode=tn_sampling_mode,
        tn_proximity_radius=tn_proximity_radius,
        no_tn_sampling=no_tn_sampling,
        predicted_delta_correction=predicted_delta_correction,
        splice_site_adjustments=splice_site_adjustments,
        return_positions_df=True,
        position_schema=position_schema,
        error_schema=error_schema,
        **kargs
    )

    print("[debug] donor_error_df shape: ", donor_error_df.shape)
    print("[debug] donor_positions_df shape: ", donor_positions_df.shape)
    
    # Add a column to indicate the splice type as 'donor'
    if 'splice_type' not in donor_error_df.columns:
        donor_error_df = donor_error_df.with_columns(
            pl.lit("donor").alias("splice_type")
        )
    
    # Add a 'splice_type' column if it doesn't already exist
    if 'splice_type' not in donor_positions_df.columns:
        donor_positions_df = donor_positions_df.with_columns(
            pl.lit("donor").alias("splice_type")
        )

    # Evaluate acceptor site errors with all three probability scores
    acceptor_error_df, acceptor_positions_df = enhanced_evaluate_acceptor_site_errors(
        annotations_df, 
        pred_results, 
        threshold=threshold,
        consensus_window=consensus_window,
        error_window=error_window, 
        collect_tn=collect_tn,
        tn_sample_factor=tn_sample_factor,
        tn_sampling_mode=tn_sampling_mode,
        tn_proximity_radius=tn_proximity_radius,
        no_tn_sampling=no_tn_sampling,
        predicted_delta_correction=predicted_delta_correction,
        splice_site_adjustments=splice_site_adjustments,
        return_positions_df=True,
        position_schema=position_schema,
        error_schema=error_schema,
        **kargs
    )

    print("[debug] acceptor_error_df shape: ", acceptor_error_df.shape)
    print("[debug] acceptor_positions_df shape: ", acceptor_positions_df.shape)
    
    # Add a column to indicate the splice type as 'acceptor'
    if 'splice_type' not in acceptor_error_df.columns:
        acceptor_error_df = acceptor_error_df.with_columns(
            pl.lit("acceptor").alias("splice_type")
        )
    
    # Add a 'splice_type' column if it doesn't already exist
    if 'splice_type' not in acceptor_positions_df.columns:
        acceptor_positions_df = acceptor_positions_df.with_columns(
            pl.lit("acceptor").alias("splice_type")
        )

    # Consolidate donor and acceptor error dataframes
    print_emphasized("Consolidating donor and acceptor error dataframes ...")
    if not is_dataframe_empty(donor_error_df) and not is_dataframe_empty(acceptor_error_df):
        # Use the utility function to ensure compatible schemas for stacking
        # Define columns that should be treated as strings to handle null/string inconsistencies
        string_columns = ['transcript_id', 'gene_id', 'gene_name', 'error_type', 'splice_type']
        
        # Prepare the dataframes for stacking
        prepared_dfs = prepare_dataframes_for_stacking(
            dataframes=[donor_error_df, acceptor_error_df],
            string_columns=string_columns,
            verbose=verbose
        )
        
        # Stack the prepared dataframes
        try:
            if len(prepared_dfs) == 2:
                error_df = prepared_dfs[0].vstack(prepared_dfs[1])
            elif len(prepared_dfs) == 1:
                error_df = prepared_dfs[0]
            else:
                # Fallback to an empty DataFrame with the error schema
                error_df = pl.DataFrame(schema=error_schema)
        except Exception as e:
            print(f"\n[ERROR] Failed to stack error dataframes: {e}")
            
            # Use the schema mismatch analysis tool to diagnose the issue
            from meta_spliceai.splice_engine.meta_models.core.schema_utils import analyze_schema_mismatch
            
            # Analyze the specific schema mismatches
            mismatch = analyze_schema_mismatch(
                donor_error_df, 
                acceptor_error_df, 
                name1="donor_error_df", 
                name2="acceptor_error_df"
            )
            
            # Output detailed diagnostics if verbose
            if verbose > 0:
                print(f"\n[DIAGNOSTIC] Schema mismatch analysis:")
                print(f"  Columns only in donor: {mismatch['columns_only_in_df1']}")
                print(f"  Columns only in acceptor: {mismatch['columns_only_in_df2']}")
                print(f"  Type mismatches: {mismatch['type_mismatches']}")
                
                if verbose > 1 and mismatch['mismatches_detail']:
                    print("\n[DETAILED DIAGNOSTICS]")
                    for detail in mismatch['mismatches_detail']:
                        print(f"  Column '{detail['column']}' mismatch:")
                        print(f"    donor type: {detail['donor_error_df_type']}")
                        print(f"    acceptor type: {detail['acceptor_error_df_type']}")
                        print(f"    is null/string issue: {detail['null_string_issue']}")
            
            print(f"\n[RECOMMENDATION] {mismatch['recommendation']}")
    elif not is_dataframe_empty(donor_error_df):
        error_df = ensure_schema(donor_error_df, error_schema)
    elif not is_dataframe_empty(acceptor_error_df):
        error_df = ensure_schema(acceptor_error_df, error_schema)
    else:
        error_df = pl.DataFrame(schema=error_schema)

    # Consolidate donor and acceptor positions dataframes
    print_emphasized("Consolidating donor and acceptor positions dataframes ...")
    if not is_dataframe_empty(donor_positions_df) and not is_dataframe_empty(acceptor_positions_df):
        # Define columns that should be treated as strings to handle null/string inconsistencies
        # Include ID columns and categorical columns that might have nulls
        string_columns = [
            'gene_id', 'transcript_id', 'splice_type', 'pred_type', 'chrom', 'strand'
        ]
        
        # Also define columns that should be integers (even with nulls)
        # This ensures consistent type handling for position columns
        int_columns = ['position', 'true_position', 'predicted_position']
        
        # Prepare the dataframes for stacking
        prepared_dfs = prepare_dataframes_for_stacking(
            dataframes=[donor_positions_df, acceptor_positions_df],
            string_columns=string_columns,
            int_columns=int_columns,
            verbose=verbose
        )
        
        # Stack the prepared dataframes
        try:
            if len(prepared_dfs) == 2:
                positions_df = prepared_dfs[0].vstack(prepared_dfs[1])
            elif len(prepared_dfs) == 1:
                positions_df = prepared_dfs[0]
            else:
                # Fallback to an empty DataFrame with the default schema
                positions_df = pl.DataFrame(schema=DEFAULT_POSITION_SCHEMA)
                
            if verbose > 0:
                # Report on which columns were added beyond the default schema
                additional_cols = [col for col in positions_df.columns if col not in DEFAULT_POSITION_SCHEMA]
                if additional_cols:
                    print(f"[INFO] Extended schema with {len(additional_cols)} additional columns: {additional_cols}")
                    
        except Exception as e:
            print(f"\n[ERROR] Failed to stack position dataframes: {e}")
            
            # Analyze the specific schema mismatches
            mismatch = analyze_schema_mismatch(
                donor_positions_df, 
                acceptor_positions_df, 
                name1="donor_positions_df", 
                name2="acceptor_positions_df"
            )
            
            # Output detailed diagnostics if verbose
            if verbose > 0:
                print(f"\n[DIAGNOSTIC] Position schema mismatch analysis:")
                print(f"  Columns only in donor: {mismatch['columns_only_in_df1']}")
                print(f"  Columns only in acceptor: {mismatch['columns_only_in_df2']}")
                print(f"  Type mismatches: {mismatch['type_mismatches']}")
                
            print(f"\n[RECOMMENDATION] {mismatch['recommendation']}")
            
            # Fallback to using extended schema method if our utility fails
            try:
                combined_schema = extend_schema(DEFAULT_POSITION_SCHEMA, donor_positions_df, acceptor_positions_df)
                donor_positions_df = ensure_schema(donor_positions_df, combined_schema)
                acceptor_positions_df = ensure_schema(acceptor_positions_df, combined_schema)
                positions_df = donor_positions_df.vstack(acceptor_positions_df)
                print("[INFO] Successfully stacked positions using fallback method")
            except Exception as fallback_error:
                print(f"[ERROR] Fallback method also failed: {fallback_error}")
                positions_df = pl.DataFrame(schema=DEFAULT_POSITION_SCHEMA)
    elif not is_dataframe_empty(donor_positions_df):
        # Build an extended schema from donor_positions_df
        combined_schema = extend_schema(DEFAULT_POSITION_SCHEMA, donor_positions_df)
        positions_df = ensure_schema(donor_positions_df, combined_schema)
    elif not is_dataframe_empty(acceptor_positions_df):
        # Build an extended schema from acceptor_positions_df
        combined_schema = extend_schema(DEFAULT_POSITION_SCHEMA, acceptor_positions_df)
        positions_df = ensure_schema(acceptor_positions_df, combined_schema)
    else:
        positions_df = pl.DataFrame(schema=DEFAULT_POSITION_SCHEMA)

    if verbose >= 2:
        print(f"Donor error count: {donor_error_df.height}, Acceptor error count: {acceptor_error_df.height}")
        print(f"Donor positions count: {donor_positions_df.height}, Acceptor positions count: {acceptor_positions_df.height}")
        print(f"Total error count: {error_df.height}, Total positions count: {positions_df.height}")
        print(f"Positions DataFrame columns: {positions_df.columns}")

        # Perform detailed gene-level analysis using the refactored function
        analyze_gene_level_positions(positions_df, consensus_window=consensus_window, verbose=verbose)

    if return_positions_df:
        return error_df, positions_df

    return error_df
