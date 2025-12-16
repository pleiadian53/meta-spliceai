"""
Utilities for verifying and validating splice site adjustments.

This module provides functions to verify that probability adjustments are correctly applied
to splice site predictions, by comparing positions before and after adjustment.
"""

import numpy as np
import polars as pl
from typing import Dict, List, Set, Tuple, Any, Optional, Union


def verify_probability_sums(donor_probs, acceptor_probs, neither_probs, 
                           strand, splice_type, adjustment_dict=None, 
                           gene_id=None, verbose=False):
    """
    Verify that probabilities sum to approximately 1.0 after adjustments.
    
    Parameters:
    -----------
    donor_probs : np.ndarray
        Array of donor probabilities
    acceptor_probs : np.ndarray
        Array of acceptor probabilities
    neither_probs : np.ndarray
        Array of neither probabilities
    strand : str
        Strand ('+' or '-')
    splice_type : str
        Type of splice site ('donor' or 'acceptor')
    adjustment_dict : dict, optional
        Dictionary with custom adjustments. If None, standard SpliceAI adjustments are assumed.
    gene_id : str, optional
        Gene ID for diagnostic messages
    verbose : bool, optional
        Whether to print detailed diagnostics
        
    Returns:
    --------
    bool
        True if all probability sums are approximately 1.0, False otherwise
    """
    # Calculate sums for all positions - we can now check the entire array since edge positions are normalized
    prob_sums = donor_probs + acceptor_probs + neither_probs
    
    # Check if sums are close to 1.0
    all_valid = np.allclose(prob_sums, 1.0, rtol=1e-5, atol=1e-5)
    
    if not all_valid and verbose:
        # Find the positions with largest deviations
        deviation = np.abs(prob_sums - 1.0)
        max_deviation_idx = np.argmax(deviation)
        
        gene_str = f" for gene {gene_id}" if gene_id else ""
        
        print(f"WARNING: Probability sum deviation detected after {splice_type} adjustment{gene_str}.")
        print(f"  Position {max_deviation_idx}: sum = {prob_sums[max_deviation_idx]:.6f}")
        print(f"  Donor: {donor_probs[max_deviation_idx]:.6f}, Acceptor: {acceptor_probs[max_deviation_idx]:.6f}, Neither: {neither_probs[max_deviation_idx]:.6f}")
        
        # Additional diagnostics if verbose is higher
        if isinstance(verbose, int) and verbose > 1:
            # Show distribution of deviations
            print(f"  Deviation statistics:")
            print(f"    Mean deviation: {np.mean(deviation):.6f}")
            print(f"    Max deviation: {np.max(deviation):.6f}")
            print(f"    Positions with deviation > 0.01: {np.sum(deviation > 0.01)}")
    else: 
        if verbose:
            print(f"  All probability sums are within tolerance: {all_valid}")
    
    return all_valid


def verify_adjustment_effect(positions_df_without_adj, positions_df_with_adj):
    """
    Verify that adjustments are actually affecting the TRUE SITE POSITIONS.
    
    Parameters:
    -----------
    positions_df_without_adj : pl.DataFrame
        DataFrame with positions without adjustments
    positions_df_with_adj : pl.DataFrame
        DataFrame with positions with adjustments
        
    Returns:
    --------
    bool
        True if the adjustments are having the expected effect, False otherwise
    """
    print("\n===== ADJUSTMENT EFFECT VERIFICATION =====")
    
    # Get the total number of positions
    total_positions_without = positions_df_without_adj.height
    total_positions_with = positions_df_with_adj.height
    print(f"Total positions without adjustment: {total_positions_without}")
    print(f"Total positions with adjustment: {total_positions_with}")
    
    if total_positions_without != total_positions_with:
        print(f"WARNING: Different number of positions in the two dataframes!")
    
    # Check for direct identity (which would indicate no changes at all)
    adjustment_effective = False
    
    # Compare TRUE splice sites for each site type separately
    for site_type in ["donor", "acceptor"]:
        print(f"\nChecking {site_type} true site positions:")
        
        # Filter to the site type and only true positives
        tp_without = positions_df_without_adj.filter(
            (pl.col("splice_type") == site_type) & 
            (pl.col("pred_type") == "TP")
        )
        
        tp_with = positions_df_with_adj.filter(
            (pl.col("splice_type") == site_type) & 
            (pl.col("pred_type") == "TP")
        )
        
        print(f"  TP count for {site_type} without adjustment: {tp_without.height}")
        print(f"  TP count for {site_type} with adjustment: {tp_with.height}")
        
        # THIS IS KEY: Instead of comparing probabilities, compare the positions of true sites
        if tp_without.height > 0 and tp_with.height > 0:
            # Get unique true site positions
            positions_without = set(tp_without["position"].to_list())
            positions_with = set(tp_with["position"].to_list())
            
            common_positions = positions_without.intersection(positions_with)
            unique_to_without = positions_without - positions_with
            unique_to_with = positions_with - positions_without
            
            print(f"  Common true {site_type} sites: {len(common_positions)}")
            print(f"  Unique to without adjustment: {len(unique_to_without)}")
            print(f"  Unique to with adjustment: {len(unique_to_with)}")
            
            # If there are different TP positions, adjustments must be working!
            if len(unique_to_without) > 0 or len(unique_to_with) > 0:
                print(f"  DETECTED: Different {site_type} sites are classified as TP!")
                adjustment_effective = True
                
                # Show examples of different sites
                if len(unique_to_without) > 0:
                    sample_pos = list(unique_to_without)[:min(3, len(unique_to_without))]
                    print(f"  Sample positions only in 'without adjustment': {sample_pos}")
                
                if len(unique_to_with) > 0:
                    sample_pos = list(unique_to_with)[:min(3, len(unique_to_with))]
                    print(f"  Sample positions only in 'with adjustment': {sample_pos}")
            else:
                print(f"  WARNING: Exact same positions for {site_type} true sites!")
                
        # Look at gene-transcript combinations - even more specific
        print(f"\n  Checking gene-transcript-position combinations:")
        
        # Create identifiers for gene-transcript-position combinations
        tp_without = tp_without.with_columns(
            pl.concat_str([
                pl.col("gene_id").fill_null(""),
                pl.lit(":"),
                pl.col("transcript_id").fill_null(""),
                pl.lit(":"),
                pl.col("position").cast(pl.Utf8)
            ]).alias("site_key")
        )
        
        tp_with = tp_with.with_columns(
            pl.concat_str([
                pl.col("gene_id").fill_null(""),
                pl.lit(":"),
                pl.col("transcript_id").fill_null(""),
                pl.lit(":"),
                pl.col("position").cast(pl.Utf8)
            ]).alias("site_key")
        )
        
        # Get unique site keys
        sites_without = set(tp_without["site_key"].to_list())
        sites_with = set(tp_with["site_key"].to_list())
        
        common_sites = sites_without.intersection(sites_with)
        unique_to_without = sites_without - sites_with
        unique_to_with = sites_with - sites_without
        
        print(f"  Common gene-transcript-position combinations: {len(common_sites)}")
        print(f"  Unique to without adjustment: {len(unique_to_without)}")
        print(f"  Unique to with adjustment: {len(unique_to_with)}")
        
        if len(unique_to_without) > 0 or len(unique_to_with) > 0:
            print(f"  DETECTED: Different gene-transcript-position combinations!")
            adjustment_effective = True
    
    if not adjustment_effective:
        print("\nWARNING: NO EFFECTIVE DIFFERENCES DETECTED IN TRUE SITE POSITIONS!")
        print("This suggests the adjustment logic is not affecting site detection.")
    else:
        print("\nVerification complete: Adjustments are affecting which positions are classified as true sites.")
        print("This is the expected behavior - adjustments should change which sites are called as TP.")
    
    print("============================================\n")
    return adjustment_effective


def verify_probability_position_shifts(
    positions_df_without_adj, 
    positions_df_with_adj, 
    adjustment_dict=None, 
    threshold=0.3,  # Lower default threshold to capture more positions
    verbose=True,
    verify_true_positions=False,  # New parameter to enable true position verification
    consensus_window=2,
    error_window=500
):
    """
    Verify that high probability positions are shifted as expected by the adjustment rules.
    
    Parameters:
    -----------
    positions_df_without_adj : pl.DataFrame
        DataFrame with positions without adjustments
    positions_df_with_adj : pl.DataFrame
        DataFrame with positions with adjustments
    adjustment_dict : dict, optional
        Dictionary with custom adjustments to verify against.
        Format: {'donor': {'plus': offset, 'minus': offset}, 
                'acceptor': {'plus': offset, 'minus': offset}}
    threshold : float, optional
        Threshold for considering a probability as "high", by default 0.3
    verbose : bool, optional
        Whether to print detailed diagnostics, by default True
    verify_true_positions : bool, optional
        Whether to also verify that true_position relationships are maintained, by default False
        
    Returns:
    --------
    dict
        Dictionary with verification results for each splice type and strand


    Memos: 
    ------
    Positions DataFrame columns: 
        ['gene_id', 'transcript_id', 'position', 'predicted_position', 'true_position', 
            'pred_type', 'score', 'strand', 'donor_score', 'acceptor_score', 
            'neither_score', 'splice_type']
        
    Column meanings:
    - position: Current position being evaluated (relative to gene start/end)
    - predicted_position: Position where SpliceAI predicted a splice site (can be None for FN/TN)
    - true_position: Actual annotated splice site position (can be None for FP/TN)
    - splice_type: 'donor' or 'acceptor' - the type of splice site
    - pred_type: 'TP', 'FP', 'FN', or 'TN' - prediction outcome
    """
    # Add at beginning of function:
    print(f"\nDiagnostic DataFrame stats before filtering:")
    print(f"  positions_df_without_adj: {positions_df_without_adj.height} rows")
    print(f"  positions_df_with_adj: {positions_df_with_adj.height} rows")
    
    # Check if the enhanced schema fields are present
    has_true_position = "true_position" in positions_df_without_adj.columns and "true_position" in positions_df_with_adj.columns
    
    if not has_true_position and verify_true_positions:
        print("WARNING: true_position field not found in DataFrames. Disabling true position verification.")
        verify_true_positions = False
    
    # For our problem gene, show actual data:
    debug_gene = "ENSG00000104435"
    debug_df = positions_df_without_adj.filter(pl.col("gene_id") == debug_gene)
    print(f"  Debug gene {debug_gene} has {debug_df.height} rows")
    if debug_df.height > 0:
        print(f"  Strand values: {debug_df['strand'].unique().to_list()}")
        print(f"  splice_type values: {debug_df['splice_type'].unique().to_list()}")
        print(f"  donor_score distribution: min={debug_df['donor_score'].min()}, max={debug_df['donor_score'].max()}")
        print(f"  High-prob rows: {debug_df.filter(pl.col('donor_score') >= 0.5).height}")
        print(f"  position field values: {debug_df['position'].is_null().sum()} null values")
        print(f"  predicted_position field values: {debug_df['predicted_position'].is_null().sum()} null values")
        
        # Add true_position stats if available
        if has_true_position:
            print(f"  true_position field values: {debug_df['true_position'].is_null().sum()} null values")
            
            # For TP cases, check if predicted_position and true_position values are close
            tp_rows = debug_df.filter(pl.col("pred_type") == "TP")
            if tp_rows.height > 0:
                close_positions = tp_rows.filter(
                    (pl.col("predicted_position").is_not_null()) & 
                    (pl.col("true_position").is_not_null()) &
                    ((pl.col("predicted_position") - pl.col("true_position")).abs() <= consensus_window)
                )
                print(f"  TP rows with predicted and true positions within {consensus_window} bp: {close_positions.height}/{tp_rows.height}")
        
        # Show a few sample rows with high probability
        high_prob = debug_df.filter(pl.col('donor_score') >= 0.9)
        if high_prob.height > 0:
            print(f"  Sample high-prob rows (up to 3):")
            print(high_prob.limit(3))

    #####################################################

    if adjustment_dict is None:
        # Default SpliceAI adjustments
        adjustment_dict = {
            'donor': {'plus': 2, 'minus': 1},
            'acceptor': {'plus': 0, 'minus': -1}
        }
    
    results = {}
    
    print("\n===== HIGH PROBABILITY POSITION SHIFT VERIFICATION =====")
    
    # Process each gene separately
    gene_ids = set(positions_df_without_adj["gene_id"].to_list()) & set(positions_df_with_adj["gene_id"].to_list())
    
    # List of genes of special interest for debugging
    debug_genes = ["ENSG00000104435"]  # STMN2
    
    for site_type in ["donor", "acceptor"]:
        results[site_type] = {'plus': {'correct': 0, 'total': 0, 'examples': []},
                             'minus': {'correct': 0, 'total': 0, 'examples': []}}
        
        # Add true position verification stats if requested
        if verify_true_positions:
            results[site_type]['plus']['true_pos_matches'] = 0
            results[site_type]['minus']['true_pos_matches'] = 0
        
        # Get the expected offset for each strand
        plus_offset = adjustment_dict.get(site_type, {}).get('plus', 0)
        minus_offset = adjustment_dict.get(site_type, {}).get('minus', 0)
        
        print(f"\nChecking {site_type.upper()} site probability shifts:")
        print(f"  Expected offsets: +strand: {plus_offset}, -strand: {minus_offset}")
        
        # Track genes by strand for verification
        plus_strand_genes = []
        minus_strand_genes = []
        
        for gene_id in gene_ids:
            # Get strand information
            gene_entries = positions_df_without_adj.filter(pl.col("gene_id") == gene_id)
            if gene_entries.height == 0:
                continue
                
            strand = gene_entries["strand"].item(0)
            if strand not in ['+', '-']:
                # continue
                raise ValueError(f"Invalid strand: {strand}")
                
            # Record which genes are on which strand
            if strand == '+':
                plus_strand_genes.append(gene_id)
            else:
                minus_strand_genes.append(gene_id)
            
            # Special debugging for genes of interest
            is_debug_gene = gene_id in debug_genes
            if is_debug_gene and verbose:
                print(f"\n  DEBUG GENE {gene_id} ({strand} strand):")
                
            # Get the expected offset based on site type and strand
            expected_offset = plus_offset if strand == '+' else minus_offset
            
            # Filter for this gene, site type, and high probability scores
            score_column = f"{site_type}_score"
            
            # Get positions with high scores before adjustment - NOW INCLUDING STRAND
            high_prob_without = positions_df_without_adj.filter(
                (pl.col("gene_id") == gene_id) &
                (pl.col("strand") == strand) &
                (pl.col("splice_type") == site_type) &
                (pl.col(score_column) >= threshold) & 
                pl.col("predicted_position").is_not_null()
            )
             
            # Get positions with high scores after adjustment - NOW INCLUDING STRAND
            high_prob_with = positions_df_with_adj.filter(
                (pl.col("gene_id") == gene_id) &
                (pl.col("strand") == strand) &
                (pl.col("splice_type") == site_type) &
                (pl.col(score_column) >= threshold) &
                pl.col("predicted_position").is_not_null()
            )
            
            # Skip if no high probability positions found
            if high_prob_without.height == 0 or high_prob_with.height == 0:
                if is_debug_gene and verbose:
                    print(f"    No high probability {site_type} positions found for gene {gene_id}:")
                    print(f"      Without adjustment: {high_prob_without.height} positions")
                    print(f"      With adjustment: {high_prob_with.height} positions")
                    
                    # Extra diagnostics for debug genes
                    gene_rows_without = positions_df_without_adj.filter(
                        (pl.col("gene_id") == gene_id) &
                        (pl.col("strand") == strand) &
                        (pl.col("splice_type") == site_type)
                    )
                    
                    if gene_rows_without.height > 0:
                        max_prob = gene_rows_without[score_column].max()
                        print(f"      Maximum {site_type} probability: {max_prob:.6f}")
                        print(f"      Threshold: {threshold}")
                        
                        # Show a few of the highest probability positions
                        top_probs = gene_rows_without.sort(score_column, descending=True).head(3)
                        print(f"      Top probability positions:")
                        for i, row in enumerate(top_probs.iter_rows(named=True)):
                            pos_value = row['predicted_position'] if row['predicted_position'] is not None else "None"
                            true_pos = row.get('true_position', None)
                            true_pos_str = f", true_pos={true_pos}" if true_pos is not None else ""
                            print(f"        {i+1}. Position {pos_value}{true_pos_str}: {row[score_column]:.6f}")
                continue
                
            # Get the position lists and sort them
            positions_without = sorted(high_prob_without["predicted_position"].drop_nulls().to_list())
            positions_with = sorted(high_prob_with["predicted_position"].drop_nulls().to_list())
            
            # Check if the number of high probability positions matches
            if len(positions_without) != len(positions_with):
                if verbose:
                    print(f"  Gene {gene_id} ({strand}): Different number of high probability positions")
                    print(f"    Without adjustment: {len(positions_without)}")
                    print(f"    With adjustment: {len(positions_with)}")
                continue
                
            # Check if positions are shifted by the expected offset
            strand_key = 'plus' if strand == '+' else 'minus'
            correct_shifts = 0
            true_pos_matches = 0 if verify_true_positions else None
            
            # Create a list of position pairs and their offset
            position_pairs = []
            for i in range(min(len(positions_without), len(positions_with))):
                pos_without = positions_without[i]
                pos_with = positions_with[i]
                observed_offset = pos_with - pos_without
                is_correct = observed_offset == expected_offset
                
                # Check true position relationships if requested
                if verify_true_positions and has_true_position:
                    # Find the rows corresponding to these positions
                    row_without = high_prob_without.filter(pl.col("predicted_position") == pos_without)
                    row_with = high_prob_with.filter(pl.col("predicted_position") == pos_with)
                    
                    if row_without.height > 0 and row_with.height > 0:
                        true_pos_without = row_without["true_position"].item(0)
                        true_pos_with = row_with["true_position"].item(0)
                        
                        # True positions should be the same across both DataFrames
                        if true_pos_without is not None and true_pos_with is not None:
                            if true_pos_without == true_pos_with:
                                true_pos_matches += 1
                
                position_pairs.append((pos_without, pos_with, observed_offset, is_correct))
                
                if is_correct:
                    correct_shifts += 1
            
            # Store results for this gene
            results[site_type][strand_key]['total'] += len(position_pairs)
            results[site_type][strand_key]['correct'] += correct_shifts
            
            # Add true position stats if requested
            if verify_true_positions and true_pos_matches is not None:
                results[site_type][strand_key]['true_pos_matches'] += true_pos_matches
            
            # Store example pairs
            if len(position_pairs) > 0:
                # Store up to 3 examples per gene
                examples = position_pairs[:min(3, len(position_pairs))]
                
                for i, (pos_without, pos_with, offset, is_correct) in enumerate(examples):
                    # Get probability values
                    prob_without = high_prob_without.filter(pl.col("predicted_position") == pos_without)[score_column].item(0)
                    prob_with = high_prob_with.filter(pl.col("predicted_position") == pos_with)[score_column].item(0)
                    
                    # Get true positions if available
                    true_pos_without = None
                    true_pos_with = None
                    true_pos_match = None
                    
                    if has_true_position:
                        row_without = high_prob_without.filter(pl.col("predicted_position") == pos_without)
                        row_with = high_prob_with.filter(pl.col("predicted_position") == pos_with)
                        
                        if row_without.height > 0 and row_with.height > 0:
                            true_pos_without = row_without["true_position"].item(0)
                            true_pos_with = row_with["true_position"].item(0)
                            true_pos_match = true_pos_without == true_pos_with if (true_pos_without is not None and true_pos_with is not None) else None
                    
                    example_data = {
                        'gene_id': gene_id,
                        'pos_without': pos_without,
                        'pos_with': pos_with,
                        'prob_without': prob_without,
                        'prob_with': prob_with,
                        'offset': offset,
                        'expected_offset': expected_offset,
                        'is_correct': is_correct
                    }
                    
                    # Add true position data if available
                    if has_true_position:
                        example_data.update({
                            'true_pos_without': true_pos_without,
                            'true_pos_with': true_pos_with,
                            'true_pos_match': true_pos_match
                        })
                    
                    results[site_type][strand_key]['examples'].append(example_data)
    
    # Calculate and print overall statistics
    print("\nSummary of high probability position shifts:")
    
    # Print statistics on which genes were processed
    print(f"  Plus strand genes: {len(plus_strand_genes)}")
    print(f"  Minus strand genes: {len(minus_strand_genes)}")
    
    if debug_genes:
        print("\nDebug gene analysis:")
        for gene_id in debug_genes:
            if gene_id in plus_strand_genes:
                print(f"  {gene_id} found on plus strand")
            elif gene_id in minus_strand_genes:
                print(f"  {gene_id} found on minus strand")
            else:
                print(f"  {gene_id} NOT FOUND in processed genes!")
    
    all_correct = True
    true_pos_correct = True if verify_true_positions else None
    
    for site_type in results:
        print(f"\n{site_type.upper()} SITES:")
        for strand in results[site_type]:
            strand_data = results[site_type][strand]
            total = strand_data['total']
            correct = strand_data['correct']
            
            if total > 0:
                accuracy = correct / total * 100
                print(f"  {strand} strand: {correct}/{total} correct shifts ({accuracy:.2f}%)")
                
                # Add true position stats if applicable
                if verify_true_positions and 'true_pos_matches' in strand_data:
                    true_pos_matches = strand_data['true_pos_matches']
                    true_pos_accuracy = true_pos_matches / total * 100 if total > 0 else 0
                    print(f"  {strand} strand true position consistency: {true_pos_matches}/{total} ({true_pos_accuracy:.2f}%)")
                    
                    if true_pos_matches < total:
                        true_pos_correct = False
                
                if correct < total:
                    all_correct = False
                    
                # Show examples
                if verbose and len(strand_data['examples']) > 0:
                    print(f"  Examples ({strand} strand):")
                    for i, ex in enumerate(strand_data['examples'][:3]):
                        status = "✓" if ex['is_correct'] else "✗"
                        print(f"    {i+1}. Gene {ex['gene_id']}: Position {ex['pos_without']} → {ex['pos_with']}")
                        print(f"       Prob: {ex['prob_without']:.4f} → {ex['prob_with']:.4f}")
                        print(f"       Offset: {ex['offset']} (Expected: {ex['expected_offset']}) {status}")
                        
                        # Add true position info if available
                        if 'true_pos_without' in ex and 'true_pos_with' in ex:
                            true_pos_status = "✓" if ex.get('true_pos_match', False) else "✗"
                            true_pos_without = ex['true_pos_without'] if ex['true_pos_without'] is not None else "None"
                            true_pos_with = ex['true_pos_with'] if ex['true_pos_with'] is not None else "None"
                            print(f"       True positions: {true_pos_without} → {true_pos_with} {true_pos_status}")
            else:
                print(f"  {strand} strand: No high probability positions found")
    
    if all_correct:
        print("\nVerification PASSED: All high probability positions are shifted as expected.")
    else:
        print("\nVerification WARNING: Some high probability positions did not shift as expected.")
        
    if verify_true_positions:
        if true_pos_correct:
            print("True position verification PASSED: All true positions are consistent across adjustments.")
        else:
            print("True position verification WARNING: Some true positions are inconsistent across adjustments.")
    
    print("========================================================\n")
    
    return results