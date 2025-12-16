"""
Utilities for analyzing position data from splice site predictions.
These functions help with diagnostics, statistics, and data quality checks
for position data structures in the splice site evaluation workflow.
"""

import numpy as np
import polars as pl
import pandas as pd

# Standard library imports
from typing import Dict, List, Set, Optional, Any, Union, Tuple

def analyze_transcript_position_stats(tp_positions, fp_positions, fn_positions, tn_collection=None, prefix="", verbose=True):
    """
    Analyze transcript and position statistics for splice site predictions.
    
    Parameters
    ----------
    tp_positions : list
        List of True Positive position dictionaries
    fp_positions : list
        List of False Positive position dictionaries  
    fn_positions : list
        List of False Negative position dictionaries
    tn_collection : list, optional
        List of True Negative position dictionaries
    prefix : str, optional
        Prefix to add to diagnostic output (e.g., gene ID or description)
    verbose : bool, optional
        Whether to print diagnostic information
        
    Returns
    -------
    dict
        Dictionary with position statistics
    """
    if tn_collection is None:
        tn_collection = []
        
    # Initialize counters
    transcript_count_per_pos = {}
    pos_type_counts = {'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0}
    
    # Count positions by type
    for pos_type, pos_list in [('TP', tp_positions), ('FP', fp_positions), ('FN', fn_positions)]:
        pos_type_counts[pos_type] = len(pos_list)
        if verbose:
            print(f"  Raw {pos_type} count: {len(pos_list)}")
        
        # Analyze transcript duplications
        for pos in pos_list:
            pos_key = f"{pos['gene_id']}:{pos['position']}"
            if pos_key not in transcript_count_per_pos:
                transcript_count_per_pos[pos_key] = set()
            if pos['transcript_id'] is not None:
                transcript_count_per_pos[pos_key].add(pos['transcript_id'])
    
    # Count TN separately
    tn_count = len(tn_collection)
    pos_type_counts['TN'] = tn_count
    if verbose:
        print(f"  Raw TN count: {tn_count}")
    
    # Analyze transcript multiplicities
    multi_transcript_pos = 0
    avg_transcripts = 0
    max_transcripts = 0
    pos_with_transcripts = 0
    
    for pos_key, transcript_set in transcript_count_per_pos.items():
        gene_id, _ = pos_key.split(":")
        if len(transcript_set) > 0:
            pos_with_transcripts += 1
            if len(transcript_set) > 1:
                multi_transcript_pos += 1
            avg_transcripts += len(transcript_set)
            max_transcripts = max(max_transcripts, len(transcript_set))
    
    if pos_with_transcripts > 0:
        avg_transcripts = avg_transcripts / pos_with_transcripts
    
    # Calculate percentage for clarity
    multi_pct = (multi_transcript_pos/len(transcript_count_per_pos)*100 
                if len(transcript_count_per_pos) > 0 else 0)
    
    if verbose:
        if prefix:
            print(f"\n[DIAGNOSTIC] {prefix} Splice site position analysis:")
        else:
            print(f"\n[DIAGNOSTIC] Splice site position analysis:")
        
        print(f"  Transcript duplication analysis:")
        print(f"    Positions with multiple transcripts: {multi_transcript_pos}/{len(transcript_count_per_pos)} ({multi_pct:.1f}%)")
        print(f"    Average transcripts per position: {avg_transcripts:.2f}")
        print(f"    Maximum transcripts for any position: {max_transcripts}")
        print(f"    Total unique positions: {len(transcript_count_per_pos)}")
        print(f"    Total rows after duplication: {sum(pos_type_counts.values())}")
    
    # Return statistics for programmatic use
    return {
        'pos_type_counts': pos_type_counts,
        'unique_positions': len(transcript_count_per_pos),
        'multi_transcript_positions': multi_transcript_pos,
        'multi_transcript_pct': multi_pct,
        'avg_transcripts_per_pos': avg_transcripts,
        'max_transcripts_per_pos': max_transcripts,
        'total_rows': sum(pos_type_counts.values())
    }

def analyze_gene_level_positions(positions_df, consensus_window=2, verbose=1):
    """
    Perform detailed gene-level analysis on positions DataFrame to verify
    prediction quality, consistency, and position relationships.
    
    Parameters
    ----------
    positions_df : pl.DataFrame
        DataFrame with position data, including gene_id, transcript_id, position,
        predicted_position, true_position, pred_type, splice_type, and score columns
    consensus_window : int, optional
        Maximum allowed distance between predicted_position and true_position to 
        be considered a match, by default 2
    verbose : int, optional
        Level of verbosity (0: none, 1: summary, 2+: detailed), by default 1
        
    Returns
    -------
    dict
        Dictionary with analysis results by gene and splice type
    """
    if verbose >= 1:
        print("\n=== DETAILED GENE-LEVEL ANALYSIS ===")
    
    # Get unique genes
    unique_genes = positions_df["gene_id"].unique().to_list()
    
    if verbose >= 1:
        print(f"Found {len(unique_genes)} unique genes in positions DataFrame")
    
    results = {}
    
    # Analyze each gene
    for gene_id in unique_genes:
        # Get gene's data
        gene_df = positions_df.filter(pl.col("gene_id") == gene_id)
        gene_strand = gene_df["strand"].unique().to_list()[0] if gene_df.height > 0 else "unknown"
        
        # Check if gene has both donor and acceptor sites
        splice_types = gene_df["splice_type"].unique() # Series is fine for 'in' check
        has_donor = "donor" in splice_types
        has_acceptor = "acceptor" in splice_types
        
        if verbose >= 1:
            print(f"\nGene: {gene_id} (Strand {gene_strand})")
            print(f"  Has donor sites: {has_donor}, Has acceptor sites: {has_acceptor}")
            print(f"  Total rows: {gene_df.height}")
        
        gene_results = {
            'strand': gene_strand,
            'total_rows': gene_df.height,
            'has_donor': has_donor,
            'has_acceptor': has_acceptor,
            'donor': {},
            'acceptor': {}
        }
        
        # Analyze each splice type within gene
        for splice_type in ["donor", "acceptor"]:
            if splice_type not in splice_types:
                if verbose >= 1:
                    print(f"  No {splice_type} sites found for this gene")
                continue
            
            # Get data for this splice type
            type_df = gene_df.filter(pl.col("splice_type") == splice_type)
            prob_column = f"{splice_type}_score"
            
            # 1. Min/max probability values
            min_prob = type_df[prob_column].min()
            max_prob = type_df[prob_column].max()
            
            # 2. Count of high probabilities (> 0.5)
            high_prob_count = type_df.filter(pl.col(prob_column) >= 0.5).height
            
            # 3-4. Counts of predicted and true splice sites
            predicted_sites = type_df.filter(pl.col("predicted_position").is_not_null()).height
            true_sites = type_df.filter(pl.col("true_position").is_not_null()).height
            
            # Analysis of position relationships
            tp_rows = type_df.filter(pl.col("pred_type") == "TP")
            position_match_count = 0
            
            if tp_rows.height > 0:
                # For TPs, check if predicted and true positions match within consensus window
                for row in tp_rows.iter_rows(named=True):
                    pred_pos = row.get('predicted_position')
                    true_pos = row.get('true_position')
                    if pred_pos is not None and true_pos is not None:
                        # Use the consensus_window parameter
                        if abs(pred_pos - true_pos) <= consensus_window:
                            position_match_count += 1
            
            position_match_pct = (position_match_count / tp_rows.height * 100 
                                 if tp_rows.height > 0 else 0)
            
            # 5. Prediction type counts
            tp_count = type_df.filter(pl.col("pred_type") == "TP").height
            fp_count = type_df.filter(pl.col("pred_type") == "FP").height
            fn_count = type_df.filter(pl.col("pred_type") == "FN").height
            tn_count = type_df.filter(pl.col("pred_type") == "TN").height
            
            # Consistency checks
            predicted_consistent = predicted_sites == (tp_count + fp_count)
            true_consistent = true_sites == (tp_count + fn_count)
            
            if verbose >= 1:
                print(f"  {splice_type.upper()} SITES:")
                print(f"    Probability statistics: min={min_prob:.6f}, max={max_prob:.6f}")
                print(f"    High probability values (≥0.5): {high_prob_count}")
                print(f"    Predicted splice sites: {predicted_sites} (TPs+FPs={tp_count+fp_count}, consistent: {predicted_consistent})")
                print(f"    True splice sites: {true_sites} (TPs+FNs={tp_count+fn_count}, consistent: {true_consistent})")
                print(f"    Prediction outcomes: TPs={tp_count}, FPs={fp_count}, FNs={fn_count}, TNs={tn_count}")
                
                if tp_count > 0:
                    print(f"    TP position match: {position_match_count}/{tp_count} ({position_match_pct:.2f}%)")
            
            # Store detailed results
            gene_results[splice_type] = {
                'min_prob': min_prob,
                'max_prob': max_prob,
                'high_prob_count': high_prob_count,
                'predicted_sites': predicted_sites,
                'true_sites': true_sites,
                'tp_count': tp_count,
                'fp_count': fp_count,
                'fn_count': fn_count,
                'tn_count': tn_count,
                'predicted_consistent': predicted_consistent,
                'true_consistent': true_consistent,
                'position_match_count': position_match_count,
                'position_match_pct': position_match_pct
            }
        
        # Add gene results to overall results
        results[gene_id] = gene_results
    
    return results

def display_position_samples(
    positions_df: pl.DataFrame,
    max_samples_per_type: int = 10,
    verbose: int = 1
) -> Dict[str, pd.DataFrame]:
    """
    Display samples of splice site positions with key information separated into identification
    and probability tables for easier analysis.
    
    This function creates two tables for each prediction type (TP, FP, FN):
    1. Identification table: Shows position information (gene_id, transcript_id, positions, etc.)
    2. Probability table: Shows probability scores and derived features
    
    Parameters
    ----------
    positions_df : pl.DataFrame
        DataFrame containing position information with both basic fields and derived features
    max_samples_per_type : int, optional
        Maximum number of samples to display for each prediction type, by default 10
    verbose : int, optional
        Verbosity level (0: return only, 1: print basic info, 2: print full tables), by default 1
        
    Returns
    -------
    Dict[str, pd.DataFrame]
        Dictionary with keys like 'TP_identity', 'TP_probabilities', etc. containing
        pandas DataFrames for each table type, for easier viewing and analysis

    Memo
    ----
    Default columns in positions dataframe: 

    ['gene_id', 'transcript_id', 'position', 'predicted_position', 'true_position', 
        'pred_type', 'score', 'strand', 
        'donor_score', 'acceptor_score', 'neither_score', 
        'splice_type']
    """
    if positions_df.height == 0:
        print("[warning] Positions DataFrame is empty, nothing to display")
        return {}
        
    # Define column sets for the two types of tables
    id_columns = [
        "gene_id", "transcript_id", "position", "predicted_position", 
        "true_position", "splice_type", "pred_type", "strand"
    ]
    # Other possible columns: "chrom"
    
    # Get all columns that aren't in id_columns
    all_columns = positions_df.columns
    prob_columns = [col for col in all_columns if col not in id_columns and col != "pred_type"]
    
    # Prediction types to sample
    pred_types = ["TP", "FP", "FN"]
    
    # Convert to pandas for easier display and handling
    positions_pd = positions_df.to_pandas()
    
    # Format integer columns 
    integer_columns = ["position"]
    if "predicted_position" in positions_pd.columns:
        positions_pd["predicted_position"] = positions_pd["predicted_position"].astype('Int64')  # pandas nullable integer
    if "true_position" in positions_pd.columns:
        positions_pd["true_position"] = positions_pd["true_position"].astype('Int64')  # pandas nullable integer
    
    result_tables = {}
    
    for pred_type in pred_types:
        # Filter to just this prediction type
        type_df = positions_pd[positions_pd["pred_type"] == pred_type]
        
        if len(type_df) == 0:
            if verbose >= 1:
                print(f"[info] No {pred_type} predictions found in the data")
            continue
            
        # Sample up to max_samples_per_type rows
        samples = type_df.sample(min(max_samples_per_type, len(type_df)))
        
        # Create the two tables
        id_table = samples[id_columns].copy()
        prob_table = samples[["position"] + prob_columns].copy()  # Keep position as reference
        
        # Format probability values to 4 decimal places for better readability
        for col in prob_table.columns:
            if col != "position" and pd.api.types.is_numeric_dtype(prob_table[col]):
                prob_table[col] = prob_table[col].round(4)
        
        # Store in result
        result_tables[f"{pred_type}_identity"] = id_table
        result_tables[f"{pred_type}_probabilities"] = prob_table
        
        # Print tables if requested
        if verbose >= 1:
            print(f"\n{'-' * 40}")
            print(f"Sample {pred_type} Predictions ({len(samples)} of {len(type_df)} total)")
            print(f"{'-' * 40}")
            
        if verbose >= 2:
            pd.set_option('display.float_format', '{:.4f}'.format)
            # Print full tables
            print("\nIdentification Information:")
            print(id_table)
            print("\nProbability Features:")
            prob_display = prob_table.copy()
            # Limit the number of columns displayed if there are too many
            if len(prob_columns) > 6:
                important_cols = ["position", "donor_score", "acceptor_score", "neither_score", 
                                  "relative_donor_probability", "splice_probability"]
                available_cols = [col for col in important_cols if col in prob_display.columns]
                other_cols = [col for col in prob_display.columns if col not in important_cols]
                # Show important columns first, then a few others
                display_cols = available_cols + other_cols[:max(0, 10-len(available_cols))]
                if len(other_cols) > 10-len(available_cols):
                    print(f"(Showing {len(display_cols)} of {len(prob_display.columns)} probability columns)")
                prob_display = prob_display[display_cols]
            print(prob_display)
            pd.reset_option('display.float_format')
        elif verbose == 1:
            # Print condensed information
            print("\nSample genes:", ", ".join(samples["gene_id"].unique()))
            print(f"Identity columns: {', '.join(id_columns)}")
            print(f"Probability columns: {len(prob_columns)} columns including {', '.join(prob_columns[:3])}...")
    
    return result_tables

def format_diagnostic_summary(positions_df: pl.DataFrame) -> str:
    """
    Generate a concise diagnostic summary string from positions DataFrame.
    
    Parameters
    ----------
    positions_df : pl.DataFrame
        DataFrame with position data including prediction types and scores
        
    Returns
    -------
    str
        Formatted summary string with key statistics
    """
    if positions_df.height == 0:
        return "No positions data available"
        
    # Count by prediction type
    type_counts = {
        pred_type: positions_df.filter(pl.col("pred_type") == pred_type).height
        for pred_type in ["TP", "FP", "FN", "TN"]
    }
    
    # Calculate average probabilities for true splice sites
    true_sites = positions_df.filter(pl.col("pred_type").is_in(["TP", "FN"]))
    if true_sites.height > 0:
        avg_donor = true_sites.filter(pl.col("splice_type") == "donor").select(pl.col("donor_score").mean()).item()
        avg_acceptor = true_sites.filter(pl.col("splice_type") == "acceptor").select(pl.col("acceptor_score").mean()).item()
        donor_count = true_sites.filter(pl.col("splice_type") == "donor").height
        acceptor_count = true_sites.filter(pl.col("splice_type") == "acceptor").height
    else:
        avg_donor, avg_acceptor = 0, 0
        donor_count, acceptor_count = 0, 0
    
    # Format the summary
    summary = [
        f"Position statistics:",
        f"  Total positions: {positions_df.height}",
        f"  By prediction type: TP={type_counts['TP']}, FP={type_counts['FP']}, FN={type_counts['FN']}, TN={type_counts['TN']}",
        f"  True donor sites: {donor_count} (avg score: {avg_donor:.4f})",
        f"  True acceptor sites: {acceptor_count} (avg score: {avg_acceptor:.4f})"
    ]
    
    return "\n".join(summary)