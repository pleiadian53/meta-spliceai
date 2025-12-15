import os
import numpy as np
from typing import Union, List, Set

import pandas as pd
import polars as pl

import random
from pybedtools import BedTool
from Bio import SeqIO
from collections import defaultdict

from .utils_fs import (
    build_gtf_database, extract_all_gtf_annotations
)
from .utils_bio import (
    normalize_strand
)

from .utils_doc import (
    print_emphasized, 
    print_with_indent, 
    print_section_separator
)

from .extract_genomic_features import (
    get_overlapping_gene_set, 
    load_overlapping_gene_metadata, 
    get_or_load_overlapping_gene_metadata,
    get_overlapping_gene_metadata,
    find_overlapping_genes, 
    SpliceAnalyzer
)

from tqdm import tqdm
from sklearn.metrics import roc_auc_score

# Refactor to utils_plot
import matplotlib.pyplot as plt 


def evaluate_spliceai_predictions_v0(annotations_df, pred_results, threshold=0.9, consensus_window=2):
    """
    Evaluate SpliceAI predictions for donor splice sites against true annotations.

    Parameters:
    - annotations_df (pl.DataFrame): DataFrame containing true splice site annotations.
      Columns: ['chrom', 'start', 'end', 'strand', 'site_type', 'gene_id', 'transcript_id']
    - pred_results (dict): The output of predict_splice_sites_for_genes(), containing per-nucleotide probabilities.
    - threshold (float): Threshold for classifying a prediction as a donor site (default is 0.9).
    - consensus_window (int): Tolerance window around true splice sites.

    Returns:
    - results (dict): A dictionary containing counts for TP, TN, FP, and FN.
    """
    results = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}

    # Step 1: Group true splice sites by gene
    grouped_annotations = annotations_df.filter(pl.col('site_type') == 'donor').group_by('gene_id').agg(
        pl.struct(['start', 'end']).alias('donor_sites')
    ).to_dict(as_series=False)

    # Step 2: Process each gene's predictions in pred_results
    for gene_id, gene_data in pred_results.items():
        if gene_id not in grouped_annotations['gene_id']:
            print(f"No donor annotations for gene: {gene_id}")
            continue
        
        donor_sites = grouped_annotations['donor_sites'][grouped_annotations['gene_id'].index(gene_id)]
        # donor_sites will be a list of dictionaries (or structs) containing the start and end positions 
        # of donor sites for a specific gene_id.
        # NOTE: 
        #  - grouped_annotations['donor_sites']:
        #    This accesses the list of donor sites for each gene. The donor sites are stored 
        #    as a list of dictionaries (or structs) containing the 'start' and 'end' positions.
        #  - grouped_annotations['gene_id'].index(gene_id): 
        #    This finds the index of the current gene_id in the list of gene IDs.

        donor_predictions = gene_data['donor_prob']  # List of probabilities for donor sites
        positions = gene_data['positions']  # List of absolute positions for each nucleotide
        gene_len = len(donor_predictions)

        # Initialize label prediction vector based on the threshold
        label_predictions = np.array([1 if prob >= threshold else 0 for prob in donor_predictions])

        # Create a set of positions within consensus windows for true splice sites
        true_donor_pos_set = set()
        for site in donor_sites:
            true_donor_pos_set.update(range(site['start'] - consensus_window, site['end'] + consensus_window + 1))
        
        # Step 3: Evaluate the prediction against true splice sites
        for i, (pred_label, position) in enumerate(zip(label_predictions, positions)):
            if position in true_donor_pos_set:
                if pred_label == 1:
                    results['TP'] += 1  # Correctly predicted within the splice site window
                else:
                    results['FN'] += 1  # Missed a true splice site (False Negative)
            else:
                if pred_label == 1:
                    results['FP'] += 1  # Incorrectly predicted a donor site outside the window (False Positive)
                else:
                    results['TN'] += 1  # Correctly predicted no splice site outside the window (True Negative)
        
        # Debugging: Check output for the gene
        print(f"Gene {gene_id} | TP: {results['TP']}, TN: {results['TN']}, FP: {results['FP']}, FN: {results['FN']}")

    return results


def reverse_complement_eager(df, sequence_col="sequence", strand_col="strand"):
    """
    Reverse-complement DNA sequences in an eager Polars DataFrame based on strand.

    Parameters:
    - df (pl.DataFrame): Polars DataFrame containing the sequence and strand information.
    - sequence_col (str): Column containing the DNA sequence.
    - strand_col (str): Column indicating the strand ('+' or '-').

    Returns:
    - pl.DataFrame: Updated Polars DataFrame with reverse-complement sequences for negative strands.
    """
    from Bio.Seq import Seq

    # Define a function to reverse complement a sequence
    def reverse_complement(seq):
        return str(Seq(seq).reverse_complement())

    # Apply the reverse complement function to the sequence column where strand is '-'
    df = df.with_columns(
        pl.when(pl.col(strand_col) == "-")
        .then(pl.col(sequence_col).apply(reverse_complement))
        .otherwise(pl.col(sequence_col))
        .alias(sequence_col)
    )
    return df


def evaluate_splice_site_predictions(annotations_df, pred_results, threshold=0.5, consensus_window=2, **kargs):
    """
    Evaluate SpliceAI predictions for both donor and acceptor splice sites against true annotations.

    Parameters:
    - annotations_df (pl.DataFrame): DataFrame containing true splice site annotations.
      Columns: ['chrom', 'start', 'end', 'strand', 'site_type', 'gene_id', 'transcript_id']
    - pred_results (dict): The output of predict_splice_sites_for_genes(), containing per-nucleotide probabilities.
    - threshold (float): Threshold for classifying a prediction as a donor or acceptor site (default is 0.5).
    - consensus_window (int): Tolerance window around true splice sites.

    Returns:
    - performance_df (pl.DataFrame): 
        A consolidated DataFrame containing counts for TP, TN, FP, FN, derived metrics per gene, 
        and splice type for both donor and acceptor sites.
    """
    verbose = kargs.get("verbose", 1)
    chromosome = kargs.get("chromosome", kargs.get("chr", "?"))  # additional information for debugging
    # overlapping_genes_metadata = kargs.get("overlapping_genes_metadata", None)
    # overlapping_gene_file_path = kargs.get("overlapping_gene_file_path", None)
    # gtf_file_path = kargs.get("gtf_file_path", None)
    return_discrepancies = kargs.get("return_discrepancies", False)
    predicted_delta_correction = kargs.get("predicted_delta_correction", True)

    if predicted_delta_correction:
        print_emphasized("[action] Applying predicted delta correction to compensate for systematic discrepancies ...")

    # Evaluate donor site predictions
    donor_performance_df, donor_delta_df = \
        evaluate_donor_site_predictions(
            annotations_df, pred_results, 
            threshold=threshold, 
            consensus_window=consensus_window, 
            # overlapping_genes_metadata=overlapping_genes_metadata,
            # gtf_file_path=gtf_file_path,
            # overlapping_gene_file_path=overlapping_gene_file_path, 
            return_discrepancies=True,
            predicted_delta_correction=predicted_delta_correction,
            chromosome=chromosome,  # information for debugging
            verbose=verbose
        )
    
    # Rename `n_donors` to `n_splice_sites` and add splice type column for donor
    donor_performance_df = donor_performance_df.rename({"n_donors": "n_splice_sites"}).with_columns(
        pl.lit("donor").alias("splice_type")
    )
    # NOTE: pl.lit():  
    #  - pl.lit(value) is a Polars function that creates a literal value (constant) that can be used in DataFrame operations.
    #    In this context, pl.lit("donor") creates a column where every row has the value "donor".

    donor_delta_df = donor_delta_df.with_columns(
        pl.lit("donor").alias("splice_type")
    )

    # Evaluate acceptor site predictions
    acceptor_performance_df, acceptor_delta_df = \
        evaluate_acceptor_site_predictions(
            annotations_df, pred_results, 
            threshold=threshold, 
            consensus_window=consensus_window, 
            # overlapping_genes_metadata=overlapping_genes_metadata,
            # gtf_file_path=gtf_file_path,
            # overlapping_gene_file_path=overlapping_gene_file_path,
            return_discrepancies=True,
            predicted_delta_correction=predicted_delta_correction,
            chromosome=chromosome,  # information for debugging
            verbose=verbose
        )

    # Rename `n_acceptors` to `n_splice_sites` and add splice type column for acceptor
    acceptor_performance_df = acceptor_performance_df.rename({"n_acceptors": "n_splice_sites"}).with_columns(
        pl.lit("acceptor").alias("splice_type")
    )

    acceptor_delta_df = acceptor_delta_df.with_columns(
        pl.lit("acceptor").alias("splice_type")
    )

    # Consolidate donor and acceptor performance dataframes
    performance_df = donor_performance_df.vstack(acceptor_performance_df)
    delta_df = donor_delta_df.vstack(acceptor_delta_df)

    if return_discrepancies:
        return performance_df, delta_df

    return performance_df



def evaluate_splice_site_errors(annotations_df, pred_results, threshold=0.5, consensus_window=2, error_window=500, **kargs):
    """
    Evaluate SpliceAI predictions for both donor and acceptor splice sites and identify FPs and FNs, including windowed regions.

    Parameters:
    - annotations_df (pl.DataFrame): DataFrame containing true splice site annotations.
      Columns: ['chrom', 'start', 'end', 'strand', 'site_type', 'gene_id', 'transcript_id']
    - pred_results (dict): The output of predict_splice_sites_for_genes(), containing per-nucleotide probabilities.
    - threshold (float): Threshold for classifying a prediction as a donor or acceptor site (default is 0.5).
    - consensus_window (int): Tolerance window around true splice sites.
    - error_window (int or tuple): Window size (or sizes) for tracking the surrounding region of FPs and FNs.
    - verbose (int): Level of verbosity.

    Returns:
    - error_df (pl.DataFrame): A consolidated DataFrame containing positions of FPs and FNs, with window coordinates,
      for both donor and acceptor sites, and an additional column indicating the splice type ('donor' or 'acceptor').
    """
    verbose = kargs.get("verbose", 1)
    chromosome = kargs.get("chromosome", kargs.get("chr", "?"))
    collect_tn = kargs.get("collect_tn", True)
    overlapping_genes_metadata = kargs.get("overlapping_genes_metadata", None)
    overlapping_gene_file_path = kargs.get("overlapping_gene_file_path", None)
    predicted_delta_correction = kargs.get("predicted_delta_correction", True)
    gtf_file_path = kargs.get("gtf_file_path", None)
    return_positions_df = kargs.get("return_positions_df", False)

    # Evaluate donor site errors
    donor_error_df, donor_positions_df = \
        evaluate_donor_site_errors(
            annotations_df, pred_results, 
            threshold=threshold, 
            consensus_window=consensus_window, 
            error_window=error_window, 
            collect_tn=collect_tn,
            overlapping_genes_metadata=overlapping_genes_metadata,
            gtf_file_path=gtf_file_path,
            overlapping_gene_file_path=overlapping_gene_file_path,  # Todo: Use a gene-specific Class to handle this
            return_positions_df=True,
            predicted_delta_correction=predicted_delta_correction,
            chromosome=chromosome,  # information for debugging
            verbose=verbose
        )

    # Add a column to indicate the splice type as 'donor'
    donor_error_df = donor_error_df.with_columns(
        pl.lit("donor").alias("splice_type")
    )
    donor_positions_df = donor_positions_df.with_columns(
        pl.lit("donor").alias("splice_type")
    )

    # Evaluate acceptor site errors
    acceptor_error_df, acceptor_positions_df = \
        evaluate_acceptor_site_errors(
            annotations_df, pred_results, 
            threshold=threshold, 
            consensus_window=consensus_window, 
            error_window=error_window, 
            collect_tn=collect_tn,
            overlapping_genes_metadata=overlapping_genes_metadata,
            gtf_file_path=gtf_file_path,
            overlapping_gene_file_path=overlapping_gene_file_path,
            return_positions_df=True,
            predicted_delta_correction=predicted_delta_correction,
            chromosome=chromosome,  # information for debugging
            verbose=verbose
        )

    # Add a column to indicate the splice type as 'acceptor'
    acceptor_error_df = acceptor_error_df.with_columns(
        pl.lit("acceptor").alias("splice_type")
    )
    acceptor_positions_df = acceptor_positions_df.with_columns(
        pl.lit("acceptor").alias("splice_type")
    )

    # Consolidate donor and acceptor error dataframes
    error_df = donor_error_df.vstack(acceptor_error_df)
    positions_df = donor_positions_df.vstack(acceptor_positions_df)

    if return_positions_df:
        return error_df, positions_df

    return error_df

####################################################################################################


def is_within_overlapping_splice_site(
    position, consensus_window, gene_id, gene_data, overlapping_genes_metadata, grouped_annotations
):
    """
    Check if a position lies within the splice site regions of overlapping genes.

    Parameters:
    - position (int): The position to check.
    - consensus_window (int): The consensus window size.
    - gene_id (str): The ID of the current gene.
    - gene_data (dict): Data for the current gene.
    - overlapping_genes_metadata (dict): Metadata for overlapping genes.
    - grouped_annotations (dict): Annotations for all genes.
       - keys: 'gene_id', 'acceptor_sites', 'donor_sites'
       - values: Lists of gene IDs, acceptor sites, and donor sites.

    Returns:
    - bool: True if the position is within an overlapping splice site, False otherwise.
    """
    if gene_id not in overlapping_genes_metadata:
        return False

    for overlap in overlapping_genes_metadata[gene_id]:  # overlapping_genes_metadata[gene_id] is a list of dictionaries
        if (overlap['start'] <= position <= overlap['end']) and (overlap['strand'] == gene_data['strand']):
            overlap_gene_id = overlap['gene_id']   # grouped_annotations['gene_id'].index(gene_id)
            if overlap_gene_id in grouped_annotations['gene_id']:
                # For each acceptor site (a dictionary), calculate the relative position within the gene
                # - can also be calculated as: ((site['start'] + site['end']) // 2) - gene_data['gene_start']
                overlapping_true_positions = [
                    site['position'] - gene_data['gene_start']
                    for site in grouped_annotations['acceptor_sites'][grouped_annotations['gene_id'].index(overlap_gene_id)]
                ]
                for true_pos in overlapping_true_positions:
                    if (true_pos - consensus_window) <= position <= (true_pos + consensus_window):
                        return True
    return False


def adjust_scores(scores, strand, splice_type):
    """
    Adjust splice site scores based on strand-specific systematic discrepancies.

    Parameters:
    - scores (np.ndarray): The array of splice site probabilities (donor or acceptor).
    - strand (str): The strand of the gene ('+' or '-').
    - splice_type (str): The type of splice site ('donor' or 'acceptor').

    Returns:
    - adjusted_scores (np.ndarray): The adjusted splice site probabilities.
    """
    adjusted_scores = scores.copy()

    if splice_type == 'donor':
        if strand == '+':
            adjusted_scores = np.roll(adjusted_scores, 2)  # Shift forward by 2nt
            adjusted_scores[:2] = 0  # Reset wrapped-around values
        elif strand == '-':
            adjusted_scores = np.roll(adjusted_scores, 1)  # Shift forward by 1nt
            adjusted_scores[:1] = 0  # Reset wrapped-around values
        else:
            raise ValueError(f"Invalid strand value: {strand}")
    elif splice_type == 'acceptor':
        if strand == '+':
            adjusted_scores = adjusted_scores  # No adjustment needed
        elif strand == '-':
            adjusted_scores = np.roll(adjusted_scores, -1)  # Shift backward by 1nt
            adjusted_scores[-1:] = 0  # Reset wrapped-around values
        else:
            raise ValueError(f"Invalid strand value: {strand}")
    else:
        raise ValueError(f"Invalid splice type: {splice_type}")

    return adjusted_scores


def adjust_labels(scores, threshold, strand, splice_type):
    """
    Generate labels based on adjusted splice site scores.

    Parameters:
    - scores (np.ndarray): The array of splice site probabilities (donor or acceptor).
    - threshold (float): The probability threshold for labeling a position as positive.
    - strand (str): The strand of the gene ('+' or '-').
    - splice_type (str): The type of splice site ('donor' or 'acceptor').

    Returns:
    - adjusted_labels (np.ndarray): The binary labels (1 for positive, 0 for negative).
    """
    adjusted_scores = adjust_scores(scores, strand, splice_type)
    adjusted_labels = np.array([1 if prob >= threshold else 0 for prob in adjusted_scores])
    return adjusted_labels


def evaluate_donor_site_predictions(annotations_df, pred_results, threshold=0.9, consensus_window=2, **kargs):
    """
    Evaluate SpliceAI predictions for donor sites against true annotations using relative positions,
    accounting for overlapping genes.

    Parameters:
    - annotations_df (pl.DataFrame): DataFrame containing true splice site annotations.
      Columns: ['chrom', 'start', 'end', 'strand', 'site_type', 'gene_id', 'transcript_id'].
    - pred_results (dict): Output of predict_splice_sites_for_genes(), containing per-nucleotide probabilities.
    - gtf_file_path (str): Path to the GTF file for identifying overlapping genes.
    - threshold (float): Threshold for classifying a prediction as a donor site (default is 0.9).
    - consensus_window (int): Tolerance window around true splice sites.

    Returns:
    - results_df (pl.DataFrame): A DataFrame containing counts for TP, TN, FP, and FN per gene.

    NOTE: 
    [1] .agg(pl.struct(['start', 'end']).alias('donor_sites')) creates a struct column 'donor_sites' containing
        the start and end positions of the true donor sites for each gene.
    """
    # from .extract_genomic_features import SpliceAnalyzer

    verbose = kargs.get('verbose', 1)
    chromosome = kargs.get('chromosome', kargs.get('chr', '?'))
    overlapping_genes_metadata = kargs.get('overlapping_genes_metadata', None)
    adjust_for_overlapping_genes = kargs.get('adjust_for_overlapping_genes', True)
    return_discrepancies = kargs.get('return_discrepancies', True)
    predicted_delta_correction = kargs.get('predicted_delta_correction', True)

    if overlapping_genes_metadata is None:
        # Check if the GTF file path is provided in kargs
        # gtf_file_path = kargs.get('gtf_file_path', None)
        # if gtf_file_path is None:
        #     raise ValueError("GTF file path must be provided to compute overlapping gene metadata.")

        # filter_valid_splice_sites = kargs.get('filter_valid_splice_sites', True)
        # min_exons = kargs.get('min_exons', 2)

        # overlapping_genes_metadata = \
        #     get_or_load_overlapping_gene_metadata(
        #         gtf_file_path,
        #         overlapping_gene_path=kargs.get('overlapping_gene_path', None),
        #         filter_valid_splice_sites=filter_valid_splice_sites,
        #         min_exons=min_exons,
        #         output_format='dict')

        sa = SpliceAnalyzer()
        overlapping_genes_metadata = sa.retrieve_overlapping_gene_metadata()

    # Prepare results
    results_list = []
    discrepancies = []  # To store discrepancies for TPs

    # Group true splice sites by gene and include the position column
    grouped_annotations = annotations_df.filter(pl.col('site_type') == 'donor').group_by('gene_id').agg(
        pl.struct(['start', 'end', 'position', 'transcript_id' ]).alias('donor_sites')
    ).to_dict(as_series=False)
    # NOTE [1]: Suppose that for each donor site, we have the start, end, and position (midpoint) of the site.
    # - The pl.struct(['start', 'end', 'position']) creates a struct (similar to a dictionary) for each group 
    #   containing the start and end columns. The alias('donor_sites') renames this struct to donor_sites.
    # - The as_series=False argument ensures that the values in the dictionary are lists rather than Series.
    # - Example data structure:
    # { 'gene_id': ['gene1', 'gene2', 'gene3'],
    #   'donor_sites': [
    #           [{'start': 100, 'end': 150, 'position': 923}, {'start': 200, 'end': 250, 'position': 1749}],
    #           [{'start': 300, 'end': 350, 'position': 2345}],
    #            [{'start': 500, 'end': 550, 'position': 3456}] 
    #       ]}

    # Process each gene's predictions in pred_results
    n_genes_processed = 0
    total_genes = len(pred_results)
    n_fp_correction = n_fn_correction = 0

    # Optional progress bar
    iter_items = pred_results.items()
    if verbose:
        try:
            from tqdm import tqdm  # type: ignore
            iter_items = tqdm(iter_items, total=total_genes, desc="Splice eval", unit="gene")
        except ImportError:
            # tqdm not installed – fall back silently
            pass

    for gene_id, gene_data in iter_items:
        # Initialize results
        gene_results = {'gene_id': gene_id, 'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0, 'n_donors': 0, 'top_k_accuracy': None}

        if gene_id not in grouped_annotations['gene_id']:
            # No donor annotations for this gene
            gene_results.update({
                'precision': None,
                'recall': None,
                'specificity': None,
                'f1_score': None,
                'fpr': None,
                'fnr': None
            })
            results_list.append(gene_results)
            continue

        # Extract donor sites from grouped_annotations
        donor_sites = grouped_annotations['donor_sites'][grouped_annotations['gene_id'].index(gene_id)]
        strand = normalize_strand(gene_data['strand'])  # Strand of the gene
       
        true_donor_positions = []  # List to store relative positions of true donor sites
        position_to_transcript = defaultdict(set)  # Dictionary to map relative positions to transcript IDs

        for site in donor_sites:
            # Calculate relative positions based on strand
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
        true_donor_positions = np.array(sorted(set(true_donor_positions)))  # sorted in ascending order i.e 5' to 3'
        position_to_transcript = {pos: list(transcripts) for pos, transcripts in position_to_transcript.items()}

        # Extract donor probabilities and positions
        donor_probabilities = np.array(gene_data['donor_prob'])  # Probabilities for all positions
        if predicted_delta_correction:
            # Adjust scores
            donor_probabilities = adjust_scores(donor_probabilities, strand, 'donor')
            # Align probabilities with donor positions
            # - This is to address systematic discrepancies between the predicted and true positions

        # Validate positions within range
        assert true_donor_positions.max() < len(donor_probabilities), \
            "true_donor_positions contain indices out of range for donor_probabilities"

        true_donor_scores = donor_probabilities[true_donor_positions]
        true_donor_scores = np.round(true_donor_scores, 3)  # Round for readability      
        
        gene_results['n_donors'] = len(true_donor_positions)

        if verbose: 
            gene_len = len(donor_probabilities)
            gene_start = gene_data['gene_start']  # Absolute start position of the gene
            gene_end = gene_data['gene_end']  # Absolute end position of the gene
            print(f"[test] Gene {gene_id} @ chr={chromosome}: start={gene_start}, end={gene_end}, length={gene_len}=?={gene_end-gene_start+1}")
            
            # assert gene_len == gene_end - gene_start + 1, f"gene_len={gene_len} <> {gene_end-gene_start+1}"
            # NOTE: Can be off by 1 

        label_predictions = np.array([1 if prob >= threshold else 0 for prob in donor_probabilities])

        # Handle strand-specific reverse 
        # if strand == '-':
        #     label_predictions = label_predictions[::-1]
        # NOTE: Already accounted for via relative positions

        positive_prediction_positions = np.where(label_predictions == 1)[0]
        positive_prediction_positions = sorted(positive_prediction_positions)  # 5' to 3' direction

        true_donor_status = {i: False for i in range(len(true_donor_positions))}
        # fp_positions, fn_positions = [], []

        # Loop over each position in the label sequence
        for i, pred_label in enumerate(label_predictions):
            found_in_window = False

            # Check if this position is a true splice site for the current gene
            for idx, true_pos in enumerate(true_donor_positions):
                if true_donor_status[idx]:  # Skip already processed true donor sites
                    continue

                window_start = true_pos - consensus_window
                window_end = true_pos + consensus_window

                if window_start <= i <= window_end:
                    found_in_window = True

                    # Get the associated transcript IDs for this true position
                    associated_transcripts = position_to_transcript[true_pos]
                    
                    # Count TP or FN based on the prediction
                    if pred_label == 1:
                        gene_results['TP'] += 1
                        true_donor_status[idx] = True

                        # Compute max probability in the window
                        window_slice = donor_probabilities[window_start:window_end+1]
                        if window_slice.size > 0:
                            score_adjusted = np.max(window_slice)
                        else:
                            score_adjusted = donor_probabilities[i]

                        assert score_adjusted >= threshold, \
                            f"Unexpected score for TP: {score_adjusted} < {threshold}"
                        # The condition is not necessarily true due to the "predicted delta"

                        # Calculate delta using the position of the max score
                        # max_idx = np.argmax(donor_probabilities[window_start:window_end+1]) + window_start
                        delta = i - true_pos  # Discrepancy between predicted and true positions

                        # Append all associated transcript IDs for this position
                        for transcript_id in associated_transcripts:
                            discrepancies.append({
                                'gene_id': gene_id,
                                'transcript_id': transcript_id,
                                'strand': strand,
                                'true_pos': true_pos,
                                'predicted_pos': i,
                                'delta': delta,
                                'score': score_adjusted
                            })
                        # NOTE: Even though there may be multiple transcripts having the same splice site,
                        #       we are not counting them separately toward TPs. 
                        #       At the gene level, the same splice site is only evaluated once

                    break
            # End looping through each donor site

            # Count non-splice sites (negative examples)
            if not found_in_window:
                is_fp = False
                
                if pred_label == 1:
                    is_fp = True

                    # Adjust FP count for overlapping genes
                    if adjust_for_overlapping_genes and is_within_overlapping_splice_site(
                        i, consensus_window, gene_id, gene_data, overlapping_genes_metadata, grouped_annotations
                    ):
                        is_fp = False  # A true splice site in overlapping gene(s) -> not a FP
                        n_fp_correction += 1  # Increment the correction count
                    
                    if is_fp:
                        gene_results['FP'] += 1 
                        # fp_positions.append(i) 
                else:
                    # Predicted negative correctly -> TN
                    gene_results['TN'] += 1

        # Count false negatives
        for idx, status in true_donor_status.items():
            if not status:
                gene_results['FN'] += 1
                # fn_positions.append(true_donor_positions[idx])

        if verbose: 
            print_emphasized(f"[test] Gene {gene_id} @ chr={chromosome}: start={gene_data['gene_start']}, strand={strand}, number of donor sites= {len(true_donor_positions)})")
            print_with_indent(f"Number of positive predictions @ threshold={threshold}: {sum(label_predictions)}", indent_level=1)
            print_with_indent(f"Relative positions of positive predictions:\n{positive_prediction_positions}\n", indent_level=2)
            print_with_indent(f"Relative positions of true donor sites:\n{true_donor_positions}\n", indent_level=2)
            print_with_indent(f"Donor scores:\n{np.round(true_donor_scores, 3)}\n", indent_level=2)

        # Top-K Accuracy Calculation
        if gene_results['n_donors'] > 0:
            # Get the top-K predicted donor sites
            top_k = gene_results['n_donors']
            sorted_pred_positions = np.argsort(-donor_probabilities)[:top_k]  # Get top K highest probability indices
            correct_top_k = sum(
                any(abs(pos - true_pos) <= consensus_window for true_pos in true_donor_positions)
                for pos in sorted_pred_positions
            )
            gene_results['top_k_accuracy'] = correct_top_k / top_k

        # Step 5: Calculate derived metrics (Precision, Recall, Specificity)
        TP, TN, FP, FN = gene_results['TP'], gene_results['TN'], gene_results['FP'], gene_results['FN']
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        fpr = 1 - specificity  # proportion of negative instances that were incorrectly predicted as positive
        fnr = 1 - recall  # proportion of positive instances that were incorrectly predicted as negative

        # Calculate ROCAUC for donor site predictions
        # if len(set(label_predictions)) > 1 and 0 in label_predictions and 1 in label_predictions:  # Ensure there are both positive and negative samples
        #     roc_auc = roc_auc_score(label_predictions, donor_probabilities)
        # else:
        #     roc_auc = None  # Set to None if ROCAUC calculation is not meaningful

        gene_results.update({
            'precision': precision, 
            'recall': recall, 
            'specificity': specificity, 
            'f1_score': f1_score, 
            # 'roc_auc': roc_auc,
            'fpr': fpr,
            'fnr': fnr
            }
        )

        # Debugging: Check output for the gene
        if verbose: 
            print_with_indent(f"TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}", indent_level=2)
            print_with_indent(f"Precision: {precision}, Recall: {recall}, Specificity: {specificity}, F1 Score: {f1_score}", indent_level=3)
            print_with_indent(f"FPR: {fpr}, FNR: {fnr}", indent_level=3)

            if n_fp_correction > 0:
                print_with_indent(f"Number of FPs corrected due to overlapping genes: {n_fp_correction}", indent_level=3)
            if n_fn_correction > 0:
                print_with_indent(f"Number of FNs corrected due to overlapping genes: {n_fn_correction}", indent_level=3)
            print_section_separator(light=True)

        # Append the results for the current gene to the results list
        results_list.append(gene_results)

        n_genes_processed += 1

    # End of loop over genes

    # Convert the results list to a DataFrame
    results_df = pl.DataFrame(results_list)
    discrepancies_df = pl.DataFrame(discrepancies)

    if return_discrepancies:
        return results_df, discrepancies_df

    return results_df


def evaluate_acceptor_site_predictions(annotations_df, pred_results, threshold=0.9, consensus_window=2, **kargs):
    """
    Evaluate SpliceAI predictions for acceptor splice sites against true annotations using relative positions.

    Parameters:
    - annotations_df (pl.DataFrame): DataFrame containing true splice site annotations.
      Columns: ['chrom', 'start', 'end', 'strand', 'site_type', 'gene_id', 'transcript_id']
    - pred_results (dict): The output of predict_splice_sites_for_genes(), containing per-nucleotide probabilities.
    - threshold (float): Threshold for classifying a prediction as an acceptor site (default is 0.9).
    - consensus_window (int): Tolerance window around true splice sites.

    Returns:
    - results_df (pl.DataFrame): A DataFrame containing counts for TP, TN, FP, and FN per gene.
    """
    # from .extract_genomic_features import get_overlapping_gene_metadata

    verbose = kargs.get('verbose', 1)
    chromosome = kargs.get('chromosome', kargs.get('chr', '?'))
    overlapping_genes_metadata = kargs.get('overlapping_genes_metadata', None)
    adjust_for_overlapping_genes = kargs.get('adjust_for_overlapping_genes', True)
    return_discrepancies = kargs.get('return_discrepancies', True)
    predicted_delta_correction = kargs.get('predicted_delta_correction', True)

    if overlapping_genes_metadata is None:
        # Check if the GTF file path is provided in kargs
        # gtf_file_path = kargs.get('gtf_file_path', None)
        # if gtf_file_path is None:
        #     raise ValueError("GTF file path must be provided to compute overlapping gene metadata.")

        # filter_valid_splice_sites = kargs.get('filter_valid_splice_sites', True)
        # min_exons = kargs.get('min_exons', 2)

        # overlapping_genes_metadata = \
        #     get_or_load_overlapping_gene_metadata(
        #         gtf_file_path,
        #         overlapping_gene_path=kargs.get('overlapping_gene_path', None),
        #         filter_valid_splice_sites=filter_valid_splice_sites,
        #         min_exons=min_exons,
        #         output_format='dict')

        sa = SpliceAnalyzer()
        overlapping_genes_metadata = sa.retrieve_overlapping_gene_metadata()

    results_list = []
    discrepancies = []  # To store discrepancies for TPs

    # Group true splice sites by gene
    grouped_annotations = annotations_df.filter(pl.col('site_type') == 'acceptor').group_by('gene_id').agg(
        pl.struct(['start', 'end', 'position', 'transcript_id']).alias('acceptor_sites')
    ).to_dict(as_series=False)

    # Process each gene's predictions in pred_results
    n_genes_processed = 0
    total_genes = len(pred_results)
    n_fp_correction = n_fn_correction = 0

    # Optional progress bar
    iter_items = pred_results.items()
    if verbose:
        try:
            from tqdm import tqdm  # type: ignore
            iter_items = tqdm(iter_items, total=total_genes, desc="Splice eval", unit="gene")
        except ImportError:
            # tqdm not installed – fall back silently
            pass

    for gene_id, gene_data in iter_items:

        # Initialize counts for the current gene
        gene_results = {'gene_id': gene_id, 'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0, 'n_acceptors': 0, 'top_k_accuracy': None}

        if gene_id not in grouped_annotations['gene_id']:
            if verbose: 
                print(f"No acceptor annotations for gene: {gene_id}")

            # Create an entry for genes with no acceptor sites
            gene_results.update({
                'precision': None,
                'recall': None,
                'specificity': None,
                'f1_score': None,
                'fpr': None,
                'fnr': None
            })
            results_list.append(gene_results)

            continue

        # Extract donor sites from grouped_annotations
        acceptor_sites = grouped_annotations['acceptor_sites'][grouped_annotations['gene_id'].index(gene_id)]
        # grouped_annotations['acceptor_sites'] is a list where each element is a list of dictionaries 
        # representing the acceptor sites for each gene
        # E.g. 
        # {
        #     'gene_id': ['gene1', 'gene2', 'gene3'],
        #     'acceptor_sites': [
        #         [{'start': 100, 'end': 150, 'position': 125}, {'start': 200, 'end': 250, 'position': 225}],
        #         [{'start': 300, 'end': 350, 'position': 325}],
        #         [{'start': 500, 'end': 550, 'position': 525}]
        #     ]
        # }
        # acceptor sites looks like [{'start': 100, 'end': 150, 'position': 125}, {'start': 200, 'end': 250, 'position': 225}]
        
        strand = normalize_strand(gene_data['strand'])
        true_acceptor_positions = []  # List to keep track of relative positions of true acceptor sites
        position_to_transcript = defaultdict(set)  # Dictionary to map relative positions to transcript IDs

        for site in acceptor_sites:
            if strand == '+':
                relative_position = site['position'] - gene_data['gene_start']
            elif strand == '-':
                relative_position = gene_data['gene_end'] - site['position']
            else:
                raise ValueError(f"Invalid strand value: {strand}")

            # Append the relative position
            true_acceptor_positions.append(relative_position)

            # Map position to transcript_id, accounting for shared positions
            position_to_transcript[relative_position].add(site['transcript_id'])

        # Sort positions and convert transcript ID sets to lists
        true_acceptor_positions = np.array(sorted(set(true_acceptor_positions)))
        position_to_transcript = {pos: list(transcripts) for pos, transcripts in position_to_transcript.items()}

        # Align probabilities with donor positions
        acceptor_probabilities = np.array(gene_data['acceptor_prob'])  # Probabilities for all positions
        if predicted_delta_correction:
            # Adjust scores
            acceptor_probabilities = adjust_scores(acceptor_probabilities, strand, 'acceptor')

        # Validate positions within range
        assert true_acceptor_positions.max() < len(acceptor_probabilities), \
            "true_acceptor_positions contain indices out of range for acceptor_probabilities"

        # Scores corresponding to the positions
        true_acceptor_scores = acceptor_probabilities[true_acceptor_positions]
        true_acceptor_scores = np.round(true_acceptor_scores, 3)  # Round for readability
        
        gene_results.update({'n_acceptors': len(true_acceptor_positions)})
        
        if verbose: 
            gene_len = len(acceptor_probabilities)
            gene_start = gene_data['gene_start']  # Absolute start position of the gene
            gene_end = gene_data['gene_end']  # Absolute end position of the gene
            print(f"[test] Gene {gene_id} @ chr={chromosome}: start={gene_start}, end={gene_end}, length={gene_len}=?={gene_end-gene_start+1}")

        # Initialize label prediction vector based on the threshold
        label_predictions = np.array([1 if prob >= threshold else 0 for prob in acceptor_probabilities])

        # Display true and predicted acceptor site positions in terms of relative positions
        positive_prediction_positions = np.where(label_predictions == 1)[0]
        positive_prediction_positions = sorted(positive_prediction_positions)  # 5' to 3' direction

        # Create a list to track if a true acceptor site is missed (for FN counting)
        true_acceptor_status = {i: False for i in range(len(true_acceptor_positions))}
        # fn_positions, fp_positions = [], [] 

        # Loop over each position in the label sequence
        for i, pred_label in enumerate(label_predictions):
            found_in_window = False

            # Check if this position is a true splice site for the current gene
            for idx, true_pos in enumerate(true_acceptor_positions):
                if true_acceptor_status[idx]:  # Skip already processed true acceptor sites
                    continue

                window_start = true_pos - consensus_window
                window_end = true_pos + consensus_window

                if window_start <= i <= window_end:
                    found_in_window = True

                    associated_transcripts = position_to_transcript[true_pos]

                    # Count TP or FN based on the prediction
                    if pred_label == 1:
                        gene_results['TP'] += 1
                        true_acceptor_status[idx] = True

                        # Compute max probability in the window                
                        window_slice = acceptor_probabilities[window_start:window_end+1]
                        if window_slice.size > 0:
                            score_adjusted = np.max(window_slice)
                        else:
                            score_adjusted = acceptor_probabilities[i]
        
                        assert score_adjusted >= threshold, \
                            f"Unexpected score for TP: {score_adjusted} < {threshold}"

                        # Calculate the delta between predicted and true positions
                        delta = i - true_pos  # Discrepancy between predicted and true positions

                        # Append all associated transcript IDs for this position
                        for transcript_id in associated_transcripts:
                            discrepancies.append({
                                'gene_id': gene_id,
                                'transcript_id': transcript_id,
                                'strand': strand,
                                'true_pos': true_pos,
                                'predicted_pos': i,
                                'delta': delta,
                                'score': score_adjusted
                            })

                    break

            # Count non-splice sites (negative examples)
            if not found_in_window:
                is_fp = False
                if pred_label == 1:
                    is_fp = True

                    # Adjust FP count for overlapping genes
                    if adjust_for_overlapping_genes and is_within_overlapping_splice_site(
                        i, consensus_window, gene_id, gene_data, overlapping_genes_metadata, grouped_annotations
                    ):
                        is_fp = False  # A true splice site in overlapping gene(s) -> not a FP
                        n_fp_correction += 1  # Increment the correction count
                    
                    if is_fp:
                        gene_results['FP'] += 1 
                        # fp_positions.append(i) 
                else:
                    # Predicted negative correctly -> TN
                    gene_results['TN'] += 1

        # Count false negatives
        for idx, status in true_acceptor_status.items():
            if not status:
                gene_results['FN'] += 1
                # fn_positions.append(true_acceptor_positions[idx])

        if verbose: 
            print_emphasized(f"[test] Gene {gene_id} @ chr={chromosome}: start={gene_data['gene_start']}, strand={strand}, number of acceptor sites= {len(true_acceptor_positions)})")
            print_with_indent(f"Number of positive predictions @ threshold={threshold}: {sum(label_predictions)}", indent_level=1)
            print_with_indent(f"Relative positions of positive predictions:\n{positive_prediction_positions}\n", indent_level=2)
            print_with_indent(f"Relative positions of true acceptor sites:\n{true_acceptor_positions}\n", indent_level=2)
            print_with_indent(f"Acceptor scores:\n{np.round(true_acceptor_scores, 3)}\n", indent_level=2)

        # Top-K Accuracy Calculation
        if gene_results['n_acceptors'] > 0:
            # Get the top-K predicted acceptor sites
            top_k = gene_results['n_acceptors']
            sorted_pred_positions = np.argsort(-acceptor_probabilities)[:top_k]  # Get top K highest probability indices
            correct_top_k = sum(
                any(abs(pos - true_pos) <= consensus_window for true_pos in true_acceptor_positions)
                for pos in sorted_pred_positions
            )
            gene_results['top_k_accuracy'] = correct_top_k / top_k

        # Step 5: Calculate derived metrics (Precision, Recall, Specificity)
        TP, TN, FP, FN = gene_results['TP'], gene_results['TN'], gene_results['FP'], gene_results['FN']
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        fpr = 1 - specificity  # proportion of negative instances that were incorrectly predicted as positive
        fnr = 1 - recall  # proportion of positive instances that were incorrectly predicted as negative

        gene_results.update({
            'precision': precision, 
            'recall': recall, 
            'specificity': specificity, 
            'f1_score': f1_score, 
            'fpr': fpr,
            'fnr': fnr}
        )

        # Debugging: Check output for the gene
        if verbose:
            print_with_indent(f"TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}", indent_level=2)
            print_with_indent(f"Precision: {precision}, Recall: {recall}, Specificity: {specificity}, F1 Score: {f1_score}", indent_level=3)
            print_with_indent(f"FPR: {fpr}, FNR: {fnr}", indent_level=3)

            if n_fp_correction > 0:
                print_with_indent(f"Number of FPs corrected due to overlapping genes: {n_fp_correction}", indent_level=3)
            print_section_separator(light=True)

        # Append the results for the current gene to the results list
        results_list.append(gene_results)

        n_genes_processed += 1

    # End of loop over genes

    # Convert the results list to a DataFrame
    results_df = pl.DataFrame(results_list)
    discrepancies_df = pl.DataFrame(discrepancies)

    if return_discrepancies:
        return results_df, discrepancies_df

    return results_df

########################################################################################


def boundary_strand_offset(pos: int, tx_start: int, tx_end: int, strand: str) -> int:
    """
    Convert absolute position -> 0-based index for the transcript
    taking into account the strand.
    """
    if strand == '+':
        return pos - tx_start
    elif strand == '-':
        return tx_end - pos
    else:
        raise ValueError(f"Invalid strand: {strand}")


def evaluate_donor_site_predictions_by_transcript(
    annotations_df: pl.DataFrame,
    pred_results: dict,
    threshold: float = 0.9,
    consensus_window: int = 2,
    predicted_delta_correction: bool = False,
    **kargs
):
    """
    Evaluate donor site predictions at the transcript level, using a consensus window 
    and checking the maximum predicted probability in that window.

    No overlapping-gene logic is applied here, because each transcript is handled independently.

    Parameters
    ----------
    annotations_df : pl.DataFrame
        Must include columns ["site_type", "transcript_id", "position", "gene_id"].
        'position' is the absolute genomic position of the donor site.
        'site_type' must contain "donor" for donor sites.
    pred_results : dict
        Keyed by transcript_id, each value is a dict with:
            {
                'strand': str,
                'tx_start': int,
                'tx_end': int,
                'donor_prob': List[float],
                'positions':  List[int],   # if needed
                'gene_id': str,            # optional but recommended
                ...
            }
    threshold : float
        Probability threshold for calling a site a predicted donor.
    consensus_window : int
        +/- window around the true donor site for matching a predicted site.
    predicted_delta_correction : bool
        If True, you could modify the raw donor_prob array via a custom adjust_scores() function.

    Additional kwargs:
    - return_discrepancies : bool
        If True, returns a second DataFrame with row-level info about TPs.
    - verbose : int
        If > 0, prints debugging statements.

    Returns
    -------
    results_df : pl.DataFrame
        One row per transcript with columns:
            [transcript_id, TP, TN, FP, FN, precision, recall, specificity, f1_score, fpr, fnr]
    discrepancies_df : pl.DataFrame
        (Only returned if return_discrepancies=True) 
        Contains row-level info about each TP.
    """
    verbose = kargs.get('verbose', 1)
    return_discrepancies = kargs.get('return_discrepancies', True)

    # Example stub for adjusting probabilities if needed
    # def adjust_scores(prob_array, strand, site_type):
    #     # E.g., pass-through or small shift-based correction
    #     return prob_array

    # 1) Group annotated donor sites by transcript
    grouped_annotations = (
        annotations_df
        .filter(pl.col("site_type") == "donor")
        .group_by("transcript_id")
        .agg(
            pl.struct(["position", "gene_id"]).alias("donor_sites")
        )
        .to_dict(as_series=False)
    )
    transcripts_with_donors = set(grouped_annotations["transcript_id"])

    results_list = []
    discrepancies = []

    # 2) Iterate over each transcript in pred_results
    for transcript_id, tx_data in pred_results.items():
        tx_results = {
            'transcript_id': transcript_id,
            'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0,
            'n_donors': 0
        }

        strand   = tx_data.get('strand', '+')
        tx_start = tx_data.get('tx_start', 0)
        tx_end   = tx_data.get('tx_end',   0)
        gene_id  = tx_data.get('gene_id',  'unknown')

        donor_probabilities = np.array(tx_data['donor_prob'], dtype=float)
        if predicted_delta_correction:
            donor_probabilities = adjust_scores(donor_probabilities, strand, 'donor')

        # If no annotated donors for this transcript
        if transcript_id not in transcripts_with_donors:
            tx_results.update({
                'precision': None, 'recall': None, 'specificity': None,
                'f1_score': None, 'fpr': None, 'fnr': None
            })
            results_list.append(tx_results)
            continue

        # 3) Get the annotated donor sites (positions)
        donor_sites_struct_list = grouped_annotations['donor_sites'][
            grouped_annotations['transcript_id'].index(transcript_id)
        ]

        # Convert absolute positions -> relative indices
        true_donor_positions = []
        for sdict in donor_sites_struct_list:
            abs_pos = sdict['position']
            rel_pos = boundary_strand_offset(abs_pos, tx_start, tx_end, strand)
            true_donor_positions.append(rel_pos)

        true_donor_positions = np.array(sorted(set(true_donor_positions)))
        tx_results['n_donors'] = len(true_donor_positions)

        # Range check
        if len(true_donor_positions) > 0:
            max_true_pos = true_donor_positions.max()
            if max_true_pos >= len(donor_probabilities):
                raise ValueError(
                    f"[Error] {transcript_id} has donor_pos out of range. "
                    f"Max donor pos {max_true_pos}, donor_prob length {len(donor_probabilities)}"
                )

        # 4) Create a threshold-based label for each position
        label_predictions = np.array([1 if p >= threshold else 0 for p in donor_probabilities])

        # Mark whether each annotated site got matched
        matched_site = [False]*len(true_donor_positions)

        # 5) Evaluate predictions at each position i
        for i, pred_label in enumerate(label_predictions):
            found_in_window = False

            for idx, tpos in enumerate(true_donor_positions):
                if matched_site[idx]:  # Skip already processed true donor sites
                    continue

                window_start = tpos - consensus_window
                window_end   = tpos + consensus_window

                if window_start <= i <= window_end:
                    found_in_window = True

                    if pred_label == 1:
                        # compute max prob in [window_start, window_end]
                        ws = max(window_start, 0)
                        we = min(window_end+1, len(donor_probabilities))
                        window_slice = donor_probabilities[ws:we]
                        score_adjusted = window_slice.max() if window_slice.size > 0 else donor_probabilities[i]

                        # Confirm the max score is above threshold
                        assert score_adjusted >= threshold, \
                            f"Unexpected: label=1, but max prob in window={score_adjusted} < {threshold}"

                        # => TP
                        tx_results['TP'] += 1
                        matched_site[idx] = True
                        delta = i - tpos

                        # Record discrepancy
                        discrepancies.append({
                            'transcript_id': transcript_id,
                            'gene_id': gene_id,
                            'strand': strand,
                            'true_pos': tpos,
                            'predicted_pos': i,
                            'delta': delta,
                            'score_adjusted': float(score_adjusted)
                        })

                    break  # no need to check other donor sites once found in window

            if not found_in_window:
                # => either FP or TN
                if pred_label == 1:
                    tx_results['FP'] += 1
                else:
                    tx_results['TN'] += 1

        # 6) Count unmatched donor sites => FN
        for idx, matched in enumerate(matched_site):
            if not matched:
                tx_results['FN'] += 1

        # 7) Compute derived metrics
        TP, TN, FP, FN = tx_results['TP'], tx_results['TN'], tx_results['FP'], tx_results['FN']
        precision   = TP / (TP + FP) if (TP + FP) > 0 else None
        recall      = TP / (TP + FN) if (TP + FN) > 0 else None
        specificity = TN / (TN + FP) if (TN + FP) > 0 else None
        f1_score    = 2*(precision*recall)/(precision+recall) if (precision and recall) else None
        fpr         = 1 - specificity if specificity is not None else None
        fnr         = 1 - recall if recall is not None else None

        tx_results.update({
            'precision': precision, 'recall': recall, 'specificity': specificity,
            'f1_score': f1_score, 'fpr': fpr, 'fnr': fnr
        })

        # 8) Store
        results_list.append(tx_results)

    # end of loop over transcripts

    # 9) Convert to Polars DataFrames
    results_df = pl.DataFrame(results_list)
    discrepancies_df = pl.DataFrame(discrepancies)

    if return_discrepancies:
        return results_df, discrepancies_df
    return results_df


def evaluate_acceptor_site_predictions_by_transcript(
    annotations_df: pl.DataFrame,
    pred_results: dict,
    threshold: float = 0.9,
    consensus_window: int = 2,
    **kargs
):
    """
    Evaluate SpliceAI predictions for acceptor splice sites at the *transcript* level.

    Parameters
    ----------
    annotations_df : pl.DataFrame
        Contains the *true* splice site annotations. Must include columns like:
          ['transcript_id', 'position', 'site_type', 'gene_id'] (plus any others).
        'site_type' must indicate "acceptor" for acceptor sites.
        'position' is typically the absolute genomic position of the acceptor site.
    pred_results : dict
        The output (in "efficient" form) of your `predict_splice_sites(..., efficient_output=True)` 
        or a similar structure, keyed by *transcript_id*, e.g.:
        {
          transcript_id: {
            'strand': '+',
            'tx_start': 12345,
            'tx_end': 12890,
            'acceptor_prob': [... per-base probabilities ...],
            'gene_id': 'GENE_XYZ',   # optional
            ...
          },
          ...
        }
    threshold : float
        Probability threshold for classifying a prediction as an acceptor site.
    consensus_window : int
        Tolerance window in +/- bases around each true acceptor site.
    **kargs : dict
        - return_discrepancies : bool
            If True, returns a second DataFrame describing the TPs with positional deltas, etc.
        - predicted_delta_correction : bool
            If True, you may adjust scores (e.g. shift them) before thresholding.

    Returns
    -------
    results_df : pl.DataFrame
        Columns: [transcript_id, TP, TN, FP, FN, n_acceptors, precision, recall, specificity, f1_score, fpr, fnr].
    discrepancies_df : pl.DataFrame
        (Only if return_discrepancies=True)
        Row-level info for each TP, e.g. [transcript_id, gene_id, strand, true_pos, predicted_pos, delta, score].
    """

    verbose = kargs.get("verbose", 1)
    return_discrepancies = kargs.get("return_discrepancies", True)
    predicted_delta_correction = kargs.get("predicted_delta_correction", True)

    # Example stub: a function to adjust acceptor probabilities if needed
    # def adjust_scores(prob_array, strand, site_type):
    #     # E.g. pass-through or shift-based correction
    #     return prob_array

    # 1) Group true acceptor sites by transcript_id
    grouped_annotations = (
        annotations_df
        .filter(pl.col("site_type") == "acceptor")
        .group_by("transcript_id")
        .agg(pl.struct(["position", "gene_id"]).alias("acceptor_sites"))
        .to_dict(as_series=False)
    )
    transcripts_with_acceptors = set(grouped_annotations["transcript_id"])

    results_list = []
    discrepancies = []

    # 2) Iterate over each transcript in pred_results
    for transcript_id, tx_data in pred_results.items():
        tx_results = {
            'transcript_id': transcript_id,
            'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0,
            'n_acceptors': 0
        }

        strand   = tx_data.get('strand', '+')
        tx_start = tx_data.get('tx_start', 0)
        tx_end   = tx_data.get('tx_end',   0)
        gene_id  = tx_data.get('gene_id',  'unknown')  # If you have gene_id in pred_results

        # 3) Retrieve acceptor probabilities
        acceptor_probabilities = np.array(tx_data['acceptor_prob'], dtype=float)
        if predicted_delta_correction:
            acceptor_probabilities = adjust_scores(acceptor_probabilities, strand, 'acceptor')

        # If transcript has no annotated acceptor sites
        if transcript_id not in transcripts_with_acceptors:
            if verbose:
                print(f"No acceptor annotations for transcript: {transcript_id}")
            # Mark metrics as None
            tx_results.update({
                'precision': None, 'recall': None, 'specificity': None,
                'f1_score': None, 'fpr': None, 'fnr': None
            })
            results_list.append(tx_results)
            continue

        # 4) Get the annotated acceptor sites for this transcript
        acceptor_sites_struct_list = grouped_annotations['acceptor_sites'][
            grouped_annotations['transcript_id'].index(transcript_id)
        ]
        # Convert absolute position -> 0-based index for this transcript
        true_acceptor_positions = []
        for site_info in acceptor_sites_struct_list:
            abs_pos = site_info['position']
            rel_pos = boundary_strand_offset(abs_pos, tx_start, tx_end, strand)
            true_acceptor_positions.append(rel_pos)

        true_acceptor_positions = np.array(sorted(set(true_acceptor_positions)))
        tx_results['n_acceptors'] = len(true_acceptor_positions)

        # 5) Check index range
        if len(true_acceptor_positions) > 0:
            max_pos = true_acceptor_positions.max()
            if max_pos >= len(acceptor_probabilities):
                raise ValueError(
                    f"[Error] {transcript_id}: acceptor pos out of range. "
                    f"Max pos {max_pos}, prob array len {len(acceptor_probabilities)}"
                )

        # 6) Label predictions based on threshold
        label_predictions = np.array([1 if p >= threshold else 0 for p in acceptor_probabilities])

        # Keep track of matched true sites => for FN count
        matched_acceptors = [False]*len(true_acceptor_positions)

        # 7) Evaluate each position i
        for i, pred_label in enumerate(label_predictions):
            found_in_window = False

            for idx, true_pos in enumerate(true_acceptor_positions):
                if matched_acceptors[idx]:  # Skip already processed true acceptor sites
                    continue

                wstart = true_pos - consensus_window
                wend   = true_pos + consensus_window

                if wstart <= i <= wend:
                    # => position i is within window of true_pos
                    found_in_window = True

                    if pred_label == 1:
                        # => candidate TP, double-check max prob in window
                        ws = max(wstart, 0)
                        we = min(wend+1, len(acceptor_probabilities))
                        window_slice = acceptor_probabilities[ws:we]

                        if window_slice.size > 0:
                            score_adjusted = np.max(window_slice)
                        else:
                            score_adjusted = acceptor_probabilities[i]

                        # Enforce that score_adjusted >= threshold
                        assert score_adjusted >= threshold, \
                            f"Unexpected TP: score_adjusted {score_adjusted} < threshold {threshold}"

                        # => Indeed a TP
                        tx_results['TP'] += 1
                        matched_acceptors[idx] = True
                        delta = i - true_pos

                        # Record a discrepancy row
                        discrepancies.append({
                            'transcript_id': transcript_id,
                            'gene_id': gene_id,
                            'strand': strand,
                            'true_pos': int(true_pos),
                            'predicted_pos': int(i),
                            'delta': int(delta),
                            'score': float(score_adjusted)
                        })

                    break  # done checking other true positions

            if not found_in_window:
                # => negative example
                if pred_label == 1:
                    tx_results['FP'] += 1
                else:
                    tx_results['TN'] += 1

        # 8) Count FNs
        for idx, matched in enumerate(matched_acceptors):
            if not matched:
                tx_results['FN'] += 1

        # 9) Compute derived metrics
        TP, TN, FP, FN = tx_results['TP'], tx_results['TN'], tx_results['FP'], tx_results['FN']
        precision   = TP / (TP + FP) if (TP + FP) > 0 else None
        recall      = TP / (TP + FN) if (TP + FN) > 0 else None
        specificity = TN / (TN + FP) if (TN + FP) > 0 else None
        f1_score    = 2 * (precision*recall)/(precision+recall) if (precision and recall) else None
        fpr         = 1 - specificity if specificity is not None else None
        fnr         = 1 - recall if recall is not None else None

        tx_results.update({
            'precision': precision,
            'recall': recall,
            'specificity': specificity,
            'f1_score': f1_score,
            'fpr': fpr,
            'fnr': fnr
        })

        if verbose > 0:
            print(f"[debug] Transcript {transcript_id} -> TP={TP}, FP={FP}, TN={TN}, FN={FN}, #acceptors={len(true_acceptor_positions)}")

        results_list.append(tx_results)

    # End for transcript_id

    # Create final Polars DataFrames
    results_df = pl.DataFrame(results_list)
    discrepancies_df = pl.DataFrame(discrepancies)

    if return_discrepancies:
        return results_df, discrepancies_df
    return results_df



########################################################################################


def evaluate_donor_site_errors(
    annotations_df, 
    pred_results, 
    threshold=0.5, 
    consensus_window=2, 
    error_window=500, 
    collect_tn=False,
    tn_sample_factor=1.2,
    tn_sampling_mode="random",
    tn_proximity_radius=50,
    **kargs
):
    """
    Evaluate SpliceAI predictions for donor splice sites and identify FPs and FNs, including windowed regions.

    Parameters:
    - annotations_df (pl.DataFrame): DataFrame containing true splice site annotations.
      Columns: ['chrom', 'start', 'end', 'strand', 'site_type', 'gene_id', 'transcript_id']
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
    - tn_proximity_radius : int
        Radius for measuring closeness if tn_sampling_mode="proximity".
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
    return_positions_df = kargs.get('return_positions_df', False)
    predicted_delta_correction = kargs.get('predicted_delta_correction', True)

    if overlapping_genes_metadata is None:
        sa = SpliceAnalyzer()
        overlapping_genes_metadata = sa.retrieve_overlapping_gene_metadata()

    error_list = []
    positions_list = []

    # We'll store TN in a separate structure until we know how to sample them:
    tn_collection = []

    # Group true donor annotations by gene
    grouped_annotations = annotations_df.filter(pl.col('site_type') == 'donor').group_by('gene_id').agg(
        pl.struct(['start', 'end', 'position', 'transcript_id']).alias('donor_sites')
    ).to_dict(as_series=False)  # 'transcript_id'

    # Process each gene's predictions in pred_results
    for gene_id, gene_data in pred_results.items():
        if gene_id not in grouped_annotations['gene_id']:
            if verbose: 
                print(f"No donor annotations for gene: {gene_id}")

            # Optional: Append an entry for these genes with no donor annotations
            error_list.append({
                'gene_id': gene_id,
                'transcript_id': None,
                'error_type': None,  # No error type, since there are no donor annotations
                'position': None,
                'window_start': None,
                'window_end': None, 
                'strand': None
            })

            positions_list.append({
                'gene_id': gene_id,
                'transcript_id': None,
                'position': None,
                'pred_type': None,
                'score': None,
                'strand': None
            })

            continue

        
        # Extract relevant info
        strand = normalize_strand(gene_data['strand'])
        donor_probabilities = np.array(gene_data['donor_prob'])  # Probabilities for all positions
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
            # Adjust scores
            donor_probabilities = adjust_scores(donor_probabilities, strand, 'donor')

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

                if window_start <= i <= window_end:
                    found_in_window = True 

                    # Get the associated transcript IDs for this true position
                    associated_transcripts = position_to_transcript[true_pos]   
                    
                    # Count TP or FN based on the prediction
                    if pred_label == 1:
                        gene_results['TP'] += 1
                        true_donor_status[idx] = True

                        assert donor_probabilities[i] >= threshold, f"Unexpected score for TP: {donor_probabilities[i]} < {threshold}"
                        
                        score = donor_probabilities[i]
                        # score = max(donor_probabilities[window_start:window_end+1])

                        # Append all associated transcript IDs for this position
                        for transcript_id in associated_transcripts:
                            tp_positions.append({
                                'gene_id': gene_id,
                                'transcript_id': transcript_id,
                                'position': i,
                                'pred_type': 'TP',

                                # max(donor_probabilities[window_start:window_end+1])
                                'score': score,

                                'strand': strand
                            })
                    break

            # Count non-splice sites (negative examples)
            if not found_in_window:
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

                        # FP positions are not associated with transcripts since they're not true splice sites
                        fp_positions.append({
                            'gene_id': gene_id,
                            'transcript_id': None,
                            'position': i,
                            'pred_type': 'FP',
                            'score': donor_probabilities[i],
                            'strand': strand,  
                        })

                else:
                    # This is a negative prediction => candidate TN
                    # We'll store it and sample later
                    tn_positions_all.append({
                        'gene_id': gene_id,
                        'transcript_id': None,
                        'position': i,
                        'pred_type': 'TN',
                        'score': donor_probabilities[i],
                        'strand': strand
                    })

                    # Predicted negative correctly -> TN
                    gene_results['TN'] += 1
        # End for each position in the label sequence

        # Count false negatives
        for idx, status in true_donor_status.items():
            if not status:
                # score = true_donor_scores[idx]
                position = true_donor_positions[idx]  # true relative position
                score = donor_probabilities[position]
                label = label_predictions[position]

                assert 0 <= position < len(label_predictions), \
                    f"Position {position} out of bounds for label_predictions array of length {len(label_predictions)}"

                # Get the associated transcript IDs for this true position
                associated_transcripts = position_to_transcript[position]

                if verbose:
                    start = max(0, position - 2)
                    end = min(len(donor_probabilities), position + 3)
                    surrounding_scores = donor_probabilities[start:end]
                    print(f"[DEBUG] FN: gene_id={gene_id}, position={position}, score={score}, label={label}  threshold={threshold}")
                    print(f"[DEBUG] Surrounding scores for FN at {position}: {surrounding_scores}")

                # assert label == 0, f"Unexpected label for FN: {label} != 0"  # Todo
                # assert score <= threshold, f"Unexpected score for FN: {score} > {threshold}, label={label}"  # Todo
                
                gene_results['FN'] += 1
                
                # Append FN positions with all associated transcript IDs
                for transcript_id in associated_transcripts:
                    fn_positions.append({
                        'gene_id': gene_id,
                        'transcript_id': transcript_id, 
                        'position': position,
                        'pred_type': 'FN',
                        'score': score,
                        'strand': strand
                    })
        # End for each true donor site and the counting of FNs

        # Add FP/FN/TP to positions_list right away
        positions_list.extend(fp_positions)
        positions_list.extend(fn_positions)
        positions_list.extend(tp_positions)

        # ------------------------------------------
        
        # If collecting TN => we handle them now
        if collect_tn:
            # (A) we can just keep them all (dangerous if gene is huge)
            # (B) sample them with preference or random
            # We'll do a two-step approach: we gather them all in `tn_positions_all` 
            # then sample below.

            # We'll store them into a separate structure for now; 
            # we do final sampling after the loop for each gene or after all genes 
            # (depending on your preference).
            # For demonstration, let's just do gene-level sampling to keep it simpler.

            # 1) figure out how many “non-FP/FN/TP” we have
            num_fp = len(fp_positions)
            num_fn = len(fn_positions)
            num_tp = len(tp_positions)
            # total of “other classes”
            total_non_tn = num_fp + num_fn + num_tp
            desired_tn_count = int(tn_sample_factor * total_non_tn)  # e.g. 2.0 * total_non_tn

            # If tn_positions_all is less than desired, keep them all
            if len(tn_positions_all) <= desired_tn_count:
                tn_collection.extend(tn_positions_all)
            else:
                # We have more than we want => sample
                if tn_sampling_mode == "random":
                    # pick random subset
                    tn_subset = random.sample(tn_positions_all, desired_tn_count)
                    tn_collection.extend(tn_subset)
                elif tn_sampling_mode == "proximity":
                    # prefer positions near TPs or FNs
                    # we can measure distance to nearest TP or FN
                    # simplest approach: build a set of “key positions” from TP/FN
                    key_positions = [p['position'] for p in (tp_positions + fn_positions)]
                    # Then compute min distance for each TN
                    if key_positions:
                        tn_with_dists = []
                        for tnrow in tn_positions_all:
                            tnpos = tnrow['position']
                            dists = [abs(tnpos - kp) for kp in key_positions]
                            mindist = min(dists) if dists else 999999
                            tn_with_dists.append((tnrow, mindist))
                        # sort by distance ascending
                        tn_with_dists.sort(key=lambda x: x[1])
                        # pick the top desired_tn_count
                        tn_subset = [x[0] for x in tn_with_dists[:desired_tn_count]]
                        tn_collection.extend(tn_subset)
                    else:
                        # if no TPs or FNs, random sample anyway
                        tn_subset = random.sample(tn_positions_all, desired_tn_count)
                        tn_collection.extend(tn_subset)
                else:
                    # fallback
                    tn_subset = random.sample(tn_positions_all, desired_tn_count)
                    tn_collection.extend(tn_subset)

        # ------------------------------------------
        # Debugging: Display true donor positions and their surrounding probabilities
        if verbose:
            # Extend donor probabilities with neighbors
            neighbor_range = 2  # Extend by ±2 nucleotides
            true_donor_scores_with_neighbors = []

            for pos in true_donor_positions:
                neighbor_scores = donor_probabilities[max(0, pos - neighbor_range):min(len(donor_probabilities), pos + neighbor_range + 1)]
                true_donor_scores_with_neighbors.append((pos, list(neighbor_scores)))

            print(f"[analysis] Gene {gene_id} @ strand={strand}, chr={chromosome}:")
            print(f"  True donor positions (relative): {true_donor_positions}")
            print(f"  Donor probabilities with neighbors (±{neighbor_range} nts):")
            print(f"     Donor score adjusted? {predicted_delta_correction}")
            for pos, scores in true_donor_scores_with_neighbors:
                print(f"    Position {pos}: {scores}")

        # ------------------------------------------

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

    # If we did collect TN => add them to positions_list
    if collect_tn and tn_collection:
        positions_list.extend(tn_collection)  # tn_collection is a subset of tn_positions_all

    # Convert error list to DataFrame
    error_df = pl.DataFrame(error_list)
    positions_df = pl.DataFrame(positions_list)

    if return_positions_df:
        return error_df, positions_df
    
    return error_df


def evaluate_acceptor_site_errors(
    annotations_df, 
    pred_results, 
    threshold=0.9, 
    consensus_window=2, 
    error_window=500, 
    collect_tn=False,
    tn_sample_factor=1.2,
    tn_sampling_mode="random",
    tn_proximity_radius=50,
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
        Threshold for classifying a position as an acceptor site (default 0.9).
    consensus_window : int
        Tolerance window around true splice sites for matching (TP or FN).
    error_window : int or tuple
        Window size for collecting the local sequence region around FPs or FNs.
        If int, we treat it as (error_window, error_window).
    collect_tn : bool
        Whether to sample a subset of TNs. Defaults to False.
    tn_sample_factor : float
        Ratio controlling how many TN rows to keep, relative to the total of (TP+FP+FN).
        E.g., if we have 100 TPs+FPs+FNs, then we keep up to 2.0 * 100 = 200 TN positions.
    tn_sampling_mode : str
        "random" => random sampling of TNs
        "proximity" => prefer positions near TPs or FNs (within tn_proximity_radius).
    tn_proximity_radius : int
        Radius used if tn_sampling_mode="proximity".
    **kargs :
        Additional arguments, e.g.:
        - overlapping_genes_metadata
        - adjust_for_overlapping_genes
        - return_positions_df
        - predicted_delta_correction
        - verbose

    Returns
    -------
    error_df : pl.DataFrame
        A DataFrame containing positions of FPs and FNs, each with a window around the error.
        Columns: ['gene_id', 'transcript_id', 'error_type', 'position', 'window_start',
                  'window_end', 'strand']
    positions_df : pl.DataFrame (only if return_positions_df=True)
        A DataFrame containing all identified sites: TP, FP, FN, and possibly TN (if collect_tn=True).
        Columns: ['gene_id', 'transcript_id', 'position', 'pred_type', 'score', 'strand']
    """
    # from .extract_genomic_features import get_overlapping_gene_metadata

    verbose = kargs.get('verbose', 1)
    chromosome = kargs.get('chromosome', kargs.get('chr', '?'))
    overlapping_genes_metadata = kargs.get('overlapping_genes_metadata', None)
    adjust_for_overlapping_genes = kargs.get('adjust_for_overlapping_genes', True)
    return_positions_df = kargs.get('return_positions_df', False)
    predicted_delta_correction = kargs.get('predicted_delta_correction', True)

    if overlapping_genes_metadata is None:
        sa = SpliceAnalyzer()
        overlapping_genes_metadata = sa.retrieve_overlapping_gene_metadata()

    error_list = []
    positions_list = []
    tn_collection = []  # We'll store final sampled TN data here

    # Group true acceptor annotations by gene
    grouped_annotations = (
        annotations_df
        .filter(pl.col('site_type') == 'acceptor')
        .group_by('gene_id')
        .agg(pl.struct(['start', 'end', 'position', 'transcript_id']).alias('acceptor_sites'))
        .to_dict(as_series=False)  # => { 'gene_id': [...], 'acceptor_sites': [...]}
    )

    # Loop over each gene in pred_results
    for gene_id, gene_data in pred_results.items():
        if gene_id not in grouped_annotations['gene_id']:
            # No acceptor annotations for this gene => store placeholders
            if verbose:
                print(f"No acceptor annotations for gene: {gene_id}")

            error_list.append({
                'gene_id': gene_id,
                'transcript_id': None,
                'error_type': None,  # no acceptor annotation
                'position': None,
                'window_start': None,
                'window_end': None,
                'strand': None
            })
            positions_list.append({
                'gene_id': gene_id,
                'transcript_id': None,
                'position': None,
                'pred_type': None,
                'score': None,
                'strand': None
            })
            continue

        # Extract acceptor annotations
        acceptor_sites = grouped_annotations['acceptor_sites'][grouped_annotations['gene_id'].index(gene_id)]
        strand = normalize_strand(gene_data['strand'])

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
        if predicted_delta_correction:
            acceptor_probabilities = adjust_scores(acceptor_probabilities, strand, 'acceptor')

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
                if true_acceptor_status[idx]:  # Skip already processed true acceptor sites
                    continue

                wstart = true_pos - consensus_window
                wend   = true_pos + consensus_window
                if wstart <= i <= wend:
                    found_in_window = True
                    # If pred_label=1 => TP
                    if pred_label == 1:
                        gene_results['TP'] += 1
                        true_acceptor_status[idx] = True

                        # Some checks
                        assert acceptor_probabilities[i] >= threshold, \
                            f"Unexpected score for TP: {acceptor_probabilities[i]} < {threshold}"

                        score_val = acceptor_probabilities[i]
                        associated_tids = position_to_transcript[true_pos]

                        # store them
                        for t_id in associated_tids:
                            tp_positions.append({
                                'gene_id': gene_id,
                                'transcript_id': t_id,
                                'position': i,
                                'pred_type': 'TP',
                                'score': score_val,
                                'strand': strand
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
                        fp_positions.append({
                            'gene_id': gene_id,
                            'transcript_id': None,
                            'position': i,
                            'pred_type': 'FP',
                            'score': acceptor_probabilities[i],
                            'strand': strand
                        })
                else:
                    # => predicted negative => candidate TN
                    gene_results['TN'] += 1
                    tn_positions_all.append({
                        'gene_id': gene_id,
                        'transcript_id': None,
                        'position': i,
                        'pred_type': 'TN',
                        'score': acceptor_probabilities[i],
                        'strand': strand
                    })

        # Now check for FN (i.e. real acceptor that never got labeled=1)
        for idx, matched in true_acceptor_status.items():
            if not matched:
                # => FN
                position = true_acceptor_positions[idx]
                score_val = acceptor_probabilities[position]
                label_val = label_predictions[position]
                gene_results['FN'] += 1

                associated_tids = position_to_transcript[position]
                if verbose:
                    rng_start = max(0, position - 2)
                    rng_end   = min(gene_len, position + 3)
                    neighbors = acceptor_probabilities[rng_start:rng_end]
                    print(f"[DEBUG] FN: gene_id={gene_id}, pos={position}, "
                          f"score={score_val}, label={label_val}, threshold={threshold}")
                    print(f"[DEBUG] Surrounding scores at pos={position}: {neighbors}")

                for t_id in associated_tids:
                    fn_positions.append({
                        'gene_id': gene_id,
                        'transcript_id': t_id,
                        'position': position,
                        'pred_type': 'FN',
                        'score': score_val,
                        'strand': strand
                    })

        # Debugging info for the neighbor distribution
        if verbose:
            neighbor_range = 2
            print(f"[analysis] Gene {gene_id} @ strand={strand}, chr={chromosome}:")
            print(f"  True acceptor positions (relative): {true_acceptor_positions}")
            if len(true_acceptor_positions) > 0:
                acceptor_scores_with_neighbors = []
                for pos in true_acceptor_positions:
                    n_start = max(0, pos - neighbor_range)
                    n_end   = min(gene_len, pos + neighbor_range + 1)
                    acceptor_scores_with_neighbors.append(
                        (pos, list(acceptor_probabilities[n_start:n_end]))
                    )
                print(f"  Acceptor probabilities ±{neighbor_range} nt around each real site:")
                for ppos, scarr in acceptor_scores_with_neighbors:
                    print(f"    pos={ppos}, neighbors={scarr}")

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

        # For final positions listing
        positions_list.extend(fp_positions)
        positions_list.extend(fn_positions)
        positions_list.extend(tp_positions)

        # Handle TN if collect_tn=True
        if collect_tn:
            # Count how many TPs+FPs+FNs we got for this gene
            num_fp = len(fp_positions)
            num_fn = len(fn_positions)
            num_tp = len(tp_positions)
            total_non_tn = num_fp + num_fn + num_tp
            desired_tn_count = int(tn_sample_factor * total_non_tn)

            if len(tn_positions_all) <= desired_tn_count:
                tn_collection.extend(tn_positions_all)
            else:
                # sampling
                if tn_sampling_mode == "random":
                    tn_subset = random.sample(tn_positions_all, desired_tn_count)
                    tn_collection.extend(tn_subset)

                elif tn_sampling_mode == "proximity":
                    # measure distance to nearest TP or FN
                    key_positions = [r['position'] for r in (tp_positions + fn_positions)]
                    if key_positions:
                        tn_with_dists = []
                        for tn_row in tn_positions_all:
                            dist_list = [abs(tn_row['position'] - kp) for kp in key_positions]
                            min_dist = min(dist_list) if dist_list else 999999
                            tn_with_dists.append((tn_row, min_dist))
                        # sort ascending
                        tn_with_dists.sort(key=lambda x: x[1])
                        tn_subset = [x[0] for x in tn_with_dists[:desired_tn_count]]
                        tn_collection.extend(tn_subset)
                    else:
                        # fallback random
                        tn_subset = random.sample(tn_positions_all, desired_tn_count)
                        tn_collection.extend(tn_subset)

                else:
                    # default fallback => random
                    tn_subset = random.sample(tn_positions_all, desired_tn_count)
                    tn_collection.extend(tn_subset)

    # End for gene_id in pred_results

    # If collecting TN => add them to positions_list
    if collect_tn and len(tn_collection) > 0:
        positions_list.extend(tn_collection)

    error_df = pl.DataFrame(error_list)
    positions_df = pl.DataFrame(positions_list)

    if kargs.get('return_positions_df', False):
        return error_df, positions_df
    return error_df


########################################################################################


def evaluate_donor_site_errors_by_transcript(
    annotations_df: pl.DataFrame,
    pred_results: dict,
    threshold: float = 0.5,
    consensus_window: int = 2,
    error_window: int = 500,
    predicted_delta_correction: bool = True,
    **kargs
):
    """
    Evaluate transcript-level donor site predictions, collecting error positions (TP, FP, FN)
    and optionally adjusting scores to account for systematic +1/+2 nt offsets.

    Parameters
    ----------
    annotations_df : pl.DataFrame
        Must have: ['transcript_id', 'position', 'site_type', 'strand', ...] 
        'site_type' must contain "donor".
        'position' is the absolute genomic position of the donor site.
    pred_results : dict
        Keyed by transcript_id. Each value is a dict, e.g.:
          {
            'strand': str ('+' or '-'),
            'tx_start': int,
            'tx_end': int,
            'donor_prob': List[float],
            ...
          }
    threshold : float
        Probability threshold for classifying a position as a donor site.
    consensus_window : int
        +/- bases around a true donor site to consider a predicted site matching it (TP).
    error_window : int or tuple
        Window size for collecting the surrounding region of FPs and FNs.
    predicted_delta_correction : bool
        If True, apply adjust_scores() to shift probabilities before thresholding.
    **kargs
        - verbose (int): Print debugging info if > 0.
        - return_positions_df (bool): If True, return both error_df and positions_df.

    Returns
    -------
    error_df : pl.DataFrame
        Rows for each FP and FN, with columns:
            [transcript_id, error_type, position, window_start, window_end, strand]
    positions_df : pl.DataFrame
        (only if return_positions_df=True)
        Contains rows for TP, FP, FN, each with columns:
            [transcript_id, position, pred_type, score, strand]
    """
    verbose = kargs.get('verbose', 1)
    return_positions_df = kargs.get('return_positions_df', False)

    # 1) Group true donor sites by transcript
    grouped_annotations = (
        annotations_df
        .filter(pl.col('site_type') == 'donor')
        .group_by("transcript_id")
        .agg(
            pl.struct(["position"]).alias("donor_sites")
        )
        .to_dict(as_series=False)
    )
    transcripts_with_donors = set(grouped_annotations["transcript_id"])

    error_list = []
    positions_list = []  # to store TPs, FPs, FNs

    # 2) Iterate over transcripts
    for transcript_id, tx_data in pred_results.items():
        # Retrieve gene_id from pred_results if present
        gene_id = tx_data.get('gene_id', 'unknown')

        # If no donor annotations, skip (or log a message)
        if transcript_id not in transcripts_with_donors:
            if verbose > 0:
                print(f"No donor annotations for transcript: {transcript_id}")

            error_list.append({
                'transcript_id': transcript_id,
                'gene_id': gene_id,
                'error_type': None,
                'position': None,
                'window_start': None,
                'window_end': None,
                'strand': None
            })

            positions_list.append(
                {
                    'transcript_id': transcript_id,
                    'gene_id': gene_id,
                    'position': None,
                    'pred_type': None,
                    'score': None,
                    'strand': None
                }
            )

            continue

        strand = tx_data.get('strand', '+')
        tx_start = tx_data.get('tx_start', 0)
        tx_end   = tx_data.get('tx_end',   0)

        # 3) Gather annotated donor sites
        donor_sites_struct_list = grouped_annotations['donor_sites'][
            grouped_annotations['transcript_id'].index(transcript_id)
        ]
        # Convert absolute -> relative index
        true_donor_positions = []
        for site_info in donor_sites_struct_list:
            abs_pos = site_info['position']
            rel_pos = boundary_strand_offset(abs_pos, tx_start, tx_end, strand)
            true_donor_positions.append(rel_pos)
        true_donor_positions = np.array(sorted(set(true_donor_positions)))

        # 4) Pull out raw donor_prob
        donor_probabilities = np.array(tx_data['donor_prob'], dtype=float)

        # If needed, apply the systematic offset correction
        if predicted_delta_correction:
            donor_probabilities = adjust_scores(donor_probabilities, strand, splice_type='donor')

        # Validate range
        if len(true_donor_positions) > 0:
            max_true = true_donor_positions.max()
            if max_true >= len(donor_probabilities):
                raise ValueError(
                    f"[Error] Transcript {transcript_id}: true donor pos out of range. "
                    f"Max pos={max_true}, len(donor_prob)={len(donor_probabilities)}"
                )

        # We'll do a simple threshold-based classification for each position
        # But we also do a max-in-window approach for TPs
        label_predictions = np.array([1 if p >= threshold else 0 for p in donor_probabilities])

        # Prepare lists
        tp_positions = []
        fp_positions = []
        fn_positions = []
        matched_donors = [False]*len(true_donor_positions)

        # 5) Evaluate each position i
        for i, pred_label in enumerate(label_predictions):
            found_in_window = False

            # Check if i is within +/- consensus_window of any true donor
            for idx, true_pos in enumerate(true_donor_positions):
                if matched_donors[idx]:  # Skip already processed true donor sites
                    continue

                wstart = true_pos - consensus_window
                wend   = true_pos + consensus_window

                if wstart <= i <= wend:
                    found_in_window = True
                    if pred_label == 1:
                        # => potential TP
                        # Optionally, compute the max probability in that window
                        ws = max(wstart, 0)
                        we = min(wend+1, len(donor_probabilities))
                        window_slice = donor_probabilities[ws:we]
                        score_adjusted = window_slice.max() if window_slice.size > 0 else donor_probabilities[i]

                        # Confirm the max prob is above threshold
                        assert score_adjusted >= threshold, (
                            f"Unexpected: predicted=1, but max prob in window={score_adjusted} < threshold={threshold}"
                        )

                        tp_positions.append({
                            'transcript_id': transcript_id,
                            'gene_id': gene_id,
                            'position': i,
                            'pred_type': 'TP',
                            'score': score_adjusted,
                            'strand': strand
                        })
                        matched_donors[idx] = True
                    break  # done checking other donors

            if not found_in_window:
                if pred_label == 1:
                    # => FP
                    fp_positions.append({
                        'transcript_id': transcript_id,
                        'gene_id': gene_id,
                        'position': i,
                        'pred_type': 'FP',
                        'score': donor_probabilities[i],
                        'strand': strand
                    })
                # else => TN => we ignore

        # 7) Count FNs: any true donor not matched
        for idx, matched in enumerate(matched_donors):
            if not matched:
                pos = true_donor_positions[idx]
                score = donor_probabilities[pos]

                fn_positions.append({
                    'transcript_id': transcript_id,
                    'gene_id': gene_id,
                    'position': pos,
                    'pred_type': 'FN',
                    'score': score,
                    'strand': strand
                })

        if verbose > 1:
            print(f"[debug] transcript={transcript_id} => #TP={len(tp_positions)}, #FP={len(fp_positions)}, #FN={len(fn_positions)}")

        # 8) error_window logic for FPs and FNs
        if isinstance(error_window, int):
            error_window = (error_window, error_window)

        donor_len = len(donor_probabilities)
        for pos_info in (fp_positions + fn_positions):
            pos_i = pos_info['position']
            error_type = pos_info['pred_type']
            strand_    = pos_info['strand']
            wstart = max(0, pos_i - error_window[0])
            wend   = min(donor_len, pos_i + error_window[1])

            error_list.append({
                'transcript_id': transcript_id,
                'gene_id': gene_id,
                'error_type': error_type,
                'position': pos_i,
                'window_start': wstart,
                'window_end': wend,
                'strand': strand_
            })

        # 9) Collect TPs, FPs, FNs in positions_list
        positions_list.extend(tp_positions + fp_positions + fn_positions)

    # end for transcript_id in pred_results

    # 10) Convert to Polars DataFrames
    error_df = pl.DataFrame(error_list)
    positions_df = pl.DataFrame(positions_list)

    if return_positions_df:
        return error_df, positions_df
    return error_df


def evaluate_acceptor_site_errors_by_transcript(
    annotations_df: pl.DataFrame,
    pred_results: dict,
    threshold: float = 0.9,
    consensus_window: int = 2,
    error_window: int = 500,
    predicted_delta_correction: bool = True,
    **kargs
):
    """
    Evaluate transcript-level acceptor site predictions, collecting error positions (TP, FP, FN),
    with an optional +/- consensus_window and an additional error_window for contextual analysis.
    Includes gene_id in both error and positions outputs.

    Parameters
    ----------
    annotations_df : pl.DataFrame
        Must have columns:
          ["transcript_id", "position", "site_type", "strand", ...]
        'site_type' must contain "acceptor".
        'position' is an absolute genomic position of the acceptor site.
    pred_results : dict
        Keyed by transcript_id. Each value is a dict like:
          {
            'gene_id': str,          # recommended
            'strand': str ('+' or '-'),
            'tx_start': int,
            'tx_end':   int,
            'acceptor_prob': List[float],
            ...
          }
        Often produced by transcript-level SpliceAI (efficient_output=True).
    threshold : float
        Probability threshold for classifying a position as an acceptor site.
    consensus_window : int
        +/- around a true acceptor site to consider a predicted site matching it => TP.
    error_window : int or tuple
        If int, interpreted as (error_window, error_window).
        Creates a [window_start, window_end] region around each FP/FN for analysis.
    predicted_delta_correction : bool
        If True, apply adjust_scores() to shift probabilities based on known offsets.
    **kargs
        - verbose (int): Print debug info if > 0.
        - return_positions_df (bool): If True, returns (error_df, positions_df).

    Returns
    -------
    error_df : pl.DataFrame
        Rows for each FP/FN, plus a placeholder row if no acceptor annotations exist,
        with columns: [transcript_id, gene_id, error_type, position, window_start, window_end, strand]
    positions_df : pl.DataFrame
        (Only if return_positions_df=True)
        Includes rows for TP, FP, FN with columns:
          [transcript_id, gene_id, position, pred_type, score, strand].
    """
    verbose = kargs.get('verbose', 1)
    return_positions_df = kargs.get('return_positions_df', False)

    # 1) Group the *true* acceptor sites by transcript_id
    grouped_annotations = (
        annotations_df
        .filter(pl.col('site_type') == 'acceptor')
        .group_by("transcript_id")
        .agg(
            pl.struct(["position"]).alias("acceptor_sites")
        )
        .to_dict(as_series=False)
    )
    transcripts_with_acceptors = set(grouped_annotations["transcript_id"])

    error_list = []
    positions_list = []  # will store TP, FP, FN

    # 2) Iterate over transcript-level predictions
    for transcript_id, tx_data in pred_results.items():
        # Retrieve gene_id from pred_results if present
        gene_id = tx_data.get('gene_id', 'unknown')

        # If no acceptor annotations exist for this transcript, 
        #   record a placeholder and continue
        if transcript_id not in transcripts_with_acceptors:
            if verbose > 0:
                print(f"No acceptor annotations for transcript: {transcript_id}")
            error_list.append({
                'transcript_id': transcript_id,
                'gene_id': gene_id,
                'error_type': None,
                'position': None,
                'window_start': None,
                'window_end': None,
                'strand': None
            })
            continue

        strand   = tx_data.get('strand', '+')
        tx_start = tx_data.get('tx_start', 0)
        tx_end   = tx_data.get('tx_end',   0)

        # 3) Retrieve the annotated acceptor sites for this transcript
        acceptor_struct_list = grouped_annotations['acceptor_sites'][
            grouped_annotations['transcript_id'].index(transcript_id)
        ]

        # Convert absolute positions -> transcript-relative
        true_acceptor_positions = []
        for site_info in acceptor_struct_list:
            abs_pos = site_info['position']
            rel_pos = boundary_strand_offset(abs_pos, tx_start, tx_end, strand)
            true_acceptor_positions.append(rel_pos)
        true_acceptor_positions = np.array(sorted(set(true_acceptor_positions)))

        # 4) Extract acceptor probabilities
        acceptor_probabilities = np.array(tx_data['acceptor_prob'], dtype=float)

        # If predicted_delta_correction => shift probabilities for known offsets
        if predicted_delta_correction:
            acceptor_probabilities = adjust_scores(acceptor_probabilities, strand, splice_type='acceptor')

        # Range check
        if len(true_acceptor_positions) > 0:
            max_pos = true_acceptor_positions.max()
            if max_pos >= len(acceptor_probabilities):
                raise ValueError(
                    f"[Error] Transcript {transcript_id}: acceptor pos out of range. "
                    f"Max pos={max_pos}, len(acceptor_prob)={len(acceptor_probabilities)}"
                )

        # 5) Create a threshold-based label for each position
        label_predictions = np.array([1 if p >= threshold else 0 for p in acceptor_probabilities])

        # We'll track TPs, FPs, FNs in separate lists
        tp_positions = []
        fp_positions = []
        fn_positions = []
        matched_acceptors = [False]*len(true_acceptor_positions)

        # 6) Evaluate each position i
        for i, pred_label in enumerate(label_predictions):
            found_in_window = False

            # Check if i is within +/- consensus_window of any true acceptor
            for idx, true_pos in enumerate(true_acceptor_positions):
                if matched_acceptors[idx]:  # Skip already processed true acceptor sites
                    continue

                wstart = true_pos - consensus_window
                wend   = true_pos + consensus_window

                if wstart <= i <= wend:
                    found_in_window = True
                    if pred_label == 1:
                        # => potential TP
                        # Optionally, compute the max probability in that window
                        ws = max(wstart, 0)
                        we = min(wend+1, len(acceptor_probabilities))
                        window_slice = acceptor_probabilities[ws:we]
                        score_adjusted = window_slice.max() if window_slice.size > 0 else acceptor_probabilities[i]

                        # Confirm the max prob is above threshold
                        assert score_adjusted >= threshold, (
                            f"Unexpected: predicted=1, but max prob in window={score_adjusted} < threshold={threshold}"
                        )

                        tp_positions.append({
                            'transcript_id': transcript_id,
                            'gene_id': gene_id,
                            'position': i,
                            'pred_type': 'TP',
                            'score': score_adjusted,
                            'strand': strand
                        })
                        matched_acceptors[idx] = True
                    break  # done checking other acceptors

            if not found_in_window:
                if pred_label == 1:
                    # => FP
                    fp_positions.append({
                        'transcript_id': transcript_id,
                        'gene_id': gene_id,
                        'position': i,
                        'pred_type': 'FP',
                        'score': acceptor_probabilities[i],
                        'strand': strand
                    })
                # else => TN => we ignore

        # 7) Count FNs: any true acceptor not matched
        for idx, matched in enumerate(matched_acceptors):
            if not matched:
                pos = true_acceptor_positions[idx]
                score = acceptor_probabilities[pos]

                fn_positions.append({
                    'transcript_id': transcript_id,
                    'gene_id': gene_id,
                    'position': pos,
                    'pred_type': 'FN',
                    'score': score,
                    'strand': strand
                })

        if verbose > 1:
            print(f"[debug] transcript={transcript_id} => #TP={len(tp_positions)}, #FP={len(fp_positions)}, #FN={len(fn_positions)}")

        # 8) error_window logic for FPs and FNs
        if isinstance(error_window, int):
            error_window = (error_window, error_window)

        acceptor_len = len(acceptor_probabilities)
        for pos_info in (fp_positions + fn_positions):
            pos_i = pos_info['position']
            error_type = pos_info['pred_type']
            strand_    = pos_info['strand']
            wstart = max(0, pos_i - error_window[0])
            wend   = min(acceptor_len, pos_i + error_window[1])

            error_list.append({
                'transcript_id': transcript_id,
                'gene_id': gene_id,
                'error_type': error_type,
                'position': pos_i,
                'window_start': wstart,
                'window_end': wend,
                'strand': strand_
            })

        # 9) Collect TPs, FPs, FNs in positions_list
        positions_list.extend(tp_positions + fp_positions + fn_positions)

    # end for transcript_id in pred_results

    # 10) Convert to Polars DataFrames
    error_df = pl.DataFrame(error_list)
    positions_df = pl.DataFrame(positions_list)

    if return_positions_df:
        return error_df, positions_df
    return error_df


########################################################################################


def evaluate_donor_site_errors_at_gene_level(annotations_df, pred_results, threshold=0.5, consensus_window=2, error_window=500, **kargs):
    """
    Evaluate SpliceAI predictions for donor splice sites and identify FPs and FNs, including windowed regions.

    Parameters:
    - annotations_df (pl.DataFrame): DataFrame containing true splice site annotations.
      Columns: ['chrom', 'start', 'end', 'strand', 'site_type', 'gene_id', 'transcript_id']
    - pred_results (dict): The output of predict_splice_sites_for_genes(), containing per-nucleotide probabilities.
    - threshold (float): Threshold for classifying a prediction as a donor site (default is 0.9).
    - consensus_window (int): Tolerance window around true splice sites.
    - error_window (int or tuple): Window size (or sizes) for tracking the surrounding region of FPs and FNs.
    - verbose (int): Level of verbosity.

    Returns:
    - error_df (pl.DataFrame): A DataFrame containing positions of FPs and FNs, with window coordinates.


    NOTE: 
    - .index: The .index(gene_id) method is used to find the index of the gene_id in the grouped_annotations['gene_id'] list. 
            This index is then used to access the corresponding donor sites in the grouped_annotations['donor_sites'] list.
    """
    # from .extract_genomic_features import get_overlapping_gene_metadata

    verbose = kargs.get('verbose', 1)
    overlapping_genes_metadata = kargs.get('overlapping_genes_metadata', None)
    adjust_for_overlapping_genes = kargs.get('adjust_for_overlapping_genes', True)
    return_positions_df = kargs.get('return_positions_df', False)
    predicted_delta_correction = kargs.get('predicted_delta_correction', True)

    if overlapping_genes_metadata is None:
        # Check if the GTF file path is provided in kargs
        # gtf_file_path = kargs.get('gtf_file_path', None)
        # if gtf_file_path is None:
        #     raise ValueError("GTF file path must be provided to compute overlapping gene metadata.")

        # filter_valid_splice_sites = kargs.get('filter_valid_splice_sites', True)
        # min_exons = kargs.get('min_exons', 2)

        # overlapping_genes_metadata = \
        #     get_or_load_overlapping_gene_metadata(
        #         gtf_file_path,
        #         overlapping_gene_path=kargs.get('overlapping_gene_path', None),
        #         filter_valid_splice_sites=filter_valid_splice_sites,
        #         min_exons=min_exons,
        #         output_format='dict')

        sa = SpliceAnalyzer()
        overlapping_genes_metadata = sa.retrieve_overlapping_gene_metadata()

    error_list = []
    positions_list = []

    # Group true donor annotations by gene
    grouped_annotations = annotations_df.filter(pl.col('site_type') == 'donor').group_by('gene_id').agg(
        pl.struct(['start', 'end', 'position', ]).alias('donor_sites')
    ).to_dict(as_series=False)

    # Process each gene's predictions in pred_results
    for gene_id, gene_data in pred_results.items():

        if gene_id not in grouped_annotations['gene_id']:
            if verbose: 
                print(f"No donor annotations for gene: {gene_id}")

            # Optional: Append an entry for these genes with no donor annotations
            error_list.append({
                'gene_id': gene_id,
                'error_type': None,  # No error type, since there are no donor annotations
                'position': None,
                'window_start': None,
                'window_end': None, 
                'strand': None
            })

            continue

        # Extract donor sites from grouped_annotations
        donor_sites = grouped_annotations['donor_sites'][grouped_annotations['gene_id'].index(gene_id)]
        strand = normalize_strand(gene_data['strand'])

        # Directly extract unique donor positions using the 'position' column, and 
        # adjust extraction of unique donor positions based on strand
        if strand == '+':
            # For positive strand, calculate relative positions based on gene_start
            true_donor_positions = np.array(
                sorted({site['position'] - gene_data['gene_start'] for site in donor_sites})
            )
        elif strand == '-':
            # For negative strand, calculate relative positions based on gene_end
            true_donor_positions = np.array(
                sorted({gene_data['gene_end'] - site['position'] for site in donor_sites})
            )
        else:
            raise ValueError(f"Invalid strand value: {strand}")

        # Align probabilities with donor positions
        donor_probabilities = np.array(gene_data['donor_prob'])  
        # Probabilities for all positions (relative to gene start or end based on strand)
        if predicted_delta_correction:
            # Adjust scores
            donor_probabilities = adjust_scores(donor_probabilities, strand, 'donor')

        # Validate positions within range
        assert true_donor_positions.max() < len(donor_probabilities), \
            "true_donor_positions contain indices out of range for donor_probabilities"

        # Scores corresponding to the positions
        true_donor_scores = donor_probabilities[true_donor_positions]
        true_donor_scores = np.round(true_donor_scores, 3)  # Round for readability

        # gene_start = gene_data['gene_start']
        # gene_end = gene_data['gene_end']
        gene_len = len(donor_probabilities)

        # assert gene_len == (gene_end - gene_start + 1), f"Gene length mismatch: {gene_len} != {gene_end - gene_start + 1}"
        # NOTE: Can be off by 1

        # Initialize label prediction vector based on the threshold
        label_predictions = np.array([1 if prob >= threshold else 0 for prob in donor_probabilities])

        # if strand == '-':
        #     label_predictions = label_predictions[::-1]

        # Initialize counts for the current gene
        gene_results = {'gene_id': gene_id, 'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}

        # Initialize a dictionary to track if a true donor site is missed (for FN counting)
        true_donor_status = {i: False for i in range(len(true_donor_positions))}

        # Track positions for FPs, FNs, and TPs
        fp_positions, fn_positions, tp_positions = [], [], []
        
        # Loop over each position in the label sequence
        for i, pred_label in enumerate(label_predictions):
            found_in_window = False

            # Check if this position is a true  splice site for the current gene
            for idx, true_pos in enumerate(true_donor_positions):
                if true_donor_status[idx]:  # Skip already processed true donor sites
                    continue

                window_start = true_pos - consensus_window
                window_end = true_pos + consensus_window

                # if i == true_pos:
                #     true_donor_scores[idx] = donor_probabilities[i]

                if window_start <= i <= window_end:
                    found_in_window = True

                    # Count TP or FN based on the prediction
                    if pred_label == 1:
                        gene_results['TP'] += 1
                        true_donor_status[idx] = True
                        
                        # Update the true donor score for this position
                        assert donor_probabilities[i] >= threshold, f"Unexpected score for TP: {donor_probabilities[i]} < {threshold}"
                        
                        tp_positions.append(
                            {'gene_id': gene_id, 'position': i, 'pred_type': 'TP', 'score': donor_probabilities[i], 'strand': strand})
                    break

            # Count non-splice sites (negative examples)
            if not found_in_window:
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
                        fp_positions.append(
                            {'gene_id': gene_id, 'position': i, 'pred_type': 'FP', 'score': donor_probabilities[i], 'strand': strand})
                else:
                    # Predicted negative correctly -> TN
                    gene_results['TN'] += 1

        # Count false negatives
        for idx, status in true_donor_status.items():
            if not status:  # True donor site missed
                # score = true_donor_scores[idx]
                position = true_donor_positions[idx]  # deduplicated position
                score = donor_probabilities[position]
                label = label_predictions[position]

                if verbose:
                    start = max(0, position - 2)
                    end = min(len(donor_probabilities), position + 3)
                    surrounding_scores = donor_probabilities[start:end]
                    print(f"[DEBUG] FN: gene_id={gene_id}, position={position}, score={score}, label={label}  threshold={threshold}")
                    print(f"[DEBUG] Surrounding scores for FN at {position}: {surrounding_scores}")
                
                # Todo
                # assert label == 0, f"Unexpected label for FN: {label} != 0"
                
                assert position < len(donor_probabilities), \
                    f"Position {position} out of bounds for predictions array of length {len(donor_probabilities)}"
                
                # Todo
                # assert score <= threshold, \
                #     f"Unexpected score for FN: {score} > {threshold}, label={label}"
                
                gene_results['FN'] += 1
                fn_positions.append(
                    {'gene_id': gene_id, 'position': position, 'pred_type': 'FN', 'score': score, 'strand': strand})

        # Debugging: Display true donor positions and their surrounding probabilities
        if verbose:
            # Extend donor probabilities with neighbors
            neighbor_range = 2  # Extend by ±2 nucleotides
            true_donor_scores_with_neighbors = []

            for pos in true_donor_positions:
                neighbor_scores = donor_probabilities[max(0, pos - neighbor_range):min(len(donor_probabilities), pos + neighbor_range + 1)]
                true_donor_scores_with_neighbors.append((pos, list(neighbor_scores)))
                
            print(f"[analysis] Gene {gene_id}:")
            print(f"  True donor positions (relative): {true_donor_positions}")
            print(f"  Donor probabilities with neighbors (±{neighbor_range} nts):")
            for pos, scores in true_donor_scores_with_neighbors:
                print(f"    Position {pos}: {scores}")
        
        # ------------------------------------------

        # Calculate windowed regions for FPs and FNs
        if isinstance(error_window, int):
            error_window = (error_window, error_window)  # Symmetric window

        for pos in fp_positions + fn_positions:  # Combine FP and FN positions (list of dicts)
            pos_type = pos['pred_type']
            strand = pos['strand']
            pos_window_start = max(0, pos['position'] - error_window[0])
            pos_window_end = min(gene_len, pos['position'] + error_window[1])
            error_list.append({
                'gene_id': pos['gene_id'],
                'error_type': pos_type, 
                'position': pos['position'],  # Relative position
                'window_start': pos_window_start,
                'window_end': pos_window_end, 
                'strand': strand
            })

        positions_list.extend(fp_positions + fn_positions + tp_positions)

    # Convert error list to DataFrame
    error_df = pl.DataFrame(error_list)
    positions_df = pl.DataFrame(positions_list)

    if return_positions_df:
        return error_df, positions_df

    return error_df


def evaluate_acceptor_site_errors_at_gene_level(annotations_df, pred_results, threshold=0.9, consensus_window=2, error_window=500, **kargs):
    """
    Evaluate SpliceAI predictions for acceptor splice sites and identify FPs and FNs, including windowed regions.

    Parameters:
    - annotations_df (pl.DataFrame): DataFrame containing true splice site annotations.
      Columns: ['chrom', 'start', 'end', 'strand', 'site_type', 'gene_id', 'transcript_id']
    - pred_results (dict): The output of predict_splice_sites_for_genes(), containing per-nucleotide probabilities.
    - threshold (float): Threshold for classifying a prediction as an acceptor site (default is 0.9).
    - consensus_window (int): Tolerance window around true splice sites.
    - error_window (int or tuple): Window size (or sizes) for tracking the surrounding region of FPs and FNs.
    - verbose (int): Level of verbosity.

    Returns:
    - error_df (pl.DataFrame): A DataFrame containing positions of FPs and FNs, with window coordinates.
    """
    # from .extract_genomic_features import get_overlapping_gene_metadata

    verbose = kargs.get('verbose', 1)
    overlapping_genes_metadata = kargs.get('overlapping_genes_metadata', None)
    adjust_for_overlapping_genes = kargs.get('adjust_for_overlapping_genes', True)
    return_positions_df = kargs.get('return_positions_df', False)
    predicted_delta_correction = kargs.get('predicted_delta_correction', True)

    if overlapping_genes_metadata is None:
        # Check if the GTF file path is provided in kargs
        # gtf_file_path = kargs.get('gtf_file_path', None)
        # if gtf_file_path is None:
        #     raise ValueError("GTF file path must be provided to compute overlapping gene metadata.")

        # filter_valid_splice_sites = kargs.get('filter_valid_splice_sites', True)
        # min_exons = kargs.get('min_exons', 2)

        # overlapping_genes_metadata = \
        #     get_or_load_overlapping_gene_metadata(
        #         gtf_file_path,
        #         overlapping_gene_path=kargs.get('overlapping_gene_path', None),
        #         filter_valid_splice_sites=filter_valid_splice_sites,
        #         min_exons=min_exons,
        #         output_format='dict')

        sa = SpliceAnalyzer()
        overlapping_genes_metadata = sa.retrieve_overlapping_gene_metadata()

    error_list = []
    positions_list = []

    # Group true acceptor annotations by gene
    grouped_annotations = annotations_df.filter(pl.col('site_type') == 'acceptor').group_by('gene_id').agg(
        pl.struct(['start', 'end', 'position', 'transcript_id']).alias('acceptor_sites')
    ).to_dict(as_series=False)  # 'transcript_id'

    # Process each gene's predictions in pred_results
    for gene_id, gene_data in pred_results.items():
        if gene_id not in grouped_annotations['gene_id']:
            if verbose: 
                print(f"No acceptor annotations for gene: {gene_id}")

            # Optional: Append an entry for these genes with no acceptor annotations
            error_list.append({
                'gene_id': gene_id,
                'error_type': None,  # No error type, since there are no acceptor annotations
                'position': None,
                'window_start': None,
                'window_end': None, 
                'strand': None
            })
            continue

        # Extract donor sites from grouped_annotations
        acceptor_sites = grouped_annotations['acceptor_sites'][grouped_annotations['gene_id'].index(gene_id)]
        strand = normalize_strand(gene_data['strand'])

        # Directly extract unique donor positions using the 'position' column, and
        # adjust extraction of unique donor positions based on strand
        # Calculate relative positions and map them to transcript_id
        true_acceptor_positions = []
        position_to_transcript = {}  # Dictionary to map relative positions to transcript IDs

        for site in acceptor_sites:
            if strand == '+':
                relative_position = site['position'] - gene_data['gene_start']
            elif strand == '-':
                relative_position = gene_data['gene_end'] - site['position']
            else:
                raise ValueError(f"Invalid strand value: {strand}")

            # Append the relative position
            true_acceptor_positions.append(relative_position)

            # Map position to transcript_id, accounting for shared positions
            if relative_position not in position_to_transcript:
                position_to_transcript[relative_position] = set()  # Use a set to avoid duplicates
            position_to_transcript[relative_position].add(site['transcript_id'])

        # Sort positions and convert transcript ID sets to lists
        true_acceptor_positions = sorted(set(true_acceptor_positions))
        position_to_transcript = {pos: list(transcripts) for pos, transcripts in position_to_transcript.items()}

        # Align probabilities with donor positions
        acceptor_probabilities = np.array(gene_data['acceptor_prob'])  # Probabilities for all positions
        if predicted_delta_correction:
            # Adjust scores
            acceptor_probabilities = adjust_scores(acceptor_probabilities, strand, 'acceptor')

        # Validate positions within range
        assert true_acceptor_positions.max() < len(acceptor_probabilities), \
            "true_acceptor_positions contain indices out of range for acceptor_probabilities"

        # Scores corresponding to the positions
        true_acceptor_scores = acceptor_probabilities[true_acceptor_positions]
        true_acceptor_scores = np.round(true_acceptor_scores, 3)  # Round for readability

        # gene_start = gene_data['gene_start']
        # gene_end = gene_data['gene_end']
        gene_len = len(acceptor_probabilities)

        # Initialize label prediction vector based on the threshold
        label_predictions = np.array([1 if prob >= threshold else 0 for prob in acceptor_probabilities])

        # if strand == '-':
        #     label_predictions = label_predictions[::-1]

        # Initialize counts for the current gene
        gene_results = {'gene_id': gene_id, 'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}
        # Initialize a dictionary to track if a true acceptor site is missed (for FN counting)
        true_acceptor_status = {i: False for i in range(len(true_acceptor_positions))}
        fp_positions, fn_positions, tp_positions = [], [], []

        # Loop over each position in the label sequence
        for i, pred_label in enumerate(label_predictions):
            found_in_window = False

            # Check if this position is a true  splice site for the current gene
            for idx, true_pos in enumerate(true_acceptor_positions):
                if true_acceptor_status[idx]:  # Skip already processed true acceptor sites
                    continue

                window_start = true_pos - consensus_window
                window_end = true_pos + consensus_window

                # if i == true_pos:
                #     # Update the true acceptor score
                #     true_acceptor_scores[idx] = acceptor_probabilities[i]

                if window_start <= i <= window_end:
                    found_in_window = True 
                    
                    # Count TP or FN based on the prediction
                    if pred_label == 1:
                        gene_results['TP'] += 1
                        true_acceptor_status[idx] = True

                        assert acceptor_probabilities[i] >= threshold, f"Unexpected score for TP: {acceptor_probabilities[i]} < {threshold}"
                        tp_positions.append(
                            {'gene_id': gene_id, 'position': i, 
                             'pred_type': 'TP', 'score': acceptor_probabilities[i], 'strand': strand})
                    break

            # Count non-splice sites (negative examples)
            if not found_in_window:
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
                        fp_positions.append(
                            {'gene_id': gene_id, 'position': i, 
                            'pred_type': 'FP', 'score': acceptor_probabilities[i], 'strand': strand})
                else:
                    # Predicted negative correctly -> TN
                    gene_results['TN'] += 1

        # Count false negatives
        for idx, status in true_acceptor_status.items():
            if not status:
                # score = true_acceptor_scores[idx]
                position = true_acceptor_positions[idx]  # true relative position
                score = acceptor_probabilities[position]
                label = label_predictions[position]

                assert 0 <= position < len(label_predictions), \
                    f"Position {position} out of bounds for label_predictions array of length {len(label_predictions)}"

                if verbose:
                    start = max(0, position - 2)
                    end = min(len(acceptor_probabilities), position + 3)
                    surrounding_scores = acceptor_probabilities[start:end]
                    print(f"[DEBUG] FN: gene_id={gene_id}, position={position}, score={score}, label={label}  threshold={threshold}")
                    print(f"[DEBUG] Surrounding scores for FN at {position}: {surrounding_scores}")

                # assert label == 0, f"Unexpected label for FN: {label} != 0"  # Todo
                # assert score <= threshold, f"Unexpected score for FN: {score} > {threshold}, label={label}"  # Todo
                
                gene_results['FN'] += 1
                fn_positions.append(
                    {'gene_id': gene_id, 'position': position, 'pred_type': 'FN', 'score': score, 'strand': strand})

        # ------------------------------------------
        # Debugging: Display true acceptor positions and their surrounding probabilities
        if verbose:
            # Extend acceptor probabilities with neighbors
            neighbor_range = 2  # Extend by ±2 nucleotides
            true_acceptor_scores_with_neighbors = []

            for pos in true_acceptor_positions:
                neighbor_scores = acceptor_probabilities[max(0, pos - neighbor_range):min(len(acceptor_probabilities), pos + neighbor_range + 1)]
                true_acceptor_scores_with_neighbors.append((pos, list(neighbor_scores)))

            print(f"[analysis] Gene {gene_id}:")
            print(f"  True acceptor positions (relative): {true_acceptor_positions}")
            print(f"  Acceptor probabilities with neighbors (±{neighbor_range} nts):")
            for pos, scores in true_acceptor_scores_with_neighbors:
                print(f"    Position {pos}: {scores}")

        # ------------------------------------------

        # Step 4: Calculate windowed regions for FPs and FNs
        if isinstance(error_window, int):
            error_window = (error_window, error_window)  # Symmetric window

        for pos in fp_positions + fn_positions:
            pos_type = pos['pred_type']
            strand = pos['strand']
            pos_window_start = max(0, pos['position'] - error_window[0])
            pos_window_end = min(gene_len, pos['position'] + error_window[1])
            error_list.append({
                'gene_id': pos['gene_id'],
                'error_type': pos_type,
                'position': pos['position'],
                'window_start': pos_window_start,
                'window_end': pos_window_end, 
                'strand': strand
            })

        positions_list.extend(fp_positions + fn_positions + tp_positions)

    # Convert error list to DataFrame
    error_df = pl.DataFrame(error_list)
    positions_df = pl.DataFrame(positions_list)

    if return_positions_df:
        return error_df, positions_df
    
    return error_df


def save_performance_df(
        performance_df, 
        output_dir, 
        chr=None, chunk_start=None, chunk_end=None, 
        separator='\t', aggregated=False, subject='splice_performance'):
    """
    Save the performance DataFrame to a file with the specified separator.

    Parameters:
    - performance_df (pl.DataFrame): The DataFrame to save.
    - output_dir (str): The directory to save the file in.
    - chr (str): The chromosome identifier (optional for aggregated DataFrame).
    - chunk_start (int): The start index of the chunk (optional for aggregated DataFrame).
    - chunk_end (int): The end index of the chunk (optional for aggregated DataFrame).
    - separator (str): The separator to use in the file (default is '\t').
    - aggregated (bool): If True, save as an aggregated file without chunk_start and chunk_end.
    """
    # Determine the file extension based on the separator
    if separator == '\t':
        file_extension = 'tsv'
    elif separator == ',':
        file_extension = 'csv'
    else:
        raise ValueError("Unsupported separator. Use '\\t' for TSV or ',' for CSV.")

    # Construct the file name and path
    if aggregated:
        performance_file = f"full_{subject}.{file_extension}"
    else:
        if chr is None or chunk_start is None or chunk_end is None:
            raise ValueError("chr, chunk_start, and chunk_end must be provided for non-aggregated DataFrame")
        performance_file = f"{subject}_{chr}_chunk_{chunk_start}_{chunk_end}.{file_extension}"
    
    performance_path = os.path.join(output_dir, performance_file)

    # Ensure the directory exists
    os.makedirs(os.path.dirname(performance_path), exist_ok=True)

    # Save the DataFrame to the file
    performance_df.write_csv(performance_path, separator=separator)

    print(f"[i/o] Performance saved to: {performance_file}")

    return performance_path


def save_error_analysis(
        error_df, output_dir, chr=None, chunk_start=None, chunk_end=None, 
        separator='\t', aggregated=False, subject='splice_errors'):
    """
    Save the error analysis DataFrame to a file with the specified separator.

    Parameters:
    - error_df (pl.DataFrame): The DataFrame to save.
    - output_dir (str): The directory to save the file in.
    - chr (str): The chromosome identifier (optional for aggregated DataFrame).
    - chunk_start (int): The start index of the chunk (optional for aggregated DataFrame).
    - chunk_end (int): The end index of the chunk (optional for aggregated DataFrame).
    - separator (str): The separator to use in the file (default is '\t').
    - aggregated (bool): If True, save as an aggregated file without chunk_start and chunk_end.

    Example usage: 
        save_error_analysis(error_df_chunk, output_dir, 'chr1', 0, 1000, separator='\t')
        save_error_analysis(full_error_df, output_dir, separator='\t', aggregated=True)
    """
    # Determine the file extension based on the separator
    if separator == '\t':
        file_extension = 'tsv'
    elif separator == ',':
        file_extension = 'csv'
    else:
        raise ValueError("Unsupported separator. Use '\\t' for TSV or ',' for CSV.")

    # Construct the file name and path
    if aggregated:
        error_analysis_file = f"full_{subject}.{file_extension}"
    else:
        if chr is None or chunk_start is None or chunk_end is None:
            raise ValueError("chr, chunk_start, and chunk_end must be provided for non-aggregated DataFrame")
        error_analysis_file = f"{subject}_{chr}_chunk_{chunk_start}_{chunk_end}.{file_extension}"
    
    error_analysis_path = os.path.join(output_dir, error_analysis_file)

    # Ensure the directory exists
    os.makedirs(os.path.dirname(error_analysis_path), exist_ok=True)

    # Save the DataFrame to the file
    error_df.write_csv(error_analysis_path, separator=separator)

    print(f"[i/o] Saved analysis data (subject={subject}) to: {error_analysis_file}")

    return error_analysis_path


def match_predicted_to_true_splice_sites(
        gene_id, gene_start, gene_end, strand, 
        label_predictions, true_donor_sites, 
        threshold_distance=10, 
        verbose=True):
    """
    Match positive splice site predictions to true donor sites and compute the distance between them.

    Parameters:
    - gene_id (str): The gene identifier.
    - gene_start (int): The starting genomic position of the gene.
    - gene_end (int): The ending genomic position of the gene.
    - strand (str): The strand of the gene ('+' or '-').
    - label_predictions (np.array): An array of binary predictions (1 for positive, 0 for negative).
    - true_donor_sites (list of dicts): A list of dictionaries containing 'start' and 'end' for true donor sites.
    - threshold_distance (int): Maximum allowable distance between predicted and true donor sites (default=10).
    - verbose (bool): If True, prints detailed summary of matches and mismatches.

    Returns:
    - match_results (dict): A dictionary containing:
        - 'matched': List of tuples (predicted_pos, true_pos, distance) for matching predictions.
        - 'unmatched_predictions': List of predicted positions that couldn't be matched to a true site within the threshold.
        - 'unmatched_true_sites': List of true donor site positions that were not predicted.
        - 'summary': A summary report of the matching results.
    """
    # Initialize the result structure
    match_results = {
        'matched': [],
        'unmatched_predictions': [],
        'unmatched_true_sites': [],
        'summary': ''
    }
    
    # Convert predicted relative positions to absolute positions
    if strand == '+':
        positive_prediction_positions = np.where(label_predictions == 1)[0] + gene_start
    else:  # For negative strand
        positive_prediction_positions = gene_end - np.where(label_predictions == 1)[0]

    # Extract the absolute positions of true donor sites
    true_donor_positions = np.array([site['start'] for site in true_donor_sites])

    if len(positive_prediction_positions) == 0:
        match_results['summary'] = f"No positive predictions for gene {gene_id}."
        return match_results

    if len(true_donor_positions) == 0:
        match_results['summary'] = f"No true donor sites for gene {gene_id}."
        return match_results

    # Match predicted positions to the nearest true donor site within the threshold distance
    for pred_pos in positive_prediction_positions:
        # Calculate the distances between this predicted position and all true donor sites
        distances = np.abs(true_donor_positions - pred_pos)
        closest_idx = np.argmin(distances)
        min_distance = distances[closest_idx]

        if min_distance <= threshold_distance:
            # Match found within the threshold distance
            true_pos = true_donor_positions[closest_idx]
            match_results['matched'].append((pred_pos, true_pos, min_distance))
        else:
            # No match within the threshold
            match_results['unmatched_predictions'].append(pred_pos)

    # Identify true donor sites that were not matched by any predictions
    for true_pos in true_donor_positions:
        if not any(true_pos == match[1] for match in match_results['matched']):
            match_results['unmatched_true_sites'].append(true_pos)

    # Generate summary report
    matched_count = len(match_results['matched'])
    unmatched_predictions_count = len(match_results['unmatched_predictions'])
    unmatched_true_sites_count = len(match_results['unmatched_true_sites'])

    summary = (
        f"Summary Report for Gene {gene_id}:\n"
        f"Total Predicted Splice Sites: {len(positive_prediction_positions)}\n"
        f"Matched Predictions: {matched_count}\n"
        f"Unmatched Predictions: {unmatched_predictions_count}\n"
        f"Unmatched True Donor Sites: {unmatched_true_sites_count}\n"
    )

    if unmatched_predictions_count > 0:
        summary += "Unmatched Predictions Details:\n"
        for pred_pos in match_results['unmatched_predictions']:
            summary += f"  Predicted Position: {pred_pos}\n"

    if unmatched_true_sites_count > 0:
        summary += "Unmatched True Donor Sites Details:\n"
        for true_pos in match_results['unmatched_true_sites']:
            summary += f"  True Donor Position: {true_pos}\n"

    match_results['summary'] = summary

    # Print summary if verbose mode is enabled
    if verbose:
        print(summary)

    return match_results


def compute_global_summary_from_chromosome_summary(file_path):
    """
    Compute the global performance summary from a chromosome-level performance summary file.

    Parameters:
    - file_path (str): Path to the chromosome performance summary file (chromosome_performance_summary.tsv).

    Returns:
    - global_summary (dict): A dictionary containing the global performance metrics.
    """
    # Step 1: Read the chromosome performance summary into a DataFrame
    chromosome_summary_df = pl.read_csv(file_path)

    # Step 2: Aggregate the counts across all chromosomes
    total_TP = chromosome_summary_df['TP'].sum()
    total_TN = chromosome_summary_df['TN'].sum()
    total_FP = chromosome_summary_df['FP'].sum()
    total_FN = chromosome_summary_df['FN'].sum()

    # Step 3: Compute global performance metrics
    global_precision = total_TP / (total_TP + total_FP) if (total_TP + total_FP) > 0 else 0
    global_recall = total_TP / (total_TP + total_FN) if (total_TP + total_FN) > 0 else 0
    global_specificity = total_TN / (total_TN + total_FP) if (total_TN + total_FP) > 0 else 0
    global_f1_score = 2 * (global_precision * global_recall) / (global_precision + global_recall) if (global_precision + global_recall) > 0 else 0

    # Step 4: Create the global summary dictionary
    global_summary = {
        'TP': total_TP,
        'TN': total_TN,
        'FP': total_FP,
        'FN': total_FN,
        'precision': global_precision,
        'recall': global_recall,
        'specificity': global_specificity,
        'f1_score': global_f1_score
    }

    # Step 5: Print the global summary for verification
    print("Global Summary:")
    for metric, value in global_summary.items():
        print(f"{metric}: {value:.4f}" if isinstance(value, float) else f"{metric}: {value}")

    return global_summary


####################################################################################################
# Utility functions for plotting ROC curves, PRC curves, and calculating AUC, etc. 

def filter_sequence_df(seq_df_path, sampled_gene_ids, format='parquet'):
    # Convert sampled_gene_ids to a set for faster membership checking
    if isinstance(sampled_gene_ids, pl.DataFrame):
        sampled_gene_ids_set = set(sampled_gene_ids['gene_id'].to_list())
    elif isinstance(sampled_gene_ids, (set, list)):
        sampled_gene_ids_set = set(sampled_gene_ids)
    else:
        raise ValueError("sampled_gene_ids must be a DataFrame, set, or list")

    # Define a lazy frame to read the sequence data
    if format == 'csv':
        lazy_seq_df = pl.scan_csv(seq_df_path)
    elif format == 'parquet':
        lazy_seq_df = pl.scan_parquet(seq_df_path)
    else:
        raise ValueError("Unsupported format. Use 'csv' or 'parquet'.")

    # Filter the lazy frame to keep only rows with gene_id in sampled_gene_ids_set
    filtered_lazy_seq_df = lazy_seq_df.filter(pl.col('gene_id').is_in(sampled_gene_ids_set))

    # Collect the filtered DataFrame
    filtered_seq_df = filtered_lazy_seq_df.collect()

    return filtered_seq_df


def filter_annotations_df(annot_df_path, sampled_gene_ids, **kargs):
    from meta_spliceai.splice_engine.utils_fs import read_splice_sites

    if isinstance(sampled_gene_ids, pl.DataFrame):
        sampled_gene_ids_set = set(sampled_gene_ids['gene_id'].to_list())
    elif isinstance(sampled_gene_ids, (set, list)):
        sampled_gene_ids_set = set(sampled_gene_ids)
    else:
        raise ValueError("sampled_gene_ids must be a DataFrame, set, or list")

    annot_df = read_splice_sites(annot_df_path, separator='\t', dtypes=None, **kargs)
    # A Polars DataFrame: ['chrom', 'start', 'end', 'strand', 'site_type', 'gene_id', 'transcript_id']

    # Subset annotation data based on the sampled gene IDs
    return annot_df.filter(pl.col('gene_id').is_in(sampled_gene_ids_set))


# TargetGeneIdsType = Union[List[str], Set[str], pl.DataFrame]

def subset_sequence_data(
    seq_df_path: str,
    target_gene_ids,  # TargetGeneIdsType
    chromosomes: List[str],
    format: str = 'parquet',
    chunk_size: int = 500,
    verbose: int = 1
) -> pl.DataFrame:
    """
    Subset sequence data from multiple chromosomes using streaming mode based on target_gene_ids.

    Parameters:
    - seq_df_path (str): Path to the sequence data file.
    - target_gene_ids (Union[List[str], Set[str], pl.DataFrame]): List, set, or DataFrame of target gene IDs to subset.
    - chromosomes (list): List of chromosomes to sample from.
    - format (str): Format of the sequence data file (default is 'parquet').
    - chunk_size (int): Size of chunks to load from each chromosome in streaming mode.

    Returns:
    - subset_df (pl.DataFrame): Subset of the input sequence data, containing the same columns.

    Example usage:
        seq_df_path is the path to the sequence data file (e.g., "path/to/seq_data.parquet")
        target_gene_ids is a list, set, or DataFrame of target gene IDs (e.g., ['gene1', 'gene2', 'gene3'])
        chromosomes is a list of chromosomes to sample from (e.g., ['1', '2', 'X'])
        subset_df = subset_sequence_data(seq_df_path, target_gene_ids=target_gene_ids, chromosomes=['1', '2', 'X'])
    """
    from meta_spliceai.splice_engine.utils_bio import load_chromosome_sequence_streaming

    # Convert target_gene_ids to a set for faster membership checking
    if isinstance(target_gene_ids, pl.DataFrame):
        target_gene_ids_set = set(target_gene_ids['gene_id'].to_list())
    elif isinstance(target_gene_ids, (set, list)):
        target_gene_ids_set = set(target_gene_ids)
    else:
        raise ValueError("target_gene_ids must be a DataFrame, set, or list")
    
    # Initialize an empty list to store the filtered rows
    subset_rows = []

    # Iterate over chromosomes
    for chr in tqdm(chromosomes, desc="Processing chromosomes"):
        # Load sequence data using streaming mode
        lazy_seq_df = load_chromosome_sequence_streaming(seq_df_path, chr, format=format)
        
        # Get the total number of genes in the current chromosome
        num_genes_in_chr = lazy_seq_df.select(pl.col('gene_id').n_unique()).collect().item()

        for chunk_start in range(0, num_genes_in_chr, chunk_size):
            chunk_end = min(chunk_start + chunk_size, num_genes_in_chr)

            # Filter the LazyFrame to process only the current chunk
            seq_chunk = lazy_seq_df.slice(chunk_start, chunk_size).collect()

            # Filter the chunk to include only rows with gene_id in target_gene_ids_set
            filtered_seq_chunk = seq_chunk.filter(pl.col('gene_id').is_in(target_gene_ids_set))

            # Append filtered rows to the subset
            subset_rows.extend(filtered_seq_chunk.iter_rows(named=True))

    # Convert subset rows back to a Polars DataFrame
    subset_df = pl.DataFrame(subset_rows)

    if verbose: 
        # Verify the number of unique genes in the resulting DataFrame
        unique_genes_count = subset_df.select(pl.col('gene_id').n_unique()).item()
        print(f"[info] Number of unique genes in the resulting DataFrame: {unique_genes_count}")

        # Check if all target genes are present in the subset
        subset_gene_ids_set = set(subset_df['gene_id'].to_list())
        missing_genes = target_gene_ids_set - subset_gene_ids_set
        if missing_genes:
            print(f"[warning] The following {len(missing_genes)} target genes are missing in the subset:\n{missing_genes}\n")
        else:
            print(f"[info] All n={unique_genes_count} target genes are present in the subset.")

    return subset_df


def create_sequence_subset_streaming(
    seq_df_path: str,
    num_genes: int,
    chromosomes: list,
    format: str = 'parquet',
    chunk_size: int = 500,
    seed: int = 42,
    verbose: int = 1
) -> pl.DataFrame:
    """
    Create a subset of sequence data from multiple chromosomes using streaming mode.

    Parameters:
    - seq_df_path (str): Path to the sequence data file.
    - num_genes (int): Total number of genes to sample from the dataset.
    - chromosomes (list): List of chromosomes to sample from.
    - format (str): Format of the sequence data file (default is 'parquet').
    - chunk_size (int): Size of chunks to load from each chromosome in streaming mode.
    - seed (int): Random seed for reproducibility (default is 42).

    Returns:
    - subset_df (pl.DataFrame): Subset of the input sequence data, containing the same columns.

    Example usage:
        seq_df_path is the path to the sequence data file (e.g., "path/to/seq_data.parquet")
        num_genes is the total number of genes to sample (e.g., 1000)
        chromosomes is a list of chromosomes to sample from (e.g., ['1', '2', 'X'])
        subset_df = create_sequence_subset_streaming(seq_df_path, num_genes=1000, chromosomes=['1', '2', 'X'])
    """
    from meta_spliceai.splice_engine.utils_bio import load_chromosome_sequence_streaming

    # Set the random seed for reproducibility
    random.seed(seed)
    # NOTE: Using the current time as a seed can be a good workaround if you want to ensure that different 
    #       runs of your experiments use different subsets or samples.
    #       seed = int(time.time())
    
    # Initialize an empty list to store the sampled rows
    subset_rows = []

    # Initialize a counter to keep track of the total number of sampled genes
    total_sampled_genes = 0

    # Iterate over chromosomes
    for chr in tqdm(chromosomes, desc="Processing chromosomes"):
        # Load sequence data using streaming mode
        lazy_seq_df = load_chromosome_sequence_streaming(seq_df_path, chr, format=format)
        
        # Get the total number of genes in the current chromosome
        num_genes_in_chr = lazy_seq_df.select(pl.col('gene_id').n_unique()).collect().item()
        # NOTE: .collect(): 
        #   - This method collects the results of the lazy computation into a DataFrame. 
        #     Polars uses lazy evaluation, so operations are not executed until .collect() is called.

        for chunk_start in range(0, num_genes_in_chr, chunk_size):
            chunk_end = min(chunk_start + chunk_size, num_genes_in_chr)

            # Filter the LazyFrame to process only the current chunk
            seq_chunk = lazy_seq_df.slice(chunk_start, chunk_size).collect()

            # Randomly sample rows from the current chunk
            num_genes_to_sample = min(chunk_size, num_genes - total_sampled_genes, len(seq_chunk))
            sampled_rows = random.sample(list(seq_chunk.iter_rows(named=True)), num_genes_to_sample)

            # Append sampled rows to the subset
            subset_rows.extend(sampled_rows)

            # Update the count of total sampled genes
            total_sampled_genes += len(sampled_rows)

            # Break the loop if we have already collected the required number of genes
            if total_sampled_genes >= num_genes:
                break

        # Break the loop if we have already collected the required number of genes
        if total_sampled_genes >= num_genes:
            break

    # Convert subset rows back to a Polars DataFrame
    subset_df = pl.DataFrame(subset_rows)

    if verbose: 
        # Verify the number of unique genes in the resulting DataFrame
        # unique_genes_count = subset_df.select(pl.col('gene_id').n_unique()).collect().item()
        unique_genes_count_df = subset_df.select(pl.col('gene_id').n_unique())
        unique_genes_count_series = unique_genes_count_df.to_series()
        unique_genes_count = unique_genes_count_series.item()

        print(f"[info] Requested number of unique genes: {num_genes}")
        print(f"... Number of unique genes in the resulting DataFrame: {unique_genes_count}")

        # Check if the count matches the requested number
        if unique_genes_count == num_genes:
            print("The resulting DataFrame has the correct number of unique genes.")
        else:
            print("The resulting DataFrame does NOT have the correct number of unique genes.")

    return subset_df


def aggregate_performance_files(
    local_dir,
    eval_dir,
    mode='gene', seq_type='full', format='parquet', save_to_file=False, 
    test_mode=False
):
    """
    Process chromosome sequences and aggregate performance results.

    Parameters:
    - mode (str): Mode of operation ('gene' or other).
    - seq_type (str): Sequence type ('minmax' or 'full').
    - format (str): File format ('tsv', 'csv', 'parquet').
    - local_dir (str): Directory for input files.
    - eval_dir (str): Directory for output files.
    - save_to_file (bool): Whether to save the results to a file.
    - test_mode (bool): Whether to run in test mode. Default is False.

    Returns:
    - global_summary (dict): Global performance summary.

    Example usage: 
        global_summary = aggregate_performance_files(
            mode='gene', 
            seq_type='full', 
            format='parquet', 
            local_dir='/path/to/local_dir', 
            eval_dir='/path/to/eval_dir', 
            save_to_file=True, 
            gtf_file='/path/to/gtf_file', 
            genome_fasta='/path/to/genome_fasta', 
            test_mode=False
        )
    """
    from meta_spliceai.splice_engine.utils_bio import load_chromosome_sequence_streaming

    if mode in ['gene', ]:
        output_file = f"gene_sequence.{format}" 
        if seq_type == 'minmax':
            output_file = f"gene_sequence_minmax.{format}"

        seq_df_path = os.path.join(local_dir, output_file)
        # gene_equence_retrieval_workflow(
        #     gtf_file, genome_fasta, gene_tx_map=None, output_file=seq_df_path, mode=seq_type)
    else: 
        seq_df_path = os.path.join(local_dir, f"tx_sequence.{format}")
        # transcript_sequence_retrieval_workflow(
        #     gtf_file, genome_fasta, gene_tx_map=None, output_file=seq_df_path)

    chromosomes = [str(i) for i in range(1, 23)] + ['X', 'Y', 'MT']

    # Initialize an empty list to store the paths of saved performance files
    performance_files = []

    for chr in tqdm(chromosomes, desc="Processing chromosomes"):

        # Load sequence data using streaming mode
        lazy_seq_df = load_chromosome_sequence_streaming(seq_df_path, chr, format=format, return_none_if_missing=True)

        if lazy_seq_df is None:
            print(f"[warning] No sequence data found for chromosome {chr}. Skipping ...")
            continue

        # Initialize chunk size
        chunk_size = 500 if not test_mode else 50  # Starting chunk size
        seq_len_avg = 50000  # Assume an average sequence length for now
        num_genes = lazy_seq_df.select(pl.col('gene_id').n_unique()).collect().item()
        n_chunk_processed = 0

        for chunk_start in range(0, num_genes, chunk_size):
            chunk_end = min(chunk_start + chunk_size, num_genes)

            splice_performance_file = f"splice_performance_{chr}_chunk_{chunk_start}_{chunk_end}.tsv"
            splice_performance_path = os.path.join(eval_dir, splice_performance_file)

            if not os.path.exists(splice_performance_path):
                print(f"... Performance file does not exist yet: {splice_performance_file}. Skipping ...")
                continue

            # Add path to the list of saved performance files
            performance_files.append(splice_performance_path)

    ########################################################

    # Group performance files by chromosome
    chromosome_performance_files = defaultdict(list)

    for performance_file in performance_files:
        # Extract chromosome information from the file name (assuming the naming pattern is consistent)
        chromosome = os.path.basename(performance_file).split('_')[2]  # Extracts the chromosome identifier
        chromosome_performance_files[chromosome].append(performance_file)

    # Debug: Print the number of files found for each chromosome
    for chromosome, files in chromosome_performance_files.items():
        print(f"[test] Chromosome {chromosome}: {len(files)} files")

    print_emphasized("Aggregate performance files by chromosome ...") 
    
    # Initialize structures to store per-chromosome and global counts
    global_counts = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}
    chromosome_summaries = []

    # Aggregate performance results per chromosome
    for chromosome, files in tqdm(chromosome_performance_files.items(), desc="Aggregating per-chromosome performance"):
        # Initialize counts for the current chromosome
        chrom_counts = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}
        
        # Aggregate the counts from each chunk for this chromosome
        for file in files:
            performance_df_chunk = pl.read_csv(file)
            
            # Update chromosome-level counts
            chrom_counts['TP'] += performance_df_chunk['TP'].sum()
            chrom_counts['TN'] += performance_df_chunk['TN'].sum()
            chrom_counts['FP'] += performance_df_chunk['FP'].sum()
            chrom_counts['FN'] += performance_df_chunk['FN'].sum()
        
        # Calculate metrics for the chromosome
        chrom_precision = chrom_counts['TP'] / (chrom_counts['TP'] + chrom_counts['FP']) if (chrom_counts['TP'] + chrom_counts['FP']) > 0 else 0
        chrom_recall = chrom_counts['TP'] / (chrom_counts['TP'] + chrom_counts['FN']) if (chrom_counts['TP'] + chrom_counts['FN']) > 0 else 0
        chrom_specificity = chrom_counts['TN'] / (chrom_counts['TN'] + chrom_counts['FP']) if (chrom_counts['TN'] + chrom_counts['FP']) > 0 else 0
        chrom_f1_score = 2 * (chrom_precision * chrom_recall) / (chrom_precision + chrom_recall) if (chrom_precision + chrom_recall) > 0 else 0
        chrom_fpr = chrom_counts['FP'] / (chrom_counts['FP'] + chrom_counts['TN']) if (chrom_counts['FP'] + chrom_counts['TN']) > 0 else 0
        chrom_fnr = chrom_counts['FN'] / (chrom_counts['FN'] + chrom_counts['TP']) if (chrom_counts['FN'] + chrom_counts['TP']) > 0 else 0
          
        # Add to the summary for this chromosome
        chromosome_summary = {
            'chromosome': chromosome,
            'TP': chrom_counts['TP'],
            'TN': chrom_counts['TN'],
            'FP': chrom_counts['FP'],
            'FN': chrom_counts['FN'],
            'precision': chrom_precision,
            'recall': chrom_recall,
            'specificity': chrom_specificity,
            'f1_score': chrom_f1_score, 
            'fpr': chrom_fpr,
            'fnr': chrom_fnr
        }
        chromosome_summaries.append(chromosome_summary)

        # Update global counts with chromosome counts
        global_counts['TP'] += chrom_counts['TP']
        global_counts['TN'] += chrom_counts['TN']
        global_counts['FP'] += chrom_counts['FP']
        global_counts['FN'] += chrom_counts['FN']

    print_emphasized("Aggregate global performance results ...")

    # Calculate global metrics after aggregating all chromosome-level counts
    global_precision = global_counts['TP'] / (global_counts['TP'] + global_counts['FP']) if (global_counts['TP'] + global_counts['FP']) > 0 else 0
    global_recall = global_counts['TP'] / (global_counts['TP'] + global_counts['FN']) if (global_counts['TP'] + global_counts['FN']) > 0 else 0
    global_specificity = global_counts['TN'] / (global_counts['TN'] + global_counts['FP']) if (global_counts['TN'] + global_counts['FP']) > 0 else 0
    global_f1_score = 2 * (global_precision * global_recall) / (global_precision + global_recall) if (global_precision + global_recall) > 0 else 0
    global_fpr = global_counts['FP'] / (global_counts['FP'] + global_counts['TN']) if (global_counts['FP'] + global_counts['TN']) > 0 else 0
    global_fnr = global_counts['FN'] / (global_counts['FN'] + global_counts['TP']) if (global_counts['FN'] + global_counts['TP']) > 0 else 0

    # Create a global summary dictionary
    global_summary = {
        'TP': global_counts['TP'],
        'TN': global_counts['TN'],
        'FP': global_counts['FP'],
        'FN': global_counts['FN'],
        'precision': global_precision,
        'recall': global_recall,
        'specificity': global_specificity,
        'f1_score': global_f1_score, 
        'fpr': global_fpr,
        'fnr': global_fnr
    }

    # Print the global summary
    global_summary_df = pl.DataFrame([global_summary])
    pl.Config.set_tbl_cols(len(global_summary_df.columns))  # Ensure all columns are shown
    print("\nGlobal Performance Summary:")
    print(global_summary_df)

    # Save per-chromosome summaries to a DataFrame and CSV
    chromosome_summaries_df = pl.DataFrame(chromosome_summaries)
    chromosome_summaries_path = os.path.join(eval_dir, "chromosome_performance_summary.tsv")

    if save_to_file:
        chromosome_summaries_df.write_csv(chromosome_summaries_path)
        print(f"[i/o] Chromosome-level performance saved to: {chromosome_summaries_path}")
    else: 
        # print chromosomal summaries
        pl.Config.set_tbl_cols(len(chromosome_summaries_df.columns))  # Ensure all columns are shown
        print("\nChromosome-wise Performance Summary:")
        print(chromosome_summaries_df)

    return global_summary


def plot_roc_curve_for_donor_site_predictions(
        annotations_df, 
        pred_results, 
        consensus_window=2, 
        thresholds=None, 
        save_path=None, 
        file_format=None, 
        **kargs):
    """
    Plot the ROC curve for SpliceAI predictions for donor splice sites against true annotations.

    Parameters:
    - annotations_df (pl.DataFrame): DataFrame containing true splice site annotations.
      Columns: ['chrom', 'start', 'end', 'strand', 'site_type', 'gene_id', 'transcript_id']
    - pred_results (dict): The output of predict_splice_sites_for_genes(), containing per-nucleotide probabilities.
    - consensus_window (int): Tolerance window around true splice sites.
    - thresholds (list, optional): List of thresholds to evaluate. Default is np.linspace(0, 1, 50).
    - save_path (str, optional): Path to save the plot. If None, the plot will be shown on the screen.
    - file_format (str, optional): File format to save the plot (e.g., 'png', 'pdf'). If None, the format is inferred from the file extension.

    Returns:
    - 
    """
    # import numpy as np
    # import matplotlib.pyplot as plt
    # from sklearn.metrics import roc_curve, auc
    # import polars as pl

    if thresholds is None:
        thresholds = np.linspace(0, 1, 50)

    # Initialize lists to store TPR and FPR for each threshold
    tpr_list = []
    fpr_list = []

    # Step 1: Group true splice sites by gene
    grouped_annotations = annotations_df.filter(pl.col('site_type') == 'donor').group_by('gene_id').agg(
        pl.struct(['start', 'end']).alias('donor_sites')
    ).to_dict(as_series=False)

    # Step 2: Process each gene's predictions in pred_results
    for threshold in tqdm(thresholds, desc="Evaluating thresholds"):
        all_tp, all_fp, all_fn, all_tn = 0, 0, 0, 0

        for gene_id, gene_data in pred_results.items():
            if gene_id not in grouped_annotations['gene_id']:
                continue

            donor_sites = grouped_annotations['donor_sites'][grouped_annotations['gene_id'].index(gene_id)]

            # Take unique donor sites based on both 'start' and 'end' positions
            donor_sites = list({(site['start'], site['end']): site for site in donor_sites}.values())

            # Calculate the precise position of each donor site (midpoint) and convert it to relative position
            true_donor_positions = sorted([
                ((site['start'] + site['end']) // 2) - gene_data['gene_start'] for site in donor_sites
            ])  # sorted in ascending order, i.e. 5' to 3' direction
            true_donor_positions = np.array(true_donor_positions)

            donor_predictions = gene_data['donor_prob']  # List of probabilities for donor sites

            # Initialize label prediction vector based on the threshold
            label_predictions = np.array([1 if prob >= threshold else 0 for prob in donor_predictions])

            if gene_data['strand'] == '-':
                label_predictions = label_predictions[::-1]

            # Create a list to track if a true donor site is missed (for FN counting)
            true_donor_status = {i: False for i in range(len(true_donor_positions))}

            # Initialize counts for the current gene
            TP, TN, FP, FN = 0, 0, 0, 0

            # Step 3: Evaluate the prediction against true splice sites using relative positions
            for i, pred_label in enumerate(label_predictions):
                rel_position = i  # Relative position is simply the index (relative to the gene start)

                found_in_window = False

                # Check if this position is within any true donor site window
                for idx, true_pos in enumerate(true_donor_positions):
                    if true_donor_status[idx]:  # Skip already processed true donor sites
                        continue

                    # Define the relaxed window around the splice site midpoint
                    window_start = true_pos - consensus_window
                    window_end = true_pos + consensus_window

                    if window_start <= rel_position <= window_end:
                        found_in_window = True

                        # If the prediction is 1 within the window, count it as a TP
                        if pred_label == 1:
                            TP += 1
                            true_donor_status[idx] = True
                        break

                # If not found in any window but predicted as 1, it's an FP
                if not found_in_window and pred_label == 1:
                    FP += 1

                # If not found in any window and predicted as 0, it's a TN
                if not found_in_window and pred_label == 0:
                    TN += 1

            # Step 4: After looping through predictions, count false negatives
            for idx, status in true_donor_status.items():
                if not status:
                    # If none of the predictions within the splice site window is 1, count it as an FN
                    FN += 1

            # Accumulate the counts across all genes
            all_tp += TP
            all_tn += TN
            all_fp += FP
            all_fn += FN

        # Step 5: Calculate TPR and FPR for the current threshold
        tpr = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0
        fpr = all_fp / (all_fp + all_tn) if (all_fp + all_tn) > 0 else 0

        tpr_list.append(tpr)
        fpr_list.append(fpr)

    # Step 6: Plot the ROC curve
    # plt.figure(figsize=(10, 6))
    # plt.plot(fpr_list, tpr_list, color='b', lw=2, label='ROC Curve (AUC = {:.2f})'.format(auc(fpr_list, tpr_list)))
    # plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=2)
    # plt.xlabel('False Positive Rate (FPR)')
    # plt.ylabel('True Positive Rate (TPR)')
    # plt.title('ROC Curve for Donor Site Predictions')
    # plt.legend(loc="lower right")
    # plt.grid()
    # plt.show()

    plot_roc_curve(fpr_list, tpr_list, save_path=save_path, file_format=file_format)


# Todo: Create a new module utils_plot
def plot_roc_curve(fpr_list, tpr_list, save_path=None, file_format=None):
    """
    Plot the ROC curve and optionally save it to a file.

    Parameters:
    - fpr_list (list): List of false positive rates.
    - tpr_list (list): List of true positive rates.
    - save_path (str, optional): Path to save the plot. If None, the plot will be shown on the screen.
    - file_format (str, optional): File format to save the plot (e.g., 'png', 'pdf'). If None, the format is inferred from the file extension.

    Example usage:
    plot_roc_curve(fpr_list, tpr_list, save_path='roc_curve.png', file_format='png')
    """
    from sklearn.metrics import auc
    
    plt.figure(figsize=(10, 6))
    plt.plot(fpr_list, tpr_list, color='b', lw=2, label='ROC Curve (AUC = {:.2f})'.format(auc(fpr_list, tpr_list)))
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=2)
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('ROC Curve for Donor Site Predictions')
    plt.legend(loc="lower right")
    plt.grid()

    if save_path:
        if file_format is None:
            file_format = os.path.splitext(save_path)[1][1:]  # Infer file format from file extension
        plt.savefig(save_path, format=file_format)
        print(f"ROC curve saved to {save_path}")
    else:
        plt.show()


def plot_prc_curve(precision_list, recall_list, save_path=None, file_format=None):
    """
    Plot the PRC curve and optionally save it to a file.

    Parameters:
    - precision_list (list): List of precision values.
    - recall_list (list): List of recall values.
    - save_path (str, optional): Path to save the plot. If None, the plot will be shown on the screen.
    - file_format (str, optional): File format to save the plot (e.g., 'png', 'pdf'). If None, the format is inferred from the file extension.

    Example usage:
    plot_prc_curve(precision_list, recall_list, save_path='prc_curve.png', file_format='png')
    """
    from sklearn.metrics import auc
    
    plt.figure(figsize=(10, 6))
    plt.plot(recall_list, precision_list, color='b', lw=2, label='PRC Curve (AUC = {:.2f})'.format(auc(recall_list, precision_list)))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve for Donor Site Predictions')
    plt.legend(loc="lower left")
    plt.grid()

    if save_path:
        if file_format is None:
            file_format = os.path.splitext(save_path)[1][1:]  # Infer file format from file extension
        plt.savefig(save_path, format=file_format)
        print(f"PRC curve saved to {save_path}")
    else:
        plt.show()


def plot_roc_curve_with_uncertainty(fpr_list, tpr_list, figsize=None, title="ROC Curve with Uncertainty", save_path=None, file_format=None):
    """
    Plot ROC curve with shaded areas representing uncertainty.

    Parameters:
    - fpr_list (list or list of lists): List of FPR values or list of lists of FPR values for each CV fold.
    - tpr_list (list or list of lists): List of TPR values or list of lists of TPR values for each CV fold.
    - figsize (tuple, optional): Figure size.
    - title (str, optional): Title of the plot.
    - save_path (str, optional): Path to save the plot. If None, the plot will be shown on the screen.
    - file_format (str, optional): File format to save the plot (e.g., 'png', 'pdf'). If None, the format is inferred from the file extension.

    Example usage:
        Example FPR and TPR lists from different CV folds
        fpr_list = [
            [0.0, 0.1, 0.2, 0.3, 1.0],
            [0.0, 0.05, 0.15, 0.25, 1.0],
            [0.0, 0.2, 0.3, 0.4, 1.0]
        ]
        tpr_list = [
            [0.0, 0.4, 0.6, 0.8, 1.0],
            [0.0, 0.5, 0.7, 0.85, 1.0],
            [0.0, 0.3, 0.5, 0.7, 1.0]
        ]
        plot_roc_curve_with_uncertainty(fpr_list, tpr_list, save_path='roc_curve.png', file_format='png')
    """
    from sklearn.metrics import auc

    plt.clf()

    # Set up the figure to plot the ROC curves
    if figsize is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    else:
        fig, ax = plt.subplots(figsize=figsize)

    # Check if the input is a list of lists or flat lists
    if isinstance(fpr_list[0], list) and isinstance(tpr_list[0], list):
        # Interpolate TPRs at common FPR points
        mean_fpr = np.linspace(0, 1, 100)
        tprs = []

        for fpr, tpr in zip(fpr_list, tpr_list):
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, alpha=0.3, label='ROC fold (AUC = %0.2f)' % roc_auc)

        # Calculate mean and standard deviation of TPRs
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std([auc(fpr, tpr) for fpr, tpr in zip(fpr_list, tpr_list)])

        # Plot the mean ROC curve
        ax.plot(mean_fpr, mean_tpr, color='b', label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc), lw=2, alpha=.8)

        # Plot the standard deviation around the mean ROC curve
        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2, label=r'$\pm$ 1 std. dev.')

    else:
        # Plot the single ROC curve
        roc_auc = auc(fpr_list, tpr_list)
        ax.plot(fpr_list, tpr_list, color='b', lw=2, label='ROC Curve (AUC = {:.2f})'.format(roc_auc))
        ax.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=2)

    # Plot random guessing (diagonal)
    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Random guessing', alpha=.8)

    # Set plot labels and legend
    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05], title=title)
    ax.set_xlabel('False Positive Rate (FPR)')
    ax.set_ylabel('True Positive Rate (TPR)')
    ax.legend(loc="lower right")

    # Save or show the plot
    if save_path:
        plt.savefig(save_path, format=file_format)
        print(f"ROC curve saved to {save_path}")
    else:
        plt.show()


####################################################################################################
def determine_optimal_threshold_for_junction_matching(donor_probs, acceptor_probs, target_junctions=43, initial_threshold=0.9):
    """
    Determine an optimal threshold for donor and acceptor probabilities to match the target number of junctions.

    Parameters:
    - donor_probs (list): List of donor site probabilities.
    - acceptor_probs (list): List of acceptor site probabilities.
    - target_junctions (int): Desired number of junctions.
    - initial_threshold (float): Starting threshold for probabilities.

    Returns:
    - float: Optimal probability threshold.
    - int: Number of junctions at the optimal threshold.

    Example usage with provided donor and acceptor probabilities:

        donor_probs = [0.9995985, 0.99946916, 0.99929726, 0.9991991, 0.999012, ...]  # Truncated for brevity
        acceptor_probs = [0.99957716, 0.9989233, 0.99875623, 0.9983406, 0.9983163, ...]  # Truncated for brevity
        optimal_threshold, junctions = determine_optimal_threshold_for_junction_matching(donor_probs, acceptor_probs, target_junctions=43)
        print(f"Optimal Threshold: {optimal_threshold:.2f}, Number of Junctions: {junctions}")

    """
    threshold = initial_threshold
    step = 0.01  # Adjust step size as needed

    while True:
        # Count donor and acceptor sites above the current threshold
        valid_donors = sum(donor_prob >= threshold for donor_prob in donor_probs)
        valid_acceptors = sum(acceptor_prob >= threshold for acceptor_prob in acceptor_probs)

        # Calculate number of valid junctions
        # Assuming each valid donor pairs with the nearest valid acceptor
        valid_junctions = min(valid_donors, valid_acceptors)

        print(f"Threshold: {threshold:.2f}, Donors: {valid_donors}, Acceptors: {valid_acceptors}, Junctions: {valid_junctions}")

        # Check if the number of junctions is close to the target
        if valid_junctions == target_junctions:
            return threshold, valid_junctions

        # Adjust threshold downwards if too few junctions, upwards if too many
        if valid_junctions > target_junctions:
            threshold += step
        else:
            threshold -= step

        # Prevent threshold from going below 0
        if threshold <= 0:
            break

    # Return the closest found threshold and number of junctions
    return threshold, valid_junctions


def is_within_tolerance(threshold, center=0.5, tolerance=0.05):
    """
    Check if the threshold is within the specified tolerance range around the center value.

    Parameters:
    - threshold (float): The threshold value to check.
    - center (float): The center value to compare against (default is 0.5).
    - tolerance (float): The tolerance range around the center value (default is 0.05).

    Returns:
    - bool: True if the threshold is within the tolerance range, False otherwise.
    """
    return center - tolerance <= threshold <= center + tolerance


def evaluate_performance_across_thresholds(annotations_df, pred_results, thresholds=np.arange(0.0, 1.05, 0.05), consensus_window=2, **kargs):
    """
    Evaluate the model's performance across a range of thresholds to compute various performance metrics.

    Parameters:
    - annotations_df (pl.DataFrame): DataFrame containing true splice site annotations.
      Columns: ['chrom', 'start', 'end', 'strand', 'site_type', 'gene_id', 'transcript_id']
    - pred_results (dict): The output of predict_splice_sites_for_genes(), containing per-nucleotide probabilities.
    - thresholds (array): List or array of threshold values to evaluate (default is np.arange(0.0, 1.05, 0.05)).
    - consensus_window (int): Tolerance window around true splice sites.

    Returns:
    - metrics_df (pl.DataFrame): DataFrame containing all performance metrics across thresholds.
    """
    metrics_data = {
        'threshold': [], 'tpr': [], 'fpr': [], 'precision': [], 'recall': [], 'f1_score': [], 'fnr': []
    }

    th_acc = 0
    for threshold in tqdm(thresholds, desc="Evaluating thresholds"):
        print_emphasized(f"[info] threshold at: {threshold:.2f}")
        results_df = \
            evaluate_splice_site_predictions(
                annotations_df, pred_results, 
                threshold=threshold, 
                consensus_window=consensus_window, 
                verbose=is_within_tolerance(threshold, 0.5, 0.05))

        # Calculate TP, TN, FP, FN across all genes
        total_tp = results_df['TP'].sum()
        total_tn = results_df['TN'].sum()
        total_fp = results_df['FP'].sum()
        total_fn = results_df['FN'].sum()

        # ROC metrics: True Positive Rate (TPR) and False Positive Rate (FPR)
        tpr = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        fpr = total_fp / (total_fp + total_tn) if (total_fp + total_tn) > 0 else 0

        # PRC metrics: Precision, Recall, F1 Score, and False Negative Rate (FNR)
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        fnr = total_fn / (total_fn + total_tp) if (total_fn + total_tp) > 0 else 0

        # Append metrics to the data dictionary
        metrics_data['threshold'].append(threshold)
        metrics_data['tpr'].append(tpr)
        metrics_data['fpr'].append(fpr)
        metrics_data['precision'].append(precision)
        metrics_data['recall'].append(recall)
        metrics_data['f1_score'].append(f1_score)
        metrics_data['fnr'].append(fnr)

        # Debugging
        print(f"Threshold: {threshold:.2f} | TPR: {tpr:.4f}, FPR: {fpr:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1_score:.4f}, FNR: {fnr:.4f}")
        th_acc += 1

    # Convert data to polars DataFrame
    metrics_df = pl.DataFrame(metrics_data)

    return metrics_df


def clean_up_intermediate_files(local_dir, chromosomes):
    """
    Delete intermediate performance files for each chunk to save space.

    Parameters:
    - local_dir (str): Directory where the intermediate files are stored.
    - chromosomes (list): List of chromosome identifiers to delete associated files.
    """
    for chr in chromosomes:
        for file in os.listdir(local_dir):
            if file.startswith(f"splice_performance_{chr}_chunk_"):
                file_path = os.path.join(local_dir, file)
                os.remove(file_path)
                print(f"[cleanup] Deleted intermediate file: {file_path}")
    print("[cleanup] All intermediate files deleted.")


def print_global_summary(global_summary):
    """
    Print the global performance summary in a readable format.

    Parameters:
    - global_summary (dict): The global performance summary.
    """
    print("\nGlobal Performance Summary:")
    print(f"{'True Positives (TP):':<30} {global_summary['TP']}")
    print(f"{'True Negatives (TN):':<30} {global_summary['TN']}")
    print(f"{'False Positives (FP):':<30} {global_summary['FP']}")
    print(f"{'False Negatives (FN):':<30} {global_summary['FN']}")
    print("-" * 85)
    print(f"{'Precision:':<30} {global_summary['precision']:.4f}")
    print(f"{'Recall:':<30} {global_summary['recall']:.4f}")
    print(f"{'Specificity:':<30} {global_summary['specificity']:.4f}")
    print(f"{'F1 Score:':<30} {global_summary['f1_score']:.4f}")
    print("-" * 85)
    print(f"{'False Positive Rate (FPR):':<30} {global_summary['fpr']:.4f}")
    print(f"{'False Negative Rate (FNR):':<30} {global_summary['fnr']:.4f}")


def demo_aggregate_performance_files(): 

    local_dir = '/path/to/meta-spliceai/data/ensembl/'
    src_dir = '/path/to/meta-spliceai/data/ensembl/'
    eval_dir = '/path/to/meta-spliceai/data/ensembl/spliceai_eval'

    # gtf_file = "/path/to/meta-spliceai/data/ensembl/Homo_sapiens.GRCh38.112.gtf"  # Replace with your GTF file path
    # genome_fasta = os.path.join(src_dir, "Homo_sapiens.GRCh38.dna.primary_assembly.fa") 
    
    global_summary = \
        aggregate_performance_files(
            local_dir, eval_dir, mode='gene', seq_type='full', format='parquet', 
            save_to_file=False, test_mode=False)

    # Print the global summary in a readable format
    print_global_summary(global_summary)

    return


def test(): 

    # Aggregate performance files and show the global summary
    demo_aggregate_performance_files()
    # Example output:
    # 
    # Global Performance Summary:
    # True Positives (TP): 475088
    # True Negatives (TN): 4041751963
    # False Positives (FP): 64914
    # False Negatives (FN): 177578
    # -------------------------------------
    # Precision: 0.8798
    # Recall: 0.7279
    # Specificity: 1.0000
    # F1 Score: 0.7967
    # -------------------------------------
    # False Positive Rate (FPR): 0.0000
    # False Negative Rate (FNR): 0.2721


if __name__ == "__main__":
    test()
