import os
import re
import numpy as np

import pandas as pd
import polars as pl

from pybedtools import BedTool
from Bio import SeqIO
from collections import defaultdict
from tqdm import tqdm

from statistics import median
from meta_spliceai.splice_engine.utils_bio import normalize_strand


def reverse_complement(seq):
    complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
    return ''.join(complement[base] for base in reversed(seq))


def generate_label_arrays_v0(annotations_df, gene_sequences):
    """
    Generate label arrays for splice site prediction based on gene sequences and annotations.

    Parameters:
    - annotations_df (pl.DataFrame): DataFrame containing annotations for splice sites (donor, acceptor).
                                     Columns: ['chrom', 'start', 'end', 'strand', 'site_type', 'gene_id', 'transcript_id']
    - gene_sequences (pl.DataFrame): DataFrame containing gene sequences with positions and gene information.
                                     Columns: ['gene_id', 'seqname', 'gene_name', 'start', 'end', 'strand', 'sequence']

    Returns:
    - labels_dict (dict): A dictionary where each key is a gene_id, and the value is a dict with:
        - 'donor_labels': Array of 0s and 1s (1 at donor sites).
        - 'acceptor_labels': Array of 0s and 1s (1 at acceptor sites).
    """

    # Pre-filter annotations_df for the relevant genes in gene_sequences
    relevant_gene_ids = gene_sequences.select('gene_id').unique().to_series().to_list()
    filtered_annotations_df = annotations_df.filter(pl.col('gene_id').is_in(relevant_gene_ids))

    # Group annotations by gene_id for efficient lookups
    grouped_annotations = filtered_annotations_df.group_by('gene_id').agg(
        pl.struct(['start', 'end', 'strand', 'site_type']).alias('annotations')
    ).to_dict(as_series=False)

    # Ensure the structure of the grouped annotations is correct
    print("> Grouped annotations dictionary:")
    print(grouped_annotations)

    labels_dict = {}

    # Iterate over gene sequences in the current chunk
    for row in gene_sequences.iter_rows(named=True):
        gene_id = row['gene_id']
        gene_seq = row['sequence']
        gene_len = len(gene_seq)
        gene_start = row['start']
        gene_end = row['end']
        strand = row['strand']

        print(f"> Processing gene: {gene_id}, Start: {gene_start}, End: {gene_end}, Length: {gene_len}")
        # assert gene_len == gene_end - gene_start + 1, f"Invalid gene length: {gene_len} != {gene_end - gene_start + 1}"

        # Initialize label arrays of length equal to the gene sequence
        donor_labels = [0] * gene_len
        acceptor_labels = [0] * gene_len

        # Get annotations for this gene from the pre-grouped dictionary
        if gene_id in grouped_annotations['gene_id']:
            # Get the corresponding annotations list for this gene
            annotations = grouped_annotations['annotations'][grouped_annotations['gene_id'].index(gene_id)]

            print("... Annotations for this gene:")
            print(annotations)

            # Iterate over the splice site annotations for this gene
            for ann_row in annotations:
                site_type = ann_row['site_type']
                site_start = ann_row['start']
                site_end = ann_row['end']
                site_strand = ann_row['strand']

                # Ensure the annotation matches the strand of the gene
                if site_strand != strand:
                    continue

                print(f"... Gene ID: {gene_id}, Splice site: {site_start}-{site_end}, Type: {site_type}")
                assert type(site_start) == int and type(site_end) == int, f"Invalid site positions: {site_start}-{site_end}"

                # Convert absolute positions (start, end) to relative positions in the gene sequence
                relative_start = max(0, site_start - gene_start)  # Ensure it's >= 0
                relative_end = min(gene_len - 1, max(0, site_end - gene_start))  # Ensure both ends are >= 0 

                print(f"... site_start: {site_start} - gene_start: {gene_start} =>  relative_start: {relative_start}")
                print(f"... site_end: {site_end} - gene_start: {gene_start} =>  relative_end: {relative_end}")
                assert relative_end > relative_start, f"Invalid relative positions: {relative_start}-{relative_end}"
                print(f"... Relative position: {relative_start}-{relative_end} <? len={gene_len})")

                if relative_start >= gene_len or relative_end < 0:
                    continue  # Ignore splice sites that are completely outside the gene sequence range

                # Mark the labels based on site_type (donor or acceptor)
                if site_type == 'donor':
                    for pos in range(relative_start, relative_end + 1):
                        donor_labels[pos] = 1
                elif site_type == 'acceptor':
                    for pos in range(relative_start, relative_end + 1):
                        acceptor_labels[pos] = 1

        # Store the label arrays in the dictionary
        labels_dict[gene_id] = {
            'donor_labels': donor_labels,
            'acceptor_labels': acceptor_labels
        }

    return labels_dict


def generate_label_arrays(annotations_df, gene_sequences):
    """
    Generate label arrays for splice site prediction based on gene sequences and annotations.

    Parameters:
    - annotations_df (pl.DataFrame): DataFrame containing annotations for splice sites (donor, acceptor).
                                     Columns: ['chrom', 'start', 'end', 'strand', 'site_type', 'gene_id', 'transcript_id']
    - gene_sequences (pl.DataFrame): DataFrame containing gene sequences with positions and gene information.
                                     Columns: ['gene_id', 'seqname', 'gene_name', 'start', 'end', 'strand', 'sequence']

    Returns:
    - labels_dict (dict): A dictionary where each key is a gene_id, and the value is a dict with:
        - 'donor_labels': Array of 0s and 1s (1 at donor sites).
        - 'acceptor_labels': Array of 0s and 1s (1 at acceptor sites).

    Memo: 
    -  This function looks up splice site annotations for each gene using the annotations_lookup dictionary, 
       processes them, and marks donor and acceptor sites in the gene sequence by creating 
       the corresponding label arrays. The gene sequences and labels are stored in a dictionary for easy retrieval.   
    """

    # Pre-filter annotations_df for the relevant genes in gene_sequences
    relevant_gene_ids = gene_sequences.select('gene_id').unique().to_series().to_list()
    print(relevant_gene_ids)
    filtered_annotations_df = annotations_df.filter(pl.col('gene_id').is_in(relevant_gene_ids))

    # Group annotations by gene_id for efficient lookups
    grouped_annotations = filtered_annotations_df.group_by('gene_id').agg(
        pl.struct(['start', 'end', 'strand', 'site_type']).alias('annotations')
    )

    # Convert the result to a dictionary where keys are gene_id, and values are the annotations
    annotations_lookup = {row['gene_id']: row['annotations'] for row in grouped_annotations.iter_rows(named=True)}
    # NOTE: annotations_lookup is a dictionary where keys are gene_id and the values are the corresponding 
    #       splice site annotations. This allows for constant-time lookup of annotations for each gene 
    #       during sequence iteration.

    # Dictionary to store the labels
    labels_dict = {}

    # Iterate over gene sequences in the current chunk
    for row in gene_sequences.iter_rows(named=True):
        gene_id = row['gene_id']
        gene_seq = row['sequence']
        gene_len = len(gene_seq)
        gene_start = row['start']
        gene_end = row['end']
        strand = row['strand']

        print(f"> Processing gene: {gene_id}, Start: {gene_start}, End: {gene_end}, Length: {gene_len}")

        # Initialize label arrays of length equal to the gene sequence
        donor_labels = [0] * gene_len
        acceptor_labels = [0] * gene_len

        # Get annotations for this gene from the pre-grouped dictionary
        annotations = annotations_lookup.get(gene_id, [])
        
        # Test the annotations for the gene
        # -----------------------------------------------------------
        print(f"... Found {len(annotations)} annotations for gene {gene_id}")

        # Count the number of donor and acceptor sites
        num_donor_sites = sum(1 for ann_row in annotations if ann_row['site_type'] == 'donor')
        num_acceptor_sites = sum(1 for ann_row in annotations if ann_row['site_type'] == 'acceptor')

        # The number of well-paired donor and acceptor sites is the minimum of the two counts
        num_annotated_junctions = min(num_donor_sites, num_acceptor_sites)

        print(f"... Number of donor sites: {num_donor_sites}")
        print(f"... Number of acceptor sites: {num_acceptor_sites}")
        print(f"... Number of annotated junctions: {num_annotated_junctions}")
        # -----------------------------------------------------------

        # Iterate over the splice site annotations for this gene
        n_splice_sites = 0
        for ann_row in annotations:
            site_type = ann_row['site_type']
            site_start = ann_row['start']
            site_end = ann_row['end']
            site_strand = ann_row['strand']

            # Ensure the annotation matches the strand of the gene
            if site_strand != strand:
                continue

            # Convert absolute positions (start, end) to relative positions in the gene sequence
            relative_start = max(0, site_start - gene_start)  # Ensure it's >= 0
            relative_end = min(gene_len - 1, max(0, site_end - gene_start))  # Ensure both ends are >= 0

            if relative_start >= gene_len or relative_end < 0:
                continue  # Ignore splice sites that are completely outside the gene sequence range

            print(f"... Gene ID: {gene_id}, Splice site: {site_start}-{site_end}, Type: {site_type}")
            print(f"... Relative position: {relative_start}-{relative_end}")

            # Mark the labels based on site_type (donor or acceptor)
            if site_type == 'donor':
                for pos in range(relative_start, relative_end + 1):
                    donor_labels[pos] = 1
            elif site_type == 'acceptor':
                for pos in range(relative_start, relative_end + 1):
                    acceptor_labels[pos] = 1

            n_splice_sites += 1

        # Store the label arrays in the dictionary
        labels_dict[gene_id] = {
            'donor_labels': donor_labels,
            'acceptor_labels': acceptor_labels
        }

    return labels_dict


def label_splice_sites_with_decoys_v0(annotations_df, gene_sequences, consensus_window=2):
    """
    Label true and decoy splice sites for each gene based on consensus sequences (CAG, TAG, AAG for acceptors, GT/GC for donors).

    Parameters:
    - annotations_df (pl.DataFrame): DataFrame containing annotations for splice sites (donor, acceptor).
                                     Columns: ['chrom', 'start', 'end', 'strand', 'site_type', 'gene_id', 'transcript_id']
    - gene_sequences (pl.DataFrame): DataFrame containing gene sequences.
                                     Columns: ['transcript_id', 'gene_id', 'gene_name', 'strand', 'seqname', 'start', 'end', 'sequence']
    - consensus_window (int): Number of nucleotides around the splice site to consider for tolerance.

    Returns:
    - labels_dict (dict): Dictionary where each key is a gene_id, and each value is a dict containing:
        - 'true_sites': List of positions for true splice sites.
        - 'decoy_sites': List of positions for decoy splice sites.
    """

    # Step 1: Create a dictionary to store the labels for each gene
    labels_dict = {}

    gene_id_counter = 0

    # Step 2: Iterate over the gene sequences
    for row in gene_sequences.iter_rows(named=True):
        gene_id = row['gene_id']
        strand = row['strand']
        gene_seq = row['sequence']
        gene_start = row['start']

        # Initialize true and decoy lists for this gene
        true_sites = []
        decoy_sites = []

        # Step 3: Get all splice sites for this gene from annotations_df
        gene_splice_sites = annotations_df.filter(pl.col('gene_id') == gene_id).to_dict(as_series=False)

        # Test: Print and inspect gene_splice_sites for the first two gene_ids
        if gene_id_counter < 2:
            print(f"Gene ID: {gene_id}")
            print("Gene Splice Sites:")
            for key, value in gene_splice_sites.items():
                print(f"{key}: {value}")
            print("\n")

        # Step 4: Search for consensus sequences in the gene sequence
        donor_motifs = ['GT', 'GC']
        acceptor_motifs = ['CAG', 'TAG', 'AAG']

        # Step 5: Label donor and acceptor sites in the gene sequence
        for match in re.finditer(r'GT|GC|CAG|TAG|AAG', gene_seq):
            match_start = match.start()
            match_end = match.end()

            # Identify if this match is a donor or acceptor motif
            motif = gene_seq[match_start:match_end]
            is_donor = motif in donor_motifs
            is_acceptor = motif in acceptor_motifs

            # Step 6: Check if this consensus sequence matches a true splice site
            true_splice_match = any(
                (gene_id == site_gene_id and match_start >= site_start - gene_start and match_end <= site_end - gene_start)
                for site_gene_id, site_start, site_end, site_type in zip(
                    gene_splice_sites['gene_id'],
                    gene_splice_sites['start'],
                    gene_splice_sites['end'],
                    gene_splice_sites['site_type']
                )
            )

            # Step 7: Assign labels (1 for true splice site, 0 for decoy)
            if true_splice_match:
                true_sites.append((match_start, match_end))  # True splice site positions
            else:
                decoy_sites.append((match_start, match_end))  # Decoy splice site positions

        # Step 8: Store the labeled positions in the dictionary for this gene
        labels_dict[gene_id] = {
            'true_sites': true_sites,
            'decoy_sites': decoy_sites
        }

        gene_id_counter += 1 

    return labels_dict


def find_consensus_sites(sequence, pattern):
    """
    Finds all locations of a given pattern (consensus sequence) in the DNA string.

    Parameters:
    - sequence (str): DNA sequence to search for the consensus pattern.
    - pattern (str): Consensus pattern (regex format) to search in the sequence.
    
    Returns:
    - matches (list of tuples): A list of tuples containing (start_position, matched_pattern).
    """
    # Use regular expression to search for the consensus sequence pattern in the given DNA sequence
    matches = [(m.start(), m.group()) for m in re.finditer(pattern, sequence)]
    return matches


def convert_relative_to_absolute_positions(gene_start, gene_end, matches, strand, **kargs):
    """
    Convert the relative match positions from the DNA sequence to absolute positions.
    Returns a list of tuples with absolute start and end positions.
    
    Parameters:
    - gene_id: The gene identifier.
    - gene_start: The starting position of the gene.
    - gene_end: The ending position of the gene.
    - matches: A list of match tuples (relative_position, matched_pattern).
    - strand: The strand of the gene ('+' or '-').
    
    Returns:
    - A list of tuples: (absolute_start, absolute_end, matched_pattern)

    Memo: 
    - On the negative strand, the coordinates from the gene annotation (start to end) are presented 
      in a smaller-to-larger order for consistency, but the biological reading of the sequence is from 3' to 5' 
      (end to start). This means that, in the case of a match on the negative strand, 
      the absolute start position (in genomic coordinates) will initially appear larger than 
      the absolute end position due to the reverse nature of the strand.


    """
    results = []
    for rel_pos, pattern in matches:
        match_start = rel_pos
        match_end = rel_pos + len(pattern) - 1  # Capture the full length of the match
        # print("**** match_start, match_end:", match_start, match_end)
        # print("******* gene_start: ", gene_start)

        if strand == '+':
            absolute_start = gene_start + match_start # - 1
            absolute_end = gene_start + match_end # - 1
            # print("********* absolute_start, absolute_end:", absolute_start, absolute_end)
        else:
            # On the negative strand, calculate as before but swap using tuple unpacking
            absolute_start = gene_end - match_start # + 1
            absolute_end = gene_end - match_end # + 1

            # Swap to ensure start < end for consistency in annotation
            absolute_start, absolute_end = min(absolute_start, absolute_end), max(absolute_start, absolute_end)
          
        results.append((absolute_start, absolute_end, pattern))

    return results


def calculate_absolute_position(gene_id, relative_position, gene_annotations):
    """

    
    Memo: 
    - 'gene_annotations' is a dictionary containing gene annotations with the following structure:
        gene_annotations = {
            'STMN2': {'start': 85000, 'end': 110000, 'strand': '+'},  # positive strand
            'UNC13A': {'start': 17601336, 'end': 17688365, 'strand': '-'}  # negative strand
        }
    """
    gene_data = gene_annotations[gene_id]
    strand = gene_data['strand']
    
    if strand == '+':
        # Positive strand: Start position + relative position - 1
        return gene_data['start'] + relative_position - 1
    elif strand == '-':
        # Negative strand: End position - relative position + 1
        return gene_data['end'] - relative_position + 1
    else:
        raise ValueError("Strand information is invalid")


def label_splice_sites_with_decoys(
        annotations_df,
        gene_sequences_df,
        donor_consensus=('GT', 'GC'), 
        acceptor_consensus=('CAG', 'TAG', 'AAG'),
        consensus_window=2, 
        save_to_file=False,
        file_path=None,
        return_df=False):
    """
    Label true and decoy splice sites by comparing consensus sequence matches to known splice site positions.
    
    Parameters:
    - gene_sequences_df (pl.DataFrame): DataFrame containing gene sequences.
      Columns: ['transcript_id', 'gene_id', 'gene_name', 'strand', 'seqname', 'start', 'end', 'sequence']

      NOTE: 'transcript_id' may not exist for gene specific sequences

    - annotations_df (pl.DataFrame): DataFrame containing annotations for splice sites.
      Columns: ['chrom', 'start', 'end', 'strand', 'site_type', 'gene_id', 'transcript_id']
    - donor_consensus (tuple): Consensus sequences to search for donor sites (e.g., 'GT', 'GC')
    - acceptor_consensus (tuple): Consensus sequences to search for acceptor sites (e.g., 'CAG', 'TAG', 'AAG')
    - consensus_window (int): Number of nucleotides to include before and after the true splice site for tolerance.
    - save_to_file (bool): Whether to save the output to a file.
    - file_path (str): Path to the file where the output should be saved.
    - return_df (bool): Whether to return the output as a DataFrame.

    Returns:
    - labeled_sites (list): A list of dicts, each containing:
      {
        'chrom': chrom,
        'start': start,  # start position of the consensus sequence match
        'end': end,      # end position of the consensus sequence match
        'strand': strand,
        'site_type': 'donor', 'decoy donor', 'acceptor', or 'decoy acceptor',
        'gene_id': gene_id,
        'transcript_id': transcript_id
      }
    """
    labeled_sites = []

    counters = {
        'donor': 0,
        'decoy donor': 0,
        'acceptor': 0,
        'decoy acceptor': 0
    }

    # Step 1: Group splice sites by gene for easier lookup
    grouped_annotations = annotations_df.group_by('gene_id').agg(
        pl.struct(['start', 'end', 'strand', 'site_type', 'transcript_id']).alias('splice_sites')
    ).to_dict(as_series=False)

    # Step 2: Process each gene in the gene_sequences_df
    for gene_row in gene_sequences_df.iter_rows(named=True):
        gene_id = gene_row['gene_id']

        gene_sequence = gene_row['sequence'] 
        # NOTE: Sequences on the negative strand are reverse complemented

        strand = gene_row['strand']
        chrom = gene_row['seqname']
        gene_start = gene_row['start']
        gene_end = gene_row['end']
        transcript_id = gene_row.get('transcript_id', 'n/a')

        # Debugging: Print gene information for verification
        print(f"\n> Processing gene: {gene_id} (Transcript: {transcript_id})")
        print(f"... Chromosome: {chrom}, Strand: {strand}, Start: {gene_start}, End: {gene_end}")
        print(f"... Gene sequence length: {len(gene_sequence)}")

        # Step 3: Retrieve true splice sites for this gene
        true_splice_sites = []
        if gene_id in grouped_annotations['gene_id']:
            index = grouped_annotations['gene_id'].index(gene_id)
            true_splice_sites = grouped_annotations['splice_sites'][index]
            print(f"> Found {len(true_splice_sites)} true splice sites for {gene_id}")
        else:
            print(f"> No splice sites found for {gene_id}.")

        # Debugging: Print out true splice sites for further inspection
        # -----------------------------------------------------------
        print("> True splice sites:")
        n_matched_donors = n_matched_acceptors = 0
        n_true_donors = n_true_acceptors = 0
        unique_splice_sites = set()

        for site in true_splice_sites:
            site_start = site['start'] - gene_start  # Adjust to relative position in gene sequence
            site_end = site['end'] - gene_start
            site_type = site['site_type']
            strand = normalize_strand(site['strand'])

            splice_site_pos = (site_start + site_end) // 2 -1  # Calculate the midpoint and adjust for 0-based indexing
            # NOTE: Need to -1 to adjust for 0-based indexing? but splice_site_pos should already be 0-based without -1

            if (site_start, site_end, site_type) in unique_splice_sites:
                continue

            # Examine the sequence within the relaxed donor site window
            relaxed_seq = gene_sequence[site_start:site_end + 1]  # Check the entire relaxed window
            print(f"[relaxed splice site]: {relaxed_seq}, splie site at: {gene_sequence[splice_site_pos]}")
            # ... ok
            
            # Determine the splice site type and check the consensus sequences
            if site_type == 'donor':
                # Donor site: Check if consensus is GT or GC at [splice_site_pos, splice_site_pos+1]
                donor_seq = gene_sequence[splice_site_pos:splice_site_pos + 2]
                n_true_donors += 1

                matches = [(splice_site_pos, donor_seq), ]  # A list of match tuples (relative_position, matched_pattern).
                donor_matches = \
                    convert_relative_to_absolute_positions(gene_start, gene_end, matches, strand)
                absolute_start, absolute_end, pattern = donor_matches[0]
                print(f"... pattern={pattern} => absolute coord? {(absolute_start, absolute_end)} (ss={splice_site_pos+gene_start})")

                if donor_seq in donor_consensus:
                    print(f"... Donor site at {splice_site_pos}: Found {donor_seq}")
                    n_matched_donors += 1

                    # Examine the sequence within the relaxed donor site window
                    relaxed_seq = gene_sequence[site_start:site_end + 1]  # Check the entire relaxed window
                    print(f"...... Matched donor site window: {relaxed_seq}")
                else:
                    print(f"... Donor site at {splice_site_pos}: Mismatch! Found {donor_seq}")

                    # Examine the sequence within the relaxed donor site window
                    relaxed_seq = gene_sequence[site_start:site_end + 1]  # Check the entire relaxed window
                    print(f"...... Relaxed donor site window: {relaxed_seq}")

            elif site_type == 'acceptor':
                # Acceptor site: Check if consensus is CAG, TAG, AAG at [splice_site_pos-2, splice_site_pos]
                a = len_acceptor_consensus = len(acceptor_consensus[0])
                acceptor_seq = gene_sequence[splice_site_pos - 2:splice_site_pos + 1]
                n_true_acceptors += 1

                matches = [(splice_site_pos, acceptor_seq), ]  # A list of match tuples (relative_position, matched_pattern).
                acceptor_matches = \
                    convert_relative_to_absolute_positions(gene_start, gene_end, matches, strand)
                absolute_start, absolute_end, pattern = acceptor_matches[0]
                print(f"... pattern={pattern} => absolute coord? {(absolute_start, absolute_end)} (ss={splice_site_pos+gene_start})")

                # If the matched acceptor sequence ends as a substring of any of the consensus sequences, 
                # then it's a match
                if any(acceptor_seq.endswith(consensus) for consensus in acceptor_consensus):
                    print(f"... Acceptor site at {splice_site_pos}: Found {acceptor_seq}")
                    n_matched_acceptors += 1

                    # Examine the sequence within the relaxed donor site window
                    relaxed_seq = gene_sequence[site_start:site_end + 1]  # Check the entire relaxed window
                    print(f"...... Matched acceptor site window: {relaxed_seq}")
                else:
                    print(f"... Acceptor site at {splice_site_pos}: Mismatch! Found {acceptor_seq}")

                    # Examine the sequence within the relaxed acceptor site window
                    relaxed_seq = gene_sequence[site_start:site_end + 1]  # Check the entire relaxed window
                    print(f"...... Relaxed acceptor site window: {relaxed_seq}")

            unique_splice_sites.add((site_start, site_end, site_type))
        
        print(f"[debug] Matched donor sites: {n_matched_donors} / n_true_donors: {n_true_donors}")
        print(f"[debug] Matched acceptor sites: {n_matched_acceptors} / n_true_acceptors: {n_true_acceptors}")

        # -----------------------------------------------------------

        # Step 4: Search for donor and acceptor consensus sequences in the gene sequence
        donor_matches = []
        for donor_seq in donor_consensus:
            donor_matches += find_consensus_sites(gene_sequence, donor_seq)
        print(f"Found {len(donor_matches)} donor consensus matches (GT/GC).")
        
        acceptor_matches = []
        for acceptor_seq in acceptor_consensus:
            acceptor_matches += find_consensus_sites(gene_sequence, acceptor_seq)
        print(f"> Found {len(acceptor_matches)} acceptor consensus matches (CAG/TAG/AAG).")

        # Step 5: Convert relative positions to absolute positions
        donor_sites_abs = convert_relative_to_absolute_positions(gene_start, gene_end, donor_matches, strand)
        acceptor_sites_abs = convert_relative_to_absolute_positions(gene_start, gene_end, acceptor_matches, strand)
        # Both are a list of tuples: (absolute_start, absolute_end, matched_pattern)

        # Debugging: Check absolute positions
        print("> Donor absolute positions:", donor_sites_abs[:10])
        print("> Acceptor absolute positions:", acceptor_sites_abs[:10])

        # Step 6: Compare inferred splice sites with true splice sites to separate true and decoy sites
        tx_ids = set()
        unique_splice_sites = set()
        for site in true_splice_sites:
            site_start = site['start'] 
            site_end = site['end'] 
            site_type = site['site_type']
            site_strand = normalize_strand(site['strand'])
            transcript_id = site['transcript_id']
            tx_ids.add(transcript_id)

            # Ensure the site matches the gene's strand
            if site_strand != strand:
                print(f"> Skipping site {site_start}-{site_end} (strand mismatch)")
                continue

            # Ensure the site is unique (not already processed)
            if (site_start, site_end, site_type) in unique_splice_sites:
                print(f"> Skipping site {site_start}-{site_end} (duplicate)") 
                
                # The same splice site may be present in multiple transcripts
                continue

            # The actual splice site is the midpoint of the start and end positions
            splice_site_pos = (site_start + site_end) // 2 -1

            # Debugging: Print the true splice site information
            # print(f"> {site['transcript_id']}: {site['site_type']} at [{site['start']}, {site['end']}] (midpoint: {splice_site_pos})")

            # Step 7: Check for true donor and acceptor sites
            if site_type == 'donor':
                for abs_start, abs_end, pattern in donor_sites_abs:
                    # Check if the matched region lies within the true splice site boundary
                     
                    # is_true_splice_site = (abs_start >= site_start and abs_end <= site_end)
                    is_true_splice_site = abs(splice_site_pos - abs_start) <= consensus_window

                    # We don't need the strand dependent logic because the negative-strand sequences are 
                    # already reverse complemented
                    # if strand == '+':
                    #     is_true_splice_site = (abs_start >= splice_site_pos and abs_end <= site_end)
                    # else: 
                    #     is_true_splice_site = (abs_start >= site_start and abs_end <= splice_site_pos)

                    assert abs_start <= abs_end, f"Donor site start is greater than end: {abs_start}, {abs_end}"
                    site_label = 'donor' if is_true_splice_site else 'decoy donor'

                    if is_true_splice_site:
                        labeled_sites.append({
                            'chrom': chrom,
                            'start': abs_start,
                            'end': abs_end,
                            'strand': strand,
                            'site_type': site_label,
                            'gene_id': gene_id,
                            'transcript_id': transcript_id
                        })
                        counters['donor'] += 1

                        print(f"... donor pattern={pattern}: {abs_start}-{abs_end} (near {splice_site_pos}?)")
                    else:
                        labeled_sites.append({
                            'chrom': chrom,
                            'start': abs_start,
                            'end': abs_end,
                            'strand': strand,
                            'site_type': site_label,
                            'gene_id': gene_id,
                            'transcript_id': transcript_id
                        })
                        counters['decoy donor'] += 1

                    # print(f"... pattern: {pattern} => L={site_label} | {abs_start}-{abs_end} (near {splice_site_pos}?)")

            elif site_type == 'acceptor':

                for abs_start, abs_end, pattern in acceptor_sites_abs:
                    # Check if the matched region lies within the true splice site boundary

                    # is_true_splice_site = (abs_start >= site_start and abs_end <= site_end) 
                    is_true_splice_site = abs(splice_site_pos - abs_start) <= consensus_window

                    assert abs_start <= abs_end, f"Acceptor site start is greater than end: {abs_start}, {abs_end}"
                    site_label = 'acceptor' if is_true_splice_site else 'decoy acceptor'

                    if is_true_splice_site:
                        labeled_sites.append({
                            'chrom': chrom,
                            'start': abs_start,
                            'end': abs_end,
                            'strand': strand,
                            'site_type': site_label,
                            'gene_id': gene_id,
                            'transcript_id': transcript_id
                        })
                        counters['acceptor'] += 1

                        print(f"... acceptor pattern={pattern}: {abs_start}-{abs_end} (near {splice_site_pos}?)")
                    else:
                        labeled_sites.append({
                            'chrom': chrom,
                            'start': abs_start,
                            'end': abs_end,
                            'strand': strand,
                            'site_type': site_label,
                            'gene_id': gene_id,
                            'transcript_id': transcript_id
                        })
                        counters['decoy acceptor'] += 1

                        if abs(splice_site_pos-abs_end) < 10: 
                            print(f"... mismatched pattern: {pattern} => L={site_label} | {abs_start}-{abs_end} (near {splice_site_pos}?)")
                            rel_start = site_start - gene_start
                            rel_end = site_end - gene_start
                            acceptor_seq = gene_sequence[rel_start:rel_end + 1]
                            print(f"... relxed acceptor site window: {acceptor_seq}")

            unique_splice_sites.add((site_start, site_end, site_type))

        print("[info] Found splice sites for n={} transcripts:\n{}\n".format(len(tx_ids), tx_ids))

    if save_to_file and file_path:
        df = pd.DataFrame(labeled_sites)

        print("[i/o] Saving labeled sites to file:\n{}\n".format(file_path))
        df.to_csv(file_path, sep='\t' if file_path.endswith('.tsv') else ',', index=False)

    if return_df:
        return pd.DataFrame(labeled_sites), counters

    return labeled_sites, counters


def label_splice_sites(gene_sequences_df, consensus_patterns):
    """
    Label true and decoy splice sites in gene sequences using consensus sequences.

    Parameters:
    - gene_sequences_df (pl.DataFrame): DataFrame containing gene sequences with positions and strand information.
                                        Columns: ['gene_id', 'seqname', 'gene_name', 'strand', 'start', 'end', 'sequence']
    - consensus_patterns (dict): Dictionary with 'donor' and 'acceptor' consensus patterns (regex).
    
    Returns:
    - results (list of dicts): List of dictionaries with labeled positions and consensus matches.
    """
    results = []

    # Iterate over each row in the gene sequence dataframe
    for row in gene_sequences_df.iter_rows(named=True):
        gene_id = row['gene_id']
        seqname = row['seqname']
        strand = row['strand']
        gene_start = row['start']
        gene_end = row['end']
        sequence = row['sequence']

        print(f"Processing gene: {gene_id} on {strand} strand")

        # Search for donor and acceptor sites in the sequence
        donor_sites = find_consensus_sites(sequence, consensus_patterns['donor'])
        acceptor_sites = find_consensus_sites(sequence, consensus_patterns['acceptor'])

        # Convert relative positions to absolute positions for donor and acceptor sites
        donor_abs_positions = convert_relative_to_absolute_positions(gene_start, gene_end, donor_sites, strand)
        acceptor_abs_positions = convert_relative_to_absolute_positions(gene_start, gene_end, acceptor_sites, strand)

        # Store the results
        results.append({
            'gene_id': gene_id,
            'seqname': seqname,
            'strand': strand,
            'donor_sites': donor_abs_positions,
            'acceptor_sites': acceptor_abs_positions
        })

    return results


def demo_find_consensus_sites(): 

    # Example DataFrame setup
    # gene_id, sequence (DNA sequence with proper orientation)
    data = {'gene_id': ['gene_1', 'gene_2'],
            'sequence': ['ATGGTGATCAGT...ATCAGT', 'CGTACGGTAGC...TGACGT']}
    df = pd.DataFrame(data)

    # Apply search for GT (donor) and AG (acceptor) consensus sequences
    df['GT_donor_sites'] = df['sequence'].apply(lambda seq: find_consensus_sites(seq, r'GT'))
    df['AG_acceptor_sites'] = df['sequence'].apply(lambda seq: find_consensus_sites(seq, r'AG'))

    # Example: show the updated DataFrame
    print(df[['gene_id', 'GT_donor_sites', 'AG_acceptor_sites']])


def demo_relative_to_absolute_positions():
    # Example gene annotation for STMN2 and UNC13A
    gene_annotations = {
        'STMN2': {'start': 85000, 'end': None, 'strand': '+'},  # positive strand
        'UNC13A': {'start': 90000, 'end': 100000, 'strand': '-'}  # negative strand
    }

    # Function to calculate absolute positions
    def calculate_absolute_position(gene_id, relative_position, gene_annotations):
        gene_data = gene_annotations[gene_id]
        strand = gene_data['strand']
        
        if strand == '+':
            # Positive strand: Start position + relative position - 1
            return gene_data['start'] + relative_position - 1
        elif strand == '-':
            # Negative strand: End position - relative position + 1
            return gene_data['end'] - relative_position + 1
        else:
            raise ValueError("Strand information is invalid")

    # Example relative positions from regex match
    relative_position_stmn2 = 15  # donor site at index 15 in STMN2 sequence
    relative_position_unc13a = 25  # donor site at index 25 in UNC13A sequence

    # Calculate absolute positions
    absolute_position_stmn2 = calculate_absolute_position('STMN2', relative_position_stmn2, gene_annotations)
    absolute_position_unc13a = calculate_absolute_position('UNC13A', relative_position_unc13a, gene_annotations)

    print(f"Absolute position for STMN2 (donor site): {absolute_position_stmn2}")
    print(f"Absolute position for UNC13A (donor site): {absolute_position_unc13a}")

    return


def demo_generate_label_arrays(): 

    # Mock annotations DataFrame
    annotations_data = {
        'chrom': ['chr1', 'chr1', 'chr1', 'chr2', 'chr2'],
        'start': [2581648, 2583368, 2583493, 3069294, 3069300],
        'end': [2581651, 2583371, 2583496, 3069297, 3069303],
        'strand': ['+', '+', '+', '+', '-'],
        'site_type': ['donor', 'acceptor', 'donor', 'donor', 'acceptor'],
        'gene_id': ['ENSG00000228037', 'ENSG00000228037', 'ENSG00000228037', 'ENSG00000142611', 'ENSG00000142611'],
        'transcript_id': ['ENST00000424215', 'ENST00000424215', 'ENST00000424215', 'ENST00000511072', 'ENST00000511072']
    }
    annotations_df = pl.DataFrame(annotations_data)

    # Mock gene sequences DataFrame
    gene_sequences_data = {
        'gene_id': ['ENSG00000228037', 'ENSG00000142611'],
        'seqname': ['chr1', 'chr2'],
        'gene_name': ['Gene1', 'Gene2'],
        'start': [2581600, 3069000],
        'end': [2584000, 3070000],
        'strand': ['+', '+'],
        'sequence': ['A' * 3000, 'T' * 1000]
    }
    gene_sequences_df = pl.DataFrame(gene_sequences_data)

    # Validate the function with mock data
    labels_dict = generate_label_arrays(annotations_df, gene_sequences_df)

    # Print the positions of 1s in the donor and acceptor labels
    for gene_id, labels in labels_dict.items():
        donor_positions = [i for i, label in enumerate(labels['donor_labels']) if label == 1]
        acceptor_positions = [i for i, label in enumerate(labels['acceptor_labels']) if label == 1]
        print(f"Gene ID: {gene_id}")
        print(f"Donor positions: {donor_positions}")
        print(f"Acceptor positions: {acceptor_positions}")
        print()

    return


def get_gene_attributes(annot_df, gene_id):
    """
    Given a gene ID, find its attributes (such as chromosome) from the annotation DataFrame.
    
    Parameters:
    - annot_df (pd.DataFrame): The annotation DataFrame containing gene information.
    - gene_id (str): The gene ID to look up.
    
    Returns:
    - dict: A dictionary containing the gene attributes if found, otherwise None.
    """
    # Look up the row for the gene of interest
    gene_row = annot_df.loc[annot_df['gene_id'] == gene_id]

    # Check if the gene_id was found
    if not gene_row.empty:
        # Convert the row to a dictionary
        gene_attributes = gene_row.iloc[0].to_dict()
        return gene_attributes
    else:
        print(f"Gene ID {gene_id} not found in the annotation DataFrame.")
        return None


def demo_generate_label_arrays_given_genes():
    from meta_spliceai.splice_engine.utils_bio import load_chromosome_sequence
    from meta_spliceai.splice_engine.extract_genomic_features import (
        read_splice_sites, 
        extract_splice_sites_workflow, 
        transcript_sequence_retrieval_workflow,
        gene_sequence_retrieval_workflow, 
    )
    from meta_spliceai.splice_engine.utils import (
        pandas_to_polars, 
        polars_to_pandas
    )

    src_dir = "/path/to/meta-spliceai/data/ensembl/"
    local_dir = data_dir = "/path/to/meta-spliceai/data/ensembl/"
    gtf_file = "/path/to/meta-spliceai/data/ensembl/Homo_sapiens.GRCh38.112.gtf"
    genome_fasta = os.path.join(src_dir, "Homo_sapiens.GRCh38.dna.primary_assembly.fa") 

    # exon, CDS, 5'UTR, and 3'UTR annotations for all transcripts
    annotation_file_path = os.path.join(data_dir, "annotations_all_transcripts.csv")
    print("[info] Transcipt Annotations (exon, CDS, 5'UTR, and 3'UTR annotations):")
    annot_df = pd.read_csv(annotation_file_path, sep=',')
    print(annot_df.head())
    # NOTE: By default, pd.read_csv()) reads the first row of the CSV file as the header; i.e. header=0 by default

    # Verify exon boudaries for all transcripts here 
    # <...> 

    use_mock = False
    extract_splice_sites = False
    gene_id_of_interest = 'ENSG00000104435'  # 'ENSG00000130477' (UNC13A), 'ENSG00000104435' (STMN2)
    
    # Look up the chromosome number for the gene of interest
    attributes = get_gene_attributes(annot_df, gene_id_of_interest)
    chromosome = attributes['chrom']
    start, end = attributes['start'], attributes['end']
    strand = attributes['strand']
    print(f"[info] Gene={gene_id_of_interest}: chr{chromosome}:{start}-{end}")  # start and end are a "min-max" gene boundary

    consensus_window = 3  # 3? if we use CAG, TAG, AAG for acceptor sites

    if use_mock: 
        # Mock annotations DataFrame for STMN2 (with consensus_window=2)
        annotations_data = {
            'chrom': ['8', '8', '8', '8', '8', '8', '8', '8'],
            'start': [79611212, 79636800, 79636895, 79641376, 79641548, 79654869, 79655060, 79664813],
            'end': [79611215, 79636803, 79636898, 79641379, 79641551, 79654872, 79655063, 79664816],
            'strand': ['+', '+', '+', '+', '+', '+', '+', '+'],
            'site_type': ['donor', 'acceptor', 'donor', 'acceptor', 'donor', 'acceptor', 'donor', 'acceptor'],
            'gene_id': ['ENSG00000104435'] * 8,
            'transcript_id': ['ENST00000220876'] * 8
        }
        annotations_df = pl.DataFrame(annotations_data)
    else:
        if extract_splice_sites: 
            extract_splice_sites_workflow(data_prefix=data_dir, gtf_file=gtf_file, consensus_window=consensus_window)

        print("[info] Loading spilce site annotations from file...")
        # Splice sites for all transcripts derived from the genome annotations (annot_df)
        splice_sites_file_path = os.path.join(data_dir, "splice_sites.tsv")
        
        # Pandas 
        # annotations_df = pd.read_csv(splice_sites_file_path, sep='\t')
        # Subset the DataFrame for the gene_id of interest
        # annotations_df = annotations_df[annotations_df['gene_id'] == gene_id_of_interest]

        # Polars
        # Read the DataFrame using Polars
        # annotations_df = pl.read_csv(splice_sites_file_path, separator='\t')
        annotations_df = read_splice_sites(splice_sites_file_path, verbose=1)

        print("[info] Splice Site Annotations prior to filtering by genes:")
        print(annotations_df.head())

        # Subset the DataFrame for the gene_id of interest
        annotations_df = annotations_df.filter(pl.col('gene_id') == gene_id_of_interest)

    print("[info] Splice Site Annotations:")
    print(annotations_df.head())

    # -----------------------------------------------------------
    # Generate sequence dataframe 
    run_sequence_generation = False

    seq_type = 'full'
    format = 'parquet'
    output_file = f"gene_sequence.{format}" 
    if seq_type == 'minmax':
        output_file = f"gene_sequence_minmax.{format}"
    seq_df_path = os.path.join(local_dir, output_file)

    if run_sequence_generation:
        gene_sequence_retrieval_workflow(gtf_file, genome_fasta, output_file=seq_df_path, mode=seq_type)

    if use_mock: 
        # Mock gene sequence DataFrame for STMN2 (ENSG00000104435)
        gene_sequences_data = {
            'gene_id': ['ENSG00000104435'],
            'seqname': ['8'],
            'gene_name': ['STMN2'],
            'start': [79611117],  # Start position of the gene in the genome
            'end': [79666158],    # End position of the gene in the genome
            'strand': ['+'],
            'sequence': ['A' * 3000 + 'G' * 3000 + 'T' * 3000 + 'C' * 3000]  # A simple gene sequence (12,000 nucleotides)
        }
        gene_sequences_df = pl.DataFrame(gene_sequences_data)

    else: 
        print("[info] Loading gene sequence from file...")
        base_file_path = os.path.join(data_dir, "gene_sequence.parquet")
        seq_df = load_chromosome_sequence(base_file_path, chromosome=chromosome, format='parquet') 
        
        # Subset the sequence dataframe for the gene_id ENSG00000104435
        # gene_id_of_interest = 'ENSG00000104435'

        # Filter the sequence DataFrame for the gene of interest, seq_df is a pl.DataFrame
        gene_sequences_df = seq_df.filter(pl.col('gene_id') == gene_id_of_interest)

    # Print the subset to verify
    print("[info] Gene Sequences:")
    print(gene_sequences_df)
    # columns: ['gene_name', 'seqname', 'gene_id', 'strand', 'start', 'end', 'sequence']

    # Assuming gene_sequences_df is the dataframe with the columns mentioned
    min_start_position = gene_sequences_df['start'].min()
    print(f"[info] Minimum start position: {min_start_position}")

    # Determine if the annotation is 1-based or 0-based
    if min_start_position == 0:
        print("The annotation is 0-based.")
    else:
        print("The annotation is 1-based.")

    # -----------------------------------------------------------

    # Add assertions to check if the inputs are Polars DataFrames
    assert isinstance(annotations_df, pl.DataFrame), "annotations_df must be a Polars DataFrame"
    assert isinstance(gene_sequences_df, pl.DataFrame), "gene_sequences_df must be a Polars DataFrame"

    method = "true_vs_decoy"  # "true_vs_decoy",  "all_nucleotides"
    consensus_window = 3  # Need 3 to capture trinucleotide acceptor sites

    if method.startswith("all"):
        # Generate the label arrays
        labels_dict = generate_label_arrays(annotations_df, gene_sequences_df)

        # Get the donor and acceptor label arrays for STMN2 (ENSG00000104435)
        stmn2_labels = labels_dict['ENSG00000104435']
        donor_labels = stmn2_labels['donor_labels']
        acceptor_labels = stmn2_labels['acceptor_labels']

        # Output the donor and acceptor positions with 1s
        donor_positions = [i for i, label in enumerate(donor_labels) if label == 1]
        acceptor_positions = [i for i, label in enumerate(acceptor_labels) if label == 1]

        # Print results
        print(f"Donor positions in sequence: {donor_positions}")
        print(f"Acceptor positions in sequence: {acceptor_positions}")
    else: 
        label_file_path = os.path.join(data_dir, "labeled_splice_sites.tsv")

        labeled_sites, counters = \
            label_splice_sites_with_decoys(
                annotations_df, gene_sequences_df, 
                donor_consensus=('GT', 'GC'), 
                acceptor_consensus=('CAG', 'TAG', 'AAG'),  # ('CAG', 'TAG', 'AAG'), ('AG', )
                consensus_window=consensus_window, 
                save_to_file=True,
                file_path=label_file_path,
                return_df=False)

        # Debugging: Print final counter values
        print("\nFinal counters:")
        print(f"True donor sites: {counters['donor']}")
        print(f"Decoy donor sites: {counters['decoy donor']}")
        print(f"True acceptor sites: {counters['acceptor']}")
        print(f"Decoy acceptor sites: {counters['decoy acceptor']}")


    return


def test(): 

    # Call the gene-specific demo function
    demo_generate_label_arrays_given_genes()

    return



if __name__ == "__main__": 
    test()