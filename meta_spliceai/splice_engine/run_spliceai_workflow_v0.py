import os, sys
import argparse
import requests
import numpy as np
import pandas as pd
import polars as pl  # pip install polars
from tqdm import tqdm
import psutil  # pip install psutil
import time, random
from datetime import timedelta


from keras.models import load_model
from pkg_resources import resource_filename
from spliceai.utils import one_hot_encode
import spliceai  # pip install spliceai

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns

from collections import defaultdict
import logging
import warnings

from meta_spliceai.splice_engine.utils import (
    calculate_duration, 
    format_time, 
    setup_logger, 
)

from meta_spliceai.splice_engine.model_evaluator import ModelEvaluationFileHandler
from meta_spliceai.splice_engine.utils_fs import (
    read_splice_sites, 
    demo_extract_sequences
)
from meta_spliceai.splice_engine.utils_df import (
    is_dataframe_empty, 
    subsample_dataframe
)

from meta_spliceai.splice_engine.extract_genomic_features import (
    extract_splice_sites_workflow, 
    transcript_sequence_retrieval_workflow,
    gene_sequence_retrieval_workflow, 
    get_overlapping_gene_metadata,
    load_overlapping_gene_metadata
)

from meta_spliceai.splice_engine.utils_bio import (
    load_sequences, 
    load_chromosome_sequence,
    load_sequences_by_chromosome, 
    load_chromosome_sequence_streaming
)

from meta_spliceai.splice_engine.evaluate_models import (
    evaluate_donor_site_predictions, 
    evaluate_acceptor_site_predictions, 
    evaluate_splice_site_predictions, 
    evaluate_splice_site_errors,
    save_performance_df, 
    save_error_analysis,
)

from meta_spliceai.splice_engine.performance_analyzer import (
    PerformanceAnalyzer, 
    sort_performance_by_metric_and_error_type, 
    # analyze_hard_genes
    # plot_single_roc_curve, 
    # plot_single_pr_curve,
    # plot_cv_roc_curve,
    # plot_cv_pr_curve,
)
    

from meta_spliceai.splice_engine.utils_doc import (
    print_emphasized,
    print_with_indent, 
    print_section_separator, 
    display,
    display_dataframe_in_chunks
)

# Set environment variables to control TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging (0 = all logs, 1 = INFO, 2 = WARNING, 3 = ERROR)
# NOTE: This environment variable setting suppresses all TensorFlow logs except for errors.
# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations to avoid related warnings

# Optionally, suppress other warnings
logging.getLogger('tensorflow').setLevel(logging.ERROR)
# This sets the logging level for TensorFlow to ERROR, further ensuring that only error messages are shown.

# Suppress specific warnings using the warnings module
# warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')
# warnings.filterwarnings('ignore', category=FutureWarning, module='tensorflow')

# Set the environment variables and logging configurations before importing TensorFlow.
import tensorflow as tf  # pip install tensorflow

from pkg_resources import resource_filename
from spliceai.utils import one_hot_encode
from meta_spliceai.utils.bio_utils import (
    demo_extract_gene_sequences, 
    demo_extract_transcript_sequences
)

def demo_basics_v0(): 

    # Load the pretrained model
    model = spliceai.SpliceAI()

    # Example: Predict splice sites for a given sequence
    sequence = "ATGCTAGCTAGCTAGCTAGCTGACTGACGTAGCTAGCTAGCATCG"
    predictions = model.predict(sequence)

    # The output will be a probability distribution for each nucleotide
    for i, (donor_prob, acceptor_prob) in enumerate(predictions):
        print(f"Nucleotide {i}: Donor={donor_prob}, Acceptor={acceptor_prob}")


def where_is_spliceai():
    import site

    # List all site-packages directories
    site_packages_dirs = site.getsitepackages()

    # Check each directory for the spliceai package
    for dir in site_packages_dirs:
        spliceai_path = os.path.join(dir, 'spliceai')
        if os.path.exists(spliceai_path):
            print(f"spliceai package found at: {spliceai_path}")
            break
    else:
        print("spliceai package not found")


def prepare_input_sequence(sequence, context):
    """
    Prepares an input DNA sequence for prediction using SpliceAI models by converting it into one-hot encoded format,
    adding flanking sequences, padding, and splitting it into overlapping blocks of the required size.

    Parameters:
    - sequence (str): The input DNA sequence (string of A, C, G, T) to be processed.
    - context (int): The context length for SpliceAI models. This defines the amount of surrounding sequence
                     (flanking) provided on both sides of each 5,000-nucleotide chunk. Valid options are typically
                     80, 400, 2000, or 10000, corresponding to SpliceAI-80nt, SpliceAI-400nt, SpliceAI-2k, and SpliceAI-10k models.

    Returns:
    - np.ndarray: A NumPy array of overlapping blocks, where each block is a one-hot encoded representation of a segment
                  of the input sequence, with flanking regions. The shape of the array is (num_blocks, block_size, 4),
                  where `block_size` = `context / 2 + 5000 + context / 2` and `num_blocks` is the number of overlapping
                  blocks generated.

    Example usage:
        sequence = "AGTCTTCTCTCTCGCTCTCTCCGCTGCTGTAGCCGGACCCTTTGCC..."
        context = 10000
        blocks = prepare_input_sequence(sequence, context)
    """

    from spliceai.utils import one_hot_encode
    import numpy as np

    # One-hot encode the input sequence
    encoded_seq = one_hot_encode(sequence)
    # NOTE: Let's assume the length of the encoded sequence is 12,345 nucleotides.
    #       The one-hot encoded sequence encoded_seq will have a shape of (12,345, 4)
    #       => pad_length = 5000 - (12345 % 5000)  # pad_length = 5000 - 2345 = 2655
    #          
    #        encoded_seq = np.pad(encoded_seq, ((0, 2655), (0, 0)), 'constant'), where
    #           ((0, 2655), (0, 0)) means:
    #           - Add 2,655 rows of padding at the end (no padding at the beginning).
    #           - No padding is added to the columns.
    #       => (encoded_seq.shape) = (12345 + 2655, 4) = (15000, 4)

    # Pad the sequence to make the length a multiple of 5,000 nucleotides
    seq_length = len(encoded_seq)
    if seq_length % 5000 != 0:
        pad_length = 5000 - (seq_length % 5000)  # number of nucleotides needed to make the sequence length a multiple of 5,000
        encoded_seq = np.pad(encoded_seq, ((0, pad_length), (0, 0)), 'constant')
    
    # Add flanking sequences of length S on both sides, where S = context / 2
    flanking = np.zeros((context // 2, 4))  # 'N' padding represented as zeros in one-hot encoding
    padded_seq = np.vstack([flanking, encoded_seq, flanking])
    
    # Split into overlapping blocks of length context / 2 + 5000 + context / 2
    block_size = context // 2 + 5000 + context // 2
    num_blocks = (len(padded_seq) - block_size) // 5000 + 1
    blocks = np.array([padded_seq[5000 * i: 5000 * i + block_size] for i in range(num_blocks)])
    
    return blocks


def predict_splice_sites_for_genes_with_pandas(gene_df, models, context=10000):
    """
    Generate splice site predictions for each gene sequence in gene_df using SpliceAI models.

    Parameters:
    - gene_df (pd.DataFrame): DataFrame with columns 'gene_id', 'gene_name', 'seqname', 'start', 'end', 'strand', 'sequence'.
    - models (list): List of loaded SpliceAI models.
    - context (int): Context length for SpliceAI (default 10000).

    Returns:
    - pd.DataFrame: DataFrame with columns 'gene_id', 'position', 'absolute_position', 'donor_prob', 'acceptor_prob', 'neither_prob'.
    """
    
    # Dictionary to store merged results by nucleotide positions
    merged_results = defaultdict(lambda: {'donor_prob': [], 'acceptor_prob': [], 'neither_prob': []})

    for idx, row in tqdm(gene_df.iterrows(), total=gene_df.shape[0], desc="Processing gene sequences"):
        gene_id = row['gene_id']
        gene_name = str(row['gene_name']) if row['gene_name'] is not None else ""  # Ensure gene_name is always a string
        sequence = row['sequence']
        seqname = row.get('seqname', row.get('chrom'))  # chromosome or sequence name
        strand = row['strand']
        seq_len = len(sequence)

        print_emphasized(f"(predict_splice_sites_gene) Processing gene (at chr={seqname}):", gene_id)
        print_with_indent(f"... Sequence Length: {seq_len}", indent_level=1)

        # Check if 'start' and 'end' positions are available for absolute positioning
        has_absolute_positions = 'start' in row and 'end' in row
        gene_start = row.get('start', None)  # Absolute start position if available
        gene_end = row.get('end', None)  # Absolute end position if available

        # Prepare input sequence using SpliceAI preprocessing logic
        input_blocks = prepare_input_sequence(sequence, context)

        print(f"[debug] Generated {len(input_blocks)} blocks for gene {gene_id}.")
        
        # Predict splice sites using SpliceAI models for each block
        for block_index, block in enumerate(input_blocks):
            x = block[None, :]  # Add batch dimension for model input

            # Predict splice sites using SpliceAI models
            y = np.mean([model.predict(x) for model in models], axis=0)
            # NOTE: y has the shape (1, block_size, 3), where 
            #       block_size is context / 2 + 5000 + context / 2, 
            #       and 3 represents the three classes (neither, acceptor, donor)
            print(f"[debug] Data shape for block {block_index}")
            print("... shape(x): ", x.shape)
            print("... shape(y): ", y.shape)

            # Extract probabilities for each nucleotide position in the block
            donor_prob = y[0, :, 2]
            acceptor_prob = y[0, :, 1]
            neither_prob = y[0, :, 0]

            # Check if predictions are generated correctly
            print(f"[debug] Extracted {len(donor_prob)} probabilities for block {block_index}.")

            # Calculate the start position of the current block relative to the original sequence
            block_start = block_index * 5000

            # Store the results with adjusted positions
            for i, (donor_p, acceptor_p, neither_p) in enumerate(zip(donor_prob, acceptor_prob, neither_prob)):
                
                if has_absolute_positions:
                    # Adjust absolute position based on the strand
                    if strand == '+':
                        absolute_position = gene_start + (block_start + i)
                    elif strand == '-':
                        # Decrement from gene_end for the negative strand
                        absolute_position = gene_end - (block_start + i)
                    else: 
                        raise ValueError(f"Invalid strand: {strand}")
                else:
                    absolute_position = None
                
                pos_key = (gene_id, absolute_position) if has_absolute_positions else (gene_id, block_start + i + 1)

                # Append probabilities for overlapping positions
                merged_results[pos_key]['donor_prob'].append(donor_p)
                merged_results[pos_key]['acceptor_prob'].append(acceptor_p)
                merged_results[pos_key]['neither_prob'].append(neither_p)
                merged_results[pos_key]['strand'] = strand
                merged_results[pos_key]['seqname'] = seqname
                merged_results[pos_key]['gene_name'] = gene_name
                merged_results[pos_key]['absolute_position'] = absolute_position if has_absolute_positions else (block_start + i + 1)

    assert len(merged_results) > 0, "No splice site predictions generated."

    # Trim unwanted probabilities:
    # Remove any positions corresponding to flanking sequences or padding
    trimmed_results = {}

    # Create a dictionary to map gene_id to its start and end positions
    gene_positions = {row['gene_id']: (row['start'], row['end']) for _, row in gene_df.iterrows()}

    for (gene_id, position), data in merged_results.items():
        # Retrieve the start and end positions for the current gene_id
        gene_start, gene_end = gene_positions[gene_id]

        # Ensure absolute positions are within the actual sequence range
        if data['absolute_position'] is not None and (data['absolute_position'] >= gene_start and data['absolute_position'] <= gene_end):
            trimmed_results[(gene_id, position)] = data

    # Consolidate results by averaging overlapping predictions
    results = []
    for (gene_id, position), data in trimmed_results.items():
        results.append({
            'seqname': data['seqname'],
            'gene_id': gene_id,
            'gene_name': data['gene_name'],
            'position': position,
            'absolute_position': data['absolute_position'],
            'donor_prob': np.mean(data['donor_prob']),
            'acceptor_prob': np.mean(data['acceptor_prob']),
            'neither_prob': np.mean(data['neither_prob']),
            'strand': data['strand']
        })
    # NOTE: Due to the overlapping nature of the input blocks, the same nucleotide position may appear 
    #       in multiple blocks, leading to multiple predictions for that position.

    # Convert results to DataFrame
    results_df = pl.DataFrame(results)

    # Sanity check: Ensure output probabilities match the input sequence lengths
    for gene_id in gene_df['gene_id'].unique():
        input_length = len(gene_df[gene_df['gene_id'] == gene_id]['sequence'].values[0])
        output_length = len(results_df[results_df['gene_id'] == gene_id])
        assert input_length == output_length, f"Length mismatch for {gene_id}: input {input_length}, output {output_length}"

    return results_df


def predict_splice_sites_for_genes(gene_df, models, context=10000, efficient_output=True, **kargs):
    """
    Generate splice site predictions for each gene sequence in gene_df using SpliceAI models.

    See predict_splice_sites_for_transcripts() for transcript-level splice-site prediction

    Parameters:
    - gene_df (pl.DataFrame): DataFrame with columns 'gene_id', 'gene_name', 'seqname', 'start', 'end', 'strand', 'sequence'.
    - models (list): List of loaded SpliceAI models.
    - context (int): Context length for SpliceAI (default 10000).
    - efficient_output (bool): If True, returns efficient_results dictionary; otherwise, returns full prediction DataFrame.

    Returns:
    - efficient_results (dict): If efficient_output is True, returns a dictionary containing splice site predictions for each gene.
    - results_df (pl.DataFrame): If efficient_output is False, returns a DataFrame with columns 'gene_id', 'position', 'absolute_position', 'donor_prob', 'acceptor_prob', 'neither_prob'.
    
    Memo: 
    - When efficient_output=False, it returns a dataframe in which
      each row in the final predictions_df represents one specific nucleotide in that gene’s sequence 
      (i.e., a single base). 
      If a gene is 10,000 nt long, you’ll see 10,000 rows for that gene, each row having 
      distinct positions and probabilities (donor_prob, acceptor_prob, neither_prob).
    """

    # Dictionary to store merged results by nucleotide positions
    merged_results = defaultdict(lambda: {'donor_prob': [], 'acceptor_prob': [], 'neither_prob': []})
    n_genes_processed = 0

    for row in tqdm(gene_df.iter_rows(named=True), total=gene_df.shape[0], desc="Processing gene sequences"):
        gene_id = row['gene_id']
        gene_name = str(row['gene_name']) if row['gene_name'] is not None else ""  # Ensure gene_name is always a string
        sequence = row['sequence']
        seqname = row.get('seqname', row.get('chrom'))  # chromosome or sequence name
        strand = row['strand']
        seq_len = len(sequence)

        print(f"(predict_splice_sites_gene) Processing gene {gene_id} (at chr={seqname}) ...")
        print("... Sequence Length:", seq_len)

        # Check if 'start' and 'end' positions are available for absolute positioning
        has_absolute_positions = 'start' in row and 'end' in row
        gene_start = row.get('start', None)  # Absolute start position if available
        gene_end = row.get('end', None)  # Absolute end position if available

        # Prepare input sequence using SpliceAI preprocessing logic
        input_blocks = prepare_input_sequence(sequence, context)
        # Each gene sequence is split into overlapping blocks of length context / 2 + 5000 + context / 2

        print(f"[debug] Generated {len(input_blocks)} blocks for gene {gene_id}.")
        
        # Predict splice sites using SpliceAI models for each block
        for block_index, block in enumerate(input_blocks):
            x = block[None, :]  # Add batch dimension for model input

            # Predict splice sites using SpliceAI models
            y = np.mean([model.predict(x) for model in models], axis=0)

            # Extract probabilities for each nucleotide position in the block
            donor_prob = y[0, :, 2]
            acceptor_prob = y[0, :, 1]
            neither_prob = y[0, :, 0]

            # Calculate the start position of the current block relative to the original sequence
            block_start = block_index * 5000

            # Store the results with adjusted positions
            for i, (donor_p, acceptor_p, neither_p) in enumerate(zip(donor_prob, acceptor_prob, neither_prob)):
                if has_absolute_positions:
                    # Adjust absolute position based on the strand
                    if strand == '+':
                        absolute_position = gene_start + (block_start + i)
                    elif strand == '-':
                        absolute_position = gene_end - (block_start + i)
                    else: 
                        raise ValueError(f"Invalid strand: {strand}")
                else:
                    absolute_position = None
                
                pos_key = (gene_id, absolute_position) if has_absolute_positions else (gene_id, block_start + i + 1)

                # Append probabilities for overlapping positions
                merged_results[pos_key]['donor_prob'].append(donor_p)
                merged_results[pos_key]['acceptor_prob'].append(acceptor_p)
                merged_results[pos_key]['neither_prob'].append(neither_p)
                merged_results[pos_key]['strand'] = strand
                merged_results[pos_key]['seqname'] = seqname
                merged_results[pos_key]['gene_name'] = gene_name
                merged_results[pos_key]['absolute_position'] = absolute_position if has_absolute_positions else (block_start + i + 1)

        # Time tracking for debugging
        if 'start_time' in kargs:
            if n_genes_processed % 10 == 0:
                print(f"[info] Processed n={n_genes_processed} genes.")
                print(f"[time] Elapsed time: {calculate_duration(kargs['start_time'])}")

        n_genes_processed += 1

    assert len(merged_results) > 0, "No splice site predictions generated."

    # Trim unwanted probabilities:
    trimmed_results = {}

    # Create a dictionary to map gene_id to its start and end positions
    gene_positions = {row['gene_id']: (row['start'], row['end']) for row in gene_df.iter_rows(named=True)}

    for (gene_id, position), data in merged_results.items():
        # Retrieve the start and end positions for the current gene_id
        gene_start, gene_end = gene_positions[gene_id]

        # Ensure absolute positions are within the actual sequence range
        if data['absolute_position'] is not None and (data['absolute_position'] >= gene_start and data['absolute_position'] <= gene_end):
            trimmed_results[(gene_id, position)] = data

    ### Output Handling ### 
    output_format = kargs.get('output_format', 'efficient')
    if output_format.startswith( ('eff', 'dict') ):
        efficient_output = True
    else: 
        efficient_output = False

    # Convert results to efficient output format if requested
    if efficient_output:
        efficient_results = defaultdict(lambda: {
            'seqname': "",
            'gene_name': "",
            'strand': "",
            'gene_start': None,
            'gene_end': None,
            'donor_prob': [],
            'acceptor_prob': [],
            'neither_prob': [],
            'positions': []
        })

        for (gene_id, position), data in trimmed_results.items():
            if gene_id not in efficient_results:
                # Initialize basic information about the gene
                efficient_results[gene_id]['seqname'] = data['seqname']
                efficient_results[gene_id]['gene_name'] = data['gene_name']
                efficient_results[gene_id]['strand'] = data['strand']
                efficient_results[gene_id]['gene_start'], efficient_results[gene_id]['gene_end'] = gene_positions[gene_id]

            # Store per-position data
            efficient_results[gene_id]['donor_prob'].append(np.mean(data['donor_prob']))
            efficient_results[gene_id]['acceptor_prob'].append(np.mean(data['acceptor_prob']))
            efficient_results[gene_id]['neither_prob'].append(np.mean(data['neither_prob']))
            efficient_results[gene_id]['positions'].append(data['absolute_position'])

        # Sanity check to ensure consistency between efficient_results and trimmed_results
        for gene_id in efficient_results.keys():
            assert len(efficient_results[gene_id]['donor_prob']) == len(efficient_results[gene_id]['positions']), \
                f"Mismatch in efficient_results for gene {gene_id}: donor_prob length doesn't match positions length"
            assert len(efficient_results[gene_id]['acceptor_prob']) == len(efficient_results[gene_id]['positions']), \
                f"Mismatch in efficient_results for gene {gene_id}: acceptor_prob length doesn't match positions length"

        return efficient_results

    # Consolidate results by averaging overlapping predictions
    results = []
    for (gene_id, position), data in trimmed_results.items():
        gene_start, gene_end = gene_positions[gene_id]
        results.append({
            'seqname': data['seqname'],
            'gene_id': gene_id,
            'gene_name': data['gene_name'],
            'position': position,
            'absolute_position': data['absolute_position'],
            'gene_start': gene_start,
            'gene_end': gene_end,
            'donor_prob': np.mean(data['donor_prob']),
            'acceptor_prob': np.mean(data['acceptor_prob']),
            'neither_prob': np.mean(data['neither_prob']),
            'strand': data['strand']
        })

    # Convert results to Polars DataFrame
    results_df = pl.DataFrame(results)

    # Sanity check: Ensure output probabilities match the input sequence lengths
    for gene_id in gene_df.select('gene_id').unique().to_series().to_list():
        input_length = len(gene_df.filter(pl.col('gene_id') == gene_id)['sequence'].to_list()[0])
        output_length = results_df.filter(pl.col('gene_id') == gene_id).shape[0]
        assert input_length == output_length, f"Length mismatch for {gene_id}: input {input_length}, output {output_length}"

    return results_df


def predict_splice_sites_for_genes_v1(gene_df, models, context=10000, efficient_output=True):
    """
    Generate splice site predictions for each gene sequence in gene_df using SpliceAI models.

    Compared to v0, this version can generate outputs in both dataframe and more efficient data structure.
    
    Parameters:
    - gene_df (pl.DataFrame): DataFrame with columns 'gene_id', 'gene_name', 'seqname', 'start', 'end', 'strand', 'sequence'.
    - models (list): List of loaded SpliceAI models.
    - context (int): Context length for SpliceAI (default 10000).
    - efficient_output (bool): If True, returns a more memory-efficient data structure. If False, returns a DataFrame.
    
    Returns:
    - If efficient_output=True: Dictionary with gene-level information and nucleotide probability vectors.
    - If efficient_output=False: pl.DataFrame with columns 'gene_id', 'position', 'absolute_position', 'donor_prob', 'acceptor_prob', 'neither_prob'.
    
    Memo: 
    - Absolute Position:
        For absolute positions, the strand determines whether you count from the start or the end of the gene.
        Positive strand (+): Count from the start.
        Negative strand (-): Count from the end.
    - Relative Position:
        Relative positions are usually calculated directly from the sequence without considering the strand.
        The relative position is simply the position within the sequence block, adjusted for zero-based indexing.

    """
    # Dictionary to store trimmed results for both cases
    trimmed_results = defaultdict(lambda: {'donor_prob': [], 'acceptor_prob': [], 'neither_prob': []})

    # Store gene-level metadata and nucleotide-level probability vectors for efficient output
    efficient_results = {}

    for row in tqdm(gene_df.iter_rows(named=True), total=gene_df.shape[0], desc="Processing gene sequences"):
        gene_id = row['gene_id']
        gene_name = str(row['gene_name']) if row['gene_name'] is not None else ""  # Ensure gene_name is always a string
        sequence = row['sequence']
        seqname = row.get('seqname', row.get('chrom'))  # chromosome or sequence name
        strand = row['strand']
        seq_len = len(sequence)

        print("(predict_splice_sites_gene) Processing gene:", gene_id)
        print("... Sequence Length:", seq_len)

        # Initialize storage for efficient output
        if efficient_output:
            efficient_results[gene_id] = {
                'seqname': seqname,
                'gene_name': gene_name,
                'strand': strand,
                'positions': [],
                'donor_prob': defaultdict(list),
                'acceptor_prob': defaultdict(list),
                'neither_prob': defaultdict(list)
            }

        # Check if 'start' and 'end' positions are available for absolute positioning
        has_absolute_positions = 'start' in row and 'end' in row
        gene_start = row.get('start', None)  # Absolute start position if available
        gene_end = row.get('end', None)  # Absolute end position if available

        # Prepare input sequence using SpliceAI preprocessing logic
        input_blocks = prepare_input_sequence(sequence, context)

        print(f"[debug] Generated {len(input_blocks)} blocks for gene {gene_id}.")
        
        # Predict splice sites using SpliceAI models for each block
        for block_index, block in enumerate(input_blocks):
            x = block[None, :]  # Add batch dimension for model input

            # Predict splice sites using SpliceAI models
            y = np.mean([model.predict(x) for model in models], axis=0)
            print(f"[debug] Data shape for block {block_index}")
            print("... shape(x): ", x.shape)
            print("... shape(y): ", y.shape)

            # Extract probabilities for each nucleotide position in the block
            donor_prob = y[0, :, 2]
            acceptor_prob = y[0, :, 1]
            neither_prob = y[0, :, 0]

            print(f"[debug] Extracted {len(donor_prob)} probabilities for block {block_index}.")

            # Calculate the start position of the current block relative to the original sequence
            block_start = block_index * 5000

            # Store the results with adjusted positions
            for i, (donor_p, acceptor_p, neither_p) in enumerate(zip(donor_prob, acceptor_prob, neither_prob)):
                absolute_position = None
                if has_absolute_positions:
                    if strand == '+':
                        absolute_position = gene_start + (block_start + i)
                    elif strand == '-':
                        absolute_position = gene_end - (block_start + i)
                    else: 
                        raise ValueError(f"Invalid strand: {strand}")
                
                pos_key = (gene_id, absolute_position) if has_absolute_positions else (gene_id, block_start + i + 1)
                # IF absolute_position isn't available, use the relative position 
                # +1 to account for 0-based indexing in Python to match the 1-based indexing in the gene sequence

                # Trim probabilities outside the gene sequence
                if has_absolute_positions and (absolute_position < gene_start or absolute_position > gene_end):
                    continue  # Skip if the position is outside the actual gene sequence

                # Efficient output: Store results as vectors
                if efficient_output:
                    pos = pos_key[1]
                    efficient_results[gene_id]['positions'].append(pos)
                    efficient_results[gene_id]['donor_prob'][pos].append(donor_p)
                    efficient_results[gene_id]['acceptor_prob'][pos].append(acceptor_p)
                    efficient_results[gene_id]['neither_prob'][pos].append(neither_p)
                else:
                    # Append probabilities for overlapping positions (DataFrame output)
                    trimmed_results[pos_key]['donor_prob'].append(donor_p)
                    trimmed_results[pos_key]['acceptor_prob'].append(acceptor_p)
                    trimmed_results[pos_key]['neither_prob'].append(neither_p)
                    trimmed_results[pos_key]['strand'] = strand
                    trimmed_results[pos_key]['seqname'] = seqname
                    trimmed_results[pos_key]['gene_name'] = gene_name
                    trimmed_results[pos_key]['absolute_position'] = absolute_position if has_absolute_positions else (block_start + i + 1)

    if efficient_output:
        # Consolidate results by averaging the probabilities for overlapping positions
        for gene_id in efficient_results:
            for pos in efficient_results[gene_id]['donor_prob']:
                efficient_results[gene_id]['donor_prob'][pos] = \
                    np.mean(efficient_results[gene_id]['donor_prob'][pos])
            for pos in efficient_results[gene_id]['acceptor_prob']:
                efficient_results[gene_id]['acceptor_prob'][pos] = \
                    np.mean(efficient_results[gene_id]['acceptor_prob'][pos])
            for pos in efficient_results[gene_id]['neither_prob']:
                efficient_results[gene_id]['neither_prob'][pos] = \
                    np.mean(efficient_results[gene_id]['neither_prob'][pos])
    else:
        # Consolidate results by averaging overlapping predictions
        results = []
        for (gene_id, position), data in trimmed_results.items():
            results.append({
                'seqname': data['seqname'],
                'gene_id': gene_id,
                'gene_name': data['gene_name'],
                'position': position,
                'absolute_position': data['absolute_position'],
                'donor_prob': np.mean(data['donor_prob']),
                'acceptor_prob': np.mean(data['acceptor_prob']),
                'neither_prob': np.mean(data['neither_prob']),
                'strand': data['strand']
            })

        # Convert to Polars DataFrame
        results_df = pl.DataFrame(results)
        return results_df


def predict_splice_sites_for_genes_v2(gene_df, models, context=10000, efficient_output=True, **kargs):
    """
    Generate splice site predictions for each gene sequence in gene_df using SpliceAI models.
    """
    # from meta_spliceai.splice_engine.utils import calculate_duration

    # Store gene-level metadata and nucleotide-level probability vectors for efficient output
    efficient_results = {}
    n_genes_processed = 0

    for row in tqdm(gene_df.iter_rows(named=True), total=gene_df.shape[0], desc="Processing gene sequences"):
        gene_id = row['gene_id']
        gene_name = str(row['gene_name']) if row['gene_name'] is not None else ""  # Ensure gene_name is always a string
        gene_start = row['start']  # Absolute start position
        gene_end = row['end']  # Absolute end position
        sequence = row['sequence']
        seqname = row.get('seqname', row.get('chrom'))  # chromosome or sequence name
        strand = row['strand']
        seq_len = len(sequence)

        print("(predict_splice_sites_gene) Processing gene:", gene_id)
        print("... Sequence Length:", seq_len)
        assert gene_end - gene_start + 1 == seq_len, (
            f"Sequence length mismatch for gene {gene_id}: "
            f"gene_start={gene_start}, gene_end={gene_end}, "
            f"expected seq_len={seq_len}, calculated seq_len={gene_end - gene_start + 1}"
        )

        # Initialize storage for efficient output
        if efficient_output:
            efficient_results[gene_id] = {
                'seqname': seqname,
                'gene_name': gene_name,
                'strand': strand,
                'gene_start': gene_start,
                'gene_end': gene_end,
                'donor_prob': np.zeros(seq_len),
                'acceptor_prob': np.zeros(seq_len),
                'neither_prob': np.zeros(seq_len)
            }

        # Initialize an array to track how many times each position is predicted
        overlap_counts = np.zeros(seq_len)

        # Prepare input sequence using SpliceAI preprocessing logic
        input_blocks = prepare_input_sequence(sequence, context)

        print(f"[info] Generated {len(input_blocks)} blocks for gene {gene_id}.")

        # Predict splice sites using SpliceAI models for each block
        for block_index, block in enumerate(input_blocks):
            x = block[None, :]  # Add batch dimension for model input

            # Predict splice sites using SpliceAI models
            y = np.mean([model.predict(x) for model in models], axis=0)
            # print(f"[debug] Data shape for block {block_index}")  
            # print("... shape(x): ", x.shape)  # (1, 15000, 4) for SpliceAI-10k; last dimension is one-hot encoding
            # print("... shape(y): ", y.shape)  # (1, 5000, 3)

            # Extract probabilities for each nucleotide position in the block
            donor_prob = y[0, :, 2]
            acceptor_prob = y[0, :, 1]
            neither_prob = y[0, :, 0]

            # print(f"[debug] Extracted {len(donor_prob)} probabilities for block {block_index}.")
            # Should be 5000 

            # Calculate the start position of the current block relative to the original sequence
            block_start = block_index * 5000

            # Store the probabilities in the continuous vectors (averaging overlapping positions)
            for i, (donor_p, acceptor_p, neither_p) in enumerate(zip(donor_prob, acceptor_prob, neither_prob)):
                if block_start + i < seq_len:
                    # Increment overlap counts for averaging
                    overlap_counts[block_start + i] += 1

                    # Accumulate probabilities for overlapping regions
                    efficient_results[gene_id]['donor_prob'][block_start + i] += donor_p
                    efficient_results[gene_id]['acceptor_prob'][block_start + i] += acceptor_p
                    efficient_results[gene_id]['neither_prob'][block_start + i] += neither_p
        # End for each block loop

        # Normalize probabilities for overlapping regions
        overlap_counts = np.clip(overlap_counts, 1, None)  # Avoid division by zero
        efficient_results[gene_id]['donor_prob'] /= overlap_counts
        efficient_results[gene_id]['acceptor_prob'] /= overlap_counts
        efficient_results[gene_id]['neither_prob'] /= overlap_counts

        # Clip probabilities to ensure they're between 0 and 1
        efficient_results[gene_id]['donor_prob'] = np.clip(efficient_results[gene_id]['donor_prob'], 0, 1)
        efficient_results[gene_id]['acceptor_prob'] = np.clip(efficient_results[gene_id]['acceptor_prob'], 0, 1)
        efficient_results[gene_id]['neither_prob'] = np.clip(efficient_results[gene_id]['neither_prob'], 0, 1)

        # Sanity check: Ensure the sum of probabilities at each position is approximately 1
        for i in range(seq_len):
            prob_sum = (
                efficient_results[gene_id]['donor_prob'][i]
                + efficient_results[gene_id]['acceptor_prob'][i]
                + efficient_results[gene_id]['neither_prob'][i]
            )
            assert np.isclose(prob_sum, 1.0, atol=1e-3), f"Probability sum at position {i} for gene {gene_id} is {prob_sum}."

        # Sanity check: Ensure the output probability vectors match the input sequence length
        assert len(efficient_results[gene_id]['donor_prob']) == seq_len, \
            f"Mismatch in donor probabilities length for gene {gene_id}"
        assert len(efficient_results[gene_id]['acceptor_prob']) == seq_len, \
            f"Mismatch in acceptor probabilities length for gene {gene_id}"
        assert len(efficient_results[gene_id]['neither_prob']) == seq_len, \
            f"Mismatch in neither probabilities length for gene {gene_id}"

        n_genes_processed += 1

        # Time tracking for debugging
        if 'start_time' in kargs:
            if n_genes_processed % 10 == 0:
                print(f"[info] Processed n={n_genes_processed} genes.")
                print(f"[time] Elapsed time: {calculate_duration(kargs['start_time'])}")

    # End for each gene loop

    return efficient_results


def predict_splice_sites_for_genes_v3(gene_df, models, context=10000, output_format='dict', **kargs):
    """
    Generate splice site predictions for each gene sequence in gene_df using SpliceAI models.

    Parameters:
    - gene_df (pd.DataFrame or pl.DataFrame): DataFrame containing gene sequences.
    - models (list): List of SpliceAI models.
    - context (int): Context length for SpliceAI.
    - output_format (str): Desired output format ('dict', 'polars', 'pandas').

    Returns:
    - dict or DataFrame: Splice site predictions.
    """
    # Store gene-level metadata and nucleotide-level probability vectors for efficient output
    efficient_results = {}
    n_genes_processed = 0

    if isinstance(gene_df, pd.DataFrame):
        gene_df = pl.DataFrame(gene_df)
        # gene_df = pl.DataFrame._from_pandas(gene_df)

    for row in tqdm(gene_df.iter_rows(named=True), total=gene_df.shape[0], desc="Processing gene sequences"):
        gene_id = row['gene_id']
        gene_name = str(row['gene_name']) if row['gene_name'] is not None else ""  # Ensure gene_name is always a string
        gene_start = row['start']  # Absolute start position
        gene_end = row['end']  # Absolute end position
        sequence = row['sequence']
        seqname = row.get('seqname', row.get('chrom'))  # chromosome or sequence name
        strand = row['strand']
        seq_len = len(sequence)

        print("(predict_splice_sites_gene) Processing gene:", gene_id)
        print("... Sequence Length:", seq_len)
        # assert gene_end - gene_start + 1 == seq_len, (
        #     f"Sequence length mismatch for gene {gene_id}: "
        #     f"gene_start={gene_start}, gene_end={gene_end}, "
        #     f"expected seq_len={seq_len}, calculated seq_len={gene_end - gene_start + 1}"
        # )

        # Initialize storage for efficient output
        efficient_results[gene_id] = {
            'seqname': seqname,
            'gene_name': gene_name,
            'strand': strand,
            'gene_start': gene_start,
            'gene_end': gene_end,
            'donor_prob': np.zeros(seq_len),
            'acceptor_prob': np.zeros(seq_len),
            'neither_prob': np.zeros(seq_len),
            'positions': np.zeros(seq_len, dtype=int)
        }

        # Initialize an array to track how many times each position is predicted
        overlap_counts = np.zeros(seq_len)

        # Prepare input sequence using SpliceAI preprocessing logic
        input_blocks = prepare_input_sequence(sequence, context)

        print(f"[info] Generated {len(input_blocks)} blocks for gene {gene_id}.")

        # Predict splice sites using SpliceAI models for each block
        for block_index, block in enumerate(input_blocks):
            x = block[None, :]  # Add batch dimension for model input

            # Predict splice sites using SpliceAI models
            y = np.mean([model.predict(x) for model in models], axis=0)

            # Extract probabilities for each nucleotide position in the block
            donor_prob = y[0, :, 2]
            acceptor_prob = y[0, :, 1]
            neither_prob = y[0, :, 0]

            # Calculate the start position of the current block relative to the original sequence
            block_start = block_index * 5000

            # Store the probabilities in the continuous vectors (averaging overlapping positions)
            for i, (donor_p, acceptor_p, neither_p) in enumerate(zip(donor_prob, acceptor_prob, neither_prob)):
                if block_start + i < seq_len:
                    # Increment overlap counts for averaging
                    overlap_counts[block_start + i] += 1

                    # Accumulate probabilities for overlapping regions
                    efficient_results[gene_id]['donor_prob'][block_start + i] += donor_p
                    efficient_results[gene_id]['acceptor_prob'][block_start + i] += acceptor_p
                    efficient_results[gene_id]['neither_prob'][block_start + i] += neither_p

                    # Calculate and store the absolute position
                    if strand == '+':
                        absolute_position = gene_start + (block_start + i)
                    elif strand == '-':
                        absolute_position = gene_end - (block_start + i)
                    else:
                        raise ValueError(f"Invalid strand: {strand}")
                    efficient_results[gene_id]['positions'][block_start + i] = absolute_position
        # End for each block loop

        # Normalize probabilities for overlapping regions
        overlap_counts = np.clip(overlap_counts, 1, None)  # Avoid division by zero
        efficient_results[gene_id]['donor_prob'] /= overlap_counts
        efficient_results[gene_id]['acceptor_prob'] /= overlap_counts
        efficient_results[gene_id]['neither_prob'] /= overlap_counts

        # Clip probabilities to ensure they're between 0 and 1
        efficient_results[gene_id]['donor_prob'] = np.clip(efficient_results[gene_id]['donor_prob'], 0, 1)
        efficient_results[gene_id]['acceptor_prob'] = np.clip(efficient_results[gene_id]['acceptor_prob'], 0, 1)
        efficient_results[gene_id]['neither_prob'] = np.clip(efficient_results[gene_id]['neither_prob'], 0, 1)

        # Sanity check: Ensure the sum of probabilities at each position is approximately 1
        for i in range(seq_len):
            prob_sum = (
                efficient_results[gene_id]['donor_prob'][i]
                + efficient_results[gene_id]['acceptor_prob'][i]
                + efficient_results[gene_id]['neither_prob'][i]
            )
            assert np.isclose(prob_sum, 1.0, atol=1e-3), f"Probability sum at position {i} for gene {gene_id} is {prob_sum}."

        # Sanity check: Ensure the output probability vectors match the input sequence length
        assert len(efficient_results[gene_id]['donor_prob']) == seq_len, \
            f"Mismatch in donor probabilities length for gene {gene_id}"
        assert len(efficient_results[gene_id]['acceptor_prob']) == seq_len, \
            f"Mismatch in acceptor probabilities length for gene {gene_id}"
        assert len(efficient_results[gene_id]['neither_prob']) == seq_len, \
            f"Mismatch in neither probabilities length for gene {gene_id}"
        assert len(efficient_results[gene_id]['positions']) == seq_len, \
            f"Mismatch in positions length for gene {gene_id}"

        n_genes_processed += 1

        # Time tracking for debugging
        if 'start_time' in kargs:
            if n_genes_processed % 10 == 0:
                print(f"[info] Processed n={n_genes_processed} genes.")
                print(f"[time] Elapsed time: {calculate_duration(kargs['start_time'])}")

    # End for each gene loop

    # Convert results to the desired output format
    if output_format == 'dict':
        return efficient_results
    else:
        results = []
        for gene_id, data in efficient_results.items():
            for i in range(len(data['positions'])):
                results.append({
                    'seqname': data['seqname'],
                    'gene_id': gene_id,
                    'gene_name': data['gene_name'],
                    'strand': data['strand'],
                    'gene_start': data['gene_start'],
                    'gene_end': data['gene_end'],
                    'position': data['positions'][i],
                    'donor_prob': data['donor_prob'][i],
                    'acceptor_prob': data['acceptor_prob'][i],
                    'neither_prob': data['neither_prob'][i]
                })

        if output_format == 'polars':
            return pl.DataFrame(results)
        elif output_format == 'pandas':
            return pd.DataFrame(results)
        else:
            raise ValueError("Unsupported output format. Please choose 'dict', 'polars', or 'pandas'.")


def predict_splice_sites_for_transcripts(transcript_df, models, context=10000): 
    return predict_splice_sites(transcript_df, models, context=context)


def predict_splice_sites(transcript_df, models, context=10000, efficient_output=True, **kargs):
    """
    Generate splice site predictions for each transcript sequence in transcript_df using SpliceAI models.

    Parameters:
    - transcript_df (pl.DataFrame): DataFrame with columns 'transcript_id', 'sequence', 'seqname', 'strand', 'start', 'end'.
    - models (list): List of loaded SpliceAI models.
    - context (int): Context length for SpliceAI (default 10000).

    Returns:
    - pl.DataFrame: DataFrame with columns 'transcript_id', 'position', 'absolute_position', 'donor_prob', 'acceptor_prob', 'neither_prob'.
    """
    # Dictionary to store merged results by nucleotide positions
    merged_results = defaultdict(lambda: {'donor_prob': [], 'acceptor_prob': [], 'neither_prob': []})
    n_trpts_processed = 0

    for row in tqdm(transcript_df.iter_rows(named=True), total=transcript_df.shape[0], desc="Processing transcript sequences"):
        # Use iter_rows(named=True) for row-by-row iteration in Polars, maintaining named rows like you had in Pandas
        
        transcript_id = row['transcript_id']
        gene_id = row.get('gene_id', 'unknown')
        gene_name = row.get('gene_name', 'unknown')
        sequence = row['sequence']
        seqname = row.get('seqname', row.get('chrom'))  # chromosome or sequence name
        strand = row['strand']
        seq_len = len(sequence)

        print("(predict_splice_sites) Processing transcript:", transcript_id)
        print("... Sequence Length:", seq_len)

        # Check if 'start' and 'end' positions are available for absolute positioning
        has_absolute_positions = 'start' in row and 'end' in row
        transcript_start = row.get('start', None)  # Absolute start position if available
        transcript_end = row.get('end', None)  # Absolute end position if available

        # Prepare input sequence using SpliceAI preprocessing logic
        input_blocks = prepare_input_sequence(sequence, context)

        print(f"[debug] Generated {len(input_blocks)} blocks for transcript {transcript_id}.")
        
        # Predict splice sites using SpliceAI models for each block
        for block_index, block in enumerate(input_blocks):
            x = block[None, :]  # Add batch dimension for model input

            # Predict splice sites using SpliceAI models
            y = np.mean([model.predict(x) for model in models], axis=0)
            print(f"[debug] Data shape for block {block_index}")
            print("... shape(x): ", x.shape)
            print("... shape(y): ", y.shape)

            # Extract probabilities
            donor_prob = y[0, :, 2]
            acceptor_prob = y[0, :, 1]
            neither_prob = y[0, :, 0]
            # NOTE: The first dimension (index 0) represents the batch size. Since we are processing one block at a time, this dimension is 1.
            #       The second dimension (index 1) represents the nucleotide positions in the block.
            #       The third dimension (index 2) represents the three classes (neither, acceptor, donor)

            print(f"[debug] Extracted {len(donor_prob)} probabilities for block {block_index}.")

            # Calculate the start position of the current block relative to the original sequence
            block_start = block_index * 5000

            # Store the results with adjusted positions
            for i, (donor_p, acceptor_p, neither_p) in enumerate(zip(donor_prob, acceptor_prob, neither_prob)):
                if has_absolute_positions:
                    # Adjust absolute position based on the strand
                    if strand == '+':
                        absolute_position = transcript_start + (block_start + i)
                    elif strand == '-':
                        absolute_position = transcript_end - (block_start + i)
                        # NOTE: For both genes and transcripts, GTF/BED annotation typically has start < end, 
                        #       even on - strand. Thus, on the - strand, you do end - offset which yields decreasing coordinates 
                        #       => 3' to 5'
                    else: 
                        raise ValueError(f"Invalid strand: {strand}")
                else:
                    absolute_position = None

                pos_key = (transcript_id, absolute_position) if has_absolute_positions else (transcript_id, block_start + i + 1)

                # Append probabilities for overlapping positions
                merged_results[pos_key]['donor_prob'].append(donor_p)
                merged_results[pos_key]['acceptor_prob'].append(acceptor_p)
                merged_results[pos_key]['neither_prob'].append(neither_p)
                merged_results[pos_key]['strand'] = strand
                merged_results[pos_key]['seqname'] = seqname
                merged_results[pos_key]['gene_id'] = gene_id
                merged_results[pos_key]['gene_name'] = gene_name
                merged_results[pos_key]['absolute_position'] = absolute_position if has_absolute_positions else (block_start + i + 1)
    
        # Time tracking for debugging
        if 'start_time' in kargs:
            if n_trpts_processed % 10 == 0:
                print(f"[info] Processed n={n_trpts_processed} genes.")
                print(f"[time] Elapsed time: {calculate_duration(kargs['start_time'])}")

        n_trpts_processed += 1
    
    # End for row in transcript dataframe

    assert len(merged_results) > 0, "No splice site predictions generated."

    # Trim unwanted probabilities:
    trimmed_results = {}

    # Create a dictionary to map transcript_id to its start and end positions
    transcript_positions = {row['transcript_id']: (row['start'], row['end']) for row in transcript_df.iter_rows(named=True)}

    for (transcript_id, position), data in merged_results.items():
        # Retrieve the start and end positions for the current transcript_id
        transcript_start, transcript_end = transcript_positions[transcript_id]

        # Ensure absolute positions are within the actual sequence range
        if data['absolute_position'] is not None and (data['absolute_position'] >= transcript_start and data['absolute_position'] <= transcript_end):
            trimmed_results[(transcript_id, position)] = data

    # Convert results to efficient output format if requested
    if efficient_output:
        efficient_results = defaultdict(lambda: {
            'seqname': "",
            'gene_id': "",
            'gene_name': "",
            'strand': "",
            'tx_start': None,
            'tx_end': None,
            'donor_prob': [],
            'acceptor_prob': [],
            'neither_prob': [],
            'positions': []
        })

        for (transcript_id, position), data in trimmed_results.items():
            if transcript_id not in efficient_results:
                # Initialize basic information about the gene
                efficient_results[transcript_id]['seqname'] = data['seqname']
                efficient_results[transcript_id]['gene_id'] = data['gene_id']
                efficient_results[transcript_id]['gene_name'] = data['gene_name']
                efficient_results[transcript_id]['strand'] = data['strand']
                efficient_results[transcript_id]['tx_start'], efficient_results[transcript_id]['tx_end'] = \
                    transcript_positions[transcript_id]

            # Store per-position data
            efficient_results[transcript_id]['donor_prob'].append(np.mean(data['donor_prob']))
            efficient_results[transcript_id]['acceptor_prob'].append(np.mean(data['acceptor_prob']))
            efficient_results[transcript_id]['neither_prob'].append(np.mean(data['neither_prob']))
            efficient_results[transcript_id]['positions'].append(data['absolute_position'])

        # Sanity check to ensure consistency between efficient_results and trimmed_results
        for transcript_id in efficient_results.keys():
            assert len(efficient_results[transcript_id]['donor_prob']) == len(efficient_results[transcript_id]['positions']), \
                f"Mismatch in efficient_results for transcript_id {transcript_id}: donor_prob length doesn't match positions length"
            assert len(efficient_results[transcript_id]['acceptor_prob']) == len(efficient_results[transcript_id]['positions']), \
                f"Mismatch in efficient_results for gene {transcript_id}: acceptor_prob length doesn't match positions length"

        return efficient_results

    # Consolidate results by averaging overlapping predictions
    results = []
    for (transcript_id, position), data in trimmed_results.items():
        tx_start, tx_end = transcript_positions[transcript_id]
        results.append({
            'seqname': data['seqname'],
            'gene_id': data['gene_id'],
            'gene_name': data['gene_name'],

            'transcript_id': transcript_id,
            'position': position,
            'absolute_position': data['absolute_position'],
            'tx_start': tx_start,
            'tx_end': tx_end,
            'donor_prob': np.mean(data['donor_prob']),
            'acceptor_prob': np.mean(data['acceptor_prob']),
            'neither_prob': np.mean(data['neither_prob']),
            'strand': data['strand']
        })

    # Convert results to Polars DataFrame
    results_df = pl.DataFrame(results)

    # Sanity check: Ensure output probabilities match the input sequence lengths
    for transcript_id in transcript_df.select('transcript_id').unique().to_series().to_list():
        # unique().to_series().to_list() construct is used to retrieve unique transcript IDs.

        input_length = len(transcript_df.filter(pl.col('transcript_id') == transcript_id)['sequence'].to_list()[0])
        # For accessing specific values, you can use .filter and .to_list() to retrieve values from Polars columns.
        
        output_length = results_df.filter(pl.col('transcript_id') == transcript_id).shape[0]
        assert input_length == output_length, f"Length mismatch for {transcript_id}: input {input_length}, output {output_length}"

    return results_df


def predict_splice_sites_v0(transcript_df, models, context=10000):
    """
    Generate splice site predictions for each transcript sequence in transcript_df using SpliceAI models.
    This version is not as efficient as the later versions, but it provides a more straightforward implementation.

    Parameters:
    - transcript_df (pd.DataFrame): DataFrame with columns 'transcript_id', 'sequence', 'seqname', 'strand', 'start', 'end'.
    - models (list): List of loaded SpliceAI models.
    - context (int): Context length for SpliceAI (default 10000).

    Returns:
    - pd.DataFrame: DataFrame with columns 'transcript_id', 'position', 'absolute_position', 'donor_prob', 'acceptor_prob', 'neither_prob'.
    """
    
    # Dictionary to store merged results by nucleotide positions
    merged_results = defaultdict(lambda: {'donor_prob': [], 'acceptor_prob': [], 'neither_prob': []})

    for idx, row in transcript_df.iterrows():
        transcript_id = row['transcript_id']
        gene_name = row.get('gene_name', 'unknown')
        sequence = row['sequence']
        seqname = row.get('seqname', row.get('chrom'))  # chromosome or sequence name  # chromosome or sequence name
        strand = row['strand']
        seq_len = len(sequence)

        print("(predict_splice_sites) Processing transcript:", transcript_id)
        print("... Sequence Length:", seq_len)

        # Check if 'start' and 'end' positions are available for absolute positioning
        has_absolute_positions = 'start' in row and 'end' in row
        transcript_start = row.get('start', None)  # Absolute start position if available
        transcript_end = row.get('end', None)  # Absolute end position if available

        # Prepare input sequence using SpliceAI preprocessing logic
        input_blocks = prepare_input_sequence(sequence, context)

        print(f"[debug] Generated {len(input_blocks)} blocks for transcript {transcript_id}.")
        
        # Predict splice sites using SpliceAI models for each block
        for block_index, block in enumerate(input_blocks):
            x = block[None, :]  # Add batch dimension for model input

            # Predict splice sites using SpliceAI models
            y = np.mean([model.predict(x) for model in models], axis=0)
            # NOTE: y has the shape (1, block_size, 3), where 
            #       block_size is context / 2 + 5000 + context / 2, 
            #       and 3 represents the three classes (neither, acceptor, donor)
            print(f"[debug] Data shape for block {block_index}")
            print("... shape(x): ", x.shape)
            print("... shape(y): ", y.shape)

            # Extract probabilities, adjusting for cropping
            # donor_prob = y[0, context // 2:context // 2 + 5000, 2]
            # acceptor_prob = y[0, context // 2:context // 2 + 5000, 1]
            # neither_prob = y[0, context // 2:context // 2 + 5000, 0]
            # NOTE: 
            #   - These predictions are only for the central 5,000 nucleotides of each chunked sequence. 
            #       This central region excludes the flanking contextual sequences added on both sides.
            #   - The slice context // 2:context // 2 + 5000 retrieves predictions for the central 5,000 nucleotides
            #   - However, SpliceAI's output does not incldue the flanking sequences, so the predictions need to be adjusted.

            donor_prob = y[0, :, 2]
            acceptor_prob = y[0, :, 1]
            neither_prob = y[0, :, 0]

            # Check if predictions are generated correctly
            print(f"[debug] Extracted {len(donor_prob)} probabilities for block {block_index}.")

            # Calculate the start position of the current block relative to the original sequence
            block_start = block_index * 5000

            # Store the results with adjusted positions
            for i, (donor_p, acceptor_p, neither_p) in enumerate(zip(donor_prob, acceptor_prob, neither_prob)):
                
                if has_absolute_positions:
                    # Adjust absolute position based on the strand
                    if strand == '+':
                        absolute_position = transcript_start + (block_start + i)
                    elif strand == '-':
                        # Decrement from transcript_end for the negative strand
                        absolute_position = transcript_end - (block_start + i)
                    else: 
                        raise ValueError(f"Invalid strand: {strand}")
                else:
                    absolute_position = None
                
                pos_key = (transcript_id, absolute_position) if has_absolute_positions else (transcript_id, block_start + i + 1)

                # Append probabilities for overlapping positions
                merged_results[pos_key]['donor_prob'].append(donor_p)
                merged_results[pos_key]['acceptor_prob'].append(acceptor_p)
                merged_results[pos_key]['neither_prob'].append(neither_p)
                merged_results[pos_key]['strand'] = strand
                merged_results[pos_key]['seqname'] = seqname
                merged_results[pos_key]['gene_name'] = gene_name
                merged_results[pos_key]['absolute_position'] = absolute_position if has_absolute_positions else (block_start + i + 1)

                # print(f"[debug] Added prediction for position {pos_key}")

    assert len(merged_results) > 0, "No splice site predictions generated."

    # Trim unwanted probabilities:
    # Remove any positions corresponding to flanking sequences or padding
    trimmed_results = {}

    # Create a dictionary to map transcript_id to its start and end positions
    transcript_positions = {row['transcript_id']: (row['start'], row['end']) for _, row in transcript_df.iterrows()}

    for (transcript_id, position), data in merged_results.items():
        # Retrieve the start and end positions for the current transcript_id
        transcript_start, transcript_end = transcript_positions[transcript_id]

        # Ensure absolute positions are within the actual sequence range
        if data['absolute_position'] is not None and (data['absolute_position'] >= transcript_start and data['absolute_position'] <= transcript_end):
            trimmed_results[(transcript_id, position)] = data

    # Consolidate results by averaging overlapping predictions
    results = []
    for (transcript_id, position), data in trimmed_results.items():
        results.append({
            'seqname': data['seqname'],
            'transcript_id': transcript_id,
            'gene_name': data['gene_name'],
            'position': position,
            'absolute_position': data['absolute_position'],
            'donor_prob': np.mean(data['donor_prob']),
            'acceptor_prob': np.mean(data['acceptor_prob']),
            'neither_prob': np.mean(data['neither_prob']),
            'strand': data['strand']
        })
    # NOTE: Due to the overlapping nature of the input blocks, the same nucleotide position may appear 
    #       in multiple blocks, leading to multiple predictions for that position.
    # chrom   start   end     strand  site_type       gene_id transcript_id

    # Convert results to DataFrame
    results_df = pl.DataFrame(results)

    # Sanity check: Ensure output probabilities match the input sequence lengths
    for transcript_id in transcript_df['transcript_id'].unique():
        input_length = len(transcript_df[transcript_df['transcript_id'] == transcript_id]['sequence'].values[0])
        output_length = len(results_df[results_df['transcript_id'] == transcript_id])
        assert input_length == output_length, f"Length mismatch for {transcript_id}: input {input_length}, output {output_length}"

    return results_df


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


def analyze_splice_site_probabilities(
        predictions_df,
        donor_threshold_percentile=99.9,
        acceptor_threshold_percentile=99.9,
        save_plot=True,
        plot_file_name='splice_site_probabilities',
        plot_format='pdf',
        output_dir='.'):
    """
    Analyze the distribution of splice site probabilities and identify "unusually large" donor and acceptor probabilities.
    
    Parameters:
    - predictions_df (pd.DataFrame): DataFrame with columns 'gene_name', 'position', 'donor_prob', 'acceptor_prob', 'neither_prob'.
    - donor_threshold_percentile (float): Percentile threshold to define "unusually large" donor probabilities (default: 99.9).
    - acceptor_threshold_percentile (float): Percentile threshold to define "unusually large" acceptor probabilities (default: 99.9).
    - save_plot (bool): Whether to save the plot to a file (default: False).
    - plot_file_name (str): Name of the plot file (default: 'splice_site_probabilities').
    - plot_format (str): Format of the plot file (default: 'png').
    - output_dir (str): Directory to save the plot file (default: current directory).
    
    Returns:
    - unusual_donor_sites (pd.DataFrame): DataFrame of positions with unusually large donor probabilities.
    - unusual_acceptor_sites (pd.DataFrame): DataFrame of positions with unusually large acceptor probabilities.
    """
    is_polars = False 
    if isinstance(predictions_df, pl.DataFrame):
        is_polars = True
        predictions_df = predictions_df.to_pandas()

    # Plotting the distribution of probabilities
    plt.figure(figsize=(12, 6))
    sns.histplot(predictions_df['donor_prob'], bins=100, color='blue', kde=True, label='Donor Probabilities')
    sns.histplot(predictions_df['acceptor_prob'], bins=100, color='red', kde=True, label='Acceptor Probabilities')
    plt.title('Distribution of Donor and Acceptor Probabilities')
    plt.xlabel('Probability Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.yscale('log')  # Use log scale for better visualization

    if save_plot:
        plot_path = os.path.join(output_dir, f"{plot_file_name}.{plot_format}")
        os.makedirs(output_dir, exist_ok=True)
        
        plt.savefig(plot_path, format=plot_format)
        print(f"[i/o] Plot saved to {plot_path}")
    else:
        plt.show()

    # Calculating thresholds for unusually large probabilities
    donor_threshold = np.percentile(predictions_df['donor_prob'], donor_threshold_percentile)
    acceptor_threshold = np.percentile(predictions_df['acceptor_prob'], acceptor_threshold_percentile)
    print(f"Donor Probability Threshold (>{donor_threshold_percentile}th Percentile): {donor_threshold}")
    print(f"Acceptor Probability Threshold (>{acceptor_threshold_percentile}th Percentile): {acceptor_threshold}")

    # Identifying unusually large donor and acceptor probabilities
    unusual_donor_sites = predictions_df[predictions_df['donor_prob'] > donor_threshold]
    unusual_acceptor_sites = predictions_df[predictions_df['acceptor_prob'] > acceptor_threshold]
    
    print(f"Number of Unusually Large Donor Sites: {len(unusual_donor_sites)}")
    print(f"Number of Unusually Large Acceptor Sites: {len(unusual_acceptor_sites)}")

    return unusual_donor_sites, unusual_acceptor_sites


def get_short_junction_name(gene, donor_pos, acceptor_pos):
    """
    This function generate names like GENE_JUNC_abc123 via hashing, which are unique but more concise.
    """
    from hashlib import md5

    # Create a short hash using the donor and acceptor positions
    short_hash = md5(f"{donor_pos}-{acceptor_pos}".encode()).hexdigest()[:6]
    return f"{gene}_JUNC_{short_hash}"


def generate_splice_junctions_bed_file_v0(predictions_df, donor_threshold_percentile=99.9, acceptor_threshold_percentile=99.9, top_n=None, output_bed_file='splice_junctions.bed', output_dir='.'):
    """
    Analyze the distribution of splice site probabilities, identify "unusually large" donor and acceptor probabilities, and generate BED file representations of the junctions.
    
    Logic: 
    - For each gene, the function selects the top N donor and acceptor sites based on their probabilities.
    - It then pairs the donor and acceptor sites to form junctions.
        - Forming Splice Junctions:
            For Positive Strand (strand == '+'): 
                The function finds the nearest downstream acceptor for each donor site. 
                It pairs them to form a junction and calculates the junction score as the average of 
                donor and acceptor probabilities.
            For Negative Strand (strand == '-'): 
                The function finds the nearest upstream acceptor for each donor site and pairs them similarly, 
                adjusting the junction score.
            Sort by Coordinates: 
                The splice junctions are sorted by coordinates to ensure they are listed from 5' to 3'.
    - The function sorts the junctions by coordinates from 5' to 3'.
    - It saves the junctions to a BED file with the gene name, donor position, acceptor position, and average probability.

    Parameters:
    - predictions_df (pd.DataFrame): DataFrame with columns 'gene_name', 'position', 'absolute_position', 'donor_prob', 'acceptor_prob', 'neither_prob', 'strand'.
    - donor_threshold_percentile (float): Percentile threshold to define "unusually large" donor probabilities (default: 99.9).
    - acceptor_threshold_percentile (float): Percentile threshold to define "unusually large" acceptor probabilities (default: 99.9).
    - top_n (int): Number of top donor and acceptor probabilities to highlight (default: None).
    - output_bed_file (str): Name of the combined output BED file (default: 'splice_junctions.bed').
    - output_dir (str): Directory to save the BED files (default: current directory).
    
    Returns:
    - unusual_donor_sites (pd.DataFrame): DataFrame of positions with unusually large donor probabilities.
    - unusual_acceptor_sites (pd.DataFrame): DataFrame of positions with unusually large acceptor probabilities.

    Example usage:
        predictions_df = pd.read_csv('predictions.csv')
        generate_splice_junctions_bed_file(predictions_df, top_n=50, output_bed_file='splice_junctions.bed', output_dir='/path/to/output')
    """
    # Calculate thresholds for unusually large probabilities
    donor_threshold = np.percentile(predictions_df['donor_prob'], donor_threshold_percentile)
    acceptor_threshold = np.percentile(predictions_df['acceptor_prob'], acceptor_threshold_percentile)
    print(f"[info] Donor Probability Threshold (>{donor_threshold_percentile}th Percentile): {donor_threshold}")
    print(f"[info] Acceptor Probability Threshold (>{acceptor_threshold_percentile}th Percentile): {acceptor_threshold}")

    # Identify unusually large donor and acceptor probabilities
    unusual_donor_sites = predictions_df[predictions_df['donor_prob'] > donor_threshold]
    unusual_acceptor_sites = predictions_df[predictions_df['acceptor_prob'] > acceptor_threshold]
    
    print(f"Number of Unusually Large Donor Sites: {len(unusual_donor_sites)}")
    print(f"Number of Unusually Large Acceptor Sites: {len(unusual_acceptor_sites)}")

    if top_n is None: 
        print("[info] Top-N donor and acceptor sites not specified.")
        top_n = min(len(unusual_donor_sites), len(unusual_acceptor_sites))
        print(f"[info] Set top_n to the minimum of unusual donor and acceptor sites: {top_n}")

    # Initialize list to store BED file rows
    combined_bed_rows = []

    # Process each gene
    for gene in predictions_df['gene_name'].unique():

        gene_df = predictions_df[predictions_df['gene_name'] == gene]

        # Check if 'seqname' column exists in the DataFrame
        if 'seqname' in gene_df.columns:
            # Get the unique seqname for the current gene
            unique_seqnames = gene_df['seqname'].unique()
            
            # Ensure that there is only one unique seqname for the gene
            if len(unique_seqnames) != 1:
                raise ValueError(f"Gene {gene} is associated with multiple chromosomes: {unique_seqnames}")
            
            # Assign the unique seqname
            seqname = unique_seqnames[0]
        else:
            seqname = gene  # seqname is not available, use gene name as seqname

        top_donor_sites = gene_df.nlargest(top_n, 'donor_prob')
        top_acceptor_sites = gene_df.nlargest(top_n, 'acceptor_prob')
        
        # Pair donor and acceptor sites to form junctions
        bed_rows = []
        for _, donor_row in top_donor_sites.iterrows():
            donor_pos = donor_row['absolute_position'] if 'absolute_position' in donor_row else donor_row['position']
            donor_prob = donor_row['donor_prob']
            strand = donor_row['strand']  # Assuming 'strand' column exists in the DataFrame
            
            if strand == '+':
                # Find the nearest acceptor site downstream of the donor site on the positive strand
                downstream_acceptors = top_acceptor_sites[top_acceptor_sites['absolute_position'] > donor_pos] if 'absolute_position' in top_acceptor_sites else top_acceptor_sites[top_acceptor_sites['position'] > donor_pos]
                
                if not downstream_acceptors.empty:
                    downstream_acceptors = downstream_acceptors.sort_values(by='absolute_position' if 'absolute_position' in downstream_acceptors else 'position')
                    nearest_acceptor = downstream_acceptors.iloc[0]
                    acceptor_pos = nearest_acceptor['absolute_position'] if 'absolute_position' in nearest_acceptor else nearest_acceptor['position']
                    acceptor_prob = nearest_acceptor['acceptor_prob']
                    junction_prob = (donor_prob + acceptor_prob) / 2
                    
                    # bed_rows.append([gene, donor_pos, acceptor_pos, f"{gene}_JUNC", junction_prob, strand])
                    bed_rows.append([seqname, donor_pos, acceptor_pos, f"{gene}_JUNC", junction_prob, strand])

                    # Alternatively, we can use a more distinctive name for the junction
                    # junction_name = get_short_junction_name(gene, donor_pos, acceptor_pos)
                    # bed_rows.append([gene, donor_pos, acceptor_pos, junction_name, junction_prob, strand])

            elif strand == '-':
                # Find the nearest acceptor site upstream of the donor site on the negative strand
                upstream_acceptors = top_acceptor_sites[top_acceptor_sites['absolute_position'] < donor_pos] if 'absolute_position' in top_acceptor_sites else top_acceptor_sites[top_acceptor_sites['position'] < donor_pos]
                
                if not upstream_acceptors.empty:
                    upstream_acceptors = upstream_acceptors.sort_values(by='absolute_position' if 'absolute_position' in upstream_acceptors else 'position', ascending=False)
                    nearest_acceptor = upstream_acceptors.iloc[0]
                    acceptor_pos = nearest_acceptor['absolute_position'] if 'absolute_position' in nearest_acceptor else nearest_acceptor['position']
                    acceptor_prob = nearest_acceptor['acceptor_prob']
                    junction_prob = (donor_prob + acceptor_prob) / 2
                    
                    # bed_rows.append([gene, acceptor_pos, donor_pos, f"{gene}_JUNC", junction_prob, strand])
                    bed_rows.append([seqname, acceptor_pos, donor_pos, f"{gene}_JUNC", junction_prob, strand])

                    # Alternatively, we can use a more distinctive name for the junction
                    # junction_name = get_short_junction_name(gene, acceptor_pos, donor_pos)
                    # bed_rows.append([gene, acceptor_pos, donor_pos, junction_name, junction_prob, strand])
        
        # Sort junctions by coordinates from 5' to 3'
        bed_rows.sort(key=lambda x: (x[1], x[2]))

        # Add junction index
        for idx, row in enumerate(bed_rows):
            if row[5] == '+':
                row[3] = f"{row[3]}_{idx+1}"
            else:
                row[3] = f"{row[3]}_{len(bed_rows)-idx}"
        
        # Save BED file for each gene
        gene_bed_file_path = os.path.join(output_dir, f"{gene}_splice_junctions.bed")
        gene_bed_df = pd.DataFrame(bed_rows, columns=['seqname', 'start', 'end', 'name', 'score', 'strand'])
        
        max_score = gene_bed_df['score'].max()
        if max_score > 0:
            gene_bed_df['score'] = (gene_bed_df['score'] / max_score) * 1000
        
        gene_bed_df.to_csv(gene_bed_file_path, sep='\t', header=False, index=False)
        print(f"[i/o] BED file for gene {gene} saved to {gene_bed_file_path}")
        
        # Add to combined list
        combined_bed_rows.extend(bed_rows)

    # Sort combined junctions by coordinates from 5' to 3'
    combined_bed_rows.sort(key=lambda x: (x[0], x[1], x[2]))
    
    # Create a DataFrame for combined BED file
    combined_bed_df = pd.DataFrame(combined_bed_rows, columns=['seqname', 'start', 'end', 'name', 'score', 'strand'])

    # Normalize the score to be between 0 and 1000
    max_score = combined_bed_df['score'].max()
    if max_score > 0:
        combined_bed_df['score'] = (combined_bed_df['score'] / max_score) * 1000
    
    # Save the combined DataFrame to a BED file
    combined_bed_file_path = os.path.join(output_dir, output_bed_file)
    combined_bed_df.to_csv(combined_bed_file_path, sep='\t', header=False, index=False)
    print(f"[i/o] Combined BED file saved to {combined_bed_file_path}")

    return combined_bed_file_path, unusual_donor_sites, unusual_acceptor_sites


def get_output_bed_file_path(output_bed_file, output_dir, add_probabilities, gene_id=None, tx_id=None):
    """
    Modify the output_bed_file name to include 'with_probabilities' before the extension if add_probabilities is True.
    Optionally, include the gene name in the file name.

    Parameters:
    - output_bed_file (str): The original output BED file name.
    - output_dir (str): The directory to save the BED file.
    - add_probabilities (bool): Whether to include probabilities in the BED file name.
    - gene_id (str): The gene name or gene ID to include in the file name (optional).

    Returns:
    - str: The modified output BED file path.
    """
    # Split the file name and extension
    file_name, file_extension = os.path.splitext(output_bed_file)
    
    # Modify the file name if add_probabilities is True
    if add_probabilities:
        file_name += "_with_probabilities"

    if tx_id: 
        file_name = f"{tx_id}_{file_name}"
    else:    
        # Include the gene name if provided
        if gene_id:
            file_name = f"{gene_id}_{file_name}"
    
    # Combine the file name and extension
    return os.path.join(output_dir, f"{file_name}{file_extension}")


####################################################################################################

def get_position(row):
    """
    Helper function to get the position of a site.
    
    Parameters:
    - row (pd.Series): A row from a DataFrame containing site information.
    
    Returns:
    - int: The position of the site.
    """
    return row['absolute_position'] if 'absolute_position' in row else row['position']


def process_gene_by_combined_proba(gene_df, gene, add_probabilities, topn, combined_prob_threshold):
    """
    Pair donor and acceptor sites by their probabilities and generate BED rows.

    Parameters:
    - gene_df (pd.DataFrame): DataFrame filtered for the specific gene.
    - gene (str): Gene name.
    - add_probabilities (bool): Whether to add donor and acceptor probabilities to the output BED rows.
    - topn (int): Number of top donor and acceptor sites to consider.
    - combined_prob_threshold (float): Minimum combined probability threshold for valid junctions.

    Returns:
    - bed_rows (list): List of BED rows for the gene.
    """
    # Check if 'seqname' column exists in the DataFrame
    if 'seqname' in gene_df.columns:
        unique_seqnames = gene_df['seqname'].unique()
        if len(unique_seqnames) != 1:
            raise ValueError(f"Gene {gene} is associated with multiple chromosomes: {unique_seqnames}")
        seqname = unique_seqnames[0]
    else:
        seqname = None

    # Combine donor and acceptor sites without filtering by donor/acceptor threshold
    potential_donor_sites = gene_df.nlargest(topn, 'donor_prob')
    potential_acceptor_sites = gene_df.nlargest(topn, 'acceptor_prob')

    # Pair donor and acceptor sites to form junctions
    bed_rows = []
    for _, donor_row in potential_donor_sites.iterrows():
        donor_pos = get_position(donor_row)
        donor_prob = donor_row['donor_prob']
        strand = donor_row['strand']

        if strand == '+':
            # Apply get_position to each row in potential_acceptor_sites
            downstream_acceptors = potential_acceptor_sites[potential_acceptor_sites.apply(get_position, axis=1) > donor_pos]
            
            if not downstream_acceptors.empty:
                sorted_acceptors = downstream_acceptors.sort_values(by=downstream_acceptors.apply(get_position, axis=1))
                nearest_acceptor = sorted_acceptors.iloc[0]
                acceptor_pos = get_position(nearest_acceptor)
                acceptor_prob = nearest_acceptor['acceptor_prob']
                junction_prob = (donor_prob + acceptor_prob) / 2

                # Apply the combined probability threshold only once
                if junction_prob > combined_prob_threshold:
                    row = [seqname if seqname else gene, donor_pos, acceptor_pos, f"{gene}_JUNC", junction_prob, strand]
                    if add_probabilities:
                        row.extend([donor_prob, acceptor_prob])
                    bed_rows.append(row)

        elif strand == '-':
            upstream_acceptors = potential_acceptor_sites[potential_acceptor_sites.apply(get_position, axis=1) < donor_pos]
            
            if not upstream_acceptors.empty:
                sorted_acceptors = upstream_acceptors.sort_values(by=upstream_acceptors.apply(get_position, axis=1), ascending=False)
                nearest_acceptor = sorted_acceptors.iloc[0]
                acceptor_pos = get_position(nearest_acceptor)
                acceptor_prob = nearest_acceptor['acceptor_prob']
                junction_prob = (donor_prob + acceptor_prob) / 2

                # Apply the combined probability threshold only once
                if junction_prob > combined_prob_threshold:
                    row = [seqname if seqname else gene, acceptor_pos, donor_pos, f"{gene}_JUNC", junction_prob, strand]
                    if add_probabilities:
                        row.extend([donor_prob, acceptor_prob])
                    bed_rows.append(row)

    # Sort junctions by coordinates from 5' to 3'
    bed_rows.sort(key=lambda x: (x[1], x[2]))  # No need to include x[0] because all rows are for the same gene

    # Add junction index
    for idx, row in enumerate(bed_rows):
        if row[5] == '+':
            row[3] = f"{row[3]}_{idx+1}"
        else:
            row[3] = f"{row[3]}_{len(bed_rows)-idx}"

    return bed_rows
    

def process_gene_by_distance_v0(gene_df, gene, site_prob_threshold=0.7, add_probabilities=False):
    """
    Pair donor and acceptor sites to form the most probable splice junctions by matching each donor to the closest acceptor.
    
    Parameters:
    - gene_df (pd.DataFrame): DataFrame filtered for the specific gene.
    - gene (str): Gene name.
    - site_prob_threshold (float): Minimum threshold for both donor and acceptor site probabilities (default: 0.7).
    - add_probabilities (bool): Whether to add donor and acceptor probabilities to the output BED rows.
    
    Returns:
    - bed_rows (list): List of BED rows for the gene.

    Key Points:

    - Filter Donors and Acceptors by site_prob_threshold:
        Only donors and acceptors with probabilities greater than or equal to say, 0.7, are considered.
        
        Match Each Donor to the Closest Acceptor:

        For each donor, find the closest downstream or upstream acceptor based on the strand.
        This approach guarantees that each donor is matched to the nearest acceptable acceptor.
    """
    # Helper function to get the position of a site
    def get_position(row):
        return row['absolute_position'] if 'absolute_position' in row else row['position']

    # Determine the sequence name ('seqname') for the gene if it exists in the DataFrame
    if 'seqname' in gene_df.columns:
        unique_seqnames = gene_df['seqname'].unique()
        if len(unique_seqnames) != 1:
            raise ValueError(f"Gene {gene} is associated with multiple chromosomes: {unique_seqnames}")
        seqname = unique_seqnames[0]
    else:
        seqname = None

    # Filter donor and acceptor sites based on the probability threshold
    qualified_donors = gene_df[gene_df['donor_prob'] >= site_prob_threshold].sort_values(by='donor_prob', ascending=False)
    qualified_acceptors = gene_df[gene_df['acceptor_prob'] >= site_prob_threshold].sort_values(by='acceptor_prob', ascending=False)

    # Pair donor and acceptor sites to form junctions
    bed_rows = []
    for _, donor_row in qualified_donors.iterrows():
        donor_pos = get_position(donor_row)
        donor_prob = donor_row['donor_prob']
        strand = donor_row['strand']

        if strand == '+':
            # Find the closest downstream acceptor site
            downstream_acceptors = qualified_acceptors[qualified_acceptors.apply(get_position, axis=1) > donor_pos]
            if not downstream_acceptors.empty:

                # Find the closest acceptor site by calculating the absolute difference between the donor position 
                # and each acceptor position, then sorting these differences and selecting the smallest one.
                closest_acceptor = downstream_acceptors.iloc[(downstream_acceptors.apply(get_position, axis=1) - donor_pos).abs().argsort().iloc[0]]
                
                acceptor_pos = get_position(closest_acceptor)
                acceptor_prob = closest_acceptor['acceptor_prob']
                junction_prob = (donor_prob + acceptor_prob) / 2

                # Add junction to bed_rows
                row = [seqname if seqname else gene, donor_pos, acceptor_pos, f"{gene}_JUNC", junction_prob, strand]
                if add_probabilities:
                    row.extend([donor_prob, acceptor_prob])
                bed_rows.append(row)

        elif strand == '-':
            # Find the closest upstream acceptor site
            upstream_acceptors = qualified_acceptors[qualified_acceptors.apply(get_position, axis=1) < donor_pos]
            if not upstream_acceptors.empty:
                closest_acceptor = upstream_acceptors.iloc[(upstream_acceptors.apply(get_position, axis=1) - donor_pos).abs().argsort().iloc[0]]
                acceptor_pos = get_position(closest_acceptor)
                acceptor_prob = closest_acceptor['acceptor_prob']
                junction_prob = (donor_prob + acceptor_prob) / 2

                # Add junction to bed_rows
                row = [seqname if seqname else gene, acceptor_pos, donor_pos, f"{gene}_JUNC", junction_prob, strand]
                if add_probabilities:
                    row.extend([donor_prob, acceptor_prob])
                bed_rows.append(row)

    # Sort junctions by coordinates from 5' to 3'
    bed_rows.sort(key=lambda x: (x[1], x[2]))

    # Add junction index 
    for idx, row in enumerate(bed_rows):
        if row[5] == '+':
            row[3] = f"{row[3]}_{idx+1}"
        else:
            row[3] = f"{row[3]}_{len(bed_rows)-idx}"

    return bed_rows


def process_gene_by_distance(gene_df, gene, num_junctions=None, initial_prob_threshold=0.7, add_probabilities=False):
    """
    Pair donor and acceptor sites to form the most probable splice junctions by matching each donor to the closest acceptor.
    Optionally, adjust site_prob_threshold to match a desired number of junctions.
    
    Parameters:
    - gene_df (pd.DataFrame): DataFrame filtered for the specific gene.
    - gene (str): Gene name.
    - num_junctions (int): Desired number of junctions to generate (default: None).
    - initial_prob_threshold (float): Initial threshold for both donor and acceptor site probabilities (default: 0.7).
    - add_probabilities (bool): Whether to add donor and acceptor probabilities to the output BED rows.
    
    Returns:
    - bed_rows (list): List of BED rows for the gene.
    """
    # Helper function to get the position of a site
    def get_position(row):
        return row['absolute_position'] if 'absolute_position' in row else row['position']

    # Test
    print("[debug] Gene:", gene)
    print("... Gene DataFrame:")
    print(gene_df.head())

    # Determine the sequence name ('seqname') for the gene if it exists in the DataFrame
    if 'seqname' in gene_df.columns:
        unique_seqnames = gene_df['seqname'].unique()
        if len(unique_seqnames) != 1:
            raise ValueError(f"Gene {gene} is associated with multiple chromosomes: {unique_seqnames}")
        seqname = unique_seqnames[0]
    else:
        seqname = None

    # Initialize probability threshold
    site_prob_threshold = initial_prob_threshold
    
    # If num_junctions is specified, adjust the site_prob_threshold to achieve the desired number of junctions
    if num_junctions is not None:
        # Start with a high probability threshold and gradually decrease it
        max_iterations = 50  # Limit to avoid infinite loops
        lower_bound = 0.1
        upper_bound = 0.99
        for _ in range(max_iterations):
            # Filter donor and acceptor sites based on the current probability threshold
            qualified_donors = gene_df[gene_df['donor_prob'] >= site_prob_threshold].sort_values(by='donor_prob', ascending=False)
            qualified_acceptors = gene_df[gene_df['acceptor_prob'] >= site_prob_threshold].sort_values(by='acceptor_prob', ascending=False)
            
            # Generate junctions based on the current threshold
            bed_rows = generate_junctions(qualified_donors, qualified_acceptors, gene, seqname, add_probabilities, get_position)
            
            # Check the number of junctions generated
            if len(bed_rows) == num_junctions:
                break  # Found the optimal threshold
            elif len(bed_rows) < num_junctions:
                site_prob_threshold -= 0.02  # Decrease threshold to include more sites
            else:
                site_prob_threshold += 0.02  # Increase threshold to include fewer sites
            
            # Keep the threshold within bounds
            site_prob_threshold = max(lower_bound, min(upper_bound, site_prob_threshold))

    print("[debug] Final Probability Threshold:", site_prob_threshold)
    
    # Generate the final bed_rows with the determined or adjusted threshold
    qualified_donors = gene_df[gene_df['donor_prob'] >= site_prob_threshold].sort_values(by='donor_prob', ascending=False)
    qualified_acceptors = gene_df[gene_df['acceptor_prob'] >= site_prob_threshold].sort_values(by='acceptor_prob', ascending=False)
    bed_rows = generate_junctions(qualified_donors, qualified_acceptors, gene, seqname, add_probabilities, get_position)

    print("[debug] Number of qualified Donors:", len(qualified_donors))
    print("[debug] Number of qualified Acceptors:", len(qualified_acceptors))
    print("[debug] Number of Junctions Generated:", len(bed_rows))

    # Sort junctions by coordinates from 5' to 3'
    bed_rows.sort(key=lambda x: (x[1], x[2]))

    # Add junction index
    for idx, row in enumerate(bed_rows):
        if row[5] == '+':
            row[3] = f"{row[3]}_{idx + 1}"
        else:
            row[3] = f"{row[3]}_{len(bed_rows) - idx}"

    return bed_rows


def process_transcript_by_distance(
        tx_df, 
        tx_id, 
        num_junctions=None, 
        initial_prob_threshold=0.7, 
        add_probabilities=False):
    """
    Pair donor and acceptor sites to form the most probable splice junctions by matching each donor to the closest acceptor.
    Optionally, adjust site_prob_threshold to match a desired number of junctions.
    
    Parameters:
    - tx_df (pd.DataFrame): DataFrame filtered for the specific transcript.
    - tx_id (str): Transcript ID. 
    - num_junctions (int): Desired number of junctions to generate (default: None).
    - initial_prob_threshold (float): Initial threshold for both donor and acceptor site probabilities (default: 0.7).
    - add_probabilities (bool): Whether to add donor and acceptor probabilities to the output BED rows.
    
    Returns:
    - bed_rows (list): List of BED rows for the gene.
    """
    # Helper function to get the position of a site
    def get_position(row):
        return row['absolute_position'] if 'absolute_position' in row else row['position']

    # Test
    print("[debug] Transcript:", tx_id)
    print("... Transcript DataFrame:")
    print(tx_df.head())

    # Determine the sequence name ('seqname') for the gene if it exists in the DataFrame
    if 'seqname' in tx_df.columns:
        unique_seqnames = tx_df['seqname'].unique()
        if len(unique_seqnames) != 1:
            raise ValueError(f"Transcript {tx_id} is associated with multiple chromosomes: {unique_seqnames}")
        seqname = unique_seqnames[0]
    else:
        seqname = None

    # Initialize probability threshold
    site_prob_threshold = initial_prob_threshold
    
    # If num_junctions is specified, adjust the site_prob_threshold to achieve the desired number of junctions
    if num_junctions is not None:
        # Start with a high probability threshold and gradually decrease it
        max_iterations = 50  # Limit to avoid infinite loops
        lower_bound = 0.1
        upper_bound = 0.99
        for _ in range(max_iterations):
            # Filter donor and acceptor sites based on the current probability threshold
            qualified_donors = tx_df[tx_df['donor_prob'] >= site_prob_threshold].sort_values(by='donor_prob', ascending=False)
            qualified_acceptors = tx_df[tx_df['acceptor_prob'] >= site_prob_threshold].sort_values(by='acceptor_prob', ascending=False)
            
            # Generate junctions based on the current threshold
            bed_rows = generate_junctions_for_transcripts(qualified_donors, qualified_acceptors, tx_id, seqname, add_probabilities, get_position)
            
            # Check the number of junctions generated
            if len(bed_rows) == num_junctions:
                break  # Found the optimal threshold
            elif len(bed_rows) < num_junctions:
                site_prob_threshold -= 0.02  # Decrease threshold to include more sites
            else:
                site_prob_threshold += 0.02  # Increase threshold to include fewer sites
            
            # Keep the threshold within bounds
            site_prob_threshold = max(lower_bound, min(upper_bound, site_prob_threshold))

    print("(process_transcript_by_distance) Final Probability Threshold: {} given n_junctions={}".format(site_prob_threshold, num_junctions))
    
    # Generate the final bed_rows with the determined or adjusted threshold
    qualified_donors = tx_df[tx_df['donor_prob'] >= site_prob_threshold].sort_values(by='donor_prob', ascending=False)
    qualified_acceptors = tx_df[tx_df['acceptor_prob'] >= site_prob_threshold].sort_values(by='acceptor_prob', ascending=False)
    bed_rows = generate_junctions(qualified_donors, qualified_acceptors, tx_id, seqname, add_probabilities, get_position)

    print("[debug] Number of qualified Donors:", len(qualified_donors))
    print("[debug] Number of qualified Acceptors:", len(qualified_acceptors))
    print("[debug] Number of Junctions Generated:", len(bed_rows))

    # Sort junctions by coordinates from 5' to 3'
    bed_rows.sort(key=lambda x: (x[1], x[2]))

    # Add junction index
    for idx, row in enumerate(bed_rows):
        if row[5] == '+':
            row[3] = f"{row[3]}_{idx + 1}"
        else:
            row[3] = f"{row[3]}_{len(bed_rows) - idx}"

    return bed_rows


def generate_junctions(qualified_donors, qualified_acceptors, gene, seqname, add_probabilities, get_position):
    """
    Helper function to generate junctions by matching each donor with the closest acceptor.
    Ensures each donor and acceptor is only used once.
    
    Parameters:
    - qualified_donors (pd.DataFrame): DataFrame of filtered donor sites.
    - qualified_acceptors (pd.DataFrame): DataFrame of filtered acceptor sites.
    - gene (str): Gene name.
    - seqname (str): Sequence name.
    - add_probabilities (bool): Whether to add donor and acceptor probabilities to the output BED rows.
    - get_position (function): Function to get the position of a site.
    
    Returns:
    - bed_rows (list): List of BED rows for the gene.
    """
    bed_rows = []
    used_donor_positions = set()  # Keep track of used donor positions
    used_acceptor_positions = set()  # Keep track of used acceptor positions

    for _, donor_row in qualified_donors.iterrows():
        donor_pos = get_position(donor_row)
        donor_prob = donor_row['donor_prob']
        strand = donor_row['strand']

        # Skip if this donor has already been used
        if donor_pos in used_donor_positions:
            continue

        if strand == '+':
            # Find the closest downstream acceptor site that hasn't been used
            downstream_acceptors = qualified_acceptors[
                (qualified_acceptors.apply(get_position, axis=1) > donor_pos) & 
                (~qualified_acceptors.apply(get_position, axis=1).isin(used_acceptor_positions))
            ]
            if not downstream_acceptors.empty:
                closest_acceptor = downstream_acceptors.iloc[
                    (downstream_acceptors.apply(get_position, axis=1) - donor_pos).abs().argsort().iloc[0]]
                acceptor_pos = get_position(closest_acceptor)
                acceptor_prob = closest_acceptor['acceptor_prob']
                junction_prob = (donor_prob + acceptor_prob) / 2

                # Mark the donor and acceptor positions as used
                used_donor_positions.add(donor_pos)
                used_acceptor_positions.add(acceptor_pos)

                # Add junction to bed_rows
                row = [seqname if seqname else gene, donor_pos, acceptor_pos, f"{gene}_JUNC", junction_prob, strand]
                if add_probabilities:
                    row.extend([donor_prob, acceptor_prob])
                bed_rows.append(row)

        elif strand == '-':
            # Find the closest upstream acceptor site that hasn't been used
            upstream_acceptors = qualified_acceptors[
                (qualified_acceptors.apply(get_position, axis=1) < donor_pos) & 
                (~qualified_acceptors.apply(get_position, axis=1).isin(used_acceptor_positions))
            ]
            if not upstream_acceptors.empty:
                closest_acceptor = upstream_acceptors.iloc[
                    (upstream_acceptors.apply(get_position, axis=1) - donor_pos).abs().argsort().iloc[0]]
                acceptor_pos = get_position(closest_acceptor)
                acceptor_prob = closest_acceptor['acceptor_prob']
                junction_prob = (donor_prob + acceptor_prob) / 2

                # Mark the donor and acceptor positions as used
                used_donor_positions.add(donor_pos)
                used_acceptor_positions.add(acceptor_pos)

                # Add junction to bed_rows
                row = [seqname if seqname else gene, acceptor_pos, donor_pos, f"{gene}_JUNC", junction_prob, strand]
                if add_probabilities:
                    row.extend([donor_prob, acceptor_prob])
                bed_rows.append(row)
    
    return bed_rows


def generate_junctions_for_transcripts(qualified_donors, qualified_acceptors, tx_id, seqname, add_probabilities, get_position):
    """
    Helper function to generate junctions by matching each donor with the closest acceptor.
    Ensures each donor and acceptor is only used once.
    
    Parameters:
    - qualified_donors (pd.DataFrame): DataFrame of filtered donor sites.
    - qualified_acceptors (pd.DataFrame): DataFrame of filtered acceptor sites.
    - tx_id (str): Transcript ID.
    - seqname (str): Sequence name.
    - add_probabilities (bool): Whether to add donor and acceptor probabilities to the output BED rows.
    - get_position (function): Function to get the position of a site.
    
    Returns:
    - bed_rows (list): List of BED rows for the transcript (tx_id).
    """
    bed_rows = []
    used_donor_positions = set()  # Keep track of used donor positions
    used_acceptor_positions = set()  # Keep track of used acceptor positions

    for _, donor_row in qualified_donors.iterrows():
        donor_pos = get_position(donor_row)
        donor_prob = donor_row['donor_prob']
        strand = donor_row['strand']

        # Skip if this donor has already been used
        if donor_pos in used_donor_positions:
            continue

        if strand == '+':
            # Find the closest downstream acceptor site that hasn't been used
            downstream_acceptors = qualified_acceptors[
                (qualified_acceptors.apply(get_position, axis=1) > donor_pos) & 
                (~qualified_acceptors.apply(get_position, axis=1).isin(used_acceptor_positions))
            ]
            if not downstream_acceptors.empty:
                closest_acceptor = downstream_acceptors.iloc[
                    (downstream_acceptors.apply(get_position, axis=1) - donor_pos).abs().argsort().iloc[0]]
                acceptor_pos = get_position(closest_acceptor)
                acceptor_prob = closest_acceptor['acceptor_prob']
                junction_prob = (donor_prob + acceptor_prob) / 2

                # Mark the donor and acceptor positions as used
                used_donor_positions.add(donor_pos)
                used_acceptor_positions.add(acceptor_pos)

                # Add junction to bed_rows
                row = [seqname if seqname else tx_id, donor_pos, acceptor_pos, f"{tx_id}_JUNC", junction_prob, strand]
                if add_probabilities:
                    row.extend([donor_prob, acceptor_prob])
                bed_rows.append(row)

        elif strand == '-':
            # Find the closest upstream acceptor site that hasn't been used
            upstream_acceptors = qualified_acceptors[
                (qualified_acceptors.apply(get_position, axis=1) < donor_pos) & 
                (~qualified_acceptors.apply(get_position, axis=1).isin(used_acceptor_positions))
            ]
            if not upstream_acceptors.empty:
                closest_acceptor = upstream_acceptors.iloc[
                    (upstream_acceptors.apply(get_position, axis=1) - donor_pos).abs().argsort().iloc[0]]
                acceptor_pos = get_position(closest_acceptor)
                acceptor_prob = closest_acceptor['acceptor_prob']
                junction_prob = (donor_prob + acceptor_prob) / 2

                # Mark the donor and acceptor positions as used
                used_donor_positions.add(donor_pos)
                used_acceptor_positions.add(acceptor_pos)

                # Add junction to bed_rows
                row = [seqname if seqname else tx_id, acceptor_pos, donor_pos, f"{tx_id}_JUNC", junction_prob, strand]
                if add_probabilities:
                    row.extend([donor_prob, acceptor_prob])
                bed_rows.append(row)
    
    return bed_rows


def process_gene_by_knn(gene_df, gene, combined_prob_threshold, top_n, k=3, add_probabilities=False):
    """
    Pair donor and acceptor sites to form the most probable splice junctions using a k-NN approach.
    
    Parameters:
    - gene_df (pd.DataFrame): DataFrame filtered for the specific gene.
    - gene (str): Gene name.
    - combined_prob_threshold (float): Minimum combined probability threshold for valid junctions.
    - top_n (int): Number of top donor and acceptor sites to consider.
    - k (int): Number of nearest acceptors to consider for pairing with each donor site (default: 3).
    - add_probabilities (bool): Whether to add donor and acceptor probabilities to the output BED rows.
    
    Returns:
    - bed_rows (list): List of BED rows for the gene.
    """
    # Helper function to get the position of a site
    def get_position(row):
        return row['absolute_position'] if 'absolute_position' in row else row['position']

    # Determine the sequence name ('seqname') for the gene if it exists in the DataFrame
    if 'seqname' in gene_df.columns:
        unique_seqnames = gene_df['seqname'].unique()
        if len(unique_seqnames) != 1:
            raise ValueError(f"Gene {gene} is associated with multiple chromosomes: {unique_seqnames}")
        seqname = unique_seqnames[0]
    else:
        seqname = None

    # Consider top candidate donor and acceptor sites based on probabilities
    potential_donor_sites = gene_df.nlargest(top_n, 'donor_prob')
    potential_acceptor_sites = gene_df.nlargest(top_n, 'acceptor_prob')

    # Pair donor and acceptor sites to form junctions
    bed_rows = []
    for _, donor_row in potential_donor_sites.iterrows():
        donor_pos = get_position(donor_row)
        donor_prob = donor_row['donor_prob']
        strand = donor_row['strand']

        # Determine acceptor sites to consider based on strand
        if strand == '+':
            # Find the top k nearest acceptor sites downstream of the donor site on the positive strand
            
            # Apply get_position to each row in potential_acceptor_sites
            downstream_acceptors = potential_acceptor_sites[potential_acceptor_sites.apply(get_position, axis=1) > donor_pos]
            # NOTE: get_position() is applied to each row of the DataFrame potential_acceptor_sites to 
            #       retrieve the position value ('absolute_position' or 'position').

            if not downstream_acceptors.empty:
                # Sort acceptors by position and select the top k
                sort_column = 'absolute_position' if 'absolute_position' in downstream_acceptors.columns else 'position'
                sorted_acceptors = downstream_acceptors.sort_values(by=sort_column)
                downstream_acceptors = sorted_acceptors.head(k)

                # Choose the acceptor with the highest combined probability, resolve ties by closest distance
                best_acceptor = None
                best_junction_prob = 0
                best_distance = float('inf')  # Initialize with a large number
                for _, acceptor_row in downstream_acceptors.iterrows():
                    acceptor_pos = get_position(acceptor_row)
                    acceptor_prob = acceptor_row['acceptor_prob']
                    junction_prob = (donor_prob + acceptor_prob) / 2
                    distance = acceptor_pos - donor_pos

                    # Apply combined probability threshold and check for the best acceptor
                    if junction_prob > combined_prob_threshold:
                        # If junction probability is higher or equal, but closer in distance, select this acceptor
                        if junction_prob > best_junction_prob or (junction_prob == best_junction_prob and distance < best_distance):
                            best_acceptor = acceptor_row
                            best_junction_prob = junction_prob
                            best_distance = distance
                
                # If a valid acceptor was found, add it to the BED rows
                if best_acceptor is not None:
                    acceptor_pos = get_position(best_acceptor)
                    acceptor_prob = best_acceptor['acceptor_prob']
                    row = [seqname if seqname else gene, donor_pos, acceptor_pos, f"{gene}_JUNC", best_junction_prob, strand]
                    if add_probabilities:
                        row.extend([donor_prob, acceptor_prob])
                    bed_rows.append(row)

        elif strand == '-':
            # Find the top k nearest acceptor sites upstream of the donor site on the negative strand
            upstream_acceptors = potential_acceptor_sites[potential_acceptor_sites.apply(get_position, axis=1) < donor_pos]

            if not upstream_acceptors.empty:
                # Sort acceptors by position and select the top k
                sort_column = 'absolute_position' if 'absolute_position' in upstream_acceptors.columns else 'position'
                sorted_acceptors = upstream_acceptors.sort_values(by=sort_column, ascending=False)
                upstream_acceptors = sorted_acceptors.head(k)

                # Choose the acceptor with the highest combined probability, resolve ties by closest distance
                best_acceptor = None
                best_junction_prob = 0
                best_distance = float('inf')  # Initialize with a large number
                for _, acceptor_row in upstream_acceptors.iterrows():
                    acceptor_pos = get_position(acceptor_row)
                    acceptor_prob = acceptor_row['acceptor_prob']
                    junction_prob = (donor_prob + acceptor_prob) / 2
                    distance = donor_pos - acceptor_pos

                    # Apply combined probability threshold and check for the best acceptor
                    if junction_prob > combined_prob_threshold:
                        # If junction probability is higher or equal, but closer in distance, select this acceptor
                        if junction_prob > best_junction_prob or (junction_prob == best_junction_prob and distance < best_distance):
                            best_acceptor = acceptor_row
                            best_junction_prob = junction_prob
                            best_distance = distance
                
                # If a valid acceptor was found, add it to the BED rows
                if best_acceptor is not None:
                    acceptor_pos = get_position(best_acceptor)
                    acceptor_prob = best_acceptor['acceptor_prob']
                    row = [seqname if seqname else gene, acceptor_pos, donor_pos, f"{gene}_JUNC", best_junction_prob, strand]
                    if add_probabilities:
                        row.extend([donor_prob, acceptor_prob])
                    bed_rows.append(row)

    # Sort junctions by coordinates from 5' to 3'
    bed_rows.sort(key=lambda x: (x[1], x[2]))  # No need to include x[0] because all rows are for the same gene

    # Add junction index 
    for idx, row in enumerate(bed_rows):
        if row[5] == '+':
            row[3] = f"{row[3]}_{idx+1}"
        else:
            row[3] = f"{row[3]}_{len(bed_rows)-idx}"
    # NOTE: The junction names in the bed_rows list are updated with unique indices based on the strand. 
    # For the + strand, indices are assigned from 1 to n, and for the - strand, they are assigned from n to 1. 
    # This indexing ensures that each junction is uniquely identified.

    return bed_rows


def generate_splice_junctions_bed_file_with_probabilities(
        predictions_df, 
        donor_threshold_percentile=99.9, 
        acceptor_threshold_percentile=99.9, 
        combined_prob_threshold=0.5,
        site_prob_threshold=None,
        top_n=None, 
        output_bed_file='splice_junctions.bed', output_dir='.'):
    """
    Analyze the distribution of splice site probabilities, identify "unusually large" donor and acceptor probabilities, and generate a BED file representation of the junctions.
    
    Parameters:
    - predictions_df (pd.DataFrame): DataFrame with columns 'gene_name', 'position', 'absolute_position', 'donor_prob', 'acceptor_prob', 'neither_prob', 'strand'.
    - donor_threshold_percentile (float): Percentile threshold to define "unusually large" donor probabilities (default: 99.9).
    - acceptor_threshold_percentile (float): Percentile threshold to define "unusually large" acceptor probabilities (default: 99.9).
    - top_n (int): Number of top donor and acceptor probabilities to highlight (default: 50).
    - output_bed_file (str): Name of the output BED file (default: 'splice_junctions.bed').
    - output_dir (str): Directory to save the BED file (default: current directory).
    
    Returns:
    - unusual_donor_sites (pd.DataFrame): DataFrame of positions with unusually large donor probabilities.
    - unusual_acceptor_sites (pd.DataFrame): DataFrame of positions with unusually large acceptor probabilities.
    """
    return generate_splice_junctions_bed_file(
            predictions_df, 
            donor_threshold_percentile=donor_threshold_percentile, 
            acceptor_threshold_percentile=acceptor_threshold_percentile,
            combined_prob_threshold=combined_prob_threshold,
            site_prob_threshold=site_prob_threshold,
            top_n=top_n,
            add_probabilities=True, 
            output_bed_file=output_bed_file,
            output_dir=output_dir)


def generate_splice_junctions_bed_file(
        predictions_df, 
        donor_threshold_percentile=99.9,
        acceptor_threshold_percentile=99.9,
        combined_prob_threshold=0.5,
        site_prob_threshold=None,
        top_n=None,
        add_probabilities=False,
        output_bed_file='splice_junctions.bed', output_dir='.'):
    """
    Analyze the distribution of splice site probabilities, identify "unusually large" donor and acceptor probabilities, and generate BED file representations of the junctions.
    
    Parameters:
    - predictions_df (pd.DataFrame): DataFrame with columns 'gene_name', 'position', 'absolute_position', 'donor_prob', 'acceptor_prob', 'neither_prob', 'strand'.
    - donor_threshold_percentile (float): Percentile threshold to define "unusually large" donor probabilities (default: 99.9).
    - acceptor_threshold_percentile (float): Percentile threshold to define "unusually large" acceptor probabilities (default: 99.9).
    - combined_prob_threshold (float): Minimum combined probability threshold for valid junctions (default: 0.5).
    
    - top_n (dict): Number of top donor and acceptor probabilities to highlight (default: None).
                    If specified, top_n is a dictionary with gene names as keys and top_n values as values.
    
    - output_bed_file (str): Name of the combined output BED file (default: 'splice_junctions.bed').
    - output_dir (str): Directory to save the BED files (default: current directory).
    
    Returns:
    - combined_bed_file_path (str): Path to the combined BED file.
    - probable_donor_sites (dict): Dictionary of where key is gene name and value is a dataframe of probable donor sites.
    - probable_acceptor_sites (dict): Dictionary of where key is gene name and value is a dataframe of probable acceptor sites.
    - n_junctions_by_gene (dict): Dictionary of where key is gene name and value is the number of junctions for that gene.
    
      NOTE: Old return values:
        - unusual_donor_sites (pd.DataFrame): DataFrame of positions with unusually large donor probabilities.
        - unusual_acceptor_sites (pd.DataFrame): DataFrame of positions with unusually large acceptor probabilities.

    Example usage:
        predictions_df = pd.read_csv('predictions.csv')
        generate_splice_junctions_bed_file(predictions_df, top_n=50, output_bed_file='splice_junctions.bed', output_dir='/path/to/output')
    """

    # Calculate thresholds for unusually large probabilities
    donor_threshold = np.percentile(predictions_df['donor_prob'], donor_threshold_percentile)
    acceptor_threshold = np.percentile(predictions_df['acceptor_prob'], acceptor_threshold_percentile)
    print(f"[info] Donor Probability Threshold (>{donor_threshold_percentile}th Percentile): {donor_threshold}")
    print(f"[info] Acceptor Probability Threshold (>{acceptor_threshold_percentile}th Percentile): {acceptor_threshold}")

    # Identify unusually large donor and acceptor probabilities
    unusual_donor_sites = predictions_df[predictions_df['donor_prob'] > donor_threshold]
    unusual_acceptor_sites = predictions_df[predictions_df['acceptor_prob'] > acceptor_threshold]
    
    print(f"Number of Unusually Large Donor Sites: {len(unusual_donor_sites)}")
    print(f"Number of Unusually Large Acceptor Sites: {len(unusual_acceptor_sites)}")

    top_n_lookup = {}
    if top_n is None: 
        top_n_combined = min(len(unusual_donor_sites), len(unusual_acceptor_sites))
        print(f"[info] Minimum of number of probable donor and acceptor sites: {top_n_combined}")
    else: 
        assert isinstance(top_n, dict), "top_n should be a dictionary with gene names as keys and top_n values as values."
        top_n_lookup = top_n

    # Initialize list to store BED file rows
    combined_bed_rows = []
    file_paths = {}
    n_junctions_by_gene = {}
    probable_donor_sites = {}
    probable_acceptor_sites = {}

    # Define the columns based on the add_probabilities flag
    columns = ['seqname', 'start', 'end', 'name', 'score', 'strand']
    if add_probabilities:
        columns.extend(['donor_prob', 'acceptor_prob'])

    # Process each gene
    for gene in predictions_df['gene_name'].unique():
        # Subset the rows associated with the same gene
        gene_df = predictions_df[predictions_df['gene_name'] == gene]

        # [test]
        print(f"[info] Processing gene {gene} with {len(gene_df)} rows.")
        print(f"... Gene DataFrame:")
        print(gene_df.head())

        # Find and print the top N donor probabilities
        top_donor_probs = gene_df['donor_prob'].nlargest(50)
        print(f"... Top Donor Probabilities: {top_donor_probs.values}")
        # Find and print the top N acceptor probabilities
        top_acceptor_probs = gene_df['acceptor_prob'].nlargest(50)
        print(f"... Top Acceptor Probabilities: {top_acceptor_probs.values}")

        donor_threshold = np.percentile(gene_df['donor_prob'], donor_threshold_percentile)
        acceptor_threshold = np.percentile(gene_df['acceptor_prob'], acceptor_threshold_percentile)
        probable_donor_sites[gene] = gene_df[gene_df['donor_prob'] > donor_threshold]
        probable_acceptor_sites[gene] = gene_df[gene_df['acceptor_prob'] > acceptor_threshold]
        if site_prob_threshold is None:
            site_prob_threshold = min(donor_threshold, acceptor_threshold)

        estimated_n_junctions = top_n_lookup.get(gene, None)
        if estimated_n_junctions is None:
            estimated_n_junctions = top_n_lookup[gene] = \
                min(len(probable_donor_sites[gene]), len(probable_acceptor_sites[gene]))
            print(f"... Seting guesstimated n_junctions to the min of probable donor and acceptor sites: {estimated_n_junctions}")
        else: 
            print(f"... Using top_n_lookup value for estimated_n_junctions: {estimated_n_junctions}")

        print(f"[info] Gene: {gene}")
        print(f"... Number of Unusually Large Donor Sites: {len(probable_donor_sites[gene])}")
        sorted_donor_probs = probable_donor_sites[gene]['donor_prob'].sort_values(ascending=False).values
        print(f"..... donor site probablities: {sorted_donor_probs}")
        print(f"... Number of Unusually Large Acceptor Sites: {len(probable_acceptor_sites[gene])}")
        sorted_acceptor_probs = probable_acceptor_sites[gene]['acceptor_prob'].sort_values(ascending=False).values
        print(f"..... acceptor site probablities: {sorted_acceptor_probs}")

        # Use site_prob_threshold to guesstimate the number of junctions or 
        # adjust the site_prob_threshold to match a desired number of junctions? 
        print("(generate_splice_junctions_bed_file) site_prob_threshold:", site_prob_threshold)

        # determine_optimal_threshold_for_junction_matching(gene_df['donor_prob'], gene_df['acceptor_prob'], target_junctions=43, initial_threshold=0.9)
 
        # Donor-Acceptor Pairing logic
        bed_rows = \
            process_gene_by_distance(
                gene_df, 
                gene, 
                num_junctions=None,  # estimated_n_junctions, 
                initial_prob_threshold=0.7,   # site_prob_threshold,
                add_probabilities=add_probabilities)

        # bed_rows = \
        #     process_gene_by_knn(
        #         gene_df, gene, 
        #         combined_prob_threshold=combined_prob_threshold,
        #         top_n=estimated_n_junctions, k=3, add_probabilities=add_probabilities)
        # process_gene(gene_df, gene, add_probabilities)
        
        assert len(bed_rows) > 0, f"No junctions found for gene {gene}."

        # Save BED file for each gene
        gene_bed_file_path = get_output_bed_file_path(output_bed_file, output_dir, add_probabilities, gene_id=gene)
        
        gene_bed_df = pd.DataFrame(bed_rows, columns=columns)
        n_junctions_by_gene[gene] = gene_bed_df['name'].nunique()
        
        max_score = gene_bed_df['score'].max()
        if max_score > 0:
            gene_bed_df['score'] = (gene_bed_df['score'] / max_score) * 1000
        
        gene_bed_df.to_csv(gene_bed_file_path, sep='\t', header=False, index=False)
        print(f"[i/o] BED file for gene {gene} saved to {gene_bed_file_path}")

        file_paths[gene] = gene_bed_file_path
        
        # Add to combined list
        combined_bed_rows.extend(bed_rows)

    # Sort combined junctions by coordinates from 5' to 3'
    combined_bed_rows.sort(key=lambda x: (x[0], x[1], x[2]))
    
    # Create a DataFrame for combined BED file
    combined_bed_df = pd.DataFrame(combined_bed_rows, columns=columns)

    # Normalize the score to be between 0 and 1000
    max_score = combined_bed_df['score'].max()
    if max_score > 0:
        combined_bed_df['score'] = (combined_bed_df['score'] / max_score) * 1000
    
    # Save the combined DataFrame to a BED file
    file_paths['combined'] = combined_bed_file_path = \
        get_output_bed_file_path(output_bed_file, output_dir, add_probabilities)
    # combined_bed_file_path = os.path.join(output_dir, output_bed_file)
    combined_bed_df.to_csv(combined_bed_file_path, sep='\t', header=False, index=False)
    print(f"[i/o] Combined BED file saved to {combined_bed_file_path}")

    return (file_paths,
            probable_acceptor_sites,
            probable_acceptor_sites,
            n_junctions_by_gene
    )


def generate_splice_junctions_bed_file_for_transcripts_with_probabilities(
        predictions_df, 
        donor_threshold_percentile=99.9, 
        acceptor_threshold_percentile=99.9, 
        combined_prob_threshold=0.5,
        site_prob_threshold=None,
        top_n=None, 
        output_bed_file='splice_junctions.bed', output_dir='.'):
    """
    Analyze the distribution of splice site probabilities, identify "unusually large" donor and acceptor probabilities, and generate a BED file representation of the junctions.
    
    Parameters:
    - predictions_df (pd.DataFrame): DataFrame with columns 'gene_name', 'position', 'absolute_position', 'donor_prob', 'acceptor_prob', 'neither_prob', 'strand'.
    - donor_threshold_percentile (float): Percentile threshold to define "unusually large" donor probabilities (default: 99.9).
    - acceptor_threshold_percentile (float): Percentile threshold to define "unusually large" acceptor probabilities (default: 99.9).
    - top_n (int): Number of top donor and acceptor probabilities to highlight (default: 50).
    - output_bed_file (str): Name of the output BED file (default: 'splice_junctions.bed').
    - output_dir (str): Directory to save the BED file (default: current directory).
    
    Returns:
    - unusual_donor_sites (pd.DataFrame): DataFrame of positions with unusually large donor probabilities.
    - unusual_acceptor_sites (pd.DataFrame): DataFrame of positions with unusually large acceptor probabilities.
    """
    return generate_splice_junctions_bed_file_for_transcripts(
            predictions_df, 
            donor_threshold_percentile=donor_threshold_percentile, 
            acceptor_threshold_percentile=acceptor_threshold_percentile,
            combined_prob_threshold=combined_prob_threshold,
            site_prob_threshold=site_prob_threshold,
            top_n=top_n,
            add_probabilities=True, 
            output_bed_file=output_bed_file,
            output_dir=output_dir)


def generate_splice_junctions_bed_file_for_transcripts(
        predictions_df, 
        donor_threshold_percentile=99.9,
        acceptor_threshold_percentile=99.9,
        combined_prob_threshold=0.5,  # junction probablity threshold
        site_prob_threshold=None,
        top_n=None,
        add_probabilities=False,
        output_bed_file='splice_junctions.bed', 
        output_dir='.'):
    """
    Analyze the distribution of splice site probabilities, identify "unusually large" donor and acceptor probabilities, and generate BED file representations of the junctions.
    
    Parameters:
    - predictions_df (pd.DataFrame): DataFrame with columns 'gene_name', 'position', 'absolute_position', 'donor_prob', 'acceptor_prob', 'neither_prob', 'strand'.
    - donor_threshold_percentile (float): Percentile threshold to define "unusually large" donor probabilities (default: 99.9).
    - acceptor_threshold_percentile (float): Percentile threshold to define "unusually large" acceptor probabilities (default: 99.9).
    - combined_prob_threshold (float): Minimum combined probability threshold for valid junctions (default: 0.5).
    
    - top_n (dict): Number of top donor and acceptor probabilities to highlight (default: None).
                    If specified, top_n is a dictionary with gene names as keys and top_n values as values.
    
    - output_bed_file (str): Name of the combined output BED file (default: 'splice_junctions.bed').
    - output_dir (str): Directory to save the BED files (default: current directory).
    
    Returns:
    - combined_bed_file_path (str): Path to the combined BED file.
    - probable_donor_sites (dict): Dictionary of where key is gene name and value is a dataframe of probable donor sites.
    - probable_acceptor_sites (dict): Dictionary of where key is gene name and value is a dataframe of probable acceptor sites.
    - n_junctions_tb (dict): Dictionary of where key is transcript ID and value is the number of junctions for that gene.

    """
    def check_transcripts_same_gene(tx_df):
        """
        Check if all transcripts in the DataFrame are associated with the same gene.

        Parameters:
        - tx_df (pd.DataFrame): DataFrame containing transcript information with 'transcript_id' and 'gene_name' columns.

        Returns:
        - bool: True if all transcripts are associated with the same gene, False otherwise.
        """
        # Group by 'transcript_id' and check the number of unique 'gene_name' values in each group
        gene_name_counts = tx_df.groupby('transcript_id')['gene_name'].nunique()
        
        # Check if all groups have only one unique 'gene_name'
        return gene_name_counts.max() == 1

    # Initialize list to store BED file rows
    combined_bed_rows = []
    file_paths = {}
    n_junctions_tb = {}
    site_prob_thresholds = {}
    probable_donor_sites = {}
    probable_acceptor_sites = {}

    top_n_lookup = top_n if top_n is not None else {}

    # Define the columns based on the add_probabilities flag
    columns = ['seqname', 'start', 'end', 'name', 'score', 'strand']
    if add_probabilities:
        columns.extend(['donor_prob', 'acceptor_prob'])

    # Process each gene
    for tx_id, tx_df in predictions_df.groupby('transcript_id'):

        unique_gene_names = tx_df['gene_name'].unique()
        assert len(unique_gene_names) == 1, f"Transcript {tx_id} is associated with multiple genes: {unique_gene_names}"

        gene = tx_df['gene_name'].iloc[0]

        # Test: Find and print the top N donor probabilities
        # ------------------------------------------------
        print(f"(generate_splice_junctions_bed_file) Processing gene {gene} (tx_id={tx_id}) with {len(tx_df)} rows.")
        print(f"... Transcript DataFrame:")
        print(tx_df.head())

        n_high_proba = 50
        top_donor_probs = tx_df['donor_prob'].nlargest(n_high_proba)
        print(f"... Top Donor Probabilities: {top_donor_probs.values}")
        # Find and print the top N acceptor probabilities
        top_acceptor_probs = tx_df['acceptor_prob'].nlargest(n_high_proba)
        print(f"... Top Acceptor Probabilities: {top_acceptor_probs.values}")
        # ------------------------------------------------

        print(f"[info] Gene: {gene} -> Transcript: {tx_id} at percentile thresholds ({donor_threshold_percentile}, {acceptor_threshold_percentile})")
        donor_threshold = np.percentile(tx_df['donor_prob'], donor_threshold_percentile)
        acceptor_threshold = np.percentile(tx_df['acceptor_prob'], acceptor_threshold_percentile)
        probable_donor_sites[tx_id] = tx_df[tx_df['donor_prob'] > donor_threshold]
        probable_acceptor_sites[tx_id] = tx_df[tx_df['acceptor_prob'] > acceptor_threshold]

        print(f"... Number of Unusually Large Donor Sites: {len(probable_donor_sites[tx_id])}")
        sorted_donor_probs = probable_donor_sites[tx_id]['donor_prob'].sort_values(ascending=False).values
        print(f"..... donor site probablities: {sorted_donor_probs}")
        print(f"... Number of Unusually Large Acceptor Sites: {len(probable_acceptor_sites[tx_id])}")
        sorted_acceptor_probs = probable_acceptor_sites[tx_id]['acceptor_prob'].sort_values(ascending=False).values
        print(f"..... acceptor site probablities: {sorted_acceptor_probs}")

        # Determine an optimal splice site probability threshold 
        # - If num_junctions is None, then use the percentile thresholds as a heuristic to 
        #   first infer the number of junctions, from which to infer the site_prob_threshold
        # - If num_junctions is given (`top_n`), then use it to infer the site_prob_threshold
        print(f"[info] Consider number of junctions to infer probability threshold ...")
        num_junctions = top_n_lookup.get(tx_id, None)
        if num_junctions is None:
            top_n_lookup[tx_id] = min(len(probable_donor_sites[tx_id]), len(probable_acceptor_sites[tx_id]))
            print(f"... Guesstimated num_junctions via percentile thresholds {num_junctions}")
        else: 
            print(f"... Using top_n_lookup value for num_junctions: {num_junctions}")

        if site_prob_threshold is None:
            site_prob_threshold, num_junctions_prime = \
                determine_optimal_threshold_for_junction_matching(
                    tx_df['donor_prob'], 
                    tx_df['acceptor_prob'], 
                    target_junctions=num_junctions, 
                    initial_threshold=0.5)  # site_prob_threshold
 
            site_prob_thresholds[tx_id] = site_prob_threshold
            print(f"[info] Given num_junctions: {num_junctions} => site_prob_threshold: {site_prob_threshold}")
            print(f"... num_junctions_prime: {num_junctions_prime} =?= {num_junctions}")

            # Vs 
            # Using just the percentile thresholds
            site_prob_threshold_min = min(donor_threshold, acceptor_threshold)
            info_message = "[info] Percentile thresholds: "
            thresholds_message = f"({donor_threshold}, {acceptor_threshold})"
            site_prob_message = f"=> site_prob_threshold: {site_prob_threshold_min}"
            print(f"{info_message}{thresholds_message} {site_prob_message}")
            # NOTE: Percentage thresholds-derived site probability threshold is probably too low

            # Update probable splice sites based on the inferred site_prob_threshold
            probable_donor_sites[tx_id] = tx_df[tx_df['donor_prob'] > site_prob_threshold]
            probable_acceptor_sites[tx_id] = tx_df[tx_df['acceptor_prob'] > site_prob_threshold]
        else: 
            print(f"[info] Using given site_prob_threshold: {site_prob_threshold}")
            site_prob_thresholds[tx_id] = site_prob_threshold

        # Donor-Acceptor Pairing logic
        bed_rows = \
            process_transcript_by_distance(
                tx_df, 
                tx_id, 
                num_junctions=None,  # if num_junctions is given, then use it to infer the site_prob_threshold
                initial_prob_threshold=site_prob_threshold,   # site_prob_threshold,  # if num_junctions=None, then use this threshold
                add_probabilities=add_probabilities)

        # kNN approach
        # bed_rows = \
        #     process_gene_by_knn(
        #         gene_df, gene, 
        #         combined_prob_threshold=combined_prob_threshold,
        #         top_n=estimated_n_junctions, k=3, add_probabilities=add_probabilities)
        # process_gene(gene_df, gene, add_probabilities)
        
        assert len(bed_rows) > 0, f"No junctions found for gene {gene} (transcript {tx_id})."

        # Save BED file for each gene
        tx_bed_file_path = get_output_bed_file_path(output_bed_file, output_dir, add_probabilities, tx_id=tx_id)
        
        tx_bed_df = pd.DataFrame(bed_rows, columns=columns)
        n_junctions_tb[tx_id] = tx_bed_df['name'].nunique()
        
        max_score = tx_bed_df['score'].max()
        if max_score > 0:
            tx_bed_df['score'] = (tx_bed_df['score'] / max_score) * 1000
        
        tx_bed_df.to_csv(tx_bed_file_path, sep='\t', header=False, index=False)
        print(f"[i/o] BED file for gene {gene} (transcript {tx_id}) saved to {tx_bed_file_path}")

        file_paths[tx_id] = tx_bed_file_path
        
        # Add to combined list
        combined_bed_rows.extend(bed_rows)

    # Sort combined junctions by coordinates from 5' to 3'
    combined_bed_rows.sort(key=lambda x: (x[0], x[1], x[2]))
    
    # Create a DataFrame for combined BED file
    combined_bed_df = pd.DataFrame(combined_bed_rows, columns=columns)

    # Normalize the score to be between 0 and 1000
    max_score = combined_bed_df['score'].max()
    if max_score > 0:
        combined_bed_df['score'] = (combined_bed_df['score'] / max_score) * 1000
    
    # Save the combined DataFrame to a BED file
    file_paths['combined'] = combined_bed_file_path = \
        get_output_bed_file_path(output_bed_file, output_dir, add_probabilities)
    # combined_bed_file_path = os.path.join(output_dir, output_bed_file)
    combined_bed_df.to_csv(combined_bed_file_path, sep='\t', header=False, index=False)
    print(f"[i/o] Combined BED file saved to {combined_bed_file_path}")

    return (file_paths,
            probable_acceptor_sites,
            probable_acceptor_sites,
            n_junctions_tb, 
            site_prob_thresholds
    )


####################################################################################################
# Plotting Functions

def plot_splice_sites(bed_file, threshold=0.5, output_file=None, verbose=1):
    """
    Plots the donor and acceptor splice sites along with the exons for a given transcript.
    
    Parameters:
    - bed_file (str): Path to the BED file with junctions and splice site predictions.
    - threshold (float): Probability threshold to filter donor and acceptor sites.
    - output_file (str, optional): Path to save the plot. If None, the plot will be displayed.

    Example Usage: 
        
        plot_splice_sites('splice_junctions.bed', threshold=0.9)

    """
    # import matplotlib.pyplot as plt
    # import matplotlib.patches as patches

    # Load the BED file into a DataFrame
    columns = ['seqname', 'start', 'end', 'name', 'score', 'strand', 'donor_prob', 'acceptor_prob']
    df = pd.read_csv(bed_file, sep='\t', header=None, names=columns)
    
    # Determine the range for plotting
    plot_start = df['start'].min()
    plot_end = df['end'].max()
    
    # Initialize plot
    fig, ax = plt.subplots(figsize=(15, 5))
    
    # Plot the transcript line
    ax.plot([plot_start, plot_end], [0, 0], color='black', lw=2)
    
    # Plot exons as black boxes
    for _, row in df.iterrows():
        exon_start = row['start']
        exon_end = row['end']
        rect = patches.Rectangle((exon_start, -0.1), exon_end - exon_start, 0.2, linewidth=1, edgecolor='black', facecolor='black')
        ax.add_patch(rect)
    
    # Plot donor (green arrows pointing up) and acceptor (red arrows pointing down) sites
    for _, row in df.iterrows():
        donor_pos = row['start']
        acceptor_pos = row['end']
        
        # Plot donor site if probability is above the threshold
        if row['donor_prob'] >= threshold:
            ax.arrow(donor_pos, -0.05, 0, 0.1, head_width=0.3, head_length=1000, fc='green', ec='green')
        
        # Plot acceptor site if probability is above the threshold
        if row['acceptor_prob'] >= threshold:
            ax.arrow(acceptor_pos, 0.05, 0, -0.1, head_width=0.3, head_length=1000, fc='red', ec='red')
    
    # Set plot limits
    ax.set_xlim([plot_start - 1000, plot_end + 1000])
    ax.set_ylim([-0.3, 0.3])
    ax.axis('off')  # Hide axes

    # Display the plot
    plt.title("SpliceAI-10k score")

    # Display or save the plot
    if output_file:
        if verbose: 
            print("[i/o] Saving plot to", output_file)
        plt.savefig(output_file)
    else:
        plt.show()

    # Close the plot to free memory
    plt.close(fig)


# Todo: Refactor to visual_analyzer
def plot_splice_probabilities_vs_annotation(
    predictions_df,
    annotated_sites,
    output_dir=".",
    show_plot=False,
    save_plot=True,
    plot_format="pdf",
    verbose=1
):
    """
    For each gene in predictions_df, create a plot of position (x-axis) vs 
    SpliceAI probability (y-axis). Overlay known donor/acceptor sites.
    """
    import matplotlib.pyplot as plt
    import os

    is_polars = False
    if isinstance(predictions_df, pl.DataFrame):
        predictions_df = predictions_df.to_pandas()
        is_polars = True

    os.makedirs(output_dir, exist_ok=True)
    all_genes = predictions_df["gene_name"].unique()

    for gene in all_genes:
        sub_df = predictions_df[predictions_df["gene_name"] == gene].copy()
        if sub_df.empty:
            continue

        sub_df.sort_values("position", inplace=True)
        positions = sub_df["position"]
        donor_prob = sub_df["donor_prob"]
        acceptor_prob = sub_df["acceptor_prob"]

        plt.figure(figsize=(12, 6))
        plt.plot(positions, donor_prob, label="Donor Prob", color='blue')
        plt.plot(positions, acceptor_prob, label="Acceptor Prob", color='red')

        # If we have annotated data for this gene:
        if gene in annotated_sites:
            donors = annotated_sites[gene]["donor_positions"]
            acceptors = annotated_sites[gene]["acceptor_positions"]

            # Mark donor sites (vertical dashed lines)
            for dpos in donors:
                plt.axvline(x=dpos, color='blue', linestyle='--', alpha=0.4)
            # Mark acceptor sites (vertical dotted lines)
            for apos in acceptors:
                plt.axvline(x=apos, color='red', linestyle=':', alpha=0.4)

            plt.legend(["Donor Prob","Acceptor Prob","Annotated Donors","Annotated Acceptors"])
        else:
            plt.legend()

        plt.xlabel("Genomic Position (absolute or gene-relative)")
        plt.ylabel("Probability")
        plt.title(f"SpliceAI Probability vs. Annotated Sites: {gene}")

        if save_plot:
            out_path = os.path.join(output_dir, f"{gene}_splice_prob.{plot_format}")
            plt.savefig(out_path, bbox_inches='tight')
            if verbose:
                print(f"[i/o] Saved plot for gene '{gene}' to {out_path}")

        if show_plot:
            plt.show()
        else:
            plt.close()


def plot_splice_probabilities_vs_annotation_enhanced(
    predictions_df,
    annotated_sites,
    output_dir=".",
    show_plot=False,
    save_plot=True,
    plot_format="pdf",
    # Color scheme adjustments
    donor_prob_color="dodgerblue",    # bright-ish blue for predicted donor
    donor_site_color="navy",          # darker blue for true donor sites
    acceptor_prob_color="tomato",     # bright-ish red/orange for predicted acceptor
    acceptor_site_color="firebrick",  # darker red for true acceptor sites
    verbose=1
):
    """
    For each gene in predictions_df, create a 2×2 grid of plots:
      - Left column for DONOR
      - Right column for ACCEPTOR
      - Top row for PREDICTED probabilities
      - Bottom row for TRUE annotated sites

    The top-left axes: predicted donor probabilities
    The bottom-left axes: vertical lines for annotated donor sites
    The top-right axes: predicted acceptor probabilities
    The bottom-right axes: vertical lines for annotated acceptor sites

    Parameters
    ----------
    predictions_df : pd.DataFrame or pl.DataFrame
        Must have columns ["gene_name", "position", "donor_prob", "acceptor_prob"].
        If polars, it will be converted to pandas for plotting syntax.
    annotated_sites : dict
        A dict keyed by gene_name with structure:
          annotated_sites[gene]["donor_positions"] = [...]
          annotated_sites[gene]["acceptor_positions"] = [...]
        Possibly from load_splice_sites_to_dict() or a similar approach.
    output_dir : str
        Where to save the plots (if save_plot=True).
    show_plot : bool
        If True, calls plt.show() after each gene's figure.
    save_plot : bool
        If True, saves each gene's figure in output_dir with format plot_format.
    plot_format : str
        File extension, e.g. "pdf", "png", etc.

    donor_prob_color : str
        Color for the predicted donor probability line (e.g. "dodgerblue").
    donor_site_color : str
        Color for the annotated donor sites (vertical dashed lines). Typically darker than donor_prob_color (e.g. "navy").
    acceptor_prob_color : str
        Color for the predicted acceptor probability line (e.g. "tomato").
    acceptor_site_color : str
        Color for the annotated acceptor sites (vertical dashed lines). Typically darker (e.g. "firebrick").

    verbose : int
        Print progress messages if > 0.
    """
    # import polars as pl

    if isinstance(predictions_df, pl.DataFrame):
        predictions_df = predictions_df.to_pandas()

    os.makedirs(output_dir, exist_ok=True)

    # Get unique genes
    all_genes = predictions_df["gene_name"].unique()
    for gene in all_genes:
        sub_df = predictions_df[predictions_df["gene_name"] == gene].copy()
        if sub_df.empty:
            continue

        # Sort by position
        sub_df.sort_values("position", inplace=True)

        # Extract numeric arrays
        positions = sub_df["position"].values
        donor_prob = sub_df["donor_prob"].values
        acceptor_prob = sub_df["acceptor_prob"].values

        # Make the figure: 2 rows × 2 columns
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8), sharex=True)
        # axes[0,0] = donor probability
        # axes[1,0] = donor annotation lines
        # axes[0,1] = acceptor probability
        # axes[1,1] = acceptor annotation lines

        # ---------- TOP ROW: predicted probabilities ----------
        # Donor prob (top-left)
        ax_donor_pred = axes[0, 0]
        ax_donor_pred.plot(positions, donor_prob, color=donor_prob_color, label="Donor Prob")
        ax_donor_pred.set_ylabel("Probability")
        ax_donor_pred.set_title("Donor Probability (Predicted)")
        ax_donor_pred.grid(True)

        # Acceptor prob (top-right)
        ax_acceptor_pred = axes[0, 1]
        ax_acceptor_pred.plot(positions, acceptor_prob, color=acceptor_prob_color, label="Acceptor Prob")
        ax_acceptor_pred.set_title("Acceptor Probability (Predicted)")
        ax_acceptor_pred.grid(True)

        # ---------- BOTTOM ROW: annotated sites ----------
        # Donor sites (bottom-left)
        ax_donor_annot = axes[1, 0]
        ax_donor_annot.set_title("Donor Sites (Annotated)")
        ax_donor_annot.set_xlabel("Position")
        ax_donor_annot.set_ylabel("Indicator")
        ax_donor_annot.set_yticks([0, 1])
        ax_donor_annot.set_yticklabels(["", ""])  # Hide y labels but keep 0..1 range
        ax_donor_annot.set_ylim(-0.1, 1.1)
        ax_donor_annot.grid(True)

        # Acceptor sites (bottom-right)
        ax_acceptor_annot = axes[1, 1]
        ax_acceptor_annot.set_title("Acceptor Sites (Annotated)")
        ax_acceptor_annot.set_xlabel("Position")
        ax_acceptor_annot.set_ylabel("Indicator")
        ax_acceptor_annot.set_yticks([0, 1])
        ax_acceptor_annot.set_yticklabels(["", ""])
        ax_acceptor_annot.set_ylim(-0.1, 1.1)
        ax_acceptor_annot.grid(True)

        # If annotated data available for this gene, mark them
        if gene in annotated_sites:
            donors_anno = annotated_sites[gene].get("donor_positions", [])
            acceptors_anno = annotated_sites[gene].get("acceptor_positions", [])

            # Mark donor sites with vertical lines
            for dpos in donors_anno:
                ax_donor_annot.axvline(x=dpos, color=donor_site_color, linestyle='--', alpha=0.7)

            # Mark acceptor sites with vertical lines
            for apos in acceptors_anno:
                ax_acceptor_annot.axvline(x=apos, color=acceptor_site_color, linestyle='--', alpha=0.7)

        # Add a main suptitle
        fig.suptitle(f"Gene: {gene}", fontsize=14)

        # Adjust layout
        fig.tight_layout(rect=[0, 0, 1, 0.95])  # leave room for suptitle

        # Save or show
        if save_plot:
            out_path = os.path.join(output_dir, f"{gene}_splice_prob.{plot_format}")
            fig.savefig(out_path, bbox_inches='tight')
            if verbose:
                print(f"[i/o] Saved plot for gene '{gene}' to {out_path}")

        if show_plot:
            plt.show()
        else:
            plt.close(fig)




####################################################################################################

def analyze_splice_sites_highlighting_topn_probabilities(predictions_df, donor_threshold_percentile=99.9, acceptor_threshold_percentile=99.9, top_n=50, save_plot=True, plot_file_name='splice_site_probabilities', plot_format='pdf', output_dir='.', color_palette='deep'):
    """
    Analyze the distribution of splice site probabilities and identify "unusually large" donor and acceptor probabilities.
    
    Parameters:
    - predictions_df (pd.DataFrame): DataFrame with columns 'gene_name', 'position', 'donor_prob', 'acceptor_prob', 'neither_prob', 'strand'.
    - donor_threshold_percentile (float): Percentile threshold to define "unusually large" donor probabilities (default: 99.9).
    - acceptor_threshold_percentile (float): Percentile threshold to define "unusually large" acceptor probabilities (default: 99.9).
    - top_n (int): Number of top donor and acceptor probabilities to highlight (default: 50).
    - save_plot (bool): Whether to save the plot to a file (default: False).
    - plot_file_name (str): Base name of the plot file (default: 'splice_site_probabilities').
    - plot_format (str): Format of the plot file (default: 'png').
    - output_dir (str): Directory to save the plot file (default: current directory).
    - color_palette (str): Color palette to use for the plot (default: 'deep').
    
    Returns:
    - unusual_donor_sites (pd.DataFrame): DataFrame of positions with unusually large donor probabilities.
    - unusual_acceptor_sites (pd.DataFrame): DataFrame of positions with unusually large acceptor probabilities.

    Example usage:
        predictions_df = pd.read_csv('predictions.csv')
        analyze_splice_site_probabilities(predictions_df, 
            top_n=50, save_plot=True, plot_file_name='splice_probabilities', plot_format='pdf', output_dir='/path/to/output', color_palette='deep')
    """
    # Set the color palette
    sns.set_palette(color_palette)
    
    # Calculate thresholds for unusually large probabilities
    donor_threshold = np.percentile(predictions_df['donor_prob'], donor_threshold_percentile)
    acceptor_threshold = np.percentile(predictions_df['acceptor_prob'], acceptor_threshold_percentile)
    print(f"[info] Donor Probability Threshold (>{donor_threshold_percentile}th Percentile): {donor_threshold}")
    print(f"[info] Acceptor Probability Threshold (>{acceptor_threshold_percentile}th Percentile): {acceptor_threshold}")

    # Identify unusually large donor and acceptor probabilities
    unusual_donor_sites = predictions_df[predictions_df['donor_prob'] > donor_threshold]
    unusual_acceptor_sites = predictions_df[predictions_df['acceptor_prob'] > acceptor_threshold]
    
    print(f"Number of Unusually Large Donor Sites: {len(unusual_donor_sites)}")
    print(f"Number of Unusually Large Acceptor Sites: {len(unusual_acceptor_sites)}")

    if top_n is None: 
        print("[info] Top-N donor and acceptor sites not specified.")
        top_n = min(len(unusual_donor_sites), len(unusual_acceptor_sites))
        print(f"[info] Set top_n to the minimum of unusual donor and acceptor sites: {top_n}")

    # Plotting the top-N pairs of donor and acceptor probabilities for each gene
    for gene in predictions_df['gene_name'].unique():
        gene_df = predictions_df[predictions_df['gene_name'] == gene]
        top_donor_sites = gene_df.nlargest(top_n, 'donor_prob')
        top_acceptor_sites = gene_df.nlargest(top_n, 'acceptor_prob')
        
        # Pair donor and acceptor sites to form junctions
        junctions = []
        for _, donor_row in top_donor_sites.iterrows():
            donor_pos = donor_row['position']
            donor_prob = donor_row['donor_prob']
            strand = donor_row['strand']  # Assuming 'strand' column exists in the DataFrame
            
            if strand == '+':
                # Find the nearest acceptor site downstream of the donor site on the positive strand
                downstream_acceptors = top_acceptor_sites[top_acceptor_sites['position'] > donor_pos].sort_values(by='position')
                if not downstream_acceptors.empty:
                    nearest_acceptor = downstream_acceptors.iloc[0]
                    acceptor_pos = nearest_acceptor['position']
                    acceptor_prob = nearest_acceptor['acceptor_prob']
                    junctions.append((donor_pos, acceptor_pos, donor_prob, acceptor_prob))
            elif strand == '-':
                # Find the nearest acceptor site upstream of the donor site on the negative strand
                upstream_acceptors = top_acceptor_sites[top_acceptor_sites['position'] < donor_pos].sort_values(by='position', ascending=False)
                if not upstream_acceptors.empty:
                    nearest_acceptor = upstream_acceptors.iloc[0]
                    acceptor_pos = nearest_acceptor['position']
                    acceptor_prob = nearest_acceptor['acceptor_prob']
                    junctions.append((donor_pos, acceptor_pos, donor_prob, acceptor_prob))
        
        # Create a DataFrame for junctions
        junctions_df = pd.DataFrame(junctions, columns=['donor_pos', 'acceptor_pos', 'donor_prob', 'acceptor_prob'])
        
        plt.figure(figsize=(12, 6))
        
        # Scatter plot for junctions with marker sizes based on combined probabilities
        plt.scatter(junctions_df['donor_pos'], junctions_df['acceptor_pos'], s=(junctions_df['donor_prob'] + junctions_df['acceptor_prob']) * 500, alpha=0.6)
        
        # Add coordinates to each paired data point
        for i, row in junctions_df.iterrows():
            plt.annotate(f"({row['donor_pos']}, {row['acceptor_pos']})", (row['donor_pos'], row['acceptor_pos']), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)
        
        plt.title(f'Top Donor and Acceptor Junctions for {gene}')
        plt.xlabel('Donor Position')
        plt.ylabel('Acceptor Position')
        plt.legend([f'Junctions ({gene})'])
        plt.yscale('log')  # Use log scale for better visualization

        if save_plot:
            plot_path = os.path.join(output_dir, f"{plot_file_name}_{gene}.{plot_format}")
            plt.savefig(plot_path, format=plot_format)
            print(f"[i/o] Plot saved to {plot_path}")
        else:
            plt.show()

    return unusual_donor_sites, unusual_acceptor_sites


def analyze_splice_sites_highlighting_topn_probabilities_v1(
        predictions_df, donor_threshold_percentile=99.9, acceptor_threshold_percentile=99.9, top_n=50, save_plot=True, plot_file_name='splice_site_probabilities', plot_format='pdf', output_dir='.', color_palette='deep'):
    """
    Analyze the distribution of splice site probabilities and identify "unusually large" donor and acceptor probabilities.
    
    Parameters:
    - predictions_df (pd.DataFrame): DataFrame with columns 'gene_name', 'position', 'donor_prob', 'acceptor_prob', 'neither_prob'.
    - donor_threshold_percentile (float): Percentile threshold to define "unusually large" donor probabilities (default: 99.9).
    - acceptor_threshold_percentile (float): Percentile threshold to define "unusually large" acceptor probabilities (default: 99.9).
    - top_n (int): Number of top donor and acceptor probabilities to highlight (default: 50).
    - save_plot (bool): Whether to save the plot to a file (default: False).
    - plot_file_name (str): Base name of the plot file (default: 'splice_site_probabilities').
    - plot_format (str): Format of the plot file (default: 'png').
    - output_dir (str): Directory to save the plot file (default: current directory).
    - color_palette (str): Color palette to use for the plot (default: 'deep').
    
    Returns:
    - unusual_donor_sites (pd.DataFrame): DataFrame of positions with unusually large donor probabilities.
    - unusual_acceptor_sites (pd.DataFrame): DataFrame of positions with unusually large acceptor probabilities.
    """
    # import seaborn as sns

    # Set the color palette
    sns.set_palette(color_palette)
    
    # Calculate thresholds for unusually large probabilities
    donor_threshold = np.percentile(predictions_df['donor_prob'], donor_threshold_percentile)
    acceptor_threshold = np.percentile(predictions_df['acceptor_prob'], acceptor_threshold_percentile)
    print(f"Donor Probability Threshold (>{donor_threshold_percentile}th Percentile): {donor_threshold}")
    print(f"Acceptor Probability Threshold (>{acceptor_threshold_percentile}th Percentile): {acceptor_threshold}")

    # Identify unusually large donor and acceptor probabilities
    unusual_donor_sites = predictions_df[predictions_df['donor_prob'] > donor_threshold]
    unusual_acceptor_sites = predictions_df[predictions_df['acceptor_prob'] > acceptor_threshold]
    
    print(f"Number of Unusually Large Donor Sites: {len(unusual_donor_sites)}")
    print(f"Number of Unusually Large Acceptor Sites: {len(unusual_acceptor_sites)}")

    # Plotting the distribution of probabilities for each gene
    for gene in predictions_df['gene_name'].unique():
        gene_df = predictions_df[predictions_df['gene_name'] == gene]
        top_donor_sites = gene_df.nlargest(top_n, 'donor_prob')
        top_acceptor_sites = gene_df.nlargest(top_n, 'acceptor_prob')
        
        plt.figure(figsize=(12, 6))
        
        # Scatter plot for top donor and acceptor probabilities with marker sizes based on probability values
        plt.scatter(top_donor_sites['position'], top_donor_sites['donor_prob'], s=top_donor_sites['donor_prob']*1000, label=f'Top Donor Probabilities ({gene})', marker='o', alpha=0.6)
        plt.scatter(top_acceptor_sites['position'], top_acceptor_sites['acceptor_prob'], s=top_acceptor_sites['acceptor_prob']*1000, label=f'Top Acceptor Probabilities ({gene})', marker='x', alpha=0.6)
        
        plt.title(f'Top Donor and Acceptor Probabilities for {gene}')
        plt.xlabel('Position')
        plt.ylabel('Probability Value')
        plt.legend()
        plt.yscale('log')  # Use log scale for better visualization

        if save_plot:
            plot_path = os.path.join(output_dir, f"{plot_file_name}_{gene}.{plot_format}")
            plt.savefig(plot_path, format=plot_format)
            print(f"[i/o] Plot saved to {plot_path}")
        else:
            plt.show()

    return unusual_donor_sites, unusual_acceptor_sites


def analyze_splice_sites_highlighting_topn_probabilities_v0(
        predictions_df, 
        donor_threshold_percentile=99.9, 
        acceptor_threshold_percentile=99.9, 
        top_n=50,
        save_plot=True, 
        plot_file_name='splice_site_probabilities', 
        plot_format='pdf', output_dir='.', color_palette='deep'):
    """
    Analyze the distribution of splice site probabilities and identify "unusually large" donor and acceptor probabilities.
    
    Parameters:
    - predictions_df (pd.DataFrame): DataFrame with columns 'gene_name', 'position', 'donor_prob', 'acceptor_prob', 'neither_prob'.
    - donor_threshold_percentile (float): Percentile threshold to define "unusually large" donor probabilities (default: 99.9).
    - acceptor_threshold_percentile (float): Percentile threshold to define "unusually large" acceptor probabilities (default: 99.9).
    - top_n (int): Number of top donor and acceptor probabilities to highlight (default: 50).
    - save_plot (bool): Whether to save the plot to a file (default: False).
    - plot_file_name (str): Name of the plot file (default: 'splice_site_probabilities').
    - plot_format (str): Format of the plot file (default: 'png').
    - output_dir (str): Directory to save the plot file (default: current directory).
    - color_palette (str): Color palette to use for the plot (default: 'deep').
    
    Returns:
    - unusual_donor_sites (pd.DataFrame): DataFrame of positions with unusually large donor probabilities.
    - unusual_acceptor_sites (pd.DataFrame): DataFrame of positions with unusually large acceptor probabilities.
    """
    # Set the color palette
    sns.set_palette(color_palette)
    
    # Calculate thresholds for unusually large probabilities
    donor_threshold = np.percentile(predictions_df['donor_prob'], donor_threshold_percentile)
    acceptor_threshold = np.percentile(predictions_df['acceptor_prob'], acceptor_threshold_percentile)
    print(f"Donor Probability Threshold (>{donor_threshold_percentile}th Percentile): {donor_threshold}")
    print(f"Acceptor Probability Threshold (>{acceptor_threshold_percentile}th Percentile): {acceptor_threshold}")

    # Identify unusually large donor and acceptor probabilities
    unusual_donor_sites = predictions_df[predictions_df['donor_prob'] > donor_threshold]
    unusual_acceptor_sites = predictions_df[predictions_df['acceptor_prob'] > acceptor_threshold]
    
    print(f"Number of Unusually Large Donor Sites: {len(unusual_donor_sites)}")
    print(f"Number of Unusually Large Acceptor Sites: {len(unusual_acceptor_sites)}")

    # Plotting the distribution of probabilities
    plt.figure(figsize=(12, 6))
    
    # Highlight top donor and acceptor probabilities
    for gene in predictions_df['gene_name'].unique():
        gene_df = predictions_df[predictions_df['gene_name'] == gene]
        top_donor_sites = gene_df.nlargest(top_n, 'donor_prob')
        top_acceptor_sites = gene_df.nlargest(top_n, 'acceptor_prob')
        
        # Scatter plot for top donor and acceptor probabilities
        plt.scatter(top_donor_sites['position'], top_donor_sites['donor_prob'], label=f'Top Donor Probabilities ({gene})', marker='o')
        plt.scatter(top_acceptor_sites['position'], top_acceptor_sites['acceptor_prob'], label=f'Top Acceptor Probabilities ({gene})', marker='x')
        
        # Add annotations for gene names and positions
        for i, row in top_donor_sites.iterrows():
            plt.annotate(row['gene_name'], (row['position'], row['donor_prob']), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)
        for i, row in top_acceptor_sites.iterrows():
            plt.annotate(row['gene_name'], (row['position'], row['acceptor_prob']), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)
    
    plt.title('Top Donor and Acceptor Probabilities by Gene')
    plt.xlabel('Position')
    plt.ylabel('Probability Value')
    plt.legend()
    plt.yscale('log')  # Use log scale for better visualization

    if save_plot:
        plot_path = os.path.join(output_dir, f"{plot_file_name}.{plot_format}")
        plt.savefig(plot_path, format=plot_format)
        print(f"[i/o] Plot saved to {plot_path}")
    else:
        plt.show()

    return unusual_donor_sites, unusual_acceptor_sites


def evaluate_top_k_accuracy(predicted_bed, known_bed, k, threshold=2):
    """
    Evaluate Top-k accuracy for predicted junctions against known canonical junctions.

    Parameters:
    - predicted_bed (pd.DataFrame): DataFrame with predicted junctions ['seqname', 'start', 'end', 'name', 'score', 'strand'].
    - known_bed (pd.DataFrame): DataFrame with known junctions ['seqname', 'start', 'end', 'name', 'score', 'strand'].
    - k (int): Number of top predictions to consider (e.g., 5 for Top-5 accuracy).
    - threshold (int): Distance threshold to consider two junctions as a match (default: 2).

    Returns:
    - top_k_accuracy (float): Top-k accuracy.
    - matched_junctions (list): List of matched junctions.

    Example Usage: 

        predicted_bed = pd.read_csv('predicted_junctions.bed', sep='\t', names=['seqname', 'start', 'end', 'name', 'score', 'strand'])
        known_bed = pd.read_csv('known_junctions.bed', sep='\t', names=['seqname', 'start', 'end', 'name', 'score', 'strand'])

        top_k = 5  # For Top-5 accuracy
        top_k_accuracy, matched_junctions = evaluate_top_k_accuracy(predicted_bed, known_bed, top_k)
        print(f"Top-{top_k} Accuracy: {top_k_accuracy * 100:.2f}%")
        print(f"Matched Junctions: {matched_junctions}")
    """
    # Sort predicted junctions by score in descending order
    top_predicted = predicted_bed.sort_values(by='score', ascending=False).head(k)

    # Convert known junctions to a set for quick lookup
    known_set = set()
    for _, row in known_bed.iterrows():
        chrom, start, end = row['seqname'], row['start'], row['end']
        known_set.add((chrom, start, end))

    # Evaluate Top-k accuracy
    matched_junctions = []
    for _, pred_row in top_predicted.iterrows():
        chrom, pred_start, pred_end = pred_row['seqname'], pred_row['start'], pred_row['end']
        
        # Check if any known junction matches within the threshold
        for known_chrom, known_start, known_end in known_set:
            if chrom == known_chrom and abs(pred_start - known_start) <= threshold and abs(pred_end - known_end) <= threshold:
                matched_junctions.append((chrom, pred_start, pred_end))
                break

    # Calculate Top-k accuracy
    top_k_accuracy = len(matched_junctions) / k

    return top_k_accuracy, matched_junctions

####################################################################################################


def filter_transcripts(known_bed, transcript_ids):
    """
    Filter the rows in known_bed to keep only the relevant transcripts.

    Parameters:
    - known_bed (pd.DataFrame): DataFrame containing junction BED file data.
    - transcript_ids (str or list): A single transcript ID (str) or a list of transcript IDs (list).

    Returns:
    - pd.DataFrame: Filtered DataFrame containing only the relevant transcripts.
    """
    if isinstance(transcript_ids, str):
        transcript_ids = [transcript_ids]

    # Extract the transcript ID from the 'name' column
    known_bed['transcript_id'] = known_bed['name'].apply(lambda x: x.split('_')[0])

    # Filter the DataFrame to keep only the relevant transcripts
    filtered_bed = known_bed[known_bed['transcript_id'].isin(transcript_ids)]

    # Drop the temporary 'transcript_id' column
    filtered_bed = filtered_bed.drop(columns=['transcript_id'])

    return filtered_bed


def save_prediction_dataframe_v0(predictions_df, output_file, use_polars=True, **kargs):
    """
    Save the prediction DataFrame to a CSV file.
    
    Parameters:
    - predictions_df (pd.DataFrame or pl.DataFrame): DataFrame containing gene name, position, donor, acceptor, neither probabilities, and strand.
    - output_file (str): Path to the output CSV file.
    - use_polars (bool): Whether to use polars for saving the DataFrame. Default is True.
    - output_dir (str, optional): Directory to save the output CSV file. Default is the current directory.
    - sep (str, optional): Delimiter to use in the CSV file (default is '\t').

    Example usage:
        predictions_df = predict_splice_sites_for_genes(seq_df, models, context)
        save_prediction_dataframe(predictions_df, "predictions.csv", output_dir='/path/to/output')
    """
    sep = kargs.get('sep', '\t')
    output_dir = kargs.get('output_dir', '.')
    output_path = os.path.join(output_dir, output_file)
    
    if use_polars:
        predictions_df.write_csv(output_path, separator=sep)
        print(f"Prediction DataFrame saved to {output_path} using polars")
    else:
        predictions_df.to_csv(output_path, sep=sep, index=False)
        print(f"Prediction DataFrame saved to {output_path} using pandas")


def save_prediction_dataframe(predictions_df, output_file, chunk_start=None, chunk_end=None, **kargs):
    """
    Save the prediction DataFrame to a CSV file, with support for chunked data.
    
    Parameters:
    - predictions_df (pd.DataFrame or pl.DataFrame): DataFrame containing gene name, position, donor, acceptor, neither probabilities, and strand.
    - output_file (str): Base name of the output file (chromosome-based files will append chunk range).
    - chunk_start (int, optional): The start index of the chunk being processed (used for chunked saving).
    - chunk_end (int, optional): The end index of the chunk being processed (used for chunked saving).
    - output_dir (str, optional): Directory to save the output CSV file. Default is the current directory.
    - sep (str, optional): Delimiter to use in the CSV file (default is '\t').

    Example usage:
        save_prediction_dataframe(predictions_df, "predictions.csv", chunk_start=1, chunk_end=1000)
    """
    sep = kargs.get('sep', '\t')
    output_dir = kargs.get('output_dir', '.')
    
    # Adjust output file to track the chunk being saved (if chunking is used)
    if chunk_start is not None and chunk_end is not None:
        base_name, ext = os.path.splitext(output_file)
        output_file = f"{base_name}_chunk_{chunk_start}_{chunk_end}{ext}"
    
    output_path = os.path.join(output_dir, output_file)
    
    if isinstance(predictions_df, pl.DataFrame):
        predictions_df.write_csv(output_path, separator=sep)
        print(f"Prediction DataFrame saved to {output_path} using polars")
    else:
        predictions_df.to_csv(output_path, sep=sep, index=False)
        print(f"Prediction DataFrame saved to {output_path} using pandas")



def load_bed_file_to_dataframe(bed_file_path, column_names=None):
    """
    Load a BED file into a pandas DataFrame.

    Parameters:
    - bed_file_path (str): Path to the BED file.
    - column_names (list, optional): List of column names for the DataFrame. If None, default names will be used.

    Returns:
    - pd.DataFrame: DataFrame containing the BED file data.
    """
    # Define default column names if none are provided
    if column_names is None:
        column_names = ['seqname', 'start', 'end', 'name', 'score', 'strand']
    # NOTE: Other possible formats for SpliceAI BED files:
    #       ['seqname', 'start', 'end', 'name', 'score', 'strand', 'donor_prob', 'acceptor_prob']
    
    # Load the BED file into a DataFrame
    bed_df = pd.read_csv(bed_file_path, sep='\t', header=None, names=column_names)
    
    return bed_df


def load_prediction_dataframe_v0(input_file, use_polars=True, **kargs):
    """
    Load the prediction DataFrame from a CSV file.
    
    Parameters:
    - input_file (str): Path to the input CSV file.
    - use_polars (bool): Whether to use polars for loading the DataFrame. Default is True.
    - sep (str, optional): Delimiter to use in the CSV file (default is '\t').
    - input_dir (str, optional): Directory to load the input CSV file from. Default is the current directory.
    
    Returns:
    - predictions_df (pd.DataFrame or pl.DataFrame): DataFrame containing gene name, position, donor, acceptor, neither probabilities, and strand.
    
    Example usage:
        predictions_df = load_prediction_dataframe("predictions.csv", input_dir='/path/to/input', use_polars=True)
    """
    sep = kargs.get('sep', '\t')
    input_dir = kargs.get('input_dir', '.')
    input_path = os.path.join(input_dir, input_file)
    
    if use_polars:
        predictions_df = pl.read_csv(input_path, separator=sep)  # Use polars for faster loading
        print(f"Prediction DataFrame loaded from {input_path} using polars")
    else:
        predictions_df = pd.read_csv(input_path, sep=sep)  # Use pandas for loading
        print(f"Prediction DataFrame loaded from {input_path} using pandas")
    
    return predictions_df


def load_prediction_dataframe(input_file, chunk_start=None, chunk_end=None, use_polars=True, **kargs):
    """
    Load the prediction DataFrame from CSV files, optionally for specific chunks.
    
    Parameters:
    - input_file (str): Base name of the input CSV file (without chunk details).
    - chunk_start (int, optional): Start of the chunk range. Default is None.
    - chunk_end (int, optional): End of the chunk range. Default is None.
    - use_polars (bool): Whether to use polars for loading the DataFrame. Default is True.
    - sep (str, optional): Delimiter to use in the CSV file (default is '\t').
    - input_dir (str, optional): Directory to load the input CSV files from. Default is the current directory.
    
    Returns:
    - predictions_df (pl.DataFrame or pd.DataFrame): DataFrame containing gene name, position, donor, acceptor, neither probabilities, and strand.
    
    Example usage:
        predictions_df = load_prediction_dataframe("predictions_chr1.tsv", chunk_start=1, chunk_end=1000, input_dir='/path/to/input')
    """
    sep = kargs.get('sep', '\t')
    input_dir = kargs.get('input_dir', '.')
    base_name, ext = os.path.splitext(input_file)
    
    # If chunking is used, modify the input file name
    if chunk_start is not None and chunk_end is not None:
        input_file = f"{base_name}_chunk_{chunk_start}_{chunk_end}{ext}"
    
    input_path = os.path.join(input_dir, input_file)
    
    if use_polars:
        predictions_df = pl.read_csv(input_path, separator=sep)  # Use polars for faster loading
        print(f"Prediction DataFrame loaded from {input_path} using polars")
    else:
        predictions_df = pd.read_csv(input_path, sep=sep)  # Use pandas for loading
        print(f"Prediction DataFrame loaded from {input_path} using pandas")
    
    return predictions_df


def save_as_bed(predictions_df, output_file_prefix, **kargs):
    """
    Save predictions as individual BED files for donor, acceptor, and neither probabilities.
    
    Parameters:
    - predictions_df (pd.DataFrame): DataFrame containing gene name, position, donor, acceptor, neither probabilities, and strand.
    - output_file_prefix (str): Prefix for the output BED files.
    - output_dir (str, optional): Directory to save the output BED files. Default is the current directory.

    Example usage:
       save_as_bed(predictions_df, "splice_predictions")

    Memo: 
    * Why multiple BED Files? 
       This function creates separate BED files for each type of probability: (donor_prob, acceptor_prob, neither_prob). 
       Each file would have the score in the fifth column, and you could then load all three files simultaneously in IGV.

    """
    output_dir = kargs.get('output_dir', '.')

    for prob_type in ['donor_prob', 'acceptor_prob', 'neither_prob']:
        bed_file = os.path.join(output_dir, f"{output_file_prefix}_{prob_type}.bed")

        print("[info] Writing BED file (prob_type={}):\n{}\n".format(prob_type, bed_file))
        with open(bed_file, 'w') as f:
            for idx, row in predictions_df.iterrows():
                score = int(row[prob_type] * 1000)  # Scale the probability to an integer score between 0-1000
                f.write(f"{row['gene_name']}\t{row['position']-1}\t{row['position']}\t{prob_type}\t{score}\t{row['strand']}\n")


def save_as_wig(predictions_df, output_file_prefix, **kargs):
    """
    Save predictions as individual WIG files for donor, acceptor, and neither probabilities.
    
    Parameters:
    - predictions_df (pd.DataFrame): DataFrame containing gene name, position, donor, acceptor, neither probabilities.
    - output_file_prefix (str): Prefix for the output WIG files.
    - output_dir (str, optional): Directory to save the output BED files. Default is the current directory.

    Example usage:
        save_as_wig(predictions_df, "splice_predictions")

    Memo: 
        WIG/BigWig Format: Consider using WIG or BigWig formats for continuous data visualization, 
        which IGV supports well.
    """
    output_dir = kargs.get('output_dir', '.')

    for prob_type in ['donor_prob', 'acceptor_prob', 'neither_prob']:
        wig_file = os.path.join(output_dir, f"{output_file_prefix}_{prob_type}.wig")

        print("[info] Writing WIG file (prob_type={}):\n{}\n".format(prob_type, wig_file))
        with open(wig_file, 'w') as f:
            f.write(f"track type=wiggle_0 name=\"{prob_type}\" description=\"{prob_type} probabilities\"\n")
            
            current_chrom = None
            current_start = None
            step = 1
            
            for idx, row in predictions_df.iterrows():
                if row['gene_name'] != current_chrom or row['position'] != current_start + step:
                    # Update current chrom and start position
                    current_chrom = row['gene_name']
                    current_start = row['position']
                    # Write the new fixedStep header
                    f.write(f"fixedStep chrom={current_chrom} start={current_start} step={step}\n")
                
                # Write the probability value for the current position
                f.write(f"{row[prob_type]}\n")
                current_start += step


def save_as_wig_with_log_transform(predictions_df, output_file_prefix, log_base=10, epsilon=1e-10, **kargs):
    """
    Save predictions as individual WIG files for donor, acceptor, and neither probabilities with log transformation.
    
    Parameters:
    - predictions_df (pd.DataFrame): DataFrame containing gene name, position, donor, acceptor, neither probabilities.
    - output_file_prefix (str): Prefix for the output WIG files.
    - log_base (float): Base of the logarithm to use (default is 10).
    - epsilon (float): Small constant added to avoid log(0) (default is 1e-10).
    """
    output_dir = kargs.get('output_dir', '.')

    for prob_type in ['donor_prob', 'acceptor_prob', 'neither_prob']:
        wig_file = os.path.join(output_dir, f"{output_file_prefix}_{prob_type}.wig")

        print("Writing WIG file (prob_type={}) with log(P):\n{}\n".format(prob_type, wig_file))
        with open(wig_file, 'w') as f:
            f.write(f"track type=wiggle_0 name=\"{prob_type}\" description=\"{prob_type} probabilities (log-transformed)\"\n")
            
            current_chrom = None
            current_start = None
            step = 1
            
            for idx, row in predictions_df.iterrows():
                if row['gene_name'] != current_chrom or row['position'] != current_start + step:
                    current_chrom = row['gene_name']
                    current_start = row['position']
                    f.write(f"fixedStep chrom={current_chrom} start={current_start} step={step}\n")
                
                log_transformed_value = -np.log10(row[prob_type] + epsilon)
                f.write(f"{log_transformed_value}\n")
                current_start += step


def display_gene_sequence_lengths(df):
    """
    Display the length of the DNA sequence for each gene in the DataFrame.

    Parameters:
    - df (pd.DataFrame or pl.DataFrame): DataFrame containing gene names and sequences with columns 'gene_name' and 'sequence'.
    """
    if isinstance(df, pd.DataFrame):
        # Handle Pandas DataFrame
        for index, row in df.iterrows():
            gene_name = row['gene_name']
            if 'transcript_id' in row:
                gene_name = f"{gene_name} ({row['transcript_id']})"
            sequence_length = len(row['sequence'])
            print(f"> Gene: {gene_name}, Sequence Length: {sequence_length}, Strand: {row.get('strand', '?')}")
    elif isinstance(df, pl.DataFrame):
        # Handle Polars DataFrame
        for row in df.iter_rows(named=True):  # Iterate over the rows of the Polars DataFrame using iter_rows with named=True
            gene_name = row['gene_name']
            if 'transcript_id' in row:
                gene_name = f"{gene_name} ({row['transcript_id']})"
            sequence_length = len(row['sequence'])
            print(f"> Gene: {gene_name}, Sequence Length: {sequence_length}, Strand: {row.get('strand', '?')}")
    else:
        raise TypeError("Unsupported DataFrame type. Please provide a Pandas or Polars DataFrame.")


def get_number_of_junctions(bed_df, id_col='name'):
    """
    Helper function to get the number of unique junctions in a BED DataFrame.
    
    Parameters:
    - bed_df (pd.DataFrame): DataFrame containing BED file data.
    - id_col (str): Column name for the junction ID (default: 'name').
    
    Returns:
    - int: Number of unique junctions.
    """
    return bed_df[id_col].nunique()


def junction_prediction_workflow(): 
    """

    
    Related
    -------
    * generate_splice_junctions_bed_file(): 
        Analyze the distribution of splice site probabilities, identify "unusually large" donor and acceptor probabilities, 
        and generate BED file representations of the junctions.

    """

    from meta_spliceai.splice_engine.utils_fs import (
        read_splice_sites, 
        demo_extract_sequences
    )
    from meta_spliceai.splice_engine.utils_bio import (
        load_sequences, 
        load_chromosome_sequence,
        load_sequences_by_chromosome, 
        load_chromosome_sequence_streaming
    )
    from meta_spliceai.splice_engine.extract_genomic_features import (
        extract_splice_sites_workflow, 
        transcript_sequence_retrieval_workflow,
        gene_sequence_retrieval_workflow, 
    )
    
    from meta_spliceai.splice_engine.extract_junctions import (
        extract_splice_junctions_workflow
    )

    # from meta_spliceai.splice_engine.evaluate_models import (
    #     evaluate_donor_site_predictions, 
    #     evaluate_acceptor_site_predictions, 
    #     evaluate_splice_site_predictions, 
    #     evaluate_splice_site_errors,
    #     save_performance_df, 
    #     save_error_analysis,
    # )

    separator = kargs.get('separator', kargs.get('sep', '\t'))

    local_dir = '/path/to/meta-spliceai/data/ensembl/'
    src_dir = '/path/to/meta-spliceai/data/ensembl/'
    eval_dir = '/path/to/meta-spliceai/data/ensembl/spliceai_eval'

    gtf_file = "/path/to/meta-spliceai/data/ensembl/Homo_sapiens.GRCh38.112.gtf"  # Replace with your GTF file path
    genome_fasta = os.path.join(src_dir, "Homo_sapiens.GRCh38.dna.primary_assembly.fa") 
    
    assert os.path.exists(gtf_file)
    assert os.path.exists(genome_fasta)
    db_file = os.path.join(local_dir, "annotations.db")
    annotation_file_path = os.path.join(local_dir, "annotations_all_transcripts.csv")
    
    print("[info] Transcipt Annotations (exon, CDS, 5'UTR, and 3'UTR annotations):")
    # annot_df = pd.read_csv(annotation_file_path, sep=',')
    # print(annot_df.head())

    extract_splice_junctions_workflow(
        data_prefix=local_dir, 
        gtf_file=gtf_file)

    pass


def load_spliceai_ensemble():
    """
    Load the SpliceAI ensemble models.
    """
    from keras.models import load_model
    from pkg_resources import resource_filename
    # from spliceai.utils import one_hot_encode
    # import spliceai  # pip install spliceai

    # Load SpliceAI models
    # context = 10000
    paths = ('models/spliceai{}.h5'.format(x) for x in range(1, 6))
    models = [load_model(resource_filename('spliceai', x)) for x in paths]

    return models


def workflow_gene_specific_spliceai_prediction(
    data_prefix,
    gtf_filename="Homo_sapiens.GRCh38.112.gtf",
    fasta_filename="Homo_sapiens.GRCh38.dna.primary_assembly.fa",
    genes=None,
    local_dir=".",
    splice_prediction_file="splice_predictions.tsv",
    do_predict=True,
    context=10000,
    plot_results=True,
    plot_dir="plots",
    **kargs
):
    """
    A workflow to:
      1) Extract gene sequences for given genes from GTF/FASTA
      2) Predict splice site probabilities with SpliceAI (or load from file)
      3) Extract donor/acceptor sites from the GTF using 'extract_splice_sites_for_all_genes'
      4) Plot splice probabilities vs. these annotated splice sites for validation.

    Parameters
    ----------
    data_prefix : str
        Directory containing GTF/FASTA/DB files.
    gtf_filename : str
        GTF filename (default: "Homo_sapiens.GRCh38.112.gtf").
    fasta_filename : str
        FASTA filename (default: "Homo_sapiens.GRCh38.dna.primary_assembly.fa").
    db_filename : str
        The gffutils FeatureDB file (default: "Homo_sapiens.GRCh38.112.db").
        If it doesn't exist, we can create it from the GTF.
    genes : list or set
        Genes to process (e.g. {"STMN2","UNC13A"}).
    local_dir : str
        Where to save intermediate files (e.g., extracted sequences, predictions).
    splice_prediction_file : str
        Basename of the prediction output file (e.g. "splice_predictions.tsv").
    do_predict : bool
        If True, run SpliceAI on the extracted sequences; if False, load existing predictions.
    context : int
        SpliceAI context size for prediction (e.g. 10000).
    plot_results : bool
        If True, plot the splice probabilities vs annotated donor/acceptor sites.
    plot_dir : str
        Where to save the plots.
    consensus_window : int
        The +/- window around the splice site. (Default: 1)

    Optional Parameters
    ------------------- 
    verbose : int
        Print progress if > 0.
    overwrite: 

    Returns
    -------
    pd.DataFrame
        The final predictions DataFrame with columns like:
        ['gene_name','position','absolute_position','donor_prob','acceptor_prob','neither_prob','strand']
    """
    from .utils_df import get_n_unique
    from .utils_bio import load_sequences
    from .workflow_utils import (
        long_to_gene_dict,   # Convert the "lanky" predictions_df from predict_splice_sites_for_genes() to a gene-centric DataFrame
        gene_long_to_single_row  # Convert the "lanky" predictions_df to a single-row-per-gene DataFrame
    )

    from .extract_genomic_features import (
        get_annotated_splice_sites, 
        load_splice_sites_to_dict
    )
    from .extract_gene_sequences import (
        run_workflow_extract_gene_sequences, 
        build_gene_id_to_name_map
    )
    from .seq_model_explainer import (
        extract_analysis_sequences_with_probabilities, 
        plot_alignment_with_probabilities
    )

    verbose = kargs.get('verbose', 1)
    overwrite = kargs.get('overwrite', False)

    os.makedirs(local_dir, exist_ok=True)

    # 1) Extract gene sequences => results in a TSV with columns: 'gene_name','seqname','start','end','strand','sequence'
    seq_file_path = os.path.join(local_dir, "gene_sequence.tsv")
    
    # Check if the sequence file already exists
    if os.path.exists(seq_file_path) and not overwrite:
        if verbose:
            print(f"[info] Gene sequences already exist at {seq_file_path} ...")
    else:
        if verbose:
            print("[action] Extracting gene sequences from GTF/FASTA ...")
        
        run_workflow_extract_gene_sequences(
            data_prefix=data_prefix,
            gtf_filename=gtf_filename,
            fasta_filename=fasta_filename,
            genes=genes,
            output_dir=local_dir,
            output_basename="gene_sequence",
            output_format="tsv",
            include_columns=['seqname', 'start', 'end', 'strand'],
            verbose=verbose
        )

    # 2) Load sequences into DataFrame
    if verbose:
        print(f"[action] Loading extracted sequences from {seq_file_path}")
    seq_df = load_sequences(seq_file_path, format='tsv')  # your existing function
    print_with_indent(f"Column: {list(seq_df.columns)}", indent_level=1) 

    if verbose:
        # seq_df['gene_name'].nunique()
        n_genes = get_n_unique(seq_df, 'gene_name')
        print(f"[info] seq_df has {n_genes} unique genes.")
        print(seq_df.head())

        # Sequence lengths
        # display_gene_sequence_lengths(seq_df)

    # 3) SpliceAI predictions
    start_time = time.time()
    splice_prediction_path = os.path.join(local_dir, splice_prediction_file)

    if do_predict:
        # Load SpliceAI models
        # (Adapt this to however you're loading your SpliceAI models)
        # e.g., models = [load_model(...) for ... in ...]
        models = load_spliceai_ensemble()  
        if verbose:
            print_emphasized(f"[action] Predicting splice sites with SpliceAI ensemble (context={context})...")
        
        predictions_df = \
            predict_splice_sites_for_genes(
                seq_df, models, context, start_time=start_time, output_format='pandas')
        
        # Save predictions
        save_prediction_dataframe(predictions_df, output_file=splice_prediction_file, output_dir=local_dir)
        if verbose:
            print(f"[i/o] Saved predictions to {splice_prediction_path}")
    else:
        if verbose:
            print(f"[action] Loading existing predictions from {splice_prediction_path}")
        predictions_df = load_prediction_dataframe(splice_prediction_file, local_dir)
    
    print_with_indent(f"Column: {list(predictions_df.columns)}", indent_level=1) 
    # Columns: ['seqname', 'gene_id', 'gene_name', 'position', 'absolute_position', 'gene_start', 'gene_end', 'donor_prob', 'acceptor_prob',
    #           'neither_prob', 'strand']

    # 4) Analyze/visualize probabilities
    if plot_results:
        # 4a) Analyze distributions (like you already do)
        out_dir_analysis = os.path.join(local_dir, plot_dir)
        os.makedirs(out_dir_analysis, exist_ok=True)
        analyze_splice_site_probabilities(
            predictions_df,
            donor_threshold_percentile=99.9,
            acceptor_threshold_percentile=99.9,
            save_plot=True,
            plot_file_name='splice_site_probabilities_distribution',
            plot_format='pdf',
            output_dir=out_dir_analysis
        )

        # 4b) Overlap with annotated sites (the new piece)
        #     We get the annotation from the GTF => exon boundaries
        # if verbose:
        #     print("[action] Loading annotated splice sites for visualization ...")
        # annotated_sites = get_annotated_splice_sites(
        #     gtf_file=os.path.join(data_prefix, gtf_filename),
        #     gene_names=genes
        # )

        retrieve_splice_sites(data_prefix, gtf_filename=gtf_filename, format='tsv')

        if verbose:
            print("[action] Building gene_id -> gene_name map for dictionary lookup ...")
        gene_map = \
            build_gene_id_to_name_map(
                gtf_file_path=os.path.join(data_prefix, gtf_filename),
                remove_version=True,
                output_path=os.path.join(data_prefix, "gene_id_name_map.csv")  # optional
            )
        # your function that returns e.g. { "ENSG00000...":"STMN2", ... }

        if verbose:
            print("[action] Loading splice-site annotation into dictionary ...")
        splice_sites_dict = load_splice_sites_to_dict(
            splice_site_file=os.path.join(data_prefix, "splice_sites.tsv"),
            key_by="gene_name",   # or "gene_id"
            gene_id_to_name_map=gene_map
        )
        # => { gene_name -> { "chrom": [...], "strand": [...],
        #                     "donor_positions": [...], "acceptor_positions": [...],
        #                     "transcripts": [...] } }

        # Finally, plot probabilities vs annotated sites
        if verbose:
            print_emphasized("[action] Plotting SpliceAI probabilities vs annotated donor/acceptor ...")
        # plot_splice_probabilities_vs_annotation(
        #     predictions_df,
        #     annotated_sites=splice_sites_dict,  
        #     output_dir=out_dir_analysis,
        #     show_plot=False,
        #     save_plot=True,
        #     plot_format='pdf',
        #     verbose=verbose
        # )

        plot_splice_probabilities_vs_annotation_enhanced(
            predictions_df,
            annotated_sites=splice_sites_dict, 
            output_dir=out_dir_analysis,
            show_plot=False,
            save_plot=True,
            plot_format="pdf",
            donor_prob_color="dodgerblue",    # bright-ish blue for predicted donor
            donor_site_color="navy",          # darker blue for true donor sites
            acceptor_prob_color="tomato",     # bright-ish red/orange for predicted acceptor
            acceptor_site_color="firebrick",  # darker red for true acceptor sites
            verbose=1
        )

    ##########################################################################
    print_emphasized(f"Retrieving predicted splice sites, given input dir: {local_dir}")
    position_df = retrieve_predicted_splice_sites(local_dir)
    print_with_indent(f"Column: {list(position_df.columns)}", indent_level=1)

    seq_prob_df = \
        extract_analysis_sequences_with_probabilities(
            sequence_df=seq_df,
            position_df=position_df,
            predictions_df=predictions_df,
            window_size=200,
            include_empty_entries=False
        )
    print_with_indent(f"Column: {list(seq_prob_df.columns)}", indent_level=1)
    #   In each row of seq_prob_df has: 
    #     - All columns from position_df + 
    #     - 'window_start' (0-based index within the gene's sequence)
    #     - 'window_end'
    #     - 'sequence' (the substring from window_start:window_end)
    #     - 'positions_list' (list of int, local positions for each base in the window)
    #     - 'donor_prob_list' (list of float)
    #     - 'acceptor_prob_list' (list of float)
    #     - 'neither_prob_list' (list of float)

    output_dir = os.path.join(local_dir, plot_dir)
    # os.path.join(local_dir, "spliceai_explanatory")

    grouped = seq_prob_df.group_by('gene_id')  # Group by gene_id
    for gene_id, group in tqdm(grouped, desc="Visualizing SpliceAI Predictions", total=n_genes):

        # Create a new directory for each gene
        gene_dir = os.path.join(output_dir, gene_id)
        os.makedirs(gene_dir, exist_ok=True)

        # Aggregator for just this transcript
        top_token_counts_local = {error_label: defaultdict(int), correct_label: defaultdict(int)}
        num_examples_local     = {error_label: 0, correct_label: 0}

        # tp_rows = group[group['label'] == 0]  # or however you store your labels

        for i, row in group.iterrows():
            # label_str = determine_label_str(row['label'])   # e.g. "FP","FN","TP","TN"
            sequence = row['sequence']
            # tx_id = row[col_tid]
            pred_type = row['pred_type']

            if pred_type.lower() == 'tp': 
            
                annotation = row.to_dict()
                # keys: ['gene_id', 'transcript_id', 'chrom', 'position', 'label', 
                #        'score', 'splice_type', 'strand', 'window_end', 'window_start', 'sequence']
                
                # annotation['error_type'] = error_label
                splice_type = annotation['splice_type']
                position = annotation['position']

                if splice_type == 'donor':
                    prob_vector = annotation['donor_prob_list']
                elif splice_type == 'acceptor':
                    prob_vector = annotation['acceptor_prob_list']
                else: 
                    prob_vector = annotation['neither_prob_list']

                    continue # skip neither sites

                output_path = os.path.join(gene_dir, f"{gene_id}_{splice_type}_{position}.pdf")
                plot_alignment_with_probabilities(
                    sequence,
                    prob_vector=prob_vector,
                    splice_site_index=None,
                    annotation=annotation,
                    output_path=output_path,
                    color_map="plasma",
                    figsize=(20, 6),
                    show_tokens=False,
                    tokenized_sequence=None,
                    attention_weights=None,
                    top_k=10,
                )

    return predictions_df


def demo_spliceai_explanatory_workflow(): 

    data_prefix = "/path/to/meta-spliceai/data/ensembl"
    local_dir = "/path/to/meta-spliceai/data/ensembl/test"
    os.makedirs(local_dir, exist_ok=True)

    gtf_file = "Homo_sapiens.GRCh38.112.gtf"
    fasta_file = "Homo_sapiens.GRCh38.dna.primary_assembly.fa"

    target_genes = {"STMN2", "UNC13A", }  # "SOD1", "TARDBP"
    splice_prediction_file = "splice_predictions_als.tsv"

    overwrite = False
    do_predict = True
    
    workflow_gene_specific_spliceai_prediction(
        data_prefix,  # Read large genomics files from this directory (on blob storage)
        gtf_filename=gtf_file,
        fasta_filename=fasta_file,
        genes=target_genes,
        local_dir=local_dir,
        splice_prediction_file=splice_prediction_file,
        do_predict=do_predict,
        context=10000,
        plot_results=True,
        plot_dir="plots",
        verbose=1, 
        overwrite=overwrite
    )


def demo_junction_prediction_workflow(): 
    """

    Updates: 
    -------
    workflow_gene_specific_spliceai_prediction()
    demo_spliceai_explanatory_workflow()
    """

    from meta_spliceai.utils.bio_utils import (
        parse_gtf_for_genes, 
        extract_genes_from_gtf, 
        extract_gene_sequences, 
        load_sequences
    )
    from meta_spliceai.splice_engine.extract_gene_sequences import  (
        run_workflow_extract_gene_sequences
    )

    # from keras.models import load_model
    # from pkg_resources import resource_filename
    # from spliceai.utils import one_hot_encode
    # from meta_spliceai.utils.bio_utils import demo_extract_gene_sequences

    # local_dir = "/path/to/meta-spliceai/data/ensembl/ALS"
    data_prefix = "/path/to/meta-spliceai/data/ensembl"
    local_dir = "/path/to/meta-spliceai/data/ensembl/test"
    os.makedirs(local_dir, exist_ok=True)

    splice_prediction_file = "splice_predictions.tsv"
    splice_prediction_path = os.path.join(local_dir, splice_prediction_file)

    gene_names = {"STMN2", "UNC13A", "MMEL1", "PANK4"}

    print("[action] Extracting gene sequences from GTF file ...")
    # Use utils.bio_utils to extract gene sequences 
    # demo_extract_gene_sequences(gene_names)
    run_workflow_extract_gene_sequences(
        data_prefix,
        gtf_filename="Homo_sapiens.GRCh38.112.gtf",
        fasta_filename="Homo_sapiens.GRCh38.dna.primary_assembly.fa",
        genes=gene_names,
        output_dir=local_dir,
        output_basename="gene_sequence",
        output_format="tsv",
        include_columns=['seqname', 'start', 'end', 'strand'],
        verbose=1
    )
    # 
    # Output: 
    # 
    # [parse] Found n=2 genes
    # ... shape(genes_df): (2, 5)
    # [i/o] Saving gene sequences to:
    # <local_dir>/gene_sequence.tsv

    format = 'tsv'
    file_path = os.path.join(local_dir, f"gene_sequence.{format}")
    seq_df = load_sequences(file_path, format=format)

    # header: 'gene_name', 'sequence', ... 
    print("[info] cols(seq_df):", list(seq_df.columns))
    
    # Drop the 'sequence' column and display the first few rows
    print(seq_df.drop(columns=['sequence']).head())  # Pandas 
    # print(seq_df.drop('sequence').head())  # Polars

    n_genes = seq_df['gene_name'].nunique()  # Pandas 
    # n_genes = seq_df.select('gene_name').n_unique()  # Polars

    print(f"[info] Found n={n_genes} genes")
    display_gene_sequence_lengths(seq_df)

    # sys.exit()
    ########################################################

    # Load SpliceAI models
    context = 10000
    paths = ('models/spliceai{}.h5'.format(x) for x in range(1, 6))
    models = [load_model(resource_filename('spliceai', x)) for x in paths]

    action = 'predict' # options: 'predict', 'load'

    # Generate splice site predictions
    if action == 'predict': 
        predictions_df = predict_splice_sites_for_genes(seq_df, models, context, output_format='pandas')
        # NOTE: 'predictions_df' includes the following columns:
        #   'gene_name', 'position', 'absolute_position', 'donor_prob', 'acceptor_prob', 'neither_prob', 'strand'
    else:
        print("[info] Loading predictions from file: ", splice_prediction_path)
        predictions_df = load_prediction_dataframe(input_file=splice_prediction_file, input_dir=local_dir)

    # Save predictions as CSV file
    save_prediction_dataframe(predictions_df, output_file=splice_prediction_file, output_dir=local_dir)
    
    # Display the results
    print("[info] Predictions DataFrame:")
    print(predictions_df.head())
    print("... cols(predictions_df):", list(predictions_df.columns))
    assert predictions_df['gene_id'].nunique() == n_genes, "Mismatch in number of genes: {} vs {}".format(predictions_df['gene_id'].nunique(), n_genes)

    plot_dir = os.path.join(local_dir, "plot")

    # Analyze splice site probabilities
    analyze_splice_site_probabilities(
        predictions_df, 
        donor_threshold_percentile=99.9, 
        acceptor_threshold_percentile=99.9, 
        save_plot=True, 
        output_dir=plot_dir)

    for color_palette in ['deep', 'bright', 'muted', 'pastel', 'dark', 'colorblind']:
        analyze_splice_sites_highlighting_topn_probabilities(
            predictions_df, 
            donor_threshold_percentile=99.9, 
            acceptor_threshold_percentile=99.9, 
            top_n=None,
            save_plot=True, 
            plot_file_name=f'splice_site_probabilities-topN-{color_palette}', 
            plot_format='pdf', 
            output_dir=plot_dir, 
            color_palette=color_palette
        )
    
    output_dir = plot_dir
    output_bed_file = "splice_predictions.bed"

    p_th = 0.01

    top_n_lookup = {}
    for gene in gene_names:
        canonical_junctions_path = os.path.join(local_dir, f"canonical/{gene}/{gene}_canonical_junctions.bed")
        known_bed = load_bed_file_to_dataframe(canonical_junctions_path)
        top_n_lookup[gene] = get_number_of_junctions(known_bed)

    bed_file_paths, probable_donors, probable_acceptors, n_junctions = \
        generate_splice_junctions_bed_file(
            predictions_df, 
            donor_threshold_percentile=99.9, 
            acceptor_threshold_percentile=99.9, 
            combined_prob_threshold=p_th, 
            site_prob_threshold=p_th, 
            # top_n=top_n_lookup, 
            output_bed_file=output_bed_file, 
            output_dir=output_dir)

    proba_bed_file_paths, probable_donors, probable_acceptors, n_junctions = \
        generate_splice_junctions_bed_file_with_probabilities(
            predictions_df, 
            donor_threshold_percentile=99.9, 
            acceptor_threshold_percentile=99.9, 
            combined_prob_threshold=p_th, 
            site_prob_threshold=p_th,
            # top_n=top_n_lookup, 
            output_bed_file=output_bed_file,  # file name will be adjusted automatically
            output_dir=output_dir)

    for gene in gene_names: 
        # splice_junctions_path = get_output_bed_file_path(output_bed_file, output_dir, add_probabilities=False, gene_id=gene)
        splice_junctions_path = bed_file_paths[gene]

        # Load predicted splice junctions from a BED file
        print("[info] Gene={} => predicted junction BED file: {}".format(gene, os.path.basename(splice_junctions_path)))
        predicted_bed = load_bed_file_to_dataframe(splice_junctions_path)
        # NOTE: Columns: ['seqname', 'start', 'end', 'name', 'score', 'strand']

        # Load known splice junctions from a BED file
        canonical_junctions_path = os.path.join(local_dir, f"canonical/{gene}/{gene}_canonical_junctions.bed")
        known_bed = load_bed_file_to_dataframe(canonical_junctions_path)

        top_k = get_number_of_junctions(known_bed)
        print("... found n={} known junctions for gene {}".format(top_k, gene))

        # Evaluate Top-k accuracy
        top_k_accuracy, matched_junctions = evaluate_top_k_accuracy(predicted_bed, known_bed, top_k, threshold=100)
        print(f"Top-{top_k} Accuracy: {top_k_accuracy * 100:.2f}%")
        print(f"Matched Junctions: {matched_junctions}")

    # top_k = 5  # For Top-5 accuracy
    # top_k_accuracy, matched_junctions = evaluate_top_k_accuracy(predicted_bed, known_bed, top_k)
    # print(f"Top-{top_k} Accuracy: {top_k_accuracy * 100:.2f}%")
    # print(f"Matched Junctions: {matched_junctions}")

    # Save predictions as BED files
    # file_path = os.path.join(local_dir, "splice_predictions.bed")
    save_as_bed(predictions_df, output_file_prefix="ssp_als_genes", output_dir=local_dir)
    
    save_as_wig(predictions_df, output_file_prefix="ssp_als_genes", output_dir=local_dir)

    save_as_wig_with_log_transform(
        predictions_df,
        output_file_prefix="ssp_als_genes_log_transform",
        output_dir=local_dir,
        log_base=10, epsilon=1e-10)


def process_chromosome(chr, output_file, format, mode, action, local_dir):
    seq_df = load_chromosome_sequence(output_file, chr, format=format)
    print("[info] Found n={} genes in chromosome {}".format(seq_df['gene_id'].nunique(), chr))

    n_trpts = None  # number of transcripts; don't apply to gene mode
    if mode in ['tx', 'transcript']:
        transcript_sequence_df = seq_df
        n_trpts = seq_df['transcript_id'].nunique()
        print(f"[info] columns(transcript_sequence_df):\n{transcript_sequence_df}\n")
        print(f"[info] Found n={n_trpts} transcripts")
        print(f"[info] Found n={transcript_sequence_df['seqname'].nunique()} seqnames")
        print(f"... seqnames: {transcript_sequence_df['seqname'].unique()}")

        transcript_sequence_df = transcript_sequence_df.with_columns(pl.col('seqname').cast(pl.Utf8))
        
    print("Step 4: Predict splice sites for the transcript sequences ...")
    splice_prediction_file = f"splice_predictions_{chr}.tsv"
    splice_prediction_path = os.path.join(local_dir, splice_prediction_file)

    if (action == 'predict') or not os.path.exists(splice_prediction_path): 
        context = 10000
        paths = ('models/spliceai{}.h5'.format(x) for x in range(1, 6))
        models = [load_model(resource_filename('spliceai', x)) for x in paths]

        if mode in ['gene']:
            predictions_df = predict_splice_sites_for_genes(seq_df, models, context=context)
        else:
            predictions_df = predict_splice_sites(seq_df, models, context=context)

        print("[i/o] Saving predictions as:", splice_prediction_file)
        save_prediction_dataframe(predictions_df, output_file=splice_prediction_file, output_dir=local_dir)
    else:
        print("[info] Loading predictions from file: ", splice_prediction_path)
        predictions_df = load_prediction_dataframe(input_file=splice_prediction_file, input_dir=local_dir)

    assert len(predictions_df) > 0, "No predictions found!"
    print("[info] shape(predictions_df):", predictions_df.shape)

    if n_trps is not None:
        assert n_trpts == predictions_df['transcript_id'].nunique(), "Mismatch in number of transcripts"
    
    print(predictions_df.head())
    print("... cols(predictions_df):", list(predictions_df.columns))
    
    return predictions_df


def predict_splice_sites_for_chromosomes(output_file, format, mode, action, local_dir):
    """
    Predict splice sites for all given genes or transcripts across multiple chromosomes.

    Parameters:
    - output_file (str): Path to the output file containing chromosome sequences.
    - format (str): Format of the input file (e.g., 'csv', 'tsv').
    - mode (str): Mode of operation ('gene' or 'tx'/'transcript').
    - action (str): Action to perform ('predict' or 'load').
    - local_dir (str): Directory to save or load prediction files.

    Returns:
    - None
    """
    chromosomes = [str(i) for i in range(1, 23)] + ['X', 'Y', 'MT']

    for chr in tqdm(chromosomes, desc="Processing chromosomes"):
        predictions_df = process_chromosome(chr, output_file, format, mode, action, local_dir)
        # You can add any additional processing for each chromosome here if needed

    return


########################################################


def check_available_memory():
    """
    Check the available memory and print it out.
    
    Returns:
    - available_memory (float): Available memory in GB.
    """
    memory_info = psutil.virtual_memory()
    available_memory = memory_info.available / (1024 ** 3)  # Convert bytes to GB
    total_memory = memory_info.total / (1024 ** 3)  # Convert total memory to GB
    print(f"[memory] Available memory: {available_memory:.2f} GB out of {total_memory:.2f} GB")
    return available_memory


def adjust_chunk_size(chunk_size, seq_len_avg):
    """
    Adjust the chunk size based on available memory and sequence length.
    
    Parameters:
    - chunk_size (int): Initial chunk size.
    - seq_len_avg (int): Average length of sequences in the dataset.

    Returns:
    - adjusted_chunk_size (int): Adjusted chunk size to fit into available memory.
    """
    available_memory = check_available_memory()

    # Rough estimate of memory needed per sequence, assuming each sequence is about 4 bytes per nucleotide.
    memory_per_sequence = seq_len_avg * 4 / (1024 ** 3)  # Memory per sequence in GB

    # Estimate how many sequences can fit into half of the available memory
    target_memory_usage = available_memory / 2
    estimated_chunk_size = int(target_memory_usage / memory_per_sequence)

    adjusted_chunk_size = min(chunk_size, estimated_chunk_size)
    print(f"[adjust_chunk_size] Adjusted chunk size: {adjusted_chunk_size}, based on available memory.")
    
    return adjusted_chunk_size


def process_and_evaluate_splice_site_predictions(
        chromosomes, seq_df_path, format, local_dir, ss_annot_df, mode='gene'):
    action = 'predict'  # options: 'predict', 'load'
    test_mode = False

    total_genes_processed = 0
    total_chunk_time = 0
    start_time = time.time()

    n_chr_processed = 0
    chromosomes = [str(i) for i in range(1, 23)] + ['X', 'Y', 'MT']
    
    # Initialize an empty DataFrame to gather all chromosome results
    full_performance_df = pl.DataFrame()

    for chr in tqdm(chromosomes, desc="Processing chromosomes"):
        # Load sequence data using streaming mode for the chromosome
        lazy_seq_df = load_chromosome_sequence_streaming(seq_df_path, chr, format=format)

        # Initialize chunk size and necessary variables
        chunk_size = 500  # Starting chunk size
        seq_len_avg = 50000  # Assume an average sequence length for now
        num_genes = lazy_seq_df.select(pl.col('gene_id').n_unique()).collect().item()
        n_chunk_processed = 0

        for chunk_start in range(0, num_genes, chunk_size):
            chunk_end = min(chunk_start + chunk_size, num_genes)

            # Track the start time for each chunk
            chunk_start_time = time.time()

            # Adjust the chunk size based on memory availability
            chunk_size = adjust_chunk_size(chunk_size, seq_len_avg)

            # Filter the LazyFrame to process only the current chunk
            seq_chunk = lazy_seq_df.slice(chunk_start, chunk_size).collect()

            print(f"[info] Processing genes {chunk_start + 1} to {chunk_end} out of {num_genes} genes")

            # Predict splice sites for the chunk of gene sequences
            splice_prediction_file = f"splice_predictions_{chr}_chunk_{chunk_start + 1}_{chunk_end}.tsv"
            splice_prediction_path = os.path.join(local_dir, splice_prediction_file)

            if (action == 'predict') or not os.path.exists(splice_prediction_path):
                context = 10000
                paths = ('models/spliceai{}.h5'.format(x) for x in range(1, 6))
                models = [load_model(resource_filename('spliceai', x)) for x in paths]

                if mode == 'gene':
                    predictions = \
                        predict_splice_sites_for_genes(seq_chunk, models, context=context, start_time=start_time)
                else:
                    predictions = predict_splice_sites(seq_chunk, models, context=context)
            else:
                # Load existing predictions if the file exists (optional based on action)
                # predictions = load_prediction_dataframe(input_file=splice_prediction_file, ...)
                pass

            assert len(predictions) > 0, "No predictions found!"
            print(f"[info] Processed {len(predictions)} predictions for chunk {chunk_start + 1} to {chunk_end}")

            # Model evaluation: Evaluate predictions and gather performance metrics for the current chunk
            performance_df_chunk = evaluate_splice_site_predictions(ss_annot_df, predictions, threshold=0.9, consensus_window=5)

            # Append chunk performance to the full chromosome performance DataFrame
            full_performance_df = full_performance_df.vstack(performance_df_chunk)

            # Save predictions and performance for the current chunk (optional)
            splice_performance_file = f"splice_performance_{chr}_chunk_{chunk_start + 1}_{chunk_end}.tsv"
            splice_performance_path = os.path.join(local_dir, splice_performance_file)
            performance_df_chunk.write_csv(splice_performance_path)
            print(f"[i/o] Saved performance to: {splice_performance_file}")

            n_chunk_processed += 1
            total_genes_processed += len(seq_chunk)

            # Track the chunk processing time and estimate total time-to-completion
            chunk_end_time = time.time()
            total_chunk_time += chunk_end_time - chunk_start_time
            print(f"[info] Chunk {n_chunk_processed} processed in {calculate_duration(chunk_start_time)}")

    # Save the final, full performance DataFrame for all chromosomes
    full_performance_file = os.path.join(local_dir, "full_splice_performance.tsv")
    full_performance_df.write_csv(full_performance_file)
    print(f"[i/o] Full performance saved to: {full_performance_file}")

    return full_performance_df


def find_optimal_threshold(metrics_df, metric='f1_score'):
    """
    Find the optimal threshold based on a user-defined metric.

    Parameters:
    - metrics_df (pl.DataFrame): DataFrame containing performance metrics across thresholds.
    - metric (str): The metric to optimize (default is 'f1_score').

    Returns:
    - optimal_threshold (float): The threshold that optimizes the specified metric.
    - best_metric_value (float): The best value of the specified metric.
    """
    # Find the row with the maximum value of the specified metric
    best_row = metrics_df.sort(metric, descending=True).head(1)
    optimal_threshold = best_row['threshold'].item()
    best_metric_value = best_row[metric].item()

    return optimal_threshold, best_metric_value


# def setup_logger(log_file_path):
#     """
#     Set up a logger that prints to the console and writes to a log file.

#     Parameters:
#     - log_file_path (str): The file path to log warnings.
#     """
#     logger = logging.getLogger('gene_id_logger')
#     logger.setLevel(logging.DEBUG)

#     # Create handlers
#     console_handler = logging.StreamHandler()
#     file_handler = logging.FileHandler(log_file_path)

#     # Create formatters and add them to the handlers
#     formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
#     console_handler.setFormatter(formatter)
#     file_handler.setFormatter(formatter)

#     # Add handlers to the logger
#     logger.addHandler(console_handler)
#     logger.addHandler(file_handler)

#     return logger


def optimize_threshold_for_splice_site_prediction(
        annotation_path, seq_df_path, *, data_dir=None, 
        num_genes_to_sample=1000, context=10000, 
        metric='f1_score',
        chromosomes=None, 
        consensus_window=3, 
        paths=None, 
        **kargs):
    """
    Determine the optimal threshold for SpliceAI predictions using a subset of sequence data.
    
    Parameters:
    - annotation_path (str): Path to the annotation file for splice sites.
    - seq_df_path (str): Path to the prepared sequence dataset.
    - data_dir (str): Directory to store temporary files.
    - num_genes_to_sample (int): Number of genes to sample across chromosomes for ROC generation.
    - context (int): Context length for SpliceAI model predictions (default: 10000).
    - metric (str): The metric to optimize (default: 'f1_score').
    - chromosomes (list): List of chromosomes to process. If None, defaults to all chromosomes.
    - consensus_window (int): Number of nucleotides to include before and after the splice site.
    - paths (list): List of paths to SpliceAI models. Defaults to SpliceAI models 1 to 6.
    - kargs: Additional keyword arguments.
    
    Returns:
    - optimal_threshold (float): The optimal threshold for classifying splice site predictions.
    """
    # from meta_spliceai.splice_engine.utils_fs import read_splice_sites

    start_time = time.time()
    # from keras.models import load_model
    # from pkg_resources import resource_filename
    # from spliceai.utils import one_hot_encode
    from sklearn.metrics import precision_recall_curve, roc_curve, f1_score
    
    from meta_spliceai.splice_engine.utils import (
        calculate_duration, 
        format_time, 
    )
    from meta_spliceai.splice_engine.evaluate_models import (
        filter_sequence_df, 
        filter_annotations_df, 
        evaluate_splice_site_predictions, 
        evaluate_performance_across_thresholds, 
        subset_sequence_data, 
        create_sequence_subset_streaming, 
        plot_roc_curve,
        plot_prc_curve,
        plot_roc_curve_for_donor_site_predictions
    )

    def test_unique_gene_ids(subset_annot_df, subset_seq_df, logger):
        """
        Test for unique gene IDs in annotation and sequence dataframes and log warnings if there is a mismatch.

        Parameters:
        - subset_annot_df (pl.DataFrame): The annotation dataframe.
        - subset_seq_df (pl.DataFrame): The sequence dataframe.
        - logger (logging.Logger): The logger to use for logging warnings.
        """
        unique_gids_in_annot = set(subset_annot_df.select('gene_id').to_series().to_list())
        unique_gids_in_seq = set(subset_seq_df.select('gene_id').to_series().to_list())

        # Find gene IDs in subset_df but not in annot_df
        missing_gene_ids = set(unique_gids_in_seq) - set(unique_gids_in_annot)

        print_emphasized(f"Number of unique genes in sequence subset: {len(unique_gids_in_seq)}")
        print_with_indent(f"Number of unique genes in annotation subset: {len(unique_gids_in_annot)}", indent_level=1)
        print_with_indent(f"Number of missing gene IDs (in seq but not in annot): {len(missing_gene_ids)}", indent_level=1)

        if unique_gids_in_annot != unique_gids_in_seq:
            warning_message = "[warning] Mismatch in gene IDs between annotation and sequence data"
            difference_message = f"Difference: {unique_gids_in_annot.symmetric_difference(unique_gids_in_seq)}"

            print_emphasized(warning_message)
            print_with_indent(difference_message, indent_level=1)

            # Log the warning messages
            logger.warning(warning_message)
            logger.warning(f"Number of unique genes in sequence subset: {len(unique_gids_in_seq)}")
            logger.warning(f"Number of unique genes in annotation subset: {len(unique_gids_in_annot)}")
            logger.warning(f"Number of missing gene IDs (in seq but not in annot): {len(missing_gene_ids)}")
            logger.warning(difference_message)

    def print_metrics_df(metrics_df):
        # Configure Polars to print all columns and rows for metrics_df
        pl.Config.set_tbl_cols(metrics_df.shape[1])
        pl.Config.set_tbl_rows(metrics_df.shape[0])  # Set to the number of rows in metrics_df

        # Print the metrics_df
        print("[info] Performance dataframe: shape={}".format(metrics_df.shape))
        print(metrics_df)

        # Restore the default configuration
        pl.Config.set_tbl_cols(10)  # Default value for columns
        pl.Config.set_tbl_rows(10)  # Default value for rows

    
    if data_dir is None:
        data_dir = os.getcwd()

    # Set default paths if not provided
    print_emphasized("[info] Loading SpliceAI models...")
    if paths is None:
        paths = ('models/spliceai{}.h5'.format(x) for x in range(1, 6))
    models = [load_model(resource_filename('spliceai', x)) for x in paths]

    print("[info] Subsetting sequence data and annotations for performance estimation...")

    print(f"> Reading annotated splice sites at:\n{annotation_path}\n")
    annot_df = read_splice_sites(annotation_path, separator='\t', dtypes=None)
    # A Polars DataFrame: ['chrom', 'start', 'end', 'strand', 'site_type', 'gene_id', 'transcript_id']

    # Subsample gene_ids from annot_df
    sampled_gene_ids = annot_df.select('gene_id').unique().sample(n=num_genes_to_sample)  # Randomly sample gene IDs
    sampled_gene_ids_set = set(sampled_gene_ids['gene_id'].to_list())
    subset_annot_df = annot_df.filter(pl.col('gene_id').is_in(sampled_gene_ids_set))

    # Subset sequence data based on the sampled gene IDs
    # Method 1: Only works for small datasets for which only one sequence file is needed
    # subset_seq_df = filter_sequence_df(seq_df_path, sampled_gene_ids, format='parquet')
    # A Polars DataFrame: ['gene_id', 'seqname', 'start', 'end', 'strand', 'sequence']

    if chromosomes is None:
        chromosomes = [str(i) for i in range(1, 23)] + ['X', 'Y',]  # 'MT']

    # Method 2: Create a subset of sequence data for the specified number of genes
    # subset_seq_df = create_sequence_subset_streaming(
    #     seq_df_path, 
    #     num_genes=num_genes_to_sample,
    #     chromosomes=chromosomes, verbose=1)
    # Method 3: Create a subset of sequence data for the specified genes (subsampled from the annnotations)
    subset_seq_df = \
        subset_sequence_data(
            seq_df_path, 
            target_gene_ids=sampled_gene_ids_set, 
            chromosomes=chromosomes)
    # A Polars DataFrame: ['gene_id', 'seqname', 'start', 'end', 'strand', 'sequence'] 

    # Step 3: Predict splice sites for the subset
    print_emphasized("[info] Generating predictions for a subset of genes to determine opt. threshold ...")
    subset_predictions = predict_splice_sites_for_genes(subset_seq_df, models, context=context)
    print("[info] Predictions for subset complete.")

    # ------------------------------------------------------
    logger = setup_logger(os.path.join(data_dir, 'gene_id_mismatch.log'))
    test_unique_gene_ids(subset_annot_df, subset_seq_df, logger)
    # unique_gene_ids = subset_seq_df.select('gene_id').unique()

    unique_gids_in_annot = set(subset_annot_df.select('gene_id').to_series().to_list())
    unique_gids_in_seq = set(subset_seq_df.select('gene_id').to_series().to_list())

    # Check for mismatch and print a warning if there is one
    if unique_gids_in_annot != unique_gids_in_seq:
        print_emphasized("[warning] Mismatch in gene IDs between annotation and sequence data")
        print_with_indent(f"Number of unique gene IDs in annotation: {len(unique_gids_in_annot)}", indent_level=1)
        print_with_indent(f"Number of unique gene IDs in sequence: {len(unique_gids_in_seq)}", indent_level=1)
        print_with_indent(f"Difference: {unique_gids_in_annot.symmetric_difference(unique_gids_in_seq)}", indent_level=2)

        # Take the intersection of the two sets to proceed
        common_gene_ids = unique_gids_in_annot.intersection(unique_gids_in_seq)

        # Filter the DataFrames to only include the common gene IDs
        subset_annot_df = subset_annot_df.filter(pl.col('gene_id').is_in(common_gene_ids))
        subset_seq_df = subset_seq_df.filter(pl.col('gene_id').is_in(common_gene_ids))
    else:
        common_gene_ids = unique_gids_in_annot

    # NOTE: The mismatch in gene IDs between annotation and sequence data can happen. 
    #       If so, proceed with the intersection of the two sets.
    # ------------------------------------------------------

    print_emphasized(f"I. Evaluating performance across thresholds based on metric={metric} ...")

    # Evaluate performance across thresholds
    metrics_df = evaluate_performance_across_thresholds(
        subset_annot_df, subset_predictions, 
        thresholds=np.arange(0.0, 1.05, 0.05), consensus_window=consensus_window
    )

    # Configure Polars to display all columns
    print_metrics_df(metrics_df)

    # Step 4: Find the optimal threshold based on the specified metric
    print("[info] Finding optimal threshold based on the specified metric...")
    optimal_threshold, best_metric_value = find_optimal_threshold(metrics_df, metric=metric)
    print_emphasized(f"[info] Optimal Threshold: {optimal_threshold:.2f}, Best Score (metric={metric}): {best_metric_value:.4f}")

    # Step 5: Plot the ROC curve
    fpr_list = metrics_df['fpr'].to_list()
    tpr_list = metrics_df['tpr'].to_list()
    output_path = os.path.join(data_dir, "ROC_curve.pdf")
    plot_roc_curve(fpr_list, tpr_list, save_path=output_path)

    # Step 6: Plot the PRC curve
    precision_list = metrics_df['precision'].to_list()
    recall_list = metrics_df['recall'].to_list()
    output_path = os.path.join(data_dir, "PRC_curve.pdf")
    plot_prc_curve(precision_list, recall_list, save_path=output_path)

    # Step 7: Evaluate splice site predictions at the optimal threshold
    print("[info] Evaluating splice site predictions for the subset at the optimal threshold...")
    evaluate_splice_site_predictions(
        subset_annot_df, subset_predictions, threshold=optimal_threshold, consensus_window=consensus_window)

    print(f"[info] Completed performance estimation and threshold optimization")
    print_emphasized(f"[info] Optimal Threshold: {optimal_threshold:.2f}, Best Score (metric={metric}): {best_metric_value:.4f}")
    print(f"[time] Overall time taken so far: {calculate_duration(time.time() - start_time)}")

    ########################################################

    # Alternative approach is to find the threshold by which the number of predicted junctions is equal to the number of known junctions
    
    print_emphasized("II. Evaluating performance using the threshold that matches the number of known junctions ...")

    # determine_optimal_threshold_for_junction_matching(donor_probs, acceptor_probs, target_junctions=43, initial_threshold=0.9)

    return optimal_threshold


def demo_optimize_threshold_for_splice_site_prediction(): 

    from meta_spliceai.splice_engine.utils_fs import read_splice_sites
    
    from meta_spliceai.splice_engine.extract_genomic_features import (
        extract_splice_sites_workflow, 
        transcript_sequence_retrieval_workflow,
        gene_sequence_retrieval_workflow, 
    )

    src_dir = '/path/to/meta-spliceai/data/ensembl/'  # Big files on Azure blob storage
    local_dir = '/path/to/meta-spliceai/data/ensembl/'
    eval_dir = '/path/to/meta-spliceai/data/ensembl/spliceai_eval'

    gtf_file = os.path.join(src_dir, "Homo_sapiens.GRCh38.112.gtf")  # Replace with your GTF file path
    genome_fasta = os.path.join(src_dir, "Homo_sapiens.GRCh38.dna.primary_assembly.fa") 
    
    assert os.path.exists(gtf_file)
    assert os.path.exists(genome_fasta)
    db_file = os.path.join(local_dir, "annotations.db")
    annotation_file_path = os.path.join(local_dir, "annotations_all_transcripts.csv")
    
    # print("[info] Transcipt Annotations (exon, CDS, 5'UTR, and 3'UTR annotations):")
    # annot_df = pd.read_csv(annotation_file_path, sep=',')
    # print(annot_df.head())
    
    # print_emphasized("[action] Extract annotations comprising exon, CDS, 5'UTR, and 3'UTR and splice sites for all transcripts")
    ss_annot_df = retrieve_splice_sites(src_dir, gtf_filename=gtf_file, format='tsv')

    # annotations_df
    print(f"[info] Splice-site datafame: shape={ss_annot_df.shape}")
    print(ss_annot_df.head())

    print_emphasized("[action] Extract DNA sequences")
    mode = 'gene'
    seq_type = 'full'  # options: 'minmax', 'full'
    format = 'parquet'
    n_gene_samples = 500
    metric = 'f1_score'
    consensus_window = 3

    seq_df_path = os.path.join(local_dir, f"gene_sequence.{format}")
    if mode in ['gene', ]:
        output_file = f"gene_sequence.{format}" 
        if seq_type == 'minmax':
            output_file = f"gene_sequence_minmax.{format}"

        seq_df_path = os.path.join(local_dir, output_file)
        # gene_sequence_retrieval_workflow(
        #     gtf_file, genome_fasta, gene_tx_map=None, output_file=seq_df_path, mode=seq_type)
    else: 
        seq_df_path = os.path.join(local_dir, f"tx_sequence.{format}")
        # transcript_sequence_retrieval_workflow(
        #     gtf_file, genome_fasta, gene_tx_map=None, output_file=seq_df_path)

    optimal_threshold = \
        optimize_threshold_for_splice_site_prediction(
            splice_sites_file_path,  # path to the splice site annotation
            seq_df_path,  # path to the sequence data
            data_dir=eval_dir, 
            num_genes_to_sample=n_gene_samples, 
            context=10000, 
            metric=metric,
            chromosomes=None, 
            consensus_window=consensus_window)


    return optimal_threshold


def error_analysis_workflow(**kargs):
    """
    Perform error analysis on precomputed splice site prediction results.

    This workflow function assumes that the error analysis files are already computed and available from the
    `splice_site_prediction_workflow()`. It performs further analysis on these precomputed results.

    Parameters:
    - separator (str, optional): The separator used in the input files. Default is '\t'.
    - local_dir (str, optional): The local directory path where the data is stored. 
        Default is '<homedir>/meta-spliceai/data/ensembl/'.
    - src_dir (str, optional): The source directory path where the data is stored. 
        Default is '/path/to/meta-spliceai/data/ensembl/'.
    - eval_dir (str, optional): The evaluation directory path where the results are stored. 
        Default is '<homedir>/meta-spliceai/data/ensembl/spliceai_eval'.
    - gtf_file (str, optional): Path to the GTF file containing gene annotations. 
        Default is '/path/to/meta-spliceai/data/ensembl/Homo_sapiens.GRCh38.112.gtf'.
    - genome_fasta (str, optional): Path to the reference genome FASTA file. 
        Default is '/path/to/meta-spliceai/data/ensembl/Homo_sapiens.GRCh38.dna.primary_assembly.fa'.
    - db_file (str, optional): Path to the annotations database file. 
        Default is '<homedir>/meta-spliceai/data/ensembl/annotations.db'.

    - mode (str, optional): Mode of sequence extraction. Options are 'gene' or 'transcript'. Default is 'gene'.
    - seq_type (str, optional): Type of sequence extraction. Options are 'minmax' or 'full'. Default is 'full'.
    - format (str, optional): File format for saving the sequences. Default is 'parquet'.
    - threshold (float, optional): Threshold for evaluating predictions. Default is 0.5.
    - consensus_window (int, optional): Window size for consensus evaluation. Default is 3.
    - error_window (int, optional): Window size for error analysis. Default is 500.

    Returns:
    - None

    Example usage:
        error_analysis_workflow(
            separator='\t',
            local_dir='/path/to/meta-spliceai/data/ensembl/',
            src_dir='/path/to/meta-spliceai/data/ensembl/',
            eval_dir='/path/to/meta-spliceai/data/ensembl/spliceai_eval',
            gtf_file='/path/to/meta-spliceai/data/ensembl/Homo_sapiens.GRCh38.112.gtf',
            genome_fasta='/path/to/meta-spliceai/data/ensembl/Homo_sapiens.GRCh38.dna.primary_assembly.fa',
            db_file='/path/to/meta-spliceai/data/ensembl/annotations.db',
            mode='gene',
            seq_type='full',
            format='parquet',
            threshold=0.5,
            consensus_window=3,
            error_window=500
        )

    Notes:
    - This function assumes that the error analysis files are already computed and available from the `splice_site_prediction_workflow()`.
    - The function will print information about the extracted gene sequences and the output file path.
    - The output file will be saved in the specified format (default is 'parquet').
    """
    from meta_spliceai.splice_engine.workflow_utils import align_and_append
    from meta_spliceai.splice_engine.analysis_utils import (
        check_and_subset_invalid_transcript_ids
    )
    # from meta_spliceai.splice_engine.model_evaluator import ModelEvaluationFileHandler
    from meta_spliceai.splice_engine.sequence_featurizer import (
        extract_error_sequences, 
        extract_analysis_sequences, 
        verify_consensus_sequences_via_analysis_data
    )
    # from meta_spliceai.splice_engine.splice_error_analyzer import verify_consensus_sequences
    from meta_spliceai.splice_engine.utils_bio import (
        load_sequences_by_chromosome, 
        load_chromosome_sequence_streaming
    )

    from meta_spliceai.splice_engine.evaluate_models import (
        # evaluate_donor_site_predictions, 
        # evaluate_acceptor_site_predictions, 
        # evaluate_splice_site_predictions, 
        # evaluate_splice_site_errors,
        save_performance_df, 
        save_error_analysis,
    )
    from meta_spliceai.splice_engine.performance_analyzer import (
        PerformanceAnalyzer, 
        sort_performance_by_metric_and_error_type, 
        # analyze_hard_genes
    )
    

    col_tid = kargs.get('col_tid', 'transcript_id')
    separator = kargs.get('separator', kargs.get('sep', '\t'))

    # Configure system I/O paths
    # Todo: Set defualt path in system classes
    splice_site_predictor = 'spliceai'
    local_dir = kargs.get('local_dir', '/path/to/meta-spliceai/data/ensembl/')
    src_dir = kargs.get('src_dir', '/path/to/meta-spliceai/data/ensembl/')
    eval_dir = \
        kargs.get(
            'eval_dir', 
            os.path.join(local_dir, f"{splice_site_predictor}_eval"))
    # f'/path/to/meta-spliceai/data/ensembl/{splice_site_predictor}_eval')
    analysis_dir = \
        kargs.get(
            'analysis_dir', 
            os.path.join(local_dir, f"{splice_site_predictor}_analysis"))
    # f'/path/to/meta-spliceai/data/ensembl/{splice_site_predictor}_analysis')

    # Configure the paths to the GTF file and genome FASTA file
    gtf_file = kargs.get('gtf_file', "/path/to/meta-spliceai/data/ensembl/Homo_sapiens.GRCh38.112.gtf")
    genome_fasta = os.path.join(src_dir, "Homo_sapiens.GRCh38.dna.primary_assembly.fa") 
    assert os.path.exists(gtf_file)
    assert os.path.exists(genome_fasta)
    db_file = os.path.join(local_dir, "annotations.db")

    print("Extract DNA sequences")
    extract_sequences = kargs.get("extract_sequences", False)
    mode = kargs.get('mode', 'gene')
    seq_type = kargs.get('seq_type', 'full')  # options: 'minmax', 'full'
    format = kargs.get('format', 'parquet')

    seq_df_path = os.path.join(local_dir, f"gene_sequence.{format}")
    if mode in ['gene', ]:
        output_file = f"gene_sequence.{format}" 
        if seq_type == 'minmax':
            output_file = f"gene_sequence_minmax.{format}"

        seq_df_path = os.path.join(local_dir, output_file)

        if extract_sequences:
            gene_sequence_retrieval_workflow(
                gtf_file, genome_fasta, gene_tx_map=None, output_file=seq_df_path, mode=seq_type)
    else: 
        seq_df_path = os.path.join(local_dir, f"tx_sequence.{format}")

        if extract_sequences:
            transcript_sequence_retrieval_workflow(
                gtf_file, genome_fasta, gene_tx_map=None, output_file=seq_df_path)

    ########################################################
    find_overlaping_genes = False
    use_precomputed_overlapping_genes = False
    overlapping_gene_file_path = os.path.join(local_dir, "overlapping_gene_counts.tsv")

    if use_precomputed_overlapping_genes:
        if find_overlaping_genes: 
            print_emphasized("[action] Precomputing overlapping genes ...")
            filter_valid_splice_sites = kargs.get('filter_valid_splice_sites', True)
            min_exons = kargs.get('min_exons', 2)

            overlapping_genes_metadata = \
                get_overlapping_gene_metadata(
                    gtf_file,
                    filter_valid_splice_sites=filter_valid_splice_sites,
                    min_exons=min_exons,
                    output_format='dict', 
                    save=True)
        else: 
            print_emphasized("[i/o] Loading precomputed overlapping genes ...")
            overlapping_genes_metadata = \
                load_overlapping_gene_metadata(overlapping_gene_file, output_format='dict', verbose=1)
    else:
        overlapping_genes_metadata = None
        print_emphasized("[info] Overlapping gene metadata will be computed dynamically ...")

    ########################################################
    threshold = kargs.get('threshold', 0.5)
    consensus_window = kargs.get('consensus_window', 3)
    error_window = kargs.get('error_window', 500)

    ########################################################
    action = kargs.get('action', 'generate')  # options: 'analysis', 'generate'
    save_intermediate_files = False  # Save intermediate files for error analysis (
    # recommended after the splice site predition workflow is completed)
    test_mode = kargs.get('test_mode', False)
    col_chrom = 'chrom'
    # separator='\t'

    print_emphasized(f"[info] Running error analysis pipeline with action={action} ...")

    # Keep track of total number of genes processed and time per chunk for estimating time-to-completion
    total_genes_processed = 0
    total_chunk_time = 0
    start_time = time.time()

    # Initialize an empty DataFrame to gather all chromosome results
    full_performance_df = pl.DataFrame()
    full_error_df = pl.DataFrame()
    full_positions_df = pl.DataFrame()
    full_error_sequence_df = pl.DataFrame()
    full_analysis_sequence_df = pl.DataFrame()

    # Initialize an empty list to store the paths of saved performance files
    performance_files = []
    error_analysis_files = []  # Used to generate error analysis files
    splice_positions_files = []  # Used to generate analysis sequences (contextual sequences surrounding predicted splice sites)

    # Gene sequence files for sequence-specific feature extraction
    analysis_sequence_files = []
    error_sequence_files = []

    n_chr_processed = 0
    chromosomes = [str(i) for i in range(1, 23)] + ['X', 'Y', ]  # 'MT'

    # Randomly subset chromosomes when test_mode is True
    if test_mode:
        if len(chromosomes) > 1: 
            chromosomes = random.sample([str(i) for i in range(1, 23)], 1)

    mefd = ModelEvaluationFileHandler(eval_dir, separator=separator)

    print_emphasized(f"[info] Looping through chromosomes {chromosomes} ...")
    for chr in tqdm(chromosomes, desc="Processing chromosomes"):
        chr = str(chr)

        # Load sequence data using streaming mode
        lazy_seq_df = load_chromosome_sequence_streaming(seq_df_path, chr, format=format)

        # Initialize chunk size
        chunk_size = 500  # if not test_mode else 50  # Starting chunk size
        seq_len_avg = 50000  # Assume an average sequence length for now
        num_genes = lazy_seq_df.select(pl.col('gene_id').n_unique()).collect().item()
        n_chunk_processed = 0

        for chunk_start in range(0, num_genes, chunk_size):
            chunk_end = min(chunk_start + chunk_size, num_genes)

            # Track the start time for each chunk
            chunk_start_time = time.time()

            # Adjust the chunk size based on memory availability
            chunk_size = adjust_chunk_size(chunk_size, seq_len_avg)

            # Filter the LazyFrame to process only the current chunk
            seq_chunk = lazy_seq_df.slice(chunk_start, chunk_size).collect()

            print_emphasized(f"[info] Processing chr={chr} | genes {chunk_start + 1} to {chunk_end} out of {num_genes} genes")

            # Predict splice sites for the chunk of gene sequences
            splice_prediction_file = f"splice_predictions_{chr}_chunk_{chunk_start+1}_{chunk_end}.tsv"
            splice_prediction_path = os.path.join(local_dir, splice_prediction_file)
            # NOTE: Prediction dataframes are not saved by default
            
            #######################################
            # chunk_start_adjusted = chunk_start + 2  # ad-hoc
            #######################################
            subject = "splice_performance"
            splice_performance_path = \
                mefd.get_performance_file_path(
                    chr=chr, chunk_start=chunk_start+1, chunk_end=chunk_end, aggregated=False)

            # Model evaluation: Evaluate predictions and gather performance metrics for the current chunk
            # performance_df_chunk = \
            #     evaluate_splice_site_predictions(
            #         ss_annot_df, predictions, 
            #         threshold=threshold, 
            #         consensus_window=consensus_window, verbose=1)
        
            if os.path.exists(splice_performance_path) and os.path.getsize(splice_performance_path) > 0:
                print_emphasized(f"Loading performance dataframe for chr={chr}, chunk {chunk_start+1} to {chunk_end}")
                print_with_indent(f"@ {splice_performance_path}", indent_level=2) 
                performance_df_chunk = \
                    mefd.load_performance_df(
                        chr=chr, 
                        chunk_start=chunk_start+1, 
                        chunk_end=chunk_end, aggregated=False)
            
                # Append chunk performance to the full chromosome performance DataFrame
                try: 
                    # Add chromosome column to performance_df_chunk
                    if col_chrom not in performance_df_chunk.columns:
                        performance_df_chunk = performance_df_chunk.with_columns(pl.lit(chr).alias(col_chrom))
                except pl.exceptions.SchemaError as e:
                    print(f"[warning] {e}")

                    # Infer the schema from full_performance_df
                    inferred_schema = full_performance_df.schema

                    # Cast columns in performance_df_chunk to match the inferred schema
                    for column, dtype in inferred_schema.items():
                        if column in performance_df_chunk.columns:
                            performance_df_chunk = performance_df_chunk.with_columns(
                                pl.col(column).cast(dtype)
                            )
                    
                    # Add chromosome column to performance_df_chunk
                    performance_df_chunk = performance_df_chunk.with_columns(pl.lit(chr).alias(col_chrom))

                except Exception as e:
                    print(f"[error] An unexpected error occurred: {e}")

                if performance_df_chunk.height > 0:

                    if action.startswith(('analysis', 'all')): 
                        # Collect the performance dataframes

                        # full_performance_df = full_performance_df.vstack(performance_df_chunk)
                        print_with_indent("[action] Collecting chunked performance dataframes ...", indent_level=1)
                        full_performance_df = \
                            align_and_append(
                                full_performance_df, 
                                performance_df_chunk, 
                                descriptions={'df1': "full performance dataframe", 'df2': "performance chunk"})
                        
                        # NOTE: 
                        # columns = [
                        #             'gene_id',        # Unique identifier for a gene
                        #             'TP',             # True Positives
                        #             'TN',             # True Negatives
                        #             'FP',             # False Positives
                        #             'FN',             # False Negatives
                        #             'n_splice_sites', # Number of splice sites
                        #             'precision',      # Precision metric
                        #             'recall',         # Recall metric
                        #             'specificity',    # Specificity metric
                        #             'f1_score',       # F1 Score metric
                        #             'fpr',            # False Positive Rate
                        #             'fnr',            # False Negative Rate
                        #             'splice_type'     # Type of splice site: donor or acceptor
                        #         ]

                    # Add path to the list of saved performance files
                    performance_files.append(splice_performance_path)
            else: 
                print(f"[warning] Could not find performance file at {splice_performance_path}")

            # Collect FP, FN datasets
            ####################################
            subject = "splice_errors"
            error_analysis_path = \
                mefd.get_error_analysis_file_path(
                    chr=chr, chunk_start=chunk_start+1, chunk_end=chunk_end, aggregated=False, subject=subject)

            # error_df_chunk = \
            #     evaluate_splice_site_errors(
            #         ss_annot_df, predictions, 
            #         threshold=threshold, 
            #         consensus_window=consensus_window, 
            #         error_window=error_window, verbose=1)
            if os.path.exists(error_analysis_path) and os.path.getsize(error_analysis_path) > 0:
                print_emphasized(f"Loading error analysis dataframe for chr={chr}, chunk {chunk_start+1} to {chunk_end}")
                print_with_indent(f"@ {error_analysis_path}", indent_level=2) 
                error_df_chunk = \
                    mefd.load_error_analysis_df(
                        chr=chr, 
                        chunk_start=chunk_start+1, 
                        chunk_end=chunk_end, aggregated=False)

                # Add chromosome column to error_df_chunk
                try: 
                    if col_chrom not in error_df_chunk.columns:
                        error_df_chunk = error_df_chunk.with_columns(pl.lit(chr).alias(col_chrom))
                except pl.exceptions.SchemaError as e:
                    print(f"[warning] {e}")

                    # Infer the schema from full_error_df
                    inferred_schema = full_error_df.schema

                    # Cast columns in error_df_chunk to match the inferred schema
                    for column, dtype in inferred_schema.items():
                        if column in error_df_chunk.columns:
                            error_df_chunk = error_df_chunk.with_columns(
                                pl.col(column).cast(dtype)
                            )

                    error_df_chunk = error_df_chunk.with_columns(pl.lit(chr).alias(col_chrom))
                except Exception as e:
                    print(f"[error] An unexpected error occurred: {e}")
                
                # ----------------------------------------------
                if error_df_chunk.height > 0:
                    
                    if action.startswith(('analysis', 'all')):
                        # In analysis mode, we collect the error dataframes

                        # full_error_df = full_error_df.vstack(error_df_chunk)
                        full_error_df = \
                            align_and_append(
                                full_error_df, 
                                error_df_chunk, 
                                descriptions={'df1': "full error dataframe", 'df2': "error chunk"})

                    # Add path to the list of error analysis files
                    error_analysis_files.append(error_analysis_path)
                    # Columns: gene_id, error_type, position, window_start, window_end, sequence, has_consensus

                    if action.startswith(('gen', 'all')):

                        print_with_indent("[action] Extracting error sequences for the current chunk ...", indent_level=1)
                        df_seq = extract_error_sequences(seq_chunk, error_df_chunk, include_empty_entries=True, verbose=1)
                        print_with_indent(f"[info] shape(df_seq): {df_seq.shape}, n(genes): {df_seq['gene_id'].n_unique()}", indent_level=2)
                        print_with_indent(f"[info] Columns: {df_seq.columns}", indent_level=2)
                        # NOTE: Columns: ['gene_id', 'error_type', 'window_start', 'window_end', 'sequence']

                        if save_intermediate_files:
                            # Save chunked error sequences
                            subject = "error_sequnces" if not test_mode else "error_sequences_test"
                            error_sequence_path = \
                                mefd.save_error_sequences(
                                    df_seq, chr=chr, chunk_start=chunk_start+1, chunk_end=chunk_end, 
                                    aggregated=False, subject=subject) 
                            error_sequence_files.append(error_sequence_path)

                        # print_with_indent("[test] Verifying consensus sequences ...", indent_level=1)
                        # for splice_type in ['donor', 'acceptor']:
                        #     print_with_indent(f"[action] Verifying consensus sequences for {splice_type} sites ...", indent_level=2)
                        #     df_seq = verify_consensus_sequences_via_analysis_data(
                        #         df_seq, splice_type, window_radius=5, col_pred_type='error_type', verbose=1)
                        #     assert 'has_consensus' in df_seq.columns
                        
                        # Error sequences are being subsumed the analysis sequences
                        # print("[info] Appending chunked error sequences to the full error sequence DataFrame ...")
                        # full_error_sequence_df = \
                        #     align_and_append(
                        #         full_error_sequence_df, 
                        #         df_seq, 
                        #         descriptions={'df1': "full error sequence dataframe", 'df2': "error sequence chunk"})

                # ----------------------------------------------
            else: 
                print_with_indent(f"[warning] Could not find error analysis file at: {error_analysis_path}", indent_level=2)
            
            # Collect TP datasets (+ FP, FN datasets)
            ####################################
            subject = "splice_positions" 
            splice_positions_path = \
                mefd.get_splice_positions_file_path(
                    chr=chr, chunk_start=chunk_start+1, chunk_end=chunk_end, aggregated=False, subject=subject)

            if os.path.exists(splice_positions_path) and os.path.getsize(splice_positions_path) > 0:
                print_emphasized(f"Loading splice positions dataframe for chr={chr}, chunk {chunk_start+1} to {chunk_end}")
                print_with_indent(f"@ {splice_positions_path}", indent_level=2)

                positions_df_chunk = \
                    mefd.load_splice_positions(
                        chr=chr, 
                        chunk_start=chunk_start+1, 
                        chunk_end=chunk_end)

                # Add chromosome column to positions_df_chunk
                positions_df_chunk = positions_df_chunk.with_columns(pl.lit(chr).alias(col_chrom))
                
                if positions_df_chunk.height > 0:
                    
                    if action.startswith(('analysis', 'all')):
                        # Append chunked analysis data to the full analysis DataFrame
                        # full_positions_df = full_positions_df.vstack(positions_df_chunk)
                        full_positions_df = \
                            align_and_append(
                                full_positions_df,  # df(i)
                                positions_df_chunk,  # df(i) + new chunk -> df(i+1)
                                descriptions={'df1': "full positions dataframe", 'df2': "positions chunk"})

                    # Add path to the list of saved splice positions files
                    splice_positions_files.append(splice_positions_path)
                    
                    if action.startswith(('gen', 'all')):
                        # Extract analysis sequences surrounding the true splice sites for TPs, FNs and predicted splice sites for FPs
                        print_with_indent("[action] Extracting analysis sequences for the current chunk ...", indent_level=1)
                        df_seq = extract_analysis_sequences(seq_chunk, positions_df_chunk, include_empty_entries=True, verbose=1)
                        print_with_indent(f"[info] shape(df_seq): {df_seq.shape}, n(genes): {df_seq['gene_id'].n_unique()}", indent_level=2)
                        print_with_indent(f"[info] Columns: {df_seq.columns}", indent_level=2)
                        # NOTE: Columns: ['gene_id', 'position', 'pred_type', 'score', 'splice_type', 'chrom'] + 
                        #                ['window_start', 'window_end', 'sequence']

                        # print_with_indent("[test] Verifying consensus sequences ...", indent_level=1)
                        # for splice_type in ['donor', 'acceptor']:
                        #     print_with_indent(f"[action] Verifying consensus sequences for {splice_type} sites ...", indent_level=2)
                        #     df_seq = verify_consensus_sequences_via_analysis_data(
                        #         df_seq, splice_type, window_radius=5, col_pred_type='error_type', verbose=1)
                        #     assert 'has_consensus' in df_seq.columns

                        if save_intermediate_files:
                            subject = "analysis_sequences" if not test_mode else "analysis_sequences_test"
                            analysis_sequence_path = \
                                mefd.save_analysis_sequences(
                                    df_seq, chr=chr, chunk_start=chunk_start+1, chunk_end=chunk_end, 
                                    aggregated=False, subject=subject) 
                            analysis_sequence_files.append(analysis_sequence_path)

                        # full_analysis_sequence_df = full_analysis_sequence_df.vstack(df_seq)
                        print("[info] Appending chunked analysis sequences to the full analysis sequence DataFrame ...")
                        full_analysis_sequence_df = \
                            align_and_append(
                                full_analysis_sequence_df, 
                                df_seq, 
                                descriptions={'df1': "full analysis sequence dataframe", 'df2': "analysis sequence chunk"})
            else: 
                print_with_indent(f"[warning] Could not find splice positions file at: {splice_positions_path}", indent_level=2)


            # Track the end time for each chunk
            chunk_end_time = time.time()
            chunk_duration = chunk_end_time - chunk_start_time
            total_chunk_time += chunk_duration
            total_genes_processed += (chunk_end - chunk_start)
            
            # Calculate the average time per gene and estimate time remaining
            avg_time_per_gene = total_chunk_time / total_genes_processed
            remaining_genes = num_genes - total_genes_processed
            estimated_remaining_time = remaining_genes * avg_time_per_gene
            print(f"[time] Time taken for this chunk: {format_time(chunk_duration)}")
            print(f"[time] Estimated time to completion: {format_time(estimated_remaining_time)}")

            n_chunk_processed += 1

            if test_mode and n_chunk_processed >= 1: 
                print(f"[test] Test mode: stopping after processing {n_chunk_processed} chunk")
                break
        # End of chunk loop

        print(f"[info] Completed processing chromosome {chr}")
        print(f"[time] Overall time taken so far: {calculate_duration(start_time)}")
        print_section_separator()

        n_chr_processed +=1 

        if test_mode and n_chr_processed >= 1: 
            print(f"[test] Test mode: stopping after processing {n_chr_processed} chromosome")
            break

    # End of chromosome loop

    n_performance_files = len(performance_files)
    n_error_analysis_files = len(error_analysis_files)
    n_splice_positions_files = len(splice_positions_files)
    n_error_sequence_files = len(error_sequence_files)
    n_analsyis_sequence_files = len(analysis_sequence_files)
    print_emphasized(f"[info] Generated n={n_error_sequence_files} error sequence files, n={n_analsyis_sequence_files} analysis sequence files")
    print_with_indent(f"[info] Processed n={n_performance_files} performance files", indent_level=1)
    print_with_indent(f"[info] Processed n={n_error_analysis_files} error analysis files", indent_level=1)
    print_with_indent(f"[info] Processed n={n_splice_positions_files} splice positions files", indent_level=1)

    # Save the final, full performance DataFrame for all chromosomes even WHEN 
    # not all chromosomes are processed
    print_with_indent("[i/o] Saving performance and error analysis data even when only partially available ...", indent_level=1)

    if action.startswith(('gen', 'all')):
        if full_error_sequence_df.height > 0:
            subject = "error_sequences" if not test_mode else "error_sequences_test"
            full_error_sequence_file = \
                mefd.save_error_sequences(
                    full_error_sequence_df, 
                    aggregated=True, subject=subject)
            # print(f"[workflow] Full error sequences saved to: {full_error_sequence_file}")
        # NOTE: Error sequences, only collected for FPs and FNs, are subsumed by the analysis sequences

        if full_analysis_sequence_df.height > 0:
            subject = "analysis_sequences" if not test_mode else "analysis_sequences_test"
            full_analysis_sequence_file = \
                mefd.save_analysis_sequences(
                    full_analysis_sequence_df, 
                    aggregated=True, subject=subject)
            # print(f"[workflow] Full analysis sequences saved to: {full_analysis_sequence_file}")
        # NOTE: Analysis sequences are collected for TPs, FPs, and FNs

    if action.startswith(('analysis', 'all')):

        if full_performance_df.height > 0:
            pa = PerformanceAnalyzer(output_dir=analysis_dir)

            # Analysis logic 
            print_emphasized(f"[workflow] Analyzing performance profile ...")
            sorted_full_performance_df = \
                sort_performance_by_metric_and_error_type(
                    full_performance_df, metric='f1_score', error_type='FP', df_type='polars')
            display_dataframe_in_chunks(sorted_full_performance_df, title="Sorted Performance DataFrame")

            print_emphasized(f"[workflow] Analyzing 'hard' genes (for which SpliceAI didn't do well) ...")
            pa.analyze_hard_genes(
                full_performance_df, 
                metric='f1_score', 
                error_type='FP', 
                n_genes=500, save=True)

            subject = "splice_performance" if not test_mode else "splice_performance_test"
            full_performance_file = \
                mefd.save_performance_df(
                    full_performance_df, 
                    aggregated=True, subject=subject)
            # NOTE: Even if splice_site_prediction_workflow() is not fully executed, the performance data can be saved
            
        if full_error_df.height > 0:
            print_emphasized(f"[workflow] Analyzing error analysis file ...")
            
            subject = "splice_errors" if not test_mode else "splice_errors_test"
            full_error_file = \
                mefd.save_error_analysis(
                    full_error_df, 
                    aggregated=True, subject=subject)
            print(f"[workflow] Full error analysis saved to: {full_error_file}")

        if full_positions_df.height > 0:
            print_emphasized(f"[workflow] Analyzing splice positions file ...")
            
            subject = "splice_positions" if not test_mode else "splice_positions_test"
            full_position_file = \
                mefd.save_splice_positions(
                    full_positions_df, 
                    aggregated=True, subject=subject)
            print(f"[workflow] Full splice positions saved to: {full_position_file}")

            abnormal_positions_df = \
                check_and_subset_invalid_transcript_ids(
                    full_positions_df, col_gid='gene_id', col_tid=col_tid, verbose=1)
            if not is_dataframe_empty(abnormal_positions_df):
                print("[test] Abnormal data in gene_fs_df:")
                # cols = ['gene_id', col_tid, 'position', 'pred_type', 'score', 'splice_type', 'strand', 'chrom']
                display_dataframe_in_chunks(
                    subsample_dataframe(abnormal_positions_df, columns=None, num_rows=10, random=True)
                )
                # raise ValueError("Abnormal data found in gene_fs_df.")

    return

######################################################

def retrieve_predicted_splice_sites(data_dir, **kargs):

    separator = kargs.get('separator', kargs.get('sep', '\t'))
    splice_site_predictor = 'spliceai'
    local_dir = data_dir  # kargs.get('local_dir', '/path/to/meta-spliceai/data/ensembl/')
    src_dir = kargs.get('src_dir', '/path/to/meta-spliceai/data/ensembl/')  # Azure blob storage
    eval_dir = kargs.get('eval_dir', None)
    # f'/path/to/meta-spliceai/data/ensembl/{splice_site_predictor}_eval'

    if eval_dir is None: 
        # Look for the evaluation directory in the order of local_dir, src_dir, and then the default
        for d in [local_dir, src_dir, ]:
            eval_dir = os.path.join(d, f"{splice_site_predictor}_eval")
            if os.path.exists(d):
                break 
    assert os.path.exists(eval_dir), f"Invalid evaluation directory: {eval_dir}"

    mefd = ModelEvaluationFileHandler(eval_dir, separator=separator)
    splice_position_df = \
        mefd.load_splice_positions(aggregated=True)
    return splice_position_df


def retrieve_splice_sites(data_dir, **kargs): 
    from meta_spliceai.splice_engine.utils_fs import read_splice_sites

    format = kargs.get('format', 'tsv')
    splice_sites_file_path = os.path.join(data_dir, f"splice_sites.{format}")

    # Load the annotations from the saved file
    print_emphasized(f"[i/o] Loading splice-site annotations from:\n{splice_sites_file_path}\n")

    try: 
        # ss_annot_df = pl.read_csv(splice_sites_file_path, separator='\t', schema_overrides={'chrom': pl.Utf8})
        ss_annot_df = read_splice_sites(splice_sites_file_path, separator='\t', dtypes=None)
    except Exception as e:
        print(f"[warning] Could not find splice-site annotations from:\n{splice_sites_file_path}")

        gtf_filename = kargs.get('gtf_filename', "Homo_sapiens.GRCh38.112.gtf")
        
        print_emphasized("[action] Extract annotations comprising exon, CDS, 5'UTR, and 3'UTR and splice sites for all transcripts")
        splice_sites_file_path = \
            extract_splice_sites_workflow(
                data_prefix=data_dir, 
                gtf_file=gtf_filename, 
                consensus_window=2)
        # Output: A Polars DataFrame: ['chrom', 'start', 'end', 'strand', 'site_type', 'gene_id', 'transcript_id']

        ss_annot_df = read_splice_sites(splice_sites_file_path, separator='\t', dtypes=None)

    # annotations_df
    print(f"[info] Splice-site datafame: shape={ss_annot_df.shape}")
    print(ss_annot_df.head())

    return ss_annot_df


def splice_site_prediction_workflow(**kargs):
    """
    Splice Site Prediction & Error Analysis Workflow

    This function orchestrates an end-to-end process for:
    1) Loading gene/transcript annotations (GTF).
    2) (Optionally) extracting splice site annotations (exons, CDS, UTR).
    3) (Optionally) extracting or loading gene sequences from a reference genome.
    4) Running SpliceAI-based splice site predictions for each gene across chromosomes.
    5) Evaluating predictions against annotated splice sites.
    6) Generating detailed performance metrics, error analyses, and saving results.

    Core Steps:
    -----------
    1. **Initialization & Configuration**
       - Reads key paths: GTF file, genome FASTA, local directories.
       - Defines parameters for chunk size, model context (e.g. 10,000 nt), thresholds, etc.

    2. **Optionally Extract Gene & Splice Site Annotations** 
       - (Via 'extract_splice_sites_workflow' or direct reading from a precomputed file.)
       - The file (e.g. "splice_sites.tsv") has annotated donor/acceptor sites per transcript.

    3. **(Optional) Gene Sequence Extraction**
       - If `run_sequence_extraction` is True, it calls a custom workflow to read or create a DataFrame of gene sequences in a desired format (full or minmax).
       - Otherwise, it reuses a precomputed file (e.g. "gene_sequence.parquet").

    4. **Chromosome Iteration & Gene Chunking**
       - Loops through each chromosome (1..22, X, Y, MT), possibly with random sampling in test mode.
       - Loads gene sequences in streaming fashion (lazy Polars).
       - Chunks the set of genes (e.g., chunk_size=500) to avoid huge memory usage.

    5. **SpliceAI Prediction per Chunk**
       - If `action='predict'`, loads SpliceAI models and calls `predict_splice_sites_for_genes(seq_chunk, models, ...)`.
       - Otherwise, may load existing prediction files (but this snippet currently bypasses the loading logic if found).
       - Produces a predictions object or DataFrame for each chunk.

    6. **Evaluate Predictions**
       - `evaluate_splice_site_predictions(...)` compares predicted vs. annotated sites, returning performance metrics (TP, FP, FN, etc.).
       - `evaluate_splice_site_errors(...)` captures error cases (FPs, FNs, etc.) plus position-level info.

    7. **Result Assembly & Intermediate Saving**
       - Appends chunk-level performance data to global DataFrames:
         `full_performance_df`, `full_delta_df`, `full_error_df`, `full_positions_df`.
       - Saves chunk-level results to disk (e.g., TSV files) via `ModelEvaluationFileHandler`:
         - Performance (`save_performance_df`)
         - Error analysis (`save_error_analysis`)
         - Splice positions (`save_splice_positions`)

    8. **Global Aggregation**
       - After processing all chromosomes, the function merges chunk-level data into final DataFrames (`full_performance_df`, etc.).
       - Saves aggregated results (per-chromosome and global).
       - Summarizes global performance metrics (precision, recall, specificity, F1, etc.) across all chromosomes.

    9. **Optional Cleanup**
       - If `remove_intermediate_files` is True, deletes the chunk-level files.

    10. **Time & Progress Reporting**
       - Tracks elapsed time for each chunk.
       - Estimates time to completion based on average time per gene.

    Parameters (via **kargs):
    -------------------------
    separator : str
        Delimiter for reading/writing CSV or TSV files (default '\t').
    local_dir : str
        Base directory for storing intermediate results (splicing site files, predictions, etc.).
    eval_dir : str
        Directory for evaluation outputs (performance summaries, error analysis).
    gtf_file : str
        Path to the GTF annotation file.
    genome_fasta : str
        Path to the reference genome FASTA file.
    extract_splice_sites : bool
        Whether to re-run extracting splice sites from the GTF.
    run_sequence_extraction : bool
        Whether to extract gene sequences from the reference genome or load existing.
    test_mode : bool
        If True, random sampling or shortened loops to limit computation.
    remove_intermediate_files : bool
        If True, cleans up chunk-level performance & error files after aggregation.
    chromosomes : list
        The set of chromosomes to process; defaults to [1..22, X, Y, MT]. 
    action : {'predict', 'load'}
        If 'predict', SpliceAI runs on each chunk. Otherwise, attempts to load existing predictions from disk (though partially unimplemented in the snippet).

    Key Outputs:
    ------------
    - Performance DataFrames: 
        * `full_performance_df` with metrics (TP, FP, FN, TN, precision, recall, etc.)
        * Additional data stored in chunk-level files
    - Error DataFrames:
        * `full_error_df` with details of false positives/negatives
        * `full_positions_df` with per-position summary
    - Aggregated chromosome-level summaries in "chromosome_performance_summary.tsv"
    - Print out global summary metrics.

    Returns:
    --------
    None
        The workflow saves its results (performance, error analyses, final summaries) to disk and prints a global summary at the end.

    Notes:
    ------
    - This function uses polars DataFrame operations extensively for chunk loading (lazy evaluation).
    - The partial or full pipeline can be toggled by booleans (extract_splice_sites, run_sequence_extraction, etc.).
    - The user must ensure GTF/FASTA files and any pretrained SpliceAI models are available.

    Example Usage:
    --------------
    splice_site_prediction_workflow(
        separator='\t',
        local_dir='/home/user/splice_results/',
        eval_dir='/home/user/splice_eval/',
        gtf_file='/path/to/Homo_sapiens.GRCh38.112.gtf',
        genome_fasta='/path/to/Homo_sapiens.GRCh38.dna.primary_assembly.fa',
        chromosomes=['1', '2', 'X'],
        test_mode=True,
        action='predict',
        remove_intermediate_files=False
    )
    """
 
    # import time
    # from keras.models import load_model
    # from pkg_resources import resource_filename
    # from spliceai.utils import one_hot_encode

    from meta_spliceai.splice_engine.workflow_utils import align_and_append
    from meta_spliceai.splice_engine.model_evaluator import ModelEvaluationFileHandler
    from meta_spliceai.splice_engine.utils_fs import (
        read_splice_sites, 
        demo_extract_sequences
    )

    separator = kargs.get('separator', kargs.get('sep', '\t'))
    local_dir = '/path/to/meta-spliceai/data/ensembl/'
    src_dir = '/path/to/meta-spliceai/data/ensembl/'
    eval_dir = '/path/to/meta-spliceai/data/ensembl/spliceai_eval'

    gtf_file = "/path/to/meta-spliceai/data/ensembl/Homo_sapiens.GRCh38.112.gtf"  # Replace with your GTF file path
    genome_fasta = os.path.join(src_dir, "Homo_sapiens.GRCh38.dna.primary_assembly.fa") 
    
    assert os.path.exists(gtf_file)
    assert os.path.exists(genome_fasta)

    # Extract gene annotations 
    db_file = 'annotations.db'
    output_file = 'annotations_all_transcripts.tsv'
    # extract_annotations(gtf_file, db_file=db_file, output_file=output_file, sep='\t')

    db_file = os.path.join(local_dir, "annotations.db")
    annotation_file_path = os.path.join(local_dir, "annotations_all_transcripts.csv")
    
    print("[info] Transcipt Annotations (exon, CDS, 5'UTR, and 3'UTR annotations):")
    # annot_df = pd.read_csv(annotation_file_path, sep=',')
    # print(annot_df.head())

    # ------------------------------------------------
    extract_splice_sites = False
    format = 'tsv'
    splice_sites_file_path = os.path.join(local_dir, f"splice_sites.{format}")

    if extract_splice_sites: 
    
        print_emphasized("[action] Extract annotations comprising exon, CDS, 5'UTR, and 3'UTR and splice sites for all transcripts")
        splice_sites_file_path = \
            extract_splice_sites_workflow(data_prefix=local_dir, gtf_file=gtf_file, consensus_window=2)
        # Output: A Polars DataFrame: ['chrom', 'start', 'end', 'strand', 'site_type', 'gene_id', 'transcript_id']

    # Load the annotations from the saved file
    print(f"[i/o] Loading splice-site annotations from:\n{splice_sites_file_path}\n")
    # splice_sites_file_path = os.path.join(local_dir, "splice_sites.tsv")

    # ss_annot_df = pl.read_csv(splice_sites_file_path, separator='\t', schema_overrides={'chrom': pl.Utf8})
    ss_annot_df = read_splice_sites(splice_sites_file_path, separator='\t', dtypes=None)

    # annotations_df
    print(f"[info] Splice-site datafame: shape={ss_annot_df.shape}")
    print(ss_annot_df.head())
    # columns: ['chrom', 'start', 'end', 'strand', 'site_type', 'gene_id', 'transcript_id']

    # sys.exit()
    # ------------------------------------------------
    run_sequence_extraction = False

    print_emphasized("[action] Extract DNA sequences")
    mode = 'gene'
    seq_type = 'full'  # options: 'minmax', 'full'
    format = 'parquet'
    threshold = 0.5
    consensus_window = 3
    error_window = 500
    collect_tn = True

    seq_df_path = os.path.join(local_dir, f"gene_sequence.{format}")
    if mode in ['gene', ]:
        output_file = f"gene_sequence.{format}" 
        if seq_type == 'minmax':
            output_file = f"gene_sequence_minmax.{format}"

        seq_df_path = os.path.join(local_dir, output_file)

        if run_sequence_extraction: 
            gene_sequence_retrieval_workflow(
                gtf_file, genome_fasta, gene_tx_map=None, output_file=seq_df_path, mode=seq_type)
    else: 
        seq_df_path = os.path.join(local_dir, f"tx_sequence.{format}")

        if run_sequence_extraction:
            transcript_sequence_retrieval_workflow(
                gtf_file, genome_fasta, gene_tx_map=None, output_file=seq_df_path)

    if not run_sequence_extraction:
        # try:
        #     if not os.path.exists(seq_df_path):
        #         raise FileNotFoundError(f"The file {seq_df_path} does not exist.")
        # except FileNotFoundError as e:
        #     print(f"Error: {e}")
        #     # other actions ... 
        #     raise
        pass
        # NOTE: gene sequences are saved in parquet format and in chromsome-specific files

    # ------------------------------------------------
    find_overlaping_genes = False
    use_precomputed_overlapping_genes = False
    overlapping_gene_file_path = os.path.join(local_dir, "overlapping_gene_counts.tsv")

    if use_precomputed_overlapping_genes:
        if find_overlaping_genes: 
            print_emphasized("[action] Precomputing overlapping genes ...")
            filter_valid_splice_sites = kargs.get('filter_valid_splice_sites', True)
            min_exons = kargs.get('min_exons', 2)

            overlapping_genes_metadata = \
                get_overlapping_gene_metadata(
                    gtf_file,
                    filter_valid_splice_sites=filter_valid_splice_sites,
                    min_exons=min_exons,
                    output_format='dict', 
                    save=True)
        else: 
            print_emphasized("[i/o] Loading precomputed overlapping genes ...")
            overlapping_genes_metadata = \
                load_overlapping_gene_metadata(overlapping_gene_file, output_format='dict', verbose=1)
    else:
        overlapping_genes_metadata = None
        print_emphasized("[info] Overlapping gene metadata will be computed dynamically ...")
        
    ########################################################
    action = 'predict'  # options: 'predict', 'load'
    remove_intermediate_files = False
    test_mode = kargs.get('test_mode', False)
    # separator='\t'

    # Keep track of total number of genes processed and time per chunk for estimating time-to-completion
    total_genes_processed = 0
    total_chunk_time = 0
    start_time = time.time()

    n_chr_processed = 0
    chromosomes = kargs.get("chromosomes", None)
    if chromosomes is None:
        chromosomes = [str(i) for i in range(1, 23)] + ['X', 'Y', 'MT']
    
    # Randomly subset chromosomes when test_mode is True
    if test_mode:
        if len(chromosomes) > 1: 
            chromosomes = random.sample([str(i) for i in range(1, 23)], 1)

    mefd = ModelEvaluationFileHandler(eval_dir, separator=separator)

    # Initialize an empty DataFrame to gather all chromosome results
    full_performance_df = pl.DataFrame()
    full_delta_df = pl.DataFrame()
    full_error_df = pl.DataFrame()
    full_positions_df = pl.DataFrame()

    # Initialize an empty list to store the paths of saved performance files
    performance_files = []
    error_analysis_files = []
    splice_position_files = []

    print_emphasized("[iterate] Looping through chromosomes ...")
    for chr in tqdm(chromosomes, desc="Processing chromosomes"):

        tqdm.write(f"Processing chromosome={chr} ...")

        # Load sequence data using streaming mode
        lazy_seq_df = load_chromosome_sequence_streaming(seq_df_path, chr, format=format)

        # Initialize chunk size
        chunk_size = 500 if not test_mode else 20  # Starting chunk size
        seq_len_avg = 50000  # Assume an average sequence length for now
        num_genes = lazy_seq_df.select(pl.col('gene_id').n_unique()).collect().item()
        n_chunk_processed = 0

        for chunk_start in range(0, num_genes, chunk_size):
            chunk_end = min(chunk_start + chunk_size, num_genes)

            # Track the start time for each chunk
            chunk_start_time = time.time()

            # Adjust the chunk size based on memory availability
            chunk_size = adjust_chunk_size(chunk_size, seq_len_avg)

            # Filter the LazyFrame to process only the current chunk
            seq_chunk = lazy_seq_df.slice(chunk_start, chunk_size).collect()
            # NOTE: calling .collect() on a lazy Polars DataFrame will load it into physical memory, which can consume significant memory resources
            
            # Estimate the memory usage of the resulting DataFrame
            # estimated_memory_usage = seq_chunk.estimated_size()
            # print(f"Estimated memory usage of seq_chunk: {estimated_memory_usage / (1024 ** 2):.2f} MB")

            print(f"[info] Processing genes {chunk_start + 1} to {chunk_end} out of {num_genes} genes at chromosome={chr} ...")

            # Predict splice sites for the chunk of gene sequences
            splice_prediction_file = f"splice_predictions_{chr}_chunk_{chunk_start+1}_{chunk_end}.tsv"
            splice_prediction_path = os.path.join(local_dir, splice_prediction_file)

            predictions = []
            if (action == 'predict') or not os.path.exists(splice_prediction_path):
                context = 10000
                paths = ('models/spliceai{}.h5'.format(x) for x in range(1, 6))
                models = [load_model(resource_filename('spliceai', x)) for x in paths]

                if mode in ['gene', ]:
                    predictions = \
                        predict_splice_sites_for_genes(
                            seq_chunk, models, context=context, start_time=start_time)

                    # NOTE: Different versions of prediciton functions
                    #   - predict_splice_sites_for_genes_v*()
                else:
                    # predictions = predict_splice_sites(seq_chunk, models, context=context, start_time=start_time)
                    raise RuntimeError(
                        "Splice site predictions evaluated at the transcript level are no longer supported by this function. "
                        "Please use the dedicated function splice_site_prediction_workflow_at_transcript_level() instead."
                    )

                # Save predictions for the current chunk (very big on Gb scale, not recommended)
                # print(f"[i/o] Saved predictions to: {splice_prediction_file}")
                # save_prediction_dataframe(predictions_df, output_file=splice_prediction_file, 
                #                           chunk_start=chunk_start, chunk_end=chunk_end, 
                #                           output_dir=local_dir, use_polars=True)

                assert len(predictions) > 0, "No predictions found!"
                print(f"[info] Processed {len(predictions)} predictions for chunk {chunk_start+1} to {chunk_end}") 
            else:
                # print("[info] Loading predictions from file: ", splice_prediction_path)
                # predictions_df = \
                #     load_prediction_dataframe(input_file=splice_prediction_file,
                #                               chunk_start=chunk_start, chunk_end=None, 
                #                               use_polars=True, input_dir=local_dir)
                pass

            # Read the annotations for the current chromosome (Todo: optimize this)
            # ss_annotat_df = pl.read_csv(splice_sites_file_path, separator=separator, schema_overrides={'chrom': pl.Utf8})
            # columns: ['chrom', 'start', 'end', 'strand', 'site_type', 'gene_id', 'transcript_id']

            # Model evaluation: Evaluate predictions and gather performance metrics for the current chunk
            # performance_df_chunk = evaluate_donor_site_predictions(ss_annot_df, predictions, threshold=threshold, consensus_window=5)
            performance_df_chunk, delta_df_chunk = \
                evaluate_splice_site_predictions(
                    ss_annot_df, predictions, 
                    threshold=threshold, 
                    consensus_window=consensus_window, 
                    # overlapping_genes_metadata=overlapping_genes_metadata, 
                    # gtf_file_path = gtf_file,
                    # overlapping_gene_file_path=overlapping_gene_file_path,  
                    return_discrepancies=True, 
                    chromosome=chr,  # for testing
                    verbose=1)

            # Append chunk performance to the full chromosome performance DataFrame
            # full_performance_df = full_performance_df.vstack(performance_df_chunk)
            full_performance_df = \
                align_and_append(
                    full_performance_df, 
                    performance_df_chunk, 
                    descriptions={'df1': "full performance dataframe", 'df2': "performance chunk"})

            # full_delta_df = full_delta_df.vstack(delta_df_chunk)
            full_delta_df = \
                align_and_append(
                    full_delta_df, 
                    delta_df_chunk, 
                    descriptions={'df1': "full predicted splice site delta dataframe", 'df2': "delta chunk"})


            error_df_chunk, positions_df_chunk = \
                evaluate_splice_site_errors(
                    ss_annot_df, predictions, 
                    threshold=threshold, 
                    consensus_window=consensus_window, 
                    error_window=error_window, 
                    collect_tn=collect_tn,

                    # gtf_file_path = gtf_file,
                    # overlapping_genes_metadata=overlapping_genes_metadata, 
                    # overlapping_gene_file_path=overlapping_gene_file_path,

                    return_positions_df=True,

                    chromosome=chr,  # for testing
                    verbose=1)

            # full_error_df = full_error_df.vstack(error_df_chunk)
            full_error_df = \
                align_and_append(
                    full_error_df, 
                    error_df_chunk, 
                    descriptions={'df1': "full error dataframe", 'df2': "error chunk"})

            # full_positions_df = full_positions_df.vstack(positions_df_chunk)
            full_positions_df = \
                align_and_append(
                    full_positions_df, 
                    positions_df_chunk, 
                    descriptions={'df1': "full positions dataframe", 'df2': "positions chunk"})
            # ----------------------------------------------

            # Save predictions and performance for the current chunk (optional)
            # splice_performance_file = f"splice_performance_{chr}_chunk_{chunk_start + 1}_{chunk_end}.tsv"
            # splice_performance_path = os.path.join(eval_dir, splice_performance_file)
            # os.makedirs(os.path.dirname(splice_performance_path), exist_ok=True)  # Ensure the directory exists
            # performance_df_chunk.write_csv(splice_performance_path)
            splice_performance_path = \
                mefd.save_performance_df(performance_df_chunk, chr, chunk_start+1, chunk_end)
            print(f"[i/o] Saved performance to: {os.path.basename(splice_performance_path)}")

            # Add path to the list of saved performance files
            performance_files.append(splice_performance_path)

            # Free up memory for the current chunk
            del performance_df_chunk, predictions, seq_chunk

            # Save error analysis file for the current chunk
            # error_analysis_file = f"splice_errors_{chr}_chunk_{chunk_start+1}_{chunk_end}.tsv"
            # error_analysis_path = os.path.join(eval_dir, error_analysis_file)
            # os.makedirs(os.path.dirname(splice_performance_path), exist_ok=True)  # Ensure the directory exists
            # error_df_chunk.write_csv(error_analysis_path)
            error_analysis_path = \
                mefd.save_error_analysis(error_df_chunk, chr, chunk_start+1, chunk_end)
            print(f"[i/o] Saved error analysis to: {os.path.basename(error_analysis_path)}")

            # Add path to the list of saved error analysis files
            error_analysis_files.append(error_analysis_path)
            # del error_df_chunk

            splice_position_path = \
                mefd.save_splice_positions(positions_df_chunk, chr, chunk_start+1, chunk_end)
            print(f"[i/o] Saved splice positions to: {os.path.basename(splice_position_path)}")

            # Add path to the list of saved splice positions files
            splice_position_files.append(splice_position_path)
            # del positions_df_chunk

            # ----------------------------------------------

            # Track the end time for each chunk
            chunk_end_time = time.time()
            chunk_duration = chunk_end_time - chunk_start_time
            total_chunk_time += chunk_duration
            total_genes_processed += (chunk_end - chunk_start)
            
            # Calculate the average time per gene and estimate time remaining
            avg_time_per_gene = total_chunk_time / total_genes_processed
            remaining_genes = num_genes - total_genes_processed
            estimated_remaining_time = remaining_genes * avg_time_per_gene
            print(f"[time] Time taken for this chunk: {format_time(chunk_duration)}")
            print(f"[time] Estimated time to completion: {format_time(estimated_remaining_time)}")

            n_chunk_processed += 1

            if test_mode and n_chunk_processed >= 1: 
                print(f"[test] Test mode: stopping after processing {n_chunk_processed} chunk")
                break
        # End of chunk loop

        print(f"[info] Completed processing chromosome {chr}")
        print(f"[time] Overall time taken so far: {calculate_duration(start_time)}")

        n_chr_processed +=1 

        if test_mode and n_chr_processed >= 1: 
            print(f"[test] Test mode: stopping after processing {n_chr_processed} chromosome")
            break

    # End of chromosome loop

    # Save the final, full performance DataFrame for all chromosomes
    # full_performance_file = os.path.join(eval_dir, "full_splice_performance.tsv")
    # full_performance_df.write_csv(full_performance_file, separator=separator)

    subject = "splice_performance" if not test_mode else "splice_performance_test"
    full_performance_file = \
        mefd.save_performance_df(full_performance_df, aggregated=True, subject=subject)
    print(f"[i/o] Full performance saved to: {full_performance_file}")

    subject = "prediction_delta" if not test_mode else "prediction_delta_test"
    full_delta_file = \
        mefd.save_performance_meta(full_delta_df, aggregated=True, subject=subject)
    print(f"[i/o] Full prediction-ground-truth delta statistics saved to: {full_delta_file}")

    # Save the final, full error analysis DataFrame for all chromosomes
    subject = "splice_errors" if not test_mode else "splice_errors_test"
    full_error_file = \
        mefd.save_error_analysis_df(full_error_df, aggregated=True, subject=subject)
    print(f"[i/o] Full error analysis saved to: {full_error_file}")

    subject = "splice_positions" if not test_mode else "splice_positions_test"
    full_position_file = \
        mefd.save_splice_positions(full_positions_df, aggregated=True, subject=subject)
    print(f"[i/o] Full splice positions saved to: {full_position_file}")

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
            performance_df_chunk = pl.read_csv(file, separator=separator)
            
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
    print("Global Summary:", global_summary)

    # Save per-chromosome summaries to a DataFrame and CSV
    chromosome_summaries_df = pl.DataFrame(chromosome_summaries)
    chromosome_summaries_path = os.path.join(eval_dir, "chromosome_performance_summary.tsv")

    if not test_mode: 
        chromosome_summaries_df.write_csv(chromosome_summaries_path, separator=separator)
        print(f"[i/o] Chromosome-level performance saved to: {chromosome_summaries_path}")

    if remove_intermediate_files:
        # Clean up intermediate performance files (optional)
        for performance_file in performance_files:
            os.remove(performance_file)
            print(f"[cleanup] Deleted intermediate file: {performance_file}")

        for error_file in error_analysis_files:
            os.remove(error_file)
            print(f"[cleanup] Deleted intermediate file: {error_file}")

        for splice_position_file in splice_position_files:
            os.remove(splice_position_file)
            print(f"[cleanup] Deleted intermediate file: {splice_position_file}")

    # Load chromosome-level performance summaries
    # global_summary = compute_global_summary_from_chromosome_summary(chromosome_summaries_path)

    # Calculate total execution time
    print(f"[time] Total execution time: {calculate_duration(start_time)}")

    return


def splice_site_prediction_workflow_at_gene_level(**kargs): 
    return splice_site_prediction_workflow(**kargs)


def splice_site_prediction_workflow_at_transcript_level(**kargs): 
    from meta_spliceai.splice_engine.workflow_utils import align_and_append
    # from meta_spliceai.splice_engine.model_evaluator import ModelEvaluationFileHandler

    separator = kargs.get('separator', kargs.get('sep', '\t'))
    local_dir = '/path/to/meta-spliceai/data/ensembl/'
    src_dir = '/path/to/meta-spliceai/data/ensembl/'
    eval_dir = '/path/to/meta-spliceai/data/ensembl/spliceai_eval'

    gtf_file = "/path/to/meta-spliceai/data/ensembl/Homo_sapiens.GRCh38.112.gtf"  # Replace with your GTF file path
    genome_fasta = os.path.join(src_dir, "Homo_sapiens.GRCh38.dna.primary_assembly.fa") 
    
    assert os.path.exists(gtf_file)
    assert os.path.exists(genome_fasta)

    # Extract gene annotations 
    db_file = 'annotations.db'
    output_file = 'annotations_all_transcripts.tsv'
    # extract_annotations(gtf_file, db_file=db_file, output_file=output_file, sep='\t')

    db_file = os.path.join(local_dir, "annotations.db")
    annotation_file_path = os.path.join(local_dir, "annotations_all_transcripts.csv")
    
    print("[info] Transcipt Annotations (exon, CDS, 5'UTR, and 3'UTR annotations):")
    # annot_df = pd.read_csv(annotation_file_path, sep=',')
    # print(annot_df.head())

    # ------------------------------------------------
    extract_splice_sites = False
    format = 'tsv'
    splice_sites_file_path = os.path.join(local_dir, f"splice_sites.{format}")

    if extract_splice_sites: 
    
        print_emphasized("[action] Extract annotations comprising exon, CDS, 5'UTR, and 3'UTR and splice sites for all transcripts")
        splice_sites_file_path = \
            extract_splice_sites_workflow(data_prefix=local_dir, gtf_file=gtf_file, consensus_window=2)
        # Output: A Polars DataFrame: ['chrom', 'start', 'end', 'strand', 'site_type', 'gene_id', 'transcript_id']

    # Load the annotations from the saved file
    print(f"[i/o] Loading splice-site annotations from:\n{splice_sites_file_path}\n")
    # splice_sites_file_path = os.path.join(local_dir, "splice_sites.tsv")

    # ss_annot_df = pl.read_csv(splice_sites_file_path, separator='\t', schema_overrides={'chrom': pl.Utf8})
    ss_annot_df = read_splice_sites(splice_sites_file_path, separator='\t', dtypes=None)

    # annotations_df
    print(f"[info] Splice-site datafame: shape={ss_annot_df.shape}")
    print(ss_annot_df.head())
    # columns: ['chrom', 'start', 'end', 'strand', 'site_type', 'gene_id', 'transcript_id']

    # ------------------------------------------------
    run_sequence_extraction = False

    print_emphasized("[action] Extract DNA sequences")
    mode = 'transcript'
    seq_type = 'full'  # options: 'minmax', 'full'
    format = 'parquet'
    threshold = 0.5
    consensus_window = 3
    error_window = 500

    seq_df_path = os.path.join(local_dir, f"tx_sequence.{format}")

    if run_sequence_extraction:
        transcript_sequence_retrieval_workflow(
            gtf_file, genome_fasta, gene_tx_map=None, output_file=seq_df_path)
    else: 
        # try:
        #     if not os.path.exists(seq_df_path):
        #         raise FileNotFoundError(f"The file {seq_df_path} does not exist.")
        # except FileNotFoundError as e:
        #     print(f"Error: {e}")
        #     # other actions ... 
        #     raise
        pass
        # NOTE: gene sequences are saved in parquet format and in chromsome-specific files

    # ------------------------------------------------
    find_overlaping_genes = False
    use_precomputed_overlapping_genes = False
    overlapping_gene_file_path = os.path.join(local_dir, "overlapping_gene_counts.tsv")

    if use_precomputed_overlapping_genes:
        if find_overlaping_genes: 
            print_emphasized("[action] Precomputing overlapping genes ...")
            filter_valid_splice_sites = kargs.get('filter_valid_splice_sites', True)
            min_exons = kargs.get('min_exons', 2)

            overlapping_genes_metadata = \
                get_overlapping_gene_metadata(
                    gtf_file,
                    filter_valid_splice_sites=filter_valid_splice_sites,
                    min_exons=min_exons,
                    output_format='dict', 
                    save=True)
        else: 
            print_emphasized("[i/o] Loading precomputed overlapping genes ...")
            overlapping_genes_metadata = \
                load_overlapping_gene_metadata(overlapping_gene_file, output_format='dict', verbose=1)
    else:
        overlapping_genes_metadata = None
        print_emphasized("[info] Overlapping gene metadata will be computed dynamically ...")
        
    ########################################################
    action = 'predict'  # options: 'predict', 'load'
    remove_intermediate_files = False
    test_mode = kargs.get('test', False)
    # separator='\t'

    # Keep track of total number of genes processed and time per chunk for estimating time-to-completion
    total_genes_processed = 0
    total_chunk_time = 0
    start_time = time.time()

    n_chr_processed = 0
    chromosomes = kargs.get("chromosomes", None)
    if chromosomes is None:
        chromosomes = [str(i) for i in range(1, 23)] + ['X', 'Y', 'MT']
    
    # Randomly subset chromosomes when test_mode is True
    if test_mode:
        if len(chromosomes) > 1: 
            chromosomes = random.sample([str(i) for i in range(1, 23)], 1)

    mefd = ModelEvaluationFileHandler(eval_dir, separator=separator)

    # Initialize an empty DataFrame to gather all chromosome results
    full_performance_df = pl.DataFrame()
    full_delta_df = pl.DataFrame()
    full_error_df = pl.DataFrame()
    full_positions_df = pl.DataFrame()

    # Initialize an empty list to store the paths of saved performance files
    performance_files = []
    error_analysis_files = []
    splice_position_files = []

    print_emphasized("[iterate] Looping through chromosomes for transcript-level evaluation ...")
    for chr in tqdm(chromosomes, desc="Processing chromosomes"):

        tqdm.write(f"Processing chromosome={chr} ...")

        # 1) Load transcript-level sequence data using streaming mode
        #    Instead of gene-level data, we assume seq_df_path points to a transcript-level dataframe:
        lazy_seq_df = load_chromosome_sequence_streaming(seq_df_path, chr, format=format)

        # 2) Initialize chunk size
        chunk_size = 500 if not test_mode else 20
        seq_len_avg = 50000
        # Instead of counting unique 'gene_id', we count unique 'transcript_id':
        num_transcripts = lazy_seq_df.select(pl.col('transcript_id').n_unique()).collect().item()
        n_chunk_processed = 0

        for chunk_start in range(0, num_transcripts, chunk_size):
            chunk_end = min(chunk_start + chunk_size, num_transcripts)

            chunk_start_time = time.time()
            chunk_size = adjust_chunk_size(chunk_size, seq_len_avg)

            # 3) Filter the LazyFrame to process only the current chunk (slice transcript-level data)
            seq_chunk = lazy_seq_df.slice(chunk_start, chunk_size).collect()

            print(f"[info] Processing transcripts {chunk_start + 1} to {chunk_end} out of {num_transcripts} at chromosome={chr} ...")

            # 4) Load or create SpliceAI models
            context = 10000
            paths = (f"models/spliceai{x}.h5" for x in range(1, 6))
            models = [load_model(resource_filename('spliceai', p)) for p in paths]

            # 5) Transcript-level splice site prediction
            #    We now call predict_splice_sites()—which is for transcript (pre-mRNA) sequences.
            predictions = []
            splice_prediction_file = f"splice_predictions_{chr}_chunk_{chunk_start+1}_{chunk_end}.tsv"
            splice_prediction_path = os.path.join(local_dir, splice_prediction_file)

            # If we're running in 'predict' action or the file is missing, generate predictions
            if (action == 'predict') or not os.path.exists(splice_prediction_path):
                predictions = predict_splice_sites(
                    seq_chunk,
                    models,
                    context=context,
                    start_time=start_time
                )
                assert len(predictions) > 0, "No transcript-level predictions found!"
                print(f"[info] Processed {len(predictions)} transcript predictions for chunk {chunk_start+1}..{chunk_end}")
            else:
                # Possibly load predictions from splice_prediction_path if you desire
                pass

            # 6) Transcript-level evaluation
            #    Instead of evaluate_donor_site_predictions() for genes,
            #    call your transcript-level function, e.g. evaluate_donor_site_predictions_by_transcript()
            #    or evaluate_splice_site_predictions_by_transcript() if you have a combined function.
            performance_df_chunk, delta_df_chunk = evaluate_splice_site_predictions_by_transcript(
                ss_annot_df,        # annotated splice sites with transcript_id
                predictions,
                threshold=threshold,
                consensus_window=consensus_window,
                return_discrepancies=True,
                verbose=1
            )

            # Append chunk performance to the full DataFrame
            full_performance_df = align_and_append(
                full_performance_df,
                performance_df_chunk,
                descriptions={'df1': "full transcript-level performance", 'df2': "performance chunk"}
            )
            full_delta_df = align_and_append(
                full_delta_df,
                delta_df_chunk,
                descriptions={'df1': "full predicted splice site delta", 'df2': "delta chunk"}
            )

            # 7) Transcript-level error analysis
            #    Instead of evaluate_splice_site_errors() for genes, call your function
            #    e.g. evaluate_acceptor_site_errors_by_transcript() or evaluate_donor_site_errors_by_transcript()
            #    or a combined function if you have one that handles both.
            error_df_chunk, positions_df_chunk = evaluate_splice_site_errors_by_transcript(
                ss_annot_df,
                predictions,
                threshold=threshold,
                consensus_window=consensus_window,
                error_window=error_window,
                return_positions_df=True,
                verbose=1
            )

            full_error_df = align_and_append(
                full_error_df,
                error_df_chunk,
                descriptions={'df1': "full error dataframe", 'df2': "transcript error chunk"}
            )
            full_positions_df = align_and_append(
                full_positions_df,
                positions_df_chunk,
                descriptions={'df1': "full positions dataframe", 'df2': "positions chunk"}
            )

            # 8) Save chunk results
            splice_performance_path = mefd.save_performance_df(performance_df_chunk, chr, chunk_start+1, chunk_end)
            print(f"[i/o] Saved performance to: {os.path.basename(splice_performance_path)}")
            performance_files.append(splice_performance_path)

            error_analysis_path = mefd.save_error_analysis(error_df_chunk, chr, chunk_start+1, chunk_end)
            print(f"[i/o] Saved error analysis to: {os.path.basename(error_analysis_path)}")
            error_analysis_files.append(error_analysis_path)

            splice_position_path = mefd.save_splice_positions(positions_df_chunk, chr, chunk_start+1, chunk_end)
            print(f"[i/o] Saved splice positions to: {os.path.basename(splice_position_path)}")
            splice_position_files.append(splice_position_path)

            # 9) Track time & clean up
            chunk_end_time = time.time()
            chunk_duration = chunk_end_time - chunk_start_time
            total_chunk_time += chunk_duration
            total_transcripts_processed += (chunk_end - chunk_start)

            avg_time_per_transcript = total_chunk_time / total_transcripts_processed
            remaining_transcripts = num_transcripts - total_transcripts_processed
            estimated_remaining_time = remaining_transcripts * avg_time_per_transcript
            print(f"[time] Time taken for this chunk: {format_time(chunk_duration)}")
            print(f"[time] Estimated time to completion: {format_time(estimated_remaining_time)}")

            n_chunk_processed += 1
            if test_mode and n_chunk_processed >= 1:
                print(f"[test] Test mode: stopping after {n_chunk_processed} chunk(s)")
                break
        # End chunk loop

        print(f"[info] Completed processing chromosome {chr}")
        print(f"[time] Overall time taken so far: {calculate_duration(start_time)}")

        n_chr_processed += 1
        if test_mode and n_chr_processed >= 1:
            print(f"[test] Test mode: stopping after {n_chr_processed} chromosome(s)")
            break
    # End chromosome loop

    # Save the final, full performance DataFrame for all chromosomes
    # full_performance_file = os.path.join(eval_dir, "full_splice_performance.tsv")
    # full_performance_df.write_csv(full_performance_file, separator=separator)

    subject = "splice_performance" if not test_mode else "splice_performance_test"
    full_performance_file = \
        mefd.save_performance_df(full_performance_df, aggregated=True, subject=subject)
    print(f"[i/o] Full performance saved to: {full_performance_file}")

    subject = "prediction_delta" if not test_mode else "prediction_delta_test"
    full_delta_file = \
        mefd.save_performance_meta(full_delta_df, aggregated=True, subject=subject)
    print(f"[i/o] Full prediction-ground-truth delta statistics saved to: {full_delta_file}")

    # Save the final, full error analysis DataFrame for all chromosomes
    subject = "splice_errors" if not test_mode else "splice_errors_test"
    full_error_file = \
        mefd.save_error_analysis_df(full_error_df, aggregated=True, subject=subject)
    print(f"[i/o] Full error analysis saved to: {full_error_file}")

    subject = "splice_positions" if not test_mode else "splice_positions_test"
    full_position_file = \
        mefd.save_splice_positions(full_positions_df, aggregated=True, subject=subject)
    print(f"[i/o] Full splice positions saved to: {full_position_file}")

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
            performance_df_chunk = pl.read_csv(file, separator=separator)
            
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
    print("Global Summary:", global_summary)

    # Save per-chromosome summaries to a DataFrame and CSV
    chromosome_summaries_df = pl.DataFrame(chromosome_summaries)
    chromosome_summaries_path = os.path.join(eval_dir, "chromosome_performance_summary.tsv")

    if not test_mode: 
        chromosome_summaries_df.write_csv(chromosome_summaries_path, separator=separator)
        print(f"[i/o] Chromosome-level performance saved to: {chromosome_summaries_path}")

    if remove_intermediate_files:
        # Clean up intermediate performance files (optional)
        for performance_file in performance_files:
            os.remove(performance_file)
            print(f"[cleanup] Deleted intermediate file: {performance_file}")

        for error_file in error_analysis_files:
            os.remove(error_file)
            print(f"[cleanup] Deleted intermediate file: {error_file}")

        for splice_position_file in splice_position_files:
            os.remove(splice_position_file)
            print(f"[cleanup] Deleted intermediate file: {splice_position_file}")

    # Load chromosome-level performance summaries
    # global_summary = compute_global_summary_from_chromosome_summary(chromosome_summaries_path)

    # Calculate total execution time
    print(f"[time] Total execution time: {calculate_duration(start_time)}")

    return


def demo_workflow_for_transcripts(): 
    from meta_spliceai.utils.bio_utils import (
        extract_transcripts_from_gtf, 
        extract_transcript_sequences, 
        truncate_sequences, 
        load_sequences
    )
    # from keras.models import load_model
    # from pkg_resources import resource_filename
    # from spliceai.utils import one_hot_encode
    from meta_spliceai.utils.bio_utils import demo_extract_gene_sequences

    local_dir = "/path/to/meta-spliceai/data/ensembl/ALS"

    splice_prediction_file = "splice_predictions.tsv"
    splice_prediction_path = os.path.join(local_dir, splice_prediction_file)

    gene_names = {"STMN2", "UNC13A"}
    tx_ids = {'STMN2': ['ENST00000220876', ], 
              'UNC13A': ['ENST00000519716', ]}

    # print("[action] Extracting transcript sequences from GTF file ...")
    # Use utils.bio_utils to extract gene sequences 
    # demo_extract_transcript_sequences()
    # 
    # Output: 
    # 
    # [parse] Found n=2 genes
    # ... shape(genes_df): (2, 5)
    # [i/o] Saving gene sequences to:
    # /path/to/meta-spliceai/data/ensembl/ALS/tx_sequence.tsv

    format = 'tsv'
    file_path = os.path.join(local_dir, f"tx_sequence.{format}")
    seq_df = load_sequences(file_path, format=format)
    # header: 'gene_name', 'sequence', ... 
    print("[info] cols(seq_df):", list(seq_df.columns))
    
    # Drop the 'sequence' column and display the first few rows
    print(seq_df.drop(columns=['sequence']).head())

    print(f"[info] Found n={seq_df['gene_name'].nunique()} genes")
    print(f"[info] Found n={seq_df['transcript_id'].nunique()} transcripts")
    display_gene_sequence_lengths(seq_df)

    # NOTE: 
    #   > Gene: STMN2 (ENST00000220876), Sequence Length: 55042, Strand: +
    #   > Gene: UNC13A (ENST00000519716), Sequence Length: 87019, Strand: -

    ########################################################


    # Step 1: Extract transcripts and their features
    
    # transcripts_df = extract_transcripts_from_gtf(gtf_file)
    # transcript_sequences_df = extract_transcript_sequences(transcripts_df, genome_fasta, output_format='dataframe')
    transcript_sequences_df = seq_df

    # Step 2: Load SpliceAI models
    context = 10000
    paths = ('models/spliceai{}.h5'.format(x) for x in range(1, 6))
    models = [load_model(resource_filename('spliceai', x)) for x in paths]

    # Step 3: Predict splice sites for the transcript sequences
    action = 'predict' # options: 'predict', 'load'

    # Generate splice site predictions
    if action == 'predict': 
        predictions_df = predict_splice_sites(transcript_sequences_df, models, context=context)
        # NOTE: 'predictions_df' includes the following columns:
        #      ['seqname', 'transcript_id', 'gene_name', 'position', 'absolute_position', 
        #       'donor_prob', 'acceptor_prob', 'neither_prob', 'strand']
    else:
        print("[info] Loading predictions from file: ", splice_prediction_path)
        predictions_df = load_prediction_dataframe(input_file=splice_prediction_file, input_dir=local_dir)

    assert len(predictions_df) > 0, "No predictions found!"
    print("[info] shape(predictions_df):", predictions_df.shape)

    # Save predictions as CSV file
    print("[i/o] Saving predictions as:", splice_prediction_file)
    save_prediction_dataframe(predictions_df, output_file=splice_prediction_file, output_dir=local_dir)
    
    # Display the results
    print(predictions_df.head())
    print("... cols(predictions_df):", list(predictions_df.columns))

    plot_dir = os.path.join(local_dir, "plot")

    # Analyze splice site probabilities
    analyze_splice_site_probabilities(
        predictions_df, 
        donor_threshold_percentile=99.9, 
        acceptor_threshold_percentile=99.9, 
        save_plot=True, 
        output_dir=plot_dir)

    for color_palette in ['deep', 'bright', 'muted', 'pastel', 'dark', 'colorblind']:
        analyze_splice_sites_highlighting_topn_probabilities(
            predictions_df, 
            donor_threshold_percentile=99.9, 
            acceptor_threshold_percentile=99.9, 
            top_n=None,
            save_plot=True, 
            plot_file_name=f'splice_site_probabilities-topN-{color_palette}', 
            plot_format='pdf', 
            output_dir=plot_dir, 
            color_palette=color_palette
        )
    
    output_dir = plot_dir
    output_bed_file = "splice_predictions.bed"

    p_th = 0.01

    top_n_lookup = {}
    for gene, tx_set in tx_ids.items():
        print("[info] Processing gene:", gene)

        for tx in tx_set:
            canonical_junctions_path = os.path.join(local_dir, f"canonical/{gene}/{gene}_canonical_junctions.bed")
            known_bed = load_bed_file_to_dataframe(canonical_junctions_path)

            known_bed = filter_transcripts(known_bed, tx)
            assert len(known_bed) > 0, "No known junctions found for transcript {}".format(tx)

            top_n_lookup[tx] = get_number_of_junctions(known_bed)
            print("[info] Found n={} known junctions for gene {} (transcript {})".format(top_n_lookup[tx], gene, tx))

            bed_file_paths, probable_donors, probable_acceptors, n_junctions, site_prob_thresholds = \
                generate_splice_junctions_bed_file_for_transcripts(
                    predictions_df, 
                    donor_threshold_percentile=99.9, 
                    acceptor_threshold_percentile=99.9, 
                    combined_prob_threshold=p_th, 
                    site_prob_threshold=None, 
                    top_n=top_n_lookup, 
                    output_bed_file=output_bed_file, 
                    output_dir=output_dir)

            proba_bed_file_paths, probable_donors, probable_acceptors, n_junctions, site_prob_thresholds = \
                generate_splice_junctions_bed_file_for_transcripts_with_probabilities(
                    predictions_df, 
                    donor_threshold_percentile=99.9, 
                    acceptor_threshold_percentile=99.9, 
                    combined_prob_threshold=p_th, 
                    site_prob_threshold=None,
                    top_n=top_n_lookup, 
                    output_bed_file=output_bed_file,  # file name will be adjusted automatically
                    output_dir=output_dir)

            splice_junctions_path = bed_file_paths[tx]

            print("[info] Gene={} -> Transcript={}: Predicted junction BED file: {}".format(gene, tx, os.path.basename(splice_junctions_path)))
            predicted_bed = load_bed_file_to_dataframe(splice_junctions_path)

            # Evaluate Top-k accuracy
            top_k = top_n_lookup[tx]
            top_k_accuracy, matched_junctions = evaluate_top_k_accuracy(predicted_bed, known_bed, top_k, threshold=10)
            print(f"[info] Top-{top_k} Accuracy: {top_k_accuracy * 100:.2f}%")
            print(f"[info] Matched Junctions: {matched_junctions}")

            proba_junctions_path = proba_bed_file_paths[tx]
            site_prob_threshold = site_prob_thresholds[tx]
            print("... transcript={}, site_prob_threshold={}".format(tx, site_prob_threshold))

            print("[i/o] Plotting splice junctions for transcript:", tx)
            
            output_file = os.path.join(output_dir, f"splice_site_probabilities-{tx}.pdf")
            plot_splice_sites(proba_junctions_path, threshold=site_prob_threshold, output_file=output_file)

    # Save predictions as BED files
    # file_path = os.path.join(local_dir, "splice_predictions.bed")
    save_as_bed(predictions_df, output_file_prefix="ssp_als", output_dir=local_dir)
    
    save_as_wig(predictions_df, output_file_prefix="ssp_als", output_dir=local_dir)

    save_as_wig_with_log_transform(
        predictions_df,
        output_file_prefix="ssp_als_log_transform",
        output_dir=local_dir,
        log_base=10, epsilon=1e-10)

    return


def demo_basics():
    """

    Memo: 
    * Model Compilation Warning: 
      
      The warning indicates that the model was not compiled because no training configuration was found 
        in the saved file. Since we are only using the model for prediction, we can ignore this warning 
        or compile the model manually if needed.  
        - The models are being loaded using load_model() from Keras, which does not require compilation
        - If you are only using the models for prediction (i.e., calling model.predict()), 
          you do not need to compile the models. Compilation is necessary only if you plan to 
          train or fine-tune the models.

    * The `resource_filename` function from the pkg_resources module is used to get 
      the absolute path to a resource file within a Python package.

    * GLIBCXX_3.4.29 missing problem
       - conda install pandas tensorflow keras

    """
    from keras.models import load_model
    from pkg_resources import resource_filename
    from spliceai.utils import one_hot_encode

    input_sequence = 'CGATCTGACGTGGGTGTCATCGCATTATCGATATTGCAT'
    # input_sequence = 'CTTTACCTCTGGTAAGGGGACC'
    # Replace this with your custom sequence

    print("[info] Input Sequence Length: {}".format(len(input_sequence)))

    context = 10000
    paths = ('models/spliceai{}.h5'.format(x) for x in range(1, 6))
    models = [load_model(resource_filename('spliceai', x)) for x in paths]
    # NOTE: 
    #   - resource_filename: Converts relative paths within a package to absolute paths
    #   - Use "pip show spliceai" or where_is_spliceai() to check the package location

    # Pad the input sequence with 'N' to match the context length
    x = one_hot_encode('N'*(context//2) + input_sequence + 'N'*(context//2))[None, :]
    # NOTE: 
    # - The input sequence is padded with Ns to create a sequence with a total context of 10,000 nucleotides, 
    #    which is typical for SpliceAI's prediction window.
    # - None is equivalent to np.newaxis. It introduces a new axis (dimension) at the specified position.
    print("[info] Data shape: ", x.shape)

    # y = np.mean([models[m].predict(x) for m in range(5)], axis=0)
    
    # Define a function for prediction to avoid retracing
    # @tf.function
    def predict_with_models(models, x):
        return np.mean([model.predict(x) for model in models], axis=0)
    # NOTE: @tf.function is a decorator that converts a Python function to a TensorFlow graph function.
    #       TF's tf.function is a way to convert Python functions into TensorFlow's computation graphs, 
    #       which is highly optimized but has certain restrictions. One of these restrictions is that 
    #       high-level Keras methods like model.predict() should not be called inside a tf.function.
    
    y = predict_with_models(models, x)

    print("[info] Data shape")
    print("... shape(X): ", x.shape)
    print("... shape(y): ", y.shape)
    # NOTE:
    # ... shape(X):  (1, 10022, 4)
    # ... shape(y):  (1, 22, 3)

    acceptor_prob = y[0, :, 1]
    donor_prob = y[0, :, 2]

    for i, (donor_prob, acceptor_prob) in enumerate(zip(donor_prob, acceptor_prob)):
        print(f"Nucleotide {i}: Donor={donor_prob}, Acceptor={acceptor_prob}")

    # Use prepare_input_sequence() to prepare the input sequence for prediction
    # input_sequence = 'CGATCTGACGTGGGTGTCATCGCATTATCGATATTGCAT'

    input_blocks = prepare_input_sequence(input_sequence, context=10000)
    print("[info] Number of input blocks:", len(input_blocks))
    

    return

def demo_basics2(): 
    from keras.models import load_model
    from pkg_resources import resource_filename
    from spliceai.utils import one_hot_encode
    import numpy as np
    import pandas as pd
    from collections import defaultdict

    # Assume the functions prepare_input_sequence() and predict_splice_sites() are already defined and available for use.

    # Test input sequence from SpliceAI demo
    input_sequence = 'CGATCTGACGTGGGTGTCATCGCATTATCGATATTGCAT'

    # Step 1: Prepare Models
    context = 10000
    paths = ('models/spliceai{}.h5'.format(x) for x in range(1, 6))
    models = [load_model(resource_filename('spliceai', x)) for x in paths]

    # Step 2: Prepare input data using prepare_input_sequence() function
    input_blocks = prepare_input_sequence(input_sequence, context)
    print(f"Generated {len(input_blocks)} blocks.")

    # Step 3: Prediction using predict_splice_sites() function
    def predict_splice_sites(sequence, models, context=10000):
        """
        Generate splice site predictions for a single sequence using SpliceAI models.
        
        Parameters:
        - sequence (str): DNA sequence.
        - models (list): List of loaded SpliceAI models.
        - context (int): Context length for SpliceAI (default 10000).

        Returns:
        - dict: A dictionary with donor and acceptor probabilities for each nucleotide position.
        """
        input_blocks = prepare_input_sequence(sequence, context)
        merged_results = defaultdict(lambda: {'donor_prob': [], 'acceptor_prob': [], 'neither_prob': []})

        for block_index, block in enumerate(input_blocks):
            x = block[None, :]  # Add batch dimension for model input
            # Predict splice sites using SpliceAI models
            y = np.mean([model.predict(x) for model in models], axis=0)
            donor_prob = y[0, :, 2]
            acceptor_prob = y[0, :, 1]
            neither_prob = y[0, :, 0]

            # Store the results
            for i in range(len(donor_prob)):
                pos_key = (block_index * 5000 + i + 1)
                merged_results[pos_key]['donor_prob'].append(donor_prob[i])
                merged_results[pos_key]['acceptor_prob'].append(acceptor_prob[i])
                merged_results[pos_key]['neither_prob'].append(neither_prob[i])

        # Consolidate results by averaging overlapping predictions
        results = {pos: {
                        'donor_prob': np.mean(data['donor_prob']),
                        'acceptor_prob': np.mean(data['acceptor_prob']),
                        'neither_prob': np.mean(data['neither_prob'])
                    } for pos, data in merged_results.items()}
        
        return results

    # Predict splice sites for the test input sequence
    results = predict_splice_sites(input_sequence, models, context)

    # Print the results
    print(f"Results for input sequence: {input_sequence}")
    for pos, probs in sorted(results.items()):
        print(f"Position {pos}: Donor={probs['donor_prob']:.4f}, Acceptor={probs['acceptor_prob']:.4f}, Neither={probs['neither_prob']:.4f}")


def demo_convert_to_excel(): 
    from meta_spliceai.splice_engine.utils_doc import convert_to_excel

    local_dir = '/path/to/meta-spliceai/data/ensembl/'
    src_dir = '/path/to/meta-spliceai/data/ensembl/'
    eval_dir = '/path/to/meta-spliceai/data/ensembl/spliceai_eval'
    target_dir = '/path/to/meta-spliceai/data/ensembl/spliceai_eval_excel'

    convert_to_excel(eval_dir, target_dir, shorten_sheet_name=True)

    

def demo(): 

    # SpliceAI's demo code
    # demo_basics()

    # Generazlizing the demo code using prepare_input_sequence() and predict_splice_sites()
    # demo_basics2()

    # where_is_spliceai()

    # demo_junction_prediction_workflow() # Applying SpliceAI to Gene Sequences (e.g., STMN2, UNC13A)
    # demo_workflow_for_transcripts() # Applying SpliceAI to Transcript Sequences (e.g., STMN2, UNC13A)

    # demo_optimize_threshold_for_splice_site_prediction()

    # Convert the evaluation results, outputs of splice_site_prediction_workflow(), to Excel format
    # demo_convert_to_excel()

    # splice_site_prediction_workflow()

    # print_emphasized("[data] Creating training data for error analysis ...")
    # error_analysis_workflow()

    return


def run_workflow(workflow='splice', **kargs):
    """
    Run the specified workflow function.

    Parameters:
    - workflow (str): Name of the workflow to run. If None, run all functions sequentially.

    Command-line Usage:

    - Splice site prediction evaluation workflow:

       $ python run_spliceai_workflow.py --workflow=splice_site_prediction

    - Junction prediction evaluation workflow
    """
    test_mode = kargs.get('test_mode', False)
    action = kargs.get('action', 'generate')

    if workflow is None or workflow.startswith("demo"):
        demo()
    elif workflow.startswith("splice"):
        print_emphasized("[workflow] Running splice site prediction workflow ...")
        chromosomes = [str(i) for i in range(1, 23)] + ['X', 'Y', 'MT']  # None, [str(i) for i in range(8, 23)] + ['X', 'Y', 'MT']
        splice_site_prediction_workflow(chromosomes=chromosomes, test_mode=test_mode)
    elif workflow.startswith(("expl", "interpr")):  # explanation, interpretation
        print_emphasized("[workflow] Running explanation/interpretation workflow ...")
        # workflow_gene_specific_spliceai_prediction()
        demo_spliceai_explanatory_workflow()
    elif workflow.startswith("junction"):
        print_emphasized("[workflow] Running junction prediction workflow ...")
        junction_prediction_workflow()
    elif workflow.startswith("error"):
        print_emphasized("[workflow] Running error analysis workflow ...")
        error_analysis_workflow(test_mode=test_mode, action=action)
    elif workflow.startswith("test"):
        print_emphasized("[workflow] Testing splice site prediction workflow ...")
        splice_site_prediction_workflow(test_mode=True)   
    elif workflow.startswith("optimize"):
        print_emphasized("[workflow] Optimizing threshold for splice site prediction ...")
        demo_optimize_threshold_for_splice_site_prediction()
    else:
        print(f"Unknown workflow: {workflow}")

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run MetaSpliceAI workflows.")
    parser.add_argument(
        "--workflow",
        type=str,
        required=True,
        help="The workflow function to run. Options: 'demo', 'splice site prediction', 'junction prediction', 'error analysis'."
    )
    parser.add_argument(
        "--test",
        action='store_true',  # args.test is True if the flag is present; otherwise, False
        help="Run in test mode."
    )
    parser.add_argument(
        "--action",
        type=str,
        default="generate",
        help="The action to perform. Options: 'generate', 'prediction', 'all', etc. Default is 'generate'."
    )

    args = parser.parse_args()

    return args

def main():
    args = parse_arguments()
    run_workflow(args.workflow, test_mode=args.test, action=args.action)

if __name__ == "__main__":
    main()