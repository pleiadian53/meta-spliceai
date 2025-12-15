import os, sys
import requests
import numpy as np
import pandas as pd
import spliceai  # pip install spliceai

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns

from collections import defaultdict
import logging
import warnings

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


def predict_splice_sites_for_genes_v0(seq_df, models, context=10000):
    """
    Generate splice site predictions for each gene sequence in seq_df.

    Parameters:
    - seq_df (pd.DataFrame): DataFrame with columns 'gene_name', 'sequence' and other optional columns.
    - models (list): List of loaded SpliceAI models.
    - context (int): Context length for SpliceAI (default 10000).

    Returns:
    - pd.DataFrame: DataFrame with columns 'gene_name', 'position', 'donor_prob', 'acceptor_prob', 'neither_prob'.

    - About Case 2:
        Let’s say context = 10000, chunk_overlap = 5000, and seq_len = 15000
            
        First Chunk:
            start = 0, end = 10000, so chunk = sequence[0:10000].
        Second Chunk:
            start = 5000, end = 15000, so chunk = sequence[5000:15000].

        Third Chunk is shorter:

            start = 10000, end = min(10000 + 10000, 15000) = 15000
            Covers nucleotides 10,001–15,000
            This chunk will be shorter, as it covers only 5,000 nt.

        Forth chunk does not exist.
    """
    from keras.models import load_model
    from pkg_resources import resource_filename
    # from spliceai.utils import one_hot_encode

    def predict_with_models(models, x):
        return np.mean([model.predict(x) for model in models], axis=0)

    results = []

    for idx, row in seq_df.iterrows():
        gene_name = row['gene_name']
        sequence = row['sequence']
        strand = row.get('strand', '+')
        seq_len = len(sequence)

        if seq_len <= context:
            # Case 1: Sequence length is less than or equal to context length
            padding_needed = context - seq_len
            left_padding = padding_needed // 2
            right_padding = padding_needed - left_padding
            padded_sequence = 'N' * left_padding + sequence + 'N' * right_padding
            x = one_hot_encode(padded_sequence)[None, :]
            print("... shape(x): ", x.shape)
            
            # Predict splice sites using SpliceAI models
            y = np.mean([model.predict(x) for model in models], axis=0)
            print("... shape(y): ", y.shape)
            
            # Extract probabilities for the original sequence length
            donor_prob = y[0, left_padding:left_padding + seq_len, 2]
            acceptor_prob = y[0, left_padding:left_padding + seq_len, 1]
            neither_prob = y[0, left_padding:left_padding + seq_len, 0]

            # Store the results
            for i, (donor_p, acceptor_p, neither_p) in enumerate(zip(donor_prob, acceptor_prob, neither_prob)):
                results.append({
                    'gene_name': gene_name,
                    'position': i + 1,  # 1-based position in the original sequence
                    'donor_prob': donor_p,
                    'acceptor_prob': acceptor_p,
                    'neither_prob': neither_p, 
                    'strand': strand
                })
            # NOTE: 
            #   `i` is the index of the current nucleotide within the current chunk of sequence being processed. 
            #   This index starts from 0 and goes up to len(chunk) - 1.
        
        else:
            # Case 2: Sequence length is greater than context length
            # Process in overlapping chunks of 'context' length
            chunk_overlap = 5000  # For example, a 5,000 nt overlap
            for start in range(0, seq_len, context - chunk_overlap):
                end = min(start + context, seq_len)
                chunk = sequence[start:end]
                
                # Pad if necessary (only the last chunk might need padding)
                if len(chunk) < context:
                    chunk = 'N' * ((context - len(chunk)) // 2) + chunk + 'N' * ((context - len(chunk) + 1) // 2)
                # NOTE:
                # Let's consider an example to verify the logic:
                #    Context Length: 10
                #    Chunk Length: 7
                # Padding Calculation:
                #    Total padding needed: 10 - 7 = 3
                #    Left padding: 3 // 2 = 1
                #    Right padding: (3 + 1) // 2 = 2
                # Resulting Chunk:
                #    Original chunk: AAA
                #    Padded chunk: 'N' * 1 + 'AAA' + 'N' * 2 = 'NAAA' + 'NN' = 'NAAANN'
                
                x = one_hot_encode(chunk)[None, :]
                print("... shape(x): ", x.shape)  # (1, 10000, 4)
                
                # Predict splice sites using SpliceAI models
                # y = np.mean([model.predict(x) for model in models], axis=0)
                y = predict_with_models(models, x)
                print("... shape(y): ", y.shape)
                
                # Extract probabilities for the original chunk length (before padding)
                donor_prob = y[0, :, 2][:len(chunk)]
                acceptor_prob = y[0, :, 1][:len(chunk)]
                neither_prob = y[0, :, 0][:len(chunk)]

                # Store the results with adjusted positions
                for i, (donor_p, acceptor_p, neither_p) in enumerate(zip(donor_prob, acceptor_prob, neither_prob)):
                    results.append({
                        'gene_name': gene_name,
                        'position': start + i + 1,  # See notes below
                        'donor_prob': donor_p,
                        'acceptor_prob': acceptor_p,
                        'neither_prob': neither_p,
                        'strand': strand
                    })

                # NOTE: 
                #   `start + i + 1` maps the position within the chunk to the corresponding position within 
                #    the entire original sequence.

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    return results_df


def predict_splice_sites_for_genes_v1(seq_df, models, context=10000, crop_size=5000):
    """
    Generate splice site predictions for each gene sequence in seq_df, handling all sequences uniformly.
    
    Parameters:
    - seq_df (pd.DataFrame): DataFrame with columns 'gene_name', 'sequence'.
    - models (list): List of loaded SpliceAI models.
    - context (int): Context length for SpliceAI (default 10000).
    - crop_size (int): Number of nucleotides to crop from each end (default 5000).

    Returns:
    - pd.DataFrame: DataFrame with columns 'gene_name', 'position', 'donor_prob', 'acceptor_prob', 'neither_prob'.

    """
    from keras.models import load_model
    from pkg_resources import resource_filename
    from spliceai.utils import one_hot_encode

    results = []

    for idx, row in seq_df.iterrows():
        gene_name = row['gene_name']
        sequence = row['sequence']
        strand = row.get('strand', '+')
        seq_len = len(sequence)

        chunk_overlap = crop_size  # Overlap based on the crop size
        min_chunk_size = context + 2 * crop_size  # Ensure enough length to accommodate cropping

        for start in range(0, seq_len, min_chunk_size - chunk_overlap):
            end = min(start + min_chunk_size, seq_len)
            chunk = sequence[start:end]
            # Memory Efficiency: 
            # Through chunking, the function avoids overloading the model with excessively long sequences,
            # which can cause memory issues and slow down processing.

            # Pad if necessary (only the last chunk might need padding)
            if len(chunk) < min_chunk_size:
                chunk = 'N' * ((min_chunk_size - len(chunk)) // 2) + chunk + 'N' * ((min_chunk_size - len(chunk) + 1) // 2)

            x = one_hot_encode(chunk)[None, :]

            # Predict splice sites using SpliceAI models
            y = np.mean([model.predict(x) for model in models], axis=0)

            # Extract probabilities, adjusting for cropping
            donor_prob = y[0, crop_size:crop_size + (end - start), 2]
            acceptor_prob = y[0, crop_size:crop_size + (end - start), 1]
            neither_prob = y[0, crop_size:crop_size + (end - start), 0]

            # Store the results with adjusted positions
            for i, (donor_p, acceptor_p, neither_p) in enumerate(zip(donor_prob, acceptor_prob, neither_prob)):
                results.append({
                    'gene_name': gene_name,
                    'position': start + i + 1,
                    'donor_prob': donor_p,
                    'acceptor_prob': acceptor_p,
                    'neither_prob': neither_p, 
                    'strand': strand
                })

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    return results_df


def predict_splice_sites_for_genes_v2(seq_df, models, context=10000, crop_size=5000):
    """
    Generate splice site predictions for each gene sequence in seq_df, handling all sequences uniformly.
    
    Parameters:
    - seq_df (pd.DataFrame): DataFrame with columns 'gene_name', 'sequence', 'start', 'end' (optional).
    - models (list): List of loaded SpliceAI models.
    - context (int): Context length for SpliceAI (default 10000).
    - crop_size (int): Number of nucleotides to crop from each end (default 5000).

    Returns:
    - pd.DataFrame: DataFrame with columns 'gene_name', 'position', 'absolute_position', 'donor_prob', 'acceptor_prob', 'neither_prob'.

    Memo:
    - Handles long sequences via chunking to ensure memory efficiency and prediction accuracy.

    Memo:
    * Handling Long Sequences via Chunking:
        E.g. context = 10,000: This is the context length SpliceAI uses.
            crop_size = 5,000: SpliceAI crops 5,000 nucleotides from each end after processing the sequence.
            
            => min_chunk_size = context + 2 * crop_size = 10,000 + 2 * 5,000 = 20,000
            This ensures that the chunk size accommodates the cropping without losing valuable data.

            chunk_overlap = crop_size = 5,000: 
            This overlap ensures that the edges of each chunk have enough context from the adjacent chunk.

            input_sequence length = 87,000 nt (as in UNC13A).

            => 6 chunks will be generated: 
                1st Chunk: 0 to 20,000 nt.
                2nd Chunk: 15,000 to 35,000 nt.
                3rd Chunk: 30,000 to 50,000 nt.
                4th Chunk: 45,000 to 65,000 nt.
                5th Chunk: 60,000 to 80,000 nt.
                6th Chunk: 75,000 to 87,000 nt.

            Final Chunk: The last chunk may be shorter than the full chunk size (20,000 nt) 
            if padding isn't applied, but it will still be processed by the model with the correct context.
    """
    from keras.models import load_model
    from pkg_resources import resource_filename
    from spliceai.utils import one_hot_encode

    results = []

    for idx, row in seq_df.iterrows():
        gene_name = row['gene_name']
        sequence = row['sequence']
        strand = row.get('strand', '+')
        seq_len = len(sequence)

        # Check if 'start' and 'end' positions are available
        has_absolute_positions = 'start' in row and 'end' in row
        gene_start = row.get('start', None)  # Absolute start position if available

        chunk_overlap = crop_size  # Overlap based on the crop size
        min_chunk_size = context + 2 * crop_size  # Ensure enough length to accommodate cropping

        for start in range(0, seq_len, min_chunk_size - chunk_overlap):
            end = min(start + min_chunk_size, seq_len)
            chunk = sequence[start:end]

            # Memory Efficiency: avoids overloading the model with excessively long sequences
            if len(chunk) < min_chunk_size:
                chunk = 'N' * ((min_chunk_size - len(chunk)) // 2) + chunk + 'N' * ((min_chunk_size - len(chunk) + 1) // 2)

            x = one_hot_encode(chunk)[None, :]

            # Predict splice sites using SpliceAI models
            y = np.mean([model.predict(x) for model in models], axis=0)

            # Extract probabilities, adjusting for cropping
            donor_prob = y[0, crop_size:crop_size + (end - start), 2]
            acceptor_prob = y[0, crop_size:crop_size + (end - start), 1]
            neither_prob = y[0, crop_size:crop_size + (end - start), 0]

            # Store the results with adjusted positions
            for i, (donor_p, acceptor_p, neither_p) in enumerate(zip(donor_prob, acceptor_prob, neither_prob)):
                absolute_position = gene_start + (start + i) if has_absolute_positions else None
                results.append({
                    'gene_name': gene_name,
                    'position': start + i + 1,
                    'absolute_position': absolute_position,  # New field for absolute position
                    'donor_prob': donor_p,
                    'acceptor_prob': acceptor_p,
                    'neither_prob': neither_p, 
                    'strand': strand
                })

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    return results_df
    

def predict_splice_sites_for_genes(seq_df, models, context=10000, crop_size=5000):
    """
    Generate splice site predictions for each gene sequence in seq_df, handling all sequences uniformly.
    
    Parameters:
    - seq_df (pd.DataFrame): DataFrame with columns 'gene_name', 'sequence', 'start', 'end' (optional).
    - models (list): List of loaded SpliceAI models.
    - context (int): Context length for SpliceAI (default 10000).
    - crop_size (int): Number of nucleotides to crop from each end (default 5000).

    Returns:
    - pd.DataFrame: DataFrame with columns 'gene_name', 'position', 'absolute_position', 'donor_prob', 'acceptor_prob', 'neither_prob'.
    """
    from keras.models import load_model
    from pkg_resources import resource_filename
    from spliceai.utils import one_hot_encode

    # Dictionary to store merged results by nucleotide positions
    merged_results = defaultdict(lambda: {'donor_prob': [], 'acceptor_prob': [], 'neither_prob': []})

    for idx, row in seq_df.iterrows():
        gene_name = row['gene_name']
        sequence = row['sequence']
        seqname = row.get('seqname', gene_name)  # chromosome or sequence name
        strand = row.get('strand', '+')
        seq_len = len(sequence)

        # Check if 'start' and 'end' positions are available
        has_absolute_positions = 'start' in row and 'end' in row
        gene_start = row.get('start', None)  # Absolute start position if available

        chunk_overlap = crop_size  # Overlap based on the crop size
        min_chunk_size = context + 2 * crop_size  # Ensure enough length to accommodate cropping

        for start in range(0, seq_len, min_chunk_size - chunk_overlap):
            end = min(start + min_chunk_size, seq_len)
            chunk = sequence[start:end]

            # Pad if necessary (only the last chunk might need padding)
            if len(chunk) < min_chunk_size:
                chunk = 'N' * ((min_chunk_size - len(chunk)) // 2) + chunk + 'N' * ((min_chunk_size - len(chunk) + 1) // 2)

            x = one_hot_encode(chunk)[None, :]

            # Predict splice sites using SpliceAI models
            y = np.mean([model.predict(x) for model in models], axis=0)

            # Extract probabilities, adjusting for cropping
            donor_prob = y[0, crop_size:crop_size + (end - start), 2]
            acceptor_prob = y[0, crop_size:crop_size + (end - start), 1]
            neither_prob = y[0, crop_size:crop_size + (end - start), 0]

            # Store the results with adjusted positions
            for i, (donor_p, acceptor_p, neither_p) in enumerate(zip(donor_prob, acceptor_prob, neither_prob)):
                absolute_position = gene_start + (start + i) if has_absolute_positions else None
                pos_key = (gene_name, absolute_position) if has_absolute_positions else (gene_name, start + i + 1)

                # Append probabilities for overlapping positions
                merged_results[pos_key]['donor_prob'].append(donor_p)
                merged_results[pos_key]['acceptor_prob'].append(acceptor_p)
                merged_results[pos_key]['neither_prob'].append(neither_p)
                merged_results[pos_key]['strand'] = strand
                merged_results[pos_key]['seqname'] = seqname
                merged_results[pos_key]['absolute_position'] = absolute_position if has_absolute_positions else (start + i + 1)

    # Consolidate results by averaging overlapping predictions
    results = []
    for (gene_name, position), data in merged_results.items():
        results.append({
            'seqname': data['seqname'],
            'gene_name': gene_name,
            'position': position,
            'absolute_position': data['absolute_position'],
            'donor_prob': np.mean(data['donor_prob']),
            'acceptor_prob': np.mean(data['acceptor_prob']),
            'neither_prob': np.mean(data['neither_prob']),
            'strand': data['strand']
        })

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    return results_df


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


def predict_splice_sites_v0(): 
    from keras.models import load_model
    from pkg_resources import resource_filename
    # import requests

    # Load pretrained SpliceAI models
    context = 10000  # Set context size for SpliceAI-10k
    paths = ('models/spliceai{}.h5'.format(x) for x in range(1, 6))
    models = [load_model(resource_filename('spliceai', x)) for x in paths]

    # Fetch the STMN2 transcript sequence and prepare it
    stmn2_sequence = "" # fetch_stmn2_sequence()
    blocks = prepare_input_sequence(stmn2_sequence, context)

    # Predict acceptor and donor site probabilities for each block
    acceptor_probs = []
    donor_probs = []

    for block in blocks:
        x = block[None, :]  # Add a batch dimension
        y = np.mean([models[m].predict(x) for m in range(5)], axis=0)  # Ensemble prediction
        acceptor_prob = y[0, :, 1]  # Extract acceptor probabilities
        donor_prob = y[0, :, 2]  # Extract donor probabilities
        acceptor_probs.append(acceptor_prob)
        donor_probs.append(donor_prob)

    # Flatten the results to match the length of the original sequence
    acceptor_probs = np.concatenate(acceptor_probs)[:len(stmn2_sequence)]
    donor_probs = np.concatenate(donor_probs)[:len(stmn2_sequence)]

    # Display or analyze the predictions for splice sites
    print("Acceptor Probabilities:", acceptor_probs)
    print("Donor Probabilities:", donor_probs)


def predict_splice_sites_for_transcripts(transcript_df, models, context=10000): 
    return predict_splice_sites(transcript_df, models, context=context)

def predict_splice_sites(transcript_df, models, context=10000):
    """
    Generate splice site predictions for each transcript sequence in transcript_df using SpliceAI models.

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
        seqname = row['seqname']  # chromosome or sequence name
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

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Sanity check: Ensure output probabilities match the input sequence lengths
    for transcript_id in transcript_df['transcript_id'].unique():
        input_length = len(transcript_df[transcript_df['transcript_id'] == transcript_id]['sequence'].values[0])
        output_length = len(results_df[results_df['transcript_id'] == transcript_id])
        assert input_length == output_length, f"Length mismatch for {transcript_id}: input {input_length}, output {output_length}"

    return results_df


####################################################################################################


def determine_optimal_threshold(donor_probs, acceptor_probs, target_junctions=43, initial_threshold=0.9):
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
        optimal_threshold, junctions = determine_optimal_threshold(donor_probs, acceptor_probs, target_junctions=43)
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

        # determine_optimal_threshold(gene_df['donor_prob'], gene_df['acceptor_prob'], target_junctions=43, initial_threshold=0.9)
 
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
                determine_optimal_threshold(
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


def save_prediction_dataframe(predictions_df, output_file, **kargs):
    """
    Save the prediction DataFrame to a CSV file.
    
    Parameters:
    - predictions_df (pd.DataFrame): DataFrame containing gene name, position, donor, acceptor, neither probabilities, and strand.
    - output_file (str): Path to the output CSV file.
    - output_dir (str, optional): Directory to save the output CSV file. Default is the current directory.

    Example usage:
        predictions_df = predict_splice_sites_for_genes(seq_df, models, context)
        save_prediction_dataframe(predictions_df, "predictions.csv", output_dir='/path/to/output')
    """
    sep = kargs.get('sep', '\t')
    output_dir = kargs.get('output_dir', '.')
    output_path = os.path.join(output_dir, output_file)
    
    predictions_df.to_csv(output_path, sep=sep, index=False)
    print(f"Prediction DataFrame saved to {output_path}")


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


def load_prediction_dataframe(input_file, **kargs):
    """
    Load the prediction DataFrame from a CSV file.
    
    Parameters:
    - input_file (str): Path to the input CSV file.
    - sep (str, optional): Delimiter to use in the CSV file (default is '\t').
    - input_dir (str, optional): Directory to load the input CSV file from. Default is the current directory.
    
    Returns:
    - predictions_df (pd.DataFrame): DataFrame containing gene name, position, donor, acceptor, neither probabilities, and strand.
    
    Example usage:
        predictions_df = load_prediction_dataframe("predictions.csv", input_dir='/path/to/input')
    """
    sep = kargs.get('sep', '\t')
    input_dir = kargs.get('input_dir', '.')
    input_path = os.path.join(input_dir, input_file)
    
    predictions_df = pd.read_csv(input_path, sep=sep)
    print(f"Prediction DataFrame loaded from {input_path}")
    
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
    - df (pd.DataFrame): DataFrame containing gene names and sequences with columns 'gene_name' and 'sequence'.
    """
    for index, row in df.iterrows():
        gene_name = row['gene_name']
        if 'transcript_id' in row:
            gene_name = f"{gene_name} ({row['transcript_id']})"
        sequence_length = len(row['sequence'])
        print(f"> Gene: {gene_name}, Sequence Length: {sequence_length}, Strand: { row.get('strand', '?') }")


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


def demo_workflow(): 
    from meta_spliceai.utils.bio_utils import (
        parse_gtf_for_genes, 
        extract_genes_from_gtf, 
        extract_gene_sequences, 
        load_sequences
    )
    from keras.models import load_model
    from pkg_resources import resource_filename
    from spliceai.utils import one_hot_encode
    # from meta_spliceai.utils.bio_utils import demo_extract_gene_sequences

    local_dir = "/path/to/meta-spliceai/data/ensembl/ALS"

    splice_prediction_file = "splice_predictions.tsv"
    splice_prediction_path = os.path.join(local_dir, splice_prediction_file)

    gene_names = {"STMN2", "UNC13A"}

    print("[action] Extracting gene sequences from GTF file ...")
    # Use utils.bio_utils to extract gene sequences 
    demo_extract_gene_sequences()
    # 
    # Output: 
    # 
    # [parse] Found n=2 genes
    # ... shape(genes_df): (2, 5)
    # [i/o] Saving gene sequences to:
    # /path/to/meta-spliceai/data/ensembl/ALS/gene_sequence.tsv

    format = 'tsv'
    file_path = os.path.join(local_dir, f"gene_sequence.{format}")
    seq_df = load_sequences(file_path, format=format)
    # header: 'gene_name', 'sequence', ... 
    print("[info] cols(seq_df):", list(seq_df.columns))
    
    # Drop the 'sequence' column and display the first few rows
    print(seq_df.drop(columns=['sequence']).head())

    print(f"[info] Found n={seq_df['gene_name'].nunique()} genes")
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
        predictions_df = predict_splice_sites_for_genes(seq_df, models, context)
        # NOTE: 'predictions_df' includes the following columns:
        #   'gene_name', 'position', 'absolute_position', 'donor_prob', 'acceptor_prob', 'neither_prob', 'strand'
    else:
        print("[info] Loading predictions from file: ", splice_prediction_path)
        predictions_df = load_prediction_dataframe(input_file=splice_prediction_file, input_dir=local_dir)

    # Save predictions as CSV file
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


def demo_workflow_for_transcripts(): 
    from meta_spliceai.utils.bio_utils import (
        extract_transcripts_from_gtf, 
        extract_transcript_sequences, 
        truncate_sequences, 
        load_sequences
    )
    from keras.models import load_model
    from pkg_resources import resource_filename
    from spliceai.utils import one_hot_encode
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
        #   'gene_name', 'position', 'absolute_position', 'donor_prob', 'acceptor_prob', 'neither_prob', 'strand'
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

    

def demo(): 

    # SpliceAI's demo code
    # demo_basics()

    # Generazlizing the demo code using prepare_input_sequence() and predict_splice_sites()
    # demo_basics2()

    # where_is_spliceai()

    # demo_workflow() # Applying SpliceAI to Gene Sequences (e.g., STMN2, UNC13A)
    demo_workflow_for_transcripts() # Applying SpliceAI to Transcript Sequences (e.g., STMN2, UNC13A)


def test(): 
    pass

if __name__ == "__main__": 
    demo()