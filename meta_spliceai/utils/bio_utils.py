
from Bio import SeqIO

import os, subprocess
import tempfile

import pandas as pd
import polars as pl

import csv
import re
from tqdm import tqdm

def count_sequences(fasta_file):
    sequences = list(SeqIO.parse(fasta_file, 'fasta'))
    return len(sequences)


def create_gtf_v0(features, output_filename):
    """
    Creates a GTF file from a list of feature dictionaries.
    """
    with open(output_filename, 'w') as gtf_fh:
        for feature in features:
            write_gtf_line(gtf_fh, **feature)


def parse_gtf_v0(file_path):
    """
    Parses a GTF file to a pandas DataFrame.

    Parameters
    ----------
    file_path : str
        The path to the GTF file.

    Returns
    -------
    df : pandas.DataFrame
        DataFrame containing the parsed GTF data.

    Example usage
    -------------
    file_path = 'path_to_your_gtf_file.gtf'
    df = parse_gtf_to_dataframe(file_path)
    print(df.head())

    """
    # Define lists to hold data
    data = {
        'chromosome': [], 'source': [], 'feature_type': [], 'start': [], 'end': [], 'strand': [],
        'gene_id': [], 'orf_id': [], 'transcript_id': [], 'reference_id': [], 'splice': [], 'splice_sites': []
    }

    # Open and read the GTF file
    with open(file_path, 'r') as file:
        for line in file:
            if line.strip() and not line.startswith('#'):
                fields = line.strip().split('\t')
                attrs = fields[8].split(';')
                attr_dict = {}
                for attr in attrs:
                    if attr.strip():
                        key, value = attr.strip().split(' ')
                        attr_dict[key] = value.strip('"')
                
                # Parse fields and attributes
                data['chromosome'].append(fields[0])
                data['source'].append(fields[1])
                data['feature_type'].append(fields[2])
                data['start'].append(int(fields[3]))
                data['end'].append(int(fields[4]))
                data['strand'].append(fields[6])
                data['gene_id'].append(attr_dict.get('gene_id', ''))
                data['orf_id'].append(attr_dict.get('orf_id', ''))
                data['transcript_id'].append(attr_dict.get('transcript_id', ''))
                data['reference_id'].append(attr_dict.get('reference_id', ''))
                
                # Handle optional splice information
                splice = 'splice' in attr_dict
                data['splice'].append(splice)
                if splice and 'sites' in attr_dict:
                    sites_str = attr_dict['sites'][1:-1]  # Remove parentheses
                    data['splice_sites'].append(sites_str)
                else:
                    data['splice_sites'].append(None)

    # Convert lists to DataFrame
    df = pd.DataFrame(data)
    
    return df


def filter_gtf_by_genes(gtf_file_path, gene_names):
    """
    Filter a GTF file by specific gene names.

    Parameters:
    - gtf_file_path (str): Path to the input GTF file.
    - gene_names (set or list): A set or list of gene names to filter.

    Returns:
    - pd.DataFrame: A DataFrame with GTF entries corresponding to the specified genes.

    Example Usage:

        gtf_file = "h38.filt.gtf"  # Path to your GTF file
        gene_names = {"STMN2", "UNC13A"}  # Replace with your set of gene names

        filtered_gtf_df = filter_gtf_by_genes(gtf_file, gene_names)
    """
    features = parse_gtf(gtf_file_path)
    filtered_features = [f for f in features if f['attributes'].get('gene_name') in gene_names]
    return pd.DataFrame(filtered_features)


def parse_gtf_for_genes(gtf_file_path):
    """
    Parse a GTF file to extract gene coordinates.

    Parameters:
    - gtf_file_path (str): Path to the input GTF file.

    Returns:
    - pd.DataFrame: A DataFrame containing gene name, chromosome, start, end, and strand.
    """
    columns = ['seqname', 'source', 'feature', 'start', 'end', 'score', 'strand', 'frame', 'attribute']
    dtype = {'seqname': str, 'source': str, 'feature': str, 'start': int, 'end': int, 'score': str, 'strand': str, 'frame': str, 'attribute': str}

    gtf_df = pd.read_csv(gtf_file_path, sep='\t', comment='#', header=None, names=columns, dtype=dtype)
    
    # Filter for gene features
    gene_df = gtf_df[gtf_df['feature'] == 'gene']
    
    # Extract gene_name from the attributes column
    gene_df['gene_name'] = gene_df['attribute'].str.extract(r'gene_name "([^"]+)"')
    
    return gene_df[['gene_name', 'seqname', 'start', 'end', 'strand']]


def parse_gtf(gtf_file_path):
    """
    Parse a GTF file and extract features into a list of dictionaries.

    Parameters:
    - gtf_file_path (str): Path to the input GTF file.

    Returns:
    - list of dict: List of dictionaries containing feature information.
    """
    # Define column names for the GTF file
    columns = [
        'seqname', 'source', 'feature', 'start', 'end', 'score', 'strand', 'frame', 'attribute'
    ]

    # Specify dtype for the columns to avoid mixed type warning
    dtype = {
        'seqname': str,  
        'source': str,
        'feature': str,
        'start': int,
        'end': int,
        'score': str,  
        'strand': str,
        'frame': str,  
        'attribute': str
    }
    
    # Read the GTF file into a DataFrame
    df = pd.read_csv(gtf_file_path, sep='\t', comment='#', header=None, names=columns, dtype=dtype)
    
    # Initialize an empty list to store features
    features = []

    # Regex to capture key-value pairs
    attribute_pattern = re.compile(
        r'(?P<key>\w+(?: \w+)*)\s'                # Match the key (can contain spaces)
        r'(?P<value>"[^"]*"|\([^)]*\))'            # Match the value (quoted string or parentheses)
    )
    
    # Iterate through each row in the DataFrame
    for _, row in df.iterrows():
        # Parse the attribute column into a dictionary
        attributes = {}
        for match in attribute_pattern.finditer(row['attribute']):
            key = match.group('key')
            value = match.group('value').strip('"')
            if value.startswith('(') and value.endswith(')'):
                # Convert the value inside parentheses into a tuple of integers
                value = tuple(map(int, value.strip('()').split(',')))
            attributes[key] = value
        
        # Create a dictionary for the current feature
        feature = {
            'seqname': row['seqname'],
            'source': row['source'],
            'feature': row['feature'],
            'start': row['start'],
            'end': row['end'],
            'score': row['score'],
            'strand': row['strand'],
            'frame': row['frame'],
            'attributes': attributes
        }
        
        # Append the feature dictionary to the list
        features.append(feature)
    
    return features


####################################################################################################
# Functions for parsing and/or filtering GTF files
# - parse_gtf()
# - parse_gtf_for_genes()
# - filter_gtf_by_genes()
# - extract_genes_from_gtf()

# Example usage:
# gtf_file = "h38.filt.gtf"
# genes_df = parse_gtf_for_genes(gtf_file)
# print(genes_df.head())

# Functions for extracting sequences from GTF files

import pandas as pd
import polars as pl

def truncate_sequences(df, max_length=200):
    """
    Truncate DNA sequences in a DataFrame for display purposes.

    Parameters:
    - df (pd.DataFrame or pl.DataFrame): DataFrame containing gene names and sequences with columns 'gene_name' and 'sequence'.
    - max_length (int): Maximum length of the sequence to display. Default is 200.

    Returns:
    - pd.DataFrame or pl.DataFrame: DataFrame with truncated sequences.

    Example usage:
        sequences_df = pd.DataFrame({
            'gene_name': ['gene1', 'gene2'],
            'sequence': [
                'ATGCGTACGTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGC',
                'CGTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTA']
        })
        truncated_df = truncate_sequences(sequences_df, max_length=10)
        print(truncated_df)
    """
    if isinstance(df, pd.DataFrame):
        truncated_df = df.copy()
        truncated_df['sequence'] = truncated_df['sequence'].apply(lambda seq: seq[:max_length] + '...' if len(seq) > max_length else seq)
        return truncated_df
    elif isinstance(df, pl.DataFrame):
        truncated_df = df.clone()
        truncated_df = truncated_df.with_columns(
            pl.col('sequence').apply(lambda seq: seq[:max_length] + '...' if len(seq) > max_length else seq, return_dtype=pl.Utf8)
        )
        return truncated_df
    else:
        raise TypeError("Unsupported DataFrame type. Please provide a Pandas or Polars DataFrame.")


def normalize_strand(strand):
    """Normalize strand to '+' or '-'."""
    if strand in [1, '+']:
        return '+'
    elif strand in [-1, '-']:
        return '-'
    else:
        raise ValueError(f"Unexpected strand value: {strand}")


def extract_gene_sequences(genes_df, genome_fasta, output_format='dataframe', include_columns=None):
    """
    Extract DNA sequences for genes from the reference genome based on coordinates.

    Parameters:
    - genes_df (pl.DataFrame): 
        Polars DataFrame with gene name, chromosome, start, end, and strand.
        Use extract_genes_from_gtf() to get the DataFrame.

    - genome_fasta (str): Path to the reference genome FASTA file.
    - output_format (str): Output format, either 'dict' or 'dataframe'. Default is 'dict'.

    Returns:
    - dict or pl.DataFrame: 
        If output_format is 'dict', returns a dictionary mapping gene names to their DNA sequences.
        If output_format is 'dataframe', returns a Polars DataFrame with the following columns:
            - 'gene_name': Name of the gene.
            - 'seqname': Chromosome or sequence name.
            - 'gene_id': Gene identifier.
            - 'strand': Strand information ('+' or '-').
            - 'start': Start position (0-based index).
            - 'end': End position.
            - 'sequence': DNA sequence of the gene.
            - Additional columns specified in include_columns, if any.

    Example usage:
        genome_fasta = "/path/to/Homo_sapiens.GRCh38.dna.primary_assembly.fa"
        gene_sequences = extract_gene_sequences(genes_df, genome_fasta)
        print(gene_sequences)
    """
    from Bio import SeqIO

    sequences = []
    
    # Load the genome into a dictionary
    genome = SeqIO.to_dict(SeqIO.parse(genome_fasta, "fasta"))

    if isinstance(genes_df, pd.DataFrame): 
        # genes_df = pl.DataFrame._from_pandas(genes_df)
        genes_df = pl.DataFrame(genes_df)
    
    nns = nps = 0
    for row in tqdm(genes_df.iter_rows(named=True), total=genes_df.height, desc="Processing genes"):
        gene_name = row['gene_name']
        gene_id = row['gene_id']
        chrom = row['seqname']
        start = row['start'] - 1  # Convert to 0-based index
        end = row['end']
        strand = normalize_strand(row['strand'])

        # Ensure the start and end positions are correctly handled
        assert end > start, (
            f"End should be greater than start: strand={strand}, start={start}, end={end}"
        )
        
        # Extract the sequence
        sequence = genome[chrom].seq[start:end]
        
        # Reverse complement if on the negative strand
        if strand == '-':
            sequence = sequence.reverse_complement()
            nns += 1
        else: 
            nps += 1
        # If the gene is on the negative strand (strand == '-'), the sequence is reverse complemented.
        
        # Collect the sequence and additional information
        # sequences[gene_name] = str(sequence)
        sequence_info = {
            'seqname': chrom,
            'gene_name': gene_name,
            'gene_id': gene_id,
            'sequence': str(sequence),
            'strand': strand, 
            'start': start, 
            'end': end
        }

        # Include additional columns if specified
        if include_columns:
            for col in include_columns:
                sequence_info[col] = row[col]

        sequences.append(sequence_info)

    print("[info] Strand information:")
    print(f"... Number of genes on the negative strand: {nns}")
    print(f"... Number of genes on the positive strand: {nps}")
    assert nns > 0 and nps > 0, "Both positive and negative strand genes should be present."

    if output_format == 'dataframe':
        sequences_df = pl.DataFrame(sequences)

        # Reorder columns to place 'sequence' as the last column
        columns = ['gene_name'] + [col for col in sequences_df.columns if col not in ['gene_name', 'sequence']] + ['sequence']
        sequences_df = sequences_df.select(columns)

        return sequences_df
    
    return {seq['gene_name']: seq['sequence'] for seq in sequences}


def extract_transcript_sequences(transcripts_df, genome_fasta, output_format='dict', include_columns=None):
    """
    Extract transcript (pre-mRNA) sequences from the reference genome based on transcript coordinates.

    Parameters:
    - transcripts_df (pd.DataFrame): 
        DataFrame with transcript_id, chromosome (seqname), start, end, and strand.
    - genome_fasta (str): Path to the reference genome FASTA file.
    - output_format (str): Output format, either 'dict' or 'dataframe'. Default is 'dict'.
    - include_columns (list, optional): Additional columns from transcripts_df to include in the output.

    Returns:
    - dict: A dictionary mapping transcript IDs to their DNA sequences or a DataFrame if output_format is 'dataframe'.

    Example usage:
        genome_fasta = "/path/to/Homo_sapiens.GRCh38.dna.primary_assembly.fa"
        transcripts_df = extract_transcripts_from_gtf("h38.filt.gtf")
        transcript_sequences = extract_transcript_sequences(transcripts_df, genome_fasta)
        print(transcript_sequences)
    """
    sequences = []
    
    # Load the genome into a dictionary
    genome = SeqIO.to_dict(SeqIO.parse(genome_fasta, "fasta"))
    
    for _, row in transcripts_df.iterrows():
        transcript_id = row['transcript_id']
        gene_name = row['gene_name']
        chrom = row['seqname']
        start = row['start'] - 1  # Convert to 0-based index for Biopython
        end = row['end']
        strand = row['strand']
        
        # Extract the sequence
        sequence = genome[chrom].seq[start:end]
        
        # Reverse complement if on the negative strand
        if strand == '-':
            sequence = sequence.reverse_complement()
        
        # Collect the sequence and additional information
        sequence_info = {
            'gene_name': gene_name,
            'transcript_id': transcript_id,
            'sequence': str(sequence),
            'strand': strand
        }

        # Include additional columns if specified
        if include_columns:
            for col in include_columns:
                sequence_info[col] = row[col]

        sequences.append(sequence_info)

    if output_format == 'dataframe':
        sequences_df = pd.DataFrame(sequences)

        # Reorder columns to place 'sequence' as the last column
        columns = ['transcript_id'] + [col for col in sequences_df.columns if col not in ['transcript_id', 'sequence']] + ['sequence']
        sequences_df = sequences_df[columns]

        return sequences_df
    
    return {seq['transcript_id']: seq['sequence'] for seq in sequences}


def save_sequences(df, output_file, format='tsv'):
    """
    Save the DNA sequence DataFrame to a file in the specified format.

    Parameters:
    - df (pd.DataFrame or pl.DataFrame): DataFrame containing gene names and sequences with columns 'gene_name' and 'sequence'.
    - output_file (str): Path to the output file.
    - format (str): Format of the output file. Can be 'tsv', 'csv', or 'parquet'. Default is 'tsv'.
    """
    if isinstance(df, pd.DataFrame):
        if format == 'tsv':
            df.to_csv(output_file, sep='\t', index=False)
        elif format == 'csv':
            df.to_csv(output_file, sep=',', index=False)
        elif format == 'parquet':
            df.to_parquet(output_file, index=False)
        else:
            raise ValueError("Unsupported format. Please choose 'tsv', 'csv', or 'parquet'.")
    elif isinstance(df, pl.DataFrame):
        if format == 'tsv':
            df.write_csv(output_file, separator='\t')
        elif format == 'csv':
            df.write_csv(output_file, separator=',')
        elif format == 'parquet':
            df.write_parquet(output_file)
        else:
            raise ValueError("Unsupported format. Please choose 'tsv', 'csv', or 'parquet'.")
    else:
        raise TypeError("Unsupported DataFrame type. Please provide a Pandas or Polars DataFrame.")


def load_sequences(file_path, format='tsv', output_format='pandas'):
    """
    Load the DNA sequence DataFrame from a file in the specified format.

    Parameters:
    - file_path (str): Path to the file.
    - format (str): Format of the file. Can be 'tsv', 'csv', or 'parquet'. Default is 'tsv'.
    - output_format (str): Desired output format ('pandas' or 'polars'). Default is 'pandas'.

    Returns:
    - pd.DataFrame or pl.DataFrame: DataFrame containing gene names and sequences.
    """
    if output_format not in ['pandas', 'polars']:
        raise ValueError("Unsupported output format. Please choose 'pandas' or 'polars'.")

    if format == 'tsv':
        if output_format == 'pandas':
            return pd.read_csv(file_path, sep='\t')
        elif output_format == 'polars':
            return pl.read_csv(file_path, separator='\t')
    elif format == 'csv':
        if output_format == 'pandas':
            return pd.read_csv(file_path)
        elif output_format == 'polars':
            return pl.read_csv(file_path)
    elif format == 'parquet':
        if output_format == 'pandas':
            return pd.read_parquet(file_path)
        elif output_format == 'polars':
            return pl.read_parquet(file_path)
    else:
        raise ValueError("Unsupported format. Please choose 'tsv', 'csv', or 'parquet'.")


def extract_transcripts_from_gtf_v0(gtf_file_path, tx_ids=None, ignore_version=True):
    """
    Extract transcript features from a GTF file, with optional filtering by transcript IDs or gene-transcript pairs.

    Parameters:
    - gtf_file_path (str): Path to the input GTF file.
    - tx_ids (set, list, or dict, optional): A set or list of transcript IDs to filter, or a dictionary where keys are gene names and values are lists of transcript IDs. If None, all transcripts are extracted.
    - ignore_version (bool): If True, ignore version numbers in transcript IDs during matching (default: True).

    Returns:
    - pd.DataFrame: A DataFrame containing transcript_id, gene_name, chromosome, start, end, and strand.

    Example usage:
        gtf_file = "hg19.filt.gtf"
        tx_ids = {"ENSG00000163558": ["ENST00000220876", "ENST00000519716"]}  # Replace with your set/dict of transcript IDs, or use None to extract all transcripts

        transcripts_df = extract_transcripts_from_gtf(gtf_file, tx_ids)
        print(transcripts_df.head())
    """
    columns = ['seqname', 'source', 'feature', 'start', 'end', 'score', 'strand', 'frame', 'attribute']
    dtype = {'seqname': str, 'source': str, 'feature': str, 'start': int, 'end': int, 'score': str, 'strand': str, 'frame': str, 'attribute': str}

    gtf_df = pd.read_csv(gtf_file_path, sep='\t', comment='#', header=None, names=columns, dtype=dtype)
    
    # Filter for transcript features
    transcript_df = gtf_df[gtf_df['feature'] == 'transcript']
    
    # Extract transcript_id and gene_name from the attributes column
    transcript_df['transcript_id'] = transcript_df['attribute'].str.extract(r'transcript_id "([^"]+)"')
    transcript_df['gene_name'] = transcript_df['attribute'].str.extract(r'gene_name "([^"]+)"')
    # NOTE: .extract()
    #  - The .extract() method is used to extract regex matches from a column containing string values. 
    #    It returns a DataFrame where each column corresponds to a capture group defined in the regex pattern.
    #  - The regex patterns are provided as raw strings (r''), which means special characters like backslashes 
    #    are treated literally.
    #  - ([^"]+): This is a capture group that matches one or more characters that are not double quotes.
    
    # If ignoring version numbers, strip them from transcript IDs
    if ignore_version:
        transcript_df['transcript_id'] = transcript_df['transcript_id'].str.split('.').str[0]
        if isinstance(tx_ids, dict):
            tx_ids = {gene: [tx.split('.')[0] for tx in transcripts] for gene, transcripts in tx_ids.items()}
        elif isinstance(tx_ids, (set, list)):
            tx_ids = {tx.split('.')[0] for tx in tx_ids}

    # Determine if filtering is needed
    if tx_ids is not None:
        if isinstance(tx_ids, dict):
            # Filtering by gene-transcript pairs
            transcript_df = transcript_df[
                transcript_df.apply(
                    lambda row: row['gene_name'] in tx_ids and row['transcript_id'] in tx_ids[row['gene_name']], axis=1
                )
            ]
        elif isinstance(tx_ids, (set, list)):
            # Filtering by transcript IDs
            transcript_df = transcript_df[transcript_df['transcript_id'].isin(tx_ids)]
        else:
            raise ValueError("tx_ids should be a set, list, or dictionary.")
    
    return transcript_df[['transcript_id', 'gene_name', 'seqname', 'start', 'end', 'strand']]


def extract_transcripts_from_gtf(gtf_file_path, tx_ids=None, ignore_version=True):
    """
    Extract transcript features from a GTF file, with optional filtering by transcript IDs or gene-transcript pairs.

    Parameters:
    - gtf_file_path (str): Path to the input GTF file.
    - tx_ids (dict, optional): A dictionary where keys are gene names or gene IDs and values are lists of transcript IDs.
                               If None, all transcripts are extracted.
    - ignore_version (bool): If True, ignore version numbers in transcript IDs during matching (default: True).

    Returns:
    - pd.DataFrame: A DataFrame containing transcript_id, gene_name, gene_id, chromosome, start, end, and strand.

    Example usage:
        gtf_file = "hg19.filt.gtf"
        tx_ids = {'STMN2': ['ENST00000220876'], 'UNC13A': ['ENST00000519716']}  # Example with gene names
        
        # or
        
        tx_ids = {'ENSG00000104435': ['ENST00000220876'], 'ENSG00000130477': ['ENST00000519716']}  # Example with gene IDs
        
        transcripts_df = extract_transcripts_from_gtf(gtf_file, tx_ids)
        print(transcripts_df.head())
    """
    columns = ['seqname', 'source', 'feature', 'start', 'end', 'score', 'strand', 'frame', 'attribute']
    dtype = {'seqname': str, 'source': str, 'feature': str, 'start': int, 'end': int, 'score': str, 'strand': str, 'frame': str, 'attribute': str}

    gtf_df = pd.read_csv(gtf_file_path, sep='\t', comment='#', header=None, names=columns, dtype=dtype)
    
    # Filter for transcript features
    transcript_df = gtf_df[gtf_df['feature'] == 'transcript']
    
    # Extract transcript_id, gene_name, and gene_id from the attributes column
    transcript_df['transcript_id'] = transcript_df['attribute'].str.extract(r'transcript_id "([^"]+)"')
    transcript_df['gene_name'] = transcript_df['attribute'].str.extract(r'gene_name "([^"]+)"')
    transcript_df['gene_id'] = transcript_df['attribute'].str.extract(r'gene_id "([^"]+)"')
    
    # If ignoring version numbers, strip them from transcript IDs
    if ignore_version:
        transcript_df['transcript_id'] = transcript_df['transcript_id'].str.split('.').str[0]
        if isinstance(tx_ids, dict):
            # Determine if the dictionary keys are gene names or gene IDs
            if all(key.startswith('ENSG') for key in tx_ids):  # Assuming gene IDs start with 'ENSG'
                # Strip version numbers from the transcript IDs in tx_ids
                tx_ids = {gene: [tx.split('.')[0] for tx in transcripts] for gene, transcripts in tx_ids.items()}
            else:  # Assume keys are gene names
                tx_ids = {gene: [tx.split('.')[0] for tx in transcripts] for gene, transcripts in tx_ids.items()}
        elif isinstance(tx_ids, (set, list)):
            tx_ids = {tx.split('.')[0] for tx in tx_ids}

    # Determine if filtering is needed
    if tx_ids is not None:
        if isinstance(tx_ids, dict):
            # Determine if the dictionary keys are gene names or gene IDs
            if all(key.startswith('ENSG') for key in tx_ids):  # Keys are gene IDs
                transcript_df = transcript_df[
                    transcript_df.apply(
                        lambda row: row['gene_id'] in tx_ids and row['transcript_id'] in tx_ids[row['gene_id']], axis=1
                    )
                ]
            else:  # Keys are gene names
                transcript_df = transcript_df[
                    transcript_df.apply(
                        lambda row: row['gene_name'] in tx_ids and row['transcript_id'] in tx_ids[row['gene_name']], axis=1
                    )
                ]
        elif isinstance(tx_ids, (set, list)):
            # Filtering by transcript IDs
            transcript_df = transcript_df[transcript_df['transcript_id'].isin(tx_ids)]
        else:
            raise ValueError("tx_ids should be a set, list, or dictionary.")
    
    return transcript_df[['transcript_id', 'gene_name', 'gene_id', 'seqname', 'start', 'end', 'strand']]


def extract_genes_from_gtf(gtf_file_path, gene_names=None, gene_ids=None):
    """
    Extract gene features from a GTF file, with optional filtering by gene names.

    Parameters:
    - gtf_file_path (str): Path to the input GTF file.
    - gene_names (set or list, optional): A set or list of gene names to filter. If None, all genes are extracted.

    Returns:
    - pd.DataFrame: A DataFrame containing gene name, chromosome, start, end, and strand.

    Example usage:
        gtf_file = "h38.filt.gtf"
        gene_names = {"STMN2", "UNC13A"}  # Replace with your set of gene names, or use None to extract all genes

        genes_df = extract_genes_from_gtf(gtf_file, gene_names)
        print(genes_df.head())
    """
    columns = ['seqname', 'source', 'feature', 'start', 'end', 'score', 'strand', 'frame', 'attribute']
    dtype = {'seqname': str, 'source': str, 'feature': str, 'start': int, 'end': int, 'score': str, 'strand': str, 'frame': str, 'attribute': str}

    gtf_df = pd.read_csv(gtf_file_path, sep='\t', comment='#', header=None, names=columns, dtype=dtype)
    
    # Filter for gene features
    gene_df = gtf_df[gtf_df['feature'] == 'gene']
    
    # Extract gene_name from the attributes column
    gene_df['gene_name'] = gene_df['attribute'].str.extract(r'gene_name "([^"]+)"')
    gene_df['gene_id'] = gene_df['attribute'].str.extract(r'gene_id "([^"]+)"')
    
    # Optional filtering by gene names
    if gene_names is not None:
        gene_df = gene_df[gene_df['gene_name'].isin(gene_names)]

    if gene_ids is not None:
        gene_df = gene_df[gene_df['gene_id'].isin(gene_ids)]
    
    return gene_df[['gene_name', 'gene_id', 'seqname', 'start', 'end', 'strand']]



def extract_sequences_for_genes(gtf_file, gene_names, reference_genome, output_fasta_file, **kargs):
    """
    Extract DNA sequences for specific genes using gffread.

    Parameters:
    - gtf_file (str): Path to the input GTF file.
    - gene_names (set or list): A set or list of gene names to filter.
    - reference_genome (str): Path to the reference genome FASTA file.
    - output_fasta_file (str): Path to the output FASTA file to store the sequences.
    """
    # import subprocess
    # import tempfile

    # Filter GTF
    filtered_gtf_df = filter_gtf_by_genes(gtf_file, gene_names)
    
    # Create a temporary file for the filtered GTF
    with tempfile.NamedTemporaryFile(delete=False, suffix=".gtf") as temp_filtered_gtf:
        filtered_gtf_file = temp_filtered_gtf.name
        filtered_gtf_df.to_csv(filtered_gtf_file, sep='\t', index=False, header=False)
    
    try:
        # Run gffread to extract sequences
        subprocess.run([
            "gffread",
            "-w", output_fasta_file,
            "-g", reference_genome,
            filtered_gtf_file
        ], check=True)
    finally:
        # Clean up the temporary file
        os.remove(filtered_gtf_file)
    # NOTE:  
    # Internally, gffread extracts sequences by:
    #   - Identifying exons or CDS entries associated with each transcript.
    #   - Concatenating these sequences based on the order defined in the GTF/GFF file.
    #   - Outputting the spliced mRNA or CDS sequence.
    # If you need to extract full gene sequences (including introns and exons), you should instead:
    #   - Parse the GTF file to get the coordinates for the entire gene (from the start of the first exon 
    #     to the end of the last exon). 
    #   - Extract the sequence from the reference genome using a tool or script that handles genomic intervals, 
    #     like using Biopython directly or bedtools.


####################################################################################################

def create_gtf(features, output_gtf_path):
    """
    Create a GTF file from a list of features.

    Parameters:
    features (list of dict): List of dictionaries containing feature information.
        Each dictionary should have the following keys:
            - seqname (str): Name of the sequence (e.g., chromosome).
            - source (str): Source of the feature (e.g., "uORFexplorer").
            - feature (str): Type of feature (e.g., "transcript", "exon", "CDS").
            - start (int): Start position of the feature.
            - end (int): End position of the feature.
            - score (str): Score of the feature (use "." if not applicable).
            - strand (str): Strand of the feature ("+" or "-").
            - frame (str): Frame of the feature (use "." if not applicable).
            - attributes (dict): Additional attributes where keys are attribute names and values are attribute values.

    output_gtf_path (str): Path to the output GTF file.
    """
    # import pandas as pd
    # import csv

    # Convert the features list into a DataFrame
    data = []
    for feature in features:
        attributes_str = ' '.join(f'{key} "{value}";' for key, value in feature['attributes'].items())
        data.append([
            feature['seqname'],
            feature['source'],
            feature['feature'],
            feature['start'],
            feature['end'],
            feature['score'],
            feature['strand'],
            feature['frame'],
            attributes_str
        ])
    
    df = pd.DataFrame(data, columns=[
        'seqname', 'source', 'feature', 'start', 'end', 'score', 'strand', 'frame', 'attribute'
    ])
    
    # Write the DataFrame to a GTF file
    df.to_csv(output_gtf_path, sep='\t', header=False, index=False, 
              quoting=csv.QUOTE_NONE, quotechar='')

    return


def parse_gtf_v0(gtf_file_path):
    """
    Parse a GTF file and extract features into a list of dictionaries.

    Parameters:
    gtf_file_path (str): Path to the input GTF file.

    Returns:
    list of dict: List of dictionaries containing feature information.
    """
    # Define column names for the GTF file
    columns = [
        'seqname', 'source', 'feature', 'start', 'end', 'score', 'strand', 'frame', 'attribute'
    ]

    # Specify dtype for the columns to avoid mixed type warning
    dtype = {

        # The name of the sequence (e.g., chromosome). This can be a string (e.g., "chr1") or 
        # an integer (e.g., "1"), so it's best to treat it as a string.
        'seqname': str,  

        'source': str,
        'feature': str,
        'start': int,
        'end': int,

        # A score between 0 and 1000. This is often a floating-point number but can also be a dot (".") 
        # if not applicable, so it's treated as a string.
        'score': str,  

        'strand': str,
        'frame': str,  # The reading frame (0, 1, 2) or a dot (".") if not applicable. This is a string.
        'attribute': str
    }
    
    # Read the GTF file into a DataFrame
    df = pd.read_csv(gtf_file_path, sep='\t', comment='#', header=None, names=columns, dtype=dtype)
    
    # Initialize an empty list to store features
    features = []
    
    # Iterate through each row in the DataFrame
    for _, row in df.iterrows():
        # Parse the attribute column into a dictionary
        attributes = {}
        for attribute in row['attribute'].split(';'):
            if attribute.strip():
                # Split on the first space after the key to handle multi-word keys
                key_value = attribute.strip().split(' ', 1)
                key = key_value[0]
                value = key_value[1].strip('"') if len(key_value) > 1 else ''
                attributes[key] = value
        
        # Create a dictionary for the current feature
        feature = {
            'seqname': row['seqname'],
            'source': row['source'],
            'feature': row['feature'],
            'start': row['start'],
            'end': row['end'],
            'score': row['score'],
            'strand': row['strand'],
            'frame': row['frame'],
            'attributes': attributes
        }
        
        # Append the feature dictionary to the list
        features.append(feature)
    
    return features


def parse_gff(gtf_file_path, file_type='GFF'):
    """
    Parse a GFF/GTF file and extract features into a list of dictionaries.

    Parameters:
    gtf_file_path (str): Path to the input GTF/GFF file.
    file_type (str): Type of the file ('GFF' or 'GTF').

    Returns:
    list of dict: List of dictionaries containing feature information.
    """
    # Define column names for the GFF/GTF file
    columns = [
        'seqname', 'source', 'feature', 'start', 'end', 'score', 'strand', 'frame', 'attribute'
    ]
    
    # Read the GFF/GTF file into a DataFrame
    df = pd.read_csv(gtf_file_path, sep='\t', comment='#', header=None, names=columns)
    
    # Initialize an empty list to store features
    features = []
    
    # Iterate through each row in the DataFrame
    for _, row in df.iterrows():
        # Parse the attribute column into a dictionary
        attributes = {}
        if file_type == 'GTF':
            # GTF format: key "value"; key2 "value2";
            for attribute in row['attribute'].split(';'):
                if attribute.strip():
                    key, value = attribute.strip().split(' ', 1)
                    attributes[key] = value.strip('"')
        elif file_type == 'GFF':
            # GFF format: key=value; key2=value2;
            for attribute in row['attribute'].split(';'):
                if attribute.strip():
                    key, value = attribute.strip().split('=', 1)
                    attributes[key] = value.strip('"')

        # Create a dictionary for the current feature
        feature = {
            'seqname': row['seqname'],
            'source': row['source'],
            'feature': row['feature'],
            'start': row['start'],
            'end': row['end'],
            'score': row['score'],
            'strand': row['strand'],
            'frame': row['frame'],
            'attributes': attributes
        }
        
        # Append the feature dictionary to the list
        features.append(feature)
    
    return features


def find_orfs(sequence, min_length=30):
    """
    Find ORFs in the given sequence.

    Parameters:
    sequence (str): The nucleotide sequence to search for ORFs.
    min_length (int): Minimum length of ORFs to consider.

    Returns:
    list of tuple: List of (start, end, frame) tuples representing the ORFs found.

    Example Usage: 

    sequence = "ATGCGTAAATGTAGATGCGTTAA"
    orfs = find_orfs(sequence)
    print(orfs)  # Output: [(0, 9, 0), (9, 18, 0), (12, 21, 0)]
    """
    # from Bio import SeqIO

    orfs = []
    start_codon = "ATG"
    stop_codons = {"TAA", "TAG", "TGA"}

    for frame in range(3):  # Check all three reading frames
        for i in range(frame, len(sequence) - 2, 3):
            codon = sequence[i:i+3]
            if codon == start_codon:
                for j in range(i, len(sequence) - 2, 3):
                    stop_codon = sequence[j:j+3]
                    if stop_codon in stop_codons:
                        orf_length = j - i + 3
                        if orf_length >= min_length:
                            orfs.append((i, j + 3, frame))
                        break
    return orfs


def subset_genes_from_gtf(gtf_file, gene_list, output_file):
    return extract_genes(gtf_file, gene_list, output_file)


def extract_genes_with_db(db_fn, gtf_file, gene_list, output_file):
    """
    Extracts specific genes and their associated features from a GTF file.

    Parameters:
    - gtf_file (str): Path to the GTF file.
    - gene_list (str): Path to a text file containing gene names to extract.
    - output_file (str): Path to the output GTF file with the extracted genes and features.
    """
    import gffutils
    # Create a database from the GTF file if it doesn't exist
    # db_fn = f"{gtf_file}.db"
    if not os.path.exists(db_fn):
        gffutils.create_db(gtf_file, db_fn, id_spec="gene_id", merge_strategy="merge", disable_infer_transcripts=True, disable_infer_genes=True)
    
    db = gffutils.FeatureDB(db_fn)

    # Read the gene list
    with open(gene_list, 'r') as f:
        genes_of_interest = {line.strip() for line in f}
    
    with open(output_file, 'w') as out:
        for gene_name in genes_of_interest:
            try:
                gene = db[gene_name]
                out.write(str(gene) + '\n')
                for related_feature in db.children(gene):
                    out.write(str(related_feature) + '\n')
            except gffutils.exceptions.FeatureNotFoundError:
                print(f"Gene {gene_name} not found in the GTF file")


def extract_genes(gtf_file, gene_list, output_file):
    """

    Example Usage: 

    gtf_file = 'Homo_sapiens.GRCh38.104.gtf'
    gene_list = 'gene_list.txt'
    output_file = 'subset.gtf'

    create_subset_gtf(gtf_file, gene_list, output_file)

    """
    import gtftools
    from gtftools.gtf import GTF

    with open(gene_list, 'r') as f:
        genes_of_interest = {line.strip() for line in f}
    
    gtf = GTF(gtf_file)
    
    with open(output_file, 'w') as out:
        for feature in gtf:
            if feature.feature_type == 'gene' and feature.attribute('gene_name') in genes_of_interest:
                out.write(str(feature) + '\n')
                for related_feature in gtf.children(feature):
                    out.write(str(related_feature) + '\n')

def demo_parse_gtf(): 
    data_path = "/path/to/meta-spliceai/data/ensembl/"
    plot_path = os.path.join(data_path, "plot")
    uorf_tx_gtf = os.path.join(data_path, "h38.final.gtf") 

    features = parse_gtf(uorf_tx_gtf)  # Parse GTF file into a list of dictionaries
        
    transcript_data = {}
    splice_site_summary = {'with_splice_sites': [], 'without_splice_sites': []}
    
    for feature in features:
        transcript_id = feature['attributes'].get('transcript_id')
        if not transcript_id:
            continue
        
        if transcript_id not in transcript_data:
            transcript_data[transcript_id] = {'exons': [], 'cds': [], 'splice_sites': []}
        
        start = feature['start']
        end = feature['end']
        feature_type = feature['feature']
        
        # Extract exons
        if feature_type == 'exon':
            transcript_data[transcript_id]['exons'].append((start, end))
        
        # Extract CDS
        elif feature_type == 'CDS':
            transcript_data[transcript_id]['cds'].append((start, end))
        
        # Extract splice sites if present
        if 'splice sites' in feature['attributes']:
            splice_sites = feature['attributes']['splice sites']
            # print(f"[test] Extracted Splice Sites: {splice_sites}")  
            transcript_data[transcript_id]['splice_sites'].append(splice_sites)
        # else: 
        #     print(f"[test] No Splice Sites Found for Transcript: {transcript_id}")
        #     print(feature['attributes'])

    # Check for splice site annotations
    for transcript_id, data in transcript_data.items():
        if data['splice_sites']:
            splice_site_summary['with_splice_sites'].append(transcript_id)
        else:
            splice_site_summary['without_splice_sites'].append(transcript_id)
    
    return transcript_data, splice_site_summary


def demo_extract_transcript_sequences():

    data_prefix = "/path/to/meta-spliceai/data/ensembl"
    local_dir = "/path/to/meta-spliceai/data/ensembl/ALS"
    genome_annot = os.path.join(data_prefix, "Homo_sapiens.GRCh38.112.gtf") 

    genome_fasta = os.path.join(
        data_prefix, "Homo_sapiens.GRCh38.dna.primary_assembly.fa") # "/path/to/Homo_sapiens.GRCh38.dna.primary_assembly.fa"
    
    gene_names = {"STMN2", "UNC13A"}
    tx_ids = {'STMN2': ['ENST00000220876', ], 
              'UNC13A': ['ENST00000519716', ]}

    print("[info] Extract transcript sequences")
    # Example inputs 
    # tx_ids = {'STMN2': ['ENST00000220876'], 'UNC13A': ['ENST00000519716']}  # Example with gene names
    # or
    # tx_ids = {'ENSG00000104435': ['ENST00000220876'], 'ENSG00000130477': ['ENST00000519716']}  # Example with gene IDs

    # Extract transcripts and their features
    gtf_file = genome_annot
    transcripts_df = extract_transcripts_from_gtf(gtf_file, tx_ids, ignore_version=True)
    print(transcripts_df.head())  # Display the extracted transcript information

    additional_columns = ['seqname', 'start', 'end', 'strand', ]
    seq_df = extract_transcript_sequences(transcripts_df, genome_fasta, output_format='dataframe', include_columns=additional_columns)
    
    format = 'tsv'
    output_path = os.path.join(local_dir, f"tx_sequence.{format}")

    print(f"[i/o] Saving transcript sequences to:\n{output_path}\n")
    print("... cols(seq_df): ", list(seq_df.columns))
    save_sequences(seq_df, output_path, format='tsv')
    
    print("[i/o] Truncated Sequences:")
    print(truncate_sequences(seq_df.to_pandas(), max_length=200))

    return seq_df


def demo_extract_gene_sequences(gene_names=None):

    data_prefix = "/path/to/meta-spliceai/data/ensembl"

    # local_dir = "/path/to/meta-spliceai/data/ensembl/ALS"
    local_dir = "/path/to/meta-spliceai/data/ensembl/test"
    # Create the directory if it doesn't exist
    os.makedirs(local_dir, exist_ok=True)

    genome_annot = os.path.join(data_prefix, "Homo_sapiens.GRCh38.112.gtf") 
    genome_fasta = os.path.join(
        data_prefix, "Homo_sapiens.GRCh38.dna.primary_assembly.fa") # "/path/to/Homo_sapiens.GRCh38.dna.primary_assembly.fa"

    genes_df = parse_gtf_for_genes(genome_annot) 
    # header: ['gene_name', 'seqname', 'start', 'end', 'strand'] 

    # Test
    # ------------------------------------------------
    print(genes_df.head())
    print(f"[parse] Found n={genes_df['gene_name'].nunique()} genes")
    print(f"... shape(genes_df): {genes_df.shape}")  # (63140, 5)

    num_null_gene_names = genes_df['gene_name'].isna().sum()
    print(f"[parse] Number of NaN or null values in 'gene_name' column: {num_null_gene_names}")  # 20470

    # ------------------------------------------------
    print("-" * 80); print()

    if gene_names is None: 
        gene_names = {"STMN2", "UNC13A"} 
    genes_df = extract_genes_from_gtf(genome_annot, gene_names=gene_names)
    # header: ['gene_name', 'seqname', 'start', 'end', 'strand'] 

    print(f"[info] Focusing on specific gene set:\n{list(gene_names)}\n ...")
    print(genes_df.head())
    print(f"[parse] Found n={genes_df['gene_name'].nunique()} genes")
    print(f"... shape(genes_df): {genes_df.shape}")

    additional_columns = ['seqname', 'start', 'end', 'strand', ]
    seq_df = extract_gene_sequences(genes_df, genome_fasta, output_format='dataframe', include_columns=additional_columns)
    # header: ['gene_name', 'sequence'] + additional columns

    format = 'tsv'
    output_path = os.path.join(local_dir, f"gene_sequence.{format}")

    print(f"[i/o] Saving gene sequences to:\n{output_path}\n")
    print("... cols(seq_df): ", list(seq_df.columns))
    save_sequences(seq_df, output_path, format='tsv')
    
    print("[i/o] Truncated Sequences:")
    print(truncate_sequences(seq_df.to_pandas(), max_length=200))


def demo(): 

    # Count sequences in the new FASTA file
    # num_sequences = count_sequences('/path/to/meta-spliceai/data/PR1/PR1.fa')
    # print(f"Number of sequences: {num_sequences}")

    # Parse GTF File (involving multiword keys and values in attribute)
    # demo_parse_gtf()

    # Extract Gene Sequences from GTF File
    # demo_extract_gene_sequences()

    # Extract Transcript Sequences from GTF File
    demo_extract_transcript_sequences()


if __name__ == "__main__":
    demo()
