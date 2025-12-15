import os
import pandas as pd
import polars as pl  # pip install polars
from tqdm import tqdm

import pybedtools
from pybedtools import BedTool
from Bio import SeqIO

def extract_gtf_annotations_by_biopython(gtf_file, transcript_id):
    """
    Extract exon, CDS, 5'UTR, and 3'UTR annotations for a specific transcript from a GTF file using Biopython.

    Parameters:
    - gtf_file (str): Path to the GTF file containing transcript annotations.
    - transcript_id (str): The transcript ID to extract annotations for.

    Returns:
    - annotations (dict): Dictionary of features (exons, CDS, UTRs) with their coordinates.
    """
    from BCBio import GFF  # pip install bcbio-gff
    from Bio import SeqIO

    annotations = {'exons': [], 'CDS': [], '5UTR': [], '3UTR': []}
    
    # Read GTF file using GFF parsing in Biopython
    with open(gtf_file) as gtf_handle:
        for record in GFF.parse(gtf_handle):
            for feature in record.features:
                if 'transcript_id' in feature.qualifiers and transcript_id in feature.qualifiers['transcript_id']:
                    # Extract exons
                    if feature.type == "exon":
                        annotations['exons'].append((record.id, feature.location.start, feature.location.end, feature.strand))

                    # Extract CDS regions
                    elif feature.type == "CDS":
                        annotations['CDS'].append((record.id, feature.location.start, feature.location.end, feature.strand))

                    # Extract 5'UTR and 3'UTR regions
                    elif feature.type == "five_prime_UTR":
                        annotations['5UTR'].append((record.id, feature.location.start, feature.location.end, feature.strand))
                    elif feature.type == "three_prime_UTR":
                        annotations['3UTR'].append((record.id, feature.location.start, feature.location.end, feature.strand))

    return annotations


def extract_gtf_annotations_v0(gtf_file, transcript_id):
    """
    Extract exon, CDS, 5'UTR, and 3'UTR annotations for a specific transcript from a GTF file using pybedtools.

    Parameters:
    - gtf_file (str): Path to the GTF file containing transcript annotations.
    - transcript_id (str): The transcript ID to extract annotations for.

    Returns:
    - annotations (dict): Dictionary of features (exons, CDS, UTRs) with their coordinates.
    """
    # from pybedtools import BedTool
    gtf = BedTool(gtf_file)
    
    # Filter the GTF for exons, CDS, 5'UTR, and 3'UTR annotations
    transcript_exons = gtf.filter(lambda x: x.fields[2] == "exon" and 'transcript_id "{}"'.format(transcript_id) in x.fields[8]).saveas()
    transcript_cds = gtf.filter(lambda x: x.fields[2] == "CDS" and 'transcript_id "{}"'.format(transcript_id) in x.fields[8]).saveas()
    transcript_5utr = gtf.filter(lambda x: x.fields[2] == "five_prime_UTR" and 'transcript_id "{}"'.format(transcript_id) in x.fields[8]).saveas()
    transcript_3utr = gtf.filter(lambda x: x.fields[2] == "three_prime_UTR" and 'transcript_id "{}"'.format(transcript_id) in x.fields[8]).saveas()
    # NOTE: When using pybedtools.BedTool and performing operations like filtering with a lambda function, 
    #       the results are typically in-memory objects. To convert these results to a DataFrame, 
    #       you need to save them to a temporary file first.

    # Convert the filtered BedTool objects to DataFrames
    def to_dataframe(bedtool_obj):
        if len(bedtool_obj) > 0:
            return bedtool_obj.to_dataframe(names=["chrom", "source", "feature", "start", "end", "score", "strand", "frame", "attribute"])
        else:
            return pd.DataFrame()  # Return an empty DataFrame if no features were found

    exon_df = to_dataframe(transcript_exons)
    cds_df = to_dataframe(transcript_cds)
    utr5_df = to_dataframe(transcript_5utr)
    utr3_df = to_dataframe(transcript_3utr)
    
    # Combine the annotations into a dictionary, checking for empty DataFrames
    annotations = {
        'exons': exon_df[['chrom', 'start', 'end', 'strand']].values.tolist() if not exon_df.empty else [],
        'CDS': cds_df[['chrom', 'start', 'end', 'strand']].values.tolist() if not cds_df.empty else [],
        '5UTR': utr5_df[['chrom', 'start', 'end', 'strand']].values.tolist() if not utr5_df.empty else [],
        '3UTR': utr3_df[['chrom', 'start', 'end', 'strand']].values.tolist() if not utr3_df.empty else []
    }
    
    return annotations


def extract_gtf_annotations(gtf_file, transcript_id):
    """
    Extract exon, CDS, 5'UTR, and 3'UTR annotations for a specific transcript from a GTF file using pybedtools.

    Parameters:
    - gtf_file (str): Path to the GTF file containing transcript annotations.
    - transcript_id (str): The transcript ID to extract annotations for.

    Returns:
    - annotations (dict): Dictionary of features (exons, CDS, UTRs) with their coordinates.
    """
    # from pybedtools import BedTool
    gtf = BedTool(gtf_file)
    
    # Filter the GTF for exons and CDS annotations
    transcript_exons = gtf.filter(lambda x: x.fields[2] == "exon" and f'transcript_id "{transcript_id}"' in x.fields[8]).saveas()
    transcript_cds = gtf.filter(lambda x: x.fields[2] == "CDS" and f'transcript_id "{transcript_id}"' in x.fields[8]).saveas()

    # Convert the filtered BedTool objects to DataFrames
    def to_dataframe(bedtool_obj):
        if len(bedtool_obj) > 0:
            return bedtool_obj.to_dataframe(names=["chrom", "source", "feature", "start", "end", "score", "strand", "frame", "attribute"])
        else:
            return pd.DataFrame()  # Return an empty DataFrame if no features were found

    exon_df = to_dataframe(transcript_exons)
    cds_df = to_dataframe(transcript_cds)

    # Sort the DataFrames by start coordinate
    exon_df = exon_df.sort_values(by='start')
    cds_df = cds_df.sort_values(by='start')

    # Get transcript coordinates
    if not exon_df.empty:
        chrom = exon_df['chrom'].iloc[0]
        strand = exon_df['strand'].iloc[0]
        transcript_start = exon_df['start'].min()
        transcript_end = exon_df['end'].max()
    else:
        return {}  # If no exons are found, return an empty dictionary

    # Infer 5'UTR and 3'UTR based on strand and CDS regions
    if not cds_df.empty:
        if strand == '+':
            # Ensure that UTR coordinates do not go out of the transcript bounds
            utr5_start, utr5_end = transcript_start, max(transcript_start, cds_df['start'].min() - 1)
            utr3_start, utr3_end = min(transcript_end, cds_df['end'].max() + 1), transcript_end
        else:
            utr5_start, utr5_end = min(transcript_end, cds_df['end'].max() + 1), transcript_end
            utr3_start, utr3_end = transcript_start, max(transcript_start, cds_df['start'].min() - 1)

        # Ensure the coordinates are valid, i.e., start < end
        utr5_df = pd.DataFrame([[chrom, utr5_start, utr5_end, strand]], columns=['chrom', 'start', 'end', 'strand']) \
            if utr5_start < utr5_end else pd.DataFrame()

        utr3_df = pd.DataFrame([[chrom, utr3_start, utr3_end, strand]], columns=['chrom', 'start', 'end', 'strand']) \
            if utr3_start < utr3_end else pd.DataFrame()
    else:
        # If no CDS is found, the entire transcript could be non-coding or incomplete
        utr5_df = pd.DataFrame()
        utr3_df = pd.DataFrame()

    # Combine the annotations into a dictionary, checking for empty DataFrames
    annotations = {
        'exons': exon_df[['chrom', 'start', 'end', 'strand']].values.tolist() if not exon_df.empty else [],
        'CDS': cds_df[['chrom', 'start', 'end', 'strand']].values.tolist() if not cds_df.empty else [],
        '5UTR': utr5_df[['chrom', 'start', 'end', 'strand']].values.tolist() if not utr5_df.empty else [],
        '3UTR': utr3_df[['chrom', 'start', 'end', 'strand']].values.tolist() if not utr3_df.empty else []
    }
    
    return annotations


def extract_transcripts_from_gtf_with_pandas(gtf_file_path, tx_ids=None, ignore_version=True):
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


def parse_attributes(attribute_str):
    """
    Parse the attribute column from a GTF file into a dictionary of key-value pairs.

    Similar to parse_attributes_simple(), but remove keys do not have values. 

    Parameters:
    attribute_str (str): A string containing the attributes from a GTF file.

    Returns:
    attributes (dict): A dictionary with attribute keys and their corresponding values.
    """
    attributes = {}
    for attribute in attribute_str.split(';'):
        if attribute.strip():
            parts = attribute.strip().split(' ', 1)
            if len(parts) == 2:  # Ensure there is a value
                key, value = parts
                attributes[key] = value.strip('"')
    return attributes


def extract_transcripts_from_gtf(gtf_file_path, tx_ids=None, ignore_version=True, use_polars=True):
    """
    Extract transcript features from a GTF file, with optional filtering by transcript IDs or gene-transcript pairs.

    Parameters:
    - gtf_file_path (str): Path to the input GTF file.
    - tx_ids (dict, optional): A dictionary where keys are gene names or gene IDs and values are lists of transcript IDs.
                               If None, all transcripts are extracted.
    - ignore_version (bool): If True, ignore version numbers in transcript IDs during matching (default: True).

    Returns:
    - pl.DataFrame: A Polars DataFrame containing transcript_id, gene_name, gene_id, chromosome, start, end, and strand.

    Example usage:
        gtf_file = "hg19.filt.gtf"
        tx_ids = {'STMN2': ['ENST00000220876'], 'UNC13A': ['ENST00000519716']}  # Example with gene names
        
        # or
        
        tx_ids = {'ENSG00000104435': ['ENST00000220876'], 'ENSG00000130477': ['ENST00000519716']}  # Example with gene IDs
        
        transcripts_df = extract_transcripts_from_gtf(gtf_file, tx_ids)
        print(transcripts_df.head())
    """
    if not use_polars:
        return extract_transcripts_from_gtf_with_pandas(gtf_file_path, tx_ids, ignore_version)

    columns = ['seqname', 'source', 'feature', 'start', 'end', 'score', 'strand', 'frame', 'attribute']

    # Read the GTF file using Polars
    gtf_df = \
        pl.read_csv(gtf_file_path, 
                    separator='\t', 
                    comment_prefix='#', 
                    has_header=False, 
                    new_columns=columns, 
                    schema_overrides={'seqname': pl.Utf8})

    # Filter for transcript features
    transcript_df = gtf_df.filter(pl.col('feature') == 'transcript')
    # Polars uses .filter() for filtering rows and .select() for selecting specific columns.

    # Extract transcript_id, gene_name, and gene_id from the attributes column
    transcript_df = transcript_df.with_columns([
        pl.col('attribute').str.extract(r'transcript_id "([^"]+)"').alias('transcript_id'),
        pl.col('attribute').str.extract(r'gene_name "([^"]+)"').alias('gene_name'),
        pl.col('attribute').str.extract(r'gene_id "([^"]+)"').alias('gene_id')
    ])
    # NOTE: Polars provides .str.extract() for extracting substrings and .str.split() 
    #       followed by .arr.first() to split and get the first part (useful for stripping version numbers).

    # If ignoring version numbers, strip them from transcript IDs
    if ignore_version:
        # transcript_df = transcript_df.with_columns(
        #     pl.col('transcript_id').str.split('.').arr.first().alias('transcript_id')
        # )
        # NOTE: This encountered (expected FixedSizeList, got list[str]), which suggests that 
        #       Polars expects a fixed-size list, but the column contains variable-length lists.

        # solution 1
        # transcript_df = transcript_df.with_columns(
        #     pl.col('transcript_id').apply(lambda x: x.split('.')[0]).alias('transcript_id')
        # )
        # the lambda function splits the string on the . character and takes the first part (index 0)

        # solution 2 (x)
        # transcript_df = transcript_df.with_columns(
        #     pl.col('transcript_id').str.split_exact(".", 1).struct.field(0).alias('transcript_id')
        # )
        # Splits the string at the specified delimiter (in this case, .) and ensures that the split results in exactly two parts.
        # .alias() renames the resulting column back to transcript_id

        # solution 3: Use .arr.get() to access the first part of the split result
        transcript_df = transcript_df.with_columns(
            pl.col('transcript_id').str.extract(r"^([^\.]+)", 1).alias('transcript_id')
        )
        # .str.extract(r"^([^\.]+)", 1: This extracts the part of the transcript_id before the first dot (.)

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
            # Filter based on gene_id or gene_name
            if all(key.startswith('ENSG') for key in tx_ids):  # Keys are gene IDs
                transcript_df = transcript_df.filter(
                    (pl.col('gene_id').is_in(list(tx_ids.keys()))) &
                    (pl.col('transcript_id').is_in([tx for sublist in tx_ids.values() for tx in sublist]))
                )
            else:  # Keys are gene names
                transcript_df = transcript_df.filter(
                    (pl.col('gene_name').is_in(list(tx_ids.keys()))) &
                    (pl.col('transcript_id').is_in([tx for sublist in tx_ids.values() for tx in sublist]))
                )
        elif isinstance(tx_ids, (set, list)):
            # Filtering by transcript IDs
            transcript_df = transcript_df.filter(pl.col('transcript_id').is_in(tx_ids))
        else:
            raise ValueError("tx_ids should be a set, list, or dictionary.")
    
    return transcript_df.select(['transcript_id', 'gene_name', 'gene_id', 'seqname', 'start', 'end', 'strand'])


def extract_genes_from_gtf_pandas(gtf_file_path, gene_names=None):
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
    
    return gene_df[['gene_name', 'gene_id', 'seqname', 'start', 'end', 'strand']]


def extract_genes_from_gtf(gtf_file_path, gene_names=None, use_polars=True):
    """
    Extract gene features from a GTF file, with optional filtering by gene names or gene IDs (starting with ENSG).

    Parameters:
    - gtf_file_path (str): Path to the input GTF file.
    - gene_names (set or list, optional): A set or list of gene names or gene IDs to filter. If None, all genes are extracted.

    Returns:
    - pl.DataFrame: A Polars DataFrame containing gene name, chromosome, start, end, and strand.

    Example usage:
        gtf_file = "h38.filt.gtf"
        gene_names = {"STMN2", "UNC13A"}  # Or use gene IDs like {"ENSG00000104435"}

        genes_df = extract_genes_from_gtf(gtf_file, gene_names)
        print(genes_df.head())
    """
    if not use_polars:
        return extract_genes_from_gtf_pandas(gtf_file_path, gene_names)

    # Define columns and types for GTF file
    columns = ['seqname', 'source', 'feature', 'start', 'end', 'score', 'strand', 'frame', 'attribute']

    # Load GTF file using Polars
    gtf_df = pl.read_csv(gtf_file_path,
                         separator='\t', 
                         comment_prefix='#', 
                         has_header=False, new_columns=columns, schema_overrides={'seqname': pl.Utf8})

    # Try to filter for gene features (standard Ensembl GTF)
    gene_df = gtf_df.filter(pl.col('feature') == 'gene')
    
    # If no gene features found, derive from transcripts (MANE format)
    if gene_df.height == 0:
        # MANE GTF only has transcript/exon features, no explicit gene features
        # Derive gene-level info by aggregating transcripts
        transcript_df = gtf_df.filter(pl.col('feature') == 'transcript')
        
        # Extract gene_id and gene_name from attributes
        transcript_df = transcript_df.with_columns([
            pl.col('attribute').str.extract(r'gene_id "([^"]+)"').alias('gene_id'),
            pl.col('attribute').str.extract(r'gene "([^"]+)"').alias('gene_name')  # MANE uses 'gene' not 'gene_name'
        ])
        
        # Aggregate by gene: min(start), max(end), first strand/seqname
        gene_df = transcript_df.group_by('gene_id').agg([
            pl.col('gene_name').first().alias('gene_name'),
            pl.col('seqname').first().alias('seqname'),
            pl.col('start').min().alias('start'),
            pl.col('end').max().alias('end'),
            pl.col('strand').first().alias('strand')
        ])
        
        # MANE GTF uses "chr1" format but Ensembl FASTA uses "1" format
        # Strip "chr" prefix to match FASTA chromosome names
        gene_df = gene_df.with_columns([
            pl.col('seqname').str.replace('^chr', '').alias('seqname')
        ])
        
        # Filter to standard chromosomes only (exclude patches/fixes like X_MU273393v1_fix)
        standard_chroms = [str(i) for i in range(1, 23)] + ['X', 'Y', 'MT']
        gene_df = gene_df.filter(pl.col('seqname').is_in(standard_chroms))
    else:
        # Standard Ensembl format: extract gene_name and gene_id from attributes
        gene_df = gene_df.with_columns([
            pl.col('attribute').str.extract(r'gene_name "([^"]+)"').alias('gene_name'),
            pl.col('attribute').str.extract(r'gene_id "([^"]+)"').alias('gene_id')
        ])

    # If gene_names is provided, filter by either gene_name or gene_id
    if gene_names is not None:
        # Check if the provided list/set contains gene IDs (starting with 'ENSG' or 'gene-')
        if all(name.startswith('ENSG') or name.startswith('gene-') for name in gene_names):
            gene_df = gene_df.filter(pl.col('gene_id').is_in(gene_names))
        else:
            gene_df = gene_df.filter(pl.col('gene_name').is_in(gene_names))

    return gene_df.select(['gene_name', 'gene_id', 'seqname', 'start', 'end', 'strand'])


def extract_exons_from_gtf(gtf_file_path, use_polars=True, output_path=None):
    """
    Extract exon features from a GTF file.

    Parameters:
    - gtf_file_path (str): Path to the input GTF file.
    - use_polars (bool): Whether to use Polars for data manipulation (default: True).
    - output_path (str, optional): Path to save the output DataFrame (default: None).

    Returns:
    - pl.DataFrame: A Polars DataFrame containing gene ID, chromosome, start, end, strand, and gene name for each exon.
    """
    if not use_polars:
        raise NotImplementedError("Non-Polars version not implemented in this example.")

    # Define columns and types for GTF file
    columns = ['seqname', 'source', 'feature', 'start', 'end', 'score', 'strand', 'frame', 'attribute']

    # Load GTF file using Polars
    gtf_df = pl.read_csv(
        gtf_file_path,
        separator='\t',
        comment_prefix='#',
        has_header=False,
        new_columns=columns,
        schema_overrides={'seqname': pl.Utf8}
    )

    # Filter for exon features
    exon_df = gtf_df.filter(pl.col('feature') == 'exon')

    # Extract gene_id and gene_name from attributes
    exon_df = exon_df.with_columns([
        pl.col("attribute").str.extract(r'gene_id "([^"]+)"').alias("gene_id"),
        pl.col("attribute").str.extract(r'gene_name "([^"]+)"').alias("gene_name"),
        pl.col("attribute").str.extract(r'transcript_id "([^"]+)"').alias("transcript_id")
    ])

    # Handle missing gene_name by substituting with gene_id or None
    exon_df = exon_df.with_columns(
        pl.when(pl.col("gene_name").is_null())
        .then(pl.col("gene_id"))
        .otherwise(pl.col("gene_name"))
        .alias("gene_name")
    )

    # Select only relevant columns
    exon_df = exon_df.select(["gene_id", "transcript_id", "gene_name",  "seqname", "start", "end", "strand"])

    # Save to output_path if provided
    if output_path:
        # Determine separator based on file extension
        separator = '\t' if output_path.endswith('.tsv') else ','
        exon_df.write_csv(output_path, separator=separator)
    
    return exon_df


def compute_exon_counts(exon_df):
    """
    Compute exon counts per gene.

    Parameters:
    - exon_df (pl.DataFrame): Polars DataFrame with exon information.

    Returns:
    - pl.DataFrame: A Polars DataFrame with gene ID and exon counts.
    """
    exon_counts = (
        exon_df
        .group_by("gene_id")
        .agg(pl.count("start").alias("exon_count"))
    )
    return exon_counts


def filter_valid_splice_site_genes(exon_counts, min_exons=2):
    """
    Filter genes with valid splice sites based on exon count.

    Parameters:
    - exon_counts (pl.DataFrame): Polars DataFrame with gene ID and exon counts.
    - min_exons (int): Minimum number of exons required for a valid splice site.

    Returns:
    - pl.DataFrame: A Polars DataFrame with gene IDs of valid splice site genes.
    """
    valid_genes = exon_counts.filter(pl.col("exon_count") >= min_exons)
    return valid_genes


####################################################################################################

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
        columns = ['gene_id', 'gene_name'] + \
                  [col for col in sequences_df.columns if col not in ['gene_id', 'gene_name', 'sequence']] + \
                  ['sequence']
        sequences_df = sequences_df.select(columns)

        return sequences_df
    
    return {seq['gene_id']: seq['sequence'] for seq in sequences}


def extract_transcript_sequences_v0(transcripts_df, genome_fasta, output_format='dict', include_columns=None):
    """
    Extract transcript (pre-mRNA) sequences from the reference genome based on transcript coordinates.

    Parameters:
    - transcripts_df (pd.DataFrame): 
        DataFrame with transcript_id, chromosome (seqname), start, end, and strand.
    - genome_fasta (str): Path to the reference genome FASTA file.
    - output_format (str): Output format, either 'dict' or 'dataframe'. Default is 'dict'.
    - include_columns (list, optional): Additional columns from transcripts_df to include in the output.

    Returns:
    - A dictionary mapping transcript IDs to their DNA sequences or 
      a DataFrame if output_format is 'dataframe'.

        - dict: If output_format is 'dict', returns a dictionary where:
            - Keys are transcript IDs (str).
            - Values are the corresponding DNA sequences (str) of the transcripts extracted from the reference genome.
        Example:
            {
                "ENST00000456328": "ATGCGT...",
                "ENST00000450305": "CGTACG..."
            }
        - pd.DataFrame: If output_format is 'dataframe', returns a DataFrame with columns:
            - 'transcript_id': Transcript ID.
            - 'gene_name': Gene name.
            - 'seqname': Chromosome name.
            - 'start': Start coordinate of the transcript.
            - 'end': End coordinate of the transcript.
            - 'strand': Strand information ('+' or '-').
            - 'sequence': DNA sequence of the transcript.
            - Additional columns specified in include_columns.


    Example usage:
        genome_fasta = "/path/to/Homo_sapiens.GRCh38.dna.primary_assembly.fa"
        transcripts_df = extract_transcripts_from_gtf("h38.filt.gtf")
        transcript_sequences = extract_transcript_sequences(transcripts_df, genome_fasta)
        print(transcript_sequences)
    """
    from Bio import SeqIO

    sequences = []
    
    # Load the genome into a dictionary
    genome = SeqIO.to_dict(SeqIO.parse(genome_fasta, "fasta"))
    
    for _, row in transcripts_df.iterrows():
        transcript_id = row['transcript_id']
        gene_name = row['gene_name']
        chrom = row['seqname']
        start = row['start']  # Convert to 0-based index for Biopython
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
        sequences_df = pl.DataFrame(sequences)

        # Reorder columns to place 'sequence' as the last column
        columns = ['transcript_id'] + [col for col in sequences_df.columns if col not in ['transcript_id', 'sequence']] + ['sequence']
        sequences_df = sequences_df[columns]

        return sequences_df
    
    return {seq['transcript_id']: seq['sequence'] for seq in sequences}


def extract_transcript_sequences(transcripts_df, genome_fasta, output_format='dict', include_columns=None):
    """
    Extract transcript (pre-mRNA) sequences from the reference genome based on transcript coordinates.

    Parameters:
    - transcripts_df (pl.DataFrame): 
        Polars DataFrame with transcript_id, chromosome (seqname), start, end, and strand.
    - genome_fasta (str): Path to the reference genome FASTA file.
    - output_format (str): Output format, either 'dict' or 'dataframe'. Default is 'dict'.
    - include_columns (list, optional): Additional columns from transcripts_df to include in the output.

    Returns:
    - A dictionary mapping transcript IDs to their DNA sequences or 
      a Polars DataFrame if output_format is 'dataframe'.

        - dict: If output_format is 'dict', returns a dictionary where:
            - Keys are transcript IDs (str).
            - Values are the corresponding DNA sequences (str) of the transcripts extracted from the reference genome.
        Example:
            {
                "ENST00000456328": "ATGCGT...",
                "ENST00000450305": "CGTACG..."
            }
        - pl.DataFrame: If output_format is 'dataframe', returns a Polars DataFrame with columns:
            - 'transcript_id': Transcript ID.
            - 'gene_name': Gene name.
            - 'seqname': Chromosome name.
            - 'start': Start coordinate of the transcript.
            - 'end': End coordinate of the transcript.
            - 'strand': Strand information ('+' or '-').
            - 'sequence': DNA sequence of the transcript.
            - Additional columns specified in include_columns.


    Example usage:
        genome_fasta = "/path/to/Homo_sapiens.GRCh38.dna.primary_assembly.fa"
        transcripts_df = extract_transcripts_from_gtf("h38.filt.gtf")
        transcript_sequences = extract_transcript_sequences(transcripts_df, genome_fasta)
        print(transcript_sequences)
    """
    # from Bio import SeqIO
    sequences = []

    # Load the genome into a dictionary
    genome = SeqIO.to_dict(SeqIO.parse(genome_fasta, "fasta"))
    
    nns = nps = 0
    for row in transcripts_df.iter_rows(named=True):
        # In Polars, iter_rows(named=True) is used to iterate over rows similar to iterrows() in pandas.

        transcript_id = row['transcript_id']
        gene_id = row.get('gene_id', "uknown")
        gene_name = row.get('gene_name', "unknown")
        chrom = row['seqname']
        start = row['start'] - 1    # Convert to 0-based index for Biopython
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
        
        # Collect the sequence and additional information
        sequence_info = {
            'seqname': chrom,
            'gene_name': gene_name,
            'gene_id': gene_id,
            'transcript_id': transcript_id,
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
    print(f"... Number of transcripts on the negative strand: {nns}")
    print(f"... Number of transcripts on the positive strand: {nps}")
    assert nns > 0 and nps > 0, "Both positive and negative strand transcripts should be present."

    if output_format == 'dataframe':
        sequences_df = pl.DataFrame(sequences)

        # Reorder columns to place 'sequence' as the last column
        # Use Polars' `select` to reorder
        columns = ['transcript_id'] + [col for col in sequences_df.columns if col not in ['transcript_id', 'sequence']] + ['sequence']
        
        sequences_df = sequences_df.select(columns)
        # Instead of reordering columns like in pandas, Polars uses the select() method to reorder the columns.

        return sequences_df
    
    # Return dictionary if 'dict' output is specified
    return {seq['transcript_id']: seq['sequence'] for seq in sequences}


def extract_gene_sequences_minmax_with_pandas(transcripts_df, genome_fasta, output_format='dict', include_columns=None):
    """
    Extract gene-specific DNA sequences from the reference genome based on the minimum start and maximum end coordinates 
    across all annotated transcripts for each gene.

    Parameters:
    - transcripts_df (pd.DataFrame): DataFrame with transcript_id, gene_id, seqname (chromosome), start, end, and strand.
    - genome_fasta (str): Path to the reference genome FASTA file.
    - output_format (str): Output format, either 'dict' or 'dataframe'. Default is 'dict'.
    - include_columns (list, optional): Additional columns from transcripts_df to include in the output.

    Returns:
    - A dictionary mapping gene IDs to their DNA sequences or 
      a DataFrame if output_format is 'dataframe'.

        - If output_format is 'dict':
            A dictionary mapping gene IDs to their DNA sequences.
            Example:
            {
                'gene1': 'ATCG...',
                'gene2': 'GGCTA...',
                ...
            }
        - If output_format is 'dataframe':
            A DataFrame with the following minimum columns:
            - gene_id: The unique identifier for each gene.
            - sequence: The extracted DNA sequence for the gene.
            - strand: The strand on which the gene is located ('+' or '-').
            - start: The 1-based start coordinate of the gene sequence.
            - end: The end coordinate of the gene sequence.
            
            Additional columns specified in include_columns will also be included if provided.

    Example usage:
        genome_fasta = "/path/to/Homo_sapiens.GRCh38.dna.primary_assembly.fa"
        gene_sequences = extract_gene_sequences_minmax(transcripts_df, genome_fasta)
        print(gene_sequences)
    """
    from Bio import SeqIO
    import pandas as pd

    sequences = []
    
    # Load the genome into a dictionary
    genome = SeqIO.to_dict(SeqIO.parse(genome_fasta, "fasta"))

    # Group transcripts by gene_id and calculate the minimum start and maximum end for each gene
    gene_coords = transcripts_df.groupby('gene_id').agg(
        seqname=('seqname', 'first'),  # assuming all transcripts of a gene are on the same chromosome
        gene_name=('gene_name', 'first'),  # assuming all transcripts of a gene have the same gene name
        strand=('strand', 'first'),  # assuming all transcripts of a gene are on the same strand
        min_start=('start', 'min'),  # For each group, calculate the minimum value in the start column
        max_end=('end', 'max')  # For each group, calculate the maximum value in the end column
    ).reset_index()  # This resets the index of the resulting DataFrame, converting the gene_id from an index to a column
    # NOTE: See demo_gropuby_and_aggregation() for an example of how groupby and aggregation work

    # Extract the gene-specific sequences using the computed coordinates
    gene_sizes = {}
    for _, row in gene_coords.iterrows():
        gene_id = row['gene_id']
        gene_name = row['gene_name']
        chrom = row['seqname']
        start = row['min_start'] - 1  # Convert to 0-based index
        end = row['max_end']
        strand = normalize_strand(row['strand'])
        
        # Extract the sequence
        sequence = genome[chrom].seq[start:end]
        
        # Reverse complement if on the negative strand
        if strand == '-':
            sequence = sequence.reverse_complement()

        # Test: Keep track of the gene sizes
        gene_sizes[gene_id] = len(sequence)
        
        # Collect the sequence and additional information
        sequence_info = {
            'seqname': chrom,
            'gene_id': gene_id,
            'gene_name': gene_name,
            'start': start + 1,  # Back to 1-based for consistent reporting
            'end': end,
            'strand': strand, 
            'sequence': str(sequence),
        }

        # Include additional columns if specified
        if include_columns:
            for col in include_columns:
                sequence_info[col] = row[col]

        sequences.append(sequence_info)

    # Gene statistics 
    print("[info] Shortest gene sequence:", min(gene_sizes.values()))
    print("[info] Longest gene sequence:", max(gene_sizes.values()))
    print("... median gene sequence length:", pd.Series(gene_sizes).median())

    if output_format == 'dataframe':
        sequences_df = pl.DataFrame(sequences)

        # Reorder columns to place 'sequence' as the last column
        columns = ['gene_id', ] + [col for col in sequences_df.columns if col not in ['gene_id', 'sequence']] + ['sequence']
        sequences_df = sequences_df[columns]

        return sequences_df
    
    return {seq['gene_id']: seq['sequence'] for seq in sequences}


def extract_gene_sequences_minmax(transcripts_df, genome_fasta, output_format='dict', include_columns=None, use_polars=True):
    """
    Extract gene-specific DNA sequences from the reference genome based on the minimum start and maximum end coordinates 
    across all annotated transcripts for each gene.

    Parameters:
    - transcripts_df (pl.DataFrame): Polars DataFrame with transcript_id, gene_id, seqname (chromosome), start, end, and strand.
    - genome_fasta (str): Path to the reference genome FASTA file.
    - output_format (str): Output format, either 'dict' or 'dataframe'. Default is 'dict'.
    - include_columns (list, optional): Additional columns from transcripts_df to include in the output.

    Returns:
    - A dictionary mapping gene IDs to their DNA sequences or 
      a DataFrame if output_format is 'dataframe'.

        - If output_format is 'dict':
            A dictionary mapping gene IDs to their DNA sequences.
            Example:
            {
                'gene1': 'ATCG...',
                'gene2': 'GGCTA...',
                ...
            }
        - If output_format is 'dataframe':
            A DataFrame with the following minimum columns:
            - gene_id: The unique identifier for each gene.
            - sequence: The extracted DNA sequence for the gene.
            - strand: The strand on which the gene is located ('+' or '-').
            - start: The 1-based start coordinate of the gene sequence.
            - end: The end coordinate of the gene sequence.
            
            Additional columns specified in include_columns will also be included if provided.

    Example usage:
        genome_fasta = "/path/to/Homo_sapiens.GRCh38.dna.primary_assembly.fa"
        gene_sequences = extract_gene_sequences_minmax(transcripts_df, genome_fasta)
        print(gene_sequences)
    """
    from Bio import SeqIO

    # Load the genome into a dictionary
    genome = SeqIO.to_dict(SeqIO.parse(genome_fasta, "fasta"))

    # Group transcripts by gene_id and calculate the minimum start and maximum end for each gene
    gene_coords = transcripts_df.group_by("gene_id").agg([
        pl.col("seqname").first().alias("seqname"),
        pl.col("gene_name").first().alias("gene_name"),
        pl.col("strand").first().alias("strand"),
        pl.col("start").min().alias("min_start"),  # minimum start across all transcripts
        pl.col("end").max().alias("max_end")  # maximum end across all transcripts
    ])
    # NOTE: The groupby and agg functions have been adapted to Polars syntax. 
    #       Aggregation is performed using pl.first(), pl.min(), and pl.max() to get the desired fields.

    # Convert to list of dictionaries for further processing
    gene_coords = gene_coords.to_dicts()
    # Polars provides a to_dicts() method to convert the grouped DataFrame into a list of dictionaries for further processing.

    sequences = []
    gene_sizes = {}

    # Extract the gene-specific sequences using the computed coordinates
    for row in tqdm(gene_coords, desc="Processing genes"):
        gene_id = row['gene_id']
        gene_name = row['gene_name']
        chrom = row['seqname']
        start = row['min_start'] - 1  # Convert to 0-based index
        end = row['max_end']
        strand = row['strand']
        
        # Extract the sequence
        sequence = genome[chrom].seq[start:end]
        
        # Reverse complement if on the negative strand
        if strand == '-':
            sequence = sequence.reverse_complement()

        # Test: Keep track of the gene sizes
        gene_sizes[gene_id] = len(sequence)
        
        # Collect the sequence and additional information
        sequence_info = {
            'seqname': chrom,
            'gene_id': gene_id,
            'gene_name': gene_name,
            'start': start + 1,  # Back to 1-based for consistent reporting
            'end': end,
            'strand': strand, 
            'sequence': str(sequence),
        }
        assert len(sequence) == end - start, f"Sequence length mismatch for gene {gene_id}"

        # Include additional columns if specified
        if include_columns:
            for col in include_columns:
                sequence_info[col] = row.get(col)

        sequences.append(sequence_info)

    # Gene statistics 
    print("[info] Shortest gene sequence:", min(gene_sizes.values()))
    print("[info] Longest gene sequence:", max(gene_sizes.values()))
    print(f"... median gene sequence length: {sorted(gene_sizes.values())[len(gene_sizes) // 2]}")

    if output_format == 'dataframe':
        sequences_df = pl.DataFrame(sequences)

        # Reorder columns to place 'sequence' as the last column
        columns = ['gene_id', ] + [col for col in sequences_df.columns if col not in ['gene_id', 'sequence']] + ['sequence']
        sequences_df = sequences_df.select(columns)
        # The select() method is used to reorder columns in Polars.

        return sequences_df
    
    return {seq['gene_id']: seq['sequence'] for seq in sequences}


def demo_gropuby_and_aggregation(): 
    # Example DataFrame
    data = {
        'gene_id': ['gene1', 'gene1', 'gene2', 'gene2', 'gene3'],
        'seqname': ['chr1', 'chr1', 'chr2', 'chr2', 'chr3'],
        'strand': ['+', '+', '-', '-', '+'],
        'start': [100, 150, 200, 250, 300],
        'end': [200, 180, 300, 280, 400]
    }
    transcripts_df = pd.DataFrame(data)

    # Group transcripts by gene_id and calculate the minimum start and maximum end for each gene
    gene_coords = transcripts_df.groupby('gene_id').agg(
        seqname=('seqname', 'first'),  # assuming all transcripts of a gene are on the same chromosome
        strand=('strand', 'first'),  # assuming all transcripts of a gene are on the same strand
        min_start=('start', 'min'),
        max_end=('end', 'max')
    ).reset_index()

    print(gene_coords)

####################################################################################################


def analyze_longest_transcripts(transcript_sequences_df: pl.DataFrame) -> pl.DataFrame:
    """
    For each gene_id, find the transcript(s) with the maximum sequence length.
    Returns a DataFrame with columns [gene_id, transcript_id, length].
    If multiple transcripts tie for the max length, you get multiple rows for that gene.
    """

    # 1) Add a 'length' column.
    df = transcript_sequences_df.with_columns(
        pl.col("sequence").apply(len).alias("length")
    )

    # 2) Group by gene_id and compute the maximum length.
    #    We also keep all transcripts that match that max length.
    #    The "filter" approach below returns a list of transcripts matching the max for each group.
    grouped = (
        df
        .groupby("gene_id", maintain_order=True)
        .agg([
            pl.col("length").max().alias("max_length"),
            pl.col("transcript_id").filter(pl.col("length") == pl.col("length").max()).alias("max_tx_ids")
        ])
    )
    # grouped has columns: gene_id, max_length, max_tx_ids

    # 3) We often want one row per transcript_id. So let’s “explode” the max_tx_ids list:
    #    If there's more than one transcript with the same max length, we get multiple rows.
    #    We'll rename 'max_tx_ids' to 'transcript_id'.
    result = (
        grouped
        .explode("max_tx_ids")
        .rename({"max_tx_ids": "transcript_id"})
    )

    # 4) Final shape: [gene_id, max_length, transcript_id]
    #    Let’s reorder columns slightly:
    result = result.select(["gene_id", "transcript_id", "max_length"])

    return result


def analyze_genes(gene_sequences_df: pl.DataFrame) -> pl.DataFrame:
    """
    Return a DataFrame with columns [gene_id, length], 
    where length is the length of the gene's extracted DNA sequence.
    """
    df = gene_sequences_df.with_columns(
        pl.col("sequence").apply(len).alias("length")
    )

    # If each gene_id is unique in gene_sequences_df, no further groupby is needed:
    result = df.select(["gene_id", "length"])
    return result


def check_available_memory():
    """
    Check the available memory and print it out.
    
    Returns:
    - available_memory (float): Available memory in GB.
    """
    import psutil  # pip install psutil
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


def add_sequence_length_column(df: pl.DataFrame, seq_col="sequence", new_col="length") -> pl.DataFrame:
    # Step 1: Extract the 'sequence' column as a list of strings
    sequences = df[seq_col].to_list()  # or .to_pylist() in newer versions

    # Step 2: Compute lengths
    lengths = [len(seq) for seq in sequences]

    # Step 3: Add as a new column
    return df.with_columns(pl.Series(new_col, lengths))


def boundary_status(gene_start, gene_end, tx_start, tx_end):
    """
    Returns:
      - 'within' if the transcript region is fully within the gene region,
      - 'no_overlap' if there is no overlap at all,
      - 'partial' otherwise (some overlap, but not fully contained).
    """
    if tx_start >= gene_start and tx_end <= gene_end:
        return "within"
    elif tx_end < gene_start or tx_start > gene_end:
        return "no_overlap"
    else:
        return "partial"


def add_boundary_status_column(
    df: pl.DataFrame,
    gene_start_col: str = "gene_start",
    gene_end_col: str   = "gene_end",
    tx_start_col: str   = "tx_start",
    tx_end_col: str     = "tx_end",
    new_col: str        = "boundary_status"
) -> pl.DataFrame:
    gene_start_list = df[gene_start_col].to_list()
    gene_end_list   = df[gene_end_col].to_list()
    tx_start_list   = df[tx_start_col].to_list()
    tx_end_list     = df[tx_end_col].to_list()

    statuses = []
    for gs, ge, ts, te in zip(gene_start_list, gene_end_list, tx_start_list, tx_end_list):
        statuses.append(boundary_status(gs, ge, ts, te))

    return df.with_columns(pl.Series(new_col, statuses))


def compare_gene_and_transcript_boundaries(
    gene_sequences_df: pl.DataFrame,
    transcript_sequences_df: pl.DataFrame
) -> pl.DataFrame:
    # Step 1) Identify the longest transcript(s) per gene
    df_tx = add_sequence_length_column(transcript_sequences_df, "sequence", "length")

    grouped = (
        df_tx
        .group_by("gene_id", maintain_order=True)
        .agg([
            pl.col("length").max().alias("max_length"),
            pl.col("transcript_id")
              .filter(pl.col("length") == pl.col("length").max())
              .alias("max_tx_ids")
        ])
    )
    longest_tx_df = (
        grouped
        .explode("max_tx_ids")
        .rename({"max_tx_ids": "transcript_id", "max_length": "tx_length"})
    )

    # Step 2) Join transcript boundary info
    tx_cols = ["transcript_id", "gene_id", "start", "end", "length"]
    longest_tx_info = longest_tx_df.join(
        df_tx.select(tx_cols),
        on=["gene_id", "transcript_id"],
        how="inner"
    ).rename({"start": "tx_start", "end": "tx_end"})

    # Step 3) Join gene DataFrame for gene boundaries
    gene_cols = ["gene_id", "start", "end"]
    genes_renamed = gene_sequences_df.select(gene_cols).rename({
        "start": "gene_start",
        "end":   "gene_end"
    })

    joined = longest_tx_info.join(genes_renamed, on="gene_id", how="inner")

    # Step 4) Compute lengths from boundaries
    joined = joined.with_columns([
        (pl.col("gene_end") - pl.col("gene_start")).alias("gene_length"),
        (pl.col("tx_end") - pl.col("tx_start")).alias("tx_length_from_coords"),
    ])

    # Step 5) delta_length via Polars expression
    joined = joined.with_columns([
        (pl.col("gene_length") - pl.col("tx_length_from_coords")).alias("delta_length")
    ])

    # Step 6) boundary_status row by row
    joined = add_boundary_status_column(
        joined,
        gene_start_col="gene_start", 
        gene_end_col="gene_end", 
        tx_start_col="tx_start", 
        tx_end_col="tx_end",
        new_col="boundary_status"
    )

    # Step 7) Final reorder
    return joined.select([
        "gene_id",
        "transcript_id",
        "gene_start",
        "gene_end",
        "tx_start",
        "tx_end",
        "gene_length",
        "tx_length_from_coords",
        "delta_length",
        "boundary_status",
    ])


def compare_gene_and_transcript_lengths(
    gene_sequences_df: pl.DataFrame,
    transcript_sequences_df: pl.DataFrame
) -> pl.DataFrame:
    """
    Returns a DataFrame with columns:
      [
        "gene_id",
        "transcript_id",
        "gene_length",
        "tx_length",
        "delta_length",
        "ratio"
      ]
    Where ratio = tx_length / gene_length.

    The function identifies the longest transcript(s) for each gene based on
    extracted sequences, then compares lengths between gene and transcript.
    """

    # 1) Add a "gene_length" column to gene_sequences_df.
    gene_len_df = add_sequence_length_column(
        gene_sequences_df, seq_col="sequence", new_col="gene_length"
    ).select(["gene_id", "gene_length"])

    # 2) Add a "tx_length" column to transcripts; 
    #    find the transcript(s) with the max length per gene.
    df_tx = add_sequence_length_column(
        transcript_sequences_df, seq_col="sequence", new_col="tx_length"
    )

    grouped = (
        df_tx
        .group_by("gene_id", maintain_order=True)
        .agg([
            pl.col("tx_length").max().alias("max_length"),
            pl.col("transcript_id")
              .filter(pl.col("tx_length") == pl.col("tx_length").max())
              .alias("max_tx_ids")
        ])
    )

    # "explode" to handle ties for the max length
    longest_tx_df = (
        grouped
        .explode("max_tx_ids")
        .rename({"max_tx_ids": "transcript_id", "max_length": "tx_length"})
        .select(["gene_id", "transcript_id", "tx_length"])
    )

    # 3) Join on gene_id to get both gene_length and tx_length.
    joined = longest_tx_df.join(gene_len_df, on="gene_id", how="inner")

    # 4) Compute difference (delta_length) and ratio.
    #    ratio = tx_length / gene_length (handle divide by zero).
    joined = joined.with_columns([
        (pl.col("gene_length") - pl.col("tx_length")).alias("delta_length"),
        pl.when(pl.col("gene_length") > 0)
          .then(pl.col("tx_length") / pl.col("gene_length"))
          .otherwise(None)
          .alias("ratio")
    ])

    # 5) Final reorder of columns
    return joined.select([
        "gene_id",
        "transcript_id",
        "gene_length",
        "tx_length",
        "delta_length",
        "ratio",
    ])



####################################################################################################


def normalize_chrom(chrom):
    """Normalize chromosome names to ensure consistency."""
    return chrom.lower().replace('chr', '')


def normalize_strand(strand):
    """Normalize strand representation to '+' and '-'."""
    if strand in ['+', 'plus', '1', 1, True]:
        return '+'
    elif strand in ['-', 'minus', '-1', -1, False]:
        return '-'
    else:
        raise ValueError(f"Unexpected strand value: {strand}")


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


def save_sequences_by_chromosome_pandas(df, output_file, format='tsv', chromosomes_to_save=None):
    """
    Save the DNA sequence DataFrame to multiple files, each corresponding to a particular chromosome value.

    Parameters:
    - df (pd.DataFrame): DataFrame containing transcript sequences.
    - output_file (str): Base name for the output files.
    - format (str): Format to save the files ('tsv', 'csv', or 'parquet').
    - chromosomes_to_save (list, optional): List of chromosomes to save. If None, all chromosomes will be saved.

    Example usage: 
        # Assuming `transcript_sequence_df` is your DataFrame and `output_file` is the base name for the output files
        output_file = "transcript_sequences.tsv"
        save_sequences_by_chromosome(transcript_sequence_df, output_file, format='tsv')
    """
    # from tqdm import tqdm

    # Ensure 'seqname' column is of string type
    df['seqname'] = df['seqname'].astype(str)

    # Get unique chromosome values
    unique_chromosomes = df['seqname'].unique()
    print(f"Found {len(unique_chromosomes)} unique chromosomes:")
    print(list(unique_chromosomes))

    # Filter chromosomes if chromosomes_to_save is provided
    if chromosomes_to_save is not None:
        unique_chromosomes = [chrom for chrom in unique_chromosomes if chrom in chromosomes_to_save]
        print(f"Saving only the following chromosomes: {unique_chromosomes}")

    # NOTE: ['1', '2', '3', '4', '5', '6', '7', '8', '9', '11', '10', '12', '13', '14', '15', '16', '17', '18', 
    #        '20', '19', '22', '21',
    #        'X', 'Y',  
    #        'MT', 'KI270728.1', 'KI270727.1', 'KI270442.1', 
    #        'GL000225.1', 'GL000009.2', 'GL000194.1', 'GL000205.2', 'GL000195.1', 
    #        'KI270733.1', 'GL000219.1', 'GL000216.2', 'KI270744.1', 'KI270734.1', 'GL000213.1', 
    #        'GL000220.1', 'GL000218.1', 'KI270731.1', 'KI270750.1', 'KI270721.1', 'KI270726.1', 'KI270711.1', 'KI270713.1']

    for chrom in tqdm(unique_chromosomes, desc="Saving sequences by chromosome"):
        # Filter the DataFrame for the current chromosome
        chrom_df = df[df['seqname'] == chrom]
        assert chrom_df.shape[0] > 0, f"No records found for chromosome {chrom}."

        n_genes = chrom_df['gene_id'].nunique()
        print("[info] Saving {} genes for chromosome {}.".format(n_genes, chrom))

        # Create the output file name
        chrom_output_file = f"{os.path.splitext(output_file)[0]}_{chrom}{os.path.splitext(output_file)[1]}"

        # Save the filtered DataFrame to the file
        if format == 'tsv':
            chrom_df.to_csv(chrom_output_file, sep='\t', index=False, encoding='utf-8')
        elif format == 'csv':
            chrom_df.to_csv(chrom_output_file, sep=',', index=False, encoding='utf-8')
        elif format == 'parquet':
            chrom_df.to_parquet(chrom_output_file, index=False)
        else:
            raise ValueError("Unsupported format. Please choose 'tsv', 'csv', or 'parquet'.")

        print(f"[i/o] Saved {chrom_df.shape[0]} records to {chrom_output_file}")


def save_sequences_by_chromosome(df, output_file, format='tsv', chromosomes_to_save=None):
    """
    Save the DNA sequence Polars DataFrame to multiple files, each corresponding to a particular chromosome value.

    Parameters:
    - df (pl.DataFrame): Polars DataFrame containing transcript sequences.
    - output_file (str): Base name for the output files.
    - format (str): Format to save the files ('tsv', 'csv', or 'parquet').
    - chromosomes_to_save (list, optional): List of chromosomes to save. If None, all chromosomes will be saved.

    Example usage: 
        output_file = "transcript_sequences.tsv"
        save_sequences_by_chromosome(transcript_sequence_df, output_file, format='tsv')
    """
    
    # Ensure 'seqname' column is of string type
    df = df.with_columns(pl.col('seqname').cast(pl.Utf8))

    # Get unique chromosome values
    unique_chromosomes = df.select('seqname').unique().to_series().to_list()
    print(f"Found {len(unique_chromosomes)} unique chromosomes:")
    print(list(unique_chromosomes))

    # Filter chromosomes if chromosomes_to_save is provided
    if chromosomes_to_save is not None:
        unique_chromosomes = [chrom for chrom in unique_chromosomes if chrom in chromosomes_to_save]
        print(f"Saving only the following chromosomes: {unique_chromosomes}")

    # Save each chromosome's sequences separately
    for chrom in tqdm(unique_chromosomes, desc="Saving sequences by chromosome"):
        # Filter the Polars DataFrame for the current chromosome
        chrom_df = df.filter(pl.col('seqname') == chrom)
        assert chrom_df.shape[0] > 0, f"No records found for chromosome {chrom}."

        n_genes = chrom_df.select('gene_id').n_unique()
        print("[info] Saving {} genes for chromosome {}.".format(n_genes, chrom))

        # Create the output file name
        chrom_output_file = f"{os.path.splitext(output_file)[0]}_{chrom}{os.path.splitext(output_file)[1]}"

        # Save the filtered DataFrame to the file
        if format == 'tsv':
            chrom_df.write_csv(chrom_output_file, separator='\t')
        elif format == 'csv':
            chrom_df.write_csv(chrom_output_file, separator=',')
        elif format == 'parquet':
            chrom_df.write_parquet(chrom_output_file)
        else:
            raise ValueError("Unsupported format. Please choose 'tsv', 'csv', or 'parquet'.")

        print(f"[i/o] Saved {chrom_df.shape[0]} records to {chrom_output_file}")


def load_sequences_v0(file_path, format='tsv'):
    """
    Load the DNA sequence DataFrame from a file in the specified format.

    Parameters:
    - file_path (str): Path to the file.
    - format (str): Format of the file. Can be 'tsv', 'csv', or 'parquet'. Default is 'tsv'.

    Returns:
    - pd.DataFrame: DataFrame containing gene names and sequences.
    """
    if format == 'tsv':
        return pd.read_csv(file_path, sep='\t')
    elif format == 'csv':
        return pd.read_csv(file_path)
    elif format == 'parquet':
        return pd.read_parquet(file_path)
    else:
        raise ValueError("Unsupported format. Please choose 'tsv', 'csv', or 'parquet'.")


def load_sequences(file_path, format='tsv', **kargs):
    """
    Load the DNA sequence Polars DataFrame from a file in the specified format.

    Parameters:
    - file_path (str): Path to the file.
    - format (str): Format of the file. Can be 'tsv', 'csv', or 'parquet'. Default is 'tsv'.

    Returns:
    - pl.DataFrame: Polars DataFrame containing gene names and sequences.
    """
    if format == 'tsv':
        return pl.read_csv(file_path, separator='\t', **kargs)
        # {'chrom': pl.Utf8, 'seqname': pl.Utf8}
    elif format == 'csv':
        return pl.read_csv(file_path, **kargs)
    elif format == 'parquet':
        return pl.read_parquet(file_path)
    else:
        raise ValueError("Unsupported format. Please choose 'tsv', 'csv', or 'parquet'.")


def load_sequences_by_chromosome_v0(base_file_path, format='tsv'):
    """
    Load the DNA sequence DataFrame from multiple files, each corresponding to a particular chromosome value.

    Parameters:
    - base_file_path (str): Base path to the files (without the chromosome suffix).
    - format (str): Format of the files. Can be 'tsv', 'csv', or 'parquet'. Default is 'tsv'.

    Returns:
    - pd.DataFrame: Concatenated DataFrame containing gene names and sequences from all chromosome files.
    """
    # Get the base name and extension of the file
    base_name, ext = os.path.splitext(base_file_path)

    # Initialize an empty list to store DataFrames
    dataframes = []

    # Iterate over possible chromosome values (assuming chromosomes 1-22, X, Y, and MT)
    chromosomes = [str(i) for i in range(1, 23)] + ['X', 'Y', 'MT']

    # Add progress tracking with tqdm
    for chrom in tqdm(chromosomes, desc="Loading sequences by chromosome"):
        # Construct the file name for the current chromosome
        chrom_file_path = f"{base_name}_{chrom}{ext}"

        # Check if the file exists
        if os.path.exists(chrom_file_path):
            # Load the DataFrame from the file
            if format == 'tsv':
                df = pd.read_csv(chrom_file_path, sep='\t')
            elif format == 'csv':
                df = pd.read_csv(chrom_file_path)
            elif format == 'parquet':
                df = pd.read_parquet(chrom_file_path)
            else:
                raise ValueError("Unsupported format. Please choose 'tsv', 'csv', or 'parquet'.")

            # Append the DataFrame to the list
            dataframes.append(df)
        else:
            print(f"File {chrom_file_path} does not exist. Skipping...")

    # Concatenate all DataFrames into a single DataFrame
    if dataframes:
        concatenated_df = pd.concat(dataframes, ignore_index=True)
        return concatenated_df
    else:
        raise FileNotFoundError("No chromosome files found.")


def load_sequences_by_chromosome(base_file_path, format='tsv', **kargs):
    """
    Load the DNA sequence Polars DataFrame from multiple files, each corresponding to a particular chromosome value.

    Parameters:
    - base_file_path (str): Base path to the files (without the chromosome suffix).
    - format (str): Format of the files. Can be 'tsv', 'csv', or 'parquet'. Default is 'tsv'.

    Returns:
    - pl.DataFrame: Concatenated Polars DataFrame containing gene names and sequences from all chromosome files.
    """
    # Get the base name and extension of the file
    base_name, ext = os.path.splitext(base_file_path)

    # Initialize an empty list to store DataFrames
    dataframes = []

    # Iterate over possible chromosome values (assuming chromosomes 1-22, X, Y, and MT)
    chromosomes = [str(i) for i in range(1, 23)] + ['X', 'Y', 'MT']

    # Add progress tracking with tqdm
    for chrom in tqdm(chromosomes, desc="Loading sequences by chromosome"):
        # Construct the file name for the current chromosome
        chrom_file_path = f"{base_name}_{chrom}{ext}"

        # Check if the file exists
        if os.path.exists(chrom_file_path):
            # Load the DataFrame from the file using Polars
            if format == 'tsv':
                df = pl.read_csv(chrom_file_path, separator='\t', **kargs)
            elif format == 'csv':
                df = pl.read_csv(chrom_file_path, **kargs)
            elif format == 'parquet':
                df = pl.read_parquet(chrom_file_path)
            else:
                raise ValueError("Unsupported format. Please choose 'tsv', 'csv', or 'parquet'.")

            # Append the DataFrame to the list
            dataframes.append(df)
        else:
            print(f"File {chrom_file_path} does not exist. Skipping...")

    # Concatenate all DataFrames into a single Polars DataFrame
    if dataframes:
        concatenated_df = pl.concat(dataframes)
        return concatenated_df
    else:
        raise FileNotFoundError("No chromosome files found.")


def load_chromosome_sequence(base_file_path, chromosome, format='tsv', **kargs):
    """
    Load the DNA sequence DataFrame for a specific chromosome using Polars.

    Parameters:
    - base_file_path (str): Base path to the files (without the chromosome suffix).
    - chromosome (str): Chromosome of interest.
    - format (str): Format of the file. Can be 'tsv', 'csv', or 'parquet'. Default is 'tsv'.

    Returns:
    - pl.DataFrame: Polars DataFrame containing gene names and sequences for the specified chromosome.
    """
    # import polars as pl

    # Get the base name and extension of the file
    base_name, ext = os.path.splitext(base_file_path)

    # Construct the file name for the specified chromosome
    chrom_file_path = f"{base_name}_{chromosome}{ext}"

    # Check if the file exists
    if os.path.exists(chrom_file_path):
        # Load the DataFrame from the file using Polars
        if format == 'tsv':
            df = pl.read_csv(chrom_file_path, separator='\t', **kargs)
        elif format == 'csv':
            df = pl.read_csv(chrom_file_path, **kargs)
        elif format == 'parquet':
            df = pl.read_parquet(chrom_file_path)
        else:
            raise ValueError("Unsupported format. Please choose 'tsv', 'csv', or 'parquet'.")

        return df
    else:
        raise FileNotFoundError(f"File {chrom_file_path} does not exist.")


# Reading the file in streaming mode (e.g., for large CSVs)
def load_chromosome_sequence_streaming(file_path, chromosome, format='tsv', return_none_if_missing=False):
    """
    Load the DNA sequence DataFrame (or the pre-mRNA sequence DataFrame for transcript-specific analysis) 
    for a specific chromosome from a file using Polars streaming mode.
    
    This enables processing large files in chunks.

    Parameters:
    - file_path (str): Path to the base file.
    - chromosome (str): Chromosome to filter.
    - format (str): Format of the file ('tsv', 'csv', 'parquet').
    - return_none_if_missing (bool): If True, return None if the file does not exist. Default is False.

    Returns:
    - pl.LazyFrame: LazyFrame for the specific chromosome, enabling chunked processing.
    """
    base_name, ext = os.path.splitext(file_path)
    chrom_file_path = f"{base_name}_{chromosome}{ext}"

    if not os.path.exists(chrom_file_path):
        if return_none_if_missing:
            print(f"File {chrom_file_path} does not exist (yet).")
            return None
        else:
            raise FileNotFoundError(f"File {chrom_file_path} not found.")

    # Read the file in streaming mode (lazy loading)
    if format == 'tsv':
        return pl.scan_csv(chrom_file_path, separator='\t', low_memory=True)
    elif format == 'csv':
        return pl.scan_csv(chrom_file_path, low_memory=True)
    elif format == 'parquet':
        return pl.scan_parquet(chrom_file_path)
    else:
        raise ValueError("Unsupported format. Please choose 'tsv', 'csv', or 'parquet'.")


####################################################################################################



def infer_exons_with_gtf(junction_bed, gtf_file, transcript_id):
    """
    Infer exon coordinates from a junction BED file using GTF annotations.

    Parameters:
    - junction_bed (str): Path to the junction BED file with donor and acceptor sites.
    - gtf_file (str): Path to the GTF file containing transcript annotations.
    - transcript_id (str): The transcript ID for which to infer exon coordinates.

    Returns:
    - exons (list): List of inferred exon coordinates in the format [chrom, start, end, strand].
    """
    # Extract GTF annotations for the given transcript ID
    annotations = extract_gtf_annotations(gtf_file, transcript_id)
    exons_gtf = annotations['exons']
    transcript_cds = annotations['CDS']
    transcript_5utr = annotations['5UTR']
    transcript_3utr = annotations['3UTR']
    
    if len(exons_gtf) == 0:
        raise ValueError(f"No exons found for transcript {transcript_id} in the GTF file.")

    # Load the junction BED file
    junction_df = pd.read_csv(junction_bed, sep='\t', header=None, names=[
        'chrom', 'start', 'end', 'name', 'score', 'strand', 'donor_prob', 'acceptor_prob'
    ])

    # Initialize the list of inferred exons
    inferred_exons = []

    # Get transcript information from GTF annotations
    chrom = exons_gtf[0][0]  # Chromosome name from the first exon entry
    strand = exons_gtf[0][3]  # Strand information from the first exon entry

    # Loop through the junctions to infer exon coordinates
    if strand == '+':
        # Positive strand: exons are between donor end and acceptor start
        for i in range(len(junction_df) - 1):
            donor_end = junction_df.iloc[i]['end']
            acceptor_start = junction_df.iloc[i + 1]['start']
            exon_start = donor_end
            exon_end = acceptor_start
            inferred_exons.append([chrom, exon_start, exon_end, strand])

        # Add the first and last exons using the transcript start and end
        transcript_start = min(exons_gtf, key=lambda x: x[1])[1]  # Start position of the first exon
        first_exon_end = junction_df.iloc[0]['start']
        inferred_exons.insert(0, [chrom, transcript_start, first_exon_end, strand])

        transcript_end = max(exons_gtf, key=lambda x: x[2])[2]  # End position of the last exon
        last_exon_start = junction_df.iloc[-1]['end']
        inferred_exons.append([chrom, last_exon_start, transcript_end, strand])

    elif strand == '-':
        # Negative strand: exons are between acceptor end and donor start
        for i in range(len(junction_df) - 1):
            acceptor_end = junction_df.iloc[i]['end']
            donor_start = junction_df.iloc[i + 1]['start']
            exon_start = donor_start
            exon_end = acceptor_end
            inferred_exons.append([chrom, exon_start, exon_end, strand])

        # Add the first and last exons using the transcript start and end
        transcript_start = min(exons_gtf, key=lambda x: x[1])[1]  # Start position of the first exon
        first_exon_start = junction_df.iloc[0]['end']
        inferred_exons.insert(0, [chrom, first_exon_start, transcript_start, strand])

        transcript_end = max(exons_gtf, key=lambda x: x[2])[2]  # End position of the last exon
        last_exon_end = junction_df.iloc[-1]['start']
        inferred_exons.append([chrom, transcript_end, last_exon_end, strand])

    # Output inferred exons along with any UTR information
    exons_with_annotations = {
        'exons': inferred_exons,
        'CDS': transcript_cds,
        '5UTR': transcript_5utr,
        '3UTR': transcript_3utr
    }

    return exons_with_annotations


def merge_exons(exons):
    """
    Merge overlapping or adjacent exons and refine boundaries.
    """
    merged_exons = []
    current_exon = exons[0]

    for next_exon in exons[1:]:
        # Merge overlapping or adjacent exons
        if next_exon[1] <= current_exon[2]:  
            current_exon = (current_exon[0], current_exon[1], max(current_exon[2], next_exon[2]), current_exon[3], current_exon[4])
        else:
            merged_exons.append(current_exon)
            current_exon = next_exon

    merged_exons.append(current_exon)
    return merged_exons


def demo_infer_exons_with_gtf():

    data_prefix = "/path/to/meta-spliceai/data/ensembl"
    local_dir = "/path/to/meta-spliceai/data/ensembl/ALS"
    genome_annot = os.path.join(data_prefix, "Homo_sapiens.GRCh38.112.gtf") 

    genome_fasta = os.path.join(
        data_prefix, "Homo_sapiens.GRCh38.dna.primary_assembly.fa")

    assert os.path.exists(genome_annot)
    assert os.path.exists(genome_fasta)
    
    # Example usage
    junction_data = {
        'chrom': ['19', '19', '19', '19'],
        'start': [17606354, 17610099, 17611855, 17617849],
        'end': [17609940, 17611763, 17617702, 17618421],
        'name': ['ENST00000519716_JUNC_42', 'ENST00000519716_JUNC_41', 'ENST00000519716_JUNC_40', 'ENST00000519716_JUNC_39'],
        'score': [996.21, 995.28, 947.42, 997.84],
        'strand': ['-', '-', '-', '-'],
        'donor_prob': [0.9952, 0.9903, 0.9151, 0.9962],
        'acceptor_prob': [0.9950, 0.9980, 0.9776, 0.9972]
    }
    junction_bed_df = pd.DataFrame(junction_data)
    gtf_file_path = genome_annot
    transcript_id = 'ENST00000519716'

    # inferred_exons = infer_exons_with_gtf(junction_bed_df, gtf_file_path, transcript_id)
    # print("Inferred Exons with UTRs and CDS:", inferred_exons)

    method = 'bedtools' # 'biopython', 'bedtools'

    if method == 'biopython':
        annotations = extract_gtf_annotations_by_biopython(gtf_file_path, transcript_id)
    else:
        annotations = extract_gtf_annotations(gtf_file_path, transcript_id)
    
    print("Exons:", annotations['exons'])
    print("CDS:", annotations['CDS'])
    print("5' UTR:", annotations['5UTR'])
    print("3' UTR:", annotations['3UTR'])

    return


def test(): 

    demo_infer_exons_with_gtf()


    return


if __name__ == "__main__":
    test() 