import os
import math
from pathlib import Path
import pandas as pd
import polars as pl
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------------------------------
# 1) parse_gtf_for_genes (pandas version)
def parse_gtf_for_genes(gtf_file_path):
    """
    Parse a GTF file to extract gene coordinates.

    Parameters
    ----------
    gtf_file_path : str
        Path to the input GTF file.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing gene_name, seqname, start, end, and strand.
    """
    columns = ['seqname', 'source', 'feature', 'start', 'end', 
               'score', 'strand', 'frame', 'attribute']
    dtype = {
        'seqname': str, 'source': str, 'feature': str,
        'start': int,  'end': int,    'score': str, 
        'strand': str, 'frame': str,  'attribute': str
    }

    gtf_df = pd.read_csv(
        gtf_file_path, sep='\t', comment='#', header=None,
        names=columns, dtype=dtype
    )
    
    # Filter for gene features
    gene_df = gtf_df[gtf_df['feature'] == 'gene'].copy()

    # Extract gene_name from 'attribute' column
    gene_df['gene_name'] = gene_df['attribute'].str.extract(r'gene_name "([^"]+)"')
    return gene_df[['gene_name', 'seqname', 'start', 'end', 'strand']]

# -----------------------------------------------------
# 2) extract_genes_from_gtf (Polars version) 
#    => handles either gene names or gene IDs
def extract_genes_from_gtf(gtf_file_path, gene_names=None, use_polars=True, strip_version=False):
    """
    Extract gene features from a GTF file, with optional filtering by either 
    gene names (e.g. 'STMN2') or Ensembl IDs (e.g. 'ENSG00000104435').

    Parameters
    ----------
    gtf_file_path : str
        Path to the input GTF file.
    gene_names : set or list, optional
        A set/list of gene identifiers. If each element starts with "ENSG",
        we'll filter on gene_id. Otherwise, we'll filter on gene_name.
    use_polars : bool
        If True, use Polars; otherwise, fall back to a pandas-based approach.

    Returns
    -------
    pl.DataFrame
        Polars DataFrame with columns: [gene_name, gene_id, seqname, start, end, strand]
    """
    if not use_polars:
        return _extract_genes_from_gtf_pandas(gtf_file_path, gene_names)

    columns = ['seqname', 'source', 'feature', 'start', 'end', 
               'score', 'strand', 'frame', 'attribute']

    # Load GTF with Polars
    gtf_df = pl.read_csv(
        gtf_file_path,
        separator='\t',
        comment_prefix='#',
        has_header=False,
        new_columns=columns,
        # ensure seqname is string
        schema_overrides={'seqname': pl.Utf8}
    )

    # Keep only 'gene' features
    gene_df = gtf_df.filter(pl.col('feature') == 'gene')

    # Extract gene_name, gene_id
    gene_df = gene_df.with_columns([
        pl.col('attribute').str.extract(r'gene_name "([^"]+)"').alias('gene_name'),
        pl.col('attribute').str.extract(r'gene_id "([^"]+)"').alias('gene_id')
    ])

    # If requested, remove version suffix from gene_id
    if strip_version:
        # gene_df = gene_df.with_columns(
        #     pl.col("gene_id").apply(_remove_ensembl_version).alias("gene_id")
        # )
        # NOTE: In recent Polars versions, you cannot directly call .apply(...) on an expression 
        #       like pl.col("gene_id") the way you would in a Pandas pipeline. 

        gene_map_df = gene_map_df.with_columns(
            pl.col("gene_id").str.replace(r"\.\d+$", "").alias("gene_id")
        )
        # finds a literal dot followed by digits at the end of the string and removes it
        
    # If we have a list of genes, filter by name or ID
    if gene_names is not None:
        # If every element looks like an ENSG ID, filter on gene_id
        if all(str(name).startswith('ENSG') for name in gene_names):
            gene_df = gene_df.filter(pl.col('gene_id').is_in(list(gene_names)))
        else:
            gene_df = gene_df.filter(pl.col('gene_name').is_in(list(gene_names)))

    return gene_df.select(['gene_name', 'gene_id', 'seqname', 'start', 'end', 'strand'])


def _extract_genes_from_gtf_pandas(gtf_file_path, gene_names=None):
    """
    Fallback function: same as extract_genes_from_gtf but using pandas only.
    """
    df = pd.read_csv(
        gtf_file_path, sep='\t', comment='#', header=None,
        names=['seqname', 'source', 'feature', 'start', 'end',
               'score', 'strand', 'frame', 'attribute']
    )
    gene_df = df[df['feature'] == 'gene'].copy()

    gene_df['gene_name'] = gene_df['attribute'].str.extract(r'gene_name "([^"]+)"')
    gene_df['gene_id']   = gene_df['attribute'].str.extract(r'gene_id "([^"]+)"')

    if gene_names is not None:
        if all(str(name).startswith('ENSG') for name in gene_names):
            gene_df = gene_df[gene_df['gene_id'].isin(gene_names)]
        else:
            gene_df = gene_df[gene_df['gene_name'].isin(gene_names)]

    # Convert to Polars
    return pl.from_pandas(
        gene_df[['gene_name','gene_id','seqname','start','end','strand']]
    )

# -----------------------------------------------------
# 3) extract_gene_sequences => same as your existing version 
def normalize_strand(strand):
    if strand in ['+', '-']:
        return strand
    # If needed, handle other cases or raise an error 
    return strand

def extract_gene_sequences(genes_df, genome_fasta, output_format='dataframe', include_columns=None):
    """
    Extract DNA sequences for each gene from a reference genome FASTA file.
    
    Parameters
    ----------
    genes_df : pl.DataFrame or pd.DataFrame
        Must contain at least [gene_name, gene_id, seqname, start, end, strand].
    genome_fasta : str
        Path to the reference genome FASTA file.
    output_format : str
        'dict' or 'dataframe' (default 'dataframe').
    include_columns : list
        Additional columns to include in the final output.
    
    Returns
    -------
    dict or pl.DataFrame
        If 'dict', returns { gene_name -> sequence }.
        If 'dataframe', returns a Polars DataFrame with 
        [gene_name, gene_id, seqname, start, end, strand, sequence, ...].
    """
    from Bio import SeqIO

    # Convert to Polars if necessary
    if isinstance(genes_df, pd.DataFrame):
        genes_df = pl.DataFrame(genes_df)

    # Load the genome as a dictionary
    genome = SeqIO.to_dict(SeqIO.parse(genome_fasta, "fasta"))

    sequences = []
    n_neg_strand = 0
    n_pos_strand = 0

    # Iterate over each gene row
    for row in tqdm(genes_df.iter_rows(named=True), total=genes_df.height, desc="Processing genes"):
        gene_name = row['gene_name']
        gene_id   = row.get('gene_id', None)  # might be None if missing
        chrom     = row['seqname']
        start     = row['start'] - 1  # 0-based
        end       = row['end']
        strand    = normalize_strand(row['strand'])
        
        # Extract the sequence from the reference
        seq_record = genome.get(chrom, None)
        if seq_record is None:
            raise ValueError(f"Chromosome '{chrom}' not found in FASTA.")
        
        sequence = seq_record.seq[start:end]
        if strand == '-':
            sequence = sequence.reverse_complement()
            n_neg_strand += 1
        else:
            n_pos_strand += 1

        seq_info = {
            'gene_name': gene_name,
            'gene_id': gene_id,
            'seqname': chrom,
            'start': start,
            'end': end,
            'strand': strand,
            'sequence': str(sequence)
        }
        
        # Optionally add more columns
        if include_columns:
            for col in include_columns:
                seq_info[col] = row[col]

        sequences.append(seq_info)

    print("[info] Strand information:")
    print(f"... # Negative strand genes: {n_neg_strand}")
    print(f"... # Positive strand genes: {n_pos_strand}")

    if output_format == 'dict':
        # Return just {gene_name -> sequence}
        return {s['gene_name']: s['sequence'] for s in sequences}

    # Return a Polars DataFrame
    sequences_df = pl.DataFrame(sequences)
    # Reorder so 'sequence' is last
    cols_order = [
        'gene_name', 
        *(c for c in sequences_df.columns if c not in ['gene_name', 'sequence']), 
        'sequence'
    ]
    return sequences_df.select(cols_order)

# -----------------------------------------------------
# 4) Utility to save sequences
def save_sequences(seq_df, output_path, format='tsv'):
    """
    Save sequences in the specified format: TSV, CSV, or Parquet.
    """
    if format == 'tsv':
        seq_df.write_csv(output_path, separator='\t')
    elif format == 'csv':
        seq_df.write_csv(output_path)
    elif format == 'parquet':
        seq_df.write_parquet(output_path)
    else:
        raise ValueError(f"Unsupported format: {format}. Supported formats are 'tsv', 'csv', and 'parquet'.")

# 5) Optional: display truncated sequences 
def truncate_sequences(df, max_length=50):
    """
    Return the same dataframe but with truncated 'sequence' column for display purposes.
    """
    df = df.copy()
    df['sequence'] = df['sequence'].apply(lambda s: s[:max_length] + '...' if len(s) > max_length else s)
    return df

# -----------------------------------------------------
# 6) High-level function to do everything
def extract_and_save_gene_sequences(
    gtf_file,
    genome_fasta,
    genes=None,      # can be gene names or gene IDs
    output_dir=None, # location to save result
    output_basename="gene_sequences",
    output_format="tsv",
    include_columns=None,
    verbose=1
):
    """
    High-level workflow to parse a GTF, filter by given genes (names or IDs),
    extract sequences from a reference FASTA, and save to disk.

    Parameters
    ----------
    gtf_file : str
        Path to GTF file (e.g., Homo_sapiens.GRCh38.112.gtf)
    genome_fasta : str
        Path to genome FASTA (e.g., Homo_sapiens.GRCh38.dna.primary_assembly.fa)
    genes : set or list
        Gene names or Ensembl IDs (both handled automatically).
    output_dir : str
        Where to save output. If None, uses current directory.
    output_basename : str
        Base name for output file, e.g., "gene_sequences"
    output_format : str
        "tsv" (default) or "csv", etc.
    include_columns : list
        Additional columns from the GTF to propagate (e.g. 'some_annotation').
    verbose : int
        Verbosity level.

    Returns
    -------
    pl.DataFrame
        DataFrame with extracted gene sequences and metadata columns.
    """
    if output_dir is None:
        output_dir = os.getcwd()
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 1) Extract gene annotation from the GTF
    filtered_genes_df = extract_genes_from_gtf(gtf_file, gene_names=genes, use_polars=True)
    if verbose:
        n_all = filtered_genes_df.height
        print(f"[extract_and_save_gene_sequences] Filtered gene records: {n_all}")

    if n_all == 0:
        print("[warning] No matching genes found. Returning empty DataFrame.")
        return pl.DataFrame()

    # 2) Extract sequences for these genes
    seq_df = extract_gene_sequences(
        filtered_genes_df,
        genome_fasta,
        output_format='dataframe',
        include_columns=include_columns
    )
    if verbose:
        print("[extract_and_save_gene_sequences] Sequence extraction complete.")
        print(f"Result columns: {seq_df.columns}")

    # 3) Save
    output_path = os.path.join(output_dir, f"{output_basename}.{output_format}")
    save_sequences(seq_df, output_path, format=output_format)
    if verbose:
        print(f"[i/o] Saved gene sequences to:\n{output_path}\n")

    return seq_df

# -----------------------------------------------------
# 7) Example usage (demo)
def demo_extract_gene_sequences():
    """
    Demonstration of extracting STMN2 & UNC13A from a GTF/FASTA 
    and saving to a local test directory.
    """
    data_prefix = "/path/to/meta-spliceai/data/ensembl"
    local_dir = "/path/to/meta-spliceai/data/ensembl/test"
    os.makedirs(local_dir, exist_ok=True)

    genome_annot = os.path.join(data_prefix, "Homo_sapiens.GRCh38.112.gtf")
    genome_fasta = os.path.join(data_prefix, "Homo_sapiens.GRCh38.dna.primary_assembly.fa")

    # Example genes to filter by name (works similarly for Ensembl IDs)
    gene_names = {"STMN2", "UNC13A"}  

    seq_df = extract_and_save_gene_sequences(
        gtf_file=genome_annot,
        genome_fasta=genome_fasta,
        genes=gene_names,
        output_dir=local_dir,
        output_basename="gene_sequence",
        output_format="tsv",
        include_columns=['seqname', 'start', 'end', 'strand'],
        verbose=1
    )

    # Show truncated versions for illustration
    print("[i/o] Truncated sequences:")
    print(truncate_sequences(seq_df.to_pandas(), max_length=100))

##################################################################
# Utilities

def build_gene_id_to_name_map(
    gtf_file_path, 
    remove_version=True, 
    use_polars=True,
    output_path=None,
    verbose=1
):
    """
    Build a dictionary mapping Ensembl gene_id -> gene_name from a GTF file,
    optionally removing the version suffix from gene_id.

    Parameters
    ----------
    gtf_file_path : str
        Path to the input GTF file.
    remove_version : bool
        If True, remove the version suffix from gene_id (e.g. ENSG0000012345.6 -> ENSG0000012345).
    use_polars : bool
        If True, use the Polars-based version of extract_genes_from_gtf(); else fallback.
    output_path : str or None
        If provided, save the resulting map as a CSV/TSV (in Polars format).
    verbose : int
        Print progress messages if > 0.

    Returns
    -------
    dict
        A dictionary: { gene_id : gene_name } (with or without version removal).
    """

    # Load the gene table from the GTF
    gene_df = extract_genes_from_gtf(
        gtf_file_path=gtf_file_path,
        gene_names=None,  # no filtering
        use_polars=use_polars
    )  # => pl.DataFrame with columns [gene_name, gene_id, seqname, start, end, strand]

    # Convert to Polars DF if it isn't already
    if not isinstance(gene_df, pl.DataFrame):
        gene_df = pl.DataFrame(gene_df)

    # Focus on just gene_id and gene_name
    gene_map_df = gene_df.select(["gene_id", "gene_name"]).unique()

    # Remove version suffix if needed
    if remove_version:
        # gene_map_df = gene_map_df.with_columns(
        #     pl.col("gene_id").apply(_remove_ensembl_version).alias("gene_id")
        # )
        # NOTE: In recent Polars versions, you cannot directly call .apply(...) on an expression 
        #       like pl.col("gene_id") the way you would in a Pandas pipeline. 

        gene_map_df = gene_map_df.with_columns(
            pl.col("gene_id").str.replace(r"\.\d+$", "").alias("gene_id")
        )
        # finds a literal dot followed by digits at the end of the string and removes it
        # NOTE: Alternate Fix: Use .map_elements(...) for a Row-Wise Python Function

    # Convert to a dictionary { gene_id: gene_name }
    pd_map = gene_map_df.to_pandas()
    gene_id_to_name = dict(zip(pd_map["gene_id"], pd_map["gene_name"]))

    # Optionally save
    if output_path:
        # Convert back to polars for easy saving
        out_pl = pl.DataFrame(pd_map)
        # e.g., out_pl.write_csv(output_path, has_header=True, separator='\t') if you want TSV
        # or just use the file extension to pick a format. For simplicity, let's do CSV:
        out_pl.write_csv(output_path)
        if verbose:
            print(f"[i/o] Saved gene_id->gene_name mapping to {output_path}")

    if verbose:
        print(f"[info] Built gene_id->gene_name map with {len(gene_id_to_name)} entries.")

    return gene_id_to_name


def _remove_ensembl_version(gene_id_str: str) -> str:
    """
    Remove the version suffix from an Ensembl gene ID (e.g. ENSG0000012345.6 -> ENSG0000012345).

    If gene_id_str doesn't contain '.', it's returned as-is.
    """
    if not gene_id_str:
        return gene_id_str
    parts = gene_id_str.split(".")
    return parts[0] if parts else gene_id_str


def lookup_ensembl_ids(
    file_path: str,
    gene_names: list,
    file_format: str = "csv",
    verbose: int = 1
) -> dict:
    """
    Look up Ensembl IDs given gene names from a CSV/TSV file.

    Parameters
    ----------
    file_path : str
        Path to the input CSV/TSV file containing gene_id and gene_name columns.
    gene_names : list
        List of gene names to look up.
    file_format : str
        Format of the file ('csv' or 'tsv').
    verbose : int
        Print progress messages if > 0.

    Returns
    -------
    dict
        A dictionary: { gene_name : ensembl_id } for the given gene names.
    """
    # Load the file into a Polars DataFrame
    if file_format == "csv":
        gene_map_df = pl.read_csv(file_path)
    elif file_format == "tsv":
        gene_map_df = pl.read_csv(file_path, sep='\t')
    else:
        raise ValueError("Unsupported file format. Use 'csv' or 'tsv'.")

    # Ensure the DataFrame has the expected columns
    if not set(["gene_id", "gene_name"]).issubset(gene_map_df.columns):
        raise ValueError("The file must contain 'gene_id' and 'gene_name' columns.")

    # Filter the DataFrame for the given gene names
    filtered_df = gene_map_df.filter(pl.col("gene_name").is_in(gene_names))

    # Convert to a dictionary { gene_name: gene_id }
    gene_name_to_id = dict(zip(filtered_df["gene_name"], filtered_df["gene_id"]))

    if verbose:
        print(f"[info] Found Ensembl IDs for {len(gene_name_to_id)} out of {len(gene_names)} gene names.")

    return gene_name_to_id


##################################################################
# Workflow function

def run_workflow_extract_gene_sequences(
    data_prefix,
    gtf_filename="Homo_sapiens.GRCh38.112.gtf",
    fasta_filename="Homo_sapiens.GRCh38.dna.primary_assembly.fa",
    genes=None,
    output_dir=None,
    output_basename="gene_sequence",
    output_format="tsv",
    include_columns=None,
    verbose=1
):
    """
    A reusable workflow function for extracting gene sequences given a GTF file, 
    a reference genome FASTA file, and an optional set of gene identifiers (names or Ensembl IDs).

    Parameters
    ----------
    data_prefix : str
        Base directory that contains the GTF and FASTA files.
    gtf_filename : str
        Name of the GTF file relative to `data_prefix`.
    fasta_filename : str
        Name of the reference genome FASTA file relative to `data_prefix`.
    genes : set or list, optional
        Gene identifiers to filter on. If each item starts with 'ENSG', filter by Ensembl gene ID; 
        otherwise, filter by gene name. If None, all genes are extracted (be mindful that this 
        could be very large).
    output_dir : str, optional
        Where to save the resulting file. If None, defaults to the current directory.
    output_basename : str
        Base filename for the output (without extension).
    output_format : str
        File extension or format, e.g. 'tsv', 'csv', etc.
    include_columns : list, optional
        Additional columns from the GTF to include in the final output (e.g. ['seqname','start','end','strand']).
    verbose : int
        Verbosity level. If > 0, prints progress messages.

    Returns
    -------
    pl.DataFrame
        A Polars DataFrame containing the extracted sequences and associated metadata 
        (e.g., gene_name, gene_id, seqname, start, end, strand, sequence).
    """
    # Construct full paths to GTF/FASTA
    gtf_file = os.path.join(data_prefix, gtf_filename)
    genome_fasta = os.path.join(data_prefix, fasta_filename)

    # Default output directory to current working directory if not specified
    if output_dir is None:
        output_dir = os.getcwd()
    os.makedirs(output_dir, exist_ok=True)

    # Use the high-level function to extract and save sequences
    seq_df = extract_and_save_gene_sequences(
        gtf_file=gtf_file,
        genome_fasta=genome_fasta,
        genes=genes,
        output_dir=output_dir,
        output_basename=output_basename,
        output_format=output_format,
        include_columns=include_columns,
        verbose=verbose
    )

    if verbose:
        print("[workflow] Sequence extraction complete. Here's a truncated view:")
        # Optionally show truncated sequences
        truncated = truncate_sequences(seq_df.to_pandas(), max_length=100)
        print(truncated.head())

    return seq_df


def demo_build_gene_id_to_name_map():
    """
    Demonstration of building a gene_id -> gene_name map from a GTF file.
    """
    data_prefix = "/path/to/meta-spliceai/data/ensembl"
    gtf_file = os.path.join(data_prefix, "Homo_sapiens.GRCh38.112.gtf")
    gene_map = \
            build_gene_id_to_name_map(
                gtf_file_path=gtf_file,
                remove_version=True,
                output_path=os.path.join(data_prefix, "gene_id_name_map.csv")  # optional
            )

    # Test gene ID lookup given a list of gene names
    gene_names = ["STMN2", "UNC13A"]
    gene_id_to_name = lookup_ensembl_ids(
        file_path=os.path.join(data_prefix, "gene_id_name_map.csv"),
        gene_names=gene_names,
        file_format="csv"
    )

    print(f"[info] gene_id_to_name: {gene_id_to_name}")

    # Test gene name lookup given a list of gene IDs
    # gene_ids = ["ENSG0000012345", "ENSG0000012346"]
    # gene_name_to_id = lookup_ensembl_ids(
    #     file_path=os.path.join(data_prefix, "gene_id_name_map.csv"),
    #     gene_ids=gene_ids,
    #     file_format="csv"
    # )
    
    return 


def demo_run_workflow_extract_gene_sequences():
    """
    Demonstration of running the high-level workflow function.
    """
    data_prefix = "/path/to/meta-spliceai/data/ensembl"
    local_dir = "/path/to/meta-spliceai/data/ensembl/test"
    os.makedirs(local_dir, exist_ok=True)

    genes = {"STMN2", "UNC13A"}

    seq_df = \
        run_workflow_extract_gene_sequences(
            data_prefix,
            gtf_filename="Homo_sapiens.GRCh38.112.gtf",
            fasta_filename="Homo_sapiens.GRCh38.dna.primary_assembly.fa",
            genes=genes,
            output_dir=local_dir,
            output_basename="gene_sequence",
            output_format="tsv",
            include_columns=['seqname', 'start', 'end', 'strand'],
            verbose=1
        )

    # format = 'tsv'
    # output_path = os.path.join(local_dir, f"gene_sequence.{format}")
    # print(f"[i/o] Saving gene sequences to:\n{output_path}\n")
    # print("... cols(seq_df): ", list(seq_df.columns))
    # save_sequences(seq_df, output_path, format='tsv')

def demo(): 

    # Demonstration of running the high-level workflow function.
    # demo_run_workflow_extract_gene_sequences()

    # Demonstration of building a gene_id -> gene_name map from a GTF file.
    demo_build_gene_id_to_name_map()



if __name__ == "__main__":
    demo()

