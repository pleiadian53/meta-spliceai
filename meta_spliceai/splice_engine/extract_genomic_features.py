import os, sys
import random
import time 

import pandas as pd
import polars as pl
import numpy as np

import gffutils
from pybedtools import BedTool
from Bio import SeqIO
from collections import defaultdict
from tqdm import tqdm

from tabulate import tabulate

from .utils_fs import (
    build_gtf_database, 
    extract_all_gtf_annotations, 
    read_splice_sites
)

from .utils_doc import (
    print_emphasized,
    print_with_indent,
    print_section_separator, 
    display
)

from .utils_bio import (
    extract_genes_from_gtf,
    extract_exons_from_gtf,
    compute_exon_counts, 
    parse_attributes,
    filter_valid_splice_site_genes
)

# NOTE: 
#    ├── splice_engine/
#    │   ├── __init__.py
#    │   ├── extract_genomic_features.py
#    │   └── utils_fs.py
# 
#    python -m splice_engine.extract_genomic_features
#


from .analyzer import Analyzer
# class Analyzer(object): 
#     source = 'ensembl'
#     version = ''
#     prefix = "/home/bchiu/work/meta-spliceai"  # Configure this to your local path (e.g. /lakehouse/default/Files on Fabric)
#     data_dir = f"{prefix}/data/ensembl"
#     gtf_file = "/path/to/meta-spliceai/data/ensembl/Homo_sapiens.GRCh38.112.gtf"


class FeatureAnalyzer(Analyzer): 

    def __init__(self, data_dir=None, *, source='ensembl', version=None, gtf_file=None, overwrite=False, **kargs):
        super().__init__()
        self.source = source
        self.version = version
        self.data_dir = data_dir or f"{FeatureAnalyzer.prefix}/data/{source}"
        self.gtf_file = gtf_file or Analyzer.gtf_file
        self.overwrite = overwrite

        self.col_tid = kargs.get("col_tid", "transcript_id")
        self.format = kargs.get("format", 'tsv')
        self.separator = kargs.get("separator", ',' if self.format == 'csv' else '\t')

    @property
    def path_to_gene_features(self):
        return os.path.join(self.analysis_dir, f'gene_features.tsv')

    @property
    def path_to_transcript_features(self): 
        return os.path.join(self.analysis_dir, f'transcript_features.tsv')

    @property
    def path_to_exon_features(self):
        return os.path.join(self.analysis_dir, f'exon_features.tsv')

    @property
    def path_to_performance_datafrane_derived_features(self):
        return os.path.join(self.analysis_dir, f'performance_df_features.tsv')

    @property
    def path_to_transcript_df_from_gtf(self):
        return os.path.join(self.analysis_dir, f'transcript_df_from_gtf.tsv')

    @property
    def path_to_exon_df_from_gtf(self):
        return os.path.join(self.analysis_dir, f'exon_df_from_gtf.tsv')

    def retrieve_gene_features(self, to_pandas=False, verbose=1):
        # sep = '\t' if format == 'tsv' else ','
        if not self.overwrite and os.path.exists(self.path_to_gene_features):
            if verbose: 
                print(f"[i/o] Loading gene features from {self.path_to_gene_features}")

            sep = ',' if self.path_to_gene_features.endswith('.csv') else '\t'
            df = pl.read_csv(
                    self.path_to_gene_features, 
                    separator=sep, 
                    schema_overrides={'chrom': pl.Utf8}
                )
            if verbose:
                print(f"[cache-hit] Gene features loaded from cache (rows={df.shape[0]:,}).")
        else:
            if verbose: 
                if not self.overwrite: 
                    print(f"[warning] Could not find gene features at:\n{self.path_to_gene_features}\n")
                print(f"[action] Extracting gene features from GTF file ...")
            
            df = extract_gene_features_from_gtf(
                self.gtf_file, 
                use_cols=['start', 'end', 'score', 'strand', 'gene_id', 'gene_name', 'gene_type', 'gene_length', 'chrom'], 
                output_file=self.path_to_gene_features)
                # Columns: ['seqname', 'source', 'feature', 'start', 'end', 'score', 'strand', 'frame', 'attribute', 'gene_id', 'gene_name', 'gene_type', 'gene_length', 'chrom']
        
        if to_pandas:
            df = df.to_pandas()

        return df
    
    def retrieve_transcript_features(self, verbose=1, to_pandas=False):
        if not self.overwrite and os.path.exists(self.path_to_transcript_features):
            if verbose: 
                print(f"[i/o] Loading transcript features from {self.path_to_transcript_features}")

            sep = ',' if self.path_to_transcript_features.endswith('.csv') else '\t'
            df = pl.read_csv(
                    self.path_to_transcript_features, 
                    separator=sep, 
                    schema_overrides={'chrom': pl.Utf8}
                )
            if verbose:
                print(f"[cache-hit] Transcript features loaded from cache (rows={df.shape[0]:,}).")
        else:
            if verbose: 
                if not self.overwrite: 
                    print(f"[warning] Could not find transcript features at:\n{self.path_to_transcript_features}\n")
                print(f"[action] Extracting transcript features from GTF file ...")

            # summarize_transcript_features(gtf_file_path=self.gtf_file, output_path=self.path_to_transcript_features)
            df = extract_transcript_features_from_gtf(
                gtf_file_path=self.gtf_file, 
                output_file=self.path_to_transcript_features)

        if to_pandas:
            df = df.to_pandas()

        return df

    def retrieve_exon_features_at_transcript_level(self, verbose=1): 
        """
        Retrieve the exon features at transcript level, with caching support.
        
        Returns
        -------
        pl.DataFrame
            The exon DataFrame

        Memo
        ----
        - What infer_schema_length does:
            When Polars reads a CSV it samples the first N rows (default = 100) 
            to guess each column’s data type. If a categorical column contains only 
            numeric-looking values in that small sample it will be inferred as Int64; 
            encountering a later non-numeric value (e.g. “X”) raises the error you saw.
        """
        if not self.overwrite and os.path.exists(self.path_to_exon_features):
            if verbose: 
                print(f"[i/o] Loading exon features from {self.path_to_exon_features}")

            sep = ',' if self.path_to_exon_features.endswith('.csv') else '\t'
            df_cached = pl.read_csv(
                self.path_to_exon_features,
                separator=sep,
                schema_overrides={
                    'chrom': pl.Utf8,
                    'seqname': pl.Utf8,
                },
                infer_schema_length=10000,
            )
            # ------------------------------------------------------------------
            # Validate that cached file contains aggregated columns; otherwise
            # regenerate from GTF to ensure we have num_exons / length summaries.
            # ------------------------------------------------------------------
            required_cols = {
                'gene_id', 'transcript_id', 'num_exons',
                'avg_exon_length', 'median_exon_length', 'total_exon_length'
            }
            if not required_cols.issubset(set(df_cached.columns)):
                if verbose:
                    print(
                        "[cache-miss] Cached exon_features file lacks aggregated columns; "
                        "recomputing summaries from GTF …"
                    )
                df_cached = summarize_exon_features_at_transcript_level(
                    gtf_file_path=self.gtf_file,
                    output_file=self.path_to_exon_features,
                    verbose=verbose,
                    save=True,
                )
            else:
                if verbose:
                    print(f"[cache-hit] Using cached transcript-level exon summary (rows={df_cached.shape[0]:,}).")
            return df_cached
        else:
            if verbose: 
                if not self.overwrite:
                    print(f"[warning] Could not find exon features at:\n{self.path_to_exon_features}\n")
                print(f"[action] Extracting exon features from GTF file ...")

            return summarize_exon_features_at_transcript_level(
                gtf_file_path=self.gtf_file, output_file=self.path_to_exon_features)

    def retrieve_exon_features(self, verbose=1):
        return self.retrieve_exon_features_at_transcript_level(verbose=verbose)

    # extract_exons_from_gtf(gtf_file_path)
    def retrieve_exon_dataframe(self, verbose=1, to_pandas=True):
        """
        Retrieve the exon dataframe, with caching support.
        
        Uses the enhanced extract_exon_features_from_gtf function with proper parameter
        filtering based on the function signature.
        
        Parameters
        ----------
        verbose : int, optional
            Verbosity level, by default 1
        to_pandas : bool, optional
            Whether to return a pandas DataFrame, by default True
            
        Returns
        -------
        Union[pd.DataFrame, pl.DataFrame]
            The exon DataFrame
        """
        import inspect
        
        # Get the signature of extract_exon_features_from_gtf
        # func_sig = inspect.signature(extract_exon_features_from_gtf)
        # func_params = set(func_sig.parameters.keys())
        # NOTE: Using inspect.signature() on a function with **kargs, 
        #       will only shows the literal parameter names in the 
        #       function definition, not the parameters that are unpacked inside.
        
        # Build parameters dict based on the function signature
        # params = {
        #     'gtf_file_path': self.gtf_file,
        #     'verbose': verbose,
        #     'to_pandas': to_pandas,
        #     'cache_file': self.path_to_exon_df_from_gtf,
        #     'use_cache': not self.overwrite,
        #     'save': True,
        #     'include_extra_features': True
        # }

        # print("(retrieve_exon_dataframe(): to_pandas =", to_pandas)
        
        # Filter only parameters that the function accepts
        # filtered_params = {k: v for k, v in params.items() if k in func_params}
        
        # Directly pass the parameters we know the function accepts
        return extract_exon_features_from_gtf(
            gtf_file_path=self.gtf_file,
            verbose=verbose,
            to_pandas=to_pandas,
            cache_file=self.path_to_exon_df_from_gtf,
            use_cache=not self.overwrite,
            save=True,
            # Save raw exon table separately so we don’t overwrite the transcript-level
            # summary stored in `exon_features.tsv`.
            output_file=self.path_to_exon_df_from_gtf,
            include_extra_features=True,
            clean_output=True,
        )

    def retrieve_gene_level_performance_features(self, **kargs):
        eval_dir = FeatureAnalyzer.eval_dir
        output_file = self.path_to_performance_datafrane_derived_features
        verbose = kargs.get('verbose', 1)

        # Determine the separator based on the file extension
        if output_file.endswith('.csv'):
            separator = ','
        else:
            separator = '\t'

        if not self.overwrite and os.path.exists(output_file):
            if verbose: 
                print(f"[i/o] Loading gene features from {output_file}")
            df = pd.read_csv(output_file, sep=separator)
        else: 
            df = extract_gene_features_from_performance_profile(
                eval_dir=eval_dir, output_file=output_file, separator='\t', verbose=verbose) 
            # NOTE: performance dataframe is saved in '\t'

            # Save the resulting dataframe
            self.save_dataframe(df, file_name=output_file, sep=separator, verbose=verbose)

        return df

    def save_dataframe(self, df, file_name, sep='\t', verbose=1):
        """
        Save a DataFrame to a file. Supports both Pandas and Polars DataFrames.

        Parameters:
        - df (pd.DataFrame or pl.DataFrame): The DataFrame to save.
        - file_name (str): The name of the file to save the DataFrame to.
        - sep (str): The separator to use in the file. Default is tab ('\t').
        - verbose (int): Verbosity level. Default is 1.
        
        Returns:
        - str: The full path where the file was saved.
        """
        # Auto-detect separator based on file extension
        if file_name.endswith('.csv'):
            sep = ','
        elif file_name.endswith('.tsv'):
            sep = '\t'

        path = os.path.join(self.analysis_dir, file_name)
        if verbose:
            print(f"[i/o] Saving DataFrame to {path}")

        if isinstance(df, pd.DataFrame):
            df.to_csv(path, sep=sep, index=False)
        elif isinstance(df, pl.DataFrame):
            df.write_csv(path, separator=sep)
        else:
            raise ValueError("Unsupported DataFrame type. Only Pandas and Polars DataFrames are supported.")
            
        return path  # Return the path where the file was saved

    def load_dataframe(self, file_name, sep=None, to_pandas=False, verbose=1):
        """
        Load a DataFrame from a file. Supports both Pandas and Polars DataFrames.

        Parameters:
        - file_name (str): The name of the file to load the DataFrame from.
        - sep (str): The separator used in the file. If None, auto-detect based on file extension.
        - to_pandas (bool): If True, load the DataFrame as a Pandas DataFrame. Default is False (load as Polars DataFrame).
        - verbose (int): Verbosity level. Default is 1.

        Returns:
        - df (pd.DataFrame or pl.DataFrame): The loaded DataFrame.
        """
        # Auto-detect separator based on file extension
        if sep is None:
            if file_name.endswith('.csv'):
                sep = ','
            elif file_name.endswith('.tsv'):
                sep = '\t'
            else:
                sep = '\t'  # Default separator

        path = os.path.join(self.analysis_dir, file_name)
        if verbose:
            print(f"[i/o] Loading DataFrame from {path}")

        if to_pandas:
            df = pd.read_csv(path, sep=sep)
        else:
            df = pl.read_csv(path, separator=sep)

        return df


class SpliceAnalyzer(Analyzer): 
    
    db_file = os.path.join(Analyzer.data_dir, "annotations.db")
    splice_sites_file = "splice_sites.tsv"

    def __init__(self, data_dir=None, *, source='ensembl', version=None):
        self.source = source
        self.version = version
        self.data_dir = data_dir or f"{SpliceAnalyzer.prefix}/data/{source}"

    @property
    def path_to_splice_sites(self):
        return os.path.join(self.data_dir, SpliceAnalyzer.splice_sites_file)

    @property
    def path_to_overlapping_gene_metadata(self):
        return os.path.join(self.data_dir, "overlapping_gene_counts.tsv")

    def load_splice_sites(self, verbose=1, raise_exception=False): 
        splice_sites_file_path = self.path_to_splice_sites
        if os.path.exists(splice_sites_file_path):
            return read_splice_sites(splice_sites_file_path, verbose=verbose)
            # Output: pl.DataFrame

        if verbose: 
            print(f"[warning] Could not find recomputed splice sites at:\n{splice_sites_file_path}\n")

        if raise_exception:
            raise FileNotFoundError(f"Splice sites file not found at: {splice_sites_file_path}")

        return None

    def retrieve_splice_sites(self, consensus_window=2, verbose=1, column_names={}):
        df_splice = self.load_splice_sites(verbose=verbose, raise_exception=False)

        if df_splice is None or df_splice.is_empty():
            if verbose: 
                print_emphasized("[action] Computing splice sites from GTF ...")
            extract_splice_sites_workflow(
                    data_prefix=self.data_dir, 
                    gtf_file=Analyzer.gtf_file, 
                    output_path=self.path_to_splice_sites,
                    consensus_window=consensus_window
                )
            df_splice = self.load_splice_sites(verbose=verbose, raise_exception=True)

        # Standardize column names: change 'site_type' to 'splice_type'
        for key, value in column_names.items():
            if key in df_splice.columns:
                df_splice = df_splice.rename({key: value})

        return df_splice

    def retrieve_overlapping_gene_metadata(self, min_exons=2, filter_valid_splice_sites=True, **kargs):
        verbose = kargs.get('verbose', 1)
        output_format = kargs.get('output_format', 'dict')
        to_pandas = kargs.get('to_pandas', False)
        result_set = \
            get_or_load_overlapping_gene_metadata(
                gtf_file_path=Analyzer.gtf_file,
                overlapping_gene_path=self.path_to_overlapping_gene_metadata, 
                filter_valid_splice_sites=filter_valid_splice_sites, 
                min_exons=min_exons, 
                output_format=output_format, verbose=verbose)
        if output_format == 'dataframe': 
            if to_pandas: 
                return result_set.to_pandas()
            return result_set
        return result_set
        

def get_gene_sequences(fasta_file, gene_names):
    """
    Retrieve DNA sequences for a set of genes from a FASTA file and return as a DataFrame.

    Parameters:
    fasta_file (str): Path to the FASTA file containing DNA sequences.
    gene_names (set or list): A set or list of gene names to retrieve sequences for.

    Returns:
    pd.DataFrame: A DataFrame with columns 'Gene' and 'Sequence'.

    Example:

    # Example usage:
    fasta_file = "your_genome_sequences.fasta"  # Replace with your FASTA file path
    gene_names = {"GENE1", "GENE2", "GENE3"}  # Replace with your set of gene names
    gene_sequences_df = get_gene_sequences_df(fasta_file, gene_names)
    print(gene_sequences_df)
    """
    data = []
    for record in SeqIO.parse(fasta_file, "fasta"):
        gene_id = record.id
        if gene_id in gene_names:
            data.append({'Gene': gene_id, 'Sequence': str(record.seq)})
    
    return pd.DataFrame(data)


def extract_splice_sites(gtf_file):
    """
    Extract splice site coordinates from a GTF file.
    
    This function extracts donor and acceptor sites from a GTF/GFF file and saves 
    them in a BED format

    Example usage:
    splice_sites = extract_splice_sites("your_annotations.gtf")
    splice_sites.to_csv("splice_sites.bed", sep="\t", index=False, header=False)
    """
    gtf = BedTool(gtf_file)
    exons = gtf.filter(lambda x: x[2] == 'exon').saveas()
    
    splice_sites = []
    
    for exon in exons:
        chrom = exon.chrom
        strand = exon.strand
        exon_start = exon.start
        exon_end = exon.end
        
        # Donor site: 3' end of the exon (start for positive strand)
        if strand == '+':
            splice_sites.append((chrom, exon_end, exon_end + 1, "donor"))
        else:
            splice_sites.append((chrom, exon_start - 1, exon_start, "donor"))
        
        # Acceptor site: 5' end of the exon (end for positive strand)
        if strand == '+':
            splice_sites.append((chrom, exon_start - 1, exon_start, "acceptor"))
        else:
            splice_sites.append((chrom, exon_end, exon_end + 1, "acceptor"))
    
    # Convert to DataFrame for further processing
    splice_sites_df = pd.DataFrame(splice_sites, columns=["chrom", "start", "end", "type"])
    return splice_sites_df


def extract_sequences(fasta_file, annotations):
    """Extract DNA sequences corresponding to splice sites."""
    sequences = {}
    for record in SeqIO.parse(fasta_file, "fasta"):
        seq_id = record.id
        if seq_id in annotations['seqname'].values:
            sequences[seq_id] = str(record.seq)
    return sequences

def label_nucleotides(annotations, sequences):
    """Label nucleotides as donor, acceptor, or neither."""
    labeled_data = []
    for _, row in annotations.iterrows():
        if row['feature'] == 'exon':
            seq_id = row['seqname']
            sequence = sequences[seq_id]
            start, end = row['start'], row['end']
            for i in range(start, end + 1):
                if i == start:
                    labeled_data.append((sequence[i-1], 'donor'))
                elif i == end:
                    labeled_data.append((sequence[i-1], 'acceptor'))
                else:
                    labeled_data.append((sequence[i-1], 'neither'))
    return labeled_data


def prepare_dataset(labeled_data):
    """Prepare the dataset for model training."""
    from sklearn.model_selection import train_test_split
    
    sequences, labels = zip(*labeled_data)
    return train_test_split(sequences, labels, test_size=0.2, random_state=42)


def prepare_data_for_spliceator(gtf_file, sequence_fasta):
    """
    Extract and format data for Spliceator.
    
    Example usage:
        spliceator_input_data = prepare_data_for_spliceator("your_annotations.gtf", "genome_sequence.fasta")
    """
    splice_sites_df = extract_splice_sites(gtf_file)
    
    # Load the corresponding DNA sequences from FASTA
    from Bio import SeqIO
    sequences = {record.id: str(record.seq) for record in SeqIO.parse(sequence_fasta, "fasta")}
    
    spliceator_data = []
    
    for _, row in splice_sites_df.iterrows():
        chrom, start, end, splice_type = row
        seq = sequences[chrom][start:end]
        
        # Example: store as tuple (sequence, label)
        spliceator_data.append((seq, splice_type))
    
    return spliceator_data


def load_splice_sites_to_dict(
    splice_site_file,
    key_by="gene_id",  # or "gene_name"
    gene_id_to_name_map=None
):
    """
    Convert a TSV of splice-site annotation into a dictionary keyed by either gene_id or gene_name.

    The splice_sites_file has columns: 
      ['chrom','start','end','strand','site_type','gene_id','transcript_id','position']

    Parameters
    ----------
    splice_site_file : str
        Path to the splice site annotation file.
    key_by : {"gene_id", "gene_name"}
        How to key the returned dictionary.
    gene_id_to_name_map : dict, optional
        If key_by="gene_name", we need a mapping gene_id -> gene_name 
        (e.g. from your GTF). If not provided and key_by="gene_name", 
        we'll skip entries lacking a known mapping.

    Returns
    -------
    dict
      e.g. 
      {
         <key> : {
             "chrom":  <string or set of chroms>,
             "strand": <string or set of strands>,
             "donor_positions": [...],
             "acceptor_positions": [...],
             "transcripts": set([...])  # optional, if you want to store transcript info
         },
         ...
      }
    """
    from .utils_fs import read_splice_sites
    # df = pl.read_csv(splice_site_file, separator='\t')  # or read_csv(...)
    df = read_splice_sites(splice_site_file, separator='\t', dtypes=None)

    # Make sure required columns exist
    required_cols = {"chrom","start","end","strand","site_type","gene_id","transcript_id","position"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"File {splice_site_file} is missing columns: {missing}")

    # We'll group by gene_id first internally
    group_col = "gene_id"
    if key_by == "gene_name" and gene_id_to_name_map is None:
        print("[warning] key_by='gene_name' but no gene_id_to_name_map provided. " 
              "We'll skip entries if we can't map them.")
    
    # Convert to pandas for grouping or do it in Polars
    pdf = df.to_pandas()

    result = {}
    for gid, sub in pdf.groupby(group_col):
        # Optionally look up gene_name
        if key_by == "gene_name":
            gene_name = gene_id_to_name_map.get(gid, None) if gene_id_to_name_map else None
            if not gene_name:
                # If we can't find it, skip or handle differently
                continue
            dict_key = gene_name
        else:
            dict_key = gid

        # Separate donor and acceptor sites
        donors = sub.loc[sub["site_type"]=="donor","position"].tolist()
        acceptors = sub.loc[sub["site_type"]=="acceptor","position"].tolist()
        chroms = set(sub["chrom"].unique())     # usually just one
        strands = set(sub["strand"].unique())   # usually just one
        tx_ids = set(sub["transcript_id"].unique())

        # Prepare the dictionary record
        if dict_key not in result:
            result[dict_key] = {
                "chrom": chroms,
                "strand": strands,
                "donor_positions": donors,
                "acceptor_positions": acceptors,
                "transcripts": tx_ids
            }
        else:
            # If you have multiple rows per gene, you might want to unify them 
            # or append them. This depends on your data structure.
            result[dict_key]["chrom"].update(chroms)
            result[dict_key]["strand"].update(strands)
            result[dict_key]["donor_positions"].extend(donors)
            result[dict_key]["acceptor_positions"].extend(acceptors)
            result[dict_key]["transcripts"].update(tx_ids)

    # Optionally sort the positions
    for k,v in result.items():
        v["donor_positions"] = sorted(set(v["donor_positions"]))
        v["acceptor_positions"] = sorted(set(v["acceptor_positions"]))
        # Convert sets to something consistent
        v["chrom"] = list(v["chrom"])
        v["strand"] = list(v["strand"])
        v["transcripts"] = list(v["transcripts"])

    return result


def get_annotated_splice_sites(
    gtf_file, 
    genes=None,               # set/list of gene names or gene IDs
    feature="exon",           # or "CDS", "UTR", etc. or a list of features
    key_by="auto"             # "auto", "gene_name", or "gene_id"
):
    """
    Extract annotated splice sites (exon boundaries) from the GTF file
    for the specified gene set. Returns a dict keyed by gene_name or gene_id.

    Parameters
    ----------
    gtf_file : str
        Path to the GTF file.
    genes : set/list, optional
        If provided, we filter to these genes. If all of them look like "ENSG", 
        we filter gene_id. Otherwise, we filter gene_name.
    feature : str or list
        GTF feature type(s) to consider. Usually "exon" (default), 
        or you can pass ["exon","CDS"] or any subset.
    key_by : {"auto", "gene_name", "gene_id"}
        If "auto", then if the user input `genes` are all "ENSG", 
        we key by gene_id; otherwise, by gene_name. 
        If you want to force a certain key, set "gene_name" or "gene_id".

    Returns
    -------
    dict
      either {gene_name -> { "chrom":..., "strand":..., "sites": sorted positions}}
      or     {gene_id   -> { "chrom":..., "strand":..., "sites": sorted positions}}
    """
    if isinstance(feature, str):
        feature = [feature]

    # Read GTF with Polars
    columns = ['seqname','source','feature','start','end','score','strand','frame','attribute']
    df = pl.read_csv(
        gtf_file, 
        separator='\t', 
        comment_prefix='#', 
        has_header=False, 
        new_columns=columns
    )

    # Keep only the desired features (e.g. "exon")
    df = df.filter(pl.col("feature").is_in(feature))

    # Extract gene_name/gene_id from attribute
    df = df.with_columns([
        pl.col("attribute").str.extract(r'gene_name "([^"]+)"').alias("gene_name"),
        pl.col("attribute").str.extract(r'gene_id "([^"]+)"').alias("gene_id")
    ])

    # Determine whether to filter by gene_name or gene_id
    filter_on_id = False
    if genes is not None:
        # If every element in genes starts with 'ENSG', assume those are gene_ids
        if all(str(g).startswith("ENSG") for g in genes):
            filter_on_id = True

        # But if key_by is explicitly set
        if key_by == "gene_id":
            filter_on_id = True
        elif key_by == "gene_name":
            filter_on_id = False

        # Now filter
        if filter_on_id:
            df = df.filter(pl.col("gene_id").is_in(list(genes)))
        else:
            df = df.filter(pl.col("gene_name").is_in(list(genes)))

    # Decide final key
    final_key = "gene_id" if filter_on_id else "gene_name"

    # Convert to pandas for grouping
    pd_df = df.to_pandas()

    annotated_sites = {}
    for key_val, sub in pd_df.groupby(final_key):
        if sub.empty:
            continue
        # For each group, gather boundary positions
        # e.g. take union of 'start' and 'end'
        starts = sub["start"].tolist()
        ends = sub["end"].tolist()
        boundaries = sorted(set(starts + ends))

        chrom = sub["seqname"].iloc[0]
        strand = sub["strand"].iloc[0]

        annotated_sites[key_val] = {
            "chrom": chrom,
            "strand": strand,
            "sites": boundaries
        }

    return annotated_sites


def extract_junctions_for_gene(db, gene_identifier, consensus_window=0):
    """
    Extract all annotated transcripts and their associated junctions for a given gene,
    considering strand orientation and exon indexing.

    Parameters:
    - db (gffutils.FeatureDB): A gffutils database created from a GTF or GFF file.
    - gene_identifier (str): Either gene name or gene ID (e.g., Ensembl ID).
    - consensus_window (int): Number of nucleotides to include before and after the splice site.

    Returns:
    - gene_junctions (pd.DataFrame): DataFrame containing junctions for all annotated transcripts.

    Example usage:
    db = gffutils.FeatureDB('path_to_gffutils_database.db')
    gene_junctions_df = extract_junctions_for_gene(db, 'ENSG00000123456')
    print(gene_junctions_df)
    """
    # import gffutils
    gene_junctions = []

    # Find the gene using the provided gene_identifier
    try:
        gene = db[gene_identifier]
    except gffutils.FeatureNotFoundError:
        print(f"Gene '{gene_identifier}' not found in the database.")
        return pd.DataFrame()

    # Loop over each transcript associated with the gene
    for transcript in db.children(gene, featuretype='transcript', order_by='start'):
        transcript_id = transcript.attributes.get('transcript_id', [transcript.id])[0]
        gene_id = transcript.attributes.get('gene_id', [''])[0]
        strand = transcript.strand
        chrom = transcript.chrom

        # Extract exons, sorted by genomic coordinates
        exons = list(db.children(transcript, featuretype='exon', order_by='start'))

        # Skip transcripts with less than 2 exons (no splicing)
        if len(exons) < 2:
            continue

        # Iterate over exon boundaries to find junctions
        if strand == '+':
            # Positive strand: Loop through exons from first to last
            for i in range(len(exons) - 1):
                donor_end = exons[i].end  # Donor site: end of the current exon
                acceptor_start = exons[i + 1].start  # Acceptor site: start of the next exon

                # Store junction information
                junction_name = f"{transcript_id}_JUNC{i:08d}"
                gene_junctions.append({
                    'chrom': chrom,
                    'start': donor_end,  # Start position in BED is 0-based
                    'end': acceptor_start,  # End position is 1-based
                    'name': junction_name,
                    'score': 0,
                    'strand': strand,
                })

        elif strand == '-':
            # Negative strand: Loop through exons in reverse order (last to first in transcript sense)
            for i in range(len(exons) - 1, 0, -1):
                donor_start = exons[i].start  # Donor site: start of the current exon
                acceptor_end = exons[i - 1].end  # Acceptor site: end of the next exon in reverse order

                # Store junction information
                junction_name = f"{transcript_id}_JUNC{i:08d}"
                gene_junctions.append({
                    'chrom': chrom,
                    'start': acceptor_end,  # Acceptor site is the end of the next exon
                    'end': donor_start,  # Donor site is the start of the current exon
                    'name': junction_name,
                    'score': 0,
                    'strand': strand,
                })

    # Convert to DataFrame
    return pd.DataFrame(gene_junctions, columns=['chrom', 'start', 'end', 'name', 'score', 'strand'])


def extract_splice_sites_for_all_genes(db, consensus_window=1, save=False, return_df=True, **kargs):
    """
    Extract all donor and acceptor sites for all genes using gffutils, accounting for strand information.

    Parameters:
    - db (gffutils.FeatureDB): A gffutils database created from a GTF or GFF file.
    - consensus_window (int): Number of nucleotides to include before and after the splice site.
    - save (bool): Whether to save the results to a file.
    - return_df (bool): Whether to return the results as a DataFrame.
    - **kargs: Additional keyword arguments for saving the file.

    Returns:
    - all_splice_sites (list of dicts by default): 
       A list where each element is a dictionary containing
      'chrom', 'start', 'end', 'strand', 'site_type', 'gene_id', 'transcript_id'.
      if return_df is set to True, a DataFrame is returned instead.

    Notes: 
    - Strand-Specific Splice Site Logic:
        * For the positive strand (+):
            - Donor Site: Located at exon_end + 1 (first base of intron), except for the last exon.
            - Acceptor Site: Located at exon_start - 1 (last base of intron), except for the first exon.
        * For the negative strand (-):
            - Donor Site: Located at exon_start - 1 (first base of intron in transcription order), except for the last exon in transcription order.
            - Acceptor Site: Located at exon_end + 1 (last base of intron in transcription order), except for the first exon in transcription order.
    

        ----xxxxxx----xxxxx---------xxxxxx---------->
                  d  a     d       a

        <---xxxxxx----xxxxx---------xxxxxx-----------
                  a  d     a       d

    """
    import gffutils

    all_splice_sites = []
    transcript_counter = 0

    for transcript in tqdm(db.features_of_type('transcript'), desc="Processing transcripts"):
        transcript_id = transcript.attributes.get('transcript_id', [transcript.id])[0]
        gene_id = transcript.attributes.get('gene_id', [''])[0]
        strand = transcript.strand
        chrom = transcript.chrom

        # Extract exons, sorted by genomic coordinates
        exons = list(db.children(transcript, featuretype='exon', order_by='start'))

        # Skip transcripts with less than 2 exons (no splicing)
        if len(exons) < 2:
            continue

        # Adjust exon order for negative strand to match transcription order
        if strand == '-':
            exons = exons[::-1]

        # Iterate over exons to find splice sites
        for i in range(len(exons)):
            exon = exons[i]
            exon_start = exon.start  # 1-based
            exon_end = exon.end      # 1-based

            # Test: Add assertions to verify exon order
            if strand == '+':
                if i > 0:
                    assert exons[i].start > exons[i - 1].start, \
                        f"Exon start positions are not increasing on positive strand: {exons[i - 1].start} !< {exons[i].start}"
            elif strand == '-':
                if i > 0:
                    assert exons[i].start < exons[i - 1].start, \
                        f"Exon start positions are not decreasing on negative strand: {exons[i - 1].start} !> {exons[i].start}"

            if strand == '+':
                # Positive strand
                # ----xxxxxx----xxxxx---------xxxxxx---------->
                #           d  a     d       a

                # Acceptor site: exon_start - 1 (except first exon)
                if i > 0:
                    position = exon_start - 1  # 1-based

                    # Extract consensus window
                    start = position - consensus_window # + 1
                    end = position + consensus_window

                    all_splice_sites.append({
                        'chrom': chrom,
                        'start': max(start, 1),  # Ensure start >= 1
                        'end': end,
                        'position': position,
                        'strand': strand,
                        'site_type': 'acceptor',
                        'gene_id': gene_id,
                        'transcript_id': transcript_id
                    })

                # Donor site: exon_end + 1 (except last exon)
                if i < len(exons) - 1:
                    position = exon_end + 1  # 1-based

                    # Extract consensus window
                    start = position - consensus_window  # + 1
                    end = position + consensus_window

                    all_splice_sites.append({
                        'chrom': chrom,
                        'start': max(start, 1),  # Ensure start >= 1
                        'end': end,
                        'position': position,
                        'strand': strand,
                        'site_type': 'donor',
                        'gene_id': gene_id,
                        'transcript_id': transcript_id
                    })

            elif strand == '-':
                # Negative strand
                # <---xxxxxx----xxxxx---------xxxxxx-----------
                #           a  d     a       d
                #                               i=0

                # Acceptor site: exon_end + 1 (except first exon in transcription order)
                if i > 0:
                    position = exon_end + 1  # 1-based

                    # Extract consensus window
                    start = position - consensus_window  # + 1
                    end = position + consensus_window

                    all_splice_sites.append({
                        'chrom': chrom,
                        'start': max(start, 1),  # Ensure start >= 1
                        'end': end,
                        'position': position,
                        'strand': strand,
                        'site_type': 'acceptor',
                        'gene_id': gene_id,
                        'transcript_id': transcript_id
                    })

                # Donor site: exon_start - 1 (except last exon in transcription order)
                if i < len(exons) - 1:
                    position = exon_start - 1  # 1-based

                    # Extract consensus window
                    start = position - consensus_window  # + 1
                    end = position + consensus_window

                    all_splice_sites.append({
                        'chrom': chrom,
                        'start': max(start, 1),  # Ensure start >= 1
                        'end': end,
                        'position': position,
                        'strand': strand,
                        'site_type': 'donor',
                        'gene_id': gene_id,
                        'transcript_id': transcript_id
                    })

        transcript_counter += 1

    # Save or return the results
    if save:
        splice_sites_df = pd.DataFrame(all_splice_sites)
        sep = kargs.get('sep', '\t')
        output_file = kargs.get('output_file', 'splice_sites.tsv')
        splice_sites_df.to_csv(output_file, index=False, sep=sep)
        print(f"[i/o] Splice sites saved to {output_file}")

    if return_df:
        return pd.DataFrame(all_splice_sites)
    else:
        return all_splice_sites


def extract_splice_sites_for_all_genes_v1(db, consensus_window=1, save=False, return_df=True, **kargs):
    """
    Extract all donor and acceptor sites for all genes using gffutils, accounting for strand information.

    Parameters:
    - db (gffutils.FeatureDB): A gffutils database created from a GTF or GFF file.
    - consensus_window (int): Number of nucleotides to include before and after the splice site.
    - save (bool): Whether to save the results to a file.
    - return_df (bool): Whether to return the results as a DataFrame.
    - **kargs: Additional keyword arguments for saving the file.

    Returns:
    - all_splice_sites (list of dicts): A list where each element is a dictionary containing
      'chrom', 'start', 'end', 'strand', 'site_type', 'gene_id', 'transcript_id'.
    """
    import gffutils

    all_splice_sites = []
    transcript_counter = 0

    for transcript in tqdm(db.features_of_type('transcript'), desc="Processing transcripts"):
        transcript_id = transcript.attributes.get('transcript_id', [transcript.id])[0]
        gene_id = transcript.attributes.get('gene_id', [''])[0]
        strand = transcript.strand
        chrom = transcript.chrom

        # Extract exons, sorted by genomic coordinates
        exons = list(db.children(transcript, featuretype='exon', order_by='start'))

        # Skip transcripts with less than 2 exons (no splicing)
        if len(exons) < 2:
            continue

        # Iterate over exon boundaries to find splice sites
        for i in range(len(exons)):
            exon = exons[i]

            if strand == '+':
                # Positive strand: Donor at exon end, acceptor at exon start
                exon_start = exon.start
                exon_end = exon.end

                # Acceptor site: start of the current exon (except for the first exon)
                if i > 0:
                    position = exon_start - 1  # The last nucleotide of the intron before the exon

                    # Extract consensus sequence region
                    acceptor_start = position - consensus_window
                    acceptor_end = position + consensus_window   

                    all_splice_sites.append({
                        'chrom': chrom,
                        'start': acceptor_start,
                        'end': acceptor_end,
                        'strand': strand,
                        'site_type': 'acceptor',
                        'gene_id': gene_id,
                        'transcript_id': transcript_id
                    })

                # Donor site: end of the current exon (except for the last exon)
                if i < len(exons) - 1:
                    position = exon_end + 1  # The first nucleotide of the intron after the exon

                    # Extract consensus sequence region
                    donor_start = position - consensus_window
                    donor_end = position + consensus_window     

                    all_splice_sites.append({
                        'chrom': chrom,
                        'start': donor_start,
                        'end': donor_end,
                        'strand': strand,
                        'site_type': 'donor',
                        'gene_id': gene_id,
                        'transcript_id': transcript_id
                    })

            elif strand == '-':
                # Negative strand: Donor at exon start, acceptor at exon end (reverse logic)
                exon_start = exon.start
                exon_end = exon.end

                # Acceptor site: end of the current exon (except for the last exon in reverse order)
                if i < len(exons) - 1:
                    position = exon_end + 1  # The first nucleotide of the intron after the exon

                    # Extract consensus sequence region
                    acceptor_start = position - consensus_window
                    acceptor_end = position + consensus_window   

                    all_splice_sites.append({
                        'chrom': chrom,
                        'start': acceptor_start,
                        'end': acceptor_end,
                        'strand': strand,
                        'site_type': 'acceptor',
                        'gene_id': gene_id,
                        'transcript_id': transcript_id
                    })

                # Donor site: start of the current exon (except for the first exon in reverse order)
                if i > 0:
                    position = exon_start - 1  # The last nucleotide of the intron before the exon

                    # Extract consensus sequence region
                    donor_start = position - consensus_window
                    donor_end = position + consensus_window     

                    all_splice_sites.append({
                        'chrom': chrom,
                        'start': donor_start,
                        'end': donor_end,
                        'strand': strand,
                        'site_type': 'donor',
                        'gene_id': gene_id,
                        'transcript_id': transcript_id
                    })

        transcript_counter += 1

    # Save or return the splice sites
    if save:
        sep = kargs.get('sep', '\t')
        format = 'csv' if sep == ',' else 'tsv'
        output_file = kargs.get('output_file', f'splice_sites.{format}')

        splice_sites_df = pd.DataFrame(all_splice_sites)
        splice_sites_df.to_csv(output_file, index=False, sep=sep)
        print(f"[i/o] Splice sites saved to {output_file}")

    return pd.DataFrame(all_splice_sites) if return_df else all_splice_sites


def extract_splice_sites_for_all_genes_v0(db, consensus_window=1, save=False, return_df=True, **kargs):
    """
    Extract all donor and acceptor sites for all genes using gffutils, accounting for strand information.

    Parameters:
    - db (gffutils.FeatureDB): A gffutils database created from a GTF or GFF file.
    - consensus_window (int): Number of nucleotides to include before and after the splice site.
    - save (bool): Whether to save the results to a file.
    - return_df (bool): Whether to return the results as a DataFrame.
    - **kargs: Additional keyword arguments for saving the file.

    Returns:
    - pd.DataFrame or list: DataFrame or list of dictionaries containing splice site information.
    """
    all_splice_sites = []

    for transcript in tqdm(db.features_of_type('transcript'), desc="Processing transcripts"):
        transcript_id = transcript.attributes.get('transcript_id', [transcript.id])[0]
        gene_id = transcript.attributes.get('gene_id', [''])[0]
        strand = transcript.strand
        chrom = transcript.chrom

        # Extract exons, sorted by genomic coordinates
        exons = list(db.children(transcript, featuretype='exon', order_by='start'))

        # Skip transcripts with less than 2 exons (no splicing)
        if len(exons) < 2:
            continue

        # Iterate over exon boundaries to find splice sites
        for i in range(len(exons)):
            exon = exons[i]
            exon_start = exon.start
            exon_end = exon.end

            if strand == '+':
                # Positive strand logic

                # Acceptor site: start of the current exon (except for the first exon)
                if i > 0:
                    position = exon_start - 1  # The last nucleotide of the intron

                    # Extract consensus sequence region
                    acceptor_start = position - consensus_window
                    acceptor_end = position + consensus_window

                    all_splice_sites.append({
                        'chrom': chrom,
                        'start': acceptor_start,
                        'end': acceptor_end,
                        'strand': strand,
                        'site_type': 'acceptor',
                        'gene_id': gene_id,
                        'transcript_id': transcript_id
                    })

                # Donor site: end of the current exon (except for the last exon)
                if i < len(exons) - 1:
                    position = exon_end + 1  # The first nucleotide of the intron

                    # Extract consensus sequence region
                    donor_start = position - consensus_window
                    donor_end = position + consensus_window

                    all_splice_sites.append({
                        'chrom': chrom,
                        'start': donor_start,
                        'end': donor_end,
                        'strand': strand,
                        'site_type': 'donor',
                        'gene_id': gene_id,
                        'transcript_id': transcript_id
                    })

            elif strand == '-':
                # Negative strand logic

                # Acceptor site: end of the current exon (except for the last exon)
                if i < len(exons) - 1:
                    position = exon_end + 1  # The first nucleotide of the intron

                    # Extract consensus sequence region
                    acceptor_start = position - consensus_window
                    acceptor_end = position + consensus_window

                    all_splice_sites.append({
                        'chrom': chrom,
                        'start': acceptor_start,
                        'end': acceptor_end,
                        'strand': strand,
                        'site_type': 'acceptor',
                        'gene_id': gene_id,
                        'transcript_id': transcript_id
                    })

                # Donor site: start of the current exon (except for the first exon)
                if i > 0:
                    position = exon_start - 1  # The last nucleotide of the intron

                    # Extract consensus sequence region
                    donor_start = position - consensus_window
                    donor_end = position + consensus_window

                    all_splice_sites.append({
                        'chrom': chrom,
                        'start': donor_start,
                        'end': donor_end,
                        'strand': strand,
                        'site_type': 'donor',
                        'gene_id': gene_id,
                        'transcript_id': transcript_id
                    })

    # Save or return the splice sites
    if save:
        sep = kargs.get('sep', '\t')
        file_format = 'csv' if sep == ',' else 'tsv'
        output_file = kargs.get('output_file', f'splice_sites.{file_format}')

        splice_sites_df = pd.DataFrame(all_splice_sites)
        splice_sites_df.to_csv(output_file, index=False, sep=sep)
        print(f"[i/o] Splice sites saved to {output_file}")

    if return_df:
        return pd.DataFrame(all_splice_sites)
    else:
        return all_splice_sites


def create_asymmetric_labels(window_size, site_type):
    """
    Create asymmetric probability labels for splice sites based on type.
    
    - window_size: Size of the consensus window around the splice site.
    - site_type: Either 'donor' or 'acceptor'.
    
    Returns:
    - A list of probability labels decreasing away from the splice site.
    """
    labels = [1]  # Exact site has label 1

    # Probability drops based on consensus window, decreasing away from the site
    for i in range(1, window_size + 1):
        if site_type == 'donor':
            labels.insert(0, 1 - i * 0.1)  # Adjust these values based on expected importance
        elif site_type == 'acceptor':
            labels.append(1 - i * 0.1)  # Adjust based on expected importance
    
    return labels


def extend_annotated_sites_with_consensus(annotated_splice_df, consensus_sequences, tolerance):
    """
    Extend annotated splice sites to account for consensus sequences around donor and acceptor sites, considering strand.

    Parameters:
    - annotated_splice_df (pd.DataFrame): Annotated splice sites DataFrame.
    - consensus_sequences (dict): A dictionary with 'donor' and 'acceptor' keys and their consensus sequences as values.
    - tolerance (int): Number of nucleotides around the splice site boundary to be considered a match.

    Returns:
    - extended_regions (dict): A dictionary of annotated regions extended by consensus sequences.
    """
    extended_regions = defaultdict(lambda: {'donor': [], 'acceptor': []})

    donor_consensus_length = len(consensus_sequences['donor'])
    acceptor_consensus_length = len(consensus_sequences['acceptor'])

    for idx, row in annotated_splice_df.iterrows():
        transcript_id = row['transcript_id']
        chrom = row['chrom']
        strand = row['strand']
        splice_type = row['splice_type']

        if splice_type == 'donor':
            if strand == '+':
                start = row['end'] - donor_consensus_length - tolerance
                end = row['end'] + tolerance
            else:
                start = row['start'] - tolerance
                end = row['start'] + donor_consensus_length + tolerance

            extended_regions[transcript_id]['donor'].append((chrom, start, end, strand))

        elif splice_type == 'acceptor':
            if strand == '+':
                start = row['start'] - tolerance
                end = row['start'] + acceptor_consensus_length + tolerance
            else:
                start = row['end'] - acceptor_consensus_length - tolerance
                end = row['end'] + tolerance

            extended_regions[transcript_id]['acceptor'].append((chrom, start, end, strand))

    return extended_regions


def match_predicted_to_annotated_sites(spliceai_predictions, annotated_sites, tolerance=2):
    """
    Match predicted splice sites to annotated splice sites within a positional tolerance.

    Parameters:
    - spliceai_predictions (pd.DataFrame): DataFrame containing SpliceAI predictions.
    - annotated_sites (pd.DataFrame): DataFrame of annotated splice sites from extract_splice_sites_for_all_genes().
    - tolerance (int): Number of nucleotides to consider around the annotated site for a match.

    Returns:
    - results_df (pd.DataFrame): DataFrame containing matched predictions with labels (TP, FP, FN).
    """
    # import pandas as pd

    # Ensure 'position' is included in annotated_sites
    annotated_sites['position'] = (annotated_sites['start'] + annotated_sites['end']) // 2

    # Create sets for quick lookup
    annotated_positions = set(zip(
        annotated_sites['chrom'],
        annotated_sites['position'],
        annotated_sites['strand'],
        annotated_sites['site_type']
    ))

    # Initialize lists to store results
    true_positives = []
    false_positives = []
    false_negatives = []

    # Iterate over predicted positions
    for idx, row in spliceai_predictions.iterrows():
        chrom = row['chrom']
        position = row['position']
        strand = row['strand']

        # Check for donor prediction
        if row['donor_prob'] >= 0.5:  # Threshold can be adjusted
            matched = False
            for delta in range(-tolerance, tolerance + 1):
                key = (chrom, position + delta, strand, 'donor')
                if key in annotated_positions:
                    true_positives.append(row)
                    matched = True
                    break
            if not matched:
                false_positives.append(row)
        # Check for acceptor prediction
        if row['acceptor_prob'] >= 0.5:
            matched = False
            for delta in range(-tolerance, tolerance + 1):
                key = (chrom, position + delta, strand, 'acceptor')
                if key in annotated_positions:
                    true_positives.append(row)
                    matched = True
                    break
            if not matched:
                false_positives.append(row)

    # Find false negatives
    predicted_positions = set(zip(
        spliceai_predictions['chrom'],
        spliceai_predictions['position'],
        spliceai_predictions['strand']
    ))
    for idx, row in annotated_sites.iterrows():
        chrom = row['chrom']
        position = row['position']
        strand = row['strand']
        site_type = row['site_type']

        matched = False
        for delta in range(-tolerance, tolerance + 1):
            key = (chrom, position + delta, strand)
            if key in predicted_positions:
                matched = True
                break
        if not matched:
            false_negatives.append(row)

    # Compile results
    tp_df = pd.DataFrame(true_positives)
    fp_df = pd.DataFrame(false_positives)
    fn_df = pd.DataFrame(false_negatives)

    return tp_df, fp_df, fn_df


def extract_annotations(gtf_file, db_file='annotations.db', output_file='annotations_all_transcripts.tsv', **kargs):
    # Step 1: Build the database
    db = build_gtf_database(gtf_file, db_file=db_file, overwrite=False)

    # Step 2: Extract exon, CDS, 5'UTR, and 3'UTR annotations for all transcripts from the DB
    annotations_df = extract_all_gtf_annotations(db)

    # Step 3: Save annotations to a file
    annotations_df.to_csv(output_file, index=False, **kargs)
    print(f"[info] Annotations extracted and saved to {output_file}")


def transcript_sequence_retrieval_workflow(gtf_file, genome_fasta, gene_tx_map=None, output_file="tx_sequence.parquet", **kargs):
    from .utils_bio import (
        extract_transcripts_from_gtf,
        extract_transcript_sequences,
        save_sequences, 
        save_sequences_by_chromosome, 
        load_sequences, 
        load_sequences_by_chromosome
    )

    # data_prefix = "/path/to/meta-spliceai/data/ensembl"
    # local_dir = "/path/to/meta-spliceai/data/ensembl"
    # genome_annot = os.path.join(data_prefix, "Homo_sapiens.GRCh38.112.gtf") 
    # genome_fasta = os.path.join(data_prefix, "Homo_sapiens.GRCh38.dna.primary_assembly.fa") # "/path/to/Homo_sapiens.GRCh38.dna.primary_assembly.fa"
    
    tx_ids = gene_tx_map  # Set to None to extract all transcripts
    # NOTE: To focus on specific genes and transcripts
    #    E.g. gene_names = {"STMN2", "UNC13A"}
    # 
    # Set tx_ids as a dictionary: 
    #
    # tx_ids = {'STMN2': ['ENST00000220876', ], 
    #           'UNC13A': ['ENST00000519716', ]}
    # or
    # 
    # tx_ids = {'ENSG00000104435': ['ENST00000220876'], 'ENSG00000130477': ['ENST00000519716']}  # Example with gene IDs

    print_emphasized("[action] Extracting pre-mRNA sequences ...")

    # Extract transcripts and their features
    transcripts_df = extract_transcripts_from_gtf(gtf_file, tx_ids=tx_ids, ignore_version=True)
    print(transcripts_df.head())  # Display the extracted transcript information
    # Columns: ['transcript_id', 'gene_name', 'gene_id', 'seqname', 'start', 'end', 'strand']

    additional_columns = ['seqname', 'start', 'end', 'strand', ]
    seq_df = extract_transcript_sequences(transcripts_df, genome_fasta, output_format='dataframe', include_columns=additional_columns)
    
    format = kargs.get('format', 'parquet')
    output_path = output_file  # os.path.join(local_dir, f"tx_sequence.{format}")
    assert os.path.exists(os.path.dirname(output_path))

    print(f"[i/o] Saving transcript sequences to:\n{output_path}\n")
    print("... cols(seq_df): ", list(seq_df.columns))
    
    
    # NOTE: Columns: ['transcript_id', 'gene_name', 'strand', 'seqname', 'start', 'end', 'sequence']
    # save_sequences(seq_df, output_path, format=format)

    chromosomes = [str(i) for i in range(1, 23)] + ['X', 'Y', 'MT']
    save_sequences_by_chromosome(seq_df, output_path, format=format, chromosomes_to_save=chromosomes)

    return 


def gene_sequence_retrieval_workflow(gtf_file, genome_fasta, gene_tx_map=None, output_file="gene_sequence.parquet", mode='minmax', **kargs): 
    """
    Extract gene-specific sequences from a GTF file and a reference genome FASTA file, 
    and save them chromosome by chromosome.

    This function extracts DNA sequences for genes based on their coordinates in a GTF file and 
    a reference genome FASTA file.
    
    The sequences are then saved chromosome by chromosome, with each chromosome getting its own file.

    Parameters:
    - gtf_file (str): Path to the GTF file containing gene annotations.
    - genome_fasta (str): Path to the reference genome FASTA file.
    - gene_tx_map (dict, optional): A dictionary mapping gene IDs to transcript IDs. Default is None.
    - output_file (str): Path to the output file where the sequences will be saved. Default is "gene_sequence.parquet".
    - mode (str): Mode of sequence extraction. Options are:
        - 'minmax': Extract sequences with the minimum start and maximum end across all transcripts.
        - 'full': Extract the entire DNA sequence for each gene. Default is 'minmax'.

    Returns:
    - None

    Example usage:
        gtf_file = "/path/to/annotations.gtf"
        genome_fasta = "/path/to/genome.fa"
        gene_sequence_retrieval_workflow(gtf_file, genome_fasta, output_file="gene_sequences.parquet", mode='full')

    Dependencies:
    - extract_genes_from_gtf: Function to extract gene information from a GTF file.
    - extract_gene_sequences: Function to extract DNA sequences for genes from the reference genome based on coordinates.
    - save_sequences_by_chromosome: Function to save sequences by chromosome.

    Notes:
    - The function will print information about the extracted gene sequences and the output file path.
    - The output file will be saved in the specified format (default is 'parquet').
    """    
    from .utils_bio import (
        extract_genes_from_gtf,
        extract_gene_sequences,
        save_sequences_by_chromosome
    )
   
    if mode == 'minmax': 
        print("[info] Extract gene-specific sequences with min start and max end across all transcripts ...")
        return gene_sequence_retrieval_minmax_workflow(gtf_file, genome_fasta, gene_tx_map=gene_tx_map, output_file=output_file)
    else: 
        print("[info] Extract the entire DNA sequence for each gene ...")

    genes_df = extract_genes_from_gtf(gtf_file)
    print("[info] Gene dataframe:")
    print(genes_df.head())  # Display the extracted gene information

    # Extract DNA sequences for genes from the reference genome based on coordinates
    seq_df = extract_gene_sequences(genes_df, genome_fasta, output_format='dataframe', include_columns=None)
    # NOTE: columns(seq_df): 
    #  [  'gene_name',  # Name of the gene
    #     'seqname',    # Chromosome or sequence name
    #     'gene_id',    # Gene identifier
    #     'strand',     # Strand information ('+' or '-')
    #     'start',      # Start position (0-based index)
    #     'end',        # End position
    #     'sequence'    # DNA sequence of the gene
    #  ]
    # NOTE: Negative-strand gene sequences are reverse-complemented

    format = kargs.get('format', 'parquet')
    output_path = output_file  

    # Check if the directory part of the output_path is empty
    output_dir = os.path.dirname(output_path)
    if not output_dir:
        # If output_path is just a file name, assume the current directory as the parent directory
        output_path = os.path.join(os.getcwd(), output_path)
        output_dir = os.getcwd()
    assert os.path.exists(output_dir), "Invalid output path: {}".format(output_path)

    print(f"[i/o] Saving the full DNA sequence for each gene to:\n{output_path}\n")
    print("... cols(seq_df): ", list(seq_df.columns))
    
    # NOTE: Columns: ['transcript_id', 'gene_id', 'gene_name', 'strand', 'seqname', 'start', 'end', 'sequence']
    # save_sequences(seq_df, output_path, format=format)

    chromosomes = [str(i) for i in range(1, 23)] + ['X', 'Y', 'MT']
    save_sequences_by_chromosome(seq_df, output_path, format=format, chromosomes_to_save=chromosomes)  

    print_emphasized("[info] Generatig supplementary output: chromosome sizes ...")
    output_path = os.path.join(output_dir, "chromosome_sizes.tsv")
    
    print_emphasized(f"[i/o] Chromosome sizes saved to {output_path}")
    generate_chromosome_sizes_file(gtf_file, genome_fasta, output_path, verbose=True)


def gene_sequence_retrieval_minmax_workflow(gtf_file, genome_fasta, gene_tx_map=None, output_file="gene_sequence.parquet"): 
    
    from .utils_bio import (
        extract_transcripts_from_gtf,
        extract_gene_sequences_minmax,
        save_sequences_by_chromosome, 
        load_sequences_by_chromosome
    )

    # src_dir = "/path/to/meta-spliceai/data/ensembl"
    # local_dir = "/path/to/meta-spliceai/data/ensembl"
    # genome_annot = os.path.join(src_dir, "Homo_sapiens.GRCh38.112.gtf") 
    # genome_fasta = os.path.join(src_dir, "Homo_sapiens.GRCh38.dna.primary_assembly.fa")  # "/path/to/Homo_sapiens.GRCh38.dna.primary_assembly.fa"

    # print("[info] Extract gene-specific sequences with min start and max end across all transcripts ...")

    # Extract transcripts and their features
    transcripts_df = extract_transcripts_from_gtf(gtf_file, tx_ids=gene_tx_map, ignore_version=True)
    print("> Transcript dataframe:")
    print(transcripts_df.head())  # Display the extracted transcript information

    # additional_columns = ['seqname', 'start', 'end', 'strand', ]
    seq_df = extract_gene_sequences_minmax(transcripts_df, genome_fasta, output_format='dataframe', include_columns=None)
    print("> DNA sequence dataframe:")
    print("... cols(seq_df): ", list(seq_df.columns))
    # A DataFrame with the following minimum columns:
    #     - gene_id: The unique identifier for each gene.
    #     - sequence: The extracted DNA sequence for the gene.
    #     - strand: The strand on which the gene is located ('+' or '-').
    #     - start: The 1-based start coordinate of the gene sequence.
    #     - end: The end coordinate of the gene sequence.

    # Assuming gene_sequences_df is the dataframe with the columns mentioned
    min_start_position = seq_df['start'].min()
    print(f"[info] Minimum start position: {min_start_position}")

    # Test: Determine if the annotation is 1-based or 0-based
    if min_start_position == 0:
        print("The annotation is 0-based.")
    else:
        print("The annotation is 1-based.")

    # Save the gene-specific DNA sequences to a file
    format = 'parquet'
    output_path = output_file  
    assert os.path.exists(os.path.dirname(output_path)), "Invalid output path: {}".format(output_path)

    print(f"[i/o] Saving gene-specific DNA sequences to:\n{output_path}\n")
    print("... cols(seq_df): ", list(seq_df.columns))
    
    # NOTE: Columns: ['transcript_id', 'gene_id', 'gene_name', 'strand', 'seqname', 'start', 'end', 'sequence']
    # save_sequences(seq_df, output_path, format=format)

    chromosomes = [str(i) for i in range(1, 23)] + ['X', 'Y', 'MT']
    save_sequences_by_chromosome(seq_df, output_path, format=format, chromosomes_to_save=chromosomes)    


def extract_splice_sites_workflow(data_prefix, gtf_file, consensus_window=2, **kargs):
    """
    Extract splice sites workflow.

    Parameters:
    - data_prefix (str): The prefix directory for data files.
    - gtf_file (str): The path to the gene annotation GTF file.
    - consensus_window (int): The consensus window size for extracting splice sites (default is 2).
    """
    import gffutils

    assert os.path.exists(gtf_file), f"GTF file not found: {gtf_file}"
    db_file = os.path.join(data_prefix, "annotations.db")
    # NOTE: extract_annotations() will create the database file if it does not exist

    print("Step 1: Build gene annotation database and extract all annotations ...")
    format = kargs.get('output_format', 'tsv')
    output_file = os.path.join(data_prefix, f"annotations_all_transcripts.{format}")
    
    # Check if the output file exists and its size
    if not os.path.exists(output_file) or os.path.getsize(output_file) == 0:
        print(f"[info] Annotation file not found at:\n{output_file}\n")

        # Extract exon, CDS, 5'UTR, and 3'UTR annotations for all transcripts from the DB
        extract_annotations(gtf_file, db_file=db_file, output_file=output_file, sep='\t')
    else:
        print(f"[info] Found the extracted annotation file at:\n{output_file}\n")

    # Load the annotations from the saved file
    # annotations_df = pd.read_csv(output_file, sep='\t')
    annotations_df = pd.read_csv(output_file, sep='\t', low_memory=False, dtype={'chrom': str})
    # NOTE: By default, pd.read_csv() reads the first row of the CSV file as the header; i.e. header=0 by default

    print("(extract_splice_sites_workflow) annotations_df:")
    print(annotations_df.head())

    print("Step 2: Extract splice sites ...")
    db = gffutils.FeatureDB(db_file)
    print(f"[info] Database loaded from {db_file}")

    # format = 'tsv'
    output_file = kargs.get('output_path', os.path.join(data_prefix, f"splice_sites.{format}"))  
    extract_splice_sites_for_all_genes(db, consensus_window=consensus_window, save=True, output_file=output_file)

    return output_file


def extract_chromosome_sizes_from_gtf(gtf_file_path, use_polars=True):
    """
    Extract chromosome sizes from a GTF file by finding the maximum 'end' coordinate for each chromosome.

    Parameters:
    - gtf_file_path (str): Path to the input GTF file.
    - use_polars (bool): If True, use Polars; otherwise, use Pandas.

    Returns:
    - dict: Dictionary with chromosome names as keys and their sizes as values.
    """
    if not use_polars:
        import pandas as pd
        gtf_df = pd.read_csv(
            gtf_file_path, sep='\t', comment='#', header=None,
            names=['seqname', 'source', 'feature', 'start', 'end', 'score', 'strand', 'frame', 'attribute']
        )
        chrom_sizes = gtf_df.groupby('seqname')['end'].max().to_dict()
        return chrom_sizes

    # Load GTF file using Polars
    columns = ['seqname', 'source', 'feature', 'start', 'end', 'score', 'strand', 'frame', 'attribute']
    gtf_df = pl.read_csv(
        gtf_file_path, 
        separator='\t', 
        comment_prefix='#', 
        has_header=False, 
        new_columns=columns, 
        schema_overrides={'seqname': pl.Utf8}
    )

    # Find the maximum 'end' position for each chromosome
    chrom_sizes = gtf_df.group_by('seqname').agg(pl.col('end').max().alias('chrom_size'))

    return {row['seqname']: row['chrom_size'] for row in chrom_sizes.to_dicts()}


def extract_chromosome_sizes_from_fasta(fasta_file_path):
    """
    Extract chromosome sizes from a FASTA file.

    Parameters:
    - fasta_file_path (str): Path to the reference genome FASTA file.

    Returns:
    - dict: Dictionary with chromosome names as keys and their sizes as values.
    """
    from Bio import SeqIO

    chrom_sizes = {}
    for record in SeqIO.parse(fasta_file_path, "fasta"):
        chrom_sizes[record.id] = len(record.seq)
    return chrom_sizes


def generate_chromosome_sizes_file(gtf_file_path, fasta_file_path, output_path, **kargs):
    """
    Generate a chromosome sizes file using GTF and FASTA inputs, with a consistency check.

    Parameters:
    - gtf_file_path (str): Path to the input GTF file.
    - fasta_file_path (str): Path to the reference genome FASTA file.
    - output_path (str): Path to save the chromosome sizes file.
    - verbose (bool): If True, print debugging and consistency check details.

    Returns:
    - None
    """
    verbose = kargs.get('verbose', 1)
    col_chrom = kargs.get('col_chrom', 'chrom')
    # chromosomes = kargs.get('chromosomes', None)

    # Extract sizes from GTF and FASTA
    gtf_sizes = extract_chromosome_sizes_from_gtf(gtf_file_path)
    fasta_sizes = extract_chromosome_sizes_from_fasta(fasta_file_path)

    # Check for consistency between GTF and FASTA sizes
    mismatches = []
    for chrom in gtf_sizes:
        if chrom in fasta_sizes:
            gtf_size = gtf_sizes[chrom]
            fasta_size = fasta_sizes[chrom]
            if gtf_size != fasta_size:
                mismatches.append((chrom, gtf_size, fasta_size))
                if verbose:
                    print(f"[Mismatch] Chromosome {chrom}: GTF size = {gtf_size}, FASTA size = {fasta_size}")
        else:
            if verbose:
                print(f"[Missing in FASTA] Chromosome {chrom} found in GTF but not in FASTA.")

    for chrom in fasta_sizes:
        if chrom not in gtf_sizes and verbose:
            print(f"[Missing in GTF] Chromosome {chrom} found in FASTA but not in GTF.")

    # If mismatches exist, report and raise a warning
    if mismatches and verbose:
        print(f"\n[Warning] Found {len(mismatches)} size mismatches between GTF and FASTA.")

    # Merge sizes, prioritizing FASTA sizes
    merged_sizes = {**gtf_sizes, **fasta_sizes}

    # Write sizes to a file
    # with open(output_path, 'w') as f:
    #     f.write("chromosome\tsize\n")
    #     for chrom, size in merged_sizes.items():
    #         f.write(f"{chrom}\t{size}\n")

    # Create a DataFrame with the merged sizes
    df = pl.DataFrame({
        col_chrom: list(merged_sizes.keys()),
        "size": list(merged_sizes.values())
    })

    # Write the DataFrame to the output file
    df.write_csv(output_path, separator='\t')

    if verbose:
        print(f"\nChromosome sizes file saved to: {output_path}")

    return df


####################################################################################################


def find_overlapping_genes(annotation_file, output_file=None, output_format="dataframe"):
    """
    Identify overlapping genes based on their coordinates and strand in the annotation file.

    Parameters:
    - annotation_file (str): Path to the input annotation file (TSV or CSV).
    - output_file (str, optional): Path to save the resulting overlapping genes file. If None, the results are not saved.
    - output_format (str): Output format, either "dict" or "dataframe" (default: "dataframe").

    Returns:
    - pd.DataFrame or dict: Overlapping genes as a DataFrame or a dictionary depending on the output_format.
    """
    # import pandas as pd

    # Load the annotation file into a DataFrame
    df = pd.read_csv(annotation_file, sep='\t')

    # Filter to only include gene-level annotations
    genes_df = df[df['feature'] == 'gene'].copy()

    # Ensure the DataFrame is sorted by chromosome, strand, and start coordinate
    genes_df.sort_values(by=['chrom', 'strand', 'start'], inplace=True)

    # Find overlapping genes
    overlapping_genes = []
    for chrom in genes_df['chrom'].unique():
        chrom_genes = genes_df[genes_df['chrom'] == chrom]

        for i, row1 in chrom_genes.iterrows():
            # Identify overlaps within the same strand
            overlaps = chrom_genes[
                (chrom_genes['strand'] == row1['strand']) &  # Ensure same strand
                (chrom_genes['start'] <= row1['end']) &  # Start of overlap
                (chrom_genes['end'] >= row1['start']) &  # End of overlap
                (chrom_genes.index != i)  # Exclude the same gene
            ]
            for _, row2 in overlaps.iterrows():
                overlapping_genes.append({
                    'gene_id_1': row1['gene_id'],
                    'gene_id_2': row2['gene_id'],
                    'chrom': row1['chrom'],
                    'start_1': row1['start'],
                    'end_1': row1['end'],
                    'start_2': row2['start'],
                    'end_2': row2['end'],
                    'strand_1': row1['strand'],
                    'strand_2': row2['strand'],
                })

    # Convert results to the requested output format
    if output_format == "dataframe":
        overlaps_df = pd.DataFrame(overlapping_genes)

        # Save to file if output_file is specified
        if output_file:
            overlaps_df.to_csv(output_file, sep='\t', index=False)
            print(f"[info] Overlapping genes saved to {output_file}")

        return overlaps_df

    elif output_format == "dict":
        overlapping_dict = {}
        for overlap in overlapping_genes:
            gene1 = overlap["gene_id_1"]
            gene2 = overlap["gene_id_2"]
            if gene1 not in overlapping_dict:
                overlapping_dict[gene1] = []
            overlapping_dict[gene1].append(gene2)
        return overlapping_dict

    else:
        raise ValueError("Invalid output_format. Choose 'dataframe' or 'dict'.")


def check_conflicting_overlaps(overlapping_genes_df):
    """
    Check if any gene_id has conflicting num_overlaps in the overlapping_genes_df.

    Parameters:
    - overlapping_genes_df (pd.DataFrame): A dataframe with columns ['gene_id', 'num_overlaps'].

    Returns:
    - pd.DataFrame or str: A dataframe with conflicting entries if conflicts are found,
                           otherwise a string indicating no conflicts.
    """
    if isinstance(overlapping_genes_df, pl.DataFrame):
        overlapping_genes_df = overlapping_genes_df.to_pandas()

    # Group by gene_id and count unique num_overlaps for each gene_id
    conflict_df = (
        overlapping_genes_df.groupby("gene_id")["num_overlaps"]
        .nunique()
        .reset_index()
        .rename(columns={"num_overlaps": "unique_counts"})
    )

    # Filter for gene_ids with more than 1 unique num_overlaps
    conflicts = conflict_df[conflict_df["unique_counts"] > 1]

    if not conflicts.empty:
        # Retrieve conflicting entries
        conflicting_gene_ids = conflicts["gene_id"].tolist()
        conflicting_entries = overlapping_genes_df[overlapping_genes_df["gene_id"].isin(conflicting_gene_ids)]
        return conflicting_entries
    else:
        print("No conflicts found. All gene_ids have consistent num_overlaps.")

    return None


def get_overlapping_gene_metadata(
    gtf_file_path=None, 
    filter_valid_splice_sites=True, 
    min_exons=2, 
    output_format='dict', 
    elim_handshake_dups=False, 
    consider_strand=False, 
    **kargs
):
    """
    Identify overlapping genes and their metadata, with optional filtering for valid splice site genes.

    Parameters:
    - gtf_file_path (str): Path to the GTF file. If None, uses Analyzer.gtf_file.
    - filter_valid_splice_sites (bool): If True, filter overlapping genes to include only those with valid splice sites.
    - min_exons (int): Minimum number of exons required for a valid splice site.
    - output_format (str): 'dict' or 'dataframe' for output format.
    - elim_handshake_dups (bool): If True, eliminate handshake duplicates for display purposes.
    - consider_strand (bool): If True, consider overlaps strand-specific.

    Returns:
    - dict or pl.DataFrame: 
        If 'dict', returns a dictionary where keys are gene IDs and values are lists of overlapping gene metadata.
        If 'dataframe', returns a Polars DataFrame with overlapping gene relationships.
    """
    verbose = kargs.get('verbose', 1)
    
    # Use Analyzer's gtf_file if not provided
    if gtf_file_path is None:
        gtf_file_path = Analyzer.gtf_file

    # Extract gene annotations
    gtf_df = extract_genes_from_gtf(gtf_file_path)

    if filter_valid_splice_sites:
        # Extract exon features and compute exon counts
        exon_df = extract_exons_from_gtf(gtf_file_path)
        exon_counts = compute_exon_counts(exon_df)
        valid_genes = filter_valid_splice_site_genes(exon_counts, min_exons=min_exons)

        # Filter GTF genes to include only valid splice site genes
        gtf_df = gtf_df.filter(pl.col("gene_id").is_in(valid_genes["gene_id"]))

    # Initialize data structure for overlapping genes
    overlapping_list = []

    for chrom in tqdm(gtf_df['seqname'].unique(), desc="Processing chromosomes"):
        tqdm.write(f"Currently processing chromosome: {chrom} ...")
        
        chrom_genes = gtf_df.filter(pl.col('seqname') == chrom).sort('start')
        for gene1 in chrom_genes.iter_rows(named=True):
            gene1_id = gene1['gene_id']
            gene1_start, gene1_end = gene1['start'], gene1['end']
            gene1_strand = gene1['strand']

            # Strand consideration
            strand_condition = (pl.col('strand') == gene1_strand) if consider_strand else True

            overlapping = chrom_genes.filter(
                (pl.col('start') <= gene1_end) &
                (pl.col('end') >= gene1_start) &
                strand_condition &
                (pl.col('gene_id') != gene1_id)
            )

            for gene2 in overlapping.iter_rows(named=True):
                # Add entries for both directions (A -> B and B -> A)
                overlapping_list.append({
                    "gene_id_1": gene1_id,
                    "gene_id_2": gene2["gene_id"],
                    "chrom": gene1["seqname"],
                    "start_1": gene1["start"],
                    "end_1": gene1["end"],
                    "start_2": gene2["start"],
                    "end_2": gene2["end"],
                    "strand_1": gene1["strand"],
                    "strand_2": gene2["strand"],
                })

    overlapping_df = pl.DataFrame(overlapping_list)

    if elim_handshake_dups:
        # Eliminate handshake duplicates (A -> B and B -> A reduced to one)
        overlapping_df = overlapping_df.with_columns(
            pl.when(pl.col("gene_id_1") > pl.col("gene_id_2"))
            .then(pl.col("gene_id_2"))
            .otherwise(pl.col("gene_id_1"))
            .alias("sorted_gene_id_1"),
            pl.when(pl.col("gene_id_1") > pl.col("gene_id_2"))
            .then(pl.col("gene_id_1"))
            .otherwise(pl.col("gene_id_2"))
            .alias("sorted_gene_id_2")
        ).drop(["gene_id_1", "gene_id_2"]).rename({"sorted_gene_id_1": "gene_id_1", "sorted_gene_id_2": "gene_id_2"}).unique()

    # Aggregate counts for each gene
    all_overlaps = (
        overlapping_df.melt(id_vars=["chrom", "start_1", "end_1", "start_2", "end_2", "strand_1", "strand_2"], 
                            value_vars=["gene_id_1", "gene_id_2"], 
                            variable_name="gene_role", 
                            value_name="gene_id")
        .group_by("gene_id")
        .agg(pl.len().alias("num_overlaps"))
    )

    # Merge num_overlaps into overlapping_df
    overlapping_df = overlapping_df.join(all_overlaps, left_on="gene_id_1", right_on="gene_id", how="left")

    output_file = kargs.get('output_file', "overlapping_gene_counts.tsv")
    save = kargs.get('save', True)

    if save and output_file:
        # Save the DataFrame to the specified output file
        print(f"[i/o] Saving overlapping gene metadata to {output_file} ...")
        file_extension = output_file.split('.')[-1]
        if file_extension == 'csv':
            overlapping_df.write_csv(output_file)
        elif file_extension == 'tsv':
            overlapping_df.write_csv(output_file, separator='\t')
        else:
            raise ValueError("Unsupported file format. Please use 'csv' or 'tsv'.")

    if output_format == 'dataframe':
        return overlapping_df

    # Prepare dict output
    overlapping_genes_dict = {}
    for row in overlapping_df.iter_rows(named=True):
        if row["gene_id_1"] not in overlapping_genes_dict:
            overlapping_genes_dict[row["gene_id_1"]] = []
        overlapping_genes_dict[row["gene_id_1"]].append({
            "gene_id": row["gene_id_2"],
            "start": row["start_2"],
            "end": row["end_2"],
            "strand": row["strand_2"],
        })

    return overlapping_genes_dict


def count_overlapping_genes(overlapping_metadata):
    """
    Count the number of overlapping genes for each key (gene_id) in overlapping_metadata.

    Parameters:
    - overlapping_metadata (dict): A dictionary where keys are gene IDs, and values are lists of dictionaries with metadata for overlapping genes.

    Returns:
    - dict: A dictionary where keys are gene IDs, and values are the counts of overlapping genes.
    """
    return {gene_id: len(overlaps) for gene_id, overlaps in overlapping_metadata.items()}


def get_overlapping_gene_metadata_v1(gtf_file_path, filter_valid_splice_sites=True, min_exons=2, output_format='dict'):
    """
    Identify overlapping genes and their metadata, with optional filtering for valid splice site genes.

    Parameters:
    - gtf_file_path (str): Path to the GTF file.
    - filter_valid_splice_sites (bool): If True, filter overlapping genes to include only those with valid splice sites.
    - min_exons (int): Minimum number of exons required for a valid splice site.
    - output_format (str): 'dict' or 'dataframe' for output format.

    Returns:
    - dict or pl.DataFrame: 
        If 'dict', returns a dictionary where keys are gene IDs and values are lists of overlapping gene metadata.
        If 'dataframe', returns a Polars DataFrame with overlapping gene relationships.
    """
    # Extract gene annotations
    gtf_df = extract_genes_from_gtf(gtf_file_path)

    if filter_valid_splice_sites:
        # Extract exon features and compute exon counts
        exon_df = extract_exons_from_gtf(gtf_file_path)
        exon_counts = compute_exon_counts(exon_df)
        valid_genes = filter_valid_splice_site_genes(exon_counts, min_exons=min_exons)

        # Filter GTF genes to include only valid splice site genes
        gtf_df = gtf_df.filter(pl.col("gene_id").is_in(valid_genes["gene_id"]))

    # Continue with overlapping gene metadata extraction as before
    overlapping_genes = {}

    for chrom in gtf_df['seqname'].unique():
        chrom_genes = gtf_df.filter(pl.col('seqname') == chrom).sort('start')
        for gene1 in chrom_genes.iter_rows(named=True):
            gene1_id = gene1['gene_id']
            gene1_start, gene1_end = gene1['start'], gene1['end']
            gene1_strand = gene1['strand']

            overlapping = chrom_genes.filter(
                (pl.col('start') <= gene1_end) &
                (pl.col('end') >= gene1_start) &
                (pl.col('strand') == gene1_strand) &
                (pl.col('gene_id') != gene1_id)
            )

            overlapping_metadata = overlapping.select(['gene_id', 'start', 'end', 'strand']).to_dicts()
            overlapping_genes[gene1_id] = overlapping_metadata

    if output_format == 'dataframe':
        df = pl.DataFrame({
            "gene_id": list(overlapping_genes.keys()),
            "overlapping_genes": [
                [gene['gene_id'] for gene in genes] for genes in overlapping_genes.values()
            ],
            "num_overlapping_genes": [len(genes) for genes in overlapping_genes.values()],
            "overlapping_metadata": [genes for genes in overlapping_genes.values()]
        })
        return df.sort("num_overlapping_genes", descending=True)

    return overlapping_genes


def get_overlapping_gene_metadata_v0(gtf_file_path, filter_valid_splice_sites=True, min_exons=2, output_format='dict', **kargs):
    """
    Identify overlapping genes and their metadata, with optional filtering for valid splice site genes.

    Parameters:
    - gtf_file_path (str): Path to the GTF file.
    - filter_valid_splice_sites (bool): If True, filter overlapping genes to include only those with valid splice sites.
    - min_exons (int): Minimum number of exons required for a valid splice site.
    - output_format (str): 'dict' or 'dataframe' for output format.

    Returns:
    - dict or pl.DataFrame: 
        If 'dict', returns a dictionary where keys are gene IDs and values are lists of overlapping gene metadata.
        If 'dataframe', returns a Polars DataFrame with overlapping gene relationships.
    """
    verbose = kargs.get('verbose', 1)

    # Extract gene annotations
    gtf_df = extract_genes_from_gtf(gtf_file_path)

    if filter_valid_splice_sites:
        # Extract exon features and compute exon counts
        exon_df = extract_exons_from_gtf(gtf_file_path)
        exon_counts = compute_exon_counts(exon_df)
        valid_genes = filter_valid_splice_site_genes(exon_counts, min_exons=min_exons)

        # Filter GTF genes to include only valid splice site genes
        gtf_df = gtf_df.filter(pl.col("gene_id").is_in(valid_genes["gene_id"]))

    # Initialize data structure for overlapping genes
    overlapping_list = []

    for chrom in gtf_df['seqname'].unique():
        chrom_genes = gtf_df.filter(pl.col('seqname') == chrom).sort('start')
        for gene1 in chrom_genes.iter_rows(named=True):
            gene1_id = gene1['gene_id']
            gene1_start, gene1_end = gene1['start'], gene1['end']
            gene1_strand = gene1['strand']

            overlapping = chrom_genes.filter(
                (pl.col('start') <= gene1_end) &
                (pl.col('end') >= gene1_start) &
                (pl.col('strand') == gene1_strand) &
                (pl.col('gene_id') != gene1_id)
            )

            for gene2 in overlapping.iter_rows(named=True):
                # Enforce consistent ordering to eliminate handshake duplicates
                gene_id_1 = min(gene1_id, gene2["gene_id"])
                gene_id_2 = max(gene1_id, gene2["gene_id"])
                overlapping_list.append({
                    "gene_id_1": gene_id_1,
                    "gene_id_2": gene_id_2,
                    "chrom": gene1["seqname"],
                    "start_1": gene1["start"],
                    "end_1": gene1["end"],
                    "start_2": gene2["start"],
                    "end_2": gene2["end"],
                    "strand_1": gene1["strand"],
                    "strand_2": gene2["strand"],
                })

    # Create Polars DataFrame
    overlapping_df = pl.DataFrame(overlapping_list).unique(subset=["gene_id_1", "gene_id_2"])

    # Comprehensive overlap count
    all_overlaps = (
        overlapping_df.melt(id_vars=["chrom", "start_1", "end_1", "start_2", "end_2", "strand_1", "strand_2"], 
                            value_vars=["gene_id_1", "gene_id_2"], 
                            variable_name="gene_role",  # column name that will contain the names of the unpivoted columns
                            value_name="gene_id")  # column name that will contain the values of the unpivoted columns
        .group_by("gene_id")
        .agg(pl.len().alias("num_overlaps"))
    )
    # pl.count() is deprecated; use pl.len() instead

    output_file = kargs.get('output_file', "overlapping_gene_counts.tsv")
    save = kargs.get('save', True)

    if output_format == 'dataframe':
        # Add metadata for output
        overlapping_metadata_df = overlapping_df.join(all_overlaps, left_on="gene_id_1", right_on="gene_id", how="left")
        
        # Sort by the number of overlapping genes in descending order
        overlapping_metadata_df = overlapping_metadata_df.sort("num_overlaps", descending=True)

        # Save to file if output_file is specified
        if save and output_file:
            print_emphasized(f"[i/o] Saving overlapping gene metadata to {output_file} ...")

            file_extension = output_file.split('.')[-1]
            if file_extension == 'csv':
                overlapping_metadata_df.write_csv(output_file)
            elif file_extension == 'tsv':
                overlapping_metadata_df.write_csv(output_file, separator='\t')
            else:
                raise ValueError("Unsupported file format. Please use 'csv' or 'tsv'.")

        return overlapping_metadata_df

    # Prepare dict output
    overlapping_genes_dict = {}
    for row in overlapping_df.iter_rows(named=True):
        if row["gene_id_1"] not in overlapping_genes_dict:
            overlapping_genes_dict[row["gene_id_1"]] = []
        overlapping_genes_dict[row["gene_id_1"]].append({
            "gene_id": row["gene_id_2"],
            "start": row["start_2"],
            "end": row["end_2"],
            "strand": row["strand_2"],
        })

    return overlapping_genes_dict


def load_overlapping_gene_metadata(file_path, output_format='dict', verbose=1):
    """
    Load previously saved overlapping gene metadata.

    Parameters:
    - file_path (str): Path to the file containing overlapping gene metadata.
    - output_format (str): Desired format of the returned data ('dict' or 'dataframe').
    - verbose (int): Verbosity level (default=1).

    Returns:
    - dict or pl.DataFrame:
        If 'dict', returns a dictionary where keys are gene IDs and values are lists of overlapping gene metadata.
        If 'dataframe', returns a Polars DataFrame with overlapping gene relationships.
    """
    # Detect file extension
    file_extension = file_path.split('.')[-1]
    if file_extension not in {'csv', 'tsv'}:
        raise ValueError("Unsupported file format. Please use 'csv' or 'tsv'.")

    # Load the file
    if verbose:
        print(f"[info] Loading overlapping gene metadata from {file_path} ...")

    separator = '\t' if file_extension == 'tsv' else ','
    metadata_df = pl.read_csv(
        file_path, 
        separator=separator, 
        schema_overrides={
            "chrom": pl.Utf8,  # Set the chrom column to string type
            # Add other columns and their types if needed
        }  
    )

    # Ensure expected columns are present
    required_columns = ['gene_id_1', 'gene_id_2', 'chrom', 'start_1', 'end_1', 'start_2', 'end_2', 
                        'strand_1', 'strand_2', 'num_overlaps']
    missing_columns = [col for col in required_columns if col not in metadata_df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in input file: {missing_columns}")

    # Return in the requested format
    if output_format == 'dataframe':
        if verbose:
            print("[info] Returning metadata as a Polars DataFrame.")
        return metadata_df

    if verbose:
        print("[info] Converting metadata to dictionary format.")
    
    # Convert to dictionary format
    overlapping_genes_dict = {}

    # Optional progress bar
    rows_iter = metadata_df.iter_rows(named=True)
    if verbose:
        try:
            from tqdm import tqdm  # type: ignore
            rows_iter = tqdm(rows_iter, total=len(metadata_df), desc="Overlap dict", unit="row")
        except ImportError:
            pass

    for row in rows_iter:
        if row["gene_id_1"] not in overlapping_genes_dict:
            overlapping_genes_dict[row["gene_id_1"]] = []
        overlapping_genes_dict[row["gene_id_1"]].append({
            "gene_id": row["gene_id_2"],
            "start": row["start_2"],
            "end": row["end_2"],
            "strand": row["strand_2"],
        })

    return overlapping_genes_dict


def get_or_load_overlapping_gene_metadata(
        gtf_file_path, 
        overlapping_gene_path=None, 
        filter_valid_splice_sites=True, 
        min_exons=2, 
        elim_handshake_dups=False, 
        consider_strand=True, 
        output_format='dict', **kargs):
    """
    Select between get_overlapping_gene_metadata() or load_overlapping_gene_metadata() depending on if the overlapping gene file exists.

    Parameters:
    - overlapping_gene_path (str): Path to the file containing overlapping gene metadata.
    - gtf_file_path (str): Path to the GTF file.
    - filter_valid_splice_sites (bool): If True, filter overlapping genes to include only those with valid splice sites.
    - min_exons (int): Minimum number of exons required for a valid splice site.
    - output_format (str): Desired format of the returned data ('dict' or 'dataframe').
    - **kargs: Additional keyword arguments for configuration.

    Returns:
    - dict or pl.DataFrame: Overlapping gene metadata in the specified format.
    """
    verbose = kargs.get("verbose", 1)
    if overlapping_gene_path is None: 
        input_dir = kargs.pop("input_dir", os.getcwd())
        overlapping_gene_path = os.path.join(input_dir, "overlapping_gene_counts.tsv")

    if os.path.exists(overlapping_gene_path):
        # Load the overlapping gene metadata from the file
        return load_overlapping_gene_metadata(overlapping_gene_path, output_format=output_format, verbose=verbose)
    else:
        # Compute the overlapping gene metadata
        return get_overlapping_gene_metadata(
            gtf_file_path,
            filter_valid_splice_sites=filter_valid_splice_sites,
            min_exons=min_exons,
            elim_handshake_dups=elim_handshake_dups, 
            consider_strand=consider_strand, 
            output_format=output_format,
            output_file=overlapping_gene_path,
            save=True,
        )


def get_overlapping_gene_relationships(gtf_file_path, output_format='dict'):
    """
    Identify overlapping genes and their metadata (coordinates, strand) from a GTF file.

    Parameters:
    - gtf_file_path (str): Path to the GTF file.
    - output_format (str): 'dict' or 'dataframe' for output format.

    Returns:
    - dict or pl.DataFrame: 
        If 'dict', returns a dictionary where keys are gene IDs and values are lists of overlapping gene metadata.
        If 'dataframe', returns a Polars DataFrame with overlapping gene relationships.
    """
    # import polars as pl
    from .utils_bio import extract_genes_from_gtf

    # Extract gene annotations
    gtf_df = extract_genes_from_gtf(gtf_file_path)

    # Dictionary to store overlapping genes with metadata
    overlapping_genes_metadata = {}
    overlapping_list = []

    # Process each chromosome separately
    for chrom in gtf_df['seqname'].unique():
        chrom_genes = gtf_df.filter(pl.col('seqname') == chrom).sort('start')
        
        # Iterate through genes on the chromosome
        for gene1 in chrom_genes.iter_rows(named=True):
            gene1_id = gene1['gene_id']
            gene1_start, gene1_end = gene1['start'], gene1['end']
            gene1_strand = gene1['strand']

            # Find overlapping genes on the same strand
            overlapping = chrom_genes.filter(
                (pl.col('start') <= gene1_end) & 
                (pl.col('end') >= gene1_start) &
                (pl.col('strand') == gene1_strand) & 
                (pl.col('gene_id') != gene1_id)
            )

            # Collect metadata for overlapping genes as a list of dictionaries
            overlapping_metadata = overlapping.select(['gene_id', 'start', 'end', 'strand']).to_dicts()
            overlapping_genes_metadata[gene1_id] = overlapping_metadata

            # Add entries to overlapping_list for DataFrame output
            for overlap in overlapping_metadata:
                overlapping_list.append({
                    'gene_id_1': gene1_id,
                    'start_1': gene1_start,
                    'end_1': gene1_end,
                    'strand_1': gene1_strand,
                    'gene_id_2': overlap['gene_id'],
                    'start_2': overlap['start'],
                    'end_2': overlap['end'],
                    'strand_2': overlap['strand']
                })

    # Return the requested output format
    if output_format == 'dataframe':
        return pl.DataFrame(overlapping_list)
    return overlapping_genes_metadata


def sort_genes_by_overlap_count(metadata_dict):
    """
    Sort genes by the number of overlapping genes they have.

    Parameters:
    - metadata_dict (dict): Dictionary where keys are gene IDs and values are lists of overlapping gene metadata.

    Returns:
    - list: List of tuples (gene_id, overlap_count) sorted by overlap count in descending order.
    """
    sorted_genes = sorted(
        [(gene_id, len(overlaps)) for gene_id, overlaps in metadata_dict.items()],
        key=lambda x: x[1], reverse=True
    )
    return sorted_genes


def sort_genes_by_overlap_count_df(metadata_df):
    """
    Sort genes by the number of overlapping genes they have using DataFrame format.

    Parameters:
    - metadata_df (pl.DataFrame): DataFrame with overlapping gene relationships.

    Returns:
    - pl.DataFrame: DataFrame with gene IDs and their overlap counts, sorted by count in descending order.
    """
    sorted_df = (
        metadata_df.lazy()
        .group_by("gene_id_1")
        .agg(pl.count("gene_id_2").alias("overlap_count"))
        .sort("overlap_count", descending=True)
        .collect()
    )
    return sorted_df


def get_overlapping_gene_set(gtf_file_path, output_format="dict", include_gene_names=False):
    """
    Identify overlapping genes from a GTF file, optionally including gene names and ensuring strand-specific overlaps.

    Parameters:
    - gtf_file_path (str): Path to the GTF file.
    - output_format (str): Output format, either "dict" or "dataframe" (default: "dict").
    - include_gene_names (bool): If True, include gene names in the output (default: False).

    Returns:
    - dict or pl.DataFrame: 
        If "dict", a dictionary where keys are gene IDs and values are sets of overlapping gene IDs (or names).
        If "dataframe", a Polars DataFrame with columns for overlapping gene pairs and their metadata.
    """
    # import polars as pl
    from .utils_bio import extract_genes_from_gtf

    # Extract gene annotations
    gtf_df = extract_genes_from_gtf(gtf_file_path)

    # Ensure gene_name column is included if requested
    if include_gene_names:
        if "gene_name" not in gtf_df.columns:
            raise ValueError("GTF file does not include gene_name information. Cannot include gene names.")

    # Prepare containers for results
    overlapping_genes = {}
    overlapping_list = []

    # Process each chromosome separately
    for chrom in gtf_df['seqname'].unique():
        chrom_genes = gtf_df.filter(pl.col('seqname') == chrom).sort('start')

        # Iterate through each gene on the chromosome
        for gene1 in chrom_genes.iter_rows(named=True):
            gene1_id = gene1['gene_id']
            gene1_start, gene1_end = gene1['start'], gene1['end']
            gene1_strand = gene1['strand']
            gene1_name = gene1['gene_name'] if include_gene_names else None

            # Find overlapping genes on the same strand
            overlaps = chrom_genes.filter(
                (pl.col('start') <= gene1_end) & 
                (pl.col('end') >= gene1_start) & 
                (pl.col('strand') == gene1_strand) & 
                (pl.col('gene_id') != gene1_id)
            )

            # Process overlaps
            for gene2 in overlaps.iter_rows(named=True):
                gene2_id = gene2['gene_id']
                gene2_name = gene2['gene_name'] if include_gene_names else None

                # Add to the list for dataframe output
                overlap_entry = {
                    "gene_id_1": gene1_id,
                    "gene_id_2": gene2_id,
                    "chrom": gene1['seqname'],
                    "strand": gene1['strand'],
                }
                if include_gene_names:
                    overlap_entry.update({
                        "gene_name_1": gene1_name,
                        "gene_name_2": gene2_name
                    })
                overlapping_list.append(overlap_entry)

                # Add to the dictionary for dict output
                if gene1_id not in overlapping_genes:
                    overlapping_genes[gene1_id] = set()
                overlapping_genes[gene1_id].add(gene2_name if include_gene_names else gene2_id)

    # Return based on the output format
    if output_format == "dict":
        return overlapping_genes
    elif output_format == "dataframe":
        return pl.DataFrame(overlapping_list)
    else:
        raise ValueError("Invalid output_format. Choose 'dict' or 'dataframe'.")


def demo_get_overlapping_gene_set(gtf_file_path=None, include_metadata=False): 
    """
    Demo function to verify the correctness of get_overlapping_gene_set.

    Parameters:
    - gtf_file_path (str): Path to the GTF file.

    Returns:
    - None
    """
    if gtf_file_path is None: 
        # Use Analyzer's gtf_file which is now dynamically configured
        gtf_file_path = Analyzer.gtf_file

    overlapping_genes_dict = get_overlapping_gene_set(
        gtf_file_path=gtf_file_path,
        output_format="dict",
        include_gene_names=False
    )
    print(overlapping_genes_dict["ENSG00000224078"])

    print_emphasized("[info] Output with gene names ...")
    overlapping_genes_with_names = get_overlapping_gene_set(
        gtf_file_path=gtf_file_path,
        output_format="dict",
        include_gene_names=True
    )
    print(overlapping_genes_with_names["ENSG00000224078"])

    print_emphasized("[info] Output as Dataframe") 
    overlapping_genes_df = get_overlapping_gene_set(
        gtf_file_path=gtf_file_path,
        output_format="dataframe",
        include_gene_names=True
    )
    print(overlapping_genes_df)

    return


######################################
# Feature extraction utilities


def extract_gene_features_from_gtf(gtf_file_path, **kargs):
    """
    Extract gene-level features from a GTF file.
    
    Automatically detects and handles different GTF formats:
    - Standard GTF (Ensembl): Has explicit 'gene' features
    - MANE GTF: No 'gene' features, aggregates from transcripts

    Parameters:
    - gtf_file_path (str): Path to the GTF file.
    - verbose (int): Verbosity level.

    Returns:
    - pl.DataFrame: A Polars DataFrame containing gene-level features.
    """
    verbose = kargs.get('verbose', 1)
    save = kargs.get('save', True)
    output_file = kargs.get('output_file', 'gene_features.csv')

    # Define columns for the GTF file
    columns = ['seqname', 'source', 'feature', 'start', 'end', 'score', 'strand', 'frame', 'attribute']

    # Load GTF file
    gtf_df = pl.read_csv(gtf_file_path, 
                         separator='\t', 
                         comment_prefix='#', 
                         has_header=False, 
                         new_columns=columns, 
                         schema_overrides={'seqname': pl.Utf8})

    # Check if this is a MANE-style GTF (no 'gene' features)
    has_gene_features = (gtf_df['feature'] == 'gene').any()
    
    if not has_gene_features:
        # MANE format - aggregate from transcripts
        if verbose:
            print("[info] Detected MANE-style GTF (no 'gene' features), aggregating from transcripts...")
        
        from meta_spliceai.splice_engine.extract_gene_features_mane import extract_gene_features_from_mane_gtf
        
        gene_df = extract_gene_features_from_mane_gtf(
            gtf_file_path,
            output_file=output_file if save else None,
            verbose=verbose
        )
        
        # Convert to expected format
        use_cols = kargs.get('use_cols', None)
        if use_cols is not None:
            gene_df = gene_df.select(use_cols)
        
        return gene_df
    
    # Standard GTF format - filter for gene features
    gene_df = gtf_df.filter(pl.col('feature') == 'gene')

    # Parse attributes to extract gene_id, gene_name, and gene_type
    gene_df = gene_df.with_columns([
        pl.col("attribute").str.extract(r'gene_id "([^"]+)"').alias("gene_id"),
        pl.col("attribute").str.extract(r'gene_name "([^"]+)"').alias("gene_name"),
        pl.col("attribute").str.extract(r'gene_biotype "([^"]+)"').alias("gene_type")
    ])

    # Compute gene length
    gene_df = gene_df.with_columns([
        (pl.col("end") - pl.col("start") + 1).alias("gene_length"),
        pl.col("strand").alias("strand"),
        pl.col("seqname").alias("chrom")
    ])

    # Select relevant columns
    use_cols = kargs.get('use_cols', None)
    # ['start', 'end', 'score', 'strand', 'gene_id', 'gene_name', 'gene_type', 'gene_length', 'chrom']

    if use_cols is not None: 
        gene_df = gene_df.select(use_cols)

    if verbose:
        print(f"[info] Extracted {gene_df.shape[0]} gene-level features.")

    # Save the DataFrame 
    if save:
        print_emphasized(f"[i/o] Saving gene-level features to a file:\n{output_file}\n ...")

        sep = '\t' if output_file.endswith('.tsv') else ','
        gene_df.write_csv(output_file, separator=sep)

    return gene_df


def extract_transcript_features_from_gtf(gtf_file_path, **kargs):
    """
    Extract transcript-level features from a GTF file.

    Parameters:
    - gtf_file_path (str): Path to the GTF file.
    - verbose (int): Verbosity level.

    Returns:
    - pl.DataFrame: A Polars DataFrame containing transcript-level features.
    """
    verbose = kargs.get('verbose', 1)
    save = kargs.get('save', True)
    output_file = kargs.get('output_file', 'transcript_features.csv')

    # Define columns for the GTF file
    columns = ['seqname', 'source', 'feature', 'start', 'end', 'score', 'strand', 'frame', 'attribute']

    # Load GTF file
    gtf_df = pl.read_csv(gtf_file_path, 
                         separator='\t', 
                         comment_prefix='#', 
                         has_header=False, 
                         new_columns=columns, 
                         schema_overrides={'seqname': pl.Utf8})

    # Filter for transcript features
    transcript_df = gtf_df.filter(pl.col('feature') == 'transcript')

    # Parse attributes to extract transcript_id, gene_id, transcript_biotype, and transcript_name
    transcript_df = transcript_df.with_columns([
        pl.col("attribute").str.extract(r'transcript_id "([^"]+)"').alias("transcript_id"),
        pl.col("attribute").str.extract(r'gene_id "([^"]+)"').alias("gene_id"),
        pl.col("attribute").str.extract(r'transcript_biotype "([^"]+)"').alias("transcript_type"),
        pl.col("attribute").str.extract(r'transcript_name "([^"]+)"').alias("transcript_name")
    ])

    # Compute transcript length
    transcript_df = transcript_df.with_columns([
        (pl.col("end") - pl.col("start") + 1).alias("transcript_length"),
        pl.col("strand").alias("strand"),
        pl.col("seqname").alias("chrom")
    ])

    # Select relevant columns
    use_cols = kargs.get('use_cols', [
        'chrom', 'start', 'end', 'strand', 'transcript_id', 
        'transcript_name', 'transcript_type', 'transcript_length', 'gene_id'
    ])
    if use_cols:
        transcript_df = transcript_df.select(use_cols)

    # Print extracted information
    if verbose:
        print(f"[info] Extracted {transcript_df.shape[0]} transcript-level features.")

    # Save the DataFrame
    if save:
        print(f"[i/o] Saving transcript-level features to a file:\n{output_file}\n...")
        sep = '\t' if output_file.endswith('.tsv') else ','
        transcript_df.write_csv(output_file, separator=sep)

    return transcript_df


def summarize_transcript_features(gtf_file_path, **kargs):
    """
    Summarize transcript-level features and aggregate them at the gene level.

    Parameters:
    - gtf_file_path (str): Path to the GTF file.
    - verbose (int): Verbosity level.

    Returns:
    - pl.DataFrame: A Polars DataFrame containing gene-level summaries of transcript features.
    """
    verbose = kargs.get('verbose', 1)
    save = kargs.get('save', True)
    output_file = kargs.get('output_file', 'transcript_features.csv')

    # Define columns for the GTF file
    columns = ['seqname', 'source', 'feature', 'start', 'end', 'score', 'strand', 'frame', 'attribute']

    # Load GTF file
    gtf_df = pl.read_csv(gtf_file_path, 
                         separator='\t', 
                         comment_prefix='#', 
                         has_header=False, 
                         new_columns=columns, 
                         schema_overrides={'seqname': pl.Utf8})

    # Filter for transcript features
    transcript_df = gtf_df.filter(pl.col('feature') == 'transcript')

    # Parse attributes to extract gene_id and transcript_biotype
    transcript_df = transcript_df.with_columns([
        pl.col("attribute").str.extract(r'gene_id "([^"]+)"').alias("gene_id"),
        pl.col("attribute").str.extract(r'transcript_biotype "([^"]+)"').alias("transcript_type"),
    ])

    # Count transcripts per type
    transcript_summary = (
        transcript_df.group_by(["gene_id", "transcript_type"])
        .count()
        .pivot(index="gene_id", columns="transcript_type", values="count")
        .fill_null(0)
    )

    # Add other features here if needed
    if verbose:
        print(f"[info] Summarized transcript-level features for {transcript_summary.shape[0]} genes.")

    # Save the DataFrame
    if save:
        print_emphasized(f"[i/o] Saving gene-level transcript features to a file:\n{output_file}\n ...")

        sep = '\t' if output_file.endswith('.tsv') else ','
        transcript_summary.write_csv(output_file, separator=sep)

    return transcript_summary


def extract_exon_features_from_gtf(gtf_file_path, **kargs):
    """
    Extract exon-level features with caching support.
    
    Parameters:
    - gtf_file_path (str): Path to the GTF file.
    - verbose (int): Verbosity level.
    - save (bool): Whether to save the output DataFrame to a file.
    - output_file (str): Output file path for saving result (if save is True).
    - cache_file (str): Path to cache file (if None, no caching is used).
    - use_cache (bool): Whether to use cached file if it exists.
    - to_pandas (bool): Whether to return a pandas DataFrame.
    - include_extra_features (bool): Whether to include additional exon features like length.
    
    Returns:
    - Union[pl.DataFrame, pd.DataFrame]: DataFrame containing exon features.
    """
    # Parse parameters
    verbose = kargs.get('verbose', 1)
    save = kargs.get('save', True)
    output_file = kargs.get('output_file', 'exon_features.csv')
    cache_file = kargs.get('cache_file', None)
    use_cache = kargs.get('use_cache', True)
    to_pandas = kargs.get('to_pandas', False)
    include_extra_features = kargs.get('include_extra_features', True)
    clean_output = kargs.get('clean_output', True)
    
    # Use cache if available and requested
    if use_cache and cache_file and os.path.exists(cache_file):
        if verbose:
            print(f"[i/o] Loading exon dataframe from cache: {cache_file}")
        
        # Use smart_read_csv for robust file loading
        from meta_spliceai.splice_engine.utils_df import smart_read_csv
        exon_df = smart_read_csv(cache_file, use_polars=True)
    else:
        # Define columns for the GTF file
        columns = ['seqname', 'source', 'feature', 'start', 'end', 'score', 'strand', 'frame', 'attribute']

        # Load GTF file
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

        # Parse attributes to extract gene_id, transcript_id, transcript_biotype, and exon_number
        exon_df = exon_df.with_columns([
            pl.col("attribute").str.extract(r'gene_id "([^"]+)"').alias("gene_id"),
            pl.col("attribute").str.extract(r'transcript_id "([^"]+)"').alias("transcript_id"),
            pl.col("attribute").str.extract(r'gene_name "([^"]+)"').alias("gene_name"),
        ])

        # Handle missing gene_name by substituting with gene_id
        exon_df = exon_df.with_columns(
            pl.when(pl.col("gene_name").is_null())
            .then(pl.col("gene_id"))
            .otherwise(pl.col("gene_name"))
            .alias("gene_name")
        )

        # Add extra features if requested
        if include_extra_features:
            exon_df = exon_df.with_columns([
                pl.col("attribute").str.extract(r'transcript_biotype "([^"]+)"').alias("transcript_type"),
                pl.col("attribute").str.extract(r'exon_number "([^"]+)"').cast(pl.Int64).alias("exon_number"),
                (pl.col("end") - pl.col("start") + 1).alias("exon_length"),
                pl.col("seqname").alias("chrom"),  # Add chromosome
                pl.col("strand").alias("strand")
            ])

        # Print the total number of exons
        if verbose:
            print(f"[info] Extracted {exon_df.shape[0]} exons with {exon_df.shape[1]} features.")

        # Clean output if requested (before saving or returning)
        if clean_output:
            # Select only the essential columns, similar to extract_exons_from_gtf in utils_bio.py
            essential_columns = ["gene_id", "transcript_id", "gene_name"]
            
            if include_extra_features:
                essential_columns.extend([
                    "transcript_type", "exon_number", "exon_length", 
                    "chrom", "seqname", "start", "end", "strand"
                ])
            else:
                essential_columns.extend(["seqname", "start", "end", "strand"])
                
            # Filter to only keep essential columns
            exon_df = exon_df.select([col for col in essential_columns if col in exon_df.columns])

        # Save to cache if requested
        if cache_file and save:
            if verbose:
                print(f"[i/o] Saving exon dataframe to cache: {cache_file}")
            
            separator = '\t' if cache_file.endswith('.tsv') else ','
            exon_df.write_csv(cache_file, separator=separator)

    # Save to output file if different from cache
    if save and output_file and (not cache_file or output_file != cache_file):
        if verbose:
            print(f"[i/o] Saving exon-level features to file: {output_file}")
            
        separator = '\t' if output_file.endswith('.tsv') else ','
        exon_df.write_csv(output_file, separator=separator)

    # print(f"[info] dtype(exon_df): {type(exon_df)}, to_pandas? {to_pandas}")

    # Convert to pandas if requested
    if to_pandas:
        return exon_df.to_pandas()

    return exon_df


def summarize_exon_features_at_gene_level(gtf_file_path, **kargs):
    """
    Summarize exon-level features and aggregate them at the gene level.

    Parameters:
    - gtf_file_path (str): Path to the GTF file.
    - verbose (int): Verbosity level.

    Returns:
    - pl.DataFrame: A Polars DataFrame containing gene-level summaries of exon features.
    """
    verbose = kargs.get('verbose', 1)
    save = kargs.get('save', True)
    output_file = kargs.get('output_file', 'transcript_features.csv')

    # Define columns for the GTF file
    columns = ['seqname', 'source', 'feature', 'start', 'end', 'score', 'strand', 'frame', 'attribute']

    # Load GTF file
    gtf_df = pl.read_csv(gtf_file_path, 
                         separator='\t', 
                         comment_prefix='#', 
                         has_header=False, 
                         new_columns=columns, 
                         schema_overrides={'seqname': pl.Utf8})

    # Filter for exon features
    exon_df = gtf_df.filter(pl.col('feature') == 'exon')

    # Parse attributes to extract gene_id
    exon_df = exon_df.with_columns([
        pl.col("attribute").str.extract(r'gene_id "([^"]+)"').alias("gene_id")
    ])

    # Compute exon length
    exon_df = exon_df.with_columns([
        (pl.col("end") - pl.col("start") + 1).alias("exon_length")
    ])

    # Summarize exon statistics per gene
    exon_summary = (
        exon_df.group_by("gene_id")
        .agg([
            pl.count("exon_length").alias("num_exons"),
            pl.mean("exon_length").alias("avg_exon_length"),
            pl.median("exon_length").alias("median_exon_length")
        ])
    )

    if verbose:
        print(f"[info] Summarized exon-level features for {exon_summary.shape[0]} genes.")

    # Save the DataFrame
    if save:
        print_emphasized(f"[i/o] Saving GTF-derived exon-specific feature set to a file:\n{output_file}\n ...")

        sep = '\t' if output_file.endswith('.tsv') else ','
        exon_summary.write_csv(output_file, separator=sep)

    return exon_summary


def summarize_exon_features_at_transcript_level(gtf_file_path, **kargs):
    """
    Summarize exon-level features and aggregate them at the transcript level.

    Parameters:
    - gtf_file_path (str): Path to the GTF file.
    - verbose (int): Verbosity level.
    - save (bool): Whether to save the output DataFrame to a file (default: True).
    - output_file (str): Output file path (default: 'transcript_exon_features.csv').

    Returns:
    - pl.DataFrame: A Polars DataFrame containing transcript-level summaries of exon features.
    """
    verbose = kargs.get('verbose', 1)
    save = kargs.get('save', True)
    output_file = kargs.get('output_file', 'transcript_exon_features.csv')

    # Define columns for the GTF file
    columns = ['seqname', 'source', 'feature', 'start', 'end', 'score', 'strand', 'frame', 'attribute']

    # Load GTF file
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

    # Parse attributes to extract transcript_id, transcript_type, and gene_id
    exon_df = exon_df.with_columns([
        pl.col("attribute").str.extract(r'gene_id "([^"]+)"').alias("gene_id"),
        pl.col("attribute").str.extract(r'transcript_id "([^"]+)"').alias("transcript_id"),
        pl.col("attribute").str.extract(r'transcript_biotype "([^"]+)"').alias("transcript_type"),
    ])

    # Compute exon length
    exon_df = exon_df.with_columns([
        (pl.col("end") - pl.col("start") + 1).alias("exon_length")
    ])

    # Summarize exon statistics per transcript
    transcript_exon_summary = (
        exon_df.group_by(["transcript_id", "transcript_type", "gene_id"])
        .agg([
            pl.count("exon_length").alias("num_exons"),
            pl.mean("exon_length").alias("avg_exon_length"),
            pl.median("exon_length").alias("median_exon_length"),
            pl.sum("exon_length").alias("total_exon_length")
        ])
    )

    if verbose:
        print(f"[info] Summarized exon-level features for {transcript_exon_summary.shape[0]} transcripts.")

    # Save the DataFrame
    if save:
        print(f"[i/o] Saving transcript-level exon features to a file:\n{output_file} ...")
        sep = '\t' if output_file.endswith('.tsv') else ','
        transcript_exon_summary.write_csv(output_file, separator=sep)

    return transcript_exon_summary


# ---------------------------------------------------------------------------
#  Streaming implementation to reduce memory usage (≥ 2025-06-17)
# ---------------------------------------------------------------------------

def _summarize_exon_features_at_transcript_level_streaming(
    gtf_file_path: str,
    *,
    verbose: int = 1,
    save: bool = True,
    output_file: str = "transcript_exon_features.csv",
) -> pl.DataFrame:
    """Stream GTF with Polars scan_csv and aggregate exon statistics per transcript.

    This approach keeps memory proportional to the number of *transcripts* (the
    aggregation keys) instead of the total number of *exons*, eliminating OOM
    conditions on machines with limited RAM.

    Memo
    ----
    Polars reads the GTF in chunks, filters to exons, updates the per-transcript aggregates 
    for that chunk, discards the chunk, and proceeds to the next.
    When the scan is finished Polars assembles the final aggregation result 
    (one row per transcript) in memory and returns it as a `transcript_exon_summary` DataFrame.
    """
    columns = [
        "seqname",
        "source",
        "feature",
        "start",
        "end",
        "score",
        "strand",
        "frame",
        "attribute",
    ]

    if verbose:
        print("[info] Streaming GTF to summarise exon features …")

    scan = pl.scan_csv(
        gtf_file_path,
        separator="\t",
        comment_prefix="#",
        has_header=False,
        new_columns=columns,
        schema_overrides={"seqname": pl.Utf8},
    )

    exons = (
        scan.filter(pl.col("feature") == "exon")
        .with_columns([
            pl.col("attribute").str.extract(r'gene_id "([^"]+)"').alias("gene_id"),
            pl.col("attribute").str.extract(r'transcript_id "([^"]+)"').alias("transcript_id"),
            pl.col("attribute").str.extract(r'transcript_biotype "([^"]+)"').alias("transcript_type"),
            (pl.col("end").cast(pl.Int64) - pl.col("start").cast(pl.Int64) + 1).alias("exon_length"),
        ])
    )

    summary_lazy = exons.group_by(["transcript_id", "transcript_type", "gene_id"]).agg([
        pl.count("exon_length").alias("num_exons"),
        pl.mean("exon_length").alias("avg_exon_length"),
        pl.median("exon_length").alias("median_exon_length"),
        pl.sum("exon_length").alias("total_exon_length"),
    ])

    transcript_exon_summary = summary_lazy.collect(streaming=True)

    if verbose:
        print(
            f"[info] Summarised exon features for {transcript_exon_summary.height} transcripts (streaming)."
        )

    if save:
        sep = "\t" if output_file.endswith(".tsv") else ","
        transcript_exon_summary.write_csv(output_file, separator=sep)

    return transcript_exon_summary

# Expose streaming version under the original public name
summarize_exon_features_at_transcript_level = _summarize_exon_features_at_transcript_level_streaming


def join_gene_features_v0(*feature_dataframes):
    """
    Join multiple gene-level feature DataFrames into one.

    Parameters:
    - feature_dataframes: Variable number of Polars DataFrames to join.

    Returns:
    - pl.DataFrame: A Polars DataFrame containing all joined features.
    """
    if not feature_dataframes:
        raise ValueError("No feature dataframes provided for joining.")

    result_df = feature_dataframes[0]
    for df in feature_dataframes[1:]:
        result_df = result_df.join(df, on="gene_id", how="left")

    return result_df


def join_gene_features(*feature_dataframes):
    """
    Join multiple gene-level feature DataFrames into one.

    Parameters:
    - feature_dataframes: Variable number of DataFrames (Pandas or Polars) to join.

    Returns:
    - pl.DataFrame: A Polars DataFrame containing all joined features.
    """
    if not feature_dataframes:
        raise ValueError("No feature dataframes provided for joining.")

    # Convert all input DataFrames to Polars DataFrames
    polars_dfs = []
    for df in feature_dataframes:
        if isinstance(df, pd.DataFrame):
            polars_dfs.append(pl.from_pandas(df))
        elif isinstance(df, pl.DataFrame):
            polars_dfs.append(df)
        else:
            raise TypeError("All input dataframes must be either Pandas or Polars DataFrames.")

    # Join the DataFrames on "gene_id"
    result_df = polars_dfs[0]
    for df in polars_dfs[1:]:
        result_df = result_df.join(df, on="gene_id", how="left")

    return result_df


def extract_gene_features_from_performance_profile(**kargs):
    """
    Extract gene features from a performance profile.

    Parameters:
    - eval_dir (str): Directory where the evaluation file is located.
    - separator (str): Separator used in the evaluation file (default is '\t').

    Returns:
    - pd.DataFrame: A DataFrame containing 'gene_id', 'n_splice_sites', and 'splice_type'.
    """
    from meta_spliceai.splice_engine.model_evaluator import ModelEvaluationFileHandler
    # import pandas as pd

    eval_dir = kargs.get('eval_dir', FeatureAnalyzer.eval_dir)
    separator = kargs.get('separator', kargs.get('sep', '\t'))

    # Initialize ModelEvaluationFileHandler
    mefd = ModelEvaluationFileHandler(eval_dir, separator=separator)

    # Load the performance dataframe
    performance_df = mefd.load_performance_df(aggregated=True)

    # Ensure the DataFrame is in Pandas format for compatibility
    if isinstance(performance_df, pl.DataFrame):
        performance_df = performance_df.to_pandas()

    # Extract relevant columns
    extracted_features = performance_df[['gene_id', 'n_splice_sites', 'splice_type']].copy()

    # Ensure consistency in column types
    extracted_features['n_splice_sites'] = extracted_features['n_splice_sites'].fillna(0).astype(int)
    extracted_features['splice_type'] = extracted_features['splice_type'].fillna('unknown')

    return extracted_features

# --------------------------------


def compute_splice_site_distances(
    splice_sites_df, transcript_features_df, match_col='position'
):
    """
    Compute distances of splice sites to the transcript start and end positions.

    Parameters:
    - splice_sites_df (pd.DataFrame): DataFrame containing splice site annotations.
      Required columns: ['gene_id', 'transcript_id', match_col, 'splice_type'].
    - transcript_features_df (pd.DataFrame): DataFrame containing transcript features.
      Required columns: ['gene_id', 'transcript_id', 'start', 'end'].
    - match_col (str): Column in `splice_sites_df` representing the splice site position.

    Returns:
    - pd.DataFrame: Updated splice_sites_df with added columns:
      ['distance_to_start', 'distance_to_end'].
    """
    # Ensure required columns are present
    required_splice_cols = ['gene_id', 'transcript_id', match_col, 'splice_type']
    required_transcript_cols = ['gene_id', 'transcript_id', 'start', 'end']

    if isinstance(splice_sites_df, pl.DataFrame):
        splice_sites_df = splice_sites_df.to_pandas()
    if isinstance(transcript_features_df, pl.DataFrame):
        transcript_features_df = transcript_features_df.to_pandas()
    
    for col in required_splice_cols:
        if col not in splice_sites_df.columns:
            raise ValueError(f"Column '{col}' is missing from splice_sites_df.")
    
    for col in required_transcript_cols:
        if col not in transcript_features_df.columns:
            raise ValueError(f"Column '{col}' is missing from transcript_features_df.")

    # Drop duplicate columns (with different meanings)
    splice_sites_df = splice_sites_df.drop(columns=['start', 'end'])

    id_columns = ['gene_id', 'transcript_id']
    
    # Merge splice sites with transcript features
    merged_df = splice_sites_df.merge(
        transcript_features_df,
        on=['gene_id', 'transcript_id'],
        how='left'
    )
    # print("[test] Merged DataFrame:")
    # print("colmns(splice_sites): ", list(splice_sites_df.columns))
    # # NOTE: ['chrom', 'start', 'end', 'position', 'strand', 'splice_type', 'gene_id', 'transcript_id']
    # print("columns(transcript):", list(transcript_features_df.columns))
    # # NOTE: ['chrom', 'start', 'end', 'strand', 'transcript_id', 'transcript_name', 'transcript_type', 'transcript_length', 'gene_id']
    # print("=> columns:", list(merged_df.columns))

    # Compute distances to transcript boundaries
    merged_df['distance_to_start'] = abs(merged_df[match_col] - merged_df['start'])
    merged_df['distance_to_end'] = abs(merged_df[match_col] - merged_df['end'])

    columns_to_keep = id_columns + [match_col, 'splice_type', 'distance_to_start', 'distance_to_end']

    merged_df = merged_df[columns_to_keep]
    print_emphasized(f"[compute_splice_site_distances] Keeping columns: {list(merged_df.columns)}")

    return merged_df


def compute_distances_to_transcript_boundaries(df, match_col='position', **kwargs):
    """
    Compute distances between splice sites and transcript or gene boundaries.

    Parameters:
    - df (pd.DataFrame): Input dataframe containing splice site metadata and coordinates.
      Required columns: ['gene_id', 'transcript_id', match_col, 'splice_type', 'tx_start', 'tx_end'].
      If transcript boundaries are unavailable, use 'gene_start' and 'gene_end' as fallback.
    - match_col (str): Column name representing splice site positions (default: 'position').

    Returns:
    - pd.DataFrame: DataFrame with added columns 'distance_to_start' and 'distance_to_end'.
    """
    # Check required columns
    required_columns = {'gene_id', 'transcript_id', match_col, 'splice_type'}
    boundary_columns = {'tx_start', 'tx_end', 'gene_start', 'gene_end'}

    if not required_columns.issubset(df.columns):
        raise ValueError(f"Input dataframe must contain columns: {required_columns}")
    if not boundary_columns.intersection(df.columns):
        raise ValueError(f"Input dataframe must contain at least transcript or gene boundaries: {boundary_columns}")

    # Determine whether to use transcript or gene boundaries
    if 'tx_start' in df.columns and 'tx_end' in df.columns:
        start_col, end_col = 'tx_start', 'tx_end'
    elif 'gene_start' in df.columns and 'gene_end' in df.columns:
        start_col, end_col = 'gene_start', 'gene_end'
    else:
        raise ValueError("No valid transcript or gene boundaries found in the dataframe.")

    # Compute distances
    df['distance_to_start'] = df[match_col] - df[start_col]
    df['distance_to_end'] = df[end_col] - df[match_col]

    # Verify distances are positive
    invalid_rows = df[(df['distance_to_start'] < 0) | (df['distance_to_end'] < 0)]
    if not invalid_rows.empty:
        raise ValueError(
            "Found invalid distances (negative values). Ensure that "
            f"{match_col} contains absolute positions. Examples:\n{invalid_rows}"
        )

    return df


def compute_distances_with_strand_adjustment(df, match_col: str = "position", **kwargs):
    """Compute absolute genomic coordinates and distances to boundaries.

    The function keeps the *original* dataframe backend (Polars ⇄ Pandas) and
    avoids any per-row Python loops or unnecessary copies, which dramatically
    lowers memory usage when operating on millions of rows.

    Parameters
    ----------
    df : pl.DataFrame | pandas.DataFrame
        Training dataframe that must contain at least the columns::
            gene_id, strand, gene_start, gene_end, <match_col>
    match_col : str, default "position"
        Column with the splice-site *relative* position, counted from
        ``gene_start`` on the + strand and from ``gene_end`` on the – strand.

    Returns
    -------
    Same type as *df* with 3 new columns::
        absolute_position, distance_to_start, distance_to_end
    """
    import pandas as pd
    from typing import Set

    col_tid = kwargs.get("col_tid", "transcript_id")

    required: Set[str] = {"gene_id", "strand", "gene_start", "gene_end", match_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"compute_distances_with_strand_adjustment: missing required columns {missing}"
        )

    # ------------------------------------------------------------------
    # Polars implementation (preferred for large datasets) -------------
    # ------------------------------------------------------------------
    if isinstance(df, pl.DataFrame):
        start_col = "gene_start" if "gene_start" in df.columns else "tx_start"
        end_col = "gene_end" if "gene_end" in df.columns else "tx_end"

        # Ensure columns used in arithmetic are numeric (Int64).  Some cached TSVs
        # may store them as strings, which triggers the Polars “arithmetic on string”
        # exception.  We cast non-strictly so any already-numeric columns stay as-is.
        numeric_cols = [start_col, end_col, match_col]
        df = df.with_columns([
            pl.col(c).cast(pl.Int64, strict=False).alias(c) for c in numeric_cols
        ])

        # Absolute genomic coordinate of the splice site
        df = df.with_columns(
            pl.when(pl.col("strand") == "+")
            .then(pl.col("gene_start") + pl.col(match_col))
            .otherwise(pl.col("gene_end") - pl.col(match_col))
            .alias("absolute_position")
        )

        # Distances to boundaries (vectorised expressions)
        df = df.with_columns(
            [
                (pl.col("absolute_position") - pl.col(start_col)).alias("distance_to_start"),
                (pl.col(end_col) - pl.col("absolute_position")).alias("distance_to_end"),
            ]
        )

        if kwargs.get("validate", True):
            n_invalid = df.filter(
                (pl.col("distance_to_start") < 0) | (pl.col("distance_to_end") < 0)
            ).height
            if n_invalid:
                raise ValueError(
                    f"Negative distances detected in {n_invalid} rows (Polars branch)."
                )
        return df

    # ------------------------------------------------------------------
    # Pandas fallback (small / legacy datasets) -------------------------
    # ------------------------------------------------------------------
    strand = df["strand"]
    rel_pos = df[match_col]

    # Vectorised absolute position
    abs_pos = df["gene_start"].where(strand == "+", df["gene_end"]) + rel_pos.where(
        strand == "+", -rel_pos
    )
    df["absolute_position"] = abs_pos

    start_col = "gene_start" if "gene_start" in df.columns else "tx_start"
    end_col = "gene_end" if "gene_end" in df.columns else "tx_end"

    df["distance_to_start"] = df["absolute_position"] - df[start_col]
    df["distance_to_end"] = df[end_col] - df["absolute_position"]

    if kwargs.get("validate", True):
        invalid_mask = (df["distance_to_start"] < 0) | (df["distance_to_end"] < 0)
        if invalid_mask.any():
            raise ValueError(
                "Negative distances detected in Pandas branch. "
                f"Examples:\n{df[invalid_mask].head()}"
            )

    return df




def compute_distance_to_transcript_boundaries_via_gtf(position_df, gtf_file_path):
    """
    Compute the distance between each splice site and the start/end of its associated transcript.

    Parameters:
    - position_df (pd.DataFrame): DataFrame with splice site information.
      Required columns: ['gene_id', 'position', 'transcript_id', 'splice_type', 'pred_type'].
    - gtf_file_path (str): Path to the GTF file containing transcript boundaries.

    Returns:
    - position_df_with_distances (pd.DataFrame): Updated DataFrame with distance to transcript start/end.
    """
    # Load GTF file into a DataFrame
    gtf_df = pd.read_csv(
        gtf_file_path, sep='\t', comment='#',
        names=['seqname', 'source', 'feature', 'start', 'end', 'score', 'strand', 'frame', 'attribute']
    )

    # Filter for transcript features
    transcript_df = gtf_df[gtf_df['feature'] == 'transcript']

    # Extract transcript_id, start, and end
    transcript_df['transcript_id'] = transcript_df['attribute'].str.extract(r'transcript_id "([^"]+)"')
    transcript_df = transcript_df[['transcript_id', 'start', 'end']].drop_duplicates()

    # Merge position_df with transcript_df on 'transcript_id'
    merged_df = pd.merge(position_df, transcript_df, on='transcript_id', how='left')

    # Check for missing transcripts
    if merged_df['start'].isnull().any() or merged_df['end'].isnull().any():
        missing_transcripts = merged_df[merged_df['start'].isnull() | merged_df['end'].isnull()]
        print(f"[WARNING] Missing transcript boundaries for {len(missing_transcripts)} splice sites.")
        print(missing_transcripts[['gene_id', 'transcript_id']].drop_duplicates())

    # Compute distances to transcript start and end
    merged_df['distance_to_start'] = merged_df['position'] - merged_df['start']
    merged_df['distance_to_end'] = merged_df['end'] - merged_df['position']

    # Return updated DataFrame
    return merged_df


def compute_intron_lengths(gtf_file_path=None):
    """
    Compute intron lengths for each transcript from a GTF file.

    Parameters:
    - gtf_file_path (str): Path to the GTF file containing genomic annotations.

    Returns:
    - intron_lengths (pd.DataFrame): A DataFrame containing transcript IDs, intron indices, and intron lengths.
                                     Columns: ['transcript_id', 'intron_index', 'intron_length']
    """
    from collections import defaultdict

    fa = FeatureAnalyzer(gtf_file=gtf_file_path)
    exons = fa.retrieve_exon_dataframe(to_pandas=True)

    # Load the GTF file into a DataFrame
    # columns = [
    #     'seqname', 'source', 'feature', 'start', 'end', 'score', 'strand', 'frame', 'attribute'
    # ]
    # gtf_df = pd.read_csv(gtf_file_path, sep='\t', comment='#', names=columns)

    # # Filter for exons
    # exons = gtf_df[gtf_df['feature'] == 'exon']

    # # Parse attributes to extract transcript_id and gene_id
    # def parse_attributes(attributes):
    #     attrs = {}
    #     for attr in attributes.split(';'):
    #         attr = attr.strip()
    #         if attr:
    #             key, value = attr.split(' ')
    #             attrs[key] = value.strip('"')
    #     return attrs

    # exons['transcript_id'] = exons['attribute'].apply(lambda x: parse_attributes(x).get('transcript_id', None))
    # exons['gene_id'] = exons['attribute'].apply(lambda x: parse_attributes(x).get('gene_id', None))

    # Group by transcript_id and calculate intron lengths
    intron_lengths = []
    for transcript_id, group in exons.groupby('transcript_id'):
        group = group.sort_values('start')
        starts = group['end'].values[:-1]
        ends = group['start'].values[1:]
        introns = ends - starts - 1  # Subtract 1 to exclude exon ends and starts
        for intron_index, intron_length in enumerate(introns, start=1):
            intron_lengths.append({
                'transcript_id': transcript_id,
                'intron_index': intron_index,
                'intron_length': intron_length
            })

    # Convert to a DataFrame
    intron_lengths_df = pd.DataFrame(intron_lengths)

    return intron_lengths_df


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


def compute_exon_lengths(gtf_file_path):
    """
    Compute exon lengths for each transcript from a GTF file.

    Parameters:
    - gtf_file_path (str): Path to the GTF file containing genomic annotations.

    Returns:
    - exon_lengths (pd.DataFrame): A DataFrame containing transcript IDs, exon indices, and exon lengths.
                                    Columns: ['transcript_id', 'exon_index', 'exon_length']
    """
    from .utils_bio import parse_attributes, extract_exons_from_gtf

    # Load the GTF file into a DataFrame
    # columns = [
    #     'seqname', 'source', 'feature', 'start', 'end', 'score', 'strand', 'frame', 'attribute'
    # ]
    # gtf_df = pd.read_csv(gtf_file_path, sep='\t', comment='#', names=columns)

    # # Filter for exons
    # exons = gtf_df[gtf_df['feature'] == 'exon']

    # # Parse attributes to extract transcript_id and gene_id
    # def parse_attributes(attributes):
    #     attrs = {}
    #     for attr in attributes.split(';'):
    #         attr = attr.strip()
    #         if attr:
    #             key, value = attr.split(' ')
    #             attrs[key] = value.strip('"')
    #     return attrs

    # exons['transcript_id'] = exons['attribute'].apply(lambda x: parse_attributes(x).get('transcript_id', None))
    # exons['gene_id'] = exons['attribute'].apply(lambda x: parse_attributes(x).get('gene_id', None))

    fa = FeatureAnalyzer(gtf_file=gtf_file_path)
    exons = fa.retrieve_exon_dataframe(to_pandas=True)
    print("[test] columns(exons):", list(exons.columns))
    # Columns: ['gene_id', 'gene_name', 'seqname', 'start', 'end', 'strand']

    # Compute exon lengths
    exons['exon_length'] = exons['end'] - exons['start'] + 1 
    # NOTE: Polars DataFrame does not support direct assignment of a Series by index
    #       exons = exons.with_column((exons['end'] - exons['start'] + 1).alias('exon_length'))

    # Group by transcript_id and sort by start position
    exon_lengths = []
    for transcript_id, group in exons.groupby('transcript_id'):
        group = group.sort_values('start')
        for exon_index, row in enumerate(group.itertuples(), start=1):
            exon_lengths.append({
                'transcript_id': transcript_id,
                'exon_index': exon_index,
                'exon_length': row.exon_length
            })

    # Convert to a DataFrame
    exon_lengths_df = pd.DataFrame(exon_lengths)

    return exon_lengths_df


def compute_total_exon_lengths(gtf_file_path):
    """
    Compute the total exon lengths for each transcript using the GTF file.

    Parameters:
    - gtf_file_path (str): Path to the GTF file containing genomic annotations.

    Returns:
    - total_exon_lengths_df (pd.DataFrame): A DataFrame with total exon lengths for each transcript.
                                            Columns: ['transcript_id', 'total_exon_length']
    """
    # Compute individual exon lengths
    exon_lengths_df = compute_exon_lengths(gtf_file_path)

    # Group by transcript_id and sum exon lengths
    total_exon_lengths_df = (
        exon_lengths_df.groupby('transcript_id')['exon_length']
        .sum()
        .reset_index()
        .rename(columns={'exon_length': 'total_exon_length'})
    )

    return total_exon_lengths_df


def compute_total_intron_lengths(gtf_file_path):
    """
    Compute the total intron lengths for each transcript using the GTF file.

    Parameters:
    - gtf_file_path (str): Path to the GTF file containing genomic annotations.

    Returns:
    - total_intron_lengths_df (pd.DataFrame): A DataFrame with total intron lengths for each transcript.
                                              Columns: ['transcript_id', 'total_intron_length']
    """
    # Compute individual intron lengths
    intron_lengths_df = compute_intron_lengths(gtf_file_path)

    # Group by transcript_id and sum intron lengths
    total_intron_lengths_df = (
        intron_lengths_df.groupby('transcript_id')['intron_length']
        .sum()
        .reset_index()
        .rename(columns={'intron_length': 'total_intron_length'})
    )

    return total_intron_lengths_df


def compute_total_lengths(gtf_file_path):
    """
    Compute the total exon and intron lengths for each transcript.

    Parameters:
    - gtf_file_path (str): Path to the GTF file containing genomic annotations.

    Returns:
    - total_lengths_df (pd.DataFrame): A DataFrame with total exon and intron lengths for each transcript.
                                       Columns: ['transcript_id', 'total_exon_length', 'total_intron_length']
    """
    # Compute total exon and intron lengths
    total_exon_lengths_df = compute_total_exon_lengths(gtf_file_path)
    total_intron_lengths_df = compute_total_intron_lengths(gtf_file_path)

    # Merge the two DataFrames
    total_lengths_df = pd.merge(
        total_exon_lengths_df,
        total_intron_lengths_df,
        on='transcript_id',
        how='outer'
    ).fillna(0)  # Fill missing values with 0 if a transcript has no exons or introns

    return total_lengths_df


def demo_compute_total_lengths(gtf_file_path=None, **kargs):
    """
    Demo function to verify the correctness of compute_total_lengths.

    Parameters:
    - gtf_file_path (str): Path to the GTF file.

    Returns:
    - None
    """
    if gtf_file_path is None:
        gtf_file_path = FeatureAnalyzer.gtf_file

    fa = FeatureAnalyzer()

    # Compute total exon lengths
    total_exon_lengths = compute_total_exon_lengths(gtf_file_path)
    print(total_exon_lengths.head())

    # Compute total intron lengths
    total_intron_lengths = compute_total_intron_lengths(gtf_file_path)
    print(total_intron_lengths.head())

    # Compute combined total lengths
    total_lengths = compute_total_lengths(gtf_file_path)
    print(total_lengths.head())

    # Save to file
    fa.save_dataframe(total_lengths, "total_lengths.csv")


######################################

def display_dataframe(df, num_rows=5, num_columns=10):
    """
    Display example rows from the DataFrame, including the header, with high readability.

    Parameters:
    - df (pl.DataFrame or pd.DataFrame): The DataFrame to display.
    - num_rows (int): Number of rows to display (default is 5).
    - num_columns (int): Number of columns to display at a time (default is 10).

    Returns:
    - None
    """
    # Convert Polars DataFrame to Pandas DataFrame if necessary
    if isinstance(df, pl.DataFrame):
        df = df.to_pandas()

    # Display the DataFrame in chunks of columns
    total_columns = df.shape[1]
    for start in range(0, total_columns, num_columns):
        end = start + num_columns
        subset_df = df.iloc[:, start:end]
        print(tabulate(subset_df.head(num_rows), headers='keys', tablefmt='psql'))
        print("\n")


def run_genomic_gtf_feature_extraction(gtf_file_path=None, **kargs):
    # from meta_spliceai.splice_engine.utils_doc import display_dataframe
    # from meta_spliceai.splice_engine.utils_fs import get_id_columns

    overwrite = kargs.get('overwrite', False)

    fa = FeatureAnalyzer(gtf_file=gtf_file_path, overwrite=overwrite)

    verbose = kargs.get('verbose', 1)
    save = kargs.get('save', True)
    output_dir = kargs.get('analysis_dir', fa.analysis_dir)
    format = kargs.get('format', 'tsv')

    # Extract features from the GTF file
    print_emphasized("[i/o] Extracting gene-level features from the GTF file ...")
    
    gene_features = fa.retrieve_gene_features()
    # gene_features = extract_gene_features_from_gtf(gtf_file_path, output_file=output_file)
    print("[test] columns(gene_features):", list(gene_features.columns))
    # NOTE: ['start', 'end', 'score', 'strand', 'gene_id', 'gene_name', 'gene_type', 'gene_length', 'chrom']
    #      - Keep: 'gene_id', 'start', 'end', 'strand', 'gene_type', 'gene_length', 'chrom'

    print_emphasized("[i/o] Summarizing transcript-level features ...")
    # output_file = os.path.join(output_dir, f'transcript_features.{format}')
    # transcript_features = summarize_transcript_features(gtf_file_path, output_file=output_file)
    transcript_features = fa.retrieve_transcript_features()
    print("[test] columns(transcript):", list(transcript_features.columns))
    # NOTE: ['chrom', 'start', 'end', 'strand', 'transcript_id', 'transcript_name', 'transcript_type', 'transcript_length', 'gene_id']
    #       - Keep 'gene_id', 'transcript_id', 'transcript_length' (and possibly 'transcript_type')

    print_emphasized("[i/o] Summarizing exon-level features ...")
    # output_file = os.path.join(output_dir, f'exon_features.{format}')
    # exon_features = summarize_exon_features(gtf_file_path, output_file=output_file)
    exon_features = fa.retrieve_exon_features()  # equivalent to retrieve_exon_features_at_transcript_level()
    # exon_features = fa.retrieve_exon_features_at_transcript_level()
    print("[test] columns(exon_features):", list(exon_features.columns))
    # NOTE: ['transcript_id', 'transcript_type', 'gene_id', 'num_exons', 'avg_exon_length', 'median_exon_length', 'total_exon_length']
    #       - Keep 'gene_id', 'transcript_id', 'num_exons', 'avg_exon_length', 'median_exon_length', 'total_exon_length'

    # Subset columns
    gene_features = gene_features.select(['gene_id', 'start', 'end', 'strand', 'gene_type', 'gene_length', 'chrom'])
    
    # rename columns
    gene_features = gene_features.rename({
        'start': 'gene_start',
        'end': 'gene_end',
        # 'gene_length': 'gene_length'
    })

    transcript_features = transcript_features.select(['gene_id', 'transcript_id', 'transcript_length', 'start', 'end'])
    # Optionally, add 'transcript_type', 'tx_start', 'tx_end'

    transcript_features = transcript_features.rename({
        'start': 'tx_start',
        'end': 'tx_end'
    })

    exon_features = exon_features.select(['gene_id', 'transcript_id', 'num_exons', 'avg_exon_length', 'median_exon_length', 'total_exon_length'])

    # Join dataframes
    combined_features = transcript_features.join(gene_features, on='gene_id', how='inner')
    combined_features = combined_features.join(exon_features, on=['gene_id', 'transcript_id'], how='inner')

    # Verify the number of unique genes and transcripts
    num_unique_genes = combined_features.select(pl.col('gene_id').n_unique()).to_series()[0]
    num_unique_transcripts = combined_features.select(pl.col('transcript_id').n_unique()).to_series()[0]

    print("[info] columns(combined_features):", list(combined_features.columns))
    print(f"[info] Number of unique genes: {num_unique_genes}")
    print(f"[info] Number of unique transcripts: {num_unique_transcripts}")
    print(f"[info] Shape of the DataFrame: {combined_features.shape}")

    # Extract features from error analysis file 

    # Extract features from error sequence file

    full_gene_features = combined_features
    # Join features
    # full_gene_features = \
    #     join_gene_features(
    #         gene_features, 
    #         transcript_features, 
    #         exon_features)

    print(full_gene_features.head())

    if verbose:
        print(f"[info] Consolidated gene-level features for {num_unique_genes} genes with {num_unique_transcripts} transcripts.")

    # Save the DataFrame
    if save:
        output_path = kargs.get('output_path', os.path.join(output_dir, f'genomic_gtf_feature_set.{format}'))

        print_emphasized(f"[i/o] Saving GTF-derived genomic features to a file:\n{output_path}\n ...")

        sep = '\t' if output_path.endswith('.tsv') else ','
        full_gene_features.write_csv(output_path, separator=sep)

    return full_gene_features


######################################

def demo_get_overlapping_gene_metadata(gtf_file_path=None, show_relationship=True): 
    """
    Demo function to verify the correctness of get_overlapping_gene_metadata.

    Parameters:
    - gtf_file_path (str): Path to the GTF file.

    Returns:
    - None
    """
    import pprint
    from meta_spliceai.mllib.sampling import sample_dict

    local_dir = '/path/to/meta-spliceai/data/ensembl/'
    output_file = os.path.join(local_dir, "overlapping_gene_counts-v2.tsv")

    if gtf_file_path is None: 
        gtf_file_path = "/path/to/meta-spliceai/data/ensembl/Homo_sapiens.GRCh38.112.gtf"  # Replace with your GTF file path

    show_overlapping_gene_rel = show_relationship 

    if not show_overlapping_gene_rel: 
        print_emphasized("[i/o] Testing overlapping genes in dataframe format ...")
        sorted_genes = get_overlapping_gene_metadata(gtf_file_path, output_format='dataframe')
        print(sorted_genes[:10])  # Top 10 genes with the most overlaps
        # Columns: gene_id_1, gene_id_2, chrom, start_1, end_1, start_2, end_2, strand_1, strand_2, num_overlaps

        if output_file:
            file_extension = output_file.split('.')[-1]
            if file_extension == 'csv':
                sorted_genes.write_csv(output_file)
            elif file_extension == 'tsv':
                sorted_genes.write_csv(output_file, separator='\t')
            else:
                raise ValueError("Unsupported file format. Please use 'csv' or 'tsv'.")

        # -----------------------------
        print_emphasized("[i/o] Including only overlapping genes with valid splice sites ...")
        sorted_genes = \
            get_overlapping_gene_metadata(
                gtf_file_path, 
                filter_valid_splice_sites=True, 
                min_exons=2,
                output_format='dataframe')
        print(sorted_genes[:10])  # Top 10 genes with the most overlaps

        # -----------------------------
        print_emphasized("[i/o] Dictionary format ...")
        metadata_dict = get_overlapping_gene_metadata(gtf_file_path, output_format='dict')
        sorted_genes = sort_genes_by_overlap_count(metadata_dict)
        print(sorted_genes[:100])  # Top 10 genes with the most overlaps

        print("[info] Example entries in metadata_dict ...")
        metadata_dict_sampled = sample_dict(metadata_dict, n_sample=10)

        # Nicely display the content of the sampled dictionaries
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(metadata_dict_sampled)
        
    else: 
        print_emphasized("[i/o] Testing overlapping genes in dataframe format ...")
        metadata_df = get_overlapping_gene_relationships(gtf_file_path, output_format='dataframe')
        print(metadata_df.head(10))  # Top 10 genes with the most overlaps

        print_with_indent("[action] Sorting genes by the number of overlapping genes")
        sorted_genes_df = sort_genes_by_overlap_count_df(metadata_df)
        print(sorted_genes_df.head(10))  # Top 10 genes with the most overlaps

        print_emphasized("[i/o] Testing overlapping genes in dictionary format ...")
        metadata_dict = get_overlapping_gene_relationships(gtf_file_path, output_format='dict')
        sorted_genes = sort_genes_by_overlap_count(metadata_dict)
        print(sorted_genes[:10])  # Top 10 genes with the most overlaps

def demo_get_overlapping_gene_metadata_via_splice_analyzer(gtf_file_path=None, show_relationship=True):

    sa = SpliceAnalyzer()
    df = sa.retrieve_overlapping_gene_metadata(output_format='dataframe')
    
    print("[info] Columns:", list(df.columns))
    display(df.head(10))



def demo_determine_chromosome_sizes():

    local_dir = '/path/to/meta-spliceai/data/ensembl/'
    src_dir = '/path/to/meta-spliceai/data/ensembl/'
    eval_dir = '/path/to/meta-spliceai/data/ensembl/spliceai_eval'

    gtf_file = "/path/to/meta-spliceai/data/ensembl/Homo_sapiens.GRCh38.112.gtf"  # Replace with your GTF file path
    genome_fasta = os.path.join(src_dir, "Homo_sapiens.GRCh38.dna.primary_assembly.fa") 
    output_path = os.path.join(local_dir, "chromosome_sizes.tsv") 
    verbose = True
    overwrite = False

    if ovewrite or not os.path.exists(output_path): 
        df = generate_chromosome_sizes_file(gtf_file, genome_fasta, output_path, verbose=verbose)
    else: 
        # Read the output file into a DataFrame
        df = pl.read_csv(output_path, separator='\t', has_header=True, new_columns=["chromosome", "size"])
    
    display(df)


def select_random_gene_ids(gene_df: pl.DataFrame, n_genes: int, k: int) -> list[str]:
    """
    From the gene_df, randomly pick k gene_ids (assuming it has a 'gene_id' column).
    Returns a list of gene_id strings.
    """
    all_gene_ids = gene_df.select(pl.col("gene_id")).unique().to_series().to_list()
    # Sample k gene_ids (without replacement):
    return random.sample(all_gene_ids, k=min(k, len(all_gene_ids)))


def compare_gene_and_premrna_sequences(gtf_file=None, genome_fasta=None, **kargs): 
    from meta_spliceai.splice_engine.utils_bio import (
            load_sequences, 
            load_chromosome_sequence,
            load_sequences_by_chromosome, 
            load_chromosome_sequence_streaming, 
            analyze_longest_transcripts, 
            analyze_genes, 
            adjust_chunk_size, 
            check_available_memory,
            compare_gene_and_transcript_boundaries, 
            compare_gene_and_transcript_lengths, 
        )
    import random, time

    separator = kargs.get('separator', kargs.get('sep', '\t'))
    test_mode = kargs.get('test_mode', True)
    local_dir = kargs.get("output_dir", os.getcwd()) 

    run_sequence_retrieval = kargs.get('run_sequence_retrieval', False)
    mode = 'gene'
    seq_type = 'full'  # options: 'minmax', 'full'
    format = 'parquet'

    gene_sequence_file = f"gene_sequence_minmax.{format}" if seq_type == 'minmax' else f"gene_sequence.{format}"
    gene_seq_df_path = os.path.join(local_dir,  gene_sequence_file)
    tx_seq_df_path = os.path.join(local_dir, f"tx_sequence.{format}")

    if run_sequence_retrieval: 
        print_emphasized("[action] Extracting DNA sequences ...")
        gene_sequence_retrieval_workflow(
            gtf_file, genome_fasta, 
            output_file=gene_seq_df_path, 
            mode=seq_type)
        
        print_emphasized("[action] Extracting pre-mRNA sequences ...")
        transcript_sequence_retrieval_workflow(
            gtf_file, 
            genome_fasta, 
            output_file=tx_seq_df_path)

    n_chr_processed = 0
    chromosomes = kargs.get("chromosomes", None)
    if chromosomes is None:
        chromosomes = ['2', ] # [str(i) for i in range(1, 23)] + ['X', 'Y', 'MT']

    for chr in tqdm(chromosomes, desc="Processing chromosomes"):

        tqdm.write(f"Processing chromosome={chr} ...")

        # Load sequence data using streaming mode
        lazy_gene_seq_df = load_chromosome_sequence_streaming(gene_seq_df_path, chr, format=format)
        lazy_tx_seq_df = load_chromosome_sequence_streaming(tx_seq_df_path, chr, format=format)

        # Initialize chunk size
        chunk_size = 500 if not test_mode else 50  # Starting chunk size
        seq_len_avg = 50000  # Assume an average sequence length for now
        num_genes = lazy_gene_seq_df.select(pl.col('gene_id').n_unique()).collect().item()
        n_chunk_processed = 0

        if test_mode:
            # Collect a small DF with all gene IDs for this chromosome:
            gene_ids_df = lazy_gene_seq_df.select(pl.col("gene_id")).unique().collect()
            total_genes = gene_ids_df.height  # or n_unique
            print("Number of genes in this chromosome:", total_genes)

            # Randomly pick chunk_size gene IDs:
            subset_ids = select_random_gene_ids(gene_ids_df, total_genes, chunk_size)

            # Now filter both gene_seq_df and tx_seq_df by these gene_ids:
            #   gene_seq_chunk will contain only those rows whose gene_id is in subset_ids
            gene_seq_chunk = (
                lazy_gene_seq_df
                .filter(pl.col("gene_id").is_in(subset_ids))
                .collect()
            )
            tx_seq_chunk = (
                lazy_tx_seq_df
                .filter(pl.col("gene_id").is_in(subset_ids))
                .collect()
            )

            # Now gene_seq_chunk and tx_seq_chunk share the same set of gene_ids
            print(f"gene_seq_chunk shape: {gene_seq_chunk.shape}")
            print(f"tx_seq_chunk shape: {tx_seq_chunk.shape}")

            # Estimate the memory usage of the resulting DataFrame
            chunk_dict = {'gene': gene_seq_chunk, 'transcript': tx_seq_chunk}
            for k, seq_chunk in chunk_dict.items():
                estimated_memory_usage = seq_chunk.estimated_size()
                print(f"Estimated memory usage of seq_chunk (type={k}): {estimated_memory_usage / (1024 ** 2):.2f} MB")

            # gene_length_df = analyze_genes(gene_seq_chunk)
            # tx_length_df = analyze_longest_transcripts(tx_seq_chunk)

            # Compare boundaries:
            boundaries_df = compare_gene_and_transcript_boundaries(gene_seq_chunk, tx_seq_chunk)

            print("Number of rows:", boundaries_df.height)
            print("Number of columns:", boundaries_df.width)
            display_dataframe(boundaries_df.head())

            # Compare lengths & ratio:
            lengths_df = compare_gene_and_transcript_lengths(gene_seq_chunk, tx_seq_chunk)

            print("Number of rows:", lengths_df.height)
            print("Number of columns:", lengths_df.width)
            display_dataframe(lengths_df.head())

        else: 

            for chunk_start in range(0, num_genes, chunk_size):  # Process chunk_size genes at a time
                chunk_end = min(chunk_start + chunk_size, num_genes)

                # Track the start time for each chunk
                chunk_start_time = time.time()

                # Adjust the chunk size based on memory availability
                chunk_size = adjust_chunk_size(chunk_size, seq_len_avg)

                # Filter the LazyFrame to process only the current chunk
                gene_seq_chunk = lazy_gene_seq_df.slice(chunk_start, chunk_size).collect()
                tx_seq_chunk = lazy_tx_seq_df.slice(chunk_start, chunk_size).collect()

                # Estimate the memory usage of the resulting DataFrame
                chunk_dict = {'gene': gene_seq_chunk, 'transcript': tx_seq_chunk}
                for k, seq_chunk in chunk_dict.items():
                    estimated_memory_usage = seq_chunk.estimated_size()
                    print(f"Estimated memory usage of seq_chunk (type={k}): {estimated_memory_usage / (1024 ** 2):.2f} MB")

                # gene_length_df = analyze_genes(gene_seq_chunk)
                # tx_length_df = analyze_longest_transcripts(tx_seq_chunk)

                # Compare boundaries:
                boundaries_df = compare_gene_and_transcript_boundaries(gene_seq_chunk, tx_seq_chunk)
                display_dataframe(boundaries_df.head())

                # Compare lengths & ratio:
                lengths_df = compare_gene_and_transcript_lengths(gene_seq_chunk, tx_seq_chunk)
                display_dataframe(lengths_df.head())
            


def demo_run_genomic_gtf_feature_extraction(gtf_file_path=None): 

    eval_dir = FeatureAnalyzer.eval_dir
    analysis_dir = FeatureAnalyzer.analysis_dir

    if gtf_file_path is None: 
        gtf_file_path = "/path/to/meta-spliceai/data/ensembl/Homo_sapiens.GRCh38.112.gtf"  # Replace with your GTF file path

    format = 'csv'
    output_path = os.path.join(analysis_dir, f'genomic_gtf_feature_set.{format}')

    df_fs = run_genomic_gtf_feature_extraction(gtf_file_path, output_file=output_path)

    display_dataframe(df_fs)

    return df_fs


def test(): 

    src_dir = '/path/to/meta-spliceai/data/ensembl/'
    local_dir = '/path/to/meta-spliceai/data/ensembl/'
    gtf_file = "/path/to/meta-spliceai/data/ensembl/Homo_sapiens.GRCh38.112.gtf"  # Replace with your GTF file path
    genome_fasta = os.path.join(src_dir, "Homo_sapiens.GRCh38.dna.primary_assembly.fa") 

    # Determine the size of each chromosome 
    # demo_determine_chromosome_sizes()

    # Determine overlapping genes for each gene from a GTF file.
    print_emphasized("[demo] Overalpping genes ...")
    # demo_get_overlapping_gene_set()
    # demo_get_overlapping_gene_metadata(show_relationship=False)  # similar to the above but also including meta data
    demo_get_overlapping_gene_metadata_via_splice_analyzer()

    # demo_run_genomic_gtf_feature_extraction()
    # demo_compute_total_lengths()

    # print_emphasized("[demo] Gene vs transcript sequences")
    # compare_gene_and_premrna_sequences(
    #     gtf_file=gtf_file, 
    #     genome_fasta=genome_fasta,
    #     output_dir=local_dir,
    #     test_mode=True, 
    #     run_sequence_retrieval=False)

    return


if __name__ == "__main__": 
    test()

