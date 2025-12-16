"""
Utility functions for handling sequence data in meta_models package.
"""
import os
import glob
import polars as pl
from typing import Optional, List, Dict, Union, Tuple, Set, Any

def safe_load_chromosome_sequence(file_path: str, chromosome: str, format: str = 'tsv', 
                                 return_none_if_missing: bool = False) -> Optional[pl.LazyFrame]:
    """
    Safely load chromosome sequence data using a format compatible with polars API.
    This is a wrapper around utils_bio.load_chromosome_sequence_streaming that handles
    the 'sep' vs 'separator' parameter correctly.
    
    Parameters:
    - file_path (str): Base path to the sequence file (without chromosome suffix)
    - chromosome (str): Chromosome identifier
    - format (str): File format ('tsv', 'csv', or 'parquet')
    - return_none_if_missing (bool): Return None if file is missing instead of raising error
    
    Returns:
    - pl.LazyFrame or None: LazyFrame with sequence data or None if file missing and return_none_if_missing=True
    """
    base_name, ext = os.path.splitext(file_path)
    chrom_file_path = f"{base_name}_{chromosome}{ext}"

    if not os.path.exists(chrom_file_path):
        if return_none_if_missing:
            print(f"File {chrom_file_path} does not exist (yet).")
            return None
        else:
            raise FileNotFoundError(f"File {chrom_file_path} not found.")

    # Read the file in streaming mode (lazy loading) with correct polars API
    if format == 'tsv':
        return pl.scan_csv(chrom_file_path, separator='\t', low_memory=True)
    elif format == 'csv':
        return pl.scan_csv(chrom_file_path, low_memory=True)
    elif format == 'parquet':
        return pl.scan_parquet(chrom_file_path)
    else:
        raise ValueError("Unsupported format. Please choose 'tsv', 'csv', or 'parquet'.")


def load_gene_sequences_for_targets(target_genes, config, seq_type="standard", prioritized_chromosomes=None):
    """
    Load gene sequences for target genes from sequence files.
    
    Parameters
    ----------
    target_genes : List[str] or Dict[str, Union[str, int]]
        Either a list of gene IDs or names to load, or a dictionary mapping gene IDs/names to 
        chromosome numbers (e.g., {'STMN2': '8', 'UNC13A': '19'})
    config : SpliceAIConfig
        Configuration object with paths and settings
    seq_type : str, optional
        Type of gene sequence files to load:
        - "standard": Regular gene start to gene end (default)
        - "minmax": Min transcript start to max transcript end
    prioritized_chromosomes : List[Union[str, int]], optional
        List of chromosome numbers to check first, e.g. ['8', '19']
        
    Returns
    -------
    pl.DataFrame
        DataFrame containing the gene sequences for target genes
    """
    # from meta_spliceai.splice_engine.run_spliceai_workflow import load_chromosome_sequence_streaming
    
    # Handle the case where target_genes is a dictionary that maps genes to chromosomes
    if isinstance(target_genes, dict):
        gene_to_chrom = target_genes.copy()
        target_genes = list(gene_to_chrom.keys())
        # Extract prioritized chromosomes from the dictionary values if not explicitly provided
        if prioritized_chromosomes is None:
            prioritized_chromosomes = list(set(str(chrom) for chrom in gene_to_chrom.values()))
    else:
        gene_to_chrom = None
        
    # Convert prioritized_chromosomes to strings if provided
    if prioritized_chromosomes:
        prioritized_chromosomes = [str(chrom) for chrom in prioritized_chromosomes]
        print(f"Prioritizing chromosomes: {', '.join(prioritized_chromosomes)}")
    
    # Sequence file is typically stored as gene_sequence.parquet or gene_sequence_[chr].parquet
    seq_format = 'parquet'  # Efficient format for sequence data
    local_dir = os.path.dirname(config.eval_dir)
    
    # Determine file pattern based on seq_type
    if seq_type.lower() == "minmax":
        main_seq_file = os.path.join(local_dir, f"gene_sequence_minmax.{seq_format}")
        chr_pattern = os.path.join(local_dir, f"gene_sequence_minmax_*.{seq_format}")
        file_prefix = "gene_sequence_minmax_"
    else:  # standard
        main_seq_file = os.path.join(local_dir, f"gene_sequence.{seq_format}")
        chr_pattern = os.path.join(local_dir, f"gene_sequence_*.{seq_format}")
        file_prefix = "gene_sequence_"
    
    # Track which genes we've found
    found_genes = set()
    
    # If main sequence file exists, use it
    if os.path.exists(main_seq_file):
        print(f"Found main sequence file: {main_seq_file}")
        try:
            # Load from main sequence file
            seq_df = pl.read_parquet(main_seq_file)
            filtered_df = seq_df.filter(
                pl.col('gene_id').is_in(target_genes) | 
                pl.col('gene_name').is_in(target_genes)
            )
            if filtered_df.height > 0:
                print(f"Found {filtered_df.height} target genes in main sequence file")
                # Track which genes we found
                found_genes.update(filtered_df.get_column('gene_id').to_list())
                found_genes.update(filtered_df.get_column('gene_name').to_list())
                
                # If we found all target genes, return immediately
                if all(gene in found_genes for gene in target_genes):
                    print("All target genes found in main sequence file")
                    return filtered_df
                else:
                    remaining_genes = [gene for gene in target_genes if gene not in found_genes]
                    print(f"Still need to find genes: {', '.join(remaining_genes)}")
            else:
                print(f"No target genes found in main sequence file")
        except Exception as e:
            print(f"Error loading sequences from main file: {e}")
    
    # If we get here, either main file doesn't exist or target genes weren't found
    print(f"Looking for chromosome-specific sequence files for {seq_type} sequences...")
    
    # Find all available chromosome-specific files
    chr_files = glob.glob(chr_pattern)
    
    if not chr_files:
        print(f"No chromosome-specific sequence files found matching pattern: {chr_pattern}")
        return None
        
    print(f"Found {len(chr_files)} chromosome-specific sequence files")
    
    # Extract chromosome numbers and map to file paths
    chromosome_files = {}
    for file_path in chr_files:
        filename = os.path.basename(file_path)
        try:
            # Extract chromosome number from filename
            chr_num = filename.replace(file_prefix, "").split(".")[0]
            
            # Skip if we detect a "minmax" prefix in a standard search or vice versa
            if seq_type.lower() == "standard" and "minmax" in filename:
                continue
            
            chromosome_files[chr_num] = file_path
            print(f"  Found sequence file for chromosome {chr_num}: {filename}")
        except Exception as e:
            print(f"Could not extract chromosome number from filename: {filename}, error: {str(e)}")
    
    if not chromosome_files:
        print("Could not determine chromosome numbers from available files")
        return None
        
    print(f"Available chromosomes: {', '.join(sorted(chromosome_files.keys()))}")
    
    # List of genes we still need to find
    remaining_genes = [gene for gene in target_genes if gene not in found_genes]
    
    # Collect sequences from each chromosome file
    all_seqs = []
    
    # Process chromosomes in prioritized order if specified
    if prioritized_chromosomes:
        # First check prioritized chromosomes
        for chr_num in prioritized_chromosomes:
            if chr_num in chromosome_files:
                found_in_chrom, seq_df = _process_chromosome(
                    chr_num, chromosome_files[chr_num], remaining_genes, seq_type
                )
                if seq_df is not None and seq_df.height > 0:
                    all_seqs.append(seq_df)
                    found_genes.update(found_in_chrom)
                    remaining_genes = [gene for gene in target_genes if gene not in found_genes]
                    
                    # If all genes are found, we can stop
                    if not remaining_genes:
                        print(f"All target genes found in prioritized chromosomes")
                        break
    
    # If we still have genes to find, check the remaining chromosomes
    if remaining_genes:
        # Sort chromosomes by number to process them in a consistent order
        for chr_num in sorted(chromosome_files.keys()):
            # Skip chromosomes we've already checked
            if prioritized_chromosomes and chr_num in prioritized_chromosomes:
                continue
                
            found_in_chrom, seq_df = _process_chromosome(
                chr_num, chromosome_files[chr_num], remaining_genes, seq_type
            )
            if seq_df is not None and seq_df.height > 0:
                all_seqs.append(seq_df)
                found_genes.update(found_in_chrom)
                remaining_genes = [gene for gene in target_genes if gene not in found_genes]
                
                # If all genes are found, we can stop
                if not remaining_genes:
                    print(f"All target genes found, stopping search")
                    break
    
    if all_seqs:
        # Combine all sequences found
        gene_df = combine_sequence_dataframes(all_seqs)
        print(f"Combined sequences for {gene_df.height} genes")
        return gene_df
    else:
        print(f"No sequences found for target genes in any {seq_type} chromosome file")
        return None


def _process_chromosome(chr_num, file_path, target_genes, seq_type):
    """
    Process a chromosome file to find target genes.
    
    Parameters
    ----------
    chr_num : str
        Chromosome number
    file_path : str
        Path to the chromosome file
    target_genes : List[str]
        List of genes to search for
    seq_type : str
        Type of sequence file (standard or minmax)
        
    Returns
    -------
    Tuple[List[str], pl.DataFrame]
        List of genes found in this chromosome and the DataFrame of sequences
    """
    print(f"  - Loading sequences from chromosome {chr_num}")
    try:
        # Use direct file path to load sequences
        seq_df = pl.read_parquet(file_path)
        
        if seq_df is not None:
            # Filter to only keep target genes
            filtered_df = seq_df.filter(
                pl.col('gene_id').is_in(target_genes) | 
                pl.col('gene_name').is_in(target_genes)
            )
            if filtered_df.height > 0:
                # Get list of genes found in this chromosome
                found_genes = set()
                found_genes.update(filtered_df.get_column('gene_id').to_list())
                found_genes.update(filtered_df.get_column('gene_name').to_list())
                found_genes = [g for g in found_genes if g in target_genes]
                
                print(f"    Found {filtered_df.height} target genes in chromosome {chr_num}: {', '.join(found_genes)}")
                return found_genes, filtered_df
            else:
                print(f"    No target genes found in chromosome {chr_num}")
                return [], None
    except Exception as e:
        print(f"Error loading chromosome {chr_num} sequences: {e}")
        return [], None


def scan_chromosome_sequence(seq_result, chromosome, format='parquet', separator='\t', verbosity=1):
    """
    Lazily load sequence data for a specific chromosome, handling both chromosome-specific
    files and combined sequence files.
    
    Parameters
    ----------
    seq_result : dict
        Dictionary with sequence file information (from prepare_genomic_sequences)
    chromosome : str
        Chromosome identifier to load
    format : str, default='parquet'
        Expected format of the sequence files
    separator : str, default='\t'
        Separator for TSV/CSV files
    verbosity : int, default=1
        Verbosity level
        
    Returns
    -------
    pl.LazyFrame
        Lazy dataframe containing the sequence data for the specified chromosome
        
    Raises
    ------
    FileNotFoundError
        If no sequence file could be found for the specified chromosome
    """
    # Ensure chromosome is a string
    chromosome = str(chromosome)
    
    # First, try to use chromosome-specific files if available
    if 'chr_sequence_files' in seq_result:
        # Find the file for this chromosome in the list of chr-specific files
        chr_pattern = f"_sequence_{chromosome}."
        chr_files = [f for f in seq_result['chr_sequence_files'] if chr_pattern in f]
        
        if chr_files:
            if verbosity >= 2:
                print(f"[debug] Using chromosome-specific file: {os.path.basename(chr_files[0])}")
            
            # Use the chromosome-specific file directly
            return pl.scan_parquet(chr_files[0]) if format == 'parquet' else pl.scan_csv(chr_files[0], separator=separator)
    
    # Fall back to combined sequence file if available
    sequence_file = seq_result.get("sequence_file") or seq_result.get("sequences_file")
    if sequence_file and os.path.exists(sequence_file):
        if verbosity >= 2:
            print(f"[debug] Using combined sequence file with filtering: {os.path.basename(sequence_file)}")
            
        # Use load_chromosome_sequence_streaming to filter the combined file
        return load_chromosome_sequence_streaming(sequence_file, chromosome, format=format)
    
    # If we got here, we couldn't find appropriate files
    raise FileNotFoundError(f"No sequence file found for chromosome {chromosome}")


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


def read_sequence_file(path, format='tsv', try_alternatives=True, verbosity=1, **kwargs):
    """
    Read a sequence file in various formats with error handling.
    
    Parameters
    ----------
    path : str
        Path to the sequence file
    format : str, default='tsv'
        Expected format of the file ('parquet', 'tsv', or 'csv')
    try_alternatives : bool, default=True
        Whether to try alternative formats if the expected format fails
    verbosity : int, default=1
        Verbosity level (0=quiet, 1=basic info, 2=debug)
        
    Returns
    -------
    DataFrame or None
        The loaded dataframe or None if loading failed
    """
    if not os.path.exists(path):
        if verbosity >= 2:
            print(f"[debug] File not found: {path}")
        return None
    
    try:
        # First try the expected format
        if format == 'parquet':
            import pyarrow.parquet as pq
            df = pq.read_table(path).to_pandas()
        else:
            separator = '\t' if format == 'tsv' else ','
            df = pl.read_csv(path, separator=separator)
        
        if verbosity >= 2:
            print(f"[info] Loaded {len(df)} sequences from {path}")
        return df
    
    except Exception as primary_err:
        if verbosity >= 2:
            print(f"[debug] Failed to load {path} as {format}: {str(primary_err)}")
        
        # Try alternative formats if requested
        if try_alternatives:
            try:
                if format != 'parquet':
                    # Try parquet if the original format wasn't parquet
                    if verbosity >= 2:
                        print(f"[debug] Trying parquet format...")
                    import pyarrow.parquet as pq
                    df = pq.read_table(path).to_pandas()
                    if verbosity >= 1:
                        print(f"[warning] File {path} is in parquet format despite {format} extension")
                    return df
                else:
                    # Try TSV if the original format was parquet
                    if verbosity >= 2:
                        print(f"[debug] Trying TSV format...")
                    df = pl.read_csv(path, separator='\t')
                    if verbosity >= 1:
                        print(f"[warning] File {path} is in TSV format despite parquet extension")
                    return df
            except Exception as alt_err:
                if verbosity >= 2:
                    print(f"[debug] Alternative format also failed: {str(alt_err)}")
        
        return None


def combine_sequence_dataframes(dfs, use_polars=None, verbosity=1):
    """
    Combine sequence dataframes with proper type conversion.
    
    Parameters
    ----------
    dfs : list
        List of dataframes to combine
    use_polars : bool, default=None
        Whether to use Polars for combining. If None, auto-detects based on input type
    verbosity : int, default=1
        Verbosity level (0=quiet, 1=basic info, 2=debug, 3=trace)
        
    Returns
    -------
    DataFrame or None
        The combined dataframe or None if combining failed
    """
    if not dfs:
        return None
    
    try:
        # Auto-detect dataframe type if not specified
        if use_polars is None:
            import polars as pl
            # Check if the first dataframe is a Polars dataframe
            use_polars = isinstance(dfs[0], pl.DataFrame)
            if verbosity >= 2:
                print(f"[debug] Auto-detected dataframe type: {'Polars' if use_polars else 'Pandas'}")
        
        if not use_polars:
            # Pandas approach
            import pandas as pd
            
            # Ensure consistent types across dataframes
            for i, df in enumerate(dfs):
                # Convert seqname/chrom to string if it exists
                chrom_col = 'seqname' if 'seqname' in df.columns else 'chrom'
                if chrom_col in df.columns:
                    if verbosity >= 3:
                        print(f"[debug] Converting {chrom_col} to string in dataframe {i+1}")
                    df[chrom_col] = df[chrom_col].astype(str)
            
            final_df = pd.concat(dfs, ignore_index=True)
            if verbosity >= 2:
                print(f"[debug] Combined {len(dfs)} dataframes, final shape: {final_df.shape}")
            return final_df
        else:
            # Polars approach
            import polars as pl
            # Convert all dataframes to have consistent types
            for i, df in enumerate(dfs):
                # Ensure seqname/chrom is always string type
                chrom_col = 'seqname' if 'seqname' in df.columns else 'chrom'
                if chrom_col in df.columns:
                    if verbosity >= 3:
                        print(f"[debug] Ensuring {chrom_col} is string type in dataframe {i+1}")
                    df = df.with_columns(pl.col(chrom_col).cast(pl.Utf8))
                dfs[i] = df
            
            final_df = pl.concat(dfs, how="vertical")
            if verbosity >= 2:
                print(f"[debug] Combined {len(dfs)} dataframes, final shape: {final_df.shape}")
            return final_df
    except Exception as e:
        if verbosity >= 1:
            print(f"[error] Failed to concatenate dataframes: {str(e)}")
        
        # Fallback to just return the first dataframe
        if dfs:
            if verbosity >= 1:
                print(f"[warning] Returning only data from the first dataframe")
            return dfs[0]
        return None
