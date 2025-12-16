"""
Data preparation utilities for splice site prediction workflows.

This module contains functions for preparing genomic data for splice site prediction,
including annotation extraction, sequence loading, and chromosome optimization.
"""

import os
import re
import pandas as pd
import polars as pl
from typing import Dict, List, Optional, Tuple, Union, Set, Any

from meta_spliceai.splice_engine.utils_fs import read_splice_sites
from meta_spliceai.splice_engine.utils_doc import (
    print_emphasized, 
    print_with_indent
)

from meta_spliceai.splice_engine.meta_models.core.analyzer import Analyzer

from meta_spliceai.splice_engine.meta_models.utils.sequence_utils import (
    read_sequence_file,
    combine_sequence_dataframes
)

from meta_spliceai.splice_engine.meta_models.utils.chrom_utils import (
    determine_target_chromosomes, 
    normalize_chromosome_names
)

from meta_spliceai.splice_engine.meta_models.openspliceai_adapter.aligned_splice_extractor import (
    AlignedSpliceExtractor
)

from meta_spliceai.splice_engine.meta_models.utils.splice_utils import (
    prepare_splice_site_adjustments
)

from meta_spliceai.splice_engine.meta_models.utils.dataframe_utils import (
    is_pandas_dataframe,
    is_dataframe_empty,
    filter_dataframe,
    get_row_count,
    get_shape,
    get_first_row,
    has_column
)

from meta_spliceai.splice_engine.meta_models.workflows.sequence_data_utils import (
    _process_gene_sequences,
    _process_transcript_sequences,
    _load_split_sequence_files
)


def prepare_gene_annotations(
    local_dir: str, 
    gtf_file: str, 
    do_extract: bool = False,
    output_filename: str = "annotations_all_transcripts.tsv",
    use_shared_db: bool = True,
    target_chromosomes: Optional[List[str]] = None,
    separator: str = "\t",
    verbosity: int = 1
) -> Dict[str, Any]:
    """
    Prepare gene annotations, from GTF file, for splice site prediction.
    
    This function extracts gene annotations from a GTF file if requested,
    and loads the annotations from a file if it exists.
    
    Parameters
    ----------
    local_dir : str
        Directory to store annotation files
    gtf_file : str
        Path to GTF file
    do_extract : bool, default=False
        Whether to extract annotations from GTF file
    output_filename : str, default="annotations_all_transcripts.tsv"
        Name of the output file for annotations
    use_shared_db : bool, default=True
        Whether to use the shared database in the parent directory of data/ensembl
    target_chromosomes : Optional[List[str]], default=None
        List of chromosomes to filter annotations to, handles both "chr1" and "1" formats
    separator : str, default="\t"
        Separator to use for output files
    verbosity : int, default=1
        Verbosity level
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing paths and dataframes for annotations

    Notes
    -----
    Annotation dataframe columns: 
    ["chrom", "start", "end", "strand", "feature", "gene_id", "transcript_id"]
    """
    result = {
        'success': False,
        'annotation_file': None,
        'annotation_df': None
    }
    
    # CRITICAL: annotations.db contains coordinates (start, end) and is BUILD-SPECIFIC
    # It should ALWAYS be in local_dir (build-specific directory), not shared_dir
    # The "use_shared_db" flag is misleading - it was intended for sharing across analyses
    # within the same build, but annotations.db should never be shared across builds.
    
    # Determine the output file path
    output_file = os.path.join(local_dir, output_filename)
    
    # ALWAYS use build-specific directory for annotations.db
    db_file = os.path.join(local_dir, 'annotations.db')
    
    # Extract gene annotations if requested
    if do_extract:
        from meta_spliceai.splice_engine.extract_genomic_features import extract_annotations
        if verbosity >= 1:
            print_emphasized("[action] Extract gene annotations...")
            print(f"[info] Using build-specific annotations.db: {db_file}")
        
        extract_annotations(gtf_file, db_file=db_file, output_file=output_file, sep=separator)
        result['annotation_file'] = output_file
    
    # If not extracting, check if the build-specific file exists
    elif not do_extract:
        if os.path.exists(output_file):
            if verbosity >= 1:
                print(f"[info] Using existing annotations file: {output_file}")
            result['annotation_file'] = output_file
        else:
            if verbosity >= 0:
                print(f"[warning] Annotations file not found: {output_file}")
                print("[warning] Setting do_extract=True to generate annotations")
            
            # Force extraction since we need the file
            from meta_spliceai.splice_engine.extract_genomic_features import extract_annotations
            if verbosity >= 1:
                print_emphasized("[action] Extract gene annotations...")
                print(f"[info] Using build-specific annotations.db: {db_file}")
            
            extract_annotations(gtf_file, db_file=db_file, output_file=output_file, sep=separator)
            result['annotation_file'] = output_file
    
    # Test loading annotations - use the same file that was created/copied
    if os.path.exists(output_file):
        if verbosity >= 1:
            print(f"[info] Loading transcript annotations from: {output_file}")
            # Transcript annotations include exons, CDS, 5'UTR, and 3'UTR, etc. 
        
        # Determine file format based on extension
        file_ext = os.path.splitext(output_file)[1].lower()
        if file_ext == '.csv':
            sep = ','
        else:  # Default to tab for .tsv or any other format
            sep = separator
            
        # Add dtype specification to avoid mixed type warnings
        # Set chromosome as string type to avoid dtype warnings
        annot_df = pd.read_csv(
            output_file, 
            sep=sep,
            low_memory=False, 
            dtype={'chrom': str}
        )
        # NOTE: low_memory=False - This prevents pandas from chunking the file and 
        #       potentially inferring different data types for different chunks
        
        # Filter by chromosomes if specified
        if target_chromosomes:
            # Normalize chromosome names
            normalized_chromosomes = normalize_chromosome_names(target_chromosomes)
            if verbosity >= 1:
                print(f"[info] Filtering annotations to chromosomes: {normalized_chromosomes}")
                
            # Filter the DataFrame
            original_count = len(annot_df)
            annot_df = annot_df[annot_df['chrom'].isin(normalized_chromosomes)]
            filtered_count = len(annot_df)
            
            if verbosity >= 1:
                print(f"[info] Filtered annotations from {original_count} to {filtered_count} rows")
                
        if verbosity >= 1:
            print(annot_df.head())
        
        result['annotation_df'] = annot_df
        result['annotation_file'] = output_file
        result['success'] = True
    else:
        if verbosity >= 1:
            print(f"[warning] Annotation file not found: {output_file}")
    
    return result


def prepare_splice_site_annotations(
    local_dir: str, 
    gtf_file: str, 
    do_extract: bool = True,
    output_filename: str = "splice_sites_enhanced.tsv",
    gene_annotations_df: Optional[Union[pd.DataFrame, pl.DataFrame]] = None,
    consensus_window: int = 2,
    separator: str = '\t',
    use_shared_db: bool = True,
    target_chromosomes: Optional[List[str]] = None,
    fasta_file: Optional[str] = None,
    verbosity: int = 1
) -> Dict[str, Any]:
    """
    Extract and prepare splice site annotations.
    
    Parameters
    ----------
    local_dir : str
        Directory to store extracted splice site annotations
    gtf_file : str
        Path to GTF file
    do_extract : bool, optional
        Whether to extract annotations from GTF file, by default True
    output_filename : str, optional
        Name of the output file, by default "splice_sites.tsv"
    gene_annotations_df : Optional[Union[pd.DataFrame, pl.DataFrame]], optional
        Pre-loaded gene annotations dataframe, useful for subsetting, by default None
    consensus_window : int, optional
        Window size for splice site consensus, by default 2
    separator : str, optional
        Separator for annotation files, by default '\t'
    use_shared_db : bool, default=True
        Whether to use the shared database in the parent directory of data/ensembl
    target_chromosomes : Optional[List[str]], default=None
        List of chromosomes to filter splice sites to, handles both "chr1" and "1" formats
    fasta_file : Optional[str], default=None
        Path to FASTA genome file, required for OpenSpliceAI fallback when splice_sites.tsv is missing
    verbosity : int, optional
        Verbosity level, by default 1
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing path and dataframe for splice site annotations
        
    Notes
    -----
    When providing gene_annotations_df, the function will filter the full splice site
    annotations to only include genes in the provided dataframe.
    """
    result = {
        'success': False,
        'splice_sites_file': None,
        'splice_sites_df': None
    }
    
    if verbosity >= 1:
        print_emphasized("[data] Loading splice site annotations...")
    
    # Default file extension format for splice site annotations
    ss_format = 'tsv'  # Maintain TSV format for splice site annotations
    
    # CRITICAL: Splice sites are BUILD-SPECIFIC, not shared across builds!
    # Only annotations.db can be shared (transcript metadata without coordinates)
    # Splice sites contain coordinates that differ between GRCh37 and GRCh38
    
    # Determine paths for truly shared resources (annotations.db only)
    if use_shared_db:
        # Use the parent directory of the Analyzer's eval_dir for shared resources
        shared_dir = Analyzer.shared_dir  # os.path.dirname(Analyzer.eval_dir)
        shared_db_file = os.path.join(shared_dir, 'annotations.db')
        
        if verbosity >= 2:
            print(f"[info] Using shared resources directory for annotations.db: {shared_dir}")
    
    # Use specified output filename
    if not output_filename.endswith(f".{ss_format}"):
        output_filename = f"{output_filename}.{ss_format}"
    
    # CRITICAL FIX: Splice sites are ALWAYS in local_dir (build-specific)
    # They contain coordinates that differ between genome builds
    splice_sites_file_path = os.path.join(local_dir, output_filename)
    result['splice_sites_file'] = splice_sites_file_path
    
    # Determine if we need to generate full splice sites
    # ALWAYS use local_dir for splice sites (build-specific)
    # Use splice_sites_enhanced.tsv as the canonical complete file
    full_splice_sites_path = os.path.join(local_dir, f"splice_sites_enhanced.{ss_format}")
    need_full_extraction = do_extract and not os.path.exists(full_splice_sites_path)
    
    # CRITICAL: Do NOT auto-enable extraction during inference!
    # If do_extract=False was explicitly set (e.g., during inference), respect it.
    # Only auto-enable if the caller didn't explicitly disable extraction.
    if not do_extract and use_shared_db and not os.path.exists(full_splice_sites_path):
        if verbosity >= 0:
            print(f"[error] Shared splice sites file not found: {full_splice_sites_path}")
            print("[error] do_extract=False was set, but required file is missing.")
            print("[error] This file should have been created during the base model pass.")
            print("[error] Please run the base model workflow first to generate genomic datasets.")
        
        # Return error instead of auto-enabling extraction
        result['success'] = False
        result['error'] = f"Required splice sites file not found: {full_splice_sites_path}"
        return result
    
    # Get list of target genes if we have a subset
    target_genes = None
    if gene_annotations_df is not None:
        # Extract unique gene IDs from the annotations
        if isinstance(gene_annotations_df, pd.DataFrame):
            target_genes = set(gene_annotations_df['gene_id'].unique())
        else:  # polars DataFrame
            target_genes = set(gene_annotations_df['gene_id'].unique().to_list())
            
        if verbosity >= 1:
            print(f"[info] Will filter splice sites to {len(target_genes)} target genes")
    
    # Extract full splice sites if needed
    if need_full_extraction:
        from meta_spliceai.splice_engine.extract_genomic_features import extract_splice_sites_workflow
        if verbosity >= 1:
            print_emphasized("[action] Extracting full splice sites (this may take a while)...")
        
        # Make sure the directory exists
        if not os.path.exists(local_dir):
            os.makedirs(local_dir, exist_ok=True)
            if verbosity >= 1:
                print(f"[info] Created build-specific directory: {local_dir}")
        
        # CRITICAL: ALWAYS extract splice sites to local_dir (build-specific)
        # Splice sites contain coordinates that differ between genome builds!
        full_splice_sites_path = extract_splice_sites_workflow(
            data_prefix=local_dir, 
            gtf_file=gtf_file, 
            consensus_window=consensus_window
        )
            
        if verbosity >= 1:
            print(f"[info] Full splice sites extracted to: {full_splice_sites_path}")
    
    # Load the full splice sites
    if not os.path.exists(full_splice_sites_path):
        if verbosity >= 0:
            print(f"Warning: Full splice sites file not found at {full_splice_sites_path}")
            
        if not do_extract and fasta_file is not None:
            # NEW: OpenSpliceAI fallback when splice_sites.tsv is missing
            if verbosity >= 1:
                print("[fallback] Using OpenSpliceAI workflow to generate splice sites...")
                print("[info] This leverages the AlignedSpliceExtractor for 100% equivalence with MetaSpliceAI")
            
            try:
                # Initialize AlignedSpliceExtractor with MetaSpliceAI coordinates
                extractor = AlignedSpliceExtractor(
                    coordinate_system="splicesurveyor",
                    enable_biotype_filtering=False,
                    verbosity=max(0, verbosity - 1)  # Reduce verbosity for cleaner output
                )
                
                # Extract splice sites with 100% MetaSpliceAI equivalence
                if verbosity >= 2:
                    print(f"[debug] Extracting splice sites using AlignedSpliceExtractor")
                    print(f"[debug] GTF file: {gtf_file}")
                    print(f"[debug] FASTA file: {fasta_file}")
                
                # Use schema adaptation for clean, systematic conversion
                splice_sites_df = extractor.extract_splice_sites(
                    gtf_file=gtf_file,
                    fasta_file=fasta_file,
                    gene_ids=None,  # Extract all genes
                    output_format="dataframe",
                    apply_schema_adaptation=True  # Use systematic schema adapter
                )
                
                # Apply chromosome filtering if specified
                if target_chromosomes:
                    normalized_chromosomes = normalize_chromosome_names(target_chromosomes)
                    if verbosity >= 1:
                        print(f"[info] Filtering splice sites to chromosomes: {normalized_chromosomes}")
                    original_count = len(splice_sites_df)
                    splice_sites_df = splice_sites_df[splice_sites_df['chrom'].isin(normalized_chromosomes)]
                    filtered_count = len(splice_sites_df)
                    if verbosity >= 1:
                        print(f"[info] Filtered splice sites from {original_count} to {filtered_count} rows by chromosomes")
                
                # Save to the expected location for future use
                if verbosity >= 1:
                    print(f"[action] Saving OpenSpliceAI-generated splice sites to: {full_splice_sites_path}")
                
                # Ensure directory exists
                os.makedirs(os.path.dirname(full_splice_sites_path), exist_ok=True)
                
                # Save in TSV format matching MetaSpliceAI's native format
                splice_sites_df.to_csv(full_splice_sites_path, sep=separator, index=False)
                
                if verbosity >= 1:
                    print(f"[success] Generated {len(splice_sites_df)} splice sites using OpenSpliceAI fallback")
                    print(f"[info] File saved to: {full_splice_sites_path}")
                
            except Exception as e:
                if verbosity >= 0:
                    print(f"Error: OpenSpliceAI fallback failed: {str(e)}")
                    print("Please set do_extract=True to generate splice sites using native MetaSpliceAI workflow.")
                return result
        
        elif not do_extract:
            if verbosity >= 0:
                print("Error: Cannot use OpenSpliceAI fallback without fasta_file parameter.")
                print("Please either:")
                print("  1. Set do_extract=True to generate splice sites, or")
                print("  2. Provide fasta_file parameter to enable OpenSpliceAI fallback")
            return result
        else:
            if verbosity >= 0:
                print("Error occurred during extraction. Check the logs for details.")
            return result
    
    # Load the full splice site annotations
    full_ss_annotations_df = read_splice_sites(full_splice_sites_path, separator=separator, dtypes=None)
    
    # Use the existing utility function from utils_df.py
    if full_ss_annotations_df is None or is_dataframe_empty(full_ss_annotations_df):
        if verbosity >= 0:
            print("Error: Failed to load splice site annotations.")
        return result
    
    if verbosity >= 2:
        print(f"[info] Full splice-site dataframe: shape={full_ss_annotations_df.shape}")
    # NOTE: Always loads the full splice site dataframe first, then 
    #       filters it to the subset of genes.
    
    # Initialize the splice site annotations dataframe
    ss_annotations_df = full_ss_annotations_df
    
    # Apply filtering in sequence, first by target genes if specified
    if target_genes:
        if verbosity >= 1:
            print(f"[action] Filtering splice sites to {len(target_genes)} target genes...")
        
        # Filter the data - handle both pandas and polars formats
        if isinstance(ss_annotations_df, pd.DataFrame):
            original_count = len(ss_annotations_df)
            ss_annotations_df = ss_annotations_df[ss_annotations_df['gene_id'].isin(target_genes)]
            filtered_count = len(ss_annotations_df)
        else:  # polars DataFrame
            original_count = ss_annotations_df.shape[0]
            ss_annotations_df = ss_annotations_df.filter(
                pl.col("gene_id").is_in(target_genes)
            )
            filtered_count = ss_annotations_df.shape[0]
        
        if verbosity >= 1:
            print(f"[info] Filtered splice sites from {original_count} to {filtered_count} rows by genes")
    
    # Then filter by chromosomes if specified
    if target_chromosomes:
        # Normalize chromosome names
        normalized_chromosomes = normalize_chromosome_names(target_chromosomes)
        if verbosity >= 1:
            print(f"[info] Filtering splice sites to chromosomes: {normalized_chromosomes}")
            
        # Filter the DataFrame - handle both pandas and polars
        if isinstance(ss_annotations_df, pd.DataFrame):
            original_count = len(ss_annotations_df)
            ss_annotations_df = ss_annotations_df[ss_annotations_df['chrom'].isin(normalized_chromosomes)]
            filtered_count = len(ss_annotations_df)
        else:  # polars DataFrame
            original_count = ss_annotations_df.shape[0]
            ss_annotations_df = ss_annotations_df.filter(
                pl.col("chrom").is_in(normalized_chromosomes)
            )
            filtered_count = ss_annotations_df.shape[0]
            
        if verbosity >= 1:
            print(f"[info] Filtered splice sites from {original_count} to {filtered_count} rows by chromosomes")
    
    # CRITICAL: During inference (do_extract=False), NEVER write to disk
    # The inference workflow should only READ existing data, not modify it
    if do_extract:
        # Save the filtered subset to the output file if it's different from the full set
        if (target_genes or target_chromosomes) or output_filename != f"splice_sites.{ss_format}":
            if isinstance(ss_annotations_df, pl.DataFrame):
                ss_annotations_df.write_csv(splice_sites_file_path, separator=separator)
            else:
                ss_annotations_df.to_csv(splice_sites_file_path, sep=separator, index=False)
                
            if verbosity >= 1:
                print(f"[info] Saved filtered splice sites to: {splice_sites_file_path}")
        else:
            # If the output filename is different from the default, make a copy
            if output_filename != f"splice_sites.{ss_format}" and not os.path.exists(splice_sites_file_path):
                import shutil
                if not os.path.exists(os.path.dirname(splice_sites_file_path)):
                    os.makedirs(os.path.dirname(splice_sites_file_path), exist_ok=True)
                shutil.copy(full_splice_sites_path, splice_sites_file_path)
                
                if verbosity >= 1:
                    print(f"[info] Copied full splice sites to: {splice_sites_file_path}")
    else:
        # Inference mode: Just use the data in memory, don't write anything
        if verbosity >= 1:
            print(f"[info] Inference mode: Using filtered splice sites in memory only (not writing to disk)")
    
    if verbosity >= 1:
        print(f"[info] Final splice-site dataframe: shape={ss_annotations_df.shape}")
        print(ss_annotations_df.head())
    
    result['splice_sites_df'] = ss_annotations_df
    result['splice_sites_file'] = splice_sites_file_path
    result['success'] = True
    
    return result


def prepare_genomic_sequences(
    local_dir: str,
    gtf_file: str,
    genome_fasta: str,
    mode: str = 'gene',
    seq_type: str = 'full',
    do_extract: bool = True,
    chromosomes: Optional[List[str]] = None,
    genes: Optional[List[str]] = None,  
    test_mode: bool = False,
    seq_format: str = 'parquet',
    single_sequence_file: bool = False,
    force_overwrite: bool = False,
    verbosity: int = 1
) -> Dict[str, Any]:
    """
    Prepare genomic sequences for splice site prediction.
    
    Parameters
    ----------
    local_dir : str
        Directory to store output files
    gtf_file : str
        Path to GTF annotation file
    genome_fasta : str
        Path to genome FASTA file
    mode : str, default='gene'
        Mode for sequence extraction ('gene' or 'transcript')
    seq_type : str, default='full'
        Type of gene sequences to extract ('full' or 'minmax')
    do_extract : bool, default=True
        Whether to extract sequences or use existing files
    chromosomes : List[str], optional
        List of chromosomes to include, if None all chromosomes are used
    single_sequence_file : bool, default=False
        Whether to extract a single sequence file for all chromosomes
    genes : List[str], optional
        List of genes to include (for future enhancement)
    test_mode : bool, default=False
        Whether to run in test mode
    seq_format : str, default='parquet'
        Format for sequence files
    force_overwrite : bool, default=False
        If True, re-extract sequences even when expected output files already exist.
    verbosity : int, default=1
        Verbosity level
        
    Returns
    -------
    Dict[str, Any]
        Dictionary with results including success status and file paths
    """
    result = {
        'success': False,
        'sequences_file': None,
        'sequences_df': None,
        'error': None
    }
    
    # Create output directory if needed
    os.makedirs(local_dir, exist_ok=True)

    standard_chroms = [str(i) for i in range(1, 23)] + ['X', 'Y']
    if chromosomes is None: 
        chromosomes = standard_chroms

    print(f"[info] do_extract: {do_extract}, force_overwrite: {force_overwrite}")
    print(f"[info] test_mode: {test_mode}")
    print(f"[info] chromosomes: {chromosomes}, genes: {genes}")
    
    try:
        # Process based on mode
        if mode == 'gene':
            seq_df_path, do_extract = _process_gene_sequences(
                local_dir=local_dir,
                gtf_file=gtf_file,
                genome_fasta=genome_fasta,
                seq_type=seq_type,
                seq_format=seq_format,
                chromosomes=chromosomes,
                do_extract=do_extract,
                force_overwrite=force_overwrite,
                single_sequence_file=single_sequence_file,
                verbosity=verbosity
            )
        else:  # mode == 'transcript'
            seq_df_path, do_extract = _process_transcript_sequences(
                local_dir=local_dir,
                gtf_file=gtf_file,
                genome_fasta=genome_fasta,
                seq_format=seq_format,
                chromosomes=chromosomes,
                do_extract=do_extract,
                force_overwrite=force_overwrite,
                single_sequence_file=single_sequence_file,
                verbosity=verbosity
            )
        
        # Success case
        result['success'] = True
        
        # Store the main combined file path (may or may not exist depending on single_sequence_file)
        result['sequences_file'] = seq_df_path
        result['seq_format'] = seq_format
        
        # Store information about per-chromosome files if not using a single file
        if not single_sequence_file:
            # Find all chromosome-specific sequence files in the directory
            chr_file_pattern = f"gene_sequence_*.{seq_format}" if mode == 'gene' else f"tx_sequence_*.{seq_format}"
            
            # Get list of chromosome-specific files
            import glob
            chr_sequence_files = glob.glob(os.path.join(local_dir, chr_file_pattern))
            
            # Add to result dict
            result['chr_sequence_files'] = chr_sequence_files
            
            if verbosity >= 2:
                print(f"[debug] Found {len(chr_sequence_files)} chromosome-specific sequence files")
                if verbosity >= 3:
                    for f in chr_sequence_files:
                        print(f"[debug]   - {os.path.basename(f)}")
        
        # Try to load the sequences dataframe
        try:
            # For gene filtering optimization
            target_chromosomes = None
            
            # If genes are specified, try to determine which chromosomes they're on
            if genes and len(genes) > 0:
                if verbosity >= 1:
                    print(f"[info] Optimizing sequence loading for {len(genes)} target genes")
                
                # Use previously implemented function to find chromosomes for target genes
                result = determine_target_chromosomes(
                    local_dir=local_dir,
                    gtf_file=gtf_file,
                    target_genes=genes,
                    chromosomes=chromosomes,
                    test_mode=test_mode,
                    verbosity=verbosity
                )
                # NOTE: Also caches the gene-chromosome mapping in a file for future use
                
                # Extract the chromosomes from the result
                target_chromosomes = result.get('chromosomes')
                
                if verbosity >= 1 and target_chromosomes:
                    print(f"[info] Target genes found on chromosomes: {target_chromosomes}")
            else:
                # No genes specified, use all chromosomes as before
                target_chromosomes = chromosomes
            
            print(f"[info] Target chromosomes: {target_chromosomes}, test_mode: {test_mode}")
            
            # Determine which file(s) to load
            sequences_df = None  # ensure defined
            try:
                if single_sequence_file:
                    # Load a single combined file if requested
                    if verbosity >= 2:
                        print(f"[debug] Loading single sequence file: {seq_df_path}")
                    
                    if os.path.exists(seq_df_path):
                        if seq_format == 'parquet':
                            sequences_df = pq.read_table(seq_df_path).to_pandas()
                        else:  # tsv/csv
                            sequences_df = pl.read_csv(seq_df_path, separator='\t' if seq_format == 'tsv' else ',')
                        
                        if verbosity >= 1:
                            print(f"[info] Loaded single sequence file with {len(sequences_df)} records")
                    else:
                        if verbosity >= 1:
                            print(f"[warning] Single sequence file not found: {seq_df_path}")
                else:
                    # Default: load per-chromosome files
                    if verbosity >= 2:
                        print(f"[debug] Loading per-chromosome sequence files")
                    
                    # Make sure we have target chromosomes defined
                    if target_chromosomes is None:
                        if verbosity >= 1:
                            print("[warning] No target chromosomes specified, using standard set")
                        target_chromosomes = standard_chroms
                    
                    # Load sequences from per-chromosome files
                    sequences_df = _load_split_sequence_files(target_chromosomes, local_dir, seq_type, seq_format, verbosity)
                    
                    if sequences_df is None or len(sequences_df) == 0:
                        if verbosity >= 1:
                            print(f"[warning] Failed to load any chromosome sequence files")
            except Exception as e:
                if verbosity >= 1:
                    print(f"[error] Failed to load sequence data: {str(e)}")
                
            # If we have sequence data, apply any requested filters
            if sequences_df is not None:
                # Log dataframe type for debugging
                if verbosity >= 2:
                    print(f"[debug] DataFrame type: {'Pandas' if is_pandas_dataframe(sequences_df) else 'Polars'}")
                
                # Apply chromosome filtering if requested
                if chromosomes and not test_mode:
                    normalized_chroms = normalize_chromosome_names(chromosomes)
                    if verbosity >= 2:
                        print(f"[info] Filtering sequences by chromosomes: {normalized_chroms}")
                    
                    chrom_col = 'seqname' if has_column(sequences_df, 'seqname') else 'chrom'
                    original_count = get_row_count(sequences_df)

                    # Filter sequence dataframe by chromosomes
                    sequences_df = filter_dataframe(sequences_df, chrom_col, normalized_chroms)
                    
                    if verbosity >= 1:
                        print(f"[info] Filtered to {get_row_count(sequences_df)} sequences on target chromosomes")
                
                # Apply gene filtering if specified
                if genes:
                    original_count = get_row_count(sequences_df)
                    
                    # Try filtering by gene_id first (if genes are Ensembl IDs)
                    if has_column(sequences_df, 'gene_id'):
                        sequences_df_by_id = filter_dataframe(sequences_df, 'gene_id', genes)
                        
                        # If no matches by gene_id, try gene_name (if genes are symbols like BRCA1)
                        if get_row_count(sequences_df_by_id) == 0 and has_column(sequences_df, 'gene_name'):
                            if verbosity >= 1:
                                print(f"[debug] No matches by gene_id, trying gene_name column")
                            sequences_df = filter_dataframe(sequences_df, 'gene_name', genes)
                        else:
                            sequences_df = sequences_df_by_id
                    elif has_column(sequences_df, 'gene_name'):
                        # Only gene_name column available
                        sequences_df = filter_dataframe(sequences_df, 'gene_name', genes)
                    
                    if verbosity >= 1:
                        filtered_count = get_row_count(sequences_df)
                        print(f"[info] Filtered to {filtered_count} sequences for {len(genes)} target genes (from {original_count})")
                        if filtered_count == 0:
                            print(f"[warning] No sequences found for target genes! Check if gene names/IDs match annotation.")
                
                # Add debug info about the final state of sequences_df
                if verbosity >= 2:
                    print(f"[debug] Final sequence dataframe shape: {get_shape(sequences_df) if sequences_df is not None else 'None'}")
                    
                    if sequences_df is not None and get_row_count(sequences_df) > 0:
                        print(f"[debug] Dataframe columns: {list(sequences_df.columns)}")
                        print(f"[debug] First row sample: {get_first_row(sequences_df)}")
            else:
                if verbosity >= 1:
                    print("[warning] No sequence data available after loading")
            
            # Add debug info about result dictionary
            if verbosity >= 2:
                print(f"[debug] Result dictionary keys before assigning sequences_df: {list(result.keys())}")
            
            result['sequences_df'] = sequences_df
            # Mark the operation as successful if we reach this point
            result['success'] = True
            
            if verbosity >= 2:
                print(f"[debug] Result dictionary keys after assigning sequences_df: {list(result.keys())}")
                print(f"[debug] Returning result from sequence loading block")
        except Exception as e:
            if verbosity >= 1:
                print(f"[error] Failed to load sequences dataframe: {str(e)}")
                import traceback
                print(f"[debug] Exception traceback: {traceback.format_exc()}")
        
        return result
    
    except Exception as e:
        result['error'] = str(e)
        if verbosity >= 1:
            print(f"[error] Failed to prepare genomic sequences: {str(e)}")
            import traceback
            print(f"[debug] Outer exception traceback: {traceback.format_exc()}")
        return result


def handle_overlapping_genes(
    local_dir: str,
    gtf_file: str,
    do_find: bool = True,
    min_exons: int = 2,
    filter_valid_splice_sites: bool = True,
    separator: str = '\t',
    output_format: str = 'pd',
    verbosity: int = 1,
    target_chromosomes: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Find or load overlapping gene information.
    
    Parameters
    ----------
    local_dir : str
        Directory to store output files
    gtf_file : str
        Path to GTF file
    do_find : bool, default=True
        Whether to find overlapping genes if not already found
    min_exons : int, default=2
        Minimum number of exons for a gene to be considered
    filter_valid_splice_sites : bool, default=True
        Whether to filter genes for valid splice sites
    separator : str, default='\t'
        Separator for output TSV file
    output_format : str, default='pd'
        Output format ('pd' for pandas, 'pl' for polars)
    verbosity : int, default=1
        Verbosity level
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing overlapping gene information
    """
    result = {
        'success': False,
        'overlapping_file': None,
        'overlapping_df': None
    }
    
    overlapping_gene_file_path = os.path.join(local_dir, "overlapping_gene_counts.tsv")
    result['overlapping_file'] = overlapping_gene_file_path
    
    # Use SpliceAnalyzer to handle overlapping gene metadata
    if do_find or not os.path.exists(overlapping_gene_file_path):
        from meta_spliceai.splice_engine.meta_models.core.analyzers.splice import SpliceAnalyzer

        if verbosity >= 1:
            print_emphasized("[action] Retrieving overlapping gene metadata...")
        
        # Create a SpliceAnalyzer instance with local_dir as its data directory
        # This makes path_to_overlapping_gene_metadata return the path we want
        analyzer = SpliceAnalyzer(data_dir=local_dir)
        
        # Retrieve overlapping gene metadata
        overlapping_df = analyzer.retrieve_overlapping_gene_metadata(
            min_exons=min_exons,
            filter_valid_splice_sites=filter_valid_splice_sites,
            output_format='dataframe',
            to_pandas=(output_format == 'pd'),
            verbose=verbosity,
            target_chromosomes=target_chromosomes  # Add chromosome filtering
        )
        
        # Save the dataframe to the expected location if needed
        if not os.path.exists(overlapping_gene_file_path):
            if isinstance(overlapping_df, pd.DataFrame):
                overlapping_df.to_csv(overlapping_gene_file_path, sep=separator, index=False)
            else:
                # Assume it's a polars dataframe
                overlapping_df.write_csv(overlapping_gene_file_path, separator=separator)
                
            if verbosity >= 1:
                print(f"[info] Saved overlapping gene metadata to: {overlapping_gene_file_path}")
        
        result['overlapping_df'] = overlapping_df
        result['success'] = True
    else:
        # Load overlapping gene information from existing file
        if verbosity >= 1:
            print(f"[info] Loading existing overlapping gene file: {overlapping_gene_file_path}")
            
        if output_format == 'pd':
            overlapping_df = pd.read_csv(overlapping_gene_file_path, sep=separator)
        else:
            # Use polars with proper schema overrides to handle chromosome names
            overlapping_df = pl.read_csv(
                overlapping_gene_file_path, 
                separator=separator,
                schema_overrides={
                    # Ensure chromosome names are always treated as strings
                    "chrom": pl.Utf8,
                    "seqname": pl.Utf8
                },
                ignore_errors=True  # Handle other potential type issues gracefully
            )
            
        if verbosity >= 1:
            print(f"[info] Overlapping genes dataframe: shape={overlapping_df.shape}")
            if verbosity >= 2:
                print(overlapping_df.head())
        
        result['overlapping_df'] = overlapping_df
        result['success'] = True
    
    return result


def load_spliceai_models(verbosity: int = 1) -> Dict[str, Any]:
    """
    Load SpliceAI models.
    
    Parameters
    ----------
    verbosity : int, optional
        Verbosity level, by default 1
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing loaded models and status
    """
    result = {
        'success': False,
        'models': None,
        'error': None
    }
    
    if verbosity >= 1:
        print_emphasized("[action] Loading SpliceAI models...")
    
    try:
        from keras.models import load_model
        from meta_spliceai.splice_engine.run_spliceai_workflow import load_spliceai_models
        
        models = load_spliceai_models()
        if verbosity >= 1:
            print_with_indent(f"[info] SpliceAI models loaded successfully (n={len(models)})", indent_level=1)
        
        result['models'] = models
        result['success'] = True
    except Exception as e:
        if verbosity >= 1:
            print_with_indent(f"[error] Failed to load SpliceAI models: {str(e)}", indent_level=1)
        result['error'] = str(e)
    
    return result
