"""
Helper functions for genomic sequence data processing, loading, and extraction.
"""

import os
import polars as pl
from typing import Dict, List, Optional, Tuple, Any

from meta_spliceai.splice_engine.utils_doc import (
    print_emphasized, 
    print_with_indent
)

from meta_spliceai.splice_engine.meta_models.utils.sequence_utils import (
    read_sequence_file,
    combine_sequence_dataframes
)

from meta_spliceai.splice_engine.extract_genomic_features import (
    gene_sequence_retrieval_workflow,
    transcript_sequence_retrieval_workflow
)


def _process_gene_sequences(
    local_dir: str,
    gtf_file: str,
    genome_fasta: str,
    seq_type: str,
    seq_format: str,
    chromosomes: Optional[List[str]],
    do_extract: bool,
    force_overwrite: bool,
    single_sequence_file: bool,
    verbosity: int
) -> Tuple[str, bool]:
    """
    Helper function to process gene sequence extraction or loading.
    
    Parameters
    ----------
    local_dir : str
        Directory for output files
    gtf_file : str
        Path to GTF annotation file
    genome_fasta : str
        Path to genome FASTA file
    seq_type : str
        Type of sequence to extract ('full' or 'minmax')
    seq_format : str
        File format to use ('parquet', 'tsv', 'csv')
    chromosomes : List[str], optional
        Chromosomes to include
    do_extract : bool
        Whether to extract sequences or load existing files
    force_overwrite : bool
        Whether to overwrite existing files
    single_sequence_file : bool
        Whether to use a single combined file instead of per-chromosome files
    verbosity : int
        Verbosity level
        
    Returns
    -------
    Tuple[str, bool]
        Path to sequence file and updated do_extract flag
    """
    # Determine output file name
    if seq_type == 'minmax':
        output_file = f"gene_sequence_minmax.{seq_format}"
    else:
        output_file = f"gene_sequence.{seq_format}"
    
    seq_df_path = os.path.join(local_dir, output_file)
    
    # Short-circuit extraction if files exist and not forcing overwrite
    if do_extract and not force_overwrite:
        files_present = False
        if single_sequence_file:
            # Check for a single combined file
            files_present = os.path.exists(seq_df_path)
            if verbosity >= 1:
                print(f"[info] Sequence file exists: {files_present}")
        elif chromosomes:
            # Check per-chromosome files
            files_present = True
            for chrom in chromosomes:
                chrom_str = chrom if chrom.startswith('chr') else chrom.lstrip('chr')
                fname = (
                    f"gene_sequence_minmax_{chrom_str}.{seq_format}"
                    if seq_type == 'minmax'
                    else f"gene_sequence_{chrom_str}.{seq_format}"
                )
                if not os.path.exists(os.path.join(local_dir, fname)):
                    files_present = False
                    break
        else: 
            # Should not reach here unless chromosomes is explicitly set to an empty list
            raise ValueError("No chromosomes specified and no standard set defined")
        
        if files_present:
            if verbosity >= 1:
                print_with_indent("[skip] Sequence files already present; use force_overwrite=True to regenerate", indent_level=1)
            do_extract = False  # cancel extraction
    
    # Extract sequences if still requested
    if do_extract:
        # Extract all sequences first (gene_sequence_retrieval_workflow doesn't support chromosome filtering)
        if verbosity >= 1:
            print_with_indent(f"Extracting {seq_type} gene sequences for all chromosomes...", indent_level=1)
            print_with_indent(f"Note: Will filter to chromosomes {chromosomes} after extraction", indent_level=1)
        
        # Process all chromosomes at once
        gene_sequence_retrieval_workflow(
            gtf_file, genome_fasta, gene_tx_map=None,
            output_file=seq_df_path, mode=seq_type, format=seq_format
        )
        # NOTE: gene_tx_map is used for mapping selected transcripts to genes; 
        #       not used in gene-level analysis
    
    return seq_df_path, do_extract


def _process_transcript_sequences(
    local_dir: str,
    gtf_file: str,
    genome_fasta: str,
    seq_format: str,
    chromosomes: Optional[List[str]],
    do_extract: bool,
    force_overwrite: bool,
    single_sequence_file: bool,
    verbosity: int
) -> Tuple[str, bool]:
    """
    Helper function to process transcript sequence extraction or loading.
    
    Parameters
    ----------
    local_dir : str
        Directory for output files
    gtf_file : str
        Path to GTF annotation file
    genome_fasta : str
        Path to genome FASTA file
    seq_format : str
        File format to use ('parquet', 'tsv', 'csv')
    chromosomes : List[str], optional
        Chromosomes to include
    do_extract : bool
        Whether to extract sequences or load existing files
    force_overwrite : bool
        Whether to overwrite existing files
    single_sequence_file : bool
        Whether to use a single combined file instead of per-chromosome files
    verbosity : int
        Verbosity level
        
    Returns
    -------
    Tuple[str, bool]
        Path to sequence file and updated do_extract flag
    """
    # No seq_type for transcript mode
    seq_df_path = os.path.join(local_dir, f"tx_sequence.{seq_format}")
    
    # Short-circuit extraction if files exist and not forcing overwrite
    if do_extract and not force_overwrite:
        files_present = False
        if single_sequence_file:
            # Check for a single combined file
            files_present = os.path.exists(seq_df_path)
            if verbosity >= 1:
                print(f"[info] Transcript sequence file exists: {files_present}")
        elif chromosomes:
            # Check per-chromosome files
            files_present = True
            for chrom in chromosomes:
                chrom_str = chrom if chrom.startswith('chr') else chrom.lstrip('chr')
                fname = f"tx_sequence_{chrom_str}.{seq_format}"
                if not os.path.exists(os.path.join(local_dir, fname)):
                    files_present = False
                    break
        else:
            # Should not reach here unless chromosomes is explicitly set to an empty list
            raise ValueError("No chromosomes specified for per-chromosome file check")
        
        if files_present:
            if verbosity >= 1:
                print_with_indent("[skip] Transcript sequence files already present; use force_overwrite=True to regenerate", indent_level=1)
            do_extract = False  # cancel extraction
    
    if do_extract:
        if verbosity >= 1:
            print_with_indent("Extracting transcript sequences...", indent_level=1)
            print_with_indent(f"Note: Will filter to chromosomes {chromosomes} after extraction", indent_level=1)
        transcript_sequence_retrieval_workflow(
            gtf_file, genome_fasta, gene_tx_map=None, 
            output_file=seq_df_path, format=seq_format
        )
        # NOTE: transcript_sequence_retrieval_workflow saves individual chromosome files
        # similar to gene_sequence_retrieval_workflow
    
    return seq_df_path, do_extract


def _load_split_sequence_files(
    chromosomes: List[str],
    local_dir: str,
    seq_type: str,
    seq_format: str,
    verbosity: int = 1,
):
    """Load per-chromosome sequence files (parquet or delimited text).
    
    Parameters
    ----------
    chromosomes : List[str]
        Target chromosome identifiers (with or without 'chr' prefix).
    local_dir : str
        Directory containing sequence files.
    seq_type : str
        Either 'full' or 'minmax'. Determines file naming pattern.
    seq_format : str
        File extension without dot (e.g. 'parquet', 'tsv').
    verbosity : int, default=1
        Verbosity level for logging.
    
    Returns
    -------
    DataFrame or None
        Concatenated dataframe of all loaded chromosome files or ``None`` if nothing loaded.
    """
    if not chromosomes:
        return None

    dfs = []
    seen = set()
    for chrom in chromosomes:
        # Normalize chromosome label (remove optional 'chr' prefix)
        chrom_str = chrom[3:] if chrom.lower().startswith("chr") else chrom
        if chrom_str in seen:
            continue
        seen.add(chrom_str)

        # Base filename
        base_fname = (
            f"gene_sequence_minmax_{chrom_str}.{seq_format}"
            if seq_type == "minmax"
            else f"gene_sequence_{chrom_str}.{seq_format}"
        )

        if verbosity >= 2:
            print(f"[debug] Looking for chromosome file with basename: {base_fname}")

        candidate_paths = [
            os.path.join(local_dir, base_fname),
            os.path.join(local_dir, f"gene_sequence_{seq_type}_{chrom_str}.{seq_format}"),
            os.path.join(local_dir, f"gene_sequence_{chrom_str}_{seq_type}.{seq_format}"),
            os.path.join(local_dir, f"{seq_type}_gene_sequence_{chrom_str}.{seq_format}"),
        ]

        # if verbosity >= 2:
        #     for i, path in enumerate(candidate_paths):
        #         exists = os.path.exists(path)
        #         print(f"[debug] Candidate path {i+1}: {path} (exists: {exists})")

        for path in candidate_paths:
            if not os.path.exists(path):
                continue
                
            # First try reading based on specified format
            try:
                # Use the helper function to read the file
                df = read_sequence_file(
                    path=path, 
                    format=seq_format, 
                    try_alternatives=True,
                    verbosity=verbosity
                )
                
                if df is not None:
                    dfs.append(df)
            except Exception as primary_err:
                # If the specified format failed, try detecting actual format
                if verbosity >= 2:
                    print(f"[debug] Failed to load {path} as {seq_format}: {str(primary_err)}")
                    
                try:
                    # Check if it's actually a parquet file with wrong extension
                    if seq_format != "parquet":
                        if verbosity >= 2:
                            print(f"[debug] Trying parquet format...")
                        import pyarrow.parquet as pq
                        df = pq.read_table(path).to_pandas()
                        if verbosity >= 1:
                            print(f"[warning] File {path} is in parquet format despite {seq_format} extension")
                        dfs.append(df)
                    else:
                        # Not a parquet file, final attempt with common text formats
                        if seq_format == "parquet":
                            try:
                                if verbosity >= 2:
                                    print(f"[debug] Trying TSV format...")
                                df = pl.read_csv(path, separator="\t", encoding="utf8-lossy")
                                if verbosity >= 1:
                                    print(f"[warning] File {path} is in TSV format despite parquet extension")
                                dfs.append(df)
                            except Exception:
                                if verbosity >= 1:
                                    print(f"[warning] Failed to load {path} in any recognized format")
                except Exception:
                    pass
            
        # End for each candidate path
        
        if not dfs:
            if verbosity >= 1:
                print("[warning] No chromosome files found for the specified chromosomes")
            return None

        if verbosity >= 2:
            print(f"[info] Loaded data from {len(dfs)} chromosome files")

    # End for each chromosome file loading
        
    try:
        # Use the helper function to combine the dataframes
        use_polars = seq_format != "parquet"
        return combine_sequence_dataframes(
            dfs=dfs, 
            use_polars=use_polars,
            verbosity=verbosity
        )
    except Exception as e:
        if verbosity >= 1:
            print(f"[error] Failed to concatenate chromosome dataframes: {str(e)}")
            print("[info] Will attempt to return just the first dataframe")
        
        # Fallback to just return the first dataframe
        if dfs:
            if verbosity >= 1:
                print(f"[warning] Returning only data from the first chromosome")
            return dfs[0]
        return None


def extract_analysis_sequences(sequence_df, position_df, window_size=250, include_empty_entries=True, 
                               essential_columns_only=False, additional_columns=None, 
                               add_transcript_count=True, drop_transcript_id=True, 
                               resolve_prediction_conflicts=True,
                               position_id_mode='genomic',  # NEW: 'genomic', 'transcript', 'hybrid'
                               preserve_transcript_list=False,  # NEW: Keep transcript metadata
                               **kargs):
    """
    Extract sequences from genes based on positions for analysis.

    Parameters:
    - sequence_df (pl.DataFrame): Polars DataFrame with columns ['gene_id', 'sequence', 'strand'].
    - position_df (pl.DataFrame): Polars DataFrame with columns ['gene_id', 'position', ...].
                                All other columns in position_df will be retained in the output.
    - window_size (int): Size of the flanking sequence window around the position.
    - include_empty_entries (bool): If True, include entries for missing or invalid gene IDs/windows.
    - essential_columns_only (bool): If True, only include essential columns to identify a splice site context.
                                  If False, include all columns from position_df in the output. Default is False for meta-models.
    - additional_columns (List[str]): Additional columns to include beyond the essential ones.
    - add_transcript_count : bool, optional
        If True (default), add a 'transcript_count' column that indicates how many 
        transcripts are associated with each position
    - drop_transcript_id : bool, optional
        If True, remove the 'transcript_id' column from the output to avoid sequence duplication
        when multiple transcripts share the same splice site position. Default is True.
    - resolve_prediction_conflicts : bool, optional
        If True, resolve conflicts when the same position has multiple prediction types by keeping
        only the highest priority prediction type (TP > FN > FP > TN). Default is True.
    - position_id_mode : str, optional
        Position identification strategy:
        - 'genomic': Current behavior - group by genomic position only (default)
        - 'transcript': Each transcript-position is unique (enables meta-learning)
        - 'hybrid': Group genomically but preserve transcript information
    - preserve_transcript_list : bool, optional
        If True and mode is 'genomic' or 'hybrid', keep list of transcript IDs for each position
    - verbose (int): Verbosity level for progress messages.

    Returns:
    - pl.DataFrame: Polars DataFrame with extracted sequences and metadata.

    If essential_columns_only=True:
        Columns: ['gene_id', 'position', 'strand', 'splice_type', 'window_start', 'window_end', 'sequence']
    Else:
        All columns from position_df plus 'window_start', 'window_end', 'sequence'
    """
    from Bio.Seq import Seq

    col_tid = kargs.get("col_tid", "transcript_id")
    col_gid = kargs.get("col_gid", "gene_id")
    col_pos = kargs.get("col_pos", "position")
    verbose = kargs.get("verbose", 1)
    
    # Ensure pred_type is included for meta-models
    if not additional_columns:
        additional_columns = []
    if 'pred_type' not in additional_columns:
        additional_columns.append('pred_type')
    if verbose:
        print(f"[info] Including all columns from position_df (essential_columns_only={essential_columns_only})")
    
    # Process transcript counts if needed
    transcript_counts = {}
    if add_transcript_count and col_tid in position_df.columns:
        if verbose:
            print(f"[info] Computing transcript counts for each position...")
        
        # Create a unique key for each position
        position_keys = []
        for row in position_df.iter_rows(named=True):
            # Key format: (gene_id, position, strand, splice_type)
            key = (row[col_gid], row[col_pos], row.get('strand'), row.get('splice_type'))
            position_keys.append((key, row.get(col_tid)))
        
        # Count transcripts for each position
        for key, tid in position_keys:
            if tid:  # Only count if transcript ID is present
                if key not in transcript_counts:
                    transcript_counts[key] = set()
                transcript_counts[key].add(tid)
        
        if verbose:
            print(f"[info] Found transcript counts for {len(transcript_counts)} unique positions")
            # Show distribution of transcript counts
            count_distribution = {}
            for transcripts in transcript_counts.values():
                count = len(transcripts)
                count_distribution[count] = count_distribution.get(count, 0) + 1
            print(f"[info] Transcript count distribution: {sorted(count_distribution.items())}")
    
    # Regular processing continues below

    # Fast lookups for gene sequences and strand
    gene_seq_lookup = dict(zip(sequence_df[col_gid], sequence_df["sequence"]))
    strand_lookup = dict(zip(sequence_df[col_gid], sequence_df["strand"]))

    if verbose:
        print(f"[info] Number of genes in sequence_df: {sequence_df.shape[0]}")
        print(f"[info] Number of positions in position_df: {position_df.shape[0]}")

    # Check required columns
    if "gene_id" not in position_df.columns or "position" not in position_df.columns:
        raise ValueError("position_df must contain at least 'gene_id' and 'position' columns.")

    # Core columns that uniquely identify a splice-site context / sequence window.
    # These are used for de-duplication and are **always** retained when
    # essential_columns_only=True.
    core_cols = [col_gid, col_pos, "strand", "splice_type"]

    # List to collect extracted sequences
    extracted_sequences = []
    processed_sequence_ids: set[tuple] = set()
    n_skipped_duplicates = 0
    n_total_rows = 0

    # Iterate over position_df rows
    for row in position_df.iter_rows(named=True):
        n_total_rows += 1
        gene_id = row[col_gid]
        position = row[col_pos]

        # Check if gene exists in sequence_df
        if gene_id not in gene_seq_lookup:
            if include_empty_entries:
                extracted_sequences.append({**row, 'window_start': None, 'window_end': None, 'sequence': None})
            continue

        sequence = gene_seq_lookup[gene_id]
        strand = strand_lookup[gene_id]

        # Test strand consistency
        assert strand == row['strand'], \
            f"[error] Strand mismatch for gene {gene_id}: expected {strand}, got {row['strand']}"

        # Calculate window boundaries, ensuring they don't exceed sequence limits
        window_start = max(position - window_size, 0)
        window_end = min(position + window_size, len(sequence))

        # Extract the sequence segment
        extracted_seq = sequence[window_start:window_end]

        # Reverse complement if on negative strand
        # if strand == '-':
        #     extracted_seq = str(Seq(extracted_seq).reverse_complement())
        # NOTE: No reverse complementing needed, since 'sequence' is pre-oriented !!!

        # Build a *core key* for this window to detect duplicates.
        # In training mode: core_key includes splice_type to keep different annotations at same position
        # In inference mode: splice_type is None for all positions, so deduplication would incorrectly
        # drop valid positions. Only deduplicate when splice_type is meaningful.
        core_key = tuple(row[c] for c in core_cols)
        
        # Only apply deduplication if splice_type is not None (training/evaluation mode)
        # In inference mode (splice_type=None), we want to keep ALL positions even if duplicate
        if row['splice_type'] is not None:
            if core_key in processed_sequence_ids:
                # Duplicate sequence window – skip to avoid redundant rows.
                n_skipped_duplicates += 1
                continue
            processed_sequence_ids.add(core_key)

        # Determine which columns to include in the output row
        if essential_columns_only:
            # Start with a fresh list each iteration to avoid mutation side-effects
            row_cols = list(core_cols)

            # Optionally keep transcript_id
            # if col_tid in row:
            #     row_cols.append(col_tid)

            # Optionally keep user-specified columns, filtering those that exist
            if additional_columns:
                row_cols.extend([c for c in additional_columns if c in row])

            result_row = {col: row[col] for col in row_cols}
        else:
            # Retain all original columns (deep copy row)
            result_row = row.copy()

        # Add the window and sequence information
        result_row.update({
            'window_start': window_start,
            'window_end': window_end,
            "sequence": extracted_seq
        })
        
        # Add transcript count if enabled and available
        if add_transcript_count and transcript_counts:
            # Build key for this position to look up in transcript_counts
            count_key = (gene_id, position, strand, row.get('splice_type'))
            if count_key in transcript_counts:
                result_row['transcript_count'] = len(transcript_counts[count_key])
            else:
                result_row['transcript_count'] = 0  # No transcripts found
            
        # Remove transcript_id if requested
        if drop_transcript_id and 'transcript_id' in result_row:
            del result_row['transcript_id']

        extracted_sequences.append(result_row)

    # DEBUG: Report extraction stats before DataFrame conversion
    n_output = len(extracted_sequences)
    if verbose or n_skipped_duplicates > 0 or (n_total_rows != n_output + n_skipped_duplicates):
        print(f"[debug] extract_analysis_sequences: {n_total_rows} input rows → {n_output} output rows")
        if n_skipped_duplicates > 0:
            print(f"[debug]   Skipped {n_skipped_duplicates} duplicate rows (splice_type != None)")
        if n_total_rows != n_output + n_skipped_duplicates:
            n_missing = n_total_rows - n_output - n_skipped_duplicates
            print(f"[debug]   ⚠️ {n_missing} rows lost for other reasons (missing gene_id, etc.)")
    
    # Convert to Polars DataFrame – infer schema over all rows to avoid type mismatches
    # When the first few rows contain only nulls, Polars may infer `Null` type and then
    # raise a ComputeError once a non-null value appears. Setting `infer_schema_length=None`
    # forces Polars to inspect the entire sequence, yielding a consistent schema.
    output_df = pl.from_dicts(extracted_sequences, infer_schema_length=None)

    # Reorder columns
    columns = output_df.columns

    # Transcript ID is optional
    first_columns = [col_gid, col_tid] if col_tid in columns else [col_gid]

    last_columns = ['sequence']
    middle_columns = [col for col in columns if col not in first_columns + last_columns]
    ordered_columns = first_columns + middle_columns + last_columns
    output_df = output_df.select(ordered_columns)

    if verbose:
        print(f"[info] Extracted sequences for {len(output_df)} positions.")
    
    # If enabled, resolve conflicts when the same position has multiple prediction types
    if resolve_prediction_conflicts and 'pred_type' in output_df.columns:
        original_count = len(output_df)
        
        # Define prediction type priorities: TP > FN > FP > TN
        pred_type_priority = {'TP': 0, 'FN': 1, 'FP': 2, 'TN': 3}
        
        # Create a numeric priority column - using a version-compatible approach
        # First add a default high value (lowest priority)
        output_df = output_df.with_columns(
            pl.lit(99).alias('pred_type_priority')
        )
        
        # Then update priority for each prediction type
        for pred_type, priority in pred_type_priority.items():
            output_df = output_df.with_columns(
                pl.when(pl.col('pred_type') == pred_type)
                .then(pl.lit(priority))
                .otherwise(pl.col('pred_type_priority'))
                .alias('pred_type_priority')
            )
        
        # Determine grouping columns based on position identification mode
        if position_id_mode == 'transcript':
            # Full transcript-specific: include transcript_id in grouping
            group_cols = ['gene_id', 'position', 'strand', 'transcript_id']
            if verbose:
                print(f"[info] Using transcript-specific position identification")
            
        elif position_id_mode == 'hybrid':
            # Hybrid: Group genomically but preserve transcript info
            group_cols = ['gene_id', 'position', 'strand']
            
            if preserve_transcript_list and 'transcript_id' in output_df.columns:
                # First collect all transcript IDs for each position
                transcript_groups = output_df.group_by(group_cols).agg([
                    pl.col('transcript_id').unique().alias('transcript_id_list'),
                    pl.col('transcript_id').count().alias('transcript_count_total'),
                    # Collect splice types seen across transcripts
                    pl.col('splice_type').unique().alias('splice_types_observed'),
                ])
                if verbose:
                    print(f"[info] Hybrid mode: preserving transcript information for {len(transcript_groups)} positions")
                
        else:  # position_id_mode == 'genomic' (default/current behavior)
            group_cols = ['gene_id', 'position', 'strand']
            if verbose:
                print(f"[info] Using genomic position identification (current behavior)")
        
        # Note: We deliberately exclude splice_type from grouping to ensure proper deduplication
        # when a position has both a specific splice_type and None (except in transcript mode)
            
        # Perform deduplication based on the chosen mode
        if position_id_mode == 'hybrid' and preserve_transcript_list:
            # Special handling for hybrid mode with transcript preservation
            
            # First, get the best prediction for each genomic position
            best_predictions = output_df.sort('pred_type_priority').group_by(group_cols).first()
            
            # Then merge with the transcript information we collected
            output_df = best_predictions.join(
                transcript_groups,
                on=group_cols,
                how='left'
            )
            
            # Add a flag indicating this position has multiple transcript contexts
            output_df = output_df.with_columns(
                (pl.col('transcript_count_total') > 1).alias('is_multi_transcript')
            )
            
            if verbose:
                multi_transcript_count = output_df.filter(pl.col('is_multi_transcript')).height
                print(f"[info] Preserved transcript information for {multi_transcript_count} multi-transcript positions")
            
        else:
            # Standard deduplication (works for both 'genomic' and 'transcript' modes)
            output_df = output_df.with_columns(
                pl.col('pred_type_priority').rank(method='dense', descending=False)
                .over(group_cols)
                .alias('group_rank')
            )
            
            output_df = output_df.filter(pl.col('group_rank') == 1)
            output_df = output_df.drop(['group_rank'])
        
        # Clean up temporary column
        output_df = output_df.drop('pred_type_priority')
        
        # Report on the deduplication
        if verbose and original_count > len(output_df):
            dedup_count = original_count - len(output_df)
            if position_id_mode == 'transcript':
                print(f"[info] Resolved {dedup_count} conflicting predictions within transcript-position pairs")
            else:
                print(f"[info] Resolved {dedup_count} conflicting prediction types through priority-based deduplication")

    return output_df