"""
Data handlers for meta models.

This module provides utilities for loading and saving meta model data
at various stages of processing.
"""

import os
import glob
import re
from tokenize import triple_quoted
from tqdm.auto import tqdm
import pandas as pd
import polars as pl
from typing import Union, Optional, Dict, List, Any, Tuple, Iterator, Set, NamedTuple

# Import the original file handler
from meta_spliceai.splice_engine.model_evaluator import ModelEvaluationFileHandler
from meta_spliceai.splice_engine.meta_models.core.analyzer import Analyzer
from meta_spliceai.splice_engine.utils_fs import read_splice_sites


class MetaModelDataHandler:
    """
    Handler for meta model data loading and saving operations.
    
    This class wraps the original ModelEvaluationFileHandler to provide
    a more focused interface for meta model operations.
    """
    
    def __init__(
        self,
        eval_dir: Optional[str] = None,
        meta_subdir: str = 'meta_models',
        separator: str = '\t',
        **kwargs
    ):
        """
        Initialize the meta model data handler.
        
        Parameters
        ----------
        eval_dir : Optional[str], optional
            Directory for evaluation data, by default None (uses ErrorAnalyzer.eval_dir)
        meta_subdir : str, optional
            Subdirectory for meta model data under eval_dir, by default 'meta_models'
        separator : str, optional
            Separator for data files, by default '\t'
        """
        self.eval_dir = eval_dir or Analyzer.eval_dir
        self.separator = separator
        
        # Create a meta_models subdirectory for all meta model outputs
        self.meta_subdir = meta_subdir
        self.meta_dir = os.path.join(self.eval_dir, self.meta_subdir)
        os.makedirs(self.meta_dir, exist_ok=True)
        
        # Create a file handler that points to the parent eval_dir
        # We'll explicitly set subdirectories when needed
        self.file_handler = ModelEvaluationFileHandler(
            self.eval_dir, separator=separator, **kwargs
        )
    
    def _get_output_dir(self, output_subdir: Optional[str] = None, use_shared_dir: bool = False) -> str:
        """
        Get the appropriate output directory, using meta_dir as default.
        
        Parameters
        ----------
        output_subdir : Optional[str], optional
            Additional subdirectory under meta_dir, by default None
        use_shared_dir : bool, optional
            If True, use the shared evaluation directory (self.eval_dir) instead of 
            the subject-specific directory (self.meta_dir), by default False
            
        Returns
        -------
        str
            Full path to the output directory
        """
        if use_shared_dir:
            # Use the shared evaluation directory
            output_dir = self.eval_dir
            if output_subdir:
                output_dir = os.path.join(output_dir, output_subdir)
        else:
            # Use the subject-specific meta_models directory (default behavior)
            if output_subdir:
                # Create a subdirectory under meta_dir
                output_dir = os.path.join(self.meta_dir, output_subdir)
            else:
                # Use meta_dir as the default
                output_dir = self.meta_dir

            # Prevent duplicate nesting when output_subdir equals meta_subdir (e.g. "meta_models/meta_models")
            if os.path.normpath(output_dir).endswith(os.path.join(self.meta_subdir, self.meta_subdir)):
                output_dir = self.meta_dir
            
        os.makedirs(output_dir, exist_ok=True)
        return output_dir
    
    def _with_output_dir(self, func, output_subdir: Optional[str] = None, use_shared_dir: bool = False, *args, **kwargs):
        """
        Execute a function with a temporarily modified output directory.
        
        This changes the file_handler's output_dir temporarily and then restores it.
        
        Parameters
        ----------
        func : Callable
            Function to execute with modified output directory
        output_subdir : Optional[str], optional
            Subdirectory to use, by default None
        use_shared_dir : bool, optional
            If True, use the shared evaluation directory (self.eval_dir) instead of 
            the subject-specific directory (self.meta_dir), by default False
            
        Returns
        -------
        Any
            Result of the function call
        """
        # Save the original output directory
        orig_output_dir = self.file_handler.output_dir
        
        try:
            # Set the new output directory
            self.file_handler.output_dir = self._get_output_dir(output_subdir, use_shared_dir)
            
            # Call the function with the new output directory
            return func(*args, **kwargs)
        finally:
            # Restore the original output directory
            self.file_handler.output_dir = orig_output_dir
    
    def load_analysis_sequences(
        self,
        aggregated: bool = True,
        error_label: str = None,
        correct_label: str = None,
        **kwargs
    ) -> pl.DataFrame:
        """
        Load analysis sequences.
        
        WARNING: This method attempts to load ALL analysis sequences into memory at once,
        which may cause memory issues for large datasets. For memory-efficient loading
        of specific genes, use `iterative_load_analysis_sequences` instead.
        
        Parameters
        ----------
        aggregated : bool, optional
            Whether to load aggregated data, by default True
        error_label : str, optional
            Label for error cases, by default None
        correct_label : str, optional
            Label for correct cases, by default None
        **kwargs
            Additional arguments to pass to load_analysis_sequences
        
        Returns
        -------
        pl.DataFrame
            DataFrame containing analysis sequences
        """
        if error_label is not None: 
            kwargs['error_label'] = error_label
        if correct_label is not None: 
            kwargs['correct_label'] = correct_label

        return self.file_handler.load_analysis_sequences(aggregated=aggregated, **kwargs)
    
    def iterative_load_analysis_sequences(
        self,
        target_gene_ids: List[str],
        subject: str = "analysis_sequences",
        output_subdir: Optional[str] = None,
        use_shared_dir: bool = False,
        show_progress: bool = True,
        error_label: Optional[str] = None,
        correct_label: Optional[str] = None,
        essential_columns_only: bool = True,
        splice_type: str = "any",
        batch_size: Optional[int] = None,
        verbose: int = 1,
        **kwargs
    ) -> pl.DataFrame:
        """
        Memory-efficient loading of analysis sequences for specific genes.
        
        This method iterates through chunked analysis sequence files, filtering
        for specific genes without loading the entire dataset into memory.
        
        Parameters
        ----------
        target_gene_ids : List[str]
            List of gene IDs to retrieve sequences for
        subject : str, optional
            Subject name for the sequence files, by default "analysis_sequences"
        output_subdir : Optional[str], optional
            Additional subdirectory under meta_dir, by default None
        use_shared_dir : bool, optional
            If True, use the shared evaluation directory, by default False
        show_progress : bool, optional
            Whether to show a progress bar, by default True
        error_label : Optional[str], optional
            Label for error cases (FP/FN), by default None
        correct_label : Optional[str], optional
            Label for correct cases (TP/TN), by default None
        essential_columns_only : bool, optional
            Whether to only include essential columns in the result, by default True
        splice_type : str, optional
            Filter by splice type ("donor", "acceptor", or "any"), by default "any"
        batch_size : Optional[int], optional
            Number of files to process in each batch, by default None (process all at once)
        verbose : int, optional
            Verbosity level, by default 1
        **kwargs
            Additional arguments for filtering
        
        Returns
        -------
        pl.DataFrame
            Filtered DataFrame containing analysis sequences for target genes
        """
        # Parameters for file handler
        if error_label is not None:
            kwargs['error_label'] = error_label
        if correct_label is not None:
            kwargs['correct_label'] = correct_label
        
        # Convert target genes to a set for faster lookups
        target_gene_set = set(target_gene_ids)
        
        # This will hold our accumulated results
        results = []
        found_gene_ids = set()
        total_rows = 0
        unified_schema = None
        
        # Helper class to store file information for sorting
        class SequenceFile(NamedTuple):
            path: str
            chromosome: str = ""
            chunk_start: int = 0
            chunk_end: int = 0
            is_aggregated: bool = False
        
        def _process_file(file_path: str) -> Optional[pl.DataFrame]:
            """Process a single analysis sequence file and extract matching genes."""
            nonlocal total_rows, found_gene_ids, unified_schema
            try:
                # We've combined both schema_overrides and filter_by_splice_type logic here
                # to avoid having to call those functions separately
                df = pl.read_csv(
                    file_path,
                    separator=self.separator,
                    schema_overrides={"chrom": pl.Utf8}
                )
                
                # Establish unified column order from first successful file
                if unified_schema is None:
                    unified_schema = df.schema
                    if verbose >= 2:
                        print(f"[analysis_sequences] Using column order from {os.path.basename(file_path)}: {len(unified_schema)} columns")
                
                # Verify column sets match (not just order)
                current_columns = set(df.columns)
                expected_columns = set(unified_schema.keys())
                
                if current_columns != expected_columns:
                    missing = expected_columns - current_columns
                    extra = current_columns - expected_columns
                    raise ValueError(
                        f"Column mismatch in {os.path.basename(file_path)}:\n"
                        f"  Missing columns: {missing}\n"
                        f"  Extra columns: {extra}\n"
                        f"  Expected columns: {len(expected_columns)}, Got: {len(current_columns)}"
                    )
                
                # Reorder columns to match the unified order
                df = df.select([col for col in unified_schema.keys()])
                
                # Filter by splice type if needed
                if splice_type != "any" and "splice_type" in df.columns:
                    df = df.filter(pl.col("splice_type") == splice_type)
                
                # Filter to target genes
                if "gene_id" in df.columns:
                    df = df.filter(pl.col("gene_id").is_in(target_gene_set))
                    
                    # Early return if no matching genes
                    if df.height == 0:
                        return None
                    
                    # Update tracking variables
                    found_genes_in_chunk = set(df["gene_id"].unique())
                    found_gene_ids.update(found_genes_in_chunk)
                    total_rows += df.height
                    
                    return df
                else:
                    if verbose >= 2:
                        print(f"[warning] No 'gene_id' column in {os.path.basename(file_path)}")
                    return None
            except Exception as e:
                if verbose >= 1:
                    print(f"[error] Failed to process {os.path.basename(file_path)}: {str(e)}")
                return None
        
        def _parse_sequence_filename(filename: str) -> SequenceFile:
            """Parse a sequence filename to extract chromosome and chunk information."""
            basename = os.path.basename(filename)
            
            # Check if this is an aggregated file
            if f"full_{subject}" in basename:
                return SequenceFile(path=filename, is_aggregated=True)
            
            # Try to parse the chunk pattern: {subject}_{chr}_chunk_{chunk_start}_{chunk_end}.{ext}
            pattern = fr"^{subject}_([^_]+)_chunk_([0-9]+)_([0-9]+)\."  # Regex for chunk files
            match = re.search(pattern, basename)
            
            if match:
                chrom, start, end = match.groups()
                return SequenceFile(
                    path=filename,
                    chromosome=chrom,
                    chunk_start=int(start),
                    chunk_end=int(end)
                )
            
            # If we couldn't parse it, just return the file with default values
            return SequenceFile(path=filename)
            
        # Use the temporary output dir context
        with self._output_dir_context(output_subdir, use_shared_dir) as output_dir:
            # Find all analysis sequence files (both chunk-level and aggregated if present)
            # Exclude non-training artifacts like 'analysis_sequences_inference.tsv'
            file_pattern = os.path.join(output_dir, f"*{subject}*.{self.file_handler.file_extension}")
            all_sequence_files = glob.glob(file_pattern)
            
            # Filter out non-training artifacts
            sequence_file_paths = []
            for file_path in all_sequence_files:
                basename = os.path.basename(file_path)
                # Skip files that don't follow the training artifact pattern
                # Training artifacts should have: {subject}_{chromosome}_chunk_{start}_{end}.tsv
                # or be aggregated files: full_{subject}.tsv
                if (f"full_{subject}" in basename or 
                    re.search(fr"^{subject}_[^_]+_chunk_[0-9]+_[0-9]+\.", basename)):
                    sequence_file_paths.append(file_path)
                elif verbose >= 2:
                    print(f"[info] Skipping non-training artifact: {basename}")
            
            if len(sequence_file_paths) == 0:
                if verbose >= 1:
                    print(f"[warning] No analysis sequence files found matching pattern: {file_pattern}")
                return pl.DataFrame()
            
            # Parse and organize files
            sequence_files = [_parse_sequence_filename(path) for path in sequence_file_paths]
            
            # Sort files for more efficient processing: first by chromosome, then by chunk start
            # This ensures we process files in a logical order, which may improve cache locality
            non_aggregated_files = [f for f in sequence_files if not f.is_aggregated]
            non_aggregated_files.sort(key=lambda f: (f.chromosome, f.chunk_start))
            
            # Get aggregated files separately
            aggregated_files = [f.path for f in sequence_files if f.is_aggregated]
            
            if verbose >= 1:
                print(f"[info] Found {len(sequence_file_paths)} analysis sequence files to process")
                print(f"[info] - {len(aggregated_files)} aggregated files")
                print(f"[info] - {len(non_aggregated_files)} chunked files across {len(set(f.chromosome for f in non_aggregated_files))} chromosomes")
            
            if aggregated_files and len(target_gene_ids) > 100:  # Only check aggregated for large gene lists
                if verbose >= 1:
                    print(f"[info] Checking aggregated file(s) first")
                    
                for agg_file in aggregated_files:
                    df = _process_file(agg_file)
                    if df is not None and df.height > 0:
                        if verbose >= 1:
                            print(f"[info] Successfully loaded {df.height} rows for {len(set(df['gene_id'].unique()))} genes from aggregated file")
                        return df
            
            # Process chunk files (in batches if requested) - already sorted by chromosome and chunk
            remaining_files = [f.path for f in non_aggregated_files]
            
            if batch_size is not None and batch_size > 0:
                # Process in batches to manage memory
                # Try to keep files from same chromosome in same batch when possible
                file_batches = []
                current_batch = []
                current_size = 0
                current_chrom = None
                
                for seq_file in non_aggregated_files:
                    # If we're starting a new batch or continuing same chromosome
                    if (current_size == 0 or 
                        (seq_file.chromosome == current_chrom and current_size < batch_size)):
                        current_batch.append(seq_file.path)
                        current_size += 1
                        current_chrom = seq_file.chromosome
                    # If we need to start a new batch
                    else:
                        if current_batch:  # Don't add empty batches
                            file_batches.append(current_batch)
                        current_batch = [seq_file.path]
                        current_size = 1
                        current_chrom = seq_file.chromosome
                
                # Add the last batch if it's not empty
                if current_batch:
                    file_batches.append(current_batch)
                
                if verbose >= 1:
                    print(f"[info] Processing {len(remaining_files)} files in {len(file_batches)} batches")
                    
                for batch_idx, file_batch in enumerate(file_batches):
                    batch_results = []
                    
                    if show_progress:
                        file_iter = tqdm(file_batch, desc=f"Batch {batch_idx+1}/{len(file_batches)}")
                    else:
                        file_iter = file_batch
                        
                    for file_path in file_iter:
                        chunk_df = _process_file(file_path)
                        if chunk_df is not None and chunk_df.height > 0:
                            batch_results.append(chunk_df)
                    
                    # Combine results from this batch and add to overall results
                    if batch_results:
                        batch_df = pl.concat(batch_results)
                        results.append(batch_df)
                        
                        # Check if we found all target genes
                        if len(found_gene_ids) == len(target_gene_set):
                            if verbose >= 1:
                                print(f"[info] Found all {len(target_gene_ids)} target genes. Stopping early.")
                            break
            else:
                # Process all files at once (with progress bar if requested)
                if show_progress:
                    file_iter = tqdm(remaining_files, desc="Processing sequence files")
                else:
                    file_iter = remaining_files
                    
                for file_path in file_iter:
                    chunk_df = _process_file(file_path)
                    if chunk_df is not None and chunk_df.height > 0:
                        results.append(chunk_df)
        
        # Combine all results
        if not results:
            if verbose >= 1:
                print(f"[warning] No matching sequences found for the {len(target_gene_ids)} target genes")
            # Return empty DataFrame with expected columns
            return pl.DataFrame()
        
        final_df = pl.concat(results)
        
        # Deduplicate if needed
        if "sequence" in final_df.columns:
            # Create a unique key based on core identifiers
            key_cols = ["gene_id", "position", "strand", "splice_type"]
            n_before = final_df.height
            
            # Check if all key columns exist
            if all(col in final_df.columns for col in key_cols):
                final_df = final_df.unique(subset=key_cols)
                n_after = final_df.height
                
                if verbose >= 1 and n_before != n_after:
                    print(f"[info] Deduplicated {n_before - n_after} rows ({n_before} → {n_after})")
        
        if verbose >= 1:
            found_genes = set(final_df["gene_id"].unique()) if "gene_id" in final_df.columns else set()
            missing_genes = target_gene_set - found_genes
            
            print(f"[info] Found sequences for {len(found_genes)}/{len(target_gene_ids)} target genes")
            print(f"[info] Total rows in result: {final_df.height}")
            
            if missing_genes and verbose >= 2:
                print(f"[info] Missing genes: {list(missing_genes)[:10]}{'...' if len(missing_genes) > 10 else ''}")
        
        return final_df
    
    def _output_dir_context(self, output_subdir=None, use_shared_dir=False):
        """Context manager for temporarily setting the output directory."""
        from contextlib import contextmanager
        
        @contextmanager
        def context():
            output_dir = self._get_output_dir(output_subdir, use_shared_dir)
            try:
                yield output_dir
            finally:
                pass  # No cleanup needed
                
        return context()
    
    def load_featurized_artifact(
        self,
        aggregated: bool = True,
        subject: str = "seq_featurized",
        error_label: str = "FP",
        correct_label: str = "TP",
        **kwargs
    ) -> pd.DataFrame:
        """
        Load a featurized dataset.
        
        Parameters
        ----------
        aggregated : bool, optional
            Whether to load aggregated data, by default True
        subject : str, optional
            Subject identifier for the dataset, by default "seq_featurized"
        error_label : str, optional
            Label for error class (positive), by default "FP"
        correct_label : str, optional
            Label for correct class (negative), by default "TP"
        
        Returns
        -------
        pd.DataFrame
            Featurized dataset
        """
        return self.file_handler.load_featurized_artifact(
            aggregated=aggregated,
            subject=subject,
            error_label=error_label,
            correct_label=correct_label,
            **kwargs
        )
    
    def save_featurized_artifact(
        self,
        df: Union[pd.DataFrame, pl.DataFrame],
        aggregated: bool = True,
        subject: str = "seq_featurized",
        error_label: str = "FP",
        correct_label: str = "TP",
        **kwargs
    ) -> str:
        """
        Save a featurized dataset.
        
        Parameters
        ----------
        df : Union[pd.DataFrame, pl.DataFrame]
            Dataset to save
        aggregated : bool, optional
            Whether to save as aggregated data, by default True
        subject : str, optional
            Subject identifier for the dataset, by default "seq_featurized"
        error_label : str, optional
            Label for error class (positive), by default "FP"
        correct_label : str, optional
            Label for correct class (negative), by default "TP"
        
        Returns
        -------
        str
            Path to the saved file
        """
        return self._with_output_dir(
            self.file_handler.save_featurized_artifact,
            df=df,
            aggregated=aggregated,
            subject=subject,
            error_label=error_label,
            correct_label=correct_label,
            **kwargs
        )

    def parametrize_subject(self, subject, **kargs):
        pred_type = kargs.get('pred_type', self.pred_type)  # Only used in training a sequence model
        if pred_type is not None and pred_type not in subject: 
            subject = f"{subject}_{pred_type.lower()}"

        # For specific taxonomy
        error_label = kargs.get("error_label", self.error_label)
        correct_label = kargs.get("correct_label", self.correct_label)
        if error_label is not None and correct_label is not None: 
            if error_label not in subject:
                subject = f"{subject}_{error_label.lower()}"
            if correct_label not in subject:
                subject = f"{subject}_{correct_label.lower()}"

        if kargs.get('test', False):
            subject = f"{subject}_test"

        return subject
    
    def save_analysis_sequences(
        self,
        df: Union[pd.DataFrame, pl.DataFrame],
        aggregated: bool = True,
        error_label: str = None,
        correct_label: str = None,
        **kwargs
    ) -> str:
        """
        Save analysis sequences.
        
        Parameters
        ----------
        df : Union[pd.DataFrame, pl.DataFrame]
            Sequences to save
        aggregated : bool, optional
            Whether to save as aggregated data, by default True
        error_label : str, optional
            Label for error class (positive), by default "FP"
        correct_label : str, optional
            Label for correct class (negative), by default "TP"
        
        Returns
        -------
        str
            Path to the saved file
        """
        if error_label is not None: 
            kwargs['error_label'] = error_label
        if correct_label is not None: 
            kwargs['correct_label'] = correct_label

        return self._with_output_dir(
            self.file_handler.save_analysis_sequences,
            analysis_df=df,
            aggregated=aggregated,
            **kwargs
        )
    
    def save_featurized_dataset(
        self,
        df: Union[pd.DataFrame, pl.DataFrame],
        aggregated: bool = True,
        error_label: str = None,
        correct_label: str = None,
        **kwargs
    ) -> str:
        """
        Save the final featurized dataset.
        
        Parameters
        ----------
        df : Union[pd.DataFrame, pl.DataFrame]
            Dataset to save
        error_label : str, optional
            Label for error class (positive), by default "FP"
        correct_label : str, optional
            Label for correct class (negative), by default "TP"
        
        Returns
        -------
        str
            Path to the saved file
        """
        if error_label is not None: 
            kwargs['error_label'] = error_label
        if correct_label is not None: 
            kwargs['correct_label'] = correct_label

        return self._with_output_dir(
            self.file_handler.save_featurized_dataset,
            df=df,
            aggregated=aggregated,
            **kwargs
        )
    
    def load_featurized_dataset(
        self,
        aggregated: bool = True,
        subject: str = "featurized_dataset",
        chr: Optional[str] = None,
        chunk_start: Optional[int] = None,
        chunk_end: Optional[int] = None,
        error_label: str = None,
        correct_label: str = None,
        convert_to_pandas: bool = True,
        output_subdir: Optional[str] = None,
        **kwargs
    ) -> Union[pd.DataFrame, pl.DataFrame]:
        """
        Load the final featurized dataset.
        
        Parameters
        ----------
        aggregated : bool, optional
            Whether to load an aggregated file, by default True
        subject : str, optional
            Subject identifier for the file, by default "featurized_dataset"
        chr : Optional[str], optional
            Chromosome identifier, by default None
        chunk_start : Optional[int], optional
            Start index of chunk, by default None
        chunk_end : Optional[int], optional
            End index of chunk, by default None
        error_label : str, optional
            Label for error class (positive), by default "FP"
        correct_label : str, optional
            Label for correct class (negative), by default "TP"
        convert_to_pandas : bool, optional
            Whether to convert to pandas DataFrame, by default True (featurized data is usually used with scikit-learn)
        output_subdir : Optional[str], optional
            Subdirectory under meta_dir to look for file, by default None
            
        Returns
        -------
        Union[pd.DataFrame, pl.DataFrame]
            Loaded DataFrame with featurized data
        """
        # Create a wrapper function to capture the file loading logic
        def _load_featurized_dataset(**kws):
            try:
                # Pass relevant parameters for subject parametrization
                kws_with_labels = {
                    **kws,
                    'error_label': error_label,
                    'correct_label': correct_label
                }
                
                # Delegate to the file handler's method
                df = self.file_handler.load_featurized_dataset(
                    chr=chr,
                    chunk_start=chunk_start,
                    chunk_end=chunk_end,
                    aggregated=aggregated,
                    subject=subject,
                    **kws_with_labels
                )
                
                # Convert to polars if needed (original returns pandas)
                if not convert_to_pandas and isinstance(df, pd.DataFrame):
                    df = pl.from_pandas(df)
                    
                return df
            except Exception as e:
                print(f"[error] Failed to load featurized dataset: {str(e)}")
                raise
        
        # Use _with_output_dir to temporarily set the output directory to meta_dir/output_subdir
        return self._with_output_dir(_load_featurized_dataset, output_subdir=output_subdir, **kwargs)
    
    def _get_column_order(self, df: pl.DataFrame, enhanced: bool = False) -> Tuple[List[str], List[str]]:
        """
        Helper to determine appropriate column ordering for splice positions DataFrame.
        
        Parameters
        ----------
        df : pl.DataFrame
            DataFrame to order columns for
        enhanced : bool, optional
            Whether this is enhanced positions with all scores, by default False
            
        Returns
        -------
        Tuple[List[str], List[str]]
            first_columns, last_columns lists for reordering
        """
        first_columns = ['gene_id']
        if 'transcript_id' in df.columns:
            first_columns.append('transcript_id')
            
        # Basic columns that should appear first
        for col in ['position', 'predicted_position', 'true_position', 'strand', 'chrom']:
            if col in df.columns and col not in first_columns:
                first_columns.append(col)
        
        # Identify the columns to place at the end
        last_columns = []
        if enhanced:
            # Enhanced version with three scores
            last_columns = ['donor_score', 'acceptor_score', 'neither_score']
        else:
            # Traditional version with single score
            if 'score' in df.columns:
                last_columns = ['score']
                
        # Add common fields that should appear at the end
        for col in ['pred_type', 'splice_type']:
            if col in df.columns and col not in last_columns:
                last_columns.append(col)
                
        # IMPORTANT: We need to make sure that any additional columns 
        # (like context_* or derived features) are preserved
        # The file_handler's reorder_columns function will handle any columns
        # not specified in first_columns or last_columns, ensuring they're
        # preserved in the output between the first and last groups
        
        return first_columns, last_columns
    
    def save_splice_positions(
        self,
        positions_df: pl.DataFrame,
        chr: Optional[str] = None,
        chunk_start: Optional[int] = None,
        chunk_end: Optional[int] = None,
        enhanced: bool = True,
        aggregated: bool = False,
        output_subdir: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Save splice positions DataFrame.
        
        Parameters
        ----------
        positions_df : pl.DataFrame
            Positions DataFrame to save
        chr : Optional[str], optional
            Chromosome, by default None
        chunk_start : Optional[int], optional
            Chunk start, by default None
        chunk_end : Optional[int], optional
            Chunk end, by default None
        enhanced : bool, optional
            Whether this is enhanced positions with all scores, by default True
        aggregated : bool, optional
            Whether this is aggregated data, by default False
        output_subdir : Optional[str], optional
            Output subdirectory, by default None
        
        Returns
        -------
        str
            Path to the saved file
        """
        # Validate enhanced positions DataFrame
        if enhanced:
            if not all(col in positions_df.columns for col in ['donor_score', 'acceptor_score', 'neither_score']):
                raise ValueError("Enhanced positions DataFrame must contain 'donor_score', 'acceptor_score', and 'neither_score' columns")
        
        # Do NOT convert `positions_df` to pandas - keep as Polars since ModelEvaluationFileHandler.save_splice_positions 
        # and reorder_columns specifically expect a Polars DataFrame
            
        # Determine the appropriate subject
        subject = "splice_positions_enhanced" if enhanced else "splice_positions"
        
        # Get appropriate column ordering for output readability and standardization
        first_columns, last_columns = self._get_column_order(positions_df, enhanced)
        
        return self._with_output_dir(
            self.file_handler.save_splice_positions,
            output_subdir=output_subdir,
            splice_positions_df=positions_df,  # Pass the original Polars DataFrame
            chr=chr,
            chunk_start=chunk_start,
            chunk_end=chunk_end,
            aggregated=aggregated,
            subject=subject,
            standardize_columns=True,  # Always standardize, but with appropriate columns
            first_columns=first_columns,
            last_columns=last_columns,
            **kwargs
        )
    
    def load_splice_positions(
        self,
        aggregated: bool = True,
        subject: str = None,
        chr: Optional[str] = None,
        chunk_start: Optional[int] = None,
        chunk_end: Optional[int] = None,
        enhanced: bool = True,
        convert_to_pandas: bool = False,
        output_subdir: Optional[str] = None,
        use_shared_dir: bool = False,
        **kwargs
    ) -> Union[pd.DataFrame, pl.DataFrame]:
        """
        Load splice position data with enhanced probability scores if requested.
        
        Parameters
        ----------
        aggregated : bool, optional
            Whether to load an aggregated file, by default True
        subject : str, optional
            Subject identifier for the file, by default None (auto-determined)
        chr : Optional[str], optional
            Chromosome identifier, by default None
        chunk_start : Optional[int], optional
            Start index of chunk, by default None
        chunk_end : Optional[int], optional
            End index of chunk, by default None
        enhanced : bool, optional
            Whether to load an enhanced file with all three probabilities, by default True
        convert_to_pandas : bool, optional
            Whether to convert to pandas DataFrame, by default False
        output_subdir : Optional[str], optional
            Subdirectory under meta_dir to look for file, by default None
        use_shared_dir : bool, optional
            If True, look in shared evaluation directory instead of meta_dir, by default False
            
        Returns
        -------
        Union[pd.DataFrame, pl.DataFrame]
            Loaded DataFrame with splice positions
        """
        verbose = kwargs.pop('verbose', 1)

        # Determine the subject based on enhanced flag if not explicitly provided
        if subject is None:
            subject = "splice_positions_enhanced" if enhanced else "splice_positions"
        
        # Create a wrapper function to capture the file loading logic
        def _load_splice_positions(chr=chr, chunk_start=chunk_start, chunk_end=chunk_end, 
                            aggregated=aggregated, subject=subject, **kws):
            try:
                # Use the file handler's method to get the file path
                file_path = self.file_handler.get_splice_positions_file_path(
                    chr=chr, 
                    chunk_start=chunk_start, 
                    chunk_end=chunk_end, 
                    aggregated=aggregated, 
                    subject=subject
                )

                if verbose > 0:
                    print(f"Loading splice positions file: {file_path}")
                
                # Check if file exists before loading
                if not os.path.exists(file_path):
                    raise FileNotFoundError(f"Splice positions file not found: {file_path}")
                
                # Ensure chromosome columns are loaded as strings and include any schema
                schema = kws.get('schema', {})
                if 'chrom' not in schema:
                    schema['chrom'] = pl.Utf8
                
                # Load with Polars
                df = pl.read_csv(
                    file_path,
                    separator=self.separator,
                    schema_overrides=schema,
                    **{k: v for k, v in kws.items() if k != 'schema'}
                )
                
                # Convert to pandas if requested
                if convert_to_pandas:
                    df = df.to_pandas()
                    
                return df
            except Exception as e:
                print(f"[error] Failed to load splice positions from {file_path}: {str(e)}")
                raise
        
        # Use _with_output_dir to temporarily set the output directory to meta_dir/output_subdir
        return self._with_output_dir(_load_splice_positions, output_subdir=output_subdir, use_shared_dir=use_shared_dir, **kwargs)

    def load_splice_annotations(
        self,
        separator: Optional[str] = None,
        dtypes: Optional[Dict] = None,
        convert_to_polars: bool = True,
        **kwargs
    ) -> Union[pd.DataFrame, pl.DataFrame]:
        """
        Load splice site annotations from the parent directory of eval_dir.
        
        Parameters
        ----------
        separator : Optional[str], optional
            Separator used in the file, by default None (uses self.separator)
        dtypes : Optional[Dict], optional
            Data types for columns, by default None
        convert_to_polars : bool, optional
            Whether to convert the result to a polars DataFrame, by default True
            
        Returns
        -------
        Union[pd.DataFrame, pl.DataFrame]
            DataFrame containing splice site annotations
        """
        separator = separator or self.separator
        
        # Get the parent directory of eval_dir
        parent_dir = os.path.dirname(self.eval_dir)
        splice_sites_file = os.path.join(parent_dir, "splice_sites.tsv")
        
        if not os.path.exists(splice_sites_file):
            # Try alternative filenames if the default isn't found
            alternative_files = [
                os.path.join(parent_dir, "splice_sites.csv"),
                os.path.join(self.eval_dir, "splice_sites.tsv"),
                os.path.join(self.eval_dir, "splice_sites.csv")
            ]
            
            for alt_file in alternative_files:
                if os.path.exists(alt_file):
                    splice_sites_file = alt_file
                    # Adjust separator if needed based on extension
                    if alt_file.endswith('.csv'):
                        separator = ','
                    break
            else:
                raise FileNotFoundError(
                    f"Splice sites file not found in parent dir ({parent_dir}) "
                    f"or eval dir ({self.eval_dir})"
                )
        
        # Determine the appropriate separator based on file extension if not already set
        if separator is None:
            if splice_sites_file.endswith('.csv'):
                separator = ','
            else:
                separator = '\t'  # Default to tab for .tsv
        
        # Use utils_fs.read_splice_sites to load the data
        df = read_splice_sites(splice_sites_file, separator=separator, dtypes=dtypes)
        
        # Convert to polars if requested
        if convert_to_polars and isinstance(df, pd.DataFrame):
            return pl.from_pandas(df)
        
        return df
    
    def load_error_analysis(
        self,
        aggregated: bool = True,
        subject: str = "splice_errors",
        chr: Optional[str] = None,
        chunk_start: Optional[int] = None,
        chunk_end: Optional[int] = None,
        convert_to_pandas: bool = False,
        output_subdir: Optional[str] = None,
        **kwargs
    ) -> Union[pd.DataFrame, pl.DataFrame]:
        """
        Load error analysis DataFrame.
        
        Parameters
        ----------
        aggregated : bool, optional
            Whether to load an aggregated file, by default True
        subject : str, optional
            Subject identifier for the file, by default "splice_errors"
        chr : Optional[str], optional
            Chromosome identifier, by default None
        chunk_start : Optional[int], optional
            Start index of chunk, by default None
        chunk_end : Optional[int], optional
            End index of chunk, by default None
        convert_to_pandas : bool, optional
            Whether to convert to pandas DataFrame, by default False
        output_subdir : Optional[str], optional
            Subdirectory under meta_dir to look for file, by default None
            
        Returns
        -------
        Union[pd.DataFrame, pl.DataFrame]
            Loaded DataFrame with error analysis data
        """
        verbose = kwargs.pop('verbose', 1)

        # Create a wrapper function to capture the file loading logic
        def _load_error_analysis(chr=chr, chunk_start=chunk_start, chunk_end=chunk_end, 
                          aggregated=aggregated, subject=subject, **kws):
            try:
                # Use the file handler's method to get the file path - need to derive from error_analysis_df
                file_path = self.file_handler.get_error_analysis_file_path(
                    chr=chr, 
                    chunk_start=chunk_start, 
                    chunk_end=chunk_end, 
                    aggregated=aggregated, 
                    subject=subject
                )
                
                # Check if file exists before loading
                if not os.path.exists(file_path):
                    raise FileNotFoundError(f"Error analysis file not found: {file_path}")
                
                # Ensure chromosome columns are loaded as strings
                schema = kws.get('schema', {})
                if 'chrom' not in schema:
                    schema['chrom'] = pl.Utf8

                if verbose >= 1:
                    print(f"[i/o] Loading error analysis from {file_path}")
                
                # Load with Polars
                df = pl.read_csv(
                    file_path,
                    separator=self.separator,
                    schema_overrides=schema,
                    **{k: v for k, v in kws.items() if k != 'schema'}
                )
                
                # Convert to pandas if requested
                if convert_to_pandas:
                    df = df.to_pandas()
                    
                return df
            except Exception as e:
                print(f"[error] Failed to load error analysis from {file_path}: {str(e)}")
                raise
        
        # Use _with_output_dir to temporarily set the output directory to meta_dir/output_subdir
        return self._with_output_dir(_load_error_analysis, output_subdir=output_subdir, **kwargs)

    def save_error_analysis(
        self,
        error_df: Union[pd.DataFrame, pl.DataFrame],
        chr: Optional[str] = None,
        chunk_start: Optional[int] = None,
        chunk_end: Optional[int] = None,
        aggregated: bool = False,
        subject: str = "splice_errors",
        output_subdir: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Save error analysis DataFrame.
        
        Parameters
        ----------
        error_df : Union[pd.DataFrame, pl.DataFrame]
            Error analysis DataFrame to save
        chr : Optional[str], optional
            Chromosome, by default None
        chunk_start : Optional[int], optional
            Chunk start, by default None
        chunk_end : Optional[int], optional
            Chunk end, by default None
        aggregated : bool, optional
            Whether this is aggregated data, by default False
        subject : str, optional
            Subject identifier, by default "splice_errors"
        output_subdir : Optional[str], optional
            Output subdirectory, by default None
        
        Returns
        -------
        str
            Path to the saved file
        """
        # Do NOT convert to pandas - keep as original format
        # The underlying save_error_analysis should handle both formats
            
        return self._with_output_dir(
            self.file_handler.save_error_analysis,
            output_subdir=output_subdir,
            error_df=error_df,  # Pass the original DataFrame
            chr=chr,
            chunk_start=chunk_start,
            chunk_end=chunk_end,
            aggregated=aggregated,
            subject=subject,
            **kwargs
        )