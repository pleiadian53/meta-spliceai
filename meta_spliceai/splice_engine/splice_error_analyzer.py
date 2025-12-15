import os, sys
import numpy as np
from typing import Union, List, Set

import pandas as pd
import polars as pl
from tabulate import tabulate

from pybedtools import BedTool
from Bio import SeqIO

from .utils_bio import (
    normalize_strand
)
from .utils_df import (
    drop_columns,
    subsample_dataframe,
    is_empty,
    is_dataframe_empty, 
    concatenate_dataframes,
    join_and_remove_duplicates
)

from .utils_doc import (
    print_emphasized, 
    print_with_indent, 
    print_section_separator, 
    display, 
    display_dataframe_in_chunks
)

from .visual_analyzer import (
    create_error_bigwig
)

from meta_spliceai.splice_engine.model_evaluator import (
    ModelEvaluationFileHandler
)
from meta_spliceai.splice_engine.analyzer import Analyzer

from .sequence_featurizer import (
    extract_error_sequences, 
    extract_analysis_sequences, 
    featurize_gene_sequences,
    verify_consensus_sequences, 
    verify_consensus_sequences_via_analysis_data,
    display_feature_set,
    harmonize_features
)
from .extract_genomic_features import (
    FeatureAnalyzer, SpliceAnalyzer, 
    run_genomic_gtf_feature_extraction, 
    # compute_splice_site_distances, 
    compute_distances_to_transcript_boundaries,
    compute_distances_with_strand_adjustment,  
    compute_total_lengths,
    check_conflicting_overlaps
)
from .performance_analyzer import PerformanceAnalyzer
from .analysis_utils import (
    check_duplicates,
    check_data_integrity,
    check_and_subset_invalid_transcript_ids,
    remove_invalid_transcript_ids, 
    filter_and_validate_ids,
    handle_duplicates, 
    count_unique_ids,
    find_missing_combinations, 
    analyze_data_labels,
    impute_missing_values, 
    map_gene_names_to_ids, 
)

import pyBigWig
from tqdm import tqdm

# Refactor to utils_plot
import matplotlib.pyplot as plt 



class ErrorAnalyzer(Analyzer):
    # eval_dir = '/path/to/meta-spliceai/data/ensembl/spliceai_eval'
    # analysis_dir = '/path/to/meta-spliceai/data/ensembl/spliceai_analysis'

    schema = {
        'chrom': pl.Utf8,
        'error_type': pl.Utf8,
        'gene_id': pl.Utf8,
        'position': pl.Int64,
        'splice_type': pl.Utf8,
        'strand': pl.Utf8,
        'transcript_id': pl.Utf8,
        'window_end': pl.Int64,
        'window_start': pl.Int64
    }

    pred_type_to_label = {
        'FP': 1,  # False Positive
        'FN': 1,  # False Negative
        'TP': 0   # True Positive
    }
    
    def __init__(self, data_dir=None, *, source='ensembl', version=None, gtf_file=None, genome_fasta=None, window_size=500, **kargs):
        super().__init__()
        self.source = source
        self.version = version
        self.data_dir = data_dir or f"{Analyzer.prefix}/data/{source}"
        self.gtf_file = gtf_file or Analyzer.gtf_file
        self.genome_fasta = genome_fasta or Analyzer.genome_fasta
        self._output_dir = None
        self.window_size = window_size

        self.feature_importance_base_model = kargs.get('feature_importance_base_model', 'xgboost')
        self.importance_type = kargs.get('importance_type', 'shap')
        self.experiment = kargs.get('experiment', None)
        self.model_type = kargs.get('model_type', None)

        self.pred_type = kargs.get('pred_type', None)
        self.correct_label = kargs.get('correct_label', "TP")
        self.error_label = kargs.get('error_label', self.pred_type)
        self.splice_type = kargs.get('splice_type', None)
        self.separator = '\t'

    def retrieve_tp_data_points(self, chr=None, chunk_start=None, chunk_end=None, aggregated=True, subject="splice_tp", **kargs):
        verbose = kargs.get('verbose', 1)
        overwrite = kargs.get('overwrite', False)

        mefd = ModelEvaluationFileHandler(ErrorAnalyzer.eval_dir, separator=self.separator)
        target_pred_type = 'TP'
        tp_data_path = mefd.get_tp_data_file_path(chr, chunk_start, chunk_end, aggregated, subject=subject)

        if not overwrite and os.path.exists(tp_data_path):
            if verbose: 
                print_emphasized(f"[i/o] Loading TP data points from {tp_data_path} ...")
            df = mefd.load_tp_data_points(chr, chunk_start, chunk_end, aggregated, subject=subject)
        else: 
            # Extract TP data points
            df = extract_tp_data_points(gtf_file=self.gtf_file, window_size=self.window_size, save=True, verbose=1)
        return df

    def retrieve_fp_data_points(self, chr=None, chunk_start=None, chunk_end=None, aggregated=True, subject="splice_fp", **kargs):
        verbose = kargs.get('verbose', 1)
        overwrite = kargs.get('overwrite', False)

        mefd = ModelEvaluationFileHandler(ErrorAnalyzer.eval_dir, separator=self.separator)
        target_pred_type = 'FP'
        fp_data_path = mefd.get_fp_data_file_path(chr, chunk_start, chunk_end, aggregated, subject=subject)

        if not overwrite and os.path.exists(fp_data_path):
            if verbose: 
                print_emphasized(f"[i/o] Loading FP data points from {fp_data_path} ...")
            df = mefd.load_fp_data_points(chr, chunk_start, chunk_end, aggregated, subject=subject)
        else: 
            # Extract FP data points
            df = extract_fp_data_points(gtf_file=self.gtf_file, window_size=self.window_size, save=True, verbose=1)
        return df

    def retrieve_fn_data_points(self, chr=None, chunk_start=None, chunk_end=None, aggregated=True, subject="splice_fn", **kargs):
        verbose = kargs.get('verbose', 1)
        overwrite = kargs.get('overwrite', False)

        mefd = ModelEvaluationFileHandler(ErrorAnalyzer.eval_dir, separator=self.separator)
        target_pred_type = 'FN'
        fp_data_path = mefd.get_fp_data_file_path(chr, chunk_start, chunk_end, aggregated, subject=subject)

        if not overwrite and os.path.exists(fp_data_path):
            if verbose: 
                print_emphasized(f"[i/o] Loading FN data points from {fp_data_path} ...")
            df = mefd.load_fp_data_points(chr, chunk_start, chunk_end, aggregated, subject=subject)
        else: 
            # Extract FN data points
            df = extract_fn_data_points(gtf_file=self.gtf_file, window_size=self.window_size, save=True, verbose=1)
        return df

    # def set_analysis_output_dir(self, pred_type=None, experiment=None, **kargs):
    #     verbose = kargs.get('verbose', 1)
    #     output_dir = ErrorAnalyzer.analysis_dir

    #     self.pred_type = pred_type
    #     self.error_label = kargs.get('error_label', pred_type)
    #     self.correct_label = kargs.get('correct_label', "TP")

    #     self.experiment = experiment
    #     if experiment: 
    #         output_dir = os.path.join(output_dir, experiment)     
            
    #     if self.error_label is not None and self.correct_label is not None: 
    #         error_analysis_type = f"{self.error_label}_vs_{self.correct_label}".lower()
    #         output_dir = os.path.join(output_dir, error_analysis_type)

    #     self.model_type = kargs.get('model_type', None)
    #     if self.model_type:
    #         output_dir = os.path.join(output_dir, self.model_type)

    #     os.makedirs(output_dir, exist_ok=True) 
    #     if verbose: 
    #         print(f"[info] Output directory set to {output_dir}")
    #     return output_dir

    def set_analysis_output_dir(self, pred_type=None, experiment=None, error_label=None, correct_label=None, **kargs):
        verbose = kargs.get('verbose', 1)
        splice_type = kargs.get('splice_type', None)
        output_dir = ErrorAnalyzer.analysis_dir

        # Update instance variables only if non-null values are provided
        if pred_type is not None:
            self.pred_type = pred_type
        if error_label is not None:
            self.error_label = error_label
        if correct_label is not None:
            self.correct_label = correct_label
        if experiment is not None:
            self.experiment = experiment
        if splice_type is not None:
            self.splice_type = splice_type
        if 'model_type' in kargs and kargs['model_type'] is not None:
            self.model_type = kargs['model_type']
    
        if self.experiment: 
            output_dir = os.path.join(output_dir, self.experiment)   

        if self.splice_type is not None:
            output_dir = os.path.join(output_dir, self.splice_type)
            
        if self.error_label is not None and self.correct_label is not None: 
            error_analysis_type = f"{self.error_label}_vs_{self.correct_label}".lower()
            output_dir = os.path.join(output_dir, error_analysis_type)

        if self.model_type:
            output_dir = os.path.join(output_dir, self.model_type)

        os.makedirs(output_dir, exist_ok=True) 
        self._output_dir = output_dir

        if verbose: 
            print(f"[info] Output directory set to {output_dir}")

        return output_dir

    def get_analysis_output_dir(self, model_type=None):
        if self._output_dir is None:
            raise ValueError("Output directory has not been set. Please call set_analysis_output_dir first.")
        return self._output_dir

    def load_motif_importance(
            self, 
            input_dir=None, 
            subject=None, 
            pred_type=None, 
            format='tsv', 
            to_pandas=True, # Convert to Pandas DataFrame because feature importance is a small dataset
            **kargs
        ):
        """
        Load motif importance data from a file.

        Parameters:
        - input_dir (str): Directory where the file is located.
        - subject (str): Subject name.
        - format (str): File format ('csv' or 'tsv'). Default is 'tsv'.
        - to_pandas (bool): If True, convert the DataFrame to Pandas. Default is True.

        Returns:
        - df (pd.DataFrame or pl.DataFrame): The loaded DataFrame.
        """
        error_label = kargs.get('error_label', self.error_label)
        correct_label = kargs.get('correct_label', self.correct_label)

        # if input_dir is None: 
        #     if self.experiment is None:
        #         input_dir = ErrorAnalyzer.analysis_dir
        #     else: 
        #         input_dir = os.path.join(ErrorAnalyzer.analysis_dir, self.experiment)
        if input_dir is None: 
            model_type = kargs.get('model_type', self.model_type)
            input_dir = \
                self.set_analysis_output_dir(
                    error_label=error_label, 
                    correct_label=correct_label, 
                    model_type=model_type)  # optional: model_type

        if subject is None: 
            error_label = kargs.get('error_label', pred_type)
            correct_label = kargs.get('correct_label', "TP")    
            if error_label:
                subject = f"{error_label.lower()}_vs_{correct_label.lower()}"
            else: 
                raise ValueError("Subject or prediction type (pred_type) must be specified.")

        model_name = self.feature_importance_base_model
        importance_type = self.importance_type

        file_name = kargs.get("file_name", None)

        if file_name is None: 
            suffix = kargs.get("suffix", None)
            file_base = f'{subject}-{model_name.lower()}-motif-importance-{importance_type.lower()}'

            if suffix: 
                file_base = f"{file_base}-{suffix}"
                
            file_name = f"{file_base}.{format}"

        file_path = os.path.join(input_dir, file_name) 

        if os.path.exists(file_path):
            print(f"[i/o] Loading motif importance data from {file_path} ...")
            if format == 'tsv':
                df = pl.read_csv(file_path, separator='\t')
            elif format == 'csv':
                df = pl.read_csv(file_path)
            else:
                raise ValueError(f"Unsupported format: {format}")

            if to_pandas:
                df = df.to_pandas()

            return df
        else:
            raise FileNotFoundError(f"File not found: {file_path}")

    def load_featurized_dataset(self, aggregated=True, error_label=None, correct_label=None, splice_type=None, **kargs):
        """
        Load a featurized dataset from a file.

        Parameters
        ----------
        aggregated : bool, optional
            If True, load aggregated data (default: True).
        error_label : str, optional
            Error label to filter by (default: None).
        correct_label : str, optional
            Correct label to filter by (default: None).
        splice_type : str, optional
            Splice type to filter by (default: None).
        **kargs : dict
            Additional keyword arguments for file loading.

        Returns
        -------
        pl.DataFrame or pd.DataFrame
            Featurized dataset loaded from file.
        """
        mefd = ModelEvaluationFileHandler(ErrorAnalyzer.eval_dir, separator=self.separator)
        
        if error_label is None: 
            error_label = self.error_label
        if correct_label is None: 
            correct_label = self.correct_label
        if splice_type is None: 
            splice_type = self.splice_type
        
        return mefd.load_featurized_dataset(
            aggregated=aggregated, 
            error_label=error_label, 
            correct_label=correct_label, 
            splice_type=splice_type, 
            **kargs
        )   


def verify_consensus_sequences_v0(error_sequence_df, splice_type, window_radius=2, **kargs):
    """
    Verify if consensus sequences are present near error positions for FPs.

    Parameters:
    - error_sequence_df (pl.DataFrame): Polars DataFrame containing extracted sequences and error metadata.
      Columns: ['gene_id', 'error_type', 'position', 'sequence', ...].
    - splice_type (str): 'donor' or 'acceptor' to specify the splice site type.
    - window_radius (int): Radius around the error position to search for consensus sequences.
    - verbose (int): If > 0, print progress and summary messages.

    Returns:
    - pl.DataFrame: Polars DataFrame with an additional column 'has_consensus' indicating whether a consensus was found.
    """
    import polars as pl

    # Define consensus sequences for donor and acceptor sites
    consensus_sequences = {
        "acceptor": {"AG", "CAG", "TAG", "AAG"},
        "donor": {"GT", "GC"}
    }
    verbose = kargs.get("verbose", 1)
    col_seq = kargs.get("col_seq", "sequence")

    if splice_type not in consensus_sequences:
        raise ValueError(f"Invalid splice_type '{splice_type}'. Must be 'donor' or 'acceptor'.")

    # Validate required columns
    required_columns = {"gene_id", "error_type", "position", col_seq}
    if not required_columns.issubset(error_sequence_df.columns):
        raise ValueError(f"Input DataFrame must contain columns: {required_columns}")

    # Ensure the input DataFrame is eager (for iteration)
    error_sequence_df = error_sequence_df.collect() if isinstance(error_sequence_df, pl.LazyFrame) else error_sequence_df

    # Function to check for consensus in a neighborhood
    def has_consensus(sequence, position):
        # Error handling for out-of-bounds positions
        if position < 0 or position >= len(sequence):
            return False
        start_idx = max(0, position - window_radius)
        end_idx = min(len(sequence), position + window_radius + 1)
        neighborhood = sequence[start_idx:end_idx]
        return any(consensus in neighborhood for consensus in consensus_sequences[splice_type])

    # Apply the consensus check
    error_sequence_df = error_sequence_df.with_columns(
        pl.col(col_seq)
        .apply(lambda seq, pos: has_consensus(seq, pos), 
               return_dtype=pl.Boolean, 
               args=(pl.col("position"),))  # Pass position column as an argument
        .alias("has_consensus")
    )

    # Summary
    if verbose:
        n_fp = error_sequence_df.filter(pl.col("error_type") == "FP").shape[0]
        n_with_consensus = error_sequence_df.filter(pl.col("has_consensus")).shape[0]
        print(f"Total FPs: {n_fp}, Consensus Detected in Neighborhood: {n_with_consensus} ({n_with_consensus / n_fp:.2%})")

    return error_sequence_df


def has_consensus(sequence, position, splice_type, window_radius):
    consensus_sequences = {
        "acceptor": {"AG", "CAG", "TAG", "AAG"},
        "donor": {"GT", "GC"}
    }
    if splice_type not in consensus_sequences:
        raise ValueError(f"Invalid splice_type '{splice_type}'. Must be 'donor' or 'acceptor'.")
    
    # Handle None position
    if position is None:
        return None
    
    # Ensure position is an integer
    position = int(position)
    
    # Error handling for out-of-bounds positions
    if position < 0 or position >= len(sequence):
        return False

    start_idx = max(0, position - window_radius)  # start_idx must be >= 0
    end_idx = min(len(sequence), position + window_radius + 1)  # end_idx must be <= len(sequence)
    neighborhood = sequence[start_idx:end_idx]

    return any(consensus in neighborhood for consensus in consensus_sequences[splice_type])


def verify_consensus_sequences(analysis_sequence_df, splice_type, window_radius=None, **kwargs):
    """
    Verify consensus sequences near splice site positions and categorize their presence for TPs, FPs, and FNs.

    Parameters:
    - analysis_sequence_df (pl.DataFrame or pd.DataFrame): DataFrame with sequences and splice site metadata.
      Required columns: ['gene_id', 'pred_type', 'position', 'sequence', 'splice_type'].
    - splice_type (str): 'donor' or 'acceptor' to specify the splice site type.
    - window_radius (int): Radius around the splice site position to check for consensus sequences.
                           If None, dynamically determined by consensus sequence lengths.
    - verbose (int): Verbosity level.

    Returns:
    - pl.DataFrame: Polars DataFrame with an additional column 'has_consensus' showing presence of consensus sequences.
    """
    # Consensus sequences
    consensus_sequences = {
        "acceptor": {"AG", "CAG", "TAG", "AAG"},
        "donor": {"GT", "GC"}
    }
    verbose = kwargs.get("verbose", 1)
    col_seq = kwargs.get("col_seq", "sequence")
    col_pred_type = kwargs.get("col_pred_type", "pred_type")

    # Validate splice type
    if splice_type not in consensus_sequences:
        raise ValueError(f"Invalid splice_type '{splice_type}'. Must be 'donor' or 'acceptor'.")

    # Dynamically set window_radius based on the maximum consensus sequence length
    if window_radius is None:
        max_consensus_length = max(len(seq) for seq in consensus_sequences[splice_type])
        window_radius = max(2, max_consensus_length)  # Ensure at least radius=2 for short motifs

    if verbose:
        print(f"[info] Window radius set to {window_radius} for splice type '{splice_type}'.")

    # Ensure required columns exist
    required_columns = {"gene_id", col_pred_type, "position", col_seq, "splice_type"}
    if not required_columns.issubset(analysis_sequence_df.columns):
        raise ValueError(f"Input DataFrame must contain columns: {required_columns}")

    # Convert to Pandas if necessary
    if isinstance(analysis_sequence_df, pl.DataFrame):
        is_polars = True
        analysis_sequence_df = analysis_sequence_df.to_pandas()
    else:
        is_polars = False

    # Subset based on splice type
    analysis_sequence_df = analysis_sequence_df[analysis_sequence_df['splice_type'] == splice_type]

    # Function to check consensus
    def check_consensus(sequence, position):
        """Check for consensus sequences within ±window_radius of the splice site."""
        if not sequence or position is None or not isinstance(position, int):
            return False  # Handle invalid positions
        
        # Ensure the position is valid
        position = int(position)
        start_idx = max(0, position - window_radius)
        end_idx = min(len(sequence), position + window_radius + 1)
        neighborhood = sequence[start_idx:end_idx]

        return any(consensus in neighborhood for consensus in consensus_sequences[splice_type])

    # Apply consensus checking
    analysis_sequence_df["has_consensus"] = analysis_sequence_df.apply(
        lambda row: check_consensus(row[col_seq], row["position"]), axis=1
    )

    # Verbose summary
    if verbose:
        summary = (
            analysis_sequence_df.groupby(col_pred_type)["has_consensus"]
            .agg(["sum", "count"])
            .rename(columns={"sum": "consensus_found", "count": "total"})
        )
        print("[Summary] Consensus Sequence Analysis:")
        print(summary)

    # Convert back to Polars if necessary
    if is_polars:
        return pl.DataFrame(analysis_sequence_df)
    return analysis_sequence_df



def analyze_performance(sort_by='FP', N=10, **kargs):
    """
    Analyze splice site prediction performance and highlight top and bottom N rows.

    Parameters:
    - sort_by (str): Column to sort by (default is 'FP').
    - N (int): Number of rows to highlight (default is 10).
    - **kargs: Additional keyword arguments for configuration.

    Returns:
    - None
    """
    # import polars as pl
    from meta_spliceai.splice_engine.model_evaluator import ModelEvaluationFileHandler

    eval_dir = kargs.get('eval_dir', ErrorAnalyzer.eval_dir)
    separator = kargs.get('separator', kargs.get('sep', '\t'))

    mefd = ModelEvaluationFileHandler(eval_dir, separator=separator)
    performance_df = mefd.load_performance_df(aggregated=True)

    if isinstance(performance_df, pl.DataFrame):
        performance_df = performance_df.to_pandas()

    # Sort the DataFrame by the specified column
    sorted_df = performance_df.sort_values(by=sort_by, ascending=False)

    # Highlight the top N and bottom N rows
    top_N = sorted_df.head(N)
    bottom_N = sorted_df.tail(N)

    # Display the highlighted rows
    print(f"Top {N} rows by {sort_by}:")
    print(top_N)

    print(f"\nBottom {N} rows by {sort_by}:")
    print(bottom_N)

    # Additional metrics and analyses
    mean_value = performance_df[sort_by].mean()
    median_value = performance_df[sort_by].median()
    std_dev_value = performance_df[sort_by].std()

    print(f"\nMean {sort_by}: {mean_value}")
    print(f"Median {sort_by}: {median_value}")
    print(f"Standard Deviation {sort_by}: {std_dev_value}")

    # Comparative analysis 
    performance_baseline = kargs.get('performance_baseline', 'full_splice_performance_minus_overlapping_genes.tsv')
    performance_df_baseline = \
        mefd.load_performance_df(
            aggregated=True, 
            performance_file=performance_baseline, 
            to_pandas=True)

    if performance_df_baseline is None or performance_df_baseline.empty:
        print(f"\nBaseline performance file not found: {performance_baseline}")
        
    else: 
        # Join performance_df and performance_df_baseline on gene_id
        merged_df = performance_df.merge(performance_df_baseline, on='gene_id', suffixes=('', '_base'))

        # Print the columns of the merged DataFrame for debugging
        print(f"\nColumns in merged DataFrame: {merged_df.columns.tolist()}")

        # Subset the columns
        columns_to_retain = ['gene_id', 'n_splice_sites', 'splice_type', sort_by, f"{sort_by}_base"]
        missing_columns = [col for col in columns_to_retain if col not in merged_df.columns]
        if missing_columns:
            raise KeyError(f"Columns not found in merged DataFrame: {missing_columns}")

        merged_df = merged_df[columns_to_retain]

        # Display the merged DataFrame
        print(f"\nMerged DataFrame with {sort_by} and {sort_by}_base:")
        print(merged_df)


def analyze_error_profile(
    error_df=None,
    save_plot=False,
    plot_format='pdf',
    plot_file='error_analysis_distribution.pdf',
    verbose=True,
    **kargs
):
    """
    Analyze SpliceAI error analysis data and provide insights.

    Parameters:
    - error_df (pd.DataFrame): DataFrame containing SpliceAI error analysis data.
    - save_plot (bool): Whether to save the plot (default is False).
    - plot_format (str): Format to save the plot (default is 'pdf').
    - plot_file (str): File name to save the plot (default is 'error_analysis_distribution.pdf').
    - verbose (bool): Whether to print progress updates (default is True).

    Returns:
    - None
    """
    # import matplotlib.pyplot as plt
    # from tabulate import tabulate
    from meta_spliceai.splice_engine.model_evaluator import ModelEvaluationFileHandler

    if error_df is None:
        if verbose:
            print("Loading error analysis data...")
        
        # Replace this with your data loading logic
        eval_dir = kargs.get('eval_dir', ErrorAnalyzer.eval_dir)
        separator = kargs.get('separator', kargs.get('sep', '\t'))

        # Initialize ModelEvaluationFileHandler
        mefd = ModelEvaluationFileHandler(eval_dir, separator=separator)

        # Load the error analysis dataframe
        error_df = mefd.load_error_analysis_df(aggregated=True)
        # Columns: gene_id, error_type, position, window_start, window_end, splice_type 
        # - used to analyze the distribution of errors, extract error sequences, etc.

    if isinstance(error_df, pl.DataFrame):
        error_df = error_df.to_pandas()

    if 'error_type' not in error_df.columns or 'splice_type' not in error_df.columns:
        raise ValueError("The input DataFrame must contain 'error_type' and 'splice_type' columns.")

    total_errors = len(error_df)
    if total_errors == 0:
        raise ValueError("No errors found in the dataset.")

    # Compute error summary statistics with ratios
    summary_stats = {
        "Total Errors": [total_errors, "100%"],
        "False Positives (FP)": [
            len(error_df[error_df['error_type'] == 'FP']), 
            f"{len(error_df[error_df['error_type'] == 'FP']) / total_errors * 100:.2f}%"
        ],
        "False Negatives (FN)": [
            len(error_df[error_df['error_type'] == 'FN']), 
            f"{len(error_df[error_df['error_type'] == 'FN']) / total_errors * 100:.2f}%"
        ],
        "Donor Errors": [
            len(error_df[error_df['splice_type'] == 'donor']), 
            f"{len(error_df[error_df['splice_type'] == 'donor']) / total_errors * 100:.2f}%"
        ],
        "Acceptor Errors": [
            len(error_df[error_df['splice_type'] == 'acceptor']), 
            f"{len(error_df[error_df['splice_type'] == 'acceptor']) / total_errors * 100:.2f}%"
        ],
    }

    # Print summary statistics in a readable format
    if verbose:
        print("\nError Analysis Summary Statistics:")
        print(tabulate(
            pd.DataFrame.from_dict(summary_stats, orient='index', columns=['Count', 'Ratio']),
            headers="keys", 
            tablefmt="pretty"
        ))

    # Visualization: Errors by Type and Splice Type
    plt.figure(figsize=(10, 6))
    error_counts = error_df.groupby(['error_type', 'splice_type']).size().unstack()
    error_counts.plot(kind='bar', stacked=True, figsize=(10, 6), alpha=0.7, edgecolor='k')
    plt.title("Error Counts by Type and Splice Type")
    plt.xlabel("Error Type")
    plt.ylabel("Count")
    plt.legend(title="Splice Type")
    plt.tight_layout()

    if save_plot:
        output_dir = kargs.get('output_dir', ErrorAnalyzer.analysis_dir)
        output_path = os.path.join(output_dir, plot_file)
        plt.savefig(output_path, format=plot_format)
        if verbose:
            print(f"Plot saved as {output_path}")
    else:
        plt.show()

####################################################################################################
# Error Analysis Functions

def retrieve_splice_sites(input_dir=None, format='tsv', **kargs): 
    from meta_spliceai.splice_engine.extract_genomic_features import SpliceAnalyzer
    
    # consensus_window = kargs.get("consensus_window", 2)

    analyzer = SpliceAnalyzer(input_dir)
    df_splice = analyzer.retrieve_splice_sites()
    # Output: A Polars DataFrame: ['chrom', 'start', 'end', 'strand', 'site_type', 'gene_id', 'transcript_id']

    return df_splice

def subset_error_analysis_file(error_df=None, **kargs):
    """
    Subset the error analysis file into two DataFrames: one for FPs and the other for FNs.

    Parameters:
    - error_df (pd.DataFrame): DataFrame containing SpliceAI error analysis data.
    - file_path (str): Path to the error analysis file.

    Returns:
    - fp_df (pl.DataFrame): DataFrame containing FP examples.
    - fn_df (pl.DataFrame): DataFrame containing FN examples.

    # Example usage
    file_path = 'full_splice_errors.tsv'
    fp_df, fn_df = subset_error_analysis_file(file_path)
    print(fp_df)
    print(fn_df)
    """
    from meta_spliceai.splice_engine.model_evaluator import ModelEvaluationFileHandler

    verbose = kargs.get("verbose", 1)

    # Load the error analysis file
    # error_df = pl.read_csv(file_path, separator='\t')

    if error_df is None:
        if verbose:
            print("[info] Loading error analysis data...")
        
        # Replace this with your data loading logic
        eval_dir = kargs.get('eval_dir', ErrorAnalyzer.eval_dir)
        separator = kargs.get('separator', kargs.get('sep', '\t'))

        # Initialize ModelEvaluationFileHandler
        mefd = ModelEvaluationFileHandler(eval_dir, separator=separator)

        # Load the error analysis dataframe
        error_df = mefd.load_error_analysis_df(aggregated=True)

    # if isinstance(error_df, pl.DataFrame):
    #     error_df = error_df.to_pandas()

    # Filter out rows with null values
    error_df = error_df.drop_nulls()

    # Subset the DataFrame into FPs and FNs
    fp_df = error_df.filter(pl.col('error_type') == 'FP')
    fn_df = error_df.filter(pl.col('error_type') == 'FN')

    # Infer the window_

    return fp_df, fn_df


def extract_fp_data_points(
        error_df=None, 
        position_df=None, 
        splice_sites_df=None,
        gene_feature_df=None,
        transcript_feature_df=None, 
        gtf_file=None,
        window_size=None,
        **kargs): 
    from meta_spliceai.splice_engine.extract_genomic_features import (
        FeatureAnalyzer, 
        SpliceAnalyzer
    )

    input_dataframes = [error_df, position_df, splice_sites_df, gene_feature_df, transcript_feature_df]
    if any(df is None for df in input_dataframes):
        result_set = retrieve_splice_site_analysis_data(gtf_file)
        error_df = result_set['error_df']  # window_start, window_end
        position_df = result_set['position_df']
        splice_sites_df = result_set['splice_sites_df']
        transcript_feature_df = result_set['transcript_feature_df']
        gene_feature_df = result_set['gene_feature_df']

    # Constant parameters
    target_pred_type = 'FP'
    eval_dir = ErrorAnalyzer.eval_dir
    separator = '\t'
    
    # Get the FNs (and their predicted positions) from the position DataFrame
    position_df = position_df.filter(pl.col('pred_type') == target_pred_type)
    print(f"[debug] Number of {target_pred_type}s identified: {len(position_df)}")

    # Incoporate transcript ID and additional features chromosome, strand, splice site strengths, etc. 
    # df = align_transcript_ids(df, splice_sites_df, tolerance=2, gene_feature_df=gene_feature_df)
    # NOTE: Use the transcript feature DataFrame to align transcript IDs by transcript boundaries
    #       This is because 'position' for FP does not represent true splice site positions
    print_emphasized("[info] Aligning transcript IDs by transcript boundaries ...")
    df = align_transcript_ids_by_boundaries(
        position_df, 
        transcript_feature_df, 
        gene_feature_df, 
        position_col='position', tolerance=2)
    display_dataframe_in_chunks(df.head(5))

    print_with_indent(f"[info] Columns in transcript-aligned position_df: {list(df.columns)}", indent_level=1)
    # NOTE: Columns: ['gene_id', 'position', 'pred_type', 'score', 'splice_type', 'gene_start', 'gene_end', 'gene_strand',
    #                 'absolute_position', 'transcript_id']

    print_emphasized("[info] Merging genomic features with predicted position dataframe ...")
    # Merge splice sites with gene features to get gene start positions
    df = df.join(
        transcript_feature_df.select(['transcript_id', 'gene_id', 'start', 'end', 'transcript_length', 'chrom', 'strand']),
        on=['gene_id', 'transcript_id'],
        how='inner'
    )
    # Verify that gene_strand and strand columns have identical values
    if (df['gene_strand'] == df['strand']).all():
        # Drop the gene_strand column as it's redundant
        df = df.drop('gene_strand')
    else: 
        raise ValueError("Strand values in gene_strand and strand columns do not match.")

    display_dataframe_in_chunks(df.head(5))
    # NOTE: Available columns in genomic feature df: 
    #       'chrom', 'start', 'end', 'strand', 'transcript_id', 
    #       'transcript_name', 'transcript_type', 'transcript_length', 'gene_id'

    # Compute window coordinates using the position column
    if window_size is None: 
        window_size = (error_df['window_end'] - error_df['position']).abs().max()
        print("[info] Window size inferred from error_df:", window_size)
        df = df.with_columns([
            error_df['window_start'],
            error_df['window_end']
        ])
    else: 
        df = df.with_columns([
            (pl.col('position') - window_size).alias('window_start'),  # Window start
            (pl.col('position') + window_size).alias('window_end')  # Window end
        ])

    # Correct window_start and window_end to be within gene boundaries
    # tx_length = (pl.col('end') - pl.col('start') + 1).alias('transcript_length')
    df = df.with_columns([
        pl.when(pl.col('window_start') < 0)
        .then(0)
        .otherwise(pl.col('window_start'))
        .alias('window_start'),

        pl.when(pl.col('window_end') > pl.col('transcript_length'))
        .then(pl.col('transcript_length'))
        .otherwise(pl.col('window_end'))
        .alias('window_end')
    ])

    # Prepare final DataFrame
    # df = df.select(['gene_id', 'transcript_id', 'position', 'window_start', 'window_end', 'score', 'splice_type'])
    df = df.with_columns(pl.lit(target_pred_type).alias('pred_type'))  # Add pred_type column for true positives

    # Display the FN-specific DataFrame
    display_dataframe_in_chunks(df.head(5))

    # Save the output DataFrame
    if kargs.get('save', False):
        # Initialize ModelEvaluationFileHandler
        mefd = ModelEvaluationFileHandler(eval_dir, separator=separator)
        output_path = mefd.save_tp_data_points(df, aggregated=True, subject=f"splice_{target_pred_type}".lower())
        print_emphasized(f"Saved {target_pred_type} data points to {output_path}")

    return df


# Todo: refactor to class method
def extract_fn_data_points(
        error_df=None, 
        position_df=None, 
        splice_sites_df=None,
        gene_feature_df=None,
        transcript_feature_df=None, 
        gtf_file=None,
        window_size=None,
        **kargs):
    
    from meta_spliceai.splice_engine.extract_genomic_features import (
        FeatureAnalyzer, 
        SpliceAnalyzer
    )

    input_dataframes = [error_df, position_df, splice_sites_df, gene_feature_df, transcript_feature_df]
    if any(df is None for df in input_dataframes):
        result_set = retrieve_splice_site_analysis_data(gtf_file, feature_type='transcript')
        error_df = result_set['error_df']  # window_start, window_end
        position_df = result_set['position_df']
        splice_sites_df = result_set['splice_sites_df']
        transcript_feature_df = result_set['transcript_feature_df']
        gene_feature_df = result_set['gene_feature_df']

    # Constant parameters
    target_pred_type = 'FN'
    eval_dir = ErrorAnalyzer.eval_dir
    separator = '\t'
    
    # Get the FNs (and their predicted positions) from the position DataFrame
    position_df = position_df.filter(pl.col('pred_type') == target_pred_type)
    print(f"[debug] Number of {target_pred_type}s identified: {len(position_df)}")

    # Incoporate transcript ID and additional features chromosome, strand, splice site strengths, etc. 
    df = align_transcript_ids(position_df, splice_sites_df, gene_feature_df=gene_feature_df, tolerance=2)
    print(f"[info] Columns in transcript-aligned position_df: {list(df.columns)}")
    # NOTE: Use the transcript feature DataFrame to align transcript IDs by transcript boundaries
    #       This is because 'position' for FP, FN does not represent true splice site positions
    # print_emphasized("[info] Aligning transcript IDs by transcript boundaries ...")
    # df = align_transcript_ids_by_boundaries(
    #     position_df, 
    #     transcript_feature_df, 
    #     gene_feature_df, 
    #     position_col='position', tolerance=2)
    display_dataframe_in_chunks(df.head(5))

    print_with_indent(f"[info] Columns in transcript-aligned position_df: {list(df.columns)}", indent_level=1)
    # NOTE: if align_transcript_ids() is applied
    # Columns: ['gene_id', 'position', 'pred_type', 'score', 'splice_type', 'start', 'end', 'strand', 'absolute_position',
    #                'transcript_id']
    #          - 'start' and 'end' represent the gene boundaries
    #       if align_transcript_ids_by_boundaries() is applied
    # NOTE: Columns: ['gene_id', 'position', 'pred_type', 'score', 'splice_type', 'gene_start', 'gene_end', 'gene_strand',
    #                 'absolute_position', 'transcript_id']

    # Distinguish between gene and transcript features
    df = df.rename({
        'start': 'gene_start',
        'end': 'gene_end', 
        'strand': 'gene_strand'
    })

    print_emphasized("[info] Merging genomic features with predicted position dataframe ...")
    # Merge splice sites with gene features to get gene start positions
    df = df.join(
        transcript_feature_df.select(['transcript_id', 'gene_id', 'start', 'end', 'transcript_length', 'chrom', 'strand']),
        on=['gene_id', 'transcript_id'],
        how='inner'
    )
    # Verify that gene_strand and strand columns have identical values
    if (df['gene_strand'] == df['strand']).all():
        # Drop the gene_strand column as it's redundant
        df = df.drop('gene_strand')
    else: 
        raise ValueError("Strand values in gene_strand and strand columns do not match.")

    display_dataframe_in_chunks(df.head(5), title="Merged DataFrame with Transcript Features")
    # NOTE: Available columns in genomic feature df: 
    #       'chrom', 'start', 'end', 'strand', 'transcript_id', 
    #       'transcript_name', 'transcript_type', 'transcript_length', 'gene_id'

    # Compute window coordinates using the position column
    if window_size is None: 
        window_size = (error_df['window_end'] - error_df['position']).abs().max()
        print("[info] Window size inferred from error_df:", window_size)
        df = df.with_columns([
            error_df['window_start'],
            error_df['window_end']
        ])
    else: 
        df = df.with_columns([
            (pl.col('position') - window_size).alias('window_start'),  # Window start
            (pl.col('position') + window_size).alias('window_end')  # Window end
        ])

    # Correct window_start and window_end to be within gene boundaries
    # tx_length = (pl.col('end') - pl.col('start') + 1).alias('transcript_length')
    df = df.with_columns([
        pl.when(pl.col('window_start') < 0)
        .then(0)
        .otherwise(pl.col('window_start'))
        .alias('window_start'),

        pl.when(pl.col('window_end') > pl.col('transcript_length'))
        .then(pl.col('transcript_length'))
        .otherwise(pl.col('window_end'))
        .alias('window_end')
    ])

    # Prepare final DataFrame
    # df = df.select(['gene_id', 'transcript_id', 'position', 'window_start', 'window_end', 'score', 'splice_type'])
    df = df.with_columns(pl.lit(target_pred_type).alias('pred_type'))  # Add pred_type column for true positives
    print_emphasized(f"[info] Columns of FN-specific DataFrame: {list(df.columns)}")

    # Display the FN-specific DataFrame
    display_dataframe_in_chunks(df.head(5), title="FN-specific DataFrame")

    # Save the output DataFrame
    if kargs.get('save', False):
        # Initialize ModelEvaluationFileHandler
        mefd = ModelEvaluationFileHandler(eval_dir, separator=separator)
        output_path = mefd.save_tp_data_points(df, aggregated=True, subject=f"splice_{target_pred_type}".lower())
        print_emphasized(f"Saved {target_pred_type} data points to {output_path}")

    return df


def extract_tp_data_points(
        error_df=None, 
        position_df=None, 
        splice_sites_df=None,
        gene_feature_df=None, 
        gtf_file=None,
        window_size=500,
        **kargs):
    """
    Derive TP-specific data points from the precomputed splice sites data.

    Parameters:
    - gtf_file_path (str): Path to the GTF file containing gene annotations.
    - window_size (int): 
      Size of the context window surrounding the true splice site (default is 500).
    - gene_feature_df (pd.DataFrame): DataFrame containing gene features.

      Note that this should be consistent with the error window used in model evaluation at: 
      splice_engine.evaluate_models.evaluate_splice_site_errors()     
 
    Returns:
    - tp_df (pl.DataFrame): DataFrame containing TP examples.
    """
    from meta_spliceai.splice_engine.extract_genomic_features import (
        FeatureAnalyzer, 
        SpliceAnalyzer
    )

    input_dataframes = [error_df, position_df, splice_sites_df, gene_feature_df]
    if any(df is None for df in input_dataframes):
        result_set = retrieve_splice_site_analysis_data(gtf_file)
        error_df = result_set['error_df']
        position_df = result_set['position_df']
        splice_sites_df = result_set['splice_sites_df']
        gene_feature_df = result_set['gene_feature_df']
        transcript_feature_df = result_set['transcript_feature_df']

    # Constant parameters
    target_pred_type = 'TP'
    eval_dir = ErrorAnalyzer.eval_dir
    separator = '\t'
    
    # position_df = position_df[position_df['pred_type'] == pred_type]
    splice_hit_df = position_df = position_df.filter(pl.col('pred_type') == target_pred_type)
    print(f"[debug] Number of {target_pred_type}s identified: {len(splice_hit_df)}")

    # Incoporate transcript ID and additional features chromosome, strand, splice site strengths, etc. 
    print_emphasized("[info] Aligning transcript IDs by splice site positions ...")
    df = align_transcript_ids(position_df, splice_sites_df, gene_feature_df=gene_feature_df, tolerance=2)
    print(f"[info] Columns in transcript-aligned position_df: {list(df.columns)}")
    # NOTE: Columns: 
    #      ['gene_id', 'position', 'pred_type', 'score', 'splice_type', 'start', 'end', 'strand', 
    #       'absolute_position', 'transcript_id']

    # Distinguish between gene and transcript features
    df = df.rename({
        'start': 'gene_start',
        'end': 'gene_end', 
        'strand': 'gene_strand'  # Rename strand to gene_strand; to drop, test only
    })

    # Step 3: Merge splice sites with transcript features
    print_emphasized("[info] Merging transcript features with predicted position dataframe ...")

    # Select columns from gene_feature_df that are not already in df
    # columns_to_add = ['gene_id',]
    # columns_to_add += [col for col in ['start', 'end', 'chrom', 'strand'] if col not in df.columns] 
    # df = df.join(
    #     gene_feature_df.select(columns_to_add),
    #     on='gene_id',
    #     how='inner'
    # )

    df = df.join(
        transcript_feature_df.select(['transcript_id', 'gene_id', 'start', 'end', 'transcript_length', 'chrom', 'strand']),
        on=['gene_id', 'transcript_id'],
        how='inner'
    )
    # Verify that gene_strand and strand columns have identical values
    if (df['gene_strand'] == df['strand']).all():
        # Drop the gene_strand column as it's redundant
        df = df.drop('gene_strand')
    else: 
        raise ValueError("Strand values in gene_strand and strand columns do not match.")
    display_dataframe_in_chunks(df.head(5), title="Merged DataFrame with Transcript Features")

    # ------------------------------------------------

    # Compute window coordinates using the position column
    df = df.with_columns([
        (pl.col('position') - window_size).alias('window_start'),  # Window start
        (pl.col('position') + window_size).alias('window_end')  # Window end
    ])

    # Correct window_start and window_end to be within gene boundaries:
    gene_length = (pl.col('end') - pl.col('start') + 1).alias('gene_length')
    df = df.with_columns([
        pl.when(pl.col('window_start') < 0)
        .then(0)
        .otherwise(pl.col('window_start'))
        .alias('window_start'),

        pl.when(pl.col('window_end') > gene_length)
        .then(gene_length)
        .otherwise(pl.col('window_end'))
        .alias('window_end')
    ])

    # Prepare final DataFrame
    # df = df.select(['gene_id', 'position', 'window_start', 'window_end', 'score', 'splice_type'])
    df = df.with_columns(pl.lit(target_pred_type).alias('pred_type'))  # Add pred_type column for true positives
    print_emphasized(f"[info] Columns of TP-specific DataFrame: {list(df.columns)}")
    # NOTE: Columns: 
    #       ['gene_id', 'position', 'pred_type', 'score', 'splice_type', 'gene_start', 'gene_end', 
    #        'absolute_position', 'transcript_id', 'start', 'end', 'transcript_length', 
    #        'chrom', 'strand', 'window_start', 'window_end']
    #       - 'position' the splice site predicted position (relative), which for TPs is the true splice site position
    #       - 'absolute_position' is the true splice site position (absolute coordinate)

    # Display the TP DataFrame
    display_dataframe_in_chunks(df.head(5), title="TP-specific DataFrame")

    # Save the output DataFrame
    if kargs.get('save', False):
        # Initialize ModelEvaluationFileHandler
        mefd = ModelEvaluationFileHandler(eval_dir, separator=separator)
        output_path = mefd.save_tp_data_points(df, aggregated=True, subject="splice_tp")
        print_emphasized(f"Saved {target_pred_type} data points to {output_path}")

    return df


def retrieve_splice_site_analysis_data_v0(gtf_file=None, **kargs): 
    from meta_spliceai.splice_engine.extract_genomic_features import (
            FeatureAnalyzer, 
            SpliceAnalyzer
        )

    print_emphasized(f"[i/o] Loading error analysis data ...")

    # Replace this with your data loading logic
    eval_dir = ErrorAnalyzer.eval_dir
    separator = kargs.get('separator', kargs.get('sep', '\t'))
    # genomic_feature_type = kargs.get('feature_type', 'gene')
    result_set = {}

    # Initialize ModelEvaluationFileHandler
    mefd = ModelEvaluationFileHandler(eval_dir, separator=separator)

    # Define datasets to load
    available_datasets = {
        'error_df': lambda: mefd.load_error_analysis_df(aggregated=True, schema=ErrorAnalyzer.schema),
        'gene_features': lambda: FeatureAnalyzer(gtf_file=gtf_file).retrieve_gene_features(),
        'transcript_features': lambda: FeatureAnalyzer(gtf_file=gtf_file).retrieve_transcript_features(),
        'splice_sites_df': lambda: SpliceAnalyzer().retrieve_splice_sites(column_names={'site_type': 'splice_type'}),
        'position_df': lambda: mefd.load_splice_positions(aggregated=True, schema=ErrorAnalyzer.schema)
    }

    # Load the error analysis dataframe
    result_set['error_df'] = error_df = mefd.load_error_analysis_df(aggregated=True, schema=ErrorAnalyzer.schema)

    fp_df, fn_df = subset_error_analysis_file(error_df=error_df, **kargs)
    gtf_file_path = gtf_file

    # Load gene features
    fa = FeatureAnalyzer(gtf_file=gtf_file_path)
    # if genomic_feature_type == 'gene': 

    print_emphasized(f"[i/o] Retrieving gene features ...")
    result_set['gene_feature_df'] = gene_features = \
        fa.retrieve_gene_features()  # Output is a Polars DataFrame
    print(f"[info] Columns in gene_features: {gene_features.columns}")
    # NOTE: gene_features is a pl.Dataframe
    #  Columns: ['start', 'end', 'score', 'strand', 'gene_id', 'gene_name', 'gene_type', 'gene_length', 'chrom']
    display_dataframe_in_chunks(gene_features.head(5))
    assert (gene_features['gene_length'] == (gene_features['end'] - gene_features['start'] + 1)).all(), \
        "Assertion failed: gene_length is not equal to end - start + 1"
    # elif genomic_feature_type.startswith( ('transcript', 'tx') ):

    print_emphasized(f"[i/o] Retrieving transcript features ...")
    result_set['transcript_feature_df'] = transcript_features = \
        fa.retrieve_transcript_features()  # Output is a Polars DataFrame
    print(f"[info] Columns in transcript_features: {transcript_features.columns}")
    # NOTE: transcript_features is a pl.Dataframe
    # Columns: 
    #       ['chrom', 'start', 'end', 
    #        'strand', 'transcript_id', 'transcript_name', 'transcript_type', 'transcript_length', 'gene_id']
    display_dataframe_in_chunks(transcript_features.head(5))

    # Load the splice sites file
    print_emphasized(f"[i/o] Loading splice sites data ...")
    analyzer = SpliceAnalyzer()
    result_set['splice_sites_df'] = splice_sites_df = \
        analyzer.retrieve_splice_sites(column_names={'site_type': 'splice_type'})  # Output is a Polars DataFrame
    
    # Standardize column names: change 'site_type' to 'splice_type'
    # if 'site_type' in splice_sites_df.columns:
    #     splice_sites_df = splice_sites_df.rename({'site_type': 'splice_type'})

    print(f"[info] Columns in splice_sites_df: {splice_sites_df.columns}")
    # NOTE: Columns: ['chrom', 'start', 'end', 'strand', 'splice_type', 'gene_id', 'transcript_id']
    display_dataframe_in_chunks(splice_sites_df.head(5))

    # Load error analysis data
    print_emphasized(f"[i/o] Loading prediction & analysis data ...")
    mefd = ModelEvaluationFileHandler(ErrorAnalyzer.eval_dir, separator='\t')
    result_set['position_df'] = position_df = \
        mefd.load_splice_positions(aggregated=True, schema=ErrorAnalyzer.schema)
    print("[info] Columns in position_df:", position_df.columns)
    display_dataframe_in_chunks(position_df.head(5))
    # Columns: ['gene_id', 'position', 'pred_type', 'score', 'splice_type']

    return result_set


def retrieve_splice_site_analysis_data(gtf_file=None, load_datasets=None, **kargs):
    from meta_spliceai.splice_engine.extract_genomic_features import (
        FeatureAnalyzer, 
        SpliceAnalyzer
    )

    print_emphasized(f"[i/o] Loading error analysis data ...")

    # Replace this with your data loading logic
    eval_dir = ErrorAnalyzer.eval_dir
    separator = kargs.get('separator', kargs.get('sep', '\t'))
    result_set = {}

    # Initialize ModelEvaluationFileHandler
    mefd = ModelEvaluationFileHandler(eval_dir, separator=separator)

    # Define datasets to load
    available_datasets = {
        'error_df': lambda: mefd.load_error_analysis_df(aggregated=True, schema=ErrorAnalyzer.schema),
        'gene_features': lambda: FeatureAnalyzer(gtf_file=gtf_file).retrieve_gene_features(),
        'transcript_features': lambda: FeatureAnalyzer(gtf_file=gtf_file).retrieve_transcript_features(),
        'splice_sites_df': lambda: SpliceAnalyzer().retrieve_splice_sites(column_names={'site_type': 'splice_type'}),
        'position_df': lambda: mefd.load_splice_positions(aggregated=True, schema=ErrorAnalyzer.schema)
    }

    # Add aliases for backward compatibility (see v0)
    aliases = {
        'gene_feature_df': 'gene_features',
        'transcript_feature_df': 'transcript_features'
    }

    # Load only the specified datasets
    if load_datasets is None:
        load_datasets = available_datasets.keys()

    # Resolve aliases
    load_datasets = [aliases.get(dataset, dataset) for dataset in load_datasets]

    for dataset in load_datasets:
        if dataset in available_datasets:
            result_set[dataset] = available_datasets[dataset]()
            print(f"[info] Loaded {dataset} with columns: {result_set[dataset].columns}")
            display_dataframe_in_chunks(result_set[dataset].head(5))
        else:
            print(f"[warning] Dataset {dataset} is not available for loading.")

    return result_set


def align_transcript_ids_for_tn_analysis(position_df, gtf_file=None, **kargs):
    """
    Identify transcript IDs for TN positions by matching them to transcript boundaries.

    This closely parallels `align_transcript_ids_for_fp_analysis()`, except it
    looks for pred_type == 'TN' instead of 'FP'.

    Parameters
    ----------
    position_df : pl.DataFrame (or pd.DataFrame)
        Must include columns: ['gene_id', 'position', 'pred_type', 'strand', 'score'].
        The function filters out rows that are not 'TN' in `pred_type`.
    gtf_file : str or None
        Path to the reference GTF file, or None to use defaults in your code.
    col_pred_type : str
        Column in `position_df` to check for 'TN' label. Defaults to 'pred_type'.
    col_tid : str
        Column name for transcript ID.  Defaults to 'transcript_id'.

    Returns
    -------
    df : pl.DataFrame
        A copy of `position_df` (restricted to TN rows) with an added column
        `transcript_id` for each matched transcript.  If a position belongs to multiple
        transcripts, you get multiple rows.  If you already had some valid transcript IDs,
        they remain intact.
    unmatched_genes : set
        Set of gene IDs for which transcripts were not found or matched.

    Notes
    -----
    - Under the hood, calls `retrieve_splice_site_analysis_data()` to get the gene_features
      and transcript_features data.
    - Then calls `align_transcript_ids_by_boundaries()`.  The logic is identical to
      your FP alignment, just substituting `pred_type='TN'`.
    - If you already have transcript_id in some rows, those remain. Rows needing transcript
      alignment get expanded if multiple transcripts match the absolute position.
    """
    col_pred_type = kargs.get('col_pred_type', 'pred_type')
    col_tid = kargs.get('col_tid', 'transcript_id')
    tolerance = kargs.get('tolerance', 10)  # or whichever default you prefer

    # Retrieve the GTF-based feature sets
    result_set = retrieve_splice_site_analysis_data(
        gtf_file, 
        load_datasets=['gene_features', 'transcript_features', 'splice_sites_df']
    )
    gene_feature_df = result_set['gene_features']
    transcript_feature_df = result_set['transcript_features']

    # Subset the DataFrame into TN
    has_tn = True
    if col_pred_type in position_df.columns:
        target_pred_type = 'TN'
        position_df = position_df.filter(pl.col(col_pred_type) == target_pred_type)
        print(f"[debug] Number of {target_pred_type} positions identified: {len(position_df)}")

        if position_df.is_empty():
            has_tn = False
    else:
        # If pred_type does not exist, assume all are TN?  Or raise an error.
        print("[warning] Column for pred_type not found. Assuming all are TN.")
        pass

    # Now align transcript IDs by boundaries

    if has_tn: 
        print_emphasized("[info] Aligning transcript IDs by transcript boundaries for TN ...")

        if col_tid not in position_df.columns:
            # If there's no transcript_id column, we call align_transcript_ids_by_boundaries
            df, unmatched_genes = align_transcript_ids_by_boundaries_enhanced(
                position_df, 
                transcript_feature_df, 
                gene_feature_df, 
                position_col='position',
                policy='longest',             # or 'closest'
                max_transcripts=1,           # default to 1
                tolerance=tolerance,
                return_unmatched=True,       # if True, also return unmatched genes
                remove_extra_cols=True, 
                min_distance_boundary=10,
                drop_unmatched_if_too_close=True
            )

        else:
            # If transcript_id already exists, we do a partial approach:
            valid_position_df, invalid_position_df = \
                filter_and_validate_ids(position_df, col_tid=col_tid,
                                        valid_prefixes=None, verbose=1, return_invalid_rows=True)

            if is_empty(invalid_position_df):
                print("[info] 'transcript_id' column already exists in the input DataFrame for all TN rows.")
                df = position_df
                unmatched_genes = set()
            else:
                # For those missing transcript_id, call align_transcript_ids_by_boundaries
                invalid_position_df = drop_columns(invalid_position_df, [col_tid])
                print("[info] 'transcript_id' column exists but a subset of TN rows are missing it.")
                
                df_aligned, unmatched_genes = align_transcript_ids_by_boundaries_enhanced(
                    invalid_position_df, 
                    transcript_feature_df, 
                    gene_feature_df, 
                    position_col='position',
                    policy='longest',             # or 'closest'
                    max_transcripts=1,           # default to 1
                    tolerance=tolerance,
                    return_unmatched=True,       # if True, also return unmatched genes
                    remove_extra_cols=True, 
                    min_distance_boundary=10,
                    drop_unmatched_if_too_close=True
                )
                df = concatenate_dataframes(valid_position_df, df_aligned, axis=0)

    else:  
        print("[info] No TN positions found. Returning original DataFrame.")
        df = position_df
        unmatched_genes = set()

    print(f"[info] Columns in transcript-aligned TN DataFrame: {list(df.columns)}")
    display_dataframe_in_chunks(df, num_rows=5)

    return df, unmatched_genes


def align_transcript_ids_for_fp_analysis(position_df, gtf_file=None, **kargs):

    col_pred_type = kargs.get('col_pred_type', 'pred_type')
    # test_mode = kargs.get('test_mode', False)
    col_tid = kargs.get('col_tid', 'transcript_id')

    result_set = retrieve_splice_site_analysis_data(gtf_file, load_datasets=['gene_features', 'transcript_features'])
    # splice_sites_df = result_set['splice_sites_df']
    gene_feature_df = result_set['gene_features']
    transcript_feature_df = result_set['transcript_features']

    # Subset the DataFrame into FPs
    if col_pred_type in position_df.columns:
        target_pred_type = 'FP'
        position_df = position_df.filter(pl.col(col_pred_type) == target_pred_type)
        print(f"[debug] Number of {target_pred_type}s identified: {len(position_df)}")
    else: 
        # Assume all positions are FPs
        pass

    # Use the transcript feature DataFrame to align transcript IDs by transcript boundaries
    # This is because 'position' for FP does not represent true splice site positions
    print_emphasized("[info] Aligning transcript IDs by transcript boundaries ...")

    if col_tid not in position_df.columns:
        df, unmatched_genes = \
            align_transcript_ids_by_boundaries(
                position_df, 
                transcript_feature_df,  # To access the transcript boundaries
                gene_feature_df, 
                position_col='position', tolerance=2, return_unmatched=True)
    else: 
        # It's possible that a subset of transcript IDs are missing even though the column exists
        valid_position_df, invalid_position_df = \
            filter_and_validate_ids(position_df, col_tid=col_tid, valid_prefixes=None, verbose=1, return_invalid_rows=True)

        if is_empty(invalid_position_df):
            print("[info] 'transcript_id' column already exists in the input DataFrame and all have valid entries.")
            df = position_df
            unmatched_genes = set()
        else: 
            invalid_position_df = drop_columns(invalid_position_df, [col_tid])

            print("[info] 'transcript_id' column already exists but a subset of transcript IDs are missing.")
            df, unmatched_genes = \
                align_transcript_ids_by_boundaries(
                    invalid_position_df, 
                    transcript_feature_df,  # To access the transcript boundaries
                    gene_feature_df, 
                    position_col='position', tolerance=2, return_unmatched=True)        
            
            # Combine valid_position_df with df
            df = concatenate_dataframes(valid_position_df, df, axis=0)
 
    print(f"[info] Columns in transcript-aligned position_df: {list(df.columns)}")
    display_dataframe_in_chunks(df, num_rows=5) 

    return df, unmatched_genes


def align_transcript_ids_for_fn_analysis(position_df, gtf_file=None, **kargs):
    
    col_pred_type = kargs.get('col_pred_type', 'pred_type')
    col_tid = kargs.get('col_tid', 'transcript_id')
    throw_exception = kargs.get('throw_exception', True)
    result_set = retrieve_splice_site_analysis_data(gtf_file, load_datasets=['gene_features', 'splice_sites_df'])
    splice_sites_df = result_set['splice_sites_df']
    gene_feature_df = result_set['gene_features']

    # Subset the DataFrame into FNs
    if col_pred_type in position_df.columns:
        target_pred_type = 'FN'
        position_df = position_df.filter(pl.col(col_pred_type) == target_pred_type)
        print(f"[debug] Number of {target_pred_type}s identified: {len(position_df)}")
    else: 
        # Assume that position_df contains only FNs
        pass

    # Align transcript IDs by splice site positions
    if col_tid not in position_df.columns:
        df, unmatched_genes = \
            align_transcript_ids(
                position_df, 
                splice_sites_df, 
                gene_feature_df=gene_feature_df, 
                tolerance=3, throw_exception=throw_exception, return_unmatched=True)
    else: 
        # It's possible that a subset of transcript IDs are missing even though the column exists
        valid_position_df, invalid_position_df = \
            filter_and_validate_ids(position_df, col_tid=col_tid, valid_prefixes=None, verbose=1, return_invalid_rows=True)

        if is_empty(invalid_position_df):
            print("[info] 'transcript_id' column already exists in the input DataFrame and all have valid entries.")
            df = position_df
            unmatched_genes = set()
        else: 
            print("[info] 'transcript_id' column already exists but a subset of transcript IDs are missing.")
            invalid_position_df = drop_columns(invalid_position_df, [col_tid])

            df, unmatched_genes = \
                align_transcript_ids(
                    invalid_position_df, 
                    splice_sites_df, 
                    gene_feature_df=gene_feature_df, 
                    tolerance=3, throw_exception=throw_exception, return_unmatched=True)      
            
            # Combine valid_position_df with df
            df = concatenate_dataframes(valid_position_df, df, axis=0)

    print(f"[info] Columns in transcript-aligned position_df: {list(df.columns)}")
    # NOTE: ['gene_id', 'transcript_id', 'chrom', 'position', 'pred_type', 'score', 'splice_type', 
    #        'strand', 'window_end', 'window_start', 'sequence', 'strand_right']

    # Subset the desired columns
    desired_columns = ['gene_id', 'transcript_id', 'chrom', 'position', 'pred_type', 'score', 'splice_type', 'strand']
    df_subset = df.select([col for col in desired_columns if col in df.columns])

    display_dataframe_in_chunks(df_subset, num_rows=5)

    return df, unmatched_genes


def align_transcript_ids_for_tp_analysis(position_df, gtf_file=None, **kargs): 

    col_pred_type = kargs.get('col_pred_type', 'pred_type')
    col_tid = kargs.get('col_tid', 'transcript_id')
    throw_exception = kargs.get('throw_exception', True)
    result_set = retrieve_splice_site_analysis_data(gtf_file, load_datasets=['gene_features', 'splice_sites_df'])
    splice_sites_df = result_set['splice_sites_df']
    gene_feature_df = result_set['gene_features']

    # Subset the DataFrame into TPs
    if col_pred_type in position_df.columns:
        target_pred_type = 'TP'
        position_df = position_df.filter(pl.col(col_pred_type) == target_pred_type)
        print(f"[debug] Number of {target_pred_type}s identified: {len(position_df)}")
    else: 
        print("[warning] Column for pred_type not found. Assuming all are TPs.")
        # Assume that position_df contains only TPs
        pass

    # Align transcript IDs by splice site positions
    if col_tid not in position_df.columns:
        df, unmatched_genes = \
            align_transcript_ids(
                position_df, 
                splice_sites_df, 
                gene_feature_df=gene_feature_df, 
                tolerance=3, throw_exception=throw_exception, return_unmatched=True)
    else:
        # It's possible that a subset of transcript IDs are missing even though the column exists
        valid_position_df, invalid_position_df = \
            filter_and_validate_ids(position_df, col_tid=col_tid, valid_prefixes=None, verbose=1, return_invalid_rows=True)

        if is_empty(invalid_position_df):
            print("[info] 'transcript_id' column already exists in the input DataFrame and all have valid entries.")
            df = position_df
            unmatched_genes = set()
        else: 
            print("[info] 'transcript_id' column already exists but a subset of transcript IDs are missing.")

            invalid_position_df = drop_columns(invalid_position_df, [col_tid])

            df, unmatched_genes = \
                align_transcript_ids(
                    invalid_position_df, 
                    splice_sites_df, 
                    gene_feature_df=gene_feature_df, 
                    tolerance=3, throw_exception=throw_exception, return_unmatched=True)      
            
            # Combine valid_position_df with df
            df = concatenate_dataframes(valid_position_df, df, axis=0)

    print(f"[info] Columns in transcript-aligned position_df: {list(df.columns)}")

    # Subset the desired columns
    desired_columns = ['gene_id', 'transcript_id', 'chrom', 'position', 'pred_type', 'score', 'splice_type', 'strand']
    df_subset = df.select([col for col in desired_columns if col in df.columns])

    display_dataframe_in_chunks(df_subset, num_rows=5)

    return df, unmatched_genes


def align_transcript_ids_v0(position_df, splice_sites_df, tolerance=2):
    """
    Align transcript IDs to the position dataframe using splice site annotations.

    Parameters:
    - position_df (pd.DataFrame): DataFrame containing splice site error information.
      Columns: ['gene_id', 'position', 'pred_type', 'score', 'splice_type'].
      The `position` column represents the true splice site position.
    - splice_sites_df (pd.DataFrame): DataFrame containing splice site annotations.
      Columns: ['gene_id', 'position', 'transcript_id', ...].
      The `position` column represents the true splice site position in the annotations.
    - tolerance (int): Number of nucleotides to allow for matching splice site positions (default: 0).

    Returns:
    - pd.DataFrame: A new DataFrame with an added `transcript_id` column, representing the transcript
      associated with each splice site position.
    """
    # Ensure necessary columns are present
    required_position_cols = ['gene_id', 'position', 'pred_type', 'score', 'splice_type']
    required_splice_cols = ['gene_id', 'position', 'transcript_id']
    for col in required_position_cols:
        if col not in position_df.columns:
            raise ValueError(f"Column '{col}' is missing from position_df.")
    for col in required_splice_cols:
        if col not in splice_sites_df.columns:
            raise ValueError(f"Column '{col}' is missing from splice_sites_df.")

    if isinstance(position_df, pl.DataFrame):
        position_df = position_df.to_pandas()
    if isinstance(splice_sites_df, pl.DataFrame):
        splice_sites_df = splice_sites_df.to_pandas()

    # Merge position_df with splice_sites_df on gene_id and position
    merged_df = position_df.merge(
        splice_sites_df,
        on=['gene_id', 'position'],
        how='left',
        suffixes=('', '_annot')
    )
    
    # Test: merged_df should be non-empty 
    assert not merged_df.empty, "Merged DataFrame is empty."

    # Handle tolerance
    if tolerance > 0:
        # Identify rows where positions do not match within tolerance
        unmatched_rows = merged_df[merged_df['transcript_id'].isna()]
        for idx, row in tqdm(unmatched_rows.iterrows(), total=unmatched_rows.shape[0], desc="Processing unmatched rows"):
            # Look for matching positions within tolerance
            gene_id = row['gene_id']
            pos = row['position']
            splice_candidates = splice_sites_df[
                (splice_sites_df['gene_id'] == gene_id) &
                (abs(splice_sites_df['position'] - pos) <= tolerance)
            ]
            if not splice_candidates.empty:
                # Assign the first matched transcript ID within tolerance
                merged_df.at[idx, 'transcript_id'] = splice_candidates.iloc[0]['transcript_id']

    return merged_df


def align_transcript_ids(
        df, splice_sites_df, gene_feature_df, 
        position_col='position', tolerance=2, throw_exception=True, return_unmatched=False):
    """
    Align transcript IDs to a general dataframe using splice site annotations, accounting
    for absolute vs relative positions and tolerance for minor discrepancies. Includes a
    validation check to ensure every gene in the input dataframe has a match in the splice
    site annotations.

    Parameters:
    - df (pd.DataFrame): Input DataFrame with a position-like column for alignment.
      Required columns: ['gene_id', position_col, 'strand'], where `position_col` represents
        a relative position within the gene, and `strand` specifies the gene's strand.
    - splice_sites_df (pd.DataFrame): DataFrame containing splice site annotations.
      Required columns: ['gene_id', 'position', 'transcript_id'], where `position`
        is the absolute genomic coordinate of the splice site.
    - position_col (str): Name of the column in `df` representing the relative position.
    - tolerance (int): Number of nucleotides to allow for matching splice site positions (default: 0).
    - gene_feature_df (pd.DataFrame): DataFrame containing gene-level features, including
      'gene_id', 'start', and 'end' for determining the absolute positions.

    Returns:
    - pd.DataFrame: A new DataFrame with an added `transcript_id` column, representing the transcript
      associated with each splice site position.
    """
    # Base case: Check if transcript_id column exists and has no null values
    if 'transcript_id' in df.columns and df['transcript_id'].notna().all():
        print("[info] 'transcript_id' already exists and is non-null. Returning the original DataFrame.")
        if return_unmatched: 
            return df, set()

        return df

    # Ensure required columns are present
    required_gene_cols = ['gene_id', 'start', 'end', 'strand']
    for col in required_gene_cols:
        if col not in gene_feature_df.columns:
            raise ValueError(f"Column '{col}' is missing from gene_feature_df.")
    required_splice_cols = ['gene_id', 'position', 'transcript_id']
    for col in required_splice_cols:
        if col not in splice_sites_df.columns:
            raise ValueError(f"Column '{col}' is missing from splice_sites_df.")
    if position_col not in df.columns:
        raise ValueError(f"Column '{position_col}' is missing from df.")

    is_polars = False
    if isinstance(df, pl.DataFrame):
        df = df.to_pandas()
        is_polars = True
    if isinstance(splice_sites_df, pl.DataFrame):
        splice_sites_df = splice_sites_df.to_pandas()
    if isinstance(gene_feature_df, pl.DataFrame):
        gene_feature_df = gene_feature_df.to_pandas()

    # Merge df with gene_feature_df to get gene start, end, and strand
    df = df.merge(
        gene_feature_df[['gene_id', 'start', 'end', ]],   # 'strand'
        on='gene_id',
        how='left'
    )

    # Compute absolute positions based on strand
    def compute_absolute_position(row):
        relative_position = row[position_col]
        strand = row['strand']
        if strand == '+':
            return row['start'] + relative_position
        elif strand == '-':
            return row['end'] - relative_position
        else:
            raise ValueError(f"Invalid strand value: {strand}")

    # Add an absolute position column to df
    df['absolute_position'] = df.apply(compute_absolute_position, axis=1)

    # Initialize the transcript_id column
    df['transcript_id'] = None

    # Perform alignment with tolerance
    unmatched_genes = set()  # Track genes that fail the validation check
    aligned_rows = []  # To store the expanded rows

    for idx, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing rows"):
        gene_id = row['gene_id']
        abs_pos = row['absolute_position']
        splice_type = row['splice_type'] if 'splice_type' in row else None

        # Find matching splice sites within the tolerance range
        matching_sites = splice_sites_df[
            (splice_sites_df['gene_id'] == gene_id) &
            (abs(splice_sites_df['position'] - abs_pos) <= tolerance)
        ]

        # Add optional condition for splice_type if it is given
        if splice_type is not None:
            # Todo: Unify the column name
            # matching_sites = matching_sites[matching_sites['site_type'] == splice_type]
            matching_sites = matching_sites[matching_sites['splice_type'] == splice_type]
            
        if not matching_sites.empty:
            # Assign the first matching transcript_id
            # df.at[idx, 'transcript_id'] = matching_sites.iloc[0]['transcript_id']

            # Keep track of all matching transcripts
            # => Create a row for each matched transcript
            for transcript_id in matching_sites['transcript_id']:
                new_row = row.copy()
                new_row['transcript_id'] = transcript_id
                aligned_rows.append(new_row)

        else:
            unmatched_genes.add(gene_id)

            print(f"Could not find a matching splice site for gene {gene_id} at position {abs_pos}, where site_type={splice_type}")
            
            # Test
            if splice_type is None:
                display(splice_sites_df[splice_sites_df['gene_id'] == gene_id].sort_values('position').head(20))
            else: 
                display(splice_sites_df[(splice_sites_df['gene_id'] == gene_id) & (splice_sites_df['splice_type'] == splice_type)].sort_values('position').head(20))

    # Validation check: Ensure all genes in df have matches in splice_sites_df
    df_gene_ids = set(df['gene_id'])
    splice_site_gene_ids = set(splice_sites_df['gene_id'])
    missing_genes = df_gene_ids - splice_site_gene_ids

    if missing_genes:
        raise ValueError(
            f"Some genes in the input dataframe are missing from splice_sites_df: {missing_genes}"
        )

    msg = (f"[warning] Some splice site positions for genes in the input dataframe "
              f"could not be matched (even with tolerance={tolerance}): {unmatched_genes}"
           )

    if unmatched_genes:
        if throw_exception: 
            raise ValueError(msg)
        else: 
            print(msg)

            # remove unmatched genes
            print_emphasized(f"Removing unmatched genes from the input dataframe (n={len(unmatched_genes)})...")
            df = df[~df['gene_id'].isin(unmatched_genes)]

    # Create a DataFrame from aligned rows
    aligned_df = pd.DataFrame(aligned_rows)

    if is_polars:
        aligned_df = pl.DataFrame(aligned_df)

    if return_unmatched: 
        return aligned_df, unmatched_genes

    return aligned_df


def align_transcript_ids_single(
    df,
    transcript_feature_df,
    gene_feature_df,
    position_col='position',
    tolerance=0,
    return_unmatched=False,
    remove_extra_cols=True
):
    """
    Like align_transcript_ids_by_boundaries, but pick exactly ONE transcript
    for each position to avoid row blow-up. Uses a 'closest boundary' strategy.
    """
    def compute_absolute_position(row):
        relative_position = row[position_col]
        strand = row['gene_strand'] if 'gene_strand' in row else row['strand']
        if strand == '+':
            return row['gene_start'] + relative_position
        elif strand == '-':
            return row['gene_end'] - relative_position
        else:
            raise ValueError(f"Invalid strand value: {strand}")

    is_polars = False
    if isinstance(df, pl.DataFrame):
        df = df.to_pandas()
        is_polars = True

    # If transcript_id is already non-null, skip
    if 'transcript_id' in df.columns and df['transcript_id'].notna().all():
        print("[info] 'transcript_id' is non-null. Returning the original DataFrame.")
        if is_polars:
            df = pl.DataFrame(df)
        if return_unmatched:
            return df, []
        return df

    if isinstance(transcript_feature_df, pl.DataFrame):
        transcript_feature_df = transcript_feature_df.to_pandas()
    if isinstance(gene_feature_df, pl.DataFrame):
        gene_feature_df = gene_feature_df.to_pandas()

    gene_feature_df = gene_feature_df.rename(columns={
        'start': 'gene_start',
        'end': 'gene_end',
        'strand': 'gene_strand',
        'chrom': 'gene_chrom'
    })
    transcript_feature_df = transcript_feature_df.rename(columns={
        'start': 'transcript_start',
        'end': 'transcript_end'
    })

    # Merge gene boundaries
    df = df.merge(
        gene_feature_df[['gene_id','gene_start','gene_end','gene_strand']],
        on='gene_id',
        how='left'
    )

    df['absolute_position'] = df.apply(compute_absolute_position, axis=1)

    # We'll store the result in a list
    aligned_rows = []

    for idx, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing rows"):
        gene_id = row['gene_id']
        abs_pos = row['absolute_position']

        candidates = transcript_feature_df[
            (transcript_feature_df['gene_id'] == gene_id) &
            (transcript_feature_df['transcript_start'] - tolerance <= abs_pos) &
            (transcript_feature_df['transcript_end']   + tolerance >= abs_pos)
        ]

        if not candidates.empty:
            # pick the transcript whose boundary is closest to abs_pos
            candidates['dist_to_boundary'] = candidates.apply(
                lambda r2: min(abs(abs_pos - r2['transcript_start']),
                               abs(abs_pos - r2['transcript_end'])),
                axis=1
            )
            chosen = candidates.loc[ candidates['dist_to_boundary'].idxmin() ]
            new_row = row.copy()
            new_row['transcript_id'] = chosen['transcript_id']
            aligned_rows.append(new_row)
        else:
            # No match => still keep this row with transcript_id=None
            new_row = row.copy()
            new_row['transcript_id'] = None
            aligned_rows.append(new_row)

    aligned_df = pd.DataFrame(aligned_rows)

    df_gene_ids = set(df['gene_id'])
    transcript_gene_ids = set(transcript_feature_df['gene_id'])
    missing_genes = df_gene_ids - transcript_gene_ids

    if remove_extra_cols:
        for col_ in ['gene_start','gene_end','gene_strand','absolute_position','dist_to_boundary']:
            if col_ in aligned_df.columns:
                aligned_df.drop(columns=col_, inplace=True)

    if is_polars:
        aligned_df = pl.DataFrame(aligned_df)

    if return_unmatched:
        return aligned_df, missing_genes
    return aligned_df


def align_transcript_ids_by_boundaries(
        df, 
        transcript_feature_df, 
        gene_feature_df, 
        position_col='position', 
        tolerance=0, return_unmatched=False, remove_extra_cols=True):
    """
    Align transcript IDs to a dataframe with relative positions using transcript boundaries.
    See also align_transcript_ids_single(), align_transcript_ids_by_boundaries_enhanced()

    Parameters:
    - df (pd.DataFrame): Input DataFrame with a position-like column for alignment.
      Required columns: ['gene_id', position_col, 'strand'], where `position_col` represents
        a relative position within the gene.
    - transcript_feature_df (pd.DataFrame): DataFrame containing transcript-level features,
      including 'transcript_id', 'start', 'end', 'gene_id'.
    - gene_feature_df (pd.DataFrame): DataFrame containing gene-level features, including
      'gene_id', 'start', 'end', 'strand'.
    - position_col (str): Name of the column in `df` representing the relative position.
    - tolerance (int): Number of nucleotides to allow for transcript matching (default: 0).
    - return_unmatched (bool): Whether to return unmatched genes.
    - remove_extra_cols (bool): If True, removes intermediate columns added during processing.

    Returns:
    - pd.DataFrame: A new DataFrame with an added `transcript_id` column.
    """
    # Compute absolute positions
    def compute_absolute_position(row):
        relative_position = row[position_col]
        strand = row['gene_strand'] if 'gene_strand' in row else row['strand']
        if strand == '+':
            return row['gene_start'] + relative_position
        elif strand == '-':
            return row['gene_end'] - relative_position
        else:
            raise ValueError(f"Invalid strand value: {strand}")

    # At the moment, this function is designed to work with Pandas DataFrames
    is_polars = False
    if isinstance(df, pl.DataFrame):
        df = df.to_pandas()
        is_polars = True

    # Base case: Check if transcript_id column exists and has no null values
    if 'transcript_id' in df.columns and df['transcript_id'].notna().all():
        print("[info] 'transcript_id' already exists and is non-null. Returning the original DataFrame.")
   
        if is_polars:
            df = pl.DataFrame(df)

        if return_unmatched: 
            return df, []
        return df

    if isinstance(transcript_feature_df, pl.DataFrame):
        transcript_feature_df = transcript_feature_df.to_pandas()
    if isinstance(gene_feature_df, pl.DataFrame):
        gene_feature_df = gene_feature_df.to_pandas()

    # Rename columns to resolve ambiguity
    gene_feature_df = gene_feature_df.rename(columns={
        'start': 'gene_start',
        'end': 'gene_end',
        'strand': 'gene_strand',
        'chrom': 'gene_chrom'
    })
    transcript_feature_df = transcript_feature_df.rename(columns={
        'start': 'transcript_start',
        'end': 'transcript_end'
    })

    # Merge gene boundaries into input dataframe
    df = df.merge(
        gene_feature_df[['gene_id', 'gene_start', 'gene_end', 'gene_strand']],
        on='gene_id',
        how='left'
    )

    df['absolute_position'] = df.apply(compute_absolute_position, axis=1)

    # Initialize the transcript_id column
    df['transcript_id'] = None

    aligned_rows = []  # To store the expanded rows

    # Perform alignment based on transcript boundaries
    for idx, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing rows"):
        gene_id = row['gene_id']
        abs_pos = row['absolute_position']

        # Find transcripts within tolerance range
        matching_transcripts = transcript_feature_df[
            (transcript_feature_df['gene_id'] == gene_id) &
            (transcript_feature_df['transcript_start'] - tolerance <= abs_pos) &
            (transcript_feature_df['transcript_end'] + tolerance >= abs_pos)
        ]

        if not matching_transcripts.empty:
            for _, transcript_row in matching_transcripts.iterrows():
                new_row = row.copy()
                new_row['transcript_id'] = transcript_row['transcript_id']
                aligned_rows.append(new_row)

    # Convert aligned rows to a DataFrame
    aligned_df = pd.DataFrame(aligned_rows)

    # Validation - Ensure all genes have matches
    df_gene_ids = set(df['gene_id'])
    transcript_gene_ids = set(transcript_feature_df['gene_id'])
    missing_genes = df_gene_ids - transcript_gene_ids

    if missing_genes:
        raise ValueError(
            f"Some genes in the input dataframe are missing from transcript_feature_df: {missing_genes}"
        )

    # Remove extra columns if requested
    if remove_extra_cols:
        cols_to_remove = ['gene_start', 'gene_end', 'gene_strand', 'absolute_position']
        aligned_df = aligned_df.drop(columns=[col for col in cols_to_remove if col in aligned_df.columns])

    if is_polars:
        aligned_df = pl.DataFrame(aligned_df)

    if return_unmatched: 
        return aligned_df, missing_genes

    return aligned_df


def align_transcript_ids_by_boundaries_enhanced(
    df,
    transcript_feature_df,
    gene_feature_df,
    position_col='position',
    tolerance=0,
    policy='longest',             # or 'closest'
    max_transcripts=1,           # default to 1
    return_unmatched=False,
    remove_extra_cols=True,
    min_distance_boundary=0,      # New parameter: minimal allowed distance from transcript boundary
    drop_unmatched_if_too_close=False
):
    """
    Enhanced version of align_transcript_ids_by_boundaries() that:
      1) finds all transcripts for a given gene that contain a position (± `tolerance`),
      2) optionally enforces a minimal distance from the transcript boundaries
         (i.e., ensure the position is at least min_distance_boundary away from start/end),
      3) picks transcripts according to the specified `policy`:
         - 'longest' => pick the transcripts with the largest length,
         - 'closest' => pick the transcripts whose boundary is closest to the position,
      4) returns up to `max_transcripts` transcripts per row.

    Parameters
    ----------
    df : pd.DataFrame or pl.DataFrame
        Must contain: 'gene_id', position_col, 'strand' (optionally).
    transcript_feature_df : pd.DataFrame or pl.DataFrame
        Must contain: 'gene_id', 'transcript_id', 'start', 'end'.
          We'll rename these to transcript_{start,end}.
    gene_feature_df : pd.DataFrame or pl.DataFrame
        Must contain: 'gene_id', 'start', 'end', 'strand'.
          We'll rename these to gene_{start,end,strand}.
    position_col : str, default='position'
        Name of the column representing the relative position within the gene.
    tolerance : int, default=0
        ± range for matching transcripts if the position is near the boundary.
    policy : {'longest','closest'}, default='longest'
        - If 'longest', we pick the transcripts with the largest (transcript_end - transcript_start).
        - If 'closest', we pick transcripts whose boundary is closest to the absolute_position.
    max_transcripts : int, default=1
        Maximum number of transcripts to keep per row after sorting by `policy`.
    return_unmatched : bool, default=False
        If True, returns (aligned_df, missing_genes). Otherwise just returns aligned_df.
    remove_extra_cols : bool, default=True
        If True, drops columns like 'gene_start','gene_end','gene_strand','absolute_position','dist_to_boundary','transcript_len'.
    min_distance_boundary : int, default=0
        Minimal required distance from transcript boundaries. If > 0, we skip candidates if
        the absolute_position is too close to transcript_{start,end}.
        e.g. if min_distance_boundary=10, the position must be at least 10 nt away from start & end.
    drop_unmatched_if_too_close : bool, default=False
        If True and all transcripts are invalid due to boundary distance, we simply
        drop the row from the final output. Otherwise we produce a row with transcript_id=None.

    Returns
    -------
    pd.DataFrame or pl.DataFrame
        A new DataFrame with an added 'transcript_id' column. Potentially multiple rows per original row
        if `max_transcripts` > 1 or if multiple transcripts tie. 
        If return_unmatched=True, also returns the set of missing genes.
    """

    def compute_absolute_position(row):
        """
        Convert the relative position in row[position_col] to an absolute genomic position
        based on gene_{start,end} and strand.
        """
        relative_position = row[position_col]
        strand = row['gene_strand'] if 'gene_strand' in row else row.get('strand', '+')
        if strand == '+':
            return row['gene_start'] + relative_position
        elif strand == '-':
            return row['gene_end'] - relative_position
        else:
            raise ValueError(f"Invalid strand value: {strand}")

    # Convert polars => pandas if needed
    is_polars = isinstance(df, pl.DataFrame)
    if is_polars:
        df = df.to_pandas()
    if isinstance(transcript_feature_df, pl.DataFrame):
        transcript_feature_df = transcript_feature_df.to_pandas()
    if isinstance(gene_feature_df, pl.DataFrame):
        gene_feature_df = gene_feature_df.to_pandas()

    # If transcript_id is non-null for all rows, no need to realign
    if 'transcript_id' in df.columns and df['transcript_id'].notna().all():
        print("[info] 'transcript_id' is non-null for all rows. Returning original DataFrame.")
        if is_polars and not return_unmatched:
            return pl.DataFrame(df)
        elif is_polars and return_unmatched:
            return pl.DataFrame(df), []
        elif not is_polars and not return_unmatched:
            return df
        else:
            return df, []

    # rename columns in gene_feature_df to avoid collisions
    gene_feature_df = gene_feature_df.rename(columns={
        'start': 'gene_start',
        'end': 'gene_end',
        'strand': 'gene_strand',
        'chrom': 'gene_chrom'
    }).copy()
    # rename columns in transcript_feature_df similarly
    transcript_feature_df = transcript_feature_df.rename(columns={
        'start': 'transcript_start',
        'end': 'transcript_end'
    }).copy()

    # Check that the required columns exist
    for col_required in ['gene_start','gene_end','gene_strand']:
        if col_required not in gene_feature_df.columns:
            raise ValueError(f"Missing column '{col_required}' in gene_feature_df.")

    # Merge gene boundaries into the input df => add gene_{start,end,strand}
    df = df.merge(
        gene_feature_df[['gene_id','gene_start','gene_end','gene_strand']],
        on='gene_id',
        how='left'
    )

    # Compute absolute_position
    df['absolute_position'] = df.apply(compute_absolute_position, axis=1)

    # Validate the policy
    if policy not in ('longest','closest'):
        raise ValueError(f"Unsupported policy='{policy}'. Must be 'longest' or 'closest'.")

    aligned_rows = []

    # Process each row in df, find matching transcripts, pick up to `max_transcripts`.
    for idx, row_i in tqdm(df.iterrows(), total=df.shape[0], desc="Processing rows"):
        gene_id = row_i['gene_id']
        abs_pos = row_i['absolute_position']

        # find transcripts that overlap [abs_pos ± tolerance]
        candidates = transcript_feature_df[
            (transcript_feature_df['gene_id'] == gene_id)
            &
            (transcript_feature_df['transcript_start'] - tolerance <= abs_pos)
            &
            (transcript_feature_df['transcript_end']   + tolerance >= abs_pos)
        ]
        if candidates.empty:
            # no transcripts => unmatched row
            if not drop_unmatched_if_too_close:
                new_row = row_i.copy()
                new_row['transcript_id'] = None
                aligned_rows.append(new_row)
            continue

        # If min_distance_boundary>0, filter out transcripts where the position is too close
        # to transcript boundaries
        if min_distance_boundary > 0:
            # We define:
            #   dist_start = abs_pos - transcript_start
            #   dist_end   = transcript_end - abs_pos
            # Each must be >= min_distance_boundary
            candidates = candidates[
                ((abs_pos - candidates['transcript_start']) >= min_distance_boundary)
                &
                ((candidates['transcript_end'] - abs_pos) >= min_distance_boundary)
            ]
            if candidates.empty:
                # again no match => unmatched or drop
                if not drop_unmatched_if_too_close:
                    new_row = row_i.copy()
                    new_row['transcript_id'] = None
                    aligned_rows.append(new_row)
                    # NOTE: If no transcripts remain after this filter, 
                    #       we either add a row with transcript_id=None or skip entirely—depending 
                    #       on the value of `drop_unmatched_if_too_close`.
                continue

        # Decide how to rank them
        candidates = candidates.copy()  # to avoid SettingWithCopy issues
        if policy=='longest':
            candidates['transcript_len'] = (candidates['transcript_end'] - candidates['transcript_start'])
            # sort descending by transcript_len
            candidates.sort_values(by='transcript_len', ascending=False, inplace=True)
        else:
            # policy == 'closest'
            # measure distance to boundary
            candidates['dist_to_boundary'] = candidates.apply(
                lambda r2: min(
                    abs(abs_pos - r2['transcript_start']),
                    abs(r2['transcript_end'] - abs_pos)
                ),
                axis=1
            )
            candidates.sort_values(by='dist_to_boundary', ascending=True, inplace=True)

        # after sorting, pick up to max_transcripts
        top_candidates = candidates.head(max_transcripts)
        for _, c_row in top_candidates.iterrows():
            new_row = row_i.copy()
            new_row['transcript_id'] = c_row['transcript_id']
            aligned_rows.append(new_row)

    # Build final DataFrame
    aligned_df = pd.DataFrame(aligned_rows)

    # Check for missing genes if relevant
    df_gene_ids = set(df['gene_id'])
    transcript_gene_ids = set(transcript_feature_df['gene_id'])
    missing_genes = df_gene_ids - transcript_gene_ids

    # Remove extra columns if requested
    if remove_extra_cols:
        drop_cols = [
            'gene_start','gene_end','gene_strand','absolute_position',
            'dist_to_boundary','transcript_len'
        ]
        for col_ in drop_cols:
            if col_ in aligned_df.columns:
                aligned_df.drop(columns=col_, inplace=True, errors='ignore')

    # Convert back to polars if original was polars
    if is_polars:
        aligned_df = pl.DataFrame(aligned_df)

    if return_unmatched:
        return aligned_df, missing_genes
    return aligned_df


def align_transcript_ids_by_boundaries_enhanced_v0(
    df,
    transcript_feature_df,
    gene_feature_df,
    position_col='position',
    tolerance=2,
    policy='longest',             # or 'closest'
    max_transcripts=1,           # default to 1
    return_unmatched=False,
    remove_extra_cols=True
):
    """
    Enhanced version of align_transcript_ids_by_boundaries() that either:
    - picks the 'longest' transcripts containing the position, or
    - picks the 'closest' transcripts by distance to boundary,
    returning up to `max_transcripts` transcripts per row.

    Parameters
    ----------
    df : pd.DataFrame or pl.DataFrame
        Contains columns: 'gene_id', position_col, 'strand' (optionally).
    transcript_feature_df : pd.DataFrame or pl.DataFrame
        Must have 'gene_id', 'transcript_id', 'start', 'end' (which we rename to transcript_{start,end}).
    gene_feature_df : pd.DataFrame or pl.DataFrame
        Must have 'gene_id', 'start', 'end', 'strand' (we rename to gene_{start,end,strand}).
    policy : str, default='longest'
        - 'longest' => pick transcripts with the largest length (transcript_end - transcript_start).
        - 'closest' => pick transcripts whose boundaries are closest to the absolute position.
    max_transcripts : int, default=1
        Cap on how many transcripts we keep per row. Use 1 if you want exactly one transcript.
    tolerance : int, default=0
        ± range for matching transcripts if the position is near the boundary.
    return_unmatched : bool, default=False
        If True, return (aligned_df, missing_genes).
    remove_extra_cols : bool, default=True
        If True, remove intermediate columns like 'gene_start','gene_end','gene_strand','absolute_position'.

    Returns
    -------
    pd.DataFrame or pl.DataFrame
        A new DataFrame with added 'transcript_id' column (may be repeated or expanded up to max_transcripts times).
        If return_unmatched=True, returns (df_aligned, missing_genes).
    """
    def compute_absolute_position(row):
        # 'strand' or 'gene_strand'
        # row[ position_col ] is the relative position
        # if strand=='+', absolute_position = gene_start + relative_position
        # if strand=='-', absolute_position = gene_end - relative_position
        # else error
        relative_position = row[position_col]
        strand = row['gene_strand'] if 'gene_strand' in row else row.get('strand', '+')
        if strand == '+':
            return row['gene_start'] + relative_position
        elif strand == '-':
            return row['gene_end'] - relative_position
        else:
            raise ValueError(f"Invalid strand value: {strand}")

    # Convert polars => pandas if needed
    is_polars = isinstance(df, pl.DataFrame)
    if is_polars:
        df = df.to_pandas()
    if isinstance(transcript_feature_df, pl.DataFrame):
        transcript_feature_df = transcript_feature_df.to_pandas()
    if isinstance(gene_feature_df, pl.DataFrame):
        gene_feature_df = gene_feature_df.to_pandas()

    # If transcript_id is non-null for all rows, we can skip
    if 'transcript_id' in df.columns and df['transcript_id'].notna().all():
        print("[info] 'transcript_id' is non-null for all rows. Returning original DataFrame.")
        if is_polars and not return_unmatched:
            return pl.DataFrame(df)
        elif is_polars and return_unmatched:
            return pl.DataFrame(df), []
        elif not is_polars and not return_unmatched:
            return df
        else:
            return df, []

    # rename columns
    gene_feature_df = gene_feature_df.rename(columns={
        'start': 'gene_start',
        'end': 'gene_end',
        'strand': 'gene_strand',
        'chrom': 'gene_chrom'
    }).copy()
    transcript_feature_df = transcript_feature_df.rename(columns={
        'start': 'transcript_start',
        'end': 'transcript_end'
    }).copy()

    # Merge gene boundaries
    merge_cols = ['gene_id']
    for col_required in ['gene_start','gene_end','gene_strand']:
        if col_required not in gene_feature_df.columns:
            raise ValueError(f"Missing column '{col_required}' in gene_feature_df.")
    df = df.merge(
        gene_feature_df[['gene_id','gene_start','gene_end','gene_strand']],
        on='gene_id',
        how='left'
    )
    print("[test] columns(df):", list(df.columns))
    print(f"[test] dim(df): {df.shape}")

    # compute absolute_position
    df['absolute_position'] = df.apply(compute_absolute_position, axis=1)

    aligned_rows = []

    # optional extra fields we might need
    if policy not in ('longest','closest'):
        raise ValueError(f"Unsupported policy='{policy}', must be 'longest' or 'closest'.")

    # Initialize a counter for debugging messages
    debug_counter = 0
    debug_limit = 3  # Set the limit for how many times to show debug messages

    # for progress bar
    for idx, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing rows"):
        gene_id = row['gene_id']
        abs_pos = row['absolute_position']

        # find all transcripts in that gene that overlap [abs_pos ± tolerance]
        candidates = transcript_feature_df[
            (transcript_feature_df['gene_id'] == gene_id)
            &
            (transcript_feature_df['transcript_start'] - tolerance <= abs_pos)
            &
            (transcript_feature_df['transcript_end']   + tolerance >= abs_pos)
        ]
        if candidates.empty:
            # keep the row with transcript_id=None
            print(f"[test] No candidates found for gene_id {gene_id} at position {abs_pos}")
            new_row = row.copy()
            new_row['transcript_id'] = None
            aligned_rows.append(new_row)
        else:
            if debug_counter < debug_limit:
                print(f"[debug] Found {len(candidates)} candidates for gene_id {gene_id} at position {abs_pos}")

            # Make a copy of the candidates DataFrame to avoid SettingWithCopyWarning
            candidates = candidates.copy()

            # decide how to rank them
            if policy=='longest':
                # Compute length
                candidates['transcript_len'] = candidates['transcript_end'] - candidates['transcript_start']
                
                # Compute length using .loc to avoid SettingWithCopyWarning? 
                # candidates.loc[:, 'transcript_len'] = candidates['transcript_end'] - candidates['transcript_start']
                
                # sort descending by transcript_len
                candidates = candidates.sort_values(by='transcript_len', ascending=False)
                
                if debug_counter < debug_limit:
                    print(f"[debug] Top 3 longest candidates for gene_id {gene_id}:\n{candidates.head(3)}")

                    # Increment the debug counter
                    debug_counter += 1

            elif policy=='closest':
                # compute distance from boundary
                # distance = min(|abs_pos - transcript_start|, |abs_pos - transcript_end|)
                candidates['dist_to_boundary'] = candidates.apply(
                    lambda r2: min(
                        abs(abs_pos - r2['transcript_start']),
                        abs(abs_pos - r2['transcript_end'])
                    ),
                    axis=1
                )
                # sort ascending by dist_to_boundary
                candidates = candidates.sort_values(by='dist_to_boundary', ascending=True)

                if debug_counter < debug_limit:
                    print(f"[debug] Top 3 closest candidates for gene_id {gene_id}:\n{candidates.head(3)}")

                    # Increment the debug counter
                    debug_counter += 1
            
            # after sorting, we pick up to max_transcripts rows
            top_candidates = candidates.head(max_transcripts)

            for _, c_row in top_candidates.iterrows():
                new_row = row.copy()
                new_row['transcript_id'] = c_row['transcript_id']
                aligned_rows.append(new_row)

    aligned_df = pd.DataFrame(aligned_rows)

    # check for missing gene matches
    df_gene_ids = set(df['gene_id'])
    transcript_gene_ids = set(transcript_feature_df['gene_id'])
    missing_genes = df_gene_ids - transcript_gene_ids

    if remove_extra_cols:
        # remove intermediate columns
        for col_ in ['gene_start','gene_end','gene_strand','absolute_position','dist_to_boundary','transcript_len']:
            if col_ in aligned_df.columns:
                aligned_df.drop(columns=col_, inplace=True, errors='ignore')

    if is_polars:
        aligned_df = pl.DataFrame(aligned_df)

    if return_unmatched:
        return aligned_df, missing_genes
    return aligned_df



############################################


def demo_analyze_splice_sites():
    from meta_spliceai.splice_engine.extract_genomic_features import (
        FeatureAnalyzer, 
        SpliceAnalyzer
    )

    # splice_sites_df = retrieve_splice_sites() 
    analyzer = SpliceAnalyzer()
    splice_sites_df = analyzer.retrieve_splice_sites()  # Output is a Polars DataFrame
    print(f"[info] Columns in splice_sites_df: {splice_sites_df.columns}")
    # NOTE: Columns: ['chrom', 'start', 'end', 'strand', 'site_type', 'gene_id', 'transcript_id']
    
    display(splice_sites_df.head(5))
    splice_sites_df = splice_sites_df.to_pandas()

    # Verify the number of splice sites
    target_genes = ['ENSG00000253267', 'ENSG00000287740', 'ENSG00000240486', 'ENSG00000253280']

    splice_type = 'donor'
    for gene_id in target_genes: 
        # Number of donor sites
        n_splice_sites = len(splice_sites_df[(splice_sites_df['gene_id'] == gene_id) & (splice_sites_df['site_type'] == splice_type)]) 
        print(f"[test] Gene {gene_id} has {n_splice_sites} splice sites.")

        # Group true splice sites by gene
        
        # Filter annotations for splice sites
        filtered_annotations = splice_sites_df[splice_sites_df['site_type'] == splice_type]

        # Group true splice sites by gene and aggregate into a list of dictionaries
        grouped_annotations = (
            filtered_annotations.groupby('gene_id')
            .apply(lambda group: group[['start', 'end']].to_dict('records'))
            .to_dict()
        )

        # Retrieve acceptor sites for a specific gene_id
        splice_sites = grouped_annotations.get(gene_id, [])

        # Remove duplicates by 'start' and 'end' while preserving the structure
        splice_sites = list({(site['start'], site['end']): site for site in splice_sites}.values())

        print("[info] Number of unique splice sites:", len(splice_sites))

        # NOTE: The number of splice sites can appear to be inconsistent because a gene can have multiple transcripts
        #       and each transcript can have multiple splice sites, which may have overlapping coordinates.
        #       E.g. ENSG00000253267
        #            (1620371, 1620375) appears twice.
        # 
        #            One from ENST00000518063.
        #            Another from ENST00000518009.


def demo_analyze_error_profile():
    error_data = None 

    analyze_error_profile(
        error_df=error_data,
        save_plot=True,
        plot_file='fp_fn_error_distribution.pdf',
        verbose=True
    )

def demo_misc(): 

    subset_error_analysis_file()


def incorporate_gene_features_v0(df_trainset, gtf_file, format='tsv', overwrite=True, **kargs):
    # from .extract_genomic_features import (
    #     FeatureAnalyzer, SpliceAnalyzer, 
    #     run_genomic_gtf_feature_extraction 
    # )
    col_tid = kargs.get('col_tid', 'transcript_id')

    fa = FeatureAnalyzer(gtf_file=gtf_file, overwrite=overwrite)

    output_path = os.path.join(ErrorAnalyzer.analysis_dir, f'genomic_gtf_feature_set.{format}')
    gene_fs_df = run_genomic_gtf_feature_extraction(gtf_file, output_file=output_path)
    print("[info] Columns in gnomic gtf-derived feature set:", list(gene_fs_df.columns))
    print_emphasized(f"[info] Saved gnomic gtf-derived feature set to {output_path}")

    shape0 = df_trainset.shape
    on_cols = ['gene_id', col_tid]
    df_trainset = join_and_remove_duplicates(df_trainset, gene_fs_df, on=on_cols, how='left', verbose=1)
    print(f"[info] shape(df_trainset) after including gene features: {shape0} -> {df_trainset.shape}")
    assert shape0[0] == df_trainset.shape[0], "Assertion failed: Number of rows changed after joining gene features."

    return df_trainset


def incorporate_exon_intron_lengths(df_trainset, gtf_file, format='tsv', overwrite=True, **kargs):
    # from .extract_genomic_features import (
    #         FeatureAnalyzer, SpliceAnalyzer, 
    #         run_genomic_gtf_feature_extraction 
    #     )
    col_tid = kargs.get('col_tid', 'transcript_id')

    fa = FeatureAnalyzer(gtf_file=gtf_file, overwrite=overwrite)
    output_path = os.path.join(fa.analysis_dir, 'total_intron_exon_lengths.tsv')
    if os.path.exists(output_path):
        length_df = pd.read_csv(output_path, sep='\t')
    else: 
        length_df = compute_total_lengths(gtf_file)
        path_length_features = fa.save_dataframe(length_df, 'total_intron_exon_lengths.tsv')
        assert output_path == path_length_features, f"File path mismatch: {output_path} != {path_length_features}"
    print(f"[i/o] Path(length_df): {output_path}")
    print(f"[test] type(length_df): {type(length_df)})")
    print(f"[test] columns(length_df): {list(length_df.columns)}")

    shape0 = df_trainset.shape
    on_cols = [col_tid, ]
    df_trainset = join_and_remove_duplicates(df_trainset, length_df, on=on_cols, how='left', verbose=1)
    print(f"[info] shape(df_trainset) after including exon-intron lengths: {shape0} -> {df_trainset.shape}")

    assert shape0[0] == df_trainset.shape[0], "Assertion failed: Number of rows changed after joining exon-intron lengths."
    return df_trainset


def incorporate_performance_features_v0(df_trainset, gtf_file=None, overwrite=True, **kargs):
    fa = FeatureAnalyzer(gtf_file=gtf_file, overwrite=overwrite)
    splice_stats_features = fa.retrieve_gene_level_performance_features()
    print("[test] columns(splice_stats_features):", list(splice_stats_features.columns))
    # NOTE: ['gene_id', 'n_splice_sites', 'splice_type']

    shape0 = df_trainset.shape
    on_cols = ['gene_id', 'splice_type']
    df_trainset = join_and_remove_duplicates(df_trainset, splice_stats_features, on=on_cols, how='left', verbose=1)
    print(f"[info] shape(df_trainset) after including performance features: {shape0} -> {df_trainset.shape}")
    print(f"[info] Columns in updated df_trainset (n={len(df_trainset.columns)}):", list(df_trainset.columns))

    assert shape0[0] == df_trainset.shape[0], "Assertion failed: Number of rows changed after joining performance features."
    return df_trainset


def incorporate_overlapping_genes(df_trainset, overwrite=True, **kargs):
    sa = SpliceAnalyzer()  # Todo: overwrite
    overlapping_genes_df = sa.retrieve_overlapping_gene_metadata(output_format='dataframe', to_pandas=True)
    print("[test] columns(overlapping_genes_df):", list(overlapping_genes_df.columns))
    print_with_indent(f"Is polars? {type(overlapping_genes_df)}", indent_level=1)

    columns = ['gene_id_1', 'num_overlaps']
    overlapping_genes_df = overlapping_genes_df[columns].rename(columns={'gene_id_1': 'gene_id'})
    print("[test] columns(overlapping_genes_df):", list(overlapping_genes_df.columns))
    print("[test] shape(overlapping_genes_df): ", overlapping_genes_df.shape)

    conflicting_counts = check_conflicting_overlaps(overlapping_genes_df)
    if conflicting_counts is not None:
        display(conflicting_counts)
        raise ValueError("Conflicting overlaps detected.")

    # De-duplicate overlapping_genes_df on gene_id
    # overlapping_genes_df = overlapping_genes_df.unique(subset=['gene_id'])  # Polars 
    overlapping_genes_df = overlapping_genes_df.drop_duplicates(subset=['gene_id'])

    shape0 = df_trainset.shape
    on_cols = ['gene_id']
    df_trainset = df_trainset.merge(overlapping_genes_df, on=on_cols, how='left').fillna(0)
    print(f"[info] shape(df_trainset) after including overlapping genes: {shape0} -> {df_trainset.shape}")

    assert shape0[0] == df_trainset.shape[0], "Assertion failed: Number of rows changed after joining overlapping genes."
    return df_trainset


def compute_distances_with_strand(df_trainset, **kwargs):
    col_tid = kwargs.get('col_tid', 'transcript_id')

    shape0 = df_trainset.shape
    df_trainset = compute_distances_with_strand_adjustment(df_trainset, match_col='position', col_tid=col_tid)
    print(f"[info] shape(df_trainset) after computing distances with strand adjustment: {shape0} -> {df_trainset.shape}")

    assert shape0[0] == df_trainset.shape[0], "Assertion failed: Number of rows changed after computing distances with strand adjustment."
    return df_trainset


# Main function to incorporate all features
def incorporate_all_features(df_trainset, gtf_file, format='tsv', overwrite=True, **kargs):
    df_trainset = incorporate_gene_features_v0(df_trainset, gtf_file, format=format, overwrite=overwrite)
    df_trainset = incorporate_exon_intron_lengths(df_trainset, gtf_file, format=format, overwrite=overwrite)
    df_trainset = incorporate_performance_features_v0(df_trainset, overwrite=overwrite)
    df_trainset = incorporate_overlapping_genes(df_trainset)
    df_trainset = compute_distances_with_strand(df_trainset)
    return df_trainset


def downsample_dataframe(df, sample_fraction=1.0, max_sample_size=None, verbose=1):
    """
    Downsample a Polars DataFrame based on a sample fraction and a maximum sample size.

    Parameters:
    - df (pl.DataFrame): The DataFrame to downsample.
    - sample_fraction (float): The fraction of data to keep (0 < sample_fraction <= 1).
    - max_sample_size (int, optional): The maximum number of rows to sample.

    Returns:
    - pl.DataFrame: The downsampled DataFrame.
    """
    original_size = df.shape[0]
    if sample_fraction < 1.0 or (max_sample_size is not None and max_sample_size < original_size):
        num_rows_to_sample = min(int(original_size * sample_fraction), max_sample_size or original_size)
        df = df.sample(n=num_rows_to_sample, with_replacement=False)
        if verbose: 
            print(f"[info] Downsampled DataFrame from {original_size} to {df.shape[0]} rows.")
    else:
        if verbose: 
            print(f"[info] No downsampling applied. DataFrame size: {original_size} rows.")

    return df


def make_error_sequence_model_dataset(analysis_sequence_df, pred_type):
    """
    Generate a FP-specific or FN-specific error sequence model dataset.

    Parameters:
    - analysis_sequence_df (pl.DataFrame or pd.DataFrame): The input DataFrame.
    - pred_type (str): The type of error analysis dataset to generate ('FP' or 'FN').

    Returns:
    - pl.DataFrame or pd.DataFrame: The filtered DataFrame.
    """
    is_polars = isinstance(analysis_sequence_df, pl.DataFrame)
    if is_polars:
        analysis_sequence_df = analysis_sequence_df.to_pandas()

    if pred_type == 'FP':
        filtered_df = analysis_sequence_df[analysis_sequence_df['pred_type'].isin(['TP', 'FP'])]
    elif pred_type == 'FN':
        filtered_df = analysis_sequence_df[analysis_sequence_df['pred_type'].isin(['TP', 'FN'])]
    else:
        raise ValueError(f"Invalid pred_type '{pred_type}'. Choose from 'FP' or 'FN'.")

    if is_polars:
        filtered_df = pl.DataFrame(filtered_df)

    return filtered_df


def make_kmer_featurized_dataset_incrementally(gtf_file=None, **kargs): 
    from meta_spliceai.splice_engine.extract_genomic_features import (
        extract_splice_sites_workflow, 
        transcript_sequence_retrieval_workflow,
        gene_sequence_retrieval_workflow, 
    )
    from meta_spliceai.splice_engine.utils_bio import (
        load_sequences_by_chromosome, 
        load_chromosome_sequence_streaming
    )
    from meta_spliceai.splice_engine.workflow_utils import (
        adjust_chunk_size
    )
    # from meta_spliceai.splice_engine.model_evaluator import ModelEvaluationFileHandler

    src_dir = '/path/to/meta-spliceai/data/ensembl/'  # Big files on Azure blob storage
    local_dir = '/path/to/meta-spliceai/data/ensembl/'
    eval_dir = '/path/to/meta-spliceai/data/ensembl/spliceai_eval'
    gtf_file = os.path.join(src_dir, "Homo_sapiens.GRCh38.112.gtf")  # Replace with your GTF file path
    genome_fasta = os.path.join(src_dir, "Homo_sapiens.GRCh38.dna.primary_assembly.fa") 

    col_tid = kargs.get('col_tid', 'transcript_id')
    overwrite = kargs.get("overwrite", True)
    mode = kargs.get("mode", 'gene')
    seq_type = kargs.get("seq_type", 'full')
    format = kargs.get("format", 'parquet') 
    test_mode = kargs.get("test_mode", False)
    
    run_sequence_extraction = kargs.get("run_sequence_extraction", False)
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

    ############################################################
    n_chr_processed = 0
    chromosomes = kargs.get("chromosomes", None)
    if chromosomes is None:
        chromosomes = [str(i) for i in range(1, 23)] + ['X', 'Y', ]  # 'MT'

    fa = FeatureAnalyzer(gtf_file=gtf_file, overwrite=overwrite, col_tid=col_tid)
    format = fa.format
    seperator = '\t' if format == 'tsv' else ','
    mefd = ModelEvaluationFileHandler(ErrorAnalyzer.eval_dir, separator=seperator)

    run_sequence_featurization = True
    
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

            print(f"[info] Processing genes {chunk_start + 1} to {chunk_end} out of {num_genes} genes at chromosome={chr} ...")

            # Use run_spliceai_workflow.error_analysis_workflow() to generate the error analysis data
            analysis_sequence_df = \
                mefd.load_analysis_sequences(chr=chr, chunk_start=chunk_start, chunk_end=chunk_end, aggregated=False)  # Output is a Polars DataFrame
            print("[info] Columns in analysis_sequence_df:", analysis_sequence_df.columns)
            # ['gene_id', 'position', 'pred_type', 'score', 'splice_type', 'chrom', 'window_start', 'window_end', 'sequence']

            # Determine the final unique number of genes in analysis_sequence_df
            final_unique_gene_ids = analysis_sequence_df.select(pl.col('gene_id')).unique()
            final_gene_ids_set = set(final_unique_gene_ids.to_series().to_list())
            
            # Check for missing custom genes
            missing_genes = [gene for gene in custom_genes if gene not in final_gene_ids_set]
            if missing_genes:
                print(f"[warning] The following custom genes were not found in the final unique gene IDs: {missing_genes}")
            else:
                print("[info] All custom genes were found in the final unique gene IDs.")
            final_num_unique_genes = final_unique_gene_ids.height
            print_emphasized(f"[info] Final number of unique gene IDs: {final_num_unique_genes}")

            # Add additional columns to the analysis_sequence_df (e.g. strand)
            if 'strand' not in analysis_sequence_df.columns:
                analysis_sequence_df = analysis_sequence_df.join(
                    gene_feature_df.select(['gene_id', 'strand']),
                    on='gene_id',
                    how='inner'
                )
            print("[info] Columns in (chunked) analysis_sequence_df:", analysis_sequence_df.columns)

            ############################################################
            subject = "seq_featurized"
            # analysis_subject = f"{error_label}_vs_{correct_label}_analysis_sequences"

            # col_tid = 'transcript_id'
            kmer_sizes = kargs.get("kmer_sizes", [6, ])   
            if kmer_sizes is None: 
                kmer_sizes = [6, ]
            elif isinstance(kmer_sizes, int):
                kmer_sizes = [kmer_sizes]
            
            df_trainset = pd.DataFrame()

            if run_sequence_featurization:
            
                print_emphasized(f"1) Aligning splice sites to transcripts  ...")

                tp_sequence_df = tn_sequence_df = None
                if correct_label.lower() == 'tp':
                    print("[info] Aligning transcript IDs for TP analysis ...")
                    tp_sequence_df, unmatched_genes = \
                        align_transcript_ids_for_tp_analysis(
                            analysis_sequence_df, gtf_file=ErrorAnalyzer.gtf_file, throw_exception=False)
                    print(f"[info] Shape of tp_sequence_df: {tp_sequence_df.shape}")
                    assert 'sequence' in tp_sequence_df.columns, "Column 'sequence' not found in tp_sequence_df."
                    print("[info] type(tp_sequence_df):", type(tp_sequence_df))
                    print(f"[info] Columns in tp_sequence_df: {list(tp_sequence_df.columns)[:100]}")

                if correct_label.lower() == 'tn': 
                    print("[info] Aligning transcript IDs for TN analysis ...")
                    tn_sequence_df, unmatched_genes = \
                        align_transcript_ids_for_tn_analysis(
                            analysis_sequence_df, gtf_file=ErrorAnalyzer.gtf_file)
                    print(f"[info] Shape of tn_sequence_df: {tn_sequence_df.shape}")
                    assert 'sequence' in tn_sequence_df.columns, "Column 'sequence' not found in tn_sequence_df."
                    print("[info] type(tn_sequence_df):", type(tn_sequence_df))
                    print(f"[info] Columns in tn_sequence_df: {list(tn_sequence_df.columns)[:100]}")
                
                # Remove unmatched genes
                if unmatched_genes: 
                    print_emphasized(f"Removing unmatched genes from the input dataframe (n={len(unmatched_genes)})...")
                    analysis_sequence_df = analysis_sequence_df.filter(~pl.col('gene_id').is_in(unmatched_genes))
                    # analysis_sequence_df = analysis_sequence_df[~analysis_sequence_df['gene_id'].isin(unmatched_genes)]

                fp_sequence_df = fn_sequence_df = None
                if error_label.lower() == 'fp':
                    print_emphasized(f"Aligning transcript IDs for FP analysis ...")
                    fp_sequence_df, unmatched_genes = \
                        align_transcript_ids_for_fp_analysis(
                            analysis_sequence_df, gtf_file=ErrorAnalyzer.gtf_file)
                    print(f"[info] Shape of fp_sequence_df: {fp_sequence_df.shape}")
                    assert 'sequence' in fp_sequence_df.columns, "Column 'sequence' not found in fp_sequence_df."
                    print("[info] type(fp_sequence_df):", type(fp_sequence_df))
                    print(f"[info] Columns in fp_sequence_df: {list(fp_sequence_df.columns)[:100]}")

                elif error_label.lower() == 'fn':
                    print_emphasized(f"Aligning transcript IDs for FN analysis ...")
                    fn_sequence_df, unmatched_genes = \
                        align_transcript_ids_for_fn_analysis(
                            analysis_sequence_df, gtf_file=ErrorAnalyzer.gtf_file)
                    print(f"[info] Shape of fn_sequence_df: {fn_sequence_df.shape}")
                    assert 'sequence' in fn_sequence_df.columns, "Column 'sequence' not found in fn_sequence_df."
                    print("[info] type(fn_sequence_df):", type(fn_sequence_df))
                    print(f"[info] Columns in fn_sequence_df: {list(fn_sequence_df.columns)[:100]}")

                ############################################################
                print_emphasized(f"2) Featurizing analysis sequences with correct_label={correct_label} ...")

                tp_featurized_df = tn_featurized_df = None
                tp_features = tn_features = None

                if correct_label.lower() == 'tp':
                    print_with_indent(f"[action] Featurizing analysis sequences of type TP ...", indent_level=1)
                    tp_featurized_df, tp_features = featurize_gene_sequences(tp_sequence_df, kmer_sizes=kmer_sizes, return_feature_set=True, verbose=1)
                if correct_label.lower() == 'tn':
                    
                    # Define the fraction of data to keep (e.g., 0.1 for 10%)
                    tn_sample_fraction = kargs.get('tn_sample_fraction', 0.5)
                    tn_max_sample_size = kargs.get('tn_max_sample_size', 50000)  # Adjust this value as needed
                    tn_sequence_df = downsample_dataframe(
                        tn_sequence_df, 
                        sample_fraction=tn_sample_fraction, 
                        max_sample_size=tn_max_sample_size, verbose=1)

                    # Check the shape of the downsampled DataFrame
                    if tn_sample_fraction < 1.0:
                        print(f"[info] Shape of (downsampled) tn_sequence_df: {tn_sequence_df.shape}")
                    
                    print_with_indent(f"[action] Featurizing analysis sequences of type TN ...", indent_level=1)
                    tn_featurized_df, tn_features = featurize_gene_sequences(tn_sequence_df, kmer_sizes=kmer_sizes, return_feature_set=True, verbose=1) 

                fp_featurized_df = fn_featurized_df = None
                if error_label.lower() == 'fp':
                    print_emphasized(f"[action] Featurizing analysis sequences of type FP ...")
                    fp_featurized_df, fp_features = featurize_gene_sequences(fp_sequence_df, kmer_sizes=kmer_sizes, return_feature_set=True, verbose=1)

                    if tp_features is not None:
                        missing_in_tp = set(fp_features) - set(tp_features)
                        missing_in_fp = set(tp_features) - set(fp_features)

                        if missing_in_tp or missing_in_fp:
                            print("[diagnostics] Feature set mismatch detected between TPs and FPs!")
                            print(f"Features missing in TP: {missing_in_tp}")
                            print(f"Features missing in FP: {missing_in_fp}")
                        else: 
                            print_emphasized("[info] Feature sets are consistent between TPs and FPs.")
                
                elif error_label.lower() == 'fn':
                    print_emphasized(f"[action] Featurizing analysis sequences of type FN ...")
                    fn_featurized_df, fn_features = featurize_gene_sequences(fn_sequence_df, kmer_sizes=kmer_sizes, return_feature_set=True, verbose=1)

                    # Define the fraction of data to keep (e.g., 0.1 for 10%)
                    fn_sample_fraction = kargs.get('fn_sample_fraction', 1.0)
                    fn_max_sample_size = kargs.get('fn_max_sample_size', 50000)  # Adjust this value as needed
                    fn_sequence_df = downsample_dataframe(
                        fn_sequence_df, 
                        sample_fraction=fn_sample_fraction, 
                        max_sample_size=fn_max_sample_size, verbose=1)

                    if tp_features is not None:
                        missing_in_tp = set(fn_features) - set(tp_features)
                        missing_in_fn = set(tp_features) - set(fn_features)

                        if missing_in_tp or missing_in_fn:
                            print("[diagnostics] Feature set mismatch detected between TPs and FNs!")
                            print(f"Features missing in TP: {missing_in_fp}")
                            print(f"Features missing in FN: {missing_in_fn}")
                        else: 
                            print_emphasized("[info] Feature sets are consistent between TPs and FNs.")

                # Keep selective meta features
                # fp_featurized_df = fp_featurized_df.rename({'gene_strand': 'strand'})  # Rename gene_strand to strand
                # fp_featurized_df = fp_featurized_df.drop('gene_strand')

                meta_columns = ['gene_id', col_tid, 'position', 'score', 'splice_type', 'chrom', 'strand']
                # NOTE: Columns in a typical analysis sequence dataframe 
                #      ['gene_id', 'transcript_id', 'chrom', 'position', 'pred_type', 'score', 'splice_type', 'strand',
                #       'window_end', 'window_start', 'sequence']

                if correct_label.lower() == 'tp':
                    tp_featurized_df = tp_featurized_df.select(meta_columns + tp_features)
                    assert isinstance(tp_featurized_df, pl.DataFrame), "Expected a Polars DataFrame."
                elif correct_label.lower() == 'tn':
                    tn_featurized_df = tn_featurized_df.select(meta_columns + tn_features)
                    assert isinstance(tn_featurized_df, pl.DataFrame), "Expected a Polars DataFrame."
                
                if error_label.lower() == 'fp':
                    fp_featurized_df = fp_featurized_df.select(meta_columns + fp_features)
                    assert isinstance(fp_featurized_df, pl.DataFrame), "Expected a Polars DataFrame."
                elif error_label.lower() == 'fn':
                    fn_featurized_df = fn_featurized_df.select(meta_columns + fn_features)
                    assert isinstance(fn_featurized_df, pl.DataFrame), "Expected a Polars DataFrame."

                print_emphasized("[info] Harmonizing features ...")
                harmonized_fps = harmonized_fns = None

                # Harmonize features for FP
                if error_label.lower() == 'fp' and correct_label.lower() == 'tp':
                    # Harmonize features for FP wrt TP
                    harmonized_tps, harmonized_fps = harmonize_features(
                        [tp_featurized_df, fp_featurized_df],
                        [tp_features, fp_features]
                    )

                    tp_featurized_df = harmonized_tps
                    fp_featurized_df = harmonized_fps

                    assert set(harmonized_tps.columns) == set(harmonized_fps.columns), \
                        "Columns in TP and FP DataFrames do not match."

                elif error_label.lower() == 'fn' and correct_label.lower() == 'tp':
                    # Harmonize features for FN wrt TP
                    harmonized_tps, harmonized_fns = harmonize_features(
                        [tp_featurized_df, fn_featurized_df],
                        [tp_features, fn_features]
                    )

                    tp_featurized_df = harmonized_tps
                    fn_featurized_df = harmonized_fns

                    assert set(harmonized_tps.columns) == set(harmonized_fns.columns), \
                        "Columns in TP and FN DataFrames do not match."

                elif error_label.lower() == 'fn' and correct_label.lower() == 'tn': 
                    # Harmonize features for FN wrt TN
                    harmonized_tns, harmonized_fns = harmonize_features(
                        [tn_featurized_df, fn_featurized_df],
                        [tn_features, fn_features]
                    )

                    tn_featurized_df = harmonized_tns
                    fn_featurized_df = harmonized_fns

                    assert set(harmonized_tns.columns) == set(harmonized_fns.columns), \
                        "Columns in TN and FN DataFrames do not match."

                else: 
                    raise ValueError(f"Invalid pred_type combinations: {error_label} vs {correct_label}")

                if tp_featurized_df is not None and isinstance(tp_featurized_df, pl.DataFrame):
                    tp_featurized_df = tp_featurized_df.to_pandas()
                if tn_featurized_df is not None and isinstance(tn_featurized_df, pl.DataFrame):
                    tn_featurized_df = tn_featurized_df.to_pandas()
                if fp_featurized_df is not None and isinstance(fp_featurized_df, pl.DataFrame):
                    fp_featurized_df = fp_featurized_df.to_pandas()
                if fn_featurized_df is not None and isinstance(fn_featurized_df, pl.DataFrame):
                    fn_featurized_df = fn_featurized_df.to_pandas()
            
                # Below we'll use Pandas features
                # assert isinstance(tp_featurized_df, pd.DataFrame), "Expected a Pandas DataFrame."
                # print("[info] Converted Polars DataFrames to Pandas DataFrames for downstream processing ...")

                # Combine datasets
                print_emphasized("[info] Combining datasets encoding labels ...")
                if error_label.lower() == 'fp' and correct_label.lower() == 'tp':
                    tp_featurized_df['label'] = 0  # Negative examples
                    fp_featurized_df['label'] = 1  # Positive examples

                    df_trainset = pd.concat([tp_featurized_df, fp_featurized_df], ignore_index=True)
                    display_dataframe_in_chunks(df_trainset, title="TP+FP training data after sequence featurization")
                    print(f"[info] Columns in TP+FP training data: {list(df_trainset.columns)[:100]}")

                elif error_label.lower() == 'fn' and correct_label.lower() == 'tp': 
                    tp_featurized_df['label'] = 0  # Negative examples
                    fn_featurized_df['label'] = 1  # Positive examples

                    df_trainset = pd.concat([tp_featurized_df, fn_featurized_df], ignore_index=True)
                    display_dataframe_in_chunks(df_trainset, title="TP+FN training data after sequence featurization")
                    print(f"[info] Columns in TP+FN training data: {list(df_trainset.columns)[:100]}")
                elif error_label.lower() == 'fn' and correct_label.lower() == 'tn': 
                    tn_featurized_df['label'] = 0  # Negative examples 
                    fn_featurized_df['label'] = 1  # Positive examples

                    df_trainset = pd.concat([tn_featurized_df, fn_featurized_df], ignore_index=True)
                    display_dataframe_in_chunks(df_trainset, title="TN+FN training data after sequence featurization")
                    print(f"[info] Columns in TN+FN training data: {list(df_trainset.columns)[:100]}")
                else: 
                    raise ValueError(f"Invalid pred_type combination: {error_label} vs {correct_label}")

                ############################################################
                # subject = f"{error_label}_vs_{correct_label}_seq_featurized"
                featurized_dataset_path = \
                    mefd.save_featurized_artifact(
                        df_trainset, 
                        chr=chr, chunk_start=chunk_start, chunk_end=chunk_end, 
                        aggregated=False, 
                        subject=subject, 
                        error_label=error_label, correct_label=correct_label)
                print_emphasized(f"[info] Saved featurized dataset (subject={subject}) to {featurized_dataset_path}")
                print_with_indent(f"Columns: {display_feature_set(df_trainset, max_kmers=100)}", indent_level=1)
                # Path: .../data/ensembl/spliceai_eval/full_{pred_type}_seq_featurized.tsv

                ############################################################
                # Additionally save the corresponding analysis sequence dataset
                if error_label.lower() == 'fp' and correct_label.lower() == 'tp':
                    analysis_subset_df = \
                        concatenate_dataframes(tp_sequence_df, fp_sequence_df, axis=0)
                    
                elif error_label.lower() == 'fn' and correct_label.lower() == 'tp': 
                    analysis_subset_df = \
                        concatenate_dataframes(tp_sequence_df, fn_sequence_df, axis=0)

                elif error_label.lower() == 'fn' and correct_label.lower() == 'tn': 
                    analysis_subset_df = \
                        concatenate_dataframes(tn_sequence_df, fn_sequence_df, axis=0)
                else: 
                    raise ValueError(f"Invalid pred_type combination: {error_label} vs {correct_label}")
                    
                analysis_subset_path = \
                    mefd.save_analysis_sequences(
                        analysis_subset_df, 
                        chr=chr, chunk_start=chunk_start, chunk_end=chunk_end, 
                        aggregated=False, 
                        error_label=error_label, correct_label=correct_label
                    )
                print_emphasized(f"[info] Saved analysis sequences (pred_type={pred_type}) to {analysis_subset_path}")
                print_with_indent(f"Columns: {list(analysis_subset_df.columns)}", indent_level=1)

            else: 
                print_emphasized("[i/o] Loading featurized dataset ...")
                df_trainset = \
                    mefd.load_featurized_artifact(
                        chr=chr, chunk_start=chunk_start, chunk_end=chunk_end, 
                        aggregated=False, 
                        subject=subject, 
                        error_label=error_label, 
                        correct_label=correct_label
                    )


def make_kmer_featurized_dataset(gtf_file=None, **kargs): 
    # from .sequence_featurizer import (
    #     extract_error_sequences, 
    #     extract_analysis_sequences, 
    #     featurize_gene_sequences,
    #     verify_consensus_sequences, 
    #     verify_consensus_sequences_via_analysis_data,
    #     display_feature_set,
    #     harmonize_features
    # )
    # from .extract_genomic_features import (
    #     FeatureAnalyzer, SpliceAnalyzer, 
    #     run_genomic_gtf_feature_extraction, 
    #     # compute_splice_site_distances, 
    #     compute_distances_to_transcript_boundaries,
    #     compute_distances_with_strand_adjustment,  
    #     compute_total_lengths,
    #     check_conflicting_overlaps
    # )
    # from .performance_analyzer import PerformanceAnalyzer
    from meta_spliceai.splice_engine.extract_gene_sequences import lookup_ensembl_ids

    pred_type = kargs.get("pred_type", 'FP')
    error_label = kargs.get("error_label", pred_type)  # Error prediction types (as positive class)
    correct_label = kargs.get("correct_label", 'TP')  # Correct prediction types (as negative class)

    overwrite = kargs.get("overwrite", True)
    # test_mode = True  # If true, load test analysis data
    subset_genes = kargs.get("subset_genes", True)  # If True, limit the number of genes to a subset
    subset_policy = kargs.get("subset_policy", 'hard') # 'random' or 'top'
    custom_genes = kargs.get('custom_genes', [])
    n_genes = kargs.get('n_genes', 1000)
    col_tid = kargs.get('col_tid', 'transcript_id')

    # Initialize analyzer objects if needed
    fa = kargs.get('fa', None)
    mefd = kargs.get('mefd', None)

    # Initialize the feature analyzer
    if fa is None: 
        fa = FeatureAnalyzer(gtf_file=gtf_file, overwrite=overwrite, col_tid=col_tid)
    format = fa.format 
    seperator = '\t' if format == 'tsv' else ','

    print_emphasized("[i/o] Accessing splice site analysis data ...")
    # A. If we want comprehensive datasets, use retrieve_splice_site_analysis_data()
    # result_set = \
    #     retrieve_splice_site_analysis_data(
    #         gtf_file, 
    #         load_datasets=['gene_features', 'transcript_features', 'splice_sites_df', ])
    # error_df = result_set['error_df']
    # position_df = result_set['position_df']
    # splice_sites_df = result_set['splice_sites_df']
    # gene_feature_df = result_set['gene_features']
    # transcript_feature_df = result_set['transcript_features']

    # B. Retrieve gene features specifically
    gene_feature_df = fa.retrieve_gene_features()

    print_section_separator()

    gtf_file_path = ErrorAnalyzer.gtf_file
    # ea = ErrorAnalyzer(window_size=500)  # Uses ModelEvaluationFileHandler to load data
    # df_tp = ea.retrieve_tp_data_points(overwrite=overwrite)
    # df_fp = ea.retrieve_fp_data_points(overwrite=overwrite)
    # df_fn = ea.retrieve_fn_data_points(overwrite=overwrite)
    # extract_tp_data_points(gtf_file=ErrorAnalyzer.gtf_file, window_size=500, save=True)
    # extract_fp_data_points(gtf_file=ErrorAnalyzer.gtf_file, window_size=500, save=True)
    # extract_fn_data_points(gtf_file=ErrorAnalyzer.gtf_file, window_size=500, save=True)

    # Initialize the model analyzer
    if mefd is None: 
        mefd = ModelEvaluationFileHandler(ErrorAnalyzer.eval_dir, separator=seperator)
    # splice_pos_df = mefd.load_splice_positions(aggregated=True)

    # Use run_spliceai_workflow.error_analysis_workflow() to generate the error analysis data
    analysis_sequence_df = \
        mefd.load_analysis_sequences(aggregated=True)  # Output is a Polars DataFrame
    print("[info] Columns in analysis_sequence_df:", analysis_sequence_df.columns)
    # ['gene_id', 'position', 'pred_type', 'score', 'splice_type', 'chrom', 'window_start', 'window_end', 'sequence']
    
    if subset_genes: 
        random_subset = subset_policy.startswith('rand')
        print_emphasized(f"[info] Limiting the number of genes to n={n_genes} ...")

        # Incorporate custom genes
        additional_gene_ids = []
        if custom_genes:
            # Determine if custom genes are IDs or names
            if all(str(gene).startswith('ENSG') for gene in custom_genes):
                additional_gene_ids = custom_genes
            else:
                # Map gene names to gene IDs
                data_prefix = "/path/to/meta-spliceai/data/ensembl"
                gene_id_to_name = lookup_ensembl_ids(
                    file_path=os.path.join(data_prefix, "gene_id_name_map.csv"),
                    gene_names=custom_genes,
                    file_format="csv"
                )
                # Todo: If gene_id_name_map.csv is not available, use the gtf_file to map gene names to IDs
                #       by calling extract_gene_sequences.build_gene_id_to_name_map()
                additional_gene_ids = list(gene_id_to_name.keys())

        if random_subset:
            # Randomly sample n_genes
            print_with_indent(f"Randomly sampling n={n_genes} genes ...")
            unique_gene_ids = analysis_sequence_df.select(pl.col('gene_id')).unique()
            num_unique_genes = unique_gene_ids.height
            print_emphasized(f"[info] Number of unique gene IDs: {num_unique_genes}")
            print_with_indent(f"Taking a subet of gene IDs n={n_genes} ...")

            sampled_gene_ids = unique_gene_ids.sample(n=n_genes, with_replacement=False)
            if len(additional_gene_ids) > 0:
                # Convert additional_gene_ids to a Polars DataFrame
                additional_gene_ids_df = pl.DataFrame({'gene_id': additional_gene_ids})
                
                # Concatenate the sampled_gene_ids with additional_gene_ids
                sampled_gene_ids = sampled_gene_ids.vstack(additional_gene_ids_df).unique()
                # In Polars, the vstack method is used to concatenate two DataFrames vertically

            # Filter the original DataFrame to include only the sampled gene_ids
            analysis_sequence_df = analysis_sequence_df.filter(pl.col('gene_id').is_in(sampled_gene_ids['gene_id']))
        elif subset_policy.startswith(('hard', 'diff', 'challen')):
            print_emphasized("[info] Selecting 'hard' genes ...")

            # pa = PerformanceAnalyzer(output_dir=analysis_dir)
            pa = PerformanceAnalyzer()

            strategy = kargs.get('strategy', 'worst')
            threshold = kargs.get('threshold', 0.5)

            df_hard_genes = pa.retrieve_hard_genes(
                error_type=pred_type, 
                n_genes=n_genes, 
                strategy=strategy, threshold=threshold, 
                overwrite=overwrite)
            display_dataframe_in_chunks(df_hard_genes, title="Hard genes")

            if df_hard_genes is not None:
                unique_genes_df = df_hard_genes.unique(subset=['gene_id'])
                hard_gene_ids = unique_genes_df.select('gene_id').to_series().to_list()

                if len(additional_gene_ids) > 0:
                    # Add additional gene IDs to the hard gene list
                    hard_gene_ids.extend(additional_gene_ids)
                    hard_gene_ids = list(set(hard_gene_ids))  # Ensure uniqueness

                # Subset analysis_sequence_df to keep only rows for the longest genes
                analysis_sequence_df = analysis_sequence_df.filter(pl.col('gene_id').is_in(hard_gene_ids))

                n_genes = analysis_sequence_df.select(pl.col('gene_id').n_unique()).to_series()[0]
                print_with_indent(f"[workflow] Analyzing n={n_genes} 'hard' genes (for which SpliceAI didn't do well) ...", indent_level=1)
            else: 
                raise ValueError(f"No hard genes found for error_type={pred_type}.")

        else: 
            print_with_indent(f"Selecting the top n={n_genes} genes by gene lengths ...", indent_level=1)
        
            # Remove duplicate rows based on gene_id
            unique_genes_df = analysis_sequence_df.unique(subset=['gene_id'])

            # Join unique_genes_df with gene_feature_df to get gene-level features and sort by gene length
            top_genes_df = unique_genes_df.join(
                gene_feature_df.select(['gene_id', 'gene_length']),
                on='gene_id',
                how='inner'
            ).sort('gene_length', descending=True).head(n_genes)

            # Display gene_id vs gene_length to verify the result
            print(top_genes_df.select(['gene_id', 'gene_length']))

            # Get the list of top gene_ids
            top_gene_ids = top_genes_df.select('gene_id').to_series().to_list()

            if len(additional_gene_ids) > 0:
                top_gene_ids.extend(additional_gene_ids)
                top_gene_ids = list(set(top_gene_ids))  # Ensure uniqueness

            # Subset analysis_sequence_df to keep only rows for the longest genes
            analysis_sequence_df = analysis_sequence_df.filter(pl.col('gene_id').is_in(top_gene_ids))

            # Drop the gene_length column
            # analysis_sequence_df = analysis_sequence_df.drop('gene_length')
    # NOTE: analysis_sequence_df is a Polars DataFrame

    # Determine the final unique number of genes in analysis_sequence_df
    final_unique_gene_ids = analysis_sequence_df.select(pl.col('gene_id')).unique()
    final_gene_ids_set = set(final_unique_gene_ids.to_series().to_list())
    
    # Check for missing custom genes
    missing_genes = [gene for gene in custom_genes if gene not in final_gene_ids_set]
    if missing_genes:
        print(f"[warning] The following custom genes were not found in the final unique gene IDs: {missing_genes}")
    else:
        print("[info] All custom genes were found in the final unique gene IDs.")
    final_num_unique_genes = final_unique_gene_ids.height
    print_emphasized(f"[info] Final number of unique gene IDs: {final_num_unique_genes}")

    # Add additional columns to the analysis_sequence_df (e.g. strand)
    if 'strand' not in analysis_sequence_df.columns:
        analysis_sequence_df = analysis_sequence_df.join(
            gene_feature_df.select(['gene_id', 'strand']),
            on='gene_id',
            how='inner'
        )
    print("[info] Columns in analysis_sequence_df:", analysis_sequence_df.columns)
    # display_dataframe_in_chunks(analysis_sequence_df, title="Analysis sequence data")
    # position: absolute position

    # Feature set format 
    # format = fa.format
    # seperator = '\t' if format == 'tsv' else ','

    ############################################################
    subject = "seq_featurized"
    # analysis_subject = f"{error_label}_vs_{correct_label}_analysis_sequences"

    run_sequence_featurization = True
    # col_tid = 'transcript_id'
    kmer_sizes = kargs.get("kmer_sizes", [6, ])   
    if kmer_sizes is None: 
        kmer_sizes = [6, ]
    elif isinstance(kmer_sizes, int):
        kmer_sizes = [kmer_sizes]
    
    df_trainset = pd.DataFrame()
    # df_trainset_fp = pd.DataFrame()
    # df_trainset_fn = pd.DataFrame()

    if run_sequence_featurization:
    
        print_emphasized(f"1) Aligning splice sites to transcripts  ...")

        tp_sequence_df = tn_sequence_df = None
        if correct_label.lower() == 'tp':
            print("[info] Aligning transcript IDs for TP analysis ...")
            tp_sequence_df, unmatched_genes = \
                align_transcript_ids_for_tp_analysis(
                    analysis_sequence_df, gtf_file=ErrorAnalyzer.gtf_file, throw_exception=False)
            print(f"[info] Shape of tp_sequence_df: {tp_sequence_df.shape}")
            assert 'sequence' in tp_sequence_df.columns, "Column 'sequence' not found in tp_sequence_df."
            print("[info] type(tp_sequence_df):", type(tp_sequence_df))
            print(f"[info] Columns in tp_sequence_df: {list(tp_sequence_df.columns)[:100]}")

        if correct_label.lower() == 'tn': 
            print("[info] Aligning transcript IDs for TN analysis ...")
            tn_sequence_df, unmatched_genes = \
                align_transcript_ids_for_tn_analysis(
                    analysis_sequence_df, gtf_file=ErrorAnalyzer.gtf_file)
            print(f"[info] Shape of tn_sequence_df: {tn_sequence_df.shape}")
            assert 'sequence' in tn_sequence_df.columns, "Column 'sequence' not found in tn_sequence_df."
            print("[info] type(tn_sequence_df):", type(tn_sequence_df))
            print(f"[info] Columns in tn_sequence_df: {list(tn_sequence_df.columns)[:100]}")
        
        # Remove unmatched genes
        if unmatched_genes: 
            print_emphasized(f"Removing unmatched genes from the input dataframe (n={len(unmatched_genes)})...")
            analysis_sequence_df = analysis_sequence_df.filter(~pl.col('gene_id').is_in(unmatched_genes))
            # analysis_sequence_df = analysis_sequence_df[~analysis_sequence_df['gene_id'].isin(unmatched_genes)]

        fp_sequence_df = fn_sequence_df = None
        if error_label.lower() == 'fp':
            print_emphasized(f"Aligning transcript IDs for FP analysis ...")
            fp_sequence_df, unmatched_genes = \
                align_transcript_ids_for_fp_analysis(
                    analysis_sequence_df, gtf_file=ErrorAnalyzer.gtf_file)
            print(f"[info] Shape of fp_sequence_df: {fp_sequence_df.shape}")
            assert 'sequence' in fp_sequence_df.columns, "Column 'sequence' not found in fp_sequence_df."
            print("[info] type(fp_sequence_df):", type(fp_sequence_df))
            print(f"[info] Columns in fp_sequence_df: {list(fp_sequence_df.columns)[:100]}")

        elif error_label.lower() == 'fn':
            print_emphasized(f"Aligning transcript IDs for FN analysis ...")
            fn_sequence_df, unmatched_genes = \
                align_transcript_ids_for_fn_analysis(
                    analysis_sequence_df, gtf_file=ErrorAnalyzer.gtf_file)
            print(f"[info] Shape of fn_sequence_df: {fn_sequence_df.shape}")
            assert 'sequence' in fn_sequence_df.columns, "Column 'sequence' not found in fn_sequence_df."
            print("[info] type(fn_sequence_df):", type(fn_sequence_df))
            print(f"[info] Columns in fn_sequence_df: {list(fn_sequence_df.columns)[:100]}")

        ################################################
        print_emphasized(f"2) Featurizing analysis sequences with correct_label={correct_label} ...")

        tp_featurized_df = tn_featurized_df = None
        tp_features = tn_features = None

        if correct_label.lower() == 'tp':
            print_with_indent(f"[action] Featurizing analysis sequences of type TP ...", indent_level=1)
            tp_featurized_df, tp_features = featurize_gene_sequences(tp_sequence_df, kmer_sizes=kmer_sizes, return_feature_set=True, verbose=1)
        if correct_label.lower() == 'tn':
            
            # Define the fraction of data to keep (e.g., 0.1 for 10%)
            tn_sample_fraction = kargs.get('tn_sample_fraction', 0.5)
            tn_max_sample_size = kargs.get('tn_max_sample_size', 50000)  # Adjust this value as needed
            tn_sequence_df = downsample_dataframe(
                tn_sequence_df, 
                sample_fraction=tn_sample_fraction, 
                max_sample_size=tn_max_sample_size, verbose=1)

            # Check the shape of the downsampled DataFrame
            if tn_sample_fraction < 1.0:
                print(f"[info] Shape of (downsampled) tn_sequence_df: {tn_sequence_df.shape}")
            
            print_with_indent(f"[action] Featurizing analysis sequences of type TN ...", indent_level=1)
            tn_featurized_df, tn_features = featurize_gene_sequences(tn_sequence_df, kmer_sizes=kmer_sizes, return_feature_set=True, verbose=1) 

        fp_featurized_df = fn_featurized_df = None
        if error_label.lower() == 'fp':
            print_emphasized(f"[action] Featurizing analysis sequences of type FP ...")
            fp_featurized_df, fp_features = featurize_gene_sequences(fp_sequence_df, kmer_sizes=kmer_sizes, return_feature_set=True, verbose=1)

            if tp_features is not None:
                missing_in_tp = set(fp_features) - set(tp_features)
                missing_in_fp = set(tp_features) - set(fp_features)

                if missing_in_tp or missing_in_fp:
                    print("[diagnostics] Feature set mismatch detected between TPs and FPs!")
                    print(f"Features missing in TP: {missing_in_tp}")
                    print(f"Features missing in FP: {missing_in_fp}")
                else: 
                    print_emphasized("[info] Feature sets are consistent between TPs and FPs.")
        
        elif error_label.lower() == 'fn':
            print_emphasized(f"[action] Featurizing analysis sequences of type FN ...")
            fn_featurized_df, fn_features = featurize_gene_sequences(fn_sequence_df, kmer_sizes=kmer_sizes, return_feature_set=True, verbose=1)

            # Define the fraction of data to keep (e.g., 0.1 for 10%)
            fn_sample_fraction = kargs.get('fn_sample_fraction', 1.0)
            fn_max_sample_size = kargs.get('fn_max_sample_size', 50000)  # Adjust this value as needed
            fn_sequence_df = downsample_dataframe(
                fn_sequence_df, 
                sample_fraction=fn_sample_fraction, 
                max_sample_size=fn_max_sample_size, verbose=1)

            if fn_sample_fraction < 1.0:
                print(f"[info] Shape of (downsampled) fn_sequence_df: {fn_sequence_df.shape}")

            if tp_features is not None:
                missing_in_tp = set(fn_features) - set(tp_features)
                missing_in_fn = set(tp_features) - set(fn_features)

                if missing_in_tp or missing_in_fn:
                    print("[diagnostics] Feature set mismatch detected between TPs and FNs!")
                    print(f"Features missing in TP: {missing_in_fp}")
                    print(f"Features missing in FN: {missing_in_fn}")
                else: 
                    print_emphasized("[info] Feature sets are consistent between TPs and FNs.")

        # Keep selective meta features
        # fp_featurized_df = fp_featurized_df.rename({'gene_strand': 'strand'})  # Rename gene_strand to strand
        # fp_featurized_df = fp_featurized_df.drop('gene_strand')

        meta_columns = ['gene_id', col_tid, 'position', 'score', 'splice_type', 'chrom', 'strand']
        # NOTE: Columns in a typical analysis sequence dataframe 
        #      ['gene_id', 'transcript_id', 'chrom', 'position', 'pred_type', 'score', 'splice_type', 'strand',
        #       'window_end', 'window_start', 'sequence']

        if correct_label.lower() == 'tp':
            tp_featurized_df = tp_featurized_df.select(meta_columns + tp_features)
            assert isinstance(tp_featurized_df, pl.DataFrame), "Expected a Polars DataFrame."
        elif correct_label.lower() == 'tn':
            tn_featurized_df = tn_featurized_df.select(meta_columns + tn_features)
            assert isinstance(tn_featurized_df, pl.DataFrame), "Expected a Polars DataFrame."
        
        if error_label.lower() == 'fp':
            fp_featurized_df = fp_featurized_df.select(meta_columns + fp_features)
            assert isinstance(fp_featurized_df, pl.DataFrame), "Expected a Polars DataFrame."
        elif error_label.lower() == 'fn':
            fn_featurized_df = fn_featurized_df.select(meta_columns + fn_features)
            assert isinstance(fn_featurized_df, pl.DataFrame), "Expected a Polars DataFrame."

        print_emphasized("[info] Harmonizing features ...")
        harmonized_fps = harmonized_fns = None

        # Harmonize features for FP
        if error_label.lower() == 'fp' and correct_label.lower() == 'tp':
            # Harmonize features for FP wrt TP
            harmonized_tps, harmonized_fps = harmonize_features(
                [tp_featurized_df, fp_featurized_df],
                [tp_features, fp_features]
            )

            tp_featurized_df = harmonized_tps
            fp_featurized_df = harmonized_fps

            assert set(harmonized_tps.columns) == set(harmonized_fps.columns), \
                "Columns in TP and FP DataFrames do not match."

        elif error_label.lower() == 'fn' and correct_label.lower() == 'tp':
            # Harmonize features for FN wrt TP
            harmonized_tps, harmonized_fns = harmonize_features(
                [tp_featurized_df, fn_featurized_df],
                [tp_features, fn_features]
            )

            tp_featurized_df = harmonized_tps
            fn_featurized_df = harmonized_fns

            assert set(harmonized_tps.columns) == set(harmonized_fns.columns), \
                "Columns in TP and FN DataFrames do not match."

        elif error_label.lower() == 'fn' and correct_label.lower() == 'tn': 
            # Harmonize features for FN wrt TN
            harmonized_tns, harmonized_fns = harmonize_features(
                [tn_featurized_df, fn_featurized_df],
                [tn_features, fn_features]
            )

            tn_featurized_df = harmonized_tns
            fn_featurized_df = harmonized_fns

            assert set(harmonized_tns.columns) == set(harmonized_fns.columns), \
                "Columns in TN and FN DataFrames do not match."

        else: 
            raise ValueError(f"Invalid pred_type combinations: {error_label} vs {correct_label}")

        if tp_featurized_df is not None and isinstance(tp_featurized_df, pl.DataFrame):
            tp_featurized_df = tp_featurized_df.to_pandas()
        if tn_featurized_df is not None and isinstance(tn_featurized_df, pl.DataFrame):
            tn_featurized_df = tn_featurized_df.to_pandas()
        if fp_featurized_df is not None and isinstance(fp_featurized_df, pl.DataFrame):
            fp_featurized_df = fp_featurized_df.to_pandas()
        if fn_featurized_df is not None and isinstance(fn_featurized_df, pl.DataFrame):
            fn_featurized_df = fn_featurized_df.to_pandas()
    
        # Below we'll use Pandas features
        # assert isinstance(tp_featurized_df, pd.DataFrame), "Expected a Pandas DataFrame."
        # print("[info] Converted Polars DataFrames to Pandas DataFrames for downstream processing ...")

        # Combine datasets
        print_emphasized("[info] Combining datasets encoding labels ...")
        if error_label.lower() == 'fp' and correct_label.lower() == 'tp':
            tp_featurized_df['label'] = 0  # Negative examples
            fp_featurized_df['label'] = 1  # Positive examples

            df_trainset = pd.concat([tp_featurized_df, fp_featurized_df], ignore_index=True)
            display_dataframe_in_chunks(df_trainset, title="TP+FP training data after sequence featurization")
            print(f"[info] Columns in TP+FP training data: {list(df_trainset.columns)[:100]}")

        elif error_label.lower() == 'fn' and correct_label.lower() == 'tp': 
            tp_featurized_df['label'] = 0  # Negative examples
            fn_featurized_df['label'] = 1  # Positive examples

            df_trainset = pd.concat([tp_featurized_df, fn_featurized_df], ignore_index=True)
            display_dataframe_in_chunks(df_trainset, title="TP+FN training data after sequence featurization")
            print(f"[info] Columns in TP+FN training data: {list(df_trainset.columns)[:100]}")
        elif error_label.lower() == 'fn' and correct_label.lower() == 'tn': 
            tn_featurized_df['label'] = 0  # Negative examples 
            fn_featurized_df['label'] = 1  # Positive examples

            df_trainset = pd.concat([tn_featurized_df, fn_featurized_df], ignore_index=True)
            display_dataframe_in_chunks(df_trainset, title="TN+FN training data after sequence featurization")
            print(f"[info] Columns in TN+FN training data: {list(df_trainset.columns)[:100]}")
        else: 
            raise ValueError(f"Invalid pred_type combination: {error_label} vs {correct_label}")
      
        # Remove some meta data columns
        # columns_to_drop = ['pred_type', 'window_start', 'window_end', 'absolute_position', 'start', 'end']
        # df_trainset = df_trainset.drop(columns=columns_to_drop)

        # subject = f"{error_label}_vs_{correct_label}_seq_featurized"
        featurized_dataset_path = \
            mefd.save_featurized_artifact(
                df_trainset, aggregated=True, 
                subject=subject, 
                error_label=error_label, correct_label=correct_label)
        print_emphasized(f"[info] Saved featurized dataset (subject={subject}) to {featurized_dataset_path}")
        print_with_indent(f"Columns: {display_feature_set(df_trainset, max_kmers=100)}", indent_level=1)
        # Path: .../data/ensembl/spliceai_eval/full_{pred_type}_seq_featurized.tsv

        ############################################################
        # analysis_subject = f"{error_label}_vs_{correct_label}_analysis_sequences"

        # Additionally save the corresponding analysis sequence dataset
        if error_label.lower() == 'fp' and correct_label.lower() == 'tp':
            analysis_subset_df = \
                concatenate_dataframes(tp_sequence_df, fp_sequence_df, axis=0)
            
        elif error_label.lower() == 'fn' and correct_label.lower() == 'tp': 
            analysis_subset_df = \
                concatenate_dataframes(tp_sequence_df, fn_sequence_df, axis=0)

        elif error_label.lower() == 'fn' and correct_label.lower() == 'tn': 
            analysis_subset_df = \
                concatenate_dataframes(tn_sequence_df, fn_sequence_df, axis=0)
        else: 
            raise ValueError(f"Invalid pred_type combination: {error_label} vs {correct_label}")
            
        analysis_subset_path = \
            mefd.save_analysis_sequences(
                analysis_subset_df, aggregated=True, 
                error_label=error_label, correct_label=correct_label
            )
        print_emphasized(f"[info] Saved analysis sequences (pred_type={pred_type}) to {analysis_subset_path}")
        print_with_indent(f"Columns: {list(analysis_subset_df.columns)}", indent_level=1)

    else: 
        print_emphasized("[i/o] Loading featurized dataset ...")
        df_trainset = \
            mefd.load_featurized_artifact(
                aggregated=True, subject=subject, 
                error_label=error_label, 
                correct_label=correct_label
            )

    return df_trainset


def run_training_data_generation_workflow(
    gtf_file=None, 
    pred_type='FP',
    kmer_sizes=[6, ],
    subset_genes=True, 
    subset_policy='hard',  # 'random', 'top', 'hard', ...
    n_genes=1000,  # relevant when subset_genes=True
    **kargs
):

    error_label = kargs.get('error_label', pred_type)
    correct_label = kargs.get('correct_label', 'TP')

    subject = "seq_featurized"
    overwrite = kargs.get("overwrite", True)
    col_tid = kargs.get('col_tid', 'transcript_id')
    custom_genes = kargs.get('custom_genes', [])

    # subset_genes = kargs.get("subset_genes", True)  # If True, limit the number of genes to a subset
    # subset_policy = kargs.get("subset_policy", 'hard') # 'random' or 'top'
    # n_genes = kargs.get('n_genes', 1000)

    # Initialize the feature analyzer
    fa = FeatureAnalyzer(gtf_file=gtf_file, overwrite=overwrite, col_tid=col_tid)
    format = fa.format 
    seperator = '\t' if format == 'tsv' else ','
    
    # Initialize the model analyzer
    mefd = ModelEvaluationFileHandler(ErrorAnalyzer.eval_dir, separator=seperator)
    
    # Stage 1: Generate the main featurized dataset with k‐mers and save it out.
    #  - Extract features from analysis sequence data
    print_emphasized("[workflow] Extract kmer features from analysis sequence data ...")
    print_with_indent(f"[params] pred_type: {pred_type}, kmer_sizes: {kmer_sizes}, subset_genes: {subset_genes}, subset_policy: {subset_policy}, n_genes: {n_genes}", indent_level=1)
    print_with_indent(f"[params] error_label: {error_label}, correct_label: {correct_label}", indent_level=1)
    print_with_indent(f"[params] custom_genes: {custom_genes}", indent_level=1)
    
    make_kmer_featurized_dataset(
        gtf_file=gtf_file, 
        
        pred_type=pred_type, 
        error_label=error_label,
        correct_label=correct_label,
        
        kmer_sizes=kmer_sizes,

        # Subsampling parameters
        subset_genes=subset_genes,
        subset_policy=subset_policy,
        custom_genes=custom_genes,
        n_genes=n_genes,
        fn_sample_fraction=kargs.get('fn_sample_fraction', 1.0),
        fn_max_sample_size=kargs.get('fn_max_sample_size', 50000),
        tn_sample_fraction=kargs.get('tn_sample_fraction', 0.5),
        tn_max_sample_size=kargs.get('tn_max_sample_size', 50000),
        overwrite=overwrite, 

        # Analyzer instances
        fa=fa, 
        mefd=mefd,
    )
    df_trainset = mefd.load_featurized_artifact(
        aggregated=True, subject=subject, 
        error_label=error_label, 
        correct_label=correct_label
    )
    # At this point df_trainset is a Polars DataFrame

    columns0 = display_feature_set(df_trainset, max_kmers=100)
    print_emphasized(f"[info] Columns in df_trainset prior to incorporating features from various data sources:\n{columns0}\n")
    # NOTE: ['gene_id', 'transcript_id', 'position', 'score', 'splice_type', 'chrom', 'strand', 
    #        '3mer_CTT', '3mer_CCC', '3mer_GTC', '3mer_AAA', '2mer_CA', '3mer_TCT', '3mer_CGA', '3mer_TCA', '3mer_AAT', ...]
    shape0 = df_trainset.shape
    print_with_indent(f"shape(df_trainset): {shape0}", indent_level=1)
    count_unique_ids(df_trainset, col_tid=col_tid)

    # --- Test ----
    # Check the data integrity for gene_id and transcript_id
    df_abnormal = check_and_subset_invalid_transcript_ids(df_trainset, col_tid=col_tid, verbose=1)
    if not is_dataframe_empty(df_abnormal): 
        print("[test] Abnormal data in training data (df_trainset):")
        cols = ['gene_id', col_tid, 'position', 'score', 'splice_type', 'chrom', 'strand']
        display(subsample_dataframe(df_abnormal, columns=cols, num_rows=10, random=True))
        # raise ValueError("Abnormal data found in df_trainset.")
        df_trainset = \
            filter_and_validate_ids(df_trainset, col_tid=col_tid, verbose=1)

    ############################################################
    # Gene-level features 

    print_emphasized("[workflow] Extracting gene-level features from GTF file ...")
    df_trainset = incorporate_gene_level_features(df_trainset, fa=fa)
    shape0 = df_trainset.shape

    ############################################################

    print_emphasized(f"[info] Incorporating exon-intron length features via length_df ...")
    df_trainset = incorporate_length_features(df_trainset, fa=fa)
    
    shape0 = df_trainset.shape
    ############################################################
    # Peroformance profile features

    # Extract features from perfomance profile
    print_emphasized("[i/o] Extracting performance-profile features ...")
    df_trainset = incorporate_performance_features(df_trainset, fa=fa)
    
    shape0 = df_trainset.shape
    ############################################################
    # Overlapping genes

    sa = SpliceAnalyzer()
    print_emphasized("[info] Incorporating overlapping gene-specific features ...")
    df_trainset = incorporate_overlapping_gene_features(df_trainset, sa=sa)

    shape0 = df_trainset.shape
    ############################################################

    print_emphasized("[info] Incorporating splice site features ...")
    df_trainset = incorporate_distance_features(df_trainset, fa)
    
    shape0 = df_trainset.shape
    ############################################################

    featurized_dataset_path = \
            mefd.save_featurized_artifact(
                df_trainset, aggregated=True, 
                subject=subject, 
                error_label=error_label, correct_label=correct_label)
    print_emphasized(f"[info] Saved featurized dataset ({error_label}_vs_{correct_label}) to {featurized_dataset_path}")
    print_with_indent(f"Columns: {display_feature_set(df_trainset, max_kmers=100)}", indent_level=1)

    # Stage 2: Load that dataset again (from featurized_dataset_path) and ...
    # ... move on to the next phase which requires analysis sequences
    df_trainset = \
        make_training_data_with_analysis_sequences(
            fa=fa, 
            pred_type=pred_type, 
            error_label=error_label, 
            correct_label=correct_label)

    return df_trainset


def incorporate_gene_level_features(df_trainset, fa, **kargs): 
    """
    Incorporate various gene-level features, intron/exon lengths, etc. into df_trainset.

    1) Run genomic_gtf_feature_extraction() to get gene-level info (gene_fs_df).
    2) Merge gene_fs_df into df_trainset on (gene_id, transcript_id).
    3) Compute or load total_intron_exon_lengths and merge into df_trainset.
    4) Return the updated df_trainset.

    Parameters
    ----------
    df_trainset : pd.DataFrame
        Already featurized dataset (e.g. with k-mer columns, label, etc.)
    gtf_file : str
        Path to the GTF file for extracting gene-level features.
    fa : FeatureAnalyzer
        FeatureAnalyzer instance with references to analysis directories, etc.
    col_tid : str
        Column name for transcript ID. Default "transcript_id".
    overwrite : bool
        Whether to overwrite existing cached data in feature extraction. Default True.
    format_ : str
        File format for saving/loading (tsv or csv). Default "tsv".

    Returns
    -------
    df_trainset : pd.DataFrame
        The updated DataFrame after merging gene-level features and intron lengths.
    """
    # import os
    # from .extract_genomic_features import (run_genomic_gtf_feature_extraction, compute_total_lengths)
    # from .utils_df import (join_and_remove_duplicates, is_dataframe_empty)
    # from .analysis_utils import (
    #    check_duplicates, 
    #    find_missing_combinations, check_and_subset_invalid_transcript_ids)
    # from .utils_df import display, display_dataframe_in_chunks
    col_tid = fa.col_tid
    overwrite = fa.overwrite
    format_ = fa.format
    gtf_file = fa.gtf_file

    # analysis_dir = ErrorAnalyzer.analysis_dir  # or pass as param if needed
    shape0 = df_trainset.shape

    print_emphasized("[workflow] Extracting gene-level features from GTF file ...")
    output_path = os.path.join(fa.analysis_dir, f'genomic_gtf_feature_set.{format_}')
    gene_fs_df = run_genomic_gtf_feature_extraction(gtf_file, output_file=output_path)  # returns Polars df
    print("[info] Columns in genomic GTF-derived feature set:", list(gene_fs_df.columns))
    
    print_emphasized(f"[info] Saved genomic GTF-derived feature set to {output_path}")

    # Test: check if gene_fs_df has invalid transcript IDs
    df_abnormal = check_and_subset_invalid_transcript_ids(
        gene_fs_df, col_gid='gene_id', col_tid=col_tid, verbose=1
    )
    if not is_dataframe_empty(df_abnormal):
        print("[test] Abnormal data in gene_fs_df:")
        cols = ['gene_id', col_tid, 'start', 'end', 'strand', 'gene_type', 'chrom']
        display(subsample_dataframe(df_abnormal, columns=cols, num_rows=10, random=True))
        raise ValueError("Abnormal data found in gene_fs_df (from GTF).")

    on_columns = ['gene_id', col_tid]

    # Test: Check duplicates in df_trainset
    num_duplicates_trainset = check_duplicates(df_trainset, subset=on_columns, verbose=2, example_limit=5)
    print(f"[diagnostics] #duplicates in df_trainset based on {on_columns}: {num_duplicates_trainset}")

    # Test: Check duplicates in gene_fs_df
    num_duplicates_gene_fs = check_duplicates(gene_fs_df, subset=on_columns, verbose=2, example_limit=5)
    print(f"[diagnostics] #duplicates in gene_fs_df based on {on_columns}: {num_duplicates_gene_fs}")

    print_emphasized("[info] Merging GTF-derived gene-level features ...")
    df_trainset = join_and_remove_duplicates(
        df_trainset, gene_fs_df, on=on_columns, how='left', verbose=1
    )
    print(f"[info] shape(df_trainset) after merging gene-level feats: {shape0} -> {df_trainset.shape}")

    # Test: Check missing combos
    df_missing = find_missing_combinations(df_trainset, gene_fs_df, on_columns)
    print("[test] Missing combos in df_trainset that are not in gene_fs_df => shape:", df_missing.shape)
    display(df_missing.head(10))

    # Ensure row count didn't unexpectedly change
    assert shape0[0] == df_trainset.shape[0], \
        "Assertion failed: row count changed after merging gene-level features."

    # Next: total intron/exon lengths
    shape0 = df_trainset.shape

    return df_trainset


def incorporate_length_features(df_trainset, fa, **kargs): 

    # fa = FeatureAnalyzer(gtf_file=gtf_file, overwrite=overwrite)

    gtf_file = fa.gtf_file
    col_tid = fa.col_tid
    overwrite = fa.overwrite
    format_ = fa.format

    print_emphasized("[info] Incorporating exon-intron length features ...")
    output_path = os.path.join(fa.analysis_dir, 'total_intron_exon_lengths.tsv')
    if os.path.exists(output_path):
        length_df = pd.read_csv(output_path, sep='\t')
    else:
        length_df = compute_total_lengths(gtf_file)  # returns e.g. a pd.DataFrame
        path_length_features = fa.save_dataframe(length_df, 'total_intron_exon_lengths.tsv')
        assert output_path == path_length_features, f"File path mismatch: {output_path} != {path_length_features}"

    print(f"[i/o] Path(length_df): {output_path}")
    print("[test] type(length_df):", type(length_df))
    print("[test] columns(length_df):", list(length_df.columns))
    # e.g. ['transcript_id','total_exon_length','total_intron_length']
    shape0 = df_trainset.shape

    # Merge with df_trainset
    df_trainset = join_and_remove_duplicates(
        df_trainset, length_df, on=[col_tid], how='left', verbose=1
    )
    print(f"[info] shape(df_trainset) after merging exon-intron lengths: {shape0} -> {df_trainset.shape}")
    assert shape0[0] == df_trainset.shape[0], \
        "Assertion failed: row count changed after merging intron-exon lengths."

    return df_trainset


def incorporate_performance_features(df_trainset, fa, **kargs):

    # splice_stats_features = extract_gene_features_from_performance_profile()
    splice_stats_features = fa.retrieve_gene_level_performance_features()
    print("[test] columns(splice_stats_features):", list(splice_stats_features.columns))
    # NOTE: ['gene_id', 'n_splice_sites', 'splice_type']
    
    # splice_sites_df = SpliceAnalyzer().retrieve_splice_sites(column_names={'site_type': 'splice_type'})
    # print_with_indent(f"[info] Merging new feature set: splice_stats_features (type={type(splice_sites_df)}) ...", indent_level=1)
    
    shape0 = df_trainset.shape
    on_columns = ['gene_id', 'splice_type']
    df_trainset = join_and_remove_duplicates(df_trainset, splice_stats_features, on=on_columns, how='left', verbose=1)
    
    columns = display_feature_set(df_trainset, max_kmers=100)
    print(f"[info] Columns in updated df_trainset (n={len(df_trainset.columns)}):", list(columns))
    print(f"[info] shape(df_trainset): {shape0} -> {df_trainset.shape}")
    assert shape0[0] == df_trainset.shape[0], "Assertion failed: Number of rows changed after joining performance features."

    return df_trainset

def incorporate_overlapping_gene_features(df_trainset, sa=None, **kargs):

    if sa is None: 
        sa = SpliceAnalyzer()

    overlapping_genes_df = sa.retrieve_overlapping_gene_metadata(output_format='dataframe', to_pandas=True)
    print("[test] columns(overlapping_genes_df):", list(overlapping_genes_df.columns))
    print_with_indent(f"Is polars? {type(overlapping_genes_df)}", indent_level=1)  # Yes, it is a Polars DataFrame

    columns = ['gene_id_1', 'num_overlaps']
    overlapping_genes_df = overlapping_genes_df[columns].rename(columns={'gene_id_1': 'gene_id'})
    print("[test] columns(overlapping_genes_df):", list(overlapping_genes_df.columns))
    print("[test] shape(overlapping_genes_df): ", overlapping_genes_df.shape)  # (6317, 2)

    # Test: Check for duplicate rows in terms of gene_id
    conflicting_counts = check_conflicting_overlaps(overlapping_genes_df)
    if conflicting_counts is not None:
        display(conflicting_counts)
        raise ValueError("Conflicting overlaps detected.")

    # De-duplicate overlapping_genes_df on gene_id
    # overlapping_genes_df = overlapping_genes_df.unique(subset=['gene_id'])
    overlapping_genes_df = overlapping_genes_df.drop_duplicates(subset=['gene_id'])

    on_columns = ['gene_id', ]
    shape0 = df_trainset.shape

    # Perform a left join and fill missing values with 0 for num_overlaps
    df_trainset = join_and_remove_duplicates(df_trainset, overlapping_genes_df, on=on_columns, how='left', verbose=1)
    # df_trainset = df_trainset.merge(overlapping_genes_df, on=on_columns, how='left').fillna(0)
    
    columns = display_feature_set(df_trainset, max_kmers=100)
    print(f"[info] Columns in updated df_trainset (n={len(df_trainset.columns)}):", list(columns))
    print(f"[info] shape(df_trainset): {shape0} -> {df_trainset.shape}")
    assert shape0[0] == df_trainset.shape[0], "Assertion failed: Number of rows changed after joining overlapping genes."
    
    shape0 = df_trainset.shape

    return df_trainset


def incorporate_distance_features(df_trainset, fa, **kargs):
    col_tid = fa.col_tid

    # mefd = ModelEvaluationFileHandler(ErrorAnalyzer.eval_dir, separator='\t')
    # if df_trainset is None: 
    #     df_trainset = mefd.load_featurized_artifact(aggregated=True, subject=subject)
    # At this point df_trainset is a Polars DataFrame
    
    # Distance to transcript boundaries
    # distance_df = compute_splice_site_distances(splice_sites_df, transcript_feature_df, match_col='position')
    # print("[info] Columns in distance_df:", list(distance_df.columns))
    # display_dataframe_in_chunks(distance_df, title="Distance to transcript boundaries")
    # NOTE: ['gene_id', 'transcript_id', 'position', 'splice_type', 'distance_to_start', 'distance_to_end']

    display_dataframe_in_chunks(df_trainset, title="Training data before incorporating distance features")
    
    # Compute distance features directly on the training set
    shape0 = df_trainset.shape
    # df_trainset = compute_distances_to_transcript_boundaries(df_trainset, match_col='position')

    # Compute distances with strand adjustment
    df_trainset = compute_distances_with_strand_adjustment(df_trainset, match_col='position', col_tid=col_tid)
    # This function assumes that 'position' is a relative position within the gene

    columns = display_feature_set(df_trainset, max_kmers=100)
    print(f"[info] Columns in updated df_trainset (n={len(df_trainset.columns)}):", list(columns))
    print(f"[info] shape(df_trainset): {shape0} -> {df_trainset.shape}")
    assert shape0[0] == df_trainset.shape[0], "Assertion failed: Number of rows changed after computing distances with strand adjustment."
    
    print_section_separator()
    shape0 = df_trainset.shape

    # featurized_dataset_path = \
    #         mefd.save_featurized_artifact(df_trainset, aggregated=True, subject=subject)

    return df_trainset


def make_training_data_with_analysis_sequences(fa, **kargs):
    import gc

    pred_type = kargs.get("pred_type", "FP")
    error_label = kargs.get("error_label", pred_type)
    correct_label = kargs.get("correct_label", "TP")

    subject = "seq_featurized"
    col_tid = kargs.get('col_tid', 'transcript_id')

    mefd = ModelEvaluationFileHandler(ErrorAnalyzer.eval_dir, separator='\t')
    ############################################################

    analysis_sequence_df = mefd.load_analysis_sequences(
        aggregated=True, error_label=error_label, correct_label=correct_label)  # Output is a Polars DataFrame
    print("[info] Columns in analysis_sequence_df:", analysis_sequence_df.columns)
    # ['gene_id', 'position', 'pred_type', 'score', 'splice_type', 'chrom', 'window_start', 'window_end', 'sequence']
  
    gene_feature_df = fa.retrieve_gene_features()

    # Augment analysis_sequence_df with 'gene_start' and 'gene_end' to verify consensus sequences
    columns_to_select = ['gene_id', 'start', 'end', 'strand']
    gene_feature_df_selected = gene_feature_df.select(columns_to_select).rename({'start': 'gene_start', 'end': 'gene_end'})
    analysis_sequence_df = join_and_remove_duplicates(analysis_sequence_df, gene_feature_df_selected, on=['gene_id'], how='left', verbose=1)

    # Verify consensus sequences
    print_emphasized("[i/o] Verifying consensus sequences ...")
    print("[info] Input columns + gene boundary:", list(analysis_sequence_df.columns))

    # Check if 'sequence' column exists before dropping it
    if 'sequence' in analysis_sequence_df.columns:
        drop_columns(analysis_sequence_df, ['sequence'])

    display_dataframe_in_chunks(analysis_sequence_df, title="Augmented analysis sequence data")
    
    # Verify consensus sequences with full gene sequences
    # sequence_df = verify_consensus_sequences(sequence_df, splice_type="donor", window_radius=5, adjust_position=True, verbose=1)
    # sequence_df = verify_consensus_sequences(sequence_df, splice_type="donor", window_radius=5, adjust_position=False, verbose=1)
    assert isinstance(analysis_sequence_df, pl.DataFrame), "Expected a Polars DataFrame."

    # Verify consensus sequences with analysis data
    print_emphasized("[info] Verifying consensus sequences by setting adjust_position=True ...")
    consensus_dfs = []
    target_columns = (
        ['gene_id', col_tid, 'position', 'splice_type', 'has_consensus']
        if col_tid in analysis_sequence_df.columns
        else ['gene_id', 'position', 'splice_type', 'has_consensus']
    )
    for splice_type in ['donor', 'acceptor']:
        print_with_indent(f"Verifying splice_type={splice_type} ...", indent_level=1)
        df_consensus, summary = \
            verify_consensus_sequences_via_analysis_data(
            analysis_sequence_df.filter(pl.col('splice_type') == splice_type), 
            splice_type=splice_type, window_radius=5, adjust_position=True, verbose=1)
        # NOTE: Do not overwrite analysis_sequence_df here
        print_emphasized(f"[info] Columns after verifying consensus sequences: {df_consensus.columns}")
        consensus_dfs.append(df_consensus.select(target_columns))
        # 'transcript_id' is not present

        visualize_consensus_ratios(
            summary, splice_type, 
            save_plot=True, 
            plot_file=f'consensus_ratios_{splice_type}.pdf', 
            plot_format='pdf', verbose=True)
    
        # print_emphasized("[info] Verifying consensus sequences by setting adjust_position=False ...")
        # verify_consensus_sequences_via_analysis_data(
        #     augmented_analysis_sequence_df.filter(pl.col('splice_type') == 'donor'), 
        #     splice_type="donor", window_radius=5, adjust_position=False, verbose=1)

    # Combine the resulting dataframes
    combined_consensus_df = pl.concat(consensus_dfs)

    # Convert to Pandas for joining
    combined_consensus_df = combined_consensus_df.to_pandas()
    
    ############################################################
    # Release memory held by analysis_sequence_df
    del analysis_sequence_df
    gc.collect()


    ############################################################
    print_emphasized("[i/o] Loading featurized dataset ...")
    df_trainset = \
        mefd.load_featurized_artifact(
            aggregated=True, subject=subject, 
            error_label=error_label, correct_label=correct_label)
    shape0 = df_trainset.shape

    if isinstance(df_trainset, pl.DataFrame):
        df_trainset = df_trainset.to_pandas()

    # Ensure the columns have the same data type
    df_trainset['gene_id'] = df_trainset['gene_id'].astype(str)
    combined_consensus_df['gene_id'] = combined_consensus_df['gene_id'].astype(str)

    # Join the combined dataframe to df_trainset
    on_cols = (
        ['gene_id', col_tid, 'position', 'splice_type'] 
        if col_tid in df_trainset.columns 
        else ['gene_id', 'position', 'splice_type']
    )
    df_trainset = df_trainset.merge(combined_consensus_df, on=on_cols, how='left')
    # NOTE: 'transcript_id' is not present in the augmented_analysis_sequence_df

    columns = display_feature_set(df_trainset, max_kmers=100)
    print(f"[info] Columns(df_trainset) after merging consensus_df (n={len(df_trainset.columns)}):", list(columns))
    print(f"[info] shape(df_trainset): {shape0} -> {df_trainset.shape}")

    df_trainset = \
        impute_missing_values(
            df_trainset, 
            imputation_map={'num_overlaps': 0, 'has_consensus': False}, 
            verbose=1)

    featurized_dataset_path = \
            mefd.save_featurized_dataset(
                df_trainset, aggregated=True, 
                error_label=error_label, correct_label=correct_label
            )
    print_emphasized(f"[info] Saved final featurized dataset to {featurized_dataset_path}")
    print_with_indent(f"Columns: {display_feature_set(df_trainset, max_kmers=100)}", indent_level=1)
    shape0 = df_trainset.shape

    # Test
    print_emphasized("[test] Testing the integrity of the final training set ...")
    df_trainset_prime = \
        mefd.load_featurized_dataset(aggregated=True, error_label=error_label, correct_label=correct_label)
    assert df_trainset_prime.shape == df_trainset.shape, "Shape mismatch detected."
    analysis_result = \
        analyze_data_labels(df_trainset_prime, label_col='label', verbose=2, handle_missing=None)

    return  df_trainset


def demo_make_training_set(gtf_file=None, **kargs): 

    pred_type = kargs.get('pred_type', "FP")
    error_label = kargs.get("error_label", pred_type)
    correct_label = kargs.get("correct_label", "TP")

    mefd = ModelEvaluationFileHandler(ErrorAnalyzer.eval_dir, separator='\t')
    
    # Extract features from analysis sequence data
    make_kmer_featurized_dataset(gtf_file=gtf_file, **kargs)

    df_trainset = \
        mefd.load_featurized_artifact(
            aggregated=True, subject=subject)

    shape0 = df_trainset.shape
    if isinstance(df_trainset, pl.DataFrame):
        df_trainset = df_trainset.to_pandas()

    print_emphasized(f"[info] Columns in df_trainset prior to incorporating features from various data sources:\n{df_trainset.columns}\n")
    # NOTE: ['gene_id', 'transcript_id', 'position', 'score', 'splice_type', 'chrom', 'strand', 
    #        '3mer_CTT', '3mer_CCC', '3mer_GTC', '3mer_AAA', '2mer_CA', '3mer_TCT', '3mer_CGA', '3mer_TCA', '3mer_AAT', ...]
    print_with_indent(f"shape(df_trainset): {shape0}", indent_level=1)
    count_unique_ids(df_trainset, col_tid=col_tid)

    # --- Test ----
    # Check the data integrity for gene_id and transcript_id
    df_abnormal = check_and_subset_invalid_transcript_ids(df_trainset, col_tid=col_tid, verbose=1)
    if not is_dataframe_empty(df_abnormal): 
        print("[test] Abnormal data in training data (df_trainset):")
        cols = ['gene_id', col_tid, 'position', 'score', 'splice_type', 'chrom', 'strand']
        display(subsample_dataframe(df_abnormal, columns=cols, num_rows=10, random=True))
        # raise ValueError("Abnormal data found in df_trainset.")
        df_trainset = \
            filter_and_validate_ids(df_trainset, col_tid=col_tid, verbose=1)

    fa = FeatureAnalyzer(gtf_file=gtf_file, overwrite=overwrite)

    # Gene-level features 
    output_path = os.path.join(ErrorAnalyzer.analysis_dir, f'genomic_gtf_feature_set.{format}')
    gene_fs_df = run_genomic_gtf_feature_extraction(gtf_file, output_file=output_path)
    print("[info] Columns in gnomic gtf-derived feature set:", list(gene_fs_df.columns))
    print_emphasized(f"[info] Saved gnomic gtf-derived feature set to {output_path}")
    # gene_fs_df is a Polars dataframe
    # NOTE: ['gene_id', 'transcript_id', 'transcript_length', 'start', 'end', 'strand', 'gene_type', 
    #        'gene_length', 'chrom', 'num_exons', 'avg_exon_length', 'median_exon_length', 'total_exon_length']

    df_abnormal = check_and_subset_invalid_transcript_ids(gene_fs_df, col_gid='gene_id', col_tid='transcript_id', verbose=1)
    if not is_dataframe_empty(df_abnormal):
        print("[test] Abnormal data in gene_fs_df:")
        cols = ['gene_id', 'transcript_id', 'start', 'end', 'strand', 'gene_type', 'chrom',]
        display(subsample_dataframe(df_abnormal, columns=cols, num_rows=10, random=True))
        raise ValueError("Abnormal data found in gene_fs_df.")
    # NOTE: GTF-derived dataframe shouldn't have abnormal data

    on_columns = ['gene_id', col_tid]
    # --- Test ----
    # Check for duplicate rows in df_trainset
    num_duplicates_trainset = check_duplicates(df_trainset, subset=on_columns, verbose=2, example_limit=5)
    print(f"[diagnostics] Number of duplicate rows in df_trainset based on {on_columns}: {num_duplicates_trainset}")

    # Check for duplicate rows in gene_fs_df
    num_duplicates_gene_fs = check_duplicates(gene_fs_df, subset=on_columns, verbose=2, example_limit=5)
    print(f"[diagnostics] Number of duplicate rows in gene_fs_df based on {on_columns}: {num_duplicates_gene_fs}")
    # -------------

    # Join the dataframes and then remove duplicate columns
    print_emphasized(f"[info] Incorporating new variables from GTF-derived gene-level feature set (type={type(gene_fs_df)}) ...")
    
    df_trainset = join_and_remove_duplicates(df_trainset, gene_fs_df, on=on_columns, how='left', verbose=1)
    
    columns = display_feature_set(df_trainset, max_kmers=100)
    print(f"[info] Columns in updated df_trainset (n={len(df_trainset.columns)}:", list(columns))
    print(f"[info] shape(df_trainset) after including gene features: {shape0} -> {df_trainset.shape}")
    
    df_missing = find_missing_combinations(df_trainset, gene_fs_df, on_columns)
    print("[test] Missing combinations in df_trainset but not in gene_fs_df:")
    # Number of missing combintations
    print(f"[test] shape(df_missing): {df_missing.shape}")
    display(df_missing.head(10))
    
    assert shape0[0] == df_trainset.shape[0], "Assertion failed: Number of rows changed after joining gene features."
    # ---------------------------
    shape0 = df_trainset.shape
    
    # Total intron lengths and exon lengths
    print_emphasized(f"[info] Incorporating exon-intron length features via length_df ...")
    output_path = os.path.join(fa.analysis_dir, 'total_intron_exon_lengths.tsv')
    if os.path.exists(output_path):
        length_df = pd.read_csv(output_path, sep='\t')
    else: 
        length_df = compute_total_lengths(gtf_file)
        path_length_features = fa.save_dataframe(length_df, 'total_intron_exon_lengths.tsv')
        assert output_path == path_length_features, f"File path mismatch: {output_path} != {path_length_features}"
    # NOTE: ['transcript_id', 'total_exon_length', 'total_intron_length']
    print(f"[i/o] Path(length_df): {output_path}")
    print(f"[test] type(length_df): {type(length_df)})")
    print(f"[test] columns(length_df): {list(length_df.columns)}")
    
    on_columns = [col_tid, ]
    df_trainset = join_and_remove_duplicates(df_trainset, length_df, on=on_columns, how='left', verbose=1)
    
    columns = display_feature_set(df_trainset, max_kmers=100)
    print(f"[info] Columns in updated df_trainset (n={len(df_trainset.columns)}:", list(columns))
    print(f"[info] shape(df_trainset): {shape0} -> {df_trainset.shape}")
    assert shape0[0] == df_trainset.shape[0], "Assertion failed: Number of rows changed after joining exon-intron lengths."
    # ---------------------------
    shape0 = df_trainset.shape

    # Extract features from perfomance profile
    print_emphasized("[i/o] Extracting performance-profile features ...")
    # splice_stats_features = extract_gene_features_from_performance_profile()
    splice_stats_features = fa.retrieve_gene_level_performance_features()
    print("[test] columns(splice_stats_features):", list(splice_stats_features.columns))
    # NOTE: ['gene_id', 'n_splice_sites', 'splice_type']
    
    print_emphasized(f"[info] Incorporating new feature set: splice_stats_features (type={type(splice_sites_df)}) ...")
    on_columns = ['gene_id', 'splice_type']
    df_trainset = join_and_remove_duplicates(df_trainset, splice_stats_features, on=on_columns, how='left', verbose=1)
    
    columns = display_feature_set(df_trainset, max_kmers=100)
    print(f"[info] Columns in updated df_trainset (n={len(df_trainset.columns)}):", list(columns))
    print(f"[info] shape(df_trainset): {shape0} -> {df_trainset.shape}")
    assert shape0[0] == df_trainset.shape[0], "Assertion failed: Number of rows changed after joining performance features."
    # ---------------------------
    shape0 = df_trainset.shape
    
    # Overlapping genes
    print_emphasized("[info] Incorporating overlapping gene-specific features ...")
    sa = SpliceAnalyzer()
    overlapping_genes_df = sa.retrieve_overlapping_gene_metadata(output_format='dataframe', to_pandas=True)
    print("[test] columns(overlapping_genes_df):", list(overlapping_genes_df.columns))
    print_with_indent(f"Is polars? {type(overlapping_genes_df)}", indent_level=1)  # Yes, it is a Polars DataFrame

    columns = ['gene_id_1', 'num_overlaps']
    overlapping_genes_df = overlapping_genes_df[columns].rename(columns={'gene_id_1': 'gene_id'})
    print("[test] columns(overlapping_genes_df):", list(overlapping_genes_df.columns))
    print("[test] shape(overlapping_genes_df): ", overlapping_genes_df.shape)  # (6317, 2)

    # Test: Check for duplicate rows in terms of gene_id
    conflicting_counts = check_conflicting_overlaps(overlapping_genes_df)
    if conflicting_counts is not None:
        display(conflicting_counts)
        raise ValueError("Conflicting overlaps detected.")

    # De-duplicate overlapping_genes_df on gene_id
    # overlapping_genes_df = overlapping_genes_df.unique(subset=['gene_id'])
    overlapping_genes_df = overlapping_genes_df.drop_duplicates(subset=['gene_id'])

    on_columns = ['gene_id', ]

    # Test
    # Known issue with .join(): 
    # - You are trying to merge on object and int64 columns for key 'gene_id'. If you wish to proceed you should use pd.concat
    # df_trainset['gene_id'] = df_trainset['gene_id'].astype(str)
    # overlapping_genes_df['gene_id'] = overlapping_genes_df['gene_id'].astype(str)
    # print("Type of df_trainset['gene_id']:", df_trainset['gene_id'].dtype)
    # print("> example values: ", df_trainset['gene_id'].head(5))
    # print(df_trainset['gene_id'].unique())
    # print("Type of overlapping_genes_df['gene_id']:", overlapping_genes_df['gene_id'].dtype)
    # print("> example values:", overlapping_genes_df['gene_id'].head(5))
    # print(overlapping_genes_df['gene_id'].unique())

    # Perform a left join and fill missing values with 0 for num_overlaps
    df_trainset = join_and_remove_duplicates(df_trainset, overlapping_genes_df, on=on_columns, how='left', verbose=1)
    # df_trainset = df_trainset.merge(overlapping_genes_df, on=on_columns, how='left').fillna(0)
    
    columns = display_feature_set(df_trainset, max_kmers=100)
    print(f"[info] Columns in updated df_trainset (n={len(df_trainset.columns)}):", list(columns))
    print(f"[info] shape(df_trainset): {shape0} -> {df_trainset.shape}")
    assert shape0[0] == df_trainset.shape[0], "Assertion failed: Number of rows changed after joining overlapping genes."
    # ---------------------------
    shape0 = df_trainset.shape

    # Distance to transcript boundaries
    # distance_df = compute_splice_site_distances(splice_sites_df, transcript_feature_df, match_col='position')
    # print("[info] Columns in distance_df:", list(distance_df.columns))
    # display_dataframe_in_chunks(distance_df, title="Distance to transcript boundaries")
    # NOTE: ['gene_id', 'transcript_id', 'position', 'splice_type', 'distance_to_start', 'distance_to_end']

    display_dataframe_in_chunks(df_trainset, title="Training data before incorporating distance features")
    
    # print_emphasized("[info] Incorporating new feature set: distance_df ...")
    # on_columns = ['gene_id', 'transcript_id', 'splice_type', ]
    # Drop position column
    # distance_df = distance_df.drop(columns=['position'])

    # df_trainset = join_and_remove_duplicates(df_trainset, distance_df, on=on_columns, how='left', verbose=1)
    # df_trainset = df_trainset.merge(distance_df, on=on_columns, how='left').fillna(0)

    # display_dataframe_in_chunks(df_trainset, title="Training data after incorporating distance features")
    
    # Compute distance features directly on the training set
    shape0 = df_trainset.shape
    # df_trainset = compute_distances_to_transcript_boundaries(df_trainset, match_col='position')

    # Compute distances with strand adjustment
    print_emphasized("[info] Computing distances with strand adjustment ...")
    df_trainset = compute_distances_with_strand_adjustment(df_trainset, match_col='position', col_tid=col_tid)
    # This function assumes that 'position' is a relative position within the gene

    columns = display_feature_set(df_trainset, max_kmers=100)
    print(f"[info] Columns in updated df_trainset (n={len(df_trainset.columns)}):", list(columns))
    print(f"[info] shape(df_trainset): {shape0} -> {df_trainset.shape}")
    assert shape0[0] == df_trainset.shape[0], "Assertion failed: Number of rows changed after computing distances with strand adjustment."
    
    print_section_separator()
    shape0 = df_trainset.shape
    # ---------------------------

    # Augment analysis_sequence_df with 'gene_start' and 'gene_end' to verify consensus sequences
    columns_to_select = ['gene_id', 'start', 'end', 'strand']
    gene_feature_df_selected = gene_feature_df.select(columns_to_select).rename({'start': 'gene_start', 'end': 'gene_end'})
    analysis_sequence_df = join_and_remove_duplicates(analysis_sequence_df, gene_feature_df_selected, on=['gene_id'], how='left', verbose=1)

    # Perform the join operation
    # augmented_analysis_sequence_df = analysis_sequence_df.join(
    #     gene_feature_df_selected,
    #     on='gene_id',
    #     how='inner',
    #     suffix='_gene'
    # )
    # Drop redundant columns if they exist
    # redundant_columns = [col for col in augmented_analysis_sequence_df.columns if col.endswith('_gene')]
    # augmented_analysis_sequence_df = augmented_analysis_sequence_df.drop(redundant_columns)

    # Verify consensus sequences
    print_emphasized("[i/o] Verifying consensus sequences ...")
    print("[info] Input columns + gene boundary:", list(analysis_sequence_df.columns))

    # Check if 'sequence' column exists before dropping it
    if 'sequence' in analysis_sequence_df.columns:
        drop_columns(analysis_sequence_df, ['sequence'])

    display_dataframe_in_chunks(analysis_sequence_df, title="Augmented analysis sequence data")
    
    # Verify consensus sequences with full gene sequences
    # sequence_df = verify_consensus_sequences(sequence_df, splice_type="donor", window_radius=5, adjust_position=True, verbose=1)
    # sequence_df = verify_consensus_sequences(sequence_df, splice_type="donor", window_radius=5, adjust_position=False, verbose=1)
    assert isinstance(analysis_sequence_df, pl.DataFrame), "Expected a Polars DataFrame."

    # Verify consensus sequences with analysis data
    print_emphasized("[info] Verifying consensus sequences by setting adjust_position=True ...")
    consensus_dfs = []
    target_columns = (
        ['gene_id', col_tid, 'position', 'splice_type', 'has_consensus']
        if col_tid in analysis_sequence_df.columns
        else ['gene_id', 'position', 'splice_type', 'has_consensus']
    )
    for splice_type in ['donor', 'acceptor']:
        print_with_indent(f"Verifying splice_type={splice_type} ...", indent_level=1)
        df_consensus, summary = \
            verify_consensus_sequences_via_analysis_data(
            analysis_sequence_df.filter(pl.col('splice_type') == splice_type), 
            splice_type=splice_type, window_radius=5, adjust_position=True, verbose=1)
        # NOTE: Do not overwrite analysis_sequence_df here
        print_emphasized(f"[info] Columns after verifying consensus sequences: {df_consensus.columns}")
        consensus_dfs.append(df_consensus.select(target_columns))
        # 'transcript_id' is not present

        visualize_consensus_ratios(
            summary, splice_type, 
            save_plot=True, 
            plot_file=f'consensus_ratios_{splice_type}.pdf', 
            plot_format='pdf', verbose=True)
    
        # print_emphasized("[info] Verifying consensus sequences by setting adjust_position=False ...")
        # verify_consensus_sequences_via_analysis_data(
        #     augmented_analysis_sequence_df.filter(pl.col('splice_type') == 'donor'), 
        #     splice_type="donor", window_radius=5, adjust_position=False, verbose=1)

    # Combine the resulting dataframes
    combined_consensus_df = pl.concat(consensus_dfs)

    # Convert to Pandas for joining
    combined_consensus_df = combined_consensus_df.to_pandas()

    # Ensure the columns have the same data type
    df_trainset['gene_id'] = df_trainset['gene_id'].astype(str)
    combined_consensus_df['gene_id'] = combined_consensus_df['gene_id'].astype(str)

    # Join the combined dataframe to df_trainset
    on_cols = (
        ['gene_id', col_tid, 'position', 'splice_type'] 
        if col_tid in df_trainset.columns 
        else ['gene_id', 'position', 'splice_type']
    )
    df_trainset = df_trainset.merge(combined_consensus_df, on=on_cols, how='left')
    # NOTE: 'transcript_id' is not present in the augmented_analysis_sequence_df

    columns = display_feature_set(df_trainset, max_kmers=100)
    print(f"[info] Columns(df_trainset) after merging consensus_df (n={len(df_trainset.columns)}):", list(columns))
    print(f"[info] shape(df_trainset): {shape0} -> {df_trainset.shape}")

    df_trainset = \
        impute_missing_values(
            df_trainset, 
            imputation_map={'num_overlaps': 0, 'has_consensus': False}, 
            verbose=1)

    featurized_dataset_path = \
            mefd.save_featurized_dataset(
                df_trainset, aggregated=True, 
                error_label=error_label, correct_label=correct_label
            )
    print_emphasized(f"[info] Saved final featurized dataset to {featurized_dataset_path}")
    print_with_indent(f"Columns: {display_feature_set(df_trainset, max_kmers=100)}", indent_level=1)
    shape0 = df_trainset.shape

    # Test
    print_emphasized("[test] Testing the integrity of the final training set ...")
    df_trainset_prime = \
        mefd.load_featurized_dataset(aggregated=True, error_label=error_label, correct_label=correct_label)
    assert df_trainset_prime.shape == df_trainset.shape, "Shape mismatch detected."
    analysis_result = \
        analyze_data_labels(df_trainset_prime, label_col='label', verbose=2, handle_missing=None)


def visualize_consensus_ratios(summary_df, splice_type, save_plot=True, plot_file='consensus_ratios.pdf', plot_format='pdf', verbose=True, **kargs):
    """
    Visualize the consensus sequence analysis summary as a bar plot.

    Parameters:
    - summary_df (pd.DataFrame): Summary DataFrame with columns ['pred_type', 'sum', 'count', 'ratio'].
    - splice_type (str): Splice site type ('donor' or 'acceptor') for plot title.

    Returns:
    - None: Displays the bar plot.
    """
    import seaborn as sns
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 6))
    ax = sns.barplot(
        data=summary_df,
        x="ratio",
        y="pred_type",
        palette="viridis",
        orient="h"
    )
    for idx, row in summary_df.iterrows():
        ax.text(
            row["ratio"] + 0.02, idx,
            f"{int(row['sum'])}/{int(row['count'])}",
            color="black", va="center"
        )
    plt.title(f"Consensus Sequence Detection by Prediction Type ({splice_type.title()})")
    plt.xlabel("Ratio of Consensus Sequences Detected")
    plt.ylabel("Prediction Type")
    plt.xlim(0, 1)
    plt.tight_layout()
    
    if save_plot:
        output_dir = kargs.get('output_dir', ErrorAnalyzer.analysis_dir)
        output_path = os.path.join(output_dir, plot_file)
        plt.savefig(output_path, format=plot_format)
        if verbose: 
            print(f"Plot saved as {output_path}")
    else:
        plt.show()


def workflow_training_data_generation(strategy='average', threshold=0.5, kmer_sizes=[6, ], **kargs):
    from itertools import product

    custom_genes = kargs.get('custom_genes', ['ENSG00000104435', 'ENSG00000130477'])  # STMN2, UNC13A
    
    n_genes = kargs.get('n_genes', 1000)
    subset_genes = kargs.get('subset_genes', True)
    subset_policy = kargs.get('subset_policy', 'hard')  # 'random', 'top', 'hard', ...

    # Define possible labels
    error_labels = ["FP", "FN"]   # ["FN"]  # ["FP", "FN"]  
    correct_labels = ["TP", "TN"]  # ["TN"]  # ["TP", "TN"]  

    # Iterate through all combinations, excluding (FP, TN)
    for error_label, correct_label in product(error_labels, correct_labels):
        if (error_label, correct_label) == ("FP", "TN"):
            continue  # Skip the unwanted combination

        run_training_data_generation_workflow(
            gtf_file=None, 

            pred_type=error_label,
            error_label=error_label,
            correct_label=correct_label,

            kmer_sizes=kmer_sizes,
            
            subset_genes=subset_genes, 
            subset_policy=subset_policy,  # 'random', 'top', 'hard', ...
            n_genes=n_genes,

            # In addition to the default, susbampled gene set, we can also specify and include custom genes
            custom_genes=custom_genes
        )
        print_section_separator()
    

def test(): 
    from itertools import product

    # analyze_performance(sort_by='FP', N=20)
    # demo_analyze_error_profile()

    # Analyze splice site data
    # demo_analyze_splice_sites()  # data/ensembl/splice_sites.tsv

    # demo_misc()

    strategy = 'average'
    threshold = 0.5
    # demo_make_training_set(pred_type='FP', strategy=strategy, threshold=threshold, kmer_sizes=[6, ])
    # print_section_separator()
    # demo_make_training_set(pred_type='FN', strategy=strategy, threshold=threshold, kmer_sizes=[6, ])

    # Define possible labels
    error_labels = ["FP", "FN"]   # ["FN"]  # ["FP", "FN"]  
    correct_labels = ["TP", "TN"]  # ["TN"]  # ["TP", "TN"]  

    # Iterate through all combinations, excluding (FP, TN)
    for error_label, correct_label in product(error_labels, correct_labels):
        if (error_label, correct_label) == ("FP", "TN"):
            continue  # Skip the unwanted combination

        run_training_data_generation_workflow(
            gtf_file=None, 

            pred_type=error_label,
            error_label=error_label,
            correct_label=correct_label,

            kmer_sizes=[3, ],
            subset_genes=True, 
            subset_policy='hard',  # 'random', 'top', 'hard', ...
            n_genes=1000,
            custom_genes=['ENSG00000104435', 'ENSG00000130477']  # STMN2, UNC13A
        )
        print_section_separator()



if __name__ == "__main__":
    test()







