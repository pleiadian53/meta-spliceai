import os
import pandas as pd
import polars as pl

from .utils_doc import (
    print_emphasized, 
    print_with_indent, 
    print_section_separator
)

class ModelEvaluationFileHandler(object):

    # Define the expected schema for performance dataframe
    performance_df_expected_schema = {
        "precision": pl.Float64,
        "recall": pl.Float64,
        "f1_score": pl.Float64,
        "specificity": pl.Float64,
        "fpr": pl.Float64,
        "fnr": pl.Float64,
        "chrom": pl.Utf8,
        "gene_id": pl.Utf8,
        # Add other columns as needed
    }

    def __init__(self, output_dir, separator='\t', **kargs):
        """
        Initialize the ModelEvaluationFileHandler.

        Parameters:
        - output_dir (str): The directory to save the files in.
        - separator (str): The separator to use in the files (default is '\t').
        """
        self.output_dir = output_dir
        self.separator = separator
        self.file_extension = self._determine_file_extension(separator)
        self.verbose = kargs.get('verbose', 1)
        self.pred_type = kargs.get('pred_type', None)
        self.error_label = kargs.get('error_label', self.pred_type)
        self.correct_label = kargs.get('correct_label', None)
        self.splice_type = kargs.get('splice_type', None)

    def _determine_file_extension(self, separator):
        """
        Determine the file extension based on the separator.

        Parameters:
        - separator (str): The separator to use in the file.

        Returns:
        - str: The file extension.
        """
        if separator == '\t':
            return 'tsv'
        elif separator == ',':
            return 'csv'
        else:
            raise ValueError("Unsupported separator. Use '\\t' for TSV or ',' for CSV.")

    def get_performance_file_path(
            self, 
            chr=None, chunk_start=None, chunk_end=None, 
            aggregated=False, performance_file=None, subject=None, test=False):
        """
        Get the file path for the performance DataFrame.

        Parameters:
        - chr (str): The chromosome identifier (optional for aggregated DataFrame).
        - chunk_start (int): The start index of the chunk (optional for aggregated DataFrame).
        - chunk_end (int): The end index of the chunk (optional for aggregated DataFrame).
        - aggregated (bool): If True, get the path for an aggregated file without chunk_start and chunk_end.

        Returns:
        - str: The file path.
        """
        if chunk_start is None and chunk_end is None:
            aggregated = True

        if subject is None: 
            subject = "splice_performance"
        if test: 
            subject = f"{subject}_test"

        if aggregated:
            if performance_file is None: 
                performance_file = f"full_{subject}.{self.file_extension}"
        else:
            if chr is None or chunk_start is None or chunk_end is None:
                raise ValueError("chr, chunk_start, and chunk_end must be provided for non-aggregated DataFrame")
            performance_file = f"{subject}_{chr}_chunk_{chunk_start}_{chunk_end}.{self.file_extension}"
        
        return os.path.join(self.output_dir, performance_file)

    def get_performance_meta_data_path(self, chr=None, chunk_start=None, chunk_end=None, aggregated=True, subject=None, test=False):
        if chunk_start is None and chunk_end is None:
            aggregated = True

        if subject is None: 
            subject = "prediction_delta"
        if test: 
            subject = f"{subject}_test"

        if aggregated: 
            prediction_delta_file = f"full_{subject}.{self.file_extension}"
        else:
            if chr is None or chunk_start is None or chunk_end is None:
                raise ValueError("chr, chunk_start, and chunk_end must be provided for non-aggregated DataFrame")
            prediction_delta_file = f"{subject}_{chr}_chunk_{chunk_start}_{chunk_end}.{self.file_extension}"
        return os.path.join(self.output_dir, prediction_delta_file)

    def save_performance_analysis(self, evaluation_df, chr=None, chunk_start=None, chunk_end=None, aggregated=False):
        return self.save_performance_df(evaluation_df, chr, chunk_start, chunk_end, aggregated, subject="splice_performance")

    def save_performance_df(
            self, 
            performance_df, 
            chr=None, 
            chunk_start=None, chunk_end=None, 
            aggregated=False, 
            performance_file=None, subject="splice_performance", test=False):
        """
        Save the performance DataFrame to a file with the specified separator.

        Parameters:
        - performance_df (pl.DataFrame): The DataFrame to save.
        - chr (str): The chromosome identifier (optional for aggregated DataFrame).
        - chunk_start (int): The start index of the chunk (optional for aggregated DataFrame).
        - chunk_end (int): The end index of the chunk (optional for aggregated DataFrame).
        - aggregated (bool): If True, save as an aggregated file without chunk_start and chunk_end.
        """
        performance_path = \
            self.get_performance_file_path(
                chr, chunk_start, chunk_end, aggregated, performance_file=performance_file, subject=subject, test=test)

        # Ensure the directory exists
        os.makedirs(os.path.dirname(performance_path), exist_ok=True)

        # Save the DataFrame to the file
        performance_df.write_csv(performance_path, separator=self.separator)

        print(f"[i/o] Performance saved to: {performance_path}")

        return performance_path

    def save_performance_meta(
            self, 
            meta_df, 
            chr=None, chunk_start=None, chunk_end=None, 
            aggregated=True, 
            subject="prediction_delta", test=False):
        performance_meta_file_path = self.get_performance_meta_data_path(
            chr, chunk_start, chunk_end, aggregated, subject=subject, test=test)
        os.makedirs(os.path.dirname(performance_meta_file_path), exist_ok=True)
        print_with_indent(f"[i/o] Saving performance meta data to (subject={subject}): {performance_meta_file_path}", indent_level=1)

        meta_df.write_csv(performance_meta_file_path, separator=self.separator)

    def load_performance_df(
            self, 
            chr=None, 
            chunk_start=None, chunk_end=None, 
            aggregated=False, 
            performance_file=None, 
            test=False,
            to_pandas=False, **kargs):
        """
        Load the performance DataFrame from a file with the specified separator.

        Parameters:
        - chr (str): The chromosome identifier (optional for aggregated DataFrame).
        - chunk_start (int): The start index of the chunk (optional for aggregated DataFrame).
        - chunk_end (int): The end index of the chunk (optional for aggregated DataFrame).
        - aggregated (bool): If True, load from an aggregated file without chunk_start and chunk_end.

        Returns:
        - pl.DataFrame: The loaded DataFrame.
        """
        performance_path = \
            self.get_performance_file_path(chr, chunk_start, chunk_end, aggregated, performance_file=performance_file, test=test)

        print(f"[i/o] Loading performance from: {performance_path}")

        schema = kargs.get('schema', self.performance_df_expected_schema)

        # Load the DataFrame from the file
        performance_df = pl.read_csv(performance_path, separator=self.separator, schema_overrides=schema)

        if to_pandas:
            return performance_df.to_pandas()

        return performance_df

    def load_performance_meta(
            self,
            chr=None,
            chunk_start=None,
            chunk_end=None,
            aggregated=True,
            test=False,
            subject="prediction_delta"):
        performance_meta_file_path = self.get_performance_meta_data_path(chr, chunk_start, chunk_end, aggregated, subject=subject)
        print(f"[i/o] Loading performance meta data from (subject={subject}): {performance_meta_file_path}")

        schema = kargs.get('schema', self.performance_df_expected_schema)
        
        performance_meta_df = pl.read_csv(performance_meta_file_path, separator=self.separator, schema_overrides=schema)
        
        return performance_meta_df

    def get_error_analysis_file_path(
            self, chr=None, chunk_start=None, chunk_end=None, 
            aggregated=False, subject=None, test=False):
        """
        Get the file path for the error analysis DataFrame.

        Parameters:
        - chr (str): The chromosome identifier (optional for aggregated DataFrame).
        - chunk_start (int): The start index of the chunk (optional for aggregated DataFrame).
        - chunk_end (int): The end index of the chunk (optional for aggregated DataFrame).
        - aggregated (bool): If True, get the path for an aggregated file without chunk_start and chunk_end.

        Returns:
        - str: The file path.
        """
        if subject is None: 
            subject = "splice_errors"
        if test: 
            subject = f"{subject}_test"
        if chunk_start is None and chunk_end is None:
            aggregated = True

        if aggregated:
            error_analysis_file = f"full_{subject}.{self.file_extension}"
        else:
            if chr is None or chunk_start is None or chunk_end is None:
                raise ValueError("chr, chunk_start, and chunk_end must be provided for non-aggregated DataFrame")
            error_analysis_file = f"{subject}_{chr}_chunk_{chunk_start}_{chunk_end}.{self.file_extension}"
        
        return os.path.join(self.output_dir, error_analysis_file)

    def get_splice_positions_file_path(self, chr=None, chunk_start=None, chunk_end=None, aggregated=True, subject="splice_positions"):
        if chunk_start is None and chunk_end is None:
            aggregated = True
        
        if aggregated: 
            splice_positions_file = f"full_{subject}.{self.file_extension}"
        else:
            if chr is None or chunk_start is None or chunk_end is None:
                raise ValueError("chr, chunk_start, and chunk_end must be provided for non-aggregated DataFrame")
            splice_positions_file = f"{subject}_{chr}_chunk_{chunk_start}_{chunk_end}.{self.file_extension}"
        return os.path.join(self.output_dir, splice_positions_file)

    def get_tp_data_file_path(self, chr=None, chunk_start=None, chunk_end=None, aggregated=True, subject="splice_tp"):
        if aggregated: 
            tp_data_file = f"full_{subject}.{self.file_extension}"
        else:
            if chr is None or chunk_start is None or chunk_end is None:
                raise ValueError("chr, chunk_start, and chunk_end must be provided for non-aggregated DataFrame")
            tp_data_file = f"{subject}_{chr}_chunk_{chunk_start}_{chunk_end}.{self.file_extension}"
        return os.path.join(self.output_dir, tp_data_file)

    def save_tp_data_points(self, tp_data, chr=None, chunk_start=None, chunk_end=None, aggregated=True, subject="splice_tp"):
        tp_data_path = self.get_tp_data_file_path(chr, chunk_start, chunk_end, aggregated, subject=subject)
        os.makedirs(os.path.dirname(tp_data_path), exist_ok=True)
        # tp_data.write_csv(tp_data_path, separator=self.separator)
        print_with_indent(f"[i/o] Saving true positive data from: {tp_data_path}", indent_level=1)
        
        self.save_dataframe(tp_data, tp_data_path, separator=self.separator)
        print(f"[i/o] Saved true positive data to: {tp_data_path}")

    def load_tp_data_points(self, chr=None, chunk_start=None, chunk_end=None, aggregated=True, subject="splice_tp"):
        tp_data_path = self.get_tp_data_file_path(chr, chunk_start, chunk_end, aggregated, subject=subject)

        print(f"[i/o] Loading true positive data from: {tp_data_path}")
        tp_data = pl.read_csv(tp_data_path, separator=self.separator)
        return tp_data

    def save_error_sequences(
            self, 
            error_df, 
            chr=None, chunk_start=None, chunk_end=None, 
            aggregated=False, subject="error_sequences", standardize_columns=True): 
        
        if standardize_columns: 
            error_df = self.reorder_columns(error_df)
        
        return self.save_error_analysis_df(error_df, chr, chunk_start, chunk_end, aggregated, subject=subject)

    def save_analysis_sequences(
            self, analysis_df, 
            chr=None, chunk_start=None, chunk_end=None, 
            aggregated=False, subject="analysis_sequences", standardize_columns=True, **kargs):

        subject = self.parametrize_subject(subject, **kargs)

        if standardize_columns: 
            analysis_df = self.reorder_columns(analysis_df)
        
        return self.save_error_analysis_df(analysis_df, chr, chunk_start, chunk_end, aggregated, subject=subject)

    @staticmethod
    def reorder_columns(df, first_columns=['gene_id'], last_columns=['sequence']):
        """
        Reorder columns in the DataFrame such that specified columns come first and last.

        Parameters:
        - df (pl.DataFrame): The DataFrame to reorder.
        - first_columns (list): List of columns to place first.
        - last_columns (list): List of columns to place last.

        Returns:
        - pl.DataFrame: The DataFrame with reordered columns.
        """
        columns = df.columns
        if 'transcript_id' in columns and 'transcript_id' not in first_columns:
            first_columns.append('transcript_id')
        middle_columns = [col for col in columns if col not in first_columns + last_columns]
        ordered_columns = first_columns + middle_columns + last_columns
        return df.select(ordered_columns)

    # Todo: Refactor to FeatureAnalyzer? 
    def save_featurized_dataset(
            self, 
            featurized_df, 
            chr=None, chunk_start=None, chunk_end=None, 
            aggregated=False, 
            separator=None, subject="featurized_dataset", **kargs):

        subject = self.parametrize_subject(subject, **kargs)

        return self.save_error_analysis_df(
            featurized_df, chr, chunk_start, chunk_end, aggregated, subject=subject, separator=separator)

    def save_featurized_artifact(
            self,
            featurized_df,
            chr=None, chunk_start=None, chunk_end=None, 
            aggregated=False, separator=None, subject="featurized_artifact", **kargs):

        subject = self.parametrize_subject(subject, **kargs)

        return self.save_error_analysis_df(
            featurized_df, chr, chunk_start, chunk_end, aggregated, subject=subject, separator=separator)

    def save_error_analysis_df(
            self, error_df, chr=None, chunk_start=None, chunk_end=None, 
            aggregated=False, subject="splice_errors", separator=None, verbose=1):
        """
        Save the error analysis DataFrame to a file with the specified separator.

        Parameters:
        - error_df (pl.DataFrame): The DataFrame to save.
        - chr (str): The chromosome identifier (optional for aggregated DataFrame).
        - chunk_start (int): The start index of the chunk (optional for aggregated DataFrame).
        - chunk_end (int): The end index of the chunk (optional for aggregated DataFrame).
        - aggregated (bool): If True, save as an aggregated file without chunk_start and chunk_end.
        """
        error_analysis_path = self.get_error_analysis_file_path(chr, chunk_start, chunk_end, aggregated, subject=subject)

        # Ensure the directory exists
        os.makedirs(os.path.dirname(error_analysis_path), exist_ok=True)

        if self.verbose: 
            print_with_indent(f"[i/o] Saving analysis data (subject={subject}) to: {error_analysis_path}", indent_level=1)
            if aggregated: 
                print_with_indent(f"Columns in the DataFrame: {error_df.columns}", indent_level=2)

        # Save the DataFrame to the file
        # error_df.write_csv(error_analysis_path, separator=self.separator)
        if separator is None:
            separator = self.separator
        self.save_dataframe(error_df, error_analysis_path, separator=separator)

        return error_analysis_path

    def save_error_analysis(self, error_df, chr=None, chunk_start=None, chunk_end=None, aggregated=False, subject="splice_errors"):
        return self.save_error_analysis_df(error_df, chr, chunk_start, chunk_end, aggregated, subject=subject)

    def save_splice_positions(
            self, splice_positions_df, 
            chr=None, chunk_start=None, chunk_end=None, 
            aggregated=False, subject="splice_positions", **kwargs):
        
        splice_positions_path = self.get_splice_positions_file_path(chr, chunk_start, chunk_end, aggregated, subject=subject)
        os.makedirs(os.path.dirname(splice_positions_path), exist_ok=True)

        if self.verbose: 
            print_with_indent(f"[i/o] Saving splice positions to: {splice_positions_path}", indent_level=1)
            if aggregated: 
                print_with_indent(f"Columns in the DataFrame: {splice_positions_df.columns}", indent_level=2)

        # Standardize columns
        standardize_columns = kwargs.get('standardize_columns', True)
        first_columns = kwargs.get('first_columns', None)
        last_columns = kwargs.get('last_columns', None)
        
        if standardize_columns: 
            # Use provided column ordering or fall back to defaults
            first_cols = first_columns if first_columns is not None else ['gene_id']
            last_cols = last_columns if last_columns is not None else ['score', 'splice_type']
            
            # Check that columns actually exist in the DataFrame
            first_cols = [col for col in first_cols if col in splice_positions_df.columns]
            last_cols = [col for col in last_cols if col in splice_positions_df.columns]
            
            splice_positions_df = self.reorder_columns(
                splice_positions_df, 
                first_columns=first_cols, last_columns=last_cols)

        # splice_positions_df.write_csv(splice_positions_path, separator=self.separator)
        self.save_dataframe(splice_positions_df, splice_positions_path, separator=self.separator)

        return splice_positions_path

    def load_error_sequences(self, chr=None, chunk_start=None, chunk_end=None, aggregated=False, subject="error_sequences", **kargs): 
        if kargs.get('test', False):
            subject = f"{subject}_test"
        return self.load_error_analysis_df(chr, chunk_start, chunk_end, aggregated, subject=subject)

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

    def load_analysis_sequences(self, chr=None, chunk_start=None, chunk_end=None, aggregated=False, subject="analysis_sequences", **kargs):
        
        subject = self.parametrize_subject(subject, **kargs)
        splice_type = kargs.get('splice_type', "any")
        verbose = kargs.get('verbose', 0)

        df = self.load_error_analysis_df(chr, chunk_start, chunk_end, aggregated, subject=subject)

        # Filter by splice type
        df = filter_by_splice_type(df, splice_type, verbose=verbose)

        return df

    def get_featurized_dataset_path(
            self, 
            chr=None, chunk_start=None, chunk_end=None, 
            aggregated=False, subject="featurized_dataset", **kargs):

        subject = self.parametrize_subject(subject, **kargs)

        return self.get_error_analysis_file_path(chr, chunk_start, chunk_end, aggregated, subject=subject)
    
    @staticmethod
    def filter_by_splice_type(df, splice_type, verbose=0):
        return filter_by_splice_type(df, splice_type, verbose)

    def load_featurized_dataset(
            self, 
            chr=None, chunk_start=None, chunk_end=None, 
            aggregated=False, subject="featurized_dataset", **kargs):
        
        subject = self.parametrize_subject(subject, **kargs)
        splice_type = kargs.get('splice_type', self.splice_type)
        verbose = kargs.get('verbose', 0)

        df = self.load_error_analysis_df(chr, chunk_start, chunk_end, aggregated, subject=subject)
        
        # Filter by splice type
        df = filter_by_splice_type(df, splice_type, verbose=verbose)
        
        return df 

    def load_featurized_artifact(
            self, 
            chr=None, chunk_start=None, chunk_end=None, 
            aggregated=False, subject="featurized_artifact", **kargs):

        subject = self.parametrize_subject(subject, **kargs)
        splice_type = kargs.get('splice_type', self.splice_type)
        verbose = kargs.get('verbose', 0)

        df = self.load_error_analysis_df(chr, chunk_start, chunk_end, aggregated, subject=subject)
        
        # Filter by splice type
        df = filter_by_splice_type(df, splice_type, verbose=verbose)

        return df

    def load_error_analysis_df(
            self, chr=None, chunk_start=None, chunk_end=None, 
            aggregated=False, subject="splice_errors", **kargs):
        """
        Load the error analysis DataFrame from a file with the specified separator.

        Parameters:
        - chr (str): The chromosome identifier (optional for aggregated DataFrame).
        - chunk_start (int): The start index of the chunk (optional for aggregated DataFrame).
        - chunk_end (int): The end index of the chunk (optional for aggregated DataFrame).
        - aggregated (bool): If True, load from an aggregated file without chunk_start and chunk_end.

        Returns:
        - pl.DataFrame: The loaded DataFrame.
        """
        if chunk_start is None and chunk_end is None:
            aggregated = True

        error_analysis_path = self.get_error_analysis_file_path(chr, chunk_start, chunk_end, aggregated, subject=subject)
        print(f"[i/o] Loading analysis data (subject={subject}) from: {error_analysis_path}")

        # Define the schema for the columns (Todo - class attribute?)
        schema = kargs.get('schema', {
            'chrom': pl.Utf8,
            'error_type': pl.Utf8,
            'gene_id': pl.Utf8,
            'position': pl.Int64,
            'splice_type': pl.Utf8,
            'strand': pl.Utf8,
            'transcript_id': pl.Utf8,
            'window_end': pl.Int64,
            'window_start': pl.Int64
        })

        # Load the DataFrame from the file
        try:
            # First attempt: Try with schema_overrides
            error_df = pl.read_csv(error_analysis_path, separator=self.separator, schema_overrides=schema)
        except pl.exceptions.ComputeError as e:
            # If that fails due to type issues with k-mer columns
            print(f"Warning: Initial schema load failed, trying alternative approach: {str(e)}")
            
            # Identify core columns that must use our schema
            core_schema = schema.copy()
            
            # Try again with more flexible type inference on non-core columns
            error_df = pl.read_csv(
                error_analysis_path, 
                separator=self.separator, 
                schema_overrides=core_schema,
                infer_schema_length=10000,  # Better type inference
                try_parse_dates=False  # Prevent date parsing
            )

        return error_df

    def load_splice_positions(self, chr=None, chunk_start=None, chunk_end=None, aggregated=False, subject="splice_positions", **kargs):
        if chunk_start is None and chunk_end is None:
            aggregated = True

        splice_positions_path = self.get_splice_positions_file_path(chr, chunk_start, chunk_end, aggregated, subject=subject)
        print(f"[i/o] Loading splice positions from: {splice_positions_path}")
        print(f"[debug] self.output_dir: {self.output_dir}")

        schema = kargs.get('schema', {
            'chrom': pl.Utf8,
            'error_type': pl.Utf8,
            'gene_id': pl.Utf8,
            'position': pl.Int64,
            'splice_type': pl.Utf8,
            'strand': pl.Utf8,
            'transcript_id': pl.Utf8,
            'window_end': pl.Int64,
            'window_start': pl.Int64
        })
        
        splice_positions_df = pl.read_csv(splice_positions_path, separator=self.separator, schema_overrides=schema)
        
        return splice_positions_df

    def load_dataframe(self, file_path, separator=','):
        # Ensure the file_path is within self.output_dir if it doesn't have a preceding directory
        if not os.path.dirname(file_path):
            file_path = os.path.join(self.output_dir, file_path)

        if file_path.endswith('.csv'):
            try:
                return pl.read_csv(file_path, separator=separator)
            except Exception:
                return pd.read_csv(file_path, sep=separator)
        else:
            raise ValueError("Unsupported file format")

    def save_dataframe(self, df, file_path, separator=','):
        # Ensure the file_path is within self.output_dir if it doesn't have a preceding directory
        if not os.path.dirname(file_path):
            file_path = os.path.join(self.output_dir, file_path)

        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        if isinstance(df, pl.DataFrame):
            df.write_csv(file_path, separator=separator)
        elif isinstance(df, pd.DataFrame):
            df.to_csv(file_path, sep=separator, index=False)
        else:
            raise ValueError("Unsupported DataFrame type")

# Example usage
# handler = ModelEvaluationFileHandler(output_dir='/path/to/output')
# performance_df = pl.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]})
# handler.save_performance_df(performance_df, chr='chr1', chunk_start=0, chunk_end=1000)
# loaded_df = handler.load_performance_df(chr='chr1', chunk_start=0, chunk_end=1000)
# print(loaded_df)


def filter_by_splice_type(df, splice_type, verbose=0):
    if splice_type in ("donor", "acceptor", "neither"):
        if verbose: 
            print(f"[info] Filtering by splice type: {splice_type}")
        # Verify if the column exists
        if "splice_type" not in df.columns:
            raise ValueError(f"splice_type column not found in the DataFrame")
        before_count = df.shape[0]
        df = df.filter(pl.col("splice_type") == splice_type)
        after_count = df.shape[0]
        if verbose:
            print_with_indent(
                f"[info] Subset data by splice_type='{splice_type}' => {after_count}/{before_count} rows retained.", 
                indent_level=1)
    return df