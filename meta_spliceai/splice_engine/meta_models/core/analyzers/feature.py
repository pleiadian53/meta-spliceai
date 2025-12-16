"""
Feature analyzer for the splice-surveyor project.

This module provides specialized analyzers for genomic feature
extraction and analysis, including support for:
- Gene, transcript, and exon feature extraction
- Performance feature analysis
- Integrated caching and file management
"""

import os
import pandas as pd
import polars as pl
from typing import Union, Optional, Dict, List, Any, Tuple

from .base import Analyzer
from meta_spliceai.splice_engine.extract_genomic_features import (
    extract_gene_features_from_gtf,
    extract_transcript_features_from_gtf,
    summarize_exon_features_at_transcript_level,
    extract_exon_features_from_gtf,
    extract_gene_features_from_performance_profile
)

class FeatureAnalyzer(Analyzer):
    """
    Specialized analyzer for genomic feature extraction and analysis.
    
    This class extends the base Analyzer with specific functionality
    for working with gene, transcript, and exon features.
    
    Attributes
    ----------
    overwrite : bool
        Whether to overwrite existing feature files
    col_tid : str
        Column name for transcript IDs
    format : str
        Default file format ('tsv' or 'csv')
    separator : str
        Default separator for file operations
    """
    
    def __init__(self, 
                data_dir: Optional[str] = None, 
                *, 
                source: str = 'ensembl', 
                version: Optional[str] = None,
                gtf_file: Optional[str] = None,
                overwrite: bool = False,
                **kwargs):
        """
        Initialize a FeatureAnalyzer.
        
        Parameters
        ----------
        data_dir : str, optional
            Directory containing feature data, if None uses default
        source : str, default='ensembl'
            Data source identifier
        version : str, optional
            Version string for the data source
        gtf_file : str, optional
            Path to GTF file, if None uses class default
        overwrite : bool, default=False
            Whether to overwrite existing feature files
        **kwargs : dict
            Additional configuration parameters
        """
        super().__init__(gtf_file=gtf_file, **kwargs)
        self.source = source
        self.version = version
        self.data_dir = data_dir or Analyzer.data_dir
        self.gtf_file = gtf_file or Analyzer.gtf_file
        self.overwrite = overwrite

        # Configure file format parameters
        self.col_tid = kwargs.get("col_tid", "transcript_id")
        self.format = kwargs.get("format", 'tsv')
        self.seq_format = kwargs.get("seq_format", 'parquet')
        self.separator = kwargs.get("separator", ',' if self.format == 'csv' else '\t')

    @property
    def path_to_gene_features(self) -> str:
        """Get the path to the gene features file."""
        return os.path.join(self.get_path('analysis'), f'gene_features.tsv')

    @property
    def path_to_transcript_features(self) -> str:
        """Get the path to the transcript features file."""
        return os.path.join(self.get_path('analysis'), f'transcript_features.tsv')

    @property
    def path_to_exon_features(self) -> str:
        """Get the path to the exon features file."""
        return os.path.join(self.get_path('analysis'), f'exon_features.tsv')

    @property
    def path_to_performance_datafrane_derived_features(self) -> str:
        """Get the path to the performance dataframe derived features file."""
        return os.path.join(self.get_path('analysis'), f'performance_df_features.tsv')

    @property
    def path_to_transcript_df_from_gtf(self) -> str:
        """Get the path to the transcript dataframe from GTF file."""
        return os.path.join(self.get_path('analysis'), f'transcript_df_from_gtf.tsv')

    @property
    def path_to_exon_df_from_gtf(self) -> str:
        """Get the path to the exon dataframe from GTF file."""
        return os.path.join(self.get_path('analysis'), f'exon_df_from_gtf.tsv')

    def retrieve_gene_features(self, 
                             to_pandas: bool = False, 
                             verbose: int = 1) -> Union[pd.DataFrame, pl.DataFrame]:
        """
        Retrieve gene features from GTF file or cached file.
        
        Parameters
        ----------
        to_pandas : bool, default=False
            Whether to return a pandas DataFrame
        verbose : int, default=1
            Verbosity level for output messages
            
        Returns
        -------
        Union[pd.DataFrame, pl.DataFrame]
            DataFrame with gene features
        """
        if not self.overwrite and os.path.exists(self.path_to_gene_features):
            if verbose: 
                print(f"[i/o] Loading gene features from {self.path_to_gene_features}")

            sep = ',' if self.path_to_gene_features.endswith('.csv') else '\t'
            df = pl.read_csv(
                    self.path_to_gene_features, 
                    separator=sep, 
                    schema_overrides={'chrom': pl.Utf8}
                )
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
    
    def retrieve_transcript_features(self, 
                                   verbose: int = 1, 
                                   to_pandas: bool = False) -> Union[pd.DataFrame, pl.DataFrame]:
        """
        Retrieve transcript features from GTF file or cached file.
        
        Parameters
        ----------
        verbose : int, default=1
            Verbosity level for output messages
        to_pandas : bool, default=False
            Whether to return a pandas DataFrame
            
        Returns
        -------
        Union[pd.DataFrame, pl.DataFrame]
            DataFrame with transcript features
        """
        if not self.overwrite and os.path.exists(self.path_to_transcript_features):
            if verbose: 
                print(f"[i/o] Loading transcript features from {self.path_to_transcript_features}")

            sep = ',' if self.path_to_transcript_features.endswith('.csv') else '\t'
            df = pl.read_csv(
                    self.path_to_transcript_features, 
                    separator=sep, 
                    schema_overrides={'chrom': pl.Utf8}
                )
        else:
            if verbose: 
                if not self.overwrite: 
                    print(f"[warning] Could not find transcript features at:\n{self.path_to_transcript_features}\n")
                print(f"[action] Extracting transcript features from GTF file ...")

            df = extract_transcript_features_from_gtf(
                gtf_file_path=self.gtf_file, 
                output_file=self.path_to_transcript_features)

        if to_pandas:
            df = df.to_pandas()

        return df

    def retrieve_exon_features_at_transcript_level(self, 
                                                verbose: int = 1) -> pl.DataFrame:
        """
        Retrieve exon features at the transcript level.
        
        Parameters
        ----------
        verbose : int, default=1
            Verbosity level for output messages
            
        Returns
        -------
        pl.DataFrame
            DataFrame with exon features at transcript level
        """
        if not self.overwrite and os.path.exists(self.path_to_exon_features):
            if verbose: 
                print(f"[i/o] Loading exon features from {self.path_to_exon_features}")

            sep = ',' if self.path_to_exon_features.endswith('.csv') else '\t'
            return pl.read_csv(
                    self.path_to_exon_features, 
                    separator=sep, 
                    schema_overrides={'chromosome': pl.Utf8}
                )
        else:
            if verbose: 
                if not self.overwrite:
                    print(f"[warning] Could not find exon features at:\n{self.path_to_exon_features}\n")
                print(f"[action] Extracting exon features from GTF file ...")

            return summarize_exon_features_at_transcript_level(
                gtf_file_path=self.gtf_file, output_file=self.path_to_exon_features)

    def retrieve_exon_features(self, 
                             verbose: int = 1) -> pl.DataFrame:
        """
        Retrieve exon features (alias for retrieve_exon_features_at_transcript_level).
        
        Parameters
        ----------
        verbose : int, default=1
            Verbosity level for output messages
            
        Returns
        -------
        pl.DataFrame
            DataFrame with exon features
        """
        return self.retrieve_exon_features_at_transcript_level(verbose=verbose)

    def retrieve_exon_dataframe(self, 
                              verbose: int = 1, 
                              to_pandas: bool = True) -> Union[pd.DataFrame, pl.DataFrame]:
        """
        Retrieve the exon dataframe, with caching support.
        
        Uses the enhanced extract_exon_features_from_gtf function with proper parameter
        filtering based on the function signature.
        
        Parameters
        ----------
        verbose : int, default=1
            Verbosity level for output messages
        to_pandas : bool, default=True
            Whether to return a pandas DataFrame
            
        Returns
        -------
        Union[pd.DataFrame, pl.DataFrame]
            The exon DataFrame
        """
        return extract_exon_features_from_gtf(
            gtf_file_path=self.gtf_file,
            verbose=verbose,
            to_pandas=to_pandas,
            cache_file=self.path_to_exon_df_from_gtf,
            use_cache=not self.overwrite,
            save=True,
            include_extra_features=True,
            clean_output=True
        )

    def retrieve_gene_level_performance_features(self, 
                                              **kwargs) -> pd.DataFrame:
        """
        Retrieve gene-level performance features.
        
        Parameters
        ----------
        **kwargs : dict
            Additional parameters passed to the feature extraction function
            
        Returns
        -------
        pd.DataFrame
            DataFrame with gene-level performance features
        """
        output_file = self.path_to_performance_datafrane_derived_features
        verbose = kwargs.get('verbose', 1)

        # Determine the separator based on the file extension
        separator = ',' if output_file.endswith('.csv') else '\t'

        if not self.overwrite and os.path.exists(output_file):
            if verbose: 
                print(f"[i/o] Loading gene features from {output_file}")
            df = pd.read_csv(output_file, sep=separator)
        else: 
            df = extract_gene_features_from_performance_profile(
                eval_dir=self.get_path('eval'), 
                output_file=output_file, 
                separator='\t', 
                verbose=verbose) 
            # NOTE: performance dataframe is saved in '\t'

            # Save the resulting dataframe
            self.save_dataframe(df, file_name=output_file, sep=separator, verbose=verbose)

        return df

    def save_dataframe(self, 
                      df: Union[pd.DataFrame, pl.DataFrame], 
                      file_name: str, 
                      sep: str = '\t', 
                      verbose: int = 1) -> str:
        """
        Save a DataFrame to a file. Supports both Pandas and Polars DataFrames.

        Parameters
        ----------
        df : Union[pd.DataFrame, pl.DataFrame]
            The DataFrame to save
        file_name : str
            The name of the file to save the DataFrame to
        sep : str, default='\t'
            The separator to use in the file
        verbose : int, default=1
            Verbosity level for output messages
        
        Returns
        -------
        str
            The full path where the file was saved
        """
        # Auto-detect separator based on file extension
        if file_name.endswith('.csv'):
            sep = ','
        elif file_name.endswith('.tsv'):
            sep = '\t'

        # Handle both relative and absolute paths
        if os.path.isabs(file_name):
            path = file_name
        else:
            path = os.path.join(self.get_path('analysis'), file_name)
            
        if verbose:
            print(f"[i/o] Saving DataFrame to {path}")

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)

        if isinstance(df, pd.DataFrame):
            df.to_csv(path, sep=sep, index=False)
        elif isinstance(df, pl.DataFrame):
            df.write_csv(path, separator=sep)
        else:
            raise ValueError("Unsupported DataFrame type. Only Pandas and Polars DataFrames are supported.")
            
        return path  # Return the path where the file was saved

    def load_dataframe(self, 
                      file_name: str, 
                      sep: Optional[str] = None, 
                      to_pandas: bool = False, 
                      verbose: int = 1) -> Union[pd.DataFrame, pl.DataFrame]:
        """
        Load a DataFrame from a file. Supports both Pandas and Polars DataFrames.

        Parameters
        ----------
        file_name : str
            The name of the file to load the DataFrame from
        sep : str, optional
            The separator used in the file, if None auto-detect based on file extension
        to_pandas : bool, default=False
            If True, load the DataFrame as a Pandas DataFrame
        verbose : int, default=1
            Verbosity level for output messages

        Returns
        -------
        Union[pd.DataFrame, pl.DataFrame]
            The loaded DataFrame
        """
        # Auto-detect separator based on file extension
        if sep is None:
            if file_name.endswith('.csv'):
                sep = ','
            elif file_name.endswith('.tsv'):
                sep = '\t'
            else:
                sep = '\t'  # Default separator

        # Handle both relative and absolute paths
        if os.path.isabs(file_name):
            path = file_name
        else:
            path = os.path.join(self.get_path('analysis'), file_name)
            
        if verbose:
            print(f"[i/o] Loading DataFrame from {path}")

        if to_pandas:
            df = pd.read_csv(path, sep=sep)
        else:
            df = pl.read_csv(path, separator=sep)

        return df
