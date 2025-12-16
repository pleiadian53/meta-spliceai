"""
Core analyzer class for splice site prediction and evaluation.

This module defines the Analyzer base class that provides configuration
and path standardization across the splice-surveyor project.

NOTE: 

This module has been refactored to its dedicated analyzers package, which
contains specialized analyzers for splice site prediction and analysis.
... 04.2025
"""

import os
from typing import Optional, Dict, Any, Union
import polars as pl
import pandas as pd


class Analyzer(object):
    """
    Base analyzer class providing standardized paths and configuration.
    
    This class serves as a foundation for specialized analyzers in the
    splice-surveyor project, ensuring consistent path structure and
    configuration across different components.
    
    Attributes
    ----------
    source : str
        Data source name (default: 'ensembl')
    version : str
        Version string for the data source
    prefix : str
        Root directory of the project
    data_dir : str
        Directory containing genomic data
    gtf_file : str
        Path to the GTF annotation file
    genome_fasta : str
        Path to the genome FASTA file
    eval_dir : str
        Directory for evaluation outputs
    analysis_dir : str
        Directory for analysis outputs
    """
    source = 'ensembl'
    version = ''
    
    # Use system config for project directory
    try:
        from meta_spliceai.system.config import Config as SystemConfig
        prefix = SystemConfig.PROJ_DIR
    except (ImportError, AttributeError):
        # Fallback to environment variable with proper package name
        try:
            from meta_spliceai.system.config import find_project_root
            prefix = os.getenv("META_SPLICEAI_ROOT", find_project_root())
        except ImportError:
            from pathlib import Path
            prefix = os.getenv("META_SPLICEAI_ROOT", str(Path(__file__).parent.parent.parent.parent))
    
    # Blob storage prefix (configurable via environment variable)
    blob_prefix = os.getenv("META_SPLICEAI_BLOB", "/mnt/nfs1/meta-spliceai")
    data_dir = os.path.join(prefix, "data", "ensembl")
    shared_dir = data_dir
    
    # Paths to genomic data files
    # Note: Now uses systematic path management instead of hardcoded paths
    try:
        from meta_spliceai.system.genomic_resources import create_systematic_manager
        _genomic_manager = create_systematic_manager()  # Uses default build/release from config
        gtf_file = str(_genomic_manager.get_gtf_path(validate=False))
        genome_fasta = str(_genomic_manager.get_fasta_path(validate=False))
    except ImportError:
        # Fallback to original hardcoded paths if genomic_resources not available
        gtf_file = os.path.join(prefix, "data", "ensembl", "Homo_sapiens.GRCh38.112.gtf")
        genome_fasta = os.path.join(prefix, "data", "ensembl", "Homo_sapiens.GRCh38.dna.primary_assembly.fa")
    
    # Evaluation and analysis directories
    eval_dir = os.path.join(data_dir, "spliceai_eval")
    analysis_dir = os.path.join(data_dir, "spliceai_analysis")
    
    # Default configuration parameters
    default_config = {
        'separator': '\t',
        'format': 'tsv',
        'threshold': 0.5,
        'consensus_window': 2,
        'error_window': 500
    }
    
    def __init__(self, 
                 gtf_file: Optional[str] = None, 
                 genome_fasta: Optional[str] = None,
                 eval_dir: Optional[str] = None,
                 **kwargs):
        """
        Initialize an Analyzer with custom paths if provided.
        
        Parameters
        ----------
        gtf_file : str, optional
            Path to GTF file, defaults to class attribute
        genome_fasta : str, optional
            Path to genome FASTA file, defaults to class attribute
        eval_dir : str, optional
            Path to evaluation directory, defaults to class attribute
        **kwargs : dict
            Additional configuration parameters
        """
        # Initialize paths
        self._gtf_file = gtf_file or self.__class__.gtf_file
        self._genome_fasta = genome_fasta or self.__class__.genome_fasta
        self._eval_dir = eval_dir or self.__class__.eval_dir
        
        # Initialize other configuration
        self._config = self.__class__.default_config.copy()
        for key, value in kwargs.items():
            if key in self._config:
                self._config[key] = value
    
    def get_path(self, path_type: str) -> str:
        """
        Get a standardized path.
        
        Parameters
        ----------
        path_type : str
            Type of path to retrieve ('gtf', 'fasta', 'eval', 'analysis')
            
        Returns
        -------
        str
            Path corresponding to the requested type
        """
        if path_type == 'gtf':
            return self._gtf_file
        elif path_type == 'fasta':
            return self._genome_fasta
        elif path_type == 'eval':
            return self._eval_dir
        elif path_type == 'analysis':
            return self.__class__.analysis_dir
        else:
            raise ValueError(f"Unknown path type: {path_type}")
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get the current configuration.
        
        Returns
        -------
        dict
            Dictionary of configuration parameters
        """
        return self._config.copy()


class ErrorAnalyzer(Analyzer):
    """
    Specialized analyzer for splice site error analysis.
    
    This class extends the base Analyzer with specific functionality for
    analyzing different types of prediction errors (FP, FN) and true positives (TP).
    
    Attributes
    ----------
    schema : dict
        Schema definition for error analysis dataframes
    pred_type_to_label : dict
        Mapping between prediction types and numerical labels
    """
    
    # Schema definition for error analysis
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

    # Mapping for prediction types to numerical labels
    pred_type_to_label = {
        'FP': 1,  # False Positive
        'FN': 1,  # False Negative
        'TP': 0   # True Positive
    }
    
    def __init__(self, 
                 data_dir: Optional[str] = None, 
                 source: str = 'ensembl',
                 version: Optional[str] = None, 
                 gtf_file: Optional[str] = None, 
                 genome_fasta: Optional[str] = None, 
                 window_size: int = 500, 
                 **kwargs):
        """
        Initialize an ErrorAnalyzer for splice site error analysis.
        
        Parameters
        ----------
        data_dir : str, optional
            Directory containing source data
        source : str, default='ensembl'
            Name of data source
        version : str, optional
            Version string for the data source
        gtf_file : str, optional
            Path to GTF file
        genome_fasta : str, optional
            Path to genome FASTA file
        window_size : int, default=500
            Window size for error analysis
        **kwargs : dict
            Additional configuration parameters
        """
        super().__init__(gtf_file=gtf_file, genome_fasta=genome_fasta)
        
        # Source and version
        self.source = source
        self.version = version
        
        # Directories
        self.data_dir = data_dir or f"{Analyzer.prefix}/data/{source}"
        self._output_dir = None
        
        # Analysis parameters
        self.window_size = window_size
        self.separator = kwargs.get('separator', '\t')
        
        # Model parameters
        self.feature_importance_base_model = kwargs.get('feature_importance_base_model', 'xgboost')
        self.importance_type = kwargs.get('importance_type', 'shap')
        self.experiment = kwargs.get('experiment', None)
        self.splice_type = kwargs.get('splice_type', None)
        self.model_type = kwargs.get('model_type', None)
        
        # Prediction type parameters
        self.pred_type = kwargs.get('pred_type', None)
        self.correct_label = kwargs.get('correct_label', "TP")
        self.error_label = kwargs.get('error_label', self.pred_type)
        self.splice_type = kwargs.get('splice_type', None)
    
    def retrieve_data_points(self, 
                            pred_type: str,
                            chr: Optional[str] = None, 
                            chunk_start: Optional[int] = None, 
                            chunk_end: Optional[int] = None, 
                            aggregated: bool = True, 
                            **kwargs) -> pl.DataFrame:
        """
        Generic method to retrieve data points of a given prediction type.
        
        Parameters
        ----------
        pred_type : str
            Type of prediction ('TP', 'FP', 'FN')
        chr : str, optional
            Chromosome name
        chunk_start : int, optional
            Start position of chunk
        chunk_end : int, optional
            End position of chunk
        aggregated : bool, default=True
            Whether to use aggregated data
        **kwargs : dict
            Additional parameters
            
        Returns
        -------
        pl.DataFrame
            DataFrame containing the requested data points
        """
        from meta_spliceai.splice_engine.model_evaluation import ModelEvaluationFileHandler
        from meta_spliceai.splice_engine.splice_error_analyzer import (
            extract_tp_data_points, extract_fp_data_points, extract_fn_data_points, print_emphasized
        )
        
        verbose = kwargs.get('verbose', 1)
        overwrite = kwargs.get('overwrite', False)
        subject = kwargs.get('subject', f"splice_{pred_type.lower()}")
        
        # Initialize file handler
        mefd = ModelEvaluationFileHandler(self.eval_dir, separator=self.separator)
        
        # Get method to retrieve data points based on pred_type
        if pred_type == 'TP':
            data_path = mefd.get_tp_data_file_path(chr, chunk_start, chunk_end, aggregated, subject=subject)
            load_method = mefd.load_tp_data_points
            extract_method = extract_tp_data_points
        elif pred_type == 'FP':
            data_path = mefd.get_fp_data_file_path(chr, chunk_start, chunk_end, aggregated, subject=subject)
            load_method = mefd.load_fp_data_points
            extract_method = extract_fp_data_points
        elif pred_type == 'FN':
            data_path = mefd.get_fn_data_file_path(chr, chunk_start, chunk_end, aggregated, subject=subject)
            load_method = mefd.load_fn_data_points
            extract_method = extract_fn_data_points
        else:
            raise ValueError(f"Unknown prediction type: {pred_type}")
        
        # Load or extract data points
        if not overwrite and os.path.exists(data_path):
            if verbose:
                print_emphasized(f"[i/o] Loading {pred_type} data points from {data_path} ...")
            df = load_method(chr, chunk_start, chunk_end, aggregated, subject=subject)
        else:
            # Extract data points
            if verbose:
                print_emphasized(f"[extract] Extracting {pred_type} data points...")
            df = extract_method(gtf_file=self.gtf_file, window_size=self.window_size, save=True, verbose=verbose)
        
        return df
    
    def retrieve_tp_data_points(self, chr=None, chunk_start=None, chunk_end=None, aggregated=True, **kwargs):
        """Retrieve True Positive data points."""
        return self.retrieve_data_points('TP', chr, chunk_start, chunk_end, aggregated, **kwargs)
    
    def retrieve_fp_data_points(self, chr=None, chunk_start=None, chunk_end=None, aggregated=True, **kwargs):
        """Retrieve False Positive data points."""
        return self.retrieve_data_points('FP', chr, chunk_start, chunk_end, aggregated, **kwargs)
    
    def retrieve_fn_data_points(self, chr=None, chunk_start=None, chunk_end=None, aggregated=True, **kwargs):
        """Retrieve False Negative data points."""
        return self.retrieve_data_points('FN', chr, chunk_start, chunk_end, aggregated, **kwargs)
    
    def set_analysis_output_dir(self, pred_type=None, experiment=None, error_label=None, correct_label=None, **kwargs):
        """
        Set the output directory for analysis results.
        
        Parameters
        ----------
        pred_type : str, optional
            Type of prediction ('FP', 'FN')
        experiment : str, optional
            Experiment name
        error_label : str, optional
            Label for error type
        correct_label : str, optional
            Label for correct type
        **kwargs : dict
            Additional parameters
            
        Returns
        -------
        str
            Path to the output directory
        """
        from meta_spliceai.splice_engine.splice_error_analyzer import print_emphasized
        
        verbose = kwargs.get('verbose', 1)
        output_dir = Analyzer.analysis_dir
        
        # Update prediction type parameters
        if pred_type is not None:
            self.pred_type = pred_type
        
        self.error_label = error_label or self.error_label or self.pred_type
        self.correct_label = correct_label or self.correct_label or "TP"
        
        # Set experiment directory
        self.experiment = experiment or self.experiment
        if self.experiment:
            output_dir = os.path.join(output_dir, self.experiment)

        # Set splice type directory
        self.splice_type = kwargs.get('splice_type', self.splice_type)
        if self.splice_type:
            output_dir = os.path.join(output_dir, self.splice_type)
        
        # Set error analysis type directory
        if self.error_label is not None and self.correct_label is not None:
            error_analysis_type = f"{self.error_label}_vs_{self.correct_label}".lower()
            output_dir = os.path.join(output_dir, error_analysis_type)
        
        # Set model type directory
        self.model_type = kwargs.get('model_type', self.model_type)
        if self.model_type:
            output_dir = os.path.join(output_dir, self.model_type)
        
        os.makedirs(output_dir, exist_ok=True)
        self._output_dir = output_dir
        
        if verbose:
            print_emphasized(f"[info] Output directory set to {output_dir}")
        
        return output_dir
    
    @property
    def output_dir(self):
        """Get the current output directory."""
        return self._output_dir


class SpliceAnalyzer(Analyzer):
    """
    Specialized analyzer for splice site prediction and analysis.
    
    This class extends the base Analyzer with specific functionality
    for working with splice site annotations, including support for
    automatic position adjustments and probability coherence.
    
    Attributes
    ----------
    db_file : str
        Path to the annotations database file
    splice_sites_file : str
        Name of the standard splice sites annotation file
    enhanced_splice_sites_file : str
        Name of the enhanced splice sites file with three-probability scores
    """
    
    # Standard file paths for splice site analysis
    db_file = os.path.join(Analyzer.data_dir, "annotations.db")
    splice_sites_file = "splice_sites.tsv"
    enhanced_splice_sites_file = "splice_sites_enhanced.tsv"
    donor_acceptor_pred_file = "donor_acceptor_pred.tsv"

    def __init__(self, 
                data_dir: Optional[str] = None, 
                *, 
                source: str = 'ensembl', 
                version: Optional[str] = None,
                **kwargs):
        """
        Initialize a SpliceAnalyzer.
        
        Parameters
        ----------
        data_dir : str, optional
            Directory containing splice site data, if None uses default
        source : str, default='ensembl'
            Data source identifier
        version : str, optional
            Version string for the data source
        **kwargs : dict
            Additional configuration parameters passed to parent
        """
        super().__init__(**kwargs)
        self.source = source
        self.version = version
        self.data_dir = data_dir or Analyzer.data_dir

    @property
    def path_to_splice_sites(self) -> str:
        """Get the path to the standard splice sites file."""
        return os.path.join(self.data_dir, self.splice_sites_file)
    
    @property
    def path_to_enhanced_splice_sites(self) -> str:
        """Get the path to the enhanced splice sites file with three-probability scores."""
        return os.path.join(self.data_dir, self.enhanced_splice_sites_file)
    
    @property
    def path_to_donor_acceptor_pred(self) -> str:
        """Get the path to the donor-acceptor prediction file."""
        return os.path.join(self.data_dir, self.donor_acceptor_pred_file)

    @property
    def path_to_overlapping_gene_metadata(self) -> str:
        """Get the path to the overlapping gene metadata file."""
        return os.path.join(self.data_dir, "overlapping_gene_counts.tsv")

    def load_splice_sites(self, 
                         verbose: int = 1, 
                         enhanced: bool = False,
                         raise_exception: bool = False) -> Union[pd.DataFrame, pl.DataFrame, None]:
        """
        Load splice site annotations.
        
        Parameters
        ----------
        verbose : int, default=1
            Verbosity level for output messages
        enhanced : bool, default=False
            Whether to load enhanced splice sites with three-probability scores
        raise_exception : bool, default=False
            Whether to raise an exception if file is not found
            
        Returns
        -------
        Union[pd.DataFrame, pl.DataFrame, None]
            Dataframe with splice site annotations or None if not found
            
        Raises
        ------
        FileNotFoundError
            If the splice sites file is not found and raise_exception is True
        """
        # Determine which file to load
        splice_sites_file_path = self.path_to_enhanced_splice_sites if enhanced else self.path_to_splice_sites
        
        if os.path.exists(splice_sites_file_path):
            from meta_spliceai.splice_engine.utils_fs import read_splice_sites
            return read_splice_sites(splice_sites_file_path, verbose=verbose)

        if verbose:
            file_type = "enhanced " if enhanced else ""
            print(f"[warning] Could not find {file_type}splice sites at:\n{splice_sites_file_path}\n")

        if raise_exception:
            raise FileNotFoundError(f"Splice sites file not found at: {splice_sites_file_path}")
        
        return None
        
    def verify_probability_coherence(self, 
                                    predictions: Union[pd.DataFrame, pl.DataFrame],
                                    tolerance: float = 0.01,
                                    verbose: int = 1) -> bool:
        """
        Verify that donor + acceptor + neither probabilities sum to approximately 1.0.
        
        This is critical for ensuring the validity of position adjustments.
        
        Parameters
        ----------
        predictions : Union[pd.DataFrame, pl.DataFrame]
            DataFrame containing donor_prob, acceptor_prob, and neither_prob columns
        tolerance : float, default=0.01
            Maximum allowed deviation from 1.0 for probability sum
        verbose : int, default=1
            Verbosity level for output messages
            
        Returns
        -------
        bool
            True if probability coherence is maintained, False otherwise
        """
        # Handle both pandas and polars DataFrames
        if isinstance(predictions, pd.DataFrame):
            # Pandas implementation
            if not all(col in predictions.columns for col in ['donor_prob', 'acceptor_prob', 'neither_prob']):
                if verbose:
                    print("[error] Missing probability columns in predictions DataFrame")
                return False
                
            # Calculate sum of probabilities
            prob_sum = predictions['donor_prob'] + predictions['acceptor_prob'] + predictions['neither_prob']
            coherent = ((1.0 - tolerance) <= prob_sum) & (prob_sum <= (1.0 + tolerance))
            
            if verbose and not coherent.all():
                incoherent_count = (~coherent).sum()
                print(f"[warning] Found {incoherent_count} positions with incoherent probabilities")
                
            return coherent.all()
            
        else:
            # Polars implementation
            required_cols = ['donor_prob', 'acceptor_prob', 'neither_prob']
            if not all(col in predictions.columns for col in required_cols):
                if verbose:
                    print("[error] Missing probability columns in predictions DataFrame")
                return False
                
            # Calculate sum and check if within tolerance
            with pl.Config(tbl_rows=10):
                check_result = predictions.select(
                    pl.col("donor_prob") + pl.col("acceptor_prob") + pl.col("neither_prob")
                ).with_column(
                    pl.when(
                        (pl.col("donor_prob") + pl.col("acceptor_prob") + pl.col("neither_prob") >= 1.0 - tolerance) &
                        (pl.col("donor_prob") + pl.col("acceptor_prob") + pl.col("neither_prob") <= 1.0 + tolerance)
                    ).then(True).otherwise(False).alias("is_coherent")
                )
                
                coherent = check_result.select(pl.all().is_coherent()).collect().to_series().all()
                
                if verbose and not coherent:
                    incoherent_count = check_result.filter(~pl.col("is_coherent")).height
                    print(f"[warning] Found {incoherent_count} positions with incoherent probabilities")
                    
                return coherent
