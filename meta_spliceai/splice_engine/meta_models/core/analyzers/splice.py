"""
Splice site analyzer for the splice-surveyor project.

This module provides specialized analyzers for splice site
prediction and analysis, including support for:
- Splice site adjustment detection
- Three-probability coherence (donor, acceptor, neither)
- Consistent transcript-to-gene relationship preservation
"""

import os
import pandas as pd
import polars as pl
from typing import Union, Optional, Dict, List, Any

from .base import Analyzer
from meta_spliceai.splice_engine.utils_fs import read_splice_sites

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
            return read_splice_sites(splice_sites_file_path, verbose=verbose)

        if verbose:
            file_type = "enhanced " if enhanced else ""
            print(f"[warning] Could not find {file_type}splice sites at:\n{splice_sites_file_path}\n")

        if raise_exception:
            raise FileNotFoundError(f"Splice sites file not found at: {splice_sites_file_path}")
        
        return None

    def retrieve_splice_sites(self, consensus_window=2, verbose=1, column_names={}):
        df_splice = self.load_splice_sites(verbose=verbose, raise_exception=False)

        if df_splice is None or df_splice.is_empty():
            from meta_spliceai.splice_engine.extract_genomic_features import extract_splice_sites_workflow
            if verbose: 
                print_emphasized("[action] Computing splice sites from GTF ...")
            extract_splice_sites_workflow(
                    data_prefix=self.data_dir, 
                    gtf_file=self.get_path('gtf'), 
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
        from meta_spliceai.splice_engine.extract_genomic_features import get_or_load_overlapping_gene_metadata
        verbose = kargs.get('verbose', 1)
        output_format = kargs.get('output_format', 'dict')
        to_pandas = kargs.get('to_pandas', False)
        result_set = \
            get_or_load_overlapping_gene_metadata(
                gtf_file_path=self.get_path('gtf'),
                overlapping_gene_path=self.path_to_overlapping_gene_metadata, 
                filter_valid_splice_sites=filter_valid_splice_sites, 
                min_exons=min_exons, 
                output_format=output_format, verbose=verbose)
        if output_format == 'dataframe': 
            if to_pandas: 
                return result_set.to_pandas()
            return result_set
        return result_set
        
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

    def get_position_adjustments(self, 
                                detector_output_path: Optional[str] = None,
                                auto_detect: bool = True,
                                verbose: int = 1) -> Dict[str, Dict[str, int]]:
        """
        Get position adjustments for different strand and site type combinations.
        
        This method either loads pre-detected position adjustments or automatically
        detects them using the splice site prediction data.
        
        Parameters
        ----------
        detector_output_path : str, optional
            Path to load pre-detected position adjustments from
        auto_detect : bool, default=True
            Whether to automatically detect position adjustments if not found
        verbose : int, default=1
            Verbosity level for output messages
            
        Returns
        -------
        Dict[str, Dict[str, int]]
            Nested dictionary mapping strand ('+'/'-') to site type ('donor'/'acceptor')
            to position adjustment values
        """
        # Default path for adjustment data if not specified
        if detector_output_path is None:
            detector_output_path = os.path.join(self.data_dir, "splice_site_adjustments.json")
            
        # Check if adjustments file exists
        if os.path.exists(detector_output_path):
            import json
            with open(detector_output_path, 'r') as f:
                adjustments = json.load(f)
                if verbose:
                    print(f"[info] Loaded position adjustments from: {detector_output_path}")
                    print(f"[info] Adjustments: {adjustments}")
                return adjustments
        
        # Auto-detect if requested and file not found
        if auto_detect:
            if verbose:
                print(f"[info] No position adjustments found at: {detector_output_path}")
                print("[info] Auto-detecting position adjustments...")
                
            try:
                # This would typically import from your auto-detection module
                from meta_spliceai.splice_engine.meta_models.utils.infer_splice_site_adjustments import (
                    auto_detect_splice_site_adjustments
                )
                
                # Load the necessary data for detection
                splice_sites_df = self.load_splice_sites(verbose=verbose, raise_exception=True)
                
                # Run the auto-detection
                adjustments = auto_detect_splice_site_adjustments(
                    splice_sites_df,
                    verbose=verbose
                )
                
                # Save the detected adjustments for future use
                import json
                with open(detector_output_path, 'w') as f:
                    json.dump(adjustments, f, indent=2)
                    
                if verbose:
                    print(f"[info] Detected and saved position adjustments to: {detector_output_path}")
                    print(f"[info] Adjustments: {adjustments}")
                    
                return adjustments
                
            except (ImportError, FileNotFoundError) as e:
                if verbose:
                    print(f"[error] Failed to auto-detect position adjustments: {str(e)}")
                return {}
        
        # Return empty adjustment dict if nothing found and auto-detect disabled
        if verbose:
            print("[warning] No position adjustments found and auto-detect disabled")
        return {}
    
    def apply_position_adjustments(self, 
                                 predictions: Union[pd.DataFrame, pl.DataFrame],
                                 adjustments: Optional[Dict[str, Dict[str, int]]] = None,
                                 verify_coherence: bool = True,
                                 verbose: int = 1) -> Union[pd.DataFrame, pl.DataFrame]:
        """
        Apply position adjustments to prediction probabilities.
        
        This method shifts the probability arrays based on strand and site type,
        ensuring that all three probabilities (donor, acceptor, neither) are
        adjusted together to maintain coherence.
        
        Parameters
        ----------
        predictions : Union[pd.DataFrame, pl.DataFrame]
            DataFrame containing prediction data with probability columns
        adjustments : Dict[str, Dict[str, int]], optional
            Position adjustments to apply, if None will auto-detect
        verify_coherence : bool, default=True
            Whether to verify probability coherence after adjustments
        verbose : int, default=1
            Verbosity level for output messages
            
        Returns
        -------
        Union[pd.DataFrame, pl.DataFrame]
            DataFrame with adjusted probability values
        """
        # Get adjustments if not provided
        if adjustments is None:
            adjustments = self.get_position_adjustments(verbose=verbose)
            
        if not adjustments:
            if verbose:
                print("[warning] No position adjustments to apply")
            return predictions
            
        if verbose:
            print(f"[info] Applying position adjustments: {adjustments}")
            
        # This would typically import from your adjustment application module
        try:
            from meta_spliceai.splice_engine.meta_models.utils.infer_splice_site_adjustments import (
                apply_auto_detected_adjustments
            )
            
            # Apply the adjustments to all three probability arrays simultaneously
            adjusted_predictions = apply_auto_detected_adjustments(
                predictions,
                adjustments,
                verbose=verbose
            )
            
            # Verify coherence if requested
            if verify_coherence:
                is_coherent = self.verify_probability_coherence(
                    adjusted_predictions,
                    verbose=verbose
                )
                
                if not is_coherent and verbose:
                    print("[error] Adjusted predictions failed probability coherence check")
            
            return adjusted_predictions
            
        except ImportError as e:
            if verbose:
                print(f"[error] Failed to apply position adjustments: {str(e)}")
            return predictions
