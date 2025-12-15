"""
User-friendly interface for running base model predictions.

This module provides an intuitive entry point for splice site prediction using
base models like SpliceAI or OpenSpliceAI. It's a thin wrapper around the
enhanced splice prediction workflow, designed for ease of use.

Examples
--------
Simple usage:
    >>> from meta_spliceai import run_base_model_predictions
    >>> results = run_base_model_predictions(target_genes=['BRCA1', 'TP53'])
    >>> print(f"Analyzed {results['positions'].height} positions")

With configuration:
    >>> from meta_spliceai import BaseModelConfig, run_base_model_predictions
    >>> config = BaseModelConfig(
    ...     mode='test',
    ...     threshold=0.5,
    ...     test_name='my_experiment'
    ... )
    >>> results = run_base_model_predictions(
    ...     config=config,
    ...     target_genes=['BRCA1']
    ... )

Quick predictions:
    >>> from meta_spliceai import predict_splice_sites
    >>> positions = predict_splice_sites('BRCA1')
    >>> print(positions.head())
"""

from typing import List, Optional, Dict, Any, Union
import polars as pl

from meta_spliceai.splice_engine.meta_models.workflows.splice_prediction_workflow import (
    run_enhanced_splice_prediction_workflow
)
from meta_spliceai.splice_engine.meta_models.core.data_types import SpliceAIConfig

__all__ = ['run_base_model_predictions', 'predict_splice_sites', 'BaseModelConfig']

# Alias for clarity - users think "base model config", not "SpliceAI config"
BaseModelConfig = SpliceAIConfig


def run_base_model_predictions(
    base_model: str = 'spliceai',
    target_genes: Optional[List[str]] = None,
    target_chromosomes: Optional[List[str]] = None,
    config: Optional[BaseModelConfig] = None,
    verbosity: int = 1,
    no_tn_sampling: bool = False,
    save_nucleotide_scores: bool = False,  # NEW: Control per-nucleotide score collection
    **kwargs
) -> Dict[str, Any]:
    """
    Run splice site predictions using a base model.
    
    This is a user-friendly wrapper around the enhanced splice prediction workflow,
    providing an intuitive interface for running base model predictions on genes
    or chromosomes of interest.
    
    Parameters
    ----------
    base_model : str, default='spliceai'
        Base model to use for predictions. Options:
        - 'spliceai': SpliceAI model (default)
        - 'openspliceai': OpenSpliceAI model (coming soon)
    
    target_genes : List[str], optional
        List of gene symbols or Ensembl IDs to analyze.
        Examples: ['BRCA1', 'TP53'], ['ENSG00000012048']
        If None, processes genes based on target_chromosomes or all genes.
    
    target_chromosomes : List[str], optional
        List of chromosomes to process.
        Examples: ['21', '22'], ['X', 'Y']
        If None and target_genes is None, processes all chromosomes.
    
    config : BaseModelConfig, optional
        Configuration object specifying workflow parameters.
        If None, uses default configuration with any kwargs overrides.
    
    verbosity : int, default=1
        Output verbosity level:
        - 0: Minimal (errors and final summary only)
        - 1: Normal (progress and important messages)
        - 2: Detailed (all processing details)
    
    no_tn_sampling : bool, default=False
        If True, keep all true negative positions without sampling.
        If False, sample true negatives to reduce dataset size.
        Set to True for validation/testing, False for training data generation.
    
    save_nucleotide_scores : bool, default=False
        If True, save per-nucleotide splice site scores for all positions.
        WARNING: This generates large data volumes (100s of MB to GBs).
        Recommended only for:
        - Visualization of splice site landscapes
        - Full-coverage inference mode
        - Detailed analysis of specific genes
        For meta-model training, keep False (default).
    
    **kwargs
        Additional configuration parameters passed to BaseModelConfig.
        Common options:
        - mode: 'test' or 'production'
        - coverage: 'gene_subset', 'chromosome', or 'full_genome'
        - test_name: Test identifier for test mode
        - threshold: Splice site score threshold (default: 0.5)
        - consensus_window: Window for consensus calling (default: 2)
        - error_window: Window for error analysis (default: 500)
    
    Returns
    -------
    dict
        Results dictionary containing:
        
        - 'success' : bool
            Whether the workflow completed successfully
        
        - 'positions' : polars.DataFrame
            All analyzed positions with predictions and features.
            Key columns: gene_id, position, splice_type, error_type,
            prob_donor, prob_acceptor, prob_neither, context_*, derived features
        
        - 'error_analysis' : polars.DataFrame
            Positions with errors (FP, FN) for focused analysis.
            Key columns: error_type, gene_id, position, splice_type, strand
        
        - 'analysis_sequences' : polars.DataFrame
            Sequences around each position for downstream analysis (if collected).
            Key columns: gene_id, position, sequence, splice_type, error_type
        
        - 'paths' : dict
            Output file paths:
            - 'eval_dir': Evaluation directory
            - 'output_dir': Output subdirectory
            - 'artifacts_dir': Artifacts directory
            - 'positions_artifact': Full positions file path
            - 'errors_artifact': Errors file path
        
        - 'artifact_manager' : dict
            Artifact management metadata:
            - 'mode': Execution mode
            - 'coverage': Coverage type
            - 'test_name': Test identifier
            - 'summary': Configuration summary
    
    Raises
    ------
    ValueError
        If unsupported base_model is specified
    
    NotImplementedError
        If OpenSpliceAI is requested (not yet implemented)
    
    Examples
    --------
    Basic usage with default settings:
    
    >>> results = run_base_model_predictions(
    ...     target_genes=['BRCA1', 'TP53', 'EGFR']
    ... )
    >>> print(f"Analyzed {results['positions'].height:,} positions")
    
    With custom configuration:
    
    >>> config = BaseModelConfig(
    ...     mode='test',
    ...     coverage='gene_subset',
    ...     test_name='validation_test',
    ...     threshold=0.5,
    ...     consensus_window=2,
    ...     use_auto_position_adjustments=True
    ... )
    >>> results = run_base_model_predictions(
    ...     config=config,
    ...     target_genes=['BRCA1'],
    ...     verbosity=1
    ... )
    
    Process entire chromosome:
    
    >>> results = run_base_model_predictions(
    ...     target_chromosomes=['21'],
    ...     mode='test',
    ...     coverage='chromosome',
    ...     test_name='chr21_test'
    ... )
    
    Production run (full genome):
    
    >>> results = run_base_model_predictions(
    ...     mode='production',
    ...     coverage='full_genome',
    ...     threshold=0.5,
    ...     use_auto_position_adjustments=True
    ... )
    
    Validation testing (keep all TNs):
    
    >>> results = run_base_model_predictions(
    ...     target_genes=['BRCA1', 'TP53'],
    ...     no_tn_sampling=True,  # Keep all true negatives
    ...     mode='test',
    ...     test_name='validation_run'
    ... )
    
    Notes
    -----
    - Test mode (default): Artifacts are overwritable, stored in test directories
    - Production mode: Artifacts are immutable, stored in production directories
    - Use no_tn_sampling=True for validation, False for training data generation
    - Position adjustments are auto-detected by default for better accuracy
    
    See Also
    --------
    predict_splice_sites : Simplified interface for quick predictions
    BaseModelConfig : Configuration class for detailed parameter control
    """
    # Validate base model
    supported_models = ['spliceai', 'openspliceai']
    base_model_lower = base_model.lower()
    
    if base_model_lower not in supported_models:
        raise ValueError(
            f"Unsupported base model: '{base_model}'. "
            f"Supported models: {supported_models}"
        )
    
    # Validate OpenSpliceAI prerequisites
    if base_model_lower == 'openspliceai':
        import os
        model_path = "data/models/openspliceai/"
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"OpenSpliceAI models not found at {model_path}. "
                f"Please download them first:\n"
                f"  ./scripts/base_model/download_openspliceai_models.sh"
            )
    
    # Create configuration if not provided
    if config is None:
        # BaseModelConfig.__post_init__ will automatically set correct paths
        # based on base_model parameter
        config = BaseModelConfig(
            base_model=base_model_lower,
            save_nucleotide_scores=save_nucleotide_scores,  # Pass through
            **kwargs
        )
    else:
        # If config provided, ensure base_model is set correctly
        from dataclasses import asdict
        config_dict = asdict(config)
        
        # Update base_model, save_nucleotide_scores, and any kwargs
        config_dict['base_model'] = base_model_lower
        config_dict['save_nucleotide_scores'] = save_nucleotide_scores
        config_dict.update(kwargs)
        
        # Recreate config - __post_init__ will handle path routing
        config = BaseModelConfig(**config_dict)
    
    # Delegate to the core workflow implementation
    return run_enhanced_splice_prediction_workflow(
        config=config,
        target_genes=target_genes,
        target_chromosomes=target_chromosomes,
        verbosity=verbosity,
        no_tn_sampling=no_tn_sampling
    )


def predict_splice_sites(
    genes: Union[str, List[str]],
    base_model: str = 'spliceai',
    threshold: float = 0.5,
    mode: str = 'test',
    verbosity: int = 0,
    **kwargs
) -> pl.DataFrame:
    """
    Quick prediction for a list of genes (simplified interface).
    
    This is a convenience function for the most common use case: getting
    splice site predictions for specific genes with minimal configuration.
    
    Parameters
    ----------
    genes : str or List[str]
        Gene symbol(s) or Ensembl ID(s) to analyze.
        Can be a single gene string or a list of genes.
        Examples: 'BRCA1' or ['BRCA1', 'TP53', 'EGFR']
    
    base_model : str, default='spliceai'
        Base model to use: 'spliceai' or 'openspliceai'
    
    threshold : float, default=0.5
        Splice site score threshold
    
    mode : str, default='test'
        Execution mode: 'test' or 'production'
    
    verbosity : int, default=0
        Output verbosity (0=minimal, 1=normal, 2=detailed)
    
    **kwargs
        Additional parameters passed to run_base_model_predictions
    
    Returns
    -------
    polars.DataFrame
        Positions DataFrame with predictions.
        Columns include: gene_id, position, splice_type, error_type,
        prob_donor, prob_acceptor, prob_neither, and derived features.
    
    Examples
    --------
    Single gene:
    
    >>> positions = predict_splice_sites('BRCA1')
    >>> print(f"Found {positions.height} positions")
    
    Multiple genes:
    
    >>> positions = predict_splice_sites(['BRCA1', 'TP53', 'EGFR'])
    >>> print(positions.head())
    
    With custom threshold:
    
    >>> positions = predict_splice_sites(
    ...     genes=['BRCA1'],
    ...     threshold=0.2,  # More sensitive
    ...     verbosity=1
    ... )
    
    Notes
    -----
    This function returns only the positions DataFrame. For full results
    including error analysis and paths, use run_base_model_predictions().
    
    See Also
    --------
    run_base_model_predictions : Full-featured interface with all options
    """
    # Handle single gene string
    if isinstance(genes, str):
        genes = [genes]
    
    # Run predictions with minimal output
    results = run_base_model_predictions(
        base_model=base_model,
        target_genes=genes,
        threshold=threshold,
        mode=mode,
        verbosity=verbosity,
        **kwargs
    )
    
    # Return just the positions DataFrame
    return results['positions']


# Convenience aliases for backward compatibility or alternative naming
run_splice_prediction = run_base_model_predictions
predict_splice_sites_for_genes = predict_splice_sites

