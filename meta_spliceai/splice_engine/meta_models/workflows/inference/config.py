"""
Configuration classes and factory functions for selective meta-model inference.

This module contains all configuration-related code extracted from selective_meta_inference.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class SelectiveInferenceConfig:
    """Configuration for selective meta-model inference."""
    
    # Core parameters
    model_path: Path
    target_genes: List[str]
    
    # Selective featurization thresholds
    uncertainty_threshold_low: float = 0.02   # Below this: confident non-splice
    uncertainty_threshold_high: float = 0.80  # Above this: confident splice
    
    # Processing limits
    max_positions_per_gene: int = 5000  # Prevent huge feature matrices
    max_analysis_rows: int = 500000     # Total position limit
    
    # Training integration
    training_dataset_path: Optional[Path] = None
    training_schema_path: Optional[Path] = None
    
    # Directory management  
    inference_base_dir: Path = None  # Will be set using GenomicResourceManager
    output_name: Optional[str] = None
    keep_artifacts_dir: Optional[Path] = None  # Directory to preserve artifacts and test data
    
    # Processing options
    use_calibration: bool = True
    cleanup_intermediates: bool = True
    verbose: int = 1
    
    # Inference mode: 'base_only', 'hybrid', 'meta_only'
    inference_mode: str = "hybrid"  # Default to hybrid for scalability
    
    # Complete coverage flag
    ensure_complete_coverage: bool = False  # Generate predictions for ALL positions
    
    # Chunked processing for true meta_only mode
    enable_chunked_processing: bool = False  # Enable chunked processing for full coverage
    chunk_size: int = 10000  # Size of each processing chunk


@dataclass
class SelectiveInferenceResults:
    """Results from selective meta-model inference."""
    
    success: bool
    config: SelectiveInferenceConfig
    
    # Output paths
    hybrid_predictions_path: Optional[Path] = None    # Combined base + meta predictions
    meta_predictions_path: Optional[Path] = None      # Meta-model predictions only
    base_predictions_path: Optional[Path] = None      # Base model predictions only
    gene_manifest_path: Optional[Path] = None         # Processing tracking
    
    # Statistics
    total_positions: int = 0
    positions_recalibrated: int = 0
    positions_reused: int = 0
    genes_processed: int = 0
    
    # Performance
    processing_time_seconds: float = 0.0
    feature_matrix_size_mb: float = 0.0
    
    # Details
    per_gene_stats: Dict[str, Dict] = None
    error_messages: List[str] = None


def create_selective_config(
    model_path: str,
    target_genes: List[str],
    training_dataset_path: Optional[str] = None,
    uncertainty_low: float = 0.02,
    uncertainty_high: float = 0.80,
    max_positions_per_gene: int = 5000,
    keep_artifacts_dir: Optional[str] = None,
    inference_mode: str = "hybrid",
    ensure_complete_coverage: bool = False,
    enable_chunked_processing: bool = False,
    chunk_size: int = 10000,
    verbose: int = 1
) -> SelectiveInferenceConfig:
    """
    Create a selective inference configuration.
    
    Parameters
    ----------
    model_path : str
        Path to the trained model directory
    target_genes : List[str]
        List of gene IDs to process
    training_dataset_path : str, optional
        Path to the training dataset
    uncertainty_low : float
        Lower threshold for uncertainty (default: 0.02)
    uncertainty_high : float
        Upper threshold for uncertainty (default: 0.80)
    max_positions_per_gene : int
        Maximum positions per gene to process (default: 5000)
    keep_artifacts_dir : str, optional
        Directory to preserve artifacts
    inference_mode : str
        Inference mode: 'base_only', 'hybrid', or 'meta_only' (default: 'hybrid')
    ensure_complete_coverage : bool
        Whether to ensure complete coverage (default: False)
    enable_chunked_processing : bool
        Enable chunked processing for large datasets (default: False)
    chunk_size : int
        Size of processing chunks (default: 10000)
    verbose : int
        Verbosity level (default: 1)
        
    Returns
    -------
    SelectiveInferenceConfig
        Configuration object for selective inference
    """
    return SelectiveInferenceConfig(
        model_path=Path(model_path),
        target_genes=target_genes,
        training_dataset_path=Path(training_dataset_path) if training_dataset_path else None,
        uncertainty_threshold_low=uncertainty_low,
        uncertainty_threshold_high=uncertainty_high,
        max_positions_per_gene=max_positions_per_gene,
        keep_artifacts_dir=Path(keep_artifacts_dir) if keep_artifacts_dir else None,
        inference_mode=inference_mode,
        ensure_complete_coverage=ensure_complete_coverage,
        enable_chunked_processing=enable_chunked_processing,
        chunk_size=chunk_size,
        verbose=verbose
    )
