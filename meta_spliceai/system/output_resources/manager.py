"""
Output Manager

Manages output directory structure and file operations.

Created: 2025-10-28
"""

from pathlib import Path
from typing import Optional, Literal
import logging

from .registry import OutputRegistry, GenePaths, InferenceMode


class OutputManager:
    """
    Manages output directory structure for inference workflow.
    
    Follows the same pattern as genomic resources management for consistency.
    
    Attributes
    ----------
    registry : OutputRegistry
        Output registry for path resolution
    mode : InferenceMode
        Inference mode (base_only, hybrid, meta_only)
    is_test : bool
        If True, outputs go to tests/ subdirectory
    logger : logging.Logger
        Logger instance
    """
    
    def __init__(
        self,
        registry: OutputRegistry,
        mode: InferenceMode,
        is_test: bool = False,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize output manager.
        
        Parameters
        ----------
        registry : OutputRegistry
            Output registry instance
        mode : InferenceMode
            Inference mode
        is_test : bool
            If True, outputs go to tests/ subdirectory
        logger : Optional[logging.Logger]
            Logger instance
        """
        self.registry = registry
        self.mode = mode
        self.is_test = is_test  # Fixed: was 'is_test = is_test'
        self.logger = logger or logging.getLogger(__name__)
    
    def get_gene_output_paths(self, gene_id: str) -> GenePaths:
        """
        Get all output paths for a gene.
        
        Parameters
        ----------
        gene_id : str
            Gene ID (e.g., ENSG00000169239)
        
        Returns
        -------
        GenePaths
            Container with all output paths
        
        Examples
        --------
        >>> manager = OutputManager(registry, "hybrid", is_test=False)
        >>> paths = manager.get_gene_output_paths("ENSG00000169239")
        >>> paths.predictions_file
        Path('predictions/hybrid/ENSG00000169239/combined_predictions.parquet')
        """
        return self.registry.get_gene_paths(self.mode, gene_id, self.is_test)
    
    def get_combined_output_path(self) -> Path:
        """
        Get path for combined predictions across all genes.
        
        Returns
        -------
        Path
            Path to combined predictions file
        """
        mode_dir = self.registry.get_mode_dir(self.mode, self.is_test)
        return mode_dir / "all_genes_combined.parquet"
    
    def cleanup_old_predictions(self, gene_id: str):
        """
        Clean up old predictions for a gene (for overwrite mode).
        
        Parameters
        ----------
        gene_id : str
            Gene ID to clean up
        """
        gene_dir = self.registry.get_gene_dir(self.mode, gene_id, self.is_test)
        
        if gene_dir.exists():
            import shutil
            shutil.rmtree(gene_dir)
            self.logger.info(f"Cleaned up old predictions for {gene_id}")
    
    def get_artifact_path(
        self,
        artifact_type: Literal['analysis_sequences', 'base_predictions'],
        filename: str
    ) -> Path:
        """
        Get path for a specific artifact file.
        
        Parameters
        ----------
        artifact_type : Literal['analysis_sequences', 'base_predictions']
            Type of artifact
        filename : str
            Filename
        
        Returns
        -------
        Path
            Full path to artifact file
        """
        return self.registry.get_artifact_path(artifact_type, filename)
    
    def log_directory_structure(self):
        """Log the current directory structure."""
        predictions_base = self.registry.resolve('predictions_base')
        artifacts_base = self.registry.resolve('artifacts')
        
        self.logger.info("="*80)
        self.logger.info("OUTPUT DIRECTORY STRUCTURE")
        self.logger.info("="*80)
        self.logger.info(f"Predictions: {predictions_base}")
        self.logger.info(f"Mode: {self.mode}")
        self.logger.info(f"Test mode: {self.is_test}")
        self.logger.info(f"")
        self.logger.info(f"Structure:")
        self.logger.info(f"  {predictions_base}/")
        self.logger.info(f"  ├── {self.mode}/")
        self.logger.info(f"  │   ├── <gene_id>/")
        self.logger.info(f"  │   │   └── combined_predictions.parquet")
        if self.is_test:
            self.logger.info(f"  │   └── tests/")
            self.logger.info(f"  │       └── <gene_id>/")
        self.logger.info(f"  └── spliceai_eval/")
        self.logger.info(f"      └── meta_models/")
        self.logger.info(f"          ├── analysis_sequences/")
        self.logger.info(f"          └── complete_base_predictions/")
        self.logger.info("="*80)
    
    @classmethod
    def from_config(
        cls,
        config: 'EnhancedSelectiveInferenceConfig',
        logger: Optional[logging.Logger] = None,
        base_model_name: str = "spliceai"
    ) -> 'OutputManager':
        """
        Create OutputManager from inference config.
        
        Parameters
        ----------
        config : EnhancedSelectiveInferenceConfig
            Inference configuration
        logger : Optional[logging.Logger]
            Logger instance
        base_model_name : str
            Name of base model (default: 'spliceai')
            Examples: 'spliceai', 'openspliceai', 'pangolin'
        
        Returns
        -------
        OutputManager
            Configured output manager
        
        Examples
        --------
        # Default SpliceAI
        manager = OutputManager.from_config(config, logger)
        
        # OpenSpliceAI
        manager = OutputManager.from_config(config, logger, base_model_name="openspliceai")
        """
        from .config import OutputConfig
        
        # Create output config with base model name
        if config.inference_base_dir:
            output_config = OutputConfig(
                predictions_base=config.inference_base_dir,
                base_model_name=base_model_name
            )
        else:
            output_config = OutputConfig(base_model_name=base_model_name)
        
        # Create registry
        registry = OutputRegistry(output_config)
        
        # Determine if this is a test run
        is_test = config.output_name and 'test' in config.output_name.lower()
        
        return cls(
            registry=registry,
            mode=config.inference_mode,
            is_test=is_test,
            logger=logger
        )

