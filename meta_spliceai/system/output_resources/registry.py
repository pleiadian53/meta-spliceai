"""
Output Registry

Central registry for output paths, following the pattern of genomic_resources.Registry

Created: 2025-10-28
"""

from pathlib import Path
from typing import Dict, Literal, Optional
from dataclasses import dataclass

from .config import OutputConfig

InferenceMode = Literal['base_only', 'hybrid', 'meta_only']


@dataclass
class GenePaths:
    """Container for all output paths for a gene."""
    
    # Main prediction file
    predictions_file: Path
    
    # Parent directory
    gene_dir: Path
    
    # Artifacts (shared across modes)
    artifacts_dir: Path
    analysis_sequences_dir: Path
    base_predictions_dir: Path


class OutputRegistry:
    """
    Central registry for output paths.
    
    Follows the same pattern as genomic_resources.Registry for consistency.
    
    Attributes
    ----------
    config : OutputConfig
        Output configuration
    """
    
    def __init__(self, config: Optional[OutputConfig] = None):
        """
        Initialize output registry.
        
        Parameters
        ----------
        config : Optional[OutputConfig]
            Output configuration (creates default if None)
        """
        self.config = config or OutputConfig()
        self._cache: Dict[str, Path] = {}
    
    def resolve(self, resource_kind: str) -> Path:
        """
        Resolve path for a resource kind.
        
        Similar to genomic_resources.Registry.resolve()
        
        Parameters
        ----------
        resource_kind : str
            Type of resource:
            - 'predictions_base': Base predictions directory
            - 'artifacts': Artifacts base directory
            - 'analysis_sequences': Analysis sequences directory
            - 'base_predictions': Base predictions directory
        
        Returns
        -------
        Path
            Resolved path
        
        Examples
        --------
        >>> registry = OutputRegistry()
        >>> registry.resolve('predictions_base')
        Path('predictions')
        """
        if resource_kind in self._cache:
            return self._cache[resource_kind]
        
        if resource_kind == 'predictions_base':
            path = self.config.predictions_base
        elif resource_kind == 'artifacts':
            path = self.config.artifacts_base
        elif resource_kind == 'analysis_sequences':
            path = self.config.artifacts_base / "analysis_sequences"
            path.mkdir(parents=True, exist_ok=True)
        elif resource_kind == 'base_predictions':
            path = self.config.artifacts_base / "complete_base_predictions"
            path.mkdir(parents=True, exist_ok=True)
        else:
            raise ValueError(f"Unknown resource kind: {resource_kind}")
        
        self._cache[resource_kind] = path
        return path
    
    def get_mode_dir(self, mode: InferenceMode, is_test: bool = False) -> Path:
        """
        Get directory for a specific mode.
        
        Parameters
        ----------
        mode : InferenceMode
            Inference mode (base_only, hybrid, meta_only)
        is_test : bool
            If True, returns test subdirectory
        
        Returns
        -------
        Path
            Mode directory
        
        Examples
        --------
        >>> registry = OutputRegistry()
        >>> registry.get_mode_dir('hybrid', is_test=False)
        Path('predictions/hybrid')
        
        >>> registry.get_mode_dir('hybrid', is_test=True)
        Path('predictions/hybrid/tests')
        """
        base = self.resolve('predictions_base')
        mode_dir = base / mode
        
        if is_test:
            mode_dir = mode_dir / "tests"
        
        mode_dir.mkdir(parents=True, exist_ok=True)
        return mode_dir
    
    def get_gene_dir(self, mode: InferenceMode, gene_id: str, is_test: bool = False) -> Path:
        """
        Get directory for a specific gene.
        
        Parameters
        ----------
        mode : InferenceMode
            Inference mode
        gene_id : str
            Gene ID (e.g., ENSG00000169239)
        is_test : bool
            If True, uses test subdirectory
        
        Returns
        -------
        Path
            Gene directory
        
        Examples
        --------
        >>> registry = OutputRegistry()
        >>> registry.get_gene_dir('hybrid', 'ENSG00000169239')
        Path('predictions/hybrid/ENSG00000169239')
        """
        mode_dir = self.get_mode_dir(mode, is_test)
        gene_dir = mode_dir / gene_id
        gene_dir.mkdir(parents=True, exist_ok=True)
        return gene_dir
    
    def get_gene_paths(self, mode: InferenceMode, gene_id: str, is_test: bool = False) -> GenePaths:
        """
        Get all output paths for a gene.
        
        Parameters
        ----------
        mode : InferenceMode
            Inference mode
        gene_id : str
            Gene ID
        is_test : bool
            If True, uses test subdirectory
        
        Returns
        -------
        GenePaths
            Container with all output paths
        
        Examples
        --------
        >>> registry = OutputRegistry()
        >>> paths = registry.get_gene_paths('hybrid', 'ENSG00000169239')
        >>> paths.predictions_file
        Path('predictions/hybrid/ENSG00000169239/combined_predictions.parquet')
        """
        gene_dir = self.get_gene_dir(mode, gene_id, is_test)
        
        return GenePaths(
            predictions_file=gene_dir / "combined_predictions.parquet",
            gene_dir=gene_dir,
            artifacts_dir=self.resolve('artifacts'),
            analysis_sequences_dir=self.resolve('analysis_sequences'),
            base_predictions_dir=self.resolve('base_predictions')
        )
    
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
            Full path to artifact
        
        Examples
        --------
        >>> registry = OutputRegistry()
        >>> registry.get_artifact_path('analysis_sequences', 'chunk_1_500.tsv')
        Path('predictions/spliceai_eval/meta_models/analysis_sequences/chunk_1_500.tsv')
        """
        base_dir = self.resolve(artifact_type)
        return base_dir / filename
    
    def clear_cache(self):
        """Clear path cache."""
        self._cache.clear()
    
    def __repr__(self) -> str:
        """String representation."""
        return f"OutputRegistry(config={self.config})"

