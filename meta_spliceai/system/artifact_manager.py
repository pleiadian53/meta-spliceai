"""Artifact and training data management for meta-models.

This module provides systematic management of:
1. Base model prediction artifacts (analysis_sequences, splice_positions, etc.)
2. Meta-model training datasets (feature matrices, labels)
3. Mode-based routing (production vs test)

Follows the same architectural patterns as genomic_resources for consistency.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Literal, Dict, List
import os
from datetime import datetime


# Type aliases for clarity
Mode = Literal["production", "test"]
Coverage = Literal["full_genome", "chromosome", "gene_subset"]


@dataclass
class ArtifactConfig:
    """Configuration for artifact management.
    
    Attributes
    ----------
    mode : Mode
        Execution mode: 'production' (immutable, cached) or 'test' (overwritable)
    coverage : Coverage
        Data coverage: 'full_genome', 'chromosome', or 'gene_subset'
    source : str
        Annotation source (e.g., 'ensembl', 'mane', 'gencode')
    build : str
        Genome build (e.g., 'GRCh37', 'GRCh38')
    base_model : str
        Base model name (e.g., 'spliceai', 'openspliceai')
    test_name : Optional[str]
        Test identifier for test mode artifacts
    data_root : Path
        Root directory for all data (typically 'data/')
    """
    mode: Mode = "test"  # Default to test for safety
    coverage: Coverage = "gene_subset"
    source: str = "ensembl"
    build: str = "GRCh37"
    base_model: str = "spliceai"
    test_name: Optional[str] = None
    data_root: Path = Path("data")
    
    def __post_init__(self):
        """Validate configuration and set defaults."""
        # Auto-detect mode from coverage if not explicitly set
        if self.coverage == "full_genome" and self.mode == "test":
            # Full genome coverage implies production mode unless explicitly overridden
            print("[artifact_manager] Auto-switching to production mode for full_genome coverage")
            self.mode = "production"
        
        # Generate test_name if in test mode and not provided
        if self.mode == "test" and self.test_name is None:
            self.test_name = f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Ensure data_root is a Path
        if not isinstance(self.data_root, Path):
            self.data_root = Path(self.data_root)


class ArtifactManager:
    """Manager for base model artifacts and meta-model training data.
    
    Directory structure:
    
    Production artifacts (immutable, cached):
        data/<source>/<build>/<base_model>_eval/meta_models/
        ├── full_splice_positions_enhanced.tsv
        ├── full_splice_errors.tsv
        └── analysis_sequences_*.tsv
    
    Test artifacts (ephemeral, overwritable):
        data/<source>/<build>/<base_model>_eval/tests/<test_name>/
        ├── sampled_genes.tsv
        └── meta_models/
            └── predictions/
    
    Parameters
    ----------
    config : ArtifactConfig
        Configuration for artifact management
    
    Examples
    --------
    # Production mode (full genome)
    >>> manager = ArtifactManager(ArtifactConfig(
    ...     mode="production",
    ...     coverage="full_genome",
    ...     source="ensembl",
    ...     build="GRCh37",
    ...     base_model="spliceai"
    ... ))
    >>> manager.get_artifacts_dir()
    Path('data/ensembl/GRCh37/spliceai_eval/meta_models')
    
    # Test mode (gene subset)
    >>> manager = ArtifactManager(ArtifactConfig(
    ...     mode="test",
    ...     test_name="comprehensive_test",
    ...     source="ensembl",
    ...     build="GRCh37",
    ...     base_model="spliceai"
    ... ))
    >>> manager.get_artifacts_dir()
    Path('data/ensembl/GRCh37/spliceai_eval/tests/comprehensive_test/meta_models/predictions')
    """
    
    def __init__(self, config: ArtifactConfig):
        self.config = config
        
        # Base directory: data/<source>/<build>/<base_model>_eval
        self.base_dir = (
            self.config.data_root / 
            self.config.source / 
            self.config.build / 
            f"{self.config.base_model}_eval"
        )
        
        # Mode-specific directories
        if self.config.mode == "production":
            self.artifacts_dir = self.base_dir / "meta_models"
        else:  # test mode
            self.artifacts_dir = (
                self.base_dir / 
                "tests" / 
                self.config.test_name / 
                "meta_models" / 
                "predictions"
            )
    
    def get_artifacts_dir(self, create: bool = False) -> Path:
        """Get the artifacts directory.
        
        Parameters
        ----------
        create : bool, default=False
            If True, create the directory if it doesn't exist
            
        Returns
        -------
        Path
            Path to artifacts directory
        """
        if create:
            self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        return self.artifacts_dir
    
    def get_artifact_path(self, artifact_name: str, create_dir: bool = False) -> Path:
        """Get path to a specific artifact.
        
        Parameters
        ----------
        artifact_name : str
            Name of the artifact file (e.g., 'full_splice_positions_enhanced.tsv')
        create_dir : bool, default=False
            If True, create parent directory if it doesn't exist
            
        Returns
        -------
        Path
            Full path to the artifact
        """
        artifacts_dir = self.get_artifacts_dir(create=create_dir)
        return artifacts_dir / artifact_name
    
    def should_overwrite(self, artifact_path: Path, force: bool = False) -> bool:
        """Determine if an artifact should be overwritten.
        
        Parameters
        ----------
        artifact_path : Path
            Path to the artifact
        force : bool, default=False
            If True, always overwrite regardless of mode
            
        Returns
        -------
        bool
            True if artifact should be overwritten
            
        Logic
        -----
        - Test mode: Always overwrite (unless artifact doesn't exist)
        - Production mode: Only overwrite if force=True
        """
        if force:
            return True
        
        if not artifact_path.exists():
            return True  # Always write if doesn't exist
        
        if self.config.mode == "test":
            return True  # Test artifacts are always overwritable
        
        # Production mode: don't overwrite by default
        return False
    
    def list_artifacts(self, pattern: str = "*.tsv") -> List[Path]:
        """List all artifacts matching a pattern.
        
        Parameters
        ----------
        pattern : str, default='*.tsv'
            Glob pattern for matching files
            
        Returns
        -------
        List[Path]
            List of artifact paths
        """
        artifacts_dir = self.get_artifacts_dir()
        if not artifacts_dir.exists():
            return []
        return sorted(artifacts_dir.glob(pattern))
    
    def get_training_data_dir(self, create: bool = False) -> Path:
        """Get directory for meta-model training data.
        
        Training data is always stored at the production level, regardless of mode.
        
        Parameters
        ----------
        create : bool, default=False
            If True, create the directory if it doesn't exist
            
        Returns
        -------
        Path
            Path to training data directory
        """
        training_dir = self.base_dir / "training_data"
        if create:
            training_dir.mkdir(parents=True, exist_ok=True)
        return training_dir
    
    def get_model_checkpoint_dir(self, model_version: str = "latest", create: bool = False) -> Path:
        """Get directory for meta-model checkpoints.
        
        Parameters
        ----------
        model_version : str, default='latest'
            Model version identifier
        create : bool, default=False
            If True, create the directory if it doesn't exist
            
        Returns
        -------
        Path
            Path to model checkpoint directory
        """
        checkpoint_dir = self.base_dir / "models" / model_version
        if create:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
        return checkpoint_dir
    
    def get_summary(self) -> Dict[str, any]:
        """Get a summary of the artifact manager configuration.
        
        Returns
        -------
        Dict[str, any]
            Summary information including paths and settings
        """
        return {
            "mode": self.config.mode,
            "coverage": self.config.coverage,
            "source": self.config.source,
            "build": self.config.build,
            "base_model": self.config.base_model,
            "test_name": self.config.test_name,
            "base_dir": str(self.base_dir),
            "artifacts_dir": str(self.artifacts_dir),
            "artifacts_exist": self.artifacts_dir.exists(),
            "artifact_count": len(self.list_artifacts()) if self.artifacts_dir.exists() else 0,
            "overwrite_policy": "always" if self.config.mode == "test" else "explicit_only"
        }
    
    def print_summary(self):
        """Print a formatted summary of the artifact manager configuration."""
        summary = self.get_summary()
        print("\n" + "="*80)
        print("ARTIFACT MANAGER CONFIGURATION")
        print("="*80)
        print(f"Mode:            {summary['mode']}")
        print(f"Coverage:        {summary['coverage']}")
        print(f"Source:          {summary['source']}")
        print(f"Build:           {summary['build']}")
        print(f"Base Model:      {summary['base_model']}")
        if summary['test_name']:
            print(f"Test Name:       {summary['test_name']}")
        print(f"\nBase Directory:      {summary['base_dir']}")
        print(f"Artifacts Directory: {summary['artifacts_dir']}")
        print(f"Artifacts Exist:     {summary['artifacts_exist']}")
        if summary['artifacts_exist']:
            print(f"Artifact Count:      {summary['artifact_count']}")
        print(f"Overwrite Policy:    {summary['overwrite_policy']}")
        print("="*80 + "\n")


def create_artifact_manager_from_workflow_config(
    mode: Optional[Mode] = None,
    coverage: Optional[Coverage] = None,
    source: str = "ensembl",
    build: str = "GRCh37",
    base_model: str = "spliceai",
    test_name: Optional[str] = None,
    data_root: Optional[Path] = None,
    **kwargs
) -> ArtifactManager:
    """Factory function to create ArtifactManager from workflow parameters.
    
    This is a convenience function for creating an ArtifactManager from
    workflow configuration parameters.
    
    Parameters
    ----------
    mode : Mode, optional
        Execution mode. If None, inferred from coverage.
    coverage : Coverage, optional
        Data coverage. If None, defaults to 'gene_subset'.
    source : str, default='ensembl'
        Annotation source
    build : str, default='GRCh37'
        Genome build
    base_model : str, default='spliceai'
        Base model name
    test_name : str, optional
        Test identifier for test mode
    data_root : Path, optional
        Root directory for data. If None, uses 'data/'.
    **kwargs
        Additional parameters (ignored)
        
    Returns
    -------
    ArtifactManager
        Configured artifact manager
    """
    # Set defaults
    if coverage is None:
        coverage = "gene_subset"
    
    if mode is None:
        # Infer mode from coverage
        mode = "production" if coverage == "full_genome" else "test"
    
    if data_root is None:
        # Try to get from system config
        try:
            from meta_spliceai.system.config import Config
            data_root = Path(Config.PROJ_DIR) / "data"
        except (ImportError, AttributeError):
            data_root = Path("data")
    
    config = ArtifactConfig(
        mode=mode,
        coverage=coverage,
        source=source,
        build=build,
        base_model=base_model,
        test_name=test_name,
        data_root=data_root
    )
    
    return ArtifactManager(config)

