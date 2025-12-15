"""
Output Configuration

Configuration for prediction outputs, following the pattern of genomic_resources.config

Created: 2025-10-28
"""

from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import os


@dataclass
class OutputConfig:
    """
    Configuration for prediction outputs.
    
    Follows the same pattern as genomic_resources.Config for consistency.
    
    Attributes
    ----------
    predictions_base : Path
        Base directory for all predictions (default: PROJECT_ROOT/predictions)
    base_model_name : str
        Name of base model for organizing artifacts (default: 'spliceai')
        Examples: 'spliceai', 'openspliceai', 'pangolin'
        Results in: predictions/{base_model_name}_eval/meta_models/
    artifacts_base : Path
        Base directory for artifacts (default: predictions/{base_model_name}_eval/meta_models)
        If explicitly set, overrides base_model_name
    use_project_root : bool
        If True, resolve paths relative to project root
    """
    
    predictions_base: Optional[Path] = None
    base_model_name: str = "spliceai"  # Configurable base model name
    artifacts_base: Optional[Path] = None
    use_project_root: bool = True
    
    def __post_init__(self):
        """Initialize paths with defaults if not provided."""
        
        # Get project root
        if self.use_project_root:
            project_root = self._get_project_root()
        else:
            project_root = Path.cwd()
        
        # Set default predictions base
        if self.predictions_base is None:
            # Check environment variable first (like genomic_resources does)
            env_predictions = os.getenv('META_SPLICEAI_PREDICTIONS')
            if env_predictions:
                self.predictions_base = Path(env_predictions)
            else:
                self.predictions_base = project_root / "predictions"
        else:
            self.predictions_base = Path(self.predictions_base)
        
        # Set default artifacts base using configurable base model name
        if self.artifacts_base is None:
            # Check environment variable for base model override
            env_base_model = os.getenv('META_SPLICEAI_BASE_MODEL')
            if env_base_model:
                self.base_model_name = env_base_model
            
            # CRITICAL: artifacts_base is under predictions/, not a separate directory
            # Structure: predictions/{base_model_name}_eval/meta_models/
            self.artifacts_base = self.predictions_base / f"{self.base_model_name}_eval" / "meta_models"
        else:
            self.artifacts_base = Path(self.artifacts_base)
        
        # Create directories if they don't exist
        self.predictions_base.mkdir(parents=True, exist_ok=True)
        self.artifacts_base.mkdir(parents=True, exist_ok=True)
    
    def _get_project_root(self) -> Path:
        """Get project root directory."""
        # Look for pyproject.toml or setup.py
        current = Path(__file__).resolve()
        
        for parent in current.parents:
            if (parent / "pyproject.toml").exists() or (parent / "setup.py").exists():
                return parent
        
        # Fallback to 3 levels up from this file
        return Path(__file__).resolve().parents[3]
    
    @classmethod
    def from_env(cls) -> 'OutputConfig':
        """
        Create config from environment variables.
        
        Environment Variables
        ---------------------
        META_SPLICEAI_PREDICTIONS : str
            Base directory for predictions
        META_SPLICEAI_BASE_MODEL : str
            Base model name (e.g., 'spliceai', 'openspliceai', 'pangolin')
        META_SPLICEAI_ARTIFACTS : str
            Base directory for artifacts (overrides base_model_name if set)
        
        Returns
        -------
        OutputConfig
            Configuration from environment
        
        Examples
        --------
        # Use default SpliceAI
        export META_SPLICEAI_PREDICTIONS=/mnt/predictions
        
        # Use OpenSpliceAI as base model
        export META_SPLICEAI_BASE_MODEL=openspliceai
        # → artifacts: /mnt/predictions/openspliceai_eval/meta_models/
        
        # Override everything
        export META_SPLICEAI_ARTIFACTS=/custom/artifacts
        """
        predictions_base = os.getenv('META_SPLICEAI_PREDICTIONS')
        base_model_name = os.getenv('META_SPLICEAI_BASE_MODEL', 'spliceai')
        artifacts_base = os.getenv('META_SPLICEAI_ARTIFACTS')
        
        return cls(
            predictions_base=Path(predictions_base) if predictions_base else None,
            base_model_name=base_model_name,
            artifacts_base=Path(artifacts_base) if artifacts_base else None
        )
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"OutputConfig(\n"
            f"  predictions_base={self.predictions_base},\n"
            f"  base_model_name='{self.base_model_name}',\n"
            f"  artifacts_base={self.artifacts_base}\n"
            f")"
        )

