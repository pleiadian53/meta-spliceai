"""
Output Resources Management

Centralized management of prediction outputs and artifacts, following the same
pattern as genomic_resources for consistency across the system.

Key Components:
- OutputRegistry: Central registry for output paths (like genomic Registry)
- OutputManager: Creates and manages output directory structure
- OutputConfig: Configuration for output locations

Created: 2025-10-28
"""

from pathlib import Path
from typing import Optional

# Import core components
from .registry import OutputRegistry
from .config import OutputConfig
from .manager import OutputManager

# Version
__version__ = "1.0.0"

# Global registry instance (singleton pattern, like genomic_resources)
_global_registry: Optional[OutputRegistry] = None


def get_output_registry() -> OutputRegistry:
    """
    Get the global output registry instance (singleton).
    
    Returns
    -------
    OutputRegistry
        Global output registry
    
    Examples
    --------
    >>> registry = get_output_registry()
    >>> paths = registry.get_gene_output_paths("hybrid", "ENSG00000169239")
    """
    global _global_registry
    
    if _global_registry is None:
        # Initialize with default config
        config = OutputConfig()
        _global_registry = OutputRegistry(config)
    
    return _global_registry


def create_output_manager(
    mode: str,
    is_test: bool = False,
    base_dir: Optional[Path] = None,
    base_model_name: str = "spliceai"
) -> OutputManager:
    """
    Create an OutputManager with consistent configuration.
    
    Parameters
    ----------
    mode : str
        Inference mode (base_only, hybrid, meta_only)
    is_test : bool
        If True, outputs go to tests/ subdirectory
    base_dir : Optional[Path]
        Override default base directory
    base_model_name : str
        Name of base model (default: 'spliceai')
        Examples: 'spliceai', 'openspliceai', 'pangolin'
        Results in: predictions/{base_model_name}_eval/meta_models/
    
    Returns
    -------
    OutputManager
        Configured output manager
    
    Examples
    --------
    # Default SpliceAI
    >>> manager = create_output_manager("hybrid", is_test=False)
    >>> paths = manager.get_gene_output_paths("ENSG00000169239")
    
    # OpenSpliceAI
    >>> manager = create_output_manager("hybrid", base_model_name="openspliceai")
    >>> # artifacts: predictions/openspliceai_eval/meta_models/
    """
    registry = get_output_registry()
    
    if base_dir or base_model_name != "spliceai":
        # Override default
        config = OutputConfig(
            predictions_base=base_dir,
            base_model_name=base_model_name
        )
        registry = OutputRegistry(config)
    
    return OutputManager(
        registry=registry,
        mode=mode,
        is_test=is_test
    )


# Convenience imports
__all__ = [
    'OutputRegistry',
    'OutputConfig',
    'OutputManager',
    'get_output_registry',
    'create_output_manager',
]

