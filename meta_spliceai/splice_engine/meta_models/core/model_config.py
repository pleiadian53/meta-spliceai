"""
Model configuration classes for splice site prediction.

This module defines the configuration hierarchy for base splice prediction models.
BaseModelConfig serves as an abstract base class, with model-specific configurations
(SpliceAIConfig, OpenSpliceAIConfig) inheriting from it to provide tailored defaults
and parameters for each model.

The design supports extensibility for custom models while maintaining backward
compatibility with existing code.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod
import os


@dataclass
class BaseModelConfig(ABC):
    """
    Abstract base configuration for all splice site prediction models.
    
    This class defines the common parameters required by all models, regardless
    of their specific implementation. Model-specific parameters should be defined
    in subclasses.
    
    Parameters
    ----------
    genome_fasta : str
        Path to reference genome FASTA file
    gtf_file : str
        Path to gene annotation file (GTF or GFF3)
    eval_dir : str
        Output directory for evaluation results
    mode : str
        Execution mode: 'test' or 'production'
    coverage : str
        Coverage type: 'sample', 'full_genome', 'chromosome', 'gene_subset'
    verbosity : int
        Output verbosity level (0=minimal, 1=normal, 2=detailed)
    
    Notes
    -----
    This is an abstract base class. Use model-specific subclasses:
    - SpliceAIConfig for SpliceAI (GRCh37/Ensembl)
    - OpenSpliceAIConfig for OpenSpliceAI (GRCh38/MANE)
    - Or create your own for custom models
    
    Examples
    --------
    Use specific config classes:
    
    >>> config = SpliceAIConfig(
    ...     genome_fasta='genome.fa',
    ...     gtf_file='annotations.gtf',
    ...     eval_dir='output/'
    ... )
    
    Or use factory:
    
    >>> config = create_model_config(
    ...     base_model='openspliceai',
    ...     genome_fasta='genome.fa',
    ...     gtf_file='annotations.gff'
    ... )
    """
    
    # Data paths (required)
    genome_fasta: Optional[str] = None
    gtf_file: Optional[str] = None
    eval_dir: str = ''
    shared_dir: str = ''
    
    # Execution control
    mode: str = 'test'
    coverage: str = 'gene_subset'
    test_name: Optional[str] = None
    
    # Directory management
    local_dir: Optional[str] = None
    output_subdir: str = 'meta_models'
    
    # File format settings
    format: str = 'tsv'
    seq_format: str = 'parquet'
    seq_mode: str = 'gene'
    seq_type: str = 'full'
    separator: str = '\t'
    
    # Data preparation flags
    do_extract_annotations: bool = False
    do_extract_sequences: bool = False
    do_extract_splice_sites: bool = False
    do_find_overlaping_genes: bool = False
    use_precomputed_overlapping_genes: bool = False
    
    # Performance tuning
    mini_batch_size: int = 50
    chunk_size: int = 500
    
    # Processing settings
    test_mode: bool = False
    chromosomes: Optional[List[str]] = None
    
    # Workflow settings
    use_threeprobabilities: bool = True
    save_example_sequences: bool = True
    save_nucleotide_scores: bool = False
    
    @property
    @abstractmethod
    def base_model(self) -> str:
        """Return the base model identifier."""
        pass
    
    @abstractmethod
    def get_model_specific_params(self) -> Dict[str, Any]:
        """
        Return model-specific parameters.
        
        Returns
        -------
        dict
            Dictionary of model-specific parameters that differentiate
            this model from others (e.g., threshold, context, etc.)
        """
        pass
    
    def get_full_eval_dir(self) -> str:
        """Get the full evaluation directory path including the output subdirectory."""
        return os.path.join(self.eval_dir, self.output_subdir)
    
    def get_analyzer(self):
        """Get an Analyzer instance with this configuration's paths."""
        from meta_spliceai.splice_engine.meta_models.core.analyzer import Analyzer
        return Analyzer(
            gtf_file=self.gtf_file,
            genome_fasta=self.genome_fasta,
            eval_dir=self.eval_dir
        )
    
    def get_artifact_manager(self):
        """Get an ArtifactManager for this configuration."""
        from meta_spliceai.system.artifact_manager import create_artifact_manager_from_workflow_config
        from pathlib import Path
        from meta_spliceai.system.config import Config
        
        # Get base model from subclass
        base_model = self.base_model.lower()
        
        # Infer source and build from base model
        if base_model == 'openspliceai':
            build = 'GRCh38'
            source = 'mane'
        else:
            build = 'GRCh37'
            source = 'ensembl'
        
        # Check for source specification in GTF path
        if self.gtf_file:
            if 'mane' in self.gtf_file.lower():
                source = 'mane'
            elif 'gencode' in self.gtf_file.lower():
                source = 'gencode'
        
        # Get data root
        try:
            data_root = Path(Config.PROJ_DIR) / 'data'
        except (ImportError, AttributeError):
            data_root = Path('data')
        
        return create_artifact_manager_from_workflow_config(
            mode=self.mode,
            coverage=self.coverage,
            source=source,
            build=build,
            base_model=base_model,
            test_name=self.test_name,
            data_root=data_root
        )


@dataclass
class SpliceAIConfig(BaseModelConfig):
    """
    Configuration for SpliceAI model.
    
    SpliceAI-specific parameters for splice site prediction using the
    SpliceAI deep learning model (Jaganathan et al., 2019).
    
    Parameters
    ----------
    threshold : float
        Minimum probability threshold for splice site calling
    consensus_window : int
        Window size for consensus calling
    error_window : int
        Window size for error tolerance in evaluation
    
    Notes
    -----
    SpliceAI was trained on GRCh37 (Ensembl annotations).
    For best results, use:
    - genome_fasta: Ensembl GRCh37 primary assembly
    - gtf_file: Ensembl release 87 GTF
    
    If paths are not provided, they will be automatically resolved
    using the genomic resources Registry.
    
    See Also
    --------
    OpenSpliceAIConfig : For OpenSpliceAI model (GRCh38)
    BaseModelConfig : Abstract base class
    
    Examples
    --------
    Basic usage with auto-resolution:
    
    >>> config = SpliceAIConfig()  # Uses defaults from Registry
    
    Custom parameters:
    
    >>> config = SpliceAIConfig(
    ...     genome_fasta='/path/to/GRCh37.fa',
    ...     gtf_file='/path/to/annotations.gtf',
    ...     threshold=0.3,  # More sensitive
    ...     consensus_window=5
    ... )
    """
    
    # Prediction parameters
    threshold: float = 0.5
    consensus_window: int = 2
    error_window: int = 500
    
    # Position adjustments
    use_auto_position_adjustments: bool = True
    
    def __post_init__(self):
        """Initialize derived values and set defaults from Registry."""
        from meta_spliceai.splice_engine.meta_models.core.analyzer import Analyzer
        from meta_spliceai.system.genomic_resources import Registry
        
        # Check if we need to set defaults
        if self.genome_fasta is None or self.gtf_file is None or not self.eval_dir:
            # Get defaults from Analyzer (which uses GRCh37)
            analyzer = Analyzer()
            
            if self.genome_fasta is None:
                self.genome_fasta = analyzer.genome_fasta
            if self.gtf_file is None:
                self.gtf_file = analyzer.gtf_file
            if not self.eval_dir:
                self.eval_dir = analyzer.eval_dir
            if not self.shared_dir:
                self.shared_dir = analyzer.shared_dir
        
        # Derive local_dir from eval_dir if not provided
        if self.local_dir is None:
            self.local_dir = os.path.dirname(self.eval_dir)
        
        # Auto-detect mode from coverage
        if self.coverage == 'full_genome' and self.mode == 'test':
            self.mode = 'production'
        
        # Generate test_name if needed
        if self.mode == 'test' and self.test_name is None:
            from datetime import datetime
            self.test_name = f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    @property
    def base_model(self) -> str:
        return 'spliceai'
    
    def get_model_specific_params(self) -> Dict[str, Any]:
        return {
            'threshold': self.threshold,
            'consensus_window': self.consensus_window,
            'error_window': self.error_window,
            'use_auto_position_adjustments': self.use_auto_position_adjustments
        }


@dataclass
class OpenSpliceAIConfig(BaseModelConfig):
    """
    Configuration for OpenSpliceAI model.
    
    OpenSpliceAI-specific parameters for splice site prediction using
    the OpenSpliceAI model (trained on GRCh38 with MANE annotations).
    
    Parameters
    ----------
    threshold : float
        Minimum probability threshold for splice site calling
    consensus_window : int
        Window size for consensus calling
    error_window : int
        Window size for error tolerance in evaluation
    
    Notes
    -----
    OpenSpliceAI was trained on GRCh38 (MANE annotations).
    For best results, use:
    - genome_fasta: RefSeq GRCh38.p14 FASTA
    - gtf_file: MANE v1.3 GFF3
    
    If paths are not provided, they will be automatically resolved
    using the genomic resources Registry.
    
    See Also
    --------
    SpliceAIConfig : For SpliceAI model (GRCh37)
    BaseModelConfig : Abstract base class
    
    Examples
    --------
    Basic usage with auto-resolution:
    
    >>> config = OpenSpliceAIConfig()  # Uses MANE/GRCh38 defaults
    
    Custom parameters:
    
    >>> config = OpenSpliceAIConfig(
    ...     genome_fasta='/path/to/GRCh38.fna',
    ...     gtf_file='/path/to/MANE.gff',
    ...     threshold=0.3,  # More sensitive
    ...     consensus_window=5
    ... )
    """
    
    # Prediction parameters (same as SpliceAI for now)
    threshold: float = 0.5
    consensus_window: int = 2
    error_window: int = 500
    
    # Position adjustments
    use_auto_position_adjustments: bool = True
    
    def __post_init__(self):
        """Initialize derived values and set defaults from Registry."""
        from meta_spliceai.system.genomic_resources import Registry
        from pathlib import Path
        
        # Use Registry to get OpenSpliceAI-specific defaults
        if self.genome_fasta is None or self.gtf_file is None or not self.eval_dir:
            registry = Registry(build='GRCh38_MANE', release='1.3')
            
            if self.genome_fasta is None:
                self.genome_fasta = str(registry.get_fasta_path())
            if self.gtf_file is None:
                self.gtf_file = str(registry.get_gtf_path())
            if not self.eval_dir:
                self.eval_dir = str(registry.data_dir / 'openspliceai_eval')
            if not self.shared_dir:
                self.shared_dir = str(registry.data_dir)
        
        # Derive local_dir from eval_dir if not provided
        if self.local_dir is None:
            self.local_dir = os.path.dirname(self.eval_dir)
        
        # Auto-detect mode from coverage
        if self.coverage == 'full_genome' and self.mode == 'test':
            self.mode = 'production'
        
        # Generate test_name if needed
        if self.mode == 'test' and self.test_name is None:
            from datetime import datetime
            self.test_name = f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    @property
    def base_model(self) -> str:
        return 'openspliceai'
    
    def get_model_specific_params(self) -> Dict[str, Any]:
        return {
            'threshold': self.threshold,
            'consensus_window': self.consensus_window,
            'error_window': self.error_window,
            'use_auto_position_adjustments': self.use_auto_position_adjustments
        }


# ============================================================================
# FACTORY FUNCTION
# ============================================================================

def create_model_config(base_model: str, **kwargs) -> BaseModelConfig:
    """
    Factory function to create appropriate configuration based on base model.
    
    Parameters
    ----------
    base_model : str
        Model identifier: 'spliceai', 'openspliceai', or custom model name
    **kwargs
        Configuration parameters passed to the config constructor
    
    Returns
    -------
    BaseModelConfig
        Appropriate configuration subclass for the specified model
    
    Raises
    ------
    ValueError
        If base_model is not recognized and cannot be loaded
    
    Examples
    --------
    Create SpliceAI config:
    
    >>> config = create_model_config(
    ...     base_model='spliceai',
    ...     threshold=0.5
    ... )
    
    Create OpenSpliceAI config:
    
    >>> config = create_model_config(
    ...     base_model='openspliceai',
    ...     threshold=0.3
    ... )
    
    Custom model (if registered):
    
    >>> config = create_model_config(
    ...     base_model='my_custom_model',
    ...     custom_param=value
    ... )
    """
    base_model = base_model.lower()
    
    if base_model == 'spliceai':
        return SpliceAIConfig(**kwargs)
    elif base_model == 'openspliceai':
        return OpenSpliceAIConfig(**kwargs)
    else:
        # Try to load custom model config
        try:
            return _load_custom_model_config(base_model, **kwargs)
        except ImportError as e:
            raise ValueError(
                f"Unknown base model: '{base_model}'. "
                f"Supported models: 'spliceai', 'openspliceai'. "
                f"For custom models, ensure the model config is properly registered. "
                f"Error: {e}"
            )


def _load_custom_model_config(base_model: str, **kwargs) -> BaseModelConfig:
    """
    Dynamically load custom model configuration.
    
    Custom models should define their config class in:
    meta_spliceai/splice_engine/models/<model_name>/config.py
    
    The config class should inherit from BaseModelConfig and follow
    the naming convention: <ModelName>Config
    """
    import importlib
    
    try:
        # Try to import custom model config
        module_path = f"meta_spliceai.splice_engine.models.{base_model}.config"
        config_module = importlib.import_module(module_path)
        
        # Look for Config class
        # Convert model name to class name: transformer_splice → TransformerSpliceConfig
        parts = base_model.split('_')
        class_name = ''.join(word.capitalize() for word in parts) + 'Config'
        
        config_class = getattr(config_module, class_name)
        
        # Verify it's a BaseModelConfig subclass
        if not issubclass(config_class, BaseModelConfig):
            raise TypeError(
                f"Config class {class_name} must inherit from BaseModelConfig"
            )
        
        return config_class(**kwargs)
    except (ImportError, AttributeError) as e:
        raise ImportError(
            f"Could not load custom config for model '{base_model}'. "
            f"Expected: meta_spliceai/splice_engine/models/{base_model}/config.py "
            f"with class {class_name}. "
            f"Error: {e}"
        )


# ============================================================================
# BACKWARD COMPATIBILITY
# ============================================================================

# For backward compatibility during migration
# Users can still import SpliceAIConfig as before
__all__ = [
    'BaseModelConfig',
    'SpliceAIConfig',
    'OpenSpliceAIConfig',
    'create_model_config'
]

