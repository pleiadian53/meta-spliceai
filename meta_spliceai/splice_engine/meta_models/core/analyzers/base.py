"""
Base analyzer class for the splice-surveyor project.

This module defines the foundation Analyzer class that provides 
standardized paths and configuration used by specialized analyzers.

Todo
----
1. Apply @dataclass to this class
2. 
"""

import os
from typing import Dict, List, Optional, Tuple, Union, Any

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
    
    # Determine prefix dynamically from system config
    try:
        from meta_spliceai.system.config import Config as SystemConfig
        prefix = SystemConfig.PROJ_DIR
    except (ImportError, AttributeError):
        try:
            from meta_spliceai.system.config import find_project_root
            prefix = os.getenv("META_SPLICEAI_ROOT", find_project_root())
        except ImportError:
            from pathlib import Path
            prefix = os.getenv("META_SPLICEAI_ROOT", str(Path(__file__).parent.parent.parent.parent))
    
    blob_prefix = os.getenv("META_SPLICEAI_BLOB", "/mnt/nfs1/meta-spliceai")
    data_dir = os.path.join(prefix, "data", "ensembl")
    shared_dir = data_dir 
    
    # Paths to genomic data files
    # Note: Uses systematic path management via Registry
    try:
        from meta_spliceai.system.genomic_resources import Registry
        _genomic_manager = Registry()
        gtf_file = str(_genomic_manager.resolve("gtf"))
        genome_fasta = str(_genomic_manager.resolve("fasta"))
    except Exception:
        # Fallback to constructed paths if Registry not available
        gtf_file = os.path.join(data_dir, "Homo_sapiens.GRCh38.112.gtf")
        genome_fasta = os.path.join(data_dir, "Homo_sapiens.GRCh38.dna.primary_assembly.fa")
    
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
