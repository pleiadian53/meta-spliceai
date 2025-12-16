"""
Configuration management for OpenSpliceAI adapter.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from pathlib import Path
import os

@dataclass
class OpenSpliceAIAdapterConfig:
    """Configuration for OpenSpliceAI integration with MetaSpliceAI."""
    
    # Input data paths (compatible with existing MetaSpliceAI defaults)
    gtf_file: str = "data/ensembl/Homo_sapiens.GRCh38.112.gtf"
    genome_fasta: str = "data/ensembl/Homo_sapiens.GRCh38.dna.primary_assembly.fa"
    
    # Output configuration
    output_dir: str = "data/openspliceai_processed"
    
    # OpenSpliceAI parameters
    flanking_size: int = 400  # Context window size (80, 400, 2000, 10000)
    biotype: str = "protein-coding"  # protein-coding, non-coding, all
    parse_type: str = "canonical"  # canonical, all_isoforms
    
    # Data processing parameters
    canonical_only: bool = False
    remove_paralogs: bool = False
    min_identity: float = 0.8  # For paralog removal
    min_coverage: float = 0.5  # For paralog removal
    
    # Chromosome and gene filtering
    chromosomes: Optional[List[str]] = None  # None = all chromosomes
    target_genes: Optional[List[str]] = None  # None = all genes
    
    # Quality control
    verify_h5: bool = True
    write_fasta: bool = False
    
    # Integration with MetaSpliceAI workflow
    preserve_transcript_ids: bool = True
    include_context_features: bool = True
    output_format: str = "hdf5"  # hdf5, parquet, tsv
    
    # Batch processing
    batch_size: int = 1000  # Genes per batch
    chunk_size: int = 100   # For data loading
    
    def __post_init__(self):
        """Validate and normalize configuration after initialization."""
        # Convert relative paths to absolute paths
        if not os.path.isabs(self.gtf_file):
            self.gtf_file = os.path.abspath(self.gtf_file)
        if not os.path.isabs(self.genome_fasta):
            self.genome_fasta = os.path.abspath(self.genome_fasta)
        if not os.path.isabs(self.output_dir):
            self.output_dir = os.path.abspath(self.output_dir)
            
        # Validate flanking size
        valid_flanking_sizes = [80, 400, 2000, 10000]
        if self.flanking_size not in valid_flanking_sizes:
            raise ValueError(f"flanking_size must be one of {valid_flanking_sizes}, got {self.flanking_size}")
            
        # Validate biotype
        valid_biotypes = ["protein-coding", "non-coding", "all"]
        if self.biotype not in valid_biotypes:
            raise ValueError(f"biotype must be one of {valid_biotypes}, got {self.biotype}")
            
        # Validate parse_type
        valid_parse_types = ["canonical", "all_isoforms"]
        if self.parse_type not in valid_parse_types:
            raise ValueError(f"parse_type must be one of {valid_parse_types}, got {self.parse_type}")
            
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
    
    def to_openspliceai_args(self) -> Dict[str, Any]:
        """Convert configuration to OpenSpliceAI command-line arguments format."""
        return {
            'annotation_gff': self.gtf_file,
            'genome_fasta': self.genome_fasta,
            'output_dir': self.output_dir,
            'parse_type': self.parse_type,
            'biotype': self.biotype,
            'flanking_size': self.flanking_size,
            'canonical_only': self.canonical_only,
            'remove_paralogs': self.remove_paralogs,
            'min_identity': self.min_identity,
            'min_coverage': self.min_coverage,
            'verify_h5': self.verify_h5,
            'write_fasta': self.write_fasta
        }
    
    def get_openspliceai_constants(self) -> Dict[str, int]:
        """Get OpenSpliceAI constants based on flanking size."""
        # Map flanking sizes to OpenSpliceAI constants
        flanking_to_constants = {
            80: {'CL_max': 160, 'SL': 400},
            400: {'CL_max': 800, 'SL': 2000}, 
            2000: {'CL_max': 4000, 'SL': 5000},
            10000: {'CL_max': 10000, 'SL': 5000}
        }
        
        return flanking_to_constants.get(self.flanking_size, {'CL_max': 10000, 'SL': 5000})
    
    @classmethod
    def from_splicesurveyor_config(cls, splicesurveyor_config: Any) -> 'OpenSpliceAIAdapterConfig':
        """Create adapter config from existing MetaSpliceAI configuration."""
        # Extract relevant parameters from MetaSpliceAI config
        config_dict = {}
        
        if hasattr(splicesurveyor_config, 'gtf_file'):
            config_dict['gtf_file'] = splicesurveyor_config.gtf_file
        if hasattr(splicesurveyor_config, 'genome_fasta'):
            config_dict['genome_fasta'] = splicesurveyor_config.genome_fasta
        if hasattr(splicesurveyor_config, 'eval_dir'):
            config_dict['output_dir'] = os.path.join(splicesurveyor_config.eval_dir, 'openspliceai_processed')
        if hasattr(splicesurveyor_config, 'chromosomes'):
            config_dict['chromosomes'] = splicesurveyor_config.chromosomes
            
        return cls(**config_dict)
