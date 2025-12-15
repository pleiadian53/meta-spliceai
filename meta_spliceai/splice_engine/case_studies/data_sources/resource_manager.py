"""
Case Study Data Resource Manager

Extends the genomic resource manager for case study-specific data sources
including ClinVar, SpliceVarDB, MutSpliceDB, and other variant databases.

Provides systematic path management for:
- Reference genomic files (FASTA, GTF) via genomic_resources
- Case study datasets (ClinVar, SpliceVarDB, etc.)
- Processed variant files and analysis results
"""

import logging
from pathlib import Path
from typing import Optional, Union, Dict, Any
from dataclasses import dataclass

from meta_spliceai.system.genomic_resources.core import (
    GenomicResourceManager, 
    StandardGenome, 
    GenomeBuild
)

logger = logging.getLogger(__name__)


@dataclass
class CaseStudyPaths:
    """Standard paths for case study data organization."""
    
    # Base directories
    case_studies_root: Path
    clinvar: Path
    splicevardb: Path  
    mutsplicedb: Path
    dbass: Path
    custom: Path
    
    # Analysis outputs
    processed: Path
    results: Path
    normalized_vcf: Path
    
    def __post_init__(self):
        """Ensure all paths are Path objects."""
        for field_name, field_value in self.__dict__.items():
            if isinstance(field_value, str):
                setattr(self, field_name, Path(field_value))


class CaseStudyResourceManager:
    """Resource manager for case study data sources and genomic references."""
    
    def __init__(self, 
                 genome_build: Union[str, GenomeBuild] = GenomeBuild.GRCH38,
                 ensembl_release: int = 110,
                 data_root: Optional[Union[str, Path]] = None,
                 create_missing_dirs: bool = True):
        """
        Initialize case study resource manager.
        
        Parameters
        ----------
        genome_build : str or GenomeBuild
            Genome build (GRCh37, GRCh38)
        ensembl_release : int
            Ensembl release number
        data_root : str or Path, optional
            Root directory for data (auto-detected if None)
        create_missing_dirs : bool
            Whether to create missing directories
        """
        # Initialize genomic resource manager
        genome = StandardGenome(
            genome_build=genome_build if isinstance(genome_build, GenomeBuild) else GenomeBuild(genome_build),
            ensembl_release=ensembl_release,
            data_root=data_root
        )
        
        self.genomic_manager = GenomicResourceManager(
            genome=genome,
            create_missing_dirs=create_missing_dirs
        )
        
        # Set up case study paths
        self.case_study_paths = self._setup_case_study_paths()
        
        if create_missing_dirs:
            self._create_directories()
            
        logger.info(f"CaseStudyResourceManager initialized")
        logger.info(f"Genome build: {self.genomic_manager.genome.genome_build}")
        logger.info(f"Case studies root: {self.case_study_paths.case_studies_root}")
    
    def _setup_case_study_paths(self) -> CaseStudyPaths:
        """Set up standardized case study directory structure."""
        base_data = self.genomic_manager.genome.base_data_dir
        ensembl_root = base_data / "ensembl"
        case_studies_root = ensembl_root / "case_studies"
        
        return CaseStudyPaths(
            case_studies_root=case_studies_root,
            clinvar=case_studies_root / "clinvar",  # ✅ ClinVar now consistent with other variant databases
            splicevardb=case_studies_root / "splicevardb", 
            mutsplicedb=case_studies_root / "mutsplicedb",
            dbass=case_studies_root / "dbass",
            custom=case_studies_root / "custom",
            processed=case_studies_root / "processed",
            results=case_studies_root / "results",
            normalized_vcf=case_studies_root / "normalized_vcf"
        )
    
    def _create_directories(self):
        """Create case study directories if they don't exist."""
        for path in self.case_study_paths.__dict__.values():
            if isinstance(path, Path):
                path.mkdir(parents=True, exist_ok=True)
                logger.debug(f"Ensured directory exists: {path}")
    
    # Genomic reference files (delegated to genomic_resources)
    def get_reference_fasta(self, validate: bool = True) -> Path:
        """Get reference FASTA file path (alias for compatibility)."""
        return self.get_fasta_path(validate=validate)
    
    def get_fasta_path(self, validate: bool = True) -> Path:
        """
        Get reference FASTA file path.
        
        Parameters
        ----------
        validate : bool
            Whether to validate file exists
            
        Returns
        -------
        Path
            Path to reference FASTA file
        """
        return self.genomic_manager.get_fasta_path(validate=validate)
    
    def get_gtf_path(self, validate: bool = True) -> Path:
        """
        Get reference GTF file path.
        
        Parameters
        ----------
        validate : bool
            Whether to validate file exists
            
        Returns
        -------
        Path
            Path to reference GTF file
        """
        return self.genomic_manager.get_gtf_path(validate=validate)
    
    # Case study data source paths
    def get_clinvar_dir(self) -> Path:
        """Get ClinVar data directory."""
        return self.case_study_paths.clinvar
    
    def get_clinvar_data_dir(self) -> Path:
        """Get ClinVar data directory (alias for compatibility)."""
        return self.get_clinvar_dir()
    
    def get_clinvar_vcf_path(self, filename: Optional[str] = None, 
                           date: Optional[str] = None) -> Path:
        """
        Get ClinVar VCF file path.
        
        Parameters
        ----------
        filename : str, optional
            Specific VCF filename
        date : str, optional
            Date string for auto-generated filename (YYYYMMDD)
            
        Returns
        -------
        Path
            Path to ClinVar VCF file
        """
        clinvar_dir = self.get_clinvar_dir()
        vcf_dir = clinvar_dir / "vcf"  # VCF files are in the vcf/ subdirectory
        
        if filename:
            return vcf_dir / filename
        elif date:
            return vcf_dir / f"clinvar_{date}.vcf.gz"
        else:
            # Look for existing files
            vcf_files = list(vcf_dir.glob("clinvar_*.vcf.gz"))
            if vcf_files:
                # Return most recent by filename
                return sorted(vcf_files)[-1]
            else:
                # Return default path
                return vcf_dir / "clinvar_latest.vcf.gz"
    
    def get_splicevardb_dir(self) -> Path:
        """Get SpliceVarDB data directory."""
        return self.case_study_paths.splicevardb
    
    def get_mutsplicedb_dir(self) -> Path:
        """Get MutSpliceDB data directory."""
        return self.case_study_paths.mutsplicedb
    
    def get_dbass_dir(self) -> Path:
        """Get DBASS data directory.""" 
        return self.case_study_paths.dbass
    
    def get_custom_dir(self) -> Path:
        """Get custom data directory."""
        return self.case_study_paths.custom
    
    # Processing and results paths
    def get_processed_dir(self) -> Path:
        """Get processed data directory."""
        return self.case_study_paths.processed
    
    def get_results_dir(self) -> Path:
        """Get results directory."""
        return self.case_study_paths.results
    
    def get_normalized_vcf_dir(self) -> Path:
        """Get normalized VCF directory."""
        return self.case_study_paths.normalized_vcf
    
    def get_normalized_vcf_path(self, source: str, filename: Optional[str] = None) -> Path:
        """
        Get normalized VCF file path for a data source.
        
        Parameters
        ----------
        source : str
            Data source name (clinvar, custom, etc.)
        filename : str, optional
            Specific filename (auto-generated if None)
            
        Returns
        -------
        Path
            Path to normalized VCF file
        """
        normalized_dir = self.get_normalized_vcf_dir()
        
        if filename:
            return normalized_dir / filename
        else:
            return normalized_dir / f"{source}_normalized.vcf.gz"
    
    # Convenience methods for common workflows
    def get_case_study_config(self) -> Dict[str, Any]:
        """
        Get configuration dictionary for case study workflows.
        
        Returns
        -------
        Dict[str, Any]
            Configuration with paths and settings
        """
        return {
            'genome_build': str(self.genomic_manager.genome.genome_build),
            'ensembl_release': self.genomic_manager.genome.ensembl_release,
            'reference_fasta': str(self.get_fasta_path(validate=False)),
            'reference_gtf': str(self.get_gtf_path(validate=False)),
            'data_sources': {
                'clinvar': str(self.get_clinvar_dir()),
                'splicevardb': str(self.get_splicevardb_dir()),
                'mutsplicedb': str(self.get_mutsplicedb_dir()),
                'dbass': str(self.get_dbass_dir()),
                'custom': str(self.get_custom_dir())
            },
            'processing': {
                'processed': str(self.get_processed_dir()),
                'results': str(self.get_results_dir()),
                'normalized_vcf': str(self.get_normalized_vcf_dir())
            }
        }
    
    def validate_setup(self) -> Dict[str, bool]:
        """
        Validate case study setup and report status.
        
        Returns
        -------
        Dict[str, bool]
            Validation results for each component
        """
        results = {}
        
        # Check genomic references
        try:
            fasta_path = self.get_fasta_path(validate=True)
            results['reference_fasta'] = fasta_path.exists()
        except:
            results['reference_fasta'] = False
            
        try:
            gtf_path = self.get_gtf_path(validate=True)
            results['reference_gtf'] = gtf_path.exists()
        except:
            results['reference_gtf'] = False
        
        # Check directories
        for name, path in self.case_study_paths.__dict__.items():
            if isinstance(path, Path):
                results[f'dir_{name}'] = path.exists()
        
        return results
    
    def __str__(self) -> str:
        """String representation of resource manager."""
        return (f"CaseStudyResourceManager("
                f"genome_build={self.genomic_manager.genome.genome_build}, "
                f"ensembl_release={self.genomic_manager.genome.ensembl_release}, "
                f"case_studies_root={self.case_study_paths.case_studies_root})")


def create_case_study_resource_manager(**kwargs) -> CaseStudyResourceManager:
    """
    Factory function to create case study resource manager.
    
    Parameters
    ----------
    **kwargs
        Arguments passed to CaseStudyResourceManager constructor
        
    Returns
    -------
    CaseStudyResourceManager
        Configured resource manager instance
    """
    return CaseStudyResourceManager(**kwargs)
