#!/usr/bin/env python3
"""
Systematic Data Resource Manager for Inference Workflow

This module provides a centralized way to locate and manage genomic data resources
for the inference workflow, following a systematic directory structure.

DIRECTORY STRUCTURE:
This manager integrates with meta_spliceai/system/genomic_resources for consistent path management.

Standard structure: <data_root>/data/<source>/<version>/

1. Global shared directory: data/ensembl/
   - Fundamental inputs: GTF + FASTA files (flexible for different genome builds)
   - Uses genomic_resources system for path resolution
   
2. Base model outputs: data/ensembl/spliceai_eval/
   - Direct outputs from SpliceAI base model pass
   - Error analysis artifacts from previous workflow iterations
   - Training artifacts, test data assembly, etc.
   
3. Meta-model artifacts: data/ensembl/spliceai_eval/meta_models/
   - analysis_sequences_X_chunk_1001_1500.tsv
   - splice_positions_enhanced_*.tsv
   - Other meta-model specific artifacts
   
4. Genomic derived datasets: data/ensembl/spliceai_analysis/
   - exon_features.tsv, gene_features.tsv, transcript_features.tsv
   - overlapping_gene_counts.tsv, splice_sites.tsv

5. Case study data sources: data/ensembl/case_studies/
   - SpliceVarDB, MutSpliceDB, DBASS, ClinVar data
   - Disease-specific validation datasets

SUPPORTED GENOME BUILDS:
- GRCh37 (Ensembl GRCh37)
- GRCh38 (Ensembl GRCh38)

SUPPORTED ENSEMBL RELEASES:
- Flexible versioning (e.g., 109, 112, etc.)
- Automatically resolves correct GTF/FASTA filenames

SUPPORTED BASE MODELS:
- SpliceAI (current default)
- OpenSpliceAI (already integrated)
- Future models (as long as they produce consistent raw splice site scores)

SUPPORTED DATA SOURCES:
- SpliceVarDB (50,000+ experimentally validated splice variants)
- MutSpliceDB (TCGA/CCLE cancer mutations)
- DBASS5/DBASS3 (cryptic splice sites)
- ClinVar (clinical significance)
- Custom variant databases
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import logging

logger = logging.getLogger(__name__)


class InferenceDataResourceManager:
    """
    Systematic manager for inference workflow data resources.
    
    Provides a centralized way to locate pre-computed genomic data
    and avoid redundant processing during inference.
    
    This manager integrates with the existing meta_spliceai/system/genomic_resources
    system to ensure consistency across different genome builds and Ensembl releases.
    
    Supports multiple base models (SpliceAI, OpenSpliceAI) and data sources
    (SpliceVarDB, MutSpliceDB, ClinVar) for comprehensive case study validation.
    """
    
    def __init__(self, project_root: Optional[Union[str, Path]] = None, 
                 genome_build: str = "GRCh38", ensembl_release: str = "112"):
        """
        Initialize the data resource manager.
        
        Parameters
        ----------
        project_root : str or Path, optional
            Project root directory. If None, will attempt to auto-detect.
        genome_build : str, optional
            Genome build to use (GRCh37, GRCh38). Default: GRCh38
        ensembl_release : str, optional
            Ensembl release version. Default: 112
        """
        if project_root is None:
            project_root = self._find_project_root()
        
        self.project_root = Path(project_root).resolve()
        self.genome_build = genome_build
        self.ensembl_release = ensembl_release
        
        # Use the existing genomic resources system for consistency
        from meta_spliceai.system.genomic_resources import create_systematic_manager
        # create_systematic_manager expects build/release args, not project_root
        self.genomic_manager = create_systematic_manager()
        
        # Define systematic directory structure using genomic manager
        data_root = Path(self.genomic_manager.cfg.data_root)
        self.directories = {
            "global": data_root,
            "base_model_outputs": data_root / "spliceai_eval",
            "meta_models": data_root / "spliceai_eval" / "meta_models",
            "analysis": data_root / "spliceai_analysis",
            "case_studies": data_root / "case_studies",
        }
        
        # Define expected file patterns using genomic manager for consistency
        self.expected_files = {
            # Global shared files - use genomic manager for flexible paths via resolve()
            # Convert to Path objects to ensure compatibility with .exists() checks
            "gtf_file": Path(self.genomic_manager.resolve("gtf")),
            "genome_fasta": Path(self.genomic_manager.resolve("fasta")),
            
            # Analysis-derived files
            "gene_features": self.directories["analysis"] / "gene_features.tsv",
            "exon_features": self.directories["analysis"] / "exon_features.tsv", 
            "transcript_features": self.directories["analysis"] / "transcript_features.tsv",
            "overlapping_genes": self.directories["analysis"] / "overlapping_gene_counts.tsv",
            "splice_sites": self.directories["global"] / "splice_sites.tsv",  # Actually in main ensembl dir
            
            # Meta-model artifacts (nested one level deeper)
            "analysis_sequences": self.directories["meta_models"] / "analysis_sequences_*.tsv",
            "splice_positions": self.directories["meta_models"] / "splice_positions_enhanced_*.tsv",
        }
        
        # Log the detected configuration
        logger.info(f"📋 Data Resource Manager initialized:")
        logger.info(f"   Genome build: {self.genome_build}")
        logger.info(f"   Ensembl release: {self.ensembl_release}")
        logger.info(f"   GTF path: {self.get_gtf_path()}")
        logger.info(f"   FASTA path: {self.get_genome_fasta_path()}")
        
        logger.debug(f"Initialized InferenceDataResourceManager with root: {self.project_root}")
    
    def _find_project_root(self) -> Path:
        """Auto-detect project root by looking for characteristic files."""
        current_dir = Path(__file__).resolve()
        
        # Look for characteristic project files (build-agnostic)
        key_files = [
            "meta_spliceai/__init__.py",
            "configs/genomic_resources.yaml",
            "data/ensembl",  # Directory existence check
        ]
        
        # Start from current directory and go up
        for parent in [current_dir] + list(current_dir.parents):
            for key_file in key_files:
                if (parent / key_file).exists():
                    logger.debug(f"Found project root: {parent}")
                    return parent
        
        # Fallback: assume current working directory
        cwd = Path.cwd()
        logger.warning(f"Could not auto-detect project root, using: {cwd}")
        return cwd
    
    def get_file_path(self, file_type: str) -> Optional[Path]:
        """
        Get the path to a specific file type.
        
        Parameters
        ----------
        file_type : str
            Type of file to locate (e.g., 'gene_features', 'gtf_file')
            
        Returns
        -------
        Path or None
            Path to the file if it exists, None otherwise
        """
        if file_type not in self.expected_files:
            logger.warning(f"Unknown file type: {file_type}")
            return None
        
        file_path = self.expected_files[file_type]
        
        if file_path.exists():
            logger.debug(f"Found {file_type}: {file_path}")
            return file_path
        else:
            logger.debug(f"File not found: {file_path}")
            return None
    
    def get_analysis_sequences_files(self, target_chromosomes: Optional[List[str]] = None) -> List[Path]:
        """
        Get analysis sequences files, optionally filtered by chromosomes.
        
        Parameters
        ----------
        target_chromosomes : List[str], optional
            List of chromosomes to filter by
            
        Returns
        -------
        List[Path]
            List of matching analysis sequences files
        """
        analysis_dir = self.directories["meta_models"]  # Use the deeper meta_models directory
        if not analysis_dir.exists():
            logger.warning(f"Meta-models directory not found: {analysis_dir}")
            return []
        
        # Find all analysis_sequences files
        pattern = "analysis_sequences_*.tsv"
        all_files = list(analysis_dir.glob(pattern))
        
        if not target_chromosomes:
            return all_files
        
        # Filter by target chromosomes
        filtered_files = []
        for file_path in all_files:
            # Extract chromosome from filename (e.g., analysis_sequences_4_chunk_1_500.tsv -> 4)
            filename = file_path.stem
            if "_" in filename:
                parts = filename.split("_")
                if len(parts) >= 3:
                    chrom = parts[2]  # analysis_sequences_4_chunk_1_500 -> 4
                    if chrom in target_chromosomes:
                        filtered_files.append(file_path)
        
        logger.debug(f"Found {len(filtered_files)} analysis sequences files for chromosomes {target_chromosomes}")
        return filtered_files
    
    def check_data_availability(self, required_files: List[str]) -> Dict[str, bool]:
        """
        Check availability of required data files.
        
        Parameters
        ----------
        required_files : List[str]
            List of file types to check
            
        Returns
        -------
        Dict[str, bool]
            Dictionary mapping file types to availability status
        """
        availability = {}
        
        for file_type in required_files:
            file_path = self.get_file_path(file_type)
            availability[file_type] = file_path is not None
            
            if file_path:
                logger.info(f"✅ {file_type}: {file_path}")
            else:
                logger.warning(f"❌ {file_type}: Not found")
        
        return availability
    
    def get_overlapping_genes_path(self) -> Optional[Path]:
        """Get the path to the overlapping genes file."""
        return self.get_file_path("overlapping_genes")
    
    def get_splice_sites_path(self) -> Optional[Path]:
        """Get the path to the splice sites file."""
        return self.get_file_path("splice_sites")
    
    def get_gene_features_path(self) -> Optional[Path]:
        """Get the path to the gene features file."""
        return self.get_file_path("gene_features")
    
    def get_gtf_path(self) -> Optional[Path]:
        """Get the path to the GTF file."""
        return self.get_file_path("gtf_file")
    
    def get_genome_fasta_path(self) -> Optional[Path]:
        """Get the path to the genome FASTA file."""
        return self.get_file_path("genome_fasta")
    
    def get_base_model_outputs_dir(self, base_model: str = "spliceai") -> Path:
        """
        Get directory for base model outputs.
        
        Parameters
        ----------
        base_model : str
            Base model name (spliceai, openspliceai, etc.)
            
        Returns
        -------
        Path
            Directory for base model outputs
        """
        return self.directories["base_model_outputs"] / base_model
    
    def get_case_study_data_dir(self, data_source: str) -> Path:
        """
        Get directory for case study data sources.
        
        Parameters
        ----------
        data_source : str
            Data source name (splicevardb, mutsplicedb, dbass, clinvar)
            
        Returns
        -------
        Path
            Directory for case study data
        """
        return self.directories["case_studies"] / data_source.lower()
    
    def get_external_database_path(self, database: str, source: str = "ensembl") -> Optional[Path]:
        """
        Get path to external database files using genomic_resources system.
        
        Parameters
        ----------
        database : str
            Database name (clinvar, splicevardb, etc.)
        source : str
            Data source (ensembl, gencode, etc.)
            
        Returns
        -------
        Path or None
            Path to database file if available
        """
        try:
            # Use genomic_resources system for external databases
            return self.genomic_manager.get_external_database_path(database, source=source)
        except Exception as e:
            logger.debug(f"Could not get external database path for {database}: {e}")
            return None
    
    def validate_base_model_compatibility(self, base_model: str) -> Dict[str, bool]:
        """
        Validate that a base model is compatible with the current setup.
        
        Parameters
        ----------
        base_model : str
            Base model name to validate
            
        Returns
        -------
        Dict[str, bool]
            Validation results for base model compatibility
        """
        validation = {}
        
        # Check if base model is supported
        supported_models = ["spliceai", "openspliceai"]
        validation["supported_model"] = base_model.lower() in supported_models
        
        # Check if base model outputs exist
        base_model_dir = self.get_base_model_outputs_dir(base_model)
        validation["has_outputs"] = base_model_dir.exists()
        
        # Check for specific model requirements
        if base_model.lower() == "openspliceai":
            # OpenSpliceAI might have specific requirements
            validation["has_openspliceai_adapter"] = True  # Placeholder
        
        return validation
    
    def validate_inference_requirements(self, target_genes: List[str]) -> Dict[str, bool]:
        """
        Validate that all required data is available for inference.
        
        Parameters
        ----------
        target_genes : List[str]
            List of target genes for inference
            
        Returns
        -------
        Dict[str, bool]
            Validation results for each requirement
        """
        validation = {}
        
        # Check core data files
        core_files = ["gene_features", "splice_sites", "overlapping_genes"]
        for file_type in core_files:
            validation[f"has_{file_type}"] = self.get_file_path(file_type) is not None
        
        # Check analysis sequences for target genes
        if target_genes:
            # Get chromosomes for target genes
            gene_features_path = self.get_gene_features_path()
            if gene_features_path:
                try:
                    import polars as pl
                    gene_features = pl.read_csv(
                        str(gene_features_path),
                        separator="\t",
                        schema_overrides={"chrom": pl.Utf8}
                    )
                    
                    target_gene_info = gene_features.filter(
                        pl.col("gene_id").is_in(target_genes)
                    ).select(["gene_id", "chrom"])
                    
                    if not target_gene_info.is_empty():
                        target_chromosomes = target_gene_info["chrom"].unique().to_list()
                        analysis_files = self.get_analysis_sequences_files(target_chromosomes)
                        validation["has_analysis_sequences"] = len(analysis_files) > 0
                        validation["target_chromosomes"] = target_chromosomes
                    else:
                        validation["has_analysis_sequences"] = False
                        validation["target_genes_found"] = False
                        
                except Exception as e:
                    logger.error(f"Error checking analysis sequences: {e}")
                    validation["has_analysis_sequences"] = False
            else:
                validation["has_analysis_sequences"] = False
        
        return validation
    
    def get_optimized_workflow_config(self, target_genes: List[str]) -> Dict[str, any]:
        """
        Get optimized workflow configuration based on available data.
        
        Parameters
        ----------
        target_genes : List[str]
            List of target genes for inference
            
        Returns
        -------
        Dict[str, any]
            Optimized configuration for the inference workflow
        """
        validation = self.validate_inference_requirements(target_genes)
        
        config = {
            # Data paths
            "gtf_file": str(self.get_gtf_path()) if self.get_gtf_path() else None,
            "genome_fasta": str(self.get_genome_fasta_path()) if self.get_genome_fasta_path() else None,
            "gene_features_path": str(self.get_gene_features_path()) if self.get_gene_features_path() else None,
            "splice_sites_path": str(self.get_splice_sites_path()) if self.get_splice_sites_path() else None,
            "overlapping_genes_path": str(self.get_overlapping_genes_path()) if self.get_overlapping_genes_path() else None,
            
            # Processing flags
            "do_extract_annotations": not validation.get("has_gene_features", False),
            "do_extract_splice_sites": not validation.get("has_splice_sites", False),
            "do_find_overlaping_genes": not validation.get("has_overlapping_genes", False),
            "do_extract_sequences": not validation.get("has_analysis_sequences", False),
            
            # Target chromosomes
            "target_chromosomes": validation.get("target_chromosomes", []),
            
            # Validation status
            "data_validation": validation
        }
        
        return config

    def get_validation_info(self) -> Dict[str, Any]:
        """
        Get basic information about GTF and FASTA files for validation purposes.
        
        This provides file paths and basic metadata without performing
        comprehensive validation. For full validation, use the system-level
        genomic_validator utility.
        
        Returns
        -------
        Dict[str, Any]
            Basic file information for validation
        """
        gtf_path = self.get_gtf_path()
        fasta_path = self.get_genome_fasta_path()
        
        return {
            "gtf_path": str(gtf_path) if gtf_path else None,
            "fasta_path": str(fasta_path) if fasta_path else None,
            "genome_build": self.genome_build,
            "ensembl_release": self.ensembl_release,
            "validation_note": "Use meta_spliceai.system.genomic_validator for comprehensive validation"
        }
    
    def validate_genomic_files(self) -> Dict[str, Any]:
        """
        Convenience method to validate GTF and FASTA files using the system validator.
        
        Returns
        -------
        Dict[str, Any]
            Validation results from the system-level genomic validator
        """
        try:
            from meta_spliceai.system.genomic_validator import validate_genomic_files
            gtf_path = self.get_gtf_path()
            fasta_path = self.get_genome_fasta_path()
            
            if gtf_path and fasta_path:
                return validate_genomic_files(gtf_path, fasta_path)
            else:
                return {
                    "compatible": False,
                    "error": "GTF or FASTA file path not found",
                    "gtf_path": str(gtf_path) if gtf_path else None,
                    "fasta_path": str(fasta_path) if fasta_path else None
                }
        except ImportError as e:
            return {
                "compatible": False,
                "error": f"Could not import system validator: {e}",
                "note": "Use meta_spliceai.system.genomic_validator directly"
            }
    



def create_inference_data_manager(project_root: Optional[Union[str, Path]] = None,
                                 genome_build: str = "GRCh38",
                                 ensembl_release: str = "112",
                                 auto_detect: bool = True) -> InferenceDataResourceManager:
    """
    Factory function to create an inference data resource manager.
    
    Parameters
    ----------
    project_root : str or Path, optional
        Project root directory
    genome_build : str, optional
        Genome build to use (GRCh37, GRCh38). Default: GRCh38
    ensembl_release : str, optional
        Ensembl release version. Default: 112
    auto_detect : bool, optional
        Whether to auto-detect configuration from genomic_resources system. Default: True
        
    Returns
    -------
    InferenceDataResourceManager
        Configured data resource manager
    """
    # Auto-detect configuration from genomic_resources system if requested
    if auto_detect:
        try:
            from meta_spliceai.system.genomic_resources import create_systematic_manager
            auto_manager = create_systematic_manager()
            # Registry API: access via cfg attribute
            detected_build = auto_manager.cfg.genome_build.value if hasattr(auto_manager.cfg, 'genome_build') else genome_build
            detected_release = auto_manager.cfg.ensembl_release if hasattr(auto_manager.cfg, 'ensembl_release') else ensembl_release
            
            # Use detected values if they differ from defaults
            if detected_build != genome_build or detected_release != ensembl_release:
                logger.info(f"🔄 Auto-detected configuration from genomic_resources:")
                logger.info(f"   Genome build: {genome_build} → {detected_build}")
                logger.info(f"   Ensembl release: {ensembl_release} → {detected_release}")
                genome_build = detected_build
                ensembl_release = detected_release
        except Exception as e:
            logger.warning(f"⚠️ Auto-detection failed, using provided defaults: {e}")
    
    return InferenceDataResourceManager(
        project_root=project_root,
        genome_build=genome_build,
        ensembl_release=ensembl_release
    )
