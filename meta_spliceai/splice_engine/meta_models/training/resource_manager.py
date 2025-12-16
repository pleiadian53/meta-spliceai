#!/usr/bin/env python3
"""
Training Resource Manager for Gene-Aware CV and Meta-Model Training

This module provides systematic resource management for training workflows,
integrating with the meta_spliceai.system.genomic_resources system for
consistent path resolution across all workflows.

INTEGRATION WITH GENOMIC_RESOURCES:
- Uses systematic path organization: <data_root>/data/<source>/<version>/
- Leverages existing genomic_resources for GTF/FASTA and derived datasets
- Provides training-specific extensions for model artifacts and configurations

SYSTEMATIC PATH RESOLUTION:
1. Genomic data (via genomic_resources):
   - splice_sites.tsv (from GTF exon boundaries)
   - gene_features.tsv, transcript_features.tsv (from GTF analysis)
   - overlapping_gene_counts.tsv (from genomic analysis)

2. Training-specific data:
   - exclude_features.txt (training configurations)
   - model artifacts (results/ directories)
   - training datasets (train_*/ directories)

3. External databases (via genomic_resources):
   - ClinVar, SpliceVarDB, MutSpliceDB integration
   - Case study validation datasets
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

logger = logging.getLogger(__name__)


class TrainingResourceManager:
    """
    Systematic resource manager for meta-model training workflows.
    
    Integrates with meta_spliceai.system.genomic_resources for consistent
    path management across all workflows in the system.
    """
    
    def __init__(self, project_root: Optional[Union[str, Path]] = None,
                 genome_build: str = "GRCh38", ensembl_release: str = "112"):
        """
        Initialize the training resource manager.
        
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
        
        # Initialize genomic resources manager for systematic path resolution
        try:
            from meta_spliceai.system.genomic_resources import create_systematic_manager
            self.genomic_manager = create_systematic_manager(str(self.project_root))
            self.has_genomic_resources = True
            logger.info(f"✅ Integrated with genomic_resources system")
        except ImportError as e:
            logger.warning(f"⚠️ genomic_resources not available, using fallback paths: {e}")
            self.genomic_manager = None
            self.has_genomic_resources = False
        
        # Define systematic directory structure
        self._setup_directories()
        
        # Log configuration
        logger.info(f"📋 Training Resource Manager initialized:")
        logger.info(f"   Project root: {self.project_root}")
        logger.info(f"   Genome build: {self.genome_build}")
        logger.info(f"   Ensembl release: {self.ensembl_release}")
        logger.info(f"   Genomic resources: {'✅ Available' if self.has_genomic_resources else '❌ Fallback mode'}")
    
    def _find_project_root(self) -> Path:
        """Auto-detect project root by looking for characteristic files."""
        current_dir = Path(__file__).resolve()
        
        # Look for characteristic project files
        key_files = [
            "meta_spliceai/__init__.py",
            "data/ensembl/splice_sites.tsv",
            "results/",
            "train_pc_1000_3mers/"
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
    
    def _setup_directories(self):
        """Setup systematic directory structure."""
        if self.has_genomic_resources:
            # Use genomic_resources for systematic paths
            self.directories = {
                "ensembl_base": self.genomic_manager.genome.get_source_dir("ensembl"),
                "analysis": self.genomic_manager.genome.get_source_dir("ensembl") / "spliceai_analysis",
                "case_studies": self.genomic_manager.genome.get_source_dir("ensembl") / "case_studies",
                "configs": self.project_root / "configs",
                "results": self.project_root / "results",
                "training_datasets": self.project_root,  # train_* directories at root level
            }
        else:
            # Fallback to hardcoded paths
            self.directories = {
                "ensembl_base": self.project_root / "data" / "ensembl",
                "analysis": self.project_root / "data" / "ensembl" / "spliceai_analysis", 
                "case_studies": self.project_root / "data" / "ensembl" / "case_studies",
                "configs": self.project_root / "configs",
                "results": self.project_root / "results",
                "training_datasets": self.project_root,
            }
    
    def get_splice_sites_path(self) -> Optional[Path]:
        """
        Get path to splice sites annotation file.
        
        This file contains splice site coordinates derived from GTF exon boundaries.
        
        Returns
        -------
        Path or None
            Path to splice_sites.tsv if available
        """
        if self.has_genomic_resources:
            # Try to get from genomic_resources system first
            try:
                # Check if genomic_resources has a specific method for splice sites
                if hasattr(self.genomic_manager, 'get_splice_sites_path'):
                    return self.genomic_manager.get_splice_sites_path()
            except Exception as e:
                logger.debug(f"genomic_resources splice_sites method failed: {e}")
        
        # Fallback to systematic path
        splice_sites_path = self.directories["ensembl_base"] / "splice_sites.tsv"
        if splice_sites_path.exists():
            return splice_sites_path
        
        logger.warning(f"splice_sites.tsv not found at: {splice_sites_path}")
        return None
    
    def get_gene_features_path(self) -> Optional[Path]:
        """
        Get path to gene features file.
        
        This file contains gene-level annotations derived from GTF analysis.
        
        Returns
        -------
        Path or None
            Path to gene_features.tsv if available
        """
        gene_features_path = self.directories["analysis"] / "gene_features.tsv"
        if gene_features_path.exists():
            return gene_features_path
        
        logger.warning(f"gene_features.tsv not found at: {gene_features_path}")
        return None
    
    def get_transcript_features_path(self) -> Optional[Path]:
        """
        Get path to transcript features file.
        
        This file contains transcript-level annotations derived from GTF analysis.
        
        Returns
        -------
        Path or None
            Path to transcript_features.tsv if available
        """
        transcript_features_path = self.directories["analysis"] / "transcript_features.tsv"
        if transcript_features_path.exists():
            return transcript_features_path
        
        logger.warning(f"transcript_features.tsv not found at: {transcript_features_path}")
        return None
    
    def get_exclude_features_path(self) -> Optional[Path]:
        """
        Get path to feature exclusion configuration file.
        
        Returns
        -------
        Path or None
            Path to exclude_features.txt if available
        """
        exclude_features_path = self.directories["configs"] / "exclude_features.txt"
        if exclude_features_path.exists():
            return exclude_features_path
        
        logger.debug(f"exclude_features.txt not found at: {exclude_features_path}")
        return None
    
    def get_gtf_path(self) -> Optional[Path]:
        """Get GTF file path via genomic_resources system."""
        if self.has_genomic_resources:
            try:
                return Path(self.genomic_manager.get_gtf_path(validate=False))
            except Exception as e:
                logger.warning(f"Could not get GTF path from genomic_resources: {e}")
        
        return None
    
    def get_genome_fasta_path(self) -> Optional[Path]:
        """Get genome FASTA path via genomic_resources system."""
        if self.has_genomic_resources:
            try:
                return Path(self.genomic_manager.get_fasta_path(validate=False))
            except Exception as e:
                logger.warning(f"Could not get FASTA path from genomic_resources: {e}")
        
        return None
    
    def get_training_defaults(self) -> Dict[str, Optional[str]]:
        """
        Get systematic default paths for training workflows.
        
        This replaces hardcoded defaults in argument parsers.
        
        Returns
        -------
        Dict[str, Optional[str]]
            Dictionary with systematic default paths for training
        """
        defaults = {
            "splice_sites_path": None,
            "gene_features_path": None,
            "transcript_features_path": None,
            "exclude_features_path": None,
            "gtf_file": None,
            "genome_fasta": None
        }
        
        # Get systematic paths
        splice_sites = self.get_splice_sites_path()
        if splice_sites:
            defaults["splice_sites_path"] = str(splice_sites)
        
        gene_features = self.get_gene_features_path()
        if gene_features:
            defaults["gene_features_path"] = str(gene_features)
        
        transcript_features = self.get_transcript_features_path()
        if transcript_features:
            defaults["transcript_features_path"] = str(transcript_features)
        
        exclude_features = self.get_exclude_features_path()
        if exclude_features:
            defaults["exclude_features_path"] = str(exclude_features)
        
        gtf_path = self.get_gtf_path()
        if gtf_path:
            defaults["gtf_file"] = str(gtf_path)
        
        fasta_path = self.get_genome_fasta_path()
        if fasta_path:
            defaults["genome_fasta"] = str(fasta_path)
        
        return defaults
    
    def validate_training_resources(self) -> Dict[str, Any]:
        """
        Validate that all required training resources are available.
        
        Returns
        -------
        Dict[str, Any]
            Validation results for training resources
        """
        validation = {
            "genomic_resources_available": self.has_genomic_resources,
            "resources": {},
            "missing": [],
            "recommendations": []
        }
        
        # Check core training resources
        core_resources = {
            "splice_sites": self.get_splice_sites_path(),
            "gene_features": self.get_gene_features_path(),
            "transcript_features": self.get_transcript_features_path(),
            "gtf_file": self.get_gtf_path(),
            "genome_fasta": self.get_genome_fasta_path()
        }
        
        for resource_name, resource_path in core_resources.items():
            if resource_path and resource_path.exists():
                validation["resources"][resource_name] = {
                    "available": True,
                    "path": str(resource_path)
                }
            else:
                validation["resources"][resource_name] = {
                    "available": False,
                    "path": str(resource_path) if resource_path else None
                }
                validation["missing"].append(resource_name)
        
        # Check optional resources
        exclude_features = self.get_exclude_features_path()
        validation["resources"]["exclude_features"] = {
            "available": exclude_features is not None and exclude_features.exists(),
            "path": str(exclude_features) if exclude_features else None,
            "optional": True
        }
        
        # Generate recommendations
        if validation["missing"]:
            if not self.has_genomic_resources:
                validation["recommendations"].append(
                    "Install genomic_resources system for systematic path management"
                )
            
            if "splice_sites" in validation["missing"]:
                validation["recommendations"].append(
                    "Run GTF analysis workflow to generate splice_sites.tsv from exon boundaries"
                )
            
            if any(x in validation["missing"] for x in ["gene_features", "transcript_features"]):
                validation["recommendations"].append(
                    "Run genomic analysis workflow to generate feature files from GTF"
                )
        
        # Overall status
        critical_resources = ["splice_sites", "gene_features", "transcript_features"]
        validation["all_critical_available"] = all(
            validation["resources"][res]["available"] for res in critical_resources
        )
        
        return validation
    
    def get_model_directory(self, model_name: str) -> Path:
        """Get systematic path for model results directory."""
        return self.directories["results"] / model_name
    
    def get_training_dataset_directory(self, dataset_name: str) -> Path:
        """Get systematic path for training dataset directory."""
        if not dataset_name.startswith("train_"):
            dataset_name = f"train_{dataset_name}"
        return self.directories["training_datasets"] / dataset_name
    
    def get_case_study_directory(self, study_name: str) -> Path:
        """Get systematic path for case study data."""
        return self.directories["case_studies"] / study_name


def create_training_resource_manager(project_root: Optional[Union[str, Path]] = None,
                                   auto_detect_config: bool = True) -> TrainingResourceManager:
    """
    Factory function to create a training resource manager.
    
    Parameters
    ----------
    project_root : str or Path, optional
        Project root directory. If None, will auto-detect.
    auto_detect_config : bool, optional
        Whether to auto-detect genome build and release from genomic_resources.
        
    Returns
    -------
    TrainingResourceManager
        Configured training resource manager
    """
    genome_build = "GRCh38"
    ensembl_release = "112"
    
    # Auto-detect configuration from genomic_resources if available
    if auto_detect_config:
        try:
            from meta_spliceai.system.genomic_resources import create_systematic_manager
            auto_manager = create_systematic_manager(project_root)
            genome_build = auto_manager.genome.genome_build.value
            ensembl_release = auto_manager.genome.ensembl_release
            logger.info(f"🔄 Auto-detected from genomic_resources: {genome_build}, Ensembl {ensembl_release}")
        except Exception as e:
            logger.debug(f"Auto-detection failed, using defaults: {e}")
    
    return TrainingResourceManager(
        project_root=project_root,
        genome_build=genome_build,
        ensembl_release=ensembl_release
    )


def get_systematic_training_defaults(project_root: Optional[Union[str, Path]] = None) -> Dict[str, Optional[str]]:
    """
    Get systematic default paths for training argument parsers.
    
    This function replaces hardcoded defaults in training scripts.
    
    Parameters
    ----------
    project_root : str or Path, optional
        Project root directory
        
    Returns
    -------
    Dict[str, Optional[str]]
        Systematic default paths for training workflows
    """
    manager = create_training_resource_manager(project_root)
    return manager.get_training_defaults()


def main():
    """Command-line interface for training resource management."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Training Resource Manager")
    parser.add_argument("--validate", action="store_true", help="Validate training resources")
    parser.add_argument("--show-defaults", action="store_true", help="Show systematic default paths")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.basicConfig(level=logging.INFO)
    
    manager = create_training_resource_manager()
    
    if args.validate:
        validation = manager.validate_training_resources()
        
        print(f"\n🔍 Training Resource Validation")
        print(f"Genomic resources system: {'✅ Available' if validation['genomic_resources_available'] else '❌ Not available'}")
        
        print(f"\nResource Status:")
        for resource_name, resource_info in validation["resources"].items():
            status = "✅" if resource_info["available"] else "❌"
            optional = " (optional)" if resource_info.get("optional") else ""
            print(f"  {status} {resource_name}{optional}: {resource_info['path']}")
        
        if validation["missing"]:
            print(f"\nMissing Resources: {validation['missing']}")
        
        if validation["recommendations"]:
            print(f"\nRecommendations:")
            for rec in validation["recommendations"]:
                print(f"  💡 {rec}")
        
        print(f"\nOverall Status: {'✅ Ready for training' if validation['all_critical_available'] else '❌ Missing critical resources'}")
    
    if args.show_defaults:
        defaults = manager.get_training_defaults()
        
        print(f"\n📋 Systematic Training Defaults")
        for key, value in defaults.items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
