#!/usr/bin/env python3
"""
Systematic Default Path Provider for Training Scripts

This module provides systematic default paths for training scripts,
replacing hardcoded paths with systematic resource management.

USAGE IN TRAINING SCRIPTS:
Instead of hardcoded defaults like:
    p.add_argument("--splice-sites-path", default="data/ensembl/splice_sites.tsv")

Use systematic defaults:
    from .systematic_defaults import get_systematic_defaults
    defaults = get_systematic_defaults()
    p.add_argument("--splice-sites-path", default=defaults["splice_sites_path"])

This ensures consistent path resolution across all training workflows
and integrates with the genomic_resources system.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Union

logger = logging.getLogger(__name__)


def get_systematic_defaults(project_root: Optional[Union[str, Path]] = None) -> Dict[str, Optional[str]]:
    """
    Get systematic default paths for training script argument parsers.
    
    This function replaces hardcoded defaults with systematic resource management
    that integrates with the genomic_resources system.
    
    Parameters
    ----------
    project_root : str or Path, optional
        Project root directory. If None, will auto-detect.
        
    Returns
    -------
    Dict[str, Optional[str]]
        Dictionary with systematic default paths:
        - splice_sites_path: Path to splice site annotations
        - gene_features_path: Path to gene features
        - transcript_features_path: Path to transcript features  
        - exclude_features_path: Path to feature exclusion config
        - gtf_file: Path to GTF file
        - genome_fasta: Path to genome FASTA file
    """
    try:
        from .resource_manager import create_training_resource_manager
        manager = create_training_resource_manager(project_root)
        defaults = manager.get_training_defaults()
        
        logger.debug(f"✅ Systematic defaults loaded via resource_manager")
        return defaults
        
    except ImportError as e:
        logger.warning(f"⚠️ resource_manager not available, using fallback defaults: {e}")
        return _get_fallback_defaults(project_root)
    except Exception as e:
        logger.warning(f"⚠️ Error getting systematic defaults, using fallback: {e}")
        return _get_fallback_defaults(project_root)


def _get_fallback_defaults(project_root: Optional[Union[str, Path]] = None) -> Dict[str, Optional[str]]:
    """
    Fallback to hardcoded defaults if systematic resource management fails.
    
    Parameters
    ----------
    project_root : str or Path, optional
        Project root directory
        
    Returns
    -------
    Dict[str, Optional[str]]
        Fallback default paths
    """
    if project_root is None:
        project_root = Path.cwd()
    else:
        project_root = Path(project_root)
    
    # Use the original hardcoded paths as fallback
    fallback_defaults = {
        "splice_sites_path": str(project_root / "data/ensembl/splice_sites.tsv"),
        "gene_features_path": str(project_root / "data/ensembl/spliceai_analysis/gene_features.tsv"),
        "transcript_features_path": str(project_root / "data/ensembl/spliceai_analysis/transcript_features.tsv"),
        "exclude_features_path": str(project_root / "configs/exclude_features.txt"),
        "gtf_file": None,  # No fallback for GTF
        "genome_fasta": None  # No fallback for FASTA
    }
    
    logger.debug(f"📋 Using fallback defaults")
    return fallback_defaults


def update_argument_parser_defaults(parser, project_root: Optional[Union[str, Path]] = None):
    """
    Update an existing argument parser with systematic defaults.
    
    This is a convenience function for updating existing parsers without
    modifying their argument definitions.
    
    Parameters
    ----------
    parser : argparse.ArgumentParser
        Argument parser to update
    project_root : str or Path, optional
        Project root directory
    """
    defaults = get_systematic_defaults(project_root)
    
    # Map systematic defaults to argument names
    argument_mapping = {
        "splice_sites_path": "--splice-sites-path",
        "gene_features_path": "--gene-features-path", 
        "transcript_features_path": "--transcript-features-path",
        "exclude_features_path": "--exclude-features"
    }
    
    updated_count = 0
    for default_key, arg_name in argument_mapping.items():
        default_value = defaults.get(default_key)
        if default_value:
            try:
                # Update the default value for this argument
                for action in parser._actions:
                    if hasattr(action, 'dest') and f"--{action.dest.replace('_', '-')}" == arg_name:
                        action.default = default_value
                        updated_count += 1
                        logger.debug(f"Updated {arg_name} default to: {default_value}")
                        break
            except Exception as e:
                logger.warning(f"Could not update default for {arg_name}: {e}")
    
    if updated_count > 0:
        logger.info(f"✅ Updated {updated_count} argument defaults with systematic paths")
    else:
        logger.warning(f"⚠️ No argument defaults were updated")


def validate_systematic_paths(project_root: Optional[Union[str, Path]] = None) -> Dict[str, bool]:
    """
    Validate that systematic paths are available and accessible.
    
    Parameters
    ----------
    project_root : str or Path, optional
        Project root directory
        
    Returns
    -------
    Dict[str, bool]
        Validation results for each systematic path
    """
    try:
        from .resource_manager import create_training_resource_manager
        manager = create_training_resource_manager(project_root)
        validation = manager.validate_training_resources()
        
        # Convert to simple boolean validation
        path_validation = {}
        for resource_name, resource_info in validation["resources"].items():
            path_validation[resource_name] = resource_info["available"]
        
        return path_validation
        
    except Exception as e:
        logger.error(f"Error validating systematic paths: {e}")
        return {}


def main():
    """Command-line interface for systematic defaults."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Systematic Defaults Provider")
    parser.add_argument("--show-defaults", action="store_true", help="Show systematic default paths")
    parser.add_argument("--validate", action="store_true", help="Validate systematic paths")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    
    if args.show_defaults:
        defaults = get_systematic_defaults()
        
        print(f"\n📋 Systematic Training Defaults")
        for key, value in defaults.items():
            status = "✅" if value else "❌"
            print(f"  {status} {key}: {value}")
    
    if args.validate:
        validation = validate_systematic_paths()
        
        print(f"\n🔍 Systematic Path Validation")
        for resource, available in validation.items():
            status = "✅" if available else "❌"
            print(f"  {status} {resource}")


if __name__ == "__main__":
    main()





