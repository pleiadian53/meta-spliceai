"""Genomic Resources Manager

This package provides utilities for managing genomic reference data (GTF, FASTA)
and derived datasets (splice sites, gene/transcript/exon features).

Key components:
- config.py: Configuration management with YAML and environment variable support
- registry.py: Path resolution for genomic resources
- download.py: Download and index Ensembl GTF/FASTA files
- derive.py: Generate derived TSV datasets from GTF
- cli.py: Command-line interface for all operations
"""

__version__ = "0.1.0"

from .config import Config, load_config, filename
from .registry import Registry, get_genomic_registry
from .derive import (
    GenomicDataDeriver,
    derive_gene_annotations,
    derive_splice_sites,
    derive_genomic_sequences,
    derive_overlapping_genes,
    derive_gene_features,
    derive_transcript_features,
    derive_exon_features,
    derive_junctions
)
from .splice_sites import (
    extract_splice_sites_from_gtf,
    extract_splice_sites_workflow
)
from .validators import (
    validate_genes_have_splice_sites,
    validate_genes_in_gtf,
    validate_gene_selection,
    assert_coordinate_policy,
    verify_gtf_coordinate_system,
    assert_splice_motif_policy,
    assert_build_alignment,
    ValidationError
)
from .schema import (
    standardize_splice_sites_schema,
    standardize_gene_features_schema,
    standardize_transcript_features_schema,
    standardize_exon_features_schema,
    standardize_all_schemas,
    get_standard_column_mapping,
    print_standard_schemas
)
from .gene_mapper import (
    GeneMapper,
    GeneInfo,
    GeneMappingResult,
    get_gene_mapper,
    reset_gene_mapper
)
from .gene_mapper_enhanced import (
    EnhancedGeneMapper,
    GeneMapping,
    MappingStrategy
)
from .external_id_mapper import (
    get_or_create_mane_ensembl_mapping,
    create_mane_to_ensembl_mapping,
    load_mane_to_ensembl_mapping
)
from .gene_selection import (
    GeneSelector,
    GeneSamplingConfig,
    GeneSamplingResult
)
from .build_naming import (
    get_standardized_build_name,
    parse_build_name,
    get_build_description,
    validate_build_name,
    get_build_for_base_model,
    SUPPORTED_SOURCES,
    SUPPORTED_BUILDS
)


def create_systematic_manager(build=None, release=None):
    """
    Create a systematic genomic resource manager.
    
    This is a convenience function that creates a Registry instance
    for managing genomic resources.
    
    Parameters
    ----------
    build : str, optional
        Genome build (e.g., 'GRCh38', 'GRCh37')
    release : str, optional
        Ensembl release number
        
    Returns
    -------
    Registry
        A Registry instance for resolving genomic resource paths
    """
    return Registry(build=build, release=release)


__all__ = [
    "Config",
    "load_config",
    "filename",
    "Registry",
    "get_genomic_registry",
    "create_systematic_manager",
    "GenomicDataDeriver",
    "derive_gene_annotations",
    "derive_splice_sites",
    "derive_genomic_sequences",
    "derive_overlapping_genes",
    "derive_gene_features",
    "derive_transcript_features",
    "derive_exon_features",
    "derive_junctions",
    "validate_genes_have_splice_sites",
    "validate_genes_in_gtf",
    "validate_gene_selection",
    "assert_coordinate_policy",
    "verify_gtf_coordinate_system",
    "assert_splice_motif_policy",
    "assert_build_alignment",
    "ValidationError",
    "standardize_splice_sites_schema",
    "standardize_gene_features_schema",
    "standardize_transcript_features_schema",
    "standardize_exon_features_schema",
    "standardize_all_schemas",
    "get_standard_column_mapping",
    "print_standard_schemas",
    "GeneMapper",
    "GeneInfo",
    "GeneMappingResult",
    "get_gene_mapper",
    "reset_gene_mapper",
    "EnhancedGeneMapper",
    "GeneMapping",
    "MappingStrategy",
    "get_or_create_mane_ensembl_mapping",
    "create_mane_to_ensembl_mapping",
    "load_mane_to_ensembl_mapping",
    "GeneSelector",
    "GeneSamplingConfig",
    "GeneSamplingResult",
    "get_standardized_build_name",
    "parse_build_name",
    "get_build_description",
    "validate_build_name",
    "get_build_for_base_model",
    "SUPPORTED_SOURCES",
    "SUPPORTED_BUILDS",
]
