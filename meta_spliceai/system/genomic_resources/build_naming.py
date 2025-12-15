"""Standardized Build Naming Convention

This module provides utilities for standardizing genomic build names across
different annotation sources.

Convention
----------
Format: {build}_{source}

Examples:
    - GRCh37 (Ensembl is default/implicit for historical reasons)
    - GRCh38_MANE
    - GRCh38_Ensembl
    - GRCh38_GENCODE
    - GRCh37_RefSeq
    - T2T_CHM13 (Telomere-to-Telomere assembly)

Rationale
---------
1. **Build First**: The genomic build (GRCh37, GRCh38) is the primary identifier
2. **Source Suffix**: The annotation source (MANE, Ensembl, GENCODE) is a qualifier
3. **Backward Compatible**: GRCh37 (without suffix) maintains compatibility
4. **Extensible**: Easy to add new sources (RefSeq, GENCODE, etc.)
5. **Clear**: Unambiguous which build and source are being used

Usage
-----
>>> from meta_spliceai.system.genomic_resources import get_standardized_build_name
>>> 
>>> # Default case (Ensembl GRCh37)
>>> get_standardized_build_name('ensembl', 'GRCh37')
'GRCh37'
>>> 
>>> # MANE on GRCh38
>>> get_standardized_build_name('mane', 'GRCh38')
'GRCh38_MANE'
>>> 
>>> # Ensembl on GRCh38
>>> get_standardized_build_name('ensembl', 'GRCh38')
'GRCh38_Ensembl'
>>> 
>>> # GENCODE on GRCh38
>>> get_standardized_build_name('gencode', 'GRCh38')
'GRCh38_GENCODE'
"""

from typing import Dict, Optional


# Supported annotation sources
SUPPORTED_SOURCES = {
    'ensembl': 'Ensembl',
    'mane': 'MANE',
    'gencode': 'GENCODE',
    'refseq': 'RefSeq',
    'ucsc': 'UCSC',
}

# Supported genomic builds
SUPPORTED_BUILDS = {
    'GRCh37': 'Genome Reference Consortium Human Build 37',
    'GRCh38': 'Genome Reference Consortium Human Build 38',
    'hg19': 'UCSC Human Genome Build 19 (equivalent to GRCh37)',
    'hg38': 'UCSC Human Genome Build 38 (equivalent to GRCh38)',
    'T2T_CHM13': 'Telomere-to-Telomere CHM13 assembly',
}


def get_standardized_build_name(source: str, build: str) -> str:
    """
    Standardize build names across different annotation sources.
    
    Convention: {build}_{source}
    - If source is 'ensembl' and build is GRCh37: return 'GRCh37' (default)
    - Otherwise: return '{build}_{source.upper()}'
    
    Parameters
    ----------
    source : str
        Annotation source (e.g., 'ensembl', 'mane', 'gencode')
    build : str
        Genomic build (e.g., 'GRCh37', 'GRCh38')
    
    Returns
    -------
    str
        Standardized build name
    
    Examples
    --------
    >>> get_standardized_build_name('ensembl', 'GRCh37')
    'GRCh37'
    
    >>> get_standardized_build_name('mane', 'GRCh38')
    'GRCh38_MANE'
    
    >>> get_standardized_build_name('ensembl', 'GRCh38')
    'GRCh38_Ensembl'
    
    >>> get_standardized_build_name('gencode', 'GRCh38')
    'GRCh38_GENCODE'
    
    >>> get_standardized_build_name('refseq', 'GRCh37')
    'GRCh37_RefSeq'
    """
    source_lower = source.lower()
    
    # Validate source
    if source_lower not in SUPPORTED_SOURCES:
        # Allow unknown sources but warn
        import warnings
        warnings.warn(
            f"Unknown annotation source '{source}'. "
            f"Supported sources: {', '.join(SUPPORTED_SOURCES.keys())}"
        )
    
    # Default case: Ensembl GRCh37 (historical default for backward compatibility)
    if source_lower == 'ensembl' and build == 'GRCh37':
        return 'GRCh37'
    
    # Get standardized source name
    source_name = SUPPORTED_SOURCES.get(source_lower, source.upper())
    
    # All other cases: Build_Source
    return f"{build}_{source_name}"


def parse_build_name(build_name: str) -> Dict[str, str]:
    """
    Parse a standardized build name into build and source components.
    
    Parameters
    ----------
    build_name : str
        Standardized build name (e.g., 'GRCh38_MANE', 'GRCh37')
    
    Returns
    -------
    Dict[str, str]
        Dictionary with 'build' and 'source' keys
    
    Examples
    --------
    >>> parse_build_name('GRCh37')
    {'build': 'GRCh37', 'source': 'ensembl'}
    
    >>> parse_build_name('GRCh38_MANE')
    {'build': 'GRCh38', 'source': 'mane'}
    
    >>> parse_build_name('GRCh38_Ensembl')
    {'build': 'GRCh38', 'source': 'ensembl'}
    """
    if '_' not in build_name:
        # Default case: GRCh37 implies Ensembl
        if build_name == 'GRCh37':
            return {'build': 'GRCh37', 'source': 'ensembl'}
        else:
            # Unknown format, assume it's just a build
            return {'build': build_name, 'source': 'unknown'}
    
    # Split by underscore
    parts = build_name.split('_', 1)
    build = parts[0]
    source_upper = parts[1]
    
    # Find matching source (case-insensitive)
    source = None
    for src_key, src_name in SUPPORTED_SOURCES.items():
        if src_name.upper() == source_upper.upper():
            source = src_key
            break
    
    if source is None:
        source = source_upper.lower()
    
    return {'build': build, 'source': source}


def get_build_description(build_name: str) -> Optional[str]:
    """
    Get a human-readable description of a build.
    
    Parameters
    ----------
    build_name : str
        Standardized build name
    
    Returns
    -------
    Optional[str]
        Description of the build, or None if unknown
    
    Examples
    --------
    >>> get_build_description('GRCh37')
    'Genome Reference Consortium Human Build 37 (Ensembl)'
    
    >>> get_build_description('GRCh38_MANE')
    'Genome Reference Consortium Human Build 38 (MANE)'
    """
    parsed = parse_build_name(build_name)
    build = parsed['build']
    source = parsed['source']
    
    build_desc = SUPPORTED_BUILDS.get(build, build)
    source_name = SUPPORTED_SOURCES.get(source, source.upper())
    
    return f"{build_desc} ({source_name})"


def validate_build_name(build_name: str) -> bool:
    """
    Validate that a build name follows the standard convention.
    
    Parameters
    ----------
    build_name : str
        Build name to validate
    
    Returns
    -------
    bool
        True if valid, False otherwise
    
    Examples
    --------
    >>> validate_build_name('GRCh37')
    True
    
    >>> validate_build_name('GRCh38_MANE')
    True
    
    >>> validate_build_name('InvalidBuild')
    False
    """
    parsed = parse_build_name(build_name)
    build = parsed['build']
    source = parsed['source']
    
    # Check if build is supported
    if build not in SUPPORTED_BUILDS:
        return False
    
    # Check if source is supported (or unknown)
    if source not in SUPPORTED_SOURCES and source != 'unknown':
        return False
    
    return True


# Convenience function for common cases
def get_build_for_base_model(base_model: str) -> str:
    """
    Get the standard build name for a base model.
    
    Parameters
    ----------
    base_model : str
        Base model name (e.g., 'spliceai', 'openspliceai')
    
    Returns
    -------
    str
        Standardized build name
    
    Examples
    --------
    >>> get_build_for_base_model('spliceai')
    'GRCh37'
    
    >>> get_build_for_base_model('openspliceai')
    'GRCh38_MANE'
    """
    # Default mappings for known base models
    model_to_build = {
        'spliceai': 'GRCh37',  # Ensembl GRCh37
        'openspliceai': 'GRCh38_MANE',  # MANE GRCh38
        'spliceai_grch38': 'GRCh38_Ensembl',  # Hypothetical
    }
    
    return model_to_build.get(base_model.lower(), 'GRCh37')


__all__ = [
    'get_standardized_build_name',
    'parse_build_name',
    'get_build_description',
    'validate_build_name',
    'get_build_for_base_model',
    'SUPPORTED_SOURCES',
    'SUPPORTED_BUILDS',
]

