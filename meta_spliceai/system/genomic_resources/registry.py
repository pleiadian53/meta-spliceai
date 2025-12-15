"""Registry for resolving genomic resource paths.

Provides a unified interface for locating GTF, FASTA, and derived TSV files
across multiple possible locations (data/ensembl/, data/ensembl/<BUILD>/, 
data/ensembl/spliceai_analysis/).
"""

import os
from pathlib import Path
from typing import Optional
from .config import load_config, filename


class Registry:
    """Registry for resolving genomic resource paths.
    
    NEW STRUCTURE (as of 2025-11-01):
    Directory structure: data / annotation_source / build
    
    Examples:
    - data/ensembl/GRCh37/
    - data/ensembl/GRCh38/
    - data/mane/GRCh38/
    
    The annotation source name (ensembl, mane, gencode) already tells us the data origin,
    so no additional "genomic_resources" level is needed.
    
    Search order for files:
    1. Explicit path from environment variable (SS_GTF_PATH, SS_FASTA_PATH, etc.)
    2. data/<annotation_source>/<BUILD>/ (build-specific directory)
    3. data/<annotation_source>/<BUILD>/spliceai_analysis/ (derived datasets)
    
    Parameters
    ----------
    build : str, optional
        Override build from config (e.g., 'GRCh38', 'GRCh37', 'GRCh38_MANE')
    release : str, optional
        Override release from config (e.g., '112', '106', '1.3')
    """
    
    def __init__(self, build: Optional[str] = None, release: Optional[str] = None):
        self.cfg = load_config()
        if build:
            self.cfg.build = build
        if release:
            self.cfg.release = release
        
        # Get annotation source for this build
        self.annotation_source = self.cfg.get_annotation_source(self.cfg.build)
        
        # NEW: Build directory structure using annotation source
        # Structure: data_root / annotation_source / build
        self.top = self.cfg.data_root / self.annotation_source
        
        # For MANE builds, strip suffix from directory name
        build_dir = self.cfg.build.replace("_MANE", "").replace("_GENCODE", "")
        self.stash = self.top / build_dir
        self.legacy = self.top / "spliceai_analysis"  # Legacy location (rarely used)
        
        # Build-specific directories
        self.data_dir = self.stash  # Alias for build-specific directory
        self.eval_dir = self.stash / "spliceai_eval"
        self.analysis_dir = self.stash / "spliceai_analysis"
    
    def resolve(self, kind: str) -> Optional[str]:
        """Resolve path for a given resource kind.
        
        Parameters
        ----------
        kind : str
            Resource kind: 'gtf', 'fasta', 'fasta_index', 'splice_sites',
            'gene_features', 'transcript_features', 'exon_features', 'junctions'
            
        Returns
        -------
        str or None
            Absolute path to the resource if found, None otherwise
            
        Search Order
        ------------
        For GTF/FASTA (primary genomic files):
            1. Environment variable override
            2. Build-specific directory (data/<source>/<build>/)
            3. Root directory (data/<source>/) - for backwards compatibility
            
        For derived datasets (gene_features, splice_sites, etc.):
            1. Environment variable override
            2. Build-specific directory ONLY (data/<source>/<build>/)
            
            NOTE: Root directory is EXCLUDED for derived datasets to prevent
            cross-build contamination. Each base model requires its own
            build-specific derived datasets.
            
        Examples
        --------
        >>> r = Registry(build='GRCh37')
        >>> r.resolve("gtf")
        '/path/to/data/ensembl/GRCh37/Homo_sapiens.GRCh37.87.gtf'
        >>> r.resolve("gene_features")
        '/path/to/data/ensembl/GRCh37/gene_features.tsv'
        """
        # Check for explicit environment variable override
        env_var = f"SS_{kind.upper()}_PATH"
        if env_var in os.environ:
            path = Path(os.environ[env_var])
            if path.exists():
                return str(path.resolve())
        
        # Determine if this is a primary genomic file or derived dataset
        is_primary_genomic = kind in ("gtf", "fasta", "fasta_index")
        
        # Determine filename
        if kind == "gtf":
            name = filename("gtf", self.cfg)
        elif kind == "fasta":
            name = filename("fasta", self.cfg)
        elif kind == "fasta_index":
            name = filename("fasta", self.cfg) + ".fai"
        else:
            # Derived datasets - use config if available, otherwise use defaults
            if self.cfg.derived_datasets and kind in self.cfg.derived_datasets:
                name = self.cfg.derived_datasets[kind]
            else:
                # Fallback to defaults if not in config
                mapping = {
                    "splice_sites": "splice_sites.tsv",
                    "gene_features": "gene_features.tsv",
                    "transcript_features": "transcript_features.tsv",
                    "exon_features": "exon_features.tsv",
                    "junctions": "junctions.tsv"
                }
                if kind not in mapping:
                    raise ValueError(f"Unknown resource kind: {kind}")
                name = mapping[kind]
        
        # Note: Enhanced versions now configured via genomic_resources.yaml
        # The config default is splice_sites_enhanced.tsv
        # Legacy behavior: still search for enhanced version if not found
        if kind == "splice_sites" and "enhanced" not in name:
            # Only if config specifies regular splice_sites.tsv, try enhanced as fallback
            enhanced_name = "splice_sites_enhanced.tsv"
            # For derived datasets, ONLY search in build-specific directory
            search_order = [self.stash]
            for root in search_order:
                enhanced_path = Path(root) / enhanced_name
                if enhanced_path.exists():
                    return str(enhanced_path.resolve())
            # Fallback to regular splice_sites.tsv if enhanced doesn't exist
        
        # CRITICAL: Search order depends on file type
        # For primary genomic files (GTF/FASTA): include root directory as fallback
        # For derived datasets: ONLY search in build-specific directory to prevent
        # cross-build contamination (e.g., GRCh38 gene_features being used for GRCh37)
        if is_primary_genomic:
            # GTF/FASTA can fall back to root directory for backwards compatibility
            search_order = [self.stash, self.top]
        else:
            # Derived datasets: STRICT build-specific search only
            # This prevents loading wrong build's gene_features.tsv, splice_sites.tsv, etc.
            search_order = [self.stash]
        
        for root in search_order:
            p = Path(root) / name
            if p.exists():
                return str(p.resolve())
        
        return None
    
    def get_gtf_path(self, validate=True) -> Optional[Path]:
        """Get the path to the GTF file.
        
        Parameters
        ----------
        validate : bool, default=True
            If True, raises FileNotFoundError if GTF file doesn't exist
            
        Returns
        -------
        Path or None
            Path to GTF file, or None if not found and validate=False
            
        Raises
        ------
        FileNotFoundError
            If validate=True and GTF file doesn't exist
        """
        gtf_path = self.resolve("gtf")
        if gtf_path is None:
            if validate:
                raise FileNotFoundError(
                    f"GTF file not found for build {self.cfg.build}, release {self.cfg.release}"
                )
            return None
        return Path(gtf_path)
    
    def get_fasta_path(self, validate=True) -> Optional[Path]:
        """Get the path to the FASTA file.
        
        Parameters
        ----------
        validate : bool, default=True
            If True, raises FileNotFoundError if FASTA file doesn't exist
            
        Returns
        -------
        Path or None
            Path to FASTA file, or None if not found and validate=False
            
        Raises
        ------
        FileNotFoundError
            If validate=True and FASTA file doesn't exist
        """
        fasta_path = self.resolve("fasta")
        if fasta_path is None:
            if validate:
                raise FileNotFoundError(
                    f"FASTA file not found for build {self.cfg.build}, release {self.cfg.release}"
                )
            return None
        return Path(fasta_path)
    
    def list_all(self) -> dict:
        """List all known resource kinds and their resolved paths.
        
        Returns
        -------
        dict
            Mapping from resource kind to resolved path (or None if not found)
        """
        kinds = [
            "gtf",
            "fasta",
            "fasta_index",
            "splice_sites",
            "gene_features",
            "transcript_features",
            "exon_features",
            "junctions",
        ]
        return {kind: self.resolve(kind) for kind in kinds}
    
    # ──────────────────────────────────────────────────────────────────────────
    # Additional helper methods for common paths
    # ──────────────────────────────────────────────────────────────────────────
    
    def get_annotations_db_path(self, validate: bool = False) -> Optional[Path]:
        """Get path to annotations.db file.
        
        Parameters
        ----------
        validate : bool, default=False
            If True, raises FileNotFoundError if file doesn't exist
            
        Returns
        -------
        Path or None
            Path to annotations.db, or None if not found and validate=False
        """
        # Search in build-specific directory ONLY (derived dataset)
        path = Path(self.stash) / "annotations.db"
        if path.exists():
            return path
        
        if validate:
            raise FileNotFoundError(
                f"annotations.db not found for build {self.cfg.build}. "
                f"Expected at: {path}"
            )
        return None
    
    def get_overlapping_genes_path(self, validate: bool = False) -> Optional[Path]:
        """Get path to overlapping_genes.tsv or overlapping_gene_counts.tsv file.
        
        Parameters
        ----------
        validate : bool, default=False
            If True, raises FileNotFoundError if file doesn't exist
            
        Returns
        -------
        Path or None
            Path to overlapping genes file, or None if not found and validate=False
        """
        # Search in build-specific directory ONLY (derived dataset)
        for filename in ["overlapping_genes.tsv", "overlapping_gene_counts.tsv"]:
            path = Path(self.stash) / filename
            if path.exists():
                return path
        
        if validate:
            raise FileNotFoundError(
                f"overlapping genes file not found for build {self.cfg.build}. "
                f"Expected at: {self.stash}"
            )
        return None
    
    def get_chromosome_sequence_path(
        self, 
        chromosome: str, 
        format: str = "parquet",
        validate: bool = False
    ) -> Optional[Path]:
        """Get path to chromosome-specific sequence file.
        
        Parameters
        ----------
        chromosome : str
            Chromosome name (e.g., '1', '21', 'X')
        format : str, default='parquet'
            File format ('parquet' or 'tsv')
        validate : bool, default=False
            If True, raises FileNotFoundError if file doesn't exist
            
        Returns
        -------
        Path or None
            Path to chromosome sequence file, or None if not found and validate=False
        """
        filename = f"gene_sequence_{chromosome}.{format}"
        
        # Search in build-specific directory ONLY (derived dataset)
        path = Path(self.stash) / filename
        if path.exists():
            return path
        
        if validate:
            raise FileNotFoundError(
                f"Chromosome {chromosome} sequence file not found for build {self.cfg.build}. "
                f"Expected at: {path}"
            )
        return None
    
    def get_local_dir(self) -> Path:
        """Get the local directory for intermediate files.
        
        This is the build-specific directory where derived datasets are stored.
        
        Returns
        -------
        Path
            Path to local directory (same as data_dir/stash)
        """
        return self.stash
    
    def get_eval_dir(self, create: bool = False) -> Path:
        """Get the evaluation directory for SpliceAI outputs.
        
        Parameters
        ----------
        create : bool, default=False
            If True, create the directory if it doesn't exist
            
        Returns
        -------
        Path
            Path to evaluation directory
        """
        if create:
            self.eval_dir.mkdir(parents=True, exist_ok=True)
        return self.eval_dir
    
    def get_analysis_dir(self, create: bool = False) -> Path:
        """Get the analysis directory for derived datasets.
        
        Parameters
        ----------
        create : bool, default=False
            If True, create the directory if it doesn't exist
            
        Returns
        -------
        Path
            Path to analysis directory
        """
        if create:
            self.analysis_dir.mkdir(parents=True, exist_ok=True)
        return self.analysis_dir
    
    # ──────────────────────────────────────────────────────────────────────────
    # Base model artifact paths (NEW - supports multiple base models)
    # ──────────────────────────────────────────────────────────────────────────
    
    def get_base_model_eval_dir(
        self, 
        base_model: str, 
        create: bool = False
    ) -> Path:
        """Get evaluation directory for a specific base model.
        
        This supports the multi-base-model architecture where each base model
        has its own evaluation directory:
        - data/ensembl/GRCh37/spliceai_eval/
        - data/mane/GRCh38/openspliceai_eval/
        
        Parameters
        ----------
        base_model : str
            Base model name (e.g., 'spliceai', 'openspliceai')
        create : bool, default=False
            If True, create the directory if it doesn't exist
            
        Returns
        -------
        Path
            Path to base model evaluation directory
            
        Examples
        --------
        >>> registry = Registry(build='GRCh38_MANE')
        >>> registry.get_base_model_eval_dir('openspliceai')
        Path('data/mane/GRCh38/openspliceai_eval')
        """
        eval_dir = self.stash / f'{base_model.lower()}_eval'
        if create:
            eval_dir.mkdir(parents=True, exist_ok=True)
        return eval_dir
    
    def get_meta_models_artifact_dir(
        self, 
        base_model: str, 
        create: bool = False
    ) -> Path:
        """Get meta_models artifact directory for a specific base model.
        
        This is the standard location for meta-layer training artifacts:
        - data/ensembl/GRCh37/spliceai_eval/meta_models/
        - data/mane/GRCh38/openspliceai_eval/meta_models/
        
        Parameters
        ----------
        base_model : str
            Base model name (e.g., 'spliceai', 'openspliceai')
        create : bool, default=False
            If True, create the directory if it doesn't exist
            
        Returns
        -------
        Path
            Path to meta_models artifact directory
            
        Examples
        --------
        >>> registry = Registry(build='GRCh38_MANE')
        >>> registry.get_meta_models_artifact_dir('openspliceai')
        Path('data/mane/GRCh38/openspliceai_eval/meta_models')
        """
        artifact_dir = self.get_base_model_eval_dir(base_model) / 'meta_models'
        if create:
            artifact_dir.mkdir(parents=True, exist_ok=True)
        return artifact_dir


# Global registry cache
_registry_cache = {}


def get_genomic_registry(source: str, build: str) -> Registry:
    """Get or create a genomic registry for a specific source and build.
    
    This function caches registry instances to avoid recreating them.
    
    Parameters
    ----------
    source : str
        Annotation source (e.g., 'ensembl', 'mane', 'gencode')
    build : str
        Genomic build (e.g., 'GRCh37', 'GRCh38')
    
    Returns
    -------
    Registry
        Registry instance for the specified source and build
    
    Examples
    --------
    >>> registry = get_genomic_registry('ensembl', 'GRCh37')
    >>> gtf_path = registry.gtf_file
    """
    key = f"{source}/{build}"
    
    if key not in _registry_cache:
        _registry_cache[key] = Registry(source=source, build=build)
    
    return _registry_cache[key]
