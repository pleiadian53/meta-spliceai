import os


class Analyzer(object): 
    """Base analyzer class providing standardized paths and configuration.
    
    CRITICAL: As of 2025-12, the Analyzer uses BUILD-SPECIFIC paths.
    
    The directory structure is:
        data/<source>/<build>/
    
    For example:
        data/ensembl/GRCh37/  (for SpliceAI)
        data/mane/GRCh38/     (for OpenSpliceAI)
    
    The Registry's stash (build-specific directory) is used for:
        - data_dir: Build-specific data directory
        - shared_dir: Same as data_dir (for derived datasets)
        - eval_dir: data_dir/spliceai_eval or openspliceai_eval
        - analysis_dir: data_dir/spliceai_analysis
    
    This ensures that derived datasets (gene_features.tsv, splice_sites.tsv, etc.)
    are always loaded from the correct build-specific location, preventing
    cross-build contamination.
    """
    source = 'ensembl'
    version = ''
    
    # Use genomic resources manager for systematic path resolution
    try:
        from meta_spliceai.system.genomic_resources import Registry
        from meta_spliceai.system.config import Config as SystemConfig
        
        prefix = SystemConfig.PROJ_DIR
        _registry = Registry()
        
        # CRITICAL: Use Registry's stash (build-specific directory), NOT data_root
        # data_root = data/, stash = data/ensembl/GRCh37/ (for GRCh37 build)
        # This ensures all derived datasets are loaded from build-specific paths
        data_dir = str(_registry.stash)
        
        # shared_dir should also be build-specific to avoid cross-contamination
        shared_dir = str(_registry.stash)
        
        gtf_file = _registry.resolve("gtf")
        genome_fasta = _registry.resolve("fasta")
        
    except (ImportError, AttributeError):
        # Fallback for environments without full setup
        try:
            from meta_spliceai.system.config import Config as SystemConfig, find_project_root
            prefix = SystemConfig.PROJ_DIR
        except (ImportError, AttributeError):
            from pathlib import Path
            prefix = os.getenv("META_SPLICEAI_ROOT", str(Path(__file__).parent.parent.parent))
        
        # Fallback: Use default GRCh38 path
        data_dir = os.path.join(prefix, "data", "ensembl", "GRCh38")
        shared_dir = data_dir
        gtf_file = os.path.join(data_dir, "Homo_sapiens.GRCh38.112.gtf")
        genome_fasta = os.path.join(data_dir, "Homo_sapiens.GRCh38.dna.primary_assembly.fa")
    
    # Blob storage prefix (configurable via environment variable)
    blob_prefix = os.getenv("META_SPLICEAI_BLOB", "/mnt/nfs1/meta-spliceai")
    
    # Analysis directories (build-specific)
    eval_dir = os.path.join(data_dir, "spliceai_eval")
    analysis_dir = os.path.join(data_dir, "spliceai_analysis")

    def __init__(self): 
        pass