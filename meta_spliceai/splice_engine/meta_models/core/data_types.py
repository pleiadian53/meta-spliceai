"""
Core data types for meta models in the MetaSpliceAI.

This module defines the key data structures used throughout the meta model pipeline,
providing a clean interface for configuration and data handling.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union, Tuple, Set
from datetime import datetime
import pandas as pd
import polars as pl
import os
from meta_spliceai.splice_engine.meta_models.core.analyzer import Analyzer


@dataclass
class MetaModelConfig:
    """Configuration for meta model training data generation and evaluation."""
    
    # Base parameters
    gtf_file: Optional[str] = None
    pred_type: str = 'FP'  # Prediction type, one of: FP, FN
    
    # Feature generation
    kmer_sizes: List[int] = field(default_factory=lambda: [6])
    
    # Dataset subsetting
    subset_genes: bool = True
    subset_policy: str = 'hard'  # 'random', 'top', 'hard'
    n_genes: int = 1000  # relevant when subset_genes=True
    custom_genes: List[str] = field(default_factory=list)
    
    # Label configuration
    error_label: Optional[str] = None  # If None, set to pred_type
    correct_label: str = 'TP'  # TP or TN depending on error type
    
    # Data handling
    col_tid: str = 'transcript_id'
    overwrite: bool = True
    
    # Sampling parameters
    fn_sample_fraction: float = 1.0
    fn_max_sample_size: int = 50000
    tn_sample_fraction: float = 0.5
    tn_max_sample_size: int = 50000
    
    # Model parameters
    model_type: str = "random_forest"  # or "gradient_boosting", "neural_network"
    use_cross_validation: bool = True
    num_folds: int = 5
    test_size: float = 0.2
    random_state: int = 42
    
    # Hyperparameters
    hyperparameter_tuning: bool = False
    max_trials: int = 10
    
    def __post_init__(self):
        """Initialize derived values after initialization."""
        if self.error_label is None:
            self.error_label = self.pred_type
    
    @property
    def feature_columns(self) -> List[str]:
        """Get columns that are features for training models."""
        return []  # Placeholder for subclasses
    
    @property
    def target_columns(self) -> List[str]:
        """Get columns that are targets for training models."""
        return []  # Placeholder for subclasses
    
    @property
    def training_params(self) -> Dict[str, Any]:
        """Get training parameters."""
        return {
            'model_type': self.model_type,
            'use_cross_validation': self.use_cross_validation,
            'num_folds': self.num_folds,
            'test_size': self.test_size,
            'random_state': self.random_state,
            'hyperparameter_tuning': self.hyperparameter_tuning,
            'max_trials': self.max_trials
        }
    
    @property
    def metadata_columns(self) -> List[str]:
        """Get metadata columns that should be preserved but aren't features."""
        return ['gene_id', 'transcript_id', 'position', 'score', 
                'splice_type', 'chrom', 'strand', 'label']


@dataclass
class MetaModelDataset:
    """Container for meta model datasets at various stages of processing."""
    
    # Original dataset before feature extraction
    raw_data: Optional[Union[pd.DataFrame, pl.DataFrame]] = None
    
    # Dataset after k-mer and other feature extraction
    featurized_data: Optional[Union[pd.DataFrame, pl.DataFrame]] = None
    
    # Dataset after feature extraction and selection
    processed_data: Optional[Union[pd.DataFrame, pl.DataFrame]] = None
    
    # Split datasets for training
    train_data: Optional[Union[pd.DataFrame, pl.DataFrame]] = None
    val_data: Optional[Union[pd.DataFrame, pl.DataFrame]] = None
    test_data: Optional[Union[pd.DataFrame, pl.DataFrame]] = None
    
    # Features present in the dataset
    feature_columns: List[str] = field(default_factory=list)
    kmer_features: List[str] = field(default_factory=list)
    meta_features: List[str] = field(default_factory=list)
    
    # Information about feature types and importance
    feature_importance: Optional[Dict[str, float]] = None
    
    def get_features(self) -> List[str]:
        """Get all feature columns (excluding metadata columns)."""
        return self.kmer_features + self.meta_features
    
    @property
    def metadata_columns(self) -> List[str]:
        """Get metadata columns that should be preserved but aren't features."""
        return ['gene_id', 'transcript_id', 'position', 'score', 
                'splice_type', 'chrom', 'strand', 'label']


class DatasetConfig:
    """Configuration for dataset preparation."""
    
    # Dataset parameters
    feature_scaling: str = "standard"  # or "minmax", "robust", "none"
    fill_strategy: str = "median"  # or "mean", "zero", "none"
    
    # Feature selection parameters
    filter_low_variance: bool = True
    variance_threshold: float = 0.01
    
    # Base features to include
    include_sequence_features: bool = True
    include_genomic_features: bool = True
    include_conservation_scores: bool = False  # requires additional data
    
    def __init__(self, **kwargs):
        """Initialize with optional parameter overrides."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    @property
    def feature_params(self) -> Dict[str, Any]:
        """Get feature engineering parameters."""
        return {
            'feature_scaling': self.feature_scaling,
            'fill_strategy': self.fill_strategy,
            'filter_low_variance': self.filter_low_variance,
            'variance_threshold': self.variance_threshold
        }
    
    @property
    def inclusion_flags(self) -> Dict[str, bool]:
        """Get feature inclusion flags."""
        return {
            'include_sequence_features': self.include_sequence_features,
            'include_genomic_features': self.include_genomic_features,
            'include_conservation_scores': self.include_conservation_scores
        }


@dataclass
class SpliceAIConfig:
    """Configuration for enhanced SpliceAI prediction workflow."""
    
    # Get defaults from Analyzer
    _analyzer = Analyzer()
    
    # File paths
    gtf_file: Optional[str] = field(default_factory=lambda: Analyzer.gtf_file)
    genome_fasta: Optional[str] = field(default_factory=lambda: Analyzer.genome_fasta)
    eval_dir: str = field(default_factory=lambda: Analyzer.eval_dir)
    shared_dir: str = field(default_factory=lambda: Analyzer.shared_dir)
    
    # New parameter with default that derives from eval_dir
    local_dir: Optional[str] = None
    output_subdir: str = "meta_models"  # Subdirectory for meta model outputs
    
    # Artifact management (NEW)
    mode: str = "test"  # Execution mode: 'production' (immutable) or 'test' (overwritable)
    coverage: str = "gene_subset"  # Data coverage: 'full_genome', 'chromosome', 'gene_subset'
    test_name: Optional[str] = None  # Test identifier for test mode artifacts
    
    # Base model selection (NEW)
    base_model: str = "spliceai"  # Base model: 'spliceai' (GRCh37) or 'openspliceai' (GRCh38)
    
    # File format settings
    format: str = "tsv"
    seq_format: str = "parquet"
    seq_mode: str = "gene"
    seq_type: str = "full"
    separator: str = "\t"
    
    # Prediction parameters
    threshold: float = 0.5
    consensus_window: int = 2
    error_window: int = 500
    
    # Processing settings
    test_mode: bool = False
    chromosomes: Optional[List[str]] = None
    
    # Data preparation switches
    do_extract_annotations: bool = False
    do_extract_splice_sites: bool = False
    do_extract_sequences: bool = False
    do_find_overlaping_genes: bool = False
    use_precomputed_overlapping_genes: bool = False
    
    # New settings for three-probability-score workflow (from memory)
    use_threeprobabilities: bool = True
    use_auto_position_adjustments: bool = True
    save_example_sequences: bool = True
    
    # Output control - NEW (Nov 7, 2025)
    save_nucleotide_scores: bool = False  # Per-nucleotide scores (disabled by default for efficiency)
    # Note: Gene manifest is always saved (lightweight and useful for debugging)

    def __post_init__(self):
        """Initialize derived values after dataclass initialization.
        
        This method sets up base model-specific defaults for genomic resources
        if they weren't explicitly provided.
        """
        # =====================================================================
        # CRITICAL: Set base model-specific defaults using Registry
        # =====================================================================
        # ALWAYS use Registry to get correct paths based on base_model
        # This is critical because:
        # - SpliceAI requires GRCh37/Ensembl paths
        # - OpenSpliceAI requires GRCh38/MANE paths
        # - Analyzer defaults may be set to wrong build at import time
        base_model_lower = self.base_model.lower()
        
        # Check if we're using default Analyzer paths
        using_default_gtf = (self.gtf_file == Analyzer.gtf_file)
        using_default_fasta = (self.genome_fasta == Analyzer.genome_fasta)
        using_default_eval = (self.eval_dir == Analyzer.eval_dir)
        
        # ALWAYS override defaults with base model-specific paths
        # Each base model has a specific genome build it was trained on
        if using_default_gtf or using_default_fasta or using_default_eval:
            from meta_spliceai.system.genomic_resources import Registry
            from pathlib import Path
            
            # Determine correct registry based on base model
            if base_model_lower == 'spliceai':
                registry = Registry(build='GRCh37', release='87')
            elif base_model_lower == 'openspliceai':
                registry = Registry(build='GRCh38_MANE', release='1.3')
            else:
                # Future models can be added here
                # Default to GRCh37 for unknown models
                registry = Registry(build='GRCh37', release='87')
            
            # Override defaults with base model-specific paths
            if using_default_gtf:
                self.gtf_file = str(registry.get_gtf_path())
            if using_default_fasta:
                self.genome_fasta = str(registry.get_fasta_path())
            if using_default_eval:
                self.eval_dir = str(registry.data_dir / f'{base_model_lower}_eval')
            
            # CRITICAL: Also update shared_dir to be build-specific
            # This ensures derived datasets are loaded from correct build
            self.shared_dir = str(registry.stash)
        
        # =====================================================================
        # Derive local_dir from eval_dir if not provided
        # =====================================================================
        if self.local_dir is None:
            self.local_dir = os.path.dirname(self.eval_dir)
        
        # =====================================================================
        # Auto-detect mode from coverage
        # =====================================================================
        if self.coverage == "full_genome" and self.mode == "test":
            # Full genome coverage typically implies production mode
            self.mode = "production"
        
        # =====================================================================
        # Generate test_name if needed
        # =====================================================================
        if self.mode == "test" and self.test_name is None:
            from datetime import datetime
            self.test_name = f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    def get_full_eval_dir(self) -> str:
        """Get the full evaluation directory path including the output subdirectory."""
        return os.path.join(self.eval_dir, self.output_subdir)
    
    def get_analyzer(self) -> Analyzer:
        """Get an Analyzer instance with this configuration's paths."""
        return Analyzer(
            gtf_file=self.gtf_file,
            genome_fasta=self.genome_fasta,
            eval_dir=self.eval_dir
        )
    
    def get_artifact_manager(self):
        """Get an ArtifactManager for this configuration.
        
        Returns
        -------
        ArtifactManager
            Configured artifact manager based on workflow settings
        """
        from meta_spliceai.system.artifact_manager import create_artifact_manager_from_workflow_config
        
        # Use base_model parameter directly - this is the PRIMARY source of truth
        base_model = self.base_model.lower()
        
        # Infer source and build from base model (primary logic)
        # These defaults match the training data for each model
        if base_model == "openspliceai":
            # OpenSpliceAI uses GRCh38 and MANE annotations
            build = "GRCh38"
            source = "mane"
        else:
            # SpliceAI uses GRCh37 and Ensembl annotations (default)
            build = "GRCh37"
            source = "ensembl"
        
        # Note: We do NOT override build based on gtf_file path because:
        # 1. The base_model determines which genomic build to use
        # 2. The workflow should use the correct genomic resources for the model
        # 3. If user wants different resources, they should explicitly set them
        #
        # However, we DO check for source specification in the GTF path
        if self.gtf_file:
            if "mane" in self.gtf_file.lower():
                source = "mane"
            elif "gencode" in self.gtf_file.lower():
                source = "gencode"
            # Keep ensembl as default if no other source is specified
        
        # Get data_root - should be the project-level data directory
        from pathlib import Path
        from meta_spliceai.system.config import Config
        
        # Use project data directory, not local_dir (which is build-specific)
        try:
            data_root = Path(Config.PROJ_DIR) / "data"
        except (ImportError, AttributeError):
            data_root = Path("data")
        
        return create_artifact_manager_from_workflow_config(
            mode=self.mode,
            coverage=self.coverage,
            source=source,
            build=build,
            base_model=base_model,
            test_name=self.test_name,
            data_root=data_root
        )


@dataclass
class GeneManifestEntry:
    """
    Single entry in the gene manifest tracking gene processing status.
    
    Attributes
    ----------
    gene_id : str
        Gene identifier (e.g., ENSG00000012048)
    gene_name : str
        Gene symbol (e.g., BRCA1)
    requested : bool
        Whether this gene was explicitly requested
    status : str
        Processing status: 'processed', 'not_in_annotation', 'no_sequence', 
        'sequence_too_short', 'prediction_failed'
    reason : Optional[str]
        Explanation if status != 'processed'
    num_positions : int
        Number of positions analyzed (0 if failed)
    num_nucleotides : int
        Total sequence length (0 if failed)
    num_splice_sites : int
        Number of annotated splice sites (0 if failed)
    processing_time_sec : float
        Time taken to process this gene
    base_model : str
        Base model used ('spliceai', 'openspliceai', etc.)
    genomic_build : str
        Genomic build ('GRCh37', 'GRCh38', etc.)
    timestamp : str
        ISO format timestamp of processing
    """
    gene_id: str
    gene_name: str
    requested: bool
    status: str  # 'processed', 'not_in_annotation', 'no_sequence', 'sequence_too_short', 'prediction_failed'
    reason: Optional[str] = None
    num_positions: int = 0
    num_nucleotides: int = 0
    num_splice_sites: int = 0
    processing_time_sec: float = 0.0
    base_model: str = "spliceai"
    genomic_build: str = "GRCh37"
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for DataFrame creation."""
        return {
            'gene_id': self.gene_id,
            'gene_name': self.gene_name,
            'requested': self.requested,
            'status': self.status,
            'reason': self.reason,
            'num_positions': self.num_positions,
            'num_nucleotides': self.num_nucleotides,
            'num_splice_sites': self.num_splice_sites,
            'processing_time_sec': self.processing_time_sec,
            'base_model': self.base_model,
            'genomic_build': self.genomic_build,
            'timestamp': self.timestamp
        }


class GeneManifest:
    """
    Tracks gene processing status across the workflow.
    
    This class maintains a record of all genes that were requested for processing,
    their processing status, and diagnostic information for failed genes.
    
    Examples
    --------
    >>> manifest = GeneManifest(base_model='spliceai', genomic_build='GRCh37')
    >>> manifest.add_requested_genes(['BRCA1', 'TP53', 'UNKNOWN_GENE'])
    >>> manifest.mark_processed('BRCA1', num_positions=1234, num_nucleotides=5592, processing_time=12.3)
    >>> manifest.mark_failed('UNKNOWN_GENE', status='not_in_annotation', reason='Gene not found in GTF')
    >>> df = manifest.to_dataframe()
    >>> manifest.save('/path/to/gene_manifest.tsv')
    """
    
    def __init__(self, base_model: str = 'spliceai', genomic_build: str = 'GRCh37'):
        """
        Initialize gene manifest.
        
        Parameters
        ----------
        base_model : str
            Base model being used
        genomic_build : str
            Genomic build being used
        """
        self.base_model = base_model
        self.genomic_build = genomic_build
        self.entries: Dict[str, GeneManifestEntry] = {}
        self._requested_genes: Set[str] = set()
    
    def add_requested_genes(self, gene_ids: Union[List[str], Set[str]]):
        """
        Mark genes as requested for processing.
        
        Parameters
        ----------
        gene_ids : List[str] or Set[str]
            Gene IDs that were requested
        """
        self._requested_genes.update(gene_ids)
        
        # Create placeholder entries for requested genes
        for gene_id in gene_ids:
            if gene_id not in self.entries:
                self.entries[gene_id] = GeneManifestEntry(
                    gene_id=gene_id,
                    gene_name=gene_id,  # Will be updated when processed
                    requested=True,
                    status='pending',
                    base_model=self.base_model,
                    genomic_build=self.genomic_build
                )
    
    def mark_processed(
        self,
        gene_id: str,
        gene_name: Optional[str] = None,
        num_positions: int = 0,
        num_nucleotides: int = 0,
        num_splice_sites: int = 0,
        processing_time: float = 0.0
    ):
        """
        Mark a gene as successfully processed.
        
        Parameters
        ----------
        gene_id : str
            Gene identifier
        gene_name : str, optional
            Gene symbol
        num_positions : int
            Number of positions analyzed
        num_nucleotides : int
            Total sequence length
        num_splice_sites : int
            Number of annotated splice sites
        processing_time : float
            Processing time in seconds
        """
        self.entries[gene_id] = GeneManifestEntry(
            gene_id=gene_id,
            gene_name=gene_name or gene_id,
            requested=gene_id in self._requested_genes,
            status='processed',
            reason=None,
            num_positions=num_positions,
            num_nucleotides=num_nucleotides,
            num_splice_sites=num_splice_sites,
            processing_time_sec=processing_time,
            base_model=self.base_model,
            genomic_build=self.genomic_build
        )
    
    def mark_failed(
        self,
        gene_id: str,
        gene_name: Optional[str] = None,
        status: str = 'prediction_failed',
        reason: Optional[str] = None
    ):
        """
        Mark a gene as failed to process.
        
        Parameters
        ----------
        gene_id : str
            Gene identifier
        gene_name : str, optional
            Gene symbol
        status : str
            Failure status ('not_in_annotation', 'no_sequence', 'sequence_too_short', 'prediction_failed')
        reason : str, optional
            Detailed reason for failure
        """
        self.entries[gene_id] = GeneManifestEntry(
            gene_id=gene_id,
            gene_name=gene_name or gene_id,
            requested=gene_id in self._requested_genes,
            status=status,
            reason=reason,
            num_positions=0,
            num_nucleotides=0,
            num_splice_sites=0,
            processing_time_sec=0.0,
            base_model=self.base_model,
            genomic_build=self.genomic_build
        )
    
    def to_dataframe(self, use_polars: bool = True) -> Union[pl.DataFrame, pd.DataFrame]:
        """
        Convert manifest to DataFrame.
        
        Parameters
        ----------
        use_polars : bool
            If True, return Polars DataFrame; otherwise Pandas
            
        Returns
        -------
        pl.DataFrame or pd.DataFrame
            Manifest as DataFrame
        """
        data = [entry.to_dict() for entry in self.entries.values()]
        
        if use_polars:
            return pl.DataFrame(data)
        else:
            return pd.DataFrame(data)
    
    def save(self, path: str, use_polars: bool = True):
        """
        Save manifest to TSV file.
        
        Parameters
        ----------
        path : str
            Output file path
        use_polars : bool
            If True, use Polars for saving; otherwise Pandas
        """
        df = self.to_dataframe(use_polars=use_polars)
        
        if use_polars:
            df.write_csv(path, separator='\t')
        else:
            df.to_csv(path, sep='\t', index=False)
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics for the manifest.
        
        Returns
        -------
        dict
            Summary statistics including counts by status
        """
        df = self.to_dataframe(use_polars=True)
        
        if df.height == 0:
            return {
                'total_genes': 0,
                'requested_genes': 0,
                'processed_genes': 0,
                'failed_genes': 0,
                'status_counts': {}
            }
        
        status_counts = df.group_by('status').agg(pl.count()).to_dict(as_series=False)
        
        return {
            'total_genes': df.height,
            'requested_genes': df.filter(pl.col('requested')).height,
            'processed_genes': df.filter(pl.col('status') == 'processed').height,
            'failed_genes': df.filter(pl.col('status') != 'processed').height,
            'status_counts': dict(zip(status_counts['status'], status_counts['count'])),
            'total_processing_time_sec': df['processing_time_sec'].sum(),
            'base_model': self.base_model,
            'genomic_build': self.genomic_build
        }
