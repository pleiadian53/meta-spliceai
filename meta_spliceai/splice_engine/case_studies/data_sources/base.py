"""
Base classes for database ingestion.

Provides common interfaces and data structures for ingesting splice mutation
data from various databases and converting to MetaSpliceAI format.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
import pandas as pd
import polars as pl
from enum import Enum


class SpliceEventType(Enum):
    """Types of splice events that can be detected."""
    CRYPTIC_DONOR = "cryptic_donor"
    CRYPTIC_ACCEPTOR = "cryptic_acceptor" 
    EXON_SKIPPING = "exon_skipping"
    INTRON_RETENTION = "intron_retention"
    PARTIAL_EXON_DELETION = "partial_exon_deletion"
    PARTIAL_EXON_INSERTION = "partial_exon_insertion"
    PSEUDOEXON_ACTIVATION = "pseudoexon_activation"
    CANONICAL_SITE_LOSS = "canonical_site_loss"


class ClinicalSignificance(Enum):
    """Clinical significance categories."""
    PATHOGENIC = "pathogenic"
    LIKELY_PATHOGENIC = "likely_pathogenic"
    BENIGN = "benign"
    LIKELY_BENIGN = "likely_benign"
    UNCERTAIN = "uncertain"
    CONFLICTING = "conflicting"


@dataclass
class SpliceMutation:
    """Standardized representation of a splice-affecting mutation."""
    
    # Genomic coordinates (1-based)
    chrom: str
    position: int  # 1-based position
    ref_allele: str
    alt_allele: str
    
    # Gene/transcript information
    gene_id: str
    gene_symbol: str
    splice_event_type: SpliceEventType
    affected_site_type: str  # "donor", "acceptor", "cryptic", etc.
    source_database: str
    source_id: str
    
    # Optional fields
    transcript_id: Optional[str] = None
    splice_site_position: Optional[int] = None  # Position of affected splice site
    clinical_significance: Optional[ClinicalSignificance] = None
    disease_context: Optional[str] = None
    experimentally_validated: bool = False
    validation_method: Optional[str] = None
    hgvs_notation: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass 
class IngestionResult:
    """Result of database ingestion process."""
    
    mutations: List[SpliceMutation]
    splice_sites_df: pd.DataFrame  # Compatible with existing annotation format
    gene_features_df: pd.DataFrame  # Compatible with existing gene features format
    transcript_features_df: Optional[pd.DataFrame] = None
    
    # Statistics
    total_records: int = 0
    successfully_parsed: int = 0
    parsing_errors: int = 0
    
    # Error tracking
    error_log: List[Dict[str, str]] = None
    
    def __post_init__(self):
        if self.error_log is None:
            self.error_log = []
        if self.total_records == 0:
            self.total_records = len(self.mutations)
        if self.successfully_parsed == 0:
            self.successfully_parsed = len(self.mutations)


class BaseIngester(ABC):
    """Base class for database ingesters."""
    
    def __init__(self, output_dir: Path, cache_dir: Optional[Path] = None):
        """
        Initialize ingester.
        
        Parameters
        ----------
        output_dir : Path
            Directory to save processed data
        cache_dir : Path, optional
            Directory for caching downloaded data
        """
        self.output_dir = Path(output_dir)
        self.cache_dir = Path(cache_dir) if cache_dir else self.output_dir / "cache"
        
        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Track processing statistics
        self.processing_stats = {
            "total_downloaded": 0,
            "total_processed": 0,
            "parsing_errors": 0,
            "validation_errors": 0
        }
    
    @abstractmethod
    def download_data(self, force_refresh: bool = False) -> Path:
        """Download data from source database."""
        pass
    
    @abstractmethod
    def parse_raw_data(self, data_path: Path) -> List[SpliceMutation]:
        """Parse raw database format to SpliceMutation objects."""
        pass
    
    @abstractmethod
    def validate_mutations(self, mutations: List[SpliceMutation]) -> List[SpliceMutation]:
        """Validate and filter parsed mutations."""
        pass
    
    def create_splice_sites_annotation(self, mutations: List[SpliceMutation]) -> pd.DataFrame:
        """
        Create splice sites annotation compatible with existing format.
        
        Expected format:
        chrom, start, end, position, strand, site_type, gene_id, transcript_id
        """
        records = []
        
        for mutation in mutations:
            # Create entries for affected splice sites
            if mutation.splice_site_position:
                # Determine site type and coordinates
                if mutation.affected_site_type in ["donor", "cryptic_donor"]:
                    site_type = "donor"
                    # Donor sites are typically GT dinucleotides
                    start = mutation.splice_site_position - 1  # 0-based start
                    end = mutation.splice_site_position + 1    # 1-based end
                elif mutation.affected_site_type in ["acceptor", "cryptic_acceptor"]:
                    site_type = "acceptor" 
                    # Acceptor sites are typically AG dinucleotides
                    start = mutation.splice_site_position - 2  # 0-based start  
                    end = mutation.splice_site_position        # 1-based end
                else:
                    continue  # Skip unknown site types
                
                record = {
                    "chrom": mutation.chrom,
                    "start": start,
                    "end": end, 
                    "position": mutation.splice_site_position,
                    "strand": "+",  # Default; should be inferred from gene data
                    "site_type": site_type,
                    "gene_id": mutation.gene_id,
                    "transcript_id": mutation.transcript_id or ""
                }
                records.append(record)
        
        return pd.DataFrame(records)
    
    def create_gene_features(self, mutations: List[SpliceMutation]) -> pd.DataFrame:
        """
        Create gene features compatible with existing format.
        
        Expected format:
        start, end, score, strand, gene_id, gene_name, gene_type, gene_length, chrom
        """
        # Group mutations by gene
        gene_groups = {}
        for mutation in mutations:
            if mutation.gene_id not in gene_groups:
                gene_groups[mutation.gene_id] = {
                    "gene_symbol": mutation.gene_symbol,
                    "chrom": mutation.chrom,
                    "positions": []
                }
            gene_groups[mutation.gene_id]["positions"].append(mutation.position)
        
        records = []
        for gene_id, info in gene_groups.items():
            positions = info["positions"]
            # Estimate gene boundaries (expand by 10kb on each side)
            start = min(positions) - 10000
            end = max(positions) + 10000
            
            record = {
                "start": max(1, start),  # Ensure positive coordinates
                "end": end,
                "score": ".",
                "strand": "+",  # Default; should be looked up
                "gene_id": gene_id,
                "gene_name": info["gene_symbol"],
                "gene_type": "protein_coding",  # Default; should be looked up
                "gene_length": end - start,
                "chrom": info["chrom"]
            }
            records.append(record)
        
        return pd.DataFrame(records)
    
    def ingest(self, force_refresh: bool = False, validate: bool = True) -> IngestionResult:
        """
        Complete ingestion workflow.
        
        Parameters
        ----------
        force_refresh : bool
            Whether to redownload data even if cached
        validate : bool
            Whether to validate parsed mutations
            
        Returns
        -------
        IngestionResult
            Processed data and statistics
        """
        # Download data
        print(f"Downloading data from {self.__class__.__name__}...")
        data_path = self.download_data(force_refresh=force_refresh)
        self.processing_stats["total_downloaded"] = 1
        
        # Parse raw data
        print("Parsing raw data...")
        mutations = self.parse_raw_data(data_path)
        self.processing_stats["total_processed"] = len(mutations)
        
        # Validate mutations if requested
        if validate:
            print("Validating mutations...")
            mutations = self.validate_mutations(mutations)
        
        # Create compatible annotation files
        print("Creating annotation files...")
        splice_sites_df = self.create_splice_sites_annotation(mutations)
        gene_features_df = self.create_gene_features(mutations)
        
        # Create result
        result = IngestionResult(
            mutations=mutations,
            splice_sites_df=splice_sites_df,
            gene_features_df=gene_features_df,
            total_records=self.processing_stats["total_processed"],
            successfully_parsed=len(mutations),
            parsing_errors=self.processing_stats["parsing_errors"]
        )
        
        # Save processed data
        self.save_processed_data(result)
        
        return result
    
    def save_processed_data(self, result: IngestionResult) -> None:
        """Save processed data to output directory."""
        
        # Save annotation files
        splice_sites_path = self.output_dir / "splice_sites.tsv"
        result.splice_sites_df.to_csv(splice_sites_path, sep="\t", index=False)
        
        gene_features_path = self.output_dir / "gene_features.tsv" 
        result.gene_features_df.to_csv(gene_features_path, sep="\t", index=False)
        
        if result.transcript_features_df is not None:
            transcript_path = self.output_dir / "transcript_features.tsv"
            result.transcript_features_df.to_csv(transcript_path, sep="\t", index=False)
        
        # Save mutations as JSON for detailed analysis
        import json
        mutations_path = self.output_dir / "mutations.json"
        mutations_data = []
        for mutation in result.mutations:
            mutation_dict = {
                "chrom": mutation.chrom,
                "position": mutation.position,
                "ref_allele": mutation.ref_allele,
                "alt_allele": mutation.alt_allele,
                "gene_id": mutation.gene_id,
                "gene_symbol": mutation.gene_symbol,
                "transcript_id": mutation.transcript_id,
                "splice_event_type": mutation.splice_event_type.value,
                "affected_site_type": mutation.affected_site_type,
                "splice_site_position": mutation.splice_site_position,
                "clinical_significance": mutation.clinical_significance.value if mutation.clinical_significance else None,
                "disease_context": mutation.disease_context,
                "experimentally_validated": mutation.experimentally_validated,
                "validation_method": mutation.validation_method,
                "source_database": mutation.source_database,
                "source_id": mutation.source_id,
                "hgvs_notation": mutation.hgvs_notation,
                "metadata": mutation.metadata
            }
            mutations_data.append(mutation_dict)
        
        with open(mutations_path, 'w') as f:
            json.dump(mutations_data, f, indent=2)
        
        print(f"Saved processed data to {self.output_dir}")
        print(f"  - splice_sites.tsv: {len(result.splice_sites_df)} entries")
        print(f"  - gene_features.tsv: {len(result.gene_features_df)} entries") 
        print(f"  - mutations.json: {len(result.mutations)} mutations") 