"""
SpliceVarDB loader for meta-layer variant-effect evaluation.

This module loads SpliceVarDB variants and prepares them for
evaluating the meta-layer's ability to detect variant effects on splicing.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)

# SpliceVarDB data file search paths (in priority order)
# 1. Standard data directory (recommended for both local and remote)
# 2. Legacy package path (backwards compatibility)

def _find_splicevardb_path() -> Path:
    """
    Find SpliceVarDB data file by searching multiple locations.
    
    Search order:
    1. data/splicevardb/ (standard data directory, works on local and remote)
    2. case_studies/workflows/splicevardb/ (legacy package location)
    
    Returns
    -------
    Path
        Path to the SpliceVarDB TSV file
    """
    from meta_spliceai.system.config import find_project_root
    
    project_root = Path(find_project_root(str(Path(__file__))))
    
    # Search locations in priority order
    search_paths = [
        # Standard data directory (recommended)
        project_root / "data" / "splicevardb" / "splicevardb.download.tsv",
        # Legacy package location (backwards compatibility)
        project_root / "meta_spliceai" / "splice_engine" / "case_studies" / "workflows" / "splicevardb" / "splicevardb.download.tsv",
        # RunPods workspace location
        Path("/workspace/meta-spliceai/data/splicevardb/splicevardb.download.tsv"),
    ]
    
    for path in search_paths:
        if path.exists():
            logger.debug(f"Found SpliceVarDB at: {path}")
            return path
    
    # Return first path (for error message purposes)
    return search_paths[0]


DEFAULT_SPLICEVARDB_PATH = _find_splicevardb_path()


@dataclass
class VariantRecord:
    """
    Single variant record from SpliceVarDB.
    
    CLASSIFICATION CONVENTION (taken directly from SpliceVarDB dataset):
    =====================================================================
    
    The `classification` field comes directly from SpliceVarDB and represents
    the experimentally/computationally validated effect of the variant:
    
    - "Splice-altering": Variant has CONFIRMED effect on splicing.
        - Evidence: RNA-seq, minigene assays, or consensus of computational tools
        - Use in training: Trust base model delta as target
    
    - "Non-splice-altering" (or "Normal"): Variant has NO effect on splicing.
        - Evidence: Same validation methods showing no effect
        - Use in training: Target = [0, 0, 0] (override base model!)
    
    - "Low-frequency": Rare variant, effect is UNCERTAIN.
        - Evidence: Not enough data to confidently classify
        - Use in training: SKIP (do not train on these)
    
    - "Conflicting": Evidence is contradictory.
        - Evidence: Multiple sources disagree
        - Use in training: SKIP (do not train on these)
    
    NOTE: These classifications are NOT position labels (donor/acceptor).
    They describe the VARIANT'S EFFECT on splicing, not what type of
    splice site the position is.
    
    See docs/data/SPLICEVARDB.md for full documentation.
    """
    
    variant_id: int
    chrom: str
    position: int
    ref_allele: str
    alt_allele: str
    gene: str
    hgvs: str
    method: str
    classification: str  # "Splice-altering", "Non-splice-altering", "Low-frequency", "Conflicting"
    location: str  # "Exonic", "Intronic"
    
    @property
    def is_splice_altering(self) -> bool:
        """Whether this variant affects splicing."""
        return self.classification == "Splice-altering"
    
    @property
    def is_non_splice_altering(self) -> bool:
        """Whether this variant does NOT affect splicing."""
        return self.classification == "Non-splice-altering"
    
    @property
    def is_low_frequency(self) -> bool:
        """Whether this is a low-frequency (uncertain) variant."""
        return self.classification == "Low-frequency"
    
    def get_coordinate_key(self) -> str:
        """Get a unique coordinate key for matching."""
        return f"{self.chrom}:{self.position}"


class SpliceVarDBLoader:
    """
    Load and parse SpliceVarDB for meta-layer evaluation.
    
    Parameters
    ----------
    data_path : Path, optional
        Path to SpliceVarDB TSV file.
    genome_build : str
        Genome build to use: 'GRCh37'/'hg19' or 'GRCh38'/'hg38'.
    
    Examples
    --------
    >>> loader = SpliceVarDBLoader(genome_build='GRCh38')
    >>> variants = loader.load_all()
    >>> print(f"Loaded {len(variants)} variants")
    >>> 
    >>> # Filter by classification
    >>> splice_altering = loader.get_splice_altering()
    >>> print(f"Splice-altering: {len(splice_altering)}")
    """
    
    def __init__(
        self,
        data_path: Optional[Path] = None,
        genome_build: str = 'GRCh38'
    ):
        self.data_path = Path(data_path) if data_path else DEFAULT_SPLICEVARDB_PATH
        
        # Normalize genome build
        if genome_build.lower() in ['grch38', 'hg38']:
            self.coord_column = 'hg38'
            self.genome_build = 'GRCh38'
        else:
            self.coord_column = 'hg19'
            self.genome_build = 'GRCh37'
        
        self._df: Optional[pl.DataFrame] = None
        self._variants: Optional[List[VariantRecord]] = None
    
    def load_dataframe(self) -> pl.DataFrame:
        """Load raw SpliceVarDB data as DataFrame."""
        if self._df is None:
            if not self.data_path.exists():
                raise FileNotFoundError(
                    f"SpliceVarDB data not found at {self.data_path}. "
                    "Please download from SpliceVarDB website."
                )
            
            self._df = pl.read_csv(self.data_path, separator='\t')
            
            # Parse coordinates from the appropriate column
            self._df = self._df.with_columns([
                # Parse variant string like "1-100107682-T-C"
                pl.col(self.coord_column).str.strip_chars('"').str.split('-').list.get(0).alias('chrom'),
                pl.col(self.coord_column).str.strip_chars('"').str.split('-').list.get(1).cast(pl.Int64).alias('position'),
                pl.col(self.coord_column).str.strip_chars('"').str.split('-').list.get(2).alias('ref_allele'),
                pl.col(self.coord_column).str.strip_chars('"').str.split('-').list.get(3).alias('alt_allele'),
            ])
            
            logger.info(f"Loaded {len(self._df)} variants from SpliceVarDB")
            logger.info(f"  Using {self.genome_build} coordinates ({self.coord_column})")
            
            # Log classification distribution
            counts = self._df.group_by('classification').count()
            for row in counts.iter_rows(named=True):
                logger.info(f"  {row['classification']}: {row['count']:,}")
        
        return self._df
    
    def load_all(self) -> List[VariantRecord]:
        """Load all variants as VariantRecord objects."""
        if self._variants is None:
            df = self.load_dataframe()
            self._variants = self._df_to_records(df)
        return self._variants
    
    def get_splice_altering(self) -> List[VariantRecord]:
        """Get only splice-altering variants."""
        df = self.load_dataframe()
        filtered = df.filter(pl.col('classification') == 'Splice-altering')
        return self._df_to_records(filtered)
    
    def get_non_splice_altering(self) -> List[VariantRecord]:
        """Get only non-splice-altering variants."""
        df = self.load_dataframe()
        filtered = df.filter(pl.col('classification') == 'Non-splice-altering')
        return self._df_to_records(filtered)
    
    def get_low_frequency(self) -> List[VariantRecord]:
        """Get low-frequency (uncertain) variants."""
        df = self.load_dataframe()
        filtered = df.filter(pl.col('classification') == 'Low-frequency')
        return self._df_to_records(filtered)
    
    def get_by_gene(self, gene: str) -> List[VariantRecord]:
        """Get variants for a specific gene."""
        df = self.load_dataframe()
        filtered = df.filter(pl.col('gene') == gene)
        return self._df_to_records(filtered)
    
    def get_by_chromosome(self, chrom: str) -> List[VariantRecord]:
        """Get variants on a specific chromosome."""
        df = self.load_dataframe()
        # Normalize chromosome name (remove 'chr' prefix if present)
        chrom_clean = chrom.replace('chr', '')
        filtered = df.filter(pl.col('chrom') == chrom_clean)
        return self._df_to_records(filtered)
    
    def iter_variants(
        self,
        classification: Optional[str] = None,
        chromosomes: Optional[List[str]] = None
    ) -> Iterator[VariantRecord]:
        """
        Iterate over variants with optional filtering.
        
        Parameters
        ----------
        classification : str, optional
            Filter to specific classification.
        chromosomes : list of str, optional
            Filter to specific chromosomes.
        
        Yields
        ------
        VariantRecord
            Variant records matching filters.
        """
        df = self.load_dataframe()
        
        if classification:
            df = df.filter(pl.col('classification') == classification)
        
        if chromosomes:
            chroms_clean = [c.replace('chr', '') for c in chromosomes]
            df = df.filter(pl.col('chrom').is_in(chroms_clean))
        
        for row in df.iter_rows(named=True):
            yield self._row_to_record(row)
    
    def get_train_test_split(
        self,
        test_chromosomes: List[str] = ['21', '22', 'X'],
        classification: Optional[str] = None
    ) -> Tuple[List[VariantRecord], List[VariantRecord]]:
        """
        Split variants into train and test by chromosome.
        
        Parameters
        ----------
        test_chromosomes : list of str
            Chromosomes to hold out for testing.
        classification : str, optional
            Filter to specific classification.
        
        Returns
        -------
        tuple
            (train_variants, test_variants)
        """
        df = self.load_dataframe()
        
        if classification:
            df = df.filter(pl.col('classification') == classification)
        
        test_chroms = [c.replace('chr', '') for c in test_chromosomes]
        
        train_df = df.filter(~pl.col('chrom').is_in(test_chroms))
        test_df = df.filter(pl.col('chrom').is_in(test_chroms))
        
        train_variants = self._df_to_records(train_df)
        test_variants = self._df_to_records(test_df)
        
        logger.info(f"Train/test split: {len(train_variants)} / {len(test_variants)}")
        
        return train_variants, test_variants
    
    def get_statistics(self) -> Dict[str, int]:
        """Get summary statistics of the dataset."""
        df = self.load_dataframe()
        
        stats = {
            'total_variants': len(df),
        }
        
        # By classification
        for row in df.group_by('classification').count().iter_rows(named=True):
            key = row['classification'].lower().replace('-', '_').replace(' ', '_')
            stats[f'n_{key}'] = row['count']
        
        # By location
        for row in df.group_by('location').count().iter_rows(named=True):
            if row['location']:
                key = row['location'].lower()
                stats[f'n_{key}'] = row['count']
        
        # By method
        for row in df.group_by('method').count().iter_rows(named=True):
            if row['method']:
                key = row['method'].lower().replace('-', '_').replace(' ', '_')
                stats[f'n_method_{key}'] = row['count']
        
        # Unique genes
        stats['n_genes'] = df['gene'].n_unique()
        
        return stats
    
    def _df_to_records(self, df: pl.DataFrame) -> List[VariantRecord]:
        """Convert DataFrame to list of VariantRecord."""
        return [self._row_to_record(row) for row in df.iter_rows(named=True)]
    
    def _row_to_record(self, row: Dict) -> VariantRecord:
        """Convert a single row to VariantRecord."""
        return VariantRecord(
            variant_id=row['variant_id'],
            chrom=row['chrom'],
            position=row['position'],
            ref_allele=row['ref_allele'],
            alt_allele=row['alt_allele'],
            gene=row.get('gene', ''),
            hgvs=row.get('hgvs', ''),
            method=row.get('method', ''),
            classification=row.get('classification', ''),
            location=row.get('location', '')
        )


def load_splicevardb(
    genome_build: str = 'GRCh38',
    data_path: Optional[Path] = None
) -> SpliceVarDBLoader:
    """
    Convenience function to load SpliceVarDB.
    
    Parameters
    ----------
    genome_build : str
        Genome build to use.
    data_path : Path, optional
        Custom path to data file.
    
    Returns
    -------
    SpliceVarDBLoader
        Loaded SpliceVarDB instance.
    """
    return SpliceVarDBLoader(data_path=data_path, genome_build=genome_build)

