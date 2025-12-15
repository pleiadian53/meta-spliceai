"""Enhanced Gene Mapping with Multiple Strategies

This module provides robust gene mapping across genomic builds using multiple
strategies beyond simple gene name matching.

Mapping Strategies:
1. Gene Symbol Matching (simple but fast)
2. Ensembl ID Matching (stable across builds)
3. Coordinate-Based Matching (gene boundaries with liftover)
4. External Cross-References (NCBI Gene, HGNC)

Key Concepts:
- **Primary Key**: Ensembl ID (most stable)
- **Secondary Key**: Gene Symbol (human-readable)
- **Tertiary Key**: Coordinates (genomic location)
- **Confidence Score**: How certain we are about the mapping

Examples:
    >>> from meta_spliceai.system.genomic_resources import EnhancedGeneMapper
    >>> 
    >>> mapper = EnhancedGeneMapper()
    >>> mapper.add_source_from_file('ensembl', 'GRCh37', 'data/ensembl/GRCh37/gene_features.tsv')
    >>> mapper.add_source_from_file('mane', 'GRCh38', 'data/mane/GRCh38/gene_features.tsv')
    >>> 
    >>> # Find mappings with confidence scores
    >>> mappings = mapper.find_mappings('ensembl/GRCh37', 'mane/GRCh38')
    >>> 
    >>> # Get high-confidence mappings only
    >>> high_conf = [m for m in mappings if m.confidence >= 0.9]
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Tuple, Union
from pathlib import Path
from enum import Enum
import polars as pl
import pandas as pd


class MappingStrategy(Enum):
    """Strategy used to map genes between sources."""
    ENSEMBL_ID = "ensembl_id"           # Matched by Ensembl stable ID
    GENE_SYMBOL = "gene_symbol"         # Matched by gene name
    COORDINATES = "coordinates"         # Matched by genomic location
    HGNC_ID = "hgnc_id"                # Matched by HGNC ID
    NCBI_GENE_ID = "ncbi_gene_id"      # Matched by NCBI Gene ID
    MULTIPLE = "multiple"               # Matched by multiple strategies


@dataclass
class GeneMapping:
    """Mapping between genes in different sources.
    
    Attributes
    ----------
    source1_key : str
        Source key (e.g., 'ensembl/GRCh37')
    source2_key : str
        Target source key (e.g., 'mane/GRCh38')
    gene_symbol : str
        Gene symbol (common identifier)
    source1_gene_id : str
        Gene ID in source 1
    source2_gene_id : str
        Gene ID in source 2
    strategy : MappingStrategy
        Strategy used for mapping
    confidence : float
        Confidence score (0.0 to 1.0)
    metadata : Dict
        Additional mapping information
    """
    source1_key: str
    source2_key: str
    gene_symbol: str
    source1_gene_id: str
    source2_gene_id: str
    strategy: MappingStrategy
    confidence: float = 1.0
    metadata: Dict = field(default_factory=dict)
    
    def __repr__(self):
        return (f"GeneMapping({self.gene_symbol}: {self.source1_gene_id} → "
                f"{self.source2_gene_id}, {self.strategy.value}, conf={self.confidence:.2f})")


@dataclass
class GeneInfo:
    """Comprehensive gene information."""
    gene_symbol: str
    gene_id: str
    ensembl_id: Optional[str] = None
    hgnc_id: Optional[str] = None
    ncbi_gene_id: Optional[str] = None
    source: str = ""
    build: str = ""
    chrom: str = ""
    start: int = 0
    end: int = 0
    strand: str = "+"
    gene_type: str = ""
    
    def overlaps(self, other: 'GeneInfo', min_overlap: float = 0.5) -> bool:
        """Check if this gene overlaps with another gene.
        
        Parameters
        ----------
        other : GeneInfo
            Another gene to check overlap with
        min_overlap : float
            Minimum fraction of overlap required (0.0 to 1.0)
        
        Returns
        -------
        bool
            True if genes overlap by at least min_overlap
        """
        if self.chrom != other.chrom or self.strand != other.strand:
            return False
        
        # Calculate overlap
        overlap_start = max(self.start, other.start)
        overlap_end = min(self.end, other.end)
        
        if overlap_start >= overlap_end:
            return False
        
        overlap_len = overlap_end - overlap_start
        self_len = self.end - self.start
        other_len = other.end - other.start
        
        # Check if overlap is sufficient
        overlap_frac_self = overlap_len / self_len if self_len > 0 else 0
        overlap_frac_other = overlap_len / other_len if other_len > 0 else 0
        
        return max(overlap_frac_self, overlap_frac_other) >= min_overlap


class EnhancedGeneMapper:
    """Enhanced gene mapper with multiple mapping strategies.
    
    This class provides robust gene mapping using:
    1. Gene symbol matching
    2. Ensembl ID matching
    3. Coordinate-based matching
    4. External ID cross-references
    
    Examples
    --------
    >>> mapper = EnhancedGeneMapper()
    >>> mapper.add_source_from_file('ensembl', 'GRCh37', 'data/ensembl/GRCh37/gene_features.tsv')
    >>> mapper.add_source_from_file('mane', 'GRCh38', 'data/mane/GRCh38/gene_features.tsv')
    >>> 
    >>> # Find all mappings
    >>> mappings = mapper.find_mappings('ensembl/GRCh37', 'mane/GRCh38')
    >>> 
    >>> # Get high-confidence mappings
    >>> high_conf = mapper.get_high_confidence_mappings(
    ...     'ensembl/GRCh37', 'mane/GRCh38', min_confidence=0.9
    ... )
    """
    
    def __init__(self):
        self.sources: Dict[str, pl.DataFrame] = {}
        self.gene_info: Dict[str, Dict[str, GeneInfo]] = {}  # {source_key: {gene_id: GeneInfo}}
        self.symbol_to_genes: Dict[str, Dict[str, List[str]]] = {}  # {symbol: {source_key: [gene_ids]}}
        self.ensembl_id_to_genes: Dict[str, Dict[str, str]] = {}  # {ensembl_id: {source_key: gene_id}}
    
    def add_source_from_file(
        self,
        source: str,
        build: str,
        file_path: Union[str, Path],
        separator: str = '\t',
        ensembl_id_col: Optional[str] = None,
        hgnc_id_col: Optional[str] = None,
        ncbi_gene_id_col: Optional[str] = None,
        external_id_mapping: Optional[Dict[str, str]] = None
    ) -> None:
        """Add a gene annotation source from file.
        
        Parameters
        ----------
        source : str
            Annotation source name (e.g., 'ensembl', 'mane')
        build : str
            Genomic build (e.g., 'GRCh37', 'GRCh38')
        file_path : Union[str, Path]
            Path to gene features file
        separator : str, default='\t'
            File separator
        ensembl_id_col : Optional[str]
            Column name for Ensembl IDs (auto-detected if None)
        hgnc_id_col : Optional[str]
            Column name for HGNC IDs
        ncbi_gene_id_col : Optional[str]
            Column name for NCBI Gene IDs
        external_id_mapping : Optional[Dict[str, str]]
            External mapping from gene_id to Ensembl ID (for MANE)
        """
        # Read file
        df = pl.read_csv(
            str(file_path),
            separator=separator,
            schema_overrides={'chrom': pl.Utf8, 'seqname': pl.Utf8}
        )
        
        source_key = f"{source}/{build}"
        self.sources[source_key] = df
        
        # Initialize structures
        self.gene_info[source_key] = {}
        
        # Auto-detect Ensembl ID column
        if ensembl_id_col is None:
            for col in df.columns:
                if 'ensembl' in col.lower() or col == 'gene_id':
                    # Check if values start with 'ENSG'
                    sample = df.select(pl.col(col).head(10)).to_series().to_list()
                    if any(str(v).startswith('ENSG') for v in sample if v):
                        ensembl_id_col = col
                        break
        
        # Process each gene
        for row in df.iter_rows(named=True):
            gene_id = row.get('gene_id', '')
            gene_symbol = row.get('gene_name', gene_id)
            
            if not gene_id or not gene_symbol:
                continue
            
            # Extract Ensembl ID
            ensembl_id = None
            if ensembl_id_col and ensembl_id_col in row:
                ensembl_id = row[ensembl_id_col]
            elif gene_id.startswith('ENSG'):
                ensembl_id = gene_id
            elif external_id_mapping and gene_id in external_id_mapping:
                # Use external mapping (e.g., MANE gene-XXX → ENSGXXX)
                ensembl_id = external_id_mapping[gene_id]
            
            # Extract other IDs
            hgnc_id = row.get(hgnc_id_col) if hgnc_id_col else None
            ncbi_gene_id = row.get(ncbi_gene_id_col) if ncbi_gene_id_col else None
            
            # Create GeneInfo
            gene_info = GeneInfo(
                gene_symbol=gene_symbol,
                gene_id=gene_id,
                ensembl_id=ensembl_id,
                hgnc_id=hgnc_id,
                ncbi_gene_id=ncbi_gene_id,
                source=source,
                build=build,
                chrom=row.get('chrom', row.get('seqname', '')),
                start=row.get('start', 0),
                end=row.get('end', 0),
                strand=row.get('strand', '+'),
                gene_type=row.get('gene_type', '')
            )
            
            # Store gene info
            self.gene_info[source_key][gene_id] = gene_info
            
            # Index by symbol
            if gene_symbol not in self.symbol_to_genes:
                self.symbol_to_genes[gene_symbol] = {}
            if source_key not in self.symbol_to_genes[gene_symbol]:
                self.symbol_to_genes[gene_symbol][source_key] = []
            self.symbol_to_genes[gene_symbol][source_key].append(gene_id)
            
            # Index by Ensembl ID
            if ensembl_id:
                if ensembl_id not in self.ensembl_id_to_genes:
                    self.ensembl_id_to_genes[ensembl_id] = {}
                self.ensembl_id_to_genes[ensembl_id][source_key] = gene_id
    
    def find_mappings(
        self,
        source1_key: str,
        source2_key: str,
        strategies: Optional[List[MappingStrategy]] = None,
        min_coordinate_overlap: float = 0.5
    ) -> List[GeneMapping]:
        """Find all gene mappings between two sources.
        
        Parameters
        ----------
        source1_key : str
            Source key (e.g., 'ensembl/GRCh37')
        source2_key : str
            Target source key (e.g., 'mane/GRCh38')
        strategies : Optional[List[MappingStrategy]]
            Strategies to use (all if None)
        min_coordinate_overlap : float, default=0.5
            Minimum overlap for coordinate-based matching
        
        Returns
        -------
        List[GeneMapping]
            List of gene mappings with confidence scores
        """
        if strategies is None:
            strategies = [
                MappingStrategy.ENSEMBL_ID,
                MappingStrategy.GENE_SYMBOL,
                MappingStrategy.COORDINATES
            ]
        
        mappings = []
        mapped_pairs = set()  # Track (source1_gene_id, source2_gene_id) to avoid duplicates
        
        # Strategy 1: Ensembl ID matching (highest confidence)
        if MappingStrategy.ENSEMBL_ID in strategies:
            for ensembl_id, source_genes in self.ensembl_id_to_genes.items():
                if source1_key in source_genes and source2_key in source_genes:
                    gene1_id = source_genes[source1_key]
                    gene2_id = source_genes[source2_key]
                    
                    if (gene1_id, gene2_id) in mapped_pairs:
                        continue
                    
                    gene1_info = self.gene_info[source1_key][gene1_id]
                    
                    mapping = GeneMapping(
                        source1_key=source1_key,
                        source2_key=source2_key,
                        gene_symbol=gene1_info.gene_symbol,
                        source1_gene_id=gene1_id,
                        source2_gene_id=gene2_id,
                        strategy=MappingStrategy.ENSEMBL_ID,
                        confidence=1.0,
                        metadata={'ensembl_id': ensembl_id}
                    )
                    mappings.append(mapping)
                    mapped_pairs.add((gene1_id, gene2_id))
        
        # Strategy 2: Gene symbol matching (medium confidence)
        if MappingStrategy.GENE_SYMBOL in strategies:
            for symbol, source_genes in self.symbol_to_genes.items():
                if source1_key in source_genes and source2_key in source_genes:
                    # Handle one-to-one mappings
                    genes1 = source_genes[source1_key]
                    genes2 = source_genes[source2_key]
                    
                    for gene1_id in genes1:
                        for gene2_id in genes2:
                            if (gene1_id, gene2_id) in mapped_pairs:
                                continue
                            
                            # Lower confidence if multiple genes per symbol
                            confidence = 0.9
                            if len(genes1) > 1 or len(genes2) > 1:
                                confidence = 0.7  # Ambiguous mapping
                            
                            mapping = GeneMapping(
                                source1_key=source1_key,
                                source2_key=source2_key,
                                gene_symbol=symbol,
                                source1_gene_id=gene1_id,
                                source2_gene_id=gene2_id,
                                strategy=MappingStrategy.GENE_SYMBOL,
                                confidence=confidence,
                                metadata={
                                    'num_genes_source1': len(genes1),
                                    'num_genes_source2': len(genes2)
                                }
                            )
                            mappings.append(mapping)
                            mapped_pairs.add((gene1_id, gene2_id))
        
        # Strategy 3: Coordinate-based matching (lower confidence, requires liftover)
        if MappingStrategy.COORDINATES in strategies:
            # Only attempt if builds are the same (no liftover needed)
            build1 = source1_key.split('/')[1].split('_')[0]  # Extract base build
            build2 = source2_key.split('/')[1].split('_')[0]
            
            if build1 == build2:
                # Same build - can use coordinates directly
                for gene1_id, gene1_info in self.gene_info[source1_key].items():
                    for gene2_id, gene2_info in self.gene_info[source2_key].items():
                        if (gene1_id, gene2_id) in mapped_pairs:
                            continue
                        
                        if gene1_info.overlaps(gene2_info, min_coordinate_overlap):
                            # Calculate confidence based on overlap
                            overlap_start = max(gene1_info.start, gene2_info.start)
                            overlap_end = min(gene1_info.end, gene2_info.end)
                            overlap_len = overlap_end - overlap_start
                            
                            gene1_len = gene1_info.end - gene1_info.start
                            gene2_len = gene2_info.end - gene2_info.start
                            
                            overlap_frac = overlap_len / max(gene1_len, gene2_len)
                            confidence = 0.5 + (overlap_frac * 0.4)  # 0.5 to 0.9
                            
                            mapping = GeneMapping(
                                source1_key=source1_key,
                                source2_key=source2_key,
                                gene_symbol=gene1_info.gene_symbol,
                                source1_gene_id=gene1_id,
                                source2_gene_id=gene2_id,
                                strategy=MappingStrategy.COORDINATES,
                                confidence=confidence,
                                metadata={
                                    'overlap_fraction': overlap_frac,
                                    'chrom': gene1_info.chrom
                                }
                            )
                            mappings.append(mapping)
                            mapped_pairs.add((gene1_id, gene2_id))
        
        return mappings
    
    def get_high_confidence_mappings(
        self,
        source1_key: str,
        source2_key: str,
        min_confidence: float = 0.9
    ) -> List[GeneMapping]:
        """Get only high-confidence gene mappings.
        
        Parameters
        ----------
        source1_key : str
            Source key
        source2_key : str
            Target source key
        min_confidence : float, default=0.9
            Minimum confidence threshold
        
        Returns
        -------
        List[GeneMapping]
            High-confidence mappings
        """
        all_mappings = self.find_mappings(source1_key, source2_key)
        return [m for m in all_mappings if m.confidence >= min_confidence]
    
    def get_mapping_summary(
        self,
        source1_key: str,
        source2_key: str
    ) -> Dict:
        """Get summary statistics about mappings between two sources.
        
        Parameters
        ----------
        source1_key : str
            Source key
        source2_key : str
            Target source key
        
        Returns
        -------
        Dict
            Summary statistics
        """
        mappings = self.find_mappings(source1_key, source2_key)
        
        # Count by strategy
        strategy_counts = {}
        for m in mappings:
            strategy_counts[m.strategy.value] = strategy_counts.get(m.strategy.value, 0) + 1
        
        # Count by confidence level
        high_conf = sum(1 for m in mappings if m.confidence >= 0.9)
        medium_conf = sum(1 for m in mappings if 0.7 <= m.confidence < 0.9)
        low_conf = sum(1 for m in mappings if m.confidence < 0.7)
        
        return {
            'total_mappings': len(mappings),
            'by_strategy': strategy_counts,
            'by_confidence': {
                'high (≥0.9)': high_conf,
                'medium (0.7-0.9)': medium_conf,
                'low (<0.7)': low_conf
            },
            'unique_genes_source1': len(set(m.source1_gene_id for m in mappings)),
            'unique_genes_source2': len(set(m.source2_gene_id for m in mappings))
        }
    
    def to_dataframe(
        self,
        source1_key: str,
        source2_key: str,
        min_confidence: float = 0.0
    ) -> pl.DataFrame:
        """Export mappings to a DataFrame.
        
        Parameters
        ----------
        source1_key : str
            Source key
        source2_key : str
            Target source key
        min_confidence : float, default=0.0
            Minimum confidence threshold
        
        Returns
        -------
        pl.DataFrame
            Mappings as DataFrame
        """
        mappings = self.find_mappings(source1_key, source2_key)
        mappings = [m for m in mappings if m.confidence >= min_confidence]
        
        data = {
            'gene_symbol': [m.gene_symbol for m in mappings],
            'source1_gene_id': [m.source1_gene_id for m in mappings],
            'source2_gene_id': [m.source2_gene_id for m in mappings],
            'strategy': [m.strategy.value for m in mappings],
            'confidence': [m.confidence for m in mappings]
        }
        
        return pl.DataFrame(data)
    
    def print_summary(self, source1_key: str, source2_key: str):
        """Print a formatted summary of mappings.
        
        Parameters
        ----------
        source1_key : str
            Source key
        source2_key : str
            Target source key
        """
        summary = self.get_mapping_summary(source1_key, source2_key)
        
        print("=" * 80)
        print(f"GENE MAPPING SUMMARY: {source1_key} → {source2_key}")
        print("=" * 80)
        print()
        print(f"Total mappings: {summary['total_mappings']:,}")
        print()
        print("By Strategy:")
        for strategy, count in summary['by_strategy'].items():
            pct = count / summary['total_mappings'] * 100 if summary['total_mappings'] > 0 else 0
            print(f"  {strategy:20s}: {count:,} ({pct:.1f}%)")
        print()
        print("By Confidence:")
        for level, count in summary['by_confidence'].items():
            pct = count / summary['total_mappings'] * 100 if summary['total_mappings'] > 0 else 0
            print(f"  {level:20s}: {count:,} ({pct:.1f}%)")
        print()
        print(f"Unique genes in {source1_key}: {summary['unique_genes_source1']:,}")
        print(f"Unique genes in {source2_key}: {summary['unique_genes_source2']:,}")
        print("=" * 80)

