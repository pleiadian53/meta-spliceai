"""Gene mapping across genomic builds and annotation sources.

This module provides utilities for mapping genes across different:
- Genomic builds (GRCh37, GRCh38)
- Annotation sources (Ensembl, MANE, GENCODE)
- Identifier systems (Ensembl IDs, gene symbols, RefSeq IDs)

Key Concepts
------------
- **Gene Symbol**: Human-readable name (e.g., 'BRCA1', 'TP53')
- **Ensembl ID**: Stable identifier (e.g., 'ENSG00000012048')
- **MANE ID**: MANE-specific format (e.g., 'gene-BRCA1')
- **Cross-Build Mapping**: Linking genes across GRCh37 ↔ GRCh38

Examples
--------
Basic usage:

    >>> mapper = GeneMapper()
    >>> mapper.add_source('ensembl', 'GRCh37', ensembl_gene_features_df)
    >>> mapper.add_source('mane', 'GRCh38', mane_gene_features_df)
    >>> 
    >>> # Find common genes
    >>> common_genes = mapper.get_common_genes(['ensembl/GRCh37', 'mane/GRCh38'])
    >>> 
    >>> # Map gene names to source-specific IDs
    >>> gene_ids = mapper.map_genes_to_source(['BRCA1', 'TP53'], 'mane/GRCh38')
    >>> # Returns: ['gene-BRCA1', 'gene-TP53']

Integration with workflows:

    >>> from meta_spliceai.system.genomic_resources import get_gene_mapper
    >>> mapper = get_gene_mapper()
    >>> 
    >>> # Sample genes that exist in both builds
    >>> common = mapper.get_common_genes(['ensembl/GRCh37', 'mane/GRCh38'])
    >>> sampled = common.sample(n=20)
    >>> 
    >>> # Map to SpliceAI (Ensembl/GRCh37)
    >>> spliceai_genes = mapper.map_genes_to_source(sampled, 'ensembl/GRCh37')
    >>> 
    >>> # Map to OpenSpliceAI (MANE/GRCh38)
    >>> openspliceai_genes = mapper.map_genes_to_source(sampled, 'mane/GRCh38')
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Tuple, Union
from pathlib import Path
import polars as pl
import pandas as pd


@dataclass
class GeneInfo:
    """Information about a gene in a specific annotation source.
    
    Attributes
    ----------
    gene_symbol : str
        Human-readable gene name (e.g., 'BRCA1')
    gene_id : str
        Source-specific gene identifier
    ensembl_id : Optional[str]
        Ensembl stable ID if available
    source : str
        Annotation source (e.g., 'ensembl', 'mane')
    build : str
        Genomic build (e.g., 'GRCh37', 'GRCh38')
    chrom : str
        Chromosome
    start : int
        Gene start position
    end : int
        Gene end position
    strand : str
        Strand ('+' or '-')
    gene_type : str
        Gene biotype (e.g., 'protein_coding', 'lncRNA')
    """
    gene_symbol: str
    gene_id: str
    ensembl_id: Optional[str] = None
    source: str = ""
    build: str = ""
    chrom: str = ""
    start: int = 0
    end: int = 0
    strand: str = "+"
    gene_type: str = ""


@dataclass
class GeneMappingResult:
    """Result of gene mapping operation.
    
    Attributes
    ----------
    gene_symbol : str
        Original gene symbol
    source_key : str
        Target source (e.g., 'mane/GRCh38')
    gene_id : Optional[str]
        Mapped gene ID in target source
    found : bool
        Whether mapping was successful
    gene_info : Optional[GeneInfo]
        Full gene information if found
    """
    gene_symbol: str
    source_key: str
    gene_id: Optional[str] = None
    found: bool = False
    gene_info: Optional[GeneInfo] = None


class GeneMapper:
    """Maps genes across genomic builds and annotation sources.
    
    This class maintains a registry of genes from different sources and provides
    methods to find common genes and map gene identifiers between sources.
    
    Attributes
    ----------
    sources : Dict[str, pl.DataFrame]
        Registry of gene information by source key (e.g., 'ensembl/GRCh37')
    symbol_to_info : Dict[str, Dict[str, GeneInfo]]
        Mapping from gene symbol to source-specific information
    
    Examples
    --------
    >>> mapper = GeneMapper()
    >>> mapper.add_source_from_file('ensembl', 'GRCh37', 'data/ensembl/gene_features.tsv')
    >>> mapper.add_source_from_file('mane', 'GRCh38', 'data/mane/GRCh38/gene_features.tsv')
    >>> 
    >>> common = mapper.get_common_genes(['ensembl/GRCh37', 'mane/GRCh38'])
    >>> print(f"Found {len(common)} genes in both sources")
    """
    
    def __init__(self):
        self.sources: Dict[str, pl.DataFrame] = {}
        self.symbol_to_info: Dict[str, Dict[str, GeneInfo]] = {}
        self._ensembl_id_to_symbol: Dict[str, str] = {}
    
    def add_source(
        self,
        source: str,
        build: str,
        gene_df: Union[pl.DataFrame, pd.DataFrame],
        gene_id_col: str = 'gene_id',
        gene_name_col: str = 'gene_name'
    ) -> None:
        """Add a gene annotation source to the mapper.
        
        Parameters
        ----------
        source : str
            Annotation source name (e.g., 'ensembl', 'mane', 'gencode')
        build : str
            Genomic build (e.g., 'GRCh37', 'GRCh38')
        gene_df : Union[pl.DataFrame, pd.DataFrame]
            DataFrame with gene information
        gene_id_col : str, default='gene_id'
            Column name for gene IDs
        gene_name_col : str, default='gene_name'
            Column name for gene symbols
        """
        # Convert to polars if needed
        if isinstance(gene_df, pd.DataFrame):
            gene_df = pl.from_pandas(gene_df)
        
        source_key = f"{source}/{build}"
        self.sources[source_key] = gene_df
        
        # Build symbol-to-info mapping
        for row in gene_df.iter_rows(named=True):
            gene_symbol = row.get(gene_name_col, '')
            gene_id = row.get(gene_id_col, '')
            
            if not gene_symbol or not gene_id:
                continue
            
            # Extract Ensembl ID if present
            ensembl_id = None
            if gene_id.startswith('ENSG'):
                ensembl_id = gene_id
            elif 'ensembl_id' in row:
                ensembl_id = row['ensembl_id']
            
            gene_info = GeneInfo(
                gene_symbol=gene_symbol,
                gene_id=gene_id,
                ensembl_id=ensembl_id,
                source=source,
                build=build,
                chrom=row.get('chrom', row.get('seqname', '')),
                start=row.get('start', 0),
                end=row.get('end', 0),
                strand=row.get('strand', '+'),
                gene_type=row.get('gene_type', '')
            )
            
            if gene_symbol not in self.symbol_to_info:
                self.symbol_to_info[gene_symbol] = {}
            
            self.symbol_to_info[gene_symbol][source_key] = gene_info
            
            # Build Ensembl ID reverse mapping
            if ensembl_id:
                self._ensembl_id_to_symbol[ensembl_id] = gene_symbol
    
    def add_source_from_file(
        self,
        source: str,
        build: str,
        file_path: Union[str, Path],
        separator: str = '\t'
    ) -> None:
        """Add a gene annotation source from a file.
        
        Parameters
        ----------
        source : str
            Annotation source name
        build : str
            Genomic build
        file_path : Union[str, Path]
            Path to gene features file
        separator : str, default='\t'
            File separator
        """
        gene_df = pl.read_csv(
            str(file_path),
            separator=separator,
            schema_overrides={'chrom': pl.Utf8, 'seqname': pl.Utf8}
        )
        self.add_source(source, build, gene_df)
    
    def get_common_genes(
        self,
        source_keys: List[str],
        min_sources: Optional[int] = None
    ) -> List[str]:
        """Find genes that exist in multiple sources.
        
        Parameters
        ----------
        source_keys : List[str]
            List of source keys to check (e.g., ['ensembl/GRCh37', 'mane/GRCh38'])
        min_sources : Optional[int]
            Minimum number of sources a gene must appear in.
            If None, requires all sources.
        
        Returns
        -------
        List[str]
            List of gene symbols found in the specified sources
        
        Examples
        --------
        >>> mapper.get_common_genes(['ensembl/GRCh37', 'mane/GRCh38'])
        ['BRCA1', 'TP53', 'EGFR', ...]
        """
        if min_sources is None:
            min_sources = len(source_keys)
        
        common_genes = []
        for gene_symbol, sources in self.symbol_to_info.items():
            # Count how many requested sources have this gene
            found_in = sum(1 for sk in source_keys if sk in sources)
            if found_in >= min_sources:
                common_genes.append(gene_symbol)
        
        return common_genes
    
    def map_genes_to_source(
        self,
        gene_symbols: List[str],
        target_source_key: str,
        return_type: str = 'gene_id'
    ) -> List[GeneMappingResult]:
        """Map gene symbols to source-specific identifiers.
        
        Parameters
        ----------
        gene_symbols : List[str]
            List of gene symbols to map
        target_source_key : str
            Target source (e.g., 'mane/GRCh38')
        return_type : str, default='gene_id'
            What to return: 'gene_id', 'full' (GeneMappingResult objects)
        
        Returns
        -------
        List[GeneMappingResult] or List[str]
            Mapping results. If return_type='gene_id', returns list of gene IDs
            (None for not found). If return_type='full', returns full mapping results.
        
        Examples
        --------
        >>> mapper.map_genes_to_source(['BRCA1', 'TP53'], 'mane/GRCh38')
        [GeneMappingResult(gene_symbol='BRCA1', gene_id='gene-BRCA1', found=True),
         GeneMappingResult(gene_symbol='TP53', gene_id='gene-TP53', found=True)]
        """
        results = []
        
        for gene_symbol in gene_symbols:
            if gene_symbol in self.symbol_to_info:
                sources = self.symbol_to_info[gene_symbol]
                if target_source_key in sources:
                    gene_info = sources[target_source_key]
                    result = GeneMappingResult(
                        gene_symbol=gene_symbol,
                        source_key=target_source_key,
                        gene_id=gene_info.gene_id,
                        found=True,
                        gene_info=gene_info
                    )
                else:
                    result = GeneMappingResult(
                        gene_symbol=gene_symbol,
                        source_key=target_source_key,
                        found=False
                    )
            else:
                result = GeneMappingResult(
                    gene_symbol=gene_symbol,
                    source_key=target_source_key,
                    found=False
                )
            
            results.append(result)
        
        if return_type == 'gene_id':
            return [r.gene_id for r in results]
        else:
            return results
    
    def get_gene_info(
        self,
        gene_symbol: str,
        source_key: Optional[str] = None
    ) -> Union[GeneInfo, Dict[str, GeneInfo], None]:
        """Get detailed information about a gene.
        
        Parameters
        ----------
        gene_symbol : str
            Gene symbol
        source_key : Optional[str]
            Specific source to query. If None, returns all sources.
        
        Returns
        -------
        Union[GeneInfo, Dict[str, GeneInfo], None]
            Gene information. If source_key specified, returns single GeneInfo.
            If source_key is None, returns dict of {source_key: GeneInfo}.
            Returns None if gene not found.
        """
        if gene_symbol not in self.symbol_to_info:
            return None
        
        if source_key:
            return self.symbol_to_info[gene_symbol].get(source_key)
        else:
            return self.symbol_to_info[gene_symbol]
    
    def get_intersection_dataframe(
        self,
        source_keys: List[str],
        include_all_info: bool = False
    ) -> pl.DataFrame:
        """Get a DataFrame of genes in the intersection with their IDs per source.
        
        Parameters
        ----------
        source_keys : List[str]
            List of source keys
        include_all_info : bool, default=False
            If True, include position and type information
        
        Returns
        -------
        pl.DataFrame
            DataFrame with columns:
            - gene_symbol: Gene name
            - <source_key>_gene_id: Gene ID for each source
            - (optional) chrom, start, end, strand, gene_type per source
        
        Examples
        --------
        >>> df = mapper.get_intersection_dataframe(['ensembl/GRCh37', 'mane/GRCh38'])
        >>> print(df.head())
        shape: (5, 3)
        ┌─────────────┬────────────────────────┬───────────────────┐
        │ gene_symbol ┆ ensembl/GRCh37_gene_id ┆ mane/GRCh38_gene_id│
        │ ---         ┆ ---                    ┆ ---                │
        │ str         ┆ str                    ┆ str                │
        ╞═════════════╪════════════════════════╪════════════════════╡
        │ BRCA1       ┆ ENSG00000012048        ┆ gene-BRCA1         │
        │ TP53        ┆ ENSG00000141510        ┆ gene-TP53          │
        └─────────────┴────────────────────────┴────────────────────┘
        """
        common_genes = self.get_common_genes(source_keys)
        
        data = {'gene_symbol': common_genes}
        
        for source_key in source_keys:
            gene_ids = []
            if include_all_info:
                chroms = []
                starts = []
                ends = []
                strands = []
                gene_types = []
            
            for gene_symbol in common_genes:
                gene_info = self.symbol_to_info[gene_symbol][source_key]
                gene_ids.append(gene_info.gene_id)
                
                if include_all_info:
                    chroms.append(gene_info.chrom)
                    starts.append(gene_info.start)
                    ends.append(gene_info.end)
                    strands.append(gene_info.strand)
                    gene_types.append(gene_info.gene_type)
            
            # Sanitize column names (replace '/' with '_')
            col_prefix = source_key.replace('/', '_')
            data[f'{col_prefix}_gene_id'] = gene_ids
            
            if include_all_info:
                data[f'{col_prefix}_chrom'] = chroms
                data[f'{col_prefix}_start'] = starts
                data[f'{col_prefix}_end'] = ends
                data[f'{col_prefix}_strand'] = strands
                data[f'{col_prefix}_gene_type'] = gene_types
        
        return pl.DataFrame(data)
    
    def get_summary(self) -> Dict[str, any]:
        """Get summary statistics about the mapper.
        
        Returns
        -------
        Dict[str, any]
            Summary with:
            - num_sources: Number of sources registered
            - num_unique_genes: Total unique gene symbols
            - sources: List of source keys
            - genes_per_source: Dict of gene counts per source
        """
        genes_per_source = {}
        for source_key, df in self.sources.items():
            genes_per_source[source_key] = df.height
        
        return {
            'num_sources': len(self.sources),
            'num_unique_genes': len(self.symbol_to_info),
            'sources': list(self.sources.keys()),
            'genes_per_source': genes_per_source
        }
    
    def print_summary(self):
        """Print a formatted summary of the mapper."""
        summary = self.get_summary()
        
        print("=" * 80)
        print("GENE MAPPER SUMMARY")
        print("=" * 80)
        print(f"Sources registered: {summary['num_sources']}")
        print(f"Unique gene symbols: {summary['num_unique_genes']:,}")
        print()
        print("Genes per source:")
        for source_key, count in summary['genes_per_source'].items():
            print(f"  {source_key:30s}: {count:,}")
        print("=" * 80)


# Global mapper instance
_global_mapper: Optional[GeneMapper] = None


def get_gene_mapper(
    auto_load: bool = True,
    sources: Optional[List[Tuple[str, str, str]]] = None
) -> GeneMapper:
    """Get or create the global gene mapper instance.
    
    Parameters
    ----------
    auto_load : bool, default=True
        If True and mapper is empty, automatically load common sources
    sources : Optional[List[Tuple[str, str, str]]]
        List of (source, build, file_path) tuples to load
        If None and auto_load=True, loads default sources
    
    Returns
    -------
    GeneMapper
        Global gene mapper instance
    
    Examples
    --------
    >>> mapper = get_gene_mapper()
    >>> common = mapper.get_common_genes(['ensembl/GRCh37', 'mane/GRCh38'])
    """
    global _global_mapper
    
    if _global_mapper is None:
        _global_mapper = GeneMapper()
        
        if auto_load:
            # Try to load default sources
            from meta_spliceai.system.genomic_resources import get_genomic_registry
            
            try:
                # Load Ensembl/GRCh37
                grch37_registry = get_genomic_registry('ensembl', 'GRCh37')
                grch37_gene_features = grch37_registry.data_dir / "gene_features.tsv"
                if grch37_gene_features.exists():
                    _global_mapper.add_source_from_file('ensembl', 'GRCh37', grch37_gene_features)
                
                # Load MANE/GRCh38
                grch38_registry = get_genomic_registry('mane', 'GRCh38')
                grch38_gene_features = grch38_registry.data_dir / "gene_features.tsv"
                if grch38_gene_features.exists():
                    _global_mapper.add_source_from_file('mane', 'GRCh38', grch38_gene_features)
            except Exception as e:
                # Silently fail if resources not available
                pass
        
        if sources:
            for source, build, file_path in sources:
                _global_mapper.add_source_from_file(source, build, file_path)
    
    return _global_mapper


def reset_gene_mapper():
    """Reset the global gene mapper instance."""
    global _global_mapper
    _global_mapper = None

