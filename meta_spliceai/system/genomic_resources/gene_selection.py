"""Gene Selection and Validation for Cross-Build Comparisons

This module provides utilities for selecting and validating genes across different
genomic builds using the Enhanced Gene Mapper.

Key Features:
- Sample genes by category (protein-coding, lncRNA, no splice sites)
- Map gene symbols to source-specific IDs (Ensembl, MANE, etc.)
- Validate gene availability across builds
- Handle intersection-based sampling for fair comparisons

Examples:
    >>> from meta_spliceai.system.genomic_resources import GeneSelector
    >>> 
    >>> # Initialize selector
    >>> selector = GeneSelector()
    >>> 
    >>> # Sample genes for comparison
    >>> result = selector.sample_genes_for_comparison(
    ...     source1='ensembl/GRCh37',
    ...     source2='mane/GRCh38',
    ...     n_protein_coding=10,
    ...     n_lncrna=5,
    ...     n_no_splice_sites=5
    ... )
    >>> 
    >>> # Get source-specific IDs
    >>> spliceai_ids = result['source1_gene_ids']
    >>> openspliceai_ids = result['source2_gene_ids']
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field

import polars as pl

from .gene_mapper_enhanced import EnhancedGeneMapper, GeneMapping
from .external_id_mapper import get_or_create_mane_ensembl_mapping
from .registry import Registry
from .build_naming import get_standardized_build_name


@dataclass
class GeneSamplingConfig:
    """Configuration for gene sampling.
    
    Attributes
    ----------
    n_protein_coding : int
        Number of protein-coding genes to sample
    n_lncrna : int
        Number of lncRNA genes to sample
    n_no_splice_sites : int
        Number of genes without splice sites to sample
    seed : Optional[int]
        Random seed for reproducibility (None for random)
    min_confidence : float
        Minimum confidence for gene mappings (0.0-1.0)
    protein_coding_filters : Dict
        Filters for protein-coding genes
    lncrna_filters : Dict
        Filters for lncRNA genes
    no_splice_sites_filters : Dict
        Filters for genes without splice sites
    """
    n_protein_coding: int = 10
    n_lncrna: int = 5
    n_no_splice_sites: int = 5
    seed: Optional[int] = None
    min_confidence: float = 0.9
    
    # Default filters
    protein_coding_filters: Dict = field(default_factory=lambda: {
        'min_splice_sites': 4,
        'min_length': 5_000,
        'max_length': 500_000
    })
    
    lncrna_filters: Dict = field(default_factory=lambda: {
        'gene_types': ['lncRNA', 'lincRNA', 'antisense', 'processed_transcript'],
        'min_splice_sites': 2,
        'min_length': 1_000,
        'max_length': 200_000
    })
    
    no_splice_sites_filters: Dict = field(default_factory=lambda: {
        'max_splice_sites': 0,
        'min_length': 500,
        'max_length': 50_000
    })


@dataclass
class GeneSamplingResult:
    """Result of gene sampling operation.
    
    Attributes
    ----------
    gene_symbols : List[str]
        Sampled gene symbols
    source1_gene_ids : List[Optional[str]]
        Gene IDs in source 1 (e.g., Ensembl IDs)
    source2_gene_ids : List[Optional[str]]
        Gene IDs in source 2 (e.g., MANE IDs)
    mappings : List[GeneMapping]
        Gene mappings with confidence scores
    sampled_by_category : Dict[str, List[str]]
        Genes sampled by category
    total_available : int
        Total genes available for sampling
    total_sampled : int
        Total genes sampled
    mapping_success_rate : float
        Fraction of genes successfully mapped
    """
    gene_symbols: List[str]
    source1_gene_ids: List[Optional[str]]
    source2_gene_ids: List[Optional[str]]
    mappings: List[GeneMapping]
    sampled_by_category: Dict[str, List[str]]
    total_available: int
    total_sampled: int
    mapping_success_rate: float


class GeneSelector:
    """Gene selection and validation for cross-build comparisons.
    
    This class provides utilities for:
    - Loading gene annotations from multiple sources
    - Finding high-confidence gene mappings
    - Sampling genes by category
    - Mapping gene symbols to source-specific IDs
    
    Examples
    --------
    >>> selector = GeneSelector()
    >>> 
    >>> # Sample genes for SpliceAI vs OpenSpliceAI comparison
    >>> result = selector.sample_genes_for_comparison(
    ...     source1='ensembl/GRCh37',
    ...     source2='mane/GRCh38',
    ...     config=GeneSamplingConfig(n_protein_coding=10, n_lncrna=5)
    ... )
    >>> 
    >>> print(f"Sampled {result.total_sampled} genes")
    >>> print(f"Mapping success: {result.mapping_success_rate:.1%}")
    """
    
    def __init__(self, data_dir: Optional[Path] = None):
        """Initialize gene selector.
        
        Parameters
        ----------
        data_dir : Optional[Path]
            Data directory (defaults to 'data')
        """
        self.data_dir = data_dir or Path('data')
        self.mapper = None
        self.sources_loaded = set()
    
    def load_sources(
        self,
        source1: str,
        source1_build: str,
        source2: str,
        source2_build: str,
        use_external_mapping: bool = True,
        verbosity: int = 1
    ) -> None:
        """Load gene annotation sources.
        
        Parameters
        ----------
        source1 : str
            First source name (e.g., 'ensembl')
        source1_build : str
            First source build (e.g., 'GRCh37')
        source2 : str
            Second source name (e.g., 'mane')
        source2_build : str
            Second source build (e.g., 'GRCh38')
        use_external_mapping : bool, default=True
            Use external ID mapping (e.g., MANE→Ensembl)
        verbosity : int, default=1
            Output verbosity (0-2)
        """
        if verbosity >= 1:
            print("🧬 Initializing Enhanced Gene Mapper...")
        
        # Load external mapping if requested
        external_mapping = None
        if use_external_mapping and source2.lower() == 'mane':
            if verbosity >= 1:
                print("Loading MANE→Ensembl ID mapping...")
            external_mapping = get_or_create_mane_ensembl_mapping(self.data_dir)
            if verbosity >= 1:
                print()
        
        # Initialize mapper
        self.mapper = EnhancedGeneMapper()
        
        # Load source 1
        if verbosity >= 1:
            print(f"Loading {source1}/{source1_build} genes...")
        
        registry1 = Registry(build=source1_build, release='87' if source1_build == 'GRCh37' else '1.3')
        gene_features_path1 = registry1.data_dir / "gene_features.tsv"
        
        if not gene_features_path1.exists():
            raise FileNotFoundError(f"Gene features not found: {gene_features_path1}")
        
        self.mapper.add_source_from_file(source1, source1_build, str(gene_features_path1))
        self.sources_loaded.add(f"{source1}/{source1_build}")
        
        if verbosity >= 1:
            print(f"✅ Loaded {source1}/{source1_build} genes")
        
        # Load source 2
        if verbosity >= 1:
            print(f"Loading {source2}/{source2_build} genes...")
        
        # Use standardized build naming convention
        build2 = get_standardized_build_name(source2, source2_build)
        
        registry2 = Registry(build=build2,
                           release='1.3' if source2_build.startswith('GRCh38') else '87')
        gene_features_path2 = registry2.data_dir / "gene_features.tsv"
        
        if not gene_features_path2.exists():
            raise FileNotFoundError(f"Gene features not found: {gene_features_path2}")
        
        self.mapper.add_source_from_file(
            source2,
            source2_build,
            str(gene_features_path2),
            external_id_mapping=external_mapping
        )
        self.sources_loaded.add(f"{source2}/{source2_build}")
        
        if verbosity >= 1:
            suffix = " (with Ensembl ID mapping)" if external_mapping else ""
            print(f"✅ Loaded {source2}/{source2_build} genes{suffix}")
            print()
    
    def get_high_confidence_mappings(
        self,
        source1_key: str,
        source2_key: str,
        min_confidence: float = 0.9,
        verbosity: int = 1
    ) -> List[GeneMapping]:
        """Get high-confidence gene mappings between sources.
        
        Parameters
        ----------
        source1_key : str
            Source 1 key (e.g., 'ensembl/GRCh37')
        source2_key : str
            Source 2 key (e.g., 'mane/GRCh38')
        min_confidence : float, default=0.9
            Minimum confidence threshold
        verbosity : int, default=1
            Output verbosity
        
        Returns
        -------
        List[GeneMapping]
            High-confidence gene mappings
        """
        if self.mapper is None:
            raise RuntimeError("Sources not loaded. Call load_sources() first.")
        
        if verbosity >= 1:
            print(f"Finding high-confidence gene mappings (confidence ≥ {min_confidence})...")
        
        try:
            # Print summary
            if verbosity >= 2:
                self.mapper.print_summary(source1_key, source2_key)
                print()
            
            # Get high-confidence mappings
            mappings = self.mapper.get_high_confidence_mappings(
                source1_key,
                source2_key,
                min_confidence=min_confidence
            )
            
            if verbosity >= 1:
                print(f"✅ Found {len(mappings):,} high-confidence gene mappings")
                
                # Show strategy breakdown
                from collections import Counter
                strategy_counts = Counter(m.strategy.value for m in mappings)
                if verbosity >= 2:
                    print("\nBy Strategy:")
                    for strategy, count in strategy_counts.most_common():
                        pct = count / len(mappings) * 100
                        print(f"  {strategy:20s}: {count:,} ({pct:.1f}%)")
                print()
            
            return mappings
        
        except Exception as e:
            print(f"❌ Error finding gene mappings: {e}")
            raise
    
    def sample_genes_by_category(
        self,
        mappings: List[GeneMapping],
        source1_key: str,
        config: GeneSamplingConfig,
        verbosity: int = 1
    ) -> GeneSamplingResult:
        """Sample genes by category from high-confidence mappings.
        
        Parameters
        ----------
        mappings : List[GeneMapping]
            High-confidence gene mappings
        source1_key : str
            Source 1 key for loading gene features
        config : GeneSamplingConfig
            Sampling configuration
        verbosity : int, default=1
            Output verbosity
        
        Returns
        -------
        GeneSamplingResult
            Sampling results with gene IDs and metadata
        """
        if verbosity >= 1:
            print("=" * 80)
            print("SAMPLING GENES BY CATEGORY")
            print("=" * 80)
            print()
        
        # Get common gene symbols
        common_gene_symbols = [m.gene_symbol for m in mappings]
        gene_mappings_dict = {m.gene_symbol: m for m in mappings}
        
        # Load gene features for source 1 (for sampling)
        source1_parts = source1_key.split('/')
        registry = Registry(build=source1_parts[1], release='87' if source1_parts[1] == 'GRCh37' else '1.3')
        gene_features_path = registry.data_dir / "gene_features.tsv"
        
        gene_features = pl.read_csv(
            str(gene_features_path),
            separator='\t',
            schema_overrides={'chrom': pl.Utf8, 'seqname': pl.Utf8}
        )
        
        # Filter to intersection
        intersection_df = gene_features.filter(
            pl.col('gene_name').is_in(common_gene_symbols)
        )
        
        # Load splice site counts
        splice_sites_path = registry.data_dir / "splice_sites_enhanced.tsv"
        if splice_sites_path.exists():
            splice_sites = pl.read_csv(
                str(splice_sites_path),
                separator='\t',
                schema_overrides={'chrom': pl.Utf8}
            )
            ss_counts = splice_sites.group_by('gene_id').agg(pl.count().alias('n_splice_sites'))
            intersection_df = intersection_df.join(ss_counts, on='gene_id', how='left')
            intersection_df = intersection_df.with_columns(pl.col('n_splice_sites').fill_null(0))
        else:
            intersection_df = intersection_df.with_columns(pl.lit(0).alias('n_splice_sites'))
        
        # Sample by category
        sampled_genes = {}
        
        # 1. Protein-coding genes
        if verbosity >= 1:
            print("PROTEIN-CODING GENES:")
        
        pc_filters = config.protein_coding_filters
        protein_coding = intersection_df.filter(
            (pl.col('gene_type') == 'protein_coding') &
            (pl.col('n_splice_sites') >= pc_filters['min_splice_sites']) &
            (pl.col('gene_length') >= pc_filters['min_length']) &
            (pl.col('gene_length') <= pc_filters['max_length'])
        )
        
        if verbosity >= 1:
            print(f"  Available: {protein_coding.height:,}")
        
        n_pc = min(config.n_protein_coding, protein_coding.height)
        if n_pc > 0:
            if config.seed is not None:
                sampled_pc = protein_coding.sample(n=n_pc, seed=config.seed)
            else:
                sampled_pc = protein_coding.sample(n=n_pc)
            sampled_genes['protein_coding'] = sampled_pc['gene_name'].to_list()
            
            if verbosity >= 1:
                print(f"  ✅ Sampled: {len(sampled_genes['protein_coding'])}")
                print(f"     Examples: {', '.join(sampled_genes['protein_coding'][:5])}")
        else:
            sampled_genes['protein_coding'] = []
            if verbosity >= 1:
                print(f"  ⚠️  No genes available")
        
        if verbosity >= 1:
            print()
        
        # 2. lncRNA genes
        if verbosity >= 1:
            print("LNCRNA GENES:")
        
        lnc_filters = config.lncrna_filters
        lncrna = intersection_df.filter(
            pl.col('gene_type').is_in(lnc_filters['gene_types']) &
            (pl.col('n_splice_sites') >= lnc_filters['min_splice_sites']) &
            (pl.col('gene_length') >= lnc_filters['min_length']) &
            (pl.col('gene_length') <= lnc_filters['max_length'])
        )
        
        if verbosity >= 1:
            print(f"  Available: {lncrna.height:,}")
        
        n_lnc = min(config.n_lncrna, lncrna.height)
        if n_lnc > 0:
            if config.seed is not None:
                sampled_lnc = lncrna.sample(n=n_lnc, seed=config.seed)
            else:
                sampled_lnc = lncrna.sample(n=n_lnc)
            sampled_genes['lncrna'] = sampled_lnc['gene_name'].to_list()
            
            if verbosity >= 1:
                print(f"  ✅ Sampled: {len(sampled_genes['lncrna'])}")
                print(f"     Examples: {', '.join(sampled_genes['lncrna'][:5])}")
        else:
            sampled_genes['lncrna'] = []
            if verbosity >= 1:
                print(f"  ⚠️  No genes available")
        
        if verbosity >= 1:
            print()
        
        # 3. Genes without splice sites
        if verbosity >= 1:
            print("GENES WITHOUT SPLICE SITES:")
        
        nss_filters = config.no_splice_sites_filters
        no_ss = intersection_df.filter(
            (pl.col('n_splice_sites') <= nss_filters['max_splice_sites']) &
            (pl.col('gene_length') >= nss_filters['min_length']) &
            (pl.col('gene_length') <= nss_filters['max_length'])
        )
        
        if verbosity >= 1:
            print(f"  Available: {no_ss.height:,}")
        
        n_nss = min(config.n_no_splice_sites, no_ss.height)
        if n_nss > 0:
            if config.seed is not None:
                sampled_nss = no_ss.sample(n=n_nss, seed=config.seed)
            else:
                sampled_nss = no_ss.sample(n=n_nss)
            sampled_genes['no_splice_sites'] = sampled_nss['gene_name'].to_list()
            
            if verbosity >= 1:
                print(f"  ✅ Sampled: {len(sampled_genes['no_splice_sites'])}")
                print(f"     Examples: {', '.join(sampled_genes['no_splice_sites'][:5])}")
        else:
            sampled_genes['no_splice_sites'] = []
            if verbosity >= 1:
                print(f"  ⚠️  No genes available")
        
        if verbosity >= 1:
            print()
        
        # Combine all sampled genes
        all_sampled_genes = (
            sampled_genes['protein_coding'] +
            sampled_genes['lncrna'] +
            sampled_genes['no_splice_sites']
        )
        
        if verbosity >= 1:
            print(f"TOTAL SAMPLED: {len(all_sampled_genes)} genes")
            print(f"  • Protein-coding: {len(sampled_genes['protein_coding'])}")
            print(f"  • lncRNA: {len(sampled_genes['lncrna'])}")
            print(f"  • No splice sites: {len(sampled_genes['no_splice_sites'])}")
            print()
        
        # Map to source-specific IDs
        source1_gene_ids = []
        source2_gene_ids = []
        gene_mappings = []
        
        for gene_symbol in all_sampled_genes:
            if gene_symbol in gene_mappings_dict:
                mapping = gene_mappings_dict[gene_symbol]
                source1_gene_ids.append(mapping.source1_gene_id)
                source2_gene_ids.append(mapping.source2_gene_id)
                gene_mappings.append(mapping)
            else:
                source1_gene_ids.append(None)
                source2_gene_ids.append(None)
                gene_mappings.append(None)
        
        # Calculate mapping success rate
        mapped_count = sum(1 for g in source1_gene_ids if g is not None)
        mapping_success_rate = mapped_count / len(all_sampled_genes) if all_sampled_genes else 0.0
        
        return GeneSamplingResult(
            gene_symbols=all_sampled_genes,
            source1_gene_ids=source1_gene_ids,
            source2_gene_ids=source2_gene_ids,
            mappings=[m for m in gene_mappings if m is not None],
            sampled_by_category=sampled_genes,
            total_available=len(common_gene_symbols),
            total_sampled=len(all_sampled_genes),
            mapping_success_rate=mapping_success_rate
        )
    
    def sample_genes_for_comparison(
        self,
        source1: str,
        source1_build: str,
        source2: str,
        source2_build: str,
        config: Optional[GeneSamplingConfig] = None,
        use_external_mapping: bool = True,
        verbosity: int = 1
    ) -> GeneSamplingResult:
        """Complete workflow: load sources, find mappings, and sample genes.
        
        Parameters
        ----------
        source1 : str
            First source name (e.g., 'ensembl')
        source1_build : str
            First source build (e.g., 'GRCh37')
        source2 : str
            Second source name (e.g., 'mane')
        source2_build : str
            Second source build (e.g., 'GRCh38')
        config : Optional[GeneSamplingConfig]
            Sampling configuration (uses defaults if None)
        use_external_mapping : bool, default=True
            Use external ID mapping
        verbosity : int, default=1
            Output verbosity
        
        Returns
        -------
        GeneSamplingResult
            Complete sampling results
        """
        if config is None:
            config = GeneSamplingConfig()
        
        # Load sources
        self.load_sources(
            source1, source1_build,
            source2, source2_build,
            use_external_mapping=use_external_mapping,
            verbosity=verbosity
        )
        
        # Get high-confidence mappings
        source1_key = f"{source1}/{source1_build}"
        source2_key = f"{source2}/{source2_build}"
        
        mappings = self.get_high_confidence_mappings(
            source1_key,
            source2_key,
            min_confidence=config.min_confidence,
            verbosity=verbosity
        )
        
        # Sample genes by category
        result = self.sample_genes_by_category(
            mappings,
            source1_key,
            config,
            verbosity=verbosity
        )
        
        return result

