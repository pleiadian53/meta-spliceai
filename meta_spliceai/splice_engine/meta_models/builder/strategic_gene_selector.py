#!/usr/bin/env python3
"""
Strategic Gene Selection Utility for Meta-Model Training

This utility provides advanced gene selection strategies that go beyond the basic
policies in incremental_builder.py. It focuses on selecting genes based on 
characteristics that optimize meta-model performance.

USAGE PHILOSOPHY:
1. Use incremental_builder for basic diversity (random, error-based)
2. Use this utility for strategic additions based on gene characteristics
3. Combine both approaches for optimal training sets

SELECTION STRATEGIES:
- Length-stratified: Select genes by length categories
- Splice-density-based: Select genes with high splice site density  
- Gene-type-focused: Advanced filtering beyond basic --gene-types
- Performance-optimized: Select genes likely to benefit from meta-model

INTEGRATION:
- Outputs gene lists compatible with --gene-ids-file
- Uses data_resource_manager for systematic path resolution
- Provides detailed statistics for training set analysis
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, TypeVar, Callable
import logging

import polars as pl

# Add project root to path for imports
project_root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(project_root))

from meta_spliceai.splice_engine.meta_models.workflows.inference.data_resource_manager import create_inference_data_manager

logger = logging.getLogger(__name__)


T = TypeVar('T')


def parse_flexible_list(values: List[str], item_type: Callable[[str], T] = str) -> List[T]:
    """
    Parse a list that can be either space-separated or comma-separated.
    
    Args:
        values: List of string values from argparse
        item_type: Function to convert string to desired type (e.g., int, str)
    
    Returns:
        List of parsed values
        
    Examples:
        # Space-separated: --gene-types protein_coding lncRNA
        parse_flexible_list(['protein_coding', 'lncRNA']) → ['protein_coding', 'lncRNA']
        
        # Comma-separated: --gene-types protein_coding,lncRNA  
        parse_flexible_list(['protein_coding,lncRNA']) → ['protein_coding', 'lncRNA']
    """
    result = []
    for value in values:
        if ',' in value:
            # Comma-separated within this value
            result.extend([item_type(x.strip()) for x in value.split(',') if x.strip()])
        else:
            # Space-separated (single value)
            result.append(item_type(value))
    return result


class FlexibleListAction(argparse.Action):
    """Custom argparse action that supports both space and comma-separated values."""
    
    def __init__(self, option_strings, dest, item_type=str, **kwargs):
        self.item_type = item_type
        super().__init__(option_strings, dest, **kwargs)
    
    def __call__(self, parser, namespace, values, option_string=None):
        if values is None:
            setattr(namespace, self.dest, [])
        else:
            parsed_values = parse_flexible_list(values, self.item_type)
            setattr(namespace, self.dest, parsed_values)


class StrategicGeneSelector:
    """
    Advanced gene selection based on characteristics that optimize meta-model performance.
    
    Based on analysis showing meta-model performs best on:
    - Longer genes (>15kb)
    - Higher splice site density (>10 sites/kb)  
    - Protein-coding and lncRNA genes
    - Genes with complex splicing patterns
    """
    
    def __init__(self, project_root: Optional[Union[str, Path]] = None, verbose: bool = False):
        """
        Initialize the strategic gene selector.
        
        Parameters
        ----------
        project_root : str or Path, optional
            Project root directory. Auto-detected if None.
        verbose : bool
            Enable verbose logging
        """
        self.verbose = verbose
        if project_root is None:
            project_root = self._find_project_root()
        
        self.project_root = Path(project_root).resolve()
        
        # Use data resource manager for systematic path resolution
        self.data_manager = create_inference_data_manager(
            project_root=self.project_root, 
            auto_detect=True
        )
        
        # Load gene features and splice sites
        self.gene_features_df = self._load_gene_features()
        self.splice_sites_df = self._load_splice_sites()
        
        # Pre-compute splice site densities
        self.gene_characteristics_df = self._compute_gene_characteristics()
        
        if self.verbose:
            print(f"📋 Strategic Gene Selector initialized")
            print(f"   Project root: {self.project_root}")
            print(f"   Gene features: {len(self.gene_features_df):,} genes")
            print(f"   Splice sites: {len(self.splice_sites_df):,} sites")
    
    def _find_project_root(self) -> Path:
        """Auto-detect project root by looking for characteristic files."""
        current_dir = Path(__file__).resolve()
        
        # Look for characteristic project files
        key_files = [
            "data/ensembl/spliceai_analysis/gene_features.tsv",
            "meta_spliceai/__init__.py",
        ]
        
        # Start from current directory and go up
        for parent in [current_dir] + list(current_dir.parents):
            for key_file in key_files:
                if (parent / key_file).exists():
                    if self.verbose:
                        print(f"✅ Found project root: {parent}")
                    return parent
        
        # Fallback: assume current working directory
        cwd = Path.cwd()
        if self.verbose:
            print(f"⚠️  Using current working directory as project root: {cwd}")
        return cwd
    
    def _load_gene_features(self) -> pl.DataFrame:
        """Load gene features using data resource manager."""
        gene_features_path = self.data_manager.get_gene_features_path()
        
        if not gene_features_path or not gene_features_path.exists():
            raise FileNotFoundError(f"Gene features not found: {gene_features_path}")
        
        return pl.read_csv(
            gene_features_path,
            separator='\t',
            schema_overrides={'chrom': pl.Utf8}
        )
    
    def _load_splice_sites(self) -> pl.DataFrame:
        """Load splice sites using data resource manager."""
        splice_sites_path = self.data_manager.get_splice_sites_path()
        
        if not splice_sites_path or not splice_sites_path.exists():
            raise FileNotFoundError(f"Splice sites not found: {splice_sites_path}")
        
        return pl.read_csv(
            splice_sites_path,
            separator='\t',
            schema_overrides={'chrom': pl.Utf8}
        )
    
    def _compute_gene_characteristics(self) -> pl.DataFrame:
        """Compute comprehensive gene characteristics for selection."""
        if self.verbose:
            print("🧮 Computing gene characteristics...")
        
        # Calculate splice site counts per gene
        splice_counts = (
            self.splice_sites_df
            .group_by("gene_id")
            .agg([
                pl.len().alias("total_splice_sites"),
                pl.col("site_type").filter(pl.col("site_type") == "donor").len().alias("donor_sites"),
                pl.col("site_type").filter(pl.col("site_type") == "acceptor").len().alias("acceptor_sites")
            ])
        )
        
        # Merge gene features with splice site counts
        characteristics = self.gene_features_df.join(
            splice_counts,
            on="gene_id",
            how="left"
        )
        
        # Fill missing splice site counts with 0
        characteristics = characteristics.with_columns([
            pl.col("total_splice_sites").fill_null(0),
            pl.col("donor_sites").fill_null(0),
            pl.col("acceptor_sites").fill_null(0)
        ])
        
        # Calculate splice site density (sites per kb)
        characteristics = characteristics.with_columns([
            pl.when(pl.col("gene_length") > 0)
            .then(pl.col("total_splice_sites") / (pl.col("gene_length") / 1000))
            .otherwise(0.0)
            .alias("splice_density_per_kb")
        ])
        
        # Add length categories
        characteristics = characteristics.with_columns(
            pl.when(pl.col("gene_length") >= 50000).then(pl.lit("very_long"))
            .when(pl.col("gene_length") >= 20000).then(pl.lit("long"))
            .when(pl.col("gene_length") >= 10000).then(pl.lit("medium"))
            .when(pl.col("gene_length") >= 5000).then(pl.lit("short"))
            .otherwise(pl.lit("very_short"))
            .alias("length_category")
        )
        
        # Add splice density categories
        characteristics = characteristics.with_columns(
            pl.when(pl.col("splice_density_per_kb") >= 15.0).then(pl.lit("very_high"))
            .when(pl.col("splice_density_per_kb") >= 10.0).then(pl.lit("high"))
            .when(pl.col("splice_density_per_kb") >= 5.0).then(pl.lit("medium"))
            .when(pl.col("splice_density_per_kb") >= 1.0).then(pl.lit("low"))
            .otherwise(pl.lit("very_low"))
            .alias("density_category")
        )
        
        if self.verbose:
            print(f"✅ Computed characteristics for {len(characteristics):,} genes")
        
        return characteristics
    
    def select_by_length_strata(self, 
                               length_ranges: List[Tuple[int, int]], 
                               counts_per_stratum: List[int],
                               gene_types: Optional[List[str]] = None,
                               exclude_genes: Optional[List[str]] = None) -> Dict[str, List[str]]:
        """
        Select genes by length strata for balanced training.
        
        Parameters
        ----------
        length_ranges : List[Tuple[int, int]]
            List of (min_length, max_length) tuples defining strata
        counts_per_stratum : List[int]
            Number of genes to select from each stratum
        gene_types : List[str], optional
            Restrict to specific gene types (e.g., ['protein_coding'])
        exclude_genes : List[str], optional
            Gene IDs to exclude from selection
            
        Returns
        -------
        Dict[str, List[str]]
            Dictionary mapping stratum names to selected gene IDs
        """
        if len(length_ranges) != len(counts_per_stratum):
            raise ValueError("length_ranges and counts_per_stratum must have same length")
        
        results = {}
        exclude_set = set(exclude_genes or [])
        
        # Start with all genes
        candidates = self.gene_characteristics_df
        
        # Filter by gene types if specified
        if gene_types:
            candidates = candidates.filter(pl.col("gene_type").is_in(gene_types))
        
        # Filter out excluded genes
        if exclude_set:
            candidates = candidates.filter(~pl.col("gene_id").is_in(list(exclude_set)))
        
        for i, ((min_len, max_len), count) in enumerate(zip(length_ranges, counts_per_stratum)):
            stratum_name = f"length_{min_len//1000}k_{max_len//1000}k"
            
            # Filter by length range
            stratum_candidates = candidates.filter(
                (pl.col("gene_length") >= min_len) & 
                (pl.col("gene_length") < max_len)
            )
            
            if len(stratum_candidates) < count:
                if self.verbose:
                    print(f"⚠️  Stratum {stratum_name}: only {len(stratum_candidates)} genes available (requested {count})")
                count = len(stratum_candidates)
            
            # Sample genes from this stratum
            if count > 0:
                selected = stratum_candidates.sample(n=count, seed=42)
                selected_genes = selected["gene_id"].to_list()
                results[stratum_name] = selected_genes
                
                # Remove selected genes from future strata to avoid overlap
                candidates = candidates.filter(~pl.col("gene_id").is_in(selected_genes))
                
                if self.verbose:
                    print(f"✅ {stratum_name}: selected {len(selected_genes)} genes")
            else:
                results[stratum_name] = []
        
        return results
    
    def select_high_splice_density(self,
                                 count: int,
                                 min_density: float = 10.0,
                                 gene_types: Optional[List[str]] = None,
                                 min_length: int = 5000,
                                 exclude_genes: Optional[List[str]] = None) -> List[str]:
        """
        Select genes with high splice site density for meta-model optimization.
        
        Parameters
        ----------
        count : int
            Number of genes to select
        min_density : float
            Minimum splice sites per kb
        gene_types : List[str], optional
            Restrict to specific gene types
        min_length : int
            Minimum gene length to ensure reliable density calculation
        exclude_genes : List[str], optional
            Gene IDs to exclude from selection
            
        Returns
        -------
        List[str]
            Selected gene IDs
        """
        exclude_set = set(exclude_genes or [])
        
        # Start with high-density candidates
        candidates = self.gene_characteristics_df.filter(
            (pl.col("splice_density_per_kb") >= min_density) &
            (pl.col("gene_length") >= min_length)
        )
        
        # Filter by gene types if specified
        if gene_types:
            candidates = candidates.filter(pl.col("gene_type").is_in(gene_types))
        
        # Filter out excluded genes
        if exclude_set:
            candidates = candidates.filter(~pl.col("gene_id").is_in(list(exclude_set)))
        
        if len(candidates) < count:
            if self.verbose:
                print(f"⚠️  High splice density: only {len(candidates)} genes available (requested {count})")
            count = len(candidates)
        
        # Sort by splice density (descending) and take top genes
        selected = candidates.sort("splice_density_per_kb", descending=True).head(count)
        selected_genes = selected["gene_id"].to_list()
        
        if self.verbose:
            density_stats = selected["splice_density_per_kb"]
            print(f"✅ High splice density: selected {len(selected_genes)} genes")
            print(f"   Density range: {density_stats.min():.1f} - {density_stats.max():.1f} sites/kb")
            print(f"   Mean density: {density_stats.mean():.1f} sites/kb")
        
        return selected_genes
    
    def select_meta_optimized(self,
                            count: int,
                            gene_types: Optional[List[str]] = None,
                            exclude_genes: Optional[List[str]] = None) -> List[str]:
        """
        Select genes optimized for meta-model performance based on analysis.
        
        Based on analysis showing meta-model works best on:
        - Longer genes (>15kb)
        - Higher splice density (>8 sites/kb)
        - Protein-coding and lncRNA genes
        
        Parameters
        ----------
        count : int
            Number of genes to select
        gene_types : List[str], optional
            Restrict to specific gene types (default: ['protein_coding', 'lncRNA'])
        exclude_genes : List[str], optional
            Gene IDs to exclude from selection
            
        Returns
        -------
        List[str]
            Selected gene IDs optimized for meta-model performance
        """
        if gene_types is None:
            gene_types = ['protein_coding', 'lncRNA']
        
        exclude_set = set(exclude_genes or [])
        
        # Apply meta-model optimization criteria
        candidates = self.gene_characteristics_df.filter(
            (pl.col("gene_length") >= 15000) &  # Longer genes
            (pl.col("splice_density_per_kb") >= 8.0) &  # High splice density
            (pl.col("gene_type").is_in(gene_types))  # Optimal gene types
        )
        
        # Filter out excluded genes
        if exclude_set:
            candidates = candidates.filter(~pl.col("gene_id").is_in(list(exclude_set)))
        
        if len(candidates) < count:
            if self.verbose:
                print(f"⚠️  Meta-optimized: only {len(candidates)} genes available (requested {count})")
                print("   Relaxing criteria...")
            
            # Relax criteria if needed
            candidates = self.gene_characteristics_df.filter(
                (pl.col("gene_length") >= 10000) &  # Slightly shorter genes
                (pl.col("splice_density_per_kb") >= 5.0) &  # Lower splice density
                (pl.col("gene_type").is_in(gene_types))
            )
            
            if exclude_set:
                candidates = candidates.filter(~pl.col("gene_id").is_in(list(exclude_set)))
        
        count = min(count, len(candidates))
        
        # Score genes by meta-model optimization potential
        # Higher score = better for meta-model
        candidates = candidates.with_columns([
            (
                # Length factor (longer is better, up to a point)
                pl.when(pl.col("gene_length") >= 50000).then(1.0)
                .when(pl.col("gene_length") >= 20000).then(0.9)
                .when(pl.col("gene_length") >= 15000).then(0.8)
                .otherwise(0.6) +
                
                # Splice density factor (higher is better, up to a point)
                pl.when(pl.col("splice_density_per_kb") >= 15).then(1.0)
                .when(pl.col("splice_density_per_kb") >= 10).then(0.8)
                .when(pl.col("splice_density_per_kb") >= 8).then(0.6)
                .otherwise(0.4) +
                
                # Gene type factor
                pl.when(pl.col("gene_type") == "protein_coding").then(0.8)
                .when(pl.col("gene_type") == "lncRNA").then(0.9)
                .otherwise(0.5)
            ).alias("meta_optimization_score")
        ])
        
        # Select top-scoring genes
        selected = candidates.sort("meta_optimization_score", descending=True).head(count)
        selected_genes = selected["gene_id"].to_list()
        
        if self.verbose:
            length_stats = selected["gene_length"]
            density_stats = selected["splice_density_per_kb"]
            score_stats = selected["meta_optimization_score"]
            gene_type_counts = selected["gene_type"].value_counts()
            
            print(f"✅ Meta-optimized: selected {len(selected_genes)} genes")
            print(f"   Length range: {length_stats.min():,} - {length_stats.max():,} bp")
            print(f"   Mean length: {length_stats.mean():.0f} bp")
            print(f"   Density range: {density_stats.min():.1f} - {density_stats.max():.1f} sites/kb")
            print(f"   Mean density: {density_stats.mean():.1f} sites/kb")
            print(f"   Optimization score range: {score_stats.min():.2f} - {score_stats.max():.2f}")
            print(f"   Gene types:")
            for row in gene_type_counts.iter_rows():
                gene_type, count_val = row
                print(f"     {gene_type}: {count_val}")
        
        return selected_genes
    
    def get_gene_statistics(self, gene_ids: List[str]) -> Dict[str, float]:
        """Get summary statistics for a list of genes."""
        if not gene_ids:
            return {}
        
        genes_df = self.gene_characteristics_df.filter(pl.col("gene_id").is_in(gene_ids))
        
        if genes_df.is_empty():
            return {}
        
        return {
            "count": len(genes_df),
            "mean_length": genes_df["gene_length"].mean(),
            "median_length": genes_df["gene_length"].median(),
            "mean_splice_density": genes_df["splice_density_per_kb"].mean(),
            "median_splice_density": genes_df["splice_density_per_kb"].median(),
            "protein_coding_pct": (genes_df["gene_type"] == "protein_coding").sum() / len(genes_df) * 100,
        }
    
    def save_gene_list(self, gene_ids: List[str], output_path: Union[str, Path], 
                      include_stats: bool = True) -> Path:
        """Save gene list to file with optional statistics."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write gene IDs
        with open(output_path, 'w') as f:
            for gene_id in gene_ids:
                f.write(f"{gene_id}\n")
        
        # Write statistics file if requested
        if include_stats:
            stats_path = output_path.with_suffix('.stats.txt')
            stats = self.get_gene_statistics(gene_ids)
            
            with open(stats_path, 'w') as f:
                f.write(f"Gene List Statistics: {output_path.name}\n")
                f.write("=" * 50 + "\n")
                f.write(f"Total genes: {stats.get('count', 0):,}\n")
                f.write(f"Mean length: {stats.get('mean_length', 0):.0f} bp\n")
                f.write(f"Median length: {stats.get('median_length', 0):.0f} bp\n")
                f.write(f"Mean splice density: {stats.get('mean_splice_density', 0):.2f} sites/kb\n")
                f.write(f"Median splice density: {stats.get('median_splice_density', 0):.2f} sites/kb\n")
                f.write(f"Protein-coding genes: {stats.get('protein_coding_pct', 0):.1f}%\n")
            
            if self.verbose:
                print(f"📊 Statistics saved to: {stats_path}")
        
        if self.verbose:
            print(f"💾 Gene list saved to: {output_path}")
        
        return output_path


def main():
    """Command-line interface for strategic gene selection."""
    parser = argparse.ArgumentParser(
        description="Strategic gene selection for meta-model training optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:

# Length-stratified selection for balanced training
python strategic_gene_selector.py length-strata \
    --ranges 5000,15000 15000,30000 30000,100000 \
    --counts 1000,1500,500 \
    --gene-types protein_coding \
    --output-dir gene_lists

# High splice density genes for meta-model optimization  
python strategic_gene_selector.py high-density \
    --count 2000 \
    --min-density 10.0 \
    --gene-types protein_coding lncRNA \
    --output high_density_genes.txt

# Meta-optimized selection (best for meta-model performance)
python strategic_gene_selector.py meta-optimized \
    --count 3000 \
    --output meta_optimized_genes.txt

INTEGRATION WITH INCREMENTAL BUILDER:

# 1. Create base diverse training set
python -m meta_spliceai.splice_engine.meta_models.builder.incremental_builder \
    --n-genes 5000 --subset-policy random --gene-types protein_coding \
    --output-dir train_base_5000_pc

# 2. Add strategic selections
python strategic_gene_selector.py meta-optimized --count 1000 \
    --exclude-file train_base_5000_pc/master/gene_manifest.csv \
    --output strategic_additions.txt

# 3. Combine for final training set  
python -m meta_spliceai.splice_engine.meta_models.builder.incremental_builder \
    --gene-ids-file strategic_additions.txt --subset-policy custom \
    --output-dir train_strategic_additions
        """
    )
    
    # Subcommands for different selection strategies
    subparsers = parser.add_subparsers(dest='strategy', help='Selection strategy')
    
    # Length-stratified selection
    length_parser = subparsers.add_parser('length-strata', help='Length-stratified selection')
    length_parser.add_argument('--ranges', nargs='+', required=True,
                              help='Length ranges as min,max pairs (e.g., 5000,15000 15000,30000)')
    length_parser.add_argument('--counts', required=True,
                              help='Comma-separated counts for each stratum (e.g., 1000,1500,500)')
    length_parser.add_argument('--gene-types', nargs='*', 
                              action=FlexibleListAction, item_type=str,
                              help='Gene types to include (space or comma-separated, e.g., protein_coding lncRNA OR protein_coding,lncRNA)')
    length_parser.add_argument('--output-dir', default='length_strata_genes',
                              help='Output directory for gene lists')
    
    # High splice density selection
    density_parser = subparsers.add_parser('high-density', help='High splice density selection')
    density_parser.add_argument('--count', type=int, required=True,
                               help='Number of genes to select')
    density_parser.add_argument('--min-density', type=float, default=10.0,
                               help='Minimum splice sites per kb')
    density_parser.add_argument('--gene-types', nargs='*',
                               action=FlexibleListAction, item_type=str,
                               help='Gene types to include (space or comma-separated)')
    density_parser.add_argument('--output', required=True,
                               help='Output file path')
    
    # Meta-optimized selection
    meta_parser = subparsers.add_parser('meta-optimized', help='Meta-model optimized selection')
    meta_parser.add_argument('--count', type=int, required=True,
                            help='Number of genes to select')
    meta_parser.add_argument('--gene-types', nargs='*',
                            action=FlexibleListAction, item_type=str,
                            help='Gene types to include (space or comma-separated, default: protein_coding lncRNA)')
    meta_parser.add_argument('--output', required=True,
                            help='Output file path')
    
    # Common arguments
    for subparser in [length_parser, density_parser, meta_parser]:
        subparser.add_argument('--exclude-file',
                              help='CSV file with gene_id column to exclude from selection')
        subparser.add_argument('--project-root',
                              help='Project root directory (auto-detected if not provided)')
        subparser.add_argument('--verbose', '-v', action='store_true',
                              help='Enable verbose output')
    
    args = parser.parse_args()
    
    if not args.strategy:
        parser.print_help()
        return
    
    # Initialize selector
    selector = StrategicGeneSelector(
        project_root=args.project_root,
        verbose=args.verbose
    )
    
    # Load genes to exclude if specified
    exclude_genes = []
    if args.exclude_file:
        try:
            exclude_df = pl.read_csv(args.exclude_file)
            if 'gene_id' in exclude_df.columns:
                exclude_genes = exclude_df['gene_id'].to_list()
                if args.verbose:
                    print(f"📋 Loaded {len(exclude_genes):,} genes to exclude from {args.exclude_file}")
        except Exception as e:
            print(f"⚠️ Warning: Could not load exclude file: {e}")
    
    # Execute selected strategy
    if args.strategy == 'length-strata':
        # Parse length ranges
        ranges = []
        for range_str in args.ranges:
            min_len, max_len = map(int, range_str.split(','))
            ranges.append((min_len, max_len))
        
        # Parse counts
        counts = list(map(int, args.counts.split(',')))
        
        # Select genes
        results = selector.select_by_length_strata(
            length_ranges=ranges,
            counts_per_stratum=counts,
            gene_types=args.gene_types,
            exclude_genes=exclude_genes
        )
        
        # Save results
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        all_genes = []
        for stratum_name, genes in results.items():
            if genes:
                output_path = output_dir / f"{stratum_name}.txt"
                selector.save_gene_list(genes, output_path)
                all_genes.extend(genes)
        
        # Save combined list
        if all_genes:
            combined_path = output_dir / "all_length_strata.txt"
            selector.save_gene_list(all_genes, combined_path)
            print(f"✅ Combined {len(all_genes):,} genes saved to {combined_path}")
    
    elif args.strategy == 'high-density':
        genes = selector.select_high_splice_density(
            count=args.count,
            min_density=args.min_density,
            gene_types=args.gene_types,
            exclude_genes=exclude_genes
        )
        selector.save_gene_list(genes, args.output)
    
    elif args.strategy == 'meta-optimized':
        genes = selector.select_meta_optimized(
            count=args.count,
            gene_types=args.gene_types,
            exclude_genes=exclude_genes
        )
        selector.save_gene_list(genes, args.output)
    
    print("🎯 Strategic gene selection completed!")


if __name__ == "__main__":
    main()
