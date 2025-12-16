#!/usr/bin/env python3
"""
Paralog gene analysis utility.

This script analyzes gene features data to identify potential paralog genes
based on gene names, chromosomal clustering, and gene family patterns.
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd


def load_gene_features(file_path: str) -> pd.DataFrame:
    """Load gene features from TSV file."""
    try:
        df = pd.read_csv(file_path, sep='\t')
        print(f"✅ Loaded {len(df):,} genes from {file_path}")
        return df
    except Exception as e:
        print(f"❌ Failed to load gene features: {e}")
        sys.exit(1)


def identify_gene_families_by_name(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Identify potential gene families based on gene name patterns.
    
    Args:
        df: DataFrame with gene information
        
    Returns:
        Dictionary mapping family names to lists of gene names
    """
    print("\n🔍 ANALYZING GENE NAME PATTERNS")
    print("-" * 40)
    
    # Filter out genes without names
    named_genes = df[df['gene_name'].notna() & (df['gene_name'] != '')].copy()
    print(f"Genes with names: {len(named_genes):,} / {len(df):,}")
    
    gene_families = defaultdict(list)
    
    # Pattern 1: Genes with numeric suffixes (e.g., HOX1A, HOX1B, HOX1C)
    numeric_pattern = re.compile(r'^([A-Z]+[A-Z0-9]*?)([0-9]+[A-Z]?[0-9]*)$')
    
    # Pattern 2: Genes with letter suffixes (e.g., HOXA1, HOXA2, HOXA3)
    letter_pattern = re.compile(r'^([A-Z]+)([A-Z][0-9]+)$')
    
    # Pattern 3: Genes with dash/underscore patterns (e.g., GENE-1, GENE_A)
    dash_pattern = re.compile(r'^([A-Z0-9]+)[-_]([A-Z0-9]+)$')
    
    # Pattern 4: Pseudogenes (processed pseudogenes are often paralogs)
    pseudo_pattern = re.compile(r'^([A-Z0-9]+)P([0-9]+)$')
    
    for _, row in named_genes.iterrows():
        gene_name = row['gene_name'].upper()
        gene_id = row['gene_id']
        gene_type = row['gene_type']
        
        # Check each pattern
        for pattern_name, pattern in [
            ('numeric_suffix', numeric_pattern),
            ('letter_suffix', letter_pattern),
            ('dash_underscore', dash_pattern),
            ('pseudogene', pseudo_pattern)
        ]:
            match = pattern.match(gene_name)
            if match:
                family_root = match.group(1)
                family_key = f"{pattern_name}:{family_root}"
                gene_families[family_key].append({
                    'gene_id': gene_id,
                    'gene_name': row['gene_name'],
                    'gene_type': gene_type,
                    'chrom': row['chrom'],
                    'start': row['start'],
                    'end': row['end']
                })
                break
    
    # Filter families with multiple members (potential paralogs)
    paralog_families = {k: v for k, v in gene_families.items() if len(v) > 1}
    
    print(f"Potential gene families identified: {len(paralog_families)}")
    
    # Show top families by size
    sorted_families = sorted(paralog_families.items(), key=lambda x: len(x[1]), reverse=True)
    print(f"\nTop 10 largest gene families:")
    for family_name, members in sorted_families[:10]:
        pattern_type, root = family_name.split(':', 1)
        print(f"  {root} ({pattern_type}): {len(members)} members")
        
        # Show first few members
        for member in members[:3]:
            print(f"    - {member['gene_name']} ({member['gene_type']}, chr{member['chrom']})")
        if len(members) > 3:
            print(f"    ... and {len(members) - 3} more")
    
    return paralog_families


def identify_chromosomal_clusters(df: pd.DataFrame, max_distance: int = 1000000) -> List[Dict]:
    """
    Identify potential paralog clusters based on chromosomal proximity.
    
    Args:
        df: DataFrame with gene information
        max_distance: Maximum distance between genes to consider them clustered
        
    Returns:
        List of cluster dictionaries
    """
    print(f"\n🧬 ANALYZING CHROMOSOMAL CLUSTERS")
    print("-" * 40)
    print(f"Max distance for clustering: {max_distance:,} bp")
    
    # Filter protein-coding genes with names for clustering analysis
    coding_genes = df[
        (df['gene_type'] == 'protein_coding') & 
        (df['gene_name'].notna()) & 
        (df['gene_name'] != '')
    ].copy()
    
    print(f"Analyzing {len(coding_genes):,} protein-coding genes with names")
    
    clusters = []
    
    # Group by chromosome
    for chrom in sorted(coding_genes['chrom'].unique()):
        chrom_genes = coding_genes[coding_genes['chrom'] == chrom].sort_values('start')
        
        if len(chrom_genes) < 2:
            continue
        
        # Find clusters of genes with similar names
        for i, (_, gene1) in enumerate(chrom_genes.iterrows()):
            cluster_genes = [gene1]
            
            # Look for nearby genes with similar names
            for j, (_, gene2) in enumerate(chrom_genes.iterrows()):
                if i == j:
                    continue
                
                # Check if genes are close enough
                distance = abs(gene1['start'] - gene2['start'])
                if distance > max_distance:
                    continue
                
                # Check if gene names are similar (share common prefix)
                name1 = gene1['gene_name'].upper()
                name2 = gene2['gene_name'].upper()
                
                # Simple similarity check - share at least 3 characters at start
                if len(name1) >= 3 and len(name2) >= 3:
                    if name1[:3] == name2[:3] or name1[:4] == name2[:4]:
                        cluster_genes.append(gene2)
            
            # If we found a cluster, add it
            if len(cluster_genes) > 1:
                # Check if this cluster is already recorded
                cluster_names = set(g['gene_name'] for g in cluster_genes)
                already_recorded = any(
                    cluster_names == set(c['gene_name'] for c in existing_cluster['genes'])
                    for existing_cluster in clusters
                )
                
                if not already_recorded:
                    clusters.append({
                        'chromosome': chrom,
                        'genes': cluster_genes,
                        'span_start': min(g['start'] for g in cluster_genes),
                        'span_end': max(g['end'] for g in cluster_genes),
                        'span_length': max(g['end'] for g in cluster_genes) - min(g['start'] for g in cluster_genes)
                    })
    
    print(f"Chromosomal clusters identified: {len(clusters)}")
    
    # Show top clusters
    sorted_clusters = sorted(clusters, key=lambda x: len(x['genes']), reverse=True)
    print(f"\nTop 10 largest chromosomal clusters:")
    for i, cluster in enumerate(sorted_clusters[:10]):
        genes = cluster['genes']
        print(f"  Cluster {i+1}: Chr{cluster['chromosome']}, {len(genes)} genes")
        print(f"    Span: {cluster['span_length']:,} bp")
        print(f"    Genes: {', '.join(g['gene_name'] for g in genes[:5])}")
        if len(genes) > 5:
            print(f"           ... and {len(genes) - 5} more")
    
    return clusters


def analyze_pseudogenes(df: pd.DataFrame) -> Dict[str, List[Dict]]:
    """
    Analyze pseudogenes as potential paralogs.
    
    Args:
        df: DataFrame with gene information
        
    Returns:
        Dictionary mapping pseudogene types to lists of genes
    """
    print(f"\n🧪 ANALYZING PSEUDOGENES")
    print("-" * 30)
    
    # Get all pseudogene types
    pseudogene_types = df[df['gene_type'].str.contains('pseudogene', na=False)]['gene_type'].unique()
    print(f"Pseudogene types found: {list(pseudogene_types)}")
    
    pseudogene_analysis = {}
    
    for pg_type in pseudogene_types:
        pseudogenes = df[df['gene_type'] == pg_type].copy()
        
        # Try to match pseudogenes to their parent genes
        parent_matches = []
        
        for _, pg in pseudogenes.iterrows():
            if pd.isna(pg['gene_name']) or pg['gene_name'] == '':
                continue
                
            pg_name = pg['gene_name']
            
            # Look for potential parent gene
            # Remove common pseudogene suffixes
            potential_parent = re.sub(r'P[0-9]+$', '', pg_name)  # Remove P1, P2, etc.
            potential_parent = re.sub(r'PS[0-9]*$', '', potential_parent)  # Remove PS, PS1, etc.
            
            if potential_parent != pg_name:
                # Look for the parent gene
                parent_candidates = df[
                    (df['gene_name'] == potential_parent) & 
                    (df['gene_type'] == 'protein_coding')
                ]
                
                if len(parent_candidates) > 0:
                    parent = parent_candidates.iloc[0]
                    parent_matches.append({
                        'pseudogene': {
                            'gene_id': pg['gene_id'],
                            'gene_name': pg['gene_name'],
                            'chrom': pg['chrom'],
                            'start': pg['start'],
                            'end': pg['end']
                        },
                        'parent': {
                            'gene_id': parent['gene_id'],
                            'gene_name': parent['gene_name'],
                            'chrom': parent['chrom'],
                            'start': parent['start'],
                            'end': parent['end']
                        }
                    })
        
        pseudogene_analysis[pg_type] = {
            'total_count': len(pseudogenes),
            'with_names': len(pseudogenes[pseudogenes['gene_name'].notna() & (pseudogenes['gene_name'] != '')]),
            'parent_matches': parent_matches
        }
        
        print(f"\n{pg_type}:")
        print(f"  Total: {pseudogene_analysis[pg_type]['total_count']:,}")
        print(f"  With names: {pseudogene_analysis[pg_type]['with_names']:,}")
        print(f"  Parent matches: {len(parent_matches):,}")
        
        # Show some examples
        for match in parent_matches[:3]:
            pg_info = match['pseudogene']
            parent_info = match['parent']
            print(f"    {pg_info['gene_name']} (chr{pg_info['chrom']}) -> {parent_info['gene_name']} (chr{parent_info['chrom']})")
    
    return pseudogene_analysis


def generate_paralog_report(gene_families: Dict, clusters: List[Dict], pseudogenes: Dict, output_file: Optional[str] = None) -> Dict:
    """Generate comprehensive paralog analysis report."""
    
    # Count total potential paralogs
    total_family_genes = sum(len(members) for members in gene_families.values())
    total_cluster_genes = sum(len(cluster['genes']) for cluster in clusters)
    total_pseudogene_matches = sum(len(data['parent_matches']) for data in pseudogenes.values())
    
    report = {
        'summary': {
            'gene_families': len(gene_families),
            'total_family_genes': total_family_genes,
            'chromosomal_clusters': len(clusters),
            'total_cluster_genes': total_cluster_genes,
            'pseudogene_types': len(pseudogenes),
            'pseudogene_parent_matches': total_pseudogene_matches
        },
        'gene_families': gene_families,
        'chromosomal_clusters': clusters,
        'pseudogene_analysis': pseudogenes
    }
    
    print(f"\n📊 PARALOG ANALYSIS SUMMARY")
    print("=" * 40)
    print(f"Gene families identified: {len(gene_families):,}")
    print(f"Genes in families: {total_family_genes:,}")
    print(f"Chromosomal clusters: {len(clusters):,}")
    print(f"Genes in clusters: {total_cluster_genes:,}")
    print(f"Pseudogene-parent matches: {total_pseudogene_matches:,}")
    
    # Save report if requested
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\n💾 Paralog analysis saved to: {output_path}")
    
    return report


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(
        description="Paralog gene analysis utility",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze paralogs in gene features
  python paralog_analyzer.py \\
    --gene-features data/ensembl/spliceai_analysis/gene_features.tsv \\
    --output paralog_analysis.json

  # Quick analysis without output file
  python paralog_analyzer.py \\
    --gene-features data/ensembl/spliceai_analysis/gene_features.tsv

  # Custom clustering distance
  python paralog_analyzer.py \\
    --gene-features data/ensembl/spliceai_analysis/gene_features.tsv \\
    --cluster-distance 500000 \\
    --output paralog_analysis.json
        """
    )
    
    parser.add_argument("--gene-features", required=True,
                       help="Path to gene features TSV file")
    parser.add_argument("--output", 
                       help="Output file for paralog analysis (JSON format)")
    parser.add_argument("--cluster-distance", type=int, default=1000000,
                       help="Maximum distance for chromosomal clustering (default: 1,000,000 bp)")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose output")
    
    args = parser.parse_args()
    
    print("🧬 PARALOG GENE ANALYSIS")
    print("=" * 50)
    
    # Load gene features
    df = load_gene_features(args.gene_features)
    
    # Run analyses
    gene_families = identify_gene_families_by_name(df)
    clusters = identify_chromosomal_clusters(df, args.cluster_distance)
    pseudogenes = analyze_pseudogenes(df)
    
    # Generate report
    report = generate_paralog_report(gene_families, clusters, pseudogenes, args.output)
    
    print(f"\n🎉 Paralog analysis completed successfully!")
    
    # Provide recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    print("-" * 20)
    if len(gene_families) > 0:
        print(f"✅ Consider gene family relationships in meta-model training")
    if len(clusters) > 0:
        print(f"✅ Account for chromosomal clustering in cross-validation splits")
    if sum(len(data['parent_matches']) for data in pseudogenes.values()) > 0:
        print(f"✅ Consider excluding pseudogenes or treating them separately")
    
    sys.exit(0)


if __name__ == "__main__":
    main()
