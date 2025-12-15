#!/usr/bin/env python3
"""
Comprehensive Base Model Test - Gene Category Comparison

Tests the base model (SpliceAI) with diverse gene categories:
- 20 protein-coding genes (expected good performance)
- 10 lncRNA genes (explore performance on non-coding genes)
- 5 edge case genes (tRNA, rRNA, etc. - expected low/no splice sites)

Validates:
- Performance by gene category (protein-coding vs lncRNA vs edge cases)
- Coordinate alignment across chromosomes
- Handling of genes without valid splice sites
- Comparison with SpliceAI paper benchmarks
"""

import sys
from pathlib import Path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

import polars as pl
from typing import List, Dict
import random

from meta_spliceai.system.genomic_resources import Registry
from meta_spliceai.splice_engine.meta_models.core.data_types import SpliceAIConfig
from meta_spliceai.splice_engine.meta_models.workflows.splice_prediction_workflow import (
    run_enhanced_splice_prediction_workflow
)


def sample_genes_by_category(
    registry: Registry,
    n_protein_coding: int = 20,
    n_lncrna: int = 10,
    n_edge_cases: int = 5,
    seed: int = 42
) -> Dict[str, List[str]]:
    """Sample genes by category for comprehensive testing.
    
    Categories:
    1. Protein-coding: Expected to have many splice sites, good performance
    2. lncRNA: Expected to have some splice sites, variable performance
    3. Edge cases: tRNA, rRNA, etc. - expected to have few/no splice sites
    
    Parameters
    ----------
    registry : Registry
        Registry for resolving paths
    n_protein_coding : int
        Number of protein-coding genes to sample
    n_lncrna : int
        Number of lncRNA genes to sample
    n_edge_cases : int
        Number of edge case genes to sample
    seed : int
        Random seed for reproducibility
        
    Returns
    -------
    Dict[str, List[str]]
        Dictionary with gene lists by category
    """
    print(f"\n{'='*80}")
    print("SAMPLING GENES BY CATEGORY")
    print(f"{'='*80}\n")
    
    random.seed(seed)
    
    # Load gene features (has gene_type column)
    gene_features_file = registry.data_dir / 'gene_features.tsv'
    
    if not gene_features_file.exists():
        raise FileNotFoundError(
            f"gene_features.tsv not found at {gene_features_file}\n"
            f"Please generate it first by running:\n"
            f"  python scripts/setup/generate_grch37_gene_features.py"
        )
    
    print(f"Loading gene features from: {gene_features_file}")
    gene_features = pl.read_csv(
        gene_features_file,
        separator='\t',
        schema_overrides={'chrom': pl.Utf8, 'seqname': pl.Utf8}
    )
    
    print(f"Total genes: {gene_features.height:,}")
    
    # Load splice sites to identify genes with splice sites
    splice_sites_file = registry.data_dir / 'splice_sites_enhanced.tsv'
    ss_df = pl.read_csv(
        splice_sites_file,
        separator='\t',
        schema_overrides={'chrom': pl.Utf8}
    )
    
    # Count splice sites per gene
    splice_site_counts = (
        ss_df.group_by('gene_id')
        .agg(pl.count().alias('n_splice_sites'))
    )
    
    print(f"Genes with splice sites: {splice_site_counts.height:,}")
    
    # Join with gene features
    gene_features = gene_features.join(
        splice_site_counts,
        on='gene_id',
        how='left'
    ).with_columns(
        pl.col('n_splice_sites').fill_null(0)
    )
    
    print(f"\nGene biotype distribution:")
    biotype_counts = gene_features.group_by('gene_type').agg(pl.len()).sort('len', descending=True)
    for row in biotype_counts.head(15).iter_rows(named=True):
        print(f"  {row['gene_type']:30s}: {row['len']:6,} genes")
    
    # ========================================================================
    # CATEGORY 1: Protein-coding genes (with splice sites)
    # ========================================================================
    print(f"\n{'='*80}")
    print("CATEGORY 1: Protein-Coding Genes")
    print(f"{'='*80}")
    
    protein_coding = gene_features.filter(
        (pl.col('gene_type') == 'protein_coding') &
        (pl.col('n_splice_sites') >= 4) &  # At least 2 exons
        (pl.col('gene_length') >= 5_000) &
        (pl.col('gene_length') <= 500_000)
    )
    
    print(f"Protein-coding genes (5kb-500kb, ≥4 splice sites): {protein_coding.height:,}")
    
    if protein_coding.height < n_protein_coding:
        print(f"⚠️  Warning: Only {protein_coding.height} protein-coding genes available")
        n_protein_coding = protein_coding.height
    
    sampled_protein_coding = protein_coding.sample(n=n_protein_coding, seed=seed)
    protein_coding_genes = sampled_protein_coding['gene_id'].to_list()
    
    print(f"✅ Sampled {len(protein_coding_genes)} protein-coding genes")
    print(f"   Splice sites per gene: min={sampled_protein_coding['n_splice_sites'].min()}, "
          f"max={sampled_protein_coding['n_splice_sites'].max()}, "
          f"mean={sampled_protein_coding['n_splice_sites'].mean():.1f}")
    
    # ========================================================================
    # CATEGORY 2: lncRNA genes (with splice sites)
    # ========================================================================
    print(f"\n{'='*80}")
    print("CATEGORY 2: lncRNA Genes")
    print(f"{'='*80}")
    
    # lncRNA can be: lincRNA, antisense, processed_transcript, etc.
    lncrna_types = ['lincRNA', 'antisense', 'processed_transcript', 'sense_intronic', 
                    'sense_overlapping', 'lncRNA', 'long_noncoding']
    
    lncrna = gene_features.filter(
        pl.col('gene_type').is_in(lncrna_types) &
        (pl.col('n_splice_sites') >= 2) &  # At least 1 intron
        (pl.col('gene_length') >= 1_000) &
        (pl.col('gene_length') <= 200_000)
    )
    
    print(f"lncRNA genes (1kb-200kb, ≥2 splice sites): {lncrna.height:,}")
    
    if lncrna.height < n_lncrna:
        print(f"⚠️  Warning: Only {lncrna.height} lncRNA genes available")
        n_lncrna = lncrna.height
    
    sampled_lncrna = lncrna.sample(n=n_lncrna, seed=seed)
    lncrna_genes = sampled_lncrna['gene_id'].to_list()
    
    print(f"✅ Sampled {len(lncrna_genes)} lncRNA genes")
    print(f"   Biotypes: {sampled_lncrna['gene_type'].unique().to_list()}")
    print(f"   Splice sites per gene: min={sampled_lncrna['n_splice_sites'].min()}, "
          f"max={sampled_lncrna['n_splice_sites'].max()}, "
          f"mean={sampled_lncrna['n_splice_sites'].mean():.1f}")
    
    # ========================================================================
    # CATEGORY 3: Edge cases (genes without/few splice sites)
    # ========================================================================
    print(f"\n{'='*80}")
    print("CATEGORY 3: Edge Case Genes (tRNA, rRNA, etc.)")
    print(f"{'='*80}")
    
    # Look for genes with no or very few splice sites
    # These include: tRNA, rRNA, snoRNA, snRNA, miRNA, etc.
    edge_case_types = ['Mt_tRNA', 'Mt_rRNA', 'tRNA', 'rRNA', 'snoRNA', 'snRNA', 
                       'miRNA', 'misc_RNA', 'scRNA', 'ribozyme', 'sRNA']
    
    edge_cases = gene_features.filter(
        pl.col('gene_type').is_in(edge_case_types) |
        (
            (pl.col('n_splice_sites') == 0) &
            (pl.col('gene_length') >= 50) &
            (pl.col('gene_length') <= 10_000)
        )
    )
    
    print(f"Edge case genes (no/few splice sites): {edge_cases.height:,}")
    
    if edge_cases.height == 0:
        print("⚠️  No edge case genes found, using genes with 0 splice sites instead")
        edge_cases = gene_features.filter(
            (pl.col('n_splice_sites') == 0) &
            (pl.col('gene_length') >= 50) &
            (pl.col('gene_length') <= 10_000)
        )
        print(f"   Found {edge_cases.height:,} genes with 0 splice sites")
    
    if edge_cases.height < n_edge_cases:
        print(f"⚠️  Warning: Only {edge_cases.height} edge case genes available")
        n_edge_cases = edge_cases.height
    
    if edge_cases.height > 0:
        sampled_edge_cases = edge_cases.sample(n=min(n_edge_cases, edge_cases.height), seed=seed)
        edge_case_genes = sampled_edge_cases['gene_id'].to_list()
        
        print(f"✅ Sampled {len(edge_case_genes)} edge case genes")
        print(f"   Biotypes: {sampled_edge_cases['gene_type'].unique().to_list()}")
        print(f"   Splice sites per gene: min={sampled_edge_cases['n_splice_sites'].min()}, "
              f"max={sampled_edge_cases['n_splice_sites'].max()}, "
              f"mean={sampled_edge_cases['n_splice_sites'].mean():.1f}")
    else:
        edge_case_genes = []
        print("❌ No edge case genes sampled")
    
    # ========================================================================
    # Summary
    # ========================================================================
    print(f"\n{'='*80}")
    print("SAMPLING SUMMARY")
    print(f"{'='*80}\n")
    
    all_genes = protein_coding_genes + lncrna_genes + edge_case_genes
    all_sampled = pl.concat([
        sampled_protein_coding.with_columns(pl.lit('protein_coding').alias('category')),
        sampled_lncrna.with_columns(pl.lit('lncRNA').alias('category')),
        sampled_edge_cases.with_columns(pl.lit('edge_case').alias('category')) if edge_case_genes else pl.DataFrame()
    ])
    
    print(f"Total genes sampled: {len(all_genes)}")
    print(f"  • Protein-coding: {len(protein_coding_genes)}")
    print(f"  • lncRNA: {len(lncrna_genes)}")
    print(f"  • Edge cases: {len(edge_case_genes)}")
    
    # Get chromosome distribution
    chroms = all_sampled['chrom'].unique().sort().to_list()
    print(f"\nChromosomes represented: {', '.join(map(str, chroms))}")
    
    return {
        'protein_coding': protein_coding_genes,
        'lncRNA': lncrna_genes,
        'edge_cases': edge_case_genes,
        'all': all_genes,
        'sampled_df': all_sampled
    }


def analyze_results_by_category(
    results: Dict,
    sampled_genes: Dict[str, List[str]],
    output_dir: Path
):
    """Analyze and compare results by gene category.
    
    Parameters
    ----------
    results : Dict
        Results from workflow
    sampled_genes : Dict
        Sampled genes by category
    output_dir : Path
        Output directory
    """
    print(f"\n{'='*80}")
    print("PERFORMANCE ANALYSIS BY GENE CATEGORY")
    print(f"{'='*80}\n")
    
    if 'positions' not in results or results['positions'] is None:
        print("❌ No positions data available for analysis")
        return
    
    positions_df = results['positions']
    
    # Add category labels to positions
    category_map = {}
    for cat, genes in sampled_genes.items():
        if cat != 'all' and cat != 'sampled_df':
            for gene in genes:
                category_map[gene] = cat
    
    # Create category column
    positions_with_cat = positions_df.with_columns(
        pl.col('gene_id').map_dict(category_map, default='unknown').alias('category')
    )
    
    # Overall statistics
    print("="*80)
    print("OVERALL STATISTICS")
    print("="*80 + "\n")
    
    total_positions = positions_with_cat.height
    print(f"Total positions analyzed: {total_positions:,}")
    
    if 'pred_type' in positions_with_cat.columns:
        pred_counts = positions_with_cat.group_by('pred_type').agg(pl.len().alias('count'))
        print("\nPrediction type distribution:")
        for row in pred_counts.iter_rows(named=True):
            pct = 100 * row['count'] / total_positions
            print(f"  {row['pred_type']:4s}: {row['count']:8,} ({pct:5.2f}%)")
    
    # Category-specific analysis
    print(f"\n{'='*80}")
    print("CATEGORY-SPECIFIC ANALYSIS")
    print(f"{'='*80}\n")
    
    categories = ['protein_coding', 'lncRNA', 'edge_cases']
    
    for category in categories:
        cat_positions = positions_with_cat.filter(pl.col('category') == category)
        
        if cat_positions.height == 0:
            print(f"\n{category.upper()}: No data")
            continue
        
        print(f"\n{category.upper()}")
        print("-" * 80)
        
        n_genes = cat_positions['gene_id'].n_unique()
        n_positions = cat_positions.height
        
        print(f"Genes: {n_genes}")
        print(f"Positions analyzed: {n_positions:,}")
        
        if 'pred_type' in cat_positions.columns:
            pred_counts = cat_positions.group_by('pred_type').agg(pl.len().alias('count'))
            
            tp = pred_counts.filter(pl.col('pred_type') == 'TP')['count'].sum() or 0
            fp = pred_counts.filter(pl.col('pred_type') == 'FP')['count'].sum() or 0
            fn = pred_counts.filter(pl.col('pred_type') == 'FN')['count'].sum() or 0
            tn = pred_counts.filter(pl.col('pred_type') == 'TN')['count'].sum() or 0
            
            print(f"\nPrediction outcomes:")
            print(f"  TP: {tp:6,}")
            print(f"  FP: {fp:6,}")
            print(f"  FN: {fn:6,}")
            print(f"  TN: {tn:6,}")
            
            # Calculate metrics
            if tp + fp > 0:
                precision = tp / (tp + fp)
                print(f"\nPrecision: {precision:.4f} ({precision*100:.2f}%)")
            else:
                print(f"\nPrecision: N/A (no positive predictions)")
            
            if tp + fn > 0:
                recall = tp / (tp + fn)
                print(f"Recall:    {recall:.4f} ({recall*100:.2f}%)")
                
                if tp + fp > 0:
                    f1 = 2 * (precision * recall) / (precision + recall)
                    print(f"F1 Score:  {f1:.4f} ({f1*100:.2f}%)")
            else:
                print(f"Recall:    N/A (no true splice sites)")
            
            # Splice site statistics
            if 'splice_type' in cat_positions.columns:
                true_sites = cat_positions.filter(pl.col('splice_type').is_not_null())
                if true_sites.height > 0:
                    n_donor = true_sites.filter(pl.col('splice_type') == 'donor').height
                    n_acceptor = true_sites.filter(pl.col('splice_type') == 'acceptor').height
                    print(f"\nTrue splice sites:")
                    print(f"  Donor:    {n_donor:6,}")
                    print(f"  Acceptor: {n_acceptor:6,}")
            
            # Average scores for true splice sites
            if 'splice_type' in cat_positions.columns and 'score' in cat_positions.columns:
                true_donors = cat_positions.filter(pl.col('splice_type') == 'donor')
                true_acceptors = cat_positions.filter(pl.col('splice_type') == 'acceptor')
                
                if true_donors.height > 0:
                    avg_donor_score = true_donors['score'].mean()
                    print(f"\nAverage scores at true sites:")
                    print(f"  Donor:    {avg_donor_score:.4f}")
                
                if true_acceptors.height > 0:
                    avg_acceptor_score = true_acceptors['score'].mean()
                    print(f"  Acceptor: {avg_acceptor_score:.4f}")
    
    # Comparative summary
    print(f"\n{'='*80}")
    print("COMPARATIVE SUMMARY")
    print(f"{'='*80}\n")
    
    summary_data = []
    
    for category in categories:
        cat_positions = positions_with_cat.filter(pl.col('category') == category)
        
        if cat_positions.height == 0:
            continue
        
        if 'pred_type' in cat_positions.columns:
            pred_counts = cat_positions.group_by('pred_type').agg(pl.len().alias('count'))
            
            tp = pred_counts.filter(pl.col('pred_type') == 'TP')['count'].sum() or 0
            fp = pred_counts.filter(pl.col('pred_type') == 'FP')['count'].sum() or 0
            fn = pred_counts.filter(pl.col('pred_type') == 'FN')['count'].sum() or 0
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            summary_data.append({
                'category': category,
                'n_genes': cat_positions['gene_id'].n_unique(),
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'tp': tp,
                'fp': fp,
                'fn': fn
            })
    
    if summary_data:
        summary_df = pl.DataFrame(summary_data)
        print(summary_df)
        
        # Save summary
        summary_file = output_dir / 'category_performance_summary.tsv'
        summary_df.write_csv(summary_file, separator='\t')
        print(f"\n✅ Saved performance summary: {summary_file}")


def main():
    """Run comprehensive gene category test."""
    print("\n" + "="*80)
    print("COMPREHENSIVE BASE MODEL TEST - GENE CATEGORY COMPARISON")
    print("="*80 + "\n")
    
    print("Testing SpliceAI base model with:")
    print("  • 20 protein-coding genes (expected good performance)")
    print("  • 10 lncRNA genes (variable performance)")
    print("  • 5 edge case genes (tRNA, rRNA, etc. - low/no splice sites)")
    print("  • Automatic coordinate adjustment detection")
    print("  • Genome-wide sampling")
    print()
    
    # Setup
    registry = Registry(build='GRCh37', release='87')
    gtf_file = registry.get_gtf_path(validate=False)
    fasta_file = registry.get_fasta_path(validate=False)
    splice_sites_file = registry.data_dir / 'splice_sites_enhanced.tsv'
    
    print(f"Build: GRCh37 (release 87)")
    print(f"GTF: {gtf_file}")
    print(f"FASTA: {fasta_file}")
    print(f"Splice sites: {splice_sites_file}")
    
    # Sample genes by category
    sampled_genes = sample_genes_by_category(
        registry,
        n_protein_coding=20,
        n_lncrna=10,
        n_edge_cases=5,
        seed=42
    )
    
    all_genes = sampled_genes['all']
    
    if not all_genes:
        print("\n❌ Error: No genes sampled. Cannot proceed.")
        return 1
    
    print(f"\n{'='*80}")
    print("RUNNING BASE MODEL PREDICTIONS")
    print(f"{'='*80}\n")
    
    # Setup output directory
    output_dir = project_root / 'results' / 'base_model_gene_categories_test'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    print()
    
    # Configure workflow
    config = SpliceAIConfig(
        gtf_file=str(gtf_file),
        genome_fasta=str(fasta_file),
        eval_dir=str(output_dir),
        output_subdir='predictions',
        local_dir=str(registry.data_dir),
        mode='test',  # Test mode: artifacts are overwritable
        coverage='gene_subset',  # Testing 35 genes
        test_name='gene_categories_test',  # Named test
        threshold=0.5,
        consensus_window=2,
        error_window=500,
        use_auto_position_adjustments=True,  # Enable automatic adjustment detection
        test_mode=False,
        do_extract_annotations=True,   # Extract fresh (fast for 35 genes)
        do_extract_splice_sites=False, # Use existing from data_dir
        do_extract_sequences=True,     # Extract for target genes
        do_find_overlaping_genes=False,
        chromosomes=None,  # Will be inferred from genes
        separator='\t',
        format='parquet',
        seq_format='parquet',
        seq_mode='gene',
        seq_type='minmax'
    )
    
    print("Configuration:")
    print(f"  • Threshold: {config.threshold}")
    print(f"  • Consensus window: {config.consensus_window}")
    print(f"  • Error window: {config.error_window}")
    print(f"  • Auto adjustments: {config.use_auto_position_adjustments}")
    print(f"  • Mode: {config.mode}")
    print(f"  • Coverage: {config.coverage}")
    print()
    
    # Run workflow
    print("Starting workflow...\n")
    
    try:
        results = run_enhanced_splice_prediction_workflow(
            config=config,
            target_genes=all_genes,
            verbosity=1,
            no_final_aggregate=False,
            no_tn_sampling=True  # Keep all positions for analysis
        )
        
        if not results.get('success'):
            print("\n❌ Workflow failed!")
            return 1
        
        print(f"\n{'='*80}")
        print("WORKFLOW COMPLETED SUCCESSFULLY")
        print(f"{'='*80}\n")
        
        # Save gene list with categories
        gene_list_file = output_dir / 'sampled_genes_by_category.tsv'
        sampled_genes['sampled_df'].write_csv(gene_list_file, separator='\t')
        print(f"✅ Saved gene list: {gene_list_file}")
        
        # Analyze results by category
        analyze_results_by_category(results, sampled_genes, output_dir)
        
        print(f"\n{'='*80}")
        print("NEXT STEPS")
        print(f"{'='*80}\n")
        print("Review the performance comparison:")
        print(f"  • Summary: {output_dir}/category_performance_summary.tsv")
        print(f"  • Full results: {output_dir}/predictions/")
        print()
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error during workflow execution: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

