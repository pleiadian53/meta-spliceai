#!/usr/bin/env python3
"""
Analyze Comprehensive Base Model Test Results

Analyzes the performance of SpliceAI base model on:
- Protein-coding genes vs lncRNA genes
- Coordinate alignment across chromosomes
- Donor vs acceptor performance
- Comparison with SpliceAI paper benchmarks
"""

import sys
from pathlib import Path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

import polars as pl
import numpy as np
from typing import Dict, Tuple


def calculate_metrics(df: pl.DataFrame, pred_type_col: str = 'pred_type') -> Dict[str, float]:
    """Calculate precision, recall, and F1 score from predictions.
    
    Parameters
    ----------
    df : pl.DataFrame
        DataFrame with prediction types (TP, FP, FN, TN)
    pred_type_col : str
        Column name containing prediction types
        
    Returns
    -------
    Dict[str, float]
        Dictionary with precision, recall, F1, and counts
    """
    if pred_type_col not in df.columns:
        return {
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'tp': 0,
            'fp': 0,
            'fn': 0,
            'tn': 0
        }
    
    counts = df.group_by(pred_type_col).agg(pl.count()).to_dict(as_series=False)
    pred_types = counts[pred_type_col]
    count_values = counts['count']
    
    type_counts = dict(zip(pred_types, count_values))
    
    tp = type_counts.get('TP', 0)
    fp = type_counts.get('FP', 0)
    fn = type_counts.get('FN', 0)
    tn = type_counts.get('TN', 0)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn
    }


def analyze_coordinate_offsets(positions_df: pl.DataFrame, annotations_df: pl.DataFrame) -> Dict:
    """Analyze coordinate offsets between predictions and annotations.
    
    Parameters
    ----------
    positions_df : pl.DataFrame
        Predictions with positions
    annotations_df : pl.DataFrame
        Ground truth annotations
        
    Returns
    -------
    Dict
        Offset statistics by site type and strand
    """
    print(f"\n{'='*80}")
    print("COORDINATE OFFSET ANALYSIS")
    print(f"{'='*80}\n")
    
    # For TPs, calculate the offset
    if 'pred_type' not in positions_df.columns:
        print("⚠️  'pred_type' column not found - skipping offset analysis")
        return {}
    
    tps = positions_df.filter(pl.col('pred_type') == 'TP')
    
    if tps.height == 0:
        print("⚠️  No TPs found - cannot analyze offsets")
        return {}
    
    print(f"Analyzing {tps.height:,} True Positive predictions\n")
    
    # Group by site type and strand
    for site_type in ['donor', 'acceptor']:
        print(f"\n{site_type.upper()} SITES:")
        print("-" * 40)
        
        site_tps = tps.filter(pl.col('splice_type') == site_type)
        
        if site_tps.height == 0:
            print(f"  No TPs for {site_type}")
            continue
        
        for strand in ['+', '-']:
            strand_tps = site_tps.filter(pl.col('strand') == strand)
            
            if strand_tps.height == 0:
                continue
            
            # Calculate offset if we have matched_position column
            if 'matched_position' in strand_tps.columns and 'position' in strand_tps.columns:
                offsets = (strand_tps['position'] - strand_tps['matched_position']).to_list()
                
                if offsets:
                    offset_counts = {}
                    for offset in offsets:
                        offset_counts[offset] = offset_counts.get(offset, 0) + 1
                    
                    print(f"\n  Strand {strand}:")
                    print(f"    Total TPs: {len(offsets):,}")
                    print(f"    Offset distribution:")
                    for offset in sorted(offset_counts.keys()):
                        count = offset_counts[offset]
                        pct = 100.0 * count / len(offsets)
                        print(f"      {offset:+3d} bp: {count:6,} ({pct:5.1f}%)")
    
    return {}


def main():
    """Analyze comprehensive base model test results."""
    
    print(f"\n{'='*80}")
    print("COMPREHENSIVE BASE MODEL TEST - ANALYSIS")
    print(f"{'='*80}\n")
    
    # Setup paths
    results_dir = project_root / 'results' / 'base_model_comprehensive_test'
    predictions_dir = results_dir / 'predictions'
    
    if not results_dir.exists():
        print(f"❌ Results directory not found: {results_dir}")
        print("Please run test_base_model_comprehensive.py first.")
        return 1
    
    print(f"Results directory: {results_dir}\n")
    
    # Load gene list
    gene_list_file = results_dir / 'sampled_genes.tsv'
    if gene_list_file.exists():
        gene_list_df = pl.read_csv(gene_list_file, separator='\t')
        print(f"Sampled genes:")
        biotype_counts = gene_list_df.group_by('biotype').agg(pl.count())
        for row in biotype_counts.iter_rows(named=True):
            print(f"  {row['biotype']:20s}: {row['count']:3d} genes")
        print()
    else:
        print("⚠️  Gene list file not found")
        gene_list_df = None
    
    # Load aggregated positions
    positions_file = predictions_dir / 'splice_positions_enhanced_aggregated.tsv'
    
    if not positions_file.exists():
        print(f"❌ Positions file not found: {positions_file}")
        return 1
    
    print(f"Loading predictions from: {positions_file.name}")
    positions_df = pl.read_csv(positions_file, separator='\t')
    
    print(f"Total positions: {positions_df.height:,}")
    print(f"Unique genes: {positions_df['gene_id'].n_unique():,}")
    print()
    
    # Overall metrics
    print(f"{'='*80}")
    print("OVERALL PERFORMANCE")
    print(f"{'='*80}\n")
    
    overall_metrics = calculate_metrics(positions_df)
    
    print(f"Precision: {overall_metrics['precision']:.4f}")
    print(f"Recall:    {overall_metrics['recall']:.4f}")
    print(f"F1 Score:  {overall_metrics['f1']:.4f}")
    print()
    print(f"True Positives:  {overall_metrics['tp']:8,}")
    print(f"False Positives: {overall_metrics['fp']:8,}")
    print(f"False Negatives: {overall_metrics['fn']:8,}")
    print(f"True Negatives:  {overall_metrics['tn']:8,}")
    
    # Performance by biotype
    if gene_list_df is not None:
        print(f"\n{'='*80}")
        print("PERFORMANCE BY BIOTYPE")
        print(f"{'='*80}\n")
        
        # Join with biotype info
        positions_with_biotype = positions_df.join(
            gene_list_df,
            on='gene_id',
            how='left'
        )
        
        for biotype in ['protein_coding', 'lncRNA']:
            biotype_df = positions_with_biotype.filter(pl.col('biotype') == biotype)
            
            if biotype_df.height == 0:
                continue
            
            print(f"{biotype.upper()}:")
            print("-" * 40)
            
            metrics = calculate_metrics(biotype_df)
            
            print(f"  Genes:     {biotype_df['gene_id'].n_unique():3d}")
            print(f"  Positions: {biotype_df.height:,}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall:    {metrics['recall']:.4f}")
            print(f"  F1 Score:  {metrics['f1']:.4f}")
            print(f"  TP: {metrics['tp']:6,}  FP: {metrics['fp']:6,}  FN: {metrics['fn']:6,}")
            print()
    
    # Performance by splice site type
    print(f"{'='*80}")
    print("PERFORMANCE BY SPLICE SITE TYPE")
    print(f"{'='*80}\n")
    
    for site_type in ['donor', 'acceptor']:
        site_df = positions_df.filter(pl.col('splice_type') == site_type)
        
        if site_df.height == 0:
            continue
        
        print(f"{site_type.upper()}:")
        print("-" * 40)
        
        metrics = calculate_metrics(site_df)
        
        print(f"  Positions: {site_df.height:,}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1 Score:  {metrics['f1']:.4f}")
        print(f"  TP: {metrics['tp']:6,}  FP: {metrics['fp']:6,}  FN: {metrics['fn']:6,}")
        print()
    
    # Performance by chromosome
    print(f"{'='*80}")
    print("PERFORMANCE BY CHROMOSOME")
    print(f"{'='*80}\n")
    
    if 'chrom' in positions_df.columns:
        chroms = positions_df['chrom'].unique().sort()
        
        print(f"{'Chrom':<8} {'Genes':>6} {'Positions':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
        print("-" * 70)
        
        for chrom in chroms:
            chrom_df = positions_df.filter(pl.col('chrom') == chrom)
            metrics = calculate_metrics(chrom_df)
            
            print(f"{chrom:<8} {chrom_df['gene_id'].n_unique():6d} {chrom_df.height:10,} "
                  f"{metrics['precision']:10.4f} {metrics['recall']:10.4f} {metrics['f1']:10.4f}")
    
    # Coordinate offset analysis
    # Load splice sites for comparison
    from meta_spliceai.system.genomic_resources import Registry
    registry = Registry(build='GRCh37', release='87')
    splice_sites_file = registry.data_dir / 'splice_sites_enhanced.tsv'
    
    if splice_sites_file.exists():
        ss_df = pl.read_csv(splice_sites_file, separator='\t')
        analyze_coordinate_offsets(positions_df, ss_df)
    
    # Comparison with SpliceAI paper
    print(f"\n{'='*80}")
    print("COMPARISON WITH SPLICEAI PAPER BENCHMARKS")
    print(f"{'='*80}\n")
    
    print("SpliceAI paper (Jaganathan et al., 2019) reported:")
    print("  • Top-k accuracy (k=1): ~95% for canonical splice sites")
    print("  • Precision: ~0.95-0.98 for high-confidence predictions")
    print("  • Recall: ~0.90-0.95 depending on threshold")
    print()
    print("Our results:")
    print(f"  • Precision: {overall_metrics['precision']:.4f}")
    print(f"  • Recall:    {overall_metrics['recall']:.4f}")
    print(f"  • F1 Score:  {overall_metrics['f1']:.4f}")
    print()
    
    if overall_metrics['f1'] >= 0.90:
        print("✅ Performance matches or exceeds SpliceAI paper benchmarks!")
    elif overall_metrics['f1'] >= 0.80:
        print("⚠️  Performance is good but slightly below paper benchmarks")
    else:
        print("⚠️  Performance is below expected benchmarks - investigate further")
    
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}\n")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())



