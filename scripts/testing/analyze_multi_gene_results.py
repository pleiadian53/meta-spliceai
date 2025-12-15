#!/usr/bin/env python3
"""
Analyze multi-gene test results by biotype and by gene.

This script:
1. Loads results from multi-gene test
2. Calculates metrics by biotype (protein-coding vs lncRNA)
3. Calculates metrics by gene
4. Tests coordinate adjustments for each gene
5. Generates comprehensive report

Date: November 2, 2025
"""

import sys
from pathlib import Path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

import json
import argparse
import polars as pl
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_curve, auc


def load_results(results_dir: Path):
    """Load results and gene info."""
    
    # Load gene info
    gene_info_path = results_dir / 'gene_info.json'
    if not gene_info_path.exists():
        print(f"[ERROR] Gene info not found: {gene_info_path}")
        return None, None
    
    with open(gene_info_path, 'r') as f:
        gene_sample = json.load(f)
    
    # Load positions
    positions_file = results_dir / 'full_splice_positions_enhanced.tsv'
    if not positions_file.exists():
        print(f"[ERROR] Positions file not found: {positions_file}")
        return None, None
    
    positions_df = pl.read_csv(str(positions_file), separator='\t')
    
    return positions_df, gene_sample


def calculate_gene_metrics(positions_df: pl.DataFrame, gene_id: str, splice_type: str):
    """Calculate metrics for a single gene and splice type."""
    
    gene_df = positions_df.filter(
        (pl.col('gene_id') == gene_id) &
        (pl.col('splice_type') == splice_type)
    )
    
    if gene_df.height == 0:
        return None
    
    # Use pred_type for classification
    tp = gene_df.filter(pl.col('pred_type') == 'TP').height
    fp = gene_df.filter(pl.col('pred_type') == 'FP').height
    fn = gene_df.filter(pl.col('pred_type') == 'FN').height
    tn = gene_df.filter(pl.col('pred_type') == 'TN').height
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Test coordinate adjustment (consensus_window=0)
    high_score_df = gene_df.filter(
        (pl.col(f'{splice_type}_score') >= 0.5) &
        (pl.col('true_position').is_not_null())
    )
    
    exact_matches = 0
    most_common_offset = None
    
    if high_score_df.height > 0:
        offset_df = high_score_df.with_columns(
            (pl.col('predicted_position') - pl.col('true_position')).alias('offset')
        )
        exact_matches = offset_df.filter(pl.col('offset') == 0).height
        
        # Get most common offset
        offset_counts = offset_df.group_by('offset').agg(pl.len()).sort('len', descending=True)
        if offset_counts.height > 0:
            most_common_offset = offset_counts['offset'][0]
    
    exact_match_pct = (exact_matches / high_score_df.height * 100) if high_score_df.height > 0 else 0
    
    return {
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'n_positions': gene_df.height,
        'n_high_score': high_score_df.height,
        'exact_matches': exact_matches,
        'exact_match_pct': exact_match_pct,
        'most_common_offset': most_common_offset
    }


def main():
    """Main function."""
    
    parser = argparse.ArgumentParser(description='Analyze multi-gene test results')
    parser.add_argument('--results-dir', type=str,
                       default='/Users/pleiadian53/work/meta-spliceai/data/ensembl/GRCh37/spliceai_eval/meta_models/multi_gene_test',
                       help='Results directory')
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    
    print(f"\n{'='*80}")
    print("MULTI-GENE TEST RESULTS ANALYSIS")
    print(f"{'='*80}\n")
    
    print(f"Loading results from: {results_dir}\n")
    
    # Load data
    positions_df, gene_sample = load_results(results_dir)
    
    if positions_df is None or gene_sample is None:
        return 1
    
    print(f"Total positions: {positions_df.height:,}")
    print(f"Unique genes: {positions_df['gene_id'].n_unique()}")
    print()
    
    # Analyze by biotype
    print(f"{'='*80}")
    print("ANALYSIS BY BIOTYPE")
    print(f"{'='*80}\n")
    
    gene_info = gene_sample['gene_info']
    
    for biotype in ['protein_coding', 'lncRNA']:
        gene_ids = [gid for gid, info in gene_info.items() if info['gene_biotype'] == biotype]
        
        if not gene_ids:
            continue
        
        print(f"{biotype.upper()} GENES ({len(gene_ids)} genes)")
        print("-" * 80)
        
        biotype_df = positions_df.filter(pl.col('gene_id').is_in(gene_ids))
        
        if biotype_df.height == 0:
            print(f"  No predictions found\n")
            continue
        
        for splice_type in ['donor', 'acceptor']:
            type_df = biotype_df.filter(pl.col('splice_type') == splice_type)
            
            if type_df.height == 0:
                continue
            
            # Calculate aggregate metrics
            tp = type_df.filter(pl.col('pred_type') == 'TP').height
            fp = type_df.filter(pl.col('pred_type') == 'FP').height
            fn = type_df.filter(pl.col('pred_type') == 'FN').height
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            print(f"\n{splice_type.capitalize()} sites:")
            print(f"  Total positions: {type_df.height:,}")
            print(f"  TP: {tp:,}, FP: {fp:,}, FN: {fn:,}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall:    {recall:.4f}")
            print(f"  F1 Score:  {f1:.4f}")
        
        print()
    
    # Analyze by gene
    print(f"{'='*80}")
    print("ANALYSIS BY GENE")
    print(f"{'='*80}\n")
    
    gene_results = []
    
    for gene_id, info in gene_info.items():
        gene_df = positions_df.filter(pl.col('gene_id') == gene_id)
        
        if gene_df.height == 0:
            continue
        
        gene_result = {
            'gene_id': gene_id,
            'gene_name': info['gene_name'],
            'biotype': info['gene_biotype'],
            'chrom': info['chrom']
        }
        
        for splice_type in ['donor', 'acceptor']:
            metrics = calculate_gene_metrics(positions_df, gene_id, splice_type)
            if metrics:
                gene_result[f'{splice_type}_f1'] = metrics['f1']
                gene_result[f'{splice_type}_precision'] = metrics['precision']
                gene_result[f'{splice_type}_recall'] = metrics['recall']
                gene_result[f'{splice_type}_exact_match_pct'] = metrics['exact_match_pct']
                gene_result[f'{splice_type}_offset'] = metrics['most_common_offset']
        
        gene_results.append(gene_result)
    
    # Display gene results
    print("Gene-level performance:\n")
    print(f"{'Gene':<20s} {'Biotype':<15s} {'Chr':<5s} {'Donor F1':<10s} {'Acc F1':<10s} {'D Exact%':<10s} {'A Exact%':<10s} {'D Off':<6s} {'A Off':<6s}")
    print("-" * 110)
    
    for result in sorted(gene_results, key=lambda x: x['biotype']):
        donor_f1 = result.get('donor_f1', 0)
        acc_f1 = result.get('acceptor_f1', 0)
        d_exact = result.get('donor_exact_match_pct', 0)
        a_exact = result.get('acceptor_exact_match_pct', 0)
        d_off = result.get('donor_offset', 'N/A')
        a_off = result.get('acceptor_offset', 'N/A')
        
        print(f"{result['gene_name']:<20s} {result['biotype']:<15s} {result['chrom']:<5s} "
              f"{donor_f1:<10.4f} {acc_f1:<10.4f} {d_exact:<10.1f} {a_exact:<10.1f} "
              f"{str(d_off):<6s} {str(a_off):<6s}")
    
    # Summary statistics
    print(f"\n{'='*80}")
    print("SUMMARY STATISTICS")
    print(f"{'='*80}\n")
    
    for biotype in ['protein_coding', 'lncRNA']:
        biotype_genes = [r for r in gene_results if r['biotype'] == biotype]
        
        if not biotype_genes:
            continue
        
        donor_f1s = [r.get('donor_f1', 0) for r in biotype_genes if 'donor_f1' in r]
        acc_f1s = [r.get('acceptor_f1', 0) for r in biotype_genes if 'acceptor_f1' in r]
        
        print(f"{biotype.upper()} ({len(biotype_genes)} genes):")
        if donor_f1s:
            print(f"  Donor F1:    Mean={np.mean(donor_f1s):.4f}, Std={np.std(donor_f1s):.4f}, Min={np.min(donor_f1s):.4f}, Max={np.max(donor_f1s):.4f}")
        if acc_f1s:
            print(f"  Acceptor F1: Mean={np.mean(acc_f1s):.4f}, Std={np.std(acc_f1s):.4f}, Min={np.min(acc_f1s):.4f}, Max={np.max(acc_f1s):.4f}")
        print()
    
    # Adjustment analysis
    print(f"{'='*80}")
    print("COORDINATE ADJUSTMENT ANALYSIS")
    print(f"{'='*80}\n")
    
    for splice_type in ['donor', 'acceptor']:
        offsets = [r.get(f'{splice_type}_offset') for r in gene_results 
                  if f'{splice_type}_offset' in r and r.get(f'{splice_type}_offset') is not None]
        
        if offsets:
            from collections import Counter
            offset_counts = Counter(offsets)
            
            print(f"{splice_type.capitalize()} offset distribution:")
            for offset, count in sorted(offset_counts.items()):
                pct = count / len(offsets) * 100
                print(f"  {offset:+3d}bp: {count:2d} genes ({pct:5.1f}%)")
            print()
    
    print(f"{'='*80}")
    print("TEST COMPLETE")
    print(f"{'='*80}\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())




