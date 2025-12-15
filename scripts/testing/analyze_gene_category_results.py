#!/usr/bin/env python3
"""
Analyze Gene Category Test Results

Analyzes the results from test_base_model_gene_categories.py and generates:
1. Performance metrics by category
2. Comparative analysis
3. Edge case validation
"""

import sys
from pathlib import Path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

import polars as pl
import numpy as np
import argparse


def analyze_results(results_dir_name='base_model_gene_categories_test'):
    """Analyze gene category test results.
    
    Parameters
    ----------
    results_dir_name : str
        Name of the results directory under results/
    """
    
    results_dir = project_root / 'results' / results_dir_name
    
    print("\n" + "="*80)
    print("GENE CATEGORY TEST RESULTS ANALYSIS")
    print("="*80 + "\n")
    
    # Load sampled genes with categories
    # Try both possible filenames
    genes_file = results_dir / 'sampled_genes_by_category.tsv'
    if not genes_file.exists():
        genes_file = results_dir / 'sampled_genes.tsv'
    
    if not genes_file.exists():
        print(f"❌ Gene list file not found in {results_dir}")
        print(f"   Tried: sampled_genes_by_category.tsv, sampled_genes.tsv")
        return 1
    
    genes_df = pl.read_csv(genes_file, separator='\t')
    print(f"Loaded {genes_df.height} genes")
    print(f"Categories: {genes_df['category'].unique().to_list()}")
    
    # Load positions
    positions_file = results_dir / 'meta_models' / 'predictions' / 'full_splice_positions_enhanced.tsv'
    if not positions_file.exists():
        print(f"❌ Positions file not found: {positions_file}")
        return 1
    
    print(f"\nLoading positions from: {positions_file}")
    print("(This may take a moment for large files...)")
    
    positions_df = pl.read_csv(
        positions_file,
        separator='\t',
        schema_overrides={'chrom': pl.Utf8}
    )
    
    print(f"✅ Loaded {positions_df.height:,} positions")
    
    # Create gene_id to category mapping
    gene_category_map = dict(zip(genes_df['gene_id'].to_list(), genes_df['category'].to_list()))
    
    # Add category column to positions
    positions_with_cat = positions_df.with_columns([
        pl.col('gene_id').replace(gene_category_map, default='unknown').alias('category')
    ])
    
    # Overall statistics
    print("\n" + "="*80)
    print("OVERALL STATISTICS")
    print("="*80 + "\n")
    
    total_positions = positions_with_cat.height
    n_genes = positions_with_cat['gene_id'].n_unique()
    
    print(f"Total positions analyzed: {total_positions:,}")
    print(f"Total genes: {n_genes}")
    
    if 'pred_type' in positions_with_cat.columns:
        pred_counts = positions_with_cat.group_by('pred_type').agg(pl.len().alias('count')).sort('count', descending=True)
        print("\nPrediction type distribution:")
        for row in pred_counts.iter_rows(named=True):
            pct = 100 * row['count'] / total_positions
            print(f"  {row['pred_type']:4s}: {row['count']:10,} ({pct:5.2f}%)")
    
    # Category-specific analysis
    print("\n" + "="*80)
    print("CATEGORY-SPECIFIC ANALYSIS")
    print("="*80)
    
    categories = ['protein_coding', 'lncRNA', 'edge_cases']
    summary_data = []
    
    for category in categories:
        cat_genes = genes_df.filter(pl.col('category') == category)
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
        
        # Get splice site counts from gene list
        avg_splice_sites = cat_genes['n_splice_sites'].mean() if 'n_splice_sites' in cat_genes.columns else 0
        print(f"Average splice sites per gene: {avg_splice_sites:.1f}")
        
        if 'pred_type' in cat_positions.columns:
            pred_counts = cat_positions.group_by('pred_type').agg(pl.len().alias('count'))
            
            tp = pred_counts.filter(pl.col('pred_type') == 'TP').select('count').sum()
            fp = pred_counts.filter(pl.col('pred_type') == 'FP').select('count').sum()
            fn = pred_counts.filter(pl.col('pred_type') == 'FN').select('count').sum()
            tn = pred_counts.filter(pl.col('pred_type') == 'TN').select('count').sum()
            
            tp = tp.item() if tp.height > 0 else 0
            fp = fp.item() if fp.height > 0 else 0
            fn = fn.item() if fn.height > 0 else 0
            tn = tn.item() if tn.height > 0 else 0
            
            print(f"\nPrediction outcomes:")
            print(f"  TP (True Positives):  {tp:8,}")
            print(f"  FP (False Positives): {fp:8,}")
            print(f"  FN (False Negatives): {fn:8,}")
            print(f"  TN (True Negatives):  {tn:8,}")
            
            # Calculate metrics
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            print(f"\n📊 Performance Metrics:")
            print(f"  Precision: {precision:.4f} ({precision*100:5.2f}%)")
            print(f"  Recall:    {recall:.4f} ({recall*100:5.2f}%)")
            print(f"  F1 Score:  {f1:.4f} ({f1*100:5.2f}%)")
            
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
                    max_donor_score = true_donors['score'].max()
                    print(f"\nScores at true donor sites:")
                    print(f"  Mean: {avg_donor_score:.4f}")
                    print(f"  Max:  {max_donor_score:.4f}")
                
                if true_acceptors.height > 0:
                    avg_acceptor_score = true_acceptors['score'].mean()
                    max_acceptor_score = true_acceptors['score'].max()
                    print(f"\nScores at true acceptor sites:")
                    print(f"  Mean: {avg_acceptor_score:.4f}")
                    print(f"  Max:  {max_acceptor_score:.4f}")
            
            # For edge cases, check max prediction score
            if category == 'edge_cases':
                if 'score' in cat_positions.columns:
                    max_score = cat_positions['score'].max()
                    high_scores = cat_positions.filter(pl.col('score') > 0.5).height
                    print(f"\n🔍 Edge Case Validation:")
                    print(f"  Max prediction score: {max_score:.4f}")
                    print(f"  Positions with score > 0.5: {high_scores:,}")
                    print(f"  False positives per gene: {fp / n_genes:.1f}")
                    
                    if max_score < 0.3:
                        print(f"  ✅ Excellent: Max score < 0.3")
                    elif max_score < 0.5:
                        print(f"  ✅ Good: Max score < 0.5")
                    else:
                        print(f"  ⚠️  Warning: Max score >= 0.5")
            
            # Store summary data
            summary_data.append({
                'category': category,
                'n_genes': n_genes,
                'n_positions': n_positions,
                'tp': tp,
                'fp': fp,
                'fn': fn,
                'tn': tn,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            })
    
    # Comparative summary
    print("\n" + "="*80)
    print("COMPARATIVE SUMMARY")
    print("="*80 + "\n")
    
    if summary_data:
        summary_df = pl.DataFrame(summary_data)
        print(summary_df)
        
        # Save summary
        summary_file = results_dir / 'category_performance_summary.tsv'
        summary_df.write_csv(summary_file, separator='\t')
        print(f"\n✅ Saved performance summary: {summary_file}")
        
        # Performance assessment
        print("\n" + "="*80)
        print("PERFORMANCE ASSESSMENT")
        print("="*80 + "\n")
        
        for row in summary_df.iter_rows(named=True):
            cat = row['category']
            f1 = row['f1_score']
            
            print(f"{cat.upper()}:")
            
            if cat == 'protein_coding':
                if f1 >= 0.95:
                    print(f"  ✅ EXCELLENT (F1={f1:.3f} >= 0.95)")
                elif f1 >= 0.90:
                    print(f"  ✅ GOOD (F1={f1:.3f} >= 0.90)")
                elif f1 >= 0.85:
                    print(f"  🔶 ACCEPTABLE (F1={f1:.3f} >= 0.85)")
                else:
                    print(f"  ❌ BELOW THRESHOLD (F1={f1:.3f} < 0.85)")
            
            elif cat == 'lncRNA':
                if f1 >= 0.85:
                    print(f"  ✅ EXCELLENT (F1={f1:.3f} >= 0.85)")
                elif f1 >= 0.75:
                    print(f"  ✅ GOOD (F1={f1:.3f} >= 0.75)")
                elif f1 >= 0.70:
                    print(f"  🔶 ACCEPTABLE (F1={f1:.3f} >= 0.70)")
                else:
                    print(f"  ❌ BELOW THRESHOLD (F1={f1:.3f} < 0.70)")
            
            elif cat == 'edge_cases':
                fp_per_gene = row['fp'] / row['n_genes']
                if fp_per_gene < 5:
                    print(f"  ✅ EXCELLENT (FP/gene={fp_per_gene:.1f} < 5)")
                elif fp_per_gene < 10:
                    print(f"  ✅ GOOD (FP/gene={fp_per_gene:.1f} < 10)")
                elif fp_per_gene < 20:
                    print(f"  🔶 ACCEPTABLE (FP/gene={fp_per_gene:.1f} < 20)")
                else:
                    print(f"  ❌ HIGH FP RATE (FP/gene={fp_per_gene:.1f} >= 20)")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80 + "\n")
    
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Analyze gene category test results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze Run 1 (default)
  python analyze_gene_category_results.py
  
  # Analyze Run 2
  python analyze_gene_category_results.py base_model_validation_run2
  
  # Analyze any results directory
  python analyze_gene_category_results.py my_custom_test
        """
    )
    
    parser.add_argument(
        'results_dir',
        nargs='?',
        default='base_model_gene_categories_test',
        help='Name of results directory under results/ (default: base_model_gene_categories_test)'
    )
    
    args = parser.parse_args()
    sys.exit(analyze_results(args.results_dir))

