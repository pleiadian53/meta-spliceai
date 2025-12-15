#!/usr/bin/env python3
"""
Analyze results from diverse gene test.

This script loads the results from the completed test and calculates metrics.
"""

import sys
from pathlib import Path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

import polars as pl
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_curve, auc


def load_results(results_dir: str):
    """Load error analysis and positions from disk."""
    results_path = Path(results_dir)
    
    # Load error analysis
    error_file = results_path / "full_splice_errors.tsv"
    positions_file = results_path / "full_splice_positions_enhanced.tsv"
    
    if not error_file.exists():
        print(f"Error file not found: {error_file}")
        return None, None
    
    if not positions_file.exists():
        print(f"Positions file not found: {positions_file}")
        return None, None
    
    error_df = pl.read_csv(str(error_file), separator='\t')
    positions_df = pl.read_csv(str(positions_file), separator='\t')
    
    return error_df, positions_df


def calculate_metrics(error_df: pl.DataFrame, positions_df: pl.DataFrame):
    """Calculate F1 scores and PR-AUC from error analysis."""
    
    print(f"\n{'='*80}")
    print("OVERALL METRICS")
    print(f"{'='*80}\n")
    
    print(f"Total positions: {positions_df.height:,}")
    print(f"Unique genes: {positions_df['gene_id'].n_unique()}")
    print(f"Total errors: {error_df.height:,}")
    print()
    
    # Get error counts
    error_counts = error_df.group_by(['splice_type', 'error_type']).agg(pl.count()).sort(['splice_type', 'error_type'])
    print("Error counts by type:")
    print(error_counts)
    print()
    
    # Calculate metrics for each splice type
    results = {}
    
    for splice_type in ['donor', 'acceptor']:
        # Filter errors for this splice type
        type_errors = error_df.filter(pl.col('splice_type') == splice_type)
        
        if type_errors.height == 0:
            print(f"\n[warning] No errors found for {splice_type} sites")
            continue
        
        # Count error types
        tp = type_errors.filter(pl.col('error_type') == 'TP').height
        fp = type_errors.filter(pl.col('error_type') == 'FP').height
        fn = type_errors.filter(pl.col('error_type') == 'FN').height
        tn = type_errors.filter(pl.col('error_type') == 'TN').height
        
        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"\n{splice_type.capitalize()} sites:")
        print(f"  True positives:  {tp:,}")
        print(f"  False positives: {fp:,}")
        print(f"  False negatives: {fn:,}")
        print(f"  True negatives:  {tn:,}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1 Score:  {f1:.4f}")
        
        # Calculate PR-AUC from positions DataFrame
        type_positions = positions_df.filter(pl.col('splice_type') == splice_type)
        
        if type_positions.height > 0:
            # Merge with error types
            type_positions_with_errors = type_positions.join(
                type_errors.select(['position', 'error_type']),
                on='position',
                how='left'
            )
            
            # Get scores and labels
            y_scores = type_positions_with_errors[f'{splice_type}_score'].to_numpy()
            y_true = type_positions_with_errors['error_type'].is_in(['TP', 'FN']).to_numpy()
            
            # Calculate PR-AUC
            precisions, recalls, _ = precision_recall_curve(y_true, y_scores)
            pr_auc = auc(recalls, precisions)
            
            print(f"  PR-AUC:    {pr_auc:.4f}")
        
        results[splice_type] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'tn': tn
        }
    
    return results


def main():
    """Main function."""
    
    print(f"\n{'='*80}")
    print("DIVERSE GENE TEST RESULTS ANALYSIS")
    print(f"{'='*80}\n")
    
    # Results directory
    results_dir = "/Users/pleiadian53/work/meta-spliceai/data/ensembl/GRCh37/spliceai_eval/meta_models/diverse_genes_test"
    
    print(f"Loading results from: {results_dir}\n")
    
    # Load results
    error_df, positions_df = load_results(results_dir)
    
    if error_df is None or positions_df is None:
        print("\n[ERROR] Failed to load results")
        return 1
    
    print(f"Loaded:")
    print(f"  Error analysis: {error_df.height:,} rows")
    print(f"  Positions:      {positions_df.height:,} rows")
    
    # Calculate metrics
    metrics = calculate_metrics(error_df, positions_df)
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}\n")
    
    if 'donor' in metrics and 'acceptor' in metrics:
        avg_f1 = (metrics['donor']['f1'] + metrics['acceptor']['f1']) / 2
        print(f"Average F1 Score: {avg_f1:.4f}")
        print(f"  Donor:    {metrics['donor']['f1']:.4f}")
        print(f"  Acceptor: {metrics['acceptor']['f1']:.4f}")
    
    print(f"\n{'='*80}")
    print("NOTE")
    print(f"{'='*80}\n")
    print("Only 1 gene (ENSG00000160216 - AGPAT3 on chr21) was processed.")
    print("This is because the workflow filtered by chromosomes containing target genes,")
    print("but only chr21 was processed in this run.")
    print()
    print("To test all 20 sampled genes, the workflow would need to process all")
    print("chromosomes where these genes are located (7, 3, 15, 5, 2, 18, 21, X, 4, 19, etc.)")
    
    print(f"\n{'='*80}")
    print("TEST COMPLETE")
    print(f"{'='*80}\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())



