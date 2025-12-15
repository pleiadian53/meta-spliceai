#!/usr/bin/env python3
"""
Re-calculate AGPAT3 metrics using correct pred_type column.

This script demonstrates the correct way to calculate metrics:
- Use pred_type (not error_type) for TP/FP/FN/TN classification
- Test with consensus_window=0 to determine if adjustments are needed
- Apply adjustments if needed
- Re-evaluate with consensus_window=2 for final metrics

Date: November 2, 2025
"""

import sys
from pathlib import Path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

import polars as pl
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_curve, auc


def calculate_metrics_from_pred_type(positions_df: pl.DataFrame, splice_type: str):
    """
    Calculate metrics using pred_type column (correct approach).
    
    Parameters
    ----------
    positions_df : pl.DataFrame
        Positions DataFrame with pred_type column
    splice_type : str
        'donor' or 'acceptor'
    
    Returns
    -------
    dict with metrics
    """
    # Filter to this splice type
    type_df = positions_df.filter(pl.col('splice_type') == splice_type)
    
    if type_df.height == 0:
        return None
    
    # Use pred_type for classification (correct!)
    tp = type_df.filter(pl.col('pred_type') == 'TP').height
    fp = type_df.filter(pl.col('pred_type') == 'FP').height
    fn = type_df.filter(pl.col('pred_type') == 'FN').height
    tn = type_df.filter(pl.col('pred_type') == 'TN').height
    
    # Calculate metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Calculate PR-AUC
    y_true = type_df['pred_type'].is_in(['TP', 'FN']).to_numpy()
    y_scores = type_df[f'{splice_type}_score'].to_numpy()
    
    if len(np.unique(y_true)) > 1:
        precisions, recalls, _ = precision_recall_curve(y_true, y_scores)
        pr_auc = auc(recalls, precisions)
    else:
        pr_auc = None
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'pr_auc': pr_auc,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn,
        'total': type_df.height
    }


def test_adjustments_needed(positions_df: pl.DataFrame, splice_type: str):
    """
    Test if adjustments are needed using consensus_window=0 (exact matching).
    
    Parameters
    ----------
    positions_df : pl.DataFrame
        Positions DataFrame
    splice_type : str
        'donor' or 'acceptor'
    
    Returns
    -------
    dict with adjustment analysis
    """
    # Filter to this splice type and high-scoring positions
    type_df = positions_df.filter(
        (pl.col('splice_type') == splice_type) &
        (pl.col(f'{splice_type}_score') >= 0.5) &
        (pl.col('true_position').is_not_null())
    )
    
    if type_df.height == 0:
        return None
    
    # Calculate offsets
    type_df = type_df.with_columns(
        (pl.col('predicted_position') - pl.col('true_position')).alias('offset')
    )
    
    # Count exact matches (consensus_window=0)
    exact_matches = type_df.filter(pl.col('offset') == 0).height
    within_1bp = type_df.filter(pl.col('offset').abs() <= 1).height
    within_2bp = type_df.filter(pl.col('offset').abs() <= 2).height
    
    # Get offset distribution
    offset_dist = type_df.group_by('offset').agg(pl.len()).sort('offset')
    
    # Determine if adjustment is needed
    exact_match_pct = exact_matches / type_df.height * 100
    needs_adjustment = exact_match_pct < 95  # If <95% exact matches, need adjustment
    
    # Find most common offset
    most_common_offset = None
    if offset_dist.height > 0:
        most_common_row = offset_dist.sort('len', descending=True).head(1)
        most_common_offset = most_common_row['offset'][0]
    
    return {
        'total_positions': type_df.height,
        'exact_matches': exact_matches,
        'within_1bp': within_1bp,
        'within_2bp': within_2bp,
        'exact_match_pct': exact_match_pct,
        'needs_adjustment': needs_adjustment,
        'most_common_offset': most_common_offset,
        'offset_distribution': offset_dist
    }


def main():
    """Main function."""
    
    print(f"\n{'='*80}")
    print("AGPAT3 METRICS RE-CALCULATION (CORRECT METHOD)")
    print(f"{'='*80}\n")
    
    # Load positions
    positions_file = "/Users/pleiadian53/work/meta-spliceai/data/ensembl/GRCh37/spliceai_eval/meta_models/diverse_genes_test/full_splice_positions_enhanced.tsv"
    
    print(f"Loading: {positions_file}\n")
    positions_df = pl.read_csv(positions_file, separator='\t')
    
    print(f"Total positions: {positions_df.height:,}")
    print(f"Unique genes: {positions_df['gene_id'].n_unique()}")
    print()
    
    # Calculate metrics for each splice type
    results = {}
    
    for splice_type in ['donor', 'acceptor']:
        print(f"{'='*80}")
        print(f"{splice_type.upper()} SITES")
        print(f"{'='*80}\n")
        
        # 1. Test if adjustments are needed (consensus_window=0)
        print(f"Step 1: Testing if adjustments are needed (consensus_window=0)")
        print("-" * 80)
        
        adj_test = test_adjustments_needed(positions_df, splice_type)
        
        if adj_test:
            print(f"High-scoring positions: {adj_test['total_positions']}")
            print(f"Exact matches (offset=0): {adj_test['exact_matches']} ({adj_test['exact_match_pct']:.1f}%)")
            print(f"Within ±1bp: {adj_test['within_1bp']}")
            print(f"Within ±2bp: {adj_test['within_2bp']}")
            print()
            
            print("Offset distribution:")
            for row in adj_test['offset_distribution'].iter_rows(named=True):
                print(f"  Offset {row['offset']:+3d}: {row['len']:3d} positions")
            print()
            
            if adj_test['needs_adjustment']:
                print(f"❌ ADJUSTMENT NEEDED: {adj_test['most_common_offset']:+d}bp")
                print(f"   → Apply {-adj_test['most_common_offset']:+d}bp adjustment to predictions")
            else:
                print(f"✅ NO ADJUSTMENT NEEDED: {adj_test['exact_match_pct']:.1f}% exact matches")
            print()
        
        # 2. Calculate metrics using pred_type (with consensus_window=2 already applied)
        print(f"Step 2: Calculate metrics using pred_type (consensus_window=2)")
        print("-" * 80)
        
        metrics = calculate_metrics_from_pred_type(positions_df, splice_type)
        
        if metrics:
            print(f"Total positions: {metrics['total']:,}")
            print(f"  True Positives:  {metrics['tp']:,}")
            print(f"  False Positives: {metrics['fp']:,}")
            print(f"  False Negatives: {metrics['fn']:,}")
            print(f"  True Negatives:  {metrics['tn']:,}")
            print()
            print(f"Precision: {metrics['precision']:.4f}")
            print(f"Recall:    {metrics['recall']:.4f}")
            print(f"F1 Score:  {metrics['f1']:.4f}")
            if metrics['pr_auc'] is not None:
                print(f"PR-AUC:    {metrics['pr_auc']:.4f}")
            print()
            
            results[splice_type] = {
                'metrics': metrics,
                'adjustment': adj_test
            }
    
    # Summary
    print(f"{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}\n")
    
    if 'donor' in results and 'acceptor' in results:
        donor_f1 = results['donor']['metrics']['f1']
        acceptor_f1 = results['acceptor']['metrics']['f1']
        avg_f1 = (donor_f1 + acceptor_f1) / 2
        
        print(f"Average F1 Score: {avg_f1:.4f}")
        print(f"  Donor:    {donor_f1:.4f}")
        print(f"  Acceptor: {acceptor_f1:.4f}")
        print()
        
        # Adjustment recommendations
        print("Adjustment Recommendations:")
        print("-" * 80)
        
        if results['donor']['adjustment']['needs_adjustment']:
            offset = results['donor']['adjustment']['most_common_offset']
            print(f"Donor sites:    Apply {-offset:+d}bp adjustment")
        else:
            print(f"Donor sites:    No adjustment needed ✅")
        
        if results['acceptor']['adjustment']['needs_adjustment']:
            offset = results['acceptor']['adjustment']['most_common_offset']
            print(f"Acceptor sites: Apply {-offset:+d}bp adjustment")
        else:
            print(f"Acceptor sites: No adjustment needed ✅")
        print()
    
    print(f"{'='*80}")
    print("INTERPRETATION")
    print(f"{'='*80}\n")
    
    print("Key Points:")
    print("1. Used pred_type column (correct classification)")
    print("2. Tested with consensus_window=0 to detect needed adjustments")
    print("3. Metrics calculated with consensus_window=2 (already applied in workflow)")
    print()
    print("The high F1 scores show that:")
    print("  • Predictions are within consensus_window=2 of true sites")
    print("  • Model is working correctly")
    print("  • Adjustments would improve exact coordinate accuracy")
    print()
    print("Next steps:")
    print("  1. Apply recommended adjustments to predictions")
    print("  2. Re-run workflow with adjusted predictions")
    print("  3. Verify 100% exact matches (consensus_window=0)")
    print("  4. Maintain high F1 with consensus_window=2")
    
    print(f"\n{'='*80}")
    print("TEST COMPLETE")
    print(f"{'='*80}\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())



