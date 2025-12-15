#!/usr/bin/env python3
"""
Test enriched inference workflow on high-complexity genes (50+ splice sites).

Characterizes SpliceAI performance across complexity spectrum.
"""

import sys
from pathlib import Path
import polars as pl

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from meta_spliceai.splice_engine.meta_models.workflows.inference.enhanced_selective_inference import (
    EnhancedSelectiveInferenceWorkflow,
    EnhancedSelectiveInferenceConfig
)


def validate_gene_predictions(gene_id: str, predictions_df: pl.DataFrame):
    """Validate predictions against ground truth."""
    
    # Load ground truth
    ss_df = pl.read_csv('data/ensembl/splice_sites_enhanced.tsv', 
                        separator='\t', 
                        schema_overrides={'chrom': pl.Utf8})
    
    gene_ss = ss_df.filter(pl.col('gene_id') == gene_id)
    
    if len(gene_ss) == 0:
        return None
    
    gene_name = predictions_df['gene_name'][0] if 'gene_name' in predictions_df.columns else 'N/A'
    
    # Get predictions with score >= 0.5
    donors_pred = predictions_df.filter(pl.col('donor_score') >= 0.5)
    acceptors_pred = predictions_df.filter(pl.col('acceptor_score') >= 0.5)
    
    # Get ground truth
    donors_true = gene_ss.filter(pl.col('site_type') == 'donor')
    acceptors_true = gene_ss.filter(pl.col('site_type') == 'acceptor')
    
    # Calculate metrics for donors (±2bp tolerance)
    pred_positions = set(donors_pred['absolute_position'].to_list())
    true_positions = set(donors_true['position'].to_list())
    
    matches = sum(1 for tp in true_positions if any(abs(tp - pp) <= 2 for pp in pred_positions))
    
    recall = (matches / len(true_positions) * 100) if len(true_positions) > 0 else 0
    precision = (matches / len(pred_positions) * 100) if len(pred_positions) > 0 else 0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0
    
    return {
        'gene_id': gene_id,
        'gene_name': gene_name,
        'n_true': len(true_positions),
        'n_pred': len(pred_positions),
        'tp': matches,
        'fn': len(true_positions) - matches,
        'fp': len(pred_positions) - matches,
        'recall': recall,
        'precision': precision,
        'f1': f1
    }


def main():
    # High-complexity test genes (50 splice sites each)
    high_complexity_genes = [
        "ENSG00000135318",  # NT5E (46kb, 50 sites)
        "ENSG00000169288",  # MRPL1 (90kb, 50 sites)
        "ENSG00000169876",  # MUC17 (39kb, 50 sites)
    ]
    
    model_path = project_root / "results/gene_cv_1000_run_15/ablation_study/full/model_full.pkl"
    
    print("=" * 80)
    print("🧬 TESTING HIGH-COMPLEXITY GENES (50 splice sites each)")
    print("=" * 80)
    print()
    
    config = EnhancedSelectiveInferenceConfig(
        model_path=model_path,
        target_genes=high_complexity_genes,
        inference_mode='base_only',
        ensure_complete_coverage=True,
        enable_memory_monitoring=True,
        max_memory_gb=8.0,
        verbose=1
    )
    
    workflow = EnhancedSelectiveInferenceWorkflow(config)
    results_obj = workflow.run()
    
    # Validate genes
    results = []
    per_gene_dir = workflow.output_dir / "per_gene"
    
    for gene_file in per_gene_dir.glob("*.parquet"):
        gene_id = gene_file.stem.replace('_predictions', '')
        df = pl.read_parquet(gene_file)
        metrics = validate_gene_predictions(gene_id, df)
        if metrics:
            results.append(metrics)
            print(f"  ✅ {metrics['gene_name']} ({gene_id}): "
                  f"Recall={metrics['recall']:.1f}%, Precision={metrics['precision']:.1f}%, F1={metrics['f1']:.1f}%")
            print(f"     ({metrics['tp']}/{metrics['n_true']} donors found, {metrics['fp']} false positives)")
    
    print()
    print("=" * 80)
    print("📊 HIGH-COMPLEXITY SUMMARY RESULTS")
    print("=" * 80)
    print()
    
    if results:
        avg_recall = sum(r['recall'] for r in results) / len(results)
        avg_precision = sum(r['precision'] for r in results) / len(results)
        avg_f1 = sum(r['f1'] for r in results) / len(results)
        
        total_tp = sum(r['tp'] for r in results)
        total_fn = sum(r['fn'] for r in results)
        total_fp = sum(r['fp'] for r in results)
        total_true = sum(r['n_true'] for r in results)
        
        print(f"HIGH-COMPLEXITY GENES (50 sites each, n={len(results)}):")
        print(f"  Average Recall:    {avg_recall:5.1f}%")
        print(f"  Average Precision: {avg_precision:5.1f}%")
        print(f"  Average F1:        {avg_f1:5.1f}%")
        print()
        print(f"  Total True Positives:  {total_tp}/{total_true} donors")
        print(f"  Total False Negatives: {total_fn}")
        print(f"  Total False Positives: {total_fp}")
        print()
        
        # Show individual results
        print("Individual gene results:")
        for r in results:
            print(f"  {r['gene_name']:15} | F1: {r['f1']:5.1f}% | "
                  f"Recall: {r['recall']:5.1f}% | Precision: {r['precision']:5.1f}%")
    
    print()
    print("=" * 80)


if __name__ == "__main__":
    main()

