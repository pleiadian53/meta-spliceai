#!/usr/bin/env python3
"""
Test enriched inference workflow on observed vs unobserved genes.

Tests base-only mode on:
1. Observed genes (in training - predicts unseen positions)
2. Unobserved genes (not in training - predicts all positions)
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


def validate_gene_predictions(gene_id: str, predictions_df: pl.DataFrame, mode: str):
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
        'mode': mode,
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
    # Test genes
    observed_genes = [
        "ENSG00000182108",  # DEXI (10 sites)
        "ENSG00000165997",  # ARL5B (10 sites)
        "ENSG00000250641",  # LY6G6F-LY6G6D (10 sites)
    ]
    
    unobserved_genes = [
        "ENSG00000255071",  # SAA2-SAA4 (10 sites)
        "ENSG00000006606",  # CCL26 (10 sites)
        "ENSG00000253250",  # C8orf88 (10 sites)
    ]
    
    model_path = project_root / "results/gene_cv_1000_run_15/ablation_study/full/model_full.pkl"
    
    results = []
    
    print("=" * 80)
    print("🧬 TESTING OBSERVED VS. UNOBSERVED GENES (BASE-ONLY MODE)")
    print("=" * 80)
    print()
    
    # Test observed genes
    print("📋 Testing OBSERVED genes (in training, predict unseen positions)...")
    print()
    
    config_observed = EnhancedSelectiveInferenceConfig(
        model_path=model_path,
        target_genes=observed_genes,
        inference_mode='base_only',
        ensure_complete_coverage=True,
        enable_memory_monitoring=True,
        max_memory_gb=8.0,
        verbose=1
    )
    
    workflow_observed = EnhancedSelectiveInferenceWorkflow(config_observed)
    results_observed = workflow_observed.run()
    
    # Validate observed genes
    per_gene_dir = workflow_observed.output_dir / "per_gene"
    for gene_file in per_gene_dir.glob("*.parquet"):
        gene_id = gene_file.stem.replace('_predictions', '')
        df = pl.read_parquet(gene_file)
        metrics = validate_gene_predictions(gene_id, df, 'observed')
        if metrics:
            results.append(metrics)
            print(f"  ✅ {metrics['gene_name']} ({gene_id}): "
                  f"Recall={metrics['recall']:.1f}%, Precision={metrics['precision']:.1f}%, F1={metrics['f1']:.1f}%")
    
    print()
    
    # Test unobserved genes
    print("📋 Testing UNOBSERVED genes (not in training, predict all positions)...")
    print()
    
    config_unobserved = EnhancedSelectiveInferenceConfig(
        model_path=model_path,
        target_genes=unobserved_genes,
        inference_mode='base_only',
        ensure_complete_coverage=True,
        enable_memory_monitoring=True,
        max_memory_gb=8.0,
        verbose=1
    )
    
    workflow_unobserved = EnhancedSelectiveInferenceWorkflow(config_unobserved)
    results_unobserved = workflow_unobserved.run()
    
    # Validate unobserved genes
    per_gene_dir = workflow_unobserved.output_dir / "per_gene"
    for gene_file in per_gene_dir.glob("*.parquet"):
        gene_id = gene_file.stem.replace('_predictions', '')
        df = pl.read_parquet(gene_file)
        metrics = validate_gene_predictions(gene_id, df, 'unobserved')
        if metrics:
            results.append(metrics)
            print(f"  ✅ {metrics['gene_name']} ({gene_id}): "
                  f"Recall={metrics['recall']:.1f}%, Precision={metrics['precision']:.1f}%, F1={metrics['f1']:.1f}%")
    
    print()
    print("=" * 80)
    print("📊 SUMMARY RESULTS")
    print("=" * 80)
    print()
    
    # Calculate averages
    observed_results = [r for r in results if r['mode'] == 'observed']
    unobserved_results = [r for r in results if r['mode'] == 'unobserved']
    
    if observed_results:
        avg_recall_obs = sum(r['recall'] for r in observed_results) / len(observed_results)
        avg_precision_obs = sum(r['precision'] for r in observed_results) / len(observed_results)
        avg_f1_obs = sum(r['f1'] for r in observed_results) / len(observed_results)
        
        print(f"OBSERVED GENES (n={len(observed_results)}):")
        print(f"  Average Recall:    {avg_recall_obs:5.1f}%")
        print(f"  Average Precision: {avg_precision_obs:5.1f}%")
        print(f"  Average F1:        {avg_f1_obs:5.1f}%")
        print()
    
    if unobserved_results:
        avg_recall_unobs = sum(r['recall'] for r in unobserved_results) / len(unobserved_results)
        avg_precision_unobs = sum(r['precision'] for r in unobserved_results) / len(unobserved_results)
        avg_f1_unobs = sum(r['f1'] for r in unobserved_results) / len(unobserved_results)
        
        print(f"UNOBSERVED GENES (n={len(unobserved_results)}):")
        print(f"  Average Recall:    {avg_recall_unobs:5.1f}%")
        print(f"  Average Precision: {avg_precision_unobs:5.1f}%")
        print(f"  Average F1:        {avg_f1_unobs:5.1f}%")
        print()
    
    # Overall
    if results:
        overall_recall = sum(r['recall'] for r in results) / len(results)
        overall_precision = sum(r['precision'] for r in results) / len(results)
        overall_f1 = sum(r['f1'] for r in results) / len(results)
        
        print(f"OVERALL (n={len(results)}):")
        print(f"  Average Recall:    {overall_recall:5.1f}%")
        print(f"  Average Precision: {overall_precision:5.1f}%")
        print(f"  Average F1:        {overall_f1:5.1f}%")
    
    print()
    print("=" * 80)


if __name__ == "__main__":
    main()

