#!/usr/bin/env python3
"""
Test all three inference modes (base_only, hybrid, meta_only) on the same genes.

Compares performance across modes to validate that meta-learning improves
predictions by reducing false positives and false negatives.

Success Criteria:
- Hybrid/meta_only should reduce FPs and FNs compared to base_only
- Should not introduce many new errors
- Overall F1 should improve or remain stable
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple
import polars as pl
from dataclasses import dataclass

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from meta_spliceai.splice_engine.meta_models.workflows.inference.enhanced_selective_inference import (
    EnhancedSelectiveInferenceWorkflow,
    EnhancedSelectiveInferenceConfig
)


@dataclass
class GeneMetrics:
    """Metrics for a single gene."""
    gene_id: str
    gene_name: str
    n_true: int
    n_pred: int
    tp: int
    fn: int
    fp: int
    recall: float
    precision: float
    f1: float
    true_positions: set
    pred_positions: set


@dataclass
class ModeComparison:
    """Comparison between two inference modes."""
    mode_a: str
    mode_b: str
    errors_fixed: int  # FPs or FNs in mode_a that are corrected in mode_b
    new_errors: int    # New FPs or FNs introduced by mode_b
    f1_improvement: float


def load_ground_truth() -> pl.DataFrame:
    """Load ground truth splice sites."""
    return pl.read_csv('data/ensembl/splice_sites_enhanced.tsv', 
                      separator='\t', 
                      schema_overrides={'chrom': pl.Utf8})


def get_predicted_sites(predictions_df: pl.DataFrame, 
                       score_col: str = 'donor_score',
                       threshold: float = 0.5) -> set:
    """Extract predicted splice site positions (with ±2bp tolerance)."""
    high_scores = predictions_df.filter(pl.col(score_col) >= threshold)
    return set(high_scores['absolute_position'].to_list())


def calculate_metrics(gene_id: str, 
                     predictions_df: pl.DataFrame,
                     ground_truth_df: pl.DataFrame,
                     tolerance: int = 2) -> GeneMetrics:
    """Calculate performance metrics for a single gene."""
    
    # Get gene info
    gene_name = predictions_df['gene_name'][0] if 'gene_name' in predictions_df.columns else 'N/A'
    
    # Get predictions with score >= 0.5
    pred_positions = get_predicted_sites(predictions_df, 'donor_score', 0.5)
    
    # Get ground truth
    gene_gt = ground_truth_df.filter(
        (pl.col('gene_id') == gene_id) & 
        (pl.col('site_type') == 'donor')
    )
    true_positions = set(gene_gt['position'].to_list())
    
    # Calculate matches with tolerance
    tp = 0
    matched_true = set()
    matched_pred = set()
    
    for true_pos in true_positions:
        for pred_pos in pred_positions:
            if abs(true_pos - pred_pos) <= tolerance:
                tp += 1
                matched_true.add(true_pos)
                matched_pred.add(pred_pos)
                break
    
    fn = len(true_positions) - len(matched_true)
    fp = len(pred_positions) - len(matched_pred)
    
    recall = (tp / len(true_positions) * 100) if len(true_positions) > 0 else 0
    precision = (tp / len(pred_positions) * 100) if len(pred_positions) > 0 else 0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0
    
    return GeneMetrics(
        gene_id=gene_id,
        gene_name=gene_name,
        n_true=len(true_positions),
        n_pred=len(pred_positions),
        tp=tp,
        fn=fn,
        fp=fp,
        recall=recall,
        precision=precision,
        f1=f1,
        true_positions=true_positions,
        pred_positions=pred_positions
    )


def compare_modes(metrics_a: GeneMetrics, metrics_b: GeneMetrics) -> Dict:
    """Compare two modes on the same gene."""
    
    # Errors fixed: FNs in mode_a that are TPs in mode_b
    fns_fixed = len(metrics_a.true_positions - metrics_a.pred_positions) - \
                len(metrics_b.true_positions - metrics_b.pred_positions)
    fns_fixed = max(0, fns_fixed)
    
    # Errors fixed: FPs in mode_a that are removed in mode_b
    fps_fixed = len(metrics_a.pred_positions - metrics_a.true_positions) - \
                len(metrics_b.pred_positions - metrics_b.true_positions)
    fps_fixed = max(0, fps_fixed)
    
    # New errors: FNs in mode_b that weren't in mode_a
    new_fns = len(metrics_b.true_positions - metrics_b.pred_positions) - \
              len(metrics_a.true_positions - metrics_a.pred_positions)
    new_fns = max(0, new_fns)
    
    # New errors: FPs in mode_b that weren't in mode_a
    new_fps = len(metrics_b.pred_positions - metrics_b.true_positions) - \
              len(metrics_a.pred_positions - metrics_a.true_positions)
    new_fps = max(0, new_fps)
    
    return {
        'fns_fixed': fns_fixed,
        'fps_fixed': fps_fixed,
        'new_fns': new_fns,
        'new_fps': new_fps,
        'f1_change': metrics_b.f1 - metrics_a.f1
    }


def run_inference_mode(genes: List[str], 
                      mode: str,
                      model_path: Path,
                      verbose: int = 1) -> Dict[str, pl.DataFrame]:
    """Run inference in specified mode and return predictions per gene."""
    
    print(f"\n{'='*80}")
    print(f"🧬 RUNNING INFERENCE MODE: {mode.upper()}")
    print(f"{'='*80}\n")
    
    config = EnhancedSelectiveInferenceConfig(
        model_path=model_path,
        target_genes=genes,
        inference_mode=mode,
        ensure_complete_coverage=True,
        enable_memory_monitoring=False,
        verbose=verbose
    )
    
    workflow = EnhancedSelectiveInferenceWorkflow(config)
    results = workflow.run()
    
    # Load per-gene predictions
    per_gene_dir = workflow.output_dir / "per_gene"
    predictions = {}
    
    for gene_file in per_gene_dir.glob("*.parquet"):
        gene_id = gene_file.stem.replace('_predictions', '')
        predictions[gene_id] = pl.read_parquet(gene_file)
    
    print(f"✅ Completed {mode} mode: {len(predictions)} genes processed\n")
    
    return predictions


def main():
    # Test genes (same as previous base-only tests)
    observed_genes = [
        "ENSG00000065413",  # GCNT4
        "ENSG00000134202",  # GSTM3
        "ENSG00000169239",  # CA5B
    ]
    
    unobserved_genes = [
        "ENSG00000255071",  # SAA2-SAA4
        "ENSG00000253250",  # C8orf88
        "ENSG00000006606",  # CCL26
    ]
    
    all_genes = observed_genes + unobserved_genes
    
    model_path = project_root / "results/gene_cv_1000_run_15/ablation_study/full/model_full.pkl"
    
    print("=" * 80)
    print("🧪 TESTING ALL THREE INFERENCE MODES")
    print("=" * 80)
    print()
    print("Test Genes:")
    print("  Observed (in training):   ", len(observed_genes))
    print("  Unobserved (not in training):", len(unobserved_genes))
    print("  Total:", len(all_genes))
    print()
    print("Modes to test:")
    print("  1. base_only  - SpliceAI predictions only")
    print("  2. hybrid     - Meta-model refinement on uncertain positions")
    print("  3. meta_only  - Pure meta-model predictions")
    print()
    
    # Load ground truth
    ground_truth = load_ground_truth()
    
    # Run all three modes
    modes = ['base_only', 'hybrid', 'meta_only']
    all_predictions = {}
    all_metrics = {}
    
    for mode in modes:
        all_predictions[mode] = run_inference_mode(all_genes, mode, model_path, verbose=1)
        
        # Calculate metrics for each gene
        all_metrics[mode] = {}
        for gene_id in all_genes:
            if gene_id in all_predictions[mode]:
                all_metrics[mode][gene_id] = calculate_metrics(
                    gene_id, 
                    all_predictions[mode][gene_id],
                    ground_truth
                )
    
    # Display results
    print("\n" + "=" * 80)
    print("📊 PERFORMANCE COMPARISON ACROSS MODES")
    print("=" * 80)
    
    # Per-gene comparison
    print("\n" + "-" * 80)
    print("PER-GENE RESULTS")
    print("-" * 80)
    
    for gene_id in all_genes:
        metrics_base = all_metrics['base_only'].get(gene_id)
        metrics_hybrid = all_metrics['hybrid'].get(gene_id)
        metrics_meta = all_metrics['meta_only'].get(gene_id)
        
        if not all([metrics_base, metrics_hybrid, metrics_meta]):
            continue
        
        gene_type = "OBS" if gene_id in observed_genes else "UNO"
        
        print(f"\n{metrics_base.gene_name} ({gene_id}) [{gene_type}]:")
        print(f"  Ground truth: {metrics_base.n_true} donor sites")
        print()
        print(f"  MODE           F1      Recall  Prec    TP   FN   FP")
        print(f"  ------------ ------ -------- ------ ---- ---- ----")
        
        for mode in modes:
            m = all_metrics[mode][gene_id]
            print(f"  {mode:12} {m.f1:6.1f}% {m.recall:6.1f}% {m.precision:6.1f}% "
                  f"{m.tp:4} {m.fn:4} {m.fp:4}")
        
        # Show improvements
        base_vs_hybrid = compare_modes(metrics_base, metrics_hybrid)
        base_vs_meta = compare_modes(metrics_base, metrics_meta)
        
        print()
        print(f"  Hybrid vs Base-only:")
        print(f"    FNs fixed: {base_vs_hybrid['fns_fixed']}, "
              f"FPs fixed: {base_vs_hybrid['fps_fixed']}, "
              f"New FNs: {base_vs_hybrid['new_fns']}, "
              f"New FPs: {base_vs_hybrid['new_fps']}, "
              f"F1 Δ: {base_vs_hybrid['f1_change']:+.1f}%")
        
        print(f"  Meta-only vs Base-only:")
        print(f"    FNs fixed: {base_vs_meta['fns_fixed']}, "
              f"FPs fixed: {base_vs_meta['fps_fixed']}, "
              f"New FNs: {base_vs_meta['new_fns']}, "
              f"New FPs: {base_vs_meta['new_fps']}, "
              f"F1 Δ: {base_vs_meta['f1_change']:+.1f}%")
    
    # Aggregate statistics
    print("\n" + "=" * 80)
    print("📈 AGGREGATE STATISTICS")
    print("=" * 80)
    
    for gene_category, genes_list in [("OBSERVED", observed_genes), 
                                      ("UNOBSERVED", unobserved_genes),
                                      ("OVERALL", all_genes)]:
        print(f"\n{gene_category} GENES (n={len(genes_list)}):")
        print(f"  {'MODE':12} {'F1':>6}  {'Recall':>7}  {'Prec':>7}  {'TP':>4}  {'FN':>4}  {'FP':>4}")
        print(f"  {'-'*12} {'-'*6}  {'-'*7}  {'-'*7}  {'-'*4}  {'-'*4}  {'-'*4}")
        
        for mode in modes:
            metrics_list = [all_metrics[mode][g] for g in genes_list if g in all_metrics[mode]]
            
            if not metrics_list:
                continue
            
            avg_f1 = sum(m.f1 for m in metrics_list) / len(metrics_list)
            avg_recall = sum(m.recall for m in metrics_list) / len(metrics_list)
            avg_precision = sum(m.precision for m in metrics_list) / len(metrics_list)
            total_tp = sum(m.tp for m in metrics_list)
            total_fn = sum(m.fn for m in metrics_list)
            total_fp = sum(m.fp for m in metrics_list)
            
            print(f"  {mode:12} {avg_f1:6.1f}% {avg_recall:7.1f}% {avg_precision:7.1f}% "
                  f"{total_tp:4} {total_fn:4} {total_fp:4}")
    
    # Summary of improvements
    print("\n" + "=" * 80)
    print("🎯 META-LEARNING IMPROVEMENT SUMMARY")
    print("=" * 80)
    
    for gene_category, genes_list in [("OBSERVED", observed_genes), 
                                      ("UNOBSERVED", unobserved_genes),
                                      ("OVERALL", all_genes)]:
        
        base_metrics = [all_metrics['base_only'][g] for g in genes_list if g in all_metrics['base_only']]
        hybrid_metrics = [all_metrics['hybrid'][g] for g in genes_list if g in all_metrics['hybrid']]
        meta_metrics = [all_metrics['meta_only'][g] for g in genes_list if g in all_metrics['meta_only']]
        
        if not all([base_metrics, hybrid_metrics, meta_metrics]):
            continue
        
        base_f1 = sum(m.f1 for m in base_metrics) / len(base_metrics)
        hybrid_f1 = sum(m.f1 for m in hybrid_metrics) / len(hybrid_metrics)
        meta_f1 = sum(m.f1 for m in meta_metrics) / len(meta_metrics)
        
        print(f"\n{gene_category}:")
        print(f"  Base-only F1:  {base_f1:6.1f}%")
        print(f"  Hybrid F1:     {hybrid_f1:6.1f}%  (Δ: {hybrid_f1-base_f1:+.1f}%)")
        print(f"  Meta-only F1:  {meta_f1:6.1f}%  (Δ: {meta_f1-base_f1:+.1f}%)")
    
    print("\n" + "=" * 80)
    print("✅ TESTING COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()

