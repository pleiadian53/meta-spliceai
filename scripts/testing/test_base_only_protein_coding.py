#!/usr/bin/env python
"""
Test base-only mode on protein-coding genes to verify SpliceAI performance.

Expected: High F1 scores (>90%) for protein-coding genes, as SpliceAI was trained on them.
"""

import sys
from pathlib import Path
import polars as pl
import numpy as np
from meta_spliceai.splice_engine.meta_models.workflows.inference.enhanced_selective_inference import (
    EnhancedSelectiveInferenceConfig,
    EnhancedSelectiveInferenceWorkflow
)
from meta_spliceai.system.genomic_resources import Registry

def load_splice_sites(gene_id: str) -> pl.DataFrame:
    """Load annotated splice sites for a gene."""
    registry = Registry()
    # Registry automatically looks for enhanced version when kind='splice_sites'
    ss_path = registry.resolve('splice_sites')
    
    if not ss_path or not Path(ss_path).exists():
        print(f"❌ Splice sites file not found: {ss_path}")
        return pl.DataFrame()
    
    # Load and filter to target gene
    # Note: chrom column contains 'X', 'Y', etc., so read as string
    ss_df = pl.read_csv(
        ss_path, 
        separator='\t',
        schema_overrides={'chrom': pl.Utf8}  # Read chromosome as string
    )
    gene_ss = ss_df.filter(pl.col('gene_id') == gene_id)
    
    return gene_ss

def calculate_metrics(predictions: pl.DataFrame, annotations: pl.DataFrame) -> dict:
    """Calculate precision, recall, F1 for donor and acceptor sites."""
    
    # Get predicted splice sites (high confidence threshold)
    threshold = 0.5
    pred_donors = set(predictions.filter(pl.col('donor_score') > threshold)['position'].to_list())
    pred_acceptors = set(predictions.filter(pl.col('acceptor_score') > threshold)['position'].to_list())
    
    # Get annotated splice sites
    annot_donors = set(annotations.filter(pl.col('site_type') == 'donor')['position'].to_list())
    annot_acceptors = set(annotations.filter(pl.col('site_type') == 'acceptor')['position'].to_list())
    
    def calc_metrics(pred_set, annot_set, site_type):
        tp = len(pred_set & annot_set)
        fp = len(pred_set - annot_set)
        fn = len(annot_set - pred_set)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            f'{site_type}_tp': tp,
            f'{site_type}_fp': fp,
            f'{site_type}_fn': fn,
            f'{site_type}_precision': precision,
            f'{site_type}_recall': recall,
            f'{site_type}_f1': f1
        }
    
    donor_metrics = calc_metrics(pred_donors, annot_donors, 'donor')
    acceptor_metrics = calc_metrics(pred_acceptors, annot_acceptors, 'acceptor')
    
    # Overall metrics
    all_pred = pred_donors | pred_acceptors
    all_annot = annot_donors | annot_acceptors
    overall_metrics = calc_metrics(all_pred, all_annot, 'overall')
    
    return {**donor_metrics, **acceptor_metrics, **overall_metrics}

def test_gene(gene_id: str, gene_name: str = None) -> dict:
    """Test a single gene in base-only mode."""
    
    print(f"\n{'='*80}")
    print(f"Testing Gene: {gene_name or gene_id} ({gene_id})")
    print('='*80)
    
    # Configure for base-only mode
    config = EnhancedSelectiveInferenceConfig(
        model_path="results/meta_model_1000genes_3mers_fresh/model_multiclass.pkl",
        target_genes=[gene_id],
        inference_mode="base_only",
        uncertainty_threshold_low=0.02,
        uncertainty_threshold_high=0.50,
        output_name=None  # Regular output
    )
    
    # Run inference
    workflow = EnhancedSelectiveInferenceWorkflow(config)
    result = workflow.run_incremental()  # target_genes already in config
    
    if not result.success:
        print(f"❌ Workflow failed for {gene_id}")
        return {'gene_id': gene_id, 'success': False}
    
    # Load predictions
    output_path = Path(f"predictions/base_only/{gene_id}/combined_predictions.parquet")
    if not output_path.exists():
        print(f"❌ Output not found: {output_path}")
        return {'gene_id': gene_id, 'success': False}
    
    predictions = pl.read_parquet(output_path)
    
    # Load annotations
    annotations = load_splice_sites(gene_id)
    
    if annotations.height == 0:
        print(f"⚠️  No annotated splice sites found for {gene_id}")
        return {'gene_id': gene_id, 'success': False, 'error': 'no_annotations'}
    
    # Calculate metrics
    metrics = calculate_metrics(predictions, annotations)
    
    # Print results
    print(f"\n📊 Results:")
    print(f"  Total positions: {predictions.height:,}")
    print(f"  Annotated donors: {len(set(annotations.filter(pl.col('site_type') == 'donor')['position'].to_list()))}")
    print(f"  Annotated acceptors: {len(set(annotations.filter(pl.col('site_type') == 'acceptor')['position'].to_list()))}")
    print(f"\n  Donor Sites:")
    print(f"    TP: {metrics['donor_tp']}, FP: {metrics['donor_fp']}, FN: {metrics['donor_fn']}")
    print(f"    Precision: {metrics['donor_precision']:.3f}, Recall: {metrics['donor_recall']:.3f}, F1: {metrics['donor_f1']:.3f}")
    print(f"\n  Acceptor Sites:")
    print(f"    TP: {metrics['acceptor_tp']}, FP: {metrics['acceptor_fp']}, FN: {metrics['acceptor_fn']}")
    print(f"    Precision: {metrics['acceptor_precision']:.3f}, Recall: {metrics['acceptor_recall']:.3f}, F1: {metrics['acceptor_f1']:.3f}")
    print(f"\n  Overall:")
    print(f"    TP: {metrics['overall_tp']}, FP: {metrics['overall_fp']}, FN: {metrics['overall_fn']}")
    print(f"    Precision: {metrics['overall_precision']:.3f}, Recall: {metrics['overall_recall']:.3f}, F1: {metrics['overall_f1']:.3f}")
    
    # Status
    status = "✅ PASS" if metrics['overall_f1'] > 0.80 else "⚠️  LOW" if metrics['overall_f1'] > 0.50 else "❌ FAIL"
    print(f"\n  Status: {status}")
    
    return {
        'gene_id': gene_id,
        'gene_name': gene_name,
        'success': True,
        'positions': predictions.height,
        **metrics
    }

def main():
    """Test base-only mode on protein-coding genes."""
    
    print("="*80)
    print("BASE-ONLY MODE TEST: Protein-Coding Genes")
    print("="*80)
    print("Expected: High F1 scores (>90%) as SpliceAI was trained on protein-coding genes")
    print()
    
    # Select protein-coding genes with good splice site annotations
    test_genes = [
        ("ENSG00000134202", "GSTM3"),      # 7,107 bp, tested earlier
        ("ENSG00000157764", "BRAF"),       # Well-studied oncogene
        ("ENSG00000141510", "TP53"),       # Tumor suppressor
    ]
    
    results = []
    for gene_id, gene_name in test_genes:
        result = test_gene(gene_id, gene_name)
        results.append(result)
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print('='*80)
    
    successful = [r for r in results if r.get('success', False)]
    
    if not successful:
        print("❌ All tests failed")
        return 1
    
    print(f"\nResults ({len(successful)}/{len(test_genes)} genes):")
    print(f"{'Gene':<20} {'Positions':<12} {'Donor F1':<10} {'Acceptor F1':<12} {'Overall F1':<12} {'Status'}")
    print("-" * 80)
    
    for r in successful:
        gene_label = f"{r['gene_name'] or r['gene_id']}"
        status = "✅" if r.get('overall_f1', 0) > 0.80 else "⚠️" if r.get('overall_f1', 0) > 0.50 else "❌"
        print(f"{gene_label:<20} {r.get('positions', 0):<12,} "
              f"{r.get('donor_f1', 0):<10.3f} {r.get('acceptor_f1', 0):<12.3f} "
              f"{r.get('overall_f1', 0):<12.3f} {status}")
    
    # Average metrics
    if successful:
        avg_f1 = np.mean([r['overall_f1'] for r in successful])
        avg_precision = np.mean([r['overall_precision'] for r in successful])
        avg_recall = np.mean([r['overall_recall'] for r in successful])
        
        print(f"\nAverage Performance:")
        print(f"  Precision: {avg_precision:.3f}")
        print(f"  Recall: {avg_recall:.3f}")
        print(f"  F1 Score: {avg_f1:.3f}")
        
        if avg_f1 > 0.85:
            print(f"\n✅ EXCELLENT: Base model performance meets expectations!")
        elif avg_f1 > 0.70:
            print(f"\n⚠️  ACCEPTABLE: Base model performance is reasonable but could be better")
        else:
            print(f"\n❌ POOR: Base model performance is below expectations")
        
        return 0 if avg_f1 > 0.70 else 1
    else:
        print("\n❌ No successful tests")
        return 1

if __name__ == "__main__":
    sys.exit(main())

