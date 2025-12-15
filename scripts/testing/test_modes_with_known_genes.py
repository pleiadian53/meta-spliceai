#!/usr/bin/env python3
"""
Test all 3 modes on KNOWN GOOD genes with verified splice sites.

These genes were selected because:
1. They have many annotated splice sites
2. They showed good performance in previous tests
3. They are well-studied genes

Created: 2025-10-28
"""

import sys
from pathlib import Path
import polars as pl
import numpy as np
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from meta_spliceai.splice_engine.meta_models.workflows.inference.enhanced_selective_inference import (
    EnhancedSelectiveInferenceWorkflow,
    EnhancedSelectiveInferenceConfig
)


# Known good genes with verified splice sites
KNOWN_GOOD_GENES = {
    'ENSG00000134202': {
        'name': 'GSTM3',
        'type': 'protein_coding',
        'length': 7107,
        'expected_sites': 16,
        'notes': 'Previously tested, F1 ~90%'
    },
    'ENSG00000141736': {
        'name': 'ERBB2',
        'type': 'protein_coding',
        'length': 37000,  # approximate
        'expected_sites': 528,
        'notes': 'Large gene, many splice sites'
    },
    'ENSG00000169239': {
        'name': 'CA5B',
        'type': 'protein_coding',
        'length': 20000,  # approximate
        'expected_sites': 46,
        'notes': 'Well-annotated'
    }
}


def run_quick_test():
    """Run quick validation test on known good genes."""
    
    print("\n" + "="*80)
    print("QUICK VALIDATION TEST: Known Good Genes")
    print("="*80)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Select one gene for quick test
    gene_id = 'ENSG00000134202'  # GSTM3 - known to work well
    gene_info = KNOWN_GOOD_GENES[gene_id]
    
    print(f"\nTest Gene: {gene_id} ({gene_info['name']})")
    print(f"  Type: {gene_info['type']}")
    print(f"  Length: {gene_info['length']:,} bp")
    print(f"  Expected sites: {gene_info['expected_sites']}")
    print(f"  Notes: {gene_info['notes']}")
    
    model_path = project_root / 'results/meta_model_1000genes_3mers_fresh/model_multiclass.pkl'
    
    if not model_path.exists():
        print(f"\n❌ Model not found: {model_path}")
        return 1
    
    # Test all 3 modes
    modes = ['base_only', 'hybrid', 'meta_only']
    results = {}
    
    for mode in modes:
        print(f"\n{'='*80}")
        print(f"Testing {mode.upper().replace('_', '-')} mode")
        print(f"{'='*80}")
        
        try:
            config = EnhancedSelectiveInferenceConfig(
                target_genes=[gene_id],
                model_path=model_path,
                inference_mode=mode,
                output_name='quick_validation',
                uncertainty_threshold_low=0.02,
                uncertainty_threshold_high=0.50,
                use_timestamped_output=False,
                verbose=0
            )
            
            workflow = EnhancedSelectiveInferenceWorkflow(config)
            result = workflow.run_incremental()
            
            if result.success:
                # Get output
                gene_paths = workflow.output_manager.get_gene_output_paths(gene_id)
                pred_file = gene_paths.predictions_file
                
                if pred_file.exists():
                    df = pl.read_parquet(pred_file)
                    
                    # Calculate metrics with LOWER threshold (0.3)
                    threshold = 0.3
                    
                    # Load splice sites
                    ss_path = project_root / 'data/ensembl/splice_sites_enhanced.tsv'
                    ss_df = pl.read_csv(ss_path, separator='\t', schema_overrides={'chrom': pl.String})
                    gene_sites = ss_df.filter(pl.col('gene_id') == gene_id)
                    
                    # True sites
                    true_donors = set(gene_sites.filter(pl.col('site_type') == 'donor')['position'].to_list())
                    true_acceptors = set(gene_sites.filter(pl.col('site_type') == 'acceptor')['position'].to_list())
                    
                    # Predicted sites
                    pred_donors = set(df.filter(pl.col('donor_meta') >= threshold)['position'].to_list())
                    pred_acceptors = set(df.filter(pl.col('acceptor_meta') >= threshold)['position'].to_list())
                    
                    # Metrics
                    donor_tp = len(true_donors & pred_donors)
                    donor_fp = len(pred_donors - true_donors)
                    donor_fn = len(true_donors - pred_donors)
                    
                    acceptor_tp = len(true_acceptors & pred_acceptors)
                    acceptor_fp = len(pred_acceptors - true_acceptors)
                    acceptor_fn = len(true_acceptors - pred_acceptors)
                    
                    overall_tp = donor_tp + acceptor_tp
                    overall_fp = donor_fp + acceptor_fp
                    overall_fn = donor_fn + acceptor_fn
                    
                    precision = overall_tp / (overall_tp + overall_fp) if (overall_tp + overall_fp) > 0 else 0
                    recall = overall_tp / (overall_tp + overall_fn) if (overall_tp + overall_fn) > 0 else 0
                    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                    
                    results[mode] = {
                        'f1': f1,
                        'precision': precision,
                        'recall': recall,
                        'tp': overall_tp,
                        'fp': overall_fp,
                        'fn': overall_fn
                    }
                    
                    print(f"  ✅ SUCCESS")
                    print(f"     Positions: {len(df):,}")
                    print(f"     Threshold: {threshold}")
                    print(f"     TP: {overall_tp}, FP: {overall_fp}, FN: {overall_fn}")
                    print(f"     Precision: {precision:.3f}")
                    print(f"     Recall: {recall:.3f}")
                    print(f"     F1: {f1:.3f}")
                else:
                    print(f"  ❌ Output file not found")
            else:
                print(f"  ❌ Workflow failed")
                
        except Exception as e:
            print(f"  ❌ Exception: {e}")
    
    # Summary
    print(f"\n{'='*80}")
    print("PERFORMANCE COMPARISON")
    print(f"{'='*80}")
    
    if len(results) == 3:
        print(f"\n{gene_info['name']} ({gene_id}):")
        for mode in modes:
            if mode in results:
                f1 = results[mode]['f1']
                print(f"  {mode:12s}: F1 = {f1:.3f}")
        
        # Check if meta-model helps
        base_f1 = results['base_only']['f1']
        hybrid_f1 = results['hybrid']['f1']
        meta_f1 = results['meta_only']['f1']
        
        print(f"\nImprovement:")
        print(f"  Hybrid vs Base: {(hybrid_f1 - base_f1):.3f}")
        print(f"  Meta vs Base: {(meta_f1 - base_f1):.3f}")
        
        if meta_f1 > base_f1:
            print(f"\n✅ Meta-model IMPROVES performance!")
        elif meta_f1 == base_f1:
            print(f"\n⚠️  Meta-model shows NO improvement")
        else:
            print(f"\n❌ Meta-model DEGRADES performance")
    
    return 0


if __name__ == '__main__':
    sys.exit(run_quick_test())

