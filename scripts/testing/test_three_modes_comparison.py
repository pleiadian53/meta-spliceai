#!/usr/bin/env python3
"""
Compare all 3 inference modes on known good genes.

This test:
1. Runs all 3 modes (base-only, hybrid, meta-only) on GSTM3
2. Compares splice site predictions across modes
3. Verifies meta-model is actually changing scores
4. Reports performance differences

Created: 2025-10-29
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


def run_mode(gene_id: str, mode: str, model_path: Path) -> tuple:
    """Run inference in specified mode and return predictions."""
    
    config = EnhancedSelectiveInferenceConfig(
        target_genes=[gene_id],
        model_path=model_path,
        inference_mode=mode,
        output_name=None,
        uncertainty_threshold_low=0.02,
        uncertainty_threshold_high=0.50,
        use_timestamped_output=False,
        verbose=0
    )
    
    workflow = EnhancedSelectiveInferenceWorkflow(config)
    result = workflow.run_incremental()
    
    if not result.success:
        return None, f"Failed: {result.message}"
    
    # Get predictions
    gene_paths = workflow.output_manager.get_gene_output_paths(gene_id)
    pred_file = gene_paths.predictions_file
    
    if not pred_file.exists():
        return None, f"File not found: {pred_file}"
    
    df = pl.read_parquet(pred_file)
    return df, None


def compare_modes():
    """Compare all 3 modes on GSTM3."""
    
    print("\n" + "="*80)
    print("THREE-MODE COMPARISON TEST")
    print("="*80)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    gene_id = 'ENSG00000134202'  # GSTM3
    model_path = project_root / 'results/meta_model_1000genes_3mers_fresh/model_multiclass.pkl'
    
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return 1
    
    print(f"Gene: {gene_id} (GSTM3)")
    print(f"Model: {model_path.name}\n")
    
    # Run all 3 modes
    modes = ['base_only', 'hybrid', 'meta_only']
    predictions = {}
    
    for mode in modes:
        print(f"Running {mode.upper().replace('_', '-')} mode...", end=' ')
        df, error = run_mode(gene_id, mode, model_path)
        
        if error:
            print(f"❌ {error}")
            return 1
        
        predictions[mode] = df
        print(f"✅ ({len(df):,} positions)")
    
    # Compare predictions
    print(f"\n{'='*80}")
    print("SCORE COMPARISON")
    print(f"{'='*80}")
    
    # Get common positions
    base_df = predictions['base_only']
    hybrid_df = predictions['hybrid']
    meta_df = predictions['meta_only']
    
    # Sort all by position for comparison
    base_df = base_df.sort('position')
    hybrid_df = hybrid_df.sort('position')
    meta_df = meta_df.sort('position')
    
    # Check if positions match
    base_pos = base_df['position'].to_list()
    hybrid_pos = hybrid_df['position'].to_list()
    meta_pos = meta_df['position'].to_list()
    
    if base_pos != hybrid_pos or base_pos != meta_pos:
        print("⚠️  WARNING: Position mismatch across modes!")
        print(f"   Base: {len(base_pos)} positions")
        print(f"   Hybrid: {len(hybrid_pos)} positions")
        print(f"   Meta: {len(meta_pos)} positions")
    else:
        print(f"✅ All modes have {len(base_pos):,} positions")
    
    # Compare base scores (should be identical)
    base_donor = base_df['donor_score'].to_numpy()
    hybrid_donor = hybrid_df['donor_score'].to_numpy()
    meta_donor = meta_df['donor_score'].to_numpy()
    
    base_identical = np.allclose(base_donor, hybrid_donor, rtol=1e-5) and \
                     np.allclose(base_donor, meta_donor, rtol=1e-5)
    
    print(f"\nBase scores identical across modes: {'✅ YES' if base_identical else '❌ NO'}")
    
    # Compare meta scores (should differ for hybrid/meta)
    base_donor_meta = base_df['donor_meta'].to_numpy()
    hybrid_donor_meta = hybrid_df['donor_meta'].to_numpy()
    meta_donor_meta = meta_df['donor_meta'].to_numpy()
    
    base_vs_hybrid_same = np.allclose(base_donor_meta, hybrid_donor_meta, rtol=1e-5)
    base_vs_meta_same = np.allclose(base_donor_meta, meta_donor_meta, rtol=1e-5)
    hybrid_vs_meta_same = np.allclose(hybrid_donor_meta, meta_donor_meta, rtol=1e-5)
    
    print(f"\nMeta scores:")
    print(f"  Base vs Hybrid: {'⚠️  IDENTICAL' if base_vs_hybrid_same else '✅ DIFFERENT'}")
    print(f"  Base vs Meta: {'⚠️  IDENTICAL' if base_vs_meta_same else '✅ DIFFERENT'}")
    print(f"  Hybrid vs Meta: {'⚠️  IDENTICAL' if hybrid_vs_meta_same else '✅ DIFFERENT'}")
    
    # Count differences
    if not base_vs_hybrid_same:
        diff_count = np.sum(~np.isclose(base_donor_meta, hybrid_donor_meta, rtol=1e-5))
        diff_pct = (diff_count / len(base_donor_meta)) * 100
        print(f"     Hybrid changed {diff_count:,} positions ({diff_pct:.1f}%)")
    
    if not base_vs_meta_same:
        diff_count = np.sum(~np.isclose(base_donor_meta, meta_donor_meta, rtol=1e-5))
        diff_pct = (diff_count / len(base_donor_meta)) * 100
        print(f"     Meta changed {diff_count:,} positions ({diff_pct:.1f}%)")
    
    # High-confidence predictions
    threshold = 0.5
    
    print(f"\n{'='*80}")
    print(f"HIGH-CONFIDENCE PREDICTIONS (threshold={threshold})")
    print(f"{'='*80}")
    
    for mode in modes:
        df = predictions[mode]
        n_donor = len(df.filter(pl.col('donor_meta') >= threshold))
        n_acceptor = len(df.filter(pl.col('acceptor_meta') >= threshold))
        print(f"{mode:12s}: {n_donor:3d} donors, {n_acceptor:3d} acceptors")
    
    # Metadata check
    print(f"\n{'='*80}")
    print("METADATA FEATURES")
    print(f"{'='*80}")
    
    metadata_cols = [
        'is_uncertain', 'is_low_confidence', 'is_high_entropy',
        'is_low_discriminability', 'max_confidence', 'score_spread',
        'score_entropy', 'confidence_category', 'predicted_type_base'
    ]
    
    for mode in modes:
        df = predictions[mode]
        present = sum(1 for col in metadata_cols if col in df.columns)
        status = "✅" if present == 9 else "❌"
        print(f"{mode:12s}: {status} {present}/9 features")
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    
    if base_identical:
        print("✅ Base scores consistent across modes")
    else:
        print("❌ Base scores inconsistent (BUG!)")
    
    if base_vs_meta_same:
        print("❌ Meta-model NOT recalibrating scores (BUG!)")
    else:
        print("✅ Meta-model successfully recalibrating scores")
    
    if hybrid_vs_meta_same:
        print("⚠️  Hybrid and meta-only produce identical results")
    else:
        print("✅ Hybrid and meta-only produce different results")
    
    print(f"\n{'='*80}")
    print("✅ TEST COMPLETE")
    print(f"{'='*80}\n")
    
    return 0


if __name__ == '__main__':
    sys.exit(compare_modes())

