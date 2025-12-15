#!/usr/bin/env python
"""
Test all 3 inference modes on GSTM3 (ENSG00000134202) to verify full coverage.

Expected outcome:
- All 3 modes produce exactly 7,107 rows (one per nucleotide position)
- Base scores are identical across modes
- Meta scores differ between modes:
  - base_only: meta = base (0% meta-model usage)
  - hybrid: meta ≈ base with selective adjustments (2-10% meta-model usage)
  - meta_only: meta ≠ base for all positions (100% meta-model usage)
"""

import sys
from pathlib import Path
import polars as pl
from meta_spliceai.splice_engine.meta_models.workflows.inference.enhanced_selective_inference import (
    EnhancedSelectiveInferenceConfig,
    EnhancedSelectiveInferenceWorkflow
)

def test_mode(mode: str, gene_id: str = "ENSG00000134202") -> dict:
    """Test a single inference mode."""
    print(f"\n{'='*80}")
    print(f"Testing {mode.upper()} mode")
    print('='*80)
    
    config = EnhancedSelectiveInferenceConfig(
        model_path="results/meta_model_1000genes_3mers_fresh/model_multiclass.pkl",
        inference_mode=mode,
        uncertainty_threshold_low=0.3,
        uncertainty_threshold_high=0.7,
        entropy_high_threshold=0.5,
        spread_low_threshold=0.3,
        output_name=None  # Regular output, not test
    )
    
    workflow = EnhancedSelectiveInferenceWorkflow(config)
    result = workflow.run_incremental(target_genes=[gene_id])
    
    if result.success:
        # Load output
        output_path = Path(f"predictions/{mode}/{gene_id}/combined_predictions.parquet")
        df = pl.read_parquet(output_path)
        
        return {
            'mode': mode,
            'success': True,
            'total_rows': df.height,
            'unique_positions': df['position'].n_unique(),
            'has_metadata': all(col in df.columns for col in [
                'is_uncertain', 'max_confidence', 'score_entropy'
            ]),
            'meta_model_usage': result.meta_model_positions,
            'meta_model_pct': (result.meta_model_positions / df.height * 100) if df.height > 0 else 0
        }
    else:
        return {
            'mode': mode,
            'success': False,
            'error': str(result)
        }

def compare_scores(gene_id: str = "ENSG00000134202"):
    """Compare scores across all 3 modes."""
    print(f"\n{'='*80}")
    print("COMPARING SCORES ACROSS MODES")
    print('='*80)
    
    modes = ['base_only', 'hybrid', 'meta_only']
    dfs = {}
    
    for mode in modes:
        path = Path(f"predictions/{mode}/{gene_id}/combined_predictions.parquet")
        if path.exists():
            dfs[mode] = pl.read_parquet(path).sort('position')
        else:
            print(f"❌ Missing output for {mode}")
            return
    
    # Check dimensions
    print("\nDimensions:")
    for mode, df in dfs.items():
        print(f"  {mode:15s}: {df.height:,} rows × {len(df.columns)} columns")
    
    # Check if positions match
    positions = {mode: set(df['position'].to_list()) for mode, df in dfs.items()}
    all_same = len(set(map(len, positions.values()))) == 1 and \
               all(positions['base_only'] == positions[mode] for mode in modes[1:])
    
    if all_same:
        print(f"✅ All modes have identical position sets ({len(positions['base_only']):,} positions)")
    else:
        print("❌ Position sets differ:")
        for mode in modes:
            print(f"  {mode}: {len(positions[mode]):,} positions")
    
    # Compare base scores (should be identical)
    print("\nBase score comparison:")
    base_cols = ['donor_score', 'acceptor_score', 'neither_score']
    for col in base_cols:
        base = dfs['base_only'][col].to_numpy()
        hybrid = dfs['hybrid'][col].to_numpy()
        meta = dfs['meta_only'][col].to_numpy()
        
        if len(base) == len(hybrid) == len(meta):
            import numpy as np
            base_hybrid_match = np.allclose(base, hybrid, rtol=1e-5)
            base_meta_match = np.allclose(base, meta, rtol=1e-5)
            
            print(f"  {col:20s}: base vs hybrid: {'✅ match' if base_hybrid_match else '❌ differ'}")
            print(f"  {' ':20s}  base vs meta:   {'✅ match' if base_meta_match else '❌ differ'}")
    
    # Compare meta scores (should differ)
    print("\nMeta score comparison:")
    meta_cols = ['donor_meta', 'acceptor_meta', 'neither_meta']
    for col in meta_cols:
        base = dfs['base_only'][col].to_numpy()
        hybrid = dfs['hybrid'][col].to_numpy()
        meta = dfs['meta_only'][col].to_numpy()
        
        if len(base) == len(hybrid) == len(meta):
            import numpy as np
            base_hybrid_diff = not np.allclose(base, hybrid, rtol=1e-5)
            base_meta_diff = not np.allclose(base, meta, rtol=1e-5)
            hybrid_meta_diff = not np.allclose(hybrid, meta, rtol=1e-5)
            
            print(f"  {col:20s}: base vs hybrid: {'✅ differ' if base_hybrid_diff else '⚠️  same'}")
            print(f"  {' ':20s}  base vs meta:   {'✅ differ' if base_meta_diff else '❌ SAME (BUG!)'}")
            print(f"  {' ':20s}  hybrid vs meta: {'✅ differ' if hybrid_meta_diff else '⚠️  same'}")

def main():
    gene_id = "ENSG00000134202"  # GSTM3, 7,107 bp
    expected_length = 7107
    
    print("="*80)
    print("FULL COVERAGE TEST: GSTM3 (ENSG00000134202)")
    print("="*80)
    print(f"Expected: {expected_length:,} positions (one per nucleotide)")
    print()
    
    # Test all 3 modes
    results = []
    for mode in ['base_only', 'hybrid', 'meta_only']:
        result = test_mode(mode, gene_id)
        results.append(result)
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print('='*80)
    
    all_success = all(r['success'] for r in results)
    all_correct_length = all(r.get('total_rows', 0) == expected_length for r in results)
    
    print("\nResults:")
    print(f"{'Mode':<15} {'Rows':<10} {'Unique Pos':<12} {'Expected':<10} {'Meta %':<10} {'Status'}")
    print("-" * 80)
    for r in results:
        if r['success']:
            status = '✅ PASS' if r['total_rows'] == expected_length else '❌ FAIL'
            print(f"{r['mode']:<15} {r['total_rows']:<10,} {r['unique_positions']:<12,} {expected_length:<10,} {r['meta_model_pct']:<10.1f} {status}")
        else:
            print(f"{r['mode']:<15} {'ERROR':<10} {'-':<12} {expected_length:<10,} {'-':<10} ❌ FAIL")
    
    if all_success and all_correct_length:
        print("\n✅ ALL TESTS PASSED: Full coverage achieved in all modes!")
        
        # Now compare scores
        compare_scores(gene_id)
        
        return 0
    else:
        print("\n❌ SOME TESTS FAILED")
        return 1

if __name__ == "__main__":
    sys.exit(main())

