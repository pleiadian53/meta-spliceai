#!/usr/bin/env python3
"""
Compare prediction tensors between hybrid and meta-only modes.

Expected behavior:
- Hybrid: Meta-model applied only to uncertain positions (~0-2%)
- Meta-only: Meta-model applied to ALL positions (100%)
- Predictions should be DIFFERENT
"""

import sys
from pathlib import Path
import polars as pl
import numpy as np

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))


def compare_modes_for_gene(gene_id: str, gene_name: str):
    """Compare hybrid vs meta-only predictions for a single gene."""
    
    print(f"\n{'='*80}")
    print(f"🔍 COMPARING: {gene_name} ({gene_id})")
    print(f"{'='*80}\n")
    
    # Load predictions from both modes
    hybrid_file = Path(f"data/ensembl/spliceai_eval/meta_models/inference/predictions/hybrid/per_gene/{gene_id}_predictions.parquet")
    meta_file = Path(f"data/ensembl/spliceai_eval/meta_models/inference/predictions/meta_only/per_gene/{gene_id}_predictions.parquet")
    
    if not hybrid_file.exists():
        print(f"  ❌ Hybrid file not found: {hybrid_file}")
        return False
    
    if not meta_file.exists():
        print(f"  ❌ Meta-only file not found: {meta_file}")
        return False
    
    hybrid_df = pl.read_parquet(hybrid_file)
    meta_df = pl.read_parquet(meta_file)
    
    print(f"Loaded predictions:")
    print(f"  Hybrid:     {len(hybrid_df):,} positions")
    print(f"  Meta-only:  {len(meta_df):,} positions")
    print()
    
    # Find common positions
    common_positions = set(hybrid_df['position'].to_list()) & set(meta_df['position'].to_list())
    
    if len(common_positions) == 0:
        print(f"  ❌ No common positions to compare")
        return False
    
    print(f"Common positions: {len(common_positions):,}")
    print()
    
    # Filter to common positions and sort
    hybrid_common = hybrid_df.filter(pl.col('position').is_in(list(common_positions))).sort('position')
    meta_common = meta_df.filter(pl.col('position').is_in(list(common_positions))).sort('position')
    
    # Compare base scores (donor_score, acceptor_score, neither_score)
    print("─" * 80)
    print("COMPARISON 1: Base Scores (donor_score, acceptor_score, neither_score)")
    print("─" * 80)
    print("Expected: Should be IDENTICAL (both use same base model)")
    print()
    
    for score_type in ['donor_score', 'acceptor_score', 'neither_score']:
        hybrid_scores = hybrid_common[score_type].to_numpy()
        meta_scores = meta_common[score_type].to_numpy()
        
        identical = np.allclose(hybrid_scores, meta_scores, atol=1e-9)
        status = "✅" if identical else "❌"
        
        print(f"  {status} {score_type:15}: {'IDENTICAL' if identical else 'DIFFERENT'}")
        
        if not identical:
            diff = np.abs(hybrid_scores - meta_scores)
            n_diff = np.sum(diff > 1e-6)
            print(f"     {n_diff:,} positions differ (max diff: {np.max(diff):.6f})")
    
    print()
    print("─" * 80)
    print("COMPARISON 2: Meta Scores (donor_meta, acceptor_meta, neither_meta)")
    print("─" * 80)
    print("Expected: Should be DIFFERENT (meta-only applies to all positions)")
    print()
    
    has_meta_cols = 'donor_meta' in hybrid_df.columns and 'donor_meta' in meta_df.columns
    
    if not has_meta_cols:
        print("  ❌ Meta score columns not found!")
        return False
    
    # Check meta scores
    for score_type in ['donor_meta', 'acceptor_meta', 'neither_meta']:
        hybrid_meta = hybrid_common[score_type].to_numpy()
        meta_meta = meta_common[score_type].to_numpy()
        
        identical = np.allclose(hybrid_meta, meta_meta, atol=1e-9)
        status = "❌" if identical else "✅"  # Note: reversed - we WANT differences
        
        print(f"  {status} {score_type:15}: {'IDENTICAL ❌' if identical else 'DIFFERENT ✅'}")
        
        if not identical:
            diff = np.abs(hybrid_meta - meta_meta)
            n_diff = np.sum(diff > 1e-6)
            print(f"     {n_diff:,} positions differ (max diff: {np.max(diff):.6f})")
        else:
            print(f"     ⚠️  This is WRONG! Meta-only should be different from hybrid!")
    
    print()
    print("─" * 80)
    print("COMPARISON 3: Meta-Model Application Rate")
    print("─" * 80)
    print()
    
    # Check how many positions have meta != base
    hybrid_donor_base = hybrid_common['donor_score'].to_numpy()
    hybrid_donor_meta = hybrid_common['donor_meta'].to_numpy()
    meta_donor_base = meta_common['donor_score'].to_numpy()
    meta_donor_meta = meta_common['donor_meta'].to_numpy()
    
    # Hybrid: meta-model applied where meta != base
    hybrid_applied = ~np.isclose(hybrid_donor_base, hybrid_donor_meta, atol=1e-6)
    n_hybrid_applied = np.sum(hybrid_applied)
    pct_hybrid_applied = n_hybrid_applied / len(hybrid_common) * 100
    
    # Meta-only: meta-model applied where meta != base
    meta_applied = ~np.isclose(meta_donor_base, meta_donor_meta, atol=1e-6)
    n_meta_applied = np.sum(meta_applied)
    pct_meta_applied = n_meta_applied / len(meta_common) * 100
    
    print(f"Hybrid mode:")
    print(f"  Positions with meta-model applied: {n_hybrid_applied:,}/{len(hybrid_common):,} ({pct_hybrid_applied:.1f}%)")
    if pct_hybrid_applied < 50:
        print(f"  ✅ Expected: Only uncertain positions (~0-10%)")
    else:
        print(f"  ⚠️  Unexpected: Should be low percentage for hybrid mode")
    
    print()
    
    print(f"Meta-only mode:")
    print(f"  Positions with meta-model applied: {n_meta_applied:,}/{len(meta_common):,} ({pct_meta_applied:.1f}%)")
    if pct_meta_applied >= 80:
        print(f"  ✅ Expected: Meta-model applied to most/all positions")
    elif pct_meta_applied > 10:
        print(f"  ⚠️  Partial: Some positions have meta-model, but not all")
    else:
        print(f"  ❌ FAILED: Meta-model barely applied! Should be ~100%")
    
    print()
    print("─" * 80)
    print("COMPARISON 4: Uncertainty Flags")
    print("─" * 80)
    print()
    
    if 'is_uncertain' in hybrid_df.columns:
        n_hybrid_uncertain = hybrid_common['is_uncertain'].sum()
        pct_hybrid_uncertain = n_hybrid_uncertain / len(hybrid_common) * 100
        print(f"Hybrid mode:")
        print(f"  Positions marked uncertain: {n_hybrid_uncertain:,}/{len(hybrid_common):,} ({pct_hybrid_uncertain:.1f}%)")
    else:
        print(f"  ⚠️  'is_uncertain' column not found in hybrid output")
    
    if 'is_uncertain' in meta_df.columns:
        n_meta_uncertain = meta_common['is_uncertain'].sum()
        pct_meta_uncertain = n_meta_uncertain / len(meta_common) * 100
        print(f"\nMeta-only mode:")
        print(f"  Positions marked uncertain: {n_meta_uncertain:,}/{len(meta_common):,} ({pct_meta_uncertain:.1f}%)")
        if pct_meta_uncertain >= 80:
            print(f"  ✅ Expected: ALL positions should be marked uncertain in meta-only mode")
        else:
            print(f"  ❌ FAILED: Should be 100% in meta-only mode!")
    else:
        print(f"  ⚠️  'is_uncertain' column not found in meta-only output")
    
    print()
    
    # Summary
    print("─" * 80)
    print("SUMMARY")
    print("─" * 80)
    print()
    
    all_meta_identical = np.allclose(hybrid_meta, meta_meta, atol=1e-9)
    
    if all_meta_identical and pct_meta_applied < 10:
        print("❌ CRITICAL ISSUE: Meta-only mode is NOT working correctly!")
        print("   - Meta scores are IDENTICAL to hybrid")
        print("   - Meta-model applied to < 10% of positions")
        print("   - Expected: 100% application in meta-only mode")
        return False
    elif not all_meta_identical and pct_meta_applied >= 80:
        print("✅ SUCCESS: Meta-only mode is working correctly!")
        print("   - Meta scores differ from hybrid")
        print("   - Meta-model applied to most positions")
        return True
    else:
        print("⚠️  PARTIAL: Meta-only mode is partially working")
        print(f"   - Meta scores {'identical' if all_meta_identical else 'different'}")
        print(f"   - Meta-model applied to {pct_meta_applied:.1f}% of positions")
        return False


def main():
    print("=" * 80)
    print("🔍 COMPARING HYBRID VS META-ONLY MODES")
    print("=" * 80)
    print()
    print("Analyzing prediction tensor differences between modes...")
    print()
    
    # Test genes
    test_genes = [
        ("ENSG00000065413", "ANKRD44"),
        ("ENSG00000134202", "GSTM3"),
        ("ENSG00000169239", "CA5B"),
    ]
    
    results = []
    for gene_id, gene_name in test_genes:
        success = compare_modes_for_gene(gene_id, gene_name)
        results.append((gene_name, success))
    
    print("\n" + "=" * 80)
    print("📊 OVERALL RESULTS")
    print("=" * 80)
    print()
    
    for gene_name, success in results:
        status = "✅" if success else "❌"
        print(f"  {status} {gene_name}")
    
    n_success = sum(1 for _, success in results if success)
    print()
    print(f"Success rate: {n_success}/{len(results)} genes")
    
    if n_success == 0:
        print()
        print("❌ CRITICAL: Meta-only mode is not working as expected!")
        print("   The override logic to force ALL positions as uncertain is not working.")
    elif n_success == len(results):
        print()
        print("✅ All genes show correct meta-only behavior!")
    else:
        print()
        print("⚠️  Inconsistent behavior across genes")
    
    print()
    print("=" * 80)


if __name__ == "__main__":
    main()

