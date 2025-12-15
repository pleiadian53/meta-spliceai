#!/usr/bin/env python3
"""
Verify that different inference modes produce different score predictions.

If meta-model is working correctly, we should see:
- base_only: donor_score, acceptor_score, neither_score from SpliceAI
- hybrid: Some positions have recalibrated scores (where meta-model was applied)
- meta_only: All positions have recalibrated scores
"""

import sys
from pathlib import Path
import polars as pl
import numpy as np

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))


def load_predictions(mode: str, gene_id: str) -> pl.DataFrame:
    """Load predictions for a specific mode and gene."""
    pred_dir = Path(f"data/ensembl/spliceai_eval/meta_models/inference/predictions/{mode}/per_gene")
    pred_file = pred_dir / f"{gene_id}_predictions.parquet"
    
    if not pred_file.exists():
        print(f"  ⚠️  File not found: {pred_file}")
        return None
    
    return pl.read_parquet(pred_file)


def compare_modes(gene_id: str, gene_name: str):
    """Compare predictions across modes for a single gene."""
    
    print(f"\n{'='*80}")
    print(f"🔍 ANALYZING: {gene_name} ({gene_id})")
    print(f"{'='*80}\n")
    
    # Load predictions from all modes
    base_df = load_predictions('base_only', gene_id)
    hybrid_df = load_predictions('hybrid', gene_id)
    meta_df = load_predictions('meta_only', gene_id)
    
    if base_df is None or hybrid_df is None or meta_df is None:
        print("  ❌ Could not load predictions for all modes")
        return
    
    print(f"Loaded predictions:")
    print(f"  Base-only:  {len(base_df):,} positions")
    print(f"  Hybrid:     {len(hybrid_df):,} positions")
    print(f"  Meta-only:  {len(meta_df):,} positions")
    print()
    
    # Check if DataFrames have the same positions
    all_same = (base_df['position'].equals(hybrid_df['position']) and 
                base_df['position'].equals(meta_df['position']))
    
    if not all_same:
        print("  ⚠️  WARNING: Position sets differ between modes!")
        print("     Comparing common positions only...")
        print()
        
        # Find common positions and sort DataFrames
        common_positions = set(base_df['position'].to_list()) & \
                          set(hybrid_df['position'].to_list()) & \
                          set(meta_df['position'].to_list())
        
        if not common_positions:
            print("  ❌ No common positions found!")
            return
        
        print(f"  Common positions: {len(common_positions):,}")
        print()
        
        base_df = base_df.filter(pl.col('position').is_in(list(common_positions))).sort('position')
        hybrid_df = hybrid_df.filter(pl.col('position').is_in(list(common_positions))).sort('position')
        meta_df = meta_df.filter(pl.col('position').is_in(list(common_positions))).sort('position')
    
    # Compare donor scores
    base_donor = base_df['donor_score'].to_numpy()
    hybrid_donor = hybrid_df['donor_score'].to_numpy()
    meta_donor = meta_df['donor_score'].to_numpy()
    
    # Check for exact equality
    base_vs_hybrid_identical = np.array_equal(base_donor, hybrid_donor)
    base_vs_meta_identical = np.array_equal(base_donor, meta_donor)
    hybrid_vs_meta_identical = np.array_equal(hybrid_donor, meta_donor)
    
    print("Score Identity Check (donor_score):")
    print(f"  Base vs Hybrid:     {'IDENTICAL ❌' if base_vs_hybrid_identical else 'DIFFERENT ✅'}")
    print(f"  Base vs Meta-only:  {'IDENTICAL ❌' if base_vs_meta_identical else 'DIFFERENT ✅'}")
    print(f"  Hybrid vs Meta-only: {'IDENTICAL ❌' if hybrid_vs_meta_identical else 'DIFFERENT ✅'}")
    print()
    
    # Check if meta columns exist and have values
    if 'donor_meta' in hybrid_df.columns:
        hybrid_meta = hybrid_df['donor_meta'].to_numpy()
        n_nonzero = np.sum(~np.isnan(hybrid_meta) & (hybrid_meta != 0))
        n_different = np.sum(~np.isnan(hybrid_meta) & (hybrid_meta != base_donor))
        
        print(f"Hybrid Mode Meta-Model Usage:")
        print(f"  Positions with donor_meta != 0:     {n_nonzero:,}")
        print(f"  Positions with donor_meta != donor_score: {n_different:,}")
        print(f"  Meta-model applied: {n_nonzero > 0}")
        print()
    
    if 'donor_meta' in meta_df.columns:
        meta_meta = meta_df['donor_meta'].to_numpy()
        n_nonzero = np.sum(~np.isnan(meta_meta) & (meta_meta != 0))
        n_different = np.sum(~np.isnan(meta_meta) & (meta_meta != base_donor))
        
        print(f"Meta-only Mode Meta-Model Usage:")
        print(f"  Positions with donor_meta != 0:     {n_nonzero:,}")
        print(f"  Positions with donor_meta != donor_score: {n_different:,}")
        print(f"  Meta-model applied: {n_nonzero > 0}")
        print()
    
    # Show score differences if they exist
    if not base_vs_hybrid_identical:
        diff = np.abs(hybrid_donor - base_donor)
        n_changed = np.sum(diff > 1e-6)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff[diff > 1e-6]) if n_changed > 0 else 0
        
        print(f"Base vs Hybrid Score Differences:")
        print(f"  Positions changed:  {n_changed:,} ({n_changed/len(base_donor)*100:.1f}%)")
        print(f"  Max difference:     {max_diff:.6f}")
        print(f"  Mean difference:    {mean_diff:.6f}")
        print()
    
    if not base_vs_meta_identical:
        diff = np.abs(meta_donor - base_donor)
        n_changed = np.sum(diff > 1e-6)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff[diff > 1e-6]) if n_changed > 0 else 0
        
        print(f"Base vs Meta-only Score Differences:")
        print(f"  Positions changed:  {n_changed:,} ({n_changed/len(base_donor)*100:.1f}%)")
        print(f"  Max difference:     {max_diff:.6f}")
        print(f"  Mean difference:    {mean_diff:.6f}")
        print()
    
    # Show example positions with high scores
    high_score_mask = base_donor >= 0.5
    if np.any(high_score_mask):
        print("Sample High-Confidence Positions (donor_score >= 0.5):")
        print(f"  {'Position':>8}  {'Base':>8}  {'Hybrid':>8}  {'Meta':>8}  {'Δ(H-B)':>8}  {'Δ(M-B)':>8}")
        print(f"  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}")
        
        high_indices = np.where(high_score_mask)[0][:10]  # Show first 10
        positions = base_df['position'].to_numpy()
        for idx in high_indices:
            pos = positions[idx]
            base_s = base_donor[idx]
            hybrid_s = hybrid_donor[idx]
            meta_s = meta_donor[idx]
            diff_h = hybrid_s - base_s
            diff_m = meta_s - base_s
            
            print(f"  {pos:8d}  {base_s:8.4f}  {hybrid_s:8.4f}  {meta_s:8.4f}  {diff_h:+8.4f}  {diff_m:+8.4f}")
        print()
    
    # Check uncertainty columns
    if 'is_uncertain' in base_df.columns:
        n_uncertain = base_df['is_uncertain'].sum()
        print(f"Uncertainty Analysis:")
        print(f"  Uncertain positions: {n_uncertain:,} ({n_uncertain/len(base_df)*100:.1f}%)")
        print()


def main():
    print("=" * 80)
    print("🔍 VERIFYING MODE DIFFERENCES")
    print("=" * 80)
    print()
    print("Expected behavior:")
    print("  • Base-only:  Uses SpliceAI scores directly")
    print("  • Hybrid:     Applies meta-model to uncertain positions only")
    print("  • Meta-only:  Applies meta-model to ALL positions")
    print()
    print("If all modes are identical, the meta-model is NOT being applied!")
    print()
    
    # Test genes
    test_genes = [
        ("ENSG00000065413", "ANKRD44"),   # Observed
        ("ENSG00000134202", "GSTM3"),     # Observed
        ("ENSG00000169239", "CA5B"),      # Observed
        ("ENSG00000255071", "SAA2-SAA4"), # Unobserved
        ("ENSG00000253250", "C8orf88"),   # Unobserved
        ("ENSG00000006606", "CCL26"),     # Unobserved
    ]
    
    for gene_id, gene_name in test_genes:
        compare_modes(gene_id, gene_name)
    
    print("\n" + "=" * 80)
    print("✅ VERIFICATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()

