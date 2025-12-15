#!/usr/bin/env python3
"""
Verify output completeness and correctness for all three inference modes.

Verifies:
1. Number of predicted scores = number of positions in gene sequence
2. Output tensor is N×3 (N positions, 3 splice types)
3. Score tensors differ between modes
4. For genes in training: full sequence output (not just unseen positions)
"""

import sys
from pathlib import Path
import polars as pl
import numpy as np

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))


def load_gene_info() -> pl.DataFrame:
    """Load gene information to get expected sequence lengths."""
    return pl.read_csv('data/ensembl/spliceai_analysis/gene_features.tsv',
                      separator='\t',
                      schema_overrides={'chrom': pl.Utf8})


def verify_gene_completeness(gene_id: str, gene_name: str, expected_length: int, 
                             is_observed: bool):
    """Verify output completeness for a single gene across all modes."""
    
    print(f"\n{'='*80}")
    print(f"🔍 VERIFYING: {gene_name} ({gene_id})")
    print(f"{'='*80}")
    print(f"  Category: {'OBSERVED (in training)' if is_observed else 'UNOBSERVED (not in training)'}")
    print(f"  Expected sequence length: {expected_length:,} bp")
    print()
    
    modes = ['base_only', 'hybrid', 'meta_only']
    mode_data = {}
    
    # Load predictions from all modes
    for mode in modes:
        pred_dir = Path(f"data/ensembl/spliceai_eval/meta_models/inference/predictions/{mode}/per_gene")
        pred_file = pred_dir / f"{gene_id}_predictions.parquet"
        
        if not pred_file.exists():
            print(f"  ❌ {mode}: File not found")
            continue
        
        df = pl.read_parquet(pred_file)
        mode_data[mode] = df
        
        n_positions = len(df)
        has_all_scores = all(col in df.columns for col in ['donor_score', 'acceptor_score', 'neither_score'])
        
        print(f"  {mode:12} | Positions: {n_positions:,} | Has 3 score types: {has_all_scores}")
    
    if len(mode_data) != 3:
        print(f"\n  ⚠️  Not all modes available for comparison")
        return
    
    print()
    print("-" * 80)
    print("CONDITION 1: Number of predictions = Gene sequence length")
    print("-" * 80)
    
    all_match = True
    for mode, df in mode_data.items():
        n_positions = len(df)
        matches = (n_positions == expected_length)
        coverage_pct = (n_positions / expected_length * 100) if expected_length > 0 else 0
        
        status = "✅" if matches else "❌"
        print(f"  {status} {mode:12}: {n_positions:,} positions ({coverage_pct:.1f}% coverage)")
        
        if not matches:
            all_match = False
            if is_observed:
                print(f"     NOTE: For observed genes, partial coverage may be expected")
                print(f"           (only unseen positions predicted, but should output full sequence)")
    
    if all_match:
        print(f"\n  ✅ All modes have complete coverage")
    else:
        print(f"\n  ⚠️  Some modes have incomplete coverage")
    
    print()
    print("-" * 80)
    print("CONDITION 2: Output tensor is N×3 (N positions, 3 splice types)")
    print("-" * 80)
    
    for mode, df in mode_data.items():
        has_donor = 'donor_score' in df.columns
        has_acceptor = 'acceptor_score' in df.columns
        has_neither = 'neither_score' in df.columns
        
        has_all = has_donor and has_acceptor and has_neither
        status = "✅" if has_all else "❌"
        
        print(f"  {status} {mode:12}: donor={has_donor}, acceptor={has_acceptor}, neither={has_neither}")
        
        if has_all:
            # Verify scores are valid probabilities
            donor = df['donor_score'].to_numpy()
            acceptor = df['acceptor_score'].to_numpy()
            neither = df['neither_score'].to_numpy()
            
            # Check if they sum to ~1.0
            score_sums = donor + acceptor + neither
            sum_ok = np.allclose(score_sums, 1.0, atol=0.01)
            
            if sum_ok:
                print(f"     ✅ Scores sum to 1.0 (valid probability distribution)")
            else:
                max_diff = np.max(np.abs(score_sums - 1.0))
                print(f"     ⚠️  Scores don't sum to 1.0 (max diff: {max_diff:.4f})")
    
    print()
    print("-" * 80)
    print("CONDITION 3: Score tensors differ between modes")
    print("-" * 80)
    
    base_df = mode_data['base_only']
    hybrid_df = mode_data['hybrid']
    meta_df = mode_data['meta_only']
    
    # Find common positions
    common_positions = set(base_df['position'].to_list()) & \
                      set(hybrid_df['position'].to_list()) & \
                      set(meta_df['position'].to_list())
    
    if len(common_positions) == 0:
        print(f"  ⚠️  No common positions to compare")
        return
    
    # Filter to common positions and sort
    base_common = base_df.filter(pl.col('position').is_in(list(common_positions))).sort('position')
    hybrid_common = hybrid_df.filter(pl.col('position').is_in(list(common_positions))).sort('position')
    meta_common = meta_df.filter(pl.col('position').is_in(list(common_positions))).sort('position')
    
    print(f"  Comparing {len(common_positions):,} common positions")
    print()
    
    # Compare donor scores
    base_donor = base_common['donor_score'].to_numpy()
    hybrid_donor = hybrid_common['donor_score'].to_numpy()
    meta_donor = meta_common['donor_score'].to_numpy()
    
    # Check if ANY position has different scores
    base_vs_hybrid_diff = ~np.isclose(base_donor, hybrid_donor, atol=1e-6)
    base_vs_meta_diff = ~np.isclose(base_donor, meta_donor, atol=1e-6)
    hybrid_vs_meta_diff = ~np.isclose(hybrid_donor, meta_donor, atol=1e-6)
    
    n_base_hybrid = np.sum(base_vs_hybrid_diff)
    n_base_meta = np.sum(base_vs_meta_diff)
    n_hybrid_meta = np.sum(hybrid_vs_meta_diff)
    
    print(f"  Donor score differences:")
    status_bh = "✅" if n_base_hybrid > 0 else "❌"
    status_bm = "✅" if n_base_meta > 0 else "❌"
    status_hm = "✅" if n_hybrid_meta > 0 else "❌"
    
    print(f"    {status_bh} Base vs Hybrid:    {n_base_hybrid:,} positions differ ({n_base_hybrid/len(common_positions)*100:.1f}%)")
    print(f"    {status_bm} Base vs Meta-only: {n_base_meta:,} positions differ ({n_base_meta/len(common_positions)*100:.1f}%)")
    print(f"    {status_hm} Hybrid vs Meta-only: {n_hybrid_meta:,} positions differ ({n_hybrid_meta/len(common_positions)*100:.1f}%)")
    
    if n_base_hybrid > 0:
        max_diff_bh = np.max(np.abs(base_donor - hybrid_donor))
        mean_diff_bh = np.mean(np.abs(base_donor[base_vs_hybrid_diff] - hybrid_donor[base_vs_hybrid_diff]))
        print(f"      Max difference: {max_diff_bh:.6f}, Mean: {mean_diff_bh:.6f}")
    
    if n_base_meta > 0:
        max_diff_bm = np.max(np.abs(base_donor - meta_donor))
        mean_diff_bm = np.mean(np.abs(base_donor[base_vs_meta_diff] - meta_donor[base_vs_meta_diff]))
        print(f"      Max difference: {max_diff_bm:.6f}, Mean: {mean_diff_bm:.6f}")
    
    print()
    
    # Special verification for meta-only mode
    print("-" * 80)
    print("SPECIAL CHECK: Meta-only mode should apply meta-model to ALL positions")
    print("-" * 80)
    
    if 'donor_meta' in meta_df.columns:
        meta_meta_scores = meta_df['donor_meta'].to_numpy()
        meta_base_scores = meta_df['donor_score'].to_numpy()
        
        # Count how many positions have meta != base
        meta_applied = ~np.isclose(meta_meta_scores, meta_base_scores, atol=1e-6)
        n_meta_applied = np.sum(meta_applied)
        pct_meta_applied = n_meta_applied / len(meta_df) * 100
        
        print(f"  Meta-only mode:")
        print(f"    Total positions: {len(meta_df):,}")
        print(f"    Positions with meta-model applied: {n_meta_applied:,} ({pct_meta_applied:.1f}%)")
        
        if pct_meta_applied >= 80:
            print(f"    ✅ Meta-model applied to most positions ({pct_meta_applied:.1f}% >= 80%)")
        elif pct_meta_applied >= 50:
            print(f"    ⚠️  Meta-model applied to {pct_meta_applied:.1f}% of positions (expected >= 80%)")
        else:
            print(f"    ❌ Meta-model applied to only {pct_meta_applied:.1f}% of positions (expected >= 80%)")
            print(f"       NOTE: In meta-only mode, ALL positions should use meta-model predictions")


def main():
    print("=" * 80)
    print("🔍 VERIFYING OUTPUT COMPLETENESS AND CORRECTNESS")
    print("=" * 80)
    print()
    print("This script verifies:")
    print("  1. Number of predicted scores = number of positions in gene sequence")
    print("  2. Output tensor is N×3 (N positions, 3 splice types)")
    print("  3. Score tensors differ between modes")
    print("  4. For genes in training: full sequence output (not just unseen positions)")
    print("  5. Meta-only mode applies meta-model to ALL positions")
    print()
    
    # Load gene information
    gene_info_df = load_gene_info()
    
    # Test genes (same as in test_all_inference_modes.py)
    test_genes = [
        ("ENSG00000065413", "ANKRD44", True),   # Observed
        ("ENSG00000134202", "GSTM3", True),     # Observed
        ("ENSG00000169239", "CA5B", True),      # Observed
        ("ENSG00000255071", "SAA2-SAA4", False), # Unobserved
        ("ENSG00000253250", "C8orf88", False),  # Unobserved
        ("ENSG00000006606", "CCL26", False),    # Unobserved
    ]
    
    for gene_id, gene_name, is_observed in test_genes:
        # Get expected length
        gene_row = gene_info_df.filter(pl.col('gene_id') == gene_id)
        if len(gene_row) == 0:
            print(f"\n⚠️  Gene {gene_id} not found in gene_features.tsv")
            continue
        
        expected_length = gene_row['gene_length'][0]
        
        verify_gene_completeness(gene_id, gene_name, expected_length, is_observed)
    
    print("\n" + "=" * 80)
    print("✅ VERIFICATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()

