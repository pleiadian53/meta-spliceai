#!/usr/bin/env python3
"""
Test script to demonstrate the new "window" TN sampling mode.

This script shows how the window-based TN sampling creates coherent contextual
sequences around splice sites, which is beneficial for:
- CRF-based recalibration
- Multimodal modeling approaches
- Deep learning models that benefit from spatial locality
"""

import sys
import numpy as np
import polars as pl
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parents[4]
sys.path.insert(0, str(project_root))

from meta_spliceai.splice_engine.meta_models.core.enhanced_workflow import enhanced_process_predictions_with_all_scores


def create_test_predictions(gene_id="ENSG00000001", gene_length=2000, true_donor_pos=[500, 1200], true_acceptor_pos=[800, 1500]):
    """
    Create synthetic test predictions with known splice sites.
    
    Parameters:
    -----------
    gene_id : str
        Gene identifier
    gene_length : int
        Length of the gene sequence
    true_donor_pos : list
        List of true donor positions
    true_acceptor_pos : list
        List of true acceptor positions
    """
    
    # Create base probability arrays (mostly low values)
    donor_prob = np.random.uniform(0.01, 0.05, gene_length)
    acceptor_prob = np.random.uniform(0.01, 0.05, gene_length)
    neither_prob = 1.0 - donor_prob - acceptor_prob
    
    # Add high probability peaks at true splice sites
    for pos in true_donor_pos:
        if 0 <= pos < gene_length:
            donor_prob[pos] = 0.95
            acceptor_prob[pos] = 0.02
            neither_prob[pos] = 0.03
    
    for pos in true_acceptor_pos:
        if 0 <= pos < gene_length:
            acceptor_prob[pos] = 0.95
            donor_prob[pos] = 0.02
            neither_prob[pos] = 0.03
    
    # Create predictions dictionary
    predictions = {
        gene_id: {
            'gene_start': 1000000,  # Genomic start position
            'gene_end': 1000000 + gene_length,  # Genomic end position
            'strand': '+',
            'donor_prob': donor_prob.tolist(),
            'acceptor_prob': acceptor_prob.tolist(),
            'neither_prob': neither_prob.tolist()
        }
    }
    
    return predictions, true_donor_pos, true_acceptor_pos


def create_test_annotations(gene_id="ENSG00000001", true_donor_pos=[500, 1200], true_acceptor_pos=[800, 1500]):
    """
    Create test splice site annotations.
    """
    annotations = []
    
    # Add donor sites
    for i, pos in enumerate(true_donor_pos):
        annotations.append({
            'chrom': 'chr1',
            'start': 1000000 + pos,
            'end': 1000000 + pos + 1,
            'position': 1000000 + pos,
            'strand': '+',
            'site_type': 'donor',
            'gene_id': gene_id,
            'transcript_id': f'ENST0000000{i+1}'
        })
    
    # Add acceptor sites
    for i, pos in enumerate(true_acceptor_pos):
        annotations.append({
            'chrom': 'chr1',
            'start': 1000000 + pos,
            'end': 1000000 + pos + 1,
            'position': 1000000 + pos,
            'strand': '+',
            'site_type': 'acceptor',
            'gene_id': gene_id,
            'transcript_id': f'ENST0000000{i+1}'
        })
    
    return pl.DataFrame(annotations)


def compare_sampling_modes():
    """
    Compare different TN sampling modes to demonstrate the window mode advantages.
    """
    print("🧬 Testing Window-Based TN Sampling Mode")
    print("=" * 60)
    
    # Create test data
    gene_id = "ENSG00000TEST"
    true_donor_pos = [500, 1200]
    true_acceptor_pos = [800, 1500]
    
    predictions, _, _ = create_test_predictions(gene_id, 2000, true_donor_pos, true_acceptor_pos)
    annotations_df = create_test_annotations(gene_id, true_donor_pos, true_acceptor_pos)
    
    print(f"📊 Test setup:")
    print(f"  - Gene length: 2000 nucleotides")
    print(f"  - True donor sites: {true_donor_pos}")
    print(f"  - True acceptor sites: {true_acceptor_pos}")
    print(f"  - Error window: 100 nucleotides")
    print()
    
    # Test different sampling modes
    sampling_modes = ["random", "proximity", "window"]
    results = {}
    
    for mode in sampling_modes:
        print(f"🔍 Testing {mode.upper()} sampling mode:")
        
        error_df, positions_df = enhanced_process_predictions_with_all_scores(
            predictions=predictions,
            ss_annotations_df=annotations_df,
            threshold=0.5,
            consensus_window=2,
            error_window=100,  # Smaller window for testing
            collect_tn=True,
            no_tn_sampling=False,
            tn_sample_factor=1.0,  # 1:1 ratio for clearer comparison
            tn_sampling_mode=mode,
            add_derived_features=False,  # Skip features for cleaner output
            verbose=1
        )
        
        # Analyze TN distribution
        tn_positions = positions_df.filter(pl.col("pred_type") == "TN")
        
        if tn_positions.height > 0:
            tn_pos_list = tn_positions.select("position").to_series().to_list()
            
            # Calculate distances to nearest true splice sites
            all_true_pos = true_donor_pos + true_acceptor_pos
            min_distances = []
            for pos in tn_pos_list:
                min_dist = min(abs(pos - true_pos) for true_pos in all_true_pos)
                min_distances.append(min_dist)
            
            avg_distance = np.mean(min_distances)
            median_distance = np.median(min_distances)
            within_window = sum(1 for d in min_distances if d <= 100)
            
            results[mode] = {
                'tn_count': len(tn_pos_list),
                'avg_distance': avg_distance,
                'median_distance': median_distance,
                'within_window': within_window,
                'positions': tn_pos_list[:10]  # First 10 positions for display
            }
            
            print(f"  ✓ TN positions collected: {len(tn_pos_list)}")
            print(f"  ✓ Average distance to nearest splice site: {avg_distance:.1f}")
            print(f"  ✓ Median distance to nearest splice site: {median_distance:.1f}")
            print(f"  ✓ TNs within error window (≤100): {within_window}")
            print(f"  ✓ Sample positions: {tn_pos_list[:10]}")
        else:
            print(f"  ⚠️  No TN positions collected")
            results[mode] = {'tn_count': 0}
        
        print()
    
    # Summary comparison
    print("📈 COMPARISON SUMMARY:")
    print("=" * 60)
    
    for mode in sampling_modes:
        if results[mode]['tn_count'] > 0:
            r = results[mode]
            print(f"{mode.upper():>10}: {r['tn_count']:>3} TNs, "
                  f"avg dist: {r['avg_distance']:>5.1f}, "
                  f"in window: {r['within_window']:>3}")
    
    print()
    print("🎯 WINDOW MODE ADVANTAGES:")
    print("  ✓ Creates coherent contextual sequences around splice sites")
    print("  ✓ Ideal for CRF-based recalibration (needs spatial locality)")
    print("  ✓ Perfect for multimodal and deep learning approaches")
    print("  ✓ Maintains biological relevance of splice site neighborhoods")


if __name__ == "__main__":
    compare_sampling_modes()

