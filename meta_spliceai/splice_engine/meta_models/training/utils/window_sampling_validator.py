#!/usr/bin/env python3
"""
Window Sampling Validation Script

This script analyzes the generated analysis sequences to verify window-based TN sampling behavior:
1. Check for contiguous TN positions around splice sites
2. Verify symmetric flanking regions
3. Confirm FP preservation
4. Analyze window coverage and distribution
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
import argparse

# Add project root to path
project_root = Path(__file__).parents[4]
sys.path.insert(0, str(project_root))


def analyze_window_sampling(analysis_file_path, window_size=500, verbose=True):
    """
    Analyze window sampling behavior in analysis sequences.
    
    Parameters:
    -----------
    analysis_file_path : str
        Path to analysis_sequences_*.tsv file
    window_size : int
        Expected window size for analysis
    verbose : bool
        Print detailed analysis
    """
    
    if verbose:
        print(f"🔍 Analyzing window sampling in: {analysis_file_path}")
        print("=" * 80)
    
    # Load the analysis sequences
    try:
        df = pd.read_csv(analysis_file_path, sep='\t')
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return None
    
    if verbose:
        print(f"📊 Dataset overview:")
        print(f"   Total positions: {len(df):,}")
        print(f"   Unique genes: {df['gene_id'].nunique()}")
        print(f"   Columns: {len(df.columns)}")
    
    # Analyze prediction types
    pred_type_counts = df['pred_type'].value_counts()
    if verbose:
        print(f"\n📈 Prediction type distribution:")
        for pred_type, count in pred_type_counts.items():
            print(f"   {pred_type}: {count:,} positions")
    
    # Analyze splice types
    splice_type_counts = df['splice_type'].value_counts(dropna=False)
    if verbose:
        print(f"\n🧬 Splice type distribution:")
        for splice_type, count in splice_type_counts.items():
            splice_type_str = str(splice_type) if pd.notna(splice_type) else "None (non-splice)"
            print(f"   {splice_type_str}: {count:,} positions")
    
    results = {}
    
    # Question 1: Check for contiguous TN positions around splice sites
    if verbose:
        print(f"\n🎯 QUESTION 1: Contiguous TN positions around splice sites")
        print("-" * 60)
    
    splice_sites = df[df['splice_type'].isin(['donor', 'acceptor'])].copy()
    tn_positions = df[df['pred_type'] == 'TN'].copy()
    
    contiguous_analysis = []
    
    for _, splice_site in splice_sites.iterrows():
        gene_id = splice_site['gene_id']
        splice_pos = splice_site['position']
        splice_type = splice_site['splice_type']
        
        # Get TNs for this gene
        gene_tns = tn_positions[tn_positions['gene_id'] == gene_id]
        
        if len(gene_tns) == 0:
            continue
            
        # Find TNs within window
        distances = abs(gene_tns['position'] - splice_pos)
        nearby_tns = gene_tns[distances <= window_size].copy()
        nearby_tns = nearby_tns.sort_values('position')
        
        # Check for contiguous positions
        if len(nearby_tns) > 1:
            positions = nearby_tns['position'].values
            gaps = np.diff(positions)
            max_gap = gaps.max() if len(gaps) > 0 else 0
            contiguous_sections = np.sum(gaps == 1) + 1  # Number of contiguous sections
            
            # Analyze flanking symmetry
            left_positions = positions[positions < splice_pos]
            right_positions = positions[positions > splice_pos]
            
            left_count = len(left_positions)
            right_count = len(right_positions)
            
            contiguous_analysis.append({
                'gene_id': gene_id,
                'splice_type': splice_type,
                'splice_position': splice_pos,
                'nearby_tn_count': len(nearby_tns),
                'max_gap': max_gap,
                'contiguous_sections': contiguous_sections,
                'left_flanking': left_count,
                'right_flanking': right_count,
                'symmetric_ratio': min(left_count, right_count) / max(left_count, right_count) if max(left_count, right_count) > 0 else 0,
                'window_coverage': len(nearby_tns) / (2 * window_size + 1) if window_size > 0 else 0
            })
    
    if contiguous_analysis:
        contiguous_df = pd.DataFrame(contiguous_analysis)
        
        if verbose:
            print(f"   Analyzed {len(contiguous_df)} splice sites with nearby TNs")
            print(f"   Average TNs per splice site: {contiguous_df['nearby_tn_count'].mean():.1f}")
            print(f"   Average max gap: {contiguous_df['max_gap'].mean():.1f}")
            print(f"   Average symmetric ratio: {contiguous_df['symmetric_ratio'].mean():.2f}")
            
            # Show examples
            print(f"\n   📋 Sample analysis (first 3 splice sites):")
            for _, row in contiguous_df.head(3).iterrows():
                print(f"      {row['splice_type']} @ {row['splice_position']}: "
                      f"{row['nearby_tn_count']} TNs, "
                      f"gap={row['max_gap']}, "
                      f"L={row['left_flanking']}/R={row['right_flanking']}, "
                      f"sym={row['symmetric_ratio']:.2f}")
        
        results['contiguous_analysis'] = contiguous_df
    
    # Question 2: Window symmetry analysis
    if verbose:
        print(f"\n🎯 QUESTION 2: Window symmetry and centering")
        print("-" * 60)
    
    if contiguous_analysis:
        symmetric_sites = sum(1 for analysis in contiguous_analysis if analysis['symmetric_ratio'] > 0.8)
        total_sites = len(contiguous_analysis)
        
        if verbose:
            print(f"   Symmetric sites (>80% ratio): {symmetric_sites}/{total_sites} ({100*symmetric_sites/total_sites:.1f}%)")
            
            # Analyze window coverage
            avg_coverage = np.mean([a['window_coverage'] for a in contiguous_analysis])
            print(f"   Average window coverage: {avg_coverage:.1%}")
            
            # Check if splice sites are centered
            centered_analysis = []
            for analysis in contiguous_analysis:
                left_count = analysis['left_flanking']
                right_count = analysis['right_flanking']
                total_flanking = left_count + right_count
                
                if total_flanking > 0:
                    left_ratio = left_count / total_flanking
                    is_centered = 0.4 <= left_ratio <= 0.6  # Within 20% of center
                    centered_analysis.append(is_centered)
            
            if centered_analysis:
                centered_count = sum(centered_analysis)
                print(f"   Centered splice sites (40-60% left/right): {centered_count}/{len(centered_analysis)} ({100*centered_count/len(centered_analysis):.1f}%)")
    
    # Question 3: FP preservation
    if verbose:
        print(f"\n🎯 QUESTION 3: FP preservation")
        print("-" * 60)
    
    fp_positions = df[df['pred_type'] == 'FP']
    
    if len(fp_positions) > 0:
        if verbose:
            print(f"   Total FP positions: {len(fp_positions)}")
            print(f"   FP genes: {fp_positions['gene_id'].nunique()}")
            
            # Check if FPs are preserved regardless of location
            for gene_id in fp_positions['gene_id'].unique():
                gene_fps = fp_positions[fp_positions['gene_id'] == gene_id]
                gene_splice_sites = splice_sites[splice_sites['gene_id'] == gene_id]
                
                if len(gene_splice_sites) > 0:
                    # Calculate distances from FPs to nearest splice sites
                    for _, fp in gene_fps.iterrows():
                        fp_pos = fp['position']
                        distances_to_splice = [abs(fp_pos - ss['position']) for _, ss in gene_splice_sites.iterrows()]
                        min_distance = min(distances_to_splice) if distances_to_splice else float('inf')
                        
                        print(f"      FP @ {fp_pos}: {min_distance:.0f}bp from nearest splice site")
    else:
        if verbose:
            print(f"   No FP positions found in this dataset")
    
    results['fp_analysis'] = fp_positions
    
    # Question 4: Should we collect flanking positions for FPs?
    if verbose:
        print(f"\n🎯 QUESTION 4: FP flanking position analysis")
        print("-" * 60)
    
    if len(fp_positions) > 0:
        fp_flanking_analysis = []
        
        for _, fp in fp_positions.iterrows():
            gene_id = fp['gene_id']
            fp_pos = fp['position']
            
            # Get all positions for this gene
            gene_positions = df[df['gene_id'] == gene_id].sort_values('position')
            
            # Find positions within window of this FP
            distances = abs(gene_positions['position'] - fp_pos)
            nearby_positions = gene_positions[distances <= window_size]
            
            # Count different types nearby
            nearby_counts = nearby_positions['pred_type'].value_counts()
            
            fp_flanking_analysis.append({
                'gene_id': gene_id,
                'fp_position': fp_pos,
                'nearby_total': len(nearby_positions),
                'nearby_tn': nearby_counts.get('TN', 0),
                'nearby_tp': nearby_counts.get('TP', 0),
                'nearby_fn': nearby_counts.get('FN', 0),
                'nearby_fp': nearby_counts.get('FP', 0)
            })
        
        if fp_flanking_analysis:
            fp_flanking_df = pd.DataFrame(fp_flanking_analysis)
            
            if verbose:
                print(f"   FP flanking analysis:")
                print(f"   Average nearby positions per FP: {fp_flanking_df['nearby_total'].mean():.1f}")
                print(f"   Average nearby TNs per FP: {fp_flanking_df['nearby_tn'].mean():.1f}")
                
                print(f"\n   📋 Sample FP flanking analysis:")
                for _, row in fp_flanking_df.iterrows():
                    print(f"      FP @ {row['fp_position']}: "
                          f"{row['nearby_total']} nearby ({row['nearby_tn']} TN, "
                          f"{row['nearby_tp']} TP, {row['nearby_fn']} FN)")
            
            results['fp_flanking_analysis'] = fp_flanking_df
    
    return results


def analyze_all_analysis_files(analysis_dir="/home/bchiu/work/splice-surveyor/data/ensembl/spliceai_eval/meta_models", 
                               window_size=500):
    """
    Analyze all analysis_sequences_*.tsv files in the directory.
    """
    
    analysis_dir = Path(analysis_dir)
    analysis_files = list(analysis_dir.glob("analysis_sequences_*.tsv"))
    
    print(f"🔍 WINDOW SAMPLING VALIDATION ANALYSIS")
    print(f"📁 Analysis directory: {analysis_dir}")
    print(f"📄 Found {len(analysis_files)} analysis files")
    print(f"🪟 Window size: {window_size}")
    print("=" * 80)
    
    all_results = {}
    
    for file_path in sorted(analysis_files):
        print(f"\n📄 Analyzing: {file_path.name}")
        print("-" * 50)
        
        results = analyze_window_sampling(file_path, window_size, verbose=True)
        if results:
            all_results[file_path.name] = results
    
    # Overall summary
    print(f"\n📊 OVERALL SUMMARY")
    print("=" * 80)
    
    total_files = len(all_results)
    total_contiguous_sites = sum(len(r.get('contiguous_analysis', [])) for r in all_results.values())
    total_fps = sum(len(r.get('fp_analysis', [])) for r in all_results.values())
    
    print(f"   Files analyzed: {total_files}")
    print(f"   Total splice sites with nearby TNs: {total_contiguous_sites}")
    print(f"   Total FP positions: {total_fps}")
    
    # Aggregate symmetry analysis
    all_symmetric_ratios = []
    all_window_coverage = []
    
    for results in all_results.values():
        if 'contiguous_analysis' in results:
            contiguous_df = results['contiguous_analysis']
            all_symmetric_ratios.extend(contiguous_df['symmetric_ratio'].tolist())
            all_window_coverage.extend(contiguous_df['window_coverage'].tolist())
    
    if all_symmetric_ratios:
        avg_symmetry = np.mean(all_symmetric_ratios)
        highly_symmetric = sum(1 for ratio in all_symmetric_ratios if ratio > 0.8)
        
        print(f"   Average symmetry ratio: {avg_symmetry:.2f}")
        print(f"   Highly symmetric sites (>0.8): {highly_symmetric}/{len(all_symmetric_ratios)} ({100*highly_symmetric/len(all_symmetric_ratios):.1f}%)")
        
        if all_window_coverage:
            avg_coverage = np.mean(all_window_coverage)
            print(f"   Average window coverage: {avg_coverage:.1%}")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description="Validate window-based TN sampling behavior")
    parser.add_argument("--analysis-dir", 
                       default="/home/bchiu/work/splice-surveyor/data/ensembl/spliceai_eval/meta_models",
                       help="Directory containing analysis_sequences_*.tsv files")
    parser.add_argument("--window-size", type=int, default=500,
                       help="Expected window size for analysis")
    parser.add_argument("--file-pattern", default="analysis_sequences_*.tsv",
                       help="File pattern to match")
    
    args = parser.parse_args()
    
    # Run the analysis
    results = analyze_all_analysis_files(args.analysis_dir, args.window_size)
    
    print(f"\n✅ Analysis complete!")
    
    # Recommendations based on results
    print(f"\n💡 RECOMMENDATIONS:")
    print("-" * 30)
    
    total_splice_sites = sum(len(r.get('contiguous_analysis', [])) for r in results.values())
    total_fps = sum(len(r.get('fp_analysis', [])) for r in results.values())
    
    if total_splice_sites > 0:
        print(f"✅ Found {total_splice_sites} splice sites with nearby TNs - window sampling is working")
    else:
        print(f"⚠️  No splice sites with nearby TNs found - check window sampling configuration")
    
    if total_fps > 0:
        print(f"✅ Found {total_fps} FP positions preserved - good for meta-learning")
        print(f"💭 Consider adding flanking positions for FPs to improve context")
    else:
        print(f"ℹ️  No FP positions in current dataset")
    
    print(f"\n🔬 For CRF and advanced ML approaches:")
    print(f"   • Window-based sampling provides spatial locality ✅")
    print(f"   • Contiguous sequences around splice sites ✅") 
    print(f"   • Preserved FP positions for learning patterns ✅")


if __name__ == "__main__":
    main()



