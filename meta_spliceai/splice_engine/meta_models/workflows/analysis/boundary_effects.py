#!/usr/bin/env python3
"""
Boundary Position Investigator

This script investigates the exact source of the +1 position discrepancy by examining
the boundary handling in the SpliceAI prediction workflow.

Key Investigation Points:
1. How are positions generated in predict_splice_sites_for_genes?
2. What happens at gene start/end boundaries?
3. Which nucleotide position is missing when there's a +1 discrepancy?
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

# Add meta_spliceai to path
sys.path.insert(0, str(Path(__file__).parents[5]))


def analyze_position_generation_logic(gene_length: int, context: int = 10000) -> Dict[str, any]:
    """
    Simulate the position generation logic from predict_splice_sites_for_genes
    to understand boundary effects.
    
    This recreates the exact logic from lines 322-351 in run_spliceai_workflow.py
    """
    print(f"🔬 ANALYZING POSITION GENERATION FOR GENE LENGTH: {gene_length}")
    print("=" * 70)
    
    # Simulate SpliceAI block processing
    seq_len = gene_length
    block_size = 5000  # From the code: block_start = block_index * 5000
    
    # Calculate number of blocks needed
    # This comes from the sequence preparation logic
    total_seq_len = seq_len + 2 * context  # Add context padding
    n_blocks = (total_seq_len + block_size - 1) // block_size  # Ceiling division
    
    print(f"📊 Sequence Processing Setup:")
    print(f"  • Original gene length: {seq_len:,} bp")
    print(f"  • Context padding: {context:,} bp each side")
    print(f"  • Total sequence length: {total_seq_len:,} bp")
    print(f"  • Block size: {block_size:,} bp")
    print(f"  • Number of blocks: {n_blocks}")
    print()
    
    # Simulate position generation for each block
    positions_generated = []
    
    print(f"🔄 Block Processing Simulation:")
    for block_index in range(n_blocks):
        block_start = block_index * block_size
        
        # The model output length depends on the actual block content
        # For simplicity, assume each block generates block_size positions
        # (this is where the actual discrepancy might occur)
        
        block_positions = []
        for i in range(block_size):
            # This is the key condition from line 699: if block_start + i < seq_len:
            if block_start + i < seq_len:
                position = block_start + i
                block_positions.append(position)
        
        positions_generated.extend(block_positions)
        
        print(f"  Block {block_index + 1}: positions {block_start} to {block_start + len(block_positions) - 1} ({len(block_positions)} positions)")
        
        # Stop if we've covered the gene sequence
        if block_start >= seq_len:
            break
    
    print()
    
    # Analyze the results
    unique_positions = sorted(set(positions_generated))
    
    print(f"📈 Position Generation Results:")
    print(f"  • Total positions generated: {len(positions_generated):,}")
    print(f"  • Unique positions: {len(unique_positions):,}")
    print(f"  • Gene length: {gene_length:,}")
    print(f"  • Difference: {len(unique_positions) - gene_length:+d}")
    print()
    
    # Check for boundary effects
    if unique_positions:
        first_pos = min(unique_positions)
        last_pos = max(unique_positions)
        expected_last = gene_length - 1  # 0-based indexing
        
        print(f"🎯 Boundary Analysis:")
        print(f"  • First position: {first_pos}")
        print(f"  • Last position: {last_pos}")
        print(f"  • Expected last position (0-based): {expected_last}")
        print(f"  • Expected last position (1-based): {gene_length}")
        print()
        
        # Check for missing positions
        expected_positions = set(range(gene_length))  # 0-based: 0 to gene_length-1
        actual_positions = set(unique_positions)
        
        missing_positions = expected_positions - actual_positions
        extra_positions = actual_positions - expected_positions
        
        print(f"🔍 Position Discrepancy Analysis:")
        if missing_positions:
            print(f"  • Missing positions: {sorted(missing_positions)}")
        if extra_positions:
            print(f"  • Extra positions: {sorted(extra_positions)}")
        if not missing_positions and not extra_positions:
            print(f"  • ✅ Perfect match - all positions present")
        print()
    
    return {
        'gene_length': gene_length,
        'positions_generated': len(positions_generated),
        'unique_positions': len(unique_positions),
        'difference': len(unique_positions) - gene_length,
        'first_position': min(unique_positions) if unique_positions else None,
        'last_position': max(unique_positions) if unique_positions else None,
        'missing_positions': sorted(expected_positions - set(unique_positions)) if unique_positions else [],
        'extra_positions': sorted(set(unique_positions) - expected_positions) if unique_positions else []
    }


def investigate_coordinate_systems():
    """
    Investigate the coordinate system handling that might cause +1 discrepancies.
    """
    print("🧮 COORDINATE SYSTEM INVESTIGATION")
    print("=" * 70)
    
    # The key lines from the code are:
    # Line 334: absolute_position = gene_start + (block_start + i)
    # Line 336: absolute_position = gene_end - (block_start + i)
    # Line 342: pos_key = (gene_id, absolute_position) if has_absolute_positions else (gene_id, block_start + i + 1)
    
    print("🔍 Key Code Analysis:")
    print("From predict_splice_sites_for_genes (lines 330-342):")
    print()
    print("```python")
    print("for i, (donor_p, acceptor_p, neither_p) in enumerate(zip(donor_prob, acceptor_prob, neither_prob)):")
    print("    if has_absolute_positions:")
    print("        if strand == '+':")
    print("            absolute_position = gene_start + (block_start + i)")
    print("        elif strand == '-':")
    print("            absolute_position = gene_end - (block_start + i)")
    print("    else:")
    print("        absolute_position = None")
    print("    ")
    print("    pos_key = (gene_id, absolute_position) if has_absolute_positions else (gene_id, block_start + i + 1)")
    print("```")
    print()
    
    print("🎯 Potential Sources of +1 Discrepancy:")
    print()
    
    print("1. **Block Position Calculation**:")
    print("   • `block_start + i` gives 0-based position within gene sequence")
    print("   • For non-absolute positions: `block_start + i + 1` (converts to 1-based)")
    print("   • This +1 conversion could cause boundary effects")
    print()
    
    print("2. **Absolute Position Calculation**:")
    print("   • Forward strand: `gene_start + (block_start + i)`")
    print("   • Reverse strand: `gene_end - (block_start + i)`")
    print("   • Depends on how gene_start/gene_end are defined")
    print()
    
    print("3. **Sequence Length vs Position Range**:")
    print("   • Gene sequence of length N has positions 0 to N-1 (0-based)")
    print("   • Or positions 1 to N (1-based)")
    print("   • Boundary handling at position 0 or position N could cause +/-1 errors")
    print()
    
    print("4. **Block Boundary Effects**:")
    print("   • Sequences are processed in 5,000 bp blocks with context padding")
    print("   • Edge effects at block boundaries could cause position shifts")
    print("   • Context padding (10,000 bp each side) affects position calculations")
    print()


def analyze_specific_genes():
    """
    Analyze the specific genes from our test results to understand the pattern.
    """
    print("🧬 SPECIFIC GENE ANALYSIS")
    print("=" * 70)
    
    # Test cases from our earlier analysis
    test_cases = [
        {'gene_id': 'ENSG00000142748', 'gene_length': 5715, 'final_positions': 5716, 'diff': 1},
        {'gene_id': 'ENSG00000000003', 'gene_length': 4535, 'final_positions': 4535, 'diff': 0},  # Perfect match
        {'gene_id': 'ENSG00000000005', 'gene_length': 1652, 'final_positions': 1653, 'diff': 1}
    ]
    
    print("📊 Gene-by-Gene Analysis:")
    print()
    
    for case in test_cases:
        gene_id = case['gene_id']
        gene_length = case['gene_length']
        final_positions = case['final_positions']
        diff = case['diff']
        
        print(f"🔍 {gene_id}:")
        print(f"  • Gene length: {gene_length:,} bp")
        print(f"  • Final positions: {final_positions:,}")
        print(f"  • Difference: {diff:+d}")
        
        # Analyze position generation for this gene
        result = analyze_position_generation_logic(gene_length)
        
        print(f"  • Simulated positions: {result['unique_positions']:,}")
        print(f"  • Simulated difference: {result['difference']:+d}")
        
        if result['missing_positions']:
            print(f"  • Missing positions: {result['missing_positions']}")
        if result['extra_positions']:
            print(f"  • Extra positions: {result['extra_positions']}")
        
        print()
    
    print("🎯 Pattern Analysis:")
    print()
    print("From the test cases, we can see:")
    print("• Some genes have perfect matches (diff = 0)")
    print("• Some genes have +1 position discrepancy")
    print("• The discrepancy appears to be gene-specific, not systematic")
    print()
    print("💡 This suggests the +1 discrepancy is related to:")
    print("• Specific boundary conditions for certain gene lengths")
    print("• Edge cases in the block processing logic")
    print("• Coordinate system handling for specific gene structures")


def investigate_enhanced_evaluation_filtering():
    """
    Investigate if the position filtering happens in enhanced_evaluate_splice_site_errors.
    """
    print("🔬 ENHANCED EVALUATION FILTERING INVESTIGATION")
    print("=" * 70)
    
    print("📋 Processing Chain:")
    print("1. **predict_splice_sites_for_genes()** generates raw positions")
    print("   • Creates positions 0 to gene_length-1 (or 1 to gene_length)")
    print("   • Should generate exactly gene_length positions")
    print()
    
    print("2. **enhanced_process_predictions_with_all_scores()** processes predictions")
    print("   • Calls enhanced_evaluate_splice_site_errors()")
    print("   • May filter positions based on annotations or thresholds")
    print()
    
    print("3. **enhanced_evaluate_splice_site_errors()** evaluates against annotations")
    print("   • Compares predictions to known splice sites")
    print("   • May exclude positions that don't meet criteria")
    print("   • Could filter out boundary positions")
    print()
    
    print("🎯 Potential Filtering Points:")
    print("• **Annotation matching**: Positions without annotation data")
    print("• **Threshold filtering**: Positions below probability thresholds")
    print("• **Boundary exclusion**: First/last positions excluded for safety")
    print("• **Window effects**: Positions too close to gene boundaries")
    print()
    
    print("💡 To confirm, we need to check:")
    print("• Does predict_splice_sites_for_genes() generate exactly gene_length positions?")
    print("• Where does the filtering happen in the evaluation chain?")
    print("• Are boundary positions systematically excluded?")


def main():
    """Run all investigations."""
    print("🔍 BOUNDARY POSITION DISCREPANCY INVESTIGATION")
    print("=" * 80)
    print()
    
    # Investigation 1: Position generation logic
    print("INVESTIGATION 1: Position Generation Logic")
    print("-" * 50)
    analyze_position_generation_logic(5715)  # ENSG00000142748
    print("\n" + "="*80 + "\n")
    
    # Investigation 2: Coordinate systems
    print("INVESTIGATION 2: Coordinate System Analysis")
    print("-" * 50)
    investigate_coordinate_systems()
    print("\n" + "="*80 + "\n")
    
    # Investigation 3: Specific genes
    print("INVESTIGATION 3: Specific Gene Analysis")
    print("-" * 50)
    analyze_specific_genes()
    print("\n" + "="*80 + "\n")
    
    # Investigation 4: Evaluation filtering
    print("INVESTIGATION 4: Enhanced Evaluation Filtering")
    print("-" * 50)
    investigate_enhanced_evaluation_filtering()
    print("\n" + "="*80 + "\n")
    
    print("🎯 KEY FINDINGS SUMMARY:")
    print("=" * 50)
    print("1. **Position Generation**: SpliceAI should generate one position per nucleotide")
    print("2. **Boundary Effects**: +1 discrepancies likely occur at gene boundaries")
    print("3. **Gene-Specific**: Some genes have perfect matches, others have +1 discrepancy")
    print("4. **Filtering Chain**: Position filtering may occur in evaluation, not generation")
    print()
    print("🔬 NEXT STEPS:")
    print("• Examine actual SpliceAI output for specific genes")
    print("• Trace position counts through the evaluation pipeline")
    print("• Check if boundary positions are systematically filtered")
    print("• Determine if this affects prediction quality")


if __name__ == '__main__':
    main()
