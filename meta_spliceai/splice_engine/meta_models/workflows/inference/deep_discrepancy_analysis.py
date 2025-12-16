#!/usr/bin/env python3
"""
Deep Discrepancy Analysis

This script investigates the specific questions about position count discrepancies:
1. Why are donor and acceptor score vectors asymmetric?
2. Where is the +1 discrepancy located (5' or 3' end)?
3. How does meta-only inference mode behave?
4. Why do some genes have no discrepancy?
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional

# Add meta_spliceai to path
sys.path.insert(0, str(Path(__file__).parents[5]))


def investigate_donor_acceptor_asymmetry():
    """
    Investigate why donor and acceptor score vectors have different lengths.
    """
    print("🔬 DONOR/ACCEPTOR ASYMMETRY INVESTIGATION")
    print("=" * 70)
    
    print("🎯 **ROOT CAUSE ANALYSIS**:")
    print()
    
    print("1. **Sequence Processing Boundary Effects**:")
    print("   • Gene sequences are processed with 10,000 bp context padding")
    print("   • Total sequence: gene_length + 20,000 bp")
    print("   • Processed in 5,000 bp overlapping blocks")
    print("   • Block boundaries may affect donor vs acceptor differently")
    print()
    
    print("2. **Strand-Specific Coordinate Calculations**:")
    print("   From predict_splice_sites_for_genes (lines 334-336):")
    print("   ```python")
    print("   if strand == '+':")
    print("       absolute_position = gene_start + (block_start + i)")
    print("   elif strand == '-':")
    print("       absolute_position = gene_end - (block_start + i)")
    print("   ```")
    print("   • Forward and reverse strand calculations differ")
    print("   • May create slight asymmetries at boundaries")
    print()
    
    print("3. **Context Window Edge Effects**:")
    print("   • SpliceAI model uses 10,000 bp context on each side")
    print("   • Donor sites may need different context than acceptor sites")
    print("   • Edge positions near context boundaries affected differently")
    print()
    
    print("4. **Block Processing Artifacts**:")
    print("   • Sequences split into 5,000 bp blocks")
    print("   • Overlapping regions averaged between blocks")
    print("   • Boundary conditions at block edges may differ for donor/acceptor")
    print()
    
    print("🔍 **SPECIFIC ASYMMETRY EXAMPLE (ENSG00000142748)**:")
    print("   • Gene length: 5,715 bp")
    print("   • Donor positions: 5,715 (exact match)")
    print("   • Acceptor positions: 5,728 (+13 positions)")
    print("   • Asymmetry: 0.227% (very small but consistent)")
    print()
    
    print("💡 **WHY THIS HAPPENS**:")
    print("   The asymmetry occurs because donor and acceptor splice sites have")
    print("   different sequence requirements and context dependencies:")
    print("   • Donor sites: GT dinucleotide with upstream context")
    print("   • Acceptor sites: AG dinucleotide with downstream context")
    print("   • Different context needs → different boundary handling")


def investigate_position_location():
    """
    Investigate where the +1 discrepancy is located in the sequence.
    """
    print("\n🎯 POSITION LOCATION INVESTIGATION")
    print("=" * 70)
    
    print("🔍 **WHERE IS THE +1 POSITION?**")
    print()
    
    print("**Hypothesis 1: 5' End (Start of Gene)**")
    print("   • Position 0 or position -1 added")
    print("   • Start codon region boundary effects")
    print("   • Upstream regulatory region inclusion")
    print("   • Likelihood: 🔥 MEDIUM")
    print()
    
    print("**Hypothesis 2: 3' End (End of Gene)**")
    print("   • Position gene_length or gene_length+1 added")
    print("   • Stop codon region boundary effects")
    print("   • Downstream regulatory region inclusion")
    print("   • Likelihood: 🔥 HIGH (most boundary effects occur here)")
    print()
    
    print("**Hypothesis 3: Internal Position**")
    print("   • Position added due to annotation mismatch")
    print("   • Exon-intron boundary adjustment")
    print("   • Alternative splicing site inclusion")
    print("   • Likelihood: 🔥 LOW (would be gene-structure dependent)")
    print()
    
    print("🧪 **EXPERIMENTAL DESIGN TO LOCATE +1 POSITION**:")
    print("```python")
    print("# Compare raw predictions vs final positions")
    print("raw_positions = set(predictions[gene_id]['positions'])")
    print("final_positions = set(positions_df['position'].to_list())")
    print("added_position = final_positions - raw_positions")
    print("print(f'Added position: {added_position}')")
    print()
    print("# Check relative to gene boundaries")
    print("gene_start = gene_info['start']")
    print("gene_end = gene_info['end']")
    print("for pos in added_position:")
    print("    if pos < gene_start:")
    print("        print(f'Position {pos} is UPSTREAM of gene start {gene_start}')")
    print("    elif pos > gene_end:")
    print("        print(f'Position {pos} is DOWNSTREAM of gene end {gene_end}')")
    print("    else:")
    print("        print(f'Position {pos} is INTERNAL to gene [{gene_start}, {gene_end}]')")
    print("```")


def analyze_meta_only_mode_behavior():
    """
    Analyze how meta-only inference mode handles position counts.
    """
    print("\n🧠 META-ONLY INFERENCE MODE ANALYSIS")
    print("=" * 70)
    
    print("🔍 **META-ONLY MODE BEHAVIOR PREDICTION**:")
    print()
    
    print("**Processing Chain in Meta-Only Mode**:")
    print("1. **Base Model Prediction**: Generates N positions (same as hybrid)")
    print("2. **Position Selection**: ALL positions marked as 'uncertain'")
    print("3. **Feature Generation**: Features generated for ALL positions")
    print("4. **Meta-Model Inference**: Meta-model predicts ALL positions")
    print("5. **Final Output**: Should have same position count as base model")
    print()
    
    print("**Expected Behavior**:")
    print("   • ✅ **Same +1 discrepancy**: Meta-only uses same evaluation pipeline")
    print("   • ✅ **Same position addition**: Evaluation adds boundary positions")
    print("   • ✅ **Same gene-specific pattern**: Depends on gene structure, not inference mode")
    print()
    
    print("**Key Insight**:")
    print("   The +1 discrepancy occurs in the EVALUATION phase, not the inference phase.")
    print("   Therefore, meta-only mode should show the SAME position count behavior.")
    print()
    
    print("🧪 **TEST COMMAND**:")
    print("```bash")
    print("python -m meta_spliceai.splice_engine.meta_models.workflows.inference.main_inference_workflow \\")
    print("    --model results/gene_cv_pc_1000_3mers_run_4 \\")
    print("    --training-dataset train_pc_1000_3mers \\")
    print("    --genes ENSG00000142748 \\")
    print("    --output-dir results/test_meta_only_verification \\")
    print("    --inference-mode meta_only \\")
    print("    --verbose")
    print("```")
    print("Expected: Same position counts as base_only mode")


def analyze_zero_discrepancy_genes():
    """
    Investigate why some genes have no position discrepancy.
    """
    print("\n🧬 ZERO DISCREPANCY GENE ANALYSIS")
    print("=" * 70)
    
    print("🔍 **OBSERVED PATTERN**:")
    print("   • ENSG00000142748 (5,715 bp): +1 discrepancy")
    print("   • ENSG00000000003 (4,535 bp): 0 discrepancy ← WHY?")
    print("   • ENSG00000000005 (1,652 bp): +1 discrepancy")
    print()
    
    print("🤔 **POTENTIAL EXPLANATIONS FOR ZERO DISCREPANCY**:")
    print()
    
    print("1. **Gene Structure Differences**:")
    print("   • **Simple genes**: Single exon, no complex splicing")
    print("   • **Clean boundaries**: No splice sites near gene ends")
    print("   • **Perfect annotations**: Gene boundaries match annotation boundaries")
    print("   • **No boundary sites**: No donor/acceptor sites at gene edges")
    print()
    
    print("2. **Annotation Quality**:")
    print("   • **Complete annotations**: All splice sites already in predictions")
    print("   • **No missing sites**: Evaluation finds no additional positions to add")
    print("   • **Boundary alignment**: Gene coordinates perfectly aligned with annotations")
    print()
    
    print("3. **Sequence Characteristics**:")
    print("   • **Length effects**: Certain gene lengths process cleanly")
    print("   • **Block alignment**: Gene length aligns well with 5,000 bp block boundaries")
    print("   • **Context effects**: No edge effects in context windows")
    print()
    
    print("4. **Evaluation Criteria**:")
    print("   • **Threshold effects**: All positions above evaluation thresholds")
    print("   • **Window effects**: No positions added by error/consensus windows")
    print("   • **Sampling effects**: TN sampling doesn't add boundary positions")
    print()
    
    print("🧪 **INVESTIGATION APPROACH**:")
    print("```python")
    print("# Compare gene characteristics")
    print("zero_discrepancy_genes = ['ENSG00000000003']")
    print("plus_one_genes = ['ENSG00000142748', 'ENSG00000000005']")
    print()
    print("for gene_set, label in [(zero_discrepancy_genes, 'Zero'), (plus_one_genes, '+1')]:")
    print("    print(f'{label} Discrepancy Genes:')")
    print("    for gene in gene_set:")
    print("        # Analyze gene structure")
    print("        exon_count = count_exons(gene)")
    print("        splice_sites = count_splice_sites(gene)")
    print("        boundary_sites = count_boundary_sites(gene)")
    print("        gene_length = get_gene_length(gene)")
    print("        ")
    print("        print(f'  {gene}: length={gene_length}, exons={exon_count}, '")
    print("              f'splice_sites={splice_sites}, boundary_sites={boundary_sites}')")
    print("```")


def propose_comprehensive_experiments():
    """
    Propose experiments to answer all four questions definitively.
    """
    print("\n🧪 COMPREHENSIVE EXPERIMENTAL DESIGN")
    print("=" * 70)
    
    print("🎯 **EXPERIMENT 1: Donor/Acceptor Asymmetry Tracing**")
    print("```python")
    print("# Add detailed logging to predict_splice_sites_for_genes")
    print("predictions = predict_splice_sites_for_genes(gene_df, models, verbose=True)")
    print("for gene_id in predictions:")
    print("    donor_len = len(predictions[gene_id]['donor_prob'])")
    print("    acceptor_len = len(predictions[gene_id]['acceptor_prob'])")
    print("    print(f'{gene_id}: donor={donor_len}, acceptor={acceptor_len}, diff={acceptor_len-donor_len}')")
    print("```")
    print("**Expected**: Identify exact source of asymmetry")
    print()
    
    print("🎯 **EXPERIMENT 2: Position Location Identification**")
    print("```python")
    print("# Track positions through the pipeline")
    print("def trace_position_addition(gene_id):")
    print("    # Get raw predictions")
    print("    predictions = predict_splice_sites_for_genes(...)")
    print("    raw_positions = sorted(predictions[gene_id]['positions'])")
    print("    ")
    print("    # Get evaluated positions")
    print("    error_df, positions_df = enhanced_evaluate_splice_site_errors(...)")
    print("    final_positions = sorted(positions_df['position'].to_list())")
    print("    ")
    print("    # Find added positions")
    print("    added = set(final_positions) - set(raw_positions)")
    print("    print(f'Gene: {gene_id}')")
    print("    print(f'Raw range: {min(raw_positions)} to {max(raw_positions)}')")
    print("    print(f'Final range: {min(final_positions)} to {max(final_positions)}')")
    print("    print(f'Added positions: {sorted(added)}')")
    print("```")
    print("**Expected**: Locate exact position of +1 discrepancy")
    print()
    
    print("🎯 **EXPERIMENT 3: Meta-Only Mode Testing**")
    print("```bash")
    print("# Test meta-only mode with same genes")
    print("for gene in ENSG00000142748 ENSG00000000003 ENSG00000000005; do")
    print("    echo 'Testing meta-only mode for' $gene")
    print("    python -m meta_spliceai...main_inference_workflow \\")
    print("        --genes $gene \\")
    print("        --inference-mode meta_only \\")
    print("        --verbose | grep 'Total positions'")
    print("done")
    print("```")
    print("**Expected**: Same position counts as base_only mode")
    print()
    
    print("🎯 **EXPERIMENT 4: Gene Structure Correlation**")
    print("```python")
    print("# Correlate discrepancy with gene features")
    print("test_genes = {")
    print("    'ENSG00000142748': {'discrepancy': 1, 'length': 5715},")
    print("    'ENSG00000000003': {'discrepancy': 0, 'length': 4535},")
    print("    'ENSG00000000005': {'discrepancy': 1, 'length': 1652}")
    print("}")
    print()
    print("for gene_id, info in test_genes.items():")
    print("    # Get gene structure from GTF")
    print("    gene_info = get_gene_structure(gene_id)")
    print("    print(f'{gene_id}:')")
    print("    print(f'  Discrepancy: {info[\"discrepancy\"]}')")
    print("    print(f'  Exons: {gene_info[\"exon_count\"]}')")
    print("    print(f'  Splice sites: {gene_info[\"splice_site_count\"]}')")
    print("    print(f'  Boundary sites: {gene_info[\"boundary_splice_sites\"]}')")
    print("```")
    print("**Expected**: Identify structural patterns associated with discrepancies")


def main():
    """Run the complete deep discrepancy analysis."""
    print("🔍 DEEP POSITION DISCREPANCY ANALYSIS")
    print("=" * 80)
    
    investigate_donor_acceptor_asymmetry()
    investigate_position_location()
    analyze_meta_only_mode_behavior()
    analyze_zero_discrepancy_genes()
    propose_comprehensive_experiments()
    
    print("\n" + "=" * 80)
    print("🎯 **SUMMARY OF INVESTIGATION PLAN**")
    print("=" * 80)
    print()
    print("**Question 1**: Donor/Acceptor Asymmetry")
    print("   → Boundary effects in sequence processing with different context needs")
    print()
    print("**Question 2**: +1 Position Location") 
    print("   → Most likely at 3' end (gene termination boundary)")
    print()
    print("**Question 3**: Meta-Only Mode Behavior")
    print("   → Should show SAME +1 discrepancy (evaluation phase effect)")
    print()
    print("**Question 4**: Zero Discrepancy Genes")
    print("   → Simple gene structure with clean boundaries and complete annotations")
    print()
    print("🔬 **NEXT STEPS**: Run the proposed experiments to validate these hypotheses!")


if __name__ == '__main__':
    main()
