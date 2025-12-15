#!/usr/bin/env python3
"""
Base Model Comparison

Compare SpliceAI and OpenSpliceAI performance across different gene categories.

This script uses modular components:
1. Gene Selection (GeneSelector) - Handles gene sampling and ID mapping
2. Base Model Runner (BaseModelRunner) - Handles model execution and comparison

MODES:
------
1. QUICK MODE (MODE='quick'):
   - Uses 5 well-known genes (BRCA1, TP53, EGFR, MYC, KRAS)
   - Fast execution (~2 minutes)
   
2. ROBUST MODE (MODE='robust'):
   - Samples genes by category from intersection
   - Comprehensive testing (~10 minutes)

Usage:
    python scripts/testing/compare_base_models.py
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from meta_spliceai.system.genomic_resources import (
    GeneSelector,
    GeneSamplingConfig
)
from meta_spliceai.system.base_model_runner import (
    BaseModelRunner,
    BaseModelConfig
)

# ============================================================================
# Configuration
# ============================================================================

# MODE SELECTION
MODE = 'robust'  # 'quick' or 'robust'

# Quick mode genes
QUICK_MODE_GENES = ['BRCA1', 'TP53', 'EGFR', 'MYC', 'KRAS']

# Robust mode sampling configuration
SAMPLING_CONFIG = GeneSamplingConfig(
    n_protein_coding=20,  # 20 protein-coding genes
    n_lncrna=5,           # 5 non-coding genes
    n_no_splice_sites=5,  # 5 genes without splice sites
    seed=None,            # Random sampling (different genes each run)
    min_confidence=0.9
)

# Base model configuration
MODEL_CONFIG = BaseModelConfig(
    mode='test',
    coverage='gene_subset',
    threshold=0.5,
    save_nucleotide_scores=True,  # Full coverage mode
    verbosity=1
)

# Models to compare
MODELS = ['spliceai', 'openspliceai']

# Generate test name
TEST_NAME = f"base_model_comparison_{MODE}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

print("=" * 80)
print(f"BASE MODEL COMPARISON - {MODE.upper()} MODE")
print("=" * 80)
print()
print(f"Test Name: {TEST_NAME}")
print(f"Mode: {MODE}")
print(f"Models: {', '.join(MODELS)}")
print(f"Nucleotide Scores: {'ENABLED' if MODEL_CONFIG.save_nucleotide_scores else 'DISABLED'}")
print()

# ============================================================================
# STEP 1: Gene Selection
# ============================================================================

print("=" * 80)
print("STEP 1: Gene Selection")
print("=" * 80)
print()

# Initialize gene selector
selector = GeneSelector()

if MODE == 'quick':
    # Quick mode: Use predefined genes
    print(f"Using {len(QUICK_MODE_GENES)} well-known genes:")
    for gene in QUICK_MODE_GENES:
        print(f"  • {gene}")
    print()
    
    # Load sources (needed for ID mapping)
    selector.load_sources(
        source1='ensembl',
        source1_build='GRCh37',
        source2='mane',
        source2_build='GRCh38',
        use_external_mapping=True,
        verbosity=1
    )
    
    # Get mappings for quick mode genes
    mappings = selector.get_high_confidence_mappings(
        'ensembl/GRCh37',
        'mane/GRCh38',
        min_confidence=0.9,
        verbosity=1
    )
    
    # Filter to quick mode genes
    gene_mappings_dict = {m.gene_symbol: m for m in mappings if m.gene_symbol in QUICK_MODE_GENES}
    
    # Create result manually
    from meta_spliceai.system.genomic_resources.gene_selection import GeneSamplingResult
    
    gene_symbols = QUICK_MODE_GENES
    source1_gene_ids = [gene_mappings_dict[g].source1_gene_id if g in gene_mappings_dict else None for g in gene_symbols]
    source2_gene_ids = [gene_mappings_dict[g].source2_gene_id if g in gene_mappings_dict else None for g in gene_symbols]
    
    sampling_result = GeneSamplingResult(
        gene_symbols=gene_symbols,
        source1_gene_ids=source1_gene_ids,
        source2_gene_ids=source2_gene_ids,
        mappings=[gene_mappings_dict[g] for g in gene_symbols if g in gene_mappings_dict],
        sampled_by_category={'protein_coding': gene_symbols, 'lncrna': [], 'no_splice_sites': []},
        total_available=len(mappings),
        total_sampled=len(gene_symbols),
        mapping_success_rate=sum(1 for g in source1_gene_ids if g is not None) / len(gene_symbols)
    )

else:
    # Robust mode: Sample from intersection
    sampling_result = selector.sample_genes_for_comparison(
        source1='ensembl',
        source1_build='GRCh37',
        source2='mane',
        source2_build='GRCh38',
        config=SAMPLING_CONFIG,
        use_external_mapping=True,
        verbosity=1
    )

# Print sampling summary
print("=" * 80)
print("GENE SELECTION SUMMARY")
print("=" * 80)
print()
print(f"Total sampled: {sampling_result.total_sampled} genes")
print(f"  • Protein-coding: {len(sampling_result.sampled_by_category['protein_coding'])}")
print(f"  • lncRNA: {len(sampling_result.sampled_by_category['lncrna'])}")
print(f"  • No splice sites: {len(sampling_result.sampled_by_category['no_splice_sites'])}")
print()
print(f"Mapping success rate: {sampling_result.mapping_success_rate:.1%}")
print()

# Save gene list
output_dir = Path(f"results/{TEST_NAME}")
output_dir.mkdir(parents=True, exist_ok=True)

with open(output_dir / "test_genes.txt", 'w') as f:
    for gene in sampling_result.gene_symbols:
        f.write(f"{gene}\n")

with open(output_dir / "sampled_genes.json", 'w') as f:
    json.dump({
        'mode': MODE,
        'genes': sampling_result.gene_symbols,
        'by_category': sampling_result.sampled_by_category,
        'total_available': sampling_result.total_available,
        'total_sampled': sampling_result.total_sampled,
        'mapping_success_rate': sampling_result.mapping_success_rate
    }, f, indent=2)

print(f"✅ Saved gene list to: {output_dir}/")
print()

# ============================================================================
# STEP 2: Run Base Models
# ============================================================================

print("=" * 80)
print("STEP 2: Run Base Models")
print("=" * 80)
print()

# Initialize runner
runner = BaseModelRunner()

# Prepare gene IDs for each model
gene_ids_by_model = {
    'spliceai': sampling_result.source1_gene_ids,
    'openspliceai': sampling_result.source2_gene_ids
}

# Run comparison
comparison_result = runner.compare_models(
    models=MODELS,
    gene_symbols=sampling_result.gene_symbols,
    gene_ids_by_model=gene_ids_by_model,
    test_name=TEST_NAME,
    config=MODEL_CONFIG,
    verbosity=1
)

# ============================================================================
# STEP 3: Print Comparison Summary
# ============================================================================

print("=" * 80)
print("STEP 3: Comparison Summary")
print("=" * 80)
print()

runner.print_comparison_summary(comparison_result, verbosity=1)

# ============================================================================
# STEP 4: Category-Specific Performance Analysis
# ============================================================================

print("=" * 80)
print("STEP 4: Performance by Gene Category")
print("=" * 80)
print()

import polars as pl

# Analyze performance by category
for model_name, model_result in comparison_result.models.items():
    if not model_result.success or model_result.positions.height == 0:
        print(f"⚠️  {model_name.upper()}: No positions to analyze")
        print()
        continue
    
    print(f"{model_name.upper()} - Performance by Category:")
    print("-" * 80)
    
    positions_df = model_result.positions
    
    # Get gene categories
    protein_coding_genes = set(sampling_result.sampled_by_category['protein_coding'])
    lncrna_genes = set(sampling_result.sampled_by_category['lncrna'])
    no_ss_genes = set(sampling_result.sampled_by_category['no_splice_sites'])
    
    # Determine which ID column to use
    gene_id_col = 'gene_id' if 'gene_id' in positions_df.columns else 'gene_name'
    
    # Map gene IDs back to symbols for categorization
    # Create a mapping from IDs to symbols
    id_to_symbol = {}
    if model_name == 'spliceai':
        for symbol, gene_id in zip(sampling_result.gene_symbols, sampling_result.source1_gene_ids):
            if gene_id:
                id_to_symbol[gene_id] = symbol
    else:  # openspliceai
        for symbol, gene_id in zip(sampling_result.gene_symbols, sampling_result.source2_gene_ids):
            if gene_id:
                id_to_symbol[gene_id] = symbol
    
    # Add category column to positions
    def get_category(gene_id):
        symbol = id_to_symbol.get(gene_id, gene_id)
        if symbol in protein_coding_genes:
            return 'protein_coding'
        elif symbol in lncrna_genes:
            return 'lncrna'
        elif symbol in no_ss_genes:
            return 'no_splice_sites'
        else:
            return 'unknown'
    
    positions_with_category = positions_df.with_columns(
        pl.col(gene_id_col).map_elements(get_category, return_dtype=pl.Utf8).alias('category')
    )
    
    # Calculate metrics per category
    categories = ['protein_coding', 'lncrna', 'no_splice_sites']
    
    print(f"{'Category':<20} {'Positions':<12} {'TP':<8} {'FP':<8} {'FN':<8} {'Precision':<12} {'Recall':<12} {'F1':<12}")
    print("-" * 80)
    
    for category in categories:
        cat_positions = positions_with_category.filter(pl.col('category') == category)
        
        if cat_positions.height == 0:
            print(f"{category:<20} {'0':<12} {'-':<8} {'-':<8} {'-':<8} {'-':<12} {'-':<12} {'-':<12}")
            continue
        
        tp = cat_positions.filter(pl.col('pred_type') == 'TP').height
        tn = cat_positions.filter(pl.col('pred_type') == 'TN').height
        fp = cat_positions.filter(pl.col('pred_type') == 'FP').height
        fn = cat_positions.filter(pl.col('pred_type') == 'FN').height
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        print(f"{category:<20} {cat_positions.height:<12,} {tp:<8,} {fp:<8,} {fn:<8,} {precision:<12.4f} {recall:<12.4f} {f1:<12.4f}")
    
    print()

print()

# Save comparison results
with open(output_dir / "comparison_results.json", 'w') as f:
    # Convert to JSON-serializable format
    results_dict = {
        'test_name': comparison_result.test_name,
        'mode': MODE,
        'models': {
            name: {
                'success': result.success,
                'runtime_seconds': result.runtime_seconds,
                'genes_processed': len(result.processed_genes),
                'genes_missing': len(result.missing_genes),
                'missing_genes': list(result.missing_genes),
                'metrics': result.metrics,
                'paths': result.paths,
                'error': result.error
            }
            for name, result in comparison_result.models.items()
        },
        'comparison_metrics': comparison_result.comparison_metrics
    }
    json.dump(results_dict, f, indent=2)

print(f"✅ Saved results to: {output_dir}/comparison_results.json")
print()

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("=" * 80)
print("FINAL SUMMARY")
print("=" * 80)
print()
print(f"✅ Completed {MODE.upper()} mode comparison:")
print()
print("   1. GENE SELECTION (GeneSelector)")
print(f"      • Sampled {sampling_result.total_sampled} genes")
print(f"      • Mapping success: {sampling_result.mapping_success_rate:.1%}")
print()
print("   2. BASE MODEL EXECUTION (BaseModelRunner)")
for model_name, result in comparison_result.models.items():
    status = "✅" if result.success else "❌"
    print(f"      {status} {model_name.upper()}: {len(result.processed_genes)}/{sampling_result.total_sampled} genes processed")
print()
print("   3. MODULAR ARCHITECTURE")
print("      • Gene selection: Separate, reusable module")
print("      • Model execution: Separate, reusable module")
print("      • Easy to extend with new models or sampling strategies")
print()
print("=" * 80)

