#!/usr/bin/env python
"""
Test OpenSpliceAI Base Model with Gene Category Analysis

This script tests OpenSpliceAI predictions across different gene categories:
- Protein-coding genes (20 genes)
- lncRNA genes (5 genes)
- Genes without splice sites (5 genes)

Compares performance across categories and validates the base model workflow.
"""

import sys
import os
sys.path.insert(0, '/Users/pleiadian53/work/meta-spliceai')

from meta_spliceai import run_base_model_predictions, BaseModelConfig
from meta_spliceai.system.genomic_resources import Registry
import polars as pl
import random
from datetime import datetime

print("=" * 80)
print("OPENSPLICEAI BASE MODEL - GENE CATEGORY TEST")
print("=" * 80)
print()

# Test configuration
TEST_NAME = f"openspliceai_gene_categories_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
BASE_MODEL = 'openspliceai'
BUILD = 'GRCh38_MANE'
RELEASE = '1.3'

print(f"Test Name: {TEST_NAME}")
print(f"Base Model: {BASE_MODEL}")
print(f"Build: {BUILD}")
print(f"Release: {RELEASE}")
print()

# Initialize registry for GRCh38 MANE
registry = Registry(build=BUILD, release=RELEASE)
print(f"Data Directory: {registry.data_dir}")
print(f"GTF Path: {registry.get_gtf_path()}")
print(f"FASTA Path: {registry.get_fasta_path()}")
print()

# ============================================================================
# STEP 1: Define Test Genes (Manual Selection for MANE)
# ============================================================================
print("=" * 80)
print("STEP 1: Define Test Genes")
print("=" * 80)
print()

# Load gene features to sample genes by category
print("Loading gene features...")
gene_features_path = registry.data_dir / "gene_features.tsv"

if not gene_features_path.exists():
    print(f"❌ Gene features not found at: {gene_features_path}")
    print("Please run: python meta_spliceai/splice_engine/extract_gene_features_mane.py")
    sys.exit(1)

gene_features_df = pl.read_csv(
    str(gene_features_path),
    separator='\t',
    schema_overrides={'chrom': pl.Utf8}
)

print(f"✅ Loaded {gene_features_df.height:,} genes")
print()

# Sample genes by category
random.seed(42)

# 1. Protein-coding genes with many splice sites (20 genes)
# Select genes with more exons (more splice sites)
protein_coding_multi = gene_features_df.filter(
    (pl.col('gene_type') == 'protein_coding') &
    (pl.col('num_transcripts') >= 1) &
    (pl.col('gene_length') > 10000)  # Longer genes typically have more exons
).sample(n=min(20, gene_features_df.height), seed=42)

print(f"✅ Sampled {protein_coding_multi.height} protein-coding genes (multi-exon)")

# 2. Genes with few splice sites (10 genes)
# MANE doesn't have lncRNAs, so use genes with fewer exons instead
few_splice_sites = gene_features_df.filter(
    (pl.col('gene_type') == 'protein_coding') &
    (pl.col('gene_length') < 5000)  # Shorter genes, likely fewer exons
).sample(n=min(10, gene_features_df.height), seed=42)

print(f"✅ Sampled {few_splice_sites.height} genes with likely fewer splice sites (short genes)")
print()

# Combine all sampled genes
sampled_genes = pl.concat([protein_coding_multi, few_splice_sites])

# Add category labels
sampled_genes = sampled_genes.with_columns([
    pl.when(pl.col('gene_length') < 5000)
      .then(pl.lit('few_splice_sites'))
      .otherwise(pl.lit('protein_coding'))
      .alias('category')
])

print(f"Total sampled genes: {sampled_genes.height}")
print()

# Extract gene list
target_genes = sampled_genes['gene_id'].to_list()

print(f"Selected {len(target_genes)} genes for testing:")
print(f"  - Protein-coding (multi-exon): {protein_coding_multi.height}")
print(f"  - Few splice sites (short): {few_splice_sites.height}")
print()
print("Note: MANE focuses on canonical protein-coding transcripts")
print("      No lncRNA genes available in MANE")
print()

# Save gene list
output_dir = f"results/{TEST_NAME}"
os.makedirs(output_dir, exist_ok=True)

gene_list_file = f"{output_dir}/test_genes.txt"
with open(gene_list_file, 'w') as f:
    for gene in target_genes:
        f.write(f"{gene}\n")

print(f"✅ Saved gene list to: {gene_list_file}")
print()

# ============================================================================
# STEP 2: Run OpenSpliceAI Predictions
# ============================================================================
print("=" * 80)
print("STEP 2: Run OpenSpliceAI Predictions")
print("=" * 80)
print()

print(f"Running predictions on {len(target_genes)} genes...")
print()

# Run predictions (let run_base_model_predictions handle config creation)
try:
    results = run_base_model_predictions(
        base_model=BASE_MODEL,
        target_genes=target_genes,
        mode='test',
        coverage='gene_subset',
        test_name=TEST_NAME,
        threshold=0.5,
        consensus_window=2,
        error_window=500,
        use_auto_position_adjustments=True,
        # Enable sequence extraction for target genes
        do_extract_sequences=True,
        test_mode=True,  # Extract only target genes
        verbosity=2,
        no_tn_sampling=False  # Sample TNs for efficiency
    )
    
    print()
    print("=" * 80)
    print("PREDICTION RESULTS")
    print("=" * 80)
    print()
    
    if results['success']:
        print("✅ Predictions completed successfully!")
        print()
        print(f"Total positions analyzed: {results['positions'].height:,}")
        print(f"Errors detected: {results['error_analysis'].height:,}")
        print()
        print(f"Artifact directory: {results['paths']['artifacts_dir']}")
        print(f"Positions file: {results['paths']['positions_artifact']}")
        print(f"Errors file: {results['paths']['errors_artifact']}")
        print()
        
        # Save results summary
        summary = {
            'test_name': TEST_NAME,
            'base_model': BASE_MODEL,
            'build': BUILD,
            'total_genes': len(target_genes),
            'total_positions': results['positions'].height,
            'total_errors': results['error_analysis'].height,
            'artifact_manager': results['artifact_manager']
        }
        
        import json
        with open(f"{output_dir}/results_summary.json", 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"✅ Saved results summary to: {output_dir}/results_summary.json")
        
    else:
        print("❌ Predictions failed!")
        sys.exit(1)
        
except Exception as e:
    print(f"❌ Error during predictions: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# STEP 3: Analyze Performance by Category
# ============================================================================
print()
print("=" * 80)
print("STEP 3: Analyze Performance by Category")
print("=" * 80)
print()

# Load positions and errors
positions_df = results['positions']
errors_df = results['error_analysis']

if positions_df.height == 0:
    print("⚠️  No positions to analyze")
else:
    # Merge with gene categories
    positions_with_category = positions_df.join(
        sampled_genes.select(['gene_id', 'category']),
        on='gene_id',
        how='left'
    )
    
    # Calculate metrics by category
    print("Performance by Gene Category:")
    print("-" * 80)
    print()
    
    categories = ['protein_coding', 'few_splice_sites']
    category_metrics = []
    
    for category in categories:
        cat_positions = positions_with_category.filter(pl.col('category') == category)
        
        if cat_positions.height == 0:
            print(f"{category}: No data")
            continue
        
        # Count by error type (use 'pred_type' column)
        tp = cat_positions.filter(pl.col('pred_type') == 'TP').height
        tn = cat_positions.filter(pl.col('pred_type') == 'TN').height
        fp = cat_positions.filter(pl.col('pred_type') == 'FP').height
        fn = cat_positions.filter(pl.col('pred_type') == 'FN').height
        
        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        num_genes = cat_positions.select(pl.col('gene_id').n_unique()).item()
        
        metrics = {
            'base_model': BASE_MODEL,
            'build': BUILD,
            'category': category,
            'num_genes': num_genes,
            'total_positions': cat_positions.height,
            'TP': tp,
            'TN': tn,
            'FP': fp,
            'FN': fn,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
        
        category_metrics.append(metrics)
        
        print(f"{category}:")
        print(f"  Genes: {num_genes}")
        print(f"  Positions: {cat_positions.height:,}")
        print(f"  TP: {tp:,}, TN: {tn:,}, FP: {fp:,}, FN: {fn:,}")
        print(f"  Precision: {precision:.3f}")
        print(f"  Recall: {recall:.3f}")
        print(f"  F1 Score: {f1:.3f}")
        print()
    
    # Save category metrics
    if category_metrics:
        category_metrics_df = pl.DataFrame(category_metrics)
        category_metrics_file = f"{output_dir}/category_performance_summary.tsv"
        category_metrics_df.write_csv(category_metrics_file, separator='\t')
        print(f"✅ Saved category metrics to: {category_metrics_file}")
        print()
    
    # Calculate overall metrics
    print("Overall Performance:")
    print("-" * 80)
    print()
    
    tp_total = positions_df.filter(pl.col('error_type') == 'TP').height
    tn_total = positions_df.filter(pl.col('error_type') == 'TN').height
    fp_total = positions_df.filter(pl.col('error_type') == 'FP').height
    fn_total = positions_df.filter(pl.col('error_type') == 'FN').height
    
    precision_total = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
    recall_total = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
    f1_total = 2 * (precision_total * recall_total) / (precision_total + recall_total) if (precision_total + recall_total) > 0 else 0.0
    
    print(f"Base Model: {BASE_MODEL}")
    print(f"Build: {BUILD}")
    print(f"Total Genes: {positions_df.select(pl.col('gene_id').n_unique()).item()}")
    print(f"Total Positions: {positions_df.height:,}")
    print(f"TP: {tp_total:,}, TN: {tn_total:,}, FP: {fp_total:,}, FN: {fn_total:,}")
    print(f"Precision: {precision_total:.3f}")
    print(f"Recall: {recall_total:.3f}")
    print(f"F1 Score: {f1_total:.3f}")
    print()

# ============================================================================
# STEP 4: Generate Summary Report
# ============================================================================
print("=" * 80)
print("SUMMARY")
print("=" * 80)
print()

print(f"Test: {TEST_NAME}")
print(f"Base Model: {BASE_MODEL} (GRCh38 MANE)")
print(f"Total Genes: {len(target_genes)}")
print(f"Total Positions: {results['positions'].height:,}")
print()

if positions_df.height > 0 and category_metrics:
    print("Performance by Category:")
    print(category_metrics_df)
    print()

print(f"Results saved to: {output_dir}/")
print()

print("=" * 80)
print("✅ TEST COMPLETE!")
print("=" * 80)
print()

print("Next Steps:")
print("1. Review category performance metrics")
print("2. Compare with SpliceAI results using compare_base_models.py")
print("3. Analyze differences between models")
print()

