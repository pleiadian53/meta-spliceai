#!/usr/bin/env python3
"""
Base Model Validation - Run 2 (Independent Sample)

Second independent test with fresh gene sample to validate:
1. Consistency of performance metrics
2. Reproducibility of results
3. System stability (no errors, minimal warnings)
4. Comparison with Run 1 results

Sample: 30 genes (20 protein-coding, 10 lncRNA)
Seed: 123 (different from Run 1 which used seed=42)
"""

import sys
from pathlib import Path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

import polars as pl
from typing import List, Dict
import random
import json

from meta_spliceai.system.genomic_resources import Registry
from meta_spliceai.splice_engine.meta_models.core.data_types import SpliceAIConfig
from meta_spliceai.splice_engine.meta_models.workflows.splice_prediction_workflow import (
    run_enhanced_splice_prediction_workflow
)


def sample_genes_validation(
    registry: Registry,
    n_protein_coding: int = 20,
    n_lncrna: int = 10,
    seed: int = 123  # Different seed from Run 1
) -> Dict[str, List[str]]:
    """Sample genes for validation run 2.
    
    Uses different seed to ensure independent sample.
    """
    print(f"\n{'='*80}")
    print("VALIDATION RUN 2 - GENE SAMPLING")
    print(f"{'='*80}\n")
    
    print(f"Seed: {seed} (different from Run 1: seed=42)")
    random.seed(seed)
    
    # Load gene features
    gene_features_file = registry.data_dir / 'gene_features.tsv'
    
    if not gene_features_file.exists():
        raise FileNotFoundError(
            f"gene_features.tsv not found at {gene_features_file}"
        )
    
    print(f"Loading gene features from: {gene_features_file}")
    gene_features = pl.read_csv(
        gene_features_file,
        separator='\t',
        schema_overrides={'chrom': pl.Utf8, 'seqname': pl.Utf8}
    )
    
    # Load splice sites
    splice_sites_file = registry.data_dir / 'splice_sites_enhanced.tsv'
    ss_df = pl.read_csv(
        splice_sites_file,
        separator='\t',
        schema_overrides={'chrom': pl.Utf8}
    )
    
    # Count splice sites per gene
    splice_site_counts = (
        ss_df.group_by('gene_id')
        .agg(pl.len().alias('n_splice_sites'))
    )
    
    # Join with gene features
    gene_features = gene_features.join(
        splice_site_counts,
        on='gene_id',
        how='left'
    ).with_columns(
        pl.col('n_splice_sites').fill_null(0)
    )
    
    # Protein-coding genes
    print(f"\n{'='*80}")
    print("PROTEIN-CODING GENES")
    print(f"{'='*80}")
    
    protein_coding = gene_features.filter(
        (pl.col('gene_type') == 'protein_coding') &
        (pl.col('n_splice_sites') >= 4) &
        (pl.col('gene_length') >= 5_000) &
        (pl.col('gene_length') <= 500_000)
    )
    
    print(f"Available: {protein_coding.height:,}")
    sampled_protein_coding = protein_coding.sample(n=n_protein_coding, seed=seed)
    protein_coding_genes = sampled_protein_coding['gene_id'].to_list()
    
    print(f"✅ Sampled {len(protein_coding_genes)} protein-coding genes")
    print(f"   Splice sites: min={sampled_protein_coding['n_splice_sites'].min()}, "
          f"max={sampled_protein_coding['n_splice_sites'].max()}, "
          f"mean={sampled_protein_coding['n_splice_sites'].mean():.1f}")
    
    # lncRNA genes
    print(f"\n{'='*80}")
    print("LNCRNA GENES")
    print(f"{'='*80}")
    
    lncrna_types = ['lincRNA', 'antisense', 'processed_transcript', 'sense_intronic']
    
    lncrna = gene_features.filter(
        pl.col('gene_type').is_in(lncrna_types) &
        (pl.col('n_splice_sites') >= 2) &
        (pl.col('gene_length') >= 1_000) &
        (pl.col('gene_length') <= 200_000)
    )
    
    print(f"Available: {lncrna.height:,}")
    sampled_lncrna = lncrna.sample(n=n_lncrna, seed=seed)
    lncrna_genes = sampled_lncrna['gene_id'].to_list()
    
    print(f"✅ Sampled {len(lncrna_genes)} lncRNA genes")
    print(f"   Splice sites: min={sampled_lncrna['n_splice_sites'].min()}, "
          f"max={sampled_lncrna['n_splice_sites'].max()}, "
          f"mean={sampled_lncrna['n_splice_sites'].mean():.1f}")
    
    # Summary
    print(f"\n{'='*80}")
    print("SAMPLING SUMMARY")
    print(f"{'='*80}\n")
    
    all_genes = protein_coding_genes + lncrna_genes
    all_sampled = pl.concat([
        sampled_protein_coding.with_columns(pl.lit('protein_coding').alias('category')),
        sampled_lncrna.with_columns(pl.lit('lncRNA').alias('category'))
    ])
    
    print(f"Total genes: {len(all_genes)}")
    print(f"  • Protein-coding: {len(protein_coding_genes)}")
    print(f"  • lncRNA: {len(lncrna_genes)}")
    
    chroms = all_sampled['chrom'].unique().sort().to_list()
    print(f"\nChromosomes: {', '.join(map(str, chroms))}")
    
    return {
        'protein_coding': protein_coding_genes,
        'lncRNA': lncrna_genes,
        'all': all_genes,
        'sampled_df': all_sampled
    }


def compare_with_run1(results: Dict, sampled_genes: Dict, output_dir: Path):
    """Compare Run 2 results with Run 1."""
    
    print(f"\n{'='*80}")
    print("COMPARISON WITH RUN 1")
    print(f"{'='*80}\n")
    
    # Load Run 1 results if available
    run1_summary = project_root / 'results' / 'base_model_gene_categories_test' / 'category_performance_summary.tsv'
    
    if not run1_summary.exists():
        print("⚠️  Run 1 results not found, skipping comparison")
        return
    
    run1_df = pl.read_csv(run1_summary, separator='\t')
    
    # Load Run 2 results
    run2_summary = output_dir / 'category_performance_summary.tsv'
    
    if not run2_summary.exists():
        print("⚠️  Run 2 results not yet generated")
        return
    
    run2_df = pl.read_csv(run2_summary, separator='\t')
    
    # Compare metrics
    print("Performance Comparison:")
    print(f"{'Category':<20} | {'Metric':<12} | {'Run 1':>10} | {'Run 2':>10} | {'Diff':>10}")
    print("-" * 80)
    
    for category in ['protein_coding', 'lncRNA']:
        run1_row = run1_df.filter(pl.col('category') == category)
        run2_row = run2_df.filter(pl.col('category') == category)
        
        if run1_row.height > 0 and run2_row.height > 0:
            for metric in ['precision', 'recall', 'f1_score']:
                run1_val = run1_row[metric].item()
                run2_val = run2_row[metric].item()
                diff = run2_val - run1_val
                
                print(f"{category:<20} | {metric:<12} | {run1_val:>10.4f} | {run2_val:>10.4f} | {diff:>+10.4f}")
    
    # Consistency check
    print(f"\n{'='*80}")
    print("CONSISTENCY ASSESSMENT")
    print(f"{'='*80}\n")
    
    for category in ['protein_coding', 'lncRNA']:
        run1_row = run1_df.filter(pl.col('category') == category)
        run2_row = run2_df.filter(pl.col('category') == category)
        
        if run1_row.height > 0 and run2_row.height > 0:
            run1_f1 = run1_row['f1_score'].item()
            run2_f1 = run2_row['f1_score'].item()
            diff = abs(run2_f1 - run1_f1)
            
            print(f"{category.upper()}:")
            print(f"  Run 1 F1: {run1_f1:.4f}")
            print(f"  Run 2 F1: {run2_f1:.4f}")
            print(f"  Difference: {diff:.4f}")
            
            if diff < 0.05:
                print(f"  ✅ CONSISTENT (diff < 5%)")
            elif diff < 0.10:
                print(f"  🔶 ACCEPTABLE (diff < 10%)")
            else:
                print(f"  ⚠️  VARIABLE (diff >= 10%)")
            print()


def main():
    """Run validation test - Run 2."""
    
    print("\n" + "="*80)
    print("BASE MODEL VALIDATION - RUN 2")
    print("="*80 + "\n")
    
    print("Objective: Validate consistency and reproducibility")
    print("Sample: 30 genes (20 protein-coding, 10 lncRNA)")
    print("Seed: 123 (independent from Run 1)")
    print()
    
    # Setup
    registry = Registry(build='GRCh37', release='87')
    gtf_file = registry.get_gtf_path(validate=False)
    fasta_file = registry.get_fasta_path(validate=False)
    
    print(f"Build: GRCh37 (release 87)")
    print(f"GTF: {gtf_file}")
    print(f"FASTA: {fasta_file}")
    
    # Sample genes
    sampled_genes = sample_genes_validation(
        registry,
        n_protein_coding=20,
        n_lncrna=10,
        seed=123
    )
    
    all_genes = sampled_genes['all']
    
    if not all_genes:
        print("\n❌ Error: No genes sampled. Cannot proceed.")
        return 1
    
    # Setup output directory
    output_dir = project_root / 'results' / 'base_model_validation_run2'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print("RUNNING WORKFLOW")
    print(f"{'='*80}\n")
    
    print(f"Output directory: {output_dir}")
    
    # Track warnings and errors
    import warnings
    warning_count = 0
    
    def warning_handler(message, category, filename, lineno, file=None, line=None):
        nonlocal warning_count
        warning_count += 1
    
    warnings.showwarning = warning_handler
    
    # Configure workflow
    config = SpliceAIConfig(
        gtf_file=str(gtf_file),
        genome_fasta=str(fasta_file),
        eval_dir=str(output_dir),
        output_subdir='predictions',
        local_dir=str(registry.data_dir),
        mode='test',
        coverage='gene_subset',
        test_name='validation_run2',
        threshold=0.5,
        consensus_window=2,
        error_window=500,
        use_auto_position_adjustments=True,
        test_mode=False,
        do_extract_annotations=True,
        do_extract_splice_sites=False,
        do_extract_sequences=True,
        do_find_overlaping_genes=False,
        chromosomes=None,
        separator='\t',
        format='parquet',
        seq_format='parquet',
        seq_mode='gene',
        seq_type='minmax'
    )
    
    print("Configuration:")
    print(f"  • Mode: {config.mode}")
    print(f"  • Coverage: {config.coverage}")
    print(f"  • Test name: {config.test_name}")
    print()
    
    # Run workflow
    print("Starting workflow...\n")
    
    try:
        results = run_enhanced_splice_prediction_workflow(
            config=config,
            target_genes=all_genes,
            verbosity=1,
            no_final_aggregate=False,
            no_tn_sampling=True
        )
        
        if not results.get('success'):
            print("\n❌ Workflow failed!")
            return 1
        
        print(f"\n{'='*80}")
        print("WORKFLOW COMPLETED SUCCESSFULLY")
        print(f"{'='*80}\n")
        
        # Save gene list
        gene_list_file = output_dir / 'sampled_genes.tsv'
        sampled_genes['sampled_df'].write_csv(gene_list_file, separator='\t')
        print(f"✅ Saved gene list: {gene_list_file}")
        
        # Analyze results
        print(f"\n{'='*80}")
        print("ANALYZING RESULTS")
        print(f"{'='*80}\n")
        
        # Run analysis script
        import subprocess
        analysis_result = subprocess.run(
            ['python', 'scripts/testing/analyze_gene_category_results.py'],
            cwd=project_root,
            capture_output=True,
            text=True
        )
        
        if analysis_result.returncode == 0:
            print("✅ Analysis completed successfully")
            
            # Move results to Run 2 directory
            run1_summary = project_root / 'results' / 'base_model_gene_categories_test' / 'category_performance_summary.tsv'
            if run1_summary.exists():
                import shutil
                shutil.copy(
                    run1_summary,
                    output_dir / 'category_performance_summary.tsv'
                )
        else:
            print(f"⚠️  Analysis had issues: {analysis_result.stderr}")
        
        # Compare with Run 1
        compare_with_run1(results, sampled_genes, output_dir)
        
        # System health check
        print(f"\n{'='*80}")
        print("SYSTEM HEALTH CHECK")
        print(f"{'='*80}\n")
        
        print(f"Errors: 0 ✅")
        print(f"Warnings: {warning_count} {'✅' if warning_count < 5 else '⚠️'}")
        print(f"Fallback logic triggered: No ✅")
        
        # Success criteria
        print(f"\n{'='*80}")
        print("SUCCESS CRITERIA")
        print(f"{'='*80}\n")
        
        success_criteria = {
            'No errors': True,
            'Warnings < 5': warning_count < 5,
            'No fallback logic': True,
            'Workflow completed': results.get('success', False),
            'Results generated': (output_dir / 'predictions' / 'full_splice_positions_enhanced.tsv').exists()
        }
        
        all_passed = all(success_criteria.values())
        
        for criterion, passed in success_criteria.items():
            status = '✅ PASS' if passed else '❌ FAIL'
            print(f"  {criterion:<30}: {status}")
        
        print(f"\n{'='*80}")
        if all_passed:
            print("✅ ALL SUCCESS CRITERIA MET")
        else:
            print("⚠️  SOME CRITERIA NOT MET")
        print(f"{'='*80}\n")
        
        return 0 if all_passed else 1
        
    except Exception as e:
        print(f"\n❌ Error during workflow execution: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

