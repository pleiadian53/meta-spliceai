#!/usr/bin/env python3
"""
Comprehensive Base Model Test - Protein-coding and lncRNA Genes

Tests the base model (SpliceAI) with a diverse set of genes:
- 15 protein-coding genes (expected good performance)
- 5 lncRNA genes (explore performance on non-coding genes)

Validates:
- Coordinate alignment across chromosomes
- Splice site prediction accuracy
- Performance by gene biotype
- Comparison with SpliceAI paper benchmarks
"""

import sys
from pathlib import Path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

import polars as pl
from typing import List, Dict
import random

from meta_spliceai.system.genomic_resources import Registry
from meta_spliceai.splice_engine.meta_models.core.data_types import SpliceAIConfig
from meta_spliceai.splice_engine.meta_models.workflows.splice_prediction_workflow import (
    run_enhanced_splice_prediction_workflow
)


def sample_genes_by_biotype(
    registry: Registry,
    n_protein_coding: int = 15,
    n_lncrna: int = 5,
    seed: int = 42
) -> Dict[str, List[str]]:
    """Sample genes by biotype using gene_features.tsv.
    
    Uses the same approach as prepare_gene_lists.py for consistency.
    
    Parameters
    ----------
    registry : Registry
        Registry for resolving paths
    n_protein_coding : int
        Number of protein-coding genes to sample
    n_lncrna : int
        Number of lncRNA genes to sample
    seed : int
        Random seed for reproducibility
        
    Returns
    -------
    Dict[str, List[str]]
        Dictionary with 'protein_coding' and 'lncRNA' gene lists
    """
    print(f"\n{'='*80}")
    print("SAMPLING GENES BY BIOTYPE")
    print(f"{'='*80}\n")
    
    # Load gene features (has gene_type column)
    # gene_features.tsv should be build-specific: data/<source>/<build>/gene_features.tsv
    gene_features_file = registry.data_dir / 'gene_features.tsv'
    
    if not gene_features_file.exists():
        raise FileNotFoundError(
            f"gene_features.tsv not found at {gene_features_file}\n"
            f"This file is build-specific and should exist at data/<source>/<build>/gene_features.tsv\n"
            f"Please generate it first by running:\n"
            f"  python scripts/setup/generate_grch37_gene_features.py"
        )
    
    print(f"Loading gene features from: {gene_features_file}")
    gene_features = pl.read_csv(
        gene_features_file,
        separator='\t',
        schema_overrides={'chrom': pl.Utf8, 'seqname': pl.Utf8}
    )
    
    print(f"Total genes: {gene_features.height:,}")
    
    # Load splice sites to filter to genes with splice sites
    splice_sites_file = registry.data_dir / 'splice_sites_enhanced.tsv'
    ss_df = pl.read_csv(
        splice_sites_file,
        separator='\t',
        schema_overrides={'chrom': pl.Utf8}
    )
    genes_with_splice_sites = set(ss_df['gene_id'].unique().to_list())
    print(f"Genes with splice sites: {len(genes_with_splice_sites):,}")
    
    # Filter to genes with splice sites
    gene_features = gene_features.filter(
        pl.col('gene_id').is_in(list(genes_with_splice_sites))
    )
    
    print(f"\nGene biotype distribution (genes with splice sites):")
    biotype_counts = gene_features.group_by('gene_type').agg(pl.count()).sort('count', descending=True)
    for row in biotype_counts.head(10).iter_rows(named=True):
        print(f"  {row['gene_type']:30s}: {row['count']:6,} genes")
    
    # Sample protein-coding genes
    protein_coding_genes = gene_features.filter(
        (pl.col('gene_type') == 'protein_coding') &
        (pl.col('gene_length') >= 5000) &  # Reasonable size
        (pl.col('gene_length') <= 500000)
    )
    
    print(f"\nProtein-coding genes (5kb-500kb): {protein_coding_genes.height:,}")
    
    if protein_coding_genes.height < n_protein_coding:
        print(f"⚠️  Warning: Only {protein_coding_genes.height} suitable protein-coding genes")
        n_protein_coding = protein_coding_genes.height
    
    if protein_coding_genes.height > 0:
        sampled_protein_coding = protein_coding_genes.sample(n_protein_coding, seed=seed)
        protein_coding_list = sampled_protein_coding['gene_id'].to_list()
    else:
        protein_coding_list = []
    
    # Sample lncRNA genes
    # Try different gene_type variations that may exist in GRCh37
    lncrna_genes = gene_features.filter(
        (pl.col('gene_type').str.contains('(?i)lncRNA|lincRNA|long_noncoding')) &
        (pl.col('gene_length') >= 500) &  # More lenient for lncRNAs
        (pl.col('gene_length') <= 500000)
    )
    
    print(f"lncRNA genes (1kb-200kb): {lncrna_genes.height:,}")
    
    if lncrna_genes.height < n_lncrna:
        print(f"⚠️  Warning: Only {lncrna_genes.height} suitable lncRNA genes")
        n_lncrna = lncrna_genes.height
    
    if lncrna_genes.height > 0:
        sampled_lncrna = lncrna_genes.sample(n_lncrna, seed=seed)
        lncrna_list = sampled_lncrna['gene_id'].to_list()
    else:
        lncrna_list = []
    
    print(f"\n✅ Sampled {len(protein_coding_list)} protein-coding genes")
    print(f"✅ Sampled {len(lncrna_list)} lncRNA genes")
    
    # Show which chromosomes are represented
    all_sampled_genes = protein_coding_list + lncrna_list
    if all_sampled_genes:
        sampled_gene_features = gene_features.filter(
            pl.col('gene_id').is_in(all_sampled_genes)
        )
        chroms = sampled_gene_features['chrom'].unique().sort()
        print(f"\nChromosomes represented: {', '.join(chroms.to_list())}")
    
    return {
        'protein_coding': protein_coding_list,
        'lncRNA': lncrna_list
    }


def main():
    """Run comprehensive base model test."""
    
    print(f"\n{'='*80}")
    print("COMPREHENSIVE BASE MODEL TEST")
    print(f"{'='*80}\n")
    
    print("Testing SpliceAI base model with:")
    print("  • 15 protein-coding genes (expect good performance)")
    print("  • 5 lncRNA genes (explore performance)")
    print("  • Automatic coordinate adjustment detection")
    print("  • Genome-wide sampling")
    print()
    
    # Setup paths
    build = 'GRCh37'
    release = '87'
    registry = Registry(build=build, release=release)
    
    gtf_file = registry.resolve('gtf')
    fasta_file = registry.resolve('fasta')
    splice_sites_file = registry.data_dir / 'splice_sites_enhanced.tsv'
    
    print(f"Build: {build} (release {release})")
    print(f"GTF: {gtf_file}")
    print(f"FASTA: {fasta_file}")
    print(f"Splice sites: {splice_sites_file}")
    
    # Verify splice sites file exists
    if not splice_sites_file.exists():
        print(f"\n❌ Error: Splice sites file not found: {splice_sites_file}")
        print("Please run the splice site extraction first.")
        return 1
    
    # Sample genes by biotype
    sampled_genes = sample_genes_by_biotype(
        registry,
        n_protein_coding=15,
        n_lncrna=5,
        seed=42
    )
    
    all_genes = sampled_genes['protein_coding'] + sampled_genes['lncRNA']
    
    if not all_genes:
        print("\n❌ Error: No genes sampled. Cannot proceed.")
        return 1
    
    print(f"\n{'='*80}")
    print("RUNNING BASE MODEL PREDICTIONS")
    print(f"{'='*80}\n")
    
    # Setup output directory
    output_dir = project_root / 'results' / 'base_model_comprehensive_test'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    print()
    
    # Configure workflow
    # Use registry.data_dir as local_dir to ensure correct paths
    config = SpliceAIConfig(
        gtf_file=str(gtf_file),
        genome_fasta=str(fasta_file),
        eval_dir=str(output_dir),
        output_subdir='predictions',
        local_dir=str(registry.data_dir),  # Use data directory for genomic resources
        mode='test',  # Test mode: artifacts are overwritable
        coverage='gene_subset',  # Testing 20 genes
        test_name='base_model_comprehensive_test',  # Named test
        threshold=0.5,
        consensus_window=2,
        error_window=500,
        use_auto_position_adjustments=True,  # Enable automatic adjustment detection
        test_mode=False,
        do_extract_annotations=True,   # Extract fresh (fast for 20 genes)
        do_extract_splice_sites=False, # Use existing from data_dir
        do_extract_sequences=True,     # Extract for target genes
        do_find_overlaping_genes=False,
        chromosomes=None,  # Will be inferred from genes
        separator='\t',
        format='parquet',
        seq_format='parquet',
        seq_mode='gene',
        seq_type='minmax'
    )
    
    print("Configuration:")
    print(f"  • Threshold: {config.threshold}")
    print(f"  • Consensus window: {config.consensus_window}")
    print(f"  • Error window: {config.error_window}")
    print(f"  • Auto adjustments: {config.use_auto_position_adjustments}")
    print()
    
    # Run workflow
    print("Starting workflow...\n")
    
    try:
        results = run_enhanced_splice_prediction_workflow(
            config=config,
            target_genes=all_genes,
            verbosity=1,
            no_final_aggregate=False,
            no_tn_sampling=True  # Keep all positions for analysis
        )
        
        if not results.get('success'):
            print("\n❌ Workflow failed!")
            return 1
        
        print(f"\n{'='*80}")
        print("WORKFLOW COMPLETED SUCCESSFULLY")
        print(f"{'='*80}\n")
        
        # Save gene list for analysis
        gene_list_file = output_dir / 'sampled_genes.tsv'
        gene_list_df = pl.DataFrame({
            'gene_id': all_genes,
            'biotype': (
                ['protein_coding'] * len(sampled_genes['protein_coding']) +
                ['lncRNA'] * len(sampled_genes['lncRNA'])
            )
        })
        gene_list_df.write_csv(gene_list_file, separator='\t')
        print(f"✅ Saved gene list: {gene_list_file}")
        
        # Summary
        if 'positions' in results and results['positions'] is not None:
            positions_df = results['positions']
            print(f"\nPositions analyzed: {positions_df.height:,}")
            print(f"Genes processed: {positions_df['gene_id'].n_unique():,}")
            
            if 'pred_type' in positions_df.columns:
                pred_counts = positions_df.group_by('pred_type').agg(pl.count())
                print("\nPrediction type distribution:")
                for row in pred_counts.iter_rows(named=True):
                    print(f"  {row['pred_type']:4s}: {row['count']:8,}")
        
        print(f"\n{'='*80}")
        print("NEXT STEPS")
        print(f"{'='*80}\n")
        print("Run analysis script to evaluate results:")
        print(f"  python scripts/testing/analyze_base_model_comprehensive.py")
        print()
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error during workflow execution: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())

