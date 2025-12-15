#!/usr/bin/env python3
"""
Comprehensive multi-gene test with VALIDATED GRCh37 genes.

This test:
1. Samples genes that ACTUALLY EXIST in GRCh37 with splice sites
2. Validates genes have both donor and acceptor sites
3. Processes ALL chromosomes where these genes are located
4. Calculates metrics by biotype and by gene

Date: November 2, 2025
"""

import sys
from pathlib import Path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

import argparse
import polars as pl
from meta_spliceai.splice_engine.meta_models.workflows.splice_prediction_workflow import (
    run_enhanced_splice_prediction_workflow
)
from meta_spliceai.splice_engine.meta_models.core.data_types import SpliceAIConfig
from meta_spliceai.system.genomic_resources import Registry


def sample_validated_genes(splice_sites_path: str, n_protein_coding: int = 15, n_lncrna: int = 5, seed: int = 42):
    """
    Sample genes that have both donor and acceptor sites in the annotation.
    """
    print(f"\n{'='*80}")
    print("SAMPLING VALIDATED GENES FROM SPLICE SITES")
    print(f"{'='*80}\n")
    
    print(f"Reading: {splice_sites_path}")
    
    # Load splice sites
    df = pl.read_csv(splice_sites_path, separator='\t')
    
    print(f"Total splice sites: {df.height:,}")
    print(f"Unique genes: {df['gene_id'].n_unique()}")
    print()
    
    # Count donors and acceptors per gene
    gene_stats = df.group_by(['gene_id', 'gene_name', 'gene_type', 'chrom']).agg([
        pl.col('site_type').filter(pl.col('site_type') == 'donor').count().alias('n_donors'),
        pl.col('site_type').filter(pl.col('site_type') == 'acceptor').count().alias('n_acceptors'),
        pl.col('site_type').count().alias('n_sites')
    ])
    
    # Filter for genes with BOTH donors and acceptors (at least 2 of each)
    valid_genes = gene_stats.filter(
        (pl.col('n_donors') >= 2) &
        (pl.col('n_acceptors') >= 2)
    )
    
    print(f"Genes with ≥2 donors AND ≥2 acceptors: {valid_genes.height:,}")
    print()
    
    # Sample protein-coding
    pc_genes = valid_genes.filter(pl.col('gene_type') == 'protein_coding')
    print(f"Protein-coding genes available: {pc_genes.height:,}")
    
    if pc_genes.height > n_protein_coding:
        pc_sample = pc_genes.sample(n=n_protein_coding, seed=seed)
    else:
        pc_sample = pc_genes.head(n_protein_coding)
        print(f"  Warning: Only {pc_sample.height} protein-coding genes available")
    
    # Sample lncRNA
    lnc_genes = valid_genes.filter(pl.col('gene_type') == 'lncRNA')
    print(f"lncRNA genes available: {lnc_genes.height:,}")
    
    if lnc_genes.height > n_lncrna:
        lnc_sample = lnc_genes.sample(n=n_lncrna, seed=seed)
    else:
        lnc_sample = lnc_genes.head(n_lncrna)
        print(f"  Warning: Only {lnc_sample.height} lncRNA genes available")
    
    # Combine
    all_samples = pl.concat([pc_sample, lnc_sample])
    
    # Get chromosomes
    chromosomes = sorted([str(c) for c in all_samples['chrom'].unique().to_list()])
    
    # Create gene info dict
    gene_info = {}
    for row in all_samples.iter_rows(named=True):
        gene_info[row['gene_id']] = {
            'gene_name': row['gene_name'],
            'gene_biotype': row['gene_type'],
            'chrom': str(row['chrom']),
            'n_donors': row['n_donors'],
            'n_acceptors': row['n_acceptors'],
            'n_sites': row['n_sites']
        }
    
    print(f"\nSampled {pc_sample.height} protein-coding genes:")
    for row in pc_sample.iter_rows(named=True):
        print(f"  {row['gene_id']:20s} {row['gene_name']:15s} chr{row['chrom']:3s} "
              f"({row['n_donors']:2d} donors, {row['n_acceptors']:2d} acceptors)")
    
    print(f"\nSampled {lnc_sample.height} lncRNA genes:")
    for row in lnc_sample.iter_rows(named=True):
        print(f"  {row['gene_id']:20s} {row['gene_name']:15s} chr{row['chrom']:3s} "
              f"({row['n_donors']:2d} donors, {row['n_acceptors']:2d} acceptors)")
    
    print(f"\nChromosomes to process: {', '.join(chromosomes)}")
    print(f"Total genes: {all_samples.height}")
    
    return {
        'protein_coding': pc_sample['gene_id'].to_list(),
        'lncrna': lnc_sample['gene_id'].to_list(),
        'all': all_samples['gene_id'].to_list(),
        'chromosomes': chromosomes,
        'gene_info': gene_info
    }


def main():
    """Main function."""
    
    parser = argparse.ArgumentParser(description='Comprehensive multi-gene test with validated genes')
    parser.add_argument('--n-protein-coding', type=int, default=15,
                       help='Number of protein-coding genes to sample')
    parser.add_argument('--n-lncrna', type=int, default=5,
                       help='Number of lncRNA genes to sample')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for gene sampling')
    parser.add_argument('--output-subdir', type=str, default='multi_gene_validated',
                       help='Output subdirectory name')
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print("COMPREHENSIVE MULTI-GENE TEST (VALIDATED GENES)")
    print(f"{'='*80}\n")
    
    print(f"Configuration:")
    print(f"  Protein-coding genes: {args.n_protein_coding}")
    print(f"  lncRNA genes:         {args.n_lncrna}")
    print(f"  Random seed:          {args.seed}")
    print(f"  Output subdir:        {args.output_subdir}")
    
    # Set up paths
    registry = Registry(build='GRCh37', release='87')
    gtf_path = registry.resolve('gtf')
    fasta_path = registry.resolve('fasta')
    eval_dir = registry.eval_dir
    splice_sites_path = registry.resolve('splice_sites_enhanced')
    
    print(f"\nGenome Resources:")
    print(f"  Build:        GRCh37")
    print(f"  Release:      87")
    print(f"  GTF:          {gtf_path}")
    print(f"  FASTA:        {fasta_path}")
    print(f"  Splice sites: {splice_sites_path}")
    print(f"  Eval dir:     {eval_dir}")
    
    # Sample genes from splice sites (ensures they exist!)
    gene_sample = sample_validated_genes(
        splice_sites_path=str(splice_sites_path),
        n_protein_coding=args.n_protein_coding,
        n_lncrna=args.n_lncrna,
        seed=args.seed
    )
    
    # Create config
    config = SpliceAIConfig(
        gtf_file=str(gtf_path),
        genome_fasta=str(fasta_path),
        eval_dir=str(eval_dir),
        output_subdir=args.output_subdir,
        threshold=0.5,
        consensus_window=2,
        error_window=500,
        test_mode=False,
        
        # Use existing data (already generated for GRCh37)
        do_extract_annotations=False,
        do_extract_splice_sites=False,
        do_extract_sequences=False,
        do_find_overlaping_genes=False,
        
        # Enable automatic adjustment detection
        use_auto_position_adjustments=True,
        
        # Build-specific local directory
        local_dir=str(registry.data_dir),
        
        # Format settings
        format='parquet',
        seq_format='parquet',
        separator='\t'
    )
    
    print(f"\n{'='*80}")
    print("RUNNING SPLICE PREDICTION WORKFLOW")
    print(f"{'='*80}\n")
    
    print(f"Processing {len(gene_sample['chromosomes'])} chromosomes:")
    print(f"  {', '.join(gene_sample['chromosomes'])}")
    print()
    print(f"Target genes: {len(gene_sample['all'])}")
    print(f"  Protein-coding: {len(gene_sample['protein_coding'])}")
    print(f"  lncRNA:         {len(gene_sample['lncrna'])}")
    print()
    print("This may take 10-30 minutes depending on the number of genes...")
    print()
    
    # Run workflow
    results = run_enhanced_splice_prediction_workflow(
        config=config,
        target_genes=gene_sample['all'],
        target_chromosomes=gene_sample['chromosomes'],
        verbosity=1,
        no_final_aggregate=False,
        no_tn_sampling=False,
    )
    
    if not results.get('success'):
        print("\n[ERROR] Workflow failed!")
        return 1
    
    print(f"\n{'='*80}")
    print("WORKFLOW COMPLETED SUCCESSFULLY")
    print(f"{'='*80}\n")
    
    positions_df = results.get('positions')
    
    if positions_df is None or positions_df.height == 0:
        print("[ERROR] No predictions generated!")
        return 1
    
    print(f"Total positions: {positions_df.height:,}")
    print(f"Unique genes processed: {positions_df['gene_id'].n_unique()}")
    
    # Check which genes were actually processed
    processed_genes = positions_df['gene_id'].unique().to_list()
    expected_genes = gene_sample['all']
    
    print()
    print(f"Expected genes: {len(expected_genes)}")
    print(f"Processed genes: {len(processed_genes)}")
    
    if len(processed_genes) < len(expected_genes):
        missing = set(expected_genes) - set(processed_genes)
        print(f"\nWarning: {len(missing)} genes were not processed:")
        for gene_id in missing:
            info = gene_sample['gene_info'][gene_id]
            print(f"  {gene_id} ({info['gene_name']}) on chr{info['chrom']}")
    
    # Save gene info for analysis
    import json
    gene_info_path = Path(eval_dir) / args.output_subdir / 'gene_info.json'
    gene_info_path.parent.mkdir(parents=True, exist_ok=True)
    with open(gene_info_path, 'w') as f:
        json.dump(gene_sample, f, indent=2)
    
    print(f"\nGene info saved to: {gene_info_path}")
    print()
    print("Next steps:")
    print(f"  python scripts/testing/analyze_multi_gene_results.py --results-dir {eval_dir}/{args.output_subdir}")
    print()
    print(f"Results location: {eval_dir}/{args.output_subdir}/")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())




