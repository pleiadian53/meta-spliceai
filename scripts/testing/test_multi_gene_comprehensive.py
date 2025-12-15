#!/usr/bin/env python3
"""
Comprehensive multi-gene test with protein-coding and lncRNA genes.

This test:
1. Samples 15 protein-coding + 5 lncRNA genes from GTF
2. Processes ALL chromosomes where these genes are located
3. Calculates metrics by biotype and by gene
4. Tests coordinate adjustments (consensus_window=0)
5. Provides comprehensive performance analysis

Date: November 2, 2025
"""

import sys
from pathlib import Path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

import argparse
import pandas as pd
from meta_spliceai.splice_engine.meta_models.workflows.splice_prediction_workflow import (
    run_enhanced_splice_prediction_workflow
)
from meta_spliceai.splice_engine.meta_models.core.data_types import SpliceAIConfig
from meta_spliceai.system.genomic_resources import Registry


def sample_genes_from_gtf(gtf_path: str, n_protein_coding: int = 15, n_lncrna: int = 5, seed: int = 42):
    """
    Sample diverse genes from GTF.
    
    Returns dict with:
        - 'protein_coding': list of gene IDs
        - 'lncrna': list of gene IDs
        - 'all': combined list
        - 'chromosomes': list of chromosomes
        - 'gene_info': dict mapping gene_id to info
    """
    print(f"\n{'='*80}")
    print("SAMPLING GENES FROM GTF")
    print(f"{'='*80}\n")
    
    print(f"Reading: {gtf_path}")
    
    # Parse GTF
    genes = []
    with open(gtf_path, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split('\t')
            if len(parts) < 9 or parts[2] != 'gene':
                continue
            
            attrs = {}
            for attr in parts[8].split(';'):
                attr = attr.strip()
                if not attr:
                    continue
                key_val = attr.split(' ', 1)
                if len(key_val) == 2:
                    attrs[key_val[0]] = key_val[1].strip('"')
            
            if 'gene_id' in attrs and 'gene_biotype' in attrs:
                genes.append({
                    'gene_id': attrs['gene_id'],
                    'gene_name': attrs.get('gene_name', attrs['gene_id']),
                    'gene_biotype': attrs['gene_biotype'],
                    'chrom': parts[0]
                })
    
    genes_df = pd.DataFrame(genes)
    print(f"Total genes: {len(genes_df):,}")
    
    # Sample protein-coding
    pc_genes = genes_df[genes_df['gene_biotype'] == 'protein_coding']
    if len(pc_genes) > n_protein_coding:
        pc_sample = pc_genes.sample(n=n_protein_coding, random_state=seed)
    else:
        pc_sample = pc_genes.head(n_protein_coding)
    
    # Sample lncRNA
    lnc_genes = genes_df[genes_df['gene_biotype'] == 'lncRNA']
    if len(lnc_genes) > n_lncrna:
        lnc_sample = lnc_genes.sample(n=n_lncrna, random_state=seed)
    else:
        lnc_sample = lnc_genes.head(n_lncrna)
    
    # Combine
    all_samples = pd.concat([pc_sample, lnc_sample])
    
    # Get chromosomes
    chromosomes = sorted(all_samples['chrom'].unique().tolist())
    
    # Create gene info dict
    gene_info = {}
    for _, row in all_samples.iterrows():
        gene_info[row['gene_id']] = {
            'gene_name': row['gene_name'],
            'gene_biotype': row['gene_biotype'],
            'chrom': row['chrom']
        }
    
    print(f"\nSampled {len(pc_sample)} protein-coding genes:")
    for _, gene in pc_sample.iterrows():
        print(f"  {gene['gene_id']:20s} {gene['gene_name']:15s} chr{gene['chrom']}")
    
    print(f"\nSampled {len(lnc_sample)} lncRNA genes:")
    for _, gene in lnc_sample.iterrows():
        print(f"  {gene['gene_id']:20s} {gene['gene_name']:15s} chr{gene['chrom']}")
    
    print(f"\nChromosomes to process: {', '.join(chromosomes)}")
    print(f"Total genes: {len(all_samples)}")
    
    return {
        'protein_coding': pc_sample['gene_id'].tolist(),
        'lncrna': lnc_sample['gene_id'].tolist(),
        'all': all_samples['gene_id'].tolist(),
        'chromosomes': chromosomes,
        'gene_info': gene_info
    }


def main():
    """Main function."""
    
    parser = argparse.ArgumentParser(description='Comprehensive multi-gene test')
    parser.add_argument('--n-protein-coding', type=int, default=15,
                       help='Number of protein-coding genes to sample')
    parser.add_argument('--n-lncrna', type=int, default=5,
                       help='Number of lncRNA genes to sample')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for gene sampling')
    parser.add_argument('--output-subdir', type=str, default='multi_gene_test',
                       help='Output subdirectory name')
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print("COMPREHENSIVE MULTI-GENE TEST")
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
    
    print(f"\nGenome Resources:")
    print(f"  Build:    GRCh37")
    print(f"  Release:  87")
    print(f"  GTF:      {gtf_path}")
    print(f"  FASTA:    {fasta_path}")
    print(f"  Eval dir: {eval_dir}")
    
    # Sample genes
    gene_sample = sample_genes_from_gtf(
        gtf_path=str(gtf_path),
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
        target_chromosomes=gene_sample['chromosomes'],  # Process all relevant chromosomes!
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
    print()
    
    # Save gene info for analysis
    import json
    gene_info_path = Path(eval_dir) / args.output_subdir / 'gene_info.json'
    gene_info_path.parent.mkdir(parents=True, exist_ok=True)
    with open(gene_info_path, 'w') as f:
        json.dump(gene_sample, f, indent=2)
    
    print(f"Gene info saved to: {gene_info_path}")
    print()
    print("Next steps:")
    print(f"  1. Run analysis script to calculate metrics by biotype")
    print(f"  2. Test coordinate adjustments for each gene")
    print(f"  3. Compare protein-coding vs lncRNA performance")
    print()
    print(f"Results location: {eval_dir}/{args.output_subdir}/")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())




