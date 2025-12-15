#!/usr/bin/env python3
"""
Test base model (SpliceAI) with diverse gene set including lncRNAs.

This script tests the splice prediction workflow with:
1. Protein-coding genes (expected high performance)
2. lncRNA genes (interesting to see how SpliceAI performs)
3. Mixed biotypes

Goal: Verify that the base model pass still works correctly after the
multi-build system update, and explore performance on different gene types.

Date: November 2, 2025
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from meta_spliceai.splice_engine.meta_models.workflows.splice_prediction_workflow import (
    run_enhanced_splice_prediction_workflow
)
from meta_spliceai.splice_engine.meta_models.core.data_types import SpliceAIConfig
from meta_spliceai.system.genomic_resources import Registry
import polars as pl
import pandas as pd


def get_diverse_gene_sample(gtf_path: str, n_protein_coding: int = 15, n_lncrna: int = 5) -> dict:
    """
    Sample diverse genes from GTF including protein-coding and lncRNA.
    
    Returns
    -------
    dict with keys:
        - 'protein_coding': list of protein-coding gene IDs
        - 'lncrna': list of lncRNA gene IDs
        - 'all': combined list
    """
    print(f"\n{'='*80}")
    print("SAMPLING DIVERSE GENES FROM GTF")
    print(f"{'='*80}\n")
    
    print(f"Reading GTF: {gtf_path}")
    
    # Read GTF and extract gene information
    genes = []
    with open(gtf_path, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split('\t')
            if len(parts) < 9 or parts[2] != 'gene':
                continue
            
            # Parse attributes
            attrs = {}
            for attr in parts[8].split(';'):
                attr = attr.strip()
                if not attr:
                    continue
                key_val = attr.split(' ', 1)
                if len(key_val) == 2:
                    key = key_val[0]
                    val = key_val[1].strip('"')
                    attrs[key] = val
            
            if 'gene_id' in attrs and 'gene_biotype' in attrs:
                genes.append({
                    'gene_id': attrs['gene_id'],
                    'gene_name': attrs.get('gene_name', attrs['gene_id']),
                    'gene_biotype': attrs['gene_biotype'],
                    'chrom': parts[0]
                })
    
    genes_df = pd.DataFrame(genes)
    print(f"Total genes in GTF: {len(genes_df):,}")
    
    # Count by biotype
    biotype_counts = genes_df['gene_biotype'].value_counts()
    print(f"\nTop 10 gene biotypes:")
    for biotype, count in biotype_counts.head(10).items():
        print(f"  {biotype:30s}: {count:6,}")
    
    # Sample protein-coding genes
    pc_genes = genes_df[genes_df['gene_biotype'] == 'protein_coding']
    if len(pc_genes) > n_protein_coding:
        pc_sample = pc_genes.sample(n=n_protein_coding, random_state=42)
    else:
        pc_sample = pc_genes
    
    print(f"\nSampled {len(pc_sample)} protein-coding genes:")
    for _, gene in pc_sample.iterrows():
        print(f"  {gene['gene_id']:20s} {gene['gene_name']:15s} chr{gene['chrom']}")
    
    # Sample lncRNA genes
    lnc_genes = genes_df[genes_df['gene_biotype'] == 'lncRNA']
    if len(lnc_genes) > n_lncrna:
        lnc_sample = lnc_genes.sample(n=n_lncrna, random_state=42)
    else:
        lnc_sample = lnc_genes.head(n_lncrna)
    
    print(f"\nSampled {len(lnc_sample)} lncRNA genes:")
    for _, gene in lnc_sample.iterrows():
        print(f"  {gene['gene_id']:20s} {gene['gene_name']:15s} chr{gene['chrom']}")
    
    result = {
        'protein_coding': pc_sample['gene_id'].tolist(),
        'lncrna': lnc_sample['gene_id'].tolist(),
        'all': pc_sample['gene_id'].tolist() + lnc_sample['gene_id'].tolist()
    }
    
    print(f"\nTotal genes selected: {len(result['all'])}")
    print(f"  - Protein-coding: {len(result['protein_coding'])}")
    print(f"  - lncRNA: {len(result['lncrna'])}")
    
    return result


def calculate_metrics_by_biotype(positions_df: pl.DataFrame, gene_biotypes: dict) -> dict:
    """
    Calculate F1 scores separately for protein-coding and lncRNA genes.
    
    Parameters
    ----------
    positions_df : pl.DataFrame
        Positions DataFrame with predictions
    gene_biotypes : dict
        Dict with 'protein_coding' and 'lncrna' gene ID lists
    
    Returns
    -------
    dict with metrics for each biotype
    """
    from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_curve, auc
    
    results = {}
    
    for biotype, gene_ids in [('protein_coding', gene_biotypes['protein_coding']),
                               ('lncrna', gene_biotypes['lncrna'])]:
        # Filter to this biotype
        biotype_df = positions_df.filter(pl.col('gene_id').is_in(gene_ids))
        
        if biotype_df.height == 0:
            print(f"\n[warning] No predictions found for {biotype} genes")
            continue
        
        print(f"\n{'='*80}")
        print(f"METRICS FOR {biotype.upper()} GENES")
        print(f"{'='*80}\n")
        
        # Calculate metrics for each splice type
        for splice_type in ['donor', 'acceptor']:
            type_df = biotype_df.filter(pl.col('splice_type') == splice_type)
            
            if type_df.height == 0:
                continue
            
            # Get true labels and predictions
            y_true = (type_df['error_type'].is_in(['TP', 'FN'])).to_numpy()
            y_scores = type_df[f'{splice_type}_score'].to_numpy()
            y_pred = (y_scores >= 0.5).astype(int)
            
            # Calculate metrics
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            
            # Calculate PR-AUC
            precisions, recalls, _ = precision_recall_curve(y_true, y_scores)
            pr_auc = auc(recalls, precisions)
            
            print(f"{splice_type.capitalize()} sites:")
            print(f"  Total positions: {type_df.height:,}")
            print(f"  True positives:  {type_df.filter(pl.col('error_type') == 'TP').height:,}")
            print(f"  False positives: {type_df.filter(pl.col('error_type') == 'FP').height:,}")
            print(f"  False negatives: {type_df.filter(pl.col('error_type') == 'FN').height:,}")
            print(f"  True negatives:  {type_df.filter(pl.col('error_type') == 'TN').height:,}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall:    {recall:.4f}")
            print(f"  F1 Score:  {f1:.4f}")
            print(f"  PR-AUC:    {pr_auc:.4f}")
            print()
            
            # Store results
            if biotype not in results:
                results[biotype] = {}
            results[biotype][splice_type] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'pr_auc': pr_auc,
                'n_positions': type_df.height,
                'n_tp': type_df.filter(pl.col('error_type') == 'TP').height,
                'n_fp': type_df.filter(pl.col('error_type') == 'FP').height,
                'n_fn': type_df.filter(pl.col('error_type') == 'FN').height,
                'n_tn': type_df.filter(pl.col('error_type') == 'TN').height,
            }
    
    return results


def main():
    """Run base model test with diverse genes."""
    
    print(f"\n{'='*80}")
    print("BASE MODEL TEST: DIVERSE GENE SET (PROTEIN-CODING + lncRNA)")
    print(f"{'='*80}\n")
    
    # Set up paths using Registry
    # GRCh37 uses Ensembl release 87
    registry = Registry(build='GRCh37', release='87')
    
    gtf_path = registry.resolve('gtf')
    fasta_path = registry.resolve('fasta')
    eval_dir = registry.eval_dir
    
    print(f"Configuration:")
    print(f"  Build:     GRCh37")
    print(f"  Release:   87")
    print(f"  GTF:       {gtf_path}")
    print(f"  FASTA:     {fasta_path}")
    print(f"  Eval dir:  {eval_dir}")
    
    # Sample diverse genes
    gene_sample = get_diverse_gene_sample(
        gtf_path=str(gtf_path),
        n_protein_coding=15,
        n_lncrna=5
    )
    
    # Create config
    config = SpliceAIConfig(
        gtf_file=str(gtf_path),
        genome_fasta=str(fasta_path),
        eval_dir=str(eval_dir),
        output_subdir="diverse_genes_test",
        threshold=0.5,
        consensus_window=2,
        error_window=2,
        test_mode=False,
        do_extract_annotations=False,  # Use existing
        do_extract_splice_sites=False,  # Use existing
        do_extract_sequences=False,     # Use existing
        do_find_overlaping_genes=False, # Use existing
        use_auto_position_adjustments=True,  # Enable automatic adjustment detection
        local_dir=str(registry.data_dir),
    )
    
    print(f"\n{'='*80}")
    print("RUNNING SPLICE PREDICTION WORKFLOW")
    print(f"{'='*80}\n")
    
    # Run workflow
    results = run_enhanced_splice_prediction_workflow(
        config=config,
        target_genes=gene_sample['all'],
        verbosity=1,
        no_final_aggregate=False,
        no_tn_sampling=False,
    )
    
    if not results.get('success'):
        print("\n[ERROR] Workflow failed!")
        return 1
    
    # Get positions DataFrame
    positions_df = results.get('positions')
    
    if positions_df is None or positions_df.height == 0:
        print("\n[ERROR] No predictions generated!")
        return 1
    
    print(f"\n{'='*80}")
    print("WORKFLOW COMPLETED SUCCESSFULLY")
    print(f"{'='*80}\n")
    
    print(f"Total positions: {positions_df.height:,}")
    print(f"Unique genes: {positions_df['gene_id'].n_unique()}")
    
    # Calculate metrics by biotype
    metrics = calculate_metrics_by_biotype(positions_df, gene_sample)
    
    # Summary comparison
    print(f"\n{'='*80}")
    print("SUMMARY: PROTEIN-CODING vs lncRNA PERFORMANCE")
    print(f"{'='*80}\n")
    
    if 'protein_coding' in metrics and 'lncrna' in metrics:
        for splice_type in ['donor', 'acceptor']:
            if splice_type in metrics['protein_coding'] and splice_type in metrics['lncrna']:
                pc_f1 = metrics['protein_coding'][splice_type]['f1']
                lnc_f1 = metrics['lncrna'][splice_type]['f1']
                pc_prauc = metrics['protein_coding'][splice_type]['pr_auc']
                lnc_prauc = metrics['lncrna'][splice_type]['pr_auc']
                
                print(f"{splice_type.capitalize()} sites:")
                print(f"  Protein-coding: F1={pc_f1:.4f}, PR-AUC={pc_prauc:.4f}")
                print(f"  lncRNA:         F1={lnc_f1:.4f}, PR-AUC={lnc_prauc:.4f}")
                print(f"  Difference:     ΔF1={pc_f1 - lnc_f1:+.4f}, ΔPR-AUC={pc_prauc - lnc_prauc:+.4f}")
                print()
    
    # Check alignment (zero adjustments expected)
    print(f"\n{'='*80}")
    print("COORDINATE ALIGNMENT CHECK")
    print(f"{'='*80}\n")
    
    print("Expected: Zero adjustments (predictions should align with GRCh37 annotations)")
    print("If non-zero adjustments were detected, this indicates a coordinate mismatch.")
    
    # Look for adjustment info in the workflow output
    # (This would be printed during the workflow run)
    
    print(f"\n{'='*80}")
    print("TEST COMPLETE")
    print(f"{'='*80}\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

