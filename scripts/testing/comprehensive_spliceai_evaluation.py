#!/usr/bin/env python3
"""
Comprehensive SpliceAI Evaluation with Proper Metrics

This script implements the 3 recommended evaluation methods:
1. PR-AUC (Precision-Recall Area Under Curve)
2. Top-k Accuracy
3. Optimal Threshold F1

Tests on 50+ protein-coding genes for robust statistics.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))

import numpy as np
import polars as pl
import logging
import argparse
from typing import Dict, List, Tuple
from sklearn.metrics import precision_recall_curve, auc
import json

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def sample_protein_coding_genes_from_gtf(
    build: str = 'GRCh38',
    release: str = '112',
    num_genes: int = 50,
    seed: int = 42
) -> List[Dict]:
    """
    Dynamically sample protein-coding genes from the GTF file.
    
    This ensures genes exist in the specified genome build.
    
    Parameters
    ----------
    build : str
        Genome build (e.g., 'GRCh38', 'GRCh37')
    release : str
        Ensembl release (e.g., '112', '87')
    num_genes : int
        Number of genes to sample
    seed : int
        Random seed for reproducibility
        
    Returns
    -------
    List[Dict]
        List of gene dictionaries with gene_id, gene_name, chr, strand
    """
    from meta_spliceai.system.genomic_resources import Registry
    import subprocess
    import random
    
    # Get GTF path
    registry = Registry(build=build, release=release)
    gtf_path = registry.resolve('gtf')
    
    if not gtf_path:
        logger.warning(f"GTF file not found for build {build}, falling back to hardcoded genes")
        return get_50_protein_coding_genes_hardcoded()
    
    logger.info(f"Sampling {num_genes} protein-coding genes from {gtf_path}...")
    
    # Extract protein-coding genes from GTF
    # Use grep to find gene entries with biotype "protein_coding"
    try:
        # First grep for protein_coding, then filter for gene entries
        result = subprocess.run(
            f'grep \'gene_biotype "protein_coding"\' {gtf_path} | grep -E "\\tgene\\t"',
            capture_output=True,
            text=True,
            timeout=60,
            shell=True
        )
        
        if not result.stdout:
            logger.warning("No protein-coding genes found, falling back to hardcoded genes")
            return get_50_protein_coding_genes_hardcoded()
        
        # Parse genes
        genes = []
        seen_gene_ids = set()
        
        for line in result.stdout.strip().split('\n'):
            if not line or line.startswith('#'):
                continue
            
            parts = line.split('\t')
            if len(parts) < 9:
                continue
            
            chrom = parts[0]
            strand = parts[6]
            attributes = parts[8]
            
            # Extract gene_id and gene_name
            gene_id = None
            gene_name = None
            
            for attr in attributes.split(';'):
                attr = attr.strip()
                if attr.startswith('gene_id'):
                    gene_id = attr.split('"')[1]
                elif attr.startswith('gene_name'):
                    gene_name = attr.split('"')[1]
            
            if gene_id and gene_id not in seen_gene_ids:
                seen_gene_ids.add(gene_id)
                genes.append({
                    'gene_id': gene_id,
                    'gene_name': gene_name or gene_id,
                    'chr': chrom,
                    'strand': strand
                })
        
        if len(genes) == 0:
            logger.warning("No genes parsed, falling back to hardcoded genes")
            return get_50_protein_coding_genes_hardcoded()
        
        logger.info(f"Found {len(genes)} protein-coding genes in {build}")
        
        # Sample genes with stratification by chromosome
        random.seed(seed)
        
        # Group by chromosome
        genes_by_chr = {}
        for gene in genes:
            chr_name = gene['chr']
            if chr_name not in genes_by_chr:
                genes_by_chr[chr_name] = []
            genes_by_chr[chr_name].append(gene)
        
        # Sample proportionally from each chromosome
        sampled_genes = []
        chromosomes = sorted(genes_by_chr.keys())
        
        # Prioritize main chromosomes (1-22, X, Y)
        main_chrs = [str(i) for i in range(1, 23)] + ['X', 'Y']
        priority_chrs = [c for c in main_chrs if c in genes_by_chr]
        other_chrs = [c for c in chromosomes if c not in priority_chrs]
        
        # Sample from priority chromosomes first
        genes_per_chr = max(1, num_genes // len(priority_chrs)) if priority_chrs else 1
        
        for chr_name in priority_chrs:
            chr_genes = genes_by_chr[chr_name]
            n_sample = min(genes_per_chr, len(chr_genes))
            sampled_genes.extend(random.sample(chr_genes, n_sample))
            
            if len(sampled_genes) >= num_genes:
                break
        
        # If we need more, sample from other chromosomes
        if len(sampled_genes) < num_genes:
            remaining = num_genes - len(sampled_genes)
            other_genes = [g for chr_name in other_chrs for g in genes_by_chr[chr_name]]
            if other_genes:
                n_sample = min(remaining, len(other_genes))
                sampled_genes.extend(random.sample(other_genes, n_sample))
        
        # Shuffle and trim to exact number
        random.shuffle(sampled_genes)
        sampled_genes = sampled_genes[:num_genes]
        
        logger.info(f"Sampled {len(sampled_genes)} genes from {len(priority_chrs)} main chromosomes")
        
        return sampled_genes
        
    except subprocess.TimeoutExpired:
        logger.warning("GTF parsing timed out, falling back to hardcoded genes")
        return get_50_protein_coding_genes_hardcoded()
    except Exception as e:
        logger.warning(f"Error sampling genes from GTF: {e}, falling back to hardcoded genes")
        return get_50_protein_coding_genes_hardcoded()


def get_50_protein_coding_genes_hardcoded() -> List[Dict]:
    """
    Hardcoded list of 50+ protein-coding genes (fallback for GRCh38).
    
    Selection criteria:
    - Protein-coding genes
    - Various sizes (small, medium, large)
    - Different chromosomes
    - Mix of + and - strands
    """
    return [
        # Chromosome 1 (5 genes)
        {'gene_id': 'ENSG00000117318', 'gene_name': 'ID3', 'chr': '1', 'strand': '-'},
        {'gene_id': 'ENSG00000162692', 'gene_name': 'VCAM1', 'chr': '1', 'strand': '+'},
        {'gene_id': 'ENSG00000073756', 'gene_name': 'PTGS2', 'chr': '1', 'strand': '-'},
        {'gene_id': 'ENSG00000143933', 'gene_name': 'CALM2', 'chr': '2', 'strand': '+'},
        {'gene_id': 'ENSG00000116044', 'gene_name': 'NFE2L2', 'chr': '2', 'strand': '+'},
        
        # Chromosome 2-3 (5 genes)
        {'gene_id': 'ENSG00000115414', 'gene_name': 'FN1', 'chr': '2', 'strand': '+'},
        {'gene_id': 'ENSG00000163930', 'gene_name': 'BAP1', 'chr': '3', 'strand': '+'},
        {'gene_id': 'ENSG00000134086', 'gene_name': 'VHL', 'chr': '3', 'strand': '+'},
        {'gene_id': 'ENSG00000196549', 'gene_name': 'MME', 'chr': '3', 'strand': '+'},
        {'gene_id': 'ENSG00000163735', 'gene_name': 'CXCL12', 'chr': '10', 'strand': '+'},
        
        # Chromosome 4-5 (5 genes)
        {'gene_id': 'ENSG00000138795', 'gene_name': 'LEF1', 'chr': '4', 'strand': '-'},
        {'gene_id': 'ENSG00000113558', 'gene_name': 'SKP2', 'chr': '5', 'strand': '-'},
        {'gene_id': 'ENSG00000164362', 'gene_name': 'TERT', 'chr': '5', 'strand': '-'},
        {'gene_id': 'ENSG00000113721', 'gene_name': 'PDGFRB', 'chr': '5', 'strand': '-'},
        {'gene_id': 'ENSG00000164308', 'gene_name': 'ERAP1', 'chr': '5', 'strand': '+'},
        
        # Chromosome 6-7 (5 genes)
        {'gene_id': 'ENSG00000204287', 'gene_name': 'HLA-A', 'chr': '6', 'strand': '+'},
        {'gene_id': 'ENSG00000146648', 'gene_name': 'EGFR', 'chr': '7', 'strand': '+'},
        {'gene_id': 'ENSG00000105974', 'gene_name': 'CAV1', 'chr': '7', 'strand': '-'},
        {'gene_id': 'ENSG00000106683', 'gene_name': 'LIMK1', 'chr': '7', 'strand': '+'},
        {'gene_id': 'ENSG00000164134', 'gene_name': 'NAV3', 'chr': '12', 'strand': '+'},
        
        # Chromosome 8-9 (5 genes)
        {'gene_id': 'ENSG00000136997', 'gene_name': 'MYC', 'chr': '8', 'strand': '+'},
        {'gene_id': 'ENSG00000147889', 'gene_name': 'CDKN2A', 'chr': '9', 'strand': '+'},
        {'gene_id': 'ENSG00000107104', 'gene_name': 'KRAS', 'chr': '12', 'strand': '-'},
        {'gene_id': 'ENSG00000185862', 'gene_name': 'ENG', 'chr': '9', 'strand': '+'},
        {'gene_id': 'ENSG00000107485', 'gene_name': 'GATA3', 'chr': '10', 'strand': '+'},
        
        # Chromosome 10-11 (5 genes)
        {'gene_id': 'ENSG00000165731', 'gene_name': 'RET', 'chr': '10', 'strand': '+'},
        {'gene_id': 'ENSG00000134250', 'gene_name': 'NOTCH2', 'chr': '1', 'strand': '-'},
        {'gene_id': 'ENSG00000149925', 'gene_name': 'ALDOA', 'chr': '16', 'strand': '-'},
        {'gene_id': 'ENSG00000110092', 'gene_name': 'CCND1', 'chr': '11', 'strand': '+'},
        {'gene_id': 'ENSG00000141736', 'gene_name': 'ERBB2', 'chr': '17', 'strand': '+'},
        
        # Chromosome 12 (5 genes)
        {'gene_id': 'ENSG00000111640', 'gene_name': 'GAPDH', 'chr': '12', 'strand': '+'},
        {'gene_id': 'ENSG00000123374', 'gene_name': 'CDK2', 'chr': '12', 'strand': '+'},
        {'gene_id': 'ENSG00000111276', 'gene_name': 'CDKN1B', 'chr': '12', 'strand': '+'},
        {'gene_id': 'ENSG00000139687', 'gene_name': 'RB1', 'chr': '13', 'strand': '+'},
        {'gene_id': 'ENSG00000100030', 'gene_name': 'MAPK1', 'chr': '22', 'strand': '+'},
        
        # Chromosome 13-17 (10 genes)
        {'gene_id': 'ENSG00000141510', 'gene_name': 'TP53', 'chr': '17', 'strand': '-'},
        {'gene_id': 'ENSG00000134202', 'gene_name': 'GSTM3', 'chr': '1', 'strand': '+'},
        {'gene_id': 'ENSG00000157764', 'gene_name': 'BRAF', 'chr': '7', 'strand': '-'},
        {'gene_id': 'ENSG00000171862', 'gene_name': 'PTEN', 'chr': '10', 'strand': '+'},
        {'gene_id': 'ENSG00000133703', 'gene_name': 'KRAS', 'chr': '12', 'strand': '-'},
        {'gene_id': 'ENSG00000142192', 'gene_name': 'APP', 'chr': '21', 'strand': '-'},
        {'gene_id': 'ENSG00000012048', 'gene_name': 'BRCA1', 'chr': '17', 'strand': '-'},
        {'gene_id': 'ENSG00000139618', 'gene_name': 'BRCA2', 'chr': '13', 'strand': '+'},
        {'gene_id': 'ENSG00000183914', 'gene_name': 'PDCD1', 'chr': '2', 'strand': '+'},
        {'gene_id': 'ENSG00000120129', 'gene_name': 'DUSP1', 'chr': '5', 'strand': '-'},
        
        # Additional genes for 50+ (10 genes)
        {'gene_id': 'ENSG00000166710', 'gene_name': 'B2M', 'chr': '15', 'strand': '+'},
        {'gene_id': 'ENSG00000101347', 'gene_name': 'SAMHD1', 'chr': '20', 'strand': '-'},
        {'gene_id': 'ENSG00000198691', 'gene_name': 'ABCA1', 'chr': '9', 'strand': '-'},
        {'gene_id': 'ENSG00000115415', 'gene_name': 'STAT1', 'chr': '2', 'strand': '-'},
        {'gene_id': 'ENSG00000168610', 'gene_name': 'STAT3', 'chr': '17', 'strand': '-'},
        {'gene_id': 'ENSG00000134954', 'gene_name': 'ETS1', 'chr': '11', 'strand': '-'},
        {'gene_id': 'ENSG00000092969', 'gene_name': 'TGFB1', 'chr': '19', 'strand': '+'},
        {'gene_id': 'ENSG00000108691', 'gene_name': 'CCL2', 'chr': '17', 'strand': '+'},
        {'gene_id': 'ENSG00000136244', 'gene_name': 'IL6', 'chr': '7', 'strand': '+'},
        {'gene_id': 'ENSG00000232810', 'gene_name': 'TNF', 'chr': '6', 'strand': '+'},
    ]


def load_predictions_and_annotations(gene_id: str, build: str = 'GRCh38', release: str = '112') -> Tuple[pl.DataFrame, pl.DataFrame]:
    """Load predictions and annotations for a gene."""
    # Load predictions
    pred_file = Path(f"predictions/spliceai_eval/meta_models/complete_base_predictions/{gene_id}/complete_predictions_{gene_id}.parquet")
    if not pred_file.exists():
        return None, None
    
    predictions_df = pl.read_parquet(pred_file)
    
    # Load annotations using Registry
    from meta_spliceai.system.genomic_resources import Registry
    registry = Registry(build=build, release=release)
    
    splice_sites_path = registry.resolve('splice_sites')
    if not splice_sites_path:
        logger.warning(f"Splice sites file not found for build {build}")
        return None, None
    
    # Load splice sites for this gene
    annotations_df = pl.read_csv(
        splice_sites_path,
        separator='\t',
        schema_overrides={'chrom': pl.Utf8}
    ).filter(pl.col('gene_id') == gene_id)
    
    if annotations_df.height == 0:
        return None, None
    
    # Rename site_type to splice_type for consistency
    if 'site_type' in annotations_df.columns:
        annotations_df = annotations_df.rename({'site_type': 'splice_type'})
    
    return predictions_df, annotations_df


def calculate_pr_auc(
    true_labels: np.ndarray,
    predicted_scores: np.ndarray
) -> float:
    """
    Calculate Precision-Recall Area Under Curve.
    
    This is the metric SpliceAI paper used (PR-AUC = 0.97).
    """
    if len(true_labels) == 0 or len(predicted_scores) == 0:
        return 0.0
    
    # Handle edge case: all negatives or all positives
    if np.sum(true_labels) == 0 or np.sum(true_labels) == len(true_labels):
        return 0.0
    
    precision, recall, _ = precision_recall_curve(true_labels, predicted_scores)
    pr_auc = auc(recall, precision)
    
    return pr_auc


def calculate_top_k_accuracy(
    true_positions: set,
    predicted_scores: np.ndarray,
    all_positions: np.ndarray
) -> Tuple[float, float]:
    """
    Calculate Top-k Accuracy.
    
    This is SpliceAI's primary metric (Top-k = 95%).
    
    Returns:
    - top_k_accuracy: Fraction of true sites in top-k predictions
    - optimal_threshold: Threshold used for top-k
    """
    if len(true_positions) == 0:
        return 0.0, 0.0
    
    k = len(true_positions)
    
    # Get top-k predictions
    top_k_indices = np.argsort(predicted_scores)[-k:]
    top_k_positions = set(all_positions[top_k_indices].tolist())
    
    # Count matches
    matches = len(true_positions & top_k_positions)
    accuracy = matches / k if k > 0 else 0.0
    
    # Find threshold (k-th highest score)
    optimal_threshold = np.sort(predicted_scores)[-k] if k <= len(predicted_scores) else 0.0
    
    return accuracy, optimal_threshold


def find_optimal_threshold_f1(
    true_positions: set,
    predicted_scores: np.ndarray,
    all_positions: np.ndarray,
    threshold_range: Tuple[float, float] = (0.05, 0.95),
    num_thresholds: int = 50
) -> Tuple[float, float, float]:
    """
    Find threshold that maximizes F1 score.
    
    Returns:
    - optimal_threshold: Best threshold
    - optimal_f1: F1 score at optimal threshold
    - f1_at_0_5: F1 score at threshold=0.5 (for comparison)
    """
    if len(true_positions) == 0:
        return 0.0, 0.0, 0.0
    
    thresholds = np.linspace(threshold_range[0], threshold_range[1], num_thresholds)
    
    best_f1 = 0.0
    best_threshold = 0.5
    f1_at_0_5 = 0.0
    
    for threshold in thresholds:
        # Get predictions at this threshold
        predicted_positions = set(all_positions[predicted_scores > threshold].tolist())
        
        # Calculate metrics
        tp = len(true_positions & predicted_positions)
        fp = len(predicted_positions - true_positions)
        fn = len(true_positions - predicted_positions)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
        
        # Track F1 at 0.5
        if abs(threshold - 0.5) < 0.01:
            f1_at_0_5 = f1
    
    return best_threshold, best_f1, f1_at_0_5


def evaluate_gene(
    gene_id: str,
    gene_name: str,
    predictions_df: pl.DataFrame,
    annotations_df: pl.DataFrame,
    splice_type: str
) -> Dict:
    """
    Evaluate a single gene for a specific splice type.
    
    Returns dictionary with all metrics.
    """
    # Get true splice sites
    true_sites = set(
        annotations_df.filter(pl.col('splice_type') == splice_type)['position'].to_list()
    )
    
    if len(true_sites) == 0:
        return None
    
    # Get predictions
    score_col = f'{splice_type}_score'
    predicted_scores = predictions_df[score_col].to_numpy()
    all_positions = predictions_df['position'].to_numpy()
    
    # Create binary labels for PR-AUC
    true_labels = np.array([1 if pos in true_sites else 0 for pos in all_positions])
    
    # Calculate all metrics
    pr_auc = calculate_pr_auc(true_labels, predicted_scores)
    top_k_acc, top_k_threshold = calculate_top_k_accuracy(true_sites, predicted_scores, all_positions)
    optimal_threshold, optimal_f1, f1_at_0_5 = find_optimal_threshold_f1(true_sites, predicted_scores, all_positions)
    
    return {
        'gene_id': gene_id,
        'gene_name': gene_name,
        'splice_type': splice_type,
        'num_true_sites': len(true_sites),
        'pr_auc': pr_auc,
        'top_k_accuracy': top_k_acc,
        'top_k_threshold': top_k_threshold,
        'optimal_threshold': optimal_threshold,
        'optimal_f1': optimal_f1,
        'f1_at_0_5': f1_at_0_5
    }


def main():
    """Main evaluation function."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Comprehensive SpliceAI Evaluation')
    parser.add_argument('--build', type=str, default='GRCh38', help='Genome build (default: GRCh38)')
    parser.add_argument('--release', type=str, default='112', help='Ensembl release (default: 112)')
    parser.add_argument('--output', type=str, default=None, help='Output file for results (optional)')
    parser.add_argument('--num-genes', type=int, default=None, help='Number of genes to test (default: all)')
    args = parser.parse_args()
    
    logger.info("="*80)
    logger.info("COMPREHENSIVE SPLICEAI EVALUATION")
    logger.info("="*80)
    logger.info("")
    logger.info(f"Genome Build: {args.build}")
    logger.info(f"Ensembl Release: {args.release}")
    logger.info("")
    logger.info("Implementing 3 evaluation methods:")
    logger.info("  1. PR-AUC (SpliceAI paper: 0.97)")
    logger.info("  2. Top-k Accuracy (SpliceAI paper: 0.95)")
    logger.info("  3. Optimal Threshold F1")
    logger.info("")
    
    # Get genes - sample dynamically from GTF
    num_genes_to_sample = args.num_genes if args.num_genes else 50
    genes = sample_protein_coding_genes_from_gtf(
        build=args.build,
        release=args.release,
        num_genes=num_genes_to_sample,
        seed=42
    )
    logger.info(f"Testing on {len(genes)} protein-coding genes")
    logger.info("")
    
    # Check if predictions exist, if not generate them
    missing_genes = []
    for gene_info in genes:
        gene_id = gene_info['gene_id']
        pred_file = Path(f"predictions/spliceai_eval/meta_models/complete_base_predictions/{gene_id}/complete_predictions_{gene_id}.parquet")
        if not pred_file.exists():
            missing_genes.append(gene_id)
    
    if missing_genes:
        logger.info(f"⚠️  {len(missing_genes)} genes missing predictions")
        logger.info(f"   Generating predictions first...")
        logger.info("")
        
        # Generate predictions
        from scripts.testing.generate_and_test_20_genes import generate_predictions_for_genes
        generate_predictions_for_genes(missing_genes, mode='base_only')
        logger.info("")
    
    # Evaluate all genes
    results = []
    successful_genes = 0
    
    for i, gene_info in enumerate(genes, 1):
        gene_id = gene_info['gene_id']
        gene_name = gene_info['gene_name']
        
        logger.info(f"[{i}/{len(genes)}] Evaluating {gene_name} ({gene_id})...")
        
        # Load data
        predictions_df, annotations_df = load_predictions_and_annotations(gene_id, args.build, args.release)
        
        if predictions_df is None or annotations_df is None:
            logger.info(f"  ⚠️  Skipping (no data)")
            continue
        
        # Evaluate for donor and acceptor
        for splice_type in ['donor', 'acceptor']:
            result = evaluate_gene(gene_id, gene_name, predictions_df, annotations_df, splice_type)
            if result:
                results.append(result)
        
        successful_genes += 1
    
    logger.info("")
    logger.info(f"✅ Evaluated {successful_genes}/{len(genes)} genes")
    logger.info("")
    
    if len(results) == 0:
        logger.error("❌ No results to analyze!")
        return
    
    # Convert to DataFrame for analysis
    results_df = pl.DataFrame(results)
    
    # Save results
    output_file = Path("predictions/comprehensive_evaluation_results.parquet")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    results_df.write_parquet(output_file)
    logger.info(f"💾 Saved results to {output_file}")
    logger.info("")
    
    # Aggregate statistics
    logger.info("="*80)
    logger.info("AGGREGATE STATISTICS")
    logger.info("="*80)
    logger.info("")
    
    # Overall statistics
    logger.info("📊 Overall Performance (All Genes, Both Splice Types)")
    logger.info("─"*80)
    logger.info(f"  PR-AUC:           {results_df['pr_auc'].mean():.3f} ± {results_df['pr_auc'].std():.3f}")
    logger.info(f"  Top-k Accuracy:   {results_df['top_k_accuracy'].mean():.3f} ± {results_df['top_k_accuracy'].std():.3f}")
    logger.info(f"  Optimal F1:       {results_df['optimal_f1'].mean():.3f} ± {results_df['optimal_f1'].std():.3f}")
    logger.info(f"  F1 at 0.5:        {results_df['f1_at_0_5'].mean():.3f} ± {results_df['f1_at_0_5'].std():.3f}")
    logger.info(f"  Optimal Threshold: {results_df['optimal_threshold'].mean():.3f} ± {results_df['optimal_threshold'].std():.3f}")
    logger.info("")
    
    # By splice type
    logger.info("📊 Performance by Splice Type")
    logger.info("─"*80)
    for splice_type in ['donor', 'acceptor']:
        subset = results_df.filter(pl.col('splice_type') == splice_type)
        logger.info(f"  {splice_type.upper()}:")
        logger.info(f"    PR-AUC:           {subset['pr_auc'].mean():.3f} ± {subset['pr_auc'].std():.3f}")
        logger.info(f"    Top-k Accuracy:   {subset['top_k_accuracy'].mean():.3f} ± {subset['top_k_accuracy'].std():.3f}")
        logger.info(f"    Optimal F1:       {subset['optimal_f1'].mean():.3f} ± {subset['optimal_f1'].std():.3f}")
        logger.info(f"    F1 at 0.5:        {subset['f1_at_0_5'].mean():.3f} ± {subset['f1_at_0_5'].std():.3f}")
        logger.info("")
    
    # Comparison to SpliceAI paper
    logger.info("="*80)
    logger.info("COMPARISON TO SPLICEAI PAPER")
    logger.info("="*80)
    logger.info("")
    logger.info("Metric                 | SpliceAI Paper | Our Results      | Difference")
    logger.info("─"*80)
    
    our_pr_auc = results_df['pr_auc'].mean()
    our_top_k = results_df['top_k_accuracy'].mean()
    
    logger.info(f"PR-AUC                 |     0.970      |     {our_pr_auc:.3f}      |   {our_pr_auc - 0.970:+.3f}")
    logger.info(f"Top-k Accuracy         |     0.950      |     {our_top_k:.3f}      |   {our_top_k - 0.950:+.3f}")
    logger.info("")
    
    # Key findings
    logger.info("="*80)
    logger.info("KEY FINDINGS")
    logger.info("="*80)
    logger.info("")
    
    if our_pr_auc >= 0.85:
        logger.info("✅ PR-AUC is excellent (≥0.85), close to SpliceAI's 0.97")
    elif our_pr_auc >= 0.75:
        logger.info("⚠️  PR-AUC is good (≥0.75) but lower than SpliceAI's 0.97")
    else:
        logger.info("❌ PR-AUC is lower than expected (<0.75)")
    
    logger.info("")
    
    optimal_f1 = results_df['optimal_f1'].mean()
    f1_at_0_5 = results_df['f1_at_0_5'].mean()
    improvement = optimal_f1 - f1_at_0_5
    
    logger.info(f"📈 F1 Score Improvement with Optimal Threshold:")
    logger.info(f"   F1 at threshold=0.5:  {f1_at_0_5:.3f}")
    logger.info(f"   F1 at optimal:        {optimal_f1:.3f}")
    if f1_at_0_5 > 0:
        logger.info(f"   Improvement:          {improvement:+.3f} ({improvement/f1_at_0_5*100:+.1f}%)")
    else:
        logger.info(f"   Improvement:          {improvement:+.3f} (N/A - baseline is zero)")
    logger.info("")
    
    optimal_threshold = results_df['optimal_threshold'].mean()
    logger.info(f"🎯 Optimal Threshold: {optimal_threshold:.3f}")
    logger.info(f"   (vs fixed threshold of 0.5)")
    logger.info("")
    
    # Save summary
    summary = {
        'num_genes': successful_genes,
        'num_evaluations': len(results),
        'overall': {
            'pr_auc_mean': float(results_df['pr_auc'].mean()),
            'pr_auc_std': float(results_df['pr_auc'].std()),
            'top_k_accuracy_mean': float(results_df['top_k_accuracy'].mean()),
            'top_k_accuracy_std': float(results_df['top_k_accuracy'].std()),
            'optimal_f1_mean': float(results_df['optimal_f1'].mean()),
            'optimal_f1_std': float(results_df['optimal_f1'].std()),
            'f1_at_0_5_mean': float(results_df['f1_at_0_5'].mean()),
            'f1_at_0_5_std': float(results_df['f1_at_0_5'].std()),
            'optimal_threshold_mean': float(results_df['optimal_threshold'].mean()),
            'optimal_threshold_std': float(results_df['optimal_threshold'].std())
        },
        'by_splice_type': {}
    }
    
    for splice_type in ['donor', 'acceptor']:
        subset = results_df.filter(pl.col('splice_type') == splice_type)
        summary['by_splice_type'][splice_type] = {
            'pr_auc_mean': float(subset['pr_auc'].mean()),
            'top_k_accuracy_mean': float(subset['top_k_accuracy'].mean()),
            'optimal_f1_mean': float(subset['optimal_f1'].mean()),
            'f1_at_0_5_mean': float(subset['f1_at_0_5'].mean())
        }
    
    summary_file = Path("predictions/comprehensive_evaluation_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"💾 Saved summary to {summary_file}")
    logger.info("")
    logger.info("="*80)
    logger.info("EVALUATION COMPLETE")
    logger.info("="*80)


if __name__ == "__main__":
    main()

