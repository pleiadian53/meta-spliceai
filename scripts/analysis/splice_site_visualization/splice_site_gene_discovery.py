#!/usr/bin/env python3
"""Gene discovery helper for splice site visualization.

This script helps identify the best genes for splice site visualization by:
1. Loading gene and transcript metadata
2. Analyzing splice site counts per gene
3. Optionally calculating FP/FN rates from CV results
4. Generating ranked lists for testing

Usage:
    python splice_site_gene_discovery.py --dataset train_pc_1000/master --output genes_for_testing.tsv
    python splice_site_gene_discovery.py --dataset train_pc_1000/master --cv-results results/gene_cv_1000_run_15/position_level_classification_results.tsv --top-n 20
"""

import argparse
import logging
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SpliceSiteGeneDiscovery:
    """Helper class for discovering genes suitable for splice site visualization."""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.gene_features = None
        self.transcript_features = None
        self.splice_sites = None
        
    def log(self, message: str, level: str = "INFO"):
        """Log messages with appropriate level."""
        if self.verbose:
            if level == "INFO":
                logger.info(message)
            elif level == "WARNING":
                logger.warning(message)
            elif level == "ERROR":
                logger.error(message)
    
    def load_gene_metadata(self, gene_features_path: str = "data/ensembl/spliceai_analysis/gene_features.tsv",
                          transcript_features_path: str = "data/ensembl/spliceai_analysis/transcript_features.tsv",
                          splice_sites_path: str = "data/ensembl/splice_sites.tsv") -> None:
        """Load gene, transcript, and splice site metadata."""
        
        # Load gene features
        if Path(gene_features_path).exists():
            self.log(f"Loading gene features from: {gene_features_path}")
            self.gene_features = pd.read_csv(gene_features_path, sep='\t')
            self.log(f"Loaded {len(self.gene_features)} gene features")
        else:
            self.log(f"Gene features file not found: {gene_features_path}", "WARNING")
            
        # Load transcript features
        if Path(transcript_features_path).exists():
            self.log(f"Loading transcript features from: {transcript_features_path}")
            self.transcript_features = pd.read_csv(transcript_features_path, sep='\t')
            self.log(f"Loaded {len(self.transcript_features)} transcript features")
        else:
            self.log(f"Transcript features file not found: {transcript_features_path}", "WARNING")
            
        # Load splice sites
        if Path(splice_sites_path).exists():
            self.log(f"Loading splice sites from: {splice_sites_path}")
            self.splice_sites = pd.read_csv(splice_sites_path, sep='\t')
            self.log(f"Loaded {len(self.splice_sites)} splice sites")
        else:
            self.log(f"Splice sites file not found: {splice_sites_path}", "WARNING")
    
    def load_dataset_genes(self, dataset_path: str, target_genes: List[str] = None) -> pd.DataFrame:
        """Load dataset with efficient gene-level sampling."""
        self.log(f"Loading dataset from: {dataset_path}")
        
        try:
            # Use hierarchical sampling to get gene-level data efficiently
            if target_genes:
                self.log(f"Loading data for {len(target_genes)} target genes")
                # For specific genes, we need to load enough data to find them
                from meta_spliceai.splice_engine.meta_models.training.label_utils import load_dataset_sample
                # Sample more genes to ensure we capture the target genes
                df = load_dataset_sample(dataset_path, sample_genes=min(500, len(target_genes) * 10), random_seed=42)
            else:
                self.log("Loading sample of genes for discovery")
                from meta_spliceai.splice_engine.meta_models.training.label_utils import load_dataset_sample
                df = load_dataset_sample(dataset_path, sample_genes=200, random_seed=42)
            
            # Convert to pandas if needed
            if hasattr(df, 'to_pandas'):
                df = df.to_pandas()
            
            self.log(f"Successfully loaded {len(df):,} samples with {len(df.columns)} columns")
            return df
            
        except Exception as e:
            self.log(f"Error loading dataset: {e}", "ERROR")
            raise
    
    def analyze_gene_splice_sites(self, df: pd.DataFrame) -> pd.DataFrame:
        """Analyze splice site counts per gene."""
        self.log("Analyzing splice site counts per gene...")
        
        # Count splice sites by gene and type
        gene_stats = []
        
        for gene_id in df['gene_id'].unique():
            gene_data = df[df['gene_id'] == gene_id]
            
            # Count splice site types
            donor_count = (gene_data['splice_type'] == 'donor').sum()
            acceptor_count = (gene_data['splice_type'] == 'acceptor').sum()
            neither_count = (gene_data['splice_type'] == 'neither').sum()
            total_positions = len(gene_data)
            
            # Get gene name if available
            gene_name = gene_id  # Default to gene_id
            if self.gene_features is not None:
                gene_info = self.gene_features[self.gene_features['gene_id'] == gene_id]
                if len(gene_info) > 0 and 'gene_name' in gene_info.columns:
                    name = gene_info['gene_name'].iloc[0]
                    if pd.notna(name) and name != gene_id:
                        gene_name = name
            
            # Get transcript count if available
            transcript_count = 0
            if self.transcript_features is not None:
                transcript_count = len(self.transcript_features[self.transcript_features['gene_id'] == gene_id])
            
            gene_stats.append({
                'gene_id': gene_id,
                'gene_name': gene_name,
                'donor_sites': donor_count,
                'acceptor_sites': acceptor_count,
                'neither_sites': neither_count,
                'total_positions': total_positions,
                'total_splice_sites': donor_count + acceptor_count,
                'transcript_count': transcript_count,
                'splice_site_density': (donor_count + acceptor_count) / total_positions if total_positions > 0 else 0
            })
        
        gene_stats_df = pd.DataFrame(gene_stats)
        self.log(f"Analyzed {len(gene_stats_df)} genes")
        
        return gene_stats_df
    
    def calculate_fp_fn_rates(self, df: pd.DataFrame, cv_results: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
        """Calculate false positive and false negative rates from CV results."""
        self.log("Calculating FP/FN rates from CV results...")
        
        # Merge dataset with CV results
        # CV results columns: base_donor_score, base_acceptor_score, meta_donor_prob, meta_acceptor_prob
        merged = df.merge(cv_results[['gene_id', 'position', 'true_label', 'base_donor_score', 'base_acceptor_score', 
                                     'meta_donor_prob', 'meta_acceptor_prob']], 
                         on=['gene_id', 'position'], how='inner')
        
        gene_error_stats = []
        
        for gene_id in merged['gene_id'].unique():
            gene_data = merged[merged['gene_id'] == gene_id]
            
            # Normalize true_label to ensure consistent string encoding
            gene_data = gene_data.copy()
            
            # First, convert to string to handle None/null values
            gene_data['true_label'] = gene_data['true_label'].astype(str)
            
            # Apply fail-safe logic: if not donor or acceptor, then it's neither
            # This covers None (becomes "None"), "0", 0, "nan", etc.
            gene_data['true_label'] = gene_data['true_label'].apply(
                lambda x: x if x in ['donor', 'acceptor'] else 'neither'
            )
            
            # True labels - use normalized true_label from CV results
            true_donors = gene_data['true_label'] == 'donor'
            true_acceptors = gene_data['true_label'] == 'acceptor'
            
            # Base model predictions
            base_pred_donors = gene_data['base_donor_score'] > threshold
            base_pred_acceptors = gene_data['base_acceptor_score'] > threshold
            
            # Meta model predictions
            meta_pred_donors = gene_data['meta_donor_prob'] > threshold
            meta_pred_acceptors = gene_data['meta_acceptor_prob'] > threshold
            
            # Calculate errors for donors
            base_donor_fp = (~true_donors & base_pred_donors).sum()
            base_donor_fn = (true_donors & ~base_pred_donors).sum()
            meta_donor_fp = (~true_donors & meta_pred_donors).sum()
            meta_donor_fn = (true_donors & ~meta_pred_donors).sum()
            
            # Calculate errors for acceptors
            base_acceptor_fp = (~true_acceptors & base_pred_acceptors).sum()
            base_acceptor_fn = (true_acceptors & ~base_pred_acceptors).sum()
            meta_acceptor_fp = (~true_acceptors & meta_pred_acceptors).sum()
            meta_acceptor_fn = (true_acceptors & ~meta_pred_acceptors).sum()
            
            # Calculate improvements
            rescued_donor_fn = base_donor_fn - meta_donor_fn
            eliminated_donor_fp = base_donor_fp - meta_donor_fp
            rescued_acceptor_fn = base_acceptor_fn - meta_acceptor_fn
            eliminated_acceptor_fp = base_acceptor_fp - meta_acceptor_fp
            
            gene_error_stats.append({
                'gene_id': gene_id,
                'base_donor_fp': base_donor_fp,
                'base_donor_fn': base_donor_fn,
                'meta_donor_fp': meta_donor_fp,
                'meta_donor_fn': meta_donor_fn,
                'rescued_donor_fn': rescued_donor_fn,
                'eliminated_donor_fp': eliminated_donor_fp,
                'base_acceptor_fp': base_acceptor_fp,
                'base_acceptor_fn': base_acceptor_fn,
                'meta_acceptor_fp': meta_acceptor_fp,
                'meta_acceptor_fn': meta_acceptor_fn,
                'rescued_acceptor_fn': rescued_acceptor_fn,
                'eliminated_acceptor_fp': eliminated_acceptor_fp,
                'total_improvements': rescued_donor_fn + eliminated_donor_fp + rescued_acceptor_fn + eliminated_acceptor_fp
            })
        
        error_stats_df = pd.DataFrame(gene_error_stats)
        self.log(f"Calculated error rates for {len(error_stats_df)} genes")
        
        return error_stats_df
    
    def create_gene_ranking(self, gene_stats: pd.DataFrame, error_stats: pd.DataFrame = None, 
                           min_splice_sites: int = 10) -> pd.DataFrame:
        """Create ranked list of genes suitable for visualization."""
        self.log("Creating gene ranking for visualization suitability...")
        
        # Start with gene stats
        ranking = gene_stats.copy()
        
        # Filter genes with sufficient splice sites
        ranking = ranking[ranking['total_splice_sites'] >= min_splice_sites]
        
        # Merge with error stats if available
        if error_stats is not None:
            ranking = ranking.merge(error_stats, on='gene_id', how='left')
            ranking['total_improvements'] = ranking['total_improvements'].fillna(0)
        
        # Calculate suitability score
        # Factors: splice site count, diversity (both donors and acceptors), improvements
        ranking['donor_acceptor_balance'] = np.minimum(ranking['donor_sites'], ranking['acceptor_sites'])
        ranking['suitability_score'] = (
            ranking['total_splice_sites'] * 0.4 +  # Raw count
            ranking['donor_acceptor_balance'] * 0.3 +  # Balance between donors/acceptors
            ranking['splice_site_density'] * 100 * 0.2  # Density
        )
        
        if error_stats is not None:
            ranking['suitability_score'] += ranking['total_improvements'] * 0.1  # Improvements
        
        # Sort by suitability score
        ranking = ranking.sort_values('suitability_score', ascending=False)
        
        self.log(f"Ranked {len(ranking)} genes by visualization suitability")
        
        return ranking
    
    def save_gene_recommendations(self, ranking: pd.DataFrame, output_path: str, top_n: int = 50) -> None:
        """Save gene recommendations to file."""
        
        # Select top N genes
        top_genes = ranking.head(top_n)
        
        # Select relevant columns for output
        output_cols = ['gene_id', 'gene_name', 'donor_sites', 'acceptor_sites', 'total_splice_sites', 
                      'transcript_count', 'splice_site_density', 'suitability_score']
        
        # Add error columns if available
        if 'total_improvements' in top_genes.columns:
            output_cols.extend(['base_donor_fp', 'base_donor_fn', 'base_acceptor_fp', 'base_acceptor_fn',
                               'rescued_donor_fn', 'eliminated_donor_fp', 'rescued_acceptor_fn', 
                               'eliminated_acceptor_fp', 'total_improvements'])
        
        # Save to file
        output_df = top_genes[output_cols].round(3)
        output_df.to_csv(output_path, sep='\t', index=False)
        
        self.log(f"Saved top {len(output_df)} gene recommendations to: {output_path}")
        
        # Print summary
        print(f"\n🧬 Top {min(10, len(output_df))} genes for splice site visualization:")
        print("=" * 80)
        for i, row in output_df.head(10).iterrows():
            improvements = f", improvements: {int(row['total_improvements'])}" if 'total_improvements' in row else ""
            print(f"{row['gene_name']} ({row['gene_id']}): {int(row['donor_sites'])} donors, "
                  f"{int(row['acceptor_sites'])} acceptors{improvements}")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Discover genes suitable for splice site visualization testing",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--dataset', required=True, help='Path to dataset directory')
    parser.add_argument('--cv-results', help='Path to CV results file for FP/FN analysis')
    parser.add_argument('--output', default='genes_for_splice_site_testing.tsv', 
                       help='Output file for gene recommendations')
    parser.add_argument('--top-n', type=int, default=50, 
                       help='Number of top genes to include in output')
    parser.add_argument('--min-splice-sites', type=int, default=10,
                       help='Minimum number of splice sites required per gene')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Probability threshold for FP/FN calculation')
    parser.add_argument('--gene-features', default='data/ensembl/spliceai_analysis/gene_features.tsv',
                       help='Path to gene features file')
    parser.add_argument('--transcript-features', default='data/ensembl/spliceai_analysis/transcript_features.tsv',
                       help='Path to transcript features file')
    parser.add_argument('--splice-sites', default='data/ensembl/splice_sites.tsv',
                       help='Path to splice sites annotation file')
    parser.add_argument('--target-genes', help='Comma-separated list of specific genes to analyze')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Create gene discovery instance
    discovery = SpliceSiteGeneDiscovery(verbose=args.verbose)
    
    try:
        # Load metadata
        discovery.load_gene_metadata(args.gene_features, args.transcript_features, args.splice_sites)
        
        # Parse target genes if provided
        target_genes = None
        if args.target_genes:
            target_genes = [g.strip() for g in args.target_genes.split(',')]
            discovery.log(f"Analyzing specific target genes: {target_genes}")
        
        # Load dataset
        df = discovery.load_dataset_genes(args.dataset, target_genes)
        
        # Analyze splice sites per gene
        gene_stats = discovery.analyze_gene_splice_sites(df)
        
        # Calculate FP/FN rates if CV results provided
        error_stats = None
        if args.cv_results and Path(args.cv_results).exists():
            cv_results = pd.read_csv(args.cv_results, sep='\t')
            error_stats = discovery.calculate_fp_fn_rates(df, cv_results, args.threshold)
        
        # Create gene ranking
        ranking = discovery.create_gene_ranking(gene_stats, error_stats, args.min_splice_sites)
        
        # Save recommendations
        discovery.save_gene_recommendations(ranking, args.output, args.top_n)
        
        discovery.log("Gene discovery completed successfully!")
        
    except Exception as e:
        discovery.log(f"Error during gene discovery: {e}", "ERROR")
        raise


if __name__ == "__main__":
    main() 