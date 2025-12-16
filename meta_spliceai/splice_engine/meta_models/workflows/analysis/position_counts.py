#!/usr/bin/env python3
"""
Position Count Analysis Script

This script analyzes position count discrepancies in SpliceAI inference results,
investigating:
1. Donor vs Acceptor position count differences
2. Total positions vs gene length consistency  
3. Systematic artifacts in sequence processing

The analysis helps understand why donor and acceptor position counts might differ
and whether final position counts match expected gene lengths.
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse
from dataclasses import dataclass
import logging

# Add the meta_spliceai package to the path
sys.path.insert(0, str(Path(__file__).parents[5]))

from meta_spliceai.splice_engine.meta_models.io.handlers import MetaModelDataHandler


@dataclass
class PositionCountStats:
    """Statistics for position count analysis."""
    gene_id: str
    gene_length: int
    donor_positions: int
    acceptor_positions: int
    total_raw_positions: int
    final_unique_positions: int
    
    # Derived metrics
    donor_acceptor_diff: int = 0
    position_gene_length_diff: int = 0
    donor_acceptor_ratio: float = 0.0
    position_coverage_ratio: float = 0.0
    
    def __post_init__(self):
        """Calculate derived metrics."""
        self.donor_acceptor_diff = self.donor_positions - self.acceptor_positions
        self.position_gene_length_diff = self.final_unique_positions - self.gene_length
        self.donor_acceptor_ratio = self.donor_positions / max(self.acceptor_positions, 1)
        self.position_coverage_ratio = self.final_unique_positions / max(self.gene_length, 1)


def analyze_position_counts(gene_ids: List[str], 
                          model_path: str = None,
                          training_dataset: str = None,
                          output_file: str = None) -> List[PositionCountStats]:
    """
    Convenience function to analyze position counts for multiple genes.
    
    Parameters
    ----------
    gene_ids : List[str]
        List of gene IDs to analyze
    model_path : str, optional
        Path to trained model
    training_dataset : str, optional
        Training dataset name
    output_file : str, optional
        Output file for analysis report
        
    Returns
    -------
    List[PositionCountStats]
        Position count statistics for all genes
    """
    analyzer = PositionCountAnalyzer(verbose=1)
    stats_list = analyzer.analyze_multiple_genes(gene_ids, model_path, training_dataset)
    
    if output_file:
        analyzer.generate_analysis_report(stats_list, output_file)
    
    return stats_list


class PositionCountAnalyzer:
    """Analyzer for position count discrepancies in SpliceAI inference."""
    
    def __init__(self, 
                 gene_features_path: str = "data/ensembl/spliceai_analysis/gene_features.tsv",
                 verbose: int = 1):
        """
        Initialize the position count analyzer.
        
        Parameters
        ----------
        gene_features_path : str
            Path to gene features file containing gene lengths
        verbose : int
            Verbosity level (0=quiet, 1=info, 2=debug)
        """
        self.gene_features_path = gene_features_path
        self.verbose = verbose
        self.logger = self._setup_logging()
        
        # Load gene features for length information
        self.gene_features = self._load_gene_features()
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.DEBUG if self.verbose >= 2 else logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def _load_gene_features(self) -> pd.DataFrame:
        """Load gene features containing gene lengths."""
        try:
            df = pd.read_csv(self.gene_features_path, sep='\t')
            if 'gene_length' in df.columns:
                length_col = 'gene_length'
            elif 'length' in df.columns:
                length_col = 'length'
            else:
                # Assume column 8 (0-indexed column 7) contains length
                length_col = df.columns[7] if len(df.columns) > 7 else None
                
            if length_col is None:
                raise ValueError("Could not identify gene length column")
                
            # Standardize column name
            if length_col != 'gene_length':
                df = df.rename(columns={length_col: 'gene_length'})
                
            self.logger.info(f"Loaded {len(df)} gene features from {self.gene_features_path}")
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to load gene features: {e}")
            return pd.DataFrame(columns=['gene_id', 'gene_length'])
    
    def get_gene_length(self, gene_id: str) -> int:
        """Get the length of a specific gene."""
        if self.gene_features.empty:
            return 0
            
        # Try different possible gene ID column names
        id_cols = ['gene_id', 'ensembl_gene_id', 'id']
        gene_row = None
        
        for col in id_cols:
            if col in self.gene_features.columns:
                gene_row = self.gene_features[self.gene_features[col] == gene_id]
                if not gene_row.empty:
                    break
        
        if gene_row is None or gene_row.empty:
            self.logger.warning(f"Gene {gene_id} not found in gene features")
            return 0
            
        return int(gene_row['gene_length'].iloc[0])
    
    def analyze_single_gene(self, gene_id: str, 
                           model_path: str = None,
                           training_dataset: str = None) -> PositionCountStats:
        """
        Analyze position counts for a single gene.
        
        Parameters
        ----------
        gene_id : str
            Ensembl gene ID to analyze
        model_path : str, optional
            Path to trained model (for live analysis)
        training_dataset : str, optional
            Training dataset name (for live analysis)
            
        Returns
        -------
        PositionCountStats
            Position count statistics for the gene
        """
        self.logger.info(f"Analyzing position counts for gene {gene_id}")
        
        # Get gene length from features
        gene_length = self.get_gene_length(gene_id)
        
        if model_path and training_dataset:
            # Perform live analysis using SpliceAI inference
            stats = self._analyze_gene_live(gene_id, gene_length, model_path, training_dataset)
        else:
            # Analyze from existing results (if available)
            stats = self._analyze_gene_from_results(gene_id, gene_length)
            
        return stats
    
    def _analyze_gene_live(self, gene_id: str, gene_length: int,
                          model_path: str, training_dataset: str) -> PositionCountStats:
        """Perform live analysis of a gene using SpliceAI inference."""
        try:
            # Load data handler for the gene
            data_handler = MetaModelDataHandler(training_dataset)
            gene_df = data_handler.get_gene_data([gene_id])
            
            if gene_df.empty:
                self.logger.error(f"No data found for gene {gene_id}")
                return PositionCountStats(
                    gene_id=gene_id, gene_length=gene_length,
                    donor_positions=0, acceptor_positions=0,
                    total_raw_positions=0, final_unique_positions=0
                )
            
            # Load models (simplified - would need proper model loading)
            # This is a placeholder - actual implementation would load trained models
            models = []  # Would load actual models here
            
            # Run SpliceAI prediction with verbose output to capture position counts
            predictions = predict_splice_sites_for_genes_v3(
                gene_df, models, context=10000, output_format='dict', verbose=2
            )
            
            # Run enhanced evaluation to get position count breakdown
            ss_annotations_df = data_handler.get_splice_site_annotations([gene_id])
            error_df, positions_df = enhanced_evaluate_splice_site_errors(
                ss_annotations_df, predictions, verbose=2, return_positions_df=True
            )
            
            # Extract position counts from the evaluation
            # This would capture the verbose output that shows donor/acceptor counts
            donor_positions = 0  # Would extract from verbose output
            acceptor_positions = 0  # Would extract from verbose output
            total_raw_positions = donor_positions + acceptor_positions
            final_unique_positions = len(positions_df) if positions_df is not None else 0
            
            return PositionCountStats(
                gene_id=gene_id, gene_length=gene_length,
                donor_positions=donor_positions, acceptor_positions=acceptor_positions,
                total_raw_positions=total_raw_positions, final_unique_positions=final_unique_positions
            )
            
        except Exception as e:
            self.logger.error(f"Failed to analyze gene {gene_id} live: {e}")
            return PositionCountStats(
                gene_id=gene_id, gene_length=gene_length,
                donor_positions=0, acceptor_positions=0,
                total_raw_positions=0, final_unique_positions=0
            )
    
    def _analyze_gene_from_results(self, gene_id: str, gene_length: int) -> PositionCountStats:
        """Analyze gene from existing result files."""
        # This would parse existing result files to extract position counts
        # For now, return a placeholder
        self.logger.warning(f"Analysis from existing results not yet implemented for {gene_id}")
        return PositionCountStats(
            gene_id=gene_id, gene_length=gene_length,
            donor_positions=0, acceptor_positions=0,
            total_raw_positions=0, final_unique_positions=0
        )
    
    def analyze_multiple_genes(self, gene_ids: List[str],
                              model_path: str = None,
                              training_dataset: str = None) -> List[PositionCountStats]:
        """
        Analyze position counts for multiple genes.
        
        Parameters
        ----------
        gene_ids : List[str]
            List of Ensembl gene IDs to analyze
        model_path : str, optional
            Path to trained model
        training_dataset : str, optional
            Training dataset name
            
        Returns
        -------
        List[PositionCountStats]
            Position count statistics for all genes
        """
        self.logger.info(f"Analyzing position counts for {len(gene_ids)} genes")
        
        results = []
        for i, gene_id in enumerate(gene_ids, 1):
            self.logger.info(f"Processing gene {i}/{len(gene_ids)}: {gene_id}")
            stats = self.analyze_single_gene(gene_id, model_path, training_dataset)
            results.append(stats)
            
        return results
    
    def generate_analysis_report(self, stats_list: List[PositionCountStats],
                               output_file: str = None) -> Dict:
        """
        Generate a comprehensive analysis report.
        
        Parameters
        ----------
        stats_list : List[PositionCountStats]
            List of position count statistics
        output_file : str, optional
            Path to save the report
            
        Returns
        -------
        Dict
            Analysis summary and statistics
        """
        if not stats_list:
            self.logger.warning("No statistics to analyze")
            return {}
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame([
            {
                'gene_id': s.gene_id,
                'gene_length': s.gene_length,
                'donor_positions': s.donor_positions,
                'acceptor_positions': s.acceptor_positions,
                'total_raw_positions': s.total_raw_positions,
                'final_unique_positions': s.final_unique_positions,
                'donor_acceptor_diff': s.donor_acceptor_diff,
                'position_gene_length_diff': s.position_gene_length_diff,
                'donor_acceptor_ratio': s.donor_acceptor_ratio,
                'position_coverage_ratio': s.position_coverage_ratio
            }
            for s in stats_list
        ])
        
        # Calculate summary statistics
        summary = {
            'total_genes_analyzed': len(df),
            'genes_with_data': len(df[df['final_unique_positions'] > 0]),
            
            # Donor vs Acceptor Analysis
            'donor_acceptor_differences': {
                'mean_diff': df['donor_acceptor_diff'].mean(),
                'std_diff': df['donor_acceptor_diff'].std(),
                'median_diff': df['donor_acceptor_diff'].median(),
                'max_abs_diff': df['donor_acceptor_diff'].abs().max(),
                'genes_with_diff': len(df[df['donor_acceptor_diff'] != 0]),
                'symmetry_percentage': len(df[df['donor_acceptor_diff'] == 0]) / len(df) * 100
            },
            
            # Position vs Gene Length Analysis
            'position_gene_length_differences': {
                'mean_diff': df['position_gene_length_diff'].mean(),
                'std_diff': df['position_gene_length_diff'].std(),
                'median_diff': df['position_gene_length_diff'].median(),
                'max_abs_diff': df['position_gene_length_diff'].abs().max(),
                'perfect_match_genes': len(df[df['position_gene_length_diff'] == 0]),
                'off_by_one_genes': len(df[df['position_gene_length_diff'].abs() == 1]),
                'coverage_ratio_mean': df['position_coverage_ratio'].mean()
            },
            
            # Distribution Analysis
            'gene_length_distribution': {
                'min': df['gene_length'].min(),
                'max': df['gene_length'].max(),
                'mean': df['gene_length'].mean(),
                'median': df['gene_length'].median()
            }
        }
        
        # Generate detailed report
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("🧬 POSITION COUNT ANALYSIS REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"📊 Analyzed {summary['total_genes_analyzed']} genes")
        report_lines.append(f"✅ Genes with data: {summary['genes_with_data']}")
        report_lines.append("")
        
        # Donor vs Acceptor Analysis
        da_stats = summary['donor_acceptor_differences']
        report_lines.append("🔍 DONOR vs ACCEPTOR POSITION ANALYSIS")
        report_lines.append("-" * 50)
        report_lines.append(f"• Symmetry rate: {da_stats['symmetry_percentage']:.1f}% of genes have equal donor/acceptor counts")
        report_lines.append(f"• Mean difference: {da_stats['mean_diff']:.2f} ± {da_stats['std_diff']:.2f}")
        report_lines.append(f"• Median difference: {da_stats['median_diff']:.0f}")
        report_lines.append(f"• Maximum absolute difference: {da_stats['max_abs_diff']:.0f}")
        report_lines.append(f"• Genes with asymmetry: {da_stats['genes_with_diff']}/{summary['total_genes_analyzed']}")
        report_lines.append("")
        
        # Position vs Gene Length Analysis  
        pl_stats = summary['position_gene_length_differences']
        report_lines.append("📏 POSITION COUNT vs GENE LENGTH ANALYSIS")
        report_lines.append("-" * 50)
        report_lines.append(f"• Perfect matches: {pl_stats['perfect_match_genes']}/{summary['total_genes_analyzed']} genes")
        report_lines.append(f"• Off-by-one matches: {pl_stats['off_by_one_genes']}/{summary['total_genes_analyzed']} genes")
        report_lines.append(f"• Mean coverage ratio: {pl_stats['coverage_ratio_mean']:.4f}")
        report_lines.append(f"• Mean difference: {pl_stats['mean_diff']:.2f} ± {pl_stats['std_diff']:.2f}")
        report_lines.append(f"• Maximum absolute difference: {pl_stats['max_abs_diff']:.0f}")
        report_lines.append("")
        
        # Gene Length Distribution
        gl_stats = summary['gene_length_distribution']
        report_lines.append("📈 GENE LENGTH DISTRIBUTION")
        report_lines.append("-" * 50)
        report_lines.append(f"• Range: {gl_stats['min']:,} - {gl_stats['max']:,} bp")
        report_lines.append(f"• Mean: {gl_stats['mean']:,.0f} bp")
        report_lines.append(f"• Median: {gl_stats['median']:,.0f} bp")
        report_lines.append("")
        
        # Detailed gene-by-gene analysis (top 10 most discrepant)
        report_lines.append("🎯 TOP DISCREPANCIES")
        report_lines.append("-" * 50)
        
        # Sort by absolute donor-acceptor difference
        top_da_discrepancies = df.nlargest(5, 'donor_acceptor_diff')
        if not top_da_discrepancies.empty:
            report_lines.append("Top Donor-Acceptor Asymmetries:")
            for _, row in top_da_discrepancies.iterrows():
                report_lines.append(f"  • {row['gene_id']}: {row['donor_positions']} donor, {row['acceptor_positions']} acceptor (diff: {row['donor_acceptor_diff']})")
        
        report_lines.append("")
        
        # Sort by absolute position-length difference  
        df_abs_diff = df.copy()
        df_abs_diff['abs_position_gene_length_diff'] = df['position_gene_length_diff'].abs()
        top_pl_discrepancies = df_abs_diff.nlargest(5, 'abs_position_gene_length_diff')
        if not top_pl_discrepancies.empty:
            report_lines.append("Top Position-Length Discrepancies:")
            for _, row in top_pl_discrepancies.iterrows():
                report_lines.append(f"  • {row['gene_id']}: {row['final_unique_positions']} positions, {row['gene_length']} bp (diff: {row['position_gene_length_diff']})")
        
        report_lines.append("=" * 80)
        
        report_text = "\n".join(report_lines)
        
        # Save report if output file specified
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_text)
            self.logger.info(f"Report saved to {output_file}")
        
        # Print report to console
        print(report_text)
        
        return {
            'summary': summary,
            'detailed_stats': df,
            'report_text': report_text
        }


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description='Analyze position count discrepancies in SpliceAI inference')
    parser.add_argument('--genes', nargs='+', required=True,
                       help='List of gene IDs to analyze')
    parser.add_argument('--gene-features', default='data/ensembl/spliceai_analysis/gene_features.tsv',
                       help='Path to gene features file')
    parser.add_argument('--model-path', 
                       help='Path to trained model for live analysis')
    parser.add_argument('--training-dataset',
                       help='Training dataset name for live analysis')
    parser.add_argument('--output', 
                       help='Output file for analysis report')
    parser.add_argument('--verbose', '-v', action='count', default=1,
                       help='Increase verbosity level')
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = PositionCountAnalyzer(
        gene_features_path=args.gene_features,
        verbose=args.verbose
    )
    
    # Analyze genes
    stats_list = analyzer.analyze_multiple_genes(
        args.genes,
        model_path=args.model_path,
        training_dataset=args.training_dataset
    )
    
    # Generate report
    analyzer.generate_analysis_report(stats_list, args.output)


if __name__ == '__main__':
    main()
