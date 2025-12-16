#!/usr/bin/env python3
"""Genomic browser-quality visualization for splice sites along gene sequences.

This script creates detailed visualizations showing the distribution of splice sites
along gene sequences, including:
- Base model predictions vs true splice sites
- Meta-model predictions (when available)
- Relative positions within genes
- Strand-dependent coordinate transformations

Memory optimization: K-mer features are filtered out by default since they're not
needed for visualization, reducing memory usage significantly.

Usage
-----
python -m meta_spliceai.splice_engine.meta_models.analysis.genomic_splice_visualizer \
    --dataset train_pc_1000/master \
    --gene-id ENSG00000205592 \
    --output-dir results/genomic_viz \
    --show-predictions

Example outputs:
- Gene track with splice sites marked
- Position-wise probability plots
- Comparison of base vs meta-model predictions
- Multi-gene overview plots
"""
from __future__ import annotations

import argparse
import logging
import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection
import seaborn as sns

# Data loading
from meta_spliceai.splice_engine.meta_models.training import datasets
from meta_spliceai.splice_engine.meta_models.training.classifier_utils import _encode_labels

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class GenomicSpliceVisualizer:
    """High-quality genomic visualizations for splice site analysis."""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.gene_features = None
        
    def log(self, message: str, level: str = "INFO"):
        """Log messages with appropriate level."""
        if self.verbose:
            if level == "INFO":
                logger.info(message)
            elif level == "WARNING":
                logger.warning(message)
            elif level == "ERROR":
                logger.error(message)
    
    def load_dataset(self, dataset_path: str, filter_kmers: bool = True) -> pd.DataFrame:
        """Load dataset with coordinate information, optionally filtering k-mer features."""
        self.log(f"Loading dataset from: {dataset_path}")
        
        try:
            df = datasets.load_dataset(dataset_path)
            
            # Convert to pandas if it's a polars DataFrame
            if hasattr(df, 'to_pandas'):
                self.log("Converting Polars DataFrame to Pandas...")
                df = df.to_pandas()
            
            original_cols = len(df.columns)
            original_memory = df.memory_usage(deep=True).sum() / (1024**2)
            
            # Filter out k-mer features to reduce memory usage
            if filter_kmers:
                kmer_patterns = ['6mer_', 'kmer_', '_mer_', 'mer_']
                kmer_cols = []
                for col in df.columns:
                    if any(pattern in col.lower() for pattern in kmer_patterns):
                        kmer_cols.append(col)
                
                if kmer_cols:
                    self.log(f"Filtering out {len(kmer_cols)} k-mer features to reduce memory usage...")
                    df = df.drop(columns=kmer_cols)
                    new_memory = df.memory_usage(deep=True).sum() / (1024**2)
                    memory_saved = original_memory - new_memory
                    self.log(f"Memory usage reduced by {memory_saved:.1f} MB ({memory_saved/original_memory*100:.1f}%)")
            
            self.log(f"Successfully loaded {len(df):,} samples with {len(df.columns)} columns")
            if filter_kmers:
                self.log(f"Filtered from {original_cols} to {len(df.columns)} columns")
            
            return df
        except Exception as e:
            self.log(f"Error loading dataset: {e}", "ERROR")
            raise
    
    def load_gene_features(self, gene_features_path: str = "data/ensembl/spliceai_analysis/gene_features.tsv") -> pd.DataFrame:
        """Load gene boundary information."""
        try:
            if os.path.exists(gene_features_path):
                self.log(f"Loading gene features from: {gene_features_path}")
                gene_features = pd.read_csv(gene_features_path, sep='\t')
                self.gene_features = gene_features
                self.log(f"Loaded {len(gene_features)} gene features")
                return gene_features
            else:
                self.log(f"Gene features file not found at: {gene_features_path}", "WARNING")
                return None
        except Exception as e:
            self.log(f"Error loading gene features: {e}", "WARNING")
            return None
    
    def calculate_absolute_coordinates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate absolute genomic coordinates from relative positions."""
        self.log("Calculating absolute genomic coordinates...")
        
        required_cols = ['gene_start', 'gene_end', 'position']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            self.log(f"Missing required columns for coordinate calculation: {missing_cols}", "ERROR")
            return df
        
        # If strand is missing, try to merge with gene features (optimize memory)
        if 'strand' not in df.columns and self.gene_features is not None:
            self.log("Merging with gene features to get strand information...")
            
            # Get unique genes to reduce merge size
            unique_genes = df['gene_id'].unique()
            gene_strand_map = self.gene_features[self.gene_features['gene_id'].isin(unique_genes)][['gene_id', 'strand']]
            
            # Create a lookup dictionary for efficiency
            strand_dict = dict(zip(gene_strand_map['gene_id'], gene_strand_map['strand']))
            
            # Map strand information
            df['strand'] = df['gene_id'].map(strand_dict).fillna('+')
            
            if df['strand'].isna().any():
                self.log("Some genes missing strand information, defaulting to '+'", "WARNING")
                df['strand'] = df['strand'].fillna('+')
        
        df_coord = df.copy()
        
        # Calculate absolute position based on strand
        def calc_absolute_pos(row):
            strand = row.get('strand', '+')  # Default to positive strand
            if strand == '+':
                return row['gene_start'] + row['position']
            else:  # strand == '-'
                return row['gene_end'] - row['position']
        
        df_coord['absolute_position'] = df_coord.apply(calc_absolute_pos, axis=1)
        
        # Also calculate relative position as fraction of gene length
        df_coord['gene_length'] = df_coord['gene_end'] - df_coord['gene_start']
        df_coord['relative_position_fraction'] = df_coord['position'] / df_coord['gene_length']
        
        self.log("Absolute coordinates calculated successfully")
        return df_coord
    
    def get_gene_data(self, df: pd.DataFrame, gene_id: str) -> pd.DataFrame:
        """Extract data for a specific gene."""
        gene_data = df[df['gene_id'] == gene_id].copy()
        if len(gene_data) == 0:
            raise ValueError(f"Gene {gene_id} not found in dataset")
        
        # Sort by position for proper visualization
        gene_data = gene_data.sort_values('position')
        
        self.log(f"Found {len(gene_data)} positions for gene {gene_id}")
        return gene_data
    
    def create_gene_track_plot(self, gene_data: pd.DataFrame, gene_id: str, 
                              output_path: Path, figsize: Tuple[int, int] = (16, 8)) -> str:
        """Create a detailed gene track visualization."""
        self.log(f"Creating gene track plot for {gene_id}...")
        
        fig, (ax_gene, ax_probs) = plt.subplots(2, 1, figsize=figsize, height_ratios=[1, 2])
        
        # Get gene information
        gene_start = gene_data['gene_start'].iloc[0]
        gene_end = gene_data['gene_end'].iloc[0]
        gene_length = gene_end - gene_start
        strand = gene_data['strand'].iloc[0] if 'strand' in gene_data.columns else '+'
        chrom = gene_data['chrom'].iloc[0] if 'chrom' in gene_data.columns else 'Unknown'
        
        # === Gene Track (Top panel) ===
        ax_gene.set_xlim(0, gene_length)
        ax_gene.set_ylim(0, 1)
        
        # Draw gene body
        gene_rect = patches.Rectangle((0, 0.3), gene_length, 0.4, 
                                    facecolor='lightblue', edgecolor='blue', alpha=0.7)
        ax_gene.add_patch(gene_rect)
        
        # Add strand arrow
        arrow_y = 0.5
        if strand == '+':
            ax_gene.annotate('', xy=(gene_length * 0.95, arrow_y), xytext=(gene_length * 0.85, arrow_y),
                           arrowprops=dict(arrowstyle='->', lw=2, color='blue'))
        else:
            ax_gene.annotate('', xy=(gene_length * 0.05, arrow_y), xytext=(gene_length * 0.15, arrow_y),
                           arrowprops=dict(arrowstyle='->', lw=2, color='blue'))
        
        # Mark splice sites
        splice_sites = gene_data[gene_data['splice_type'].isin(['donor', 'acceptor'])]
        
        for _, site in splice_sites.iterrows():
            pos = site['position']
            site_type = site['splice_type']
            color = 'red' if site_type == 'donor' else 'green'
            marker = '|' if site_type == 'donor' else '|'
            
            ax_gene.axvline(x=pos, ymin=0.1, ymax=0.9, color=color, linewidth=2, alpha=0.8)
            ax_gene.text(pos, 0.95, site_type[0].upper(), ha='center', va='bottom', 
                        color=color, fontweight='bold', fontsize=8)
        
        # Gene track labels and formatting
        ax_gene.set_title(f'Gene {gene_id} (Chr{chrom}, {strand} strand)\n'
                         f'Position: {gene_start:,} - {gene_end:,} (Length: {gene_length:,} bp)', 
                         fontsize=12, fontweight='bold')
        ax_gene.set_ylabel('Gene Track')
        ax_gene.set_xticks([])
        ax_gene.spines['top'].set_visible(False)
        ax_gene.spines['right'].set_visible(False)
        ax_gene.spines['bottom'].set_visible(False)
        
        # === Probability Scores (Bottom panel) ===
        positions = gene_data['position'].values
        
        # Plot probability scores if available
        prob_cols = ['donor_score', 'acceptor_score', 'neither_score']
        colors = ['red', 'green', 'blue']
        labels = ['Donor', 'Acceptor', 'Neither']
        
        for prob_col, color, label in zip(prob_cols, colors, labels):
            if prob_col in gene_data.columns:
                ax_probs.plot(positions, gene_data[prob_col], color=color, alpha=0.7, 
                            linewidth=1, label=f'{label} Score')
        
        # Highlight actual splice sites
        for _, site in splice_sites.iterrows():
            pos = site['position']
            site_type = site['splice_type']
            color = 'red' if site_type == 'donor' else 'green'
            
            ax_probs.axvline(x=pos, color=color, linestyle='--', alpha=0.8, linewidth=1)
            
            # Add site label
            y_pos = 0.9 if site_type == 'donor' else 0.8
            ax_probs.text(pos, y_pos, site_type[0].upper(), ha='center', va='center',
                         bbox=dict(boxstyle='round,pad=0.2', facecolor=color, alpha=0.7),
                         fontsize=8, color='white', fontweight='bold')
        
        # Format probability plot
        ax_probs.set_xlim(0, gene_length)
        ax_probs.set_ylim(0, 1)
        ax_probs.set_xlabel('Position in Gene (bp)')
        ax_probs.set_ylabel('Probability Score')
        ax_probs.legend(loc='upper right')
        ax_probs.grid(True, alpha=0.3)
        
        # Add position ticks
        n_ticks = 10
        tick_positions = np.linspace(0, gene_length, n_ticks)
        ax_probs.set_xticks(tick_positions)
        ax_probs.set_xticklabels([f'{int(pos/1000)}k' for pos in tick_positions])
        
        plt.tight_layout()
        
        # Save plot
        plot_file = output_path / f"gene_track_{gene_id}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.log(f"Gene track plot saved: {plot_file}")
        return str(plot_file)
    
    def create_splice_density_plot(self, gene_data: pd.DataFrame, gene_id: str,
                                 output_path: Path, window_size: int = 1000) -> str:
        """Create splice site density plot along the gene."""
        self.log(f"Creating splice density plot for {gene_id}...")
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        gene_length = gene_data['gene_end'].iloc[0] - gene_data['gene_start'].iloc[0]
        
        # Create position bins
        n_bins = max(50, gene_length // window_size)
        bins = np.linspace(0, gene_length, n_bins)
        
        # Get splice sites
        donors = gene_data[gene_data['splice_type'] == 'donor']['position']
        acceptors = gene_data[gene_data['splice_type'] == 'acceptor']['position']
        
        # Create histograms
        donor_hist, _ = np.histogram(donors, bins=bins)
        acceptor_hist, _ = np.histogram(acceptors, bins=bins)
        
        # Plot as step plots
        bin_centers = (bins[:-1] + bins[1:]) / 2
        ax.step(bin_centers, donor_hist, where='mid', color='red', alpha=0.7, 
               linewidth=2, label=f'Donor Sites (n={len(donors)})')
        ax.step(bin_centers, acceptor_hist, where='mid', color='green', alpha=0.7, 
               linewidth=2, label=f'Acceptor Sites (n={len(acceptors)})')
        
        # Fill areas for better visualization
        ax.fill_between(bin_centers, donor_hist, step='mid', alpha=0.3, color='red')
        ax.fill_between(bin_centers, acceptor_hist, step='mid', alpha=0.3, color='green')
        
        # Formatting
        ax.set_xlim(0, gene_length)
        ax.set_xlabel('Position in Gene (bp)')
        ax.set_ylabel(f'Splice Sites per {window_size}bp window')
        ax.set_title(f'Splice Site Density Distribution - Gene {gene_id}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Format x-axis
        n_ticks = 10
        tick_positions = np.linspace(0, gene_length, n_ticks)
        ax.set_xticks(tick_positions)
        ax.set_xticklabels([f'{int(pos/1000)}k' for pos in tick_positions])
        
        plt.tight_layout()
        
        # Save plot
        plot_file = output_path / f"splice_density_{gene_id}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.log(f"Splice density plot saved: {plot_file}")
        return str(plot_file)
    
    def create_prediction_comparison_plot(self, gene_data: pd.DataFrame, gene_id: str,
                                        output_path: Path, meta_predictions: pd.DataFrame = None) -> str:
        """Create comparison plot of base model vs meta-model predictions."""
        self.log(f"Creating prediction comparison plot for {gene_id}...")
        
        fig, axes = plt.subplots(3, 1, figsize=(16, 12))
        
        positions = gene_data['position'].values
        true_labels = _encode_labels(gene_data['splice_type'])
        
        # Base model predictions (argmax of probability scores)
        prob_cols = ['neither_score', 'donor_score', 'acceptor_score']
        if all(col in gene_data.columns for col in prob_cols):
            base_probs = gene_data[prob_cols].values
            base_preds = base_probs.argmax(axis=1)
        else:
            self.log("Missing probability columns for base model predictions", "WARNING")
            base_preds = np.zeros(len(gene_data))
        
        # === Panel 1: True vs Base Model ===
        ax = axes[0]
        
        # Plot probability scores
        for i, (col, color, label) in enumerate(zip(prob_cols, ['blue', 'red', 'green'], 
                                                   ['Neither', 'Donor', 'Acceptor'])):
            if col in gene_data.columns:
                ax.plot(positions, gene_data[col], color=color, alpha=0.6, linewidth=1, label=f'{label} Score')
        
        # Mark true splice sites
        true_donors = positions[true_labels == 1]
        true_acceptors = positions[true_labels == 2]
        
        for pos in true_donors:
            ax.axvline(x=pos, color='red', linestyle='-', alpha=0.8, linewidth=2)
        for pos in true_acceptors:
            ax.axvline(x=pos, color='green', linestyle='-', alpha=0.8, linewidth=2)
        
        ax.set_title(f'Base Model Predictions vs True Labels - Gene {gene_id}')
        ax.set_ylabel('Probability')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # === Panel 2: Error Analysis ===
        ax = axes[1]
        
        # Calculate errors
        base_errors = (true_labels != base_preds)
        fp_positions = positions[(base_preds != 0) & (true_labels == 0)]  # False positives
        fn_positions = positions[(base_preds == 0) & (true_labels != 0)]  # False negatives
        
        # Plot error positions
        ax.scatter(fp_positions, np.ones(len(fp_positions)), color='orange', s=50, 
                  alpha=0.7, label=f'False Positives (n={len(fp_positions)})', marker='^')
        ax.scatter(fn_positions, np.zeros(len(fn_positions)), color='purple', s=50, 
                  alpha=0.7, label=f'False Negatives (n={len(fn_positions)})', marker='v')
        
        # Background probability
        max_prob = gene_data[['donor_score', 'acceptor_score']].max(axis=1)
        ax.plot(positions, max_prob, color='gray', alpha=0.3, linewidth=1, label='Max Splice Probability')
        
        ax.set_title('Base Model Errors')
        ax.set_ylabel('Error Type')
        ax.set_ylim(-0.5, 1.5)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # === Panel 3: Meta-Model Comparison (if available) ===
        ax = axes[2]
        
        if meta_predictions is not None:
            # Plot meta-model predictions
            meta_gene_data = meta_predictions[meta_predictions['gene_id'] == gene_id]
            if len(meta_gene_data) > 0:
                meta_pos = meta_gene_data['position'].values
                meta_probs = meta_gene_data[['neither_score', 'donor_score', 'acceptor_score']].values
                meta_preds = meta_probs.argmax(axis=1)
                
                # Plot meta probabilities
                for i, (color, label) in enumerate(zip(['blue', 'red', 'green'], 
                                                     ['Neither', 'Donor', 'Acceptor'])):
                    ax.plot(meta_pos, meta_probs[:, i], color=color, alpha=0.8, 
                           linewidth=1.5, linestyle='--', label=f'Meta {label}')
                
                # Compare errors
                meta_errors = positions[np.isin(positions, meta_pos)]
                if len(meta_errors) > 0:
                    # Calculate rescued sites (FN -> correct) and new FPs
                    rescued_sites = []  # Would need detailed comparison logic
                    ax.text(0.02, 0.95, f'Meta-model comparison available', 
                           transform=ax.transAxes, fontsize=10, 
                           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
            else:
                ax.text(0.5, 0.5, 'No meta-model predictions available for this gene', 
                       transform=ax.transAxes, ha='center', va='center', fontsize=12)
        else:
            ax.text(0.5, 0.5, 'Meta-model predictions not provided', 
                   transform=ax.transAxes, ha='center', va='center', fontsize=12)
        
        ax.set_title('Meta-Model Comparison')
        ax.set_xlabel('Position in Gene (bp)')
        ax.set_ylabel('Probability')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Format all x-axes
        gene_length = gene_data['gene_end'].iloc[0] - gene_data['gene_start'].iloc[0]
        for ax in axes:
            ax.set_xlim(0, gene_length)
            n_ticks = 8
            tick_positions = np.linspace(0, gene_length, n_ticks)
            ax.set_xticks(tick_positions)
            ax.set_xticklabels([f'{int(pos/1000)}k' for pos in tick_positions])
        
        plt.tight_layout()
        
        # Save plot
        plot_file = output_path / f"prediction_comparison_{gene_id}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.log(f"Prediction comparison plot saved: {plot_file}")
        return str(plot_file)
    
    def create_multi_gene_overview(self, df: pd.DataFrame, top_genes: List[str],
                                 output_path: Path, max_genes: int = 6) -> str:
        """Create overview plot showing multiple genes."""
        self.log(f"Creating multi-gene overview for {len(top_genes)} genes...")
        
        # Limit number of genes for readability
        genes_to_plot = top_genes[:max_genes]
        
        fig, axes = plt.subplots(len(genes_to_plot), 1, figsize=(16, 3*len(genes_to_plot)))
        if len(genes_to_plot) == 1:
            axes = [axes]
        
        for i, gene_id in enumerate(genes_to_plot):
            ax = axes[i]
            gene_data = self.get_gene_data(df, gene_id)
            
            if len(gene_data) == 0:
                continue
            
            positions = gene_data['position'].values
            gene_length = gene_data['gene_end'].iloc[0] - gene_data['gene_start'].iloc[0]
            
            # Plot probability scores as filled areas
            prob_cols = ['donor_score', 'acceptor_score']
            colors = ['red', 'green']
            labels = ['Donor', 'Acceptor']
            
            for prob_col, color, label in zip(prob_cols, colors, labels):
                if prob_col in gene_data.columns:
                    ax.fill_between(positions, 0, gene_data[prob_col], 
                                  color=color, alpha=0.4, label=f'{label} Score')
            
            # Mark actual splice sites
            splice_sites = gene_data[gene_data['splice_type'].isin(['donor', 'acceptor'])]
            for _, site in splice_sites.iterrows():
                pos = site['position']
                site_type = site['splice_type']
                color = 'red' if site_type == 'donor' else 'green'
                ax.axvline(x=pos, color=color, linewidth=1, alpha=0.8)
            
            # Format subplot
            ax.set_xlim(0, gene_length)
            ax.set_ylim(0, 1)
            ax.set_ylabel(f'{gene_id}\nProbability')
            ax.grid(True, alpha=0.3)
            
            # Add gene info
            n_splice_sites = len(splice_sites)
            strand = gene_data['strand'].iloc[0] if 'strand' in gene_data.columns else '+'
            chrom = gene_data['chrom'].iloc[0] if 'chrom' in gene_data.columns else '?'
            
            ax.text(0.02, 0.95, f'Chr{chrom} ({strand}) | {n_splice_sites} sites | {gene_length/1000:.1f}kb', 
                   transform=ax.transAxes, fontsize=9, 
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # X-axis formatting
            n_ticks = 6
            tick_positions = np.linspace(0, gene_length, n_ticks)
            ax.set_xticks(tick_positions)
            if i == len(genes_to_plot) - 1:  # Only show x-labels on bottom plot
                ax.set_xticklabels([f'{int(pos/1000)}k' for pos in tick_positions])
                ax.set_xlabel('Position in Gene (bp)')
            else:
                ax.set_xticklabels([])
        
        # Add legend to top plot
        if len(genes_to_plot) > 0:
            axes[0].legend(loc='upper right')
        
        plt.suptitle(f'Multi-Gene Splice Site Overview (Top {len(genes_to_plot)} Genes)', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save plot
        plot_file = output_path / "multi_gene_overview.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.log(f"Multi-gene overview plot saved: {plot_file}")
        return str(plot_file)
    
    def generate_gene_report(self, df: pd.DataFrame, gene_id: str, output_dir: Path,
                           meta_predictions: pd.DataFrame = None) -> Dict[str, str]:
        """Generate comprehensive visualization report for a single gene."""
        self.log(f"Generating comprehensive report for gene {gene_id}...")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if gene exists first
        if gene_id not in df['gene_id'].values:
            available_genes = df['gene_id'].value_counts().head(10)
            self.log(f"Gene {gene_id} not found in dataset. Available genes: {available_genes.index.tolist()}", "ERROR")
            raise ValueError(f"Gene {gene_id} not found in dataset. Try one of: {available_genes.index.tolist()[:5]}")
        
        # Calculate coordinates
        df_coord = self.calculate_absolute_coordinates(df)
        
        # Get gene-specific data
        gene_data = self.get_gene_data(df_coord, gene_id)
        
        # Generate all visualizations
        plot_files = {}
        
        try:
            plot_files['gene_track'] = self.create_gene_track_plot(gene_data, gene_id, output_dir)
        except Exception as e:
            self.log(f"Error creating gene track plot: {e}", "WARNING")
        
        try:
            plot_files['splice_density'] = self.create_splice_density_plot(gene_data, gene_id, output_dir)
        except Exception as e:
            self.log(f"Error creating splice density plot: {e}", "WARNING")
        
        try:
            plot_files['prediction_comparison'] = self.create_prediction_comparison_plot(
                gene_data, gene_id, output_dir, meta_predictions)
        except Exception as e:
            self.log(f"Error creating prediction comparison plot: {e}", "WARNING")
        
        # Generate summary statistics
        summary = self.get_gene_summary_stats(gene_data, gene_id)
        
        # Save summary as JSON
        import json
        summary_file = output_dir / f"gene_summary_{gene_id}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        plot_files['summary'] = str(summary_file)
        
        self.log(f"Gene report generated: {len(plot_files)} files created")
        return plot_files
    
    def get_gene_summary_stats(self, gene_data: pd.DataFrame, gene_id: str) -> Dict[str, Any]:
        """Generate summary statistics for a gene."""
        
        # Basic gene info
        gene_start = gene_data['gene_start'].iloc[0]
        gene_end = gene_data['gene_end'].iloc[0]
        gene_length = gene_end - gene_start
        strand = gene_data['strand'].iloc[0] if 'strand' in gene_data.columns else '+'
        chrom = gene_data['chrom'].iloc[0] if 'chrom' in gene_data.columns else 'Unknown'
        
        # Splice site counts
        splice_counts = gene_data['splice_type'].value_counts().to_dict()
        total_positions = len(gene_data)
        
        # Position statistics
        positions = gene_data['position'].values
        
        summary = {
            'gene_id': gene_id,
            'chromosome': chrom,
            'strand': strand,
            'gene_start': int(gene_start),
            'gene_end': int(gene_end),
            'gene_length': int(gene_length),
            'total_positions_analyzed': total_positions,
            'positions_per_kb': total_positions / (gene_length / 1000),
            'splice_site_counts': splice_counts,
            'position_stats': {
                'min_position': int(positions.min()),
                'max_position': int(positions.max()),
                'position_range': int(positions.max() - positions.min()),
                'position_coverage': (positions.max() - positions.min()) / gene_length
            }
        }
        
        # Probability score statistics if available
        prob_cols = ['donor_score', 'acceptor_score', 'neither_score']
        if all(col in gene_data.columns for col in prob_cols):
            summary['probability_stats'] = {}
            for col in prob_cols:
                summary['probability_stats'][col] = {
                    'mean': float(gene_data[col].mean()),
                    'std': float(gene_data[col].std()),
                    'max': float(gene_data[col].max()),
                    'min': float(gene_data[col].min())
                }
        
        return summary


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Genomic browser-quality visualization for splice sites",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --dataset train_pc_1000/master --gene-id ENSG00000205592 --output-dir results/genomic_viz
  %(prog)s --dataset train_pc_1000/master --top-genes 5 --output-dir results/multi_gene_viz
  %(prog)s --dataset train_pc_1000/master --gene-id ENSG00000205592 --meta-predictions results/meta_preds.csv
        """
    )
    
    parser.add_argument(
        '--dataset', 
        required=True,
        help='Path to dataset directory or file'
    )
    
    parser.add_argument(
        '--gene-id',
        help='Specific gene ID to visualize (e.g., ENSG00000205592)'
    )
    
    parser.add_argument(
        '--top-genes',
        type=int,
        help='Number of top genes (by splice site count) to visualize'
    )
    
    parser.add_argument(
        '--output-dir',
        required=True,
        help='Output directory for visualizations'
    )
    
    parser.add_argument(
        '--meta-predictions',
        help='Path to meta-model predictions file (CSV/TSV)'
    )
    
    parser.add_argument(
        '--gene-features',
        default='data/ensembl/spliceai_analysis/gene_features.tsv',
        help='Path to gene features file with boundaries'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Create visualizer
    visualizer = GenomicSpliceVisualizer(verbose=args.verbose)
    
    try:
        # Load dataset
        df = visualizer.load_dataset(args.dataset)
        
        # For memory efficiency, sample if dataset is very large
        if len(df) > 100000:
            visualizer.log(f"Large dataset detected ({len(df):,} rows). Sampling 100K rows for visualization.")
            df = df.sample(n=100000, random_state=42)
        
        # Load gene features if available
        gene_features = visualizer.load_gene_features(args.gene_features)
        
        # Load meta-predictions if provided
        meta_predictions = None
        if args.meta_predictions and os.path.exists(args.meta_predictions):
            visualizer.log(f"Loading meta-predictions from: {args.meta_predictions}")
            meta_predictions = pd.read_csv(args.meta_predictions)
        
        output_dir = Path(args.output_dir)
        
        if args.gene_id:
            # Single gene analysis
            visualizer.log(f"Generating detailed analysis for gene: {args.gene_id}")
            gene_output_dir = output_dir / args.gene_id
            plot_files = visualizer.generate_gene_report(df, args.gene_id, gene_output_dir, meta_predictions)
            
            visualizer.log(f"Gene analysis complete. Files saved to: {gene_output_dir}")
            visualizer.log(f"Generated files: {list(plot_files.keys())}")
            
        elif args.top_genes:
            # Multi-gene analysis
            visualizer.log(f"Generating analysis for top {args.top_genes} genes")
            
            # Find top genes by splice site count
            df_coord = visualizer.calculate_absolute_coordinates(df)
            splice_sites = df_coord[df_coord['splice_type'].isin(['donor', 'acceptor'])]
            top_genes_counts = splice_sites.groupby('gene_id').size().sort_values(ascending=False)
            top_gene_list = top_genes_counts.head(args.top_genes).index.tolist()
            
            visualizer.log(f"Top genes by splice site count: {dict(zip(top_gene_list, top_genes_counts.head(args.top_genes)))}")
            
            # Create multi-gene overview
            output_dir.mkdir(parents=True, exist_ok=True)
            overview_file = visualizer.create_multi_gene_overview(df_coord, top_gene_list, output_dir)
            
            # Generate individual reports for each top gene
            for gene_id in top_gene_list:
                gene_output_dir = output_dir / gene_id
                try:
                    visualizer.generate_gene_report(df_coord, gene_id, gene_output_dir, meta_predictions)
                except Exception as e:
                    visualizer.log(f"Error generating report for gene {gene_id}: {e}", "WARNING")
            
            visualizer.log(f"Multi-gene analysis complete. Files saved to: {output_dir}")
            
        else:
            raise ValueError("Must specify either --gene-id or --top-genes")
        
    except Exception as e:
        visualizer.log(f"Error during visualization: {e}", "ERROR")
        raise


if __name__ == "__main__":
    main() 