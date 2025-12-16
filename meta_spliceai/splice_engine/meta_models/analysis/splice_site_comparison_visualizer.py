#!/usr/bin/env python3
"""Publication-quality splice site comparison visualizer.

This module creates genomic browser-style visualizations comparing predicted vs true splice sites,
specifically designed to show the benefits of meta-learning over base models (like SpliceAI).

Key features:
- Side-by-side gene comparisons
- Separate donor (GT) and acceptor (AG) site plots
- Predicted vs observed splice sites in paired panels
- Highlights rescued FNs and eliminated FPs
- Publication-ready output

Usage
-----
python -m meta_spliceai.splice_engine.meta_models.analysis.splice_site_comparison_visualizer \
    --dataset train_pc_1000/master \
    --genes ENSG00000131018,ENSG00000114270 \
    --meta-predictions results/meta_predictions.csv \
    --output-dir results/splice_comparison \
    --threshold 0.5

Example outputs:
- Donor site comparison plots
- Acceptor site comparison plots
- Rescued site annotations
- False positive elimination highlights
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
from matplotlib.patches import Rectangle
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

# Plot styling constants
DONOR_COLOR = '#E74C3C'  # Red for donors (GT)
ACCEPTOR_COLOR = '#3498DB'  # Blue for acceptors (AG)
TRUE_SITE_COLOR = '#2C3E50'  # Dark gray for true sites
RESCUED_COLOR = '#27AE60'  # Green for rescued sites
ELIMINATED_COLOR = '#F39C12'  # Orange for eliminated FPs
BACKGROUND_COLOR = '#F8F9FA'  # Light gray background


class SpliceSiteComparisonVisualizer:
    """Publication-quality splice site comparison visualizer."""
    
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
    
    def load_dataset(self, dataset_path: str, filter_kmers: bool = True, 
                     target_genes: List[str] = None, sample_genes: int = None) -> pd.DataFrame:
        """Load dataset with hierarchical gene-level sampling for visualization."""
        self.log(f"Loading dataset from: {dataset_path}")
        
        try:
            # For visualization, we need COMPLETE gene data, not random row sampling
            # Use hierarchical sampling instead of the broken row-level sampling
            if sample_genes is not None:
                self.log(f"Using hierarchical sampling: {sample_genes} genes (preserves all splice sites)")
                from meta_spliceai.splice_engine.meta_models.training.label_utils import load_dataset_sample
                df = load_dataset_sample(dataset_path, sample_genes=sample_genes, random_seed=42)
            else:
                # For small datasets or when we need specific genes, load full dataset
                # Override the broken row-level sampling by setting very high limit
                import os
                original_row_cap = os.environ.get('SS_MAX_ROWS', None)
                
                if target_genes:
                    # When targeting specific genes, we need the full dataset to find them
                    self.log("Loading full dataset to find target genes...")
                    # Set a very high row cap to avoid sampling that might miss target genes
                    os.environ['SS_MAX_ROWS'] = '2000000'  # Set very high limit instead of 0
                else:
                    # For visualization, use gene-level sampling with reasonable size
                    self.log("Loading dataset with gene-level sampling (100 genes)...")
                    from meta_spliceai.splice_engine.meta_models.training.label_utils import load_dataset_sample
                    df = load_dataset_sample(dataset_path, sample_genes=100, random_seed=42)
                    
                    # Convert to pandas if needed
                    if hasattr(df, 'to_pandas'):
                        df = df.to_pandas()
                        
                    # Restore original setting
                    if original_row_cap is not None:
                        os.environ['SS_MAX_ROWS'] = original_row_cap
                    else:
                        if 'SS_MAX_ROWS' in os.environ:
                            del os.environ['SS_MAX_ROWS']
                    
                    # Skip the rest of the function for hierarchical sampling
                    if filter_kmers:
                        original_cols = len(df.columns)
                        kmer_patterns = ['6mer_', 'kmer_', '_mer_', 'mer_']
                        kmer_cols = [col for col in df.columns 
                                   if any(pattern in col.lower() for pattern in kmer_patterns)]
                        
                        if kmer_cols:
                            self.log(f"Filtering out {len(kmer_cols)} k-mer features for memory efficiency...")
                            df = df.drop(columns=kmer_cols)
                            self.log(f"Filtered from {original_cols} to {len(df.columns)} columns")
                    
                    self.log(f"Successfully loaded {len(df):,} samples with {len(df.columns)} columns via hierarchical sampling")
                    return df
                
                # Full dataset loading (only when target_genes specified)
                df = datasets.load_dataset(dataset_path)
                
                # Restore original setting
                if original_row_cap is not None:
                    os.environ['SS_MAX_ROWS'] = original_row_cap
                else:
                    if 'SS_MAX_ROWS' in os.environ:
                        del os.environ['SS_MAX_ROWS']
            
            # Convert to pandas if it's a polars DataFrame
            if hasattr(df, 'to_pandas'):
                self.log("Converting Polars DataFrame to Pandas...")
                df = df.to_pandas()
            
            # Verify target genes are present
            if target_genes:
                available_genes = set(df['gene_id'].unique())
                for gene in target_genes:
                    if gene not in available_genes:
                        self.log(f"WARNING: Target gene {gene} not found in dataset", "WARNING")
                    else:
                        self.log(f"✓ Target gene {gene} found in dataset")
            
            # Filter out k-mer features to reduce memory usage
            if filter_kmers:
                original_cols = len(df.columns)
                kmer_patterns = ['6mer_', 'kmer_', '_mer_', 'mer_']
                kmer_cols = [col for col in df.columns 
                           if any(pattern in col.lower() for pattern in kmer_patterns)]
                
                if kmer_cols:
                    self.log(f"Filtering out {len(kmer_cols)} k-mer features for memory efficiency...")
                    df = df.drop(columns=kmer_cols)
                    self.log(f"Filtered from {original_cols} to {len(df.columns)} columns")
            
            self.log(f"Successfully loaded {len(df):,} samples with {len(df.columns)} columns")
            return df
            
        except Exception as e:
            self.log(f"Error loading dataset: {e}", "ERROR")
            raise
    
    def load_meta_predictions(self, meta_predictions_path: str) -> pd.DataFrame:
        """Load meta-model predictions."""
        try:
            if os.path.exists(meta_predictions_path):
                self.log(f"Loading meta-model predictions from: {meta_predictions_path}")
                meta_df = pd.read_csv(meta_predictions_path, sep='\t')
                self.log(f"Loaded {len(meta_df):,} meta-model predictions")
                return meta_df
            else:
                self.log(f"Meta-predictions file not found: {meta_predictions_path}", "WARNING")
                return None
        except Exception as e:
            self.log(f"Error loading meta-predictions: {e}", "WARNING")
            return None
    
    def load_cv_results(self, cv_results_path: str) -> pd.DataFrame:
        """Load cross-validation results with both base and meta predictions."""
        try:
            if os.path.exists(cv_results_path):
                self.log(f"Loading CV results from: {cv_results_path}")
                cv_df = pd.read_csv(cv_results_path, sep='\t')
                self.log(f"Loaded {len(cv_df):,} CV results with base and meta predictions")
                return cv_df
            else:
                self.log(f"CV results file not found: {cv_results_path}", "WARNING")
                return None
        except Exception as e:
            self.log(f"Error loading CV results: {e}", "WARNING")
            return None
    
    def format_cv_results_as_meta_data(self, cv_df: pd.DataFrame) -> pd.DataFrame:
        """Format CV results to match expected meta prediction format."""
        if cv_df is None:
            return None
        
        # Create meta data format matching the base data structure
        meta_data = cv_df.copy()
        
        # Rename columns to match expected format
        column_mapping = {
            'meta_donor_prob': 'donor_score',
            'meta_acceptor_prob': 'acceptor_score', 
            'meta_neither_prob': 'neither_score',
            'true_label': 'splice_type'
        }
        
        # Apply column mapping
        for old_col, new_col in column_mapping.items():
            if old_col in meta_data.columns:
                meta_data[new_col] = meta_data[old_col]
        
        # Convert numeric true_label to splice_type strings if needed
        if 'true_label' in meta_data.columns:
            label_map = {0: 'neither', 1: 'donor', 2: 'acceptor'}
            if meta_data['true_label'].dtype in ['int64', 'int32']:
                meta_data['splice_type'] = meta_data['true_label'].map(label_map)
        
        # Ensure required columns exist
        required_cols = ['gene_id', 'position', 'splice_type', 'donor_score', 'acceptor_score', 'neither_score']
        missing_cols = [col for col in required_cols if col not in meta_data.columns]
        if missing_cols:
            self.log(f"Warning: Missing columns in CV results: {missing_cols}", "WARNING")
        
        # Add gene boundaries if available
        if self.gene_features is not None:
            gene_info = self.gene_features[['gene_id', 'start', 'end', 'strand']].rename(
                columns={'start': 'gene_start', 'end': 'gene_end'})
            meta_data = meta_data.merge(gene_info, on='gene_id', how='left')
        else:
            # If no gene features, try to get boundaries from the original base data
            self.log("No gene features available, trying to get boundaries from base data", "WARNING")
        
        self.log(f"Formatted {len(meta_data)} meta predictions")
        return meta_data
    
    def load_gene_features(self, gene_features_path: str = "data/ensembl/spliceai_analysis/gene_features.tsv") -> pd.DataFrame:
        """Load gene features for coordinate mapping."""
        try:
            if os.path.exists(gene_features_path):
                self.log(f"Loading gene features from: {gene_features_path}")
                gene_features = pd.read_csv(gene_features_path, sep='\t')
                self.gene_features = gene_features
                self.log(f"Loaded {len(gene_features)} gene features")
                return gene_features
            else:
                self.log(f"Gene features file not found: {gene_features_path}", "WARNING")
                return None
        except Exception as e:
            self.log(f"Error loading gene features: {e}", "WARNING")
            return None
    
    def get_gene_data(self, df: pd.DataFrame, gene_id: str) -> pd.DataFrame:
        """Extract and prepare data for a specific gene."""
        gene_data = df[df['gene_id'] == gene_id].copy()
        if len(gene_data) == 0:
            raise ValueError(f"Gene {gene_id} not found in dataset")
        
        # Sort by position for proper visualization
        gene_data = gene_data.sort_values('position')
        
        # Add strand information if not present
        if 'strand' not in gene_data.columns and self.gene_features is not None:
            gene_strand = self.gene_features[self.gene_features['gene_id'] == gene_id]['strand'].iloc[0]
            gene_data['strand'] = gene_strand
        
        self.log(f"Extracted {len(gene_data)} positions for gene {gene_id}")
        return gene_data
    
    def get_gene_display_name(self, gene_id: str) -> str:
        """Get display name for gene (with gene name if available)."""
        if self.gene_features is not None:
            gene_info = self.gene_features[self.gene_features['gene_id'] == gene_id]
            if len(gene_info) > 0 and 'gene_name' in gene_info.columns:
                gene_name = gene_info['gene_name'].iloc[0]
                if pd.notna(gene_name) and gene_name != gene_id:
                    return f"{gene_name}"
        return gene_id
    
    def identify_splice_site_changes(self, base_data: pd.DataFrame, meta_data: pd.DataFrame, 
                                   threshold: float = 0.5) -> Dict[str, List[int]]:
        """Identify rescued FNs and eliminated FPs between base and meta models."""
        changes = {
            'rescued_donors': [],
            'rescued_acceptors': [],
            'eliminated_fp_donors': [],
            'eliminated_fp_acceptors': []
        }
        
        # Convert true labels to binary indicators
        true_labels = _encode_labels(base_data['splice_type'])
        true_donors = set(base_data[true_labels == 1]['position'])
        true_acceptors = set(base_data[true_labels == 2]['position'])
        
        # Base model predictions
        base_donor_preds = set(base_data[base_data['donor_score'] > threshold]['position'])
        base_acceptor_preds = set(base_data[base_data['acceptor_score'] > threshold]['position'])
        
        # Meta model predictions (if available)
        if meta_data is not None and len(meta_data) > 0:
            meta_donor_preds = set(meta_data[meta_data['donor_score'] > threshold]['position'])
            meta_acceptor_preds = set(meta_data[meta_data['acceptor_score'] > threshold]['position'])
            
            # Rescued sites (FN -> TP)
            changes['rescued_donors'] = list((true_donors - base_donor_preds) & meta_donor_preds)
            changes['rescued_acceptors'] = list((true_acceptors - base_acceptor_preds) & meta_acceptor_preds)
            
            # Eliminated FPs (FP -> TN)
            changes['eliminated_fp_donors'] = list((base_donor_preds - true_donors) - meta_donor_preds)
            changes['eliminated_fp_acceptors'] = list((base_acceptor_preds - true_acceptors) - meta_acceptor_preds)
        
        return changes
    
    def create_splice_site_panel(self, ax: plt.Axes, gene_data: pd.DataFrame, 
                               site_type: str, model_type: str, color: str,
                               highlight_positions: List[int] = None) -> None:
        """Create a single splice site panel (predicted or true sites)."""
        
        positions = gene_data['position'].values
        gene_length = gene_data['gene_end'].iloc[0] - gene_data['gene_start'].iloc[0]
        
        if model_type == 'predicted':
            # Show predicted sites with probability scores as line heights
            score_col = f"{site_type}_score"
            if score_col in gene_data.columns:
                scores = gene_data[score_col].values
                
                # Create vertical lines for all positions
                for pos, score in zip(positions, scores):
                    if score > 0.01:  # Only show significant scores
                        ax.vlines(pos, 0, score, colors=color, alpha=0.7, linewidth=1.5)
                
                # Highlight special positions if provided
                if highlight_positions:
                    for pos in highlight_positions:
                        if pos in positions:
                            idx = np.where(positions == pos)[0][0]
                            score = scores[idx]
                            ax.vlines(pos, 0, score, colors=RESCUED_COLOR, alpha=0.9, linewidth=3)
                            ax.scatter(pos, score, color=RESCUED_COLOR, s=50, zorder=5, marker='o')
            
            ax.set_ylim(0, 1)
            ax.set_ylabel('Probability', fontsize=12, fontweight='bold')
            ax.tick_params(axis='y', labelsize=10)
            
        else:  # true sites
            # Show true sites as binary indicators with better visual design
            true_labels = _encode_labels(gene_data['splice_type'])
            site_label = 1 if site_type == 'donor' else 2
            true_positions = positions[true_labels == site_label]
            
            # Create prominent vertical lines for true sites
            for pos in true_positions:
                ax.vlines(pos, 0, 1, colors=TRUE_SITE_COLOR, alpha=0.9, linewidth=3)
            
            # Fill background to clearly distinguish from prediction panel
            ax.axhspan(0, 1, alpha=0.1, color=TRUE_SITE_COLOR)
            
            ax.set_ylim(0, 1)
            ax.set_ylabel('True Sites', fontsize=12, fontweight='bold')
            ax.set_yticks([0.5])
            ax.set_yticklabels(['●'], fontsize=14)
        
        # Common formatting
        ax.set_xlim(0, gene_length)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_facecolor(BACKGROUND_COLOR)
        
        # Enhanced border
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
        
        # X-axis formatting - cleaner approach
        n_ticks = 6
        tick_positions = np.linspace(0, gene_length, n_ticks)
        ax.set_xticks(tick_positions)
        if model_type == 'true':  # Only show x-labels on bottom panels
            ax.set_xticklabels([f'{int(pos/1000)}k' for pos in tick_positions], fontsize=10)
            ax.set_xlabel('Position (kb)', fontsize=11, fontweight='bold')
        else:
            ax.set_xticklabels([])
    
    def validate_gene_data_consistency(self, gene_data_base: pd.DataFrame, 
                                      gene_data_meta: pd.DataFrame, gene_id: str) -> bool:
        """Validate that base and meta data are consistent for the same gene."""
        
        if gene_data_meta is None or len(gene_data_meta) == 0:
            self.log(f"Warning: No meta data available for gene {gene_id}", "WARNING")
            return False
        
        # Check if we have the same positions
        base_positions = set(gene_data_base['position'])
        meta_positions = set(gene_data_meta['position'])
        
        self.log(f"Base data positions: {len(base_positions)}")
        self.log(f"Meta data positions: {len(meta_positions)}")
        
        # Find intersection and differences
        common_positions = base_positions & meta_positions
        base_only = base_positions - meta_positions
        meta_only = meta_positions - base_positions
        
        self.log(f"Common positions: {len(common_positions)}")
        if base_only:
            self.log(f"Positions only in base: {len(base_only)}", "WARNING")
        if meta_only:
            self.log(f"Positions only in meta: {len(meta_only)}", "WARNING")
        
        # Check true splice sites consistency for common positions
        if len(common_positions) > 0:
            # Get data for common positions only
            base_common = gene_data_base[gene_data_base['position'].isin(common_positions)]
            meta_common = gene_data_meta[gene_data_meta['position'].isin(common_positions)]
            
            # Sort by position for comparison
            base_common = base_common.sort_values('position')
            meta_common = meta_common.sort_values('position')
            
            # Compare true labels
            base_true = base_common['splice_type'].values
            meta_true = meta_common['splice_type'].values
            
            if not np.array_equal(base_true, meta_true):
                self.log("ERROR: True splice sites don't match between base and meta data!", "ERROR")
                # Show differences
                mismatches = sum(base_true != meta_true)
                self.log(f"Found {mismatches} mismatched true labels out of {len(common_positions)}", "ERROR")
                return False
            else:
                self.log("✓ True splice sites are consistent between base and meta data")
        
        return len(common_positions) > 0
    
    def align_gene_datasets(self, gene_data_base: pd.DataFrame, 
                           gene_data_meta: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Align base and meta datasets to common positions for fair comparison."""
        
        if gene_data_meta is None or len(gene_data_meta) == 0:
            return gene_data_base, None
        
        # Find common positions
        base_positions = set(gene_data_base['position'])
        meta_positions = set(gene_data_meta['position'])
        common_positions = base_positions & meta_positions
        
        if len(common_positions) == 0:
            self.log("WARNING: No common positions between base and meta data", "WARNING")
            return gene_data_base, None
        
        # Filter both datasets to common positions
        base_aligned = gene_data_base[gene_data_base['position'].isin(common_positions)].copy()
        meta_aligned = gene_data_meta[gene_data_meta['position'].isin(common_positions)].copy()
        
        # Sort by position
        base_aligned = base_aligned.sort_values('position')
        meta_aligned = meta_aligned.sort_values('position')
        
        self.log(f"Aligned datasets to {len(common_positions)} common positions")
        
        return base_aligned, meta_aligned
    
    def create_gene_comparison_plot(self, gene_data_base: pd.DataFrame, 
                                  gene_data_meta: pd.DataFrame,
                                  gene_id: str, output_path: Path, 
                                  threshold: float = 0.5) -> Dict[str, str]:
        """Create comprehensive comparison plot for a single gene."""
        
        gene_name = self.get_gene_display_name(gene_id)
        self.log(f"Creating comparison plot for gene {gene_name} ({gene_id})")
        
        # Validate data consistency and align datasets
        if gene_data_meta is not None:
            is_valid = self.validate_gene_data_consistency(gene_data_base, gene_data_meta, gene_id)
            if is_valid:
                # Align datasets for fair comparison
                gene_data_base_aligned, gene_data_meta_aligned = self.align_gene_datasets(gene_data_base, gene_data_meta)
                gene_data_base = gene_data_base_aligned
                gene_data_meta = gene_data_meta_aligned
        
        # Create figure with subplots - use GridSpec for better control
        fig = plt.figure(figsize=(16, 12))
        
        # Create custom subplot layout with different height ratios
        # Prediction panels (3 units high), True site panels (1 unit high)
        from matplotlib.gridspec import GridSpec
        gs = GridSpec(10, 2, height_ratios=[3, 1, 0.8, 3, 1, 0.8, 0.2, 0.2, 0.2, 0.2], 
                     hspace=0.4, wspace=0.25, top=0.92, bottom=0.08)
        
        # === 5' SPLICE SITES (DONORS - GT) ===
        # Add title above the donor plots with proper positioning (higher up to avoid overlap)
        fig.text(0.5, 0.98, "5' Splice sites \"GT\"", fontsize=16, fontweight='bold', 
                color=DONOR_COLOR, ha='center', transform=fig.transFigure)
        
        # Base model - donors (larger height for predictions)
        ax1 = fig.add_subplot(gs[0, 0])  # Base predicted (tall)
        ax2 = fig.add_subplot(gs[1, 0])  # Base true (short)
        
        # Meta model - donors
        ax3 = fig.add_subplot(gs[0, 1])  # Meta predicted (tall)
        ax4 = fig.add_subplot(gs[1, 1])  # Meta true (short)
        
        # Log splice site counts for verification
        if gene_data_base is not None:
            base_donors = (gene_data_base['splice_type'] == 'donor').sum()
            base_acceptors = (gene_data_base['splice_type'] == 'acceptor').sum()
            self.log(f"Base data - Donors: {base_donors}, Acceptors: {base_acceptors}")
        
        if gene_data_meta is not None:
            meta_donors = (gene_data_meta['splice_type'] == 'donor').sum()
            meta_acceptors = (gene_data_meta['splice_type'] == 'acceptor').sum()
            self.log(f"Meta data - Donors: {meta_donors}, Acceptors: {meta_acceptors}")
        
        # Get changes for highlighting
        changes = self.identify_splice_site_changes(gene_data_base, gene_data_meta, threshold)
        
        # Plot base model donors
        self.create_splice_site_panel(ax1, gene_data_base, 'donor', 'predicted', DONOR_COLOR)
        ax1.set_title(f'{gene_name} - Base Model', fontsize=12, fontweight='bold')
        
        self.create_splice_site_panel(ax2, gene_data_base, 'donor', 'true', DONOR_COLOR)
        ax2.set_title(f'{gene_name} - Observed', fontsize=12)
        
        # Plot meta model donors (if available, otherwise show base model for comparison)
        if gene_data_meta is not None and len(gene_data_meta) > 0:
            self.create_splice_site_panel(ax3, gene_data_meta, 'donor', 'predicted', DONOR_COLOR,
                                        highlight_positions=changes['rescued_donors'])
            ax3.set_title(f'{gene_name} - Meta Model', fontsize=12, fontweight='bold')
            
            self.create_splice_site_panel(ax4, gene_data_meta, 'donor', 'true', DONOR_COLOR)
            ax4.set_title(f'{gene_name} - Observed', fontsize=12)
        else:
            # Show base model data for comparison when meta model not available
            self.create_splice_site_panel(ax3, gene_data_base, 'donor', 'predicted', DONOR_COLOR)
            ax3.set_title(f'{gene_name} - Meta Model (Base Data)', fontsize=12, fontweight='bold', color='gray')
            
            self.create_splice_site_panel(ax4, gene_data_base, 'donor', 'true', DONOR_COLOR)
            ax4.set_title(f'{gene_name} - Observed', fontsize=12)
        
        # === 3' SPLICE SITES (ACCEPTORS - AG) ===
        # Add title above the acceptor plots with proper positioning (adjusted to avoid overlap)
        fig.text(0.5, 0.52, "3' Splice sites \"AG\"", fontsize=16, fontweight='bold', 
                color=ACCEPTOR_COLOR, ha='center', transform=fig.transFigure)
        
        # Base model - acceptors (larger height for predictions)
        ax5 = fig.add_subplot(gs[3, 0])   # Base predicted (tall)
        ax6 = fig.add_subplot(gs[4, 0])   # Base true (short)
        
        # Meta model - acceptors
        ax7 = fig.add_subplot(gs[3, 1])   # Meta predicted (tall)
        ax8 = fig.add_subplot(gs[4, 1])   # Meta true (short)
        
        # Plot base model acceptors
        self.create_splice_site_panel(ax5, gene_data_base, 'acceptor', 'predicted', ACCEPTOR_COLOR)
        ax5.set_title(f'{gene_name} - Base Model', fontsize=12, fontweight='bold')
        
        self.create_splice_site_panel(ax6, gene_data_base, 'acceptor', 'true', ACCEPTOR_COLOR)
        ax6.set_title(f'{gene_name} - Observed', fontsize=12)
        ax6.set_xlabel('Position', fontsize=10)
        
        # Plot meta model acceptors (if available, otherwise show base model for comparison)
        if gene_data_meta is not None and len(gene_data_meta) > 0:
            self.create_splice_site_panel(ax7, gene_data_meta, 'acceptor', 'predicted', ACCEPTOR_COLOR,
                                        highlight_positions=changes['rescued_acceptors'])
            ax7.set_title(f'{gene_name} - Meta Model', fontsize=12, fontweight='bold')
            
            self.create_splice_site_panel(ax8, gene_data_meta, 'acceptor', 'true', ACCEPTOR_COLOR)
            ax8.set_title(f'{gene_name} - Observed', fontsize=12)
        else:
            # Show base model data for comparison when meta model not available
            self.create_splice_site_panel(ax7, gene_data_base, 'acceptor', 'predicted', ACCEPTOR_COLOR)
            ax7.set_title(f'{gene_name} - Meta Model (Base Data)', fontsize=12, fontweight='bold', color='gray')
            
            self.create_splice_site_panel(ax8, gene_data_base, 'acceptor', 'true', ACCEPTOR_COLOR)
            ax8.set_title(f'{gene_name} - Observed', fontsize=12)
        
        # Add summary statistics
        n_rescued_donors = len(changes['rescued_donors'])
        n_rescued_acceptors = len(changes['rescued_acceptors'])
        n_eliminated_fp_donors = len(changes['eliminated_fp_donors'])
        n_eliminated_fp_acceptors = len(changes['eliminated_fp_acceptors'])
        
        if gene_data_meta is not None and len(gene_data_meta) > 0:
            summary_text = f"""Meta-learning improvements:
• Rescued donors: {n_rescued_donors}
• Rescued acceptors: {n_rescued_acceptors}  
• Eliminated FP donors: {n_eliminated_fp_donors}
• Eliminated FP acceptors: {n_eliminated_fp_acceptors}"""
        else:
            # Count true splice sites for base model comparison
            true_labels = _encode_labels(gene_data_base['splice_type'])
            n_true_donors = (true_labels == 1).sum()
            n_true_acceptors = (true_labels == 2).sum()
            summary_text = f"""Base model only:
• True donors: {n_true_donors}
• True acceptors: {n_true_acceptors}
• Total positions: {len(gene_data_base)}
• Meta model: Not available"""
        
        fig.text(0.98, 0.03, summary_text, fontsize=10, ha='right', va='bottom',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8),
                transform=fig.transFigure)
        
        # Save plot
        plot_file = output_path / f"splice_comparison_{gene_id}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.log(f"Comparison plot saved: {plot_file}")
        return {
            'plot_file': str(plot_file),
            'changes': changes
        }
    
    def create_multi_gene_comparison(self, base_data: pd.DataFrame, meta_data: pd.DataFrame,
                                   gene_ids: List[str], output_path: Path, 
                                   threshold: float = 0.5) -> Dict[str, Any]:
        """Create side-by-side comparison plots for multiple genes."""
        
        self.log(f"Creating multi-gene comparison for {len(gene_ids)} genes")
        
        # Create large figure for side-by-side gene comparison
        fig = plt.figure(figsize=(8 * len(gene_ids), 12))
        
        all_changes = {}
        
        for i, gene_id in enumerate(gene_ids):
            gene_name = self.get_gene_display_name(gene_id)
            
            # Get gene data
            gene_data_base = self.get_gene_data(base_data, gene_id)
            gene_data_meta = None
            if meta_data is not None:
                try:
                    gene_data_meta = self.get_gene_data(meta_data, gene_id)
                except ValueError:
                    self.log(f"Gene {gene_id} not found in meta data", "WARNING")
            
            # Calculate subplot positions
            col_offset = i * 2
            
            # === 5' SPLICE SITES (DONORS) ===
            if i == 0:  # Only add title for first gene
                fig.text(0.02, 0.85, "5' Splice sites \"GT\"", fontsize=16, fontweight='bold', 
                        color=DONOR_COLOR, transform=fig.transFigure)
            
            # Base model donors
            ax1 = plt.subplot(8, len(gene_ids) * 2, col_offset + 1)
            ax2 = plt.subplot(8, len(gene_ids) * 2, col_offset + 1 + len(gene_ids) * 2)
            
            # Meta model donors
            ax3 = plt.subplot(8, len(gene_ids) * 2, col_offset + 2)
            ax4 = plt.subplot(8, len(gene_ids) * 2, col_offset + 2 + len(gene_ids) * 2)
            
            # Get changes for highlighting
            changes = self.identify_splice_site_changes(gene_data_base, gene_data_meta, threshold)
            all_changes[gene_id] = changes
            
            # Plot donors
            self.create_splice_site_panel(ax1, gene_data_base, 'donor', 'predicted', DONOR_COLOR)
            ax1.set_title(f'{gene_name} - Base', fontsize=10, fontweight='bold')
            
            self.create_splice_site_panel(ax2, gene_data_base, 'donor', 'true', DONOR_COLOR)
            ax2.set_title(f'{gene_name} - Observed', fontsize=10)
            
            if gene_data_meta is not None:
                self.create_splice_site_panel(ax3, gene_data_meta, 'donor', 'predicted', DONOR_COLOR,
                                            highlight_positions=changes['rescued_donors'])
                ax3.set_title(f'{gene_name} - Meta', fontsize=10, fontweight='bold')
                
                self.create_splice_site_panel(ax4, gene_data_meta, 'donor', 'true', DONOR_COLOR)
                ax4.set_title(f'{gene_name} - Observed', fontsize=10)
            
            # === 3' SPLICE SITES (ACCEPTORS) ===
            if i == 0:  # Only add title for first gene
                fig.text(0.02, 0.45, "3' Splice sites \"AG\"", fontsize=16, fontweight='bold', 
                        color=ACCEPTOR_COLOR, transform=fig.transFigure)
            
            # Acceptor subplots
            ax5 = plt.subplot(8, len(gene_ids) * 2, col_offset + 1 + len(gene_ids) * 4)
            ax6 = plt.subplot(8, len(gene_ids) * 2, col_offset + 1 + len(gene_ids) * 6)
            ax7 = plt.subplot(8, len(gene_ids) * 2, col_offset + 2 + len(gene_ids) * 4)
            ax8 = plt.subplot(8, len(gene_ids) * 2, col_offset + 2 + len(gene_ids) * 6)
            
            # Plot acceptors
            self.create_splice_site_panel(ax5, gene_data_base, 'acceptor', 'predicted', ACCEPTOR_COLOR)
            ax5.set_title(f'{gene_name} - Base', fontsize=10, fontweight='bold')
            
            self.create_splice_site_panel(ax6, gene_data_base, 'acceptor', 'true', ACCEPTOR_COLOR)
            ax6.set_title(f'{gene_name} - Observed', fontsize=10)
            ax6.set_xlabel('Position', fontsize=10)
            
            if gene_data_meta is not None:
                self.create_splice_site_panel(ax7, gene_data_meta, 'acceptor', 'predicted', ACCEPTOR_COLOR,
                                            highlight_positions=changes['rescued_acceptors'])
                ax7.set_title(f'{gene_name} - Meta', fontsize=10, fontweight='bold')
                
                self.create_splice_site_panel(ax8, gene_data_meta, 'acceptor', 'true', ACCEPTOR_COLOR)
                ax8.set_title(f'{gene_name} - Observed', fontsize=10)
                ax8.set_xlabel('Position', fontsize=10)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.92, bottom=0.08, left=0.05, right=0.98)
        
        # Save plot
        plot_file = output_path / f"multi_gene_splice_comparison.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.log(f"Multi-gene comparison plot saved: {plot_file}")
        
        return {
            'plot_file': str(plot_file),
            'all_changes': all_changes
        }
    
    def generate_summary_report(self, changes_dict: Dict[str, Dict], 
                              output_path: Path) -> str:
        """Generate a summary report of meta-learning improvements."""
        
        report_lines = [
            "# Meta-Learning Splice Site Prediction Improvements",
            "=" * 60,
            ""
        ]
        
        total_rescued_donors = 0
        total_rescued_acceptors = 0
        total_eliminated_fp_donors = 0
        total_eliminated_fp_acceptors = 0
        
        for gene_id, changes in changes_dict.items():
            gene_name = self.get_gene_display_name(gene_id)
            
            rescued_donors = len(changes['rescued_donors'])
            rescued_acceptors = len(changes['rescued_acceptors'])
            eliminated_fp_donors = len(changes['eliminated_fp_donors'])
            eliminated_fp_acceptors = len(changes['eliminated_fp_acceptors'])
            
            total_rescued_donors += rescued_donors
            total_rescued_acceptors += rescued_acceptors
            total_eliminated_fp_donors += eliminated_fp_donors
            total_eliminated_fp_acceptors += eliminated_fp_acceptors
            
            report_lines.extend([
                f"## {gene_name} ({gene_id})",
                f"- Rescued donor sites: {rescued_donors}",
                f"- Rescued acceptor sites: {rescued_acceptors}",
                f"- Eliminated FP donors: {eliminated_fp_donors}",
                f"- Eliminated FP acceptors: {eliminated_fp_acceptors}",
                ""
            ])
        
        # Add summary
        report_lines.extend([
            "## Overall Summary",
            f"- Total rescued donor sites: {total_rescued_donors}",
            f"- Total rescued acceptor sites: {total_rescued_acceptors}",
            f"- Total eliminated FP donors: {total_eliminated_fp_donors}",
            f"- Total eliminated FP acceptors: {total_eliminated_fp_acceptors}",
            f"- Total improvements: {total_rescued_donors + total_rescued_acceptors + total_eliminated_fp_donors + total_eliminated_fp_acceptors}",
            ""
        ])
        
        # Save report
        report_file = output_path / "meta_learning_improvements_summary.txt"
        with open(report_file, 'w') as f:
            f.write('\n'.join(report_lines))
        
        self.log(f"Summary report saved: {report_file}")
        return str(report_file)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Publication-quality splice site comparison visualizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --dataset train_pc_1000/master --genes ENSG00000131018,ENSG00000114270 --output-dir results/splice_comparison
  %(prog)s --dataset train_pc_1000/master --genes ENSG00000131018 --meta-predictions results/meta_preds.csv --threshold 0.3
        """
    )
    
    parser.add_argument(
        '--dataset', 
        required=True,
        help='Path to dataset directory or file'
    )
    
    parser.add_argument(
        '--genes',
        required=True,
        help='Comma-separated list of gene IDs to analyze'
    )
    
    parser.add_argument(
        '--meta-predictions',
        help='Path to meta-model predictions file (CSV/TSV)'
    )
    
    parser.add_argument(
        '--cv-results',
        help='Path to CV results file with both base and meta predictions (e.g., position_level_classification_results.tsv)'
    )
    
    parser.add_argument(
        '--output-dir',
        required=True,
        help='Output directory for visualizations'
    )
    
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.5,
        help='Probability threshold for splice site prediction (default: 0.5)'
    )
    
    parser.add_argument(
        '--gene-features',
        default='data/ensembl/spliceai_analysis/gene_features.tsv',
        help='Path to gene features file'
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
    visualizer = SpliceSiteComparisonVisualizer(verbose=args.verbose)
    
    try:
        # Parse gene list
        gene_ids = [g.strip() for g in args.genes.split(',')]
        visualizer.log(f"Analyzing {len(gene_ids)} genes: {gene_ids}")
        
        # Load data with target gene verification
        base_data = visualizer.load_dataset(args.dataset, target_genes=gene_ids)
        
        # Load gene features first (needed for meta data formatting)
        visualizer.load_gene_features(args.gene_features)
        
        # Load meta predictions (prioritize CV results over separate meta predictions file)
        meta_data = None
        if args.cv_results:
            cv_results = visualizer.load_cv_results(args.cv_results)
            if cv_results is not None:
                meta_data = visualizer.format_cv_results_as_meta_data(cv_results)
        elif args.meta_predictions:
            meta_data = visualizer.load_meta_predictions(args.meta_predictions)
        
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate visualizations
        all_changes = {}
        
        if len(gene_ids) == 1:
            # Single gene detailed analysis
            gene_id = gene_ids[0]
            gene_data_base = visualizer.get_gene_data(base_data, gene_id)
            gene_data_meta = None
            if meta_data is not None:
                try:
                    gene_data_meta = visualizer.get_gene_data(meta_data, gene_id)
                except ValueError:
                    visualizer.log(f"Gene {gene_id} not found in meta data", "WARNING")
            
            result = visualizer.create_gene_comparison_plot(
                gene_data_base, gene_data_meta, gene_id, output_dir, args.threshold
            )
            all_changes[gene_id] = result['changes']
            
        else:
            # Multi-gene comparison
            result = visualizer.create_multi_gene_comparison(
                base_data, meta_data, gene_ids, output_dir, args.threshold
            )
            all_changes = result['all_changes']
        
        # Generate summary report
        visualizer.generate_summary_report(all_changes, output_dir)
        
        visualizer.log("Splice site comparison visualization complete!")
        
    except Exception as e:
        visualizer.log(f"Error during visualization: {e}", "ERROR")
        raise


if __name__ == "__main__":
    main() 