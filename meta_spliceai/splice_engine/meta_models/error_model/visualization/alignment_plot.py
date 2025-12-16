"""
Token-level Integrated Gradients alignment visualization.

This module provides visualization tools for displaying token-level
IG attributions aligned with DNA sequences for interpretability analysis.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap, Normalize
import seaborn as sns

# Add project root to path for imports
project_root = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(project_root))


class AlignmentPlotter:
    """
    Creates alignment plots showing token-level IG attributions.
    
    Features:
    1. Sequence alignment with color-coded attributions
    2. Heatmap visualization of token importance
    3. Comparative plots for error vs correct predictions
    4. Customizable color schemes and layouts
    """
    
    def __init__(
        self,
        figsize: Tuple[int, int] = (15, 8),
        colormap: str = 'RdBu_r',
        font_size: int = 10
    ):
        """
        Initialize alignment plotter.
        
        Parameters
        ----------
        figsize : Tuple[int, int], default (15, 8)
            Figure size for plots
        colormap : str, default 'RdBu_r'
            Colormap for attribution visualization
        font_size : int, default 10
            Font size for text elements
        """
        self.figsize = figsize
        self.colormap = colormap
        self.font_size = font_size
        self.logger = logging.getLogger(__name__)
        
        # Setup plotting style
        plt.style.use('default')
        sns.set_palette("husl")
    
    def plot_sequence_attribution(
        self,
        sequence: str,
        tokens: List[str],
        attributions: List[float],
        title: str = "Token Attribution Alignment",
        max_tokens_per_line: int = 50,
        show_values: bool = True,
        save_path: Optional[Path] = None
    ) -> plt.Figure:
        """
        Plot sequence with token-level attributions.
        
        Parameters
        ----------
        sequence : str
            Original DNA sequence
        tokens : List[str]
            Tokenized sequence
        attributions : List[float]
            Attribution values for each token
        title : str, default "Token Attribution Alignment"
            Plot title
        max_tokens_per_line : int, default 50
            Maximum tokens per line
        show_values : bool, default True
            Whether to show attribution values
        save_path : Path, optional
            Path to save the plot
            
        Returns
        -------
        plt.Figure
            The created figure
        """
        # Prepare data
        tokens_clean = [token.replace('▁', '').replace('[CLS]', '').replace('[SEP]', '') 
                       for token in tokens if token not in ['[PAD]', '']]
        attributions_clean = [attr for token, attr in zip(tokens, attributions) 
                             if token not in ['[PAD]', '']]
        
        # Normalize attributions for color mapping
        attr_array = np.array(attributions_clean)
        abs_max = max(abs(attr_array.min()), abs(attr_array.max()))
        norm = Normalize(vmin=-abs_max, vmax=abs_max)
        
        # Create colormap
        cmap = plt.cm.get_cmap(self.colormap)
        
        # Calculate number of lines needed
        n_lines = (len(tokens_clean) + max_tokens_per_line - 1) // max_tokens_per_line
        
        # Create figure
        fig, axes = plt.subplots(n_lines, 1, figsize=(self.figsize[0], self.figsize[1] * n_lines))
        if n_lines == 1:
            axes = [axes]
        
        fig.suptitle(title, fontsize=self.font_size + 2, fontweight='bold')
        
        # Plot each line
        for line_idx in range(n_lines):
            ax = axes[line_idx]
            
            start_idx = line_idx * max_tokens_per_line
            end_idx = min(start_idx + max_tokens_per_line, len(tokens_clean))
            
            line_tokens = tokens_clean[start_idx:end_idx]
            line_attrs = attributions_clean[start_idx:end_idx]
            
            # Create token positions
            positions = np.arange(len(line_tokens))
            
            # Plot tokens as colored rectangles
            for i, (token, attr) in enumerate(zip(line_tokens, line_attrs)):
                color = cmap(norm(attr))
                
                # Draw rectangle
                rect = patches.Rectangle(
                    (i - 0.4, -0.4), 0.8, 0.8,
                    linewidth=1, edgecolor='black', facecolor=color, alpha=0.8
                )
                ax.add_patch(rect)
                
                # Add token text
                ax.text(i, 0, token, ha='center', va='center', 
                       fontsize=self.font_size, fontweight='bold')
                
                # Add attribution value if requested
                if show_values:
                    ax.text(i, -0.7, f'{attr:.3f}', ha='center', va='center',
                           fontsize=self.font_size - 2, style='italic')
            
            # Customize axis
            ax.set_xlim(-0.5, len(line_tokens) - 0.5)
            ax.set_ylim(-1, 0.5)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_ylabel(f'Line {line_idx + 1}', fontsize=self.font_size)
            
            # Remove spines
            for spine in ax.spines.values():
                spine.set_visible(False)
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=axes, orientation='horizontal', 
                           fraction=0.05, pad=0.1, aspect=30)
        cbar.set_label('Attribution Score', fontsize=self.font_size)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Saved alignment plot to {save_path}")
        
        return fig
    
    def plot_attribution_heatmap(
        self,
        attributions_data: List[Dict[str, Any]],
        max_sequences: int = 20,
        max_tokens: int = 100,
        title: str = "Attribution Heatmap",
        save_path: Optional[Path] = None
    ) -> plt.Figure:
        """
        Create heatmap of attributions across multiple sequences.
        
        Parameters
        ----------
        attributions_data : List[Dict[str, Any]]
            Attribution data from IG analysis
        max_sequences : int, default 20
            Maximum number of sequences to display
        max_tokens : int, default 100
            Maximum number of tokens per sequence
        title : str, default "Attribution Heatmap"
            Plot title
        save_path : Path, optional
            Path to save the plot
            
        Returns
        -------
        plt.Figure
            The created figure
        """
        # Prepare data matrix
        n_seqs = min(len(attributions_data), max_sequences)
        attribution_matrix = []
        sequence_labels = []
        
        for i in range(n_seqs):
            data = attributions_data[i]
            attrs = data['attributions'][:max_tokens]
            
            # Pad or truncate to max_tokens
            if len(attrs) < max_tokens:
                attrs.extend([0.0] * (max_tokens - len(attrs)))
            else:
                attrs = attrs[:max_tokens]
            
            attribution_matrix.append(attrs)
            sequence_labels.append(f"Seq {i+1} (Label: {data['label']})")
        
        attribution_matrix = np.array(attribution_matrix)
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot heatmap
        im = ax.imshow(attribution_matrix, cmap=self.colormap, aspect='auto',
                      interpolation='nearest')
        
        # Customize axes
        ax.set_xlabel('Token Position', fontsize=self.font_size)
        ax.set_ylabel('Sequence', fontsize=self.font_size)
        ax.set_title(title, fontsize=self.font_size + 2, fontweight='bold')
        
        # Set y-axis labels
        ax.set_yticks(range(n_seqs))
        ax.set_yticklabels(sequence_labels, fontsize=self.font_size - 2)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Attribution Score', fontsize=self.font_size)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Saved heatmap to {save_path}")
        
        return fig
    
    def plot_comparative_attribution(
        self,
        error_attributions: List[Dict[str, Any]],
        correct_attributions: List[Dict[str, Any]],
        n_examples: int = 5,
        title: str = "Error vs Correct Attribution Comparison",
        save_path: Optional[Path] = None
    ) -> plt.Figure:
        """
        Create comparative plot of error vs correct attributions.
        
        Parameters
        ----------
        error_attributions : List[Dict[str, Any]]
            Attribution data for error predictions
        correct_attributions : List[Dict[str, Any]]
            Attribution data for correct predictions
        n_examples : int, default 5
            Number of examples to show for each class
        title : str, default "Error vs Correct Attribution Comparison"
            Plot title
        save_path : Path, optional
            Path to save the plot
            
        Returns
        -------
        plt.Figure
            The created figure
        """
        fig, axes = plt.subplots(2, n_examples, figsize=(n_examples * 3, 8))
        fig.suptitle(title, fontsize=self.font_size + 2, fontweight='bold')
        
        # Plot error examples
        for i in range(min(n_examples, len(error_attributions))):
            ax = axes[0, i]
            data = error_attributions[i]
            
            # Get top tokens for visualization
            top_tokens = data.get('top_tokens', [])[:20]  # Show top 20
            if top_tokens:
                tokens = [t['token'] for t in top_tokens]
                attrs = [t['attribution'] for t in top_tokens]
                positions = range(len(tokens))
                
                # Color bars by attribution
                colors = [plt.cm.RdBu_r(0.5 + 0.5 * attr / max(abs(min(attrs)), abs(max(attrs))))
                         for attr in attrs]
                
                bars = ax.bar(positions, attrs, color=colors, alpha=0.7)
                ax.set_xticks(positions)
                ax.set_xticklabels(tokens, rotation=45, ha='right', fontsize=8)
                ax.set_title(f'Error Example {i+1}', fontsize=self.font_size)
                ax.set_ylabel('Attribution', fontsize=self.font_size - 1)
            else:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'Error Example {i+1}', fontsize=self.font_size)
        
        # Plot correct examples
        for i in range(min(n_examples, len(correct_attributions))):
            ax = axes[1, i]
            data = correct_attributions[i]
            
            # Get top tokens for visualization
            top_tokens = data.get('top_tokens', [])[:20]  # Show top 20
            if top_tokens:
                tokens = [t['token'] for t in top_tokens]
                attrs = [t['attribution'] for t in top_tokens]
                positions = range(len(tokens))
                
                # Color bars by attribution
                colors = [plt.cm.RdBu_r(0.5 + 0.5 * attr / max(abs(min(attrs)), abs(max(attrs))))
                         for attr in attrs]
                
                bars = ax.bar(positions, attrs, color=colors, alpha=0.7)
                ax.set_xticks(positions)
                ax.set_xticklabels(tokens, rotation=45, ha='right', fontsize=8)
                ax.set_title(f'Correct Example {i+1}', fontsize=self.font_size)
                ax.set_ylabel('Attribution', fontsize=self.font_size - 1)
            else:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'Correct Example {i+1}', fontsize=self.font_size)
        
        # Add row labels
        axes[0, 0].text(-0.1, 0.5, 'ERROR\nPREDICTIONS', rotation=90, ha='center', va='center',
                       transform=axes[0, 0].transAxes, fontsize=self.font_size + 1, fontweight='bold')
        axes[1, 0].text(-0.1, 0.5, 'CORRECT\nPREDICTIONS', rotation=90, ha='center', va='center',
                       transform=axes[1, 0].transAxes, fontsize=self.font_size + 1, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Saved comparative plot to {save_path}")
        
        return fig
    
    def plot_positional_analysis(
        self,
        analysis_results: Dict[str, Any],
        title: str = "Positional Attribution Analysis",
        save_path: Optional[Path] = None
    ) -> plt.Figure:
        """
        Plot positional patterns in attributions.
        
        Parameters
        ----------
        analysis_results : Dict[str, Any]
            Results from IGAnalyzer.analyze_error_patterns
        title : str, default "Positional Attribution Analysis"
            Plot title
        save_path : Path, optional
            Path to save the plot
            
        Returns
        -------
        plt.Figure
            The created figure
        """
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        fig.suptitle(title, fontsize=self.font_size + 2, fontweight='bold')
        
        error_pos = analysis_results['positional_analysis']['error_positions']
        correct_pos = analysis_results['positional_analysis']['correct_positions']
        
        # Position distribution comparison
        ax = axes[0, 0]
        if error_pos and correct_pos:
            ax.bar(['Error', 'Correct'], 
                  [error_pos.get('mean_position', 0), correct_pos.get('mean_position', 0)],
                  yerr=[error_pos.get('std_position', 0), correct_pos.get('std_position', 0)],
                  capsize=5, alpha=0.7, color=['red', 'blue'])
            ax.set_ylabel('Mean Token Position')
            ax.set_title('Average Attribution Position')
        
        # Attribution magnitude comparison
        ax = axes[0, 1]
        if error_pos and correct_pos:
            ax.bar(['Error', 'Correct'],
                  [error_pos.get('mean_attribution', 0), correct_pos.get('mean_attribution', 0)],
                  yerr=[error_pos.get('std_attribution', 0), correct_pos.get('std_attribution', 0)],
                  capsize=5, alpha=0.7, color=['red', 'blue'])
            ax.set_ylabel('Mean Attribution Magnitude')
            ax.set_title('Average Attribution Strength')
        
        # Overall statistics
        ax = axes[1, 0]
        error_stats = analysis_results['summary']['error_stats']
        correct_stats = analysis_results['summary']['correct_stats']
        
        stats_names = ['Total', 'Mean', 'Max', 'Min']
        error_values = [
            error_stats.get('mean_total_attribution', 0),
            error_stats.get('mean_mean_attribution', 0),
            error_stats.get('mean_max_attribution', 0),
            error_stats.get('mean_min_attribution', 0)
        ]
        correct_values = [
            correct_stats.get('mean_total_attribution', 0),
            correct_stats.get('mean_mean_attribution', 0),
            correct_stats.get('mean_max_attribution', 0),
            correct_stats.get('mean_min_attribution', 0)
        ]
        
        x = np.arange(len(stats_names))
        width = 0.35
        
        ax.bar(x - width/2, error_values, width, label='Error', alpha=0.7, color='red')
        ax.bar(x + width/2, correct_values, width, label='Correct', alpha=0.7, color='blue')
        
        ax.set_xlabel('Attribution Statistics')
        ax.set_ylabel('Attribution Value')
        ax.set_title('Attribution Statistics Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(stats_names)
        ax.legend()
        
        # Sample counts
        ax = axes[1, 1]
        counts = [analysis_results['summary']['n_error_samples'],
                 analysis_results['summary']['n_correct_samples']]
        ax.pie(counts, labels=['Error', 'Correct'], autopct='%1.1f%%',
              colors=['red', 'blue'], alpha=0.7)
        ax.set_title('Sample Distribution')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Saved positional analysis to {save_path}")
        
        return fig


def create_alignment_plot(
    sequence: str,
    tokens: List[str],
    attributions: List[float],
    title: str = "Token Attribution Alignment",
    save_path: Optional[Path] = None,
    **kwargs
) -> plt.Figure:
    """
    Convenience function to create a single alignment plot.
    
    Parameters
    ----------
    sequence : str
        Original DNA sequence
    tokens : List[str]
        Tokenized sequence
    attributions : List[float]
        Attribution values for each token
    title : str, default "Token Attribution Alignment"
        Plot title
    save_path : Path, optional
        Path to save the plot
    **kwargs
        Additional arguments for AlignmentPlotter
        
    Returns
    -------
    plt.Figure
        The created figure
    """
    plotter = AlignmentPlotter(**kwargs)
    return plotter.plot_sequence_attribution(
        sequence=sequence,
        tokens=tokens,
        attributions=attributions,
        title=title,
        save_path=save_path
    )
