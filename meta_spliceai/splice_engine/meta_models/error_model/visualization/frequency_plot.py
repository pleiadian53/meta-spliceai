"""
Token frequency comparison visualization for IG analysis.

This module provides visualization tools for comparing token frequencies
and importance between error and correct predictions.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle

# Add project root to path for imports
project_root = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(project_root))


class FrequencyPlotter:
    """
    Creates frequency comparison plots for token analysis.
    
    Features:
    1. Token frequency bar charts
    2. Attribution magnitude comparisons
    3. Ratio analysis plots
    4. Top-k token importance visualization
    """
    
    def __init__(
        self,
        figsize: Tuple[int, int] = (12, 8),
        style: str = 'whitegrid',
        palette: str = 'husl'
    ):
        """
        Initialize frequency plotter.
        
        Parameters
        ----------
        figsize : Tuple[int, int], default (12, 8)
            Figure size for plots
        style : str, default 'whitegrid'
            Seaborn style
        palette : str, default 'husl'
            Color palette
        """
        self.figsize = figsize
        self.logger = logging.getLogger(__name__)
        
        # Setup plotting style
        sns.set_style(style)
        sns.set_palette(palette)
        plt.rcParams['figure.figsize'] = figsize
    
    def plot_token_frequency_comparison(
        self,
        analysis_results: Dict[str, Any],
        top_k: int = 20,
        title: str = "Token Frequency Comparison",
        save_path: Optional[Path] = None
    ) -> plt.Figure:
        """
        Plot token frequency comparison between error and correct predictions.
        
        Parameters
        ----------
        analysis_results : Dict[str, Any]
            Results from IGAnalyzer.analyze_error_patterns
        top_k : int, default 20
            Number of top tokens to display
        title : str, default "Token Frequency Comparison"
            Plot title
        save_path : Path, optional
            Path to save the plot
            
        Returns
        -------
        plt.Figure
            The created figure
        """
        # Extract token data
        error_freq = analysis_results['token_analysis']['error_token_freq']
        correct_freq = analysis_results['token_analysis']['correct_token_freq']
        token_ratios = analysis_results['token_analysis']['token_ratios']
        
        # Get top tokens by ratio
        top_tokens = list(token_ratios.keys())[:top_k]
        
        # Prepare data
        token_data = []
        for token in top_tokens:
            error_stats = error_freq.get(token, {})
            correct_stats = correct_freq.get(token, {})
            
            token_data.append({
                'token': token,
                'error_freq': error_stats.get('frequency', 0),
                'correct_freq': correct_stats.get('frequency', 0),
                'error_rel_freq': error_stats.get('relative_frequency', 0.0),
                'correct_rel_freq': correct_stats.get('relative_frequency', 0.0),
                'ratio': token_ratios.get(token, 0.0),
                'error_mean_attr': error_stats.get('mean_attribution', 0.0),
                'correct_mean_attr': correct_stats.get('mean_attribution', 0.0)
            })
        
        df = pd.DataFrame(token_data)
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(self.figsize[0], self.figsize[1] * 1.2))
        fig.suptitle(title, fontsize=14, fontweight='bold')
        
        # 1. Absolute frequency comparison
        ax = axes[0, 0]
        x = np.arange(len(df))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, df['error_freq'], width, label='Error', alpha=0.8, color='red')
        bars2 = ax.bar(x + width/2, df['correct_freq'], width, label='Correct', alpha=0.8, color='blue')
        
        ax.set_xlabel('Tokens')
        ax.set_ylabel('Absolute Frequency')
        ax.set_title('Absolute Token Frequencies')
        ax.set_xticks(x)
        ax.set_xticklabels(df['token'], rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Relative frequency comparison
        ax = axes[0, 1]
        bars1 = ax.bar(x - width/2, df['error_rel_freq'], width, label='Error', alpha=0.8, color='red')
        bars2 = ax.bar(x + width/2, df['correct_rel_freq'], width, label='Correct', alpha=0.8, color='blue')
        
        ax.set_xlabel('Tokens')
        ax.set_ylabel('Relative Frequency')
        ax.set_title('Relative Token Frequencies')
        ax.set_xticks(x)
        ax.set_xticklabels(df['token'], rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Frequency ratio
        ax = axes[1, 0]
        colors = ['red' if ratio > 1 else 'blue' for ratio in df['ratio']]
        bars = ax.bar(x, df['ratio'], alpha=0.8, color=colors)
        
        ax.axhline(y=1, color='black', linestyle='--', alpha=0.5)
        ax.set_xlabel('Tokens')
        ax.set_ylabel('Frequency Ratio (Error/Correct)')
        ax.set_title('Token Frequency Ratios')
        ax.set_xticks(x)
        ax.set_xticklabels(df['token'], rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
        
        # Add ratio values on bars
        for i, (bar, ratio) in enumerate(zip(bars, df['ratio'])):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                   f'{ratio:.2f}', ha='center', va='bottom', fontsize=8)
        
        # 4. Mean attribution comparison
        ax = axes[1, 1]
        bars1 = ax.bar(x - width/2, df['error_mean_attr'], width, label='Error', alpha=0.8, color='red')
        bars2 = ax.bar(x + width/2, df['correct_mean_attr'], width, label='Correct', alpha=0.8, color='blue')
        
        ax.set_xlabel('Tokens')
        ax.set_ylabel('Mean Attribution')
        ax.set_title('Mean Token Attributions')
        ax.set_xticks(x)
        ax.set_xticklabels(df['token'], rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Saved frequency comparison to {save_path}")
        
        return fig
    
    def plot_attribution_distribution(
        self,
        analysis_results: Dict[str, Any],
        title: str = "Attribution Distribution Analysis",
        save_path: Optional[Path] = None
    ) -> plt.Figure:
        """
        Plot distribution of attribution values.
        
        Parameters
        ----------
        analysis_results : Dict[str, Any]
            Results from IGAnalyzer.analyze_error_patterns
        title : str, default "Attribution Distribution Analysis"
            Plot title
        save_path : Path, optional
            Path to save the plot
            
        Returns
        -------
        plt.Figure
            The created figure
        """
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        fig.suptitle(title, fontsize=14, fontweight='bold')
        
        # Extract statistics
        error_stats = analysis_results['summary']['error_stats']
        correct_stats = analysis_results['summary']['correct_stats']
        
        # 1. Total attribution comparison
        ax = axes[0, 0]
        categories = ['Error', 'Correct']
        means = [error_stats.get('mean_total_attribution', 0),
                correct_stats.get('mean_total_attribution', 0)]
        stds = [error_stats.get('std_total_attribution', 0),
               correct_stats.get('std_total_attribution', 0)]
        
        bars = ax.bar(categories, means, yerr=stds, capsize=5, alpha=0.8, color=['red', 'blue'])
        ax.set_ylabel('Total Attribution')
        ax.set_title('Total Attribution per Sequence')
        ax.grid(True, alpha=0.3)
        
        # Add values on bars
        for bar, mean, std in zip(bars, means, stds):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.01,
                   f'{mean:.3f}±{std:.3f}', ha='center', va='bottom', fontsize=10)
        
        # 2. Mean attribution comparison
        ax = axes[0, 1]
        means = [error_stats.get('mean_mean_attribution', 0),
                correct_stats.get('mean_mean_attribution', 0)]
        
        bars = ax.bar(categories, means, alpha=0.8, color=['red', 'blue'])
        ax.set_ylabel('Mean Attribution per Token')
        ax.set_title('Average Token Attribution')
        ax.grid(True, alpha=0.3)
        
        # Add values on bars
        for bar, mean in zip(bars, means):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                   f'{mean:.4f}', ha='center', va='bottom', fontsize=10)
        
        # 3. Max attribution comparison
        ax = axes[1, 0]
        means = [error_stats.get('mean_max_attribution', 0),
                correct_stats.get('mean_max_attribution', 0)]
        
        bars = ax.bar(categories, means, alpha=0.8, color=['red', 'blue'])
        ax.set_ylabel('Max Attribution')
        ax.set_title('Maximum Token Attribution')
        ax.grid(True, alpha=0.3)
        
        # Add values on bars
        for bar, mean in zip(bars, means):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{mean:.3f}', ha='center', va='bottom', fontsize=10)
        
        # 4. Min attribution comparison
        ax = axes[1, 1]
        means = [error_stats.get('mean_min_attribution', 0),
                correct_stats.get('mean_min_attribution', 0)]
        
        bars = ax.bar(categories, means, alpha=0.8, color=['red', 'blue'])
        ax.set_ylabel('Min Attribution')
        ax.set_title('Minimum Token Attribution')
        ax.grid(True, alpha=0.3)
        
        # Add values on bars
        for bar, mean in zip(bars, means):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height - 0.01,
                   f'{mean:.3f}', ha='center', va='top', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Saved attribution distribution to {save_path}")
        
        return fig
    
    def plot_top_tokens_analysis(
        self,
        analysis_results: Dict[str, Any],
        top_k: int = 15,
        title: str = "Top Tokens Analysis",
        save_path: Optional[Path] = None
    ) -> plt.Figure:
        """
        Plot detailed analysis of top important tokens.
        
        Parameters
        ----------
        analysis_results : Dict[str, Any]
            Results from IGAnalyzer.analyze_error_patterns
        top_k : int, default 15
            Number of top tokens to analyze
        title : str, default "Top Tokens Analysis"
            Plot title
        save_path : Path, optional
            Path to save the plot
            
        Returns
        -------
        plt.Figure
            The created figure
        """
        # Extract token data
        error_freq = analysis_results['token_analysis']['error_token_freq']
        correct_freq = analysis_results['token_analysis']['correct_token_freq']
        token_ratios = analysis_results['token_analysis']['token_ratios']
        
        # Get top tokens by ratio
        top_tokens = list(token_ratios.keys())[:top_k]
        
        # Prepare detailed data
        detailed_data = []
        for token in top_tokens:
            error_stats = error_freq.get(token, {})
            correct_stats = correct_freq.get(token, {})
            
            detailed_data.append({
                'token': token,
                'error_freq': error_stats.get('frequency', 0),
                'correct_freq': correct_stats.get('frequency', 0),
                'error_rel_freq': error_stats.get('relative_frequency', 0.0),
                'correct_rel_freq': correct_stats.get('relative_frequency', 0.0),
                'ratio': token_ratios.get(token, 0.0),
                'error_mean_attr': error_stats.get('mean_attribution', 0.0),
                'correct_mean_attr': correct_stats.get('mean_attribution', 0.0),
                'error_std_attr': error_stats.get('std_attribution', 0.0),
                'correct_std_attr': correct_stats.get('std_attribution', 0.0),
                'error_total_attr': error_stats.get('total_attribution', 0.0),
                'correct_total_attr': correct_stats.get('total_attribution', 0.0)
            })
        
        df = pd.DataFrame(detailed_data)
        
        # Create figure with custom layout
        fig = plt.figure(figsize=(self.figsize[0] * 1.5, self.figsize[1] * 1.2))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # 1. Token importance heatmap
        ax1 = fig.add_subplot(gs[0, :])
        
        # Create heatmap data
        heatmap_data = df[['error_rel_freq', 'correct_rel_freq', 'ratio', 
                          'error_mean_attr', 'correct_mean_attr']].T
        heatmap_data.columns = df['token']
        
        # Normalize each row for better visualization
        heatmap_normalized = heatmap_data.div(heatmap_data.abs().max(axis=1), axis=0)
        
        im = ax1.imshow(heatmap_normalized.values, cmap='RdBu_r', aspect='auto')
        ax1.set_xticks(range(len(df)))
        ax1.set_xticklabels(df['token'], rotation=45, ha='right')
        ax1.set_yticks(range(len(heatmap_data)))
        ax1.set_yticklabels(['Error Rel Freq', 'Correct Rel Freq', 'Ratio', 
                            'Error Mean Attr', 'Correct Mean Attr'])
        ax1.set_title('Token Importance Heatmap (Normalized)')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax1, orientation='horizontal', pad=0.1, aspect=30)
        cbar.set_label('Normalized Value')
        
        # 2. Scatter plot: Frequency vs Attribution
        ax2 = fig.add_subplot(gs[1, 0])
        
        # Error points
        ax2.scatter(df['error_rel_freq'], df['error_mean_attr'], 
                   s=100, alpha=0.7, color='red', label='Error', edgecolors='black')
        
        # Correct points
        ax2.scatter(df['correct_rel_freq'], df['correct_mean_attr'], 
                   s=100, alpha=0.7, color='blue', label='Correct', edgecolors='black')
        
        # Add token labels
        for i, row in df.iterrows():
            ax2.annotate(row['token'], 
                        (row['error_rel_freq'], row['error_mean_attr']),
                        xytext=(5, 5), textcoords='offset points', fontsize=8, alpha=0.8)
        
        ax2.set_xlabel('Relative Frequency')
        ax2.set_ylabel('Mean Attribution')
        ax2.set_title('Frequency vs Attribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Token ratio ranking
        ax3 = fig.add_subplot(gs[1, 1])
        
        # Sort by ratio for better visualization
        df_sorted = df.sort_values('ratio', ascending=True)
        y_pos = np.arange(len(df_sorted))
        
        colors = ['red' if r > 1 else 'blue' for r in df_sorted['ratio']]
        bars = ax3.barh(y_pos, df_sorted['ratio'], alpha=0.8, color=colors)
        
        ax3.axvline(x=1, color='black', linestyle='--', alpha=0.5)
        ax3.set_yticks(y_pos)
        ax3.set_yticklabels(df_sorted['token'])
        ax3.set_xlabel('Frequency Ratio (Error/Correct)')
        ax3.set_title('Token Importance Ranking')
        ax3.grid(True, alpha=0.3)
        
        # Add ratio values
        for i, (bar, ratio) in enumerate(zip(bars, df_sorted['ratio'])):
            width = bar.get_width()
            ax3.text(width + 0.05, bar.get_y() + bar.get_height()/2,
                    f'{ratio:.2f}', ha='left', va='center', fontsize=8)
        
        # 4. Attribution variability
        ax4 = fig.add_subplot(gs[2, :])
        
        x = np.arange(len(df))
        width = 0.35
        
        # Error bars with error bars
        bars1 = ax4.bar(x - width/2, df['error_mean_attr'], width, 
                       yerr=df['error_std_attr'], capsize=3,
                       label='Error', alpha=0.8, color='red')
        
        # Correct bars with error bars
        bars2 = ax4.bar(x + width/2, df['correct_mean_attr'], width,
                       yerr=df['correct_std_attr'], capsize=3,
                       label='Correct', alpha=0.8, color='blue')
        
        ax4.set_xlabel('Tokens')
        ax4.set_ylabel('Mean Attribution ± Std')
        ax4.set_title('Attribution Variability by Token')
        ax4.set_xticks(x)
        ax4.set_xticklabels(df['token'], rotation=45, ha='right')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Saved top tokens analysis to {save_path}")
        
        return fig
    
    def create_summary_report(
        self,
        analysis_results: Dict[str, Any],
        output_dir: Path,
        report_name: str = "ig_analysis_report"
    ) -> Dict[str, Path]:
        """
        Create comprehensive summary report with all visualizations.
        
        Parameters
        ----------
        analysis_results : Dict[str, Any]
            Results from IGAnalyzer.analyze_error_patterns
        output_dir : Path
            Output directory for report
        report_name : str, default "ig_analysis_report"
            Base name for report files
            
        Returns
        -------
        Dict[str, Path]
            Paths to generated report files
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        report_files = {}
        
        # 1. Token frequency comparison
        fig1 = self.plot_token_frequency_comparison(analysis_results)
        path1 = output_dir / f"{report_name}_frequency_comparison.png"
        fig1.savefig(path1, dpi=300, bbox_inches='tight')
        plt.close(fig1)
        report_files['frequency_comparison'] = path1
        
        # 2. Attribution distribution
        fig2 = self.plot_attribution_distribution(analysis_results)
        path2 = output_dir / f"{report_name}_attribution_distribution.png"
        fig2.savefig(path2, dpi=300, bbox_inches='tight')
        plt.close(fig2)
        report_files['attribution_distribution'] = path2
        
        # 3. Top tokens analysis
        fig3 = self.plot_top_tokens_analysis(analysis_results)
        path3 = output_dir / f"{report_name}_top_tokens.png"
        fig3.savefig(path3, dpi=300, bbox_inches='tight')
        plt.close(fig3)
        report_files['top_tokens'] = path3
        
        self.logger.info(f"Created comprehensive IG analysis report in {output_dir}")
        return report_files


def create_frequency_plot(
    analysis_results: Dict[str, Any],
    plot_type: str = "frequency_comparison",
    save_path: Optional[Path] = None,
    **kwargs
) -> plt.Figure:
    """
    Convenience function to create frequency plots.
    
    Parameters
    ----------
    analysis_results : Dict[str, Any]
        Results from IGAnalyzer.analyze_error_patterns
    plot_type : str, default "frequency_comparison"
        Type of plot: "frequency_comparison", "attribution_distribution", "top_tokens"
    save_path : Path, optional
        Path to save the plot
    **kwargs
        Additional arguments for FrequencyPlotter
        
    Returns
    -------
    plt.Figure
        The created figure
    """
    plotter = FrequencyPlotter(**kwargs)
    
    if plot_type == "frequency_comparison":
        return plotter.plot_token_frequency_comparison(analysis_results, save_path=save_path)
    elif plot_type == "attribution_distribution":
        return plotter.plot_attribution_distribution(analysis_results, save_path=save_path)
    elif plot_type == "top_tokens":
        return plotter.plot_top_tokens_analysis(analysis_results, save_path=save_path)
    else:
        raise ValueError(f"Unknown plot_type: {plot_type}")
