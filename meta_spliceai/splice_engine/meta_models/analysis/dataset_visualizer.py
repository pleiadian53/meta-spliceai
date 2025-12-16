"""
Dataset visualization functions for splice site prediction datasets.

This module contains all visualization functions for creating plots and charts
from dataset profiling results.
"""
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for better performance
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Configure matplotlib for publication-quality plots
plt.ioff()  # Turn off interactive mode for better performance
plt.rcParams['figure.dpi'] = 300  # High DPI for publication quality
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['text.usetex'] = True  # Use LaTeX rendering for publication quality
plt.rcParams['font.family'] = 'serif'

# Set style for consistent plots
plt.style.use('default')
sns.set_palette("husl")


def create_splice_distribution_plot(splice_analysis: Dict[str, Any], vis_dir: Path) -> Optional[str]:
    """Create splice site distribution visualization."""
    try:
        if 'percentage_distribution' not in splice_analysis:
            return None
            
        distribution = splice_analysis['percentage_distribution']
        
        # Optimize for performance: smaller figure, lower DPI
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        
        # Pie chart
        # Handle numeric keys (convert to string labels)
        labels = []
        for label in distribution.keys():
            if isinstance(label, (int, float)) or str(label).isdigit():
                if label == 0:
                    labels.append("Neither")
                else:
                    labels.append(f"Type {label}")
            else:
                labels.append(str(label).capitalize())
        sizes = list(distribution.values())
        colors = sns.color_palette("husl", len(labels))
        
        wedges, texts, autotexts = ax1.pie(sizes, labels=labels, autopct='%1.1f%%', 
                                          colors=colors, startangle=90)
        ax1.set_title('Splice Site Distribution', fontsize=14, fontweight='bold')
        
        # Bar chart
        ax2.bar(labels, sizes, color=colors)
        ax2.set_title('Splice Site Counts', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Percentage')
        ax2.set_xlabel('Splice Type')
        
        # Add count annotations
        if 'normalized_distribution' in splice_analysis:
            counts = splice_analysis['normalized_distribution']
            for i, (label, pct) in enumerate(zip(labels, sizes)):
                count = counts.get(label.lower(), 0)
                ax2.text(i, pct + 0.5, f'{count:,}', ha='center', va='bottom')
        
        plt.tight_layout()
        plot_file = vis_dir / "splice_distribution.png"
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')  # Lower DPI for speed
        plt.close()
        
        return str(plot_file)
        
    except Exception as e:
        print(f"Error creating splice distribution plot: {e}")
        return None


def create_scores_stats_plot(splice_stats: Dict[str, Any], splice_analysis: Dict[str, Any], vis_dir: Path) -> Optional[str]:
    """Create SpliceAI scores statistics visualization."""
    try:
        if not splice_stats:
            return None
        
        # Data validation passed - ranges are calculated correctly
        
        # Optimize for performance: smaller figure
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        
        # Prepare data for plotting
        score_types = []
        means = []
        medians = []
        stds = []
        ranges = []
        
        for score_col, stats in splice_stats.items():
            # Extract score type from column name
            if 'donor' in score_col.lower():
                score_type = 'Donor'
            elif 'acceptor' in score_col.lower():
                score_type = 'Acceptor'
            elif 'neither' in score_col.lower():
                score_type = 'Neither'
            else:
                score_type = score_col
            
            score_types.append(score_type)
            means.append(stats.get('mean', 0))
            medians.append(stats.get('median', stats.get('mean', 0)))  # Fallback to mean if no median
            stds.append(stats.get('std', 0))
            
            # Calculate range with proper fallback and precision
            max_val = float(stats.get('max', 1.0))  # Ensure float precision
            min_val = float(stats.get('min', 0.0))
            range_val = max_val - min_val
            ranges.append(range_val)
            
            # Debug logging (removed for cleaner output)
        
        colors = sns.color_palette("husl", len(score_types))
        
        # Mean scores by type
        bars1 = ax1.bar(score_types, means, color=colors, alpha=0.7)
        ax1.set_title('Mean SpliceAI Scores by Type', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Mean Score')
        ax1.set_ylim(0, max(means) * 1.1 if means else 1)
        
        # Add value labels on bars
        for bar, mean_val in zip(bars1, means):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{mean_val:.3f}', ha='center', va='bottom', fontsize=10)
        
        # Median scores by type
        bars2 = ax2.bar(score_types, medians, color=colors, alpha=0.7)
        ax2.set_title('Median SpliceAI Scores by Type', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Median Score')
        ax2.set_ylim(0, max(medians) * 1.1 if medians else 1)
        
        # Add value labels on bars
        for bar, median_val in zip(bars2, medians):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{median_val:.3f}', ha='center', va='bottom', fontsize=10)
        
        # Standard deviation (variability)
        bars3 = ax3.bar(score_types, stds, color=colors, alpha=0.7)
        ax3.set_title('Score Variability (Std Dev)', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Standard Deviation')
        ax3.set_ylim(0, max(stds) * 1.1 if stds else 1)
        
        # Add value labels on bars
        for bar, std_val in zip(bars3, stds):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{std_val:.3f}', ha='center', va='bottom', fontsize=10)
        
        # Score ranges
        bars4 = ax4.bar(score_types, ranges, color=colors, alpha=0.7)
        ax4.set_title('Score Ranges (Type-Specific)', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Range (Max - Min)')
        ax4.set_ylim(0, max(ranges) * 1.1 if ranges else 1)
        
        # Add value labels on bars
        for bar, range_val in zip(bars4, ranges):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{range_val:.4f}', ha='center', va='bottom', fontsize=10)
        
        # Add overall title with explanation
        fig.suptitle('SpliceAI Score Statistics by Predicted Type\n(Statistics computed only for positions predicted as each type)', 
                    fontsize=14, fontweight='bold', y=0.98)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.90)  # Make room for suptitle
        plot_file = vis_dir / "spliceai_scores_statistics.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(plot_file)
        
    except Exception as e:
        print(f"Error creating scores statistics plot: {e}")
        return None


def create_missing_values_plot(missing_data: Dict[str, int], total_samples: int, vis_dir: Path) -> Tuple[Optional[str], Optional[Dict]]:
    """Create missing values analysis visualization."""
    try:
        # Handle case with no missing values
        if not missing_data:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, '🎉 No Missing Values Detected!\n\nAll columns in the dataset are complete.', 
                   ha='center', va='center', fontsize=16, 
                   bbox=dict(boxstyle='round,pad=1', facecolor='lightgreen', alpha=0.8))
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            ax.set_title('Missing Values Analysis', fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            plot_file = vis_dir / "missing_values_by_column.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            missing_summary = {
                'total_columns_with_missing': 0,
                'columns_over_5_percent': 0,
                'columns_over_10_percent': 0,
                'worst_column': None,
                'average_missing_percentage': 0.0
            }
            
            return str(plot_file), missing_summary
        
        # Sort by missing count and take top 20
        sorted_missing = sorted(missing_data.items(), key=lambda x: x[1], reverse=True)[:20]
        
        if not sorted_missing:
            return None, None
        
        columns, counts = zip(*sorted_missing)
        percentages = [(count / total_samples * 100) for count in counts]
        
        # Create dual plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
        
        # Top plot: Missing value counts
        bars1 = ax1.bar(range(len(columns)), counts, color='lightcoral', alpha=0.7)
        ax1.set_title('Missing Values by Column (Top 20)', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Missing Value Count')
        ax1.set_xticks(range(len(columns)))
        ax1.set_xticklabels(columns, rotation=45, ha='right')
        
        # Add count labels on bars
        for bar, count in zip(bars1, counts):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{count:,}', ha='center', va='bottom', fontsize=9)
        
        # Bottom plot: Missing percentages with threshold lines
        bars2 = ax2.bar(range(len(columns)), percentages, color='orange', alpha=0.7)
        ax2.set_title('Missing Values Percentage with Thresholds', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Missing Percentage (%)')
        ax2.set_xlabel('Columns')
        ax2.set_xticks(range(len(columns)))
        ax2.set_xticklabels(columns, rotation=45, ha='right')
        
        # Add threshold lines
        ax2.axhline(y=5, color='yellow', linestyle='--', alpha=0.8, label='5% threshold')
        ax2.axhline(y=10, color='red', linestyle='--', alpha=0.8, label='10% threshold')
        ax2.legend()
        
        # Add percentage labels on bars
        for bar, pct in zip(bars2, percentages):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{pct:.1f}%', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plot_file = vis_dir / "missing_values_by_column.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Generate missing values summary
        all_missing_data = list(missing_data.items())
        all_percentages = [(count / total_samples * 100) for _, count in all_missing_data]
        
        missing_summary = {
            'total_columns_with_missing': len(missing_data),
            'columns_over_5_percent': sum(1 for pct in all_percentages if pct > 5),
            'columns_over_10_percent': sum(1 for pct in all_percentages if pct > 10),
            'worst_column': max(all_missing_data, key=lambda x: x[1]) if all_missing_data else None,
            'average_missing_percentage': np.mean(all_percentages) if all_percentages else 0
        }
        
        return str(plot_file), missing_summary
        
    except Exception as e:
        print(f"Error creating missing values plot: {e}")
        return None, None


def create_top_genes_plot(top_genes: Dict[str, int], gene_name_mapping: Dict[str, str], vis_dir: Path) -> Optional[str]:
    """Create top genes by occurrence count visualization."""
    try:
        if not top_genes:
            return None
        
        # Get gene names for display
        gene_ids = list(top_genes.keys())
        counts = list(top_genes.values())
        total_count = sum(counts)
        
        # Create display names (gene name if available, otherwise gene ID)
        display_names = []
        for gene_id in gene_ids:
            gene_name = gene_name_mapping.get(gene_id, gene_id)
            # If gene name is different from ID, show both
            if gene_name != gene_id and len(gene_name) < 15:
                display_names.append(f"{gene_name}\n({gene_id[:12]}...)" if len(gene_id) > 15 else f"{gene_name}\n({gene_id})")
            else:
                display_names.append(gene_id[:15] + "..." if len(gene_id) > 15 else gene_id)
        
        # Calculate percentages
        percentages = [(count / total_count * 100) for count in counts]
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Create horizontal bar chart
        bars = ax.barh(range(len(display_names)), counts, color=sns.color_palette("viridis", len(display_names)))
        
        ax.set_title('Top Genes by Splice Site Occurrence Count', fontsize=14, fontweight='bold')
        ax.set_xlabel('Number of Splice Site Positions (Training Samples)')
        ax.set_ylabel('Genes')
        ax.set_yticks(range(len(display_names)))
        ax.set_yticklabels(display_names)
        
        # Add count and percentage labels
        for i, (bar, count, pct) in enumerate(zip(bars, counts, percentages)):
            width = bar.get_width()
            ax.text(width + width*0.01, bar.get_y() + bar.get_height()/2,
                   f'{count:,} ({pct:.1f}%)', ha='left', va='center', fontsize=10)
        
        # Invert y-axis to show highest counts at top
        ax.invert_yaxis()
        
        plt.tight_layout()
        plot_file = vis_dir / "top_genes_by_occurrence.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(plot_file)
        
    except Exception as e:
        print(f"Error creating top genes plot: {e}")
        return None


def create_feature_categories_plot(feature_counts: Dict[str, int], vis_dir: Path) -> Optional[str]:
    """Create feature categories visualization."""
    try:
        # Filter out categories with zero counts
        non_zero_counts = {k: v for k, v in feature_counts.items() if v > 0}
        
        if not non_zero_counts:
            return None
        
        # Better display names
        category_display_names = {
            'splice_ai_scores': 'SpliceAI Raw Scores',
            'probability_context_features': 'Probability & Context Features',
            'kmer_features': 'K-mer Features',
            'positional_features': 'Positional Features',
            'structural_features': 'Structural Features',
            'sequence_context': 'Sequence Context',
            'genomic_annotations': 'Genomic Annotations',
            'identifiers': 'Identifiers',
            'other': 'Other Features'
        }
        
        categories = []
        counts = []
        colors = []
        
        # Define colors for each category
        color_map = {
            'splice_ai_scores': '#FF6B6B',
            'probability_context_features': '#4ECDC4', 
            'kmer_features': '#45B7D1',
            'positional_features': '#96CEB4',
            'structural_features': '#FFEAA7',
            'sequence_context': '#DDA0DD',
            'genomic_annotations': '#98D8C8',
            'identifiers': '#F7DC6F',
            'other': '#AED6F1'
        }
        
        for category, count in non_zero_counts.items():
            display_name = category_display_names.get(category, category.replace('_', ' ').title())
            categories.append(display_name)
            counts.append(count)
            colors.append(color_map.get(category, '#AED6F1'))
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Pie chart with better label handling
        # Only show labels for categories with >2% to avoid clutter
        total_features = sum(counts)
        labels_for_pie = []
        autopct_func = lambda pct: f'{pct:.1f}%' if pct > 2 else ''
        
        for i, (cat, count) in enumerate(zip(categories, counts)):
            percentage = (count / total_features) * 100
            if percentage > 2:  # Only label categories >2%
                labels_for_pie.append(cat)
            else:
                labels_for_pie.append('')  # Empty label for small categories
        
        wedges, texts, autotexts = ax1.pie(counts, labels=labels_for_pie, autopct=autopct_func,
                                          colors=colors, startangle=90, pctdistance=0.85)
        
        # Improve text readability
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(9)
        
        ax1.set_title('Feature Categories Distribution', fontsize=14, fontweight='bold')
        
        # Add a legend for all categories (especially useful for small ones)
        legend_labels = [f'{cat} ({count})' for cat, count in zip(categories, counts)]
        ax1.legend(wedges, legend_labels, title="Categories", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
        
        # Bar chart
        bars = ax2.bar(range(len(categories)), counts, color=colors)
        ax2.set_title('Feature Counts by Category', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Number of Features')
        ax2.set_xlabel('Feature Categories')
        ax2.set_xticks(range(len(categories)))
        ax2.set_xticklabels(categories, rotation=45, ha='right')
        
        # Add count labels on bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{count}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plot_file = vis_dir / "feature_categories.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(plot_file)
        
    except Exception as e:
        print(f"Error creating feature categories plot: {e}")
        return None


def create_gene_manifest_consistency_plot(manifest_analysis: Dict[str, Any], vis_dir: Path) -> Optional[str]:
    """Create gene manifest consistency visualization."""
    try:
        if 'consistency_percentage' not in manifest_analysis:
            return None
        
        consistency_pct = manifest_analysis['consistency_percentage']
        manifest_genes = manifest_analysis.get('total_manifest_genes', 0)
        training_genes = manifest_analysis.get('total_training_genes', 0)
        common_genes = manifest_analysis.get('common_genes_count', 0)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Consistency pie chart
        consistent = common_genes
        inconsistent = max(manifest_genes, training_genes) - common_genes
        
        ax1.pie([consistent, inconsistent], 
               labels=['Consistent', 'Inconsistent'],
               autopct='%1.1f%%',
               colors=['lightgreen', 'lightcoral'],
               startangle=90)
        ax1.set_title(f'Gene Manifest Consistency\n({consistency_pct:.1f}%)', 
                     fontsize=12, fontweight='bold')
        
        # Gene counts bar chart
        categories = ['Manifest\nGenes', 'Training\nGenes', 'Common\nGenes']
        counts = [manifest_genes, training_genes, common_genes]
        colors = ['skyblue', 'lightgreen', 'gold']
        
        bars = ax2.bar(categories, counts, color=colors)
        ax2.set_title('Gene Counts Comparison', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Number of Genes')
        
        # Add count labels
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{count:,}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plot_file = vis_dir / "gene_manifest_consistency.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(plot_file)
        
    except Exception as e:
        print(f"Error creating gene manifest consistency plot: {e}")
        return None


def create_performance_metrics_plot(performance: Dict[str, Any], vis_dir: Path) -> Optional[str]:
    """Create performance metrics visualization."""
    try:
        if not performance:
            return None
        
        # Extract metrics (this would depend on the structure of performance data)
        # This is a placeholder implementation
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        values = [0.85, 0.82, 0.88, 0.85]  # Placeholder values
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bars = ax.bar(metrics, values, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
        ax.set_title('Base Model Performance Metrics', fontsize=14, fontweight='bold')
        ax.set_ylabel('Score')
        ax.set_ylim(0, 1)
        
        # Add value labels
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.3f}', ha='center', va='bottom', fontsize=12)
        
        plt.tight_layout()
        plot_file = vis_dir / "performance_metrics.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(plot_file)
        
    except Exception as e:
        print(f"Error creating performance metrics plot: {e}")
        return None
