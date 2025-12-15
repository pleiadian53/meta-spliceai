#!/usr/bin/env python3
"""
Visualize Nucleotide-Level Scores

This script demonstrates how to visualize the full splice site landscape
for genes using the nucleotide-level scores output.

Usage:
    python scripts/testing/visualize_nucleotide_scores.py --gene BRCA1
    python scripts/testing/visualize_nucleotide_scores.py --nucleotide-scores-file path/to/nucleotide_scores.tsv
"""

import sys
import argparse
from pathlib import Path
import polars as pl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def load_nucleotide_scores(file_path: str) -> pl.DataFrame:
    """Load nucleotide scores from TSV file."""
    return pl.read_csv(file_path, separator='\t')


def plot_gene_splice_landscape(
    nucleotide_df: pl.DataFrame,
    gene_name: str,
    output_file: str = None,
    show_annotations: bool = True
):
    """
    Plot the splice site landscape for a gene.
    
    Parameters
    ----------
    nucleotide_df : pl.DataFrame
        Nucleotide-level scores DataFrame
    gene_name : str
        Gene name to plot
    output_file : str, optional
        Path to save the plot
    show_annotations : bool
        Whether to show annotated splice sites
    """
    # Filter for the specified gene
    gene_df = nucleotide_df.filter(
        (pl.col('gene_name') == gene_name) | (pl.col('gene_id') == gene_name)
    )
    
    if gene_df.height == 0:
        print(f"Error: Gene '{gene_name}' not found in nucleotide scores")
        return
    
    # Convert to pandas for plotting
    gene_pd = gene_df.to_pandas()
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
    
    # Get gene info
    gene_id = gene_pd['gene_id'].iloc[0]
    chrom = gene_pd['chrom'].iloc[0]
    strand = gene_pd['strand'].iloc[0]
    
    # Plot donor scores
    ax1.plot(gene_pd['position'], gene_pd['donor_score'], 
             color='#2E86AB', linewidth=0.8, label='Donor')
    ax1.fill_between(gene_pd['position'], gene_pd['donor_score'], 
                      alpha=0.3, color='#2E86AB')
    ax1.set_ylabel('Donor Probability', fontsize=11, fontweight='bold')
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(loc='upper right')
    
    # Plot acceptor scores
    ax2.plot(gene_pd['position'], gene_pd['acceptor_score'], 
             color='#A23B72', linewidth=0.8, label='Acceptor')
    ax2.fill_between(gene_pd['position'], gene_pd['acceptor_score'], 
                      alpha=0.3, color='#A23B72')
    ax2.set_ylabel('Acceptor Probability', fontsize=11, fontweight='bold')
    ax2.set_xlabel('Position in Gene (nt)', fontsize=11, fontweight='bold')
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(loc='upper right')
    
    # Add title
    fig.suptitle(
        f'Splice Site Landscape: {gene_name} ({gene_id})\n'
        f'Chr{chrom} | Strand: {strand} | Length: {len(gene_pd):,} nt',
        fontsize=14,
        fontweight='bold'
    )
    
    # Highlight high-confidence predictions (score > 0.5)
    donor_peaks = gene_pd[gene_pd['donor_score'] > 0.5]
    acceptor_peaks = gene_pd[gene_pd['acceptor_score'] > 0.5]
    
    if len(donor_peaks) > 0:
        ax1.scatter(donor_peaks['position'], donor_peaks['donor_score'], 
                   color='red', s=30, alpha=0.6, zorder=5)
    
    if len(acceptor_peaks) > 0:
        ax2.scatter(acceptor_peaks['position'], acceptor_peaks['acceptor_score'], 
                   color='red', s=30, alpha=0.6, zorder=5)
    
    # Add statistics
    stats_text = (
        f"Donor peaks (>0.5): {len(donor_peaks)}\n"
        f"Acceptor peaks (>0.5): {len(acceptor_peaks)}\n"
        f"Max donor score: {gene_pd['donor_score'].max():.3f}\n"
        f"Max acceptor score: {gene_pd['acceptor_score'].max():.3f}"
    )
    
    ax1.text(0.02, 0.98, stats_text, 
             transform=ax1.transAxes,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
             fontsize=9)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✅ Saved plot to: {output_file}")
    else:
        plt.show()
    
    plt.close()


def plot_combined_scores(
    nucleotide_df: pl.DataFrame,
    gene_name: str,
    output_file: str = None
):
    """
    Plot all three scores (donor, acceptor, neither) on the same plot.
    
    Parameters
    ----------
    nucleotide_df : pl.DataFrame
        Nucleotide-level scores DataFrame
    gene_name : str
        Gene name to plot
    output_file : str, optional
        Path to save the plot
    """
    # Filter for the specified gene
    gene_df = nucleotide_df.filter(
        (pl.col('gene_name') == gene_name) | (pl.col('gene_id') == gene_name)
    )
    
    if gene_df.height == 0:
        print(f"Error: Gene '{gene_name}' not found in nucleotide scores")
        return
    
    # Convert to pandas for plotting
    gene_pd = gene_df.to_pandas()
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(15, 6))
    
    # Get gene info
    gene_id = gene_pd['gene_id'].iloc[0]
    chrom = gene_pd['chrom'].iloc[0]
    strand = gene_pd['strand'].iloc[0]
    
    # Plot all three scores
    ax.plot(gene_pd['position'], gene_pd['donor_score'], 
            color='#2E86AB', linewidth=1.2, label='Donor', alpha=0.8)
    ax.plot(gene_pd['position'], gene_pd['acceptor_score'], 
            color='#A23B72', linewidth=1.2, label='Acceptor', alpha=0.8)
    ax.plot(gene_pd['position'], gene_pd['neither_score'], 
            color='#CCCCCC', linewidth=0.8, label='Neither', alpha=0.5)
    
    ax.set_ylabel('Probability', fontsize=12, fontweight='bold')
    ax.set_xlabel('Position in Gene (nt)', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper right', fontsize=11)
    
    # Add title
    ax.set_title(
        f'Complete Splice Site Landscape: {gene_name} ({gene_id})\n'
        f'Chr{chrom} | Strand: {strand} | Length: {len(gene_pd):,} nt',
        fontsize=14,
        fontweight='bold',
        pad=20
    )
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✅ Saved combined plot to: {output_file}")
    else:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Visualize nucleotide-level splice site scores'
    )
    parser.add_argument(
        '--nucleotide-scores-file',
        type=str,
        help='Path to nucleotide_scores.tsv file'
    )
    parser.add_argument(
        '--gene',
        type=str,
        default='BRCA1',
        help='Gene name to visualize (default: BRCA1)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='output/figures',
        help='Output directory for plots (default: output/figures)'
    )
    parser.add_argument(
        '--combined',
        action='store_true',
        help='Create combined plot with all three scores'
    )
    
    args = parser.parse_args()
    
    # Find nucleotide scores file
    if args.nucleotide_scores_file:
        scores_file = Path(args.nucleotide_scores_file)
    else:
        # Try to find the most recent nucleotide_scores.tsv
        data_dir = project_root / 'data'
        scores_files = list(data_dir.rglob('nucleotide_scores.tsv'))
        
        if not scores_files:
            print("Error: No nucleotide_scores.tsv file found")
            print("Please specify --nucleotide-scores-file")
            return 1
        
        # Use the most recently modified file
        scores_file = max(scores_files, key=lambda p: p.stat().st_mtime)
        print(f"Using nucleotide scores file: {scores_file}")
    
    if not scores_file.exists():
        print(f"Error: File not found: {scores_file}")
        return 1
    
    # Load scores
    print(f"Loading nucleotide scores...")
    nucleotide_df = load_nucleotide_scores(str(scores_file))
    print(f"✅ Loaded {nucleotide_df.height:,} nucleotides for {nucleotide_df['gene_id'].n_unique()} genes")
    print()
    
    # List available genes
    genes = nucleotide_df['gene_name'].unique().to_list()
    print(f"Available genes ({len(genes)}):")
    for gene in sorted(genes)[:20]:  # Show first 20
        print(f"  - {gene}")
    if len(genes) > 20:
        print(f"  ... and {len(genes) - 20} more")
    print()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate plots
    gene_name = args.gene
    print(f"Generating plots for {gene_name}...")
    print()
    
    if args.combined:
        output_file = output_dir / f'{gene_name}_combined_scores.png'
        plot_combined_scores(nucleotide_df, gene_name, str(output_file))
    else:
        output_file = output_dir / f'{gene_name}_splice_landscape.png'
        plot_gene_splice_landscape(nucleotide_df, gene_name, str(output_file))
    
    print()
    print("✅ Visualization complete!")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())


