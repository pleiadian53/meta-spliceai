#!/usr/bin/env python3
"""
Generate global summary statistics for FP/FN analysis across all genes.

This script reads the gene-level FP/FN results and provides:
1. Global totals across all genes
2. Summary statistics
3. Top genes by improvement metrics
"""

import pandas as pd
import argparse
from pathlib import Path


def generate_global_summary(input_file: str, output_file: str = None):
    """Generate global FP/FN summary statistics."""
    
    # Read gene-level results
    df = pd.read_csv(input_file, sep='\t')
    
    # Check if FP/FN columns exist
    fp_fn_cols = ['base_donor_fp', 'base_donor_fn', 'base_acceptor_fp', 'base_acceptor_fn',
                  'rescued_donor_fn', 'eliminated_donor_fp', 'rescued_acceptor_fn', 
                  'eliminated_acceptor_fp']
    
    if not all(col in df.columns for col in fp_fn_cols):
        print("❌ Error: FP/FN columns not found in input file")
        print("Available columns:", list(df.columns))
        return
    
    # Calculate global totals
    global_stats = {}
    
    # Base model errors
    global_stats['total_base_donor_fp'] = df['base_donor_fp'].sum()
    global_stats['total_base_donor_fn'] = df['base_donor_fn'].sum()
    global_stats['total_base_acceptor_fp'] = df['base_acceptor_fp'].sum()
    global_stats['total_base_acceptor_fn'] = df['base_acceptor_fn'].sum()
    
    # Meta model improvements
    global_stats['total_rescued_donor_fn'] = df['rescued_donor_fn'].sum()
    global_stats['total_eliminated_donor_fp'] = df['eliminated_donor_fp'].sum()
    global_stats['total_rescued_acceptor_fn'] = df['rescued_acceptor_fn'].sum()
    global_stats['total_eliminated_acceptor_fp'] = df['eliminated_acceptor_fp'].sum()
    
    # Total improvements
    global_stats['total_improvements'] = (
        global_stats['total_rescued_donor_fn'] + 
        global_stats['total_eliminated_donor_fp'] +
        global_stats['total_rescued_acceptor_fn'] + 
        global_stats['total_eliminated_acceptor_fp']
    )
    
    # Calculate improvement rates
    total_base_errors = (
        global_stats['total_base_donor_fp'] + global_stats['total_base_donor_fn'] +
        global_stats['total_base_acceptor_fp'] + global_stats['total_base_acceptor_fn']
    )
    
    improvement_rate = (global_stats['total_improvements'] / total_base_errors * 100) if total_base_errors > 0 else 0
    
    # Print summary
    print("🧬 Global FP/FN Summary Statistics")
    print("=" * 50)
    print(f"📊 Total genes analyzed: {len(df):,}")
    print(f"📊 Total splice sites: {df['total_splice_sites'].sum():,}")
    print()
    
    print("🔴 Base Model Errors:")
    print(f"  • Donor FP: {global_stats['total_base_donor_fp']:,}")
    print(f"  • Donor FN: {global_stats['total_base_donor_fn']:,}")
    print(f"  • Acceptor FP: {global_stats['total_base_acceptor_fp']:,}")
    print(f"  • Acceptor FN: {global_stats['total_base_acceptor_fn']:,}")
    print(f"  • Total base errors: {total_base_errors:,}")
    print()
    
    print("🟢 Meta Model Improvements:")
    print(f"  • Rescued donor FN: {global_stats['total_rescued_donor_fn']:,}")
    print(f"  • Eliminated donor FP: {global_stats['total_eliminated_donor_fp']:,}")
    print(f"  • Rescued acceptor FN: {global_stats['total_rescued_acceptor_fn']:,}")
    print(f"  • Eliminated acceptor FP: {global_stats['total_eliminated_acceptor_fp']:,}")
    print(f"  • Total improvements: {global_stats['total_improvements']:,}")
    print(f"  • Improvement rate: {improvement_rate:.1f}%")
    print()
    
    # Top genes by improvement
    print("🏆 Top 10 Genes by Total Improvements:")
    print("-" * 50)
    top_genes = df.nlargest(10, 'total_improvements')[['gene_name', 'gene_id', 'total_improvements', 
                                                       'rescued_donor_fn', 'eliminated_donor_fp',
                                                       'rescued_acceptor_fn', 'eliminated_acceptor_fp']]
    
    for i, row in top_genes.iterrows():
        print(f"{row['gene_name']} ({row['gene_id']}): {int(row['total_improvements'])} improvements")
        print(f"  └─ Rescued: {int(row['rescued_donor_fn'])} donor FN, {int(row['rescued_acceptor_fn'])} acceptor FN")
        print(f"  └─ Eliminated: {int(row['eliminated_donor_fp'])} donor FP, {int(row['eliminated_acceptor_fp'])} acceptor FP")
        print()
    
    # Save detailed summary if output file specified
    if output_file:
        summary_df = pd.DataFrame([global_stats])
        summary_df.to_csv(output_file, sep='\t', index=False)
        print(f"📁 Detailed summary saved to: {output_file}")
    
    return global_stats


def main():
    parser = argparse.ArgumentParser(description="Generate global FP/FN summary statistics")
    parser.add_argument("input_file", help="Input TSV file with gene-level FP/FN results")
    parser.add_argument("--output", "-o", help="Output file for detailed summary (optional)")
    
    args = parser.parse_args()
    
    if not Path(args.input_file).exists():
        print(f"❌ Error: Input file not found: {args.input_file}")
        return
    
    generate_global_summary(args.input_file, args.output)


if __name__ == "__main__":
    main() 