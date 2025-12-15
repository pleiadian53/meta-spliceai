#!/usr/bin/env python3
"""
Query a gene manifest to find information about genes in the training dataset.

This script helps you look up genes in your training dataset manifest.

Usage:
    python scripts/query_gene_manifest.py /path/to/gene_manifest.csv --gene STMN2
    python scripts/query_gene_manifest.py /path/to/gene_manifest.csv --gene-id ENSG00000104435
    python scripts/query_gene_manifest.py /path/to/gene_manifest.csv --list-genes
    python scripts/query_gene_manifest.py /path/to/gene_manifest.csv --stats
"""

import argparse
import sys
import pandas as pd
from pathlib import Path


def load_manifest(manifest_path: str) -> pd.DataFrame:
    """Load the gene manifest CSV file."""
    try:
        df = pd.read_csv(manifest_path)
        return df
    except Exception as e:
        print(f"❌ Error loading manifest: {e}", file=sys.stderr)
        sys.exit(1)


def query_by_gene_name(df: pd.DataFrame, gene_name: str) -> pd.DataFrame:
    """Query manifest by gene name."""
    return df[df['gene_name'].str.contains(gene_name, case=False, na=False)]


def query_by_gene_id(df: pd.DataFrame, gene_id: str) -> pd.DataFrame:
    """Query manifest by gene ID."""
    return df[df['gene_id'].str.contains(gene_id, case=False, na=False)]


def show_stats(df: pd.DataFrame):
    """Show statistics about the manifest."""
    print(f"📊 Manifest Statistics:")
    print(f"   • Total unique genes: {len(df):,}")
    print(f"   • Genes with names: {df['gene_name'].notna().sum():,}")
    print(f"   • Genes without names: {df['gene_name'].isna().sum():,}")
    print(f"   • Files containing genes: {df['file_index'].nunique():,}")
    
    if len(df) > 0:
        print(f"\n📁 Files in dataset:")
        file_counts = df.groupby('file_name').size().sort_values(ascending=False)
        for file_name, count in file_counts.head(10).items():
            print(f"   • {file_name}: {count:,} genes")
        
        if len(file_counts) > 10:
            print(f"   • ... and {len(file_counts) - 10} more files")


def list_genes(df: pd.DataFrame, limit: int = 20):
    """List genes in the manifest."""
    print(f"🧬 Genes in training dataset (showing first {limit}):")
    print()
    
    for _, row in df.head(limit).iterrows():
        gene_name = row['gene_name'] if pd.notna(row['gene_name']) else "N/A"
        print(f"   {row['global_index']:4d}. {gene_name} ({row['gene_id']})")
        print(f"        File: {row['file_name']}")
        print()
    
    if len(df) > limit:
        print(f"... and {len(df) - limit} more genes")


def main():
    parser = argparse.ArgumentParser(
        description="Query a gene manifest to find information about genes in the training dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "manifest_path",
        type=str,
        help="Path to the gene manifest CSV file",
    )
    parser.add_argument(
        "--gene", "-g",
        type=str,
        help="Search for genes by name (partial match, case-insensitive)",
    )
    parser.add_argument(
        "--gene-id", "-i",
        type=str,
        help="Search for genes by ID (partial match, case-insensitive)",
    )
    parser.add_argument(
        "--list-genes", "-l",
        action="store_true",
        help="List all genes in the manifest",
    )
    parser.add_argument(
        "--stats", "-s",
        action="store_true",
        help="Show statistics about the manifest",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Limit number of results shown (for --list-genes)",
    )
    
    args = parser.parse_args()
    
    # Load manifest
    df = load_manifest(args.manifest_path)
    
    if args.stats:
        show_stats(df)
        return
    
    if args.list_genes:
        list_genes(df, args.limit)
        return
    
    if args.gene:
        results = query_by_gene_name(df, args.gene)
        if len(results) > 0:
            print(f"🔍 Found {len(results)} gene(s) matching '{args.gene}':")
            print()
            for _, row in results.iterrows():
                gene_name = row['gene_name'] if pd.notna(row['gene_name']) else "N/A"
                print(f"   • {gene_name} ({row['gene_id']})")
                print(f"     Index: {row['global_index']}, File: {row['file_name']}")
                print()
        else:
            print(f"❌ No genes found matching '{args.gene}'")
        return
    
    if args.gene_id:
        results = query_by_gene_id(df, args.gene_id)
        if len(results) > 0:
            print(f"🔍 Found {len(results)} gene(s) matching '{args.gene_id}':")
            print()
            for _, row in results.iterrows():
                gene_name = row['gene_name'] if pd.notna(row['gene_name']) else "N/A"
                print(f"   • {gene_name} ({row['gene_id']})")
                print(f"     Index: {row['global_index']}, File: {row['file_name']}")
                print()
        else:
            print(f"❌ No genes found matching '{args.gene_id}'")
        return
    
    # If no specific query, show stats
    print("📋 Gene Manifest Query Tool")
    print("=" * 50)
    print()
    show_stats(df)
    print()
    print("💡 Usage examples:")
    print("   • python query_gene_manifest.py manifest.csv --gene STMN2")
    print("   • python query_gene_manifest.py manifest.csv --gene-id ENSG00000104435")
    print("   • python query_gene_manifest.py manifest.csv --list-genes")
    print("   • python query_gene_manifest.py manifest.csv --stats")


if __name__ == "__main__":
    main() 