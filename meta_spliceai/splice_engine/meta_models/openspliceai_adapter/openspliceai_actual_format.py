#!/usr/bin/env python3
"""
OpenSpliceAI Actual Format Demonstration

This script shows exactly what OpenSpliceAI produces and expects,
compared to your splice_sites.tsv format.
"""

import h5py
import numpy as np
import pandas as pd
from typing import Dict, List, Any

def demonstrate_openspliceai_format():
    """
    Show the exact format that OpenSpliceAI produces and expects.
    """
    
    print("="*80)
    print("OPENSPLICEAI ACTUAL FORMAT vs YOUR SPLICE_SITES.TSV")
    print("="*80)
    
    print("\n1. WHAT OPENSPLICEAI ACTUALLY PRODUCES:")
    print("-" * 50)
    
    # OpenSpliceAI creates HDF5 files with these exact datasets
    openspliceai_format = {
        'file_format': 'HDF5 (.h5)',
        'datasets': {
            'NAME': 'Gene IDs (e.g., ENSG00000228037)',
            'CHROM': 'Chromosome names (e.g., 1, 2, X, Y)',
            'STRAND': 'Strand orientation (+, -)',
            'TX_START': 'Transcription start position (string)',
            'TX_END': 'Transcription end position (string)',
            'SEQ': 'Full gene sequence (ATCG...)',
            'LABEL': 'Splice site labels as string (000111222000...)'
        },
        'label_encoding': {
            '0': 'neither (no splice site)',
            '1': 'acceptor site',
            '2': 'donor site'
        }
    }
    
    print("OpenSpliceAI HDF5 Structure:")
    print("File: datafile_train.h5 / datafile_test.h5")
    print("")
    for dataset, description in openspliceai_format['datasets'].items():
        print(f"  {dataset:10} : {description}")
    
    print("\nLabel Encoding in SEQ:")
    for label, meaning in openspliceai_format['label_encoding'].items():
        print(f"  '{label}' = {meaning}")
    
    print("\n2. YOUR CURRENT FORMAT (splice_sites.tsv):")
    print("-" * 50)
    
    your_format = {
        'file_format': 'TSV (Tab-separated values)',
        'columns': [
            'chrom', 'start', 'end', 'position', 'strand', 
            'site_type', 'gene_id', 'transcript_id'
        ],
        'site_type_values': ['donor', 'acceptor'],
        'example_row': {
            'chrom': '1',
            'start': '2581649',
            'end': '2581653', 
            'position': '2581651',
            'strand': '+',
            'site_type': 'donor',
            'gene_id': 'ENSG00000228037',
            'transcript_id': 'ENST00000424215'
        }
    }
    
    print("Your splice_sites.tsv Structure:")
    print("File: data/ensembl/splice_sites.tsv")
    print("")
    print("Columns:", " | ".join(your_format['columns']))
    print("")
    print("Example row:")
    for col, val in your_format['example_row'].items():
        print(f"  {col:15} : {val}")
    
    print("\n3. KEY DIFFERENCES:")
    print("-" * 50)
    
    differences = [
        {
            'aspect': 'Data Structure',
            'your_format': 'Individual splice site records (one per row)',
            'openspliceai': 'Full gene sequences with embedded splice labels'
        },
        {
            'aspect': 'File Format',
            'your_format': 'TSV (human-readable)',
            'openspliceai': 'HDF5 (binary, efficient for ML)'
        },
        {
            'aspect': 'Splice Site Representation',
            'your_format': 'Explicit coordinates (position, start, end)',
            'openspliceai': 'Labels embedded in sequence (position-indexed)'
        },
        {
            'aspect': 'Granularity',
            'your_format': 'Only splice sites (2.8M records)',
            'openspliceai': 'Every nucleotide in gene (millions per gene)'
        },
        {
            'aspect': 'Usage',
            'your_format': 'Direct analysis, feature extraction',
            'openspliceai': 'Deep learning model training/inference'
        }
    ]
    
    for diff in differences:
        print(f"\n{diff['aspect']}:")
        print(f"  Your format:  {diff['your_format']}")
        print(f"  OpenSpliceAI: {diff['openspliceai']}")
    
    print("\n4. WHAT IS THE EQUIVALENT OF YOUR splice_sites.tsv?")
    print("-" * 50)
    
    print("OpenSpliceAI does NOT produce a direct equivalent to your splice_sites.tsv.")
    print("Instead, it produces:")
    print("")
    print("A. HDF5 files with full gene sequences and labels:")
    print("   - datafile_train.h5")
    print("   - datafile_test.h5")
    print("")
    print("B. To extract splice sites (like your TSV), you would need to:")
    print("   1. Read the HDF5 file")
    print("   2. Parse each gene's SEQ and LABEL")
    print("   3. Find positions where LABEL = '1' (acceptor) or '2' (donor)")
    print("   4. Convert to genomic coordinates")
    print("   5. Create a TSV similar to yours")
    
    print("\n5. EXAMPLE OPENSPLICEAI DATA:")
    print("-" * 50)
    
    # Create a realistic example
    example_gene = {
        'NAME': 'ENSG00000228037',
        'CHROM': '1', 
        'STRAND': '+',
        'TX_START': '2581649',
        'TX_END': '2584126',
        'SEQ': 'ATCGATCG...GT...AG...ATCG',  # Simplified
        'LABEL': '00000000...2....1....0000'   # Simplified
    }
    
    print("Example gene in OpenSpliceAI format:")
    for key, value in example_gene.items():
        if key in ['SEQ', 'LABEL']:
            print(f"  {key:10} : {value} (truncated)")
        else:
            print(f"  {key:10} : {value}")
    
    print("\nWhere in the LABEL string:")
    print("  Position with '2' = donor site")
    print("  Position with '1' = acceptor site") 
    print("  Position with '0' = neither")
    
    return openspliceai_format, your_format, differences


def create_splice_sites_from_openspliceai_example():
    """
    Show how you would extract splice sites from OpenSpliceAI format
    to create something like your splice_sites.tsv
    """
    
    print("\n6. CONVERTING OPENSPLICEAI TO YOUR FORMAT:")
    print("-" * 50)
    
    print("Hypothetical conversion process:")
    print("")
    print("```python")
    print("def extract_splice_sites_from_openspliceai(h5_file):")
    print("    with h5py.File(h5_file, 'r') as f:")
    print("        names = f['NAME'][:]")
    print("        chroms = f['CHROM'][:]") 
    print("        strands = f['STRAND'][:]")
    print("        tx_starts = f['TX_START'][:]")
    print("        seqs = f['SEQ'][:]")
    print("        labels = f['LABEL'][:]")
    print("        ")
    print("    splice_sites = []")
    print("    for i in range(len(names)):")
    print("        gene_start = int(tx_starts[i])")
    print("        label_str = labels[i]")
    print("        ")
    print("        for pos, label in enumerate(label_str):")
    print("            if label in ['1', '2']:  # acceptor or donor")
    print("                genomic_pos = gene_start + pos")
    print("                site_type = 'acceptor' if label == '1' else 'donor'")
    print("                ")
    print("                splice_sites.append({")
    print("                    'chrom': chroms[i],")
    print("                    'position': genomic_pos,")
    print("                    'strand': strands[i],")
    print("                    'site_type': site_type,")
    print("                    'gene_id': names[i]")
    print("                })")
    print("    ")
    print("    return pd.DataFrame(splice_sites)")
    print("```")
    
    print("\nThis would produce a DataFrame similar to your splice_sites.tsv!")


def main():
    """Main demonstration."""
    
    openspliceai_format, your_format, differences = demonstrate_openspliceai_format()
    create_splice_sites_from_openspliceai_example()
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    print("\n✅ WHAT OPENSPLICEAI EXPECTS:")
    print("   Input:  GTF + FASTA files (same as your inputs)")
    print("   Output: HDF5 files with gene sequences and splice labels")
    print("")
    print("❌ WHAT OPENSPLICEAI DOES NOT PRODUCE:")
    print("   A direct equivalent to your splice_sites.tsv")
    print("")
    print("🔧 WHAT THE ADAPTER DOES:")
    print("   1. Runs OpenSpliceAI preprocessing (GTF + FASTA → HDF5)")
    print("   2. Extracts splice sites from HDF5 (similar to conversion above)")
    print("   3. Converts to your splice_sites.tsv format")
    print("   4. Handles coordinate system differences")
    print("")
    print("🎯 RESULT:")
    print("   You get the benefits of OpenSpliceAI preprocessing")
    print("   while maintaining compatibility with your existing workflow!")


if __name__ == "__main__":
    main()
