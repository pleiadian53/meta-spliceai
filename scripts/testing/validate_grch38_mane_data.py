#!/usr/bin/env python3
"""
Validate GRCh38 MANE Data Quality

This script performs comprehensive validation of the extracted GRCh38 MANE data:
1. Sequence length validation
2. Splice site consensus sequence analysis (GT-AG dinucleotides)
3. Statistical comparison with biological expectations
4. Comparison with GRCh37/Ensembl baseline

Based on the validation approach used for GRCh37/Ensembl data.

Usage:
    python scripts/testing/validate_grch38_mane_data.py
"""

import sys
from pathlib import Path
import polars as pl
import pandas as pd
from typing import Dict, List, Tuple
from collections import Counter
from pyfaidx import Fasta

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from meta_spliceai.system.genomic_resources import Registry


def validate_gene_sequences(registry: Registry) -> Dict:
    """Validate extracted gene sequences."""
    print("=" * 80)
    print("1. GENE SEQUENCE VALIDATION")
    print("=" * 80)
    print()
    
    results = {
        'total_genes': 0,
        'total_sequence_length': 0,
        'avg_gene_length': 0,
        'min_gene_length': 0,
        'max_gene_length': 0,
        'genes_per_chromosome': {},
        'validation_passed': False
    }
    
    # Load all sequence files
    data_dir = registry.data_dir
    sequence_files = list(data_dir.glob("gene_sequence_*.parquet"))
    
    if not sequence_files:
        print("❌ No sequence files found!")
        return results
    
    print(f"Found {len(sequence_files)} chromosome sequence files")
    print()
    
    all_genes = []
    for seq_file in sorted(sequence_files):
        df = pl.read_parquet(str(seq_file))
        all_genes.append(df)
        chrom = seq_file.stem.replace('gene_sequence_', '')
        results['genes_per_chromosome'][chrom] = df.height
        print(f"  {chrom}: {df.height:,} genes")
    
    # Combine all genes
    combined = pl.concat(all_genes)
    
    # Calculate statistics
    results['total_genes'] = combined.height
    
    # Add sequence length column
    combined = combined.with_columns([
        pl.col('sequence').str.len_bytes().alias('seq_length')
    ])
    
    results['total_sequence_length'] = combined['seq_length'].sum()
    results['avg_gene_length'] = combined['seq_length'].mean()
    results['min_gene_length'] = combined['seq_length'].min()
    results['max_gene_length'] = combined['seq_length'].max()
    
    print()
    print("Summary Statistics:")
    print(f"  Total genes: {results['total_genes']:,}")
    print(f"  Total sequence length: {results['total_sequence_length']:,} bp")
    print(f"  Average gene length: {results['avg_gene_length']:,.0f} bp")
    print(f"  Min gene length: {results['min_gene_length']:,} bp")
    print(f"  Max gene length: {results['max_gene_length']:,} bp")
    print()
    
    # Validation checks
    checks_passed = []
    
    # Check 1: Total genes reasonable (MANE has ~19K canonical genes)
    if 18000 <= results['total_genes'] <= 20000:
        print("✅ Total genes within expected range (18K-20K)")
        checks_passed.append(True)
    else:
        print(f"⚠️  Total genes ({results['total_genes']:,}) outside expected range")
        checks_passed.append(False)
    
    # Check 2: Average gene length reasonable (human genes ~27kb on average)
    if 10000 <= results['avg_gene_length'] <= 100000:
        print("✅ Average gene length reasonable")
        checks_passed.append(True)
    else:
        print(f"⚠️  Average gene length ({results['avg_gene_length']:,.0f}) unusual")
        checks_passed.append(False)
    
    # Check 3: No empty sequences
    empty_seqs = combined.filter(pl.col('seq_length') == 0).height
    if empty_seqs == 0:
        print("✅ No empty sequences found")
        checks_passed.append(True)
    else:
        print(f"❌ Found {empty_seqs} empty sequences!")
        checks_passed.append(False)
    
    # Check 4: All chromosomes present
    expected_chroms = [str(i) for i in range(1, 23)] + ['X', 'Y']
    missing_chroms = set(expected_chroms) - set(results['genes_per_chromosome'].keys())
    if not missing_chroms:
        print("✅ All 24 chromosomes present")
        checks_passed.append(True)
    else:
        print(f"❌ Missing chromosomes: {missing_chroms}")
        checks_passed.append(False)
    
    results['validation_passed'] = all(checks_passed)
    print()
    
    return results, combined


def extract_splice_site_consensus(
    splice_sites_df: pl.DataFrame,
    fasta_path: Path,
    site_type: str,
    max_sites: int = 10000
) -> Tuple[List[str], Dict]:
    """Extract consensus sequences around splice sites."""
    
    print(f"\nExtracting {site_type} consensus sequences...")
    
    # Load FASTA
    fasta = Fasta(str(fasta_path), as_raw=True, sequence_always_upper=True)
    
    # Filter by site type
    sites = splice_sites_df.filter(pl.col('splice_type') == site_type)
    
    # Sample if too many
    if sites.height > max_sites:
        sites = sites.sample(n=max_sites, seed=42)
    
    print(f"  Analyzing {sites.height:,} {site_type} sites")
    
    sequences = []
    stats = {
        'total_sites': sites.height,
        'extracted': 0,
        'failed': 0,
        'canonical_count': 0,
        'non_canonical_count': 0
    }
    
    for row in sites.iter_rows(named=True):
        chrom = str(row['chrom'])
        pos = int(row['position'])
        strand = row['strand']
        
        try:
            if site_type == 'donor':
                # Extract 3 exonic + 6 intronic bases (9-mer)
                if strand == '+':
                    start = pos - 3 - 1  # 0-based
                    end = pos + 6 - 1
                    seq = str(fasta[chrom][start:end])
                else:
                    start = pos - 6
                    end = pos + 3
                    seq = str(fasta[chrom][start:end])
                    seq = reverse_complement(seq)
                
                # Check for canonical GT (at positions 3-4 in 0-based)
                if len(seq) >= 5 and seq[3:5] == 'GT':
                    stats['canonical_count'] += 1
                else:
                    stats['non_canonical_count'] += 1
                    
            elif site_type == 'acceptor':
                # Extract 20 intronic + 2 (AG) + 3 exonic bases (25-mer)
                if strand == '+':
                    start = (pos - 2) - 20
                    end = (pos - 1) + 3 + 1
                    seq = str(fasta[chrom][start:end])
                else:
                    start = pos - 3 - 1
                    end = pos + 20 + 1
                    seq = str(fasta[chrom][start:end])
                    seq = reverse_complement(seq)
                
                # Check for canonical AG (at positions 20-21 in 0-based)
                if len(seq) >= 22 and seq[20:22] == 'AG':
                    stats['canonical_count'] += 1
                else:
                    stats['non_canonical_count'] += 1
            
            sequences.append(seq)
            stats['extracted'] += 1
            
        except (KeyError, IndexError) as e:
            stats['failed'] += 1
            continue
    
    return sequences, stats


def reverse_complement(seq: str) -> str:
    """Reverse complement a DNA sequence."""
    complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'N': 'N'}
    return ''.join(complement.get(base, 'N') for base in reversed(seq))


def analyze_consensus_sequences(sequences: List[str], site_type: str) -> Dict:
    """Analyze consensus sequences and calculate statistics."""
    
    if not sequences:
        return {}
    
    # Calculate position frequency matrix
    seq_length = len(sequences[0])
    pfm = [{} for _ in range(seq_length)]
    
    for seq in sequences:
        if len(seq) != seq_length:
            continue
        for i, base in enumerate(seq):
            pfm[i][base] = pfm[i].get(base, 0) + 1
    
    # Calculate consensus and percentages
    consensus = []
    for pos_counts in pfm:
        if not pos_counts:
            consensus.append('N')
        else:
            consensus.append(max(pos_counts, key=pos_counts.get))
    
    consensus_seq = ''.join(consensus)
    
    # Calculate dinucleotide percentages
    results = {
        'consensus_sequence': consensus_seq,
        'num_sequences': len(sequences),
        'pfm': pfm
    }
    
    if site_type == 'donor':
        # GT at positions 3-4 (0-based)
        gt_count = sum(1 for seq in sequences if len(seq) > 4 and seq[3:5] == 'GT')
        gc_count = sum(1 for seq in sequences if len(seq) > 4 and seq[3:5] == 'GC')
        
        results['GT_percentage'] = (gt_count / len(sequences)) * 100
        results['GC_percentage'] = (gc_count / len(sequences)) * 100
        results['canonical_dinucleotide'] = 'GT'
        
    elif site_type == 'acceptor':
        # AG at positions 20-21 (0-based)
        ag_count = sum(1 for seq in sequences if len(seq) > 21 and seq[20:22] == 'AG')
        
        results['AG_percentage'] = (ag_count / len(sequences)) * 100
        results['canonical_dinucleotide'] = 'AG'
    
    return results


def validate_splice_sites(registry: Registry) -> Dict:
    """Validate splice site consensus sequences."""
    print("=" * 80)
    print("2. SPLICE SITE CONSENSUS VALIDATION")
    print("=" * 80)
    print()
    
    # Load splice sites
    splice_sites_file = registry.data_dir / "splice_sites_enhanced.tsv"
    if not splice_sites_file.exists():
        print(f"❌ Splice sites file not found: {splice_sites_file}")
        return {}
    
    splice_sites_df = pl.read_csv(
        str(splice_sites_file),
        separator='\t',
        schema_overrides={'chrom': pl.Utf8}
    )
    
    print(f"Loaded {splice_sites_df.height:,} splice sites")
    print()
    
    # Standardize schema (site_type -> splice_type if needed)
    if 'site_type' in splice_sites_df.columns and 'splice_type' not in splice_sites_df.columns:
        splice_sites_df = splice_sites_df.rename({'site_type': 'splice_type'})
    
    # Strip "chr" prefix from chromosome names to match FASTA
    splice_sites_df = splice_sites_df.with_columns([
        pl.col('chrom').str.replace('^chr', '').alias('chrom')
    ])
    
    # Count by type
    donor_count = splice_sites_df.filter(pl.col('splice_type') == 'donor').height
    acceptor_count = splice_sites_df.filter(pl.col('splice_type') == 'acceptor').height
    
    print(f"  Donor sites: {donor_count:,}")
    print(f"  Acceptor sites: {acceptor_count:,}")
    print()
    
    # Get FASTA path
    fasta_path = registry.get_fasta_path()
    
    results = {}
    
    # Analyze donors
    donor_seqs, donor_stats = extract_splice_site_consensus(
        splice_sites_df, fasta_path, 'donor', max_sites=10000
    )
    donor_analysis = analyze_consensus_sequences(donor_seqs, 'donor')
    results['donor'] = {**donor_stats, **donor_analysis}
    
    # Analyze acceptors
    acceptor_seqs, acceptor_stats = extract_splice_site_consensus(
        splice_sites_df, fasta_path, 'acceptor', max_sites=10000
    )
    acceptor_analysis = analyze_consensus_sequences(acceptor_seqs, 'acceptor')
    results['acceptor'] = {**acceptor_stats, **acceptor_analysis}
    
    # Print results
    print()
    print("=" * 80)
    print("CONSENSUS SEQUENCE ANALYSIS")
    print("=" * 80)
    print()
    
    if 'consensus_sequence' in results['donor']:
        print("Donor Sites (9-mer: 3 exonic + 6 intronic):")
        print(f"  Consensus: {results['donor']['consensus_sequence']}")
        print(f"  GT percentage: {results['donor']['GT_percentage']:.2f}%")
        print(f"  GC percentage: {results['donor']['GC_percentage']:.2f}%")
        print()
    else:
        print("⚠️  Donor consensus analysis failed - no sequences extracted")
        print(f"  Stats: {results['donor']}")
        print()
    
    if 'consensus_sequence' in results['acceptor']:
        print("Acceptor Sites (25-mer: 20 intronic + AG + 3 exonic):")
        print(f"  Consensus: {results['acceptor']['consensus_sequence']}")
        print(f"  AG percentage: {results['acceptor']['AG_percentage']:.2f}%")
        print()
    else:
        print("⚠️  Acceptor consensus analysis failed - no sequences extracted")
        print(f"  Stats: {results['acceptor']}")
        print()
    
    # Validation checks
    print("=" * 80)
    print("BIOLOGICAL VALIDATION")
    print("=" * 80)
    print()
    
    checks_passed = []
    
    # Expected: ~98.5% GT donors (literature: 98.5-99.7%)
    if 'GT_percentage' in results['donor']:
        if 97.0 <= results['donor']['GT_percentage'] <= 100.0:
            print(f"✅ Donor GT percentage ({results['donor']['GT_percentage']:.2f}%) within expected range (97-100%)")
            checks_passed.append(True)
        else:
            print(f"⚠️  Donor GT percentage ({results['donor']['GT_percentage']:.2f}%) outside expected range")
            checks_passed.append(False)
    else:
        print("⚠️  Donor GT percentage not available")
        checks_passed.append(False)
    
    # Expected: ~1% GC donors (GC-AG introns)
    if 'GC_percentage' in results['donor']:
        if 0.1 <= results['donor']['GC_percentage'] <= 3.0:
            print(f"✅ Donor GC percentage ({results['donor']['GC_percentage']:.2f}%) within expected range (0.1-3%)")
            checks_passed.append(True)
        else:
            print(f"⚠️  Donor GC percentage ({results['donor']['GC_percentage']:.2f}%) outside expected range")
            checks_passed.append(False)
    else:
        print("⚠️  Donor GC percentage not available")
        checks_passed.append(False)
    
    # Expected: ~99.6% AG acceptors (literature: 99.6-99.8%)
    if 'AG_percentage' in results['acceptor']:
        if 98.0 <= results['acceptor']['AG_percentage'] <= 100.0:
            print(f"✅ Acceptor AG percentage ({results['acceptor']['AG_percentage']:.2f}%) within expected range (98-100%)")
            checks_passed.append(True)
        else:
            print(f"⚠️  Acceptor AG percentage ({results['acceptor']['AG_percentage']:.2f}%) outside expected range")
            checks_passed.append(False)
    else:
        print("⚠️  Acceptor AG percentage not available")
        checks_passed.append(False)
    
    results['validation_passed'] = all(checks_passed)
    print()
    
    return results


def compare_with_grch37(mane_results: Dict) -> None:
    """Compare MANE results with GRCh37/Ensembl baseline."""
    print("=" * 80)
    print("3. COMPARISON WITH GRCh37/ENSEMBL BASELINE")
    print("=" * 80)
    print()
    
    # Expected GRCh37/Ensembl values (from previous validation)
    grch37_baseline = {
        'total_genes': 35306,
        'total_splice_sites': 1998526,
        'donor_gt_pct': 99.7,
        'donor_gc_pct': 0.8,
        'acceptor_ag_pct': 99.8
    }
    
    mane_values = {
        'total_genes': mane_results.get('gene_sequences', {}).get('total_genes', 0),
        'total_splice_sites': 369918,  # From derivation
        'donor_gt_pct': mane_results.get('splice_sites', {}).get('donor', {}).get('GT_percentage', 0),
        'donor_gc_pct': mane_results.get('splice_sites', {}).get('donor', {}).get('GC_percentage', 0),
        'acceptor_ag_pct': mane_results.get('splice_sites', {}).get('acceptor', {}).get('AG_percentage', 0)
    }
    
    print("| Metric | GRCh37/Ensembl | GRCh38/MANE | Ratio |")
    print("|--------|----------------|-------------|-------|")
    
    for key, grch37_val in grch37_baseline.items():
        mane_val = mane_values.get(key, 0)
        if grch37_val > 0:
            ratio = mane_val / grch37_val
            print(f"| {key} | {grch37_val:,.1f} | {mane_val:,.1f} | {ratio:.2f}x |")
    
    print()
    print("Key Insights:")
    print("  • MANE has ~54% of Ensembl genes (canonical vs all isoforms)")
    print("  • MANE has ~18% of Ensembl splice sites (canonical transcripts only)")
    print("  • Consensus sequences should be similar (both human genome)")
    print("  • GT/AG percentages should be comparable (same biology)")
    print()


def main():
    print("=" * 80)
    print("GRCh38 MANE DATA VALIDATION")
    print("=" * 80)
    print()
    
    # Initialize registry
    registry = Registry(build='GRCh38_MANE', release='1.3')
    
    print(f"Data directory: {registry.data_dir}")
    print(f"GTF: {registry.get_gtf_path()}")
    print(f"FASTA: {registry.get_fasta_path()}")
    print()
    
    results = {}
    
    # 1. Validate gene sequences
    gene_results, gene_df = validate_gene_sequences(registry)
    results['gene_sequences'] = gene_results
    
    # 2. Validate splice sites
    splice_results = validate_splice_sites(registry)
    results['splice_sites'] = splice_results
    
    # 3. Compare with GRCh37
    compare_with_grch37(results)
    
    # Final summary
    print("=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    print()
    
    gene_passed = results['gene_sequences'].get('validation_passed', False)
    splice_passed = results['splice_sites'].get('validation_passed', False)
    
    print(f"Gene Sequences: {'✅ PASS' if gene_passed else '❌ FAIL'}")
    print(f"Splice Sites: {'✅ PASS' if splice_passed else '❌ FAIL'}")
    print()
    
    if gene_passed and splice_passed:
        print("=" * 80)
        print("✅ ALL VALIDATIONS PASSED!")
        print("=" * 80)
        print()
        print("GRCh38 MANE data is ready for OpenSpliceAI predictions.")
        print()
        sys.exit(0)
    else:
        print("=" * 80)
        print("⚠️  SOME VALIDATIONS FAILED")
        print("=" * 80)
        print()
        print("Please review the results above and address any issues.")
        print()
        sys.exit(1)


if __name__ == '__main__':
    main()

