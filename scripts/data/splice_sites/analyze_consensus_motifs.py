#!/usr/bin/env python3
"""
Analyze consensus motifs at splice sites.

This script extracts and analyzes the extended consensus sequences at donor 
and acceptor sites to validate known splice site motifs:

- Donor sites: 9-mer "MAG|GTRAGT" (3 exon + 6 intron)
  - M = A or C
  - R = A or G  
  - | = exon-intron boundary
  
- Acceptor sites: 23-mer with polypyrimidine tract
  - Extended form: (Y)n YYYYYYYYYYYYYYAG (polypyrimidine tract + AG)
  - Shorter analysis: YNY|AG (3 intron + 2 intron, showing branch point region)
  - Y = C or T (pyrimidines)
  - N = any nucleotide
  - | = intron-exon boundary

References:
- Shapiro & Senapathy (1987). RNA splice junctions of different classes of eukaryotes.
- Burge & Karlin (1997). Prediction of complete gene structures in human genomic DNA.
"""

import sys
import numpy as np
import polars as pl
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, List
from collections import Counter
from pyfaidx import Fasta

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from meta_spliceai.system.genomic_resources import Registry


# IUPAC nucleotide codes
IUPAC_CODES = {
    'A': ['A'],
    'C': ['C'],
    'G': ['G'],
    'T': ['T'],
    'R': ['A', 'G'],  # puRine
    'Y': ['C', 'T'],  # pYrimidine
    'M': ['A', 'C'],  # aMino
    'K': ['G', 'T'],  # Keto
    'S': ['G', 'C'],  # Strong
    'W': ['A', 'T'],  # Weak
    'H': ['A', 'C', 'T'],  # not G
    'B': ['C', 'G', 'T'],  # not A
    'V': ['A', 'C', 'G'],  # not T
    'D': ['A', 'G', 'T'],  # not C
    'N': ['A', 'C', 'G', 'T']  # aNy
}


def reverse_complement(seq: str) -> str:
    """Reverse complement a DNA sequence."""
    complement = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G', 'N': 'N'}
    return ''.join(complement.get(base, 'N') for base in reversed(seq))


def extract_donor_motif(
    fasta: Fasta,
    chrom: str,
    position: int,
    strand: str,
    exon_bases: int = 3,
    intron_bases: int = 6
) -> str:
    """
    Extract donor site motif: MAG|GTRAGT
    
    Parameters
    ----------
    fasta : Fasta
        Indexed FASTA file
    chrom : str
        Chromosome name
    position : int
        1-based position of the splice site (first base of intron)
    strand : str
        + or -
    exon_bases : int
        Number of bases to extract from exon (before boundary)
    intron_bases : int
        Number of bases to extract from intron (after boundary)
        
    Returns
    -------
    str
        Sequence motif (e.g., "CAGGTATGT" for canonical donor)
    """
    # Get chromosome sequence
    try:
        chrom_seq = fasta[chrom]
    except KeyError:
        if chrom.startswith('chr'):
            chrom_seq = fasta[chrom[3:]]
        else:
            chrom_seq = fasta[f'chr{chrom}']
    
    if strand == '+':
        # Positive strand
        # Position is first base of intron (1-based)
        # Extract: [position - exon_bases : position + intron_bases]
        start = position - exon_bases - 1  # Convert to 0-based
        end = position + intron_bases - 1
        seq = chrom_seq[start:end].seq.upper()
    else:
        # Negative strand
        # Position is genomically at exon start, but in transcription it's intron start
        # Extract and reverse complement
        start = position - intron_bases
        end = position + exon_bases
        seq = chrom_seq[start:end].seq.upper()
        seq = reverse_complement(seq)
    
    return seq


def extract_acceptor_motif(
    fasta: Fasta,
    chrom: str,
    position: int,
    strand: str,
    intron_bases: int = 20,
    exon_bases: int = 3
) -> str:
    """
    Extract acceptor site motif with polypyrimidine tract.
    
    For acceptor sites:
    - Position is the last base of the intron (the G in AG)
    - We want: [intron_bases before AG][AG dinucleotide][exon_bases]
    
    Parameters
    ----------
    fasta : Fasta
        Indexed FASTA file
    chrom : str
        Chromosome name
    position : int
        1-based position of the splice site (last base of intron, the G in AG)
    strand : str
        + or -
    intron_bases : int
        Number of bases to extract from intron (before AG)
    exon_bases : int
        Number of bases to extract from exon (after boundary)
        
    Returns
    -------
    str
        Sequence motif: [intron_bases][A][G][exon_bases]
    """
    # Get chromosome sequence
    try:
        chrom_seq = fasta[chrom]
    except KeyError:
        if chrom.startswith('chr'):
            chrom_seq = fasta[chrom[3:]]
        else:
            chrom_seq = fasta[f'chr{chrom}']
    
    if strand == '+':
        # Positive strand
        # IMPORTANT: position is the FIRST BASE OF THE EXON (1-based), not the last base of intron!
        # Based on dinucleotide extraction: seq = chrom_seq[position-2:position]
        # So AG is at 1-based [position-1, position-0] = 0-based [position-2, position-1]
        # We want: [intron_bases][A][G][exon_bases]
        # Total length: intron_bases + 2 (AG) + exon_bases = 25 bases
        
        # AG is at 0-based [position-2, position-1]
        # Start 20 bases before A
        start = (position - 2) - intron_bases
        # End after G + exon_bases (Python slicing is exclusive, so +1)
        end = (position - 1) + exon_bases + 1
        seq = chrom_seq[start:end].seq.upper()
    else:
        # Negative strand
        # Position still points to first base of exon in genomic coords
        # Based on dinucleotide extraction: seq = chrom_seq[position-1:position+1] then RC
        # So in genomic coords, the 2-base boundary is at [position-1, position] (CT)
        # After RC, this becomes AG in pre-mRNA 5'->3'
        
        # For pre-mRNA we want: [intron_20][A][G][exon_3]
        # In genomic (before RC): [exon_3][C][T][intron_20]
        # So extract: [position - exon_3 - 1, position + intron_20 + 1]
        start = position - exon_bases - 1
        end = position + intron_bases + 1
        seq = chrom_seq[start:end].seq.upper()
        seq = reverse_complement(seq)
    
    return seq


def matches_consensus(seq: str, consensus: str) -> bool:
    """
    Check if sequence matches IUPAC consensus.
    
    Parameters
    ----------
    seq : str
        Observed sequence
    consensus : str
        Consensus sequence with IUPAC codes
        
    Returns
    -------
    bool
        True if sequence matches consensus
    """
    if len(seq) != len(consensus):
        return False
    
    for obs, cons in zip(seq, consensus):
        if obs not in IUPAC_CODES.get(cons, []):
            return False
    
    return True


def calculate_position_frequencies(sequences: List[str]) -> pd.DataFrame:
    """
    Calculate nucleotide frequencies at each position.
    
    Parameters
    ----------
    sequences : List[str]
        List of sequences (all same length)
        
    Returns
    -------
    pd.DataFrame
        Frequency matrix (positions x nucleotides)
    """
    if not sequences:
        return pd.DataFrame()
    
    seq_len = len(sequences[0])
    counts = {pos: Counter() for pos in range(seq_len)}
    
    for seq in sequences:
        if len(seq) == seq_len:  # Skip sequences with wrong length
            for pos, base in enumerate(seq):
                counts[pos][base] += 1
    
    # Convert to dataframe
    bases = ['A', 'C', 'G', 'T']
    data = []
    for pos in range(seq_len):
        total = sum(counts[pos].values())
        if total > 0:
            row = {base: counts[pos].get(base, 0) / total for base in bases}
            row['position'] = pos
            data.append(row)
    
    df = pd.DataFrame(data)
    if not df.empty:
        df = df[['position', 'A', 'C', 'G', 'T']]
    
    return df


def derive_consensus_from_frequencies(freq_df: pd.DataFrame, threshold: float = 0.5) -> str:
    """
    Derive IUPAC consensus sequence from frequency matrix.
    
    Parameters
    ----------
    freq_df : pd.DataFrame
        Frequency matrix from calculate_position_frequencies
    threshold : float
        Minimum frequency to include a base in consensus
        
    Returns
    -------
    str
        IUPAC consensus sequence
    """
    consensus = []
    
    for _, row in freq_df.iterrows():
        bases_above_threshold = []
        for base in ['A', 'C', 'G', 'T']:
            if row[base] >= threshold:
                bases_above_threshold.append(base)
        
        # Convert to IUPAC code
        bases_set = set(bases_above_threshold)
        
        # Find matching IUPAC code
        iupac = 'N'
        for code, allowed in IUPAC_CODES.items():
            if set(allowed) == bases_set:
                iupac = code
                break
        
        consensus.append(iupac)
    
    return ''.join(consensus)


def analyze_motifs(
    splice_sites_file: Path,
    fasta_file: Path,
    sample_size: int = None,
    verbose: bool = True
) -> Dict:
    """
    Analyze extended consensus motifs at splice sites.
    
    Parameters
    ----------
    splice_sites_file : Path
        Path to splice sites TSV file
    fasta_file : Path
        Path to FASTA file
    sample_size : int, optional
        If provided, analyze only a random sample
    verbose : bool
        Print progress information
        
    Returns
    -------
    dict
        Analysis results
    """
    print("=" * 80)
    print("CONSENSUS MOTIF ANALYSIS")
    print("=" * 80)
    
    # Load splice sites
    print(f"\n📂 Loading splice sites from: {splice_sites_file}")
    df = pl.read_csv(
        splice_sites_file,
        separator='\t',
        schema_overrides={'chrom': pl.Utf8}
    )
    
    print(f"   Total splice sites: {len(df):,}")
    
    # Sample if requested
    if sample_size:
        print(f"   Sampling {sample_size:,} sites for analysis...")
        df = df.sample(n=sample_size, seed=42)
    
    # Load FASTA
    print(f"\n📂 Loading FASTA file: {fasta_file}")
    fasta = Fasta(str(fasta_file))
    
    # Separate donor and acceptor sites
    donor_df = df.filter(pl.col('site_type') == 'donor')
    acceptor_df = df.filter(pl.col('site_type') == 'acceptor')
    
    print(f"\n   Donor sites: {len(donor_df):,}")
    print(f"   Acceptor sites: {len(acceptor_df):,}")
    
    # ========================================================================
    # DONOR SITE ANALYSIS
    # ========================================================================
    print(f"\n{'=' * 80}")
    print("DONOR SITE ANALYSIS: MAG|GTRAGT (9-mer)")
    print("=" * 80)
    
    print("\n🔄 Extracting donor motifs...")
    donor_motifs = []
    
    for i, row in enumerate(donor_df.iter_rows(named=True)):
        if verbose and (i + 1) % 50000 == 0:
            print(f"   Processed {i+1:,} / {len(donor_df):,} donor sites...")
        
        try:
            motif = extract_donor_motif(
                fasta,
                row['chrom'],
                row['position'],
                row['strand'],
                exon_bases=3,
                intron_bases=6
            )
            if len(motif) == 9:  # Sanity check
                donor_motifs.append(motif)
        except Exception as e:
            pass
    
    print(f"   ✅ Extracted {len(donor_motifs):,} donor motifs")
    
    # Calculate frequencies
    print("\n📊 Calculating position-specific frequencies...")
    donor_freq = calculate_position_frequencies(donor_motifs)
    
    print("\nDonor Motif Frequencies (9-mer: positions -3 to +6):")
    print("Position: -3  -2  -1  | +1  +2  +3  +4  +5  +6")
    print("          Exon        | Intron")
    print("-" * 60)
    
    # Format output
    for _, row in donor_freq.iterrows():
        pos = int(row['position'])
        label = f"{pos-3:+3d}" if pos < 3 else f"{pos-2:+3d}"
        
        # Find dominant base
        max_base = max(['A', 'C', 'G', 'T'], key=lambda b: row[b])
        max_freq = row[max_base]
        
        freq_str = f"A:{row['A']:.2f} C:{row['C']:.2f} G:{row['G']:.2f} T:{row['T']:.2f}"
        print(f"{label}: {freq_str}  → {max_base} ({max_freq:.1%})")
        
        if pos == 2:  # Boundary
            print("          " + "-" * 50)
    
    # Derive consensus
    observed_consensus = derive_consensus_from_frequencies(donor_freq, threshold=0.5)
    print(f"\nObserved Consensus (≥50% frequency): {observed_consensus[:3]}|{observed_consensus[3:]}")
    print(f"Expected Consensus:                   MAG|GTRAGT")
    
    # Check matches to expected
    expected_donor = "MAGGTRAGT"
    matches = sum(1 for motif in donor_motifs if matches_consensus(motif, expected_donor))
    print(f"\nMotifs matching MAG|GTRAGT: {matches:,} / {len(donor_motifs):,} ({100*matches/len(donor_motifs):.2f}%)")
    
    # Canonical GT at +1,+2
    gt_motifs = [m for m in donor_motifs if m[3:5] == 'GT']
    print(f"Motifs with GT at +1,+2:    {len(gt_motifs):,} / {len(donor_motifs):,} ({100*len(gt_motifs)/len(donor_motifs):.2f}%)")
    
    # GC at +1,+2 (non-canonical)
    gc_motifs = [m for m in donor_motifs if m[3:5] == 'GC']
    print(f"Motifs with GC at +1,+2:    {len(gc_motifs):,} / {len(donor_motifs):,} ({100*len(gc_motifs)/len(donor_motifs):.2f}%)")
    
    # ========================================================================
    # ACCEPTOR SITE ANALYSIS
    # ========================================================================
    print(f"\n{'=' * 80}")
    print("ACCEPTOR SITE ANALYSIS: Polypyrimidine Tract + AG")
    print("=" * 80)
    
    print("\n🔄 Extracting acceptor motifs...")
    acceptor_motifs = []
    
    for i, row in enumerate(acceptor_df.iter_rows(named=True)):
        if verbose and (i + 1) % 50000 == 0:
            print(f"   Processed {i+1:,} / {len(acceptor_df):,} acceptor sites...")
        
        try:
            motif = extract_acceptor_motif(
                fasta,
                row['chrom'],
                row['position'],
                row['strand'],
                intron_bases=20,  # 20 bases before AG
                exon_bases=3      # 3 bases after AG
            )
            if len(motif) == 25:  # 20 + 2 (AG) + 3
                acceptor_motifs.append(motif)
        except Exception as e:
            pass
    
    print(f"   ✅ Extracted {len(acceptor_motifs):,} acceptor motifs")
    
    # Calculate frequencies
    print("\n📊 Calculating position-specific frequencies...")
    acceptor_freq = calculate_position_frequencies(acceptor_motifs)
    
    print("\nAcceptor Motif Frequencies (25-mer):")
    print("Positions -20 to -1 (intron), +1 to +2 (AG), +3 to +5 (exon)")
    print("-" * 80)
    
    # Show key positions
    key_positions = list(range(0, 5)) + [18, 19] + list(range(20, 25))
    
    for pos in key_positions:
        if pos >= len(acceptor_freq):
            continue
        
        row = acceptor_freq.iloc[pos]
        
        # Label positions: -20 to -1 (intron), then A, G, +1, +2, +3 (exon)
        if pos < 20:
            label = f"{pos-20:+3d}"
        elif pos == 20:
            label = "  A"  # A in AG
        elif pos == 21:
            label = "  G"  # G in AG
        else:
            label = f" {pos-21:+2d}"  # Exon positions
        
        max_base = max(['A', 'C', 'G', 'T'], key=lambda b: row[b])
        max_freq = row[max_base]
        
        freq_str = f"A:{row['A']:.2f} C:{row['C']:.2f} G:{row['G']:.2f} T:{row['T']:.2f}"
        print(f"{label}: {freq_str}  → {max_base} ({max_freq:.1%})")
        
        if pos == 4:
            print("     ...")
        if pos == 19:
            print("          " + "-" * 50 + " (intron|exon boundary)")
    
    # Polypyrimidine content
    print("\n📊 Polypyrimidine Tract Analysis (positions -20 to -3):")
    pyrimidine_content = []
    for i in range(18):  # Positions -20 to -3
        row = acceptor_freq.iloc[i]
        py_content = row['C'] + row['T']
        pyrimidine_content.append(py_content)
    
    mean_py = np.mean(pyrimidine_content)
    print(f"   Average pyrimidine (C+T) content: {mean_py:.1%}")
    print(f"   Range: {min(pyrimidine_content):.1%} to {max(pyrimidine_content):.1%}")
    
    # AG dinucleotide (should be at positions 20 and 21)
    ag_count = sum(1 for m in acceptor_motifs if len(m) > 21 and m[20:22] == 'AG')
    print(f"\n📊 AG Dinucleotide at boundary:")
    print(f"   Motifs with AG at positions 20-21: {ag_count:,} / {len(acceptor_motifs):,} ({100*ag_count/len(acceptor_motifs):.2f}%)")
    
    # YAG motif (last intron base + AG)
    print(f"\n📊 YAG Motif Analysis (positions -1 + AG):")
    yag_motifs = [m[19:22] for m in acceptor_motifs if len(m) > 21]
    yag_counter = Counter(yag_motifs)
    
    print("   Top 10 YAG motifs:")
    for motif, count in yag_counter.most_common(10):
        pct = 100 * count / len(acceptor_motifs)
        y_status = "✓" if motif[0] in ['C', 'T'] else "✗"
        ag_status = "✓" if motif[1:3] == 'AG' else "✗"
        print(f"     {motif}: {count:,} ({pct:.2f}%) - Y:{y_status} AG:{ag_status}")
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print("=" * 80)
    
    print("\n✅ Donor Sites (MAG|GTRAGT):")
    print(f"   - Canonical GT at +1,+2: {100*len(gt_motifs)/len(donor_motifs):.2f}%")
    print(f"   - Non-canonical GC at +1,+2: {100*len(gc_motifs)/len(donor_motifs):.2f}%")
    print(f"   - Full MAG|GTRAGT match: {100*matches/len(donor_motifs):.2f}%")
    
    print("\n✅ Acceptor Sites:")
    print(f"   - Average polypyrimidine content: {mean_py:.1%}")
    print(f"   - AG dinucleotide at boundary: {100*ag_count/len(acceptor_motifs):.2f}%")
    
    results = {
        'donor': {
            'motifs': donor_motifs,
            'frequencies': donor_freq,
            'consensus': observed_consensus[:3] + '|' + observed_consensus[3:],
            'gt_percentage': 100*len(gt_motifs)/len(donor_motifs),
            'gc_percentage': 100*len(gc_motifs)/len(donor_motifs)
        },
        'acceptor': {
            'motifs': acceptor_motifs,
            'frequencies': acceptor_freq,
            'pyrimidine_content': mean_py,
            'ag_percentage': 100*ag_count/len(acceptor_motifs)
        }
    }
    
    return results


def main():
    """Main analysis function with CLI argument parsing."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Analyze extended consensus motifs at splice sites",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze all splice sites (full dataset)
  %(prog)s --full
  
  # Analyze 50,000 randomly sampled sites (faster)
  %(prog)s --sample 50000
  
  # Analyze specific chromosomes only
  %(prog)s --full --chromosomes 1 2 X Y
  
  # Save results to custom location
  %(prog)s --full --output /path/to/output.txt
  
  # Use custom input files
  %(prog)s --splice-sites /path/to/splice_sites.tsv --fasta /path/to/genome.fa
        """
    )
    
    # Input files
    parser.add_argument(
        "--splice-sites",
        type=Path,
        help="Path to splice sites TSV file (default: auto-detect from config)"
    )
    parser.add_argument(
        "--fasta",
        type=Path,
        help="Path to reference genome FASTA file (default: from genomic_resources config)"
    )
    
    # Sampling options
    sample_group = parser.add_mutually_exclusive_group()
    sample_group.add_argument(
        "--sample",
        type=int,
        metavar="N",
        help="Analyze N randomly sampled splice sites (default: 50000)"
    )
    sample_group.add_argument(
        "--full",
        action="store_true",
        help="Analyze all splice sites (no sampling)"
    )
    
    # Filtering options
    parser.add_argument(
        "--chromosomes",
        nargs="+",
        metavar="CHR",
        help="Analyze specific chromosomes only (e.g., 1 2 X Y)"
    )
    parser.add_argument(
        "--site-types",
        nargs="+",
        choices=["donor", "acceptor"],
        default=["donor", "acceptor"],
        help="Site types to analyze (default: both donor and acceptor)"
    )
    
    # Output options
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        metavar="FILE",
        help="Save output to file (default: print to stdout)"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose progress output"
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress all output except errors"
    )
    
    args = parser.parse_args()
    
    # Setup paths
    registry = Registry()
    
    if args.splice_sites:
        splice_sites_file = args.splice_sites
    else:
        splice_sites_file = registry.cfg.data_root / "splice_sites_enhanced.tsv"
        if not splice_sites_file.exists():
            splice_sites_file = registry.cfg.data_root / "splice_sites.tsv"
    
    if args.fasta:
        fasta_file = args.fasta
    else:
        fasta_file = registry.get_fasta_path(validate=True)
    
    # Validate input files
    if not splice_sites_file.exists():
        print(f"❌ Error: Splice sites file not found: {splice_sites_file}", file=sys.stderr)
        return 1
    
    if not fasta_file.exists():
        print(f"❌ Error: FASTA file not found: {fasta_file}", file=sys.stderr)
        return 1
    
    # Determine sample size
    if args.full:
        sample_size = None
    elif args.sample:
        sample_size = args.sample
    else:
        sample_size = 50000  # Default
    
    # Print configuration (unless quiet)
    if not args.quiet:
        print(f"Splice sites file: {splice_sites_file}")
        print(f"FASTA file: {fasta_file}")
        if sample_size:
            print(f"Sample size: {sample_size:,}")
        else:
            print(f"Sample size: Full dataset")
        if args.chromosomes:
            print(f"Chromosomes: {', '.join(args.chromosomes)}")
        if len(args.site_types) < 2:
            print(f"Site types: {', '.join(args.site_types)}")
    
    # Setup output redirection if requested
    original_stdout = sys.stdout
    if args.output:
        output_file = open(args.output, 'w')
        sys.stdout = output_file
    
    try:
        # Run analysis
        results = analyze_motifs(
            splice_sites_file,
            fasta_file,
            sample_size=sample_size,
            verbose=(args.verbose or not args.quiet)
        )
        
        print(f"\n{'=' * 80}")
        print("ANALYSIS COMPLETE")
        print("=" * 80)
        
        return 0
    
    except KeyboardInterrupt:
        print("\n\n❌ Analysis interrupted by user", file=sys.stderr)
        return 130
    
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        return 1
    
    finally:
        # Restore stdout and close output file if opened
        if args.output:
            sys.stdout = original_stdout
            output_file.close()
            if not args.quiet:
                print(f"✅ Output saved to: {args.output}")


if __name__ == "__main__":
    sys.exit(main())

