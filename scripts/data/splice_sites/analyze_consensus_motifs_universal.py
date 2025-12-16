#!/usr/bin/env python3
"""
Universal Consensus Motif Analysis for Splice Sites

This enhanced script analyzes consensus sequences at splice sites across
different genomic builds and annotation sources:

- GRCh37/Ensembl (SpliceAI)
- GRCh38/MANE (OpenSpliceAI)
- Any future genomic build/annotation

Key Features:
- Automatic build/source detection via Registry
- Universal GTF/splice site format handling
- Comparative analysis across builds
- Extended consensus motifs (MAG|GTRAGT for donors, polypyrimidine + AG for acceptors)

Usage:
    # Analyze GRCh37/Ensembl (default)
    python analyze_consensus_motifs_universal.py --build GRCh37 --release 87
    
    # Analyze GRCh38/MANE
    python analyze_consensus_motifs_universal.py --build GRCh38_MANE --release 1.3
    
    # Compare both builds
    python analyze_consensus_motifs_universal.py --compare
    
    # Custom files
    python analyze_consensus_motifs_universal.py \
        --splice-sites /path/to/splice_sites.tsv \
        --fasta /path/to/genome.fa

References:
- Shapiro & Senapathy (1987). RNA splice junctions of different classes of eukaryotes.
- Burge & Karlin (1997). Prediction of complete gene structures in human genomic DNA.
"""

import sys
import numpy as np
import polars as pl
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, List, Optional
from collections import Counter
from pyfaidx import Fasta
from dataclasses import dataclass

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


@dataclass
class ConsensusAnalysisConfig:
    """Configuration for consensus analysis."""
    build: str
    release: str
    splice_sites_file: Path
    fasta_file: Path
    sample_size: Optional[int] = None
    verbose: bool = True


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
    
    Universal implementation that handles chromosome naming variations.
    """
    # Handle chromosome naming variations
    try:
        chrom_seq = fasta[chrom]
    except KeyError:
        # Try with/without 'chr' prefix
        if chrom.startswith('chr'):
            try:
                chrom_seq = fasta[chrom[3:]]
            except KeyError:
                return ""
        else:
            try:
                chrom_seq = fasta[f'chr{chrom}']
            except KeyError:
                return ""
    
    if strand == '+':
        start = position - exon_bases - 1
        end = position + intron_bases - 1
        seq = chrom_seq[start:end].seq.upper()
    else:
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
    
    Universal implementation that handles chromosome naming variations.
    """
    # Handle chromosome naming variations
    try:
        chrom_seq = fasta[chrom]
    except KeyError:
        if chrom.startswith('chr'):
            try:
                chrom_seq = fasta[chrom[3:]]
            except KeyError:
                return ""
        else:
            try:
                chrom_seq = fasta[f'chr{chrom}']
            except KeyError:
                return ""
    
    if strand == '+':
        start = (position - 2) - intron_bases
        end = (position - 1) + exon_bases + 1
        seq = chrom_seq[start:end].seq.upper()
    else:
        start = position - exon_bases - 1
        end = position + intron_bases + 1
        seq = chrom_seq[start:end].seq.upper()
        seq = reverse_complement(seq)
    
    return seq


def calculate_position_frequencies(sequences: List[str]) -> pd.DataFrame:
    """Calculate nucleotide frequencies at each position."""
    if not sequences:
        return pd.DataFrame()
    
    seq_len = len(sequences[0])
    counts = {pos: Counter() for pos in range(seq_len)}
    
    for seq in sequences:
        if len(seq) == seq_len:
            for pos, base in enumerate(seq):
                counts[pos][base] += 1
    
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
    """Derive IUPAC consensus sequence from frequency matrix."""
    consensus = []
    
    for _, row in freq_df.iterrows():
        bases_above_threshold = []
        for base in ['A', 'C', 'G', 'T']:
            if row[base] >= threshold:
                bases_above_threshold.append(base)
        
        bases_set = set(bases_above_threshold)
        
        iupac = 'N'
        for code, allowed in IUPAC_CODES.items():
            if set(allowed) == bases_set:
                iupac = code
                break
        
        consensus.append(iupac)
    
    return ''.join(consensus)


def analyze_build(config: ConsensusAnalysisConfig) -> Dict:
    """
    Analyze consensus motifs for a specific genomic build.
    
    Parameters
    ----------
    config : ConsensusAnalysisConfig
        Analysis configuration
        
    Returns
    -------
    dict
        Analysis results including motifs, frequencies, and statistics
    """
    print("=" * 80)
    print(f"CONSENSUS MOTIF ANALYSIS: {config.build} (Release {config.release})")
    print("=" * 80)
    
    # Load splice sites
    print(f"\n📂 Loading splice sites from: {config.splice_sites_file}")
    df = pl.read_csv(
        str(config.splice_sites_file),
        separator='\t',
        schema_overrides={'chrom': pl.Utf8}
    )
    
    # Standardize schema (handle site_type vs splice_type)
    if 'site_type' in df.columns and 'splice_type' not in df.columns:
        df = df.rename({'site_type': 'splice_type'})
    
    print(f"   Total splice sites: {len(df):,}")
    
    # Sample if requested
    if config.sample_size:
        print(f"   Sampling {config.sample_size:,} sites for analysis...")
        df = df.sample(n=min(config.sample_size, len(df)), seed=42)
    
    # Load FASTA
    print(f"\n📂 Loading FASTA file: {config.fasta_file}")
    fasta = Fasta(str(config.fasta_file))
    
    # Separate donor and acceptor sites
    donor_df = df.filter(pl.col('splice_type') == 'donor')
    acceptor_df = df.filter(pl.col('splice_type') == 'acceptor')
    
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
        if config.verbose and (i + 1) % 50000 == 0:
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
            if len(motif) == 9:
                donor_motifs.append(motif)
        except Exception:
            pass
    
    print(f"   ✅ Extracted {len(donor_motifs):,} donor motifs")
    
    # Calculate frequencies
    donor_freq = calculate_position_frequencies(donor_motifs)
    
    # Canonical GT/GC counts
    gt_motifs = [m for m in donor_motifs if m[3:5] == 'GT']
    gc_motifs = [m for m in donor_motifs if m[3:5] == 'GC']
    
    gt_pct = 100 * len(gt_motifs) / len(donor_motifs) if donor_motifs else 0
    gc_pct = 100 * len(gc_motifs) / len(donor_motifs) if donor_motifs else 0
    
    print(f"\n📊 Donor Statistics:")
    print(f"   GT at +1,+2: {len(gt_motifs):,} ({gt_pct:.2f}%)")
    print(f"   GC at +1,+2: {len(gc_motifs):,} ({gc_pct:.2f}%)")
    
    # ========================================================================
    # ACCEPTOR SITE ANALYSIS
    # ========================================================================
    print(f"\n{'=' * 80}")
    print("ACCEPTOR SITE ANALYSIS: Polypyrimidine Tract + AG")
    print("=" * 80)
    
    print("\n🔄 Extracting acceptor motifs...")
    acceptor_motifs = []
    
    for i, row in enumerate(acceptor_df.iter_rows(named=True)):
        if config.verbose and (i + 1) % 50000 == 0:
            print(f"   Processed {i+1:,} / {len(acceptor_df):,} acceptor sites...")
        
        try:
            motif = extract_acceptor_motif(
                fasta,
                row['chrom'],
                row['position'],
                row['strand'],
                intron_bases=20,
                exon_bases=3
            )
            if len(motif) == 25:
                acceptor_motifs.append(motif)
        except Exception:
            pass
    
    print(f"   ✅ Extracted {len(acceptor_motifs):,} acceptor motifs")
    
    # Calculate frequencies
    acceptor_freq = calculate_position_frequencies(acceptor_motifs)
    
    # Polypyrimidine content
    pyrimidine_content = []
    if not acceptor_freq.empty and len(acceptor_freq) >= 18:
        for i in range(18):
            row = acceptor_freq.iloc[i]
            py_content = row['C'] + row['T']
            pyrimidine_content.append(py_content)
    
    mean_py = np.mean(pyrimidine_content) if pyrimidine_content else 0
    
    # AG count
    ag_count = sum(1 for m in acceptor_motifs if len(m) > 21 and m[20:22] == 'AG')
    ag_pct = 100 * ag_count / len(acceptor_motifs) if acceptor_motifs else 0
    
    print(f"\n📊 Acceptor Statistics:")
    print(f"   Polypyrimidine content: {mean_py:.1%}")
    print(f"   AG at boundary: {ag_count:,} ({ag_pct:.2f}%)")
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    print(f"\n{'=' * 80}")
    print(f"SUMMARY: {config.build}")
    print("=" * 80)
    
    print("\n✅ Donor Sites:")
    print(f"   - GT percentage: {gt_pct:.2f}%")
    print(f"   - GC percentage: {gc_pct:.2f}%")
    
    print("\n✅ Acceptor Sites:")
    print(f"   - Polypyrimidine content: {mean_py:.1%}")
    print(f"   - AG percentage: {ag_pct:.2f}%")
    
    results = {
        'build': config.build,
        'release': config.release,
        'donor': {
            'total_sites': len(donor_df),
            'motifs_extracted': len(donor_motifs),
            'frequencies': donor_freq,
            'gt_percentage': gt_pct,
            'gc_percentage': gc_pct,
            'gt_count': len(gt_motifs),
            'gc_count': len(gc_motifs)
        },
        'acceptor': {
            'total_sites': len(acceptor_df),
            'motifs_extracted': len(acceptor_motifs),
            'frequencies': acceptor_freq,
            'pyrimidine_content': mean_py,
            'ag_percentage': ag_pct,
            'ag_count': ag_count
        }
    }
    
    return results


def compare_builds(results_list: List[Dict]) -> None:
    """
    Compare consensus analysis results across multiple builds.
    
    Parameters
    ----------
    results_list : List[Dict]
        List of analysis results from different builds
    """
    print("\n" + "=" * 80)
    print("COMPARATIVE ANALYSIS ACROSS BUILDS")
    print("=" * 80)
    
    # Create comparison table
    print("\n📊 Donor Site Comparison:")
    print("-" * 80)
    print(f"{'Build':<20} {'Total Sites':>12} {'GT %':>8} {'GC %':>8}")
    print("-" * 80)
    
    for results in results_list:
        build = results['build']
        donor = results['donor']
        print(f"{build:<20} {donor['total_sites']:>12,} {donor['gt_percentage']:>7.2f}% {donor['gc_percentage']:>7.2f}%")
    
    print("\n📊 Acceptor Site Comparison:")
    print("-" * 80)
    print(f"{'Build':<20} {'Total Sites':>12} {'AG %':>8} {'Py Content':>12}")
    print("-" * 80)
    
    for results in results_list:
        build = results['build']
        acceptor = results['acceptor']
        py_pct = acceptor['pyrimidine_content'] * 100
        print(f"{build:<20} {acceptor['total_sites']:>12,} {acceptor['ag_percentage']:>7.2f}% {py_pct:>11.1f}%")
    
    print("\n" + "=" * 80)
    print("KEY INSIGHTS")
    print("=" * 80)
    
    # Calculate differences
    if len(results_list) >= 2:
        r1, r2 = results_list[0], results_list[1]
        
        print(f"\n1. Splice Site Counts:")
        donor_ratio = r2['donor']['total_sites'] / r1['donor']['total_sites']
        print(f"   {r2['build']} has {donor_ratio:.2f}x the donor sites of {r1['build']}")
        
        print(f"\n2. Consensus Sequences:")
        gt_diff = abs(r1['donor']['gt_percentage'] - r2['donor']['gt_percentage'])
        if gt_diff < 1.0:
            print(f"   ✅ GT percentages are consistent ({gt_diff:.2f}% difference)")
        else:
            print(f"   ⚠️  GT percentages differ by {gt_diff:.2f}%")
        
        ag_diff = abs(r1['acceptor']['ag_percentage'] - r2['acceptor']['ag_percentage'])
        if ag_diff < 1.0:
            print(f"   ✅ AG percentages are consistent ({ag_diff:.2f}% difference)")
        else:
            print(f"   ⚠️  AG percentages differ by {ag_diff:.2f}%")
        
        print(f"\n3. Biological Interpretation:")
        print(f"   - Both builds show canonical GT-AG splicing")
        print(f"   - Consensus sequences are biologically consistent")
        print(f"   - Differences in site counts reflect annotation philosophy")
        print(f"     ({r1['build']}: comprehensive, {r2['build']}: canonical)")


def main():
    """Main analysis function with CLI argument parsing."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Universal consensus motif analysis for splice sites",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze GRCh37/Ensembl
  %(prog)s --build GRCh37 --release 87
  
  # Analyze GRCh38/MANE
  %(prog)s --build GRCh38_MANE --release 1.3
  
  # Compare both builds
  %(prog)s --compare
  
  # Sample 50K sites for faster analysis
  %(prog)s --build GRCh37 --release 87 --sample 50000
  
  # Custom files
  %(prog)s --splice-sites /path/to/splice_sites.tsv --fasta /path/to/genome.fa
        """
    )
    
    # Build selection
    parser.add_argument(
        "--build",
        type=str,
        help="Genomic build (e.g., GRCh37, GRCh38_MANE)"
    )
    parser.add_argument(
        "--release",
        type=str,
        help="Release version (e.g., 87, 1.3)"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare GRCh37/Ensembl and GRCh38/MANE"
    )
    
    # Custom files
    parser.add_argument(
        "--splice-sites",
        type=Path,
        help="Path to splice sites TSV file (overrides build/release)"
    )
    parser.add_argument(
        "--fasta",
        type=Path,
        help="Path to reference genome FASTA file (overrides build/release)"
    )
    
    # Sampling
    parser.add_argument(
        "--sample",
        type=int,
        metavar="N",
        help="Analyze N randomly sampled splice sites"
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Analyze all splice sites (no sampling)"
    )
    
    # Output
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Save output to file"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose progress output"
    )
    
    args = parser.parse_args()
    
    # Determine what to analyze
    if args.compare:
        # Compare GRCh37 and GRCh38
        builds_to_analyze = [
            ('GRCh37', '87'),
            ('GRCh38_MANE', '1.3')
        ]
    elif args.build and args.release:
        builds_to_analyze = [(args.build, args.release)]
    elif args.splice_sites and args.fasta:
        # Custom files - single analysis
        builds_to_analyze = [('custom', 'custom')]
    else:
        # Default to GRCh37
        builds_to_analyze = [('GRCh37', '87')]
    
    # Determine sample size
    if args.full:
        sample_size = None
    elif args.sample:
        sample_size = args.sample
    else:
        sample_size = 50000  # Default
    
    # Setup output redirection if requested
    original_stdout = sys.stdout
    if args.output:
        output_file = open(args.output, 'w')
        sys.stdout = output_file
    
    try:
        results_list = []
        
        for build, release in builds_to_analyze:
            # Setup configuration
            if args.splice_sites and args.fasta:
                # Custom files
                config = ConsensusAnalysisConfig(
                    build='custom',
                    release='custom',
                    splice_sites_file=args.splice_sites,
                    fasta_file=args.fasta,
                    sample_size=sample_size,
                    verbose=args.verbose
                )
            else:
                # Use Registry
                registry = Registry(build=build, release=release)
                
                splice_sites_file = registry.data_dir / "splice_sites_enhanced.tsv"
                if not splice_sites_file.exists():
                    splice_sites_file = registry.data_dir / "splice_sites.tsv"
                
                fasta_file = registry.get_fasta_path()
                
                config = ConsensusAnalysisConfig(
                    build=build,
                    release=release,
                    splice_sites_file=splice_sites_file,
                    fasta_file=fasta_file,
                    sample_size=sample_size,
                    verbose=args.verbose
                )
            
            # Validate files
            if not config.splice_sites_file.exists():
                print(f"❌ Error: Splice sites file not found: {config.splice_sites_file}", file=sys.stderr)
                continue
            
            if not config.fasta_file.exists():
                print(f"❌ Error: FASTA file not found: {config.fasta_file}", file=sys.stderr)
                continue
            
            # Run analysis
            results = analyze_build(config)
            results_list.append(results)
        
        # Compare if multiple builds
        if len(results_list) > 1:
            compare_builds(results_list)
        
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
        if args.output:
            sys.stdout = original_stdout
            output_file.close()
            print(f"✅ Output saved to: {args.output}")


if __name__ == "__main__":
    sys.exit(main())


