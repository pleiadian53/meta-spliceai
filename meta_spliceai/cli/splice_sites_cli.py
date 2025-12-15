#!/usr/bin/env python3
"""
Meta-SpliceAI Splice Sites CLI

Command-line interface for generating and validating splice site annotations
from GTF files with enhanced metadata support.

Usage:
    meta-spliceai-splice-sites --gtf <path> --output <path> [options]
    meta-spliceai-splice-sites --build mane-grch38 [options]
    meta-spliceai-splice-sites --build ensembl-grch37 --validate [options]

Examples:
    # Generate splice sites with validation
    meta-spliceai-splice-sites \\
        --gtf data/mane/GRCh38/MANE.GRCh38.v1.3.refseq_genomic.gtf \\
        --output data/mane/GRCh38/splice_sites_enhanced.tsv \\
        --validate

    # Quick generation for known build
    meta-spliceai-splice-sites --build mane-grch38 --validate

    # Generate with custom consensus window
    meta-spliceai-splice-sites \\
        --build ensembl-grch37 \\
        --consensus-window 3 \\
        --validate \\
        --verbose
"""

import sys
import argparse
from pathlib import Path
from typing import Optional, Dict

# Add project root to path for imports
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from meta_spliceai.system.genomic_resources import extract_splice_sites_from_gtf


# Predefined build configurations
BUILD_CONFIGS = {
    'mane-grch38': {
        'name': 'MANE GRCh38',
        'gtf': 'data/mane/GRCh38/MANE.GRCh38.v1.3.refseq_genomic.gtf',
        'output': 'data/mane/GRCh38/splice_sites_enhanced.tsv',
        'description': 'MANE Select v1.3 (GRCh38/hg38) - Used by OpenSpliceAI'
    },
    'ensembl-grch37': {
        'name': 'Ensembl GRCh37',
        'gtf': 'data/ensembl/GRCh37/Homo_sapiens.GRCh37.87.gtf',
        'output': 'data/ensembl/GRCh37/splice_sites_enhanced.tsv',
        'description': 'Ensembl Release 87 (GRCh37/hg19) - Used by SpliceAI'
    },
    'ensembl-grch38': {
        'name': 'Ensembl GRCh38',
        'gtf': 'data/ensembl/GRCh38/Homo_sapiens.GRCh38.112.gtf',
        'output': 'data/ensembl/GRCh38/splice_sites_enhanced.tsv',
        'description': 'Ensembl Release 112 (GRCh38/hg38)'
    }
}


def validate_splice_sites_consistency(gtf_path: Path, ss_path: Path, verbosity: int = 1) -> bool:
    """
    Validate splice sites file consistency with GTF.
    
    Parameters
    ----------
    gtf_path : Path
        Path to GTF file
    ss_path : Path
        Path to splice sites file
    verbosity : int
        Verbosity level (0=quiet, 1=normal, 2=detailed)
    
    Returns
    -------
    bool
        True if all validation tests pass, False otherwise
    """
    try:
        # Import validation functions
        sys.path.insert(0, str(project_root / "scripts/testing"))
        from validate_splice_sites_consistency import (
            count_transcripts_in_gtf,
            analyze_splice_sites_file,
            validate_consistency
        )
        
        if verbosity >= 1:
            print("\n" + "=" * 80)
            print("VALIDATING SPLICE SITES CONSISTENCY")
            print("=" * 80)
        
        # Analyze GTF
        gtf_stats = count_transcripts_in_gtf(gtf_path, min_exons=2, verbosity=verbosity)
        
        if verbosity >= 1:
            print("\nGTF Summary:")
            print(f"  Total transcripts:             {gtf_stats['total_transcripts']:,}")
            print(f"  Transcripts with splicing:     {gtf_stats['transcripts_with_splicing']:,}")
            print(f"  Single-exon transcripts:       {gtf_stats['transcripts_single_exon']:,}")
            print(f"  Unique genes (total):          {gtf_stats['unique_genes']:,}")
            print(f"  Unique genes (with splicing):  {gtf_stats['unique_genes_with_splicing']:,}")
        
        # Analyze splice sites file
        ss_stats = analyze_splice_sites_file(ss_path, verbosity=verbosity)
        
        if verbosity >= 1:
            print("\nSplice Sites Summary:")
            print(f"  Total splice sites:            {ss_stats['total_splice_sites']:,}")
            print(f"  Unique genes:                  {ss_stats['unique_genes']:,}")
            print(f"  Unique transcripts:            {ss_stats['unique_transcripts']:,}")
            print(f"  Donor sites:                   {ss_stats['donor_sites']:,}")
            print(f"  Acceptor sites:                {ss_stats['acceptor_sites']:,}")
        
        # Validate consistency
        success = validate_consistency(gtf_stats, ss_stats, verbosity=verbosity)
        
        return success
        
    except ImportError as e:
        print(f"❌ Error: Could not import validation module: {e}")
        print("   Make sure you're running from the project root.")
        return False
    except Exception as e:
        print(f"❌ Validation error: {e}")
        return False


def generate_splice_sites(
    gtf_path: Path,
    output_path: Path,
    consensus_window: int = 2,
    verbosity: int = 1,
    validate: bool = False
) -> bool:
    """
    Generate splice site annotations from GTF file.
    
    Parameters
    ----------
    gtf_path : Path
        Path to input GTF file
    output_path : Path
        Path to output TSV file
    consensus_window : int
        Consensus window size (default: 2)
    verbosity : int
        Verbosity level (0=quiet, 1=normal, 2=detailed)
    validate : bool
        Whether to validate after generation
    
    Returns
    -------
    bool
        True if successful, False otherwise
    """
    try:
        # Check if GTF file exists
        if not gtf_path.exists():
            print(f"❌ Error: GTF file not found: {gtf_path}")
            return False
        
        # Create output directory if needed
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Generate splice sites
        if verbosity >= 1:
            print("\n" + "=" * 80)
            print("GENERATING SPLICE SITES WITH ENHANCED METADATA")
            print("=" * 80)
            print(f"\nInput GTF:  {gtf_path}")
            print(f"Output TSV: {output_path}")
            print(f"Consensus window: ±{consensus_window} nucleotides")
            print(f"Validation: {'Enabled' if validate else 'Disabled'}")
            print()
        
        # Extract splice sites
        df = extract_splice_sites_from_gtf(
            gtf_path=str(gtf_path),
            consensus_window=consensus_window,
            output_file=str(output_path),
            verbosity=verbosity
        )
        
        if verbosity >= 1:
            print(f"\n✓ Successfully generated: {output_path}")
            print(f"  Rows: {len(df):,}")
            print(f"  Columns: {len(df.columns)}")
        
        # Validate if requested
        if validate:
            if verbosity >= 1:
                print("\n" + "-" * 80)
            success = validate_splice_sites_consistency(gtf_path, output_path, verbosity)
            if not success:
                print("\n⚠ Warning: Validation found issues. Please review the output above.")
                return False
        
        if verbosity >= 1:
            print("\n" + "=" * 80)
            print("✅ SPLICE SITES GENERATION COMPLETE")
            print("=" * 80)
        
        return True
        
    except Exception as e:
        print(f"❌ Error during splice site generation: {e}")
        import traceback
        if verbosity >= 2:
            traceback.print_exc()
        return False


def list_builds(verbose: bool = False):
    """List available genomic build configurations."""
    print("\n" + "=" * 80)
    print("AVAILABLE GENOMIC BUILDS")
    print("=" * 80)
    
    for build_id, config in BUILD_CONFIGS.items():
        print(f"\n{build_id}")
        print(f"  Name: {config['name']}")
        print(f"  Description: {config['description']}")
        if verbose:
            print(f"  GTF: {config['gtf']}")
            print(f"  Output: {config['output']}")
    
    print("\nUsage:")
    print("  meta-spliceai-splice-sites --build <build-id> [options]")
    print("\nExample:")
    print("  meta-spliceai-splice-sites --build mane-grch38 --validate")
    print()


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate and validate splice site annotations from GTF files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate with validation
  %(prog)s --gtf data.gtf --output sites.tsv --validate

  # Use predefined build
  %(prog)s --build mane-grch38 --validate

  # Custom consensus window
  %(prog)s --build ensembl-grch37 --consensus-window 3

  # List available builds
  %(prog)s --list-builds
        """
    )
    
    # Input options
    input_group = parser.add_argument_group('Input Options')
    input_group.add_argument(
        '--gtf',
        type=str,
        help='Path to input GTF file'
    )
    input_group.add_argument(
        '--build',
        type=str,
        choices=list(BUILD_CONFIGS.keys()),
        help='Use predefined genomic build configuration'
    )
    
    # Output options
    output_group = parser.add_argument_group('Output Options')
    output_group.add_argument(
        '--output',
        '-o',
        type=str,
        help='Path to output TSV file (default: auto-determined from build)'
    )
    
    # Processing options
    proc_group = parser.add_argument_group('Processing Options')
    proc_group.add_argument(
        '--consensus-window',
        '-w',
        type=int,
        default=2,
        help='Consensus window size in nucleotides (default: 2)'
    )
    proc_group.add_argument(
        '--validate',
        action='store_true',
        help='Validate splice sites consistency after generation'
    )
    
    # Utility options
    util_group = parser.add_argument_group('Utility Options')
    util_group.add_argument(
        '--list-builds',
        action='store_true',
        help='List available genomic build configurations'
    )
    util_group.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help='Enable verbose output'
    )
    util_group.add_argument(
        '--quiet',
        '-q',
        action='store_true',
        help='Suppress output (errors only)'
    )
    
    return parser.parse_args()


def main():
    """Main CLI entry point."""
    args = parse_args()
    
    # Determine verbosity
    if args.quiet:
        verbosity = 0
    elif args.verbose:
        verbosity = 2
    else:
        verbosity = 1
    
    # List builds and exit
    if args.list_builds:
        list_builds(verbose=args.verbose)
        return 0
    
    # Determine input/output paths
    if args.build:
        # Use predefined build configuration
        config = BUILD_CONFIGS[args.build]
        gtf_path = project_root / config['gtf']
        output_path = project_root / config['output']
        
        # Allow override of output path
        if args.output:
            output_path = Path(args.output)
            if not output_path.is_absolute():
                output_path = project_root / output_path
        
        if verbosity >= 1:
            print(f"\nUsing build configuration: {config['name']}")
            print(f"  {config['description']}")
    
    elif args.gtf:
        # Use custom GTF file
        gtf_path = Path(args.gtf)
        if not gtf_path.is_absolute():
            gtf_path = project_root / gtf_path
        
        if not args.output:
            print("❌ Error: --output is required when using --gtf")
            return 1
        
        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_path = project_root / output_path
    
    else:
        print("❌ Error: Either --gtf or --build must be specified")
        print("   Use --help for usage information")
        print("   Use --list-builds to see available builds")
        return 1
    
    # Generate splice sites
    success = generate_splice_sites(
        gtf_path=gtf_path,
        output_path=output_path,
        consensus_window=args.consensus_window,
        verbosity=verbosity,
        validate=args.validate
    )
    
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())

