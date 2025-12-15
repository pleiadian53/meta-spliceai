"""Command-line interface for genomic resources management.

Provides subcommands:
- audit: Check status of all genomic resources
- bootstrap: Download GTF/FASTA from Ensembl
- derive: Generate derived TSV datasets (future)
- set-current: Switch between builds (future)
"""

import sys
import argparse
from pathlib import Path
from typing import Optional

from .config import load_config
from .registry import Registry
from .download import fetch_ensembl
from .derive import GenomicDataDeriver


def cmd_audit(args):
    """Audit genomic resources and show status."""
    print("=" * 80)
    print("Genomic Resources Audit")
    print("=" * 80)
    
    # Load configuration
    cfg = load_config()
    print(f"\nConfiguration:")
    print(f"  Species:     {cfg.species}")
    print(f"  Build:       {cfg.build}")
    print(f"  Release:     {cfg.release}")
    print(f"  Data root:   {cfg.data_root}")
    
    # Check registry
    print(f"\nResource Status:")
    registry = Registry(build=args.build, release=args.release)
    
    resources = registry.list_all()
    
    # Format output
    max_kind_len = max(len(k) for k in resources.keys())
    
    for kind, path in resources.items():
        status = "✓ FOUND" if path else "✗ MISSING"
        path_str = path if path else "(not found)"
        
        # Color coding (if terminal supports it)
        if path:
            print(f"  {kind:<{max_kind_len}}  {status:10}  {path_str}")
        else:
            print(f"  {kind:<{max_kind_len}}  {status:10}  {path_str}")
    
    # Summary
    found = sum(1 for p in resources.values() if p)
    total = len(resources)
    
    print(f"\n{'=' * 80}")
    print(f"Summary: {found}/{total} resources found")
    
    # Check critical resources
    critical = ["gtf", "fasta", "splice_sites", "gene_features", 
                "transcript_features", "exon_features"]
    missing_critical = [k for k in critical if not resources.get(k)]
    
    if missing_critical:
        print(f"\n⚠️  Missing critical resources: {', '.join(missing_critical)}")
        print(f"\nRecommended actions:")
        if "gtf" in missing_critical or "fasta" in missing_critical:
            print(f"  1. Run: python -m meta_spliceai.system.genomic_resources.cli bootstrap")
        if any(k in missing_critical for k in ["splice_sites", "gene_features", 
                                                "transcript_features", "exon_features"]):
            print(f"  2. Run: python -m meta_spliceai.system.genomic_resources.cli derive")
        return 1
    else:
        print(f"\n✅ All critical resources present!")
        return 0


def cmd_bootstrap(args):
    """Bootstrap genomic resources by downloading from Ensembl."""
    print("=" * 80)
    print("Bootstrap Genomic Resources")
    print("=" * 80)
    
    print(f"\nConfiguration:")
    print(f"  Build:       {args.build}")
    print(f"  Release:     {args.release}")
    print(f"  Fetch GTF:   {args.gtf or args.all}")
    print(f"  Fetch FASTA: {args.fasta or args.all}")
    print(f"  Index FASTA: {args.index}")
    print(f"  Force:       {args.force}")
    print(f"  Dry run:     {args.dry_run}")
    print()
    
    # Determine what to fetch
    fetch_gtf = args.gtf or args.all
    fetch_fasta = args.fasta or args.all
    
    # If nothing specified, fetch both
    if not fetch_gtf and not fetch_fasta:
        fetch_gtf = fetch_fasta = True
    
    try:
        gtf_path, fasta_path, fai_path = fetch_ensembl(
            build=args.build,
            release=args.release,
            fetch_gtf=fetch_gtf,
            fetch_fasta=fetch_fasta,
            index_fasta=args.index,
            force=args.force,
            dry_run=args.dry_run,
            verbose=True,
        )
        
        if not args.dry_run:
            print("\n" + "=" * 80)
            if gtf_path or fasta_path:
                print("✅ Bootstrap completed successfully!")
                print("\nNext steps:")
                print("  1. Verify files: python -m meta_spliceai.system.genomic_resources.cli audit")
                print("  2. Generate derived datasets: python -m meta_spliceai.system.genomic_resources.cli derive")
                return 0
            else:
                print("⚠️  Bootstrap completed with errors")
                return 1
        else:
            return 0
            
    except Exception as e:
        print(f"\n❌ Bootstrap failed: {e}")
        import traceback
        if args.verbose:
            traceback.print_exc()
        return 1


def cmd_derive(args):
    """Generate derived TSV datasets from GTF/FASTA."""
    print("=" * 80)
    print("Genomic Data Derivation")
    print("=" * 80)
    
    # Load configuration with overrides
    cfg = load_config()
    if args.build:
        cfg.build = args.build
    if args.release:
        cfg.release = args.release
    
    print(f"\nConfiguration:")
    print(f"  Build:       {cfg.build}")
    print(f"  Release:     {cfg.release}")
    print(f"  Data root:   {cfg.data_root}")
    
    # Create registry with build/release overrides
    from .registry import Registry
    registry = Registry(build=cfg.build, release=cfg.release)
    
    # Create deriver with build-specific data directory
    data_dir = cfg.data_root / cfg.build
    deriver = GenomicDataDeriver(
        data_dir=data_dir,
        config=cfg,
        registry=registry,
        verbosity=args.verbose
    )
    
    # Determine what to derive
    derive_all = args.all or not (
        args.annotations or args.splice_sites or args.sequences or 
        args.overlapping_genes or args.gene_features or 
        args.transcript_features or args.exon_features or args.junctions
    )
    
    results = {}
    
    try:
        if derive_all:
            print("\n🔄 Deriving all datasets...")
            results = deriver.derive_all(
                consensus_window=args.consensus_window,
                target_chromosomes=args.chromosomes,
                force_overwrite=args.force
            )
        else:
            if args.annotations:
                print("\n🔄 Deriving gene annotations...")
                results['annotations'] = deriver.derive_gene_annotations(
                    target_chromosomes=args.chromosomes,
                    force_overwrite=args.force
                )
            
            if args.gene_features:
                print("\n🔄 Deriving gene features...")
                results['gene_features'] = deriver.derive_gene_features(
                    target_chromosomes=args.chromosomes,
                    force_overwrite=args.force
                )
            
            if args.transcript_features:
                print("\n🔄 Deriving transcript features...")
                results['transcript_features'] = deriver.derive_transcript_features(
                    target_chromosomes=args.chromosomes,
                    force_overwrite=args.force
                )
            
            if args.exon_features:
                print("\n🔄 Deriving exon features...")
                results['exon_features'] = deriver.derive_exon_features(
                    target_chromosomes=args.chromosomes,
                    force_overwrite=args.force
                )
            
            if args.splice_sites:
                print("\n🔄 Deriving splice sites...")
                results['splice_sites'] = deriver.derive_splice_sites(
                    consensus_window=args.consensus_window,
                    target_chromosomes=args.chromosomes,
                    force_overwrite=args.force
                )
            
            if args.junctions:
                print("\n🔄 Deriving junctions...")
                results['junctions'] = deriver.derive_junctions(
                    target_chromosomes=args.chromosomes,
                    force_overwrite=args.force
                )
            
            if args.sequences:
                print("\n🔄 Deriving genomic sequences...")
                results['sequences'] = deriver.derive_genomic_sequences(
                    target_chromosomes=args.chromosomes,
                    force_overwrite=args.force
                )
            
            if args.overlapping_genes:
                print("\n🔄 Deriving overlapping genes...")
                results['overlapping_genes'] = deriver.derive_overlapping_genes(
                    target_chromosomes=args.chromosomes,
                    force_overwrite=args.force
                )
        
        # Check if all succeeded
        all_success = all(r.get('success', False) for r in results.values())
        
        if all_success:
            print("\n✅ All derivations completed successfully!")
            return 0
        else:
            print("\n⚠️  Some derivations failed. Check output above for details.")
            return 1
            
    except Exception as e:
        print(f"\n❌ Error during derivation: {e}")
        return 1


def cmd_set_current(args):
    """Set current build by copying/linking files."""
    print("=" * 80)
    print("Set Current Build")
    print("=" * 80)
    print("\n⚠️  This command is not yet implemented.")
    print(f"\nThis will set the current build to: {args.build}")
    print("\nPlease implement Stage 4 of the rebuild plan.")
    return 1


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Genomic Resources Manager - manage GTF, FASTA, and derived datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # audit command
    audit_parser = subparsers.add_parser(
        "audit",
        help="Check status of genomic resources",
    )
    audit_parser.add_argument(
        "--build",
        type=str,
        default=None,
        help="Override build (default: from config)",
    )
    audit_parser.add_argument(
        "--release",
        type=str,
        default=None,
        help="Override release (default: from config)",
    )
    audit_parser.set_defaults(func=cmd_audit)
    
    # bootstrap command
    bootstrap_parser = subparsers.add_parser(
        "bootstrap",
        help="Download GTF/FASTA from Ensembl",
    )
    bootstrap_parser.add_argument(
        "--build",
        type=str,
        default="GRCh38",
        help="Genome build (default: GRCh38)",
    )
    bootstrap_parser.add_argument(
        "--release",
        type=str,
        default="112",
        help="Ensembl release (default: 112)",
    )
    bootstrap_parser.add_argument(
        "--gtf",
        action="store_true",
        help="Download only GTF",
    )
    bootstrap_parser.add_argument(
        "--fasta",
        action="store_true",
        help="Download only FASTA",
    )
    bootstrap_parser.add_argument(
        "--all",
        action="store_true",
        help="Download both GTF and FASTA (default if neither --gtf nor --fasta specified)",
    )
    bootstrap_parser.add_argument(
        "--no-index",
        dest="index",
        action="store_false",
        default=True,
        help="Skip FASTA indexing",
    )
    bootstrap_parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if files exist",
    )
    bootstrap_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without doing it",
    )
    bootstrap_parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output",
    )
    bootstrap_parser.set_defaults(func=cmd_bootstrap)
    
    # derive command
    derive_parser = subparsers.add_parser(
        "derive",
        help="Generate derived TSV datasets from GTF/FASTA",
    )
    derive_parser.add_argument(
        "--all",
        action="store_true",
        help="Derive all datasets (annotations, features, splice sites, junctions, sequences, overlapping genes)",
    )
    derive_parser.add_argument(
        "--annotations",
        action="store_true",
        help="Derive gene annotations only",
    )
    derive_parser.add_argument(
        "--gene-features",
        action="store_true",
        help="Derive gene features only",
    )
    derive_parser.add_argument(
        "--transcript-features",
        action="store_true",
        help="Derive transcript features only",
    )
    derive_parser.add_argument(
        "--exon-features",
        action="store_true",
        help="Derive exon features only",
    )
    derive_parser.add_argument(
        "--splice-sites",
        action="store_true",
        help="Derive splice sites only",
    )
    derive_parser.add_argument(
        "--junctions",
        action="store_true",
        help="Derive splice junctions only",
    )
    derive_parser.add_argument(
        "--sequences",
        action="store_true",
        help="Derive genomic sequences only",
    )
    derive_parser.add_argument(
        "--overlapping-genes",
        action="store_true",
        help="Derive overlapping gene metadata only",
    )
    derive_parser.add_argument(
        "--build",
        type=str,
        default=None,
        help="Override build (default: from config)",
    )
    derive_parser.add_argument(
        "--release",
        type=str,
        default=None,
        help="Override release (default: from config)",
    )
    derive_parser.add_argument(
        "--consensus-window",
        type=int,
        default=2,
        help="Consensus window for splice sites (default: 2)",
    )
    derive_parser.add_argument(
        "--chromosomes",
        nargs="+",
        help="Target specific chromosomes (e.g., 1 2 X Y)",
    )
    derive_parser.add_argument(
        "--force",
        action="store_true",
        help="Force regeneration even if files exist",
    )
    derive_parser.add_argument(
        "-v", "--verbose",
        action="count",
        default=1,
        help="Increase verbosity (can be repeated)",
    )
    derive_parser.set_defaults(func=cmd_derive)
    
    # set-current command (placeholder)
    set_current_parser = subparsers.add_parser(
        "set-current",
        help="Set current build (not yet implemented)",
    )
    set_current_parser.add_argument(
        "--build",
        type=str,
        required=True,
        help="Build to set as current (e.g., GRCh38, GRCh37)",
    )
    set_current_parser.set_defaults(func=cmd_set_current)
    
    # Parse and execute
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
