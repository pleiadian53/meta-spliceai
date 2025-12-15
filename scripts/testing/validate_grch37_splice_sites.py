#!/usr/bin/env python3
"""Validate GRCh37 splice sites file consistency with GTF.

Same as validate_splice_sites_consistency.py but configured for GRCh37.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

# Import main validation function
from validate_splice_sites_consistency import (
    count_transcripts_in_gtf,
    analyze_splice_sites_file,
    validate_consistency
)


def main():
    """Main validation function for GRCh37."""
    print("=" * 80)
    print("SPLICE SITES CONSISTENCY VALIDATION - Ensembl GRCh37")
    print("=" * 80)
    print()
    
    # Setup paths for GRCh37
    gtf_path = project_root / "data/ensembl/GRCh37/Homo_sapiens.GRCh37.87.gtf"
    ss_path = project_root / "data/ensembl/GRCh37/splice_sites_enhanced.tsv"
    
    # Check files exist
    if not gtf_path.exists():
        print(f"❌ GTF file not found: {gtf_path}")
        return False
    
    if not ss_path.exists():
        print(f"❌ Splice sites file not found: {ss_path}")
        return False
    
    # Analyze GTF
    gtf_stats = count_transcripts_in_gtf(gtf_path, min_exons=2, verbosity=1)
    
    print("\nGTF Summary:")
    print(f"  Total transcripts:             {gtf_stats['total_transcripts']:,}")
    print(f"  Transcripts with splicing:     {gtf_stats['transcripts_with_splicing']:,}")
    print(f"  Single-exon transcripts:       {gtf_stats['transcripts_single_exon']:,}")
    print(f"  Unique genes (total):          {gtf_stats['unique_genes']:,}")
    print(f"  Unique genes (with splicing):  {gtf_stats['unique_genes_with_splicing']:,}")
    
    # Analyze splice sites file
    ss_stats = analyze_splice_sites_file(ss_path, verbosity=1)
    
    print("\nSplice Sites Summary:")
    print(f"  Total splice sites:            {ss_stats['total_splice_sites']:,}")
    print(f"  Unique genes:                  {ss_stats['unique_genes']:,}")
    print(f"  Unique transcripts:            {ss_stats['unique_transcripts']:,}")
    print(f"  Donor sites:                   {ss_stats['donor_sites']:,}")
    print(f"  Acceptor sites:                {ss_stats['acceptor_sites']:,}")
    
    # Validate consistency
    success = validate_consistency(gtf_stats, ss_stats, verbosity=1)
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)


