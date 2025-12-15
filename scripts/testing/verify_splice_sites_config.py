#!/usr/bin/env python3
"""
Verify that the genomic resources manager is configured to use
splice_sites_enhanced.tsv by default.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from meta_spliceai.system.genomic_resources import Registry

def main():
    print("=" * 80)
    print("🔍 VERIFYING SPLICE SITES CONFIGURATION")
    print("=" * 80)
    print()
    
    # Create registry
    registry = Registry()
    
    # Get splice sites path using resolve()
    splice_sites_path = registry.resolve("splice_sites")
    
    print(f"Registry configuration:")
    print(f"  Data root: {registry.cfg.data_root}")
    print(f"  Build: {registry.cfg.build}")
    print(f"  Release: {registry.cfg.release}")
    print()
    
    if registry.cfg.derived_datasets:
        print(f"Derived datasets configuration:")
        for key, value in registry.cfg.derived_datasets.items():
            print(f"  {key}: {value}")
        print()
    
    print(f"Resolved splice sites path:")
    print(f"  {splice_sites_path}")
    print()
    
    # Check if it's the enhanced version
    if "enhanced" in splice_sites_path:
        print("✅ SUCCESS: Using splice_sites_enhanced.tsv")
        
        # Verify file exists and check columns
        if Path(splice_sites_path).exists():
            print("✅ File exists")
            
            # Check for enhanced columns
            import polars as pl
            df = pl.read_csv(splice_sites_path, separator='\t', n_rows=1,
                           schema_overrides={'chrom': pl.Utf8})
            
            print(f"\nColumns in splice_sites_enhanced.tsv:")
            for col in df.columns:
                print(f"  - {col}")
            
            # Check for enhanced metadata columns (scores are computed during base model pass)
            enhanced_cols = ['gene_biotype', 'transcript_biotype', 'exon_number', 
                           'exon_rank']
            missing_cols = [col for col in enhanced_cols if col not in df.columns]
            
            if missing_cols:
                print(f"\n⚠️  Missing expected enhanced metadata columns: {missing_cols}")
            else:
                print(f"\n✅ All expected enhanced metadata columns present!")
                print(f"   (Splice scores like donor_score, acceptor_score are computed during base model pass)")
        else:
            print("❌ File does not exist!")
    else:
        print(f"⚠️  Using regular splice_sites.tsv (not enhanced)")
        print(f"   Expected: splice_sites_enhanced.tsv")
        print(f"   Check if the enhanced file exists at:")
        print(f"   {registry.cfg.data_root}/splice_sites_enhanced.tsv")
    
    print()
    print("=" * 80)
    print("VERIFICATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()

