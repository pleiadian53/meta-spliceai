#!/usr/bin/env python3
"""
Minimal check script to verify MetaSpliceAI dependencies and datasets.
This script imports nothing project-specific and simply checks for required resources.
"""

import os
import sys
from pathlib import Path

def check_file_exists(filepath, description):
    """Check if a file exists and is readable."""
    path = Path(filepath)
    if path.exists() and path.is_file():
        size = path.stat().st_size
        print(f"✅ {description}: {path} ({size:,} bytes)")
        return True
    else:
        print(f"❌ {description}: {path} (MISSING)")
        return False

def check_directory_exists(dirpath, description):
    """Check if a directory exists."""
    path = Path(dirpath)
    if path.exists() and path.is_dir():
        file_count = len(list(path.iterdir()))
        print(f"✅ {description}: {path} ({file_count} items)")
        return True
    else:
        print(f"❌ {description}: {path} (MISSING)")
        return False

def main():
    """Run minimal checks for MetaSpliceAI resources."""
    print("=" * 80)
    print("MetaSpliceAI Minimal Resource Check")
    print("=" * 80)
    
    # Get project root
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data"
    ensembl_dir = data_dir / "ensembl"
    spliceai_analysis_dir = ensembl_dir / "spliceai_analysis"
    
    print(f"\nProject root: {project_root}")
    print(f"Data directory: {data_dir}")
    
    # Check core directories
    print("\n📁 Directory Structure:")
    check_directory_exists(data_dir, "Data directory")
    check_directory_exists(ensembl_dir, "Ensembl directory")
    check_directory_exists(spliceai_analysis_dir, "SpliceAI analysis directory")
    
    # Check derived datasets (required)
    print("\n📊 Derived Datasets:")
    required_files = [
        (ensembl_dir / "splice_sites.tsv", "Splice sites"),
        (spliceai_analysis_dir / "gene_features.tsv", "Gene features"),
        (spliceai_analysis_dir / "transcript_features.tsv", "Transcript features"),
        (spliceai_analysis_dir / "exon_features.tsv", "Exon features"),
    ]
    
    all_derived_exist = True
    for filepath, description in required_files:
        if not check_file_exists(filepath, description):
            all_derived_exist = False
    
    # Check source files (GTF/FASTA)
    print("\n🧬 Source Files:")
    
    # Check for Ensembl GTF files (Homo_sapiens.GRCh38.*.gtf pattern)
    ensembl_gtf_files = list(data_dir.glob("Homo_sapiens.GRCh38.*.gtf")) + list(data_dir.glob("Homo_sapiens.GRCh38.*.gtf.gz"))
    ensembl_gtf_files.extend(list(ensembl_dir.glob("Homo_sapiens.GRCh38.*.gtf")) + list(ensembl_dir.glob("Homo_sapiens.GRCh38.*.gtf.gz")))
    
    # Check for Ensembl FASTA files (Homo_sapiens.GRCh38.dna.primary_assembly.fa pattern)
    ensembl_fasta_files = list(data_dir.glob("Homo_sapiens.GRCh38.dna.primary_assembly.fa")) + list(data_dir.glob("Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz"))
    ensembl_fasta_files.extend(list(ensembl_dir.glob("Homo_sapiens.GRCh38.dna.primary_assembly.fa")) + list(ensembl_dir.glob("Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz")))
    
    # Check for any other GTF/FASTA files (examples, etc.)
    other_gtf_files = [f for f in (list(data_dir.glob("*.gtf")) + list(data_dir.glob("*.gtf.gz"))) 
                      if not f.name.startswith("Homo_sapiens.GRCh38")]
    other_fasta_files = [f for f in (list(data_dir.glob("*.fa")) + list(data_dir.glob("*.fa.gz")) + 
                                    list(data_dir.glob("*.fasta")) + list(data_dir.glob("*.fasta.gz"))) 
                        if not f.name.startswith("Homo_sapiens.GRCh38")]
    
    # Report Ensembl files (the important ones)
    if ensembl_gtf_files:
        print(f"✅ Ensembl GTF files found: {len(ensembl_gtf_files)} files")
        for f in ensembl_gtf_files:
            print(f"   - {f.name}")
    else:
        print("❌ No Ensembl GTF files found (Homo_sapiens.GRCh38.*.gtf)")
    
    if ensembl_fasta_files:
        print(f"✅ Ensembl FASTA files found: {len(ensembl_fasta_files)} files")
        for f in ensembl_fasta_files:
            print(f"   - {f.name}")
    else:
        print("❌ No Ensembl FASTA files found (Homo_sapiens.GRCh38.dna.primary_assembly.fa)")
    
    # Report other files (examples, etc.)
    if other_gtf_files:
        print(f"📝 Other GTF files found: {len(other_gtf_files)} files")
        for f in other_gtf_files[:3]:  # Show first 3
            print(f"   - {f.name}")
    
    if other_fasta_files:
        print(f"📝 Other FASTA files found: {len(other_fasta_files)} files")
        for f in other_fasta_files[:3]:  # Show first 3
            print(f"   - {f.name}")
    
    # Check SpliceAI evaluation data
    print("\n🔬 SpliceAI Evaluation Data:")
    spliceai_eval_dir = ensembl_dir / "spliceai_eval"
    check_directory_exists(spliceai_eval_dir, "SpliceAI evaluation directory")
    
    if spliceai_eval_dir.exists():
        # Count analysis files
        analysis_files = list(spliceai_eval_dir.glob("**/*_analysis_sequences_*.tsv"))
        print(f"   Analysis sequence files: {len(analysis_files)}")
        
        if analysis_files:
            print(f"   Sample files:")
            for f in analysis_files[:3]:
                print(f"   - {f.relative_to(spliceai_eval_dir)}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY:")
    print("=" * 80)
    
    if all_derived_exist:
        print("✅ All derived datasets are present")
    else:
        print("❌ Some derived datasets are missing")
    
    if ensembl_gtf_files and ensembl_fasta_files:
        print("✅ Ensembl source GTF/FASTA files are present")
    else:
        print("❌ Ensembl source GTF/FASTA files are missing")
    
    if spliceai_eval_dir.exists() and analysis_files:
        print("✅ SpliceAI evaluation data is present")
    else:
        print("❌ SpliceAI evaluation data is missing")
    
    # Overall status
    if all_derived_exist and ensembl_gtf_files and ensembl_fasta_files:
        print("\n🎉 READY: All critical resources are present!")
        return 0
    else:
        print("\n⚠️  INCOMPLETE: Some resources are missing")
        return 1

if __name__ == "__main__":
    sys.exit(main())
