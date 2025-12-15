#!/usr/bin/env python3
"""
VCF Coordinate Verification Tutorial

This tutorial demonstrates how to verify VCF coordinates against reference FASTA
sequences, helping you understand coordinate systems and catch potential issues.

Key Learning Points:
1. VCF coordinates are 1-based, inclusive
2. pyfaidx uses 0-based indexing internally
3. Always verify REF alleles match the reference FASTA
4. Chromosome naming can vary (chr1 vs 1)
"""

import sys
from pathlib import Path
from pyfaidx import Fasta
import pysam

# Add the tools directory to path
sys.path.append(str(Path(__file__).parent.parent / "tools"))
from vcf_coordinate_verifier import VCFCoordinateVerifier


def basic_coordinate_check():
    """Basic example: verify a single position."""
    print("="*60)
    print("BASIC COORDINATE VERIFICATION")
    print("="*60)
    
    # Example from your question: chr1:94,062,595 G→A
    fasta_path = "data/ensembl/Homo_sapiens.GRCh38.dna.primary_assembly.fa"
    
    try:
        # Method 1: Direct pyfaidx usage
        print("\n1. Direct pyfaidx verification:")
        fasta = Fasta(fasta_path)
        
        # VCF position 94062595 (1-based) → pyfaidx index 94062594 (0-based)
        chrom = "1"
        vcf_pos = 94062595
        expected_ref = "G"
        
        # Extract the base (convert 1-based VCF to 0-based pyfaidx)
        actual_ref = str(fasta[chrom][vcf_pos-1:vcf_pos]).upper()
        
        print(f"  Chromosome: {chrom}")
        print(f"  VCF position: {vcf_pos} (1-based)")
        print(f"  pyfaidx index: {vcf_pos-1}:{vcf_pos} (0-based)")
        print(f"  Expected REF: {expected_ref}")
        print(f"  Actual REF: {actual_ref}")
        print(f"  Match: {actual_ref == expected_ref}")
        
        # Show context
        context_start = vcf_pos - 11
        context_end = vcf_pos + 10
        context = str(fasta[chrom][context_start-1:context_end]).upper()
        marked_context = context[:10] + f"[{context[10]}]" + context[11:]
        print(f"  Context: {marked_context}")
        
        # Method 2: Using our verification tool
        print("\n2. Using VCFCoordinateVerifier:")
        verifier = VCFCoordinateVerifier(fasta_path)
        result = verifier.verify_single_position(chrom, vcf_pos, expected_ref)
        
        for key, value in result.items():
            print(f"  {key}: {value}")
            
    except FileNotFoundError:
        print(f"⚠️  FASTA file not found: {fasta_path}")
        print("Please ensure the reference FASTA is available.")
    except Exception as e:
        print(f"❌ Error: {e}")


def vcf_batch_verification():
    """Verify multiple variants from a VCF file."""
    print("\n" + "="*60)
    print("BATCH VCF VERIFICATION")
    print("="*60)
    
    fasta_path = "data/ensembl/Homo_sapiens.GRCh38.dna.primary_assembly.fa"
    vcf_path = "data/ensembl/clinvar/vcf/clinvar_20250831_main_chroms.vcf.gz"
    
    try:
        verifier = VCFCoordinateVerifier(fasta_path)
        
        print(f"\nVerifying first 5 variants from: {vcf_path}")
        results_df = verifier.verify_vcf_variants(vcf_path, max_variants=5)
        
        # Display results
        print("\nVerification Results:")
        print("-" * 80)
        for _, row in results_df.iterrows():
            status_icon = "✅" if row['match'] else "❌"
            print(f"{status_icon} {row['chromosome']}:{row['position']} "
                  f"{row['expected_ref']}→{row['vcf_alt']} "
                  f"(Expected: {row['expected_ref']}, Actual: {row['actual_ref']})")
            if 'context' in row:
                print(f"   Context: {row['context']}")
        
        # Summary
        matches = sum(results_df['match'])
        total = len(results_df)
        print(f"\nSummary: {matches}/{total} variants verified successfully")
        
        if matches != total:
            print("⚠️  Some variants failed verification!")
            print("This could indicate:")
            print("  - Coordinate system mismatch")
            print("  - Wrong reference genome version")
            print("  - VCF preprocessing issues")
        
    except FileNotFoundError as e:
        print(f"⚠️  File not found: {e}")
        print("Please ensure both FASTA and VCF files are available.")
    except Exception as e:
        print(f"❌ Error: {e}")


def coordinate_system_explanation():
    """Explain coordinate system differences."""
    print("\n" + "="*60)
    print("COORDINATE SYSTEM EXPLANATION")
    print("="*60)
    
    print("""
Key Coordinate System Facts:

1. VCF Format (1-based, inclusive):
   - Position 94062595 refers to the 94,062,595th base
   - REF/ALT alleles start at this position
   - This is what you see in genome browsers

2. pyfaidx/Python (0-based, half-open):
   - Position 94062594:94062595 extracts the same base
   - Index 94062594 is the 94,062,595th base (0-based counting)
   - slice [start:end) excludes the end position

3. Verification Process:
   ✓ Extract REF allele from VCF (1-based position)
   ✓ Convert to 0-based for pyfaidx: pos-1
   ✓ Extract same position from FASTA
   ✓ Compare: they should match exactly

4. Common Issues:
   ❌ Chromosome naming: "chr1" vs "1"
   ❌ Wrong reference genome version
   ❌ Coordinate system confusion
   ❌ Strand orientation (shouldn't affect REF allele)

5. Why This Matters for SpliceAI/MetaSpliceAI:
   - WT sequences must use correct reference bases
   - ALT sequences apply variants to correct positions
   - Coordinate errors lead to wrong splice predictions
    """)


def strand_orientation_demo():
    """Demonstrate strand orientation concepts."""
    print("\n" + "="*60)
    print("STRAND ORIENTATION DEMO")
    print("="*60)
    
    print("""
Understanding Strand Orientation:

1. VCF REF/ALT are ALWAYS on the forward strand:
   - REF allele matches the reference FASTA exactly
   - ALT allele is the variant on the forward strand
   - No strand flipping in VCF format

2. Genome Browser Display:
   - May show variants on either strand for visualization
   - UCSC might display "G→A" or "C→T" for the same variant
   - This is just a display choice, not the actual data

3. For SpliceAI Analysis:
   - Always use VCF REF/ALT directly
   - Don't worry about browser strand display
   - Trust the VCF coordinates and alleles

Example:
  VCF: chr1:100 G→A (forward strand)
  Browser might show: G→A (forward view) or C→T (reverse view)
  SpliceAI input: Use G→A from VCF, ignore browser display
    """)


def practical_workflow():
    """Show practical workflow for variant verification."""
    print("\n" + "="*60)
    print("PRACTICAL VERIFICATION WORKFLOW")
    print("="*60)
    
    print("""
Recommended Workflow for Variant Analysis:

1. Before Analysis:
   □ Verify VCF coordinates against reference FASTA
   □ Check chromosome naming consistency
   □ Confirm reference genome version matches

2. During Analysis:
   □ Use VCF REF/ALT alleles directly
   □ Apply coordinate conversion correctly (1-based → 0-based)
   □ Extract sequences with proper context windows

3. Quality Control:
   □ Spot-check variants in genome browser
   □ Verify REF alleles match reference
   □ Compare results across different tools

4. Troubleshooting:
   □ If REF doesn't match: check coordinate system
   □ If positions seem off: verify genome build
   □ If results differ: check strand handling

Command Examples:
  # Verify specific variant
  python vcf_coordinate_verifier.py --verify-position chr1:94062595:G:A --fasta GRCh38.fa
  
  # Batch verify VCF
  python vcf_coordinate_verifier.py --vcf clinvar.vcf.gz --fasta GRCh38.fa --variants 100
  
  # Generate detailed report
  python vcf_coordinate_verifier.py --vcf clinvar.vcf.gz --fasta GRCh38.fa --variants 1000 --output verification_report.tsv
    """)


def main():
    """Run the complete tutorial."""
    print("VCF COORDINATE VERIFICATION TUTORIAL")
    print("This tutorial helps you understand and verify VCF coordinates")
    print("against reference FASTA sequences.\n")
    
    # Run all tutorial sections
    coordinate_system_explanation()
    basic_coordinate_check()
    vcf_batch_verification()
    strand_orientation_demo()
    practical_workflow()
    
    print("\n" + "="*60)
    print("TUTORIAL COMPLETE")
    print("="*60)
    print("""
Next Steps:
1. Try the verification tool on your own VCF files
2. Use this workflow before running SpliceAI analysis
3. Always verify coordinates when results seem unexpected

Remember: Trust the VCF, verify with FASTA, ignore browser strand flips!
    """)


if __name__ == "__main__":
    main()
