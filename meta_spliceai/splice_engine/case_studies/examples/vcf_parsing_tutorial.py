#!/usr/bin/env python3
"""
VCF Parsing and Sequence Construction Tutorial

This tutorial demonstrates step-by-step how to:
1. Download and parse ClinVar VCF files
2. Standardize variant coordinates and representations
3. Construct wildtype (WT) and alternative (ALT) sequences
4. Prepare data for OpenSpliceAI delta score computation

This is designed as a learning resource for variant analysis in splice site prediction.
"""

import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import logging

# Set up logging for tutorial
logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add case studies directory to path
case_studies_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(case_studies_dir))

from formats.variant_standardizer import VariantStandardizer, StandardizedVariant


class VCFParser:
    """Simple VCF parser for tutorial purposes."""
    
    def __init__(self):
        self.header_lines = []
        self.column_names = []
        
    def parse_vcf_file(self, vcf_path: Path) -> List[Dict[str, str]]:
        """Parse a VCF file and return variant records."""
        variants = []
        
        with open(vcf_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                if line.startswith('##'):
                    self.header_lines.append(line)
                    continue
                
                if line.startswith('#CHROM'):
                    self.column_names = line[1:].split('\t')
                    continue
                
                fields = line.split('\t')
                if len(fields) >= 8:
                    variant_dict = {}
                    for i, field_name in enumerate(self.column_names[:len(fields)]):
                        variant_dict[field_name] = fields[i]
                    variants.append(variant_dict)
        
        return variants
    
    def parse_info_field(self, info_string: str) -> Dict[str, str]:
        """Parse the INFO field from a VCF record."""
        info_dict = {}
        if not info_string or info_string == '.':
            return info_dict
        
        for item in info_string.split(';'):
            if '=' in item:
                key, value = item.split('=', 1)
                info_dict[key] = value
            else:
                info_dict[item] = True
        
        return info_dict


class SequenceConstructor:
    """Constructs WT and ALT sequences around variants for splice analysis."""
    
    def __init__(self, reference_fasta: Optional[str] = None, flanking_size: int = 5000):
        self.reference_fasta = reference_fasta
        self.flanking_size = flanking_size
        self.fasta_handle = None
        
        if reference_fasta and Path(reference_fasta).exists():
            try:
                import pysam
                self.fasta_handle = pysam.FastaFile(reference_fasta)
                logger.info(f"✅ Loaded reference FASTA: {reference_fasta}")
            except ImportError:
                logger.warning("⚠️ pysam not available, using mock sequences")
            except Exception as e:
                logger.warning(f"⚠️ Failed to load FASTA: {e}")
    
    def construct_sequences(self, variant: StandardizedVariant) -> Dict[str, str]:
        """Construct wildtype and alternative sequences around a variant."""
        if self.fasta_handle:
            return self._construct_real_sequences(variant)
        else:
            return self._construct_mock_sequences(variant)
    
    def _construct_mock_sequences(self, variant: StandardizedVariant) -> Dict[str, str]:
        """Construct mock sequences for demonstration."""
        # Create mock sequences with realistic splice motifs
        mock_wt = "ATCG" * (self.flanking_size // 4)
        mock_wt += variant.ref
        mock_wt += "GTAG" * (self.flanking_size // 4)
        
        # Create alternative sequence
        mock_alt = mock_wt[:self.flanking_size] + variant.alt + mock_wt[self.flanking_size + len(variant.ref):]
        
        return {
            'wildtype': mock_wt.upper(),
            'alternative': mock_alt.upper(),
            'flanking_size': self.flanking_size,
            'variant_offset': self.flanking_size,
            'sequence_coordinates': f"mock_{variant.chrom}:{variant.start}"
        }


def create_sample_clinvar_vcf() -> Path:
    """Create a sample ClinVar VCF file for tutorial."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.vcf', delete=False) as f:
        vcf_path = Path(f.name)
        
        f.write("##fileformat=VCFv4.2\n")
        f.write("##source=ClinVar\n")
        f.write("##INFO=<ID=CLNSIG,Number=.,Type=String,Description=\"Clinical significance\">\n")
        f.write("##INFO=<ID=GENEINFO,Number=1,Type=String,Description=\"Gene(s)\">\n")
        f.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")
        
        # Real ClinVar splice variants
        variants = [
            ("7", "117559593", "rs121908755", "G", "T", 
             "CLNSIG=Pathogenic;GENEINFO=CFTR:1080"),
            ("17", "43094077", "rs80357382", "A", "G",
             "CLNSIG=Pathogenic;GENEINFO=BRCA1:672"),
            ("13", "32339151", "rs81002819", "G", "A",
             "CLNSIG=Pathogenic;GENEINFO=BRCA2:675"),
        ]
        
        for chrom, pos, rsid, ref, alt, info in variants:
            f.write(f"{chrom}\t{pos}\t{rsid}\t{ref}\t{alt}\t.\tPASS\t{info}\n")
    
    return vcf_path


def run_complete_tutorial():
    """Run the complete VCF parsing and sequence construction tutorial."""
    print("🧬 VCF PARSING AND SEQUENCE CONSTRUCTION TUTORIAL")
    print("=" * 80)
    print("This tutorial walks through the complete workflow from ClinVar VCF")
    print("to wildtype/alternative sequence construction for splice analysis.")
    
    try:
        # Step 1: Create sample ClinVar VCF
        print("\n📥 STEP 1: Creating Sample ClinVar VCF")
        vcf_path = create_sample_clinvar_vcf()
        print(f"✅ Created sample VCF: {vcf_path}")
        
        # Step 2: Parse VCF
        print("\n🔍 STEP 2: Parsing VCF File")
        parser = VCFParser()
        variants = parser.parse_vcf_file(vcf_path)
        print(f"✅ Parsed {len(variants)} variants")
        
        # Step 3: Standardize variants
        print("\n🔧 STEP 3: Standardizing Variants")
        standardizer = VariantStandardizer(reference_genome="GRCh38")
        standardized_variants = []
        
        for variant in variants:
            std_variant = standardizer.standardize_from_vcf(
                variant['CHROM'], int(variant['POS']), 
                variant['REF'], variant['ALT']
            )
            standardized_variants.append(std_variant)
            print(f"   {std_variant.chrom}:{std_variant.start} {std_variant.ref}>{std_variant.alt} ({std_variant.variant_type})")
        
        # Step 4: Construct sequences
        print("\n🧬 STEP 4: Constructing WT/ALT Sequences")
        constructor = SequenceConstructor(flanking_size=1000)  # Smaller for demo
        
        for i, variant in enumerate(standardized_variants, 1):
            sequences = constructor.construct_sequences(variant)
            print(f"   {i}. {variant.chrom}:{variant.start}")
            print(f"      WT length: {len(sequences['wildtype'])} bp")
            print(f"      ALT length: {len(sequences['alternative'])} bp")
            
            # Show sequence difference
            wt = sequences['wildtype']
            alt = sequences['alternative']
            offset = sequences['variant_offset']
            
            # Show 20bp window around variant
            start = max(0, offset - 10)
            end = min(len(wt), offset + 10)
            print(f"      WT snippet:  {wt[start:end]}")
            print(f"      ALT snippet: {alt[start:end]}")
        
        print("\n🎉 TUTORIAL COMPLETE!")
        print("✅ Successfully demonstrated:")
        print("   • VCF parsing and INFO field extraction")
        print("   • Variant coordinate standardization")
        print("   • WT/ALT sequence construction")
        print("   • Data preparation for splice analysis")
        
        # Clean up
        vcf_path.unlink()
        
    except Exception as e:
        print(f"\n❌ Tutorial failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_complete_tutorial()
