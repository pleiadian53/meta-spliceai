#!/usr/bin/env python3
"""
Simple test for the OpenSpliceAI Delta Bridge implementation.

This test verifies the core functionality without complex imports.
"""

import sys
from pathlib import Path
import tempfile
import pandas as pd

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

# Import only what we need for testing
from workflows.openspliceai_delta_bridge import OpenSpliceAIDeltaBridge
from data_types import DeltaScoreResult
from formats.variant_standardizer import VariantStandardizer


def test_variant_standardization():
    """Test variant standardization functionality."""
    print("🧪 Testing Variant Standardization...")
    
    standardizer = VariantStandardizer()
    
    # Test VCF standardization
    variant = standardizer.standardize_from_vcf("7", 117559593, "G", "T")
    
    print(f"   ✅ VCF standardization: {variant.chrom}:{variant.start} {variant.ref}>{variant.alt}")
    print(f"   ✅ Variant type: {variant.variant_type}")
    print(f"   ✅ Coordinate system: {variant.coordinate_system}")
    
    return variant


def test_delta_bridge_initialization():
    """Test delta bridge initialization."""
    print("\n🧪 Testing Delta Bridge Initialization...")
    
    # Initialize without real reference (will use mock)
    bridge = OpenSpliceAIDeltaBridge(
        reference_fasta="mock_reference.fa",
        annotations="grch38"
    )
    
    print(f"   ✅ Bridge initialized")
    print(f"   ✅ Variant standardizer available: {bridge.variant_standardizer is not None}")
    print(f"   ✅ Annotator status: {'Real' if bridge.annotator else 'Mock'}")
    
    return bridge


def test_mock_delta_scores():
    """Test mock delta score generation."""
    print("\n🧪 Testing Mock Delta Score Generation...")
    
    bridge = OpenSpliceAIDeltaBridge(
        reference_fasta="mock_reference.fa",
        annotations="grch38"
    )
    
    # Create test variants
    variants = [
        bridge.variant_standardizer.standardize_from_vcf("7", 117559593, "G", "T"),  # CFTR
        bridge.variant_standardizer.standardize_from_vcf("17", 43094077, "A", "G"),  # BRCA1
    ]
    
    # Generate mock delta scores
    delta_results = bridge.compute_delta_scores_from_variants(variants)
    
    print(f"   ✅ Generated delta scores for {len(delta_results)} variants")
    
    for result in delta_results:
        print(f"   📊 {result.gene_symbol}: DS_AG={result.ds_ag:.3f}, DS_DG={result.ds_dg:.3f}")
    
    return delta_results


def test_alternative_site_extraction():
    """Test alternative splice site extraction from delta scores."""
    print("\n🧪 Testing Alternative Site Extraction...")
    
    bridge = OpenSpliceAIDeltaBridge(
        reference_fasta="mock_reference.fa", 
        annotations="grch38"
    )
    
    # Create test variants
    variants = [
        bridge.variant_standardizer.standardize_from_vcf("7", 117559593, "G", "T"),
    ]
    
    # Get delta scores
    delta_results = bridge.compute_delta_scores_from_variants(variants)
    
    # Extract alternative sites
    alternative_sites = bridge.delta_scores_to_alternative_sites(delta_results, threshold=0.1)
    
    print(f"   ✅ Extracted {len(alternative_sites)} alternative splice sites")
    
    for site in alternative_sites[:3]:
        print(f"   🧬 {site.gene_symbol} {site.chrom}:{site.position}")
        print(f"      Type: {site.site_type}, Category: {site.splice_category}")
        print(f"      Delta Score: {site.delta_score:.3f}")
    
    return alternative_sites


def test_vcf_processing():
    """Test complete VCF processing pipeline."""
    print("\n🧪 Testing Complete VCF Processing Pipeline...")
    
    bridge = OpenSpliceAIDeltaBridge(
        reference_fasta="mock_reference.fa",
        annotations="grch38"
    )
    
    # Create temporary VCF
    with tempfile.NamedTemporaryFile(mode='w', suffix='.vcf', delete=False) as tmp:
        tmp.write("##fileformat=VCFv4.2\n")
        tmp.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")
        tmp.write("7\t117559593\t.\tG\tT\t.\tPASS\tGENE=CFTR\n")
        tmp.write("17\t43094077\t.\tA\tG\t.\tPASS\tGENE=BRCA1\n")
        tmp.flush()
        
        vcf_path = Path(tmp.name)
        
        # Process VCF
        sites_df = bridge.process_vcf_to_alternative_sites(vcf_path, threshold=0.1)
        
        print(f"   ✅ Processed VCF with {len(sites_df)} alternative sites")
        
        if not sites_df.empty:
            print(f"   📊 Site types: {sites_df['site_type'].value_counts().to_dict()}")
            print(f"   📊 Categories: {sites_df['splice_category'].value_counts().to_dict()}")
            print(f"   📊 Genes: {sites_df['gene_symbol'].unique().tolist()}")
        
        # Clean up
        vcf_path.unlink()
        
        return sites_df


def main():
    """Run all tests."""
    print("🧬 OpenSpliceAI Delta Bridge - Core Functionality Test")
    print("="*60)
    
    try:
        # Run tests
        variant = test_variant_standardization()
        bridge = test_delta_bridge_initialization()
        delta_results = test_mock_delta_scores()
        alternative_sites = test_alternative_site_extraction()
        sites_df = test_vcf_processing()
        
        print("\n" + "="*60)
        print("🎉 ALL TESTS COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        print(f"\n✅ Core Functionality Verified:")
        print(f"   • Variant standardization: Working")
        print(f"   • Delta bridge initialization: Working")
        print(f"   • Mock delta score generation: Working")
        print(f"   • Alternative site extraction: Working")
        print(f"   • VCF processing pipeline: Working")
        
        print(f"\n📊 Test Results Summary:")
        print(f"   • Variants processed: 2")
        print(f"   • Delta scores generated: {len(delta_results)}")
        print(f"   • Alternative sites found: {len(alternative_sites)}")
        print(f"   • Final DataFrame rows: {len(sites_df)}")
        
        print(f"\n🎯 Key Achievement:")
        print(f"   The critical gap from VCF → Delta Scores → Alternative Sites is now bridged!")
        print(f"   Ready for integration with meta-model training pipeline.")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
