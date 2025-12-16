#!/usr/bin/env python3
"""
Variant Analysis Example using AlignedSpliceExtractor

This script demonstrates how to use the AlignedSpliceExtractor for
variant analysis with external databases like ClinVar and SpliceVarDB.

Key features demonstrated:
1. Coordinate system reconciliation for different variant databases
2. Splice site extraction with coordinate alignment
3. Variant impact assessment with proper coordinate handling
4. Integration with meta-learning workflows

This is essential for accurate mutation impact analysis where even
1-2nt coordinate differences can lead to false results.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Tuple

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from meta_spliceai.splice_engine.meta_models.openspliceai_adapter.aligned_splice_extractor import (
    AlignedSpliceExtractor,
    extract_aligned_splice_sites,
    reconcile_variant_coordinates_from_clinvar
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def create_mock_clinvar_data() -> pd.DataFrame:
    """Create mock ClinVar data for demonstration."""
    
    return pd.DataFrame([
        {
            'variant_id': 'ClinVar_001',
            'gene_symbol': 'BRCA1',
            'gene_id': 'ENSG00000012048',
            'chromosome': '17',
            'position': 43094077,  # Known BRCA1 splice site variant
            'strand': '-',
            'splice_type': 'donor',
            'ref_allele': 'G',
            'alt_allele': 'A',
            'clinical_significance': 'Pathogenic',
            'review_status': '4_stars'
        },
        {
            'variant_id': 'ClinVar_002',
            'gene_symbol': 'BRCA2', 
            'gene_id': 'ENSG00000139618',
            'chromosome': '13',
            'position': 32339151,  # Known BRCA2 splice site variant
            'strand': '+',
            'splice_type': 'acceptor',
            'ref_allele': 'C',
            'alt_allele': 'T',
            'clinical_significance': 'Likely_pathogenic',
            'review_status': '3_stars'
        },
        {
            'variant_id': 'ClinVar_003',
            'gene_symbol': 'TP53',
            'gene_id': 'ENSG00000141510', 
            'chromosome': '17',
            'position': 7674220,  # Known TP53 splice site variant
            'strand': '-',
            'splice_type': 'acceptor',
            'ref_allele': 'G',
            'alt_allele': 'C',
            'clinical_significance': 'Pathogenic',
            'review_status': '4_stars'
        }
    ])


def create_mock_splicevardb_data() -> pd.DataFrame:
    """Create mock SpliceVarDB data for demonstration."""
    
    return pd.DataFrame([
        {
            'variant_id': 'SVD_001',
            'gene_symbol': 'CFTR',
            'gene_id': 'ENSG00000001626',
            'chromosome': '7',
            'position': 117559593,  # Known CFTR splice variant
            'strand': '+',
            'splice_type': 'donor',
            'ref_allele': 'G',
            'alt_allele': 'T',
            'splice_effect': 'abolish_donor',
            'evidence_level': 'strong'
        },
        {
            'variant_id': 'SVD_002',
            'gene_symbol': 'DMD',
            'gene_id': 'ENSG00000198947',
            'chromosome': 'X',
            'position': 31137344,  # Known DMD splice variant
            'strand': '-',
            'splice_type': 'acceptor',
            'ref_allele': 'A',
            'alt_allele': 'G',
            'splice_effect': 'weaken_acceptor',
            'evidence_level': 'moderate'
        }
    ])


def demonstrate_coordinate_reconciliation():
    """Demonstrate coordinate reconciliation for variant databases."""
    
    print("🔄 DEMONSTRATING COORDINATE RECONCILIATION")
    print("=" * 60)
    
    # Create mock variant data
    clinvar_data = create_mock_clinvar_data()
    splicevardb_data = create_mock_splicevardb_data()
    
    print(f"ClinVar variants: {len(clinvar_data)}")
    print(f"SpliceVarDB variants: {len(splicevardb_data)}")
    
    # Initialize extractor
    extractor = AlignedSpliceExtractor(verbosity=1)
    
    # Reconcile ClinVar coordinates
    print("\n📊 Reconciling ClinVar coordinates...")
    clinvar_reconciled = extractor.reconcile_variant_coordinates(
        variant_df=clinvar_data,
        source_system="clinvar",
        target_system="splicesurveyor"
    )
    
    # Reconcile SpliceVarDB coordinates
    print("\n📊 Reconciling SpliceVarDB coordinates...")
    splicevardb_reconciled = extractor.reconcile_variant_coordinates(
        variant_df=splicevardb_data,
        source_system="splicevardb",
        target_system="splicesurveyor"
    )
    
    # Show reconciliation results
    print("\n📋 RECONCILIATION RESULTS")
    print("-" * 40)
    
    print("ClinVar variants (before → after):")
    for _, row in clinvar_reconciled.iterrows():
        original_pos = clinvar_data.loc[clinvar_data['variant_id'] == row['variant_id'], 'position'].iloc[0]
        adjustment = row['position'] - original_pos
        print(f"  {row['variant_id']}: {original_pos} → {row['position']} ({adjustment:+d}nt)")
    
    print("\nSpliceVarDB variants (before → after):")
    for _, row in splicevardb_reconciled.iterrows():
        original_pos = splicevardb_data.loc[splicevardb_data['variant_id'] == row['variant_id'], 'position'].iloc[0]
        adjustment = row['position'] - original_pos
        print(f"  {row['variant_id']}: {original_pos} → {row['position']} ({adjustment:+d}nt)")
    
    return clinvar_reconciled, splicevardb_reconciled


def demonstrate_splice_site_extraction():
    """Demonstrate splice site extraction with coordinate alignment."""
    
    print("\n🧬 DEMONSTRATING SPLICE SITE EXTRACTION")
    print("=" * 60)
    
    gtf_file = "/home/bchiu/work/splice-surveyor/data/ensembl/Homo_sapiens.GRCh38.112.gtf"
    fasta_file = "/home/bchiu/work/splice-surveyor/data/ensembl/Homo_sapiens.GRCh38.dna.primary_assembly.fa"
    
    # Extract splice sites for genes of interest (cancer-related genes)
    cancer_genes = [
        "ENSG00000012048",  # BRCA1
        "ENSG00000139618",  # BRCA2  
        "ENSG00000141510"   # TP53
    ]
    
    print(f"Extracting splice sites for {len(cancer_genes)} cancer genes...")
    
    # Extract with MetaSpliceAI coordinates (reference)
    print("\n📊 Extracting with MetaSpliceAI coordinates...")
    extractor_ss = AlignedSpliceExtractor(
        coordinate_system="splicesurveyor",
        enable_biotype_filtering=False,
        verbosity=1
    )
    splice_sites_ss = extractor_ss.extract_splice_sites(gtf_file, fasta_file, cancer_genes)
    
    # Extract with OpenSpliceAI coordinates (for comparison)
    print("\n📊 Extracting with OpenSpliceAI coordinates...")
    extractor_osai = AlignedSpliceExtractor(
        coordinate_system="openspliceai",
        enable_biotype_filtering=False,
        verbosity=1
    )
    splice_sites_osai = extractor_osai.extract_splice_sites(gtf_file, fasta_file, cancer_genes)
    
    # Compare extraction results
    print(f"\n📋 EXTRACTION RESULTS")
    print("-" * 40)
    print(f"MetaSpliceAI sites: {len(splice_sites_ss)}")
    print(f"OpenSpliceAI sites: {len(splice_sites_osai)}")
    
    # Show coordinate differences
    if len(splice_sites_ss) == len(splice_sites_osai):
        print("\n🔍 Coordinate system differences:")
        
        # Sample a few sites for demonstration
        sample_sites = splice_sites_ss.head(5)
        for _, site in sample_sites.iterrows():
            # Find corresponding site in OpenSpliceAI data
            mask = (
                (splice_sites_osai['gene_id'] == site['gene_id']) &
                (splice_sites_osai['transcript_id'] == site['transcript_id']) &
                (splice_sites_osai['splice_type'] == site['splice_type']) &
                (splice_sites_osai['strand'] == site['strand'])
            )
            
            if mask.sum() > 0:
                osai_site = splice_sites_osai.loc[mask].iloc[0]
                coord_diff = osai_site['position'] - site['position']
                print(f"  {site['gene_id']} {site['splice_type']}_{site['strand']}: {coord_diff:+d}nt")
    
    return splice_sites_ss, splice_sites_osai


def demonstrate_variant_impact_analysis(splice_sites: pd.DataFrame, 
                                      variants: pd.DataFrame) -> pd.DataFrame:
    """Demonstrate variant impact analysis with proper coordinate handling."""
    
    print("\n🎯 DEMONSTRATING VARIANT IMPACT ANALYSIS")
    print("=" * 60)
    
    # Find variants that overlap with known splice sites
    variant_impacts = []
    
    for _, variant in variants.iterrows():
        print(f"\nAnalyzing variant: {variant['variant_id']} ({variant['gene_symbol']})")
        
        # Find splice sites in the same gene
        gene_splice_sites = splice_sites[splice_sites['gene_id'] == variant['gene_id']]
        
        if len(gene_splice_sites) == 0:
            print(f"  ⚠️  No splice sites found for gene {variant['gene_id']}")
            continue
        
        # Find exact matches (variant at splice site)
        exact_matches = gene_splice_sites[
            (gene_splice_sites['chromosome'] == variant['chromosome']) &
            (gene_splice_sites['position'] == variant['position']) &
            (gene_splice_sites['strand'] == variant['strand']) &
            (gene_splice_sites['splice_type'] == variant['splice_type'])
        ]
        
        # Find nearby sites (within 10nt)
        nearby_sites = gene_splice_sites[
            (gene_splice_sites['chromosome'] == variant['chromosome']) &
            (gene_splice_sites['strand'] == variant['strand']) &
            (abs(gene_splice_sites['position'] - variant['position']) <= 10)
        ]
        
        # Determine impact
        if len(exact_matches) > 0:
            impact_type = "direct_splice_site"
            impact_severity = "high"
            distance = 0
            print(f"  🎯 DIRECT HIT: Variant directly affects splice site")
        elif len(nearby_sites) > 0:
            impact_type = "splice_region"
            impact_severity = "moderate"
            distance = nearby_sites['position'].sub(variant['position']).abs().min()
            print(f"  📍 NEARBY: Closest splice site at {distance}nt distance")
        else:
            impact_type = "no_splice_impact"
            impact_severity = "low"
            distance = None
            print(f"  ✅ No direct splice site impact detected")
        
        # Record impact analysis
        variant_impact = {
            'variant_id': variant['variant_id'],
            'gene_id': variant['gene_id'],
            'gene_symbol': variant['gene_symbol'],
            'impact_type': impact_type,
            'impact_severity': impact_severity,
            'distance_to_splice_site': distance,
            'exact_matches': len(exact_matches),
            'nearby_sites': len(nearby_sites),
            'total_gene_splice_sites': len(gene_splice_sites)
        }
        
        variant_impacts.append(variant_impact)
    
    impact_df = pd.DataFrame(variant_impacts)
    
    # Summary
    print(f"\n📋 VARIANT IMPACT SUMMARY")
    print("-" * 40)
    print(f"Total variants analyzed: {len(impact_df)}")
    
    if len(impact_df) > 0:
        print(f"Direct splice site hits: {(impact_df['impact_type'] == 'direct_splice_site').sum()}")
        print(f"Splice region variants: {(impact_df['impact_type'] == 'splice_region').sum()}")
        print(f"No splice impact: {(impact_df['impact_type'] == 'no_splice_impact').sum()}")
    else:
        print("No variants could be analyzed (missing gene data)")
    
    return impact_df


def demonstrate_coordinate_discrepancy_detection():
    """Demonstrate systematic coordinate discrepancy detection."""
    
    print("\n🔍 DEMONSTRATING COORDINATE DISCREPANCY DETECTION")
    print("=" * 60)
    
    gtf_file = "/home/bchiu/work/splice-surveyor/data/ensembl/Homo_sapiens.GRCh38.112.gtf"
    fasta_file = "/home/bchiu/work/splice-surveyor/data/ensembl/Homo_sapiens.GRCh38.dna.primary_assembly.fa"
    
    test_genes = ["ENSG00000012048", "ENSG00000139618"]  # BRCA1, BRCA2
    
    # Extract sites with different coordinate systems
    extractor = AlignedSpliceExtractor(verbosity=1)
    
    sites_ss = extractor.extract_splice_sites(
        gtf_file, fasta_file, test_genes, output_format="dataframe"
    )
    
    # Simulate sites from another system (with known offsets)
    sites_other = sites_ss.copy()
    sites_other.loc[sites_other['splice_type'] == 'donor', 'position'] -= 1
    sites_other.loc[sites_other['splice_type'] == 'acceptor', 'position'] += 1
    sites_other['coordinate_system'] = 'other_system'
    
    # Detect discrepancies
    print("🔍 Detecting coordinate discrepancies...")
    discrepancy_report = extractor.detect_coordinate_discrepancies(
        reference_sites=sites_ss,
        comparison_sites=sites_other,
        reference_system="splicesurveyor",
        comparison_system="other_system"
    )
    
    print(f"\n📋 DISCREPANCY DETECTION RESULTS")
    print("-" * 40)
    print(f"Reference sites: {discrepancy_report['reference_sites_count']}")
    print(f"Comparison sites: {discrepancy_report['comparison_sites_count']}")
    
    if 'detected_offsets' in discrepancy_report:
        print("\nDetected coordinate offsets:")
        for splice_strand, offset_info in discrepancy_report['detected_offsets'].items():
            print(f"  {splice_strand}: {offset_info['offset']:+d}nt (confidence: {offset_info['confidence']:.1%})")
    
    return discrepancy_report


def main():
    """Run comprehensive variant analysis demonstration."""
    
    print("🎯 VARIANT ANALYSIS WITH ALIGNED SPLICE EXTRACTOR")
    print("=" * 80)
    print("Demonstrating coordinate reconciliation for ClinVar and SpliceVarDB")
    print("=" * 80)
    
    try:
        # Step 1: Coordinate reconciliation
        clinvar_reconciled, splicevardb_reconciled = demonstrate_coordinate_reconciliation()
        
        # Step 2: Splice site extraction
        splice_sites_ss, splice_sites_osai = demonstrate_splice_site_extraction()
        
        # Step 3: Variant impact analysis
        print("\n" + "=" * 60)
        print("VARIANT IMPACT ANALYSIS")
        print("=" * 60)
        
        # Analyze ClinVar variants
        print("\n🧬 ClinVar Variant Impact Analysis:")
        clinvar_impacts = demonstrate_variant_impact_analysis(splice_sites_ss, clinvar_reconciled)
        
        # Analyze SpliceVarDB variants  
        print("\n🧬 SpliceVarDB Variant Impact Analysis:")
        splicevardb_impacts = demonstrate_variant_impact_analysis(splice_sites_ss, splicevardb_reconciled)
        
        # Step 4: Coordinate discrepancy detection
        discrepancy_report = demonstrate_coordinate_discrepancy_detection()
        
        # Final summary
        print("\n" + "=" * 80)
        print("🎉 DEMONSTRATION COMPLETE")
        print("=" * 80)
        print("✅ Coordinate reconciliation: Successfully demonstrated")
        print("✅ Splice site extraction: Successfully demonstrated")
        print("✅ Variant impact analysis: Successfully demonstrated")
        print("✅ Discrepancy detection: Successfully demonstrated")
        print("\n🎯 The AlignedSpliceExtractor is ready for production variant analysis!")
        print("🔬 Key benefits:")
        print("  • Systematic coordinate reconciliation")
        print("  • Support for multiple variant databases")
        print("  • Accurate splice site impact assessment")
        print("  • Robust handling of coordinate discrepancies")
        
        return True
        
    except Exception as e:
        print(f"\n❌ DEMONSTRATION FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
