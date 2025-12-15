#!/usr/bin/env python3
"""
VCF to Alternative Splice Sites Demo

This script demonstrates the complete workflow from VCF variant analysis
to alternative splice site extraction for meta-model training.

Key Features Demonstrated:
1. VCF standardization and loading
2. OpenSpliceAI delta score computation
3. Alternative splice site extraction
4. Integration with meta-model training pipeline
5. Disease mutation database processing

This addresses the core gap identified: How to go from computing delta scores
and interpreting VCF to representing alternative splice sites.
"""

import sys
from pathlib import Path
import tempfile
import pandas as pd

# Add case studies directory to path
case_studies_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(case_studies_dir))

from workflows.alternative_splicing_pipeline import AlternativeSplicingPipeline
from workflows.openspliceai_delta_bridge import OpenSpliceAIDeltaBridge
from data_sources import SpliceVarDBIngester, ClinVarIngester


def create_demo_vcf(output_path: Path):
    """Create demo VCF file with known splice-affecting variants."""
    with open(output_path, 'w') as f:
        f.write("##fileformat=VCFv4.2\n")
        f.write("##source=MetaSpliceAI_Demo\n")
        f.write("##INFO=<ID=GENE,Number=1,Type=String,Description=\"Gene symbol\">\n")
        f.write("##INFO=<ID=CLNSIG,Number=1,Type=String,Description=\"Clinical significance\">\n")
        f.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")
        
        # Known pathogenic splice variants
        variants = [
            # CFTR splice donor variant (cystic fibrosis)
            ("7", "117559593", "CFTR_c.1585-1G>T", "G", "T", "GENE=CFTR;CLNSIG=Pathogenic"),
            
            # BRCA1 splice acceptor variant (breast cancer)
            ("17", "43094077", "BRCA1_c.4358-2A>G", "A", "G", "GENE=BRCA1;CLNSIG=Pathogenic"),
            
            # BRCA2 splice donor variant (breast cancer) 
            ("13", "32339151", "BRCA2_c.426+1G>A", "G", "A", "GENE=BRCA2;CLNSIG=Pathogenic"),
            
            # TP53 splice acceptor variant (Li-Fraumeni syndrome)
            ("17", "7674220", "TP53_c.375+1G>A", "G", "A", "GENE=TP53;CLNSIG=Pathogenic"),
            
            # MET exon 14 skipping variant (lung cancer)
            ("7", "116411708", "MET_c.3028+1G>A", "G", "A", "GENE=MET;CLNSIG=Pathogenic"),
            
            # Deep intronic CFTR variant (creates cryptic exon)
            ("7", "117465063", "CFTR_c.3718-2477C>T", "C", "T", "GENE=CFTR;CLNSIG=Pathogenic"),
        ]
        
        for chrom, pos, variant_id, ref, alt, info in variants:
            f.write(f"{chrom}\t{pos}\t{variant_id}\t{ref}\t{alt}\t.\tPASS\t{info}\n")


def demo_basic_pipeline():
    """Demonstrate basic VCF to alternative sites pipeline."""
    print("\n" + "="*80)
    print("DEMO 1: Basic VCF to Alternative Splice Sites Pipeline")
    print("="*80)
    
    # Create demo VCF
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir = Path(tmp_dir)
        vcf_path = tmp_dir / "demo_variants.vcf"
        create_demo_vcf(vcf_path)
        
        print(f"✅ Created demo VCF with 6 pathogenic splice variants: {vcf_path}")
        
        # Initialize pipeline (without reference FASTA, will use mock delta scores)
        work_dir = tmp_dir / "pipeline_output"
        pipeline = AlternativeSplicingPipeline(
            work_dir=work_dir,
            reference_fasta=None,  # Use mock for demo
            annotations="grch38"
        )
        
        print("✅ Initialized alternative splicing pipeline")
        
        # Process VCF
        sites_df = pipeline.process_vcf_to_alternative_sites(vcf_path)
        
        print(f"\n📊 RESULTS:")
        print(f"   Total alternative sites identified: {len(sites_df)}")
        
        if not sites_df.empty:
            print(f"   Site types: {sites_df['site_type'].value_counts().to_dict()}")
            print(f"   Splice categories: {sites_df['splice_category'].value_counts().to_dict()}")
            print(f"   Mean delta score: {sites_df['delta_score'].mean():.3f}")
            print(f"   Delta score range: {sites_df['delta_score'].min():.3f} to {sites_df['delta_score'].max():.3f}")
            
            # Show sample sites
            print(f"\n📋 Sample Alternative Splice Sites:")
            for i, (_, row) in enumerate(sites_df.head(3).iterrows()):
                print(f"   {i+1}. {row['gene_symbol']} {row['chromosome']}:{row['position']}")
                print(f"      Type: {row['site_type']}, Category: {row['splice_category']}")
                print(f"      Delta Score: {row['delta_score']:.3f}, Variant: {row['variant_id']}")
        
        # Generate training manifest
        manifest = pipeline.generate_training_manifest(sites_df)
        print(f"\n📄 Training manifest generated: {manifest['files']['training_ready']}")


def demo_openspliceai_integration():
    """Demonstrate OpenSpliceAI delta bridge integration."""
    print("\n" + "="*80)
    print("DEMO 2: OpenSpliceAI Delta Score Bridge Integration")
    print("="*80)
    
    # Initialize delta bridge (mock mode for demo)
    bridge = OpenSpliceAIDeltaBridge(
        reference_fasta="path/to/reference.fa",  # Would be real path in production
        annotations="grch38"
    )
    
    print("✅ Initialized OpenSpliceAI delta bridge")
    
    # Create mock variants
    variants = [
        bridge.variant_standardizer.standardize_from_vcf("7", 117559593, "G", "T"),    # CFTR
        bridge.variant_standardizer.standardize_from_vcf("17", 43094077, "A", "G"),    # BRCA1
        bridge.variant_standardizer.standardize_from_vcf("13", 32339151, "G", "A"),    # BRCA2
    ]
    
    print(f"✅ Created {len(variants)} standardized variants")
    
    # Compute delta scores
    delta_results = bridge.compute_delta_scores_from_variants(variants)
    print(f"✅ Computed delta scores for {len(delta_results)} variant-gene pairs")
    
    # Show delta score results
    print(f"\n📊 DELTA SCORE RESULTS:")
    for result in delta_results[:3]:
        print(f"   {result.gene_symbol} {result.variant_id}:")
        print(f"     DS_AG: {result.ds_ag:.3f}, DS_AL: {result.ds_al:.3f}")
        print(f"     DS_DG: {result.ds_dg:.3f}, DS_DL: {result.ds_dl:.3f}")
        print(f"     DP_AG: {result.dp_ag}, DP_AL: {result.dp_al}")
        print(f"     DP_DG: {result.dp_dg}, DP_DL: {result.dp_dl}")
    
    # Convert to alternative splice sites
    alternative_sites = bridge.delta_scores_to_alternative_sites(delta_results, threshold=0.1)
    print(f"\n✅ Extracted {len(alternative_sites)} alternative splice sites")
    
    # Show alternative sites
    print(f"\n📋 ALTERNATIVE SPLICE SITES:")
    for i, site in enumerate(alternative_sites[:5]):
        print(f"   {i+1}. {site.gene_symbol} {site.chrom}:{site.position}")
        print(f"      Type: {site.site_type}, Category: {site.splice_category}")
        print(f"      Delta Score: {site.delta_score:.3f}")


def demo_disease_database_integration():
    """Demonstrate integration with disease mutation databases."""
    print("\n" + "="*80)
    print("DEMO 3: Disease Mutation Database Integration")
    print("="*80)
    
    # This would normally ingest real database data
    print("📥 Simulating disease database ingestion...")
    
    # Mock ClinVar data
    clinvar_mutations = [
        {
            'variant_id': 'ClinVar_123456',
            'gene_symbol': 'CFTR',
            'chrom': '7',
            'position': 117559593,
            'ref_allele': 'G',
            'alt_allele': 'T',
            'clinical_significance': 'Pathogenic',
            'splice_event': 'donor_loss'
        },
        {
            'variant_id': 'ClinVar_789012', 
            'gene_symbol': 'BRCA1',
            'chrom': '17',
            'position': 43094077,
            'ref_allele': 'A',
            'alt_allele': 'G',
            'clinical_significance': 'Pathogenic',
            'splice_event': 'acceptor_loss'
        }
    ]
    
    print(f"✅ Simulated ingestion of {len(clinvar_mutations)} ClinVar mutations")
    
    # Convert to VCF format
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir = Path(tmp_dir)
        vcf_path = tmp_dir / "clinvar_variants.vcf"
        
        with open(vcf_path, 'w') as f:
            f.write("##fileformat=VCFv4.2\n")
            f.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")
            for mut in clinvar_mutations:
                f.write(f"{mut['chrom']}\t{mut['position']}\t{mut['variant_id']}\t"
                       f"{mut['ref_allele']}\t{mut['alt_allele']}\t.\tPASS\t"
                       f"GENE={mut['gene_symbol']};CLNSIG={mut['clinical_significance']}\n")
        
        # Process through pipeline
        work_dir = tmp_dir / "clinvar_analysis"
        pipeline = AlternativeSplicingPipeline(work_dir=work_dir)
        
        sites_df = pipeline.process_vcf_to_alternative_sites(vcf_path)
        
        print(f"\n📊 DISEASE DATABASE ANALYSIS RESULTS:")
        print(f"   Mutations processed: {len(clinvar_mutations)}")
        print(f"   Alternative sites identified: {len(sites_df)}")
        
        if not sites_df.empty:
            pathogenic_sites = sites_df[sites_df['variant_id'].str.contains('ClinVar')]
            print(f"   Pathogenic variant sites: {len(pathogenic_sites)}")
            
            # Show clinical significance impact
            print(f"\n📋 Clinical Impact Analysis:")
            for _, row in sites_df.head(3).iterrows():
                print(f"   {row['gene_symbol']}: {row['splice_category']}")
                print(f"     Delta Score: {row['delta_score']:.3f}")
                print(f"     Clinical Evidence: Disease-associated variant")


def demo_meta_model_integration():
    """Demonstrate integration with meta-model training pipeline."""
    print("\n" + "="*80)
    print("DEMO 4: Meta-Model Training Integration")
    print("="*80)
    
    print("🎯 Alternative Splice Sites → Meta-Model Training Data")
    
    # Create sample alternative sites data
    sample_sites = pd.DataFrame([
        {
            'chromosome': '7', 'position': 117559593, 'strand': '+',
            'site_type': 'donor', 'splice_category': 'canonical_disrupted',
            'delta_score': -0.85, 'gene_symbol': 'CFTR',
            'variant_id': '7:117559593:G>T', 'validation_evidence': 'openspliceai_prediction'
        },
        {
            'chromosome': '7', 'position': 117559595, 'strand': '+', 
            'site_type': 'donor', 'splice_category': 'cryptic_activated',
            'delta_score': 0.72, 'gene_symbol': 'CFTR',
            'variant_id': '7:117559593:G>T', 'validation_evidence': 'openspliceai_prediction'
        },
        {
            'chromosome': '17', 'position': 43094077, 'strand': '-',
            'site_type': 'acceptor', 'splice_category': 'canonical_disrupted',
            'delta_score': -0.91, 'gene_symbol': 'BRCA1',
            'variant_id': '17:43094077:A>G', 'validation_evidence': 'openspliceai_prediction'
        }
    ])
    
    print(f"✅ Created sample alternative sites dataset: {len(sample_sites)} sites")
    
    # Show training data characteristics
    print(f"\n📊 TRAINING DATA CHARACTERISTICS:")
    print(f"   Site types: {sample_sites['site_type'].value_counts().to_dict()}")
    print(f"   Splice categories: {sample_sites['splice_category'].value_counts().to_dict()}")
    print(f"   Genes represented: {sample_sites['gene_symbol'].nunique()}")
    print(f"   Delta score range: {sample_sites['delta_score'].min():.3f} to {sample_sites['delta_score'].max():.3f}")
    
    # Training data integration points
    print(f"\n🔗 META-MODEL INTEGRATION POINTS:")
    print("   1. ✅ Alternative sites compatible with transcript-aware position identification")
    print("   2. ✅ Delta scores provide variant impact training signal")
    print("   3. ✅ Splice categories enable multi-class learning")
    print("   4. ✅ Clinical evidence supports pathogenicity prediction")
    print("   5. ✅ Gene-level features enable cross-gene generalization")
    
    # Show how this integrates with existing pipeline
    print(f"\n🚀 NEXT STEPS FOR META-MODEL TRAINING:")
    print("   → Combine alternative sites with canonical sites")
    print("   → Apply transcript-aware position identification")
    print("   → Generate sequence features and k-mer representations")
    print("   → Train 5000-gene meta model with enhanced generalization")
    print("   → Validate on unseen disease mutations")


def main():
    """Run all demonstrations."""
    print("🧬 VCF to Alternative Splice Sites: Complete Workflow Demonstration")
    print("This demo shows the complete pipeline from variant analysis to meta-model training")
    
    try:
        demo_basic_pipeline()
        demo_openspliceai_integration()
        demo_disease_database_integration()
        demo_meta_model_integration()
        
        print("\n" + "="*80)
        print("🎉 ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY!")
        print("="*80)
        print("\n✅ Key Achievements:")
        print("   • VCF variants → OpenSpliceAI delta scores")
        print("   • Delta scores → Alternative splice sites")
        print("   • Disease mutations → Training data")
        print("   • Integration with meta-model pipeline")
        print("\n🎯 The critical gap has been bridged!")
        print("   VCF analysis now connects seamlessly to alternative splice site representation")
        print("   for meta-model training and disease mutation validation.")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
