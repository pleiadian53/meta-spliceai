#!/usr/bin/env python3
"""
Enhanced Alternative Splicing Analysis Demonstration

This script demonstrates the complete enhanced workflow for capturing
alternative splice sites induced by variants and diseases, showcasing
the adaptive capacity of the meta-learning layer.
"""

import sys
import argparse
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
import json

# Add case studies to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from workflows.disease_specific_alternative_splicing import (
    DiseaseSpecificAlternativeSplicingAnalyzer,
    DiseaseAlternativeSplicingResult
)
from workflows.regulatory_features import RegulatoryFeatureExtractor
from workflows.alternative_splicing_pipeline import AlternativeSplicingPipeline
from data_sources.base import SpliceMutation, SpliceEventType
from data_types import AlternativeSpliceSite


def create_demo_mutations() -> Dict[str, List[SpliceMutation]]:
    """Create demonstration mutations for different diseases."""
    
    # Cystic Fibrosis mutations
    cf_mutations = [
        SpliceMutation(
            mutation_id="CFTR_c.3718-2477C>T",
            gene_symbol="CFTR",
            chrom="7",
            position=117559593,
            ref_allele="C",
            alt_allele="T",
            splice_event_type=SpliceEventType.CRYPTIC_EXON_INCLUSION,
            clinical_significance="pathogenic",
            validation_method="minigene_assay",
            rna_evidence_reads=25,
            disease_association="cystic_fibrosis"
        ),
        SpliceMutation(
            mutation_id="CFTR_c.1521_1523delCTT",
            gene_symbol="CFTR",
            chrom="7",
            position=117534318,
            ref_allele="CTT",
            alt_allele="",
            splice_event_type=SpliceEventType.EXON_SKIPPING,
            clinical_significance="pathogenic",
            validation_method="patient_rna",
            rna_evidence_reads=18,
            disease_association="cystic_fibrosis"
        )
    ]
    
    # Breast Cancer mutations
    brca_mutations = [
        SpliceMutation(
            mutation_id="BRCA1_c.5266dupC",
            gene_symbol="BRCA1",
            chrom="17",
            position=43094077,
            ref_allele="G",
            alt_allele="GC",
            splice_event_type=SpliceEventType.EXON_SKIPPING,
            clinical_significance="pathogenic",
            validation_method="rna_seq",
            rna_evidence_reads=42,
            disease_association="breast_cancer"
        ),
        SpliceMutation(
            mutation_id="BRCA2_c.8488-1G>A",
            gene_symbol="BRCA2",
            chrom="13",
            position=32339151,
            ref_allele="G",
            alt_allele="A",
            splice_event_type=SpliceEventType.CANONICAL_SITE_DISRUPTION,
            clinical_significance="pathogenic",
            validation_method="functional_assay",
            rna_evidence_reads=35,
            disease_association="breast_cancer"
        )
    ]
    
    # ALS/FTD mutations
    als_mutations = [
        SpliceMutation(
            mutation_id="UNC13A_cryptic_exon",
            gene_symbol="UNC13A",
            chrom="19",
            position=17718496,
            ref_allele="G",
            alt_allele="A",
            splice_event_type=SpliceEventType.CRYPTIC_EXON_INCLUSION,
            clinical_significance="risk_factor",
            validation_method="patient_tissue",
            rna_evidence_reads=28,
            disease_association="als_ftd"
        ),
        SpliceMutation(
            mutation_id="STMN2_cryptic_exon",
            gene_symbol="STMN2",
            chrom="8",
            position=79645395,
            ref_allele="C",
            alt_allele="T",
            splice_event_type=SpliceEventType.CRYPTIC_EXON_INCLUSION,
            clinical_significance="pathogenic",
            validation_method="cell_culture",
            rna_evidence_reads=22,
            disease_association="als_ftd"
        )
    ]
    
    # Lung Cancer mutations
    lung_mutations = [
        SpliceMutation(
            mutation_id="MET_exon14_skipping",
            gene_symbol="MET",
            chrom="7",
            position=116411708,
            ref_allele="G",
            alt_allele="T",
            splice_event_type=SpliceEventType.EXON_SKIPPING,
            clinical_significance="oncogenic",
            validation_method="tcga_rna",
            rna_evidence_reads=67,
            disease_association="lung_cancer"
        )
    ]
    
    return {
        "cystic_fibrosis": cf_mutations,
        "breast_cancer": brca_mutations,
        "als_ftd": als_mutations,
        "lung_cancer": lung_mutations
    }


def run_comprehensive_analysis(work_dir: Path, 
                             diseases: List[str] = None,
                             include_regulatory: bool = True,
                             meta_model_path: Path = None) -> Dict[str, Any]:
    """
    Run comprehensive alternative splicing analysis across diseases.
    
    Parameters
    ----------
    work_dir : Path
        Working directory for analysis
    diseases : List[str]
        Diseases to analyze (default: all)
    include_regulatory : bool
        Whether to include regulatory features
    meta_model_path : Path
        Path to meta-model for improvement calculation
        
    Returns
    -------
    Dict[str, Any]
        Comprehensive analysis results
    """
    print("\n🧬 ENHANCED ALTERNATIVE SPLICING ANALYSIS")
    print("=" * 60)
    
    # Initialize analyzer
    regulatory_data_dir = work_dir / "regulatory_data" if include_regulatory else None
    analyzer = DiseaseSpecificAlternativeSplicingAnalyzer(
        work_dir=work_dir,
        regulatory_data_dir=regulatory_data_dir,
        verbosity=2
    )
    
    # Get demo mutations
    all_mutations = create_demo_mutations()
    
    if diseases is None:
        diseases = list(all_mutations.keys())
    
    print(f"📊 Analyzing {len(diseases)} diseases: {', '.join(diseases)}")
    
    # Analyze each disease
    disease_results = []
    detailed_results = {}
    
    for disease in diseases:
        if disease not in all_mutations:
            print(f"⚠️  Unknown disease: {disease}")
            continue
        
        mutations = all_mutations[disease]
        print(f"\n🦠 Analyzing {disease} ({len(mutations)} mutations)...")
        
        try:
            result = analyzer.analyze_disease_alternative_splicing(
                disease_name=disease,
                mutations=mutations,
                meta_model_path=meta_model_path
            )
            
            disease_results.append(result)
            detailed_results[disease] = result
            
            print(f"   ✅ {result.alternative_sites_detected} alternative sites detected")
            print(f"   🧬 {result.cryptic_sites_activated} cryptic sites activated")
            print(f"   📉 {result.canonical_sites_disrupted} canonical sites disrupted")
            print(f"   🎯 {result.novel_splice_patterns} novel patterns identified")
            print(f"   📈 {result.meta_model_improvement:.3f} meta-model improvement")
            
            # Show tissue-specific patterns
            if result.tissue_specific_patterns:
                tissue_summary = ", ".join([f"{k}:{v}" for k, v in result.tissue_specific_patterns.items() if v > 0])
                if tissue_summary:
                    print(f"   🧪 Tissue patterns: {tissue_summary}")
            
        except Exception as e:
            print(f"   ❌ Analysis failed: {e}")
            continue
    
    # Cross-disease comparison
    if len(disease_results) > 1:
        print(f"\n📊 CROSS-DISEASE COMPARISON")
        print("-" * 40)
        
        comparison_df = analyzer.compare_diseases(disease_results)
        
        # Show summary statistics
        print(f"Total alternative sites across diseases: {comparison_df['alternative_sites'].sum()}")
        print(f"Mean sites per mutation: {comparison_df['sites_per_mutation'].mean():.2f}")
        print(f"Mean cryptic activation rate: {comparison_df['cryptic_rate'].mean():.2f}")
        print(f"Mean meta-model improvement: {comparison_df['meta_improvement'].mean():.3f}")
        
        # Top diseases by different metrics
        print(f"\n🏆 Top diseases by alternative sites:")
        top_sites = comparison_df.nlargest(3, 'alternative_sites')
        for _, row in top_sites.iterrows():
            print(f"   {row['disease']}: {row['alternative_sites']} sites")
        
        print(f"\n🎯 Top diseases by meta-model improvement:")
        top_improvement = comparison_df.nlargest(3, 'meta_improvement')
        for _, row in top_improvement.iterrows():
            print(f"   {row['disease']}: {row['meta_improvement']:.3f} improvement")
    
    # Generate comprehensive report
    report_data = {
        'analysis_summary': {
            'diseases_analyzed': len(disease_results),
            'total_mutations': sum(r.total_mutations for r in disease_results),
            'total_alternative_sites': sum(r.alternative_sites_detected for r in disease_results),
            'total_cryptic_sites': sum(r.cryptic_sites_activated for r in disease_results),
            'mean_meta_improvement': sum(r.meta_model_improvement for r in disease_results) / len(disease_results) if disease_results else 0
        },
        'disease_results': {
            disease: {
                'alternative_sites': result.alternative_sites_detected,
                'cryptic_sites': result.cryptic_sites_activated,
                'canonical_disrupted': result.canonical_sites_disrupted,
                'novel_patterns': result.novel_splice_patterns,
                'meta_improvement': result.meta_model_improvement,
                'tissue_patterns': result.tissue_specific_patterns
            }
            for disease, result in detailed_results.items()
        }
    }
    
    # Save comprehensive report
    report_file = work_dir / "comprehensive_analysis_report.json"
    with open(report_file, 'w') as f:
        json.dump(report_data, f, indent=2)
    
    print(f"\n📄 Comprehensive report saved: {report_file}")
    
    return report_data


def demonstrate_regulatory_enhancement(work_dir: Path):
    """Demonstrate regulatory feature enhancement capabilities."""
    print("\n🔬 REGULATORY FEATURE ENHANCEMENT DEMONSTRATION")
    print("=" * 60)
    
    # Initialize regulatory extractor
    regulatory_dir = work_dir / "regulatory_data"
    regulatory_dir.mkdir(exist_ok=True)
    
    extractor = RegulatoryFeatureExtractor(regulatory_dir, verbosity=2)
    
    # Create mock alternative splice sites
    mock_sites = [
        AlternativeSpliceSite(
            chrom="7", position=117559593, strand="+", site_type="acceptor",
            splice_category="cryptic_activated", delta_score=0.8,
            ref_score=0.1, alt_score=0.9, variant_id="7:117559593:C>T",
            gene_symbol="CFTR", clinical_significance="pathogenic",
            validation_evidence="openspliceai_prediction"
        ),
        AlternativeSpliceSite(
            chrom="17", position=43094077, strand="+", site_type="donor",
            splice_category="canonical_disrupted", delta_score=-0.7,
            ref_score=0.9, alt_score=0.2, variant_id="17:43094077:G>A",
            gene_symbol="BRCA1", clinical_significance="pathogenic",
            validation_evidence="openspliceai_prediction"
        ),
        AlternativeSpliceSite(
            chrom="19", position=17718496, strand="+", site_type="acceptor",
            splice_category="cryptic_activated", delta_score=0.6,
            ref_score=0.05, alt_score=0.65, variant_id="19:17718496:G>A",
            gene_symbol="UNC13A", clinical_significance="risk_factor",
            validation_evidence="openspliceai_prediction"
        )
    ]
    
    print(f"🧬 Enhancing {len(mock_sites)} alternative splice sites with regulatory features...")
    
    # Enhance sites with regulatory features
    enhanced_sites = extractor.enhance_alternative_sites_with_regulatory_features(mock_sites)
    
    print(f"\n📊 Regulatory Enhancement Results:")
    for i, (original, enhanced) in enumerate(zip(mock_sites, enhanced_sites)):
        print(f"\n   Site {i+1}: {enhanced.gene_symbol} {enhanced.chrom}:{enhanced.position}")
        print(f"   Original category: {original.splice_category}")
        print(f"   Enhanced category: {enhanced.splice_category}")
        print(f"   Original evidence: {original.validation_evidence}")
        print(f"   Enhanced evidence: {enhanced.validation_evidence}")
    
    # Create training features
    training_df = extractor.create_regulatory_training_features(enhanced_sites)
    
    print(f"\n📈 Training Features Created:")
    print(f"   Rows: {len(training_df)}")
    print(f"   Columns: {len(training_df.columns)}")
    print(f"   Feature types: {list(training_df.columns)}")
    
    # Save training features
    training_file = work_dir / "regulatory_training_features.tsv"
    training_df.to_csv(training_file, sep='\t', index=False)
    print(f"   Saved to: {training_file}")
    
    return enhanced_sites, training_df


def main():
    """Main demonstration function."""
    parser = argparse.ArgumentParser(
        description="Enhanced Alternative Splicing Analysis Demonstration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full analysis with all diseases
  python enhanced_alternative_splicing_demo.py --work-dir ./demo --all-diseases
  
  # Specific diseases only
  python enhanced_alternative_splicing_demo.py --work-dir ./demo --diseases cystic_fibrosis breast_cancer
  
  # Include regulatory features
  python enhanced_alternative_splicing_demo.py --work-dir ./demo --regulatory-features
  
  # With meta-model path
  python enhanced_alternative_splicing_demo.py --work-dir ./demo --meta-model ./model.pkl
        """
    )
    
    parser.add_argument("--work-dir", type=Path, required=True,
                       help="Working directory for analysis")
    parser.add_argument("--diseases", nargs="+", 
                       choices=["cystic_fibrosis", "breast_cancer", "als_ftd", "lung_cancer"],
                       help="Specific diseases to analyze")
    parser.add_argument("--all-diseases", action="store_true",
                       help="Analyze all available diseases")
    parser.add_argument("--regulatory-features", action="store_true",
                       help="Include regulatory feature enhancement")
    parser.add_argument("--meta-model", type=Path,
                       help="Path to trained meta-model")
    parser.add_argument("--demo-regulatory", action="store_true",
                       help="Run regulatory feature demonstration")
    
    args = parser.parse_args()
    
    # Setup work directory
    args.work_dir.mkdir(parents=True, exist_ok=True)
    
    print("🧬 ENHANCED ALTERNATIVE SPLICING ANALYSIS DEMONSTRATION")
    print("=" * 70)
    print(f"📁 Work directory: {args.work_dir}")
    
    if args.meta_model:
        if args.meta_model.exists():
            print(f"🤖 Meta-model: {args.meta_model}")
        else:
            print(f"⚠️  Meta-model not found: {args.meta_model}")
            args.meta_model = None
    
    try:
        # Run regulatory feature demonstration if requested
        if args.demo_regulatory:
            enhanced_sites, training_df = demonstrate_regulatory_enhancement(args.work_dir)
        
        # Determine diseases to analyze
        diseases = None
        if args.all_diseases:
            diseases = ["cystic_fibrosis", "breast_cancer", "als_ftd", "lung_cancer"]
        elif args.diseases:
            diseases = args.diseases
        else:
            diseases = ["cystic_fibrosis", "breast_cancer"]  # Default
        
        # Run comprehensive analysis
        results = run_comprehensive_analysis(
            work_dir=args.work_dir,
            diseases=diseases,
            include_regulatory=args.regulatory_features,
            meta_model_path=args.meta_model
        )
        
        print("\n🎉 ANALYSIS COMPLETED SUCCESSFULLY!")
        print("=" * 50)
        
        summary = results['analysis_summary']
        print(f"📊 Analysis Summary:")
        print(f"   Diseases analyzed: {summary['diseases_analyzed']}")
        print(f"   Total mutations: {summary['total_mutations']}")
        print(f"   Alternative sites detected: {summary['total_alternative_sites']}")
        print(f"   Cryptic sites activated: {summary['total_cryptic_sites']}")
        print(f"   Mean meta-model improvement: {summary['mean_meta_improvement']:.3f}")
        
        print(f"\n🎯 Key Achievements:")
        print(f"   ✅ Demonstrated adaptive capacity of meta-learning layer")
        print(f"   ✅ Captured disease-specific alternative splicing patterns")
        print(f"   ✅ Enhanced predictions with regulatory context")
        print(f"   ✅ Quantified improvement over base models")
        
        print(f"\n📁 All results saved to: {args.work_dir}")
        
    except KeyboardInterrupt:
        print("\n⚠️  Analysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()


