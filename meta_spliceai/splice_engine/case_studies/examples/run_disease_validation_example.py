#!/usr/bin/env python3
"""
Example script for running disease validation case studies.

This script demonstrates how to use the MetaSpliceAI case study infrastructure
to validate meta-model performance on disease-specific splice mutations.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

# Add the package to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from meta_spliceai.splice_engine.case_studies.workflows.disease_validation import (
    DiseaseValidationWorkflow, run_met_exon14_case_study
)
from meta_spliceai.splice_engine.case_studies.data_sources import (
    SpliceVarDBIngester, MutSpliceDBIngester, DBASSIngester, ClinVarIngester
)


def run_comprehensive_disease_validation(work_dir: Path, 
                                       meta_model_path: Optional[Path] = None,
                                       diseases: List[str] = None,
                                       databases: List[str] = None) -> None:
    """
    Run comprehensive disease validation across multiple databases.
    
    Parameters
    ----------
    work_dir : Path
        Working directory for analysis
    meta_model_path : Path, optional
        Path to trained meta-model
    diseases : List[str], optional
        Diseases to focus on
    databases : List[str], optional
        Databases to use
    """
    print("🧬 MetaSpliceAI Disease Validation Case Study")
    print("=" * 60)
    
    # Default parameters
    if diseases is None:
        diseases = [
            "lung_cancer", "NSCLC", "breast_cancer", "BRCA", 
            "cystic_fibrosis", "muscular_dystrophy"
        ]
    
    if databases is None:
        databases = ["SpliceVarDB", "MutSpliceDB", "DBASS", "ClinVar"]
    
    print(f"🎯 Target diseases: {', '.join(diseases)}")
    print(f"📊 Data sources: {', '.join(databases)}")
    print(f"📁 Work directory: {work_dir}")
    
    if meta_model_path:
        print(f"🤖 Meta-model: {meta_model_path}")
    else:
        print("⚠️  No meta-model provided - will use base model only")
    
    print("\n")
    
    # Initialize workflow
    workflow = DiseaseValidationWorkflow(work_dir, meta_model_path)
    
    try:
        # Run disease-specific validation
        print("🔄 Starting disease-specific validation...")
        results = workflow.run_disease_specific_validation(
            diseases=diseases,
            databases=databases,
            min_rna_evidence=5
        )
        
        # Print summary results
        print("\n📈 VALIDATION RESULTS SUMMARY")
        print("-" * 40)
        
        for disease, result in results.items():
            print(f"\n🦠 {disease.upper()}:")
            print(f"   Total mutations analyzed: {result.total_mutations}")
            print(f"   Base model accuracy: {result.base_accuracy:.3f}")
            print(f"   Meta model accuracy: {result.meta_accuracy:.3f}")
            print(f"   Improvement: {result.improvement:+.3f}")
            print(f"   Pathogenic accuracy: {result.pathogenic_accuracy:.3f}")
            
            # Event type breakdown
            if result.event_type_metrics:
                print("   Event type performance:")
                for event_type, metrics in result.event_type_metrics.items():
                    accuracy = metrics.get('accuracy', 0)
                    count = metrics.get('total', 0)
                    print(f"     • {event_type}: {accuracy:.3f} ({count} cases)")
        
        print(f"\n✅ Analysis completed successfully!")
        print(f"📁 Detailed results saved to: {work_dir}")
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        raise


def run_specific_case_studies(work_dir: Path, 
                            meta_model_path: Optional[Path] = None) -> None:
    """Run specific well-known case studies."""
    
    print("🔬 Running Specific Case Studies")
    print("=" * 40)
    
    case_studies_dir = work_dir / "specific_case_studies"
    case_studies_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. MET Exon 14 Skipping Case Study
    print("\n1️⃣  MET Exon 14 Skipping Analysis")
    print("-" * 30)
    
    try:
        met_dir = case_studies_dir / "met_exon14"
        met_result = run_met_exon14_case_study(met_dir, meta_model_path)
        
        if met_result:
            print(f"   ✅ MET analysis completed")
            print(f"   📊 Accuracy: {met_result.accuracy:.3f}")
            print(f"   🎯 Improvement: {met_result.improvement:+.3f}")
        else:
            print("   ⚠️  MET analysis returned no results")
            
    except Exception as e:
        print(f"   ❌ MET analysis failed: {e}")
    
    # 2. CFTR Cryptic Exon Analysis
    print("\n2️⃣  CFTR Cryptic Exon Analysis")
    print("-" * 30)
    
    try:
        from meta_spliceai.splice_engine.case_studies.workflows.mutation_analysis import analyze_cftr_cryptic_exon
        
        cftr_dir = case_studies_dir / "cftr_cryptic_exon"
        cftr_result = analyze_cftr_cryptic_exon(cftr_dir, meta_model_path)
        
        print(f"   ✅ CFTR analysis completed")
        print(f"   🧬 Gene: {cftr_result.mutation.gene_symbol}")
        print(f"   🔄 Score improvement: {cftr_result.score_improvement:+.3f}")
        print(f"   🎯 Cryptic sites detected: {len(cftr_result.cryptic_sites_detected)}")
        
    except Exception as e:
        print(f"   ❌ CFTR analysis failed: {e}")


def demonstrate_database_ingestion(work_dir: Path) -> None:
    """Demonstrate individual database ingestion."""
    
    print("🗄️  Database Ingestion Demonstration")
    print("=" * 40)
    
    ingestion_dir = work_dir / "database_ingestion"
    ingestion_dir.mkdir(parents=True, exist_ok=True)
    
    databases = [
        ("SpliceVarDB", SpliceVarDBIngester),
        ("MutSpliceDB", MutSpliceDBIngester),
        ("DBASS", DBASSIngester),
        ("ClinVar", ClinVarIngester)
    ]
    
    for db_name, ingester_class in databases:
        print(f"\n📥 Ingesting {db_name}...")
        
        try:
            db_dir = ingestion_dir / db_name.lower()
            ingester = ingester_class(db_dir)
            
            # Run ingestion
            result = ingester.ingest(force_refresh=False)
            
            print(f"   ✅ {db_name} ingestion completed")
            print(f"   📊 Total mutations: {len(result.mutations)}")
            print(f"   🧬 Splice sites: {len(result.splice_sites_df)}")
            print(f"   🧬 Genes: {len(result.gene_features_df)}")
            
            # Display sample mutations
            if result.mutations:
                sample_mutations = result.mutations[:3]
                print(f"   📝 Sample mutations:")
                for mut in sample_mutations:
                    print(f"     • {mut.gene_symbol}: {mut.ref_allele}>{mut.alt_allele} ({mut.splice_event_type.value})")
            
        except Exception as e:
            print(f"   ❌ {db_name} ingestion failed: {e}")


def main():
    """Main entry point for the example script."""
    parser = argparse.ArgumentParser(
        description="MetaSpliceAI Disease Validation Case Study Example",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run comprehensive disease validation
  python run_disease_validation_example.py --work-dir ./case_study_results --comprehensive
  
  # Run specific case studies only
  python run_disease_validation_example.py --work-dir ./case_study_results --specific-cases
  
  # Run with trained meta-model
  python run_disease_validation_example.py --work-dir ./results --meta-model ./model.pkl --comprehensive
  
  # Test database ingestion
  python run_disease_validation_example.py --work-dir ./test --demo-ingestion
        """
    )
    
    parser.add_argument(
        "--work-dir", type=Path, required=True,
        help="Working directory for case study results"
    )
    
    parser.add_argument(
        "--meta-model", type=Path, default=None,
        help="Path to trained meta-model (optional)"
    )
    
    parser.add_argument(
        "--comprehensive", action="store_true",
        help="Run comprehensive disease validation across all databases"
    )
    
    parser.add_argument(
        "--specific-cases", action="store_true",
        help="Run specific well-known case studies (MET, CFTR, etc.)"
    )
    
    parser.add_argument(
        "--demo-ingestion", action="store_true",
        help="Demonstrate database ingestion capabilities"
    )
    
    parser.add_argument(
        "--diseases", nargs="+", default=None,
        help="Specific diseases to focus on (for comprehensive validation)"
    )
    
    parser.add_argument(
        "--databases", nargs="+", default=None,
        choices=["SpliceVarDB", "MutSpliceDB", "DBASS", "ClinVar"],
        help="Specific databases to use"
    )
    
    args = parser.parse_args()
    
    # Create work directory
    args.work_dir.mkdir(parents=True, exist_ok=True)
    
    print("🧬 MetaSpliceAI Case Study Infrastructure")
    print("🔬 Disease Validation Example")
    print("=" * 60)
    print(f"📁 Work directory: {args.work_dir}")
    print("")
    
    # Validate meta-model path if provided
    if args.meta_model and not args.meta_model.exists():
        print(f"⚠️  Meta-model file not found: {args.meta_model}")
        print("Continuing with base model only...\n")
        args.meta_model = None
    
    try:
        # Run requested analyses
        if args.comprehensive:
            print("🎯 Running comprehensive disease validation...")
            run_comprehensive_disease_validation(
                work_dir=args.work_dir,
                meta_model_path=args.meta_model,
                diseases=args.diseases,
                databases=args.databases
            )
        
        if args.specific_cases:
            print("\n🔬 Running specific case studies...")
            run_specific_case_studies(
                work_dir=args.work_dir,
                meta_model_path=args.meta_model
            )
        
        if args.demo_ingestion:
            print("\n🗄️  Demonstrating database ingestion...")
            demonstrate_database_ingestion(args.work_dir)
        
        # If no specific mode selected, run a basic demo
        if not any([args.comprehensive, args.specific_cases, args.demo_ingestion]):
            print("🚀 Running basic demonstration...")
            print("   Use --help to see available options\n")
            
            # Run database ingestion demo
            demonstrate_database_ingestion(args.work_dir)
            
            # Run specific cases if meta-model available
            if args.meta_model:
                run_specific_case_studies(args.work_dir, args.meta_model)
        
        print("\n🎉 Case study demonstration completed successfully!")
        print(f"📁 All results saved to: {args.work_dir}")
        
    except KeyboardInterrupt:
        print("\n⚠️  Analysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Analysis failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 