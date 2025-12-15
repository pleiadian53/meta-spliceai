#!/usr/bin/env python3
"""
Main entry point for MetaSpliceAI case studies.

This script provides a unified interface for running various case study
analyses to validate the meta-learning model against disease-specific
splice mutation databases.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any
import json
import time

# Package imports
from .workflows.disease_validation import DiseaseValidationWorkflow
from .workflows.mutation_analysis import MutationAnalysisWorkflow
from .data_sources import (
    SpliceVarDBIngester, MutSpliceDBIngester, 
    DBASSIngester, ClinVarIngester
)


def print_banner():
    """Print the MetaSpliceAI case study banner."""
    banner = """
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║    🧬 MetaSpliceAI Case Studies Infrastructure 🧬           ║
║                                                               ║
║    Validating Meta-Learning Models on Disease Mutations      ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
"""
    print(banner)


def setup_case_study_environment(work_dir: Path) -> Dict[str, Path]:
    """Setup directory structure for case studies."""
    directories = {
        "work_dir": work_dir,
        "data_sources": work_dir / "data_sources",
        "validation_results": work_dir / "validation_results", 
        "mutation_analysis": work_dir / "mutation_analysis",
        "visualizations": work_dir / "visualizations",
        "reports": work_dir / "reports"
    }
    
    for name, path in directories.items():
        path.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Case study environment set up at: {work_dir}")
    return directories


def run_data_ingestion(directories: Dict[str, Path], 
                      databases: List[str],
                      force_refresh: bool = False) -> Dict[str, Any]:
    """Run data ingestion from specified databases."""
    print("\n🗄️  PHASE 1: DATA INGESTION")
    print("=" * 50)
    
    ingestion_results = {}
    database_classes = {
        "SpliceVarDB": SpliceVarDBIngester,
        "MutSpliceDB": MutSpliceDBIngester,
        "DBASS": DBASSIngester,
        "ClinVar": ClinVarIngester
    }
    
    for db_name in databases:
        if db_name not in database_classes:
            print(f"❌ Unknown database: {db_name}")
            continue
            
        print(f"\n📥 Ingesting {db_name}...")
        
        try:
            db_dir = directories["data_sources"] / db_name.lower()
            ingester_class = database_classes[db_name]
            ingester = ingester_class(db_dir)
            
            start_time = time.time()
            result = ingester.ingest(force_refresh=force_refresh)
            elapsed_time = time.time() - start_time
            
            ingestion_results[db_name] = result
            
            print(f"   ✅ {db_name} completed in {elapsed_time:.1f}s")
            print(f"   📊 {len(result.mutations)} mutations")
            print(f"   🧬 {len(result.splice_sites_df)} splice sites")
            print(f"   🧬 {len(result.gene_features_df)} genes")
            
            # Display sample mutations
            if result.mutations:
                print(f"   📝 Sample mutations:")
                for i, mut in enumerate(result.mutations[:3]):
                    print(f"     {i+1}. {mut.gene_symbol}: {mut.ref_allele}>{mut.alt_allele} ({mut.splice_event_type.value})")
                    
        except Exception as e:
            print(f"   ❌ {db_name} failed: {e}")
            continue
    
    return ingestion_results


def run_disease_validation(directories: Dict[str, Path],
                          meta_model_path: Optional[Path],
                          diseases: List[str],
                          databases: List[str],
                          min_evidence: int = 5) -> Dict[str, Any]:
    """Run disease-specific validation analysis."""
    print("\n🦠 PHASE 2: DISEASE VALIDATION")
    print("=" * 50)
    
    print(f"🎯 Target diseases: {', '.join(diseases)}")
    print(f"📊 Using databases: {', '.join(databases)}")
    print(f"🔬 Min RNA evidence: {min_evidence}")
    
    validation_dir = directories["validation_results"]
    workflow = DiseaseValidationWorkflow(validation_dir, meta_model_path)
    
    try:
        start_time = time.time()
        results = workflow.run_disease_specific_validation(
            diseases=diseases,
            databases=databases,
            min_rna_evidence=min_evidence
        )
        elapsed_time = time.time() - start_time
        
        print(f"\n📈 VALIDATION COMPLETED in {elapsed_time:.1f}s")
        print("-" * 40)
        
        summary_stats = {
            "total_diseases": len(results),
            "total_mutations": sum(r.total_mutations for r in results.values()),
            "mean_accuracy": sum(r.meta_accuracy for r in results.values()) / len(results) if results else 0,
            "mean_improvement": sum(r.improvement for r in results.values()) / len(results) if results else 0,
        }
        
        for disease, result in results.items():
            print(f"\n🦠 {disease.upper()}:")
            print(f"   Mutations: {result.total_mutations}")
            print(f"   Base accuracy: {result.base_accuracy:.3f}")
            print(f"   Meta accuracy: {result.meta_accuracy:.3f}")
            print(f"   Improvement: {result.improvement:+.3f}")
            print(f"   Pathogenic accuracy: {result.pathogenic_accuracy:.3f}")
        
        # Save summary
        summary_file = validation_dir / "phase2_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary_stats, f, indent=2)
            
        return {"results": results, "summary": summary_stats}
        
    except Exception as e:
        print(f"❌ Disease validation failed: {e}")
        return {}


def run_mutation_analysis(directories: Dict[str, Path],
                         meta_model_path: Optional[Path],
                         target_genes: List[str] = None) -> Dict[str, Any]:
    """Run detailed mutation analysis."""
    print("\n🔬 PHASE 3: DETAILED MUTATION ANALYSIS")
    print("=" * 50)
    
    analysis_dir = directories["mutation_analysis"]
    workflow = MutationAnalysisWorkflow(analysis_dir, meta_model_path)
    
    analysis_results = {}
    
    # Run specific case studies
    case_studies = [
        ("CFTR Cryptic Exon", "analyze_cftr_cryptic_exon"),
        ("MET Exon 14 Cohort", "analyze_met_exon14_cohort")
    ]
    
    for study_name, function_name in case_studies:
        print(f"\n🧪 Running {study_name}...")
        
        try:
            # Import the specific analysis function
            if function_name == "analyze_cftr_cryptic_exon":
                from .workflows.mutation_analysis import analyze_cftr_cryptic_exon
                result = analyze_cftr_cryptic_exon(analysis_dir / "cftr", meta_model_path)
                
                analysis_results[study_name] = {
                    "gene": result.mutation.gene_symbol,
                    "score_improvement": result.score_improvement,
                    "cryptic_sites": len(result.cryptic_sites_detected),
                    "experimental_evidence": result.experimental_evidence
                }
                
                print(f"   ✅ {study_name} completed")
                print(f"   🧬 Gene: {result.mutation.gene_symbol}")
                print(f"   📈 Score improvement: {result.score_improvement:+.3f}")
                print(f"   🎯 Cryptic sites: {len(result.cryptic_sites_detected)}")
                
            elif function_name == "analyze_met_exon14_cohort":
                from .workflows.mutation_analysis import analyze_met_exon14_cohort
                result = analyze_met_exon14_cohort(analysis_dir / "met", meta_model_path)
                
                if result:
                    analysis_results[study_name] = {
                        "cohort_size": result.get("total_mutations", 0),
                        "success_rate": result.get("analysis_success_rate", 0),
                        "mean_improvement": result.get("mean_score_improvement", 0)
                    }
                    
                    print(f"   ✅ {study_name} completed")
                    print(f"   👥 Cohort size: {result.get('total_mutations', 0)}")
                    print(f"   📊 Success rate: {result.get('analysis_success_rate', 0):.2%}")
                
        except Exception as e:
            print(f"   ❌ {study_name} failed: {e}")
            continue
    
    return analysis_results


def generate_final_report(directories: Dict[str, Path], 
                         phase_results: Dict[str, Any]) -> Path:
    """Generate comprehensive final report."""
    print("\n📋 GENERATING FINAL REPORT")
    print("=" * 50)
    
    report_dir = directories["reports"]
    report_file = report_dir / "case_study_final_report.json"
    
    final_report = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "work_directory": str(directories["work_dir"]),
            "analysis_phases": list(phase_results.keys())
        },
        "executive_summary": {},
        "phase_results": phase_results
    }
    
    # Generate executive summary
    if "phase2" in phase_results and phase_results["phase2"]:
        validation_summary = phase_results["phase2"].get("summary", {})
        final_report["executive_summary"] = {
            "total_diseases_analyzed": validation_summary.get("total_diseases", 0),
            "total_mutations_analyzed": validation_summary.get("total_mutations", 0),
            "overall_meta_model_accuracy": validation_summary.get("mean_accuracy", 0),
            "mean_improvement_over_base": validation_summary.get("mean_improvement", 0),
            "analysis_successful": validation_summary.get("mean_improvement", 0) > 0
        }
    
    # Save report
    with open(report_file, 'w') as f:
        json.dump(final_report, f, indent=2)
    
    # Create human-readable summary
    summary_file = report_dir / "executive_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("MetaSpliceAI Case Study Executive Summary\n")
        f.write("=" * 50 + "\n\n")
        
        exec_summary = final_report["executive_summary"]
        if exec_summary:
            f.write(f"Diseases analyzed: {exec_summary.get('total_diseases_analyzed', 0)}\n")
            f.write(f"Mutations analyzed: {exec_summary.get('total_mutations_analyzed', 0)}\n")
            f.write(f"Meta-model accuracy: {exec_summary.get('overall_meta_model_accuracy', 0):.3f}\n")
            f.write(f"Improvement over base: {exec_summary.get('mean_improvement_over_base', 0):+.3f}\n")
            f.write(f"Analysis successful: {exec_summary.get('analysis_successful', False)}\n")
        
        f.write(f"\nDetailed results available in: {report_file}\n")
    
    print(f"📄 Final report saved to: {report_file}")
    print(f"📝 Executive summary: {summary_file}")
    
    return report_file


def main():
    """Main case study execution."""
    parser = argparse.ArgumentParser(
        description="MetaSpliceAI Case Studies - Comprehensive Analysis Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline with all databases
  python run_case_study.py --work-dir ./case_studies --meta-model ./model.pkl --all-phases
  
  # Data ingestion only
  python run_case_study.py --work-dir ./data --phase1-only --databases SpliceVarDB MutSpliceDB
  
  # Disease validation for specific diseases
  python run_case_study.py --work-dir ./validation --phase2-only --diseases lung_cancer breast_cancer
  
  # Mutation analysis only
  python run_case_study.py --work-dir ./analysis --phase3-only --meta-model ./model.pkl
        """
    )
    
    # Main arguments
    parser.add_argument("--work-dir", type=Path, required=True,
                       help="Working directory for case study")
    parser.add_argument("--meta-model", type=Path, default=None,
                       help="Path to trained meta-model")
    
    # Phase control
    parser.add_argument("--all-phases", action="store_true",
                       help="Run all analysis phases")
    parser.add_argument("--phase1-only", action="store_true",
                       help="Run only data ingestion (Phase 1)")
    parser.add_argument("--phase2-only", action="store_true", 
                       help="Run only disease validation (Phase 2)")
    parser.add_argument("--phase3-only", action="store_true",
                       help="Run only mutation analysis (Phase 3)")
    
    # Data source options
    parser.add_argument("--databases", nargs="+", 
                       default=["SpliceVarDB", "MutSpliceDB", "DBASS", "ClinVar"],
                       choices=["SpliceVarDB", "MutSpliceDB", "DBASS", "ClinVar"],
                       help="Databases to use for analysis")
    parser.add_argument("--force-refresh", action="store_true",
                       help="Force re-download of database data")
    
    # Analysis options
    parser.add_argument("--diseases", nargs="+",
                       default=["lung_cancer", "breast_cancer", "cystic_fibrosis"],
                       help="Diseases to focus validation on")
    parser.add_argument("--min-evidence", type=int, default=5,
                       help="Minimum RNA evidence threshold")
    parser.add_argument("--target-genes", nargs="+", default=None,
                       help="Specific genes for mutation analysis")
    
    args = parser.parse_args()
    
    # Banner and setup
    print_banner()
    print(f"🚀 Starting case study analysis...")
    print(f"📁 Work directory: {args.work_dir}")
    
    if args.meta_model:
        if args.meta_model.exists():
            print(f"🤖 Meta-model: {args.meta_model}")
        else:
            print(f"⚠️  Meta-model not found: {args.meta_model}")
            print("Continuing with base model only...")
            args.meta_model = None
    
    # Setup environment
    directories = setup_case_study_environment(args.work_dir)
    
    # Determine which phases to run
    if not any([args.all_phases, args.phase1_only, args.phase2_only, args.phase3_only]):
        args.all_phases = True  # Default to all phases
    
    phase_results = {}
    
    try:
        # Phase 1: Data Ingestion
        if args.all_phases or args.phase1_only:
            ingestion_results = run_data_ingestion(
                directories, args.databases, args.force_refresh
            )
            phase_results["phase1"] = {
                "databases_processed": list(ingestion_results.keys()),
                "total_mutations": sum(len(r.mutations) for r in ingestion_results.values())
            }
        
        # Phase 2: Disease Validation
        if args.all_phases or args.phase2_only:
            validation_results = run_disease_validation(
                directories, args.meta_model, args.diseases, 
                args.databases, args.min_evidence
            )
            phase_results["phase2"] = validation_results
        
        # Phase 3: Mutation Analysis
        if args.all_phases or args.phase3_only:
            mutation_results = run_mutation_analysis(
                directories, args.meta_model, args.target_genes
            )
            phase_results["phase3"] = mutation_results
        
        # Generate final report
        if phase_results:
            final_report = generate_final_report(directories, phase_results)
            
            print("\n🎉 CASE STUDY ANALYSIS COMPLETED SUCCESSFULLY!")
            print("=" * 60)
            print(f"📁 All results saved to: {args.work_dir}")
            print(f"📄 Final report: {final_report}")
            
            # Display key outcomes
            if "phase2" in phase_results and phase_results["phase2"]:
                summary = phase_results["phase2"].get("summary", {})
                improvement = summary.get("mean_improvement", 0)
                
                if improvement > 0:
                    print(f"✅ Meta-model shows {improvement:+.3f} average improvement!")
                else:
                    print(f"⚠️  Meta-model shows {improvement:+.3f} average change")
        
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