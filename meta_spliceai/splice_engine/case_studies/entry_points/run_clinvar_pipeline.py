#!/usr/bin/env python3
"""
ClinVar Pipeline Runner

A comprehensive ClinVar processing pipeline that integrates with the MetaSpliceAI
genomic resources system to provide systematic path discovery and prepare data
for delta score computation using either base models (SpliceAI, OpenSpliceAI) 
or meta models via the inference workflow.

Key Features:
- Systematic path discovery for VCF and reference files
- Supports multiple input formats (filename, date, full path)
- Integrates with MetaSpliceAI's resource management
- Prepares data for both base model and meta model delta score computation
- Clean, simple command-line interface

Usage Examples:
    # Simple usage with systematic discovery (run from project root)
    python meta_spliceai/splice_engine/case_studies/entry_points/run_clinvar_pipeline.py clinvar_20250831.vcf.gz results/clinvar_pipeline
    
    # With specific reference genome
    python meta_spliceai/splice_engine/case_studies/entry_points/run_clinvar_pipeline.py clinvar_20250831.vcf.gz results/clinvar_pipeline --reference Homo_sapiens.GRCh38.dna.primary_assembly.fa
    
    # Using date format
    python meta_spliceai/splice_engine/case_studies/entry_points/run_clinvar_pipeline.py 20250831 results/clinvar_pipeline
    
    # Research mode with all variants
    python meta_spliceai/splice_engine/case_studies/entry_points/run_clinvar_pipeline.py clinvar_20250831.vcf.gz results/research --research-mode
    
    # Test mode with limited variants
    python meta_spliceai/splice_engine/case_studies/entry_points/run_clinvar_pipeline.py 20250831 results/test --max-variants 1000

Author: Barnett Chiu
"""

import sys
import argparse
from pathlib import Path
from typing import Optional, Union

# Add the project root to the path using systematic detection
from project_root_utils import setup_entry_point_imports
setup_entry_point_imports(__file__)

from meta_spliceai.splice_engine.case_studies.workflows.complete_clinvar_pipeline import (
    CompleteClinVarPipeline,
    CompletePipelineConfig
)
from meta_spliceai.splice_engine.case_studies.data_sources.resource_manager import (
    CaseStudyResourceManager
)
from meta_spliceai.system.genomic_resources import create_systematic_manager


def find_vcf_file(vcf_input: str) -> Path:
    """
    Find VCF file using systematic discovery.
    
    Parameters
    ----------
    vcf_input : str
        VCF file identifier (filename, date, or full path)
        
    Returns
    -------
    Path
        Resolved VCF file path
        
    Raises
    ------
    FileNotFoundError
        If VCF file cannot be found
    """
    vcf_input = vcf_input.strip()
    
    # If it's already a full path, use it directly
    if Path(vcf_input).is_absolute() or '/' in vcf_input:
        vcf_path = Path(vcf_input)
        if vcf_path.exists():
            return vcf_path
        else:
            raise FileNotFoundError(f"VCF file not found: {vcf_path}")
    
    # Try systematic discovery using resource manager
    try:
        resource_manager = CaseStudyResourceManager()
        
        # If it looks like a date (YYYYMMDD), use it directly
        if vcf_input.isdigit() and len(vcf_input) == 8:
            vcf_path = resource_manager.get_clinvar_vcf_path(date=vcf_input)
        # If it's a filename, use it directly
        elif vcf_input.endswith('.vcf.gz') or vcf_input.endswith('.vcf'):
            vcf_path = resource_manager.get_clinvar_vcf_path(filename=vcf_input)
        # Otherwise, try to find the most recent file
        else:
            vcf_path = resource_manager.get_clinvar_vcf_path()
        
        if vcf_path.exists():
            return vcf_path
        else:
            raise FileNotFoundError(f"VCF file not found: {vcf_path}")
            
    except Exception as e:
        # Fallback: try current directory
        current_dir_vcf = Path(vcf_input)
        if current_dir_vcf.exists():
            return current_dir_vcf
        else:
            raise FileNotFoundError(f"VCF file not found: {vcf_input}. Error: {e}")


def find_reference_fasta(reference_input: Optional[str]) -> Optional[Path]:
    """
    Find reference FASTA file using systematic discovery.
    
    Parameters
    ----------
    reference_input : str, optional
        Reference file identifier (filename or full path)
        
    Returns
    -------
    Path, optional
        Resolved reference FASTA path, or None if not found
    """
    if not reference_input:
        return None
    
    reference_input = reference_input.strip()
    
    # If it's already a full path, use it directly
    if Path(reference_input).is_absolute() or '/' in reference_input:
        ref_path = Path(reference_input)
        if ref_path.exists():
            return ref_path
        else:
            print(f"Warning: Reference FASTA not found: {ref_path}")
            return None
    
    # Try systematic discovery using genomic resources manager
    try:
        genomic_manager = create_systematic_manager()
        ref_path = genomic_manager.get_fasta_path(validate=False)
        
        if ref_path.exists():
            return ref_path
        else:
            print(f"Warning: Reference FASTA not found: {ref_path}")
            return None
            
    except Exception as e:
        print(f"Warning: Could not find reference FASTA: {e}")
        return None


def resolve_output_directory(output_input: str) -> Path:
    """
    Resolve output directory path.
    
    Parameters
    ----------
    output_input : str
        Output directory path
        
    Returns
    -------
    Path
        Resolved output directory path
    """
    output_path = Path(output_input)
    
    # If it's a relative path, make it relative to project root
    if not output_path.is_absolute():
        project_root = Path.cwd()  # Assuming we're running from project root
        output_path = project_root / output_path
    
    return output_path


def main():
    """Main command-line interface with systematic path discovery."""
    parser = argparse.ArgumentParser(
        description="ClinVar Pipeline: Raw VCF → WT/ALT Ready Data for Delta Score Computation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Simple usage with systematic discovery
  python run_clinvar_pipeline.py clinvar_20250831.vcf.gz results/clinvar_pipeline
  
  # With specific reference genome
  python run_clinvar_pipeline.py clinvar_20250831.vcf.gz results/clinvar_pipeline \\
      --reference Homo_sapiens.GRCh38.dna.primary_assembly.fa
  
  # Using date format
  python run_clinvar_pipeline.py 20250831 results/clinvar_pipeline
  
  # Research mode with all variants
  python run_clinvar_pipeline.py clinvar_20250831.vcf.gz results/research --research-mode
  
  # Test mode with limited variants
  python run_clinvar_pipeline.py 20250831 results/test --max-variants 1000
  
  # Pathogenic variants only
  python run_clinvar_pipeline.py clinvar_20250831.vcf.gz results/pathogenic --pathogenic-only
  
  # Benign variants only
  python run_clinvar_pipeline.py clinvar_20250831.vcf.gz results/benign --benign-only
  
  # Both pathogenic and benign for classification evaluation
  python run_clinvar_pipeline.py clinvar_20250831.vcf.gz results/classification --pathogenic-and-benign
  
  # High-confidence variants only (practice guidelines + expert panels)
  python run_clinvar_pipeline.py clinvar_20250831.vcf.gz results/high_confidence \\
      --pathogenic-and-benign --review-status practice_guideline expert_panel
  
  # Specific splice events (donor gain/loss)
  python run_clinvar_pipeline.py clinvar_20250831.vcf.gz results/donor_events \\
      --pathogenic-and-benign --splice-events DG DL

Next Steps:
  After running this pipeline, your data will be ready for:
  • Base model delta scores: SpliceAI, OpenSpliceAI
  • Meta model delta scores: MetaSpliceAI inference workflow
  • Recalibrated per-nucleotide splice site scores
  • Classification evaluation: ROC-AUC/PR-AUC for pathogenic vs benign
        """
    )
    
    # Positional arguments
    parser.add_argument('vcf_input', 
                       help='VCF file identifier (filename, date YYYYMMDD, or full path)')
    parser.add_argument('output_dir',
                       help='Output directory for results')
    
    # Key options
    parser.add_argument('--reference', '-r',
                       help='Reference FASTA identifier (filename or full path)')
    parser.add_argument('--research-mode', action='store_true',
                       help='Include all variants, not just splice-affecting')
    parser.add_argument('--max-variants', type=int,
                       help='Limit variants for testing (e.g., 1000)')
    parser.add_argument('--no-sequences', action='store_true',
                       help='Skip WT/ALT sequence construction')
    parser.add_argument('--pathogenic-only', action='store_true',
                       help='Only process pathogenic/likely pathogenic variants')
    parser.add_argument('--benign-only', action='store_true',
                       help='Only process benign/likely benign variants')
    parser.add_argument('--pathogenic-and-benign', action='store_true',
                       help='Process both pathogenic and benign variants for classification')
    parser.add_argument('--review-status', nargs='+',
                       choices=['practice_guideline', 'expert_panel', 'multiple_submitters', 'single_submitter'],
                       help='Filter by review status (e.g., practice_guideline expert_panel)')
    parser.add_argument('--splice-events', nargs='+',
                       choices=['DG', 'DL', 'AG', 'AL'],
                       help='Filter by specific splice events (DG=donor_gain, DL=donor_loss, AG=acceptor_gain, AL=acceptor_loss)')
    parser.add_argument('--min-distance', type=int, default=50,
                       help='Minimum distance from chromosome ends (default: 50)')
    parser.add_argument('--threads', type=int, default=4,
                       help='Number of threads (default: 4)')
    parser.add_argument('--quiet', action='store_true',
                       help='Reduce output verbosity')
    
    args = parser.parse_args()
    
    print("🚀 ClinVar Pipeline: Preparing Data for Delta Score Computation")
    print("=" * 70)
    
    try:
        # Resolve VCF file
        print(f"🔍 Resolving VCF file: {args.vcf_input}")
        vcf_path = find_vcf_file(args.vcf_input)
        print(f"✅ Found VCF: {vcf_path}")
        
        # Resolve reference FASTA
        reference_path = None
        if args.reference:
            print(f"🔍 Resolving reference FASTA: {args.reference}")
            reference_path = find_reference_fasta(args.reference)
            if reference_path:
                print(f"✅ Found reference: {reference_path}")
            else:
                print("⚠️  Reference FASTA not found, will use auto-detection")
        
        # Resolve output directory
        output_path = resolve_output_directory(args.output_dir)
        print(f"📁 Output directory: {output_path}")
        
        # Validate filtering arguments
        if args.pathogenic_only and args.benign_only:
            print("❌ Error: Cannot specify both --pathogenic-only and --benign-only")
            return 1
        
        if args.pathogenic_and_benign and (args.pathogenic_only or args.benign_only):
            print("❌ Error: Cannot specify --pathogenic-and-benign with --pathogenic-only or --benign-only")
            return 1
        
        # Determine pathogenicity filtering
        if args.pathogenic_and_benign:
            pathogenicity_filter = ['Pathogenic', 'Likely_pathogenic', 'Benign', 'Likely_benign']
        elif args.pathogenic_only:
            pathogenicity_filter = ['Pathogenic', 'Likely_pathogenic']
        elif args.benign_only:
            pathogenicity_filter = ['Benign', 'Likely_benign']
        else:
            pathogenicity_filter = None if args.research_mode else ['Pathogenic', 'Likely_pathogenic']
        
        # Create configuration
        config = CompletePipelineConfig(
            input_vcf=vcf_path,
            output_dir=output_path,
            reference_fasta=reference_path,
            research_mode=args.research_mode,
            pathogenic_only=args.pathogenic_only,
            max_variants=args.max_variants,
            include_sequences=not args.no_sequences,
            threads=args.threads,
            verbose=not args.quiet,
            output_formats=['tsv', 'parquet']
        )
        
        # Add custom filtering options to config
        config.pathogenicity_filter = pathogenicity_filter
        config.review_status_filter = args.review_status
        config.splice_events_filter = args.splice_events
        config.min_distance = args.min_distance
        
        print("\n📋 Configuration:")
        print(f"   • Input VCF: {vcf_path}")
        print(f"   • Output Dir: {output_path}")
        print(f"   • Reference: {reference_path or 'Auto-detected'}")
        print(f"   • Research Mode: {args.research_mode}")
        print(f"   • Max Variants: {args.max_variants or 'All'}")
        print(f"   • Threads: {args.threads}")
        
        # Display filtering configuration
        if pathogenicity_filter:
            print(f"   • Pathogenicity Filter: {', '.join(pathogenicity_filter)}")
        if args.review_status:
            print(f"   • Review Status Filter: {', '.join(args.review_status)}")
        if args.splice_events:
            print(f"   • Splice Events Filter: {', '.join(args.splice_events)}")
        if args.min_distance != 50:
            print(f"   • Min Distance from Chrom Ends: {args.min_distance} bp")
        
        if args.research_mode:
            print("🔬 Research mode: Including all variants")
        if args.max_variants:
            print(f"🧪 Test mode: Limited to {args.max_variants:,} variants")
        if args.pathogenic_only:
            print("⚠️  Pathogenic-only mode: Only pathogenic/likely pathogenic variants")
        elif args.benign_only:
            print("⚠️  Benign-only mode: Only benign/likely benign variants")
        elif args.pathogenic_and_benign:
            print("🎯 Classification mode: Both pathogenic and benign variants for evaluation")
        
        print("\n🚀 Starting pipeline...")
        
        # Run the pipeline
        pipeline = CompleteClinVarPipeline(config)
        results = pipeline.run_complete_pipeline()
        
        print("\n🎉 Pipeline completed successfully!")
        print("\n📊 Next Steps for Delta Score Computation:")
        print("   • Base Models: Use WT/ALT sequences with SpliceAI or OpenSpliceAI")
        print("   • Meta Models: Use MetaSpliceAI inference workflow for recalibrated scores")
        print("   • Output files: Check results directory for processed data")
        
        return 0
        
    except FileNotFoundError as e:
        print(f"\n❌ File not found: {e}")
        print("\n💡 Tips:")
        print("   • For VCF files: Use filename (e.g., 'clinvar_20250831.vcf.gz') or date (e.g., '20250831')")
        print("   • For reference: Use filename (e.g., 'Homo_sapiens.GRCh38.dna.primary_assembly.fa')")
        print("   • Full paths are also supported")
        return 1
        
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        if not args.quiet:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
