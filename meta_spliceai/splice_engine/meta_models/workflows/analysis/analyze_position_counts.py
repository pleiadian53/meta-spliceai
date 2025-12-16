#!/usr/bin/env python3
"""
Position Count Analysis Driver Script

This is the main driver script for analyzing position count behavior in SpliceAI inference.
It provides a simple interface to run comprehensive position count analysis without
needing to remember multiple module names.

USAGE:
    # Quick analysis of specific genes
    python analyze_position_counts.py --genes ENSG00000142748 ENSG00000000003
    
    # Comprehensive analysis with all tools
    python analyze_position_counts.py --genes ENSG00000142748 --comprehensive
    
    # Cross-mode validation
    python analyze_position_counts.py --genes ENSG00000142748 --validate-modes
    
    # Generate detailed report
    python analyze_position_counts.py --genes ENSG00000142748 --report output_report.txt
"""

import sys
import argparse
from pathlib import Path
from typing import List, Optional

# Add meta_spliceai to path
sys.path.insert(0, str(Path(__file__).parents[5]))

from .position_counts import PositionCountAnalyzer
from .inference_validation import InferenceModeValidator
from .detailed_analysis import main as run_detailed_analysis
from .boundary_effects import main as run_boundary_analysis
from .pipeline_tracing import main as run_pipeline_analysis


def run_quick_analysis(gene_ids: List[str], model_path: str = None, training_dataset: str = None):
    """Run quick position count analysis for specified genes."""
    print("🚀 QUICK POSITION COUNT ANALYSIS")
    print("=" * 60)
    
    analyzer = PositionCountAnalyzer(verbose=1)
    
    print(f"📊 Analyzing {len(gene_ids)} genes:")
    for gene_id in gene_ids:
        gene_length = analyzer.get_gene_length(gene_id)
        print(f"   • {gene_id}: {gene_length:,} bp")
    print()
    
    # Simulate expected position counts
    print("📈 Expected Position Count Behavior:")
    for gene_id in gene_ids:
        gene_length = analyzer.get_gene_length(gene_id)
        if gene_length > 0:
            expected_raw = gene_length * 2  # Donor + acceptor
            expected_final = gene_length + 1  # +1 boundary enhancement
            
            print(f"   {gene_id}:")
            print(f"     • Raw predictions: ~{expected_raw:,} (donor + acceptor)")
            print(f"     • Final positions: ~{expected_final:,} (consolidated + boundary)")
            print(f"     • Expected pattern: {expected_raw:,} → {expected_final:,}")
    
    print()
    print("💡 Key Insights:")
    print("   • Raw count = ~2x gene length (donor + acceptor predictions)")
    print("   • Final count = ~gene length + 1 (boundary enhancement)")
    print("   • Small asymmetries (0.1-0.3%) are normal and expected")


def run_comprehensive_analysis(gene_ids: List[str]):
    """Run comprehensive analysis using all available tools."""
    print("🔬 COMPREHENSIVE POSITION COUNT ANALYSIS")
    print("=" * 70)
    
    print("Running detailed analysis...")
    run_detailed_analysis()
    
    print("\n" + "="*70)
    print("Running boundary effects analysis...")
    run_boundary_analysis()
    
    print("\n" + "="*70)
    print("Running pipeline tracing analysis...")
    run_pipeline_analysis()
    
    print("\n" + "="*70)
    print("🎉 Comprehensive analysis complete!")
    print("All analysis tools have been executed. Review the output above for insights.")


def run_mode_validation(gene_ids: List[str], model_path: str, training_dataset: str):
    """Run cross-mode validation to ensure consistency."""
    print("🔬 INFERENCE MODE CONSISTENCY VALIDATION")
    print("=" * 70)
    
    validator = InferenceModeValidator(model_path, training_dataset, verbose=True)
    results = validator.validate_inference_consistency(gene_ids)
    
    if results['validation_passed']:
        print("\n✅ VALIDATION PASSED")
        print("All inference modes show consistent position count behavior!")
    else:
        print("\n❌ VALIDATION FAILED")
        print("Found inconsistencies that require investigation:")
        for issue in results['consistency_issues']:
            print(f"   • {issue['gene_id']}: {issue['position_counts']}")
    
    return results


def generate_analysis_report(gene_ids: List[str], output_file: str, 
                           model_path: str = None, training_dataset: str = None):
    """Generate a comprehensive analysis report."""
    print(f"📋 Generating comprehensive analysis report: {output_file}")
    
    with open(output_file, 'w') as f:
        f.write("# Position Count Analysis Report\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"## Analyzed Genes\n")
        f.write(f"Total genes: {len(gene_ids)}\n")
        f.write(f"Gene list: {', '.join(gene_ids)}\n\n")
        
        # Add gene length information
        analyzer = PositionCountAnalyzer(verbose=0)
        f.write("## Gene Information\n")
        for gene_id in gene_ids:
            gene_length = analyzer.get_gene_length(gene_id)
            f.write(f"- **{gene_id}**: {gene_length:,} bp\n")
        f.write("\n")
        
        # Add expected behavior
        f.write("## Expected Position Count Behavior\n")
        f.write("Based on SpliceAI processing pipeline:\n\n")
        f.write("1. **Raw Predictions**: ~2x gene length (separate donor + acceptor counts)\n")
        f.write("2. **Final Positions**: ~gene length + 1 (consolidated + boundary enhancement)\n")
        f.write("3. **Asymmetry**: 0.1-0.3% donor/acceptor asymmetry is normal\n")
        f.write("4. **Consistency**: All inference modes should show identical position counts\n\n")
        
        # Add analysis results if model provided
        if model_path and training_dataset:
            f.write("## Cross-Mode Validation Results\n")
            try:
                validator = InferenceModeValidator(model_path, training_dataset, verbose=False)
                validation_results = validator.validate_inference_consistency(gene_ids)
                
                if validation_results['validation_passed']:
                    f.write("✅ **VALIDATION PASSED**: All modes show consistent behavior\n\n")
                else:
                    f.write("❌ **VALIDATION FAILED**: Found inconsistencies requiring investigation\n\n")
                    for issue in validation_results['consistency_issues']:
                        f.write(f"- {issue['gene_id']}: {issue['position_counts']}\n")
                    f.write("\n")
                    
            except Exception as e:
                f.write(f"⚠️ Could not run cross-mode validation: {e}\n\n")
        
        f.write("## Analysis Tools Used\n")
        f.write("This report was generated using the following analysis tools:\n")
        f.write("- `position_counts.py`: Core analysis framework\n")
        f.write("- `inference_validation.py`: Cross-mode consistency checking\n")
        f.write("- `boundary_effects.py`: Boundary position investigation\n")
        f.write("- `pipeline_tracing.py`: Evaluation pipeline analysis\n")
        f.write("- `detailed_analysis.py`: Comprehensive question answering\n\n")
        
        f.write("For detailed analysis, run the individual tools or use --comprehensive mode.\n")
    
    print(f"✅ Report saved to: {output_file}")


def main():
    """Main driver function."""
    parser = argparse.ArgumentParser(
        description='Analyze position count behavior in SpliceAI inference',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Quick analysis
    python analyze_position_counts.py --genes ENSG00000142748
    
    # Comprehensive analysis with all tools
    python analyze_position_counts.py --genes ENSG00000142748 --comprehensive
    
    # Cross-mode validation
    python analyze_position_counts.py --genes ENSG00000142748 \\
        --validate-modes --model results/gene_cv_pc_1000_3mers_run_4 \\
        --training-dataset train_pc_1000_3mers
    
    # Generate report
    python analyze_position_counts.py --genes ENSG00000142748 ENSG00000000003 \\
        --report analysis_report.md
        """
    )
    
    parser.add_argument('--genes', nargs='+', required=True,
                       help='Gene IDs to analyze')
    parser.add_argument('--comprehensive', action='store_true',
                       help='Run comprehensive analysis with all tools')
    parser.add_argument('--validate-modes', action='store_true',
                       help='Validate consistency across inference modes')
    parser.add_argument('--report', 
                       help='Generate analysis report to specified file')
    parser.add_argument('--model', 
                       help='Path to trained model (required for mode validation)')
    parser.add_argument('--training-dataset',
                       help='Training dataset name (required for mode validation)')
    parser.add_argument('--quiet', action='store_true',
                       help='Reduce output verbosity')
    
    args = parser.parse_args()
    
    # Validate required arguments
    if args.validate_modes and (not args.model or not args.training_dataset):
        print("❌ Error: --validate-modes requires --model and --training-dataset")
        sys.exit(1)
    
    # Run requested analysis
    if args.comprehensive:
        run_comprehensive_analysis(args.genes)
    elif args.validate_modes:
        run_mode_validation(args.genes, args.model, args.training_dataset)
    elif args.report:
        generate_analysis_report(args.genes, args.report, args.model, args.training_dataset)
    else:
        run_quick_analysis(args.genes, args.model, args.training_dataset)


if __name__ == '__main__':
    main()
