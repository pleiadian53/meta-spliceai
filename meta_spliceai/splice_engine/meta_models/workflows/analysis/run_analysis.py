#!/usr/bin/env python3
"""
Main Analysis Driver Script

This is the primary entry point for all position count analysis tasks.
It provides a unified interface to access all analysis tools and capabilities.

USAGE:
    # Interactive mode - guides you through analysis options
    python run_analysis.py
    
    # Quick gene analysis
    python run_analysis.py --quick --genes ENSG00000142748
    
    # Full comprehensive analysis
    python run_analysis.py --comprehensive --genes ENSG00000142748 ENSG00000000003
    
    # Cross-mode validation
    python run_analysis.py --validate --genes ENSG00000142748 \\
        --model results/gene_cv_pc_1000_3mers_run_4 \\
        --training-dataset train_pc_1000_3mers
"""

import sys
import argparse
from pathlib import Path
from typing import List, Optional

# Add meta_spliceai to path
sys.path.insert(0, str(Path(__file__).parents[5]))


def show_analysis_menu():
    """Show interactive analysis menu."""
    print("🧬 SPLICE SURVEYOR - POSITION COUNT ANALYSIS")
    print("=" * 60)
    print()
    print("Available Analysis Tools:")
    print("1. 🚀 Quick Analysis - Basic position count explanation")
    print("2. 🔬 Comprehensive Analysis - All analysis tools")
    print("3. 🧪 Cross-Mode Validation - Test inference mode consistency")
    print("4. 📋 Generate Report - Create detailed analysis report")
    print("5. 🎯 Specific Questions - Answer common position count questions")
    print("6. 🔍 Boundary Investigation - Analyze boundary effects")
    print("7. 🔬 Pipeline Tracing - Trace evaluation pipeline")
    print()
    
    while True:
        try:
            choice = input("Select analysis type (1-7, or 'q' to quit): ").strip()
            if choice.lower() == 'q':
                print("👋 Goodbye!")
                sys.exit(0)
            
            choice_num = int(choice)
            if 1 <= choice_num <= 7:
                return choice_num
            else:
                print("❌ Please enter a number between 1-7")
        except ValueError:
            print("❌ Please enter a valid number")


def get_gene_input() -> List[str]:
    """Get gene IDs from user input."""
    while True:
        genes_input = input("Enter gene ID(s) separated by spaces: ").strip()
        if genes_input:
            genes = genes_input.split()
            print(f"✅ Will analyze: {', '.join(genes)}")
            return genes
        print("❌ Please enter at least one gene ID")


def get_model_info() -> tuple[str, str]:
    """Get model and training dataset information."""
    print("\nFor cross-mode validation, we need model information:")
    
    model_path = input("Model path (or press Enter for default): ").strip()
    if not model_path:
        model_path = "results/gene_cv_pc_1000_3mers_run_4"
    
    training_dataset = input("Training dataset (or press Enter for default): ").strip()
    if not training_dataset:
        training_dataset = "train_pc_1000_3mers"
    
    return model_path, training_dataset


def run_interactive_mode():
    """Run interactive analysis mode."""
    print("🎯 INTERACTIVE ANALYSIS MODE")
    print("=" * 40)
    
    # Get analysis type
    choice = show_analysis_menu()
    
    # Get gene IDs
    genes = get_gene_input()
    
    # Run selected analysis
    if choice == 1:  # Quick Analysis
        from .analyze_position_counts import run_quick_analysis
        run_quick_analysis(genes)
        
    elif choice == 2:  # Comprehensive Analysis
        from .analyze_position_counts import run_comprehensive_analysis
        run_comprehensive_analysis(genes)
        
    elif choice == 3:  # Cross-Mode Validation
        model_path, training_dataset = get_model_info()
        from .analyze_position_counts import run_mode_validation
        run_mode_validation(genes, model_path, training_dataset)
        
    elif choice == 4:  # Generate Report
        output_file = input("Report filename (or press Enter for default): ").strip()
        if not output_file:
            output_file = f"position_analysis_report_{'_'.join(genes)}.md"
        
        model_path, training_dataset = get_model_info()
        from .analyze_position_counts import generate_analysis_report
        generate_analysis_report(genes, output_file, model_path, training_dataset)
        
    elif choice == 5:  # Specific Questions
        print("\n🎯 Running specific questions analysis...")
        run_detailed_analysis()
        
    elif choice == 6:  # Boundary Investigation
        print("\n🔍 Running boundary investigation...")
        run_boundary_analysis()
        
    elif choice == 7:  # Pipeline Tracing
        print("\n🔬 Running pipeline tracing...")
        run_pipeline_analysis()


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Position Count Analysis - Main Driver Script',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Analysis Types:
    --quick         Quick position count explanation
    --comprehensive Full analysis with all tools  
    --validate      Cross-mode validation
    --report        Generate detailed report

Examples:
    # Interactive mode
    python run_analysis.py
    
    # Quick analysis
    python run_analysis.py --quick --genes ENSG00000142748
    
    # Comprehensive analysis
    python run_analysis.py --comprehensive --genes ENSG00000142748 ENSG00000000003
    
    # Cross-mode validation
    python run_analysis.py --validate --genes ENSG00000142748 \\
        --model results/gene_cv_pc_1000_3mers_run_4 \\
        --training-dataset train_pc_1000_3mers
        """
    )
    
    # Analysis type (mutually exclusive)
    analysis_group = parser.add_mutually_exclusive_group()
    analysis_group.add_argument('--quick', action='store_true',
                               help='Run quick position count analysis')
    analysis_group.add_argument('--comprehensive', action='store_true',
                               help='Run comprehensive analysis with all tools')
    analysis_group.add_argument('--validate', action='store_true',
                               help='Validate consistency across inference modes')
    analysis_group.add_argument('--report', 
                               help='Generate analysis report to specified file')
    
    # Common arguments
    parser.add_argument('--genes', nargs='+',
                       help='Gene IDs to analyze (required for non-interactive mode)')
    parser.add_argument('--model', 
                       help='Path to trained model (required for validation)')
    parser.add_argument('--training-dataset',
                       help='Training dataset name (required for validation)')
    parser.add_argument('--quiet', action='store_true',
                       help='Reduce output verbosity')
    
    args = parser.parse_args()
    
    # Check if any analysis type specified
    analysis_specified = any([args.quick, args.comprehensive, args.validate, args.report])
    
    if not analysis_specified:
        # Interactive mode
        run_interactive_mode()
        return
    
    # Validate required arguments for non-interactive mode
    if not args.genes:
        print("❌ Error: --genes required for non-interactive mode")
        sys.exit(1)
    
    if args.validate and (not args.model or not args.training_dataset):
        print("❌ Error: --validate requires --model and --training-dataset")
        sys.exit(1)
    
    # Run requested analysis
    if args.quick:
        from .analyze_position_counts import run_quick_analysis
        run_quick_analysis(args.genes, args.model, args.training_dataset)
        
    elif args.comprehensive:
        from .analyze_position_counts import run_comprehensive_analysis
        run_comprehensive_analysis(args.genes)
        
    elif args.validate:
        from .analyze_position_counts import run_mode_validation
        run_mode_validation(args.genes, args.model, args.training_dataset)
        
    elif args.report:
        from .analyze_position_counts import generate_analysis_report
        generate_analysis_report(args.genes, args.report, args.model, args.training_dataset)


if __name__ == '__main__':
    main()

