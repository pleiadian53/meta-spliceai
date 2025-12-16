#!/usr/bin/env python3
"""
Main Position Count Analysis Driver

Simple, reliable driver script for position count analysis.
Provides easy access to all analysis capabilities.
"""

import sys
import os
import subprocess
from pathlib import Path

# Add meta_spliceai to path
sys.path.insert(0, str(Path(__file__).parents[5]))


def show_menu():
    """Show analysis options menu."""
    print("🧬 POSITION COUNT ANALYSIS - MAIN DRIVER")
    print("=" * 60)
    print()
    print("Available Analysis Tools:")
    print()
    print("1. 🚀 Quick Explanation - Understand position count behavior")
    print("2. 🔬 Detailed Analysis - Answer specific questions") 
    print("3. 🧪 Cross-Mode Validation - Test inference mode consistency")
    print("4. 🎯 Boundary Investigation - Analyze boundary effects")
    print("5. 🔍 Pipeline Tracing - Trace evaluation pipeline")
    print("6. 📊 Debug Tool - Focused debugging explanations")
    print()
    
    return input("Select analysis type (1-6, or 'q' to quit): ").strip()


def run_analysis_tool(tool_name: str, args: list = None):
    """Run a specific analysis tool."""
    script_path = Path(__file__).parent / f"{tool_name}.py"
    
    if not script_path.exists():
        print(f"❌ Analysis tool not found: {script_path}")
        return False
    
    cmd = ["python", str(script_path)]
    if args:
        cmd.extend(args)
    
    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"❌ Analysis tool failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Error running analysis tool: {e}")
        return False


def get_gene_input() -> list:
    """Get gene IDs from user."""
    while True:
        genes_input = input("Enter gene ID(s) separated by spaces (or press Enter for default): ").strip()
        if not genes_input:
            return ["ENSG00000142748"]  # Default gene
        
        genes = genes_input.split()
        if genes:
            return genes
        print("❌ Please enter valid gene ID(s)")


def get_model_info() -> tuple:
    """Get model information for validation."""
    print("\nFor cross-mode validation, enter model information:")
    
    model = input("Model path (or press Enter for default): ").strip()
    if not model:
        model = "results/gene_cv_pc_1000_3mers_run_4"
    
    dataset = input("Training dataset (or press Enter for default): ").strip()
    if not dataset:
        dataset = "train_pc_1000_3mers"
    
    return model, dataset


def main():
    """Main interactive driver."""
    while True:
        choice = show_menu()
        
        if choice.lower() == 'q':
            print("👋 Goodbye!")
            break
        
        try:
            choice_num = int(choice)
        except ValueError:
            print("❌ Please enter a valid number (1-6)")
            continue
        
        if choice_num == 1:  # Quick Explanation
            print("\n🚀 RUNNING QUICK EXPLANATION")
            print("=" * 40)
            run_analysis_tool("debug_position_counts", ["--case-study", "--asymmetry-analysis"])
            
        elif choice_num == 2:  # Detailed Analysis
            print("\n🔬 RUNNING DETAILED ANALYSIS")
            print("=" * 40)
            run_analysis_tool("detailed_analysis")
            
        elif choice_num == 3:  # Cross-Mode Validation
            genes = get_gene_input()
            model, dataset = get_model_info()
            
            print(f"\n🧪 RUNNING CROSS-MODE VALIDATION")
            print("=" * 40)
            
            # Run validation for each gene
            for gene in genes:
                print(f"\nTesting {gene} across inference modes...")
                
                modes = ['base_only', 'meta_only']
                for mode in modes:
                    print(f"🔬 Testing {mode} mode...")
                    cmd = [
                        "python", "-m", 
                        "meta_spliceai.splice_engine.meta_models.workflows.inference.main_inference_workflow",
                        "--model", model,
                        "--training-dataset", dataset,
                        "--genes", gene,
                        "--output-dir", f"temp_validation_{mode}_{gene}",
                        "--inference-mode", mode,
                        "--verbose"
                    ]
                    
                    try:
                        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
                        if result.returncode == 0:
                            # Extract position count
                            for line in result.stdout.split('\n'):
                                if '📊 Total positions:' in line:
                                    print(f"   ✅ {mode}: {line.strip()}")
                                    break
                        else:
                            print(f"   ❌ {mode}: Failed")
                    except subprocess.TimeoutExpired:
                        print(f"   ⏱️ {mode}: Timeout")
                    except Exception as e:
                        print(f"   ❌ {mode}: Error - {e}")
            
        elif choice_num == 4:  # Boundary Investigation
            print("\n🎯 RUNNING BOUNDARY INVESTIGATION")
            print("=" * 40)
            run_analysis_tool("boundary_effects")
            
        elif choice_num == 5:  # Pipeline Tracing
            print("\n🔍 RUNNING PIPELINE TRACING")
            print("=" * 40)
            run_analysis_tool("pipeline_tracing")
            
        elif choice_num == 6:  # Debug Tool
            genes = get_gene_input()
            print(f"\n📊 RUNNING DEBUG ANALYSIS")
            print("=" * 40)
            run_analysis_tool("debug_position_counts", ["--all"])
            
        else:
            print("❌ Please select a number between 1-6")
            continue
        
        print("\n" + "="*60)
        print("Analysis complete! Press Enter to continue or 'q' to quit...")
        next_action = input().strip()
        if next_action.lower() == 'q':
            print("👋 Goodbye!")
            break


if __name__ == '__main__':
    # Check if command line arguments provided
    if len(sys.argv) > 1:
        print("For command-line usage, use analyze_position_counts.py")
        print("This script provides interactive analysis mode.")
        print()
        print("Starting interactive mode...")
        print()
    
    main()

