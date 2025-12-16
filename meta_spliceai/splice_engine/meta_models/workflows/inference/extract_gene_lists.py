#!/usr/bin/env python3
"""
Gene List Extractor for Inference Workflow

Automatically extracts gene lists from find_test_genes.sh output and creates
individual gene files ready for use with main_inference_workflow.py.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional
import sys

def extract_gene_lists(
    input_file: str,
    output_dir: str = ".",
    scenarios: Optional[List[str]] = None,
    prefix: str = "",
    create_combined: bool = False
) -> Dict[str, str]:
    """
    Extract gene lists from test gene identification results.
    
    Args:
        input_file: JSON file from find_test_genes.sh
        output_dir: Directory to save gene list files
        scenarios: List of scenarios to extract (default: all)
        prefix: Prefix for output filenames
        create_combined: Whether to create a combined file with all genes
        
    Returns:
        Dict mapping scenario names to output file paths
    """
    # Load gene discovery results
    try:
        with open(input_file, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"❌ Error: Input file not found: {input_file}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"❌ Error: Invalid JSON in input file: {e}")
        sys.exit(1)
    
    # Default to all scenarios if none specified
    if scenarios is None:
        scenarios = ['scenario1', 'scenario2a', 'scenario2b']
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Extract genes for each scenario
    output_files = {}
    all_genes = []
    
    for scenario in scenarios:
        if scenario not in data:
            print(f"⚠️ Warning: {scenario} not found in input file")
            continue
        
        scenario_data = data[scenario]
        if not scenario_data.get('genes'):
            print(f"⚠️ Warning: No genes found for {scenario}")
            continue
        
        # Extract gene IDs
        genes = [g['gene_id'] for g in scenario_data['genes']]
        all_genes.extend(genes)
        
        # Create output filename
        if prefix:
            filename = f"{prefix}_{scenario}_genes.txt"
        else:
            filename = f"{scenario}_genes.txt"
        
        output_file = output_path / filename
        
        # Write gene list
        with open(output_file, 'w') as f:
            for gene in genes:
                f.write(f"{gene}\n")
        
        output_files[scenario] = str(output_file)
        
        print(f"✅ {scenario}: {len(genes)} genes → {output_file}")
    
    # Create combined file if requested
    if create_combined and all_genes:
        combined_filename = f"{prefix}_all_genes.txt" if prefix else "all_genes.txt"
        combined_file = output_path / combined_filename
        
        with open(combined_file, 'w') as f:
            for gene in all_genes:
                f.write(f"{gene}\n")
        
        output_files['combined'] = str(combined_file)
        print(f"✅ Combined: {len(all_genes)} genes → {combined_file}")
    
    return output_files

def generate_usage_examples(output_files: Dict[str, str], model_path: str, 
                          training_dataset: str, study_name: str) -> None:
    """Generate usage examples for the extracted gene files."""
    
    print("\n" + "="*80)
    print("🧪 READY-TO-USE INFERENCE COMMANDS")
    print("="*80)
    
    # Generate commands for each scenario
    for scenario, gene_file in output_files.items():
        if scenario == 'combined':
            continue
            
        print(f"\n# {scenario.upper()} - {Path(gene_file).name}")
        
        for mode in ['base_only', 'hybrid', 'meta_only']:
            print(f"\n# {mode} mode")
            print(f"python -m meta_spliceai.splice_engine.meta_models.workflows.inference.main_inference_workflow \\")
            print(f"    --model {model_path} \\")
            print(f"    --training-dataset {training_dataset} \\")
            print(f"    --genes-file {gene_file} \\")
            print(f"    --output-dir results/{study_name}_{scenario}_{mode} \\")
            print(f"    --inference-mode {mode} \\")
            print(f"    --enable-chunked-processing \\")
            print(f"    --chunk-size 5000 \\")
            
            # Add mode-specific options
            if mode == "meta_only":
                print(f"    --complete-coverage \\")
            elif mode == "hybrid":
                print(f"    --uncertainty-low 0.02 \\")
                print(f"    --uncertainty-high 0.80 \\")
            
            print(f"    --verbose \\")
            print(f"    --mlflow-enable \\")
            print(f"    --mlflow-experiment \"{study_name}\" \\")
            print(f"    2>&1 | tee logs/{scenario}_{mode}.log")
    
    # Combined analysis command
    if len(output_files) > 1:
        print(f"\n" + "="*80)
        print("📊 ANALYSIS COMMANDS")
        print("="*80)
        
        print(f"\n# Analyze all results")
        print(f"python -m meta_spliceai.splice_engine.meta_models.workflows.inference.inference_analyzer \\")
        print(f"    --results-dir results \\")
        print(f"    --base-suffix {study_name}_scenario2b_base_only \\")  # Use scenario2b as default
        print(f"    --hybrid-suffix {study_name}_scenario2b_hybrid \\")
        print(f"    --meta-suffix {study_name}_scenario2b_meta_only \\")
        print(f"    --output-dir {study_name}_analysis_results \\")
        print(f"    --batch-size 25 \\")
        print(f"    --verbose")
        
        print(f"\n# Statistical comparison")
        print(f"python -m meta_spliceai.splice_engine.meta_models.workflows.inference.batch_comparator \\")
        print(f"    --analysis-results {study_name}_analysis_results/detailed_report.json \\")
        print(f"    --output-dir {study_name}_statistical_comparison \\")
        print(f"    --reference-mode base_only \\")
        print(f"    --create-plots")

def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description="Extract gene lists from test gene discovery results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:

# Basic extraction (creates scenario1_genes.txt, scenario2a_genes.txt, scenario2b_genes.txt)
python extract_gene_lists.py --input test_genes.json

# Extract with custom prefix and combined file
python extract_gene_lists.py \\
    --input large_study_genes.json \\
    --prefix large_study \\
    --create-combined \\
    --output-dir gene_lists

# Extract only specific scenarios
python extract_gene_lists.py \\
    --input test_genes.json \\
    --scenarios scenario2b \\
    --prefix unseen_only

# Extract and generate usage examples
python extract_gene_lists.py \\
    --input test_genes.json \\
    --prefix my_study \\
    --study-name "my_analysis" \\
    --show-examples
        """
    )
    
    parser.add_argument("--input", "-i", required=True,
                       help="Input JSON file from find_test_genes.sh")
    parser.add_argument("--output-dir", "-o", default=".",
                       help="Output directory for gene list files (default: current directory)")
    parser.add_argument("--scenarios", nargs="+", 
                       choices=["scenario1", "scenario2a", "scenario2b"],
                       help="Scenarios to extract (default: all)")
    parser.add_argument("--prefix", default="",
                       help="Prefix for output filenames")
    parser.add_argument("--create-combined", action="store_true",
                       help="Create a combined file with all genes")
    parser.add_argument("--show-examples", action="store_true",
                       help="Show usage examples for the extracted gene files")
    parser.add_argument("--study-name", default="comparison_study",
                       help="Study name for example commands (default: comparison_study)")
    parser.add_argument("--model-path", default="results/gene_cv_pc_1000_3mers_run_4",
                       help="Model path for example commands")
    parser.add_argument("--training-dataset", default="train_pc_1000_3mers",
                       help="Training dataset for example commands")
    
    args = parser.parse_args()
    
    # Extract gene lists
    print("🧬 EXTRACTING GENE LISTS FOR INFERENCE WORKFLOW")
    print("="*60)
    
    output_files = extract_gene_lists(
        input_file=args.input,
        output_dir=args.output_dir,
        scenarios=args.scenarios,
        prefix=args.prefix,
        create_combined=args.create_combined
    )
    
    if not output_files:
        print("❌ No gene files created")
        sys.exit(1)
    
    print(f"\n✅ Successfully created {len(output_files)} gene list files")
    
    # Show usage examples if requested
    if args.show_examples:
        generate_usage_examples(
            output_files=output_files,
            model_path=args.model_path,
            training_dataset=args.training_dataset,
            study_name=args.study_name
        )
    else:
        print(f"\n💡 Use --show-examples to see ready-to-use inference commands")
    
    print(f"\n📁 Gene list files saved to: {args.output_dir}")

if __name__ == "__main__":
    main()




