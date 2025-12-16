#!/usr/bin/env python3
"""
Generate Model Comparison Workflow Commands

This utility generates the complete set of commands needed to run a model comparison
study, from gene discovery through statistical analysis.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

def generate_workflow_commands(
    gene_file: str,
    scenario: str = "scenario2b",
    study_name: str = "comparison_study",
    model_path: str = "results/gene_cv_pc_1000_3mers_run_4",
    training_dataset: str = "train_pc_1000_3mers",
    chunk_size: int = 5000,
    enable_mlflow: bool = True,
    output_file: str = None
) -> List[str]:
    """Generate workflow commands."""
    
    commands = []
    
    # Header
    commands.append("#!/bin/bash")
    commands.append("# Generated Model Comparison Workflow")
    commands.append(f"# Study: {study_name}")
    commands.append(f"# Gene file: {gene_file}")
    commands.append(f"# Scenario: {scenario}")
    commands.append("")
    commands.append("# Activate environment")
    commands.append("mamba activate surveyor")
    commands.append("")
    
    # Create directories
    commands.append("# Create output directories")
    commands.append("mkdir -p logs")
    commands.append(f"mkdir -p results")
    commands.append("")
    
    # Extract genes from JSON if needed
    if gene_file.endswith('.json'):
        commands.append("# Extract gene list from discovery results")
        commands.append('python3 -c "')
        commands.append('import json')
        commands.append(f'with open(\\"{gene_file}\\") as f:')
        commands.append('    data = json.load(f)')
        commands.append(f'genes = [g[\\"gene_id\\"] for g in data[\\"{scenario}\\"][\\"genes\\"]]')
        commands.append(f'with open(\\"{scenario}_genes.txt\\", \\"w\\") as f:')
        commands.append('    for gene in genes:')
        commands.append('        f.write(f\\"{gene}\\\\n\\")')
        commands.append(f'print(f\\"Extracted {{len(genes)}} genes for {scenario}\\")')
        commands.append('"')
        commands.append("")
        gene_file_arg = f"{scenario}_genes.txt"
    else:
        gene_file_arg = gene_file
    
    # MLflow options
    mlflow_args = []
    if enable_mlflow:
        mlflow_args = [
            "--mlflow-enable",
            f'--mlflow-experiment "{study_name}"',
            f'--mlflow-tags study_type={scenario} scenario={scenario}'
        ]
    
    # Generate inference commands for each mode
    modes = [
        ("base_only", "Base-only (SpliceAI only)"),
        ("hybrid", "Hybrid (SpliceAI + Meta-model for uncertain positions)"), 
        ("meta_only", "Meta-only (Meta-model for all positions)")
    ]
    
    for mode, description in modes:
        commands.append(f"# {description}")
        commands.append(f"echo \"Running {mode} mode...\"")
        
        cmd_parts = [
            "python -m meta_spliceai.splice_engine.meta_models.workflows.inference.main_inference_workflow",
            f"    --model {model_path}",
            f"    --training-dataset {training_dataset}",
            f"    --genes-file {gene_file_arg}",
            f"    --output-dir results/{study_name}_{mode}",
            f"    --inference-mode {mode}",
            f"    --enable-chunked-processing",
            f"    --chunk-size {chunk_size}",
        ]
        
        # Add mode-specific options
        if mode == "meta_only":
            cmd_parts.append("    --complete-coverage")
        elif mode == "hybrid":
            cmd_parts.extend([
                "    --uncertainty-low 0.02",
                "    --uncertainty-high 0.80"
            ])
        
        cmd_parts.extend([
            "    --verbose",
        ])
        
        # Add MLflow options
        cmd_parts.extend(f"    {arg}" for arg in mlflow_args)
        
        # Add log redirection
        cmd_parts.append(f"    2>&1 | tee logs/{mode}_inference.log")
        
        # Join command with backslashes
        command = " \\\n".join(cmd_parts)
        commands.append(command)
        commands.append("")
        commands.append(f'echo "Completed {mode} mode"')
        commands.append("")
    
    # Analysis commands
    commands.append("# Analyze results")
    commands.append("python -m meta_spliceai.splice_engine.meta_models.workflows.inference.inference_analyzer \\")
    commands.append("    --results-dir results \\")
    commands.append(f"    --base-suffix {study_name}_base_only \\")
    commands.append(f"    --hybrid-suffix {study_name}_hybrid \\")
    commands.append(f"    --meta-suffix {study_name}_meta_only \\")
    commands.append(f"    --output-dir {study_name}_analysis_results \\")
    commands.append("    --batch-size 25 \\")
    commands.append("    --verbose")
    commands.append("")
    
    # Statistical comparison
    commands.append("# Statistical comparison")
    commands.append("python -m meta_spliceai.splice_engine.meta_models.workflows.inference.batch_comparator \\")
    commands.append(f"    --analysis-results {study_name}_analysis_results/detailed_report.json \\")
    commands.append(f"    --output-dir {study_name}_statistical_comparison \\")
    commands.append("    --reference-mode base_only \\")
    commands.append("    --significance-level 0.05 \\")
    commands.append("    --primary-metric ap_score \\")
    commands.append("    --include-effect-sizes \\")
    commands.append("    --create-plots \\")
    commands.append("    --verbose")
    commands.append("")
    
    # Summary
    commands.append("echo \"\"")
    commands.append("echo \"✅ Model comparison workflow completed!\"")
    commands.append(f"echo \"📁 Results available in:\"")
    commands.append(f"echo \"   - results/{study_name}_*/\"")
    commands.append(f"echo \"   - {study_name}_analysis_results/\"") 
    commands.append(f"echo \"   - {study_name}_statistical_comparison/\"")
    if enable_mlflow:
        commands.append("echo \"   - MLflow: http://localhost:5000\"")
    
    return commands

def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description="Generate model comparison workflow commands",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:

# Generate workflow for scenario2b genes from gene discovery
python generate_comparison_workflow.py \\
    --gene-file test_genes.json \\
    --scenario scenario2b \\
    --study-name unseen_genes_study \\
    --output workflow_unseen_genes.sh

# Generate workflow for custom gene list
python generate_comparison_workflow.py \\
    --gene-file my_genes.txt \\
    --study-name my_custom_study \\
    --chunk-size 3000 \\
    --output workflow_custom.sh

# Generate workflow without MLflow
python generate_comparison_workflow.py \\
    --gene-file test_genes.json \\
    --scenario scenario1 \\
    --study-name training_genes_study \\
    --no-mlflow \\
    --output workflow_no_mlflow.sh
        """
    )
    
    parser.add_argument("--gene-file", required=True,
                       help="Gene file (JSON from gene discovery or TXT with gene IDs)")
    parser.add_argument("--scenario", default="scenario2b",
                       choices=["scenario1", "scenario2a", "scenario2b"],
                       help="Scenario to extract from JSON file (default: scenario2b)")
    parser.add_argument("--study-name", default="comparison_study",
                       help="Name for the study (used in directory names)")
    parser.add_argument("--model-path", default="results/gene_cv_pc_1000_3mers_run_4",
                       help="Path to trained meta-model")
    parser.add_argument("--training-dataset", default="train_pc_1000_3mers",
                       help="Training dataset path")
    parser.add_argument("--chunk-size", type=int, default=5000,
                       help="Chunk size for processing")
    parser.add_argument("--no-mlflow", action="store_true",
                       help="Disable MLflow tracking")
    parser.add_argument("--output", default="model_comparison_workflow.sh",
                       help="Output shell script file")
    
    args = parser.parse_args()
    
    # Generate commands
    commands = generate_workflow_commands(
        gene_file=args.gene_file,
        scenario=args.scenario,
        study_name=args.study_name,
        model_path=args.model_path,
        training_dataset=args.training_dataset,
        chunk_size=args.chunk_size,
        enable_mlflow=not args.no_mlflow,
        output_file=args.output
    )
    
    # Write to file
    with open(args.output, 'w') as f:
        f.write('\n'.join(commands))
    
    # Make executable
    import os
    os.chmod(args.output, 0o755)
    
    print(f"✅ Generated workflow script: {args.output}")
    print(f"📋 Run with: bash {args.output}")
    print(f"🔧 Edit the script to customize parameters before running")

if __name__ == "__main__":
    main()
