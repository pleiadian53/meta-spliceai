#!/usr/bin/env python3
"""Unified Demo Runner for Meta-Model Inference Workflows
========================================================

This script provides a convenient way to run all inference workflow demos
or specific combinations with consistent parameters.

EXAMPLE USAGE:
    # Run all demos with default settings
    python run_inference_demos.py \
        --model results/gene_cv_pc_1000_3mers_run_4/model_multiclass.pkl \
        --training-dataset train_pc_1000_3mers \
        --genes ENSG00000104435,ENSG00000006420

    # Run specific demos
    python run_inference_demos.py \
        --model /path/to/model.pkl \
        --training-dataset /path/to/training \
        --genes-file test_genes.txt \
        --demos accuracy,scalability \
        --output-dir ./demo_results

    # Quick sanity check only
    python run_inference_demos.py \
        --model /path/to/model.pkl \
        --training-dataset /path/to/training \
        --genes ENSG00000104435 \
        --demos sanity \
        --positions 10000

FEATURES:
- Run all demos or specific subsets
- Consistent parameter handling across demos
- Consolidated results and reporting
- Error handling and recovery
- Progress tracking and timing
"""

import argparse
import sys
import json
import time
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any
import tempfile

# Add project root to Python path
import os
project_root = os.path.join(os.path.expanduser("~"), "work/splice-surveyor")
sys.path.insert(0, project_root)


class InferenceDemoRunner:
    """Unified runner for all inference workflow demos."""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.demo_scripts = {
            'accuracy': 'demo_accuracy_evaluation.py',
            'data': 'demo_data_management.py', 
            'sanity': 'demo_sanity_checks.py',
            'scalability': 'demo_scalability_analysis.py'
        }
        self.results = {}
        self.execution_times = {}
        self.errors = {}
    
    def build_command(
        self,
        script_name: str,
        model_path: Path,
        training_dataset_path: Path,
        genes_arg: Optional[str] = None,
        genes_file: Optional[str] = None,
        output_dir: Path = None,
        extra_args: Dict[str, Any] = None
    ) -> List[str]:
        """Build command line for a specific demo script."""
        
        script_path = Path(__file__).parent / script_name
        
        cmd = [
            sys.executable,
            str(script_path),
            '--model', str(model_path),
            '--training-dataset', str(training_dataset_path)
        ]
        
        # Gene specification
        if genes_file:
            cmd.extend(['--genes-file', genes_file])
        elif genes_arg:
            cmd.extend(['--genes', genes_arg])
        else:
            raise ValueError("Must provide either genes or genes-file")
        
        # Output directory
        if output_dir:
            cmd.extend(['--output-dir', str(output_dir)])
        
        # Add extra arguments specific to each demo
        if extra_args:
            for key, value in extra_args.items():
                if value is True:  # Boolean flags
                    cmd.append(f'--{key}')
                elif value is not False and value is not None:  # Value arguments
                    cmd.extend([f'--{key}', str(value)])
        
        # Verbosity
        if self.verbose:
            cmd.append('--verbose')
        else:
            cmd.append('--quiet')
        
        return cmd
    
    def run_demo_script(
        self,
        demo_name: str,
        model_path: Path,
        training_dataset_path: Path,
        genes_arg: Optional[str] = None,
        genes_file: Optional[str] = None,
        output_dir: Path = None,
        extra_args: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Run a single demo script and capture results."""
        
        if demo_name not in self.demo_scripts:
            return {
                'success': False,
                'error': f"Unknown demo: {demo_name}",
                'available_demos': list(self.demo_scripts.keys())
            }
        
        script_name = self.demo_scripts[demo_name]
        
        if self.verbose:
            print(f"\n🚀 Running {demo_name} demo ({script_name})")
        
        try:
            # Build command
            cmd = self.build_command(
                script_name=script_name,
                model_path=model_path,
                training_dataset_path=training_dataset_path,
                genes_arg=genes_arg,
                genes_file=genes_file,
                output_dir=output_dir,
                extra_args=extra_args or {}
            )
            
            if self.verbose:
                print(f"   🔧 Command: {' '.join(cmd)}")
            
            # Run the demo script
            start_time = time.time()
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout per demo
            )
            end_time = time.time()
            
            execution_time = end_time - start_time
            self.execution_times[demo_name] = execution_time
            
            if result.returncode == 0:
                if self.verbose:
                    print(f"   ✅ {demo_name} demo completed successfully ({execution_time:.1f}s)")
                
                # Try to find and read result file
                result_file = None
                if output_dir:
                    result_files = {
                        'accuracy': output_dir / 'accuracy_evaluation_results.json',
                        'data': output_dir / 'data_management_demo_results.json',
                        'sanity': output_dir / 'sanity_check_results.json',
                        'scalability': output_dir / 'scalability_analysis_results.json'
                    }
                    
                    if demo_name in result_files and result_files[demo_name].exists():
                        result_file = result_files[demo_name]
                
                demo_result = {
                    'success': True,
                    'execution_time': execution_time,
                    'stdout': result.stdout,
                    'stderr': result.stderr,
                    'command': cmd,
                    'result_file': str(result_file) if result_file else None
                }
                
                # Try to parse JSON results if available
                if result_file:
                    try:
                        with open(result_file, 'r') as f:
                            demo_result['parsed_results'] = json.load(f)
                    except Exception:
                        pass  # JSON parsing failed, but that's ok
                
                return demo_result
                
            else:
                error_msg = f"Demo failed with exit code {result.returncode}"
                if result.stderr:
                    error_msg += f": {result.stderr}"
                
                if self.verbose:
                    print(f"   ❌ {demo_name} demo failed ({execution_time:.1f}s)")
                    print(f"      Error: {error_msg}")
                
                return {
                    'success': False,
                    'error': error_msg,
                    'execution_time': execution_time,
                    'stdout': result.stdout,
                    'stderr': result.stderr,
                    'exit_code': result.returncode,
                    'command': cmd
                }
        
        except subprocess.TimeoutExpired:
            error_msg = f"Demo timed out after 5 minutes"
            if self.verbose:
                print(f"   ⏱️  {demo_name} demo timed out")
            
            return {
                'success': False,
                'error': error_msg,
                'timeout': True
            }
        
        except Exception as e:
            error_msg = f"Unexpected error: {e}"
            if self.verbose:
                print(f"   ❌ {demo_name} demo error: {error_msg}")
            
            return {
                'success': False,
                'error': error_msg,
                'exception': str(e)
            }
    
    def run_demos(
        self,
        demos_to_run: List[str],
        model_path: Path,
        training_dataset_path: Path,
        genes_arg: Optional[str] = None,
        genes_file: Optional[str] = None,
        output_dir: Path = None,
        demo_configs: Dict[str, Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Run specified demos and compile results."""
        
        print("🎯 Meta-Model Inference Workflow Demo Suite")
        print("=" * 50)
        print(f"   📁 Model: {model_path}")
        print(f"   📁 Training data: {training_dataset_path}")
        print(f"   🧬 Genes: {genes_arg or genes_file}")
        print(f"   🎬 Demos to run: {', '.join(demos_to_run)}")
        print(f"   📁 Output: {output_dir}")
        
        # Validate demos
        invalid_demos = [d for d in demos_to_run if d not in self.demo_scripts]
        if invalid_demos:
            print(f"❌ Invalid demos: {invalid_demos}")
            print(f"   Available: {list(self.demo_scripts.keys())}")
            return {
                'success': False,
                'error': f"Invalid demos: {invalid_demos}"
            }
        
        # Ensure output directory exists
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # Run each demo
        suite_start_time = time.time()
        demo_results = {}
        successful_demos = []
        failed_demos = []
        
        for demo_name in demos_to_run:
            # Get demo-specific configuration
            extra_args = demo_configs.get(demo_name, {}) if demo_configs else {}
            
            # Create demo-specific output directory
            demo_output_dir = output_dir / demo_name if output_dir else None
            if demo_output_dir:
                demo_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Run the demo
            result = self.run_demo_script(
                demo_name=demo_name,
                model_path=model_path,
                training_dataset_path=training_dataset_path,
                genes_arg=genes_arg,
                genes_file=genes_file,
                output_dir=demo_output_dir,
                extra_args=extra_args
            )
            
            demo_results[demo_name] = result
            
            if result['success']:
                successful_demos.append(demo_name)
            else:
                failed_demos.append(demo_name)
                self.errors[demo_name] = result.get('error', 'Unknown error')
        
        suite_end_time = time.time()
        total_time = suite_end_time - suite_start_time
        
        # Compile suite results
        suite_results = {
            'success': len(failed_demos) == 0,
            'total_execution_time': total_time,
            'demos_requested': demos_to_run,
            'demos_successful': successful_demos,
            'demos_failed': failed_demos,
            'demo_results': demo_results,
            'execution_times': self.execution_times,
            'errors': self.errors,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'config': {
                'model_path': str(model_path),
                'training_dataset_path': str(training_dataset_path),
                'genes_arg': genes_arg,
                'genes_file': genes_file,
                'output_dir': str(output_dir) if output_dir else None
            }
        }
        
        # Print summary
        if self.verbose:
            print(f"\n📊 Demo Suite Summary")
            print(f"   ⏱️  Total time: {total_time:.1f}s")
            print(f"   ✅ Successful: {len(successful_demos)}/{len(demos_to_run)}")
            
            if successful_demos:
                print(f"   🎉 Completed demos: {', '.join(successful_demos)}")
            
            if failed_demos:
                print(f"   ❌ Failed demos: {', '.join(failed_demos)}")
                for demo in failed_demos:
                    print(f"      {demo}: {self.errors.get(demo, 'Unknown error')}")
        
        # Save consolidated results
        if output_dir:
            results_file = output_dir / 'demo_suite_results.json'
            with open(results_file, 'w') as f:
                json.dump(suite_results, f, indent=2, default=str)
            
            if self.verbose:
                print(f"   💾 Suite results saved: {results_file}")
        
        return suite_results


def parse_genes_input(genes_arg: str, genes_file: Optional[str]) -> tuple:
    """Parse and validate gene input."""
    if genes_file:
        genes_path = Path(genes_file)
        if not genes_path.exists():
            raise FileNotFoundError(f"Gene file not found: {genes_file}")
        return None, genes_file
    elif genes_arg:
        return genes_arg, None
    else:
        raise ValueError("Must provide either --genes or --genes-file")


def parse_demos_list(demos_arg: str) -> List[str]:
    """Parse comma-separated list of demos."""
    if demos_arg.lower() == 'all':
        return ['accuracy', 'data', 'sanity', 'scalability']
    
    demos = [d.strip().lower() for d in demos_arg.split(',') if d.strip()]
    if not demos:
        raise ValueError("No valid demos specified")
    
    # Map common aliases
    demo_aliases = {
        'acc': 'accuracy',
        'eval': 'accuracy',
        'evaluation': 'accuracy',
        'data-mgmt': 'data',
        'management': 'data',
        'sanity-check': 'sanity',
        'checks': 'sanity',
        'scale': 'scalability',
        'perf': 'scalability',
        'performance': 'scalability'
    }
    
    normalized_demos = []
    for demo in demos:
        normalized_demos.append(demo_aliases.get(demo, demo))
    
    return normalized_demos


def main():
    parser = argparse.ArgumentParser(
        description="Unified Demo Runner for Meta-Model Inference Workflows",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLE USAGE:
    # RECOMMENDED: Run from project root (paths relative to project root)
    python -m meta_spliceai.splice_engine.meta_models.workflows.inference.run_inference_demos \\
        --model results/gene_cv_pc_1000_3mers_run_4/model_multiclass.pkl \\
        --training-dataset train_pc_1000_3mers \\
        --genes ENSG00000104435,ENSG00000006420 \\
        --demos all

    # ALTERNATIVE: Run script from project root
    cd /home/bchiu/work/splice-surveyor  # Change to your project root
    python meta_spliceai/splice_engine/meta_models/workflows/inference/run_inference_demos.py \\
        --model results/gene_cv_pc_1000_3mers_run_4/model_multiclass.pkl \\
        --training-dataset train_pc_1000_3mers \\
        --genes-file test_genes.txt \\
        --demos accuracy,scalability \\
        --output-dir ./results

    # Quick sanity check
    python -m meta_spliceai.splice_engine.meta_models.workflows.inference.run_inference_demos \\
        --model results/gene_cv_pc_1000_3mers_run_4/model_multiclass.pkl \\
        --training-dataset train_pc_1000_3mers \\
        --genes ENSG00000104435 \\
        --demos sanity \\
        --positions 5000

NOTE: Paths like "results/..." are relative to project root.
      Use absolute paths if running from different directories.

AVAILABLE DEMOS:
    - accuracy: F1-based evaluation (appropriate for imbalanced data)
    - data: Artifact storage and organization verification
    - sanity: Input-output consistency and reliability checks
    - scalability: Computational efficiency and selective featurization analysis

DEMO ALIASES:
    - all: Run all available demos
    - acc, eval, evaluation: accuracy demo
    - data-mgmt, management: data demo  
    - sanity-check, checks: sanity demo
    - scale, perf, performance: scalability demo
        """
    )
    
    # Required arguments
    parser.add_argument('--model', '-m', required=True, type=Path,
                       help='Path to pre-trained meta-model (.pkl file)')
    parser.add_argument('--training-dataset', '-t', required=True, type=Path,
                       help='Path to training dataset directory')
    
    # Gene specification (mutually exclusive)
    gene_group = parser.add_mutually_exclusive_group(required=True)
    gene_group.add_argument('--genes', '-g', type=str,
                           help='Comma-separated list of gene IDs')
    gene_group.add_argument('--genes-file', '-gf', type=str,
                           help='Path to file containing gene IDs (one per line)')
    
    # Demo selection
    parser.add_argument('--demos', '-d', type=str, default='all',
                       help='Comma-separated list of demos to run (default: all)')
    
    # Common optional arguments
    parser.add_argument('--output-dir', '-o', type=Path,
                       default=Path(tempfile.gettempdir()) / "inference_demos",
                       help='Output directory for all demo results')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Suppress all output except errors')
    
    # Demo-specific arguments
    parser.add_argument('--positions', type=int, default=10000,
                       help='Expected positions per gene (for sanity checks)')
    parser.add_argument('--gene-sizes', type=str, default='10000,25000,50000',
                       help='Gene sizes for scalability testing (comma-separated)')
    parser.add_argument('--max-positions', type=int,
                       help='Maximum positions per gene (memory limit)')
    parser.add_argument('--verify-reusability', action='store_true',
                       help='Test cache reusability in data management demo')
    parser.add_argument('--memory-profiling', action='store_true',
                       help='Enable memory profiling in scalability demo')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.model.exists():
        print(f"❌ Model file not found: {args.model}")
        return 1
    
    if not args.training_dataset.exists():
        print(f"❌ Training dataset not found: {args.training_dataset}")
        return 1
    
    # Parse genes
    try:
        genes_arg, genes_file = parse_genes_input(args.genes, args.genes_file)
    except (FileNotFoundError, ValueError) as e:
        print(f"❌ Gene input error: {e}")
        return 1
    
    # Parse demos
    try:
        demos_to_run = parse_demos_list(args.demos)
    except ValueError as e:
        print(f"❌ Demo selection error: {e}")
        return 1
    
    # Set verbosity
    verbose = args.verbose and not args.quiet
    
    try:
        # Create demo configurations
        demo_configs = {
            'sanity': {
                'positions': args.positions,
                'check-consistency': True
            },
            'scalability': {
                'gene-sizes': args.gene_sizes,
                'test-selective-featurization': True,
                'memory-profiling': args.memory_profiling
            },
            'data': {
                'verify-reusability': args.verify_reusability
            },
            'accuracy': {}
        }
        
        # Add common arguments
        if args.max_positions:
            demo_configs['sanity']['max-positions'] = args.max_positions
            demo_configs['accuracy']['max-positions'] = args.max_positions
        
        # Create runner and execute
        runner = InferenceDemoRunner(verbose=verbose)
        
        results = runner.run_demos(
            demos_to_run=demos_to_run,
            model_path=args.model,
            training_dataset_path=args.training_dataset,
            genes_arg=genes_arg,
            genes_file=genes_file,
            output_dir=args.output_dir,
            demo_configs=demo_configs
        )
        
        # Return appropriate exit code
        if results['success']:
            if not args.quiet:
                successful = len(results['demos_successful'])
                total = len(results['demos_requested'])
                print(f"\n✅ Demo suite completed successfully! ({successful}/{total} demos)")
            return 0
        else:
            failed_count = len(results['demos_failed'])
            total_count = len(results['demos_requested'])
            print(f"❌ Demo suite completed with {failed_count}/{total_count} failures")
            return 1
            
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())