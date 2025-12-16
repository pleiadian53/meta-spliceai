#!/usr/bin/env python3
"""Demo: Meta-Model Inference Scalability Analysis
================================================

This script demonstrates computational scalability analysis for meta-model inference,
focusing on selective featurization efficiency and memory optimization.

EXAMPLE USAGE:
    # RECOMMENDED: Run as module from project root (paths relative to project root):
    python -m meta_spliceai.splice_engine.meta_models.workflows.inference.demo_scalability_analysis \
        --model results/gene_cv_pc_1000_3mers_run_4/model_multiclass.pkl \
        --training-dataset train_pc_1000_3mers \
        --genes ENSG00000104435 \
        --gene-sizes 10000,25000,50000

    # ALTERNATIVE: Run script directly from project root:
    cd /home/bchiu/work/splice-surveyor  # Change to your project root
    python meta_spliceai/splice_engine/meta_models/workflows/inference/demo_scalability_analysis.py \
        --model results/gene_cv_pc_1000_3mers_run_4/model_multiclass.pkl \
        --training-dataset train_pc_1000_3mers \
        --genes ENSG00000104435 \
        --gene-sizes 10000,25000,50000

    # COMPREHENSIVE SCALABILITY TESTING:
    python -m meta_spliceai.splice_engine.meta_models.workflows.inference.demo_scalability_analysis \
        --model results/gene_cv_pc_1000_3mers_run_4/model_multiclass.pkl \
        --training-dataset train_pc_1000_3mers \
        --genes-file scalability_test_genes.txt \
        --gene-sizes 5000,10000,25000,50000,100000 \
        --test-selective-featurization \
        --memory-profiling \
        --verbose

FEATURES:
- Selective featurization efficiency measurement
- Memory usage analysis and optimization validation
- Runtime scalability across different gene sizes
- Feature matrix size reduction quantification
- Computational efficiency comparisons
- Throughput analysis (positions/second)

REQUIREMENTS:
- Pre-trained meta-model (.pkl file)  
- Training dataset directory
- Target genes for scalability testing
- Sufficient memory for large gene processing
"""

import argparse
import sys
import json
import time
import psutil
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import tempfile
import threading
import gc

import numpy as np
import polars as pl
import pandas as pd

# Add project root to Python path
import os
project_root = os.path.join(os.path.expanduser("~"), "work/splice-surveyor")
sys.path.insert(0, project_root)

from meta_spliceai.splice_engine.meta_models.workflows.selective_meta_inference import (
    run_selective_meta_inference,
    SelectiveInferenceConfig
)


class MemoryMonitor:
    """Monitor memory usage during operations."""
    
    def __init__(self):
        self.monitoring = False
        self.memory_samples = []
        self.peak_memory = 0
        self.initial_memory = 0
    
    def start_monitoring(self):
        """Start memory monitoring in background thread."""
        self.monitoring = True
        self.initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        self.peak_memory = self.initial_memory
        self.memory_samples = [self.initial_memory]
        
        def monitor():
            while self.monitoring:
                current_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
                self.memory_samples.append(current_memory)
                self.peak_memory = max(self.peak_memory, current_memory)
                time.sleep(0.1)  # Sample every 100ms
        
        self.thread = threading.Thread(target=monitor, daemon=True)
        self.thread.start()
    
    def stop_monitoring(self):
        """Stop memory monitoring and return statistics."""
        self.monitoring = False
        if hasattr(self, 'thread'):
            self.thread.join(timeout=1.0)
        
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        return {
            'initial_memory_mb': self.initial_memory,
            'peak_memory_mb': self.peak_memory,
            'final_memory_mb': final_memory,
            'memory_increase_mb': self.peak_memory - self.initial_memory,
            'samples_collected': len(self.memory_samples)
        }


class ScalabilityAnalyzer:
    """Analyze computational scalability of inference workflows."""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.results = {}
    
    def estimate_traditional_approach(self, gene_size: int, features_per_position: int = 500) -> Dict[str, Any]:
        """Estimate traditional approach resource requirements."""
        
        # Traditional approach: featurize all positions
        total_features = gene_size * features_per_position
        
        # Estimate memory (assuming float32)
        memory_mb = total_features * 4 / 1024 / 1024  # 4 bytes per float32
        
        # Estimate compute time (heuristic based on feature generation + model prediction)
        feature_generation_time = gene_size * 0.001  # 1ms per position
        model_prediction_time = gene_size * 0.0005    # 0.5ms per position prediction
        total_time = feature_generation_time + model_prediction_time
        
        return {
            'approach': 'traditional',
            'gene_size': gene_size,
            'positions_featurized': gene_size,
            'total_features': total_features,
            'estimated_memory_mb': memory_mb,
            'estimated_time_seconds': total_time,
            'features_per_position': features_per_position
        }
    
    def run_selective_inference_benchmark(
        self,
        model_path: Path,
        training_dataset_path: Path,
        gene_id: str,
        gene_size: int,
        uncertainty_low: float = 0.02,
        uncertainty_high: float = 0.80,
        monitor_memory: bool = True
    ) -> Dict[str, Any]:
        """Run selective inference and measure performance."""
        
        if self.verbose:
            print(f"      🔬 Running selective inference for {gene_id} ({gene_size:,} bp)")
        
        # Create temporary output directory
        with tempfile.TemporaryDirectory(prefix=f"scalability_{gene_id}_") as temp_dir:
            temp_dir = Path(temp_dir)
            
            # Start memory monitoring
            memory_monitor = MemoryMonitor() if monitor_memory else None
            if memory_monitor:
                memory_monitor.start_monitoring()
            
            try:
                # Configure selective inference
                selective_config = SelectiveInferenceConfig(
                    model_path=str(model_path),
                    target_genes=[gene_id],
                    training_dataset_path=str(training_dataset_path) if training_dataset_path else None,
                    uncertainty_threshold_low=uncertainty_low,
                    uncertainty_threshold_high=uncertainty_high,
                    max_positions_per_gene=gene_size,
                    inference_base_dir=temp_dir / "artifacts",
                    verbose=0,  # Suppress verbose output for cleaner benchmarking
                    cleanup_intermediates=False
                )
                
                # Run inference with timing
                start_time = time.time()
                workflow_results = run_selective_meta_inference(selective_config)
                end_time = time.time()
                
                # Stop memory monitoring
                memory_stats = memory_monitor.stop_monitoring() if memory_monitor else {}
                
                # Force garbage collection
                gc.collect()
                
                if not workflow_results.get('success', False):
                    raise RuntimeError("Selective inference failed")
                
                # Analyze results
                predictions = workflow_results['predictions']
                gene_predictions = predictions.filter(pl.col('gene_id') == gene_id)
                
                runtime_seconds = end_time - start_time
                positions_processed = len(gene_predictions)
                throughput = positions_processed / runtime_seconds if runtime_seconds > 0 else 0
                
                # Calculate efficiency metrics
                efficiency_ratio = positions_processed / gene_size
                memory_per_position = memory_stats.get('memory_increase_mb', 0) / positions_processed if positions_processed > 0 else 0
                
                benchmark_results = {
                    'approach': 'selective',
                    'gene_id': gene_id,
                    'gene_size': gene_size,
                    'positions_processed': positions_processed,
                    'positions_skipped': gene_size - positions_processed,
                    'efficiency_ratio': efficiency_ratio,
                    'runtime_seconds': runtime_seconds,
                    'throughput_positions_per_second': throughput,
                    'memory_per_position_mb': memory_per_position,
                    'uncertainty_thresholds': [uncertainty_low, uncertainty_high],
                    'memory_stats': memory_stats,
                    'success': True
                }
                
                if self.verbose:
                    print(f"         ✅ Processed {positions_processed:,}/{gene_size:,} positions ({efficiency_ratio:.1%})")
                    print(f"         ⏱️  Runtime: {runtime_seconds:.2f}s ({throughput:.0f} pos/s)")
                    if memory_stats:
                        print(f"         💾 Memory: {memory_stats.get('memory_increase_mb', 0):.1f} MB")
                
                return benchmark_results
                
            except Exception as e:
                if memory_monitor:
                    memory_monitor.stop_monitoring()
                
                return {
                    'approach': 'selective',
                    'gene_id': gene_id,
                    'gene_size': gene_size,
                    'success': False,
                    'error': str(e),
                    'runtime_seconds': 0
                }
    
    def compare_approaches(
        self,
        selective_results: Dict[str, Any],
        features_per_position: int = 500
    ) -> Dict[str, Any]:
        """Compare selective vs traditional approaches."""
        
        if not selective_results.get('success', False):
            return {
                'comparison_valid': False,
                'error': 'Selective inference failed'
            }
        
        gene_size = selective_results['gene_size']
        
        # Get traditional estimates
        traditional = self.estimate_traditional_approach(gene_size, features_per_position)
        
        # Calculate improvements
        positions_ratio = selective_results['positions_processed'] / traditional['positions_featurized']
        memory_reduction = 1 - (selective_results.get('memory_stats', {}).get('memory_increase_mb', 0) / traditional['estimated_memory_mb'])
        
        # Estimate speedup (selective only processes uncertain positions)
        time_reduction = 1 - positions_ratio  # Assuming time is proportional to positions processed
        estimated_speedup = 1 / (1 - time_reduction) if time_reduction < 1 else float('inf')
        
        comparison = {
            'comparison_valid': True,
            'gene_size': gene_size,
            'traditional': traditional,
            'selective': selective_results,
            'efficiency_gains': {
                'positions_reduction_ratio': 1 - positions_ratio,
                'memory_reduction_ratio': max(0, memory_reduction),  # Don't go negative
                'estimated_speedup': min(estimated_speedup, 100),   # Cap at 100x
                'positions_processed_selective': selective_results['positions_processed'],
                'positions_processed_traditional': traditional['positions_featurized'],
                'memory_saved_mb': max(0, traditional['estimated_memory_mb'] - selective_results.get('memory_stats', {}).get('memory_increase_mb', 0))
            }
        }
        
        return comparison
    
    def run_scalability_analysis(
        self,
        model_path: Path,
        training_dataset_path: Path,
        target_genes: List[str],
        gene_sizes: List[int],
        output_dir: Path,
        test_selective: bool = True,
        memory_profiling: bool = True,
        features_per_position: int = 500
    ) -> Dict[str, Any]:
        """Run comprehensive scalability analysis."""
        
        print("⚡ Meta-Model Inference Scalability Analysis")
        print("=" * 50)
        print(f"   📁 Model: {model_path}")
        print(f"   🧬 Target genes: {len(target_genes)}")
        print(f"   📏 Gene sizes: {gene_sizes}")
        print(f"   💾 Memory profiling: {memory_profiling}")
        
        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)
        
        analysis_results = {
            'config': {
                'model_path': str(model_path),
                'target_genes': target_genes,
                'gene_sizes': gene_sizes,
                'test_selective': test_selective,
                'memory_profiling': memory_profiling,
                'features_per_position': features_per_position
            },
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'gene_analysis': {},
            'scalability_trends': {},
            'efficiency_summary': {}
        }
        
        try:
            # Step 1: Analyze each gene at different sizes
            if self.verbose:
                print(f"\n🔬 STEP 1: Gene-by-Gene Scalability Analysis")
            
            for gene_id in target_genes:
                if self.verbose:
                    print(f"\n   📊 Analyzing gene: {gene_id}")
                
                gene_results = {
                    'gene_id': gene_id,
                    'size_benchmarks': {},
                    'scalability_metrics': {}
                }
                
                # Test different gene sizes
                for gene_size in gene_sizes:
                    if self.verbose:
                        print(f"      📏 Testing size: {gene_size:,} bp")
                    
                    # Run selective inference benchmark
                    if test_selective:
                        selective_results = self.run_selective_inference_benchmark(
                            model_path=model_path,
                            training_dataset_path=training_dataset_path,
                            gene_id=gene_id,
                            gene_size=gene_size,
                            monitor_memory=memory_profiling
                        )
                        
                        # Compare with traditional approach
                        comparison = self.compare_approaches(selective_results, features_per_position)
                        
                        gene_results['size_benchmarks'][gene_size] = {
                            'selective_results': selective_results,
                            'comparison': comparison
                        }
                    
                    # Brief pause to allow system recovery
                    time.sleep(0.5)
                
                # Calculate scalability metrics for this gene
                if gene_results['size_benchmarks']:
                    sizes = list(gene_results['size_benchmarks'].keys())
                    runtimes = [gene_results['size_benchmarks'][s]['selective_results']['runtime_seconds'] 
                               for s in sizes if gene_results['size_benchmarks'][s]['selective_results'].get('success')]
                    throughputs = [gene_results['size_benchmarks'][s]['selective_results']['throughput_positions_per_second']
                                  for s in sizes if gene_results['size_benchmarks'][s]['selective_results'].get('success')]
                    
                    if len(runtimes) >= 2:
                        # Calculate scaling relationship
                        size_ratios = [sizes[i] / sizes[0] for i in range(len(sizes))]
                        time_ratios = [runtimes[i] / runtimes[0] for i in range(len(runtimes))]
                        
                        # Linear scaling would have time_ratio = size_ratio
                        # Sublinear scaling has time_ratio < size_ratio (better)
                        scaling_efficiency = np.mean([size_ratios[i] / time_ratios[i] for i in range(len(time_ratios)) if time_ratios[i] > 0])
                        
                        gene_results['scalability_metrics'] = {
                            'scaling_efficiency': scaling_efficiency,  # >1.0 is sublinear (good)
                            'mean_throughput': np.mean(throughputs),
                            'throughput_variance': np.var(throughputs),
                            'sizes_tested': sizes,
                            'successful_tests': len(runtimes)
                        }
                
                analysis_results['gene_analysis'][gene_id] = gene_results
            
            # Step 2: Aggregate scalability trends
            if self.verbose:
                print(f"\n📈 STEP 2: Aggregating Scalability Trends")
            
            all_comparisons = []
            size_performance = {size: {'runtimes': [], 'memory_usage': [], 'efficiency_ratios': []} 
                              for size in gene_sizes}
            
            for gene_id, gene_data in analysis_results['gene_analysis'].items():
                for size, benchmark_data in gene_data['size_benchmarks'].items():
                    if benchmark_data['selective_results'].get('success'):
                        selective = benchmark_data['selective_results']
                        comparison = benchmark_data['comparison']
                        
                        all_comparisons.append(comparison)
                        
                        size_performance[size]['runtimes'].append(selective['runtime_seconds'])
                        size_performance[size]['memory_usage'].append(
                            selective.get('memory_stats', {}).get('memory_increase_mb', 0)
                        )
                        size_performance[size]['efficiency_ratios'].append(selective['efficiency_ratio'])
            
            # Calculate aggregate trends
            aggregate_trends = {}
            for size, metrics in size_performance.items():
                if metrics['runtimes']:
                    aggregate_trends[size] = {
                        'mean_runtime': np.mean(metrics['runtimes']),
                        'mean_memory_mb': np.mean(metrics['memory_usage']),
                        'mean_efficiency_ratio': np.mean(metrics['efficiency_ratios']),
                        'samples': len(metrics['runtimes'])
                    }
            
            analysis_results['scalability_trends'] = aggregate_trends
            
            # Step 3: Efficiency Summary
            if all_comparisons:
                valid_comparisons = [c for c in all_comparisons if c.get('comparison_valid', False)]
                
                if valid_comparisons:
                    efficiency_gains = [c['efficiency_gains'] for c in valid_comparisons]
                    
                    summary = {
                        'mean_positions_reduction': np.mean([e['positions_reduction_ratio'] for e in efficiency_gains]),
                        'mean_memory_reduction': np.mean([e['memory_reduction_ratio'] for e in efficiency_gains]),
                        'mean_estimated_speedup': np.mean([e['estimated_speedup'] for e in efficiency_gains]),
                        'median_positions_reduction': np.median([e['positions_reduction_ratio'] for e in efficiency_gains]),
                        'median_memory_reduction': np.median([e['memory_reduction_ratio'] for e in efficiency_gains]),
                        'total_comparisons': len(valid_comparisons),
                        'genes_analyzed': len(target_genes),
                        'sizes_tested': len(gene_sizes)
                    }
                    
                    analysis_results['efficiency_summary'] = summary
                    
                    if self.verbose:
                        print(f"   ✅ Efficiency Summary:")
                        print(f"      Mean position reduction: {summary['mean_positions_reduction']:.1%}")
                        print(f"      Mean memory reduction: {summary['mean_memory_reduction']:.1%}")
                        print(f"      Mean estimated speedup: {summary['mean_estimated_speedup']:.1f}x")
            
            analysis_results['success'] = True
            
            # Save results
            results_file = output_dir / "scalability_analysis_results.json"
            with open(results_file, 'w') as f:
                json.dump(analysis_results, f, indent=2, default=str)
            
            if self.verbose:
                print(f"\n✅ Scalability Analysis Completed!")
                print(f"   📁 Results saved: {results_file}")
                if 'efficiency_summary' in analysis_results:
                    summary = analysis_results['efficiency_summary']
                    print(f"   ⚡ Average efficiency gain: {summary['mean_estimated_speedup']:.1f}x speedup")
                    print(f"   💾 Average memory reduction: {summary['mean_memory_reduction']:.1%}")
            
            return analysis_results
            
        except Exception as e:
            analysis_results['success'] = False
            analysis_results['error'] = str(e)
            
            if self.verbose:
                print(f"\n❌ Scalability analysis failed: {e}")
            
            return analysis_results


def parse_genes_input(genes_arg: str, genes_file: Optional[str]) -> List[str]:
    """Parse gene input from command line or file."""
    if genes_file:
        genes_path = Path(genes_file)
        if not genes_path.exists():
            raise FileNotFoundError(f"Gene file not found: {genes_file}")
        
        with open(genes_path, 'r') as f:
            genes = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        
        if not genes:
            raise ValueError(f"No genes found in file: {genes_file}")
        
        return genes
    
    elif genes_arg:
        genes = [g.strip() for g in genes_arg.split(',') if g.strip()]
        if not genes:
            raise ValueError("No valid genes provided")
        return genes
    
    else:
        raise ValueError("Must provide either --genes or --genes-file")


def parse_gene_sizes(sizes_arg: str) -> List[int]:
    """Parse gene sizes from comma-separated string."""
    try:
        sizes = [int(s.strip()) for s in sizes_arg.split(',') if s.strip()]
        if not sizes:
            raise ValueError("No valid sizes provided")
        if any(s <= 0 for s in sizes):
            raise ValueError("All sizes must be positive")
        return sorted(sizes)
    except ValueError as e:
        raise ValueError(f"Invalid gene sizes format: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Demo: Meta-Model Inference Scalability Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLE USAGE:
    # RECOMMENDED: Run as module from project root:
    python -m meta_spliceai.splice_engine.meta_models.workflows.inference.demo_scalability_analysis \\
        --model results/gene_cv_pc_1000_3mers_run_4/model_multiclass.pkl \\
        --training-dataset train_pc_1000_3mers \\
        --genes ENSG00000104435 \\
        --gene-sizes 10000,25000,50000

    # ALTERNATIVE: Run script from project root:
    cd /home/bchiu/work/splice-surveyor  # Change to your project root
    python meta_spliceai/splice_engine/meta_models/workflows/inference/demo_scalability_analysis.py \\
        --model results/gene_cv_pc_1000_3mers_run_4/model_multiclass.pkl \\
        --training-dataset train_pc_1000_3mers \\
        --genes-file scalability_genes.txt \\
        --gene-sizes 5000,10000,25000,50000,100000 \\
        --test-selective-featurization \\
        --memory-profiling \\
        --verbose

NOTE: Paths like "results/..." are relative to project root.
      Use absolute paths if running from different directories.
        """
    )
    
    # Required arguments
    parser.add_argument('--model', '-m', required=True, type=Path,
                       help='Path to pre-trained meta-model (.pkl file)')
    parser.add_argument('--training-dataset', '-t', required=True, type=Path,
                       help='Path to training dataset directory')
    parser.add_argument('--gene-sizes', '-s', required=True, type=str,
                       help='Comma-separated list of gene sizes to test (e.g., 10000,25000,50000)')
    
    # Gene specification (mutually exclusive)
    gene_group = parser.add_mutually_exclusive_group(required=True)
    gene_group.add_argument('--genes', '-g', type=str,
                           help='Comma-separated list of gene IDs')
    gene_group.add_argument('--genes-file', '-gf', type=str,
                           help='Path to file containing gene IDs (one per line)')
    
    # Optional arguments
    parser.add_argument('--output-dir', '-o', type=Path,
                       default=Path(tempfile.gettempdir()) / "scalability_analysis",
                       help='Output directory for results')
    parser.add_argument('--features-per-position', type=int, default=500,
                       help='Expected features per position for traditional approach (default: 500)')
    parser.add_argument('--test-selective-featurization', action='store_true',
                       help='Test selective featurization efficiency')
    parser.add_argument('--memory-profiling', action='store_true',
                       help='Enable detailed memory profiling')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Suppress all output except errors')
    
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
        target_genes = parse_genes_input(args.genes, args.genes_file)
    except (FileNotFoundError, ValueError) as e:
        print(f"❌ Gene input error: {e}")
        return 1
    
    # Parse gene sizes
    try:
        gene_sizes = parse_gene_sizes(args.gene_sizes)
    except ValueError as e:
        print(f"❌ Gene sizes error: {e}")
        return 1
    
    # Set verbosity
    verbose = args.verbose and not args.quiet
    
    try:
        # Create analyzer and run
        analyzer = ScalabilityAnalyzer(verbose=verbose)
        
        results = analyzer.run_scalability_analysis(
            model_path=args.model,
            training_dataset_path=args.training_dataset,
            target_genes=target_genes,
            gene_sizes=gene_sizes,
            output_dir=args.output_dir,
            test_selective=args.test_selective_featurization,
            memory_profiling=args.memory_profiling,
            features_per_position=args.features_per_position
        )
        
        if results['success']:
            if not args.quiet:
                print(f"\n✅ Scalability analysis completed!")
                if 'efficiency_summary' in results:
                    summary = results['efficiency_summary']
                    print(f"⚡ Average speedup: {summary['mean_estimated_speedup']:.1f}x")
                    print(f"💾 Memory reduction: {summary['mean_memory_reduction']:.1%}")
                    print(f"📊 Genes analyzed: {summary['genes_analyzed']}")
            return 0
        else:
            print(f"❌ Analysis failed: {results.get('error', 'Unknown error')}")
            return 1
            
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())