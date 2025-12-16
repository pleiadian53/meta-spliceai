#!/usr/bin/env python3
"""
Performance analysis utility for meta-model training results.

This script analyzes training results and generates comprehensive performance reports
including CV metrics, meta vs base comparisons, and file structure validation.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional

import pandas as pd


def analyze_training_performance(results_dir: str, output_file: Optional[str] = None) -> Dict:
    """
    Analyze training performance from results directory.
    
    Args:
        results_dir: Directory containing training results
        output_file: Optional output file for detailed analysis
        
    Returns:
        Dictionary containing performance analysis results
    """
    results_dir = Path(results_dir)
    
    print("📊 TRAINING PERFORMANCE ANALYSIS")
    print("=" * 60)
    print(f"Analyzing results from: {results_dir}")
    
    if not results_dir.exists():
        print(f"❌ Results directory not found: {results_dir}")
        return {'error': 'Results directory not found'}
    
    # Initialize results
    analysis = {
        'results_dir': str(results_dir),
        'cv_performance': {},
        'meta_vs_base': {},
        'file_structure': {},
        'summary': {},
        'issues': []
    }
    
    # Check for key files
    key_files = {
        'cv_metrics': 'gene_cv_metrics.csv',
        'meta_vs_base': 'meta_vs_base_performance.tsv',
        'model': 'model_multiclass.pkl',
        'roc_curves': 'roc_curves_meta.pdf',
        'pr_curves': 'pr_curves_meta.pdf',
        'training_summary_txt': 'training_summary.txt',
        'training_summary_json': 'training_summary.json'
    }
    
    print(f"\n📁 FILE STRUCTURE VALIDATION:")
    print("-" * 40)
    
    for file_key, filename in key_files.items():
        file_path = results_dir / filename
        exists = file_path.exists()
        analysis['file_structure'][file_key] = {
            'filename': filename,
            'exists': exists,
            'path': str(file_path) if exists else None
        }
        
        status = "✅" if exists else "❌"
        print(f"  {status} {filename}")
        
        if not exists:
            analysis['issues'].append(f"Missing file: {filename}")
    
    # Analyze CV performance
    cv_file = results_dir / 'gene_cv_metrics.csv'
    if cv_file.exists():
        try:
            cv_metrics = pd.read_csv(cv_file)
            
            print(f"\n🎯 CV PERFORMANCE (5-fold):")
            print("-" * 30)
            
            # Key metrics
            metrics = ['test_macro_f1', 'test_accuracy', 'top_k_accuracy']
            cv_stats = {}
            
            for metric in metrics:
                if metric in cv_metrics.columns:
                    mean_val = cv_metrics[metric].mean()
                    std_val = cv_metrics[metric].std()
                    cv_stats[metric] = {'mean': mean_val, 'std': std_val}
                    
                    metric_name = metric.replace('test_', '').replace('_', ' ').title()
                    print(f"  {metric_name}: {mean_val:.3f} ± {std_val:.3f}")
            
            analysis['cv_performance'] = {
                'n_folds': len(cv_metrics),
                'metrics': cv_stats,
                'raw_data': cv_metrics.to_dict('records')
            }
            
            # Check for performance issues
            if 'test_macro_f1' in cv_stats:
                f1_mean = cv_stats['test_macro_f1']['mean']
                f1_std = cv_stats['test_macro_f1']['std']
                
                if f1_mean < 0.5:
                    analysis['issues'].append(f"Low F1 score: {f1_mean:.3f}")
                if f1_std > 0.1:
                    analysis['issues'].append(f"High F1 variance: {f1_std:.3f}")
            
        except Exception as e:
            analysis['issues'].append(f"Failed to analyze CV metrics: {e}")
            print(f"  ❌ Failed to analyze CV metrics: {e}")
    
    # Analyze meta vs base performance
    meta_vs_base_file = results_dir / 'meta_vs_base_performance.tsv'
    if meta_vs_base_file.exists():
        try:
            meta_vs_base = pd.read_csv(meta_vs_base_file, sep='\t')
            
            print(f"\n🧬 META vs BASE PERFORMANCE:")
            print("-" * 35)
            
            # Key metrics
            base_metrics = ['f1_score', 'precision', 'recall', 'accuracy']
            meta_stats = {}
            
            for metric in base_metrics:
                if metric in meta_vs_base.columns:
                    mean_val = meta_vs_base[metric].mean()
                    std_val = meta_vs_base[metric].std()
                    meta_stats[metric] = {'mean': mean_val, 'std': std_val}
                    
                    metric_name = metric.replace('_', ' ').title()
                    print(f"  Meta {metric_name}: {mean_val:.3f} ± {std_val:.3f}")
            
            analysis['meta_vs_base'] = {
                'n_samples': len(meta_vs_base),
                'metrics': meta_stats,
                'raw_data': meta_vs_base.to_dict('records')
            }
            
        except Exception as e:
            analysis['issues'].append(f"Failed to analyze meta vs base: {e}")
            print(f"  ❌ Failed to analyze meta vs base: {e}")
    
    # Load training summary if available
    summary_json = results_dir / 'training_summary.json'
    if summary_json.exists():
        try:
            with open(summary_json) as f:
                training_summary = json.load(f)
            
            print(f"\n⚙️ TRAINING CONFIGURATION:")
            print("-" * 30)
            
            # Extract key info
            dataset_info = training_summary.get('dataset_info', {})
            training_params = training_summary.get('training_parameters', {})
            system_info = training_summary.get('system_info', {})
            
            if dataset_info:
                print(f"  Dataset: {dataset_info.get('dataset_path', 'N/A')}")
                print(f"  Genes: {dataset_info.get('n_genes', 'N/A')}")
            
            if training_params:
                print(f"  Estimators: {training_params.get('n_estimators', 'N/A')}")
                print(f"  Row cap: {training_params.get('row_cap', 'N/A')}")
            
            if system_info:
                print(f"  Platform: {system_info.get('platform', 'N/A')}")
                print(f"  Memory: {system_info.get('memory_gb', 'N/A')} GB")
            
            analysis['training_summary'] = training_summary
            
        except Exception as e:
            analysis['issues'].append(f"Failed to load training summary: {e}")
            print(f"  ❌ Failed to load training summary: {e}")
    
    # Generate overall summary
    n_files_present = sum(1 for f in analysis['file_structure'].values() if f['exists'])
    n_files_total = len(key_files)
    
    analysis['summary'] = {
        'files_present': n_files_present,
        'files_total': n_files_total,
        'completeness_score': (n_files_present / n_files_total) * 100,
        'n_issues': len(analysis['issues']),
        'overall_status': 'success' if len(analysis['issues']) == 0 else 'warning'
    }
    
    print(f"\n📈 OVERALL SUMMARY:")
    print("-" * 25)
    print(f"  Files present: {n_files_present}/{n_files_total}")
    print(f"  Completeness: {analysis['summary']['completeness_score']:.1f}%")
    print(f"  Issues found: {len(analysis['issues'])}")
    
    if analysis['issues']:
        print(f"\n⚠️ ISSUES DETECTED:")
        for issue in analysis['issues']:
            print(f"  - {issue}")
    else:
        print(f"\n✅ NO ISSUES DETECTED")
    
    # Save results if requested
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        print(f"\n💾 Performance analysis saved to: {output_path}")
    
    return analysis


def generate_performance_summary(results_dir: str):
    """Generate a formatted performance summary (legacy function for compatibility)."""
    results_dir = Path(results_dir)
    
    print('=' * 60)
    print('META-MODEL TRAINING SUMMARY')
    print('=' * 60)
    
    # CV Performance
    cv_file = results_dir / 'gene_cv_metrics.csv'
    if cv_file.exists():
        cv_metrics = pd.read_csv(cv_file)
        print(f'\n🎯 CV Performance (5-fold):')
        print(f'  Mean F1 Score: {cv_metrics["test_macro_f1"].mean():.3f} ± {cv_metrics["test_macro_f1"].std():.3f}')
        if 'top_k_accuracy' in cv_metrics.columns:
            print(f'  Mean Top-k Accuracy: {cv_metrics["top_k_accuracy"].mean():.3f} ± {cv_metrics["top_k_accuracy"].std():.3f}')
    
    # Meta vs Base
    meta_vs_base_file = results_dir / 'meta_vs_base_performance.tsv'
    if meta_vs_base_file.exists():
        meta_vs_base = pd.read_csv(meta_vs_base_file, sep='\t')
        print(f'\n🧬 Meta vs Base Performance:')
        print(f'  Meta F1: {meta_vs_base["f1_score"].mean():.3f}')
        print(f'  Meta Precision: {meta_vs_base["precision"].mean():.3f}')
        print(f'  Meta Recall: {meta_vs_base["recall"].mean():.3f}')
    
    # File structure
    print(f'\n📁 Generated Files:')
    key_files = [
        'model_multiclass.pkl',
        'gene_cv_metrics.csv',
        'meta_vs_base_performance.tsv',
        'roc_curves_meta.pdf',
        'pr_curves_meta.pdf'
    ]
    
    for file in key_files:
        if (results_dir / file).exists():
            print(f'  ✅ {file}')
        else:
            print(f'  ❌ {file} (missing)')
    
    print('=' * 60)


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(
        description="Training performance analysis utility",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze training results
  python performance_analyzer.py \\
    --results-dir results/gene_cv_pc_5000_3mers_diverse_run1 \\
    --output performance_report.json

  # Quick summary (legacy format)
  python performance_analyzer.py \\
    --results-dir results/gene_cv_pc_5000_3mers_diverse_run1 \\
    --legacy-summary
        """
    )
    
    parser.add_argument("--results-dir", required=True,
                       help="Directory containing training results")
    parser.add_argument("--output", 
                       help="Output file for performance analysis (JSON format)")
    parser.add_argument("--legacy-summary", action="store_true",
                       help="Generate legacy-style summary output")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose output")
    
    args = parser.parse_args()
    
    if args.legacy_summary:
        # Generate legacy summary format
        generate_performance_summary(args.results_dir)
    else:
        # Run comprehensive analysis
        results = analyze_training_performance(args.results_dir, args.output)
        
        if 'error' in results:
            print(f"❌ Performance analysis failed: {results['error']}")
            sys.exit(1)
        
        # Exit with appropriate code based on issues
        n_issues = results['summary'].get('n_issues', 0)
        if n_issues > 0:
            print(f"\n⚠️ Analysis completed with {n_issues} issues.")
            sys.exit(1)
        else:
            print(f"\n🎉 Performance analysis completed successfully!")
            sys.exit(0)


if __name__ == "__main__":
    main()




