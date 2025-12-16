#!/usr/bin/env python3
"""
CV Run Comparison Script - Post-Training Analysis Tool

This script compares results from multiple CV runs to assess reproducibility,
identify trends, and validate model performance consistency.

Usage:
    python -m meta_spliceai.splice_engine.meta_models.analysis.compare_cv_runs \
        --run1 results/gene_cv_pc_1000_3mers_run_2_more_genes \
        --run2 results/gene_cv_pc_1000_3mers_run_3 \
        --output comparison_report.html

Features:
    - Statistical significance testing
    - Performance trend analysis
    - Reproducibility assessment
    - Detailed metric comparisons
    - HTML report generation
    - Visualization of differences
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Add project root to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent.parent))

class CVRunComparator:
    """Comprehensive CV run comparison and analysis tool."""
    
    def __init__(self, run1_path: Path, run2_path: Path, output_dir: Path):
        self.run1_path = Path(run1_path)
        self.run2_path = Path(run2_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load CV summary data
        self.run1_data = self._load_cv_summary(self.run1_path)
        self.run2_data = self._load_cv_summary(self.run2_path)
        
        # Load detailed fold metrics
        self.run1_folds = self._load_fold_metrics(self.run1_path)
        self.run2_folds = self._load_fold_metrics(self.run2_path)
        
        # Load gene-level data if available
        self.run1_genes = self._load_gene_data(self.run1_path)
        self.run2_genes = self._load_gene_data(self.run2_path)
    
    def _load_cv_summary(self, run_path: Path) -> Dict[str, Any]:
        """Load CV metrics summary from a run directory."""
        summary_file = run_path / "cv_metrics_visualization" / "cv_metrics_summary.txt"
        
        if not summary_file.exists():
            raise FileNotFoundError(f"CV summary not found: {summary_file}")
        
        data = {}
        
        with open(summary_file, 'r') as f:
            lines = f.readlines()
        
        # Parse the summary file
        for line in lines:
            line = line.strip()
            if not line or line.startswith('=') or line.startswith('-'):
                continue
            
            # Parse key metrics
            if 'Base Model F1 Score:' in line:
                parts = line.split(':')[1].strip().split('±')
                data['base_f1'] = float(parts[0].strip())
                data['base_f1_std'] = float(parts[1].strip())
            
            elif 'Meta Model F1 Score:' in line:
                parts = line.split(':')[1].strip().split('±')
                data['meta_f1'] = float(parts[0].strip())
                data['meta_f1_std'] = float(parts[1].strip())
            
            elif 'Base Model ROC AUC:' in line:
                parts = line.split(':')[1].strip().split('±')
                data['base_auc'] = float(parts[0].strip())
                data['base_auc_std'] = float(parts[1].strip())
            
            elif 'Meta Model ROC AUC:' in line:
                parts = line.split(':')[1].strip().split('±')
                data['meta_auc'] = float(parts[0].strip())
                data['meta_auc_std'] = float(parts[1].strip())
            
            elif 'Base Model Average Precision:' in line:
                parts = line.split(':')[1].strip().split('±')
                data['base_ap'] = float(parts[0].strip())
                data['base_ap_std'] = float(parts[1].strip())
            
            elif 'Meta Model Average Precision:' in line:
                parts = line.split(':')[1].strip().split('±')
                data['meta_ap'] = float(parts[0].strip())
                data['meta_ap_std'] = float(parts[1].strip())
            
            elif 'F1 Score Improvement:' in line:
                parts = line.split(':')[1].strip().split('±')
                data['f1_improvement'] = float(parts[0].strip())
                data['f1_improvement_std'] = float(parts[1].strip())
            
            elif 'F1 Score Improvement (%):' in line:
                data['f1_improvement_pct'] = float(line.split(':')[1].strip().replace('%', ''))
            
            elif 'False Positive Reduction (per fold):' in line:
                parts = line.split(':')[1].strip().split('±')
                data['fp_reduction_per_fold'] = float(parts[0].strip())
                data['fp_reduction_std'] = float(parts[1].strip())
            
            elif 'False Negative Reduction (per fold):' in line:
                parts = line.split(':')[1].strip().split('±')
                data['fn_reduction_per_fold'] = float(parts[0].strip())
                data['fn_reduction_std'] = float(parts[1].strip())
            
            elif 'Total FP Reduction (all folds):' in line:
                data['total_fp_reduction'] = int(line.split(':')[1].strip())
            
            elif 'Total FN Reduction (all folds):' in line:
                data['total_fn_reduction'] = int(line.split(':')[1].strip())
            
            elif 'Baseline False Positives:' in line:
                data['baseline_fp'] = int(line.split(':')[1].strip())
            
            elif 'Baseline False Negatives:' in line:
                data['baseline_fn'] = int(line.split(':')[1].strip())
            
            elif 'FP Reduction:' in line and '%' in line and '(' not in line:
                data['fp_reduction_pct'] = float(line.split('%')[0].split(':')[1].strip())
            
            elif 'FN Reduction:' in line and '%' in line and '(' not in line:
                data['fn_reduction_pct'] = float(line.split('%')[0].split(':')[1].strip())
            
            elif 'test samples per fold' in line:
                data['test_samples_per_fold'] = int(line.split('~')[1].split()[0])
            
            elif 'Combined Top-k Accuracy:' in line:
                parts = line.split(':')[1].strip().split('±')
                data['combined_topk'] = float(parts[0].strip())
                data['combined_topk_std'] = float(parts[1].strip())
            
            elif 'Donor Top-k Accuracy:' in line:
                parts = line.split(':')[1].strip().split('±')
                data['donor_topk'] = float(parts[0].strip())
                data['donor_topk_std'] = float(parts[1].strip())
            
            elif 'Acceptor Top-k Accuracy:' in line:
                parts = line.split(':')[1].strip().split('±')
                data['acceptor_topk'] = float(parts[0].strip())
                data['acceptor_topk_std'] = float(parts[1].strip())
        
        return data
    
    def _load_fold_metrics(self, run_path: Path) -> Optional[pd.DataFrame]:
        """Load detailed fold-level metrics."""
        metrics_file = run_path / "metrics_folds.tsv"
        
        if metrics_file.exists():
            return pd.read_csv(metrics_file, sep='\t')
        return None
    
    def _load_gene_data(self, run_path: Path) -> Optional[pd.DataFrame]:
        """Load gene-level performance data."""
        gene_file = run_path / "gene_deltas.csv"
        
        if gene_file.exists():
            return pd.read_csv(gene_file)
        return None
    
    def compare_metrics(self) -> Dict[str, Any]:
        """Perform comprehensive metric comparison."""
        
        # Define metrics to compare
        metrics = [
            ('Base F1 Score', 'base_f1', 'base_f1_std'),
            ('Meta F1 Score', 'meta_f1', 'meta_f1_std'),
            ('Base ROC AUC', 'base_auc', 'base_auc_std'),
            ('Meta ROC AUC', 'meta_auc', 'meta_auc_std'),
            ('Base Average Precision', 'base_ap', 'base_ap_std'),
            ('Meta Average Precision', 'meta_ap', 'meta_ap_std'),
            ('F1 Improvement', 'f1_improvement', 'f1_improvement_std'),
            ('F1 Improvement (%)', 'f1_improvement_pct', None),
            ('FP Reduction per Fold', 'fp_reduction_per_fold', 'fp_reduction_std'),
            ('FN Reduction per Fold', 'fn_reduction_per_fold', 'fn_reduction_std'),
            ('FP Reduction (%)', 'fp_reduction_pct', None),
            ('FN Reduction (%)', 'fn_reduction_pct', None),
            ('Combined Top-k', 'combined_topk', 'combined_topk_std'),
            ('Donor Top-k', 'donor_topk', 'donor_topk_std'),
            ('Acceptor Top-k', 'acceptor_topk', 'acceptor_topk_std')
        ]
        
        comparison_results = {
            'metrics': [],
            'significant_differences': [],
            'non_significant_differences': [],
            'similarity_score': 0,
            'overall_assessment': ''
        }
        
        similar_metrics = 0
        total_metrics = len(metrics)
        
        for metric_name, metric_key, std_key in metrics:
            run1_val = self.run1_data.get(metric_key, 0)
            run2_val = self.run2_data.get(metric_key, 0)
            
            if std_key:
                run1_std = self.run1_data.get(std_key, 0)
                run2_std = self.run2_data.get(std_key, 0)
                
                # Statistical significance test
                pooled_se = np.sqrt((run1_std**2 + run2_std**2) / 2)
                t_stat = abs(run2_val - run1_val) / pooled_se if pooled_se > 0 else 0
                p_value = 2 * (1 - stats.t.cdf(t_stat, 4)) if pooled_se > 0 else 1.0
                
                is_significant = t_stat > 2.776  # 95% confidence, 4 df
                
                metric_result = {
                    'name': metric_name,
                    'run1_value': run1_val,
                    'run2_value': run2_val,
                    'run1_std': run1_std,
                    'run2_std': run2_std,
                    'difference': run2_val - run1_val,
                    'percent_change': ((run2_val - run1_val) / run1_val * 100) if run1_val != 0 else 0,
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'is_significant': is_significant
                }
                
                if is_significant:
                    comparison_results['significant_differences'].append(metric_result)
                else:
                    comparison_results['non_significant_differences'].append(metric_result)
            else:
                metric_result = {
                    'name': metric_name,
                    'run1_value': run1_val,
                    'run2_value': run2_val,
                    'difference': run2_val - run1_val,
                    'percent_change': ((run2_val - run1_val) / run1_val * 100) if run1_val != 0 else 0,
                    'is_significant': None
                }
            
            comparison_results['metrics'].append(metric_result)
            
            # Check if within 2% for similarity score
            rel_diff = abs(run2_val - run1_val) / run1_val if run1_val != 0 else 0
            if rel_diff < 0.02:
                similar_metrics += 1
        
        comparison_results['similarity_score'] = (similar_metrics / total_metrics) * 100
        
        # Overall assessment
        if comparison_results['similarity_score'] >= 90:
            comparison_results['overall_assessment'] = 'HIGHLY SIMILAR - Excellent reproducibility'
        elif comparison_results['similarity_score'] >= 80:
            comparison_results['overall_assessment'] = 'SIMILAR - Good reproducibility'
        elif comparison_results['similarity_score'] >= 70:
            comparison_results['overall_assessment'] = 'MODERATELY SIMILAR - Some variation'
        else:
            comparison_results['overall_assessment'] = 'DIFFERENT - Poor reproducibility'
        
        return comparison_results
    
    def create_visualizations(self, comparison_results: Dict[str, Any]):
        """Create comparison visualizations."""
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('CV Run Comparison Analysis', fontsize=16, fontweight='bold')
        
        # 1. Metric comparison bar chart
        ax1 = axes[0, 0]
        metrics = [m['name'] for m in comparison_results['metrics']]
        run1_values = [m['run1_value'] for m in comparison_results['metrics']]
        run2_values = [m['run2_value'] for m in comparison_results['metrics']]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        ax1.bar(x - width/2, run1_values, width, label='Run 1', alpha=0.8)
        ax1.bar(x + width/2, run2_values, width, label='Run 2', alpha=0.8)
        
        ax1.set_xlabel('Metrics')
        ax1.set_ylabel('Values')
        ax1.set_title('Metric Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels(metrics, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Percent change analysis
        ax2 = axes[0, 1]
        percent_changes = [m['percent_change'] for m in comparison_results['metrics']]
        colors = ['red' if abs(pc) > 2 else 'green' for pc in percent_changes]
        
        bars = ax2.bar(metrics, percent_changes, color=colors, alpha=0.7)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax2.axhline(y=2, color='red', linestyle='--', alpha=0.5, label='2% threshold')
        ax2.axhline(y=-2, color='red', linestyle='--', alpha=0.5)
        
        ax2.set_xlabel('Metrics')
        ax2.set_ylabel('Percent Change (%)')
        ax2.set_title('Percent Change Analysis')
        ax2.set_xticklabels(metrics, rotation=45, ha='right')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Statistical significance plot
        ax3 = axes[1, 0]
        significant_metrics = [m['name'] for m in comparison_results['significant_differences']]
        non_significant_metrics = [m['name'] for m in comparison_results['non_significant_differences']]
        
        sig_counts = [len(significant_metrics), len(non_significant_metrics)]
        labels = ['Significant', 'Non-Significant']
        colors = ['red', 'green']
        
        ax3.pie(sig_counts, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax3.set_title('Statistical Significance Distribution')
        
        # 4. Similarity score
        ax4 = axes[1, 1]
        similarity = comparison_results['similarity_score']
        
        ax4.bar(['Similarity Score'], [similarity], color='blue', alpha=0.7)
        ax4.set_ylabel('Similarity (%)')
        ax4.set_title(f'Overall Similarity: {similarity:.1f}%')
        ax4.set_ylim(0, 100)
        ax4.grid(True, alpha=0.3)
        
        # Add text annotation
        ax4.text(0, similarity + 2, f'{similarity:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'cv_comparison_visualization.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_html_report(self, comparison_results: Dict[str, Any]):
        """Generate comprehensive HTML report."""
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>CV Run Comparison Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .metric-table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                .metric-table th, .metric-table td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                .metric-table th {{ background-color: #f2f2f2; }}
                .significant {{ background-color: #ffebee; }}
                .non-significant {{ background-color: #e8f5e8; }}
                .summary {{ background-color: #e3f2fd; padding: 15px; border-radius: 5px; margin: 20px 0; }}
                .visualization {{ text-align: center; margin: 20px 0; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>CV Run Comparison Report</h1>
                <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p><strong>Run 1:</strong> {self.run1_path.name}</p>
                <p><strong>Run 2:</strong> {self.run2_path.name}</p>
            </div>
            
            <div class="summary">
                <h2>Executive Summary</h2>
                <p><strong>Overall Assessment:</strong> {comparison_results['overall_assessment']}</p>
                <p><strong>Similarity Score:</strong> {comparison_results['similarity_score']:.1f}%</p>
                <p><strong>Significant Differences:</strong> {len(comparison_results['significant_differences'])}</p>
                <p><strong>Non-Significant Differences:</strong> {len(comparison_results['non_significant_differences'])}</p>
            </div>
            
            <h2>Detailed Metric Comparison</h2>
            <table class="metric-table">
                <tr>
                    <th>Metric</th>
                    <th>Run 1</th>
                    <th>Run 2</th>
                    <th>Difference</th>
                    <th>% Change</th>
                    <th>Significant</th>
                </tr>
        """
        
        for metric in comparison_results['metrics']:
            row_class = 'significant' if metric.get('is_significant') else 'non-significant'
            significance = 'Yes' if metric.get('is_significant') else 'No' if metric.get('is_significant') is not None else 'N/A'
            
            html_content += f"""
                <tr class="{row_class}">
                    <td>{metric['name']}</td>
                    <td>{metric['run1_value']:.4f}</td>
                    <td>{metric['run2_value']:.4f}</td>
                    <td>{metric['difference']:+.4f}</td>
                    <td>{metric['percent_change']:+.2f}%</td>
                    <td>{significance}</td>
                </tr>
            """
        
        html_content += """
            </table>
            
            <div class="visualization">
                <h2>Visualization</h2>
                <img src="cv_comparison_visualization.png" alt="CV Comparison Visualization" style="max-width: 100%;">
            </div>
            
            <h2>Key Insights</h2>
            <ul>
        """
        
        # Add key insights
        meta_f1_diff = next(m['difference'] for m in comparison_results['metrics'] if m['name'] == 'Meta F1 Score')
        html_content += f"<li>Meta F1 Score difference: {meta_f1_diff:+.4f}</li>"
        
        if comparison_results['significant_differences']:
            html_content += "<li>Statistically significant differences found in some metrics</li>"
        else:
            html_content += "<li>No statistically significant differences found</li>"
        
        html_content += f"""
            <li>Overall reproducibility: {comparison_results['similarity_score']:.1f}%</li>
            </ul>
            
            <h2>Recommendations</h2>
            <ul>
        """
        
        if comparison_results['similarity_score'] >= 90:
            html_content += "<li>Excellent reproducibility - model is highly stable</li>"
        elif comparison_results['similarity_score'] >= 80:
            html_content += "<li>Good reproducibility - minor variations are normal</li>"
        else:
            html_content += "<li>Moderate variation - consider investigating differences</li>"
        
        html_content += """
            </ul>
        </body>
        </html>
        """
        
        with open(self.output_dir / 'cv_comparison_report.html', 'w') as f:
            f.write(html_content)
    
    def run_comparison(self):
        """Run the complete comparison analysis."""
        
        print("=== CV Run Comparison Analysis ===")
        print(f"Run 1: {self.run1_path.name}")
        print(f"Run 2: {self.run2_path.name}")
        print(f"Output: {self.output_dir}")
        print()
        
        # Perform comparison
        comparison_results = self.compare_metrics()
        
        # Print summary
        print("=== Executive Summary ===")
        print(f"Overall Assessment: {comparison_results['overall_assessment']}")
        print(f"Similarity Score: {comparison_results['similarity_score']:.1f}%")
        print(f"Significant Differences: {len(comparison_results['significant_differences'])}")
        print(f"Non-Significant Differences: {len(comparison_results['non_significant_differences'])}")
        print()
        
        # Print detailed results
        print("=== Detailed Results ===")
        for metric in comparison_results['metrics']:
            status = "✓" if metric.get('is_significant') else "○" if metric.get('is_significant') is not None else "-"
            print(f"{status} {metric['name']:<25} | {metric['run1_value']:8.4f} | {metric['run2_value']:8.4f} | {metric['difference']:+8.4f} | {metric['percent_change']:+6.2f}%")
        
        print()
        
        # Create visualizations
        print("Creating visualizations...")
        self.create_visualizations(comparison_results)
        
        # Generate HTML report
        print("Generating HTML report...")
        self.generate_html_report(comparison_results)
        
        # Save JSON results
        with open(self.output_dir / 'comparison_results.json', 'w') as f:
            json.dump(comparison_results, f, indent=2, default=str)
        
        print(f"\n=== Analysis Complete ===")
        print(f"Results saved to: {self.output_dir}")
        print(f"HTML Report: {self.output_dir / 'cv_comparison_report.html'}")
        print(f"Visualization: {self.output_dir / 'cv_comparison_visualization.png'}")
        print(f"JSON Data: {self.output_dir / 'comparison_results.json'}")


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Compare CV runs and assess reproducibility",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Compare two CV runs
    python -m meta_spliceai.splice_engine.meta_models.analysis.compare_cv_runs \\
        --run1 results/gene_cv_pc_1000_3mers_run_2_more_genes \\
        --run2 results/gene_cv_pc_1000_3mers_run_3 \\
        --output cv_comparison_results
    
    # Compare with custom output directory
    python -m meta_spliceai.splice_engine.meta_models.analysis.compare_cv_runs \\
        --run1 results/run_1 \\
        --run2 results/run_2 \\
        --output analysis/comparison_report
        """
    )
    
    parser.add_argument(
        '--run1', 
        type=Path, 
        required=True,
        help='Path to first CV run results directory'
    )
    
    parser.add_argument(
        '--run2', 
        type=Path, 
        required=True,
        help='Path to second CV run results directory'
    )
    
    parser.add_argument(
        '--output', 
        type=Path, 
        default=Path('cv_comparison_results'),
        help='Output directory for comparison results (default: cv_comparison_results)'
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.run1.exists():
        print(f"Error: Run 1 directory not found: {args.run1}")
        sys.exit(1)
    
    if not args.run2.exists():
        print(f"Error: Run 2 directory not found: {args.run2}")
        sys.exit(1)
    
    # Run comparison
    comparator = CVRunComparator(args.run1, args.run2, args.output)
    comparator.run_comparison()


if __name__ == "__main__":
    main() 