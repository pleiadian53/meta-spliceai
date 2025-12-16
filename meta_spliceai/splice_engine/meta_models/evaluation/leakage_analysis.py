#!/usr/bin/env python3
"""
Comprehensive Data Leakage Analysis Module

This module provides comprehensive data leakage detection and visualization for
splice site prediction models, including correlation analysis, visualization plots,
and structured reporting.
"""
from __future__ import annotations

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
import warnings

# Statistical imports
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import LabelEncoder

# Set plotting style
plt.style.use('default')
sns.set_palette("husl")

class LeakageAnalyzer:
    """
    Comprehensive data leakage analyzer with multiple detection methods and visualizations.
    """
    
    def __init__(self, output_dir: str = None, subject: str = "leakage_analysis"):
        """
        Initialize the leakage analyzer.
        
        Parameters
        ----------
        output_dir : str, optional
            Directory to save results and plots
        subject : str, optional
            Subject name for output files
        """
        self.output_dir = output_dir or os.getcwd()
        self.subject = subject
        self.results = {}
        
        # Create output directory structure
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Create subdirectories
        self.correlations_dir = self.output_dir / "correlations"
        self.visualizations_dir = self.output_dir / "visualizations" 
        self.reports_dir = self.output_dir / "reports"
        
        for dir_path in [self.correlations_dir, self.visualizations_dir, self.reports_dir]:
            dir_path.mkdir(exist_ok=True, parents=True)
        
        # Set up plotting parameters
        self.figsize = (12, 8)
        self.plot_params = {
            'dpi': 300,
            'bbox_inches': 'tight',
            'facecolor': 'white',
            'edgecolor': 'none'
        }
    
    def run_comprehensive_analysis(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        threshold: float = 0.95,
        methods: List[str] = None,
        top_n: int = 50,
        verbose: int = 1
    ) -> Dict[str, Any]:
        """
        Run comprehensive leakage analysis using multiple correlation methods.
        
        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix
        y : pd.Series
            Target variable
        threshold : float, optional
            Correlation threshold for detecting leaky features
        methods : List[str], optional
            Correlation methods to use. If None, uses ['pearson', 'spearman']
        top_n : int, optional
            Number of top correlated features to analyze in detail
        verbose : int, optional
            Verbosity level
            
        Returns
        -------
        Dict[str, Any]
            Dictionary containing all analysis results
        """
        if methods is None:
            methods = ['pearson', 'spearman']
        
        if verbose:
            print(f"[Leakage Analysis] Starting comprehensive analysis...")
            print(f"  Data shape: {X.shape}")
            print(f"  Target distribution: {y.value_counts().to_dict()}")
            print(f"  Threshold: {threshold}")
            print(f"  Methods: {methods}")
        
        results = {}
        
        # Ensure target is binary for correlation analysis
        y_binary = self._prepare_target(y, verbose)
        
        # 1. Correlation Analysis
        for method in methods:
            if verbose:
                print(f"\n[Leakage Analysis] Running {method} correlation analysis...")
            
            correlation_results = self._analyze_correlations(
                X, y_binary, method=method, threshold=threshold, top_n=top_n, verbose=verbose
            )
            results[method] = correlation_results
            
            # Create correlation plots
            self._plot_correlation_results(
                correlation_results, method=method, threshold=threshold, top_n=top_n
            )
        
        # 2. Create comparison plots if multiple methods
        if len(methods) > 1:
            if verbose:
                print(f"\n[Leakage Analysis] Creating method comparison plots...")
            self._plot_method_comparison(results, threshold=threshold, top_n=top_n)
        
        # 3. Generate comprehensive report
        if verbose:
            print(f"\n[Leakage Analysis] Generating comprehensive report...")
        report = self._generate_comprehensive_report(results, threshold=threshold, X=X, y=y_binary)
        
        # 4. Save all results
        self._save_results(results, report)
        
        if verbose:
            print(f"\n[Leakage Analysis] Analysis complete!")
            print(f"  Output directory: {self.output_dir}")
            total_leaky = sum(len(r['leaky_features']) for r in results.values())
            print(f"  Total potentially leaky features detected: {total_leaky}")
        
        return {
            'correlation_results': results,
            'comprehensive_report': report,
            'output_directory': str(self.output_dir)
        }
    
    def _prepare_target(self, y: pd.Series, verbose: int = 1) -> pd.Series:
        """Convert target to binary format for correlation analysis."""
        if verbose > 1:
            print(f"[Leakage Analysis] Preparing target variable...")
            print(f"  Original target values: {y.value_counts().to_dict()}")
        
        # Handle different target formats
        if pd.api.types.is_numeric_dtype(y):
            # Assume 0 = negative class, any other value = positive class
            y_binary = (y != 0).astype(int)
        else:
            # String labels: convert to binary
            if set(y.unique()) == {'neither', 'donor', 'acceptor'}:
                # Three-class format: neither vs splice (donor/acceptor)
                y_binary = (y != 'neither').astype(int)
            else:
                # General case: use label encoder then binarize
                le = LabelEncoder()
                y_encoded = le.fit_transform(y)
                y_binary = (y_encoded != 0).astype(int)
        
        if verbose > 1:
            print(f"  Binary target distribution: {pd.Series(y_binary).value_counts().to_dict()}")
        
        return pd.Series(y_binary, index=y.index, name='target_binary')
    
    def _analyze_correlations(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        method: str = 'pearson',
        threshold: float = 0.95,
        top_n: int = 50,
        verbose: int = 1
    ) -> Dict[str, Any]:
        """Analyze feature-target correlations using specified method."""
        
        results = []
        skipped_features = []
        
        for feature in X.columns:
            # Check if feature is numeric and has variance
            feature_data = X[feature]
            
            if not pd.api.types.is_numeric_dtype(feature_data):
                skipped_features.append((feature, "non_numeric"))
                continue
            
            if feature_data.nunique() <= 1:
                skipped_features.append((feature, "constant"))
                continue
            
            # Calculate correlation
            try:
                if method == 'pearson':
                    corr_coef, p_value = pearsonr(feature_data.fillna(0), y)
                elif method == 'spearman':
                    corr_coef, p_value = spearmanr(feature_data.fillna(0), y, nan_policy='omit')
                else:
                    raise ValueError(f"Unknown correlation method: {method}")
                
                results.append({
                    'feature': feature,
                    'correlation': corr_coef,
                    'abs_correlation': abs(corr_coef),
                    'p_value': p_value,
                    'is_leaky': abs(corr_coef) >= threshold
                })
                
            except Exception as e:
                skipped_features.append((feature, f"error: {str(e)}"))
                continue
        
        # Convert to DataFrame and sort
        if results:
            results_df = pd.DataFrame(results)
            results_df = results_df.sort_values('abs_correlation', ascending=False).reset_index(drop=True)
        else:
            results_df = pd.DataFrame(columns=['feature', 'correlation', 'abs_correlation', 'p_value', 'is_leaky'])
        
        # Identify leaky features
        leaky_features = results_df[results_df['is_leaky']].copy()
        
        # Save correlation results
        output_file = self.correlations_dir / f"feature_correlations_{method}.csv"
        results_df.to_csv(output_file, index=False)
        
        if verbose:
            print(f"  Analyzed {len(results_df)} features")
            print(f"  Skipped {len(skipped_features)} features")
            print(f"  Found {len(leaky_features)} potentially leaky features (threshold: {threshold})")
            print(f"  Saved results to: {output_file}")
        
        return {
            'full_results': results_df,
            'top_n': results_df.head(top_n),
            'leaky_features': leaky_features,
            'skipped_features': skipped_features,
            'threshold': threshold,
            'method': method,
            'summary_stats': {
                'total_features': len(results_df),
                'leaky_count': len(leaky_features),
                'max_correlation': results_df['abs_correlation'].max() if len(results_df) > 0 else 0,
                'mean_correlation': results_df['abs_correlation'].mean() if len(results_df) > 0 else 0,
                'median_correlation': results_df['abs_correlation'].median() if len(results_df) > 0 else 0
            }
        }
    
    def _plot_correlation_results(
        self,
        results: Dict[str, Any],
        method: str,
        threshold: float,
        top_n: int = 20
    ):
        """Create visualization plots for correlation results."""
        
        results_df = results['full_results']
        leaky_features = results['leaky_features']
        
        if len(results_df) == 0:
            print(f"[Leakage Analysis] No correlation results to plot for {method}")
            return
        
        # 1. Top correlations bar plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Left plot: Top correlations
        top_features = results_df.head(top_n)
        colors = ['red' if is_leaky else 'blue' for is_leaky in top_features['is_leaky']]
        
        bars = ax1.barh(range(len(top_features)), top_features['abs_correlation'], color=colors, alpha=0.7)
        ax1.set_yticks(range(len(top_features)))
        ax1.set_yticklabels(top_features['feature'], fontsize=8)
        ax1.set_xlabel('Absolute Correlation', fontweight='bold')
        ax1.set_title(f'Top {top_n} Feature Correlations ({method.title()})', fontweight='bold')
        ax1.axvline(x=threshold, color='red', linestyle='--', alpha=0.8, label=f'Threshold ({threshold})')
        ax1.legend()
        ax1.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax1.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{width:.3f}', ha='left', va='center', fontsize=8)
        
        # Right plot: Correlation distribution
        ax2.hist(results_df['abs_correlation'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.axvline(x=threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold ({threshold})')
        ax2.set_xlabel('Absolute Correlation', fontweight='bold')
        ax2.set_ylabel('Frequency', fontweight='bold')
        ax2.set_title(f'Distribution of Feature Correlations ({method.title()})', fontweight='bold')
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        plot_file = self.visualizations_dir / f"correlation_analysis_{method}.png"
        plt.savefig(plot_file, **self.plot_params)
        plt.close()
        
        # 2. Scatter plot of correlations vs p-values (if p-values available)
        if 'p_value' in results_df.columns and not results_df['p_value'].isna().all():
            plt.figure(figsize=self.figsize)
            
            # Create scatter plot
            scatter = plt.scatter(
                results_df['abs_correlation'], 
                -np.log10(results_df['p_value'] + 1e-300),  # Add small constant to avoid log(0)
                c=results_df['is_leaky'].map({True: 'red', False: 'blue'}),
                alpha=0.6,
                s=30
            )
            
            plt.axvline(x=threshold, color='red', linestyle='--', alpha=0.8, label=f'Correlation Threshold ({threshold})')
            plt.axhline(y=-np.log10(0.05), color='orange', linestyle='--', alpha=0.8, label='p < 0.05')
            
            plt.xlabel('Absolute Correlation', fontweight='bold')
            plt.ylabel('-log10(p-value)', fontweight='bold')
            plt.title(f'Feature Correlations vs Statistical Significance ({method.title()})', fontweight='bold')
            plt.legend()
            plt.grid(alpha=0.3)
            
            # Annotate top leaky features
            if len(leaky_features) > 0:
                for _, row in leaky_features.head(5).iterrows():
                    plt.annotate(
                        row['feature'][:15] + ('...' if len(row['feature']) > 15 else ''),
                        (row['abs_correlation'], -np.log10(row['p_value'] + 1e-300)),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, ha='left'
                    )
            
            plt.tight_layout()
            plot_file = self.visualizations_dir / f"correlation_significance_{method}.png"
            plt.savefig(plot_file, **self.plot_params)
            plt.close()
        
        # 3. Leaky features detail plot (if any found)
        if len(leaky_features) > 0:
            plt.figure(figsize=(12, max(6, len(leaky_features) * 0.3)))
            
            colors = plt.cm.Reds(np.linspace(0.4, 1, len(leaky_features)))
            bars = plt.barh(range(len(leaky_features)), leaky_features['abs_correlation'], color=colors)
            
            plt.yticks(range(len(leaky_features)), leaky_features['feature'])
            plt.xlabel('Absolute Correlation', fontweight='bold')
            plt.title(f'Potentially Leaky Features ({method.title()})\n'
                     f'{len(leaky_features)} features with |correlation| ≥ {threshold}', fontweight='bold')
            plt.axvline(x=threshold, color='red', linestyle='--', alpha=0.8, label=f'Threshold ({threshold})')
            
            # Add value labels
            for i, bar in enumerate(bars):
                width = bar.get_width()
                plt.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                        f'{width:.3f}', ha='left', va='center', fontsize=9)
            
            plt.legend()
            plt.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            
            plot_file = self.visualizations_dir / f"leaky_features_{method}.png"
            plt.savefig(plot_file, **self.plot_params)
            plt.close()
    
    def _plot_method_comparison(self, results: Dict[str, Any], threshold: float, top_n: int = 20):
        """Create comparison plots between different correlation methods."""
        
        methods = list(results.keys())
        if len(methods) < 2:
            return
        
        # 1. Method overlap heatmap
        method_features = {}
        for method, method_results in results.items():
            leaky_features = set(method_results['leaky_features']['feature'])
            method_features[method.title()] = leaky_features
        
        if any(len(features) > 0 for features in method_features.values()):
            # Create overlap matrix
            method_names = list(method_features.keys())
            overlap_matrix = np.zeros((len(method_names), len(method_names)))
            
            for i, method1 in enumerate(method_names):
                for j, method2 in enumerate(method_names):
                    if i == j:
                        overlap_matrix[i, j] = len(method_features[method1])
                    else:
                        overlap = len(method_features[method1] & method_features[method2])
                        overlap_matrix[i, j] = overlap
            
            plt.figure(figsize=(8, 6))
            sns.heatmap(
                overlap_matrix,
                annot=True,
                fmt='g',
                cmap='Reds',
                xticklabels=method_names,
                yticklabels=method_names,
                square=True,
                cbar_kws={'label': 'Number of Overlapping Leaky Features'}
            )
            
            plt.title(f'Leaky Feature Overlap Between Methods\n(Threshold: {threshold})', 
                     fontweight='bold', pad=20)
            plt.tight_layout()
            
            plot_file = self.visualizations_dir / "method_overlap.png"
            plt.savefig(plot_file, **self.plot_params)
            plt.close()
        
        # 2. Correlation comparison scatter plot (for Pearson vs Spearman)
        if len(methods) == 2 and 'pearson' in results and 'spearman' in results:
            pearson_df = results['pearson']['full_results']
            spearman_df = results['spearman']['full_results']
            
            # Merge on feature name
            merged_df = pd.merge(
                pearson_df[['feature', 'abs_correlation', 'is_leaky']],
                spearman_df[['feature', 'abs_correlation', 'is_leaky']],
                on='feature',
                suffixes=('_pearson', '_spearman')
            )
            
            if len(merged_df) > 0:
                plt.figure(figsize=self.figsize)
                
                # Create scatter plot
                colors = merged_df.apply(
                    lambda row: 'red' if row['is_leaky_pearson'] or row['is_leaky_spearman'] else 'blue',
                    axis=1
                )
                
                plt.scatter(
                    merged_df['abs_correlation_pearson'],
                    merged_df['abs_correlation_spearman'],
                    c=colors,
                    alpha=0.6,
                    s=30
                )
                
                # Add diagonal line
                max_corr = max(merged_df['abs_correlation_pearson'].max(), 
                              merged_df['abs_correlation_spearman'].max())
                plt.plot([0, max_corr], [0, max_corr], 'k--', alpha=0.5, label='y = x')
                
                # Add threshold lines
                plt.axvline(x=threshold, color='red', linestyle='--', alpha=0.8)
                plt.axhline(y=threshold, color='red', linestyle='--', alpha=0.8, label=f'Threshold ({threshold})')
                
                plt.xlabel('Pearson Absolute Correlation', fontweight='bold')
                plt.ylabel('Spearman Absolute Correlation', fontweight='bold')
                plt.title('Pearson vs Spearman Correlation Comparison', fontweight='bold')
                plt.legend()
                plt.grid(alpha=0.3)
                
                plt.tight_layout()
                plot_file = self.visualizations_dir / "pearson_vs_spearman.png"
                plt.savefig(plot_file, **self.plot_params)
                plt.close()
    
    def _generate_comprehensive_report(
        self, 
        results: Dict[str, Any], 
        threshold: float,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Dict[str, Any]:
        """Generate comprehensive leakage analysis report."""
        
        report = {
            'analysis_summary': {
                'dataset_info': {
                    'n_samples': len(X),
                    'n_features': len(X.columns),
                    'target_distribution': y.value_counts().to_dict()
                },
                'analysis_parameters': {
                    'threshold': threshold,
                    'methods_used': list(results.keys())
                }
            },
            'method_results': {},
            'overall_summary': {},
            'recommendations': []
        }
        
        total_leaky = 0
        all_leaky_features = set()
        
        # Process each method
        for method, method_results in results.items():
            method_summary = method_results['summary_stats'].copy()
            method_summary['leaky_features_list'] = method_results['leaky_features']['feature'].tolist()
            
            report['method_results'][method] = method_summary
            
            total_leaky += method_summary['leaky_count']
            all_leaky_features.update(method_summary['leaky_features_list'])
        
        # Overall summary
        unique_leaky_features = len(all_leaky_features)
        report['overall_summary'] = {
            'unique_leaky_features': unique_leaky_features,
            'total_leaky_detections': total_leaky,
            'leakage_rate': unique_leaky_features / len(X.columns) if len(X.columns) > 0 else 0,
            'consensus_leaky_features': []
        }
        
        # Find consensus leaky features (detected by multiple methods)
        if len(results) > 1:
            feature_counts = {}
            for method_results in results.values():
                for feature in method_results['leaky_features']['feature']:
                    feature_counts[feature] = feature_counts.get(feature, 0) + 1
            
            consensus_features = [
                feature for feature, count in feature_counts.items() 
                if count >= len(results) / 2  # At least half of methods agree
            ]
            report['overall_summary']['consensus_leaky_features'] = consensus_features
        
        # Generate recommendations
        recommendations = []
        
        if unique_leaky_features == 0:
            recommendations.append("✅ No potentially leaky features detected. Data appears clean.")
        elif unique_leaky_features < 5:
            recommendations.append(f"⚠️ Found {unique_leaky_features} potentially leaky features. Review these carefully.")
        else:
            recommendations.append(f"🚨 Found {unique_leaky_features} potentially leaky features. This suggests significant data leakage concerns.")
        
        if report['overall_summary']['leakage_rate'] > 0.1:
            recommendations.append("🚨 High leakage rate (>10% of features). Consider data preprocessing review.")
        
        if len(report['overall_summary']['consensus_leaky_features']) > 0:
            recommendations.append(f"🎯 {len(report['overall_summary']['consensus_leaky_features'])} features detected as leaky by multiple methods - high priority for investigation.")
        
        # Method-specific recommendations
        if 'pearson' in results and 'spearman' in results:
            pearson_leaky = set(results['pearson']['leaky_features']['feature'])
            spearman_leaky = set(results['spearman']['leaky_features']['feature'])
            spearman_only = spearman_leaky - pearson_leaky
            
            if len(spearman_only) > 0:
                recommendations.append(f"📊 {len(spearman_only)} features detected only by Spearman correlation - may indicate non-linear leakage patterns.")
        
        report['recommendations'] = recommendations
        
        return report
    
    def _save_results(self, results: Dict[str, Any], report: Dict[str, Any]):
        """Save all analysis results to files."""
        
        # 1. Save individual method results
        for method, method_results in results.items():
            # Full results
            full_file = self.correlations_dir / f"full_results_{method}.csv"
            method_results['full_results'].to_csv(full_file, index=False)
            
            # Leaky features only
            if len(method_results['leaky_features']) > 0:
                leaky_file = self.correlations_dir / f"leaky_features_{method}.csv"
                method_results['leaky_features'].to_csv(leaky_file, index=False)
        
        # 2. Save comprehensive report
        report_file = self.reports_dir / "comprehensive_leakage_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # 3. Save human-readable summary
        summary_file = self.reports_dir / "leakage_analysis_summary.txt"
        self._write_summary_report(report, summary_file)
        
        print(f"[Leakage Analysis] Results saved to:")
        print(f"  Correlations: {self.correlations_dir}")
        print(f"  Visualizations: {self.visualizations_dir}")
        print(f"  Reports: {self.reports_dir}")
    
    def _write_summary_report(self, report: Dict[str, Any], output_file: Path):
        """Write human-readable summary report."""
        
        with open(output_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("COMPREHENSIVE DATA LEAKAGE ANALYSIS REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            # Dataset info
            f.write("DATASET INFORMATION\n")
            f.write("-" * 30 + "\n")
            dataset_info = report['analysis_summary']['dataset_info']
            f.write(f"Number of samples: {dataset_info['n_samples']:,}\n")
            f.write(f"Number of features: {dataset_info['n_features']:,}\n")
            f.write(f"Target distribution: {dataset_info['target_distribution']}\n\n")
            
            # Analysis parameters
            f.write("ANALYSIS PARAMETERS\n")
            f.write("-" * 30 + "\n")
            params = report['analysis_summary']['analysis_parameters']
            f.write(f"Correlation threshold: {params['threshold']}\n")
            f.write(f"Methods used: {', '.join(params['methods_used'])}\n\n")
            
            # Overall summary
            f.write("OVERALL RESULTS\n")
            f.write("-" * 30 + "\n")
            overall = report['overall_summary']
            f.write(f"Unique potentially leaky features: {overall['unique_leaky_features']}\n")
            f.write(f"Total leaky detections: {overall['total_leaky_detections']}\n")
            f.write(f"Leakage rate: {overall['leakage_rate']:.2%}\n")
            
            if overall['consensus_leaky_features']:
                f.write(f"Consensus leaky features: {len(overall['consensus_leaky_features'])}\n")
                for feature in overall['consensus_leaky_features']:
                    f.write(f"  - {feature}\n")
            f.write("\n")
            
            # Method-specific results
            f.write("METHOD-SPECIFIC RESULTS\n")
            f.write("-" * 30 + "\n")
            for method, method_results in report['method_results'].items():
                f.write(f"\n{method.upper()} CORRELATION:\n")
                f.write(f"  Features analyzed: {method_results['total_features']}\n")
                f.write(f"  Potentially leaky features: {method_results['leaky_count']}\n")
                f.write(f"  Maximum correlation: {method_results['max_correlation']:.4f}\n")
                f.write(f"  Mean correlation: {method_results['mean_correlation']:.4f}\n")
                f.write(f"  Median correlation: {method_results['median_correlation']:.4f}\n")
                
                if method_results['leaky_features_list']:
                    f.write(f"  Leaky features:\n")
                    for feature in method_results['leaky_features_list'][:10]:  # Top 10
                        f.write(f"    - {feature}\n")
                    if len(method_results['leaky_features_list']) > 10:
                        f.write(f"    ... and {len(method_results['leaky_features_list']) - 10} more\n")
            
            f.write("\n")
            
            # Recommendations
            f.write("RECOMMENDATIONS\n")
            f.write("-" * 30 + "\n")
            for i, recommendation in enumerate(report['recommendations'], 1):
                f.write(f"{i}. {recommendation}\n")
            
            f.write("\n")
            f.write("=" * 80 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 80 + "\n")


def run_comprehensive_leakage_analysis(
    dataset_path: str | Path,
    run_dir: str | Path,
    threshold: float = 0.95,
    methods: List[str] = None,
    sample: int = None,
    top_n: int = 50,
    verbose: int = 1
) -> Dict[str, Any]:
    """
    Run comprehensive leakage analysis on a trained model's features.
    
    Parameters
    ----------
    dataset_path : str | Path
        Path to the dataset
    run_dir : str | Path
        Directory containing model and feature manifest
    threshold : float, optional
        Correlation threshold for leakage detection
    methods : List[str], optional
        Correlation methods to use
    sample : int, optional
        Number of samples to use for analysis
    top_n : int, optional
        Number of top features to analyze in detail
    verbose : int, optional
        Verbosity level
        
    Returns
    -------
    Dict[str, Any]
        Analysis results and output directory
    """
    # FIXED: Import from the correct training module
    from meta_spliceai.splice_engine.meta_models.training import datasets
    from meta_spliceai.splice_engine.meta_models.builder import preprocessing
    
    if methods is None:
        methods = ['pearson', 'spearman']
    
    run_dir = Path(run_dir)
    
    # Create leakage analysis directory
    leakage_dir = run_dir / "leakage_analysis"
    leakage_dir.mkdir(exist_ok=True, parents=True)
    
    if verbose:
        print(f"[Comprehensive Leakage Analysis] Starting analysis...")
        print(f"  Dataset: {dataset_path}")
        print(f"  Output directory: {leakage_dir}")
        print(f"  Threshold: {threshold}")
        print(f"  Methods: {methods}")
    
    # Load dataset
    if sample is not None:
        if verbose:
            print(f"[Comprehensive Leakage Analysis] Loading dataset sample ({sample} rows)...")
        # Set environment variable for row limit
        import os
        original_max_rows = os.environ.get("SS_MAX_ROWS")
        os.environ["SS_MAX_ROWS"] = str(sample)
        df = datasets.load_dataset(dataset_path)
        if original_max_rows is not None:
            os.environ["SS_MAX_ROWS"] = original_max_rows
        else:
            os.environ.pop("SS_MAX_ROWS", None)
    else:
        if verbose:
            print(f"[Comprehensive Leakage Analysis] Loading full dataset...")
        df = datasets.load_dataset(dataset_path)
    
    # Prepare training data
    X_df, y_series = preprocessing.prepare_training_data(
        df, 
        label_col="splice_type", 
        return_type="pandas", 
        verbose=verbose,
        encode_chrom=True  # Include encoded chromosome as a feature
    )
    
    if verbose:
        print(f"[Comprehensive Leakage Analysis] Prepared data: {X_df.shape}")
    
    # Initialize analyzer
    analyzer = LeakageAnalyzer(
        output_dir=leakage_dir,
        subject="comprehensive_leakage"
    )
    
    # Run comprehensive analysis
    results = analyzer.run_comprehensive_analysis(
        X=X_df,
        y=y_series,
        threshold=threshold,
        methods=methods,
        top_n=top_n,
        verbose=verbose
    )
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run comprehensive data leakage analysis")
    parser.add_argument("dataset", help="Path to dataset")
    parser.add_argument("run_dir", help="Path to model run directory")
    parser.add_argument("--threshold", type=float, default=0.95, help="Correlation threshold")
    parser.add_argument("--methods", nargs="+", default=["pearson", "spearman"], help="Correlation methods")
    parser.add_argument("--sample", type=int, help="Sample size")
    parser.add_argument("--top-n", type=int, default=50, help="Number of top features")
    parser.add_argument("--verbose", type=int, default=1, help="Verbosity level")
    
    args = parser.parse_args()
    
    results = run_comprehensive_leakage_analysis(
        dataset_path=args.dataset,
        run_dir=args.run_dir,
        threshold=args.threshold,
        methods=args.methods,
        sample=args.sample,
        top_n=args.top_n,
        verbose=args.verbose
    )
    
    print("\n" + "="*60)
    print("COMPREHENSIVE LEAKAGE ANALYSIS COMPLETE")
    print("="*60)
    print(f"Output directory: {results['output_directory']}")
    print("Check the reports/ subdirectory for detailed analysis results.") 