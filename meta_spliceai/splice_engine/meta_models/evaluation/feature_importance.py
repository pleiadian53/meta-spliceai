"""
Feature Importance Analysis Module

This module provides comprehensive feature importance analysis using multiple methods:
1. XGBoost internal feature importance
2. Hypothesis testing (statistical significance)
3. Effect size measurement
4. Mutual information

Each method provides both quantitative results and publication-ready visualizations.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
from collections import defaultdict
import re

# Statistical imports
from scipy.stats import mannwhitneyu, ttest_ind, chi2_contingency, fisher_exact, shapiro
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import OrdinalEncoder
from statsmodels.stats.multitest import multipletests

# Plotting configuration
plt.style.use('default')
sns.set_palette("husl")

class FeatureImportanceAnalyzer:
    """
    Comprehensive feature importance analyzer with multiple methods and publication-ready plots.
    """
    
    def __init__(self, output_dir: str = None, subject: str = "feature_analysis"):
        """
        Initialize the feature importance analyzer.
        
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
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
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
        model,
        X: pd.DataFrame,
        y: pd.Series,
        top_k: int = 20,
        methods: List[str] = None,
        feature_categories: Dict = None,
        verbose: int = 1
    ) -> Dict:
        """
        Run comprehensive feature importance analysis using multiple methods.
        
        Parameters
        ----------
        model : sklearn-compatible model
            Trained model for analysis
        X : pd.DataFrame
            Feature matrix
        y : pd.Series
            Target variable
        top_k : int, optional
            Number of top features to analyze
        methods : List[str], optional
            Methods to use. If None, uses all available methods
        feature_categories : Dict, optional
            Dictionary categorizing features by type
        verbose : int, optional
            Verbosity level
            
        Returns
        -------
        Dict
            Dictionary containing all analysis results
        """
        if methods is None:
            methods = ['xgboost', 'hypothesis_testing', 'effect_sizes', 'mutual_info']
        
        results = {}
        
        # 1. XGBoost Feature Importance
        if 'xgboost' in methods:
            if verbose:
                print("Running XGBoost feature importance analysis...")
            results['xgboost'] = self.analyze_xgboost_importance(
                model, X, top_k=top_k, verbose=verbose
            )
        
        # 2. Hypothesis Testing
        if 'hypothesis_testing' in methods:
            if verbose:
                print("Running hypothesis testing analysis...")
            results['hypothesis_testing'] = self.analyze_hypothesis_testing(
                X, y, top_k=top_k, feature_categories=feature_categories, verbose=verbose
            )
        
        # 3. Effect Sizes
        if 'effect_sizes' in methods:
            if verbose:
                print("Running effect size analysis...")
            results['effect_sizes'] = self.analyze_effect_sizes(
                X, y, top_k=top_k, feature_categories=feature_categories, verbose=verbose
            )
        
        # 4. Mutual Information
        if 'mutual_info' in methods:
            if verbose:
                print("Running mutual information analysis...")
            results['mutual_info'] = self.analyze_mutual_information(
                X, y, top_k=top_k, feature_categories=feature_categories, verbose=verbose
            )
        
        # Create comparison plots
        self.create_comparison_plots(results, top_k=top_k)
        
        # Store results
        self.results = results
        
        return results
    
    def analyze_xgboost_importance(
        self,
        model,
        X: pd.DataFrame,
        top_k: int = 20,
        importance_types: List[str] = None,
        verbose: int = 1
    ) -> Dict:
        """
        Analyze XGBoost feature importance using multiple importance types.
        """
        if importance_types is None:
            importance_types = ['weight', 'gain', 'cover', 'total_gain', 'total_cover']
        
        results = {}
        
        for importance_type in importance_types:
            if verbose:
                print(f"  Analyzing {importance_type} importance...")
            
            # Get feature importance
            feature_importance_dict = model.get_booster().get_score(importance_type=importance_type)
            
            # Convert to DataFrame
            importance_df = pd.DataFrame.from_dict(
                feature_importance_dict, orient='index', columns=['importance_score']
            ).reset_index().rename(columns={'index': 'feature'})
            
            # Add missing features with zero importance
            missing_features = set(X.columns) - set(importance_df['feature'])
            if missing_features:
                missing_df = pd.DataFrame({
                    'feature': list(missing_features), 
                    'importance_score': 0.0
                })
                importance_df = pd.concat([importance_df, missing_df], ignore_index=True)
            
            # Sort by importance
            importance_df = importance_df.sort_values(
                by='importance_score', ascending=False
            ).reset_index(drop=True)
            
            results[importance_type] = {
                'full_df': importance_df,
                'top_k': importance_df.head(top_k)
            }
            
            # Create publication-ready plot
            self._plot_xgboost_importance(
                importance_df.head(top_k), 
                importance_type, 
                top_k
            )
        
        return results
    
    def analyze_hypothesis_testing(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        top_k: int = 20,
        feature_categories: Dict = None,
        alpha: float = 0.05,
        verbose: int = 1
    ) -> Dict:
        """
        Analyze feature importance via hypothesis testing with comprehensive significance summary.
        
        Uses appropriate statistical tests (t-test, Mann-Whitney U, chi-square, Fisher's exact)
        and applies Benjamini-Hochberg FDR correction for multiple testing.
        """
        # Classify features if not provided
        if feature_categories is None:
            feature_categories = self._classify_features(X, y)
        
        if verbose:
            print(f"[Hypothesis Testing] Analyzing {len(X.columns)} features...")
        
        results = []
        
        for idx, feature in enumerate(X.columns):
            # FIXED: Reduce verbosity - only show progress for first few features
            if verbose and idx < 5:
                print(f"  Testing feature: {feature}")
            
            pos_class = X.loc[y == 1, feature]
            neg_class = X.loc[y == 0, feature]
            
            test_result = self._perform_statistical_test(
                pos_class, neg_class, feature, feature_categories, alpha
            )
            
            # FIXED: Reduce verbosity - only show details for first few features
            if verbose and idx < 5:
                print(f"    Test result: p-value={test_result['p_value']:.6f}, statistic={test_result['test_statistic']:.4f}")
            
            results.append(test_result)
        
        # Convert to DataFrame and sort
        results_df = pd.DataFrame(results)
        
        # Apply Benjamini-Hochberg FDR correction
        if len(results_df) > 0:
            # FIXED: Handle NaN p-values before FDR correction
            invalid_mask = (results_df['p_value'].isna() | 
                           (results_df['p_value'] < 0) | 
                           (results_df['p_value'] > 1))
            
            if invalid_mask.any():
                print(f"[FDR Correction] Found {invalid_mask.sum()} invalid p-values, setting to 1.0")
                results_df.loc[invalid_mask, 'p_value'] = 1.0
            
            rejected, p_values_corrected, alpha_sidak, alpha_bonf = multipletests(
                results_df['p_value'], 
                alpha=alpha, 
                method='fdr_bh'
            )
            
            # FIXED: Use robust approach to handle NaN values and prevent log(0)
            p_values_corrected = np.nan_to_num(p_values_corrected, nan=1.0)
            
            results_df['p_value_fdr'] = p_values_corrected
            results_df['neg_log10_p_fdr'] = -np.log10(np.maximum(p_values_corrected, 1e-300))
            results_df['significant_fdr'] = rejected
            results_df['significant_uncorrected'] = results_df['p_value'] < alpha
            
            # Sort by FDR-corrected p-value
            results_df = results_df.sort_values('p_value_fdr').reset_index(drop=True)
            
            # Generate comprehensive significance summary
            significance_summary = self._generate_significance_summary(
                results_df, alpha, verbose
            )
            
            if verbose:
                print(f"\n" + "="*60)
                print("MULTIPLE TESTING CORRECTION SUMMARY")
                print("="*60)
                print(f"Total features tested: {len(results_df)}")
                print(f"Significance level (α): {alpha}")
                print(f"Multiple testing correction: Benjamini-Hochberg (FDR)")
                print(f"Expected false discovery rate: ≤ {alpha*100:.1f}%")
                print()
                print(f"Significant features (uncorrected p < {alpha}): {results_df['significant_uncorrected'].sum()}")
                print(f"Significant features (FDR-corrected p < {alpha}): {results_df['significant_fdr'].sum()}")
                print(f"Features rejected by multiple testing correction: {results_df['significant_uncorrected'].sum() - results_df['significant_fdr'].sum()}")
                
                if results_df['significant_fdr'].sum() > 0:
                    print(f"\n" + "="*60)
                    print("SIGNIFICANT FEATURES AFTER FDR CORRECTION")
                    print("="*60)
                    
                    significant_features = results_df[results_df['significant_fdr']].copy()
                    # FIXED: Limit output to first 10 features to avoid overwhelming output
                    display_features = significant_features.head(10)
                    for i, (_, row) in enumerate(display_features.iterrows()):
                        print(f"{i+1:2d}. {row['feature']}")
                        print(f"     Test: {row['test_type']}")
                        print(f"     Raw p-value: {row['p_value']:.6f}")
                        print(f"     FDR-adjusted p-value: {row['p_value_fdr']:.6f}")
                        print(f"     Test statistic: {row['test_statistic']:.4f}")
                        print(f"     -log10(p_FDR): {row['neg_log10_p_fdr']:.2f}")
                        print()
                    
                    # If there are more than 10 significant features, show count
                    if len(significant_features) > 10:
                        print(f"... and {len(significant_features) - 10} more significant features")
                        print()
                else:
                    print(f"\n⚠️  No features remain significant after FDR correction.")
                    print(f"   This suggests that the initially significant results may be false positives.")
                    print(f"   Consider: 1) Larger sample size, 2) More stringent α level, 3) Effect size analysis")
                
                print(f"="*60)
        
        # Create publication-ready plot
        if len(results_df) > 0:
            self._plot_hypothesis_testing_results(results_df, top_k)
        else:
            print("[Hypothesis Testing] Warning: No results to plot")
        
        return {
            'full_df': results_df,
            'top_k': results_df.head(top_k),
            'num_significant': results_df['significant_fdr'].sum() if len(results_df) > 0 else 0,
            'num_significant_uncorrected': results_df['significant_uncorrected'].sum() if len(results_df) > 0 else 0,
            'significant_features': results_df[results_df['significant_fdr']].copy() if len(results_df) > 0 else pd.DataFrame(),
            'significance_summary': significance_summary if len(results_df) > 0 else {}
        }
    
    def analyze_effect_sizes(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        top_k: int = 20,
        feature_categories: Dict = None,
        verbose: int = 1
    ) -> Dict:
        """
        Analyze feature importance via effect size measurement.
        """
        if feature_categories is None:
            feature_categories = self._classify_features(X, y)
        
        if verbose:
            print(f"[Effect Sizes] Analyzing {len(X.columns)} features...")
            print(f"[Effect Sizes] Target distribution: {y.value_counts().to_dict()}")
        
        results = []
        
        for idx, feature in enumerate(X.columns):
            # FIXED: Reduce verbosity - only show progress for first few features
            if verbose and idx < 5:
                print(f"  Computing effect size for: {feature}")
            
            pos_class = X.loc[y == 1, feature]
            neg_class = X.loc[y == 0, feature]
            
            # FIXED: Reduce verbosity - only show details for first few features
            if verbose and idx < 5:
                print(f"    Positive class: {len(pos_class)} samples, mean={pos_class.mean():.4f}")
                print(f"    Negative class: {len(neg_class)} samples, mean={neg_class.mean():.4f}")
            
            # Handle potential errors and skip problematic features
            try:
                effect_result = self._compute_effect_size(
                    pos_class, neg_class, feature, feature_categories
                )
                
                # FIXED: Reduce verbosity - only show details for first few features
                if verbose and idx < 5:
                    print(f"    Effect size: {effect_result['effect_size']:.4f} ({effect_result['effect_size_interpretation']})")
                
                results.append(effect_result)
            except Exception as e:
                # FIXED: Only print error messages for verbose mode
                if verbose:
                    print(f"Error computing effect size for {feature}: {str(e)}")
                
                # Add a dummy result to maintain data structure
                results.append({
                    'feature': feature,
                    'effect_size': 0,
                    'effect_type': 'Error',
                    'feature_type': 'Unknown',
                    'effect_size_interpretation': 'error'
                })
        
        # Convert to DataFrame and sort
        results_df = pd.DataFrame(results)
        
        if len(results_df) > 0:
            # Sort by absolute effect size
            results_df['abs_effect_size'] = results_df['effect_size'].abs()
            results_df = results_df.sort_values('abs_effect_size', ascending=False).reset_index(drop=True)
            
            if verbose:
                large_effects = (results_df['abs_effect_size'] >= 0.8).sum()
                medium_effects = ((results_df['abs_effect_size'] >= 0.5) & (results_df['abs_effect_size'] < 0.8)).sum()
                small_effects = ((results_df['abs_effect_size'] >= 0.2) & (results_df['abs_effect_size'] < 0.5)).sum()
                
                print(f"[Effect Sizes] Large effect sizes (≥0.8): {large_effects}")
                print(f"[Effect Sizes] Medium effect sizes (0.5-0.8): {medium_effects}")
                print(f"[Effect Sizes] Small effect sizes (0.2-0.5): {small_effects}")
                
                # FIXED: Show only top 5 to avoid overwhelming output
                print(f"[Effect Sizes] Top 5 largest effect sizes:")
                for i, row in results_df.head(5).iterrows():
                    print(f"  {row['feature']}: {row['effect_size']:.4f} ({row['effect_size_interpretation']})")
        
        # Create publication-ready plot
        if len(results_df) > 0:
            self._plot_effect_sizes(results_df, top_k)
        else:
            print("[Effect Sizes] Warning: No results to plot")
        
        return {
            'full_df': results_df,
            'top_k': results_df.head(top_k),
            'num_large_effects': (results_df['abs_effect_size'] >= 0.8).sum() if len(results_df) > 0 else 0,
            'num_medium_effects': ((results_df['abs_effect_size'] >= 0.5) & (results_df['abs_effect_size'] < 0.8)).sum() if len(results_df) > 0 else 0
        }
    
    def analyze_mutual_information(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        top_k: int = 20,
        feature_categories: Dict = None,
        verbose: int = 1
    ) -> Dict:
        """
        Analyze feature importance via mutual information.
        """
        if feature_categories is None:
            feature_categories = self._classify_features(X, y)
        
        results = []
        encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        
        categorical_features = set(
            feature_categories.get('categorical_features', []) + 
            feature_categories.get('derived_categorical_features', [])
        )
        
        for feature in X.columns:
            if verbose and len(results) < 10:
                print(f"  Computing mutual information for: {feature}")
            
            is_categorical = feature in categorical_features
            
            # Prepare feature data
            feature_data = X[feature].values.reshape(-1, 1)
            if is_categorical:
                try:
                    feature_data = encoder.fit_transform(feature_data)
                except:
                    feature_data = np.array([str(x) for x in X[feature]]).reshape(-1, 1)
                    feature_data = encoder.fit_transform(feature_data)
            
            # Calculate mutual information
            mi_score = mutual_info_classif(
                feature_data, y.astype(int), discrete_features=is_categorical
            )[0]
            
            results.append({
                'feature': feature,
                'mi_score': mi_score,
                'feature_type': 'Categorical' if is_categorical else 'Numerical'
            })
        
        # Convert to DataFrame and sort
        results_df = pd.DataFrame(results).sort_values(
            by='mi_score', ascending=False
        ).reset_index(drop=True)
        
        # Create publication-ready plot
        self._plot_mutual_information(results_df, top_k)
        
        return {
            'full_df': results_df,
            'top_k': results_df.head(top_k)
        }
    
    def create_comparison_plots(self, results: Dict, top_k: int = 20):
        """
        Create comparison plots across all methods.
        """
        # Extract top features from each method
        method_features = {}
        
        if 'xgboost' in results:
            # Use 'gain' as primary XGBoost importance
            primary_xgb = results['xgboost'].get('gain', results['xgboost'].get('weight'))
            if primary_xgb:
                method_features['XGBoost'] = set(primary_xgb['top_k']['feature'].tolist())
        
        if 'hypothesis_testing' in results:
            method_features['Hypothesis Testing'] = set(
                results['hypothesis_testing']['top_k']['feature'].tolist()
            )
        
        if 'effect_sizes' in results:
            method_features['Effect Sizes'] = set(
                results['effect_sizes']['top_k']['feature'].tolist()
            )
        
        if 'mutual_info' in results:
            method_features['Mutual Information'] = set(
                results['mutual_info']['top_k']['feature'].tolist()
            )
        
        # Create overlap heatmap
        self._plot_method_overlap(method_features, top_k)
        
        # Create ranking comparison
        self._plot_ranking_comparison(results, top_k)
    
    def _plot_xgboost_importance(self, importance_df: pd.DataFrame, importance_type: str, top_k: int):
        """Create publication-ready XGBoost importance plot."""
        plt.figure(figsize=self.figsize)
        
        # FIXED: Reverse order so most important features appear at TOP
        importance_df_reversed = importance_df.iloc[::-1]
        
        # Create color gradient
        colors = plt.cm.viridis(np.linspace(0, 1, len(importance_df_reversed)))
        
        # Create horizontal bar plot
        bars = plt.barh(
            range(len(importance_df_reversed)), 
            importance_df_reversed['importance_score'],
            color=colors,
            edgecolor='black',
            linewidth=0.5
        )
        
        # Customize plot
        plt.yticks(range(len(importance_df_reversed)), importance_df_reversed['feature'])
        plt.xlabel('Importance Score', fontsize=14, fontweight='bold')
        plt.ylabel('Features (Most Important → Top)', fontsize=14, fontweight='bold')
        plt.title(f'XGBoost Feature Importance ({importance_type.title()})\nTop {top_k} Features', 
                 fontsize=16, fontweight='bold', pad=20)
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(width + width*0.01, bar.get_y() + bar.get_height()/2, 
                    f'{width:.3f}', ha='left', va='center', fontsize=10)
        
        # Grid and styling
        plt.grid(axis='x', alpha=0.3, linestyle='--')
        plt.tight_layout()
        
        # Save plot
        filename = f'{self.subject}_xgboost_importance_{importance_type}.png'
        plt.savefig(os.path.join(self.output_dir, filename), **self.plot_params)
        plt.close()
    
    def _plot_hypothesis_testing_results(self, results_df: pd.DataFrame, top_k: int):
        """Create publication-ready hypothesis testing results plot."""
        if len(results_df) == 0:
            print(f"[Feature Importance] Warning: No hypothesis testing results to plot")
            return
        
        if verbose_debugging := True:  # Enable for debugging
            print(f"[Feature Importance] Plotting hypothesis testing results:")
            print(f"  - DataFrame shape: {results_df.shape}")
            print(f"  - Columns: {list(results_df.columns)}")
            print(f"  - Top 5 p-values: {results_df['p_value_fdr'].head().tolist()}")
            print(f"  - Significant features: {results_df['significant_fdr'].sum()}")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Plot 1: -log10(p-value) with significance threshold
        top_results = results_df.head(top_k)
        
        if len(top_results) == 0:
            print(f"[Feature Importance] Warning: No top results to plot")
            plt.close(fig)
            return
        
        # FIXED: Reverse order so most important features appear at TOP
        top_results_reversed = top_results.iloc[::-1]
        
        # FIXED: Use fallback logic for FDR-adjusted p-values if they're too small
        fdr_values = top_results_reversed['neg_log10_p_fdr']
        fdr_max = fdr_values.max()
        
        if fdr_max < 2.0:  # If FDR-adjusted values are too small (p > 0.01)
            print(f"[Hypothesis Testing Plot] FDR-adjusted p-values too small for visualization (max -log10 = {fdr_max:.3f})")
            print(f"[Hypothesis Testing Plot] Using raw p-values for better visibility")
            # Use raw p-values instead
            plot_values = -np.log10(np.maximum(top_results_reversed['p_value'], 1e-300))
            ylabel_text = '-log₁₀(Raw p-value)'
            title_text = 'Statistical Significance of Features (Raw p-values)'
        else:
            plot_values = fdr_values
            ylabel_text = '-log₁₀(FDR-adjusted p-value)'
            title_text = 'Statistical Significance of Features'
        
        colors = ['red' if sig else 'blue' for sig in top_results_reversed['significant_fdr']]
        
        bars1 = ax1.barh(
            range(len(top_results_reversed)), 
            plot_values,
            color=colors,
            alpha=0.7,
            edgecolor='black',
            linewidth=0.5
        )
        
        ax1.set_yticks(range(len(top_results_reversed)))
        ax1.set_yticklabels(top_results_reversed['feature'])
        ax1.set_xlabel(ylabel_text, fontsize=12, fontweight='bold')
        ax1.set_ylabel('Features (Most Significant → Top)', fontsize=12, fontweight='bold')
        ax1.set_title(title_text, fontsize=14, fontweight='bold')
        
        # Add significance threshold line
        if fdr_max < 2.0:
            # For raw p-values, use 0.05 threshold
            threshold_line = -np.log10(0.05)
        else:
            # For FDR-corrected, use 0.05 threshold
            threshold_line = -np.log10(0.05)
        
        ax1.axvline(x=threshold_line, color='red', linestyle='--', alpha=0.5, label='p=0.05')
        ax1.legend()
        ax1.grid(axis='x', alpha=0.3)
        
        # Add value labels on bars for top results
        for i, bar in enumerate(bars1):
            width = bar.get_width()
            if width > 0:  # Only label non-zero bars
                ax1.text(width + width*0.02, bar.get_y() + bar.get_height()/2, 
                        f'{width:.2f}', ha='left', va='center', fontsize=8)
        
        # Plot 2: Test statistics (also reversed)
        bars2 = ax2.barh(
            range(len(top_results_reversed)), 
            top_results_reversed['test_statistic'],
            color=plt.cm.plasma(np.linspace(0, 1, len(top_results_reversed))),
            edgecolor='black',
            linewidth=0.5
        )
        
        ax2.set_yticks(range(len(top_results_reversed)))
        ax2.set_yticklabels(top_results_reversed['feature'])
        ax2.set_xlabel('Test Statistic', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Features (Most Significant → Top)', fontsize=12, fontweight='bold')
        ax2.set_title('Test Statistics', fontsize=14, fontweight='bold')
        ax2.grid(axis='x', alpha=0.3)
        
        # Add value labels on bars for test statistics
        for i, bar in enumerate(bars2):
            width = bar.get_width()
            if abs(width) > 0.001:  # Only label meaningful values
                ax2.text(width + width*0.02 if width > 0 else width - abs(width)*0.02, 
                        bar.get_y() + bar.get_height()/2, 
                        f'{width:.2f}', ha='left' if width > 0 else 'right', va='center', fontsize=8)
        
        plt.tight_layout()
        
        # Save plot
        filename = f'{self.subject}_hypothesis_testing_results.png'
        plot_path = os.path.join(self.output_dir, filename)
        plt.savefig(plot_path, **self.plot_params)
        
        if verbose_debugging:
            print(f"[Feature Importance] Hypothesis testing plot saved to: {plot_path}")
            
        plt.close()  # Explicitly close the figure to free memory
    
    def _plot_effect_sizes(self, results_df: pd.DataFrame, top_k: int):
        """Create publication-ready effect sizes plot."""
        if len(results_df) == 0:
            print(f"[Feature Importance] Warning: No effect size results to plot")
            return
        
        if verbose_debugging := True:  # Enable for debugging
            print(f"[Feature Importance] Plotting effect sizes:")
            print(f"  - DataFrame shape: {results_df.shape}")
            print(f"  - Columns: {list(results_df.columns)}")
            print(f"  - Effect size range: [{results_df['effect_size'].min():.4f}, {results_df['effect_size'].max():.4f}]")
            print(f"  - Large effects (≥0.8): {(results_df['abs_effect_size'] >= 0.8).sum()}")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        top_results = results_df.head(top_k)
        
        if len(top_results) == 0:
            print(f"[Feature Importance] Warning: No top effect size results to plot")
            plt.close(fig)
            return
        
        # FIXED: Reverse order so most important features appear at TOP
        top_results_reversed = top_results.iloc[::-1]
        
        # Plot 1: Effect sizes by magnitude
        colors = ['red' if x < 0 else 'blue' for x in top_results_reversed['effect_size']]
        
        bars1 = ax1.barh(
            range(len(top_results_reversed)), 
            top_results_reversed['effect_size'],
            color=colors,
            alpha=0.7,
            edgecolor='black',
            linewidth=0.5
        )
        
        ax1.set_yticks(range(len(top_results_reversed)))
        ax1.set_yticklabels(top_results_reversed['feature'])
        ax1.set_xlabel('Effect Size', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Features (Largest Effect → Top)', fontsize=12, fontweight='bold')
        ax1.set_title('Effect Sizes by Feature', fontsize=14, fontweight='bold')
        ax1.axvline(x=0, color='black', linestyle='-', alpha=0.5)
        ax1.grid(axis='x', alpha=0.3)
        
        # Add effect size magnitude guidelines
        for threshold, label in [(0.2, 'Small'), (0.5, 'Medium'), (0.8, 'Large')]:
            ax1.axvline(x=threshold, color='gray', linestyle='--', alpha=0.5)
            ax1.axvline(x=-threshold, color='gray', linestyle='--', alpha=0.5)
        
        # Add value labels on bars
        for i, bar in enumerate(bars1):
            width = bar.get_width()
            if abs(width) > 0.001:  # Only label meaningful values
                ax1.text(width + width*0.02 if width > 0 else width - abs(width)*0.02, 
                        bar.get_y() + bar.get_height()/2, 
                        f'{width:.3f}', ha='left' if width > 0 else 'right', va='center', fontsize=8)
        
        # Plot 2: Absolute effect sizes (also reversed)
        bars2 = ax2.barh(
            range(len(top_results_reversed)), 
            np.abs(top_results_reversed['effect_size']),
            color=plt.cm.viridis(np.linspace(0, 1, len(top_results_reversed))),
            edgecolor='black',
            linewidth=0.5
        )
        
        ax2.set_yticks(range(len(top_results_reversed)))
        ax2.set_yticklabels(top_results_reversed['feature'])
        ax2.set_xlabel('Absolute Effect Size', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Features (Largest Effect → Top)', fontsize=12, fontweight='bold')
        ax2.set_title('Effect Size Magnitudes', fontsize=14, fontweight='bold')
        ax2.grid(axis='x', alpha=0.3)
        
        # Add value labels on bars
        for i, bar in enumerate(bars2):
            width = bar.get_width()
            if width > 0.001:
                ax2.text(width + width*0.02, bar.get_y() + bar.get_height()/2, 
                        f'{width:.3f}', ha='left', va='center', fontsize=8)
        
        plt.tight_layout()
        
        # Save plot
        filename = f'{self.subject}_effect_sizes.png'
        plot_path = os.path.join(self.output_dir, filename)
        plt.savefig(plot_path, **self.plot_params)
        
        if verbose_debugging:
            print(f"[Feature Importance] Effect sizes plot saved to: {plot_path}")
            
        plt.close()
    
    def _plot_mutual_information(self, results_df: pd.DataFrame, top_k: int):
        """Create publication-ready mutual information plot."""
        if len(results_df) == 0:
            print(f"[Feature Importance] Warning: No mutual information results to plot")
            return
        
        if verbose_debugging := True:  # Enable for debugging
            print(f"[Feature Importance] Plotting mutual information:")
            print(f"  - DataFrame shape: {results_df.shape}")
            print(f"  - Columns: {list(results_df.columns)}")
            print(f"  - MI score range: [{results_df['mi_score'].min():.4f}, {results_df['mi_score'].max():.4f}]")
            print(f"  - Categorical features: {(results_df['feature_type'] == 'Categorical').sum()}")
            print(f"  - Numerical features: {(results_df['feature_type'] == 'Numerical').sum()}")
        
        plt.figure(figsize=self.figsize)
        
        top_results = results_df.head(top_k)
        
        if len(top_results) == 0:
            print(f"[Feature Importance] Warning: No top mutual information results to plot")
            plt.close()
            return
        
        # FIXED: Reverse order so most important features appear at TOP
        top_results_reversed = top_results.iloc[::-1]
        
        # Color by feature type
        colors = ['red' if ft == 'Categorical' else 'blue' for ft in top_results_reversed['feature_type']]
        
        bars = plt.barh(
            range(len(top_results_reversed)), 
            top_results_reversed['mi_score'],
            color=colors,
            alpha=0.7,
            edgecolor='black',
            linewidth=0.5
        )
        
        plt.yticks(range(len(top_results_reversed)), top_results_reversed['feature'])
        plt.xlabel('Mutual Information Score', fontsize=14, fontweight='bold')
        plt.ylabel('Features (Highest MI → Top)', fontsize=14, fontweight='bold')
        plt.title(f'Mutual Information Analysis\nTop {top_k} Features', 
                 fontsize=16, fontweight='bold', pad=20)
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            if width > 0.001:
                plt.text(width + width*0.02, bar.get_y() + bar.get_height()/2, 
                        f'{width:.3f}', ha='left', va='center', fontsize=8)
        
        # Add legend
        categorical_patch = plt.Rectangle((0, 0), 1, 1, facecolor='red', alpha=0.7)
        numerical_patch = plt.Rectangle((0, 0), 1, 1, facecolor='blue', alpha=0.7)
        plt.legend([categorical_patch, numerical_patch], ['Categorical', 'Numerical'], 
                  loc='lower right')
        
        # Grid and styling
        plt.grid(axis='x', alpha=0.3, linestyle='--')
        plt.tight_layout()
        
        # Save plot
        filename = f'{self.subject}_mutual_information.png'
        plot_path = os.path.join(self.output_dir, filename)
        plt.savefig(plot_path, **self.plot_params)
        
        if verbose_debugging:
            print(f"[Feature Importance] Mutual information plot saved to: {plot_path}")
            
        plt.close()
    
    def _plot_method_overlap(self, method_features: Dict, top_k: int):
        """Create method overlap heatmap."""
        methods = list(method_features.keys())
        if len(methods) < 2:
            return
        
        # Get all unique features
        all_features = set()
        for features in method_features.values():
            all_features.update(features)
        
        # Create overlap matrix
        overlap_matrix = np.zeros((len(methods), len(methods)))
        
        for i, method1 in enumerate(methods):
            for j, method2 in enumerate(methods):
                if i == j:
                    overlap_matrix[i, j] = len(method_features[method1])
                else:
                    overlap = len(method_features[method1] & method_features[method2])
                    overlap_matrix[i, j] = overlap
        
        # Create heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            overlap_matrix,
            annot=True,
            fmt='g',
            cmap='Blues',
            xticklabels=methods,
            yticklabels=methods,
            square=True,
            cbar_kws={'label': 'Number of Overlapping Features'}
        )
        
        plt.title(f'Feature Overlap Between Methods\n(Top {top_k} Features)', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Methods', fontsize=14, fontweight='bold')
        plt.ylabel('Methods', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save plot
        filename = f'{self.subject}_method_overlap.png'
        plt.savefig(os.path.join(self.output_dir, filename), **self.plot_params)
        plt.close()
    
    def _plot_ranking_comparison(self, results: Dict, top_k: int):
        """Create ranking comparison plot."""
        # Extract rankings from each method
        rankings = {}
        
        if 'xgboost' in results:
            primary_xgb = results['xgboost'].get('gain', results['xgboost'].get('weight'))
            if primary_xgb:
                rankings['XGBoost'] = primary_xgb['top_k'][['feature', 'importance_score']]
        
        if 'hypothesis_testing' in results:
            ht_data = results['hypothesis_testing']['top_k']
            
            # FIXED: Use fallback importance measure if FDR-corrected values are too small
            if 'neg_log10_p_fdr' in ht_data.columns:
                fdr_max = ht_data['neg_log10_p_fdr'].max()
                if fdr_max > 0.01:  # If we have some meaningful FDR-corrected values
                    ht_df = ht_data[['feature', 'neg_log10_p_fdr']].copy()
                    ht_df.columns = ['feature', 'importance_score']
                else:
                    # Fallback to raw p-values when FDR correction makes everything insignificant
                    print(f"[Feature Importance] FDR-corrected p-values too small (max={fdr_max:.6f}), using raw p-values for visualization")
                    ht_df = ht_data[['feature', 'p_value']].copy()
                    ht_df['importance_score'] = -np.log10(np.maximum(ht_df['p_value'], 1e-300))
                    ht_df = ht_df[['feature', 'importance_score']]
            else:
                # Fallback if column doesn't exist
                ht_df = ht_data[['feature', 'p_value']].copy()
                ht_df['importance_score'] = -np.log10(np.maximum(ht_df['p_value'], 1e-300))
                ht_df = ht_df[['feature', 'importance_score']]
            
            rankings['Hypothesis Testing'] = ht_df
        
        if 'effect_sizes' in results:
            es_data = results['effect_sizes']['top_k']
            
            # FIXED: Use absolute effect size with better handling for small values
            if 'effect_size' in es_data.columns:
                es_df = es_data[['feature', 'effect_size']].copy()
                es_df['importance_score'] = np.abs(es_df['effect_size'])
                
                # If all effect sizes are very small, scale them up for visibility
                max_effect = es_df['importance_score'].max()
                if max_effect < 0.01:  # Very small effect sizes
                    print(f"[Feature Importance] Effect sizes very small (max={max_effect:.6f}), scaling for visibility")
                    # Scale to make the largest effect size = 1.0 for better visualization
                    if max_effect > 0:
                        es_df['importance_score'] = es_df['importance_score'] / max_effect
                    else:
                        # If all are zero, create artificial ranking based on order
                        es_df['importance_score'] = np.linspace(0.1, 1.0, len(es_df))
                
                es_df = es_df[['feature', 'importance_score']]
            else:
                # Fallback: create artificial ranking
                es_df = es_data[['feature']].copy()
                es_df['importance_score'] = np.linspace(0.1, 1.0, len(es_df))
            
            rankings['Effect Sizes'] = es_df
        
        if 'mutual_info' in results:
            mi_df = results['mutual_info']['top_k'][['feature', 'mi_score']]
            mi_df.columns = ['feature', 'importance_score']
            rankings['Mutual Information'] = mi_df
        
        if len(rankings) < 2:
            return
        
        # Create ranking comparison plot
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        method_names = list(rankings.keys())
        
        # Store debug information
        debug_info = {}
        
        for i, method in enumerate(method_names):
            if i < len(axes):
                df = rankings[method]
                
                # FIXED: Reverse order so most important features appear at TOP
                df_reversed = df.iloc[::-1]
                
                # Get raw values for debugging
                raw_values = df_reversed['importance_score'].values
                max_raw = df['importance_score'].max()
                min_raw = df['importance_score'].min()
                
                # Store debug info
                debug_info[method] = {
                    'max_raw': max_raw,
                    'min_raw': min_raw,
                    'range': max_raw - min_raw,
                    'top_5_values': raw_values[-5:] if len(raw_values) >= 5 else raw_values
                }
                
                # FIXED: Better handling for all value ranges including zero
                if max_raw > 0:
                    # Always normalize to [0, 1] range for consistent visualization
                    normalized_scores = df_reversed['importance_score'] / max_raw
                    xlabel = 'Normalized Importance Score'
                else:
                    # If all values are zero, create a minimal artificial gradient for visibility
                    print(f"[Feature Importance] All {method} importance scores are zero, creating artificial ranking")
                    normalized_scores = np.linspace(0.1, 1.0, len(df_reversed)) / len(df_reversed)
                    xlabel = 'Artificial Ranking'
                
                # Ensure all bars are visible (minimum width)
                min_visible_width = 0.05
                normalized_scores = np.maximum(normalized_scores, min_visible_width)
                
                bars = axes[i].barh(
                    range(len(df_reversed)), 
                    normalized_scores,
                    color=plt.cm.Set3(i),
                    alpha=0.7,
                    edgecolor='black',
                    linewidth=0.5
                )
                
                axes[i].set_yticks(range(len(df_reversed)))
                axes[i].set_yticklabels(df_reversed['feature'], fontsize=8)
                axes[i].set_xlabel(xlabel, fontsize=10)
                axes[i].set_ylabel('Features (Most Important → Top)', fontsize=10)
                axes[i].set_title(f'{method}', fontsize=12, fontweight='bold')
                axes[i].grid(axis='x', alpha=0.3)
                
                # Set explicit axis limits to ensure bars are visible
                axes[i].set_xlim(0, max(1.1, max(normalized_scores) * 1.1))
                
                # Add value labels on bars showing the original values
                for j, (bar, orig_val) in enumerate(zip(bars, raw_values)):
                    width = bar.get_width()
                    if width > 0.01:  # Only label bars that are visible
                        if orig_val < 0.001:
                            label = f'{orig_val:.4f}'
                        elif orig_val < 0.1:
                            label = f'{orig_val:.3f}'
                        else:
                            label = f'{orig_val:.2f}'
                        
                        axes[i].text(
                            width + 0.02, 
                            bar.get_y() + bar.get_height()/2, 
                            label,
                            ha='left', va='center', fontsize=7
                        )
        
        # Remove empty subplots
        for i in range(len(method_names), len(axes)):
            fig.delaxes(axes[i])
        
        plt.suptitle(f'Feature Ranking Comparison\n(Top {top_k} Features)', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save plot
        filename = f'{self.subject}_ranking_comparison.png'
        plot_path = os.path.join(self.output_dir, filename)
        plt.savefig(plot_path, **self.plot_params)
        
        # Print debug information
        print(f"[Feature Importance] Ranking comparison plot saved to: {plot_path}")
        print(f"[Feature Importance] Debug info for small values:")
        for method, info in debug_info.items():
            print(f"  {method}: max={info['max_raw']:.4f}, min={info['min_raw']:.4f}, range={info['range']:.4f}")
            print(f"    Top 5 values: {[f'{v:.4f}' for v in info['top_5_values']]}")
        
        plt.close()
    
    def _classify_features(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Classify features by type."""
        numerical_features = []
        categorical_features = []
        
        for col in X.columns:
            # K-mer features are always numerical (they represent counts)
            if re.match(r'^\d+mer_', col):
                numerical_features.append(col)
            # Chromosome is always categorical, even when encoded as numbers
            elif col == 'chrom':
                categorical_features.append(col)
            # Gene type is always categorical, even when encoded as numbers  
            elif col == 'gene_type':
                categorical_features.append(col)
            # Other numeric columns
            elif X[col].dtype in ['int64', 'float64']:
                # Only use unique value threshold for features that are truly categorical 
                # in nature (like other encoded categorical variables)
                if X[col].nunique() <= 10 and not col.endswith('_score') and not col.endswith('_length') and not col.endswith('_position'):
                    categorical_features.append(col)
                else:
                    numerical_features.append(col)
            else:
                categorical_features.append(col)
        
        return {
            'numerical_features': numerical_features,
            'categorical_features': categorical_features,
            'derived_categorical_features': [],
            'motif_features': [col for col in X.columns if re.match(r'^\d+mer_', col)]
        }
    
    def _perform_statistical_test(self, pos_class, neg_class, feature, feature_categories, alpha):
        """Perform appropriate statistical test based on feature type."""
        numerical_features = set(feature_categories.get('numerical_features', []))
        categorical_features = set(feature_categories.get('categorical_features', []))
        
        # Initialize result
        result = {
            'feature': feature,
            'test_type': '',
            'test_statistic': 0,
            'p_value': 1.0,
            'significant': False
        }
        
        try:
            if feature in numerical_features:
                # Test for normality
                _, shapiro_p = shapiro(pos_class.tolist() + neg_class.tolist())
                
                if shapiro_p >= alpha:  # Normal distribution
                    stat, p_val = ttest_ind(pos_class, neg_class, equal_var=False)
                    result['test_type'] = 't-test'
                else:  # Non-normal distribution
                    stat, p_val = mannwhitneyu(pos_class, neg_class, alternative='two-sided')
                    result['test_type'] = 'Mann-Whitney U'
                
                result['test_statistic'] = stat
                result['p_value'] = p_val
                
            elif feature in categorical_features:
                # FIXED: Create contingency table - reset index to avoid duplicate labels
                pos_class_reset = pos_class.reset_index(drop=True)
                neg_class_reset = neg_class.reset_index(drop=True)
                
                # Create combined data with proper labeling
                combined_values = pd.concat([pos_class_reset, neg_class_reset], ignore_index=True)
                combined_labels = pd.concat([
                    pd.Series([1] * len(pos_class_reset)), 
                    pd.Series([0] * len(neg_class_reset))
                ], ignore_index=True)
                
                contingency = pd.crosstab(combined_values, combined_labels)
                
                if contingency.shape == (2, 2):
                    # Fisher's exact test for 2x2 tables
                    stat, p_val = fisher_exact(contingency)
                    result['test_type'] = "Fisher's Exact"
                else:
                    # Chi-square test for larger tables
                    stat, p_val, _, _ = chi2_contingency(contingency)
                    result['test_type'] = 'Chi-square'
                
                result['test_statistic'] = stat
                result['p_value'] = p_val
            
            result['significant'] = result['p_value'] < alpha
            
        except Exception as e:
            print(f"Error testing feature {feature}: {str(e)}")
            result['p_value'] = 1.0
            result['test_type'] = 'Error'
        
        return result
    
    def _apply_fdr_correction(self, results_df: pd.DataFrame, alpha: float):
        """Apply FDR correction to p-values."""
        # FIXED: Handle NaN p-values before applying FDR correction
        if len(results_df) == 0:
            return results_df
        
        # Check for invalid p-values and fix them
        invalid_mask = (results_df['p_value'].isna() | 
                       (results_df['p_value'] < 0) | 
                       (results_df['p_value'] > 1))
        
        if invalid_mask.any():
            print(f"[FDR Correction] Found {invalid_mask.sum()} invalid p-values, setting to 1.0")
            results_df.loc[invalid_mask, 'p_value'] = 1.0
        
        # Apply Benjamini-Hochberg correction
        try:
            rejected, pvals_corrected, _, _ = multipletests(
                results_df['p_value'], alpha=alpha, method='fdr_bh'
            )
            
            # Handle NaN values in corrected p-values
            pvals_corrected = np.nan_to_num(pvals_corrected, nan=1.0)
            
            results_df['p_value_fdr'] = pvals_corrected
            results_df['significant_fdr'] = rejected
            results_df['neg_log10_p_fdr'] = -np.log10(np.maximum(pvals_corrected, 1e-300))
            
        except Exception as e:
            print(f"[FDR Correction] Error applying FDR correction: {e}")
            # Fallback: no correction
            results_df['p_value_fdr'] = results_df['p_value']
            results_df['significant_fdr'] = results_df['p_value'] < alpha
            results_df['neg_log10_p_fdr'] = -np.log10(np.maximum(results_df['p_value'], 1e-300))
        
        return results_df.sort_values('p_value_fdr').reset_index(drop=True)
    
    def _compute_effect_size(self, pos_class, neg_class, feature, feature_categories):
        """Compute effect size based on feature type."""
        numerical_features = set(feature_categories.get('numerical_features', []))
        categorical_features = set(feature_categories.get('categorical_features', []))
        
        result = {
            'feature': feature,
            'effect_size': 0,
            'effect_type': '',
            'feature_type': '',
            'effect_size_interpretation': 'negligible'
        }
        
        try:
            if feature in numerical_features:
                # Cohen's d
                mean_diff = pos_class.mean() - neg_class.mean()
                pooled_std = np.sqrt(
                    ((len(pos_class)-1)*pos_class.var() + (len(neg_class)-1)*neg_class.var()) / 
                    (len(pos_class) + len(neg_class) - 2)
                )
                
                if pooled_std > 0:
                    cohens_d = mean_diff / pooled_std
                else:
                    cohens_d = 0
                
                result['effect_size'] = cohens_d
                result['effect_type'] = "Cohen's d"
                result['feature_type'] = 'Numerical'
                
                # Add interpretation
                abs_effect = abs(cohens_d)
                if abs_effect >= 0.8:
                    result['effect_size_interpretation'] = 'large'
                elif abs_effect >= 0.5:
                    result['effect_size_interpretation'] = 'medium'
                elif abs_effect >= 0.2:
                    result['effect_size_interpretation'] = 'small'
                else:
                    result['effect_size_interpretation'] = 'negligible'
                
            elif feature in categorical_features:
                # FIXED: Cramer's V - reset index to avoid duplicate labels
                pos_class_reset = pos_class.reset_index(drop=True)
                neg_class_reset = neg_class.reset_index(drop=True)
                
                # Create combined data with proper labeling
                combined_values = pd.concat([pos_class_reset, neg_class_reset], ignore_index=True)
                combined_labels = pd.concat([
                    pd.Series([1] * len(pos_class_reset)), 
                    pd.Series([0] * len(neg_class_reset))
                ], ignore_index=True)
                
                contingency = pd.crosstab(combined_values, combined_labels)
                
                chi2, _, _, _ = chi2_contingency(contingency)
                n = contingency.sum().sum()
                
                if n > 0:
                    phi2 = chi2 / n
                    r, k = contingency.shape
                    if min(k - 1, r - 1) > 0:
                        cramers_v = np.sqrt(phi2 / (min(k - 1, r - 1)))
                    else:
                        cramers_v = 0
                else:
                    cramers_v = 0
                
                result['effect_size'] = cramers_v
                result['effect_type'] = "Cramer's V"
                result['feature_type'] = 'Categorical'
                
                # Add interpretation (same thresholds as Cohen's d)
                if cramers_v >= 0.8:
                    result['effect_size_interpretation'] = 'large'
                elif cramers_v >= 0.5:
                    result['effect_size_interpretation'] = 'medium'
                elif cramers_v >= 0.2:
                    result['effect_size_interpretation'] = 'small'
                else:
                    result['effect_size_interpretation'] = 'negligible'
                
        except Exception as e:
            print(f"Error computing effect size for {feature}: {str(e)}")
            result['effect_size'] = 0
            result['effect_type'] = 'Error'
            result['effect_size_interpretation'] = 'error'
        
        return result
    
    def _generate_significance_summary(self, results_df: pd.DataFrame, alpha: float, verbose: int = 1) -> Dict:
        """Generate comprehensive summary of significance testing results."""
        
        significant_fdr = results_df[results_df['significant_fdr']].copy()
        significant_uncorrected = results_df[results_df['significant_uncorrected']].copy()
        
        # Classify significant features by test type
        test_type_summary = {}
        if len(significant_fdr) > 0:
            test_type_counts = significant_fdr['test_type'].value_counts()
            for test_type, count in test_type_counts.items():
                test_type_summary[test_type] = {
                    'count': count,
                    'features': significant_fdr[significant_fdr['test_type'] == test_type]['feature'].tolist()
                }
        
        # Effect size bins for significant features
        effect_strength_summary = {
            'very_strong': 0,  # -log10(p) >= 5 (p <= 1e-5)
            'strong': 0,       # -log10(p) >= 3 (p <= 1e-3)
            'moderate': 0,     # -log10(p) >= 2 (p <= 1e-2)
            'weak': 0          # -log10(p) < 2
        }
        
        if len(significant_fdr) > 0:
            for _, row in significant_fdr.iterrows():
                neg_log10_p = row['neg_log10_p_fdr']
                if neg_log10_p >= 5:
                    effect_strength_summary['very_strong'] += 1
                elif neg_log10_p >= 3:
                    effect_strength_summary['strong'] += 1
                elif neg_log10_p >= 2:
                    effect_strength_summary['moderate'] += 1
                else:
                    effect_strength_summary['weak'] += 1
        
        # Multiple testing impact
        multiple_testing_impact = {
            'total_features': len(results_df),
            'significant_uncorrected': len(significant_uncorrected),
            'significant_fdr_corrected': len(significant_fdr),
            'features_rejected_by_correction': len(significant_uncorrected) - len(significant_fdr),
            'correction_stringency': (len(significant_uncorrected) - len(significant_fdr)) / max(len(significant_uncorrected), 1),
            'alpha_level': alpha
        }
        
        summary = {
            'test_type_summary': test_type_summary,
            'effect_strength_summary': effect_strength_summary,
            'multiple_testing_impact': multiple_testing_impact,
            'most_significant_feature': significant_fdr.iloc[0]['feature'] if len(significant_fdr) > 0 else None,
            'strongest_effect_pvalue': significant_fdr.iloc[0]['p_value_fdr'] if len(significant_fdr) > 0 else None
        }
        
        return summary
    
    def save_results(self, filename: str = None):
        """Save all results to files including significance summaries."""
        if filename is None:
            filename = f'{self.subject}_feature_importance_results.xlsx'
        
        filepath = os.path.join(self.output_dir, filename)
        
        # Check if we have any results to save
        if not self.results:
            print(f"[Feature Importance] No results to save to Excel file")
            return
        
        sheets_written = 0
        
        try:
            with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                for method_name, method_results in self.results.items():
                    if method_name == 'xgboost':
                        for imp_type, imp_results in method_results.items():
                            if 'full_df' in imp_results and not imp_results['full_df'].empty:
                                sheet_name = f'{method_name}_{imp_type}'
                                imp_results['full_df'].to_excel(writer, sheet_name=sheet_name, index=False)
                                sheets_written += 1
                    else:
                        if 'full_df' in method_results and not method_results['full_df'].empty:
                            method_results['full_df'].to_excel(writer, sheet_name=method_name, index=False)
                            sheets_written += 1
                            
                            # Save significant features for hypothesis testing
                            if method_name == 'hypothesis_testing' and 'significant_features' in method_results:
                                if len(method_results['significant_features']) > 0:
                                    sheet_name = f'{method_name}_significant'
                                    method_results['significant_features'].to_excel(writer, sheet_name=sheet_name, index=False)
                                    sheets_written += 1
                
                # If no sheets were written, create a placeholder sheet
                if sheets_written == 0:
                    placeholder_df = pd.DataFrame({
                        'info': ['No feature importance results available'],
                        'reason': ['All analysis methods failed or produced empty results']
                    })
                    placeholder_df.to_excel(writer, sheet_name='placeholder', index=False)
                    sheets_written += 1
                    print(f"[Feature Importance] Created placeholder sheet due to no valid results")
            
            if sheets_written > 0:
                print(f"[Feature Importance] Results saved to: {filepath}")
                print(f"[Feature Importance] Sheets written: {sheets_written}")
            else:
                print(f"[Feature Importance] Warning: No sheets were written to Excel file")
                
        except Exception as e:
            print(f"[Feature Importance] Error saving Excel file: {e}")
            print(f"[Feature Importance] Attempting to save individual method results as CSV files...")
            
            # Fallback: save as CSV files
            for method_name, method_results in self.results.items():
                if method_name == 'xgboost':
                    for imp_type, imp_results in method_results.items():
                        if 'full_df' in imp_results and not imp_results['full_df'].empty:
                            csv_filename = f'{self.subject}_{method_name}_{imp_type}.csv'
                            csv_path = os.path.join(self.output_dir, csv_filename)
                            imp_results['full_df'].to_csv(csv_path, index=False)
                            print(f"[Feature Importance] Saved {csv_filename} as CSV fallback")
                else:
                    if 'full_df' in method_results and not method_results['full_df'].empty:
                        csv_filename = f'{self.subject}_{method_name}.csv'
                        csv_path = os.path.join(self.output_dir, csv_filename)
                        method_results['full_df'].to_csv(csv_path, index=False)
                        print(f"[Feature Importance] Saved {csv_filename} as CSV fallback")
        
        # Save detailed significance summary for hypothesis testing
        if 'hypothesis_testing' in self.results:
            try:
                self._save_significance_summary()
            except Exception as e:
                print(f"[Feature Importance] Error saving significance summary: {e}")
    
    def _save_significance_summary(self):
        """Save detailed significance summary to text file."""
        if 'hypothesis_testing' not in self.results:
            return
        
        ht_results = self.results['hypothesis_testing']
        if 'significance_summary' not in ht_results or not ht_results['significance_summary']:
            return
        
        summary = ht_results['significance_summary']
        summary_file = os.path.join(self.output_dir, f'{self.subject}_significance_summary.txt')
        
        with open(summary_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("STATISTICAL SIGNIFICANCE ANALYSIS SUMMARY\n")
            f.write("=" * 80 + "\n\n")
            
            # Multiple testing correction summary
            mti = summary['multiple_testing_impact']
            f.write("MULTIPLE TESTING CORRECTION (Benjamini-Hochberg FDR)\n")
            f.write("-" * 50 + "\n")
            f.write(f"Total features tested: {mti['total_features']}\n")
            f.write(f"Significance level (α): {mti['alpha_level']}\n")
            f.write(f"Expected false discovery rate: ≤ {mti['alpha_level']*100:.1f}%\n\n")
            
            f.write(f"Significant features (uncorrected): {mti['significant_uncorrected']}\n")
            f.write(f"Significant features (FDR-corrected): {mti['significant_fdr_corrected']}\n")
            f.write(f"Features rejected by correction: {mti['features_rejected_by_correction']}\n")
            f.write(f"Correction stringency: {mti['correction_stringency']:.1%}\n\n")
            
            # Test type breakdown
            if summary['test_type_summary']:
                f.write("SIGNIFICANT FEATURES BY TEST TYPE\n")
                f.write("-" * 50 + "\n")
                for test_type, info in summary['test_type_summary'].items():
                    f.write(f"{test_type}: {info['count']} features\n")
                    for feature in info['features']:
                        f.write(f"  - {feature}\n")
                    f.write("\n")
            
            # Effect strength summary
            ess = summary['effect_strength_summary']
            f.write("EFFECT STRENGTH DISTRIBUTION (by -log10(p_FDR))\n")
            f.write("-" * 50 + "\n")
            f.write(f"Very Strong (≥5.0, p≤1e-5): {ess['very_strong']} features\n")
            f.write(f"Strong (≥3.0, p≤1e-3): {ess['strong']} features\n") 
            f.write(f"Moderate (≥2.0, p≤1e-2): {ess['moderate']} features\n")
            f.write(f"Weak (<2.0): {ess['weak']} features\n\n")
            
            # Detailed significant features
            if len(ht_results['significant_features']) > 0:
                f.write("DETAILED SIGNIFICANT FEATURES LIST\n")
                f.write("-" * 50 + "\n")
                sig_features = ht_results['significant_features']
                for i, (_, row) in enumerate(sig_features.iterrows(), 1):
                    f.write(f"{i:2d}. {row['feature']}\n")
                    f.write(f"     Test: {row['test_type']}\n")
                    f.write(f"     Raw p-value: {row['p_value']:.6f}\n")
                    f.write(f"     FDR-adjusted p-value: {row['p_value_fdr']:.6f}\n")
                    f.write(f"     Test statistic: {row['test_statistic']:.4f}\n")
                    f.write(f"     -log10(p_FDR): {row['neg_log10_p_fdr']:.2f}\n\n")
            else:
                f.write("NO FEATURES REMAIN SIGNIFICANT AFTER FDR CORRECTION\n")
                f.write("-" * 50 + "\n")
                f.write("This suggests that initially significant results may be false positives.\n")
                f.write("Recommendations:\n")
                f.write("1. Increase sample size for more statistical power\n")
                f.write("2. Consider more stringent significance level\n")
                f.write("3. Focus on effect size analysis instead of p-values\n")
                f.write("4. Use domain knowledge to prioritize features\n\n")
            
            f.write("=" * 80 + "\n")
            f.write(f"Analysis generated by Feature Importance Analyzer\n")
            f.write(f"Subject: {self.subject}\n")
            f.write("=" * 80 + "\n")
        
        print(f"Significance summary saved to: {summary_file}")


def analyze_feature_importance(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    output_dir: str = None,
    subject: str = "feature_analysis",
    top_k: int = 20,
    methods: List[str] = None,
    verbose: int = 1
) -> Dict:
    """
    Convenience function to run comprehensive feature importance analysis.
    
    Parameters
    ----------
    model : sklearn-compatible model
        Trained model
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target variable
    output_dir : str, optional
        Directory to save results
    subject : str, optional
        Subject name for files
    top_k : int, optional
        Number of top features
    methods : List[str], optional
        Methods to use
    verbose : int, optional
        Verbosity level
        
    Returns
    -------
    Dict
        Results dictionary
    """
    analyzer = FeatureImportanceAnalyzer(output_dir=output_dir, subject=subject)
    
    results = analyzer.run_comprehensive_analysis(
        model=model,
        X=X,
        y=y,
        top_k=top_k,
        methods=methods,
        verbose=verbose
    )
    
    # Save results
    analyzer.save_results()
    
    return results 