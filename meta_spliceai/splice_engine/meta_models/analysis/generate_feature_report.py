#!/usr/bin/env python3
"""
Feature Analysis Report Generator

This script generates comprehensive reports explaining the probability-based features
and their importance in splice site prediction, with specific focus on the features
mentioned in SHAP analysis.

Usage:
    python generate_feature_report.py --data-file enhanced_positions.tsv --output-dir reports/
"""

import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass

@dataclass
class FeatureInfo:
    """Information about a specific feature."""
    name: str
    category: str
    description: str
    formula: str
    interpretation: str
    use_case: str
    good_range: str
    poor_range: str

# Define feature catalog
FEATURE_CATALOG = {
    # Basic probability features
    'donor_score': FeatureInfo(
        name='donor_score',
        category='Basic Probability',
        description='Raw SpliceAI probability for donor splice site',
        formula='Direct output from SpliceAI model',
        interpretation='Higher values indicate stronger donor signal',
        use_case='Primary donor site detection',
        good_range='> 0.5 (above threshold)',
        poor_range='< 0.1 (very weak signal)'
    ),
    
    'acceptor_score': FeatureInfo(
        name='acceptor_score', 
        category='Basic Probability',
        description='Raw SpliceAI probability for acceptor splice site',
        formula='Direct output from SpliceAI model',
        interpretation='Higher values indicate stronger acceptor signal', 
        use_case='Primary acceptor site detection',
        good_range='> 0.5 (above threshold)',
        poor_range='< 0.1 (very weak signal)'
    ),
    
    'splice_probability': FeatureInfo(
        name='splice_probability',
        category='Derived Probability',
        description='Combined probability of any splice site (donor or acceptor)',
        formula='(donor_score + acceptor_score) / (donor_score + acceptor_score + neither_score)',
        interpretation='Overall confidence that position is a splice site',
        use_case='General splice site detection, threshold optimization',
        good_range='> 0.7 (high confidence)',
        poor_range='< 0.3 (low confidence)'
    ),
    
    'splice_neither_diff': FeatureInfo(
        name='splice_neither_diff',
        category='Derived Probability', 
        description='Difference between splice signal and neither signal',
        formula='(max(donor_score, acceptor_score) - neither_score) / max(all_scores)',
        interpretation='Strength of splice signal relative to background',
        use_case='Distinguishing true splice sites from background noise',
        good_range='> 0.3 (strong splice signal)',
        poor_range='< 0.1 (weak discrimination)'
    ),
    
    # Peak detection features
    'donor_peak_height_ratio': FeatureInfo(
        name='donor_peak_height_ratio',
        category='Peak Detection',
        description='How many times higher the donor peak is compared to neighbors',
        formula='donor_score / (neighbor_mean + epsilon)',
        interpretation='Peak prominence - sharp peaks indicate true sites',
        use_case='Distinguishing sharp true peaks from broad false signals',
        good_range='> 2.0 (prominent peak)',
        poor_range='< 1.5 (weak or broad peak)'
    ),
    
    'acceptor_peak_height_ratio': FeatureInfo(
        name='acceptor_peak_height_ratio',
        category='Peak Detection',
        description='How many times higher the acceptor peak is compared to neighbors',
        formula='acceptor_score / (neighbor_mean + epsilon)',
        interpretation='Peak prominence - sharp peaks indicate true sites',
        use_case='Distinguishing sharp true peaks from broad false signals',
        good_range='> 2.0 (prominent peak)',
        poor_range='< 1.5 (weak or broad peak)'
    ),
    
    # Signal processing features
    'donor_second_derivative': FeatureInfo(
        name='donor_second_derivative',
        category='Signal Processing',
        description='Curvature of the donor signal at the splice site',
        formula='(donor_score - context_m1) - (context_p1 - donor_score)',
        interpretation='Positive = sharp peak (concave up), Negative = broad signal (concave down)',
        use_case='Detecting characteristic sharp peaks of true splice sites',
        good_range='> 0.05 (sharp curvature)',
        poor_range='< 0 (broad/flat signal)'
    ),
    
    'acceptor_second_derivative': FeatureInfo(
        name='acceptor_second_derivative',
        category='Signal Processing', 
        description='Curvature of the acceptor signal at the splice site',
        formula='(acceptor_score - context_m1) - (context_p1 - acceptor_score)',
        interpretation='Positive = sharp peak (concave up), Negative = broad signal (concave down)',
        use_case='Detecting characteristic sharp peaks of true splice sites',
        good_range='> 0.05 (sharp curvature)',
        poor_range='< 0 (broad/flat signal)'
    ),
    
    'donor_signal_strength': FeatureInfo(
        name='donor_signal_strength',
        category='Signal Processing',
        description='Donor signal above local background level',
        formula='donor_score - neighbor_mean',
        interpretation='How much the signal stands out from background',
        use_case='Background subtraction to identify true signals',
        good_range='> 0.2 (strong signal)',
        poor_range='< 0.05 (weak signal)'
    ),
    
    'acceptor_signal_strength': FeatureInfo(
        name='acceptor_signal_strength',
        category='Signal Processing',
        description='Acceptor signal above local background level', 
        formula='acceptor_score - neighbor_mean',
        interpretation='How much the signal stands out from background',
        use_case='Background subtraction to identify true signals',
        good_range='> 0.2 (strong signal)',
        poor_range='< 0.05 (weak signal)'
    ),
    
    # Cross-type comparison features
    'type_signal_difference': FeatureInfo(
        name='type_signal_difference',
        category='Cross-Type Comparison',
        description='Difference between donor and acceptor signal strengths',
        formula='donor_signal_strength - acceptor_signal_strength',
        interpretation='Positive = donor preferred, Negative = acceptor preferred',
        use_case='Distinguishing donor from acceptor splice sites',
        good_range='|value| > 0.1 (clear type preference)',
        poor_range='|value| < 0.05 (ambiguous type)'
    ),
    
    'donor_acceptor_peak_ratio': FeatureInfo(
        name='donor_acceptor_peak_ratio',
        category='Cross-Type Comparison',
        description='Ratio of donor to acceptor peak heights',
        formula='donor_peak_height_ratio / (acceptor_peak_height_ratio + epsilon)',
        interpretation='> 1 = donor dominant, < 1 = acceptor dominant',
        use_case='Type discrimination in ambiguous cases',
        good_range='> 2.0 or < 0.5 (clear preference)',
        poor_range='0.8 - 1.2 (ambiguous)'
    ),
    
    'score_difference_ratio': FeatureInfo(
        name='score_difference_ratio',
        category='Cross-Type Comparison',
        description='Normalized difference between donor and acceptor scores',
        formula='(donor_score - acceptor_score) / (donor_score + acceptor_score + epsilon)',
        interpretation='Ranges from -1 (pure acceptor) to +1 (pure donor)',
        use_case='Type classification with symmetric interpretation',
        good_range='|value| > 0.5 (strong type preference)',
        poor_range='|value| < 0.2 (weak preference)'
    ),
}

def analyze_feature_importance(df: pl.DataFrame, feature_name: str) -> Dict:
    """Analyze the importance and characteristics of a specific feature."""
    
    # Convert to pandas for easier analysis
    df_pd = df.to_pandas()
    
    if feature_name not in df_pd.columns:
        return {'error': f'Feature {feature_name} not found in data'}
    
    feature_data = df_pd[feature_name].dropna()
    
    analysis = {
        'feature_name': feature_name,
        'total_values': len(feature_data),
        'missing_values': df_pd[feature_name].isna().sum(),
        'basic_stats': {
            'mean': feature_data.mean(),
            'std': feature_data.std(),
            'min': feature_data.min(),
            'max': feature_data.max(),
            'median': feature_data.median(),
            'q25': feature_data.quantile(0.25),
            'q75': feature_data.quantile(0.75)
        }
    }
    
    # Analyze by prediction type if available
    if 'pred_type' in df_pd.columns:
        type_analysis = {}
        for pred_type in ['TP', 'FP', 'FN', 'TN']:
            if pred_type in df_pd['pred_type'].values:
                subset = df_pd[df_pd['pred_type'] == pred_type][feature_name].dropna()
                if len(subset) > 0:
                    type_analysis[pred_type] = {
                        'count': len(subset),
                        'mean': subset.mean(),
                        'std': subset.std(),
                        'median': subset.median()
                    }
        analysis['by_prediction_type'] = type_analysis
    
    # Analyze by splice type if available  
    if 'splice_type' in df_pd.columns:
        splice_analysis = {}
        for splice_type in ['donor', 'acceptor', 'neither']:
            if splice_type in df_pd['splice_type'].values:
                subset = df_pd[df_pd['splice_type'] == splice_type][feature_name].dropna()
                if len(subset) > 0:
                    splice_analysis[splice_type] = {
                        'count': len(subset),
                        'mean': subset.mean(),
                        'std': subset.std(),
                        'median': subset.median()
                    }
        analysis['by_splice_type'] = splice_analysis
    
    return analysis

def generate_feature_report(df: pl.DataFrame, output_dir: Path) -> str:
    """Generate a comprehensive feature analysis report."""
    
    # Create report content
    report_lines = []
    
    # Header
    report_lines.extend([
        "# Probability Feature Analysis Report",
        "",
        f"**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Dataset:** {df.height:,} positions with {len(df.columns)} features",
        "",
        "This report analyzes the probability-based features derived from SpliceAI predictions",
        "and their utility for meta-model training and splice site error correction.",
        "",
        "---",
        ""
    ])
    
    # Dataset overview
    df_pd = df.to_pandas()
    
    report_lines.extend([
        "## Dataset Overview",
        "",
        f"- **Total positions:** {df.height:,}",
        f"- **Total features:** {len(df.columns)}",
        ""
    ])
    
    # Prediction type distribution
    if 'pred_type' in df_pd.columns:
        pred_counts = df_pd['pred_type'].value_counts()
        report_lines.extend([
            "### Prediction Type Distribution",
            ""
        ])
        for pred_type, count in pred_counts.items():
            percentage = count / len(df_pd) * 100
            report_lines.append(f"- **{pred_type}:** {count:,} ({percentage:.1f}%)")
        report_lines.append("")
    
    # Splice type distribution
    if 'splice_type' in df_pd.columns:
        splice_counts = df_pd['splice_type'].value_counts()
        report_lines.extend([
            "### Splice Type Distribution",
            ""
        ])
        for splice_type, count in splice_counts.items():
            percentage = count / len(df_pd) * 100
            report_lines.append(f"- **{splice_type}:** {count:,} ({percentage:.1f}%)")
        report_lines.append("")
    
    # Analyze key features
    report_lines.extend([
        "---",
        "",
        "## Feature Analysis",
        "",
        "Analysis of key probability-based features and their characteristics.",
        ""
    ])
    
    # Get available features from our catalog
    available_features = [name for name in FEATURE_CATALOG.keys() if name in df_pd.columns]
    
    if not available_features:
        report_lines.extend([
            "**Warning:** No catalogued features found in dataset.",
            "",
            "Available columns:",
        ])
        for col in df_pd.columns[:20]:  # Show first 20 columns
            report_lines.append(f"- {col}")
        if len(df_pd.columns) > 20:
            report_lines.append(f"- ... and {len(df_pd.columns) - 20} more")
        report_lines.append("")
    else:
        # Analyze each available feature
        for feature_name in available_features:
            feature_info = FEATURE_CATALOG[feature_name]
            analysis = analyze_feature_importance(df, feature_name)
            
            if 'error' in analysis:
                continue
                
            report_lines.extend([
                f"### {feature_name.replace('_', ' ').title()}",
                "",
                f"**Category:** {feature_info.category}",
                "",
                f"**Description:** {feature_info.description}",
                "",
                f"**Formula:** `{feature_info.formula}`",
                "",
                f"**Interpretation:** {feature_info.interpretation}",
                "",
                f"**Use Case:** {feature_info.use_case}",
                "",
                "#### Statistical Summary",
                "",
                f"- **Total values:** {analysis['total_values']:,}",
                f"- **Missing values:** {analysis['missing_values']:,}",
                f"- **Mean:** {analysis['basic_stats']['mean']:.4f}",
                f"- **Std Dev:** {analysis['basic_stats']['std']:.4f}",
                f"- **Range:** {analysis['basic_stats']['min']:.4f} to {analysis['basic_stats']['max']:.4f}",
                f"- **Median:** {analysis['basic_stats']['median']:.4f}",
                f"- **IQR:** {analysis['basic_stats']['q25']:.4f} to {analysis['basic_stats']['q75']:.4f}",
                ""
            ])
            
            # Add analysis by prediction type
            if 'by_prediction_type' in analysis and analysis['by_prediction_type']:
                report_lines.extend([
                    "#### Performance by Prediction Type",
                    ""
                ])
                
                for pred_type in ['TP', 'FP', 'FN', 'TN']:
                    if pred_type in analysis['by_prediction_type']:
                        stats = analysis['by_prediction_type'][pred_type]
                        report_lines.append(
                            f"- **{pred_type}:** μ={stats['mean']:.4f}, "
                            f"σ={stats['std']:.4f}, n={stats['count']:,}"
                        )
                report_lines.append("")
            
            # Add analysis by splice type
            if 'by_splice_type' in analysis and analysis['by_splice_type']:
                report_lines.extend([
                    "#### Performance by Splice Type",
                    ""
                ])
                
                for splice_type in ['donor', 'acceptor', 'neither']:
                    if splice_type in analysis['by_splice_type']:
                        stats = analysis['by_splice_type'][splice_type]
                        report_lines.append(
                            f"- **{splice_type}:** μ={stats['mean']:.4f}, "
                            f"σ={stats['std']:.4f}, n={stats['count']:,}"
                        )
                report_lines.append("")
            
            # Add interpretation guidance
            report_lines.extend([
                "#### Interpretation Guidance",
                "",
                f"- **Good range:** {feature_info.good_range}",
                f"- **Poor range:** {feature_info.poor_range}",
                "",
                "---",
                ""
            ])
    
    # Feature relationships and correlations
    if len(available_features) > 1:
        report_lines.extend([
            "## Feature Relationships",
            "",
            "Analysis of correlations between key features.",
            ""
        ])
        
        # Calculate correlation matrix
        corr_matrix = df_pd[available_features].corr()
        
        # Find strongest correlations
        corr_pairs = []
        for i in range(len(available_features)):
            for j in range(i+1, len(available_features)):
                feat1, feat2 = available_features[i], available_features[j]
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.3:  # Only show moderate to strong correlations
                    corr_pairs.append((feat1, feat2, corr_val))
        
        # Sort by absolute correlation strength
        corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        
        if corr_pairs:
            report_lines.extend([
                "### Notable Feature Correlations",
                "",
                "Correlations with |r| > 0.3:",
                ""
            ])
            
            for feat1, feat2, corr_val in corr_pairs[:10]:  # Top 10
                relationship = "positively" if corr_val > 0 else "negatively"
                strength = "strongly" if abs(corr_val) > 0.7 else "moderately"
                report_lines.append(
                    f"- **{feat1}** vs **{feat2}:** r={corr_val:.3f} "
                    f"({strength} {relationship} correlated)"
                )
            report_lines.append("")
    
    # Usage recommendations
    report_lines.extend([
        "---",
        "",
        "## Usage Recommendations",
        "",
        "### For False Positive Reduction",
        "",
        "Focus on these features to identify likely false positives:",
        "",
        "- **Peak quality:** Low `peak_height_ratio` (< 1.5) indicates weak peaks",
        "- **Signal shape:** Negative `second_derivative` indicates broad/flat signals", 
        "- **Background:** Low `signal_strength` indicates poor signal-to-noise ratio",
        "- **Overall confidence:** Low `splice_probability` (< 0.3) indicates weak predictions",
        "",
        "### For False Negative Rescue",
        "",
        "Look for these patterns in 'neither' predictions:",
        "",
        "- **Moderate confidence:** `splice_probability` between 0.3-0.5 (below threshold)",
        "- **Good signal quality:** Positive `second_derivative` with reasonable `peak_height_ratio`",
        "- **Type preference:** Strong `type_signal_difference` indicating clear type",
        "",
        "### For Type Discrimination",
        "",
        "Use these features to distinguish donor from acceptor sites:",
        "",
        "- **Primary:** `type_signal_difference` (positive = donor, negative = acceptor)",
        "- **Secondary:** `donor_acceptor_peak_ratio` (> 2.0 = donor, < 0.5 = acceptor)",
        "- **Normalized:** `score_difference_ratio` (ranges from -1 to +1)",
        "",
        "---",
        "",
        "## Technical Notes",
        "",
        "### Feature Engineering Principles",
        "",
        "1. **Biological relevance:** Features reflect known splice site properties",
        "2. **Signal processing:** Peak detection and noise reduction techniques",
        "3. **Numerical stability:** Epsilon values prevent division by zero",
        "4. **Interpretability:** Clear mathematical and biological meaning",
        "",
        "### Computational Details",
        "",
        "- **Context positions:** m2=-2, m1=-1, pos=0, p1=+1, p2=+2",
        "- **Epsilon value:** 1e-10 for safe division operations",
        "- **Peak threshold:** 0.001 minimum for local peak detection",
        "- **Missing values:** Handled by dropping or filling based on context",
        "",
        "### Validation Approaches",
        "",
        "- **SHAP analysis:** Feature importance in trained meta-models",
        "- **Correlation analysis:** Independence and redundancy checking", 
        "- **Biological validation:** Consistency with known splice site biology",
        "- **Performance metrics:** Impact on FP reduction and FN rescue rates",
        ""
    ])
    
    # Save report
    report_path = output_dir / "feature_analysis_report.md"
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    
    return str(report_path)

def create_summary_table(df: pl.DataFrame, output_dir: Path) -> str:
    """Create a summary table of all available features."""
    
    df_pd = df.to_pandas()
    
    # Get all numeric columns
    numeric_cols = df_pd.select_dtypes(include=[np.number]).columns.tolist()
    
    # Create summary statistics
    summary_data = []
    
    for col in numeric_cols:
        if col in ['position', 'window_start', 'window_end']:  # Skip positional columns
            continue
            
        data = df_pd[col].dropna()
        if len(data) == 0:
            continue
            
        # Get feature info if available
        feature_info = FEATURE_CATALOG.get(col, None)
        category = feature_info.category if feature_info else "Unknown"
        description = feature_info.description if feature_info else "No description available"
        
        summary_data.append({
            'Feature': col,
            'Category': category,
            'Description': description,
            'Count': len(data),
            'Missing': df_pd[col].isna().sum(),
            'Mean': data.mean(),
            'Std': data.std(),
            'Min': data.min(),
            'Max': data.max(),
            'Q25': data.quantile(0.25),
            'Q75': data.quantile(0.75)
        })
    
    # Create DataFrame and save
    summary_df = pd.DataFrame(summary_data)
    
    # Save as CSV
    csv_path = output_dir / "feature_summary_table.csv"
    summary_df.to_csv(csv_path, index=False)
    
    # Save as formatted markdown table
    md_path = output_dir / "feature_summary_table.md"
    with open(md_path, 'w') as f:
        f.write("# Feature Summary Table\n\n")
        f.write(summary_df.to_markdown(index=False, floatfmt=".4f"))
        f.write("\n")
    
    return str(csv_path)

def main():
    """Main function to generate feature analysis reports."""
    
    parser = argparse.ArgumentParser(description="Generate comprehensive feature analysis reports")
    parser.add_argument("--data-file", type=str, required=True,
                       help="Path to enhanced positions data file")
    parser.add_argument("--output-dir", type=str, default="results/probability_feature_analysis/reports",
                       help="Output directory for reports")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating feature analysis reports...")
    print(f"Data file: {args.data_file}")
    print(f"Output directory: {output_dir}")
    
    # Load data
    try:
        if args.data_file.endswith('.parquet'):
            df = pl.read_parquet(args.data_file)
        else:
            df = pl.read_csv(args.data_file, separator='\t')
        
        print(f"Loaded {df.height:,} positions with {len(df.columns)} features")
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Generate comprehensive report
    print("\n1. Generating comprehensive feature analysis report...")
    report_path = generate_feature_report(df, output_dir)
    print(f"   Saved: {report_path}")
    
    # Generate summary table
    print("\n2. Generating feature summary table...")
    table_path = create_summary_table(df, output_dir)
    print(f"   Saved: {table_path}")
    
    print(f"\n✅ Feature analysis complete!")
    print(f"Check the output directory: {output_dir}")

if __name__ == "__main__":
    main() 