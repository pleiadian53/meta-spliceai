#!/usr/bin/env python3
"""
Demo script for the comprehensive feature importance analysis module.

This script demonstrates how to use the FeatureImportanceAnalyzer with 
gene-wise CV data to perform multi-method feature importance analysis.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split

# Import the new feature importance module
from meta_spliceai.splice_engine.meta_models.evaluation.feature_importance import (
    FeatureImportanceAnalyzer,
    analyze_feature_importance
)

def load_sample_data():
    """
    Load sample data for demonstration.
    In practice, you would load your actual gene-wise CV data.
    """
    # Create sample data with realistic features
    np.random.seed(42)
    n_samples = 1000
    
    # Create some realistic feature names
    feature_names = (
        # Numerical features
        ['score', 'num_exons', 'transcript_length', 'gc_content', 'conservation_score'] +
        # Categorical features  
        ['gene_type', 'strand', 'chrom_type'] +
        # K-mer features
        [f'{k}mer_{seq}' for k in [3, 4, 5] for seq in ['GT', 'AG', 'GC', 'AT', 'TA', 'CG']] +
        # Additional numerical features
        [f'feature_{i}' for i in range(20)]
    )
    
    # Generate realistic data
    data = {}
    
    # Numerical features
    data['score'] = np.random.beta(2, 1, n_samples)  # Biased towards higher values
    data['num_exons'] = np.random.poisson(8, n_samples) + 1
    data['transcript_length'] = np.random.lognormal(8, 1, n_samples)
    data['gc_content'] = np.random.normal(0.5, 0.1, n_samples)
    data['conservation_score'] = np.random.gamma(2, 2, n_samples)
    
    # Categorical features (encode as integers for XGBoost compatibility)
    gene_type_map = {'protein_coding': 0, 'lncRNA': 1, 'miRNA': 2}
    strand_map = {'+': 0, '-': 1}
    chrom_type_map = {'autosome': 0, 'X': 1, 'Y': 2}
    
    gene_types = np.random.choice(['protein_coding', 'lncRNA', 'miRNA'], n_samples, p=[0.7, 0.2, 0.1])
    strands = np.random.choice(['+', '-'], n_samples)
    chrom_types = np.random.choice(['autosome', 'X', 'Y'], n_samples, p=[0.9, 0.08, 0.02])
    
    data['gene_type'] = [gene_type_map[x] for x in gene_types]
    data['strand'] = [strand_map[x] for x in strands]
    data['chrom_type'] = [chrom_type_map[x] for x in chrom_types]
    
    # K-mer features (count data)
    for feature in feature_names:
        if feature.endswith('mer_GT') or feature.endswith('mer_AG'):
            # These should be higher for true splice sites
            data[feature] = np.random.poisson(3, n_samples)
        elif 'mer_' in feature:
            data[feature] = np.random.poisson(1, n_samples)
    
    # Additional numerical features
    for i in range(20):
        data[f'feature_{i}'] = np.random.normal(0, 1, n_samples)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Create target variable (splice site vs non-splice site)
    # Make it correlated with some features
    y_prob = (
        0.3 * (df['score'] > 0.7) +
        0.2 * (df['num_exons'] > 10) +
        0.2 * (df['3mer_GT'] > 2) +
        0.1 * (df['4mer_GT'] > 2) +
        0.2 * np.random.random(n_samples)
    )
    y = (y_prob > 0.5).astype(int)
    
    return df, y

def create_mock_model(X, y):
    """
    Create a mock XGBoost model for demonstration.
    """
    try:
        import xgboost as xgb
        
        # Create and train XGBoost model
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            eval_metric='logloss'
        )
        
        # Fit the model
        model.fit(X, y)
        
        return model
    
    except ImportError:
        print("XGBoost not available. Creating mock model...")
        # Create a mock model object that has the required methods
        class MockModel:
            def __init__(self):
                self.feature_names = X.columns.tolist()
                # Create random importance scores
                np.random.seed(42)
                self.importance_scores = {
                    'weight': {name: np.random.random() for name in self.feature_names},
                    'gain': {name: np.random.random() for name in self.feature_names},
                    'cover': {name: np.random.random() for name in self.feature_names},
                    'total_gain': {name: np.random.random() for name in self.feature_names},
                    'total_cover': {name: np.random.random() for name in self.feature_names},
                }
            
            def get_booster(self):
                return self
            
            def get_score(self, importance_type='weight'):
                return self.importance_scores.get(importance_type, self.importance_scores['weight'])
        
        return MockModel()

def demo_basic_usage():
    """
    Demonstrate basic usage of the feature importance analyzer.
    """
    print("=== Feature Importance Analysis Demo ===")
    print("Loading sample data...")
    
    # Load sample data
    X, y = load_sample_data()
    print(f"Data shape: {X.shape}, Target distribution: {y.sum()}/{len(y)}")
    
    # Create model
    print("Creating and training model...")
    model = create_mock_model(X, y)
    
    # Initialize analyzer
    output_dir = "demo_feature_importance_results"
    analyzer = FeatureImportanceAnalyzer(
        output_dir=output_dir,
        subject="demo_analysis"
    )
    
    # Run comprehensive analysis
    print("Running comprehensive feature importance analysis...")
    results = analyzer.run_comprehensive_analysis(
        model=model,
        X=X,
        y=y,
        top_k=15,
        methods=['xgboost', 'hypothesis_testing', 'effect_sizes', 'mutual_info'],
        verbose=1
    )
    
    # Save results
    analyzer.save_results()
    
    print(f"\nAnalysis complete! Results saved to: {output_dir}")
    print(f"Generated {len(os.listdir(output_dir))} files")
    
    # Print summary of results
    print("\n=== Analysis Summary ===")
    for method_name, method_results in results.items():
        if method_name == 'xgboost':
            print(f"\n{method_name.upper()}:")
            for imp_type, imp_results in method_results.items():
                top_features = imp_results['top_k']['feature'].head(5).tolist()
                print(f"  {imp_type}: {top_features}")
        else:
            top_features = method_results['top_k']['feature'].head(5).tolist()
            print(f"\n{method_name.upper()}: {top_features}")
    
    return results

def demo_custom_analysis():
    """
    Demonstrate custom analysis with specific methods.
    """
    print("\n=== Custom Feature Importance Analysis Demo ===")
    
    # Load data
    X, y = load_sample_data()
    model = create_mock_model(X, y)
    
    # Run analysis with only specific methods
    print("Running analysis with hypothesis testing and effect sizes only...")
    
    results = analyze_feature_importance(
        model=model,
        X=X,
        y=y,
        output_dir="demo_custom_analysis",
        subject="custom_demo",
        top_k=10,
        methods=['hypothesis_testing', 'effect_sizes'],
        verbose=1
    )
    
    print("Custom analysis complete!")
    return results

def demo_with_real_data():
    """
    Demonstrate with actual gene-wise CV data if available.
    """
    print("\n=== Real Data Demo ===")
    
    # Try to load real data
    try:
        # This would be the path to your actual gene-wise CV results
        # model_path = "path/to/your/model.pkl"
        # data_path = "path/to/your/processed_data.csv"
        
        # model = joblib.load(model_path)
        # data = pd.read_csv(data_path)
        # X = data.drop(['target'], axis=1)
        # y = data['target']
        
        print("Real data not available for demo. Using sample data instead.")
        return demo_basic_usage()
        
    except Exception as e:
        print(f"Could not load real data: {e}")
        print("Using sample data instead.")
        return demo_basic_usage()

def main():
    """
    Main demo function.
    """
    print("Feature Importance Analysis Demo")
    print("=" * 50)
    
    # Run basic demo
    results1 = demo_basic_usage()
    
    # Run custom demo
    results2 = demo_custom_analysis()
    
    # Try real data demo
    # results3 = demo_with_real_data()
    
    print("\n" + "=" * 50)
    print("Demo complete!")
    print("\nCheck the output directories for generated plots and results:")
    print("- demo_feature_importance_results/")
    print("- demo_custom_analysis/")
    
    print("\nGenerated files include:")
    print("- XGBoost importance plots (multiple types)")
    print("- Hypothesis testing results")
    print("- Effect size analysis")
    print("- Mutual information analysis")
    print("- Method comparison plots")
    print("- Comprehensive results Excel file")

if __name__ == "__main__":
    main() 