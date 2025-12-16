#!/usr/bin/env python3
"""
Compare Feature Preprocessing Between Gene CV and Ablation Scripts

This script reproduces the exact preprocessing from run_gene_cv_sigmoid.py
to identify where the ablation script diverges.

Usage:
    python -m meta_spliceai.splice_engine.meta_models.analysis.compare_preprocessing \
      --dataset train_pc_1000/master \
      --sample-size 5000
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import os

def reproduce_gene_cv_preprocessing(dataset_path: str, sample_size: int = 5000):
    """Reproduce the exact preprocessing from run_gene_cv_sigmoid.py."""
    
    print("🧬 Reproducing Gene CV Preprocessing")
    print("=" * 60)
    
    # Step 1: Set environment variable (like gene CV does)
    if not os.getenv("SS_MAX_ROWS"):
        os.environ["SS_MAX_ROWS"] = str(sample_size)
    
    # Step 2: Load dataset (exactly like gene CV)
    try:
        from meta_spliceai.splice_engine.meta_models.training import datasets
        df = datasets.load_dataset(dataset_path)
        print(f"✅ Dataset loaded: {df.shape}")
        print(f"   Columns: {len(df.columns)}")
        print(f"   Data type: {type(df)}")
    except Exception as e:
        print(f"❌ Dataset loading failed: {e}")
        return None, None
    
    # Step 3: Prepare training data (exactly like gene CV) 
    try:
        from meta_spliceai.splice_engine.meta_models.builder import preprocessing
        
        X_df, y_series = preprocessing.prepare_training_data(
            df, 
            label_col="splice_type", 
            return_type="pandas", 
            verbose=1,
            preserve_transcript_columns=False,  # Set to False for simplicity
            encode_chrom=True  # Include encoded chromosome as a feature
        )
        
        print(f"✅ Training data prepared: X={X_df.shape}, y={len(y_series)}")
        print(f"   Feature data type: {type(X_df)}")
        print(f"   Feature dtypes: {X_df.dtypes.value_counts().to_dict()}")
        
    except Exception as e:
        print(f"❌ Training data preparation failed: {e}")
        return None, None
    
    # Step 4: Check for problematic columns (like gene CV would find)
    print(f"\n🔍 Data Quality Analysis:")
    
    # Check data types
    print(f"Column data types:")
    for dtype, count in X_df.dtypes.value_counts().items():
        print(f"  {dtype}: {count} columns")
    
    # Check for non-numeric columns  
    non_numeric_cols = X_df.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric_cols:
        print(f"⚠️ Non-numeric columns found: {len(non_numeric_cols)}")
        for col in non_numeric_cols[:10]:  # Show first 10
            print(f"  - {col}: {X_df[col].dtype}")
    else:
        print(f"✅ All columns are numeric")
    
    # Check for missing values
    missing_cols = X_df.columns[X_df.isnull().any()].tolist()
    if missing_cols:
        print(f"⚠️ Columns with missing values: {len(missing_cols)}")
        for col in missing_cols[:5]:
            missing_pct = X_df[col].isnull().sum() / len(X_df) * 100
            print(f"  - {col}: {missing_pct:.1f}% missing")
    else:
        print(f"✅ No missing values")
    
    # Step 5: Feature categorization (like ablation needs to do)
    print(f"\n📊 Feature Categorization:")
    
    feature_names = list(X_df.columns)
    
    # Identify feature types (same logic as ablation should use)
    prob_features = []
    kmer_features = []
    other_features = []
    
    for col in feature_names:
        if any(term in col.lower() for term in ['score', 'prob', 'signal', 'context']):
            prob_features.append(col)
        elif col.startswith('6mer_'):
            kmer_features.append(col)
        else:
            other_features.append(col)
    
    print(f"  Probability features: {len(prob_features)}")
    print(f"  K-mer features: {len(kmer_features)}")
    print(f"  Other features: {len(other_features)}")
    
    # Show examples
    print(f"\n📝 Feature Examples:")
    print(f"  Probability (first 5): {prob_features[:5]}")
    print(f"  K-mer (first 5): {kmer_features[:5]}")
    print(f"  Other (first 5): {other_features[:5]}")
    
    # Step 6: Test ablation modes
    print(f"\n🧪 Testing Ablation Feature Sets:")
    
    ablation_modes = {
        'full': prob_features + kmer_features + other_features,
        'no_probs': kmer_features + other_features,
        'no_kmer': prob_features + other_features,
        'only_kmer': kmer_features
    }
    
    for mode, features in ablation_modes.items():
        print(f"  {mode}: {len(features)} features")
        
        # Check if all features exist in X_df
        available = [f for f in features if f in X_df.columns]
        missing = [f for f in features if f not in X_df.columns]
        
        print(f"    Available: {len(available)}")
        if missing:
            print(f"    Missing: {len(missing)} (first 3: {missing[:3]})")
    
    # Step 7: Check target encoding
    print(f"\n🎯 Target Analysis:")
    print(f"  Target type: {type(y_series)}")
    print(f"  Target distribution:")
    
    if hasattr(y_series, 'value_counts'):
        for value, count in y_series.value_counts().items():
            pct = count / len(y_series) * 100
            print(f"    {value}: {count:,} ({pct:.1f}%)")
    
    return X_df, y_series

def test_simple_model(X_df, y_series):
    """Test that the processed data works with a simple model."""
    
    print(f"\n🤖 Testing Model Training:")
    
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, f1_score
        
        # Encode target like gene CV does
        from meta_spliceai.splice_engine.meta_models.training.classifier_utils import _encode_labels
        y_encoded = _encode_labels(y_series)
        
        print(f"  Encoded target distribution:")
        unique, counts = np.unique(y_encoded, return_counts=True)
        for val, count in zip(unique, counts):
            pct = count / len(y_encoded) * 100
            label = {0: 'neither', 1: 'donor', 2: 'acceptor'}.get(val, f'unknown_{val}')
            print(f"    {val} ({label}): {count:,} ({pct:.1f}%)")
        
        # Filter to numeric columns only
        numeric_cols = X_df.select_dtypes(include=[np.number]).columns
        X_numeric = X_df[numeric_cols]
        
        print(f"  Using {len(numeric_cols)} numeric features out of {len(X_df.columns)} total")
        
        # Handle missing/infinite values
        if X_numeric.isnull().any().any():
            print(f"  Filling missing values with 0")
            X_numeric = X_numeric.fillna(0)
        
        if np.isinf(X_numeric.values).any():
            print(f"  Replacing infinite values with 0")
            X_numeric = X_numeric.replace([np.inf, -np.inf], 0)
        
        # Split and train
        X_train, X_test, y_train, y_test = train_test_split(
            X_numeric, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        print(f"  Train/test split: {X_train.shape} / {X_test.shape}")
        
        # Train model
        clf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=1)
        clf.fit(X_train, y_train)
        
        # Evaluate
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average='macro')
        
        print(f"  ✅ Model training successful!")
        print(f"     Accuracy: {accuracy:.3f}")
        print(f"     Macro F1: {f1_macro:.3f}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Model training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main comparison function."""
    
    parser = argparse.ArgumentParser(description="Compare preprocessing pipelines")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--sample-size", type=int, default=5000)
    
    args = parser.parse_args()
    
    print("🔬 Feature Preprocessing Comparison")
    print("=" * 80)
    print(f"Dataset: {args.dataset}")
    print(f"Sample size: {args.sample_size}")
    print("=" * 80)
    
    # Reproduce gene CV preprocessing
    X_df, y_series = reproduce_gene_cv_preprocessing(args.dataset, args.sample_size)
    
    if X_df is not None and y_series is not None:
        # Test model training
        model_success = test_simple_model(X_df, y_series)
        
        print(f"\n" + "=" * 80)
        print(f"📋 SUMMARY")
        print("=" * 80)
        
        if model_success:
            print(f"✅ Gene CV preprocessing pipeline works correctly")
            print(f"✅ Data is suitable for model training")
            print(f"✅ The ablation script should use identical preprocessing")
            print(f"\n💡 The ablation script bug is likely:")
            print(f"   1. Different data loading or preprocessing")
            print(f"   2. Missing chromosome encoding (encode_chrom=True)")
            print(f"   3. Not filtering non-numeric columns properly")
            print(f"   4. Different feature exclusion logic")
        else:
            print(f"❌ Even gene CV preprocessing has issues")
            print(f"❌ Need to debug the preprocessing pipeline")
        
    else:
        print(f"\n❌ Could not reproduce gene CV preprocessing")
        print(f"❌ Check dataset path and dependencies")

if __name__ == "__main__":
    main() 