#!/usr/bin/env python3
"""
Test Ablation Data Processing

This script replicates the ablation study data processing to identify
where the catastrophic performance drop is coming from.

Usage:
    python -m meta_spliceai.splice_engine.meta_models.analysis.test_ablation_data \
      --dataset train_pc_1000/master \
      --sample-size 10000
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

def load_and_process_data(dataset_dir: str, sample_size: int = 10000):
    """Load and process data exactly like the ablation study."""
    
    print(f"📊 Loading and Processing Data")
    print("=" * 50)
    
    # Load data
    data_path = Path(dataset_dir)
    parquet_files = list(data_path.glob("batch_*.parquet"))
    
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {dataset_dir}")
    
    print(f"Loading from {parquet_files[0]}")
    df = pd.read_parquet(parquet_files[0])
    
    print(f"Original shape: {df.shape}")
    print(f"Columns: {len(df.columns)}")
    
    # Sample data
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)
    print(f"Sample shape: {df.shape}")
    
    # Check splice_type distribution
    print(f"\n🎯 Target Variable Analysis:")
    splice_dist = df['splice_type'].value_counts()
    print(f"splice_type distribution:")
    for value, count in splice_dist.items():
        pct = count / len(df) * 100
        print(f"  {value}: {count:,} ({pct:.1f}%)")
    
    # Handle target encoding (support both "0" and "neither")
    print(f"\n🔄 Target Encoding:")
    
    def encode_splice_type(x):
        """Encode splice type with backward compatibility."""
        if x == 'donor':
            return 1
        elif x == 'acceptor':
            return 2
        elif x in ['0', 'neither', 0]:  # Support both formats
            return 0
        else:
            print(f"Warning: Unknown splice_type value: {x}")
            return 0
    
    df['target'] = df['splice_type'].apply(encode_splice_type)
    
    target_dist = df['target'].value_counts()
    print(f"Encoded target distribution:")
    for value, count in target_dist.items():
        pct = count / len(df) * 100
        label = {0: 'neither', 1: 'donor', 2: 'acceptor'}.get(value, f'unknown_{value}')
        print(f"  {value} ({label}): {count:,} ({pct:.1f}%)")
    
    # Check for data quality issues
    print(f"\n🔍 Data Quality Checks:")
    
    # Check for missing values
    missing_cols = df.columns[df.isnull().any()].tolist()
    if missing_cols:
        print(f"⚠️ Columns with missing values: {len(missing_cols)}")
        for col in missing_cols[:5]:  # Show first 5
            missing_pct = df[col].isnull().sum() / len(df) * 100
            print(f"  {col}: {missing_pct:.1f}% missing")
    else:
        print(f"✅ No missing values found")
    
    # Check for infinite values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    inf_cols = []
    for col in numeric_cols:
        if np.isinf(df[col]).any():
            inf_cols.append(col)
    
    if inf_cols:
        print(f"⚠️ Columns with infinite values: {len(inf_cols)}")
        for col in inf_cols[:5]:
            inf_count = np.isinf(df[col]).sum()
            print(f"  {col}: {inf_count} infinite values")
    else:
        print(f"✅ No infinite values found")
    
    return df

def test_feature_processing(df: pd.DataFrame):
    """Test feature processing like the ablation study."""
    
    print(f"\n🔧 Feature Processing Test")
    print("=" * 50)
    
    # Identify feature types
    prob_features = [col for col in df.columns if any(term in col.lower() 
                    for term in ['score', 'prob', 'signal', 'context'])]
    kmer_features = [col for col in df.columns if col.startswith('6mer_')]
    other_features = [col for col in df.columns if col not in prob_features + kmer_features 
                     and col not in ['splice_type', 'target']]
    
    print(f"Feature categories:")
    print(f"  Probability features: {len(prob_features)}")
    print(f"  K-mer features: {len(kmer_features)}")
    print(f"  Other features: {len(other_features)}")
    
    # Show examples
    print(f"\nExample probability features: {prob_features[:5]}")
    print(f"Example k-mer features: {kmer_features[:5]}")
    print(f"Example other features: {other_features[:5]}")
    
    # Test different feature combinations
    feature_sets = {
        'full': prob_features + kmer_features + other_features,
        'no_probs': kmer_features + other_features,
        'no_kmer': prob_features + other_features,
        'only_kmer': kmer_features
    }
    
    print(f"\nFeature set sizes:")
    for name, features in feature_sets.items():
        print(f"  {name}: {len(features)} features")
    
    return feature_sets

def test_model_training(df: pd.DataFrame, feature_sets: dict):
    """Test model training with different feature sets."""
    
    print(f"\n🤖 Model Training Test")
    print("=" * 50)
    
    # Prepare target
    y = df['target']
    
    print(f"Target distribution for training:")
    target_dist = y.value_counts()
    for value, count in target_dist.items():
        pct = count / len(y) * 100
        label = {0: 'neither', 1: 'donor', 2: 'acceptor'}.get(value, f'unknown_{value}')
        print(f"  {value} ({label}): {count:,} ({pct:.1f}%)")
    
    results = {}
    
    for set_name, feature_list in feature_sets.items():
        print(f"\n--- Testing {set_name} ({len(feature_list)} features) ---")
        
        if len(feature_list) == 0:
            print(f"⚠️ No features available for {set_name}")
            continue
        
        # Check if features exist in dataframe
        available_features = [f for f in feature_list if f in df.columns]
        missing_features = [f for f in feature_list if f not in df.columns]
        
        print(f"Available features: {len(available_features)}")
        if missing_features:
            print(f"Missing features: {len(missing_features)} (first 5: {missing_features[:5]})")
        
        if len(available_features) == 0:
            print(f"❌ No valid features found for {set_name}")
            continue
        
        # Prepare features
        X = df[available_features]
        
        # Handle data types
        print(f"Data types: {X.dtypes.value_counts().to_dict()}")
        
        # Identify and handle non-numeric columns
        non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
        if non_numeric_cols:
            print(f"⚠️ Non-numeric columns found: {len(non_numeric_cols)}")
            print(f"  Examples: {non_numeric_cols[:5]}")
            # Try to convert to numeric, drop if can't
            for col in non_numeric_cols:
                try:
                    X[col] = pd.to_numeric(X[col], errors='coerce')
                except:
                    print(f"  Dropping non-convertible column: {col}")
                    X = X.drop(columns=[col])
        
        # Now select only numeric columns
        X = X.select_dtypes(include=[np.number])
        print(f"Final numeric features: {X.shape[1]}")
        
        if X.shape[1] == 0:
            print(f"❌ No numeric features left after filtering")
            continue
        
        # Check for data issues
        if X.isnull().any().any():
            print(f"⚠️ Missing values found, filling with 0")
            X = X.fillna(0)
        
        # Check for infinite values (only on numeric data)
        if np.isinf(X.values).any():
            print(f"⚠️ Infinite values found, replacing with 0")
            X = X.replace([np.inf, -np.inf], 0)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Train/test split: {X_train.shape} / {X_test.shape}")
        
        # Train simple model
        try:
            # Use simple random forest for quick test
            clf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=1)
            clf.fit(X_train, y_train)
            
            # Predict
            y_pred = clf.predict(X_test)
            
            # Evaluate
            accuracy = accuracy_score(y_test, y_pred)
            f1_macro = f1_score(y_test, y_pred, average='macro')
            
            print(f"✅ Results:")
            print(f"  Accuracy: {accuracy:.3f}")
            print(f"  Macro F1: {f1_macro:.3f}")
            
            results[set_name] = {
                'accuracy': accuracy,
                'f1_macro': f1_macro,
                'n_features': len(available_features)
            }
            
            # Show class-wise performance
            print(f"  Classification report:")
            report = classification_report(y_test, y_pred, 
                                         target_names=['neither', 'donor', 'acceptor'],
                                         output_dict=True)
            for class_name in ['neither', 'donor', 'acceptor']:
                if class_name in report:
                    f1 = report[class_name]['f1-score']
                    print(f"    {class_name}: F1={f1:.3f}")
            
        except Exception as e:
            print(f"❌ Training failed: {e}")
            results[set_name] = {'error': str(e)}
    
    return results

def main():
    """Main testing function."""
    
    parser = argparse.ArgumentParser(description="Test ablation data processing")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--sample-size", type=int, default=10000)
    
    args = parser.parse_args()
    
    print("🧪 Ablation Data Processing Test")
    print("=" * 60)
    
    try:
        # Load and process data
        df = load_and_process_data(args.dataset, args.sample_size)
        
        # Test feature processing
        feature_sets = test_feature_processing(df)
        
        # Test model training
        results = test_model_training(df, feature_sets)
        
        # Summary
        print(f"\n📊 SUMMARY")
        print("=" * 50)
        
        print(f"Results comparison:")
        for set_name, result in results.items():
            if 'accuracy' in result:
                acc = result['accuracy']
                f1 = result['f1_macro']
                n_feat = result['n_features']
                print(f"  {set_name:10s}: Acc={acc:.3f}, F1={f1:.3f}, Features={n_feat}")
            else:
                print(f"  {set_name:10s}: ERROR - {result.get('error', 'Unknown')}")
        
        # Analysis
        print(f"\n🔍 Analysis:")
        if any('accuracy' in r for r in results.values()):
            best_acc = max([r['accuracy'] for r in results.values() if 'accuracy' in r])
            if best_acc > 0.8:
                print(f"✅ Good performance achieved ({best_acc:.3f}) - ablation script has a bug")
            elif best_acc > 0.5:
                print(f"⚠️ Moderate performance ({best_acc:.3f}) - data or feature issues")
            else:
                print(f"❌ Poor performance ({best_acc:.3f}) - major data/target issues")
        
        print(f"\n💡 Next steps:")
        print(f"1. If performance is good here, the issue is in the ablation script")
        print(f"2. If performance is bad here, check data preprocessing")
        print(f"3. Compare feature handling with successful gene CV")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 