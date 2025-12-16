#!/usr/bin/env python3
"""
Test Realistic Ablation Study

This script tests the ablation study with:
1. Realistic class distribution (90% neither, 10% splice sites)
2. Proper feature categorization based on user's feature list
3. Verification that SpliceAI features are truly essential
"""

import pandas as pd
import polars as pl
import numpy as np
from pathlib import Path

def create_realistic_dataset(source_path: str, target_size: int = 10000):
    """Create a realistic dataset with 90% neither, 10% splice sites."""
    
    print("🔍 Creating Realistic Dataset (90% neither)")
    print("=" * 60)
    
    # Load source data
    df = pd.read_parquet(source_path)
    print(f"Source dataset: {df.shape}")
    
    dist = df['splice_type'].value_counts()
    print(f"Source distribution:")
    for value, count in dist.items():
        pct = count / len(df) * 100
        print(f"  {value}: {count:,} ({pct:.1f}%)")
    
    # Separate by type
    donor_df = df[df['splice_type'] == 'donor']
    acceptor_df = df[df['splice_type'] == 'acceptor']
    neither_df = df[df['splice_type'] == '0']
    
    # Realistic proportions: 90% neither, 5% donor, 5% acceptor
    n_total = target_size
    n_neither = int(n_total * 0.90)
    n_donor = int(n_total * 0.05)
    n_acceptor = n_total - n_neither - n_donor  # Remainder
    
    print(f"\nTarget realistic distribution:")
    print(f"  Neither: {n_neither:,} (90.0%)")
    print(f"  Donor: {n_donor:,} ({n_donor/n_total*100:.1f}%)")
    print(f"  Acceptor: {n_acceptor:,} ({n_acceptor/n_total*100:.1f}%)")
    
    # Sample data
    donor_sample = donor_df.sample(n=min(n_donor, len(donor_df)), random_state=42)
    acceptor_sample = acceptor_df.sample(n=min(n_acceptor, len(acceptor_df)), random_state=42)
    neither_sample = neither_df.sample(n=min(n_neither, len(neither_df)), random_state=42)
    
    # Combine and shuffle
    realistic_df = pd.concat([donor_sample, acceptor_sample, neither_sample], ignore_index=True)
    realistic_df = realistic_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"\nActual realistic dataset:")
    final_dist = realistic_df['splice_type'].value_counts()
    for value, count in final_dist.items():
        pct = count / len(realistic_df) * 100
        print(f"  {value}: {count:,} ({pct:.1f}%)")
    
    return realistic_df

def categorize_features_properly(feature_names, user_prob_features):
    """
    Categorize features based on the user's explicit list of probability features.
    
    Args:
        feature_names: All feature names from preprocessing
        user_prob_features: User's explicit list of probability features
    
    Returns:
        Dict with proper feature categorization
    """
    
    print("🔧 Proper Feature Categorization")
    print("=" * 50)
    
    # Parse user's probability features
    user_prob_set = set(user_prob_features)
    
    # Categorize based on user's explicit list
    prob_features = [f for f in feature_names if f in user_prob_set]
    kmer_features = [f for f in feature_names if f.startswith('6mer_')]
    other_features = [f for f in feature_names if f not in prob_features and f not in kmer_features]
    
    print(f"Feature categorization based on user's list:")
    print(f"  Probability features: {len(prob_features):,}")
    print(f"  K-mer features: {len(kmer_features):,}")
    print(f"  Other features: {len(other_features):,}")
    
    # Critical check: Are SpliceAI raw scores properly categorized?
    critical_spliceai = ['donor_score', 'acceptor_score', 'neither_score']
    
    print(f"\n🚨 Critical SpliceAI Features Check:")
    for feat in critical_spliceai:
        if feat in prob_features:
            print(f"  ✅ {feat}: Correctly in probability features")
        elif feat in other_features:
            print(f"  ❌ {feat}: WRONGLY in other features!")
        elif feat in kmer_features:
            print(f"  ❌ {feat}: WRONGLY in k-mer features!")
        else:
            print(f"  ❓ {feat}: NOT FOUND in dataset")
    
    # Show examples of each category
    print(f"\n📋 Feature Examples:")
    print(f"  Probability (first 10): {prob_features[:10]}")
    print(f"  K-mer (first 5): {kmer_features[:5]}")
    print(f"  Other (all): {other_features}")
    
    return {
        'prob_features': prob_features,
        'kmer_features': kmer_features,
        'other_features': other_features
    }

def test_ablation_on_realistic_data(df, feature_categories):
    """Test ablation modes on realistic data with proper feature categorization."""
    
    print(f"\n🧪 Testing Ablation on Realistic Data")
    print("=" * 60)
    
    # Preprocess data
    df_polars = pl.from_pandas(df)
    
    from meta_spliceai.splice_engine.meta_models.builder import preprocessing
    
    X_df, y_series = preprocessing.prepare_training_data(
        df_polars, 
        label_col="splice_type", 
        return_type="pandas", 
        verbose=0,
        preserve_transcript_columns=False,
        encode_chrom=True
    )
    
    print(f"Preprocessed data: X={X_df.shape}, y={len(y_series)}")
    
    # Check target distribution
    print(f"Target distribution:")
    target_dist = y_series.value_counts()
    for value, count in target_dist.items():
        pct = count / len(y_series) * 100
        print(f"  {value}: {count:,} ({pct:.1f}%)")
    
    # Use proper feature categorization
    prob_features = feature_categories['prob_features']
    kmer_features = feature_categories['kmer_features']
    other_features = feature_categories['other_features']
    
    # Define ablation modes with proper categorization
    ablation_modes = {
        'full': prob_features + kmer_features + other_features,
        'no_probs': kmer_features + other_features,  # Excludes ALL SpliceAI features
        'no_kmer': prob_features + other_features,   # Excludes sequence features
        'only_kmer': kmer_features,                  # Only sequence features
        'only_probs': prob_features                  # Only SpliceAI features
    }
    
    print(f"\n🎯 Ablation Mode Definitions:")
    for mode, features in ablation_modes.items():
        # Check available features
        available = [f for f in features if f in X_df.columns]
        
        # Check for critical SpliceAI features
        has_donor_score = 'donor_score' in available
        has_acceptor_score = 'acceptor_score' in available
        has_splice_prob = 'splice_probability' in available
        
        print(f"  {mode}: {len(available)} features available")
        print(f"    SpliceAI scores: donor={has_donor_score}, acceptor={has_acceptor_score}, splice_prob={has_splice_prob}")
        
        if mode == 'no_probs' and (has_donor_score or has_acceptor_score):
            print(f"    🚨 BUG: no_probs still has SpliceAI scores!")
    
    # Train and evaluate models
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, f1_score, classification_report
    from meta_spliceai.splice_engine.meta_models.training.classifier_utils import _encode_labels
    
    y_encoded = _encode_labels(y_series)
    
    results = {}
    
    for mode, feature_list in ablation_modes.items():
        print(f"\n--- Testing {mode} ---")
        
        # Filter to available features
        available_features = [f for f in feature_list if f in X_df.columns]
        
        if len(available_features) == 0:
            print(f"❌ No features available for {mode}")
            continue
        
        print(f"Using {len(available_features)} features")
        
        X_mode = X_df[available_features]
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_mode, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Train model
        clf = RandomForestClassifier(
            n_estimators=100, 
            random_state=42, 
            n_jobs=1,
            class_weight='balanced'  # Handle class imbalance
        )
        clf.fit(X_train, y_train)
        
        # Evaluate
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average='macro')
        f1_weighted = f1_score(y_test, y_pred, average='weighted')
        
        results[mode] = {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'n_features': len(available_features)
        }
        
        print(f"✅ Results: Acc={accuracy:.3f}, F1_macro={f1_macro:.3f}, F1_weighted={f1_weighted:.3f}")
        
        # Show classification report for critical modes
        if mode in ['no_probs', 'only_probs']:
            print(f"Classification report for {mode}:")
            print(classification_report(y_test, y_pred))
    
    return results

def main():
    """Main function to test realistic ablation study."""
    
    print("🧪 Realistic Ablation Study Test")
    print("=" * 80)
    
    # User's explicit probability feature list
    user_prob_features = [
        'acceptor_context_diff_ratio', 'acceptor_diff_m1', 'acceptor_diff_m2', 
        'acceptor_diff_p1', 'acceptor_diff_p2', 'acceptor_is_local_peak', 
        'acceptor_peak_height_ratio', 'acceptor_score', 'acceptor_second_derivative',
        'acceptor_signal_strength', 'acceptor_surge_ratio', 'acceptor_weighted_context',
        'context_asymmetry', 'context_max', 'context_neighbor_mean', 
        'context_score_m1', 'context_score_m2', 'context_score_p1', 'context_score_p2',
        'donor_acceptor_diff', 'donor_acceptor_logodds', 'donor_acceptor_peak_ratio',
        'donor_context_diff_ratio', 'donor_diff_m1', 'donor_diff_m2', 'donor_diff_p1', 
        'donor_diff_p2', 'donor_is_local_peak', 'donor_peak_height_ratio', 'donor_score',
        'donor_second_derivative', 'donor_signal_strength', 'donor_surge_ratio', 
        'donor_weighted_context', 'neither_score', 'probability_entropy', 
        'relative_donor_probability', 'score_difference_ratio', 'signal_strength_ratio',
        'splice_neither_diff', 'splice_neither_logodds', 'splice_probability', 
        'type_signal_difference'
    ]
    
    print(f"User provided {len(user_prob_features)} probability features")
    
    # Load hierarchical balanced dataset
    source_path = "results/ablation_study_fixed/hierarchical_balanced_dataset.parquet"
    
    if not Path(source_path).exists():
        print(f"❌ Source dataset not found: {source_path}")
        print("Please run the fixed ablation study first")
        return
    
    # Create realistic dataset
    realistic_df = create_realistic_dataset(source_path, target_size=10000)
    
    # Save realistic dataset
    realistic_path = "results/ablation_study_fixed/realistic_dataset.parquet"
    Path(realistic_path).parent.mkdir(parents=True, exist_ok=True)
    realistic_df.to_parquet(realistic_path, index=False)
    print(f"💾 Saved realistic dataset: {realistic_path}")
    
    # Preprocess to get feature names
    df_polars = pl.from_pandas(realistic_df)
    from meta_spliceai.splice_engine.meta_models.builder import preprocessing
    
    X_df, y_series = preprocessing.prepare_training_data(
        df_polars, 
        label_col="splice_type", 
        return_type="pandas", 
        verbose=0,
        preserve_transcript_columns=False,
        encode_chrom=True
    )
    
    # Proper feature categorization
    feature_categories = categorize_features_properly(
        list(X_df.columns), 
        user_prob_features
    )
    
    # Test ablation on realistic data
    results = test_ablation_on_realistic_data(realistic_df, feature_categories)
    
    print(f"\n" + "=" * 80)
    print("📊 REALISTIC ABLATION RESULTS")
    print("=" * 80)
    
    print(f"{'Mode':<12} {'Accuracy':<10} {'F1 Macro':<10} {'F1 Weighted':<12} {'Features':<10}")
    print("-" * 65)
    
    for mode, result in results.items():
        acc = result['accuracy']
        f1_macro = result['f1_macro']
        f1_weighted = result['f1_weighted']
        n_feat = result['n_features']
        print(f"{mode:<12} {acc:.3f}      {f1_macro:.3f}      {f1_weighted:.3f}        {n_feat:<10}")
    
    print(f"\n💡 Expected Biologically Sensible Results:")
    print(f"  🎯 'full' should perform best (~95%+ accuracy)")
    print(f"  🚨 'no_probs' should perform MUCH worse (~60-70% accuracy)")
    print(f"  📊 'only_probs' should perform well (~90%+ accuracy)")
    print(f"  🧬 'only_kmer' should perform moderately (~70-80% accuracy)")
    
    print(f"\n🔍 Diagnostic Insights:")
    if 'no_probs' in results and 'full' in results:
        no_probs_acc = results['no_probs']['accuracy']
        full_acc = results['full']['accuracy']
        
        if abs(no_probs_acc - full_acc) < 0.1:
            print(f"  🚨 PROBLEM: no_probs ({no_probs_acc:.3f}) ≈ full ({full_acc:.3f})")
            print(f"      This suggests SpliceAI features aren't essential - biologically impossible!")
        else:
            print(f"  ✅ GOOD: no_probs ({no_probs_acc:.3f}) << full ({full_acc:.3f})")
            print(f"      This confirms SpliceAI features are essential")

if __name__ == "__main__":
    main() 