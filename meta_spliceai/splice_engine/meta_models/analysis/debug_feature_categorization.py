#!/usr/bin/env python3
"""
Debug Feature Categorization in Ablation Study

This script investigates exactly which features are being categorized
in each ablation mode to understand why 'no_probs' still performs well.
"""

import pandas as pd
import polars as pl
from pathlib import Path

def debug_feature_categorization():
    """Debug the feature categorization logic from the ablation study."""
    
    print("🔍 Debugging Feature Categorization")
    print("=" * 60)
    
    # Load the hierarchical balanced dataset
    balanced_data_path = "results/ablation_study_fixed/hierarchical_balanced_dataset.parquet"
    
    if not Path(balanced_data_path).exists():
        print(f"❌ Balanced dataset not found at: {balanced_data_path}")
        print("Please run the fixed ablation study first")
        return
    
    # Load and preprocess data (same as ablation script)
    print("Loading balanced dataset...")
    df = pd.read_parquet(balanced_data_path)
    print(f"Dataset shape: {df.shape}")
    
    # Convert to polars and preprocess
    df_polars = pl.from_pandas(df)
    
    from meta_spliceai.splice_engine.meta_models.builder import preprocessing
    
    X_df, y_series = preprocessing.prepare_training_data(
        df_polars, 
        label_col="splice_type", 
        return_type="pandas", 
        verbose=0,  # Reduce verbosity
        preserve_transcript_columns=False,
        encode_chrom=True
    )
    
    print(f"Preprocessed data: X={X_df.shape}, y={len(y_series)}")
    
    # Apply the EXACT same categorization logic as ablation script
    feature_names = list(X_df.columns)
    
    # Probability/signal features (EXACT same logic)
    prob_features = [col for col in feature_names if any(term in col.lower() 
                    for term in ['score', 'prob', 'signal', 'context', 'peak', 'difference'])]
    
    # K-mer features
    kmer_features = [col for col in feature_names if col.startswith('6mer_')]
    
    # Other features
    other_features = [col for col in feature_names if col not in prob_features + kmer_features]
    
    print(f"\n📊 Feature Categorization Results:")
    print(f"  Total features: {len(feature_names):,}")
    print(f"  Probability/signal features: {len(prob_features):,}")
    print(f"  K-mer features: {len(kmer_features):,}")
    print(f"  Other features: {len(other_features):,}")
    
    # Show detailed breakdown
    print(f"\n🔍 Detailed Probability Feature Analysis:")
    
    # Group probability features by pattern
    score_features = [f for f in prob_features if 'score' in f.lower()]
    prob_only_features = [f for f in prob_features if 'prob' in f.lower() and 'score' not in f.lower()]
    signal_features = [f for f in prob_features if 'signal' in f.lower()]
    context_features = [f for f in prob_features if 'context' in f.lower()]
    peak_features = [f for f in prob_features if 'peak' in f.lower()]
    diff_features = [f for f in prob_features if 'difference' in f.lower()]
    
    print(f"  'score' features ({len(score_features)}):")
    for f in sorted(score_features)[:10]:  # Show first 10
        print(f"    {f}")
    if len(score_features) > 10:
        print(f"    ... and {len(score_features) - 10} more")
    
    print(f"\n  'prob' features ({len(prob_only_features)}):")
    for f in sorted(prob_only_features):
        print(f"    {f}")
    
    print(f"\n  'signal' features ({len(signal_features)}):")
    for f in sorted(signal_features):
        print(f"    {f}")
    
    print(f"\n  'context' features ({len(context_features)}):")
    for f in sorted(context_features):
        print(f"    {f}")
    
    print(f"\n  'peak' features ({len(peak_features)}):")
    for f in sorted(peak_features):
        print(f"    {f}")
    
    print(f"\n  'difference' features ({len(diff_features)}):")
    for f in sorted(diff_features):
        print(f"    {f}")
    
    # Check critical SpliceAI features
    print(f"\n🚨 Critical SpliceAI Feature Check:")
    
    critical_features = [
        'donor_score', 'acceptor_score', 'neither_score',
        'splice_probability', 'relative_donor_probability'
    ]
    
    for feat in critical_features:
        if feat in prob_features:
            print(f"  ✅ {feat}: In probability features")
        elif feat in other_features:
            print(f"  ⚠️ {feat}: In OTHER features (BUG!)")
        elif feat in kmer_features:
            print(f"  ❌ {feat}: In k-mer features (WRONG!)")
        else:
            print(f"  ❓ {feat}: NOT FOUND in dataset")
    
    # Show what's in "other" features
    print(f"\n📋 'Other' Features ({len(other_features)}):")
    for f in sorted(other_features):
        print(f"    {f}")
    
    # Test ablation modes
    print(f"\n🧪 Ablation Mode Feature Counts:")
    
    ablation_modes = {
        'full': prob_features + kmer_features + other_features,
        'no_probs': kmer_features + other_features,
        'no_kmer': prob_features + other_features,
        'only_kmer': kmer_features
    }
    
    for mode, features in ablation_modes.items():
        print(f"  {mode}: {len(features):,} features")
        
        # Check if key SpliceAI features are included
        has_donor_score = 'donor_score' in features
        has_acceptor_score = 'acceptor_score' in features
        has_splice_prob = 'splice_probability' in features
        
        print(f"    Has donor_score: {has_donor_score}")
        print(f"    Has acceptor_score: {has_acceptor_score}")
        print(f"    Has splice_probability: {has_splice_prob}")
        
        if mode == 'no_probs' and (has_donor_score or has_acceptor_score):
            print(f"    🚨 BUG: 'no_probs' still has SpliceAI scores!")
    
    # Return feature lists for further analysis
    return {
        'prob_features': prob_features,
        'kmer_features': kmer_features,
        'other_features': other_features,
        'ablation_modes': ablation_modes
    }

def test_realistic_class_balance():
    """Test ablation with realistic class balance (90% neither)."""
    
    print(f"\n" + "🔄" * 60)
    print("Testing with Realistic Class Balance (90% neither)")
    print("🔄" * 60)
    
    # Load the original balanced dataset
    balanced_data_path = "results/ablation_study_fixed/hierarchical_balanced_dataset.parquet"
    df = pd.read_parquet(balanced_data_path)
    
    print(f"Original balanced dataset:")
    dist = df['splice_type'].value_counts()
    for value, count in dist.items():
        pct = count / len(df) * 100
        print(f"  {value}: {count:,} ({pct:.1f}%)")
    
    # Create realistic imbalanced dataset (90% neither)
    print(f"\n🎯 Creating realistic imbalanced dataset (90% neither)...")
    
    # Separate by type
    donor_df = df[df['splice_type'] == 'donor']
    acceptor_df = df[df['splice_type'] == 'acceptor']
    neither_df = df[df['splice_type'] == '0']
    
    # Keep fewer splice sites, sample more neither sites
    n_splice_sites = min(1000, len(donor_df) + len(acceptor_df))  # Smaller sample
    n_donor = min(500, len(donor_df))
    n_acceptor = min(500, len(acceptor_df))
    n_neither = int(n_splice_sites * 9)  # 90% neither
    
    print(f"Target composition:")
    print(f"  Donors: {n_donor:,}")
    print(f"  Acceptors: {n_acceptor:,}")
    print(f"  Neither: {n_neither:,}")
    print(f"  Total: {n_donor + n_acceptor + n_neither:,}")
    
    # Sample data
    donor_sample = donor_df.sample(n=n_donor, random_state=42)
    acceptor_sample = acceptor_df.sample(n=n_acceptor, random_state=42)
    neither_sample = neither_df.sample(n=min(n_neither, len(neither_df)), random_state=42)
    
    # Combine
    realistic_df = pd.concat([donor_sample, acceptor_sample, neither_sample], ignore_index=True)
    realistic_df = realistic_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"\nRealistic dataset:")
    realistic_dist = realistic_df['splice_type'].value_counts()
    for value, count in realistic_dist.items():
        pct = count / len(realistic_df) * 100
        print(f"  {value}: {count:,} ({pct:.1f}%)")
    
    # Save realistic dataset
    realistic_path = "results/ablation_study_fixed/realistic_balanced_dataset.parquet"
    realistic_df.to_parquet(realistic_path, index=False)
    print(f"💾 Saved realistic dataset to: {realistic_path}")
    
    # Test ablation modes on realistic data
    print(f"\n🧪 Testing Ablation Modes on Realistic Data:")
    
    # Convert to polars and preprocess
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
    
    # Feature categorization
    feature_names = list(X_df.columns)
    prob_features = [col for col in feature_names if any(term in col.lower() 
                    for term in ['score', 'prob', 'signal', 'context', 'peak', 'difference'])]
    kmer_features = [col for col in feature_names if col.startswith('6mer_')]
    other_features = [col for col in feature_names if col not in prob_features + kmer_features]
    
    ablation_modes = {
        'full': prob_features + kmer_features + other_features,
        'no_probs': kmer_features + other_features,
        'only_kmer': kmer_features
    }
    
    # Quick test with Random Forest
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, f1_score
    from meta_spliceai.splice_engine.meta_models.training.classifier_utils import _encode_labels
    
    y_encoded = _encode_labels(y_series)
    
    results = {}
    
    for mode, selected_features in ablation_modes.items():
        print(f"\n--- Testing {mode} ({len(selected_features)} features) ---")
        
        if len(selected_features) == 0:
            continue
        
        X_mode = X_df[selected_features]
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_mode, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Train model
        clf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=1)
        clf.fit(X_train, y_train)
        
        # Evaluate
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average='macro')
        
        results[mode] = {'accuracy': accuracy, 'f1_macro': f1_macro}
        
        print(f"✅ {mode}: Acc={accuracy:.3f}, F1={f1_macro:.3f}")
    
    print(f"\n📊 Realistic Data Results Summary:")
    print(f"{'Mode':<12} {'Accuracy':<10} {'F1 Macro':<10}")
    print("-" * 35)
    for mode, result in results.items():
        acc = result['accuracy']
        f1 = result['f1_macro']
        print(f"{mode:<12} {acc:.3f}      {f1:.3f}")
    
    print(f"\n💡 Expected behavior:")
    print(f"  - If 'no_probs' still performs well, there's a categorization bug")
    print(f"  - If 'only_kmer' performs well on realistic data, that's suspicious")
    print(f"  - We should see much lower performance on 90% neither data")

def main():
    """Main function."""
    
    print("🔍 Feature Categorization Debug Analysis")
    print("=" * 80)
    
    # Debug feature categorization
    feature_info = debug_feature_categorization()
    
    # Test with realistic class balance
    test_realistic_class_balance()
    
    print(f"\n" + "=" * 80)
    print("🔍 DIAGNOSIS COMPLETE")
    print("=" * 80)
    
    print(f"🎯 Key Questions to Answer:")
    print(f"  1. Are critical SpliceAI features (donor_score, acceptor_score) properly excluded from 'no_probs'?")
    print(f"  2. Does 'no_probs' still perform well on realistic data (90% neither)?")
    print(f"  3. What features in 'other' category might be doing the heavy lifting?")
    
    print(f"\n💡 If 'no_probs' still performs well on realistic data:")
    print(f"  - There's likely a feature categorization bug")
    print(f"  - Or genomic position features are more powerful than expected")
    print(f"  - Or k-mer features alone are surprisingly effective")

if __name__ == "__main__":
    main() 