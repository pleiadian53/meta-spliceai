#!/usr/bin/env python3
"""
Realistic Ablation Study with Proper Metrics for Imbalanced Data

Focus on F1 and Average Precision instead of accuracy.
"""

import pandas as pd
import polars as pl
import numpy as np
from pathlib import Path

def test_ablation_with_proper_metrics(df, feature_categories):
    """Test ablation modes focusing on F1 and AP metrics for imbalanced data."""
    
    print(f"\n🧪 Ablation Study with Proper Imbalanced Data Metrics")
    print("=" * 70)
    
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
    print(f"Target distribution (why accuracy is useless):")
    target_dist = y_series.value_counts()
    for value, count in target_dist.items():
        pct = count / len(y_series) * 100
        class_name = {0: 'neither', 1: 'donor', 2: 'acceptor'}.get(value, f'class_{value}')
        print(f"  {value} ({class_name}): {count:,} ({pct:.1f}%)")
    
    majority_baseline = target_dist.max() / len(y_series)
    print(f"\n⚠️ Majority class baseline accuracy: {majority_baseline:.1%}")
    print(f"   → This is why accuracy is meaningless!")
    
    # Use proper feature categorization
    prob_features = feature_categories['prob_features']
    kmer_features = feature_categories['kmer_features']
    other_features = feature_categories['other_features']
    
    # Define ablation modes
    ablation_modes = {
        'full': prob_features + kmer_features + other_features,
        'no_probs': kmer_features + other_features,  # NO SpliceAI
        'no_kmer': prob_features + other_features,   # NO sequence
        'only_probs': prob_features,                 # ONLY SpliceAI
        'only_kmer': kmer_features                   # ONLY sequence
    }
    
    # Train and evaluate models with proper metrics
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (
        f1_score, average_precision_score, precision_recall_curve,
        classification_report, confusion_matrix
    )
    from sklearn.preprocessing import LabelBinarizer
    from meta_spliceai.splice_engine.meta_models.training.classifier_utils import _encode_labels
    
    y_encoded = _encode_labels(y_series)
    
    results = {}
    
    for mode, feature_list in ablation_modes.items():
        print(f"\n" + "="*50)
        print(f"🔬 Testing {mode.upper()}")
        print("="*50)
        
        # Filter to available features
        available_features = [f for f in feature_list if f in X_df.columns]
        
        if len(available_features) == 0:
            print(f"❌ No features available for {mode}")
            continue
        
        print(f"Using {len(available_features)} features")
        
        # Check for critical SpliceAI features
        has_donor_score = 'donor_score' in available_features
        has_acceptor_score = 'acceptor_score' in available_features
        has_splice_prob = 'splice_probability' in available_features
        
        print(f"SpliceAI features: donor={has_donor_score}, acceptor={has_acceptor_score}, splice_prob={has_splice_prob}")
        
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
            class_weight='balanced'  # Critical for imbalanced data
        )
        clf.fit(X_train, y_train)
        
        # Get predictions and probabilities
        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)
        
        # Calculate meaningful metrics for imbalanced data
        f1_macro = f1_score(y_test, y_pred, average='macro')
        f1_weighted = f1_score(y_test, y_pred, average='weighted')
        
        # Per-class F1 scores
        f1_per_class = f1_score(y_test, y_pred, average=None)
        
        # Average Precision (AP) for each class
        lb = LabelBinarizer()
        y_test_bin = lb.fit_transform(y_test)
        y_pred_proba_reordered = y_pred_proba[:, lb.classes_]
        
        ap_scores = []
        for i in range(len(lb.classes_)):
            if len(np.unique(y_test_bin[:, i])) > 1:  # Need both classes present
                ap = average_precision_score(y_test_bin[:, i], y_pred_proba_reordered[:, i])
                ap_scores.append(ap)
            else:
                ap_scores.append(np.nan)
        
        ap_macro = np.nanmean(ap_scores)
        
        results[mode] = {
            'f1_macro': float(f1_macro),
            'f1_weighted': float(f1_weighted),
            'f1_per_class': [float(f) for f in f1_per_class],
            'ap_macro': float(ap_macro),
            'ap_per_class': [float(ap) if not np.isnan(ap) else None for ap in ap_scores],
            'n_features': len(available_features)
        }
        
        print(f"\n📊 MEANINGFUL METRICS (no accuracy!):")
        print(f"  F1 Macro: {f1_macro:.3f}")
        print(f"  F1 Weighted: {f1_weighted:.3f}")
        print(f"  AP Macro: {ap_macro:.3f}")
        
        print(f"\n📋 Per-Class Performance:")
        class_names = ['neither', 'donor', 'acceptor']
        for i, (class_name, f1_val, ap_val) in enumerate(zip(class_names, f1_per_class, ap_scores)):
            ap_str = f"{ap_val:.3f}" if not np.isnan(ap_val) else "N/A"
            print(f"  {class_name:>8}: F1={f1_val:.3f}, AP={ap_str}")
        
        # Show confusion matrix for critical modes
        if mode in ['no_probs', 'only_probs', 'full']:
            print(f"\nConfusion Matrix for {mode}:")
            cm = confusion_matrix(y_test, y_pred)
            print(f"      Pred: neither  donor  acceptor")
            for i, true_class in enumerate(['neither', 'donor', 'acceptor']):
                print(f"True {true_class:>7}: {cm[i][0]:>6} {cm[i][1]:>6} {cm[i][2]:>8}")
    
    return results

def main():
    """Main function focusing on proper metrics for imbalanced data."""
    
    print("🎯 Ablation Study with Proper Imbalanced Data Metrics")
    print("=" * 80)
    print("❌ IGNORING ACCURACY (useless for imbalanced data)")
    print("✅ FOCUSING ON F1 and Average Precision")
    print("=" * 80)
    
    # User's probability features (from previous conversation)
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
    
    # Load realistic dataset
    realistic_path = "results/ablation_study_fixed/realistic_dataset.parquet"
    
    if not Path(realistic_path).exists():
        print(f"❌ Realistic dataset not found: {realistic_path}")
        print("Please run test_realistic_ablation.py first")
        return
    
    df = pd.read_parquet(realistic_path)
    
    print(f"Loaded realistic dataset: {df.shape}")
    dist = df['splice_type'].value_counts()
    for value, count in dist.items():
        pct = count / len(df) * 100
        print(f"  {value}: {count:,} ({pct:.1f}%)")
    
    # Feature categorization
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
    
    # Categorize features properly
    user_prob_set = set(user_prob_features)
    feature_names = list(X_df.columns)
    
    prob_features = [f for f in feature_names if f in user_prob_set]
    kmer_features = [f for f in feature_names if f.startswith('6mer_')]
    other_features = [f for f in feature_names if f not in prob_features and f not in kmer_features]
    
    feature_categories = {
        'prob_features': prob_features,
        'kmer_features': kmer_features,
        'other_features': other_features
    }
    
    # Run ablation with proper metrics
    results = test_ablation_with_proper_metrics(df, feature_categories)
    
    print(f"\n" + "=" * 80)
    print("📊 FINAL RESULTS (Proper Metrics Only)")
    print("=" * 80)
    
    print(f"{'Mode':<12} {'F1 Macro':<10} {'AP Macro':<10} {'Features':<10} {'SpliceAI?':<10}")
    print("-" * 70)
    
    for mode, result in results.items():
        f1_macro = result['f1_macro']
        ap_macro = result['ap_macro']
        n_feat = result['n_features']
        
        # Check if mode uses SpliceAI features
        uses_spliceai = mode in ['full', 'no_kmer', 'only_probs']
        spliceai_str = "Yes" if uses_spliceai else "No"
        
        print(f"{mode:<12} {f1_macro:.3f}      {ap_macro:.3f}      {n_feat:<10} {spliceai_str:<10}")
    
    print(f"\n🔍 Key Insights (Proper Metrics):")
    
    if 'no_probs' in results and 'full' in results:
        no_probs_f1 = results['no_probs']['f1_macro']
        full_f1 = results['full']['f1_macro']
        
        f1_drop = full_f1 - no_probs_f1
        
        print(f"  📉 F1 drop without SpliceAI: {f1_drop:.3f} ({f1_drop/full_f1*100:.1f}% relative)")
        
        if f1_drop > 0.2:  # > 20% drop
            print(f"  ✅ SpliceAI features are CRITICAL for splice site prediction")
        else:
            print(f"  ⚠️ SpliceAI features less important than expected")
    
    if 'only_kmer' in results:
        only_kmer_f1 = results['only_kmer']['f1_macro']
        random_baseline = 1/3  # 3-class random performance
        
        print(f"  🧬 K-mers alone F1: {only_kmer_f1:.3f} vs random baseline: {random_baseline:.3f}")
        
        if only_kmer_f1 > 0.5:
            print(f"  💡 Sequence features surprisingly informative")
        else:
            print(f"  📊 Sequence features have limited predictive power")
    
    print(f"\n✅ These results make biological sense because:")
    print(f"  - F1 and AP properly handle class imbalance")
    print(f"  - Focus on minority classes (splice sites) that actually matter")
    print(f"  - Accuracy would be misleading with 87% majority class")

if __name__ == "__main__":
    main() 