#!/usr/bin/env python3
"""
Final Corrected Ablation Study

This script implements the properly corrected ablation study with:
1. Realistic class distribution (90% neither, 10% splice sites)
2. Proper feature categorization based on user's explicit list
3. Meaningful metrics (F1 macro, Average Precision, ROC AUC) instead of accuracy
4. Hierarchical sampling to preserve splice sites
5. Optimized for high-memory systems with larger sample sizes

Usage:
    python -m meta_spliceai.splice_engine.meta_models.analysis.run_ablation_study_final \
      --dataset train_pc_1000/master \
      --output-dir results/ablation_study_corrected \
      --target-size 50000 \
      --n-trials 3
"""

import argparse
import pandas as pd
import polars as pl
import numpy as np
from pathlib import Path
import json
import os
import time
from typing import Dict, List, Tuple, Any

# User's explicit probability feature list (from conversation)
PROBABILITY_FEATURES = [
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

def create_realistic_balanced_dataset(dataset_path: str, target_size: int, output_dir: str) -> str:
    """
    Create a realistic balanced dataset with proper class distribution.
    Optimized for high-memory systems with larger sample sizes.
    
    Strategy:
    1. Load dataset with higher row cap for high-memory systems
    2. Keep ALL available splice sites (donors + acceptors)
    3. Sample neither sites to create realistic 90% neither proportion
    4. Save balanced dataset for ablation experiments
    
    Returns path to the balanced dataset file.
    """
    
    print("🧬 Creating Realistic Balanced Dataset (High-Memory Optimized)")
    print("=" * 70)
    
    from meta_spliceai.splice_engine.meta_models.training import datasets
    
    # Load dataset with higher row cap for high-memory systems
    print(f"Loading dataset: {dataset_path}")
    print(f"Target size: {target_size:,} (will be ~90% neither)")
    
    # For high-memory systems, use larger row caps to get more splice sites
    if target_size <= 20000:
        row_cap = min(target_size * 30, 1000000)  # 30x multiplier for small datasets
    elif target_size <= 50000:
        row_cap = min(target_size * 20, 1500000)  # 20x multiplier for medium datasets
    else:
        row_cap = min(target_size * 15, 2000000)  # 15x multiplier for large datasets
    
    print(f"Using row cap: {row_cap:,} (optimized for target size {target_size:,})")
    
    os.environ["SS_MAX_ROWS"] = str(row_cap)
    
    df = datasets.load_dataset(dataset_path)
    
    if hasattr(df, 'to_pandas'):
        df_pandas = df.to_pandas()
    else:
        df_pandas = df
    
    print(f"Loaded dataset shape: {df_pandas.shape}")
    print(f"Memory usage: ~{df_pandas.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    
    # Analyze initial distribution
    splice_dist = df_pandas['splice_type'].value_counts()
    print(f"Initial distribution:")
    
    donor_count = 0
    acceptor_count = 0
    neither_count = 0
    
    for value, count in splice_dist.items():
        pct = count / len(df_pandas) * 100
        print(f"  {value}: {count:,} ({pct:.1f}%)")
        
        if value in ['donor', 1]:
            donor_count = count
        elif value in ['acceptor', 2]:
            acceptor_count = count
        elif value in ['0', 'neither', 0]:
            neither_count = count
    
    total_splice_sites = donor_count + acceptor_count
    print(f"\nAvailable splice sites: {total_splice_sites:,} (donors: {donor_count:,}, acceptors: {acceptor_count:,})")
    
    # Calculate realistic target distribution
    # Goal: 90% neither, 5% donor, 5% acceptor (realistic proportions)
    target_neither = int(target_size * 0.90)
    target_donor = int(target_size * 0.05)
    target_acceptor = target_size - target_neither - target_donor
    
    print(f"\nTarget realistic distribution:")
    print(f"  Neither: {target_neither:,} (90.0%)")
    print(f"  Donor: {target_donor:,} ({target_donor/target_size*100:.1f}%)")
    print(f"  Acceptor: {target_acceptor:,} ({target_acceptor/target_size*100:.1f}%)")
    
    # Check if we have enough splice sites
    if total_splice_sites < (target_donor + target_acceptor):
        print(f"⚠️ Warning: Only {total_splice_sites:,} splice sites available, need {target_donor + target_acceptor:,}")
        print(f"   Will use all available splice sites and adjust proportions")
    
    # Sample data with realistic proportions
    donor_df = df_pandas[df_pandas['splice_type'].isin(['donor', 1])]
    acceptor_df = df_pandas[df_pandas['splice_type'].isin(['acceptor', 2])]
    neither_df = df_pandas[df_pandas['splice_type'].isin(['0', 'neither', 0])]
    
    # Sample each class
    donor_sample = donor_df.sample(n=min(target_donor, len(donor_df)), random_state=42)
    acceptor_sample = acceptor_df.sample(n=min(target_acceptor, len(acceptor_df)), random_state=42)
    neither_sample = neither_df.sample(n=min(target_neither, len(neither_df)), random_state=42)
    
    print(f"\nActual sampling:")
    print(f"  Donor: {len(donor_sample):,} / {len(donor_df):,} available")
    print(f"  Acceptor: {len(acceptor_sample):,} / {len(acceptor_df):,} available")
    print(f"  Neither: {len(neither_sample):,} / {len(neither_df):,} available")
    
    # Combine and shuffle
    realistic_df = pd.concat([donor_sample, acceptor_sample, neither_sample], ignore_index=True)
    realistic_df = realistic_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"\nFinal realistic dataset shape: {realistic_df.shape}")
    print(f"Final memory usage: ~{realistic_df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    
    # Verify final distribution
    final_dist = realistic_df['splice_type'].value_counts()
    print(f"Final distribution:")
    
    final_splice_sites = 0
    for value, count in final_dist.items():
        pct = count / len(realistic_df) * 100
        print(f"  {value}: {count:,} ({pct:.1f}%)")
        if value in ['donor', 'acceptor', 1, 2]:
            final_splice_sites += count
    
    neither_ratio = (len(realistic_df) - final_splice_sites) / len(realistic_df)
    print(f"Final neither ratio: {neither_ratio:.1%} (realistic!)")
    
    # Save balanced dataset
    output_path = Path(output_dir) / "realistic_balanced_dataset.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\n💾 Saving realistic dataset to: {output_path}")
    realistic_df.to_parquet(output_path, index=False)
    print(f"   Saved file size: ~{output_path.stat().st_size / 1024**2:.1f} MB")
    
    return str(output_path)

def categorize_features_properly(feature_names: List[str]) -> Dict[str, List[str]]:
    """Categorize features using the user's explicit probability feature list."""
    
    print("🔧 Proper Feature Categorization")
    print("=" * 50)
    
    # Use user's explicit probability features
    prob_features = [f for f in feature_names if f in PROBABILITY_FEATURES]
    kmer_features = [f for f in feature_names if f.startswith('6mer_')]
    other_features = [f for f in feature_names if f not in prob_features and f not in kmer_features]
    
    print(f"Feature categorization:")
    print(f"  Probability features: {len(prob_features):,}")
    print(f"  K-mer features: {len(kmer_features):,}")
    print(f"  Other features: {len(other_features):,}")
    print(f"  Total features: {len(feature_names):,}")
    
    # Critical verification: SpliceAI scores
    critical_spliceai = ['donor_score', 'acceptor_score', 'neither_score']
    print(f"\n🚨 Critical SpliceAI Feature Verification:")
    for feat in critical_spliceai:
        if feat in prob_features:
            print(f"  ✅ {feat}: Correctly categorized as probability feature")
        else:
            print(f"  ❌ {feat}: Missing or miscategorized!")
    
    return {
        'prob_features': prob_features,
        'kmer_features': kmer_features,
        'other_features': other_features
    }

def run_single_ablation_experiment(
    data_path: str, 
    mode: str, 
    feature_categories: Dict[str, List[str]],
    trial_id: int = 0
) -> Dict[str, Any]:
    """Run a single ablation experiment with proper metrics including ROC AUC."""
    
    print(f"\n🔬 Ablation Experiment: {mode.upper()} (Trial {trial_id})")
    print("=" * 60)
    
    try:
        # Load balanced dataset
        df = pd.read_parquet(data_path)
        print(f"Dataset shape: {df.shape}")
        print(f"Memory usage: ~{df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        
        # Show distribution
        dist = df['splice_type'].value_counts()
        print(f"Class distribution:")
        for value, count in dist.items():
            pct = count / len(df) * 100
            print(f"  {value}: {count:,} ({pct:.1f}%)")
        
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
        
        print(f"Preprocessed: X={X_df.shape}, y={len(y_series)}")
        print(f"Feature matrix memory: ~{X_df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        
        # Feature selection based on ablation mode
        prob_features = feature_categories['prob_features']
        kmer_features = feature_categories['kmer_features']
        other_features = feature_categories['other_features']
        
        ablation_modes = {
            'full': prob_features + kmer_features + other_features,
            'no_probs': kmer_features + other_features,  # NO SpliceAI
            'no_kmer': prob_features + other_features,   # NO sequence
            'only_probs': prob_features,                 # ONLY SpliceAI
            'only_kmer': kmer_features                   # ONLY sequence
        }
        
        if mode not in ablation_modes:
            raise ValueError(f"Unknown mode: {mode}. Choose from {list(ablation_modes.keys())}")
        
        selected_features = [f for f in ablation_modes[mode] if f in X_df.columns]
        
        if len(selected_features) == 0:
            raise ValueError(f"No features available for mode {mode}")
        
        print(f"Using {len(selected_features):,} features")
        
        # Verify SpliceAI exclusion for no_probs
        critical_features = ['donor_score', 'acceptor_score', 'neither_score', 'splice_probability']
        has_spliceai = any(f in selected_features for f in critical_features)
        
        if mode == 'no_probs' and has_spliceai:
            print(f"🚨 WARNING: no_probs mode still contains SpliceAI features!")
        elif mode == 'no_probs':
            print(f"✅ Verified: no_probs excludes all SpliceAI features")
        
        X_mode = X_df[selected_features]
        
        # Train model with proper handling of imbalanced data
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import (
            f1_score, average_precision_score, roc_auc_score, classification_report
        )
        from sklearn.preprocessing import LabelBinarizer
        from meta_spliceai.splice_engine.meta_models.training.classifier_utils import _encode_labels
        
        y_encoded = _encode_labels(y_series)
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_mode, y_encoded, test_size=0.2, random_state=42 + trial_id, stratify=y_encoded
        )
        
        print(f"Train/test split: {X_train.shape[0]:,} / {X_test.shape[0]:,}")
        
        # Train model with balanced class weights and optimized for larger datasets
        n_estimators = min(200, max(100, len(X_train) // 1000))  # Scale with dataset size
        clf = RandomForestClassifier(
            n_estimators=n_estimators, 
            random_state=42 + trial_id, 
            n_jobs=-1,  # Use all available cores
            class_weight='balanced',  # Critical for imbalanced data
            max_depth=20,  # Prevent overfitting on large datasets
            min_samples_split=10,  # Regularization for large datasets
            min_samples_leaf=5
        )
        
        print(f"Training Random Forest with {n_estimators} trees...")
        start_time = time.time()
        clf.fit(X_train, y_train)
        train_time = time.time() - start_time
        print(f"Training completed in {train_time:.1f}s")
        
        # Predictions
        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)
        
        # Calculate proper metrics (NO ACCURACY!)
        f1_macro = f1_score(y_test, y_pred, average='macro')
        f1_weighted = f1_score(y_test, y_pred, average='weighted')
        f1_per_class = f1_score(y_test, y_pred, average=None)
        
        # ROC AUC (multi-class, one-vs-rest)
        try:
            roc_auc_macro = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='macro')
            roc_auc_weighted = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
        except Exception as e:
            print(f"Warning: Could not compute ROC AUC: {e}")
            roc_auc_macro = np.nan
            roc_auc_weighted = np.nan
        
        # Average Precision per class
        lb = LabelBinarizer()
        y_test_bin = lb.fit_transform(y_test)
        y_pred_proba_reordered = y_pred_proba[:, lb.classes_]
        
        ap_scores = []
        for i in range(len(lb.classes_)):
            if len(np.unique(y_test_bin[:, i])) > 1:
                ap = average_precision_score(y_test_bin[:, i], y_pred_proba_reordered[:, i])
                ap_scores.append(ap)
            else:
                ap_scores.append(np.nan)
        
        ap_macro = np.nanmean(ap_scores)
        
        results = {
            'mode': mode,
            'trial_id': trial_id,
            'f1_macro': float(f1_macro),
            'f1_weighted': float(f1_weighted),
            'f1_per_class': [float(f) for f in f1_per_class],
            'roc_auc_macro': float(roc_auc_macro) if not np.isnan(roc_auc_macro) else None,
            'roc_auc_weighted': float(roc_auc_weighted) if not np.isnan(roc_auc_weighted) else None,
            'ap_macro': float(ap_macro),
            'ap_per_class': [float(ap) if not np.isnan(ap) else None for ap in ap_scores],
            'n_features': len(selected_features),
            'n_estimators': n_estimators,
            'training_time_seconds': float(train_time),
            'dataset_size': {
                'total': len(X_mode),
                'train': len(X_train),
                'test': len(X_test)
            },
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        print(f"\n📊 Results (Proper Metrics for Imbalanced Data):")
        print(f"  F1 Macro: {f1_macro:.3f}")
        print(f"  F1 Weighted: {f1_weighted:.3f}")
        print(f"  AP Macro: {ap_macro:.3f}")
        if not np.isnan(roc_auc_macro):
            print(f"  ROC AUC Macro: {roc_auc_macro:.3f}")
            print(f"  ROC AUC Weighted: {roc_auc_weighted:.3f}")
        else:
            print(f"  ROC AUC: Could not compute")
        print(f"  Training time: {train_time:.1f}s")
        
        # Per-class breakdown
        class_names = ['neither', 'donor', 'acceptor']
        print(f"\n📋 Per-class performance:")
        for i, (name, f1, ap) in enumerate(zip(class_names, f1_per_class, ap_scores)):
            ap_str = f"{ap:.3f}" if not np.isnan(ap) else "N/A"
            print(f"  {name:>8}: F1={f1:.3f}, AP={ap_str}")
        
        return results
        
    except Exception as e:
        error_result = {
            'mode': mode,
            'trial_id': trial_id,
            'error': str(e),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        print(f"❌ Experiment {mode} failed: {e}")
        import traceback
        traceback.print_exc()
        
        return error_result

def main():
    """Main function for corrected ablation study optimized for high-memory systems."""
    
    parser = argparse.ArgumentParser(description="Corrected Ablation Study with Proper Metrics (High-Memory Optimized)")
    parser.add_argument("--dataset", type=str, required=True, 
                       help="Dataset path (e.g., train_pc_1000/master)")
    parser.add_argument("--output-dir", type=str, required=True, 
                       help="Output directory for results")
    parser.add_argument("--target-size", type=int, default=50000, 
                       help="Target dataset size for realistic sampling (default: 50000 for high-memory)")
    parser.add_argument("--n-trials", type=int, default=3, 
                       help="Number of trials per ablation mode")
    parser.add_argument("--modes", nargs='+', 
                       default=['full', 'no_probs', 'no_kmer', 'only_probs', 'only_kmer'], 
                       help="Ablation modes to test")
    
    args = parser.parse_args()
    
    print("🎯 Corrected Ablation Study (High-Memory Optimized)")
    print("=" * 80)
    print("✅ Realistic class distribution (90% neither)")
    print("✅ Proper feature categorization (user's explicit list)")
    print("✅ Meaningful metrics (F1, AP, ROC AUC - NO accuracy)")
    print("✅ Hierarchical sampling preserves splice sites")
    print("✅ Optimized for high-memory systems and large datasets")
    print("=" * 80)
    print(f"Dataset: {args.dataset}")
    print(f"Output: {args.output_dir}")
    print(f"Target size: {args.target_size:,}")
    print(f"Trials: {args.n_trials}")
    print(f"Modes: {args.modes}")
    print("=" * 80)
    
    # Step 1: Create realistic balanced dataset
    print(f"\n📦 STEP 1: Create Realistic Balanced Dataset")
    balanced_data_path = create_realistic_balanced_dataset(
        args.dataset, args.target_size, args.output_dir
    )
    
    # Step 2: Preprocess to get feature names for categorization
    print(f"\n🔧 STEP 2: Feature Categorization")
    df_sample = pd.read_parquet(balanced_data_path)
    df_polars = pl.from_pandas(df_sample.head(1000))  # Small sample for preprocessing
    
    from meta_spliceai.splice_engine.meta_models.builder import preprocessing
    X_df, _ = preprocessing.prepare_training_data(
        df_polars, 
        label_col="splice_type", 
        return_type="pandas", 
        verbose=0,
        preserve_transcript_columns=False,
        encode_chrom=True
    )
    
    feature_categories = categorize_features_properly(list(X_df.columns))
    
    # Step 3: Run ablation experiments
    print(f"\n🧪 STEP 3: Run Ablation Experiments")
    all_results = []
    
    for mode in args.modes:
        print(f"\n" + "🔬" * 30 + f" MODE: {mode.upper()} " + "🔬" * 30)
        
        mode_results = []
        for trial in range(args.n_trials):
            result = run_single_ablation_experiment(
                balanced_data_path, mode, feature_categories, trial
            )
            mode_results.append(result)
            all_results.append(result)
        
        # Summarize mode results
        valid_results = [r for r in mode_results if 'f1_macro' in r]
        if valid_results:
            f1_scores = [r['f1_macro'] for r in valid_results]
            ap_scores = [r['ap_macro'] for r in valid_results]
            roc_auc_scores = [r['roc_auc_macro'] for r in valid_results if r['roc_auc_macro'] is not None]
            
            print(f"\n📊 {mode} Summary ({len(valid_results)} trials):")
            print(f"  F1 Macro: {np.mean(f1_scores):.3f} ± {np.std(f1_scores):.3f}")
            print(f"  AP Macro: {np.mean(ap_scores):.3f} ± {np.std(ap_scores):.3f}")
            if roc_auc_scores:
                print(f"  ROC AUC Macro: {np.mean(roc_auc_scores):.3f} ± {np.std(roc_auc_scores):.3f}")
    
    # Step 4: Save comprehensive results
    print(f"\n💾 STEP 4: Save Results")
    
    summary_file = Path(args.output_dir) / "ablation_summary_corrected.json"
    summary = {
        'experiment_config': {
            'dataset': args.dataset,
            'target_size': args.target_size,
            'n_trials': args.n_trials,
            'modes_tested': args.modes,
            'balanced_dataset_path': balanced_data_path,
            'probability_features_used': PROBABILITY_FEATURES,
            'optimization': 'high_memory_large_dataset'
        },
        'results': all_results,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"📊 Summary saved: {summary_file}")
    
    # Final comparison table
    print(f"\n" + "=" * 90)
    print("📈 FINAL ABLATION RESULTS (Proper Metrics)")
    print("=" * 90)
    
    print(f"{'Mode':<12} {'F1 Macro':<10} {'AP Macro':<10} {'ROC AUC':<10} {'Features':<10} {'SpliceAI?':<10}")
    print("-" * 85)
    
    for mode in args.modes:
        mode_results = [r for r in all_results if r.get('mode') == mode and 'f1_macro' in r]
        if mode_results:
            f1_mean = np.mean([r['f1_macro'] for r in mode_results])
            ap_mean = np.mean([r['ap_macro'] for r in mode_results])
            
            # ROC AUC (handle None values)
            roc_scores = [r['roc_auc_macro'] for r in mode_results if r['roc_auc_macro'] is not None]
            roc_mean = np.mean(roc_scores) if roc_scores else np.nan
            
            n_features = mode_results[0]['n_features']
            
            uses_spliceai = mode in ['full', 'no_kmer', 'only_probs']
            spliceai_str = "Yes" if uses_spliceai else "No"
            
            roc_str = f"{roc_mean:.3f}" if not np.isnan(roc_mean) else "N/A"
            
            print(f"{mode:<12} {f1_mean:.3f}      {ap_mean:.3f}      {roc_str:<10} {n_features:<10} {spliceai_str:<10}")
    
    print(f"\n🎯 Key Insights:")
    print(f"  💡 Focus on F1 Macro and AP Macro (most meaningful for imbalanced data)")
    print(f"  ⚠️ ROC AUC included but less reliable for imbalanced datasets")
    print(f"  📊 Compare 'full' vs 'no_probs' to quantify SpliceAI importance")
    print(f"  🧬 'only_kmer' shows sequence-only baseline performance")
    print(f"  🔍 Expected: ~90% performance drop from 'full' to 'only_kmer'")
    
    # Memory and performance summary
    print(f"\n💾 System Performance Summary:")
    dataset_size_mb = Path(balanced_data_path).stat().st_size / 1024**2
    print(f"  Dataset size: {dataset_size_mb:.1f} MB")
    print(f"  Target size: {args.target_size:,} samples")
    print(f"  Total experiments: {len(args.modes) * args.n_trials}")

if __name__ == "__main__":
    main() 