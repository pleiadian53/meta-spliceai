#!/usr/bin/env python3
"""
Fixed Ablation Study with Hierarchical Sampling

This script fixes the critical issues in the original ablation study:
1. Uses hierarchical sampling (preserve all splice sites, sample neither sites)
2. Uses same preprocessing pipeline as gene CV (which works)
3. Handles "0" encoding for neither/non-splice sites properly

Usage:
    python -m meta_spliceai.splice_engine.meta_models.analysis.run_ablation_study_fixed \
      --dataset train_pc_1000/master \
      --output-dir results/ablation_study_fixed \
      --target-size 50000 \
      --n-trials 3
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import json
import os
import time
from typing import Dict, List, Tuple, Any

def setup_hierarchical_sampling(dataset_path: str, target_size: int, output_dir: str) -> str:
    """
    Create hierarchically sampled dataset and save to disk.
    
    Strategy:
    1. Load full dataset
    2. Keep ALL splice sites (donor + acceptor)
    3. Sample neither sites to reach target size
    4. Save balanced dataset for ablation experiments
    
    Returns path to the balanced dataset file.
    """
    
    print("🧬 Setting up Hierarchical Sampling")
    print("=" * 50)
    
    from meta_spliceai.splice_engine.meta_models.training import datasets
    
    # Load dataset with a reasonable row cap for memory efficiency
    print(f"Loading dataset: {dataset_path}")
    print("Note: Using row cap for memory efficiency, but will preserve all splice sites")
    
    # Set a higher row cap to get more splice sites
    os.environ["SS_MAX_ROWS"] = str(min(target_size * 10, 500000))
    
    df = datasets.load_dataset(dataset_path)
    
    if hasattr(df, 'to_pandas'):
        df_pandas = df.to_pandas()
    else:
        df_pandas = df
    
    print(f"Loaded dataset shape: {df_pandas.shape}")
    
    # Analyze distribution
    splice_dist = df_pandas['splice_type'].value_counts()
    print(f"Original distribution:")
    
    splice_sites_total = 0
    donor_count = 0
    acceptor_count = 0
    neither_count = 0
    
    for value, count in splice_dist.items():
        pct = count / len(df_pandas) * 100
        print(f"  {value}: {count:,} ({pct:.1f}%)")
        
        if value in ['donor', 1]:
            donor_count = count
            splice_sites_total += count
        elif value in ['acceptor', 2]:
            acceptor_count = count
            splice_sites_total += count
        elif value in ['0', 'neither', 0]:
            neither_count = count
    
    print(f"\nSplice site breakdown:")
    print(f"  Donor sites: {donor_count:,}")
    print(f"  Acceptor sites: {acceptor_count:,}")
    print(f"  Total splice sites: {splice_sites_total:,}")
    print(f"  Neither sites: {neither_count:,}")
    
    # Apply hierarchical sampling
    print(f"\n🎯 Applying Hierarchical Sampling (target: {target_size:,})")
    
    # Separate splice sites from neither sites
    donor_sites = df_pandas[df_pandas['splice_type'].isin(['donor', 1])]
    acceptor_sites = df_pandas[df_pandas['splice_type'].isin(['acceptor', 2])]
    neither_sites = df_pandas[df_pandas['splice_type'].isin(['0', 'neither', 0])]
    
    # Keep ALL splice sites
    all_splice_sites = pd.concat([donor_sites, acceptor_sites], ignore_index=True)
    
    print(f"Preserving ALL splice sites: {len(all_splice_sites):,}")
    
    # Calculate neither sites to sample
    remaining_slots = target_size - len(all_splice_sites)
    
    if remaining_slots <= 0:
        print(f"Warning: Target size {target_size:,} is smaller than splice sites {len(all_splice_sites):,}")
        print(f"Using only splice sites for balanced dataset")
        balanced_df = all_splice_sites
    else:
        # Sample neither sites
        neither_sample_size = min(remaining_slots, len(neither_sites))
        print(f"Sampling {neither_sample_size:,} neither sites from {len(neither_sites):,} available")
        
        neither_sampled = neither_sites.sample(n=neither_sample_size, random_state=42)
        
        # Combine data
        balanced_df = pd.concat([all_splice_sites, neither_sampled], ignore_index=True)
    
    # Shuffle the dataset
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"\nFinal balanced dataset shape: {balanced_df.shape}")
    
    # Check final distribution
    final_dist = balanced_df['splice_type'].value_counts()
    print(f"Hierarchical sampling result:")
    
    final_splice_sites = 0
    for value, count in final_dist.items():
        pct = count / len(balanced_df) * 100
        print(f"  {value}: {count:,} ({pct:.1f}%)")
        if value in ['donor', 'acceptor', 1, 2]:
            final_splice_sites += count
    
    splice_ratio = final_splice_sites / len(balanced_df)
    print(f"Final splice site ratio: {splice_ratio:.1%}")
    
    if splice_ratio > 0.15:
        print(f"✅ Excellent balance: {splice_ratio:.1%} splice sites - model can learn effectively!")
    else:
        print(f"⚠️ Still imbalanced: {splice_ratio:.1%} splice sites - may need larger target size")
    
    # Save balanced dataset
    output_path = Path(output_dir) / "hierarchical_balanced_dataset.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\n💾 Saving balanced dataset to: {output_path}")
    balanced_df.to_parquet(output_path, index=False)
    
    return str(output_path)

def run_ablation_experiment(
    data_path: str, 
    mode: str, 
    output_dir: str,
    trial_id: int = 0
) -> Dict[str, Any]:
    """
    Run a single ablation experiment using the same preprocessing as gene CV.
    
    Args:
        data_path: Path to the hierarchically sampled balanced dataset
        mode: One of 'full', 'no_probs', 'no_kmer', 'only_kmer'
        output_dir: Output directory for results
        trial_id: Trial number for multiple runs
    
    Returns:
        Dictionary with results
    """
    
    print(f"\n🧪 Running Ablation Experiment: {mode} (Trial {trial_id})")
    print("=" * 60)
    
    try:
        # Load the balanced dataset
        print(f"Loading balanced dataset: {data_path}")
        df = pd.read_parquet(data_path)
        print(f"Balanced dataset shape: {df.shape}")
        
        # Check distribution
        dist = df['splice_type'].value_counts()
        print(f"Dataset distribution:")
        for value, count in dist.items():
            pct = count / len(df) * 100
            print(f"  {value}: {count:,} ({pct:.1f}%)")
        
        # Convert to polars for preprocessing (matching gene CV pipeline)
        import polars as pl
        df_polars = pl.from_pandas(df)
        
        # Apply the SAME preprocessing as gene CV
        print(f"\n📊 Applying Gene CV Preprocessing Pipeline")
        from meta_spliceai.splice_engine.meta_models.builder import preprocessing
        
        X_df, y_series = preprocessing.prepare_training_data(
            df_polars, 
            label_col="splice_type", 
            return_type="pandas", 
            verbose=1,
            preserve_transcript_columns=False,  # Same as gene CV
            encode_chrom=True  # Same as gene CV
        )
        
        print(f"Preprocessed data: X={X_df.shape}, y={len(y_series)}")
        
        # Verify no string columns (the bug we found)
        non_numeric_cols = []
        for col in X_df.columns:
            if X_df[col].dtype == 'object' or X_df[col].dtype.name.startswith('string'):
                non_numeric_cols.append(col)
        
        if non_numeric_cols:
            print(f"❌ ERROR: Found non-numeric columns that will break ML models:")
            for col in non_numeric_cols:
                print(f"   {col}: {X_df[col].dtype}")
            raise ValueError(f"Non-numeric columns detected: {non_numeric_cols}")
        else:
            print(f"✅ All features are numeric - ready for ML models")
        
        # Check target distribution after preprocessing
        print(f"Target distribution after preprocessing:")
        if hasattr(y_series, 'value_counts'):
            target_dist = y_series.value_counts()
            for value, count in target_dist.items():
                pct = count / len(y_series) * 100
                print(f"  {value}: {count:,} ({pct:.1f}%)")
        
        # Feature categorization for ablation modes
        feature_names = list(X_df.columns)
        
        # Probability/signal features
        prob_features = [col for col in feature_names if any(term in col.lower() 
                        for term in ['score', 'prob', 'signal', 'context', 'peak', 'difference'])]
        
        # K-mer features
        kmer_features = [col for col in feature_names if col.startswith('6mer_')]
        
        # Other features (genomic position, etc.)
        other_features = [col for col in feature_names if col not in prob_features + kmer_features]
        
        print(f"\nFeature categorization:")
        print(f"  Probability/signal features: {len(prob_features)}")
        print(f"  K-mer features: {len(kmer_features)}")
        print(f"  Other features: {len(other_features)}")
        
        # Select features based on ablation mode
        ablation_modes = {
            'full': prob_features + kmer_features + other_features,
            'no_probs': kmer_features + other_features,
            'no_kmer': prob_features + other_features,
            'only_kmer': kmer_features
        }
        
        if mode not in ablation_modes:
            raise ValueError(f"Unknown ablation mode: {mode}. Choose from {list(ablation_modes.keys())}")
        
        selected_features = ablation_modes[mode]
        print(f"\nAblation mode '{mode}': Using {len(selected_features)} features")
        
        if len(selected_features) == 0:
            raise ValueError(f"No features selected for mode '{mode}'")
        
        # Filter features
        X_mode = X_df[selected_features]
        
        # Train and evaluate model
        print(f"\n🤖 Training Model")
        
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split, cross_val_score
        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
        from meta_spliceai.splice_engine.meta_models.training.classifier_utils import _encode_labels
        
        # Encode targets (handle "0" encoding properly)
        y_encoded = _encode_labels(y_series)
        print(f"Target encoding: {pd.Series(y_encoded).value_counts().to_dict()}")
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_mode, y_encoded, test_size=0.2, random_state=42 + trial_id, stratify=y_encoded
        )
        
        print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
        
        # Train Random Forest (same as your other experiments)
        clf = RandomForestClassifier(
            n_estimators=100, 
            random_state=42 + trial_id, 
            n_jobs=2,  # Reduced for memory efficiency
            class_weight='balanced'  # Handle class imbalance
        )
        
        start_time = time.time()
        clf.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        # Evaluate
        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average='macro')
        f1_weighted = f1_score(y_test, y_pred, average='weighted')
        
        # ROC AUC (handle multiclass)
        try:
            if len(np.unique(y_encoded)) > 2:
                roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='macro')
            else:
                roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
        except Exception as e:
            print(f"Warning: Could not compute ROC AUC: {e}")
            roc_auc = np.nan
        
        # Cross-validation for robustness
        cv_scores = cross_val_score(clf, X_mode, y_encoded, cv=3, scoring='f1_macro', n_jobs=1)
        
        results = {
            'mode': mode,
            'trial_id': trial_id,
            'n_features': len(selected_features),
            'feature_types': {
                'prob_features': len(prob_features),
                'kmer_features': len(kmer_features),
                'other_features': len(other_features),
                'selected_prob': len([f for f in selected_features if f in prob_features]),
                'selected_kmer': len([f for f in selected_features if f in kmer_features]),
                'selected_other': len([f for f in selected_features if f in other_features])
            },
            'data_shape': {
                'total_samples': len(X_mode),
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'n_features': X_mode.shape[1]
            },
            'class_distribution': target_dist.to_dict() if hasattr(y_series, 'value_counts') else {},
            'performance': {
                'accuracy': float(accuracy),
                'f1_macro': float(f1_macro),
                'f1_weighted': float(f1_weighted),
                'roc_auc': float(roc_auc) if not np.isnan(roc_auc) else None,
                'cv_f1_mean': float(cv_scores.mean()),
                'cv_f1_std': float(cv_scores.std())
            },
            'training_time_seconds': float(train_time),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        print(f"\n📊 Results for {mode} (Trial {trial_id}):")
        print(f"  Accuracy: {accuracy:.3f}")
        print(f"  F1 (macro): {f1_macro:.3f}")
        print(f"  F1 (weighted): {f1_weighted:.3f}")
        print(f"  ROC AUC: {roc_auc:.3f}" if not np.isnan(roc_auc) else "  ROC AUC: N/A")
        print(f"  CV F1: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        print(f"  Training time: {train_time:.1f}s")
        print(f"  Features used: {len(selected_features)}")
        
        # Save detailed classification report
        class_report = classification_report(y_test, y_pred, output_dict=True)
        results['classification_report'] = class_report
        
        # Save results
        output_file = Path(output_dir) / f"ablation_{mode}_trial_{trial_id}.json"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"💾 Results saved to: {output_file}")
        
        return results
        
    except Exception as e:
        error_result = {
            'mode': mode,
            'trial_id': trial_id,
            'error': str(e),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        print(f"❌ Ablation experiment {mode} failed: {e}")
        import traceback
        traceback.print_exc()
        
        return error_result

def main():
    """Main function to run the fixed ablation study."""
    
    parser = argparse.ArgumentParser(description="Fixed Ablation Study with Hierarchical Sampling")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset path (e.g., train_pc_1000/master)")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for results")
    parser.add_argument("--target-size", type=int, default=50000, help="Target dataset size for hierarchical sampling")
    parser.add_argument("--n-trials", type=int, default=3, help="Number of trials per ablation mode")
    parser.add_argument("--modes", nargs='+', default=['full', 'no_probs', 'no_kmer', 'only_kmer'], 
                       help="Ablation modes to test")
    
    args = parser.parse_args()
    
    print("🧬 Fixed Ablation Study with Hierarchical Sampling")
    print("=" * 80)
    print(f"Dataset: {args.dataset}")
    print(f"Output directory: {args.output_dir}")
    print(f"Target size: {args.target_size:,}")
    print(f"Number of trials: {args.n_trials}")
    print(f"Ablation modes: {args.modes}")
    print("=" * 80)
    
    # Step 1: Create hierarchically sampled dataset
    balanced_dataset_path = setup_hierarchical_sampling(
        args.dataset, 
        args.target_size, 
        args.output_dir
    )
    
    # Step 2: Run ablation experiments
    all_results = []
    
    for mode in args.modes:
        print(f"\n" + "🧪" * 20 + f" MODE: {mode.upper()} " + "🧪" * 20)
        
        mode_results = []
        for trial in range(args.n_trials):
            result = run_ablation_experiment(
                balanced_dataset_path,
                mode,
                args.output_dir,
                trial_id=trial
            )
            mode_results.append(result)
            all_results.append(result)
        
        # Summarize mode results
        if mode_results and 'performance' in mode_results[0]:
            valid_results = [r for r in mode_results if 'performance' in r]
            if valid_results:
                accuracies = [r['performance']['accuracy'] for r in valid_results]
                f1_scores = [r['performance']['f1_macro'] for r in valid_results]
                
                print(f"\n📊 Summary for {mode}:")
                print(f"  Accuracy: {np.mean(accuracies):.3f} ± {np.std(accuracies):.3f}")
                print(f"  F1 (macro): {np.mean(f1_scores):.3f} ± {np.std(f1_scores):.3f}")
    
    # Step 3: Save overall summary
    summary_file = Path(args.output_dir) / "ablation_summary.json"
    summary = {
        'experiment_config': {
            'dataset': args.dataset,
            'target_size': args.target_size,
            'n_trials': args.n_trials,
            'modes_tested': args.modes,
            'balanced_dataset_path': balanced_dataset_path
        },
        'results': all_results,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n" + "=" * 80)
    print(f"🎯 ABLATION STUDY COMPLETE")
    print("=" * 80)
    print(f"📁 All results saved to: {args.output_dir}")
    print(f"📊 Summary: {summary_file}")
    print(f"📦 Balanced dataset: {balanced_dataset_path}")
    
    # Final comparison table
    print(f"\n📈 FINAL COMPARISON:")
    print(f"{'Mode':<12} {'Accuracy':<10} {'F1 Macro':<10} {'Features':<10}")
    print("-" * 50)
    
    for mode in args.modes:
        mode_results = [r for r in all_results if r.get('mode') == mode and 'performance' in r]
        if mode_results:
            accuracies = [r['performance']['accuracy'] for r in mode_results]
            f1_scores = [r['performance']['f1_macro'] for r in mode_results]
            n_features = mode_results[0]['n_features']
            
            acc_mean = np.mean(accuracies)
            f1_mean = np.mean(f1_scores)
            
            print(f"{mode:<12} {acc_mean:.3f}      {f1_mean:.3f}      {n_features:<10}")
    
    print(f"\n💡 Key Insights:")
    print(f"   - Compare 'full' vs 'no_probs' to see importance of probability features")
    print(f"   - Compare 'full' vs 'no_kmer' to see importance of k-mer features")  
    print(f"   - 'only_kmer' shows how much sequence alone can achieve")
    print(f"   - All modes should now show reasonable performance (not 0.7%!)")

if __name__ == "__main__":
    main() 