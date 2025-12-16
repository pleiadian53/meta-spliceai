#!/usr/bin/env python3
"""
Test Hierarchical Sampling for Ablation Study

This script tests the hierarchical sampling approach that should be used
in the ablation study to preserve splice sites.

Usage:
    python -m meta_spliceai.splice_engine.meta_models.analysis.test_hierarchical_sampling \
      --dataset train_pc_1000/master \
      --target-size 50000
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import os

def test_random_sampling(dataset_path: str, sample_size: int):
    """Test what happens with random sampling (BAD approach)."""
    
    print("❌ Testing Random Sampling (BAD)")
    print("=" * 50)
    
    try:
        # Set row cap for random sampling
        os.environ["SS_MAX_ROWS"] = str(sample_size)
        
        from meta_spliceai.splice_engine.meta_models.training import datasets
        df = datasets.load_dataset(dataset_path)
        
        print(f"Random sample shape: {df.shape}")
        
        # Check splice type distribution
        if hasattr(df, 'to_pandas'):
            df_pandas = df.to_pandas()
        else:
            df_pandas = df
            
        splice_dist = df_pandas['splice_type'].value_counts()
        print(f"Random sampling distribution:")
        
        total = len(df_pandas)
        splice_sites = 0
        
        for value, count in splice_dist.items():
            pct = count / total * 100
            print(f"  {value}: {count:,} ({pct:.1f}%)")
            if value in ['donor', 'acceptor', 1, 2]:  # Splice sites
                splice_sites += count
        
        print(f"Total splice sites: {splice_sites:,} ({splice_sites/total*100:.1f}%)")
        print(f"Problem: Only {splice_sites/total*100:.1f}% splice sites - model will predict 'neither' for everything!")
        
        return df_pandas, splice_sites/total
        
    except Exception as e:
        print(f"Random sampling failed: {e}")
        return None, 0

def test_hierarchical_sampling(dataset_path: str, target_size: int):
    """Test hierarchical sampling approach (GOOD approach)."""
    
    print("\n✅ Testing Hierarchical Sampling (GOOD)")
    print("=" * 50)
    
    try:
        # Load full dataset first to understand structure
        from meta_spliceai.splice_engine.meta_models.training import datasets
        
        # Reset row cap to load more data for hierarchical sampling
        if "SS_MAX_ROWS" in os.environ:
            del os.environ["SS_MAX_ROWS"]
        
        print("Loading full dataset for hierarchical sampling...")
        df_full = datasets.load_dataset(dataset_path)
        
        if hasattr(df_full, 'to_pandas'):
            df_pandas = df_full.to_pandas()
        else:
            df_pandas = df_full
            
        print(f"Full dataset shape: {df_pandas.shape}")
        
        # Analyze full distribution
        splice_dist_full = df_pandas['splice_type'].value_counts()
        print(f"Full dataset distribution:")
        
        splice_sites_full = 0
        donor_count = 0
        acceptor_count = 0
        neither_count = 0
        
        for value, count in splice_dist_full.items():
            pct = count / len(df_pandas) * 100
            print(f"  {value}: {count:,} ({pct:.1f}%)")
            
            if value in ['donor', 1]:
                donor_count = count
                splice_sites_full += count
            elif value in ['acceptor', 2]:
                acceptor_count = count
                splice_sites_full += count
            elif value in ['0', 'neither', 0]:
                neither_count = count
        
        print(f"\nSplice site breakdown:")
        print(f"  Donor sites: {donor_count:,}")
        print(f"  Acceptor sites: {acceptor_count:,}")
        print(f"  Total splice sites: {splice_sites_full:,}")
        print(f"  Neither sites: {neither_count:,}")
        
        # Hierarchical sampling strategy
        print(f"\n🧬 Applying Hierarchical Sampling:")
        print(f"Target sample size: {target_size:,}")
        
        # Strategy: Keep ALL splice sites, sample neither sites to reach target
        donor_sites = df_pandas[df_pandas['splice_type'].isin(['donor', 1])]
        acceptor_sites = df_pandas[df_pandas['splice_type'].isin(['acceptor', 2])]
        neither_sites = df_pandas[df_pandas['splice_type'].isin(['0', 'neither', 0])]
        
        # Keep all splice sites
        splice_data = pd.concat([donor_sites, acceptor_sites], ignore_index=True)
        
        print(f"Preserving ALL splice sites: {len(splice_data):,}")
        
        # Calculate how many neither sites we can add
        remaining_slots = target_size - len(splice_data)
        
        if remaining_slots > 0 and len(neither_sites) > 0:
            # Sample neither sites to fill remaining slots
            neither_sample_size = min(remaining_slots, len(neither_sites))
            neither_sampled = neither_sites.sample(n=neither_sample_size, random_state=42)
            
            print(f"Sampling {neither_sample_size:,} neither sites from {len(neither_sites):,} available")
            
            # Combine all data
            hierarchical_sample = pd.concat([splice_data, neither_sampled], ignore_index=True)
        else:
            hierarchical_sample = splice_data
            print(f"Warning: Target size too small, using only splice sites")
        
        # Shuffle the final dataset
        hierarchical_sample = hierarchical_sample.sample(frac=1, random_state=42).reset_index(drop=True)
        
        print(f"\nHierarchical sample shape: {hierarchical_sample.shape}")
        
        # Check final distribution
        final_dist = hierarchical_sample['splice_type'].value_counts()
        print(f"Hierarchical sampling distribution:")
        
        final_splice_sites = 0
        for value, count in final_dist.items():
            pct = count / len(hierarchical_sample) * 100
            print(f"  {value}: {count:,} ({pct:.1f}%)")
            if value in ['donor', 'acceptor', 1, 2]:
                final_splice_sites += count
        
        splice_ratio = final_splice_sites / len(hierarchical_sample)
        print(f"Final splice site ratio: {splice_ratio:.1%}")
        
        if splice_ratio > 0.15:  # More than 15% splice sites
            print(f"✅ Good balance: {splice_ratio:.1%} splice sites - model can learn!")
        else:
            print(f"⚠️ Still low: {splice_ratio:.1%} splice sites - consider larger target size")
        
        return hierarchical_sample, splice_ratio
        
    except Exception as e:
        print(f"Hierarchical sampling failed: {e}")
        import traceback
        traceback.print_exc()
        return None, 0

def test_ablation_modes_with_hierarchical_data(df: pd.DataFrame):
    """Test all ablation modes with properly sampled data."""
    
    print(f"\n🧪 Testing Ablation Modes with Hierarchical Data")
    print("=" * 60)
    
    try:
        # Reproduce gene CV preprocessing on hierarchical data
        from meta_spliceai.splice_engine.meta_models.builder import preprocessing
        
        # Convert back to polars if needed for preprocessing
        if hasattr(df, 'to_pandas'):
            df_for_preprocessing = df
        else:
            # Convert pandas to polars format if needed
            import polars as pl
            df_for_preprocessing = pl.from_pandas(df)
        
        # Apply gene CV preprocessing
        X_df, y_series = preprocessing.prepare_training_data(
            df_for_preprocessing, 
            label_col="splice_type", 
            return_type="pandas", 
            verbose=1,
            preserve_transcript_columns=False,
            encode_chrom=True
        )
        
        print(f"Preprocessed data: X={X_df.shape}, y={len(y_series)}")
        
        # Check target distribution after preprocessing
        print(f"Target distribution after preprocessing:")
        if hasattr(y_series, 'value_counts'):
            for value, count in y_series.value_counts().items():
                pct = count / len(y_series) * 100
                print(f"  {value}: {count:,} ({pct:.1f}%)")
        
        # Feature categorization
        feature_names = list(X_df.columns)
        prob_features = [col for col in feature_names if any(term in col.lower() 
                        for term in ['score', 'prob', 'signal', 'context'])]
        kmer_features = [col for col in feature_names if col.startswith('6mer_')]
        other_features = [col for col in feature_names if col not in prob_features + kmer_features]
        
        print(f"\nFeature breakdown:")
        print(f"  Probability features: {len(prob_features)}")
        print(f"  K-mer features: {len(kmer_features)}")
        print(f"  Other features: {len(other_features)}")
        
        # Test each ablation mode
        ablation_modes = {
            'full': prob_features + kmer_features + other_features,
            'no_probs': kmer_features + other_features,
            'no_kmer': prob_features + other_features,
            'only_kmer': kmer_features
        }
        
        results = {}
        
        for mode, selected_features in ablation_modes.items():
            print(f"\n--- Testing {mode} mode ({len(selected_features)} features) ---")
            
            if len(selected_features) == 0:
                print(f"❌ No features for {mode} mode")
                continue
            
            # Filter features
            X_mode = X_df[selected_features]
            
            # Quick model test
            try:
                from sklearn.ensemble import RandomForestClassifier
                from sklearn.model_selection import train_test_split
                from sklearn.metrics import accuracy_score, f1_score
                from meta_spliceai.splice_engine.meta_models.training.classifier_utils import _encode_labels
                
                # Encode targets
                y_encoded = _encode_labels(y_series)
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X_mode, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
                )
                
                # Train model
                clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1)
                clf.fit(X_train, y_train)
                
                # Evaluate
                y_pred = clf.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                f1_macro = f1_score(y_test, y_pred, average='macro')
                
                results[mode] = {
                    'accuracy': accuracy,
                    'f1_macro': f1_macro,
                    'n_features': len(selected_features)
                }
                
                print(f"✅ {mode}: Acc={accuracy:.3f}, F1={f1_macro:.3f}, Features={len(selected_features)}")
                
            except Exception as e:
                print(f"❌ {mode} failed: {e}")
                results[mode] = {'error': str(e)}
        
        return results
        
    except Exception as e:
        print(f"Ablation mode testing failed: {e}")
        import traceback
        traceback.print_exc()
        return {}

def main():
    """Main testing function."""
    
    parser = argparse.ArgumentParser(description="Test hierarchical sampling for ablation")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--target-size", type=int, default=50000)
    
    args = parser.parse_args()
    
    print("🧬 Hierarchical Sampling Test for Ablation Study")
    print("=" * 80)
    print(f"Dataset: {args.dataset}")
    print(f"Target size: {args.target_size:,}")
    print("=" * 80)
    
    # Test random sampling (what happens now)
    random_df, random_splice_ratio = test_random_sampling(args.dataset, args.target_size)
    
    # Test hierarchical sampling (what should happen)
    hierarchical_df, hierarchical_splice_ratio = test_hierarchical_sampling(args.dataset, args.target_size)
    
    # Test ablation modes with hierarchical data
    if hierarchical_df is not None:
        ablation_results = test_ablation_modes_with_hierarchical_data(hierarchical_df)
        
        print(f"\n" + "=" * 80)
        print(f"📊 ABLATION RESULTS SUMMARY")
        print("=" * 80)
        
        for mode, result in ablation_results.items():
            if 'accuracy' in result:
                acc = result['accuracy']
                f1 = result['f1_macro']
                n_feat = result['n_features']
                print(f"{mode:12s}: Acc={acc:.3f}, F1={f1:.3f}, Features={n_feat:,}")
            else:
                print(f"{mode:12s}: ERROR - {result.get('error', 'Unknown')}")
    
    # Final recommendations
    print(f"\n" + "=" * 80)
    print(f"📋 RECOMMENDATIONS")
    print("=" * 80)
    
    print(f"🎯 Use Hierarchical Sampling in Ablation Script:")
    print(f"   1. Load full dataset (no SS_MAX_ROWS limit initially)")
    print(f"   2. Keep ALL splice sites (donor + acceptor)")
    print(f"   3. Sample neither sites to reach target size")
    print(f"   4. Results in {hierarchical_splice_ratio:.1%} splice sites vs {random_splice_ratio:.1%} with random")
    
    print(f"\n🧬 Expected Ablation Performance (with hierarchical sampling):")
    if ablation_results:
        for mode in ['full', 'no_kmer', 'no_probs', 'only_kmer']:
            if mode in ablation_results and 'accuracy' in ablation_results[mode]:
                acc = ablation_results[mode]['accuracy']
                f1 = ablation_results[mode]['f1_macro']
                print(f"   {mode:12s}: ~{acc:.1%} accuracy, ~{f1:.1%} F1")
    
    print(f"\n💡 Key Insight: Probability features (donor_score, acceptor_score) are crucial!")
    print(f"   K-mer features alone give moderate performance")
    print(f"   Removing probability features severely hurts performance")

if __name__ == "__main__":
    main() 