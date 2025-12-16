#!/usr/bin/env python3
"""
Diagnose Ablation Study Issues

This script analyzes the ablation study output to identify potential problems
with sampling, data processing, or model training.

Usage:
    python -m meta_spliceai.splice_engine.meta_models.analysis.diagnose_ablation_issue \
      --ablation-dir results/gene_cv_1000_run_15/ablation_study \
      --dataset train_pc_1000/master
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import json

def analyze_ablation_results(ablation_dir: str):
    """Analyze the ablation study output files."""
    
    ablation_path = Path(ablation_dir)
    
    print("🔍 Analyzing Ablation Study Results")
    print("=" * 50)
    
    # Check if results exist
    summary_file = ablation_path / "ablation_summary.csv"
    report_file = ablation_path / "ablation_report.json"
    
    if not summary_file.exists():
        print(f"❌ Missing ablation summary: {summary_file}")
        return
    
    # Load results
    try:
        df = pd.read_csv(summary_file)
        print(f"✅ Loaded ablation results from: {summary_file}")
        print(f"\nResults overview:")
        print(df)
        
        # Check for suspicious patterns
        print(f"\n🚨 Issue Analysis:")
        
        # Issue 1: Performance too low
        max_accuracy = df['accuracy'].max()
        max_f1 = df['macro_f1'].max()
        
        if max_accuracy < 0.1:
            print(f"❌ CRITICAL: Max accuracy is {max_accuracy:.3f} - this is impossible!")
            print(f"   Expected: >0.90 based on gene CV results")
        
        if max_f1 < 0.5:
            print(f"❌ CRITICAL: Max F1 is {max_f1:.3f} - worse than random!")
            print(f"   Expected: >0.85 based on gene CV results")
        
        # Issue 2: Feature counts
        print(f"\n📊 Feature Count Analysis:")
        for _, row in df.iterrows():
            mode = row['mode']
            n_features = row['n_features']
            print(f"  {mode}: {n_features} features")
            
            if mode == 'full' and n_features < 4000:
                print(f"    ⚠️ Full model has fewer features than expected")
            elif mode == 'no_kmer' and n_features > 100:
                print(f"    ⚠️ no_kmer has too many features (should be ~65)")
            elif mode == 'only_kmer' and n_features < 4000:
                print(f"    ⚠️ only_kmer has too few features")
        
        # Issue 3: Performance patterns
        print(f"\n🎯 Performance Pattern Analysis:")
        
        full_acc = df[df['mode'] == 'full']['accuracy'].iloc[0]
        no_probs_acc = df[df['mode'] == 'no_probs']['accuracy'].iloc[0]
        no_kmer_acc = df[df['mode'] == 'no_kmer']['accuracy'].iloc[0]
        only_kmer_acc = df[df['mode'] == 'only_kmer']['accuracy'].iloc[0]
        
        if abs(full_acc - no_kmer_acc) < 0.001:
            print(f"❌ SUSPICIOUS: full and no_kmer have identical performance")
            print(f"   This suggests k-mer features aren't being used properly")
        
        if no_probs_acc > full_acc:
            print(f"❌ IMPOSSIBLE: no_probs performs better than full model")
            print(f"   This indicates a data processing error")
        
    except Exception as e:
        print(f"❌ Error reading ablation results: {e}")

def check_data_sampling(dataset_dir: str, sample_size: int = 100000):
    """Check if the hierarchical sampling is working correctly."""
    
    print(f"\n🧬 Analyzing Hierarchical Sampling")
    print("=" * 50)
    
    try:
        from pathlib import Path
        data_path = Path(dataset_dir)
        parquet_files = list(data_path.glob("batch_*.parquet"))
        
        if not parquet_files:
            print(f"❌ No parquet files found in {dataset_dir}")
            return
        
        # Load a sample to check the data structure
        print(f"Loading sample from {parquet_files[0]}")
        df = pd.read_parquet(parquet_files[0])
        
        print(f"📊 Original data shape: {df.shape}")
        
        # Check splice type distribution
        if 'splice_type' in df.columns:
            splice_dist = df['splice_type'].value_counts()
            print(f"\n🎯 Original splice type distribution:")
            for splice_type, count in splice_dist.items():
                pct = count / len(df) * 100
                print(f"  {splice_type}: {count:,} ({pct:.1f}%)")
            
            # Check if sampling preserves splice sites
            total_splice_sites = splice_dist.get('donor', 0) + splice_dist.get('acceptor', 0)
            total_neither = splice_dist.get('neither', 0)
            
            print(f"\n📈 Sampling Analysis:")
            print(f"  Total splice sites: {total_splice_sites:,}")
            print(f"  Total neither sites: {total_neither:,}")
            print(f"  Splice site ratio: {total_splice_sites/len(df):.3f}")
            
            if total_splice_sites < 1000:
                print(f"❌ CRITICAL: Too few splice sites in original data!")
            
            # Expected distribution after hierarchical sampling
            expected_sample_size = min(sample_size, len(df))
            if total_splice_sites < expected_sample_size * 0.05:
                print(f"⚠️ WARNING: Splice sites may be under-represented after sampling")
                print(f"   Consider increasing --sample-size or using stratified sampling")
        
        else:
            print(f"❌ No 'splice_type' column found. Available columns:")
            print(f"   {list(df.columns)}")
            
    except Exception as e:
        print(f"❌ Error analyzing data sampling: {e}")

def check_target_encoding():
    """Check common target encoding issues."""
    
    print(f"\n🎯 Target Encoding Diagnostics")
    print("=" * 50)
    
    print(f"Expected target encoding:")
    print(f"  donor = 1")
    print(f"  acceptor = 2") 
    print(f"  neither = 0")
    print(f"")
    print(f"🔍 Check ablation logs for:")
    print(f"  1. Target distribution during training")
    print(f"  2. Class weights or imbalance handling")
    print(f"  3. Binary model training for each class")
    print(f"  4. Prediction aggregation method")

def suggest_fixes():
    """Suggest potential fixes for the ablation study."""
    
    print(f"\n🛠️  Suggested Fixes")
    print("=" * 50)
    
    fixes = [
        {
            "issue": "Hierarchical Sampling",
            "fix": "Ensure all splice sites are preserved during sampling",
            "command": "--preserve-splice-sites --min-splice-sites 5000"
        },
        {
            "issue": "Feature Exclusion", 
            "fix": "Check if critical features were incorrectly excluded",
            "command": "--no-auto-exclude or review configs/exclude_features.txt"
        },
        {
            "issue": "Sample Size",
            "fix": "Use larger sample to ensure adequate representation",
            "command": "--row-cap 500000"
        },
        {
            "issue": "Target Encoding",
            "fix": "Verify target variable is correctly encoded",
            "command": "Check target preprocessing in ablation script"
        },
        {
            "issue": "Model Training",
            "fix": "Use same parameters as successful gene CV",
            "command": "Copy training params from gene CV script"
        }
    ]
    
    for i, fix in enumerate(fixes, 1):
        print(f"{i}. **{fix['issue']}**")
        print(f"   Fix: {fix['fix']}")
        print(f"   Command: {fix['command']}")
        print()

def main():
    """Main diagnostic function."""
    
    parser = argparse.ArgumentParser(description="Diagnose ablation study issues")
    parser.add_argument("--ablation-dir", type=str, required=True,
                       help="Directory containing ablation results")
    parser.add_argument("--dataset", type=str, 
                       help="Original dataset directory (optional)")
    
    args = parser.parse_args()
    
    print("🚨 Ablation Study Diagnostic Tool")
    print("=" * 60)
    
    # Analyze ablation results
    analyze_ablation_results(args.ablation_dir)
    
    # Check data sampling if dataset provided
    if args.dataset:
        check_data_sampling(args.dataset)
    
    # Check target encoding
    check_target_encoding()
    
    # Suggest fixes
    suggest_fixes()
    
    print(f"\n💡 Next Steps:")
    print(f"1. Review the diagnostic output above")
    print(f"2. Try the suggested fixes one by one")
    print(f"3. Compare with successful gene CV parameters")
    print(f"4. Test with a smaller sample first to debug faster")

if __name__ == "__main__":
    main() 