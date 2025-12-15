#!/usr/bin/env python3
"""
Validation script for train_pc_7000_3mers_opt dataset.

This script demonstrates how to load and validate the dataset,
providing examples of common usage patterns and quality checks.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
from typing import Dict, List, Tuple

def validate_dataset(dataset_path: str) -> Dict[str, bool]:
    """
    Validate the train_pc_7000_3mers_opt dataset.
    
    Args:
        dataset_path: Path to the dataset directory
        
    Returns:
        Dictionary of validation results
    """
    results = {}
    dataset_dir = Path(dataset_path)
    master_dir = dataset_dir / "master"
    
    print(f"🔍 Validating dataset: {dataset_path}")
    print("=" * 60)
    
    # 1. Check directory structure
    print("📁 Checking directory structure...")
    results['directory_structure'] = (
        dataset_dir.exists() and 
        master_dir.exists() and
        (master_dir / "gene_manifest.csv").exists()
    )
    print(f"   Directory structure: {'✅' if results['directory_structure'] else '❌'}")
    
    # 2. Load and validate gene manifest
    print("\n📊 Validating gene manifest...")
    try:
        manifest = pd.read_csv(master_dir / "gene_manifest.csv")
        
        # Check expected columns
        expected_cols = [
            'global_index', 'gene_id', 'gene_name', 'gene_type', 'chrom',
            'strand', 'gene_length', 'start', 'end', 'total_splice_sites',
            'donor_sites', 'acceptor_sites', 'splice_density_per_kb',
            'file_index', 'file_name'
        ]
        
        results['manifest_columns'] = all(col in manifest.columns for col in expected_cols)
        results['manifest_gene_type'] = (manifest['gene_type'] == 'protein_coding').all()
        results['manifest_splice_balance'] = (
            manifest['donor_sites'] + manifest['acceptor_sites'] == 
            manifest['total_splice_sites']
        ).all()
        
        print(f"   Manifest columns: {'✅' if results['manifest_columns'] else '❌'}")
        print(f"   Gene types (protein_coding): {'✅' if results['manifest_gene_type'] else '❌'}")
        print(f"   Splice site balance: {'✅' if results['manifest_splice_balance'] else '❌'}")
        print(f"   Total genes: {len(manifest):,}")
        
        # Show enhanced statistics
        if 'gene_type' in manifest.columns:
            gene_types = manifest['gene_type'].value_counts()
            print(f"   Gene types: {len(gene_types)} types, primary: {dict(gene_types.head(3))}")
        if 'splice_density_per_kb' in manifest.columns:
            density_mean = manifest['splice_density_per_kb'].mean()
            density_median = manifest['splice_density_per_kb'].median()
            print(f"   Splice density: mean {density_mean:.2f}, median {density_median:.2f} sites/kb")
        if 'gene_length' in manifest.columns:
            length_mean = manifest['gene_length'].mean()
            length_median = manifest['gene_length'].median()
            print(f"   Gene length: mean {length_mean:,.0f}, median {length_median:,.0f} bp")
        
    except Exception as e:
        print(f"   ❌ Error loading manifest: {e}")
        results['manifest_columns'] = False
        results['manifest_gene_type'] = False
        results['manifest_splice_balance'] = False
    
    # 3. Check batch files
    print("\n📦 Checking batch files...")
    batch_files = list(master_dir.glob("batch_*.parquet"))
    results['batch_files_exist'] = len(batch_files) > 0
    
    if results['batch_files_exist']:
        print(f"   Found {len(batch_files)} batch files")
        
        # Load first batch for schema validation
        try:
            sample_batch = pd.read_parquet(batch_files[0])
            
            # Check key features exist
            key_features = [
                'splice_type', 'pred_type', 'donor_score', 'acceptor_score',
                'neither_score', 'gene_id', 'chrom', 'position'
            ]
            
            results['batch_schema'] = all(col in sample_batch.columns for col in key_features)
            results['batch_feature_count'] = len(sample_batch.columns) >= 140  # Should be 143
            
            # Check data types and ranges
            score_cols = ['donor_score', 'acceptor_score', 'neither_score']
            results['score_ranges'] = all(
                (sample_batch[col] >= 0).all() and (sample_batch[col] <= 1).all()
                for col in score_cols if col in sample_batch.columns
            )
            
            print(f"   Batch schema: {'✅' if results['batch_schema'] else '❌'}")
            print(f"   Feature count (≥140): {'✅' if results['batch_feature_count'] else '❌'}")
            print(f"   Score ranges [0,1]: {'✅' if results['score_ranges'] else '❌'}")
            print(f"   Sample batch shape: {sample_batch.shape}")
            
        except Exception as e:
            print(f"   ❌ Error loading batch file: {e}")
            results['batch_schema'] = False
            results['batch_feature_count'] = False
            results['score_ranges'] = False
    else:
        print("   ❌ No batch files found")
        results['batch_schema'] = False
        results['batch_feature_count'] = False
        results['score_ranges'] = False
    
    return results

def demonstrate_usage(dataset_path: str) -> None:
    """Demonstrate common usage patterns for the dataset."""
    
    print("\n" + "=" * 60)
    print("📚 USAGE DEMONSTRATION")
    print("=" * 60)
    
    dataset_dir = Path(dataset_path)
    master_dir = dataset_dir / "master"
    
    # Load gene manifest
    print("\n1️⃣ Loading gene manifest...")
    manifest = pd.read_csv(master_dir / "gene_manifest.csv")
    print(f"   Loaded {len(manifest):,} genes")
    
    # Show chromosome distribution
    chrom_dist = manifest['chrom'].value_counts().sort_index()
    print(f"   Chromosome distribution: {dict(chrom_dist.head())}")
    
    # Load sample batch
    print("\n2️⃣ Loading sample batch...")
    batch_files = list(master_dir.glob("batch_*.parquet"))
    if batch_files:
        sample_batch = pd.read_parquet(batch_files[0])
        print(f"   Loaded batch with shape: {sample_batch.shape}")
        
        # Show splice type distribution
        splice_dist = sample_batch['splice_type'].value_counts()
        print(f"   Splice type distribution: {dict(splice_dist)}")
        
        # Show prediction performance
        if 'pred_type' in sample_batch.columns:
            pred_dist = sample_batch['pred_type'].value_counts()
            print(f"   Prediction distribution: {dict(pred_dist)}")
    
    # Memory usage estimation
    print("\n3️⃣ Memory usage estimation...")
    total_size_mb = sum(f.stat().st_size for f in batch_files) / (1024 * 1024)
    print(f"   Total dataset size: {total_size_mb:.1f} MB")
    print(f"   Estimated RAM needed: {total_size_mb * 2:.1f} MB (for full loading)")

def main():
    """Main validation function."""
    
    # Default dataset path (relative to script location)
    default_path = "../../../../train_pc_7000_3mers_opt"
    dataset_path = sys.argv[1] if len(sys.argv) > 1 else default_path
    
    print("🧬 TRAIN_PC_7000_3MERS_OPT DATASET VALIDATION")
    print("=" * 60)
    print(f"Dataset path: {dataset_path}")
    
    # Validate dataset
    results = validate_dataset(dataset_path)
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 VALIDATION SUMMARY")
    print("=" * 60)
    
    passed = sum(results.values())
    total = len(results)
    
    for check, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {check.replace('_', ' ').title()}: {status}")
    
    print(f"\n🎯 Overall: {passed}/{total} checks passed")
    
    if passed == total:
        print("🎉 Dataset validation successful!")
        demonstrate_usage(dataset_path)
    else:
        print("⚠️  Some validation checks failed. Please review the dataset.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
