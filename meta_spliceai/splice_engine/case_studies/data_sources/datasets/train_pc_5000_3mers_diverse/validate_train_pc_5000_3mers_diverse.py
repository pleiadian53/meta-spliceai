#!/usr/bin/env python3
"""
Validation script for train_pc_5000_3mers_diverse dataset.

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
    Validate the train_pc_5000_3mers_diverse dataset.
    
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
        
        # Check expected columns (enhanced manifest format)
        expected_cols = [
            'global_index', 'gene_id', 'gene_name', 'gene_type', 'chrom', 'strand',
            'gene_length', 'start', 'end', 'total_splice_sites', 'donor_sites', 
            'acceptor_sites', 'splice_density_per_kb', 'file_index', 'file_name'
        ]
        
        results['manifest_columns'] = all(col in manifest.columns for col in expected_cols)
        results['manifest_unique_genes'] = len(manifest['gene_id'].unique()) == len(manifest)
        results['manifest_file_references'] = all(
            (master_dir / fname).exists() for fname in manifest['file_name'].unique()
        )
        
        # Check enhanced manifest data quality
        results['manifest_splice_balance'] = True
        results['manifest_positive_lengths'] = True
        if 'donor_sites' in manifest.columns and 'acceptor_sites' in manifest.columns:
            results['manifest_splice_balance'] = (
                manifest['donor_sites'] == manifest['acceptor_sites']
            ).all()
        if 'gene_length' in manifest.columns:
            results['manifest_positive_lengths'] = (manifest['gene_length'] > 0).all()
        
        print(f"   Manifest columns: {'✅' if results['manifest_columns'] else '❌'}")
        print(f"   Unique genes: {'✅' if results['manifest_unique_genes'] else '❌'}")
        print(f"   File references: {'✅' if results['manifest_file_references'] else '❌'}")
        print(f"   Splice site balance: {'✅' if results['manifest_splice_balance'] else '❌'}")
        print(f"   Positive gene lengths: {'✅' if results['manifest_positive_lengths'] else '❌'}")
        print(f"   Total genes: {len(manifest):,}")
        
        # Show enhanced statistics
        if 'gene_type' in manifest.columns:
            gene_types = manifest['gene_type'].value_counts()
            print(f"   Gene types: {len(gene_types)} types, top: {dict(gene_types.head(3))}")
        if 'splice_density_per_kb' in manifest.columns:
            density_mean = manifest['splice_density_per_kb'].mean()
            print(f"   Mean splice density: {density_mean:.2f} sites/kb")
        
    except Exception as e:
        print(f"   ❌ Error loading manifest: {e}")
        results['manifest_columns'] = False
        results['manifest_unique_genes'] = False
        results['manifest_file_references'] = False
        results['manifest_splice_balance'] = False
        results['manifest_positive_lengths'] = False
    
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
                'gene_id', 'gene_type', 'chrom', 'strand', 'position',
                'donor_score', 'acceptor_score', 'neither_score'
            ]
            
            results['batch_schema'] = all(col in sample_batch.columns for col in key_features)
            results['batch_feature_count'] = len(sample_batch.columns) >= 140  # Should be 143-148
            
            # Check data types and ranges
            score_cols = ['donor_score', 'acceptor_score', 'neither_score']
            results['score_ranges'] = all(
                (sample_batch[col] >= 0).all() and (sample_batch[col] <= 1).all()
                for col in score_cols if col in sample_batch.columns
            )
            
            # Check gene type diversity
            gene_types = sample_batch['gene_type'].unique()
            results['gene_type_diversity'] = len(gene_types) > 1  # Should have multiple gene types
            
            # Check chromosome coverage
            chromosomes = sample_batch['chrom'].unique()
            results['chromosome_coverage'] = len(chromosomes) > 10  # Should cover many chromosomes
            
            print(f"   Batch schema: {'✅' if results['batch_schema'] else '❌'}")
            print(f"   Feature count (≥140): {'✅' if results['batch_feature_count'] else '❌'}")
            print(f"   Score ranges [0,1]: {'✅' if results['score_ranges'] else '❌'}")
            print(f"   Gene type diversity: {'✅' if results['gene_type_diversity'] else '❌'}")
            print(f"   Chromosome coverage: {'✅' if results['chromosome_coverage'] else '❌'}")
            print(f"   Sample batch shape: {sample_batch.shape}")
            print(f"   Gene types found: {list(gene_types)[:5]}{'...' if len(gene_types) > 5 else ''}")
            
        except Exception as e:
            print(f"   ❌ Error loading batch file: {e}")
            results['batch_schema'] = False
            results['batch_feature_count'] = False
            results['score_ranges'] = False
            results['gene_type_diversity'] = False
            results['chromosome_coverage'] = False
    else:
        print("   ❌ No batch files found")
        results['batch_schema'] = False
        results['batch_feature_count'] = False
        results['score_ranges'] = False
        results['gene_type_diversity'] = False
        results['chromosome_coverage'] = False
    
    # 4. Check 3-mer features
    print("\n🧬 Validating 3-mer features...")
    try:
        if results['batch_files_exist']:
            sample_batch = pd.read_parquet(batch_files[0])
            kmer_cols = [col for col in sample_batch.columns if col.startswith('3mer_')]
            
            results['kmer_features'] = len(kmer_cols) == 64  # Should have all 64 3-mers
            results['kmer_ranges'] = all(
                (sample_batch[col] >= 0).all()  # 3-mers are counts, not frequencies
                for col in kmer_cols
            ) if kmer_cols else False
            
            print(f"   3-mer feature count (64): {'✅' if results['kmer_features'] else '❌'}")
            print(f"   3-mer value ranges (≥0): {'✅' if results['kmer_ranges'] else '❌'}")
            print(f"   Found {len(kmer_cols)} 3-mer features")
        else:
            results['kmer_features'] = False
            results['kmer_ranges'] = False
            
    except Exception as e:
        print(f"   ❌ Error validating 3-mer features: {e}")
        results['kmer_features'] = False
        results['kmer_ranges'] = False
    
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
    
    # Load sample batch for analysis
    print("\n2️⃣ Loading sample batch...")
    batch_files = list(master_dir.glob("batch_*.parquet"))
    if batch_files:
        sample_batch = pd.read_parquet(batch_files[0])
        print(f"   Loaded batch with shape: {sample_batch.shape}")
        
        # Show gene type distribution
        if 'gene_type' in sample_batch.columns:
            gene_type_dist = sample_batch['gene_type'].value_counts()
            print(f"   Gene type distribution: {dict(gene_type_dist.head())}")
        
        # Show chromosome distribution
        if 'chrom' in sample_batch.columns:
            chrom_dist = sample_batch['chrom'].value_counts().sort_index()
            print(f"   Chromosome distribution: {dict(chrom_dist.head())}")
        
        # Show splice score statistics
        score_cols = ['donor_score', 'acceptor_score', 'neither_score']
        for col in score_cols:
            if col in sample_batch.columns:
                mean_score = sample_batch[col].mean()
                print(f"   {col} mean: {mean_score:.3f}")
    
    # Memory usage estimation
    print("\n3️⃣ Memory usage estimation...")
    total_size_mb = sum(f.stat().st_size for f in batch_files) / (1024 * 1024)
    print(f"   Total dataset size: {total_size_mb:.1f} MB")
    print(f"   Estimated RAM needed: {total_size_mb * 2:.1f} MB (for full loading)")
    
    # Feature analysis
    print("\n4️⃣ Feature analysis...")
    if batch_files:
        sample_batch = pd.read_parquet(batch_files[0])
        
        # Count feature categories
        spliceai_cols = [col for col in sample_batch.columns 
                        if any(x in col.lower() for x in ['donor', 'acceptor', 'score', 'context'])]
        kmer_cols = [col for col in sample_batch.columns if col.startswith('3mer_')]
        genomic_cols = [col for col in sample_batch.columns 
                       if any(x in col.lower() for x in ['gene', 'transcript', 'position', 'chrom'])]
        
        print(f"   SpliceAI features: {len(spliceai_cols)}")
        print(f"   3-mer features: {len(kmer_cols)}")
        print(f"   Genomic features: {len(genomic_cols)}")
        print(f"   Total features: {len(sample_batch.columns)}")

def main():
    """Main validation function."""
    
    # Default dataset path (relative to script location)
    default_path = "../../../../train_pc_5000_3mers_diverse"
    dataset_path = sys.argv[1] if len(sys.argv) > 1 else default_path
    
    print("🧬 TRAIN_PC_5000_3MERS_DIVERSE DATASET VALIDATION")
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
