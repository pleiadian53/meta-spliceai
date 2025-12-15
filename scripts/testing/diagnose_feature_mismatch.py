#!/usr/bin/env python3
"""
Diagnose feature dimension mismatch between training and inference.

This script compares:
1. Features used during training (from model metadata)
2. Features generated during inference
3. Identifies missing and extra features
"""

import sys
from pathlib import Path
import pandas as pd
import polars as pl
import pickle

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def load_model_metadata(model_path: Path):
    """Load model and its feature manifest."""
    print(f"\n{'='*60}")
    print(f"LOADING MODEL METADATA")
    print(f"{'='*60}")
    
    # Load model
    print(f"\n1. Loading model from: {model_path}")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # Get feature names from model
    if hasattr(model, 'feature_names_in_'):
        training_features = list(model.feature_names_in_)
    elif hasattr(model, 'feature_name_'):
        training_features = list(model.feature_name_)
    elif hasattr(model, 'get_booster') and hasattr(model.get_booster(), 'feature_names'):
        training_features = model.get_booster().feature_names
    else:
        print("  ⚠️  Could not extract feature names from model")
        training_features = []
    
    print(f"  Model expects: {len(training_features)} features")
    
    # Load feature manifest if available
    model_dir = model_path.parent
    manifest_path = model_dir / "feature_manifest.json"
    manifest_csv_path = model_dir / "feature_manifest.csv"
    
    manifest_features = []
    if manifest_path.exists():
        import json
        print(f"\n2. Loading feature manifest: {manifest_path.name}")
        with open(manifest_path) as f:
            manifest_data = json.load(f)
        manifest_features = [f['name'] for f in manifest_data['features']]
        print(f"  Manifest lists: {len(manifest_features)} features")
    elif manifest_csv_path.exists():
        print(f"\n2. Loading feature manifest (CSV): {manifest_csv_path.name}")
        manifest_df = pd.read_csv(manifest_csv_path)
        manifest_features = manifest_df['feature'].tolist()
        print(f"  Manifest lists: {len(manifest_features)} features")
    else:
        print(f"\n2. No feature manifest found")
    
    # Load exclusions file
    exclusions_path = model_dir / "global_excluded_features.txt"
    excluded_features = []
    if exclusions_path.exists():
        print(f"\n3. Loading exclusions: {exclusions_path.name}")
        with open(exclusions_path) as f:
            excluded_features = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        print(f"  Excluded: {len(excluded_features)} features")
    else:
        print(f"\n3. No exclusions file found")
    
    return {
        'training_features': training_features,
        'manifest_features': manifest_features,
        'excluded_features': excluded_features,
        'model': model
    }


def analyze_inference_features(analysis_file: Path):
    """Analyze features generated during inference."""
    print(f"\n{'='*60}")
    print(f"ANALYZING INFERENCE FEATURES")
    print(f"{'='*60}")
    
    print(f"\n1. Loading inference analysis file: {analysis_file.name}")
    
    # Load the analysis file (TSV format)
    df = pl.read_csv(analysis_file, separator='\t')
    
    print(f"  Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Get all columns
    all_columns = df.columns
    
    print(f"\n2. Column breakdown:")
    print(f"  Total columns: {len(all_columns)}")
    
    # Categorize columns
    from meta_spliceai.splice_engine.meta_models.builder.preprocessing import (
        LEAKAGE_COLUMNS, METADATA_COLUMNS, SEQUENCE_COLUMNS, REDUNDANT_COLUMNS
    )
    from meta_spliceai.splice_engine.meta_models.builder.feature_schema import (
        is_kmer_feature
    )
    
    leakage = [c for c in all_columns if c in LEAKAGE_COLUMNS]
    metadata = [c for c in all_columns if c in METADATA_COLUMNS]
    sequence = [c for c in all_columns if c in SEQUENCE_COLUMNS]
    redundant = [c for c in all_columns if c in REDUNDANT_COLUMNS]
    kmers = [c for c in all_columns if is_kmer_feature(c)]
    
    base_exclusions = set(leakage + metadata + sequence + redundant)
    potential_features = [c for c in all_columns if c not in base_exclusions]
    
    print(f"    Leakage columns: {len(leakage)}")
    print(f"    Metadata columns: {len(metadata)}")
    print(f"    Sequence columns: {len(sequence)}")
    print(f"    Redundant columns: {len(redundant)}")
    print(f"    K-mer columns: {len(kmers)}")
    print(f"    Potential features: {len(potential_features)}")
    
    return {
        'all_columns': all_columns,
        'potential_features': potential_features,
        'kmers': kmers,
        'leakage': leakage,
        'metadata': metadata,
        'sequence': sequence,
        'redundant': redundant,
        'df': df
    }


def compare_features(model_info, inference_info):
    """Compare training vs inference features."""
    print(f"\n{'='*60}")
    print(f"FEATURE COMPARISON")
    print(f"{'='*60}")
    
    training_set = set(model_info['training_features'])
    inference_set = set(inference_info['potential_features'])
    
    # Apply exclusions to inference set
    excluded_set = set(model_info['excluded_features'])
    inference_set_filtered = inference_set - excluded_set
    
    missing_in_inference = training_set - inference_set_filtered
    extra_in_inference = inference_set_filtered - training_set
    common = training_set & inference_set_filtered
    
    print(f"\n1. Feature counts:")
    print(f"  Training expects: {len(training_set)} features")
    print(f"  Inference has (before exclusion): {len(inference_set)} features")
    print(f"  Inference has (after exclusion): {len(inference_set_filtered)} features")
    print(f"  Common: {len(common)} features")
    print(f"  Missing in inference: {len(missing_in_inference)} features")
    print(f"  Extra in inference: {len(extra_in_inference)} features")
    
    # Critical issue check
    if missing_in_inference:
        print(f"\n⚠️  CRITICAL: {len(missing_in_inference)} features missing in inference!")
        print(f"  This will cause prediction failures.")
        print(f"\n  Missing features (first 20):")
        for feature in sorted(missing_in_inference)[:20]:
            print(f"    - {feature}")
        if len(missing_in_inference) > 20:
            print(f"    ... and {len(missing_in_inference) - 20} more")
    
    if extra_in_inference:
        print(f"\n✅ Info: {len(extra_in_inference)} extra features in inference")
        print(f"  These can be safely dropped.")
        print(f"\n  Extra features (first 20):")
        for feature in sorted(extra_in_inference)[:20]:
            print(f"    - {feature}")
        if len(extra_in_inference) > 20:
            print(f"    ... and {len(extra_in_inference) - 20} more")
    
    # Check k-mer features specifically
    print(f"\n2. K-mer feature analysis:")
    inference_kmers = set(inference_info['kmers'])
    training_kmers = set([f for f in training_set if len(f) == 3 and f.isupper() and all(c in 'ACGT' for c in f)])
    
    print(f"  Training k-mers: {len(training_kmers)}")
    print(f"  Inference k-mers: {len(inference_kmers)}")
    
    missing_kmers = training_kmers - inference_kmers
    if missing_kmers:
        print(f"  ⚠️  Missing k-mers: {len(missing_kmers)}")
        print(f"    Examples: {sorted(missing_kmers)[:10]}")
    else:
        print(f"  ✅ All training k-mers present in inference")
    
    return {
        'missing': sorted(missing_in_inference),
        'extra': sorted(extra_in_inference),
        'common': sorted(common),
        'missing_kmers': sorted(missing_kmers) if missing_kmers else []
    }


def main():
    """Run feature mismatch diagnosis."""
    print("\n" + "="*60)
    print("FEATURE DIMENSION MISMATCH DIAGNOSIS")
    print("="*60)
    
    # Configuration
    model_path = Path("results/meta_model_1000genes_3mers_fresh/model_multiclass.pkl")
    analysis_file = Path("predictions/test_meta_only/predictions/meta_only/complete_base_predictions/ENSG00000141736/meta_models/complete_inference/analysis_sequences_17_chunk_1001_1500.tsv")
    
    # Check if files exist
    if not model_path.exists():
        print(f"\n❌ Model not found: {model_path}")
        return 1
    
    if not analysis_file.exists():
        print(f"\n❌ Analysis file not found: {analysis_file}")
        print(f"  Run inference first to generate this file")
        return 1
    
    # Load model metadata
    model_info = load_model_metadata(model_path)
    
    # Analyze inference features
    inference_info = analyze_inference_features(analysis_file)
    
    # Compare
    comparison = compare_features(model_info, inference_info)
    
    # Summary
    print(f"\n{'='*60}")
    print(f"DIAGNOSIS SUMMARY")
    print(f"{'='*60}")
    
    if comparison['missing']:
        print(f"\n❌ PROBLEM IDENTIFIED:")
        print(f"  Inference is missing {len(comparison['missing'])} features that the model expects")
        print(f"  Root cause: Incomplete feature generation in inference workflow")
        print(f"\n  Solution:")
        print(f"    1. Ensure all feature generation steps run during inference")
        print(f"    2. Check that k-mer generation is enabled")
        print(f"    3. Verify genomic feature enrichment is complete")
        print(f"    4. Compare feature generation code between training and inference")
        return 1
    elif comparison['extra']:
        print(f"\n✅ NO CRITICAL ISSUES:")
        print(f"  Inference has {len(comparison['extra'])} extra features (can be dropped)")
        print(f"  All {len(comparison['common'])} required features are present")
        print(f"\n  Action: Filter extra features before calling model.predict()")
        return 0
    else:
        print(f"\n✅ PERFECT MATCH:")
        print(f"  All {len(comparison['common'])} features match exactly")
        print(f"  No missing or extra features")
        return 0


if __name__ == "__main__":
    sys.exit(main())

