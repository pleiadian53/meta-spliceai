#!/usr/bin/env python3
"""
Direct test of the three fixed methods without running full inference pipeline.
"""

import sys
from pathlib import Path
import polars as pl
import numpy as np

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from meta_spliceai.splice_engine.meta_models.workflows.inference.enhanced_selective_inference import (
    EnhancedSelectiveInferenceConfig,
    EnhancedSelectiveInferenceWorkflow
)
from meta_spliceai.splice_engine.meta_models.workflows.inference_workflow_utils import load_model_with_calibration

def main():
    print("=" * 80)
    print("🧪 TESTING META-MODEL APPLICATION METHODS DIRECTLY")
    print("=" * 80)
    print()
    
    # Load base-only predictions that already have all required features
    print("Loading existing base-only predictions...")
    base_pred_file = Path("data/ensembl/spliceai_eval/meta_models/inference/predictions/base_only/per_gene/ENSG00000134202_predictions.parquet")
    
    if not base_pred_file.exists():
        print(f"❌ Base predictions not found: {base_pred_file}")
        print("   Run the base-only mode test first!")
        return
    
    df = pl.read_parquet(base_pred_file)
    print(f"✅ Loaded {len(df)} positions with {len(df.columns)} columns")
    print()
    
    # Create workflow instance
    print("Creating workflow instance...")
    config = EnhancedSelectiveInferenceConfig(
        model_path=Path("results/gene_cv_1000_run_15/ablation_study/full/model_full.pkl"),
        inference_mode="meta_only",
        target_genes=["ENSG00000134202"],
        inference_base_dir=Path("data/ensembl/spliceai_eval/meta_models/inference"),
    )
    workflow = EnhancedSelectiveInferenceWorkflow(config)
    print("✅ Workflow created")
    print()
    
    # Test 1: Feature Generation
    print("─" * 80)
    print("TEST 1: _generate_features_for_uncertain_positions()")
    print("─" * 80)
    print()
    
    # Use a subset of positions for testing
    test_positions = df.head(100)
    
    try:
        features = workflow._generate_features_for_uncertain_positions(test_positions)
        
        if features is not None and len(features) > 0:
            print(f"✅ SUCCESS: Generated features for {len(features)} positions")
            print(f"   Feature columns: {len(features.columns)}")
            print(f"   Sample features: {list(features.columns[:10])}")
            
            # Check for k-mer features
            kmer_cols = [c for c in features.columns if 'mer_' in c.lower()]
            print(f"   K-mer features: {len(kmer_cols)}")
            
            # Check for base scores
            base_scores = [c for c in features.columns if c in ['donor_score', 'acceptor_score', 'neither_score']]
            print(f"   Base scores: {base_scores}")
        else:
            print(f"❌ FAILED: No features generated")
            return
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print()
    
    # Test 2: Meta-Model Application
    print("─" * 80)
    print("TEST 2: _apply_meta_model_to_features()")
    print("─" * 80)
    print()
    
    try:
        # Load model
        print("Loading meta-model...")
        model = load_model_with_calibration(
            config.model_path,
            use_calibration=config.use_calibration
        )
        print(f"✅ Model loaded: {type(model)}")
        print()
        
        # Apply model
        print("Applying meta-model to features...")
        predictions = workflow._apply_meta_model_to_features(model, features)
        
        if predictions is not None and len(predictions) > 0:
            print(f"✅ SUCCESS: Generated predictions for {len(predictions)} positions")
            print(f"   Prediction shape: {predictions.shape}")
            print(f"   Expected shape: ({len(features)}, 3)")
            
            # Check if probabilities sum to 1
            row_sums = predictions.sum(axis=1)
            all_valid = np.allclose(row_sums, 1.0, atol=1e-3)
            print(f"   Probabilities sum to 1.0: {'✅ YES' if all_valid else '❌ NO'}")
            
            # Show sample predictions
            print(f"\n   Sample predictions (first 3):")
            for i in range(min(3, len(predictions))):
                print(f"     Position {i}: neither={predictions[i,0]:.4f}, donor={predictions[i,1]:.4f}, acceptor={predictions[i,2]:.4f}")
        else:
            print(f"❌ FAILED: No predictions generated")
            return
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print()
    
    # Test 3: Score Update
    print("─" * 80)
    print("TEST 3: _update_uncertain_positions_with_meta_predictions()")
    print("─" * 80)
    print()
    
    try:
        # Create a mock result DataFrame with default meta scores
        result_df = test_positions.with_columns([
            pl.col('donor_score').alias('donor_meta'),
            pl.col('acceptor_score').alias('acceptor_meta'),
            pl.col('neither_score').alias('neither_meta'),
            pl.lit(0).alias('is_adjusted')
        ])
        
        print(f"Before update:")
        print(f"  Positions with is_adjusted=1: {result_df['is_adjusted'].sum()}")
        
        # Check if meta scores are identical to base scores
        donor_base = result_df['donor_score'].to_numpy()
        donor_meta_before = result_df['donor_meta'].to_numpy()
        n_diff_before = np.sum(np.abs(donor_base - donor_meta_before) > 1e-6)
        print(f"  Positions where donor_meta ≠ donor_score: {n_diff_before}")
        print()
        
        # Apply update
        print("Applying score update...")
        updated_df = workflow._update_uncertain_positions_with_meta_predictions(
            result_df,
            test_positions,
            predictions
        )
        
        print(f"\nAfter update:")
        print(f"  Positions with is_adjusted=1: {updated_df['is_adjusted'].sum()}")
        
        # Check if meta scores are now different
        donor_meta_after = updated_df['donor_meta'].to_numpy()
        n_diff_after = np.sum(np.abs(donor_base - donor_meta_after) > 1e-6)
        print(f"  Positions where donor_meta ≠ donor_score: {n_diff_after}")
        
        if n_diff_after > n_diff_before:
            print(f"\n✅ SUCCESS: Meta scores updated correctly!")
            print(f"   {n_diff_after - n_diff_before} positions recalibrated")
        else:
            print(f"\n❌ FAILED: Meta scores not updated")
            return
            
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print()
    print("=" * 80)
    print("✅ ALL TESTS PASSED - META-MODEL APPLICATION IS WORKING!")
    print("=" * 80)
    print()
    print("The fix successfully implemented:")
    print("  1. ✅ Feature generation from uncertain positions")
    print("  2. ✅ Meta-model prediction on features")
    print("  3. ✅ Score update with meta-model predictions")
    print()
    print("Next step: Re-run full inference comparison tests")
    print("=" * 80)


if __name__ == "__main__":
    main()

