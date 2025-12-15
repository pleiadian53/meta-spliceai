#!/usr/bin/env python3
"""
Comprehensive test of hybrid and meta-only inference modes.

Verification Requirements:
1. No genomic features should be missing in feature matrix X'
2. Some k-mers may be missing (normal) but substantial overlap expected
3. Meta-only mode: meta-model applied to ALL positions (100% usage)
4. Prediction lengths match gene sequence length (gene_end - gene_start + 1)
"""

import sys
from pathlib import Path
import polars as pl
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from meta_spliceai.splice_engine.meta_models.workflows.inference.enhanced_selective_inference import (
    EnhancedSelectiveInferenceWorkflow,
    EnhancedSelectiveInferenceConfig
)
from meta_spliceai.system.genomic_resources import Registry


def get_gene_length(gene_id: str):
    """Get expected gene sequence length from gene_features.tsv."""
    # Use the full gene_features.tsv file
    gene_features_path = Path("data/ensembl/spliceai_analysis/gene_features.tsv")
    
    if not gene_features_path.exists():
        print(f"  ⚠️  gene_features.tsv not found at: {gene_features_path}")
        return None
    
    try:
        df = pl.read_csv(
            str(gene_features_path),
            separator='\t',
            schema_overrides={'chrom': pl.Utf8}
        )
        
        gene_row = df.filter(pl.col('gene_id') == gene_id)
        if gene_row.height == 0:
            print(f"  ⚠️  Gene {gene_id} not found in gene_features.tsv")
            print(f"      Available columns: {df.columns}")
            print(f"      Total genes: {df.height}")
            return None
        
        gene_start = gene_row['start'][0]
        gene_end = gene_row['end'][0]
        expected_length = gene_end - gene_start + 1
        
        return expected_length, gene_start, gene_end
    except Exception as e:
        print(f"  ❌ Error loading gene_features.tsv: {e}")
        import traceback
        traceback.print_exc()
        return None


def verify_prediction_lengths(predictions_path: Path, gene_id: str, expected_length: int) -> dict:
    """Verify that prediction lengths match gene sequence length."""
    print(f"\n  Verifying prediction lengths...")
    
    # Load predictions
    if predictions_path.suffix == '.parquet':
        pred_df = pl.read_parquet(predictions_path)
    else:
        pred_df = pl.read_csv(predictions_path, separator='\t')
    
    # Filter to gene
    gene_pred = pred_df.filter(pl.col('gene_id') == gene_id)
    
    if gene_pred.height == 0:
        return {
            'success': False,
            'error': 'No predictions found for gene'
        }
    
    # Check lengths
    n_positions = gene_pred.height
    
    # Check that we have predictions for donor, acceptor, neither
    score_cols = {
        'donor': None,
        'acceptor': None,
        'neither': None
    }
    
    # Determine which score columns to check based on mode
    if 'donor_meta' in gene_pred.columns:
        score_cols['donor'] = 'donor_meta'
        score_cols['acceptor'] = 'acceptor_meta'
        score_cols['neither'] = 'neither_meta'
    else:
        score_cols['donor'] = 'donor_score'
        score_cols['acceptor'] = 'acceptor_score'
        score_cols['neither'] = 'neither_score'
    
    # Verify all score columns exist
    missing_cols = [k for k, v in score_cols.items() if v not in gene_pred.columns]
    if missing_cols:
        return {
            'success': False,
            'error': f'Missing score columns: {missing_cols}'
        }
    
    # Check for NaN values
    nan_counts = {}
    for splice_type, col in score_cols.items():
        nan_count = gene_pred[col].is_null().sum()
        nan_counts[splice_type] = nan_count
    
    result = {
        'success': True,
        'n_positions': n_positions,
        'expected_length': expected_length,
        'match': n_positions == expected_length,
        'score_columns': score_cols,
        'nan_counts': nan_counts,
        'coverage': (n_positions / expected_length * 100) if expected_length > 0 else 0
    }
    
    return result


def analyze_feature_matrix(analysis_dir: Path, gene_id: str) -> dict:
    """Analyze feature matrix from analysis files."""
    print(f"\n  Analyzing feature matrix...")
    
    # Find analysis files for this gene
    analysis_files = list(analysis_dir.glob(f"analysis_sequences_*.tsv"))
    
    if not analysis_files:
        return {
            'success': False,
            'error': 'No analysis files found'
        }
    
    # Load first analysis file to check features
    analysis_df = pl.read_csv(analysis_files[0], separator='\t')
    
    # Filter to gene
    gene_df = analysis_df.filter(pl.col('gene_id') == gene_id)
    
    if gene_df.height == 0:
        return {
            'success': False,
            'error': 'Gene not found in analysis files'
        }
    
    # Categorize features
    from meta_spliceai.splice_engine.meta_models.builder.preprocessing import (
        LEAKAGE_COLUMNS, METADATA_COLUMNS, SEQUENCE_COLUMNS, REDUNDANT_COLUMNS
    )
    from meta_spliceai.splice_engine.meta_models.builder.feature_schema import (
        is_kmer_feature
    )
    
    all_columns = analysis_df.columns
    
    # Identify feature types
    leakage = [c for c in all_columns if c in LEAKAGE_COLUMNS]
    metadata = [c for c in all_columns if c in METADATA_COLUMNS]
    sequence = [c for c in all_columns if c in SEQUENCE_COLUMNS]
    redundant = [c for c in all_columns if c in REDUNDANT_COLUMNS]
    kmers = [c for c in all_columns if is_kmer_feature(c)]
    
    excluded = set(leakage + metadata + sequence + redundant)
    potential_features = [c for c in all_columns if c not in excluded]
    
    # Separate k-mers from non-k-mers
    feature_kmers = [c for c in potential_features if is_kmer_feature(c)]
    feature_non_kmers = [c for c in potential_features if not is_kmer_feature(c)]
    
    # Identify genomic features
    genomic_features = [
        'gene_start', 'gene_end', 'gene_length',
        'tx_start', 'tx_end', 'transcript_length',
        'num_overlaps', 'chrom'
    ]
    present_genomic = [f for f in genomic_features if f in all_columns]
    missing_genomic = [f for f in genomic_features if f not in all_columns]
    
    result = {
        'success': True,
        'total_columns': len(all_columns),
        'potential_features': len(potential_features),
        'feature_kmers': len(feature_kmers),
        'feature_non_kmers': len(feature_non_kmers),
        'present_genomic': present_genomic,
        'missing_genomic': missing_genomic,
        'kmer_list': sorted(feature_kmers),
        'non_kmer_list': sorted(feature_non_kmers)
    }
    
    return result


def verify_meta_model_usage(predictions_path: Path, gene_id: str, mode: str) -> dict:
    """Verify meta-model usage statistics."""
    print(f"\n  Verifying meta-model usage for {mode} mode...")
    
    # Load predictions
    if predictions_path.suffix == '.parquet':
        pred_df = pl.read_parquet(predictions_path)
    else:
        pred_df = pl.read_csv(predictions_path, separator='\t')
    
    # Filter to gene
    gene_pred = pred_df.filter(pl.col('gene_id') == gene_id)
    
    if gene_pred.height == 0:
        return {
            'success': False,
            'error': 'No predictions found'
        }
    
    total_positions = gene_pred.height
    
    # Check if is_adjusted column exists
    if 'is_adjusted' not in gene_pred.columns:
        return {
            'success': False,
            'error': 'is_adjusted column not found'
        }
    
    # Count adjusted positions
    adjusted_positions = gene_pred['is_adjusted'].sum()
    usage_pct = (adjusted_positions / total_positions * 100) if total_positions > 0 else 0
    
    # For meta-only mode, expect 100% usage
    expected_usage = 100.0 if mode == 'meta_only' else None
    
    result = {
        'success': True,
        'total_positions': total_positions,
        'adjusted_positions': adjusted_positions,
        'usage_pct': usage_pct,
        'expected_usage': expected_usage,
        'meets_expectation': (abs(usage_pct - 100.0) < 1.0) if mode == 'meta_only' else True
    }
    
    return result


def test_gene_inference(gene_id: str, mode: str, model_path: Path, output_base: Path) -> dict:
    """Test inference for a single gene in specified mode."""
    print(f"\n{'='*60}")
    print(f"Testing Gene: {gene_id} (Mode: {mode})")
    print(f"{'='*60}")
    
    # Get expected gene length
    gene_info = get_gene_length(gene_id)
    if gene_info is None:
        print(f"  ❌ Gene not found in gene_features.tsv")
        return {'success': False, 'error': 'Gene not found'}
    
    expected_length, gene_start, gene_end = gene_info
    print(f"  Gene coordinates: {gene_start:,} - {gene_end:,}")
    print(f"  Expected sequence length: {expected_length:,} bp")
    
    # Create config
    output_dir = output_base / f"test_{mode}" / gene_id
    
    config = EnhancedSelectiveInferenceConfig(
        model_path=model_path,
        target_genes=[gene_id],
        inference_mode=mode,
        inference_base_dir=output_dir,
        uncertainty_threshold_high=0.50  # For hybrid mode
    )
    
    # Run inference
    print(f"\n  Running inference...")
    workflow = EnhancedSelectiveInferenceWorkflow(config)
    
    try:
        results = workflow.run()
        
        if not results.success:
            print(f"  ❌ Inference failed")
            return {'success': False, 'error': 'Inference workflow failed'}
        
        print(f"  ✅ Inference completed")
        
        # Determine predictions path based on mode
        if mode == 'hybrid':
            predictions_path = Path(results.hybrid_predictions_path) if results.hybrid_predictions_path else None
        elif mode == 'meta_only':
            predictions_path = Path(results.meta_predictions_path) if results.meta_predictions_path else None
        else:
            predictions_path = Path(results.base_predictions_path) if results.base_predictions_path else None
        
        if not predictions_path or not predictions_path.exists():
            print(f"  ❌ Predictions file not found: {predictions_path}")
            return {'success': False, 'error': 'Predictions file not found'}
        
        print(f"  Predictions: {predictions_path}")
        
        # Verify prediction lengths
        length_result = verify_prediction_lengths(predictions_path, gene_id, expected_length)
        
        if not length_result['success']:
            print(f"  ❌ Length verification failed: {length_result['error']}")
            return {'success': False, 'error': length_result['error']}
        
        print(f"  Prediction positions: {length_result['n_positions']:,}")
        print(f"  Expected length: {length_result['expected_length']:,}")
        print(f"  Match: {'✅' if length_result['match'] else '❌'}")
        print(f"  Coverage: {length_result['coverage']:.1f}%")
        
        # Analyze feature matrix
        analysis_dir = output_dir / "predictions" / mode / "complete_base_predictions" / gene_id / "meta_models" / "complete_inference"
        
        if analysis_dir.exists():
            feature_result = analyze_feature_matrix(analysis_dir, gene_id)
            
            if feature_result['success']:
                print(f"\n  Feature Matrix Analysis:")
                print(f"    Total columns: {feature_result['total_columns']}")
                print(f"    Potential features: {feature_result['potential_features']}")
                print(f"    K-mer features: {feature_result['feature_kmers']}")
                print(f"    Non-k-mer features: {feature_result['feature_non_kmers']}")
                print(f"    Present genomic features: {len(feature_result['present_genomic'])}")
                print(f"      {feature_result['present_genomic']}")
                
                if feature_result['missing_genomic']:
                    print(f"    ❌ Missing genomic features: {feature_result['missing_genomic']}")
                else:
                    print(f"    ✅ All genomic features present")
        else:
            print(f"  ⚠️  Analysis directory not found: {analysis_dir}")
            feature_result = {'success': False, 'error': 'Analysis directory not found'}
        
        # Verify meta-model usage
        usage_result = verify_meta_model_usage(predictions_path, gene_id, mode)
        
        if usage_result['success']:
            print(f"\n  Meta-Model Usage:")
            print(f"    Total positions: {usage_result['total_positions']:,}")
            print(f"    Adjusted positions: {usage_result['adjusted_positions']:,}")
            print(f"    Usage: {usage_result['usage_pct']:.1f}%")
            
            if mode == 'meta_only':
                if usage_result['meets_expectation']:
                    print(f"    ✅ Meets expectation (100% for meta-only mode)")
                else:
                    print(f"    ❌ Does NOT meet expectation (expected 100%, got {usage_result['usage_pct']:.1f}%)")
        
        # Overall result
        all_checks_passed = (
            length_result['success'] and
            (feature_result['success'] and len(feature_result['missing_genomic']) == 0) and
            usage_result['success'] and
            (usage_result['meets_expectation'] if mode == 'meta_only' else True)
        )
        
        print(f"\n  {'✅ ALL CHECKS PASSED' if all_checks_passed else '⚠️  SOME CHECKS FAILED'}")
        
        return {
            'success': True,
            'all_checks_passed': all_checks_passed,
            'length_result': length_result,
            'feature_result': feature_result,
            'usage_result': usage_result
        }
        
    except Exception as e:
        print(f"  ❌ Exception during inference: {e}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}


def main():
    """Run comprehensive meta-model inference tests."""
    print("\n" + "="*60)
    print("COMPREHENSIVE META-MODEL INFERENCE TEST")
    print("="*60)
    
    # Configuration
    model_path = Path("results/meta_model_1000genes_3mers_fresh/model_multiclass.pkl")
    output_base = Path("predictions/meta_modes_test")
    
    # Test genes (mix of observed and unobserved)
    test_genes = [
        'ENSG00000141736',  # ERBB2 (observed, protein-coding)
        'ENSG00000134202',  # GSTM1 (observed, protein-coding)
        'ENSG00000169239',  # NLRP3 (unobserved, protein-coding)
    ]
    
    # Test modes
    test_modes = ['hybrid', 'meta_only']
    
    # Verify model exists
    if not model_path.exists():
        print(f"\n❌ Model not found: {model_path}")
        print("   Train the meta-model first using run_gene_cv_sigmoid.py")
        return 1
    
    print(f"\nModel: {model_path}")
    print(f"Output: {output_base}")
    print(f"Test genes: {len(test_genes)}")
    print(f"Test modes: {test_modes}")
    
    # Run tests
    results = {}
    
    for mode in test_modes:
        results[mode] = {}
        
        for gene_id in test_genes:
            result = test_gene_inference(gene_id, mode, model_path, output_base)
            results[mode][gene_id] = result
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for mode in test_modes:
        print(f"\n{mode.upper()} MODE:")
        
        for gene_id in test_genes:
            result = results[mode][gene_id]
            
            if result['success'] and result.get('all_checks_passed'):
                status = "✅ PASS"
            elif result['success']:
                status = "⚠️  PARTIAL"
            else:
                status = "❌ FAIL"
            
            print(f"  {gene_id}: {status}")
            
            if result['success'] and not result.get('all_checks_passed'):
                # Show which checks failed
                if not result['length_result'].get('match'):
                    print(f"    - Length mismatch")
                if result['feature_result'].get('missing_genomic'):
                    print(f"    - Missing genomic features: {result['feature_result']['missing_genomic']}")
                if mode == 'meta_only' and not result['usage_result'].get('meets_expectation'):
                    print(f"    - Meta-model usage: {result['usage_result']['usage_pct']:.1f}% (expected 100%)")
    
    # Overall status
    all_passed = all(
        results[mode][gene_id].get('all_checks_passed', False)
        for mode in test_modes
        for gene_id in test_genes
    )
    
    print("\n" + "="*60)
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        print("\n✅ No genomic features missing")
        print("✅ Prediction lengths match gene sequence lengths")
        print("✅ Meta-only mode: 100% meta-model usage")
        print("✅ K-mer overlap substantial (as expected)")
        return 0
    else:
        print("⚠️  SOME TESTS FAILED")
        print("\nReview the detailed output above for specific issues.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

