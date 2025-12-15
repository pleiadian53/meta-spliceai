#!/usr/bin/env python
"""
Comprehensive test of all 3 inference modes.

Tests:
1. Base-only mode
2. Hybrid mode  
3. Meta-only mode

For each mode, verifies:
- No fallback logic is triggered
- Feature alignment works correctly
- Predictions are generated successfully
- Output files are created
"""

import sys
from pathlib import Path
import polars as pl
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from meta_spliceai.splice_engine.meta_models.workflows.inference.enhanced_selective_inference import (
    EnhancedSelectiveInferenceWorkflow,
    EnhancedSelectiveInferenceConfig
)

def verify_splice_sites_complete():
    """Verify splice sites file is complete before testing."""
    splice_sites_file = project_root / 'data/ensembl/splice_sites_enhanced.tsv'
    
    if not splice_sites_file.exists():
        print(f"❌ Splice sites file not found: {splice_sites_file}")
        return False
    
    # Check file size
    file_size_mb = splice_sites_file.stat().st_size / (1024 * 1024)
    
    if file_size_mb < 100:  # Should be ~193MB
        print(f"❌ Splice sites file too small: {file_size_mb:.1f} MB (expected ~193 MB)")
        return False
    
    # Check line count
    with open(splice_sites_file) as f:
        line_count = sum(1 for _ in f)
    
    if line_count < 2_000_000:  # Should be ~2.8M
        print(f"❌ Splice sites file has too few lines: {line_count:,} (expected ~2.8M)")
        return False
    
    print(f"✅ Splice sites file verified: {file_size_mb:.1f} MB, {line_count:,} lines")
    return True

def test_mode(mode: str, test_genes: list, model_path: Path) -> dict:
    """
    Test a specific inference mode.
    
    OutputManager will automatically handle directory structure.
    With output_name containing 'test', predictions go to: predictions/{mode}/tests/{gene_id}/
    
    Returns:
        dict with test results
    """
    print(f"\n{'='*80}")
    print(f"TESTING {mode.upper().replace('_', '-')} MODE")
    print(f"{'='*80}")
    
    results = {
        'mode': mode,
        'genes': [],
        'success_count': 0,
        'failure_count': 0,
        'fallback_triggered': False,
        'feature_mismatch': False,
        'errors': []
    }
    
    for gene_id in test_genes:
        print(f"\n--- Testing gene: {gene_id} ---")
        
        try:
            # Create config
            # OutputManager automatically creates test directory when 'test' in output_name
            config = EnhancedSelectiveInferenceConfig(
                target_genes=[gene_id],
                model_path=model_path,
                inference_mode=mode,
                output_name=f'test_comprehensive',  # 'test' triggers test mode
                uncertainty_threshold_low=0.02,
                uncertainty_threshold_high=0.50,  # Lower to enable meta-model
                use_timestamped_output=False  # Overwrite for clean testing
            )
            
            # Run inference
            workflow = EnhancedSelectiveInferenceWorkflow(config)
            result = workflow.run_incremental()
            
            # Check results
            if hasattr(result, 'success') and result.success:
                print(f"  ✅ {gene_id}: SUCCESS")
                
                # Verify output files
                if mode == 'base_only':
                    pred_file = result.base_predictions_path
                else:
                    pred_file = result.hybrid_predictions_path or result.base_predictions_path
                
                if pred_file and Path(pred_file).exists():
                    # Load and verify predictions
                    df = pl.read_parquet(pred_file)
                    print(f"     Predictions: {len(df):,} positions")
                    
                    # Verify required columns
                    required_cols = ['gene_id', 'position', 'donor_score', 'acceptor_score', 'neither_score']
                    missing_cols = [c for c in required_cols if c not in df.columns]
                    
                    if missing_cols:
                        print(f"     ⚠️  Missing columns: {missing_cols}")
                        results['errors'].append(f"{gene_id}: Missing columns {missing_cols}")
                    else:
                        print(f"     ✅ All required columns present")
                        results['success_count'] += 1
                else:
                    print(f"     ⚠️  Output file not found: {pred_file}")
                    results['errors'].append(f"{gene_id}: Output file not found")
                    results['failure_count'] += 1
            else:
                print(f"  ❌ {gene_id}: FAILED")
                results['failure_count'] += 1
                if hasattr(result, 'error'):
                    results['errors'].append(f"{gene_id}: {result.error}")
                    print(f"     Error: {result.error}")
                
        except Exception as e:
            print(f"  ❌ {gene_id}: EXCEPTION - {e}")
            results['failure_count'] += 1
            results['errors'].append(f"{gene_id}: {str(e)}")
            
            # Check for specific error types
            error_str = str(e).lower()
            if 'fallback' in error_str or 'regenerat' in error_str:
                results['fallback_triggered'] = True
                print(f"     🚨 FALLBACK LOGIC DETECTED")
            
            if 'feature' in error_str and 'mismatch' in error_str:
                results['feature_mismatch'] = True
                print(f"     🚨 FEATURE MISMATCH DETECTED")
        
        results['genes'].append(gene_id)
    
    return results

def main():
    print("\n" + "="*80)
    print("COMPREHENSIVE INFERENCE WORKFLOW TEST")
    print("="*80)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Project: {project_root}")
    
    # Pre-flight checks
    print("\n" + "="*80)
    print("PRE-FLIGHT CHECKS")
    print("="*80)
    
    if not verify_splice_sites_complete():
        print("\n❌ Pre-flight checks failed. Please regenerate splice sites file.")
        sys.exit(1)
    
    # Test configuration
    test_genes = [
        'ENSG00000141736',  # ERBB2 - large gene, 528 splice sites
        'ENSG00000134202',  # GSTM3 - medium gene, 58 splice sites
        'ENSG00000169239',  # CA5B - large gene, 46 splice sites
    ]
    
    model_path = project_root / 'results/meta_model_1000genes_3mers_fresh/model_multiclass.pkl'
    
    # Verify model exists
    if not model_path.exists():
        print(f"\n❌ Model not found: {model_path}")
        sys.exit(1)
    
    print(f"\n✅ Model found: {model_path}")
    print(f"✅ Output: OutputManager will create predictions/{mode}/tests/{gene_id}/")
    print(f"✅ Test genes: {len(test_genes)}")
    for gene in test_genes:
        print(f"   - {gene}")
    
    # Run tests for all modes
    all_results = {}
    modes = ['base_only', 'hybrid', 'meta_only']
    
    for mode in modes:
        # OutputManager handles directory structure automatically
        results = test_mode(mode, test_genes, model_path)
        all_results[mode] = results
    
    # Summary report
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    total_success = 0
    total_failure = 0
    critical_issues = []
    
    for mode, results in all_results.items():
        success_rate = (results['success_count'] / len(test_genes) * 100) if test_genes else 0
        
        print(f"\n{mode.upper().replace('_', '-')} MODE:")
        print(f"  Success: {results['success_count']}/{len(test_genes)} ({success_rate:.0f}%)")
        print(f"  Failures: {results['failure_count']}/{len(test_genes)}")
        
        if results['fallback_triggered']:
            print(f"  🚨 CRITICAL: Fallback logic was triggered!")
            critical_issues.append(f"{mode}: Fallback logic triggered")
        
        if results['feature_mismatch']:
            print(f"  🚨 CRITICAL: Feature mismatch detected!")
            critical_issues.append(f"{mode}: Feature mismatch")
        
        if results['errors']:
            print(f"  Errors:")
            for error in results['errors'][:5]:  # Show first 5
                print(f"    - {error}")
        
        total_success += results['success_count']
        total_failure += results['failure_count']
    
    # Final verdict
    print("\n" + "="*80)
    print("FINAL VERDICT")
    print("="*80)
    
    total_tests = len(modes) * len(test_genes)
    overall_success_rate = (total_success / total_tests * 100) if total_tests else 0
    
    print(f"Overall: {total_success}/{total_tests} tests passed ({overall_success_rate:.0f}%)")
    
    if critical_issues:
        print(f"\n🚨 CRITICAL ISSUES DETECTED:")
        for issue in critical_issues:
            print(f"  - {issue}")
        print(f"\n❌ TEST SUITE FAILED")
        sys.exit(1)
    elif total_failure > 0:
        print(f"\n⚠️  SOME TESTS FAILED ({total_failure} failures)")
        sys.exit(1)
    else:
        print(f"\n✅ ALL TESTS PASSED!")
        sys.exit(0)

if __name__ == '__main__':
    main()

