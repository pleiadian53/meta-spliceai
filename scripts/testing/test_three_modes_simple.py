#!/usr/bin/env python
"""
Simple, direct test of all 3 modes using the working command-line approach.

NOTE: This script uses the main_inference_workflow.py CLI, which now internally
uses OutputManager for consistent path management. Output paths are now:
  predictions/{mode}/tests/{gene_id}/combined_predictions.parquet
"""
import subprocess
import sys
from pathlib import Path

def run_inference(gene_id: str, mode: str) -> dict:
    """Run inference for a gene in a specific mode."""
    
    cmd = [
        'conda', 'run', '-n', 'surveyor', '--no-capture-output',
        'python', '-m', 'meta_spliceai.splice_engine.meta_models.workflows.inference.main_inference_workflow',
        '--genes', gene_id,
        '--model-path', 'results/meta_model_1000genes_3mers_fresh/model_multiclass.pkl',
        '--inference-mode', mode,
        '--output-dir', f'predictions/test_{mode}'
    ]
    
    print(f"\n{'='*80}")
    print(f"Testing {gene_id} in {mode.upper().replace('_', '-')} mode")
    print(f"{'='*80}")
    
    try:
        result = subprocess.run(
            cmd,
            cwd='/Users/pleiadian53/work/meta-spliceai',
            capture_output=True,
            text=True,
            timeout=300  # 5 minutes
        )
        
        if result.returncode == 0:
            print(f"✅ SUCCESS")
            # Check for output file
            if mode == 'base_only':
                pred_file = Path(f'predictions/test_{mode}/{gene_id}/predictions/base_only/combined_predictions.parquet')
            else:
                pred_file = Path(f'predictions/test_{mode}/{gene_id}/predictions/{mode}/combined_predictions.parquet')
            
            if pred_file.exists():
                print(f"   Output file: {pred_file}")
                return {'success': True, 'gene': gene_id, 'mode': mode}
            else:
                print(f"   ⚠️  Output file not found: {pred_file}")
                return {'success': False, 'gene': gene_id, 'mode': mode, 'error': 'Output file not found'}
        else:
            print(f"❌ FAILED (exit code: {result.returncode})")
            # Look for specific errors in output
            error_lines = [line for line in result.stderr.split('\n') if 'error' in line.lower() or 'failed' in line.lower()]
            if error_lines:
                print(f"   Errors: {error_lines[:3]}")
            return {'success': False, 'gene': gene_id, 'mode': mode, 'error': result.stderr[-500:]}
            
    except subprocess.TimeoutExpired:
        print(f"❌ TIMEOUT")
        return {'success': False, 'gene': gene_id, 'mode': mode, 'error': 'Timeout'}
    except Exception as e:
        print(f"❌ EXCEPTION: {e}")
        return {'success': False, 'gene': gene_id, 'mode': mode, 'error': str(e)}

def main():
    print("\n" + "="*80)
    print("SIMPLE 3-MODE INFERENCE TEST")
    print("="*80)
    
    test_genes = ['ENSG00000134202']  # Start with one small gene
    modes = ['base_only', 'hybrid', 'meta_only']
    
    results = []
    for mode in modes:
        for gene_id in test_genes:
            result = run_inference(gene_id, mode)
            results.append(result)
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    success_count = sum(1 for r in results if r['success'])
    total = len(results)
    
    print(f"Passed: {success_count}/{total}")
    
    for result in results:
        status = "✅" if result['success'] else "❌"
        print(f"{status} {result['gene']} - {result['mode']}")
    
    if success_count == total:
        print("\n✅ ALL TESTS PASSED")
        sys.exit(0)
    else:
        print(f"\n⚠️  {total - success_count} tests failed")
        sys.exit(1)

if __name__ == '__main__':
    main()

