#!/usr/bin/env python3
"""
Test all 3 inference modes on diverse gene set (protein-coding + lncRNA)
Verify metadata preservation and prediction success

Created: 2025-10-28
"""

import sys
from pathlib import Path
import polars as pl
import json

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from meta_spliceai.splice_engine.meta_models.workflows.inference.enhanced_selective_inference import (
    EnhancedSelectiveInferenceWorkflow,
    EnhancedSelectiveInferenceConfig
)


# Expected metadata features
METADATA_FEATURES = [
    'is_uncertain', 'is_low_confidence', 'is_high_entropy',
    'is_low_discriminability', 'max_confidence', 'score_spread',
    'score_entropy', 'confidence_category', 'predicted_type_base'
]

# Expected core prediction columns
CORE_COLUMNS = [
    'gene_id', 'position', 
    'donor_score', 'acceptor_score', 'neither_score',
    'donor_meta', 'acceptor_meta', 'neither_meta',
    'splice_type', 'is_adjusted'
]


def select_test_genes():
    """
    Select diverse test genes from gene_features.tsv
    
    Strategy:
    - 3 protein-coding genes (varied complexity)
    - 3 lncRNA genes (varied length)
    - Mix of observed/unobserved in training data
    """
    gene_features_path = project_root / 'data/ensembl/spliceai_analysis/gene_features.tsv'
    
    if not gene_features_path.exists():
        print(f"❌ Gene features not found: {gene_features_path}")
        return None
    
    # Load gene features
    df = pl.read_csv(
        gene_features_path,
        separator='\t',
        schema_overrides={'chrom': pl.String}
    )
    
    print("="*80)
    print("SELECTING DIVERSE TEST GENES")
    print("="*80)
    
    # Filter for genes with splice sites
    splice_sites_path = project_root / 'data/ensembl/splice_sites_enhanced.tsv'
    ss_df = pl.read_csv(
        splice_sites_path,
        separator='\t',
        schema_overrides={'chrom': pl.String}
    )
    
    genes_with_sites = ss_df.select('gene_id').unique()
    df = df.join(genes_with_sites, on='gene_id', how='inner')
    
    print(f"Total genes with splice sites: {len(df):,}")
    
    # Get gene_type counts
    gene_type_counts = df.group_by('gene_type').agg(pl.len().alias('count')).sort('count', descending=True)
    print(f"\nTop gene types:")
    for row in gene_type_counts.head(10).iter_rows(named=True):
        print(f"  {row['gene_type']:30s} {row['count']:,}")
    
    selected_genes = {}
    
    # 1. Select 3 protein-coding genes (different sizes)
    pc_genes = df.filter(pl.col('gene_type') == 'protein_coding').sort('gene_length')
    
    if len(pc_genes) >= 3:
        # Small, medium, large
        small_idx = len(pc_genes) // 4
        medium_idx = len(pc_genes) // 2
        large_idx = 3 * len(pc_genes) // 4
        
        selected_genes['protein_coding'] = [
            {
                'gene_id': pc_genes[small_idx, 'gene_id'],
                'gene_name': pc_genes[small_idx, 'gene_name'],
                'length': pc_genes[small_idx, 'gene_length'],
                'size_class': 'small'
            },
            {
                'gene_id': pc_genes[medium_idx, 'gene_id'],
                'gene_name': pc_genes[medium_idx, 'gene_name'],
                'length': pc_genes[medium_idx, 'gene_length'],
                'size_class': 'medium'
            },
            {
                'gene_id': pc_genes[large_idx, 'gene_id'],
                'gene_name': pc_genes[large_idx, 'gene_name'],
                'length': pc_genes[large_idx, 'gene_length'],
                'size_class': 'large'
            }
        ]
    
    # 2. Select 3 lncRNA genes (different lengths)
    lnc_genes = df.filter(pl.col('gene_type') == 'lncRNA').sort('gene_length')
    
    if len(lnc_genes) >= 3:
        small_idx = len(lnc_genes) // 4
        medium_idx = len(lnc_genes) // 2
        large_idx = 3 * len(lnc_genes) // 4
        
        selected_genes['lncRNA'] = [
            {
                'gene_id': lnc_genes[small_idx, 'gene_id'],
                'gene_name': lnc_genes[small_idx, 'gene_name'],
                'length': lnc_genes[small_idx, 'gene_length'],
                'size_class': 'small'
            },
            {
                'gene_id': lnc_genes[medium_idx, 'gene_id'],
                'gene_name': lnc_genes[medium_idx, 'gene_name'],
                'length': lnc_genes[medium_idx, 'gene_length'],
                'size_class': 'medium'
            },
            {
                'gene_id': lnc_genes[large_idx, 'gene_id'],
                'gene_name': lnc_genes[large_idx, 'gene_name'],
                'length': lnc_genes[large_idx, 'gene_length'],
                'size_class': 'large'
            }
        ]
    
    return selected_genes


def verify_metadata_preservation(predictions_file: Path) -> dict:
    """
    Verify that metadata features are preserved in output
    
    Returns dict with:
    - present: list of present metadata features
    - missing: list of missing metadata features
    - stats: statistics for each present feature
    """
    if not predictions_file.exists():
        return {'error': f'File not found: {predictions_file}'}
    
    try:
        df = pl.read_parquet(predictions_file)
    except Exception as e:
        return {'error': f'Failed to read file: {e}'}
    
    result = {
        'total_positions': len(df),
        'total_columns': len(df.columns),
        'present': [],
        'missing': [],
        'stats': {}
    }
    
    for feature in METADATA_FEATURES:
        if feature in df.columns:
            result['present'].append(feature)
            
            # Get statistics based on type
            if df[feature].dtype == pl.Boolean:
                true_count = df[feature].sum()
                pct = (true_count / len(df) * 100) if len(df) > 0 else 0
                result['stats'][feature] = {
                    'type': 'boolean',
                    'true_count': true_count,
                    'percentage': pct
                }
            elif df[feature].dtype in [pl.Float64, pl.Float32]:
                result['stats'][feature] = {
                    'type': 'float',
                    'mean': float(df[feature].mean()),
                    'std': float(df[feature].std()),
                    'min': float(df[feature].min()),
                    'max': float(df[feature].max())
                }
            elif df[feature].dtype in [pl.String, pl.Utf8]:
                value_counts = df[feature].value_counts().sort('count', descending=True)
                result['stats'][feature] = {
                    'type': 'string',
                    'unique_values': len(value_counts),
                    'top_values': [
                        {'value': row[0], 'count': row[1]}
                        for row in value_counts.head(3).iter_rows()
                    ]
                }
            else:
                result['stats'][feature] = {
                    'type': str(df[feature].dtype),
                    'dtype': str(df[feature].dtype)
                }
        else:
            result['missing'].append(feature)
    
    return result


def run_test_mode(gene_id: str, gene_info: dict, mode: str) -> dict:
    """Run inference for a single gene in a specific mode"""
    
    model_path = project_root / 'results/meta_model_1000genes_3mers_fresh/model_multiclass.pkl'
    
    gene_name = gene_info.get('gene_name') or 'N/A'
    print(f"\n{'='*80}")
    print(f"Testing {mode.upper()} mode on {gene_id} ({gene_name})")
    print(f"  Gene type: {gene_info.get('gene_type', 'unknown')}")
    print(f"  Size class: {gene_info['size_class']}")
    print(f"  Length: {gene_info['length']:,} bp")
    print(f"{'='*80}")
    
    try:
        # Create config (OutputManager will handle directory structure)
        config = EnhancedSelectiveInferenceConfig(
            target_genes=[gene_id],
            model_path=model_path,
            inference_mode=mode,
            # Use test output to separate from production runs
            output_name=f'diverse_test',  # OutputManager will use this to determine test mode
            uncertainty_threshold_low=0.02,
            uncertainty_threshold_high=0.50,
            use_timestamped_output=False
        )
        
        # Run workflow
        workflow = EnhancedSelectiveInferenceWorkflow(config)
        results = workflow.run_incremental()
        
        # Use workflow's OutputManager to get correct output path
        # With output_name='diverse_test', it's automatically detected as a test run
        # Path: predictions/{mode}/tests/{gene_id}/combined_predictions.parquet
        gene_paths = workflow.output_manager.get_gene_output_paths(gene_id)
        predictions_file = gene_paths.predictions_file
        
        if not predictions_file.exists():
            return {
                'status': 'failed',
                'error': f'Output file not found: {predictions_file}',
                'gene_id': gene_id,
                'mode': mode
            }
        
        # Verify metadata
        metadata_check = verify_metadata_preservation(predictions_file)
        
        return {
            'status': 'success',
            'gene_id': gene_id,
            'gene_name': gene_info.get('gene_name') or 'N/A',
            'mode': mode,
            'predictions_file': str(predictions_file),
            'metadata_check': metadata_check
        }
        
    except Exception as e:
        return {
            'status': 'failed',
            'error': str(e),
            'gene_id': gene_id,
            'mode': mode
        }


def main():
    """Run comprehensive test on diverse genes"""
    
    print("\n" + "="*80)
    print("COMPREHENSIVE INFERENCE TEST: DIVERSE GENES + METADATA VERIFICATION")
    print("="*80)
    
    # Select test genes
    test_genes = select_test_genes()
    
    if not test_genes:
        print("\n❌ Failed to select test genes")
        return 1
    
    # Print selected genes
    print("\n" + "="*80)
    print("SELECTED TEST GENES")
    print("="*80)
    
    for gene_type, genes in test_genes.items():
        print(f"\n{gene_type.upper()}:")
        for gene in genes:
            gene_name = gene.get('gene_name') or 'N/A'
            print(f"  {gene['gene_id']:20s} {str(gene_name):15s} "
                  f"{gene['size_class']:8s} {gene['length']:,} bp")
    
    # Run tests
    modes = ['base_only', 'hybrid', 'meta_only']
    results = []
    
    for gene_type, genes in test_genes.items():
        for gene in genes:
            gene['gene_type'] = gene_type
            
            for mode in modes:
                result = run_test_mode(gene['gene_id'], gene, mode)
                results.append(result)
                
                # Print immediate result
                if result['status'] == 'success':
                    mc = result['metadata_check']
                    print(f"\n✅ SUCCESS")
                    print(f"   Positions: {mc['total_positions']:,}")
                    print(f"   Metadata: {len(mc['present'])}/{len(METADATA_FEATURES)} features present")
                    if mc['missing']:
                        print(f"   Missing: {mc['missing']}")
                else:
                    print(f"\n❌ FAILED: {result.get('error', 'Unknown error')}")
    
    # Summary report
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    
    success_count = sum(1 for r in results if r['status'] == 'success')
    total_count = len(results)
    
    print(f"\nOverall: {success_count}/{total_count} tests passed "
          f"({success_count/total_count*100:.1f}%)")
    
    # Group by mode
    for mode in modes:
        mode_results = [r for r in results if r['mode'] == mode]
        mode_success = sum(1 for r in mode_results if r['status'] == 'success')
        print(f"\n{mode.upper()}: {mode_success}/{len(mode_results)} passed")
        
        for r in mode_results:
            if r['status'] == 'success':
                mc = r['metadata_check']
                metadata_status = f"{len(mc['present'])}/{len(METADATA_FEATURES)} metadata"
                print(f"  ✅ {r['gene_id']} - {mc['total_positions']:,} positions, {metadata_status}")
            else:
                print(f"  ❌ {r['gene_id']} - {r.get('error', 'Failed')}")
    
    # Metadata preservation summary
    print("\n" + "="*80)
    print("METADATA PRESERVATION SUMMARY")
    print("="*80)
    
    successful_results = [r for r in results if r['status'] == 'success']
    
    if successful_results:
        # Aggregate metadata presence across all tests
        metadata_presence = {feature: 0 for feature in METADATA_FEATURES}
        
        for r in successful_results:
            for feature in r['metadata_check']['present']:
                metadata_presence[feature] += 1
        
        print(f"\nMetadata feature presence across {len(successful_results)} successful tests:")
        for feature, count in sorted(metadata_presence.items(), key=lambda x: -x[1]):
            pct = (count / len(successful_results) * 100)
            status = "✅" if pct == 100 else "⚠️" if pct > 0 else "❌"
            print(f"  {status} {feature:30s} {count}/{len(successful_results)} ({pct:.0f}%)")
    
    # Save detailed results
    results_file = project_root / 'results' / 'diverse_genes_test_results.json'
    results_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n📁 Detailed results saved to: {results_file}")
    
    # Return exit code
    if success_count == total_count:
        print("\n🎉 ALL TESTS PASSED!")
        return 0
    else:
        print(f"\n⚠️  {total_count - success_count} tests failed")
        return 1


if __name__ == '__main__':
    sys.exit(main())

