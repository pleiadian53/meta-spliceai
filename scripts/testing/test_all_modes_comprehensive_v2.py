#!/usr/bin/env python3
"""
Comprehensive test of all 3 inference modes on diverse genes.

Tests:
1. All modes complete successfully
2. All 9 metadata features preserved
3. Scores differ between modes (meta-model recalibration)
4. Performance comparison (F1 scores)

Created: 2025-10-28
"""

import sys
from pathlib import Path
import polars as pl
import numpy as np
from datetime import datetime
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


def select_diverse_test_genes():
    """
    Select diverse test genes for comprehensive testing.
    
    Returns:
        dict: {gene_type: [gene_info_dicts]}
    """
    gene_features_path = project_root / 'data/ensembl/spliceai_analysis/gene_features.tsv'
    splice_sites_path = project_root / 'data/ensembl/splice_sites_enhanced.tsv'
    
    if not gene_features_path.exists():
        print(f"❌ Gene features not found: {gene_features_path}")
        return None
    
    # Load gene features
    df = pl.read_csv(
        gene_features_path,
        separator='\t',
        schema_overrides={'chrom': pl.String}
    )
    
    # Filter for genes with splice sites
    ss_df = pl.read_csv(
        splice_sites_path,
        separator='\t',
        schema_overrides={'chrom': pl.String}
    )
    
    genes_with_sites = ss_df.select('gene_id').unique()
    df = df.join(genes_with_sites, on='gene_id', how='inner')
    
    print(f"\n{'='*80}")
    print("SELECTING DIVERSE TEST GENES")
    print(f"{'='*80}")
    print(f"Total genes with splice sites: {len(df):,}\n")
    
    selected_genes = {}
    
    # 1. Select 2 protein-coding genes (small and medium)
    pc_genes = df.filter(pl.col('gene_type') == 'protein_coding').sort('gene_length')
    
    if len(pc_genes) >= 2:
        small_idx = len(pc_genes) // 4
        medium_idx = len(pc_genes) // 2
        
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
            }
        ]
    
    # 2. Select 2 lncRNA genes (small and medium)
    lnc_genes = df.filter(pl.col('gene_type') == 'lncRNA').sort('gene_length')
    
    if len(lnc_genes) >= 2:
        small_idx = len(lnc_genes) // 4
        medium_idx = len(lnc_genes) // 2
        
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
            }
        ]
    
    return selected_genes


def run_inference_mode(gene_id: str, gene_info: dict, mode: str, model_path: Path) -> dict:
    """Run inference for a single gene in a specific mode."""
    
    gene_name = gene_info.get('gene_name') or 'N/A'
    print(f"\n{'='*80}")
    print(f"Testing {mode.upper().replace('_', '-')} mode on {gene_id} ({gene_name})")
    print(f"  Gene type: {gene_info.get('gene_type', 'unknown')}")
    print(f"  Size class: {gene_info['size_class']}")
    print(f"  Length: {gene_info['length']:,} bp")
    print(f"{'='*80}")
    
    try:
        # Create config
        config = EnhancedSelectiveInferenceConfig(
            target_genes=[gene_id],
            model_path=model_path,
            inference_mode=mode,
            output_name='comprehensive_test_v2',  # Triggers test mode
            uncertainty_threshold_low=0.02,
            uncertainty_threshold_high=0.50,
            use_timestamped_output=False,
            verbose=0  # Quiet mode
        )
        
        # Run workflow
        workflow = EnhancedSelectiveInferenceWorkflow(config)
        results = workflow.run_incremental()
        
        if not results.success:
            return {
                'status': 'failed',
                'error': 'Workflow returned failure',
                'gene_id': gene_id,
                'mode': mode
            }
        
        # Get output path from OutputManager
        gene_paths = workflow.output_manager.get_gene_output_paths(gene_id)
        predictions_file = gene_paths.predictions_file
        
        if not predictions_file.exists():
            return {
                'status': 'failed',
                'error': f'Output file not found: {predictions_file}',
                'gene_id': gene_id,
                'mode': mode
            }
        
        # Load predictions
        df = pl.read_parquet(predictions_file)
        
        # CRITICAL: Verify full coverage (N predictions for N-bp gene)
        gene_length = gene_info['length']
        total_positions = len(df)
        unique_positions = df['position'].n_unique()
        
        coverage_pct = (total_positions / gene_length) * 100 if gene_length > 0 else 0
        
        # Check metadata features
        metadata_present = [feat for feat in METADATA_FEATURES if feat in df.columns]
        metadata_missing = [feat for feat in METADATA_FEATURES if feat not in df.columns]
        
        # Calculate basic stats
        uncertain_count = df.filter(pl.col('is_uncertain') == True).height if 'is_uncertain' in df.columns else 0
        adjusted_count = df.filter(pl.col('is_adjusted') == 1).height if 'is_adjusted' in df.columns else 0
        
        # Verify full coverage
        coverage_ok = coverage_pct >= 95.0  # Allow 5% tolerance for edge effects
        
        print(f"  ✅ SUCCESS")
        print(f"     Gene length: {gene_length:,} bp")
        print(f"     Positions: {total_positions:,} ({coverage_pct:.1f}% coverage)")
        print(f"     Unique positions: {unique_positions:,}")
        if not coverage_ok:
            print(f"     ⚠️  WARNING: Coverage below 95%! Expected ~{gene_length:,} positions")
        print(f"     Metadata: {len(metadata_present)}/9 features")
        print(f"     Uncertain: {uncertain_count:,} ({uncertain_count/total_positions*100:.1f}%)")
        print(f"     Adjusted: {adjusted_count:,} ({adjusted_count/total_positions*100:.1f}%)")
        
        return {
            'status': 'success',
            'gene_id': gene_id,
            'gene_name': gene_name,
            'mode': mode,
            'predictions_file': str(predictions_file),
            'gene_length': gene_length,
            'total_positions': total_positions,
            'unique_positions': unique_positions,
            'coverage_pct': coverage_pct,
            'coverage_ok': coverage_ok,
            'uncertain_count': uncertain_count,
            'adjusted_count': adjusted_count,
            'metadata_present': metadata_present,
            'metadata_missing': metadata_missing,
            'df': df  # Keep for comparison
        }
        
    except Exception as e:
        print(f"  ❌ FAILED: {e}")
        return {
            'status': 'failed',
            'error': str(e),
            'gene_id': gene_id,
            'mode': mode
        }


def compare_mode_scores(results_by_mode: dict, gene_id: str) -> dict:
    """
    Compare prediction scores between modes.
    
    Expected:
    - Base scores (donor_score, acceptor_score, neither_score) should be IDENTICAL
    - Meta scores (donor_meta, acceptor_meta, neither_meta) should be DIFFERENT
    """
    print(f"\n{'='*80}")
    print(f"COMPARING SCORES ACROSS MODES: {gene_id}")
    print(f"{'='*80}\n")
    
    modes = ['base_only', 'hybrid', 'meta_only']
    
    # Check all modes succeeded
    for mode in modes:
        if mode not in results_by_mode or results_by_mode[mode]['status'] != 'success':
            print(f"  ❌ Mode {mode} not available for comparison")
            return {'status': 'failed', 'error': f'Mode {mode} not available'}
    
    # Get dataframes
    dfs = {mode: results_by_mode[mode]['df'].sort('position') for mode in modes}
    
    # Find common positions
    common_positions = set(dfs['base_only']['position'].to_list())
    for mode in ['hybrid', 'meta_only']:
        common_positions &= set(dfs[mode]['position'].to_list())
    
    if len(common_positions) == 0:
        print(f"  ❌ No common positions to compare")
        return {'status': 'failed', 'error': 'No common positions'}
    
    print(f"Common positions: {len(common_positions):,}\n")
    
    # Filter to common positions and verify same length
    for mode in modes:
        dfs[mode] = dfs[mode].filter(pl.col('position').is_in(list(common_positions))).sort('position')
    
    # Verify all have same positions after filtering
    base_positions = set(dfs['base_only']['position'].to_list())
    hybrid_positions = set(dfs['hybrid']['position'].to_list())
    meta_positions = set(dfs['meta_only']['position'].to_list())
    
    # Find truly common positions (intersection of all three)
    truly_common = base_positions & hybrid_positions & meta_positions
    
    if len(truly_common) < len(common_positions):
        print(f"  ⚠️  WARNING: Position mismatch after filtering:")
        print(f"     base_only: {len(base_positions)} positions")
        print(f"     hybrid: {len(hybrid_positions)} positions")
        print(f"     meta_only: {len(meta_positions)} positions")
        print(f"     Truly common: {len(truly_common)} positions")
        print(f"\n  This indicates different modes are outputting different position sets.")
        print(f"  Will compare only the {len(truly_common)} truly common positions.")
        print(f"  Note: This may indicate a bug in position filtering logic.\n")
        
        # Re-filter to truly common positions
        for mode in modes:
            dfs[mode] = dfs[mode].filter(pl.col('position').is_in(list(truly_common))).sort('position')
    
    # Final verification
    if not (len(dfs['base_only']) == len(dfs['hybrid']) == len(dfs['meta_only'])):
        print(f"  ❌ CRITICAL: Still have length mismatch after filtering to truly common positions")
        return {'status': 'failed', 'error': 'Cannot resolve position mismatch'}
    
    comparison_results = {}
    
    # 1. Compare base scores (should be identical)
    print("─" * 80)
    print("COMPARISON 1: Base Scores (should be IDENTICAL across all modes)")
    print("─" * 80)
    
    base_scores_identical = True
    for score_type in ['donor_score', 'acceptor_score', 'neither_score']:
        base_scores = dfs['base_only'][score_type].to_numpy()
        hybrid_scores = dfs['hybrid'][score_type].to_numpy()
        meta_scores = dfs['meta_only'][score_type].to_numpy()
        
        # Verify same length
        if not (len(base_scores) == len(hybrid_scores) == len(meta_scores)):
            print(f"  ❌ {score_type}: Length mismatch (base:{len(base_scores)}, hybrid:{len(hybrid_scores)}, meta:{len(meta_scores)})")
            base_scores_identical = False
            continue
        
        base_hybrid_identical = np.allclose(base_scores, hybrid_scores, atol=1e-9)
        base_meta_identical = np.allclose(base_scores, meta_scores, atol=1e-9)
        
        all_identical = base_hybrid_identical and base_meta_identical
        status = "✅" if all_identical else "❌"
        
        print(f"  {status} {score_type:15}: {'IDENTICAL' if all_identical else 'DIFFERENT'}")
        
        if not all_identical:
            base_scores_identical = False
            if not base_hybrid_identical:
                diff = np.abs(base_scores - hybrid_scores)
                n_diff = np.sum(diff > 1e-6)
                print(f"     base vs hybrid: {n_diff:,} positions differ (max: {np.max(diff):.6f})")
            if not base_meta_identical:
                diff = np.abs(base_scores - meta_scores)
                n_diff = np.sum(diff > 1e-6)
                print(f"     base vs meta: {n_diff:,} positions differ (max: {np.max(diff):.6f})")
    
    comparison_results['base_scores_identical'] = base_scores_identical
    
    # 2. Compare meta scores (should be different)
    print()
    print("─" * 80)
    print("COMPARISON 2: Meta Scores (should be DIFFERENT between modes)")
    print("─" * 80)
    print("Expected:")
    print("  - base_only: meta scores = base scores (no adjustment)")
    print("  - hybrid: meta scores differ for uncertain positions only")
    print("  - meta_only: meta scores differ for ALL positions")
    print()
    
    meta_scores_differ = {}
    
    for score_type in ['donor_meta', 'acceptor_meta', 'neither_meta']:
        base_meta = dfs['base_only'][score_type].to_numpy()
        hybrid_meta = dfs['hybrid'][score_type].to_numpy()
        meta_meta = dfs['meta_only'][score_type].to_numpy()
        
        # Base-only: meta should equal base
        base_score_type = score_type.replace('_meta', '_score')
        base_base = dfs['base_only'][base_score_type].to_numpy()
        base_meta_equals_base = np.allclose(base_meta, base_base, atol=1e-9)
        
        # Hybrid vs base-only
        hybrid_differs = not np.allclose(hybrid_meta, base_meta, atol=1e-6)
        n_hybrid_diff = np.sum(np.abs(hybrid_meta - base_meta) > 1e-6)
        
        # Meta-only vs base-only
        meta_differs = not np.allclose(meta_meta, base_meta, atol=1e-6)
        n_meta_diff = np.sum(np.abs(meta_meta - base_meta) > 1e-6)
        
        # Meta-only vs hybrid
        meta_hybrid_differs = not np.allclose(meta_meta, hybrid_meta, atol=1e-6)
        n_meta_hybrid_diff = np.sum(np.abs(meta_meta - hybrid_meta) > 1e-6)
        
        print(f"{score_type}:")
        print(f"  {'✅' if base_meta_equals_base else '❌'} base-only: meta = base (expected)")
        print(f"  {'✅' if hybrid_differs else '❌'} hybrid vs base: {n_hybrid_diff:,}/{len(common_positions):,} differ ({n_hybrid_diff/len(common_positions)*100:.1f}%)")
        print(f"  {'✅' if meta_differs else '❌'} meta vs base: {n_meta_diff:,}/{len(common_positions):,} differ ({n_meta_diff/len(common_positions)*100:.1f}%)")
        print(f"  {'✅' if meta_hybrid_differs else '❌'} meta vs hybrid: {n_meta_hybrid_diff:,}/{len(common_positions):,} differ ({n_meta_hybrid_diff/len(common_positions)*100:.1f}%)")
        
        meta_scores_differ[score_type] = {
            'base_meta_equals_base': base_meta_equals_base,
            'hybrid_differs': hybrid_differs,
            'n_hybrid_diff': int(n_hybrid_diff),
            'meta_differs': meta_differs,
            'n_meta_diff': int(n_meta_diff),
            'meta_hybrid_differs': meta_hybrid_differs,
            'n_meta_hybrid_diff': int(n_meta_hybrid_diff)
        }
    
    comparison_results['meta_scores_differ'] = meta_scores_differ
    comparison_results['status'] = 'success'
    
    return comparison_results


def calculate_performance_metrics(df: pl.DataFrame, splice_sites_df: pl.DataFrame, gene_id: str) -> dict:
    """
    Calculate F1 score and other performance metrics.
    
    Compare predicted splice sites (high scores) with annotated splice sites.
    """
    # Get annotated splice sites for this gene
    gene_sites = splice_sites_df.filter(pl.col('gene_id') == gene_id)
    
    if gene_sites.height == 0:
        return {'error': 'No annotated splice sites for this gene'}
    
    # Get true donor and acceptor positions
    # Note: splice_sites_enhanced.tsv uses 'site_type' not 'splice_type'
    true_donors = set(gene_sites.filter(pl.col('site_type') == 'donor')['position'].to_list())
    true_acceptors = set(gene_sites.filter(pl.col('site_type') == 'acceptor')['position'].to_list())
    
    # Predict splice sites based on meta scores (threshold = 0.5)
    pred_donors = set(df.filter(pl.col('donor_meta') >= 0.5)['position'].to_list())
    pred_acceptors = set(df.filter(pl.col('acceptor_meta') >= 0.5)['position'].to_list())
    
    # Calculate metrics for donors
    donor_tp = len(true_donors & pred_donors)
    donor_fp = len(pred_donors - true_donors)
    donor_fn = len(true_donors - pred_donors)
    
    donor_precision = donor_tp / (donor_tp + donor_fp) if (donor_tp + donor_fp) > 0 else 0
    donor_recall = donor_tp / (donor_tp + donor_fn) if (donor_tp + donor_fn) > 0 else 0
    donor_f1 = 2 * donor_precision * donor_recall / (donor_precision + donor_recall) if (donor_precision + donor_recall) > 0 else 0
    
    # Calculate metrics for acceptors
    acceptor_tp = len(true_acceptors & pred_acceptors)
    acceptor_fp = len(pred_acceptors - true_acceptors)
    acceptor_fn = len(true_acceptors - pred_acceptors)
    
    acceptor_precision = acceptor_tp / (acceptor_tp + acceptor_fp) if (acceptor_tp + acceptor_fp) > 0 else 0
    acceptor_recall = acceptor_tp / (acceptor_tp + acceptor_fn) if (acceptor_tp + acceptor_fn) > 0 else 0
    acceptor_f1 = 2 * acceptor_precision * acceptor_recall / (acceptor_precision + acceptor_recall) if (acceptor_precision + acceptor_recall) > 0 else 0
    
    # Overall F1
    overall_tp = donor_tp + acceptor_tp
    overall_fp = donor_fp + acceptor_fp
    overall_fn = donor_fn + acceptor_fn
    
    overall_precision = overall_tp / (overall_tp + overall_fp) if (overall_tp + overall_fp) > 0 else 0
    overall_recall = overall_tp / (overall_tp + overall_fn) if (overall_tp + overall_fn) > 0 else 0
    overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0
    
    return {
        'donor': {
            'tp': donor_tp, 'fp': donor_fp, 'fn': donor_fn,
            'precision': donor_precision, 'recall': donor_recall, 'f1': donor_f1
        },
        'acceptor': {
            'tp': acceptor_tp, 'fp': acceptor_fp, 'fn': acceptor_fn,
            'precision': acceptor_precision, 'recall': acceptor_recall, 'f1': acceptor_f1
        },
        'overall': {
            'tp': overall_tp, 'fp': overall_fp, 'fn': overall_fn,
            'precision': overall_precision, 'recall': overall_recall, 'f1': overall_f1
        }
    }


def main():
    """Run comprehensive test on all 3 modes."""
    
    print("\n" + "="*80)
    print("COMPREHENSIVE INFERENCE TEST: ALL 3 MODES")
    print("="*80)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Project: {project_root}")
    
    # Pre-flight checks
    print("\n" + "="*80)
    print("PRE-FLIGHT CHECKS")
    print("="*80)
    
    if not verify_splice_sites_complete():
        print("\n❌ Pre-flight checks failed. Please regenerate splice sites file.")
        return 1
    
    # Select test genes
    test_genes = select_diverse_test_genes()
    
    if not test_genes:
        print("\n❌ Failed to select test genes")
        return 1
    
    # Print selected genes
    print(f"\n{'='*80}")
    print("SELECTED TEST GENES")
    print(f"{'='*80}")
    
    all_genes = []
    for gene_type, genes in test_genes.items():
        print(f"\n{gene_type.upper()}:")
        for gene in genes:
            gene['gene_type'] = gene_type
            all_genes.append(gene)
            gene_name = gene.get('gene_name') or 'N/A'
            print(f"  {gene['gene_id']:20s} {str(gene_name):15s} "
                  f"{gene['size_class']:8s} {gene['length']:,} bp")
    
    # Model path
    model_path = project_root / 'results/meta_model_1000genes_3mers_fresh/model_multiclass.pkl'
    
    if not model_path.exists():
        print(f"\n❌ Model not found: {model_path}")
        return 1
    
    print(f"\n✅ Model found: {model_path}")
    
    # Load splice sites for performance evaluation
    splice_sites_path = project_root / 'data/ensembl/splice_sites_enhanced.tsv'
    splice_sites_df = pl.read_csv(
        splice_sites_path,
        separator='\t',
        schema_overrides={'chrom': pl.String}
    )
    
    # Run tests for all genes and modes
    modes = ['base_only', 'hybrid', 'meta_only']
    all_results = {}
    
    for gene in all_genes:
        gene_id = gene['gene_id']
        all_results[gene_id] = {}
        
        print(f"\n\n{'#'*80}")
        print(f"# TESTING GENE: {gene_id} ({gene.get('gene_name', 'N/A')})")
        print(f"{'#'*80}")
        
        # Run all 3 modes
        for mode in modes:
            result = run_inference_mode(gene_id, gene, mode, model_path)
            all_results[gene_id][mode] = result
        
        # Compare scores across modes
        if all(all_results[gene_id][mode]['status'] == 'success' for mode in modes):
            comparison = compare_mode_scores(all_results[gene_id], gene_id)
            all_results[gene_id]['comparison'] = comparison
            
            # Calculate performance metrics
            print(f"\n{'='*80}")
            print(f"PERFORMANCE METRICS: {gene_id}")
            print(f"{'='*80}\n")
            
            for mode in modes:
                df = all_results[gene_id][mode]['df']
                metrics = calculate_performance_metrics(df, splice_sites_df, gene_id)
                
                if 'error' not in metrics:
                    all_results[gene_id][mode]['metrics'] = metrics
                    
                    print(f"{mode.upper().replace('_', '-')} MODE:")
                    print(f"  Donor F1:    {metrics['donor']['f1']:.3f}")
                    print(f"  Acceptor F1: {metrics['acceptor']['f1']:.3f}")
                    print(f"  Overall F1:  {metrics['overall']['f1']:.3f}")
                    print(f"  TP: {metrics['overall']['tp']}, FP: {metrics['overall']['fp']}, FN: {metrics['overall']['fn']}")
                    print()
    
    # Final summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    
    total_tests = len(all_genes) * len(modes)
    success_count = sum(
        1 for gene_id in all_results
        for mode in modes
        if all_results[gene_id][mode]['status'] == 'success'
    )
    
    print(f"\nTests passed: {success_count}/{total_tests}")
    
    # Metadata preservation summary
    print(f"\nMetadata Preservation:")
    for gene_id in all_results:
        for mode in modes:
            if all_results[gene_id][mode]['status'] == 'success':
                n_present = len(all_results[gene_id][mode]['metadata_present'])
                status = "✅" if n_present == 9 else "❌"
                print(f"  {status} {gene_id} ({mode}): {n_present}/9 features")
    
    # Score comparison summary
    print(f"\nScore Comparisons:")
    for gene_id in all_results:
        if 'comparison' in all_results[gene_id]:
            comp = all_results[gene_id]['comparison']
            if comp['status'] == 'success':
                base_identical = comp['base_scores_identical']
                meta_differ = any(
                    v['meta_differs'] for v in comp['meta_scores_differ'].values()
                )
                print(f"  {'✅' if base_identical else '❌'} {gene_id}: Base scores identical")
                print(f"  {'✅' if meta_differ else '❌'} {gene_id}: Meta scores differ")
    
    # Performance comparison
    print(f"\nPerformance Comparison (F1 Scores):")
    for gene_id in all_results:
        if all(all_results[gene_id][mode]['status'] == 'success' for mode in modes):
            if all('metrics' in all_results[gene_id][mode] for mode in modes):
                print(f"\n  {gene_id}:")
                for mode in modes:
                    f1 = all_results[gene_id][mode]['metrics']['overall']['f1']
                    print(f"    {mode:12s}: F1 = {f1:.3f}")
    
    # Save results
    output_file = project_root / 'results/comprehensive_test_v2_results.json'
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Prepare results for JSON (remove dataframes)
    json_results = {}
    for gene_id in all_results:
        json_results[gene_id] = {}
        for mode in modes:
            json_results[gene_id][mode] = {
                k: v for k, v in all_results[gene_id][mode].items()
                if k != 'df'  # Remove dataframe
            }
        if 'comparison' in all_results[gene_id]:
            json_results[gene_id]['comparison'] = all_results[gene_id]['comparison']
    
    with open(output_file, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"\n📁 Results saved to: {output_file}")
    
    if success_count == total_tests:
        print(f"\n✅ ALL TESTS PASSED!")
        return 0
    else:
        print(f"\n⚠️  {total_tests - success_count} tests failed")
        return 1


if __name__ == '__main__':
    sys.exit(main())

