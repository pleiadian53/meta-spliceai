#!/usr/bin/env python3
"""
Verify Base Model Outputs for Meta-Learning Training

This script verifies that base model outputs meet the requirements for
meta-learning training dataset generation:

1. Nucleotide scores consistency: Number of scores per gene equals gene length
2. Gene manifest completeness: All processed genes are tracked
3. Splice site annotation consistency: Genes with splice sites are identified
4. Data integrity: No missing or corrupted data

Usage:
    python scripts/testing/verify_base_model_outputs.py \
        --test-dir results/base_model_comparison_robust_YYYYMMDD_HHMMSS
"""

import sys
import json
from pathlib import Path
from typing import Dict, List, Optional
import polars as pl

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def verify_nucleotide_scores(
    nucleotide_scores_path: Path,
    gene_features_path: Path,
    model_name: str
) -> Dict:
    """Verify nucleotide scores consistency with gene lengths.
    
    Parameters
    ----------
    nucleotide_scores_path : Path
        Path to nucleotide_scores.tsv
    gene_features_path : Path
        Path to gene_features.tsv
    model_name : str
        Model name (for reporting)
    
    Returns
    -------
    Dict
        Verification results
    """
    print(f"\n{'='*80}")
    print(f"VERIFYING NUCLEOTIDE SCORES: {model_name}")
    print(f"{'='*80}\n")
    
    if not nucleotide_scores_path.exists():
        print(f"❌ Nucleotide scores file not found: {nucleotide_scores_path}")
        return {
            'success': False,
            'error': 'File not found',
            'genes_verified': 0,
            'genes_failed': 0
        }
    
    # Load nucleotide scores
    print(f"Loading nucleotide scores from: {nucleotide_scores_path}")
    nucleotide_scores = pl.read_csv(
        str(nucleotide_scores_path),
        separator='\t',
        schema_overrides={'chrom': pl.Utf8}
    )
    
    print(f"✅ Loaded {nucleotide_scores.height:,} nucleotide score rows")
    print(f"   Columns: {nucleotide_scores.columns}")
    print()
    
    # Load gene features
    print(f"Loading gene features from: {gene_features_path}")
    gene_features = pl.read_csv(
        str(gene_features_path),
        separator='\t',
        schema_overrides={'chrom': pl.Utf8, 'seqname': pl.Utf8}
    )
    
    print(f"✅ Loaded {gene_features.height:,} genes")
    print()
    
    # Group nucleotide scores by gene
    print("Verifying nucleotide score counts per gene...")
    score_counts = (
        nucleotide_scores
        .group_by('gene_id')
        .agg([
            pl.count().alias('n_scores'),
            pl.col('gene_name').first()
        ])
    )
    
    # Join with gene features to get gene lengths
    verification = score_counts.join(
        gene_features.select(['gene_id', 'gene_length']),
        on='gene_id',
        how='left'
    )
    
    # Check consistency
    verification = verification.with_columns([
        (pl.col('n_scores') == pl.col('gene_length')).alias('is_consistent'),
        (pl.col('n_scores') - pl.col('gene_length')).alias('difference')
    ])
    
    # Count results
    total_genes = verification.height
    consistent_genes = verification.filter(pl.col('is_consistent')).height
    inconsistent_genes = total_genes - consistent_genes
    
    print(f"\nVerification Results:")
    print(f"  Total genes:        {total_genes}")
    print(f"  ✅ Consistent:      {consistent_genes} ({consistent_genes/total_genes*100:.1f}%)")
    print(f"  ❌ Inconsistent:    {inconsistent_genes} ({inconsistent_genes/total_genes*100:.1f}%)")
    print()
    
    # Show inconsistent genes
    if inconsistent_genes > 0:
        print("Inconsistent genes (showing first 10):")
        print("-" * 80)
        print(f"{'Gene ID':<20} {'Gene Name':<15} {'Scores':<10} {'Length':<10} {'Diff':<10}")
        print("-" * 80)
        
        inconsistent = verification.filter(~pl.col('is_consistent')).head(10)
        for row in inconsistent.iter_rows(named=True):
            print(f"{row['gene_id']:<20} {row['gene_name']:<15} {row['n_scores']:<10} "
                  f"{row['gene_length']:<10} {row['difference']:<10}")
        print()
    
    # Calculate statistics
    if inconsistent_genes > 0:
        diff_stats = verification.filter(~pl.col('is_consistent')).select([
            pl.col('difference').abs().mean().alias('mean_abs_diff'),
            pl.col('difference').abs().max().alias('max_abs_diff'),
            pl.col('difference').abs().min().alias('min_abs_diff')
        ]).to_dicts()[0]
        
        print(f"Difference Statistics:")
        print(f"  Mean absolute difference: {diff_stats['mean_abs_diff']:.1f}")
        print(f"  Max absolute difference:  {diff_stats['max_abs_diff']}")
        print(f"  Min absolute difference:  {diff_stats['min_abs_diff']}")
        print()
    
    success = inconsistent_genes == 0
    
    if success:
        print(f"✅ VERIFICATION PASSED: All nucleotide scores are consistent with gene lengths")
    else:
        print(f"⚠️  VERIFICATION WARNING: {inconsistent_genes} genes have inconsistent nucleotide scores")
    
    return {
        'success': success,
        'total_genes': total_genes,
        'consistent_genes': consistent_genes,
        'inconsistent_genes': inconsistent_genes,
        'consistency_rate': consistent_genes / total_genes if total_genes > 0 else 0.0,
        'inconsistent_details': verification.filter(~pl.col('is_consistent')).to_dicts() if inconsistent_genes > 0 else []
    }


def verify_gene_manifest(
    manifest_path: Path,
    model_name: str
) -> Dict:
    """Verify gene manifest completeness.
    
    Parameters
    ----------
    manifest_path : Path
        Path to gene_manifest.tsv
    model_name : str
        Model name (for reporting)
    
    Returns
    -------
    Dict
        Verification results
    """
    print(f"\n{'='*80}")
    print(f"VERIFYING GENE MANIFEST: {model_name}")
    print(f"{'='*80}\n")
    
    if not manifest_path.exists():
        print(f"❌ Gene manifest file not found: {manifest_path}")
        return {
            'success': False,
            'error': 'File not found'
        }
    
    # Load manifest
    print(f"Loading gene manifest from: {manifest_path}")
    manifest = pl.read_csv(
        str(manifest_path),
        separator='\t'
    )
    
    print(f"✅ Loaded {manifest.height:,} gene entries")
    print(f"   Columns: {manifest.columns}")
    print()
    
    # Analyze manifest
    print("Manifest Summary:")
    
    # Count by status
    status_counts = manifest.group_by('status').agg(pl.count().alias('count'))
    print("\nBy Status:")
    for row in status_counts.iter_rows(named=True):
        print(f"  {row['status']:<30}: {row['count']:,}")
    
    # Count requested vs processed
    requested = manifest.filter(pl.col('requested')).height
    processed = manifest.filter(pl.col('status') == 'processed').height
    
    print(f"\nProcessing Summary:")
    print(f"  Requested:  {requested:,}")
    print(f"  Processed:  {processed:,} ({processed/requested*100:.1f}%)")
    print()
    
    # Genes with splice sites
    with_splice_sites = manifest.filter(pl.col('num_splice_sites') > 0).height
    print(f"Genes with splice sites: {with_splice_sites:,} ({with_splice_sites/manifest.height*100:.1f}%)")
    print()
    
    # Check for missing critical fields
    missing_gene_id = manifest.filter(pl.col('gene_id').is_null()).height
    missing_gene_name = manifest.filter(pl.col('gene_name').is_null()).height
    
    if missing_gene_id > 0 or missing_gene_name > 0:
        print(f"⚠️  Missing critical fields:")
        if missing_gene_id > 0:
            print(f"   Missing gene_id: {missing_gene_id}")
        if missing_gene_name > 0:
            print(f"   Missing gene_name: {missing_gene_name}")
        print()
    
    success = (processed > 0 and missing_gene_id == 0 and missing_gene_name == 0)
    
    if success:
        print(f"✅ VERIFICATION PASSED: Gene manifest is complete and well-formed")
    else:
        print(f"❌ VERIFICATION FAILED: Gene manifest has issues")
    
    return {
        'success': success,
        'total_genes': manifest.height,
        'requested': requested,
        'processed': processed,
        'processing_rate': processed / requested if requested > 0 else 0.0,
        'with_splice_sites': with_splice_sites,
        'status_counts': status_counts.to_dicts(),
        'missing_gene_id': missing_gene_id,
        'missing_gene_name': missing_gene_name
    }


def verify_test_outputs(test_dir: Path) -> Dict:
    """Verify all outputs from a base model comparison test.
    
    Parameters
    ----------
    test_dir : Path
        Test directory (e.g., results/base_model_comparison_robust_YYYYMMDD_HHMMSS)
    
    Returns
    -------
    Dict
        Complete verification results
    """
    print(f"\n{'='*80}")
    print(f"VERIFYING TEST OUTPUTS")
    print(f"{'='*80}\n")
    print(f"Test Directory: {test_dir}")
    print()
    
    if not test_dir.exists():
        print(f"❌ Test directory not found: {test_dir}")
        return {'success': False, 'error': 'Test directory not found'}
    
    # Find SpliceAI and OpenSpliceAI test directories
    spliceai_dir = None
    openspliceai_dir = None
    
    for subdir in test_dir.iterdir():
        if subdir.is_dir():
            if 'spliceai' in subdir.name.lower() and 'open' not in subdir.name.lower():
                spliceai_dir = subdir
            elif 'openspliceai' in subdir.name.lower():
                openspliceai_dir = subdir
    
    if spliceai_dir is None:
        print("⚠️  SpliceAI test directory not found")
    else:
        print(f"✅ Found SpliceAI directory: {spliceai_dir.name}")
    
    if openspliceai_dir is None:
        print("⚠️  OpenSpliceAI test directory not found")
    else:
        print(f"✅ Found OpenSpliceAI directory: {openspliceai_dir.name}")
    
    print()
    
    results = {
        'test_dir': str(test_dir),
        'spliceai': {},
        'openspliceai': {}
    }
    
    # Verify SpliceAI
    if spliceai_dir:
        # Find artifacts directory
        artifacts_dir = spliceai_dir / 'artifacts'
        if not artifacts_dir.exists():
            print(f"⚠️  SpliceAI artifacts directory not found")
        else:
            # Verify nucleotide scores
            nucleotide_scores_path = artifacts_dir / 'nucleotide_scores.tsv'
            gene_features_path = Path('data/ensembl/GRCh37/gene_features.tsv')
            
            results['spliceai']['nucleotide_scores'] = verify_nucleotide_scores(
                nucleotide_scores_path,
                gene_features_path,
                'SpliceAI'
            )
            
            # Verify gene manifest
            manifest_path = artifacts_dir / 'gene_manifest.tsv'
            results['spliceai']['gene_manifest'] = verify_gene_manifest(
                manifest_path,
                'SpliceAI'
            )
    
    # Verify OpenSpliceAI
    if openspliceai_dir:
        # Find artifacts directory
        artifacts_dir = openspliceai_dir / 'artifacts'
        if not artifacts_dir.exists():
            print(f"⚠️  OpenSpliceAI artifacts directory not found")
        else:
            # Verify nucleotide scores
            nucleotide_scores_path = artifacts_dir / 'nucleotide_scores.tsv'
            gene_features_path = Path('data/mane/GRCh38/gene_features.tsv')
            
            results['openspliceai']['nucleotide_scores'] = verify_nucleotide_scores(
                nucleotide_scores_path,
                gene_features_path,
                'OpenSpliceAI'
            )
            
            # Verify gene manifest
            manifest_path = artifacts_dir / 'gene_manifest.tsv'
            results['openspliceai']['gene_manifest'] = verify_gene_manifest(
                manifest_path,
                'OpenSpliceAI'
            )
    
    # Overall success
    spliceai_success = (
        results['spliceai'].get('nucleotide_scores', {}).get('success', False) and
        results['spliceai'].get('gene_manifest', {}).get('success', False)
    )
    openspliceai_success = (
        results['openspliceai'].get('nucleotide_scores', {}).get('success', False) and
        results['openspliceai'].get('gene_manifest', {}).get('success', False)
    )
    
    results['overall_success'] = spliceai_success and openspliceai_success
    
    # Print final summary
    print(f"\n{'='*80}")
    print(f"FINAL VERIFICATION SUMMARY")
    print(f"{'='*80}\n")
    
    if spliceai_dir:
        print(f"SpliceAI:")
        print(f"  Nucleotide Scores: {'✅ PASS' if results['spliceai'].get('nucleotide_scores', {}).get('success') else '❌ FAIL'}")
        print(f"  Gene Manifest:     {'✅ PASS' if results['spliceai'].get('gene_manifest', {}).get('success') else '❌ FAIL'}")
    
    if openspliceai_dir:
        print(f"\nOpenSpliceAI:")
        print(f"  Nucleotide Scores: {'✅ PASS' if results['openspliceai'].get('nucleotide_scores', {}).get('success') else '❌ FAIL'}")
        print(f"  Gene Manifest:     {'✅ PASS' if results['openspliceai'].get('gene_manifest', {}).get('success') else '❌ FAIL'}")
    
    print(f"\n{'='*80}")
    if results['overall_success']:
        print("✅ ALL VERIFICATIONS PASSED - Ready for meta-learning training")
    else:
        print("❌ SOME VERIFICATIONS FAILED - Review issues above")
    print(f"{'='*80}\n")
    
    # Save results
    results_file = test_dir / 'verification_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"💾 Saved verification results to: {results_file}")
    
    return results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Verify base model outputs for meta-learning training')
    parser.add_argument('--test-dir', type=str, required=True,
                       help='Test directory (e.g., results/base_model_comparison_robust_YYYYMMDD_HHMMSS)')
    
    args = parser.parse_args()
    
    test_dir = Path(args.test_dir)
    results = verify_test_outputs(test_dir)
    
    # Exit with appropriate code
    sys.exit(0 if results.get('overall_success', False) else 1)




