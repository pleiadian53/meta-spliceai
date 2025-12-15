#!/usr/bin/env python3
"""
Test Gene Manifest Functionality

This script tests the gene manifest tracking system by running predictions
on a mix of valid and invalid genes to verify that:
1. Requested genes are tracked
2. Processed genes are marked correctly
3. Missing genes are reported with appropriate status
4. Nucleotide-level scores are captured

Usage:
    python scripts/testing/test_gene_manifest.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from meta_spliceai import run_base_model_predictions

def test_gene_manifest():
    """Test gene manifest with a mix of valid and invalid genes."""
    
    print("=" * 80)
    print("Gene Manifest Test")
    print("=" * 80)
    print()
    
    # Test genes: mix of valid and invalid
    test_genes = [
        'BRCA1',           # Valid protein-coding gene
        'TP53',            # Valid protein-coding gene
        'MALAT1',          # Valid lncRNA
        'UNKNOWN_GENE_1',  # Invalid - should fail
        'FAKE_GENE_XYZ',   # Invalid - should fail
    ]
    
    print(f"Testing with {len(test_genes)} genes:")
    for gene in test_genes:
        print(f"  - {gene}")
    print()
    
    # Run predictions
    print("Running base model predictions...")
    print("NOTE: save_nucleotide_scores=True for testing (disabled by default)")
    print()
    
    results = run_base_model_predictions(
        base_model='spliceai',
        target_genes=test_genes,
        save_nucleotide_scores=True,  # Enable for testing (disabled by default)
        test_mode=True,  # Use test mode for faster execution
        do_extract_sequences=True,  # Force sequence extraction
        verbosity=2
    )
    
    print()
    print("=" * 80)
    print("Results")
    print("=" * 80)
    print()
    
    # Check if workflow succeeded
    if not results.get('success', False):
        print("❌ Workflow failed!")
        return False
    
    print("✅ Workflow completed successfully")
    print()
    
    # Analyze gene manifest
    manifest_df = results.get('gene_manifest')
    if manifest_df is None or manifest_df.height == 0:
        print("❌ Gene manifest is empty!")
        return False
    
    print(f"📋 Gene Manifest ({manifest_df.height} genes)")
    print("-" * 80)
    print(manifest_df)
    print()
    
    # Analyze manifest summary
    manifest_summary = results.get('manifest_summary', {})
    print("📊 Manifest Summary:")
    print(f"  Total genes: {manifest_summary.get('total_genes', 0)}")
    print(f"  Requested genes: {manifest_summary.get('requested_genes', 0)}")
    print(f"  Processed genes: {manifest_summary.get('processed_genes', 0)}")
    print(f"  Failed genes: {manifest_summary.get('failed_genes', 0)}")
    print()
    
    if manifest_summary.get('status_counts'):
        print("  Status breakdown:")
        for status, count in manifest_summary['status_counts'].items():
            print(f"    {status}: {count}")
        print()
    
    # Check nucleotide scores
    nucleotide_scores_df = results.get('nucleotide_scores')
    if nucleotide_scores_df is not None and nucleotide_scores_df.height > 0:
        print(f"🧬 Nucleotide Scores ({nucleotide_scores_df.height:,} nucleotides)")
        print(f"  Genes: {nucleotide_scores_df['gene_id'].n_unique()}")
        print(f"  Columns: {nucleotide_scores_df.columns}")
        print()
        
        # Show sample
        print("  Sample (first 10 rows):")
        print(nucleotide_scores_df.head(10))
        print()
    else:
        print("⚠️  No nucleotide scores captured")
        print()
    
    # Verify expectations
    print("=" * 80)
    print("Verification")
    print("=" * 80)
    print()
    
    checks_passed = 0
    checks_total = 0
    
    # Check 1: All requested genes should be in manifest
    checks_total += 1
    manifest_genes = set(manifest_df['gene_id'].to_list())
    requested_genes = set(test_genes)
    
    # Note: gene_id might differ from gene_name, so check both
    manifest_gene_names = set(manifest_df['gene_name'].to_list())
    all_manifest_identifiers = manifest_genes | manifest_gene_names
    
    missing_from_manifest = requested_genes - all_manifest_identifiers
    if not missing_from_manifest:
        print("✅ Check 1: All requested genes are in manifest")
        checks_passed += 1
    else:
        print(f"❌ Check 1: Missing genes from manifest: {missing_from_manifest}")
    
    # Check 2: Valid genes should be processed
    checks_total += 1
    valid_genes = {'BRCA1', 'TP53', 'MALAT1'}
    processed_genes = set(
        manifest_df.filter(manifest_df['status'] == 'processed')['gene_name'].to_list()
    )
    
    valid_processed = valid_genes & processed_genes
    if len(valid_processed) >= 2:  # At least 2 valid genes should be processed
        print(f"✅ Check 2: Valid genes processed ({len(valid_processed)}/{len(valid_genes)})")
        checks_passed += 1
    else:
        print(f"❌ Check 2: Not enough valid genes processed ({len(valid_processed)}/{len(valid_genes)})")
    
    # Check 3: Invalid genes should fail
    checks_total += 1
    invalid_genes = {'UNKNOWN_GENE_1', 'FAKE_GENE_XYZ'}
    failed_genes = set(
        manifest_df.filter(manifest_df['status'] != 'processed')['gene_name'].to_list()
    )
    
    invalid_failed = invalid_genes & failed_genes
    if len(invalid_failed) >= 1:  # At least 1 invalid gene should fail
        print(f"✅ Check 3: Invalid genes failed ({len(invalid_failed)}/{len(invalid_genes)})")
        checks_passed += 1
    else:
        print(f"❌ Check 3: Invalid genes not marked as failed")
    
    # Check 4: Nucleotide scores should be captured for processed genes
    checks_total += 1
    if nucleotide_scores_df is not None and nucleotide_scores_df.height > 0:
        nucleotide_genes = set(nucleotide_scores_df['gene_name'].unique().to_list())
        if nucleotide_genes & valid_genes:
            print(f"✅ Check 4: Nucleotide scores captured for processed genes")
            checks_passed += 1
        else:
            print(f"❌ Check 4: No nucleotide scores for valid genes")
    else:
        print(f"❌ Check 4: No nucleotide scores captured")
    
    print()
    print(f"Checks passed: {checks_passed}/{checks_total}")
    print()
    
    # Print artifact paths
    print("=" * 80)
    print("Artifact Paths")
    print("=" * 80)
    print()
    paths = results.get('paths', {})
    print(f"  Artifacts directory: {paths.get('artifacts_dir')}")
    print(f"  Gene manifest: {paths.get('manifest_artifact')}")
    print(f"  Nucleotide scores: {paths.get('nucleotide_scores_artifact')}")
    print(f"  Positions: {paths.get('positions_artifact')}")
    print()
    
    return checks_passed == checks_total


if __name__ == '__main__':
    success = test_gene_manifest()
    sys.exit(0 if success else 1)

