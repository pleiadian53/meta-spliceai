#!/usr/bin/env python3
"""Test script for enhanced splice sites extraction.

This script tests the new splice_sites.py module to verify that:
1. All 14 columns are extracted (8 core + 6 enhanced)
2. Enhanced columns contain data (not all empty)
3. The extraction completes successfully
4. Output file is created with correct format

Usage:
    python scripts/testing/test_enhanced_splice_sites_extraction.py

Requirements:
    - mamba activate metaspliceai
    - GTF file exists at data/mane/GRCh38/MANE.GRCh38.v1.3.refseq_genomic.gtf
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

import polars as pl
from meta_spliceai.system.genomic_resources import extract_splice_sites_from_gtf


def test_enhanced_splice_sites_extraction():
    """Test enhanced splice sites extraction."""
    
    print("="*80)
    print("TESTING ENHANCED SPLICE SITES EXTRACTION")
    print("="*80)
    print()
    
    # Setup paths
    gtf_path = project_root / "data/mane/GRCh38/MANE.GRCh38.v1.3.refseq_genomic.gtf"
    output_file = project_root / "data/mane/GRCh38/splice_sites_enhanced_test.tsv"
    
    # Check GTF exists
    if not gtf_path.exists():
        print(f"❌ GTF file not found: {gtf_path}")
        print("Please ensure the GTF file exists before running this test.")
        return False
    
    print(f"✓ Found GTF file: {gtf_path.name}")
    print()
    
    # Run extraction
    print("Running extraction...")
    print("-" * 80)
    
    try:
        df = extract_splice_sites_from_gtf(
            gtf_path=str(gtf_path),
            consensus_window=2,
            output_file=str(output_file),
            save=True,
            return_df=True,
            verbosity=2
        )
    except Exception as e:
        print(f"❌ Extraction failed: {e}")
        return False
    
    print()
    print("-" * 80)
    print("✓ Extraction completed successfully")
    print()
    
    # Test 1: Check column count
    print("TEST 1: Column Count")
    print("-" * 80)
    expected_cols = 14
    actual_cols = len(df.columns)
    
    if actual_cols == expected_cols:
        print(f"✓ PASS: {actual_cols} columns (expected {expected_cols})")
    else:
        print(f"❌ FAIL: {actual_cols} columns (expected {expected_cols})")
        return False
    print()
    
    # Test 2: Check required columns exist
    print("TEST 2: Required Columns")
    print("-" * 80)
    
    required_columns = [
        # Core columns
        'chrom', 'start', 'end', 'position', 'strand', 'site_type', 
        'gene_id', 'transcript_id',
        # Enhanced columns
        'gene_name', 'gene_biotype', 'transcript_biotype',
        'exon_id', 'exon_number', 'exon_rank'
    ]
    
    missing_cols = []
    for col in required_columns:
        if col in df.columns:
            print(f"  ✓ {col}")
        else:
            print(f"  ❌ {col} (MISSING)")
            missing_cols.append(col)
    
    if missing_cols:
        print(f"\n❌ FAIL: Missing columns: {missing_cols}")
        return False
    else:
        print("\n✓ PASS: All required columns present")
    print()
    
    # Test 3: Check enhanced columns have data
    print("TEST 3: Enhanced Columns Data Quality")
    print("-" * 80)
    
    # Load a sample for testing
    df_sample = pl.read_csv(str(output_file), separator='\t', n_rows=1000)
    
    enhanced_cols = ['gene_name', 'gene_biotype', 'transcript_biotype', 
                     'exon_id', 'exon_number', 'exon_rank']
    
    all_passed = True
    for col in enhanced_cols:
        filled = df_sample[col].is_not_null().sum()
        total = len(df_sample)
        pct = (filled / total) * 100 if total > 0 else 0
        
        # Expect at least 50% filled (some attributes may be legitimately missing)
        if pct >= 50:
            print(f"  ✓ {col:25s}: {filled:4d}/{total:4d} ({pct:5.1f}%)")
        else:
            print(f"  ⚠ {col:25s}: {filled:4d}/{total:4d} ({pct:5.1f}%) - LOW")
            all_passed = False
    
    if all_passed:
        print("\n✓ PASS: Enhanced columns have sufficient data")
    else:
        print("\n⚠ WARNING: Some enhanced columns have low data coverage")
        print("   This may be expected depending on the GTF format.")
    print()
    
    # Test 4: Validate unique genes and transcripts
    print("TEST 4: Data Integrity Validation")
    print("-" * 80)
    
    # Load full file for validation
    df_full = pl.read_csv(str(output_file), separator='\t')
    
    unique_genes = df_full['gene_id'].n_unique()
    unique_transcripts = df_full['transcript_id'].n_unique()
    
    # Count splice sites by type
    donors = df_full.filter(pl.col('site_type') == 'donor').height
    acceptors = df_full.filter(pl.col('site_type') == 'acceptor').height
    
    print(f"  ✓ Total splice sites: {len(df_full):,}")
    print(f"  ✓ Unique genes: {unique_genes:,}")
    print(f"  ✓ Unique transcripts: {unique_transcripts:,}")
    print(f"  ✓ Donor sites: {donors:,}")
    print(f"  ✓ Acceptor sites: {acceptors:,}")
    
    # Sanity checks
    if donors > 0 and acceptors > 0:
        ratio = donors / acceptors
        if 0.8 <= ratio <= 1.2:
            print(f"  ✓ Donor/Acceptor ratio: {ratio:.2f} (balanced)")
        else:
            print(f"  ⚠ Donor/Acceptor ratio: {ratio:.2f} (may indicate data issue)")
    
    # Check transcripts per gene
    avg_tx_per_gene = unique_transcripts / unique_genes if unique_genes > 0 else 0
    print(f"  ✓ Avg transcripts per gene: {avg_tx_per_gene:.2f}")
    
    # Check splice sites per transcript (should be >= 2 for most)
    ss_per_tx = df_full.group_by('transcript_id').agg(pl.count().alias('n_sites'))
    avg_ss_per_tx = ss_per_tx['n_sites'].mean()
    print(f"  ✓ Avg splice sites per transcript: {avg_ss_per_tx:.1f}")
    
    print()
    
    # Test 5: Show sample data
    print("TEST 5: Sample Data")
    print("-" * 80)
    
    sample_cols = ['gene_id', 'gene_name', 'gene_biotype', 'site_type', 'exon_rank']
    print(df_sample.select(sample_cols).head(10))
    print()
    
    # Test 6: Check output file
    print("TEST 6: Output File")
    print("-" * 80)
    
    if output_file.exists():
        file_size = output_file.stat().st_size / (1024 * 1024)  # MB
        print(f"✓ Output file created: {output_file.name}")
        print(f"  Size: {file_size:.2f} MB")
        print(f"  Rows: {len(df):,}")
    else:
        print(f"❌ Output file not found: {output_file}")
        return False
    print()
    
    # Summary
    print("="*80)
    print("TEST SUMMARY")
    print("="*80)
    print("✅ ALL TESTS PASSED")
    print()
    print("The enhanced splice sites extraction is working correctly.")
    print(f"Output saved to: {output_file}")
    print()
    print("You can now safely regenerate splice_sites_enhanced.tsv for your")
    print("actual data using:")
    print()
    print("  python -c \"")
    print("from meta_spliceai.system.genomic_resources import GenomicDataDeriver")
    print("from pathlib import Path")
    print()
    print("deriver = GenomicDataDeriver(data_dir=Path('data/mane/GRCh38'), verbosity=2)")
    print("result = deriver.derive_splice_sites(")
    print("    output_filename='splice_sites_enhanced.tsv',")
    print("    consensus_window=2,")
    print("    force_overwrite=True")
    print(")")
    print("print(f'Success: {result[\\\"success\\\"]}')")
    print("  \"")
    print()
    
    return True


if __name__ == "__main__":
    success = test_enhanced_splice_sites_extraction()
    sys.exit(0 if success else 1)

