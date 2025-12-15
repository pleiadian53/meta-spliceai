#!/usr/bin/env python3
"""Quick verification of test results."""

import sys
from pathlib import Path
import polars as pl

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def verify_predictions(pred_file: Path, gene_id: str, expected_length: int):
    """Verify prediction file."""
    print(f"\n{'='*60}")
    print(f"Verifying: {pred_file.name}")
    print(f"Gene: {gene_id}, Expected length: {expected_length:,} bp")
    print(f"{'='*60}")
    
    if not pred_file.exists():
        print(f"❌ File not found")
        return False
    
    # Load predictions
    df = pl.read_parquet(pred_file)
    
    print(f"\n📊 Basic Stats:")
    print(f"  Total rows: {df.height:,}")
    print(f"  Columns: {len(df.columns)}")
    
    # Check gene
    genes = df['gene_id'].unique().to_list()
    print(f"  Genes: {genes}")
    
    # Check positions
    gene_df = df.filter(pl.col('gene_id') == gene_id)
    n_positions = gene_df.height
    print(f"\n📍 Positions:")
    print(f"  Predicted: {n_positions:,}")
    print(f"  Expected: {expected_length:,}")
    print(f"  Match: {'✅' if n_positions == expected_length else '❌'}")
    print(f"  Coverage: {(n_positions / expected_length * 100):.1f}%")
    
    # Check score columns
    print(f"\n🎯 Score Columns:")
    score_cols = ['donor_score', 'acceptor_score', 'neither_score']
    for col in score_cols:
        if col in df.columns:
            print(f"  ✅ {col}")
        else:
            print(f"  ❌ {col} MISSING")
    
    # Check meta columns
    print(f"\n🧠 Meta-Model Columns:")
    meta_cols = ['donor_meta', 'acceptor_meta', 'neither_meta', 'is_adjusted']
    for col in meta_cols:
        if col in df.columns:
            if col == 'is_adjusted':
                n_adjusted = gene_df[col].sum()
                pct = (n_adjusted / n_positions * 100) if n_positions > 0 else 0
                print(f"  ✅ {col}: {n_adjusted:,} ({pct:.1f}%)")
            else:
                print(f"  ✅ {col}")
        else:
            print(f"  ❌ {col} MISSING")
    
    # Check additional columns
    print(f"\n📋 Additional Columns:")
    add_cols = ['chrom', 'strand', 'transcript_id', 'splice_type']
    for col in add_cols:
        if col in df.columns:
            print(f"  ✅ {col}")
    
    return True

def main():
    print("\n" + "="*60)
    print("TEST RESULTS VERIFICATION")
    print("="*60)
    
    # Test cases that succeeded
    test_cases = [
        {
            'gene_id': 'ENSG00000141736',
            'expected_length': 42513,
            'mode': 'hybrid',
            'file': Path('predictions/meta_modes_test/test_hybrid/ENSG00000141736/predictions/hybrid/combined_predictions.parquet')
        }
    ]
    
    for test in test_cases:
        verify_predictions(test['file'], test['gene_id'], test['expected_length'])
    
    print("\n" + "="*60)
    print("VERIFICATION COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()

