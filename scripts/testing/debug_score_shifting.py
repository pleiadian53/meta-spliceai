#!/usr/bin/env python3
"""
Debug score-shifting to understand why F1=0.

This will show:
1. Raw predictions before adjustment
2. Adjusted predictions after score-shifting
3. GTF annotations
4. Which predictions match annotations
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))

import polars as pl

# Test on GSTM3 (small gene)
gene_id = 'ENSG00000134202'

print("="*80)
print(f"DEBUG: Score-Shifting for {gene_id}")
print("="*80)

# Load predictions
pred_file = Path(f"predictions/base_only/{gene_id}/{gene_id}_predictions.parquet")
if pred_file.exists():
    predictions_df = pl.read_parquet(pred_file)
    
    print(f"\nPredictions loaded: {predictions_df.height} rows")
    print(f"Columns: {predictions_df.columns}")
    
    # Show sample of high-scoring positions
    print(f"\nTop 10 donor scores:")
    top_donors = predictions_df.sort('donor_score', descending=True).head(10)
    for row in top_donors.iter_rows(named=True):
        print(f"  Position {row['position']}: donor={row['donor_score']:.3f}, acceptor={row['acceptor_score']:.3f}")
    
    print(f"\nTop 10 acceptor scores:")
    top_acceptors = predictions_df.sort('acceptor_score', descending=True).head(10)
    for row in top_acceptors.iter_rows(named=True):
        print(f"  Position {row['position']}: donor={row['donor_score']:.3f}, acceptor={row['acceptor_score']:.3f}")
else:
    print(f"Predictions file not found: {pred_file}")

# Load GTF annotations
print(f"\n{'='*80}")
print("GTF Annotations")
print("="*80)

from meta_spliceai.splice_engine.meta_models.workflows.data_preparation import load_splice_sites_from_gtf

gtf_path = "data/ensembl/Homo_sapiens.GRCh38.112.gtf"
annotations_df = load_splice_sites_from_gtf(gtf_path, gene_ids=[gene_id])

print(f"\nAnnotations loaded: {annotations_df.height} rows")

donors = annotations_df.filter(pl.col('splice_type') == 'donor')
acceptors = annotations_df.filter(pl.col('splice_type') == 'acceptor')

print(f"\nDonor sites ({len(donors)}):")
for row in donors.head(10).iter_rows(named=True):
    print(f"  Position {row['position']}")

print(f"\nAcceptor sites ({len(acceptors)}):")
for row in acceptors.head(10).iter_rows(named=True):
    print(f"  Position {row['position']}")

# Check if any predictions match annotations
if pred_file.exists():
    print(f"\n{'='*80}")
    print("Matching Analysis")
    print("="*80)
    
    donor_positions = set(donors['position'].to_list())
    acceptor_positions = set(acceptors['position'].to_list())
    
    # Check predictions at threshold 0.5
    pred_donors = set(predictions_df.filter(pl.col('donor_score') > 0.5)['position'].to_list())
    pred_acceptors = set(predictions_df.filter(pl.col('acceptor_score') > 0.5)['position'].to_list())
    
    print(f"\nAt threshold 0.5:")
    print(f"  Annotated donors: {len(donor_positions)}")
    print(f"  Predicted donors: {len(pred_donors)}")
    print(f"  Matches (TP): {len(donor_positions & pred_donors)}")
    print(f"  FP: {len(pred_donors - donor_positions)}")
    print(f"  FN: {len(donor_positions - pred_donors)}")
    
    print(f"\n  Annotated acceptors: {len(acceptor_positions)}")
    print(f"  Predicted acceptors: {len(pred_acceptors)}")
    print(f"  Matches (TP): {len(acceptor_positions & pred_acceptors)}")
    print(f"  FP: {len(pred_acceptors - acceptor_positions)}")
    print(f"  FN: {len(acceptor_positions - pred_acceptors)}")
    
    # Show specific examples
    if len(donor_positions) > 0:
        example_donor = list(donor_positions)[0]
        print(f"\nExample donor site: position {example_donor}")
        
        # Check scores around this position
        window = predictions_df.filter(
            (pl.col('position') >= example_donor - 5) &
            (pl.col('position') <= example_donor + 5)
        ).sort('position')
        
        print(f"  Scores around annotated donor site:")
        for row in window.iter_rows(named=True):
            marker = " ← ANNOTATED" if row['position'] == example_donor else ""
            print(f"    Pos {row['position']}: donor={row['donor_score']:.3f}, acceptor={row['acceptor_score']:.3f}{marker}")
    
    if len(acceptor_positions) > 0:
        example_acceptor = list(acceptor_positions)[0]
        print(f"\nExample acceptor site: position {example_acceptor}")
        
        # Check scores around this position
        window = predictions_df.filter(
            (pl.col('position') >= example_acceptor - 5) &
            (pl.col('position') <= example_acceptor + 5)
        ).sort('position')
        
        print(f"  Scores around annotated acceptor site:")
        for row in window.iter_rows(named=True):
            marker = " ← ANNOTATED" if row['position'] == example_acceptor else ""
            print(f"    Pos {row['position']}: donor={row['donor_score']:.3f}, acceptor={row['acceptor_score']:.3f}{marker}")

print("\n" + "="*80)

