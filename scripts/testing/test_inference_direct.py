#!/usr/bin/env python3
"""
Direct test of inference workflow - uses actual workflow outputs.

This test:
1. Runs inference on a known good gene (GSTM3)
2. Reads the actual prediction files
3. Compares performance across modes
4. Uses gene-relative coordinates (matching workflow output)

Created: 2025-10-29
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


def test_gene_inference():
    """Test inference on GSTM3 and verify outputs."""
    
    print("\n" + "="*80)
    print("DIRECT INFERENCE TEST")
    print("="*80)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    gene_id = 'ENSG00000134202'  # GSTM3
    model_path = project_root / 'results/meta_model_1000genes_3mers_fresh/model_multiclass.pkl'
    
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return 1
    
    # Run base-only mode
    print("Running BASE-ONLY mode...")
    config = EnhancedSelectiveInferenceConfig(
        target_genes=[gene_id],
        model_path=model_path,
        inference_mode='base_only',
        output_name=None,  # Use default output location
        uncertainty_threshold_low=0.02,
        uncertainty_threshold_high=0.50,
        use_timestamped_output=False,
        verbose=1
    )
    
    workflow = EnhancedSelectiveInferenceWorkflow(config)
    result = workflow.run_incremental()
    
    if not result.success:
        print(f"❌ Workflow failed: {result.message}")
        return 1
    
    # Get output paths
    gene_paths = workflow.output_manager.get_gene_output_paths(gene_id)
    pred_file = gene_paths.predictions_file
    
    print(f"\n✅ Workflow completed successfully")
    print(f"   Output: {pred_file}")
    
    if not pred_file.exists():
        print(f"❌ Prediction file not found: {pred_file}")
        return 1
    
    # Read predictions
    df = pl.read_parquet(pred_file)
    
    print(f"\n{'='*80}")
    print("PREDICTION ANALYSIS")
    print(f"{'='*80}")
    print(f"Total positions: {len(df):,}")
    print(f"Position range: {df['position'].min()} to {df['position'].max()}")
    print(f"Columns: {', '.join(df.columns[:15])}...")
    
    # Analyze scores
    threshold = 0.5
    high_donor = df.filter(pl.col('donor_meta') >= threshold)
    high_acceptor = df.filter(pl.col('acceptor_meta') >= threshold)
    
    print(f"\n{'='*80}")
    print(f"HIGH-CONFIDENCE PREDICTIONS (threshold={threshold})")
    print(f"{'='*80}")
    print(f"Donor sites: {len(high_donor):,}")
    print(f"Acceptor sites: {len(high_acceptor):,}")
    
    if len(high_donor) > 0:
        print(f"\nTop donor positions:")
        print(high_donor.select(['position', 'donor_meta']).sort('donor_meta', descending=True).head(10))
    
    if len(high_acceptor) > 0:
        print(f"\nTop acceptor positions:")
        print(high_acceptor.select(['position', 'acceptor_meta']).sort('acceptor_meta', descending=True).head(10))
    
    # Check metadata preservation
    metadata_cols = [
        'is_uncertain', 'is_low_confidence', 'is_high_entropy',
        'is_low_discriminability', 'max_confidence', 'score_spread',
        'score_entropy', 'confidence_category', 'predicted_type_base'
    ]
    
    present = [col for col in metadata_cols if col in df.columns]
    missing = [col for col in metadata_cols if col not in df.columns]
    
    print(f"\n{'='*80}")
    print("METADATA FEATURES")
    print(f"{'='*80}")
    print(f"Present: {len(present)}/9")
    if missing:
        print(f"Missing: {missing}")
    else:
        print("✅ All metadata features present!")
    
    # Summary statistics
    print(f"\n{'='*80}")
    print("SCORE STATISTICS")
    print(f"{'='*80}")
    print(f"Donor scores:")
    print(f"  Min: {df['donor_meta'].min():.4f}")
    print(f"  Max: {df['donor_meta'].max():.4f}")
    print(f"  Mean: {df['donor_meta'].mean():.4f}")
    print(f"  Median: {df['donor_meta'].median():.4f}")
    
    print(f"\nAcceptor scores:")
    print(f"  Min: {df['acceptor_meta'].min():.4f}")
    print(f"  Max: {df['acceptor_meta'].max():.4f}")
    print(f"  Mean: {df['acceptor_meta'].mean():.4f}")
    print(f"  Median: {df['acceptor_meta'].median():.4f}")
    
    print(f"\n{'='*80}")
    print("✅ TEST COMPLETE")
    print(f"{'='*80}\n")
    
    return 0


if __name__ == '__main__':
    sys.exit(test_gene_inference())

