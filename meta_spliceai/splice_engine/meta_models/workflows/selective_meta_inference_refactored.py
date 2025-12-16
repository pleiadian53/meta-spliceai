"""
Selective Meta-Model Inference System (Refactored)
=================================================

This module provides the main driver for selective meta-model inference.
The implementation has been refactored into modular components for better
maintainability and reduced complexity.

Key Features:
1. **Complete Coverage Capability**: Can predict at every nucleotide
2. **Selective Featurization**: Only generates features for uncertain positions  
3. **Base Model Reuse**: Directly uses confident base model predictions
4. **Hybrid Prediction System**: Combines base + meta predictions seamlessly
5. **Structured Data Management**: Organized artifacts and gene tracking
"""

from __future__ import annotations

import os
import sys
import time
import shutil
import datetime
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Any
from collections import defaultdict

import polars as pl
import pandas as pd
import numpy as np

# Optional MLflow import
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

# Import modular components
from .inference import (
    SelectiveInferenceConfig,
    SelectiveInferenceResults,
    verify_selective_featurization,
    setup_inference_directories,
    track_processed_genes,
    create_gene_manifest,
    get_test_data_directory,
    combine_predictions_for_complete_coverage,
    generate_selective_meta_predictions,
    save_predictions,
    save_inference_metadata,
    preserve_artifacts
)

# Import existing infrastructure
from .splice_inference_workflow import run_enhanced_splice_inference_workflow
from .inference_workflow_utils import load_model_with_calibration


def run_selective_meta_inference(config: SelectiveInferenceConfig) -> SelectiveInferenceResults:
    """
    Run selective meta-model inference with complete coverage capability.
    
    This function implements the streamlined strategy:
    1. Generate base model predictions for all positions
    2. Selectively featurize only uncertain positions  
    3. Apply meta-model recalibration to uncertain positions
    4. Combine base + meta predictions for complete coverage
    5. Track processed genes and manage artifacts efficiently
    
    Parameters
    ----------
    config : SelectiveInferenceConfig
        Complete configuration for selective inference
        
    Returns
    -------
    SelectiveInferenceResults
        Comprehensive results with all output paths and statistics
    """
    start_time = time.time()
    
    if config.verbose >= 1:
        print(f"\n🎯 SELECTIVE META-MODEL INFERENCE")
        print(f"=" * 60)
        print(f"📊 Target genes: {len(config.target_genes)}")
        print(f"🎚️  Uncertainty range: [{config.uncertainty_threshold_low:.3f}, {config.uncertainty_threshold_high:.3f})")
        print(f"🧠 Model: {config.model_path}")
        print(f"🔬 Strategy: Selective featurization + base model reuse")
    
    # Set up output directory structure
    if config.inference_base_dir is not None:
        output_dir = config.inference_base_dir
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_name = config.output_name or f"selective_inference_{timestamp}"
        test_data_dir = get_test_data_directory(config.training_dataset_path)
        output_dir = test_data_dir / "predictions" / output_name
    
    output_dir.mkdir(parents=True, exist_ok=True)
    directories = setup_inference_directories(output_dir)
    
    # Initialize results
    results = SelectiveInferenceResults(
        success=False,
        config=config,
        error_messages=[]
    )
    
    # Initialize variables for cleanup
    preserve_artifacts_flag = config.keep_artifacts_dir is not None
    temp_dir = None
    
    try:
        # Step 1: Load training coverage for selective exclusion
        training_positions = _load_training_coverage(config)
        
        # Step 2: Run workflow to get base predictions and analysis sequences
        workflow_results, temp_dir = _run_inference_workflow(
            config, training_positions, preserve_artifacts_flag
        )
        
        # Step 3: Load complete base model predictions
        complete_base_pd = _load_base_predictions(workflow_results, config)
        
        # Step 4: Generate meta-model predictions for uncertain positions
        meta_predictions_pd = generate_selective_meta_predictions(
            config, complete_base_pd, workflow_results, config.verbose >= 1
        )
        
        # Step 5: Combine predictions for complete coverage
        hybrid_predictions = combine_predictions_for_complete_coverage(
            complete_base_pd,
            meta_predictions_pd,
            config.uncertainty_threshold_low,
            config.uncertainty_threshold_high,
            config.inference_mode
        )
        
        # Step 6: Save results and calculate statistics
        results = _save_results_and_statistics(
            hybrid_predictions,
            meta_predictions_pd,
            complete_base_pd,
            directories,
            config,
            results
        )
        
        # Step 7: Organize and preserve artifacts
        _organize_artifacts(
            complete_base_pd,
            temp_dir,
            config,
            directories
        )
        
        # Log to MLflow if available
        _log_to_mlflow(results, start_time)
        
        results.success = True
        
    except Exception as e:
        results.success = False
        results.error_messages = [str(e)]
        if config.verbose >= 1:
            print(f"\n❌ Error in selective inference: {e}")
            import traceback
            traceback.print_exc()
        raise
        
    finally:
        # Clean up temporary directory if needed
        if temp_dir and temp_dir.exists() and not preserve_artifacts_flag:
            if config.cleanup_intermediates:
                shutil.rmtree(temp_dir)
                if config.verbose >= 1:
                    print(f"   🗑️  Cleaned up temporary directory: {temp_dir}")
        
        results.processing_time_seconds = time.time() - start_time
    
    if config.verbose >= 1:
        print(f"\n🎉 SELECTIVE INFERENCE COMPLETE!")
        print(f"=" * 60)
        print(f"⏱️  Processing time: {results.processing_time_seconds:.1f}s")
        print(f"📁 Output directory: {output_dir}")
        print(f"✨ Complete coverage achieved with selective efficiency!")
    
    return results


def _load_training_coverage(config: SelectiveInferenceConfig) -> Dict[str, set]:
    """Load training positions for selective exclusion."""
    training_positions = defaultdict(set)
    
    if config.training_dataset_path and config.training_dataset_path.exists():
        if config.verbose >= 1:
            print(f"\n📚 STEP 1: Loading training coverage")
        
        try:
            # Load training positions
            if config.training_dataset_path.is_dir():
                train_files = list(config.training_dataset_path.glob("*.parquet"))
                if train_files:
                    train_dfs = [pl.read_parquet(f, columns=['gene_id', 'position']) 
                                for f in train_files]
                    train_df = pl.concat(train_dfs).unique()
                else:
                    train_df = pl.DataFrame({'gene_id': [], 'position': []})
            else:
                train_df = pl.read_parquet(
                    config.training_dataset_path, 
                    columns=['gene_id', 'position']
                ).unique()
            
            # Filter to target genes and build lookup
            train_df = train_df.filter(pl.col('gene_id').is_in(config.target_genes))
            for row in train_df.iter_rows():
                gene_id, position = row
                training_positions[gene_id].add(position)
            
            if config.verbose >= 1:
                total_training = sum(len(positions) for positions in training_positions.values())
                print(f"   ✅ Loaded {total_training:,} training positions across {len(training_positions)} genes")
                
        except Exception as e:
            if config.verbose >= 1:
                print(f"   ⚠️  Could not load training coverage: {e}")
    
    return dict(training_positions)


def _run_inference_workflow(
    config: SelectiveInferenceConfig,
    training_positions: Dict[str, set],
    preserve_artifacts: bool
) -> tuple[Dict[str, Any], Path]:
    """Run the base inference workflow."""
    if config.verbose >= 1:
        print(f"\n🔬 STEP 2: Running inference workflow")
    
    # Create temporary directory for workflow
    if preserve_artifacts:
        temp_dir = Path(config.keep_artifacts_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)
    else:
        temp_dir = Path(tempfile.mkdtemp(prefix="selective_inference_"))
    
    if config.verbose >= 1:
        print(f"   🗂️  Working directory: {temp_dir}")
    
    # Configure workflow based on coverage mode
    if getattr(config, 'ensure_complete_coverage', False):
        # Use complete coverage workflow
        from .splice_inference_workflow import run_complete_coverage_inference_workflow
        
        workflow_results = run_complete_coverage_inference_workflow(
            target_genes=config.target_genes,
            target_chromosomes=None,
            verbosity=max(0, config.verbose - 1),
            max_positions_per_gene=0,  # No limit for complete coverage
            no_final_aggregate=True,
            model_path=config.model_path,
            use_calibration=config.use_calibration,
            output_dir=temp_dir / "complete_coverage_output",
            train_schema_dir=config.training_schema_path,
            # Reuse existing genomic datasets
            do_extract_annotations=False,
            do_extract_sequences=False,
            do_extract_splice_sites=False,
            do_find_overlaping_genes=False,
        )
    else:
        # Use selective workflow
        workflow_results = run_enhanced_splice_inference_workflow(
            target_genes=config.target_genes,
            output_dir=temp_dir / "workflow_output",
            covered_pos=training_positions if training_positions else None,
            t_low=config.uncertainty_threshold_low,
            t_high=config.uncertainty_threshold_high,
            max_positions_per_gene=config.max_positions_per_gene,
            max_analysis_rows=config.max_analysis_rows,
            model_path=config.model_path,
            use_calibration=config.use_calibration,
            train_schema_dir=config.training_schema_path,
            do_prepare_sequences=False,
            do_prepare_annotations=False,
            cleanup=False,
            verbosity=max(0, config.verbose - 1)
        )
    
    if not workflow_results.get('success', False):
        raise RuntimeError("Inference workflow failed")
    
    return workflow_results, temp_dir


def _load_base_predictions(
    workflow_results: Dict[str, Any],
    config: SelectiveInferenceConfig
) -> pd.DataFrame:
    """Load complete base model predictions from workflow results."""
    if config.verbose >= 1:
        print(f"\n📊 STEP 3: Loading complete base model predictions")
    
    # Handle different result structures
    if workflow_results.get("analysis_sequences") is not None:
        # Complete coverage workflow
        complete_base_df = workflow_results["analysis_sequences"]
        complete_base_pd = complete_base_df.to_pandas() if hasattr(complete_base_df, 'to_pandas') else complete_base_df
    else:
        # Selective workflow - load from files
        workflow_output_dir = Path(workflow_results['paths']['output_dir'])
        analysis_files = list(workflow_output_dir.glob("analysis_sequences_*.tsv"))
        
        if not analysis_files:
            raise RuntimeError("No base model prediction files found")
        
        # Load and combine all base model predictions
        base_dfs = []
        for file_path in analysis_files:
            df = pl.read_csv(
                file_path, 
                separator="\t", 
                infer_schema_length=1000,
                schema_overrides={"chrom": pl.Utf8}
            )
            # Filter to target genes
            df = df.filter(pl.col("gene_id").is_in(config.target_genes))
            base_dfs.append(df)
        
        complete_base_df = pl.concat(base_dfs).unique(subset=['gene_id', 'position'])
        complete_base_pd = complete_base_df.to_pandas()
    
    if config.verbose >= 1:
        print(f"   📊 Loaded {len(complete_base_pd):,} complete base model predictions")
    
    return complete_base_pd


def _save_results_and_statistics(
    hybrid_predictions: pd.DataFrame,
    meta_predictions_pd: pd.DataFrame,
    complete_base_pd: pd.DataFrame,
    directories: Dict[str, Path],
    config: SelectiveInferenceConfig,
    results: SelectiveInferenceResults
) -> SelectiveInferenceResults:
    """Save results and calculate statistics."""
    if config.verbose >= 1:
        print(f"\n💾 STEP 6: Saving results")
    
    # Calculate statistics
    total_positions = len(hybrid_predictions)
    recalibrated_positions = (hybrid_predictions['prediction_source'] == 'meta_model').sum()
    reused_positions = total_positions - recalibrated_positions
    
    if config.verbose >= 1:
        print(f"   📊 Complete coverage: {total_positions:,} positions")
        print(f"   🤖 Meta-model recalibrated: {recalibrated_positions:,} ({recalibrated_positions/total_positions:.1%})")
        print(f"   🔄 Base model reused: {reused_positions:,} ({reused_positions/total_positions:.1%})")
    
    # Save predictions
    pred_dir = directories['predictions']
    
    # Save hybrid predictions
    hybrid_path = pred_dir / "complete_coverage_predictions.parquet"
    save_predictions(hybrid_predictions, hybrid_path)
    results.hybrid_predictions_path = hybrid_path
    
    # Save meta-model predictions
    if not meta_predictions_pd.empty:
        meta_path = pred_dir / "meta_model_predictions.parquet"
        save_predictions(meta_predictions_pd, meta_path)
        results.meta_predictions_path = meta_path
    
    # Save base model predictions
    base_path = pred_dir / "base_model_predictions.parquet"
    save_predictions(complete_base_pd, base_path)
    results.base_predictions_path = base_path
    
    # Generate per-gene statistics
    per_gene_stats = {}
    for gene_id in config.target_genes:
        gene_data = hybrid_predictions[hybrid_predictions['gene_id'] == gene_id]
        if len(gene_data) > 0:
            per_gene_stats[gene_id] = {
                'total_positions': len(gene_data),
                'recalibrated_positions': (gene_data['prediction_source'] == 'meta_model').sum(),
                'reused_positions': (gene_data['prediction_source'] == 'base_model').sum(),
                'uncertain_positions': gene_data['is_uncertain'].sum()
            }
    
    # Track processed genes
    manifest_path = directories['manifests'] / "selective_inference_manifest.csv"
    track_processed_genes(manifest_path, per_gene_stats, config)
    results.gene_manifest_path = manifest_path
    
    # Update results
    results.total_positions = total_positions
    results.positions_recalibrated = recalibrated_positions
    results.positions_reused = reused_positions
    results.genes_processed = len(per_gene_stats)
    results.per_gene_stats = per_gene_stats
    
    if config.verbose >= 1:
        print(f"   ✅ Saved hybrid predictions: {hybrid_path}")
        if results.meta_predictions_path:
            print(f"   ✅ Saved meta predictions: {results.meta_predictions_path}")
        print(f"   ✅ Saved base predictions: {base_path}")
        print(f"   📋 Updated gene manifest: {manifest_path}")
    
    return results


def _organize_artifacts(
    complete_base_pd: pd.DataFrame,
    temp_dir: Path,
    config: SelectiveInferenceConfig,
    directories: Dict[str, Path]
) -> None:
    """Organize and preserve artifacts."""
    if config.verbose >= 1:
        print(f"\n💾 STEP 7: Organizing artifacts")
    
    # Always save test data to systematic directory
    test_data_dir = get_test_data_directory(config.training_dataset_path)
    test_data_dir.mkdir(parents=True, exist_ok=True)
    
    # Create test data structure
    master_dir = test_data_dir / "master"
    master_dir.mkdir(parents=True, exist_ok=True)
    
    # Save test data in training-like format
    genes_per_batch = 100
    gene_list = config.target_genes
    batch_num = 1
    
    for i in range(0, len(gene_list), genes_per_batch):
        batch_genes = gene_list[i:i+genes_per_batch]
        batch_data = complete_base_pd[complete_base_pd['gene_id'].isin(batch_genes)]
        
        if len(batch_data) > 0:
            batch_file = master_dir / f"batch_{batch_num:05d}.parquet"
            save_predictions(batch_data, batch_file, ensure_string_gene_id=True)
            if config.verbose >= 1:
                print(f"   📦 Saved batch {batch_num}: {len(batch_data)} positions from {len(batch_genes)} genes")
            batch_num += 1
    
    # Create gene manifest
    create_gene_manifest(master_dir, config.target_genes, complete_base_pd, config.verbose >= 1)
    
    # Preserve additional artifacts if requested
    if config.keep_artifacts_dir:
        artifacts_dir = Path(config.keep_artifacts_dir)
        preserve_artifacts(temp_dir, artifacts_dir, verbose=config.verbose >= 1)
    
    # Save metadata
    metadata_path = directories['cache'] / "inference_metadata.json"
    save_inference_metadata(
        metadata_path,
        config,
        {'total_positions': len(complete_base_pd)},
        {'artifacts_preserved': config.keep_artifacts_dir is not None}
    )
    
    if config.verbose >= 1:
        print(f"   ✅ Test data saved to: {test_data_dir}")
        print(f"   📋 Metadata saved to: {metadata_path}")


def _log_to_mlflow(results: SelectiveInferenceResults, start_time: float) -> None:
    """Log results to MLflow if available."""
    if MLFLOW_AVAILABLE and mlflow.active_run():
        try:
            # Log metrics
            mlflow.log_metrics({
                "selective_total_positions": results.total_positions,
                "selective_uncertain_positions": results.positions_recalibrated,
                "selective_reused_positions": results.positions_reused,
                "selective_efficiency": (results.positions_reused / results.total_positions * 100) 
                                      if results.total_positions > 0 else 0,
                "selective_processing_time": time.time() - start_time
            })
            
            # Log artifacts
            if results.hybrid_predictions_path:
                mlflow.log_artifact(str(results.hybrid_predictions_path), "selective_inference")
            if results.meta_predictions_path:
                mlflow.log_artifact(str(results.meta_predictions_path), "selective_inference")
            if results.base_predictions_path:
                mlflow.log_artifact(str(results.base_predictions_path), "selective_inference")
                
        except Exception as e:
            print(f"   ⚠️  Failed to log to MLflow: {e}")


# Re-export key functions for backward compatibility
from .inference import (
    create_selective_config,
    verify_no_label_leakage,
    get_excluded_columns_for_inference
)

__all__ = [
    "SelectiveInferenceConfig",
    "SelectiveInferenceResults", 
    "run_selective_meta_inference",
    "create_selective_config",
    "combine_predictions_for_complete_coverage",
    "setup_inference_directories",
    "track_processed_genes",
    "get_excluded_columns_for_inference",
    "verify_no_label_leakage",
    "verify_selective_featurization"
]


if __name__ == "__main__":
    # Example usage
    config = create_selective_config(
        model_path="results/gene_cv_run_3/best_model.json",
        target_genes=["ENSG00000104435", "ENSG00000130477"],  # STMN2, UNC13A
        training_dataset_path="train_pc_1000_3mers/master",
        max_positions_per_gene=3000
    )
    
    results = run_selective_meta_inference(config)
    
    if results.success:
        print(f"✅ Success! {results.positions_recalibrated:,} positions recalibrated, {results.positions_reused:,} reused")
        print(f"📊 Complete coverage: {results.total_positions:,} total positions")
    else:
        print(f"❌ Failed: {results.error_messages}")
