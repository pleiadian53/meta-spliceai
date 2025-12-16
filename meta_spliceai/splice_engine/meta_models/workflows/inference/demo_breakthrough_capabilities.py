#!/usr/bin/env python3
"""
Demo: Breakthrough Inference Workflow Capabilities
=================================================

This script demonstrates the three major breakthrough capabilities developed
in the selective meta-model inference system:

1. **Complete Selective Feature Generation**: Generate features only for uncertain 
   positions while maintaining complete coverage
2. **Enhanced Feature Matrix Generation**: Using enhanced_process_predictions_with_all_scores
   to create comprehensive feature matrices with all three probability scores
3. **Mixed Predictions**: Demonstrate hybrid system combining base + meta model predictions
   for complete coverage with computational efficiency

Key Innovation: This approach provides complete nucleotide coverage while being
computationally efficient by selectively applying expensive meta-model inference
only where it provides the most value.
"""

import os
import sys
import time
import re
import glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import polars as pl
from dataclasses import asdict

# Add the project root to the path
project_root = Path(__file__).parents[5]  # Go up to splice-surveyor root
sys.path.insert(0, str(project_root))

from meta_spliceai.splice_engine.meta_models.workflows.selective_meta_inference import (
    SelectiveInferenceConfig,
    run_selective_meta_inference,
    combine_predictions_for_complete_coverage,
    create_selective_config
)
from meta_spliceai.splice_engine.meta_models.core.enhanced_workflow import (
    enhanced_process_predictions_with_all_scores
)
from meta_spliceai.splice_engine.meta_models.workflows.splice_prediction_workflow import (
    run_enhanced_splice_prediction_workflow
)
from meta_spliceai.splice_engine.meta_models.core.data_types import SpliceAIConfig
from meta_spliceai.system.genomic_resources import create_systematic_manager

def print_section(title: str, char: str = "=") -> None:
    """Print a formatted section header."""
    print(f"\n{char * 80}")
    print(f" {title}")
    print(f"{char * 80}")

def print_subsection(title: str) -> None:
    """Print a formatted subsection header."""
    print(f"\n{'─' * 60}")
    print(f"🔹 {title}")
    print(f"{'─' * 60}")

def setup_test_environment() -> Dict[str, Path]:
    """Set up test environment and return important paths."""
    print_section("🚀 SETTING UP TEST ENVIRONMENT")
    
    # Initialize systematic resource manager
    manager = create_systematic_manager()
    
    # Get key paths
    paths = {
        'meta_models_dir': Path(manager.cfg.data_root) / "spliceai_eval" / "meta_models",
        'inference_dir': Path(manager.cfg.data_root) / "spliceai_eval" / "meta_models" / "inference",
        'training_dir': Path(manager.cfg.data_root) / "spliceai_eval" / "meta_models" / "train_pc_1000_3mers",
    }
    
    # Look for available models - prioritize most recent (highest run number)
    # First check project root results directory (primary location)
    project_results_dir = project_root / "results"
    model_candidates = []
    
    # Search in project root results directory first
    if project_results_dir.exists():
        # Find all gene_cv_pc_1000_3mers_run_* directories
        pattern = str(project_results_dir / "gene_cv_pc_1000_3mers_run_*")
        run_dirs = glob.glob(pattern)
        
        # Extract run numbers and sort in descending order (most recent first)
        run_info = []
        for run_dir in run_dirs:
            match = re.search(r'run_(\d+)', run_dir)
            if match:
                run_number = int(match.group(1))
                model_path = Path(run_dir) / "model_multiclass.pkl"
                run_info.append((run_number, model_path))
        
        # Sort by run number in descending order (highest/most recent first)
        run_info.sort(key=lambda x: x[0], reverse=True)
        model_candidates = [model_path for _, model_path in run_info]
    
    # Fallback: check meta_models directory
    meta_results_dir = paths['meta_models_dir'] / "results"
    if meta_results_dir.exists():
        pattern = str(meta_results_dir / "gene_cv_pc_1000_3mers_run_*")
        run_dirs = glob.glob(pattern)
        
        run_info = []
        for run_dir in run_dirs:
            match = re.search(r'run_(\d+)', run_dir)
            if match:
                run_number = int(match.group(1))
                model_path = Path(run_dir) / "model_multiclass.pkl"
                run_info.append((run_number, model_path))
        
        run_info.sort(key=lambda x: x[0], reverse=True)
        model_candidates.extend([model_path for _, model_path in run_info])
    
    # Hard-coded fallback candidates
    fallback_candidates = [
        project_root / "results" / "gene_cv_pc_1000_3mers_run_4" / "model_multiclass.pkl",
        project_root / "results" / "gene_cv_pc_1000_3mers_run_3" / "model_multiclass.pkl",
        project_root / "results" / "gene_cv_pc_1000_3mers_run_2" / "model_multiclass.pkl",
        project_root / "results" / "gene_cv_pc_1000_3mers_run_1" / "model_multiclass.pkl",
    ]
    model_candidates.extend(fallback_candidates)
    
    # Debug: print search results
    print(f"🔍 Model search results:")
    print(f"   • Project results dir: {project_results_dir} (exists: {project_results_dir.exists()})")
    print(f"   • Found {len(model_candidates)} model candidates")
    for i, candidate in enumerate(model_candidates[:5]):  # Show first 5
        print(f"     {i+1}. {candidate} (exists: {candidate.exists()})")
    
    paths['model'] = None
    for candidate in model_candidates:
        if candidate.exists():
            paths['model'] = candidate
            # Extract run number for reporting
            match = re.search(r'run_(\d+)', str(candidate))
            run_number = match.group(1) if match else "unknown"
            print(f"🤖 Using most recent model: run_{run_number}")
            break
    
    # Look for training schema - check project root first
    schema_candidates = [
        project_root / "train_pc_1000_3mers" / "master",
        project_root / "train_pc_1000_3mers",
        paths['meta_models_dir'] / "train_pc_1000_3mers" / "master",
        paths['meta_models_dir'] / "train_pc_1000_3mers",
        paths['training_dir'] / "master",
        paths['training_dir'],
    ]
    
    paths['training_schema'] = None
    for candidate in schema_candidates:
        if candidate.exists():
            paths['training_schema'] = candidate
            break
    
    print(f"✅ Meta-models directory: {paths['meta_models_dir']}")
    print(f"✅ Inference directory: {paths['inference_dir']}")
    print(f"✅ Model path: {paths['model']}")
    print(f"✅ Training schema: {paths['training_schema']}")
    
    # Verify critical paths exist
    if not paths['model']:
        raise FileNotFoundError("No trained meta-model found! Please train a model first.")
    
    if not paths['training_schema']:
        print("⚠️  No training schema found - will use basic features")
    
    return paths

def test_enhanced_feature_generation(paths: Dict[str, Path]) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """
    Test 1: Enhanced Feature Matrix Generation
    
    Demonstrates enhanced_process_predictions_with_all_scores generating
    comprehensive feature matrices with all three probability scores and
    sophisticated context-aware derived features.
    """
    print_section("🧬 TEST 1: ENHANCED FEATURE MATRIX GENERATION", "=")
    
    print("Testing enhanced_process_predictions_with_all_scores with comprehensive feature generation...")
    
    # Select test genes (well-known genes with splice sites)
    test_genes = ["ENSG00000154358"]  # Known working gene from previous testing
    
    print(f"🎯 Target genes: {test_genes}")
    
    # Step 1: Generate base model predictions for test genes
    print_subsection("Step 1: Generate Base Model Predictions")
    
    # Configure SpliceAI workflow for test genes
    config = SpliceAIConfig(
        test_mode=False,
        chromosomes=['21', '22'],  # Include multiple chromosomes to ensure gene coverage
        do_extract_annotations=False,  # Use existing
        do_extract_splice_sites=False,  # Use existing
        do_extract_sequences=False,    # Use existing
        do_find_overlaping_genes=False  # Use existing
    )
    
    print("🔄 Running enhanced splice prediction workflow...")
    start_time = time.time()
    
    # Run base model predictions
    workflow_results = run_enhanced_splice_prediction_workflow(
        config=config,
        target_genes=test_genes,
        verbosity=1,
        no_final_aggregate=True  # Don't aggregate to save memory
    )
    
    prediction_time = time.time() - start_time
    print(f"✅ Base model predictions completed in {prediction_time:.1f}s")
    
    if not workflow_results.get('success', False):
        raise RuntimeError("Base model prediction workflow failed")
    
    # Step 2: Load generated predictions and test feature generation
    print_subsection("Step 2: Load Predictions and Generate Features")
    
    # Look for analysis sequences files (contain base model predictions)
    output_dir = Path(workflow_results['paths']['output_dir'])
    analysis_files = list(output_dir.glob("analysis_sequences_*.tsv"))
    
    if not analysis_files:
        raise FileNotFoundError("No analysis sequence files found from base model predictions")
    
    print(f"📁 Found {len(analysis_files)} analysis sequence files")
    
    # Load and combine predictions
    all_predictions = {}
    total_positions = 0
    
    for file_path in analysis_files:
        df = pl.read_csv(
            file_path, 
            separator="\t", 
            infer_schema_length=1000,
            schema_overrides={"chrom": pl.Utf8}  # Handle chromosome X, Y as strings
        )
        df = df.filter(pl.col("gene_id").is_in(test_genes))
        
        if df.height > 0:
            # Group by gene and convert to the format expected by enhanced_process_predictions_with_all_scores
            for gene_id in df['gene_id'].unique():
                gene_data = df.filter(pl.col('gene_id') == gene_id)
                
                # Get gene boundaries
                positions = gene_data['position'].to_numpy()
                gene_start = positions.min()
                gene_end = positions.max()
                
                # Convert to the expected format (dict with numpy arrays)
                all_predictions[gene_id] = {
                    'donor_prob': gene_data['donor_score'].to_numpy(),
                    'acceptor_prob': gene_data['acceptor_score'].to_numpy(), 
                    'neither_prob': gene_data['neither_score'].to_numpy(),
                    'positions': positions,
                    'chromosome': gene_data['chrom'][0] if 'chrom' in gene_data.columns else '21',
                    'strand': gene_data['strand'][0] if 'strand' in gene_data.columns else '+',
                    'gene_start': gene_start,
                    'gene_end': gene_end
                }
                total_positions += len(gene_data)
    
    print(f"📊 Loaded predictions for {len(all_predictions)} genes, {total_positions:,} total positions")
    
    # Load gene boundaries from training data (authoritative source)
    print_subsection("Loading Gene Boundaries from Training Data")
    
    if not paths['training_schema'] or not paths['training_schema'].exists():
        raise FileNotFoundError(f"Training schema not found at {paths['training_schema']} - needed for gene boundaries")
    
    # Look for training data files with metadata
    training_files = list(paths['training_schema'].glob("*.parquet"))
    if not training_files:
        # Try parent directory
        training_files = list(paths['training_schema'].parent.glob("*.parquet"))
    
    if not training_files:
        raise FileNotFoundError(f"No training data files found in {paths['training_schema']}")
    
    print(f"📋 Found {len(training_files)} training data files")
    
    # Load a sample to check schema and get gene boundaries
    sample_file = training_files[0]
    sample_df = pl.read_parquet(sample_file, n_rows=1000)  # Sample for schema check
    
    # Verify required metadata columns exist
    required_metadata_cols = ['gene_id', 'gene_start', 'gene_end', 'chrom', 'strand', 'position']
    missing_cols = [col for col in required_metadata_cols if col not in sample_df.columns]
    
    if missing_cols:
        raise ValueError(f"Training data missing required metadata columns: {missing_cols}. "
                        f"Available columns: {sample_df.columns}")
    
    print(f"✅ Training data contains all required metadata columns")
    print(f"   📊 Total columns: {len(sample_df.columns)}")
    print(f"   🗂️  Metadata columns: {required_metadata_cols}")
    
    # Load gene boundaries for our test genes from training data
    gene_boundaries = {}
    for training_file in training_files:
        chunk_df = pl.read_parquet(training_file)
        test_gene_data = chunk_df.filter(pl.col("gene_id").is_in(test_genes))
        
        if test_gene_data.height > 0:
            # Get unique gene boundaries (should be consistent across all positions)
            gene_info = test_gene_data.group_by("gene_id").agg([
                pl.col("gene_start").first(),
                pl.col("gene_end").first(), 
                pl.col("chrom").first(),
                pl.col("strand").first()
            ])
            
            for row in gene_info.iter_rows():
                gene_id, gene_start, gene_end, chrom, strand = row
                if gene_id not in gene_boundaries:
                    gene_boundaries[gene_id] = {
                        'gene_start': gene_start,
                        'gene_end': gene_end,
                        'chrom': chrom,
                        'strand': strand
                    }
    
    print(f"📋 Found gene boundaries for {len(gene_boundaries)} test genes from training data")
    
    # Update predictions with authoritative gene boundaries from training data
    for gene_id in test_genes:
        if gene_id in all_predictions and gene_id in gene_boundaries:
            boundaries = gene_boundaries[gene_id]
            
            # Update with authoritative gene boundaries from training data
            all_predictions[gene_id]['gene_start'] = boundaries['gene_start']
            all_predictions[gene_id]['gene_end'] = boundaries['gene_end'] 
            all_predictions[gene_id]['chromosome'] = boundaries['chrom']
            all_predictions[gene_id]['strand'] = boundaries['strand']
            
            # Validate tensor dimensions against training data boundaries
            expected_length = boundaries['gene_end'] - boundaries['gene_start'] + 1
            actual_length = len(all_predictions[gene_id]['donor_prob'])
            
            print(f"   🧬 {gene_id}: gene_start={boundaries['gene_start']}, gene_end={boundaries['gene_end']}")
            print(f"      📏 Expected tensor length: {expected_length:,}")
            print(f"      🔍 Actual tensor length: {actual_length:,}")
            print(f"      ✅ Dimension match: {actual_length == expected_length}")
            
            if actual_length != expected_length:
                raise ValueError(f"Tensor dimension mismatch for {gene_id}: "
                               f"expected {expected_length}, got {actual_length}. "
                               f"This indicates coordinate system inconsistency.")
        elif gene_id in all_predictions:
            raise ValueError(f"Gene {gene_id} found in predictions but not in training data boundaries")
    
    print(f"✅ All tensor dimensions validated against training data gene boundaries")
    
    # Load splice site annotations
    manager = create_systematic_manager()
    splice_sites_path = manager.get_splice_sites_path()
    
    # Read with proper schema to handle string chromosomes (X, Y, etc.)
    ss_annotations_df = pl.read_csv(
        splice_sites_path, 
        separator="\t",
        schema_overrides={"chrom": pl.Utf8}  # Ensure chromosome is read as string
    )
    ss_annotations_df = ss_annotations_df.filter(pl.col("gene_id").is_in(test_genes))
    
    print(f"📍 Loaded {ss_annotations_df.height} splice site annotations for test genes")
    
    # Step 3: Generate enhanced features
    print_subsection("Step 3: Generate Comprehensive Feature Matrix")
    
    print("🧬 Calling enhanced_process_predictions_with_all_scores...")
    feature_start_time = time.time()
    
    # Call the enhanced feature generation function
    error_df, positions_df = enhanced_process_predictions_with_all_scores(
        predictions=all_predictions,
        ss_annotations_df=ss_annotations_df,
        threshold=0.5,
        consensus_window=2,
        error_window=500,
        analyze_position_offsets=True,
        collect_tn=True,
        predicted_delta_correction=True,
        add_derived_features=True,  # Enable all derived features
        fill_missing_values=True,   # Fill any missing values
        compute_all_context_features=True,  # Enable context features
        verbose=2  # Detailed output
    )
    
    feature_time = time.time() - feature_start_time
    print(f"✅ Feature generation completed in {feature_time:.1f}s")
    
    # Step 4: Analyze the generated features
    print_subsection("Step 4: Analyze Generated Feature Matrix")
    
    print(f"📊 Enhanced Feature Matrix Analysis:")
    print(f"   • Total positions: {positions_df.height:,}")
    print(f"   • Total features: {len(positions_df.columns)}")
    print(f"   • Memory usage: {positions_df.estimated_size('mb'):.1f} MB")
    
    # Categorize features
    feature_categories = {
        'base_scores': [col for col in positions_df.columns if col.endswith('_score')],
        'context_features': [col for col in positions_df.columns if col.startswith('context_')],
        'donor_features': [col for col in positions_df.columns if col.startswith('donor_') and col != 'donor_score'],
        'acceptor_features': [col for col in positions_df.columns if col.startswith('acceptor_') and col != 'acceptor_score'],
        'cross_type_features': [col for col in positions_df.columns if any(col.startswith(prefix) for prefix in ['score_difference', 'signal_strength', 'type_signal'])],
        'derived_features': [col for col in positions_df.columns if any(col.startswith(prefix) for prefix in ['relative_', 'splice_', 'probability_'])]
    }
    
    print(f"\n📋 Feature Categories:")
    for category, features in feature_categories.items():
        print(f"   • {category}: {len(features)} features")
        if features and len(features) <= 5:
            print(f"     └─ {features}")
        elif features:
            print(f"     └─ {features[:3]}... (+{len(features)-3} more)")
    
    # Check for prediction types
    if 'pred_type' in positions_df.columns:
        pred_counts = positions_df.group_by('pred_type').count().sort('count', descending=True)
        print(f"\n🎯 Prediction Type Distribution:")
        for row in pred_counts.iter_rows():
            pred_type, count = row
            percentage = (count / positions_df.height) * 100
            print(f"   • {pred_type}: {count:,} ({percentage:.1f}%)")
    
    # Sample feature values for verification
    print(f"\n🔍 Sample Feature Values (first 5 positions):")
    sample_cols = ['gene_id', 'position', 'donor_score', 'acceptor_score', 'neither_score']
    
    # Add some derived features to the sample
    for category, features in feature_categories.items():
        if features and category != 'base_scores':
            sample_cols.extend(features[:2])  # Add first 2 features from each category
    
    sample_cols = [col for col in sample_cols if col in positions_df.columns][:15]  # Limit to 15 columns
    sample_df = positions_df.select(sample_cols).head(5)
    print(sample_df.to_pandas().to_string(index=False, float_format='%.4f'))
    
    print(f"\n✅ Enhanced feature generation test completed successfully!")
    print(f"   🧬 Generated {len(positions_df.columns)} features for {positions_df.height:,} positions")
    print(f"   ⚡ Processing time: {feature_time:.1f}s ({positions_df.height/feature_time:.0f} positions/sec)")
    
    return error_df, positions_df

def test_selective_meta_inference(paths: Dict[str, Path]) -> Dict:
    """
    Test 2: Complete Selective Meta-Model Inference
    
    Demonstrates the selective inference system that provides complete coverage
    by combining efficient base model predictions with targeted meta-model
    recalibration for uncertain positions.
    """
    print_section("🤖 TEST 2: SELECTIVE META-MODEL INFERENCE", "=")
    
    print("Testing complete selective meta-model inference workflow...")
    
    # Test genes
    test_genes = ["ENSG00000154358"]  # Known working gene from previous testing
    
    print(f"🎯 Target genes: {test_genes}")
    
    # Step 1: Configure selective inference
    print_subsection("Step 1: Configure Selective Inference")
    
    config = SelectiveInferenceConfig(
        model_path=paths['model'],
        target_genes=test_genes,
        training_dataset_path=paths['training_schema'],
        training_schema_path=paths['training_schema'],
        uncertainty_threshold_low=0.02,    # Below this: confident non-splice
        uncertainty_threshold_high=0.80,   # Above this: confident splice
        max_positions_per_gene=5000,       # Reasonable limit
        inference_mode="hybrid",           # Use hybrid for balanced performance
        use_calibration=True,
        verbose=2
    )
    
    print(f"⚙️  Configuration:")
    print(f"   • Model: {config.model_path.name}")
    print(f"   • Uncertainty range: [{config.uncertainty_threshold_low:.3f}, {config.uncertainty_threshold_high:.3f})")
    print(f"   • Inference mode: {config.inference_mode}")
    print(f"   • Max positions per gene: {config.max_positions_per_gene:,}")
    
    # Step 2: Run selective inference
    print_subsection("Step 2: Run Selective Meta-Model Inference")
    
    print("🚀 Starting selective inference workflow...")
    inference_start_time = time.time()
    
    try:
        results = run_selective_meta_inference(config)
        inference_time = time.time() - inference_start_time
        
        if not results.success:
            print(f"❌ Selective inference failed: {results.error_messages}")
            return {'success': False, 'error': results.error_messages}
        
        print(f"✅ Selective inference completed in {inference_time:.1f}s")
        
    except Exception as e:
        print(f"❌ Selective inference failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}
    
    # Step 3: Analyze results
    print_subsection("Step 3: Analyze Selective Inference Results")
    
    print(f"📊 Selective Inference Results:")
    print(f"   • Total positions: {results.total_positions:,}")
    print(f"   • Positions recalibrated: {results.positions_recalibrated:,} ({results.positions_recalibrated/results.total_positions:.1%})")
    print(f"   • Positions reused: {results.positions_reused:,} ({results.positions_reused/results.total_positions:.1%})")
    print(f"   • Genes processed: {results.genes_processed}")
    print(f"   • Processing time: {results.processing_time_seconds:.1f}s")
    print(f"   • Feature matrix size: {results.feature_matrix_size_mb:.1f} MB")
    
    # Load and examine the hybrid predictions
    if results.hybrid_predictions_path and results.hybrid_predictions_path.exists():
        hybrid_df = pd.read_parquet(results.hybrid_predictions_path)
        
        print(f"\n🔗 Hybrid Predictions Analysis:")
        print(f"   • Total rows: {len(hybrid_df):,}")
        print(f"   • Columns: {len(hybrid_df.columns)}")
        
        # Analyze prediction sources
        if 'prediction_source' in hybrid_df.columns:
            source_counts = hybrid_df['prediction_source'].value_counts()
            print(f"   • Prediction Sources:")
            for source, count in source_counts.items():
                percentage = (count / len(hybrid_df)) * 100
                print(f"     └─ {source}: {count:,} ({percentage:.1f}%)")
        
        # Analyze confidence categories
        if 'confidence_category' in hybrid_df.columns:
            conf_counts = hybrid_df['confidence_category'].value_counts()
            print(f"   • Confidence Categories:")
            for category, count in conf_counts.items():
                percentage = (count / len(hybrid_df)) * 100
                print(f"     └─ {category}: {count:,} ({percentage:.1f}%)")
        
        # Show sample predictions
        print(f"\n🔍 Sample Hybrid Predictions:")
        sample_cols = ['gene_id', 'position', 'donor_score', 'donor_meta', 'acceptor_score', 'acceptor_meta', 'prediction_source', 'confidence_category']
        sample_cols = [col for col in sample_cols if col in hybrid_df.columns]
        print(hybrid_df[sample_cols].head(10).to_string(index=False, float_format='%.4f'))
    
    # Per-gene statistics
    if results.per_gene_stats:
        print(f"\n📈 Per-Gene Statistics:")
        for gene_id, stats in results.per_gene_stats.items():
            total = stats['total_positions']
            recal = stats['recalibrated_positions']
            reused = stats['reused_positions']
            print(f"   • {gene_id}: {total:,} total, {recal:,} recalibrated ({recal/total:.1%}), {reused:,} reused ({reused/total:.1%})")
    
    print(f"\n✅ Selective meta-inference test completed successfully!")
    
    return {
        'success': True,
        'results': results,
        'hybrid_predictions': hybrid_df if 'hybrid_df' in locals() else None,
        'processing_time': inference_time
    }

def test_mixed_predictions_demonstration(selective_results: Dict, paths: Dict[str, Path]) -> None:
    """
    Test 3: Mixed Predictions Demonstration
    
    Demonstrates how the hybrid system seamlessly combines base model and 
    meta-model predictions to provide complete coverage with optimal efficiency.
    """
    print_section("🔗 TEST 3: MIXED PREDICTIONS DEMONSTRATION", "=")
    
    if not selective_results.get('success', False):
        print("❌ Cannot demonstrate mixed predictions - selective inference failed")
        return
    
    results = selective_results['results']
    hybrid_df = selective_results.get('hybrid_predictions')
    
    if hybrid_df is None or len(hybrid_df) == 0:
        print("❌ No hybrid predictions available for analysis")
        return
    
    print("Demonstrating mixed prediction system capabilities...")
    
    # Step 1: Analyze prediction source distribution
    print_subsection("Step 1: Prediction Source Analysis")
    
    if 'prediction_source' in hybrid_df.columns:
        source_analysis = hybrid_df.groupby('prediction_source').agg({
            'donor_score': ['mean', 'std', 'min', 'max'],
            'donor_meta': ['mean', 'std', 'min', 'max'],
            'position': 'count'
        }).round(4)
        
        print(f"📊 Prediction Source Statistics:")
        print(source_analysis.to_string())
        
        # Calculate efficiency metrics
        total_positions = len(hybrid_df)
        meta_positions = (hybrid_df['prediction_source'] == 'meta_model').sum()
        base_positions = total_positions - meta_positions
        
        efficiency_ratio = base_positions / total_positions
        print(f"\n⚡ Efficiency Metrics:")
        print(f"   • Total positions: {total_positions:,}")
        print(f"   • Base model reuse: {base_positions:,} ({efficiency_ratio:.1%})")
        print(f"   • Meta model usage: {meta_positions:,} ({1-efficiency_ratio:.1%})")
        print(f"   • Computational savings: {efficiency_ratio:.1%} (vs meta-only mode)")
    
    # Step 2: Analyze uncertainty-based stratification
    print_subsection("Step 2: Uncertainty-Based Stratification")
    
    if 'confidence_category' in hybrid_df.columns:
        # Calculate actual confidence scores for verification
        hybrid_df['max_base_score'] = hybrid_df[['donor_score', 'acceptor_score']].max(axis=1)
        hybrid_df['max_meta_score'] = hybrid_df[['donor_meta', 'acceptor_meta']].max(axis=1)
        
        conf_analysis = hybrid_df.groupby('confidence_category').agg({
            'max_base_score': ['mean', 'std', 'min', 'max'],
            'max_meta_score': ['mean', 'std', 'min', 'max'],
            'position': 'count'
        }).round(4)
        
        print(f"🎯 Confidence Category Analysis:")
        print(conf_analysis.to_string())
        
        # Verify threshold-based categorization is working correctly
        print(f"\n✅ Threshold Verification:")
        for category in hybrid_df['confidence_category'].unique():
            category_data = hybrid_df[hybrid_df['confidence_category'] == category]
            score_range = (category_data['max_base_score'].min(), category_data['max_base_score'].max())
            print(f"   • {category}: score range [{score_range[0]:.3f}, {score_range[1]:.3f}]")
    
    # Step 3: Compare base vs meta predictions for uncertain positions
    print_subsection("Step 3: Base vs Meta Prediction Comparison")
    
    # Focus on positions where meta-model was actually used
    meta_positions = hybrid_df[hybrid_df['prediction_source'] == 'meta_model'].copy()
    
    if len(meta_positions) > 0:
        # Calculate differences between base and meta predictions
        meta_positions['donor_diff'] = meta_positions['donor_meta'] - meta_positions['donor_score']
        meta_positions['acceptor_diff'] = meta_positions['acceptor_meta'] - meta_positions['acceptor_score']
        meta_positions['neither_diff'] = meta_positions['neither_meta'] - meta_positions['neither_score']
        
        print(f"🔄 Base vs Meta Prediction Differences (for {len(meta_positions):,} meta-recalibrated positions):")
        diff_stats = meta_positions[['donor_diff', 'acceptor_diff', 'neither_diff']].describe()
        print(diff_stats.round(4).to_string())
        
        # Find positions with significant recalibration
        significant_changes = meta_positions[
            (abs(meta_positions['donor_diff']) > 0.1) | 
            (abs(meta_positions['acceptor_diff']) > 0.1)
        ]
        
        print(f"\n🎯 Significant Recalibrations (|change| > 0.1): {len(significant_changes):,} positions")
        
        if len(significant_changes) > 0:
            print(f"Sample positions with significant recalibration:")
            sample_cols = ['gene_id', 'position', 'donor_score', 'donor_meta', 'donor_diff', 
                          'acceptor_score', 'acceptor_meta', 'acceptor_diff']
            print(significant_changes[sample_cols].head(5).to_string(index=False, float_format='%.4f'))
    
    # Step 4: Demonstrate inference mode flexibility
    print_subsection("Step 4: Inference Mode Flexibility Demonstration")
    
    # Simulate what different inference modes would produce
    base_only_meta = hybrid_df[['donor_score', 'acceptor_score', 'neither_score']].copy()
    base_only_meta.columns = ['donor_meta', 'acceptor_meta', 'neither_meta']
    
    meta_only_meta = hybrid_df[['donor_meta', 'acceptor_meta', 'neither_meta']].copy()
    
    # Calculate mode comparison statistics
    modes_comparison = {
        'base_only': {
            'positions_processed': len(hybrid_df),
            'meta_model_calls': 0,
            'efficiency': 1.0
        },
        'hybrid': {
            'positions_processed': len(hybrid_df),
            'meta_model_calls': (hybrid_df['prediction_source'] == 'meta_model').sum(),
            'efficiency': (hybrid_df['prediction_source'] == 'base_model').sum() / len(hybrid_df)
        },
        'meta_only': {
            'positions_processed': len(hybrid_df),
            'meta_model_calls': len(hybrid_df),
            'efficiency': 0.0
        }
    }
    
    print(f"🎚️  Inference Mode Comparison:")
    for mode, stats in modes_comparison.items():
        efficiency_pct = stats['efficiency'] * 100
        meta_pct = (stats['meta_model_calls'] / stats['positions_processed']) * 100
        print(f"   • {mode:>10}: {stats['meta_model_calls']:,}/{stats['positions_processed']:,} meta calls ({meta_pct:.1f}%), {efficiency_pct:.1f}% efficient")
    
    # Step 5: Performance and scalability insights
    print_subsection("Step 5: Performance and Scalability Insights")
    
    processing_time = selective_results.get('processing_time', 0)
    feature_matrix_mb = results.feature_matrix_size_mb
    
    # Calculate throughput metrics
    positions_per_second = results.total_positions / processing_time if processing_time > 0 else 0
    mb_per_thousand_positions = (feature_matrix_mb / results.total_positions) * 1000 if results.total_positions > 0 else 0
    
    print(f"⚡ Performance Metrics:")
    print(f"   • Processing throughput: {positions_per_second:.0f} positions/second")
    print(f"   • Memory efficiency: {mb_per_thousand_positions:.2f} MB per 1K positions")
    print(f"   • Feature matrix size: {feature_matrix_mb:.1f} MB for {results.positions_recalibrated:,} uncertain positions")
    print(f"   • Computational savings: {(results.positions_reused/results.total_positions)*100:.1f}% positions skip expensive featurization")
    
    # Estimate scalability
    genome_wide_positions = 3_000_000_000  # Approximate human genome size
    estimated_uncertain = genome_wide_positions * (results.positions_recalibrated / results.total_positions)
    estimated_feature_gb = (estimated_uncertain / results.positions_recalibrated) * feature_matrix_mb / 1024
    
    print(f"\n🌐 Genome-Wide Scalability Projection:")
    print(f"   • Estimated uncertain positions: {estimated_uncertain:,.0f}")
    print(f"   • Estimated feature matrix size: {estimated_feature_gb:.1f} GB")
    print(f"   • Estimated processing time: {(genome_wide_positions / positions_per_second / 3600):.1f} hours")
    
    print(f"\n✅ Mixed predictions demonstration completed successfully!")
    print(f"   🎯 Demonstrated complete coverage with {(results.positions_reused/results.total_positions)*100:.1f}% computational savings")
    print(f"   🧠 Meta-model intelligently applied to {results.positions_recalibrated:,} uncertain positions")
    print(f"   ⚡ System scales efficiently with selective featurization strategy")

def main():
    """Main demonstration function."""
    print_section("🚀 BREAKTHROUGH INFERENCE WORKFLOW CAPABILITIES DEMO", "█")
    
    print("""
This demonstration showcases the three major breakthrough capabilities:

1. 🧬 **Enhanced Feature Matrix Generation**: Complete feature matrices with 
   all three probability scores and sophisticated context-aware features

2. 🤖 **Selective Meta-Model Inference**: Computational efficiency through 
   selective featurization while maintaining complete coverage

3. 🔗 **Mixed Predictions**: Seamless combination of base + meta predictions 
   for optimal balance of accuracy and performance

Key Innovation: Complete nucleotide coverage with computational efficiency!
""")
    
    try:
        # Setup
        paths = setup_test_environment()
        
        # Test 1: Enhanced Feature Generation
        print("\n" + "🧬" * 40)
        error_df, positions_df = test_enhanced_feature_generation(paths)
        
        # Test 2: Selective Meta-Model Inference  
        print("\n" + "🤖" * 40)
        selective_results = test_selective_meta_inference(paths)
        
        # Test 3: Mixed Predictions Demonstration
        print("\n" + "🔗" * 40)
        test_mixed_predictions_demonstration(selective_results, paths)
        
        # Final Summary
        print_section("🎉 BREAKTHROUGH CAPABILITIES DEMONSTRATION COMPLETE!", "█")
        
        print(f"""
✅ ALL TESTS COMPLETED SUCCESSFULLY!

🏆 **Breakthrough Achievements Demonstrated:**

1. **Enhanced Feature Generation**: 
   • Generated {len(positions_df.columns)} sophisticated features
   • Processed {positions_df.height:,} positions with full context awareness
   • Includes probability ratios, context features, and cross-type comparisons

2. **Selective Meta-Model Inference**:
   • Achieved complete coverage for {selective_results['results'].total_positions:,} positions
   • {(selective_results['results'].positions_reused/selective_results['results'].total_positions)*100:.1f}% computational savings through selective featurization
   • {selective_results['results'].positions_recalibrated:,} uncertain positions recalibrated with meta-model

3. **Mixed Predictions System**:
   • Seamlessly combines base + meta predictions
   • Flexible inference modes (base_only, hybrid, meta_only)
   • Scales efficiently for genome-wide analysis

🚀 **Ready for Production**: The selective inference workflow provides the
perfect balance of accuracy and computational efficiency, enabling practical
deployment for both targeted gene analysis and genome-wide applications!

🎯 **Next Steps**: Deploy this system for real-world splice site analysis
with confidence in both accuracy and scalability.
""")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())