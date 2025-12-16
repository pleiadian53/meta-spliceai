"""
Feature processing for selective meta-model inference.

This module handles feature generation for selective positions using the StandardizedFeaturizer.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional

from meta_spliceai.splice_engine.meta_models.features.standardized_featurizer import (
    StandardizedFeaturizer,
    create_featurizer_from_model_path
)
from ..inference_workflow_utils import load_model_with_calibration
from ..selective_feature_generator import _load_training_schema
from ..chunked_meta_processor import create_chunked_processor
from .config import SelectiveInferenceConfig
from .verification import verify_selective_featurization, verify_no_label_leakage


class FeatureGeneratorWrapper:
    """
    Thin wrapper around StandardizedFeaturizer for compatibility with existing code.
    """
    def __init__(self, model_path=None, verbose=False):
        self.model_path = model_path
        self.verbose = verbose
        
    def generate_features(self, analysis_df, base_df, training_schema=None):
        """Generate features for the given positions.
        
        Uses the StandardizedFeaturizer to ensure consistent feature generation
        across training and inference workflows.
        """
        # Create featurizer with training schema and excluded features
        if self.model_path:
            # Use helper to load schema and excluded features from model directory
            featurizer = create_featurizer_from_model_path(
                self.model_path,
                verbose=self.verbose
            )
        else:
            # Fallback to basic featurizer
            featurizer = StandardizedFeaturizer(
                kmer_sizes=[3],  # Default to 3-mers, could be made configurable
                training_schema=training_schema,
                verbose=self.verbose
            )
        
        # Generate features using the standardized pipeline
        features_df = featurizer.featurize_from_analysis_df(
            analysis_df=analysis_df,
            base_predictions_df=base_df,
            include_probability_features=True,
            include_context_features=True,
            include_kmer_features=True,
            include_genomic_features=True,
            harmonize_with_training=(training_schema is not None)
        )
        
        return features_df


def generate_chunked_meta_predictions(
    config: SelectiveInferenceConfig,
    complete_base_pd: pd.DataFrame,
    workflow_results: Dict,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Generate meta-model predictions for ALL positions using chunked processing.
    
    This function enables true "meta_only" mode by processing positions in chunks
    to avoid memory issues.
    
    Parameters
    ----------
    config : SelectiveInferenceConfig
        Configuration for selective inference
    complete_base_pd : pd.DataFrame
        Complete base model predictions
    workflow_results : Dict
        Results from the inference workflow
    verbose : bool, default=True
        Enable verbose output
        
    Returns
    -------
    pd.DataFrame
        Meta-model predictions for all positions
    """
    if verbose:
        print("   📊 Initializing chunked meta-model processing for full coverage")
    
    # Load model (which may include calibration)
    model = load_model_with_calibration(
        config.model_path,
        use_calibration=config.use_calibration
    )
    
    # The model itself may be a calibrated ensemble
    calibrator = None  # Calibration is handled within the model
    
    # Load training schema if available
    training_schema = None
    if config.training_schema_path:
        try:
            # Use the new model resource manager for systematic schema loading
            from .model_resource_manager import create_model_resource_manager
            model_manager = create_model_resource_manager()
            training_schema = model_manager.load_feature_schema(config.training_schema_path)
            
            if training_schema is None:
                # Fallback to old method
                schema_path = config.training_schema_path
                if schema_path.name == "master" and schema_path.parent.exists():
                    schema_path = schema_path.parent
                training_schema = _load_training_schema(schema_path)
                
            if verbose and training_schema:
                print(f"   📋 Loaded training schema from {config.training_schema_path}")
        except Exception as e:
            if verbose:
                print(f"   ⚠️  Could not load training schema: {e}")
            training_schema = None
    
    # Get analysis sequences
    if 'analysis_sequences' not in workflow_results:
        if config.inference_mode == "meta_only":
            # For meta_only mode, we must have analysis sequences - don't fall back to base predictions
            raise RuntimeError("Meta-only mode requires analysis_sequences but none were found. "
                             "This indicates a problem with the complete coverage workflow.")
        else:
            if verbose:
                print("   ⚠️  No analysis sequences available; returning empty predictions")
            return pd.DataFrame(columns=['gene_id', 'position', 'donor_meta', 'acceptor_meta', 'neither_meta'])
    
    analysis_sequences = workflow_results['analysis_sequences']
    analysis_sequences_pd = (analysis_sequences.to_pandas()
                             if hasattr(analysis_sequences, 'to_pandas') 
                             else analysis_sequences)
    
    # Create feature generator with model path for loading excluded features
    model_dir = config.model_path if config.model_path.is_dir() else config.model_path.parent
    feature_generator = FeatureGeneratorWrapper(model_path=model_dir, verbose=verbose > 1)
    
    # Create chunked processor with inference mode
    processor = create_chunked_processor(
        chunk_size=config.chunk_size,
        inference_mode=config.inference_mode,
        verbose=config.verbose
    )
    
    # Get unique genes
    genes = complete_base_pd['gene_id'].unique()
    
    if verbose:
        total_positions = len(complete_base_pd)
        print(f"   🧬 Processing {total_positions:,} positions across {len(genes)} genes")
        print(f"   📦 Using chunk size: {config.chunk_size:,} positions")
    
    # Process all genes
    meta_predictions = processor.process_multiple_genes(
        gene_ids=genes.tolist(),
        base_predictions=complete_base_pd,
        analysis_sequences=analysis_sequences_pd,
        model=model,
        feature_generator=feature_generator,
        training_schema=training_schema,
        calibrator=calibrator,
        parallel=False  # Can be made configurable
    )
    
    if verbose:
        print(f"   ✅ Generated meta-model predictions for {len(meta_predictions):,} positions")
    
    return meta_predictions


def generate_selective_meta_predictions(
    config: SelectiveInferenceConfig,
    complete_base_pd: pd.DataFrame,
    workflow_results: Dict,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Generate meta-model predictions for uncertain positions using selective feature generation.
    
    Parameters
    ----------
    config : SelectiveInferenceConfig
        Configuration for selective inference
    complete_base_pd : pd.DataFrame
        Complete base model predictions
    workflow_results : Dict
        Results from the inference workflow
    verbose : bool, default=True
        Enable verbose output
        
    Returns
    -------
    pd.DataFrame
        Meta-model predictions for uncertain positions
    """
    if verbose:
        print("   🧬 Generating selective features for uncertain positions")

    # 1) Identify uncertain positions
    max_scores = np.maximum(complete_base_pd['donor_score'].values,
                            complete_base_pd['acceptor_score'].values)
    if config.inference_mode == "base_only":
        uncertain_mask = np.zeros(len(complete_base_pd), dtype=bool)
    elif config.inference_mode == "meta_only":
        # Check if we should use chunked processing for full coverage
        total_positions = len(complete_base_pd)
        if total_positions > config.max_positions_per_gene and getattr(config, 'enable_chunked_processing', False):
            # Use chunked processor for true full coverage
            if verbose:
                print(f"   🔄 Using chunked processing for {total_positions:,} positions")
            return generate_chunked_meta_predictions(config, complete_base_pd, workflow_results, verbose)
        else:
            # All positions are uncertain in meta_only mode
            uncertain_mask = np.ones(len(complete_base_pd), dtype=bool)
    else:  # hybrid mode
        # Identify uncertain positions based on thresholds
        uncertain_mask = (
            (max_scores >= config.uncertainty_threshold_low) & 
            (max_scores < config.uncertainty_threshold_high)
        )
    
    uncertain_positions = complete_base_pd[uncertain_mask].copy()
    
    # Verify selective featurization
    if verbose:
        verify_selective_featurization(complete_base_pd, uncertain_positions, config, verbose=True)
    
    if len(uncertain_positions) == 0:
        if verbose:
            print("   ✅ No uncertain positions identified for feature generation")
        return pd.DataFrame(columns=['gene_id', 'position', 'donor_meta', 'acceptor_meta', 'neither_meta'])
    
    if verbose:
        print(f"   🎯 Identified {len(uncertain_positions)} uncertain positions for meta-model inference")
        print(f"   📊 Uncertainty range: [{config.uncertainty_threshold_low:.3f}, {config.uncertainty_threshold_high:.3f})")
    
    # Limit positions to prevent memory issues (unless chunked processing is enabled)
    if len(uncertain_positions) > config.max_positions_per_gene and not getattr(config, 'enable_chunked_processing', False):
        if verbose:
            print(f"   ⚠️  Limiting to {config.max_positions_per_gene} positions (memory constraint)")
        uncertain_positions = uncertain_positions.head(config.max_positions_per_gene)
    
    # 2) Obtain analysis rows for uncertain positions
    if 'analysis_sequences' not in workflow_results:
        if config.inference_mode == "meta_only":
            # For meta_only mode, we must have analysis sequences
            raise RuntimeError("Meta-only mode requires analysis_sequences but none were found. "
                             "This indicates a problem with the workflow configuration.")
        else:
            if verbose:
                print("   ⚠️  No analysis sequences available; skipping meta-model recalibration")
            return pd.DataFrame(columns=['gene_id', 'position', 'donor_meta', 'acceptor_meta', 'neither_meta'])

    analysis_sequences = workflow_results['analysis_sequences']
    analysis_sequences_pd = (analysis_sequences.to_pandas()
                             if hasattr(analysis_sequences, 'to_pandas') else analysis_sequences)
    
    # Merge to get only uncertain positions
    uncertain_analysis = analysis_sequences_pd.merge(
        uncertain_positions[['gene_id', 'position']],
        on=['gene_id', 'position'],
        how='inner'
    )
    
    if len(uncertain_analysis) == 0:
        if config.inference_mode == "meta_only":
            # For meta_only mode, this is a critical error
            raise RuntimeError(f"Meta-only mode requires analysis sequences for all positions but found none. "
                             f"Expected {len(uncertain_positions)} positions but got 0 analysis sequences. "
                             f"This indicates the complete coverage workflow failed to generate proper analysis_sequences.")
        else:
            if verbose:
                print("   ⚠️  No analysis sequences found for uncertain positions")
            return pd.DataFrame(columns=['gene_id', 'position', 'donor_meta', 'acceptor_meta', 'neither_meta'])
    
    # 3) Generate features using StandardizedFeaturizer
    # Use the helper function to create featurizer with proper configuration
    # This will load training schema and excluded features from the model directory
    if config.model_path:
        # Get model directory (handle both file and directory paths)
        model_dir = config.model_path if config.model_path.is_dir() else config.model_path.parent
        featurizer = create_featurizer_from_model_path(
            model_dir,
            verbose=verbose
        )
    else:
        # Fallback: Load training schema if available
        training_schema = None
        if config.training_schema_path:
            schema_path = config.training_schema_path
            if schema_path.name == "master" and schema_path.parent.exists():
                schema_path = schema_path.parent
            training_schema = _load_training_schema(schema_path)
        
        # Create basic featurizer
        featurizer = StandardizedFeaturizer(
            kmer_sizes=[3],
            training_schema=training_schema,
            verbose=verbose
        )
    
    # Generate features
    features_df = featurizer.featurize_from_analysis_df(
        analysis_df=uncertain_analysis,
        base_predictions_df=uncertain_positions,
        include_probability_features=True,
        include_context_features=True,
        include_kmer_features=True,
        include_genomic_features=True,
        harmonize_with_training=True
    )
    
    if verbose:
        print(f"   📊 Feature matrix: {len(features_df)} positions × {len(features_df.columns) - 2} features")
        verify_no_label_leakage(features_df, verbose=verbose)
    
    # 4) Prepare feature matrix for prediction
    # The StandardizedFeaturizer has already harmonized features and excluded what's needed
    # We just need to remove the metadata columns that are always kept (gene_id, position)
    metadata_cols = {'gene_id', 'position'}
    
    feature_cols = [col for col in features_df.columns if col not in metadata_cols]
    X = features_df[feature_cols].values.astype(np.float32)
    
    if verbose:
        print(f"   📊 Feature matrix shape: {X.shape}")
    
    # 5) Generate predictions
    model = load_model_with_calibration(config.model_path, use_calibration=config.use_calibration)
    meta_probs = model.predict_proba(X)
    
    if meta_probs.shape[1] != 3:
        if verbose:
            print(f"   ⚠️  Unexpected model output shape: {meta_probs.shape}")
        return pd.DataFrame(columns=['gene_id', 'position', 'donor_meta', 'acceptor_meta', 'neither_meta'])
    
    # Create meta predictions with additional informative columns
    meta_predictions_df = pd.DataFrame({
        'gene_id': features_df['gene_id'].values,
        'position': features_df['position'].values,
        # Meta model predictions
        'donor_meta': meta_probs[:, 1],
        'acceptor_meta': meta_probs[:, 2],
        'neither_meta': meta_probs[:, 0],
    })
    
    # Add score_meta (max of the three meta scores)
    meta_predictions_df['score_meta'] = meta_predictions_df[['donor_meta', 'acceptor_meta', 'neither_meta']].max(axis=1)
    
    # Add original base model scores if available
    if 'donor_score' in uncertain_positions.columns:
        # Merge back the base scores
        meta_predictions_df = meta_predictions_df.merge(
            uncertain_positions[['gene_id', 'position', 'donor_score', 'acceptor_score', 'neither_score']],
            on=['gene_id', 'position'],
            how='left'
        )
    
    # Add label-related columns if available
    if 'splice_type' in uncertain_analysis.columns:
        meta_predictions_df = meta_predictions_df.merge(
            uncertain_analysis[['gene_id', 'position', 'splice_type']],
            on=['gene_id', 'position'],
            how='left'
        )
    if 'pred_type' in uncertain_analysis.columns:
        meta_predictions_df = meta_predictions_df.merge(
            uncertain_analysis[['gene_id', 'position', 'pred_type']],
            on=['gene_id', 'position'],
            how='left'
        )
    
    if verbose:
        print(f"   ✅ Generated meta-model predictions for {len(meta_predictions_df)} positions")
    
    return meta_predictions_df
