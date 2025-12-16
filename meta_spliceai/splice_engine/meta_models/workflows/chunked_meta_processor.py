#!/usr/bin/env python3
"""
Chunked Meta-Model Processor for Full Coverage

This module provides memory-efficient processing of all positions through the meta-model
by splitting large gene sequences into manageable chunks.
"""

import numpy as np
import pandas as pd
import polars as pl
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
from tqdm import tqdm
import gc

# Import feature consistency checker for robustness
try:
    from .inference.feature_consistency_checker import check_early_chunk_consistency
except ImportError:
    check_early_chunk_consistency = None

logger = logging.getLogger(__name__)


@dataclass
class ChunkedProcessingConfig:
    """Configuration for chunked meta-model processing."""
    
    # Chunk size for processing
    chunk_size: int = 10000  # Process 10k positions at a time
    
    # Memory management
    max_memory_gb: float = 8.0  # Maximum memory usage in GB
    gc_frequency: int = 5  # Run garbage collection every N chunks
    
    # Processing options
    overlap_size: int = 0  # Overlap between chunks (if needed for context)
    use_progress_bar: bool = True
    inference_mode: str = "hybrid"  # One of: base_only, hybrid, meta_only
    verbose: int = 1


class ChunkedMetaProcessor:
    """
    Process large gene sequences through meta-model in memory-efficient chunks.
    
    This class enables true "meta_only" mode by processing ALL positions
    without memory constraints.
    """
    
    def __init__(self, config: ChunkedProcessingConfig):
        """
        Initialize the chunked processor.
        
        Parameters
        ----------
        config : ChunkedProcessingConfig
            Configuration for chunked processing
        """
        self.config = config
        self.logger = logger
        
    def process_gene_in_chunks(
        self,
        gene_id: str,
        base_predictions: pd.DataFrame,
        analysis_sequences: pd.DataFrame,
        model,
        feature_generator,
        training_schema: Optional[Dict] = None,
        calibrator = None
    ) -> pd.DataFrame:
        """
        Process a single gene's positions in chunks through the meta-model.
        
        Parameters
        ----------
        gene_id : str
            Gene identifier
        base_predictions : pd.DataFrame
            Base model predictions for all positions
        analysis_sequences : pd.DataFrame
            Analysis sequences with features for all positions
        model : object
            Trained meta-model
        feature_generator : object
            Feature generator for creating meta-model features
        training_schema : Dict, optional
            Training schema for feature consistency
        calibrator : object, optional
            Calibration model for probability adjustment
            
        Returns
        -------
        pd.DataFrame
            Meta-model predictions for all positions
        """
        # Filter to this gene
        gene_base = base_predictions[base_predictions['gene_id'] == gene_id].copy()
        gene_analysis = analysis_sequences[analysis_sequences['gene_id'] == gene_id].copy()
        
        if len(gene_base) == 0:
            self.logger.warning(f"No base predictions found for gene {gene_id}")
            return pd.DataFrame()
            
        total_positions = len(gene_base)
        n_chunks = (total_positions + self.config.chunk_size - 1) // self.config.chunk_size
        
        if self.config.verbose >= 1:
            print(f"\n📊 Processing {gene_id}: {total_positions:,} positions in {n_chunks} chunks")
            
        # Sort by position to ensure sequential processing
        gene_base = gene_base.sort_values('position')
        gene_analysis = gene_analysis.sort_values('position')
        
        # Process each chunk
        chunk_results = []
        progress_bar = None
        
        if self.config.use_progress_bar and n_chunks > 1:
            progress_bar = tqdm(total=n_chunks, desc=f"Processing {gene_id}", unit="chunk")
        
        for chunk_idx in range(n_chunks):
            start_idx = chunk_idx * self.config.chunk_size
            end_idx = min(start_idx + self.config.chunk_size, total_positions)
            
            if self.config.verbose >= 2:
                print(f"  Chunk {chunk_idx + 1}/{n_chunks}: positions {start_idx:,}-{end_idx:,}")
            
            # Get chunk of data
            chunk_base = gene_base.iloc[start_idx:end_idx]
            chunk_positions = set(chunk_base['position'].values)
            chunk_analysis = gene_analysis[gene_analysis['position'].isin(chunk_positions)]
            
            if len(chunk_analysis) == 0:
                self.logger.warning(f"  No analysis sequences for chunk {chunk_idx + 1}")
                # Still include base predictions for these positions
                chunk_meta = chunk_base[['gene_id', 'position']].copy()
                chunk_meta['donor_meta'] = chunk_base['donor_score']
                chunk_meta['acceptor_meta'] = chunk_base['acceptor_score']
                chunk_meta['neither_meta'] = chunk_base['neither_score']
                chunk_meta['prediction_source'] = 'base_model_fallback'
            else:
                # Process chunk through meta-model
                chunk_meta = self._process_single_chunk(
                    chunk_base,
                    chunk_analysis,
                    model,
                    feature_generator,
                    training_schema,
                    calibrator,
                    chunk_idx=chunk_idx  # Pass chunk index for consistency checking
                )
            
            chunk_results.append(chunk_meta)
            
            # Memory management
            if (chunk_idx + 1) % self.config.gc_frequency == 0:
                gc.collect()
                if self.config.verbose >= 2:
                    print(f"  Memory cleanup after chunk {chunk_idx + 1}")
            
            if progress_bar:
                progress_bar.update(1)
        
        if progress_bar:
            progress_bar.close()
        
        # Combine all chunks
        if chunk_results:
            combined_results = pd.concat(chunk_results, ignore_index=True)
            
            if self.config.verbose >= 1:
                print(f"✅ Processed {len(combined_results):,} positions for {gene_id}")
                
            return combined_results
        else:
            return pd.DataFrame()
    
    def _process_single_chunk(
        self,
        chunk_base: pd.DataFrame,
        chunk_analysis: pd.DataFrame,
        model,
        feature_generator,
        training_schema: Optional[Dict],
        calibrator,
        chunk_idx: int = 0
    ) -> pd.DataFrame:
        """
        Process a single chunk through the meta-model.
        
        Parameters
        ----------
        chunk_base : pd.DataFrame
            Base predictions for this chunk
        chunk_analysis : pd.DataFrame
            Analysis sequences for this chunk
        model : object
            Trained meta-model
        feature_generator : object
            Feature generator
        training_schema : Dict, optional
            Training schema
        calibrator : object, optional
            Calibration model
            
        Returns
        -------
        pd.DataFrame
            Meta-model predictions for this chunk
        """
        try:
            # Generate features for this chunk
            if hasattr(feature_generator, 'generate_features'):
                features_df = feature_generator.generate_features(
                    chunk_analysis,
                    chunk_base,
                    training_schema=training_schema
                )
            else:
                # Fallback to direct feature generation
                from meta_spliceai.splice_engine.meta_models.workflows.selective_feature_generator import (
                    _generate_features_for_missing_positions
                )
                features_df = _generate_features_for_missing_positions(
                    chunk_analysis,
                    training_schema_path=None,  # Not needed for positions with analysis sequences
                    verbose=False
                )
            
            if features_df is None or len(features_df) == 0:
                # In meta_only mode, feature generation failure is a critical error
                if self.config.inference_mode == "meta_only":
                    raise RuntimeError(
                        "Feature generation failed in meta_only mode. "
                        "Cannot fall back to base model."
                    )
                
                # For other modes, fall back to base predictions
                result = chunk_base[['gene_id', 'position']].copy()
                result['donor_meta'] = chunk_base['donor_score']
                result['acceptor_meta'] = chunk_base['acceptor_score']
                result['neither_meta'] = chunk_base['neither_score']
                result['prediction_source'] = 'feature_generation_failed'
                
                # Add score_meta
                result['score_meta'] = result[['donor_meta', 'acceptor_meta', 'neither_meta']].max(axis=1)
                
                # Add original base model scores
                result['donor_score'] = chunk_base['donor_score'].values
                result['acceptor_score'] = chunk_base['acceptor_score'].values
                result['neither_score'] = chunk_base['neither_score'].values
                
                # Add label-related columns if available
                if 'splice_type' in chunk_base.columns:
                    result['splice_type'] = chunk_base['splice_type'].values
                if 'pred_type' in chunk_base.columns:
                    result['pred_type'] = chunk_base['pred_type'].values
                    
                return result
            
            # Feature consistency check for early chunks (hybrid and meta_only modes)
            if check_early_chunk_consistency and self.config.inference_mode in ["hybrid", "meta_only"]:
                # Get model path from feature generator if available
                model_path = getattr(feature_generator, 'model_path', None)
                if model_path and chunk_idx < 2:  # Check first 2 chunks only
                    try:
                        check_early_chunk_consistency(
                            features_df, 
                            model_path, 
                            chunk_idx=chunk_idx,
                            max_checks=2,
                            verbose=(self.config.verbose >= 2),
                            strict_mode=False  # Allow missing features for unseen genes
                        )
                    except ValueError as e:
                        # Critical feature inconsistency detected
                        self.logger.error(f"Feature consistency check failed: {e}")
                        if self.config.inference_mode == "meta_only":
                            raise  # Re-raise in meta_only mode
                        # In hybrid mode, log but continue (will use base predictions)
            
            # Prepare feature matrix for model prediction
            # The StandardizedFeaturizer (via FeatureGeneratorWrapper) has already:
            # 1. Harmonized features with training schema
            # 2. Excluded features from excluded_features.txt
            # 3. Handled chrom encoding
            # We just need to remove the metadata columns that are always kept (gene_id, position)
            metadata_cols = {'gene_id', 'position'}
            
            feature_cols = [col for col in features_df.columns if col not in metadata_cols]
            X = features_df[feature_cols].values.astype(np.float32)
            
            # Make predictions
            if hasattr(model, 'predict_proba'):
                # Multiclass model
                meta_probs = model.predict_proba(X)
                
                if meta_probs.shape[1] == 3:
                    # [neither, donor, acceptor]
                    donor_meta = meta_probs[:, 1]
                    acceptor_meta = meta_probs[:, 2]
                    neither_meta = meta_probs[:, 0]
                else:
                    # Unexpected shape
                    self.logger.warning(f"Unexpected prediction shape: {meta_probs.shape}")
                    donor_meta = chunk_base['donor_score'].values
                    acceptor_meta = chunk_base['acceptor_score'].values
                    neither_meta = chunk_base['neither_score'].values
            else:
                # Binary models or other format
                donor_meta = chunk_base['donor_score'].values
                acceptor_meta = chunk_base['acceptor_score'].values
                neither_meta = chunk_base['neither_score'].values
            
            # Apply calibration if available
            if calibrator is not None:
                try:
                    donor_meta = calibrator.transform(donor_meta.reshape(-1, 1)).ravel()
                    acceptor_meta = calibrator.transform(acceptor_meta.reshape(-1, 1)).ravel()
                except Exception as e:
                    self.logger.warning(f"Calibration failed: {e}")
            
            # Create result DataFrame with additional informative columns
            result = pd.DataFrame({
                'gene_id': chunk_base['gene_id'].values,
                'position': chunk_base['position'].values,
                'donor_meta': donor_meta,
                'acceptor_meta': acceptor_meta,
                'neither_meta': neither_meta,
                'prediction_source': 'meta_model'
            })
            
            # Add score_meta (max of the three meta scores)
            result['score_meta'] = result[['donor_meta', 'acceptor_meta', 'neither_meta']].max(axis=1)
            
            # Add original base model scores
            result['donor_score'] = chunk_base['donor_score'].values
            result['acceptor_score'] = chunk_base['acceptor_score'].values
            result['neither_score'] = chunk_base['neither_score'].values
            
            # Add label-related columns if available
            if 'splice_type' in chunk_base.columns:
                result['splice_type'] = chunk_base['splice_type'].values
            if 'pred_type' in chunk_base.columns:
                result['pred_type'] = chunk_base['pred_type'].values
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing chunk: {e}")
            self.logger.error(f"Error type: {type(e).__name__}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            
            # In meta_only mode, we should NOT fall back to base predictions
            # This would hide the real problem
            if self.config.inference_mode == "meta_only":
                raise RuntimeError(
                    f"Meta-model prediction failed in meta_only mode. "
                    f"Cannot fall back to base model. Error: {e}"
                ) from e
            
            # For other modes (hybrid, base_only), fall back to base predictions
            result = chunk_base[['gene_id', 'position']].copy()
            result['donor_meta'] = chunk_base['donor_score']
            result['acceptor_meta'] = chunk_base['acceptor_score']
            result['neither_meta'] = chunk_base['neither_score']
            result['prediction_source'] = 'processing_error_fallback'
            
            # Add score_meta
            result['score_meta'] = result[['donor_meta', 'acceptor_meta', 'neither_meta']].max(axis=1)
            
            # Add original base model scores
            result['donor_score'] = chunk_base['donor_score'].values
            result['acceptor_score'] = chunk_base['acceptor_score'].values
            result['neither_score'] = chunk_base['neither_score'].values
            
            # Add label-related columns if available
            if 'splice_type' in chunk_base.columns:
                result['splice_type'] = chunk_base['splice_type'].values
            if 'pred_type' in chunk_base.columns:
                result['pred_type'] = chunk_base['pred_type'].values
                
            return result
    
    def process_multiple_genes(
        self,
        gene_ids: List[str],
        base_predictions: pd.DataFrame,
        analysis_sequences: pd.DataFrame,
        model,
        feature_generator,
        training_schema: Optional[Dict] = None,
        calibrator = None,
        parallel: bool = False
    ) -> pd.DataFrame:
        """
        Process multiple genes through the meta-model in chunks.
        
        Parameters
        ----------
        gene_ids : List[str]
            List of gene identifiers
        base_predictions : pd.DataFrame
            Base model predictions for all genes
        analysis_sequences : pd.DataFrame
            Analysis sequences for all genes
        model : object
            Trained meta-model
        feature_generator : object
            Feature generator
        training_schema : Dict, optional
            Training schema
        calibrator : object, optional
            Calibration model
        parallel : bool, default=False
            Whether to process genes in parallel
            
        Returns
        -------
        pd.DataFrame
            Combined meta-model predictions for all genes
        """
        all_results = []
        
        for gene_id in gene_ids:
            gene_results = self.process_gene_in_chunks(
                gene_id=gene_id,
                base_predictions=base_predictions,
                analysis_sequences=analysis_sequences,
                model=model,
                feature_generator=feature_generator,
                training_schema=training_schema,
                calibrator=calibrator
            )
            
            if len(gene_results) > 0:
                all_results.append(gene_results)
            
            # Memory cleanup between genes
            gc.collect()
        
        if all_results:
            combined = pd.concat(all_results, ignore_index=True)
            
            if self.config.verbose >= 1:
                print(f"\n🎯 Total processed: {len(combined):,} positions across {len(gene_ids)} genes")
                
            return combined
        else:
            return pd.DataFrame()


def create_chunked_processor(
    chunk_size: int = 10000,
    inference_mode: str = "hybrid",
    verbose: int = 1
) -> ChunkedMetaProcessor:
    """
    Create a chunked processor with default settings.
    
    Parameters
    ----------
    chunk_size : int, default=10000
        Size of each processing chunk
    inference_mode : str, default="hybrid"
        One of: base_only, hybrid, meta_only
    verbose : int, default=1
        Verbosity level
        
    Returns
    -------
    ChunkedMetaProcessor
        Configured processor instance
    """
    config = ChunkedProcessingConfig(
        chunk_size=chunk_size,
        inference_mode=inference_mode,
        verbose=verbose
    )
    return ChunkedMetaProcessor(config)
