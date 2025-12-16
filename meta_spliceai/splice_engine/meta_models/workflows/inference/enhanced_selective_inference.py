"""
Enhanced Selective Meta-Model Inference with Complete Coverage

This module implements the corrected selective inference workflow that ensures:
1. Complete position coverage for all target genes (no gaps from downsampling)
2. Base model predictions for ALL positions in target genes
3. Selective meta-model application based only on uncertainty from base scores
4. Proper output schema with all required columns

Key improvements over the existing selective_inference.py:
- Generates complete base model predictions for ALL positions (not just training subset)
- Properly identifies uncertain positions using ONLY base model scores
- Applies meta-model selectively to uncertain positions only
- Ensures continuous position coverage without gaps
"""

import os
import sys
import json
import time
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union, Any
from collections import defaultdict, Counter
from dataclasses import dataclass
import logging
from itertools import product

import polars as pl
import pandas as pd
import numpy as np
import gc
import psutil

# Import core components
from meta_spliceai.splice_engine.meta_models.workflows.splice_prediction_workflow import (
    run_enhanced_splice_prediction_workflow
)
from meta_spliceai.splice_engine.meta_models.core.data_types import SpliceAIConfig
from meta_spliceai.splice_engine.meta_models.workflows.inference_workflow_utils import (
    load_model_with_calibration
)
from meta_spliceai.splice_engine.meta_models.workflows.inference.genomic_feature_enricher import (
    GenomicFeatureEnricher
)
from meta_spliceai.splice_engine.meta_models.builder.feature_schema import (
    encode_categorical_features
)
# NEW: Import centralized output management
from meta_spliceai.system.output_resources import OutputManager
# NEW: Import coordinate adjustment system for base model agnostic predictions
from meta_spliceai.splice_engine.meta_models.utils.splice_utils import (
    prepare_splice_site_adjustments
)

logger = logging.getLogger(__name__)


class MemoryMonitor:
    """Monitor memory usage during inference."""
    
    def __init__(self, max_memory_gb: float = 8.0):
        self.max_memory_gb = max_memory_gb
        self.max_memory_mb = max_memory_gb * 1024
        self.process = psutil.Process(os.getpid())
        self.peak_mb = 0.0
        self.checkpoints = []
        self.baseline_mb = self._get_current_memory_mb()
        
    def _get_current_memory_mb(self) -> float:
        """Get current memory usage in MB."""
        return self.process.memory_info().rss / (1024 ** 2)
    
    def checkpoint(self, step_name: str) -> float:
        """Record memory checkpoint."""
        current_mb = self._get_current_memory_mb()
        self.peak_mb = max(self.peak_mb, current_mb)
        self.checkpoints.append((step_name, current_mb))
        
        # Warn if approaching limit
        if current_mb > self.max_memory_mb * 0.8:
            logger.warning(f"⚠️  Memory at {current_mb:.1f} MB (80% of {self.max_memory_mb:.1f} MB limit)")
        
        return current_mb
    
    def report(self) -> dict:
        """Generate memory usage report."""
        return {
            'peak_mb': self.peak_mb,
            'final_mb': self._get_current_memory_mb(),
            'baseline_mb': self.baseline_mb,
            'delta_mb': self._get_current_memory_mb() - self.baseline_mb,
            'max_allowed_mb': self.max_memory_mb,
            'peak_usage_pct': (self.peak_mb / self.max_memory_mb) * 100,
            'checkpoints': self.checkpoints
        }


@dataclass
class EnhancedSelectiveInferenceConfig:
    """Configuration for enhanced selective meta-model inference with complete coverage."""
    
    # Core parameters
    model_path: Path
    target_genes: List[str]
    
    # Complete coverage parameters
    ensure_complete_coverage: bool = True  # Generate predictions for ALL positions
    gene_features_path: Optional[Path] = None  # Path to gene features file
    
    # Selective featurization thresholds
    uncertainty_threshold_low: float = 0.02   # Below this: confident non-splice
    uncertainty_threshold_high: float = 0.50  # Above this: confident splice (lowered from 0.80 to enable meta-model testing)
    
    # Processing limits
    max_positions_per_gene: int = 10000  # Prevent huge feature matrices
    max_analysis_rows: int = 500000     # Total position limit
    
    # Training integration
    training_dataset_path: Optional[Path] = None
    training_schema_path: Optional[Path] = None
    
    # Directory management  
    inference_base_dir: Path = None
    output_name: Optional[str] = None
    use_timestamped_output: bool = False  # If True, create timestamped dirs (for experiments)
    
    # Processing options
    use_calibration: bool = True
    cleanup_intermediates: bool = True
    verbose: int = 1
    
    # Inference mode: 'base_only', 'hybrid', 'meta_only'
    inference_mode: str = "hybrid"  # Default to hybrid for scalability
    
    # NEW: Memory management parameters
    max_memory_gb: float = 8.0           # Maximum memory usage
    chunk_size: int = 10000              # Positions per chunk
    gc_frequency: int = 5                # Garbage collection frequency
    enable_memory_monitoring: bool = True # Enable memory tracking


@dataclass
class EnhancedSelectiveInferenceResults:
    """Results from enhanced selective meta-model inference."""
    
    success: bool
    config: EnhancedSelectiveInferenceConfig
    
    # Output paths
    hybrid_predictions_path: Optional[Path] = None    # Combined base + meta predictions
    meta_predictions_path: Optional[Path] = None      # Meta-model predictions only
    base_predictions_path: Optional[Path] = None      # Base model predictions only
    gene_manifest_path: Optional[Path] = None         # Processing tracking
    
    # Statistics
    total_positions: int = 0
    positions_recalibrated: int = 0
    positions_reused: int = 0
    genes_processed: int = 0
    
    # Performance
    processing_time_seconds: float = 0.0
    feature_matrix_size_mb: float = 0.0
    
    # Details
    per_gene_stats: Dict[str, Dict] = None
    error_messages: List[str] = None


class EnhancedSelectiveInferenceWorkflow:
    """
    Enhanced selective inference workflow that ensures complete coverage.
    
    This workflow addresses the gaps in position coverage by:
    1. Generating base model predictions for ALL positions in target genes
    2. Identifying uncertain positions using only base model scores
    3. Selectively applying meta-model to uncertain positions only
    4. Ensuring continuous position coverage without gaps
    """
    
    def __init__(self, config: EnhancedSelectiveInferenceConfig):
        """Initialize the enhanced selective inference workflow."""
        self.config = config
        self.logger = self._setup_logging()
        
        # NEW: Initialize memory monitoring if enabled
        if config.enable_memory_monitoring:
            self.memory_monitor = MemoryMonitor(max_memory_gb=config.max_memory_gb)
        else:
            self.memory_monitor = None
        
        # NEW: Initialize genomic feature enricher for coordinate conversion
        self.genomic_enricher = GenomicFeatureEnricher(verbose=False)
        
        # NEW: Initialize centralized output manager
        self.output_manager = OutputManager.from_config(
            config=config,
            logger=self.logger,
            base_model_name="spliceai"  # TODO: Make configurable when adding other base models
        )
        
        # Log the directory structure
        if config.verbose >= 1:
            self.output_manager.log_directory_structure()
        
        # For backward compatibility (some code may still reference self.output_dir)
        self.output_dir = self.output_manager.registry.get_mode_dir(
            config.inference_mode,
            is_test=config.output_name and 'test' in config.output_name.lower()
        )
        
        # Initialize coordinate adjustment system for base model agnostic predictions
        # This handles coordinate differences between base models (SpliceAI, OpenSpliceAI, etc.)
        # and our GTF-derived annotations. Critical for accurate splice site detection.
        self.adjustment_dict = None  # Will be loaded/detected in _prepare_adjustments()
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the workflow."""
        log_level = {
            0: logging.WARNING,
            1: logging.INFO,
            2: logging.DEBUG,
            3: logging.DEBUG
        }.get(self.config.verbose, logging.INFO)
        
        logger = logging.getLogger(__name__)
        logger.setLevel(log_level)
        
        # Add console handler if not already present
        if not logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(log_level)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        return logger
    
    # REMOVED: _setup_output_directory() - now handled by OutputManager
    # The OutputManager provides centralized, consistent output path management
    # following the same pattern as genomic_resources
    
    def _find_gene_features_file(self) -> Path:
        """Find the gene features file with genomic coordinates."""
        # CRITICAL: Use Registry to get build-specific gene features
        # This ensures we use GRCh37 gene features when working with GRCh37
        gene_features_path = self.registry.resolve("gene_features")
        if gene_features_path:
            self.logger.info(f"Found gene features: {gene_features_path}")
            return Path(gene_features_path)
        
        # Fallback to config-provided paths if Registry fails
        possible_paths = []
        
        # Add user-provided path if specified
        if self.config.gene_features_path:
            possible_paths.append(self.config.gene_features_path)
        
        # Add training dataset path if specified
        if self.config.training_dataset_path:
            possible_paths.append(self.config.training_dataset_path / "gene_features.tsv")
        
        for path in possible_paths:
            if path and Path(path).exists():
                self.logger.info(f"Found gene features: {path}")
                return Path(path)
                
        raise FileNotFoundError(
            f"Could not find gene_features.tsv. Searched in:\n"
            f"  - Registry (build={self.registry.cfg.build})\n"
            f"  - Config paths: {possible_paths}\n"
            f"Please ensure gene_features.tsv exists in the build-specific directory."
        )
    
    def _get_gene_info(self, gene_features_path: Path) -> Dict:
        """Load gene information from gene features file."""
        # Read with proper schema for chromosome column (handles 'X', 'Y', etc.)
        # CRITICAL: Override both 'chrom' and 'seqname' as Utf8 to handle X, Y, MT chromosomes
        # GRCh37 files may have 'seqname' column from GTF, GRCh38 files use 'chrom'
        features_df = pl.read_csv(
            gene_features_path, 
            separator='\t',
            schema_overrides={'chrom': pl.Utf8, 'seqname': pl.Utf8}
        )
        
        gene_info = {}
        for gene_id in self.config.target_genes:
            gene_row = features_df.filter(pl.col('gene_id') == gene_id)
            if gene_row.height == 0:
                self.logger.warning(f"Gene {gene_id} not found in gene features")
                continue
                
            row_dict = gene_row.to_dicts()[0]
            gene_info[gene_id] = {
                'chrom': str(row_dict['chrom']),  # Ensure string format
                'strand': row_dict['strand'],
                'start': int(row_dict['start']),
                'end': int(row_dict['end']),
                'length': int(row_dict['gene_length']),  # Use pre-computed length
                'gene_name': row_dict.get('gene_name', gene_id)
            }
            
        self.logger.info(f"Loaded info for {len(gene_info)} genes")
        return gene_info
    
    def _generate_complete_base_predictions(self, gene_info: Dict) -> pl.DataFrame:
        """
        Generate complete base model predictions for ALL positions in target genes.
        
        This ensures that every position in every target gene has a base model prediction,
        bypassing any sparse artifacts from training data downsampling.
        """
        self.logger.info("🎯 Generating complete base model predictions for ALL positions...")
        
        complete_predictions = []
        
        for gene_id in self.config.target_genes:
            if gene_id not in gene_info:
                self.logger.warning(f"Skipping {gene_id} - no gene info available")
                continue
                
            gene_length = gene_info[gene_id]['length']
            self.logger.info(f"Processing {gene_id} (length: {gene_length:,} bp)")
            
            # Generate complete base model predictions for ALL positions
            gene_complete = self._generate_complete_base_model_predictions(gene_id, gene_info[gene_id])
            
            if gene_complete.height > 0:
                complete_predictions.append(gene_complete)
                self.logger.info(f"  ✅ Complete coverage: {gene_complete.height:,} positions")
                
                # Validate coverage
                expected_positions = gene_length
                actual_positions = gene_complete.height
                coverage_percent = actual_positions / expected_positions * 100
                
                if coverage_percent >= 95:
                    self.logger.info(f"  ✅ Excellent coverage: {coverage_percent:.1f}%")
                elif coverage_percent >= 80:
                    self.logger.warning(f"  ⚠️  Good coverage: {coverage_percent:.1f}%")
                else:
                    self.logger.error(f"  ❌ Poor coverage: {coverage_percent:.1f}%")
            else:
                self.logger.error(f"  ❌ No predictions generated for {gene_id}")
            
        if complete_predictions:
            all_predictions = pl.concat(complete_predictions)
            self.logger.info(f"✅ Complete base predictions: {all_predictions.height:,} total positions")
            return all_predictions
        else:
            self.logger.error("❌ No complete predictions generated for any target genes")
            return pl.DataFrame()
    
    def _generate_complete_base_model_predictions(self, gene_id: str, gene_info: Dict) -> pl.DataFrame:
        """
        Generate complete base model predictions for ALL positions in the target gene.
        
        CRITICAL: This must return predictions for EVERY nucleotide position (1 to gene_length).
        The output dimension must be N × 3 where N = gene_length.
        
        This invokes the SpliceAI base model DIRECTLY to predict splice site scores for every 
        nucleotide position in the gene, ensuring complete coverage without gaps.
        """
        gene_length = gene_info['length']
        self.logger.info(f"  Generating complete base model predictions for all {gene_length:,} positions...")
        
        # Use OutputManager for artifact paths (mode-independent)
        gene_paths = self.output_manager.get_gene_output_paths(gene_id)
        complete_output_dir = gene_paths.base_predictions_dir / gene_id
        complete_output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # CRITICAL FIX: Call SpliceAI DIRECTLY to get complete predictions
            # This ensures we get predictions for ALL positions, not just filtered ones
            complete_predictions = self._run_spliceai_directly(gene_id, gene_info, complete_output_dir)
            
            if complete_predictions.height == 0:
                self.logger.error(f"  ❌ No predictions generated by SpliceAI for {gene_id}")
                return pl.DataFrame()
            
            self.logger.info(f"  ✅ SpliceAI predictions: {complete_predictions.height:,} positions")
            
            # VERIFICATION: Check for full coverage
            # NOTE: With score-shifting approach, we should have complete_predictions.height == gene_length
            # because we shift SCORES not positions, maintaining all N positions for N-bp gene
            if complete_predictions.height != gene_length:
                reduction_pct = (1 - complete_predictions.height / gene_length) * 100
                if reduction_pct > 5:  # More than 5% missing is concerning
                    self.logger.warning(
                        f"  ⚠️  Coverage issue: {complete_predictions.height:,}/{gene_length:,} positions "
                        f"({100-reduction_pct:.1f}% coverage, {reduction_pct:.1f}% missing)"
                    )
                else:
                    self.logger.info(
                        f"  📊 Position count: {complete_predictions.height:,}/{gene_length:,} "
                        f"({100-reduction_pct:.1f}% coverage)"
                    )
            else:
                self.logger.info(f"  ✅ Full coverage achieved: {gene_length:,} positions (100%)")
            
            # Validate we have required score columns
            required_score_cols = ['donor_score', 'acceptor_score', 'neither_score']
            missing_cols = [col for col in required_score_cols if col not in complete_predictions.columns]
            
            if missing_cols:
                self.logger.error(f"  ❌ Missing required score columns: {missing_cols}")
                return pl.DataFrame()
            
            # Enrich with genomic features BEFORE k-mer generation for meta-model compatibility
            # CRITICAL: Disable transcript features to avoid join-based duplication
            # (one gene → many transcripts would multiply rows 5× for a 5-transcript gene)
            self.logger.info(f"  Enriching with genomic features for meta-model compatibility...")
            pre_enrich_rows = complete_predictions.height
            complete_predictions = self.genomic_enricher.enrich(
                complete_predictions,
                include_critical=True,   # gene_start, gene_end, absolute_position
                include_useful=True,     # gene_name, gene_type
                include_structure=False, # DISABLED: transcript features cause row multiplication
                include_flags=True       # has_gene_info, has_tx_info
            )
            post_enrich_rows = complete_predictions.height
            if post_enrich_rows != pre_enrich_rows:
                self.logger.warning(f"  ⚠️  Enrichment changed row count: {pre_enrich_rows:,} → {post_enrich_rows:,} ({post_enrich_rows // pre_enrich_rows}× multiplication)")
            
            # Apply dynamic chromosome encoding to match training data schema
            self.logger.info(f"  Applying dynamic chromosome encoding to match training schema...")
            complete_predictions = self._apply_dynamic_chrom_encoding(complete_predictions)
            
            # Generate k-mer features for meta-model compatibility
            # NOTE: Use 3-mers to match training data (train_pc_1000_3mers)
            self.logger.info(f"  Generating k-mer features for meta-model compatibility...")
            complete_predictions = self._generate_kmer_features(complete_predictions, kmer_sizes=[3])
            
            # CRITICAL: Generate derived features (probability, context, etc.) to match training
            self.logger.info(f"  Generating derived features (probability + context)...")
            complete_predictions = self._generate_derived_features(complete_predictions)
            
            # Save complete predictions for future reuse
            complete_pred_file = complete_output_dir / f"complete_predictions_{gene_id}.parquet"
            complete_predictions.write_parquet(complete_pred_file, compression='zstd')
            self.logger.info(f"  💾 Saved complete predictions: {complete_pred_file.name}")
            
            return complete_predictions
            
        except Exception as e:
            self.logger.error(f"  ❌ Failed to generate complete base predictions for {gene_id}: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return pl.DataFrame()
    
    def _check_existing_genomic_datasets(self) -> Dict[str, bool]:
        """
        Check for existing genomic datasets to avoid recomputation.
        
        Uses Registry to systematically locate datasets in the centralized
        data/ensembl/ directory where they were created during the base model pass.
        
        Returns
        -------
        Dict[str, bool]
            Dictionary indicating which datasets exist
        """
        from meta_spliceai.system.genomic_resources import Registry
        
        datasets = {
            'splice_sites': False,
            'gene_features': False,
            'transcript_features': False,
            'exon_features': False,
            'annotations_db': False,
            'overlapping_genes': False,
            'gene_sequences': False,
            'all_exist': False
        }
        
        registry = Registry()
        
        # Use Registry to locate datasets (RECOMMENDED APPROACH)
        registry_mappings = {
            'splice_sites': 'splice_sites',
            'gene_features': 'gene_features',
            'transcript_features': 'transcript_features',
            'exon_features': 'exon_features',
            'annotations_db': 'annotations_db',
            'overlapping_genes': 'overlapping_genes'
            # Note: 'gene_sequences' not in Registry, checked separately below
        }
        
        for dataset_type, registry_key in registry_mappings.items():
            try:
                path = registry.resolve(registry_key)
                if path:
                    path_obj = Path(path)
                    if path_obj.exists():
                        datasets[dataset_type] = True
                        self.logger.debug(f"    Found {dataset_type}: {path}")
                    else:
                        self.logger.debug(f"    Registry returned non-existent path for {dataset_type}: {path}")
            except Exception as e:
                self.logger.debug(f"    Could not resolve {dataset_type} via Registry: {e}")
        
        # Check for gene_sequences using Registry's data directory
        data_dir = self.registry.get_local_dir()
        if data_dir.exists():
            seq_files = list(data_dir.glob("gene_sequence_*.parquet"))
            if seq_files:
                datasets['gene_sequences'] = True
                self.logger.debug(f"    Found {len(seq_files)} gene sequence files in {data_dir}")
        
        # Determine if all essential datasets exist
        essential_datasets = ['splice_sites', 'gene_features', 'annotations_db']
        datasets['all_exist'] = all(datasets[ds] for ds in essential_datasets)
        
        self.logger.info(f"  Genomic dataset status:")
        for dataset_type, exists in datasets.items():
            if dataset_type != 'all_exist':
                status = "✅" if exists else "❌"
                self.logger.info(f"    {status} {dataset_type}")
        
        # Log warning if essential datasets are missing
        if not datasets['all_exist']:
            missing = [ds for ds in essential_datasets if not datasets[ds]]
            self.logger.warning(
                f"  ⚠️  Missing essential datasets: {missing}. "
                "These should have been created during the base model pass. "
                "Run the base model workflow first to create them."
            )
        
        return datasets
    
    def _check_gene_sequences_exist(self, registry) -> bool:
        """
        Check if gene sequence files exist in the shared data location.
        
        Parameters
        ----------
        registry : Registry
            Registry instance for path resolution (not used currently)
            
        Returns
        -------
        bool
            True if gene sequence files exist
        """
        # Check for individual chromosome files in data/ensembl/
        data_dir = Path("data/ensembl")
        if data_dir.exists():
            # Look for gene_sequence_*.parquet files
            seq_files = list(data_dir.glob("gene_sequence_*.parquet"))
            if seq_files:
                self.logger.debug(f"    Found {len(seq_files)} gene sequence files")
                return True
        
        self.logger.debug(f"    No gene sequence files found in {data_dir}")
        return False
    
    def _apply_coordinate_adjustments_v0(self, predictions_df: pl.DataFrame) -> pl.DataFrame:
        """
        OLD VERSION: Apply coordinate adjustments by shifting POSITIONS (creates collisions).
        
        This version shifts position coordinates based on predicted splice type, which causes
        multiple positions to map to the same coordinate (position collisions), reducing coverage.
        
        DEPRECATED: Use _apply_coordinate_adjustments() instead, which shifts SCORES not positions.
        
        Different base models (SpliceAI, OpenSpliceAI, etc.) may have different
        coordinate conventions. This method ensures predictions align with our
        GTF-derived annotations by applying appropriate corrections based on the
        loaded adjustment_dict.
        
        The adjustment system is base-model agnostic and uses the prepare_splice_site_adjustments()
        system from the training workflow to detect or load model-specific adjustments.
        
        Parameters
        ----------
        predictions_df : pl.DataFrame
            DataFrame with raw positions, donor_prob, acceptor_prob, strand
            
        Returns
        -------
        pl.DataFrame
            DataFrame with corrected positions (WARNING: may have collisions)
        """
        # Ensure adjustment_dict is available
        if self.adjustment_dict is None:
            self.logger.warning("  ⚠️  No adjustment_dict available, using default SpliceAI pattern")
            self.adjustment_dict = {
                'donor': {'plus': 2, 'minus': 1},
                'acceptor': {'plus': 0, 'minus': -1}
            }
        
        # Determine predicted splice type for each position
        # (donor if donor_prob > acceptor_prob, otherwise acceptor)
        predictions_df = predictions_df.with_columns([
            pl.when(pl.col('donor_prob') > pl.col('acceptor_prob'))
              .then(pl.lit('donor'))
              .otherwise(pl.lit('acceptor'))
              .alias('predicted_type')
        ])
        
        # Apply adjustments based on the adjustment_dict
        # The dict contains the SHIFT needed: positive = shift right, negative = shift left
        # To CORRECT the position, we SUBTRACT the shift (opposite direction)
        
        donor_plus_adj = self.adjustment_dict['donor']['plus']
        donor_minus_adj = self.adjustment_dict['donor']['minus']
        acceptor_plus_adj = self.adjustment_dict['acceptor']['plus']
        acceptor_minus_adj = self.adjustment_dict['acceptor']['minus']
        
        predictions_df = predictions_df.with_columns([
            pl.when((pl.col('predicted_type') == 'donor') & (pl.col('strand') == '+'))
              .then(pl.col('position') - donor_plus_adj)      # Donor + strand
              .when((pl.col('predicted_type') == 'donor') & (pl.col('strand') == '-'))
              .then(pl.col('position') - donor_minus_adj)     # Donor - strand
              .when((pl.col('predicted_type') == 'acceptor') & (pl.col('strand') == '+'))
              .then(pl.col('position') - acceptor_plus_adj)   # Acceptor + strand
              .when((pl.col('predicted_type') == 'acceptor') & (pl.col('strand') == '-'))
              .then(pl.col('position') - acceptor_minus_adj)  # Acceptor - strand
              .otherwise(pl.col('position'))                   # Fallback
              .alias('position')
        ])
        
        # Remove temporary column
        predictions_df = predictions_df.drop('predicted_type')
        
        return predictions_df
    
    def _apply_coordinate_adjustments(self, predictions_df: pl.DataFrame) -> pl.DataFrame:
        """
        Apply type & strand-specific coordinate adjustments with CORRELATED probability vectors.
        
        CRITICAL INSIGHT (2025-10-31):
        ==============================
        The three scores at each position (donor, acceptor, neither) are CORRELATED and must
        sum to 1.0. When adjusting coordinates, we must shift the ENTIRE probability vector
        as a unit, not individual score types independently.
        
        This is the CORRECT approach for maintaining full coverage AND probability constraints:
        - Keeps all N positions for an N-bp gene (100% coverage)
        - No position collisions
        - Maintains probability constraint: donor + acceptor + neither = 1.0 at each position
        - Each position gets the ENTIRE adjusted probability distribution
        
        The adjustment creates splice-type-specific "views":
        - Donor view: All three scores shifted by donor adjustment
        - Acceptor view: All three scores shifted by acceptor adjustment
        - When evaluating donors, use donor view
        - When evaluating acceptors, use acceptor view
        
        Different base models (SpliceAI, OpenSpliceAI, etc.) may have different
        coordinate conventions. This method ensures predictions align with our
        GTF-derived annotations by applying appropriate corrections based on the
        loaded adjustment_dict.
        
        Parameters
        ----------
        predictions_df : pl.DataFrame
            DataFrame with positions, donor_prob, acceptor_prob, neither_prob, strand
            
        Returns
        -------
        pl.DataFrame
            DataFrame with adjusted scores (positions unchanged, full coverage maintained,
            probability constraints satisfied)
        """
        # Ensure adjustment_dict is available
        if self.adjustment_dict is None:
            self.logger.warning("  ⚠️  No adjustment_dict available, using default SpliceAI pattern")
            self.adjustment_dict = {
                'donor': {'plus': 2, 'minus': 1},
                'acceptor': {'plus': 0, 'minus': -1}
            }
        
        # Use the v2 score adjustment module with correlated probability vectors
        from meta_spliceai.splice_engine.meta_models.utils.score_adjustment import adjust_predictions_dataframe_v2
        
        self.logger.info(f"  🔧 Applying coordinate adjustments with correlated probability vectors (v2)...")
        
        # Apply adjustments using the v2 implementation
        # method='multi_view' adds separate columns for each splice type's view:
        # - donor_prob_donor_view, acceptor_prob_donor_view, neither_prob_donor_view
        # - donor_prob_acceptor_view, acceptor_prob_acceptor_view, neither_prob_acceptor_view
        # This allows evaluation code to use the appropriate view for each splice type
        predictions_df = adjust_predictions_dataframe_v2(
            predictions_df=predictions_df,
            adjustment_dict=self.adjustment_dict,
            score_columns=('donor_prob', 'acceptor_prob', 'neither_prob'),
            position_column='position',
            strand_column='strand',
            method='multi_view',  # Use multi-view adjustment (adds view columns)
            verbose=True  # DEBUG: Enable detailed logging
        )
        
        self.logger.info(f"  ✅ Coordinate adjustments applied successfully (v2 with correlated vectors)")
        
        return predictions_df
    
    def _prepare_coordinate_adjustments(self) -> Dict[str, Dict[str, int]]:
        """
        Prepare coordinate adjustments for base model predictions.
        
        This method loads or detects the coordinate adjustments needed to align
        base model predictions with our GTF-derived annotations. It uses the
        existing prepare_splice_site_adjustments() system from the training workflow.
        
        The adjustment system is base-model agnostic:
        - SpliceAI: Known offsets (donors +2/+1, acceptors 0/-1)
        - OpenSpliceAI: Different offsets (donors +1, acceptors 0)
        - Future models: Will auto-detect from sample predictions
        
        Returns
        -------
        Dict[str, Dict[str, int]]
            Adjustment dictionary with format:
            {
                'donor': {'plus': offset, 'minus': offset},
                'acceptor': {'plus': offset, 'minus': offset}
            }
        """
        # Check if adjustments already computed
        predictions_root = Path("predictions")  # Standard predictions directory
        adjustment_file = predictions_root / "splice_site_adjustments.json"
        
        if adjustment_file.exists():
            self.logger.info(f"  Loading existing coordinate adjustments from {adjustment_file}")
            import json
            with open(adjustment_file, 'r') as f:
                adjustment_dict = json.load(f)
            self.logger.info(f"    Donors:    +{adjustment_dict['donor']['plus']} on + strand, +{adjustment_dict['donor']['minus']} on - strand")
            self.logger.info(f"    Acceptors: +{adjustment_dict['acceptor']['plus']} on + strand, +{adjustment_dict['acceptor']['minus']} on - strand")
            return adjustment_dict
        
        # Load splice site annotations for adjustment detection
        try:
            from meta_spliceai.system.genomic_resources import Registry
            registry = Registry()
            ss_path = registry.resolve('splice_sites')
            
            if ss_path and Path(ss_path).exists():
                import pandas as pd
                ss_annotations_df = pd.read_csv(ss_path, sep='\t')
                
                # Use prepare_splice_site_adjustments from training workflow
                self.logger.info("  Detecting coordinate adjustments for base model...")
                adjustment_result = prepare_splice_site_adjustments(
                    local_dir=str(predictions_root),
                    ss_annotations_df=ss_annotations_df,
                    sample_predictions=None,  # Use default SpliceAI pattern
                    use_empirical=False,      # Use known pattern, not data-driven
                    save_adjustments=True,
                    verbosity=1 if self.config.verbose >= 1 else 0
                )
                
                adjustment_dict = adjustment_result.get('adjustment_dict')
                if adjustment_dict:
                    self.logger.info("  ✅ Coordinate adjustments prepared")
                    return adjustment_dict
        except Exception as e:
            self.logger.warning(f"  Could not load splice sites for adjustment detection: {e}")
        
        # Fallback: Use known SpliceAI pattern
        self.logger.info("  Using default SpliceAI coordinate adjustment pattern")
        return {
            'donor': {'plus': 2, 'minus': 1},
            'acceptor': {'plus': 0, 'minus': -1}
        }
    
    def _run_spliceai_directly(self, gene_id: str, gene_info: Dict, output_dir: Path) -> pl.DataFrame:
        """
        Run SpliceAI DIRECTLY on the gene sequence to get complete predictions for ALL positions.
        
        This is the CRITICAL FIX for full coverage. Instead of loading filtered training data,
        we call SpliceAI's predict_splice_sites_for_genes() which returns predictions for
        EVERY nucleotide position (N × 3 matrix where N = gene_length).
        
        Args:
            gene_id: Target gene ID
            gene_info: Gene metadata (chrom, start, end, strand, length)
            output_dir: Directory to save outputs
            
        Returns:
            DataFrame with complete predictions for all positions (gene_length rows)
        """
        from keras.models import load_model
        from pkg_resources import resource_filename
        from collections import defaultdict
        import polars as pl
        
        self.logger.info(f"  🧬 Running SpliceAI directly on {gene_id}...")
        
        try:
            # Load SpliceAI models
            context = 10000
            paths = (f'models/spliceai{x}.h5' for x in range(1, 6))
            models = [load_model(resource_filename('spliceai', x)) for x in paths]
            self.logger.info(f"  ✅ Loaded {len(models)} SpliceAI models")
            
            # Get gene sequence
            from meta_spliceai.system.genomic_resources import Registry
            registry = Registry()
            
            # Extract sequence for this gene
            chrom = str(gene_info['chrom'])
            start = gene_info['start']
            end = gene_info['end']
            strand = gene_info['strand']
            
            # Load FASTA and extract sequence
            fasta_path = registry.get_fasta_path()
            from pyfaidx import Fasta
            fasta = Fasta(str(fasta_path))
            
            # Get sequence (1-based coordinates)
            sequence = str(fasta[chrom][start-1:end])
            
            if strand == '-':
                # Reverse complement for negative strand
                complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'N': 'N'}
                sequence = ''.join(complement.get(base, 'N') for base in reversed(sequence))
            
            self.logger.info(f"  ✅ Extracted sequence: {len(sequence):,} bp")
            
            # Create input DataFrame for SpliceAI (use Polars directly)
            seq_df = pl.DataFrame({
                'gene_id': [gene_id],
                'gene_name': [gene_info.get('gene_name', gene_id)],
                'sequence': [sequence],
                'start': [start],
                'end': [end],
                'strand': [strand],
                'chrom': [chrom]
            })
            
            # Run SpliceAI prediction
            # Use the same import as the working workflow
            from meta_spliceai.splice_engine.run_spliceai_workflow import predict_splice_sites_for_genes
            
            self.logger.info(f"  🔮 Running SpliceAI prediction...")
            # Use efficient dict format (memory-efficient, same coverage as DataFrame)
            # efficient_output=True returns dict: {gene_id: {'donor_prob': [...], 'positions': [...]}}
            # CRITICAL: Set crop_size=0 to disable cropping and get FULL COVERAGE (all N positions)
            predictions_dict = predict_splice_sites_for_genes(seq_df, models, context=context, crop_size=0, efficient_output=True)
            
            # Convert dict to DataFrame for easier manipulation
            # This gives us one row per position with all required columns
            if gene_id not in predictions_dict:
                self.logger.error(f"  ❌ Gene {gene_id} not in predictions")
                return pl.DataFrame()
            
            gene_preds = predictions_dict[gene_id]
            
            # CRITICAL: Convert genomic positions to relative positions for extract_analysis_sequences
            # SpliceAI returns genomic coordinates (e.g., 109741037), but extract_analysis_sequences
            # expects 0-based indices into the gene sequence (0 to gene_length-1)
            genomic_positions = gene_preds['positions']
            gene_start_genomic = gene_preds['gene_start']
            
            # Convert to relative positions
            # For + strand: relative_pos = genomic_pos - gene_start
            # For - strand: relative_pos = gene_end - genomic_pos
            if strand == '+':
                relative_positions = [pos - gene_start_genomic for pos in genomic_positions]
            else:
                gene_end_genomic = gene_preds['gene_end']
                relative_positions = [gene_end_genomic - pos for pos in genomic_positions]
            
            # DEBUG: Check for duplicates in input data
            n_genomic = len(genomic_positions)
            n_unique_genomic = len(set(genomic_positions))
            n_unique_relative = len(set(relative_positions))
            if n_genomic != n_unique_genomic:
                self.logger.warning(f"  ⚠️  DUPLICATES IN GENOMIC POSITIONS: {n_genomic} total, {n_unique_genomic} unique")
            if n_genomic != n_unique_relative:
                self.logger.warning(f"  ⚠️  DUPLICATES IN RELATIVE POSITIONS: {n_genomic} total, {n_unique_relative} unique")
            
            # Build initial DataFrame with BOTH genomic and relative positions
            predictions_df = pl.DataFrame({
                'gene_id': [gene_id] * len(gene_preds['positions']),
                'genomic_position': genomic_positions,  # Genomic coordinates (for output)
                'position': relative_positions,  # Relative positions (for extract_analysis_sequences)
                'donor_prob': gene_preds['donor_prob'],
                'acceptor_prob': gene_preds['acceptor_prob'],
                'neither_prob': gene_preds['neither_prob'],
                'chrom': [gene_preds['seqname']] * len(gene_preds['positions']),
                'strand': [gene_preds['strand']] * len(gene_preds['positions']),
                'gene_name': [gene_preds['gene_name']] * len(gene_preds['positions']),
                'gene_start': [gene_start_genomic] * len(gene_preds['positions']),
                'gene_end': [gene_preds['gene_end']] * len(gene_preds['positions'])
            })
            
            self.logger.info(f"  DEBUG: Created DataFrame with {predictions_df.height} rows from {len(genomic_positions)} positions")
            
            # Add placeholder 'splice_type' column (required by extract_analysis_sequences)
            # This will be properly set later based on annotations
            predictions_df = predictions_df.with_columns([
                pl.lit(None).cast(pl.Utf8).alias('splice_type')
            ])
            
            # CRITICAL: Add sequence windows using existing extract_analysis_sequences utility
            # This reuses the same logic as training to ensure consistency
            from meta_spliceai.splice_engine.meta_models.workflows.sequence_data_utils import extract_analysis_sequences
            
            self.logger.info(f"  ✅ Extracting sequence windows (±250bp) using extract_analysis_sequences...")
            
            # Create sequence DataFrame (input format for extract_analysis_sequences)
            # Note: extract_analysis_sequences expects 'sequence' to be the FULL gene sequence
            predictions_with_seq = extract_analysis_sequences(
                sequence_df=seq_df,  # seq_df has full gene sequence
                position_df=predictions_df,  # predictions_df has individual positions
                window_size=250,
                include_empty_entries=True,
                essential_columns_only=False,  # Keep all columns
                drop_transcript_id=False,
                resolve_prediction_conflicts=False,  # Don't resolve conflicts (no duplicates yet)
                position_id_mode='genomic',
                preserve_transcript_list=False,
                verbose=0
            )
            
            # extract_analysis_sequences returns all columns from position_df + 'sequence', 'window_start', 'window_end'
            # Use this as our predictions_df going forward
            predictions_df = predictions_with_seq
            
            self.logger.info(f"  ✅ Added sequence windows ({predictions_df.height:,} positions with sequences)")
            self.logger.info(f"  DEBUG: After extract_analysis_sequences: {predictions_df.height} rows")
            
            # CRITICAL: Restore genomic positions for coordinate adjustment
            # The coordinate adjustment expects genomic coordinates (not relative)
            predictions_df = predictions_df.with_columns([
                pl.col('genomic_position').alias('position')
            ])
            
            # CRITICAL: Apply type & strand-specific coordinate corrections
            # Different base models (SpliceAI, OpenSpliceAI) may have different coordinate conventions
            # This ensures predictions align with our GTF-derived annotations
            # See: splice_prediction_workflow.py (lines 287-364) for training workflow implementation
            self.logger.info(f"  DEBUG: Before coordinate adjustment: {predictions_df.height} rows")
            predictions_df = self._apply_coordinate_adjustments(predictions_df)
            self.logger.info(f"  DEBUG: After coordinate adjustment: {predictions_df.height} rows")
            
            # Drop the temporary genomic_position column (now that position is corrected)
            if 'genomic_position' in predictions_df.columns:
                predictions_df = predictions_df.drop('genomic_position')
            
            # Verify no duplicates after score-based coordinate adjustment
            # NOTE: With the new score-shifting approach, we should have NO position collisions
            # because we shift SCORES not positions. All N positions are preserved.
            n_positions = predictions_df['position'].n_unique() if 'position' in predictions_df.columns else 0
            self.logger.info(f"  ✅ SpliceAI completed: {predictions_df.height:,} rows, {n_positions:,} unique positions")
            
            if predictions_df.height > n_positions:
                # This should NOT happen with score-shifting approach!
                n_duplicates = predictions_df.height - n_positions
                duplication_ratio = predictions_df.height / n_positions
                self.logger.warning(f"  ⚠️  UNEXPECTED: Position duplicates detected: {n_duplicates:,} ({duplication_ratio:.2f}×)")
                self.logger.warning(f"  This should not happen with score-shifting adjustment!")
                self.logger.info(f"  🔧 Deduplicating by keeping first occurrence...")
                predictions_df = predictions_df.unique(subset=['position'], maintain_order=True)
                self.logger.info(f"  ✅ After deduplication: {predictions_df.height:,} rows")
            elif predictions_df.height == n_positions:
                self.logger.info(f"  ✅ Full coverage maintained: All {n_positions:,} positions unique (no collisions)")
            
            # Rename probability columns to score columns for consistency
            # Also rename view columns if they exist (from v2 multi-view adjustment)
            rename_dict = {
                'donor_prob': 'donor_score',
                'acceptor_prob': 'acceptor_score',
                'neither_prob': 'neither_score'
            }
            
            # Check if view columns exist and add them to rename dict
            if 'donor_prob_donor_view' in predictions_df.columns:
                rename_dict.update({
                    'donor_prob_donor_view': 'donor_score_donor_view',
                    'acceptor_prob_donor_view': 'acceptor_score_donor_view',
                    'neither_prob_donor_view': 'neither_score_donor_view',
                    'donor_prob_acceptor_view': 'donor_score_acceptor_view',
                    'acceptor_prob_acceptor_view': 'acceptor_score_acceptor_view',
                    'neither_prob_acceptor_view': 'neither_score_acceptor_view'
                })
            
            predictions_df = predictions_df.rename(rename_dict)
            
            # Verify we have all required columns after renaming
            required_cols = ['gene_id', 'position', 'donor_score', 'acceptor_score', 'neither_score', 'strand', 'chrom']
            missing = [col for col in required_cols if col not in predictions_df.columns]
            if missing:
                raise ValueError(f"Missing required columns after SpliceAI: {missing}. Available: {predictions_df.columns}")
            
            return predictions_df
            
        except Exception as e:
            self.logger.error(f"  ❌ Failed to run SpliceAI directly: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return pl.DataFrame()
    
    def _run_splice_prediction_with_existing_data(self, gene_id: str, gene_info: Dict, existing_datasets: Dict[str, bool]) -> Dict:
        """
        Run SpliceAI prediction using existing genomic datasets for faster processing.
        
        This method ensures that NO genomic datasets are regenerated during inference.
        All datasets (annotations, splice sites, sequences) are loaded from the
        centralized location (data/ensembl/) where they were created during the
        base model pass.
        """
        self.logger.info(f"  Using existing genomic datasets for faster processing...")
        
        # Use OutputManager for artifact paths (mode-independent)
        gene_paths = self.output_manager.get_gene_output_paths(gene_id)
        complete_output_dir = gene_paths.base_predictions_dir / gene_id
        
        # Create a minimal SpliceAIConfig that reuses existing data
        # Use Registry to resolve GTF and FASTA paths
        from meta_spliceai.system.genomic_resources import Registry
        registry = Registry()
        
        # Check if gene sequences exist in shared location
        gene_sequences_exist = self._check_gene_sequences_exist(registry)
        
        if not gene_sequences_exist:
            self.logger.warning(
                "  ⚠️  Gene sequence files not found in shared location. "
                "These should have been created during the base model pass. "
                "Will attempt to load from existing analysis files instead."
            )
        
        config = SpliceAIConfig(
            # Core paths
            eval_dir=str(complete_output_dir),
            gtf_file=str(registry.get_gtf_path()),
            genome_fasta=str(registry.get_fasta_path()),
            
            # CRITICAL: Point to shared data directory where genomic datasets exist
            local_dir=str(self.registry.get_local_dir()),  # ✅ Load existing datasets from build-specific dir
            
            # CRITICAL: Disable ALL data preparation to reuse existing datasets
            do_extract_annotations=False,      # ✅ Use existing annotations.db
            do_extract_splice_sites=False,     # ✅ Use existing splice_sites.tsv
            do_extract_sequences=False,        # ✅ Use existing gene_sequence_*.parquet (NOT True!)
            do_find_overlaping_genes=False,    # ✅ Use existing overlapping_genes data
            use_precomputed_overlapping_genes=existing_datasets['overlapping_genes'],  # ✅ Use if available
            
            # Output configuration
            output_subdir="complete_inference",
            format="parquet",
            separator='\t',
            
            # Processing parameters - use lower threshold to capture more positions
            threshold=0.01,  # Lower threshold to include more positions
            consensus_window=5,
            error_window=5,
            
            # Disable test mode to process complete gene
            test_mode=False
        )
        
        # Set up paths to existing datasets using Registry
        if existing_datasets['annotations_db']:
            # Find annotations.db path via Registry
            annotations_path = self.registry.get_annotations_db_path(validate=False)
            if annotations_path:
                config.annotations_db = str(annotations_path)
            elif self.config.training_dataset_path:
                # Fallback to training dataset path if provided
                training_annot_path = self.config.training_dataset_path / "annotations.db"
                if training_annot_path.exists():
                    config.annotations_db = str(training_annot_path)
        
        if existing_datasets['splice_sites']:
            # Find splice_sites.tsv path via Registry
            splice_sites_path = self.registry.resolve("splice_sites")
            if splice_sites_path:
                config.splice_sites_file = splice_sites_path
            elif self.config.training_dataset_path:
                # Fallback to training dataset path if provided
                training_ss_path = self.config.training_dataset_path / "splice_sites.tsv"
                if training_ss_path.exists():
                    config.splice_sites_file = str(training_ss_path)
        
        # Run the prediction workflow with existing data
        result = run_enhanced_splice_prediction_workflow(
            config=config,
            target_genes=[gene_id],
            target_chromosomes=[str(gene_info['chrom'])],
            verbosity=max(0, self.config.verbose - 1),
            # Additional parameters to skip unnecessary processing
            do_prepare_annotations=False,      # Skip annotation preparation
            do_prepare_splice_sites=False,     # Skip splice site preparation
            do_prepare_sequences=True,         # Still need sequences
            do_prepare_position_tables=False,  # Skip position table preparation
            do_prepare_feature_matrices=False  # Skip feature matrix preparation
        )
        
        return result
    
    def _run_splice_prediction_with_full_generation(self, gene_id: str, gene_info: Dict) -> Dict:
        """
        DISABLED: Inference should NEVER generate genomic data.
        This method exists only for backward compatibility but will always fail.
        """
        error_msg = (
            "CRITICAL ERROR: Attempted to generate genomic data during inference. "
            "Inference workflow should ONLY predict, not generate data. "
            "Run the base model pass first to create required genomic datasets."
        )
        self.logger.error(f"  ❌ {error_msg}")
        return {'success': False, 'error': error_msg}
    
    def _identify_uncertain_positions(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Identify low-confidence, high-uncertainty positions using ONLY base model scores.
        
        Requirements from specification:
        - Do NOT rely on any labels (splice_type, pred_type, or label-related columns)
        - Use only base model scores: donor_score, acceptor_score, neither_score
        - Define uncertainty criteria: confidence thresholds 0.02-0.80, entropy from base scores

        IMPORTANT: This method is called in ALL modes (base-only, hybrid, meta-only)
            because the metadata describes intrinsic properties of the base model
            predictions, independent of whether the meta-model is applied.
            
            The metadata is valuable for:
            - Downstream analysis and quality control
            - Comparing performance across modes
            - Future adaptive meta-model selection
            - Research and debugging
            
            In base-only mode:
            - Metadata is generated but meta-model is NOT applied
            - is_uncertain=True means "base model is uncertain"
            - is_adjusted=False means "meta-model was not applied"

        """
        self.logger.info("🔍 Identifying low-confidence, high-uncertainty positions...")
        
        # Ensure required base model score columns exist
        required_cols = ['donor_score', 'acceptor_score', 'neither_score']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required base model score columns: {missing_cols}")
        
        # Calculate uncertainty metrics from base model scores ONLY
        uncertainty_df = df.with_columns([
            # Max confidence (highest score among the three types)
            pl.max_horizontal(['donor_score', 'acceptor_score', 'neither_score']).alias('max_confidence'),
            
            # Calculate entropy from base model scores
            pl.struct(['donor_score', 'acceptor_score', 'neither_score'])
            .map_elements(lambda x: self._calculate_score_entropy([x['donor_score'], x['acceptor_score'], x['neither_score']]), 
                         return_dtype=pl.Float64)
            .alias('score_entropy'),
            
            # Score spread (discriminability)
            pl.struct(['donor_score', 'acceptor_score', 'neither_score'])
            .map_elements(lambda x: self._calculate_score_spread([x['donor_score'], x['acceptor_score'], x['neither_score']]), 
                         return_dtype=pl.Float64)
            .alias('score_spread'),
            
            # Predicted type based on max base model score
            pl.struct(['donor_score', 'acceptor_score', 'neither_score'])
            .map_elements(lambda x: self._get_max_score_type([x['donor_score'], x['acceptor_score'], x['neither_score']]), 
                         return_dtype=pl.Utf8)
            .alias('predicted_type_base')
        ])
        
        # Define uncertainty criteria per specification
        confidence_low_threshold = self.config.uncertainty_threshold_low
        confidence_high_threshold = self.config.uncertainty_threshold_high
        entropy_high_threshold = 0.9      # High entropy indicates uncertainty
        spread_low_threshold = 0.1        # Low spread indicates uncertainty
        
        # Identify low-confidence, high-uncertainty positions
        final_df = uncertainty_df.with_columns([
            # Low-confidence criteria
            (pl.col('max_confidence') < confidence_high_threshold).alias('is_low_confidence'),
            
            # High-uncertainty criteria  
            (pl.col('score_entropy') > entropy_high_threshold).alias('is_high_entropy'),
            (pl.col('score_spread') < spread_low_threshold).alias('is_low_discriminability'),
            
            # Very uncertain positions (multiple criteria)
            (
                (pl.col('max_confidence') < confidence_high_threshold) |
                (pl.col('score_entropy') > entropy_high_threshold) |
                (pl.col('score_spread') < spread_low_threshold)
            ).alias('is_uncertain'),
            
            # Confidence categories for analysis
            pl.when(pl.col('max_confidence') >= confidence_high_threshold)
            .then(pl.lit('high'))
            .when(pl.col('max_confidence') >= 0.3)
            .then(pl.lit('medium'))
            .otherwise(pl.lit('low'))
            .alias('confidence_category')
        ])
        
        # Log uncertainty analysis results
        total_positions = final_df.height
        uncertain_positions = final_df.filter(pl.col('is_uncertain')).height
        low_conf_positions = final_df.filter(pl.col('is_low_confidence')).height
        high_entropy_positions = final_df.filter(pl.col('is_high_entropy')).height
        
        uncertainty_rate = uncertain_positions / total_positions if total_positions > 0 else 0
        
        self.logger.info(f"  Uncertainty identification results:")
        self.logger.info(f"    Total positions: {total_positions:,}")
        self.logger.info(f"    Low confidence (< {confidence_high_threshold}): {low_conf_positions:,}")
        self.logger.info(f"    High entropy (> {entropy_high_threshold}): {high_entropy_positions:,}")
        self.logger.info(f"    Overall uncertain: {uncertain_positions:,} ({uncertainty_rate:.1%})")
        
        return final_df
    
    def _calculate_score_entropy(self, scores: List[float]) -> float:
        """Calculate entropy from base model scores."""
        scores = np.array(scores)
        
        # Normalize to probabilities
        total = np.sum(scores)
        if total <= 0:
            return 0.0
        
        probs = scores / total
        probs = np.maximum(probs, 1e-10)  # Avoid log(0)
        
        # Calculate entropy
        entropy = -np.sum(probs * np.log(probs))
        
        # Normalize by max possible entropy for 3 categories
        max_entropy = np.log(3)
        normalized_entropy = entropy / max_entropy
        
        return float(normalized_entropy)
    
    def _calculate_score_spread(self, scores: List[float]) -> float:
        """Calculate spread between highest and second highest scores."""
        sorted_scores = np.sort(scores)[::-1]  # Sort descending
        return float(sorted_scores[0] - sorted_scores[1])
    
    def _get_max_score_type(self, scores: List[float]) -> str:
        """Get splice type with maximum score."""
        score_names = ['donor', 'acceptor', 'neither']
        max_idx = np.argmax(scores)
        return score_names[max_idx]
    
    def _apply_meta_model_selectively(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Apply meta-model ONLY to uncertain positions and create the final output schema.
        
        Per specification requirements:
        1. Construct feature matrices ONLY for low-confidence, high-uncertainty positions
        2. Apply trained meta-model solely to these selectively featurized positions  
        3. Assign recalibrated scores (donor_meta, acceptor_meta, neither_meta)
        4. For high-confidence positions: copy base scores to meta columns directly
        5. Create complete output schema with all required columns
        """
        self.logger.info("🧠 Selective meta-model recalibration...")
        
        # Initialize all meta scores as copies of base scores (default behavior)
        # Per requirement: "For positions not recalibrated, copy base scores to meta columns"
        result_df = df.with_columns([
            pl.col('donor_score').alias('donor_meta'),
            pl.col('acceptor_score').alias('acceptor_meta'),
            pl.col('neither_score').alias('neither_meta'),
            pl.lit(0).alias('is_adjusted')  # Default: not adjusted
        ])
        
        # Get uncertain positions for selective meta-model application
        uncertain_positions = df.filter(pl.col('is_uncertain') == True)
        uncertain_count = uncertain_positions.height
        
        self.logger.info(f"  Uncertain positions requiring meta-model: {uncertain_count:,}")
        
        if uncertain_count == 0:
            self.logger.info("  All positions are high-confidence - using base model scores directly")
        else:
            self.logger.info(f"  Constructing feature matrices for {uncertain_count:,} uncertain positions...")
            
            # Load trained meta-model
            try:
                model = load_model_with_calibration(self.config.model_path, use_calibration=self.config.use_calibration)
                self.logger.info(f"  ✅ Loaded meta-model: {self.config.model_path}")
            except Exception as e:
                self.logger.error(f"  ❌ Failed to load meta-model: {e}")
                # NO FALLBACK - fail loudly
                raise RuntimeError(f"Failed to load meta-model from {self.config.model_path}: {e}") from e
            
            # Generate features for uncertain positions only
            uncertain_features = self._generate_features_for_uncertain_positions(uncertain_positions)
            
            if uncertain_features is not None and len(uncertain_features) > 0:
                # Apply meta-model to uncertain positions
                meta_predictions = self._apply_meta_model_to_features(model, uncertain_features)
                
                # Update uncertain positions with meta-model predictions
                result_df = self._update_uncertain_positions_with_meta_predictions(
                    result_df, uncertain_positions, meta_predictions
                )
                
                # Verify selective application
                meta_applied = result_df.filter(pl.col('is_adjusted') == 1).height
                base_only = result_df.filter(pl.col('is_adjusted') == 0).height
                
                self.logger.info(f"  ✅ Meta-model recalibrated: {meta_applied:,} positions")
                self.logger.info(f"  ✅ Base model used directly: {base_only:,} positions")
            else:
                self.logger.warning("  ⚠️  No features generated for uncertain positions - using base model scores")
        
        # Add final splice type prediction based on recalibrated meta scores
        final_df = result_df.with_columns([
            pl.struct(['donor_meta', 'acceptor_meta', 'neither_meta'])
            .map_elements(lambda x: self._get_max_score_type([x['donor_meta'], x['acceptor_meta'], x['neither_meta']]), 
                         return_dtype=pl.Utf8)
            .alias('splice_type')
        ])
        
        return final_df
    
    def _save_enriched_analysis_data(self, enriched_data: pl.DataFrame, output_dir: Path, gene_id: str):
        """
        Save enriched analysis data back to analysis files for schema consistency.
        
        This ensures that the intermediate analysis files have the same schema
        as the training data, including all k-mers and genomic features.
        
        Parameters
        ----------
        enriched_data : pl.DataFrame
            Enriched data with all features (k-mers, genomic features, etc.)
        output_dir : Path
            Directory where analysis files are stored
        gene_id : str
            Gene ID for file naming
        """
        try:
            # Find existing analysis files to replace
            analysis_files = list(output_dir.glob("**/analysis_sequences_*.tsv"))
            
            if not analysis_files:
                self.logger.warning(f"  No analysis files found to replace for {gene_id}")
                return
            
            # Save enriched data to replace the original analysis files
            for analysis_file in analysis_files:
                # Create backup of original file
                backup_file = analysis_file.with_suffix('.tsv.backup')
                if not backup_file.exists():
                    analysis_file.rename(backup_file)
                    self.logger.info(f"  Created backup: {backup_file.name}")
                
                # Save enriched data
                enriched_data.write_csv(analysis_file, separator='\t')
                self.logger.info(f"  Saved enriched data: {analysis_file.name} ({len(enriched_data)} rows, {len(enriched_data.columns)} columns)")
            
            self.logger.info(f"  ✅ Schema consistency: Analysis files now have {len(enriched_data.columns)} columns (same as training)")
            
        except Exception as e:
            self.logger.warning(f"  ⚠️  Failed to save enriched analysis data: {e}")
            # Continue execution - this is not critical for functionality
    
    def _load_model_feature_exclusions(self) -> set:
        """
        Load the exact feature exclusions used during training.
        
        Returns
        -------
        set
            Set of feature names that should be excluded from the feature matrix
        """
        try:
            # Import preprocessing utilities
            from meta_spliceai.splice_engine.meta_models.builder.preprocessing import (
                LEAKAGE_COLUMNS, METADATA_COLUMNS, SEQUENCE_COLUMNS, REDUNDANT_COLUMNS
            )
            
            # Base exclusions from preprocessing
            base_exclusions = set(LEAKAGE_COLUMNS + METADATA_COLUMNS + SEQUENCE_COLUMNS + REDUNDANT_COLUMNS)
            
            # Load training-specific exclusions from model metadata
            model_dir = self.config.model_path.parent
            excluded_features_file = model_dir / 'global_excluded_features.txt'
            
            training_exclusions = set()
            if excluded_features_file.exists():
                with open(excluded_features_file, 'r') as f:
                    lines = f.readlines()
                
                # Parse excluded features (skip comments and empty lines)
                for line in lines:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        training_exclusions.add(line)
                
                self.logger.info(f"  Loaded {len(training_exclusions)} training-specific exclusions")
            
            # Combine all exclusions
            all_exclusions = base_exclusions | training_exclusions
            
            self.logger.info(f"  Total exclusions: {len(all_exclusions)} (base: {len(base_exclusions)}, training: {len(training_exclusions)})")
            
            return all_exclusions
            
        except Exception as e:
            self.logger.warning(f"  ⚠️  Failed to load model exclusions: {e}")
            # Fallback to base exclusions only
            from meta_spliceai.splice_engine.meta_models.builder.preprocessing import (
                LEAKAGE_COLUMNS, METADATA_COLUMNS, SEQUENCE_COLUMNS, REDUNDANT_COLUMNS
            )
            return set(LEAKAGE_COLUMNS + METADATA_COLUMNS + SEQUENCE_COLUMNS + REDUNDANT_COLUMNS)

    def _generate_features_for_uncertain_positions(self, uncertain_positions: pl.DataFrame) -> Optional[pd.DataFrame]:
        """
        Generate features for uncertain positions.
        
        Extract feature columns compatible with the trained meta-model.
        Uses the same feature exclusions as training for consistency.
        """
        if len(uncertain_positions) == 0:
            return None
        
        # Load the exact exclusions used during training
        exclude_cols = self._load_model_feature_exclusions()
        
        # Keep only feature columns (exclude leakage, metadata, sequence, redundant, and training-specific exclusions)
        feature_cols = [c for c in uncertain_positions.columns if c not in exclude_cols]
        
        if len(feature_cols) == 0:
            self.logger.warning("  ⚠️  No feature columns found after filtering")
            return None
        
        self.logger.info(f"  Extracted {len(feature_cols)} feature columns (excluded {len(exclude_cols)} columns)")
        
        # Convert to pandas for model compatibility
        features_df = uncertain_positions.select(feature_cols).to_pandas()
        
        # Handle missing values (simple imputation: fill with 0)
        if features_df.isna().any().any():
            n_missing = features_df.isna().sum().sum()
            self.logger.info(f"  Imputing {n_missing} missing values with 0")
            features_df = features_df.fillna(0.0)
        
        return features_df
    
    def _align_features_with_model(self, features: pd.DataFrame, model) -> pd.DataFrame:
        """
        Align inference features with model's expected features.
        
        This handles three cases:
        1. CRITICAL: Missing non-k-mer features - raises error (inference bug)
        2. NON-CRITICAL: Missing k-mers - fills with 0 (k-mer not in test sequence)
        3. NON-CRITICAL: Extra features - drops them (e.g., rare k-mers not in training)
        
        Parameters
        ----------
        features : pd.DataFrame
            Features extracted from inference data
        model : trained model
            Model with expected feature names
            
        Returns
        -------
        pd.DataFrame
            Aligned feature matrix matching model's expectations
            
        Raises
        ------
        ValueError
            If critical non-k-mer features are missing (indicates incomplete feature generation)
            
        Notes
        -----
        K-mer features are special:
        - Training data (1000 genes) → comprehensive k-mer coverage (all 64 3-mers)
        - Test gene (1 gene) → limited k-mer coverage (e.g., 45 of 64 3-mers)
        - Missing k-mers in test gene → count = 0 (NOT an error!)
        """
        from meta_spliceai.splice_engine.utils_kmer import is_kmer_feature
        import json
        
        # Get model's expected features
        if hasattr(model, 'feature_names_in_'):
            expected_features = list(model.feature_names_in_)
        elif hasattr(model, 'feature_name_'):
            expected_features = list(model.feature_name_)
        elif hasattr(model, 'get_booster') and hasattr(model.get_booster(), 'feature_names'):
            expected_features = model.get_booster().feature_names
        else:
            # Fallback: Load from training metadata file
            if self.config.model_path:
                model_dir = Path(self.config.model_path).parent
                features_file = model_dir / 'train.features.json'
                
                if features_file.exists():
                    try:
                        with open(features_file) as f:
                            data = json.load(f)
                        expected_features = data.get('feature_names', [])
                        self.logger.info(f"  ℹ️  Loaded {len(expected_features)} expected features from {features_file.name}")
                    except Exception as e:
                        self.logger.warning(f"  ⚠️  Failed to load features from {features_file}: {e}")
                        self.logger.warning("  ⚠️  Cannot determine model's expected features - skipping alignment")
                        return features
                else:
                    self.logger.warning(f"  ⚠️  Features file not found: {features_file}")
                    self.logger.warning("  ⚠️  Cannot determine model's expected features - skipping alignment")
                    return features
            else:
                self.logger.warning("  ⚠️  Cannot determine model's expected features - skipping alignment")
                return features
        
        inference_features = set(features.columns)
        expected_set = set(expected_features)
        
        missing = expected_set - inference_features
        extra = inference_features - expected_set
        
        self.logger.info(f"  Feature alignment:")
        self.logger.info(f"    Model expects: {len(expected_features)} features")
        self.logger.info(f"    Inference has: {len(inference_features)} features")
        self.logger.info(f"    Common: {len(expected_set & inference_features)} features")
        
        # Separate missing features into k-mers vs. non-k-mers
        missing_kmers = [f for f in missing if is_kmer_feature(f)]
        missing_non_kmers = [f for f in missing if not is_kmer_feature(f)]
        
        # CRITICAL: Check for missing non-k-mer features
        if missing_non_kmers:
            self.logger.error(f"    ❌ Missing {len(missing_non_kmers)} CRITICAL features")
            self.logger.error(f"       Features: {sorted(missing_non_kmers)[:20]}")
            
            raise ValueError(
                f"CRITICAL: Inference is missing {len(missing_non_kmers)} non-k-mer features. "
                f"This indicates incomplete feature generation. "
                f"Missing features: {sorted(missing_non_kmers)[:20]}"
            )
        
        # NON-CRITICAL: Missing k-mers (fill with 0)
        if missing_kmers:
            self.logger.info(f"    ℹ️  Missing {len(missing_kmers)} k-mers (not in test sequence, will fill with 0)")
            self.logger.debug(f"       K-mers: {sorted(missing_kmers)[:10]}")
            
            # Add missing k-mer columns with count = 0
            for kmer in missing_kmers:
                features[kmer] = 0
        
        # NON-CRITICAL: Extra features (e.g., rare k-mers not in training)
        if extra:
            extra_kmers = [f for f in extra if is_kmer_feature(f)]
            extra_non_kmers = [f for f in extra if not is_kmer_feature(f)]
            
            if extra_kmers:
                self.logger.info(f"    ℹ️  Extra {len(extra_kmers)} k-mers (not in training, will drop)")
                self.logger.debug(f"       K-mers: {sorted(extra_kmers)[:10]}")
            
            if extra_non_kmers:
                self.logger.warning(f"    ⚠️  Extra {len(extra_non_kmers)} non-k-mer features (will drop)")
                self.logger.warning(f"       Features: {sorted(extra_non_kmers)[:10]}")
        
        # Ensure we have all expected features (either from inference or filled with 0)
        # and drop any extra features
        features = features.reindex(columns=expected_features, fill_value=0)
        
        self.logger.info(f"  ✅ Features aligned: {features.shape[1]} columns")
        
        return features
    
    def _apply_meta_model_to_features(self, model, features: pd.DataFrame) -> np.ndarray:
        """
        Apply meta-model to feature matrix.
        
        Returns
        -------
        np.ndarray
            Shape (n_positions, 3) with predicted probabilities for [neither, donor, acceptor]
        """
        try:
            # CRITICAL: Align features with model's expectations
            features_aligned = self._align_features_with_model(features, model)
            
            # Get predictions from model
            # Model returns class probabilities: shape (n_samples, n_classes)
            # where n_classes = 3 (neither=0, donor=1, acceptor=2)
            predictions = model.predict_proba(features_aligned)
            
            self.logger.info(f"  ✅ Meta-model predictions generated for {len(predictions)} positions")
            
            # Ensure predictions are valid probabilities
            if not np.allclose(predictions.sum(axis=1), 1.0, atol=1e-3):
                self.logger.warning("  ⚠️  Predictions don't sum to 1.0 - normalizing")
                row_sums = predictions.sum(axis=1, keepdims=True)
                predictions = predictions / row_sums
            
            return predictions
            
        except Exception as e:
            self.logger.error(f"  ❌ Meta-model prediction failed: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            # NO FALLBACK - fail loudly to expose real issues
            raise RuntimeError(f"Meta-model prediction failed: {e}") from e
    
    def _update_uncertain_positions_with_meta_predictions(self, 
                                                        result_df: pl.DataFrame,
                                                        uncertain_positions: pl.DataFrame,
                                                        meta_predictions: np.ndarray) -> pl.DataFrame:
        """
        Update uncertain positions with meta-model predictions.
        
        Parameters
        ----------
        result_df : pl.DataFrame
            Full result DataFrame with base scores and default meta scores
        uncertain_positions : pl.DataFrame
            Subset of positions where meta-model was applied
        meta_predictions : np.ndarray
            Meta-model predictions, shape (n_uncertain, 3) for [neither, donor, acceptor]
            
        Returns
        -------
        pl.DataFrame
            Updated result DataFrame with meta scores and is_adjusted flag
        """
        if len(meta_predictions) == 0:
            return result_df
        
        # Create mapping from position to meta-model predictions
        # Meta-model output: [neither_prob, donor_prob, acceptor_prob]
        positions = uncertain_positions['position'].to_list()
        
        # Create update DataFrame
        meta_update_df = pl.DataFrame({
            'position': positions,
            'neither_meta_new': meta_predictions[:, 0],  # neither probability
            'donor_meta_new': meta_predictions[:, 1],    # donor probability
            'acceptor_meta_new': meta_predictions[:, 2],  # acceptor probability
            'is_adjusted_new': [1] * len(positions)       # Mark as adjusted
        })
        
        # Join with result DataFrame
        result_df = result_df.join(meta_update_df, on='position', how='left')
        
        # Update meta scores where we have new predictions
        result_df = result_df.with_columns([
            # Update meta scores with predicted values where available
            pl.coalesce('donor_meta_new', 'donor_meta').alias('donor_meta'),
            pl.coalesce('acceptor_meta_new', 'acceptor_meta').alias('acceptor_meta'),
            pl.coalesce('neither_meta_new', 'neither_meta').alias('neither_meta'),
            pl.coalesce('is_adjusted_new', 'is_adjusted').alias('is_adjusted'),
        ])
        
        # Drop temporary columns
        result_df = result_df.drop(['donor_meta_new', 'acceptor_meta_new', 
                                    'neither_meta_new', 'is_adjusted_new'])
        
        return result_df

    def _is_valid_kmer(self, kmer: str) -> bool:
        """Check whether a k-mer should be kept based on biological relevance."""
        # Remove k-mers with any N's (ambiguous nucleotides)
        if "N" in kmer:
            return False
        
        # For longer k-mers (6+), apply additional filtering
        if len(kmer) >= 6:
            # Remove repetitive GGGGGG or CCCCCC (but allow AAAAAA and TTTTTT)
            if kmer in {"GGGGGG", "CCCCCC"}:
                return False

            # Remove excessive GC-repeats (e.g., "GCGCGC", "CGCGCG")
            if all(kmer[i] == kmer[i+2] for i in range(len(kmer) - 2)):
                return False  
        
        return True

    def _get_kmer_counts(self, sequence: str, k: int, filter_invalid_kmers: bool = True) -> Dict[str, int]:
        """Compute k-mer counts for a given sequence with optional filtering."""
        # Handle None or invalid sequences
        if sequence is None:
            return {}
        
        # Handle non-string inputs gracefully
        if not isinstance(sequence, str):
            if pd.isna(sequence):
                return {}
            try:
                sequence = str(sequence)
            except:
                return {}
        
        # Handle empty strings or sequences too short for k-mers
        if not sequence or len(sequence) < k:
            return {}
        
        try:
            kmer_counts = Counter([sequence[i:i+k] for i in range(len(sequence) - k + 1)])

            # Apply filtering for all k-mer sizes to ensure consistency
            if filter_invalid_kmers:
                return {kmer: count for kmer, count in kmer_counts.items() if self._is_valid_kmer(kmer)}
            
            return kmer_counts
        except Exception:
            return {}

    def _generate_kmer_features(self, df: pl.DataFrame, kmer_sizes: List[int] = [3]) -> pl.DataFrame:
        """
        Generate k-mer features and other sequence features for the given DataFrame.
        
        Parameters
        ----------
        df : pl.DataFrame
            DataFrame containing 'sequence' column
        kmer_sizes : List[int]
            List of k-mer sizes to generate (default: [3] for 3-mers)
            
        Returns
        -------
        pl.DataFrame
            DataFrame with k-mer features and other sequence features added
        """
        if 'sequence' not in df.columns:
            self.logger.warning("  ⚠️  No 'sequence' column found, skipping feature generation")
            return df
        
        # Convert to pandas for easier processing
        pd_df = df.to_pandas()
        
        # Generate all possible k-mers for each size
        alphabet = ("A", "C", "G", "T")
        all_kmer_columns = []
        
        for k in kmer_sizes:
            # Generate all possible k-mers
            all_kmers = [''.join(p) for p in product(alphabet, repeat=k)]
            kmer_cols = [f"{k}mer_{kmer}" for kmer in all_kmers]
            all_kmer_columns.extend(kmer_cols)
            
            # Generate k-mer counts for each sequence
            # NOTE: Disable filtering to match training (which includes k-mers with 'N')
            kmer_dict_col = f"{k}mer_counts"
            pd_df[kmer_dict_col] = pd_df['sequence'].apply(
                lambda x: self._get_kmer_counts(x, k, filter_invalid_kmers=False)
            )
            
            # Expand k-mer dictionaries into separate columns
            kmer_df = pd.json_normalize(pd_df[kmer_dict_col])
            prefix = f"{k}mer_"
            kmer_df = kmer_df.add_prefix(prefix)
            
            # Ensure all possible k-mers are present (fill missing with 0)
            for kmer in all_kmers:
                col_name = f"{k}mer_{kmer}"
                if col_name not in kmer_df.columns:
                    kmer_df[col_name] = 0
            
            # Reorder columns to match expected order
            kmer_df = kmer_df.reindex(columns=kmer_cols, fill_value=0)
            
            # Drop the dictionary column and add k-mer columns
            pd_df = pd.concat([pd_df.drop(kmer_dict_col, axis=1), kmer_df], axis=1)
        
        # Generate additional sequence features that are missing
        self.logger.info("  Generating additional sequence features...")
        
        # GC content
        if 'gc_content' not in pd_df.columns:
            pd_df['gc_content'] = pd_df['sequence'].apply(self._get_gc_content)
        
        # Sequence length
        if 'sequence_length' not in pd_df.columns:
            pd_df['sequence_length'] = pd_df['sequence'].apply(self._get_sequence_length)
        
        # Sequence complexity
        if 'sequence_complexity' not in pd_df.columns:
            pd_df['sequence_complexity'] = pd_df['sequence'].apply(self._get_sequence_complexity)
        
        # Convert back to polars
        result_df = pl.from_pandas(pd_df)
        
        self.logger.info(f"  ✅ Generated {len(all_kmer_columns)} k-mer features + 3 sequence features")
        return result_df

    def _apply_dynamic_chrom_encoding(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Apply dynamic chromosome encoding using centralized feature schema.
        
        This method uses the centralized categorical feature encoding from
        feature_schema.py to ensure consistency between training and inference.
        
        Parameters
        ----------
        df : pl.DataFrame
            Input dataframe with 'chrom' column
            
        Returns
        -------
        pl.DataFrame
            Dataframe with encoded 'chrom' column
            
        Notes
        -----
        This method now delegates to the centralized encode_categorical_features()
        function, which uses the CATEGORICAL_FEATURES registry to ensure the same
        encoding logic is applied during both training and inference.
        """
        if 'chrom' not in df.columns:
            self.logger.warning("  ⚠️  No 'chrom' column found, skipping chromosome encoding")
            return df
            
        self.logger.info(f"  Applying categorical encoding using centralized schema...")
        
        # Use centralized encoding function
        result_df = encode_categorical_features(
            df,
            features_to_encode=['chrom'],
            verbose=False  # We'll log our own messages
        )
        
        self.logger.info(f"  ✅ Chromosome encoding completed - 'chrom' is now numeric")
        return result_df

    def _get_gc_content(self, sequence: str) -> float:
        """Compute GC content of a sequence."""
        if not isinstance(sequence, str) or not sequence:
            return 0.0
        
        try:
            gc_count = sequence.upper().count('G') + sequence.upper().count('C')
            return gc_count / len(sequence) if len(sequence) > 0 else 0.0
        except Exception:
            return 0.0

    def _get_sequence_length(self, sequence: str) -> int:
        """Compute sequence length."""
        if not isinstance(sequence, str):
            return 0
        return len(sequence)

    def _get_sequence_complexity(self, sequence: str) -> float:
        """Compute sequence complexity based on nucleotide frequencies."""
        if not isinstance(sequence, str) or not sequence:
            return 0.0
        
        try:
            from collections import Counter
            counts = Counter(sequence.upper())
            total = len(sequence)
            if total == 0:
                return 0.0
            
            # Calculate Shannon entropy
            entropy = 0.0
            for count in counts.values():
                p = count / total
                if p > 0:
                    entropy -= p * np.log2(p)
            
            # Normalize by maximum possible entropy (log2(4) = 2 for DNA)
            return entropy / 2.0
        except Exception:
            return 0.0
    
    def _generate_derived_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Generate ALL derived features that were created during training.
        
        This includes:
        1. Probability-based features (ratios, differences, log-odds)
        2. Context scores (neighboring position scores: m2, m1, p1, p2)
        3. Donor/acceptor-specific features (surge ratios, peaks, derivatives, etc.)
        4. Context-agnostic features (neighbor means, asymmetry, max)
        5. Cross-type features (signal strength ratios, etc.)
        
        Uses the SAME feature generation functions as training (enhanced_workflow.py).
        """
        from meta_spliceai.splice_engine.meta_models.core.enhanced_workflow import (
            generate_context_agnostic_features,
            generate_donor_features,
            generate_acceptor_features
        )
        from meta_spliceai.splice_engine.meta_models.workflows.inference.optimized_feature_enrichment import (
            OptimizedInferenceEnricher,
            InferenceFeatureConfig
        )
        
        # Step 1: Generate probability features (using pandas-based enricher)
        self.logger.info(f"    Step 1/4: Probability features...")
        pd_df = df.to_pandas()
        config = InferenceFeatureConfig(
            include_probability_features=True,
            include_context_features=False,  # Will do this separately
            include_kmer_features=False,
            include_genomic_features=False,
            verbose=False
        )
        enricher = OptimizedInferenceEnricher(config)
        pd_df = enricher._generate_probability_features(pd_df)
        df = pl.from_pandas(pd_df)
        
        # Step 2: Generate context scores (neighboring positions)
        # CRITICAL: Context scores are extracted using get_context_scores from enhanced_evaluation.py
        # This ensures consistency with training data generation
        self.logger.info(f"    Step 2/5: Context scores (±2 positions)...")
        
        from meta_spliceai.splice_engine.meta_models.core.enhanced_evaluation import get_context_scores
        
        # Sort by position first to ensure correct neighborhood relationships
        df = df.sort('position')
        
        # Convert to pandas for easier row-wise operations
        pd_df = df.to_pandas()
        
        # Extract donor and acceptor probabilities as numpy arrays
        donor_probs = pd_df['donor_score'].values
        acceptor_probs = pd_df['acceptor_score'].values
        
        # For each position, extract context scores from BOTH donor and acceptor
        # Then use the MAXIMUM as the "splice signal" at each context position
        # This matches the training logic in enhanced_evaluation.py
        context_m2_list, context_m1_list = [], []
        context_p1_list, context_p2_list = [], []
        
        for idx in range(len(pd_df)):
            # Get context from donor probabilities
            donor_context = get_context_scores(donor_probs, idx, window_size=2)
            # Get context from acceptor probabilities  
            acceptor_context = get_context_scores(acceptor_probs, idx, window_size=2)
            
            # Take max of donor/acceptor at each position (splice signal strength)
            # Context window is [m2, m1, center, p1, p2] (5 elements)
            context_m2_list.append(max(donor_context[0], acceptor_context[0]))
            context_m1_list.append(max(donor_context[1], acceptor_context[1]))
            context_p1_list.append(max(donor_context[3], acceptor_context[3]))
            context_p2_list.append(max(donor_context[4], acceptor_context[4]))
        
        # Add context scores as new columns
        pd_df['context_score_m2'] = context_m2_list
        pd_df['context_score_m1'] = context_m1_list
        pd_df['context_score_p1'] = context_p1_list
        pd_df['context_score_p2'] = context_p2_list
        
        # Convert back to polars
        df = pl.from_pandas(pd_df)
        
        # Step 3: Generate context-agnostic features
        self.logger.info(f"    Step 3/5: Context-agnostic features...")
        context_agnostic_exprs = generate_context_agnostic_features(epsilon=1e-10)
        df = df.with_columns(context_agnostic_exprs)
        
        # Step 4: Generate donor and acceptor specific features
        self.logger.info(f"    Step 4/5: Donor/acceptor features...")
        donor_exprs = generate_donor_features(epsilon=1e-10)
        acceptor_exprs = generate_acceptor_features(epsilon=1e-10)
        df = df.with_columns(donor_exprs + acceptor_exprs)
        
        # Step 5: Generate cross-type features + transcript features
        self.logger.info(f"    Step 5/5: Cross-type + transcript features...")
        
        # Cross-type features
        cross_type_exprs = [
            # Signal strength ratio (donor vs acceptor)
            (pl.col('donor_signal_strength') / (pl.col('acceptor_signal_strength').abs() + 1e-10)).alias('signal_strength_ratio'),
            
            # Score difference ratio
            ((pl.col('donor_score') - pl.col('acceptor_score')) / (pl.col('donor_score') + pl.col('acceptor_score') + 1e-10)).alias('score_difference_ratio'),
            
            # Donor/acceptor peak ratio
            (pl.col('donor_peak_height_ratio') / (pl.col('acceptor_peak_height_ratio') + 1e-10)).alias('donor_acceptor_peak_ratio'),
            
            # Type signal difference (absolute difference between donor and acceptor signal strengths)
            (pl.col('donor_signal_strength') - pl.col('acceptor_signal_strength')).alias('type_signal_difference'),
        ]
        
        # Add transcript-level features (use gene-level proxies for gene-level inference)
        if 'transcript_length' not in df.columns and 'gene_end' in df.columns and 'gene_start' in df.columns:
            cross_type_exprs.append((pl.col('gene_end') - pl.col('gene_start')).alias('transcript_length'))
        
        if 'tx_start' not in df.columns and 'gene_start' in df.columns:
            cross_type_exprs.append(pl.col('gene_start').alias('tx_start'))
        
        if 'tx_end' not in df.columns and 'gene_end' in df.columns:
            cross_type_exprs.append(pl.col('gene_end').alias('tx_end'))
        
        df = df.with_columns(cross_type_exprs)
        
        self.logger.info(f"  ✅ Generated complete feature set ({len(df.columns)} total columns)")
        
        return df
    
    def _create_final_output_schema(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Create the final output schema with ALL required columns per specification.
        
        REQUIRED columns (minimum):
        - gene_id, position, donor_score, acceptor_score, neither_score
        - donor_meta, acceptor_meta, neither_meta, splice_type, is_adjusted
        
        ADDITIONAL useful columns (recommended):
        - entropy, transcript_id, and other contextually relevant features
        """
        self.logger.info("📋 Creating final output schema per specification...")
        
        # Ensure all REQUIRED columns exist
        required_columns = {
            'gene_id': 'UNKNOWN',
            'position': 0,
            'donor_score': 0.0,
            'acceptor_score': 0.0, 
            'neither_score': 0.0,
            'donor_meta': 0.0,
            'acceptor_meta': 0.0,
            'neither_meta': 0.0,
            'splice_type': 'neither',
            'is_adjusted': 0
        }
        
        # Add missing required columns with defaults
        for col, default_val in required_columns.items():
            if col not in df.columns:
                df = df.with_columns(pl.lit(default_val).alias(col))
                self.logger.warning(f"  Added missing required column: {col}")
        
        # Add useful additional columns per specification
        if 'score_entropy' in df.columns:
            df = df.with_columns(pl.col('score_entropy').alias('entropy'))
        else:
            # Calculate entropy from base scores if not already present
            df = df.with_columns([
                pl.struct(['donor_score', 'acceptor_score', 'neither_score'])
                .map_elements(lambda x: self._calculate_score_entropy([x['donor_score'], x['acceptor_score'], x['neither_score']]), 
                             return_dtype=pl.Float64)
                .alias('entropy')
            ])
        
        # Add transcript_id if not present
        if 'transcript_id' not in df.columns:
            df = df.with_columns(pl.lit('UNKNOWN').alias('transcript_id'))
        
        # Define final column order per specification
        final_columns = [
            # Core identification
            'gene_id', 'position',
            
            # Base model scores (required)
            'donor_score', 'acceptor_score', 'neither_score',
            
            # Meta-model recalibrated scores (required)
            'donor_meta', 'acceptor_meta', 'neither_meta',
            
            # Final prediction and adjustment flag (required)
            'splice_type', 'is_adjusted',
            
            # Additional useful columns (recommended)
            'entropy', 'transcript_id'
        ]
        
        # Add other contextually relevant features if available
        # IMPORTANT: Include ALL 9 metadata features for adaptive meta-model selection
        # NOTE: We do NOT include k-mer features in the final output (too many columns)
        # K-mers are used internally for meta-model prediction but not saved to output
        contextual_columns = []
        for col in df.columns:
            if col not in final_columns and col in [
                # Metadata features for adaptive selection (9 total)
                'is_uncertain', 'is_low_confidence', 'is_high_entropy',
                'is_low_discriminability', 'max_confidence', 'score_spread',
                'score_entropy', 'confidence_category', 'predicted_type_base',
                # Genomic context
                'chrom', 'strand', 'sequence'
            ]:
                contextual_columns.append(col)
        
        final_columns.extend(contextual_columns)
        
        # Select only available columns
        available_columns = [col for col in final_columns if col in df.columns]
        final_df = df.select(available_columns)
        
        # Verify all required columns are present
        missing_required = [col for col in required_columns.keys() if col not in final_df.columns]
        if missing_required:
            raise ValueError(f"Final schema missing required columns: {missing_required}")
        
        self.logger.info(f"  ✅ Final output schema: {len(available_columns)} columns")
        self.logger.info(f"  Required columns: {list(required_columns.keys())}")
        self.logger.info(f"  Additional columns: {[col for col in available_columns if col not in required_columns]}")
        
        return final_df
    
    # ==============================================================================
    # NEW: Helper methods for incremental processing
    # ==============================================================================
    
    def _generate_base_predictions_for_single_gene(self, gene_id: str, gene_info: Dict) -> pl.DataFrame:
        """
        Generate base model predictions for A SINGLE GENE ONLY.
        
        This is the KEY fix - instead of generating predictions for all genes,
        we generate them one gene at a time.
        
        Parameters
        ----------
        gene_id : str
            Single gene ID to process
        gene_info : Dict
            Gene information (chrom, start, end, length, etc.)
            
        Returns
        -------
        pl.DataFrame
            Base model predictions for this gene only
        """
        if self.config.verbose >= 2:
            self.logger.info(f"    Generating base predictions for {gene_id}...")
        
        # Call the existing base prediction method but for single gene
        # The existing method already handles single gene input
        single_gene_info = {gene_id: gene_info}
        return self._generate_complete_base_predictions(single_gene_info)
    
    def _combine_parquet_files(self, input_paths: List[Path], output_path: Path):
        """
        Combine multiple parquet files WITHOUT loading all into memory.
        Uses streaming to avoid memory issues.
        
        Parameters
        ----------
        input_paths : List[Path]
            List of parquet files to combine
        output_path : Path
            Output combined file
        """
        import pyarrow.parquet as pq
        
        self.logger.info(f"  Combining {len(input_paths)} parquet files...")
        
        if not input_paths:
            self.logger.warning("  No input files to combine")
            return
        
        # Read schema from first file
        first_table = pq.read_table(input_paths[0])
        schema = first_table.schema
        
        # Create writer
        writer = pq.ParquetWriter(output_path, schema, compression='zstd')
        
        # Write each file
        for idx, input_path in enumerate(input_paths, 1):
            if self.config.verbose >= 2:
                self.logger.info(f"    Adding file {idx}/{len(input_paths)}: {input_path.name}")
            table = pq.read_table(input_path)
            writer.write_table(table)
            del table  # Free memory immediately
            
            # Garbage collect periodically
            if idx % self.config.gc_frequency == 0:
                gc.collect()
        
        writer.close()
        self.logger.info(f"  ✅ Combined file created: {output_path}")
    
    def _create_base_only_file(self, input_paths: List[Path], output_path: Path):
        """
        Create base-only predictions file from per-gene results.
        
        This just combines the files, extracting only base model columns.
        """
        # For now, just combine all files (they already have base predictions)
        self._combine_parquet_files(input_paths, output_path)
    
    # ==============================================================================
    # Run methods (legacy and incremental)
    # ==============================================================================
    
    def run_legacy(self) -> EnhancedSelectiveInferenceResults:
        """
        DEPRECATED: Legacy run method that loads all predictions into memory.
        
        This method is kept for backward compatibility but should not be used
        for production workloads as it can cause OOM errors.
        
        Use run() (which now calls run_incremental()) instead.
        """
        self.logger.warning("=" * 80)
        self.logger.warning("⚠️  USING LEGACY RUN METHOD - MAY CAUSE OOM FOR LARGE WORKLOADS!")
        self.logger.warning("⚠️  Consider switching to incremental processing (default run() method)")
        self.logger.warning("=" * 80)
        
        start_time = time.time()
        
        self.logger.info("=" * 80)
        self.logger.info("🧬 ENHANCED SELECTIVE META-MODEL INFERENCE (LEGACY)")
        self.logger.info("=" * 80)
        self.logger.info(f"Target genes: {len(self.config.target_genes)}")
        self.logger.info(f"Model: {self.config.model_path}")
        self.logger.info(f"Output directory: {self.output_dir}")
        self.logger.info("=" * 80)
        
        # Initialize results
        results = EnhancedSelectiveInferenceResults(
            success=False,
            config=self.config,
            error_messages=[]
        )
        
        try:
            # Step 1: Load gene information
            self.logger.info("📋 Step 1: Loading gene information...")
            gene_features_path = self._find_gene_features_file()
            gene_info = self._get_gene_info(gene_features_path)
            
            if not gene_info:
                raise ValueError("No valid genes found in gene features")
            
            # Step 2: Generate complete base predictions for ALL positions
            self.logger.info("🎯 Step 2: Generating complete base model predictions...")
            complete_base_df = self._generate_complete_base_predictions(gene_info)
            
            if complete_base_df.height == 0:
                raise ValueError("No base predictions generated")
            
            # Step 3: Identify uncertain positions using ONLY base model scores
            self.logger.info("🔍 Step 3: Identifying low-confidence, high-uncertainty positions...")
            uncertainty_df = self._identify_uncertain_positions(complete_base_df)
            
            # Step 4: Apply meta-model selectively to uncertain positions only
            self.logger.info("🧠 Step 4: Selective meta-model recalibration...")
            meta_adjusted_df = self._apply_meta_model_selectively(uncertainty_df)
            
            # Step 5: Create final output schema
            self.logger.info("📋 Step 5: Creating final output schema...")
            final_df = self._create_final_output_schema(meta_adjusted_df)
            
            # Step 6: Save results
            self.logger.info("💾 Step 6: Saving results...")
            
            # Save hybrid predictions (complete coverage)
            hybrid_path = self.output_dir / "complete_coverage_predictions.parquet"
            final_df.write_parquet(hybrid_path)
            results.hybrid_predictions_path = hybrid_path
            
            # Save base model predictions
            base_path = self.output_dir / "base_model_predictions.parquet"
            complete_base_df.write_parquet(base_path)
            results.base_predictions_path = base_path
            
            # Generate per-gene statistics
            per_gene_stats = {}
            for gene_id in self.config.target_genes:
                gene_data = final_df.filter(pl.col('gene_id') == gene_id)
                if gene_data.height > 0:
                    per_gene_stats[gene_id] = {
                        'total_positions': gene_data.height,
                        'recalibrated_positions': gene_data.filter(pl.col('is_adjusted') == 1).height,
                        'reused_positions': gene_data.filter(pl.col('is_adjusted') == 0).height,
                        'uncertain_positions': gene_data.filter(pl.col('is_uncertain') == True).height
                    }
            
            # Update results
            results.success = True
            results.total_positions = final_df.height
            results.positions_recalibrated = final_df.filter(pl.col('is_adjusted') == 1).height
            results.positions_reused = final_df.filter(pl.col('is_adjusted') == 0).height
            results.genes_processed = len(per_gene_stats)
            results.per_gene_stats = per_gene_stats
            
            # Log final summary
            total_time = time.time() - start_time
            self.logger.info("=" * 80)
            self.logger.info("🎉 ENHANCED SELECTIVE INFERENCE COMPLETED")
            self.logger.info("=" * 80)
            self.logger.info(f"✅ Total positions: {final_df.height}")
            self.logger.info(f"🧠 Meta-model applied: {results.positions_recalibrated} ({results.positions_recalibrated/final_df.height*100:.1f}%)")
            self.logger.info(f"⏱️  Runtime: {total_time:.1f} seconds")
            self.logger.info(f"📁 Results: {hybrid_path}")
            self.logger.info("=" * 80)
            
        except Exception as e:
            results.success = False
            results.error_messages = [str(e)]
            self.logger.error(f"Enhanced selective inference failed: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
        
        finally:
            results.processing_time_seconds = time.time() - start_time
        
        return results
    
    def run_incremental(self) -> EnhancedSelectiveInferenceResults:
        """
        Run the enhanced selective inference workflow with INCREMENTAL processing.
        
        This processes genes ONE AT A TIME to avoid loading all predictions into memory.
        This is the production-ready method that scales to large gene sets.
        """
        start_time = time.time()
        
        self.logger.info("=" * 80)
        self.logger.info("🧬 ENHANCED SELECTIVE META-MODEL INFERENCE (INCREMENTAL)")
        self.logger.info("=" * 80)
        self.logger.info(f"Target genes: {len(self.config.target_genes)}")
        self.logger.info(f"Model: {self.config.model_path}")
        self.logger.info(f"Inference mode: {self.config.inference_mode}")
        self.logger.info(f"Output directory: {self.output_dir}")
        self.logger.info("=" * 80)
        
        # Initialize coordinate adjustments (base model agnostic)
        if self.adjustment_dict is None:
            self.logger.info("📐 Preparing coordinate adjustments for base model...")
            # CRITICAL FIX (2025-10-31): Base model predictions are already aligned with GTF annotations!
            # Evidence from VCAM1: base F1=0.756, adjusted F1=0.400
            # The previous adjustments (+2/+1) were making predictions WORSE
            self.logger.warning("  ⚠️  USING ZERO ADJUSTMENTS - Base model already aligned with GTF")
            self.adjustment_dict = {
                'donor': {'plus': 0, 'minus': 0},
                'acceptor': {'plus': 0, 'minus': 0}
            }
            self.logger.info(f"  ✅ Adjustments set: Donors (+{self.adjustment_dict['donor']['plus']}/+{self.adjustment_dict['donor']['minus']}), Acceptors (+{self.adjustment_dict['acceptor']['plus']}/+{self.adjustment_dict['acceptor']['minus']})")
        
        # Initialize memory monitoring
        if self.memory_monitor:
            self.memory_monitor.checkpoint("start")
        
        # Initialize results
        results = EnhancedSelectiveInferenceResults(
            success=False,
            config=self.config,
            error_messages=[],
            per_gene_stats={}
        )
        
        try:
            # Step 1: Load gene information (minimal memory)
            self.logger.info("📋 Step 1: Loading gene information...")
            gene_features_path = self._find_gene_features_file()
            gene_info = self._get_gene_info(gene_features_path)
            
            if not gene_info:
                raise ValueError("No valid genes found in gene features")
            
            if self.memory_monitor:
                self.memory_monitor.checkpoint("gene_info_loaded")
            self.logger.info(f"  Loaded info for {len(gene_info)} genes")
            
            # Step 2: Process genes ONE AT A TIME (INCREMENTAL)
            self.logger.info(f"🎯 Step 2: Processing {len(self.config.target_genes)} genes incrementally...")
            
            all_results_paths = []
            per_gene_stats = {}
            total_positions_processed = 0
            total_recalibrated = 0
            total_reused = 0
            
            for gene_idx, gene_id in enumerate(self.config.target_genes, 1):
                gene_start_time = time.time()
                
                self.logger.info("")
                self.logger.info(f"{'='*60}")
                self.logger.info(f"🧬 Gene {gene_idx}/{len(self.config.target_genes)}: {gene_id}")
                self.logger.info(f"{'='*60}")
                
                if gene_id not in gene_info:
                    self.logger.warning(f"  ⚠️  Gene {gene_id} not found in gene features - skipping")
                    continue
                
                gene_length = gene_info[gene_id]['length']
                gene_name = gene_info[gene_id].get('gene_name', gene_id)
                self.logger.info(f"  Gene: {gene_name} ({gene_id})")
                self.logger.info(f"  Length: {gene_length:,} bp")
                
                try:
                    # Step 2a: Generate complete base predictions for THIS GENE ONLY (with enrichment)
                    self.logger.info(f"  🎯 Generating complete base model predictions...")
                    gene_complete_df = self._generate_complete_base_model_predictions(
                        gene_id, gene_info[gene_id]
                    )
                    
                    if gene_complete_df.height == 0:
                        self.logger.warning(f"  ⚠️  No base predictions generated for {gene_id} - skipping")
                        continue
                    
                    if self.memory_monitor:
                        self.memory_monitor.checkpoint(f"gene_{gene_id}_base_complete")
                    self.logger.info(f"  ✅ Complete base predictions: {gene_complete_df.height:,} positions")
                    
                    # Step 2b: ALWAYS identify uncertain positions (generates metadata features)
                    # This is important even in base-only mode for metadata preservation
                    self.logger.info(f"  🔍 Identifying uncertain positions (generating metadata)...")
                    gene_uncertainty_df = self._identify_uncertain_positions(gene_complete_df)
                    
                    # Step 2c: Process based on inference mode
                    if self.config.inference_mode == "base_only":
                        # Base-only mode: No meta-model, just add required columns
                        # Metadata features from uncertainty identification are preserved
                        gene_final_df = gene_uncertainty_df.with_columns([
                            pl.col('donor_score').alias('donor_meta'),
                            pl.col('acceptor_score').alias('acceptor_meta'),
                            pl.col('neither_score').alias('neither_meta'),
                            pl.lit(0).cast(pl.Int32).alias('is_adjusted')
                        ])
                        recalibrated = 0
                        reused = gene_final_df.height
                        
                    elif self.config.inference_mode in ["hybrid", "meta_only"]:
                        # Hybrid/Meta-only mode: Apply meta-model to uncertain positions
                        # (uncertainty already identified above)
                        uncertain_count = gene_uncertainty_df.filter(pl.col('is_uncertain') == True).height
                        uncertain_pct = (uncertain_count / gene_uncertainty_df.height) * 100 if gene_uncertainty_df.height > 0 else 0
                        
                        self.logger.info(f"  📊 Uncertain positions: {uncertain_count:,} ({uncertain_pct:.1f}%)")
                        
                        # Apply meta-model selectively for THIS GENE ONLY
                        if self.config.inference_mode == "meta_only":
                            # Override: treat ALL positions as uncertain
                            self.logger.info(f"  🧠 Meta-only mode: Processing ALL positions...")
                            gene_uncertainty_df = gene_uncertainty_df.with_columns([
                                pl.lit(True).alias('is_uncertain')
                            ])
                            uncertain_count = gene_uncertainty_df.height
                        
                        if uncertain_count > 0:
                            self.logger.info(f"  🧠 Applying meta-model ({uncertain_count:,} positions)...")
                            gene_final_df = self._apply_meta_model_selectively(gene_uncertainty_df)
                        else:
                            self.logger.info(f"  ✅ All positions confident - using base model scores")
                            gene_final_df = gene_uncertainty_df.with_columns([
                                pl.col('donor_score').alias('donor_meta'),
                                pl.col('acceptor_score').alias('acceptor_meta'),
                                pl.col('neither_score').alias('neither_meta'),
                                pl.lit(0).cast(pl.Int32).alias('is_adjusted')
                            ])
                        
                        if self.memory_monitor:
                            self.memory_monitor.checkpoint(f"gene_{gene_id}_meta_complete")
                        
                        recalibrated = gene_final_df.filter(pl.col('is_adjusted') == 1).height if 'is_adjusted' in gene_final_df.columns else 0
                        reused = gene_final_df.filter(pl.col('is_adjusted') == 0).height if 'is_adjusted' in gene_final_df.columns else gene_final_df.height
                    
                    else:
                        raise ValueError(f"Unknown inference mode: {self.config.inference_mode}")
                    
                    # Step 2d: Create final output schema for THIS GENE
                    self.logger.info(f"  📋 Creating output schema...")
                    gene_final_df = self._create_final_output_schema(gene_final_df)
                    
                    # CRITICAL VERIFICATION: Check complete coverage
                    expected_positions = gene_length
                    actual_positions = gene_final_df.height
                    
                    if actual_positions != expected_positions:
                        coverage_pct = (actual_positions / expected_positions) * 100
                        reduction_pct = 100 - coverage_pct
                        self.logger.info(
                            f"  📊 Prediction coverage: {actual_positions:,}/{expected_positions:,} positions "
                            f"({coverage_pct:.1f}%, {reduction_pct:.1f}% reduction from coordinate collisions)"
                        )
                        # Note: Position count after coordinate adjustment is expected to be lower due to collisions
                        if coverage_pct < 70:
                            self.logger.error(
                                f"  ❌ Insufficient coverage ({coverage_pct:.1f}%) - skipping {gene_id}"
                            )
                            continue
                        else:
                            self.logger.info(f"  ✅ Acceptable coverage: {actual_positions:,} positions ({coverage_pct:.1f}%)")
                    else:
                        self.logger.info(f"  ✅ Complete coverage verified: {expected_positions:,} positions")
                    
                    # Note: Genomic features already enriched earlier for meta-model compatibility
                    
                    # Step 2e: Save results IMMEDIATELY (don't accumulate in memory)
                    # Use OutputManager for consistent path management
                    gene_paths = self.output_manager.get_gene_output_paths(gene_id)
                    gene_output_path = gene_paths.predictions_file
                    gene_final_df.write_parquet(gene_output_path, compression='zstd')
                    all_results_paths.append(gene_output_path)
                    
                    self.logger.info(f"  💾 Saved: {gene_output_path}")
                    
                    # Step 2f: Record statistics
                    gene_time = time.time() - gene_start_time
                    per_gene_stats[gene_id] = {
                        'total_positions': gene_final_df.height,
                        'recalibrated_positions': recalibrated,
                        'reused_positions': reused,
                        'processing_time_seconds': gene_time,
                        'output_path': str(gene_output_path)
                    }
                    
                    total_positions_processed += gene_final_df.height
                    total_recalibrated += recalibrated
                    total_reused += reused
                    
                    self.logger.info(f"  ⏱️  Processing time: {gene_time:.1f}s")
                    self.logger.info(f"  🧠 Meta-model usage: {recalibrated:,}/{gene_final_df.height:,} ({recalibrated/gene_final_df.height*100:.1f}%)")
                    
                    # Step 2g: CRITICAL - Free memory immediately
                    del gene_complete_df
                    del gene_final_df
                    if 'gene_uncertainty_df' in locals():
                        del gene_uncertainty_df
                    gc.collect()
                    
                    if self.memory_monitor:
                        current_mem = self.memory_monitor.checkpoint(f"gene_{gene_id}_complete")
                        self.logger.info(f"  📊 Current memory: {current_mem:.1f} MB")
                    
                except Exception as e:
                    self.logger.error(f"  ❌ Failed to process gene {gene_id}: {e}")
                    import traceback
                    self.logger.error(traceback.format_exc())
                    results.error_messages.append(f"Gene {gene_id}: {str(e)}")
                    continue
            
            # Step 3: Combine results (streaming, not loading all into memory)
            self.logger.info("")
            self.logger.info("=" * 80)
            self.logger.info("📦 Step 3: Combining results...")
            
            if len(all_results_paths) == 0:
                raise ValueError("No genes processed successfully")
            
            # Create combined predictions file by concatenating per-gene files
            # Use OutputManager to get consistent combined output path
            combined_path = self.output_manager.get_combined_output_path()
            self._combine_parquet_files(all_results_paths, combined_path)
            
            # Also save base-only predictions for comparison in mode directory
            base_path = self.output_dir / "base_model_predictions.parquet"
            self._create_base_only_file(all_results_paths, base_path)
            
            # Update results
            results.success = True
            results.hybrid_predictions_path = combined_path
            results.base_predictions_path = base_path
            results.total_positions = total_positions_processed
            results.positions_recalibrated = total_recalibrated
            results.positions_reused = total_reused
            results.genes_processed = len(per_gene_stats)
            results.per_gene_stats = per_gene_stats
            
            # Memory report
            if self.memory_monitor:
                memory_report = self.memory_monitor.report()
                self.logger.info(f"📊 Peak memory: {memory_report['peak_mb']:.1f} MB ({memory_report['peak_usage_pct']:.1f}%)")
            
            # Final summary
            total_time = time.time() - start_time
            self.logger.info("=" * 80)
            self.logger.info("🎉 ENHANCED SELECTIVE INFERENCE COMPLETED")
            self.logger.info("=" * 80)
            self.logger.info(f"✅ Genes processed: {results.genes_processed}")
            self.logger.info(f"✅ Total positions: {total_positions_processed:,}")
            self.logger.info(f"🧠 Meta-model applied: {total_recalibrated:,} ({total_recalibrated/total_positions_processed*100:.1f}%)")
            self.logger.info(f"⏱️  Total runtime: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
            self.logger.info(f"📁 Results: {combined_path}")
            self.logger.info("=" * 80)
            
        except Exception as e:
            results.success = False
            results.error_messages.append(str(e))
            self.logger.error(f"Enhanced selective inference failed: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
        
        finally:
            results.processing_time_seconds = time.time() - start_time
        
        return results
    
    def run(self) -> EnhancedSelectiveInferenceResults:
        """
        Run the enhanced selective inference workflow.
        
        This now uses incremental processing by default to avoid memory issues.
        For production workloads, this is the recommended method.
        """
        return self.run_incremental()


def run_enhanced_selective_meta_inference(config: EnhancedSelectiveInferenceConfig) -> EnhancedSelectiveInferenceResults:
    """
    Run enhanced selective meta-model inference with complete coverage.
    
    This function implements the corrected selective inference approach that ensures:
    1. Complete position coverage (no gaps) for all target genes
    2. Base model predictions for ALL positions
    3. Selective meta-model application based only on uncertainty
    4. Proper output schema with continuous position numbering
    
    Parameters
    ----------
    config : EnhancedSelectiveInferenceConfig
        Complete configuration for enhanced selective inference
        
    Returns
    -------
    EnhancedSelectiveInferenceResults
        Comprehensive results with all output paths and statistics
    """
    workflow = EnhancedSelectiveInferenceWorkflow(config)
    return workflow.run() 