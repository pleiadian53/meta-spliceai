#!/usr/bin/env python3
"""
Practical Main-Entry Meta-Model Inference Workflow

A comprehensive, production-ready script for performing meta-model inference 
on arbitrary genes using pre-trained meta-models. Supports both selective 
processing for efficiency and complete coverage capability.

FEATURES:
- Flexible parameterization for models, datasets, and target genes
- Efficient selective processing (reuses confident base model predictions)
- Complete coverage capability (generates full probability tensors)
- Structured data management with gene manifests
- Dynamic uncertainty thresholds and performance reporting
- Robust error handling and verbose logging
- Optional parallel execution capability

KEY USAGE EXAMPLES:

1. Basic inference on specific genes:
   python -m meta_spliceai.splice_engine.meta_models.workflows.inference.main_inference_workflow \
       --model results/gene_cv_pc_1000_3mers_run_4/model_multiclass.pkl \
       --training-dataset train_pc_1000_3mers \
       --genes ENSG00000154358,ENSG00000100490 \
       --output-dir ./inference_results

2. High-performance parallel processing:
   python -m meta_spliceai.splice_engine.meta_models.workflows.inference.main_inference_workflow \
       --model results/gene_cv_pc_1000_3mers_run_4/model_multiclass.pkl \
       --training-dataset train_pc_1000_3mers \
       --genes-file top_error_prone_genes.txt \
       --parallel-workers 4 \
       --strategy UNCERTAINTY_FOCUSED \
       --output-dir ./production_inference \
       --verbose

3. Complete coverage analysis (all positions):
   python -m meta_spliceai.splice_engine.meta_models.workflows.inference.main_inference_workflow \
       --model results/gene_cv_pc_1000_3mers_run_4/model_multiclass.pkl \
       --genes ENSG00000154358 \
       --complete-coverage \
       --uncertainty-low 0.01 \
       --uncertainty-high 0.90 \
       --output-dir ./comprehensive_analysis

4. Novel gene discovery (genes not in training):
   python -m meta_spliceai.splice_engine.meta_models.workflows.inference.main_inference_workflow \
       --model results/gene_cv_pc_1000_3mers_run_4/model_multiclass.pkl \
       --genes ENSG00000198888,ENSG00000198899 \
       --strategy SELECTIVE \
       --max-positions 50000 \
       --output-dir ./novel_gene_analysis

PROCESSING STRATEGIES:
- SELECTIVE (default): ~80-95% memory reduction, selective recalibration
- COMPLETE: Full coverage, all positions processed  
- UNCERTAINTY_FOCUSED: Minimal processing, highest uncertainty only
- TRAINING_GAPS: Focus on positions not in original training data

OUTPUT STRUCTURE:
inference_results/
├── inference_workflow.log              # Detailed execution log
├── gene_manifest.json                  # Gene processing tracking
├── inference_summary.json              # Results summary
├── performance_report.txt              # Performance metrics
├── genes/                              # Individual gene results
│   ├── GENE_ID/
│   │   ├── GENE_ID_predictions.parquet
│   │   └── GENE_ID_statistics.json
└── selective_inference/                # Intermediate artifacts

PERFORMANCE:
- Memory efficiency: 80-95% reduction vs traditional approaches
- Parallel processing: Multi-core support with --parallel-workers
- Incremental processing: Reuses existing results via gene manifest
- Selective featurization: Only processes uncertain positions

For comprehensive documentation, see MAIN_INFERENCE_WORKFLOW.md

Author: Splice Surveyor Team
License: MIT
"""

import argparse
import json
import logging
import multiprocessing
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import traceback
import polars as pl

# Add project root to path for imports
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parents[4]  # Go up to splice-surveyor root
sys.path.insert(0, str(project_root))

import pandas as pd

# Optional MLflow import - only import if enabled
try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
import polars as pl
from .complete_coverage_predictor import CompleteCoveragePredictor, run_complete_coverage_inference
from .uncertainty_analyzer import UncertaintyAnalyzer, analyze_prediction_uncertainty
from tqdm import tqdm

# Import core inference components
from meta_spliceai.splice_engine.meta_models.workflows.selective_meta_inference import (
    SelectiveInferenceConfig,
    SelectiveInferenceResults,
    run_selective_meta_inference
)
from meta_spliceai.splice_engine.meta_models.workflows.inference.performance_analysis import (
    PerformanceAnalyzer
)

from meta_spliceai.splice_engine.meta_models.training.meta_evaluation_utils import (
    enhanced_generate_per_nucleotide_meta_scores
)
from meta_spliceai.splice_engine.meta_models.core.data_types import SpliceAIConfig
from meta_spliceai.splice_engine.utils import setup_logger, calculate_duration, format_time


def resolve_model_path(model_arg: str, keywords: Optional[List[str]] = None) -> str:
    """Resolve a --model argument that may be a file or a directory.

    If a directory is provided, attempt to find the most appropriate trained
    meta-model file within it, preferring well-known filenames and extensions.

    Preference order (by scoring heuristic):
      1) Exact name: model_multiclass.pkl
      2) Names containing ordered keywords (default: ["multiclass", "model", "best", "final"]).
         Earlier keywords in the list carry higher weight, and scores are additive.
      Then prefer .pkl over .joblib over .sav, and break ties by newest mtime.

    Parameters
    ----------
    model_arg : str
        Path to a model file or a directory that contains model files.

    Returns
    -------
    str
        Resolved absolute path to the chosen model file.
    """
    base_path = Path(model_arg).expanduser().resolve()
    # Default keyword precedence if none provided
    kw_list: List[str] = [k.strip().lower() for k in (keywords or ["multiclass", "model", "best", "final"]) if k.strip()]
    if base_path.is_file():
        return str(base_path)

    if not base_path.exists():
        raise FileNotFoundError(f"Model path does not exist: {base_path}")

    if not base_path.is_dir():
        raise FileNotFoundError(f"Model path is not a file or directory: {base_path}")

    def score_candidate(path: Path) -> Tuple[int, float]:
        name = path.name.lower()
        score = 0
        if name == "model_multiclass.pkl":
            score += 100
        # Keyword precedence scoring
        # Earlier keywords get higher weight; matches accumulate
        # Example weights: len(kw_list)-idx down to 1, scaled by 10
        for idx, kw in enumerate(kw_list):
            if kw and kw in name:
                score += (len(kw_list) - idx) * 10
        # Extension preference
        if path.suffix == ".pkl":
            score += 5
        elif path.suffix == ".joblib":
            score += 3
        elif path.suffix == ".sav":
            score += 1
        # mtime as tie-breaker (newer is preferred)
        try:
            mtime = path.stat().st_mtime
        except Exception:
            mtime = 0.0
        return score, mtime

    # First pass: look only at top-level files
    top_level_files = [
        p for p in base_path.glob("*")
        if p.is_file() and p.suffix in {".pkl", ".joblib", ".sav"}
    ]

    candidates = top_level_files

    # If no candidates at top-level, do a recursive search
    if not candidates:
        candidates = [
            p for p in base_path.rglob("*")
            if p.is_file() and p.suffix in {".pkl", ".joblib", ".sav"}
        ]

    if not candidates:
        raise FileNotFoundError(
            f"No model files found under directory: {base_path}. "
            f"Please provide a specific model file (e.g., model_multiclass.pkl)."
        )

    # Prefer the highest scoring, newest candidate
    candidates_sorted = sorted(
        candidates,
        key=lambda p: score_candidate(p),
        reverse=True,
    )
    return str(candidates_sorted[0].resolve())


class InferenceWorkflowConfig:
    """Configuration class for the main inference workflow."""
    
    def __init__(
        self,
        model_path: str,
        training_dataset_path: Optional[str] = None,
        target_genes: Optional[List[str]] = None,
        genes_file: Optional[str] = None,
        output_dir: str = "./inference_results",
        uncertainty_threshold_low: float = 0.02,
        uncertainty_threshold_high: float = 0.80,
        # Enhanced uncertainty selection parameters
        target_meta_selection_rate: float = 0.10,  # Target ~10% of positions for meta-model
        uncertainty_selection_strategy: str = "hybrid_entropy",  # "confidence_only", "entropy_only", "hybrid_entropy"
        enable_adaptive_uncertainty_tuning: bool = True,
        max_positions_per_gene: int = 10000,
        complete_coverage: bool = False,
        # Chunked processing for true meta_only mode
        enable_chunked_processing: bool = False,
        chunk_size: int = 10000,
        selective_strategy: str = "SELECTIVE",
        parallel_workers: int = 1,
        verbose: int = 1,
        cleanup_intermediates: bool = True,
        performance_reporting: bool = True,
        force_recompute: bool = False,
        inference_mode: str = "hybrid",
        keep_artifacts_dir: Optional[str] = None,
        preserve_artifacts: bool = False,
        model_keywords: Optional[List[str]] = None,
        # MLflow options (optional; auto-enable when provided)
        mlflow_enable: bool = False,
        mlflow_experiment: Optional[str] = None,
        mlflow_tracking_uri: Optional[str] = None,
        mlflow_tags: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize inference workflow configuration.
        
        Parameters
        ----------
        model_path : str
            Path to pre-trained meta-model (.pkl file)
        training_dataset_path : str, optional
            Path to original training dataset directory
        target_genes : List[str], optional
            List of gene IDs to analyze
        genes_file : str, optional
            Path to file containing gene IDs (one per line)
        output_dir : str
            Output directory for predictions and artifacts
        uncertainty_threshold_low : float
            Lower threshold for uncertainty-based selection (default: 0.02)
        uncertainty_threshold_high : float
            Upper threshold for uncertainty-based selection (default: 0.80)
        target_meta_selection_rate : float
            Target percentage of positions for meta-model inference (default: 0.10 = 10%)
        uncertainty_selection_strategy : str
            Selection strategy: "confidence_only", "entropy_only", "hybrid_entropy" (default: "hybrid_entropy")
        enable_adaptive_uncertainty_tuning : bool
            Enable adaptive threshold tuning to achieve target selection rate (default: True)
        max_positions_per_gene : int
            Maximum positions to process per gene (default: 10000)
        complete_coverage : bool
            Generate predictions for ALL positions (default: False for efficiency)
        selective_strategy : str
            Strategy for position selection: SELECTIVE, COMPLETE, TRAINING_GAPS, UNCERTAINTY_FOCUSED
        parallel_workers : int
            Number of parallel workers (default: 1)
        verbose : int
            Verbosity level (0=quiet, 1=normal, 2=detailed, 3=debug)
        cleanup_intermediates : bool
            Clean up intermediate files after processing
        performance_reporting : bool
            Generate performance reports
        force_recompute : bool
            Force recomputation even if results exist
        inference_mode : str
            Inference strategy: 'base_only', 'hybrid' (default), 'meta_only'
        model_keywords : List[str], optional
            Ordered keyword precedence for picking a model within a directory.
            Defaults to ["multiclass", "model", "best", "final"].
        """
        self.model_path = model_path
        self.training_dataset_path = training_dataset_path
        self.target_genes = target_genes or []
        self.genes_file = genes_file
        self.output_dir = output_dir
        self.uncertainty_threshold_low = uncertainty_threshold_low
        self.uncertainty_threshold_high = uncertainty_threshold_high
        self.target_meta_selection_rate = target_meta_selection_rate
        self.uncertainty_selection_strategy = uncertainty_selection_strategy
        self.enable_adaptive_uncertainty_tuning = enable_adaptive_uncertainty_tuning
        self.max_positions_per_gene = max_positions_per_gene
        self.complete_coverage = complete_coverage
        self.enable_chunked_processing = enable_chunked_processing
        self.chunk_size = chunk_size
        self.selective_strategy = selective_strategy
        self.parallel_workers = parallel_workers
        self.verbose = verbose
        self.cleanup_intermediates = cleanup_intermediates
        self.performance_reporting = performance_reporting
        self.force_recompute = force_recompute
        self.inference_mode = inference_mode
        self.keep_artifacts_dir = keep_artifacts_dir
        self.preserve_artifacts = preserve_artifacts
        self.model_keywords = model_keywords
        # Store MLflow options; validation will finalize defaults/auto-enable
        self.mlflow_enable = mlflow_enable
        self.mlflow_experiment = mlflow_experiment
        self.mlflow_tracking_uri = mlflow_tracking_uri
        self.mlflow_tags = mlflow_tags or {}
        
        # Validate inputs
        self._validate_config()
    
    def _validate_config(self):
        """Validate configuration parameters."""
        # Resolve model path: allow directory input and auto-pick best model file
        try:
            potential = Path(self.model_path).expanduser().resolve()
            if potential.is_dir():
                self.model_path = resolve_model_path(str(potential), keywords=self.model_keywords)
            else:
                # If not a directory, must be an existing file
                if not potential.exists() or not potential.is_file():
                    raise FileNotFoundError
                self.model_path = str(potential)
        except FileNotFoundError:
            raise FileNotFoundError(f"Model file not found or resolvable: {self.model_path}")
        
        # Check training dataset path if provided
        if self.training_dataset_path and not os.path.exists(self.training_dataset_path):
            raise FileNotFoundError(f"Training dataset path not found: {self.training_dataset_path}")
        
        # Check genes file if provided
        if self.genes_file and not os.path.exists(self.genes_file):
            raise FileNotFoundError(f"Genes file not found: {self.genes_file}")
        
        # Validate thresholds
        if not (0 <= self.uncertainty_threshold_low <= 1):
            raise ValueError("uncertainty_threshold_low must be between 0 and 1")
        if not (0 <= self.uncertainty_threshold_high <= 1):
            raise ValueError("uncertainty_threshold_high must be between 0 and 1")
        if self.uncertainty_threshold_low >= self.uncertainty_threshold_high:
            raise ValueError("uncertainty_threshold_low must be less than uncertainty_threshold_high")
        
        # Validate strategy
        valid_strategies = ["SELECTIVE", "COMPLETE", "TRAINING_GAPS", "UNCERTAINTY_FOCUSED"]
        if self.selective_strategy not in valid_strategies:
            raise ValueError(f"selective_strategy must be one of: {valid_strategies}")
        
        # Validate inference mode
        valid_modes = ["base_only", "hybrid", "meta_only"]
        if self.inference_mode not in valid_modes:
            raise ValueError(f"inference_mode must be one of: {valid_modes}")
        
        # Auto-enable complete coverage for ALL inference workflows (required for complete position coverage)
        # 
        # CRITICAL INSIGHT: Inference workflows fundamentally differ from training workflows:
        # 1. Training artifacts have TN downsampling (missing many positions)
        # 2. Inference needs ALL positions for complete coverage (Scenarios 1, 2a, 2b)
        # 3. This applies to ALL inference modes (base_only, hybrid, meta_only)
        # 4. Selective workflow only works with complete pre-computed artifacts
        #
        # Therefore, inference workflows should ALWAYS use complete coverage to ensure:
        # - Scenario 1: Observed genes get complete position coverage (not just training subset)  
        # - Scenario 2a: Unseen genes with artifacts get fresh complete predictions
        # - Scenario 2b: Unseen genes without artifacts get fresh complete predictions
        if not self.complete_coverage:
            self.complete_coverage = True
            if self.verbose >= 1:
                print("🔄 Auto-enabled complete coverage for inference (required for complete position coverage)")
                print("   This ensures all scenarios work: observed genes (unseen positions) + unseen genes")
        
        # Auto-enable chunked processing for complete coverage to prevent OOM
        # Complete coverage can generate large feature matrices, so chunked processing is essential
        if self.complete_coverage and not self.enable_chunked_processing:
            self.enable_chunked_processing = True
            if self.verbose >= 1:
                print("🔄 Auto-enabled chunked processing for complete coverage (prevents OOM for large genes)")
        
        # Validate parallel workers
        if self.parallel_workers < 1:
            raise ValueError("parallel_workers must be at least 1")
        if self.parallel_workers > multiprocessing.cpu_count():
            logging.warning(f"parallel_workers ({self.parallel_workers}) exceeds CPU count ({multiprocessing.cpu_count()})")
        
        # MLflow configuration (optional)
        self.mlflow_enable = getattr(self, 'mlflow_enable', False)
        self.mlflow_experiment = getattr(self, 'mlflow_experiment', None)
        self.mlflow_tracking_uri = getattr(self, 'mlflow_tracking_uri', None)
        self.mlflow_tags = getattr(self, 'mlflow_tags', {})

        # Auto-enable MLflow if tracking-related options were provided
        if not self.mlflow_enable and (self.mlflow_tracking_uri is not None or self.mlflow_experiment is not None):
            self.mlflow_enable = True

        # Apply default experiment name if enabled but not specified
        if self.mlflow_enable and not self.mlflow_experiment:
            self.mlflow_experiment = 'surveyor-inference'


class InferenceWorkflowRunner:
    """Main runner class for the inference workflow."""
    
    def __init__(self, config: InferenceWorkflowConfig):
        """Initialize the workflow runner."""
        self.config = config
        self.logger = None
        self.start_time = None
        self.performance_stats = {}
        self.mlflow_run = None
        
        # Setup output directory
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        
        # Setup MLflow if enabled
        self._setup_mlflow()
        
        # Load target genes
        self._load_target_genes()
    
    def _setup_logging(self):
        """Setup logging configuration."""
        log_level = {
            0: logging.WARNING,
            1: logging.INFO,
            2: logging.DEBUG,
            3: logging.DEBUG
        }.get(self.config.verbose, logging.INFO)
        
        log_file = os.path.join(self.config.output_dir, "inference_workflow.log")
        self.logger = setup_logger(log_file)
        
        # Set the logging level for the logger
        self.logger.setLevel(log_level)
        
        # Also set console handler level if it exists
        for handler in self.logger.handlers:
            if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
                handler.setLevel(log_level)
        
        self.logger.info("=" * 80)
        self.logger.info("🧬 SPLICE SURVEYOR - META-MODEL INFERENCE WORKFLOW")
        self.logger.info("=" * 80)
    
    def _load_target_genes(self):
        """Load target genes from file or command line."""
        if self.config.genes_file:
            self.logger.info(f"📋 Loading target genes from: {self.config.genes_file}")
            with open(self.config.genes_file, 'r') as f:
                file_genes = [line.strip() for line in f if line.strip()]
            self.config.target_genes.extend(file_genes)
            self.logger.info(f"   Loaded {len(file_genes)} genes from file")
        
        if not self.config.target_genes:
            raise ValueError("No target genes specified. Use --genes or --genes-file.")
        
        # Remove duplicates while preserving order
        seen = set()
        unique_genes = []
        for gene in self.config.target_genes:
            if gene not in seen:
                seen.add(gene)
                unique_genes.append(gene)
        
        self.config.target_genes = unique_genes
        self.logger.info(f"📊 Total unique target genes: {len(self.config.target_genes)}")
    
    def _setup_mlflow(self):
        """Setup MLflow tracking if enabled."""
        if not self.config.mlflow_enable:
            return
        
        if not MLFLOW_AVAILABLE:
            self.logger.warning("⚠️  MLflow is not installed. Tracking disabled.")
            self.config.mlflow_enable = False
            return
        
        try:
            # Set tracking URI if provided
            if self.config.mlflow_tracking_uri:
                mlflow.set_tracking_uri(self.config.mlflow_tracking_uri)
                self.logger.info(f"🔗 MLflow tracking URI: {self.config.mlflow_tracking_uri}")
            
            # Set experiment
            mlflow.set_experiment(self.config.mlflow_experiment)
            self.logger.info(f"📊 MLflow experiment: {self.config.mlflow_experiment}")
            
            # Start MLflow run
            run_name = f"inference_{time.strftime('%Y%m%d_%H%M%S')}"
            self.mlflow_run = mlflow.start_run(run_name=run_name)
            
            # Log initial parameters
            mlflow.log_params({
                "model_path": self.config.model_path,
                "training_dataset": self.config.training_dataset_path or "unknown",
                "num_target_genes": len(self.config.target_genes),
                "uncertainty_threshold_low": self.config.uncertainty_threshold_low,
                "uncertainty_threshold_high": self.config.uncertainty_threshold_high,
                "max_positions_per_gene": self.config.max_positions_per_gene,
                "complete_coverage": self.config.complete_coverage,
                "selective_strategy": self.config.selective_strategy,
                "inference_mode": self.config.inference_mode,
                "parallel_workers": self.config.parallel_workers,
                "output_dir": self.config.output_dir
            })
            
            # Log tags
            tags = self.config.mlflow_tags.copy()
            tags["workflow"] = "inference"
            tags["model_type"] = "meta_model"
            mlflow.set_tags(tags)
            
            self.logger.info(f"✅ MLflow run started: {self.mlflow_run.info.run_id}")
            
        except Exception as e:
            self.logger.warning(f"⚠️  Failed to setup MLflow: {e}")
            self.config.mlflow_enable = False
    
    def _create_gene_manifest(self) -> Path:
        """Create or load gene manifest file."""
        manifest_path = Path(self.config.output_dir) / "gene_manifest.json"
        
        if manifest_path.exists() and not self.config.force_recompute:
            self.logger.info(f"📖 Loading existing gene manifest: {manifest_path}")
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
        else:
            self.logger.info(f"📝 Creating new gene manifest: {manifest_path}")
            manifest = {
                "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                "model_path": self.config.model_path,
                "training_dataset_path": self.config.training_dataset_path,
                "config": self._config_to_dict(),
                "processed_genes": {},
                "performance_summary": {}
            }
        
        return manifest_path, manifest
    
    def _config_to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for serialization."""
        return {
            "uncertainty_threshold_low": self.config.uncertainty_threshold_low,
            "uncertainty_threshold_high": self.config.uncertainty_threshold_high,
            "max_positions_per_gene": self.config.max_positions_per_gene,
            "complete_coverage": self.config.complete_coverage,
            "selective_strategy": self.config.selective_strategy,
            "parallel_workers": self.config.parallel_workers,
            "cleanup_intermediates": self.config.cleanup_intermediates
        }
    
    def _save_gene_manifest(self, manifest_path: Path, manifest: Dict[str, Any]):
        """Save gene manifest to file."""
        manifest["last_updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        self.logger.debug(f"💾 Saved gene manifest: {manifest_path}")
    
    def _check_gene_processed(self, manifest: Dict[str, Any], gene_id: str) -> bool:
        """Check if gene has already been processed."""
        if self.config.force_recompute:
            return False
        
        gene_entry = manifest.get("processed_genes", {}).get(gene_id)
        if not gene_entry:
            return False
        
        # Check if config matches
        current_config = self._config_to_dict()
        stored_config = gene_entry.get("config", {})
        
        # Key parameters that affect results
        key_params = [
            "uncertainty_threshold_low", "uncertainty_threshold_high",
            "max_positions_per_gene", "complete_coverage", "selective_strategy"
        ]
        
        for param in key_params:
            if current_config.get(param) != stored_config.get(param):
                self.logger.debug(f"   Config mismatch for {gene_id}: {param}")
                return False
        
        # Check if output files exist
        output_file = gene_entry.get("output_file")
        if output_file and os.path.exists(output_file):
            return True
        
        return False
    
    def _process_single_gene(self, gene_id: str) -> Tuple[str, Dict[str, Any]]:
        """Process a single gene and return results."""
        gene_start_time = time.time()
        
        try:
            self.logger.info(f"🧬 Processing gene: {gene_id}")
            self.logger.debug(f"   Using model file: {self.config.model_path}")
            
            # Use selective inference with complete coverage flag
            if self.config.complete_coverage:
                self.logger.info(f"  Using selective inference with complete coverage")
            else:
                self.logger.info(f"  Using selective inference for efficiency")
            
            # For complete coverage, don't limit positions (use chunked processing instead)
            # For selective inference, use position limits for efficiency
            max_positions = (
                0 if self.config.complete_coverage
                else self.config.max_positions_per_gene
            )
            
            selective_config = SelectiveInferenceConfig(
                model_path=Path(self.config.model_path),
                target_genes=[gene_id],
                training_dataset_path=Path(self.config.training_dataset_path) if self.config.training_dataset_path else None,
                training_schema_path=Path(self.config.model_path).parent if self.config.model_path else None,
                uncertainty_threshold_low=self.config.uncertainty_threshold_low,
                uncertainty_threshold_high=self.config.uncertainty_threshold_high,
                max_positions_per_gene=max_positions,
                inference_base_dir=Path(self.config.output_dir) / "selective_inference",
                keep_artifacts_dir=Path(self.config.keep_artifacts_dir) if self.config.keep_artifacts_dir else None,
                verbose=max(0, self.config.verbose - 1),
                cleanup_intermediates=self.config.cleanup_intermediates,
                inference_mode=self.config.inference_mode,
                ensure_complete_coverage=self.config.complete_coverage,
                enable_chunked_processing=self.config.enable_chunked_processing,
                chunk_size=self.config.chunk_size
            )
            
            # Run selective meta inference
            workflow_results = run_selective_meta_inference(selective_config)
            
            if not workflow_results.success:
                error_msg = workflow_results.error_messages[0] if workflow_results.error_messages else "Unknown error"
                raise RuntimeError(f"Selective inference failed: {error_msg}")
            
            # Save gene-specific results
            gene_output_dir = Path(self.config.output_dir) / "genes" / gene_id
            os.makedirs(gene_output_dir, exist_ok=True)
            
            output_file = gene_output_dir / f"{gene_id}_predictions.parquet"
            
            if workflow_results.hybrid_predictions_path is not None:
                # Load the predictions from the file
                import polars as pl
                hybrid_predictions = pl.read_parquet(workflow_results.hybrid_predictions_path)
                
                # Save predictions to gene-specific location
                hybrid_predictions.write_parquet(output_file)
                
                # Generate summary statistics
                stats = self._generate_gene_statistics(hybrid_predictions, gene_id)
                
                # Save statistics
                stats_file = gene_output_dir / f"{gene_id}_statistics.json"
                with open(stats_file, 'w') as f:
                    json.dump(stats, f, indent=2)
                
                processing_time = time.time() - gene_start_time
                
                result = {
                    "success": True,
                    "output_file": str(output_file),
                    "statistics_file": str(stats_file),
                    "processing_time": processing_time,
                    "statistics": stats,
                    "config": self._config_to_dict(),
                    "processed_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }
                
                self.logger.info(f"   ✅ {gene_id} completed in {processing_time:.1f}s")
                
                # Log to MLflow if enabled
                if self.config.mlflow_enable and MLFLOW_AVAILABLE:
                    try:
                        # Log per-gene metrics
                        mlflow.log_metrics({
                            f"gene_{gene_id}_positions": stats["total_positions"],
                            f"gene_{gene_id}_meta_positions": stats.get("prediction_sources", {}).get("meta_model_positions", 0),
                            f"gene_{gene_id}_processing_time": processing_time
                        })
                        
                        # Log gene artifacts
                        mlflow.log_artifact(str(output_file), f"genes/{gene_id}")
                        mlflow.log_artifact(str(stats_file), f"genes/{gene_id}")
                        
                    except Exception as e:
                        self.logger.debug(f"Failed to log gene {gene_id} to MLflow: {e}")
                
                return gene_id, result
            
            else:
                raise RuntimeError("No predictions generated")
                
        except Exception as e:
            processing_time = time.time() - gene_start_time
            error_msg = f"Failed to process {gene_id}: {str(e)}"
            self.logger.error(error_msg)
            self.logger.debug(f"   Full traceback: {traceback.format_exc()}")
            
            result = {
                "success": False,
                "error": error_msg,
                "processing_time": processing_time,
                "config": self._config_to_dict(),
                "processed_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            return gene_id, result
    
    def _generate_gene_statistics(self, predictions_df: pl.DataFrame, gene_id: str) -> Dict[str, Any]:
        """Generate comprehensive statistics for a gene's predictions."""
        try:
            total_positions = len(predictions_df)
            
            # Count predictions by type (use pred_type or confidence_category if available)
            if "pred_type" in predictions_df.columns:
                prediction_counts = predictions_df.group_by("pred_type").agg(
                    pl.len().alias("count")
                ).to_pandas()
                
                class_distribution = {
                    row["pred_type"]: int(row["count"]) 
                    for _, row in prediction_counts.iterrows()
                }
            elif "confidence_category" in predictions_df.columns:
                prediction_counts = predictions_df.group_by("confidence_category").agg(
                    pl.len().alias("count")
                ).to_pandas()
                
                class_distribution = {
                    row["confidence_category"]: int(row["count"]) 
                    for _, row in prediction_counts.iterrows()
                }
            else:
                class_distribution = {"unknown": total_positions}
            
            # Generate before/after comparison for meta-model impact
            base_only_distribution = self._calculate_base_only_classification(predictions_df)
            
            # Calculate confidence statistics (use consistent _meta columns)
            confidence_stats = {}
            for class_name in ["donor", "acceptor", "neither"]:
                # Try different column naming conventions
                meta_col = f"{class_name}_meta"
                hybrid_col = f"{class_name}_hybrid"
                
                if meta_col in predictions_df.columns:
                    probs = predictions_df.select(pl.col(meta_col)).to_pandas()[meta_col]
                elif hybrid_col in predictions_df.columns:
                    probs = predictions_df.select(pl.col(hybrid_col)).to_pandas()[hybrid_col]
                else:
                    continue
                    
                confidence_stats[class_name] = {
                    "mean": float(probs.mean()),
                    "std": float(probs.std()),
                    "min": float(probs.min()),
                    "max": float(probs.max()),
                    "median": float(probs.median())
                }
            
            # Check for base vs meta model usage
            meta_used = predictions_df.filter(pl.col("prediction_source") == "meta_model").height
            base_used = predictions_df.filter(pl.col("prediction_source") == "base_model").height
            
            return {
                "gene_id": gene_id,
                "total_positions": total_positions,
                "class_distribution": class_distribution,
                "base_only_distribution": base_only_distribution,
                "confidence_statistics": confidence_stats,
                "prediction_sources": {
                    "meta_model_positions": meta_used,
                    "base_model_positions": base_used,
                    "meta_model_percentage": (meta_used / total_positions * 100) if total_positions > 0 else 0
                },
                "genomic_coverage": {
                    "start_position": int(predictions_df.select(pl.col("position").min()).item()),
                    "end_position": int(predictions_df.select(pl.col("position").max()).item()),
                    "span": int(predictions_df.select(pl.col("position").max()).item() - 
                               predictions_df.select(pl.col("position").min()).item() + 1)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error generating statistics for {gene_id}: {e}")
            return {
                "gene_id": gene_id,
                "error": str(e),
                "total_positions": len(predictions_df) if predictions_df is not None else 0
            }
    
    def _calculate_base_only_classification(self, predictions_df: pl.DataFrame) -> Dict[str, int]:
        """Calculate what the classification would be using only base model scores.
        
        This accurately simulates base-only performance by comparing base model scores
        to determine what predictions would have been made without meta-model intervention.
        """
        try:
            if "pred_type" not in predictions_df.columns:
                return {"unknown": len(predictions_df)}
            
            # Check if we have the necessary columns
            required_cols = ["donor_score", "acceptor_score", "neither_score", "splice_type"]
            if not all(col in predictions_df.columns for col in required_cols):
                # If we don't have base scores, return current distribution
                prediction_counts = predictions_df.group_by("pred_type").agg(
                    pl.len().alias("count")
                ).to_pandas()
                return {
                    row["pred_type"]: int(row["count"]) 
                    for _, row in prediction_counts.iterrows()
                }
            
            # Calculate base-only predictions for all positions
            base_only_dist = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}
            
            for row in predictions_df.to_dicts():
                donor_score = row.get("donor_score", 0)
                acceptor_score = row.get("acceptor_score", 0)
                neither_score = row.get("neither_score", 1)
                splice_type = row.get("splice_type")
                
                # Determine what base model would predict
                max_score = max(donor_score, acceptor_score, neither_score)
                
                if splice_type is not None:
                    # This is an actual splice site
                    if splice_type == "donor":
                        # For donor sites, check if donor score wins
                        if donor_score == max_score and donor_score > neither_score:
                            base_only_dist["TP"] += 1
                        else:
                            base_only_dist["FN"] += 1
                    elif splice_type == "acceptor":
                        # For acceptor sites, check if acceptor score wins
                        if acceptor_score == max_score and acceptor_score > neither_score:
                            base_only_dist["TP"] += 1
                        else:
                            base_only_dist["FN"] += 1
                    else:
                        # Unknown splice type, use conservative approach
                        if max(donor_score, acceptor_score) > neither_score:
                            base_only_dist["TP"] += 1
                        else:
                            base_only_dist["FN"] += 1
                else:
                    # This is not a splice site (neither position)
                    if neither_score == max_score and neither_score > max(donor_score, acceptor_score):
                        base_only_dist["TN"] += 1
                    else:
                        base_only_dist["FP"] += 1
            
            # Store detailed impact analysis for reporting
            meta_positions = predictions_df.filter(pl.col("prediction_source") == "meta_model")
            if meta_positions.height > 0:
                improvements = 0
                regressions = 0
                
                for row in meta_positions.to_dicts():
                    splice_type = row.get("splice_type")
                    if splice_type:
                        # Check if meta model improved this position
                        if splice_type == "donor":
                            base_correct = row["donor_score"] > row["neither_score"]
                            meta_correct = row["donor_meta"] > row["neither_meta"]
                        elif splice_type == "acceptor":
                            base_correct = row["acceptor_score"] > row["neither_score"]
                            meta_correct = row["acceptor_meta"] > row["neither_meta"]
                        else:
                            base_correct = max(row["donor_score"], row["acceptor_score"]) > row["neither_score"]
                            meta_correct = max(row["donor_meta"], row["acceptor_meta"]) > row["neither_meta"]
                        
                        if not base_correct and meta_correct:
                            improvements += 1
                        elif base_correct and not meta_correct:
                            regressions += 1
                
                self._meta_model_impact = {
                    "positions_processed": meta_positions.height,
                    "improvements": improvements,
                    "regressions": regressions,
                    "net_impact": improvements - regressions
                }
            
            return base_only_dist
            
        except Exception as e:
            self.logger.debug(f"Error calculating base-only classification: {e}")
            # Fallback: return current distribution
            if "pred_type" in predictions_df.columns:
                prediction_counts = predictions_df.group_by("pred_type").agg(
                    pl.len().alias("count")
                ).to_pandas()
                
                return {
                    row["pred_type"]: int(row["count"]) 
                    for _, row in prediction_counts.iterrows()
                }
            else:
                return {"unknown": len(predictions_df)}
    
    def _process_genes_parallel(self, genes_to_process: List[str]) -> Dict[str, Dict[str, Any]]:
        """Process genes in parallel."""
        if self.config.parallel_workers == 1:
            # Sequential processing
            results = {}
            for gene_id in tqdm(genes_to_process, desc="Processing genes"):
                gene_id, result = self._process_single_gene(gene_id)
                results[gene_id] = result
            return results
        
        else:
            # Parallel processing
            self.logger.info(f"🔄 Processing {len(genes_to_process)} genes with {self.config.parallel_workers} workers")
            
            with multiprocessing.Pool(self.config.parallel_workers) as pool:
                gene_results = pool.map(self._process_single_gene, genes_to_process)
            
            return dict(gene_results)
    
    def _generate_summary_report(self, all_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Generate comprehensive summary report."""
        successful_genes = [gene for gene, result in all_results.items() if result.get("success", False)]
        failed_genes = [gene for gene, result in all_results.items() if not result.get("success", False)]
        
        total_processing_time = sum(result.get("processing_time", 0) for result in all_results.values())
        
        # Aggregate statistics from successful genes
        total_positions = 0
        total_meta_positions = 0
        class_totals = {}
        base_only_totals = {}
        
        for gene in successful_genes:
            result = all_results[gene]
            stats = result.get("statistics", {})
            
            total_positions += stats.get("total_positions", 0)
            
            source_stats = stats.get("prediction_sources", {})
            total_meta_positions += source_stats.get("meta_model_positions", 0)
            
            class_dist = stats.get("class_distribution", {})
            for class_name, count in class_dist.items():
                class_totals[class_name] = class_totals.get(class_name, 0) + count
            
            base_only_dist = stats.get("base_only_distribution", {})
            for class_name, count in base_only_dist.items():
                base_only_totals[class_name] = base_only_totals.get(class_name, 0) + count
        
        summary = {
            "execution_summary": {
                "total_genes_requested": len(self.config.target_genes),
                "successful_genes": len(successful_genes),
                "failed_genes": len(failed_genes),
                "success_rate": len(successful_genes) / len(self.config.target_genes) * 100,
                "total_processing_time": total_processing_time,
                "average_time_per_gene": total_processing_time / len(self.config.target_genes),
                "workflow_start_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.start_time)),
                "workflow_end_time": time.strftime("%Y-%m-%d %H:%M:%S")
            },
            "prediction_summary": {
                "total_positions_analyzed": total_positions,
                "meta_model_positions": total_meta_positions,
                "base_model_positions": total_positions - total_meta_positions,
                "meta_model_usage_percentage": (total_meta_positions / total_positions * 100) if total_positions > 0 else 0,
                "class_distribution": class_totals,
                "base_only_distribution": base_only_totals
            },
            "configuration": self._config_to_dict(),
            "successful_genes": successful_genes,
            "failed_genes": failed_genes
        }
        
        if failed_genes:
            summary["failure_reasons"] = {
                gene: all_results[gene].get("error", "Unknown error")
                for gene in failed_genes
            }
        
        return summary
    
    def run(self) -> Dict[str, Any]:
        """Run the complete inference workflow."""
        self.start_time = time.time()
        
        try:
            # Create gene manifest
            manifest_path, manifest = self._create_gene_manifest()
            
            # Determine which genes need processing
            genes_to_process = []
            skipped_genes = []
            
            for gene_id in self.config.target_genes:
                if self._check_gene_processed(manifest, gene_id):
                    skipped_genes.append(gene_id)
                    self.logger.info(f"⏭️  Skipping {gene_id} (already processed)")
                else:
                    genes_to_process.append(gene_id)
            
            self.logger.info(f"📊 Processing plan: {len(genes_to_process)} new, {len(skipped_genes)} skipped")
            
            if not genes_to_process:
                self.logger.info("✅ All requested genes already processed!")
                return {"success": True, "message": "All genes already processed"}
            
            # Process genes
            self.logger.info(f"🚀 Starting inference on {len(genes_to_process)} genes...")
            
            gene_results = self._process_genes_parallel(genes_to_process)
            
            # Update manifest with new results
            for gene_id, result in gene_results.items():
                manifest["processed_genes"][gene_id] = result
            
            # Include skipped genes in final results
            all_results = gene_results.copy()
            for gene_id in skipped_genes:
                all_results[gene_id] = manifest["processed_genes"][gene_id]
            
            # Generate summary report
            summary = self._generate_summary_report(all_results)
            manifest["performance_summary"] = summary
            
            # Save manifest
            self._save_gene_manifest(manifest_path, manifest)
            
            # Save summary report
            summary_file = Path(self.config.output_dir) / "inference_summary.json"
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            
            # Performance reporting
            if self.config.performance_reporting:
                self._generate_performance_report(summary)
                
                # Enhanced performance analysis
                try:
                    self.logger.info("📊 Generating enhanced performance analysis...")
                    analyzer = PerformanceAnalyzer(self.config.output_dir, verbose=self.config.verbose >= 1)
                    
                    # Collect all gene predictions and statistics
                    all_predictions = {}
                    all_gene_stats = {}
                    
                    for gene_id, result in all_results.items():
                        if result.get("success", False):
                            # Load predictions for this gene
                            pred_file = Path(self.config.output_dir) / "genes" / gene_id / f"{gene_id}_predictions.parquet"
                            if pred_file.exists():
                                all_predictions[gene_id] = pl.read_parquet(pred_file)
                                all_gene_stats[gene_id] = result.get("statistics", {})
                                
                                # Generate per-gene performance report
                                analyzer.generate_per_gene_report(gene_id, all_predictions[gene_id], all_gene_stats[gene_id])
                    
                    # Generate consolidated report
                    if all_predictions:
                        consolidated_report = analyzer.generate_consolidated_report(all_gene_stats, all_predictions)
                        self.logger.info(f"   ✅ Consolidated report: {consolidated_report}")
                        
                        # Generate ROC/PR curves if we have enough data
                        n_total_positions = sum(len(df) for df in all_predictions.values())
                        if n_total_positions >= 100:
                            # Determine which modes to compare based on actual inference mode
                            comparison_modes = ["base_only"]
                            
                            # Add the actual inference mode that was run
                            if self.config.inference_mode == "hybrid":
                                comparison_modes.append("hybrid")
                            elif self.config.inference_mode == "meta_only":
                                # Only add meta_only if we have meta scores for all positions
                                sample_df = next(iter(all_predictions.values()))
                                if 'donor_meta' in sample_df.columns and 'acceptor_meta' in sample_df.columns:
                                    # Check if meta scores are available for all positions
                                    if not sample_df['donor_meta'].is_null().any():
                                        comparison_modes.append("meta_only")
                            elif self.config.inference_mode == "base_only":
                                # Only compare base_only (no meta model involved)
                                pass  # comparison_modes already has base_only
                            
                            curve_results = analyzer.generate_roc_pr_curves(all_predictions, comparison_modes)
                            self.logger.info(f"   ✅ ROC/PR curves: {curve_results['plot']}")
                            
                            # Log curves to MLflow
                            if self.config.mlflow_enable and MLFLOW_AVAILABLE:
                                mlflow.log_artifact(str(curve_results['plot']), "performance_analysis")
                                mlflow.log_artifact(str(curve_results['metrics']), "performance_analysis")
                        else:
                            self.logger.info(f"   ⚠️  Insufficient data for ROC/PR curves ({n_total_positions} positions)")
                            self.logger.info(f"      Consider running inference on more genes for reliable curves")
                        
                        # Log all performance reports to MLflow
                        if self.config.mlflow_enable and MLFLOW_AVAILABLE:
                            perf_dir = Path(self.config.output_dir) / "performance_analysis"
                            if perf_dir.exists():
                                for report_file in perf_dir.glob("*.txt"):
                                    mlflow.log_artifact(str(report_file), "performance_analysis")
                                for report_file in perf_dir.glob("*.json"):
                                    mlflow.log_artifact(str(report_file), "performance_analysis")
                                for plot_file in perf_dir.glob("*.pdf"):
                                    mlflow.log_artifact(str(plot_file), "performance_analysis")
                                for plot_file in perf_dir.glob("*.png"):
                                    mlflow.log_artifact(str(plot_file), "performance_analysis")
                    
                    self.logger.info("   ✅ Enhanced performance analysis completed")
                    
                except Exception as e:
                    self.logger.warning(f"   ⚠️  Enhanced performance analysis failed: {e}")
                    if self.config.verbose >= 2:
                        import traceback
                        traceback.print_exc()
            
            # Log final summary with proper success/failure determination
            total_time = time.time() - self.start_time
            successful_count = summary['execution_summary']['successful_genes']
            total_count = summary['execution_summary']['total_genes_requested']
            
            # Determine if workflow was truly successful
            workflow_success = successful_count > 0
            
            self.logger.info("=" * 80)
            if workflow_success:
                self.logger.info("🎉 INFERENCE WORKFLOW COMPLETED SUCCESSFULLY")
            else:
                self.logger.error("❌ INFERENCE WORKFLOW FAILED - NO GENES PROCESSED SUCCESSFULLY")
            self.logger.info("=" * 80)
            self.logger.info(f"📊 Total genes processed: {successful_count}/{total_count}")
            self.logger.info(f"⏱️  Total runtime: {format_time(total_time)}")
            self.logger.info(f"📁 Results saved to: {self.config.output_dir}")
            self.logger.info("=" * 80)
            
            # Log final metrics and artifacts to MLflow
            if self.config.mlflow_enable and MLFLOW_AVAILABLE:
                try:
                    # Log summary metrics
                    mlflow.log_metrics({
                        "total_genes_requested": summary['execution_summary']['total_genes_requested'],
                        "successful_genes": summary['execution_summary']['successful_genes'],
                        "failed_genes": summary['execution_summary']['failed_genes'],
                        "success_rate": summary['execution_summary']['success_rate'],
                        "total_positions_analyzed": summary['prediction_summary']['total_positions_analyzed'],
                        "meta_model_positions": summary['prediction_summary']['meta_model_positions'],
                        "meta_model_usage_percentage": summary['prediction_summary']['meta_model_usage_percentage'],
                        "total_processing_time": total_time
                    })
                    
                    # Log class distribution metrics
                    for class_name, count in summary['prediction_summary']['class_distribution'].items():
                        mlflow.log_metric(f"class_{class_name}_count", count)
                    
                    # Log artifacts
                    mlflow.log_artifact(str(manifest_path))
                    mlflow.log_artifact(str(summary_file))
                    if Path(self.config.output_dir, "performance_report.txt").exists():
                        mlflow.log_artifact(str(Path(self.config.output_dir) / "performance_report.txt"))
                    if Path(self.config.output_dir, "inference_workflow.log").exists():
                        mlflow.log_artifact(str(Path(self.config.output_dir) / "inference_workflow.log"))
                    # Optionally log console capture if present
                    console_log_candidates = [
                        Path(self.config.output_dir) / "inference_console.log",
                    ]
                    # Allow override via env var (e.g., SURVEYOR_CONSOLE_LOG=/path/to/tee.log)
                    env_console = os.environ.get("SURVEYOR_CONSOLE_LOG")
                    if env_console:
                        console_log_candidates.append(Path(env_console))
                    for path in console_log_candidates:
                        try:
                            if path and path.exists():
                                mlflow.log_artifact(str(path))
                        except Exception:
                            pass
                    
                    # Log entire output directory structure
                    self._log_directory_tree_to_mlflow()
                    
                    self.logger.info(f"📊 MLflow artifacts logged successfully")
                    
                    # End MLflow run successfully
                    mlflow.end_run(status="FINISHED")
                    self.mlflow_run = None
                    
                except Exception as e:
                    self.logger.warning(f"Failed to log final results to MLflow: {e}")
            
            return {
                "success": workflow_success,
                "summary": summary,
                "manifest_path": str(manifest_path),
                "summary_file": str(summary_file),
                "output_directory": self.config.output_dir,
                "genes_processed": successful_count,
                "genes_total": total_count
            }
            
        except Exception as e:
            total_time = time.time() - self.start_time
            error_msg = f"Workflow failed after {format_time(total_time)}: {str(e)}"
            self.logger.error(error_msg)
            self.logger.debug(f"Full traceback: {traceback.format_exc()}")
            
            # Close MLflow run if active
            if self.config.mlflow_enable and MLFLOW_AVAILABLE and self.mlflow_run:
                try:
                    mlflow.log_metric("workflow_failed", 1)
                    mlflow.end_run(status="FAILED")
                except:
                    pass
            
            return {
                "success": False,
                "error": error_msg,
                "runtime": total_time
            }
    
    def _log_directory_tree_to_mlflow(self):
        """Create and log a directory tree snapshot to MLflow."""
        try:
            tree_file = Path(self.config.output_dir) / "directory_tree.txt"
            with open(tree_file, 'w') as f:
                f.write(f"Directory structure of {self.config.output_dir}\n")
                f.write("=" * 60 + "\n\n")
                
                for root, dirs, files in os.walk(self.config.output_dir):
                    level = root.replace(self.config.output_dir, '').count(os.sep)
                    indent = ' ' * 2 * level
                    f.write(f"{indent}{os.path.basename(root)}/\n")
                    subindent = ' ' * 2 * (level + 1)
                    for file in files[:20]:  # Limit to first 20 files per directory
                        size = os.path.getsize(os.path.join(root, file))
                        size_str = f"{size/1024:.1f}KB" if size < 1024*1024 else f"{size/1024/1024:.1f}MB"
                        f.write(f"{subindent}{file} ({size_str})\n")
                    if len(files) > 20:
                        f.write(f"{subindent}... and {len(files)-20} more files\n")
            
            mlflow.log_artifact(str(tree_file))
        except Exception as e:
            self.logger.debug(f"Failed to create directory tree: {e}")
    
    def __del__(self):
        """Cleanup MLflow run if still active."""
        if hasattr(self, 'config') and self.config.mlflow_enable and MLFLOW_AVAILABLE and hasattr(self, 'mlflow_run') and self.mlflow_run:
            try:
                mlflow.end_run()
            except:
                pass
    
    def _generate_performance_report(self, summary: Dict[str, Any]):
        """Generate detailed performance report."""
        report_file = Path(self.config.output_dir) / "performance_report.txt"
        
        with open(report_file, 'w') as f:
            f.write("📊 SPLICE SURVEYOR - INFERENCE PERFORMANCE REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            # Execution summary
            exec_summary = summary["execution_summary"]
            f.write("🔍 EXECUTION SUMMARY\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total genes requested: {exec_summary['total_genes_requested']}\n")
            f.write(f"Successfully processed: {exec_summary['successful_genes']}\n")
            f.write(f"Failed: {exec_summary['failed_genes']}\n")
            f.write(f"Success rate: {exec_summary['success_rate']:.1f}%\n")
            f.write(f"Total processing time: {format_time(exec_summary['total_processing_time'])}\n")
            f.write(f"Average time per gene: {exec_summary['average_time_per_gene']:.1f}s\n\n")
            
            # Prediction summary
            pred_summary = summary["prediction_summary"]
            f.write("🧬 PREDICTION SUMMARY\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total positions analyzed: {pred_summary['total_positions_analyzed']:,}\n")
            f.write(f"Meta-model positions: {pred_summary['meta_model_positions']:,}\n")
            f.write(f"Base-model positions: {pred_summary['base_model_positions']:,}\n")
            f.write(f"Meta-model usage: {pred_summary['meta_model_usage_percentage']:.1f}%\n\n")
            
            # Class distribution - Before/After comparison
            f.write("📈 CLASS DISTRIBUTION: BEFORE vs AFTER META-MODEL\n")
            f.write("-" * 50 + "\n")
            
            base_only_dist = pred_summary.get("base_only_distribution", {})
            hybrid_dist = pred_summary["class_distribution"]
            
            # Calculate totals for percentage calculations
            base_only_total = sum(base_only_dist.values()) if base_only_dist else 0
            hybrid_total = sum(hybrid_dist.values()) if hybrid_dist else 0
            
            # Get all unique classes from both distributions
            all_classes = set(base_only_dist.keys()) | set(hybrid_dist.keys())
            
            if base_only_dist and hybrid_dist:
                f.write("BEFORE (Base Model Only) → AFTER (With Meta-Model Recalibration):\n\n")
                
                for class_name in sorted(all_classes):
                    base_count = base_only_dist.get(class_name, 0)
                    hybrid_count = hybrid_dist.get(class_name, 0)
                    
                    base_pct = (base_count / base_only_total * 100) if base_only_total > 0 else 0
                    hybrid_pct = (hybrid_count / hybrid_total * 100) if hybrid_total > 0 else 0
                    
                    change = hybrid_count - base_count
                    change_symbol = "📈" if change > 0 else "📉" if change < 0 else "➡️"
                    
                    f.write(f"{class_name.upper():>3}: {base_count:>4} ({base_pct:>5.1f}%) → {hybrid_count:>4} ({hybrid_pct:>5.1f}%) {change_symbol} {change:+d}\n")
                
                # Calculate improvement metrics
                tp_improvement = hybrid_dist.get("TP", 0) - base_only_dist.get("TP", 0)
                tn_improvement = hybrid_dist.get("TN", 0) - base_only_dist.get("TN", 0)
                fp_reduction = base_only_dist.get("FP", 0) - hybrid_dist.get("FP", 0)
                fn_reduction = base_only_dist.get("FN", 0) - hybrid_dist.get("FN", 0)
                
                f.write(f"\n🎯 META-MODEL IMPACT SUMMARY:\n")
                if tp_improvement > 0:
                    f.write(f"   • True Positives gained: +{tp_improvement}\n")
                if tn_improvement > 0:
                    f.write(f"   • True Negatives gained: +{tn_improvement}\n")
                if fp_reduction > 0:
                    f.write(f"   • False Positives reduced: -{fp_reduction}\n")
                if fn_reduction > 0:
                    f.write(f"   • False Negatives reduced: -{fn_reduction}\n")
                
                net_improvement = tp_improvement + tn_improvement + fp_reduction + fn_reduction
                f.write(f"   • Net improvement: {net_improvement:+d} positions\n")
                
            else:
                # Fallback to single distribution if comparison not available
                f.write("Current Classification (Hybrid Predictions):\n")
                for class_name, count in hybrid_dist.items():
                    percentage = (count / hybrid_total * 100) if hybrid_total > 0 else 0
                    f.write(f"{class_name.upper()}: {count:,} ({percentage:.1f}%)\n")
            
            f.write(f"\n📁 Detailed results available in: {self.config.output_dir}\n")
        
        self.logger.info(f"📋 Performance report saved: {report_file}")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Practical Main-Entry Meta-Model Inference Workflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:

# Basic inference on specific genes
python main_inference_workflow.py \\
    --model results/gene_cv_pc_1000_3mers_run_4/model_multiclass.pkl \\
    --training-dataset train_pc_1000_3mers \\
    --genes ENSG00000104435,ENSG00000006420 \\
    --output-dir /tmp/inference_demo

# Inference with genes from file and custom thresholds
python main_inference_workflow.py \\
    --model results/gene_cv_pc_1000_3mers_run_4/model_multiclass.pkl \\
    --training-dataset train_pc_1000_3mers \\
    --genes-file top_error_prone_genes.txt \\
    --uncertainty-low 0.01 \\
    --uncertainty-high 0.85 \\
    --complete-coverage \\
    --output-dir ./detailed_inference

# High-performance parallel processing
python main_inference_workflow.py \\
    --model results/gene_cv_pc_1000_3mers_run_4/model_multiclass.pkl \\
    --training-dataset train_pc_1000_3mers \\
    --genes-file large_gene_list.txt \\
    --parallel-workers 4 \\
    --strategy UNCERTAINTY_FOCUSED \\
    --output-dir ./parallel_inference

# Run as module from project root
python -m meta_spliceai.splice_engine.meta_models.workflows.inference.main_inference_workflow \\
    --model results/gene_cv_pc_1000_3mers_run_4/model_multiclass.pkl \\
    --training-dataset train_pc_1000_3mers \\
    --genes ENSG00000104435 \\
    --output-dir inference_results

NOTE: Paths like "results/..." are relative to project root.
      Use absolute paths if running from different directories.
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--model", "--model-path",
        required=True,
        help=(
            "Path to pre-trained meta-model file or directory containing it. "
            "If a directory is provided, the workflow will attempt to select the "
            "best candidate (e.g., model_multiclass.pkl)."
        )
    )

    parser.add_argument(
        "--model-keywords",
        nargs="+",
        help=(
            "Ordered keywords (by precedence) used to select a model file when --model is a directory. "
            "Defaults to: multiclass model best final"
        )
    )
    
    # Gene specification (mutually exclusive groups would be ideal, but we'll validate manually)
    parser.add_argument(
        "--genes",
        help="Comma-separated list of gene IDs to analyze"
    )
    
    parser.add_argument(
        "--genes-file",
        help="Path to file containing gene IDs (one per line)"
    )
    
    # Optional arguments
    parser.add_argument(
        "--training-dataset", "--training-dataset-path",
        help="Path to original training dataset directory"
    )
    
    parser.add_argument(
        "--output-dir",
        default="./inference_results",
        help="Output directory for predictions and artifacts (default: ./inference_results)"
    )

    # Artifact preservation
    parser.add_argument(
        "--keep-artifacts-dir",
        help="Directory to preserve intermediate artifacts and assembled test data (optional)"
    )
    parser.add_argument(
        "--preserve-artifacts",
        action="store_true",
        help="Preserve intermediate artifacts (alias to enabling --keep-artifacts-dir when provided)"
    )
    
    # MLflow tracking arguments
    parser.add_argument(
        "--mlflow-enable",
        action="store_true",
        help="Enable MLflow experiment tracking"
    )
    
    parser.add_argument(
        "--mlflow-experiment",
        default="surveyor-inference",
        help="MLflow experiment name (default: surveyor-inference)"
    )
    
    parser.add_argument(
        "--mlflow-tracking-uri",
        help="MLflow tracking server URI (optional, defaults to local mlruns)"
    )
    
    parser.add_argument(
        "--mlflow-tags",
        nargs="+",
        help="MLflow tags as key=value pairs (e.g., scenario=training_gene model_version=v1)"
    )
    
    # Uncertainty thresholds
    parser.add_argument(
        "--uncertainty-low",
        type=float,
        default=0.02,
        help="Lower uncertainty threshold for selective processing (default: 0.02)"
    )
    
    parser.add_argument(
        "--uncertainty-high",
        type=float,
        default=0.80,
        help="Upper uncertainty threshold for selective processing (default: 0.80)"
    )
    
    # Enhanced uncertainty selection parameters
    parser.add_argument(
        "--target-meta-rate",
        type=float,
        default=0.10,
        help="Target percentage of positions for meta-model inference (default: 0.10 = 10%%)"
    )
    
    parser.add_argument(
        "--uncertainty-strategy",
        type=str,
        default="hybrid_entropy",
        choices=["confidence_only", "entropy_only", "hybrid_entropy"],
        help="Uncertainty selection strategy (default: hybrid_entropy)"
    )
    
    parser.add_argument(
        "--disable-adaptive-tuning",
        action="store_true",
        help="Disable adaptive threshold tuning for target selection rate"
    )
    
    # Processing options
    parser.add_argument(
        "--max-positions",
        type=int,
        default=10000,
        help="Maximum positions to process per gene (default: 10000)"
    )
    
    parser.add_argument(
        "--complete-coverage",
        action="store_true",
        help="Generate predictions for ALL positions (slower but comprehensive)"
    )
    
    parser.add_argument(
        "--strategy",
        choices=["SELECTIVE", "COMPLETE", "TRAINING_GAPS", "UNCERTAINTY_FOCUSED"],
        default="SELECTIVE",
        help="Position selection strategy (default: SELECTIVE)"
    )
    
    parser.add_argument(
        "--inference-mode",
        choices=["base_only", "hybrid", "meta_only"],
        default="hybrid",
        help="Inference strategy: base_only (fast), hybrid (default, balanced), meta_only (comprehensive but memory-intensive)"
    )
    
    parser.add_argument(
        "--enable-chunked-processing",
        action="store_true",
        help="Enable chunked processing for true full coverage in meta_only mode (processes ALL positions)"
    )
    
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=10000,
        help="Size of each processing chunk when chunked processing is enabled (default: 10000)"
    )
    
    # Performance options
    parser.add_argument(
        "--parallel-workers",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1)"
    )
    
    parser.add_argument(
        "--no-cleanup",
        action="store_true",
        help="Don't clean up intermediate files"
    )
    
    parser.add_argument(
        "--no-performance-report",
        action="store_true",
        help="Don't generate performance reports"
    )
    
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help="Force recomputation even if results exist"
    )
    
    # Verbosity
    parser.add_argument(
        "--verbose", "-v",
        action="count",
        default=1,
        help="Increase verbosity (use -v, -vv, or -vvv)"
    )
    
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Minimal output (overrides --verbose)"
    )
    
    return parser.parse_args()


def main():
    """Main entry point for the inference workflow."""
    args = parse_arguments()
    
    # Handle verbosity
    if args.quiet:
        verbosity = 0
    else:
        verbosity = args.verbose
    
    # Parse genes
    target_genes = []
    if args.genes:
        target_genes = [gene.strip() for gene in args.genes.split(",") if gene.strip()]
    
    # Validate gene inputs
    if not target_genes and not args.genes_file:
        print("❌ Error: Must specify either --genes or --genes-file")
        sys.exit(1)
    
    # Parse MLflow tags
    mlflow_tags = {}
    if args.mlflow_tags:
        for tag in args.mlflow_tags:
            if '=' in tag:
                key, value = tag.split('=', 1)
                mlflow_tags[key] = value
            else:
                print(f"⚠️  Warning: Invalid MLflow tag format '{tag}', expected key=value")
    
    try:
        # Create configuration
        config = InferenceWorkflowConfig(
            model_path=args.model,
            model_keywords=args.model_keywords,
            training_dataset_path=args.training_dataset,
            target_genes=target_genes,
            genes_file=args.genes_file,
            output_dir=args.output_dir,
            uncertainty_threshold_low=args.uncertainty_low,
            uncertainty_threshold_high=args.uncertainty_high,
            target_meta_selection_rate=args.target_meta_rate,
            uncertainty_selection_strategy=args.uncertainty_strategy,
            enable_adaptive_uncertainty_tuning=not args.disable_adaptive_tuning,
            max_positions_per_gene=args.max_positions,
            complete_coverage=args.complete_coverage,
            enable_chunked_processing=args.enable_chunked_processing,
            chunk_size=args.chunk_size,
            selective_strategy=args.strategy,
            parallel_workers=args.parallel_workers,
            verbose=verbosity,
            cleanup_intermediates=not args.no_cleanup,
            performance_reporting=not args.no_performance_report,
            force_recompute=args.force_recompute,
            inference_mode=args.inference_mode,
            keep_artifacts_dir=args.keep_artifacts_dir,
            preserve_artifacts=args.preserve_artifacts,

            # MLflow tracking
            mlflow_enable=args.mlflow_enable,
            mlflow_experiment=args.mlflow_experiment,
            mlflow_tracking_uri=args.mlflow_tracking_uri,
            mlflow_tags=mlflow_tags
        )
        
        # Run workflow
        runner = InferenceWorkflowRunner(config)
        result = runner.run()
        
        if result["success"]:
            genes_processed = result.get("genes_processed", 0)
            genes_total = result.get("genes_total", 0)
            print(f"✅ Inference workflow completed successfully!")
            print(f"📊 Successfully processed {genes_processed}/{genes_total} genes")
            if "output_directory" in result:
                print(f"📁 Results available in: {result['output_directory']}")
            else:
                print(f"📁 Results available in: {args.output_dir}")
            sys.exit(0)
        else:
            # Check if this was a gene processing failure vs a fatal error
            if "genes_processed" in result:
                genes_processed = result.get("genes_processed", 0)
                genes_total = result.get("genes_total", 0)
                print(f"❌ Inference workflow failed: No genes processed successfully ({genes_processed}/{genes_total})")
                print(f"📁 Partial results and error logs available in: {result.get('output_directory', args.output_dir)}")
            else:
                print(f"❌ Inference workflow failed: {result['error']}")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        if verbosity >= 2:
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()