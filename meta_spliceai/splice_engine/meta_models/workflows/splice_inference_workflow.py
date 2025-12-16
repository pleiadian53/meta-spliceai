"""Splice-inference workflow
=================================
A lightweight variant of :pyfunc:`run_enhanced_splice_prediction_workflow` that
runs the base SpliceAI model **only** to obtain per-nucleotide scores required
for meta-model *inference*.

Differences from the full prediction workflow
---------------------------------------------
1.  Optionally skips heavy preprocessing steps (annotation extraction,
    sequence FASTA dumps, overlapping-gene analysis) – we merely *verify* that
    their artefacts already exist when they are disabled.
2.  Can accept a *covered-position* dictionary so that downstream callers can
    decide which positions are "unseen" and therefore need to be evaluated by
    the meta-model.
    (Filtering itself will be implemented in a follow-up patch.)
3.  Supports automatic clean-up of bulky per-position artefacts once the meta
    scores have been produced.
"""
from __future__ import annotations

import os
import shutil
import json as _json
import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union, Any

import polars as pl
import pandas as pd
import math
import time
import numpy as np
import pickle

from meta_spliceai.splice_engine.meta_models.workflows.inference_workflow_utils import (
    validate_score_columns,
    load_model_with_calibration,
    get_model_info,
    perform_neighborhood_analysis,
    standardize_label_encoding,
    diagnostic_sampling
)
from meta_spliceai.splice_engine.meta_models.workflows.splice_prediction_workflow import (
    run_enhanced_splice_prediction_workflow,
)
from meta_spliceai.splice_engine.meta_models.constants import EXPECTED_MIN_COLUMNS

# ---------------------------------------------------------------------------
# Helper – load canonical feature column list from a training directory
# ---------------------------------------------------------------------------
def _load_training_schema(train_dir: Path) -> List[str]:
    """Return list of feature column names stored by a finished training run.

    Search order (first hit wins):
    1. ``train_dir / "feature_manifest.csv"``
    2. ``train_dir / "columns.json"``
    3. ``train_dir / "features" / "meta" / "columns.json"``
    4. Arrow schema of the Parquet dataset at ``train_dir / "master"``
    5. Header of ``train_dir / "column_stats.txt"`` (tab-separated)
    """
    # 0 - Feature manifest CSV (preferred method)
    manifest_path = train_dir / "feature_manifest.csv"
    if manifest_path.exists():
        try:
            import pandas as pd
            manifest_df = pd.read_csv(manifest_path)
            if "feature_name" in manifest_df.columns:
                return manifest_df["feature_name"].tolist()
        except Exception as e:
            print(f"[warning] Failed to read feature manifest: {e}")
    
    # 1 & 2 – JSON file
    for rel in ["columns.json", "features/meta/columns.json"]:
        p = train_dir / rel
        if p.exists():
            with open(p) as fh:
                cols = _json.load(fh)
            if isinstance(cols, list) and all(isinstance(c, str) for c in cols):
                return cols
    # 3 – Parquet dataset
    try:
        import pyarrow.dataset as ds  # type: ignore
        # Check master subdirectory first
        master_dir = train_dir / "master"
        if master_dir.exists():
            return ds.dataset(master_dir).schema.names
        # Check train_dir directly if master doesn't exist
        elif train_dir.exists():
            parquet_files = list(train_dir.glob("*.parquet"))
            if parquet_files:
                return ds.dataset(train_dir).schema.names
    except Exception:
        pass  # fall through
    # 4 – column_stats.txt header
    stats_path = train_dir / "column_stats.txt"
    if stats_path.exists():
        with open(stats_path) as fh:
            header = fh.readline().rstrip("\n").split("\t")
        return header
    raise FileNotFoundError(
        f"Could not determine feature columns – checked feature_manifest.csv, JSON, Parquet, and column_stats.txt under {train_dir}"
    )

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_enhanced_splice_inference_workflow(
    *,
    covered_pos: Dict[str, Set[int]] | None = None,
    t_low: float = 0.02,
    t_high: float = 0.80,
    target_genes: List[str] | None = None,
    do_prepare_sequences: bool = False,
    do_prepare_position_tables: bool = False,
    do_prepare_feature_matrices: bool = False,
    do_prepare_annotations: bool = False,
    do_handle_overlaps: bool = False,
    # House-keeping ----------------------------------------------------------
    cleanup: bool = True,
    cleanup_features: bool = False,  # Whether to remove feature matrices after prediction
    verbosity: int = 1,
    max_analysis_rows: int = 500000,
    max_positions_per_gene: int = 0,
    no_final_aggregate: bool = True,
    # Model and prediction options -------------------------------------------
    model_path: Optional[Path] = None,
    use_calibration: bool = True,
    # Analysis options -------------------------------------------------------
    neigh_sample: int = 0,
    neigh_window: int = 50,
    diag_sample: int = 0,
    # Directory options ------------------------------------------------------
    output_dir: Optional[str | Path] = None,
    feature_dir: Optional[str | Path] = None,  # Custom directory for feature matrices
    train_schema_dir: Optional[str | Path] = None,  # Training schema directory for feature consistency
    **kwargs,
) -> Dict[str, Any]:
    """Run the SpliceAI base model for inference-time feature generation.

    Parameters
    ----------
    covered_pos
        Mapping ``gene_id -> set(relative_position)`` denoting positions that
        were *already* covered in the meta-model's training matrix.  This
        object is *not* used directly in this first iteration but is accepted
        now so that the caller API is future-proof: a subsequent patch will
        use it to filter the per-nucleotide DataFrame before serialisation.
    t_low, t_high
        Placeholder thresholds for the "ambiguous" score zone.  Not yet used.
    target_genes
        If provided, restrict inference to this subset – keeps run-time down.
    do_prepare_* flags
        When *False* (default) the corresponding heavyweight preparation step
        is skipped: these flags are designed to avoid unnecessarily repeating
        expensive computations when this inference workflow is called multiple
        times across different meta-models.
    model_path
        Path to the model file to use for predictions. Can be either an XGBoost
        model (.json) or a pickled calibrated ensemble model (.pkl).
    use_calibration
        Whether to use calibration when available in the model. If False and a
        calibrated model is provided, will extract the base model.
    neigh_sample
        Number of positions to sample for neighborhood analysis. If 0, no
        neighborhood analysis is performed.
    neigh_window
        Window size (in nucleotides) around each sampled position for
        neighborhood analysis.
    diag_sample
        Number of positions to sample for detailed diagnostics. If 0, no
        diagnostic sampling is performed.
    cleanup
        Delete bulky *analysis_sequences* / *splice_positions_enhanced* files
        after the workflow completes. Set to *False* when running an
        evaluation where you need to inspect the raw artefacts.
    cleanup_features
        Delete feature matrix files after prediction and analysis. Set to *True*
        to prevent accumulation of large feature files that are no longer needed.
        Default is *False* to preserve feature files for further analysis.
    verbosity
        0 = silent   1 = normal   2 = verbose.
    output_dir
        Custom output directory for workflow results. If None, a default
        directory will be used.
    feature_dir
        Custom directory for feature matrices. If None, they will be stored in
        the output directory under 'master'. Using a custom directory can help
        organize features and prevent large files from accumulating in unwanted locations.
    train_schema_dir
        Directory containing training schema information for feature consistency.
        If provided, will load feature column names from this directory.

    Returns
    -------
    A dictionary containing:
        - 'success': Boolean indicating whether the workflow completed successfully
        - 'paths': Dictionary of important file paths generated during workflow
        - 'model_info': Information about the model used for prediction
        - 'neighborhood_results': Results of neighborhood analysis if performed
        - 'diagnostic_results': Results of diagnostic sampling if performed
    """
    
    # CRITICAL DEBUG: Check if this function is being called at all
    if verbosity >= 1:
        print("🔥🔥🔥 [CRITICAL] run_enhanced_splice_inference_workflow FUNCTION CALLED! 🔥🔥🔥")
        print(f"🔥🔥🔥 [CRITICAL] Target genes: {target_genes}")
        print(f"🔥🔥🔥 [CRITICAL] Output dir: {output_dir}")
        print(f"🔥🔥🔥 [CRITICAL] train_schema_dir: {train_schema_dir}")
    
    # ------------------------------------------------------------------
    # Handle output directories ----------------------------------------
    # ------------------------------------------------------------------
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if output_dir is None:
        output_dir = Path(f"inference_workflow_output_{timestamp}")
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up feature directory
    if feature_dir is None:
        # Use default location in output directory
        feature_base_dir = output_dir / "features"
    else:
        # Use custom feature directory
        feature_base_dir = Path(feature_dir)
    
    feature_base_dir.mkdir(parents=True, exist_ok=True)
    
    # Create model-specific feature directory
    if model_path is not None:
        model_name = Path(model_path).stem
        master_dataset_dir = feature_base_dir / f"features_{model_name}_{timestamp}"
    else:
        master_dataset_dir = feature_base_dir / f"features_{timestamp}"
    
    master_dataset_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a file to track feature directory origin
    with open(master_dataset_dir / "feature_info.txt", "w") as f:
        f.write(f"Created: {datetime.datetime.now().isoformat()}\n")
        f.write(f"Source: {output_dir}\n")
        if model_path is not None:
            f.write(f"Model: {model_path}\n")
        f.write(f"Command: inference workflow\n")
    
    # ------------------------------------------------------------------
    # Delegate to the existing enhanced prediction workflow -------------
    # ------------------------------------------------------------------
    _wk = dict(kwargs)  # make a copy
    _wk.update(
        {
            # Pre-processing switches map directly onto SpliceAIConfig fields
            # Note: Only include parameters that exist in SpliceAIConfig dataclass
            # For inference, we should use pre-computed resources, not re-extract everything
            "do_extract_annotations": False,  # Use existing annotations.db and splice_sites.tsv
            "do_extract_sequences": False,    # Use existing gene_sequence_*.parquet files
            "do_extract_splice_sites": False, # Use existing splice_sites.tsv
            "do_find_overlaping_genes": False, # Use existing overlapping gene data
            
            # Gene filter
            "target_genes": target_genes,
            # Use a dedicated subdirectory so inference artefacts never collide
            "eval_dir": str(output_dir),  # SpliceAIConfig uses eval_dir, not output_dir
            # Disable memory-heavy aggregation inside prediction workflow unless caller overrides
            "no_final_aggregate": _wk.get("no_final_aggregate", no_final_aggregate),
        }
    )
    
    # Store custom parameters that don't belong to SpliceAIConfig separately
    custom_params = {
        "do_extract_position_tables": do_prepare_position_tables,
        "do_extract_feature_matrices": do_prepare_feature_matrices,
        "do_handle_overlaps": do_handle_overlaps,
        "spliceai_args": {
            "no_chunk": True,  # Faster processing
            "prefix": "sai",
        },
    }
    
    if verbosity >= 2:
        print(f"[inference-workflow] Custom parameters (not passed to SpliceAIConfig): {custom_params}")
    
    if verbosity >= 1:
        print(
            f"[inference-workflow] Starting enhanced splice prediction workflow "
            f"for {len(target_genes) if target_genes else 'all'} genes …"
        )

    # For inference, skip the heavyweight prediction workflow and directly use pre-computed data
    # This avoids redundant preprocessing steps that take 10+ minutes
    if verbosity >= 1:
        print("[inference-workflow] Using streamlined approach with pre-computed resources")
        print(f"[inference-workflow] 🚀 CHECKPOINT 1: Starting streamlined approach")
    
    # Load existing splice sites data directly
    from meta_spliceai.splice_engine.utils_fs import read_splice_sites
    from meta_spliceai.splice_engine.meta_models.utils.sequence_utils import scan_chromosome_sequence
    from meta_spliceai.splice_engine.meta_models.io.handlers import MetaModelDataHandler
    
    if verbosity >= 1:
        print(f"[inference-workflow] 🚀 CHECKPOINT 2: Imports completed")
    
    # Set up data handler (kept for compatibility)
    data_handler = MetaModelDataHandler(str(output_dir), separator="\t")
    
    if verbosity >= 1:
        print(f"[inference-workflow] 🚀 CHECKPOINT 3: Data handler created")
    
    # Load splice sites directly from existing file
    if verbosity >= 1:
        print(f"[inference-workflow] Loading splice sites from pre-computed data")
        print(f"[inference-workflow] 🚀 CHECKPOINT 4: About to start try block")
    

    try:
        # CORRECTED APPROACH: Load existing analysis_sequences artifacts directly
        # This is much more efficient than recreating them from training data
        if verbosity >= 1:
            print(f"[inference-workflow] 🎯 Loading existing analysis_sequences artifacts for target genes")
        
        # Step 1: Find chromosomes for target genes using gene_features.tsv
        import polars as pl_local
        
        # Use systematic data resource management
        from meta_spliceai.splice_engine.meta_models.workflows.inference.data_resource_manager import create_inference_data_manager
        data_manager = create_inference_data_manager()
        gene_features_path = data_manager.get_gene_features_path()
        
        if verbosity >= 1:
            print(f"[inference-workflow] 📊 Looking up gene chromosomes in {gene_features_path}")
            
        # Fix: Specify that 'chrom' column should be read as string, not int
        gene_features_df = pl_local.read_csv(
            str(gene_features_path), 
            separator="\t",
            schema_overrides={"chrom": pl_local.Utf8}  # Force chrom to be string to handle X/Y chromosomes
        )
        
        # Find chromosomes for target genes
        target_gene_info = gene_features_df.filter(
            pl_local.col("gene_id").is_in(target_genes)
        ).select(["gene_id", "chrom", "gene_name"])
        
        if target_gene_info.is_empty():
            raise RuntimeError(f"Target genes not found in gene_features.tsv: {target_genes}")
            
        target_chromosomes = target_gene_info["chrom"].unique().to_list()
        
        if verbosity >= 1:
            gene_chrom_info = target_gene_info.to_dict(as_series=False)
            for i in range(len(gene_chrom_info["gene_id"])):
                gene_id = gene_chrom_info["gene_id"][i]
                chrom = gene_chrom_info["chrom"][i]
                gene_name = gene_chrom_info["gene_name"][i]
                print(f"[inference-workflow] 🧬 {gene_id} ({gene_name}) is on chromosome {chrom}")
            print(f"[inference-workflow] 📍 Target chromosomes: {target_chromosomes}")
        
        # Step 2: Load analysis_sequences from chromosome-specific artifact files
        meta_models_dir = data_manager.directories["meta_models"]  # Use systematic path
        analysis_sequences_list = []
        
        for chrom in target_chromosomes:
            chrom_files = list(meta_models_dir.glob(f"analysis_sequences_{chrom}_*.tsv"))
            if verbosity >= 1:
                print(f"[inference-workflow] 🔍 Found {len(chrom_files)} analysis_sequences files for chromosome {chrom}")
            
            for file_path in chrom_files:
                if verbosity >= 2:
                    print(f"[inference-workflow] 📄 Loading {file_path.name}")
                    
                # Load and filter to target genes
                df_chunk = pl_local.read_csv(file_path, separator="\t")
                if "gene_id" in df_chunk.columns:
                    gene_matches = df_chunk.filter(pl_local.col("gene_id").is_in(target_genes))
                    if len(gene_matches) > 0:
                        analysis_sequences_list.append(gene_matches)
                        if verbosity >= 1:
                            print(f"[inference-workflow] ✅ Found {len(gene_matches)} rows for target genes in {file_path.name}")
        
        if not analysis_sequences_list:
            if verbosity >= 1:
                print(f"[inference-workflow] ⚠️ No pre-computed analysis_sequences data found for target genes")
                print(f"[inference-workflow] 🧬 Target genes: {target_genes}")
                print(f"[inference-workflow] 📍 Target chromosomes: {target_chromosomes}")
                print(f"[inference-workflow] 📁 Searched in: {meta_models_dir}")
                print(f"[inference-workflow] 💡 This is expected for unseen genes that haven't been processed yet")
                print(f"[inference-workflow] 🔄 Will fall back to full prediction workflow to generate data")
            raise RuntimeError(f"No analysis_sequences data found for target genes {target_genes} in chromosomes {target_chromosomes}")
        
        # Step 3: Combine all matching data
        analysis_sequences_df = pl_local.concat(analysis_sequences_list)
        
        if verbosity >= 1:
            print(f"[inference-workflow] ✅ Loaded {len(analysis_sequences_df)} existing analysis_sequences rows for target genes")
            print(f"[inference-workflow] 🚀 Skipped chromosome processing for {35 - len(target_chromosomes)} irrelevant chromosomes!")
        
        # Step 4: Save filtered results to output directory
        os.makedirs(output_dir, exist_ok=True)
        analysis_sequences_file = Path(output_dir) / "analysis_sequences_filtered.tsv"
        
        analysis_sequences_df.write_csv(str(analysis_sequences_file), separator="\t")
        if verbosity >= 1:
            print(f"[inference-workflow] 💾 Saved {len(analysis_sequences_df)} analysis sequences to {analysis_sequences_file}")
        
        # Create minimal result structure to satisfy downstream code
        # Since we're using pre-computed data, create an empty features directory
        features_dir = Path(output_dir) / "features"
        features_dir.mkdir(exist_ok=True)
        
        result = {
            "success": True,
            "paths": {
                "eval_dir": str(output_dir),
                "output_dir": str(output_dir),
                "feature_dir": str(features_dir),
                "analysis_sequences": [str(analysis_sequences_file)]
            },
            "splice_sites_df": analysis_sequences_df,  # Use the loaded analysis_sequences
            "error_analysis": pl_local.DataFrame(),
            "positions": pl_local.DataFrame(),
            "analysis_sequences": analysis_sequences_df
        }
        
        if verbosity >= 1:
            print(f"[inference-workflow] ✅ Streamlined workflow completed successfully using pre-computed data")
            print(f"[inference-workflow] 🚀 Skipped redundant preprocessing - saved ~10+ minutes!")
        
        # Return the streamlined result immediately
        return result
            
    except Exception as e:
        if verbosity >= 1:
            print(f"[inference-workflow] ❌ Streamlined approach failed: {e}")
            print(f"[inference-workflow] 🐛 Exception type: {type(e).__name__}")
            
            # Provide specific guidance based on the error
            if "No analysis_sequences data found" in str(e):
                print("[inference-workflow] 💡 REASON: Target genes don't have pre-computed analysis_sequences data")
                print("[inference-workflow] 📝 This is normal for unseen genes that haven't been processed before")
                print("[inference-workflow] 🔄 SOLUTION: Falling back to full prediction workflow to generate the data")
                print("[inference-workflow] ⏱️  NOTE: This will take longer (~1-2 minutes per gene) but will work correctly")
            else:
                print("[inference-workflow] 💡 This may indicate missing pre-computed data or configuration issues")
            
            print("[inference-workflow] ⚠️  Falling back to full prediction workflow (this will take longer)")
            print("[inference-workflow] 📊 Note: This fallback will process relevant chromosomes and generate missing data")
        
        # Add detailed traceback for debugging
        import traceback
        if verbosity >= 2:
            print(f"[inference-workflow] 🔍 Full traceback:")
            traceback.print_exc()
        
        # Fallback to the original approach if streamlined fails
        # Use systematic data manager to determine target chromosomes and optimize fallback
        target_chromosomes = None
        skip_overlapping_genes = False
        
        if target_genes:
            try:
                # Use the data manager we already created
                validation = data_manager.validate_inference_requirements(target_genes)
                target_chromosomes = validation.get("target_chromosomes", [])
                
                # If we have overlapping genes data available, skip processing
                if validation.get("has_overlapping_genes", False):
                    skip_overlapping_genes = True
                    if verbosity >= 1:
                        print(f"[inference-workflow] 🎯 Fallback will skip overlapping gene processing (data available)")
                
                if target_chromosomes:
                    if verbosity >= 1:
                        print(f"[inference-workflow] 🎯 Fallback will process only chromosomes: {target_chromosomes}")

            except Exception as chrom_lookup_error:
                if verbosity >= 1:
                    print(f"[inference-workflow] ⚠️ Could not determine target chromosomes: {chrom_lookup_error}")
                    print("[inference-workflow] ⚠️ Fallback will process all chromosomes")

        # Prepare fallback configuration with optimizations
        fallback_config = _wk.copy()
        fallback_config.update({
            "verbosity": max(0, verbosity - 1),
            "target_chromosomes": target_chromosomes,
            "do_find_overlaping_genes": not skip_overlapping_genes,  # Skip if data available
            # Use existing genomic data to avoid reprocessing
            "local_dir": "data/ensembl",
        })
        
        # Remove parameters that SpliceAIConfig doesn't accept
        invalid_params = ["force_overwrite", "auto_detect"]
        for param in invalid_params:
            fallback_config.pop(param, None)
        
        result = run_enhanced_splice_prediction_workflow(**fallback_config)
        if not result.get("success", False):
            raise RuntimeError("Enhanced splice prediction workflow failed; cannot continue.")
        
    # Get artefact directory from prediction workflow result
    artefact_dir = Path(result["paths"].get("output_dir"))
    if not artefact_dir.exists():
        # Create the directory if it doesn't exist (for streamlined approach)
        artefact_dir.mkdir(parents=True, exist_ok=True)
        if verbosity >= 1:
            print(f"[inference-workflow] Created output directory: {artefact_dir}")
    
    if not artefact_dir.is_dir():
        raise RuntimeError("No output directory provided by subprocess.")

    # ------------------------------------------------------------------
    # Build filtered *analysis_sequences* for unseen & ambiguous positions
    # ------------------------------------------------------------------
    if verbosity >= 1:
        print(f"[inference-workflow] Loading analysis_sequences chunk files (cap {max_analysis_rows:,} rows)…")
    analysis_df: pl.DataFrame = result.get("analysis_sequences", pl.DataFrame())
    if analysis_df.is_empty():
        # Fallback: load analysis_sequences files from disk (chunk-level TSVs)
        # Use systematic data manager to find the correct directory
        global_artifacts_dir = data_manager.directories["meta_models"]
        tsv_paths = []
        
        # Check local artifact directory first (for newly generated files)
        tsv_paths = [p for p in artefact_dir.glob("analysis_sequences_*.*") 
                    if p.suffix in {".tsv", ".tsv.gz", ".parquet"} and "inference" not in p.name]
        if tsv_paths and verbosity >= 1:
            print(f"[inference-workflow] Found {len(tsv_paths)} analysis_sequences files in local directory: {artefact_dir}")
        
        # Also check meta_models subdirectory of local artifact directory
        if not tsv_paths:
            meta_models_subdir = artefact_dir / "meta_models"
            if meta_models_subdir.exists():
                tsv_paths = [p for p in meta_models_subdir.glob("analysis_sequences_*.*") 
                            if p.suffix in {".tsv", ".tsv.gz", ".parquet"} and "inference" not in p.name]
                if tsv_paths and verbosity >= 1:
                    print(f"[inference-workflow] Found {len(tsv_paths)} analysis_sequences files in local meta_models: {meta_models_subdir}")
        
        # If not found locally, check global directory
        if not tsv_paths and global_artifacts_dir.exists():
            tsv_paths = [p for p in global_artifacts_dir.glob("analysis_sequences_*.*") 
                        if p.suffix in {".tsv", ".tsv.gz", ".parquet"} and "inference" not in p.name]
            if tsv_paths and verbosity >= 1:
                print(f"[inference-workflow] Found {len(tsv_paths)} analysis_sequences files in global directory: {global_artifacts_dir}")
        
        if not tsv_paths:
            raise RuntimeError("No analysis_sequences files available for filtering – cannot proceed. "
                             f"Checked: {global_artifacts_dir} and {artefact_dir}")
        else: 
            if verbosity >= 1:
                source_dir = global_artifacts_dir if global_artifacts_dir.exists() and any(p.parent == global_artifacts_dir for p in tsv_paths) else artefact_dir
                print(f"[inference-workflow] analysis_sequences in RAM is empty; loading {len(tsv_paths)} chunk files from disk from\n{source_dir}\n ...")
        
        # ---- Streaming load & on-the-fly sampling to avoid OOM --------------
        required_cols = set(EXPECTED_MIN_COLUMNS)
        ln2 = math.log(2.0)
        eps = 1e-9
        selected_rows: list[pl.DataFrame] = []  # final tiny collection
        per_gene_buffers: Dict[str, list] = {}

        def _update_buffer(gid: str, df_sub: pl.DataFrame):
            """Keep at most *max_positions_per_gene* most-uncertain rows per gene."""
            if max_positions_per_gene <= 0:
                # No limit ⇒ just append
                per_gene_buffers.setdefault(gid, []).append(df_sub)
                return
            existing = per_gene_buffers.get(gid)
            if existing is None:
                per_gene_buffers[gid] = [df_sub]
            else:
                existing.append(df_sub)
            # Concatenate and keep top N
            df_cat = pl.concat(per_gene_buffers[gid]) # stacks rows (concatenates vertically)
            df_cat = (
                df_cat.sort(["_neg_entropy", "_dist_thr"])  # already have helper cols
                        .head(max_positions_per_gene)
            )
            per_gene_buffers[gid] = [df_cat]

        for p in tsv_paths:
            if verbosity >= 2:
                print(f"[stream] Processing {p.name} …")
            if p.suffix == ".parquet":
                df_p = pl.read_parquet(p)
                # Select only required columns after loading
                df_p = df_p.select([col for col in required_cols if col in df_p.columns])
            else:
                # Load all columns first, then select required ones
                df_p = pl.read_csv(p, separator="\t", infer_schema_length=100)
                # Select only required columns that exist
                available_required_cols = [col for col in required_cols if col in df_p.columns]
                if len(available_required_cols) < len(required_cols):
                    missing_cols = [col for col in required_cols if col not in df_p.columns]
                    if verbosity >= 1:
                        print(f"[stream] ⚠️ Missing columns in {p.name}: {missing_cols}")
                df_p = df_p.select(available_required_cols)
                
                # Cast score columns to float in case the CSV reader inferred them as Utf8
                df_p = df_p.with_columns(
                    pl.col(["donor_score", "acceptor_score", "neither_score"]).cast(pl.Float64)
                )
                validate_score_columns(df_p, p, verbose=verbosity)

            # Optional gene filter ---------------------------------------------------
            if target_genes:
                df_p = df_p.filter(pl.col("gene_id").is_in(target_genes))

            if df_p.is_empty():
                continue
            # Compute helpers
            df_p = (
                df_p.with_columns([
                    pl.max_horizontal("donor_score", "acceptor_score").alias("_max_score"),
                ]).filter((pl.col("_max_score") >= t_low) & (pl.col("_max_score") < t_high))
            )
            if df_p.is_empty():
                continue
            # Exclude already covered positions (those that are already in the training set)
            if covered_pos is not None:
                df_p = df_p.filter(~pl.struct(["gene_id", "position"]).apply(lambda s: s["position"] in covered_pos.get(s["gene_id"], set())))
            if df_p.is_empty():
                continue
            # Entropy + dist
            df_p = df_p.with_columns([
                (
                    -(
                        pl.when(pl.col("donor_score") < eps).then(eps).otherwise(pl.col("donor_score")) * pl.when(pl.col("donor_score") < eps).then(eps).otherwise(pl.col("donor_score")).log()
                        + pl.when(pl.col("acceptor_score") < eps).then(eps).otherwise(pl.col("acceptor_score")) * pl.when(pl.col("acceptor_score") < eps).then(eps).otherwise(pl.col("acceptor_score")).log()
                        + pl.when(pl.col("neither_score") < eps).then(eps).otherwise(pl.col("neither_score")) * pl.when(pl.col("neither_score") < eps).then(eps).otherwise(pl.col("neither_score")).log()
                    ) / ln2
                ).alias("_entropy"),
            ]).with_columns([
                (-pl.col("_entropy")).alias("_neg_entropy"),
                pl.min_horizontal(
                    (pl.col("_max_score") - t_low).abs(),
                    (pl.col("_max_score") - t_high).abs(),
                ).alias("_dist_thr"),
            ])
            # Buffer per gene
            for gid, subdf in df_p.group_by("gene_id"):
                _update_buffer(gid, subdf)

            # `subdf` is the gene-specific slice of the current chunk after all filters:
            # It contains only rows whose
            # – gene_id == gid
            # – max(donor, acceptor) lies in [t_low, t_high)
            # – (optionally) are not in covered_pos.

            # Early stop if we reached cap rows total
            current_rows = sum(buf[0].height for buf in per_gene_buffers.values())
            if max_analysis_rows > 0 and current_rows >= max_analysis_rows:
                break

        # Merge buffers – ensure consistent column order across gene DataFrames
        canonical_cols = list(EXPECTED_MIN_COLUMNS) + [
            "_entropy",
            "_neg_entropy",
            "_dist_thr",
            "_max_score",
        ]
        # Build union of columns to avoid dropping any additional features
        union_cols: set[str] = set(canonical_cols)
        tmp_dfs: list[pl.DataFrame] = []
        for buf in per_gene_buffers.values():
            df_cat = pl.concat(buf)
            union_cols.update(df_cat.columns)
            tmp_dfs.append(df_cat)

        # Final ordered column list – canonical first, then the rest alphabetically
        ordered_cols = [c for c in canonical_cols if c in union_cols] + sorted(col for col in union_cols if col not in canonical_cols)

        filtered_pieces: list[pl.DataFrame] = []
        for df_cat in tmp_dfs:
            missing = [c for c in ordered_cols if c not in df_cat.columns]
            if missing:
                df_cat = df_cat.with_columns([pl.lit(None).alias(c) for c in missing])
            filtered_pieces.append(df_cat.select(ordered_cols))

        analysis_df = pl.concat(filtered_pieces) if filtered_pieces else pl.DataFrame()
        # NOTE: Polars concat (vertical stack) demands that all DataFrames share 
        #       the exact same column set and order.

        # Drop helper cols – they will be recomputed later if needed
        if not analysis_df.is_empty():
            analysis_df = analysis_df.drop(["_entropy", "_neg_entropy", "_dist_thr", "_max_score"])
            # Enforce global cap once more in case last chunk pushed us over
            if max_analysis_rows > 0 and analysis_df.height > max_analysis_rows:
                if verbosity >= 1:
                    print(f"[info] Final row cap: trimming to {max_analysis_rows:,} rows (was {analysis_df.height:,})")
                analysis_df = analysis_df.head(max_analysis_rows)

        # ── Diagnostics & schema validation ────────────────────────────────────
        # required_cols = {"splice_type", "sequence", "donor_score", "acceptor_score", "neither_score"}
        required_cols = set(EXPECTED_MIN_COLUMNS)
        
        if verbosity >= 2:
            print(f"[diagnostic] Concatenated analysis_sequences: {analysis_df.height} rows × {len(analysis_df.columns)} columns")
            print(f"[diagnostic] Column sample: {analysis_df.columns[:25]}")
            print(f"[diagnostic] Available columns: {sorted(analysis_df.columns)}")
        
        missing_required = [c for c in required_cols if c not in analysis_df.columns]
        if missing_required:
            if verbosity >= 1:
                print(f"[diagnostic] ❌ Missing required columns: {missing_required}")
                print(f"[diagnostic] 📋 Available columns: {sorted(analysis_df.columns)}")
                print(f"[diagnostic] 🔍 Required columns: {sorted(required_cols)}")
                
                # Check if it's just a column order/naming issue
                available_cols = set(analysis_df.columns)
                if len(available_cols & required_cols) > 0:
                    print(f"[diagnostic] 💡 Some required columns found: {sorted(available_cols & required_cols)}")
                    print(f"[diagnostic] 💡 This may be a column order or naming issue")
                
            raise RuntimeError(f"Required columns missing in concatenated analysis_sequences: {missing_required}")

        if analysis_df.is_empty():
            raise RuntimeError("Loaded analysis_sequences files but resulting dataframe is empty – cannot continue.")

        # ------------------------------------------------------------------
        # Apply ambiguity filter irrespective of in-memory vs streaming load
        # ------------------------------------------------------------------
        analysis_df = (
            analysis_df.with_columns(
                pl.max_horizontal("donor_score", "acceptor_score").alias("_max_score")
            )
            .filter((pl.col("_max_score") >= t_low) & (pl.col("_max_score") < t_high))
            .drop("_max_score")
        )
        if analysis_df.is_empty():
            raise RuntimeError(
                "No candidate positions remain after applying ambiguity thresholds "
                f"[t_low={t_low}, t_high={t_high}]. "
                "Consider loosening the thresholds or increasing --max-analysis-rows."
            )

    if covered_pos is not None:
        # Convert to polars LazyFrame for efficient filtering
        lf = analysis_df.lazy()
        # Compute max donor / acceptor score per row
        lf = lf.with_columns(
            pl.max_horizontal("donor_score", "acceptor_score").alias("_max_score")
        )
        # Ambiguous mask
        lf = lf.filter((pl.col("_max_score") >= t_low) & (pl.col("_max_score") < t_high))

        # ↑ This answers "how close is this position to being called a splice site"

        # ------------------------------------------------------------------
        # Exclude already-covered positions --------------------------------
        # ------------------------------------------------------------------
        # We'll materialise per gene because building a giant Boolean mask is
        # memory-expensive.
        filtered_chunks: list[pl.DataFrame] = []

        # --- Cheap logging & timed collect ---------------------------------
        if verbosity >= 1:
            print(f"[filter] Collecting Polars lazyframe of {analysis_df.height:,} rows …")
        _t0 = time.time()
        _collected = lf.collect()
        if verbosity >= 1:
            print(f"[filter] Polars collect finished in {time.time() - _t0:.1f}s")

        for gid, subdf in _collected.group_by("gene_id"):
            if gid in covered_pos:
                seen = covered_pos[gid]
                subdf = subdf.filter(~pl.col("position").is_in(sorted(seen)))
            filtered_chunks.append(subdf)
        filtered_df = pl.concat(filtered_chunks)
    else:
        filtered_df = analysis_df  # fallback – keep everything

    # Optional per-gene position cap to keep feature matrix small
    if max_positions_per_gene and max_positions_per_gene > 0:
        if verbosity >= 1:
            print(f"[sampling] Selecting {max_positions_per_gene} most-uncertain positions per gene (entropy)…")

        ln2 = math.log(2.0)
        eps = 1e-9
        filtered_df = (
            filtered_df.with_columns([
                (
                    -(
                        pl.when(pl.col("donor_score") < eps).then(eps).otherwise(pl.col("donor_score")) * pl.when(pl.col("donor_score") < eps).then(eps).otherwise(pl.col("donor_score")).log()
                        + pl.when(pl.col("acceptor_score") < eps).then(eps).otherwise(pl.col("acceptor_score")) * pl.when(pl.col("acceptor_score") < eps).then(eps).otherwise(pl.col("acceptor_score")).log()
                        + pl.when(pl.col("neither_score") < eps).then(eps).otherwise(pl.col("neither_score")) * pl.when(pl.col("neither_score") < eps).then(eps).otherwise(pl.col("neither_score")).log()
                    ) / ln2
                ).alias("_entropy"),
                pl.max_horizontal("donor_score", "acceptor_score").alias("_max_score")
            ])
            .with_columns([
                pl.min_horizontal(
                    (pl.col("_max_score") - t_low).abs(),
                    (pl.col("_max_score") - t_high).abs()
                ).alias("_dist_thr"),
                (-pl.col("_entropy")).alias("_neg_entropy")
            ])
            .sort(["gene_id", "_neg_entropy", "_dist_thr"])  # descending entropy, then closest to threshold
            .group_by("gene_id").head(max_positions_per_gene)
            .drop(["_entropy", "_neg_entropy", "_dist_thr", "_max_score"])
        )

    # Sanity check
    if verbosity >= 1:
        print(
            f"[inference-workflow] Filtered analysis_sequences: "
            f"{len(filtered_df):,} rows (→ unseen & ambiguous)"
        )
        # ↑ This count equals the number of positions whose feature rows
        #   will be handed to incremental_builder for meta-model re-scoring

    # Write filtered TSV so incremental builder can pick it up
    out_tsv = artefact_dir / "analysis_sequences_inference.tsv"
    if verbosity >= 1:
        print(f"[write] Writing {filtered_df.height:,} rows to {out_tsv} …")
    filtered_df.write_csv(out_tsv, separator="\t")
    if verbosity >= 1:
        print("[write] Done")

    # ------------------------------------------------------------------
    # Load canonical training schema (optional) -------------------------
    # ------------------------------------------------------------------
    schema_cols: list[str] | None = None
    if train_schema_dir is not None:
        try:
            # Use the new model resource manager for systematic schema loading
            from meta_spliceai.splice_engine.meta_models.workflows.inference.model_resource_manager import create_model_resource_manager
            model_manager = create_model_resource_manager()
            schema_cols = model_manager.load_feature_schema(train_schema_dir)
            
            if schema_cols is None:
                # Fallback to old method
                schema_cols = _load_training_schema(Path(train_schema_dir))
                
        except Exception as exc:
            raise RuntimeError(f"Failed to load training schema from {train_schema_dir}: {exc}") from exc

    # ------------------------------------------------------------------
    # Build feature matrix with incremental_builder ---------------------
    # ------------------------------------------------------------------
    if verbosity >= 1:
        print("[inference-workflow] Building enriched feature matrix via incremental_builder …")

    from meta_spliceai.splice_engine.meta_models.builder.incremental_builder import (
        incremental_build_training_dataset,
    )

    gene_list = filtered_df.select("gene_id").unique().get_column("gene_id").to_list()
    feature_dir = artefact_dir / "features"
    master_dataset_dir = incremental_build_training_dataset(
        eval_dir=result["paths"].get("eval_dir"),
        output_dir=feature_dir,
        subset_policy="custom",
        additional_gene_ids=gene_list,
        run_workflow=False,
        overwrite=True,
        verbose=max(0, verbosity - 1),
        initial_schema_cols=schema_cols,
    )

    if verbosity >= 1:
        print(f"[inference-workflow] Feature dataset ready at: {master_dataset_dir}")

    # ------------------------------------------------------------------
    # Optional clean-up --------------------------------------------------
    # ------------------------------------------------------------------
    if cleanup:
        patterns = [
            # Remove *original* heavy files only – keep the newly written TSV
            "analysis_sequences_*.tsv",
            "analysis_sequences_*.tsv.gz",
            "analysis_sequences_*.parquet",
            "splice_positions_enhanced_*.tsv",
            "splice_positions_enhanced_*.tsv.gz",
            "splice_positions_enhanced_*.parquet",
        ]
        removed = 0
        for pat in patterns:
            for f in artefact_dir.glob(pat):
                # Skip the inference TSV we just wrote
                if f.name == out_tsv.name:
                    continue
                try:
                    f.unlink(missing_ok=True)
                    removed += 1
                except Exception as exc:  # pragma: no cover – best-effort
                    print(f"[cleanup-warn] Could not remove {f}: {exc}")
        if verbosity >= 1:
            print(f"[inference-workflow] Clean-up removed {removed} bulky artefact files from {artefact_dir}")

    if verbosity >= 1:
        print(f"[inference-workflow] Base workflow completed. Artefacts in: {artefact_dir}")
        
    # ------------------------------------------------------------------
    # Load model for prediction (if provided) ---------------------------
    # ------------------------------------------------------------------
    model = None
    model_info = None
    if model_path is not None:
        model_path = Path(model_path)
        if verbosity >= 1:
            print(f"[inference-workflow] Loading model from {model_path}")
            
        try:
            model = load_model_with_calibration(model_path, use_calibration=use_calibration)
            model_info = get_model_info(model_path)
            if verbosity >= 1:
                cal_status = "with calibration" if model_info.get("has_calibration", False) else "without calibration"
                model_type = model_info.get("type", "unknown")
                print(f"[inference-workflow] Loaded {model_type} model {cal_status}")
        except Exception as e:
            print(f"[warning] Failed to load model: {e}")
            model = None
    
    # ------------------------------------------------------------------
    # Load dataset for prediction --------------------------------------
    # ------------------------------------------------------------------
    feature_dataset = None
    feature_X = None
    feature_meta = None
    
    # Only load if we have a model and will perform analysis
    if model is not None and (neigh_sample > 0 or diag_sample > 0):
        if verbosity >= 1:
            print(f"[inference-workflow] Loading feature matrix for prediction")
        
        try:
            # Load parquet dataset
            import glob
            parquet_files = list(glob.glob(str(master_dataset_dir / "*.parquet")))
            
            if parquet_files:
                # Combine all parquet files
                feature_dfs = []
                meta_cols = ["gene_id", "chrom", "position", "strand"]
                
                for pf in parquet_files:
                    df = pd.read_parquet(pf)
                    feature_dfs.append(df)
                
                if feature_dfs:
                    # Combine all feature dataframes
                    feature_df = pd.concat(feature_dfs, ignore_index=True)
                    
                    # Extract metadata columns and feature matrix
                    feature_meta = {}
                    for col in meta_cols:
                        if col in feature_df.columns:
                            feature_meta[col] = feature_df[col]
                            feature_df = feature_df.drop(columns=[col])
                    
                    # Extract label if present
                    y = None
                    if "label" in feature_df.columns:
                        feature_meta["label"] = feature_df["label"]
                        feature_df = feature_df.drop(columns=["label"])
                    
                    # Convert to numpy array for prediction
                    feature_X = feature_df.values
                    
                    if verbosity >= 1:
                        print(f"[inference-workflow] Loaded feature matrix with shape {feature_X.shape}")
        except Exception as e:
            print(f"[warning] Failed to load feature matrix: {e}")
            feature_X = None
    
    # ------------------------------------------------------------------
    # Apply diagnostic sampling (if requested) -------------------------
    # ------------------------------------------------------------------
    diagnostic_results = None
    if model is not None and feature_X is not None and diag_sample > 0:
        if verbosity >= 1:
            print(f"[inference-workflow] Performing diagnostic sampling with {diag_sample} samples")
            
        try:
            # Extract labels if available
            y = feature_meta.get("label", None)
            
            # Perform diagnostic sampling
            X_sample, y_sample, meta_sample = diagnostic_sampling(
                feature_X, y, feature_meta, 
                sample_size=diag_sample,
                stratify=(y is not None)
            )
            
            # Make predictions on sampled data
            y_proba = model.predict_proba(X_sample)
            
            # Store results
            diagnostic_results = {
                "X_sample": X_sample,
                "y_sample": y_sample,
                "meta_sample": meta_sample,
                "predictions": y_proba,
                "sample_size": len(X_sample)
            }
            
            # Save diagnostic results to CSV if metadata available
            if meta_sample is not None:
                diag_dir = artefact_dir / "diagnostics"
                diag_dir.mkdir(parents=True, exist_ok=True)
                
                # Create diagnostic dataframe
                diag_data = {}
                for k, v in meta_sample.items():
                    diag_data[k] = v
                
                if y_proba is not None:
                    diag_data["prob_donor"] = y_proba[:, 0]
                    diag_data["prob_acceptor"] = y_proba[:, 1]
                    diag_data["prob_neither"] = y_proba[:, 2]
                
                diag_df = pd.DataFrame(diag_data)
                diag_df.to_csv(diag_dir / "diagnostic_samples.csv", index=False)
                
                if verbosity >= 1:
                    print(f"[inference-workflow] Saved diagnostic results to {diag_dir / 'diagnostic_samples.csv'}")
        except Exception as e:
            print(f"[warning] Failed to perform diagnostic sampling: {e}")
            diagnostic_results = {"error": str(e)}
    
    # ------------------------------------------------------------------
    # Perform neighborhood analysis (if requested) ---------------------
    # ------------------------------------------------------------------
    neighborhood_results = None
    if model is not None and feature_X is not None and neigh_sample > 0 and neigh_window > 0:
        if verbosity >= 1:
            print(f"[inference-workflow] Performing neighborhood analysis with {neigh_sample} samples and window size {neigh_window}")
            
        try:
            # Ensure required metadata columns exist
            if "position" not in feature_meta or ("chrom" not in feature_meta and "gene_id" not in feature_meta):
                raise ValueError("Neighborhood analysis requires 'position' and either 'chrom' or 'gene_id' columns")
            
            # Choose chromosome column
            chrom_col = "chrom" if "chrom" in feature_meta else "gene_id"
            
            # Create dataframe from feature_meta
            df_meta = pd.DataFrame(feature_meta)
            
            # Perform neighborhood analysis
            neigh_dir = artefact_dir / "neighborhood"
            neighborhood_results = perform_neighborhood_analysis(
                model=model,
                X=feature_X,
                df=df_meta,
                chrom_col=chrom_col,
                pos_col="position",
                sample_count=neigh_sample,
                window_size=neigh_window,
                out_dir=neigh_dir,
                plot_title=f"Neighborhood Analysis (window={neigh_window})"
            )
            
            if verbosity >= 1 and neighborhood_results:
                print(f"[inference-workflow] Completed neighborhood analysis with {neighborhood_results.get('neighborhood_samples', 0)} positions")
        except Exception as e:
            print(f"[warning] Failed to perform neighborhood analysis: {e}")
            neighborhood_results = {"error": str(e)}
    
    # ------------------------------------------------------------------
    # Optional cleanup of feature matrix files ----------------------------
    # ------------------------------------------------------------------
    if cleanup_features and feature_X is not None:
        feature_files_removed = 0
        patterns = [
            "*.npz",         # Sparse matrices
            "*.npy",         # Dense matrices
            "*.feather",     # Feature metadata
            "*.parquet",     # Parquet files
            "metadata.json", # Metadata files
        ]
        # Only clean up feature files in the feature directory
        # Keep diagnostic and neighborhood analysis results
        feature_dir = artefact_dir / "features"
        if feature_dir.exists() and feature_dir.is_dir():
            for pattern in patterns:
                for f in feature_dir.glob(pattern):
                    try:
                        # Skip files in subdirectories for neighborhood or diagnostics
                        if "neighborhood" in str(f) or "diagnostics" in str(f):
                            continue
                        f.unlink(missing_ok=True)
                        feature_files_removed += 1
                    except Exception as exc:  # pragma: no cover – best-effort
                        print(f"[cleanup-warn] Could not remove feature file {f}: {exc}")
            
            if verbosity >= 1 and feature_files_removed > 0:
                print(f"[inference-workflow] Feature cleanup removed {feature_files_removed} files from {feature_dir}")
    
    if verbosity >= 1:
        print(f"[inference-workflow] Workflow completed. Artefacts in: {artefact_dir}")

    # Return results dictionary
    return {
        "success": True,
        "paths": {
            "output_dir": artefact_dir,
            "feature_dir": master_dataset_dir,
        },
        "model_info": model_info,
        "neighborhood_results": neighborhood_results,
        "diagnostic_results": diagnostic_results
    }

def run_complete_coverage_inference_workflow(
    *,
    target_genes: List[str] | None = None,
    target_chromosomes: List[str] | None = None,
    verbosity: int = 1,
    max_positions_per_gene: int = 0,
    no_final_aggregate: bool = True,
    # Model and prediction options
    model_path: Optional[Path] = None,
    use_calibration: bool = True,
    # Directory options
    output_dir: Optional[str | Path] = None,
    train_schema_dir: Optional[str | Path] = None,
    # Data preparation switches - reuse existing datasets when possible
    do_extract_annotations: bool = False,
    do_extract_sequences: bool = False,
    do_extract_splice_sites: bool = False,
    do_find_overlaping_genes: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """
    Run the complete coverage inference workflow that generates base model scores for ALL positions.
    
    This function replicates the prediction workflow's nested for loop to generate base model
    scores for positions that were downsampled during training. It reuses existing genomic
    datasets for efficiency but runs the full prediction pipeline for complete coverage.
    
    CRITICAL: This function sets no_tn_sampling=True to disable TN-specific position subsampling,
    ensuring that ALL positions in the gene receive base model scores, not just the downsampled subset.
    
    Parameters:
    ----------
    target_genes : List[str], optional
        List of gene IDs to process. If None, processes all genes.
    target_chromosomes : List[str], optional
        List of chromosomes to process. If None, processes all chromosomes.
    verbosity : int, default=1
        Controls output verbosity.
    max_positions_per_gene : int, default=0
        Maximum positions per gene to process. 0 means no limit.
    no_final_aggregate : bool, default=True
        Whether to skip final aggregation to save memory.
    model_path : Path, optional
        Path to the model file for predictions.
    use_calibration : bool, default=True
        Whether to use calibration when available.
    output_dir : Path, optional
        Output directory for results.
    train_schema_dir : Path, optional
        Training schema directory for feature consistency.
    do_extract_* : bool
        Flags to control data preparation steps. Set to False to reuse existing datasets.
    **kwargs : dict
        Additional parameters passed to the prediction workflow.
        
    Returns:
    -------
    Dict[str, Any]
        Dictionary containing workflow results and paths.
    """
    
    if verbosity >= 1:
        print("🔥🔥🔥 [COMPLETE COVERAGE] Starting complete coverage inference workflow 🔥🔥🔥")
        print(f"🔥🔥🔥 [COMPLETE COVERAGE] Target genes: {target_genes}")
        print(f"🔥🔥🔥 [COMPLETE COVERAGE] Target chromosomes: {target_chromosomes}")
        print(f"🔥🔥🔥 [COMPLETE COVERAGE] Output dir: {output_dir}")
    
    # ------------------------------------------------------------------
    # Handle output directories
    # ------------------------------------------------------------------
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if output_dir is None:
        output_dir = Path(f"complete_coverage_inference_{timestamp}")
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ------------------------------------------------------------------
    # Prepare configuration for the prediction workflow
    # ------------------------------------------------------------------
    _wk = dict(kwargs)  # make a copy
    # Check if key genomic datasets already exist
    key_datasets = [
        "data/ensembl/splice_sites.tsv",
        "data/ensembl/spliceai_analysis/gene_features.tsv", 
        "data/ensembl/spliceai_analysis/transcript_features.tsv",
        "data/ensembl/spliceai_analysis/exon_features.tsv",
        "data/ensembl/overlapping_gene_counts.tsv"
    ]
    
    existing_datasets = [ds for ds in key_datasets if Path(ds).exists()]
    missing_datasets = [ds for ds in key_datasets if not Path(ds).exists()]
    
    if verbosity >= 1:
        print(f"[complete-coverage] 📊 Checking existing genomic datasets...")
        print(f"[complete-coverage] ✅ Found {len(existing_datasets)} existing datasets")
        if missing_datasets:
            print(f"[complete-coverage] ⚠️ Missing {len(missing_datasets)} datasets: {missing_datasets}")
    
    # If all key datasets exist, skip data preparation for efficiency
    if len(existing_datasets) == len(key_datasets):
        if verbosity >= 1:
            print(f"[complete-coverage] 🚀 All key datasets exist - skipping data preparation for efficiency")
        do_extract_annotations = False
        do_extract_sequences = False
        do_extract_splice_sites = False
        do_find_overlaping_genes = False
    else:
        if verbosity >= 1:
            print(f"[complete-coverage] ⚠️ Some datasets missing - will prepare missing data")
    
    _wk.update({
        # Data preparation switches - reuse existing datasets when possible
        "do_extract_annotations": do_extract_annotations,
        "do_extract_sequences": do_extract_sequences,
        "do_extract_splice_sites": do_extract_splice_sites,
        "do_find_overlaping_genes": do_find_overlaping_genes,
        
        # Gene filter
        "target_genes": target_genes,
        "target_chromosomes": target_chromosomes,
        
        # Use a dedicated subdirectory for complete coverage
        "eval_dir": str(output_dir),
        
        # CRITICAL: Set local_dir to global data directory to find existing datasets
        "local_dir": "data/ensembl",
        
        # Use parquet format for sequences (matching existing files)
        "seq_format": "parquet",
        
        # Disable memory-heavy aggregation for efficiency
        "no_final_aggregate": no_final_aggregate,
        
        # CRITICAL: Disable TN sampling to preserve ALL positions for complete coverage
        "no_tn_sampling": True,
    })
    
    if verbosity >= 1:
        print(f"[complete-coverage] Configuration prepared")
        print(f"[complete-coverage] Data preparation: annotations={do_extract_annotations}, sequences={do_extract_sequences}, splice_sites={do_extract_splice_sites}, overlaps={do_find_overlaping_genes}")
    
    # ------------------------------------------------------------------
    # Run the enhanced prediction workflow for complete coverage
    # ------------------------------------------------------------------
    if verbosity >= 1:
        print(f"[complete-coverage] 🚀 Starting enhanced prediction workflow for complete coverage...")
    
    try:
        # Run the full prediction workflow to generate base model scores for ALL positions
        result = run_enhanced_splice_prediction_workflow(
            verbosity=max(0, verbosity - 1), 
            **_wk
        )
        
        if not result.get("success", False):
            raise RuntimeError("Enhanced splice prediction workflow failed; cannot continue.")
        
        if verbosity >= 1:
            print(f"[complete-coverage] ✅ Enhanced prediction workflow completed successfully")
            
        # Get the output directory from the result
        artefact_dir = Path(result["paths"].get("output_dir", output_dir))
        
        # ------------------------------------------------------------------
        # Load and process the generated analysis sequences
        # ------------------------------------------------------------------
        if verbosity >= 1:
            print(f"[complete-coverage] 📊 Loading generated analysis sequences...")
        
        # Load analysis_sequences files from disk (chunk-level TSVs)
        tsv_paths = [p for p in artefact_dir.glob("analysis_sequences_*.*") 
                    if p.suffix in {".tsv", ".tsv.gz", ".parquet"} and "inference" not in p.name]
        
        if not tsv_paths:
            raise RuntimeError("No analysis_sequences files generated by prediction workflow")
        
        if verbosity >= 1:
            print(f"[complete-coverage] 📁 Found {len(tsv_paths)} analysis_sequences files")
        
        # Load and combine all analysis sequences
        analysis_sequences_list = []
        total_rows = 0
        
        for p in tsv_paths:
            if verbosity >= 2:
                print(f"[complete-coverage] 📄 Loading {p.name}...")
            
            try:
                if p.suffix == ".parquet":
                    df_chunk = pl.read_parquet(p)
                else:
                    df_chunk = pl.read_csv(p, separator="\t")
                
                # Filter to target genes if specified
                if target_genes:
                    df_chunk = df_chunk.filter(pl.col("gene_id").is_in(target_genes))
                
                if not df_chunk.is_empty():
                    analysis_sequences_list.append(df_chunk)
                    total_rows += len(df_chunk)
                    
                    if verbosity >= 2:
                        print(f"[complete-coverage] ✅ Loaded {len(df_chunk)} rows from {p.name}")
                        
            except Exception as e:
                if verbosity >= 1:
                    print(f"[complete-coverage] ⚠️ Failed to load {p.name}: {e}")
                continue
        
        if not analysis_sequences_list:
            raise RuntimeError("No analysis_sequences data loaded from generated files")
        
        # Combine all analysis sequences with consistent schema
        if len(analysis_sequences_list) > 1:
            # Ensure consistent schema across all dataframes
            first_schema = analysis_sequences_list[0].schema
            
            # Convert all dataframes to have the same schema as the first one
            aligned_sequences = []
            for i, df in enumerate(analysis_sequences_list):
                try:
                    # Cast to the same schema as the first dataframe
                    aligned_df = df.cast(first_schema)
                    aligned_sequences.append(aligned_df)
                except Exception as e:
                    if verbosity >= 1:
                        print(f"[complete-coverage] ⚠️ Schema alignment failed for dataframe {i}: {e}")
                    # Fallback: try to cast problematic columns
                    try:
                        # Common problematic columns that might have type mismatches
                        df_fixed = df
                        for col in df.columns:
                            if col in first_schema:
                                expected_type = first_schema[col]
                                if expected_type == pl.Utf8 and df[col].dtype != pl.Utf8:
                                    df_fixed = df_fixed.with_columns(pl.col(col).cast(pl.Utf8))
                                elif expected_type == pl.Int64 and df[col].dtype != pl.Int64:
                                    df_fixed = df_fixed.with_columns(pl.col(col).cast(pl.Int64))
                        aligned_sequences.append(df_fixed)
                    except Exception as e2:
                        if verbosity >= 1:
                            print(f"[complete-coverage] ⚠️ Fallback schema alignment also failed: {e2}")
                        # Last resort: just append as is
                        aligned_sequences.append(df)
            analysis_sequences_df = pl.concat(aligned_sequences)
        else:
            analysis_sequences_df = analysis_sequences_list[0]
        
        if verbosity >= 1:
            print(f"[complete-coverage] ✅ Loaded {len(analysis_sequences_df)} total analysis sequences")
            print(f"[complete-coverage] 🧬 Genes covered: {analysis_sequences_df['gene_id'].n_unique()}")
            print(f"[complete-coverage] 📍 Position range: {analysis_sequences_df['position'].min()} to {analysis_sequences_df['position'].max()}")
        
        # ------------------------------------------------------------------
        # Save the complete analysis sequences
        # ------------------------------------------------------------------
        complete_analysis_file = output_dir / "complete_analysis_sequences.tsv"
        analysis_sequences_df.write_csv(str(complete_analysis_file), separator="\t")
        
        if verbosity >= 1:
            print(f"[complete-coverage] 💾 Saved complete analysis sequences to {complete_analysis_file}")
        
        # ------------------------------------------------------------------
        # Create result structure
        # ------------------------------------------------------------------
        result = {
            "success": True,
            "paths": {
                "output_dir": str(output_dir),
                "artefact_dir": str(artefact_dir),
                "complete_analysis_sequences": str(complete_analysis_file),
            },
            "analysis_sequences": analysis_sequences_df,
            "total_positions": len(analysis_sequences_df),
            "genes_covered": analysis_sequences_df["gene_id"].n_unique(),
            "position_range": {
                "min": analysis_sequences_df["position"].min(),
                "max": analysis_sequences_df["position"].max()
            }
        }
        
        if verbosity >= 1:
            print(f"[complete-coverage] 🎉 Complete coverage inference workflow completed successfully!")
            print(f"[complete-coverage] 📊 Total positions generated: {len(analysis_sequences_df):,}")
            print(f"[complete-coverage] 🧬 Genes covered: {analysis_sequences_df['gene_id'].n_unique()}")
        
        return result
        
    except Exception as e:
        if verbosity >= 1:
            print(f"[complete-coverage] ❌ Complete coverage inference workflow failed: {e}")
            import traceback
            traceback.print_exc()
        
        return {
            "success": False,
            "error": str(e),
            "paths": {
                "output_dir": str(output_dir)
            }
        }
