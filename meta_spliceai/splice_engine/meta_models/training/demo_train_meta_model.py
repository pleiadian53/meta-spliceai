"""Quick-start demo for training and evaluating a meta-model.

Usage
-----
python -m meta_spliceai.splice_engine.meta_models.training.demo_train_meta_model \
       /path/to/dataset/master[.parquet]                                   \
       /path/to/spliceai/full_splice_performance.tsv                       \
       --prune-features --model xgboost --out-dir models/demo_xgb

Features
~~~~~~~~
1. Optional feature pruning: removes zero-variance columns and drops pairs with
   |corr| > 0.99 (numeric only).  The trimmed dataset is saved alongside the
   output directory so that future runs can reuse it.
2. Wraps the high-level :class:`Trainer` and prints both raw metrics **and** the
   per-gene Δ compared to the base SpliceAI profile.
3. Works on datasets up to ~3 M rows.  For bigger sets use
   ``training.incremental.IncrementalTrainer``.
"""
from __future__ import annotations

import gc
import argparse
import json
import sys
from pathlib import Path
from typing import Tuple
import inspect
import os
import psutil
import random

import numpy as np
import polars as pl
import pyarrow as pa # For potential direct Arrow operations
import pyarrow.parquet as pq # For ParquetWriter
from sklearn.feature_selection import VarianceThreshold

from meta_spliceai.splice_engine.meta_models.builder import preprocessing as prep


def log_memory_usage(stage: str):
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    print(f"[MemoryLog] Stage: {stage} - RSS: {mem_info.rss / 1024 ** 2:.2f} MB, VMS: {mem_info.vms / 1024 ** 2:.2f} MB")

from meta_spliceai.splice_engine.meta_models.training import Trainer
from meta_spliceai.splice_engine.meta_models.io.handlers import MetaModelDataHandler


# ---------------------------------------------------------------------------
# Feature-pruning helpers ----------------------------------------------------
# ---------------------------------------------------------------------------

def _high_corr_cols(df: pl.DataFrame, *, threshold: float = 0.99) -> set[str]:
    """Return columns whose Pearson corr > *threshold* (keep upper triangle)."""
    num_cols = [c for c in df.columns if df[c].dtype.is_numeric()]
    if len(num_cols) < 2:
        return set()

    corr_mat = df.select(num_cols).corr()  # numpy ndarray
    upper = np.triu_indices_from(corr_mat, k=1)
    to_drop = {
        num_cols[j]
        for i, j in zip(*upper)
        if abs(corr_mat[i, j]) >= threshold
    }
    return to_drop

# ---------------------------------------------------------------------------
# Memory-friendly shard-wise pruning implementation
# ---------------------------------------------------------------------------

def _prune_dataset_shardwise(
    src: Path,
    dst: Path,
    *,
    corr_thr: float = 0.99,
    sample_rows_per_shard: int,  # Changed from sample_rows
    max_corr_cols: int = 1500,
    label_col: str = "splice_type",
    verbose: int = 1,
) -> Tuple[int, int]:
    """Prune columns shard-by-shard to keep peak RSS low.

    The algorithm samples *at most* a few rows from **each** Parquet shard to
    determine candidate zero-variance and high-correlation columns, then writes
    the full dataset with those columns dropped in a second streaming pass.
    """
    print(f"[MemoryLog] Using Polars version: {pl.__version__}")
    log_memory_usage("prune_dataset[shard]: Start")
    # ------------------------------------------------------------------
    # 1. Discover shards ------------------------------------------------
    # ------------------------------------------------------------------
    shard_files: list[Path]
    if src.is_file():
        shard_files = [src]
    else:
        shard_files = sorted(src.glob("*.parquet"))
        if not shard_files:
            raise FileNotFoundError(f"No Parquet shards found under {src}")

    rows_per_shard = sample_rows_per_shard

    const_cols: set[str] = set()
    corr_drop: set[str] = set()

    for i, pf in enumerate(shard_files):
        log_memory_usage(
            f"prune_dataset[shard]: Reading {rows_per_shard} rows from {pf.name} ({i+1}/{len(shard_files)})"
        )
        try:
            df_sample = pl.read_parquet(pf, n_rows=rows_per_shard)
        except Exception as e:
            print(f"[MemoryLog] Failed to read shard {pf}: {e}; skipping.")
            continue

        # Drop label column – not a feature.
        if label_col in df_sample.columns:
            df_sample = df_sample.drop(label_col)

        # ----- Zero-variance ------------------------------------------
        uniques = df_sample.select(pl.all().n_unique()).to_dicts()[0]
        zero_cols = [col for col, nuni in uniques.items() if nuni <= 1]
        const_cols.update(zero_cols)

        # ----- High correlation (optional & width-limited) -------------
        numeric_cols = [
            c for c in df_sample.columns if df_sample[c].dtype.is_numeric() and c not in const_cols
        ]
        if corr_thr < 1.0 and numeric_cols and len(numeric_cols) <= max_corr_cols:
            corr_drop.update(_high_corr_cols(df_sample.select(numeric_cols), threshold=corr_thr))

        del df_sample  # free sample memory

    if verbose:
        print(
            f"[prune-shard] identified {len(const_cols)} zero-variance + {len(corr_drop)} high-corr columns"
        )

    cols_to_drop = list(const_cols | corr_drop)

    # ------------------------------------------------------------------
    # 2. Stream full dataset, drop columns, and write -------------------
    # ------------------------------------------------------------------
    log_memory_usage("prune_dataset[shard]: Scanning full dataset for write")
    scan_kwargs: dict[str, object] = {}
    if "missing_columns" in inspect.signature(pl.scan_parquet).parameters:
        scan_kwargs["missing_columns"] = "insert"

    if src.is_file():
        ldf_all = pl.scan_parquet(str(src), **scan_kwargs)
    else:
        ldf_all = pl.scan_parquet(str(src / "*.parquet"), low_memory=True, **scan_kwargs)

    if cols_to_drop:
        ldf_pruned = ldf_all.drop(cols_to_drop)
    else:
        ldf_pruned = ldf_all

    log_memory_usage("prune_dataset[shard]: Determining target schema for pruned dataset")
    try:
        target_arrow_schema = ldf_pruned.limit(0).collect(engine='streaming').to_arrow().schema
    except Exception as e_schema_target:
        print(f"[MemoryLog] CRITICAL: Could not determine target schema from ldf_pruned: {e_schema_target}. Output may be incorrect or empty.")
        target_arrow_schema = pa.schema([]) # Fallback to empty schema

    dst_path_str = str(dst if dst.suffix == ".parquet" else f"{dst}.parquet")

    if not target_arrow_schema.names: # Check if the schema has any fields
        log_memory_usage(f"prune_dataset[shard]: Target schema is empty. Writing a minimal empty Parquet file to {dst_path_str}.")
        pl.DataFrame().write_parquet(dst_path_str, compression="zstd")
        return len(const_cols), len(corr_drop)

    # Proceed only if target_arrow_schema has fields
    writer = None
    try:
        if not shard_files:
            log_memory_usage("[MemoryLog] No source files found. Writing empty Parquet file with target schema.")
            # target_arrow_schema is guaranteed to have names here
            pq.write_table(pa.Table.from_batches([], schema=target_arrow_schema), dst_path_str, compression='zstd', compression_level=1, version='2.6')
            return len(const_cols), len(corr_drop)

        writer = pq.ParquetWriter(dst_path_str, target_arrow_schema, compression='zstd', compression_level=1, version='2.6')
        log_memory_usage(f"prune_dataset[shard_append]: Initialized ParquetWriter for {dst_path_str} with target schema ({len(target_arrow_schema.names)} columns).")

        for i, shard_file_path in enumerate(shard_files):
            log_memory_usage(f"prune_dataset[shard_append]: Reading source shard {i+1}/{len(shard_files)}: {shard_file_path.name}")
            df_shard_eager = pl.read_parquet(shard_file_path)
            
            # Apply initial pruning based on cols_to_drop
            df_after_initial_drop = df_shard_eager
            if cols_to_drop:
                existing_cols_in_shard_to_drop = [col for col in cols_to_drop if col in df_shard_eager.columns]
                if existing_cols_in_shard_to_drop:
                    df_after_initial_drop = df_shard_eager.drop(existing_cols_in_shard_to_drop)
            
            del df_shard_eager  # Free memory of original shard
            gc.collect()
            log_memory_usage(f"prune_dataset[shard_append]: Pruned shard {i+1}. Conforming to target schema.")

            # Conform df_after_initial_drop to target_arrow_schema
            select_expressions = []
            for field_name in target_arrow_schema.names:  # Iterate in target order
                if field_name in df_after_initial_drop.columns:
                    # Keep column as-is. We'll rely on a final Arrow cast for type alignment.
                    select_expressions.append(pl.col(field_name))
                else:
                    # Column is absent in this shard – add a null column placeholder.
                    select_expressions.append(pl.lit(None).alias(field_name))
            
            df_conformed = df_after_initial_drop.select(select_expressions)
            del df_after_initial_drop
            gc.collect()
            log_memory_usage(f"prune_dataset[shard_append]: Conformed shard {i+1}. Converting to Arrow.")
            
            # Convert to Arrow and cast to the unified target schema to ensure
            # consistent column ordering and data types across shards.
            arrow_table_to_write = df_conformed.to_arrow().cast(target_arrow_schema, safe=False)
            del df_conformed
            
            if arrow_table_to_write.num_rows > 0:
                writer.write_table(arrow_table_to_write)
                log_memory_usage(f"prune_dataset[shard_append]: Wrote {arrow_table_to_write.num_rows} rows from shard {i+1} to {dst_path_str}")
            else:
                log_memory_usage(f"prune_dataset[shard_append]: Shard {i+1} resulted in an empty table after conforming. Not writing this shard's data.")
            del arrow_table_to_write
            gc.collect()
        
    except Exception as e_write:
        print(f"[MemoryLog] Error during shard-by-shard Parquet writing: {e_write}")
        # Ensure partial file is removed or handled if error occurs mid-write
        if writer is not None:
            try:
                writer.close() # Try to close, may error if state is bad
            except Exception as e_close_on_error:
                print(f"[MemoryLog] Error closing writer after another error: {e_close_on_error}")
            writer = None # Prevent further use in finally
        # Consider removing dst_path_str if it's incomplete, though this can be risky.
        # For now, just re-raise.
        raise
    finally:
        if writer:
            writer.close()
            log_memory_usage("prune_dataset[shard_append]: Closed ParquetWriter")

    log_memory_usage("prune_dataset[shard]: Done")
    return len(const_cols), len(corr_drop)


def prune_dataset(
    src: Path,
    dst: Path,
    *,
    corr_thr: float = 0.99,
    sample_rows_per_shard: int,  # Changed from sample_rows
    max_corr_cols: int = 1500,
    label_col: str = "splice_type",
    verbose: int = 1,
) -> Tuple[int, int]:
    """Prune feature *columns* using *shard-wise* strategy to minimise memory usage.

    This function is now a thin wrapper around :func:`_prune_dataset_shardwise`.
    """

    # Delegate to the new shard-wise implementation
    return _prune_dataset_shardwise(
        src,
        dst,
        corr_thr=corr_thr,
        sample_rows_per_shard=sample_rows_per_shard,  # Changed
        max_corr_cols=max_corr_cols,
        label_col=label_col,
        verbose=verbose,
    )


# ---------------------------------------------------------------------------
def _prune_dataset_sample_based(*args, **kwargs):
    """Legacy sample-based pruning function (disabled)."""
    raise NotImplementedError(
        "Sample-based pruning has been disabled; use shard-wise pruning instead."
    )

# CLI -----------------------------------------------------------------------
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:  # noqa: D401
    parser = argparse.ArgumentParser(
        description="Train a meta-model and compare to SpliceAI baseline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("dataset", help="Master Parquet directory or file.")
    parser.add_argument("baseline_tsv", nargs="?", help="SpliceAI per-gene performance TSV (auto-detect if omitted).")
    parser.add_argument("--model", default="xgboost", help="Model spec name or JSON dict.")
    parser.add_argument("--out-dir", default="models/demo_run", help="Where to write model & metrics.")
    parser.add_argument("--group-col", default="gene_id", help="Column holding gene IDs for group split.")
    parser.add_argument("--prune-features", action="store_true", help="Apply zero-variance & high-corr pruning.")
    parser.add_argument("--corr-thr", type=float, default=0.99, help="Correlation threshold for pruning.")
    parser.add_argument(
        "--pruning-sample-rows-per-shard",
        type=int,
        default=25_000,
        help="Number of rows to sample from each Parquet shard for pruning analysis. "
             "Lower values reduce memory but may be less representative.",
    )
    parser.add_argument(
        "--pruning-max-corr-cols",
        type=int,
        default=1500,
        help="Maximum number of numeric columns to consider for correlation analysis during pruning. "
             "Reduces memory if many numeric columns exist.",
    )
    parser.add_argument("--shard", type=int, default=None, help="Only load this shard index (1-based) when DATASET is a directory of shards.")

    args = parser.parse_args(argv)

    dataset_path = Path(args.dataset)

    # ------------------------------------------------------------------
    # Shard selection ----------------------------------------------------
    # ------------------------------------------------------------------
    if args.shard is not None:
        if not dataset_path.is_dir():
            parser.error("--shard can only be used when DATASET points to a directory containing batch_*.parquet files.")
        shard_file = dataset_path / f"batch_{args.shard:05d}.parquet"
        if not shard_file.exists():
            parser.error(f"Shard file {shard_file} not found.")
        print(f"[demo] Using shard file {shard_file}")
        dataset_path_selected = shard_file
    else:
        dataset_path_selected = dataset_path
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Optional feature pruning                                          
    # ------------------------------------------------------------------
    if args.prune_features:
        trimmed_path = out_dir / "dataset_trimmed.parquet"
        print("[demo] Pruning features …")
        n_const, n_corr = prune_dataset(
            dataset_path_selected,
            trimmed_path,
            corr_thr=args.corr_thr,
            sample_rows_per_shard=args.pruning_sample_rows_per_shard,
            max_corr_cols=args.pruning_max_corr_cols,
            # verbose=1, # Assuming default verbosity or could be linked to a general verbose CLI arg
        )
        print(f"[demo] Removed {n_const} constant + {n_corr} highly-corr columns; new dataset → {trimmed_path}")
        dataset_for_training = trimmed_path
    else:
        dataset_for_training = dataset_path_selected

    # ------------------------------------------------------------------
    # Train model                                                       
    # ------------------------------------------------------------------
    trainer = Trainer(
        model_spec=args.model,
        out_dir=out_dir,
        group_col=args.group_col,
    )
    trainer.fit(dataset_for_training)
    trainer.save()

    # --------------------------------------------
    # Feature manifest for reproducibility
    # --------------------------------------------
    import pandas as _pd  # local import to avoid extra dependency at top level

    feat_manifest_df = _pd.DataFrame({
        "feature": trainer.feature_names_,
        "description": "",  # placeholder to be filled manually
    })
    manifest_path = out_dir / "feature_manifest.csv"
    feat_manifest_df.to_csv(manifest_path, index=False)
    print(f"[demo] Feature manifest → {manifest_path} ({len(feat_manifest_df)} features)")

    print("\n================ METRICS ================")
    print(json.dumps(trainer.metrics, indent=2))

    # ------------------------------------------------------------------
    # Baseline comparison                                               
    # ------------------------------------------------------------------
    print("\n============= Δ vs SpliceAI =============")
    # ------------------------------------------------------------------
    # Detect baseline TSV if missing                                     
    # ------------------------------------------------------------------
    if args.baseline_tsv is None:
        handler = MetaModelDataHandler()
        auto_path = Path(handler.eval_dir) / "full_splice_performance.tsv"
        if not auto_path.exists():
            parser.error("Baseline TSV not supplied and auto-detection failed – please provide path explicitly.")
        baseline_tsv_path = auto_path
        print(f"[demo] Auto-detected baseline TSV at {baseline_tsv_path}")
    else:
        baseline_tsv_path = Path(args.baseline_tsv)
    delta_df = trainer.compare_baseline(baseline_tsv_path)
    print(delta_df.sort_values("delta", ascending=False).head(15).to_string(index=False))

    # Save for later plotting
    delta_df.to_csv(out_dir / "delta_vs_baseline.csv", index=False)

    # ------------------------------------------------------------------
    # Feature importance                                                 
    # ------------------------------------------------------------------
    print("\n============= Feature Importance (top 20) =============")
    imp_df = trainer.feature_importance(max_samples=2000)
    print(imp_df.head(20).to_string(index=False))
    imp_df.to_csv(out_dir / "feature_importance_all.csv", index=False)

    print("\n[demo] Completed. Artifacts in", out_dir)


if __name__ == "__main__":  # pragma: no cover
    main(sys.argv[1:])
