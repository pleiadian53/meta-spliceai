from __future__ import annotations

"""Incremental Meta-Model Training Dataset Builder
=======================================================
Builds the full-genome meta-model training dataset in small, memory-friendly
batches.  Each batch follows the build → enrich → downsample → append pattern
so that enormous numbers of *easy* true-negative rows never need to be
materialised at once.

High-level algorithm
--------------------
1. Determine the list of target genes (via ``subset_analysis_sequences`` or a
   user-supplied list).
2. Split the gene list into fixed-size batches.
3. For each batch:
   a. Create a *raw* Parquet with k-mer & mandatory base features.
   b. Enrich it with additional gene / performance / overlap features.
   c. Down-sample the true negatives to balance the dataset.
   d. Append (partitioned by chromosome) to a master Arrow dataset directory.

The master directory can later be memory-mapped by most ML libraries and can be
incrementally extended or re-written on failure recovery.
"""

from pathlib import Path
from typing import Iterable, List, Sequence, Optional, Tuple, Dict
import os
import itertools

import polars as pl
import pyarrow.dataset as ds
import pyarrow as pa
import pyarrow.parquet as pq

from meta_spliceai.splice_engine.meta_models.io.handlers import (
    MetaModelDataHandler,
)
from meta_spliceai.splice_engine.meta_models.features.gene_selection import (
    subset_analysis_sequences,
    subset_positions_dataframe,
)
from meta_spliceai.splice_engine.meta_models.builder.dataset_builder import (
    build_training_dataset,
)
from meta_spliceai.splice_engine.meta_models.features.feature_enrichment import (
    apply_feature_enrichers,
)
from meta_spliceai.splice_engine.meta_models.builder.downsample_tn import (
    downsample_tn,
)
from meta_spliceai.splice_engine.meta_models.builder import builder_utils
from meta_spliceai.splice_engine.meta_models.analysis.shared_analysis_utils import (
    check_genomic_files_exist,
)
from meta_spliceai.splice_engine.meta_models.workflows.splice_prediction_workflow import (
    run_enhanced_splice_prediction_workflow,
)

__all__ = [
    "build_base_dataset",
    "incremental_build_training_dataset",
]

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _chunks(seq: Sequence[str], size: int) -> Iterable[List[str]]:
    """Yield successive *size*-chunk lists from *seq*."""
    for i in range(0, len(seq), size):
        yield list(seq[i : i + size])

# ---------------------------------------------------------------------------
# Stage 1 – build + enrich per-batch dataset
# ---------------------------------------------------------------------------

def build_base_dataset(
    gene_ids: Sequence[str],
    output_path: Path | str,
    *,
    data_handler: MetaModelDataHandler,
    kmer_sizes: Sequence[int] | None = (6,),
    enrichers: Sequence[str] | None = None,
    batch_rows: int = 500_000,
    overwrite: bool = False,
    verbose: int = 1,
) -> Path:
    """Create a *single-batch* enriched Parquet training dataset.

    Parameters
    ----------
    gene_ids
        List of Ensembl gene IDs to include in this batch.
    output_path
        Destination Parquet (enriched, *before* TN down-sampling).
    data_handler
        Gives path access to the ``*_analysis_sequences_*.tsv`` files.
    kmer_sizes, enrichers, batch_rows, overwrite, verbose
        Behaviour mirrors the arguments from previous utilities.
    """

    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"{output_path} exists – use --overwrite to regenerate")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Temporary location for the raw k-mer dataset (removed afterwards)
    tmp_path = output_path.with_suffix(".tmp.parquet")
    if tmp_path.exists():
        tmp_path.unlink()

    # ---------------------------------------------------------------------
    # 1) Assemble k-mer table ------------------------------------------------
    # ---------------------------------------------------------------------
    build_training_dataset(
        analysis_tsv_dir=str(Path(data_handler.meta_dir)),
        output_path=str(tmp_path),
        mode="none",
        target_genes=list(gene_ids),
        top_n_genes=None,
        kmer_sizes=list(kmer_sizes) if kmer_sizes else None,
        batch_rows=batch_rows,
        keep_sequence=False,
        verbose=max(0, verbose - 1),
    )

    # ---------------------------------------------------------------------
    # 2) Feature enrichment --------------------------------------------------
    # ---------------------------------------------------------------------
    df = pl.read_parquet(tmp_path)
    df_enriched = apply_feature_enrichers(
        df,
        enrichers=enrichers,
        verbose=max(0, verbose - 1),
        fa=None,
        sa=None,
    )
    df_enriched.write_parquet(output_path, compression="zstd")

    # Clean up temp file to save disk space
    tmp_path.unlink(missing_ok=True)

    if verbose:
        print(f"      • built + enriched dataset → {output_path} ({df_enriched.height:,} rows)")
    return output_path

# ---------------------------------------------------------------------------
# Stage 2 – orchestrate incremental build over many batches
# ---------------------------------------------------------------------------

def incremental_build_training_dataset(
    *,
    eval_dir: str | None = None,
    output_dir: str | Path = "train_dataset_trimmed",
    n_genes: int = 20_000,
    subset_policy: str = "error_total",
    batch_size: int = 1_000,
    kmer_sizes: Sequence[int] | None = (6,),
    enrichers: Sequence[str] | None = None,
    downsample_kwargs: Optional[Dict] = None,
    batch_rows: int = 500_000,
    gene_types: Optional[Sequence[str]] = None,
    run_workflow: bool = False,
    workflow_kwargs: Optional[Dict] = None,
    overwrite: bool = False,
    verbose: int = 1,
) -> Path:
    """Full incremental build pipeline.

    Parameters
    ----------
    eval_dir
        Root *spliceai_eval* directory (auto-detected if *None*).
    output_dir
        Directory that will contain per-batch Parquets and the *master* dataset.
    n_genes, subset_policy
        Controls which genes are selected (mirrors existing API).
    batch_size
        Number of genes per batch.
    kmer_sizes, enrichers
        Feature extraction parameters.
    downsample_kwargs
        Passed straight to ``downsample_tn`` (e.g. ``dict(hard_prob_thresh=0.15)``).
    overwrite
        If *True*, regenerate existing batch / master artefacts.

    Returns
    -------
    Path
        Path to the *master* partitioned dataset directory.
    """

    output_dir = Path(output_dir)
    batch_dir = output_dir
    master_dir = output_dir / "master"
    batch_dir.mkdir(parents=True, exist_ok=True)
    master_dir.mkdir(parents=True, exist_ok=True)

    # Optionally run the upstream prediction workflow ------------------
    if run_workflow:
        if verbose:
            print("[incremental-builder] Running enhanced prediction workflow …")

        _wk = dict(workflow_kwargs or {})  # copy so we can mutate safely

        # ------------------------------------------------------------------
        # Optional: restrict prediction to specific gene types
        # ------------------------------------------------------------------
        # --------------------------------------------------------------
        # Determine whether genomic files need extraction -------------
        # --------------------------------------------------------------
        if "do_extract_sequences" not in _wk and "do_extract_splice_sites" not in _wk:
            existing_files = check_genomic_files_exist()
            _wk["do_extract_sequences"] = not existing_files.get("genomic_sequences", False)
            _wk["do_extract_annotations"] = not existing_files.get("annotations", False)
            _wk["do_extract_splice_sites"] = not existing_files.get("splice_sites", False)
            _wk["do_find_overlaping_genes"] = True  # safe default
            if verbose:
                print(
                    "[incremental-builder] Genomic file status → extract_sequences=%s, annotations=%s, splice_sites=%s"
                    % (
                        _wk["do_extract_sequences"],
                        _wk["do_extract_annotations"],
                        _wk["do_extract_splice_sites"],
                    )
                )

        # ------------------------------------------------------------------
        # Add gene-type derived target_genes if requested -------------------
        # ------------------------------------------------------------------
        if gene_types is not None and "target_genes" not in _wk:
            try:
                # Derive gene_id list for requested gene types using gene_features.tsv
                from meta_spliceai.system.config import Config

                gf_path = os.path.join(
                    Config.DATA_DIR,
                    "ensembl",
                    "spliceai_analysis",
                    "gene_features.tsv",
                )
            except Exception:
                home = os.environ.get("HOME", "")
                gf_path = os.path.join(
                    home,
                    "work",
                    "splice-surveyor",
                    "data",
                    "ensembl",
                    "spliceai_analysis",
                    "gene_features.tsv",
                )

            if not os.path.exists(gf_path):
                raise FileNotFoundError(
                    "gene_features.tsv not found – cannot infer gene IDs for gene_types filter. "
                    "Either generate the gene features table or omit --gene-types when --run-workflow is used."
                )

            _gf = pl.read_csv(gf_path, separator="\t", schema_overrides={"chrom": pl.Utf8})
            gene_id_list = (
                _gf.filter(pl.col("gene_type").is_in(list(gene_types)))
                .select("gene_id")
                .to_series()
                .to_list()
            )
            if verbose:
                print(
                    f"[incremental-builder] Passing {len(gene_id_list):,} target genes to prediction workflow (gene types filter)."
                )
            _wk["target_genes"] = gene_id_list

        run_enhanced_splice_prediction_workflow(verbosity=max(0, verbose - 1), **_wk)

    # ------------------------------------------------------------------
    # Gene selection ----------------------------------------------------
    # ------------------------------------------------------------------
    dh = MetaModelDataHandler(eval_dir=eval_dir)

    if gene_types is None:
        # Use standard helper (faster because it streams minimal columns)
        _, all_gene_ids = subset_analysis_sequences(
            data_handler=dh,
            n_genes=n_genes,
            subset_policy=subset_policy,
            aggregated=True,
            additional_gene_ids=None,
            use_effective_counts=True,
            verbose=max(0, verbose - 1),
        )
    else:
        # Need gene-type filtering → operate on full positions_df
        pos_df = dh.load_splice_positions(aggregated=True)
        _, all_gene_ids = subset_positions_dataframe(
            pos_df,
            n_genes=n_genes,
            subset_policy=subset_policy,
            gene_types=list(gene_types),
            additional_gene_ids=None,
            use_effective_counts=True,
            verbose=max(0, verbose - 1),
        )
    # ------------------------------------------------------------------
    # Quick verification (optional) -------------------------------------
    # ------------------------------------------------------------------
    try:
        builder_utils.verify_gene_selection(
            dh,
            gene_ids=all_gene_ids,
            expected_gene_types=gene_types,
            raise_error=True,
            verbose=max(0, verbose - 1),
        )
    except Exception as e:
        # Fail fast so that errors are obvious during incremental builds
        raise

    if verbose:
        print(f"[incremental-builder] Selected {len(all_gene_ids)} genes via policy '{subset_policy}'.")

    batches = _chunks(all_gene_ids, batch_size)
    ds_kwargs = dict(format="parquet", partitioning="hive", existing_data_behavior="overwrite_or_ignore")
    downsample_kwargs = downsample_kwargs or {}

    for batch_ix, gene_batch in enumerate(batches, 1):
        prefix = f"batch_{batch_ix:05d}"
        raw_path = batch_dir / f"{prefix}_raw.parquet"
        trim_path = batch_dir / f"{prefix}_trim.parquet"

        if trim_path.exists() and not overwrite:
            if verbose:
                print(f"[{prefix}] trim Parquet exists - skipping …")
            # Still append to master (idempotent thanks to overwrite_or_ignore)
            ds.write_dataset(ds.dataset(trim_path), base_dir=master_dir, **ds_kwargs)
            continue

        if verbose:
            print(f"[{prefix}] processing {len(gene_batch)} genes …")

        # 1. Build + enrich -------------------------------------------------
        build_base_dataset(
            gene_ids=gene_batch,
            output_path=raw_path,
            data_handler=dh,
            kmer_sizes=kmer_sizes,
            enrichers=enrichers,
            batch_rows=batch_rows,
            overwrite=overwrite,
            verbose=verbose,
        )

        # 2. Down-sample TNs ----------------------------------------------
        if verbose:
            print(f"[{prefix}] down-sampling TNs …")
        _ = downsample_tn(raw_path, trim_path, **downsample_kwargs)

        # Optionally remove the raw Parquet to save disk after trimming
        raw_path.unlink(missing_ok=True)

        # 3 Append to master ---------------------------------------------
        if verbose:
            print(f"[{prefix}] appending to master dataset …")
        ds.write_dataset(ds.dataset(trim_path), base_dir=master_dir, **ds_kwargs)

    if verbose:
        print(f"[incremental-builder] Completed.  Master dataset at → {master_dir}")
    return master_dir

# ---------------------------------------------------------------------------
# CLI entry-point -----------------------------------------------------------
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse, sys, json

    p = argparse.ArgumentParser(
        description="Incrementally build the meta-model training dataset in gene batches.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--n-genes", 
        type=int, 
        default=20_000, 
        help="Total number of genes to include in the training dataset. Lower values (e.g., 1000) recommended for testing."
    )
    p.add_argument(
        "--subset-policy", 
        type=str, 
        default="error_total", 
        help="Gene selection strategy. Valid options: 'error_total' (genes with most errors), 'error_fp' (most false positives), "
             "'error_fn' (most false negatives), 'random' (random sampling), 'custom' (use provided gene ids)."
    )
    p.add_argument(
        "--batch-size", 
        type=int, 
        default=1_000, 
        help="Number of genes to process in each batch. Lower values reduce memory usage but increase processing time."
    )
    p.add_argument(
        "--output-dir", 
        type=str, 
        default="train_dataset_trimmed", 
        help="Output directory for the trimmed training dataset. Can be absolute or relative to working directory."
    )
    p.add_argument(
        "--overwrite", 
        action="store_true", 
        help="Overwrite existing artefacts. If not set, will attempt to resume previous run."
    )
    p.add_argument(
        "--run-workflow", 
        action="store_true", 
        help="Run the enhanced splice prediction workflow first to generate required prediction files before dataset building."
    )
    p.add_argument(
        "--workflow-kwargs", 
        type=str, 
        default="{}", 
        help="JSON dict with kwargs for the prediction workflow. Common options: {'eval_dir': '/path/to/dir', "
             "'gene_types': ['protein_coding'], 'target_genes': ['STMN2', 'UNC13A']}."
    )
    p.add_argument(
        "--gene-types", 
        type=str, 
        nargs="*", 
        default=None, 
        help="Restrict gene selection to these gene types. Common values: 'protein_coding', 'lncRNA', 'pseudogene', 'miRNA', etc."
    )
    p.add_argument(
        "--hard-prob", 
        type=float, 
        default=0.15, 
        help="TN down-sampling hard_prob_thresh: probability threshold to identify 'hard' negatives that will be preserved."
    )
    p.add_argument(
        "--window", 
        type=int, 
        default=75, 
        help="TN down-sampling window_nt: nucleotide window around true positives to preserve as 'neighborhood' negatives."
    )
    p.add_argument(
        "--easy-ratio", 
        type=float, 
        default=0.5, 
        help="TN down-sampling easy_neg_ratio: fraction of 'easy' negatives to randomly keep after preserving hard/neighborhood negatives."
    )
    p.add_argument(
        "--verbose", 
        "-v", 
        action="count", 
        default=1, 
        help="Increase verbosity. Use -v for standard output, -vv for detailed output, -vvv for debug."
    )
    args = p.parse_args(sys.argv[1:])

    try:
        wk = json.loads(args.workflow_kwargs)
    except json.JSONDecodeError as e:
        print(f"Invalid JSON for --workflow-kwargs: {e}", file=sys.stderr)
        sys.exit(1)

    ds_kwargs = dict(
        hard_prob_thresh=args.hard_prob,
        window_nt=args.window,
        easy_neg_ratio=args.easy_ratio,
    )

    incremental_build_training_dataset(
        n_genes=args.n_genes,
        subset_policy=args.subset_policy,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
        downsample_kwargs=ds_kwargs,
        gene_types=args.gene_types,
        run_workflow=args.run_workflow,
        workflow_kwargs=wk,
        verbose=args.verbose,
    )
