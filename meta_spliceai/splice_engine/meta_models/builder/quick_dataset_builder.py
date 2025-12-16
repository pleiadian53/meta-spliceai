"""Quick utility to assemble **temporary** meta-model training datasets.

This script is intended for *rapid iteration* while the full incremental
builder job is still running.  It lets you cherry-pick a set of genes—either by
explicit list or by scanning already-written `analysis_sequences_*` TSV
chunks—and run the standard `build_training_dataset` pipeline to produce a
Parquet file that mirrors the final schema but at much smaller scale.

The output files are placed in the existing *meta_models* directory with a
unique prefix so they do **not** overwrite the official datasets.
"""
from __future__ import annotations

import argparse
import glob
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Sequence

import polars as pl

from meta_spliceai.splice_engine.meta_models.builder.build_training_dataset import (
    build_meta_model_training_dataset,
)
from meta_spliceai.splice_engine.meta_models.io.handlers import MetaModelDataHandler

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _collect_genes_from_tsv(tsv_paths: Iterable[str], sample_n: int | None = None, verbose: int = 1) -> List[str]:
    """Return **unique** gene IDs present in one or more TSV chunks.

    Parameters
    ----------
    tsv_paths
        Iterable of file paths (supports glob expansion beforehand).
    sample_n
        If provided, randomly sample *sample_n* gene IDs from the full set.
    verbose
        Print a short summary.
    """
    genes: set[str] = set()
    for p in tsv_paths:
        # Use Polars' lazy CSV reader for speed and low memory
        s = (
            pl.scan_csv(p, separator="\t", has_header=True, comment_prefix=None)
            .select("gene_id")
            .unique()
            .collect()
            .to_series()
        )
        genes.update(s.to_list())
    gene_list = sorted(genes)
    if sample_n is not None and sample_n < len(gene_list):
        gene_list = random.sample(gene_list, sample_n)
    if verbose:
        print(f"[quick-builder] Collected {len(gene_list):,} unique genes from TSV chunks.")
    return gene_list


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------


def build_temp_dataset(
    *,
    eval_dir: str | Path | None = None,
    gene_ids: Sequence[str] | None = None,
    tsv_glob: str | None = None,
    sample_n: int | None = None,
    output_prefix: str | None = None,
    kmer_sizes: Sequence[int] | None = (6,),
    enrichers: Sequence[str] | None = None,
    batch_rows: int = 500_000,
    overwrite: bool = True,
    verbose: int = 1,
):
    """High-level convenience wrapper.

    At least one of *gene_ids* or *tsv_glob* must be supplied.
    """
    if eval_dir is None:
        # Resolve to default eval_dir used across MetaSpliceAI utilities
        eval_dir = MetaModelDataHandler().eval_dir
    if gene_ids is None and tsv_glob is None:
        raise ValueError("Supply either gene_ids or tsv_glob.")

    # ------------------------------------------------------------------
    # 1. Resolve gene ID list
    # ------------------------------------------------------------------
    if gene_ids is None:
        # If tsv_glob is a bare filename pattern, prepend meta_models path
        if os.sep not in tsv_glob.strip(os.sep):
            meta_dir = MetaModelDataHandler(eval_dir=eval_dir).meta_dir
            tsv_glob = str(Path(meta_dir) / tsv_glob)
        paths = glob.glob(tsv_glob)
        if not paths:
            raise FileNotFoundError(f"No files matched pattern: {tsv_glob}")
        gene_ids = _collect_genes_from_tsv(paths, sample_n=sample_n, verbose=verbose)
    else:
        gene_ids = list(gene_ids)
        if sample_n is not None and sample_n < len(gene_ids):
            gene_ids = random.sample(gene_ids, sample_n)

    # ------------------------------------------------------------------
    # 2. Unique output prefix to avoid clashes with official datasets
    # ------------------------------------------------------------------
    if output_prefix is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_prefix = f"temp_dataset_{len(gene_ids)}g_{ts}"

    # ------------------------------------------------------------------
    # 3. Delegate to the canonical builder
    # ------------------------------------------------------------------
    enriched_path = build_meta_model_training_dataset(
        eval_dir=str(eval_dir) if eval_dir else None,
        n_genes=len(gene_ids),  # ignored because subset_policy=custom
        subset_policy="custom",
        kmer_sizes=kmer_sizes,
        enrichers=enrichers,
        output_prefix=output_prefix,
        overwrite=overwrite,
        additional_gene_ids=gene_ids,
        run_workflow=False,  # we rely on already-generated TSVs
        batch_rows=batch_rows,
        verbose=verbose,
    )
    print(f"[quick-builder] Temporary training dataset ready: {enriched_path}")
    return enriched_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Assemble a temporary meta-model training dataset from existing TSV chunks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--eval-dir", type=str, help="Root spliceai_eval directory.")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--gene-ids-file", type=str, help="Plain text OR CSV/TSV file containing the gene list.")
    group.add_argument("--tsv-glob", type=str, help="Glob pattern to analysis_sequences TSV chunks.")
    p.add_argument("--gene-col", type=str, default="gene_id", help="Column name holding gene IDs when --gene-ids-file is a CSV/TSV.")
    p.add_argument("--sample-n", type=int, default=None, help="Randomly sample N genes from the collected set.")
    p.add_argument("--output-prefix", type=str, default=None, help="Custom basename for the Parquet outputs.")
    p.add_argument("--kmer-sizes", type=int, nargs="*", default=[6], help="K-mer sizes to extract; pass 0 to skip k-mer extraction.")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing output files of the same prefix.")
    p.add_argument("--batch-rows", type=int, default=500_000, help="Row buffer size fed to the Parquet writer; lower it to reduce RAM peak (default 500k).")
    p.add_argument("--verbose", "-v", action="count", default=1)
    return p.parse_args()


def _cli_entry() -> None:
    ns = _parse_cli()

    # Load IDs if file provided
    ids: List[str] | None = None
    if ns.gene_ids_file:
        # Try CSV/TSV first – fall back to simple line-by-line read
        try:
            sep = "\t" if ns.gene_ids_file.endswith(".tsv") else "," if ns.gene_ids_file.endswith(".csv") else None
            if sep:
                df_ids = pl.read_csv(ns.gene_ids_file, separator=sep)
                if ns.gene_col not in df_ids.columns:
                    raise KeyError
                ids = df_ids[ns.gene_col].drop_nulls().unique().to_list()
            else:
                raise FileNotFoundError  # force fallback
        except Exception:
            with open(ns.gene_ids_file) as fh:
                ids = [ln.strip() for ln in fh if ln.strip()]

    build_temp_dataset(
        eval_dir=ns.eval_dir,
        gene_ids=ids,
        tsv_glob=ns.tsv_glob,
        sample_n=ns.sample_n,
        output_prefix=ns.output_prefix,
        kmer_sizes=(None if ns.kmer_sizes == [0] else ns.kmer_sizes),
        overwrite=ns.overwrite,
        batch_rows=ns.batch_rows,
        verbose=ns.verbose,
    )


if __name__ == "__main__":
    _cli_entry()
