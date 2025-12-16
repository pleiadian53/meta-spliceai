"""Extract unique gene IDs, transcript IDs, and gene–transcript pairs
from one or more `analysis_sequences_*` TSV chunks.

The resulting file `gene_tx_candidates.tsv` contains three blocks so that it can
be re-used directly by other helper scripts (e.g. as input to
`quick_dataset_builder`).

Usage
-----
$ python -m meta_spliceai.splice_engine.meta_models.examples.extract_gene_tx_ids \
    "analysis_sequences_6_chunk_2001_2500.tsv" -o gene_tx_candidates.tsv

$ python -m ...extract_gene_tx_ids "analysis_sequences_*_chunk_*.tsv" -o list.tsv

The script is fast because it uses **Polars**' lazy CSV reader and operates on a
minimal set of columns.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path
from typing import Iterable, List, Optional

import polars as pl
from tqdm import tqdm  # progress bar
from meta_spliceai.splice_engine.meta_models.io.handlers import MetaModelDataHandler

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TSV_EXTS = (".tsv", ".tsv.gz")
_CSV_EXTS = (".csv", ".csv.gz")

def _detect_sep(path: str) -> str:
    """Return the delimiter based on *path* extension."""
    lower = path.lower()
    if lower.endswith(_TSV_EXTS):
        return "\t"
    if lower.endswith(_CSV_EXTS):
        return ","
    # default to tab – all repo artefacts are TSV
    return "\t"


def _paths_from_patterns(patterns: Iterable[str], base_dir: Optional[str] = None) -> List[str]:
    paths: List[str] = []
    for pat in patterns:
        # If pat is relative (no path separator) and base_dir provided, prepend base_dir
        if base_dir and os.sep not in pat and not pat.startswith("."):
            pat_full = os.path.join(base_dir, pat)
        else:
            pat_full = pat
        expanded = glob.glob(pat_full)
        paths.extend(expanded)
    unique_paths = sorted(set(paths))
    if not unique_paths:
        sys.exit("No files matched the provided pattern(s).")
    return unique_paths


def _load_unique(paths: List[str], column: str) -> pl.Series:
    """Return a Polars Series of unique values for *column* across *paths*."""
    lazy_frames = []
    for p in tqdm(paths, desc=f"Collect {column}", unit="file", leave=False):
        sep = _detect_sep(p)
        if column not in ("gene_id", "transcript_id"):
            raise ValueError("column must be gene_id or transcript_id")
        # Some files (e.g. splice_positions_enhanced) may lack transcript_id
        try:
            lazy_frames.append(pl.scan_csv(p, separator=sep).select(column))
        except pl.ColumnNotFoundError:
            continue
    if not lazy_frames:
        return pl.Series(name=column, values=[])
    df = pl.concat(lazy_frames).unique().collect()  # convert LazyFrame -> DataFrame
    return df[column]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_gene_tx_ids(
    patterns: List[str] | None = None,
    *,
    eval_dir: str | None = None,
    n_genes: int | None = None,
    seed: int = 42,
):
    """Return (genes_series, transcripts_series, pair_df) for *paths*.

    *paths* should be a list of files matching analysis_sequences or
    splice_positions TSV/CSV chunks.
    """
    # Determine base directory for meta models
    # Always resolve meta_models base dir via MetaModelDataHandler (falls back to Analyzer.eval_dir)
    base_dir = MetaModelDataHandler(eval_dir=eval_dir).meta_dir
    # Default patterns if none supplied
    if not patterns:
        patterns = ["analysis_sequences_*.tsv", "splice_positions_enhanced_*.tsv"]
    paths = _paths_from_patterns(patterns, base_dir=base_dir)
    genes = _load_unique(paths, "gene_id").sort()
    txs = _load_unique(paths, "transcript_id").sort()

    original_gene_count = len(genes)
    # Warn if requested more genes than available ------------------------------------
    if n_genes is not None and n_genes > original_gene_count:
        print(
            f"[extract_gene_tx_ids] WARNING: Requested --n-genes {n_genes} but only "
            f"{original_gene_count} genes are available. Using all available genes."
        )
        n_genes = None

    # ------------------------------------------------------------------
    # Optional random subsampling of genes ------------------------------
    # ------------------------------------------------------------------
    if n_genes is not None and n_genes < original_gene_count:
        import random

        rng = random.Random(seed)
        sampled_genes = rng.sample(genes.to_list(), n_genes)
        sampled_genes_set = set(sampled_genes)
        genes = pl.Series("gene_id", sorted(sampled_genes))
    else:
        sampled_genes_set = set(genes.to_list())
        # No subsampling requested

    pair_frames = []
    for p in tqdm(paths, desc="Collect gene–transcript pairs", unit="file"):
        sep = _detect_sep(p)
        try:
            pair_frames.append(pl.scan_csv(p, separator=sep).select(["gene_id", "transcript_id"]))
        except pl.ColumnNotFoundError:
            continue
    if pair_frames:
        pairs = (
            pl.concat(pair_frames)
            .drop_nulls(["transcript_id"])
            .unique()
            .sort(["gene_id", "transcript_id"])
            .collect()
        )
    else:
        pairs = pl.DataFrame({"gene_id": [], "transcript_id": []})

    # If subsampled genes, filter pairs and transcripts accordingly
    if n_genes is not None:
        if pairs.height > 0:
            pairs = pairs.filter(pl.col("gene_id").is_in(sampled_genes_set))
        if pairs.height > 0:
            txs = pairs["transcript_id"].unique().sort()
        else:
            txs = pl.Series(name="transcript_id", values=[])

    return genes, txs, pairs

# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Extract unique gene / transcript IDs from analysis_sequences TSV chunks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("patterns", nargs="*", help="Glob patterns (relative or absolute) to analysis_sequences or splice_positions chunks. If omitted, scans all default chunks in --eval-dir's meta_models directory.")
    ap.add_argument("--eval-dir", type=str, default=None, help="Root spliceai_eval directory (to locate meta_models). If omitted, patterns must be absolute or cwd-relative.")
    ap.add_argument("-o", "--out", default="gene_tx_candidates.tsv", help="Output TSV file path.")
    ap.add_argument(
        "-n",
        "--n-genes",
        type=int,
        default=None,
        help="Randomly subsample to N genes if more are present in the artifacts.",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for gene subsampling (default: 42).",
    )
    args = ap.parse_args()

    genes, txs, df_pairs = extract_gene_tx_ids(
        args.patterns, eval_dir=args.eval_dir, n_genes=args.n_genes, seed=args.seed
    )

    out_path = Path(args.out)
    with out_path.open("w") as fh:
        fh.write("type\tid\tgene_id\ttranscript_id\n")
        for g in genes:
            fh.write(f"gene\t{g}\t{g}\t\n")
        for t in txs:
            fh.write(f"transcript\t{t}\t\t{t}\n")
        for row in df_pairs.iter_rows():
            fh.write(f"pair\t{row[0]}|{row[1]}\t{row[0]}\t{row[1]}\n")

    print(
        f"[extract] Wrote {out_path} – {genes.len()} genes, {txs.len()} transcripts, {df_pairs.height} pairs." )


if __name__ == "__main__":
    main()
