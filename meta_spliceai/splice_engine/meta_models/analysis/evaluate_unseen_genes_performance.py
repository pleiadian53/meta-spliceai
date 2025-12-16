#!/usr/bin/env python
"""Evaluate meta-model generalisation on *unseen genes*.

A convenient wrapper that

1. Determines which genes of selected *gene_type* were **not** part of a
   training dataset (e.g. the 1 000-gene protein-coding training-set).
2. Samples *N* of these genes (optionally all) so that run-time stays tractable.
3. Runs the **enhanced splice inference workflow** restricted to those genes
   and applies a pre-trained meta-model to obtain predictions.
4. Outputs a small CSV / JSON summary with macro-F1, AUROC/AUPRC per class as
   computed by the demo inference helper.

The script is intentionally lightweight – it re-uses the existing
``demo_splice_inference`` implementation to avoid duplicating evaluation logic.

Example
-------
$ conda run -n surveyor python -m meta_spliceai.splice_engine.meta_models.analysis.evaluate_unseen_genes_performance \
      --model-run runs/gene_cv_sigmoid_2 \
      --train-master train_pc_1000/master \
      --train-schema-dir train_pc_1000 \
      --gene-features data/ensembl/spliceai_analysis/gene_features.tsv \
      --gene-type protein_coding \
      --n-genes 100 \
      --max-positions-per-gene 250
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import List, Sequence

from rich import print as rprint

from meta_spliceai.splice_engine.meta_models.analysis.list_unseen_genes import (
    load_gene_universe,
    load_training_genes,
)
from meta_spliceai.splice_engine.meta_models.workflows.demo_splice_inference import (
    demo as _run_demo,
)

# ---------------------------------------------------------------------------
# Helper – sample genes reproducibly
# ---------------------------------------------------------------------------

def _sample_genes(genes: Sequence[str], n: int, seed: int = 42) -> List[str]:
    if n <= 0 or n >= len(genes):
        return list(genes)
    rng = random.Random(seed)
    sample = list(genes)
    rng.shuffle(sample)
    return sample[:n]


# ---------------------------------------------------------------------------
# Main evaluation routine
# ---------------------------------------------------------------------------

def evaluate(
    model_run: Path,
    train_master: Path,
    train_schema_dir: Path | None,
    gene_features_tsv: Path,
    gene_types: Sequence[str] | None = None,
    n_genes: int = 100,
    max_positions_per_gene: int = 0,
    max_analysis_rows: int = 500_000,
    seed: int = 42,
    verbosity: int = 1,
) -> None:
    gene_types = list(gene_types or ["protein_coding"])

    if verbosity:
        rprint(f"[eval] Loading gene universe of types {gene_types}…")
    universe = load_gene_universe(gene_features_tsv, gene_types)

    if verbosity:
        rprint(f"[eval] Universe size = {len(universe):,} genes.")

    training_genes = load_training_genes(train_master)
    unseen_genes = sorted(universe - training_genes)

    if not unseen_genes:
        raise RuntimeError("No unseen genes found – check gene_types and training dataset")

    if verbosity:
        rprint(f"[eval] Unseen genes available = {len(unseen_genes):,}.")

    target_genes = _sample_genes(unseen_genes, n_genes, seed)
    if verbosity:
        rprint(f"[eval] Selected {len(target_genes):,} genes for evaluation.")

    # ------------------------------------------------------------------
    # Resolve schema dir default
    # ------------------------------------------------------------------
    if train_schema_dir is None:
        train_schema_dir = train_master.parent
        if verbosity:
            rprint(f"[eval] Using {train_schema_dir} as schema directory (derived from --train-master parent).")
    # ------------------------------------------------------------------
    # Delegate to demo inference helper
    # ------------------------------------------------------------------
    _run_demo(
        run_dir=model_run,
        eval_dir=Path.cwd(),  # placeholder – not currently used by demo
        target_genes=target_genes,
        max_positions_per_gene=max_positions_per_gene,
        max_analysis_rows=max_analysis_rows,
        train_schema_dir=train_schema_dir,
        verbosity=verbosity,
    )


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate meta-model on genes unseen during training.")
    p.add_argument("--model-run", required=True, type=Path, help="Path to directory containing the trained meta-model run (model_multiclass.*)")
    p.add_argument("--train-master", required=True, type=Path, help="<train_root>/master directory with Parquet shards used in training")
    p.add_argument("--train-schema-dir", type=Path, default=None,
                   help="Directory whose Parquet schema defines the canonical feature columns. Defaults to the *parent* directory of --train-master if omitted.")
    p.add_argument("--gene-features", required=True, type=Path, help="gene_features.tsv path with gene_type column")
    p.add_argument("--gene-type", nargs="*", default=["protein_coding"], help="Gene type(s) to include in evaluation (default: protein_coding)")
    p.add_argument("--n-genes", type=int, default=100, help="Number of unseen genes to sample (<=0 ⇒ all)")
    p.add_argument("--max-positions-per-gene", type=int, default=0, help="Cap on least-confident positions per gene (0 = unlimited)")
    p.add_argument("--max-analysis-rows", type=int, default=500_000, help="Global cap on total positions analysed (0 = unlimited)")
    p.add_argument("--seed", type=int, default=42, help="Random seed for gene sampling")
    p.add_argument("-v", "--verbose", action="count", default=1, help="Increase verbosity")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:  # pragma: no cover – CLI only
    ns = _parse_args(argv or sys.argv[1:])
    evaluate(
        model_run=ns.model_run.expanduser(),
        train_master=ns.train_master.expanduser(),
        train_schema_dir=(ns.train_schema_dir.expanduser() if ns.train_schema_dir is not None else None),
        gene_features_tsv=ns.gene_features.expanduser(),
        gene_types=ns.gene_type,
        n_genes=ns.n_genes,
        max_positions_per_gene=ns.max_positions_per_gene,
        max_analysis_rows=ns.max_analysis_rows,
        seed=ns.seed,
        verbosity=ns.verbose,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
