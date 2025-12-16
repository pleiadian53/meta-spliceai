#!/usr/bin/env python
"""List genes of specific *gene_type* that are *not* present in a training dataset.

Typical use-case:
    1) Gather the protein-coding genes from the canonical `gene_features.tsv`.
    2) Subtract the genes that were used to train the 1000-gene meta-model.

Example
-------
$ conda run -n surveyor python -m meta_spliceai.splice_engine.meta_models.analysis.list_unseen_genes \
      --gene-features data/ensembl/spliceai_analysis/gene_features.tsv \
      --train-master train_pc_1000/master \
      --gene-type protein_coding \
      --limit 50
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path
import sys, pathlib
from typing import Sequence

import polars as pl
from rich.progress import track

# ------------------------------------------------------------------
# Allow invocation as a module while still resolving project-root imports
# ------------------------------------------------------------------
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
if (PROJECT_ROOT / "meta_spliceai").exists():
    sys.path.insert(0, str(PROJECT_ROOT))


def load_gene_universe(gene_features_tsv: Path, gene_types: Sequence[str]) -> set[str]:
    """Return the set of gene_ids whose *gene_type* is in *gene_types*."""
    # Only read the columns we need to avoid dtype inference issues on e.g. 'chrom'.
    df = pl.read_csv(
        gene_features_tsv,
        separator="\t",
        columns=["gene_id", "gene_type"],
    )
    sub = df.filter(pl.col("gene_type").is_in(gene_types))
    return set(sub["gene_id"].unique())


def load_training_genes(master_dir: Path) -> set[str]:
    """Return the set of gene_ids found in all Parquet shards in *master_dir*."""
    if not master_dir.is_dir():
        raise FileNotFoundError(master_dir)
    gene_ids: set[str] = set()
    parquet_paths = sorted(master_dir.glob("*.parquet"))
    if not parquet_paths:
        raise RuntimeError(f"No parquet files found in {master_dir}")

    for p in track(parquet_paths, description="Scanning training shards …"):
        # read only the column we need to keep memory usage minimal
        df = pl.read_parquet(p, columns=["gene_id"])
        gene_ids.update(df["gene_id"].unique())
    return gene_ids


def main() -> None:
    parser = argparse.ArgumentParser(description="List unseen genes of given type(s) vs. training dataset")
    parser.add_argument("--gene-features", required=True, type=Path, help="Path to gene_features.tsv")
    parser.add_argument("--train-master", required=True, type=Path, help="<train_root>/master directory with parquet shards")
    parser.add_argument("--gene-type", nargs="*", default=["protein_coding"], help="Gene types to include (default: protein_coding)")
    parser.add_argument("--limit", type=int, default=0, help="Print at most N gene IDs (0 = unlimited)")
    parser.add_argument("--output", type=Path, default=None, help="Optional path to write the gene list (one per line)")
    args = parser.parse_args()

    universe = load_gene_universe(args.gene_features, args.gene_type)
    training = load_training_genes(args.train_master)

    unseen = sorted(universe - training)
    if args.limit > 0:
        unseen = unseen[: args.limit]

    if args.output is not None:
        args.output.write_text("\n".join(unseen) + "\n")
        print(f"Wrote {len(unseen)} genes → {args.output}")
    else:
        print("\n".join(unseen))


if __name__ == "__main__":
    main()
