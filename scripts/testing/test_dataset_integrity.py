#!/usr/bin/env python
"""Regression check for meta-model training datasets.

Verifies that:
1. No *feature* columns listed in ``builder_utils._FEATURE_LEVEL`` contain nulls.
2. The ``n_splice_sites`` column exactly matches the authoritative counts
   derived from ``splice_sites.tsv`` for genes present in that table.

Usage
-----
    conda run -n surveyor python scripts/test_dataset_integrity.py <DATASET_DIR>

If DATASET_DIR is omitted, it defaults to `train_pc_1000` in the project root.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from meta_spliceai.splice_engine.meta_models.builder import builder_utils as bu

pl.Config.set_tbl_rows(10)


def load_master_df(root: Path) -> pl.DataFrame:
    """Load the full *master* Parquet dataset into a Polars DataFrame."""
    master_dir = root / "master"
    if not master_dir.exists():
        raise SystemExit(f"Master dataset dir not found: {master_dir}")

    # Efficiently concatenate via Polars scan to avoid memory spike
    lazy_frames = [pl.scan_parquet(p) for p in sorted(master_dir.glob("*.parquet"))]
    if not lazy_frames:
        raise SystemExit("No Parquet files found in master directory.")

    return pl.concat(lazy_frames).collect()


def check_nulls(df: pl.DataFrame) -> None:
    """Assert that required feature columns have *zero* null values."""
    feature_cols = [c for c in bu._FEATURE_LEVEL.keys() if c in df.columns]
    offending = {c: df[c].null_count() for c in feature_cols if df[c].null_count() > 0}
    if offending:
        msg_lines = ["[FAIL] Null values detected in feature columns:"]
        msg_lines += [f"  • {col}: {cnt} nulls" for col, cnt in offending.items()]
        raise AssertionError("\n".join(msg_lines))
    print("[PASS] No nulls found in critical feature columns.")


def check_n_splice_sites(df: pl.DataFrame) -> None:
    """Verify n_splice_sites agrees with splice_sites.tsv counts."""
    if "gene_id" not in df.columns or "n_splice_sites" not in df.columns:
        print("[WARN] gene_id or n_splice_sites column missing – skipping count verification.")
        return

    auth_df = bu._compute_n_splice_sites_df()

    merged = df.select(["gene_id", "n_splice_sites"]).join(
        auth_df, on="gene_id", how="inner", suffix="_auth"
    )
    mism = merged.filter(pl.col("n_splice_sites") != pl.col("n_splice_sites_auth"))
    if mism.height > 0:
        sample = mism.head(10)
        raise AssertionError(
            f"[FAIL] n_splice_sites mismatch for {mism.height} genes. Sample:\n{sample}"
        )
    print("[PASS] n_splice_sites matches authoritative splice_sites.tsv counts.")


def check_gene_types(df: pl.DataFrame, expected: str) -> None:
    """Verify that gene_type matches the expected value (e.g. 'protein_coding')."""
    if "gene_type" not in df.columns:
        print("[WARN] `gene_type` column not found, skipping check.")
        return

    # Genes with a type other than expected (and not 'unknown', which is a patchable state)
    violating_df = df.filter(
        (~pl.col("gene_type").eq(expected)) & (~pl.col("gene_type").eq("unknown"))
    )

    if violating_df.height > 0:
        sample = violating_df.select(["gene_id", "gene_type"]).unique().head(10)
        raise AssertionError(
            f"[FAIL] Found {violating_df.height} records with unexpected gene_type. Sample:\n{sample}"
        )
    print(f"[PASS] All gene_type values match expected value ('{expected}').")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run regression checks on a meta-model training dataset."
    )
    parser.add_argument(
        "dataset_dir",
        nargs="?",
        default="train_pc_1000",
        help="Path to the dataset directory (default: train_pc_1000)",
    )
    parser.add_argument(
        "--expected-gene-type",
        default="protein_coding",
        help="Expected dominant gene_type for diagnostics (default: protein_coding)",
    )
    args = parser.parse_args()
    root = Path(args.dataset_dir).expanduser().resolve()

    print("\n[step] Loading master dataset …")
    df = load_master_df(root)
    print(f"Loaded DataFrame with shape {df.shape}\n")

    print("[step] Checking nulls …")
    check_nulls(df)

    print("[step] Checking n_splice_sites integrity …")
    check_n_splice_sites(df)

    print("[step] Checking gene types …")
    check_gene_types(df, expected=args.expected_gene_type)

    print("\n[success] Dataset integrity checks passed.\n")


if __name__ == "__main__":
    main()
