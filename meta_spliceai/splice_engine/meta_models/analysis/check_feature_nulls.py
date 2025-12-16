#!/usr/bin/env python
"""check_feature_nulls.py – report feature columns that contain null or empty-string values.

Usage (module path):
    python -m meta_spliceai.splice_engine.meta_models.analysis.check_feature_nulls \
        --parquet /path/to/dataset/dir_or_file \
        [--feature-list /path/to/feature_names.txt] \
        [--limit 50000]

If --feature-list is omitted, all columns in the parquet file are inspected.
The script prints a table of columns with at least one null/empty value and their counts.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import polars as pl


def _load_feature_list(path: Path) -> List[str]:
    """Load feature names from a text file (one per line) or JSON list."""
    if not path.exists():
        raise FileNotFoundError(f"Feature list file not found: {path}")

    if path.suffix == ".json":
        import json

        return json.loads(path.read_text())

    # Default: plain text, one feature per line
    return [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]


def main() -> None:  # noqa: D401
    parser = argparse.ArgumentParser(
        description="Check null / empty-string counts per feature column.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--parquet",
        required=True,
        help="Path to a Parquet file OR directory containing Parquet files.",
    )
    parser.add_argument(
        "--feature-list",
        help="Optional path to text / JSON file with feature names to inspect.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Number of rows to sample (0 = load all rows).",
    )

    args = parser.parse_args()
    pq_path = Path(args.parquet)

    if pq_path.is_dir():
        try:
            pq_path = next(pq_path.glob("*.parquet"))
        except StopIteration as exc:
            raise FileNotFoundError(
                f"No .parquet files found in directory {pq_path}"
            ) from exc

    n_rows = None if args.limit <= 0 else args.limit
    df = pl.read_parquet(pq_path, n_rows=n_rows)

    # Determine which columns to inspect ----------------------------------
    if args.feature_list:
        features = _load_feature_list(Path(args.feature_list))
        missing = [f for f in features if f not in df.columns]
        if missing:
            print(
                f"Warning: {len(missing)} features not present in dataframe: "
                f"{', '.join(missing[:10])}{' …' if len(missing) > 10 else ''}"
            )
        cols = [c for c in features if c in df.columns]
    else:
        cols = df.columns

    # Compute null / empty-string counts ----------------------------------
    exprs = []
    schema = df.schema
    for c in cols:
        cond = pl.col(c).is_null()
        if schema.get(c) == pl.Utf8:
            # Only string columns can be compared safely to ""
            cond = cond | (pl.col(c) == "")
        exprs.append(cond.sum().alias(c))

    counts = df.select(exprs)

    offenders = [(c, counts[0, c]) for c in cols if counts[0, c] > 0]

    if offenders:
        print(
            f"Found {len(offenders)} feature columns with null / empty values (sample size = {df.height:,} rows):"
        )
        for col, cnt in sorted(offenders, key=lambda x: -x[1]):
            print(f"{col:35s} {cnt}")
    else:
        print(
            f"✅ No null or empty-string values detected in the inspected columns (sample size = {df.height:,} rows)."
        )


if __name__ == "__main__":
    main()
