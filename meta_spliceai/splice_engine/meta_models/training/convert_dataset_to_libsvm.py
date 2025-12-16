#!/usr/bin/env python3
"""Command-line wrapper for converting a featurised Parquet dataset to a single
LibSVM file suitable for XGBoost **external-memory** training.

Internally delegates to
`meta_spliceai.splice_engine.meta_models.training.external_memory_utils.convert_dataset_to_libsvm`.

Example
-------
>>> python -m meta_spliceai.splice_engine.meta_models.training.convert_dataset_to_libsvm \
        --dataset train_pc_1000/master \
        --out     train_pc_1000/master/global.libsvm
"""
from __future__ import annotations

import argparse
from pathlib import Path

from meta_spliceai.splice_engine.meta_models.training.external_memory_utils import (
    convert_dataset_to_libsvm as _convert,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:  # pragma: no cover
    p = argparse.ArgumentParser(description="Convert Parquet dataset to LibSVM for external-memory XGBoost.")
    p.add_argument("--dataset", required=True, help="Dataset directory or single Parquet file")
    p.add_argument("--out", required=True, help="Output .libsvm filename (will also write .features.json)")
    p.add_argument("--chunk-rows", type=int, default=250_000, help="Chunk rows within large shards (memory bound)")
    p.add_argument("--verbose", type=int, default=1, choices=[0, 1], help="Verbosity level")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:  # pragma: no cover
    args = _parse_args(argv)
    out_path = _convert(
        args.dataset,
        args.out,
        chunk_rows=args.chunk_rows,
        verbose=args.verbose,
    )
    print("[convert_dataset_to_libsvm] Done →", out_path)


if __name__ == "__main__":  # pragma: no cover
    main()
