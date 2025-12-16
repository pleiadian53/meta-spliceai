"""Utility script to append k-mer feature columns to an existing Parquet
training dataset produced by *incremental_builder* / *quick_dataset_builder*
when it was created with ``--kmer-sizes 0``.

The script streams the input Parquet file in small record batches to keep
RAM usage low, computes k-mer features via ``make_kmer_features`` and writes
out a new Parquet file.  It therefore avoids the very high peak memory that
occurs when k-mer generation is performed on the full raw dataset.

Example
-------
$ python -m meta_spliceai.splice_engine.meta_models.builder.add_kmers \
        train_pc_1000_trimmed.parquet \
        --kmer-sizes 6 \
        --batch-rows 20000 -v
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence, Optional

import pyarrow as pa
import pyarrow.parquet as pq

from meta_spliceai.splice_engine.meta_models.features.kmer_features import (
    make_kmer_features,
)

DEFAULT_BATCH_ROWS = 50_000


def add_kmer_features_to_parquet(
    input_path: str | Path,
    output_path: str | Path | None = None,
    *,
    kmer_sizes: Optional[Sequence[int]] = (6,),
    batch_rows: int = DEFAULT_BATCH_ROWS,
    keep_sequence: bool = False,
    compression: str | None = "zstd",
    overwrite: bool = False,
    verbose: int = 1,
) -> Path:
    """Stream *input_path* and write k-mer-augmented dataset to *output_path*.

    Parameters
    ----------
    input_path, output_path
        Parquet paths.  If *output_path* is None the file will be written next
        to *input_path* with a ``_kmers`` suffix.
    kmer_sizes
        Iterable of integers (e.g. ``(4, 6)``).  If ``None`` or empty, the
        function simply copies *input_path* to *output_path*.
    batch_rows
        Number of rows to load into pandas at once for k-mer extraction.
    keep_sequence
        If False the raw "sequence" column is dropped after features are
        generated.
    overwrite
        Whether an existing *output_path* may be replaced.
    verbose
        ``0`` – silent, ``1`` – progress per batch, ``2`` – additional info.
    Returns
    -------
    Path to the written Parquet file.
    """

    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    if output_path is None:
        output_path = input_path.with_name(f"{input_path.stem}_kmers.parquet")
    output_path = Path(output_path)

    if output_path.exists():
        if overwrite:
            output_path.unlink()
        else:
            raise FileExistsError(
                f"Output Parquet {output_path} exists.  Use --overwrite to replace."
            )

    # Handle pass-through (copy) mode ---------------------------------------
    if not kmer_sizes or kmer_sizes == [0]:
        if verbose:
            print("[add-kmers] No k-mer sizes provided; copying file …")
        pq.copy_file(str(input_path), str(output_path))
        return output_path

    pf = pq.ParquetFile(input_path)
    total_rows = pf.metadata.num_rows
    if verbose:
        sizes_txt = ", ".join(map(str, kmer_sizes))
        print(
            f"[add-kmers] Appending k-mer features (k = {sizes_txt}) to "
            f"{total_rows:,} rows from {input_path} …"
        )

    writer: pq.ParquetWriter | None = None
    rows_written = 0

    for batch_idx, record_batch in enumerate(pf.iter_batches(batch_size=batch_rows), start=1):
        # Arrow RecordBatch -> pandas for k-mer feature generation ------------
        pd_batch = record_batch.to_pandas()
        pd_batch, _ = make_kmer_features(
            pd_batch,
            kmer_sizes=kmer_sizes,
            return_feature_set=True,
            verbose=0,
        )
        if not keep_sequence and "sequence" in pd_batch.columns:
            pd_batch = pd_batch.drop(columns=["sequence"])

        # Back to Arrow and write -------------------------------------------
        table = pa.Table.from_pandas(pd_batch, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(output_path, table.schema, compression=compression)
        writer.write_table(table)
        rows_written += table.num_rows

        if verbose:
            pct = rows_written / total_rows * 100
            print(
                f"  • wrote batch {batch_idx} with {table.num_rows:,} rows "
                f"(total {rows_written:,}/{total_rows:,}; {pct:5.1f} %)"
            )

    if writer is not None:
        writer.close()
    if verbose:
        print(f"[add-kmers] Completed: {output_path} written.")
    return output_path


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Append k-mer features to an existing training Parquet dataset.")
    p.add_argument("input", help="Input Parquet path (must contain 'sequence' column).")
    p.add_argument("--output", "-o", help="Output Parquet path. Defaults to '<input>_kmers.parquet'.")
    p.add_argument(
        "--kmer-sizes",
        type=int,
        nargs="*",
        default=[6],
        help="One or more k-mer sizes (integers). Use 0 to skip and just copy.",
    )
    p.add_argument(
        "--batch-rows",
        type=int,
        default=DEFAULT_BATCH_ROWS,
        help="Number of rows to process per batch (controls RAM).",
    )
    p.add_argument(
        "--keep-sequence",
        action="store_true",
        help="Keep the raw 'sequence' column after k-mer extraction (default: drop).",
    )
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing output file if present.")
    p.add_argument("-v", "--verbose", action="count", default=1, help="Increase verbosity.")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None):
    args = _parse_args(argv)
    add_kmer_features_to_parquet(
        input_path=args.input,
        output_path=args.output,
        kmer_sizes=(None if args.kmer_sizes == [0] else args.kmer_sizes),
        batch_rows=args.batch_rows,
        keep_sequence=args.keep_sequence,
        overwrite=args.overwrite,
        verbose=args.verbose,
    )


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
