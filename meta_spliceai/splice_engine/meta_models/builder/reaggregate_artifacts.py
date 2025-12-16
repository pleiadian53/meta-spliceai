"""Re-aggregate full splice-site artefacts from chunk-level files.

This utility is intended for recovery scenarios where the canonical
aggregated TSVs (e.g. ``full_splice_positions_enhanced.tsv``) were
accidentally overwritten.  It reloads all chunk files that remain on
disk and concatenates them back into the full dataset.

Example
-------
$ python -m meta_spliceai.splice_engine.meta_models.builder.reaggregate_artifacts \
    --eval-dir /path/to/data/ensembl/spliceai_eval \
    --output-subdir meta_models

The script will locate e.g. ``splice_positions_enhanced_chr*.tsv``
files under ``<eval_dir>/<output_subdir>/`` and rebuild the corresponding
``full_splice_positions_enhanced.tsv`` and ``full_splice_errors.tsv``.

It is safe to run multiple times; existing full files will be
overwritten **only if** you pass ``--overwrite``.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import polars as pl
from typing import Optional

from meta_spliceai.splice_engine.meta_models.io.handlers import (
    MetaModelDataHandler,
)


def reaggregate(
    eval_dir: str,
    output_subdir: Optional[str] = None,
    enhanced: bool = True,
    overwrite: bool = False,
    verbosity: int = 1,
) -> None:
    """Recreate aggregated splice-site artefacts.

    Parameters
    ----------
    eval_dir
        Root *spliceai_eval* directory (default: ``Analyzer.eval_dir``).
    output_subdir
        The sub-directory that hosts the chunk files (``meta_models`` by
        default).  Use the same value that was active when the chunks were
        generated.  For inference runs this might be ``inference``.
    enhanced
        If *True* expect the three-probability enhanced artefacts.
    overwrite
        Overwrite existing full TSVs if they already exist.
    verbosity
        0 = silent, 1 = info, 2 = verbose diagnostics.
    """
    dh = MetaModelDataHandler(eval_dir)

    # Resolve the directory that holds the chunk files.
    root_dir = Path(dh._get_output_dir(None if output_subdir in (None, "", "meta_models") else output_subdir))

    subject_pos = "splice_positions_enhanced" if enhanced else "splice_positions"
    subject_err = "splice_errors"

    # Collect chunk-level splice positions.
    if verbosity:
        print("[reaggregate] Scanning for chunk-level splice positions …")
    pos_paths = sorted(root_dir.glob(f"{subject_pos}_*_chunk_*.tsv"))
    pos_paths += sorted(root_dir.glob(f"{subject_pos}_*_chunk_*.parquet"))
    if not pos_paths:
        raise RuntimeError("No splice-position chunk files found in " + str(root_dir))

    pos_dfs = []
    all_pos_cols: set[str] = set()
    try:
        from tqdm import tqdm
        _iter_pos = tqdm(pos_paths, desc="concat positions", unit="file") if verbosity >= 1 else pos_paths
    except ImportError:
        _iter_pos = pos_paths

    for p in _iter_pos:
        df_p = pl.read_parquet(p) if p.suffix == ".parquet" else pl.read_csv(p, separator="\t", infer_schema_length=0)
        all_pos_cols.update(df_p.columns)
        pos_dfs.append(df_p)
    # harmonise schemas
    ordered_pos_cols = sorted(all_pos_cols)
    aligned_pos_dfs: list[pl.DataFrame] = []
    for df_p in pos_dfs:
        missing = [c for c in ordered_pos_cols if c not in df_p.columns]
        if missing:
            df_p = df_p.with_columns([pl.lit(None).alias(c) for c in missing])
        aligned_pos_dfs.append(df_p.select(ordered_pos_cols))
    pos_df = pl.concat(aligned_pos_dfs, rechunk=True)
    if verbosity:
        print(f"[reaggregate] Loaded {pos_df.height:,} rows from {len(pos_df.columns)} columns across {len(pos_paths)} files.")
        try:
            n_genes = pos_df.select(pl.col("gene_id").n_unique()).item()
            print(f"[reaggregate] Unique genes: {n_genes:,}")
            if "gene_type" in pos_df.columns:
                gt_counts = (
                    pos_df.group_by("gene_type")
                          .agg(pl.col("gene_id").n_unique().alias("n_genes"))
                          .sort("n_genes", descending=True)
                )
                top_types = gt_counts.head(10)
                type_str = ", ".join(f"{row['gene_type']}={row['n_genes']}" for row in top_types.to_dicts())
                print(f"[reaggregate] Gene types (top): {type_str}")
        except Exception:
            pass

    # Collect chunk-level error analysis.
    if verbosity:
        print("[reaggregate] Scanning for chunk-level error analysis …")
    err_paths = sorted(root_dir.glob(f"{subject_err}_*_chunk_*.tsv"))
    err_paths += sorted(root_dir.glob(f"{subject_err}_*_chunk_*.parquet"))
    if not err_paths:
        raise RuntimeError("No splice-error chunk files found in " + str(root_dir))

    err_dfs = []
    all_err_cols: set[str] = set()
    try:
        from tqdm import tqdm
        _iter_err = tqdm(err_paths, desc="concat errors", unit="file") if verbosity >= 1 else err_paths
    except ImportError:
        _iter_err = err_paths

    for p in _iter_err:
        df_e = pl.read_parquet(p) if p.suffix == ".parquet" else pl.read_csv(p, separator="\t", infer_schema_length=0)
        all_err_cols.update(df_e.columns)
        err_dfs.append(df_e)
    ordered_err_cols = sorted(all_err_cols)
    aligned_err_dfs: list[pl.DataFrame] = []
    for df_e in err_dfs:
        missing = [c for c in ordered_err_cols if c not in df_e.columns]
        if missing:
            df_e = df_e.with_columns([pl.lit(None).alias(c) for c in missing])
        aligned_err_dfs.append(df_e.select(ordered_err_cols))
    err_df = pl.concat(aligned_err_dfs, rechunk=True)
    if verbosity:
        print(f"[reaggregate] Loaded {err_df.height:,} error rows across {len(err_paths)} files.")

    # Determine output filenames & directory
    out_dir = Path(dh._get_output_dir(output_subdir))
    pos_fname = "full_splice_positions_enhanced.tsv" if enhanced else "full_splice_positions.tsv"
    err_fname = "full_splice_errors.tsv"
    pos_path = out_dir / pos_fname
    err_path = out_dir / err_fname

    for path in (pos_path, err_path):
        if path.exists() and not overwrite:
            raise FileExistsError(
                f"{path} already exists – use --overwrite to replace it.")

    if verbosity:
        print(f"[reaggregate] Writing {pos_path.relative_to(out_dir.parent)} …")
    pos_df.write_csv(pos_path, separator="\t")

    if verbosity:
        print(f"[reaggregate] Writing {err_path.relative_to(out_dir.parent)} …")
    err_df.write_csv(err_path, separator="\t")

    if verbosity:
        print("[reaggregate] Done.")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Re-aggregate splice-site artefacts from chunk files.")
    p.add_argument("--eval-dir", default=None, help="Path to spliceai_eval directory. Default: Analyzer.eval_dir")
    p.add_argument("--output-subdir", default=None, help="Subdirectory (e.g. meta_models, inference)")
    p.add_argument("--no-enhanced", action="store_true", help="Expect legacy one-score artefacts instead of enhanced three-score files.")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing aggregated TSVs if they exist.")
    p.add_argument("-v", "--verbosity", type=int, default=1, choices=[0, 1, 2])
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    reaggregate(
        eval_dir=args.eval_dir,
        output_subdir=args.output_subdir,
        enhanced=not args.no_enhanced,
        overwrite=args.overwrite,
        verbosity=args.verbosity,
    )


if __name__ == "__main__":
    main()
