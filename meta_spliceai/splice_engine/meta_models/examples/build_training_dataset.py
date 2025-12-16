"""Example script to assemble a small training dataset for the meta-model.

This utility glues together the following pipeline stages already present in
MetaSpliceAI:

1. *Gene selection* via ``subset_analysis_sequences`` – choose the most
   informative genes (e.g. highest FN counts) from the consolidated
   ``full_splice_positions_enhanced.tsv``.
2. *Dataset assembly* with ``build_training_dataset`` – streams the many
   ``*_analysis_sequences_*.tsv[.gz]`` files and optionally adds k-mer features.
3. *Feature enrichment* with ``apply_feature_enrichers`` – augments the table
   with gene-level length, performance, overlap, distance … features thanks to
   the registry pattern defined in ``feature_enrichment.py``.

Run this after executing *run_fn_rescue_pipeline.py* (or its FP counterpart) so
that the chunked ``analysis_sequences_*`` and consolidated positions files are
available under the *meta_models* directory.

Example
-------
$ python -m meta_spliceai.splice_engine.meta_models.examples.build_training_dataset \
    --n-genes 5 --subset-policy error_fn --kmer-sizes 6 5

This will create two Parquet files in the *meta_models* directory:
    training_dataset_… .parquet         (base k-mer table)
    training_dataset_…_enriched.parquet (with extra feature sets)
"""
from __future__ import annotations

import os
import argparse
from pathlib import Path
from typing import List, Optional, Sequence

import polars as pl

# ────────────────────────────────────────────────────────────────────────────
# Internal imports – we rely only on existing public helpers so that this
# script remains a thin orchestration layer.
# ────────────────────────────────────────────────────────────────────────────
from meta_spliceai.splice_engine.meta_models.io.handlers import MetaModelDataHandler
from meta_spliceai.splice_engine.meta_models.features.gene_selection import (
    subset_analysis_sequences,
)
from meta_spliceai.splice_engine.meta_models.dataset_builder import (
    build_training_dataset,
)
from meta_spliceai.splice_engine.meta_models.features.feature_enrichment import (
    apply_feature_enrichers,
)
# NEW: optionally kick off the enhanced prediction workflow
from meta_spliceai.splice_engine.meta_models.workflows.splice_prediction_workflow import (
    run_enhanced_splice_prediction_workflow,
)

# ---------------------------------------------------------------------------
# Core driver
# ---------------------------------------------------------------------------

def build_meta_model_training_dataset(
    *,
    eval_dir: Optional[str] = None,
    n_genes: int = 2000,
    subset_policy: str = "error_total",
    kmer_sizes: Optional[Sequence[int]] = (6,),
    enrichers: Optional[Sequence[str]] = None,
    output_prefix: Optional[str] = None,
    overwrite: bool = False,
    batch_rows: int = 500_000,
    additional_gene_ids: Optional[Sequence[str]] = None,
    use_effective_counts: bool = True,
    run_workflow: bool = False,
    workflow_kwargs: Optional[dict] = None,
    verbose: int = 1,
) -> Path:
    """Assemble a complete training dataset for the meta-model.

    Parameters
    ----------
    eval_dir
        Root *spliceai_eval* directory.  If *None*, uses the default path from
        :class:`Analyzer`.
    n_genes, subset_policy
        Passed to :pyfunc:`subset_analysis_sequences`.
    kmer_sizes
        List of k-mer sizes.  If *None* or empty, k-mer extraction is skipped.
    enrichers
        Names of additional feature-set enrichers as registered in
        ``feature_enrichment``.  *None* > run all registered enrichers.
    output_prefix
        Basename (without extension) for the resulting Parquet files.
    overwrite
        Overwrite existing output files if they already exist.
    batch_rows
        How many rows to buffer between disk writes when streaming TSVs.
    additional_gene_ids
        Force-include these gene IDs (implies subset-policy=custom if used alone).
    use_effective_counts
        Use effective error counts instead of raw error counts when ranking genes.
    run_workflow
        Run the enhanced prediction workflow before building dataset.
    workflow_kwargs
        Keyword arguments to pass to the enhanced prediction workflow.
    verbose
        Verbosity level.
    """

    # 0. Optionally run the upstream enhanced prediction workflow -------------
    if run_workflow:
        if verbose:
            print("[builder] Running enhanced splice-prediction workflow …")
        workflow_kwargs = workflow_kwargs or {}
        # If a custom gene list is supplied, forward it, otherwise run on all
        if additional_gene_ids:
            workflow_kwargs.setdefault("target_genes", list(additional_gene_ids))
        run_enhanced_splice_prediction_workflow(verbosity=max(0, verbose - 1), **workflow_kwargs)

    # 1. Resolve paths and helper objects -------------------------------------------------
    data_handler = MetaModelDataHandler(eval_dir=eval_dir)
    meta_dir = Path(data_handler.meta_dir)  # contains the *_analysis_sequences_* TSVs

    # 2. Gene selection ------------------------------------------------------------------
    _, gene_ids = subset_analysis_sequences(
        data_handler=data_handler,
        n_genes=n_genes,
        subset_policy=subset_policy,
        aggregated=True,
        additional_gene_ids=additional_gene_ids,
        use_effective_counts=use_effective_counts,
        verbose=verbose,
    )
    if verbose:
        print(f"[builder] Selected {len(gene_ids)} genes via policy '{subset_policy}'.")

    # 3. Build base dataset with k-mers ---------------------------------------------------
    if output_prefix is None:
        output_prefix = f"training_dataset_{subset_policy}_{len(gene_ids)}g"
    base_path = meta_dir / f"{output_prefix}.parquet"

    if base_path.exists() and not overwrite:
        raise FileExistsError(
            f"{base_path} exists. Use --overwrite to regenerate the dataset.")

    build_training_dataset(
        analysis_tsv_dir=str(meta_dir),
        output_path=str(base_path),
        mode="none",              # we already determined the gene subset
        target_genes=gene_ids,
        top_n_genes=None,
        kmer_sizes=list(kmer_sizes) if kmer_sizes else None,
        batch_rows=batch_rows,
        keep_sequence=False,
        verbose=verbose,
    )
    if verbose:
        print(f"[builder] Base training dataset saved to: {base_path}")

    # 4. Load Parquet and run feature enrichers -----------------------------------------
    df = pl.read_parquet(base_path)
    rows_before = df.height

    df_enriched = apply_feature_enrichers(
        df,
        enrichers=enrichers,
        verbose=verbose,
        fa=None,
        sa=None,
    )

    assert df_enriched.height == rows_before, \
       f"Row mismatch: {rows_before} -> {df_enriched.height}"

    enriched_path = meta_dir / f"{output_prefix}_enriched.parquet"
    if enriched_path.exists() and not overwrite:
        raise FileExistsError(
            f"{enriched_path} exists. Use --overwrite to regenerate the enriched dataset.")

    df_enriched.write_parquet(enriched_path, compression="zstd")
    if verbose:
        print(f"[builder] Enriched training dataset saved to: {enriched_path}")

    return enriched_path


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------

def _parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Assemble a k-mer + enriched feature training dataset for the meta-model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--n-genes", type=int, default=2000, help="Number of genes to include.")
    parser.add_argument(
        "--subset-policy",
        type=str,
        default="error_total",
        choices=[
            "error_total",
            "error_fp",
            "error_fn",
            "random",
            "rand",
            "custom",
        ],
        help="Gene selection strategy.",
    )
    parser.add_argument("--kmer-sizes", nargs="*", type=int, default=[6], help="k-mer sizes.")
    parser.add_argument(
        "--enrichers",
        nargs="*",
        default=None,
        help="Subset of enrichers to run (default: all).",
    )
    parser.add_argument(
        "--eval-dir",
        type=str,
        default=None,
        help="Path to the spliceai_eval directory (auto-detected if omitted).",
    )
    parser.add_argument("--output-prefix", type=str, default=None, help="Output file basename.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files.")
    parser.add_argument("--batch-rows", type=int, default=500_000, help="Batch size for streaming.")
    parser.add_argument(
        "--verbose",
        "-v",
        nargs="?",           # optional value
        const=1,              # if flag provided with no value → 1
        default=1,
        type=int,
        help="Verbosity level (omit value to set to 1; higher is more verbose).",
    )
    # NEW options
    parser.add_argument("--additional-genes", nargs="*", default=None, help="Force-include these gene IDs (implies subset-policy=custom if used alone).")
    parser.add_argument("--no-effective-counts", action="store_true", help="Use raw error counts instead of effective counts when ranking genes.")
    parser.add_argument("--run-workflow", action="store_true", help="Run the enhanced prediction workflow before building dataset.")
    return parser.parse_args()


def main() -> None:
    args = _parse_cli()

    build_meta_model_training_dataset(
        eval_dir=args.eval_dir,
        n_genes=args.n_genes,
        subset_policy=args.subset_policy,
        kmer_sizes=args.kmer_sizes,
        enrichers=args.enrichers,
        output_prefix=args.output_prefix,
        overwrite=args.overwrite,
        batch_rows=args.batch_rows,
        additional_gene_ids=args.additional_genes,
        use_effective_counts=not args.no_effective_counts,
        run_workflow=args.run_workflow,
        workflow_kwargs={},
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
