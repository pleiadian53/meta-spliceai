"""Gene selection utilities for training-data generation workflows.

This module provides a lightweight, reusable interface for sub-setting the
analysis-sequence DataFrame (produced by ``MetaModelDataHandler``) down to a
user-specified number of *target* genes according to several policies
(e.g. random sampling, hard-gene selection, and **new error-count based
ranking**).

The goal is to minimise the volume of training data for quick iterations while
focusing on informative or challenging genes.
"""
from __future__ import annotations

from typing import Iterable, List, Literal, Optional, Tuple

import os
import polars as pl

from meta_spliceai.splice_engine.meta_models.io.handlers import MetaModelDataHandler
from meta_spliceai.splice_engine.meta_models.analysis.error_counting import count_effective_errors

# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

ErrorPolicy = Literal["error_fp", "error_fn", "error_total"]
RandomPolicy = Literal["rand", "random"]
HardPolicy = Literal["hard"]  # placeholder – implemented elsewhere

SubsetPolicy = Literal[
    "rand",
    "random",
    "hard",  # Performance-based hard genes
    "error_fp",
    "error_fn",
    "error_total",
    "custom",  # Custom gene list provided by the user
]


def subset_analysis_sequences(
    data_handler: MetaModelDataHandler,
    n_genes: int = 1_000,
    subset_policy: SubsetPolicy = "error_total",
    *,
    analysis_sequence_df: Optional[pl.DataFrame] = None,
    positions_df: Optional[pl.DataFrame] = None,
    aggregated: bool = True,
    additional_gene_ids: Optional[Iterable[str]] = None,
    use_effective_counts: bool = True,
    verbose: int = 1,
) -> Tuple[pl.DataFrame, List[str]]:
    """Return a subset of *analysis-sequence* rows restricted to ``n_genes``.

    Parameters
    ----------
    data_handler
        Instance of :class:`~meta_spliceai.splice_engine.meta_models.io.handlers.MetaModelDataHandler`.
    n_genes
        Desired number of genes.
    subset_policy
        Strategy for prioritising which genes are kept.  Supported values::

            'rand'|'random'   - Random sample of *n_genes*
            'hard'            - Use existing *PerformanceAnalyzer* logic (delegated)
            'error_fp'        - Genes with the most **false positives**
            'error_fn'        - Genes with the most **false negatives**
            'error_total'     - Genes with the highest *FP + FN* count (default)
            'custom'          - Use only genes provided in `additional_gene_ids` (ignores n_genes)

    analysis_sequence_df
        Pre-loaded analysis-sequence DataFrame.  If omitted, it will be loaded
        via *data_handler*.
    positions_df
        Pre-loaded splice-positions DataFrame.  If omitted, it will be loaded
        via *data_handler*.
    aggregated
        Whether to load the aggregated (genome-wide) file when
        ``analysis_sequence_df`` is *None*.
    additional_gene_ids
        Gene IDs that must **always** be included in the final set (e.g. for
        custom benchmarking).
    use_effective_counts
        If True, use effective error counts that deduplicate errors at the same genomic
        position across multiple transcripts (recommended).
    verbose
        Verbosity level.

    Returns
    -------
    tuple
        ``(subset_df, gene_ids)`` where ``subset_df`` contains rows only for the
        selected genes and ``gene_ids`` is the list of unique gene IDs retained.
    """

    policy = subset_policy.lower()
    additional_gene_ids = list(additional_gene_ids or [])

    # Short-circuit when we *only* need a custom gene list – avoids loading the
    # very large genome-wide splice-positions table and therefore saves >10 GB
    # peak RAM during meta-model evaluation.
    if policy == "custom":
        if not additional_gene_ids:
            raise ValueError("'custom' policy requires non-empty additional_gene_ids parameter")
        if verbose:
            print(f"[gene_selection] Using custom gene list with {len(additional_gene_ids)} genes (no positions_df load)")
        # Minimal stub DataFrame – callers rarely use the contents
        subset_df = pl.DataFrame({"gene_id": additional_gene_ids})
        return subset_df, additional_gene_ids

    # ------------------------------------------------------------------
    # Load splice *positions* DataFrame – this contains richer metadata (e.g.
    # `pred_type`, error labels, scores) that are no longer present in the
    # compact `analysis_sequence_df`.
    # ------------------------------------------------------------------
    if positions_df is None:
        if verbose:
            print("[gene_selection] Loading splice-positions DataFrame…")
        
        try:
            positions_df = data_handler.load_splice_positions(aggregated=aggregated)
            
            # Check if the aggregated file is complete (should have many genes for genome-wide data)
            n_genes_in_file = positions_df.select(pl.col("gene_id")).n_unique()
            if n_genes_in_file < 1000:  # Suspiciously low for genome-wide data
                if verbose:
                    print(f"[gene_selection] WARNING: Aggregated file only contains {n_genes_in_file} genes (suspiciously low)")
                    print(f"[gene_selection] This suggests the aggregated file is incomplete/corrupted from a previous run")
                    print(f"[gene_selection] Assembling fresh dataset from chunked files...")
                raise FileNotFoundError("Aggregated file is incomplete - will reassemble from chunks")
            else:
                if verbose:
                    print(f"[gene_selection] Using aggregated file with {n_genes_in_file:,} genes")
                    
        except FileNotFoundError:
            # If the aggregated file doesn't exist OR is incomplete, assemble it from chunked files
            if verbose:
                print("[gene_selection] Aggregated positions file not found or incomplete. Assembling from chunked files...")
            
            import glob
            import os
            
            # Find chunked splice_positions files
            meta_dir = data_handler.meta_dir
            chunk_pattern = os.path.join(meta_dir, "splice_positions_enhanced_*_chunk_*.tsv")
            chunk_files = glob.glob(chunk_pattern)
            
            if not chunk_files:
                raise FileNotFoundError(
                    f"No splice positions files found. "
                    f"Expected either aggregated file or chunked files matching pattern: {chunk_pattern}"
                )
            
            if verbose:
                print(f"[gene_selection] Found {len(chunk_files)} chunked positions files. Loading...")
            
            # Load and combine all chunked files
            chunk_dfs = []
            unified_schema = None
            
            for i, chunk_file in enumerate(chunk_files):
                try:
                    chunk_df = pl.read_csv(
                        chunk_file,
                        separator=data_handler.separator,
                        schema_overrides={"chrom": pl.Utf8}
                    )
                    
                    # Establish unified column order from first successful file
                    if unified_schema is None:
                        unified_schema = chunk_df.schema
                        if verbose >= 2:
                            print(f"[gene_selection] Using column order from {os.path.basename(chunk_file)}: {len(unified_schema)} columns")
                    
                    # Verify column sets match (not just order)
                    current_columns = set(chunk_df.columns)
                    expected_columns = set(unified_schema.keys())
                    
                    if current_columns != expected_columns:
                        missing = expected_columns - current_columns
                        extra = current_columns - expected_columns
                        raise ValueError(
                            f"Column mismatch in {os.path.basename(chunk_file)}:\n"
                            f"  Missing columns: {missing}\n"
                            f"  Extra columns: {extra}\n"
                            f"  Expected columns: {len(expected_columns)}, Got: {len(current_columns)}"
                        )
                    
                    # Reorder columns to match the unified order
                    aligned_df = chunk_df.select([col for col in unified_schema.keys()])
                    chunk_dfs.append(aligned_df)
                    
                    if verbose and (i + 1) % 20 == 0:
                        print(f"[gene_selection] Loaded {i + 1}/{len(chunk_files)} chunk files...")
                        
                except Exception as e:
                    if verbose >= 2:
                        print(f"[gene_selection] Warning: Failed to load {os.path.basename(chunk_file)}: {e}")
            
            if not chunk_dfs:
                raise RuntimeError("Failed to load any chunked splice positions files")
            
            if verbose:
                print(f"[gene_selection] Combining {len(chunk_dfs)} chunk files...")
            positions_df = pl.concat(chunk_dfs)
            n_genes = positions_df.select(pl.col("gene_id")).n_unique()
            
            if verbose:
                print(f"[gene_selection] Successfully assembled complete dataset: {n_genes:,} genes, {positions_df.height:,} positions")
                
            # Save the assembled aggregated file for future use (overwrite the corrupted one)
            try:
                output_path = os.path.join(meta_dir, "full_splice_positions_enhanced.tsv")
                positions_df.write_csv(output_path, separator=data_handler.separator)
                if verbose:
                    print(f"[gene_selection] Saved complete aggregated positions to: {output_path}")
            except Exception as e:
                if verbose >= 2:
                    print(f"[gene_selection] Warning: Could not save aggregated file: {e}")

    if "gene_id" not in positions_df.columns:
        raise ValueError("positions_df must contain a 'gene_id' column.")

    policy = subset_policy.lower()
    additional_gene_ids = list(additional_gene_ids or [])

    # ------------------------------------------------------------------
    # Policy: Custom gene list (only use genes in additional_gene_ids)
    # ------------------------------------------------------------------
    if policy == "custom":
        if not additional_gene_ids:
            raise ValueError("'custom' policy requires non-empty additional_gene_ids parameter")
        
        if verbose:
            print(f"[gene_selection] Using custom gene list with {len(additional_gene_ids)} genes")
        
        gene_ids = list(additional_gene_ids)

    # ------------------------------------------------------------------
    # Policy: Random sampling
    # ------------------------------------------------------------------
    elif policy in ("rand", "random"):
        if verbose:
            print(f"[gene_selection] Randomly sampling n={n_genes} genes …")
        unique_gene_ids = positions_df.select(pl.col("gene_id")).unique()
        sampled_gene_ids = (
            unique_gene_ids.sample(n=n_genes, with_replacement=False)
            if n_genes < unique_gene_ids.height
            else unique_gene_ids
        )
        gene_ids = sampled_gene_ids["gene_id"].to_list()

    # ------------------------------------------------------------------
    # Policy: Error-count based ranking (new)
    # ------------------------------------------------------------------
    elif policy.startswith("error"):
        if "pred_type" not in positions_df.columns:
            raise ValueError("'pred_type' column missing – cannot compute error counts.")
            
        if use_effective_counts:
            # Use the specialized error counting implementation that deduplicates by position
            if verbose > 0:
                print(f"[gene_selection] Using effective error counts (deduplicated by position)")
                
            # Map the policy to specific error type(s)
            error_types = ["FP", "FN"]  # default for error_total
            if policy == "error_fp":
                error_types = ["FP"]
            elif policy == "error_fn":
                error_types = ["FN"]
                
            # Count effective errors
            error_df = count_effective_errors(
                positions_df, 
                group_by=["gene_id"],
                count_by_position=True,
                error_types=error_types,
                verbose=max(0, verbose - 1)  # reduce verbosity level for sub-function
            )
            
            # Determine which column to sort by based on policy
            sort_col = "total_effective_errors"  # default for error_total
            if policy == "error_fp" and "FP" in error_df.columns:
                sort_col = "FP"
            elif policy == "error_fn" and "FN" in error_df.columns:
                sort_col = "FN"
                
            # Filter to only genes with actual errors (count > 0)
            if verbose:
                total_genes_with_errors = error_df.filter(pl.col(sort_col) > 0).height
                print(f"[gene_selection] Found {total_genes_with_errors} genes with actual errors out of {error_df.height} total genes")
            
            # Only consider genes with actual errors
            genes_with_errors = error_df.filter(pl.col(sort_col) > 0)
            
            if genes_with_errors.height == 0:
                if verbose:
                    print(f"[gene_selection] WARNING: No genes found with {policy} errors")
                gene_ids = []
            else:
                # Sort and take top N genes from those with actual errors
                gene_ids = genes_with_errors.sort(sort_col, descending=True).head(n_genes)["gene_id"].to_list()
            
            if verbose:
                print(f"[gene_selection] Selected top {len(gene_ids)} genes by effective '{policy}' count.")
        else:
            # Traditional error counting (may count the same position multiple times)
            if verbose > 0:
                print(f"[gene_selection] Using raw error counts (may include transcript duplicates)")
                
            error_mask = positions_df["pred_type"].is_in(["FP", "FN"])
            if policy == "error_fp":
                error_mask = positions_df["pred_type"] == "FP"
            elif policy == "error_fn":
                error_mask = positions_df["pred_type"] == "FN"
            # else: keep both FP + FN

            error_df = (
                positions_df.filter(error_mask)
                .group_by("gene_id")
                .len()
                .sort("len", descending=True)
            )
            
            # Filter to only genes with actual errors (count > 0)
            if verbose:
                total_genes_with_errors = error_df.filter(pl.col("len") > 0).height
                print(f"[gene_selection] Found {total_genes_with_errors} genes with actual errors out of {error_df.height} total genes")
            
            # Only consider genes with actual errors
            genes_with_errors = error_df.filter(pl.col("len") > 0)
            
            if genes_with_errors.height == 0:
                if verbose:
                    print(f"[gene_selection] WARNING: No genes found with {policy} errors")
                gene_ids = []
            else:
                # Sort and take top N genes from those with actual errors
                top_genes_df = genes_with_errors.head(n_genes)
                gene_ids = top_genes_df["gene_id"].to_list()

            if verbose:
                print(f"[gene_selection] Selected top {len(gene_ids)} genes by raw '{policy}' count.")

    # ------------------------------------------------------------------
    # Policy: Hard genes (delegated to PerformanceAnalyzer)
    # ------------------------------------------------------------------
    elif policy == "hard":
        from meta_spliceai.splice_engine.performance_analyzer import PerformanceAnalyzer

        if verbose:
            print("[gene_selection] Delegating hard-gene retrieval to PerformanceAnalyzer …")
        pa = PerformanceAnalyzer()
        df_hard_genes = pa.retrieve_hard_genes(n_genes=n_genes, **{})
        if df_hard_genes is None or df_hard_genes.is_empty():
            raise RuntimeError("PerformanceAnalyzer returned no hard genes.")
        gene_ids = df_hard_genes.select("gene_id").to_series().to_list()

    elif policy in ("custom", "explicit"):
        # Custom policy: use only the genes provided in additional_gene_ids
        if not additional_gene_ids:
            raise ValueError("Custom policy requires additional_gene_ids to be provided")
        gene_ids = []  # Start with empty list, will be populated by additional_gene_ids below

    elif policy == "all":
        # All policy: use all available genes (ignore n_genes and additional_gene_ids)
        unique_gene_ids = df.select(pl.col("gene_id")).unique()
        gene_ids = unique_gene_ids["gene_id"].to_list()
        if verbose:
            print(f"[gene_selection] Selected all {len(gene_ids):,} available genes")
        # For 'all' policy, we don't add additional_gene_ids (they're already included)
        return gene_ids

    else:
        raise ValueError(f"Unknown subset_policy='{subset_policy}'.")

    # ------------------------------------------------------------------
    # Ensure additional genes are included
    # ------------------------------------------------------------------
    gene_ids = list(set(gene_ids) | set(additional_gene_ids))

    # ------------------------------------------------------------------
    # Load analysis-sequence DataFrame *after* gene IDs have been determined.
    # ------------------------------------------------------------------
    # Process analysis sequences efficiently for the selected gene IDs
    # ------------------------------------------------------------------
    if analysis_sequence_df is None:
        if verbose:
            print(f"[gene_selection] Loading analysis sequences for {len(gene_ids)} genes using memory-efficient loading...")
        # Use the new iterative loading method that only loads sequences for target genes
        analysis_sequence_df = data_handler.iterative_load_analysis_sequences(
            target_gene_ids=gene_ids,
            output_subdir=None,  # Use default directory
            use_shared_dir=False,  # Use subject-specific directory
            show_progress=(verbose > 0),
            verbose=verbose,
            aggregated=aggregated  # Pass through the aggregated flag
        )

    if "gene_id" not in analysis_sequence_df.columns:
        raise ValueError("analysis_sequence_df must contain a 'gene_id' column.")
        
    # Already filtered by gene_id during loading, but let's make sure
    subset_df = analysis_sequence_df.filter(pl.col("gene_id").is_in(gene_ids))

    if verbose:
        n_unique = subset_df.select(pl.col("gene_id").n_unique()).item()
        print(f"[gene_selection] Final gene count: {n_unique}")

    return subset_df, gene_ids


# -----------------------------------------------------------------------------
# Splice *positions* subsetting utility
# -----------------------------------------------------------------------------

def subset_positions_dataframe(
    positions_df: pl.DataFrame,
    n_genes: int = 1_000,
    subset_policy: SubsetPolicy = "error_total",
    *,
    gene_types: Optional[Iterable[str]] = None,
    gene_features_df: Optional[pl.DataFrame] = None,
    gene_features_path: Optional[str] = None,
    additional_gene_ids: Optional[Iterable[str]] = None,
    use_effective_counts: bool = True,
    verbose: int = 1,
) -> Tuple[pl.DataFrame, List[str]]:
    """Subset a *splice-positions* DataFrame to the most informative genes.

    This mirrors the behaviour of :func:`subset_analysis_sequences` but operates
    directly on the **positions_df**, which contains richer error labels and
    probability features.  Additional filtering by *gene type* is supported via
    an external *gene features* table.
    
    When ``use_effective_counts=True`` (the default), genes are prioritized using
    effective error counts that deduplicate errors at the same genomic position
    across multiple transcripts.
    """

    if "gene_id" not in positions_df.columns:
        raise ValueError("positions_df must contain a 'gene_id' column.")

    # ------------------------------------------------------------------
    # Optional gene-type filtering
    # ------------------------------------------------------------------
    if gene_types is not None:
        gene_types = list(gene_types)

        # Case 1: positions_df already carries `gene_type`
        if "gene_type" in positions_df.columns:
            positions_df = positions_df.filter(pl.col("gene_type").is_in(gene_types))
        else:
            # Need external mapping (gene_id -> gene_type)
            if gene_features_df is None:
                if gene_features_path is None:
                    # Derive a sensible default location
                    try:
                        from meta_spliceai.system.config import Config

                        gene_features_path = os.path.join(
                            Config.DATA_DIR,
                            "ensembl",
                            "spliceai_analysis",
                            "gene_features.tsv",
                        )
                    except Exception:
                        home = os.environ.get("HOME", "")
                        gene_features_path = os.path.join(
                            home,
                            "work",
                            "splice-surveyor",
                            "data",
                            "ensembl",
                            "spliceai_analysis",
                            "gene_features.tsv",
                        )

                if verbose:
                    print(f"[gene_selection] Loading gene features from: {gene_features_path}")

                gene_features_df = pl.read_csv(
                    gene_features_path,
                    separator="\t",
                    schema_overrides={"chrom": pl.Utf8},
                )

            filtered_gene_ids = (
                gene_features_df.filter(pl.col("gene_type").is_in(gene_types))
                .select("gene_id")
                .to_series()
                .to_list()
            )

            positions_df = positions_df.filter(pl.col("gene_id").is_in(filtered_gene_ids))

        if verbose:
            n_after = positions_df.select(pl.count()).item()
            print(f"[gene_selection] After gene-type filter: {n_after:,} rows")

    # ------------------------------------------------------------------
    # Gene selection according to policy
    # ------------------------------------------------------------------
    policy = subset_policy.lower()
    additional_gene_ids = list(additional_gene_ids or [])

    if policy in ("rand", "random"):
        unique_gene_ids = positions_df.select(pl.col("gene_id")).unique()
        sampled_gene_ids = (
            unique_gene_ids.sample(n=n_genes, with_replacement=False)
            if n_genes < unique_gene_ids.height
            else unique_gene_ids
        )
        gene_ids = sampled_gene_ids["gene_id"].to_list()

    elif policy.startswith("error"):
        if "pred_type" not in positions_df.columns:
            raise ValueError("'pred_type' column missing – cannot compute error counts.")
            
        if use_effective_counts:
            # Use the specialized error counting implementation that deduplicates by position
            if verbose > 0:
                print(f"[gene_selection] Using effective error counts (deduplicated by position)")
                
            # Map the policy to specific error type(s)
            error_types = ["FP", "FN"]  # default for error_total
            if policy == "error_fp":
                error_types = ["FP"]
            elif policy == "error_fn":
                error_types = ["FN"]
                
            # Count effective errors
            error_df = count_effective_errors(
                positions_df, 
                group_by=["gene_id"],
                count_by_position=True,
                error_types=error_types,
                verbose=max(0, verbose - 1)  # reduce verbosity level for sub-function
            )
            
            # Determine which column to sort by based on policy
            sort_col = "total_effective_errors"  # default for error_total
            if policy == "error_fp" and "FP" in error_df.columns:
                sort_col = "FP"
            elif policy == "error_fn" and "FN" in error_df.columns:
                sort_col = "FN"
                
            # Sort and take top N genes
            gene_ids = error_df.sort(sort_col, descending=True).head(n_genes)["gene_id"].to_list()
            
        else:
            # Traditional error counting (may count the same position multiple times)
            if verbose > 0:
                print(f"[gene_selection] Using raw error counts (may include transcript duplicates)")
                
            error_mask = positions_df["pred_type"].is_in(["FP", "FN"])
            if policy == "error_fp":
                error_mask = positions_df["pred_type"] == "FP"
            elif policy == "error_fn":
                error_mask = positions_df["pred_type"] == "FN"

            error_df = (
                positions_df.filter(error_mask)
                .group_by("gene_id")
                .len()
                .sort("len", descending=True)
            )
            gene_ids = error_df.head(n_genes)["gene_id"].to_list()

    elif policy == "hard":
        from meta_spliceai.splice_engine.performance_analyzer import PerformanceAnalyzer

        pa = PerformanceAnalyzer()
        df_hard_genes = pa.retrieve_hard_genes(n_genes=n_genes)
        if df_hard_genes is None or df_hard_genes.is_empty():
            raise RuntimeError("PerformanceAnalyzer returned no hard genes.")
        gene_ids = df_hard_genes.select("gene_id").to_series().to_list()

    elif policy in ("custom", "explicit"):
        # Custom policy: use only the genes provided in additional_gene_ids
        if not additional_gene_ids:
            raise ValueError("Custom policy requires additional_gene_ids to be provided")
        gene_ids = []  # Start with empty list, will be populated by additional_gene_ids below

    elif policy == "all":
        # All policy: use all available genes (ignore n_genes and additional_gene_ids)
        unique_gene_ids = positions_df.select(pl.col("gene_id")).unique()
        gene_ids = unique_gene_ids["gene_id"].to_list()
        if verbose:
            print(f"[gene_selection] Selected all {len(gene_ids):,} available genes")
        # For 'all' policy, we don't add additional_gene_ids (they're already included)
        subset_df = positions_df.filter(pl.col("gene_id").is_in(gene_ids))
        return subset_df, gene_ids

    else:
        raise ValueError(f"Unknown subset_policy='{subset_policy}'.")

    # Ensure extras are included
    gene_ids = list(set(gene_ids) | set(additional_gene_ids))

    subset_df = positions_df.filter(pl.col("gene_id").is_in(gene_ids))

    if verbose:
        n_unique = subset_df.select(pl.col("gene_id").n_unique()).item()
        print(f"[gene_selection] Final gene count: {n_unique}")

    return subset_df, gene_ids


# -----------------------------------------------------------------------------
# Convenience re-exports for *features* package
# -----------------------------------------------------------------------------
__all__ = [
    "subset_analysis_sequences",
    "subset_positions_dataframe",
]
