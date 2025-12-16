from __future__ import annotations

"""Incremental Meta-Model Training Dataset Builder
=======================================================
Builds the full-genome meta-model training dataset in small, memory-friendly
batches.  Each batch follows the build → enrich → downsample → append pattern
so that enormous numbers of *easy* true-negative rows never need to be
materialised at once.

High-level algorithm
--------------------
1. Determine the list of target genes (via ``subset_analysis_sequences`` or a
   user-supplied list).
2. Split the gene list into fixed-size batches.
3. For each batch:
   a. Create a *raw* Parquet with k-mer & mandatory base features.
   b. Enrich it with additional gene / performance / overlap features.
   c. Down-sample the true negatives to balance the dataset.
   d. Append (partitioned by chromosome) to a master Arrow dataset directory.

The master directory can later be memory-mapped by most ML libraries and can be
incrementally extended or re-written on failure recovery.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple, Any, Iterable, TypeVar, Callable
from functools import lru_cache

import polars as pl
import pandas as pd

# Add project root to path for imports
project_root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(project_root))

import itertools

import polars as pl
import pyarrow.dataset as ds
import pyarrow as pa
import pyarrow.parquet as pq

from meta_spliceai.splice_engine.meta_models.io.handlers import (
    MetaModelDataHandler,
)
from meta_spliceai.splice_engine.meta_models.features.gene_selection import (
    subset_analysis_sequences,
    subset_positions_dataframe,
)
from meta_spliceai.splice_engine.meta_models.builder.dataset_builder import (
    build_training_dataset,
)
from meta_spliceai.splice_engine.meta_models.features.feature_enrichment import (
    apply_feature_enrichers,
)
from meta_spliceai.splice_engine.meta_models.builder.downsample_tn import (
    downsample_tn,
)
from meta_spliceai.splice_engine.meta_models.builder import builder_utils
from meta_spliceai.splice_engine.meta_models.analysis.shared_analysis_utils import (
    check_genomic_files_exist,
)
from meta_spliceai.splice_engine.meta_models.workflows.splice_prediction_workflow import (
    run_enhanced_splice_prediction_workflow,
)
from meta_spliceai.splice_engine.utils.output_enhancement import create_output_enhancer

# Import transcript-aware position identification (optional enhancement)
try:
    from .transcript_aware_positions import TranscriptAwareConfig
    TRANSCRIPT_AWARE_AVAILABLE = True
except ImportError:
    TRANSCRIPT_AWARE_AVAILABLE = False

__all__ = [
    "build_base_dataset",
    "incremental_build_training_dataset",
    "generate_gene_manifest",
]

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _chunks(seq: Sequence[str], size: int) -> Iterable[List[str]]:
    """Yield successive *size*-chunk lists from *seq*."""
    for i in range(0, len(seq), size):
        yield list(seq[i : i + size])

def _generate_gene_manifest(
    master_dir: Path,
    output_path: Path | None = None,
    verbose: int = 1,
) -> Path:
    """Generate an enhanced gene manifest with comprehensive gene characteristics.
    
    This function creates a detailed manifest that includes gene names, types, lengths,
    splice site counts, and other characteristics useful for downstream analysis.
    
    Parameters
    ----------
    master_dir
        Path to the master dataset directory containing Parquet files.
    output_path
        Path for the manifest file. If None, creates 'gene_manifest.csv' in master_dir.
    verbose
        Verbosity level.
        
    Returns
    -------
    Path
        Path to the generated manifest file.
    """
    if output_path is None:
        output_path = master_dir / "gene_manifest.csv"
    
    if verbose:
        print(f"[manifest] Generating enhanced gene manifest from {master_dir} ...")
    
    # Use data resource manager for systematic path resolution
    try:
        # Import the data resource manager from the inference workflows
        project_root = Path(__file__).resolve().parents[4]  # Go up to project root
        sys.path.insert(0, str(project_root))
        from meta_spliceai.splice_engine.meta_models.workflows.inference.data_resource_manager import create_inference_data_manager
        
        data_manager = create_inference_data_manager(project_root=project_root, auto_detect=True)
        
        # Get paths using systematic approach
        gene_features_path = data_manager.get_gene_features_path()
        splice_sites_path = data_manager.get_splice_sites_path()
        
        if verbose:
            print(f"[manifest] Using systematic path resolution:")
            print(f"  Gene features: {gene_features_path}")
            print(f"  Splice sites: {splice_sites_path}")
            
    except Exception as e:
        if verbose:
            print(f"[manifest] Warning: Could not use data resource manager: {e}")
            print(f"[manifest] Falling back to Config/relative paths")
        
        # Fallback to previous approach
        try:
            from meta_spliceai.system.config import Config
            gene_features_path = Path(Config.DATA_DIR) / "ensembl" / "spliceai_analysis" / "gene_features.tsv"
            splice_sites_path = Path(Config.DATA_DIR) / "ensembl" / "splice_sites.tsv"
        except Exception:
            gene_features_path = Path("data/ensembl/spliceai_analysis/gene_features.tsv")
            splice_sites_path = Path("data/ensembl/splice_sites.tsv")
    
    # Load gene features
    gene_features_df = None
    if gene_features_path and gene_features_path.exists():
        try:
            gene_features_df = pl.read_csv(
                gene_features_path, 
                separator='\t',
                schema_overrides={'chrom': pl.Utf8}
            )
            if verbose:
                print(f"[manifest] Loaded gene features: {gene_features_df.height:,} genes")
        except Exception as e:
            if verbose:
                print(f"[manifest] Warning: Could not load gene features: {e}")
            gene_features_df = None
    
    # Load splice sites for density calculations
    splice_sites_df = None
    if splice_sites_path and splice_sites_path.exists():
        try:
            splice_sites_df = pl.read_csv(
                splice_sites_path,
                separator='\t',
                schema_overrides={'chrom': pl.Utf8}
            )
            if verbose:
                print(f"[manifest] Loaded splice sites: {splice_sites_df.height:,} sites")
        except Exception as e:
            if verbose:
                print(f"[manifest] Warning: Could not load splice sites: {e}")
            splice_sites_df = None
    
    # Collect all gene information from Parquet files
    gene_info = []
    total_rows = 0
    
    parquet_paths = sorted(master_dir.glob("*.parquet"))
    if not parquet_paths:
        raise RuntimeError(f"No parquet files found in {master_dir}")
    
    for i, pq_path in enumerate(parquet_paths, 1):
        if verbose:
            print(f"[manifest] Processing {pq_path.name} ({i}/{len(parquet_paths)}) ...")
        
        # First check what columns are available
        df_schema = pl.scan_parquet(pq_path).schema
        available_columns = ["gene_id"]  # gene_id should always be present
        
        if "gene_name" in df_schema:
            available_columns.append("gene_name")
        
        # Read only the columns we need to keep memory usage minimal
        df = pl.read_parquet(pq_path, columns=available_columns)
        
        # Add gene_name column as null if it doesn't exist
        if "gene_name" not in df.columns:
            df = df.with_columns(pl.lit(None).alias("gene_name"))
        
        # Get unique genes from this file
        unique_genes = df.unique(subset=["gene_id"])
        
        # Add file index information
        unique_genes = unique_genes.with_columns([
            pl.lit(i).alias("file_index"),
            pl.lit(pq_path.name).alias("file_name")
        ])
        
        gene_info.append(unique_genes)
        total_rows += df.height
    
    if not gene_info:
        raise RuntimeError("No gene data found in Parquet files")
    
    # Combine all gene information
    manifest_df = pl.concat(gene_info)
    
    # Remove duplicates (genes that appear in multiple files)
    manifest_df = manifest_df.unique(subset=["gene_id"], maintain_order=True)
    
    # Calculate splice site density for each gene
    splice_site_counts = None
    if splice_sites_df is not None:
        try:
            # Group splice sites by gene and count
            splice_site_counts = (
                splice_sites_df
                .group_by("gene_id")
                .agg([
                    pl.len().alias("total_splice_sites"),
                    pl.col("site_type").filter(pl.col("site_type") == "donor").len().alias("donor_sites"),
                    pl.col("site_type").filter(pl.col("site_type") == "acceptor").len().alias("acceptor_sites")
                ])
            )
            if verbose:
                print(f"[manifest] Calculated splice site counts for {splice_site_counts.height:,} genes")
        except Exception as e:
            if verbose:
                print(f"[manifest] Warning: Could not calculate splice site counts: {e}")
            splice_site_counts = None
    
    # Enrich manifest with comprehensive gene characteristics
    if gene_features_df is not None:
        if verbose:
            print(f"[manifest] Enriching manifest with gene characteristics...")
        
        # Prepare gene features for joining (select all relevant columns)
        gene_characteristics = gene_features_df.select([
            "gene_id", "gene_name", "gene_type", "gene_length", "chrom", "strand", "start", "end"
        ]).unique(subset=["gene_id"])
        
        # Left join to get comprehensive gene information
        manifest_df = manifest_df.join(
            gene_characteristics, 
            on="gene_id", 
            how="left", 
            suffix="_lookup"
        )
        
        # Use lookup name if original is null, otherwise keep original
        manifest_df = manifest_df.with_columns(
            pl.coalesce([pl.col("gene_name"), pl.col("gene_name_lookup")]).alias("gene_name")
        ).drop("gene_name_lookup")
        
        if verbose:
            enriched_count = manifest_df.filter(pl.col("gene_type").is_not_null()).height
            print(f"[manifest] Successfully enriched {enriched_count:,} genes with characteristics")
    
    # Add splice site information if available
    if splice_site_counts is not None:
        if verbose:
            print(f"[manifest] Adding splice site density information...")
        
        manifest_df = manifest_df.join(
            splice_site_counts,
            on="gene_id",
            how="left"
        )
        
        # Calculate splice site density (sites per kb)
        manifest_df = manifest_df.with_columns([
            pl.when(pl.col("gene_length").is_not_null() & (pl.col("gene_length") > 0))
            .then(pl.col("total_splice_sites") / (pl.col("gene_length") / 1000))
            .otherwise(None)
            .alias("splice_density_per_kb")
        ])
        
        # Fill missing splice site counts with 0
        manifest_df = manifest_df.with_columns([
            pl.col("total_splice_sites").fill_null(0),
            pl.col("donor_sites").fill_null(0),
            pl.col("acceptor_sites").fill_null(0),
            pl.col("splice_density_per_kb").fill_null(0.0)
        ])
        
        if verbose:
            genes_with_splice_data = manifest_df.filter(pl.col("total_splice_sites") > 0).height
            print(f"[manifest] Added splice site data for {genes_with_splice_data:,} genes")
    
    # Add global index
    manifest_df = manifest_df.with_row_index("global_index")
    
    # Reorder columns for comprehensive readability
    base_columns = ["global_index", "gene_id", "gene_name", "gene_type", "chrom", "strand"]
    size_columns = ["gene_length", "start", "end"]
    splice_columns = ["total_splice_sites", "donor_sites", "acceptor_sites", "splice_density_per_kb"]
    file_columns = ["file_index", "file_name"]
    
    # Only include columns that exist in the DataFrame
    available_columns = manifest_df.columns
    ordered_columns = []
    
    for col_group in [base_columns, size_columns, splice_columns, file_columns]:
        for col in col_group:
            if col in available_columns:
                ordered_columns.append(col)
    
    # Add any remaining columns that weren't in our predefined groups
    for col in available_columns:
        if col not in ordered_columns:
            ordered_columns.append(col)
    
    manifest_df = manifest_df.select(ordered_columns)
    
    # Write manifest
    manifest_df.write_csv(output_path)
    
    if verbose:
        print(f"[manifest] Generated enhanced manifest with {manifest_df.height:,} unique genes")
        print(f"[manifest] Manifest saved to: {output_path}")
        print(f"[manifest] Total rows across all files: {total_rows:,}")
        
        # Show comprehensive statistics
        if "gene_name" in manifest_df.columns:
            with_names = manifest_df.filter(pl.col("gene_name").is_not_null()).height
            without_names = manifest_df.filter(pl.col("gene_name").is_null()).height
            print(f"[manifest] Genes with names: {with_names:,}, without names: {without_names:,}")
        
        if "gene_type" in manifest_df.columns:
            gene_type_counts = manifest_df.filter(pl.col("gene_type").is_not_null())["gene_type"].value_counts()
            print(f"[manifest] Gene type distribution:")
            for row in gene_type_counts.iter_rows():
                gene_type, count = row
                print(f"  {gene_type}: {count:,}")
        
        if "gene_length" in manifest_df.columns:
            length_stats = manifest_df.filter(pl.col("gene_length").is_not_null())["gene_length"]
            if not length_stats.is_empty():
                print(f"[manifest] Gene length statistics:")
                print(f"  Mean: {length_stats.mean():.0f} bp")
                print(f"  Median: {length_stats.median():.0f} bp")
                print(f"  Min: {length_stats.min():,} bp, Max: {length_stats.max():,} bp")
        
        if "splice_density_per_kb" in manifest_df.columns:
            density_stats = manifest_df.filter(pl.col("splice_density_per_kb").is_not_null())["splice_density_per_kb"]
            if not density_stats.is_empty():
                print(f"[manifest] Splice site density statistics:")
                print(f"  Mean: {density_stats.mean():.2f} sites/kb")
                print(f"  Median: {density_stats.median():.2f} sites/kb")
                
        print(f"[manifest] Columns included: {', '.join(manifest_df.columns)}")
    
    return output_path

# ---------------------------------------------------------------------------
# Stage 1 – build + enrich per-batch dataset
# ---------------------------------------------------------------------------

def build_base_dataset(
    *,
    gene_ids: Sequence[str],
    output_path: Path | str,
    data_handler: MetaModelDataHandler,
    kmer_sizes: Sequence[int] | None = (6,),
    enrichers: Sequence[str] | None = None,
    batch_rows: int = 500_000,
    overwrite: bool = False,
    initial_schema_cols: Optional[List[str]] = None,
    position_id_mode: str = 'genomic',  # NEW: Position identification strategy
    verbose: int = 1,
) -> Path:
    """Create a *single-batch* enriched Parquet training dataset.

    Parameters
    ----------
    gene_ids
        List of Ensembl gene IDs to include in this batch.
    output_path
        Destination Parquet (enriched, *before* TN down-sampling).
    data_handler
        Gives path access to the ``*_analysis_sequences_*.tsv`` files.
    position_id_mode
        Position identification strategy:
        - 'genomic': Current behavior (default, backward compatible)
        - 'transcript': Transcript-aware position identification
        - 'splice_aware': Same as transcript (emphasizes meta-learning)
    kmer_sizes, enrichers, batch_rows, overwrite, verbose
        Behaviour mirrors the arguments from previous utilities.
    """

    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"{output_path} exists – use --overwrite to regenerate")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------------
    # Configure position identification mode (transcript-aware enhancement)
    # ---------------------------------------------------------------------
    transcript_config = None
    if position_id_mode != 'genomic':
        if not TRANSCRIPT_AWARE_AVAILABLE:
            if verbose:
                print(f"      • Warning: transcript-aware module not available, falling back to genomic mode")
            position_id_mode = 'genomic'
        else:
            transcript_config = TranscriptAwareConfig(mode=position_id_mode)
            if verbose:
                print(f"      • Using {position_id_mode} position identification mode")
                print(f"      • Grouping columns: {transcript_config.get_grouping_columns()}")

    # Temporary location for the raw k-mer dataset (removed afterwards)
    tmp_path = output_path.with_suffix(".tmp.parquet")
    if tmp_path.exists():
        tmp_path.unlink()

    # ---------------------------------------------------------------------
    # 1) Assemble k-mer table ------------------------------------------------
    # ---------------------------------------------------------------------
    # Pass transcript-aware configuration to dataset builder
    dataset_kwargs = {
        'analysis_tsv_dir': str(Path(data_handler.meta_dir)),
        'output_path': str(tmp_path),
        'mode': "none",
        'target_genes': list(gene_ids),
        'top_n_genes': None,
        'kmer_sizes': list(kmer_sizes) if kmer_sizes else None,
        'batch_rows': batch_rows,
        'keep_sequence': False,
        'initial_schema_cols': initial_schema_cols,
        'compression': None,  # write tmp file uncompressed for maximum robustness
        'verbose': max(0, verbose - 1),
    }
    
    # Add transcript-aware configuration if available
    if transcript_config is not None:
        dataset_kwargs['transcript_config'] = transcript_config
        if verbose:
            print(f"      • Passing transcript config to dataset builder: {transcript_config.mode}")
        
    build_training_dataset(**dataset_kwargs)

    # ---------------------------------------------------------------------
    # 2) Feature enrichment --------------------------------------------------
    # ---------------------------------------------------------------------
    
    # Check if build_training_dataset actually created any output
    if not tmp_path.exists():
        if verbose:
            print(f"      • Warning: No training data found for batch genes: {list(gene_ids)[:5]}...")
        print(f"      • This usually means the selected genes don't have artifacts in {data_handler.meta_dir}")
        print(f"      • Skipping this batch (no training data to process)")
        
        # Create an empty parquet file so the rest of the pipeline doesn't break
        empty_df = pl.DataFrame({
            "gene_id": [],
            "position": [],
            "pred_type": [],
        })
        empty_df.write_parquet(output_path, compression="zstd")
        return output_path
    
    df = pl.read_parquet(tmp_path)
    
    # If the dataset is empty, handle gracefully
    if df.height == 0:
        if verbose:
            print(f"      • Warning: Empty dataset for batch genes: {list(gene_ids)[:5]}...")
        # Write empty dataset and return
        df.write_parquet(output_path, compression="zstd")
        tmp_path.unlink(missing_ok=True)
        return output_path
    
    df_enriched = apply_feature_enrichers(
        df,
        enrichers=enrichers,
        verbose=max(0, verbose - 1),
        fa=None,
        sa=None,
    )

    # Patch missing categorical / structural features ----------------------
    # (Originally performed post-hoc via `scripts/patch_gene_type.py` and
    #  `scripts/patch_structural_features.py`; integrating it here means
    #  every dataset created by the incremental builder—training or inference—
    #  is immediately fully patched.)
    df_enriched = builder_utils.fill_missing_gene_type(df_enriched)
    df_enriched = builder_utils.fill_missing_structural_features(df_enriched)

    # Authoritative n_splice_sites overwrite (splice_sites.tsv)
    df_enriched = builder_utils.update_n_splice_sites(df_enriched)

    df_enriched.write_parquet(output_path, compression="zstd")

    # Clean up temp file to save disk space
    tmp_path.unlink(missing_ok=True)

    if verbose:
        print(f"      • built + enriched dataset → {output_path} ({df_enriched.height:,} rows)")
    return output_path

def generate_gene_manifest(
    dataset_dir: str | Path,
    output_path: str | Path | None = None,
    verbose: int = 1,
) -> Path:
    """Generate a gene manifest for an existing training dataset.
    
    This function can be used to generate a manifest for datasets that were
    created without the manifest feature, or to regenerate the manifest.
    
    Parameters
    ----------
    dataset_dir
        Path to the training dataset directory (should contain a 'master' subdirectory
        with Parquet files).
    output_path
        Path for the manifest file. If None, creates 'gene_manifest.csv' in dataset_dir.
    verbose
        Verbosity level.
        
    Returns
    -------
    Path
        Path to the generated manifest file.
    """
    dataset_dir = Path(dataset_dir)
    master_dir = dataset_dir / "master"
    
    if not master_dir.exists():
        raise FileNotFoundError(f"Master directory not found: {master_dir}")
    
    if output_path is None:
        output_path = dataset_dir / "gene_manifest.csv"
    
    return _generate_gene_manifest(master_dir, output_path, verbose)

# ---------------------------------------------------------------------------
# Stage 2 – orchestrate incremental build over many batches
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _load_gene_features_df() -> Optional[pl.DataFrame]:
    """Load gene features DataFrame for ID/name lookups."""
    try:
        from meta_spliceai.system.config import Config
        gene_features_path = Path(Config.DATA_DIR) / "ensembl" / "spliceai_analysis" / "gene_features.tsv"
    except Exception:
        # Fallback to relative path
        gene_features_path = Path("data/ensembl/spliceai_analysis/gene_features.tsv")
    
    if not gene_features_path.exists():
        # Try alternative paths
        alternative_paths = [
            Path("data/ensembl/spliceai_analysis/gene_features.tsv"),
            Path("../data/ensembl/spliceai_analysis/gene_features.tsv"),
            Path("../../data/ensembl/spliceai_analysis/gene_features.tsv"),
        ]
        
        for alt_path in alternative_paths:
            if alt_path.exists():
                gene_features_path = alt_path
                break
        else:
            return None
    
    try:
        # Use schema_overrides to handle chrom column with string values like 'X'
        return pl.read_csv(
            gene_features_path, 
            separator="\t",
            schema_overrides={"chrom": pl.Utf8}
        )
    except Exception as e:
        print(f"[debug] Failed to load gene_features.tsv: {e}")
        return None


def _normalize_gene_identifiers(gene_ids: Set[str], verbose: int = 1) -> Tuple[Set[str], Dict[str, str]]:
    """
    Normalize gene identifiers to Ensembl IDs and create a mapping for display.
    
    Parameters
    ----------
    gene_ids
        Set of gene identifiers (could be Ensembl IDs or gene names)
    verbose
        Verbosity level
        
    Returns
    -------
    Tuple[Set[str], Dict[str, str]]
        Normalized gene IDs (Ensembl format) and mapping from original to normalized
    """
    if not gene_ids:
        return set(), {}
    
    # Load gene features for lookup
    gene_features_df = _load_gene_features_df()
    if gene_features_df is None:
        if verbose:
            print("[validation] Warning: Could not load gene_features.tsv for ID/name lookup")
            print("[validation] This means gene names cannot be converted to Ensembl IDs")
            print("[validation] Please ensure gene_features.tsv exists or use Ensembl IDs directly")
        
        # Check if any gene IDs look like gene names (not Ensembl IDs)
        gene_names = {gid for gid in gene_ids if not gid.startswith("ENSG")}
        if gene_names:
            if verbose:
                print(f"[validation] Found gene names that cannot be converted: {list(gene_names)}")
                print("[validation] These genes will be skipped. Please use Ensembl IDs instead.")
            # Only keep genes that look like Ensembl IDs
            ensembl_ids = {gid for gid in gene_ids if gid.startswith("ENSG")}
            return ensembl_ids, {gid: gid for gid in ensembl_ids}
        else:
            # All look like Ensembl IDs, proceed
            return gene_ids, {gid: gid for gid in gene_ids}
    
    # Create lookup dictionaries
    id_to_name = dict(zip(gene_features_df["gene_id"], gene_features_df["gene_name"]))
    name_to_id = dict(zip(gene_features_df["gene_name"], gene_features_df["gene_id"]))
    
    # Remove empty gene names
    id_to_name = {k: v for k, v in id_to_name.items() if v}
    name_to_id = {k: v for k, v in name_to_id.items() if v}
    
    normalized_ids = set()
    original_to_normalized = {}
    failed_conversions = []
    
    for gene_id in gene_ids:
        normalized_id = None
        
        # Check if it's already an Ensembl ID
        if gene_id.startswith("ENSG"):
            normalized_id = gene_id
        # Check if it's a gene name that needs conversion
        elif gene_id in name_to_id:
            normalized_id = name_to_id[gene_id]
        # If neither, this is a problem
        else:
            failed_conversions.append(gene_id)
            continue
        
        if normalized_id:
            normalized_ids.add(normalized_id)
            original_to_normalized[gene_id] = normalized_id
    
    if failed_conversions and verbose:
        print(f"[validation] Warning: Could not convert {len(failed_conversions)} gene identifiers:")
        print(f"  Failed conversions: {failed_conversions}")
        print("  These genes will be skipped. Please use valid Ensembl IDs or gene names.")
    
    if verbose >= 2:
        converted_count = len([gid for gid in gene_ids if gid in name_to_id])
        if converted_count > 0:
            print(f"[validation] Converted {converted_count} gene names to Ensembl IDs")
    
    return normalized_ids, original_to_normalized


def _get_display_names(gene_ids: Set[str], id_to_name_mapping: Dict[str, str]) -> List[str]:
    """Get display names for gene IDs, preferring gene names over Ensembl IDs."""
    display_names = []
    for gene_id in gene_ids:
        # Try to get gene name, fallback to Ensembl ID
        display_name = id_to_name_mapping.get(gene_id, gene_id)
        display_names.append(display_name)
    return display_names


def incremental_build_training_dataset(
    *,
    patch_dataset: bool = False,
    eval_dir: str | None = None,
    output_dir: str | Path = "train_dataset_trimmed",
    n_genes: int = 20_000,
    subset_policy: str = "error_total",
    batch_size: int = 1_000,
    kmer_sizes: Sequence[int] | None = (6,),
    enrichers: Sequence[str] | None = None,
    downsample_kwargs: Optional[Dict] = None,
    batch_rows: int = 500_000,
    gene_types: Optional[Sequence[str]] = None,
    additional_gene_ids: Optional[Sequence[str]] = None,
    initial_schema_cols: Optional[List[str]] = None,
    run_workflow: bool = False,
    workflow_kwargs: Optional[Dict] = None,
    overwrite: bool = False,
    generate_manifest: bool = True,
    position_id_mode: str = 'genomic',  # NEW: Position identification strategy
    verbose: int = 1,
) -> Path:

    """Full incremental build pipeline.

    Parameters
    ----------
    eval_dir
        Root *spliceai_eval* directory (auto-detected if *None*).
    output_dir
        Directory that will contain per-batch Parquets and the *master* dataset.
    n_genes, subset_policy
        Controls which genes are selected (mirrors existing API).
    batch_size
        Number of genes per batch.
    kmer_sizes, enrichers
        Feature extraction parameters.
    downsample_kwargs
        Passed straight to ``downsample_tn`` (e.g. ``dict(hard_prob_thresh=0.15)``).
    overwrite
        If *True*, regenerate existing batch / master artefacts.

    Returns
    -------
    Path
        Path to the *master* partitioned dataset directory.
    """

    # Map alias: allow 'explicit' as synonym for 'custom' gene list policy
    if subset_policy.lower() == "explicit":
        subset_policy = "custom"

    # ------------------------------------------------------------------
    # Resolve output path ------------------------------------------------
    # ------------------------------------------------------------------
    output_dir = Path(output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = Path.cwd() / output_dir

    # Create the directory tree -----------------------------------------------------------------
    # (raises a helpful error if the *parent* of an absolute path doesn't exist)
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Parent directory for --output-dir does not exist: {output_dir.parent}. "
            "Please create it first or mount the remote path." 
        ) from exc
    batch_dir = output_dir
    master_dir = output_dir / "master"
    master_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Normalize gene identifiers FIRST (before gene selection)
    # ------------------------------------------------------------------
    if verbose:
        print(f"[incremental-builder] Normalizing gene identifiers...")
    
    # Normalize additional gene IDs to Ensembl IDs
    normalized_additional_genes, id_mapping = _normalize_gene_identifiers(set(additional_gene_ids or []), verbose)
    
    # Create reverse mapping for display
    reverse_mapping = {v: k for k, v in id_mapping.items()}
    
    if verbose:
        print(f"[incremental-builder] Normalized {len(additional_gene_ids or [])} additional genes to {len(normalized_additional_genes)} Ensembl IDs")
        if id_mapping:
            # Show only a summary and first few examples to avoid cluttering output
            num_mappings = len(id_mapping)
            if num_mappings <= 10:
                print(f"[incremental-builder] Gene name mappings:")
                for gene_name, ensembl_id in id_mapping.items():
                    print(f"  {gene_name} → {ensembl_id}")
            else:
                print(f"[incremental-builder] Gene name mappings ({num_mappings} total, showing first 5):")
                for i, (gene_name, ensembl_id) in enumerate(id_mapping.items()):
                    if i >= 5:
                        break
                    print(f"  {gene_name} → {ensembl_id}")
                print(f"  ... and {num_mappings - 5} more")

    # ------------------------------------------------------------------
    # Gene selection (MOVED BEFORE workflow execution) ----------------
    # ------------------------------------------------------------------
    dh = MetaModelDataHandler(eval_dir=eval_dir)

    # Check if we need to expand beyond existing artifacts (regardless of gene_types)
    if run_workflow and subset_policy.lower() in ["error_total", "error_fp", "error_fn", "random", "rand", "all"]:
        # When --run-workflow is specified, select from full genome instead of just existing artifacts
        if verbose:
            gene_type_msg = f" (gene types: {gene_types})" if gene_types else ""
            print(f"[incremental-builder] --run-workflow specified: selecting {n_genes} genes from full genome{gene_type_msg}")
            print(f"[incremental-builder] Will generate artifacts for selected genes during workflow execution")
            
        # Use full gene features for selection instead of existing artifacts
        try:
            gene_features_df = _load_gene_features_df()
            if gene_features_df is not None:
                # Apply gene type filter if specified
                if gene_types:
                    gene_features_df = gene_features_df.filter(pl.col("gene_type").is_in(gene_types))
                    if verbose:
                        print(f"[incremental-builder] Filtered to gene types {gene_types}: {len(gene_features_df):,} genes available")
                
                # Pre-filter genes to only include those with splice sites (critical for splicing analysis)
                try:
                    from meta_spliceai.system.genomic_resources import Registry
                    registry = Registry()
                    splice_sites_path = Path(registry.cfg.data_root) / "splice_sites_enhanced.tsv"
                    
                    if not splice_sites_path.exists():
                        # Fallback to regular splice_sites.tsv
                        splice_sites_path = Path(registry.cfg.data_root) / "splice_sites.tsv"
                    
                    if splice_sites_path.exists():
                        if verbose:
                            print(f"[incremental-builder] Pre-filtering genes by splice site coverage using: {splice_sites_path.name}")
                        
                        # Load splice sites and aggregate by gene
                        splice_sites_df = pl.read_csv(
                            splice_sites_path,
                            separator='\t',
                            schema_overrides={'chrom': pl.Utf8}
                        )
                        
                        # Count splice sites per gene
                        gene_splice_counts = splice_sites_df.group_by("gene_id").agg([
                            pl.len().alias("splice_site_count")
                        ])
                        
                        # Filter to genes with at least 1 splice site
                        genes_with_splice_sites = set(
                            gene_splice_counts
                            .filter(pl.col("splice_site_count") > 0)["gene_id"]
                            .to_list()
                        )
                        
                        # Filter gene_features_df to only include genes with splice sites
                        initial_count = len(gene_features_df)
                        gene_features_df = gene_features_df.filter(
                            pl.col("gene_id").is_in(genes_with_splice_sites)
                        )
                        filtered_count = len(gene_features_df)
                        
                        if verbose:
                            print(f"[incremental-builder] Pre-filtered {initial_count:,} → {filtered_count:,} genes (removed {initial_count - filtered_count:,} without splice sites)")
                    else:
                        if verbose:
                            print(f"[incremental-builder] Warning: splice sites file not found, skipping pre-filter")
                except Exception as e:
                    if verbose:
                        print(f"[incremental-builder] Warning: Could not pre-filter by splice sites: {e}")
                        print(f"[incremental-builder] Continuing with all genes (validation will filter later)")
                
                available_genes = gene_features_df["gene_id"].to_list()
                
                if subset_policy.lower() == "all":
                    # Select all available genes (ignore n_genes and additional genes)
                    all_gene_ids = available_genes
                    if verbose:
                        print(f"[incremental-builder] Selected all {len(all_gene_ids):,} available genes")
                else:
                    # Add additional genes first
                    selected_genes = list(normalized_additional_genes)
                    
                    # Then select genes based on policy
                    remaining_genes = [g for g in available_genes if g not in selected_genes]
                    additional_needed = max(0, n_genes - len(selected_genes))
                        
                    if additional_needed > 0 and len(remaining_genes) >= additional_needed:
                        if subset_policy.lower() in ["random", "rand"]:
                            # Random selection for diversity
                            import random
                            random.seed(42)  # Reproducible selection
                            selected_genes.extend(random.sample(remaining_genes, additional_needed))
                            if verbose:
                                print(f"[incremental-builder] Using random selection for biological diversity")
                        else:
                            # For error-based policies, use random selection from full genome
                            # (since we don't have error data for all genes yet)
                            import random
                            random.seed(42)  # Reproducible selection
                            selected_genes.extend(random.sample(remaining_genes, additional_needed))
                            if verbose:
                                print(f"[incremental-builder] Using random selection from full genome (error data not available for all genes)")
                        
                    all_gene_ids = selected_genes
                if verbose:
                    print(f"[incremental-builder] Selected {len(all_gene_ids)} genes from full genome for artifact generation")
            else:
                raise Exception("Could not load gene features for full genome selection")
                    
        except Exception as e:
            if verbose:
                print(f"[incremental-builder] Warning: Could not select from full genome: {e}")
                print(f"[incremental-builder] Falling back to existing artifact-based selection")
            
            # Fallback to existing artifact-based selection
            if gene_types is None:
                _, all_gene_ids = subset_analysis_sequences(
                    data_handler=dh,
                    n_genes=n_genes,
                    subset_policy=subset_policy,
                    aggregated=True,
                    additional_gene_ids=list(normalized_additional_genes),
                    use_effective_counts=True,
                    verbose=max(0, verbose - 1),
                )
            else:
                # Need gene-type filtering → operate on full positions_df
                pos_df = dh.load_splice_positions(aggregated=True)
                _, all_gene_ids = subset_positions_dataframe(
                    pos_df,
                    n_genes=n_genes,
                    subset_policy=subset_policy,
                    gene_types=list(gene_types),
                    additional_gene_ids=list(normalized_additional_genes),
                    use_effective_counts=True,
                    verbose=max(0, verbose - 1),
                )
    elif gene_types is None:
        # Standard artifact-based selection (no gene types, no workflow)
        _, all_gene_ids = subset_analysis_sequences(
            data_handler=dh,
            n_genes=n_genes,
            subset_policy=subset_policy,
            aggregated=True,
            additional_gene_ids=list(normalized_additional_genes),  # Use normalized Ensembl IDs
            use_effective_counts=True,
            verbose=max(0, verbose - 1),
        )
    else:
        # Gene-type filtering with existing artifacts (no workflow)
        pos_df = dh.load_splice_positions(aggregated=True)
        _, all_gene_ids = subset_positions_dataframe(
            pos_df,
            n_genes=n_genes,
            subset_policy=subset_policy,
            gene_types=list(gene_types),
            additional_gene_ids=list(normalized_additional_genes),  # Use normalized Ensembl IDs
            use_effective_counts=True,
            verbose=max(0, verbose - 1),
        )
    
    # ------------------------------------------------------------------
    # Quick verification (optional) -------------------------------------
    # Heavy verification loads the large splice-positions table.  Skip this
    # step when we already have an explicit/custom gene list so that we avoid
    # multi-GB memory spikes during meta-model evaluation.
    # ------------------------------------------------------------------
    if subset_policy.lower() not in ("custom", "explicit"):
        try:
            builder_utils.verify_gene_selection(
                dh,
                gene_ids=all_gene_ids,
                expected_gene_types=gene_types,
                raise_error=True,
                verbose=max(0, verbose - 1),
            )
        except Exception as e:
            # Fail fast so that errors are obvious during incremental builds
            raise
    elif verbose:
        print("[incremental-builder] Skipping verify_gene_selection for explicit/custom gene list")

    if verbose:
        # Use enhanced output utility
        output_enhancer = create_output_enhancer(verbose)
        output_enhancer.print_gene_selection_summary(all_gene_ids, n_genes, subset_policy, normalized_additional_genes)
    
    # ------------------------------------------------------------------
    # Validate artifact coverage for requested genes
    # ------------------------------------------------------------------
    if verbose:
        print(f"[incremental-builder] Validating artifact coverage for {len(all_gene_ids)} selected genes...")
    
    # Check what genes are actually available in the analysis sequence artifacts
    available_genes = set()
    try:
        # First try to use aggregated positions file for complete gene list
        try:
            positions_path = os.path.join(dh.meta_dir, "full_splice_positions_enhanced.tsv")
            if os.path.exists(positions_path):
                # Use aggregated file - much faster and complete
                positions_df = pl.scan_csv(
                    positions_path,
                    separator=dh.separator,
                    schema_overrides={"chrom": pl.Utf8}
                )
                available_genes = set(positions_df.select("gene_id").unique().collect()["gene_id"].to_list())
                if verbose:
                    print(f"[incremental-builder] Found {len(available_genes)} unique genes in aggregated positions file")
            else:
                raise FileNotFoundError("Aggregated positions file not found")
        except Exception:
            # Fallback: Use aggregated positions file if chunked sampling fails
            # This is more reliable than random chunk sampling which can miss genes
            try:
                positions_path = os.path.join(dh.meta_dir, "full_splice_positions_enhanced.tsv")
                if os.path.exists(positions_path):
                    # Use aggregated file as fallback - much more reliable
                    positions_df = pl.scan_csv(
                        positions_path,
                        separator=dh.separator,
                        schema_overrides={"chrom": pl.Utf8}
                    )
                    available_genes = set(positions_df.select("gene_id").unique().collect()["gene_id"].to_list())
                    if verbose:
                        print(f"[incremental-builder] Found {len(available_genes)} unique genes in aggregated positions file (fallback)")
                else:
                    if verbose:
                        print("[incremental-builder] Warning: No aggregated positions file found for validation")
                    # If no aggregated file exists, skip validation entirely
                    available_genes = set()
            except Exception as e:
                if verbose >= 2:
                    print(f"[incremental-builder] Could not validate artifact coverage: {e}")
                # Skip validation entirely if both methods fail
                available_genes = set()
        
        # Now all_gene_ids are already Ensembl IDs, so we can compare directly
        missing_genes = set(all_gene_ids) - available_genes
        
        # Ensure missing_genes is always defined for later use
        if 'missing_genes' not in locals():
            missing_genes = set()
        
        if missing_genes and len(available_genes) > 0:  # Only check if we found some genes
            missing_count = len(missing_genes)
            coverage_pct = (len(set(all_gene_ids) - missing_genes) / len(set(all_gene_ids))) * 100
            
            if verbose:
                print(f"[incremental-builder] Artifact coverage: {coverage_pct:.1f}% ({len(set(all_gene_ids)) - missing_count}/{len(set(all_gene_ids))} genes)")
                
            if missing_count > 0:
                if verbose:
                    print(f"[incremental-builder] Missing {missing_count} genes from artifacts")
                    
                    # Get display names for missing genes
                    gene_features_df = _load_gene_features_df()
                    id_to_name = {}
                    if gene_features_df is not None:
                        id_to_name = dict(zip(gene_features_df["gene_id"], gene_features_df["gene_name"]))
                        id_to_name = {k: v for k, v in id_to_name.items() if v}
                    
                    # Convert missing gene IDs to display names
                    missing_display_names = []
                    for gene_id in missing_genes:
                        display_name = id_to_name.get(gene_id, gene_id)
                        missing_display_names.append(display_name)
                    
                    if verbose >= 2 and missing_count <= 10:
                        print(f"  Missing genes: {missing_display_names}")
                    elif verbose >= 2:
                        print(f"  Missing genes (first 10): {missing_display_names[:10]}...")
                
                # If ANY genes are missing, we need to run the workflow
                if missing_count > 0:
                    if not run_workflow:
                        print(f"[incremental-builder] CRITICAL: Missing {missing_count} requested genes from artifacts")
                        print(f"[incremental-builder] Automatically enabling --run-workflow to generate missing gene data")
                        run_workflow = True
                    else:
                        print(f"[incremental-builder] Note: {missing_count} genes missing, but --run-workflow is already enabled")
        
    except Exception as e:
        if verbose >= 2:
            print(f"[incremental-builder] Could not validate artifact coverage: {e}")
        # Continue anyway - the iterative loading will handle missing genes gracefully
    
    # ------------------------------------------------------------------
    # Critical validation: Ensure gene count meets expectations
    # ------------------------------------------------------------------
    expected_max_genes = n_genes + len(normalized_additional_genes)
    expected_min_genes = max(n_genes, len(normalized_additional_genes))  # At least n_genes, or all additional genes if more
    
    # For error-based policies, we should get at least n_genes total
    # Additional genes might overlap with top-N genes, so total could be anywhere from n_genes to n_genes + additional_genes
    strict_policies = ["error_total", "error_fp", "error_fn"]
    is_strict_policy = subset_policy.lower() in strict_policies
    
    # Skip validation for 'all' policy since it intentionally selects all available genes
    if subset_policy.lower() != "all" and len(all_gene_ids) > expected_max_genes:
        raise ValueError(
            f"Gene selection returned {len(all_gene_ids)} genes, which exceeds the expected maximum of {expected_max_genes} "
            f"(n_genes={n_genes} + additional_genes={len(normalized_additional_genes)}). "
            f"This would be too costly to proceed. Please check your gene selection logic or reduce n_genes."
        )
    elif subset_policy.lower() != "all" and len(all_gene_ids) < expected_min_genes:
        if is_strict_policy and not run_workflow:
            raise ValueError(
                f"Gene selection returned only {len(all_gene_ids)} genes, but expected at least {expected_min_genes} "
                f"(n_genes={n_genes}, additional_genes={len(normalized_additional_genes)}). "
                f"For policy '{subset_policy}', we should be able to find at least {n_genes} genes. "
                f"This suggests an issue with the gene selection logic or insufficient genes in the dataset."
            )
        elif len(all_gene_ids) < expected_min_genes and run_workflow:
            if verbose:
                print(f"[incremental-builder] Note: Selected {len(all_gene_ids)} genes from existing artifacts, "
                      f"but --run-workflow is enabled to generate artifacts for additional genes.")
                print(f"[incremental-builder] Will expand to full gene set during workflow execution.")
        else:
            if verbose:
                print(f"[incremental-builder] Note: Selected {len(all_gene_ids)} genes, which is less than the requested minimum {expected_min_genes}. This can happen with policies like 'random' or 'custom' when insufficient genes are available.")
    
    # Informative message about gene overlap
    if len(normalized_additional_genes) > 0:
        overlap_count = len(normalized_additional_genes) - (len(all_gene_ids) - max(0, len(all_gene_ids) - n_genes))
        if overlap_count > 0 and verbose:
            print(f"[incremental-builder] Note: {overlap_count} additional genes are already among the top {n_genes} genes selected by policy '{subset_policy}'.")

    # ------------------------------------------------------------------
    # Enhanced splice prediction workflow (IF requested) --------------
    # ------------------------------------------------------------------
    if run_workflow:
        if verbose:
            output_enhancer.print_workflow_start(missing_genes)
            print("[incremental-builder] Running enhanced splice-prediction workflow on selected genes …")

        # --------------------------------------------------------------
        # PRE-FLIGHT VALIDATION: Check genes have splice sites --------
        # --------------------------------------------------------------
        genes_to_validate = list(missing_genes) if missing_genes else list(all_gene_ids)
        if genes_to_validate:
            try:
                from meta_spliceai.system.genomic_resources.validators import validate_gene_selection
                from meta_spliceai.system.config import Config
                
                # Get data directory from system config
                data_dir = Path(Config.PROJ_DIR) / "data" / "ensembl"
                valid_genes, invalid_summary = validate_gene_selection(
                    genes_to_validate,
                    data_dir,
                    min_splice_sites=1,
                    fail_on_invalid=False,  # Filter out invalid genes instead of failing
                    verbose=(verbose >= 1)
                )
                
                # Update gene lists to only include valid genes
                if missing_genes:
                    missing_genes = set(valid_genes)
                    if verbose and invalid_summary['no_splice_sites']:
                        print(f"[incremental-builder] ⚠️  Filtered out {len(invalid_summary['no_splice_sites'])} genes without splice sites")
                
                # Update all_gene_ids to only include valid genes
                all_gene_ids = [g for g in all_gene_ids if g in valid_genes]
                
                if not valid_genes:
                    print("[incremental-builder] ❌ ERROR: No genes with splice sites found!")
                    print("[incremental-builder] Consider using --gene-types protein_coding lncRNA")
                    raise SystemExit(1)
                    
            except ImportError:
                if verbose:
                    print("[incremental-builder] ⚠️  Splice site validation unavailable (validators.py not found)")
        
        _wk = workflow_kwargs.copy() if workflow_kwargs else {}

        # --------------------------------------------------------------
        # Determine whether genomic files need extraction -------------
        # --------------------------------------------------------------
        if "do_extract_sequences" not in _wk and "do_extract_splice_sites" not in _wk:
            existing_files = check_genomic_files_exist()
            _wk["do_extract_sequences"] = not existing_files.get("genomic_sequences", False)
            _wk["do_extract_annotations"] = not existing_files.get("annotations", False)
            _wk["do_extract_splice_sites"] = not existing_files.get("splice_sites", False)
            _wk["do_find_overlaping_genes"] = True  # safe default
            if verbose:
                print(
                    "[incremental-builder] Genomic file status → extract_sequences=%s, annotations=%s, splice_sites=%s"
                    % (
                        _wk["do_extract_sequences"],
                        _wk["do_extract_annotations"],
                        _wk["do_extract_splice_sites"],
                    )
                )

        # ------------------------------------------------------------------
        # CRITICAL FIX: Pass selected genes to workflow (not just missing genes)
        # ------------------------------------------------------------------
        if "target_genes" not in _wk:
            # For new dataset creation (like 5000-gene model), process ALL selected genes
            if run_workflow and len(all_gene_ids) > len(missing_genes if 'missing_genes' in locals() else []):
                # We're building a new dataset - process all selected genes
                _wk["target_genes"] = list(all_gene_ids)
                if verbose:
                    print(f"[incremental-builder] Passing {len(all_gene_ids)} SELECTED genes to prediction workflow.")
                    print(f"[incremental-builder] This will generate artifacts for the complete 5000-gene dataset.")
                    if verbose >= 2 and len(all_gene_ids) <= 10:
                        print(f"  Selected genes to process: {list(all_gene_ids)}")
            elif missing_genes:
                # Standard case - only process missing genes
                _wk["target_genes"] = list(missing_genes)
                if verbose:
                    print(f"[incremental-builder] Passing {len(missing_genes)} MISSING genes to prediction workflow.")
                    if verbose >= 2 and len(missing_genes) <= 10:
                        print(f"  Missing genes to process: {list(missing_genes)}")
            else:
                # If no missing genes and not building new dataset, skip workflow
                if verbose:
                    print("[incremental-builder] No missing genes detected - skipping workflow execution.")
                run_workflow = False
        else:
            # If target_genes is explicitly set, use it but warn if it doesn't match
            if verbose:
                print(f"[incremental-builder] Using explicit target_genes: {len(_wk['target_genes'])} genes")

        # Note: gene_types filtering is now handled in the gene selection phase above
        # The selected gene list (all_gene_ids) already incorporates any gene type filtering

        # Add position identification mode to workflow kwargs
        _wk['position_id_mode'] = position_id_mode
        
        run_enhanced_splice_prediction_workflow(verbosity=max(0, verbose - 1), **_wk)

    # ------------------------------------------------------------------
    # Dataset building with selected genes ----------------------------
    # ------------------------------------------------------------------

    batches = _chunks(all_gene_ids, batch_size)

    # ds_kwargs = dict(format="parquet", partitioning="hive", existing_data_behavior="overwrite_or_ignore")
    # NOTE: 
    # PyArrow 15.0+ requires a Partitioning object or list; passing the string
    # 'hive' now raises a ValueError.  We do not currently rely on directory
    # partitioning, so omit the argument entirely and allow a flat layout.
    ds_kwargs = dict(format="parquet", existing_data_behavior="overwrite_or_ignore")
    
    downsample_kwargs = downsample_kwargs or {}

    # Collect all trimmed Parquet paths so that we can write the complete
    # master dataset in a single call *after* the loop.  This avoids the
    # clobber-by-filename problem that previously dropped earlier batches.
    trim_paths: list[Path] = []

    # Enhanced batch processing with color-coded progress
    output_enhancer = create_output_enhancer(verbose)
    batches_list = list(batches)  # Convert generator to list once
    total_batches = len(batches_list)  # Get count from the list
    
    for batch_ix, gene_batch in enumerate(batches_list, 1):
        prefix = f"batch_{batch_ix:05d}"
        raw_path = batch_dir / f"{prefix}_raw.parquet"
        trim_path = batch_dir / f"{prefix}_trim.parquet"

        if trim_path.exists() and not overwrite:
            if verbose:
                print(f"[{prefix}] trim Parquet exists – skipping …")
            # Re-use the existing trimmed Parquet later when we assemble the master dataset
            trim_paths.append(trim_path)
            continue

        if verbose:
            output_enhancer.print_batch_header(batch_ix, total_batches, len(gene_batch))

        # 1. Build + enrich -------------------------------------------------
        build_base_dataset(
            gene_ids=gene_batch,
            output_path=raw_path,
            data_handler=dh,
            kmer_sizes=kmer_sizes,
            enrichers=enrichers,
            batch_rows=batch_rows,
            overwrite=overwrite,
            initial_schema_cols=initial_schema_cols,
            position_id_mode=position_id_mode,  # Pass through transcript-aware config
            verbose=verbose,
        )

        # 2. Down-sample TNs ----------------------------------------------
        if verbose:
            print(f"[{prefix}] down-sampling TNs …")
        df_trim = downsample_tn(raw_path, trim_path, **downsample_kwargs)

        if verbose:
            print(f"[{prefix}] trimmed dataset → {trim_path} ({len(df_trim):,} rows)")

        # Optionally remove the raw Parquet to save disk after trimming
        raw_path.unlink(missing_ok=True)

        # Remember path for final master assembly ---------------------------------
        trim_paths.append(trim_path)

    # ------------------------------------------------------------------
    # Write combined master dataset in a single pass -------------------
    # ------------------------------------------------------------------
    if trim_paths:
        if verbose:
            print(f"\n[incremental-builder] Linking {len(trim_paths):,} trimmed batches into master dataset …")
        master_dir.mkdir(parents=True, exist_ok=True)
        # import os, shutil
        for i, pth in enumerate(trim_paths, 1):
            dest = master_dir / f"batch_{i:05d}.parquet"
            if dest.exists():
                if overwrite:
                    dest.unlink()
                else:
                    continue
            try:
                os.link(pth, dest)  # hard-link, O(1) and space-efficient
            except OSError as e:
                # os.link fails if source and destination are on different filesystems or hard-link not supported
                # shutil.copy2 will error if src and dest are the *same* file, so guard against that
                if dest.resolve() == pth.resolve():
                    # Already linked/copied in a previous run – nothing to do
                    continue
                shutil.copy2(pth, dest)

    if verbose:
        # Attempt to summarise the assembled master dataset
        try:
            master_ds = ds.dataset(master_dir)
            total_rows = master_ds.count_rows()  # PyArrow ≥ 14.0
        except Exception:
            try:
                total_rows = len(master_ds.to_table())  # fallback
            except Exception:
                total_rows = "?"
        rows_display = (f"{total_rows:,}" if isinstance(total_rows, (int, float)) else str(total_rows))
        
        # Use enhanced output utility for completion summary
        output_enhancer.print_completion_summary(master_dir, rows_display, all_gene_ids, len(trim_paths))
    
    # ------------------------------------------------------------------
    # Generate gene manifest --------------------------------------------
    # ------------------------------------------------------------------
    if generate_manifest and trim_paths:
        try:
            # Check if the dataset has gene_id column (raw dataset) or is downsampled (training dataset)
            sample_path = trim_paths[0]
            sample_schema = pl.scan_parquet(sample_path).collect_schema()
            
            if "gene_id" in sample_schema:
                # Raw dataset with gene information - generate full manifest
                manifest_path = _generate_gene_manifest(master_dir, verbose=verbose)
                
                # Validate final dataset contains only expected genes
                if verbose:
                    print("[incremental-builder] Validating final dataset gene count...")
                    manifest_df = pl.read_csv(manifest_path, separator="\t")
                    final_gene_count = manifest_df.height
                    
                    print(f"[incremental-builder] Final dataset validation:")
                    print(f"  Expected genes: {len(all_gene_ids)}")
                    print(f"  Actual genes in dataset: {final_gene_count}")
                    
                    if final_gene_count > len(all_gene_ids):
                        print(f"[incremental-builder] WARNING: Dataset contains {final_gene_count - len(all_gene_ids)} more genes than expected!")
                        print(f"  This may indicate an issue with gene filtering during dataset building.")
                    elif final_gene_count < len(all_gene_ids):
                        print(f"[incremental-builder] Note: Dataset contains {len(all_gene_ids) - final_gene_count} fewer genes than selected.")
                        print(f"  This can happen if some selected genes had no valid training data.")
                    else:
                        print(f"[incremental-builder] ✅ Gene count validation passed!")
            else:
                # Downsampled training dataset - skip manifest generation
                if verbose:
                    print("[incremental-builder] Skipping gene manifest generation for downsampled training dataset")
                    print("  (Training datasets don't contain gene metadata columns)")
                    
        except Exception as e:
            if verbose:
                print(f"[incremental-builder] Warning: Failed to generate gene manifest: {e}")
    
    # ------------------------------------------------------------------
    # Optional post-build dataset patching ------------------------------
    # ------------------------------------------------------------------
    if patch_dataset:
        if verbose:
            print("[incremental-builder] Running post-build patch scripts …")
        import subprocess, sys
        # Project root is 4 levels up: .../splice-surveyor
        project_root = Path(__file__).resolve().parents[4]
        patch_scripts = [
            "patch_structural_features.py",
            "patch_gene_type.py",
        ]
        for script in patch_scripts:
            script_path = project_root / "scripts" / script
            if not script_path.exists():
                if verbose:
                    print(f"  – Warning: {script_path} not found; skipping.")
                continue
            if verbose:
                print(f"  – {script} …")
            try:
                subprocess.run([sys.executable, str(script_path), str(master_dir), "--inplace"], check=True)
            except subprocess.CalledProcessError as e:
                print(f"[incremental-builder] Patch script {script} failed: {e}", file=sys.stderr)
                # Do not abort entire build; continue with next script
    return master_dir

# ---------------------------------------------------------------------------
# CLI entry-point -----------------------------------------------------------
# ---------------------------------------------------------------------------

T = TypeVar('T')


def parse_flexible_list(values: List[str], item_type: Callable[[str], T] = str) -> List[T]:
    """
    Parse a list that can be either space-separated or comma-separated.
    
    Args:
        values: List of string values from argparse
        item_type: Function to convert string to desired type (e.g., int, str)
    
    Returns:
        List of parsed values
        
    Examples:
        # Space-separated: --kmer-sizes 3 5
        parse_flexible_list(['3', '5'], int) → [3, 5]
        
        # Comma-separated: --kmer-sizes 3,5  
        parse_flexible_list(['3,5'], int) → [3, 5]
    """
    result = []
    for value in values:
        if ',' in value:
            # Comma-separated within this value
            result.extend([item_type(x.strip()) for x in value.split(',') if x.strip()])
        else:
            # Space-separated (single value)
            result.append(item_type(value))
    return result


class FlexibleListAction(argparse.Action):
    """Custom argparse action that supports both space and comma-separated values."""
    
    def __init__(self, option_strings, dest, item_type=str, **kwargs):
        self.item_type = item_type
        super().__init__(option_strings, dest, **kwargs)
    
    def __call__(self, parser, namespace, values, option_string=None):
        if values is None:
            setattr(namespace, self.dest, [])
        else:
            parsed_values = parse_flexible_list(values, self.item_type)
            setattr(namespace, self.dest, parsed_values)


if __name__ == "__main__":
    import argparse, sys, json

    p = argparse.ArgumentParser(
        description="Incrementally build the meta-model training dataset in gene batches.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--n-genes", 
        type=int, 
        default=20_000, 
        help="Total number of genes to include in the training dataset. Lower values (e.g., 1000) recommended for testing."
    )
    p.add_argument(
        "--subset-policy", 
        type=str, 
        default="error_total", 
        help="Gene selection strategy. Valid options: 'error_total' (genes with most errors), 'error_fp' (most false positives), "
             "'error_fn' (most false negatives), 'random' (random sampling), 'custom' (use provided gene ids), 'all' (use all available genes)."
    )
    p.add_argument(
        "--batch-size", 
        type=int, 
        default=1_000, 
        help="Number of genes to process in each batch. Lower values reduce memory usage but increase processing time."
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default="train_dataset_trimmed",
        help=(
            "Directory to write per-batch Parquet files and the final master dataset. "
            "If the value is an *absolute* path (starts with / or ~), it will be used as-is. "
            "Otherwise it is interpreted relative to the current working directory."
        ),
    )
    p.add_argument(
        "--overwrite", 
        action="store_true", 
        help="Overwrite existing artefacts. If not set, will attempt to resume previous run."
    )
    p.add_argument(
        "--run-workflow", 
        action="store_true", 
        help="Run the enhanced splice prediction workflow first to generate required prediction files before dataset building."
    )
    p.add_argument(
        "--workflow-kwargs", 
        type=str, 
        default="{}", 
        help="JSON dict with kwargs for the prediction workflow. Common options: {'eval_dir': '/path/to/dir', "
             "'gene_types': ['protein_coding'], 'target_genes': ['STMN2', 'UNC13A']}."
    )
    p.add_argument(
        "--gene-types", 
        type=str, 
        nargs="*", 
        default=None, 
        help="Restrict gene selection to these gene types. Common values: 'protein_coding', 'lncRNA', 'pseudogene', 'miRNA', etc."
    )
    p.add_argument(
        "--gene-ids-file",
        type=str,
        default=None,
        help="Plain text or CSV/TSV file containing custom gene IDs to build. Implies --subset-policy custom if used."
    )
    p.add_argument(
        "--gene-col",
        type=str,
        default="gene_id",
        help="Column name with gene IDs when --gene-ids-file points to a CSV/TSV file."
    )
    p.add_argument(
        "--hard-prob", 
        type=float, 
        default=0.15, 
        help="TN down-sampling hard_prob_thresh: probability threshold to identify 'hard' negatives that will be preserved."
    )
    p.add_argument(
        "--window", 
        type=int, 
        default=75, 
        help="TN down-sampling window_nt: nucleotide window around true positives to preserve as 'neighborhood' negatives."
    )
    p.add_argument(
        "--easy-ratio", 
        type=float, 
        default=0.5, 
        help="TN down-sampling easy_neg_ratio: fraction of 'easy' negatives to randomly keep after preserving hard/neighborhood negatives."
    )
    p.add_argument(
        "--kmer-sizes",
        nargs="*",
        action=FlexibleListAction,
        item_type=int,
        default=[6],
        help="One or more k-mer sizes to extract. Provide multiple integers separated by space OR comma (e.g. --kmer-sizes 4 6 OR --kmer-sizes 4,6). Use 0 or omit to skip k-mer extraction entirely."
    )
    p.add_argument(
        "--patch-dataset",
        action="store_true",
        help="Run built-in patch scripts (structural features & gene_type) on the assembled dataset to fill missing values."
    )
    p.add_argument(
        "--no-manifest",
        action="store_true",
        help="Skip generation of gene manifest file (gene_manifest.csv)."
    )
    p.add_argument(
        "--batch-rows",
        type=int,
        default=500_000,
        help="Maximum number of rows to buffer in memory before flushing a Parquet row group when streaming TSVs.  Lower values reduce peak RAM at the cost of slower I/O."
    )
    p.add_argument(
        "--position-id-mode",
        type=str,
        default="genomic",
        choices=["genomic", "transcript", "splice_aware", "hybrid"],
        help="Position identification strategy: 'genomic' (current/default), 'transcript' (transcript-aware for meta-learning), 'splice_aware' (same as transcript), 'hybrid' (transition mode)."
    )
    p.add_argument(
        "--verbose", 
        "-v", 
        action="count", 
        default=1, 
        help="Increase verbosity. Use -v for standard output, -vv for detailed output, -vvv for debug."
    )
    args = p.parse_args(sys.argv[1:])

    # --------------------------------------------------------------
    # Load custom gene IDs file (if provided) ----------------------
    # --------------------------------------------------------------
    ids = None
    if args.gene_ids_file:
        import polars as pl, os
        try:
            sep = "\t" if args.gene_ids_file.endswith(".tsv") else "," if args.gene_ids_file.endswith(".csv") else None
            if sep:
                df_ids = pl.read_csv(args.gene_ids_file, separator=sep)
                if args.gene_col not in df_ids.columns:
                    raise KeyError
                ids = df_ids[args.gene_col].drop_nulls().unique().to_list()
            else:
                with open(args.gene_ids_file) as fh:
                    ids = [ln.strip() for ln in fh if ln.strip()]
        except Exception as _e:
            print(f"[incremental-builder] Failed to load gene IDs from {args.gene_ids_file}: {_e}", file=sys.stderr)
            sys.exit(1)
        
        # Handle gene selection logic when --gene-ids-file is provided
        # This implements the intended behavior:
        # - "--gene-ids-file file.txt" (no --n-genes, no --subset-policy) = custom mode (only genes from file)
        # - "--gene-ids-file file.txt --subset-policy custom" = custom mode (only genes from file)
        # - "--subset-policy all" = all available genes (ignores --gene-ids-file with warning)
        # - "--n-genes 5000 --gene-ids-file file.txt" = 5000 genes + additional genes from file
        # - "--gene-ids-file file.txt --subset-policy random" = ERROR (conflicting parameters)
        n_genes_explicitly_set = "--n-genes" in sys.argv
        subset_policy_explicitly_set = "--subset-policy" in sys.argv
        
        # Handle 'all' policy first (special case)
        if args.subset_policy == "all":
            if args.gene_ids_file:
                print(f"[incremental-builder] WARNING: --gene-ids-file ignored when using --subset-policy all", file=sys.stderr)
                print(f"[incremental-builder] Using all available genes (gene-ids-file will be ignored)")
            else:
                print(f"[incremental-builder] Using all available genes")
            # For 'all' policy, we don't need to validate n_genes or gene_ids_file
            
        elif not n_genes_explicitly_set:
            # No --n-genes provided, gene-ids-file should determine behavior
            if not subset_policy_explicitly_set:
                # Auto-switch to custom mode
                args.subset_policy = "custom"
                print(f"[incremental-builder] Auto-setting --subset-policy custom because only --gene-ids-file was provided")
                print(f"[incremental-builder] Using {len(ids)} custom genes only")
            elif args.subset_policy == "custom":
                # Explicitly set to custom - this is fine
                print(f"[incremental-builder] Using {len(ids)} custom genes with policy '{args.subset_policy}'")
            else:
                # Conflicting parameters: gene-ids-file without n-genes but with non-custom policy
                print(f"[incremental-builder] ERROR: --subset-policy '{args.subset_policy}' requires --n-genes to be specified", file=sys.stderr)
                print(f"[incremental-builder] When using --gene-ids-file without --n-genes, only --subset-policy custom or all is allowed", file=sys.stderr)
                print(f"[incremental-builder] Either:", file=sys.stderr)
                print(f"[incremental-builder]   1. Remove --subset-policy (will auto-set to custom)", file=sys.stderr)
                print(f"[incremental-builder]   2. Use --subset-policy custom explicitly", file=sys.stderr)
                print(f"[incremental-builder]   3. Use --subset-policy all to select all genes", file=sys.stderr)
                print(f"[incremental-builder]   4. Add --n-genes to use '{args.subset_policy}' policy", file=sys.stderr)
                sys.exit(1)
        else:
            # --n-genes provided, normal behavior
            if args.subset_policy == "custom":
                print(f"[incremental-builder] Using {len(ids)} custom genes with policy '{args.subset_policy}'")
            else:
                print(f"[incremental-builder] Using policy '{args.subset_policy}' to select {args.n_genes} genes total (including {len(ids)} from user file)")

    try:
        wk = json.loads(args.workflow_kwargs)
    except json.JSONDecodeError as e:
        print(f"Invalid JSON for --workflow-kwargs: {e}", file=sys.stderr)
        sys.exit(1)

    ds_kwargs = dict(
        hard_prob_thresh=args.hard_prob,
        window_nt=args.window,
        easy_neg_ratio=args.easy_ratio,
    )

    incremental_build_training_dataset(
        n_genes=args.n_genes,
        subset_policy=args.subset_policy,
        batch_size=args.batch_size,
        kmer_sizes=(None if args.kmer_sizes == [0] else args.kmer_sizes),
        output_dir=args.output_dir,
        overwrite=args.overwrite,
        downsample_kwargs=ds_kwargs,
        gene_types=args.gene_types,
        additional_gene_ids=ids,
        run_workflow=args.run_workflow,
        workflow_kwargs=wk,
        batch_rows=args.batch_rows,
        verbose=args.verbose,
        patch_dataset=args.patch_dataset,
        generate_manifest=not args.no_manifest,
        position_id_mode=args.position_id_mode,  # Pass through transcript-aware mode
    )
