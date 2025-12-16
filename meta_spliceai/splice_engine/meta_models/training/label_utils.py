"""Central utilities for splice-site class labels.

Provides a *single* source of truth for the canonical label mapping and helper
functions to encode / remap labels.  Import this module instead of defining
`_LABEL_MAP_STR` repeatedly.

Canonical mapping
-----------------
0 : neither (background / non-splice)
1 : donor   (5' splice site)
2 : acceptor (3' splice site)

The old, legacy numeric encoding used by some historic datasets instead had
0 = donor, 1 = acceptor, 2 = neither.  The helpers below auto-detect that
pattern and silently remap to the canonical order while issuing a
RuntimeWarning so the caller is aware.
"""
from __future__ import annotations

from typing import Any, Sequence, Dict
import numpy as np
import warnings

# ---------------------------------------------------------------------------
# 1. Canonical mapping tables
# ---------------------------------------------------------------------------

LABEL_MAP_STR: Dict[str, int] = {"neither": 0, "donor": 1, "acceptor": 2}
INT_TO_LABEL: Dict[int, str] = {v: k for k, v in LABEL_MAP_STR.items()}

__all__ = [
    "LABEL_MAP_STR",
    "INT_TO_LABEL",
    "encode_labels",
    "swap_0_2",
    "load_dataset_sample",
    "load_genes_subset",
]

# ---------------------------------------------------------------------------
# 2. Helper functions
# ---------------------------------------------------------------------------

def swap_0_2(arr: np.ndarray) -> np.ndarray:
    """Return a copy of *arr* with labels 0 and 2 swapped.

    Useful for converting between canonical and legacy numeric encodings.
    """
    swapped = arr.copy()
    mask0 = swapped == 0
    mask2 = swapped == 2
    swapped[mask0] = 2
    swapped[mask2] = 0
    return swapped


def encode_labels(arr: Sequence[Any]) -> np.ndarray:
    """Convert raw *splice_type* labels to canonical integers 0/1/2.

    Accepts a mix of strings ("donor"/"acceptor"/"neither"), integers, or
    numeric strings.  Automatically detects and remaps the legacy order
    (0=donor,2=neither) to the canonical order.
    """
    arr = np.asarray(arr)

    # ------------------------------------------------------------------
    # Step 1 – convert non-integer representations to ints via the map.
    # ------------------------------------------------------------------
    if arr.dtype.kind not in ("i", "u"):
        conv: list[int] = []
        unknown: set[str] = set()
        for x in arr:
            # direct ints
            if isinstance(x, (int, np.integer)):
                conv.append(int(x))
                continue
            # numeric strings
            try:
                xi = int(x)
                conv.append(xi)
                continue
            except (ValueError, TypeError):
                pass
            # canonical text labels (case-insensitive)
            val = LABEL_MAP_STR.get(str(x).lower())
            if val is None:
                unknown.add(str(x))
            else:
                conv.append(val)
        if unknown:
            raise ValueError(
                f"Unrecognised splice_type labels: {', '.join(sorted(unknown))}."
            )
        arr = np.asarray(conv, dtype=int)

    # ------------------------------------------------------------------
    # Step 2 – validate & remap numeric labels if necessary.
    # ------------------------------------------------------------------
    uniq = np.unique(arr)
    if not set(uniq).issubset({0, 1, 2}):
        raise ValueError(
            f"Unexpected numeric splice_type labels {uniq}. Expected subset of {{0,1,2}}."
        )

    # Heuristic: in canonical encoding *neither* (0) is usually the largest
    # class.  If label 2 dominates we assume legacy order and swap 0↔2.
    counts = np.bincount(arr.astype(int), minlength=3)
    if counts[0] < counts[2]:
        arr = swap_0_2(arr)
        warnings.warn(
            "Detected legacy numeric encoding (0=donor,2=neither); auto-remapped to canonical order.",
            RuntimeWarning,
        )
    return arr.astype(int)


def load_dataset_sample(dataset_path: str, sample_size: int = None, sample_genes: int = None, 
                       random_seed: int = 42) -> Any:
    """
    Load and sample a dataset while preserving gene-level structure.
    
    This function samples complete genes rather than individual rows, ensuring that
    all splice sites (donors, acceptors) and non-splice sites from a gene are preserved.
    
    Args:
        dataset_path: Path to the dataset directory or file
        sample_size: Number of total rows to sample (if gene sampling not specified)
        sample_genes: Number of genes to sample (takes precedence over sample_size)
        random_seed: Random seed for reproducibility
    
    Returns:
        Sampled dataframe (same type as input - Pandas or Polars)
    """
    import os
    import random
    import numpy as np
    from pathlib import Path
    
    try:
        # First try importing our dataframe utility functions
        from meta_spliceai.splice_engine.meta_models.utils.dataframe_utils import (
            is_pandas_dataframe, get_row_count, filter_dataframe, has_column
        )
    except ImportError:
        # Fallback to simple type checking if utils not available
        import pandas as pd
        def is_pandas_dataframe(df): return isinstance(df, pd.DataFrame)
        def get_row_count(df): return len(df) if is_pandas_dataframe(df) else df.height
        def filter_dataframe(df, column, values):
            if is_pandas_dataframe(df):
                return df[df[column].isin(values)]
            else:
                import polars as pl
                return df.filter(pl.col(column).is_in(values))
        def has_column(df, column):
            if is_pandas_dataframe(df):
                return column in df.columns
            else:
                return column in df.columns
    
    # Set random seed for reproducibility
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    # Load the dataset
    if os.path.isdir(dataset_path):
        # It's a directory, check for parquet or CSV files
        path = Path(dataset_path)
        parquet_files = list(path.glob("*.parquet")) + list(path.glob("**/*.parquet"))
        csv_files = list(path.glob("*.csv")) + list(path.glob("**/*.csv")) + \
                    list(path.glob("*.tsv")) + list(path.glob("**/*.tsv"))
        
        if parquet_files:
            # Try with polars first for better performance
            try:
                import polars as pl
                df = pl.read_parquet(str(parquet_files[0]))
                is_polars = True
            except ImportError:
                import pandas as pd
                df = pd.read_parquet(str(parquet_files[0]))
                is_polars = False
        elif csv_files:
            try:
                import polars as pl
                df = pl.read_csv(str(csv_files[0]), infer_schema_length=10000)
                is_polars = True
            except ImportError:
                import pandas as pd
                df = pd.read_csv(str(csv_files[0]))
                is_polars = False
        else:
            # Try master.csv/master.tsv
            master_path = path / "master.csv"
            if not master_path.exists():
                master_path = path / "master.tsv"
            
            if master_path.exists():
                try:
                    import polars as pl
                    df = pl.read_csv(str(master_path), infer_schema_length=10000, 
                                    separator="," if master_path.suffix == ".csv" else "\t")
                    is_polars = True
                except ImportError:
                    import pandas as pd
                    df = pd.read_csv(str(master_path), sep="," if master_path.suffix == ".csv" else "\t")
                    is_polars = False
            else:
                raise FileNotFoundError(f"Could not find any data files in {dataset_path}")
    else:
        # It's a file, try to load it
        if dataset_path.endswith(".parquet"):
            try:
                import polars as pl
                df = pl.read_parquet(dataset_path)
                is_polars = True
            except ImportError:
                import pandas as pd
                df = pd.read_parquet(dataset_path)
                is_polars = False
        elif dataset_path.endswith(".csv") or dataset_path.endswith(".tsv"):
            sep = "," if dataset_path.endswith(".csv") else "\t"
            try:
                import polars as pl
                df = pl.read_csv(dataset_path, separator=sep, infer_schema_length=10000)
                is_polars = True
            except ImportError:
                import pandas as pd
                df = pd.read_csv(dataset_path, sep=sep)
                is_polars = False
        else:
            raise ValueError(f"Unsupported file format: {dataset_path}")
    
    # Check if the required gene column exists
    gene_col = None
    for col_name in ["gene_id", "gene_name", "gene", "transcript_id"]:
        if has_column(df, col_name):
            gene_col = col_name
            break
    
    if gene_col is None:
        print("[WARNING] No gene or transcript identifier column found. Falling back to random sampling.")
        if sample_size is None and sample_genes is None:
            return df  # Return the entire dataset if no sample size specified
        
        # Random sampling if no gene column
        total_rows = get_row_count(df)
        sample_rows = min(sample_size or total_rows, total_rows)
        
        if is_pandas_dataframe(df):
            return df.sample(n=sample_rows, random_state=random_seed)
        else:
            import polars as pl
            return df.sample(n=sample_rows, seed=random_seed)
    
    # Get unique genes
    if is_pandas_dataframe(df):
        unique_genes = df[gene_col].unique().tolist()
    else:
        import polars as pl
        unique_genes = df.select(pl.col(gene_col).unique()).to_series().to_list()
    
    total_genes = len(unique_genes)
    print(f"[INFO] Found {total_genes} unique genes in the dataset")
    
    # Determine how many genes to sample
    genes_to_sample = sample_genes
    if genes_to_sample is None and sample_size is not None:
        # Estimate number of genes needed to reach sample_size
        avg_rows_per_gene = get_row_count(df) / total_genes
        genes_to_sample = max(1, min(total_genes, int(sample_size / avg_rows_per_gene)))
    
    # Default to all genes if neither sample_genes nor sample_size specified
    if genes_to_sample is None:
        return df
    
    # Sample genes
    genes_to_sample = min(genes_to_sample, total_genes)
    sampled_genes = random.sample(unique_genes, genes_to_sample)
    
    print(f"[INFO] Sampling {genes_to_sample} genes out of {total_genes} total genes")
    
    # Filter to only include the sampled genes
    sampled_df = filter_dataframe(df, gene_col, sampled_genes)
    
    # If we have a specific sample_size target and we're over, subsample rows
    if sample_size is not None:
        actual_size = get_row_count(sampled_df)
        if actual_size > sample_size:
            if is_pandas_dataframe(sampled_df):
                sampled_df = sampled_df.sample(n=sample_size, random_state=random_seed)
            else:
                import polars as pl
                sampled_df = sampled_df.sample(n=sample_size, seed=random_seed)
            print(f"[INFO] Further sampled to {sample_size} rows from {actual_size}")
        else:
            print(f"[INFO] Selected {actual_size} rows from {genes_to_sample} genes")
    else:
        actual_size = get_row_count(sampled_df)
        print(f"[INFO] Selected {actual_size} rows from {genes_to_sample} genes")
    
    return sampled_df


def load_genes_subset(dataset_path: str, gene_list: List[str], return_polars: bool = False):
    """
    Load data for a specific subset of genes.
    
    This function efficiently loads only the data for specified genes,
    avoiding memory issues with large datasets.
    
    Parameters
    ----------
    dataset_path : str
        Path to the dataset directory containing parquet files
    gene_list : List[str]
        List of specific gene IDs to load
    return_polars : bool, default False
        If True, return polars DataFrame. If False, return pandas DataFrame.
        
    Returns
    -------
    DataFrame
        DataFrame containing data only for the specified genes
    """
    import polars as pl
    
    # Load only the specified genes
    lf = pl.scan_parquet(f"{dataset_path}/*.parquet", extra_columns='ignore')
    df = lf.filter(pl.col("gene_id").is_in(gene_list)).collect()
    
    if return_polars:
        return df
    else:
        # Convert to pandas for compatibility
        return df.to_pandas()
