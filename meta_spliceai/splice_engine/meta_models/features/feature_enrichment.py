"""feature_enrichment.py
A lightweight, plug-and-play **feature-augmentation registry** for meta-model
training tables.

Motivation
----------
`make_kmer_featurized_dataset` generates k-mer (+ optional gene) features.  As
research progresses we often want to bolt on new, *orthogonal* feature groups
(e.g. transcript-length metrics, exon–intron geometry, performance profiles,
external conservation scores).  Hard-coding each call leads to duplication and
rigid workflows.

This helper provides:

1.  A **decorator-based registry** so any module can expose an *enricher*:

        @register_enricher("length_features")
        def incorporate_length_features(df: pl.DataFrame, fa, **kw): ...

2.  A single `apply_feature_enrichers(df, enrichers=[...], **ctx)` dispatcher
    that executes enrichers in order, passing along shared context objects
    (feature-analyzer, splice-analyzer, etc.).

3.  Utility functions to query available enrichers.

The approach keeps code loosely coupled – new enrichers live next to their
feature logic, not inside the central k-mer script.
"""
from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Optional, Sequence

import pandas as pd
import polars as pl

# ---------------------------------------------------------------------------
# Registry machinery
# ---------------------------------------------------------------------------

_Enricher = Callable[..., pl.DataFrame]
_ENRICHERS: Dict[str, _Enricher] = {}


def register_enricher(name: Optional[str] = None):  # decorator factory
    """Decorator to register a feature-enrichment function.

    Usage
    -----
        @register_enricher()               # name defaults to function.__name__
        def incorporate_length_features(df, fa=None, **kw): ...
    """

    def _decorator(func: _Enricher) -> _Enricher:
        key = name or func.__name__
        if key in _ENRICHERS:
            raise ValueError(f"Enricher '{key}' already registered.")
        _ENRICHERS[key] = func
        return func

    return _decorator


def list_enrichers() -> List[str]:
    """Return names of all registered enrichers in registration order."""
    
    # Preserve registration (insertion) order so that dependencies like gene-level
    # features are computed *before* they are used by downstream enrichers (e.g.
    # distance_features requires columns added by gene_level).
    # Users can still sort manually if a different order is needed.
    
    return list(_ENRICHERS)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def apply_feature_enrichers(
    df: pl.DataFrame,
    enrichers: Optional[Sequence[str]] = None,
    *,
    verbose: int = 1,
    # contextual objects that many enrichers need
    fa=None,
    sa=None,
    **kwargs,
) -> pl.DataFrame:
    """Apply *enrichers* sequentially to *df* and return the augmented table.

    Parameters
    ----------
    df
        Base feature table (typically output of make_kmer_featurized_dataset).
    enrichers
        List of enricher names.  If *None*, all registered enrichers are run in
        registration order.
    verbose
        0 = silent, 1 = log enricher names.
    fa, sa
        FeatureAnalyzer / SpliceAnalyzer helpers – passed to each enricher so
        they don't have to be re-constructed.
    **kwargs
        Additional context forwarded to each enricher.
    """

    if enrichers is None:
        enrichers = list_enrichers()

    out = df
    for key in enrichers:
        if key not in _ENRICHERS:
            raise KeyError(f"Enricher '{key}' not registered. Call list_enrichers()"
                           " to see available options.")
        func = _ENRICHERS[key]
        if verbose:
            print(f"[enrichment] Running '{key}' …")
        out = func(out, fa=fa, sa=sa, verbose=verbose, **kwargs)
        # Convert to polars DataFrame if needed
        if isinstance(out, pd.DataFrame):
            out = pl.from_pandas(out)
        elif not isinstance(out, pl.DataFrame):
            raise TypeError(
                f"Enricher '{key}' must return a polars or pandas DataFrame, got {type(out)}"
            )
    return out


# ---------------------------------------------------------------------------
# Optional: auto-register built-in enrichers if available
# ---------------------------------------------------------------------------

try:
    from meta_spliceai.splice_engine.meta_models.workflows.data_generation import (
        incorporate_gene_level_features,
        incorporate_length_features,
        incorporate_performance_features,
        incorporate_overlapping_gene_features,
        incorporate_distance_features,
    )

    @register_enricher("gene_level")
    def _gene_level(df, fa=None, **kw):
        return incorporate_gene_level_features(df, fa=fa)

    @register_enricher("length_features")
    def _length(df, fa=None, **kw):
        return incorporate_length_features(df, fa=fa)

    @register_enricher("performance_features")
    def _perf(df, fa=None, **kw):
        return incorporate_performance_features(df, fa=fa)

    @register_enricher("overlap_features")
    def _overlap(df, sa=None, **kw):
        return incorporate_overlapping_gene_features(df, sa=sa)

    @register_enricher("distance_features")
    def _dist(df, fa=None, **kw):
        return incorporate_distance_features(df, fa)

except ModuleNotFoundError:
    # Older installations might not have those helpers – ignore.
    pass

__all__ = [
    "register_enricher",
    "list_enrichers",
    "apply_feature_enrichers",
]
