"""Utility helpers for k-mer feature naming conventions.

Provides `is_kmer_feature()` to detect columns that encode fixed-length DNA k-mers
using the `<k>mer_<SEQ>` convention, e.g. `6mer_AAAATA`.
"""
from __future__ import annotations

import re
from typing import Final

__all__: list[str] = [
    "is_kmer_feature",
]

_KMER_RE: Final[re.Pattern[str]] = re.compile(r"^(\d+)mer_([ACGTN]+)$", flags=re.IGNORECASE)

def is_kmer_feature(col: str) -> bool:
    """Return ``True`` if *col* follows the `<k>mer_<SEQ>` naming scheme.

    The prefix integer *k* must equal the length of the nucleotide sequence, and
    the sequence may contain only the canonical bases *A*, *C*, *G*, *T*, and *N*
    (ambiguous base, case-insensitive).

    Examples
    --------
    >>> is_kmer_feature("6mer_AAAATA")
    True
    >>> is_kmer_feature("4mer_acgt")
    True
    >>> is_kmer_feature("6mer_ACGT")  # length mismatch –> False
    False
    >>> is_kmer_feature("gene_id")
    False
    """
    m = _KMER_RE.match(col)
    if not m:
        return False
    return int(m.group(1)) == len(m.group(2))
