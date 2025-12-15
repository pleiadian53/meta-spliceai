"""
Common data types for case studies.

This module contains shared data structures to avoid circular imports.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class AlternativeSpliceSite:
    """Represents an alternative splice site derived from variant analysis."""
    chrom: str
    position: int
    strand: str
    site_type: str  # 'donor', 'acceptor'
    splice_category: str  # 'canonical', 'cryptic_activated', 'disease_associated'
    delta_score: float
    ref_score: float
    alt_score: float
    variant_id: str
    gene_symbol: str
    clinical_significance: Optional[str] = None
    validation_evidence: Optional[str] = None


@dataclass
class DeltaScoreResult:
    """Results from OpenSpliceAI delta score computation."""
    variant_id: str
    gene_symbol: str
    chrom: str
    position: int
    ref_allele: str
    alt_allele: str
    
    # Delta scores (DS)
    ds_ag: Optional[float] = None  # Acceptor Gain
    ds_al: Optional[float] = None  # Acceptor Loss  
    ds_dg: Optional[float] = None  # Donor Gain
    ds_dl: Optional[float] = None  # Donor Loss
    
    # Delta positions (DP) - relative to variant
    dp_ag: Optional[int] = None
    dp_al: Optional[int] = None
    dp_dg: Optional[int] = None
    dp_dl: Optional[int] = None
