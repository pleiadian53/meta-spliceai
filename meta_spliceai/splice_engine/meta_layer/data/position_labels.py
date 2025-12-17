"""
Position Label Derivation for Multi-Step Framework Step 3.

This module provides utilities for deriving position labels (where splice effects occur)
from SpliceVarDB variants, using two complementary approaches:

1. **HGVS Parsing** (weak labels): Extract position hints from HGVS notation
2. **Base Model Delta Analysis** (recommended): Derive positions from base model delta peaks

These labels can be used to train the Position Localization model (Step 3) in the
Multi-Step Framework.

Usage
-----
>>> from meta_spliceai.splice_engine.meta_layer.data.position_labels import (
...     derive_position_labels_from_delta,
...     derive_position_labels_from_hgvs,
...     create_position_attention_target
... )
>>>
>>> # From base model deltas (recommended)
>>> affected_positions = derive_position_labels_from_delta(
...     ref_seq, alt_seq, base_model, threshold=0.1
... )
>>>
>>> # From HGVS notation (weak labels)
>>> hint = derive_position_labels_from_hgvs(variant, variant_position_in_window=250)

See Also
--------
- docs/methods/MULTI_STEP_FRAMEWORK.md: Multi-Step Framework documentation
- data/splicevardb_loader.py: SpliceVarDB loading and HGVS parsing
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from .splicevardb_loader import HGVSPositionHint, VariantRecord, parse_hgvs_position_hint

logger = logging.getLogger(__name__)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class AffectedPosition:
    """
    A single position affected by a variant.
    
    Attributes
    ----------
    position : int
        Position in the sequence (0-indexed)
    effect_type : str
        Type of effect: 'donor_gain', 'donor_loss', 'acceptor_gain', 'acceptor_loss'
    delta_value : float
        Magnitude of the delta (signed: positive=gain, negative=loss)
    channel : str
        Which channel: 'donor' or 'acceptor'
    confidence : str
        Confidence level: 'high', 'medium', 'low'
    source : str
        Where this label came from: 'base_model_delta', 'hgvs_parsing'
    """
    position: int
    effect_type: str
    delta_value: float
    channel: str
    confidence: str = 'medium'
    source: str = 'base_model_delta'


@dataclass
class PositionLabelResult:
    """
    Result of position label derivation for a variant.
    
    Attributes
    ----------
    variant_id : str
        Identifier for the variant
    affected_positions : List[AffectedPosition]
        List of affected positions with their effects
    attention_target : Optional[np.ndarray]
        Normalized attention target [L] for training
    peak_position : Optional[int]
        Position of maximum effect
    peak_effect_type : Optional[str]
        Type of effect at peak position
    sequence_length : int
        Length of the analyzed sequence
    """
    variant_id: str
    affected_positions: List[AffectedPosition]
    attention_target: Optional[np.ndarray] = None
    peak_position: Optional[int] = None
    peak_effect_type: Optional[str] = None
    sequence_length: int = 0


# =============================================================================
# BASE MODEL DELTA ANALYSIS (RECOMMENDED)
# =============================================================================

def derive_position_labels_from_delta(
    ref_seq: Union[str, np.ndarray],
    alt_seq: Union[str, np.ndarray],
    base_model,
    device: str = 'cpu',
    threshold: float = 0.1,
    window_around_variant: Optional[Tuple[int, int]] = None,
    variant_position: Optional[int] = None
) -> List[AffectedPosition]:
    """
    Derive affected positions from base model delta analysis.
    
    This is the RECOMMENDED approach for deriving position labels. It uses the
    base model (SpliceAI/OpenSpliceAI) to identify where splice probabilities
    change significantly due to the variant.
    
    Parameters
    ----------
    ref_seq : str or np.ndarray
        Reference sequence (string or one-hot encoded)
    alt_seq : str or np.ndarray
        Alternate sequence with variant
    base_model : nn.Module or callable
        Base splice prediction model (or list for ensemble)
    device : str
        Device for inference
    threshold : float
        Minimum absolute delta to consider significant (default 0.1)
    window_around_variant : tuple of (start, end), optional
        Only consider positions within this window. If None, uses ±50bp from center.
    variant_position : int, optional
        Position of variant in sequence. If None, assumes center.
    
    Returns
    -------
    List[AffectedPosition]
        List of positions with significant delta effects
    
    Examples
    --------
    >>> affected = derive_position_labels_from_delta(
    ...     ref_seq, alt_seq, base_model, threshold=0.1
    ... )
    >>> for pos in affected:
    ...     print(f"Position {pos.position}: {pos.effect_type} (Δ={pos.delta_value:.3f})")
    """
    # Convert to one-hot if needed
    if isinstance(ref_seq, str):
        ref_oh = _one_hot_encode(ref_seq)
        alt_oh = _one_hot_encode(alt_seq)
    else:
        ref_oh = ref_seq
        alt_oh = alt_seq
    
    # Get base model predictions
    ref_scores = _get_base_model_scores(ref_oh, base_model, device)
    alt_scores = _get_base_model_scores(alt_oh, base_model, device)
    
    # Compute delta
    delta = alt_scores - ref_scores  # [L, 3] for [neither, acceptor, donor]
    
    # Determine search window
    seq_len = len(delta)
    if window_around_variant is not None:
        start, end = window_around_variant
    elif variant_position is not None:
        start = max(0, variant_position - 50)
        end = min(seq_len, variant_position + 51)
    else:
        # Assume variant is at center
        center = seq_len // 2
        start = max(0, center - 50)
        end = min(seq_len, center + 51)
    
    # Find affected positions
    affected = []
    
    for pos in range(start, end):
        # OpenSpliceAI output: [neither, acceptor, donor]
        donor_delta = delta[pos, 2]
        acceptor_delta = delta[pos, 1]
        
        # Check donor effect
        if abs(donor_delta) >= threshold:
            effect_type = 'donor_gain' if donor_delta > 0 else 'donor_loss'
            confidence = 'high' if abs(donor_delta) >= 0.3 else 'medium'
            
            affected.append(AffectedPosition(
                position=pos,
                effect_type=effect_type,
                delta_value=float(donor_delta),
                channel='donor',
                confidence=confidence,
                source='base_model_delta'
            ))
        
        # Check acceptor effect
        if abs(acceptor_delta) >= threshold:
            effect_type = 'acceptor_gain' if acceptor_delta > 0 else 'acceptor_loss'
            confidence = 'high' if abs(acceptor_delta) >= 0.3 else 'medium'
            
            affected.append(AffectedPosition(
                position=pos,
                effect_type=effect_type,
                delta_value=float(acceptor_delta),
                channel='acceptor',
                confidence=confidence,
                source='base_model_delta'
            ))
    
    # Sort by absolute delta (strongest effects first)
    affected.sort(key=lambda x: abs(x.delta_value), reverse=True)
    
    return affected


def create_position_attention_target(
    affected_positions: List[AffectedPosition],
    sequence_length: int,
    sigma: float = 3.0,
    normalize: bool = True
) -> np.ndarray:
    """
    Create an attention-style target distribution from affected positions.
    
    This creates a soft target for training position localization models,
    where each affected position contributes a Gaussian peak weighted by
    its delta magnitude.
    
    Parameters
    ----------
    affected_positions : List[AffectedPosition]
        List of affected positions from delta analysis
    sequence_length : int
        Length of the output sequence
    sigma : float
        Standard deviation of Gaussian peaks (default 3.0 = ±3bp spread)
    normalize : bool
        Whether to normalize to sum to 1 (default True)
    
    Returns
    -------
    np.ndarray
        Attention target of shape [sequence_length]
    
    Examples
    --------
    >>> attention_target = create_position_attention_target(
    ...     affected_positions, sequence_length=501, sigma=3.0
    ... )
    >>> # Use as soft target for attention/localization training
    >>> loss = kl_divergence(pred_attention, attention_target)
    """
    target = np.zeros(sequence_length, dtype=np.float32)
    
    if not affected_positions:
        # No affected positions: uniform distribution
        if normalize:
            target[:] = 1.0 / sequence_length
        return target
    
    # Create Gaussian peaks at each affected position
    x = np.arange(sequence_length)
    
    for pos_info in affected_positions:
        pos = pos_info.position
        weight = abs(pos_info.delta_value)
        
        if 0 <= pos < sequence_length:
            # Add Gaussian peak
            gaussian = np.exp(-0.5 * ((x - pos) / sigma) ** 2)
            target += weight * gaussian
    
    # Normalize
    if normalize and target.sum() > 0:
        target = target / target.sum()
    
    return target


def create_binary_position_mask(
    affected_positions: List[AffectedPosition],
    sequence_length: int,
    expand_by: int = 2
) -> np.ndarray:
    """
    Create a binary mask of affected positions.
    
    Parameters
    ----------
    affected_positions : List[AffectedPosition]
        List of affected positions
    sequence_length : int
        Length of the output sequence
    expand_by : int
        Expand each affected position by this many bp on each side (default 2)
    
    Returns
    -------
    np.ndarray
        Binary mask of shape [sequence_length] with 1s at affected positions
    """
    mask = np.zeros(sequence_length, dtype=np.float32)
    
    for pos_info in affected_positions:
        pos = pos_info.position
        start = max(0, pos - expand_by)
        end = min(sequence_length, pos + expand_by + 1)
        mask[start:end] = 1.0
    
    return mask


# =============================================================================
# HGVS PARSING APPROACH (WEAK LABELS)
# =============================================================================

def derive_position_labels_from_hgvs(
    variant: VariantRecord,
    variant_position_in_window: int = 250,
    window_size: int = 501
) -> Optional[AffectedPosition]:
    """
    Derive position label from HGVS notation.
    
    This is a WEAK labeling approach - it provides approximate position information
    based on HGVS notation parsing. Use `derive_position_labels_from_delta` for
    more accurate labels.
    
    Parameters
    ----------
    variant : VariantRecord
        Variant record with HGVS notation
    variant_position_in_window : int
        Where the variant is located in the context window (default 250 = center)
    window_size : int
        Size of the context window (default 501)
    
    Returns
    -------
    AffectedPosition or None
        Inferred affected position, or None if cannot be determined
    
    Notes
    -----
    HGVS notation like "c.670-1G>T" tells us:
    - The variant is 1bp before an exon boundary
    - This is likely affecting an acceptor site
    - The affected splice site is at variant_position + 1
    
    However, this is approximate because:
    - We don't know the exact exon coordinates
    - The variant might create a cryptic site elsewhere
    - Complex effects are not captured
    """
    hint = variant.get_position_hint()
    
    if hint.site_type == 'unknown':
        return None
    
    if hint.site_type == 'deep_intronic':
        # Deep intronic variants might create cryptic sites
        # Position is uncertain - return variant position itself
        return AffectedPosition(
            position=variant_position_in_window,
            effect_type='cryptic_unknown',
            delta_value=0.0,  # Unknown magnitude
            channel='unknown',
            confidence='low',
            source='hgvs_parsing'
        )
    
    if hint.site_type == 'exonic':
        # Exonic variants may affect ESE/ESS
        # Effect location is less certain
        return AffectedPosition(
            position=variant_position_in_window,
            effect_type='exonic_unknown',
            delta_value=0.0,
            channel='unknown',
            confidence='low',
            source='hgvs_parsing'
        )
    
    # For donor/acceptor sites, we can estimate the affected position
    if hint.site_type == 'donor':
        # Variant is after exon (intron side)
        # The donor splice site is at the exon boundary
        # If variant is at +1, the donor site is at variant_position - 1
        affected_pos = variant_position_in_window - hint.distance_from_boundary
        
        # Determine if this is gain or loss
        # Canonical mutations (+1, +2) usually cause loss
        # Other mutations might create cryptic donors
        if hint.is_canonical_region:
            effect_type = 'donor_loss'  # Disrupting existing donor
        else:
            effect_type = 'donor_gain'  # Possibly creating new donor
        
        return AffectedPosition(
            position=affected_pos,
            effect_type=effect_type,
            delta_value=0.0,
            channel='donor',
            confidence=hint.confidence,
            source='hgvs_parsing'
        )
    
    if hint.site_type == 'acceptor':
        # Variant is before exon (intron side)
        # The acceptor splice site is at the exon boundary
        # If variant is at -1, the acceptor site is at variant_position + 1
        affected_pos = variant_position_in_window + hint.distance_from_boundary
        
        # Canonical mutations (-1, -2) usually cause loss
        if hint.is_canonical_region:
            effect_type = 'acceptor_loss'
        else:
            effect_type = 'acceptor_gain'
        
        return AffectedPosition(
            position=affected_pos,
            effect_type=effect_type,
            delta_value=0.0,
            channel='acceptor',
            confidence=hint.confidence,
            source='hgvs_parsing'
        )
    
    return None


# =============================================================================
# COMBINED APPROACH
# =============================================================================

def derive_position_labels(
    variant: VariantRecord,
    ref_seq: Optional[str] = None,
    alt_seq: Optional[str] = None,
    base_model=None,
    device: str = 'cpu',
    threshold: float = 0.1,
    use_hgvs_fallback: bool = True
) -> PositionLabelResult:
    """
    Derive position labels using the best available method.
    
    This combines both approaches:
    1. If base model and sequences are provided: use delta analysis (recommended)
    2. Otherwise, fall back to HGVS parsing (weak labels)
    
    Parameters
    ----------
    variant : VariantRecord
        The variant to analyze
    ref_seq : str, optional
        Reference sequence
    alt_seq : str, optional
        Alternate sequence
    base_model : nn.Module, optional
        Base model for delta analysis
    device : str
        Device for inference
    threshold : float
        Delta threshold for significance
    use_hgvs_fallback : bool
        Whether to use HGVS parsing if delta analysis is not available
    
    Returns
    -------
    PositionLabelResult
        Complete position label information
    """
    affected_positions = []
    
    # Try delta analysis first (recommended)
    if ref_seq is not None and alt_seq is not None and base_model is not None:
        affected_positions = derive_position_labels_from_delta(
            ref_seq, alt_seq, base_model, device, threshold
        )
    elif use_hgvs_fallback:
        # Fall back to HGVS parsing
        hgvs_result = derive_position_labels_from_hgvs(variant)
        if hgvs_result is not None:
            affected_positions = [hgvs_result]
    
    # Find peak position
    peak_position = None
    peak_effect_type = None
    if affected_positions:
        peak = max(affected_positions, key=lambda x: abs(x.delta_value))
        peak_position = peak.position
        peak_effect_type = peak.effect_type
    
    # Create attention target if we have sequence info
    seq_len = len(ref_seq) if ref_seq else 501
    attention_target = None
    if affected_positions:
        attention_target = create_position_attention_target(
            affected_positions, seq_len
        )
    
    return PositionLabelResult(
        variant_id=variant.get_coordinate_key(),
        affected_positions=affected_positions,
        attention_target=attention_target,
        peak_position=peak_position,
        peak_effect_type=peak_effect_type,
        sequence_length=seq_len
    )


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _one_hot_encode(seq: str) -> np.ndarray:
    """One-hot encode a DNA sequence."""
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 0}
    seq = seq.upper()
    indices = [mapping.get(b, 0) for b in seq]
    oh = np.zeros((len(seq), 4), dtype=np.float32)
    oh[np.arange(len(seq)), indices] = 1
    return oh


def _get_base_model_scores(
    seq_oh: np.ndarray,
    base_model,
    device: str
) -> np.ndarray:
    """
    Get base model scores for a one-hot encoded sequence.
    
    Handles both single models and ensembles.
    """
    # Convert to tensor [1, 4, L]
    x = torch.tensor(seq_oh.T, dtype=torch.float32).unsqueeze(0)
    x = x.to(device)
    
    # Handle ensemble
    if isinstance(base_model, list):
        models = base_model
    else:
        models = [base_model]
    
    with torch.no_grad():
        preds = []
        for model in models:
            model.eval()
            pred = model(x).cpu()
            preds.append(pred)
        
        # Average ensemble predictions
        avg = torch.mean(torch.stack(preds), dim=0)
        
        # Convert to probabilities [B, C, L] -> [L, C]
        probs = F.softmax(avg.permute(0, 2, 1), dim=-1)
    
    return probs[0].numpy()  # [L, 3]


# =============================================================================
# STATISTICS AND ANALYSIS
# =============================================================================

def analyze_position_label_distribution(
    variants: List[VariantRecord],
    use_hgvs: bool = True
) -> Dict[str, int]:
    """
    Analyze the distribution of position labels from HGVS parsing.
    
    Parameters
    ----------
    variants : List[VariantRecord]
        List of variants to analyze
    use_hgvs : bool
        Whether to parse HGVS notation
    
    Returns
    -------
    Dict[str, int]
        Counts by site type and confidence level
    """
    stats = {
        'total': len(variants),
        'donor': 0,
        'acceptor': 0,
        'deep_intronic': 0,
        'exonic': 0,
        'unknown': 0,
        'canonical_region': 0,
        'extended_region': 0,
        'high_confidence': 0,
        'medium_confidence': 0,
        'low_confidence': 0,
    }
    
    for variant in variants:
        if use_hgvs:
            hint = variant.get_position_hint()
            stats[hint.site_type] = stats.get(hint.site_type, 0) + 1
            
            if hint.is_canonical_region:
                stats['canonical_region'] += 1
            if hint.is_extended_region:
                stats['extended_region'] += 1
            
            stats[f'{hint.confidence}_confidence'] += 1
    
    return stats

