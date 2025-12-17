"""
Position Label Derivation for Multi-Step Framework Step 3.

This module provides utilities for deriving position-level labels for
position localization training. Labels can be derived from:

1. Base model delta analysis (RECOMMENDED)
2. HGVS parsing (if available)
3. Effect type annotation (for specific site types)

Usage
-----
>>> from meta_spliceai.splice_engine.meta_layer.data.position_labels import (
...     derive_position_labels_from_delta,
...     create_position_attention_target,
...     create_binary_position_mask
... )
>>> 
>>> # From base model delta
>>> affected = derive_position_labels_from_delta(ref_seq, alt_seq, models, device)
>>> 
>>> # Create training target
>>> attention_target = create_position_attention_target(affected, seq_length=501)

See Also
--------
- docs/methods/MULTI_STEP_FRAMEWORK.md
- models/position_localizer.py
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class AffectedPosition:
    """
    Represents a position affected by a splice-altering variant.
    
    Attributes
    ----------
    position : int
        Position in the sequence (relative to center)
    delta_value : float
        Delta magnitude at this position
    channel : int
        Which channel (0=neither, 1=acceptor, 2=donor)
    effect_type : str
        'donor_gain', 'donor_loss', 'acceptor_gain', 'acceptor_loss', 'unknown'
    """
    position: int
    delta_value: float
    channel: int
    effect_type: str
    
    @property
    def is_gain(self) -> bool:
        return self.delta_value > 0
    
    @property
    def is_loss(self) -> bool:
        return self.delta_value < 0
    
    @property
    def is_donor(self) -> bool:
        return self.channel == 2
    
    @property
    def is_acceptor(self) -> bool:
        return self.channel == 1


# =============================================================================
# DELTA-BASED DERIVATION (RECOMMENDED)
# =============================================================================

def derive_position_labels_from_delta(
    ref_seq: str,
    alt_seq: str,
    models: List,
    device: str,
    threshold: float = 0.1,
    window: int = 50,
    max_positions: int = 5
) -> List[AffectedPosition]:
    """
    Derive affected positions from base model delta analysis.
    
    This is the RECOMMENDED approach for position label derivation because
    it directly identifies where the splice site probability changes most.
    
    Parameters
    ----------
    ref_seq : str
        Reference DNA sequence (should be 10K+ for base model)
    alt_seq : str
        Alternate DNA sequence with variant embedded
    models : List
        List of base models (OpenSpliceAI or similar)
    device : str
        Device for inference
    threshold : float
        Minimum |delta| to consider significant
    window : int
        Window around center to search (±window bp)
    max_positions : int
        Maximum number of affected positions to return
    
    Returns
    -------
    List[AffectedPosition]
        Affected positions sorted by delta magnitude (descending)
    
    Example
    -------
    >>> affected = derive_position_labels_from_delta(ref, alt, models, 'cuda')
    >>> for ap in affected:
    ...     print(f"Position {ap.position}: {ap.effect_type} (Δ={ap.delta_value:.3f})")
    """
    # One-hot encode
    def one_hot(seq: str) -> np.ndarray:
        mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 0}
        indices = [mapping.get(b.upper(), 0) for b in seq]
        oh = np.zeros((4, len(seq)), dtype=np.float32)
        oh[indices, np.arange(len(seq))] = 1
        return oh
    
    # Run base models
    def predict(seq: str) -> np.ndarray:
        x = torch.tensor(one_hot(seq), dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            preds = [m(x).cpu() for m in models]
            avg = torch.mean(torch.stack(preds), dim=0)
            # OpenSpliceAI outputs [B, 3, L], softmax along channel
            probs = F.softmax(avg, dim=1)
        return probs[0].permute(1, 0).numpy()  # [L, 3]
    
    ref_probs = predict(ref_seq)
    alt_probs = predict(alt_seq)
    delta = alt_probs - ref_probs  # [L, 3]
    
    # Focus on center region
    center = len(delta) // 2
    start = max(0, center - window)
    end = min(len(delta), center + window + 1)
    
    # Channel names: [neither, acceptor, donor]
    channel_names = ['neither', 'acceptor', 'donor']
    
    # Find positions with significant delta
    affected = []
    
    for pos in range(start, end):
        rel_pos = pos - center  # Position relative to center
        
        for ch in [1, 2]:  # Only acceptor (1) and donor (2)
            d = delta[pos, ch]
            
            if abs(d) >= threshold:
                # Determine effect type
                if ch == 2:  # donor
                    effect_type = 'donor_gain' if d > 0 else 'donor_loss'
                else:  # acceptor
                    effect_type = 'acceptor_gain' if d > 0 else 'acceptor_loss'
                
                affected.append(AffectedPosition(
                    position=rel_pos,
                    delta_value=d,
                    channel=ch,
                    effect_type=effect_type
                ))
    
    # Sort by delta magnitude
    affected.sort(key=lambda x: abs(x.delta_value), reverse=True)
    
    return affected[:max_positions]


def derive_position_labels_per_channel(
    ref_seq: str,
    alt_seq: str,
    models: List,
    device: str,
    window: int = 50
) -> Dict[str, AffectedPosition]:
    """
    Derive the single most affected position PER CHANNEL.
    
    This is an alternative to derive_position_labels_from_delta that
    separately finds the max delta for donor and acceptor channels.
    
    Returns
    -------
    Dict[str, AffectedPosition]
        Keys: 'donor', 'acceptor'. May be empty if no significant delta.
    """
    # One-hot encode
    def one_hot(seq: str) -> np.ndarray:
        mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 0}
        indices = [mapping.get(b.upper(), 0) for b in seq]
        oh = np.zeros((4, len(seq)), dtype=np.float32)
        oh[indices, np.arange(len(seq))] = 1
        return oh
    
    def predict(seq: str) -> np.ndarray:
        x = torch.tensor(one_hot(seq), dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            preds = [m(x).cpu() for m in models]
            avg = torch.mean(torch.stack(preds), dim=0)
            probs = F.softmax(avg, dim=1)
        return probs[0].permute(1, 0).numpy()
    
    ref_probs = predict(ref_seq)
    alt_probs = predict(alt_seq)
    delta = alt_probs - ref_probs
    
    center = len(delta) // 2
    start = max(0, center - window)
    end = min(len(delta), center + window + 1)
    
    results = {}
    
    # Donor channel (2)
    donor_delta = delta[start:end, 2]
    max_donor_idx = np.abs(donor_delta).argmax()
    max_donor_val = donor_delta[max_donor_idx]
    if abs(max_donor_val) > 0.01:  # Minimum threshold
        results['donor'] = AffectedPosition(
            position=max_donor_idx - window,  # Relative to center
            delta_value=max_donor_val,
            channel=2,
            effect_type='donor_gain' if max_donor_val > 0 else 'donor_loss'
        )
    
    # Acceptor channel (1)
    acc_delta = delta[start:end, 1]
    max_acc_idx = np.abs(acc_delta).argmax()
    max_acc_val = acc_delta[max_acc_idx]
    if abs(max_acc_val) > 0.01:
        results['acceptor'] = AffectedPosition(
            position=max_acc_idx - window,
            delta_value=max_acc_val,
            channel=1,
            effect_type='acceptor_gain' if max_acc_val > 0 else 'acceptor_loss'
        )
    
    return results


# =============================================================================
# TRAINING TARGET GENERATION
# =============================================================================

def create_position_attention_target(
    affected_positions: List[AffectedPosition],
    seq_length: int,
    center: Optional[int] = None,
    sigma: float = 3.0,
    delta_weighted: bool = True
) -> np.ndarray:
    """
    Create soft attention target distribution for position localization.
    
    Parameters
    ----------
    affected_positions : List[AffectedPosition]
        List of affected positions (from derive_position_labels_from_delta)
    seq_length : int
        Length of the sequence context
    center : int, optional
        Center position in sequence. Defaults to seq_length // 2
    sigma : float
        Standard deviation of Gaussian for each peak
    delta_weighted : bool
        If True, weight Gaussians by delta magnitude
    
    Returns
    -------
    np.ndarray
        Soft attention distribution [seq_length], sums to 1
    
    Example
    -------
    >>> target = create_position_attention_target(affected, seq_length=501)
    >>> assert target.shape == (501,)
    >>> assert np.isclose(target.sum(), 1.0)
    """
    if center is None:
        center = seq_length // 2
    
    x = np.arange(seq_length)
    attention = np.zeros(seq_length, dtype=np.float32)
    
    if not affected_positions:
        # Fallback: uniform Gaussian at center
        attention = np.exp(-0.5 * ((x - center) / sigma) ** 2)
    else:
        for ap in affected_positions:
            # Convert relative position to absolute
            abs_pos = center + ap.position
            
            if 0 <= abs_pos < seq_length:
                # Weight by delta magnitude if requested
                weight = abs(ap.delta_value) if delta_weighted else 1.0
                
                # Add Gaussian centered at this position
                gaussian = np.exp(-0.5 * ((x - abs_pos) / sigma) ** 2)
                attention += weight * gaussian
    
    # Normalize to sum to 1
    if attention.sum() > 0:
        attention = attention / attention.sum()
    else:
        # Fallback: uniform
        attention = np.ones(seq_length) / seq_length
    
    return attention


def create_binary_position_mask(
    affected_positions: List[AffectedPosition],
    seq_length: int,
    center: Optional[int] = None,
    radius: int = 2
) -> np.ndarray:
    """
    Create binary mask target for position segmentation.
    
    Parameters
    ----------
    affected_positions : List[AffectedPosition]
        List of affected positions
    seq_length : int
        Length of the sequence context
    center : int, optional
        Center position in sequence
    radius : int
        Radius around each affected position to mark as 1
    
    Returns
    -------
    np.ndarray
        Binary mask [seq_length]
    
    Example
    -------
    >>> mask = create_binary_position_mask(affected, seq_length=501, radius=2)
    >>> assert mask.shape == (501,)
    >>> assert set(np.unique(mask)) <= {0, 1}
    """
    if center is None:
        center = seq_length // 2
    
    mask = np.zeros(seq_length, dtype=np.float32)
    
    for ap in affected_positions:
        abs_pos = center + ap.position
        
        # Mark positions within radius
        for i in range(max(0, abs_pos - radius), min(seq_length, abs_pos + radius + 1)):
            mask[i] = 1.0
    
    return mask


def create_offset_target(
    affected_positions: List[AffectedPosition],
    seq_length: int = 1
) -> np.ndarray:
    """
    Create offset regression target (distance from variant to effect).
    
    For simple offset prediction (Step 3 alternative), returns the
    offset in bp from the variant position to the most significant
    affected position.
    
    Parameters
    ----------
    affected_positions : List[AffectedPosition]
        List of affected positions (sorted by delta magnitude)
    seq_length : int
        Not used, for API consistency
    
    Returns
    -------
    np.ndarray
        [offset_donor, offset_acceptor] or [offset] if single prediction
    """
    offset_donor = 0
    offset_acceptor = 0
    
    for ap in affected_positions:
        if ap.channel == 2 and offset_donor == 0:  # Donor
            offset_donor = ap.position
        elif ap.channel == 1 and offset_acceptor == 0:  # Acceptor
            offset_acceptor = ap.position
    
    return np.array([offset_donor, offset_acceptor], dtype=np.float32)


# =============================================================================
# HGVS-BASED DERIVATION (ALTERNATIVE)
# =============================================================================

def derive_position_from_hgvs(
    hgvs: str,
    effect_type: Optional[str] = None
) -> Optional[AffectedPosition]:
    """
    Derive affected position from HGVS notation (when available).
    
    HGVS can encode exact splice site positions (e.g., c.123+1G>A).
    This is useful when HGVS annotations are available in SpliceVarDB.
    
    Parameters
    ----------
    hgvs : str
        HGVS notation (e.g., 'c.123+1G>A', 'c.456-2A>G')
    effect_type : str, optional
        Known effect type from annotation
    
    Returns
    -------
    Optional[AffectedPosition]
        Affected position if parseable, None otherwise
    """
    import re
    
    # Pattern for intronic positions: c.X+Y or c.X-Y
    intronic_pattern = r'c\.(\d+)([+-])(\d+)([ACGT])>([ACGT])'
    match = re.search(intronic_pattern, hgvs)
    
    if match:
        exon_pos = int(match.group(1))
        direction = match.group(2)
        offset = int(match.group(3))
        ref_base = match.group(4)
        alt_base = match.group(5)
        
        # Determine if this is donor (+) or acceptor (-) region
        if direction == '+':
            # Donor site region (5' end of intron)
            if offset <= 8:  # Close to splice site
                channel = 2  # donor
                inferred_type = 'donor_loss'  # Mutation in canonical site
            else:
                channel = 0  # neither
                inferred_type = 'unknown'
        else:  # direction == '-'
            # Acceptor site region (3' end of intron)
            if offset <= 8:
                channel = 1  # acceptor
                inferred_type = 'acceptor_loss'
            else:
                channel = 0
                inferred_type = 'unknown'
        
        # Use provided effect type if available
        final_type = effect_type if effect_type else inferred_type
        
        return AffectedPosition(
            position=0,  # HGVS doesn't give absolute position easily
            delta_value=-0.5,  # Placeholder - we don't know actual delta
            channel=channel,
            effect_type=final_type
        )
    
    return None


# =============================================================================
# UTILITIES
# =============================================================================

def effect_type_to_channel(effect_type: str) -> int:
    """Convert effect type string to channel index."""
    mapping = {
        'donor_gain': 2,
        'donor_loss': 2,
        'acceptor_gain': 1,
        'acceptor_loss': 1,
        'unknown': 0,
        'neither': 0
    }
    return mapping.get(effect_type.lower(), 0)


def channel_to_effect_type(channel: int, is_gain: bool) -> str:
    """Convert channel index and direction to effect type string."""
    if channel == 2:
        return 'donor_gain' if is_gain else 'donor_loss'
    elif channel == 1:
        return 'acceptor_gain' if is_gain else 'acceptor_loss'
    else:
        return 'unknown'


def summarize_affected_positions(
    affected_positions: List[AffectedPosition]
) -> Dict[str, any]:
    """
    Summarize affected positions for logging/debugging.
    
    Returns
    -------
    Dict containing:
        - n_positions: Number of affected positions
        - effect_types: List of unique effect types
        - max_delta: Maximum absolute delta
        - positions: List of position offsets
    """
    if not affected_positions:
        return {
            'n_positions': 0,
            'effect_types': [],
            'max_delta': 0.0,
            'positions': []
        }
    
    return {
        'n_positions': len(affected_positions),
        'effect_types': list(set(ap.effect_type for ap in affected_positions)),
        'max_delta': max(abs(ap.delta_value) for ap in affected_positions),
        'positions': [ap.position for ap in affected_positions]
    }
