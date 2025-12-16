"""
Biological evaluation for delta prediction models.

This module evaluates delta scores by:
1. Applying delta to base model scores
2. Finding splice site peaks in adjusted scores
3. Comparing predicted peaks to canonical annotations
4. Evaluating variant-specific effects (gain/loss)
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.signal import find_peaks
from sklearn.metrics import precision_score, recall_score, f1_score

logger = logging.getLogger(__name__)


@dataclass
class SpliceSitePeak:
    """A detected splice site peak."""
    position: int
    probability: float
    site_type: str  # 'donor' or 'acceptor'
    is_canonical: bool = False  # Whether it matches an annotation


@dataclass
class PeakEvaluationResult:
    """Results from peak-based evaluation."""
    precision: float
    recall: float
    f1: float
    true_positives: int
    false_positives: int
    false_negatives: int
    total_predicted: int
    total_annotated: int


def find_splice_site_peaks(
    scores: np.ndarray,
    threshold: float = 0.5,
    min_distance: int = 10,
    prominence: float = 0.1,
    site_type: str = 'donor'
) -> List[SpliceSitePeak]:
    """
    Find splice site peaks in probability scores.
    
    Parameters
    ----------
    scores : np.ndarray, shape [L]
        Probability scores for one site type (donor or acceptor)
    threshold : float
        Minimum probability to consider
    min_distance : int
        Minimum distance between peaks (in nucleotides)
    prominence : float
        Minimum prominence of peaks
    site_type : str
        'donor' or 'acceptor'
    
    Returns
    -------
    List[SpliceSitePeak]
        Detected peaks with positions and probabilities
    """
    # Filter by threshold
    above_threshold = scores >= threshold
    
    if not np.any(above_threshold):
        return []
    
    # Find peaks using scipy
    peaks, properties = find_peaks(
        scores,
        height=threshold,
        distance=min_distance,
        prominence=prominence
    )
    
    # Convert to SpliceSitePeak objects
    result = []
    for peak_idx in peaks:
        result.append(SpliceSitePeak(
            position=int(peak_idx),
            probability=float(scores[peak_idx]),
            site_type=site_type
        ))
    
    return result


def match_peaks_to_annotations(
    predicted_peaks: List[SpliceSitePeak],
    annotated_positions: List[int],
    tolerance: int = 2
) -> Tuple[List[SpliceSitePeak], List[SpliceSitePeak], List[int]]:
    """
    Match predicted peaks to annotated splice sites.
    
    Parameters
    ----------
    predicted_peaks : List[SpliceSitePeak]
        Predicted splice site peaks
    annotated_positions : List[int]
        Positions of annotated splice sites
    tolerance : int
        Maximum distance for a match (in nucleotides)
    
    Returns
    -------
    Tuple of (true_positives, false_positives, false_negatives)
    """
    true_positives = []
    false_positives = []
    matched_annotated = set()
    
    # Match each predicted peak
    for peak in predicted_peaks:
        matched = False
        for ann_pos in annotated_positions:
            if abs(peak.position - ann_pos) <= tolerance:
                peak.is_canonical = True
                true_positives.append(peak)
                matched_annotated.add(ann_pos)
                matched = True
                break
        
        if not matched:
            false_positives.append(peak)
    
    # Find false negatives (annotated but not predicted)
    false_negatives = [
        pos for pos in annotated_positions
        if pos not in matched_annotated
    ]
    
    return true_positives, false_positives, false_negatives


def evaluate_peak_detection(
    predicted_peaks: List[SpliceSitePeak],
    annotated_positions: List[int],
    tolerance: int = 2
) -> PeakEvaluationResult:
    """
    Evaluate splice site peak detection against annotations.
    
    Parameters
    ----------
    predicted_peaks : List[SpliceSitePeak]
        Predicted splice site peaks
    annotated_positions : List[int]
        Positions of annotated splice sites
    tolerance : int
        Maximum distance for a match
    
    Returns
    -------
    PeakEvaluationResult
        Evaluation metrics
    """
    tp, fp, fn = match_peaks_to_annotations(
        predicted_peaks, annotated_positions, tolerance
    )
    
    precision = len(tp) / max(1, len(predicted_peaks))
    recall = len(tp) / max(1, len(annotated_positions))
    f1 = 2 * precision * recall / max(1e-10, precision + recall)
    
    return PeakEvaluationResult(
        precision=precision,
        recall=recall,
        f1=f1,
        true_positives=len(tp),
        false_positives=len(fp),
        false_negatives=len(fn),
        total_predicted=len(predicted_peaks),
        total_annotated=len(annotated_positions)
    )


def compare_base_vs_adjusted_peaks(
    base_scores: np.ndarray,
    adjusted_scores: np.ndarray,
    annotated_positions: List[int],
    threshold: float = 0.5,
    tolerance: int = 2
) -> Dict[str, PeakEvaluationResult]:
    """
    Compare splice site detection between base and adjusted scores.
    
    Parameters
    ----------
    base_scores : np.ndarray, shape [L, 3]
        Base model scores [donor, acceptor, neither]
    adjusted_scores : np.ndarray, shape [L, 3]
        Delta-adjusted scores [donor, acceptor, neither]
    annotated_positions : List[int]
        Positions of annotated splice sites (for this site type)
    threshold : float
        Probability threshold for peak detection
    tolerance : int
        Match tolerance
    
    Returns
    -------
    Dict with 'base' and 'adjusted' evaluation results
    """
    # Extract donor and acceptor scores
    base_donor = base_scores[:, 0]
    base_acceptor = base_scores[:, 1]
    adj_donor = adjusted_scores[:, 0]
    adj_acceptor = adjusted_scores[:, 1]
    
    # Find peaks
    base_donor_peaks = find_splice_site_peaks(base_donor, threshold, site_type='donor')
    base_acceptor_peaks = find_splice_site_peaks(base_acceptor, threshold, site_type='acceptor')
    adj_donor_peaks = find_splice_site_peaks(adj_donor, threshold, site_type='donor')
    adj_acceptor_peaks = find_splice_site_peaks(adj_acceptor, threshold, site_type='acceptor')
    
    # Evaluate (assuming annotated_positions contains both donor and acceptor)
    # In practice, you'd separate them by site type
    base_result = evaluate_peak_detection(
        base_donor_peaks + base_acceptor_peaks,
        annotated_positions,
        tolerance
    )
    adjusted_result = evaluate_peak_detection(
        adj_donor_peaks + adj_acceptor_peaks,
        annotated_positions,
        tolerance
    )
    
    return {
        'base': base_result,
        'adjusted': adjusted_result
    }


def evaluate_variant_effect(
    variant_classification: str,
    base_peaks: List[SpliceSitePeak],
    adjusted_peaks: List[SpliceSitePeak],
    annotated_positions: List[int]
) -> Dict[str, any]:
    """
    Evaluate variant-specific effects (gain/loss detection).
    
    Parameters
    ----------
    variant_classification : str
        'Splice-altering' or 'Normal'
    base_peaks : List[SpliceSitePeak]
        Peaks in base model scores
    adjusted_peaks : List[SpliceSitePeak]
        Peaks in adjusted scores
    annotated_positions : List[int]
        Canonical annotated positions
    
    Returns
    -------
    Dict with variant effect metrics
    """
    base_positions = {p.position for p in base_peaks}
    adjusted_positions = {p.position for p in adjusted_peaks}
    
    # Find changes
    new_peaks = adjusted_positions - base_positions
    lost_peaks = base_positions - adjusted_positions
    
    # Check if new peaks are canonical
    new_canonical = sum(
        1 for pos in new_peaks
        if any(abs(pos - ann) <= 2 for ann in annotated_positions)
    )
    
    if variant_classification == 'Splice-altering':
        # Should see gain or loss
        return {
            'has_gain': len(new_peaks) > 0,
            'has_loss': len(lost_peaks) > 0,
            'new_peaks_count': len(new_peaks),
            'new_canonical_count': new_canonical,
            'lost_peaks_count': len(lost_peaks),
            'gain_detected': len(new_peaks) > 0 and new_canonical > 0
        }
    else:  # Normal
        # Should NOT create spurious peaks
        return {
            'spurious_peaks': len(new_peaks),
            'maintains_canonical': len(base_positions & adjusted_positions) / max(1, len(base_positions)),
            'false_gain_rate': len(new_peaks) / max(1, len(adjusted_peaks))
        }


def aggregate_biological_metrics(
    results: List[Dict[str, any]]
) -> Dict[str, float]:
    """
    Aggregate biological evaluation metrics across all variants.
    
    Parameters
    ----------
    results : List[Dict]
        Per-variant evaluation results
    
    Returns
    -------
    Dict with aggregated metrics
    """
    # Peak detection metrics
    base_precisions = [r['base'].precision for r in results if 'base' in r]
    base_recalls = [r['base'].recall for r in results if 'base' in r]
    adj_precisions = [r['adjusted'].precision for r in results if 'adjusted' in r]
    adj_recalls = [r['adjusted'].recall for r in results if 'adjusted' in r]
    
    # Variant effect metrics
    splice_altering = [r for r in results if r.get('classification') == 'Splice-altering']
    normal = [r for r in results if r.get('classification') == 'Normal']
    
    gain_detected = sum(
        1 for r in splice_altering
        if r.get('variant_effect', {}).get('gain_detected', False)
    )
    
    spurious_gains = sum(
        r.get('variant_effect', {}).get('spurious_peaks', 0)
        for r in normal
    )
    
    return {
        'base_precision': np.mean(base_precisions) if base_precisions else 0.0,
        'base_recall': np.mean(base_recalls) if base_recalls else 0.0,
        'adjusted_precision': np.mean(adj_precisions) if adj_precisions else 0.0,
        'adjusted_recall': np.mean(adj_recalls) if adj_recalls else 0.0,
        'precision_improvement': np.mean(adj_precisions) - np.mean(base_precisions) if adj_precisions and base_precisions else 0.0,
        'recall_improvement': np.mean(adj_recalls) - np.mean(base_recalls) if adj_recalls and base_recalls else 0.0,
        'gain_detection_rate': gain_detected / max(1, len(splice_altering)),
        'false_gain_rate': spurious_gains / max(1, len(normal)),
        'n_splice_altering': len(splice_altering),
        'n_normal': len(normal)
    }

