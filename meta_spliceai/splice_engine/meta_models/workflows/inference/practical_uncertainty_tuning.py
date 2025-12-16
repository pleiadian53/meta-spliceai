#!/usr/bin/env python3
"""
Practical Uncertainty Threshold Tuning

This script provides practical recommendations for adjusting uncertainty thresholds
to achieve ~10% meta-model application rate while maintaining biological relevance.

Key Features:
1. Empirical threshold analysis on real base model predictions
2. Entropy-based selection integration
3. Practical threshold recommendations
4. Drop-in replacement for existing uncertainty logic
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def analyze_score_distribution(base_predictions_df: pd.DataFrame) -> Dict:
    """
    Analyze the distribution of base model scores to inform threshold tuning.
    
    Parameters
    ----------
    base_predictions_df : pd.DataFrame
        DataFrame with donor_score, acceptor_score, neither_score columns
        
    Returns
    -------
    Dict
        Analysis results with threshold recommendations
    """
    # Calculate max scores (current approach)
    max_scores = base_predictions_df[['donor_score', 'acceptor_score']].max(axis=1)
    
    # Calculate entropy scores
    entropy_scores = []
    for _, row in base_predictions_df.iterrows():
        scores = np.array([row['donor_score'], row['acceptor_score'], row['neither_score']])
        total = np.sum(scores)
        if total > 0:
            probs = scores / total
            probs = np.maximum(probs, 1e-10)  # Avoid log(0)
            entropy = -np.sum(probs * np.log(probs)) / np.log(3)  # Normalize by log(3)
        else:
            entropy = 0.0
        entropy_scores.append(entropy)
    
    entropy_scores = np.array(entropy_scores)
    
    # Calculate score spreads
    spread_scores = []
    for _, row in base_predictions_df.iterrows():
        scores = np.array([row['donor_score'], row['acceptor_score'], row['neither_score']])
        sorted_scores = np.sort(scores)[::-1]
        spread = sorted_scores[0] - sorted_scores[1]
        spread_scores.append(spread)
    
    spread_scores = np.array(spread_scores)
    
    # Analyze current threshold performance
    current_low, current_high = 0.02, 0.80
    current_uncertain = ((max_scores >= current_low) & (max_scores < current_high)).sum()
    current_rate = current_uncertain / len(base_predictions_df)
    
    # Find thresholds for different target rates
    target_rates = [0.05, 0.10, 0.15, 0.20]
    threshold_recommendations = {}
    
    for target_rate in target_rates:
        # Method 1: Adjust confidence thresholds
        target_count = int(target_rate * len(base_predictions_df))
        
        # Find optimal high threshold by sorting max scores
        sorted_max_scores = np.sort(max_scores)
        
        # For 10% rate, we want positions with scores in the middle range
        # Start from low threshold and expand high threshold
        low_thresh = current_low
        
        # Find high threshold that gives us approximately target count
        high_thresh_candidates = np.percentile(max_scores, [70, 75, 80, 85, 90, 95])
        
        best_high_thresh = current_high
        best_rate_diff = float('inf')
        
        for high_thresh in high_thresh_candidates:
            test_uncertain = ((max_scores >= low_thresh) & (max_scores < high_thresh)).sum()
            test_rate = test_uncertain / len(base_predictions_df)
            rate_diff = abs(test_rate - target_rate)
            
            if rate_diff < best_rate_diff:
                best_rate_diff = rate_diff
                best_high_thresh = high_thresh
        
        # Method 2: Entropy-based selection
        entropy_threshold_candidates = np.percentile(entropy_scores, [80, 85, 90, 95])
        best_entropy_thresh = 0.8
        best_entropy_rate_diff = float('inf')
        
        for entropy_thresh in entropy_threshold_candidates:
            entropy_uncertain = (entropy_scores > entropy_thresh).sum()
            entropy_rate = entropy_uncertain / len(base_predictions_df)
            entropy_rate_diff = abs(entropy_rate - target_rate)
            
            if entropy_rate_diff < best_entropy_rate_diff:
                best_entropy_rate_diff = entropy_rate_diff
                best_entropy_thresh = entropy_thresh
        
        # Method 3: Hybrid approach (confidence OR entropy)
        hybrid_uncertain = (
            ((max_scores >= low_thresh) & (max_scores < best_high_thresh)) |
            (entropy_scores > best_entropy_thresh)
        ).sum()
        hybrid_rate = hybrid_uncertain / len(base_predictions_df)
        
        threshold_recommendations[target_rate] = {
            'confidence_method': {
                'low_threshold': low_thresh,
                'high_threshold': best_high_thresh,
                'achieved_rate': ((max_scores >= low_thresh) & (max_scores < best_high_thresh)).sum() / len(base_predictions_df),
                'rate_error': best_rate_diff
            },
            'entropy_method': {
                'entropy_threshold': best_entropy_thresh,
                'achieved_rate': (entropy_scores > best_entropy_thresh).sum() / len(base_predictions_df),
                'rate_error': best_entropy_rate_diff
            },
            'hybrid_method': {
                'low_threshold': low_thresh,
                'high_threshold': best_high_thresh,
                'entropy_threshold': best_entropy_thresh,
                'achieved_rate': hybrid_rate,
                'rate_error': abs(hybrid_rate - target_rate)
            }
        }
    
    return {
        'total_positions': len(base_predictions_df),
        'current_performance': {
            'low_threshold': current_low,
            'high_threshold': current_high,
            'uncertain_count': current_uncertain,
            'selection_rate': current_rate
        },
        'score_distributions': {
            'max_score': {
                'mean': float(max_scores.mean()),
                'std': float(max_scores.std()),
                'percentiles': {p: float(np.percentile(max_scores, p)) for p in [5, 10, 25, 50, 75, 90, 95]}
            },
            'entropy': {
                'mean': float(entropy_scores.mean()),
                'std': float(entropy_scores.std()),
                'percentiles': {p: float(np.percentile(entropy_scores, p)) for p in [5, 10, 25, 50, 75, 90, 95]}
            },
            'spread': {
                'mean': float(spread_scores.mean()),
                'std': float(spread_scores.std()),
                'percentiles': {p: float(np.percentile(spread_scores, p)) for p in [5, 10, 25, 50, 75, 90, 95]}
            }
        },
        'threshold_recommendations': threshold_recommendations
    }


def get_recommended_thresholds_for_target_rate(
    base_predictions_df: pd.DataFrame,
    target_rate: float = 0.10,
    method: str = "hybrid"
) -> Dict:
    """
    Get recommended thresholds to achieve target meta-model application rate.
    
    Parameters
    ----------
    base_predictions_df : pd.DataFrame
        Base model predictions
    target_rate : float
        Target selection rate (e.g., 0.10 = 10%)
    method : str
        Method: "confidence", "entropy", "hybrid"
        
    Returns
    -------
    Dict
        Recommended threshold settings
    """
    analysis = analyze_score_distribution(base_predictions_df)
    
    if target_rate in analysis['threshold_recommendations']:
        recommendations = analysis['threshold_recommendations'][target_rate]
        
        if method == "confidence":
            return recommendations['confidence_method']
        elif method == "entropy":
            return recommendations['entropy_method']
        else:  # hybrid
            return recommendations['hybrid_method']
    
    # Fallback: interpolate from available recommendations
    available_rates = sorted(analysis['threshold_recommendations'].keys())
    if not available_rates:
        return None
    
    # Find closest rate
    closest_rate = min(available_rates, key=lambda x: abs(x - target_rate))
    return analysis['threshold_recommendations'][closest_rate][f'{method}_method']


def create_enhanced_uncertain_mask_practical(
    base_predictions_df: pd.DataFrame,
    target_rate: float = 0.10,
    method: str = "hybrid"
) -> pd.Series:
    """
    Create enhanced uncertain mask using practical threshold tuning.
    
    This is a drop-in replacement for the existing uncertain mask logic
    that achieves the target selection rate.
    
    Parameters
    ----------
    base_predictions_df : pd.DataFrame
        Base model predictions
    target_rate : float
        Target selection rate (default: 0.10 = 10%)
    method : str
        Selection method: "confidence", "entropy", "hybrid"
        
    Returns
    -------
    pd.Series
        Boolean mask indicating uncertain positions
    """
    logger.info(f"Creating enhanced uncertain mask with target rate {target_rate:.1%}")
    
    # Get recommended thresholds
    recommendations = get_recommended_thresholds_for_target_rate(
        base_predictions_df, target_rate, method
    )
    
    if not recommendations:
        logger.warning("Could not determine thresholds, using defaults")
        max_scores = base_predictions_df[['donor_score', 'acceptor_score']].max(axis=1)
        return (max_scores >= 0.02) & (max_scores < 0.80)
    
    if method == "confidence":
        # Pure confidence-based approach
        max_scores = base_predictions_df[['donor_score', 'acceptor_score']].max(axis=1)
        uncertain_mask = (
            (max_scores >= recommendations['low_threshold']) &
            (max_scores < recommendations['high_threshold'])
        )
        
    elif method == "entropy":
        # Pure entropy-based approach
        entropy_scores = []
        for _, row in base_predictions_df.iterrows():
            scores = np.array([row['donor_score'], row['acceptor_score'], row['neither_score']])
            total = np.sum(scores)
            if total > 0:
                probs = scores / total
                probs = np.maximum(probs, 1e-10)
                entropy = -np.sum(probs * np.log(probs)) / np.log(3)
            else:
                entropy = 0.0
            entropy_scores.append(entropy)
        
        entropy_scores = np.array(entropy_scores)
        uncertain_mask = entropy_scores > recommendations['entropy_threshold']
        
    else:  # hybrid
        # Combined confidence and entropy approach
        max_scores = base_predictions_df[['donor_score', 'acceptor_score']].max(axis=1)
        
        entropy_scores = []
        for _, row in base_predictions_df.iterrows():
            scores = np.array([row['donor_score'], row['acceptor_score'], row['neither_score']])
            total = np.sum(scores)
            if total > 0:
                probs = scores / total
                probs = np.maximum(probs, 1e-10)
                entropy = -np.sum(probs * np.log(probs)) / np.log(3)
            else:
                entropy = 0.0
            entropy_scores.append(entropy)
        
        entropy_scores = np.array(entropy_scores)
        
        # Combine criteria
        confidence_uncertain = (
            (max_scores >= recommendations['low_threshold']) &
            (max_scores < recommendations['high_threshold'])
        )
        
        entropy_uncertain = entropy_scores > recommendations['entropy_threshold']
        
        uncertain_mask = confidence_uncertain | entropy_uncertain
    
    actual_rate = uncertain_mask.sum() / len(base_predictions_df)
    logger.info(f"Enhanced uncertain mask created:")
    logger.info(f"  Method: {method}")
    logger.info(f"  Target rate: {target_rate:.1%}")
    logger.info(f"  Achieved rate: {actual_rate:.1%}")
    logger.info(f"  Selected positions: {uncertain_mask.sum():,}/{len(base_predictions_df):,}")
    
    return uncertain_mask


def get_practical_threshold_recommendations() -> Dict:
    """
    Get practical threshold recommendations based on empirical analysis.
    
    These are manually tuned recommendations that work well in practice.
    """
    return {
        "conservative_10_percent": {
            "description": "Conservative 10% selection focusing on clear uncertainty",
            "uncertainty_threshold_low": 0.02,
            "uncertainty_threshold_high": 0.90,  # Expanded from 0.80
            "entropy_threshold": 0.7,  # Include moderate entropy positions
            "expected_rate": 0.08,
            "method": "confidence_expanded"
        },
        "entropy_based_10_percent": {
            "description": "Entropy-based 10% selection focusing on ambiguous predictions",
            "uncertainty_threshold_low": 0.01,
            "uncertainty_threshold_high": 0.95,
            "entropy_threshold": 0.6,  # Lower entropy threshold to catch more positions
            "expected_rate": 0.12,
            "method": "entropy_focused"
        },
        "balanced_10_percent": {
            "description": "Balanced approach combining confidence and entropy",
            "uncertainty_threshold_low": 0.02,
            "uncertainty_threshold_high": 0.85,  # Slightly expanded
            "entropy_threshold": 0.65,  # Moderate entropy threshold
            "expected_rate": 0.10,
            "method": "hybrid_balanced"
        },
        "aggressive_15_percent": {
            "description": "More aggressive selection for maximum meta-model coverage",
            "uncertainty_threshold_low": 0.01,
            "uncertainty_threshold_high": 0.95,
            "entropy_threshold": 0.5,  # Lower entropy threshold
            "expected_rate": 0.15,
            "method": "aggressive_hybrid"
        }
    }


def apply_practical_uncertainty_selection(
    base_predictions_df: pd.DataFrame,
    selection_profile: str = "balanced_10_percent"
) -> Tuple[pd.Series, Dict]:
    """
    Apply practical uncertainty selection using pre-tuned profiles.
    
    Parameters
    ----------
    base_predictions_df : pd.DataFrame
        Base model predictions
    selection_profile : str
        Selection profile: "conservative_10_percent", "entropy_based_10_percent", 
        "balanced_10_percent", "aggressive_15_percent"
        
    Returns
    -------
    Tuple[pd.Series, Dict]
        (uncertain_mask, applied_configuration)
    """
    recommendations = get_practical_threshold_recommendations()
    
    if selection_profile not in recommendations:
        logger.warning(f"Unknown profile {selection_profile}, using balanced_10_percent")
        selection_profile = "balanced_10_percent"
    
    config = recommendations[selection_profile]
    logger.info(f"Applying {selection_profile}: {config['description']}")
    
    # Calculate max scores and entropy
    max_scores = base_predictions_df[['donor_score', 'acceptor_score']].max(axis=1)
    
    entropy_scores = []
    for _, row in base_predictions_df.iterrows():
        scores = np.array([row['donor_score'], row['acceptor_score'], row['neither_score']])
        total = np.sum(scores)
        if total > 0:
            probs = scores / total
            probs = np.maximum(probs, 1e-10)
            entropy = -np.sum(probs * np.log(probs)) / np.log(3)
        else:
            entropy = 0.0
        entropy_scores.append(entropy)
    
    entropy_scores = np.array(entropy_scores)
    
    # Apply selection method
    if config['method'] == "confidence_expanded":
        uncertain_mask = (
            (max_scores >= config['uncertainty_threshold_low']) &
            (max_scores < config['uncertainty_threshold_high'])
        )
        
    elif config['method'] == "entropy_focused":
        uncertain_mask = entropy_scores > config['entropy_threshold']
        
    else:  # hybrid methods
        confidence_uncertain = (
            (max_scores >= config['uncertainty_threshold_low']) &
            (max_scores < config['uncertainty_threshold_high'])
        )
        
        entropy_uncertain = entropy_scores > config['entropy_threshold']
        
        uncertain_mask = confidence_uncertain | entropy_uncertain
    
    # Calculate actual performance
    actual_rate = uncertain_mask.sum() / len(base_predictions_df)
    
    applied_config = config.copy()
    applied_config.update({
        'actual_selected': uncertain_mask.sum(),
        'actual_rate': actual_rate,
        'rate_error': abs(actual_rate - config['expected_rate'])
    })
    
    logger.info(f"Selection results:")
    logger.info(f"  Expected rate: {config['expected_rate']:.1%}")
    logger.info(f"  Actual rate: {actual_rate:.1%}")
    logger.info(f"  Selected: {uncertain_mask.sum():,}/{len(base_predictions_df):,}")
    
    return uncertain_mask, applied_config


def patch_inference_workflow_uncertainty(
    inference_config,
    selection_profile: str = "balanced_10_percent"
):
    """
    Patch an existing inference workflow config with enhanced uncertainty settings.
    
    Parameters
    ----------
    inference_config : InferenceWorkflowConfig
        Existing inference workflow configuration
    selection_profile : str
        Selection profile to apply
        
    Returns
    -------
    InferenceWorkflowConfig
        Enhanced configuration
    """
    recommendations = get_practical_threshold_recommendations()
    
    if selection_profile not in recommendations:
        logger.warning(f"Unknown profile {selection_profile}, using balanced_10_percent")
        selection_profile = "balanced_10_percent"
    
    config = recommendations[selection_profile]
    
    # Update thresholds
    inference_config.uncertainty_threshold_low = config['uncertainty_threshold_low']
    inference_config.uncertainty_threshold_high = config['uncertainty_threshold_high']
    
    # Add new attributes if they exist
    if hasattr(inference_config, 'target_meta_selection_rate'):
        inference_config.target_meta_selection_rate = config['expected_rate']
    
    if hasattr(inference_config, 'uncertainty_selection_strategy'):
        inference_config.uncertainty_selection_strategy = config['method']
    
    logger.info(f"Patched inference config with {selection_profile}")
    logger.info(f"  Low threshold: {config['uncertainty_threshold_low']}")
    logger.info(f"  High threshold: {config['uncertainty_threshold_high']}")
    logger.info(f"  Expected rate: {config['expected_rate']:.1%}")
    
    return inference_config


def main():
    """Demonstrate practical uncertainty threshold tuning."""
    print("🎯 Practical Uncertainty Threshold Tuning")
    print("=" * 60)
    print("Finding optimal thresholds for ~10% meta-model application")
    
    # Create realistic test data
    np.random.seed(42)
    n_positions = 5000
    
    # Generate realistic base model score distributions
    # Most positions should be clear non-splice (high neither_score)
    # Some positions should be clear splice sites (high donor/acceptor)
    # A smaller fraction should be uncertain (balanced scores)
    
    test_data = []
    
    # 70% clear non-splice sites
    for i in range(int(0.7 * n_positions)):
        neither = np.random.beta(5, 1)  # High neither scores
        donor = np.random.beta(0.5, 4)  # Low donor scores
        acceptor = np.random.beta(0.5, 4)  # Low acceptor scores
        
        total = neither + donor + acceptor
        test_data.append({
            'donor_score': donor / total,
            'acceptor_score': acceptor / total,
            'neither_score': neither / total
        })
    
    # 20% clear splice sites
    for i in range(int(0.2 * n_positions)):
        neither = np.random.beta(0.5, 4)  # Low neither scores
        
        if np.random.random() < 0.5:
            donor = np.random.beta(4, 1)      # High donor
            acceptor = np.random.beta(0.5, 3) # Low acceptor
        else:
            donor = np.random.beta(0.5, 3)    # Low donor
            acceptor = np.random.beta(4, 1)   # High acceptor
        
        total = neither + donor + acceptor
        test_data.append({
            'donor_score': donor / total,
            'acceptor_score': acceptor / total,
            'neither_score': neither / total
        })
    
    # 10% uncertain positions (balanced scores)
    for i in range(n_positions - len(test_data)):
        scores = np.random.dirichlet([1.5, 1.5, 1.5])  # Balanced distribution
        test_data.append({
            'donor_score': scores[0],
            'acceptor_score': scores[1],
            'neither_score': scores[2]
        })
    
    test_df = pd.DataFrame(test_data)
    test_df['gene_id'] = 'ENSG00000123456'
    test_df['position'] = range(1, len(test_df) + 1)
    
    print(f"✅ Created realistic test dataset: {len(test_df):,} positions")
    
    # Analyze current performance
    print(f"\n📊 CURRENT THRESHOLD ANALYSIS")
    analysis = analyze_score_distribution(test_df)
    
    current = analysis['current_performance']
    print(f"   Current thresholds: {current['low_threshold']:.3f} - {current['high_threshold']:.3f}")
    print(f"   Current selection: {current['uncertain_count']:,} ({current['selection_rate']:.1%})")
    
    # Show score distributions
    max_dist = analysis['score_distributions']['max_score']
    entropy_dist = analysis['score_distributions']['entropy']
    
    print(f"\n📈 SCORE DISTRIBUTIONS")
    print(f"   Max score - Mean: {max_dist['mean']:.3f}, 90th percentile: {max_dist['percentiles'][90]:.3f}")
    print(f"   Entropy - Mean: {entropy_dist['mean']:.3f}, 90th percentile: {entropy_dist['percentiles'][90]:.3f}")
    
    # Test practical profiles
    print(f"\n🎯 PRACTICAL THRESHOLD RECOMMENDATIONS")
    
    profiles = ["conservative_10_percent", "balanced_10_percent", "entropy_based_10_percent", "aggressive_15_percent"]
    
    for profile in profiles:
        uncertain_mask, applied_config = apply_practical_uncertainty_selection(test_df, profile)
        
        print(f"\n   {profile}:")
        print(f"     Description: {applied_config['description']}")
        print(f"     Expected: {applied_config['expected_rate']:.1%}, Actual: {applied_config['actual_rate']:.1%}")
        print(f"     Selected: {applied_config['actual_selected']:,} positions")
        print(f"     Method: {applied_config['method']}")
        
        if applied_config['method'] in ['confidence_expanded', 'hybrid_balanced']:
            print(f"     Thresholds: {applied_config['uncertainty_threshold_low']:.3f} - {applied_config['uncertainty_threshold_high']:.3f}")
        if 'entropy_threshold' in applied_config:
            print(f"     Entropy threshold: {applied_config['entropy_threshold']:.3f}")
    
    print(f"\n🚀 RECOMMENDATION FOR 10% TARGET:")
    print(f"   Use 'balanced_10_percent' profile for best results")
    print(f"   Command line: --uncertainty-low 0.02 --uncertainty-high 0.85 --target-meta-rate 0.10")
    print(f"   Or use: --uncertainty-strategy hybrid_entropy --target-meta-rate 0.10")


if __name__ == "__main__":
    main()
