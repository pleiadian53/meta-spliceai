"""
Enhanced Uncertainty Selection for Meta-Model Inference

This module implements improved uncertainty detection that combines:
1. Traditional confidence-based thresholds
2. Entropy-based uncertainty detection
3. Score spread analysis
4. Adaptive threshold tuning to achieve target selection rates

Goal: Select ~10% of positions for meta-model inference (vs current ~3%)
while focusing on positions where meta-model can provide maximum benefit.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class EnhancedUncertaintyConfig:
    """Configuration for enhanced uncertainty selection."""
    
    # Traditional confidence thresholds (current system)
    confidence_threshold_low: float = 0.02
    confidence_threshold_high: float = 0.80
    
    # Entropy-based thresholds
    entropy_threshold: float = 0.6  # Lowered from 0.8 to catch more uncertain positions
    
    # Score spread thresholds (discriminability)
    spread_threshold_low: float = 0.2  # Increased from 0.15 to catch more ambiguous cases
    
    # Score variance threshold
    variance_threshold: float = 0.08  # Lowered from 0.1 to be more sensitive
    
    # Selection strategy
    selection_strategy: str = "hybrid_entropy"  # "confidence_only", "entropy_only", "hybrid_entropy"
    target_selection_rate: float = 0.10  # Target ~10% of positions for meta-model
    
    # Adaptive threshold tuning
    enable_adaptive_tuning: bool = True
    tuning_iterations: int = 3


class EnhancedUncertaintySelector:
    """
    Enhanced uncertainty selector that combines multiple uncertainty metrics
    to achieve better meta-model position selection.
    """
    
    def __init__(self, config: EnhancedUncertaintyConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
    def calculate_entropy(self, donor_score: float, acceptor_score: float, neither_score: float) -> float:
        """Calculate normalized entropy of score distribution."""
        scores = np.array([donor_score, acceptor_score, neither_score])
        
        # Normalize scores to probabilities
        total = np.sum(scores)
        if total <= 0:
            return 0.0
            
        probs = scores / total
        
        # Add small epsilon to avoid log(0)
        probs = np.maximum(probs, 1e-10)
        
        # Calculate entropy
        entropy = -np.sum(probs * np.log(probs))
        
        # Normalize by max possible entropy (log(3) for 3 categories)
        max_entropy = np.log(3)
        normalized_entropy = entropy / max_entropy
        
        return float(normalized_entropy)
    
    def calculate_score_spread(self, donor_score: float, acceptor_score: float, neither_score: float) -> float:
        """Calculate spread between highest and second highest scores."""
        scores = np.array([donor_score, acceptor_score, neither_score])
        sorted_scores = np.sort(scores)[::-1]  # Sort descending
        return float(sorted_scores[0] - sorted_scores[1])
    
    def calculate_score_variance(self, donor_score: float, acceptor_score: float, neither_score: float) -> float:
        """Calculate variance of the three scores."""
        scores = np.array([donor_score, acceptor_score, neither_score])
        return float(np.var(scores))
    
    def identify_uncertain_positions_vectorized(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Vectorized identification of uncertain positions using multiple criteria.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with donor_score, acceptor_score, neither_score columns
            
        Returns
        -------
        pd.DataFrame
            DataFrame with added uncertainty analysis columns
        """
        result_df = df.copy()
        
        # Calculate max scores (existing logic)
        result_df['max_score'] = df[['donor_score', 'acceptor_score']].max(axis=1)
        
        # Calculate entropy for each position
        entropy_scores = []
        spread_scores = []
        variance_scores = []
        
        for _, row in df.iterrows():
            entropy = self.calculate_entropy(row['donor_score'], row['acceptor_score'], row['neither_score'])
            spread = self.calculate_score_spread(row['donor_score'], row['acceptor_score'], row['neither_score'])
            variance = self.calculate_score_variance(row['donor_score'], row['acceptor_score'], row['neither_score'])
            
            entropy_scores.append(entropy)
            spread_scores.append(spread)
            variance_scores.append(variance)
        
        result_df['score_entropy'] = entropy_scores
        result_df['score_spread'] = spread_scores
        result_df['score_variance'] = variance_scores
        
        # Apply uncertainty criteria based on selection strategy
        if self.config.selection_strategy == "confidence_only":
            # Traditional confidence-based selection
            uncertain_mask = (
                (result_df['max_score'] >= self.config.confidence_threshold_low) &
                (result_df['max_score'] < self.config.confidence_threshold_high)
            )
            
        elif self.config.selection_strategy == "entropy_only":
            # Pure entropy-based selection
            uncertain_mask = (
                result_df['score_entropy'] > self.config.entropy_threshold
            )
            
        else:  # "hybrid_entropy" (default)
            # Combined approach: expand the traditional confidence zone with entropy criteria
            confidence_uncertain = (
                (result_df['max_score'] >= self.config.confidence_threshold_low) &
                (result_df['max_score'] < self.config.confidence_threshold_high)
            )
            
            entropy_uncertain = (
                result_df['score_entropy'] > self.config.entropy_threshold
            )
            
            spread_uncertain = (
                result_df['score_spread'] < self.config.spread_threshold_low
            )
            
            variance_uncertain = (
                result_df['score_variance'] > self.config.variance_threshold
            )
            
            # Combine criteria: position is uncertain if it meets ANY of these conditions
            uncertain_mask = (
                confidence_uncertain |
                entropy_uncertain |
                spread_uncertain |
                variance_uncertain
            )
        
        result_df['is_uncertain'] = uncertain_mask
        
        # Add uncertainty reasons for debugging
        uncertainty_reasons = []
        for _, row in result_df.iterrows():
            reasons = []
            
            if (row['max_score'] >= self.config.confidence_threshold_low and 
                row['max_score'] < self.config.confidence_threshold_high):
                reasons.append('confidence_zone')
                
            if row['score_entropy'] > self.config.entropy_threshold:
                reasons.append(f'high_entropy({row["score_entropy"]:.3f})')
                
            if row['score_spread'] < self.config.spread_threshold_low:
                reasons.append(f'low_spread({row["score_spread"]:.3f})')
                
            if row['score_variance'] > self.config.variance_threshold:
                reasons.append(f'high_variance({row["score_variance"]:.3f})')
            
            uncertainty_reasons.append(';'.join(reasons) if reasons else '')
        
        result_df['uncertainty_reasons'] = uncertainty_reasons
        
        return result_df
    
    def adaptive_threshold_tuning(self, df: pd.DataFrame) -> EnhancedUncertaintyConfig:
        """
        Adaptively tune thresholds to achieve target selection rate.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with base model scores
            
        Returns
        -------
        EnhancedUncertaintyConfig
            Tuned configuration
        """
        if not self.config.enable_adaptive_tuning:
            return self.config
        
        self.logger.info(f"Adaptive tuning to achieve {self.config.target_selection_rate:.1%} selection rate")
        
        # Start with current config
        tuned_config = EnhancedUncertaintyConfig(
            confidence_threshold_low=self.config.confidence_threshold_low,
            confidence_threshold_high=self.config.confidence_threshold_high,
            entropy_threshold=self.config.entropy_threshold,
            spread_threshold_low=self.config.spread_threshold_low,
            variance_threshold=self.config.variance_threshold,
            selection_strategy=self.config.selection_strategy,
            target_selection_rate=self.config.target_selection_rate,
            enable_adaptive_tuning=False  # Disable recursion
        )
        
        best_config = tuned_config
        best_rate_diff = float('inf')
        
        # Try different threshold combinations
        entropy_candidates = [0.4, 0.5, 0.6, 0.7, 0.8]
        spread_candidates = [0.1, 0.15, 0.2, 0.25, 0.3]
        confidence_high_candidates = [0.70, 0.75, 0.80, 0.85, 0.90]
        
        for entropy_thresh in entropy_candidates:
            for spread_thresh in spread_candidates:
                for conf_high_thresh in confidence_high_candidates:
                    # Create test config
                    test_config = EnhancedUncertaintyConfig(
                        confidence_threshold_low=tuned_config.confidence_threshold_low,
                        confidence_threshold_high=conf_high_thresh,
                        entropy_threshold=entropy_thresh,
                        spread_threshold_low=spread_thresh,
                        variance_threshold=tuned_config.variance_threshold,
                        selection_strategy=tuned_config.selection_strategy,
                        enable_adaptive_tuning=False
                    )
                    
                    # Test selection rate
                    test_selector = EnhancedUncertaintySelector(test_config)
                    test_result = test_selector.identify_uncertain_positions_vectorized(df)
                    selection_rate = test_result['is_uncertain'].sum() / len(test_result)
                    
                    rate_diff = abs(selection_rate - self.config.target_selection_rate)
                    
                    if rate_diff < best_rate_diff:
                        best_rate_diff = rate_diff
                        best_config = test_config
                        
                        self.logger.debug(f"Better config found: entropy={entropy_thresh}, "
                                        f"spread={spread_thresh}, conf_high={conf_high_thresh}, "
                                        f"rate={selection_rate:.3f}, diff={rate_diff:.3f}")
        
        final_selector = EnhancedUncertaintySelector(best_config)
        final_result = final_selector.identify_uncertain_positions_vectorized(df)
        final_rate = final_result['is_uncertain'].sum() / len(final_result)
        
        self.logger.info(f"Adaptive tuning complete:")
        self.logger.info(f"  Target rate: {self.config.target_selection_rate:.3f}")
        self.logger.info(f"  Achieved rate: {final_rate:.3f}")
        self.logger.info(f"  Best entropy threshold: {best_config.entropy_threshold}")
        self.logger.info(f"  Best spread threshold: {best_config.spread_threshold_low}")
        self.logger.info(f"  Best confidence high: {best_config.confidence_threshold_high}")
        
        return best_config
    
    def analyze_uncertainty_distribution(self, df: pd.DataFrame) -> Dict:
        """
        Analyze the distribution of uncertainty metrics for debugging and optimization.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with uncertainty analysis
            
        Returns
        -------
        Dict
            Analysis results
        """
        result_df = self.identify_uncertain_positions_vectorized(df)
        
        total_positions = len(result_df)
        uncertain_positions = result_df['is_uncertain'].sum()
        selection_rate = uncertain_positions / total_positions
        
        # Analyze uncertainty reasons
        all_reasons = []
        for reasons_str in result_df['uncertainty_reasons']:
            if reasons_str:
                all_reasons.extend(reasons_str.split(';'))
        
        from collections import Counter
        reason_counts = Counter(all_reasons)
        
        # Score distribution statistics
        stats = {
            'total_positions': total_positions,
            'uncertain_positions': uncertain_positions,
            'selection_rate': selection_rate,
            'target_rate': self.config.target_selection_rate,
            'rate_difference': abs(selection_rate - self.config.target_selection_rate),
            'uncertainty_reasons': dict(reason_counts),
            'score_statistics': {
                'max_score': {
                    'mean': result_df['max_score'].mean(),
                    'std': result_df['max_score'].std(),
                    'min': result_df['max_score'].min(),
                    'max': result_df['max_score'].max()
                },
                'entropy': {
                    'mean': result_df['score_entropy'].mean(),
                    'std': result_df['score_entropy'].std(),
                    'min': result_df['score_entropy'].min(),
                    'max': result_df['score_entropy'].max()
                },
                'spread': {
                    'mean': result_df['score_spread'].mean(),
                    'std': result_df['score_spread'].std(),
                    'min': result_df['score_spread'].min(),
                    'max': result_df['score_spread'].max()
                }
            },
            'thresholds_used': {
                'confidence_low': self.config.confidence_threshold_low,
                'confidence_high': self.config.confidence_threshold_high,
                'entropy': self.config.entropy_threshold,
                'spread': self.config.spread_threshold_low,
                'variance': self.config.variance_threshold
            }
        }
        
        return stats


def enhance_uncertainty_selection_for_inference(
    base_predictions_df: pd.DataFrame,
    target_selection_rate: float = 0.10,
    selection_strategy: str = "hybrid_entropy",
    enable_adaptive_tuning: bool = True
) -> Tuple[pd.DataFrame, Dict]:
    """
    Main function to enhance uncertainty selection for meta-model inference.
    
    This function replaces the simple threshold-based approach with a more
    sophisticated multi-criteria uncertainty detection system.
    
    Parameters
    ----------
    base_predictions_df : pd.DataFrame
        DataFrame with donor_score, acceptor_score, neither_score columns
    target_selection_rate : float
        Target percentage of positions to select for meta-model (default: 0.10 = 10%)
    selection_strategy : str
        Selection strategy: "confidence_only", "entropy_only", "hybrid_entropy"
    enable_adaptive_tuning : bool
        Whether to adaptively tune thresholds to achieve target rate
        
    Returns
    -------
    Tuple[pd.DataFrame, Dict]
        (DataFrame with uncertainty analysis, analysis statistics)
    """
    # Create enhanced config
    config = EnhancedUncertaintyConfig(
        target_selection_rate=target_selection_rate,
        selection_strategy=selection_strategy,
        enable_adaptive_tuning=enable_adaptive_tuning
    )
    
    # Initialize selector
    selector = EnhancedUncertaintySelector(config)
    
    # Adaptive tuning if enabled
    if enable_adaptive_tuning:
        config = selector.adaptive_threshold_tuning(base_predictions_df)
        selector = EnhancedUncertaintySelector(config)
    
    # Perform uncertainty analysis
    result_df = selector.identify_uncertain_positions_vectorized(base_predictions_df)
    
    # Generate analysis report
    analysis_stats = selector.analyze_uncertainty_distribution(base_predictions_df)
    
    logger.info(f"Enhanced uncertainty selection complete:")
    logger.info(f"  Strategy: {selection_strategy}")
    logger.info(f"  Target rate: {target_selection_rate:.1%}")
    logger.info(f"  Achieved rate: {analysis_stats['selection_rate']:.1%}")
    logger.info(f"  Selected positions: {analysis_stats['uncertain_positions']:,}/{analysis_stats['total_positions']:,}")
    
    return result_df, analysis_stats


def create_enhanced_uncertain_mask(
    base_predictions_df: pd.DataFrame,
    uncertainty_threshold_low: float = 0.02,
    uncertainty_threshold_high: float = 0.80,
    target_selection_rate: float = 0.10
) -> pd.Series:
    """
    Create enhanced uncertain mask that can be used as a drop-in replacement
    for the existing uncertainty detection logic.
    
    This function maintains compatibility with existing code while providing
    enhanced uncertainty detection.
    
    Parameters
    ----------
    base_predictions_df : pd.DataFrame
        DataFrame with base model predictions
    uncertainty_threshold_low : float
        Lower confidence threshold (for compatibility)
    uncertainty_threshold_high : float  
        Upper confidence threshold (for compatibility)
    target_selection_rate : float
        Target selection rate for meta-model application
        
    Returns
    -------
    pd.Series
        Boolean mask indicating uncertain positions
    """
    result_df, _ = enhance_uncertainty_selection_for_inference(
        base_predictions_df,
        target_selection_rate=target_selection_rate,
        selection_strategy="hybrid_entropy",
        enable_adaptive_tuning=True
    )
    
    return result_df['is_uncertain']


if __name__ == "__main__":
    # Demo/test the enhanced uncertainty selection
    import logging
    
    logging.basicConfig(level=logging.INFO)
    
    # Create mock data for testing
    np.random.seed(42)
    n_positions = 1000
    
    # Generate realistic base model scores
    mock_data = {
        'gene_id': ['ENSG00000123456'] * n_positions,
        'position': range(1, n_positions + 1),
        'donor_score': np.random.beta(0.5, 2, n_positions),  # Skewed towards low scores
        'acceptor_score': np.random.beta(0.5, 2, n_positions),
        'neither_score': np.random.beta(2, 0.5, n_positions)  # Skewed towards high scores
    }
    
    # Normalize scores to sum to 1 for each position (more realistic)
    for i in range(n_positions):
        total = mock_data['donor_score'][i] + mock_data['acceptor_score'][i] + mock_data['neither_score'][i]
        mock_data['donor_score'][i] /= total
        mock_data['acceptor_score'][i] /= total
        mock_data['neither_score'][i] /= total
    
    test_df = pd.DataFrame(mock_data)
    
    print("🧪 Testing Enhanced Uncertainty Selection")
    print("=" * 50)
    
    # Test different strategies
    strategies = ["confidence_only", "entropy_only", "hybrid_entropy"]
    
    for strategy in strategies:
        print(f"\n📊 Testing strategy: {strategy}")
        result_df, stats = enhance_uncertainty_selection_for_inference(
            test_df,
            target_selection_rate=0.10,
            selection_strategy=strategy,
            enable_adaptive_tuning=True
        )
        
        print(f"   Selection rate: {stats['selection_rate']:.1%}")
        print(f"   Uncertainty reasons: {stats['uncertainty_reasons']}")
    
    print("\n✅ Enhanced uncertainty selection testing complete!")

