"""
Uncertainty Analyzer for Inference Mode

This module implements sophisticated uncertainty detection based ONLY on base model scores,
without access to ground truth labels (as would be the case in real inference scenarios).

Key principle: Uncertainty must be inferred from patterns in base model predictions:
- Low confidence (max score below threshold)
- High entropy (scores spread across multiple types)
- Low discriminability (small difference between top scores)
- Contextual patterns (unusual scores relative to neighbors)
"""

import logging
import numpy as np
import polars as pl
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class UncertaintyMetrics:
    """Container for uncertainty metrics calculated from base model scores."""
    max_score: float           # Highest score among the three types
    score_entropy: float       # Entropy of score distribution
    score_spread: float        # Difference between max and second max
    score_variance: float      # Variance of the three scores
    predicted_type: str        # Predicted splice type (donor/acceptor/neither)
    confidence_level: str      # high/medium/low confidence
    is_uncertain: bool         # Overall uncertainty flag
    uncertainty_reasons: List[str]  # Reasons for uncertainty classification


class UncertaintyAnalyzer:
    """
    Analyzes uncertainty in base model predictions for selective meta-model application.
    
    Uses multiple complementary metrics to identify positions where meta-model
    recalibration is most likely to improve predictions.
    """
    
    def __init__(self, 
                 confidence_threshold: float = 0.5,
                 entropy_threshold: float = 0.8,
                 spread_threshold: float = 0.15,
                 variance_threshold: float = 0.1):
        """
        Initialize uncertainty analyzer with configurable thresholds.
        
        Args:
            confidence_threshold: Minimum max score for confident predictions
            entropy_threshold: Maximum entropy for confident predictions
            spread_threshold: Minimum spread between top scores for confident predictions
            variance_threshold: Maximum variance for confident predictions
        """
        self.confidence_threshold = confidence_threshold
        self.entropy_threshold = entropy_threshold  
        self.spread_threshold = spread_threshold
        self.variance_threshold = variance_threshold
        
    def calculate_uncertainty_metrics(self, 
                                    donor_score: float,
                                    acceptor_score: float, 
                                    neither_score: float) -> UncertaintyMetrics:
        """
        Calculate comprehensive uncertainty metrics for a single position.
        
        Args:
            donor_score: Base model donor score
            acceptor_score: Base model acceptor score
            neither_score: Base model neither score
            
        Returns:
            UncertaintyMetrics object with all calculated metrics
        """
        scores = np.array([donor_score, acceptor_score, neither_score])
        score_names = ['donor', 'acceptor', 'neither']
        
        # Basic metrics
        max_score = float(np.max(scores))
        max_idx = np.argmax(scores)
        predicted_type = score_names[max_idx]
        
        # Score distribution metrics
        score_entropy = self._calculate_entropy(scores)
        score_spread = self._calculate_spread(scores)
        score_variance = float(np.var(scores))
        
        # Uncertainty classification
        uncertainty_reasons = []
        
        # Check various uncertainty conditions
        if max_score < self.confidence_threshold:
            uncertainty_reasons.append(f'low_confidence_score ({max_score:.3f})')
            
        if score_entropy > self.entropy_threshold:
            uncertainty_reasons.append(f'high_entropy ({score_entropy:.3f})')
            
        if score_spread < self.spread_threshold:
            uncertainty_reasons.append(f'low_discriminability ({score_spread:.3f})')
            
        if score_variance > self.variance_threshold:
            uncertainty_reasons.append(f'high_variance ({score_variance:.3f})')
        
        # Overall uncertainty flag
        is_uncertain = len(uncertainty_reasons) > 0
        
        # Confidence level categorization
        if max_score >= 0.8 and score_spread >= 0.3:
            confidence_level = 'high'
        elif max_score >= 0.5 and score_spread >= 0.1:
            confidence_level = 'medium'
        else:
            confidence_level = 'low'
            
        return UncertaintyMetrics(
            max_score=max_score,
            score_entropy=score_entropy,
            score_spread=score_spread,
            score_variance=score_variance,
            predicted_type=predicted_type,
            confidence_level=confidence_level,
            is_uncertain=is_uncertain,
            uncertainty_reasons=uncertainty_reasons
        )
    
    def _calculate_entropy(self, scores: np.ndarray) -> float:
        """Calculate normalized entropy of score distribution."""
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
    
    def _calculate_spread(self, scores: np.ndarray) -> float:
        """Calculate spread between highest and second highest scores."""
        sorted_scores = np.sort(scores)[::-1]  # Sort descending
        return float(sorted_scores[0] - sorted_scores[1])
    
    def analyze_dataframe(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Analyze uncertainty for all positions in a DataFrame.
        
        Args:
            df: DataFrame with donor_score, acceptor_score, neither_score columns
            
        Returns:
            DataFrame with added uncertainty analysis columns
        """
        logger.info(f"Analyzing uncertainty for {df.height} positions")
        
        # Calculate metrics for each row
        metrics_list = []
        for row in df.iter_rows(named=True):
            metrics = self.calculate_uncertainty_metrics(
                donor_score=row['donor_score'],
                acceptor_score=row['acceptor_score'],
                neither_score=row['neither_score']
            )
            metrics_list.append(metrics)
        
        # Convert metrics to DataFrame columns
        uncertainty_df = pl.DataFrame({
            'max_score': [m.max_score for m in metrics_list],
            'score_entropy': [m.score_entropy for m in metrics_list],
            'score_spread': [m.score_spread for m in metrics_list],
            'score_variance': [m.score_variance for m in metrics_list],
            'predicted_splice_type': [m.predicted_type for m in metrics_list],
            'confidence_level': [m.confidence_level for m in metrics_list],
            'is_uncertain': [m.is_uncertain for m in metrics_list],
            'uncertainty_reasons': [';'.join(m.uncertainty_reasons) for m in metrics_list]
        })
        
        # Combine with original DataFrame
        result_df = pl.concat([df, uncertainty_df], how='horizontal')
        
        # Log summary statistics
        uncertain_count = result_df.filter(pl.col('is_uncertain')).height
        uncertainty_rate = uncertain_count / df.height if df.height > 0 else 0
        
        confidence_dist = result_df.group_by('confidence_level').agg(pl.len().alias('count'))
        
        logger.info(f"Uncertainty analysis complete:")
        logger.info(f"  Uncertain positions: {uncertain_count}/{df.height} ({uncertainty_rate:.1%})")
        logger.info(f"  Confidence distribution: {confidence_dist.to_dicts()}")
        
        return result_df
    
    def get_meta_model_candidates(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Get positions that are candidates for meta-model application.
        
        Args:
            df: DataFrame with uncertainty analysis
            
        Returns:
            DataFrame filtered to uncertain positions only
        """
        candidates = df.filter(pl.col('is_uncertain'))
        
        logger.info(f"Selected {candidates.height} positions for meta-model application")
        
        return candidates
    
    def analyze_contextual_uncertainty(self, df: pl.DataFrame, window_size: int = 5) -> pl.DataFrame:
        """
        Analyze uncertainty in the context of neighboring positions.
        
        Adds contextual features that may help identify positions where
        the local genomic context creates uncertainty.
        
        Args:
            df: DataFrame with base scores, sorted by position
            window_size: Size of neighborhood window for context analysis
            
        Returns:
            DataFrame with added contextual uncertainty features
        """
        logger.info(f"Analyzing contextual uncertainty with window size {window_size}")
        
        # Ensure DataFrame is sorted by position
        df_sorted = df.sort(['gene_id', 'position'])
        
        # Calculate rolling statistics for context analysis
        contextual_df = df_sorted.with_columns([
            # Rolling mean of max scores (local confidence level)
            pl.col('max_score').rolling_mean(window_size, center=True).alias('local_confidence_mean'),
            
            # Rolling std of max scores (local confidence stability)
            pl.col('max_score').rolling_std(window_size, center=True).alias('local_confidence_std'),
            
            # Rolling mean of entropy (local uncertainty level)
            pl.col('score_entropy').rolling_mean(window_size, center=True).alias('local_entropy_mean'),
            
            # Rolling std of entropy (local uncertainty variability)
            pl.col('score_entropy').rolling_std(window_size, center=True).alias('local_entropy_std')
        ])
        
        # Add contextual uncertainty flags
        contextual_enhanced = contextual_df.with_columns([
            # Position is in a locally uncertain region
            (
                (pl.col('local_confidence_mean') < self.confidence_threshold) |
                (pl.col('local_entropy_mean') > self.entropy_threshold)
            ).alias('in_uncertain_region'),
            
            # Position shows high local variability
            (
                (pl.col('local_confidence_std') > 0.2) |
                (pl.col('local_entropy_std') > 0.3)
            ).alias('high_local_variability'),
            
            # Combined contextual uncertainty
            (
                pl.col('is_uncertain') |
                (pl.col('local_confidence_mean') < self.confidence_threshold) |
                (pl.col('local_entropy_mean') > self.entropy_threshold)
            ).alias('contextually_uncertain')
        ])
        
        # Update uncertainty reasons with contextual information
        def update_uncertainty_reasons(row):
            reasons = row['uncertainty_reasons'].split(';') if row['uncertainty_reasons'] else []
            
            if row['in_uncertain_region'] and 'contextual_uncertainty' not in reasons:
                reasons.append('contextual_uncertainty')
                
            if row['high_local_variability'] and 'local_variability' not in reasons:
                reasons.append('local_variability')
                
            return ';'.join(reasons)
        
        final_df = contextual_enhanced.with_columns([
            pl.struct(['uncertainty_reasons', 'in_uncertain_region', 'high_local_variability'])
            .map_elements(update_uncertainty_reasons)
            .alias('enhanced_uncertainty_reasons')
        ])
        
        # Log contextual analysis summary
        contextual_uncertain = final_df.filter(pl.col('contextually_uncertain')).height
        contextual_rate = contextual_uncertain / df.height if df.height > 0 else 0
        
        logger.info(f"Contextual uncertainty analysis:")
        logger.info(f"  Contextually uncertain: {contextual_uncertain}/{df.height} ({contextual_rate:.1%})")
        
        return final_df
    
    def generate_uncertainty_report(self, df: pl.DataFrame) -> Dict:
        """
        Generate comprehensive uncertainty analysis report.
        
        Args:
            df: DataFrame with uncertainty analysis
            
        Returns:
            Dictionary with detailed uncertainty statistics
        """
        total_positions = df.height
        uncertain_positions = df.filter(pl.col('is_uncertain')).height
        
        # Confidence distribution
        confidence_dist = df.group_by('confidence_level').agg(pl.len().alias('count')).to_dicts()
        confidence_summary = {item['confidence_level']: item['count'] for item in confidence_dist}
        
        # Uncertainty reasons analysis
        all_reasons = []
        for reasons_str in df['uncertainty_reasons'].to_list():
            if reasons_str:
                all_reasons.extend(reasons_str.split(';'))
        
        from collections import Counter
        reason_counts = Counter(all_reasons)
        
        # Score distribution analysis
        score_stats = {
            'max_score': {
                'mean': float(df['max_score'].mean()),
                'std': float(df['max_score'].std()),
                'min': float(df['max_score'].min()),
                'max': float(df['max_score'].max())
            },
            'score_entropy': {
                'mean': float(df['score_entropy'].mean()),
                'std': float(df['score_entropy'].std()),
                'min': float(df['score_entropy'].min()),
                'max': float(df['score_entropy'].max())
            },
            'score_spread': {
                'mean': float(df['score_spread'].mean()),
                'std': float(df['score_spread'].std()),
                'min': float(df['score_spread'].min()),
                'max': float(df['score_spread'].max())
            }
        }
        
        return {
            'total_positions': total_positions,
            'uncertain_positions': uncertain_positions,
            'uncertainty_rate': uncertain_positions / total_positions if total_positions > 0 else 0,
            'confidence_distribution': confidence_summary,
            'uncertainty_reasons': dict(reason_counts),
            'score_statistics': score_stats,
            'thresholds_used': {
                'confidence_threshold': self.confidence_threshold,
                'entropy_threshold': self.entropy_threshold,
                'spread_threshold': self.spread_threshold,
                'variance_threshold': self.variance_threshold
            }
        }


def analyze_prediction_uncertainty(df: pl.DataFrame,
                                 confidence_threshold: float = 0.5,
                                 entropy_threshold: float = 0.8,
                                 spread_threshold: float = 0.15,
                                 variance_threshold: float = 0.1,
                                 include_contextual: bool = True) -> Tuple[pl.DataFrame, Dict]:
    """
    Main function to analyze prediction uncertainty in a DataFrame.
    
    Args:
        df: DataFrame with donor_score, acceptor_score, neither_score columns
        confidence_threshold: Minimum max score for confident predictions
        entropy_threshold: Maximum entropy for confident predictions
        spread_threshold: Minimum spread between top scores for confident predictions
        variance_threshold: Maximum variance for confident predictions
        include_contextual: Whether to include contextual uncertainty analysis
        
    Returns:
        Tuple of (enhanced DataFrame, uncertainty report)
    """
    analyzer = UncertaintyAnalyzer(
        confidence_threshold=confidence_threshold,
        entropy_threshold=entropy_threshold,
        spread_threshold=spread_threshold,
        variance_threshold=variance_threshold
    )
    
    # Basic uncertainty analysis
    analyzed_df = analyzer.analyze_dataframe(df)
    
    # Add contextual analysis if requested
    if include_contextual:
        analyzed_df = analyzer.analyze_contextual_uncertainty(analyzed_df)
    
    # Generate report
    report = analyzer.generate_uncertainty_report(analyzed_df)
    
    return analyzed_df, report