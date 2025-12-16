"""
Streaming statistics utilities for dataset profiling.

This module contains the StreamingStats class for memory-efficient statistical
calculations and feature categorization functions.
"""
import random
import re
from typing import List, Dict, Any, Optional, Union
from collections import Counter
import numpy as np
import pandas as pd


class StreamingStats:
    """Compute statistics incrementally without storing all values."""
    
    def __init__(self, track_values_for_median: bool = False):
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0  # Sum of squares of deviations from mean
        self.min_val = float('inf')
        self.max_val = float('-inf')
        
        # For median calculation (reservoir sampling)
        self.track_values_for_median = track_values_for_median
        self.values_sample = []
        self.sample_size = 10000  # Keep a sample for median estimation
        
    def update(self, values: Union[List[float], np.ndarray, pd.Series]):
        """Update statistics with new values."""
        if isinstance(values, (pd.Series, np.ndarray)):
            values = values.tolist()
        elif not isinstance(values, list):
            values = [values]
        
        for value in values:
            if pd.isna(value):
                continue
                
            self.count += 1
            
            # Update min/max
            self.min_val = min(self.min_val, value)
            self.max_val = max(self.max_val, value)
            
            # Update mean and variance using Welford's algorithm
            delta = value - self.mean
            self.mean += delta / self.count
            delta2 = value - self.mean
            self.m2 += delta * delta2
            
            # Reservoir sampling for median estimation
            if self.track_values_for_median:
                if len(self.values_sample) < self.sample_size:
                    self.values_sample.append(value)
                else:
                    # Replace random element with probability sample_size/count
                    if random.random() < self.sample_size / self.count:
                        idx = random.randint(0, self.sample_size - 1)
                        self.values_sample[idx] = value
    
    def get_variance(self) -> float:
        """Get sample variance."""
        if self.count < 2:
            return 0.0
        return self.m2 / (self.count - 1)
    
    def get_std(self) -> float:
        """Get sample standard deviation."""
        return np.sqrt(self.get_variance())
    
    def get_median(self) -> Optional[float]:
        """Get estimated median (only if tracking values)."""
        if not self.track_values_for_median or not self.values_sample:
            return None
        return np.median(self.values_sample)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get all computed statistics."""
        stats = {
            'count': self.count,
            'mean': self.mean,
            'std': self.get_std(),
            'min': self.min_val if self.count > 0 else None,
            'max': self.max_val if self.count > 0 else None,
            'variance': self.get_variance()
        }
        
        if self.track_values_for_median:
            stats['median'] = self.get_median()
            
        return stats


def find_splice_column(df: pd.DataFrame) -> Optional[str]:
    """Find the splice type column in the dataframe."""
    # Exact matches first (highest priority)
    exact_matches = ['splice_type', 'label', 'target', 'y', 'class']
    for col in df.columns:
        if col.lower() in [match.lower() for match in exact_matches]:
            return col
    
    # Partial matches (lower priority)
    potential_cols = ['splice_type', 'label', 'target', 'y', 'class']
    for col in df.columns:
        col_lower = col.lower()
        if any(potential in col_lower for potential in potential_cols):
            return col
    
    return None


def find_gene_column(df: pd.DataFrame) -> Optional[str]:
    """Find the gene ID column in the dataframe."""
    for col in df.columns:
        col_lower = col.lower()
        if 'gene_id' in col_lower or 'gene' in col_lower:
            return col
    
    return None


def categorize_features(columns: List[str]) -> Dict[str, Any]:
    """Categorize dataset features into different types."""
    feature_categories = {
        'splice_ai_scores': [],
        'probability_context_features': [],
        'kmer_features': [],
        'positional_features': [],
        'structural_features': [],
        'sequence_context': [],
        'genomic_annotations': [],
        'identifiers': [],
        'other': []
    }
    
    # Probability and context-derived features from enhanced workflow
    probability_context_patterns = [
        # Context-agnostic features
        'context_neighbor_mean', 'context_asymmetry', 'context_max',
        'context_score_m1', 'context_score_m2', 'context_score_p1', 'context_score_p2',
        
        # Donor-specific features
        'donor_diff_m1', 'donor_diff_m2', 'donor_diff_p1', 'donor_diff_p2',
        'donor_surge_ratio', 'donor_is_local_peak', 'donor_weighted_context',
        'donor_peak_height_ratio', 'donor_second_derivative', 'donor_signal_strength',
        'donor_context_diff_ratio',
        
        # Acceptor-specific features
        'acceptor_diff_m1', 'acceptor_diff_m2', 'acceptor_diff_p1', 'acceptor_diff_p2',
        'acceptor_surge_ratio', 'acceptor_is_local_peak', 'acceptor_weighted_context',
        'acceptor_peak_height_ratio', 'acceptor_second_derivative', 'acceptor_signal_strength',
        'acceptor_context_diff_ratio',
        
        # Cross-type features
        'donor_acceptor_peak_ratio', 'donor_acceptor_diff', 'donor_acceptor_logodds',
        'type_signal_difference', 'score_difference_ratio', 'signal_strength_ratio',
        
        # Probability-derived features
        'probability_entropy', 'relative_donor_probability', 'splice_probability',
        'splice_neither_diff', 'splice_neither_logodds',
        
        # Max score feature (derived from raw scores)
        'score',
        
        # General probability/context patterns (fallback)
        'prob', 'probability', 'context', 'signal', 'strength', 'ratio', 'diff',
        'peak', 'surge', 'weighted', 'derivative', 'asymmetry', 'neighbor', 'entropy',
        'logodds', 'splice_neither', 'relative_donor'
    ]
    
    # Categorize each column
    for col in columns:
        col_lower = col.lower()
        
        # SpliceAI raw scores
        if any(score in col_lower for score in ['donor_score', 'acceptor_score', 'neither_score']):
            feature_categories['splice_ai_scores'].append(col)
        
        # Probability and context-derived features
        elif any(pattern in col_lower for pattern in probability_context_patterns):
            feature_categories['probability_context_features'].append(col)
        
        # K-mer features (regex pattern for k-mer columns like 6mer_GGATCN or pure nucleotide sequences)
        elif (re.match(r'^[ATCG]+$', col) or 
              re.match(r'^\d+mer_[ATCGN]+$', col) or 
              any(kmer in col_lower for kmer in ['kmer', '_mer'])):
            feature_categories['kmer_features'].append(col)
        
        # Positional features
        elif any(pos in col_lower for pos in ['position', 'distance', 'offset', 'coord', 'start', 'end']):
            feature_categories['positional_features'].append(col)
        
        # Structural features
        elif any(struct in col_lower for struct in ['structure', 'fold', 'secondary', 'stem', 'loop']):
            feature_categories['structural_features'].append(col)
        
        # Sequence context
        elif any(seq in col_lower for seq in ['upstream', 'downstream', 'flanking', 'window', 'context']):
            feature_categories['sequence_context'].append(col)
        
        # Genomic annotations
        elif any(annot in col_lower for annot in ['exon', 'intron', 'utr', 'cds', 'transcript', 'annotation']):
            feature_categories['genomic_annotations'].append(col)
        
        # Identifiers
        elif any(id_col in col_lower for id_col in ['id', 'name', 'symbol', 'ensembl', 'gene', 'transcript']):
            feature_categories['identifiers'].append(col)
        
        # Everything else
        else:
            feature_categories['other'].append(col)
    
    # Calculate feature counts
    feature_counts = {category: len(features) for category, features in feature_categories.items()}
    
    return {
        'feature_categories': feature_categories,
        'feature_counts': feature_counts,
        'total_features': sum(feature_counts.values())
    }
