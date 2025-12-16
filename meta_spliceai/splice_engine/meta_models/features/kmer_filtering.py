"""
K-mer filtering strategies for splice site prediction.

This module provides domain-specific filtering approaches for k-mer features
that are more appropriate than conventional variance/correlation filtering
for genomic sequence data and splice site prediction tasks.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union, Set
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectKBest, f_classif
import logging

from meta_spliceai.splice_engine.utils_kmer import is_kmer_feature as _is_kmer

logger = logging.getLogger(__name__)

# Known splice site consensus motifs
DONOR_MOTIFS = {
    'GT': 'Donor site consensus (GU in RNA)',
    'GC': 'Alternative donor site',
    'AT': 'Rare donor variant'
}

ACCEPTOR_MOTIFS = {
    'AG': 'Acceptor site consensus',
    'CG': 'Alternative acceptor site'
}

# Position-specific importance for 6-mers
SPLICE_POSITIONS = {
    'donor': {
        'critical': [0, 1],  # GT at positions 1-2
        'important': [3, 4],  # GC at positions 4-5
        'context': [2, 5]     # Flanking regions
    },
    'acceptor': {
        'critical': [3, 4],  # AG at positions 4-5
        'context': [0, 1, 2, 5]  # Flanking regions
    }
}


class KmerFilteringStrategy:
    """Base class for k-mer filtering strategies."""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
    
    def filter(self, X: np.ndarray, y: np.ndarray, kmer_features: List[str], **kwargs) -> List[str]:
        """Filter k-mer features based on the strategy."""
        raise NotImplementedError


class MotifBasedFilter(KmerFilteringStrategy):
    """
    Filter k-mers based on known splice site motifs.
    
    This approach preserves biologically relevant features that are known
    to be important for splice site recognition, even if they have low variance.
    """
    
    def __init__(self, splice_type: str = "both", include_context: bool = True):
        super().__init__(
            name="motif_based",
            description=f"Filter based on {splice_type} splice site motifs"
        )
        self.splice_type = splice_type
        self.include_context = include_context
    
    def filter(self, X: np.ndarray, y: np.ndarray, kmer_features: List[str], **kwargs) -> List[str]:
        """Filter k-mers based on splice site motifs."""
        selected_features = []
        
        for kmer in kmer_features:
            if not _is_kmer(kmer):
                continue
                
            if self._is_splice_relevant_kmer(kmer):
                selected_features.append(kmer)
        
        logger.info(f"Motif-based filtering: {len(selected_features)}/{len(kmer_features)} k-mers selected")
        return selected_features
    
    def _is_splice_relevant_kmer(self, kmer: str) -> bool:
        """Check if k-mer contains splice site motifs."""
        if not kmer.startswith(('6mer_', '4mer_')):
            return False
        
        # Extract sequence from k-mer name
        if kmer.startswith('6mer_'):
            seq = kmer[6:]
            k = 6
        elif kmer.startswith('4mer_'):
            seq = kmer[5:]
            k = 4
        else:
            return False
        
        # Check for donor motifs
        if self.splice_type in ['donor', 'both']:
            if self._contains_donor_motif(seq, k):
                return True
        
        # Check for acceptor motifs
        if self.splice_type in ['acceptor', 'both']:
            if self._contains_acceptor_motif(seq, k):
                return True
        
        return False
    
    def _contains_donor_motif(self, seq: str, k: int) -> bool:
        """Check if sequence contains donor site motifs."""
        if k == 6:
            # Check for GT at positions 1-2 or GC at positions 4-5
            if seq[0:2] in DONOR_MOTIFS or seq[3:5] in DONOR_MOTIFS:
                return True
        elif k == 4:
            # Check for GT or GC anywhere in 4-mer
            for i in range(len(seq) - 1):
                if seq[i:i+2] in DONOR_MOTIFS:
                    return True
        return False
    
    def _contains_acceptor_motif(self, seq: str, k: int) -> bool:
        """Check if sequence contains acceptor site motifs."""
        if k == 6:
            # Check for AG at positions 4-5
            if seq[3:5] in ACCEPTOR_MOTIFS:
                return True
        elif k == 4:
            # Check for AG anywhere in 4-mer
            for i in range(len(seq) - 1):
                if seq[i:i+2] in ACCEPTOR_MOTIFS:
                    return True
        return False


class MutualInformationFilter(KmerFilteringStrategy):
    """
    Filter k-mers based on mutual information with splice site labels.
    
    This approach is better than correlation because it:
    - Captures non-linear relationships
    - Handles sparse features well (many k-mers are rare)
    - Preserves informative rare motifs
    """
    
    def __init__(self, threshold: float = 0.01, top_k: Optional[int] = None):
        super().__init__(
            name="mutual_information",
            description=f"Filter based on MI threshold {threshold}"
        )
        self.threshold = threshold
        self.top_k = top_k
    
    def filter(self, X: np.ndarray, y: np.ndarray, kmer_features: List[str], **kwargs) -> List[str]:
        """Filter k-mers based on mutual information."""
        # Get k-mer columns
        kmer_indices = [i for i, name in enumerate(kmer_features) if _is_kmer(name)]
        if not kmer_indices:
            return []
        
        X_kmer = X[:, kmer_indices]
        kmer_names = [kmer_features[i] for i in kmer_indices]
        
        # Calculate mutual information
        mi_scores = mutual_info_classif(X_kmer, y, random_state=42)
        
        # Create feature-score pairs
        feature_scores = list(zip(kmer_names, mi_scores))
        
        # Sort by MI score (descending)
        feature_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Apply filtering
        if self.top_k is not None:
            selected_features = [feature for feature, _ in feature_scores[:self.top_k]]
        else:
            selected_features = [
                feature for feature, score in feature_scores 
                if score >= self.threshold
            ]
        
        logger.info(f"MI filtering: {len(selected_features)}/{len(kmer_names)} k-mers selected")
        return selected_features


class SparsityAwareFilter(KmerFilteringStrategy):
    """
    Filter k-mers based on their sparsity patterns.
    
    For splice sites, we want:
    - Not too rare (informative when present)
    - Not too common (discriminative)
    - Preserve motifs that are rare but biologically important
    """
    
    def __init__(self, min_occurrence_rate: float = 0.001, max_occurrence_rate: float = 0.95):
        super().__init__(
            name="sparsity_aware",
            description=f"Filter based on occurrence rate [{min_occurrence_rate}, {max_occurrence_rate}]"
        )
        self.min_occurrence_rate = min_occurrence_rate
        self.max_occurrence_rate = max_occurrence_rate
    
    def filter(self, X: np.ndarray, y: np.ndarray, kmer_features: List[str], **kwargs) -> List[str]:
        """Filter k-mers based on sparsity patterns."""
        selected_features = []
        
        for i, feature in enumerate(kmer_features):
            if not _is_kmer(feature):
                continue
                
            # Calculate occurrence rate
            occurrence_rate = np.mean(X[:, i] > 0)
            
            # Check if within acceptable range
            if self.min_occurrence_rate <= occurrence_rate <= self.max_occurrence_rate:
                selected_features.append(feature)
        
        logger.info(f"Sparsity filtering: {len(selected_features)}/{len([f for f in kmer_features if _is_kmer(f)])} k-mers selected")
        return selected_features


class VarianceFilter(KmerFilteringStrategy):
    """
    Traditional variance-based filtering with domain-specific adjustments.
    
    This is a fallback to conventional methods but with adjustments
    for the sparse nature of k-mer features.
    """
    
    def __init__(self, threshold: float = 0.001):
        super().__init__(
            name="variance_based",
            description=f"Filter based on variance threshold {threshold}"
        )
        self.threshold = threshold
    
    def filter(self, X: np.ndarray, y: np.ndarray, kmer_features: List[str], **kwargs) -> List[str]:
        """Filter k-mers based on variance."""
        selected_features = []
        
        for i, feature in enumerate(kmer_features):
            if not _is_kmer(feature):
                continue
                
            # Calculate variance
            variance = np.var(X[:, i])
            
            if variance >= self.threshold:
                selected_features.append(feature)
        
        logger.info(f"Variance filtering: {len(selected_features)}/{len([f for f in kmer_features if _is_kmer(f)])} k-mers selected")
        return selected_features


class EnsembleFilter(KmerFilteringStrategy):
    """
    Combine multiple filtering approaches for robust feature selection.
    
    This approach uses multiple strategies and takes the union or intersection
    of their results, providing more robust feature selection.
    """
    
    def __init__(self, strategies: List[KmerFilteringStrategy], 
                 combination_method: str = "union"):
        super().__init__(
            name="ensemble",
            description=f"Ensemble of {len(strategies)} strategies using {combination_method}"
        )
        self.strategies = strategies
        self.combination_method = combination_method
    
    def filter(self, X: np.ndarray, y: np.ndarray, kmer_features: List[str], **kwargs) -> List[str]:
        """Apply ensemble filtering."""
        selected_sets = []
        
        for strategy in self.strategies:
            try:
                selected = strategy.filter(X, y, kmer_features, **kwargs)
                selected_sets.append(set(selected))
                logger.info(f"Strategy '{strategy.name}': {len(selected)} features selected")
            except Exception as e:
                logger.warning(f"Strategy '{strategy.name}' failed: {e}")
                selected_sets.append(set())
        
        # Combine results
        if self.combination_method == "union":
            final_selected = set.union(*selected_sets)
        elif self.combination_method == "intersection":
            final_selected = set.intersection(*selected_sets)
        else:
            raise ValueError(f"Unknown combination method: {self.combination_method}")
        
        logger.info(f"Ensemble filtering: {len(final_selected)}/{len([f for f in kmer_features if _is_kmer(f)])} k-mers selected")
        return list(final_selected)


class KmerFilterManager:
    """
    Manager class for applying k-mer filtering strategies.
    
    This class provides a unified interface for applying different
    filtering strategies to k-mer features in datasets.
    """
    
    def __init__(self):
        self.strategies = {}
        self._register_default_strategies()
    
    def _register_default_strategies(self):
        """Register default filtering strategies."""
        self.strategies.update({
            'motif': MotifBasedFilter(),
            'mi': MutualInformationFilter(),
            'sparsity': SparsityAwareFilter(),
            'variance': VarianceFilter(),
            'ensemble': EnsembleFilter([
                MotifBasedFilter(),
                MutualInformationFilter(threshold=0.005),
                SparsityAwareFilter()
            ])
        })
    
    def register_strategy(self, name: str, strategy: KmerFilteringStrategy):
        """Register a custom filtering strategy."""
        self.strategies[name] = strategy
    
    def get_strategy(self, name: str) -> KmerFilteringStrategy:
        """Get a filtering strategy by name."""
        if name not in self.strategies:
            raise ValueError(f"Unknown strategy '{name}'. Available: {list(self.strategies.keys())}")
        return self.strategies[name]
    
    def filter_kmers(self, X: np.ndarray, y: np.ndarray, 
                    feature_names: List[str], strategy_name: str, **kwargs) -> Tuple[np.ndarray, List[str]]:
        """
        Apply k-mer filtering to a feature matrix.
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Target labels
        feature_names : List[str]
            List of feature names
        strategy_name : str
            Name of the filtering strategy to use
        **kwargs
            Additional arguments for the filtering strategy
            
        Returns:
        --------
        Tuple[np.ndarray, List[str]]
            Filtered feature matrix and feature names
        """
        # Separate k-mer and non-k-mer features
        kmer_indices = [i for i, name in enumerate(feature_names) if _is_kmer(name)]
        non_kmer_indices = [i for i, name in enumerate(feature_names) if not _is_kmer(name)]
        
        if not kmer_indices:
            logger.warning("No k-mer features found in dataset")
            return X, feature_names
        
        # Get k-mer features
        kmer_features = [feature_names[i] for i in kmer_indices]
        X_kmer = X[:, kmer_indices]
        
        # Apply filtering strategy
        strategy = self.get_strategy(strategy_name)
        selected_kmers = strategy.filter(X_kmer, y, kmer_features, **kwargs)
        
        # Get indices of selected k-mers
        selected_kmer_indices = [
            kmer_indices[i] for i, name in enumerate(kmer_features) 
            if name in selected_kmers
        ]
        
        # Combine non-k-mer and selected k-mer indices
        final_indices = non_kmer_indices + selected_kmer_indices
        final_features = [feature_names[i] for i in final_indices]
        
        logger.info(f"K-mer filtering complete: {len(final_features)}/{len(feature_names)} features retained")
        
        return X[:, final_indices], final_features
    
    def get_filtering_stats(self, X: np.ndarray, feature_names: List[str]) -> Dict[str, any]:
        """Get statistics about k-mer features in the dataset."""
        kmer_features = [name for name in feature_names if _is_kmer(name)]
        non_kmer_features = [name for name in feature_names if not _is_kmer(name)]
        
        stats = {
            'total_features': len(feature_names),
            'kmer_features': len(kmer_features),
            'non_kmer_features': len(non_kmer_features),
            'kmer_ratio': len(kmer_features) / len(feature_names) if feature_names else 0
        }
        
        if kmer_features:
            # Analyze k-mer sizes
            kmer_sizes = {}
            for kmer in kmer_features:
                if kmer.startswith('6mer_'):
                    kmer_sizes['6mer'] = kmer_sizes.get('6mer', 0) + 1
                elif kmer.startswith('4mer_'):
                    kmer_sizes['4mer'] = kmer_sizes.get('4mer', 0) + 1
            
            stats['kmer_size_distribution'] = kmer_sizes
        
        return stats


# Convenience functions for easy integration
def create_filter_manager() -> KmerFilterManager:
    """Create a KmerFilterManager with default strategies."""
    return KmerFilterManager()


def filter_dataset_kmers(df: pd.DataFrame, 
                        target_col: str,
                        strategy_name: str = 'ensemble',
                        **kwargs) -> pd.DataFrame:
    """
    Filter k-mers in a pandas DataFrame.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame with features
    target_col : str
        Name of the target column
    strategy_name : str
        Name of the filtering strategy
    **kwargs
        Additional arguments for the filtering strategy
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with filtered k-mer features
    """
    manager = create_filter_manager()
    
    # Prepare data
    feature_names = [col for col in df.columns if col != target_col]
    X = df[feature_names].values
    y = df[target_col].values
    
    # Apply filtering
    X_filtered, filtered_features = manager.filter_kmers(
        X, y, feature_names, strategy_name, **kwargs
    )
    
    # Create filtered DataFrame
    filtered_df = pd.DataFrame(X_filtered, columns=filtered_features)
    filtered_df[target_col] = y
    
    return filtered_df


def get_kmer_filtering_stats(df: pd.DataFrame) -> Dict[str, any]:
    """Get statistics about k-mer features in a DataFrame."""
    manager = create_filter_manager()
    feature_names = df.columns.tolist()
    X = df.values
    return manager.get_filtering_stats(X, feature_names) 