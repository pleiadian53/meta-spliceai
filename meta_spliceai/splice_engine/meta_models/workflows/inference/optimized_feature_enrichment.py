"""
Optimized Feature Enrichment for Inference Workflow

This module provides a streamlined, inference-specific feature enrichment
pipeline that avoids the heavy computational overhead of the full training
feature enrichment while maintaining compatibility with meta-model requirements.

Key optimizations:
1. Pre-computed feature caching at gene level
2. Selective enrichment only for uncertain positions
3. Direct feature matrix generation bypassing coordinate system conversions
4. Minimal I/O operations using in-memory caching
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import pickle
import logging
from dataclasses import dataclass

# Set up logging
logger = logging.getLogger(__name__)

@dataclass
class InferenceFeatureConfig:
    """Configuration for inference feature enrichment"""
    use_cache: bool = True
    cache_dir: Optional[Path] = None
    verbose: bool = True
    
    # Feature groups to include
    include_probability_features: bool = True
    include_genomic_features: bool = True
    include_kmer_features: bool = True
    include_context_features: bool = True

class OptimizedInferenceEnricher:
    """
    Optimized feature enricher specifically designed for inference workflow.
    
    This enricher:
    1. Bypasses the heavy enhanced_process_predictions_with_all_scores workflow
    2. Directly generates feature matrices from base model predictions
    3. Uses cached genomic features to avoid repeated I/O
    4. Maintains compatibility with training feature manifests
    """
    
    def __init__(self, config: InferenceFeatureConfig):
        self.config = config
        self._genomic_cache = {}
        self._feature_manifest = None
        
        # Initialize Registry for build-specific path resolution
        from meta_spliceai.system.genomic_resources import Registry
        self.registry = Registry()
        
        if config.verbose:
            logger.info("🚀 Initialized OptimizedInferenceEnricher")
    
    def load_feature_manifest(self, model_path: Union[str, Path]) -> List[str]:
        """Load the feature manifest from the trained model directory"""
        model_dir = Path(model_path).parent
        manifest_path = model_dir / "feature_manifest.csv"
        
        if manifest_path.exists():
            df = pd.read_csv(manifest_path)
            features = df['feature'].tolist()
            self._feature_manifest = features
            if self.config.verbose:
                logger.info(f"📋 Loaded {len(features)} features from manifest")
            return features
        else:
            raise FileNotFoundError(f"Feature manifest not found: {manifest_path}")
    
    def _load_genomic_features_for_gene(self, gene_id: str) -> Optional[Dict]:
        """Load cached genomic features for a specific gene"""
        if gene_id in self._genomic_cache:
            return self._genomic_cache[gene_id]
        
        # Try to load from gene features file using Registry
        try:
            gene_features_path_str = self.registry.resolve("gene_features")
            if gene_features_path_str:
                gene_features_path = Path(gene_features_path_str)
                gene_df = pd.read_csv(gene_features_path, sep='\t')
                gene_row = gene_df[gene_df['gene_id'] == gene_id]
                if not gene_row.empty:
                    gene_data = gene_row.iloc[0].to_dict()
                    self._genomic_cache[gene_id] = gene_data
                    if self.config.verbose:
                        logger.info(f"✅ Loaded genomic features for {gene_id} from {gene_features_path}")
                    return gene_data
                else:
                    if self.config.verbose:
                        logger.warning(f"Gene {gene_id} not found in gene_features.tsv")
            else:
                if self.config.verbose:
                    logger.warning(f"Gene features file not found for build {self.registry.cfg.build}")
        except Exception as e:
            if self.config.verbose:
                logger.warning(f"Could not load genomic features for {gene_id}: {e}")
        
        return None
    
    def _generate_probability_features(self, predictions_df: pd.DataFrame) -> pd.DataFrame:
        """Generate probability-based features from base model predictions"""
        df = predictions_df.copy()
        
        # Basic probability features
        df['relative_donor_probability'] = df['donor_score'] / (df['donor_score'] + df['acceptor_score'] + 1e-8)
        df['splice_probability'] = df['donor_score'] + df['acceptor_score']
        df['donor_acceptor_diff'] = df['donor_score'] - df['acceptor_score']
        df['splice_neither_diff'] = df['splice_probability'] - df['neither_score']
        
        # Log-odds ratios (with smoothing)
        epsilon = 1e-8
        df['donor_acceptor_logodds'] = np.log((df['donor_score'] + epsilon) / (df['acceptor_score'] + epsilon))
        df['splice_neither_logodds'] = np.log((df['splice_probability'] + epsilon) / (df['neither_score'] + epsilon))
        
        # Probability entropy
        probs = df[['donor_score', 'acceptor_score', 'neither_score']].values
        probs = probs / (probs.sum(axis=1, keepdims=True) + 1e-8)  # Normalize
        entropy = -np.sum(probs * np.log(probs + 1e-8), axis=1)
        df['probability_entropy'] = entropy
        
        return df
    
    def _generate_context_features(self, predictions_df: pd.DataFrame) -> pd.DataFrame:
        """Generate context-aware features"""
        df = predictions_df.copy()
        
        # Sort by position to ensure proper context calculation
        df = df.sort_values('position').reset_index(drop=True)
        
        # Context neighbor features (using rolling windows)
        window_size = 5
        for score_col in ['donor_score', 'acceptor_score']:
            # Rolling mean excluding current position
            df[f'{score_col.split("_")[0]}_context_mean'] = (
                df[score_col].rolling(window=window_size, center=True).mean()
            )
        
        # Context asymmetry (difference between upstream and downstream)
        for score_col in ['donor_score', 'acceptor_score']:
            col_prefix = score_col.split('_')[0]
            upstream = df[score_col].shift(2)  # 2 positions upstream
            downstream = df[score_col].shift(-2)  # 2 positions downstream
            df[f'{col_prefix}_context_asymmetry'] = upstream - downstream
        
        # Local peak detection
        for score_col in ['donor_score', 'acceptor_score']:
            col_prefix = score_col.split('_')[0]
            scores = df[score_col].values
            
            # Simple local peak detection
            is_peak = np.zeros(len(scores), dtype=bool)
            for i in range(1, len(scores) - 1):
                if scores[i] > scores[i-1] and scores[i] > scores[i+1]:
                    is_peak[i] = True
            df[f'{col_prefix}_is_local_peak'] = is_peak.astype(int)
        
        return df
    
    def _generate_kmer_features(self, predictions_df: pd.DataFrame, k: int = 3) -> pd.DataFrame:
        """Generate k-mer features - for inference, we'll use zeros as placeholders"""
        df = predictions_df.copy()
        
        # Generate all possible k-mers
        bases = ['A', 'T', 'G', 'C']
        kmers = []
        
        def generate_kmers(length):
            if length == 1:
                return bases
            else:
                smaller_kmers = generate_kmers(length - 1)
                return [base + kmer for base in bases for kmer in smaller_kmers]
        
        kmers = generate_kmers(k)
        
        # For inference, we'll set all k-mer features to 0
        # The feature harmonization step will handle any missing k-mers
        for kmer in kmers:
            df[f'{k}mer_{kmer}'] = 0.0
        
        if self.config.verbose:
            logger.info(f"🧬 Generated {len(kmers)} {k}-mer features (initialized to 0)")
        
        return df
    
    def _generate_genomic_features(self, predictions_df: pd.DataFrame, gene_id: str) -> pd.DataFrame:
        """Generate genomic features using cached data"""
        df = predictions_df.copy()
        
        # Load genomic features for this gene
        genomic_data = self._load_genomic_features_for_gene(gene_id)
        
        if genomic_data:
            # Add genomic features as constants for all positions in this gene
            for key, value in genomic_data.items():
                if key in ['gene_start', 'gene_end', 'gene_length', 'transcript_length',
                          'tx_start', 'tx_end', 'num_exons', 'avg_exon_length',
                          'median_exon_length', 'total_exon_length', 'total_intron_length',
                          'n_splice_sites', 'num_overlaps']:
                    df[key] = value
                elif key == 'chrom':
                    df[key] = value
        else:
            # Set default values if genomic data not available
            default_genomic = {
                'gene_start': 0, 'gene_end': 10000, 'gene_length': 10000,
                'transcript_length': 10000, 'tx_start': 0, 'tx_end': 10000,
                'num_exons': 10, 'avg_exon_length': 100, 'median_exon_length': 100,
                'total_exon_length': 1000, 'total_intron_length': 9000,
                'n_splice_sites': 20, 'num_overlaps': 0, 'chrom': '1'
            }
            for key, value in default_genomic.items():
                df[key] = value
            
            if self.config.verbose:
                logger.warning(f"⚠️ Using default genomic features for {gene_id}")
        
        return df
    
    def generate_features_for_uncertain_positions(
        self,
        uncertain_positions_df: pd.DataFrame,
        gene_id: str,
        model_path: Union[str, Path]
    ) -> pd.DataFrame:
        """
        Generate complete feature matrix for uncertain positions only.
        
        This is the main entry point for inference feature generation.
        """
        if self.config.verbose:
            logger.info(f"🎯 Generating features for {len(uncertain_positions_df)} uncertain positions")
        
        # Load feature manifest to know what features we need
        if self._feature_manifest is None:
            self.load_feature_manifest(model_path)
        
        # Start with the base predictions
        feature_df = uncertain_positions_df.copy()
        
        # Generate different types of features
        if self.config.include_probability_features:
            if self.config.verbose:
                logger.info("📊 Generating probability features...")
            feature_df = self._generate_probability_features(feature_df)
        
        if self.config.include_context_features:
            if self.config.verbose:
                logger.info("🔍 Generating context features...")
            feature_df = self._generate_context_features(feature_df)
        
        if self.config.include_genomic_features:
            if self.config.verbose:
                logger.info("🧬 Generating genomic features...")
            feature_df = self._generate_genomic_features(feature_df, gene_id)
        
        if self.config.include_kmer_features:
            if self.config.verbose:
                logger.info("🔤 Generating k-mer features...")
            feature_df = self._generate_kmer_features(feature_df)
        
        # Harmonize with training features
        feature_df = self._harmonize_with_training_features(feature_df)
        
        if self.config.verbose:
            logger.info(f"✅ Generated {feature_df.shape[1]} features for {feature_df.shape[0]} positions")
        
        return feature_df
    
    def _harmonize_with_training_features(self, feature_df: pd.DataFrame) -> pd.DataFrame:
        """Ensure feature matrix matches training feature manifest exactly"""
        if self._feature_manifest is None:
            if self.config.verbose:
                logger.warning("⚠️ No feature manifest loaded - skipping harmonization")
            return feature_df
        
        # Separate features by type for different handling
        kmer_pattern = r'^\d+mer_'
        kmer_features = [f for f in self._feature_manifest if pd.Series([f]).str.match(kmer_pattern).iloc[0]]
        other_features = [f for f in self._feature_manifest if f not in kmer_features]
        
        # Handle missing k-mer features (fill with zeros)
        missing_kmers = set(kmer_features) - set(feature_df.columns)
        if missing_kmers:
            if self.config.verbose:
                logger.info(f"🔤 Adding {len(missing_kmers)} missing k-mer features (filled with 0)")
            for kmer in missing_kmers:
                feature_df[kmer] = 0.0
        
        # Handle missing other features (warn but use defaults)
        missing_other = set(other_features) - set(feature_df.columns)
        if missing_other:
            if self.config.verbose:
                logger.warning(f"⚠️ Missing {len(missing_other)} non-kmer features: {list(missing_other)[:5]}...")
            # Fill with reasonable defaults based on feature type
            for feature in missing_other:
                if 'length' in feature.lower():
                    feature_df[feature] = 1000.0  # Reasonable sequence length
                elif 'complexity' in feature.lower():
                    feature_df[feature] = 0.5  # Medium complexity
                elif 'gc_content' in feature.lower():
                    feature_df[feature] = 0.5  # 50% GC content
                else:
                    feature_df[feature] = 0.0  # Default to 0
        
        # Remove extra features not in manifest
        extra_features = set(feature_df.columns) - set(self._feature_manifest)
        if extra_features:
            if self.config.verbose:
                logger.info(f"✂️ Removing {len(extra_features)} extra features not in training manifest")
            feature_df = feature_df.drop(columns=list(extra_features))
        
        # Reorder columns to match training exactly
        feature_df = feature_df[self._feature_manifest]
        
        return feature_df


def create_optimized_enricher(verbose: bool = True) -> OptimizedInferenceEnricher:
    """Factory function to create an optimized inference enricher"""
    config = InferenceFeatureConfig(
        use_cache=True,
        verbose=verbose,
        include_probability_features=True,
        include_genomic_features=True,
        include_kmer_features=True,
        include_context_features=True
    )
    return OptimizedInferenceEnricher(config)