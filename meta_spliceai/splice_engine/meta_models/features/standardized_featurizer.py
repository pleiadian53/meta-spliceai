"""
Standardized Per-Nucleotide Featurization Module
================================================

This module provides a unified featurization pipeline for meta-model training and inference.
It ensures consistent feature generation across different scenarios:
1. Training data assembly (incremental_builder.py)
2. Inference on unseen positions (scenario 1)
3. Inference on unseen genes (scenarios 2A and 2B)

The featurization process includes:
- Raw probability scores from base model
- Advanced probability features (entropy, ratios, etc.)
- Context-aware features from neighboring positions
- Sequence motifs (k-mers, GC content, etc.)
- Genomic features (gene length, splice sites, etc.)
- Feature enrichment (gene-level, performance, overlaps, etc.)
"""

from __future__ import annotations

import pandas as pd
import polars as pl
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union, Any

from meta_spliceai.splice_engine.meta_models.features.kmer_features import make_kmer_features
from meta_spliceai.splice_engine.meta_models.features.feature_enrichment import (
    apply_feature_enrichers, list_enrichers
)
from meta_spliceai.splice_engine.meta_models.core.enhanced_workflow import (
    enhanced_process_predictions_with_all_scores
)
from meta_spliceai.splice_engine.meta_models.builder.preprocessing import (
    drop_unwanted_columns, impute_nulls
)


class StandardizedFeaturizer:
    """
    Unified featurization pipeline for meta-model training and inference.
    
    This class ensures consistent feature generation across all use cases,
    handling the complete feature extraction pipeline from raw predictions
    to enriched feature matrices.
    """
    
    def __init__(
        self,
        kmer_sizes: Optional[List[int]] = None,
        enrichers: Optional[List[str]] = None,
        training_schema: Optional[pd.DataFrame] = None,
        excluded_features: Optional[List[str]] = None,
        verbose: int = 1
    ):
        """
        Initialize the standardized featurizer.
        
        Parameters
        ----------
        kmer_sizes : List[int], optional
            K-mer sizes to extract (e.g., [3, 6]). If None, defaults to [3].
        enrichers : List[str], optional
            Feature enrichers to apply. If None, uses all available enrichers.
        training_schema : pd.DataFrame, optional
            Training schema for feature harmonization during inference.
            Should contain 'feature' column with expected feature names.
        excluded_features : List[str], optional
            List of features to exclude (e.g., from leakage analysis during training).
        verbose : int
            Verbosity level (0=silent, 1=normal, 2=detailed).
        """
        self.kmer_sizes = kmer_sizes or [3]
        self.enrichers = enrichers
        self.training_schema = training_schema
        self.excluded_features = set(excluded_features) if excluded_features else set()
        self.verbose = verbose
        
        # Cache training feature names if schema provided
        self.training_features = None
        self.training_features_ordered = None  # Preserve original order
        if training_schema is not None:
            # Handle both DataFrame with 'feature' column and list of feature names
            if isinstance(training_schema, pd.DataFrame):
                self.training_features_ordered = training_schema['feature'].tolist()  # Keep order
                self.training_features = set(self.training_features_ordered)  # For fast lookup
            elif isinstance(training_schema, list):
                self.training_features_ordered = training_schema  # Keep order
                self.training_features = set(training_schema)  # For fast lookup
            elif isinstance(training_schema, set):
                # If given a set, we can't preserve order, so sort for consistency
                self.training_features_ordered = sorted(training_schema)
                self.training_features = training_schema
            else:
                raise ValueError(f"training_schema must be DataFrame or list/set, got {type(training_schema)}")
            
            if self.verbose:
                print(f"[StandardizedFeaturizer] Loaded training schema with {len(self.training_features)} features")
                if self.excluded_features:
                    print(f"[StandardizedFeaturizer] Will exclude {len(self.excluded_features)} features: {sorted(self.excluded_features)}")
    
    def featurize_from_analysis_df(
        self,
        analysis_df: pd.DataFrame,
        base_predictions_df: Optional[pd.DataFrame] = None,
        include_probability_features: bool = True,
        include_context_features: bool = True,
        include_kmer_features: bool = True,
        include_genomic_features: bool = True,
        harmonize_with_training: bool = True
    ) -> pd.DataFrame:
        """
        Generate complete feature matrix from analysis sequences dataframe.
        
        This is the main entry point for inference workflows where we have
        analysis_sequences artifacts with existing probability/context features.
        
        Parameters
        ----------
        analysis_df : pd.DataFrame
            Analysis sequences dataframe containing at minimum:
            - gene_id, position, sequence
            - Probability scores and context features (if from enhanced workflow)
        base_predictions_df : pd.DataFrame, optional
            Base model predictions with donor_score, acceptor_score, neither_score.
            If not provided, assumes these are already in analysis_df.
        include_probability_features : bool
            Whether to include/verify probability-based features.
        include_context_features : bool
            Whether to include/verify context-based features.
        include_kmer_features : bool
            Whether to extract k-mer features from sequences.
        include_genomic_features : bool
            Whether to add genomic features via enrichment.
        harmonize_with_training : bool
            Whether to harmonize features with training schema.
            
        Returns
        -------
        pd.DataFrame
            Complete feature matrix ready for model input.
        """
        if self.verbose:
            print(f"[StandardizedFeaturizer] Starting featurization for {len(analysis_df)} positions")
        
        # Start with a copy to avoid modifying input
        features_df = analysis_df.copy()
        
        # Step 1: Ensure base probability scores are present
        if base_predictions_df is not None:
            features_df = self._merge_base_scores(features_df, base_predictions_df)
        
        # Check what features we already have
        existing_features = self._check_existing_features(features_df)
        
        # Step 2: Generate/verify probability and context features
        if include_probability_features or include_context_features:
            features_df = self._ensure_probability_context_features(
                features_df, 
                existing_features,
                include_probability_features,
                include_context_features
            )
        
        # Step 3: Extract k-mer features from sequences
        if include_kmer_features and 'sequence' in features_df.columns:
            features_df = self._add_kmer_features(features_df)
        
        # Step 4: Apply genomic feature enrichment
        if include_genomic_features:
            features_df = self._add_genomic_features(features_df)
        
        # Step 5: Harmonize with training schema if needed
        if harmonize_with_training and self.training_features is not None:
            features_df = self._harmonize_features(features_df)
        
        # Step 6: Handle chromosome encoding
        if 'chrom' in features_df.columns:
            features_df = self._encode_chromosome(features_df)
        
        if self.verbose:
            feature_cols = [c for c in features_df.columns if c not in ['gene_id', 'position']]
            print(f"[StandardizedFeaturizer] Generated {len(feature_cols)} features")
        
        return features_df
    
    def featurize_from_predictions(
        self,
        predictions_dict: Dict[str, Any],
        ss_annotations_df: Optional[pl.DataFrame] = None,
        **enhanced_workflow_kwargs
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate features from raw SpliceAI predictions using enhanced workflow.
        
        This is used during training data generation where we start from
        raw predictions and need to generate all features from scratch.
        
        Parameters
        ----------
        predictions_dict : Dict[str, Any]
            Raw predictions from SpliceAI model.
        ss_annotations_df : pl.DataFrame, optional
            Splice site annotations for labeling.
        **enhanced_workflow_kwargs
            Additional arguments for enhanced_process_predictions_with_all_scores.
            
        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            (positions_df, error_analysis_df)
        """
        if self.verbose:
            print("[StandardizedFeaturizer] Processing predictions through enhanced workflow")
        
        # Use enhanced workflow to generate all probability/context features
        positions_df, error_df = enhanced_process_predictions_with_all_scores(
            predictions_dict,
            ss_annotations_df=ss_annotations_df,
            add_derived_features=True,
            compute_all_context_features=True,
            verbose=self.verbose,
            **enhanced_workflow_kwargs
        )
        
        # Convert to pandas for consistency
        if isinstance(positions_df, pl.DataFrame):
            positions_df = positions_df.to_pandas()
        if isinstance(error_df, pl.DataFrame):
            error_df = error_df.to_pandas()
        
        return positions_df, error_df
    
    def _check_existing_features(self, df: pd.DataFrame) -> Dict[str, Set[str]]:
        """Check what feature categories are already present."""
        columns = set(df.columns)
        
        # Define feature patterns
        probability_features = {
            'donor_score', 'acceptor_score', 'neither_score',
            'relative_donor_probability', 'probability_entropy',
            'splice_probability', 'donor_acceptor_diff'
        }
        
        context_features = {
            'context_score_m2', 'context_score_m1', 
            'context_score_p1', 'context_score_p2',
            'context_neighbor_mean', 'context_asymmetry',
            'donor_surge_ratio', 'acceptor_surge_ratio'
        }
        
        kmer_pattern_cols = [c for c in columns if 'mer_' in c]
        genomic_features = {
            'gene_length', 'transcript_length', 'n_splice_sites',
            'gene_start', 'gene_end', 'num_overlaps'
        }
        
        return {
            'probability': probability_features & columns,
            'context': context_features & columns,
            'kmer': set(kmer_pattern_cols),
            'genomic': genomic_features & columns
        }
    
    def _merge_base_scores(
        self, 
        features_df: pd.DataFrame, 
        base_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Merge base model scores if not already present."""
        score_cols = ['donor_score', 'acceptor_score', 'neither_score']
        missing_scores = [c for c in score_cols if c not in features_df.columns]
        
        if missing_scores and all(c in base_df.columns for c in score_cols):
            if self.verbose >= 2:
                print(f"[StandardizedFeaturizer] Merging base scores: {missing_scores}")
            
            merge_cols = ['position']
            if 'gene_id' in features_df.columns and 'gene_id' in base_df.columns:
                merge_cols.append('gene_id')
            
            features_df = features_df.merge(
                base_df[merge_cols + score_cols],
                on=merge_cols,
                how='left'
            )
        
        return features_df
    
    def _ensure_probability_context_features(
        self,
        features_df: pd.DataFrame,
        existing_features: Dict[str, Set[str]],
        include_probability: bool,
        include_context: bool
    ) -> pd.DataFrame:
        """Ensure all probability and context features are present."""
        # If we already have most features, assume they're correct
        if len(existing_features['probability']) >= 5 and len(existing_features['context']) >= 6:
            if self.verbose >= 2:
                print("[StandardizedFeaturizer] Probability and context features already present")
            return features_df
        
        # Otherwise, we'd need to regenerate them
        # This would require the full enhanced workflow with neighboring positions
        if self.verbose:
            print("[StandardizedFeaturizer] Warning: Missing probability/context features")
            print(f"  Found probability features: {len(existing_features['probability'])}")
            print(f"  Found context features: {len(existing_features['context'])}")
        
        # For now, add placeholders for critical missing features
        # In production, this should trigger a full recomputation
        critical_features = {
            'relative_donor_probability': 0.0,
            'probability_entropy': 0.0,
            'splice_probability': 0.0,
            'context_neighbor_mean': 0.0,
            'donor_surge_ratio': 0.0,
            'acceptor_surge_ratio': 0.0
        }
        
        for feat, default_val in critical_features.items():
            if feat not in features_df.columns:
                features_df[feat] = default_val
        
        return features_df
    
    def _add_kmer_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Extract k-mer features from sequences."""
        if self.verbose:
            print(f"[StandardizedFeaturizer] Extracting k-mer features (k={self.kmer_sizes})")
        
        try:
            # Use the existing kmer feature extraction
            kmer_result = make_kmer_features(
                features_df,
                kmer_sizes=self.kmer_sizes,
                verbose=0
            )
            
            # Handle both tuple and DataFrame return types
            if isinstance(kmer_result, tuple):
                kmer_df, _ = kmer_result
            else:
                kmer_df = kmer_result
            
            # Merge k-mer features back
            merge_cols = ['gene_id', 'position']
            if 'gene_id' not in features_df.columns:
                merge_cols = ['position']
            
            # Get k-mer columns (excluding merge columns and sequence)
            kmer_cols = [c for c in kmer_df.columns 
                        if c not in merge_cols + ['sequence']]
            
            # Add k-mer features to main dataframe
            for col in kmer_cols:
                if col not in features_df.columns:
                    features_df[col] = kmer_df[col]
            
            if self.verbose >= 2:
                print(f"[StandardizedFeaturizer] Added {len(kmer_cols)} k-mer features")
            
        except Exception as e:
            if self.verbose:
                print(f"[StandardizedFeaturizer] K-mer extraction failed: {e}")
        
        return features_df
    
    def _add_genomic_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Add genomic features through feature enrichment."""
        if self.verbose:
            print("[StandardizedFeaturizer] Applying feature enrichment")
        
        try:
            # Convert to polars for enrichment
            if isinstance(features_df, pd.DataFrame):
                features_pl = pl.from_pandas(features_df)
            else:
                features_pl = features_df
            
            # Apply enrichers
            enrichers_to_use = self.enrichers
            if enrichers_to_use is None:
                enrichers_to_use = list_enrichers()
            
            if self.verbose >= 2:
                print(f"[StandardizedFeaturizer] Using enrichers: {enrichers_to_use}")
            
            enriched_df = apply_feature_enrichers(
                features_pl,
                enrichers=enrichers_to_use,
                verbose=0
            )
            
            # Convert back to pandas
            features_df = enriched_df.to_pandas()
            
        except Exception as e:
            if self.verbose:
                print(f"[StandardizedFeaturizer] Feature enrichment failed: {e}")
                print("[StandardizedFeaturizer] Adding default genomic features")
            
            # Add default values as fallback
            genomic_defaults = {
                'gene_start': 0, 'gene_end': 10000, 'gene_length': 10000,
                'transcript_length': 10000, 'tx_start': 0, 'tx_end': 10000,
                'num_overlaps': 0, 'n_splice_sites': 20
            }
            
            for feat, default_val in genomic_defaults.items():
                if feat not in features_df.columns:
                    features_df[feat] = default_val
        
        return features_df
    
    def _harmonize_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Harmonize features with training schema."""
        if self.verbose:
            print("[StandardizedFeaturizer] Harmonizing features with training schema")
        
        # First, remove any features that were excluded during training
        if self.excluded_features:
            cols_to_drop = [col for col in self.excluded_features if col in features_df.columns]
            if cols_to_drop:
                if self.verbose >= 2:
                    print(f"[StandardizedFeaturizer] Dropping excluded features: {cols_to_drop}")
                features_df = features_df.drop(columns=cols_to_drop)
        
        # Identify missing and extra features
        current_features = set(features_df.columns)
        feature_cols = current_features - {'gene_id', 'position'}
        
        missing_features = self.training_features - feature_cols
        extra_features = feature_cols - self.training_features
        
        if self.verbose >= 2:
            print(f"[StandardizedFeaturizer] Missing features: {len(missing_features)}")
            if missing_features:
                print(f"[StandardizedFeaturizer]   Missing: {sorted(missing_features)[:5]}...")
            print(f"[StandardizedFeaturizer] Extra features: {len(extra_features)}")
            if extra_features:
                print(f"[StandardizedFeaturizer]   Extra: {sorted(extra_features)[:5]}...")
        
        # Add missing features with appropriate default values
        for feat in missing_features:
            # Use appropriate defaults based on feature type
            if feat == 'chrom':
                # For unseen genes, default chromosome to 0 (unknown)
                features_df[feat] = 0
            elif feat in ['gene_length', 'transcript_length', 'gene_start', 'gene_end', 
                          'tx_start', 'tx_end', 'n_splice_sites']:
                # For genomic coordinate features, use -1 to indicate missing
                features_df[feat] = -1
            elif feat == 'num_overlaps':
                # Default to 0 overlapping genes
                features_df[feat] = 0
            else:
                # Default to 0.0 for other features
                features_df[feat] = 0.0
                
            if self.verbose >= 2:
                print(f"[StandardizedFeaturizer]   Added missing feature '{feat}' with default value")
        
        # Select only training features plus metadata, preserving original order
        # Use training_features_ordered if available to maintain the exact order from training
        if self.training_features_ordered:
            ordered_features = self.training_features_ordered
        else:
            # Fallback to sorted if we don't have the original order
            ordered_features = sorted(self.training_features)
            
        # Ensure all ordered features exist in the dataframe
        final_features = []
        for feat in ordered_features:
            if feat in features_df.columns:
                final_features.append(feat)
            else:
                # This shouldn't happen after adding missing features, but be safe
                if self.verbose >= 1:
                    print(f"[StandardizedFeaturizer] Warning: Feature '{feat}' still missing after harmonization")
                # Add it with default value
                features_df[feat] = 0.0
                final_features.append(feat)
                
        final_cols = ['gene_id', 'position'] + final_features
        
        return features_df[final_cols]
    
    def _encode_chromosome(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Encode chromosome column to numeric if needed."""
        if features_df['chrom'].dtype == 'object':
            if self.verbose >= 2:
                print("[StandardizedFeaturizer] Encoding chromosome to numeric")
            
            chrom_map = {
                '1': 1, '2': 2, '3': 3, '4': 4, '5': 5,
                '6': 6, '7': 7, '8': 8, '9': 9, '10': 10,
                '11': 11, '12': 12, '13': 13, '14': 14, '15': 15,
                '16': 16, '17': 17, '18': 18, '19': 19, '20': 20,
                '21': 21, '22': 22, 'X': 23, 'Y': 24, 'MT': 25, 'M': 25
            }
            
            features_df['chrom'] = features_df['chrom'].map(chrom_map).fillna(0).astype(int)
        
        return features_df


def create_featurizer_from_model_path(
    model_path: Union[str, Path],
    verbose: int = 1
) -> StandardizedFeaturizer:
    """
    Create a StandardizedFeaturizer configured for a specific model.
    
    This helper function loads the training schema and configuration
    from a model directory to ensure consistent featurization.
    
    Parameters
    ----------
    model_path : str or Path
        Path to the trained model directory.
    verbose : int
        Verbosity level.
        
    Returns
    -------
    StandardizedFeaturizer
        Configured featurizer instance.
    """
    model_path = Path(model_path)
    
    # Load training schema
    training_schema = None
    schema_path = model_path / "feature_manifest.csv"
    if schema_path.exists():
        training_schema = pd.read_csv(schema_path)
        if verbose:
            print(f"[create_featurizer] Loaded training schema from {schema_path}")
    
    # Load excluded features
    excluded_features = []
    excluded_path = model_path / "excluded_features.txt"
    if excluded_path.exists():
        with open(excluded_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    excluded_features.append(line)
        if verbose and excluded_features:
            print(f"[create_featurizer] Loaded {len(excluded_features)} excluded features from {excluded_path}")
    
    # Load model config to get k-mer sizes
    kmer_sizes = [3]  # Default
    config_path = model_path / "model_config.json"
    if config_path.exists():
        import json
        with open(config_path) as f:
            config = json.load(f)
            if 'kmer_sizes' in config:
                kmer_sizes = config['kmer_sizes']
    
    return StandardizedFeaturizer(
        kmer_sizes=kmer_sizes,
        training_schema=training_schema,
        excluded_features=excluded_features,
        verbose=verbose
    )
