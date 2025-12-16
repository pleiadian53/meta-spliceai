"""
Dataset preparation for deep error models.

This module prepares position-centric training datasets from meta-model artifacts
for transformer-based error classification models.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import logging

import pandas as pd
import polars as pl
import numpy as np
from sklearn.model_selection import train_test_split

# Add project root to path for imports
project_root = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(project_root))

from meta_spliceai.splice_engine.meta_models.error_model.config import ErrorModelConfig
from meta_spliceai.splice_engine.meta_models.io.handlers import MetaModelDataHandler
from meta_spliceai.splice_engine.model_evaluator import ModelEvaluationFileHandler
from meta_spliceai.splice_engine.analysis_utils import label_analysis_dataset


class ErrorDatasetPreparer:
    """
    Prepares position-centric datasets for deep error model training.
    
    This class handles:
    1. Loading analysis_sequences_* and splice_positions_enhanced_* artifacts
    2. Extracting contextual sequences around target positions
    3. Including base model scores and derived features
    4. Creating train/val/test splits
    5. Formatting data for transformer models
    """
    
    def __init__(self, config: ErrorModelConfig, eval_dir: Optional[str] = None):
        """
        Initialize the dataset preparer.
        
        Parameters
        ----------
        config : ErrorModelConfig
            Configuration for the error model.
        eval_dir : Optional[str], optional
            Path to evaluation directory containing artifacts.
            If None, uses default from MetaModelDataHandler.
        """
        self.config = config
        self.eval_dir = eval_dir
        
        # Initialize data handlers - use eval_dir directly for single directory input
        self.data_handler = MetaModelDataHandler(eval_dir=eval_dir)
        self.file_handler = ModelEvaluationFileHandler(
            eval_dir or self.data_handler.eval_dir, 
            separator='\t'
        )
        
        self.logger = logging.getLogger(__name__)
        
    def prepare_dataset(
        self, 
        gene_ids: Optional[List[str]] = None,
        facet: str = "simple",
        return_splits: bool = True
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
        """
        Prepare complete dataset for error model training.
        
        Parameters
        ----------
        gene_ids : List[str], optional
            Specific gene IDs to include. If None, uses all available genes.
        facet : str, default "simple"
            Dataset facet to load (e.g., "simple", "trimmed")
        return_splits : bool, default True
            Whether to return train/val/test splits or full dataset
            
        Returns
        -------
        Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]
            If return_splits=True: (train_df, val_df, test_df)
            If return_splits=False: full_df
        """
        self.logger.info("Starting dataset preparation...")
        
        # Step 1: Load analysis sequences
        analysis_df = self._load_analysis_sequences(facet=facet)
        
        # Step 2: Filter by gene IDs if specified
        if gene_ids is not None:
            analysis_df = analysis_df[analysis_df['gene_id'].isin(gene_ids)]
            self.logger.info(f"Filtered to {len(gene_ids)} genes, {len(analysis_df)} positions")
        
        # Step 3: Extract contextual sequences
        analysis_df = self._extract_contextual_sequences(analysis_df)
        
        # Step 4: Add labels for binary classification
        analysis_df = self._add_binary_labels(analysis_df)
        
        # Step 5: Extract features from analysis sequences
        analysis_df = self._extract_features_from_analysis_sequences(analysis_df)
        
        # Step 6: Final preprocessing
        analysis_df = self._preprocess_features(analysis_df)
        
        self.logger.info(f"Final dataset shape: {analysis_df.shape}")
        self.logger.info(f"Label distribution:\n{analysis_df['label'].value_counts()}")
        
        if not return_splits:
            return analysis_df
            
        # Step 7: Create train/val/test splits
        return self._create_splits(analysis_df)
    
    def prepare_dataset_from_dataframe(
        self,
        df: pd.DataFrame,
        error_label: str = "FP",
        correct_label: str = "TP"
    ) -> Dict[str, Any]:
        """
        Prepare dataset from existing DataFrame (for workflow integration).
        
        Parameters
        ----------
        df : pd.DataFrame
            Pre-loaded analysis sequences DataFrame
        error_label : str, default "FP"
            Label for error class
        correct_label : str, default "TP"
            Label for correct class
            
        Returns
        -------
        Dict[str, Any]
            Dictionary containing datasets and metadata
        """
        self.logger.info("Preparing dataset from provided DataFrame...")
        
        # Ensure we have required columns
        required_cols = ['gene_id', 'transcript_id', 'sequence']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Add binary labels based on prediction_type if available
        if 'prediction_type' in df.columns:
            df = df.copy()
            df['label'] = (df['prediction_type'] == error_label).astype(int)
        elif 'pred_type' in df.columns:
            df = df.copy()
            df['label'] = (df['pred_type'] == error_label).astype(int)
        elif 'label' not in df.columns:
            # Try to infer from existing error analysis
            df = label_analysis_dataset(df, positive_class=error_label)
        
        # Use sequence column as context_sequence (they are the same in analysis artifacts)
        if 'context_sequence' not in df.columns:
            df['context_sequence'] = df['sequence']
        
        # Extract features from analysis sequences
        df = self._extract_features_from_analysis_sequences(df)
        
        # Preprocess features
        df = self._preprocess_features(df)
        
        # Create splits
        train_df, val_df, test_df = self._create_splits(df)
        
        # Convert to dataset format using proper config-aware method
        from meta_spliceai.splice_engine.meta_models.error_model.dataset.data_utils import DataUtils
        from transformers import AutoTokenizer
        
        # Initialize tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_name)
        
        # Use the proper method that incorporates config feature settings
        train_dataset, val_dataset, test_dataset = DataUtils.prepare_datasets_from_splits(
            train_df=train_df,
            val_df=val_df, 
            test_df=test_df,
            tokenizer=tokenizer,
            config=self.config  # This ensures feature config is used!
        )
        
        datasets = {
            'train': train_dataset,
            'val': val_dataset,
            'test': test_dataset
        }
        
        return {
            'datasets': datasets,
            'dataframes': {
                'train': train_df,
                'val': val_df,
                'test': test_df
            },
            'metadata': {
                'total_samples': len(df),
                'error_samples': (df['label'] == 1).sum(),
                'correct_samples': (df['label'] == 0).sum(),
                'error_label': error_label,
                'correct_label': correct_label
            }
        }
    
    def _load_analysis_sequences(self, facet: str = "simple") -> pd.DataFrame:
        """Load analysis sequences from artifacts."""
        self.logger.info(f"Loading analysis sequences (facet: {facet})...")
        
        subject = f"analysis_sequences_{facet}"
        
        # Use the file handler to load analysis sequences
        analysis_df = self.file_handler.load_analysis_sequences(
            aggregated=True,
            subject=subject,
            error_label=self.config.error_label,
            correct_label=self.config.correct_label,
            splice_type=self.config.splice_type
        )
        
        # Convert Polars to Pandas if needed
        if isinstance(analysis_df, pl.DataFrame):
            analysis_df = analysis_df.to_pandas()
        
        self.logger.info(f"Loaded {len(analysis_df)} sequences")
        return analysis_df
    
    def _extract_contextual_sequences(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract contextual sequences around target positions.
        
        The original 'sequence' column contains full gene sequences.
        We need to extract symmetric context around the target position.
        """
        self.logger.info(f"Extracting contextual sequences (±{self.config.context_radius} nt)...")
        
        def extract_context(row):
            """Extract context around target position."""
            sequence = row['sequence']
            position = row['position']  # Relative position within sequence
            
            # Calculate context boundaries
            start_pos = max(0, position - self.config.context_radius)
            end_pos = min(len(sequence), position + self.config.context_radius + 1)
            
            # Extract context sequence
            context_seq = sequence[start_pos:end_pos]
            
            # Pad if necessary to maintain consistent length
            if len(context_seq) < self.config.context_length:
                # Pad with 'N' nucleotides
                pad_left = max(0, self.config.context_radius - position)
                pad_right = max(0, (position + self.config.context_radius + 1) - len(sequence))
                context_seq = 'N' * pad_left + context_seq + 'N' * pad_right
            
            return context_seq
        
        # Apply context extraction
        df['context_sequence'] = df.apply(extract_context, axis=1)
        
        # Validate context lengths
        context_lengths = df['context_sequence'].str.len()
        if not all(context_lengths == self.config.context_length):
            self.logger.warning(f"Context length variation detected: {context_lengths.describe()}")
        
        self.logger.info(f"Extracted contexts with length {self.config.context_length}")
        return df
    
    def _add_binary_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add binary labels for error classification."""
        self.logger.info("Adding binary labels...")
        
        # Use existing label_analysis_dataset function
        if isinstance(df, pd.DataFrame):
            # Convert to Polars for processing
            df_pl = pl.from_pandas(df)
            df_pl = label_analysis_dataset(df_pl, positive_class=self.config.error_label)
            df = df_pl.to_pandas()
        else:
            df = label_analysis_dataset(df, positive_class=self.config.error_label)
        
        # Ensure we have the expected label mapping
        label_counts = df['label'].value_counts()
        self.logger.info(f"Binary label distribution: {dict(label_counts)}")
        
        return df
    
    def _extract_features_from_analysis_sequences(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract and select features directly from analysis_sequences_* files.
        
        The analysis_sequences_* files contain all necessary features including:
        - Base model scores (score, donor_score, acceptor_score, neither_score)
        - Context features (context_score_*, context_neighbor_mean, etc.)
        - Probability-derived features (donor_*, acceptor_*, probability_entropy, etc.)
        - Statistical features (signal_strength_ratio, splice_probability, etc.)
        
        Excludes metadata and leakage columns that should not be used for training.
        """
        self.logger.info("Extracting features from analysis sequences...")
        
        # Import column definitions from preprocessing module
        from meta_spliceai.splice_engine.meta_models.builder.preprocessing import (
            METADATA_COLUMNS, LEAKAGE_COLUMNS, SEQUENCE_COLUMNS
        )
        
        # All features are already in the analysis_sequences_* files
        # No need to load additional files - just select the appropriate columns
        
        available_columns = set(df.columns)
        excluded_columns = set(METADATA_COLUMNS + LEAKAGE_COLUMNS + SEQUENCE_COLUMNS)
        
        # Add error model specific exclusions
        excluded_columns.update({
            'prediction_type', 'error_type', 'correct_type',  # Labels
            'splice_type',  # Target label
            # Note: 'sequence' is NOT excluded - it's the primary feature for transformers
        })
        
        selected_features = []
        
        # Base model scores
        if self.config.include_base_scores:
            base_score_cols = ['score', 'donor_score', 'acceptor_score', 'neither_score']
            selected_features.extend([col for col in base_score_cols if col in available_columns])
        
        # Context features
        if self.config.include_context_features:
            context_cols = [
                'context_score_m1', 'context_score_p1', 'context_score_m2', 'context_score_p2',
                'context_neighbor_mean', 'context_asymmetry', 'context_max'
            ]
            selected_features.extend([col for col in context_cols if col in available_columns])
        
        # Probability-derived donor features
        if self.config.include_donor_features:
            donor_cols = [
                'donor_diff_m1', 'donor_surge_ratio', 'donor_peak_height_ratio', 
                'donor_signal_strength', 'donor_context_diff_ratio', 'donor_local_max_ratio',
                'donor_context_asymmetry', 'donor_relative_strength'
            ]
            selected_features.extend([col for col in donor_cols if col in available_columns])
        
        # Probability-derived acceptor features
        if self.config.include_acceptor_features:
            acceptor_cols = [
                'acceptor_diff_m1', 'acceptor_surge_ratio', 'acceptor_peak_height_ratio',
                'acceptor_signal_strength', 'acceptor_context_diff_ratio', 'acceptor_local_max_ratio',
                'acceptor_context_asymmetry', 'acceptor_relative_strength'
            ]
            selected_features.extend([col for col in acceptor_cols if col in available_columns])
        
        # Statistical and probability-derived features
        if self.config.include_derived_features:
            derived_cols = [
                'probability_entropy', 'splice_probability', 'signal_strength_ratio',
                'type_signal_difference', 'relative_donor_probability', 'relative_acceptor_probability',
                'max_probability_difference', 'probability_variance'
            ]
            selected_features.extend([col for col in derived_cols if col in available_columns])
        
        # Filter out any excluded columns that might have been selected
        selected_features = [col for col in selected_features if col not in excluded_columns]
        
        # Deduplicate while preserving order
        selected_features = list(dict.fromkeys(selected_features))
        
        # Call the new clean logging function
        from .data_utils import DataUtils
        DataUtils.log_feature_summary(
            feature_columns=selected_features,
            num_samples=len(df),
            feature_dim_after_transform=None  # Will be determined by the model architecture
        )
        
        # Ensure essential columns are retained
        essential_cols = ['gene_id', 'transcript_id', 'position', 'label', 'context_sequence', 'sequence']
        for col in essential_cols:
            if col in df.columns and col not in selected_features:
                df = df.copy()  # Avoid modifying original
        
        return df[selected_features + essential_cols]
    
    def _preprocess_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess features for model training."""
        self.logger.info("Preprocessing features...")
        
        # Ensure required columns exist
        required_cols = ['context_sequence', 'label']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Clean sequence data
        df['context_sequence'] = df['context_sequence'].fillna('N' * self.config.context_length)
        
        # Handle missing values in additional features
        if self.config.include_base_scores:
            score_cols = [col for col in df.columns if 'score' in col.lower()]
            for col in score_cols:
                if col in df.columns:
                    df[col] = df[col].fillna(0.0)
        
        # Remove any remaining NaN values in critical columns
        df = df.dropna(subset=['context_sequence', 'label'])
        
        self.logger.info(f"Preprocessed dataset shape: {df.shape}")
        return df
    
    def _create_splits(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Create train/validation/test splits."""
        self.logger.info("Creating train/val/test splits...")
        
        if self.config.gene_level_split:
            # Split by genes to avoid data leakage
            unique_genes = df['gene_id'].unique()
            
            # First split: train vs (val + test)
            train_genes, temp_genes = train_test_split(
                unique_genes,
                test_size=(self.config.val_split + self.config.test_split),
                random_state=self.config.random_seed,
                shuffle=True
            )
            
            # Second split: val vs test
            val_genes, test_genes = train_test_split(
                temp_genes,
                test_size=self.config.test_split / (self.config.val_split + self.config.test_split),
                random_state=self.config.random_seed,
                shuffle=True
            )
            
            # Create splits based on gene membership
            train_df = df[df['gene_id'].isin(train_genes)].copy()
            val_df = df[df['gene_id'].isin(val_genes)].copy()
            test_df = df[df['gene_id'].isin(test_genes)].copy()
            
        else:
            # Split by positions (standard approach)
            # First split: train vs (val + test)
            train_df, temp_df = train_test_split(
                df,
                test_size=(self.config.val_split + self.config.test_split),
                random_state=self.config.random_seed,
                stratify=df['label'],
                shuffle=True
            )
            
            # Second split: val vs test
            val_df, test_df = train_test_split(
                temp_df,
                test_size=self.config.test_split / (self.config.val_split + self.config.test_split),
                random_state=self.config.random_seed,
                stratify=temp_df['label'],
                shuffle=True
            )
        
        # Log split statistics
        for name, split_df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
            label_dist = split_df['label'].value_counts()
            self.logger.info(f"{name} split: {len(split_df)} samples, labels: {dict(label_dist)}")
        
        return train_df, val_df, test_df
    
    def save_splits(
        self, 
        train_df: pd.DataFrame, 
        val_df: pd.DataFrame, 
        test_df: pd.DataFrame,
        output_dir: Path
    ) -> Dict[str, Path]:
        """Save train/val/test splits to files."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        saved_files = {}
        for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
            file_path = output_dir / f"{name}_dataset.parquet"
            df.to_parquet(file_path, index=False)
            saved_files[name] = file_path
            self.logger.info(f"Saved {name} split to {file_path}")
        
        return saved_files
