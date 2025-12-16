"""
Data handling utilities for deep error models.

This module provides utility functions for data preprocessing, tokenization,
and dataset management for transformer-based error classification.
"""

import re
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import logging


class DNASequenceDataset(Dataset):
    """
    PyTorch Dataset for DNA sequences with additional features.
    
    This dataset handles:
    1. DNA sequence tokenization
    2. Additional numerical features (scores, derived features)
    3. Binary classification labels
    """
    
    def __init__(
        self,
        sequences: List[str],
        labels: List[int],
        tokenizer: AutoTokenizer,
        max_length: int = 512,
        additional_features: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None
    ):
        """
        Initialize DNA sequence dataset.
        
        Parameters
        ----------
        sequences : List[str]
            List of DNA sequences
        labels : List[int]
            Binary labels (0 or 1)
        tokenizer : AutoTokenizer
            Tokenizer for DNA sequences
        max_length : int, default 512
            Maximum sequence length for tokenization
        additional_features : np.ndarray, optional
            Additional numerical features (n_samples x n_features)
        feature_names : List[str], optional
            Names of additional features
        """
        self.sequences = sequences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.additional_features = additional_features
        self.feature_names = feature_names or []
        
        # Validate inputs
        if len(sequences) != len(labels):
            raise ValueError("Number of sequences must match number of labels")
        
        if additional_features is not None:
            if len(additional_features) != len(sequences):
                raise ValueError("Number of additional features must match number of sequences")
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single item from the dataset."""
        sequence = self.sequences[idx]
        label = self.labels[idx]
        
        # Tokenize sequence
        encoding = self.tokenizer(
            sequence,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        item = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }
        
        # Add additional features if available
        if self.additional_features is not None:
            item['additional_features'] = torch.tensor(
                self.additional_features[idx], 
                dtype=torch.float32
            )
        
        return item


class DataUtils:
    """Utility functions for data preprocessing and management."""
    
    @staticmethod
    def clean_dna_sequence(sequence: str) -> str:
        """
        Clean DNA sequence by removing invalid characters and normalizing.
        
        Parameters
        ----------
        sequence : str
            Raw DNA sequence
            
        Returns
        -------
        str
            Cleaned DNA sequence
        """
        # Convert to uppercase
        sequence = sequence.upper()
        
        # Replace invalid characters with 'N'
        sequence = re.sub(r'[^ATCGN]', 'N', sequence)
        
        return sequence
    
    @staticmethod
    def validate_sequences(sequences: List[str], min_length: int = 50) -> Tuple[List[str], List[int]]:
        """
        Validate DNA sequences and return valid sequences with their indices.
        
        Parameters
        ----------
        sequences : List[str]
            List of DNA sequences
        min_length : int, default 50
            Minimum sequence length
            
        Returns
        -------
        Tuple[List[str], List[int]]
            (valid_sequences, valid_indices)
        """
        valid_sequences = []
        valid_indices = []
        
        for i, seq in enumerate(sequences):
            cleaned_seq = DataUtils.clean_dna_sequence(seq)
            
            # Check minimum length
            if len(cleaned_seq) >= min_length:
                valid_sequences.append(cleaned_seq)
                valid_indices.append(i)
        
        return valid_sequences, valid_indices
    
    @staticmethod
    def log_feature_summary(feature_columns: List[str], num_samples: int, feature_dim_after_transform: int = None):
        """
        Log a clean summary of features being used in the model.
        
        Parameters
        ----------
        feature_columns : List[str]
            List of feature column names
        num_samples : int
            Number of samples in the dataset
        feature_dim_after_transform : int, optional
            Dimension after transformation/embedding layers
        """
        import logging
        logger = logging.getLogger(__name__)
        
        # Group features by category
        feature_groups = {
            'Base Scores': [],
            'Context Features': [],
            'Donor Features': [],
            'Acceptor Features': [],
            'Statistical Features': [],
            'Other Features': []
        }
        
        for feat in sorted(feature_columns):
            if 'score' in feat.lower() and not any(x in feat.lower() for x in ['context', 'donor', 'acceptor']):
                feature_groups['Base Scores'].append(feat)
            elif 'context' in feat.lower():
                feature_groups['Context Features'].append(feat)
            elif 'donor' in feat.lower():
                feature_groups['Donor Features'].append(feat)
            elif 'acceptor' in feat.lower():
                feature_groups['Acceptor Features'].append(feat)
            elif any(kw in feat.lower() for kw in ['entropy', 'variance', 'probability', 'signal']):
                feature_groups['Statistical Features'].append(feat)
            else:
                feature_groups['Other Features'].append(feat)
        
        # Log clean summary
        logger.info("\n" + "=" * 70)
        logger.info("FEATURE SUMMARY:")
        logger.info("=" * 70)
        
        # Log feature groups
        total_features = len(feature_columns)
        for group_name, features in feature_groups.items():
            if features:
                logger.info(f"  {group_name:20s} [{len(features):2d} features]")
                # Show first few features as examples
                if len(features) <= 3:
                    for feat in features:
                        logger.info(f"    • {feat}")
                else:
                    for feat in features[:2]:
                        logger.info(f"    • {feat}")
                    logger.info(f"    • ... ({len(features)-2} more)")
        
        logger.info("-" * 70)
        logger.info("DIMENSIONALITY:")
        logger.info(f"  Input features:      [{num_samples:,} samples × {total_features} features]")
        if feature_dim_after_transform:
            logger.info(f"  After embedding:     [batch_size × {feature_dim_after_transform}]")
        logger.info("  Primary sequence:    Handled separately by transformer")
        logger.info("=" * 70 + "\n")
    
    @staticmethod
    def extract_additional_features(
        df: pd.DataFrame,
        include_base_scores: bool = True,
        include_derived_features: bool = True,
        include_genomic_features: bool = True,
        include_context_features: bool = True,
        include_donor_features: bool = True,
        include_acceptor_features: bool = True
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Extract additional numerical features from dataframe.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe with features
        include_base_scores : bool, default True
            Include raw base model prediction scores (acceptor_score, donor_score, neither_score, score)
        include_context_features : bool, default True
            Include context-specific features (context_score_*, context_neighbor_mean, etc.)
        include_donor_features : bool, default True
            Include donor-specific features (donor_diff_*, donor_surge_ratio, etc.)
        include_acceptor_features : bool, default True
            Include acceptor-specific features (acceptor_diff_*, acceptor_surge_ratio, etc.)
        include_derived_features : bool, default True
            Include general statistical/entropy features not covered by specific groups above
        include_genomic_features : bool, default True
            Include genomic features (gene-level, transcript-level)
            
        Returns
        -------
        Tuple[np.ndarray, List[str]]
            (feature_matrix, feature_names)
        """
        feature_columns = set()  # Use set for automatic deduplication
        
        # Splice Prediction Scores
        if include_base_scores:
            splice_score_features = [
                'acceptor_score', 'donor_score', 'neither_score', 'score'
            ]
            available_scores = [col for col in splice_score_features if col in df.columns]
            feature_columns.update(available_scores)
            
            # Also include any other score columns not in the explicit list
            other_score_cols = [col for col in df.columns 
                              if 'score' in col.lower() 
                              and col not in splice_score_features
                              and df[col].dtype in ['float64', 'float32', 'int64', 'int32']]
            feature_columns.update(other_score_cols)
        
        # Context Features
        if include_context_features:
            context_features = [
                'context_score_m2', 'context_score_m1', 'context_score_p1', 'context_score_p2',
                'context_neighbor_mean', 'context_asymmetry', 'context_max'
            ]
            available_context = [col for col in context_features if col in df.columns]
            feature_columns.update(available_context)
        
        # Donor-specific Features
        if include_donor_features:
            donor_features = [
                'donor_diff_m1', 'donor_diff_m2', 'donor_diff_p1', 'donor_diff_p2',
                'donor_surge_ratio', 'donor_is_local_peak', 'donor_weighted_context',
                'donor_peak_height_ratio', 'donor_second_derivative', 'donor_signal_strength',
                'donor_context_diff_ratio'
            ]
            available_donor = [col for col in donor_features if col in df.columns]
            feature_columns.update(available_donor)
        
        # Acceptor-specific Features
        if include_acceptor_features:
            acceptor_features = [
                'acceptor_diff_m1', 'acceptor_diff_m2', 'acceptor_diff_p1', 'acceptor_diff_p2',
                'acceptor_surge_ratio', 'acceptor_is_local_peak', 'acceptor_weighted_context',
                'acceptor_peak_height_ratio', 'acceptor_second_derivative', 'acceptor_signal_strength',
                'acceptor_context_diff_ratio'
            ]
            available_acceptor = [col for col in acceptor_features if col in df.columns]
            feature_columns.update(available_acceptor)
        
        # Additional derived features (catch-all for other useful features)
        if include_derived_features:
            # General statistical/entropy features - now more inclusive since duplicates are handled
            derived_keywords = ['entropy', 'variance', 'std', 'mean', 'median', 'skew', 'kurtosis', 
                               'ratio', 'diff', 'max', 'min', 'surge', 'peak', 'signal']
            derived_cols = [col for col in df.columns 
                           if any(keyword in col.lower() for keyword in derived_keywords)
                           and df[col].dtype in ['float64', 'float32', 'int64', 'int32']]
            feature_columns.update(derived_cols)
        
        # Genomic features
        if include_genomic_features:
            genomic_cols = [col for col in df.columns 
                           if any(keyword in col.lower() for keyword in ['exon', 'intron', 'gene', 'transcript', 'gc'])
                           and df[col].dtype in ['float64', 'float32', 'int64', 'int32']]
            feature_columns.update(genomic_cols)
        
        # Convert back to list and sort for consistent ordering
        feature_columns = sorted(list(feature_columns))
        
        if not feature_columns:
            return np.array([]).reshape(len(df), 0), []
        
        # Extract feature matrix
        feature_matrix = df[feature_columns].fillna(0.0).values.astype(np.float32)
        
        return feature_matrix, feature_columns
    
    @staticmethod
    def create_data_loader(
        dataset: DNASequenceDataset,
        batch_size: int = 16,
        shuffle: bool = True,
        num_workers: int = 4,
        pin_memory: bool = True
    ) -> DataLoader:
        """
        Create PyTorch DataLoader for the dataset.
        
        Parameters
        ----------
        dataset : DNASequenceDataset
            Dataset to create loader for
        batch_size : int, default 16
            Batch size
        shuffle : bool, default True
            Whether to shuffle data
        num_workers : int, default 4
            Number of worker processes
        pin_memory : bool, default True
            Whether to pin memory for GPU transfer
            
        Returns
        -------
        DataLoader
            PyTorch DataLoader
        """
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=DataUtils._collate_fn
        )
    
    @staticmethod
    def _collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Custom collate function for batching."""
        collated = {}
        
        # Standard keys
        for key in ['input_ids', 'attention_mask', 'labels']:
            if key in batch[0]:
                collated[key] = torch.stack([item[key] for item in batch])
        
        # Additional features (if present)
        if 'additional_features' in batch[0]:
            collated['additional_features'] = torch.stack([item['additional_features'] for item in batch])
        
        return collated
    
    @staticmethod
    def prepare_datasets_from_splits(
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        tokenizer: AutoTokenizer,
        config: 'ErrorModelConfig'
    ) -> Tuple[DNASequenceDataset, DNASequenceDataset, DNASequenceDataset]:
        """
        Prepare PyTorch datasets from dataframe splits.
        
        Parameters
        ----------
        train_df, val_df, test_df : pd.DataFrame
            Train, validation, and test dataframes
        tokenizer : AutoTokenizer
            Tokenizer for sequences
        config : ErrorModelConfig
            Configuration object
            
        Returns
        -------
        Tuple[DNASequenceDataset, DNASequenceDataset, DNASequenceDataset]
            (train_dataset, val_dataset, test_dataset)
        """
        datasets = []
        
        for df in [train_df, val_df, test_df]:
            # Extract sequences and labels
            sequences = df['context_sequence'].tolist()
            labels = df['label'].tolist()
            
            # Clean and validate sequences
            sequences = [DataUtils.clean_dna_sequence(seq) for seq in sequences]
            
            # Extract additional features
            additional_features, feature_names = DataUtils.extract_additional_features(
                df,
                include_base_scores=config.include_base_scores,
                include_derived_features=config.include_derived_features,
                include_genomic_features=config.include_genomic_features,
                include_context_features=config.include_context_features,
                include_donor_features=config.include_donor_features,
                include_acceptor_features=config.include_acceptor_features
            )
            
            # Create dataset
            dataset = DNASequenceDataset(
                sequences=sequences,
                labels=labels,
                tokenizer=tokenizer,
                max_length=config.max_length,
                additional_features=additional_features if additional_features.size > 0 else None,
                feature_names=feature_names
            )
            
            datasets.append(dataset)
        
        return tuple(datasets)
    
    @staticmethod
    def compute_class_weights(labels: List[int]) -> torch.Tensor:
        """
        Compute class weights for imbalanced datasets.
        
        Parameters
        ----------
        labels : List[int]
            Binary labels
            
        Returns
        -------
        torch.Tensor
            Class weights tensor
        """
        from sklearn.utils.class_weight import compute_class_weight
        
        unique_labels = np.unique(labels)
        class_weights = compute_class_weight(
            'balanced',
            classes=unique_labels,
            y=labels
        )
        
        return torch.tensor(class_weights, dtype=torch.float32)
    
    def dataframe_to_dataset(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Convert DataFrame to dataset format for workflow compatibility.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with sequences and labels
            
        Returns
        -------
        List[Dict[str, Any]]
            List of sample dictionaries
        """
        dataset = []
        
        for _, row in df.iterrows():
            sample = {
                'sequence': row.get('context_sequence', row.get('sequence', '')),
                'label': int(row.get('label', 0))
            }
            
            # Add additional features if available
            feature_cols = [col for col in df.columns if col.startswith(('donor_', 'acceptor_', 'neither_', 'entropy', 'max_ratio'))]
            if feature_cols:
                features = [row[col] for col in feature_cols if pd.notna(row[col])]
                if features:
                    sample['additional_features'] = features
            
            # Add metadata
            for col in ['gene_id', 'transcript_id', 'chrom', 'position']:
                if col in row:
                    sample[col] = row[col]
            
            dataset.append(sample)
        
        return dataset
    
    @staticmethod
    def log_dataset_statistics(dataset: DNASequenceDataset, name: str = "Dataset"):
        """Log statistics about the dataset."""
        logger = logging.getLogger(__name__)
        
        # Basic statistics
        logger.info(f"{name} statistics:")
        logger.info(f"  - Total samples: {len(dataset)}")
        
        # Label distribution
        labels = [dataset[i]['labels'].item() for i in range(len(dataset))]
        unique, counts = np.unique(labels, return_counts=True)
        label_dist = dict(zip(unique, counts))
        logger.info(f"  - Label distribution: {label_dist}")
        
        # Sequence length statistics
        seq_lengths = [len(dataset.sequences[i]) for i in range(len(dataset))]
        logger.info(f"  - Sequence length: min={min(seq_lengths)}, max={max(seq_lengths)}, mean={np.mean(seq_lengths):.1f}")
        
        # Additional features
        if dataset.additional_features is not None:
            logger.info(f"  - Additional features: {dataset.additional_features.shape[1]} features")
            logger.info(f"  - Feature names: {dataset.feature_names}")
        else:
            logger.info("  - No additional features")
