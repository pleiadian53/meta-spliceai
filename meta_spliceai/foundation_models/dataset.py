"""
PyTorch datasets for splice site prediction models.

This module provides PyTorch dataset implementations for training and evaluating
the splice site prediction models with per-nucleotide predictions.
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class SpliceSiteDataset(Dataset):
    """PyTorch dataset for splice site prediction with per-nucleotide labels."""
    
    def __init__(self, sequences, donor_labels=None, acceptor_labels=None, transform=None):
        """
        Initialize the dataset.
        
        Args:
            sequences (list): List of DNA sequences
            donor_labels (list, optional): Per-nucleotide labels for donor sites (0/1)
            acceptor_labels (list, optional): Per-nucleotide labels for acceptor sites (0/1)
            transform (callable, optional): Optional transform to apply to the data
        """
        self.sequences = sequences
        self.donor_labels = donor_labels
        self.acceptor_labels = acceptor_labels
        self.transform = transform
        
        # Auto-generate empty labels if not provided
        if donor_labels is None and sequences:
            self.donor_labels = [np.zeros(len(seq)) for seq in sequences]
        
        if acceptor_labels is None and sequences:
            self.acceptor_labels = [np.zeros(len(seq)) for seq in sequences]
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        """Get item from dataset."""
        seq = self.sequences[idx]
        donor_label = self.donor_labels[idx]
        acceptor_label = self.acceptor_labels[idx]
        
        # One-hot encode the sequence
        encoded_seq = self._one_hot_encode(seq)
        
        # Apply transformations if needed
        if self.transform:
            encoded_seq = self.transform(encoded_seq)
        
        # Convert to PyTorch tensors
        x = torch.tensor(encoded_seq, dtype=torch.float32)
        donor_y = torch.tensor(donor_label, dtype=torch.float32)
        acceptor_y = torch.tensor(acceptor_label, dtype=torch.float32)
        
        return x, (donor_y, acceptor_y)
    
    def _one_hot_encode(self, sequence):
        """
        One-hot encode a DNA sequence.
        
        Args:
            sequence (str): DNA sequence
            
        Returns:
            np.ndarray: One-hot encoded sequence (shape: [seq_length, 4])
        """
        # Mapping of nucleotides to indices
        nuc_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
        
        # Initialize encoded array
        encoded = np.zeros((len(sequence), 4), dtype=np.float32)
        
        # Fill in encoded values
        for i, nuc in enumerate(sequence.upper()):
            if nuc in nuc_map:
                encoded[i, nuc_map[nuc]] = 1.0
        
        return encoded


class SpliceDataProcessor:
    """Process and convert genomic data to PyTorch datasets for splice site prediction."""
    
    def __init__(self, window_size=400, label_smoothing=0.1):
        """
        Initialize the data processor.
        
        Args:
            window_size (int): Size of window around splice sites for per-nucleotide labeling
            label_smoothing (float): Amount of label smoothing to apply
        """
        self.window_size = window_size
        self.label_smoothing = label_smoothing
    
    def prepare_from_windows(self, windows_df, context_length=10000):
        """
        Prepare PyTorch dataset from genomic windows.
        
        Args:
            windows_df (pd.DataFrame): DataFrame with splice site windows
            context_length (int): Length to pad/truncate sequences to
            
        Returns:
            SpliceSiteDataset: PyTorch dataset for training
        """
        # Extract sequences
        sequences = windows_df['sequence'].tolist()
        
        # Initialize empty label arrays
        donor_labels = []
        acceptor_labels = []
        
        # Process each sequence to generate donor and acceptor labels
        for i, row in windows_df.iterrows():
            sequence = row['sequence']
            seq_len = len(sequence)
            
            # Ensure sequence is exactly context_length
            if seq_len < context_length:
                # Pad with N's
                sequence = sequence + 'N' * (context_length - seq_len)
            elif seq_len > context_length:
                # Truncate
                sequence = sequence[:context_length]
            
            # Create empty label arrays
            donor_label = np.zeros(context_length)
            acceptor_label = np.zeros(context_length)
            
            # Set the label arrays
            if 'center_pos' in row and 'type' in row:
                center_pos = row['center_pos']
                
                # Make sure center_pos is within bounds
                if 0 <= center_pos < context_length:
                    # For positive examples, set the appropriate label array
                    if row.get('label', 0) == 1:
                        # Use a window around the center position for smoother gradients
                        half_window = self.window_size // 2
                        start_pos = max(0, center_pos - half_window)
                        end_pos = min(context_length, center_pos + half_window + 1)
                        
                        # Create gaussian-like distribution centered at splice site
                        positions = np.arange(start_pos, end_pos)
                        values = np.exp(-0.5 * ((positions - center_pos) / (half_window / 3))**2)
                        
                        # Normalize to max value of 1.0
                        values = values / np.max(values)
                        
                        # Apply label smoothing
                        if self.label_smoothing > 0:
                            values = values * (1.0 - self.label_smoothing) + self.label_smoothing
                        
                        # Set values in the appropriate label array
                        if row['type'] == 'donor':
                            donor_label[start_pos:end_pos] = values
                        elif row['type'] == 'acceptor':
                            acceptor_label[start_pos:end_pos] = values
            
            donor_labels.append(donor_label)
            acceptor_labels.append(acceptor_label)
            
            # Update sequence
            sequences[i] = sequence
        
        # Create PyTorch dataset
        dataset = SpliceSiteDataset(sequences, donor_labels, acceptor_labels)
        logger.info(f"Created dataset with {len(dataset)} samples")
        
        return dataset
    
    def prepare_from_genomic_processor(self, genomic_processor, data_dict, batch_size=32, shuffle=True):
        """
        Prepare PyTorch datasets from GenomicDataProcessor output.
        
        Args:
            genomic_processor (GenomicDataProcessor): The genomic data processor
            data_dict (dict): Dictionary with train, validation, and test dataframes
            batch_size (int): Batch size for DataLoader
            shuffle (bool): Whether to shuffle the datasets
            
        Returns:
            dict: Dictionary with train, validation, and test PyTorch datasets
        """
        datasets = {}
        
        for split, df in data_dict.items():
            datasets[split] = self.prepare_from_windows(df)
        
        return datasets
    
    @staticmethod
    def convert_to_torch_dataloader(dataset, batch_size=32, shuffle=True):
        """
        Convert a PyTorch dataset to DataLoader.
        
        Args:
            dataset (SpliceSiteDataset): PyTorch dataset
            batch_size (int): Batch size
            shuffle (bool): Whether to shuffle the data
            
        Returns:
            DataLoader: PyTorch DataLoader
        """
        return torch.utils.data.DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=shuffle,
            num_workers=4,
            pin_memory=True
        )
