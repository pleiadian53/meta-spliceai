"""
Models for splice site prediction.

This module contains PyTorch implementations of various neural network architectures
for splice site prediction, including:
1. Dilated CNN
2. Transformer-based models
3. HyenaDNA model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Tuple, Dict, Optional, Union


class DilatedCNN(nn.Module):
    """
    Dilated CNN model for splice site prediction.
    
    The model uses dilated convolutions to capture both local patterns
    and long-range dependencies in genomic sequences. This architecture
    is inspired by SpliceAI and produces per-nucleotide predictions for
    donor and acceptor splice sites.
    
    Args:
        seq_length: Length of the input sequence
        num_filters: Number of convolutional filters
        kernel_size: Size of the convolutional kernel
        dilation_rates: List of dilation rates for the convolutional layers
        dropout_rate: Dropout rate for regularization
    """
    
    def __init__(
        self,
        seq_length: int = 10000,
        num_filters: int = 256,
        kernel_size: int = 11,
        dilation_rates: List[int] = [1, 2, 4, 8, 16, 32],
        dropout_rate: float = 0.2
    ):
        super(DilatedCNN, self).__init__()
        
        # Input shape: (batch_size, 4, seq_length) - PyTorch uses channels-first format
        self.seq_length = seq_length
        
        # Input embedding: convert one-hot encoded nucleotides to embedding
        self.embedding = nn.Conv1d(4, num_filters, kernel_size=1)
        
        # Create dilated convolutional blocks with residual connections
        self.conv_blocks = nn.ModuleList()
        self.skip_connections = nn.ModuleList()
        
        for dilation_rate in dilation_rates:
            # Calculate padding to maintain sequence length
            padding = (kernel_size - 1) * dilation_rate // 2
            
            # Each block contains two conv layers with a residual connection
            block = nn.ModuleDict({
                'conv1': nn.Conv1d(
                    in_channels=num_filters,
                    out_channels=num_filters,
                    kernel_size=kernel_size,
                    dilation=dilation_rate,
                    padding=padding
                ),
                'bn1': nn.BatchNorm1d(num_filters),
                'conv2': nn.Conv1d(
                    in_channels=num_filters,
                    out_channels=num_filters,
                    kernel_size=kernel_size,
                    dilation=dilation_rate,
                    padding=padding
                ),
                'bn2': nn.BatchNorm1d(num_filters),
                'dropout': nn.Dropout(dropout_rate)
            })
            self.conv_blocks.append(block)
            
            # Skip connection to output layer (1x1 conv to adjust dimensions if needed)
            self.skip_connections.append(
                nn.Conv1d(num_filters, num_filters, kernel_size=1)
            )
        
        # Output layer for all three classes (donor, acceptor, neither)
        self.output_layer = nn.Conv1d(num_filters, 3, kernel_size=1)
    
    def forward(self, x):
        """
        Forward pass of the model.
        
        Args:
            x: Input tensor of shape (batch_size, seq_length, 4)
            
        Returns:
            Tensor of shape (batch_size, seq_length, 3) where the last dimension
            represents [donor_prob, acceptor_prob, neither_prob] for each position.
            The probabilities sum to 1 for each position.
        """
        # Input shape: (batch_size, seq_length, 4)
        # Convert to channels-first format: (batch_size, 4, seq_length)
        x = x.permute(0, 2, 1)
        
        # Initial embedding
        x = self.embedding(x)
        
        # Apply dilated convolution blocks with skip connections
        skip_outputs = []
        for block in self.conv_blocks:
            # Residual connection
            residual = x
            
            # First conv layer with activation
            x = F.relu(block['bn1'](block['conv1'](x)))
            
            # Second conv layer without activation (applied after residual)
            x = block['bn2'](block['conv2'](x))
            
            # Add residual connection
            x = F.relu(x + residual)
            
            # Apply dropout
            x = block['dropout'](x)
            
            # Store output for skip connection
            skip_outputs.append(x)
        
        # Combine skip connections
        combined_skip = sum(skip_conn(output) for skip_conn, output 
                          in zip(self.skip_connections, skip_outputs))
        
        # Get logits for all three classes
        logits = self.output_layer(combined_skip)  # Shape: (batch_size, 3, seq_length)
        
        # Permute to (batch_size, seq_length, 3) and apply softmax across classes
        logits = logits.permute(0, 2, 1)  # Shape: (batch_size, seq_length, 3)
        probs = F.softmax(logits, dim=2)  # Apply softmax to get probabilities that sum to 1
        
        return probs


class PositionalEncoding(nn.Module):
    """
    Positional encoding for the Transformer model.
    
    This class implements the sinusoidal positional encoding described in
    "Attention Is All You Need" (Vaswani et al., 2017).
    
    Args:
        d_model: Dimension of the model
        max_len: Maximum sequence length
        dropout: Dropout rate
    """
    
    def __init__(self, d_model: int, max_len: int = 10000, dropout: float = 0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        # Apply sin/cos functions
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension and register as buffer (not a parameter)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """Forward pass of the positional encoding."""
        # Add positional encoding to input
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class DNATransformer(nn.Module):
    """
    Transformer-based model for splice site prediction.
    
    This model uses a Transformer encoder to process genomic sequences and produces
    per-nucleotide predictions for donor and acceptor splice sites.
    
    Args:
        seq_length: Length of the input sequence
        embed_dim: Dimension of the embedding
        num_heads: Number of attention heads
        ff_dim: Dimension of the feed-forward network
        num_transformer_blocks: Number of transformer blocks
        dropout_rate: Dropout rate for regularization
    """
    
    def __init__(
        self,
        seq_length: int = 10000,
        embed_dim: int = 64,
        num_heads: int = 8,
        ff_dim: int = 128,
        num_transformer_blocks: int = 6,
        dropout_rate: float = 0.1
    ):
        super(DNATransformer, self).__init__()
        
        # Input shape: (batch_size, seq_length, 4)
        self.seq_length = seq_length
        
        # Embedding layer to convert one-hot encoding to embedding
        self.embedding = nn.Linear(4, embed_dim)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(
            d_model=embed_dim,
            max_len=seq_length,
            dropout=dropout_rate
        )
        
        # Transformer encoder layers
        self.transformer_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=ff_dim,
                dropout=dropout_rate,
                batch_first=True
            )
            for _ in range(num_transformer_blocks)
        ])
        
        # Skip connections from each transformer block
        self.skip_connections = nn.ModuleList([
            nn.Linear(embed_dim, embed_dim) 
            for _ in range(num_transformer_blocks)
        ])
        
        # Output projection layer for all three classes (donor, acceptor, neither)
        self.output_projection = nn.Linear(embed_dim, 3)
        
    def forward(self, x):
        """
        Forward pass of the model.
        
        Args:
            x: Input tensor of shape (batch_size, seq_length, 4)
            
        Returns:
            Tensor of shape (batch_size, seq_length, 3) where the last dimension
            represents [donor_prob, acceptor_prob, neither_prob] for each position.
            The probabilities sum to 1 for each position.
        """
        # Input shape: (batch_size, seq_length, 4)
        batch_size, seq_len, _ = x.shape
        
        # Apply embedding
        x = self.embedding(x)
        
        # Apply positional encoding
        x = self.positional_encoding(x)
        
        # Store outputs for skip connections
        skip_outputs = []
        
        # Apply transformer blocks
        for transformer_block in self.transformer_blocks:
            # Apply transformer block
            x_residual = x
            x = transformer_block(x)
            x = x + x_residual  # Residual connection
            skip_outputs.append(x)
        
        # Combine skip connections
        combined_features = sum(skip_conn(output) for skip_conn, output 
                             in zip(self.skip_connections, skip_outputs))
        
        # Apply output projection for all three classes
        logits = self.output_projection(combined_features)  # Shape: (batch_size, seq_length, 3)
        
        # Apply softmax to get probabilities that sum to 1
        probs = F.softmax(logits, dim=2)
        
        return probs


class HyenaFilter(nn.Module):
    """
    Hyena filter for long-range dependencies in genomic sequences.
    
    Implements a simplified version of the HyenaDNA filter as described in
    "HyenaDNA: Long-Range Genomic Sequence Modeling at Single Nucleotide Resolution"
    (Nguyen et al., 2023).
    
    Args:
        d_model: Dimension of the model
        filter_order: Order of the filter
        seq_len: Maximum sequence length
    """
    
    def __init__(self, d_model: int, filter_order: int = 4, seq_len: int = 10000):
        super(HyenaFilter, self).__init__()
        
        self.d_model = d_model
        self.filter_order = filter_order
        self.seq_len = seq_len
        
        # Learnable projections
        self.projection = nn.Parameter(torch.randn(filter_order, d_model))
        
        # Initialize filter coefficients
        self.filter_coeffs = nn.Parameter(torch.randn(filter_order, d_model))
        
        # Make filters more long-range
        for i in range(filter_order):
            self.filter_coeffs.data[i] = self.filter_coeffs.data[i] * 0.5 ** i
    
    def forward(self, x):
        """Forward pass of the Hyena filter."""
        # Input shape: (batch_size, seq_len, d_model)
        batch_size, seq_len, d_model = x.shape
        
        # Compute filter responses for each order
        responses = []
        
        # Apply filters of different orders
        for i in range(self.filter_order):
            # Project input
            proj = F.linear(x, self.projection[i].unsqueeze(0))
            
            # Apply exponential decay filter
            coeffs = torch.exp(self.filter_coeffs[i])
            
            # Simple implementation of causal convolution with exponential decay
            output = torch.zeros_like(proj)
            for t in range(seq_len):
                for t_prev in range(t + 1):
                    decay = coeffs ** (t - t_prev)
                    output[:, t, :] += proj[:, t_prev, :] * decay
            
            responses.append(output)
        
        # Sum responses from all filters
        return sum(responses)


class HyenaDNA(nn.Module):
    """
    HyenaDNA model for splice site prediction.
    
    A simplified implementation of the HyenaDNA architecture for
    processing long genomic sequences with single nucleotide resolution.
    Produces per-nucleotide predictions for donor and acceptor splice sites.
    
    Args:
        seq_length: Length of the input sequence
        embed_dim: Dimension of the embedding
        filter_order: Order of the Hyena filter
        num_heads: Number of attention heads
        num_layers: Number of model layers
        droppath_rate: Drop path rate for regularization
    """
    
    def __init__(
        self,
        seq_length: int = 10000,
        embed_dim: int = 64,
        filter_order: int = 4,
        num_heads: int = 8,
        num_layers: int = 4,
        droppath_rate: float = 0.1
    ):
        super(HyenaDNA, self).__init__()
        
        # Input shape: (batch_size, seq_length, 4)
        self.seq_length = seq_length
        
        # Embedding layer to convert one-hot encoding to embedding
        self.embedding = nn.Linear(4, embed_dim)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(
            d_model=embed_dim,
            max_len=seq_length,
            dropout=droppath_rate
        )
        
        # HyenaDNA layers
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.ModuleDict({
                # Layer normalization
                'norm1': nn.LayerNorm(embed_dim),
                'norm2': nn.LayerNorm(embed_dim),
                
                # HyenaFilter for sequence modeling
                'hyena_filter': HyenaFilter(embed_dim, filter_order, seq_length),
                
                # Feed forward network
                'ffn': nn.Sequential(
                    nn.Linear(embed_dim, embed_dim * 4),
                    nn.GELU(),
                    nn.Linear(embed_dim * 4, embed_dim),
                    nn.Dropout(droppath_rate)
                )
            }))
        
        # Skip connections from each layer
        self.skip_connections = nn.ModuleList([
            nn.Linear(embed_dim, embed_dim) 
            for _ in range(num_layers)
        ])
        
        # Output projection for all three classes (donor, acceptor, neither)
        self.output_projection = nn.Linear(embed_dim, 3)
        
    def forward(self, x):
        """
        Forward pass of the model.
        
        Args:
            x: Input tensor of shape (batch_size, seq_length, 4)
            
        Returns:
            Tensor of shape (batch_size, seq_length, 3) where the last dimension
            represents [donor_prob, acceptor_prob, neither_prob] for each position.
            The probabilities sum to 1 for each position.
        """
        # Input shape: (batch_size, seq_length, 4)
        
        # Apply embedding
        x = self.embedding(x)
        
        # Apply positional encoding
        x = self.positional_encoding(x)
        
        # Store outputs for skip connections
        skip_outputs = []
        
        # Apply HyenaDNA layers
        for layer in self.layers:
            # First sub-layer: HyenaFilter with residual connection
            residual = x
            x = layer['norm1'](x)
            x = layer['hyena_filter'](x)
            x = x + residual  # Residual connection
            
            # Second sub-layer: FFN with residual connection
            residual = x
            x = layer['norm2'](x)
            x = layer['ffn'](x)
            x = x + residual  # Residual connection
            
            # Store output for skip connection
            skip_outputs.append(x)
        
        # Combine skip connections
        combined_features = sum(skip_conn(output) for skip_conn, output 
                              in zip(self.skip_connections, skip_outputs))
        
        # Apply output projection for all three classes
        logits = self.output_projection(combined_features)  # Shape: (batch_size, seq_length, 3)
        
        # Apply softmax to get probabilities that sum to 1
        probs = F.softmax(logits, dim=2)
        
        return probs


def create_model(
    model_type: str,
    **kwargs
):
    """
    Create a model for splice site prediction.
    
    Args:
        model_type: Type of model to create ('cnn', 'transformer', or 'hyenadna')
        **kwargs: Additional arguments to pass to the model constructor
    
    Returns:
        A PyTorch model for splice site prediction
    
    Raises:
        ValueError: If an invalid model type is specified
    """
    # Create the specified model type
    if model_type.lower() == 'cnn':
        return DilatedCNN(**kwargs)
    elif model_type.lower() == 'transformer':
        return DNATransformer(**kwargs)
    elif model_type.lower() == 'hyenadna':
        return HyenaDNA(**kwargs)
    else:
        valid_types = ['cnn', 'transformer', 'hyenadna']
        raise ValueError(f"Invalid model type: {model_type}. Must be one of {valid_types}")
