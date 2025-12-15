# Splice Site Prediction Model Architecture Instructions

This document provides detailed instructions for recreating the neural network models for splice site prediction as defined in the MetaSpliceAI project. These models are designed to output three probabilities (donor, acceptor, neither) for each nucleotide position.

## Model Overview

The project implements three different model architectures for splice site prediction:

1. **DilatedCNN**: A convolutional neural network with dilated convolutions for capturing long-range dependencies
2. **DNATransformer**: A transformer-based model adapted for processing DNA sequences 
3. **HyenaDNA**: A model based on the HyenaDNA architecture for long-range genomic sequence modeling

All models share these common characteristics:
- Input: One-hot encoded DNA sequences of shape (batch_size, seq_length, 4)
- Output: Probabilities for each nucleotide position of shape (batch_size, seq_length, 3) where the last dimension represents [donor_prob, acceptor_prob, neither_prob]
- The output probabilities sum to 1 at each position (softmax activation)
- Skip connections to improve gradient flow during training

## 1. DilatedCNN Architecture

### Components
- **Input**: One-hot encoded DNA sequence (batch_size, seq_length, 4)
- **Embedding**: 1x1 convolution to project input to hidden dimension
- **Dilated Convolution Blocks**: Series of dilated convolution blocks with increasing dilation rates
- **Skip Connections**: From each block to the output layer
- **Output Layer**: 1x1 convolution to produce logits for three classes

### Detailed Architecture
```python
class DilatedCNN(nn.Module):
    def __init__(
        self,
        seq_length: int = 10000,
        num_filters: int = 256,
        kernel_size: int = 11,
        dilation_rates: List[int] = [1, 2, 4, 8, 16, 32],
        dropout_rate: float = 0.2
    ):
        super(DilatedCNN, self).__init__()
        
        # Input embedding
        self.embedding = nn.Conv1d(4, num_filters, kernel_size=1)
        
        # Dilated convolution blocks
        self.conv_blocks = nn.ModuleList()
        self.skip_connections = nn.ModuleList()
        
        # Add blocks with increasing dilation rates
        for dilation in dilation_rates:
            self.conv_blocks.append(nn.ModuleDict({
                'conv1': nn.Conv1d(
                    num_filters, num_filters, kernel_size=kernel_size,
                    padding=dilation * (kernel_size // 2), dilation=dilation
                ),
                'bn1': nn.BatchNorm1d(num_filters),
                'conv2': nn.Conv1d(
                    num_filters, num_filters, kernel_size=kernel_size,
                    padding=dilation * (kernel_size // 2), dilation=dilation
                ),
                'bn2': nn.BatchNorm1d(num_filters),
                'dropout': nn.Dropout(dropout_rate)
            }))
            
            # Skip connection
            self.skip_connections.append(
                nn.Conv1d(num_filters, num_filters, kernel_size=1)
            )
        
        # Output layer for all three classes (donor, acceptor, neither)
        self.output_layer = nn.Conv1d(num_filters, 3, kernel_size=1)
    
    def forward(self, x):
        # Convert to channels-first: (batch_size, 4, seq_length)
        x = x.permute(0, 2, 1)
        
        # Initial embedding
        x = self.embedding(x)
        
        # Apply dilated convolution blocks
        skip_outputs = []
        for block in self.conv_blocks:
            # Residual connection
            residual = x
            
            # First conv with activation
            x = F.relu(block['bn1'](block['conv1'](x)))
            
            # Second conv without activation yet
            x = block['bn2'](block['conv2'](x))
            
            # Add residual and activate
            x = F.relu(x + residual)
            
            # Apply dropout
            x = block['dropout'](x)
            
            # Store for skip connection
            skip_outputs.append(x)
        
        # Combine skip connections
        combined_skip = sum(skip_conn(output) for skip_conn, output 
                          in zip(self.skip_connections, skip_outputs))
        
        # Get logits for all three classes
        logits = self.output_layer(combined_skip)  # (batch_size, 3, seq_length)
        
        # Reshape and apply softmax
        logits = logits.permute(0, 2, 1)  # (batch_size, seq_length, 3)
        probs = F.softmax(logits, dim=2)
        
        return probs
```

## 2. DNATransformer Architecture

### Components
- **Input**: One-hot encoded DNA sequence (batch_size, seq_length, 4)
- **Embedding**: Linear projection from 4 to embed_dim
- **Positional Encoding**: Sinusoidal position embeddings
- **Transformer Encoder Blocks**: Multiple transformer encoder layers
- **Skip Connections**: From each transformer block to the output layer
- **Output Layer**: Linear layer to produce logits for three classes

### Detailed Architecture
```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=10000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class DNATransformer(nn.Module):
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
        
        # Embedding layer
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
        
        # Skip connections
        self.skip_connections = nn.ModuleList([
            nn.Linear(embed_dim, embed_dim) 
            for _ in range(num_transformer_blocks)
        ])
        
        # Output projection for all three classes
        self.output_projection = nn.Linear(embed_dim, 3)
    
    def forward(self, x):
        # Apply embedding
        x = self.embedding(x)
        
        # Apply positional encoding
        x = self.positional_encoding(x)
        
        # Store outputs for skip connections
        skip_outputs = []
        
        # Apply transformer blocks
        for transformer_block in self.transformer_blocks:
            # Apply transformer with residual
            x_residual = x
            x = transformer_block(x)
            x = x + x_residual
            skip_outputs.append(x)
        
        # Combine skip connections
        combined_features = sum(skip_conn(output) for skip_conn, output 
                             in zip(self.skip_connections, skip_outputs))
        
        # Apply output projection
        logits = self.output_projection(combined_features)
        
        # Apply softmax
        probs = F.softmax(logits, dim=2)
        
        return probs
```

## 3. HyenaDNA Architecture

### Components
- **Input**: One-hot encoded DNA sequence (batch_size, seq_length, 4)
- **Embedding**: Linear projection from 4 to embed_dim
- **Positional Encoding**: Sinusoidal position embeddings
- **HyenaDNA Layers**: Series of layers with HyenaFilter and feed-forward network
- **Skip Connections**: From each layer to the output
- **Output Layer**: Linear projection to produce logits for three classes

### Detailed Architecture
```python
class HyenaFilter(nn.Module):
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
        # Input: (batch_size, seq_len, d_model)
        batch_size, seq_len, d_model = x.shape
        
        # Apply filters of different orders
        responses = []
        for i in range(self.filter_order):
            # Project input
            proj = F.linear(x, self.projection[i].unsqueeze(0))
            
            # Apply exponential decay filter
            coeffs = torch.exp(self.filter_coeffs[i])
            
            # Causal convolution with exponential decay
            output = torch.zeros_like(proj)
            for t in range(seq_len):
                for t_prev in range(t + 1):
                    decay = coeffs ** (t - t_prev)
                    output[:, t, :] += proj[:, t_prev, :] * decay
            
            responses.append(output)
        
        # Sum responses
        return sum(responses)

class HyenaDNA(nn.Module):
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
        
        # Embedding layer
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
                
                # HyenaFilter
                'hyena_filter': HyenaFilter(embed_dim, filter_order, seq_length),
                
                # Feed forward network
                'ffn': nn.Sequential(
                    nn.Linear(embed_dim, embed_dim * 4),
                    nn.GELU(),
                    nn.Linear(embed_dim * 4, embed_dim),
                    nn.Dropout(droppath_rate)
                )
            }))
        
        # Skip connections
        self.skip_connections = nn.ModuleList([
            nn.Linear(embed_dim, embed_dim) 
            for _ in range(num_layers)
        ])
        
        # Output projection for all three classes
        self.output_projection = nn.Linear(embed_dim, 3)
    
    def forward(self, x):
        # Apply embedding
        x = self.embedding(x)
        
        # Apply positional encoding
        x = self.positional_encoding(x)
        
        # Store outputs for skip connections
        skip_outputs = []
        
        # Apply HyenaDNA layers
        for layer in self.layers:
            # First sub-layer: HyenaFilter with residual
            residual = x
            x = layer['norm1'](x)
            x = layer['hyena_filter'](x)
            x = x + residual
            
            # Second sub-layer: FFN with residual
            residual = x
            x = layer['norm2'](x)
            x = layer['ffn'](x)
            x = x + residual
            
            # Store for skip connection
            skip_outputs.append(x)
        
        # Combine skip connections
        combined_features = sum(skip_conn(output) for skip_conn, output 
                              in zip(self.skip_connections, skip_outputs))
        
        # Apply output projection
        logits = self.output_projection(combined_features)
        
        # Apply softmax
        probs = F.softmax(logits, dim=2)
        
        return probs
```

## Model Factory Function

```python
def create_model(model_type: str, **kwargs):
    """
    Create a model for splice site prediction.
    
    Args:
        model_type: Type of model to create ('cnn', 'transformer', or 'hyenadna')
        **kwargs: Additional arguments to pass to the model constructor
    
    Returns:
        A PyTorch model for splice site prediction
    """
    if model_type.lower() == 'cnn':
        return DilatedCNN(**kwargs)
    elif model_type.lower() == 'transformer':
        return DNATransformer(**kwargs)
    elif model_type.lower() == 'hyenadna':
        return HyenaDNA(**kwargs)
    else:
        valid_types = ['cnn', 'transformer', 'hyenadna']
        raise ValueError(f"Invalid model type: {model_type}. Must be one of {valid_types}")
```

## Key Design Considerations

1. **Three-class Probability Output**: Each model outputs probabilities for three classes (donor, acceptor, neither) for each nucleotide position. The probabilities sum to 1 for each position.

2. **Skip Connections**: All models use skip connections to improve gradient flow during training.

3. **Residual Connections**: Each model incorporates residual connections to facilitate training of deeper networks.

4. **Common API**: All models have a consistent API, taking the same input format and producing the same output format.

5. **Flexible Architecture**: Models can be configured with different hyperparameters to adjust capacity and performance.

## Implementation Notes

- The HyenaFilter implementation is a simplified version of the original HyenaDNA architecture
- The DNATransformer uses vanilla transformer encoder layers from PyTorch
- All models are designed to work on CPU or GPU without code changes
