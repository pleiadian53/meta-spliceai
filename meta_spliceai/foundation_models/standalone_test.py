import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Create a minimal version of the DilatedCNN model for testing
class DilatedCNN(nn.Module):
    def __init__(self, seq_length=100, num_filters=16, num_conv_layers=2):
        super(DilatedCNN, self).__init__()
        
        # Input shape: (batch_size, seq_length, 4)
        self.seq_length = seq_length
        
        # Initial embedding layer (1x1 convolution)
        self.embedding = nn.Conv1d(4, num_filters, kernel_size=1)
        
        # Create dilated convolution blocks with skip connections
        self.conv_blocks = nn.ModuleList()
        self.skip_connections = nn.ModuleList()
        
        # Add dilated convolution blocks with increasing dilation rates
        for i in range(num_conv_layers):
            # Dilation rate increases exponentially (1, 2, 4, 8, ...)
            dilation = 2 ** i
            
            # Add block with two dilated convolutions and skip connection
            self.conv_blocks.append(nn.ModuleDict({
                'conv1': nn.Conv1d(
                    num_filters, num_filters, kernel_size=3,
                    padding=dilation, dilation=dilation
                ),
                'bn1': nn.BatchNorm1d(num_filters),
                'conv2': nn.Conv1d(
                    num_filters, num_filters, kernel_size=3,
                    padding=dilation, dilation=dilation
                ),
                'bn2': nn.BatchNorm1d(num_filters),
                'dropout': nn.Dropout(0.1)
            }))
            
            # Skip connection to output layer (1x1 conv to adjust dimensions if needed)
            self.skip_connections.append(
                nn.Conv1d(num_filters, num_filters, kernel_size=1)
            )
        
        # Output layer for all three classes (donor, acceptor, neither)
        self.output_layer = nn.Conv1d(num_filters, 3, kernel_size=1)
    
    def forward(self, x):
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

def test_model_output():
    """Test if the model produces the expected output format."""
    print("\n=== Testing DilatedCNN model ===")
    
    # Create a small model instance
    seq_length = 100
    model = DilatedCNN(seq_length=seq_length, num_filters=16, num_conv_layers=2)
    
    # Create a batch of random one-hot encoded sequences
    batch_size = 2
    x = torch.zeros(batch_size, seq_length, 4)
    # Fill with random one-hot encoded nucleotides
    for i in range(batch_size):
        for j in range(seq_length):
            idx = np.random.randint(0, 4)
            x[i, j, idx] = 1.0
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(x)
    
    # Check output shape
    expected_shape = (batch_size, seq_length, 3)
    actual_shape = output.shape
    print(f"Output shape: {actual_shape} (expected: {expected_shape})")
    assert actual_shape == expected_shape, f"Output shape {actual_shape} != expected {expected_shape}"
    
    # Check if probabilities sum to 1 (approximately) for each position
    sum_probs = torch.sum(output, dim=2)
    all_close_to_one = torch.allclose(sum_probs, torch.ones_like(sum_probs), rtol=1e-5)
    print(f"Probabilities sum to 1: {all_close_to_one}")
    assert all_close_to_one, "Probabilities don't sum to 1 for each position"
    
    # Check if values are in valid range [0, 1]
    valid_range = torch.all((output >= 0) & (output <= 1))
    print(f"All probabilities in range [0, 1]: {valid_range}")
    assert valid_range, "Some probabilities are outside the [0, 1] range"
    
    # Check a sample of outputs
    print("Sample output (first sequence, first 5 positions):")
    for i in range(min(5, seq_length)):
        probs = output[0, i].detach().numpy()
        print(f"Position {i}: Donor={probs[0]:.4f}, Acceptor={probs[1]:.4f}, Neither={probs[2]:.4f}, Sum={sum(probs):.4f}")
    
    print("DilatedCNN test PASSED\n")
    return model

def main():
    """Run the standalone test."""
    print("Starting model test...")
    test_model_output()
    print("All tests completed!")

if __name__ == "__main__":
    main()
