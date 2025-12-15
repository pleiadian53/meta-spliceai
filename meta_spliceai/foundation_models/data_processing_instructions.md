# Splice Site Data Processing Instructions

This document provides detailed instructions for implementing the data processing pipeline for splice site prediction as used in the MetaSpliceAI project. This covers data preparation, encoding, and dataset creation.

## Overview

The data processing pipeline transforms raw DNA sequences and splice site annotations into a format suitable for training neural network models. The pipeline includes:

1. **DNA Sequence Encoding**: Converting nucleotide sequences to one-hot encoding
2. **Target Label Generation**: Creating labels for donor, acceptor, and neither splice sites
3. **Dataset Creation**: Building PyTorch datasets and dataloaders
4. **Data Augmentation**: Implementing techniques to enhance model performance and generalization

## 1. DNA Sequence Encoding

### One-Hot Encoding of DNA Sequences

DNA sequences are represented as one-hot encoded vectors where each nucleotide (A, C, G, T) is encoded as a binary vector:
- A: [1, 0, 0, 0]
- C: [0, 1, 0, 0]
- G: [0, 0, 1, 0]
- T: [0, 0, 0, 1]

Unknown or non-standard nucleotides (N) can be encoded as [0.25, 0.25, 0.25, 0.25].

```python
def one_hot_encode(seq):
    """
    Convert a DNA sequence to one-hot encoding.
    
    Args:
        seq: String of DNA sequence
        
    Returns:
        numpy array of shape (len(seq), 4)
    """
    mapping = {
        'A': [1, 0, 0, 0],
        'C': [0, 1, 0, 0],
        'G': [0, 0, 1, 0],
        'T': [0, 0, 0, 1],
        'N': [0.25, 0.25, 0.25, 0.25]
    }
    
    # Create empty array
    one_hot = np.zeros((len(seq), 4), dtype=np.float32)
    
    # Fill with one-hot encodings
    for i, nucleotide in enumerate(seq.upper()):
        if nucleotide in mapping:
            one_hot[i] = mapping[nucleotide]
        else:
            # Default to N for any non-standard nucleotide
            one_hot[i] = mapping['N']
            
    return one_hot
```

## 2. Target Label Generation

### Creating Three-Class Labels

For each nucleotide position, we create a one-hot encoded vector representing three possible classes:
- Donor site: [1, 0, 0]
- Acceptor site: [0, 1, 0]
- Neither (non-splice site): [0, 0, 1]

```python
def create_splice_labels(seq_length, donor_positions, acceptor_positions):
    """
    Create one-hot encoded labels for splice sites.
    
    Args:
        seq_length: Length of the sequence
        donor_positions: List of donor splice site positions (0-indexed)
        acceptor_positions: List of acceptor splice site positions (0-indexed)
        
    Returns:
        numpy array of shape (seq_length, 3) with one-hot encoded labels
    """
    # Initialize with all "neither" class
    labels = np.zeros((seq_length, 3), dtype=np.float32)
    labels[:, 2] = 1  # Set "neither" as default
    
    # Set donor positions
    for pos in donor_positions:
        if 0 <= pos < seq_length:
            labels[pos] = [1, 0, 0]
    
    # Set acceptor positions
    for pos in acceptor_positions:
        if 0 <= pos < seq_length:
            labels[pos] = [0, 1, 0]
            
    return labels
```

### Handling Overlapping Splice Sites

In rare cases where a position could be both a donor and acceptor site (e.g., in very short introns or alternative splicing), a priority rule should be applied:

```python
def resolve_overlapping_sites(donor_positions, acceptor_positions):
    """
    Resolve positions that appear in both donor and acceptor lists.
    
    Args:
        donor_positions: List of donor splice site positions
        acceptor_positions: List of acceptor splice site positions
        
    Returns:
        Tuple of (resolved_donor_positions, resolved_acceptor_positions)
    """
    # Find overlapping positions
    overlaps = set(donor_positions).intersection(set(acceptor_positions))
    
    if not overlaps:
        return donor_positions, acceptor_positions
    
    # Clone the lists to avoid modifying the originals
    resolved_donors = donor_positions.copy()
    resolved_acceptors = acceptor_positions.copy()
    
    # Apply priority rule (e.g., prioritize donor sites)
    for pos in overlaps:
        # Remove from acceptor list to prioritize donor
        resolved_acceptors.remove(pos)
        
    return resolved_donors, resolved_acceptors
```

## 3. Dataset Creation

### PyTorch Dataset Implementation

```python
class SpliceSiteDataset(Dataset):
    """Dataset for splice site prediction."""
    
    def __init__(self, sequences, labels, transform=None):
        """
        Initialize dataset.
        
        Args:
            sequences: List of one-hot encoded sequences, each shape (seq_length, 4)
            labels: List of one-hot encoded labels, each shape (seq_length, 3)
            transform: Optional transform to apply to samples
        """
        self.sequences = sequences
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.sequences)
        
    def __getitem__(self, idx):
        """Get a sample from the dataset."""
        seq = self.sequences[idx]
        label = self.labels[idx]
        
        # Convert to tensors
        seq_tensor = torch.tensor(seq, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.float32)
        
        # Apply transform if specified
        if self.transform:
            seq_tensor, label_tensor = self.transform(seq_tensor, label_tensor)
            
        return seq_tensor, label_tensor

def create_dataloaders(sequences, labels, batch_size=32, val_split=0.2, test_split=0.1, 
                      random_seed=42):
    """
    Create train, validation, and test dataloaders.
    
    Args:
        sequences: List of one-hot encoded sequences
        labels: List of one-hot encoded labels
        batch_size: Batch size for dataloaders
        val_split: Fraction of data for validation
        test_split: Fraction of data for testing
        random_seed: Random seed for reproducibility
        
    Returns:
        Dictionary with train, validation, and test dataloaders
    """
    # Set random seed for reproducibility
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    
    # Shuffle data
    indices = np.arange(len(sequences))
    np.random.shuffle(indices)
    
    # Split indices for train, validation, and test
    test_size = int(len(indices) * test_split)
    val_size = int(len(indices) * val_split)
    train_size = len(indices) - val_size - test_size
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    # Create datasets
    train_dataset = SpliceSiteDataset(
        [sequences[i] for i in train_indices],
        [labels[i] for i in train_indices]
    )
    
    val_dataset = SpliceSiteDataset(
        [sequences[i] for i in val_indices],
        [labels[i] for i in val_indices]
    )
    
    test_dataset = SpliceSiteDataset(
        [sequences[i] for i in test_indices],
        [labels[i] for i in test_indices]
    )
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }
```

## 4. Data Augmentation

### Sequence Augmentation Techniques

Several data augmentation techniques can be applied to DNA sequences to improve model generalization:

```python
class SequenceAugmentation:
    """Augmentation techniques for DNA sequences."""
    
    @staticmethod
    def reverse_complement(seq_tensor, label_tensor=None):
        """
        Create reverse complement of a sequence.
        
        Args:
            seq_tensor: One-hot encoded sequence tensor (seq_length, 4)
            label_tensor: One-hot encoded label tensor (seq_length, 3)
            
        Returns:
            Augmented sequence and label tensors
        """
        # Reverse the sequence
        seq_reversed = torch.flip(seq_tensor, [0])
        
        # Complement the nucleotides (A<->T, C<->G)
        # In one-hot encoding: [A,C,G,T] -> [T,G,C,A]
        seq_complement = seq_reversed[:, [3, 2, 1, 0]]
        
        # For labels, donor sites become acceptor sites and vice versa in the reverse complement
        if label_tensor is not None:
            # Reverse the labels
            label_reversed = torch.flip(label_tensor, [0])
            
            # Swap donor and acceptor labels [donor,acceptor,neither] -> [acceptor,donor,neither]
            label_reversed_swapped = label_reversed[:, [1, 0, 2]]
            
            return seq_complement, label_reversed_swapped
        
        return seq_complement
    
    @staticmethod
    def add_random_mutations(seq_tensor, label_tensor, mutation_rate=0.01):
        """
        Add random mutations to a sequence.
        
        Args:
            seq_tensor: One-hot encoded sequence tensor (seq_length, 4)
            label_tensor: One-hot encoded label tensor
            mutation_rate: Probability of mutating each nucleotide
            
        Returns:
            Augmented sequence tensor (labels unchanged)
        """
        seq_length = seq_tensor.size(0)
        mutated_seq = seq_tensor.clone()
        
        # Generate random indices to mutate
        num_mutations = int(seq_length * mutation_rate)
        mutation_indices = torch.randperm(seq_length)[:num_mutations]
        
        # For each mutation site, randomly replace with another nucleotide
        for idx in mutation_indices:
            # Generate a random one-hot vector
            new_nucleotide = torch.zeros(4)
            new_idx = torch.randint(0, 4, (1,))
            new_nucleotide[new_idx] = 1.0
            
            # Replace the nucleotide
            mutated_seq[idx] = new_nucleotide
            
        return mutated_seq, label_tensor
```

### Handling Class Imbalance

Splice sites typically constitute a very small fraction of the genome, leading to class imbalance:

```python
def calculate_class_weights(labels):
    """
    Calculate class weights inversely proportional to class frequencies.
    
    Args:
        labels: List of one-hot encoded labels, each shape (seq_length, 3)
        
    Returns:
        Tensor of shape (3,) with class weights
    """
    # Concatenate all labels
    all_labels = np.concatenate([l.reshape(-1, 3) for l in labels], axis=0)
    
    # Count class occurrences
    class_counts = all_labels.sum(axis=0)
    
    # Calculate weights inversely proportional to frequency
    weights = 1.0 / class_counts
    
    # Normalize weights
    weights = weights / weights.sum() * 3
    
    return torch.tensor(weights, dtype=torch.float32)
```

## 5. Sequence Context Handling

### Window-Based Approach

For models that can't process very long sequences, a window-based approach can be used:

```python
def create_context_windows(sequence, labels, window_size=1000, stride=100):
    """
    Create overlapping windows from a long sequence.
    
    Args:
        sequence: One-hot encoded sequence, shape (seq_length, 4)
        labels: One-hot encoded labels, shape (seq_length, 3)
        window_size: Size of each window
        stride: Step size between windows
        
    Returns:
        Lists of windowed sequences and labels
    """
    seq_length = len(sequence)
    windowed_sequences = []
    windowed_labels = []
    
    for start in range(0, seq_length - window_size + 1, stride):
        end = start + window_size
        window_seq = sequence[start:end]
        window_labels = labels[start:end]
        
        windowed_sequences.append(window_seq)
        windowed_labels.append(window_labels)
        
    return windowed_sequences, windowed_labels
```

## Complete Data Processing Pipeline

### Example Pipeline Integration

```python
def process_genome_data(fasta_file, annotation_file, output_dir, window_size=10000, stride=5000):
    """
    Process genome data from FASTA and annotation files.
    
    Args:
        fasta_file: Path to FASTA file with genomic sequences
        annotation_file: Path to annotation file with splice site positions
        output_dir: Directory to save processed data
        window_size: Size of sequence windows
        stride: Stride between windows
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load genome sequences
    genome_sequences = load_fasta(fasta_file)
    
    # Load splice site annotations
    annotations = load_annotations(annotation_file)
    
    processed_sequences = []
    processed_labels = []
    
    # Process each chromosome/sequence
    for chrom, sequence in genome_sequences.items():
        if chrom not in annotations:
            continue
            
        # Get donor and acceptor positions for this chromosome
        donor_positions = annotations[chrom]['donors']
        acceptor_positions = annotations[chrom]['acceptors']
        
        # Resolve any overlapping positions
        donor_positions, acceptor_positions = resolve_overlapping_sites(
            donor_positions, acceptor_positions
        )
        
        # Create one-hot encoded sequence
        one_hot_sequence = one_hot_encode(sequence)
        
        # Create labels
        labels = create_splice_labels(
            len(sequence), donor_positions, acceptor_positions
        )
        
        # Create overlapping windows
        window_sequences, window_labels = create_context_windows(
            one_hot_sequence, labels, window_size=window_size, stride=stride
        )
        
        processed_sequences.extend(window_sequences)
        processed_labels.extend(window_labels)
    
    # Save processed data
    np.save(os.path.join(output_dir, 'sequences.npy'), np.array(processed_sequences))
    np.save(os.path.join(output_dir, 'labels.npy'), np.array(processed_labels))
    
    print(f"Processed {len(processed_sequences)} sequence windows")
    
    # Calculate and save class weights
    class_weights = calculate_class_weights(processed_labels)
    np.save(os.path.join(output_dir, 'class_weights.npy'), class_weights.numpy())
    
    print(f"Class weights: {class_weights}")
```

## Key Considerations

1. **Balanced Sampling**: Ensure training batches contain a balanced mix of positive (splice sites) and negative examples

2. **Sequence Context**: Splice site prediction depends on the surrounding sequence context, so provide sufficient context on both sides of potential splice sites

3. **Genome-Wide Representation**: Include sequences from various chromosomes and genomic regions to ensure model generalization

4. **Species-Specific Patterns**: Different species may have different splice site patterns, so train species-specific models when needed

5. **Data Leakage Prevention**: Ensure that sequences from the same gene or highly similar sequences are not split between train/validation/test sets

6. **Memory Efficiency**: For genome-scale data, use data generators or on-the-fly processing to manage memory usage
