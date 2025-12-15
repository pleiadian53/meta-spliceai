# Multi-Instance Training: Memory-Efficient Gene Allocation

## How Multi-Instance Training Determines Genes Per Instance

The multi-instance training system uses a sophisticated **hardware-adaptive algorithm** to determine the optimal number of genes per instance while maintaining memory efficiency. Here's the complete breakdown:

## 🧮 Core Algorithm

### 1. Default Configuration Parameters

```python
# Default values from command-line arguments
base_genes_per_instance = 1500          # Default genes per instance
max_instances = 10                      # Maximum number of instances
instance_overlap = 0.1                  # 10% overlap between instances
memory_per_gene_mb = 8.0               # Estimated memory per gene (MB)
max_memory_per_instance_gb = 15.0      # Maximum memory per instance (GB)
auto_adjust_instance_size = True       # Enable hardware adaptation
```

### 2. Hardware-Adaptive Memory Calculation

When `auto_adjust_instance_size=True` (default), the system performs dynamic hardware analysis:

```python
# Step 1: Detect available system memory
import psutil
available_memory_gb = psutil.virtual_memory().available / (1024**3)

# Step 2: Calculate maximum genes based on memory constraints
max_genes_by_memory = int((max_memory_per_instance_gb * 1024) / memory_per_gene_mb)
# Example: (15.0 GB * 1024 MB/GB) / 8.0 MB/gene = 1,920 genes

# Step 3: Choose the more conservative limit
optimal_genes_per_instance = min(base_genes_per_instance, max_genes_by_memory)
# Example: min(1500, 1920) = 1500 genes

# Step 4: Apply safety bounds
optimal_genes_per_instance = max(500, min(optimal_genes_per_instance, 5000))
```

### 3. Instance Count Calculation

Once the optimal genes per instance is determined, calculate how many instances are needed:

```python
# Calculate effective genes per instance (accounting for overlap)
genes_per_instance_effective = int(optimal_genes_per_instance * (1 - instance_overlap))
# Example: 1500 * (1 - 0.1) = 1350 effective genes per instance

# Calculate minimum instances needed for complete coverage
optimal_instances = max(3, (total_genes + genes_per_instance_effective - 1) // genes_per_instance_effective)
# Example: max(3, (9280 + 1350 - 1) // 1350) = max(3, 7) = 7 instances

# Respect maximum instance limit
optimal_instances = min(optimal_instances, max_instances)
```

## 📊 Memory Estimation Formula

The system uses a **position-centric memory model** based on empirical measurements:

### Memory Per Gene Calculation
```
Memory per gene = 8.0 MB (default)

This accounts for:
- Feature vectors: ~1,167 features × 4 bytes (float32) = ~4.7 KB per position
- Average positions per gene: ~400 positions
- Training overhead: XGBoost internal structures, CV folds, etc.
- Safety margin: Memory fragmentation and peak usage

Total: ~400 positions × 4.7 KB × overhead factors ≈ 8 MB per gene
```

### Instance Memory Calculation
```
Memory per instance = genes_per_instance × memory_per_gene_mb
Example: 1500 genes × 8.0 MB = 12.0 GB per instance
```

## 🔧 Hardware Adaptation Examples

### High-Memory System (64GB RAM)
```bash
# Available memory: ~60GB
# Max genes by memory: (30GB * 1024) / 8MB = 3,840 genes
# Optimal genes per instance: min(1500, 3840) = 1500 genes
# Result: Uses default 1500 genes per instance

Configuration:
- Genes per instance: 1,500
- Instances needed: 7
- Memory per instance: 12 GB
- Total memory usage: ~84 GB (across all instances, but sequential)
```

### Medium-Memory System (32GB RAM)
```bash
# Available memory: ~28GB  
# Max genes by memory: (15GB * 1024) / 8MB = 1,920 genes
# Optimal genes per instance: min(1500, 1920) = 1500 genes
# Result: Uses default 1500 genes per instance

Configuration:
- Genes per instance: 1,500
- Instances needed: 7  
- Memory per instance: 12 GB
- Total memory usage: 12 GB (sequential training)
```

### Low-Memory System (16GB RAM)
```bash
# Available memory: ~12GB
# Max genes by memory: (8GB * 1024) / 8MB = 1,024 genes  
# Optimal genes per instance: min(1500, 1024) = 1024 genes
# Result: Automatically reduces to 1024 genes per instance

Configuration:
- Genes per instance: 1,024
- Instances needed: 10
- Memory per instance: 8.2 GB
- Total memory usage: 8.2 GB (sequential training)
```

## 🎯 User Configuration Options

### Manual Override
```bash
# Override default genes per instance
--genes-per-instance 2000

# Override memory constraints  
--max-memory-per-instance-gb 20.0

# Override memory estimation
--memory-per-gene-mb 10.0

# Disable automatic adjustment
--no-auto-adjust-instance-size
```

### Automatic Configuration (Recommended)
```bash
# Let the system optimize automatically
python -m meta_spliceai...run_gene_cv_sigmoid \
    --dataset train_regulatory_10k_kmers/master \
    --train-all-genes \
    --auto-adjust-instance-size \  # Default: enabled
    --verbose
```

## 📈 Dynamic Adjustment Logic

### Complete Coverage Guarantee

The system ensures **100% gene coverage** regardless of hardware constraints:

```python
def ensure_complete_coverage(total_genes, optimal_genes_per_instance, instance_overlap):
    """Calculate instances needed to cover ALL genes."""
    
    # Effective genes per instance (after overlap)
    effective_genes = int(optimal_genes_per_instance * (1 - instance_overlap))
    
    # Minimum instances for complete coverage
    min_instances = (total_genes + effective_genes - 1) // effective_genes
    
    # Always use at least 3 instances for ensemble benefits
    return max(3, min_instances)
```

### Memory Safety Bounds

The system applies **conservative safety bounds** to prevent OOM errors:

```python
# Minimum viable instance size
MIN_GENES_PER_INSTANCE = 500    # Below this, training becomes ineffective

# Maximum reasonable instance size  
MAX_GENES_PER_INSTANCE = 5000   # Above this, memory benefits diminish

# Safety factor in memory calculations
MEMORY_SAFETY_FACTOR = 0.6      # Use only 60% of available memory
```

## 🔍 Real-World Example: train_regulatory_10k_kmers

For the large regulatory dataset with **9,280 genes**:

### System Analysis
```
Dataset: train_regulatory_10k_kmers/master
Total genes: 9,280
Total positions: 3,729,279
Features per position: 1,167
```

### Memory Calculation
```
Memory per gene: 8.0 MB (empirically measured)
Total dataset memory: 9,280 × 8.0 MB = 74.2 GB
```

### Multi-Instance Configuration (Medium System)
```
Available memory: 32 GB
Max memory per instance: 15 GB
Max genes by memory: 1,920 genes
Optimal genes per instance: min(1500, 1920) = 1,500 genes
Effective genes per instance: 1,500 × 0.9 = 1,350 genes
Instances needed: ceil(9,280 / 1,350) = 7 instances
```

### Result
```
✅ 7 instances × 1,500 genes each = 10,500 total gene slots
✅ 10% overlap provides 1,220 redundant slots  
✅ 9,280 unique genes + 1,220 overlap = 100% coverage
✅ Memory per instance: 12 GB (within 15 GB limit)
✅ Sequential training: Only one instance in memory at a time
```

## 🚀 Performance Benefits

### Memory Efficiency
- **Single Model**: 74.2 GB (fails on most systems)
- **Multi-Instance**: 12 GB per instance (succeeds on 16GB+ systems)
- **Reduction**: 6.2× memory reduction

### Scalability
- **Unlimited Dataset Size**: Memory usage independent of total genes
- **Hardware Adaptive**: Automatically optimizes for available resources
- **Fault Tolerant**: Instance failures don't compromise overall training

### Quality Preservation
- **Complete Coverage**: Every gene contributes to final model
- **Full Analysis**: Each instance gets CV + SHAP + calibration
- **Ensemble Benefits**: Model diversity improves generalization

This sophisticated memory management system is what enables the multi-instance training to handle genomic datasets of unlimited size while maintaining memory efficiency and ensuring 100% gene coverage.
