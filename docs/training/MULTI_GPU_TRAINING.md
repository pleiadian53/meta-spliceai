# Multi-GPU Training for Error Sequence Models

This document describes the implementation of distributed training capabilities for error sequence models in the MetaSpliceAI toolkit using DeepSpeed and PyTorch Distributed.

## Overview

We refactored the original `error_sequence_model_dist.py` module into smaller, more focused components to enable efficient multi-GPU training, particularly optimized for hardware with memory constraints like Tesla M60 GPUs (8GB VRAM).

## Module Structure

The distributed training functionality is organized across several modules:

### 1. `error_sequence_distributed.py`

This is the core module that implements distributed training capabilities with DeepSpeed integration.

**Key Components:**

- **Memory Management Functions**:
  - `log_memory_usage()`: Tracks RAM usage
  - `log_gpu_memory_detailed()`: Detailed GPU memory monitoring
  - `set_device()`: Configures the correct device based on local rank

- **Sequence Bucketing**:
  - `create_dynamic_buckets()`: Groups sequences of similar lengths to optimize memory usage and batch sizes

- **Training Functions**:
  - `train_with_deepspeed()`: Main training function that handles distributed training with DeepSpeed
  - `get_optimal_deepspeed_config()`: Generates an optimized DeepSpeed configuration

### 2. `error_sequence_demo.py`

This module provides demonstration workflows for different training scenarios.

**Key Functions:**

- `demo_pretrain_finetune_dist()`: Demonstrates the entire pipeline for distributed training including:
  - Data loading and preprocessing
  - Model initialization
  - Training with DeepSpeed
  - Evaluation and visualization

### 3. `run_multi_gpu_training.py`

This is a CLI wrapper script that makes it easy to launch distributed training from the command line.

**Features:**

- Command-line argument parsing for training parameters
- Integration with torchrun for distributed process management
- Environment variable configuration for process safety
- Memory usage reporting

### 4. Configuration Files

- `deepspeed_config.json`: Defines DeepSpeed optimization settings including:
  - ZeRO optimization stage
  - Gradient accumulation
  - Mixed precision settings
  - Memory offloading configurations

## How the Modules Interact

1. The user executes `run_multi_gpu_training.py` through `torchrun`
2. The script calls `demo_pretrain_finetune_dist()` from `error_sequence_demo.py`
3. The demo function handles data preparation and calls `train_with_deepspeed()` from `error_sequence_distributed.py`
4. The training function manages:
   - Sequence bucketing for memory efficiency
   - Model initialization and configuration
   - Integration with DeepSpeed optimizer
   - Training loop execution
   - Model saving and evaluation

## Hardware Considerations

The implementation is specifically optimized for Tesla M60 GPUs with 8GB VRAM through:

1. **Dynamic Sequence Bucketing**: Handles sequences of different lengths with appropriate batch sizes
2. **Memory-Efficient Settings**:
   - ZeRO Stage 2 optimization
   - Gradient checkpointing
   - CPU offloading for optimizer states
   - Mixed precision training (FP16)
3. **Conservative Defaults**:
   - Small batch sizes (2, 1, 1) for different sequence length buckets
   - Increased gradient accumulation steps (16) to compensate
   - Reduced learning rate (1e-5) for better stability

## Usage Instructions

### Basic Usage

```bash
# Set environment variables for stability
export TOKENIZERS_PARALLELISM=false
export POLARS_ALLOW_FORKING_THREAD=1

# Launch training on 2 GPUs
torchrun --nproc_per_node=2 run_multi_gpu_training.py \
  --experiment hard_genes \
  --pred_type FP \
  --epochs 3
```

### Advanced Configuration

```bash
torchrun --nproc_per_node=2 run_multi_gpu_training.py \
  --experiment hard_genes \
  --pred_type FP \
  --correct_label TP \
  --model_name zhihan1996/DNABERT-2-117M \
  --epochs 3 \
  --max_length 768 \
  --batch_sizes 2,1,1 \
  --bucket_boundaries 256,512,768 \
  --deepspeed_config deepspeed_config.json
```

### Available Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--experiment` | Experiment name | "hard_genes" |
| `--pred_type` | Type of prediction to analyze (FP, FN) | "FP" |
| `--correct_label` | Ground truth label for comparison | "TP" |
| `--model_name` | Pretrained model name | "zhihan1996/DNABERT-2-117M" |
| `--epochs` | Number of training epochs | 3 |
| `--max_length` | Maximum sequence length | 768 |
| `--batch_sizes` | Comma-separated batch sizes for each bucket | "4,2,1" |
| `--bucket_boundaries` | Sequence length thresholds for buckets | "256,512,768" |
| `--deepspeed_config` | Path to DeepSpeed configuration file | "deepspeed_config.json" |

## Common Issues and Solutions

1. **Out of Memory Errors**:
   - Reduce batch sizes
   - Increase gradient accumulation steps
   - Reduce maximum sequence length

2. **Process Deadlocks**:
   - Ensure `multiprocessing.set_start_method('spawn')` is used instead of 'fork'
   - Set `TOKENIZERS_PARALLELISM=false` and `POLARS_ALLOW_FORKING_THREAD=1`

3. **NCCL Warnings**:
   - These are resolved by properly cleaning up the distributed process group using `dist.destroy_process_group()`

## Benchmarks

On Tesla M60 GPUs (8GB VRAM each), the implementation achieves:

- Stable training with sequence lengths up to 768 tokens
- Peak memory usage ~2GB per GPU
- Effective batch size of 32-64 depending on configuration
- Training accuracy improvements over single-GPU baseline

## Future Improvements

1. Dynamic gradient accumulation based on sequence length
2. Support for even larger sequence lengths through further optimizations
3. Additional DeepSpeed ZeRO stages with fine-tuned configurations
4. More sophisticated dynamic bucketing strategies

## References

1. [DeepSpeed Documentation](https://www.deepspeed.ai/)
2. [PyTorch Distributed Training Guide](https://pytorch.org/tutorials/beginner/dist_overview.html)
3. [NVIDIA NCCL Documentation](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/index.html)
