"""
DeepSpeed-optimized distributed training utilities for error sequence models.

This module provides optimized training functions for multi-GPU setups,
specifically designed for limited memory environments like Tesla M60 GPUs.
"""

import os
import torch
import numpy as np
from typing import Dict, List, Union, Tuple, Optional
import warnings
import psutil

def log_memory_usage():
    """
    Log system memory usage (RAM).
    
    Returns detailed information about current memory usage:
    - RSS (Resident Set Size): Memory occupied by the process in RAM
    - VMS (Virtual Memory Size): Total virtual memory used
    """
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    
    # Convert to gigabytes for readability
    gb_factor = 1024 ** 3
    rss_gb = memory_info.rss / gb_factor
    vms_gb = memory_info.vms / gb_factor
    
    print(f"[Memory Usage] RSS: {rss_gb:.2f} GB, VMS: {vms_gb:.2f} GB")
    return memory_info

def log_gpu_memory_detailed():
    """
    Detailed GPU memory logging with breakdown of allocations.
    Helps identify memory leaks and bottlenecks.
    """
    if not torch.cuda.is_available():
        print("CUDA not available for memory logging")
        return
        
    # Basic memory stats
    for i in range(torch.cuda.device_count()):
        memory_allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)
        memory_reserved = torch.cuda.memory_reserved(i) / (1024 ** 3)
        max_memory_allocated = torch.cuda.max_memory_allocated(i) / (1024 ** 3)
        
        print(f"GPU {i}:")
        print(f"  Current allocation: {memory_allocated:.2f} GB")
        print(f"  Reserved memory: {memory_reserved:.2f} GB")
        print(f"  Peak allocation: {max_memory_allocated:.2f} GB")
        
        # Memory by tensor type (optional)
        if hasattr(torch.cuda, 'memory_stats'):
            stats = torch.cuda.memory_stats(i)
            if 'allocated_bytes.all.current' in stats:
                print(f"  Detailed allocation: {stats['allocated_bytes.all.current'] / (1024 ** 3):.2f} GB")

def set_device():
    """
    Determine the device to use (CPU or GPU) for distributed training.
    
    Returns:
        device (torch.device): The device to use for training
        local_rank (int): The local rank of the process
    """
    if torch.cuda.is_available():
        # If launched with torchrun or launch, 'LOCAL_RANK' is set
        local_rank_env = os.environ.get('LOCAL_RANK', None)
        if local_rank_env is not None:
            local_rank = int(local_rank_env)
            device = torch.device(f"cuda:{local_rank}")
            torch.cuda.set_device(local_rank)
            print(f"[INFO] Using CUDA device: {device} (LOCAL_RANK={local_rank})")
        else:
            # Single-GPU or a manual environment without local_rank
            local_rank = 0
            device = torch.device("cuda:0")
            print(f"[INFO] Using CUDA device: {device} (single GPU mode)")
    else:
        # CPU-only case
        device = torch.device("cpu")
        local_rank = -1
        print("[INFO] CUDA not available. Using CPU.")

    return device, local_rank

def is_distributed_training():
    """
    Detect if the current training session is in distributed mode.

    Returns:
        distributed (bool): True if distributed training is enabled and initialized
    """
    return torch.distributed.is_available() and torch.distributed.is_initialized()

def setup_device_and_model(model, distributed=False, verbose=1):
    """
    Set up the device and model for single-GPU, multi-GPU, or CPU training.

    Parameters:
        model: The model to set up
        distributed (bool): Whether to use DistributedDataParallel
        verbose (int): Verbosity level

    Returns:
        model: The model, moved to the appropriate device and wrapped if necessary
        device: The torch.device used for training
    """
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # Detect device
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}" if distributed else "cuda")
        torch.cuda.set_device(local_rank)  # Ensure correct GPU is set for this process
    else:
        device = torch.device("cpu")

    # Move model to the detected device if not already there
    current_device = next(model.parameters()).device
    if current_device != device:
        model = model.to(device)

    # Wrap model in DDP if distributed training is enabled
    if distributed:
        if verbose:
            print(f"[INFO] Distributed training enabled. Using device: {device}")
        if not isinstance(model, torch.nn.parallel.DistributedDataParallel):
            # Only set `device_ids` for CUDA devices
            ddp_args = {"device_ids": [local_rank]} if device.type == "cuda" else {}
            model = torch.nn.parallel.DistributedDataParallel(model, **ddp_args)
    else:
        if verbose:
            print(f"[INFO] Using device: {device}")

    return model, device

def create_dynamic_buckets(df, tokenizer, sequence_column="sequence", 
                          bucket_boundaries=[256, 512, 768, 1024], 
                          max_batch_sizes=[8, 4, 2, 1]):
    """
    Create dynamically sized buckets based on sequence length 
    with different batch sizes for memory efficiency.
    
    Args:
        df: DataFrame containing sequences
        tokenizer: Tokenizer for encoding
        sequence_column: Column containing sequence data
        bucket_boundaries: Boundaries for sequence length buckets
        max_batch_sizes: Maximum batch size for each bucket (lower for longer sequences)
        
    Returns:
        List of datasets with batch size information
    """
    from .error_sequence_model_dist import prepare_dataset_for_huggingface
    
    sequence_lengths = [len(tokenizer.encode(seq)) for seq in df[sequence_column]]
    df['seq_len'] = sequence_lengths
    
    buckets = []
    prev_boundary = 0
    
    for i, boundary in enumerate(bucket_boundaries):
        # Create bucket for sequences between prev_boundary and boundary
        bucket_df = df[(df['seq_len'] > prev_boundary) & (df['seq_len'] <= boundary)]
        if len(bucket_df) > 0:
            # Create dataset for this bucket
            bucket_dataset = prepare_dataset_for_huggingface(
                bucket_df, tokenizer, max_length=boundary
            )
            batch_size = max_batch_sizes[i] if i < len(max_batch_sizes) else 1
            buckets.append({
                'dataset': bucket_dataset,
                'max_length': boundary,
                'batch_size': batch_size,
                'count': len(bucket_df)
            })
        prev_boundary = boundary
    
    # Handle sequences longer than the last boundary
    long_seq_df = df[df['seq_len'] > bucket_boundaries[-1]]
    if len(long_seq_df) > 0:
        long_seq_dataset = prepare_dataset_for_huggingface(
            long_seq_df, tokenizer, max_length=max(long_seq_df['seq_len'])
        )
        buckets.append({
            'dataset': long_seq_dataset,
            'max_length': max(long_seq_df['seq_len']),
            'batch_size': 1,  # Use smallest batch size for longest sequences
            'count': len(long_seq_df)
        })
    
    return buckets

def get_optimal_deepspeed_config(tesla_m60=True, train_batch_size=2, grad_acc_steps=16):
    """
    Generate optimal DeepSpeed configuration for older GPU hardware like Tesla M60.
    
    Args:
        tesla_m60: If True, optimize for Tesla M60 GPUs (8GB memory)
        train_batch_size: Per-device train batch size
        grad_acc_steps: Gradient accumulation steps
        
    Returns:
        Dictionary with DeepSpeed configuration
    """
    config = {
        "zero_optimization": {
            "stage": 3,
            "offload_optimizer": {
                "device": "cpu",
                "pin_memory": True
            },
            "offload_param": {
                "device": "cpu",
                "pin_memory": True
            },
            "overlap_comm": True,
            "contiguous_gradients": True,
            "reduce_bucket_size": tesla_m60 and 5e7 or 5e8,
            "stage3_prefetch_bucket_size": tesla_m60 and 5e7 or 5e8,
            "stage3_param_persistence_threshold": 1e4
        },
        "train_micro_batch_size_per_gpu": train_batch_size,
        "gradient_accumulation_steps": grad_acc_steps,
        "gradient_clipping": 1.0,
        "fp16": {
            "enabled": False  # Tesla M60 doesn't support FP16 well
        },
        "bf16": {
            "enabled": False
        },
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": 2e-5,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": 0.01
            }
        },
        "scheduler": {
            "type": "WarmupLR",
            "params": {
                "warmup_min_lr": 0,
                "warmup_max_lr": 2e-5,
                "warmup_num_steps": 500
            }
        },
        "steps_per_print": 50,
        "wall_clock_breakdown": False
    }
    
    return config

def train_with_deepspeed(
    train_df,
    val_dataset,
    tokenizer,
    model_name,
    output_dir,
    num_train_epochs=3,
    max_length=512,
    local_rank=-1,
    num_labels=2,
    bucket_boundaries=[256, 512, 768],
    max_batch_sizes=[4, 2, 1],
    deepspeed_config_path=None
):
    """
    Train model with DeepSpeed optimization, specifically for limited GPU memory.
    
    Args:
        train_df: Training dataframe
        val_dataset: Validation dataset (already tokenized)
        tokenizer: Tokenizer instance
        model_name: Pretrained model name
        output_dir: Directory to save outputs
        num_train_epochs: Number of training epochs
        max_length: Maximum sequence length
        local_rank: Local rank for distributed training
        num_labels: Number of classification labels
        bucket_boundaries: Sequence length thresholds for buckets
        max_batch_sizes: Batch sizes for each bucket
        deepspeed_config_path: Path to DeepSpeed config file
        
    Returns:
        Trained model and trainer
    """
    import os
    import torch
    import numpy as np
    from datasets import Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        TrainingArguments,
        Trainer,
        DataCollatorWithPadding
    )
    import torch.distributed as dist
    
    # Setup device/distributed environment
    device, local_rank = set_device()
    
    # Log memory usage before initializing model
    log_memory_usage()
    log_gpu_memory_detailed()
    
    # Create dynamic buckets for sequences of different lengths
    train_buckets = create_dynamic_buckets(
        train_df,
        tokenizer,
        sequence_column="sequence",
        bucket_boundaries=bucket_boundaries, 
        max_batch_sizes=max_batch_sizes
    )
    
    # Log bucket info
    for i, bucket in enumerate(train_buckets):
        print(f"  Bucket {i+1}: {bucket['count']} sequences, max_length={bucket['max_length']}, batch_size={bucket['batch_size']}")
    
    # Base training arguments with conservatively optimized parameters for old GPUs
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=50,
        
        # These will be overridden per bucket
        per_device_train_batch_size=2,  # Default small batch size
        per_device_eval_batch_size=2,   # Default small batch size
        
        # Conservative learning parameters for stability
        learning_rate=1e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        
        # For memory efficiency
        gradient_checkpointing=True,
        gradient_accumulation_steps=8,
        fp16=True,  # Enable mixed precision
        
        # DeepSpeed integration
        deepspeed=deepspeed_config_path,
        
        # Other optimization settings
        ddp_find_unused_parameters=False,
        dataloader_num_workers=1,  # Reduce for stability
        
        # Load best model at the end
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        
        # For multi-GPU training
        local_rank=local_rank
    )
    
    # Log effective batch size
    effective_batch_size = training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps
    if torch.cuda.device_count() > 1:
        effective_batch_size *= torch.cuda.device_count()
    print(f"[INFO] Effective batch size: {effective_batch_size}")
    
    # Train in a more stable way - one model for all buckets with data concatenation
    # avoiding re-initialization which can cause gradient issues
    final_model = None
    final_trainer = None
    
    # Load the base model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        trust_remote_code=False
    )
    
    # Reset the checkpoints to save space and avoid duplication
    os.makedirs(output_dir, exist_ok=True)
    checkpoints = [output_dir]
    
    # Combine all buckets into one dataset to avoid re-initializing models
    combined_dataset = None
    combined_batch_size = 2  # Small default batch size
    
    print(f"[INFO] Creating combined dataset from all buckets")
    for bucket in train_buckets:
        if combined_dataset is None:
            combined_dataset = bucket['dataset']
            # Use the smallest bucket batch size for safety
            combined_batch_size = min(combined_batch_size, bucket['batch_size'])
        else:
            combined_dataset = combined_dataset.concatenate(bucket['dataset'])
    
    print(f"[INFO] Combined dataset has {len(combined_dataset)} examples")
    print(f"[INFO] Using batch size: {combined_batch_size}")
    
    # Update training arguments with combined batch size
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=50,
        
        # Set to conservative batch size
        per_device_train_batch_size=combined_batch_size,
        per_device_eval_batch_size=combined_batch_size,
        
        # Conservative learning parameters
        learning_rate=1e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        
        # For memory efficiency
        gradient_checkpointing=True,
        gradient_accumulation_steps=16,  # Increase to compensate for smaller batch
        fp16=True,  # Enable mixed precision
        
        # DeepSpeed integration
        deepspeed=deepspeed_config_path,
        
        # Other optimization settings
        ddp_find_unused_parameters=False,
        dataloader_num_workers=1,
        
        # Load best model at the end
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        
        # For multi-GPU training
        local_rank=local_rank,
        
        # Number of epochs
        num_train_epochs=num_train_epochs
    )
    
    # Create a data collator that dynamically pads the batch
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=combined_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator
    )
    
    print(f"[INFO] Starting training for {num_train_epochs} epochs")
    trainer.train()
    
    # After training, save model
    final_model_path = os.path.join(output_dir, "final_model")
    os.makedirs(final_model_path, exist_ok=True)
    trainer.save_model(final_model_path)
    print(f"[INFO] Model saved to {final_model_path}")
    
    # Final memory check
    log_memory_usage()
    log_gpu_memory_detailed()
    
    # NOTE: We're NOT destroying the process group here anymore
    # We'll let the caller handle that after evaluation is complete
    
    return model, trainer, final_model_path
