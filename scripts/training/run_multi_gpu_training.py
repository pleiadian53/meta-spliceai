#!/usr/bin/env python
"""
Multi-GPU training script for error sequence models using DeepSpeed.

This script demonstrates how to run distributed training on Tesla M60 GPUs
with memory constraints using DeepSpeed for optimization.

Example usage:
    # Single-process testing (for development):
    python run_multi_gpu_training.py
    
    # Multi-GPU training with torchrun:
    torchrun --nproc_per_node=2 run_multi_gpu_training.py
"""

import os
import sys
import argparse
import torch
import multiprocessing
from pathlib import Path
import torch.distributed as dist

# Force multiprocessing to use 'spawn' method instead of 'fork'
# This prevents Polars deadlocks on Linux
multiprocessing.set_start_method('spawn', force=True)

# Set environment variables to avoid warnings
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['POLARS_ALLOW_FORKING_THREAD'] = '1'

# Import splicesurvey modules
from meta_spliceai.splice_engine.error_sequence_demo import demo_pretrain_finetune_dist
from meta_spliceai.splice_engine.error_sequence_distributed import (
    log_memory_usage, 
    log_gpu_memory_detailed
)

def main():
    parser = argparse.ArgumentParser(description="Run multi-GPU training for error sequence models")
    
    # Dataset and model parameters
    parser.add_argument("--experiment", type=str, default="hard_genes", 
                        help="Experiment name")
    parser.add_argument("--pred_type", type=str, default="FP", 
                        help="Prediction type (FP, FN)")
    parser.add_argument("--correct_label", type=str, default="TP", 
                        help="Correct label for comparison")
    parser.add_argument("--model_name", type=str, default="zhihan1996/DNABERT-2-117M", 
                        help="Pretrained model name")
    
    # Training parameters
    parser.add_argument("--epochs", type=int, default=3, 
                        help="Number of training epochs")
    parser.add_argument("--max_length", type=int, default=768, 
                        help="Maximum sequence length")
    parser.add_argument("--batch_sizes", type=str, default="4,2,1", 
                        help="Comma-separated batch sizes for each bucket")
    parser.add_argument("--bucket_boundaries", type=str, default="256,512,768", 
                        help="Comma-separated sequence length boundaries for buckets")
    
    # DeepSpeed configuration
    parser.add_argument("--deepspeed_config", type=str, 
                        default="deepspeed_config.json", 
                        help="Path to DeepSpeed configuration file")
    
    args = parser.parse_args()
    
    # Process batch sizes and bucket boundaries
    max_batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    bucket_boundaries = [int(x) for x in args.bucket_boundaries.split(",")]
    
    # Ensure DeepSpeed config exists
    if not os.path.exists(args.deepspeed_config):
        print(f"[WARNING] DeepSpeed config not found at {args.deepspeed_config}")
        print("[INFO] Will generate default config during training")
    
    # Log available GPUs
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"[INFO] Found {gpu_count} GPUs available for training")
        for i in range(gpu_count):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("[WARNING] No GPUs available. Training will run on CPU only.")
    
    # Log initial memory usage
    print("[INFO] Initial memory usage:")
    log_memory_usage()
    log_gpu_memory_detailed()
    
    # Run the training
    print(f"[INFO] Starting training with DeepSpeed on experiment: {args.experiment}")
    print(f"[INFO] Prediction type: {args.pred_type} vs {args.correct_label}")
    print(f"[INFO] Using model: {args.model_name}")
    print(f"[INFO] Training for {args.epochs} epochs")
    print(f"[INFO] Max sequence length: {args.max_length}")
    print(f"[INFO] Bucket boundaries: {bucket_boundaries}")
    print(f"[INFO] Max batch sizes: {max_batch_sizes}")
    
    model, trainer, model_path = demo_pretrain_finetune_dist(
        experiment=args.experiment,
        pred_type=args.pred_type,
        correct_label=args.correct_label,
        model_name=args.model_name,
        num_train_epochs=args.epochs,
        max_length=args.max_length,
        bucket_boundaries=bucket_boundaries,
        max_batch_sizes=max_batch_sizes,
    )
    
    print(f"[INFO] Training complete!")
    print(f"[INFO] Model saved to: {model_path}")
    
    # Final memory usage
    print("[INFO] Final memory usage:")
    log_memory_usage()
    log_gpu_memory_detailed()
    
    # Properly clean up distributed resources
    if dist.is_initialized():
        try:
            print("[INFO] Cleaning up distributed process group")
            # Wait for all processes to reach this point
            dist.barrier()
            # Then destroy the process group
            dist.destroy_process_group()
        except Exception as e:
            print(f"[WARNING] Error while cleaning up distributed resources: {e}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
