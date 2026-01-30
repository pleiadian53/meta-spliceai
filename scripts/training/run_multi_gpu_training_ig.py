#!/usr/bin/env python
"""
Multi-GPU training script for Integrated Gradients analysis using DeepSpeed.

This script runs Integrated Gradients analysis on trained models to identify
important sequence patterns in DNA that influence error predictions.

Usage:
    torchrun --nproc_per_node=<num_gpus> run_multi_gpu_training_ig.py [options]

Options:
    --experiment EXPERIMENT       Experiment name (default: 'hard_genes')
    --pred_type {FP,FN}           Prediction type to analyze (default: 'FP')
    --splice_type {any,donor,acceptor}  Splice site type (default: 'any')
    --correct_label {TP,TN}       Correct label to compare against (default: 'TP')
    --n_steps N_STEPS             Number of IG steps (default: 50)
    --batch_size BATCH_SIZE       Batch size for analysis (default: 2)
    --max_samples MAX_SAMPLES     Maximum number of samples to analyze (default: 50)
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

# Import from meta_spliceai
sys.path.insert(0, str(Path(__file__).resolve().parent))
from meta_spliceai.splice_engine.error_sequence_demo_ig import demo_pretrain_finetune_ig_dist
from meta_spliceai.splice_engine.error_sequence_distributed import log_memory_usage, log_gpu_memory_detailed

def main():
    """
    Main entry point for multi-GPU Integrated Gradients analysis.
    
    Parses command-line arguments and runs the demo_pretrain_finetune_ig_dist
    function with appropriate parameters.
    """
    parser = argparse.ArgumentParser(description='Run multi-GPU IG analysis on error sequence models.')
    
    # Add arguments
    parser.add_argument('--experiment', type=str, default='hard_genes',
                        help='Experiment name (default: hard_genes)')
    parser.add_argument('--pred_type', type=str, default='FP', choices=['FP', 'FN'],
                        help='Prediction type to analyze (default: FP)')
    parser.add_argument('--splice_type', type=str, default='any', choices=['any', 'donor', 'acceptor'],
                        help='Splice site type (default: any)')
    parser.add_argument('--correct_label', type=str, default='TP', choices=['TP', 'TN'],
                        help='Correct label to compare against (default: TP)')
    parser.add_argument('--n_steps', type=int, default=50,
                        help='Number of IG steps (default: 50)')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='Batch size for analysis (default: 2)')
    parser.add_argument('--max_samples', type=int, default=50,
                        help='Maximum number of samples to analyze (default: 50)')

    args = parser.parse_args()
    
    # Set up the local rank
    local_rank = int(os.environ.get('LOCAL_RANK', '-1'))
    
    # Determine if this is the main process
    is_main_process = local_rank in [0, -1]
    
    if is_main_process:
        print(f"[INFO] Starting multi-GPU IG analysis for prediction type: {args.pred_type}")
        print(f"[INFO] Configuration:")
        print(f"  - Experiment: {args.experiment}")
        print(f"  - Error label: {args.pred_type}")
        print(f"  - Correct label: {args.correct_label}")
        print(f"  - Splice type: {args.splice_type}")
        print(f"  - IG steps: {args.n_steps}")
        print(f"  - Batch size: {args.batch_size}")
        print(f"  - Max samples: {args.max_samples}")
        
        # Initial memory usage
        print("[INFO] Initial memory usage:")
        log_memory_usage()
        log_gpu_memory_detailed()
    
    # Run the IG analysis
    results = demo_pretrain_finetune_ig_dist(
        experiment=args.experiment,
        pred_type=args.pred_type,
        error_label=args.pred_type,
        correct_label=args.correct_label,
        splice_type=args.splice_type,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        max_samples=args.max_samples
    )
    
    # Final memory usage
    if is_main_process:
        print("[INFO] Final memory usage:")
        log_memory_usage()
        log_gpu_memory_detailed()
    
    # Properly clean up distributed resources
    if dist.is_initialized():
        try:
            if is_main_process:
                print("[INFO] Cleaning up distributed process group")
            # Wait for all processes to reach this point
            dist.barrier()
            # Then destroy the process group
            dist.destroy_process_group()
        except Exception as e:
            if is_main_process:
                print(f"[WARNING] Error while cleaning up distributed resources: {e}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
