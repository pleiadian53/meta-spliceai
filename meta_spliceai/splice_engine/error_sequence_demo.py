"""
Demonstration functions for error sequence model training,
including DeepSpeed-enabled multi-GPU training capabilities.
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
import polars as pl
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm.auto import tqdm

# Import from other modules
from .error_sequence_distributed import (
    train_with_deepspeed,
    log_memory_usage,
    log_gpu_memory_detailed,
    set_device
)

def demo_pretrain_finetune_dist(**kargs): 
    """
    DeepSpeed-optimized version of pretrain_finetune for multi-GPU training.
    
    Specifically designed for limited memory environments like Tesla M60 GPUs.
    Uses memory optimization techniques including:
    - ZeRO Stage 3 optimization
    - Parameter and optimizer state offloading
    - Dynamic sequence bucketing
    - Gradient checkpointing
    - Adjusted batch sizes based on sequence length
    
    Args:
        **kargs: Keyword arguments including:
            tokenizer_name: Name of the tokenizer to use
            model_name: Pretrained model name
            model_type: Type of model (e.g., 'DNABERT')
            facet: Dataset facet to use
            pred_type: Prediction type to analyze
            experiment: Experiment name
            batch_size: Base batch size (will be adjusted based on sequence length)
            num_train_epochs: Number of training epochs
    """
    from meta_spliceai.splice_engine.analysis_utils import (
        abridge_sequence, 
        analyze_data_labels,
        label_analysis_dataset, 
        subsample_genes, 
        verify_label_classes, 
        show_predicted_splice_sites, 
        find_transcripts_with_both_labels
    )
    from meta_spliceai.splice_engine.seq_model_utils import (
        trim_contextual_sequences
    )
    from meta_spliceai.splice_engine.splice_error_analyzer import ErrorAnalyzer
    from meta_spliceai.splice_engine.utils_df import (
        estimate_dataframe_size, 
        estimate_memory_usage
    )
    
    from .error_sequence_model_dist import (
        evaluate_and_plot_metrics,
        prepare_dataset_for_huggingface,
        prepare_transformer_datasets
    )
    
    # Fixed import for ModelEvaluationFileHandler
    from meta_spliceai.splice_engine.model_evaluator import ModelEvaluationFileHandler
    
    # ----- Configuration -----
    do_fine_tuning = True
    do_manual_evaluation = False
    device, local_rank = set_device()

    # Get model configuration from arguments
    tokenizer_name = kargs.get('tokenizer_name', "zhihan1996/DNABERT-2-117M")
    model_name = kargs.get('model_name', "zhihan1996/DNABERT-2-117M")
    model_type = kargs.get('model_type', 'DNABERT')

    # Dataset configuration
    facet = kargs.get("facet", "simple")
    subject = f"analysis_sequences_{facet}"
    
    pred_type = kargs.get('pred_type', 'FP')
    error_label = kargs.get('error_label', pred_type)
    correct_label = kargs.get('correct_label', "TP")
    splice_type = kargs.get('splice_type', "any")

    pred_type_to_label = {error_label: 1, correct_label: 0}
    gene_level_split = False
    retrieve_motifs = False
    
    # Training parameters
    max_length = kargs.get('max_length', 768)  # Reduced for memory efficiency
    bucket_boundaries = kargs.get('bucket_boundaries', [256, 512, 768])
    max_batch_sizes = kargs.get('max_batch_sizes', [4, 2, 1])  # Different batch sizes by sequence length
    num_train_epochs = kargs.get('num_train_epochs', 3)
    experiment_name = kargs.get('experiment', 'hard_genes')
    
    col_tid = 'transcript_id'
    col_label = 'label'

    # ----- Setup output directories -----
    analyzer = ErrorAnalyzer(experiment=experiment_name, model_type=model_type.lower())
    output_dir = analyzer.set_analysis_output_dir(
        error_label=error_label, 
        correct_label=correct_label, 
        splice_type=splice_type
    )
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"[INFO] Output directory set to: {output_dir}")

    logging_dir = os.path.join(output_dir, "logs")
    os.makedirs(logging_dir, exist_ok=True)
    print(f"[INFO] Logging directory set to: {logging_dir}")

    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"[INFO] Checkpoint directory set to: {checkpoint_dir}")

    # ----- Load and preprocess data -----
    mefd = ModelEvaluationFileHandler(ErrorAnalyzer.eval_dir, separator='\t')
    
    print(f"[INFO] Loading the analysis sequence dataset from {facet}...")
    
    # Load the analysis sequence dataset
    analysis_sequence_df = mefd.load_analysis_sequences(
        aggregated=True, 
        subject=subject, 
        error_label=error_label, 
        correct_label=correct_label, 
        splice_type=splice_type
    )
    
    # Label the dataset
    analysis_sequence_df = label_analysis_dataset(analysis_sequence_df, positive_class=error_label)
    print(f"[INFO] Dataset shape: {analysis_sequence_df.shape}")

    # Verify labels and sequences
    verify_label_classes(analysis_sequence_df, label_col=col_label)
    analyze_data_labels(analysis_sequence_df, label_col=col_label, verbose=2, handle_missing=None)
    
    # Check and adjust sequence lengths
    max_seq_length = verify_sequence_lengths(analysis_sequence_df, sequence_col='sequence')
    if max_seq_length < max_length:
        print(f"[INFO] Maximum sequence length: {max_seq_length} < requested {max_length}")
        max_length = max_seq_length
    
    # ----- Prepare for training -----
    print(f"[INFO] Setting up tokenizer with max_length={max_length}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, model_max_length=max_length)
    
    # Prepare datasets
    print(f"[INFO] Preparing training and validation datasets...")
    train_dataset, val_dataset, train_df, val_df = prepare_transformer_datasets(
        training_df=analysis_sequence_df,
        tokenizer=tokenizer,
        test_size=0.2, 
        return_dataframes=True,
        dataset_format="huggingface",
        gene_level_split=gene_level_split
    )
    
    # ----- DeepSpeed configuration -----
    deepspeed_config = os.path.join(ErrorAnalyzer.analysis_dir, "deepspeed_config.json")
    if not os.path.exists(deepspeed_config):
        print(f"[WARNING] DeepSpeed config not found at {deepspeed_config}")
        deepspeed_config = None
    else:
        print(f"[INFO] Using DeepSpeed config: {deepspeed_config}")
    
    # ----- Multi-GPU training -----
    print(f"[INFO] Starting DeepSpeed-optimized training...")
    model_id = f"{experiment_name.lower()}_{model_type.lower()}_{pred_type}"
    
    # Log initial memory usage
    log_memory_usage()
    log_gpu_memory_detailed()
    
    # Train with DeepSpeed
    if torch.cuda.device_count() > 1:
        print(f"[INFO] Multi-GPU training with {torch.cuda.device_count()} GPUs")
        model, trainer, model_path = train_with_deepspeed(
            train_df=train_df,
            val_dataset=val_dataset,
            tokenizer=tokenizer,
            model_name=model_name,
            output_dir=output_dir,
            num_train_epochs=num_train_epochs,
            max_length=max_length,
            local_rank=local_rank,
            num_labels=2,
            bucket_boundaries=bucket_boundaries,
            max_batch_sizes=max_batch_sizes,
            deepspeed_config_path=deepspeed_config
        )
    else:
        print(f"[INFO] Single-GPU training")
        from .error_sequence_model_dist import train_transformer_model
        
        # Use regular training with memory optimizations
        training_arguments = {
            "output_dir": output_dir,
            "eval_strategy": 'epoch',
            "save_strategy": 'epoch',
            "per_device_train_batch_size": max_batch_sizes[0],
            "per_device_eval_batch_size": max_batch_sizes[0],
            "num_train_epochs": 1,  # Handle epochs manually
            "weight_decay": 0.01,
            "learning_rate": 2e-5,
            "warmup_steps": 500,
            "gradient_accumulation_steps": 8,
            "load_best_model_at_end": True,
            "metric_for_best_model": "eval_loss",
            "logging_dir": logging_dir,
            "logging_steps": 100,
            "ddp_find_unused_parameters": False,
            "gradient_checkpointing": True,
            "local_rank": local_rank,
        }
        
        model, trainer, model_path = train_transformer_model(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            training_arguments=training_arguments,
            model_name=model_name,
            num_labels=2,
            num_train_epochs=num_train_epochs,
            device=device,
            model_id=model_id,
            return_path_to_model=True,
            train_model=do_fine_tuning,
        )
    
    # ----- Evaluation -----
    print(f"[INFO] Final model saved to: {model_path}")
    print(f"[INFO] Model configuration:")
    print(model.config)
    
    # Final evaluation with plotting
    print(f"[INFO] Evaluating model on validation set...")
    validation_report = evaluate_and_plot_metrics(
        trainer, val_dataset, output_dir, 
        label_names=[correct_label, error_label]
    )
    print(f"[INFO] Validation report: {validation_report}")
    
    # Return the trained model and trainer
    return model, trainer, model_path

def verify_sequence_lengths(df, sequence_col='sequence'):
    """
    Verify the lengths of sequences in the dataframe and return the maximum length.
    """
    if isinstance(df, pl.DataFrame):
        # For Polars DataFrame
        seq_lengths = df[sequence_col].map_elements(len).to_numpy()
    else:
        # For Pandas DataFrame
        seq_lengths = df[sequence_col].str.len().values
        
    min_len = np.min(seq_lengths)
    max_len = np.max(seq_lengths)
    avg_len = np.mean(seq_lengths)
    
    print(f"[INFO] Sequence lengths - Min: {min_len}, Max: {max_len}, Avg: {avg_len:.1f}")
    
    return max_len

def display_dataframe_in_chunks(df, num_rows=10, num_cols=None):
    """Display a dataframe in chunks for better readability."""
    if num_cols is None:
        num_cols = df.shape[1]
        
    num_rows_to_display = min(num_rows, df.shape[0])
    df_display = df.iloc[:num_rows_to_display, :num_cols]
    
    print(f"Displaying {num_rows_to_display} rows x {num_cols} columns")
    print(df_display)
    print(f"Total shape: {df.shape}")
    
def print_emphasized(text):
    """Print text with emphasis."""
    print("\n" + "="*80)
    print(text)
    print("="*80)
    
def print_with_indent(text, indent_level=0):
    """Print text with indentation."""
    indent = "  " * indent_level
    print(f"{indent}{text}")
