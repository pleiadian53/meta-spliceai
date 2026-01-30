"""
DeepSpeed optimization patch for error_sequence_model_dist.py.
This file contains the changes needed to make DeepSpeed work with 2x Tesla M60 GPUs.

Instructions:
1. This is a reference implementation - copy the relevant parts to error_sequence_model_dist.py
2. Focus on modifying the training_arguments dictionary in demo_pretrain_finetune()
3. Apply the memory optimizations to train_transformer_model()
"""

# PATCH 1: Modified training arguments dictionary for demo_pretrain_finetune()
# Replace the training_arguments dictionary around line 5934 with this:

training_arguments = {
    "output_dir": output_dir,
    "eval_strategy": 'epoch',  # Evaluation strategy ('steps' or 'epoch')
    "save_strategy": 'epoch',  # Save strategy ('steps' or 'epoch')
    
    "per_device_train_batch_size": 2,  # Small batch size per GPU for Tesla M60 8GB
    "per_device_eval_batch_size": 2,   # Small batch size for evaluation
    
    "num_train_epochs": 1,  # Handle epochs manually
    "weight_decay": 0.01,   # L2 regularization coefficient
    "learning_rate": 2e-5,  # Slightly lower learning rate for stability
    "warmup_steps": 500,    # Warmup steps for learning rate scheduler
    
    "gradient_accumulation_steps": 16,  # Accumulate gradients over more steps (2*16*2 GPUs = effective batch size of 64)
    
    "load_best_model_at_end": True,    # Automatically load the best model at the end
    "metric_for_best_model": "eval_loss",  # Metric for identifying best model
    
    "logging_dir": logging_dir,  # Directory for logging
    "logging_steps": 100,       # More frequent logging for monitoring
    
    "ddp_find_unused_parameters": False,  # Optimize DDP for unused parameters
    "remove_unused_columns": False,      # Keep all columns in dataset
    "dataloader_drop_last": True,        # Drop last incomplete batch
    "local_rank": local_rank,            # From torchrun
    
    "gradient_checkpointing": True,      # Critical for memory savings
    "deepspeed": deepspeed_config,       # Enable DeepSpeed with config file
    "eval_accumulation_steps": 8,        # For memory efficiency during evaluation
    
    # Memory optimizations for outdated hardware
    "dataloader_num_workers": 2,         # Limit workers for older systems
    "max_grad_norm": 1.0,                # Gradient clipping for stability
}

# PATCH 2: Modified sequence bucketing for better memory efficiency
# Add this function to create dynamically sized buckets based on sequence length

def create_dynamic_buckets(df, tokenizer, sequence_column="sequence", 
                          bucket_boundaries=[256, 512, 768, 1024], max_batch_sizes=[8, 4, 2, 1]):
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

# PATCH 3: Monitoring memory usage during training
# Add this to log GPU memory during training

def log_gpu_memory_detailed():
    """
    Detailed GPU memory logging with breakdown of allocations.
    Helps identify memory leaks and bottlenecks.
    """
    import torch
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

# MODIFIED WORKFLOW EXAMPLE:
# Insert this example workflow in demo_pretrain_finetune() 
# to replace the standard training approach with DeepSpeed-optimized workflow

"""
# DeepSpeed-optimized workflow
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs with DeepSpeed")
    
    # 1. Log memory before training
    log_gpu_memory_detailed()
    
    # 2. Set smaller sequence length for memory efficiency
    if max_length > 768:
        print(f"[Warning] Reducing max_length from {max_length} to 768 for memory efficiency")
        max_length = 768
        
    # 3. Create smaller bucketed datasets
    bucket_boundaries = [256, 512, 768]
    max_batch_sizes = [4, 2, 1]  # Smaller batches for longer sequences
    
    train_buckets = create_dynamic_buckets(
        train_df, tokenizer, 
        bucket_boundaries=bucket_boundaries,
        max_batch_sizes=max_batch_sizes
    )
    
    # 4. Use DeepSpeed-enabled trainer
    deepspeed_model = None
    trainer = None
    
    for epoch in range(num_train_epochs):
        print(f"[INFO] Starting training epoch {epoch+1}/{num_train_epochs}")
        
        # Train on each bucket with appropriate batch size
        for i, bucket in enumerate(train_buckets):
            print(f"[INFO] Training on bucket {i+1}/{len(train_buckets)}")
            print(f"  - Sequences: {bucket['count']}")
            print(f"  - Max length: {bucket['max_length']}")
            print(f"  - Batch size: {bucket['batch_size']}")
            
            # Update batch size for this bucket
            training_arguments['per_device_train_batch_size'] = bucket['batch_size']
            
            # Create trainer for this bucket
            bucket_model, bucket_trainer = train_transformer_model(
                train_dataset=bucket['dataset'],
                val_dataset=val_dataset,  # Same validation set for all buckets
                training_arguments=training_arguments,
                model_name=model_name,
                num_labels=2,
                num_train_epochs=1,  # Just one epoch per bucket per global epoch
                device=device,
                return_path_to_model=False,
                train_model=True
            )
            
            # Save the latest model state
            if deepspeed_model is None:
                deepspeed_model = bucket_model
                trainer = bucket_trainer
            else:
                # Copy weights from bucket_model to deepspeed_model
                for target_param, param in zip(deepspeed_model.parameters(), bucket_model.parameters()):
                    target_param.data.copy_(param.data)
                
                # Update trainer
                trainer = bucket_trainer
            
            # Log memory after training on this bucket
            log_gpu_memory_detailed()
            
            # Clear cache to free memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # Final evaluation after all epochs
    model = deepspeed_model
else:
    # Standard single-GPU training
    model, trainer = train_transformer_model(...)
"""
