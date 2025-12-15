from transformers import (
    AutoTokenizer,  # pip install transformers
    AutoModelForSequenceClassification,  # pip install einops
    Trainer, 
    TrainingArguments
)

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

from meta_spliceai.splice_engine.utils_doc import (
    print_emphasized, 
    print_with_indent, 
    print_section_separator, 
    display, 
    display_dataframe_in_chunks
)

from typing import Union, List, Set
from tabulate import tabulate
from tqdm import tqdm

import os
import sys
import numpy as np
import random
import psutil  # pip install psutil

import pandas as pd
import polars as pl


import torch
print("[info] PyTorch version:", torch.__version__)

import transformers
print("[info] Transformers version:", transformers.__version__)

# Update torch and transformers to the latest versions as needed
# pip install --upgrade torch transformers


def verify_tokenizer_max_length(tokenizer_name, model_max_length=512):
    """
    Utility function to verify if the tokenizer and model support the specified max_length.

    Parameters:
    - tokenizer_name (str): Name or path of the tokenizer to verify.
    - model_max_length (int): Maximum sequence length to verify (default: 512).

    Returns:
    - bool: True if the tokenizer supports the specified max_length, False otherwise.
    """
    # Load the tokenizer with specified max_length
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, model_max_length=model_max_length)

    # Print tokenizer configuration details
    print(f"[info] Tokenizer loaded: {tokenizer_name}")
    print(f"[info] Tokenizer max length: {tokenizer.model_max_length}")

    # Verify max_length compatibility
    if model_max_length > tokenizer.model_max_length:
        print(f"[warn] Specified max_length ({model_max_length}) exceeds tokenizer's maximum ({tokenizer.model_max_length}).")
        return False

    print(f"[info] Specified max_length ({model_max_length}) is compatible with the tokenizer.")
    return True


def trim_contextual_sequences_v0(df, target_length=200):
    """
    Trim contextual sequences in the DataFrame to a specified target length.

    Parameters:
    - df (pd.DataFrame): DataFrame containing the columns 'position', 'window_start',
                         'window_end', and 'sequence'.
    - target_length (int): Desired length of the trimmed sequence (default: 200).

    Returns:
    - pd.DataFrame: Updated DataFrame with trimmed sequences and adjusted start/end windows.
    """
    is_polars = isinstance(df, pl.DataFrame)
    if is_polars:
        df = df.to_pandas()

    half_length = target_length // 2  # Half of the target length for symmetric trimming

    # Verify necessary columns exist
    required_columns = ["position", "window_start", "window_end", "sequence"]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' is missing in the input DataFrame.")

    def trim_row(row):
        position = row["position"]
        window_start = row["window_start"]
        window_end = row["window_end"]
        sequence = row["sequence"]

        # Current flanking lengths
        left_flank = position - window_start
        right_flank = window_end - position

        # Validate initial alignment
        if not (left_flank + right_flank + 1 == len(sequence)):
            raise ValueError(
                f"[error] Misalignment detected. Sequence length: {len(sequence)}, "
                f"Left flank: {left_flank}, Right flank: {right_flank}, "
                f"Position: {position}, Window Start: {window_start}, Window End: {window_end}"
            )

        # Determine trimming boundaries
        new_start = max(position - half_length, window_start)  # Avoid going beyond transcript start
        new_end = min(position + half_length, window_end)      # Avoid going beyond transcript end

        # Trim the sequence
        trim_start_idx = new_start - window_start
        trim_end_idx = trim_start_idx + (new_end - new_start)
        trimmed_sequence = sequence[trim_start_idx:trim_end_idx + 1]

        # Adjust the parameters to match the trimmed sequence
        trimmed_left_flank = position - new_start
        trimmed_right_flank = new_end - position

        # Sanity check
        if trimmed_left_flank + trimmed_right_flank + 1 != len(trimmed_sequence):
            raise ValueError(
                f"[error] Post-trim misalignment. Trimmed sequence length: {len(trimmed_sequence)}, "
                f"Left flank: {trimmed_left_flank}, Right flank: {trimmed_right_flank}, "
                f"Position: {position}, New Start: {new_start}, New End: {new_end}"
            )

        # Update the row with new trimmed values
        row["window_start"] = new_start
        row["window_end"] = new_end
        row["sequence"] = trimmed_sequence
        return row

    # Apply trimming row by row
    trimmed_df = df.apply(trim_row, axis=1)

    # Convert back to Polars if the input was Polars
    if is_polars:
        trimmed_df = pl.DataFrame(trimmed_df)

    return trimmed_df

# Notes on the trimming logic v0: 
# The predicted splice site position (column position) is part of the contextual sequence. Specifically:
# The position corresponds to the index of the splice site within the original sequence, and it is inclusive 
# in the current flanking window ...

# See splice_engine.sequence_featurizer.extract_analysis_sequences() for the original sequence extraction logic.
def trim_contextual_sequences(df, target_length=200):
    """
    Trim contextual sequences in the DataFrame to a specified target length.

    Parameters:
    - df (pd.DataFrame): DataFrame containing the columns 'position', 'window_start',
                         'window_end', and 'sequence'.
    - target_length (int): Desired length of the trimmed sequence (default: 200).

    Returns:
    - pd.DataFrame: Updated DataFrame with trimmed sequences and adjusted start/end windows.
    """
    is_polars = isinstance(df, pl.DataFrame)
    if is_polars:
        df = df.to_pandas()

    half_length = target_length // 2  # Half of the target length for symmetric trimming

    # Verify necessary columns exist
    required_columns = ["position", "window_start", "window_end", "sequence"]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' is missing in the input DataFrame.")

    def trim_row(row):
        position = row["position"]
        window_start = row["window_start"]
        window_end = row["window_end"]
        sequence = row["sequence"]

        # Current flanking lengths
        left_flank = position - window_start
        right_flank = window_end - position - 1  # Adjusted for inclusive counting

        # Validate extracted sequence length
        expected_length = left_flank + right_flank + 1  # Include the nucleotide at 'position'
        if len(sequence) != expected_length:
            raise ValueError(
                f"[error] Misalignment detected. Sequence length: {len(sequence)}, "
                f"Expected length: {expected_length}, "
                f"Left flank: {left_flank}, Right flank: {right_flank}, "
                f"Position: {position}, Window Start: {window_start}, Window End: {window_end}"
            )

        # Determine trimming boundaries
        new_start = max(position - half_length, window_start)  # Avoid going beyond sequence start
        new_end = min(position + half_length - 1, window_end - 1)  # Corrected for inclusive slicing

        # Trim the sequence
        trim_start_idx = new_start - window_start
        trim_end_idx = trim_start_idx + (new_end - new_start) + 1  # Include the endpoint
        trimmed_sequence = sequence[trim_start_idx:trim_end_idx]

        # Adjust the parameters to match the trimmed sequence
        trimmed_left_flank = position - new_start
        trimmed_right_flank = new_end - position

        # Validate trimmed sequence length
        trimmed_expected_length = trimmed_left_flank + trimmed_right_flank + 1
        if len(trimmed_sequence) != trimmed_expected_length:
            raise ValueError(
                f"[error] Post-trim misalignment. Trimmed sequence length: {len(trimmed_sequence)}, "
                f"Expected length: {trimmed_expected_length}, "
                f"Left flank: {trimmed_left_flank}, Right flank: {trimmed_right_flank}, "
                f"Position: {position}, New Start: {new_start}, New End: {new_end}"
            )

        # Update the row with new trimmed values
        row["window_start"] = new_start
        row["window_end"] = new_end + 1  # Adjust back to inclusive indexing
        row["sequence"] = trimmed_sequence
        return row

    # Apply trimming row by row
    trimmed_df = df.apply(trim_row, axis=1)

    # Convert back to Polars if the input was Polars
    if is_polars:
        trimmed_df = pl.DataFrame(trimmed_df)

    return trimmed_df


def prepare_dataset(df, tokenizer, pred_type='FP', label_column='label', sequence_column='sequence', max_length=512):
    """
    Prepare dataset for transformer-based models with explicit max_length.

    Parameters:
    - max_length (int): Maximum length for tokenization (default is 512 for DNABERT).
    """
    # Check if required columns are present
    required_columns = [label_column, sequence_column]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")

    is_polars = isinstance(df, pl.DataFrame)
    if is_polars: 
        df = df.to_pandas()

    # Encode sequences
    encodings = tokenizer(
        list(df[sequence_column]),
        padding=True,
        truncation=True,
        max_length=max_length,  # Explicit truncation length
        return_tensors="pt"
    )
    
    # Check if the label column is already encoded in integers
    if df[label_column].dtype == int:
        labels = torch.tensor(df[label_column].values)
    else:
        labels = torch.tensor((df[label_column] == pred_type).astype(int).values)  # FP=1, TP=0 given pred_type='FP'

    # Create PyTorch dataset
    dataset = torch.utils.data.TensorDataset(encodings['input_ids'], encodings['attention_mask'], labels)
    # NOTE: While this works with PyTorch DataLoader, it is incompatible with the Hugging Face Trainer.

    return dataset


def prepare_dataset_for_huggingface(
        df, tokenizer, pred_type='FP', label_column='label', sequence_column='sequence', max_length=512
    ):
    """
    Prepare dataset for transformer-based models with explicit max_length.

    Returns:
    - Hugging Face Dataset object compatible with the Trainer API.
    """
    from datasets import Dataset  # pip install datasets

    # Check if required columns are present
    required_columns = [label_column, sequence_column]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")

    is_polars = isinstance(df, pl.DataFrame)
    if is_polars:
        df = df.to_pandas()

    # Encode sequences
    encodings = tokenizer(
        list(df[sequence_column]),
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="np"  # Use NumPy for easier conversion to Hugging Face Dataset
    )

    # Add labels
    # Check if the label column is already encoded in integers
    if df[label_column].dtype == int:
        labels = df[label_column].values
    else:
        labels = (df[label_column] == pred_type).astype(int).values  # FP=1, TP=0 given pred_type='FP'

    # Combine encodings and labels into a dictionary
    dataset_dict = {
        'input_ids': encodings['input_ids'],
        'attention_mask': encodings['attention_mask'],
        'labels': labels
    }

    # Convert to Hugging Face Dataset
    dataset = Dataset.from_dict(dataset_dict)
    return dataset


def create_buckets(df, bucket_size=5000, sequence_column="sequence", tokenizer=None):
    """
    Create buckets for training based on sequence length.

    Parameters:
    - df (pd.DataFrame): Input DataFrame with sequences.
    - bucket_size (int): Number of samples per bucket.
    - sequence_column (str): Column containing sequences.
    - tokenizer: Tokenizer for sequence encoding (optional).

    Returns:
    - train_buckets (list): List of training DataFrames for each bucket.
    - val_buckets (list): List of validation DataFrames for each bucket.
    """
    # Add a column for sequence length
    df["sequence_length"] = df[sequence_column].apply(len)

    # Sort DataFrame by sequence length
    df = df.sort_values("sequence_length").reset_index(drop=True)

    # Create buckets by splitting the sorted DataFrame
    num_buckets = (len(df) + bucket_size - 1) // bucket_size  # Ceiling division
    buckets = [
        df.iloc[i * bucket_size: (i + 1) * bucket_size] for i in range(num_buckets)
    ]

    # Split each bucket into train/validation
    train_buckets, val_buckets = [], []
    for bucket in buckets:
        train_df, val_df = train_test_split(bucket, test_size=0.2, random_state=42)
        train_buckets.append(train_df)
        val_buckets.append(val_df)

    return train_buckets, val_buckets

####################################################################################################

def verify_sequence_lengths(df, sequence_col='sequence'):
    """
    Verify the length of the sequences from the specified sequence column.

    Parameters:
    - df (pl.DataFrame or pd.DataFrame): The input DataFrame.
    - sequence_col (str): The column name for sequences (default is 'sequence').

    Returns:
    - None
    """
    is_polars = isinstance(df, pl.DataFrame)

    if is_polars:
        # Use .apply on Polars expressions and collect the result as a Series
        # sequence_lengths = df.with_columns(
        #     pl.col(sequence_col).apply(lambda x: len(x)).alias('sequence_length')
        # )['sequence_length']
        # min_length = sequence_lengths.min()
        # max_length = sequence_lengths.max()
        # mean_length = sequence_lengths.mean()
        # median_length = sequence_lengths.median()
        df = df.to_pandas()
    
    sequence_lengths = df[sequence_col].apply(len)
    min_length = sequence_lengths.min()
    max_length = sequence_lengths.max()
    mean_length = sequence_lengths.mean()
    median_length = sequence_lengths.median()

    print_with_indent(f"[info] Sequence length statistics:", indent_level=1)
    print_with_indent(f"  - Min length: {min_length}", indent_level=2)
    print_with_indent(f"  - Max length: {max_length}", indent_level=2)
    print_with_indent(f"  - Mean length: {mean_length}", indent_level=2)
    print_with_indent(f"  - Median length: {median_length}", indent_level=2)

    return max_length


def validate_sequences(df, sequence_col="sequence"):
    invalid_rows = df[~df[sequence_col].str.contains("^[ACGTN]+$", regex=True)]
    if not invalid_rows.empty:
        print(f"Invalid sequences detected: {len(invalid_rows)}")
        raise ValueError("Sequence data contains invalid characters.")


def compute_attention_rollout_v0(attentions):
    """
    Aggregate attention weights across layers for cumulative importance.
    
    Parameters:
    - attentions (list): List of attention matrices (num_layers, batch_size, num_heads, seq_len, seq_len).
    
    Returns:
    - cumulative_attention: Aggregated attention matrix (seq_len, seq_len).
    """
    import numpy as np

    cumulative_attention = np.eye(attentions[0].shape[-1])  # Identity matrix
    for layer_attention in attentions:
        avg_layer_attention = layer_attention.mean(axis=1).detach().numpy()  # Average over heads
        cumulative_attention = avg_layer_attention @ cumulative_attention  # Recursive aggregation

    cumulative_attention /= cumulative_attention.sum(axis=-1, keepdims=True)  # Normalize
    return cumulative_attention


def compute_attention_rollout(attention_weights, include_residual=True):
    """
    Perform attention rollout to compute global attention scores.

    Parameters:
    - attention_weights (list of torch.Tensor): List of attention tensors from each layer.
      Each tensor has shape (batch_size, num_heads, seq_len, seq_len).
    - include_residual (bool): Whether to include residual connections in the attention scores.

    Returns:
    - numpy.ndarray: Rolled out attention map (seq_len x seq_len).
    """
    device = attention_weights[0].device
    global_attention = torch.eye(
        attention_weights[0].size(-1),
        device=device
    )

    for layer_attention in attention_weights:
        # 1) Average across heads
        layer_attention = layer_attention.mean(dim=1)  # (batch, seq_len, seq_len)

        # 2) Optional: add identity (residual)
        if include_residual:
            layer_attention += torch.eye(
                layer_attention.size(-1),
                device=layer_attention.device
            )

        # 3) Normalize row-wise
        layer_attention = layer_attention / layer_attention.sum(dim=-1, keepdim=True)

        # 4) "Rollout" = multiply cumulative attention with this layer's attention
        global_attention = torch.matmul(global_attention, layer_attention)

    # Remove batch dimension (assumes batch=1) and return NumPy array
    return global_attention.squeeze(0).detach().cpu().numpy()

# Considering residual connections, both max and average attention scores
def compute_attention_rollout_v1(
    attention_weights,
    include_residual=True,
    residual_scale=0.1,
    return_max_and_avg=False,
    remove_batch_dim=True
):
    """
    Improved rollout computation with normalized attention scores.

    Updates: 
    - Fixed the computation logic to ensure row-wise and global normalization of attention scores.
    """
    seq_len = attention_weights[0].size(-1)
    global_attention = torch.eye(seq_len).to(attention_weights[0].device)
    num_layers = len(attention_weights)

    max_scores = torch.zeros(seq_len).to(attention_weights[0].device)
    avg_scores = torch.zeros_like(max_scores)

    for layer_attention in attention_weights:
        layer_attention = layer_attention.mean(dim=1)  # Average across heads
        if include_residual:
            residual = residual_scale * torch.eye(seq_len).to(layer_attention.device)
            layer_attention += residual

        # Normalize attention scores row-wise
        layer_attention = layer_attention / layer_attention.sum(dim=-1, keepdim=True)

        # Aggregate global attention
        global_attention = torch.matmul(global_attention, layer_attention)

        # Compute max and average scores per token
        token_scores = layer_attention.sum(dim=0)  # Aggregate over queries
        max_scores = torch.max(max_scores, token_scores)
        avg_scores += token_scores / num_layers

    # Final normalization for global attention
    global_attention = global_attention / global_attention.sum(dim=-1, keepdim=True)

    if remove_batch_dim:
        global_attention = global_attention.mean(dim=0)  # Average over batch

    if return_max_and_avg:
        return (
            global_attention.detach().cpu().numpy(),
            max_scores.detach().cpu().numpy(),
            avg_scores.detach().cpu().numpy(),
        )
    return global_attention.detach().cpu().numpy()


def get_attention_weights_with_rollout(model, tokenizer, sequence, max_length=512, include_rollout=True, **kwargs):
    """
    Extract attention weights with optional attention rollout for a DNA sequence.

    Parameters:
    - model: Trained transformer model.
    - tokenizer: Tokenizer for the model.
    - sequence (str): DNA sequence.
    - max_length (int): Maximum length for tokenization (default: 512).
    - include_rollout (bool): Whether to compute attention rollout.

    Returns:
    - global_attention (numpy.ndarray): Rolled out attention map (if include_rollout=True).
    - tokenized_sequence (list): Tokenized sequence as a list of tokens.
    """
    # Tokenize the input sequence
    inputs = tokenizer(
        sequence,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
        **kwargs
    )

    # Forward pass with attention outputs enabled
    outputs = model(**inputs, output_attentions=True)

    # Ensure attentions are returned
    if not hasattr(outputs, "attentions") or outputs.attentions is None:
        raise ValueError(
            "The model did not return attention weights. "
            "Ensure `output_attentions=True` during model initialization."
        )

    # Extract tokenized sequence
    tokenized_sequence = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

    if include_rollout:
        global_attention = compute_attention_rollout(outputs.attentions)
        return global_attention, tokenized_sequence
    else:
        return outputs.attentions, tokenized_sequence


def map_and_validate_motifs_with_rollout(
    motifs, original_sequence, global_attention, tokenized_sequence
):
    """
    Map externally derived motifs to the original sequence and estimate attention scores using rollout.

    Parameters:
    - motifs: List of important motifs (strings).
    - original_sequence: Original DNA sequence (string).
    - global_attention: Rolled out attention map (seq_len x seq_len).
    - tokenized_sequence: Tokenized sequence as a list of tokens.

    Returns:
    - validated_motifs: List of motifs with attention scores and positions.
    """
    import re

    validated_motifs = []

    for motif in motifs:
        matches = [m.start() for m in re.finditer(re.escape(motif), original_sequence)]
        for match_pos in matches:
            motif_start, motif_end = match_pos, match_pos + len(motif)

            # Find the best token that fully overlaps or encloses the motif
            best_token = None
            best_token_score = 0

            for idx, token in enumerate(tokenized_sequence):
                token_start = original_sequence.find(token)
                token_end = token_start + len(token)

                # Check for full overlap or enclosure
                if token_start <= motif_start and token_end >= motif_end:
                    token_score = global_attention[idx, :].mean()  # Use row mean as the score
                    if token_score > best_token_score:
                        best_token = token
                        best_token_score = token_score

            # Record the best match
            if best_token:
                validated_motifs.append({
                    "motif": motif,
                    "start": motif_start,
                    "end": motif_end,
                    "best_token": best_token,
                    "attention_score": best_token_score
                })

    return validated_motifs





####################################################################################################

def log_memory_usage():
    """

    RSS (Resident Set Size): 
    The portion of memory occupied by a process that is held in RAM. 
    It includes all the memory that the process has allocated and is currently using, 
    including code, data, and stack. It does not include memory that is swapped out to disk.

    VMS (Virtual Memory Size): 
    The total amount of virtual memory used by a process. 
    It includes all the memory that the process has allocated, including memory that is swapped out to disk, 
    memory that is allocated but not used, and memory that is shared with other processes.
    """

    # import psutil
    process = psutil.Process()
    mem_info = process.memory_info()
    print(f"[Memory] RSS: {mem_info.rss / (1024 ** 2):.2f} MB | VMS: {mem_info.vms / (1024 ** 2):.2f} MB")


def log_gpu_memory_usage():
    # import torch 
    
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 2)
        reserved = torch.cuda.memory_reserved() / (1024 ** 2)
        print(f"[GPU Memory] Allocated: {allocated:.2f} MB | Reserved: {reserved:.2f} MB")
    else:
        print("[GPU Memory] No GPU detected.")


def compute_metrics(eval_pred):
    """
    Custom function to compute metrics during evaluation.

    Parameters:
    - eval_pred: A tuple (logits, labels) where logits may be part of a larger output structure.

    Returns:
    - A dictionary of metrics: F1, AUC, accuracy, etc.
    """
    outputs, labels = eval_pred

    # Extract logits if outputs is a tuple
    if isinstance(outputs, tuple):
        logits = outputs[0]  # Typically, the first element is the logits
    else:
        logits = outputs

    # Compute predictions
    predictions = logits.argmax(axis=-1)

    # Compute metrics
    f1 = f1_score(labels, predictions, average="weighted")
    auc = roc_auc_score(labels, logits[:, 1]) if logits.shape[1] > 1 else 0.0
    acc = accuracy_score(labels, predictions)
    return {"f1": f1, "auc": auc, "accuracy": acc}


def train_transformer_model(
    train_dataset,
    val_dataset,
    model_name="zhihan1996/DNABERT-2-117M",
    output_dir="./model_output",
    num_labels=2,
    num_train_epochs=3,

    batch_size=16,
    weight_decay=0.01,
    learning_rate=5e-5,
    warmup_steps=100,
    
    # save_steps=500,
    save_steps=0,  # Disable intermediate checkpoints by default
    save_best_model=True,  # Save only the best model

    logging_dir="./logs",
    logging_steps=100,
    eval_strategy="steps",

    lr_scheduler_type="reduce_on_plateau",  # Built-in scheduler
    early_stopping_patience=8,  # Number of epochs with no improvement for early stopping
    scheduler_metric="eval_loss",  # Metric for learning rate scheduler
    scheduler_mode="min",  # "min" for loss, "max" for metrics like F1

    use_auto_model=True,
    model_class=None,
    tokenizer=None,
    **kwargs
):
    """
    Pre-train and fine-tune a transformer model for sequence classification with manual epoch control.
    """
    from transformers import TrainingArguments, Trainer, EarlyStoppingCallback
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    # from torch.optim.lr_scheduler import ReduceLROnPlateau
    # from sklearn.metrics import f1_score, accuracy_score

    # Load tokenizer
    if tokenizer is None:
        # This is likely unnecessary if the inputs have already been tokenized
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load model
    if use_auto_model:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels, trust_remote_code=True
        )
    elif model_class:
        model = model_class.from_pretrained(
            model_name, num_labels=num_labels, output_attentions=True, attn_implementation="eager"
        )
    else:
        raise ValueError("Either use_auto_model must be True or model_class must be provided.")

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy=eval_strategy,
        save_steps=save_steps,
        
        per_device_train_batch_size=batch_size,  # Mini-batch size
        gradient_accumulation_steps=4,  # Accumulate gradients over 4 steps
        num_train_epochs=1,  # Handle epochs manually
        
        weight_decay=weight_decay,
        
        logging_dir=logging_dir,
        logging_steps=logging_steps,

        learning_rate=learning_rate,
        warmup_steps=warmup_steps,

        load_best_model_at_end=save_best_model,  # Load the best model at the end
        metric_for_best_model=scheduler_metric,  # Specify a metric to determine the best model

        lr_scheduler_type="reduce_lr_on_plateau",  # Corrected scheduler type

        **kwargs
    )

    # Define Trainer
    trainer = Trainer(
        model=model,
        args=training_args,

        train_dataset=train_dataset,
        eval_dataset=val_dataset,

        compute_metrics=compute_metrics,  # Pass the custom metrics function
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)
        ]
    )

    print_emphasized("[info] Starting training...")
    log_memory_usage()
    log_gpu_memory_usage()

    # Initialize variables for early stopping
    best_metric = np.inf if scheduler_mode in ("min", None) else -np.inf
    best_epoch = 0  # Track the epoch of the best metric
    no_improve_count = 0
    min_delta = 1e-4  # Minimal improvement threshold

    # Train model manually for multiple epochs
    for epoch in range(num_train_epochs):
        # print_with_indent(f"Starting epoch {epoch + 1}/{num_train_epochs}...", indent_level=1)
        print(f"=================== Epoch {epoch + 1}/{num_train_epochs} ===================")
        
        # Train for one epoch
        trainer.train()  # No checkpoint resumption
        # NOTE: 
        #  - Each call to trainer.train() will process the entire training dataset once unless interrupted or limited by specific arguments.
        #  - By setting num_train_epochs=1 in TrainingArguments, each call ensures a single epoch is executed without skipping.

        # Compute training metrics
        train_results = trainer.evaluate(train_dataset)
        train_f1 = train_results["eval_f1"]
        train_acc = train_results["eval_accuracy"]
        train_auc = train_results["eval_auc"]
        train_loss = train_results["eval_loss"]

        # Compute validation metrics
        val_results = trainer.evaluate(val_dataset)
        val_f1 = val_results["eval_f1"]
        val_acc = val_results["eval_accuracy"]
        val_auc = val_results["eval_auc"]
        val_loss = val_results["eval_loss"]

        # Early Stopping Logic
        if scheduler_mode is None: 
            no_improve_count = 0  # Disable early stopping
        else: 
            # improvement = (best_metric - val_loss) if scheduler_mode == "min" else (val_f1 - best_metric)
            if scheduler_mode == "min" and val_loss < best_metric:
                best_metric = val_loss
                best_epoch = epoch + 1
                no_improve_count = 0
            elif scheduler_mode == "max" and val_f1 > best_metric:
                best_metric = val_f1  # Choose a different metric for comparison
                no_improve_count = 0
            else:
                no_improve_count += 1

        combined_metrics = {
            "train_loss": train_loss, "eval_loss": val_loss,
            "train_f1": train_f1, "eval_f1": val_f1,
            "train_auc": train_auc, "eval_auc": val_auc,
            "train_accuracy": train_acc, "eval_accuracy": val_acc,
            "epoch": epoch + 1
        }
        
        print("=======================================================")
        print(f"Loss: {train_loss:.4f} (Train) -> {val_loss:.4f} (Val)")
        print(f"F1: {train_f1:.4f} (Train) -> {val_f1:.4f} (Val)")
        # print(f"Accuracy: {train_acc:.4f} (Train) -> {val_acc:.4f} (Val)")
        print(f"ROC AUC: {train_auc:.4f} (Train) -> {val_auc:.4f} (Val)")
        print("=======================================================")

        # Save metrics (training and validation) to a file for post-analysis
        metrics_log_path = os.path.join(output_dir, "training_metrics.log")
        with open(metrics_log_path, "a") as log_file:
            log_file.write(f"Epoch {epoch + 1}: {combined_metrics}\n")
        
        if save_best_model:
            mode_id = kwargs.get("model_id", "best_model")
            if no_improve_count == 0: 
                # Save only the best model
                best_model_path = os.path.join(output_dir, mode_id)
                trainer.save_model(best_model_path)
                print_with_indent(f"Best model saved to: {best_model_path}", indent_level=2)
            else: 
                pass
        else: 
            # Save model after each epoch
            epoch_output_dir = os.path.join(output_dir, f"epoch_{epoch + 1}")
            trainer.save_model(epoch_output_dir)
            print_with_indent(f"Model saved to: {epoch_output_dir}", indent_level=2)

        # Check for early stopping
        if no_improve_count >= early_stopping_patience:
            print("[info] Early stopping triggered.")
            print(f"[info] Early stopping triggered at epoch {epoch + 1}.")
            print(f"[info] Best metric ({scheduler_metric}): {best_metric:.4f} at epoch {best_epoch}.")
            break

        # Log memory usage
        log_memory_usage()
        log_gpu_memory_usage()

        # Clear GPU cache to free memory (only if a GPU is available)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    # End of training loop

    print(f"[info] Training completed. Best metric ({scheduler_metric}): {best_metric:.4f} at epoch {best_epoch}.")

    # Return the best model and trainer
    if save_best_model and best_model_path and os.path.exists(best_model_path):
        print_emphasized(f"[info] Loading the best model from: {best_model_path}")
        model = AutoModelForSequenceClassification.from_pretrained(best_model_path)

    return model, trainer




def demo_train_model_with_bucketing(analysis_sequence_df, **kargs): 
    from transformers.models.bert import BertForSequenceClassification
    from meta_spliceai.splice_engine.splice_error_analyzer import ErrorAnalyzer

    subject = "analysis_sequences_sampled"
    tokenizer_name = "zhihan1996/DNABERT-2-117M"
    model_name = "zhihan1996/DNABERT-2-117M"
    model_type = 'DNABERT'
    pred_type = "FP"
    experiment_name = f"hard_genes"
    
    max_length = 1000  # Maximum sequence length for tokenization   
    col_tid = 'transcript_id'

    output_dir = os.path.join(ErrorAnalyzer.analysis_dir, experiment_name)
    os.makedirs(output_dir, exist_ok=True)

    logging_dir = os.path.join(output_dir, "logs")
    os.makedirs(logging_dir, exist_ok=True)

    #########################################################

    # Step 1: Create Buckets
    train_buckets, val_buckets = create_buckets(
        analysis_sequence_df, bucket_size=5000, sequence_column="sequence"
    )

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, model_max_length=max_length)

    # Step 2: Train on Buckets from Shorter to Longer
    for bucket_idx, (train_bucket, val_bucket) in enumerate(zip(train_buckets, val_buckets), start=1):
        print(f"[info] Training on Bucket {bucket_idx}/{len(train_buckets)}")

        model_output_dir = os.path.join(output_dir, f"bucket_{bucket_idx}")
        
        model, trainer = train_transformer_model(
            train_dataset=prepare_dataset_for_huggingface(train_bucket, tokenizer),
            val_dataset=prepare_dataset_for_huggingface(val_bucket, tokenizer),
            model_name=model_name,
            model_class=BertForSequenceClassification,
            output_dir=model_output_dir,
            num_labels=2,
            num_train_epochs=10,
            batch_size=8,
            learning_rate=3e-5,
            warmup_steps=500,
            logging_steps=500,
            save_best_model=True,
            save_steps=1000
        )

    # Optional: Evaluate on All Buckets
    for bucket_idx, val_bucket in enumerate(val_buckets, start=1):
        val_results = trainer.evaluate(prepare_dataset_for_huggingface(val_bucket, tokenizer))
        print(f"Bucket {bucket_idx} Validation Results: {val_results}")
