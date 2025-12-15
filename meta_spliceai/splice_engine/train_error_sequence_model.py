import os, sys
import numpy as np
import random
import psutil  # pip install psutil
from typing import Union, List, Set

import pandas as pd
import polars as pl
from tabulate import tabulate
from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

from meta_spliceai.splice_engine.utils_bio import (
    normalize_strand
)

from meta_spliceai.splice_engine.utils_doc import (
    print_emphasized, 
    print_with_indent, 
    print_section_separator, 
    display, 
    display_dataframe_in_chunks
)

from meta_spliceai.splice_engine.visual_analyzer import (
    create_error_bigwig
)

from meta_spliceai.mllib import ModelTracker

from meta_spliceai.splice_engine.model_evaluator import ModelEvaluationFileHandler
from meta_spliceai.splice_engine.analyzer import Analyzer
from meta_spliceai.splice_engine.splice_error_analyzer import ErrorAnalyzer
from meta_spliceai.splice_engine.seq_model_utils import (
    verify_sequence_lengths, 
    compute_attention_rollout
)

import torch
print("[info] PyTorch version:", torch.__version__)

import transformers
print("[info] Transformers version:", transformers.__version__)

# Update torch and transformers to the latest versions as needed
# pip install --upgrade torch transformers

# May need to run: Your currently installed version of Keras is Keras 3, but this is not yet supported in Transformers.
# pip install tf-keras

from transformers import AutoConfig
from transformers import (
    AutoTokenizer,  # pip install transformers
    AutoModelForSequenceClassification,  # pip install einops
    Trainer, 
    TrainingArguments
)

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score


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


def create_buckets(df, tokenizer, bucket_thresholds, dataset_format="huggingface", **kwargs):
    """
    Create buckets based on sequence length and prepare datasets.

    Parameters:
    - df (pd.DataFrame): DataFrame with sequences and labels.
    - tokenizer: Tokenizer for encoding sequences.
    - bucket_thresholds (list): Sequence length thresholds for creating buckets.
    - dataset_format (str): Dataset format ("huggingface" or "pytorch").
    - kwargs: Additional arguments passed to dataset preparation functions.

    Returns:
    - buckets (list): List of datasets for each bucket.
    """
    buckets = []
    for threshold in bucket_thresholds:
        bucket = df[df["sequence_length"] <= threshold]
        if not bucket.empty:
            if dataset_format == "huggingface":
                bucket_dataset = prepare_dataset_for_huggingface(bucket, tokenizer, **kwargs)
            else:
                bucket_dataset = prepare_dataset(bucket, tokenizer, **kwargs)
            buckets.append(bucket_dataset)
        df = df[df["sequence_length"] > threshold]

    # Handle sequences longer than the last threshold
    if not df.empty:
        if dataset_format == "huggingface":
            longer_bucket = prepare_dataset_for_huggingface(df, tokenizer, **kwargs)
        else:
            longer_bucket = prepare_dataset(df, tokenizer, **kwargs)
        buckets.append(longer_bucket)

    return buckets


def create_buckets_by_sequence_length(df, bucket_size=5000, sequence_column="sequence", tokenizer=None):
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


def prepare_transformer_datasets_with_buckets(
    training_df,
    tokenizer_name="zhihan1996/DNABERT-2-117M",
    max_length=512,
    test_size=0.2,
    bucket_thresholds=[512, 1024],
    dataset_format="huggingface",
    return_dataframes=False,
    random_state=42,
    **kwargs
):
    """
    Prepare training and validation datasets with shuffling and bucketing.

    Parameters:
    - training_df (pd.DataFrame or pl.DataFrame): DataFrame containing the training data.
    - tokenizer_name: Name of the tokenizer.
    - max_length: Maximum sequence length for tokenization.
    - test_size: Proportion of the dataset to include in the validation split.
    - bucket_thresholds: Sequence length thresholds for creating buckets.
    - dataset_format: Dataset format ("huggingface" or "pytorch").
    - return_dataframes: Whether to return raw train/validation DataFrames.
    - random_state: Random seed for reproducibility.

    Returns:
    - train_buckets, val_buckets: Buckets of datasets for training and validation.
    """
    is_polars = isinstance(training_df, pl.DataFrame)
    if is_polars:
        training_df = training_df.to_pandas()

    # Shuffle the dataset
    training_df = training_df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    # Define tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, model_max_length=max_length)

    # Tokenize sequences and compute lengths
    training_df["sequence_length"] = training_df["sequence"].apply(len)

    # Split dataset into training and validation sets
    train_df, val_df = train_test_split(training_df, test_size=test_size, random_state=random_state)

    # Create buckets for training and validation datasets
    train_buckets = create_buckets(train_df, tokenizer, bucket_thresholds, dataset_format, **kwargs)
    val_buckets = create_buckets(val_df, tokenizer, bucket_thresholds, dataset_format, **kwargs)

    # Return dataframes and datasets if requested
    if return_dataframes:
        return train_buckets, val_buckets, train_df, val_df

    return train_buckets, val_buckets


def prepare_transformer_datasets(
    training_df, 
    tokenizer_name="zhihan1996/DNABERT-2-117M", 
    tokenizer=None,  
    test_size=0.2, 
    return_dataframes=False, 
    random_state=42,
    gene_level_split=False,  # New parameter: if True, split on gene level
    **kargs
):
    """
    Prepare training and validation datasets for transformer-based models.

    Parameters:
    - training_df (pd.DataFrame or pl.DataFrame): DataFrame containing the training data.
    - tokenizer_name (str): Name of the Hugging Face tokenizer to use.
    - tokenizer: Pre-initialized tokenizer (optional).
    - test_size (float): Proportion of the dataset to include in the validation split.
    - return_dataframes (bool): Whether to return train_df and val_df in addition to PyTorch datasets.
    - random_state (int): Random seed for reproducibility.
    - gene_level_split (bool): If True, subsample the data on the level of genes rather than rows.

    Returns:
    - train_dataset: PyTorch Dataset for training.
    - val_dataset: PyTorch Dataset for validation.
    - (optional) train_df: Training DataFrame (if return_dataframes=True).
    - (optional) val_df: Validation DataFrame (if return_dataframes=True).
    """
    col_label = kargs.get("col_label", "label")

    # Handle Polars DataFrames
    is_polars = isinstance(training_df, pl.DataFrame)
    if is_polars:
        training_df = training_df.to_pandas()

    # Validate required columns
    required_columns = ["sequence", col_label]
    missing_columns = [col for col in required_columns if col not in training_df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in the input DataFrame: {missing_columns}")

    # Split dataset: either gene-level or row-level
    if gene_level_split:
        if "gene_id" not in training_df.columns:
            raise ValueError("gene_level_split is enabled but the 'gene_id' column is missing in the DataFrame.")
        unique_genes = training_df["gene_id"].unique()
        train_genes, val_genes = train_test_split(unique_genes, test_size=test_size, random_state=random_state)
        train_df = training_df[training_df["gene_id"].isin(train_genes)]
        val_df = training_df[training_df["gene_id"].isin(val_genes)]
    else:
        train_df, val_df = train_test_split(training_df, test_size=test_size, random_state=random_state)
    
    # Shuffle training set
    train_df = train_df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    # Initialize tokenizer
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, model_max_length=kargs.get("max_length", 512))
        print(f"[info] Initialized tokenizer with model_max_length={kargs.get('max_length', 512)}")
    else:
        print(f"[info] Using pre-initialized tokenizer: {tokenizer}")

    # Prepare datasets in the desired format (default: Hugging Face datasets)
    dataset_format = kargs.get("dataset_format", "huggingface")
    if dataset_format.startswith("hugging"):
        train_dataset = prepare_dataset_for_huggingface(train_df, tokenizer, pred_type=kargs.get("pred_type", "FP"))
        val_dataset = prepare_dataset_for_huggingface(val_df, tokenizer, pred_type=kargs.get("pred_type", "FP"))
    else:
        train_dataset = prepare_dataset(train_df, tokenizer, pred_type=kargs.get("pred_type", "FP"))
        val_dataset = prepare_dataset(val_df, tokenizer, pred_type=kargs.get("pred_type", "FP"))

    if return_dataframes:
        return train_dataset, val_dataset, train_df, val_df

    return train_dataset, val_dataset


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


def define_training_arguments(output_dir, eval_strategy, save_steps, batch_size, weight_decay, logging_dir, logging_steps, learning_rate, warmup_steps, save_best_model, scheduler_metric, **kwargs):
    """
    Define training arguments for the transformer model.

    Parameters:
    - output_dir (str): Directory to save the model.
    - eval_strategy (str): Evaluation strategy.
    - save_steps (int): Number of steps between model saves.
    - batch_size (int): Mini-batch size.
    - weight_decay (float): Weight decay for regularization.
    - logging_dir (str): Directory for logging.
    - logging_steps (int): Number of steps between logging.
    - learning_rate (float): Learning rate.
    - warmup_steps (int): Number of warmup steps.
    - save_best_model (bool): Whether to save the best model at the end.
    - scheduler_metric (str): Metric to determine the best model.
    - **kwargs: Additional optional parameters.

    Returns:
    - TrainingArguments: The training arguments.
    """
    # Print the contents of **kwargs
    print("[info] Additional optional parameters passed to TrainingArguments:")
    for key, value in kwargs.items():
        print(f"  {key}: {value}")

    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy=eval_strategy,
        save_steps=save_steps,
        per_device_train_batch_size=batch_size,  # Mini-batch size
        # gradient_accumulation_steps=4,  # Accumulate gradients over 4 steps
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

    return training_args


def is_distributed_training():
    """
    Detect if the current training session is in distributed mode.

    Returns:
    - distributed (bool): True if distributed training is enabled and initialized, False otherwise.
    """
    # Check if distributed mode is supported and initialized
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def set_device_v0():
    """
    Determine the device to use (CPU or GPU) and set it for distributed training if applicable.

    Returns:
    - device (torch.device): The device to use for training.
    - local_rank (int): The local rank of the process (default 0 for single process).
    """
    import torch

    # Get the local rank from the environment variable set by torchrun
    local_rank = int(os.environ.get("LOCAL_RANK", 0))  # Default to 0 if not found

    if torch.cuda.is_available():
        # Use CUDA if available
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        print(f"[INFO] Using CUDA device: {device}")
    else:
        # Fallback to CPU
        device = torch.device("cpu")
        print("[INFO] CUDA not available. Using CPU.")

    return device, local_rank


def set_device():
    # import torch
    # import os

    if torch.cuda.is_available():
        # If launched with torchrun or launch, 'LOCAL_RANK' is set
        local_rank_env = os.environ.get('LOCAL_RANK', None)
        if local_rank_env is not None:
            local_rank = int(local_rank_env)
            device = torch.device(f"cuda:{local_rank}")
        else:
            # Single-GPU or a manual environment without local_rank
            local_rank = 0
            device = torch.device("cuda:0")
    else:
        # CPU-only case
        device = torch.device("cpu")
        local_rank = -1

    return device, local_rank


def setup_device_and_model(model, distributed=False, verbose=1):
    """
    Set up the device and model for single-GPU, multi-GPU, or CPU training.

    Parameters:
    - model: The model to set up.
    - distributed (bool): Whether to use DistributedDataParallel.

    Returns:
    - model: The model, moved to the appropriate device and optionally wrapped in DDP.
    - device: The torch.device used for training.
    """
    import torch

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


def train_transformer_model(
    train_dataset,  # Tokenized training dataset
    val_dataset,  # Tokenized validation dataset
    
    # Paramters for TrainingArguments
    training_arguments=None,  # Additional parameters for TrainingArguments

    # Additional parameters for training
    model_name="zhihan1996/DNABERT-2-117M",
    num_labels=2,  # Number of target labels for classification (binary by default)
    num_train_epochs=3,  # Number of training epochs but not to be passed to TrainingArguments
    model_class=None,  # Custom model class (if not using AutoModel)
    tokenizer=None,  # Custom tokenizer (optional)
    train_model=True,  # If False, skip training and directly load the best model
    **kwargs
):
    """
    Fine-tune a transformer model for sequence classification.

    This function is designed for CPU, single-GPU, or multi-GPU (DistributedDataParallel) setups.
    It supports evaluation at both step-level and epoch-level intervals and can handle early stopping.

    Parameters:
        - train_dataset: Tokenized dataset for training.
        - val_dataset: Tokenized dataset for validation.
        - model_name: Hugging Face model name or path to a pretrained model.
        - eval_strategy: Strategy for evaluation (e.g., "steps" or "epochs").
        - scheduler_mode: Mode for learning rate scheduler ("min" or "max").
        - return_path_to_model: Return model path in addition to the model and trainer.

    Returns:
        - model: Fine-tuned model.
        - trainer: Hugging Face Trainer object for the model.

    Memo: 
    - Pre-tokenized Datasets:
        Input datasets already include input IDs and attention masks (and possibly token type IDs) 
        produced by a tokenizer. The Trainer and model work directly with these tensors.
    - Trainer API: 
        The Hugging Face Trainer doesn't require the tokenizer to perform training or evaluation. 
        The tokenizer is primarily needed during data preprocessing (e.g., tokenizing raw text) 
        and for post-processing (e.g., decoding predictions for display or metrics computation).

    """
    from transformers import TrainingArguments, Trainer, EarlyStoppingCallback
    from torch.nn.parallel import DistributedDataParallel as DDP
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    # from sklearn.metrics import f1_score, accuracy_score

    # --- Distributed Training Setup ---
    device = kwargs.get("device", None)
    if device is None: 
        device, local_rank = set_device()
        if 'local_rank' not in training_arguments:
            training_arguments['local_rank'] = local_rank

    # --- GPU Detection ---
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        print(f"[info] Detected {num_gpus} GPUs. Running multi-GPU if 'torchrun' was used.")
    elif num_gpus == 1:
        print("[info] Single GPU available.")
    else:
        print("[info] Using CPU-only mode.")

    # --- Tokenizer Initialization ---
    # Load tokenizer
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    # --- Model Initialization ---
    use_auto_model = kwargs.get("use_auto_model", True)
    if model_class is not None: 
        use_auto_model = False

    # Load model
    if use_auto_model:
        # (A) Load config from the remote checkpoint
        # config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        # Override fields if needed
        # config.num_labels = num_labels
        # config.output_attentions = True

        # print(f"[debug] Loaded config:\n{config}\n")

        # (B) Instantiate the model from the pretrained checkpoint + config
        #     (This ensures custom code from the DNABERT repo is used.)
        # model = AutoModelForSequenceClassification.from_pretrained(
        #     model_name, 
        #     config=config, 
        #     trust_remote_code=True
        # )

        # --- Don't use config ---
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, trust_remote_code=False,
            num_labels=2,               # merges into config.num_labels
            output_attentions=True,      # merges into config.output_attentions
            attn_implementation="eager"  # This tells Transformers: “Use the manual (aka ‘eager’) attention, skip the SDPA logic.”
        )
        # model.config.num_labels = 2
        # model.config.output_attentions = True
        # model.config.attn_implementation = "eager"

        # NOTE: 
        # trust_remote_code=True is only relevant when auto_map is specified in config.json
        # - The following entry is currently removed from the config.json file:
        #     "auto_map":
        #   {"AutoConfig": "configuration_bert.BertConfig",
        #     "AutoModel": "bert_layers.BertModel",
        #     "AutoModelForMaskedLM": "bert_layers.BertForMaskedLM",
        #     "AutoModelForSequenceClassification": "bert_layers.BertForSequenceClassification"
        #   }
        # - "model_type": "bert" is added to the config.json file

        # Load config + weights directly
        # model = AutoModelForSequenceClassification.from_pretrained(
        #     model_name,
        #     trust_remote_code=True,
        #     num_labels=num_labels,          # override label count
        #     output_attentions=True          # if you want attention maps
        # )
        # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    elif model_class:
        model = model_class.from_pretrained(
            model_name, num_labels=num_labels, output_attentions=True
            # attn_implementation="eager"
        )
        # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    else:
        raise ValueError("Either use_auto_model must be True or model_class must be provided.")

    # Test
    print_emphasized("[info] Additional optional parameters passed to TrainingArguments:")
    for key, value in kwargs.items():
        print_with_indent(f"  {key}: {value}", indent_level=1)

    # Additional Trainer parameters (not to be included in TrainingArguments)
    early_stopping_patience = kwargs.get("early_stopping_patience", 5)

    # --- Training Arguments ---
    training_args = TrainingArguments(
        **training_arguments,
        # **kwargs
    )

    # --- Trainer Initialization ---
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

    output_dir = training_arguments["output_dir"]
    model_id = kwargs.get("model_id", "best_model")  # Identifier for the best model
    best_model_path = os.path.join(output_dir, model_id)
    scheduler_metric = kwargs.get("scheduler_metric", training_arguments.get("metric_for_best_model", "eval_loss"))
    scheduler_mode = kwargs.get("scheduler_mode", "min")
    save_best_model = kwargs.get("save_best_model", training_arguments.get("load_best_model_at_end", True))
    eval_strategy = training_arguments.get("eval_strategy", "epoch")

    # --- Training Logic ---
    if train_model: 
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
            
            trainer.train()  # Train for one epoch 
            # NOTE: 
            #  - Each call to trainer.train() will process the entire training dataset once unless interrupted or limited by specific arguments.
            #  - By setting num_train_epochs=1 in TrainingArguments, each call ensures a single epoch is executed without skipping.

            # --- Epoch-Level Evaluation ---
            if eval_strategy.startswith("epoch"):
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
                        best_metric = val_f1  # Todo: Choose a different metric for comparison
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
                    if no_improve_count == 0: 
                        # Save only the best model
                        # best_model_path = os.path.join(output_dir, model_id)
                        trainer.save_model(best_model_path)
                        print_with_indent(f"Best model saved to: {best_model_path}", indent_level=2)
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

            # --- Step-Level Evaluation ---
            elif eval_strategy.startswith("step"):
                # Let Trainer handle evaluation automatically
                print("[info] Using 'steps' evaluation strategy. Trainer will evaluate automatically at specified steps.")
                if trainer.state.best_model_checkpoint:
                    print_with_indent(f"Best model checkpoint: {trainer.state.best_model_checkpoint}", indent_level=1)
                else:
                    print("[warning] No best model checkpoint identified. Check training arguments.")
            else:
                raise ValueError(f"Unsupported evaluation strategy: {eval_strategy}")

            # Log memory usage
            log_memory_usage()
            log_gpu_memory_usage()

            # Clear GPU cache to free memory (only if a GPU is available)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        # End of training loop

        print(f"[info] Training completed. Best metric ({scheduler_metric}): {best_metric:.4f} at epoch {best_epoch}.")
    else: 
        # --- Load Pretrained Model ---
        # Return the best model and trainer
        if best_model_path and os.path.exists(best_model_path):
            print_emphasized(f"[info] Loading the best model from: {best_model_path}")
            model = AutoModelForSequenceClassification.from_pretrained(best_model_path)

            distributed = is_distributed_training()
            model, device = setup_device_and_model(model, distributed=distributed)
        else: 
            raise FileNotFoundError(f"Best model not found at: {best_model_path}")

    # --- Return Results ---
    if kwargs.get("return_path_to_model", False):
        return model, trainer, best_model_path

    return model, trainer


def evaluate_and_plot_metrics(trainer, val_dataset, output_dir, label_names=None):
    """
    Evaluate model on the validation set and plot ROC AUC and PRC AUC.
    """
    from sklearn.metrics import (
        classification_report, 
        roc_auc_score, 
        precision_recall_curve, 
        auc, 
        RocCurveDisplay, 
        PrecisionRecallDisplay
    )
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    # Run predictions
    predictions = trainer.predict(val_dataset)
    logits = predictions.predictions
    labels = predictions.label_ids

    # Debug logits
    # print(f"[DEBUG] Logits type: {type(logits)}")
    # print(f"[DEBUG] Logits content: {logits}")

    # Ensure logits is a NumPy array or list of arrays
    if isinstance(logits, dict):
        logits = logits["logits"]  # Extract logits from dictionary
    elif isinstance(logits, tuple):
        logits = logits[0]  # Extract logits from tuple
        print("[info] Number of structures in tuple:", len(logits))
    elif isinstance(logits, list) and isinstance(logits[0], np.ndarray):
        logits = np.concatenate(logits, axis=0)
    elif not isinstance(logits, np.ndarray):
        raise ValueError("Logits must be a NumPy array or a list of arrays.")

    # Ensure logits has correct dimensions
    if logits.ndim != 2:
        raise ValueError(f"Logits must be a 2D array, but got shape: {logits.shape}")

    if labels is None:
        raise ValueError("Labels not available in predictions.")

    # Compute probabilities using softmax
    probs = np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)

    # Compute predicted classes
    predictions = np.argmax(logits, axis=-1)

    # Classification report
    validation_report = classification_report(labels, predictions, target_names=label_names, output_dict=True)
    print("Validation Classification Report:\n", validation_report)

    # ROC AUC
    roc_auc = roc_auc_score(labels, probs[:, 1])  # Assuming binary classification
    RocCurveDisplay.from_predictions(labels, probs[:, 1])
    plt.title(f"ROC AUC (Validation): {roc_auc:.4f}")
    roc_path = os.path.join(output_dir, "roc_auc_validation.pdf")
    plt.savefig(roc_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"[INFO] ROC AUC plot saved to {roc_path}")

    # Precision-Recall Curve (PRC AUC)
    precision, recall, _ = precision_recall_curve(labels, probs[:, 1])
    pr_auc = auc(recall, precision)
    PrecisionRecallDisplay(precision=precision, recall=recall, average_precision=pr_auc).plot()
    plt.title(f"PRC AUC (Validation): {pr_auc:.4f}")
    prc_path = os.path.join(output_dir, "prc_auc_validation.pdf")
    plt.savefig(prc_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"[INFO] PRC AUC plot saved to {prc_path}")

    return validation_report


def get_attention_weights(model, tokenizer, sequence, layer=None, head=None, max_length=512, policy="mean", **kwargs):
    """
    Extract and normalize attention weights for a sequence.

    Parameters:
    - model: Trained transformer model.
    - tokenizer: Tokenizer for the model.
    - sequence (str): Input sequence.
    - layer (int or None): Specific transformer layer (None = aggregate across all layers).
    - head (int or None): Specific attention head (None = average across all heads).
    - max_length (int): Maximum sequence length for tokenization.
    - policy (str): Aggregation policy ("mean", "rollout").

    Returns:
    - attention_weights (numpy.ndarray): 2D attention matrix (seq_len x seq_len).
    - tokenized_sequence (list): List of tokenized tokens.
    """
    import numpy as np
    import torch

    # Tokenize the input sequence
    inputs = tokenizer(sequence, truncation=True, max_length=max_length, return_tensors="pt", **kwargs)

    # Figure out which device the model is on
    model_device = next(model.parameters()).device

    # Move inputs to the model's device
    inputs = {k: v.to(model_device) for k, v in inputs.items()}

    # Forward pass with attention outputs
    with torch.no_grad():
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            output_attentions=True
        )
    # NOTE: If the model is on cuda:0 (or cuda:1) but inputs["input_ids"] is still on CPU, PyTorch raises a mismatch error

    # Ensure attentions are returned
    if not hasattr(outputs, "attentions") or outputs.attentions is None:
        raise ValueError("Model did not return attention weights. Enable `output_attentions=True`.")

    # Extract tokenized sequence
    tokenized_sequence = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    # Aggregate attention across layers and heads
    attentions = outputs.attentions
    if policy == "mean":
        # Average attention across all layers and heads
        attention_weights = torch.stack(attentions).mean(dim=0).mean(dim=1).squeeze(0).cpu().numpy()
    elif policy == "rollout":
        # Compute attention rollout
        print("[info] Computing attention rollout...")
        attention_weights = compute_attention_rollout(attentions)
    elif layer is not None:
        # Attention from a specific layer
        print(f"[info] Extracting attention from layer {layer}...")
        attention_weights = attentions[layer]
        if head is not None:
            # Attention from a specific head
            print(f"[info] Extracting attention from head {head}...")
            attention_weights = attention_weights[:, head].squeeze(0).cpu().numpy()
        else:
            # Average attention across all heads
            attention_weights = attention_weights.mean(dim=1).squeeze(0).cpu().numpy()
    else:
        raise ValueError("Invalid policy. Use 'mean', 'rollout', or specify layer and head.")

    # Normalize attention scores
    attention_weights = (attention_weights - attention_weights.min()) / (attention_weights.max() - attention_weights.min() + 1e-8)

    return attention_weights, tokenized_sequence


# Adding max and average attention scores
def get_attention_weights_v1(
    model,
    tokenizer,
    sequence,
    layer=0,
    head=0,
    max_length=512,
    policy="rollout",
    return_max_and_avg=True,
    **kwargs,
):
    """
    Get attention weights and optionally max and average attention scores.

    Parameters:
    - model: Trained transformer model.
    - tokenizer: Tokenizer for the model.
    - sequence: Input sequence (string).
    - layer: Transformer layer to extract attention weights from.
    - head: Attention head to visualize.
    - max_length: Maximum token length for input.
    - policy: Aggregation policy ("rollout" or "mean").
    - return_max_and_avg (bool): Whether to return max and average scores.

    Returns:
    - If return_max_and_avg=False: Attention weights and tokenized sequence.
    - If return_max_and_avg=True: Tuple (attention_weights, tokenized_sequence, max_scores, avg_scores).

    - attention_weights is 2D NumPy array of shape (seq_len, seq_len)
        - Aggregated or rolled-out attention weights where the (i, j) entry represents 
          the attention the i-th query token pays to the j-th key token.
    - tokenized_sequence is a list of tokens (strings) obtained from the tokenizer.
        - Includes special tokens (e.g., [CLS], [SEP]) used by the tokenizer, if applicable.

    Memos: 
    - Key updates: 
        Flexibility: Allows you to retrieve both max and average scores, enabling better interpretation 
        of token importance.
        Compatibility: Works seamlessly with visualize_attention_regions and visualize_attention_weights.
        Improved Interpretability: Distinguishes between global significance (average) 
        and localized peaks (maximum).

    """
    inputs = tokenizer(sequence, truncation=True, max_length=max_length, return_tensors="pt", **kwargs)

    # Forward pass with attention outputs enabled
    outputs = model(**inputs, output_attentions=True)

    attentions = outputs.attentions  # Extract attention tensors
    if policy == "rollout":
        results = compute_attention_rollout(
            attentions, return_max_and_avg=return_max_and_avg, remove_batch_dim=True)
        if return_max_and_avg:
            global_attention, max_scores, avg_scores = results
            return global_attention, tokenizer.convert_ids_to_tokens(inputs["input_ids"][0]), max_scores, avg_scores
        return results, tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    elif policy == "mean":
        layer_attention = attentions[layer][:, head]  # Extract specific layer and head
        attention_weights = layer_attention.squeeze(0).detach().cpu().numpy()
        return attention_weights, tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    elif layer is not None:
        # Attention from a specific layer
        attention_weights = attentions[layer]
        if head is not None:
            # Attention from a specific head
            attention_weights = attention_weights[:, head].squeeze(0).cpu().numpy()
        else:
            # Average attention across all heads
            attention_weights = attention_weights.mean(dim=1).squeeze(0).cpu().numpy()
    else:
        raise ValueError("Invalid policy. Use 'mean', 'rollout', or specify layer and head.")

    raise ValueError(f"Unknown policy: {policy}")


def visualize_attention_weights_v0(
    attention_weights,
    tokenized_sequence,
    output_path=None,
    file_format="pdf",
    highlight_top_k=None,
    color_map="coolwarm"  # Custom color scheme
):
    """
    Visualize attention weights as a heatmap with optional highlighting of important subsequences.

    Parameters:
    - attention_weights: 2D NumPy array of attention weights for a specific layer and head.
    - tokenized_sequence: Tokenized sequence as a list of tokens.
    - output_path (str): File path to save the heatmap.
    - file_format (str): File format for saving the plot (default: 'pdf').
    - highlight_top_k (int): Number of top tokens to highlight based on aggregate attention scores.
    - color_map (str): Color map for the heatmap.

    Raises:
    - ValueError: If the attention_weights format is unexpected.

    Memo: 
    - Why normalize attention weights?
        attention_weights are values computed directly from the attention matrices of the transformer model.
        These values tend to be very small and close to uniform because the attention mechanism often uses softmax normalization, 
        leading to relatively flat distributions across tokens.
    - The column-wise aggregation focuses on the importance of tokens as keys, which aligns with:
        - Interpreting Influential Tokens: Helps identify tokens in the sequence that the model uses most to make predictions.
        - Highlighting Regions of Interest: Regions with high aggregate scores might correspond to 
          biologically meaningful motifs or splice site indicators.

    - Interpretation of Scores in the Heatmap
        Rows: Represent queries (tokens asking questions). The heatmap entries in row i show how much the query token i 
              attends to other tokens (keys).
        Columns: Represent keys (tokens being focused on). The normalized aggregate score for column j indicates 
                how important token j is as a key in the sequence.
        Annotated Tokens: 
            The annotated normalized score for each token reflects how much that token influences 
            the self-attention computation as a key across all queries.

    """
    # Validate attention_weights format
    if not isinstance(attention_weights, np.ndarray) or attention_weights.ndim != 2:
        raise ValueError("Expected attention_weights to be a 2D NumPy array.")

    # Validate sequence length matches attention dimensions
    seq_len = len(tokenized_sequence)
    if attention_weights.shape != (seq_len, seq_len):
        raise ValueError(
            f"Tokenized sequence length ({seq_len}) does not match attention dimensions {attention_weights.shape}."
        )

    # Normalize attention weights for consistent scaling
    attention_weights_normalized = (attention_weights - np.min(attention_weights)) / (
        np.max(attention_weights) - np.min(attention_weights) + 1e-8
    )

    # Compute aggregate attention scores for each token
    aggregate_scores = attention_weights.sum(axis=0)  # Sum over all query rows

    # Normalize scores
    aggregate_scores_normalized = (aggregate_scores - np.min(aggregate_scores)) / (
        np.max(aggregate_scores) - np.min(aggregate_scores) + 1e-8
    )

    # Highlight top-k tokens based on aggregate scores
    highlight_indices = None
    if highlight_top_k:
        highlight_indices = np.argsort(aggregate_scores_normalized)[-highlight_top_k:]

    # Annotate tokens with their normalized scores
    annotated_tokens = [
        f"{token} ({aggregate_scores_normalized[i]:.2f})" for i, token in enumerate(tokenized_sequence)
    ]

    # Plot heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        attention_weights_normalized,
        xticklabels=annotated_tokens,
        yticklabels=annotated_tokens,
        cmap=color_map
    )
    plt.title("Attention Weights Heatmap")
    plt.xlabel("Tokenized Sequence Position")
    plt.ylabel("Tokenized Sequence Position")

    # Highlight important tokens
    if highlight_indices is not None:
        for idx in highlight_indices:
            plt.axvline(x=idx + 0.5, color="red", linestyle="--", linewidth=1)
            plt.axhline(y=idx + 0.5, color="red", linestyle="--", linewidth=1)

    # Save or show the plot
    if output_path:
        plt.savefig(output_path, format=file_format, bbox_inches="tight", dpi=200)
        print(f"[info] Attention weights heatmap saved to: {output_path}")
        plt.close()
    else:
        plt.show()

# Added functionality to highlight off-diagonal attention weights
def visualize_attention_weights(
    attention_weights,
    tokenized_sequence,
    output_path=None,
    file_format="pdf",
    highlight_top_k=None,
    color_map="coolwarm",  # Custom color scheme
    amplify_off_diagonal=True,  # Amplify off-diagonal weights
    amplification_factor=2.0  # Factor to amplify off-diagonal weights
):
    """
    Visualize attention weights as a heatmap with enhanced off-diagonal highlights.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Validate attention_weights format
    if not isinstance(attention_weights, np.ndarray) or attention_weights.ndim != 2:
        raise ValueError("Expected attention_weights to be a 2D NumPy array.")

    seq_len = len(tokenized_sequence)
    if attention_weights.shape != (seq_len, seq_len):
        raise ValueError(
            f"Tokenized sequence length ({seq_len}) does not match attention dimensions {attention_weights.shape}."
        )

    # Normalize attention weights for consistent scaling
    attention_weights_normalized = (attention_weights - np.min(attention_weights)) / (
        np.max(attention_weights) - np.min(attention_weights) + 1e-8
    )

    # Optionally amplify off-diagonal weights
    if amplify_off_diagonal:
        diag_mask = np.eye(seq_len)
        attention_weights_normalized += amplification_factor * (1 - diag_mask) * attention_weights_normalized

    # Compute aggregate attention scores for each token
    aggregate_scores = attention_weights.sum(axis=0)

    # Normalize scores
    aggregate_scores_normalized = (aggregate_scores - np.min(aggregate_scores)) / (
        np.max(aggregate_scores) - np.min(aggregate_scores) + 1e-8
    )

    # Highlight top-k tokens based on aggregate scores
    highlight_indices = None
    if highlight_top_k:
        highlight_indices = np.argsort(aggregate_scores_normalized)[-highlight_top_k:]

    # Annotate tokens with their normalized scores
    annotated_tokens = [
        f"{token} ({aggregate_scores_normalized[i]:.2f})" for i, token in enumerate(tokenized_sequence)
    ]

    # Plot heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        attention_weights_normalized,
        xticklabels=annotated_tokens,
        yticklabels=annotated_tokens,
        cmap=color_map
    )
    plt.title("Enhanced Attention Weights Heatmap")
    plt.xlabel("Tokenized Sequence Position")
    plt.ylabel("Tokenized Sequence Position")

    # Highlight important tokens
    if highlight_indices is not None:
        for idx in highlight_indices:
            plt.axvline(x=idx + 0.5, color="red", linestyle="--", linewidth=1)
            plt.axhline(y=idx + 0.5, color="red", linestyle="--", linewidth=1)

    # Save or show the plot
    if output_path:
        plt.savefig(output_path, format=file_format, bbox_inches="tight", dpi=200)
        print(f"[info] Attention weights heatmap saved to: {output_path}")
        plt.close()
    else:
        plt.show()


def visualize_attention_weights_with_special_tokens_filtered(
    attention_weights,
    tokenized_sequence,
    output_path=None,
    file_format="pdf",
    highlight_top_k=None,
    color_map="coolwarm",  # Custom color scheme
    amplify_off_diagonal=True,  # Amplify off-diagonal weights
    amplification_factor=2.0,  # Factor to amplify off-diagonal weights
    special_tokens=("[CLS]", "[SEP]", "[PAD]", "[UNK]")
):
    """
    Visualize attention weights as a heatmap with enhanced off-diagonal highlights,
    while excluding special tokens (rows/columns) from the plot.

    Parameters
    ----------
    attention_weights : np.ndarray, shape (seq_len, seq_len)
        2D attention matrix for each pair of tokens.
    tokenized_sequence : list of str
        List of tokens, length = seq_len. May contain special tokens.
    output_path : str or None
        If set, save the figure as a PDF (default) or the specified file_format.
    file_format : str
        Format to save the figure, e.g. "pdf" or "png".
    highlight_top_k : int or None
        If set, highlight the top-k tokens by aggregate attention. 
    color_map : str
        Seaborn/Matplotlib colormap, e.g. "coolwarm".
    amplify_off_diagonal : bool
        If True, multiply off-diagonal elements by 'amplification_factor'.
    amplification_factor : float
        Factor to amplify off-diagonal elements if amplify_off_diagonal=True.
    special_tokens : tuple or set
        Tokens to exclude from the heatmap, e.g. ("[CLS]", "[SEP]", "[PAD]", "[UNK]").

    Returns
    -------
    None
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    # 1) Validate input
    if not isinstance(attention_weights, np.ndarray) or attention_weights.ndim != 2:
        raise ValueError("Expected attention_weights to be a 2D NumPy array.")

    seq_len = len(tokenized_sequence)
    if attention_weights.shape != (seq_len, seq_len):
        raise ValueError(
            f"Tokenized sequence length ({seq_len}) does not match attention dimensions {attention_weights.shape}."
        )

    # 2) Identify valid (non-special) tokens
    if special_tokens is None: 
        special_tokens = []
    special_tokens_set = set(special_tokens) 
    valid_indices = [
        i for i, tk in enumerate(tokenized_sequence) 
        if tk not in special_tokens_set
    ]
    # If all tokens are special, we can't plot anything
    if not valid_indices:
        print("[warning] All tokens were special. No heatmap to display.")
        return

    # 3) Subset both the token list and the attention matrix
    filtered_tokens = [tokenized_sequence[i] for i in valid_indices]
    filtered_attn   = attention_weights[np.ix_(valid_indices, valid_indices)]
    filtered_seq_len= len(filtered_tokens)

    # 4) Normalize the filtered attention weights
    min_val = filtered_attn.min()
    max_val = filtered_attn.max()
    denom = (max_val - min_val) + 1e-8
    attn_norm = (filtered_attn - min_val) / denom

    # Optionally amplify off-diagonal
    if amplify_off_diagonal:
        diag_mask = np.eye(filtered_seq_len)
        attn_norm += amplification_factor * (1 - diag_mask) * attn_norm

    # 5) Aggregate scores for each token (sum across rows or columns)
    #    e.g., sum across columns to see how strongly each token is attended
    aggregate_scores = filtered_attn.sum(axis=0)
    # Normalize aggregate scores
    aggr_min, aggr_max = aggregate_scores.min(), aggregate_scores.max()
    aggr_denom = (aggr_max - aggr_min) + 1e-8
    aggregate_scores_normalized = (aggregate_scores - aggr_min) / aggr_denom

    # 6) Decide if we highlight top-k tokens
    highlight_indices = None
    if highlight_top_k and highlight_top_k < filtered_seq_len:
        highlight_indices = np.argsort(aggregate_scores_normalized)[-highlight_top_k:]
    elif highlight_top_k and highlight_top_k >= filtered_seq_len:
        # If highlight_top_k >= number of filtered tokens, we highlight them all
        highlight_indices = np.arange(filtered_seq_len)

    # 7) Annotate tokens with their normalized scores
    annotated_tokens = [
        f"{tk} ({aggregate_scores_normalized[i]:.2f})"
        for i, tk in enumerate(filtered_tokens)
    ]

    # 8) Plot the heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        attn_norm,
        xticklabels=annotated_tokens,
        yticklabels=annotated_tokens,
        cmap=color_map
    )
    plt.title("Enhanced Attention Weights Heatmap (Filtered Special Tokens)")
    plt.xlabel("Tokens (Filtered)")
    plt.ylabel("Tokens (Filtered)")

    # 9) Highlight the top-k tokens by drawing lines
    if highlight_indices is not None:
        for idx in highlight_indices:
            plt.axvline(x=idx + 0.5, color="red", linestyle="--", linewidth=1)
            plt.axhline(y=idx + 0.5, color="red", linestyle="--", linewidth=1)

    # 10) Save or show
    if output_path:
        plt.savefig(output_path, format=file_format, bbox_inches="tight", dpi=200)
        print(f"[info] Attention weights heatmap saved to: {output_path}")
        plt.close()
    else:
        plt.show()

####################################################################################################

def visualize_attention_regions(
    original_sequence,
    attention_weights,
    tokenized_sequence,
    annotation=None,  # Optional annotation for the sequence
    output_path=None,
    top_k=10,  # Number of top regions to highlight
    color_map="viridis",  # Color scheme for visualization
    dynamic_band_height=True,  # Adjust band heights based on attention scores
    add_legend=True,  # Include a legend for top tokens
):
    """
    Visualize important regions in a sequence based on attention scores.

    Parameters:
    - original_sequence: The full DNA sequence (string).
    - attention_weights: Attention scores (1D or 2D array).
    - tokenized_sequence: Tokenized representation of the sequence.
    - annotation (dict): Optional annotation for the sequence.
    - output_path (str): File path to save the visualization.
    - top_k (int): Number of top regions to highlight.
    - color_map (str): Colormap for the visualization.
    - dynamic_band_height (bool): Adjust band heights based on attention scores.
    - add_legend (bool): Include a legend summarizing the top tokens.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np

    # Handle 2D attention weights
    if isinstance(attention_weights, np.ndarray) and attention_weights.ndim == 2:
        attention_weights = attention_weights.sum(axis=0)  # Sum over query rows

    # Normalize attention weights
    attention_weights = np.array(attention_weights)
    attention_weights_normalized = (attention_weights - np.min(attention_weights)) / (
        np.max(attention_weights) - np.min(attention_weights) + 1e-8
    )

    # Identify top-k regions by attention scores
    top_indices = np.argsort(attention_weights_normalized)[-top_k:][::-1]  # Sort descending
    top_scores = attention_weights_normalized[top_indices]
    top_tokens = [tokenized_sequence[int(idx)] for idx in top_indices]

    # Create figure
    plt.figure(figsize=(15, 6))

    # Highlight important regions
    for idx, (start, token, score) in enumerate(zip(top_indices, top_tokens, top_scores)):
        height = score if dynamic_band_height else 1.0  # Adjust band height
        color = plt.get_cmap(color_map)(score)  # Map score to color
        plt.axvspan(start, start + len(token), 0, height, color=color, alpha=0.7)

        # Annotate token and score
        plt.text(
            start + len(token) / 2,
            height + 0.05,
            f"{token} ({score:.2f})",
            fontsize=9,
            ha="center",
            rotation=45,
        )

    # Plot sequence structure
    plt.plot(range(len(original_sequence)), [0] * len(original_sequence), "k-", alpha=0.5)

    # Adjust y-axis dynamically
    max_score = max(top_scores)
    y_max = max_score + (0.1 if dynamic_band_height else 0.05)
    plt.ylim(0, y_max)

    # Add legend for top tokens
    if add_legend:
        token_summary = {}
        for token, score in zip(top_tokens, top_scores):
            if token not in token_summary:
                token_summary[token] = {"max_score": score, "total_score": score, "count": 1}
            else:
                token_summary[token]["max_score"] = max(token_summary[token]["max_score"], score)
                token_summary[token]["total_score"] += score
                token_summary[token]["count"] += 1

        # Compute average scores
        legend_entries = []
        for token, stats in token_summary.items():
            avg_score = stats["total_score"] / stats["count"]
            legend_entries.append(
                mpatches.Patch(
                    color=plt.get_cmap(color_map)(stats["max_score"]),
                    label=f"{token} (Max: {stats['max_score']:.2f}, Avg: {avg_score:.2f})",
                )
            )

        plt.legend(handles=legend_entries, loc="upper right", title="Top Tokens")

    # Formatting
    plt.title("Important Regions in the Sequence Based on Attention Scores")
    plt.xlabel("Position in Sequence")
    plt.ylabel("Attention Weight")
    plt.tight_layout()

    # Save or display the plot
    if output_path:
        plt.savefig(output_path, format="pdf", bbox_inches="tight", dpi=300)
        print(f"[info] Visualization saved to: {output_path}")
        plt.close()
    else:
        plt.show()


# Introducing token-wise max scores and average scores
def visualize_attention_regions_v1(
    original_sequence,
    attention_weights,
    tokenized_sequence,
    max_scores=None,
    avg_scores=None,
    output_path=None,
    top_k=10,
    color_map="viridis",
    dynamic_band_height=True
):
    """
    Enhanced visualization with adaptive y-axis and improved legends.

    Updates: 
    - Y-Axis Range:
       Dynamically adjust the y-axis upper limit to the maximum attention score with a buffer for annotations.
    - Legend Improvements:
       Integrated a comprehensive legend with token statistics (max and avg scores) for easy interpretation.
    
    - Compatibility with Updated Outputs:
    - Ensured compatibility with outputs from compute_attention_rollout and get_attention_weights.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    if attention_weights.ndim == 2:
        attention_weights = attention_weights.sum(axis=0)  # Aggregate to 1D

    attention_weights_normalized = (attention_weights - np.min(attention_weights)) / (
        np.max(attention_weights) - np.min(attention_weights) + 1e-8
    )

    # Use max_scores and avg_scores if available
    token_importance = [
        {
            "token": tokenized_sequence[idx],
            "start": idx,
            "max": max_scores[idx] if max_scores is not None else attention_weights_normalized[idx],
            "avg": avg_scores[idx] if avg_scores is not None else attention_weights_normalized[idx],
        }
        for idx in range(len(tokenized_sequence))
    ]

    # Identify top-k tokens by max scores
    token_importance = sorted(token_importance, key=lambda x: x["max"], reverse=True)
    top_tokens = token_importance[:top_k]

    # Determine y-axis limits dynamically
    y_max = max(entry["max"] for entry in token_importance) * 1.1

    # Create the figure
    plt.figure(figsize=(15, 6))
    for entry in token_importance:
        token, start, max_score, avg_score = entry["token"], entry["start"], entry["max"], entry["avg"]
        height = max_score if dynamic_band_height else 1.0
        color = plt.get_cmap(color_map)(max_score)
        plt.axvspan(start, start + len(token), 0, height / y_max, color=color, alpha=0.7)

        if entry in top_tokens:
            plt.text(
                start + len(token) / 2,
                height / y_max + 0.05,
                f"{token} (Max: {max_score:.2f}, Avg: {avg_score:.2f})",
                fontsize=9,
                ha="center",
                rotation=45,
            )

    # Sequence structure
    plt.plot(range(len(original_sequence)), [0] * len(original_sequence), "k-", alpha=0.5)

    # Formatting
    plt.title("Important Regions in the Sequence Based on Attention Scores")
    plt.xlabel("Position in Sequence")
    plt.ylabel("Attention Weight")
    plt.ylim(0, y_max)
    plt.tight_layout()

    # Add legend
    if top_tokens:
        legend_patches = [
            f"{entry['token']} (Max: {entry['max']:.2f}, Avg: {entry['avg']:.2f})" for entry in top_tokens
        ]
        plt.legend(
            handles=[],
            title="Top Tokens",
            loc="upper right",
            labels=legend_patches,
            fontsize=8,
            frameon=False,
        )

    if output_path:
        plt.savefig(output_path, format="pdf", bbox_inches="tight", dpi=300)
        print(f"[info] Visualization saved to: {output_path}")
        plt.close()
    else:
        plt.show()


def visualize_attention_regions_static(
    original_sequence,
    attention_weights,
    tokenized_sequence,
    annotation=None,  # dict with e.g. {'transcript_id':..., 'strand':..., 'splice_type':..., 'pred_type':..., 'window_start':..., 'window_end':..., 'position':...}
    output_path=None,
    top_k=10,
    color_map="viridis",
    dynamic_band_height=True,
    add_legend=True,
    label_rotation=45,
    hide_overlapping_labels=True,
    diagnostic_check=False,
    # new parameters
    filter_special_tokens=False,
    special_tokens=("[CLS]","[SEP]","[UNK]","[PAD]"),
    extra_margin=3  # spacing at 3' end
):
    """
    Visualize important regions in a sequence (5'→3') based on attention scores.

    Key improvements:
      - The “Splice Site” label no longer has a confusing arrow.
      - 5' and 3' ends are labeled with gene-based coordinates if provided in 'annotation' along with 'strand'.
      - Legend is placed outside (right side) so tokens near the 3' end don't get covered.

    Parameters
    ----------
    original_sequence : str
        Contextual DNA sequence, oriented 5'→3' (length = window_end - window_start).
    attention_weights : np.ndarray
        1D or 2D array (if 2D, we sum across one axis). Must match len(tokenized_sequence).
    tokenized_sequence : list of str
        Variable-length tokens. If you want to skip special tokens, set filter_special_tokens=True.
    annotation : dict, optional
        e.g. {
          'transcript_id':..., 'strand':..., 'splice_type':..., 'pred_type':...,
          'window_start':..., 'window_end':..., 'position':...
        }
        'position' is the predicted splice site in gene-based coords. 'strand' can be '+' or '-'.
        'window_start' and 'window_end' define how this context was extracted from the gene.
    output_path : str or None
        If set, save the figure. Otherwise plt.show().
    top_k : int
        Number of top tokens to highlight.
    color_map : str
        A Matplotlib colormap (e.g. "viridis","plasma","coolwarm").
    dynamic_band_height : bool
        If True, band height = attention score. Otherwise band height=1 for all tokens.
    add_legend : bool
        If True, build a patch legend for top tokens by average score.
    label_rotation : int
        Rotation angle (e.g. 45 or 90) for the token labels.
    hide_overlapping_labels : bool
        If True, skip placing labels that overlap each other.
    diagnostic_check : bool
        Print extra info about left vs. right half attention distribution.
    filter_special_tokens : bool
        If True, skip tokens like [CLS],[SEP],[UNK],[PAD].
    special_tokens : tuple
        The set of special tokens to remove if filter_special_tokens=True.
    extra_margin : int or float
        Additional margin at the right (3' end) so tokens near the end aren't cut off.

    Returns
    -------
    None
    """
    # import numpy as np
    # import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    # 1) Handle 2D => 1D
    if attention_weights.ndim == 2:
        attention_weights = attention_weights.sum(axis=0)
    if len(attention_weights) != len(tokenized_sequence):
        raise ValueError(
            f"Mismatched lengths: attention_weights({len(attention_weights)}) "
            f"vs. tokenized_sequence({len(tokenized_sequence)})"
        )

    # 2) Optionally remove special tokens
    if filter_special_tokens:
        special_set = set(special_tokens)
        filtered_tokens = []
        filtered_scores = []
        for tk, sc in zip(tokenized_sequence, attention_weights):
            if tk not in special_set:
                filtered_tokens.append(tk)
                filtered_scores.append(sc)
        tokenized_sequence = filtered_tokens
        attention_weights = np.array(filtered_scores, dtype=float)
        if not tokenized_sequence:
            print("[warning] All tokens removed by filtering. Exiting.")
            return

    # 3) Normalize attention
    smin, smax = attention_weights.min(), attention_weights.max()
    denom = (smax - smin) + 1e-8
    norm_scores = (attention_weights - smin) / denom

    # 4) Identify top-k
    idx_sorted = np.argsort(norm_scores)  # ascending
    top_indices = idx_sorted[-top_k:][::-1]  # last k desc
    top_scores = norm_scores[top_indices]
    top_tokens = [tokenized_sequence[i] for i in top_indices]

    seq_len = len(original_sequence)
    midpoint = seq_len // 2

    # 5) Diagnostic if needed
    if diagnostic_check:
        left_scores = norm_scores[:midpoint]
        right_scores= norm_scores[midpoint:]
        avg_left = left_scores.mean() if len(left_scores)>0 else 0
        avg_right= right_scores.mean() if len(right_scores)>0 else 0
        left_count= sum(i<midpoint for i in top_indices)
        right_count= top_k - left_count
        print(f"[diagnostic] avg attn left={avg_left:.3f}, right={avg_right:.3f}")
        print(f"[diagnostic] top_k={top_k} => left={left_count}, right={right_count}")

    # 6) Prepare figure
    fig, ax = plt.subplots(figsize=(15,6))
    title_str= "Important Regions (5'→3')"
    # extract annotation
    gene_id= "N/A"
    tx_id=   "N/A"
    splice_type= "N/A"
    pred_type=   "N/A"
    strand=      "+"
    window_start=0
    window_end=  seq_len
    position=    seq_len//2
    if annotation:
        gene_id= annotation.get("gene_id","N/A")
        tx_id=   annotation.get("transcript_id","N/A")
        splice_type= annotation.get("splice_type","N/A")
        pred_type= annotation.get("pred_type","N/A")
        strand= annotation.get("strand","+")

        # if present
        window_start= annotation.get("window_start",0)
        window_end=   annotation.get("window_end",seq_len)
        position=     annotation.get("position",(window_start+window_end)//2)

    title_str += f"\nGene: {gene_id}, Tx: {tx_id}, Splice: {splice_type}, Pred: {pred_type}"
    ax.set_title(title_str, fontsize=12)

    # 7) Plot a baseline line at y=0
    ax.plot(range(seq_len), [0]*seq_len, "k-", alpha=0.5)

    # Label 5' / 3' ends with coordinates
    if strand=="+":
        left_coord=  window_start
        right_coord= window_end
    else:
        left_coord=  window_end
        right_coord= window_start
    ax.text(0, 0.05,  f"5' end: {left_coord}",  ha="left", va="bottom", color="blue", fontsize=9)
    ax.text(seq_len, 0.05, f"3' end: {right_coord}", ha="right",va="bottom", color="blue", fontsize=9)

    # Splice site line
    # local_splice_idx in [0.. seq_len]
    if strand=="+":
        local_splice_idx= position - window_start
    else:
        local_splice_idx= (window_end-1) - position

    if 0<= local_splice_idx < seq_len:
        ax.axvline(local_splice_idx, color="red", linestyle="--", alpha=0.6)
        # remove arrow, just label "Splice Site"
        ax.text(local_splice_idx+1, 0.1, "Splice Site", color="red", ha="left", va="bottom", fontsize=9)

    # 8) Build naive prefix sums for alignment
    prefix_sums=[0]
    for tk in tokenized_sequence:
        prefix_sums.append(prefix_sums[-1] + len(tk))

    # collect top tokens => (index,score,token)
    top_data= list(zip(top_indices, top_scores, top_tokens))
    # sort left->right by naive offset
    top_data.sort(key=lambda x: prefix_sums[x[0]])

    # function to find best substring
    def find_closest_substring(token, seq, approx_x):
        best_idx= -1
        best_diff= 1e9
        start=0
        while True:
            idx= seq.find(token, start)
            if idx<0: break
            diff= abs(idx- approx_x)
            if diff< best_diff:
                best_diff= diff
                best_idx= idx
            start= idx+1
        return best_idx

    placed_labels=[]
    cmap= plt.get_cmap(color_map)

    for idx_, sc_, tk_ in top_data:
        approx_start= prefix_sums[idx_]
        real_start= find_closest_substring(tk_, original_sequence, approx_start)
        if real_start<0:
            real_start= approx_start  # fallback
        width= len(tk_)
        color= cmap(sc_)
        # band height
        band_height= sc_ if dynamic_band_height else 1.0

        # highlight region with axvspan
        ax.axvspan(real_start, real_start+width, ymin=0, ymax=band_height, color=color, alpha=0.7)

        # place text label
        label_x= real_start+ width/2
        label_y= band_height+0.02
        label_str= f"{tk_} ({sc_:.2f})"

        if hide_overlapping_labels:
            text_width= width*0.35
            text_height= 0.06
            overlap_found= False
            for (ox,oy,ow,oh) in placed_labels:
                if (abs(label_x-ox)< (text_width+ow)) and (abs(label_y-oy)< (text_height+oh)):
                    overlap_found=True
                    break
            if not overlap_found:
                ax.text(label_x, label_y, label_str, fontsize=9, ha="center", rotation=label_rotation)
                placed_labels.append((label_x,label_y, text_width,text_height))
        else:
            ax.text(label_x, label_y, label_str, fontsize=9, ha="center", rotation=label_rotation)

    # 9) Build legend if needed
    if add_legend and top_indices.size>0:
        token_summary={}
        for idx_ in top_indices:
            tk= tokenized_sequence[idx_]
            sc= norm_scores[idx_]
            token_summary.setdefault(tk,[]).append(sc)
        patches=[]
        for tk, sc_list in token_summary.items():
            avg_score= np.mean(sc_list)
            patch_col= cmap(avg_score)
            label_ = f"{tk} (Avg: {avg_score:.2f})"
            patches.append(mpatches.Patch(color=patch_col, label=label_))

        def parse_avg(lbl):
            # parse "(Avg: 0.XX)"
            part= lbl.split("Avg: ")[1].rstrip(")")
            return float(part)
        patches= sorted(patches, key=lambda p: parse_avg(p.get_label()), reverse=True)

        # place legend outside to the right
        ax.legend(handles=patches, loc="upper left", bbox_to_anchor=(1.0,1.0),
                  borderaxespad=0., title=f"Top {top_k} Tokens (Avg Scores)")

    # 10) final formatting
    max_score= max(top_scores) if top_scores.size>0 else 1.0
    ax.set_ylim(0, max_score+0.2)
    ax.set_xlim(0, seq_len+ extra_margin)
    ax.set_xlabel("Position in Sequence (5' → 3')")
    ax.set_ylabel("Attention Weight")

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, format="pdf", bbox_inches="tight", dpi=300)
        print(f"[info] Visualization saved to: {output_path}")
        plt.close()
    else:
        plt.show()


def visualize_attention_regions_static_v0(
    original_sequence,
    attention_weights,
    tokenized_sequence,
    annotation=None,
    output_path=None,
    top_k=10,
    color_map="viridis",
    dynamic_band_height=True,
    add_legend=True,
    label_rotation=45,
    hide_overlapping_labels=True,
    diagnostic_check=True,
    # 1) new params
    filter_special_tokens=True,
    special_tokens=("[CLS]", "[SEP]", "[UNK]", "[PAD]"),
    extra_margin=3  # margin at 3' end so last token isn't cut off
):
    """
    Visualize important regions in a sequence based on attention scores (static Matplotlib),
    oriented 5'→3'. We highlight top-k tokens by color-coded vertical spans 
    at their actual substring offsets.

    Parameters:
    -----------
    original_sequence : str
        The full DNA sequence, oriented 5'→3'.
    attention_weights : np.ndarray
        1D or 2D array. If 2D, we sum across one axis to get a single score per token.
        The result must match the length of tokenized_sequence.
    tokenized_sequence : list of str
        Tokenized representation (variable length tokens). If there's an overlap or
        special tokens, we handle them accordingly.
    annotation : dict
        Optional: e.g. {'transcript_id':..., 'strand':..., 'splice_type':..., 'pred_type':...}.
    output_path : str or None
        If provided, save a PDF. Otherwise show interactively.
    top_k : int
        Number of top tokens to highlight by color.
    color_map : str
        A Matplotlib colormap (e.g. "viridis", "plasma").
    dynamic_band_height : bool
        If True, the vertical band height is scaled by the token's attention score in [0..1].
        Otherwise use a fixed height=1 for all tokens.
    add_legend : bool
        If True, show a legend summarizing the top tokens and their average attention scores.
    label_rotation : int
        Angle to rotate text labels for tokens (e.g., 45, 90).
    hide_overlapping_labels : bool
        If True, skip labels that overlap in the plot.
    diagnostic_check : bool
        Print extra debugging info about left vs. right half.
    filter_special_tokens : bool
        If True, remove special tokens like [CLS],[SEP],[UNK],[PAD].
    special_tokens : tuple
        The special tokens to remove if filter_special_tokens=True.
    extra_margin : float
        Additional margin at the 3' end to avoid cutting off last tokens.

    Returns:
    --------
    None
    """
    import matplotlib.patches as mpatches

    # 1) Sum 2D => 1D if needed
    if attention_weights.ndim == 2:
        attention_weights = attention_weights.sum(axis=0)
    if len(attention_weights) != len(tokenized_sequence):
        raise ValueError(
            f"Length mismatch: attention_weights({len(attention_weights)}) "
            f"vs. tokenized_sequence({len(tokenized_sequence)})"
        )

    # 2) Optionally filter out special tokens
    special_set = set(special_tokens)
    filtered_tokens = []
    filtered_scores = []
    if filter_special_tokens:
        for tk, sc in zip(tokenized_sequence, attention_weights):
            if tk not in special_set:
                filtered_tokens.append(tk)
                filtered_scores.append(sc)
    else:
        filtered_tokens = tokenized_sequence
        filtered_scores = attention_weights

    if not filtered_tokens:
        print("[warning] No tokens left after filtering special tokens. Exiting.")
        return

    tokenized_sequence = filtered_tokens
    attention_weights = np.array(filtered_scores, dtype=float)

    # 3) Normalize scores => [0..1]
    smin, smax = attention_weights.min(), attention_weights.max()
    denom = (smax - smin) + 1e-8
    norm_scores = (attention_weights - smin) / denom

    # 4) Identify top-k tokens
    idx_sorted = np.argsort(norm_scores)  # ascending
    top_indices = idx_sorted[-top_k:][::-1]  # last k in descending order
    top_scores = norm_scores[top_indices]
    top_tokens = [tokenized_sequence[i] for i in top_indices]

    # 5) Diagnostic check
    seq_len = len(original_sequence)
    midpoint = seq_len // 2
    if diagnostic_check:
        left_scores = norm_scores[:midpoint]
        right_scores= norm_scores[midpoint:]
        avg_left   = left_scores.mean() if len(left_scores)>0 else 0
        avg_right  = right_scores.mean() if len(right_scores)>0 else 0
        left_count = sum(idx<midpoint for idx in top_indices)
        right_count= top_k - left_count
        print(f"[diagnostic] average attn left={avg_left:.3f}, right={avg_right:.3f}")
        print(f"[diagnostic] top_k={top_k} => left={left_count}, right={right_count} tokens.")

    # 6) set up the figure
    plt.figure(figsize=(15, 6))
    title_str = "Important Regions (5'→3')"
    if annotation:
        tr_id = annotation.get("transcript_id","N/A")
        st    = annotation.get("strand","N/A")
        sp    = annotation.get("splice_type","N/A")
        pt    = annotation.get("pred_type","N/A")
        title_str += f"\nTranscript: {tr_id}, Strand: {st}, Splice: {sp}, Pred: {pt}"
    plt.title(title_str)

    # baseline line at y=0
    plt.plot(range(seq_len), [0]*seq_len, "k-", alpha=0.5)

    # Label 5' / 3' ends
    plt.text(0, 0.05,  "5' end", ha="left", va="bottom", color="blue", fontsize=9)
    plt.text(seq_len, 0.05, "3' end", ha="right", va="bottom", color="blue", fontsize=9)

    # Splice site line
    plt.axvline(midpoint, color="red", linestyle="--", alpha=0.6)
    plt.text(midpoint+1, 0.1, "Splice Site →", color="red", ha="left", va="bottom", fontsize=9)

    # 7) We'll do a substring-match approach: build prefix sums as an approximate offset, then refine
    prefix_sums = [0]
    for tk in tokenized_sequence:
        prefix_sums.append(prefix_sums[-1] + len(tk))

    # pick out top tokens
    top_data = list(zip(top_indices, top_scores, top_tokens))
    # sort left->right by prefix_sums
    top_data.sort(key=lambda x: prefix_sums[x[0]])

    # We'll find the best actual substring match near prefix_sums offset
    def find_closest_substring(token, seq, approx_x):
        # gather all occurrences, pick the one closest to approx_x
        best_idx = -1
        best_diff= 1e9
        start=0
        while True:
            idx = seq.find(token, start)
            if idx<0:
                break
            diff= abs(idx-approx_x)
            if diff< best_diff:
                best_diff= diff
                best_idx= idx
            start = idx+1
        return best_idx
    # Gathers all occurrences of token in the original_sequence.
    # Picks whichever occurrence is closest to approx_x.
    # If none found (-1), we fallback to the naive offset.
    # This ensures if you have repeated substring or if the naive offset is off by a few nucleotides, we adjust.

    placed_labels= []
    cmap= plt.get_cmap(color_map)

    for (idx_, sc_, tk_) in top_data:
        approx_start = prefix_sums[idx_]  # naive offset
        match_start  = find_closest_substring(tk_, original_sequence, approx_start)
        if match_start<0:
            # fallback => just use approx_start
            real_start= approx_start
        else:
            real_start= match_start

        width= len(tk_)
        color= cmap(sc_)
        height= sc_ if dynamic_band_height else 1.0

        # highlight region with axvspan
        plt.axvspan(real_start, real_start+width, ymin=0, ymax=height, color=color, alpha=0.7)

        # place text label
        label_x = real_start+ width/2
        label_y = height+0.02
        label_str = f"{tk_} ({sc_:.2f})"

        if hide_overlapping_labels:
            text_width=width*0.35
            text_height=0.06
            overlap_found= False
            for (ox,oy,ow,oh) in placed_labels:
                if (abs(label_x-ox)<(text_width+ow)) and (abs(label_y-oy)<(text_height+oh)):
                    overlap_found=True
                    break
            if not overlap_found:
                plt.text(label_x, label_y, label_str, fontsize=9, ha="center", rotation=label_rotation)
                placed_labels.append((label_x,label_y,text_width,text_height))
        else:
            plt.text(label_x, label_y, label_str, fontsize=9, ha="center", rotation=label_rotation)

    # 8) Legend
    if add_legend and len(top_data)>0:
        token_summary= {}
        for idx_ in top_indices:
            tk= tokenized_sequence[idx_]
            sc= norm_scores[idx_]
            token_summary.setdefault(tk, []).append(sc)
        patches=[]
        for tk, sc_list in token_summary.items():
            avg_sc= np.mean(sc_list)
            patch_color= cmap(avg_sc)
            label_= f"{tk} (Avg: {avg_sc:.2f})"
            patches.append(mpatches.Patch(color=patch_color, label=label_))

        def parse_avg(lbl):
            # parse (Avg: 0.XX)
            part= lbl.split("Avg: ")[1].rstrip(")")
            return float(part)
        patches= sorted(patches, key=lambda p: parse_avg(p.get_label()), reverse=True)

        plt.legend(handles=patches, loc="upper right", title=f"Top {top_k} Tokens (Avg Scores)")

    # 9) Final formatting
    max_score= max(top_scores) if len(top_scores)>0 else 1.0
    plt.ylim(0, max_score+0.2)
    plt.xlim(0, seq_len+extra_margin)
    plt.xlabel("Position in Sequence (5' → 3')")
    plt.ylabel("Attention Weight")

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, format="pdf", bbox_inches="tight", dpi=300)
        print(f"[info] Visualization saved to: {output_path}")
        plt.close()
    else:
        plt.show()


def visualize_attention_regions_interactive(
    original_sequence,
    attention_weights,
    tokenized_sequence,
    annotation=None,
    top_k=10,
    color_map="Viridis",
    dynamic_band_height=True,
    show_midpoint=True
):
    """
    Create an interactive Plotly figure showing attention-based important regions.
    Hovering over tokens reveals details (token text + score).

    Parameters:
    - original_sequence (str)
    - attention_weights (1D or 2D np.ndarray)
    - tokenized_sequence (list of str)
    - annotation (dict) optional metadata
    - top_k (int): number of tokens to highlight
    - color_map (str): a Plotly colormap name (e.g., "Viridis", "Cividis", etc.)
    - dynamic_band_height (bool): use token's score for rectangle height
    - show_midpoint (bool): draw a dashed vertical line at sequence midpoint
    """
    import plotly.graph_objects as go

    if attention_weights.ndim == 2:
        attention_weights = attention_weights.sum(axis=0)

    attention_weights = np.array(attention_weights)
    min_val, max_val = attention_weights.min(), attention_weights.max()
    denom = (max_val - min_val) + 1e-8
    attention_weights_normalized = (attention_weights - min_val) / denom

    # Identify top-k
    top_indices = np.argsort(attention_weights_normalized)[-top_k:][::-1]
    top_scores = attention_weights_normalized[top_indices]
    top_tokens = [tokenized_sequence[int(idx)] for idx in top_indices]

    fig = go.Figure()

    # baseline "sequence line"
    seq_len = len(original_sequence)
    fig.add_trace(
        go.Scatter(
            x=list(range(seq_len)),
            y=[0]*seq_len,
            mode="lines",
            line=dict(color="black", width=1),
            name="Sequence baseline",
        )
    )

    # Add rectangles for top tokens
    for idx, (pos, token, score) in enumerate(zip(top_indices, top_tokens, top_scores)):
        height = score if dynamic_band_height else 1.0
        # color scale: we can map the normalized score to a color
        # Plotly can do a uniform fillcolor, or we can map the color using plotly.colors
        color = f"rgba(68, 1, 84, {0.5 + 0.5*score})"  # naive approach: alpha ~ score
        # Alternatively, use a built-in colorscale library, but let's keep it simple

        # We'll add a rectangle shape from x=pos to x=pos+len(token)
        fig.add_shape(
            type="rect",
            x0=pos,
            x1=pos+len(token),
            y0=0,
            y1=height,
            fillcolor=color,
            opacity=0.7,
            line=dict(width=0),
            name=token,
        )

        # We can add an invisible scatter point for hover text
        # so that when you hover over it, you see token info
        fig.add_trace(go.Scatter(
            x=[(pos + pos+len(token))/2],
            y=[height/2],
            mode="markers",
            marker=dict(size=5, color="rgba(0,0,0,0)"),
            showlegend=False,
            hovertemplate=f"Token: {token}<br>Score: {score:.2f}",
        ))

    # Midpoint
    if show_midpoint:
        mid_x = seq_len // 2
        # We add a shape line or annotation
        fig.add_shape(
            type="line",
            x0=mid_x,
            x1=mid_x,
            y0=0,
            y1=1.2 * (max(top_scores) if len(top_scores)>0 else 1.0),
            line=dict(color="red", dash="dash"),
        )
        fig.add_annotation(
            x=mid_x,
            y=(max(top_scores) if len(top_scores)>0 else 1.0),
            text="Splice Site",
            showarrow=True,
            arrowhead=2,
            yshift=10,
        )

    # Title and layout
    title_str = "Interactive Attention-Based Regions"
    if annotation:
        splice_type = annotation.get("splice_type", "N/A")
        pred_type = annotation.get("pred_type", "N/A")
        title_str += f" - {splice_type} - {pred_type}"

    fig.update_layout(
        title=title_str,
        xaxis=dict(title="Position in Sequence (base-level)"),
        yaxis=dict(title="Attention Weight (normalized)", range=[0, max(top_scores)+0.2 if len(top_scores)>0 else 1]),
        shapes_layer="below",  # put the rectangles behind the baseline line
    )

    fig.show()
    return fig

########################################################################################
# Alignment-style plots with top-K tokens above the raw sequence


def find_best_substring_match(token, context, approx_x):
    best_idx = -1
    best_diff= float('inf')
    search_start = 0
    while True:
        idx = context.find(token, search_start)
        if idx < 0:
            break
        diff = abs(idx - approx_x)
        if diff < best_diff:
            best_idx = idx
            best_diff= diff
        search_start = idx + 1  # continue searching for next occurrence
    return best_idx


def plot_alignment_with_scores_v0(
    original_sequence,
    attention_weights,
    tokenized_sequence,
    annotation=None,
    output_path=None,
    top_k=10,
    color_map="plasma",  # e.g. a "temperature" style colormap
    dynamic_band_height=True,
    add_legend=True,
    hide_overlapping_labels=True,
    token_label_rotation=0,
    figsize=(20, 6),

    filter_special_tokens=True,
    special_tokens=("[CLS]", "[SEP]", "[PAD]", "[UNK]"),

    draw_alignment_arrows=True,
    arrow_line_color="lightgrey",
    arrow_end_y=0.3,        # Where the arrow meets near the base row
    rect_baseline_height=0.3,
    rect_scale_factor=0.7,

    # 1) Add local coordinate label
    show_token_positions=False,   # If True, we show e.g. "start=12" on top of the rectangle
    position_label_fontsize=7,    # smaller font for the coordinate label
    extra_x_margin=3              # 3-nucleotide margin at 3' end to avoid cutoff
):
    """
    Plot an alignment-style visualization of the top-k tokens above a local 0..L axis,
    where L = len(original_sequence). The substring is oriented 5'→3'.
    We compute the local splice index from annotation (window_start, window_end, position, strand).

    Additional features:
    - filter_special_tokens: if True, skip e.g. [CLS],[SEP],[UNK],[PAD].
    - draw_alignment_arrows: dashed lines from each token to the best substring match in the raw sequence.
    - color_map: "plasma" (temperature-based) or "magma", "coolwarm", etc.
    - show_token_positions: label the rectangle with local start coordinate (and/or end).

    Also we add some extra margin at the right side (3' end) = extra_x_margin nucleotides,
    so the last token doesn't get chopped off.

    Parameters
    ----------
    original_sequence : str
        The context substring, oriented 5'→3'.
    attention_weights : np.ndarray
        1D array of attention scores, one per token. (If 2D, sum over rows first.)
    tokenized_sequence : list of str
        The tokens (possibly variable-length subwords or k-mers).
    annotation : dict
        e.g. {'window_start','window_end','position','strand','gene_id','transcript_id','splice_type','pred_type'}
    output_path : str or None
        If set, save figure. Otherwise plt.show().
    top_k : int
        Number of top tokens to highlight by color.
    color_map : str
        A Matplotlib colormap name, e.g. "plasma","magma","coolwarm".
    dynamic_band_height : bool
        If True, rectangle height depends on attention score => baseline + scaled factor.
    add_legend : bool
        If True, build a patch legend summarizing top tokens by average scores.
    hide_overlapping_labels : bool
        If True, skip token text labels that overlap.
    token_label_rotation : float
        Rotation angle in degrees for token text.
    figsize : (width, height)
        Figure size in inches.
    filter_special_tokens : bool
        If True, remove tokens in special_tokens from the plotting.
    special_tokens : tuple
        Set of special tokens like ("[CLS]","[SEP]","[PAD]","[UNK]").
    draw_alignment_arrows : bool
        If True, draw dashed arrows from each top token to the best substring match in original_sequence.
    arrow_line_color : str
        Color of arrow lines. "lightgrey" or "black" etc.
    arrow_end_y : float
        The y-level near the base row for arrow endpoints.
    rect_baseline_height : float
        Minimum rectangle height if dynamic_band_height is True.
    rect_scale_factor : float
        Additional scaling if dynamic_band_height is True.
    show_token_positions : bool
        If True, label each token's rectangle with "start=XX" or "pos=XX" to show local coords.
    position_label_fontsize : int
        Font size for the position label if show_token_positions=True.
    extra_x_margin : float
        Additional margin at the right (3' end) so last token doesn't get cut off.

    Returns
    -------
    None

    Updates
    -------
    - Add local coordinate label (Parameter: show_token_positions)
    - The rectangle width = token_len. That encloses the spelled-out token horizontally.
    - Ensures we have a small buffer so tokens near the 3′ end are not clipped.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np

    # 1) Handle 2D => 1D if needed
    if attention_weights.ndim == 2:
        attention_weights = attention_weights.sum(axis=0)
    elif attention_weights.ndim != 1:
        raise ValueError("attention_weights must be 1D or 2D.")

    if len(attention_weights) != len(tokenized_sequence):
        raise ValueError(
            f"Size mismatch: len(attention_weights)={len(attention_weights)} "
            f"vs len(tokenized_sequence)={len(tokenized_sequence)}"
        )

    # 2) Optionally filter out special tokens
    filtered_tokens = []
    filtered_scores = []
    special_set = set(special_tokens)
    if filter_special_tokens:
        for tk, sc in zip(tokenized_sequence, attention_weights):
            if tk not in special_set:
                filtered_tokens.append(tk)
                filtered_scores.append(sc)
    else:
        filtered_tokens = tokenized_sequence
        filtered_scores = attention_weights

    if len(filtered_tokens) == 0:
        print("[warning] After filtering special tokens, no tokens left to plot.")
        return

    tokenized_sequence = filtered_tokens
    attention_weights = np.array(filtered_scores, dtype=float)

    # 3) Extract annotation
    if annotation is None:
        annotation = {}
    seq_len = len(original_sequence)
    window_start = annotation.get("window_start", 0)
    window_end   = annotation.get("window_end", seq_len)
    position     = annotation.get("position", (window_start+window_end)//2)
    strand       = annotation.get("strand", "+")
    gene_id      = annotation.get("gene_id", "N/A")
    tx_id        = annotation.get("transcript_id", "N/A")
    splice_type  = annotation.get("splice_type", "N/A")
    pred_type    = annotation.get("pred_type", "N/A")

    # local splice idx
    if strand == '+':
        local_splice_idx = position - window_start
    else:
        local_splice_idx = (window_end -1) - position

    # 4) Normalize attention scores
    scores = attention_weights
    smin, smax = scores.min(), scores.max()
    denom = (smax - smin) + 1e-8
    norm_scores = (scores - smin)/denom

    # 5) Identify top-k
    idx_sorted = np.argsort(norm_scores)
    top_indices = idx_sorted[-top_k:][::-1]  # last k desc
    top_scores = norm_scores[top_indices]
    top_tokens = [tokenized_sequence[i] for i in top_indices]

    # 6) Create figure
    fig, ax = plt.subplots(figsize=figsize)
    title_str = (
        f"Alignment Plot (Gene: {gene_id}, Tx: {tx_id}, Strand: {strand})\n"
        f"Splice Type: {splice_type}, Pred Type: {pred_type}"
    )
    ax.set_title(title_str, fontsize=12)

    # 7) draw raw base row
    for i, base in enumerate(original_sequence):
        ax.text(i+0.5, 0, base, ha="center", va="center", fontsize=8)

    # mark 5' and 3'
    if strand == '+':
        left_coord = window_start
        right_coord= window_end
    else:
        left_coord = window_end -1
        right_coord= window_start

    ax.text(0, 0.2, f"5' end: {left_coord}", color="blue", ha="left", va="bottom", fontsize=9)
    ax.text(seq_len, 0.2, f"3' end: {right_coord}", color="blue", ha="right", va="bottom", fontsize=9)

    # dashed line for splice site
    if 0 <= local_splice_idx < seq_len:
        ax.axvline(local_splice_idx, color="red", linestyle="--", alpha=0.6)
        ax.text(local_splice_idx+1, 0.4, "Splice Site", color="red", ha="left", va="bottom", fontsize=9)

    # 8) build prefix sums
    prefix_sums = [0]
    for tk_ in tokenized_sequence:
        prefix_sums.append(prefix_sums[-1] + len(tk_))

    # 9) collect top tokens => (index,score,token), sort by prefix_sums
    top_data = list(zip(top_indices, top_scores, top_tokens))
    top_data.sort(key=lambda x: prefix_sums[x[0]])  # left->right

    # function to find best substring occurrence near prefix sums
    def find_closest_match_in_seq(token, seq, approx_x):
        best_idx = -1
        best_dist= 1e9
        start    = 0
        while True:
            idx = seq.find(token, start)
            if idx < 0:
                break
            dist = abs(idx - approx_x)
            if dist < best_dist:
                best_idx = idx
                best_dist= dist
            start = idx +1
        return best_idx

    y_offset = 1.0
    placed_labels = []
    cmap = plt.get_cmap(color_map)

    for idx_, sc_, tk_ in top_data:
        start_x = prefix_sums[idx_]
        token_len = len(tk_)

        # improved vertical dimension
        if dynamic_band_height:
            rect_height = rect_baseline_height + sc_*rect_scale_factor
        else:
            rect_height = rect_baseline_height + rect_scale_factor

        color = cmap(sc_)

        # draw rectangle
        rect = plt.Rectangle(
            (start_x, y_offset),
            token_len,
            rect_height,
            color=color,
            alpha=0.8,
            lw=0
        )
        ax.add_patch(rect)

        # position for token text in center
        label_x = start_x + token_len/2
        label_y = y_offset + rect_height/2
        label_str = f"{tk_} ({sc_:.2f})"

        # place token text
        if hide_overlapping_labels:
            # bounding box check
            text_width = token_len * 0.45
            text_height= rect_height
            overlap_found = False
            for (ox, oy, ow, oh) in placed_labels:
                if (abs(label_x - ox)<(text_width+ow)) and (abs(label_y-oy)<(text_height+oh)):
                    overlap_found=True
                    break
            if not overlap_found:
                ax.text(
                    label_x, label_y, label_str,
                    fontsize=9, ha="center", va="center",
                    rotation=token_label_rotation
                )
                placed_labels.append((label_x, label_y, text_width, text_height))
        else:
            ax.text(
                label_x, label_y, label_str,
                fontsize=9, ha="center", va="center",
                rotation=token_label_rotation
            )

        # 1) show local coordinate
        if show_token_positions:
            pos_str = f"pos={start_x}"
            ax.text(
                label_x, y_offset+rect_height+0.1,
                pos_str,
                fontsize=position_label_fontsize,
                ha="center",
                va="bottom",
                color="black",
                rotation=0
            )

        # optional arrow
        if draw_alignment_arrows:
            # find best match near start_x
            match_idx = find_closest_match_in_seq(tk_, original_sequence, start_x)
            if match_idx>=0:
                arrow_start = (label_x, label_y)
                arrow_end   = (match_idx + token_len/2, arrow_end_y)
                ax.annotate(
                    "",
                    xy=arrow_end, xycoords="data",
                    xytext=arrow_start, textcoords="data",
                    arrowprops=dict(arrowstyle="->", color=arrow_line_color, linestyle="--", alpha=0.8)
                )

        y_offset += (rect_height + 0.25)

    # 3) extra margin so last token not cut off
    total_seq_length = prefix_sums[-1]
    x_max = max(seq_len, total_seq_length)+ extra_x_margin
    ax.set_xlim(0, x_max)
    ax.set_ylim(-0.5, y_offset + 0.5)
    ax.set_xlabel("Local Index (0 = 5' end, L = 3' end)")
    ax.set_ylabel("Token Rows")

    # legend
    if add_legend and len(top_data)>0:
        # aggregator
        token_summary = {}
        for idx_, sc_, tk_ in top_data:
            token_summary.setdefault(tk_, []).append(sc_)
        patches=[]
        for tk_, sc_list in token_summary.items():
            avg_score = np.mean(sc_list)
            patch_color= cmap(avg_score)
            label_ = f"{tk_} (Avg: {avg_score:.2f})"
            patches.append(mpatches.Patch(color=patch_color, label=label_))

        # sort descending
        def parse_avg(lbl):
            p= lbl.split("Avg: ")[1].rstrip(")")
            return float(p)
        patches= sorted(patches, key=lambda p: parse_avg(p.get_label()), reverse=True)

        ax.legend(
            handles=patches,
            loc="upper left",
            bbox_to_anchor=(1.0,1.0),
            borderaxespad=0.,
            title=f"Top {top_k} Tokens"
        )

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"[info] Saved alignment plot to: {output_path}")
        plt.close()
    else:
        plt.show()


def plot_alignment_with_scores(
    original_sequence,
    attention_weights,
    tokenized_sequence,
    annotation=None,
    output_path=None,
    top_k=10,
    color_map="plasma",  # e.g. a "temperature" style colormap
    dynamic_band_height=True,
    add_legend=True,
    hide_overlapping_labels=True,
    token_label_rotation=0,
    figsize=(20, 6),

    # Special tokens & filtering
    filter_special_tokens=True,
    special_tokens=("[CLS]", "[SEP]", "[PAD]", "[UNK]"),

    # Arrows and rectangle logic
    draw_alignment_arrows=True,
    arrow_line_color="lightgrey",
    arrow_end_y=0.3,        # Where the arrow meets near the base row
    rect_baseline_height=0.3,
    rect_scale_factor=0.7,

    # Local coordinate labeling
    show_token_positions=False,
    position_label_fontsize=7,
    extra_x_margin=3,  # 3-nucleotide margin at 3' end to avoid cutoff

    # Score display logic
    show_token_scores=False,
    token_score_fontsize=8
):
    """
    Plot an alignment-style visualization of the top-k tokens above a local 0..L axis,
    where L = len(original_sequence). The substring is oriented 5'→3'.

    Score Placement Rules:
    1) By default (show_token_scores=False), we place the numeric score to the
       **right** of the rectangle (side).
    2) If show_token_scores=True, we place the numeric score **on top** of the rectangle.
    3) If BOTH show_token_scores=True and show_token_positions=True, we combine
       them above the rectangle in a small multi-line style to avoid clutter.

    Other Features:
      - 2D→1D sum if needed (for attention/IG).
      - Optionally filter out special tokens (e.g., [CLS],[SEP]).
      - Temperature-based colormap. Rectangle width = token length.
      - Dashed arrows from each token's rectangle to the best substring match.
      - 5' end, 3' end labeled with gene-based coords (annotation).
      - Additional margin at 3' end so tokens near boundary aren't clipped.

    Updates: 
    - The rectangle always covers exactly [real_start .. real_start + token_len].
    - The token name stays inside the rectangle.
    - Scores and/or positions go either above or to the right—never appended to the token string where they could horizontally overflow.

    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np

    # 1) Handle 2D => 1D if needed
    if attention_weights.ndim == 2:
        attention_weights = attention_weights.sum(axis=0)
    elif attention_weights.ndim != 1:
        raise ValueError("attention_weights must be 1D or 2D.")

    if len(attention_weights) != len(tokenized_sequence):
        raise ValueError(
            f"Size mismatch: len(attention_weights)={len(attention_weights)} "
            f"vs len(tokenized_sequence)={len(tokenized_sequence)}"
        )

    # 2) Optionally filter out special tokens
    special_set = set(special_tokens)
    if filter_special_tokens:
        filtered_tokens = []
        filtered_scores = []
        for tk, sc in zip(tokenized_sequence, attention_weights):
            if tk not in special_set:
                filtered_tokens.append(tk)
                filtered_scores.append(sc)
    else:
        filtered_tokens  = tokenized_sequence
        filtered_scores  = attention_weights

    if not filtered_tokens:
        print("[warning] After filtering special tokens, no tokens left to plot.")
        return

    tokenized_sequence = filtered_tokens
    attention_weights  = np.array(filtered_scores, dtype=float)

    # 3) Extract annotation info
    if annotation is None:
        annotation = {}
    seq_len      = len(original_sequence)
    window_start = annotation.get("window_start", 0)
    window_end   = annotation.get("window_end", seq_len)
    position     = annotation.get("position", (window_start + window_end)//2)
    strand       = annotation.get("strand", "+")
    gene_id      = annotation.get("gene_id", "N/A")
    tx_id        = annotation.get("transcript_id", "N/A")
    splice_type  = annotation.get("splice_type", "N/A")
    pred_type    = annotation.get("pred_type", "N/A")
    error_type   = annotation.get("error_type", "N/A")
    reference    = annotation.get("reference", "Null Sequence")

    # local splice idx
    if strand == '+':
        local_splice_idx = position - window_start
    else:
        local_splice_idx = (window_end - 1) - position

    # 4) Normalize the attention/IG scores
    scores = attention_weights
    smin, smax = scores.min(), scores.max()
    denom = (smax - smin) + 1e-8
    norm_scores = (scores - smin) / denom

    # 5) Identify top-k
    idx_sorted   = np.argsort(norm_scores)       # ascending
    top_indices  = idx_sorted[-top_k:][::-1]     # last k descending
    top_scores   = norm_scores[top_indices]
    top_tokens   = [tokenized_sequence[i] for i in top_indices]

    # 6) Plot setup
    fig, ax = plt.subplots(figsize=figsize)
    title_str = (
        f"Alignment Plot (Gene: {gene_id}, Tx: {tx_id}, Strand: {strand})\n"
        f"Splice Type: {splice_type}, Pred Type: {pred_type}"
    )
    if reference and reference.lower() not in ["null", "n/a"]:
        title_str += f", Ref: {reference}"

    ax.set_title(title_str, fontsize=12)

    # 7) Draw the raw DNA sequence as text at y=0
    for i, base in enumerate(original_sequence):
        ax.text(i + 0.5, 0, base, ha="center", va="center", fontsize=8)

    # Label 5' and 3' ends with gene-based coords
    if strand == '+':
        left_coord  = window_start
        right_coord = window_end
    else:
        left_coord  = window_end - 1
        right_coord = window_start
    ax.text(0,      0.2, f"5' end: {left_coord}",  ha="left",  va="bottom", color="blue", fontsize=9)
    ax.text(seq_len,0.2, f"3' end: {right_coord}", ha="right", va="bottom", color="blue", fontsize=9)

    # Mark predicted splice site
    if 0 <= local_splice_idx < seq_len:
        ax.axvline(local_splice_idx, color="red", linestyle="--", alpha=0.6)
        ax.text(local_splice_idx+1, 0.4, "Splice Site", color="red", ha="left", va="bottom", fontsize=9)

    # 8) Build naive prefix sums
    prefix_sums = [0]
    for tk in tokenized_sequence:
        prefix_sums.append(prefix_sums[-1] + len(tk))

    # build top_data => (index,score,token)
    top_data = list(zip(top_indices, top_scores, top_tokens))
    # Sort left->right by naive offset
    top_data.sort(key=lambda x: prefix_sums[x[0]])

    # function to refine offset by substring match
    def find_closest_match_in_seq(token, seq, approx_x):
        best_idx = -1
        best_diff= 1e9
        start=0
        while True:
            idx = seq.find(token, start)
            if idx < 0:
                break
            dist = abs(idx - approx_x)
            if dist < best_diff:
                best_diff= dist
                best_idx= idx
            start= idx+1
        return best_idx

    y_offset= 1.0
    placed_labels = []
    cmap = plt.get_cmap(color_map)

    # 9) Plot top tokens
    for idx_, sc_, tk_ in top_data:
        approx_start = prefix_sums[idx_]
        real_start   = find_closest_match_in_seq(tk_, original_sequence, approx_start)
        if real_start < 0:
            real_start = approx_start  # fallback if not found

        token_len  = len(tk_)
        # rectangle height
        if dynamic_band_height:
            rect_height = rect_baseline_height + sc_ * rect_scale_factor
        else:
            rect_height = rect_baseline_height + rect_scale_factor

        color = cmap(sc_)

        # Add rectangle for the token
        rect = plt.Rectangle(
            (real_start, y_offset),
            token_len,
            rect_height,
            color=color,
            alpha=0.8,
            lw=0
        )
        ax.add_patch(rect)

        # Place the token text in the center of the bar
        label_x = real_start + token_len / 2
        label_y = y_offset + rect_height / 2

        if hide_overlapping_labels:
            text_width  = token_len * 0.45
            text_height = rect_height
            overlap_found = False
            for (ox, oy, ow, oh) in placed_labels:
                if (abs(label_x - ox) < (text_width + ow)) and (abs(label_y - oy) < (text_height + oh)):
                    overlap_found = True
                    break
            if not overlap_found:
                ax.text(
                    label_x, label_y, tk_,
                    fontsize=9, ha="center", va="center",
                    rotation=token_label_rotation
                )
                placed_labels.append((label_x, label_y, text_width, text_height))
        else:
            ax.text(
                label_x, label_y, tk_,
                fontsize=9, ha="center", va="center",
                rotation=token_label_rotation
            )

        # Next: position and/or score
        # Decide logic based on show_token_scores + show_token_positions
        # 1) If show_token_scores=False => put the numeric score to the "right" of the rectangle
        # 2) If show_token_scores=True => put it "on top" of the rectangle
        # 3) If both show_token_scores and show_token_positions => Place both on top but side by side

        # Score string
        score_str = f"{sc_:.2f}"
        # Position string
        pos_str   = f"pos={real_start}"

        # We'll define short helper to place text. 
        # We'll keep fonts small enough if multi-line
        base_font_size = token_score_fontsize

        if show_token_scores and show_token_positions:

            # Place both on top but side by side
            mid_y = y_offset + rect_height + 0.05

            # Shift score left a bit, position right a bit
            score_x = label_x - 0.3
            pos_x   = label_x + 0.3

            ax.text(
                score_x, mid_y,
                score_str,
                fontsize=token_score_fontsize,
                ha="right", va="bottom", color="black"
            )
            ax.text(
                pos_x, mid_y,
                pos_str,
                fontsize=position_label_fontsize,
                ha="left", va="bottom", color="black"
            )

        elif show_token_scores and not show_token_positions:
            # put score on top
            ax.text(
                label_x,
                y_offset + rect_height + 0.05,
                score_str,
                fontsize=base_font_size,
                ha="center",
                va="bottom",
                color="black"
            )

        elif not show_token_scores and show_token_positions:
            # we put position on top
            ax.text(
                label_x,
                y_offset + rect_height + 0.05,
                pos_str,
                fontsize=position_label_fontsize,
                ha="center",
                va="bottom",
                color="black"
            )
        else:
            # default => show score to the side (right)
            score_x = real_start + token_len + 0.2
            score_y = label_y
            ax.text(
                score_x, score_y,
                score_str,
                fontsize=base_font_size,
                ha="left",
                va="center",
                color="black"
            )

        # Optional arrow
        if draw_alignment_arrows:
            arrow_start = (label_x, label_y)
            arrow_end   = (real_start + token_len/2, arrow_end_y)
            ax.annotate(
                "",
                xy=arrow_end, xycoords="data",
                xytext=arrow_start, textcoords="data",
                arrowprops=dict(arrowstyle="->", color=arrow_line_color, linestyle="--", alpha=0.8)
            )

        # Update vertical offset for next token
        y_offset += (rect_height + 0.25)

    # final axes limits
    total_seq_length = prefix_sums[-1]
    x_max = max(seq_len, total_seq_length) + extra_x_margin
    ax.set_xlim(0, x_max)
    ax.set_ylim(-0.5, y_offset + 0.5)
    ax.set_xlabel("Local Index (0 = 5' end, L = 3' end)")
    ax.set_ylabel("Token Rows")

    # 10) Legend
    if add_legend and len(top_data) > 0:
        token_summary={}
        for idx_ in top_indices:
            tk= tokenized_sequence[idx_]
            sc= norm_scores[idx_]
            token_summary.setdefault(tk,[]).append(sc)
        patches=[]
        for tk_, sc_list in token_summary.items():
            avg_score = np.mean(sc_list)
            patch_col = cmap(avg_score)
            label_    = f"{tk_} (Avg: {avg_score:.2f})"
            patches.append(mpatches.Patch(color=patch_col, label=label_))

        def parse_avg(lbl):
            part = lbl.split("Avg: ")[1].rstrip(")")
            return float(part)

        patches= sorted(patches, key=lambda p: parse_avg(p.get_label()), reverse=True)

        # Place legend outside right
        ax.legend(
            handles=patches,
            loc="upper left",
            bbox_to_anchor=(1.0,1.0),
            borderaxespad=0.,
            title=f"Top {top_k} Tokens"
        )

    plt.tight_layout()

    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"[info] Saved alignment plot to: {output_path}")
        plt.close()
    else:
        plt.show()


def plot_alignment_with_ig_scores(
    original_sequence,
    attention_weights,
    tokenized_sequence,
    annotation=None,
    output_path=None,
    top_k=10,
    color_map="plasma",  # e.g. a "temperature" style colormap
    dynamic_band_height=True,
    add_legend=True,
    hide_overlapping_labels=True,
    token_label_rotation=0,
    figsize=(20, 6),

    # Special tokens & filtering
    filter_special_tokens=True,
    special_tokens=("[CLS]", "[SEP]", "[PAD]", "[UNK]"),

    # Arrows and rectangle logic
    draw_alignment_arrows=True,
    arrow_line_color="lightgrey",
    arrow_end_y=0.3,        # Where the arrow meets near the base row
    rect_baseline_height=0.3,
    rect_scale_factor=0.7,

    # Local coordinate labeling
    show_token_positions=False,
    position_label_fontsize=7,
    extra_x_margin=3,  # 3-nucleotide margin at 3' end to avoid cutoff

    # Score display logic
    show_token_scores=False,
    token_score_fontsize=8
):
    """
    Plot an alignment-style visualization of the top-k tokens above a local 0..L axis,
    where L = len(original_sequence). The substring is oriented 5'→3'.

    Score Placement Rules:
    1) By default (show_token_scores=False), we place the numeric score to the
       **right** of the rectangle (side).
    2) If show_token_scores=True, we place the numeric score **on top** of the rectangle.
    3) If BOTH show_token_scores=True and show_token_positions=True, we combine
       them above the rectangle in a small multi-line style to avoid clutter.

    Other Features:
      - 2D→1D sum if needed (for attention/IG).
      - Optionally filter out special tokens (e.g., [CLS],[SEP]).
      - Temperature-based colormap. Rectangle width = token length.
      - Dashed arrows from each token's rectangle to the best substring match.
      - 5' end, 3' end labeled with gene-based coords (annotation).
      - Additional margin at 3' end so tokens near boundary aren't clipped.

    Updates: 
    - The rectangle always covers exactly [real_start .. real_start + token_len].
    - The token name stays inside the rectangle.
    - Scores and/or positions go either above or to the right—never appended to the token string where they could horizontally overflow.

    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np

    # 1) Handle 2D => 1D if needed
    if attention_weights.ndim == 2:
        attention_weights = attention_weights.sum(axis=0)
    elif attention_weights.ndim != 1:
        raise ValueError("attention_weights must be 1D or 2D.")

    if len(attention_weights) != len(tokenized_sequence):
        raise ValueError(
            f"Size mismatch: len(attention_weights)={len(attention_weights)} "
            f"vs len(tokenized_sequence)={len(tokenized_sequence)}"
        )

    # 2) Optionally filter out special tokens
    special_set = set(special_tokens)
    if filter_special_tokens:
        filtered_tokens = []
        filtered_scores = []
        for tk, sc in zip(tokenized_sequence, attention_weights):
            if tk not in special_set:
                filtered_tokens.append(tk)
                filtered_scores.append(sc)
    else:
        filtered_tokens  = tokenized_sequence
        filtered_scores  = attention_weights

    if not filtered_tokens:
        print("[warning] After filtering special tokens, no tokens left to plot.")
        return

    tokenized_sequence = filtered_tokens
    attention_weights  = np.array(filtered_scores, dtype=float)

    # 3) Extract annotation info
    if annotation is None:
        annotation = {}
    seq_len      = len(original_sequence)
    window_start = annotation.get("window_start", 0)
    window_end   = annotation.get("window_end", seq_len)
    position     = annotation.get("position", (window_start + window_end)//2)
    strand       = annotation.get("strand", "+")
    gene_id      = annotation.get("gene_id", "N/A")
    tx_id        = annotation.get("transcript_id", "N/A")
    splice_type  = annotation.get("splice_type", "N/A")
    error_type   = annotation.get("error_type", "N/A")
    reference    = annotation.get("reference", "Null Sequence")

    # local splice idx
    if strand == '+':
        local_splice_idx = position - window_start
    else:
        local_splice_idx = (window_end - 1) - position

    # 4) Normalize the attention/IG scores
    scores = attention_weights
    smin, smax = scores.min(), scores.max()
    denom = (smax - smin) + 1e-8
    norm_scores = (scores - smin) / denom

    # 5) Identify top-k
    idx_sorted   = np.argsort(norm_scores)       # ascending
    top_indices  = idx_sorted[-top_k:][::-1]     # last k descending
    top_scores   = norm_scores[top_indices]
    top_tokens   = [tokenized_sequence[i] for i in top_indices]

    # 6) Plot setup
    fig, ax = plt.subplots(figsize=figsize)
    title_str = (
        f"Alignment Plot (Gene: {gene_id}, Tx: {tx_id}, Strand: {strand})\n"
        f"Splice Type: {splice_type}, Error Type: {error_type}, Ref: {reference}"
    )
    ax.set_title(title_str, fontsize=12)

    # 7) Draw the raw DNA sequence as text at y=0
    for i, base in enumerate(original_sequence):
        ax.text(i + 0.5, 0, base, ha="center", va="center", fontsize=8)

    # Label 5' and 3' ends with gene-based coords
    if strand == '+':
        left_coord  = window_start
        right_coord = window_end
    else:
        left_coord  = window_end - 1
        right_coord = window_start
    ax.text(0,      0.2, f"5' end: {left_coord}",  ha="left",  va="bottom", color="blue", fontsize=9)
    ax.text(seq_len,0.2, f"3' end: {right_coord}", ha="right", va="bottom", color="blue", fontsize=9)

    # Mark predicted splice site
    if 0 <= local_splice_idx < seq_len:
        ax.axvline(local_splice_idx, color="red", linestyle="--", alpha=0.6)
        ax.text(local_splice_idx+1, 0.4, "Splice Site", color="red", ha="left", va="bottom", fontsize=9)

    # 8) Build naive prefix sums
    prefix_sums = [0]
    for tk in tokenized_sequence:
        prefix_sums.append(prefix_sums[-1] + len(tk))

    # build top_data => (index,score,token)
    top_data = list(zip(top_indices, top_scores, top_tokens))
    # Sort left->right by naive offset
    top_data.sort(key=lambda x: prefix_sums[x[0]])

    # function to refine offset by substring match
    def find_closest_match_in_seq(token, seq, approx_x):
        best_idx = -1
        best_diff= 1e9
        start=0
        while True:
            idx = seq.find(token, start)
            if idx < 0:
                break
            dist = abs(idx - approx_x)
            if dist < best_diff:
                best_diff= dist
                best_idx= idx
            start= idx+1
        return best_idx

    y_offset= 1.0
    placed_labels = []
    cmap = plt.get_cmap(color_map)

    # 9) Plot top tokens
    for idx_, sc_, tk_ in top_data:
        approx_start = prefix_sums[idx_]
        real_start   = find_closest_match_in_seq(tk_, original_sequence, approx_start)
        if real_start < 0:
            real_start = approx_start  # fallback if not found

        token_len  = len(tk_)
        # rectangle height
        if dynamic_band_height:
            rect_height = rect_baseline_height + sc_ * rect_scale_factor
        else:
            rect_height = rect_baseline_height + rect_scale_factor

        color = cmap(sc_)

        # Add rectangle for the token
        rect = plt.Rectangle(
            (real_start, y_offset),
            token_len,
            rect_height,
            color=color,
            alpha=0.8,
            lw=0
        )
        ax.add_patch(rect)

        # Place the token text in the center of the bar
        label_x = real_start + token_len / 2
        label_y = y_offset + rect_height / 2

        if hide_overlapping_labels:
            text_width  = token_len * 0.45
            text_height = rect_height
            overlap_found = False
            for (ox, oy, ow, oh) in placed_labels:
                if (abs(label_x - ox) < (text_width + ow)) and (abs(label_y - oy) < (text_height + oh)):
                    overlap_found = True
                    break
            if not overlap_found:
                ax.text(
                    label_x, label_y, tk_,
                    fontsize=9, ha="center", va="center",
                    rotation=token_label_rotation
                )
                placed_labels.append((label_x, label_y, text_width, text_height))
        else:
            ax.text(
                label_x, label_y, tk_,
                fontsize=9, ha="center", va="center",
                rotation=token_label_rotation
            )

        # Next: position and/or score
        # Decide logic based on show_token_scores + show_token_positions
        # 1) If show_token_scores=False => put the numeric score to the "right" of the rectangle
        # 2) If show_token_scores=True => put it "on top" of the rectangle
        # 3) If both show_token_scores and show_token_positions => 
        #    multi-line approach above the rectangle OR 
        #    side-by-side if there's enough space

        # Score string
        score_str = f"{sc_:.2f}"
        # Position string
        pos_str   = f"pos={real_start}"

        # We'll define short helper to place text. 
        # We'll keep fonts small enough if multi-line
        base_font_size = token_score_fontsize

        if show_token_scores and show_token_positions:
            # multi-line approach above the rectangle
            # small fonts
            # line1_y = y_offset + rect_height + 0.05
            # line2_y = line1_y + 0.20

            # # line1 for score
            # ax.text(
            #     label_x, line1_y,
            #     score_str,
            #     fontsize=base_font_size,
            #     ha="center", va="bottom", color="black"
            # )
            # # line2 for pos
            # ax.text(
            #     label_x, line2_y,
            #     pos_str,
            #     fontsize=position_label_fontsize,
            #     ha="center", va="bottom", color="black"
            # )

            # Place both on top but side by side
            mid_y = y_offset + rect_height + 0.05

            # Shift score left a bit, position right a bit
            score_x = label_x - 0.3
            pos_x   = label_x + 0.3

            ax.text(
                score_x, mid_y,
                score_str,
                fontsize=token_score_fontsize,
                ha="right", va="bottom", color="black"
            )
            ax.text(
                pos_x, mid_y,
                pos_str,
                fontsize=position_label_fontsize,
                ha="left", va="bottom", color="black"
            )

        elif show_token_scores and not show_token_positions:
            # put score on top
            ax.text(
                label_x,
                y_offset + rect_height + 0.05,
                score_str,
                fontsize=base_font_size,
                ha="center",
                va="bottom",
                color="black"
            )

        elif not show_token_scores and show_token_positions:
            # we put position on top
            ax.text(
                label_x,
                y_offset + rect_height + 0.05,
                pos_str,
                fontsize=position_label_fontsize,
                ha="center",
                va="bottom",
                color="black"
            )
        else:
            # default => show score to the side (right)
            score_x = real_start + token_len + 0.2
            score_y = label_y
            ax.text(
                score_x, score_y,
                score_str,
                fontsize=base_font_size,
                ha="left",
                va="center",
                color="black"
            )

        # Optional arrow
        if draw_alignment_arrows:
            arrow_start = (label_x, label_y)
            arrow_end   = (real_start + token_len/2, arrow_end_y)
            ax.annotate(
                "",
                xy=arrow_end, xycoords="data",
                xytext=arrow_start, textcoords="data",
                arrowprops=dict(arrowstyle="->", color=arrow_line_color, linestyle="--", alpha=0.8)
            )

        # Update vertical offset for next token
        y_offset += (rect_height + 0.25)

    # final axes limits
    total_seq_length = prefix_sums[-1]
    x_max = max(seq_len, total_seq_length) + extra_x_margin
    ax.set_xlim(0, x_max)
    ax.set_ylim(-0.5, y_offset + 0.5)
    ax.set_xlabel("Local Index (0 = 5' end, L = 3' end)")
    ax.set_ylabel("Token Rows")

    # 10) Legend
    if add_legend and len(top_data) > 0:
        token_summary={}
        for idx_ in top_indices:
            tk= tokenized_sequence[idx_]
            sc= norm_scores[idx_]
            token_summary.setdefault(tk,[]).append(sc)
        patches=[]
        for tk_, sc_list in token_summary.items():
            avg_score = np.mean(sc_list)
            patch_col = cmap(avg_score)
            label_    = f"{tk_} (Avg: {avg_score:.2f})"
            patches.append(mpatches.Patch(color=patch_col, label=label_))

        def parse_avg(lbl):
            part = lbl.split("Avg: ")[1].rstrip(")")
            return float(part)

        patches= sorted(patches, key=lambda p: parse_avg(p.get_label()), reverse=True)

        # Place legend outside right
        ax.legend(
            handles=patches,
            loc="upper left",
            bbox_to_anchor=(1.0,1.0),
            borderaxespad=0.,
            title=f"Top {top_k} Tokens"
        )

    plt.tight_layout()

    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"[info] Saved alignment plot to: {output_path}")
        plt.close()
    else:
        plt.show()


def visualize_alignment_attention_regions(
    original_sequence,
    attention_weights,
    tokenized_sequence,
    annotation=None,
    output_path=None,
    top_k=10,
    color_map="viridis",
    dynamic_band_height=False,
    add_legend=True,
    hide_overlapping_labels=True,
    token_label_rotation=0,
    figsize=(20, 6),
):
    """
    Plot an alignment-style visualization of the top-k tokens above a local 0..L axis,
    where L is the length of the original_sequence (contextual substring).
    The tokens are positioned using a cumulative sum of token lengths so that each token’s
    rectangle aligns exactly with the raw sequence. The splice site is marked and 5'/3' ends are annotated.

    Parameters
    ----------
    original_sequence : str
        The context substring (or reference) oriented 5'→3'. Typically, length = window_end - window_start.
    attention_weights : np.ndarray
        1D array of attention scores, one per token. (If 2D, summed over rows.)
    tokenized_sequence : list of str
        List of tokens (e.g. k-mers) covering the sequence. When concatenated (in order),
        they reconstruct the original sequence.
    annotation : dict
        Should include keys:
            - 'window_start', 'window_end' (int)
            - 'position' (int): gene-based coordinate of the splice site
            - 'strand': '+' or '-'
            - Optionally: 'gene_id', 'transcript_id', 'splice_type', 'pred_type'
    output_path : str or None
        If provided, the plot is saved as a PDF; else, it is shown.
    top_k : int
        Number of top tokens (by attention score) to highlight.
    color_map : str
        Name of the matplotlib colormap for token colors.
    dynamic_band_height : bool
        If True, the height of each token’s rectangle scales with its attention score.
    add_legend : bool
        If True, add a legend summarizing the top tokens and their average scores.
    hide_overlapping_labels : bool
        If True, use adjustText to reposition overlapping token labels.
    token_label_rotation : float
        Rotation angle (in degrees) for token label text.
    figsize : tuple
        Size of the figure, e.g., (width, height).

    Returns
    -------
    None
    """
    # import numpy as np
    # import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from adjustText import adjust_text  # pip install adjustText

    # 1) If attention_weights is 2D, sum over rows to get a 1D array.
    if attention_weights.ndim == 2:
        attention_weights = attention_weights.sum(axis=0)
    elif attention_weights.ndim != 1:
        raise ValueError("attention_weights must be 1D or 2D.")

    if len(attention_weights) != len(tokenized_sequence):
        raise ValueError("Mismatch: len(attention_weights) != len(tokenized_sequence).")

    # 2) Extract annotation values with defaults.
    if annotation is None:
        annotation = {}
    window_start = annotation.get("window_start", 0)
    window_end   = annotation.get("window_end", len(original_sequence))
    position     = annotation.get("position", (window_start + window_end) // 2)
    strand       = annotation.get("strand", "+")
    gene_id      = annotation.get("gene_id", "N/A")
    tx_id        = annotation.get("transcript_id", "N/A")
    splice_type  = annotation.get("splice_type", "N/A")
    pred_type    = annotation.get("pred_type", "N/A")
    seq_len = len(original_sequence)

    # 3) Compute local splice index for the dashed line.
    if strand == '+':
        local_splice_idx = position - window_start
    else:
        local_splice_idx = (window_end - 1) - position
    # (If out-of-range, that may be acceptable for edge cases.)

    # 4) Normalize attention scores.
    scores = attention_weights
    min_val, max_val = scores.min(), scores.max()
    denom = (max_val - min_val) + 1e-8
    norm_scores = (scores - min_val) / denom

    # 5) Identify top-k tokens.
    idx_sorted = np.argsort(norm_scores)
    top_indices = idx_sorted[-top_k:][::-1]  # highest scores, descending order.
    top_scores = norm_scores[top_indices]
    top_tokens = [tokenized_sequence[i] for i in top_indices]

    # 6) Compute cumulative token lengths to determine token offsets.
    # prefix_sums[i] = sum(len(tokenized_sequence[j]) for j in 0..i-1)
    prefix_sums = [0]
    for tk in tokenized_sequence:
        prefix_sums.append(prefix_sums[-1] + len(tk))
    # Thus, token i occupies [prefix_sums[i], prefix_sums[i+1]) in the reconstructed sequence.

    # 7) Prepare the figure and title.
    fig, ax = plt.subplots(figsize=figsize)
    title_str = (f"Alignment Plot (Gene: {gene_id}, Tx: {tx_id}, Strand: {strand})\n"
                 f"Splice: {splice_type}, Pred: {pred_type}")
    ax.set_title(title_str, fontsize=12)

    # 8) Display the raw sequence as text along the x-axis.
    for i, base in enumerate(original_sequence):
        ax.text(i + 0.5, 0, base, ha="center", va="center", fontsize=8, color="black")
    ax.hlines(y=0, xmin=0, xmax=seq_len, color="gray", linestyle="--", linewidth=0.5)

    # Annotate 5' and 3' ends (using gene-based coordinates).
    if strand == '+':
        left_coord = window_start
        right_coord = window_end
    else:
        left_coord = window_end - 1
        right_coord = window_start
    ax.text(0, 0.2, f"5' end: {left_coord}", color="blue", ha="left", va="bottom", fontsize=9)
    ax.text(seq_len, 0.2, f"3' end: {right_coord}", color="blue", ha="right", va="bottom", fontsize=9)

    # 9) Draw the splice site as a red dashed vertical line with an offset label.
    if 0 <= local_splice_idx < seq_len:
        ax.axvline(local_splice_idx, color="red", linestyle="--", alpha=0.6)
        # Place the splice site label slightly above the raw sequence.
        ax.text(local_splice_idx + 1, 0.4, "Splice Site", color="red", ha="left", va="bottom", fontsize=9)

    # 10) Gather top-k tokens along with their cumulative positions.
    # Each token i is placed at x = prefix_sums[i] with width = len(tokenized_sequence[i]).
    top_data = list(zip(top_indices, top_scores, top_tokens))
    top_data.sort(key=lambda x: prefix_sums[x[0]])  # sort left-to-right based on true offset

    # 11) Plot each top token on its own row (starting well above the raw sequence).
    y_offset = 2.0  # Start above the raw sequence row.
    token_texts = []  # Collect text objects for adjustText.
    cmap = plt.get_cmap(color_map)

    for idx_, sc_, tk_ in top_data:
        start_x = prefix_sums[idx_]
        token_len = len(tk_)
        # End position is prefix_sums[idx_ + 1]
        end_x = prefix_sums[idx_ + 1]
        color = cmap(sc_)

        # Determine rectangle height.
        if dynamic_band_height:
            # Optionally, you could add nonlinear scaling here.
            rect_height = sc_  # Simple linear scaling.
        else:
            rect_height = 0.8

        # Add rectangle for token.
        ax.add_patch(plt.Rectangle((start_x, y_offset), token_len, rect_height,
                                     color=color, alpha=0.7, lw=0))
        # Label at the center of the rectangle.
        label_x = start_x + token_len / 2
        label_y = y_offset + rect_height / 2
        label_str = f"{tk_} ({sc_:.2f})"
        t = ax.text(label_x, label_y, label_str, fontsize=9, ha="center", va="center",
                    rotation=token_label_rotation)
        token_texts.append(t)

        # Optionally, add a thin connector line from the bottom of the token rectangle to the raw sequence.
        ax.plot([label_x, label_x], [y_offset, 0.5], linestyle="--", color="gray", linewidth=0.7)

        y_offset += rect_height + 0.2

    # 12) If enabled, use adjustText to reposition overlapping token labels.
    if hide_overlapping_labels and token_texts:
        adjust_text(token_texts, arrowprops=dict(arrowstyle="-", color="black", lw=0.5))

    # 13) Build a legend summarizing the top tokens.
    if add_legend and top_data:
        token_summary = {}
        for idx_, sc_, tk_ in top_data:
            if tk_ not in token_summary:
                token_summary[tk_] = {"total": sc_, "count": 1}
            else:
                token_summary[tk_]["total"] += sc_
                token_summary[tk_]["count"] += 1

        legend_entries = []
        for tk_, stats in token_summary.items():
            avg_score = stats["total"] / stats["count"]
            patch_color = cmap(avg_score)
            label_ = f"{tk_} (Avg: {avg_score:.2f})"
            legend_entries.append(mpatches.Patch(color=patch_color, label=label_))
        # Sort patches by descending average score.
        def parse_avg(lbl):
            return float(lbl.split("Avg: ")[1].rstrip(")"))
        legend_entries = sorted(legend_entries, key=lambda p: parse_avg(p.get_label()), reverse=True)
        ax.legend(handles=legend_entries, loc="upper right", title=f"Top {top_k} Tokens")

    # 14) Final formatting.
    total_seq_length = prefix_sums[-1]
    ax.set_xlim(0, total_seq_length)
    ax.set_ylim(-0.5, y_offset + 0.5)
    ax.set_xlabel("Local Index (0 = 5' end, L = 3' end)")
    ax.set_ylabel("Token Rows")
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"[info] Saved alignment plot to: {output_path}")
        plt.close()
    else:
        plt.show()

####################################################################################################
# Plotting functions for IG and token frequency analysis


def build_freq_list_for_type(result_list, err_type="FP"):
    """
    Convert result_list => a list of (token, err_freq, tp_freq, diff)
    err_type can be 'FP' or 'FN' 
    """
    data = []
    for (tk, fp_freq, fn_freq, tp_freq, fp_diff, fn_diff) in result_list:
        if err_type == "FP":
            data.append((tk, fp_freq, tp_freq, fp_diff))
        else:  # FN
            data.append((tk, fn_freq, tp_freq, fn_diff))
    # sort by absolute diff
    data.sort(key=lambda x: abs(x[3]), reverse=True)
    return data


def build_token_frequency_comparison(
    top_token_counts,
    num_examples,
    error_label="FP",
    correct_label="TP"
):
    """
    Compare token frequencies between error and correct examples.

    Parameters:
    - top_token_counts (dict): A dictionary with token counts for error and correct examples.
    - num_examples (dict): A dictionary with the number of examples for error and correct labels.
    - error_label (str): The label for error examples (default is "FP").
    - correct_label (str): The label for correct examples (default is "TP").

    Returns:
    - list: A list of tuples (token, error_freq, correct_freq, diff) sorted by absolute difference.
        - token (str): The token.
        - error_freq (float): Frequency of the token in error examples.
        - correct_freq (float): Frequency of the token in correct examples.
        - diff (float): Difference between error_freq and correct_freq.
    """
    import math
    from collections import defaultdict

    # gather all tokens from both sets
    all_tokens = set(top_token_counts[error_label].keys()) | set(top_token_counts[correct_label].keys())

    result_list = []
    e_count = num_examples[error_label]
    c_count = num_examples[correct_label]

    for tk in all_tokens:
        e_freq = 0.0
        c_freq = 0.0
        if e_count > 0:
            e_freq = top_token_counts[error_label][tk] / e_count
        if c_count > 0:
            c_freq = top_token_counts[correct_label][tk] / c_count
        diff = e_freq - c_freq
        result_list.append((tk, e_freq, c_freq, diff))

    # sort by absolute difference
    result_list.sort(key=lambda x: abs(x[3]), reverse=True)
    return result_list


def bar_chart_freq(result_list, error_label="FP", correct_label="TP", top_n=20, output_path=None):
    """
    Plot a bar chart comparing token frequencies between error and correct examples.
    """

    import matplotlib.pyplot as plt
    import numpy as np

    # slice top_n
    data = result_list[:top_n]
    tokens = [x[0] for x in data]
    efreqs = [x[1] for x in data]
    cfreqs = [x[2] for x in data]
    diffs  = [x[3] for x in data]

    x = np.arange(len(data))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12,6))
    rects_e = ax.bar(x - width/2, efreqs, width, label=error_label)
    rects_c = ax.bar(x + width/2, cfreqs, width, label=correct_label)

    ax.set_xticks(x)
    ax.set_xticklabels(tokens, rotation=45, ha="right")
    ax.set_ylabel("Frequency in Top-K IG tokens")
    ax.set_title(f"Comparison: {error_label} vs {correct_label} (top {top_n} tokens by abs diff)")
    ax.legend()

    # optional numeric annotations
    for rect in rects_e:
        height = rect.get_height()
        ax.annotate(f"{height:.2f}",
                    xy=(rect.get_x() + rect.get_width()/2, height),
                    xytext=(0,3),
                    textcoords="offset points",
                    ha="center", va="bottom", fontsize=8)

    for rect in rects_c:
        height = rect.get_height()
        ax.annotate(f"{height:.2f}",
                    xy=(rect.get_x() + rect.get_width()/2, height),
                    xytext=(0,3),
                    textcoords="offset points",
                    ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"[info] Saved token frequency barchart to: {output_path}")
        plt.close()
    else:
        plt.show()


def plot_token_frequency_bar(result_list, top_n=20, sort_by="diff", output_path=None):
    """
    result_list: list of (token, fp_freq, tp_freq, diff_or_ratio)
    top_n: how many tokens to show
    sort_by: either "diff" or "ratio" depending on what you store in the 4th element
    """
    # 1) Slice top N (already sorted by abs diff or ratio)
    displayed = result_list[:top_n]

    # 2) Separate data
    tokens   = [d[0] for d in displayed]
    fp_freqs = [d[1] for d in displayed]
    tp_freqs = [d[2] for d in displayed]
    diffs    = [d[3] for d in displayed]  # might be diff or ratio

    # 3) X positions
    x = np.arange(len(displayed))  # 0.. top_n-1
    width = 0.35

    # 4) Plot
    fig, ax = plt.subplots(figsize=(12, 6))

    # FP bars on the left
    rects_fp = ax.bar(x - width/2, fp_freqs, width, label='FP freq')
    # TP bars on the right
    rects_tp = ax.bar(x + width/2, tp_freqs, width, label='TP freq')

    # 5) Labeling
    ax.set_ylabel("Frequency in Top-K IG Tokens")
    ax.set_title("Token Frequency for FPs vs TPs (Top N by Difference)")
    ax.set_xticks(x)
    ax.set_xticklabels(tokens, rotation=45, ha="right")
    ax.legend()

    # Optionally add text above bars
    for rect in rects_fp:
        height = rect.get_height()
        ax.annotate(f"{height:.2f}",
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

    for rect in rects_tp:
        height = rect.get_height()
        ax.annotate(f"{height:.2f}",
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"[info] Saved token frequency plot to: {output_path}")
        plt.close()
    else:
        plt.show()


####################################################################################################


def map_token_positions_with_context(
    attention_weights,
    tokenized_sequence,
    original_sequence,
    kmer_size=6
):
    """
    Map all token positions in the original sequence with context-aware attention scores.

    Parameters:
    - attention_weights: 2D NumPy array of attention weights (seq_len x seq_len).
    - tokenized_sequence: Tokenized sequence as a list of tokens.
    - original_sequence: Original DNA sequence.
    - kmer_size: K-mer size used during tokenization.

    Returns:
    - token_positions: List of dictionaries with token, start, end, and context-aware attention score.
    """
    import re

    seq_len = len(tokenized_sequence)
    token_positions = []

    # Compute aggregate attention scores for each token (column-wise)
    aggregate_scores = attention_weights.sum(axis=0)

    # Normalize attention scores
    normalized_scores = (aggregate_scores - np.min(aggregate_scores)) / (
        np.max(aggregate_scores) - np.min(aggregate_scores) + 1e-8
    )

    # Map tokens back to the original sequence
    for idx, token in enumerate(tokenized_sequence):
        token_matches = [m.start() for m in re.finditer(token, original_sequence)]
        for match_pos in token_matches:
            context_score = normalized_scores[idx]
            token_positions.append({
                "token": token,
                "start": match_pos,
                "end": match_pos + kmer_size,
                "attention_score": context_score
            })

    return token_positions


def map_token_positions_with_variable_lengths(
    attention_weights,
    tokenized_sequence,
    original_sequence
):
    """
    Map variable-length tokens to positions in the original sequence with context-aware attention scores.

    Parameters:
    - attention_weights: 2D NumPy array of attention weights (seq_len x seq_len).
    - tokenized_sequence: Tokenized sequence as a list of tokens.
    - original_sequence: Original DNA sequence.

    Returns:
    - token_positions: List of dictionaries with token, start, end, and context-aware attention score.
    """
    import re

    seq_len = len(tokenized_sequence)
    token_positions = []

    # Compute aggregate attention scores for each token (column-wise)
    aggregate_scores = attention_weights.sum(axis=0)

    # Normalize attention scores
    normalized_scores = (aggregate_scores - np.min(aggregate_scores)) / (
        np.max(aggregate_scores) - np.min(aggregate_scores) + 1e-8
    )

    # Map tokens back to the original sequence
    for idx, token in enumerate(tokenized_sequence):
        token_matches = [m.start() for m in re.finditer(re.escape(token), original_sequence)]
        for match_pos in token_matches:
            
            context_score = normalized_scores[idx]
            # This value represents the aggregated attention score for a specific token at position idx 
            # in the tokenized sequence.
            # The score reflects how much attention this token receives across all queries in 
            # the attention mechanism for the specified layer and head.

            # Problem: 
            #   - If the same token (e.g., "TGAA") occurs multiple times in the original sequence, 
            #   its attention score does not change for different positions within the same tokenized sequence 
            #   because normalized_scores[idx] is fixed for that token in the heatmap
            #   - This score reflects the model's "global attention" to the token based on the heatmap 
            #     at the specified layer and head.

            token_positions.append({
                "token": token,
                "start": match_pos,
                "end": match_pos + len(token),  # Use token length
                "attention_score": context_score
            })

    return token_positions


def map_token_positions_with_context(
    attention_weights,
    tokenized_sequence,
    original_sequence
):
    """
    Map all token positions in the original sequence with context-specific attention scores.

    Parameters:
    - attention_weights: 2D NumPy array of attention weights (seq_len x seq_len).
    - tokenized_sequence: Tokenized sequence as a list of tokens.
    - original_sequence: Original DNA sequence.

    Returns:
    - token_positions: List of dictionaries with token, start, end, and context-specific attention scores.

    Memo: 
    
    - Each token occurrence is scored independently based on its attention contribution to 
       specific query tokens in the heatmap.
    - If "TGAA" occurs multiple times, each occurrence will now have its own contextual attention score 
      based on its role in the tokenized sequence.
    """
    import re

    seq_len = len(tokenized_sequence)
    token_positions = []

    # Map tokens back to the original sequence
    for idx, token in enumerate(tokenized_sequence):
        # Find all occurrences of the token in the original sequence
        token_matches = [m.start() for m in re.finditer(re.escape(token), original_sequence)]
        for match_pos in token_matches:
            # Extract context-specific attention score (row-wise attention)
            context_score = attention_weights[idx, :]  # Row for the current token
            context_score_normalized = (context_score - np.min(context_score)) / (
                np.max(context_score) - np.min(context_score) + 1e-8
            )
            token_positions.append({
                "token": token,
                "start": match_pos,
                "end": match_pos + len(token),
                "attention_score": context_score_normalized[idx]
            })

    return token_positions


def generate_motifs_from_sequence(sequence, min_k=2, max_k=4):
    """
    Generate k-mers from the input sequence.

    Parameters:
    - sequence: Input DNA sequence (string).
    - min_k: Minimum k-mer size.
    - max_k: Maximum k-mer size.

    Returns:
    - motifs: List of k-mers (2-mers to 4-mers).
    """
    motifs = set()
    for k in range(min_k, max_k + 1):
        for i in range(len(sequence) - k + 1):
            motifs.add(sequence[i:i + k])
    return sorted(motifs)  # Sort motifs for consistent order


def map_motifs_to_sequence(motifs, original_sequence):
    """
    Map important motifs to positions in the original DNA sequence.

    Parameters:
    - motifs: List of important motifs (strings) to map.
    - original_sequence: Original DNA sequence (string).

    Returns:
    - mapped_motifs: List of dictionaries with motif, start, end, and occurrence count.
    """
    import re

    mapped_motifs = []

    for motif in motifs:
        # Find all occurrences of the motif
        matches = [m.start() for m in re.finditer(re.escape(motif), original_sequence)]
        for match_pos in matches:
            mapped_motifs.append({
                "motif": motif,
                "start": match_pos,
                "end": match_pos + len(motif),
            })

    return mapped_motifs


def validate_motifs_with_attention(mapped_motifs, attention_weights, tokenized_sequence, original_sequence):
    """
    Validate mapped motifs using context-aware attention weights.

    Parameters:
    - mapped_motifs: List of motifs mapped to their positions in the original sequence.
    - attention_weights: 2D NumPy array of attention weights (seq_len x seq_len).
    - tokenized_sequence: Tokenized sequence as a list of tokens.
    - original_sequence: Original DNA sequence.

    Returns:
    - validated_motifs: List of motifs with their contextual attention scores.
    """
    validated_motifs = []

    for motif_info in mapped_motifs:
        start, end = motif_info["start"], motif_info["end"]
        motif = motif_info["motif"]

        # Find corresponding tokens in the tokenized sequence
        context_scores = []
        for idx, token in enumerate(tokenized_sequence):
            token_start = original_sequence.find(token)
            token_end = token_start + len(token)
            if token_start >= start and token_end <= end:
                # Extract context-aware attention score for the token
                context_scores.append(attention_weights[idx, :].mean())  # Aggregate row attention

        # Compute mean contextual attention score for the motif
        if context_scores:
            mean_attention_score = sum(context_scores) / len(context_scores)
            validated_motifs.append({
                "motif": motif,
                "start": start,
                "end": end,
                "mean_attention_score": mean_attention_score,
            })

    return validated_motifs


def visualize_validated_motifs(original_sequence, validated_motifs, output_path=None):
    """
    Visualize the original sequence with validated motifs highlighted.

    Parameters:
    - original_sequence: Original DNA sequence.
    - validated_motifs: List of validated motifs with their positions and attention scores.
    - output_path (str): File path to save the visualization.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # Create a figure
    plt.figure(figsize=(15, 4))
    plt.plot(range(len(original_sequence)), [0] * len(original_sequence), "k-", alpha=0.5)

    # Highlight motifs with intensity based on attention scores
    for motif in validated_motifs:
        start, end = motif["start"], motif["end"]
        attention_score = motif["mean_attention_score"]
        plt.axvspan(start, end, color="red", alpha=attention_score, label=f"{motif['motif']} ({attention_score:.2f})")

    plt.title("Validated Motifs in Context")
    plt.xlabel("Position in Sequence")
    plt.ylabel("Attention Weight")
    plt.legend(loc="upper right", bbox_to_anchor=(1.1, 1))

    # Save or display the plot
    if output_path:
        plt.savefig(output_path, bbox_inches="tight", dpi=200)
        print(f"[info] Visualization saved to: {output_path}")
        plt.close()
    else:
        plt.show()


def visualize_contextual_importance_v1(
    original_sequence,
    token_positions,
    output_path=None,
    top_tokens=10,
    color_map="magma", 
    motif_key="motif", 
    attention_key="attention_score"
):
    """
    Visualize the original sequence with context-aware importance highlights.

    Parameters:
    - original_sequence: Original DNA sequence.
    - token_positions: List of token positions with attention scores.
    - output_path (str): File path to save the visualization.
    - top_tokens (int): Number of most important tokens to include in the legend.
    - color_map (str): Color map for highlighting important regions.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np

    # Dynamically retrieve attention scores
    def get_attention_score(token):
        for key, value in token.items():
            if "attention_score" in key:
                return value
        return 0  # Default to 0 if no attention score is found

    # Sort tokens by attention score (descending order)
    token_positions = sorted(token_positions, key=get_attention_score, reverse=True)

    # Limit to top tokens for the legend
    unique_tokens = {}
    for pos in token_positions:
        if len(unique_tokens) < top_tokens:
            unique_tokens[pos[motif_key]] = get_attention_score(pos)

    # Create a figure
    plt.figure(figsize=(15, 4))
    plt.plot(range(len(original_sequence)), [0] * len(original_sequence), "k-", alpha=0.5)

    # Highlight tokens using a color map
    cmap = plt.get_cmap(color_map)
    for pos in token_positions:
        start, end = pos["start"], pos["end"]
        attention_score = get_attention_score(pos)
        color = cmap(attention_score)
        plt.axvspan(start, end, color=color, label=f"{pos[motif_key]}")

    # Format the plot
    plt.title("Context-Aware Highlight of Important Regions")
    plt.xlabel("Position in Sequence")
    plt.ylabel("Attention Weight")
    plt.ylim([-0.1, 0.1])  # Adjust y-limits to focus on the highlights

    # Add a custom legend outside the plot
    legend_patches = [
        mpatches.Patch(color=cmap(score), label=f"{token} ({score:.2f})")
        for token, score in unique_tokens.items()
    ]
    plt.legend(handles=legend_patches, loc="upper left", bbox_to_anchor=(1, 1), title="Top Tokens")

    # Save or display the plot
    if output_path:
        plt.savefig(output_path, bbox_inches="tight", dpi=200)
        print(f"[info] Visualization saved to: {output_path}")
        plt.close()
    else:
        plt.show()


def visualize_contextual_importance(
    original_sequence,
    token_positions,
    output_path=None,
    top_tokens=10,
    color_map="plasma",  # Aesthetic color map
    motif_key="motif",
    attention_key="mean_attention_score"
):
    """
    Visualize the original sequence with context-aware importance highlights.

    Parameters:
    - original_sequence: Original DNA sequence.
    - token_positions: List of token positions with attention scores.
    - output_path (str): File path to save the visualization.
    - top_tokens (int): Number of most important tokens to include in the legend.
    - color_map (str): Color map for highlighting important regions.
    - motif_key: Key for accessing the motif name in token_positions.
    - attention_key: Key for accessing the attention score in token_positions.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np

    # Retrieve attention scores dynamically
    def get_attention_score(token):
        return token.get(attention_key, 0)

    # Sort tokens by attention score (descending order)
    token_positions = sorted(token_positions, key=get_attention_score, reverse=True)

    # Normalize attention scores for consistent visualization
    scores = [get_attention_score(pos) for pos in token_positions]
    min_score, max_score = min(scores), max(scores)
    normalized_scores = [(score - min_score) / (max_score - min_score + 1e-8) for score in scores]

    # Aggregate motif occurrences for the legend
    motif_occurrences = {}
    for pos, norm_score in zip(token_positions, normalized_scores):
        motif = pos.get(motif_key, "Unknown")
        if motif not in motif_occurrences:
            motif_occurrences[motif] = []
        motif_occurrences[motif].append(norm_score)

    # Limit the legend to top tokens
    top_motifs = sorted(motif_occurrences.keys(), key=lambda x: max(motif_occurrences[x]), reverse=True)[:top_tokens]

    # Create a figure
    plt.figure(figsize=(15, 4))
    plt.plot(range(len(original_sequence)), [0] * len(original_sequence), "k-", alpha=0.5)

    # Highlight tokens using a color map
    cmap = plt.get_cmap(color_map)
    for pos, norm_score in zip(token_positions, normalized_scores):
        start, end = pos["start"], pos["end"]
        color = cmap(norm_score)
        plt.axvspan(start, end, color=color, label=f"{pos[motif_key]}")

    # Format the plot
    plt.title("Context-Aware Highlight of Important Regions")
    plt.xlabel("Position in Sequence")
    plt.ylabel("Attention Weight")
    plt.ylim([-0.1, 0.1])  # Adjust y-limits to focus on the highlights

    # Add a custom legend outside the plot
    # legend_patches = [
    #     mpatches.Patch(color=cmap(score), label=f"{token} ({score:.2f})")
    #     for token, score in unique_tokens.items()
    # ]
    # plt.legend(handles=legend_patches, loc="upper left", bbox_to_anchor=(1, 1), title="Top Motifs")
    legend_patches = [
        mpatches.Patch(
            color=cmap(max(motif_occurrences[motif])),
            label=f"{motif} (Max Score: {max(motif_occurrences[motif]):.2f})"
        )
        for motif in top_motifs
    ]
    plt.legend(handles=legend_patches, loc="upper left", bbox_to_anchor=(1, 1), title="Top Motifs")

    # Save or display the plot
    if output_path:
        plt.savefig(output_path, bbox_inches="tight", dpi=200)
        print(f"[info] Visualization saved to: {output_path}")
        plt.close()
    else:
        plt.show()


# But externally derived motifs (e.g., from SHAP or other methods) may not align perfectly with 
# the tokens generated by the tokenizer. 
# This mismatch poses a challenge when attempting to assign context-aware attention scores to the motifs ...

def map_and_validate_motifs_with_attention_v0(
    motifs, original_sequence, attention_weights, tokenized_sequence, fallback=True
):
    """
    Map externally derived motifs to the original sequence and estimate context-aware attention scores.

    Parameters:
    - motifs: List of important motifs (strings).
    - original_sequence: Original DNA sequence (string).
    - attention_weights: 2D NumPy array of attention weights (seq_len x seq_len).
    - tokenized_sequence: Tokenized sequence as a list of tokens.
    - fallback (bool): Whether to allow partial overlaps as a fallback.

    Returns:
    - validated_motifs: List of motifs with context-aware attention scores and positions.
    """
    import re

    validated_motifs = []

    for motif in motifs:
        matches = [m.start() for m in re.finditer(re.escape(motif), original_sequence)]
        for match_pos in matches:
            motif_start, motif_end = match_pos, match_pos + len(motif)

            # Find the best overlapping token
            best_token = None
            best_token_score = 0
            best_overlap = 0

            for idx, token in enumerate(tokenized_sequence):
                token_start = original_sequence.find(token)
                token_end = token_start + len(token)

                # Check for full overlap or complete enclosure
                if token_start <= motif_start and token_end >= motif_end:
                    overlap_length = motif_end - motif_start  # Fully enclosed
                    token_score = attention_weights[idx, :].mean()

                    # Update best match
                    if overlap_length > best_overlap or (overlap_length == best_overlap and token_score > best_token_score):
                        best_token = token
                        best_token_score = token_score
                        best_overlap = overlap_length

            # Fallback to partial overlaps if no exact match
            if fallback and not best_token:
                for idx, token in enumerate(tokenized_sequence):
                    token_start = original_sequence.find(token)
                    token_end = token_start + len(token)

                    # Check for partial overlap
                    if token_start < motif_end and token_end > motif_start:
                        overlap_length = min(motif_end, token_end) - max(motif_start, token_start)
                        token_score = attention_weights[idx, :].mean()

                        # Update best match
                        if overlap_length > best_overlap or (overlap_length == best_overlap and token_score > best_token_score):
                            best_token = token
                            best_token_score = token_score
                            best_overlap = overlap_length

            # Record the best match
            if best_token:
                validated_motifs.append({
                    "motif": motif,
                    "start": motif_start,
                    "end": motif_end,
                    "best_token": best_token,
                    "best_token_score": best_token_score,
                    "overlap_length": best_overlap,
                    "attention_score": best_token_score
                })

    return validated_motifs


def map_and_validate_motifs_with_attention(
    motifs, original_sequence, attention_weights, tokenized_sequence, fallback=True
):
    """
    Map externally derived motifs to the original sequence and estimate context-aware attention scores.
    """

    import re
    validated_motifs = []

    # Normalize attention weights
    max_score = np.max(attention_weights)
    min_score = np.min(attention_weights)
    normalized_weights = (attention_weights - min_score) / (max_score - min_score + 1e-8)

    for motif in motifs:
        matches = [m.start() for m in re.finditer(re.escape(motif), original_sequence)]
        for match_pos in matches:
            motif_start, motif_end = match_pos, match_pos + len(motif)
            best_token = None
            best_token_score = 0
            best_overlap = 0

            for idx, token in enumerate(tokenized_sequence):
                token_start = original_sequence.find(token)
                token_end = token_start + len(token)

                if token_start <= motif_start and token_end >= motif_end:
                    overlap_length = motif_end - motif_start
                    token_score = normalized_weights[idx, :].mean()

                    if overlap_length > best_overlap or (
                        overlap_length == best_overlap and token_score > best_token_score
                    ):
                        best_token = token
                        best_token_score = token_score
                        best_overlap = overlap_length

            if fallback and not best_token:
                for idx, token in enumerate(tokenized_sequence):
                    token_start = original_sequence.find(token)
                    token_end = token_start + len(token)

                    if token_start < motif_end and token_end > motif_start:
                        overlap_length = min(motif_end, token_end) - max(motif_start, token_start)
                        token_score = normalized_weights[idx, :].mean()

                        if overlap_length > best_overlap or (
                            overlap_length == best_overlap and token_score > best_token_score
                        ):
                            best_token = token
                            best_token_score = token_score
                            best_overlap = overlap_length

            if best_token:
                validated_motifs.append({
                    "motif": motif,
                    "start": motif_start,
                    "end": motif_end,
                    "best_token": best_token,
                    "best_token_score": best_token_score,
                    "overlap_length": best_overlap,
                    "attention_score": best_token_score
                })

    return validated_motifs


def visualize_contextual_importance_with_annotations(
    original_sequence,
    validated_motifs,
    output_path=None,
    top_motifs=10,
    color_map="viridis",
    dynamic_band_height=True,
    motif_key="motif", # Key to retrieve motif sequence
    attention_key="attention_score"  # Key to retrieve attention scores
):
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np

    # Retrieve attention scores dynamically
    def get_attention_score(token):
        return token.get(attention_key, 0)

    validated_motifs = sorted(validated_motifs, key=lambda x: x["attention_score"], reverse=True)
    top_motifs = validated_motifs[:top_motifs]

    cmap = plt.get_cmap(color_map)
    max_score = max(motif["attention_score"] for motif in validated_motifs)

    plt.figure(figsize=(15, 6))
    plt.plot(range(len(original_sequence)), [0] * len(original_sequence), "k-", alpha=0.5)

    for motif in validated_motifs:
        start, end = motif["start"], motif["end"]
        score = motif["attention_score"]
        height = score if dynamic_band_height else 1.0
        color = cmap(score / max_score)
        plt.axvspan(start, end, 0, height, color=color, alpha=0.7)

    for i, motif in enumerate(top_motifs):
        start, end = motif["start"], motif["end"]
        x_pos = (start + end) // 2
        y_pos = max_score + 0.05 + (i * 0.1) % 0.5
        plt.annotate(
            motif["motif"],
            xy=(x_pos, 0),
            xytext=(x_pos, y_pos),
            arrowprops=dict(facecolor="black", arrowstyle="->"),
            fontsize=9,
            ha="center",
        )

    legend_patches = [
        mpatches.Patch(
            color=cmap(motif["attention_score"] / max_score),
            label=f"{motif['motif']} (Score: {motif['attention_score']:.2f})",
        )
        for motif in top_motifs
    ]
    plt.legend(handles=legend_patches, loc="upper right", title="Top Motifs")

    plt.title("Motif Visualization with Attention Scores")
    plt.xlabel("Position in Sequence")
    plt.ylabel("Attention Weight")
    plt.ylim(0, max_score + 0.3)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, bbox_inches="tight", dpi=300)
        print(f"[info] Visualization saved to: {output_path}")
        plt.close()
    else:
        plt.show()


def visualize_contextual_importance_with_annotations_v0(
    original_sequence,
    validated_motifs,
    output_path=None,
    top_motifs=10,  # Limit the number of motifs in the legend
    color_map="viridis",  # Perceptually uniform colormap
    dynamic_band_height=True,  # Adjust band heights based on attention scores
    motif_key="motif", # Key to retrieve motif sequence
    attention_key="attention_score"  # Key to retrieve attention scores
):
    """
    Visualize motifs with context-aware attention scores and annotations.

    Parameters:
    - original_sequence: The full DNA sequence (string).
    - validated_motifs: List of motifs with positions and attention scores.
    - output_path (str): File path to save the visualization.
    - top_motifs (int): Number of motifs to include in the legend.
    - color_map (str): Colormap for motif highlighting.
    - dynamic_band_height (bool): Adjust band heights based on attention scores.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np

    # Retrieve attention scores dynamically
    def get_attention_score(token):
        return token.get(attention_key, 0)

    # Sort motifs by attention score and limit to top motifs
    validated_motifs = sorted(validated_motifs, key=lambda x: x["attention_score"], reverse=True)
    top_motifs = validated_motifs[:top_motifs]

    # Prepare color mapping for motifs
    cmap = plt.get_cmap(color_map)
    max_score = max(motif["attention_score"] for motif in validated_motifs)

    # Create the figure
    plt.figure(figsize=(15, 6))
    plt.plot(range(len(original_sequence)), [0] * len(original_sequence), "k-", alpha=0.5)

    # Highlight motifs with dynamic band heights
    for motif in validated_motifs:
        start, end = motif["start"], motif["end"]
        score = motif["attention_score"]
        height = score if dynamic_band_height else 1.0  # Full height if dynamic band height is off
        color = cmap(score / max_score)  # Normalize score to colormap
        plt.axvspan(start, end, 0, height, color=color, alpha=0.7)

    # Annotate motifs with arrows
    for i, motif in enumerate(top_motifs):
        start, end = motif["start"], motif["end"]
        x_pos = (start + end) // 2
        y_pos = max_score + 0.05 + (i * 0.1) % 0.5  # Stagger annotations vertically within a small range
        plt.annotate(
            motif[motif_key],
            xy=(x_pos, 0),
            xytext=(x_pos, y_pos),
            arrowprops=dict(facecolor="black", arrowstyle="->"),
            fontsize=9,
            ha="center",
        )

    ### Add a legend for the top motifs

    # Prepare legend data
    motif_scores = {}
    for motif in validated_motifs:
        if motif[motif_key] not in motif_scores:
            motif_scores[motif[motif_key]] = {
                "max_score": motif[attention_key],
                "mean_score": motif[attention_key],
                "count": 1,
            }
        else:
            motif_scores[motif[motif_key]]["max_score"] = max(
                motif_scores[motif[motif_key]]["max_score"], motif[attention_key]
            )
            motif_scores[motif[motif_key]]["mean_score"] += motif[attention_key]
            motif_scores[motif[motif_key]]["count"] += 1

    # Compute final mean scores
    for motif, scores in motif_scores.items():
        scores["mean_score"] /= scores["count"]

    # Limit legend to top motifs
    top_motifs_set = set([motif[motif_key] for motif in top_motifs])
    legend_motifs = {
        motif: scores
        for motif, scores in motif_scores.items()
        if motif in top_motifs_set
    }

    # Configure the legend
    legend_patches = [
        mpatches.Patch(
            color=cmap(scores["max_score"] / max_score),
            label=f"{motif} (Max: {scores['max_score']:.2f}, Mean: {scores['mean_score']:.2f})",
        )
        for motif, scores in legend_motifs.items()
    ]
    plt.legend(handles=legend_patches, loc="upper right", title="Top Motifs")

    # Format the plot
    plt.title("Motif Visualization with Context-Aware Attention")
    plt.xlabel("Position in Sequence")
    plt.ylabel("Attention Weight")
    plt.ylim(0, max_score + 0.3)
    plt.tight_layout()

    # Save or display the plot
    if output_path:
        plt.savefig(output_path, bbox_inches="tight", dpi=300)
        print(f"[info] Visualization saved to: {output_path}")
        plt.close()
    else:
        plt.show()


####################################################################################################


def visualize_motifs_and_attention(sequence, motif_positions, attention_weights):
    """
    Visualize motifs and attention weights in a DNA sequence.

    Parameters:
    - sequence (str): DNA sequence.
    - motif_positions (list): List of tuples (motif, start, end).
    - attention_weights (numpy.ndarray): Attention weights for the sequence.
    """
    seq_len = len(sequence)
    attention_scores = attention_weights.mean(axis=0)[:seq_len]  # Average attention over sequence positions

    # Initialize figure
    plt.figure(figsize=(15, 5))

    # Plot attention weights
    plt.bar(range(seq_len), attention_scores, alpha=0.5, label="Attention Weight", color="blue")

    # Highlight motifs
    for motif, start, end in motif_positions:
        plt.axvspan(start, end, color="red", alpha=0.3, label=f"Motif: {motif}")

    # Add labels and legend
    plt.xlabel("Sequence Position")
    plt.ylabel("Attention Weight")
    plt.title("Attention Weights and Motif Positions in DNA Sequence")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()


def find_motif_positions(sequence, motifs):
    """
    Find positions of important motifs in a DNA sequence.

    Parameters:
    - sequence (str): DNA sequence.
    - motifs (list): List of important motifs (e.g., k-mers).

    Returns:
    - motif_positions (list): List of tuples (motif, start, end).
    """
    motif_positions = []
    for motif in motifs:
        start = 0
        while (start := sequence.find(motif, start)) != -1:
            motif_positions.append((motif, start, start + len(motif)))
            start += 1  # Move to the next position
    return motif_positions


####################################################################################################

# Strategies to Reduce Visual Clutter and Highlight Important Motifs
def filter_motifs_by_attention(sequence, motif_positions, attention_weights, threshold_percentile=90):
    """
    Filter motifs based on average attention weights in their sequence regions.

    Parameters:
    - sequence (str): DNA sequence.
    - motif_positions (list): List of tuples (motif, start, end).
    - attention_weights (numpy.ndarray): Attention weights for the sequence.
    - threshold_percentile (float): Percentile threshold for filtering.

    Returns:
    - filtered_motifs (list): List of motifs that pass the threshold.
    """
    # Calculate the attention threshold
    threshold = np.percentile(attention_weights, threshold_percentile)

    filtered_motifs = []
    for motif, start, end in motif_positions:
        # Calculate the average attention weight for the motif region
        avg_attention = attention_weights[start:end].mean()
        if avg_attention > threshold:
            filtered_motifs.append((motif, start, end, avg_attention))

    return filtered_motifs

# For motifs that overlap in the sequence, group them into clusters and represent each cluster by 
# the motif with the highest attention weight.

def group_overlapping_motifs(motif_positions):
    """
    Group overlapping motifs and retain the one with the highest attention weight.

    Parameters:
    - motif_positions (list): List of tuples (motif, start, end, avg_attention).

    Returns:
    - grouped_motifs (list): List of non-overlapping motifs.
    """
    # Sort motifs by start position
    motif_positions = sorted(motif_positions, key=lambda x: (x[1], -x[3]))  # Sort by start and descending attention

    grouped_motifs = []
    current_group = []

    for motif in motif_positions:
        if not current_group or motif[1] > current_group[-1][2]:  # No overlap
            if current_group:
                # Append the motif with the highest attention from the current group
                grouped_motifs.append(max(current_group, key=lambda x: x[3]))
            current_group = [motif]
        else:
            current_group.append(motif)

    # Add the last group
    if current_group:
        grouped_motifs.append(max(current_group, key=lambda x: x[3]))

    return grouped_motifs

# Sort motifs by their average attention weight and display only the top N motifs in the plot.

def select_top_motifs(motif_positions, top_n=10):
    """
    Select the top N motifs based on their average attention weights.

    Parameters:
    - motif_positions (list): List of tuples (motif, start, end, avg_attention).
    - top_n (int): Number of motifs to select.

    Returns:
    - top_motifs (list): List of the top N motifs.
    """
    return sorted(motif_positions, key=lambda x: -x[3])[:top_n]


def visualize_motifs_and_attention_filtered(sequence, motif_positions, attention_weights):
    """
    Visualize filtered motifs and attention weights in a DNA sequence.

    Parameters:
    - sequence (str): DNA sequence.
    - motif_positions (list): List of filtered motifs (motif, start, end, avg_attention).
    - attention_weights (numpy.ndarray): Attention weights for the sequence.
    """
    seq_len = len(sequence)
    attention_scores = attention_weights.mean(axis=0)[:seq_len]  # Average attention over sequence positions

    # Initialize figure
    plt.figure(figsize=(15, 5))

    # Plot attention weights
    plt.bar(range(seq_len), attention_scores, alpha=0.5, label="Attention Weight", color="blue")

    # Highlight motifs
    for motif, start, end, avg_attention in motif_positions:
        plt.axvspan(start, end, color="red", alpha=0.3, label=f"Motif: {motif} ({avg_attention:.2f})")

    # Add labels and legend
    plt.xlabel("Sequence Position")
    plt.ylabel("Attention Weight")
    plt.title("Attention Weights and Top Motifs in DNA Sequence")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()


def prepare_motifs_for_visualization(motif_df, motif_column='motif', score_column='importance_score', return_scores=False):
    """
    Prepare motifs for visualization by converting them to the correct format for find_motif_positions().

    Parameters:
    - motif_df (pd.DataFrame): DataFrame containing motifs and their importance scores.
    - motif_column (str): Name of the column containing motif features (e.g., '3mer_TTT').
    - score_column (str): Name of the column containing importance scores.
    - return_scores (bool): Whether to return scores along with the motifs.

    Returns:
    - motifs (list): List of motif strings (e.g., ['TTT', 'AGA', ...]).
    - (optional) motifs_with_scores (list of tuples): List of (motif, score) tuples.
    """
    # Sort the DataFrame by the importance score in descending order
    motif_df = motif_df.sort_values(by=score_column, ascending=False)

    # Extract only the motif sequence (e.g., '3mer_TTT' -> 'TTT')
    motif_df['parsed_motif'] = motif_df[motif_column].str.extract(r'mer_(.*)$')[0]

    if return_scores:
        # Return motifs along with their scores
        motifs_with_scores = list(zip(motif_df['parsed_motif'], motif_df[score_column]))
        return motifs_with_scores

    # Return only the motifs
    motifs = motif_df['parsed_motif'].tolist()
    return motifs


####################################################################################################
# Integrated gradients


def forward_fn(input_ids, attention_mask, model):
    """
    Return the logit for the positive class (index=1) in binary classification.
    input_ids: (batch_size, seq_len)
    attention_mask: (batch_size, seq_len)
    model: The fine-tuned DNABERT model
    """
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits  # shape (batch_size, num_labels)
    # We'll interpret the logit for class index 1 (i.e., predicted=1)
    # For multi-class, you might specify the class of interest or do a for-loop
    class_logit = logits[:, 1]
    return class_logit


def compute_ig_attributions(
    model,
    tokenizer,
    sequence,
    device,
    baseline_seq=None,
    n_steps=50,
    max_length=512
):
    """
    Compute Integrated Gradients attributions for a single DNA sequence input.

    :param model: Fine-tuned DNABERT model (on device).
    :param tokenizer: DNABERT tokenizer.
    :param sequence: Raw DNA sequence (str).
    :param device: torch device (e.g. "cuda").
    :param baseline_seq: Baseline sequence (str) or None for an all-"N" baseline.
    :param n_steps: Number of steps for IG approximation.
    :param max_length: Truncation limit for the sequence.

    :return: A tuple (attributions, tokens)
        attributions: np.ndarray of shape (seq_len,) with IG values per token.
        tokens: the list of token strings from the DNABERT tokenizer.
    """
    from captum.attr import IntegratedGradients  # pip install captum
    from torch.nn import functional as F

    # 1) Tokenize
    inputs = tokenizer(
        sequence,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_length
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    # 2) Build baseline
    if baseline_seq is None:
        baseline_seq = "A" * len(sequence)  # same length as actual sequence

    baseline_inputs = tokenizer(
        baseline_seq,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_length
    )
    baseline_ids = baseline_inputs["input_ids"].to(device)

    print("input_ids shape:", input_ids.shape)
    print("baseline_ids shape:", baseline_ids.shape)
    
    # We typically can reuse the same attention_mask for baseline if the length is the same,
    # or we build a baseline_mask similarly:
    baseline_mask = baseline_inputs["attention_mask"].to(device)

    # 3) Define a custom forward function
    def captum_forward(inputs_ids_batch):
        """
        inputs_ids_batch: shape (batch_size, seq_len)
        We'll reuse attention_mask expanded to match batch_size.
        """
        batch_size, seq_len = inputs_ids_batch.shape
        # Expand the original attention_mask or build a similarly shaped mask
        # if you want to handle baseline_mask as well. For a single example,
        # we can just broadcast:
        attn_mask_expanded = attention_mask.expand(batch_size, seq_len)

        outputs = model(input_ids=inputs_ids_batch, attention_mask=attn_mask_expanded)
        logits = outputs.logits  # shape (batch_size, num_labels)
        class_logit = logits[:, 1]  # interpret class index 1 = "positive" class
        return class_logit

    # 4) Create Integrated Gradients
    ig = IntegratedGradients(captum_forward)

    # 5) Compute attributions
    attributions = ig.attribute(
        inputs=input_ids,            # shape: (1, seq_len)
        baselines=baseline_ids,      # must match shape
        n_steps=n_steps,
        internal_batch_size=1,
    )
    # shape => (batch=1, seq_len)

    attributions = attributions[0].detach().cpu().numpy()  # shape (seq_len,)

    # Convert input_ids to tokens
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    return attributions, tokens


def compute_ig_attributions_with_lig(
    model,
    tokenizer,
    sequence,
    device,
    baseline_seq=None,
    n_steps=50,
    max_length=512
):
    from captum.attr import LayerIntegratedGradients  # pip install captum

    # 1) Tokenize actual input
    inputs = tokenizer(sequence, return_tensors="pt", padding="max_length", truncation=True, max_length=max_length)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    # 2) Build baseline
    if baseline_seq is None:
        baseline_seq = "A" * len(sequence)
    baseline_inputs = tokenizer(baseline_seq, return_tensors="pt", padding="max_length", truncation=True, max_length=max_length)
    baseline_ids = baseline_inputs["input_ids"].to(device)

    # 3) Locate DNABERT embedding layer
    # If it's a standard BERT architecture, might be something like model.bert.embeddings.word_embeddings
    # For DNABERT:
    embedding_layer = model.bert.embeddings.word_embeddings  # or check model's code

    # 4) Create a wrapper forward function capturing your model
    def lig_forward(ids_batch):
        # ids_batch shape => (batch_size, seq_len)
        # we also pass attention_mask expanded similarly
        batch_size, seq_len = ids_batch.shape
        att_mask_expanded = attention_mask.expand(batch_size, seq_len)
        return forward_fn(ids_batch, att_mask_expanded, model)

    lig = LayerIntegratedGradients(lig_forward, embedding_layer)

    # 5) Compute attributions
    # We pass (input_ids, baseline_ids) for the embedding input, plus additional_forward_args for the model
    attributions = lig.attribute(
        inputs=input_ids,
        baselines=baseline_ids,
        n_steps=n_steps,
    )
    # shape => (batch=1, seq_len, embedding_dim)

    # sum across embedding dim
    attributions = attributions.sum(dim=-1)  # => (1, seq_len)
    attributions = attributions[0].detach().cpu().numpy()  # => (seq_len,)

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    return attributions, tokens


def compute_shap_attributions_with_deepshap(
    model,
    tokenizer,
    sequence,
    device,
    baseline_seq=None,
    max_length=512,
    target_class=1,
    batch_size=1,
    verbose=True
):
    """
    Compute per-token SHAP (DeepSHAP) attributions for a single DNA sequence input.

    This function parallels your `compute_ig_attributions_with_lig(...)` logic,
    but uses DeepSHAP from the `shap` library instead of Captum's LayerIntegratedGradients.

    Parameters
    ----------
    model : nn.Module (e.g., a huggingface AutoModelForSequenceClassification)
        The fine-tuned model used for predictions (binary or multi-class).
    tokenizer : PreTrainedTokenizer
        The tokenizer for your model (e.g. DNABERT).
    sequence : str
        The raw DNA sequence to interpret.
    device : str
        "cuda" or "cpu".
    baseline_seq : str or None
        Baseline sequence to represent "neutral" input. If None, we use a simple "A..." or "N..." approach.
    max_length : int
        Max sequence length for tokenization.
    target_class : int
        Class index for which we want attributions (binary default = 1).
    batch_size : int
        If you want to handle multiple examples in a batch, adapt accordingly. 
        Here we show single example usage only.
    verbose : bool
        If True, prints debugging info.

    Returns
    -------
    shap_values : np.ndarray (shape: (seq_len,))
        The per-token SHAP values for the selected target_class.
    tokens : list of str
        The corresponding tokens from the tokenizer.

    Notes
    -----
    1) You may need to adapt your forward pass or reference baseline if your model 
       is multi-class or has special embeddings.
    2) If binary classification, `target_class=1` typically corresponds to the "positive" logit.
    3) The actual usage of DeepSHAP in `shap` can vary with version. This snippet illustrates 
       the typical pattern of passing a reference distribution.

    Example:
        shap_vals, tokens = compute_shap_attributions_with_deepshap(
            model, tokenizer, "ACGTAA...", device="cuda"
        )
        plot_alignment_with_ig_scores(
            original_sequence="ACGTAA...",
            attention_weights=shap_vals,
            tokenized_sequence=tokens,
            ...
        )
    """
    import shap
    import torch

    # 1) Tokenize the input
    inputs = tokenizer(
        sequence,
        return_tensors="pt",
        truncation=True,
        max_length=max_length
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    # 2) Create baseline if not provided
    #    e.g. all-"N" or "A" of same length as `sequence`
    if baseline_seq is None:
        baseline_seq = "N" * len(sequence)  # or "A"*len(sequence), etc.

    baseline_inputs = tokenizer(
        baseline_seq,
        return_tensors="pt",
        truncation=True,
        max_length=max_length
    )
    baseline_ids = baseline_inputs["input_ids"].to(device)
    baseline_mask = baseline_inputs["attention_mask"].to(device)

    # Possibly adapt if you want multiple baseline sequences for a reference distribution
    # For simplicity, we use single baseline here.

    # 3) Define a custom forward function for shap
    #    This function should return the logit for `target_class` 
    #    (if multi-class or binary).
    def forward_func(input_ids_batch, attention_mask_batch):
        outputs = model(input_ids=input_ids_batch, attention_mask=attention_mask_batch)
        logits = outputs.logits  # shape: (batch, num_labels)
        # pick the logit for target_class
        return logits[:, target_class]

    # We'll create a "reference dataset" for DeepSHAP (the baseline).
    # In shap, we typically pass a list of references. If we have just 1 baseline,
    # we might do something like:
    reference_input_ids = baseline_ids
    reference_attn_mask = baseline_mask

    # 4) Build the data tuple. Some versions of shap require passing multiple inputs
    #    if your model forward expects (input_ids, attention_mask).
    #    We'll wrap them into a single combined input.
    #    We'll store them as a tuple => DeepExplainer can handle multiple data inputs.
    # 
    # For example, the explainer expects something like:
    # "explanations = explainer.shap_values((test_input_ids, test_attention_mask))"
    # 
    # We'll create a small function to handle the forward pass with two inputs.
    # We'll define a partial that returns just the logit for target_class.

    # 5) Create DeepExplainer
    # If we pass a "reference" dataset, it must be the same shape as a batch. 
    # Typically, you might have multiple baseline sequences. Here we'll just expand dims to 2D.
    reference_input_ids = reference_input_ids.unsqueeze(0) if reference_input_ids.ndimension() < 2 else reference_input_ids
    reference_attn_mask = reference_attn_mask.unsqueeze(0) if reference_attn_mask.ndimension() < 2 else reference_attn_mask

    # shape => (1, seq_len)
    # We'll replicate if needed, but let's keep it simple for single baseline
    # Some shap versions allow a "data tuple" baseline like [ (ids, mask) ] repeated. 
    # We'll do:
    reference_data = (reference_input_ids, reference_attn_mask)

    # We'll define a small wrapper:
    class HFDeepExplainerWrapper(shap.explainers.deep.DeepExplainer):
        def forward(self, inputs):
            """
            Expects inputs as a tuple: (input_ids, attention_mask)
            Returns forward_func(...) results. 
            """
            (i_ids, i_mask) = inputs
            return forward_func(i_ids, i_mask)

    # Initialize the wrapper explainer:
    explainer = HFDeepExplainerWrapper(
        model=forward_func,  # or we can pass None, but see usage
        data=reference_data
    )

    # 6) Prepare test_data for our single sequence
    # shape => (1, seq_len) for input_ids, etc.
    test_input_ids = input_ids
    test_attention_mask = attention_mask
    test_data = (test_input_ids, test_attention_mask)

    # 7) Compute shap values
    # shap_values_batch => might be shape (batch, seq_len)
    shap_values_batch = explainer.shap_values(test_data)

    # The result is typically a numpy array or list, depending on shap version.
    # We'll assume we get e.g. (1, seq_len)
    if isinstance(shap_values_batch, list):
        # Some versions might produce a list of arrays for each output dimension
        # If we're only returning 1 dimension, we might do:
        shap_values = shap_values_batch[0][0]
    else:
        # If it's just an array
        shap_values = shap_values_batch[0]

    # 8) Convert to CPU + numpy if needed
    if torch.is_tensor(shap_values):
        shap_values = shap_values.detach().cpu().numpy()

    # 9) Convert input IDs to tokens
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    if verbose:
        print(f"[DeepSHAP] shape of shap_values: {shap_values.shape}")
        print(f"[DeepSHAP] # tokens = {len(tokens)} => example tokens: {tokens[:10]}")

    return shap_values, tokens


####################################################################################################

# Generate synthetic DNA sequence dataset
def generate_synthetic_dataset_v0(num_samples=1000, sequence_length=200, col_label='label'):
    import random

    bases = ['A', 'T', 'C', 'G']
    synthetic_data = []

    for _ in range(num_samples):
        # Random DNA sequence
        sequence = ''.join(random.choices(bases, k=sequence_length))

        # Add specific motifs for FP or TP
        if random.random() < 0.5:
            label = 1
            sequence = sequence[:50] + "GTGAG" + sequence[55:]  # Insert a motif
        else:
            label = 0
            sequence = sequence[:50] + "AGGTA" + sequence[55:]  # Insert a motif

        synthetic_data.append({"sequence": sequence, col_label: label})

    return pd.DataFrame(synthetic_data)


# Introducing more complex inter-token relationships and realistic scenarios to encourage attention 
# across multiple tokens
def generate_synthetic_dataset(
    num_samples=1000,
    sequence_length=200,
    col_label="label",
    tp_motif="GTGAGT",  # Motif associated with TPs
    fp_motif="AGGTAG",  # Motif associated with FPs
    regulatory_motif="CACACA",  # Common regulatory motif
    inter_token_distance=5,  # Distance between motifs for interaction
    random_noise=True,  # Add random noise to sequences
    noise_probability=0.05,  # Probability of introducing random mutations
):
    """
    Generate a synthetic DNA sequence dataset with complex inter-token dependencies.

    Parameters:
    - num_samples: Number of samples to generate.
    - sequence_length: Length of each DNA sequence.
    - col_label: Column name for the class label.
    - tp_motif: Motif associated with TPs.
    - fp_motif: Motif associated with FPs.
    - regulatory_motif: Common regulatory motif associated with inter-token dependencies.
    - inter_token_distance: Distance between motifs to simulate dependencies.
    - random_noise: Whether to introduce random mutations in the sequence.
    - noise_probability: Probability of introducing mutations in the sequence.

    Returns:
    - pd.DataFrame: DataFrame containing the synthetic dataset.
    """
    bases = ["A", "T", "C", "G"]
    synthetic_data = []

    for _ in range(num_samples):
        # Start with a random sequence
        sequence = "".join(random.choices(bases, k=sequence_length))

        if random.random() < 0.5:
            # True Positive (TP)
            label = 1
            # Insert TP motif at a random position
            tp_position = random.randint(10, sequence_length - len(tp_motif) - inter_token_distance - 10)
            sequence = sequence[:tp_position] + tp_motif + sequence[tp_position + len(tp_motif):]

            # Add a regulatory motif near the TP motif to encourage inter-token dependencies
            regulatory_position = tp_position + len(tp_motif) + inter_token_distance
            if regulatory_position + len(regulatory_motif) <= sequence_length:
                sequence = sequence[:regulatory_position] + regulatory_motif + sequence[regulatory_position + len(regulatory_motif):]
        else:
            # False Positive (FP)
            label = 0
            # Insert FP motif at a random position
            fp_position = random.randint(10, sequence_length - len(fp_motif) - inter_token_distance - 10)
            sequence = sequence[:fp_position] + fp_motif + sequence[fp_position + len(fp_motif):]

            # Add a noisy or incomplete regulatory motif near the FP motif
            noisy_regulatory = "".join(random.choices(bases, k=len(regulatory_motif)))  # Randomized "regulatory" motif
            noisy_position = fp_position + len(fp_motif) + inter_token_distance
            if noisy_position + len(noisy_regulatory) <= sequence_length:
                sequence = sequence[:noisy_position] + noisy_regulatory + sequence[noisy_position + len(noisy_regulatory):]

        # Introduce random mutations in the sequence
        if random_noise:
            sequence = "".join(
                random.choice(bases) if random.random() < noise_probability else base
                for base in sequence
            )

        # Add the generated sequence and label to the dataset
        synthetic_data.append({"sequence": sequence, col_label: label})

    return pd.DataFrame(synthetic_data)


def merge_sequence_and_feature_datasets(
        feature_df, sequence_df, 
        pred_type='FP', 
        label_col='label', 
        pred_type_col='pred_type', 
        return_feature_set=False, return_subset_sequence_df=False):
    """
    Merge sequence and feature datasets, retaining only rows in feature_df that match with those in sequence_df.

    Parameters:
    - feature_df (pd.DataFrame or pl.DataFrame): DataFrame containing feature data.
    - sequence_df (pd.DataFrame or pl.DataFrame): DataFrame containing sequence data.
    - label_col (str): Column name for the label in feature_df (default is 'label').
    - pred_type_col (str): Column name for the prediction type in sequence_df (default is 'pred_type').

    Returns:
    - merged_df (pd.DataFrame or pl.DataFrame): Merged DataFrame.
    - feature_columns (list): List of feature columns.
    """
    from meta_spliceai.splice_engine.analysis_utils import classify_features

    is_polars = isinstance(sequence_df, pl.DataFrame)
    if is_polars:
        sequence_df = sequence_df.to_pandas()
    if isinstance(feature_df, pl.DataFrame):
        feature_df = feature_df.to_pandas()

    # Map the pred_type column in sequence_df to 0 and 1
    if label_col not in sequence_df.columns:
        # sequence_df[label_col] = sequence_df[pred_type_col].apply(lambda x: 1 if x == pred_type else 0)
        raise ValueError("The sequence dataset has not been labeled yet")
    if label_col not in feature_df.columns:
        raise ValueError("The feature dataset has not been labeled yet")

    # Test: prediction type column should have been used to label the dataset and dropped
    assert pred_type_col not in sequence_df.columns, "The prediction type column should have been dropped"
    assert pred_type_col not in feature_df.columns, "The prediction type column should have been dropped"

    # Test: Unique values of the prediction type column
    print(f"[test] Unique values in the prediction type column in sequence_df: {sequence_df[label_col].unique()}")
    print(f"[test] Unique values in the prediction type column in feature_df: {feature_df[label_col].unique()}")

    # Merge the DataFrames on the primary keys
    merged_df = pd.merge(
        feature_df,
        sequence_df,
        on=['gene_id', 'transcript_id', 'position', 'splice_type', ],
        how='inner',
        suffixes=('', '_dup'), 
        indicator=True
    )

    # Extract the subset of sequence_df that was taken
    subset_sequence_df = merged_df[merged_df['_merge'] == 'both'].drop(columns=['_merge'])

    # Drop the indicator column from the merged DataFrame
    merged_df = merged_df.drop(columns=['_merge'])

    # Verify that the labeling between the two data sources is consistent
    assert merged_df[label_col].equals(merged_df[f'{label_col}_dup']), "Inconsistent labels between sequence and feature datasets"
    # Log: ok

    # Drop duplicate columns with suffix '_dup'
    merged_df = merged_df.loc[:, ~merged_df.columns.str.endswith('_dup')]

    # Convert back to Polars if the input was Polars
    if is_polars:
        merged_df = pl.DataFrame(merged_df)

    if return_feature_set: 
        # Classify features to identify feature columns
        feature_categories = classify_features(merged_df, label_col=label_col)
        feature_columns = (
            feature_categories['categorical_features'] +
            feature_categories['numerical_features'] +
            feature_categories['derived_categorical_features'] + 
            ['sequence', ]
        )
        return merged_df, feature_columns

    if return_subset_sequence_df:
        return merged_df, subset_sequence_df

    return merged_df


####################################################################################################


def augmented_analysis_sequence_dataset(experiment='hard_genes', pred_type='FP', **kargs): 
    from meta_spliceai.splice_engine.train_error_model import load_error_classifier_dataset
    from meta_spliceai.splice_engine.analysis_utils import (
        classify_features, 
        count_unique_ids,
        subset_non_motif_features, 
        label_analysis_dataset
    )
    # from meta_spliceai.splice_engine.seq_model_utils import (
    #     verify_sequence_lengths, 
    # )

    error_label = kargs.get("error_label", pred_type)
    correct_label = kargs.get("correct_label", "TP")

    col_label = kargs.get('col_label', 'label')
    col_tid = kargs.get('col_tid', 'transcript_id')
    verbose = kargs.get('verbose', 1)
    # subject = kargs.get('subject', "analysis_sequences")
    
    facet = kargs.get("facet", "simple")
    subject = f"featurized_dataset_{facet}"

    print_emphasized("[info] Loading error classifier dataset ...")
    df_trainset = load_error_classifier_dataset(
        pred_type=pred_type, 
        error_label=error_label, correct_label=correct_label, 
        verbose=1)
    print_with_indent(f"Columns: {list(df_trainset.columns)}", indent_level=1)

    if col_label not in df_trainset.columns:
        # Training data has not been labeled yet
        df_trainset = label_analysis_dataset(df_trainset, positive_class=error_label)

    # feature_categories = classify_features(df, label_col=col_label, verbose=verbose)
    print_emphasized("[info] Analyzing feature categories and take non-motif features ...")
    df_non_motif = subset_non_motif_features(df_trainset, label_col=col_label, max_unique_for_categorical=10)
    print_with_indent(f"Columns: {list(df_non_motif.columns)}", indent_level=1)

    result = count_unique_ids(df_non_motif, col_gid='gene_id', col_tid=col_tid, verbose=1, return_ids=False)
    n_genes = result['num_unique_genes']
    n_trpts = result['num_unique_transcripts']
    print_with_indent(f"[info] Number of unique gene IDs in featurized dataset: {result['num_unique_genes']}", indent_level=2)
    print_with_indent(f"[info] Number of unique transcript IDs in featurized dataset: {result['num_unique_transcripts']}", indent_level=2)

    # ---- Load the analysis sequence dataset ----

    print_emphasized("[info] Loading analysis sequence dataset ...")
    mefd = ModelEvaluationFileHandler(ErrorAnalyzer.eval_dir, separator='\t')
    # splice_pos_df = mefd.load_splice_positions(aggregated=True)

    subject = f"analysis_sequences_{facet}"
    # NOTE: Use run_spliceai_workflow.error_analysis_workflow() to generate the error analysis data 
    analysis_sequence_df = \
        mefd.load_analysis_sequences(
            aggregated=True, subject=subject, 
            error_label=error_label, correct_label=correct_label
        )  # Output is a Polars DataFrame
    
    # Label the analysis sequence dataset
    analysis_sequence_df = label_analysis_dataset(analysis_sequence_df, positive_class=error_label)

    # Verify sequence lengths
    verify_sequence_lengths(analysis_sequence_df, sequence_col='sequence')

    result = count_unique_ids(analysis_sequence_df, col_gid='gene_id', col_tid=col_tid, verbose=1, return_ids=False)
    # n_genes = result['num_unique_genes']
    # n_trpts = result['num_unique_transcripts']

    print_with_indent(f"[info] Columns: {list(analysis_sequence_df.columns)}", indent_level=1)
    print_with_indent(f"[info] Shape of the analysis sequence DataFrame: {analysis_sequence_df.shape}", indent_level=2)
    print_with_indent(f"[info] Number of unique gene IDs: {result['num_unique_genes']}", indent_level=2)
    print_with_indent(f"[info] Number of unique transcript IDs: {result['num_unique_transcripts']}", indent_level=2)
    
    # NOTE: Total set with Ensembl: (2,915,508, 11) ~ 3M rows, 11 columns

    # Merge the sequence and feature datasets
    # Precondition: Both datasets have been labeled
    assert col_label in df_non_motif.columns, "The feature dataset has not been labeled yet"
    assert col_label in analysis_sequence_df.columns, "The analysis sequence dataset has not been labeled yet"

    print_emphasized("[info] Merging sequence and feature datasets ...")
    augmented_sequence_df = \
        merge_sequence_and_feature_datasets(df_non_motif, analysis_sequence_df, label_col='label', pred_type_col='pred_type')

    result = count_unique_ids(augmented_sequence_df, col_gid='gene_id', col_tid=col_tid, verbose=1, return_ids=False)
    # assert result['num_unique_genes'] == n_genes, f"Expected {n_genes} unique genes but found {result['num_unique_genes']}"
    
    print_with_indent(f"[info] Columns: {list(augmented_sequence_df.columns)}", indent_level=1)
    print_with_indent(f"[info] Shape of the merged DataFrame: {augmented_sequence_df.shape}", indent_level=2)
    print_with_indent(f"[info] Number of unique gene IDs: {result['num_unique_genes']}", indent_level=2)
    print_with_indent(f"[info] Number of unique transcript IDs: {result['num_unique_transcripts']}", indent_level=2)

    subject = f"augmented_sequences_{facet}"
    if kargs.get('save', True):
        # augmented_sequences_path = \
        #         mefd.save_analysis_sequences(
        #             augmented_sequence_df, 
        #             aggregated=True, 
        #             subject="augmented_sequences"
        #         )
        featurized_sequence_path = \
            mefd.save_featurized_dataset(
                augmented_sequence_df, 
                aggregated=True, 
                subject=subject,
                error_label=error_label, correct_label=correct_label,
            )

        print_emphasized(f"[info] Saved final augmented sequence dataset to {featurized_sequence_path}")
        print_with_indent(f"Columns: {list(augmented_sequence_df.columns)}", indent_level=1)
        shape0 = augmented_sequence_df.shape

        # Test
        augmented_sequence_df_prime = \
            mefd.load_featurized_dataset(
                aggregated=True, 
                subject=subject,
                error_label=error_label, correct_label=correct_label,
            )
        assert augmented_sequence_df_prime.shape == augmented_sequence_df.shape, "Shape mismatch detected."
        # analysis_result = \
        #     analyze_data_labels(df_trainset_prime, label_col='label', verbose=2, handle_missing=None)

    return augmented_sequence_df


def demo_visualize_attention_weights():

    # Example DNA sequence and motifs
    sequence = "ACGTACGTACGTACGTACGTACGTACGT"
    important_motifs = ["ACGT", "TACG"]

    # Step 1: Find motif positions
    motif_positions = find_motif_positions(sequence, important_motifs)

    # Step 2: Extract attention weights
    # Assuming `model` is a fine-tuned transformer model and `tokenizer` is its tokenizer
    attention_weights = get_attention_weights(
        model, tokenizer, sequence, layer=0, head=0)

    # Step 3: Visualize motifs and attention
    visualize_motifs_and_attention(sequence, motif_positions, attention_weights)


def demo_visualize_attention_weights_filtered():
    # Example DNA sequence and motifs
    sequence = "ACGTACGTACGTACGTACGTACGTACGT"
    important_motifs = ["ACGT", "TACG"]

    # Step 1: Identify Motif Positions
    motif_positions = find_motif_positions(sequence, important_motifs)

    # Step 2: Extract Attention Weights
    attention_weights = get_attention_weights(model, tokenizer, sequence, layer=0, head=0)

    # Step 3: Filter and Refine Motifs
    filtered_motifs = filter_motifs_by_attention(sequence, motif_positions, attention_weights, threshold_percentile=90)
    grouped_motifs = group_overlapping_motifs(filtered_motifs)
    top_motifs = select_top_motifs(grouped_motifs, top_n=10)

    # Step 4: Visualize
    visualize_motifs_and_attention(sequence, top_motifs, attention_weights)


def demo_process_training_dataset(**kargs): 
    from meta_spliceai.splice_engine.splice_error_analyzer import (
        ErrorAnalyzer, 
        make_error_sequence_model_dataset
    )
    from meta_spliceai.splice_engine.seq_model_utils import (
        verify_sequence_lengths, 
        trim_contextual_sequences
    )
    from meta_spliceai.splice_engine.analysis_utils import (
        analyze_data_labels, 
        count_unique_ids, 
        subset_training_data, 
        subsample_genes, 
        show_predicted_splice_sites, 
        filter_analysis_dataframe
    )
    from meta_spliceai.splice_engine.utils_df import (
        estimate_dataframe_size, 
        estimate_memory_usage
    )

    print_emphasized("Demo: Load Training Dataset")
    
    # Prelinimary steps:
    # Use splice_engine.run_spliceai_workflow.error_analysis_workflow() to generate the error analysis data
    # Use splice_engine.splice_error_analyzer.demo_make_training_set() to generate featurized dataset
    
    overwrite = kargs.get('overwrite', True)
    col_tid = kargs.get('col_tid', 'transcript_id')
    verbose = kargs.get('verbose', 1)

    experiment = kargs.get('experiment', 'hard_genes')
    facet = kargs.get("facet", "simple")  #'trimmed', "simple"  
    # 'simple': subsampled and trimmed
    subject = f"featurized_dataset_{facet}"

    mefd = ModelEvaluationFileHandler(ErrorAnalyzer.eval_dir, separator='\t')
    pred_type = kargs.get('pred_type', 'FP')
    error_label = kargs.get('error_label', pred_type)
    correct_label = kargs.get('correct_label', "TP")

    col_label = kargs.get('col_label', 'label')
    # n_samples = kargs.get('n_samples', 1000)
    n_genes = kargs.get('n_genes', 100)
    target_length = kargs.get('target_length', 200)  # Used to trim the contextual sequences (prior: 1000nt)

    input_path = mefd.get_featurized_dataset_path(
        aggregated=True, subject=subject, 
        error_label=error_label, correct_label=correct_label)

    if overwrite or not os.path.exists(input_path): 
        print_emphasized(f"[info] Loading the full featurized dataset given pred_type={pred_type} ...")
        df_trainset = \
            mefd.load_featurized_dataset(
                aggregated=True, 
                error_label=error_label, correct_label=correct_label)
        # subject set to the default value associated with full dataset
        
        analysis_result = \
            analyze_data_labels(df_trainset, label_col=col_label, verbose=2, handle_missing=None)
        print("[info] type(df_trainset): {}".format(type(df_trainset)))

        result_set = count_unique_ids(df_trainset, col_tid=col_tid, verbose=verbose, return_ids=True)

        Ng0 = result_set['num_unique_genes']
        Nt0 = result_set['num_unique_transcripts']
        gene_ids = result_set['unique_gene_ids']
        tx_ids = result_set['unique_transcript_ids']
        print_with_indent(f"Number of unique genes: {Ng0}", indent_level=1)
        print_with_indent(f"Number of unique transcripts: {Nt0}", indent_level=1)

        df_subset = df_trainset
        subsampling_applied = False

        if n_genes is not None and (n_genes < Ng0): 
            print_emphasized(f"[action] Subsampling the training dataset to n_genes={n_genes} ...")

            # df_subset = subset_training_data(df_trainset, group_cols=['gene_id', ], target_sample_size=n_samples, verbose=1)
            df_subset = subsample_genes(df_trainset,  N=n_genes)
            
            result_set = count_unique_ids(df_subset, col_tid=col_tid, verbose=verbose, return_ids=True)   

            Ng1 = result_set['num_unique_genes']
            Nt1 = result_set['num_unique_transcripts']
            gene_ids = result_set['unique_gene_ids']
            tx_ids = result_set['unique_transcript_ids']     

            print_with_indent(f"Number of unique genes after subsampling: {Ng0} -> {Ng1}", indent_level=1)
            print_with_indent(f"Number of unique transcripts after subsampling: {Nt0} -> {Nt1}", indent_level=1)
            print_with_indent(f"Size of the training dataset after subsampling: {df_trainset.shape[0]} -> {df_subset.shape[0]}", indent_level=1)    

            memory_usage_mb = estimate_memory_usage(df_subset)
            print_with_indent(f"Estimated memory usage of df_subset: {memory_usage_mb:.2f} MB", indent_level=1)

            estimated_size_mb = estimate_dataframe_size(df_subset, file_format='tsv')
            print_with_indent(f"Estimated disk size of df_subset: {estimated_size_mb:.2f} MB", indent_level=1)

            subject = f"featurized_dataset_{facet}"  # experimental dataset with subsampled genes
            featurized_dataset_path = \
                mefd.save_featurized_dataset(
                    df_subset, 
                    aggregated=True, 
                    subject=subject, 
                    error_label=error_label, correct_label=correct_label
                ) 
            print_with_indent(f"[info] Saved the featurized dataset to: {featurized_dataset_path}", indent_level=1)

            subsampling_applied = True
        # ------------------------------------------------
        # Based on the featurized dataset, we can now subset the analysis sequences to match the training dataset

        print_emphasized("[info] Loading the analysis sequence dataset ...")
        analysis_sequence_df = \
            mefd.load_analysis_sequences(
                aggregated=True, 
                error_label=error_label, correct_label=correct_label)  # Output is a Polars DataFrame
        print_with_indent(f"[info] Shape: {analysis_sequence_df.shape}", indent_level=1)
        print_with_indent(f"[info] Columns: {list(analysis_sequence_df.columns)}", indent_level=1)
        verify_sequence_lengths(analysis_sequence_df, sequence_col='sequence')

        # print_emphasized(f"[action] Subsetting the analysis sequence dataset based on prediction type: {pred_type} ...")
        # analysis_sequence_df = make_error_sequence_model_dataset(analysis_sequence_df, pred_type=pred_type)        
        
        print_emphasized("[action] Subsetting the analysis sequence dataset ...")

        if subsampling_applied:
            # Subset the DataFrame
            # Take sequences associated with gene IDs and tx IDs found in the featurized dataset
            analysis_subset_df = analysis_sequence_df.filter(
                (pl.col('gene_id').is_in(gene_ids)) & 
                (pl.col('transcript_id').is_in(tx_ids))
            )
            print_with_indent(f"[info] Shape (after subsetting): {analysis_subset_df.shape}")

            # Test: Verify the number of unique genes/transcripts
            result_set = count_unique_ids(analysis_subset_df, col_tid=col_tid, verbose=verbose, return_ids=True)
            Ng2 = result_set['num_unique_genes']
            Nt2 = result_set['num_unique_transcripts']
            if not (Ng1 == Ng2 and Nt1 >= Nt2): 
                print(f"Number of unique genes mismach? {Ng1} =?= {Ng2}, transcripts mismatch? {Nt1} =?= {Nt2}")
            assert Ng1 == Ng2 and Nt1 >= Nt2, \
                f"Number of unique genes/transcripts mismatch: {Ng1} != {Ng2} or {Nt1} != {Nt2}"
        else:
            analysis_subset_df = analysis_sequence_df

        print_emphasized(f"[action] Trimming the contextual sequences to target length={target_length} ...")
        analysis_subset_df = trim_contextual_sequences(analysis_subset_df, target_length=target_length)

        # Test: Verify sequence lengths
        verify_sequence_lengths(analysis_subset_df, sequence_col='sequence')

        # Verify labels 
        analyze_data_labels(analysis_subset_df, label_col='pred_type', verbose=2, handle_missing=None)

        subject = f"analysis_sequences_{facet}"
        analysis_subset_path = \
            mefd.save_analysis_sequences(
                analysis_subset_df, 
                aggregated=True, 
                subject=subject, 
                error_label=error_label, correct_label=correct_label
            )  
        print_with_indent(f"[info] Saved sampled analysis sequences to: {analysis_subset_path}", indent_level=1)

        # Test: Show the subsampled dataset
        focused_pred_types = [error_label, correct_label]
        print_emphasized(f"[test] Show predicted splice site profile in the subsampled dataset  ...")
        print_with_indent(f"[test] Focused prediction types: {focused_pred_types}", indent_level=1)
        analysis_subset_df_prime = \
            filter_analysis_dataframe(
                df=mefd.load_analysis_sequences(aggregated=True),
                gene_ids=gene_ids, 
                transcript_ids=tx_ids, verbose=0)
        show_predicted_splice_sites(
            analysis_subset_df_prime,
            focused_pred_types=focused_pred_types,
            num_transcripts=10)

        subject = f"augmented_sequences_{facet}"
        print_emphasized(f"[action] Create augmented sequence dataset with facet={facet} ...")
        # augmented_analysis_sequence_dataset(experiment=experiment, facet=facet, pred_type=pred_type, save=True)


def demo_retrieve_motif_importance(pred_type='FP'):

    use_mock = False

    if use_mock: 
        df_motif = pd.DataFrame({
            'motif': ['3mer_TTT', '3mer_AGA', '3mer_TCT', '2mer_GT', '3mer_GTT'],
            'importance_score': [0.195, 0.169, 0.163, 0.161, 0.152]
       })
    else: 
        analyzer = ErrorAnalyzer(experiment='hard_genes')
        df_motif = analyzer.load_motif_importance(pred_type=pred_type)
        
    display_dataframe_in_chunks(df_motif, num_rows=df_motif.shape[0])

    motifs = prepare_motifs_for_visualization(df_motif)
    print(motifs)

    motifs_with_scores = prepare_motifs_for_visualization(df_motif, return_scores=True)
    print(motifs_with_scores)


def demo_pretrain_finetune(**kargs): 
    """
    Pretrain and fine-tune a transformer model for splice error prediction using DNABERT.

    Memo: 
    - GPU Support: 
       The model and input tensors (input_ids, attention_mask) are moved to the appropriate device (cuda or cpu).
       Model inference (outputs) is performed on the selected device.
       Default device is dynamically set using torch.device.
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
    from transformers.models.bert import BertForSequenceClassification
    from meta_spliceai.splice_engine.splice_error_analyzer import ErrorAnalyzer
    from meta_spliceai.splice_engine.utils_df import (
        estimate_dataframe_size, 
        estimate_memory_usage
    )
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report

    do_fine_tuning = True
    do_manual_evaluation = False
    device, local_rank = set_device()

    tokenizer_name = "zhihan1996/DNABERT-2-117M"
    model_name = "zhihan1996/DNABERT-2-117M"
    model_type = 'DNABERT'

    facet = kargs.get("facet", "simple")    # 'trimmed',  "simple" (subsampled + trimmed)
    subject = f"analysis_sequences_{facet}"
    pred_type = kargs.get('pred_type', 'FP')
    error_label = kargs.get('error_label', pred_type)
    correct_label = kargs.get('correct_label', "TP")

    pred_type_to_label = {error_label: 1, correct_label: 0}
    gene_level_split = False
    # {k: v for k, v in ErrorAnalyzer.pred_type_to_label if k in [pred_type, 'TP']}

    experiment_name = kargs.get('experiment', 'hard_genes')
    
    max_length = 1000  # Maximum sequence length for tokenization   
    col_tid = 'transcript_id'
    col_label = 'label'

    analyzer  = ErrorAnalyzer(experiment=experiment_name, model_type=model_type.lower())
    # error_analysis_type = f"{pred_type}_vs_TP".lower()
    output_dir = analyzer.set_analysis_output_dir(error_label=error_label, correct_label=correct_label)
    os.makedirs(output_dir, exist_ok=True)
    print(f"[info] Output directory set to: {output_dir}")

    logging_dir = os.path.join(output_dir, "logs")
    os.makedirs(logging_dir, exist_ok=True)
    print(f"[info] Logging directory set to: {logging_dir}")

    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"[info] Checkpoint directory set to: {checkpoint_dir}")

    mefd = ModelEvaluationFileHandler(ErrorAnalyzer.eval_dir, separator='\t')
    # splice_pos_df = mefd.load_splice_positions(aggregated=True)

    # Use run_spliceai_workflow.error_analysis_workflow() to generate the error analysis data
    print_emphasized("[info] Loading the analysis sequence dataset ...")

    # Loading the analysis sequence dataset which contains sequences and their associated labels
    analysis_sequence_df = mefd.load_analysis_sequences(
        aggregated=True, subject=subject, 
        error_label=error_label, correct_label=correct_label
    )  # Output is a Polars DataFrame
    # analysis_sequence_df has the following columns: 
    # ['gene_id', 'transcript_id', 'chrom', 'position', 'pred_type', 'score', 'splice_type', 'strand', 
    #  'window_end', 'window_start', 'sequence']

    # Label the dataset based on the prediction type
    analysis_sequence_df = label_analysis_dataset(analysis_sequence_df, positive_class=error_label)
    # Shape of the DataFrame
    print_with_indent(f"[info] Shape of the analysis sequence DataFrame: {analysis_sequence_df.shape}", indent_level=1)
    # NOTE: Total set with Ensembl: (2,915,508, 11) ~ 3M rows, 11 columns

    # Verify the label classes
    verify_label_classes(analysis_sequence_df, label_col=col_label)

    # Analyze the data labels
    analyze_data_labels(analysis_sequence_df, label_col=col_label, verbose=2, handle_missing=None)

    # Verify sequence lengths
    max_length_prime = verify_sequence_lengths(analysis_sequence_df, sequence_col='sequence')
    if max_length_prime < max_length:
        print(f"[info] Maximum sequence length: {max_length_prime} < requested {max_length}")
        max_length = max_length_prime

    # Optionally trim the sequence to a maximum length
    # analysis_sequence_df = trim_contextual_sequences(analysis_sequence_df, target_length=200)

    print_emphasized("[info] Preparing the training and validation datasets ...")
  
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, model_max_length=max_length)
    train_dataset, val_dataset, train_df, val_df = \
        prepare_transformer_datasets(
            training_df=analysis_sequence_df,
            
            # Tokenizer parameteres
            # tokenizer_name=tokenizer_name, 
            # max_length=max_length,  # Adjust max length for longer sequences
            tokenizer=tokenizer,
            
            test_size=0.2, 
            return_dataframes=True,
            dataset_format="huggingface",  # Change to "pytorch" for PyTorch Datasets
            gene_level_split=gene_level_split
        )

    ###### Training the Transformer Model ######

    print_emphasized("[info] Training the transformer model ...")
    
    batch_size = kargs.get('batch_size', 8)
    num_train_epochs = kargs.get('num_train_epochs', 10)

    model_id = f"{experiment_name.lower()}_{model_type.lower()}_{pred_type}"
    model_path = os.path.join(output_dir, model_id)
    model = trainer = None
    deepspeed_config = os.path.join(ErrorAnalyzer.analysis_dir, "deepspeed_config.json")
    print_with_indent(f"[info] DeepSpeed configuration file: {deepspeed_config}", indent_level=1)

    print_emphasized("[info] Configuring the training parameters ...")
    training_arguments = {
        "output_dir": output_dir,

        "eval_strategy": 'epoch', # Evaluation strategy ('steps' or 'epoch')
        "save_strategy": 'epoch', # Save strategy ('steps' or 'epoch')
        # --load_best_model_at_end requires the save and eval strategy to match

        "save_steps": 0,  # Save checkpoint every `save_steps` steps; 0 disables intermediate checkpoints

        "per_device_train_batch_size": batch_size,  # Batch size per device (GPU or CPU)
        # per_device_train_batch_size=2,  # Reduce batch size for training
        # per_device_eval_batch_size=2,  # Reduce batch size for evaluation

        "num_train_epochs": 1,  # Handle epochs manually
        "weight_decay": 0.01,  # L2 regularization coefficient
        "learning_rate": 5e-5,
        "warmup_steps": 500,  # Warmup steps for learning rate scheduler

        "gradient_accumulation_steps": 4,  # Accumulate gradients over N steps

        "load_best_model_at_end": True, # Automatically load the best model at the end
        "metric_for_best_model": "eval_loss",  # Specify metric for identifying the best model (e.g., "eval_loss" or "eval_f1")
        # "lr_scheduler_type": "reduce_lr_on_plateau",  # Corrected scheduler type
        
        "logging_dir": logging_dir,  # Directory for logging
        "logging_steps": 1000,  # Log metrics every `logging_steps` steps

        "ddp_find_unused_parameters": False,  # Optimize DDP for unused parameters
        "remove_unused_columns": False,  # Remove unused columns from the dataset
        "dataloader_drop_last": True,  # Drop the last incomplete batch
        "local_rank": local_rank,  # Automatically detected by torchrun
        # When local_rank is passed to TrainingArguments, the Trainer ensures the model is placed on the appropriate device 
        # and uses torch.nn.parallel.DistributedDataParallel (DDP) for distributed training.

        # fp16=True,  # Mixed precision for memory efficiency
        # NOTE: Use fp16=True on GPUs with Tensor Core support, such as NVIDIA V100, A100, or 
        #       RTX 20/30/40 series GPUs.
        
        # "gradient_checkpointing": True,   # Enable gradient checkpointing
        # "deepspeed": deepspeed_config,  # Optional: Use DeepSpeed
        # "eval_accumulation_steps": 16,  # or higher for memory efficiency

        # "prediction_loss_only": True,  # Turn off Full-Logits Storage
        # This only tells the Trainer not to store and concatenate all the model logits during evaluation
    }
    
    model, trainer, model_path = \
        train_transformer_model(
            train_dataset=train_dataset,  # Tokenized training dataset
            val_dataset=val_dataset,  # Tokenized validation dataset
            training_arguments=training_arguments,  # Training arguments
            
            model_name=model_name,  # Pre-trained DNABERT
            # model_class=BertForSequenceClassification,  # If specified, use_auto_model=False
            
            # tokenizer=tokenizer,  # Tokenizer for the model (not used)
            # tokenizer is unecessary because train_dataset and val_dataset are pre-tokenized

            use_auto_model=True,  # Set to True to use AutoModelForSequenceClassification
            
            num_labels=2,  # Binary classification (FP/FN vs. TP)
            num_train_epochs=num_train_epochs,  # Number of training epochs for "manual" training (not to be passed to Trainer)
            model_id=model_id,  # Model ID for saving
            device=device,  # Device (cuda or cpu)
            return_path_to_model = True,
            train_model=do_fine_tuning,
        )

    ###### Pre-Model Evaluation ######

    # (Test) Verify model consistency
    print_emphasized("[INFO] Model Configuration:")
    print(model.config)
    # for name, param in model.named_parameters():
    #     print(f"Parameter {name}, Shape: {param.shape}")

    # Evaluate the model and plot metrics (e.g. ROC-AUC, PR-AUC curves)
    print_emphasized("[INFO] Evaluating the model on the validation set ...")
    validation_report = \
        evaluate_and_plot_metrics(trainer, val_dataset, output_dir, label_names=["TP", pred_type])
    print_with_indent(f"[INFO] Validation Report: {validation_report}\n", indent_level=1)

    # Choose top motifs for visualization
    print_emphasized("[info] Retrieving important motifs ...")
    # analyzer = ErrorAnalyzer(experiment=experiment_name)
    df_motif = \
        analyzer.load_motif_importance(
            error_label=error_label, correct_label=correct_label, 
            model_type='xgboost')  # Retrieve important motifs derived from xgboost workflow
    
    display_dataframe_in_chunks(df_motif, num_rows=df_motif.shape[0])

    print_with_indent("[info] Processing sequences for motif visualization ...", indent_level=1)
    motifs_with_scores = \
        prepare_motifs_for_visualization(df_motif, return_scores=True)

    print_with_indent(f"Motifs with scores: {motifs_with_scores}", indent_level=2)
    motifs = [motif for motif, _ in motifs_with_scores]
    print_with_indent(f"Motifs: {motifs}", indent_level=3)
    
    important_motifs = motifs[:5]

    # Subsample the validation DataFrame
    print_emphasized("[info] Subset validation dataframe for visualization ...")
    val_policy = "subsample"

    if val_policy == "subsample":
        # Subsample the validation DataFrame
        subsampled_val_df = subsample_genes(analysis_sequence_df, gene_col='gene_id', N=10)
    else: 
        subsampled_val_df = find_transcripts_with_both_labels(analysis_sequence_df, num_transcripts=10)
        # subsampled_val_df = val_df.filter(pl.col('transcript_id').is_in(tx_set.keys()))

        # Find the number of unique transcripts with both labels
        num_trpts_with_both_labels = subsampled_val_df['transcript_id'].nunique()
        print(f"[info] Number of transcripts with both labels: {num_trpts_with_both_labels}")

    if isinstance(subsampled_val_df, pl.DataFrame):
        subsampled_val_df = subsampled_val_df.to_pandas()
    
    ### Model Inference and Visualization ###

    # After training is complete
    model.eval()  # put the model in inference mode, outside all loops
    # model.eval() changes layers to evaluation behavior (disabling dropout, 
    # freezing batchnorm running means, etc.).

    # Dictionary that maps labels back to prediction types
    label_to_pred_type = {v: k for k, v in pred_type_to_label.items()}

    # Loop through the sequence column and track progress using tqdm
    # for i, row in tqdm(subsampled_val_df.iterrows(), total=subsampled_val_df.shape[0], desc="Processing validation sequences"):
    
    # Loop through the genes and track progress using tqdm
    grouped = subsampled_val_df.groupby('gene_id')  # Group by gene_id
    for gene_id, group in tqdm(grouped, total=len(grouped), desc="Processing validation sequences by gene"):
        # Create a new directory for each gene
        gene_dir = os.path.join(output_dir, gene_id)
        os.makedirs(gene_dir, exist_ok=True)
        
        for i, row in group.iterrows():
            sequence = row['sequence']
            gene_id = row['gene_id']
            tx_id = row[col_tid]
            label = row[col_label]

            print("[info] label -> pred_type: ", label_to_pred_type)

            annotation = row.to_dict()
            # keys: ['gene_id', 'transcript_id', 'chrom', 'position', 'label', 
            #        'score', 'splice_type', 'strand', 'window_end', 'window_start', 'sequence']
            annotation['pred_type'] = label_to_pred_type[label]
            splice_type = annotation['splice_type']
            position = annotation['position']
            
            print_with_indent(f"Processing sequence {i + 1}/{len(val_df['sequence'])}...", indent_level=1)
            print_with_indent(f"Gene ID: {gene_id}, Transcript ID: {tx_id}", indent_level=2)

            encoded_sequence = tokenizer(
                sequence,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            )

            input_ids = encoded_sequence["input_ids"].to(device)
            attention_mask = encoded_sequence["attention_mask"].to(device)

            # Get prediction for the example
            with torch.no_grad(): 
                output = model(input_ids, attention_mask=attention_mask)
                prediction = torch.argmax(output.logits).item()
            # This ensures no gradients are computed/stored at runtime (disabling autograd graph creation).

            # Simplify the sequence for print messages
            seq = abridge_sequence(sequence, max_length=100)
            print_with_indent(f"Sequence: {seq}")
            print_with_indent(f"Predicted class: {pred_type if prediction == 1 else 'TP'}")

            # Retrieve attention weights 
            attention_weights, tokenized_sequence = \
                get_attention_weights(
                    model=model,
                    tokenizer=tokenizer,
                    sequence=sequence,
                    max_length=max_length,
                    # layer=None,  # Use all layers for rollout
                    # head=None,  # Use all heads for rollout
                    # policy="rollout"
                    layer=0,  # Use a layer
                    head=2,  # Use a head
                )

            ### Model Explanation & Visualization ###
            tx_dir = os.path.join(gene_dir, tx_id)
            os.makedirs(tx_dir, exist_ok=True)

            # Visualize attention weights as a heatmap
            print_emphasized("Visualizing Attention Weights in Heatmap ...")

            heatmap_path = os.path.join(tx_dir, f"attention_heatmap_{position}_{splice_type}_{label}.pdf")
            visualize_attention_weights_with_special_tokens_filtered(
                attention_weights,
                tokenized_sequence,
                highlight_top_k=5,
                amplify_off_diagonal=False,  # Amplify off-diagonal weights
                output_path=heatmap_path, 
                special_tokens=("[CLS]", "[SEP]", "[PAD]", "[UNK]")
            )

            # Visualize regions of high attention scores
            print_emphasized("Visualizing High Attention Regions ...")
            region_path = os.path.join(tx_dir, f"attention_regions_{position}_{splice_type}_{label}.pdf")
            visualize_attention_regions_static(
                sequence,
                attention_weights=attention_weights,  # 2D attention weights
                tokenized_sequence=tokenized_sequence,
                # max_scores=max_scores,  # Use max attention scores for visualization
                # avg_scores=avg_scores,  # Use average attention scores for visualization
                annotation=annotation,
                output_path=region_path,
                top_k=10,  # Highlight the top 10 regions
                color_map="viridis",  # Color map for the heatmap
                dynamic_band_height=True,
                add_legend=True,
                label_rotation=45,
                hide_overlapping_labels=False,  
                diagnostic_check=True, 
                filter_special_tokens=True,
            )

            # Alignment plot 
            print_emphasized("Visualizing Alignment Attention Regions ...")
            alignment_path = os.path.join(tx_dir, f"alignment_attention_{position}_{splice_type}_{label}.pdf")
            # visualize_alignment_attention_regions(
            #     sequence,
            #     attention_weights=attention_weights,
            #     tokenized_sequence=tokenized_sequence,
            #     annotation=annotation,
            #     output_path=alignment_path,
            #     top_k=10,
            #     color_map="viridis",

            #     dynamic_band_height=True,
            #     # nonlinear_scaling=True,
            #     # scaling_factor=10,

            #     add_legend=True,
            #     token_label_rotation=0,
            #     figsize=(20, 6), 
            #     hide_overlapping_labels=True, 
            #     # offsets=None  # Optional: list of (start, end) positions for each token
            # )
            plot_alignment_with_scores(
                sequence,
                attention_weights=attention_weights,
                tokenized_sequence=tokenized_sequence,
                annotation=annotation,
                output_path=alignment_path,
                top_k=10,
                color_map="viridis",  # "viridis", "plasma", "coolwarm"
                dynamic_band_height=True,
                add_legend=True,
                hide_overlapping_labels=False,
                token_label_rotation=0,
                figsize=(20, 6),

                # Additional toggles:
                filter_special_tokens=True,  
                special_tokens=("[CLS]", "[SEP]", "[PAD]", "[UNK]"),

                draw_alignment_arrows=True,    # If True, draw subtle dashed line arrows
                arrow_line_color="lightgrey",  # Arrow line color
                arrow_end_y=0.3,               # Where arrow ends in y dimension (just above context row)

                rect_baseline_height=0.3,      # A baseline rectangle height
                rect_scale_factor=0.7,         # Additional scale factor if dynamic_band_height is True
                # show_token_positions=True,     # Show token positions on the x-axis
                show_token_scores=True        # Show token scores on the x-axis
            )

            # Map important motifs to sequence 
            # print_emphasized("Mapping important motifs to sequence and addressing the misalignment issues between motifs and tokens ...")        
            # validated_motifs = map_and_validate_motifs_with_attention(
            #     important_motifs, sequence, attention_weights, tokenized_sequence
            # )
        
            # # Output validated motifs
            # print_with_indent("Validated Motifs:", indent_level=1)
            # for motif in validated_motifs:
            #     print(motif)

            # # Visualize motifs with annotations
            # output_path = os.path.join(tx_dir, f"validated_motifs_with_annotations_attention_{position}_{splice_type}_{label}.pdf")
            # visualize_contextual_importance_with_annotations(
            #     sequence, 
            #     validated_motifs, 
            #     output_path=output_path, 
            #     top_motifs=10, 
            #     dynamic_band_height=True,  # Enable variable band heights
            #     motif_key="motif", 
            #     attention_key="attention_score"
            # )

            print_emphasized("Integrated Gradients Analysis with layer-based approach ...")

            try: 
                ig_attributions, ig_tokens = compute_ig_attributions_with_lig(
                    model=model,
                    tokenizer=tokenizer,
                    sequence=sequence,
                    device=device,
                    baseline_seq=None,       # or e.g. "N"*len(sequence)
                    n_steps=50,
                    max_length=max_length
                )
            except Exception as e:
                print(f"Error in compute_ig_attributions_with_lig(): {e}")
                ig_attributions, ig_tokens = None, None

            # Inspect top tokens by absolute attributions
            top_positions = np.argsort(-np.abs(ig_attributions))[:10]
            print(f"[IG-Analysis] Top 10 tokens by absolute attribution for {gene_id} / {tx_id} / i={i}:")
            for pos_ in top_positions:
                print(f"  Token='{ig_tokens[pos_]}', Attribution={ig_attributions[pos_]:.4f}")

        # End foreach row in gene-specific group
    # End foreach gene

    # Loop through the genes again and track progress using tqdm
    # grouped = subsampled_val_df.groupby('gene_id')  # Group by gene_id

    print_emphasized("[info] Advanced IG Analysis ...")
    # See demo_pretrain_finetune_ig()

    # Workflow: 
    # - Gather IG for each example (looping over rows).
    # - Identify top k tokens by absolute IG.
    # - Aggregate occurrences of each token in those top-k sets for FPs vs TPs.
    # - Post-process at the end to generate a frequency table or ratio table.
        
    print_emphasized("[info] Finished processing all sequences.")


def demo_pretrain_finetune_ig(**kargs):
    """
    Pretrain and fine-tune a transformer model for splice error prediction using DNABERT 
    and analyze the model using Integrated Gradients.

    Workflow: 
    - Gather IG for each example (looping over rows).
    - Identify top k tokens by absolute IG.
    - Aggregate occurrences of each token in those top-k sets for FPs vs TPs.
    - Post-process at the end to generate a frequency table or ratio table.

    Memo: 
    - GPU Support: 
       The model and input tensors (input_ids, attention_mask) are moved to the appropriate device (cuda or cpu).
       Model inference (outputs) is performed on the selected device.
       Default device is dynamically set using torch.device.
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
    from transformers.models.bert import BertForSequenceClassification
    from meta_spliceai.splice_engine.splice_error_analyzer import ErrorAnalyzer
    from meta_spliceai.splice_engine.utils_df import (
        estimate_dataframe_size, 
        estimate_memory_usage
    )
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report

    do_fine_tuning = True
    do_manual_evaluation = False
    device, local_rank = set_device()

    facet = kargs.get("facet", "simple")    # 'trimmed',  "simple" (subsampled + trimmed)
    subject = f"analysis_sequences_{facet}"
    tokenizer_name = "zhihan1996/DNABERT-2-117M"
    model_name = "zhihan1996/DNABERT-2-117M"
    model_type = 'DNABERT'
    
    pred_type = kargs.get('pred_type', 'FP')
    error_label = kargs.get('error_label', pred_type)
    correct_label = kargs.get('correct_label', 'TP')
    gene_level_split = False

    pred_type_to_label = {error_label: 1, correct_label: 0}
    # {k: v for k, v in ErrorAnalyzer.pred_type_to_label if k in [pred_type, 'TP']}

    experiment_name = kargs.get('experiment', 'hard_genes')
    
    max_length = 1000  # Maximum sequence length for tokenization   
    col_tid = 'transcript_id'
    col_label = 'label'

    print_emphasized("[workflow] Pretrain and fine-tune a transformer model for splice error prediction using DNABERT ...")
    analyzer  = ErrorAnalyzer(experiment=experiment_name, model_type=model_type.lower())
    # error_analysis_type = f"{pred_type}_vs_TP".lower()
    output_dir = analyzer.set_analysis_output_dir(pred_type=error_label)
    os.makedirs(output_dir, exist_ok=True)
    print_with_indent(f"[info] Output directory set to: {output_dir}", indent_level=1)

    logging_dir = os.path.join(output_dir, "logs")
    os.makedirs(logging_dir, exist_ok=True)
    print_with_indent(f"[info] Logging directory set to: {logging_dir}", indent_level=1)

    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    print_with_indent(f"[info] Checkpoint directory set to: {checkpoint_dir}", indent_level=1)

    mefd = ModelEvaluationFileHandler(ErrorAnalyzer.eval_dir, separator='\t')
    # splice_pos_df = mefd.load_splice_positions(aggregated=True)

    # Use run_spliceai_workflow.error_analysis_workflow() to generate the error analysis data
    print_emphasized("[i/o] Loading and processing the analysis sequence dataset ...")

    # Loading the analysis sequence dataset which contains sequences and their associated labels
    analysis_sequence_df = mefd.load_analysis_sequences(
        aggregated=True, 
        subject=subject, 
        error_label=error_label, correct_label=correct_label
    )  # Output is a Polars DataFrame
    # analysis_sequence_df has the following columns: 
    # ['gene_id', 'transcript_id', 'chrom', 'position', 'pred_type', 'score', 'splice_type', 'strand', 
    #  'window_end', 'window_start', 'sequence']

    # Label the dataset based on the prediction type
    analysis_sequence_df = label_analysis_dataset(analysis_sequence_df, positive_class=error_label)
    # Shape of the DataFrame
    print_with_indent(f"[info] Shape of the analysis sequence DataFrame: {analysis_sequence_df.shape}", indent_level=1)
    # NOTE: Total set with Ensembl: (2,915,508, 11) ~ 3M rows, 11 columns

    # Verify the label classes
    verify_label_classes(analysis_sequence_df, label_col=col_label)

    # Analyze the data labels
    analyze_data_labels(analysis_sequence_df, label_col=col_label, verbose=2, handle_missing=None)

    # Verify sequence lengths
    max_length_prime = verify_sequence_lengths(analysis_sequence_df, sequence_col='sequence')
    if max_length_prime < max_length:
        print(f"[info] Maximum sequence length: {max_length_prime} < requested {max_length}")
        max_length = max_length_prime

    # Optionally trim the sequence to a maximum length
    # analysis_sequence_df = trim_contextual_sequences(analysis_sequence_df, target_length=200)

    print_emphasized("[info] Preparing the training and validation datasets ...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, model_max_length=max_length)
    
    train_dataset, val_dataset, train_df, val_df = \
        prepare_transformer_datasets(
            training_df=analysis_sequence_df,
            
            # Tokenizer parameteres
            tokenizer=tokenizer,
            
            test_size=0.2, 
            return_dataframes=True,
            dataset_format="huggingface",  # Change to "pytorch" for PyTorch Datasets
            gene_level_split=gene_level_split
        )

    ###### Training the Transformer Model ######

    print_emphasized("[info] Training the transformer model ...")
    
    batch_size = kargs.get('batch_size', 8)
    num_train_epochs = kargs.get('num_train_epochs', 10)

    model_id = f"{experiment_name.lower()}_{model_type.lower()}_{error_label}"
    model_path = os.path.join(output_dir, model_id)
    model = trainer = None
    deepspeed_config = os.path.join(ErrorAnalyzer.analysis_dir, "deepspeed_config.json")
    print_with_indent(f"[info] DeepSpeed configuration file: {deepspeed_config}", indent_level=1)

    print_emphasized("[info] Configuring the training parameters ...")
    training_arguments = {
        "output_dir": output_dir,

        "eval_strategy": 'epoch', # Evaluation strategy ('steps' or 'epoch')
        "save_strategy": 'epoch', # Save strategy ('steps' or 'epoch')
        # --load_best_model_at_end requires the save and eval strategy to match

        "save_steps": 0,  # Save checkpoint every `save_steps` steps; 0 disables intermediate checkpoints

        "per_device_train_batch_size": batch_size,  # Batch size per device (GPU or CPU)
        # per_device_train_batch_size=2,  # Reduce batch size for training
        # per_device_eval_batch_size=2,  # Reduce batch size for evaluation

        "num_train_epochs": 1,  # Handle epochs manually
        "weight_decay": 0.01,  # L2 regularization coefficient
        "learning_rate": 5e-5,
        "warmup_steps": 500,  # Warmup steps for learning rate scheduler

        "gradient_accumulation_steps": 4,  # Accumulate gradients over N steps

        "load_best_model_at_end": True, # Automatically load the best model at the end
        "metric_for_best_model": "eval_loss",  # Specify metric for identifying the best model (e.g., "eval_loss" or "eval_f1")
        # "lr_scheduler_type": "reduce_lr_on_plateau",  # Corrected scheduler type
        
        "logging_dir": logging_dir,  # Directory for logging
        "logging_steps": 1000,  # Log metrics every `logging_steps` steps

        "ddp_find_unused_parameters": False,  # Optimize DDP for unused parameters
        "remove_unused_columns": False,  # Remove unused columns from the dataset
        "dataloader_drop_last": True,  # Drop the last incomplete batch
        "local_rank": local_rank,  # Automatically detected by torchrun
        # When local_rank is passed to TrainingArguments, the Trainer ensures the model is placed on the appropriate device 
        # and uses torch.nn.parallel.DistributedDataParallel (DDP) for distributed training.

        # fp16=True,  # Mixed precision for memory efficiency
        # NOTE: Use fp16=True on GPUs with Tensor Core support, such as NVIDIA V100, A100, or 
        #       RTX 20/30/40 series GPUs.
        
        # "gradient_checkpointing": True,   # Enable gradient checkpointing
        # "deepspeed": deepspeed_config  # Optional: Use DeepSpeed
    }
    
    model, trainer, model_path = \
        train_transformer_model(
            train_dataset=train_dataset,  # Tokenized training dataset
            val_dataset=val_dataset,  # Tokenized validation dataset
            training_arguments=training_arguments,  # Training arguments
            
            model_name=model_name,  # Pre-trained DNABERT
            # model_class=BertForSequenceClassification,  # If specified, use_auto_model=False
            
            # tokenizer=tokenizer,  # Tokenizer for the model (not used)
            # tokenizer is unecessary because train_dataset and val_dataset are pre-tokenized

            use_auto_model=True,  # Set to True to use AutoModelForSequenceClassification
            
            num_labels=2,  # Binary classification (FP/FN vs. TP)
            num_train_epochs=num_train_epochs,  # Number of training epochs for "manual" training (not to be passed to Trainer)
            model_id=model_id,  # Model ID for saving
            device=device,  # Device (cuda or cpu)
            return_path_to_model = True,
            train_model=do_fine_tuning,
        )

    ###### Pre-Model Evaluation ######

    # (Test) Verify model consistency
    print_emphasized("[INFO] Model Configuration:")
    print(model.config)

    # Evaluate the model and plot metrics (e.g. ROC-AUC, PR-AUC curves)
    print_emphasized("[INFO] Evaluating the model on the validation set ...")
    validation_report = \
        evaluate_and_plot_metrics(trainer, val_dataset, output_dir, label_names=[correct_label, error_label])
    print_with_indent(f"[INFO] Validation Report: {validation_report}\n", indent_level=1)

    # Choose top motifs for visualization
    print_emphasized("[info] Retrieving important motifs ...")
    # analyzer = ErrorAnalyzer(experiment=experiment_name)
    df_motif = analyzer.load_motif_importance(
        error_label=error_label, correct_label=correct_label, 
        model_type='xgboost')  # Retrieve important motifs
    display_dataframe_in_chunks(df_motif, num_rows=df_motif.shape[0])

    print_with_indent("[info] Processing sequences for motif visualization ...", indent_level=1)
    motifs_with_scores = \
        prepare_motifs_for_visualization(df_motif, return_scores=True)

    print_with_indent(f"Motifs with scores: {motifs_with_scores}", indent_level=2)
    motifs = [motif for motif, _ in motifs_with_scores]
    print_with_indent(f"Motifs: {motifs}", indent_level=3)
    
    important_motifs = motifs[:5]

    # Subsample the validation DataFrame
    print_emphasized("[info] Subset validation dataframe for visualization ...")
    val_policy = "subsample"

    if val_policy == "subsample":
        # Subsample the validation DataFrame
        subsampled_val_df = subsample_genes(analysis_sequence_df, gene_col='gene_id', N=10)
    else: 
        subsampled_val_df = find_transcripts_with_both_labels(analysis_sequence_df, num_transcripts=10)
        # subsampled_val_df = val_df.filter(pl.col('transcript_id').is_in(tx_set.keys()))

        # Find the number of unique transcripts with both labels
        num_trpts_with_both_labels = subsampled_val_df['transcript_id'].nunique()
        print(f"[info] Number of transcripts with both labels: {num_trpts_with_both_labels}")

    if isinstance(subsampled_val_df, pl.DataFrame):
        subsampled_val_df = subsampled_val_df.to_pandas()
    
    ### Model Inference and Visualization ###

    # After training is complete
    model.eval()  # put the model in inference mode, outside all loops

    # Dictionary that maps labels back to prediction types
    label_to_pred_type = {v: k for k, v in pred_type_to_label.items()}

    # Global aggregator for top token frequencies

    # We store counts of tokens for each label type (FP,FN,TP,TN).
    top_token_counts = {
        "FP": defaultdict(int),
        "FN": defaultdict(int),
        "TP": defaultdict(int),
        "TN": defaultdict(int)
    }

    # Keep track of how many examples we have for each type
    num_examples = {
        "FP": 0,
        "FN": 0,
        "TP": 0,
        "TN": 0
    }

    # NOTE: Summaries vs. Local Plots
    # - Global Bar Chart (bar_chart_freq(...))
    #  
    #     - Summarizes all TPs vs. FPs in the entire dataset, so you see the “decoy tokens” or 
    #       “common motifs” that cause errors across many genes/transcripts.
    #     
    #     - Transcript-Level Summaries
    #       The same aggregator approach, but grouped by (gene_id, transcript_id) to see if certain transcripts have unique error motifs.
    #     
    #     - Per-Example Alignment or Regions Plots
    #       For a single row, a single context sequence, you show how each token’s IG distribution 
    #       reveals local “pushes” or “pulls” on the logit. This is more qualitative, letting you 
    #       see exactly which substring was crucial in that example’s classification.


    def determine_label_str(label): 
        if label == 0: 
            return correct_label
        if label == 1: 
            return error_label

    def print_freq_results(result_list, error_label="FP", correct_label="TP", top_n=20):
        print(f"\n=== {error_label} vs {correct_label} ===")
        print("token\terror_freq\tcorrect_freq\tdiff")
        for tk, ef, cf, df in result_list[:top_n]:
            print(f"{tk}\t{ef:.3f}\t{cf:.3f}\t{df:.3f}")

    # We'll keep top_k at 10 for example
    TOP_K = 10
    TOP_K_GLOBAL = 30

    grouped = subsampled_val_df.groupby('gene_id')

    for gene_id, group in tqdm(grouped, desc="IG Analysis by Gene"):

        # Create a new directory for each gene
        gene_dir = os.path.join(output_dir, gene_id)
        os.makedirs(gene_dir, exist_ok=True)

        # Aggregator for just this transcript
        top_token_counts_local = {error_label: defaultdict(int), correct_label: defaultdict(int)}
        num_examples_local     = {error_label: 0, correct_label: 0}

        tp_rows = group[group['label'] == 0]  # or however you store your labels

        # Gather a TP for baseline
        baseline_seq = None
        # if not tp_rows.empty:
        #     baseline_seq = tp_rows.iloc[0]['sequence']  # pick first

        for i, row in group.iterrows():
            label_str = determine_label_str(row['label'])   # e.g. "FP","FN","TP","TN"
            sequence = row['sequence']
            tx_id = row[col_tid]
            
            annotation = row.to_dict()
            # keys: ['gene_id', 'transcript_id', 'chrom', 'position', 'label', 
            #        'score', 'splice_type', 'strand', 'window_end', 'window_start', 'sequence']
            
            annotation['error_type'] = error_label
            splice_type = annotation['splice_type']
            position = annotation['position']

            # decide if we skip or handle
            if label_str not in ["FP","FN","TP","TN"]:
                continue  # ignore or handle differently

            # pick baseline logic
            if label_str in ["FP","FN"] and baseline_seq is not None:
                # error with a "TP" baseline
                used_baseline = baseline_seq
            else:
                # fallback baseline, e.g. None => "A"*len(seq)
                used_baseline = None

            # compute IG
            try:
                ig_attributions, ig_tokens = compute_ig_attributions_with_lig(
                    model=model,
                    tokenizer=tokenizer,
                    sequence=sequence,
                    device=device,
                    baseline_seq=used_baseline,
                    n_steps=50,
                    max_length=max_length
                )
            except Exception as e:
                print(f"[Error IG] {gene_id} row={i} {label_str}: {e}")
                continue

            # top-k by absolute IG
            top_positions = np.argsort(-np.abs(ig_attributions))[:TOP_K]
            top_tokens = [ig_tokens[pos] for pos in top_positions]

            # Update local aggregator
            num_examples_local[label_str]+=1
            for tk in top_tokens:
                top_token_counts_local[label_str][tk] += 1

            # Update global aggregator
            num_examples[label_str] += 1
            for tk in top_tokens:
                top_token_counts[label_str][tk] += 1

            # debugging
            print(f"[IG] gene={gene_id}, row={i}, label={label_str}, top tokens:")
            for pos_ in top_positions:
                print(f"   {ig_tokens[pos_]} => {ig_attributions[pos_]:.4f}")

            tx_dir = os.path.join(gene_dir, tx_id)
            os.makedirs(tx_dir, exist_ok=True)

            # Optionally visualize the IG attributions
            scores = np.abs(ig_attributions)   # ig_attributions   
            alignment_path = os.path.join(tx_dir, f"alignment_ig_attributions_{position}_{splice_type}_{label_str}.pdf")
            plot_alignment_with_ig_scores(
                sequence,
                attention_weights=scores,   
                tokenized_sequence=ig_tokens,
                annotation=annotation,
                output_path=alignment_path,
                top_k=10,
                color_map="viridis",  # "viridis", "plasma", "coolwarm"
                dynamic_band_height=True,
                add_legend=True,
                hide_overlapping_labels=False,
                token_label_rotation=0,
                figsize=(20, 6),

                # Additional toggles:
                filter_special_tokens=True,  
                special_tokens=("[CLS]", "[SEP]", "[PAD]", "[UNK]"),

                draw_alignment_arrows=True,    # If True, draw subtle dashed line arrows
                arrow_line_color="lightgrey",  # Arrow line color
                arrow_end_y=0.3,               # Where arrow ends in y dimension (just above context row)

                rect_baseline_height=0.3,      # A baseline rectangle height
                rect_scale_factor=0.7,         # Additional scale factor if dynamic_band_height is True
                
                show_token_scores=True,       # Show token scores on the x-axis
                show_token_positions=False     # Show token positions on the x-axis
            )

        # Now build frequencies for local
        # e.g. do a local FP vs TP comparison
        local_comparison = build_token_frequency_comparison(
            top_token_counts_local, num_examples_local, error_label=error_label, correct_label=correct_label
        )

        # Optionally print or create a bar chart just for this transcript
        print(f"\n=== gene={gene_id}, transcript={tx_id}: {error_label} vs {correct_label} frequencies ===")
        print_freq_results(local_comparison, error_label, correct_label, top_n=TOP_K)

        chart_path = os.path.join(tx_dir, f"token_frequency_{error_label}_vs_{correct_label}.pdf")
        bar_chart_freq(local_comparison, error_label, correct_label, top_n=TOP_K, output_path=chart_path)

    # End foreach gene

    print_emphasized("[info] Post-processing token frequencies: Generate the specific comparisons ...")
    chart_path = os.path.join(output_dir, f"token_frequency_{error_label}_vs_{correct_label}.pdf")
    
    global_comparison = \
        build_token_frequency_comparison(
            top_token_counts, num_examples, error_label=error_label, correct_label=correct_label
    )
    # A list of tuples (token, error_freq, correct_freq, diff) sorted by descending absolute difference
    print_freq_results(global_comparison, error_label, correct_label, top_n=TOP_K_GLOBAL)
    bar_chart_freq(global_comparison, error_label, correct_label, top_n=TOP_K_GLOBAL, output_path=chart_path)
    # NOTE: Global bar chart 
    # - Summarizes all TPs vs. FPs in the entire dataset, so you see the “decoy tokens” or “common motifs” 
    #   that cause errors across many genes/transcripts.


    # err_count = top_token_counts[pred_type][tk]
    # tp_count = top_token_counts["TP"][tk]

    # err_freq = err_count / num_err_examples  # fraction of FP or FN examples that had 'tk' in top-10
    # tp_freq = tp_count / num_tp_examples  # fraction of TP examples that had 'tk' in top-10

    # result_list = []
    # all_tokens = set(top_token_counts[pred_type].keys()) | set(top_token_counts["TP"].keys())
    # for tk in all_tokens:
    #     fp_c = top_token_counts[pred_type][tk]
    #     tp_c = top_token_counts["TP"][tk]
    #     err_freq = fp_c / num_err_examples if num_err_examples>0 else 0
    #     tp_freq = tp_c / num_tp_examples if num_tp_examples>0 else 0
    #     diff = err_freq - tp_freq
    #     result_list.append((tk, err_freq, tp_freq, diff))

    # # sort by absolute difference or by descending diff
    # result_list.sort(key=lambda x: abs(x[3]), reverse=True)

    # print("\n=== Token-Level Frequency Comparison ===")
    # err_freq = f"{pred_type}_freq".lower()
    # print(f"token\t{err_freq}\ttp_freq\tdiff")
    # for tk, ff, tf, df in result_list[:50]:
    #     print(f"{tk}\t{ff:.3f}\t{tf:.3f}\t{df:.3f}")

    # frequency_path = os.path.join(output_dir, f"token_frequency_{pred_type}.tsv")
    # plot_token_frequency_bar(
    #     result_list, 
    #     top_n=20, 
    #     output_path=frequency_path)
    # A grouped bar chart: for each token (x-axis), a bar for fp_freq and a bar for tp_freq. 
    # This visually highlights tokens that are heavily skewed toward errors (FP or FN) vs. correct predictions (TP).


def verify_installation(): 
    # from transformers import AutoTokenizer, AutoModelForSequenceClassification
    # tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M")
    # model = AutoModelForSequenceClassification.from_pretrained(
    #     "zhihan1996/DNABERT-2-117M", num_labels=2, trust_remote_code=True)
    # NOTE: The model class you are passing has a `config_class` attribute that is not consistent with the config class you passed

    from transformers import AutoTokenizer

    # Import custom configuration and model classes
    from transformers import AutoConfig
    from transformers.models.bert import BertForSequenceClassification

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)

    # Load the custom configuration
    config = AutoConfig.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
    # NOTE: loads the custom configuration from DNABERT's repository.

    # Load the model using the correct configuration
    model = BertForSequenceClassification.from_pretrained(
        "zhihan1996/DNABERT-2-117M",
        config=config,
        # trust_remote_code=True
    )
    # NOTE: Use BertForSequenceClassification explicitly instead of AutoModelForSequenceClassification 
    #       to bypass the class mismatch.


def demo_synthetic_dataset():
    from transformers.models.bert import BertForSequenceClassification
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report

    col_tid = 'transcript_id'
    col_label = 'label'
    pred_type = 'FP'
    error_label = pred_type

    # Generate synthetic data
    print_emphasized("Generating synthetic dataset ...")
    synthetic_df = generate_synthetic_dataset(
        num_samples=1000,
        sequence_length=200,
        tp_motif="GTGAGT",
        fp_motif="AGGTAG",
        regulatory_motif="CACACA",
        inter_token_distance=5,
        random_noise=True,
        noise_probability=0.05
    )
    print(synthetic_df.head())
    print(synthetic_df[col_label].value_counts())

    # Verify if synthetic data is labeled 
    if col_label not in synthetic_df.columns:
        # Synthetic data has not been labeled yet
        synthetic_df = label_analysis_dataset(synthetic_df, positive_class=error_label)

    tokenizer_name = "zhihan1996/DNABERT-2-117M"
    model_name = "zhihan1996/DNABERT-2-117M"

    experiment_name = f"test"
    output_dir = os.path.join(ErrorAnalyzer.analysis_dir, experiment_name)
    os.makedirs(output_dir, exist_ok=True)

    logging_dir = os.path.join(output_dir, "logs")
    os.makedirs(logging_dir, exist_ok=True)

    # Split synthetic data into train/validation
    print_emphasized("Preparing the training and validation datasets ...")
    train_dataset, val_dataset, train_df, val_df = prepare_transformer_datasets(
        synthetic_df, 
        tokenizer_name=tokenizer_name, 
        test_size=0.2, 
        return_dataframes=True
    )

    print(f"Training dataset size: {len(train_df)}")
    print(f"Validation dataset size: {len(val_df)}")

    model, trainer = train_transformer_model(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            model_name=model_name,
            output_dir=output_dir,
            logging_dir=logging_dir,
            model_class=BertForSequenceClassification,
            use_auto_model=False,
            num_labels=2,
            num_train_epochs=10,
            batch_size=8,  
            gradient_accumulation_steps=4,  # Effective batch size of 64
            early_stopping_patience=5,
            learning_rate=5e-5,  # Adjusted learning rate
            warmup_steps=100,  # Added warmup steps
            logging_steps=50
       )

    print_emphasized("Evaluating the model on the validation set ...")

    # Prepare tokenizer for encoding validation sequences
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    max_length = 512

    # Encode validation sequences
    val_encodings = tokenizer(
        list(val_df['sequence']),
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )

    # Get predictions
    model.eval()
    with torch.no_grad():
        outputs = model(val_encodings['input_ids'], attention_mask=val_encodings['attention_mask'])
        preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()

    # Evaluate predictions
    # true_labels = (val_df['pred_type'] == "FP").astype(int).values  # Convert FP/TP to binary labels
    true_labels = (val_df[col_label] == 1).values  
    print(classification_report(true_labels, preds, target_names=["TP", pred_type]))

    # Test Predictions on Example Sequences
    print_emphasized("Example Predictions:")

    # example_sequence = val_df.iloc[0]['sequence']
    # example_sequence = "ATCGTCTAGGTAGTCTGAGTAGGTACGTAGTGTAGAGTCTAGGTA"   # Simulate a TP
    # A, TCGTC, TAGG, TAGTC, TGAGTA, ... 

    # Replace example_sequence with the new test examples
    tp_sequence = "ATCGTCT" + "GTGAGT" + "GCA" + "CACACA" + "TGTAGTCTAGGTA"  # TP: True Positive (label=0)
    fp_sequence = "TGCATAG" + "AGGTAG" + "TTG" + "GATCAG" + "GACTCGTGTAC"   # FP: False Positive (label=1)

    # Test both TP and FP sequences
    for example_sequence, encoded_label in [(tp_sequence, 0), (fp_sequence, 1)]:
        label = "TP" if encoded_label == 0 else "FP"  # Human-readable class label for display
        print(f"\nEvaluating {label} Sequence (Encoded as {encoded_label}):")

        # Generate motifs
        motifs = generate_motifs_from_sequence(example_sequence, min_k=2, max_k=4)
        N = 10  # Specify the number of motifs to select
        important_motifs = random.sample(motifs, N)
        print(f"Important motifs: {important_motifs}")

        # Encode the sequence
        encoded_example = tokenizer(
            example_sequence,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )

        # Predict class for the example
        with torch.no_grad():
            output = model(encoded_example['input_ids'], attention_mask=encoded_example['attention_mask'])
            prediction = torch.argmax(output.logits).item()

        predicted_label = "FP" if prediction == 1 else "TP"
        print(f"Sequence: {example_sequence}")
        print(f"Predicted class: {predicted_label} (Encoded as {prediction})")

        original_sequence = example_sequence

        # Extract attention weights using rollout
        # attention_weights, tokenized_sequence, max_scores, avg_scores = \
        # get_attention_weights(
        #     model=model,
        #     tokenizer=tokenizer,
        #     sequence=original_sequence,
        #     layer=None,  # Use all layers for rollout
        #     head=None,  # Use all heads for rollout
        #     max_length=max_length,
        #     return_max_and_avg=True,
        #     policy="rollout"
        # )
        attention_weights, tokenized_sequence = \
            get_attention_weights(
                model=model,
                tokenizer=tokenizer,
                sequence=original_sequence,
                layer=None,  # Use all layers for rollout
                head=None,  # Use all heads for rollout
                max_length=max_length,
                policy="rollout"
            )

        # Visualize attention weights as a heatmap
        print_emphasized("Visualizing Attention Weights in Heatmap ...")
        heatmap_path = os.path.join(output_dir, f"{label}_attention_heatmap.pdf")
        visualize_attention_weights(
            attention_weights,
            tokenized_sequence,
            highlight_top_k=5,
            amplify_off_diagonal=False,  # Amplify off-diagonal weights
            output_path=heatmap_path
        )

        # Map motifs and validate with attention

        # Alternative visualization: Focusing on regions of high attention scores
        print_emphasized("Visualizing High Attention Regions ...")
        region_path = os.path.join(output_dir, f"{label}_attention_regions.pdf")
        visualize_attention_regions(
            original_sequence,
            attention_weights=attention_weights,  # 2D attention weights
            tokenized_sequence=tokenized_sequence,
            # max_scores=max_scores,  # Use max attention scores for visualization
            # avg_scores=avg_scores,  # Use average attention scores for visualization
            top_k=10,  # Highlight the top 10 regions
            dynamic_band_height=True,
            output_path=region_path
        )

        print_emphasized("Mapping important motifs to sequence and addressing the misalignment issues between motifs and tokens ...")        
    
        validated_motifs = map_and_validate_motifs_with_attention(
            important_motifs, original_sequence, attention_weights, tokenized_sequence
        )
    
        # Output validated motifs
        print_with_indent("Validated Motifs:", indent_level=1)
        for motif in validated_motifs:
            print(motif)

        # Visualize validated motifs
        # output_path = os.path.join(output_dir, "example_validated_motifs.pdf")
        # visualize_contextual_importance(
        #     original_sequence, 
        #     validated_motifs, 
        #     output_path=output_path, 
        #     motif_key="motif", 
        #     attention_key="mean_attention_score")

        # Visualize validated motifs with annotations
        output_path = os.path.join(output_dir, "validated_motifs_with_annotations.pdf")
        visualize_contextual_importance_with_annotations(
            original_sequence, 
            validated_motifs, 
            output_path=output_path, 
            top_motifs=10, 
            dynamic_band_height=False,  # Enable variable band heights
            motif_key="motif", 
            attention_key="attention_score"
        )

        output_path = os.path.join(output_dir, "validated_motifs_with_annotations_attention_weighted_height.pdf")
        visualize_contextual_importance_with_annotations(
            original_sequence, 
            validated_motifs, 
            output_path=output_path, 
            top_motifs=10, 
            dynamic_band_height=True,  # Enable variable band heights
            motif_key="motif", 
            attention_key="attention_score"
        )


def demo():
    # from meta_spliceai.splice_engine.splice_error_analyzer import demo_make_training_set
    correct_label = 'TP'
    pred_type = 'FP'
    error_label = pred_type
    facet = 'simple'

    n_epochs = 8
    batch_size = 10

    # Creating featurized training dataset based on gene selection criteria (e.g. hard genes)
    print_emphasized("Demo: Creating featurized dataset based on gene selection criteira ...")
    # demo_make_training_set(pred_type=pred_type)

    print_emphasized("Demo: Processing Training Dataset ...")
    demo_process_training_dataset(pred_type=pred_type, facet=facet)

    print_with_indent("Retrieve Motif Importance ...", indent_level=1)
    # demo_retrieve_motif_importance(pred_type=pred_type)

    print_with_indent("Verify Installation ...", indent_level=1)
    # verify_installation()

    print_with_indent("Augment sequence dataset ...", indent_level=1)
    subject = 'analysis_sequences'
    # augmented_analysis_sequence_dataset(experiment='hard_genes', pred_type=pred_type)

    print_with_indent("Testing the workflow utilities with synthetic datasets ...", indent_level=1)
    # demo_synthetic_dataset()

    # Define a focused set of transcripts for visualization and testing
    focus_tx_set = None
    if pred_type == 'FN': 
        # FN vs TP: 
        # Transcript ID: ENST00000247866, Gene ID: ENSG00000090266 (strand: +)
        # Transcript ID: ENST00000261652, Gene ID: ENSG00000240505 (strand: -)
        focus_tx_set = ['ENST00000247866', 'ENST00000261652', ]

    print_emphasized("Demo: Pretraining and Finetuning Transformer Models")
    demo_pretrain_finetune(
        pred_type=pred_type, 
        facet=facet, 
        num_train_epochs=n_epochs, batch_size=batch_size, 
        # focused_tx_set=focus_tx_set
    )

    print_emphasized("Demo: Advanced Integrated Gradients Analysis")
    demo_pretrain_finetune_ig(
        error_label=error_label, 
        correct_label=correct_label,
        facet=facet, 
        num_train_epochs=n_epochs, batch_size=batch_size, 
        # focused_tx_set=focus_tx_set
    )


if __name__ == "__main__":
    demo()

