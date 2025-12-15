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
    compute_attention_rollout, 
    # prepare_dataset,
    # prepare_dataset_for_huggingface, 
)

import torch
print("[info] PyTorch version:", torch.__version__)

import transformers
print("[info] Transformers version:", transformers.__version__)

from transformers.integrations import TensorBoardCallback

# Update torch and transformers to the latest versions as needed
# pip install --upgrade torch transformers

from transformers import (
    AutoTokenizer,  # pip install transformers
    AutoModelForSequenceClassification,  # pip install einops
    Trainer, 
    TrainingArguments
)

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score


import torch.nn as nn

class HybridTransformerModel(nn.Module):
    def __init__(self, transformer_model, num_additional_features, num_labels):
        super(HybridTransformerModel, self).__init__()
        self.transformer = transformer_model
        self.fc_features = nn.Linear(num_additional_features, 128)  # Embedding for additional features
        self.fc_sequence = nn.Linear(self.transformer.config.hidden_size + 128, 256)  # Combine sequence and features
        self.classifier = nn.Linear(256, num_labels)  # Final classification layer
        self.dropout = nn.Dropout(0.3)

    def forward(self, input_ids, attention_mask, additional_features):
        # Transformer outputs
        transformer_outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        sequence_embedding = transformer_outputs.last_hidden_state[:, 0, :]  # [CLS] token embedding

        # Process additional features
        feature_embedding = self.fc_features(additional_features)

        # Combine sequence and features
        combined_embedding = torch.cat([sequence_embedding, feature_embedding], dim=-1)
        combined_embedding = self.dropout(combined_embedding)

        # Classification head
        logits = self.classifier(self.fc_sequence(combined_embedding))
        return logits


def train_hybrid_transformer_model(
    train_dataset, val_dataset, transformer_model_name, num_additional_features, num_labels, training_args
):
    # Load transformer
    transformer_model = AutoModel.from_pretrained(transformer_model_name)

    # Initialize hybrid model
    model = HybridTransformerModel(
        transformer_model=transformer_model,
        num_additional_features=num_additional_features,
        num_labels=num_labels
    )

    # Use Hugging Face's Trainer API for training
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )

    trainer.train()
    return model, trainer


def prepare_dataset_for_huggingface(
    df, tokenizer, pred_type='FP', label_column='label', sequence_column='sequence',
    feature_columns=None, max_length=512
):
    """
    Prepare dataset for transformer-based models with explicit max_length and external features.

    Returns:
    - Hugging Face Dataset object compatible with the Trainer API.
    """
    from datasets import Dataset

    # Check if required columns are present
    required_columns = [label_column, sequence_column]
    if feature_columns:
        required_columns.extend(feature_columns)

    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")

    # Tokenize sequences
    tokenized = tokenizer(
        list(df[sequence_column]),
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="np"  # Use NumPy for compatibility with Hugging Face Dataset
    )

    # Extract labels
    if df[label_column].dtype == int:
        labels = df[label_column].values
    else:
        labels = (df[label_column] == pred_type).astype(int).values  # FP=1, TP=0

    # Extract features if available
    if feature_columns:
        features = df[feature_columns].to_numpy(dtype=np.float32)
    else:
        features = None

    # Combine encodings, features, and labels into a dictionary
    dataset_dict = {
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"],
        "labels": labels,
    }

    if features is not None:
        dataset_dict["features"] = features

    # Convert to Hugging Face Dataset
    dataset = Dataset.from_dict(dataset_dict)
    return dataset


def prepare_transformer_datasets(
    training_df, 
    tokenizer_name="zhihan1996/DNABERT-2-117M", 
    tokenizer=None,  
    test_size=0.2, 
    return_dataframes=False, 
    random_state=42,
    feature_columns=None,  # Add feature_columns
    **kargs
):
    """
    Prepare training and validation datasets for transformer-based models, 
    supporting Hugging Face or PyTorch datasets, with optional external features.
    """
    col_label = kargs.get("col_label", "label")
    sequence_column = kargs.get("sequence_column", "sequence")
    dataset_format = kargs.get("dataset_format", "huggingface")
    max_length = kargs.get("max_length", 512)

    # Handle Polars DataFrames
    is_polars = isinstance(training_df, pl.DataFrame)
    if is_polars:
        training_df = training_df.to_pandas()

    # Validate required columns
    required_columns = [col_label, sequence_column]
    if feature_columns:
        required_columns.extend(feature_columns)

    missing_columns = [col for col in required_columns if col not in training_df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in the input DataFrame: {missing_columns}")

    # Split dataset into training and validation
    train_df, val_df = train_test_split(training_df, test_size=test_size, random_state=random_state)
    train_df = train_df.sample(frac=1, random_state=random_state).reset_index(drop=True)  # Shuffle training set

    # Initialize tokenizer
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, model_max_length=max_length)
        print(f"[info] Initialized tokenizer with model_max_length={max_length}")
    else:
        print(f"[info] Using pre-initialized tokenizer: {tokenizer}")

    # Prepare datasets
    if dataset_format.startswith("hugging"):
        train_dataset = prepare_dataset_for_huggingface(
            train_df, tokenizer, pred_type=kargs.get("pred_type", "FP"), 
            label_column=col_label, sequence_column=sequence_column, 
            feature_columns=feature_columns, max_length=max_length
        )
        val_dataset = prepare_dataset_for_huggingface(
            val_df, tokenizer, pred_type=kargs.get("pred_type", "FP"), 
            label_column=col_label, sequence_column=sequence_column, 
            feature_columns=feature_columns, max_length=max_length
        )
    else:
        raise ValueError(f"Dataset format '{dataset_format}' is not yet supported for hybrid models.")

    if return_dataframes:
        return train_dataset, val_dataset, train_df, val_df

    return train_dataset, val_dataset


def get_attention_weights(model, tokenizer, sequence, layer=None, head=None, max_length=512, policy="mean"):
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
    inputs = tokenizer(sequence, truncation=True, max_length=max_length, return_tensors="pt")

    # Forward pass with attention outputs
    with torch.no_grad():
        outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], output_attentions=True)

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
        attention_weights = compute_attention_rollout(attentions)
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

    # Normalize attention scores
    attention_weights = (attention_weights - attention_weights.min()) / (attention_weights.max() - attention_weights.min() + 1e-8)

    return attention_weights, tokenized_sequence


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


def train_hybrid_transformer_model(
    train_dataset,
    val_dataset,
    model_name="zhihan1996/DNABERT-2-117M",
    output_dir="./model_output",
    num_labels=2,
    num_additional_features=0,
    num_train_epochs=20,

    batch_size=16,
    weight_decay=0.01,
    learning_rate=5e-5,
    warmup_steps=100,

    save_steps=0,  # Disable intermediate checkpoints by default
    save_best_model=True,  # Save only the best model

    logging_dir="./logs",
    logging_steps=100,
    eval_strategy="steps",

    lr_scheduler_type="reduce_on_plateau",  # Built-in scheduler
    early_stopping_patience=8,  # Number of epochs with no improvement for early stopping
    scheduler_metric="eval_loss",  # Metric for learning rate scheduler
    scheduler_mode="min",  # "min" for loss, "max" for metrics like F1

    enable_checkpointing=False, 

    model_class=None,
    **kwargs
):
    """
    Train a hybrid transformer model with external features and sequence inputs.

    Parameters:
    - train_dataset: Training dataset with tokenized sequences and additional features.
    - val_dataset: Validation dataset with tokenized sequences and additional features.
    - model_name: Pre-trained transformer model name.
    - num_labels: Number of output labels (e.g., 2 for binary classification).
    - num_additional_features: Number of additional input features.
    - num_train_epochs: Number of training epochs.
    - batch_size: Batch size for training.
    - weight_decay: Weight decay for the optimizer.
    - learning_rate: Learning rate.
    - warmup_steps: Number of warmup steps for the scheduler.
    - save_steps: Save checkpoint every `save_steps` steps.
    - save_best_model: Save the best model based on validation performance.
    - logging_dir: Directory for logging metrics.
    - logging_steps: Log metrics every `logging_steps` steps.
    - eval_strategy: Evaluation strategy (e.g., "steps").
    - lr_scheduler_type: Learning rate scheduler type.
    - early_stopping_patience: Patience for early stopping.
    - scheduler_metric: Metric for early stopping and scheduler (e.g., "eval_loss").
    - scheduler_mode: Direction for metric optimization ("min" or "max").
    - model_class: Custom model class for hybrid models.
    """
    from transformers import TrainingArguments, Trainer, EarlyStoppingCallback
    # from torch.optim.lr_scheduler import ReduceLROnPlateau
    # from transformers.integrations import TensorBoardCallback
    from transformers import AutoModel
    import numpy as np

    # Load the transformer model
    if model_class is not None:
        model = model_class.from_pretrained(model_name, num_labels=num_labels)
    else:
        raise ValueError("A custom hybrid model class must be provided.")

    # Define training arguments
    # training_args = TrainingArguments(
    #     output_dir=output_dir,  # Directory to save model checkpoints
    #     evaluation_strategy="epoch",  # Evaluate at the end of each epoch
    #     save_strategy="epoch" if enable_checkpointing else "no",  # Conditional checkpointing
    #     save_total_limit=1 if enable_checkpointing else None,  # Retain only 1 checkpoint if enabled
    #     load_best_model_at_end=enable_checkpointing,  # Load best model only if checkpointing is enabled
    #     metric_for_best_model=scheduler_metric,  # Metric to determine the best model
    #     greater_is_better=(scheduler_mode == "max"),  # Whether higher values indicate better performance

    #     per_device_train_batch_size=batch_size,  # Mini-batch size per device
    #     gradient_accumulation_steps=4,  # Effective batch size is batch_size * 4
    #     learning_rate=learning_rate,  # Initial learning rate
    #     weight_decay=weight_decay,  # Apply weight decay to optimize generalization
    #     num_train_epochs=num_train_epochs,  # Total number of training epochs
    #     warmup_steps=warmup_steps,  # Number of steps for learning rate warmup

    #     logging_dir=logging_dir,  # Directory to save training logs
    #     logging_steps=logging_steps,  # Log metrics every specified number of steps

    #     report_to="tensorboard",  # Log metrics to TensorBoard for visualization
    #     fp16=True,  # Enable mixed precision training for faster performance on supported GPUs
    # )
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch" if enable_checkpointing else "no",
        save_total_limit=1 if enable_checkpointing else None,
        load_best_model_at_end=enable_checkpointing,
        metric_for_best_model=scheduler_metric,
        greater_is_better=(scheduler_mode == "max"),

        per_device_train_batch_size=16,  # Increase batch size if memory allows
        gradient_accumulation_steps=4,  # Effective batch size = 16 * 4
        learning_rate=3e-5,  # Lower learning rate for large dataset
        weight_decay=weight_decay,
        num_train_epochs=10,  # Start with fewer epochs and monitor performance
        warmup_steps=1000,  # More warmup steps for stable training

        logging_dir=logging_dir,
        logging_steps=logging_steps,

        report_to="tensorboard",
        fp16=True,  # Mixed precision training
    )

    # Custom data collator for hybrid input
    class HybridDataCollator:
        def __call__(self, batch):
            input_ids = torch.stack([item["input_ids"] for item in batch])
            attention_mask = torch.stack([item["attention_mask"] for item in batch])
            additional_features = torch.stack([item["additional_features"] for item in batch])
            labels = torch.tensor([item["labels"] for item in batch])
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "additional_features": additional_features,
                "labels": labels,
            }
    # The HybridDataCollator class appropriately stacks these tensors into a batch. 
    # Ensure that prepare_transformer_datasets() correctly populates the additional_features field 
    # as a tensor with shape (num_samples, num_additional_features).

    # Define the trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=HybridDataCollator(),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience), TensorBoardCallback()],
    )
    # TensorBoardCallback enables visualization of training metrics in TensorBoard

    print("[INFO] Starting training...")

    # Track best performance metrics
    best_metric = np.inf if scheduler_mode == "min" else -np.inf
    best_epoch = 0
    no_improve_count = 0

    # Train for multiple epochs with manual control
    for epoch in range(num_train_epochs):
        print(f"==== Starting Epoch {epoch + 1}/{num_train_epochs} ====")
        trainer.train()

        # Evaluate on training and validation sets
        train_results = trainer.evaluate(train_dataset)
        val_results = trainer.evaluate(val_dataset)

        train_loss = train_results["eval_loss"]
        val_loss = val_results["eval_loss"]
        train_f1 = train_results["eval_f1"]
        val_f1 = val_results["eval_f1"]
        train_auc = train_results["eval_auc"]
        val_auc = val_results["eval_auc"]

        # Monitor metrics
        print("=======================================================")
        print(f"Epoch {epoch + 1} Metrics:")
        print(f"  Train Loss: {train_loss:.4f} -> Val Loss: {val_loss:.4f}")
        print(f"  Train F1: {train_f1:.4f} -> Val F1: {val_f1:.4f}")
        print(f"  Train AUC: {train_auc:.4f} -> Val AUC: {val_auc:.4f}")
        print("=======================================================")

        # Early stopping logic
        if scheduler_mode == "min" and val_loss < best_metric:
            best_metric = val_loss
            best_epoch = epoch + 1
            no_improve_count = 0
        elif scheduler_mode == "max" and val_f1 > best_metric:
            best_metric = val_f1
            best_epoch = epoch + 1
            no_improve_count = 0
        else:
            no_improve_count += 1

        # Save the best model
        if save_best_model and no_improve_count == 0:
            best_model_path = f"{output_dir}/best_model"
            trainer.save_model(best_model_path)
            print(f"[INFO] Saved best model to {best_model_path}")

        # Early stopping
        if no_improve_count >= early_stopping_patience:
            print(f"[INFO] Early stopping at epoch {epoch + 1}. Best epoch: {best_epoch}.")
            break

    print(f"[INFO] Training completed. Best metric: {best_metric:.4f} at epoch {best_epoch}.")
    return model, trainer


def demo_augment_sequence_data():
    from meta_spliceai.splice_engine.train_error_sequence_model import (
        augmented_analysis_sequence_dataset
    )

    augmented_analysis_sequence_dataset(experiment='hard_genes', pred_type='FP', save=True)


def demo_pretrain_finetune(): 
    from meta_spliceai.splice_engine.train_error_sequence_model import augmented_analysis_sequence_dataset
    from meta_spliceai.splice_engine.analysis_utils import (
        abridge_sequence, 
        label_analysis_dataset
    )
    from meta_spliceai.splice_engine.seq_modeling_utils import (
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

    subject = "analysis_sequences_sampled"
    tokenizer_name = "zhihan1996/DNABERT-2-117M"
    model_name = "zhihan1996/DNABERT-2-117M"
    model_type = 'DNABERT'
    pred_type = "FP"
    experiment_name = f"hard_genes"
    
    max_length = 1000  # Maximum sequence length for tokenization   
    col_tid = 'transcript_id'
    col_label = 'label'

    output_dir = os.path.join(ErrorAnalyzer.analysis_dir, experiment_name)
    os.makedirs(output_dir, exist_ok=True)

    logging_dir = os.path.join(output_dir, "logs")
    os.makedirs(logging_dir, exist_ok=True)

    mefd = ModelEvaluationFileHandler(ErrorAnalyzer.eval_dir, separator='\t')

    augmented_analysis_sequence_dataset(experiment='hard_genes', pred_type='FP')


def demo(): 

    demo_augment_sequence_data()


if __name__ == "__main__":
    demo()