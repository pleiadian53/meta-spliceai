"""
Integrated Gradients Analysis for DNA Sequence Models

This module provides functions for performing Integrated Gradients (IG) analysis
of DNA sequence models, designed to work with multi-GPU setups using DeepSpeed.
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from captum.attr import IntegratedGradients, visualization
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import polars as pl
from collections import defaultdict

# Import from other modules
from .error_sequence_distributed import (
    log_memory_usage,
    log_gpu_memory_detailed,
    set_device,
    is_distributed_training
)

from .error_sequence_model_dist import (
    prepare_transformer_datasets,
    evaluate_and_plot_metrics,
    plot_alignment_with_ig_scores,
    build_token_frequency_comparison,
    bar_chart_freq
)

from .model_evaluator import ModelEvaluationFileHandler


def run_ig_analysis(model, tokenized_examples, device, target_class=1, n_steps=50, baseline_type="zero"):
    """Run Integrated Gradients analysis on a single tokenized example.
    
    Args:
        model: The model to analyze
        tokenized_examples: A tuple/list of (input_ids, attention_mask) or a dictionary with these keys
                           Can be either a single example or a batch
        device: Device to run analysis on
        target_class: Target class for attribution (0 or 1)
        n_steps: Number of steps for IG
        baseline_type: Type of baseline to use - "zero" for zero embeddings or "random" for random
        
    Returns:
        A dictionary with attributions for each token
    """
    from captum.attr import IntegratedGradients, visualization
    
    # Extract input_ids and attention_mask from tokenized_examples
    if isinstance(tokenized_examples, (tuple, list)) and len(tokenized_examples) >= 2:
        input_ids, attention_mask = tokenized_examples[0], tokenized_examples[1]
    elif isinstance(tokenized_examples, dict):
        input_ids = tokenized_examples.get("input_ids")
        attention_mask = tokenized_examples.get("attention_mask") 
    else:
        raise ValueError("tokenized_examples must be a tuple, list, or dict with input_ids and attention_mask")
    
    # Convert to proper tensor type and device if necessary
    if not isinstance(input_ids, torch.Tensor):
        input_ids = torch.tensor(input_ids, dtype=torch.long)
    
    if not isinstance(attention_mask, torch.Tensor):
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        
    # Ensure input_ids are Long type (required for embedding layer)
    input_ids = input_ids.long().to(device)
    attention_mask = attention_mask.long().to(device)
    
    # Handle batch dimension or single example
    is_batched = len(input_ids.shape) > 1
    
    # If not batched, add batch dimension
    if not is_batched:
        input_ids = input_ids.unsqueeze(0)
        attention_mask = attention_mask.unsqueeze(0)
    
    # Process one example at a time for IG analysis (to avoid batch size issues)
    all_token_attributions = []
    for i in range(input_ids.shape[0]):
        # Get single example
        single_input_ids = input_ids[i:i+1].clone().detach()  # Keep batch dimension
        single_attention_mask = attention_mask[i:i+1].clone().detach()
        
        # Ensure they're on the correct device and have the right type
        single_input_ids = single_input_ids.to(device).long()
        single_attention_mask = single_attention_mask.to(device).long()
        
        try:
            # Create a wrapped model that includes the embedding layer
            class ModelWithEmbedding(torch.nn.Module):
                def __init__(self, base_model):
                    super().__init__()
                    self.base_model = base_model
                    # Extract the embedding layer from the base model
                    self.embedding = base_model.get_input_embeddings()
                
                def forward(self, embeddings, attention_mask):
                    # Skip the embedding lookup in the model and use provided embeddings directly
                    outputs = self.base_model(inputs_embeds=embeddings, attention_mask=attention_mask)
                    return outputs.logits[:, target_class]
            
            # Create wrapped model
            wrapped_model = ModelWithEmbedding(model)
            
            # Get embeddings from model for the input IDs
            with torch.no_grad():
                # Get the embedding weights
                embedding_layer = model.get_input_embeddings()
                # Convert token IDs to embeddings
                token_embeddings = embedding_layer(single_input_ids)
            
            # Create baseline embeddings based on specified type
            if baseline_type == "zero":
                # Zero embedding - zero tensor with same shape as token_embeddings
                baseline_embeddings = torch.zeros_like(token_embeddings)
            elif baseline_type == "random":
                # Random embeddings - random values with same shape as token_embeddings
                baseline_embeddings = torch.randn_like(token_embeddings)
            else:
                raise ValueError(f"Unknown baseline type: {baseline_type}")
            
            # Initialize IntegratedGradients
            ig = IntegratedGradients(wrapped_model)
            
            # Apply IG to compute attributions
            attributions = ig.attribute(
                inputs=token_embeddings,
                baselines=baseline_embeddings,
                additional_forward_args=(single_attention_mask,),
                n_steps=n_steps,
                internal_batch_size=1  # Process one step at a time
            )
            
            # Sum attributions across embedding dimensions
            token_attributions = attributions.sum(dim=-1).detach().cpu()
            all_token_attributions.append(token_attributions)
            
        except Exception as e:
            print(f"Error processing batch {i}: {e}")
            # Return empty attributions in case of error
            token_attributions = torch.zeros_like(single_input_ids).float().cpu()
            all_token_attributions.append(token_attributions)
    
    # Stack results if multiple examples
    if len(all_token_attributions) > 1:
        return torch.cat(all_token_attributions, dim=0)
    elif len(all_token_attributions) > 0:
        return all_token_attributions[0]
    else:
        # Return empty tensor if no attributions could be computed
        return torch.zeros((0,), device='cpu')


def aggregate_token_attributions(
    attributions,
    tokenizer,
    tokens,
    attention_mask=None,
    top_k=20
):
    """
    Aggregate token attribution scores and identify top tokens.
    
    Args:
        attributions: Attribution scores from Integrated Gradients
        tokenizer: HuggingFace tokenizer
        tokens: List of tokens
        attention_mask: Attention mask to identify valid tokens
        top_k: Number of top tokens to return
        
    Returns:
        top_tokens: List of (token, score) tuples for top tokens
        top_indices: List of indices for top tokens
    """
    # Convert tensor to numpy if needed
    if isinstance(attributions, torch.Tensor):
        attributions = attributions.squeeze().cpu().numpy()
    
    # Process attention mask if provided
    if attention_mask is not None:
        if isinstance(attention_mask, torch.Tensor):
            attention_mask = attention_mask.cpu().numpy()
        
        # Only consider positions where attention_mask is 1
        masked_attributions = attributions.copy()
        masked_attributions[attention_mask == 0] = 0
    else:
        masked_attributions = attributions.copy()
    
    # Filter out special tokens
    special_tokens = {"[CLS]", "[SEP]", "[PAD]", "[UNK]"}
    filtered_attributions = masked_attributions.copy()
    
    for i, token in enumerate(tokens):
        if token in special_tokens:
            filtered_attributions[i] = 0
    
    # Get indices of top-k tokens by absolute attribution value
    top_indices = np.argsort(-np.abs(filtered_attributions))[:top_k]
    
    # Create list of (token, score) tuples
    top_tokens = [(tokens[i], float(filtered_attributions[i])) for i in top_indices]
    
    # Sort by absolute attribution value (descending)
    top_tokens = sorted(top_tokens, key=lambda x: -abs(x[1]))
    
    return top_tokens, top_indices


def visualize_sequence_attributions(
    sequence,
    tokens,
    attributions,
    tokenizer,
    output_path=None,
    highlighted_positions=None
):
    """
    Visualize the token attributions from Integrated Gradients analysis.
    
    Args:
        sequence: Original sequence text (if available)
        tokens: List of tokens from the tokenizer
        attributions: Numpy array of attribution values for each token
        tokenizer: The tokenizer used for the model
        output_path: Path to save the visualization
        highlighted_positions: List of token positions to highlight
        
    Returns:
        None, but saves visualization to output_path if provided
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Filter out any special tokens if needed
    special_tokens = {"[CLS]", "[SEP]", "[PAD]", "[UNK]"}
    
    # Get original sequence if not provided
    if sequence is None:
        # Try to reconstruct from tokens
        sequence = tokenizer.convert_tokens_to_string(tokens)
    
    # Create a more readable version of tokens with attribution scores
    readable_tokens = []
    filtered_attributions = []
    filtered_positions = []
    
    for i, (token, attr) in enumerate(zip(tokens, attributions)):
        # Skip special tokens
        if token in special_tokens:
            continue
        
        readable_tokens.append(token)
        filtered_attributions.append(attr)
        filtered_positions.append(i)
    
    # Handle highlighted positions
    if highlighted_positions is None:
        # If not provided, highlight top 10 tokens by absolute attribution
        abs_attrs = np.abs(filtered_attributions)
        highlighted_positions = np.argsort(-abs_attrs)[:10]
    
    # Create bar chart of attribution scores
    plt.figure(figsize=(20, 6))
    
    # Plot attributions
    bars = plt.bar(
        range(len(filtered_attributions)), 
        filtered_attributions,
        color=['red' if i in highlighted_positions else 'blue' for i in range(len(filtered_attributions))]
    )
    
    # Add token labels
    plt.xticks(
        range(len(readable_tokens)), 
        readable_tokens, 
        rotation=45, 
        ha='right'
    )
    
    # Add title and labels
    plt.title("Token Attribution Scores from Integrated Gradients")
    plt.xlabel("Tokens")
    plt.ylabel("Attribution Score")
    plt.tight_layout()
    
    # Save or show plot
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    # We'll skip the Captum visualization for now as it's causing errors
    # The correct parameters for Captum's visualization may be different


def load_and_preprocess_validation_data(dataset_dir=None, tokenizer=None, max_samples=100):
    """Load and preprocess validation data for IG analysis.
    
    Args:
        dataset_dir: Directory containing datasets
        tokenizer: Tokenizer for encoding sequences
        max_samples: Maximum number of samples to process
    
    Returns:
        TensorDataset with input_ids and attention_mask
    """
    if dataset_dir is None:
        # If no dataset_dir is provided, create a simple synthetic dataset for testing
        print("Using synthetic dataset for testing...")
        sequences = [
            "ACGTACGTACGTACGTACGT" * 10,
            "TGCATGCATGCATGCATGCA" * 10,
            "GTAGTAGTAGTAGTAGTAGT" * 10
        ]
        labels = [1, 0, 1]
        
        # Take only max_samples
        sequences = sequences[:min(len(sequences), max_samples)]
        labels = labels[:min(len(labels), max_samples)]
        
        encoded = tokenizer(
            sequences,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
        input_ids = encoded["input_ids"].long()
        attention_mask = encoded["attention_mask"].long()
        label_tensor = torch.tensor(labels, dtype=torch.long)
        
        return torch.utils.data.TensorDataset(input_ids, attention_mask, label_tensor)
    
    # Load dataset from CSV files
    try:
        import pandas as pd
        
        # Try to find validation files
        validation_files = []
        for filename in os.listdir(dataset_dir):
            if "validation" in filename and filename.endswith(".csv"):
                validation_files.append(os.path.join(dataset_dir, filename))
        
        if not validation_files:
            raise FileNotFoundError("No validation files found in dataset directory")
        
        # Load first validation file
        df = pd.read_csv(validation_files[0])
        
        # Get sequence column name (try common names)
        sequence_col = None
        for col in ["sequence", "Sequence", "seq", "dna", "DNA"]:
            if col in df.columns:
                sequence_col = col
                break
        
        if sequence_col is None:
            # Just use the first column that's not a label
            for col in df.columns:
                if col.lower() not in ["label", "class", "target", "y"]:
                    sequence_col = col
                    break
        
        # Get label column name (try common names)
        label_col = None
        for col in ["label", "Label", "class", "Class", "target", "Target", "y", "Y"]:
            if col in df.columns:
                label_col = col
                break
        
        if label_col is None and len(df.columns) > 1:
            # Just use the last column
            label_col = df.columns[-1]
        
        # If we still don't have label column, create dummy labels
        if label_col is None:
            df["label"] = 1
            label_col = "label"
            
        print(f"Using sequence column: {sequence_col}")
        print(f"Using label column: {label_col}")
        
        # Ensure we have the columns we need
        if sequence_col is None:
            raise ValueError("Could not identify sequence column in validation data")
        
        # Sample data if needed
        if max_samples and max_samples < len(df):
            # Try to balance classes
            if label_col in df.columns:
                # Get equal samples from each class if possible
                classes = df[label_col].unique()
                if len(classes) > 1:
                    samples_per_class = max_samples // len(classes)
                    balanced_samples = []
                    for cls in classes:
                        cls_samples = df[df[label_col] == cls].sample(
                            min(samples_per_class, (df[label_col] == cls).sum())
                        )
                        balanced_samples.append(cls_samples)
                    df = pd.concat(balanced_samples)
                else:
                    df = df.sample(max_samples)
            else:
                df = df.sample(max_samples)
        
        # Extract sequences and labels
        sequences = df[sequence_col].tolist()
        labels = df[label_col].astype(int).tolist()
        
        # Tokenize sequences
        encoded = tokenizer(
            sequences,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
        input_ids = encoded["input_ids"].long()
        attention_mask = encoded["attention_mask"].long()
        label_tensor = torch.tensor(labels, dtype=torch.long)
        
        return torch.utils.data.TensorDataset(input_ids, attention_mask, label_tensor)
        
    except Exception as e:
        print(f"Error loading validation data: {e}")
        print("Using synthetic dataset instead...")
        return load_and_preprocess_validation_data(dataset_dir=None, tokenizer=tokenizer, max_samples=max_samples)


def demo_pretrain_finetune_ig_dist(
    experiment=None,
    pred_type=None,
    error_label=None,
    correct_label="TP",
    splice_type="any",
    output_dir=None,
    checkpoint_dir=None,
    dataset_dir=None,
    n_steps=50,
    max_samples=100,
    batch_size=8,
    local_rank=-1,
    local_world_size=1,
    global_rank=-1,
    model_name="zhihan1996/DNABERT-2-117M",
    **kargs
):
    """Distributed implementation of the IG analysis for error sequence models.
    
    Args:
        experiment: Experiment name
        pred_type: Prediction type to analyze
        error_label: Error class label
        correct_label: Correct class label
        splice_type: Splice type to analyze
        output_dir: Directory to save results
        checkpoint_dir: Directory containing model checkpoint
        dataset_dir: Directory containing datasets
        n_steps: Number of steps for IG analysis
        max_samples: Maximum number of samples to analyze
        batch_size: Batch size
        local_rank: Local rank of this process
        local_world_size: Number of processes per node
        global_rank: Global rank of this process
        model_name: Original pretrained model name
        
    Returns:
        Dict with analysis results
    """
    # Setup device for multi-GPU training
    from .error_sequence_distributed import set_device
    device, local_rank = set_device()
    is_main_process = local_rank in [0, -1]
    
    # Determine model path from experiment if checkpoint_dir not provided
    if checkpoint_dir is None and experiment is not None:
        from .error_sequence_model_dist import ErrorAnalyzer
        analyzer = ErrorAnalyzer(experiment=experiment, model_type="dnabert")
        output_dir = analyzer.set_analysis_output_dir(
            error_label=error_label or pred_type, 
            correct_label=correct_label, 
            splice_type=splice_type
        )
        checkpoint_dir = os.path.join(output_dir, "final_model")
        
    if checkpoint_dir is None:
        raise ValueError("Either checkpoint_dir or experiment must be provided")
    
    if is_main_process:
        print(f"Loading model from {checkpoint_dir}...")
        print(f"Using tokenizer from {model_name}...")
    
    # Load tokenizer from pretrained model (not from checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load model from checkpoint
    try:
        model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir)
    except Exception as e:
        if is_main_process:
            print(f"Error loading model from checkpoint: {e}")
            print(f"Using pretrained model from {model_name} instead...")
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    
    model.eval()  # Set model to evaluation mode
    model = model.to(device)
    
    # Load validation dataset
    print("Loading validation dataset...")
    val_dataset = load_and_preprocess_validation_data(
        dataset_dir=dataset_dir,
        tokenizer=tokenizer,
        max_samples=max_samples
    )
    
    # Create DataLoader with proper distributed sampler
    from torch.utils.data import DataLoader, DistributedSampler
    
    # Initialize sampler for distributed training
    sampler = None
    if is_distributed_training():
        sampler = DistributedSampler(
            val_dataset,
            num_replicas=local_world_size,
            rank=local_rank
        )
    
    # Create the dataloader
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False if sampler else True
    )
    
    # Create output directory if it doesn't exist
    if output_dir and is_main_process:
        os.makedirs(os.path.join(output_dir, "ig_analysis"), exist_ok=True)
    
    # Run IG analysis on each batch
    results = []
    
    # Dictionaries for token frequency analysis
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
    
    # Special tokens to filter out
    special_tokens = {"[CLS]", "[SEP]", "[PAD]", "[UNK]"}
    
    # Settings for token analysis
    TOP_K = 30
    TOP_K_GLOBAL = 50
    
    # Process all validation batches
    print("Running IG analysis...")
    for batch_idx, batch in enumerate(tqdm(val_dataloader, desc="IG Analysis")):
        # Debug batch structure if needed
        if batch_idx == 0 and is_main_process:
            print("[DEBUG] Batch type:", type(batch))
            print("[DEBUG] Batch length:", len(batch))
            for i, item in enumerate(batch):
                print(f"[DEBUG] Batch item {i} type: {type(item)}, shape: {item.shape}")
        
        # Unpack batch data
        input_ids, attention_mask, labels = batch
        batch_size = input_ids.size(0)
        
        # Process each example in the batch
        for i in range(batch_size):
            # Extract single example
            sample_input_ids = input_ids[i:i+1].to(device)
            sample_attention_mask = attention_mask[i:i+1].to(device)
            
            # Handle scalar conversion safely
            try:
                sample_label = int(labels[i].item())
            except:
                # Default to class 1 if there's an issue
                print(f"Warning: Could not convert label to integer for batch {batch_idx}, sample {i}. Using default class 1.")
                sample_label = 1
            
            # Determine label string
            label_str = error_label if sample_label == 1 else correct_label
            
            # Convert tokens to text for visualization
            tokens = tokenizer.convert_ids_to_tokens(sample_input_ids[0].cpu().tolist())
            
            # Skip batch if any token is None (should not happen, but just in case)
            if None in tokens:
                print(f"Warning: None token found in batch {batch_idx}, sample {i}. Skipping.")
                continue
            
            # Original sequence (reconstruct from tokens if needed)
            sequence = tokenizer.decode(sample_input_ids[0], skip_special_tokens=True)
            
            try:
                # Run IG analysis on this single sample
                attributions = run_ig_analysis(
                    model=model,
                    tokenized_examples={
                        "input_ids": sample_input_ids.clone().detach().long(),
                        "attention_mask": sample_attention_mask.clone().detach().long()
                    },
                    device=device,
                    target_class=sample_label,
                    n_steps=n_steps,
                    baseline_type="zero"
                )
                
                # Aggregate attributions
                top_tokens, token_indices = aggregate_token_attributions(
                    attributions=attributions,
                    tokenizer=tokenizer,
                    tokens=tokens,
                    attention_mask=sample_attention_mask[0].cpu().numpy(),
                    top_k=20
                )
                
                # Create visualization
                if output_dir and is_main_process:
                    viz_path = os.path.join(output_dir, "ig_analysis", f"sample_{batch_idx}_{i}.png")
                    visualize_sequence_attributions(
                        sequence=sequence,
                        tokens=tokens,
                        attributions=attributions,
                        tokenizer=tokenizer,
                        output_path=viz_path,
                        highlighted_positions=token_indices
                    )
                
                # Create alignment visualization
                if output_dir and is_main_process:
                    alignment_path = os.path.join(output_dir, "ig_analysis", f"alignment_{batch_idx}_{i}.pdf")
                    
                    # Create a simple annotation dictionary
                    annotation = {
                        "pred_type": label_str,
                    }
                    
                    from .error_sequence_model_dist import plot_alignment_with_ig_scores
                    
                    # Plot alignment visualization
                    plot_alignment_with_ig_scores(
                        sequence,
                        attention_weights=np.abs(attributions),
                        tokenized_sequence=tokens,
                        annotation=annotation,
                        output_path=alignment_path,
                        top_k=TOP_K,
                        color_map="viridis",
                        dynamic_band_height=True,
                        add_legend=True,
                        hide_overlapping_labels=False,
                        token_label_rotation=0,
                        figsize=(20, 6),
                        filter_special_tokens=True,
                        special_tokens=special_tokens,
                        draw_alignment_arrows=True,
                        arrow_line_color="lightgrey",
                        arrow_end_y=0.3,
                        rect_baseline_height=0.3,
                        rect_scale_factor=0.7,
                        show_token_scores=True,
                        show_token_positions=False
                    )
                
                # Update token frequency analysis
                # Get top-k by absolute IG values
                top_positions = np.argsort(-np.abs(attributions))[:TOP_K]
                
                # Collect tokens that are not special tokens
                top_filtered_tokens = []
                for pos_idx in top_positions:
                    # Convert tensor to integer if needed
                    if isinstance(pos_idx, (np.ndarray, torch.Tensor)):
                        pos_idx = int(pos_idx)
                    
                    # Make sure index is valid
                    if 0 <= pos_idx < len(tokens):
                        tk = tokens[pos_idx]
                        if tk not in special_tokens:
                            top_filtered_tokens.append(tk)
                
                # Update global token counts
                num_examples[label_str] += 1
                for tk in top_filtered_tokens:
                    top_token_counts[label_str][tk] += 1
                
                # Store results
                result = {
                    "batch_idx": batch_idx,
                    "sample_idx": i,
                    "true_class": sample_label,
                    "label_str": label_str,
                    "top_tokens": top_tokens
                }
                results.append(result)
                
            except Exception as e:
                print(f"Error processing batch {batch_idx}, sample {i}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    # Perform global token frequency analysis
    if is_main_process and output_dir:
        print("Generating token frequency comparison...")
        from .error_sequence_model_dist import build_token_frequency_comparison, bar_chart_freq
        
        # Create output directory for token frequency analysis
        freq_dir = os.path.join(output_dir, "token_frequency")
        os.makedirs(freq_dir, exist_ok=True)
        
        # Generate global comparison
        chart_path = os.path.join(freq_dir, f"global_token_frequency_{error_label}_vs_{correct_label}.pdf")
        global_comparison = build_token_frequency_comparison(
            top_token_counts, 
            num_examples, 
            error_label=error_label, 
            correct_label=correct_label, 
            verbose=1
        )
        
        # Print frequency results
        print(f"\n=== {error_label} vs {correct_label} token frequencies ===")
        print("token\terror_freq\tcorrect_freq\tdiff")
        for tk, ef, cf, df in global_comparison[:TOP_K_GLOBAL]:
            print(f"{tk}\t{ef:.3f}\t{cf:.3f}\t{df:.3f}")
        
        # Generate bar chart
        bar_chart_freq(
            global_comparison, 
            error_label=error_label, 
            correct_label=correct_label, 
            top_n=TOP_K_GLOBAL, 
            output_path=chart_path
        )
    
    print(f"Completed IG analysis on {len(results)} samples")
    
    # Return results
    return {
        "results": results,
        "top_token_counts": top_token_counts,
        "num_examples": num_examples
    }
