"""
Sequence-based error model interface.

This module provides a clean interface to the existing error sequence model functionality,
serving as a bridge between the new package structure and the existing implementation.
"""

import os
import sys
from typing import Dict, List, Tuple, Union, Optional, Any, Callable

# Import from existing modules - only import functions that actually exist
from ..error_sequence_model import (
    train_transformer_model,  # Instead of train_validate_sequence_model
    # analyze_error_tokens and explain_sequence_model_errors don't exist
    # Importing alternative relevant functions:
    get_attention_weights,
    visualize_attention_regions
)


class ErrorSequenceModel:
    """
    Wrapper for sequence-based error model functionality.
    
    This class provides a clean object-oriented interface to the existing
    sequence-based error model functionality, making it easier to train
    and analyze error models for splice site prediction.
    
    Parameters
    ----------
    error_label : str
        Label for error cases (e.g., "FP" for False Positives)
    correct_label : str
        Label for correct cases (e.g., "TP" for True Positives)
    splice_type : str
        Type of splice site ("donor", "acceptor", or "any")
    experiment : str, optional
        Experiment name for organizing results
    model_type : str, default="transformer"
        Type of sequence model to use
    output_dir : str, optional
        Directory for saving model and results
    verbose : int, default=1
        Verbosity level
    """
    
    def __init__(
        self,
        error_label: str,
        correct_label: str,
        splice_type: str,
        experiment: Optional[str] = None,
        model_type: str = "transformer",
        output_dir: Optional[str] = None,
        verbose: int = 1
    ):
        self.error_label = error_label
        self.correct_label = correct_label
        self.splice_type = splice_type
        self.experiment = experiment
        self.model_type = model_type
        self.verbose = verbose
        
        # Configure output directory
        if output_dir is None and experiment is not None:
            self.output_dir = os.path.join(
                "results",
                experiment,
                splice_type,
                f"{error_label.lower()}_vs_{correct_label.lower()}",
                "sequence_model"
            )
        else:
            self.output_dir = output_dir
            
        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
            
        # Initialize model and results
        self.model = None
        self.tokenizer = None
        self.results = None
        
    def train(
        self,
        train_dataset=None,
        val_dataset=None,
        model_name: str = "zhihan1996/DNABERT-2-117M",
        num_labels: int = 2,
        num_train_epochs: int = 3,
        **kwargs
    ):
        """
        Train the sequence-based error model.
        
        Parameters
        ----------
        train_dataset : Dataset
            Training dataset (if None, will be loaded from data sources)
        val_dataset : Dataset
            Validation dataset (if None, will be split from train_dataset)
        model_name : str, default="zhihan1996/DNABERT-2-117M"
            Name of pre-trained model to use
        num_labels : int, default=2
            Number of output labels
        num_train_epochs : int, default=3
            Number of training epochs
        **kwargs : dict
            Additional parameters for training
            
        Returns
        -------
        dict
            Results from training
        """
        # Call the existing training function
        model, trainer = train_transformer_model(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            model_name=model_name,
            num_labels=num_labels,
            num_train_epochs=num_train_epochs,
            **kwargs
        )
        
        self.model = model
        # Tokenizer is typically part of the train_transformer_model result
        # but we'll handle that in a future update
        
        # Store results for future reference
        self.results = {
            "model": model,
            "trainer": trainer
        }
        
        return self.results
        
    def analyze_errors(
        self,
        sequence: str,
        layer: Optional[int] = None,
        head: Optional[int] = None,
        max_length: int = 512,
        policy: str = "mean",
        **kwargs
    ):
        """
        Analyze errors using attention mechanisms.
        
        Parameters
        ----------
        sequence : str
            Input sequence to analyze
        layer : int, optional
            Specific transformer layer (None = aggregate across all layers)
        head : int, optional
            Specific attention head (None = average across all heads)
        max_length : int, default=512
            Maximum sequence length for tokenization
        policy : str, default="mean"
            Aggregation policy ("mean", "rollout")
        **kwargs : dict
            Additional parameters for analysis
            
        Returns
        -------
        dict
            Analysis results including attention weights and tokenized sequence
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call train() first.")
            
        # Use get_attention_weights instead of analyze_error_tokens
        attention_weights, tokenized_sequence = get_attention_weights(
            self.model,
            self.tokenizer,
            sequence,
            layer=layer,
            head=head,
            max_length=max_length,
            policy=policy,
            **kwargs
        )
        
        # Store results for future reference
        analysis_results = {
            "attention_weights": attention_weights,
            "tokenized_sequence": tokenized_sequence
        }
        
        return analysis_results
        
    def explain_errors(
        self,
        original_sequence: str,
        attention_weights,
        tokenized_sequence,
        annotation: Optional[Dict] = None,
        output_path: Optional[str] = None,
        top_k: int = 10,
        color_map: str = "viridis",
        dynamic_band_height: bool = True,
        add_legend: bool = True,
        **kwargs
    ):
        """
        Explain errors by visualizing attention regions.
        
        Parameters
        ----------
        original_sequence : str
            The full DNA sequence
        attention_weights : array
            Attention scores (1D or 2D array)
        tokenized_sequence : list
            Tokenized representation of the sequence
        annotation : dict, optional
            Optional annotation for the sequence
        output_path : str, optional
            File path to save the visualization
        top_k : int, default=10
            Number of top regions to highlight
        color_map : str, default="viridis"
            Colormap for the visualization
        dynamic_band_height : bool, default=True
            Adjust band heights based on attention scores
        add_legend : bool, default=True
            Include a legend summarizing the top tokens
        **kwargs : dict
            Additional parameters for visualization
            
        Returns
        -------
        None
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call train() first.")
            
        # Use visualize_attention_regions instead of explain_sequence_model_errors
        visualize_attention_regions(
            original_sequence,
            attention_weights,
            tokenized_sequence,
            annotation=annotation,
            output_path=output_path,
            top_k=top_k,
            color_map=color_map,
            dynamic_band_height=dynamic_band_height,
            add_legend=add_legend,
            **kwargs
        )
        
        return None


class DistributedErrorSequenceModel(ErrorSequenceModel):
    """
    Distributed training interface for sequence-based error models.
    
    This class extends ErrorSequenceModel to support distributed training
    across multiple GPUs using the error_sequence_distributed.py implementation.
    """
    
    def train_distributed(
        self,
        model_type: str = "transformer",
        n_epochs: int = 10,
        batch_size: int = 32,
        learning_rate: float = 5e-5,
        num_gpus: int = 2,
        **kwargs
    ) -> Dict:
        """
        Train a sequence-based error model using distributed training.
        
        Parameters
        ----------
        model_type : str, default="transformer"
            Type of model architecture to use (transformer preferred for distributed)
        n_epochs : int, default=10
            Number of training epochs
        batch_size : int, default=32
            Batch size for training (per GPU)
        learning_rate : float, default=5e-5
            Learning rate for optimizer
        num_gpus : int, default=2
            Number of GPUs to use for distributed training
        **kwargs : dict
            Additional keyword arguments passed to the distributed training function
            
        Returns
        -------
        dict
            Results from distributed training
        """
        from ..error_sequence_distributed import train_distributed
        
        # Call the distributed training function
        model, tokenizer, results = train_distributed(
            error_label=self.error_label,
            correct_label=self.correct_label,
            splice_type=self.splice_type,
            output_dir=self.output_dir,
            model_type=model_type,
            n_epochs=n_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            num_gpus=num_gpus,
            verbose=self.verbose,
            **kwargs
        )
        
        self.model = model
        self.tokenizer = tokenizer
        self.results = results
        
        return results
