"""
Transformer model trainer for deep error models.

This module handles fine-tuning of pre-trained DNA language models
(DNABERT, HyenaDNA) for splice site error classification.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
from pathlib import Path
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    TrainerCallback
)
from transformers.modeling_outputs import SequenceClassifierOutput
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import numpy as np

# Add project root to path for imports
project_root = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(project_root))

from meta_spliceai.splice_engine.meta_models.error_model.config import ErrorModelConfig
from meta_spliceai.splice_engine.meta_models.error_model.dataset.data_utils import DNASequenceDataset, DataUtils


class MultiModalTransformerModel(nn.Module):
    """
    Transformer model that combines sequence embeddings with additional features.
    
    This model extends a pre-trained DNA transformer to include additional
    numerical features (base model scores, derived features, genomic features).
    """
    
    def __init__(
        self,
        model_name: str,
        num_additional_features: int = 0,
        num_labels: int = 2,
        dropout_rate: float = 0.1
    ):
        """
        Initialize multi-modal transformer model.
        
        Parameters
        ----------
        model_name : str
            Name of pre-trained transformer model
        num_additional_features : int, default 0
            Number of additional numerical features
        num_labels : int, default 2
            Number of output classes
        dropout_rate : float, default 0.1
            Dropout rate for additional feature layers
        """
        super().__init__()
        
        # Load pre-trained transformer
        self.transformer = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            output_hidden_states=True
        )
        
        # Get hidden size from transformer
        self.hidden_size = self.transformer.config.hidden_size
        
        # Additional feature processing
        self.num_additional_features = num_additional_features
        if num_additional_features > 0:
            self.feature_projection = nn.Sequential(
                nn.Linear(num_additional_features, self.hidden_size // 2),  # e.g., 42 → 384
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(self.hidden_size // 2, self.hidden_size // 4)  # e.g., 384 → 192
            )
            
            # Combined classifier
            combined_size = self.hidden_size + self.hidden_size // 4
            self.classifier = nn.Sequential(
                nn.Linear(combined_size, self.hidden_size // 2),  # e.g., 960 → 384
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(self.hidden_size // 2, num_labels)  # e.g., 384 → 2
            )
        else:
            # Use transformer's original classifier
            self.classifier = None
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        additional_features: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> SequenceClassifierOutput:
        """Forward pass through the model."""
        
        # Get transformer outputs
        transformer_outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        
        # Get sequence representation (model-agnostic approach)
        sequence_repr = self._extract_sequence_representation(transformer_outputs, input_ids)
        # Handles different foundation model architectures automatically

        if self.num_additional_features > 0 and additional_features is not None:
            # Process additional features
            feature_repr = self.feature_projection(additional_features)  # [batch_size, hidden_size//4]
            # Input:  [batch_size, num_additional_features] (e.g., [16, 42])
            # Output: [batch_size, hidden_size//4] (e.g., [16, 192])
            
            # Combine representations
            combined_repr = torch.cat([sequence_repr, feature_repr], dim=1)
            # sequence_repr: [batch_size, hidden_size] (e.g., [16, 768])
            # feature_repr:  [batch_size, hidden_size//4] (e.g., [16, 192])
            # combined_repr: [batch_size, hidden_size + hidden_size//4] (e.g., [16, 960])
            
            # Get logits
            logits = self.classifier(combined_repr)
            # Input:  [batch_size, 960]
            # Output: [batch_size, 2] (binary classification: TP vs FP/FN)
        else:
            # Use transformer's logits directly
            logits = transformer_outputs.logits
        
        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.transformer.config.num_labels), labels.view(-1))
        
        # Return SequenceClassifierOutput for compatibility with HuggingFace Trainer
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions
        )
    
    def _extract_sequence_representation(self, transformer_outputs, input_ids):
        """
        Extract sequence representation in a model-agnostic way.
        
        Handles different foundation model architectures:
        - BERT-style (DNABERT, ESM): Uses CLS token at position 0
        - HyenaDNA, Caduceus: Uses global average pooling
        - T5-based: Uses encoder outputs with average pooling
        - Causal models: Uses last position
        
        Parameters
        ----------
        transformer_outputs : transformers output object
            Outputs from the transformer model
        input_ids : torch.Tensor
            Input token IDs [batch_size, seq_length]
            
        Returns
        -------
        torch.Tensor
            Sequence representation [batch_size, hidden_size]
        """
        last_hidden_state = transformer_outputs.hidden_states[-1]  # [batch_size, seq_length, hidden_size]
        
        # Detect model architecture based on model name/config
        model_name = self.transformer.config._name_or_path.lower() if hasattr(self.transformer.config, '_name_or_path') else ""
        model_type = getattr(self.transformer.config, 'model_type', '').lower()
        
        # BERT-style models (DNABERT, DNABERT-2, ESM, etc.)
        if any(bert_indicator in model_name for bert_indicator in ['bert', 'esm']) or model_type in ['bert', 'esm']:
            # Use CLS token at position 0
            sequence_repr = last_hidden_state[:, 0, :]  # [batch_size, hidden_size]
            
        # HyenaDNA and similar convolution/state-space models
        elif any(hyena_indicator in model_name for hyena_indicator in ['hyena', 'caduceus', 'mamba']) or model_type in ['hyena', 'mamba']:
            # Use global average pooling (excluding padding tokens)
            attention_mask = transformer_outputs.get('attention_mask', torch.ones_like(input_ids))
            mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            sum_embeddings = torch.sum(last_hidden_state * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            sequence_repr = sum_embeddings / sum_mask  # [batch_size, hidden_size]
            
        # T5-based models (Nucleotide Transformer)
        elif 't5' in model_name or model_type == 't5':
            # Use average pooling of encoder outputs
            attention_mask = transformer_outputs.get('attention_mask', torch.ones_like(input_ids))
            mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            sum_embeddings = torch.sum(last_hidden_state * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            sequence_repr = sum_embeddings / sum_mask  # [batch_size, hidden_size]
            
        # Causal/GPT-style models
        elif any(gpt_indicator in model_name for gpt_indicator in ['gpt', 'llama', 'mistral']) or model_type in ['gpt2', 'llama']:
            # Use last non-padding position
            sequence_lengths = input_ids.ne(self.transformer.config.pad_token_id).sum(dim=1) - 1
            sequence_repr = last_hidden_state[torch.arange(last_hidden_state.size(0)), sequence_lengths]
            
        else:
            # Default fallback: try CLS token first, then average pooling
            try:
                # Try CLS token approach
                sequence_repr = last_hidden_state[:, 0, :]  # [batch_size, hidden_size]
            except:
                # Fallback to average pooling
                sequence_repr = last_hidden_state.mean(dim=1)  # [batch_size, hidden_size]
                
        return sequence_repr


class TransformerTrainer:
    """
    Trainer for transformer-based error classification models.
    
    This class handles:
    1. Model initialization and configuration
    2. Training loop with validation
    3. Model evaluation and metrics
    4. Checkpointing and model saving
    """
    
    def __init__(self, config: ErrorModelConfig):
        """
        Initialize transformer trainer.
        
        Parameters
        ----------
        config : ErrorModelConfig
            Configuration for training
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
        
        # Add special tokens if needed
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Device setup
        self.device = self._setup_device()
        
        # Model will be initialized during training
        self.model = None
        self.trainer = None
        
    def _setup_device(self) -> torch.device:
        """
        Setup computation device with comprehensive CPU/single-GPU/multi-GPU support.
        
        Returns
        -------
        torch.device
            Primary device for model placement
        """
        import torch
        
        # Detect available hardware
        cuda_available = torch.cuda.is_available()
        num_gpus = torch.cuda.device_count() if cuda_available else 0
        
        self.logger.info(f"Hardware detection: CUDA available: {cuda_available}, GPUs: {num_gpus}")
        
        if self.config.device == "auto":
            if num_gpus > 1:
                # Multi-GPU setup
                device = torch.device("cuda:0")  # Primary GPU
                self.use_multi_gpu = True
                self.logger.info(f"Auto-detected multi-GPU setup: {num_gpus} GPUs available")
            elif num_gpus == 1:
                # Single GPU setup
                device = torch.device("cuda:0")
                self.use_multi_gpu = False
                self.logger.info("Auto-detected single GPU setup")
            else:
                # CPU-only setup
                device = torch.device("cpu")
                self.use_multi_gpu = False
                self.logger.info("Auto-detected CPU-only setup")
                
        elif self.config.device == "cpu":
            # Force CPU mode
            device = torch.device("cpu")
            self.use_multi_gpu = False
            self.logger.info("Forced CPU mode")
            
        elif self.config.device.startswith("cuda"):
            # Specific GPU or multi-GPU mode
            if self.config.device == "cuda" and num_gpus > 1:
                # Multi-GPU mode
                device = torch.device("cuda:0")
                self.use_multi_gpu = True
                self.logger.info(f"Multi-GPU mode: using {num_gpus} GPUs")
            else:
                # Single specific GPU
                device = torch.device(self.config.device)
                self.use_multi_gpu = False
                self.logger.info(f"Single GPU mode: {device}")
        else:
            # Fallback
            device = torch.device("cpu")
            self.use_multi_gpu = False
            self.logger.warning(f"Unknown device '{self.config.device}', falling back to CPU")
        
        # Store device info for later use
        self.num_gpus = num_gpus
        self.primary_device = device
        
        return device
    
    def _log_training_configuration(self, train_dataset: Dataset, num_train_samples: int, val_dataset: Dataset = None):
        """
        Log comprehensive training configuration and parameters.
        
        Parameters
        ----------
        train_dataset : Dataset
            Training dataset to analyze
        num_train_samples : int
            Number of training samples
        val_dataset : Dataset, optional
            Validation dataset
        """
        import numpy as np
        
        # Analyze context lengths in the dataset
        sequence_lengths = []
        if hasattr(train_dataset, 'sequences') and train_dataset.sequences:
            sequence_lengths = [len(seq) for seq in train_dataset.sequences[:1000]]  # Sample first 1000
        
        self.logger.info("="*80)
        self.logger.info("TRAINING CONFIGURATION SUMMARY")
        self.logger.info("="*80)
        
        # Model Architecture
        self.logger.info("\nMODEL ARCHITECTURE:")
        self.logger.info("-"*40)
        self.logger.info(f"  Base Model:          {self.config.model_name}")
        self.logger.info(f"  Model Type:          Transformer + MLP Classifier")
        self.logger.info(f"  Hidden Size:         {getattr(self.model.transformer.config, 'hidden_size', 'N/A')}")
        self.logger.info(f"  Num Layers:          {getattr(self.model.transformer.config, 'num_hidden_layers', 'N/A')}")
        self.logger.info(f"  Num Attention Heads: {getattr(self.model.transformer.config, 'num_attention_heads', 'N/A')}")
        self.logger.info(f"  Vocab Size:          {getattr(self.model.transformer.config, 'vocab_size', 'N/A')}")
        self.logger.info(f"  Max Position Embeds: {getattr(self.model.transformer.config, 'max_position_embeddings', 512)}")
        
        # Context Length Analysis
        if sequence_lengths:
            self.logger.info("\nCONTEXT LENGTH ANALYSIS:")
            self.logger.info("-"*40)
            self.logger.info(f"  Training Context:    {self.config.context_length} nt (±{self.config.context_length//2} around splice site)")
            self.logger.info(f"  Max Tokenization:    {self.config.max_length} tokens")
            self.logger.info(f"  Extracted Sequences:")
            self.logger.info(f"    - Average:         {np.mean(sequence_lengths):.1f} nt")
            self.logger.info(f"    - Median:          {np.median(sequence_lengths):.1f} nt")
            self.logger.info(f"    - Min:             {np.min(sequence_lengths)} nt")
            self.logger.info(f"    - Max:             {np.max(sequence_lengths)} nt")
            self.logger.info(f"    - Std Dev:         {np.std(sequence_lengths):.1f} nt")
            # Note: sequence_lengths should be the extracted context (200nt), not raw artifacts
            if any(l != self.config.context_length for l in sequence_lengths[:100]):
                self.logger.info(f"  ⚠️  Context length variation detected (expected {self.config.context_length} nt)")
        
        # Training Hyperparameters
        self.logger.info("\nTRAINING HYPERPARAMETERS:")
        self.logger.info("-"*40)
        self.logger.info(f"  Learning Rate:       {self.config.learning_rate}")
        self.logger.info(f"  Batch Size:          {self.config.batch_size}")
        self.logger.info(f"  Num Epochs:          {self.config.num_epochs}")
        self.logger.info(f"  Warmup Steps:        {self.config.warmup_steps}")
        self.logger.info(f"  Weight Decay:        {self.config.weight_decay}")
        self.logger.info(f"  Gradient Accumulation: {getattr(self.config, 'gradient_accumulation_steps', 1)} steps")
        self.logger.info(f"  FP16 Training:       {getattr(self.config, 'fp16', False)}")
        self.logger.info(f"  Max Grad Norm:       {getattr(self.config, 'max_grad_norm', 1.0)}")
        
        # Optimization Details
        self.logger.info("\nOPTIMIZATION DETAILS:")
        self.logger.info("-"*40)
        self.logger.info(f"  Optimizer:           AdamW")
        self.logger.info(f"  Loss Function:       CrossEntropyLoss")
        self.logger.info(f"  LR Scheduler:        Linear warmup + decay")
        self.logger.info(f"  Evaluation Strategy: Every {getattr(self.config, 'evaluation_steps', 100)} steps")
        self.logger.info(f"  Save Strategy:       Every {getattr(self.config, 'save_steps', 500)} steps")
        early_stopping = getattr(self.config, 'early_stopping', False)
        self.logger.info(f"  Early Stopping:      {'Enabled' if early_stopping else 'Disabled'}")
        if early_stopping:
            self.logger.info(f"    - Patience:        {getattr(self.config, 'patience', 3)} evaluations")
            self.logger.info(f"    - Min Delta:       {getattr(self.config, 'min_delta', 0.001)}")
        
        # Data Configuration
        self.logger.info("\nDATA CONFIGURATION:")
        self.logger.info("-"*40)
        self.logger.info(f"  Train Samples:       {num_train_samples:,}")
        self.logger.info(f"  Validation Samples:  {len(val_dataset) if val_dataset else 'N/A'}")
        self.logger.info(f"  Feature Dimension:   {getattr(train_dataset, 'feature_dim', 'N/A')}")
        self.logger.info(f"  Class Distribution:")
        if hasattr(train_dataset, 'labels'):
            unique, counts = np.unique(train_dataset.labels, return_counts=True)
            for label, count in zip(unique, counts):
                self.logger.info(f"    - Class {label}:        {count:,} ({100*count/num_train_samples:.1f}%)")
        
        # Hardware Configuration
        self.logger.info("\nHARDWARE CONFIGURATION:")
        self.logger.info("-"*40)
        self.logger.info(f"  Device:              {self.device}")
        self.logger.info(f"  Multi-GPU:           {self.use_multi_gpu}")
        if self.use_multi_gpu:
            self.logger.info(f"  Number of GPUs:      {self.num_gpus}")
        
        # Memory Estimates (rough)
        param_count = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info("\nMODEL SIZE:")
        self.logger.info("-"*40)
        self.logger.info(f"  Total Parameters:    {param_count:,}")
        self.logger.info(f"  Trainable Params:    {trainable_params:,}")
        self.logger.info(f"  Model Memory:        ~{param_count * 4 / 1024**3:.2f} GB (FP32)")
        
        self.logger.info("="*80 + "\n")
    
    def train(
        self,
        train_dataset: Dataset,
        val_dataset: Dataset = None,
        output_dir: str = None
    ) -> Dict[str, Any]:
        """
        Train the transformer model.
        
        Parameters
        ----------
        train_dataset : Dataset
            Training dataset
        val_dataset : Dataset, optional
            Validation dataset
        output_dir : str, optional
            Output directory for checkpoints and logs
            
        Returns
        -------
        Dict[str, Any]
            Training results and metrics
        """
        self.logger.info("Starting transformer training...")
        
        # Setup output directory
        if output_dir is None:
            output_dir = self.config.get_output_dir()
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup experiment tracking
        tracking_config = self.config.setup_experiment_tracking()
        self.logger.info(f"Experiment tracking config: {tracking_config}")
        
        # Initialize model
        num_additional_features = 0
        if train_dataset.additional_features is not None:
            num_additional_features = train_dataset.additional_features.shape[1]
        
        # Log feature summary if we have additional features
        if num_additional_features > 0 and hasattr(train_dataset, 'feature_names'):
            from meta_spliceai.splice_engine.meta_models.error_model.dataset.data_utils import DataUtils
            from transformers import AutoConfig
            # Get hidden size from config to calculate feature embedding dimension
            model_config = AutoConfig.from_pretrained(self.config.model_name)
            feature_dim_after_transform = model_config.hidden_size // 4
            DataUtils.log_feature_summary(
                feature_columns=train_dataset.feature_names,
                num_samples=len(train_dataset),
                feature_dim_after_transform=feature_dim_after_transform
            )
        
        self.model = MultiModalTransformerModel(
            model_name=self.config.model_name,
            num_additional_features=num_additional_features,
            num_labels=2,  # Binary classification
            dropout_rate=0.1
        )

        # Inspect architecture - comprehensive diagnostics
        print("=" * 80)
        print("MULTI-MODAL TRANSFORMER ARCHITECTURE")
        print("=" * 80)
        print(f"Model: {self.config.model_name}")
        print(f"Hidden size: {self.model.hidden_size}")
        print(f"Num layers: {self.model.transformer.config.num_hidden_layers}")
        print(f"Max sequence length: {self.config.max_length}")
        print(f"Device: {self.device}")
        print()
        print("ARCHITECTURE SUMMARY:")
        print(f"  Sequence embedding: [batch_size, {self.model.hidden_size}]")
        if self.model.num_additional_features > 0:
            feature_dim = self.model.hidden_size // 4
            combined_dim = self.model.hidden_size + feature_dim
            print(f"  Feature embedding:  [batch_size, {feature_dim}]")
            print(f"  Combined embedding: [batch_size, {combined_dim}]")
            print(f"  Classification:     [batch_size, 2] (TP vs Error)")
        else:
            print(f"  Classification:     [batch_size, 2] (sequence-only mode)")
        print("=" * 80)
        
        # Move model to device and setup multi-GPU if available
        self.model.to(self.device)
        
        # Setup multi-GPU training if available
        if self.use_multi_gpu and self.num_gpus > 1:
            self.logger.info(f"Wrapping model with DataParallel for {self.num_gpus} GPUs")
            self.model = torch.nn.DataParallel(self.model)
            # Adjust batch size for multi-GPU
            effective_batch_size = self.config.batch_size * self.num_gpus
            self.logger.info(f"Effective batch size with {self.num_gpus} GPUs: {effective_batch_size}")
        
        # Flag to show batch diagnostics only once
        self._batch_diagnostics_shown = False
        
        # Log comprehensive training configuration
        self._log_training_configuration(train_dataset, len(train_dataset), val_dataset)
        
        # Setup training arguments with multi-GPU support
        training_args = TrainingArguments(
            output_dir=str(output_dir / "checkpoints"),
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            warmup_steps=self.config.warmup_steps,
            weight_decay=self.config.weight_decay,
            learning_rate=self.config.learning_rate,
            logging_dir=str(output_dir / "logs"),
            logging_steps=100,
            evaluation_strategy="steps",
            eval_steps=500,
            save_strategy="steps",
            save_steps=500,
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_f1",
            greater_is_better=True,
            report_to=self.config.tracking_integrations,  # Use configured tracking integrations
            dataloader_num_workers=self.config.num_workers,
            dataloader_pin_memory=self.config.pin_memory,
            remove_unused_columns=False,
            # Multi-GPU configuration
            dataloader_drop_last=False,  # Keep all samples
            ddp_find_unused_parameters=False,  # Optimize for performance
            # Automatic mixed precision for better GPU utilization
            fp16=torch.cuda.is_available() and self.config.device != "cpu" and self.config.use_mixed_precision,
        )
        
        # Compute class weights for imbalanced data
        train_labels = [train_dataset[i]['labels'].item() for i in range(len(train_dataset))]
        class_weights = DataUtils.compute_class_weights(train_labels)
        
        # Custom callback to show batch diagnostics once
        class BatchDiagnosticsCallback(TrainerCallback):
            def __init__(self, trainer_instance):
                self.trainer_instance = trainer_instance
                
            def on_step_begin(self, args, state, control, **kwargs):
                if not self.trainer_instance._batch_diagnostics_shown and state.global_step == 0:
                    # Get a sample batch to show actual dimensions
                    train_dataloader = kwargs.get('train_dataloader')
                    if train_dataloader is not None:
                        sample_batch = next(iter(train_dataloader))
                        print()
                        print("ACTUAL BATCH DIMENSIONS (First Training Batch):")
                        print(f"  Input IDs shape:        {sample_batch['input_ids'].shape}")
                        print(f"  Attention mask shape:   {sample_batch['attention_mask'].shape}")
                        if 'additional_features' in sample_batch and sample_batch['additional_features'] is not None:
                            print(f"  Additional features:    {sample_batch['additional_features'].shape}")
                        else:
                            print(f"  Additional features:    None (sequence-only mode)")
                        print(f"  Labels shape:           {sample_batch['labels'].shape}")
                        print("=" * 80)
                        self.trainer_instance._batch_diagnostics_shown = True
        
        # Initialize trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self._compute_metrics,
            callbacks=[
                EarlyStoppingCallback(early_stopping_patience=3),
                BatchDiagnosticsCallback(self)
            ],
        )
        
        # Train model
        self.logger.info("Starting training...")
        train_result = self.trainer.train()
        
        # Save final model
        self.trainer.save_model(str(output_dir / "final_model"))
        self.tokenizer.save_pretrained(str(output_dir / "final_model"))
        
        # Save training configuration
        config_path = output_dir / "training_config.json"
        with open(config_path, 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2, default=str)
        
        self.logger.info(f"Training completed. Model saved to {output_dir / 'final_model'}")
        
        return {
            'train_result': train_result,
            'model_path': output_dir / "final_model",
            'config_path': config_path
        }
    
    def evaluate(
        self,
        test_dataset: DNASequenceDataset,
        model_path: Optional[Path] = None
    ) -> Dict[str, float]:
        """
        Evaluate the trained model.
        
        Parameters
        ----------
        test_dataset : DNASequenceDataset
            Test dataset
        model_path : Path, optional
            Path to saved model. If None, uses current model.
            
        Returns
        -------
        Dict[str, float]
            Evaluation metrics
        """
        self.logger.info("Evaluating model...")
        
        # Load model if path provided
        if model_path is not None:
            self.load_model(model_path)
        
        if self.model is None:
            raise ValueError("No model available for evaluation")
        
        # Evaluate using trainer
        if self.trainer is not None:
            eval_result = self.trainer.evaluate(test_dataset)
        else:
            # Manual evaluation
            eval_result = self._manual_evaluate(test_dataset)
        
        self.logger.info(f"Evaluation results: {eval_result}")
        return eval_result
    
    def _manual_evaluate(self, dataset: DNASequenceDataset) -> Dict[str, float]:
        """Manual evaluation when trainer is not available."""
        self.model.eval()
        
        dataloader = DataUtils.create_data_loader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory
        )
        
        all_predictions = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for batch in dataloader:
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(**batch)
                logits = outputs['logits']
                
                # Get predictions and probabilities
                probs = torch.softmax(logits, dim=-1)
                predictions = torch.argmax(logits, dim=-1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(batch['labels'].cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of positive class
        
        # Compute metrics
        metrics = self._compute_metrics_from_arrays(
            np.array(all_predictions),
            np.array(all_labels),
            np.array(all_probs)
        )
        
        return metrics
    
    def _compute_metrics(self, eval_pred) -> Dict[str, float]:
        """Compute metrics for trainer."""
        predictions, labels = eval_pred
        
        # Handle case where predictions might be a tuple (logits, labels)
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        
        # Ensure predictions is a numpy array with the right shape
        if not isinstance(predictions, np.ndarray):
            predictions = np.array(predictions)
        
        # Get predicted classes
        pred_classes = np.argmax(predictions, axis=1)
        
        # Get probabilities for AUC
        probs = torch.softmax(torch.tensor(predictions), dim=-1)[:, 1].numpy()
        
        return self._compute_metrics_from_arrays(pred_classes, labels, probs)
    
    def _compute_metrics_from_arrays(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        probs: np.ndarray
    ) -> Dict[str, float]:
        """Compute metrics from arrays."""
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
        auc = roc_auc_score(labels, probs)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc
        }
    
    def load_model(self, model_path: Path) -> None:
        """Load a saved model."""
        self.logger.info(f"Loading model from {model_path}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        
        # Load model configuration
        config_path = model_path / "training_config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                saved_config = json.load(f)
            # Update current config with saved parameters
            for key, value in saved_config.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
        
        # Initialize and load model
        self.model = MultiModalTransformerModel(
            model_name=str(model_path),
            num_additional_features=0,  # Will be updated based on data
            num_labels=2
        )
        
        self.model.to(self.device)
        self.model.eval()
    
    def predict(
        self,
        sequences: List[str],
        additional_features: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions on new sequences.
        
        Parameters
        ----------
        sequences : List[str]
            DNA sequences to predict
        additional_features : np.ndarray, optional
            Additional features for sequences
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (predictions, probabilities)
        """
        if self.model is None:
            raise ValueError("No model loaded for prediction")
        
        self.model.eval()
        
        # Create temporary dataset
        dummy_labels = [0] * len(sequences)  # Dummy labels
        temp_dataset = DNASequenceDataset(
            sequences=sequences,
            labels=dummy_labels,
            tokenizer=self.tokenizer,
            max_length=self.config.max_length,
            additional_features=additional_features
        )
        
        # Create dataloader
        dataloader = DataUtils.create_data_loader(
            temp_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory
        )
        
        all_predictions = []
        all_probs = []
        
        with torch.no_grad():
            for batch in dataloader:
                # Remove labels from batch
                batch = {k: v.to(self.device) for k, v in batch.items() if k != 'labels'}
                
                # Forward pass
                outputs = self.model(**batch)
                logits = outputs['logits']
                
                # Get predictions and probabilities
                probs = torch.softmax(logits, dim=-1)
                predictions = torch.argmax(logits, dim=-1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        return np.array(all_predictions), np.array(all_probs)
