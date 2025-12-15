"""
Training and evaluation pipeline for splice site prediction model.

This module handles the training, evaluation, and benchmarking of
splice site prediction models using PyTorch.
"""

import os
import time
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report, roc_auc_score
)

from .models import create_model

logger = logging.getLogger(__name__)


class SpliceSiteLoss(nn.Module):
    """
    Loss function for splice site prediction with three classes (donor, acceptor, neither).
    Uses a combination of cross-entropy and focal loss to handle class imbalance.
    """
    
    def __init__(self, gamma=2.0, class_weights=None, reduction='mean'):
        """
        Initialize splice site loss function.
        
        Args:
            gamma: Focusing parameter for focal loss that adjusts the rate at which easy examples are down-weighted
            class_weights: Optional tensor of shape (3,) with weights for [donor, acceptor, neither] classes
                           If None, weights are automatically computed based on class frequency
            reduction: Specifies the reduction to apply to the output ('none', 'mean', 'sum')
        """
        super(SpliceSiteLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.class_weights = class_weights
    
    def forward(self, inputs, targets):
        """
        Forward pass for splice site loss.
        
        Args:
            inputs: Predicted probabilities with shape (batch_size, seq_len, 3) for [donor, acceptor, neither]
            targets: Ground truth one-hot encoded labels with shape (batch_size, seq_len, 3)
            
        Returns:
            Loss value
        """
        # Ensure inputs are valid probabilities
        epsilon = 1e-7
        inputs = torch.clamp(inputs, epsilon, 1 - epsilon)
        
        # Calculate cross entropy manually for more control
        ce_loss = -torch.sum(targets * torch.log(inputs), dim=2)  # Shape: (batch_size, seq_len)
        
        # Apply focal loss modulation
        p_t = torch.exp(-ce_loss)
        focal_loss = (1 - p_t) ** self.gamma * ce_loss
        
        # Apply class weights if provided
        if self.class_weights is not None:
            # Compute weighted loss
            weight = torch.sum(targets * self.class_weights.unsqueeze(0).unsqueeze(0), dim=2)
            focal_loss = focal_loss * weight
        
        # Apply reduction
        if self.reduction == 'mean':
            return torch.mean(focal_loss)
        elif self.reduction == 'sum':
            return torch.sum(focal_loss)
        else:
            return focal_loss


class SpliceSiteTrainer:
    """Trainer for splice site prediction models."""
    
    def __init__(self, model=None, model_type='transformer', seq_length=10000, 
                 learning_rate=1e-3, weight_decay=1e-4, focal_loss_gamma=2.0, 
                 checkpoint_dir='checkpoints', tensorboard_dir='logs', device=None, **model_kwargs):
        """
        Initialize the trainer.
        
        Args:
            model (nn.Module): Pre-built PyTorch model (optional)
            model_type (str): Type of model to create if model is None
            seq_length (int): Length of input sequences
            learning_rate (float): Initial learning rate
            weight_decay (float): Weight decay factor
            focal_loss_gamma (float): Gamma parameter for focal loss
            checkpoint_dir (str): Directory to save checkpoints
            tensorboard_dir (str): Directory for TensorBoard logs
            device (str): Device to run the model on ('cuda' or 'cpu', defaults to 'cuda' if available)
            **model_kwargs: Additional arguments for model creation
        """
        self.seq_length = seq_length
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.focal_loss_gamma = focal_loss_gamma
        self.checkpoint_dir = checkpoint_dir
        self.tensorboard_dir = tensorboard_dir
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        logger.info(f"Using device: {self.device}")
        
        # Create model if not provided
        if model is None:
            self.model = create_model(model_type=model_type, seq_length=seq_length, **model_kwargs)
        else:
            self.model = model
            
        # Move model to device
        self.model = self.model.to(self.device)
            
        # Create directories
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(tensorboard_dir, exist_ok=True)
            
        # Initialize metrics
        self.metrics = {
            'train_loss': [],
            'val_loss': [],
            'train_accuracy': [],
            'val_accuracy': [],
            'val_auroc': [],
            'val_auprc': []
        }
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def compile_model(self, learning_rate=None, use_focal_loss=True, alpha=0.5):
        """
        Set up the model for training with optimizer and loss function.
        
        Args:
            learning_rate (float): Learning rate (uses self.learning_rate if None)
            use_focal_loss (bool): Whether to use focal loss
            alpha (float): Weight balancing factor between donor and acceptor losses (default 0.5 = equal weight)
        """
        if learning_rate is None:
            learning_rate = self.learning_rate
            
        # Create optimizer with weight decay
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.999),
            weight_decay=self.weight_decay
        )
        
        # Set loss function
        if use_focal_loss:
            self.criterion = SpliceSiteLoss(gamma=self.focal_loss_gamma)
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        logger.info(f"Model compiled with learning_rate={learning_rate}, use_focal_loss={use_focal_loss}")
    
    def lr_scheduler(self, optimizer, patience=5, factor=0.1, min_lr=1e-6):
        """
        Create a learning rate scheduler.
        
        Args:
            optimizer: PyTorch optimizer
            patience: Number of epochs with no improvement after which learning rate will be reduced
            factor: Factor by which the learning rate will be reduced
            min_lr: Lower bound on the learning rate
            
        Returns:
            PyTorch learning rate scheduler
        """
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=factor, 
            patience=patience, 
            verbose=True, 
            min_lr=min_lr
        )
    
    def train(self, train_dataset, validation_dataset, epochs=50, batch_size=32, 
              patience=10, use_lr_scheduler=True, class_weights=None):
        """
        Train the model.
        
        Args:
            train_dataset (torch.utils.data.Dataset): Training dataset
            validation_dataset (torch.utils.data.Dataset): Validation dataset
            epochs (int): Number of epochs
            batch_size (int): Batch size
            patience (int): Patience for early stopping
            use_lr_scheduler (bool): Whether to use learning rate scheduler
            class_weights (dict): Class weights for imbalanced datasets
            
        Returns:
            dict: Training history
        """
        # Compile model if not already compiled
        if not hasattr(self, 'optimizer'):
            self.compile_model()
            
        # Setup data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
        
        # Setup callbacks
        callbacks = [
            # Early stopping
            EarlyStopping(
                patience=patience,
                verbose=True
            )
        ]
        
        # Add learning rate scheduler if requested
        if use_lr_scheduler:
            scheduler = self.lr_scheduler(self.optimizer)
            callbacks.append(scheduler)
            
        # Train model
        logger.info(f"Starting training for {epochs} epochs with batch size {batch_size}")
        
        for epoch in range(epochs):
            # Train epoch
            self.model.train()
            total_loss = 0
            correct = 0
            total_nucleotides = 0
            
            for batch in train_loader:
                x, y = batch
                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(x)
                
                # Calculate loss
                loss = self.criterion(outputs, y)
                loss.backward()
                self.optimizer.step()
                
                # Track metrics
                total_loss += loss.item()
                
                # Calculate accuracy
                _, predicted = torch.max(outputs, 2)
                correct += (predicted == torch.argmax(y, 2)).sum().item()
                total_nucleotides += y.shape[0] * y.shape[1]
                
            # Average train metrics
            avg_loss = total_loss / len(train_loader)
            avg_accuracy = correct / total_nucleotides
            self.metrics['train_loss'].append(avg_loss)
            self.metrics['train_accuracy'].append(avg_accuracy)
            
            # Validate epoch
            self.model.eval()
            total_loss = 0
            correct = 0
            total_nucleotides = 0
            y_true = []
            y_pred = []
            
            with torch.no_grad():
                for batch in val_loader:
                    x, y = batch
                    x, y = x.to(self.device), y.to(self.device)
                    outputs = self.model(x)
                    
                    # Calculate loss
                    loss = self.criterion(outputs, y)
                    total_loss += loss.item()
                    
                    # Calculate accuracy
                    _, predicted = torch.max(outputs, 2)
                    correct += (predicted == torch.argmax(y, 2)).sum().item()
                    total_nucleotides += y.shape[0] * y.shape[1]
                    
                    # Collect for AUROC/AUPRC calculation
                    y_true.extend(torch.argmax(y, 2).cpu().numpy().flatten())
                    y_pred.extend(torch.max(outputs, 2)[1].cpu().numpy().flatten())
            
            # Calculate validation metrics
            avg_loss = total_loss / len(val_loader)
            avg_accuracy = correct / total_nucleotides
            
            # Calculate AUROC and AUPRC
            auroc = roc_auc_score(y_true, y_pred)
            auprc = average_precision_score(y_true, y_pred)
            
            # Update metrics
            self.metrics['val_loss'].append(avg_loss)
            self.metrics['val_accuracy'].append(avg_accuracy)
            self.metrics['val_auroc'].append(auroc)
            self.metrics['val_auprc'].append(auprc)
            
            # Log progress
            logger.info(f"Epoch {epoch+1}/{epochs}")
            logger.info(f"Train Loss: {self.metrics['train_loss'][-1]:.4f}, Accuracy: {self.metrics['train_accuracy'][-1]:.4f}")
            logger.info(f"Val Loss: {self.metrics['val_loss'][-1]:.4f}, Accuracy: {self.metrics['val_accuracy'][-1]:.4f}")
            logger.info(f"Val AUROC: {auroc:.4f}, AUPRC: {auprc:.4f}")
            
            # Check for early stopping
            if callbacks[0].step(avg_loss):
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
            
            # Update learning rate scheduler
            if use_lr_scheduler:
                scheduler.step(avg_loss)
        
        logger.info("Training completed")
        return self.metrics
    
    def evaluate(self, test_dataset, threshold=0.5):
        """
        Evaluate the model on test data.
        
        Args:
            test_dataset (torch.utils.data.Dataset): Test dataset
            threshold (float): Classification threshold
            
        Returns:
            dict: Evaluation metrics
        """
        logger.info("Evaluating model on test data")
        
        # Get predictions
        y_true = []
        y_pred_proba = []
        
        self.model.eval()
        with torch.no_grad():
            for batch in test_dataset:
                x, y = batch
                x, y = x.to(self.device), y.to(self.device)
                outputs = self.model(x)
                y_pred_proba.extend(torch.max(outputs, 2)[1].cpu().numpy())
                y_true.extend(torch.argmax(y, 2).cpu().numpy())
        
        y_true = np.array(y_true)
        y_pred_proba = np.array(y_pred_proba)
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        # Calculate metrics
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        pr_auc = average_precision_score(y_true, y_pred_proba)
        
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # Calculate additional metrics
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1_score = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
        
        results = {
            'accuracy': accuracy,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'precision': precision,
            'f1_score': f1_score,
            'auroc': roc_auc,
            'auprc': pr_auc,
            'true_positives': tp,
            'false_positives': fp,
            'true_negatives': tn,
            'false_negatives': fn,
            'fpr': fpr,
            'tpr': tpr,
            'precision_curve': precision,
            'recall_curve': recall
        }
        
        # Log results
        logger.info(f"Evaluation results:")
        for key, value in results.items():
            if isinstance(value, (int, float)):
                logger.info(f"{key}: {value}")
        
        return results
    
    def plot_metrics(self, save_dir=None):
        """
        Plot training metrics.
        
        Args:
            save_dir (str): Directory to save plots
        """
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            
        # Plot loss
        plt.figure(figsize=(10, 6))
        plt.plot(self.metrics['train_loss'], label='Training Loss')
        plt.plot(self.metrics['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        if save_dir:
            plt.savefig(os.path.join(save_dir, 'loss.png'))
        plt.close()
        
        # Plot accuracy
        plt.figure(figsize=(10, 6))
        plt.plot(self.metrics['train_accuracy'], label='Training Accuracy')
        plt.plot(self.metrics['val_accuracy'], label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.legend()
        plt.grid(True)
        if save_dir:
            plt.savefig(os.path.join(save_dir, 'accuracy.png'))
        plt.close()
        
        # Plot AUROC if available
        if self.metrics['val_auroc']:
            plt.figure(figsize=(10, 6))
            plt.plot(self.metrics['val_auroc'], label='Validation AUROC')
            plt.xlabel('Epoch')
            plt.ylabel('AUROC')
            plt.title('Validation AUROC')
            plt.legend()
            plt.grid(True)
            if save_dir:
                plt.savefig(os.path.join(save_dir, 'auroc.png'))
            plt.close()
    
    def plot_roc_curve(self, fpr, tpr, roc_auc, save_path=None):
        """
        Plot ROC curve.
        
        Args:
            fpr (np.ndarray): False positive rates
            tpr (np.ndarray): True positive rates
            roc_auc (float): Area under ROC curve
            save_path (str): Path to save plot
        """
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    
    def plot_precision_recall_curve(self, precision, recall, pr_auc, save_path=None):
        """
        Plot precision-recall curve.
        
        Args:
            precision (np.ndarray): Precision values
            recall (np.ndarray): Recall values
            pr_auc (float): Area under precision-recall curve
            save_path (str): Path to save plot
        """
        plt.figure(figsize=(10, 8))
        plt.plot(recall, precision, color='darkred', lw=2, label='PR curve (area = %0.2f)' % pr_auc)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
            
    def plot_confusion_matrix(self, cm, save_path=None):
        """
        Plot confusion matrix.
        
        Args:
            cm (np.ndarray): Confusion matrix
            save_path (str): Path to save plot
        """
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        
        classes = ['Negative', 'Positive']
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes)
        plt.yticks(tick_marks, classes)
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
            
    def save_model(self, filepath):
        """
        Save the model.
        
        Args:
            filepath (str): Path to save model
        """
        torch.save(self.model.state_dict(), filepath)
        logger.info(f"Model saved to {filepath}")
        
    def load_model(self, filepath):
        """
        Load a model.
        
        Args:
            filepath (str): Path to load model from
            
        Returns:
            nn.Module: Loaded model
        """
        self.model.load_state_dict(torch.load(filepath))
        logger.info(f"Model loaded from {filepath}")
        return self.model
        
    def benchmark_against_spliceai(self, test_data, spliceai_predictions, threshold=0.5):
        """
        Benchmark against SpliceAI.
        
        Args:
            test_data (dict): Test data with sequences and labels
            spliceai_predictions (dict): SpliceAI predictions
            threshold (float): Classification threshold
            
        Returns:
            dict: Benchmark results
        """
        # Get model predictions
        model_y_true = []
        model_y_pred = []
        
        self.model.eval()
        with torch.no_grad():
            for batch in test_data:
                x, y = batch
                x, y = x.to(self.device), y.to(self.device)
                outputs = self.model(x)
                model_y_true.extend(y.cpu().numpy())
                model_y_pred.extend(torch.max(outputs, 2)[1].cpu().numpy())
        
        model_y_true = np.array(model_y_true)
        model_y_pred = np.array(model_y_pred)
        
        # Get SpliceAI metrics
        spliceai_y_true = spliceai_predictions['true_labels']
        spliceai_y_pred = spliceai_predictions['predictions']
        
        # Calculate ROC curves
        model_fpr, model_tpr, _ = roc_curve(model_y_true, model_y_pred)
        model_roc_auc = auc(model_fpr, model_tpr)
        
        spliceai_fpr, spliceai_tpr, _ = roc_curve(spliceai_y_true, spliceai_y_pred)
        spliceai_roc_auc = auc(spliceai_fpr, spliceai_tpr)
        
        # Calculate PR curves
        model_precision, model_recall, _ = precision_recall_curve(model_y_true, model_y_pred)
        model_pr_auc = average_precision_score(model_y_true, model_y_pred)
        
        spliceai_precision, spliceai_recall, _ = precision_recall_curve(spliceai_y_true, spliceai_y_pred)
        spliceai_pr_auc = average_precision_score(spliceai_y_true, spliceai_y_pred)
        
        # Calculate confusion matrices
        model_cm = confusion_matrix(model_y_true, model_y_pred)
        spliceai_cm = confusion_matrix(spliceai_y_true, (spliceai_y_pred >= threshold).astype(int))
        
        # Plot comparison ROC curve
        plt.figure(figsize=(10, 8))
        plt.plot(model_fpr, model_tpr, color='darkorange', lw=2, 
                label=f'Our Model (AUC = {model_roc_auc:.3f})')
        plt.plot(spliceai_fpr, spliceai_tpr, color='darkgreen', lw=2, 
                label=f'SpliceAI (AUC = {spliceai_roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve Comparison')
        plt.legend(loc="lower right")
        plt.grid(True)
        
        # Calculate additional metrics for both models
        results = {
            'our_model': {
                'auroc': model_roc_auc,
                'auprc': model_pr_auc,
                'fpr': model_fpr,
                'tpr': model_tpr,
                'precision': model_precision,
                'recall': model_recall,
                'confusion_matrix': model_cm
            },
            'spliceai': {
                'auroc': spliceai_roc_auc,
                'auprc': spliceai_pr_auc,
                'fpr': spliceai_fpr,
                'tpr': spliceai_tpr,
                'precision': spliceai_precision,
                'recall': spliceai_recall,
                'confusion_matrix': spliceai_cm
            }
        }
        
        # Log benchmark results
        logger.info(f"Benchmark results:")
        logger.info(f"Our Model - AUROC: {model_roc_auc:.4f}, AUPRC: {model_pr_auc:.4f}")
        logger.info(f"SpliceAI - AUROC: {spliceai_roc_auc:.4f}, AUPRC: {spliceai_pr_auc:.4f}")
        
        return results


class EarlyStopping:
    """Early stopping callback."""
    
    def __init__(self, patience=5, min_delta=0.001, verbose=False):
        """
        Initialize early stopping callback.
        
        Args:
            patience (int): Number of epochs with no improvement after which training will be stopped
            min_delta (float): Minimum change in the monitored quantity to qualify as an improvement
            verbose (bool): Whether to print messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def step(self, loss):
        """
        Update early stopping callback.
        
        Args:
            loss (float): Current loss
            
        Returns:
            bool: Whether to stop training
        """
        if self.best_score is None:
            self.best_score = loss
        elif loss > self.best_score + self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = loss
            self.counter = 0
        return self.early_stop
