# Splice Site Model Training and Evaluation Instructions

This document provides detailed instructions for training and evaluating splice site prediction models used in the MetaSpliceAI project.

## Overview

The training pipeline includes:

1. **Training Loop**: Implementation of model training with appropriate loss functions
2. **Evaluation Metrics**: Methods to assess model performance
3. **Hyperparameter Tuning**: Approaches for optimizing model performance
4. **Inference Pipeline**: Steps for making predictions on new sequences

## 1. Model Training

### Loss Function Implementation

For the three-class splice site prediction task, we use a specialized loss function:

```python
class SpliceSiteLoss(nn.Module):
    """
    Loss function for splice site prediction with three classes (donor, acceptor, neither).
    Uses a combination of cross-entropy and focal loss to handle class imbalance.
    """
    
    def __init__(self, gamma=2.0, class_weights=None, reduction='mean'):
        """
        Initialize splice site loss function.
        
        Args:
            gamma: Focusing parameter for focal loss that adjusts how easy examples are down-weighted
            class_weights: Optional tensor of shape (3,) with weights for [donor, acceptor, neither]
            reduction: Reduction method ('none', 'mean', 'sum')
        """
        super(SpliceSiteLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.class_weights = class_weights
    
    def forward(self, inputs, targets):
        """
        Forward pass for splice site loss.
        
        Args:
            inputs: Predicted probabilities shape (batch_size, seq_len, 3) for [donor, acceptor, neither]
            targets: Ground truth one-hot labels shape (batch_size, seq_len, 3)
        """
        # Ensure inputs are valid probabilities
        epsilon = 1e-7
        inputs = torch.clamp(inputs, epsilon, 1 - epsilon)
        
        # Cross entropy
        ce_loss = -torch.sum(targets * torch.log(inputs), dim=2)
        
        # Focal loss modulation
        p_t = torch.exp(-ce_loss)
        focal_loss = (1 - p_t) ** self.gamma * ce_loss
        
        # Apply class weights if provided
        if self.class_weights is not None:
            weight = torch.sum(targets * self.class_weights.unsqueeze(0).unsqueeze(0), dim=2)
            focal_loss = focal_loss * weight
        
        # Apply reduction
        if self.reduction == 'mean':
            return torch.mean(focal_loss)
        elif self.reduction == 'sum':
            return torch.sum(focal_loss)
        else:
            return focal_loss
```

### Training Loop

```python
class SpliceSiteTrainer:
    """Trainer for splice site prediction models."""
    
    def __init__(
        self,
        model_type: str,
        device: str = None,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        use_focal_loss: bool = True,
        focal_loss_gamma: float = 2.0,
        alpha: float = 0.5,
        **model_kwargs
    ):
        """
        Initialize trainer.
        
        Args:
            model_type: Type of model ('cnn', 'transformer', 'hyenadna')
            device: Device to use ('cuda', 'mps', 'cpu')
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for regularization
            use_focal_loss: Whether to use focal loss
            focal_loss_gamma: Gamma parameter for focal loss
            alpha: Weight balancing factor between donor and acceptor losses
            **model_kwargs: Additional arguments for model creation
        """
        # Set device (automatically detect if not specified)
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else \
                         'mps' if torch.backends.mps.is_available() else 'cpu'
        else:
            self.device = device
            
        # Create model
        self.model = create_model(model_type, **model_kwargs)
        self.model = self.model.to(self.device)
        
        # Set hyperparameters
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.focal_loss_gamma = focal_loss_gamma
        
        # Set optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Set loss function
        if use_focal_loss:
            self.criterion = SpliceSiteLoss(gamma=self.focal_loss_gamma)
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        # Initialize metrics tracking
        self.metrics = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': [],
            'val_auroc': [],
            'val_auprc': []
        }
    
    def train(self, train_dataset, validation_dataset, epochs=50, batch_size=32, 
              patience=10, use_lr_scheduler=True, class_weights=None):
        """
        Train the model.
        
        Args:
            train_dataset: Training dataset
            validation_dataset: Validation dataset
            epochs: Number of training epochs
            batch_size: Batch size
            patience: Patience for early stopping
            use_lr_scheduler: Whether to use learning rate scheduler
            class_weights: Optional class weights for loss function
        
        Returns:
            Dictionary of training history
        """
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        val_loader = DataLoader(
            validation_dataset, batch_size=batch_size, shuffle=False
        )
        
        # Set up learning rate scheduler if requested
        if use_lr_scheduler:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
            )
        
        # Set up early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        # Training loop
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
            
            # Calculate average metrics
            avg_loss = total_loss / len(train_loader)
            avg_accuracy = correct / total_nucleotides
            self.metrics['train_loss'].append(avg_loss)
            self.metrics['train_accuracy'].append(avg_accuracy)
            
            # Validation epoch
            self.model.eval()
            val_loss, val_accuracy, auroc, auprc = self.evaluate(val_loader)
            
            # Update metrics
            self.metrics['val_loss'].append(val_loss)
            self.metrics['val_accuracy'].append(val_accuracy)
            self.metrics['val_auroc'].append(auroc)
            self.metrics['val_auprc'].append(auprc)
            
            # Update learning rate scheduler
            if use_lr_scheduler:
                scheduler.step(val_loss)
            
            # Check for early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
            
            # Log progress
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"Train Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}")
            print(f"Val AUROC: {auroc:.4f}, AUPRC: {auprc:.4f}")
        
        # Restore best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            
        return self.metrics
    
    def evaluate(self, dataloader):
        """
        Evaluate the model on a dataset.
        
        Args:
            dataloader: DataLoader for evaluation
            
        Returns:
            Tuple of (loss, accuracy, auroc, auprc)
        """
        total_loss = 0
        correct = 0
        total_nucleotides = 0
        y_true = []
        y_pred = []
        
        with torch.no_grad():
            for batch in dataloader:
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
                
                # Collect predictions for AUROC/AUPRC
                y_true.extend(torch.argmax(y, 2).cpu().numpy().flatten())
                y_pred.extend(torch.max(outputs, 2)[1].cpu().numpy().flatten())
        
        # Calculate metrics
        avg_loss = total_loss / len(dataloader)
        accuracy = correct / total_nucleotides
        
        # Calculate ROC and PR curves
        try:
            auroc = roc_auc_score(y_true, y_pred)
            auprc = average_precision_score(y_true, y_pred)
        except:
            # If there's an issue with the curves (e.g., only one class present)
            auroc = 0.0
            auprc = 0.0
        
        return avg_loss, accuracy, auroc, auprc
```

## 2. Model Evaluation

### Evaluation Metrics

For splice site prediction, several metrics are particularly relevant:

1. **Accuracy**: The overall accuracy across all three classes
2. **AUROC**: Area Under the Receiver Operating Characteristic curve
3. **AUPRC**: Area Under the Precision-Recall curve (especially important for imbalanced data)
4. **Per-Class Metrics**: Precision, recall, and F1 score for each class

```python
def calculate_performance_metrics(y_true, y_pred_proba, threshold=0.5):
    """
    Calculate comprehensive performance metrics.
    
    Args:
        y_true: Ground truth labels (one-hot encoded)
        y_pred_proba: Predicted probabilities from model
        threshold: Classification threshold
        
    Returns:
        Dictionary of metrics
    """
    # Convert one-hot to class indices
    y_true_class = np.argmax(y_true, axis=2).flatten()
    
    # Get class predictions
    y_pred_class = np.argmax(y_pred_proba, axis=2).flatten()
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true_class, y_pred_class)
    
    # Calculate per-class metrics
    precision = precision_score(y_true_class, y_pred_class, average=None)
    recall = recall_score(y_true_class, y_pred_class, average=None)
    f1 = f1_score(y_true_class, y_pred_class, average=None)
    
    # Calculate AUROC and AUPRC (one-vs-rest for each class)
    auroc = []
    auprc = []
    
    for i in range(3):  # For each class (donor, acceptor, neither)
        # Convert to binary problem (one-vs-rest)
        y_binary = (y_true_class == i).astype(int)
        y_score = y_pred_proba.reshape(-1, 3)[:, i]
        
        # Calculate ROC and PR curves
        auroc.append(roc_auc_score(y_binary, y_score))
        auprc.append(average_precision_score(y_binary, y_score))
    
    return {
        'confusion_matrix': cm,
        'accuracy': accuracy_score(y_true_class, y_pred_class),
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auroc': auroc,
        'auprc': auprc
    }
```

### Visualization

Visualize model performance with relevant plots:

```python
def plot_performance_metrics(metrics, class_names=None):
    """
    Plot performance metrics.
    
    Args:
        metrics: Dictionary of metrics from calculate_performance_metrics
        class_names: Optional list of class names ['Donor', 'Acceptor', 'Neither']
    """
    if class_names is None:
        class_names = ['Donor', 'Acceptor', 'Neither']
    
    # Set up figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Confusion matrix
    im = axes[0, 0].imshow(metrics['confusion_matrix'], cmap='Blues')
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            axes[0, 0].text(j, i, metrics['confusion_matrix'][i, j],
                           ha="center", va="center", color="white" if metrics['confusion_matrix'][i, j] > metrics['confusion_matrix'].max() / 2 else "black")
    
    axes[0, 0].set_xticks(np.arange(len(class_names)))
    axes[0, 0].set_yticks(np.arange(len(class_names)))
    axes[0, 0].set_xticklabels(class_names)
    axes[0, 0].set_yticklabels(class_names)
    axes[0, 0].set_xlabel('Predicted')
    axes[0, 0].set_ylabel('True')
    axes[0, 0].set_title('Confusion Matrix')
    
    # Per-class precision, recall, f1
    x = np.arange(len(class_names))
    width = 0.25
    
    axes[0, 1].bar(x - width, metrics['precision'], width, label='Precision')
    axes[0, 1].bar(x, metrics['recall'], width, label='Recall')
    axes[0, 1].bar(x + width, metrics['f1'], width, label='F1')
    
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(class_names)
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].set_ylabel('Score')
    axes[0, 1].set_title('Precision, Recall, and F1 Score by Class')
    axes[0, 1].legend()
    
    # AUROC
    axes[1, 0].bar(x, metrics['auroc'], color='blue')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(class_names)
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].set_ylabel('AUROC')
    axes[1, 0].set_title('Area Under ROC Curve by Class')
    
    # AUPRC
    axes[1, 1].bar(x, metrics['auprc'], color='green')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(class_names)
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].set_ylabel('AUPRC')
    axes[1, 1].set_title('Area Under PR Curve by Class')
    
    fig.tight_layout()
    
    return fig
```

## 3. Hyperparameter Tuning

### Grid Search Implementation

```python
def hyperparameter_tuning(train_dataset, val_dataset, model_type='cnn',
                          param_grid=None, fixed_params=None, num_epochs=10):
    """
    Perform hyperparameter tuning using grid search.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        model_type: Type of model to tune
        param_grid: Dictionary of parameters to search
        fixed_params: Dictionary of fixed parameters
        num_epochs: Number of epochs for each trial
        
    Returns:
        Dictionary of best parameters and results
    """
    if param_grid is None:
        if model_type == 'cnn':
            param_grid = {
                'num_filters': [64, 128, 256],
                'kernel_size': [7, 11, 15],
                'learning_rate': [1e-3, 1e-4],
                'dropout_rate': [0.1, 0.2, 0.3]
            }
        elif model_type == 'transformer':
            param_grid = {
                'embed_dim': [64, 128],
                'num_heads': [4, 8],
                'num_transformer_blocks': [3, 6],
                'learning_rate': [1e-3, 1e-4],
                'dropout_rate': [0.1, 0.2]
            }
    
    if fixed_params is None:
        fixed_params = {}
    
    # Generate all parameter combinations
    param_combinations = list(product(*param_grid.values()))
    param_names = list(param_grid.keys())
    
    best_val_auprc = 0
    best_params = None
    best_metrics = None
    
    # Perform grid search
    for i, params in enumerate(param_combinations):
        param_dict = {**fixed_params, **{name: value for name, value in zip(param_names, params)}}
        
        # Create model-specific parameters
        model_params = {k: v for k, v in param_dict.items() 
                       if k not in ['learning_rate', 'weight_decay']}
        
        # Extract trainer parameters
        lr = param_dict.get('learning_rate', 1e-3)
        weight_decay = param_dict.get('weight_decay', 1e-4)
        
        # Create and train model
        trainer = SpliceSiteTrainer(
            model_type=model_type,
            learning_rate=lr,
            weight_decay=weight_decay,
            **model_params
        )
        
        print(f"Trial {i+1}/{len(param_combinations)}: {param_dict}")
        
        # Train for specified epochs
        metrics = trainer.train(
            train_dataset, 
            val_dataset,
            epochs=num_epochs,
            patience=num_epochs  # Disable early stopping for consistent comparison
        )
        
        # Get final validation AUPRC
        val_auprc = metrics['val_auprc'][-1]
        
        if val_auprc > best_val_auprc:
            best_val_auprc = val_auprc
            best_params = param_dict
            best_metrics = metrics
        
        print(f"Validation AUPRC: {val_auprc:.4f}")
        print("-----------------------------")
    
    print(f"Best parameters: {best_params}")
    print(f"Best validation AUPRC: {best_val_auprc:.4f}")
    
    return {
        'best_params': best_params,
        'best_metrics': best_metrics
    }
```

## 4. Model Saving and Loading

```python
def save_model(model, file_path, metadata=None):
    """
    Save model and metadata.
    
    Args:
        model: PyTorch model to save
        file_path: Path to save the model
        metadata: Optional dictionary of metadata
    """
    model_state = model.state_dict()
    
    # Prepare save dictionary
    save_dict = {
        'model_state': model_state,
        'model_class': model.__class__.__name__,
        'model_config': {k: v for k, v in model.__dict__.items() 
                        if not k.startswith('_') and not callable(v)}
    }
    
    if metadata:
        save_dict['metadata'] = metadata
    
    # Save to file
    torch.save(save_dict, file_path)
    print(f"Model saved to {file_path}")

def load_model(file_path, device=None):
    """
    Load saved model.
    
    Args:
        file_path: Path to saved model
        device: Device to load model to
        
    Returns:
        Loaded model and metadata
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else \
              'mps' if torch.backends.mps.is_available() else 'cpu'
    
    # Load from file
    save_dict = torch.load(file_path, map_location=device)
    
    # Get model class and config
    model_class_name = save_dict['model_class']
    model_config = save_dict.get('model_config', {})
    
    # Create model instance
    model_cls = globals()[model_class_name]
    model = model_cls(**model_config)
    
    # Load state
    model.load_state_dict(save_dict['model_state'])
    model = model.to(device)
    
    return model, save_dict.get('metadata', None)
```

## 5. Inference Pipeline

### Sequence Prediction

```python
def predict_sequence(model, sequence, window_size=1000, stride=500, batch_size=32, device=None):
    """
    Make predictions on a long sequence by using a sliding window approach.
    
    Args:
        model: Trained model
        sequence: One-hot encoded sequence of shape (seq_length, 4)
        window_size: Size of sliding window
        stride: Stride between windows
        batch_size: Batch size for predictions
        device: Device to use for predictions
        
    Returns:
        Predicted probabilities for each position of shape (seq_length, 3)
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else \
               'mps' if torch.backends.mps.is_available() else 'cpu'
    
    seq_length = len(sequence)
    model = model.to(device)
    model.eval()
    
    # Create sliding windows
    windows = []
    window_starts = []
    
    for start in range(0, seq_length - window_size + 1, stride):
        end = start + window_size
        window = sequence[start:end]
        windows.append(window)
        window_starts.append(start)
    
    # Add final window if needed
    if seq_length - window_starts[-1] - window_size > 0:
        start = seq_length - window_size
        window = sequence[start:]
        windows.append(window)
        window_starts.append(start)
    
    # Convert to batches
    num_windows = len(windows)
    num_batches = (num_windows + batch_size - 1) // batch_size
    
    # Initialize output array
    predictions = np.zeros((seq_length, 3), dtype=np.float32)
    counts = np.zeros(seq_length, dtype=np.int32)
    
    # Process each batch
    with torch.no_grad():
        for i in range(num_batches):
            batch_start = i * batch_size
            batch_end = min((i + 1) * batch_size, num_windows)
            
            batch_windows = windows[batch_start:batch_end]
            batch_starts = window_starts[batch_start:batch_end]
            
            # Convert to tensor
            batch_tensor = torch.tensor(np.array(batch_windows), dtype=torch.float32).to(device)
            
            # Get predictions
            batch_preds = model(batch_tensor).cpu().numpy()
            
            # Aggregate predictions
            for j, start in enumerate(batch_starts):
                end = start + window_size
                predictions[start:end] += batch_preds[j]
                counts[start:end] += 1
    
    # Average overlapping predictions
    predictions = predictions / counts.reshape(-1, 1)
    
    return predictions
```

## 6. Model Comparison and Benchmarking

```python
def compare_models(models_dict, test_dataset, metrics=['accuracy', 'auroc', 'auprc']):
    """
    Compare multiple models on the same test dataset.
    
    Args:
        models_dict: Dictionary mapping model names to model objects
        test_dataset: Test dataset
        metrics: List of metrics to compare
        
    Returns:
        DataFrame of comparison results
    """
    results = {}
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    for model_name, model in models_dict.items():
        print(f"Evaluating {model_name}...")
        model.eval()
        
        # Make predictions
        all_y_true = []
        all_y_pred = []
        
        with torch.no_grad():
            for x, y in test_loader:
                outputs = model(x)
                
                all_y_true.append(y.numpy())
                all_y_pred.append(outputs.numpy())
        
        # Concatenate results
        y_true = np.concatenate(all_y_true, axis=0)
        y_pred = np.concatenate(all_y_pred, axis=0)
        
        # Calculate metrics
        performance = calculate_performance_metrics(y_true, y_pred)
        
        # Store results
        results[model_name] = {
            'accuracy': performance['accuracy'],
            'auroc_donor': performance['auroc'][0],
            'auroc_acceptor': performance['auroc'][1],
            'auroc_neither': performance['auroc'][2],
            'auprc_donor': performance['auprc'][0],
            'auprc_acceptor': performance['auprc'][1],
            'auprc_neither': performance['auprc'][2],
        }
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results).T
    
    return results_df
```

## Key Considerations

1. **Balanced Evaluation**: Use metrics like AUPRC that are robust to class imbalance

2. **Resource Management**: Train with appropriate batch sizes for your hardware constraints

3. **Learning Rate Scheduling**: Use learning rate schedulers to improve convergence

4. **Early Stopping**: Implement early stopping to prevent overfitting

5. **Model Comparison**: Compare multiple architectures on the same dataset

6. **Hyperparameter Sensitivity**: Test models with different hyperparameters to ensure robustness
