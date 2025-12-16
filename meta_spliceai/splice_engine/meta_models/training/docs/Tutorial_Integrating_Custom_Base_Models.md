# Tutorial: Integrating Custom Base Models into MetaSpliceAI

**A Step-by-Step Guide to Extending the Splice Prediction Workflow**

---

## 📚 Prerequisites

Before starting this tutorial, you should:
- Understand the [Base Model Pass architecture](Base_Model_Pass_Gene_Level_Evaluation.md)
- Have a pre-trained model that can predict splice sites
- Be familiar with Python class inheritance

## 🎯 Goal

By the end of this tutorial, you'll be able to integrate any splice site prediction model (OpenSpliceAI, fine-tuned Evo2, or your custom model) into MetaSpliceAI's meta-learning pipeline.

---

## Part 1: Understanding the Base Model Interface

### The Contract: What Your Model Must Provide

Any base model integrated into MetaSpliceAI must produce three probability scores for each nucleotide position:

```python
# Required output for position i in a sequence
output[i] = {
    'donor_prob': 0.15,     # P(donor splice site)
    'acceptor_prob': 0.03,  # P(acceptor splice site)  
    'neither_prob': 0.82    # P(not a splice site)
}
# Constraint: donor_prob + acceptor_prob + neither_prob ≈ 1.0
```

---

## Part 2: Creating the Base Model Adapter Pattern

### Step 1: Define the Abstract Base Class

```python
# File: meta_spliceai/splice_engine/meta_models/base_models/base_adapter.py

from abc import ABC, abstractmethod
from typing import Dict, List, Any
import numpy as np

class BaseModelAdapter(ABC):
    """Abstract base class for splice site prediction models."""
    
    def __init__(self, model_config: Dict[str, Any] = None):
        self.config = model_config or {}
        self.model = None
        self.is_loaded = False
    
    @abstractmethod
    def load_model(self) -> None:
        """Load the pre-trained model weights."""
        pass
    
    @abstractmethod
    def predict_gene(self, sequence: str, gene_info: Dict) -> Dict[str, np.ndarray]:
        """
        Generate predictions for a single gene.
        
        Returns:
            Dictionary with keys:
            - donor_prob: np.ndarray of shape (sequence_length,)
            - acceptor_prob: np.ndarray of shape (sequence_length,)
            - neither_prob: np.ndarray of shape (sequence_length,)
        """
        pass
    
    def predict_splice_sites_for_genes(
        self, genes: List[Dict], sequences: Dict[str, str]
    ) -> Dict[str, Dict]:
        """Generate predictions for multiple genes (main entry point)."""
        if not self.is_loaded:
            self.load_model()
        
        predictions = {}
        for gene in genes:
            gene_id = gene['gene_id']
            sequence = sequences.get(gene_id)
            
            if sequence:
                gene_predictions = self.predict_gene(sequence, gene)
                gene_predictions.update({
                    'gene_start': gene['gene_start'],
                    'gene_end': gene['gene_end'],
                    'strand': gene['strand'],
                    'chromosome': gene['chromosome']
                })
                predictions[gene_id] = gene_predictions
        
        return predictions
```

---

## Part 3: Implementing Model-Specific Adapters

### Example 1: SpliceAI Adapter

```python
# File: meta_spliceai/splice_engine/meta_models/base_models/spliceai_adapter.py

class SpliceAIAdapter(BaseModelAdapter):
    """Adapter for the original SpliceAI model."""
    
    def load_model(self):
        from meta_spliceai.splice_engine.run_spliceai_workflow import load_spliceai_models
        self.models = load_spliceai_models(
            model_dir=self.config.get('model_path', 'models/spliceai/')
        )
        self.is_loaded = True
    
    def predict_gene(self, sequence: str, gene_info: Dict) -> Dict[str, np.ndarray]:
        from meta_spliceai.splice_engine.run_spliceai_workflow import (
            predict_splice_sites_single_gene
        )
        
        predictions = predict_splice_sites_single_gene(
            sequence=sequence,
            models=self.models,
            gene_info=gene_info,
            context_window=self.config.get('context_window', 10000)
        )
        
        return {
            'donor_prob': predictions['donor_prob'],
            'acceptor_prob': predictions['acceptor_prob'],
            'neither_prob': predictions['neither_prob']
        }
```

### Example 2: OpenSpliceAI Adapter

```python
# File: meta_spliceai/splice_engine/meta_models/base_models/openspliceai_adapter.py

class OpenSpliceAIAdapter(BaseModelAdapter):
    """
    Adapter for OpenSpliceAI models.
    
    OpenSpliceAI uses PyTorch state dictionaries (.pth/.pt files) similar to SpliceAI.
    Pre-trained models like OSAIMANE-10000nt need to be downloaded separately from:
    https://github.com/Kuanhao-Chao/OpenSpliceAI/tree/main/models/
    """
    
    def load_model(self):
        """Load OpenSpliceAI model(s) from state dict files."""
        from meta_spliceai.openspliceai.predict.predict import load_pytorch_models
        from meta_spliceai.openspliceai.constants import consts
        import torch
        
        # Set device
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device_str)
        
        # Load model(s) - can be single file or directory of models
        model_path = self.config.get('model_path')  # Path to .pth file or directory
        flanking_size = self.config.get('flanking_size', 10000)  # Context window
        
        # OpenSpliceAI's load_pytorch_models handles both single files and directories
        self.models, self.params = load_pytorch_models(
            model_path=model_path,
            device=self.device,
            SL=consts['SL'],  # Sequence length from OpenSpliceAI constants
            CL=flanking_size  # Context length
        )
        
        # If multiple models loaded (ensemble), we'll average their predictions
        self.use_ensemble = len(self.models) > 1
        self.is_loaded = True
    
    def predict_gene(self, sequence: str, gene_info: Dict) -> Dict[str, np.ndarray]:
        """Generate OpenSpliceAI predictions, handling ensemble if multiple models."""
        import torch
        import numpy as np
        
        # Convert sequence to one-hot encoding
        one_hot_seq = self._encode_sequence(sequence)
        
        # Prepare input tensor
        X = torch.tensor(one_hot_seq, dtype=torch.float32).unsqueeze(0)
        X = X.transpose(1, 2).to(self.device)  # Shape: [batch, channels, length]
        
        # Get predictions from all models
        all_predictions = []
        with torch.no_grad():
            for model in self.models:
                output = model(X)
                all_predictions.append(output)
        
        # Average predictions if ensemble
        if self.use_ensemble:
            avg_output = torch.mean(torch.stack(all_predictions), dim=0)
        else:
            avg_output = all_predictions[0]
        
        # Convert to numpy and extract probabilities
        output_np = avg_output.squeeze(0).cpu().numpy()  # Shape: [3, seq_length]
        
        # OpenSpliceAI outputs 3 channels: neither, acceptor, donor
        neither_prob = output_np[0, :]
        acceptor_prob = output_np[1, :]
        donor_prob = output_np[2, :]
        
        # Apply softmax to get proper probabilities
        import scipy.special
        probs = np.stack([neither_prob, acceptor_prob, donor_prob], axis=0)
        probs = scipy.special.softmax(probs, axis=0)
        
        return {
            'donor_prob': probs[2, :],     # Donor channel
            'acceptor_prob': probs[1, :],   # Acceptor channel
            'neither_prob': probs[0, :]     # Neither channel
        }
    
    def _encode_sequence(self, sequence: str) -> np.ndarray:
        """Convert DNA string to one-hot encoding for OpenSpliceAI."""
        mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
        encoded = np.zeros((4, len(sequence)))  # 4 channels for A, C, G, T
        
        for i, nucleotide in enumerate(sequence.upper()):
            if nucleotide in mapping:
                encoded[mapping[nucleotide], i] = 1
            # Unknown nucleotides left as all zeros
        
        return encoded
```

**Note on Pre-trained Models:**
- OpenSpliceAI models are NOT included in the codebase
- Download pre-trained models from the OpenSpliceAI GitHub repository
- The OSAIMANE-10000nt model is recommended for human genomics
- Models are PyTorch state dictionaries (.pth files), similar to SpliceAI
```

---

## Part 4: Creating the Model Registry

```python
# File: meta_spliceai/splice_engine/meta_models/base_models/model_registry.py

from typing import Dict, Type

class ModelRegistry:
    """Registry for available base models."""
    
    _models: Dict[str, Type[BaseModelAdapter]] = {}
    
    @classmethod
    def register(cls, name: str, adapter_class: Type[BaseModelAdapter]):
        """Register a new model adapter."""
        cls._models[name] = adapter_class
    
    @classmethod
    def get_model(cls, name: str, config: Dict = None) -> BaseModelAdapter:
        """Get an initialized model adapter."""
        if name not in cls._models:
            available = ', '.join(cls._models.keys())
            raise ValueError(f"Unknown model: {name}. Available: {available}")
        
        return cls._models[name](config)

# Register built-in models
ModelRegistry.register('spliceai', SpliceAIAdapter)
ModelRegistry.register('openspliceai', OpenSpliceAIAdapter)
```

---

## Part 5: Updating the Main Workflow

```python
# File: meta_spliceai/splice_engine/meta_models/workflows/splice_prediction_workflow.py

from ..base_models.model_registry import ModelRegistry

def run_enhanced_splice_prediction_workflow(
    genes: List[Dict],
    sequences: Dict[str, str],
    base_model: str = 'spliceai',
    model_config: Dict = None,
    meta_model_path: str = None,
    **kwargs
):
    """
    Run splice prediction with pluggable base model.
    
    Args:
        base_model: Name of base model ('spliceai', 'openspliceai', etc.)
        model_config: Configuration for the base model
    """
    
    # Step 1: Get the appropriate model adapter
    print(f"🔧 Loading base model: {base_model}")
    base_adapter = ModelRegistry.get_model(base_model, model_config)
    
    # Step 2: Generate base model predictions
    print(f"🧬 Generating predictions for {len(genes)} genes...")
    base_predictions = base_adapter.predict_splice_sites_for_genes(genes, sequences)
    
    # Step 3: Evaluate predictions (existing code)
    from .enhanced_workflow import enhanced_process_predictions_with_all_scores
    
    error_df, positions_df = enhanced_process_predictions_with_all_scores(
        pred_results=base_predictions,
        genes=genes,
        **kwargs
    )
    
    # Step 4: Apply meta-model if provided
    if meta_model_path:
        print(f"🎯 Applying meta-model corrections...")
        positions_df = apply_meta_model(positions_df, meta_model_path)
    
    return {
        'base_predictions': base_predictions,
        'error_df': error_df,
        'positions_df': positions_df
    }
```

---

## Part 6: Adding Your Custom Model - Quick Guide

### Step 1: Create Your Adapter

```python
# File: your_model_adapter.py

class YourModelAdapter(BaseModelAdapter):
    def load_model(self):
        # Load your model
        from your_package import YourModel
        self.model = YourModel.load(self.config['checkpoint_path'])
        self.is_loaded = True
    
    def predict_gene(self, sequence: str, gene_info: Dict) -> Dict:
        # Your prediction logic
        raw_output = self.model.predict(sequence)
        
        # Convert to required format
        return {
            'donor_prob': raw_output['donor'],
            'acceptor_prob': raw_output['acceptor'],
            'neither_prob': 1.0 - (raw_output['donor'] + raw_output['acceptor'])
        }
```

### Step 2: Register Your Model

```python
from model_registry import ModelRegistry
from your_model_adapter import YourModelAdapter

ModelRegistry.register('your_model', YourModelAdapter)
```

### Step 3: Use Your Model

```python
results = run_enhanced_splice_prediction_workflow(
    genes=my_genes,
    sequences=my_sequences,
    base_model='your_model',  # Your registered model name
    model_config={'checkpoint_path': 'path/to/model.ckpt'}
)
```

---

## Part 7: Testing Your Integration

```python
# File: test_integration.py

import unittest
import numpy as np

class TestModelIntegration(unittest.TestCase):
    def test_output_protocol(self):
        """Test that output conforms to protocol."""
        model = ModelRegistry.get_model('your_model')
        
        test_gene = {
            'gene_id': 'TEST001',
            'gene_start': 1000,
            'gene_end': 2000,
            'strand': '+',
            'chromosome': 'chr1'
        }
        test_sequence = 'ATCG' * 250  # 1000bp
        
        predictions = model.predict_gene(test_sequence, test_gene)
        
        # Check required keys
        self.assertIn('donor_prob', predictions)
        self.assertIn('acceptor_prob', predictions)
        self.assertIn('neither_prob', predictions)
        
        # Check probability sum
        prob_sum = (predictions['donor_prob'] + 
                   predictions['acceptor_prob'] + 
                   predictions['neither_prob'])
        np.testing.assert_allclose(prob_sum, 1.0, atol=0.01)
```

---

## Part 8: CLI Integration

```python
# Add to your training script
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    '--base-model',
    type=str,
    default='spliceai',
    choices=['spliceai', 'openspliceai', 'your_model'],
    help='Base model for splice site prediction'
)
parser.add_argument(
    '--model-checkpoint',
    type=str,
    help='Path to model checkpoint'
)

args = parser.parse_args()

# Run with selected model
results = run_enhanced_splice_prediction_workflow(
    genes=genes,
    sequences=sequences,
    base_model=args.base_model,
    model_config={'checkpoint_path': args.model_checkpoint}
)
```

---

## Part 9: Performance Tips

### Batching for Efficiency

```python
class BatchedAdapter(BaseModelAdapter):
    def predict_splice_sites_for_genes(self, genes, sequences):
        """Optimized batch prediction."""
        batch_size = self.config.get('batch_size', 32)
        predictions = {}
        
        for i in range(0, len(genes), batch_size):
            batch_genes = genes[i:i+batch_size]
            batch_seqs = [sequences[g['gene_id']] for g in batch_genes]
            
            # Batch prediction
            batch_preds = self.predict_batch(batch_seqs, batch_genes)
            
            for gene, pred in zip(batch_genes, batch_preds):
                predictions[gene['gene_id']] = pred
        
        return predictions
```

### Caching Predictions

```python
import pickle
from pathlib import Path

class CachedAdapter(BaseModelAdapter):
    def predict_gene(self, sequence: str, gene_info: Dict) -> Dict:
        # Check cache first
        cache_key = f"{gene_info['gene_id']}_{hash(sequence)}"
        cache_path = Path(f"cache/{cache_key}.pkl")
        
        if cache_path.exists():
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        
        # Generate predictions
        predictions = super().predict_gene(sequence, gene_info)
        
        # Save to cache
        cache_path.parent.mkdir(exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump(predictions, f)
        
        return predictions
```

---

## Summary

You now have a complete framework for integrating any splice site prediction model:

1. **Create an adapter** inheriting from `BaseModelAdapter`
2. **Implement two methods**: `load_model()` and `predict_gene()`
3. **Register your model** with the `ModelRegistry`
4. **Use the unified workflow** with your model name

The key is that all models must produce the same output format (donor, acceptor, neither probabilities), but their internal implementation is completely flexible.

---

## Next Steps

- Review the comprehensive [Base Model Pass documentation](Base_Model_Pass_Gene_Level_Evaluation.md)
- Test your integration with the provided templates
- Explore multi-model ensembles for improved performance

---

*Last Updated: September 2025*
