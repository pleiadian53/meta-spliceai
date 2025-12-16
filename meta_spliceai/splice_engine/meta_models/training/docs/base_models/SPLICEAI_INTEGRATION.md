# SpliceAI Integration Guide

This document provides a comprehensive guide to SpliceAI model integration in the MetaSpliceAI meta-learning system.

## 🎯 Overview

SpliceAI is the default base model used in MetaSpliceAI for per-nucleotide splice site prediction. It provides the foundation layer that meta-models build upon to improve accuracy and reduce prediction errors.

## 📁 Model Structure

### **Systematic Path Convention**
SpliceAI models follow the MetaSpliceAI systematic path convention:
- **Pattern**: `data/models/<base_model_name>/`
- **SpliceAI**: `data/models/spliceai/`
- **Future Models**: `data/models/openspliceai/`, `data/models/custom_model/`, etc.

### **Default Model Path**
```
data/models/spliceai/
├── spliceai1.h5            # SpliceAI ensemble model 1
├── spliceai2.h5            # SpliceAI ensemble model 2
├── spliceai3.h5            # SpliceAI ensemble model 3
├── spliceai4.h5            # SpliceAI ensemble model 4
└── spliceai5.h5            # SpliceAI ensemble model 5
```

**Note**: This directory follows the systematic convention `data/models/<base_model_name>/` and is a symlink to the SpliceAI package models, maintaining consistency while enabling future base model integration.

### **Benefits of Systematic Path Convention**
- **Consistency**: All base models follow the same `data/models/<base_model_name>/` pattern
- **Discoverability**: Resource managers can systematically locate models
- **Extensibility**: Easy to add new base models (OpenSpliceAI, custom models)
- **Maintainability**: Clear organization and predictable structure

### **Model Files**
- **Format**: Keras/TensorFlow HDF5 (`.h5`)
- **Architecture**: Deep neural network with attention mechanisms
- **Input**: DNA sequence with one-hot encoding
- **Output**: Per-nucleotide splice site probabilities (donor/acceptor)

## 🔧 Loading Mechanism

### **Core Loading Function**
```python
from meta_spliceai.splice_engine.meta_models.utils.model_utils import load_spliceai_ensemble

# Load models with specific context
models = load_spliceai_ensemble(context=10_000)
```

### **Implementation Details**

#### **1. Model Discovery**
```python
def load_spliceai_ensemble(context: int = 10000) -> List:
    """
    Load SpliceAI ensemble models from systematic path.
    
    Parameters
    ----------
    context : int
        Context window size in base pairs (default: 10,000)
        Note: Context is used for input processing, not model selection
        
    Returns
    -------
    List
        List of 5 loaded SpliceAI model instances (ensemble)
    """
    from keras.models import load_model
    import os
    
    # Load SpliceAI ensemble models from systematic path
    # Follows convention: data/models/<base_model_name>/
    model_dir = "data/models/spliceai/"
    paths = [os.path.join(model_dir, f"spliceai{i}.h5") for i in range(1, 6)]
    models = [load_model(path) for path in paths]
    
    # Note: The context parameter is included for API consistency,
    # but doesn't affect model loading itself, only how they're used
    
    return models
```

#### **2. Context Window Support**
SpliceAI uses a **single ensemble of 5 models** that can handle different context windows:

- **Context Window**: Determined by input sequence length, not separate model files
- **Default Context**: 10,000 bp (recommended)
- **Supported Contexts**: Any length up to model's maximum input size
- **Ensemble**: All 5 models are loaded and used together for prediction

#### **3. Model Validation**
```python
def validate_spliceai_model(model_path: str) -> bool:
    """Validate SpliceAI model file integrity."""
    try:
        from keras.models import load_model
        model = load_model(model_path)
        
        # Check model architecture
        assert model.input_shape[1] == 4  # One-hot encoded DNA
        assert len(model.output_shape) == 2  # Donor and acceptor outputs
        
        return True
    except Exception as e:
        print(f"Model validation failed: {e}")
        return False
```

## 🚀 Usage Examples

### **Basic Usage**
```python
# Load SpliceAI models
from meta_spliceai.splice_engine.meta_models.utils.model_utils import load_spliceai_ensemble

models = load_spliceai_ensemble(context=10_000)
print(f"Loaded {len(models)} SpliceAI model(s)")

# Use in inference workflow
from meta_spliceai.splice_engine.meta_models.workflows.inference.sequence_inference import SequenceInference

inference = SequenceInference(
    models=models,
    mode="base_only",  # Use only base models
    context_window=10_000
)
```

### **Advanced Configuration**
```python
# Load multiple context models
models_5k = load_spliceai_ensemble(context=5_000)
models_10k = load_spliceai_ensemble(context=10_000)
models_50k = load_spliceai_ensemble(context=50_000)

# Combine for ensemble prediction
all_models = models_5k + models_10k + models_50k

# Use in hybrid mode
inference = SequenceInference(
    models=all_models,
    mode="hybrid",  # Base + meta models
    context_window=10_000
)
```

### **Delta Score Computation**
```python
# Compute delta scores for variant analysis
from meta_spliceai.splice_engine.meta_models.workflows.inference.sequence_inference import SequenceInference

# Load models
models = load_spliceai_ensemble(context=10_000)

# Create inference instance
inference = SequenceInference(models=models, mode="base_only")

# Compute delta scores
delta_scores = inference.compute_delta_scores(
    wt_sequence="ATCGATCG...",
    alt_sequence="ATCGATCG...",
    variant_position=1000
)

print(f"Delta scores: {delta_scores}")
```

## 🔍 Model Specifications

### **Input Requirements**
- **Sequence Format**: DNA sequence as string
- **Encoding**: One-hot encoding (A=1000, T=0100, G=0010, C=0001)
- **Context Window**: Must match model training context
- **Sequence Length**: Must be exactly the context window size

### **Output Format**
- **Donor Sites**: Per-nucleotide probability of donor splice sites
- **Acceptor Sites**: Per-nucleotide probability of acceptor splice sites
- **Range**: [0, 1] probability values
- **Resolution**: Per-nucleotide predictions

### **Performance Characteristics**
- **Inference Speed**: ~100-1000 sequences/second (depending on context)
- **Memory Usage**: ~2-8 GB (depending on context and batch size)
- **Accuracy**: State-of-the-art splice site prediction performance

## 🛠️ Configuration Options

### **Environment Variables**
```bash
# Set model directory (follows data/models/<base_model_name>/ convention)
export SPLICEAI_MODEL_DIR="data/models/spliceai/"

# Set default context
export SPLICEAI_DEFAULT_CONTEXT="10000"

# Enable GPU acceleration
export CUDA_VISIBLE_DEVICES="0"
```

### **Model Parameters**
```python
# Model loading parameters
model_config = {
    "context_window": 10_000,
    "batch_size": 32,
    "use_gpu": True,
    "model_dir": "data/models/spliceai/",  # Follows data/models/<base_model_name>/ convention
    "validation": True
}
```

## 🔧 Troubleshooting

### **Common Issues**

#### **1. Model Not Found**
```python
# Error: Model file not found
FileNotFoundError: [Errno 2] No such file or directory: 'data/models/spliceai/spliceai1.h5'

# Solution: Check model path and file existence
import os
model_path = "data/models/spliceai/spliceai1.h5"
if not os.path.exists(model_path):
    print(f"Model not found: {model_path}")
    print("Please ensure SpliceAI models are available in data/models/spliceai/")
    print("This directory should contain spliceai1.h5 through spliceai5.h5")
```

#### **2. Context Mismatch**
```python
# Error: Context window mismatch
ValueError: Expected context 10000, got 5000

# Solution: Use correct context window
models = load_spliceai_ensemble(context=5_000)  # Match your sequence length
```

#### **3. Memory Issues**
```python
# Error: Out of memory
ResourceExhaustedError: OOM when allocating tensor

# Solution: Reduce batch size or use smaller context
model_config = {
    "batch_size": 16,  # Reduce from default 32
    "context_window": 5_000  # Use smaller context
}
```

### **Debug Commands**
```bash
# Check model files (follows data/models/<base_model_name>/ convention)
ls -la data/models/spliceai/

# Validate model integrity
python -c "from meta_spliceai.splice_engine.meta_models.utils.model_utils import load_spliceai_ensemble; models = load_spliceai_ensemble(); print('Models loaded successfully')"

# Check GPU availability
python -c "import tensorflow as tf; print('GPU available:', tf.config.list_physical_devices('GPU'))"
```

## 📊 Performance Monitoring

### **Model Loading Time**
```python
import time

start_time = time.time()
models = load_spliceai_ensemble(context=10_000)
load_time = time.time() - start_time

print(f"Model loading time: {load_time:.2f} seconds")
```

### **Inference Performance**
```python
# Benchmark inference speed
import time

sequences = ["ATCG" * 2500] * 100  # 100 sequences of 10k bp

start_time = time.time()
for seq in sequences:
    predictions = inference.predict(seq)
inference_time = time.time() - start_time

print(f"Inference time: {inference_time:.2f} seconds")
print(f"Sequences per second: {len(sequences) / inference_time:.2f}")
```

## 🔗 Integration Points

### **Meta Model Training**
- SpliceAI predictions serve as features for meta-model training
- Delta scores computed for variant analysis
- Per-nucleotide scores used for error correction

### **Inference Workflows**
- Base model pass in sequence inference
- Delta score computation for variants
- Integration with meta-model predictions

### **Resource Management**
- Model loading and validation
- Path discovery and configuration
- Performance monitoring and optimization

## 📚 References

- **SpliceAI Paper**: [Jaganathan et al., 2019](https://www.cell.com/cell/fulltext/S0092-8674(19)30329-0)
- **Model Repository**: [SpliceAI GitHub](https://github.com/Illumina/SpliceAI)
- **Performance Benchmarks**: [SpliceAI Performance](https://github.com/Illumina/SpliceAI#performance)

---

*This documentation is part of the MetaSpliceAI meta-learning system. For questions or contributions, please refer to the main project documentation.*
