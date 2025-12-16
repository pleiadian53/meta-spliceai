# Base Models Documentation

This directory contains documentation for base model integration and management in the MetaSpliceAI meta-learning system.

## 📁 Directory Structure

```
base_models/
├── README.md                           # This file - overview of base models
├── SPLICEAI_INTEGRATION.md            # SpliceAI model loading and usage
├── OPENSEPLICEAI_INTEGRATION.md       # OpenSpliceAI integration (future)
├── MODEL_LOADING_ARCHITECTURE.md      # Technical architecture for model loading
└── BASE_MODEL_RESOURCE_MANAGEMENT.md  # Resource management for base models
```

## 🎯 Overview

Base models serve as the foundation layer for meta-learning in MetaSpliceAI. They provide per-nucleotide splice site predictions that are then used by meta-models to improve accuracy and reduce errors.

### **Supported Base Models**

#### **1. SpliceAI (Current Default)**
- **Status**: ✅ Fully Integrated
- **Documentation**: [SpliceAI Integration](./SPLICEAI_INTEGRATION.md)
- **Default Path**: `data/models/spliceai/` (symlink to package models)
- **Models**: `spliceai1.h5` through `spliceai5.h5` (ensemble)

#### **2. OpenSpliceAI (Planned)**
- **Status**: 🚧 Planned Integration
- **Documentation**: [OpenSpliceAI Integration](./OPENSEPLICEAI_INTEGRATION.md)
- **Expected Path**: `data/models/openspliceai/`

### **Key Features**

- **Systematic Model Loading**: Automatic discovery and loading of pre-trained models
- **Multi-Model Support**: Support for different base model architectures
- **Resource Management**: Centralized management of model files and metadata
- **Context Flexibility**: Support for different context window sizes
- **Delta Score Computation**: Integration with delta score calculation workflows

## 🔧 Technical Architecture

### **Model Loading Flow**
1. **Discovery**: Systematically locate pre-trained model files
2. **Validation**: Verify model files and dependencies
3. **Loading**: Load models with appropriate context windows
4. **Integration**: Make models available to inference workflows

### **Resource Management**
- **Model Files**: Pre-trained model weights and configurations
- **Metadata**: Model specifications, training info, and performance metrics
- **Dependencies**: Required libraries and system requirements
- **Validation**: Health checks and compatibility verification

## 📚 Documentation Files

### **Core Documentation**
- **[SpliceAI Integration](./SPLICEAI_INTEGRATION.md)**: Complete guide to SpliceAI model loading and usage
- **[Model Loading Architecture](./MODEL_LOADING_ARCHITECTURE.md)**: Technical details of the model loading system
- **[Base Model Resource Management](./BASE_MODEL_RESOURCE_MANAGEMENT.md)**: Resource management patterns and best practices

### **Integration Guides**
- **[OpenSpliceAI Integration](./OPENSEPLICEAI_INTEGRATION.md)**: Planned integration for OpenSpliceAI models
- **[Custom Base Models](./CUSTOM_BASE_MODELS.md)**: Guide for integrating new base model types

## 🚀 Quick Start

### **Using SpliceAI (Default)**
```python
from meta_spliceai.splice_engine.meta_models.utils.model_utils import load_spliceai_ensemble

# Load SpliceAI models with default context
models = load_spliceai_ensemble(context=10_000)

# Use in inference workflow
from meta_spliceai.splice_engine.meta_models.workflows.inference.sequence_inference import SequenceInference
inference = SequenceInference(models=models, mode="base_only")
```

### **Model Resource Management**
```python
from meta_spliceai.splice_engine.meta_models.workflows.inference.model_resource_manager import ModelResourceManager

# Create resource manager
manager = ModelResourceManager()

# Validate model resources
validation = manager.validate_model_resources("data/models/spliceai/")
print(validation["summary"])
```

## 🔍 Model Discovery

The system automatically discovers base models using these patterns:

### **SpliceAI Models**
- **Path Pattern**: `data/models/spliceai/` (symlink to package)
- **File Pattern**: `spliceai1.h5` through `spliceai5.h5` (ensemble)
- **Context Support**: Single ensemble handles all context windows

### **OpenSpliceAI Models (Planned)**
- **Path Pattern**: `data/models/openspliceai/`
- **File Pattern**: `*.pth` (PyTorch models)
- **Context Support**: Configurable context windows

## 📋 Best Practices

### **Model Organization**
- Store models in systematic directory structures
- Use consistent naming conventions
- Maintain model metadata and documentation
- Version control model configurations

### **Resource Management**
- Use the ModelResourceManager for systematic access
- Validate model resources before use
- Monitor model loading performance
- Implement proper error handling

### **Integration Patterns**
- Follow the established model loading interfaces
- Support multiple context window sizes
- Enable easy switching between base models
- Maintain backward compatibility

## 🔗 Related Documentation

- **[Meta Model Training](../README.md)**: Overview of meta-model training
- **[Inference Workflows](../../workflows/inference/)**: Inference workflow documentation
- **[Model Utils](../../utils/)**: Utility functions for model management
- **[Resource Management](../../../case_studies/data_sources/)**: Data resource management patterns

## 📝 Contributing

When adding new base model support:

1. **Create Integration Guide**: Document the new model in this directory
2. **Update Resource Manager**: Add model discovery patterns
3. **Implement Loading Logic**: Add model loading functions
4. **Update Documentation**: Update this README and related files
5. **Add Tests**: Ensure proper testing of new functionality

## 🐛 Troubleshooting

### **Common Issues**
- **Model Not Found**: Check model path configuration
- **Loading Errors**: Verify model file integrity and dependencies
- **Context Mismatch**: Ensure context window compatibility
- **Memory Issues**: Monitor memory usage during model loading

### **Debug Commands**
```bash
# Validate model resources
python -m meta_spliceai.splice_engine.meta_models.workflows.inference.model_resource_manager --validate data/models/spliceai/

# Check model loading
python -c "from meta_spliceai.splice_engine.meta_models.utils.model_utils import load_spliceai_ensemble; print('Models loaded successfully')"
```

---

*This documentation is part of the MetaSpliceAI meta-learning system. For questions or contributions, please refer to the main project documentation.*
