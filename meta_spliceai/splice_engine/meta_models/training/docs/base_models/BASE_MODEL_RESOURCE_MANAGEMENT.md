# Base Model Resource Management Analysis

This document analyzes the current state of base model resource management in MetaSpliceAI and provides recommendations for improvement.

## 🔍 Current State Analysis

### **Model Resource Manager (`model_resource_manager.py`)**

#### **✅ What's Working Well**
- **Meta-Model Focus**: Excellent support for trained meta-models and their resources
- **Systematic Path Resolution**: Robust path discovery and validation
- **Resource Validation**: Comprehensive validation of model resources
- **Feature Schema Management**: Good support for feature manifests and training schemas

#### **❌ Missing Base Model Support**
The current `ModelResourceManager` is **focused on meta-models** and lacks specific support for base models:

```python
# Current model file patterns (meta-model focused)
self.model_file_patterns = {
    "multiclass": ["model_multiclass.pkl", "model_multiclass.joblib"],
    "binary": ["model_binary.pkl", "model_binary.joblib"],
    "best": ["best_model.pkl", "best_model.joblib"],
    "final": ["final_model.pkl", "final_model.joblib"],
    "calibrated": ["calibrated_model.pkl", "calibrated_model.joblib"]
}
```

**Missing Base Model Patterns:**
- No SpliceAI model patterns (`.h5` files)
- No OpenSpliceAI model patterns (`.pth` files)
- No base model directory structure
- No context window management

### **Data Resource Manager (`data_resource_manager.py`)**

#### **✅ Base Model Awareness**
The data resource manager has some base model awareness:

```python
# Base model output directories
"base_model_outputs": self.genomic_manager.genome.get_source_dir("ensembl") / "spliceai_eval",
"meta_models": self.genomic_manager.genome.get_source_dir("ensembl") / "spliceai_eval" / "meta_models",

# Supported base models
SUPPORTED BASE MODELS:
- SpliceAI (current default)
- OpenSpliceAI (already integrated)
```

#### **❌ Limited Base Model Management**
- **No Model Discovery**: No systematic discovery of base model files
- **No Model Loading**: No integration with model loading functions
- **No Context Management**: No support for different context windows
- **No Model Validation**: No validation of base model files

## 🚀 Recommended Improvements

### **1. Extend Model Resource Manager for Base Models**

#### **Add Base Model File Patterns**
```python
# Add to ModelResourceManager.__init__()
self.base_model_patterns = {
    "spliceai": {
        "file_patterns": ["spliceai1.h5", "spliceai2.h5", "spliceai3.h5", "spliceai4.h5", "spliceai5.h5"],
        "context_windows": [10000],  # Single ensemble handles all contexts
        "model_dir": "data/models/spliceai/",
        "metadata_file": "metadata/model_info.json"
    },
    "openspliceai": {
        "file_patterns": ["openspliceai_*.pth"],
        "context_windows": [10000, 50000],
        "model_dir": "data/models/openspliceai/",
        "metadata_file": "metadata/model_info.json"
    }
}
```

#### **Add Base Model Discovery Methods**
```python
def discover_base_models(self, base_model_type: str = "spliceai") -> Dict[str, Any]:
    """
    Discover available base model files for a specific base model type.
    
    Parameters
    ----------
    base_model_type : str
        Type of base model (spliceai, openspliceai)
        
    Returns
    -------
    Dict[str, Any]
        Dictionary with discovered model information
    """
    if base_model_type not in self.base_model_patterns:
        raise ValueError(f"Unsupported base model type: {base_model_type}")
    
    model_config = self.base_model_patterns[base_model_type]
    model_dir = self.project_root / model_config["model_dir"]
    
    discovered_models = {
        "model_directory": str(model_dir),
        "available_models": [],
        "context_windows": [],
        "metadata": None
    }
    
    # Discover model files
    for pattern in model_config["file_patterns"]:
        model_files = list(model_dir.glob(pattern))
        for model_file in model_files:
            # Extract context window from filename
            context = self._extract_context_from_filename(model_file.name)
            discovered_models["available_models"].append({
                "file_path": str(model_file),
                "context_window": context,
                "file_size": model_file.stat().st_size
            })
            if context not in discovered_models["context_windows"]:
                discovered_models["context_windows"].append(context)
    
    # Load metadata if available
    metadata_file = model_dir / model_config["metadata_file"]
    if metadata_file.exists():
        try:
            with open(metadata_file, 'r') as f:
                discovered_models["metadata"] = json.load(f)
        except Exception as e:
            logger.warning(f"Could not load metadata: {e}")
    
    return discovered_models

def _extract_context_from_filename(self, filename: str) -> int:
    """Extract context window from model filename."""
    import re
    
    # Pattern for SpliceAI: spliceai_5k.h5, spliceai_10k.h5, etc.
    match = re.search(r'(\d+)k', filename)
    if match:
        return int(match.group(1)) * 1000
    
    # Pattern for OpenSpliceAI: openspliceai_10000.pth, etc.
    match = re.search(r'(\d+)\.pth', filename)
    if match:
        return int(match.group(1))
    
    return None
```

#### **Add Base Model Validation**
```python
def validate_base_model(self, base_model_type: str, context_window: int = None) -> Dict[str, Any]:
    """
    Validate base model resources for a specific type and context.
    
    Parameters
    ----------
    base_model_type : str
        Type of base model to validate
    context_window : int, optional
        Specific context window to validate
        
    Returns
    -------
    Dict[str, Any]
        Validation results
    """
    validation = {
        "base_model_type": base_model_type,
        "context_window": context_window,
        "model_directory_found": False,
        "model_files_found": False,
        "metadata_found": False,
        "validation_passed": False,
        "issues": [],
        "recommendations": []
    }
    
    try:
        # Discover models
        discovered = self.discover_base_models(base_model_type)
        validation["model_directory_found"] = True
        validation["model_directory"] = discovered["model_directory"]
        
        # Check for specific context window
        if context_window:
            matching_models = [m for m in discovered["available_models"] 
                             if m["context_window"] == context_window]
            if matching_models:
                validation["model_files_found"] = True
                validation["model_file"] = matching_models[0]["file_path"]
            else:
                validation["issues"].append(f"No model found for context {context_window}")
                validation["recommendations"].append(f"Available contexts: {discovered['context_windows']}")
        else:
            # Check for any models
            if discovered["available_models"]:
                validation["model_files_found"] = True
                validation["available_models"] = discovered["available_models"]
            else:
                validation["issues"].append("No model files found")
        
        # Check metadata
        if discovered["metadata"]:
            validation["metadata_found"] = True
            validation["metadata"] = discovered["metadata"]
        
        # Overall validation
        validation["validation_passed"] = (
            validation["model_directory_found"] and 
            validation["model_files_found"]
        )
        
    except Exception as e:
        validation["issues"].append(f"Validation error: {e}")
    
    return validation
```

### **2. Create Base Model Resource Manager**

#### **New Module: `base_model_resource_manager.py`**
```python
#!/usr/bin/env python3
"""
Base Model Resource Manager

Specialized manager for base model resources including:
- SpliceAI model discovery and loading
- OpenSpliceAI model discovery and loading
- Context window management
- Model validation and health checks
"""

class BaseModelResourceManager:
    """Specialized manager for base model resources."""
    
    def __init__(self, project_root: Optional[Union[str, Path]] = None):
        self.project_root = Path(project_root) if project_root else self._find_project_root()
        
        # Base model configurations
        self.base_model_configs = {
            "spliceai": {
                "model_dir": "data/models/spliceai/",
                "file_patterns": ["spliceai_5k.h5", "spliceai_10k.h5", "spliceai_50k.h5"],
                "context_windows": [5000, 10000, 50000],
                "loader_function": "load_spliceai_ensemble",
                "metadata_file": "metadata/model_info.json"
            },
            "openspliceai": {
                "model_dir": "data/models/openspliceai/",
                "file_patterns": ["openspliceai_*.pth"],
                "context_windows": [10000, 50000],
                "loader_function": "load_openspliceai_ensemble",
                "metadata_file": "metadata/model_info.json"
            }
        }
    
    def get_base_model_path(self, base_model_type: str, context_window: int = None) -> Optional[Path]:
        """Get path to base model file."""
        # Implementation here
        pass
    
    def load_base_model(self, base_model_type: str, context_window: int = None) -> Any:
        """Load base model with specified context window."""
        # Implementation here
        pass
    
    def validate_base_model_resources(self, base_model_type: str) -> Dict[str, Any]:
        """Validate base model resources."""
        # Implementation here
        pass
```

### **3. Integration with Existing Systems**

#### **Update Model Utils**
```python
# In model_utils.py
def load_spliceai_ensemble(context: int = 10000) -> List:
    """Load SpliceAI models with systematic path discovery."""
    from meta_spliceai.splice_engine.meta_models.workflows.inference.base_model_resource_manager import BaseModelResourceManager
    
    manager = BaseModelResourceManager()
    model_path = manager.get_base_model_path("spliceai", context)
    
    if not model_path:
        raise FileNotFoundError(f"SpliceAI model not found for context {context}")
    
    # Load model using existing logic
    from keras.models import load_model
    model = load_model(str(model_path))
    return [model]
```

#### **Update Inference Workflows**
```python
# In sequence_inference.py
def __init__(self, models: Optional[List] = None, base_model_type: str = "spliceai", 
             context_window: int = 10000, mode: str = "base_only"):
    """Initialize with systematic base model loading."""
    if models is None:
        from meta_spliceai.splice_engine.meta_models.workflows.inference.base_model_resource_manager import BaseModelResourceManager
        
        manager = BaseModelResourceManager()
        models = manager.load_base_model(base_model_type, context_window)
    
    self.models = models
    self.base_model_type = base_model_type
    self.context_window = context_window
    self.mode = mode
```

## 📋 Implementation Plan

### **Phase 1: Extend Model Resource Manager**
1. Add base model file patterns
2. Add base model discovery methods
3. Add base model validation
4. Update existing methods to support base models

### **Phase 2: Create Base Model Resource Manager**
1. Create specialized base model manager
2. Implement model discovery and loading
3. Add context window management
4. Add model validation and health checks

### **Phase 3: Integration**
1. Update model utils to use new manager
2. Update inference workflows
3. Add configuration management
4. Add documentation and examples

### **Phase 4: Testing and Validation**
1. Test with existing SpliceAI models
2. Test with OpenSpliceAI models
3. Validate error handling
4. Performance testing

## 🎯 Benefits

### **Systematic Model Management**
- Consistent path discovery across base models
- Centralized model validation
- Easy switching between base model types

### **Context Window Management**
- Support for multiple context windows
- Automatic context window detection
- Context-specific model loading

### **Extensibility**
- Easy addition of new base model types
- Consistent interface for all base models
- Future-proof architecture

### **Integration**
- Seamless integration with existing workflows
- Backward compatibility
- Consistent error handling

## 🔗 Related Documentation

- **[SpliceAI Integration](./SPLICEAI_INTEGRATION.md)**: SpliceAI-specific integration guide
- **[Model Loading Architecture](./MODEL_LOADING_ARCHITECTURE.md)**: Technical architecture details
- **[Base Models README](./README.md)**: Overview of base model system

---

*This analysis is part of the MetaSpliceAI meta-learning system. For questions or contributions, please refer to the main project documentation.*
