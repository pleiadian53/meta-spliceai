"""
Flexible loader for splice site prediction foundation models.

Supports:
- External packages (installed via pip, e.g., spliceai)
- Internal packages (nested under meta_spliceai.foundation_models)
"""

import importlib
from meta_spliceai.system import FOUNDATION_MODEL as DEFAULT_MODEL
from meta_spliceai.system import MODEL_REGISTRY


def load_model(name: str = None, **model_params):
    model_name = (name or DEFAULT_MODEL).lower()
    
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"No model registered as '{model_name}'. Update MODEL_REGISTRY.")

    # Retrieve the module and class names from the registry
    module_path = MODEL_REGISTRY[model_name]['module_path']
    class_name = MODEL_REGISTRY[model_name]['class_name']

    # Dynamically import the module and retrieve the class
    model_module = importlib.import_module(module_path)
    ModelClass = getattr(model_module, class_name)

    # Instantiate and return the model instance
    return ModelClass(**model_params)


def get_utils_module(model_name: str = None):
    model_name = (model_name or FOUNDATION_MODEL).lower()

    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Model '{model_name}' not registered. Check MODEL_REGISTRY.")

    utils_module_path = MODEL_REGISTRY[model_name]['utils_module']
    return importlib.import_module(utils_module_path)
