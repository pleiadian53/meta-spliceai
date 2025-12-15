"""
System module initialization.

This module provides system-wide configurations and utilities.
"""

# Import the Config class
from meta_spliceai.system.config import Config

# Import or create FOUNDATION_MODEL directly
try:
    # First try to import from the system.config module
    from meta_spliceai.system.config import FOUNDATION_MODEL
except ImportError:
    try:
        # Then try from sys_config.Config if it exists
        from meta_spliceai.system.sys_config.Config import FOUNDATION_MODEL
    except ImportError:
        # If not found, use the value from our Config class
        FOUNDATION_MODEL = Config.FOUNDATION_MODEL

from meta_spliceai.system.config import MODEL_REGISTRY

# Note: BaseModelRunner imports are moved to avoid circular imports
# Import directly from meta_spliceai.system.base_model_runner if needed

# This allows importing directly from the system module:
# from meta_spliceai.system import Config, FOUNDATION_MODEL
__all__ = [
    'Config',
    'FOUNDATION_MODEL',
    'MODEL_REGISTRY'
]