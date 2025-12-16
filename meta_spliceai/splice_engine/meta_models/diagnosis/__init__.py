"""
Diagnostic tools for meta-model training and evaluation.

This package provides utilities for:
- Memory usage assessment before training
- System resource validation
- Dataset size analysis
- OOM risk evaluation
"""

from .memory_checker import MemoryChecker, assess_oom_risk

__all__ = ['MemoryChecker', 'assess_oom_risk']
