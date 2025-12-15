"""
MetaSpliceAI Engine Module
"""

# Expose distributed training modules
try:
    from .error_sequence_distributed import (
        set_device,
        log_memory_usage,
        log_gpu_memory_detailed,
        train_with_deepspeed,
        create_dynamic_buckets,
    )
except (ModuleNotFoundError, ImportError):  # e.g., torch missing or cannot load native libs
    def _missing_dep(*_args, **_kwargs):  # noqa: D401, ANN001
        raise ImportError(
            "Distributed training utilities require optional dependencies like 'torch'. "
            "Install them or avoid calling distributed helpers."
        )

    set_device = _missing_dep  # type: ignore
    log_memory_usage = _missing_dep  # type: ignore
    log_gpu_memory_detailed = _missing_dep  # type: ignore
    train_with_deepspeed = _missing_dep  # type: ignore
    create_dynamic_buckets = _missing_dep  # type: ignore

# Expose demo functions
try:
    from .error_sequence_demo import demo_pretrain_finetune_dist
except (ModuleNotFoundError, ImportError, AttributeError, RuntimeError) as e:
    # optional demo depends on torch/transformers/pandas compatibility
    def demo_pretrain_finetune_dist(*_args, **_kwargs):  # type: ignore
        raise ImportError(
            f"demo_pretrain_finetune_dist requires optional dependencies. Error: {e}"
        )

# Version
__version__ = "0.1.0"