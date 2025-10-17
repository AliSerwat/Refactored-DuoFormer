"""
DuoFormer: Multi-Scale Vision Transformer for Medical Imaging

A production-ready implementation of DuoFormer for general medical image classification
with enterprise-grade MLOps practices.

Original work: https://github.com/xiaoyatang/duoformer_TCGA
Refactored by: https://github.com/AliSerwat/Refactored-DuoFormer
"""

import sys

# Runtime Python version check
if sys.version_info < (3, 10):
    raise RuntimeError(
        f"DuoFormer requires Python 3.10 or higher. "
        f"You are using Python {sys.version_info.major}.{sys.version_info.minor}. "
        f"Please upgrade your Python installation."
    )

__version__ = "1.1.0"
__author__ = "AliSerwat"
__email__ = "ali.serwat@example.com"
__description__ = "Multi-Scale Vision Transformer for Medical Imaging"


# Lazy imports to avoid circular dependencies
def __getattr__(name):
    """Lazy import for better performance and to avoid circular dependencies."""
    if name == "build_model_no_extra_params":
        from .models import build_model_no_extra_params

        return build_model_no_extra_params
    elif name == "count_parameters":
        from .models import count_parameters

        return count_parameters
    elif name in [
        "ModelConfig",
        "DEFAULT_CONFIG",
        "LIGHTWEIGHT_CONFIG",
        "PERFORMANCE_CONFIG",
    ]:
        from .config import (
            ModelConfig,
            DEFAULT_CONFIG,
            LIGHTWEIGHT_CONFIG,
            PERFORMANCE_CONFIG,
        )

        return locals()[name]
    elif name == "Trainer":
        from .training.trainer import Trainer

        return Trainer
    elif name == "setup_device_environment":
        from .utils.device import setup_device_environment

        return setup_device_environment
    elif name == "setup_logging":
        from .utils.logging import setup_logging

        return setup_logging
    else:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
    # Models
    "build_model_no_extra_params",
    "count_parameters",
    # Configuration
    "ModelConfig",
    "DEFAULT_CONFIG",
    "LIGHTWEIGHT_CONFIG",
    "PERFORMANCE_CONFIG",
    # Training
    "Trainer",
    # Utilities
    "setup_device_environment",
    "setup_logging",
    # Metadata
    "__version__",
    "__author__",
    "__email__",
    "__description__",
]
