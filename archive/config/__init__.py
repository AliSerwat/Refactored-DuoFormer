"""
⚙️ DuoFormer Configuration

Configuration management for models and training.
"""

from .model_config import (
    ModelConfig,
    BackboneConfig,
    TransformerConfig,
    MultiScaleConfig,
    TrainingConfig,
    DataConfig,
    LoggingConfig,
    DEFAULT_CONFIG,
    LIGHTWEIGHT_CONFIG,
    PERFORMANCE_CONFIG,
    FINETUNE_CONFIG,
)

__all__ = [
    "ModelConfig",
    "BackboneConfig",
    "TransformerConfig",
    "MultiScaleConfig",
    "TrainingConfig",
    "DataConfig",
    "LoggingConfig",
    "DEFAULT_CONFIG",
    "LIGHTWEIGHT_CONFIG",
    "PERFORMANCE_CONFIG",
    "FINETUNE_CONFIG",
]
