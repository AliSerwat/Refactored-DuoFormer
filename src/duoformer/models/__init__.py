"""
DuoFormer Models Package

This package contains all model-related components including:
- Backbone networks (ResNet-18, ResNet-50)
- Multi-scale transformers
- Attention mechanisms
- Projection heads
- Main DuoFormer model
"""

from .model_wo_extra_params import MyModel_no_extra_params

# Re-export for backward compatibility
build_model_no_extra_params = MyModel_no_extra_params


def count_parameters(model):
    """Count trainable and total parameters in a model."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


__all__ = [
    "MyModel_no_extra_params",
    "build_model_no_extra_params",
    "count_parameters",
]
