"""
Mock data generators for efficient testing.

Creates lightweight synthetic data without requiring actual datasets or GPUs.
"""

import torch
import numpy as np
from typing import Tuple, Optional


def create_mock_image_batch(
    batch_size: int = 4,
    channels: int = 3,
    height: int = 224,
    width: int = 224,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Create mock image batch for testing.

    Args:
        batch_size: Number of images
        channels: Number of channels
        height: Image height
        width: Image width
        device: Device ('cpu' recommended for tests)

    Returns:
        Mock image tensor
    """
    return torch.randn(batch_size, channels, height, width, device=device)


def create_mock_labels(
    batch_size: int = 4, num_classes: int = 10, device: str = "cpu"
) -> torch.Tensor:
    """
    Create mock labels for testing.

    Args:
        batch_size: Number of labels
        num_classes: Number of classes
        device: Device ('cpu' recommended for tests)

    Returns:
        Mock label tensor
    """
    return torch.randint(0, num_classes, (batch_size,), device=device)


def create_mock_dataset(
    num_samples: int = 100, num_classes: int = 10, image_size: int = 224
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create lightweight mock dataset for testing.

    Args:
        num_samples: Number of samples
        num_classes: Number of classes
        image_size: Image size

    Returns:
        Tuple of (images, labels)
    """
    images = torch.randn(num_samples, 3, image_size, image_size)
    labels = torch.randint(0, num_classes, (num_samples,))
    return images, labels


class MockModel(torch.nn.Module):
    """
    Lightweight mock model for testing training pipelines without heavy computation.
    """

    def __init__(self, input_dim: int = 768, num_classes: int = 10):
        super().__init__()
        self.fc = torch.nn.Linear(input_dim, num_classes)
        self.name = "MockModel"
        self.num_layers = 1
        self.proj_dim = input_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Simple pooling + FC for fast testing
        if x.dim() == 4:  # (B, C, H, W)
            x = x.mean(dim=[2, 3])  # Global average pooling
        return self.fc(x)


def get_minimal_model_config() -> dict:
    """
    Get minimal configuration for fast testing.

    Returns:
        Configuration dictionary with minimal settings
    """
    return {
        "depth": 2,  # Minimal depth
        "embed_dim": 64,  # Small embedding
        "num_heads": 2,  # Minimal heads
        "num_classes": 5,  # Few classes
        "num_layers": 2,  # Minimal scales
        "num_patches": 49,
        "proj_dim": 64,
        "mlp_ratio": 2.0,  # Smaller MLP
        "attn_drop_rate": 0.0,
        "proj_drop_rate": 0.0,
        "freeze_backbone": True,
        "backbone": "r18",  # Lightest backbone
        "pretrained": False,  # Don't download weights
    }
