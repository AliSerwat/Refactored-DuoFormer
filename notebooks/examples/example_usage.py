#!/usr/bin/env python3
"""
Refactored DuoFormer - Example Usage

Demonstrates all features for general medical image classification.

This script shows:
- Model creation for various medical imaging tasks
- Configuration loading
- Training with professional Trainer class
- Checkpoint management
- TensorBoard logging

Usage:
    python examples/example_usage.py

Original work: https://github.com/xiaoyatang/duoformer_TCGA
"""

import torch
import torch.nn as nn
from pathlib import Path
import sys

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))


def example_model_creation():
    """Example: Create DuoFormer model."""
    print("\n" + "=" * 60)
    print("Example 1: Model Creation")
    print("=" * 60)

    from models import build_model_no_extra_params, count_parameters

    # Create model
    model = build_model_no_extra_params(
        depth=12,
        embed_dim=768,
        num_heads=12,
        num_classes=10,
        num_layers=2,
        num_patches=49,
        proj_dim=768,
        freeze_backbone=True,
        backbone="r50",
        pretrained=False,  # Set True to download pretrained weights
    )

    # Count parameters
    trainable, total = count_parameters(model)
    print(f"\nModel: {model.name}")
    print(f"Trainable parameters: {trainable:.2f}M")
    print(f"Total parameters: {total:.2f}M")

    return model


def example_configuration():
    """Example: Load configuration."""
    print("\n" + "=" * 60)
    print("Example 2: Configuration Management")
    print("=" * 60)

    from config import ModelConfig, DEFAULT_CONFIG, LIGHTWEIGHT_CONFIG

    # Use default config
    config = DEFAULT_CONFIG
    print(f"\nBackbone: {config.backbone.name}")
    print(f"Depth: {config.transformer.depth}")
    print(f"Learning rate: {config.training.learning_rate}")

    # Or load from YAML
    yaml_path = Path("config/default_config.yaml")
    if yaml_path.exists():
        config = ModelConfig.from_yaml(str(yaml_path))
        print(f"\nLoaded from YAML: {yaml_path}")

    return config


def example_synthetic_data():
    """Example: Create synthetic dataset for testing."""
    print("\n" + "=" * 60)
    print("Example 3: Synthetic Dataset")
    print("=" * 60)

    import torch
    from torch.utils.data import TensorDataset, DataLoader
    import numpy as np

    # Create synthetic data
    num_samples = 100
    num_classes = 10
    image_size = 224

    np.random.seed(42)
    images = torch.randn(num_samples, 3, image_size, image_size)
    labels = torch.randint(0, num_classes, (num_samples,))

    # Create dataset and loader
    dataset = TensorDataset(images, labels)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    print(f"\nDataset size: {len(dataset)}")
    print(f"Batches: {len(loader)}")
    print(f"Batch shape: {next(iter(loader))[0].shape}")

    return loader


def example_training():
    """Example: Training with Trainer class."""
    print("\n" + "=" * 60)
    print("Example 4: Training Pipeline")
    print("=" * 60)

    try:
        import torch
        import torch.nn as nn
        from models import build_model_no_extra_params
        from utils import (
            Trainer,
            create_optimizer,
            create_scheduler,
            setup_device_environment,
        )

        # Setup (hardware-agnostic)
        device = setup_device_environment("auto", verbose=True)

        # Create model
        model = build_model_no_extra_params(
            depth=6,
            embed_dim=384,
            num_heads=6,
            num_classes=10,
            num_layers=2,
            backbone="r18",
            pretrained=False,
        ).to(device)

        # Create optimizer and scheduler
        optimizer = create_optimizer(model, "adamw", lr=1e-4)
        scheduler = create_scheduler(optimizer, "cosine", epochs=10)

        # Create trainer
        trainer = Trainer(
            model=model,
            criterion=nn.CrossEntropyLoss(),
            optimizer=optimizer,
            device=device,
            scheduler=scheduler,
            checkpoint_dir="./example_checkpoints",
            use_amp=False,
        )

        print("\nTrainer created successfully!")
        print(f"Checkpoint dir: {trainer.checkpoint_dir}")
        print(f"TensorBoard: {trainer.writer.log_dir}")

        # Note: To actually train, you would call:
        # trainer.fit(train_loader, val_loader, epochs=10)

        return trainer

    except ImportError as e:
        print(f"\nMissing dependencies: {e}")
        print("Run: python setup_environment.py")
        return None


def example_config_export():
    """Example: Export configuration."""
    print("\n" + "=" * 60)
    print("Example 5: Configuration Export")
    print("=" * 60)

    try:
        from config import ModelConfig, DEFAULT_CONFIG

        # Modify config
        config = DEFAULT_CONFIG
        config.exp_name = "my_experiment"
        config.training.epochs = 50
        config.training.batch_size = 64

        # Save to YAML
        output_path = "config/my_custom_config.yaml"
        config.to_yaml(output_path)
        print(f"\nConfiguration saved to: {output_path}")

        # Save to JSON
        json_path = "config/my_custom_config.json"
        config.to_json(json_path)
        print(f"Configuration saved to: {json_path}")

        return config
    except Exception as e:
        print(f"\nError: {e}")
        return None


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("Refactored DuoFormer - Usage Examples")
    print("=" * 60)

    try:
        # Example 1: Model creation
        model = example_model_creation()

        # Example 2: Configuration
        config = example_configuration()

        # Example 3: Synthetic data
        loader = example_synthetic_data()

        # Example 4: Training (requires dependencies)
        trainer = example_training()

        # Example 5: Config export
        exported_config = example_config_export()

        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        print("=" * 60)
        print("\nYou can now:")
        print("  1. Modify config files in config/")
        print("  2. Run training: python train.py --config config/default_config.yaml")
        print("  3. Monitor with: tensorboard --logdir=runs")

    except ImportError as e:
        print(f"\n\nMissing dependencies: {e}")
        print("\nPlease install dependencies first:")
        print("  python setup_environment.py")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
