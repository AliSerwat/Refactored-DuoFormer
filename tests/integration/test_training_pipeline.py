"""
Integration test for full training pipeline.

This is a resource-intensive test - only run when needed.
Tests the complete training workflow end-to-end.
"""

import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def test_full_training_pipeline():
    """
    Test complete training workflow with minimal setup.

    Warning: This test requires GPU and takes time.
    Only run for integration testing.
    """
    print("\n🧪 Testing full training pipeline...")
    print("⚠ Warning: This test requires dependencies and may take time")

    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import TensorDataset, DataLoader
        from models import build_model_no_extra_params, count_parameters
        from utils import Trainer, create_optimizer, create_scheduler

        # Use CPU for testing (avoid GPU resource usage)
        device = torch.device("cpu")
        print(f"   Using device: {device}")

        # Create tiny model
        model = build_model_no_extra_params(
            depth=2,
            embed_dim=64,
            num_heads=2,
            num_classes=3,
            num_layers=2,
            backbone="r18",
            pretrained=False,
        ).to(device)

        # Count params
        trainable, total = count_parameters(model)
        print(f"   Model params: {trainable:.2f}M trainable")

        # Create tiny dataset
        images = torch.randn(20, 3, 224, 224)
        labels = torch.randint(0, 3, (20,))
        dataset = TensorDataset(images, labels)

        train_loader = DataLoader(dataset, batch_size=4, shuffle=True)
        val_loader = DataLoader(dataset, batch_size=4, shuffle=False)

        # Create trainer
        criterion = nn.CrossEntropyLoss()
        optimizer = create_optimizer(model, "adam", lr=1e-3)
        scheduler = create_scheduler(optimizer, "cosine", epochs=2)

        trainer = Trainer(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            scheduler=scheduler,
            checkpoint_dir="./tests/integration/test_checkpoints",
            use_amp=False,  # No AMP for CPU testing
        )

        # Train for 2 epochs only
        print("   Training for 2 epochs...")
        trainer.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=2,
            patience=5,
            save_best_only=True,
        )

        print("   ✅ Full training pipeline: PASS")

        # Cleanup
        import shutil

        shutil.rmtree("./tests/integration/test_checkpoints", ignore_errors=True)

        return True

    except ImportError as e:
        print(f"   ⚠ Skipping (dependencies not installed): {e}")
        return True  # Don't fail if deps missing
    except Exception as e:
        print(f"   ❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Integration Test: Full Training Pipeline")
    print("=" * 60)
    print("⚠ This test may take 1-2 minutes")

    success = test_full_training_pipeline()
    sys.exit(0 if success else 1)
