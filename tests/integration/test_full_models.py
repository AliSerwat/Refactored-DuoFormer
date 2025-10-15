"""
🧪 Refactored DuoFormer - Full Model Integration Tests

Complete model initialization and forward pass tests.
These tests may download pretrained weights and take longer.

Usage:
    python tests/integration/test_full_models.py
    pytest tests/integration/test_full_models.py -v

Original work: https://github.com/xiaoyatang/duoformer_TCGA
"""

import torch
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import build_model_no_extra_params, count_parameters


def test_model_init_resnet50():
    """Test model initialization with ResNet-50 backbone."""
    print("\n🧪 Testing ResNet-50 model initialization...")

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
        pretrained=False,  # Don't download weights for testing
    )

    assert model is not None
    assert model.num_layers == 2
    assert model.proj_dim == 768
    print("   ✅ Model initialized successfully")


def test_model_forward():
    """Test model forward pass."""
    print("\n🧪 Testing model forward pass...")

    model = build_model_no_extra_params(
        depth=6,  # Smaller for faster testing
        embed_dim=384,
        num_heads=6,
        num_classes=10,
        num_layers=2,
        num_patches=49,
        proj_dim=384,
        freeze_backbone=True,
        backbone="r18",  # Smaller backbone
        pretrained=False,
    )

    model.eval()

    # Create dummy input
    batch_size = 4
    x = torch.randn(batch_size, 3, 224, 224)

    # Forward pass
    with torch.no_grad():
        output = model(x)

    assert output.shape == (batch_size, 10)
    print(f"   ✅ Forward pass successful: {x.shape} -> {output.shape}")


def test_parameter_count():
    """Test parameter counting."""
    print("\n🧪 Testing parameter counting...")

    model = build_model_no_extra_params(
        depth=12,
        embed_dim=768,
        num_heads=12,
        num_classes=10,
        num_layers=2,
        proj_dim=768,
        freeze_backbone=True,
        backbone="r50",
        pretrained=False,
    )

    trainable_params, total_params = count_parameters(model)

    assert trainable_params > 0
    assert total_params >= trainable_params
    print(f"   ✅ Trainable: {trainable_params:.2f}M, Total: {total_params:.2f}M")


def test_different_backbones():
    """Test different backbone architectures."""
    print("\n🧪 Testing different backbones...")

    backbones = ["r50", "r18"]

    for backbone in backbones:
        model = build_model_no_extra_params(
            depth=6,
            embed_dim=384,
            num_heads=6,
            num_classes=10,
            num_layers=2,
            backbone=backbone,
            pretrained=False,
        )
        assert model is not None
        print(f"   ✅ {backbone} backbone initialized")


def test_different_scales():
    """Test different number of scales."""
    print("\n🧪 Testing different scales...")

    for num_layers in [2, 3, 4]:
        model = build_model_no_extra_params(
            depth=6,
            embed_dim=384,
            num_heads=6,
            num_classes=10,
            num_layers=num_layers,
            backbone="r18",
            pretrained=False,
        )
        assert model.num_layers == num_layers
        print(f"   ✅ {num_layers} scales initialized")


def run_all_tests():
    """Run all model integration tests."""
    print("=" * 80)
    print("🚀 Running Refactored DuoFormer - Full Model Tests")
    print("=" * 80)

    try:
        test_model_init_resnet50()
        test_model_forward()
        test_parameter_count()
        test_different_backbones()
        test_different_scales()

        print("\n" + "=" * 80)
        print("✅ All tests passed!")
        print("=" * 80)
        return True

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
