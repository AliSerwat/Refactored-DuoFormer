"""
Lightweight model tests using minimal configurations.

These tests use tiny models to verify functionality without wasting resources.
"""

import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def test_minimal_model_creation():
    """Test model with minimal config (fastest)."""
    print("\n🧪 Testing minimal model creation...")

    try:
        import torch
        from models import build_model_no_extra_params

        # Absolute minimal config for speed
        model = build_model_no_extra_params(
            depth=2,  # Just 2 blocks
            embed_dim=64,  # Tiny embedding
            num_heads=2,
            num_classes=3,
            num_layers=2,
            num_patches=49,
            proj_dim=64,
            backbone="r18",
            pretrained=False,
        )

        assert model is not None
        print(f"   Model: {model.name}")
        print("   ✅ Minimal model creation: PASS")

    except ImportError as e:
        print(f"   ⚠ Skipping (dependencies not installed): {e}")


def test_minimal_forward_pass():
    """Test forward pass with tiny batch."""
    print("\n🧪 Testing minimal forward pass...")

    try:
        import torch
        from models import build_model_no_extra_params

        model = build_model_no_extra_params(
            depth=2,
            embed_dim=64,
            num_heads=2,
            num_classes=3,
            num_layers=2,
            backbone="r18",
            pretrained=False,
        )
        model.eval()

        # Tiny batch for speed
        x = torch.randn(2, 3, 224, 224)  # Just 2 images

        with torch.no_grad():
            output = model(x)

        assert output.shape == (2, 3)
        print(f"   Input: {x.shape} → Output: {output.shape}")
        print("   ✅ Forward pass: PASS")

    except ImportError:
        print("   ⚠ Skipping (dependencies not installed)")


def run_all_tests():
    """Run all lightweight tests."""
    print("\n" + "=" * 60)
    print("Running Lightweight Unit Tests")
    print("=" * 60)
    print("These tests use minimal models for speed")

    try:
        test_minimal_model_creation()
        test_minimal_forward_pass()

        print("\n" + "=" * 60)
        print("✅ All lightweight tests passed!")
        print("=" * 60)
        return True
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
