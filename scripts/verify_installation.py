#!/usr/bin/env python3
"""
Refactored DuoFormer - Installation Verification

Verifies that all components can be imported and initialized correctly.
Run after installing dependencies.

Original work: https://github.com/xiaoyatang/duoformer_TCGA
"""

import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))


def verify_imports():
    """Verify all critical imports work."""
    print("Checking imports...")

    try:
        # Core dependencies
        import torch
        import torchvision
        import timm
        import einops
        import matplotlib
        import numpy
        import PIL
        import tqdm
        import sklearn

        print("  Core dependencies: OK")

        # Project modules
        from models import build_model_no_extra_params, count_parameters

        print("  Models module: OK")

        from utils import Trainer, create_optimizer, create_scheduler

        print("  Utils module: OK")

        from config import ModelConfig, DEFAULT_CONFIG

        print("  Config module: OK")

        return True
    except ImportError as e:
        print(f"  Import error: {e}")
        print("\n  Run: python setup_environment.py")
        return False


def verify_model_creation():
    """Verify model can be created."""
    print("\nVerifying model creation...")

    try:
        import torch
        from models import build_model_no_extra_params

        model = build_model_no_extra_params(
            depth=6,
            embed_dim=384,
            num_heads=6,
            num_classes=10,
            num_layers=2,
            backbone="r18",
            pretrained=False,
        )
        print("  Model creation: OK")

        # Test forward pass
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            output = model(x)
        print(f"  Forward pass: OK ({x.shape} -> {output.shape})")

        return True
    except Exception as e:
        print(f"  Error: {e}")
        return False


def main():
    """Main verification."""
    print("=" * 60)
    print("Refactored DuoFormer - Installation Verification")
    print("=" * 60 + "\n")

    imports_ok = verify_imports()

    if imports_ok:
        model_ok = verify_model_creation()

        if model_ok:
            print("\n" + "=" * 60)
            print("✅ Status: ALL CHECKS PASSED")
            print("=" * 60)
            print("\nYour installation is working correctly!")
            print("\nNext steps:")
            print("  1. Try demo: jupyter notebook demo_duoformer.ipynb")
            print("  2. Run fast tests: python tests/run_tests.py --unit")
            print("  3. Check system: python scripts/check_system.py")
            print("  4. Train model: python train.py --help")
            return True

    print("\n" + "=" * 60)
    print("Status: INSTALLATION INCOMPLETE")
    print("=" * 60)
    print("\nPlease run: python setup_environment.py")
    return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
