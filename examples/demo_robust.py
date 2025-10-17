#!/usr/bin/env python3
"""
Refactored DuoFormer - Robust Demo

Platform and hardware agnostic demo for general medical imaging.
This script demonstrates automatic hardware detection and optimization.

Features:
- Auto-detects best available device (CUDA/MPS/CPU)
- Platform-specific DataLoader settings
- Optimal batch size and worker count
- Memory checking
- Works on Windows, Linux, macOS

Usage:
    python demo_robust.py
"""

import torch
import torch.nn as nn
from pathlib import Path
import sys

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from duoformer.models import build_model_no_extra_params, count_parameters
from utils import (
    setup_device_environment,
    print_platform_info,
    print_device_info,
    get_optimal_num_workers,
    get_platform_specific_settings,
)


def main():
    """Run robust demo."""
    print("\n" + "=" * 80)
    print("🚀 Refactored DuoFormer - Robust Demo (Platform-Agnostic)")
    print("=" * 80)

    # Step 1: Print platform and device info
    print("\n📊 System Information:")
    print_platform_info()
    print_device_info()

    # Step 2: Auto-configure device
    print("\n🔧 Configuring Device...")
    device = setup_device_environment("auto", verbose=False)
    print(f"   Selected device: {device}")

    # Step 3: Get platform-specific settings
    settings = get_platform_specific_settings()
    print(f"\n⚙️  Platform-Specific Settings:")
    print(f"   Optimal num_workers: {settings['num_workers']}")
    print(f"   Pin memory: {settings['pin_memory']}")
    print(f"   Persistent workers: {settings['persistent_workers']}")

    # Step 4: Create model
    print(f"\n🏗️  Building Model...")
    model = build_model_no_extra_params(
        depth=6,  # Smaller for demo
        embed_dim=384,
        num_heads=6,
        num_classes=10,
        num_layers=2,
        proj_dim=384,
        backbone="r18",  # Lighter backbone for demo
        pretrained=False,  # Skip download for demo
        freeze_backbone=True,
    )
    model = model.to(device)

    # Count parameters
    trainable, total = count_parameters(model)
    print(f"   Model: {model.name}")
    print(f"   Trainable params: {trainable:.2f}M")
    print(f"   Total params: {total:.2f}M")

    # Step 5: Test forward pass
    print(f"\n🧪 Testing Forward Pass...")
    batch_size = 4
    x = torch.randn(batch_size, 3, 224, 224, device=device)

    with torch.no_grad():
        model.eval()
        output = model(x)

    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Forward pass: SUCCESS ✅")

    # Step 6: Memory info
    if device.type == "cuda":
        memory_allocated = torch.cuda.memory_allocated(device) / 1e9
        memory_reserved = torch.cuda.memory_reserved(device) / 1e9
        print(f"\n💾 GPU Memory Usage:")
        print(f"   Allocated: {memory_allocated:.3f} GB")
        print(f"   Reserved: {memory_reserved:.3f} GB")

    # Success
    print("\n" + "=" * 80)
    print("✅ Demo Complete - All Systems Working!")
    print("=" * 80)
    print("\n📋 Your system is ready for:")
    print(f"   • Device: {device}")
    print(f"   • Recommended batch size: 16-32 (adjust based on memory)")
    print(f"   • Recommended num_workers: {settings['num_workers']}")
    print("\n💡 Next steps:")
    print("   python train.py --help")
    print("   jupyter notebook demo_duoformer.ipynb")
    print("\n")


if __name__ == "__main__":
    try:
        main()
    except ImportError as e:
        print(f"\n❌ Missing dependencies: {e}")
        print("\nPlease install dependencies first:")
        print("  python setup_environment.py")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
