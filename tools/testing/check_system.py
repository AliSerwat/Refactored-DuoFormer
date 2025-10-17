#!/usr/bin/env python3
"""
Refactored DuoFormer - System Checker

Checks system capabilities and provides hardware-specific recommendations
for medical image classification tasks.

Original work: https://github.com/xiaoyatang/duoformer_TCGA
"""

import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


def main():
    """Check system and provide recommendations."""
    print("\n" + "=" * 80)
    print("🔍 Refactored DuoFormer - System Check")
    print("=" * 80)

    # Check if dependencies installed
    try:
        import torch
        from utils import print_platform_info, print_device_info
        from utils.platform_utils import get_optimal_batch_size

        # Print info
        print_platform_info()
        print_device_info()

        # Recommendations
        print("\n" + "=" * 80)
        print("💡 Recommendations for Your System")
        print("=" * 80)

        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9

            # Estimate batch sizes for different configs
            configs = [
                ("Lightweight (ResNet-18, 2 scales)", 30),
                ("Standard (ResNet-50, 2 scales)", 50),
                ("Performance (ResNet-50, 4 scales)", 70),
            ]

            print(f"\nGPU Memory: {gpu_memory:.2f} GB")
            print("\nRecommended Batch Sizes:")

            for config_name, model_size in configs:
                batch = get_optimal_batch_size(model_size, gpu_memory)
                print(f"   • {config_name}: batch_size={batch}")

            print("\nRecommended Commands:")
            print(f"   # Lightweight")
            print(
                f"   python train.py --config config/lightweight_config.yaml --batch_size 64"
            )
            print(f"   # Standard")
            print(
                f"   python train.py --config config/default_config.yaml --batch_size 32"
            )
            print(f"   # Performance")
            print(
                f"   python train.py --config config/performance_config.yaml --batch_size 16 --amp"
            )
        else:
            print("\nNo GPU detected - CPU recommendations:")
            print("   • Use lightweight config")
            print("   • Reduce batch size to 8-16")
            print("   • Consider using fewer transformer layers")
            print("\nRecommended Command:")
            print(
                "   python train.py --config config/lightweight_config.yaml --batch_size 8 --device cpu"
            )

        print("\n" + "=" * 80)
        print("✅ System check complete!")
        print("=" * 80 + "\n")

    except ImportError as e:
        print(f"\n❌ Dependencies not installed: {e}")
        print("\nPlease run:")
        print("   python setup_environment.py")
        print("\nThen run this script again.")
        sys.exit(1)


if __name__ == "__main__":
    main()
