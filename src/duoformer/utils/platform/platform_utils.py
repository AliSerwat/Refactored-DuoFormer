"""
🌐 Platform Utilities

Platform and system detection utilities for cross-platform compatibility.

Features:
- OS detection
- CPU info
- Memory detection
- Optimal settings per platform
"""

import os
import sys
import platform
from typing import Dict, Any
import multiprocessing


def get_platform_info() -> Dict[str, Any]:
    """
    Get comprehensive platform information.

    Returns:
        Dictionary with platform details
    """
    info = {
        "system": platform.system(),  # 'Windows', 'Linux', 'Darwin' (macOS)
        "platform": platform.platform(),
        "machine": platform.machine(),  # 'x86_64', 'AMD64', etc.
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "os_name": os.name,  # 'nt' (Windows), 'posix' (Unix)
        "cpu_count": multiprocessing.cpu_count(),
    }

    # Try to get more detailed CPU info
    try:
        if hasattr(os, "sched_getaffinity"):
            info["available_cpus"] = len(os.sched_getaffinity(0))
        else:
            info["available_cpus"] = multiprocessing.cpu_count()
    except:
        info["available_cpus"] = multiprocessing.cpu_count()

    # Memory info
    try:
        import psutil

        vm = psutil.virtual_memory()
        info["total_memory_gb"] = vm.total / 1e9
        info["available_memory_gb"] = vm.available / 1e9
    except ImportError:
        info["total_memory_gb"] = None
        info["available_memory_gb"] = None

    return info


def is_windows() -> bool:
    """Check if running on Windows."""
    return platform.system() == "Windows" or os.name == "nt"


def is_linux() -> bool:
    """Check if running on Linux."""
    return platform.system() == "Linux"


def is_macos() -> bool:
    """Check if running on macOS."""
    return platform.system() == "Darwin"


def get_optimal_batch_size(
    model_params_millions: float,
    gpu_memory_gb: float = 8.0,
    image_size: int = 224,
    safety_factor: float = 0.7,
) -> int:
    """
    Estimate optimal batch size based on model size and GPU memory.

    Args:
        model_params_millions: Model parameters in millions
        gpu_memory_gb: Available GPU memory in GB
        image_size: Input image size
        safety_factor: Safety factor (0.7 = use 70% of memory)

    Returns:
        Recommended batch size

    Note:
        This is a rough estimate. Actual optimal batch size depends on:
        - Model architecture
        - Activation memory
        - Gradient storage
        - Optimizer state
    """
    # Rough estimation formula
    # Memory per image ≈ (image_size^2 * 3 * 4 bytes) + (model_params * 4 * 3)
    # Factor of 3 for gradients and optimizer states

    bytes_per_param = 4  # float32
    image_memory = (image_size**2) * 3 * bytes_per_param / 1e9
    model_memory = model_params_millions * bytes_per_param * 3 / 1000

    available_memory = gpu_memory_gb * safety_factor

    # Estimate batch size
    estimated_batch = int(available_memory / (image_memory + model_memory * 0.1))

    # Round to nearest power of 2 for efficiency
    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    optimal = max([b for b in batch_sizes if b <= estimated_batch], default=1)

    return optimal


def get_platform_specific_settings() -> Dict[str, Any]:
    """
    Get platform-specific optimal settings.

    Returns:
        Dictionary with recommended settings
    """
    settings = {
        "num_workers": get_optimal_num_workers(),
        "pin_memory": True,
        "persistent_workers": False,
    }

    if is_windows():
        # Windows specific
        settings["pin_memory"] = False  # Can cause issues on Windows
        settings["num_workers"] = min(4, multiprocessing.cpu_count())
        settings["persistent_workers"] = False
    elif is_linux():
        # Linux specific
        settings["pin_memory"] = True
        settings["num_workers"] = min(8, multiprocessing.cpu_count())
        settings["persistent_workers"] = True
    elif is_macos():
        # macOS specific
        settings["pin_memory"] = True
        settings["num_workers"] = min(4, multiprocessing.cpu_count())
        settings["persistent_workers"] = False

    return settings


def get_optimal_num_workers(max_workers: int = 8) -> int:
    """
    Get optimal number of DataLoader workers for current platform.

    Args:
        max_workers: Maximum number of workers

    Returns:
        Optimal number of workers
    """
    cpu_count = multiprocessing.cpu_count()

    if is_windows():
        # Windows can have issues with multi-process loading
        return min(4, cpu_count, max_workers)
    elif is_macos():
        # macOS works well but don't over-subscribe
        return min(4, cpu_count // 2, max_workers)
    else:  # Linux
        # Linux handles multi-process well
        return min(cpu_count, max_workers)


def print_platform_info():
    """Print platform information."""
    info = get_platform_info()
    settings = get_platform_specific_settings()

    print("\n" + "=" * 70)
    print("🌐 Platform Information")
    print("=" * 70)
    print(f"\nSystem: {info['system']}")
    print(f"Platform: {info['platform']}")
    print(f"Machine: {info['machine']}")
    print(f"Python: {info['python_version']} ({info['python_implementation']})")
    print(f"\nCPU Cores: {info['cpu_count']}")
    print(f"Available CPUs: {info['available_cpus']}")

    if info["total_memory_gb"]:
        print(f"\nTotal Memory: {info['total_memory_gb']:.2f} GB")
        print(f"Available Memory: {info['available_memory_gb']:.2f} GB")

    print(f"\n📊 Recommended Settings for this Platform:")
    print(f"   num_workers: {settings['num_workers']}")
    print(f"   pin_memory: {settings['pin_memory']}")
    print(f"   persistent_workers: {settings['persistent_workers']}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    print_platform_info()

    # Also print device info
    try:
        import torch

        try:
            from ..device.device_utils import print_device_info

            print_device_info()
        except ImportError:
            print("Install PyTorch to see GPU information:")
            print("  python setup_environment.py")
    except ImportError:
        print("Install PyTorch to see GPU information:")
        print("  python setup_environment.py")
