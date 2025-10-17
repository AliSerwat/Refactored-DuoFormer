"""
DuoFormer Utilities Package

This package contains utility functions for device management, platform detection,
and other helper functions.
"""

# Import functions from their respective modules
from .device import (
    get_device,
    get_optimal_num_workers,
    setup_device_environment,
    print_device_info,
    get_device_info,
    check_gpu_memory,
    auto_select_device,
)

from .platform import (
    is_windows,
    is_linux,
    is_macos,
    get_platform_specific_settings,
    print_platform_info,
    get_platform_info,
    get_optimal_batch_size,
)

__all__ = [
    "get_device",
    "get_optimal_num_workers",
    "setup_device_environment",
    "print_device_info",
    "get_device_info",
    "check_gpu_memory",
    "auto_select_device",
    "is_windows",
    "is_linux",
    "is_macos",
    "get_platform_specific_settings",
    "print_platform_info",
    "get_platform_info",
    "get_optimal_batch_size",
]
