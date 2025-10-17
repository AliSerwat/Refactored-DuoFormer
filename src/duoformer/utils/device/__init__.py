"""
Device utilities for DuoFormer.

Hardware-agnostic device management and configuration.
"""

from .device_utils import (
    get_device,
    get_optimal_num_workers,
    setup_device_environment,
    set_cuda_visible_devices,
    check_gpu_memory,
    get_device_info,
    print_device_info,
    auto_select_device,
)

__all__ = [
    "get_device",
    "get_optimal_num_workers",
    "setup_device_environment",
    "set_cuda_visible_devices",
    "check_gpu_memory",
    "get_device_info",
    "print_device_info",
    "auto_select_device",
]
