"""
Platform utilities for DuoFormer.

Platform and system detection utilities for cross-platform compatibility.
"""

from .platform_utils import (
    get_platform_info,
    is_windows,
    is_linux,
    is_macos,
    get_optimal_batch_size,
    get_platform_specific_settings,
    get_optimal_num_workers,
    print_platform_info,
)

__all__ = [
    "get_platform_info",
    "is_windows",
    "is_linux",
    "is_macos",
    "get_optimal_batch_size",
    "get_platform_specific_settings",
    "get_optimal_num_workers",
    "print_platform_info",
]
