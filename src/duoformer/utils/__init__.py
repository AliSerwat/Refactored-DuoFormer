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
)

from .platform import (
    is_windows,
    is_linux,
    is_macos,
    get_platform_specific_settings,
)

__all__ = [
    "get_device",
    "get_optimal_num_workers",
    "setup_device_environment",
    "is_windows",
    "is_linux",
    "is_macos",
    "get_platform_specific_settings",
]
