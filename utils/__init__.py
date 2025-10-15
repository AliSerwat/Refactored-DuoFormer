"""
🛠️ Refactored DuoFormer Utilities

Training utilities, dataset handling, and helper functions.
Refactored for general medical imaging applications.
"""

from .trainer import Trainer, create_optimizer, create_scheduler
from .dataset import (
    MedicalImageDataset,
    MedicalImageAugmentation,
    create_dataloaders,
    build_dataset,
)
from .device_utils import (
    get_device,
    get_optimal_num_workers,
    set_cuda_visible_devices,
    check_gpu_memory,
    get_device_info,
    print_device_info,
    setup_device_environment,
    auto_select_device,
)
from .platform_utils import (
    get_platform_info,
    is_windows,
    is_linux,
    is_macos,
    get_platform_specific_settings,
    print_platform_info,
)

__all__ = [
    "Trainer",
    "create_optimizer",
    "create_scheduler",
    "MedicalImageDataset",
    "MedicalImageAugmentation",
    "create_dataloaders",
    "build_dataset",
    "get_device",
    "get_optimal_num_workers",
    "setup_device_environment",
    "auto_select_device",
    "print_device_info",
    "get_platform_info",
    "is_windows",
    "is_linux",
    "is_macos",
    "get_platform_specific_settings",
    "print_platform_info",
]
