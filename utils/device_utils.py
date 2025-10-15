"""
🖥️ Device Utilities

Hardware-agnostic device management and configuration.

Features:
- Automatic device detection
- Multi-GPU support
- CPU fallback
- Memory checking
- Platform-agnostic
"""

import torch
import os
from typing import Optional, Union, List


def get_device(
    device_str: Optional[str] = None, gpu_id: Optional[int] = None, verbose: bool = True
) -> torch.device:
    """
    Get PyTorch device in a hardware-agnostic way.

    Args:
        device_str: Device string ('cuda', 'cpu', 'cuda:0', etc.)
                   If None, automatically selects best available
        gpu_id: Specific GPU ID to use (0, 1, 2, etc.)
        verbose: Print device information

    Returns:
        torch.device instance

    Examples:
        >>> device = get_device()  # Auto-detect best device
        >>> device = get_device('cuda')  # Use CUDA if available, else CPU
        >>> device = get_device(gpu_id=1)  # Use GPU 1
    """
    # Auto-detect if not specified
    if device_str is None:
        if torch.cuda.is_available():
            if gpu_id is not None:
                device_str = f"cuda:{gpu_id}"
            else:
                device_str = "cuda"
        else:
            device_str = "cpu"

    # Handle CUDA request when not available
    if "cuda" in device_str and not torch.cuda.is_available():
        if verbose:
            print(f"⚠️  CUDA requested but not available. Falling back to CPU.")
        device_str = "cpu"

    # Create device
    device = torch.device(device_str)

    if verbose:
        print(f"\n🖥️  Device Configuration:")
        print(f"   Device: {device}")

        if device.type == "cuda":
            gpu_idx = device.index if device.index is not None else 0
            print(f"   GPU Name: {torch.cuda.get_device_name(gpu_idx)}")
            print(
                f"   GPU Memory: {torch.cuda.get_device_properties(gpu_idx).total_memory / 1e9:.2f} GB"
            )
            print(f"   CUDA Version: {torch.version.cuda}")
            print(f"   cuDNN Version: {torch.backends.cudnn.version()}")
            print(f"   Number of GPUs: {torch.cuda.device_count()}")
        else:
            print(f"   Running on CPU")
            print(f"   Note: Training will be slower without GPU")

    return device


def get_optimal_num_workers(max_workers: int = 8) -> int:
    """
    Get optimal number of workers for DataLoader based on available CPUs.

    Args:
        max_workers: Maximum number of workers to use

    Returns:
        Optimal number of workers

    Notes:
        - On Windows, num_workers > 0 can cause issues
        - On Unix, we can use more workers
        - Generally: min(cpu_count, max_workers)
    """
    import multiprocessing

    cpu_count = multiprocessing.cpu_count()

    # Platform-specific adjustments
    if os.name == "nt":  # Windows
        # Windows can have issues with multi-process loading
        optimal = min(4, cpu_count, max_workers)
    else:  # Unix-like (Linux, macOS)
        optimal = min(cpu_count, max_workers)

    return optimal


def set_cuda_visible_devices(gpu_ids: Union[int, List[int], str, None] = None):
    """
    Set CUDA_VISIBLE_DEVICES environment variable in a flexible way.

    Args:
        gpu_ids: GPU ID(s) to use. Can be:
                - int: Single GPU (e.g., 0)
                - list: Multiple GPUs (e.g., [0, 1, 2])
                - str: Comma-separated (e.g., '0,1,2')
                - None: Use all available GPUs

    Examples:
        >>> set_cuda_visible_devices(0)  # Use GPU 0
        >>> set_cuda_visible_devices([0, 1])  # Use GPUs 0 and 1
        >>> set_cuda_visible_devices('0,1,2')  # Use GPUs 0, 1, 2
        >>> set_cuda_visible_devices(None)  # Use all GPUs

    Note:
        Call this BEFORE importing torch for it to take effect properly.
    """
    if gpu_ids is None:
        # Use all available GPUs
        return

    if isinstance(gpu_ids, int):
        gpu_str = str(gpu_ids)
    elif isinstance(gpu_ids, list):
        gpu_str = ",".join(map(str, gpu_ids))
    elif isinstance(gpu_ids, str):
        gpu_str = gpu_ids
    else:
        raise ValueError(f"Invalid gpu_ids type: {type(gpu_ids)}")

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_str
    print(f"🎯 CUDA_VISIBLE_DEVICES set to: {gpu_str}")


def check_gpu_memory(device: torch.device, required_gb: float = 8.0) -> bool:
    """
    Check if GPU has sufficient memory.

    Args:
        device: PyTorch device
        required_gb: Required memory in GB

    Returns:
        True if sufficient memory available
    """
    if device.type != "cuda":
        return True  # CPU doesn't have this constraint

    gpu_idx = device.index if device.index is not None else 0
    total_memory = torch.cuda.get_device_properties(gpu_idx).total_memory / 1e9

    if total_memory < required_gb:
        print(
            f"⚠️  Warning: GPU has {total_memory:.2f} GB, but {required_gb:.2f} GB recommended"
        )
        print(f"   Consider reducing batch size or using CPU")
        return False

    return True


def get_device_info() -> dict:
    """
    Get comprehensive device information.

    Returns:
        Dictionary with device information
    """
    info = {
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "cudnn_available": torch.backends.cudnn.is_available(),
        "cudnn_version": (
            torch.backends.cudnn.version()
            if torch.backends.cudnn.is_available()
            else None
        ),
        "num_gpus": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "cpu_count": os.cpu_count(),
        "platform": os.name,
    }

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            info[f"gpu_{i}_name"] = torch.cuda.get_device_name(i)
            info[f"gpu_{i}_memory_gb"] = (
                torch.cuda.get_device_properties(i).total_memory / 1e9
            )

    return info


def print_device_info():
    """Print comprehensive device information."""
    info = get_device_info()

    print("\n" + "=" * 70)
    print("🖥️  Hardware Information")
    print("=" * 70)

    print(f"\nPlatform: {info['platform']}")
    print(f"CPU Cores: {info['cpu_count']}")

    if info["cuda_available"]:
        print(f"\nCUDA Available: Yes")
        print(f"CUDA Version: {info['cuda_version']}")
        print(f"cuDNN Version: {info['cudnn_version']}")
        print(f"Number of GPUs: {info['num_gpus']}")

        for i in range(info["num_gpus"]):
            print(f"\n  GPU {i}:")
            print(f"    Name: {info[f'gpu_{i}_name']}")
            print(f"    Memory: {info[f'gpu_{i}_memory_gb']:.2f} GB")
    else:
        print(f"\nCUDA Available: No")
        print(f"Running on CPU only")

    print("=" * 70 + "\n")


def setup_device_environment(
    preferred_device: str = "auto",
    gpu_ids: Optional[Union[int, List[int]]] = None,
    verbose: bool = True,
) -> torch.device:
    """
    Setup device environment in a robust, platform-agnostic way.

    Args:
        preferred_device: 'auto', 'cuda', 'cpu', or specific like 'cuda:0'
        gpu_ids: Optional GPU IDs to restrict to
        verbose: Print information

    Returns:
        Configured device

    Example:
        >>> device = setup_device_environment('auto')
        >>> device = setup_device_environment('cuda', gpu_ids=[0, 1])
    """
    # Set CUDA visible devices if specified
    if gpu_ids is not None:
        set_cuda_visible_devices(gpu_ids)

    # Get device
    if preferred_device == "auto":
        device = get_device(verbose=verbose)
    else:
        device = get_device(preferred_device, verbose=verbose)

    return device


# Auto-detection helper
def auto_select_device() -> torch.device:
    """
    Automatically select the best available device.

    Priority:
    1. CUDA GPU if available
    2. MPS (Apple Silicon) if available
    3. CPU as fallback
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        # Apple Silicon support (PyTorch 1.12+)
        return torch.device("mps")
    else:
        return torch.device("cpu")
