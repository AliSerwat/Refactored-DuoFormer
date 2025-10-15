"""
Unit tests for utility functions.

Fast tests for device detection, platform utils, etc.
No GPU or heavy models required.
"""

import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils import (
    get_optimal_num_workers,
    is_windows,
    is_linux,
    is_macos,
    get_platform_specific_settings,
)


def test_platform_detection():
    """Test platform detection functions."""
    # At least one should be True
    platforms = [is_windows(), is_linux(), is_macos()]
    assert any(platforms), "Should detect at least one platform"
    assert sum(platforms) == 1, "Should detect exactly one platform"
    print("✓ Platform detection: PASS")


def test_optimal_num_workers():
    """Test num_workers detection."""
    workers = get_optimal_num_workers(max_workers=8)
    assert isinstance(workers, int)
    assert workers > 0
    assert workers <= 8
    print(f"✓ Optimal num_workers: {workers} PASS")


def test_platform_settings():
    """Test platform-specific settings."""
    settings = get_platform_specific_settings()

    assert "num_workers" in settings
    assert "pin_memory" in settings
    assert isinstance(settings["num_workers"], int)
    assert isinstance(settings["pin_memory"], bool)
    print("✓ Platform settings: PASS")


def test_device_selection_cpu():
    """Test device selection (CPU only, no GPU required)."""
    try:
        import torch
        from utils import get_device

        # Force CPU for testing
        device = get_device("cpu", verbose=False)
        assert device.type == "cpu"
        print("✓ Device selection (CPU): PASS")
    except ImportError:
        print("⚠ Skipping device test (torch not installed)")


def run_all_tests():
    """Run all utility tests."""
    print("\n" + "=" * 60)
    print("Running Utility Unit Tests (No GPU Required)")
    print("=" * 60)

    try:
        test_platform_detection()
        test_optimal_num_workers()
        test_platform_settings()
        test_device_selection_cpu()

        print("\n" + "=" * 60)
        print("✅ All utility tests passed!")
        print("=" * 60)
        return True
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
