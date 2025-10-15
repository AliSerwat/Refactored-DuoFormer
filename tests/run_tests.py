#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test runner for Refactored DuoFormer.

Runs tests in order from fastest to slowest, with clear separation.

Usage:
    python tests/run_tests.py              # Run all tests
    python tests/run_tests.py --unit       # Only fast unit tests
    python tests/run_tests.py --integration # Only integration tests
"""

import sys
import argparse
from pathlib import Path
import time

# Fix Windows encoding
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def run_unit_tests(verbose=False):
    """Run fast unit tests (no GPU needed)."""
    print("\n" + "=" * 70)
    print("🚀 Running Unit Tests (Fast, No GPU Required)")
    print("=" * 70)

    unit_tests = [
        "tests/unit/test_config.py",
        "tests/unit/test_utils.py",
        "tests/unit/test_lightweight.py",
    ]

    passed = 0
    failed = 0

    for test_file in unit_tests:
        test_path = Path(test_file)
        if test_path.exists():
            print(f"\n▶ Running: {test_file}")
            import subprocess

            result = subprocess.run(
                [sys.executable, str(test_path)], capture_output=not verbose
            )
            if result.returncode == 0:
                passed += 1
                print(f"  ✅ PASS")
            else:
                failed += 1
                print(f"  ❌ FAIL")
        else:
            print(f"\n⚠ Skipping {test_file} (not found)")

    return passed, failed


def run_integration_tests(verbose=False):
    """Run integration tests (may require GPU, takes longer)."""
    print("\n" + "=" * 70)
    print("🔬 Running Integration Tests (Slower, Resource-Intensive)")
    print("=" * 70)
    print("⚠ These tests may take several minutes")

    integration_tests = [
        "tests/integration/test_full_models.py",
        "tests/integration/test_training_pipeline.py",
    ]

    passed = 0
    failed = 0

    for test_file in integration_tests:
        test_path = Path(test_file)
        if test_path.exists():
            print(f"\n▶ Running: {test_file}")
            import subprocess

            result = subprocess.run(
                [sys.executable, str(test_path)], capture_output=not verbose
            )
            if result.returncode == 0:
                passed += 1
                print(f"  ✅ PASS")
            else:
                failed += 1
                print(f"  ❌ FAIL")
        else:
            print(f"\n⚠ Skipping {test_file} (not found)")

    return passed, failed


def main():
    """Main test runner."""
    parser = argparse.ArgumentParser(description="Run DuoFormer tests")
    parser.add_argument("--unit", action="store_true", help="Run only unit tests")
    parser.add_argument(
        "--integration", action="store_true", help="Run only integration tests"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    start_time = time.time()

    unit_passed, unit_failed = 0, 0
    int_passed, int_failed = 0, 0

    if args.unit or (not args.unit and not args.integration):
        unit_passed, unit_failed = run_unit_tests(args.verbose)

    if args.integration or (not args.unit and not args.integration):
        int_passed, int_failed = run_integration_tests(args.verbose)

    elapsed = time.time() - start_time

    # Summary
    total_passed = unit_passed + int_passed
    total_failed = unit_failed + int_failed
    total_tests = total_passed + total_failed

    print("\n" + "=" * 70)
    print("📊 Test Summary")
    print("=" * 70)
    print(f"\nUnit Tests:        {unit_passed} passed, {unit_failed} failed")
    print(f"Integration Tests: {int_passed} passed, {int_failed} failed")
    print(f"\nTotal:             {total_passed}/{total_tests} passed")
    print(f"Time:              {elapsed:.2f} seconds")

    if total_failed == 0:
        print("\n✅ All tests passed!")
        print("=" * 70 + "\n")
        return True
    else:
        print(f"\n❌ {total_failed} test(s) failed")
        print("=" * 70 + "\n")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
