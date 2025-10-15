TESTS DIRECTORY
===============

Organized test suite with separation of concerns for efficiency.

Structure:
----------
tests/
├── unit/           Fast tests, no GPU required (~seconds)
│   ├── test_config.py         Configuration tests
│   ├── test_utils.py          Utility function tests
│   └── test_lightweight.py    Minimal model tests
│
├── integration/    Full workflow tests, may need GPU (~minutes)
│   ├── test_full_models.py       Complete model tests
│   └── test_training_pipeline.py Training workflow tests
│
├── fixtures/       Mock data and test utilities
│   └── mock_data.py           Lightweight test data generators
│
└── run_tests.py    Test runner

Usage:
------
# Run all tests (recommended)
python tests/run_tests.py

# Fast unit tests only (no GPU, <30 seconds)
python tests/run_tests.py --unit

# Integration tests only (may need GPU, slower)
python tests/run_tests.py --integration

# Specific test file
python tests/unit/test_config.py
python tests/unit/test_utils.py
python tests/integration/test_full_models.py

# With pytest (if installed)
pytest tests/unit/ -v           # Fast tests
pytest tests/integration/ -v    # Slow tests
pytest tests/ -v                # All tests

Test Types:
-----------
UNIT TESTS (Fast, <30s total):
  • Configuration validation
  • Platform detection
  • Device utilities
  • Mock model operations
  • No GPU required
  • No model weights downloaded

INTEGRATION TESTS (Slow, minutes):
  • Full model initialization
  • Complete forward passes
  • Training pipeline
  • May use GPU if available
  • Downloads pretrained weights

Efficiency:
-----------
✓ Unit tests run in seconds
✓ No wasted GPU resources for simple checks
✓ Mock data for fast validation
✓ Integration tests separated for when needed
✓ Clear test organization

