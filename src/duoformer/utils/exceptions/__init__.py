"""
Custom Exception Classes for DuoFormer

This module provides structured exception handling with proper error messages,
logging, and recovery mechanisms for the medical AI framework.
"""

from .exceptions import (
    DuoFormerException,
    ConfigurationError,
    ModelError,
    DataError,
    TrainingError,
    DeviceError,
    ValidationError,
    DependencyError,
    handle_exception,
    safe_execute,
    validate_file_exists,
    validate_directory_exists,
    validate_positive_number,
    validate_range,
)

__all__ = [
    "DuoFormerException",
    "ConfigurationError",
    "ModelError",
    "DataError",
    "TrainingError",
    "DeviceError",
    "ValidationError",
    "DependencyError",
    "handle_exception",
    "safe_execute",
    "validate_file_exists",
    "validate_directory_exists",
    "validate_positive_number",
    "validate_range",
]
