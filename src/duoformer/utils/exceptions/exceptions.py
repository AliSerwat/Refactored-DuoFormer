#!/usr/bin/env python3
"""
🔧 Custom Exception Classes for Refactored DuoFormer

Provides structured exception handling with proper error messages,
logging, and recovery mechanisms for the medical AI framework.
"""

import logging
from typing import Optional, Dict, Any
from pathlib import Path
from functools import wraps


class DuoFormerException(Exception):
    """Base exception class for all DuoFormer-related errors."""

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize DuoFormer exception.

        Args:
            message: Human-readable error message
            error_code: Machine-readable error code
            details: Additional error details
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}

        # Log the exception
        logger = logging.getLogger(self.__class__.__module__)
        logger.error(f"DuoFormer Error [{error_code}]: {message}")
        if details:
            logger.debug(f"Error details: {details}")


class ConfigurationError(DuoFormerException):
    """Raised when configuration is invalid or missing."""

    def __init__(
        self,
        message: str,
        config_file: Optional[Path] = None,
        config_key: Optional[str] = None,
    ):
        """
        Initialize configuration error.

        Args:
            message: Error message
            config_file: Path to configuration file
            config_key: Configuration key that caused the error
        """
        details = {}
        if config_file:
            details["config_file"] = str(config_file)
        if config_key:
            details["config_key"] = config_key

        super().__init__(message, "CONFIG_ERROR", details)


class ModelError(DuoFormerException):
    """Raised when model-related operations fail."""

    def __init__(
        self,
        message: str,
        model_name: Optional[str] = None,
        operation: Optional[str] = None,
    ):
        """
        Initialize model error.

        Args:
            message: Error message
            model_name: Name of the model
            operation: Operation that failed
        """
        details = {}
        if model_name:
            details["model_name"] = model_name
        if operation:
            details["operation"] = operation

        super().__init__(message, "MODEL_ERROR", details)


class DataError(DuoFormerException):
    """Raised when data-related operations fail."""

    def __init__(
        self,
        message: str,
        data_path: Optional[Path] = None,
        data_format: Optional[str] = None,
    ):
        """
        Initialize data error.

        Args:
            message: Error message
            data_path: Path to data file/directory
            data_format: Expected data format
        """
        details = {}
        if data_path:
            details["data_path"] = str(data_path)
        if data_format:
            details["data_format"] = data_format

        super().__init__(message, "DATA_ERROR", details)


class TrainingError(DuoFormerException):
    """Raised when training-related operations fail."""

    def __init__(
        self,
        message: str,
        epoch: Optional[int] = None,
        checkpoint_path: Optional[Path] = None,
    ):
        """
        Initialize training error.

        Args:
            message: Error message
            epoch: Training epoch when error occurred
            checkpoint_path: Path to checkpoint file
        """
        details: Dict[str, Any] = {}
        if epoch is not None:
            details["epoch"] = epoch
        if checkpoint_path:
            details["checkpoint_path"] = str(checkpoint_path)

        super().__init__(message, "TRAINING_ERROR", details)


class DeviceError(DuoFormerException):
    """Raised when device-related operations fail."""

    def __init__(
        self,
        message: str,
        device: Optional[str] = None,
        operation: Optional[str] = None,
    ):
        """
        Initialize device error.

        Args:
            message: Error message
            device: Device that caused the error
            operation: Operation that failed
        """
        details = {}
        if device:
            details["device"] = device
        if operation:
            details["operation"] = operation

        super().__init__(message, "DEVICE_ERROR", details)


class ValidationError(DuoFormerException):
    """Raised when input validation fails."""

    def __init__(
        self,
        message: str,
        parameter: Optional[str] = None,
        value: Optional[Any] = None,
        expected: Optional[Any] = None,
    ):
        """
        Initialize validation error.

        Args:
            message: Error message
            parameter: Parameter name that failed validation
            value: Actual value that failed validation
            expected: Expected value or type
        """
        details = {}
        if parameter:
            details["parameter"] = parameter
        if value is not None:
            details["value"] = str(value)
        if expected is not None:
            details["expected"] = str(expected)

        super().__init__(message, "VALIDATION_ERROR", details)


class DependencyError(DuoFormerException):
    """Raised when required dependencies are missing."""

    def __init__(
        self, message: str, package: Optional[str] = None, version: Optional[str] = None
    ):
        """
        Initialize dependency error.

        Args:
            message: Error message
            package: Missing package name
            version: Required version
        """
        details = {}
        if package:
            details["package"] = package
        if version:
            details["version"] = version

        super().__init__(message, "DEPENDENCY_ERROR", details)


def handle_exception(exc: Exception, context: str = "") -> None:
    """
    Handle exceptions with proper logging and user-friendly messages.

    Args:
        exc: Exception to handle
        context: Additional context about where the exception occurred
    """
    logger = logging.getLogger(__name__)

    if isinstance(exc, DuoFormerException):
        # Already logged in the exception constructor
        logger.info(f"Handling DuoFormer exception in context: {context}")
    else:
        # Log unexpected exceptions
        logger.error(f"Unexpected error in context '{context}': {exc}", exc_info=True)


def safe_execute(func, *args, default_return=None, error_message: str = "", **kwargs):
    """
    Safely execute a function with proper exception handling.

    Args:
        func: Function to execute
        *args: Function arguments
        default_return: Default return value if function fails
        error_message: Custom error message
        **kwargs: Function keyword arguments

    Returns:
        Function result or default_return if execution fails
    """
    logger = logging.getLogger(__name__)

    try:
        return func(*args, **kwargs)
    except DuoFormerException:
        # Re-raise DuoFormer exceptions
        raise
    except Exception as exc:
        message = error_message or f"Error executing {func.__name__}"
        logger.error(f"{message}: {exc}", exc_info=True)

        if default_return is not None:
            return default_return
        else:
            # Convert to DuoFormer exception
            raise DuoFormerException(message) from exc


def validate_file_exists(file_path: Path, description: str = "file") -> None:
    """
    Validate that a file exists, raising appropriate exception if not.

    Args:
        file_path: Path to file to validate
        description: Description of the file for error message

    Raises:
        DataError: If file does not exist
    """
    if not file_path.exists():
        raise DataError(
            f"{description.capitalize()} not found: {file_path}", data_path=file_path
        )

    if not file_path.is_file():
        raise DataError(
            f"{description.capitalize()} is not a file: {file_path}",
            data_path=file_path,
        )


def validate_directory_exists(dir_path: Path, description: str = "directory") -> None:
    """
    Validate that a directory exists, raising appropriate exception if not.

    Args:
        dir_path: Path to directory to validate
        description: Description of the directory for error message

    Raises:
        DataError: If directory does not exist
    """
    if not dir_path.exists():
        raise DataError(
            f"{description.capitalize()} not found: {dir_path}", data_path=dir_path
        )

    if not dir_path.is_dir():
        raise DataError(
            f"{description.capitalize()} is not a directory: {dir_path}",
            data_path=dir_path,
        )


def validate_positive_number(
    value: Any, parameter_name: str, allow_zero: bool = False
) -> None:
    """
    Validate that a number is positive (or non-negative if allow_zero=True).

    Args:
        value: Value to validate
        parameter_name: Name of the parameter for error message
        allow_zero: Whether zero is allowed

    Raises:
        ValidationError: If value is not a valid positive number
    """
    try:
        num_value = float(value)
    except (ValueError, TypeError):
        raise ValidationError(
            f"{parameter_name} must be a number, got {type(value).__name__}",
            parameter=parameter_name,
            value=value,
            expected="number",
        )

    if allow_zero and num_value < 0:
        raise ValidationError(
            f"{parameter_name} must be non-negative, got {num_value}",
            parameter=parameter_name,
            value=value,
            expected=">= 0",
        )
    elif not allow_zero and num_value <= 0:
        raise ValidationError(
            f"{parameter_name} must be positive, got {num_value}",
            parameter=parameter_name,
            value=value,
            expected="> 0",
        )


def validate_range(
    value: Any, parameter_name: str, min_val: float, max_val: float
) -> None:
    """
    Validate that a value is within a specified range.

    Args:
        value: Value to validate
        parameter_name: Name of the parameter for error message
        min_val: Minimum allowed value (inclusive)
        max_val: Maximum allowed value (inclusive)

    Raises:
        ValidationError: If value is not within the specified range
    """
    try:
        num_value = float(value)
    except (ValueError, TypeError):
        raise ValidationError(
            f"{parameter_name} must be a number, got {type(value).__name__}",
            parameter=parameter_name,
            value=value,
            expected="number",
        )

    if not (min_val <= num_value <= max_val):
        raise ValidationError(
            f"{parameter_name} must be between {min_val} and {max_val}, got {num_value}",
            parameter=parameter_name,
            value=value,
            expected=f"[{min_val}, {max_val}]",
        )
