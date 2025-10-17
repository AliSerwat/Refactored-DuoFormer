#!/usr/bin/env python3
"""
🔧 Centralized Logging Configuration for Refactored DuoFormer

Provides structured logging with proper formatting, levels, and handlers.
Replaces print statements with professional logging throughout the codebase.

Usage:
    from utils.logging_config import get_logger

    logger = get_logger(__name__)
    logger.info("Training started")
    logger.error("Model failed to load", exc_info=True)
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for different log levels."""

    # Color codes
    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
        "RESET": "\033[0m",  # Reset
    }

    def format(self, record):
        # Add color to level name
        if hasattr(record, "levelname"):
            color = self.COLORS.get(record.levelname, self.COLORS["RESET"])
            record.levelname = f"{color}{record.levelname}{self.COLORS['RESET']}"

        return super().format(record)


def setup_logging(
    level: str = "INFO",
    log_file: Optional[Path] = None,
    log_dir: Path = Path("logs"),
    enable_console: bool = True,
    enable_file: bool = True,
    max_file_size: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
) -> None:
    """
    Setup centralized logging configuration.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Specific log file path (auto-generated if None)
        log_dir: Directory for log files
        enable_console: Enable console logging
        enable_file: Enable file logging
        max_file_size: Maximum log file size before rotation
        backup_count: Number of backup files to keep
    """
    # Create log directory
    if enable_file:
        log_dir.mkdir(parents=True, exist_ok=True)

        # Generate log file name with timestamp
        if log_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = log_dir / f"duoformer_{timestamp}.log"

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))

    # Clear existing handlers
    root_logger.handlers.clear()

    # Console handler
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, level.upper()))

        # Use colored formatter for console
        console_formatter = ColoredFormatter(
            "%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)

    # File handler with rotation
    if enable_file and log_file:
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=max_file_size, backupCount=backup_count, encoding="utf-8"
        )
        file_handler.setLevel(logging.DEBUG)  # File gets all levels

        # Detailed formatter for file
        file_formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)-20s | %(funcName)-15s | %(lineno)-4d | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)

    # Log the setup
    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured - Level: {level}, File: {log_file}")


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the specified name.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)


def log_function_call(func):
    """
    Decorator to log function calls with parameters and return values.

    Args:
        func: Function to decorate

    Returns:
        Decorated function with logging
    """
    import functools

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)

        # Log function entry
        logger.debug(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")

        try:
            result = func(*args, **kwargs)
            logger.debug(f"{func.__name__} completed successfully")
            return result
        except Exception as e:
            logger.error(f"{func.__name__} failed with error: {e}", exc_info=True)
            raise

    return wrapper


def log_training_progress(
    epoch: int,
    total_epochs: int,
    train_loss: float,
    val_loss: Optional[float] = None,
    val_acc: Optional[float] = None,
) -> None:
    """
    Log training progress in a structured format.

    Args:
        epoch: Current epoch number
        total_epochs: Total number of epochs
        train_loss: Training loss
        val_loss: Validation loss (optional)
        val_acc: Validation accuracy (optional)
    """
    logger = get_logger("training")

    progress = f"Epoch {epoch}/{total_epochs}"
    train_info = f"Train Loss: {train_loss:.4f}"

    if val_loss is not None:
        train_info += f" | Val Loss: {val_loss:.4f}"
    if val_acc is not None:
        train_info += f" | Val Acc: {val_acc:.4f}"

    logger.info(f"{progress} | {train_info}")


def log_model_info(model_name: str, num_parameters: int, model_size_mb: float) -> None:
    """
    Log model information in a structured format.

    Args:
        model_name: Name of the model
        num_parameters: Number of model parameters
        model_size_mb: Model size in MB
    """
    logger = get_logger("model")

    logger.info(f"Model: {model_name}")
    logger.info(f"Parameters: {num_parameters:,}")
    logger.info(f"Size: {model_size_mb:.2f} MB")


def log_system_info() -> None:
    """Log system information for debugging."""
    import platform
    import torch

    logger = get_logger("system")

    logger.info("System Information:")
    logger.info(f"  Platform: {platform.platform()}")
    logger.info(f"  Python: {platform.python_version()}")
    logger.info(f"  PyTorch: {torch.__version__}")

    if torch.cuda.is_available():
        logger.info(f"  CUDA: {torch.version.cuda}")
        logger.info(f"  GPU Count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            logger.info(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        logger.info("  CUDA: Not available")


# Initialize logging on import
setup_logging()
