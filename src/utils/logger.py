"""
Logging utilities for CampusGPT
Centralized logging configuration
"""

import logging
import sys
from typing import Optional
from pathlib import Path


def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    """
    Get configured logger instance
    
    Args:
        name: Logger name (usually __name__)
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Avoid adding multiple handlers
    if logger.handlers:
        return logger
    
    # Set logging level
    log_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(log_level)
    
    # Create formatter
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger


def setup_file_logging(
    logger: logging.Logger,
    log_file: str,
    level: str = "INFO",
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
) -> None:
    """
    Add file logging to existing logger
    
    Args:
        logger: Logger instance
        log_file: Path to log file
        level: Logging level for file handler
        max_bytes: Maximum log file size before rotation
        backup_count: Number of backup files to keep
    """
    from logging.handlers import RotatingFileHandler
    
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # File handler with rotation
    file_handler = RotatingFileHandler(
        log_path,
        maxBytes=max_bytes,
        backupCount=backup_count
    )
    
    log_level = getattr(logging, level.upper(), logging.INFO)
    file_handler.setLevel(log_level)
    
    # File formatter (more detailed)
    file_formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    
    logger.addHandler(file_handler)


def configure_training_logging(log_dir: str = "./logs") -> logging.Logger:
    """
    Configure logging for training processes
    
    Args:
        log_dir: Directory for log files
        
    Returns:
        Configured logger for training
    """
    logger = get_logger("campusgpt.training", level="INFO")
    
    # Add file logging
    log_file = Path(log_dir) / "training.log"
    setup_file_logging(logger, str(log_file), level="DEBUG")
    
    return logger


def configure_api_logging(log_dir: str = "./logs") -> logging.Logger:
    """
    Configure logging for API server
    
    Args:
        log_dir: Directory for log files
        
    Returns:
        Configured logger for API
    """
    logger = get_logger("campusgpt.api", level="INFO")
    
    # Add file logging
    log_file = Path(log_dir) / "api.log"
    setup_file_logging(logger, str(log_file), level="INFO")
    
    return logger


def disable_transformers_warnings():
    """Disable noisy transformers warnings"""
    transformers_logger = logging.getLogger("transformers")
    transformers_logger.setLevel(logging.WARNING)
    
    # Also disable tokenizers warnings
    tokenizers_logger = logging.getLogger("tokenizers")
    tokenizers_logger.setLevel(logging.WARNING)