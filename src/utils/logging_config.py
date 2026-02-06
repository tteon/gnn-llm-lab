"""
Structured logging configuration for GNN+LLM experiments.

Provides consistent logging across all modules with:
- Console output with color support
- Optional file logging with rotation
- Structured format with timestamps
- Different log levels for different environments
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


# ANSI color codes for terminal output
class LogColors:
    """ANSI color codes for log levels."""
    RESET = "\033[0m"
    DEBUG = "\033[36m"      # Cyan
    INFO = "\033[32m"       # Green
    WARNING = "\033[33m"    # Yellow
    ERROR = "\033[31m"      # Red
    CRITICAL = "\033[35m"   # Magenta
    BOLD = "\033[1m"


class ColoredFormatter(logging.Formatter):
    """Custom formatter with color support for console output."""

    LEVEL_COLORS = {
        logging.DEBUG: LogColors.DEBUG,
        logging.INFO: LogColors.INFO,
        logging.WARNING: LogColors.WARNING,
        logging.ERROR: LogColors.ERROR,
        logging.CRITICAL: LogColors.CRITICAL,
    }

    def __init__(self, fmt: str, datefmt: str = None, use_colors: bool = True):
        super().__init__(fmt, datefmt)
        self.use_colors = use_colors

    def format(self, record: logging.LogRecord) -> str:
        if self.use_colors and sys.stdout.isatty():
            color = self.LEVEL_COLORS.get(record.levelno, LogColors.RESET)
            record.levelname = f"{color}{record.levelname}{LogColors.RESET}"
            record.name = f"{LogColors.BOLD}{record.name}{LogColors.RESET}"
        return super().format(record)


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    log_dir: str = "results/logs",
    module_name: str = "gnnllm",
    use_colors: bool = True,
) -> logging.Logger:
    """
    Configure logging for the application.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file name (auto-generated if None)
        log_dir: Directory for log files
        module_name: Name for the root logger
        use_colors: Enable colored console output

    Returns:
        Configured logger instance
    """
    # Parse log level
    log_level = getattr(logging, level.upper(), logging.INFO)

    # Get or create logger
    logger = logging.getLogger(module_name)
    logger.setLevel(log_level)

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    # Console handler with colors
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_format = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    console_formatter = ColoredFormatter(console_format, datefmt="%H:%M:%S", use_colors=use_colors)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file is not None or log_dir:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)

        if log_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = f"{timestamp}_experiment.log"

        file_handler = logging.FileHandler(log_path / log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)  # Always capture all levels to file
        file_format = "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s"
        file_formatter = logging.Formatter(file_format, datefmt="%Y-%m-%d %H:%M:%S")
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    # Prevent propagation to root logger
    logger.propagate = False

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a child logger with the given name.

    Args:
        name: Logger name (will be prefixed with 'gnnllm.')

    Returns:
        Logger instance
    """
    return logging.getLogger(f"gnnllm.{name}")


# Module-level convenience functions
_default_logger: Optional[logging.Logger] = None


def log_info(msg: str, *args, **kwargs):
    """Log info message."""
    global _default_logger
    if _default_logger is None:
        _default_logger = setup_logging()
    _default_logger.info(msg, *args, **kwargs)


def log_warning(msg: str, *args, **kwargs):
    """Log warning message."""
    global _default_logger
    if _default_logger is None:
        _default_logger = setup_logging()
    _default_logger.warning(msg, *args, **kwargs)


def log_error(msg: str, *args, **kwargs):
    """Log error message."""
    global _default_logger
    if _default_logger is None:
        _default_logger = setup_logging()
    _default_logger.error(msg, *args, **kwargs)


def log_debug(msg: str, *args, **kwargs):
    """Log debug message."""
    global _default_logger
    if _default_logger is None:
        _default_logger = setup_logging()
    _default_logger.debug(msg, *args, **kwargs)
