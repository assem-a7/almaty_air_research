"""
Utility functions for Almaty Air Quality Forecasting & Analysis project.

This module provides helper functions for logging, file operations,
and common data handling tasks.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

import pandas as pd


def setup_logging(
    log_path: Optional[Path] = None,
    log_level: str = "INFO",
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    date_format: str = "%Y-%m-%d %H:%M:%S",
) -> logging.Logger:
    """
    Set up logging configuration with both file and console handlers.
    
    Creates a logger that writes to both a file (if path provided) and console.
    The file handler includes all messages, while console shows INFO and above.
    
    Args:
        log_path: Path to log file. If None, only console logging is enabled.
        log_level: Minimum logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_format: Format string for log messages.
        date_format: Format string for timestamps.
    
    Returns:
        Configured logger instance.
    
    Examples:
        >>> logger = setup_logging(Path("logs/app.log"))
        >>> logger.info("Application started")
    """
    # Create logger
    logger = logging.getLogger("almaty_air_quality")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(log_format, datefmt=date_format)
    
    # Console handler (INFO and above)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (all levels) if path provided
    if log_path is not None:
        # Ensure parent directory exists
        safe_mkdir(log_path.parent)
        
        file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        logger.info(f"Logging initialized. Log file: {log_path}")
    else:
        logger.info("Logging initialized (console only)")
    
    return logger


def safe_mkdir(path: Path) -> Path:
    """
    Create directory if it doesn't exist, including parent directories.
    
    This function is idempotent - it's safe to call multiple times.
    Uses exist_ok=True to avoid errors if directory already exists.
    
    Args:
        path: Directory path to create.
    
    Returns:
        The created/existing directory path.
    
    Raises:
        OSError: If directory creation fails due to permissions or other issues.
    
    Examples:
        >>> data_dir = safe_mkdir(Path("data/raw"))
        >>> print(data_dir.exists())
        True
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_df(
    df: pd.DataFrame,
    path: Path,
    index: bool = False,
    logger: Optional[logging.Logger] = None,
    **kwargs,
) -> None:
    """
    Save DataFrame to CSV file with automatic directory creation and logging.
    
    Ensures parent directory exists before saving. Logs DataFrame shape
    and destination path for tracking.
    
    Args:
        df: DataFrame to save.
        path: Destination file path (should end in .csv).
        index: Whether to include DataFrame index in output.
        logger: Logger instance for output. If None, prints to console.
        **kwargs: Additional arguments passed to pandas.to_csv().
    
    Raises:
        ValueError: If DataFrame is empty.
        IOError: If file write fails.
    
    Examples:
        >>> df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        >>> save_df(df, Path("data/output.csv"))
        Saved DataFrame (3, 2) to data/output.csv
    """
    if df.empty:
        raise ValueError("Cannot save empty DataFrame")
    
    path = Path(path)
    
    # Ensure parent directory exists
    safe_mkdir(path.parent)
    
    # Save DataFrame
    df.to_csv(path, index=index, **kwargs)
    
    # Log success
    message = f"Saved DataFrame {df.shape} to {path}"
    if logger:
        logger.info(message)
    else:
        print(message)


def load_df(
    path: Path,
    logger: Optional[logging.Logger] = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Load DataFrame from CSV file with logging.
    
    Args:
        path: Source file path.
        logger: Logger instance for output. If None, prints to console.
        **kwargs: Additional arguments passed to pandas.read_csv().
    
    Returns:
        Loaded DataFrame.
    
    Raises:
        FileNotFoundError: If file doesn't exist.
        pd.errors.EmptyDataError: If file is empty.
    
    Examples:
        >>> df = load_df(Path("data/input.csv"))
        Loaded DataFrame (100, 5) from data/input.csv
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    
    # Load DataFrame
    df = pd.read_csv(path, **kwargs)
    
    # Log success
    message = f"Loaded DataFrame {df.shape} from {path}"
    if logger:
        logger.info(message)
    else:
        print(message)
    
    return df


def validate_date_range(start_date: str, end_date: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """
    Validate and convert date strings to pandas Timestamps.
    
    Args:
        start_date: Start date string (YYYY-MM-DD format).
        end_date: End date string (YYYY-MM-DD format).
    
    Returns:
        Tuple of (start_timestamp, end_timestamp).
    
    Raises:
        ValueError: If dates are invalid or end_date < start_date.
    
    Examples:
        >>> start, end = validate_date_range("2024-01-01", "2024-12-31")
        >>> print(start, end)
        2024-01-01 00:00:00 2024-12-31 00:00:00
    """
    try:
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
    except Exception as e:
        raise ValueError(f"Invalid date format: {e}")
    
    if end < start:
        raise ValueError(f"End date {end_date} is before start date {start_date}")
    
    return start, end


def get_data_info(df: pd.DataFrame, logger: Optional[logging.Logger] = None) -> dict:
    """
    Get comprehensive information about a DataFrame.
    
    Args:
        df: DataFrame to analyze.
        logger: Logger instance for output.
    
    Returns:
        Dictionary with shape, columns, dtypes, missing values, and memory usage.
    """
    info = {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": df.dtypes.to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "missing_percentage": (df.isnull().sum() / len(df) * 100).to_dict(),
        "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024**2,
    }
    
    if logger:
        logger.info(f"DataFrame shape: {info['shape']}")
        logger.info(f"Memory usage: {info['memory_usage_mb']:.2f} MB")
        
        missing_cols = {k: v for k, v in info['missing_percentage'].items() if v > 0}
        if missing_cols:
            logger.warning(f"Missing values: {missing_cols}")
    
    return info