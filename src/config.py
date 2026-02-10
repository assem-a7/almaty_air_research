"""
Configuration module for Almaty Air Quality Forecasting & Analysis project.

This module centralizes all configuration parameters including locations,
date ranges, paths, and operational settings.
"""

from pathlib import Path
from typing import Final

# Project root directory
PROJECT_ROOT: Final[Path] = Path(__file__).parent.parent

# ============================================================================
# Location Configuration
# ============================================================================

CITY: Final[str] = "Almaty"
COUNTRY: Final[str] = "Kazakhstan"
TIMEZONE: Final[str] = "Asia/Almaty"

# Almaty coordinates
LATITUDE: Final[float] = 43.2220
LONGITUDE: Final[float] = 76.8512

# ============================================================================
# Date Range Configuration
# ============================================================================

START_DATE: Final[str] = "2024-01-01"
END_DATE: Final[str] = "2025-12-31"

# ============================================================================
# Data Processing Configuration
# ============================================================================

RANDOM_SEED: Final[int] = 42

# Retry settings for API calls and network operations
MAX_RETRIES: Final[int] = 3
RETRY_DELAY: Final[int] = 5  # seconds
TIMEOUT: Final[int] = 30  # seconds

# ============================================================================
# Directory Structure
# ============================================================================

# Data directories
DATA_DIR: Final[Path] = PROJECT_ROOT / "data"
RAW_DATA_DIR: Final[Path] = DATA_DIR / "raw"
INTERIM_DATA_DIR: Final[Path] = DATA_DIR / "interim"
PROCESSED_DATA_DIR: Final[Path] = DATA_DIR / "processed"

# Report directories
REPORTS_DIR: Final[Path] = PROJECT_ROOT / "reports"
FIGURES_DIR: Final[Path] = REPORTS_DIR / "figures"
TABLES_DIR: Final[Path] = REPORTS_DIR / "tables"

# Log directory
LOGS_DIR: Final[Path] = PROJECT_ROOT / "logs"

# Source code directory
SRC_DIR: Final[Path] = PROJECT_ROOT / "src"

# Notebooks directory
NOTEBOOKS_DIR: Final[Path] = PROJECT_ROOT / "notebooks"

# ============================================================================
# File Naming Conventions
# ============================================================================

# Default filename patterns
RAW_DATA_FILENAME: Final[str] = f"air_quality_{CITY.lower()}_raw.csv"
PROCESSED_DATA_FILENAME: Final[str] = f"air_quality_{CITY.lower()}_processed.csv"
MODEL_FILENAME: Final[str] = "forecast_model.pkl"

# ============================================================================
# Logging Configuration
# ============================================================================

LOG_FORMAT: Final[str] = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DATE_FORMAT: Final[str] = "%Y-%m-%d %H:%M:%S"
LOG_LEVEL: Final[str] = "INFO"

# ============================================================================
# Data Quality Thresholds
# ============================================================================

# Maximum allowable missing data percentage
MAX_MISSING_PERCENTAGE: Final[float] = 30.0

# Minimum required data points for analysis
MIN_DATA_POINTS: Final[int] = 100

# ============================================================================
# Model Configuration
# ============================================================================

# Train/test split ratio
TEST_SIZE: Final[float] = 0.2

# Cross-validation folds
CV_FOLDS: Final[int] = 5

# Feature engineering parameters
ROLLING_WINDOW_SIZES: Final[tuple[int, ...]] = (3, 7, 14, 30)
LAG_FEATURES: Final[tuple[int, ...]] = (1, 2, 3, 7, 14)


def get_config_summary() -> dict[str, str | float | int]:
    """
    Get a summary of key configuration parameters.
    
    Returns:
        Dictionary containing key configuration values.
    """
    return {
        "city": CITY,
        "country": COUNTRY,
        "timezone": TIMEZONE,
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "random_seed": RANDOM_SEED,
        "project_root": str(PROJECT_ROOT),
    }


if __name__ == "__main__":
    # Display configuration when run directly
    print("=" * 70)
    print("Almaty Air Quality Forecasting & Analysis - Configuration")
    print("=" * 70)
    
    config = get_config_summary()
    for key, value in config.items():
        print(f"{key:20s}: {value}")
    
    print("\n" + "=" * 70)
    print("Directory Structure:")
    print("=" * 70)
    
    directories = [
        RAW_DATA_DIR,
        INTERIM_DATA_DIR,
        PROCESSED_DATA_DIR,
        FIGURES_DIR,
        TABLES_DIR,
        LOGS_DIR,
        SRC_DIR,
        NOTEBOOKS_DIR,
    ]
    
    for directory in directories:
        rel_path = directory.relative_to(PROJECT_ROOT)
        print(f"  {rel_path}")