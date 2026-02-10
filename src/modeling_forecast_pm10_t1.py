"""
PM10 1-Hour-Ahead Forecasting Model
====================================

This module implements a production-quality machine learning pipeline for
forecasting PM10 concentrations 1 hour ahead in Almaty, Kazakhstan.

Uses classical ML models (Random Forest, Gradient Boosting, Linear Regression)
with proper time-series cross-validation and feature engineering.

Includes:
- Naive baseline (persistence model)
- Time-series cross-validation for all models
- Proper scaling (only for linear models)

Designed for inclusion in academic research reports.

Author: Research Data Science Team
Date: 2026-02-06
"""

import json
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from scipy import stats
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from src.config import FIGURES_DIR, LOGS_DIR, RAW_DATA_DIR, TABLES_DIR, CITY, COUNTRY, TIMEZONE

warnings.filterwarnings('ignore')

# Configure matplotlib
plt.style.use('default')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 10

FIG_DPI = 300
RANDOM_STATE = 42
N_CV_SPLITS = 5


# ============================================================================
# 1. LOGGING SETUP
# ============================================================================


def setup_logger(log_path: Path) -> logging.Logger:
    """
    Set up logging configuration with console and file handlers.
    
    Args:
        log_path: Path to log file
    
    Returns:
        Configured logger instance
    """
    # Ensure log directory exists
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger("modeling_forecast_pm10_t1")
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # File handler
    file_handler = logging.FileHandler(log_path, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    return logger


# ============================================================================
# 2. DATA LOADING AND PREPROCESSING
# ============================================================================


def load_and_validate_data(logger: logging.Logger) -> pd.DataFrame:
    """
    Load pollution data with robust timezone handling and validation.
    
    Args:
        logger: Logger instance
    
    Returns:
        Validated DataFrame with datetime, pm25, pm10 columns
    """
    logger.info("=" * 80)
    logger.info("LOADING AND VALIDATING DATA")
    logger.info("=" * 80)
    
    pollution_path = RAW_DATA_DIR / "pollution.csv"
    logger.info(f"\nLoading data from: {pollution_path}")
    
    if not pollution_path.exists():
        raise FileNotFoundError(f"Data file not found: {pollution_path}")
    
    # Load data
    df = pd.read_csv(pollution_path)
    logger.info(f"✓ Loaded {len(df):,} records")
    logger.info(f"  Columns: {df.columns.tolist()}")
    
    # Parse datetime with UTC first (handles mixed timezone offsets)
    logger.info("\nParsing datetime with robust timezone handling...")
    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce', utc=True)
    
    # Convert to target timezone
    df['datetime'] = df['datetime'].dt.tz_convert(TIMEZONE)
    logger.info(f"  ✓ Converted to {TIMEZONE}")
    
    # Drop NaT datetime rows
    nat_count = df['datetime'].isna().sum()
    if nat_count > 0:
        df = df.dropna(subset=['datetime'])
        logger.warning(f"  Dropped {nat_count} rows with invalid datetime")
    
    # Sort by datetime
    df = df.sort_values('datetime').reset_index(drop=True)
    logger.info(f"  ✓ Sorted by datetime")
    
    # Validate temporal properties
    logger.info("\nValidating temporal properties...")
    time_diffs = df['datetime'].diff().dropna()
    median_diff = time_diffs.median()
    logger.info(f"  Median time interval: {median_diff}")
    
    # Check if approximately hourly
    expected_hourly = pd.Timedelta(hours=1)
    if abs((median_diff - expected_hourly).total_seconds()) > 600:  # Allow 10min tolerance
        logger.warning(f"  Warning: Median interval ({median_diff}) differs from 1 hour")
    else:
        logger.info(f"  ✓ Confirmed hourly frequency")
    
    # Date range
    logger.info(f"  Date range: {df['datetime'].min()} to {df['datetime'].max()}")
    logger.info(f"  Time span: {(df['datetime'].max() - df['datetime'].min()).days} days")
    
    # Validate pm10
    logger.info("\nValidating PM10 data...")
    df['pm10'] = pd.to_numeric(df['pm10'], errors='coerce')
    pm10_valid = df['pm10'].notna().sum()
    pm10_missing_pct = (df['pm10'].isna().sum() / len(df)) * 100
    logger.info(f"  PM10: {pm10_valid:,} valid values ({pm10_missing_pct:.2f}% missing)")
    logger.info(f"  PM10 range: [{df['pm10'].min():.2f}, {df['pm10'].max():.2f}] µg/m³")
    logger.info(f"  PM10 mean: {df['pm10'].mean():.2f} µg/m³")
    
    # Validate pm25 if exists
    if 'pm25' in df.columns:
        logger.info("\nValidating PM2.5 data...")
        df['pm25'] = pd.to_numeric(df['pm25'], errors='coerce')
        pm25_valid = df['pm25'].notna().sum()
        pm25_missing_pct = (df['pm25'].isna().sum() / len(df)) * 100
        logger.info(f"  PM2.5: {pm25_valid:,} valid values ({pm25_missing_pct:.2f}% missing)")
        logger.info(f"  PM2.5 range: [{df['pm25'].min():.2f}, {df['pm25'].max():.2f}] µg/m³")
    
    logger.info(f"\n✓ Final dataset: {len(df):,} records")
    logger.info("=" * 80)
    
    return df


# ============================================================================
# 3. FEATURE ENGINEERING
# ============================================================================


def create_features(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """
    Create time-series features for PM10 forecasting.
    
    Features include:
    - Calendar features (hour, day_of_week, month, year)
    - Lag features for pm10 (1, 2, 3, 6, 12, 24, 48, 168 hours)
    - Rolling statistics (mean, std) for 24h and 168h windows
    - PM2.5 lag features if available (exogenous predictor)
    - Target: pm10 shifted -1 hour (1-hour-ahead prediction)
    
    Args:
        df: Input DataFrame with datetime, pm25, pm10 columns
        logger: Logger instance
    
    Returns:
        DataFrame with engineered features
    """
    logger.info("\n" + "=" * 80)
    logger.info("FEATURE ENGINEERING")
    logger.info("=" * 80)
    
    df = df.copy()
    
    # Calendar features
    logger.info("\n1. Creating calendar features...")
    df['hour'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['month'] = df['datetime'].dt.month
    df['year'] = df['datetime'].dt.year
    df['day_of_year'] = df['datetime'].dt.dayofyear
    
    # Cyclical encoding for hour and month
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    logger.info("  ✓ Created: hour, day_of_week, month, year, day_of_year")
    logger.info("  ✓ Created: hour_sin, hour_cos, month_sin, month_cos (cyclical)")
    
    # PM10 lag features
    logger.info("\n2. Creating PM10 lag features...")
    lag_hours = [1, 2, 3, 6, 12, 24, 48, 168]
    
    for lag in lag_hours:
        df[f'pm10_lag_{lag}'] = df['pm10'].shift(lag)
        logger.info(f"  ✓ Created: pm10_lag_{lag}")
    
    # Rolling statistics (using shift to prevent leakage)
    logger.info("\n3. Creating PM10 rolling statistics...")
    
    # 24-hour rolling stats
    df['pm10_roll_mean_24'] = df['pm10'].shift(1).rolling(window=24, min_periods=12).mean()
    df['pm10_roll_std_24'] = df['pm10'].shift(1).rolling(window=24, min_periods=12).std()
    df['pm10_roll_min_24'] = df['pm10'].shift(1).rolling(window=24, min_periods=12).min()
    df['pm10_roll_max_24'] = df['pm10'].shift(1).rolling(window=24, min_periods=12).max()
    logger.info("  ✓ Created: pm10_roll_mean_24, pm10_roll_std_24, pm10_roll_min_24, pm10_roll_max_24")
    
    # 168-hour (7-day) rolling stats
    df['pm10_roll_mean_168'] = df['pm10'].shift(1).rolling(window=168, min_periods=84).mean()
    df['pm10_roll_std_168'] = df['pm10'].shift(1).rolling(window=168, min_periods=84).std()
    logger.info("  ✓ Created: pm10_roll_mean_168, pm10_roll_std_168")
    
    # Rate of change features
    df['pm10_diff_1'] = df['pm10'].diff(1)
    df['pm10_diff_24'] = df['pm10'].diff(24)
    logger.info("  ✓ Created: pm10_diff_1, pm10_diff_24 (rate of change)")
    
    # PM2.5 exogenous features (if available)
    if 'pm25' in df.columns and df['pm25'].notna().sum() > 0:
        logger.info("\n4. Creating PM2.5 exogenous features...")
        df['pm25_lag_1'] = df['pm25'].shift(1)
        df['pm25_roll_mean_24'] = df['pm25'].shift(1).rolling(window=24, min_periods=12).mean()
        logger.info("  ✓ Created: pm25_lag_1, pm25_roll_mean_24 (exogenous predictor)")
    
    # Target variable: 1-hour-ahead PM10
    logger.info("\n5. Creating target variable...")
    df['target_pm10_t_plus_1'] = df['pm10'].shift(-1)
    logger.info("  ✓ Created: target_pm10_t_plus_1 (pm10 shifted -1 hour)")
    
    # Drop rows with NaN target (last row)
    initial_len = len(df)
    df = df.dropna(subset=['target_pm10_t_plus_1'])
    logger.info(f"  ✓ Dropped {initial_len - len(df)} row(s) with NaN target")
    
    # Summary
    logger.info(f"\n✓ Feature engineering complete")
    logger.info(f"  Total features: {len(df.columns)}")
    logger.info(f"  Records with features: {len(df):,}")
    
    # Check for NaN in features
    nan_summary = df.isna().sum()
    nan_features = nan_summary[nan_summary > 0]
    if len(nan_features) > 0:
        logger.info(f"\n  Features with NaN values:")
        for feat, count in nan_features.items():
            pct = (count / len(df)) * 100
            logger.info(f"    {feat}: {count:,} ({pct:.2f}%)")
    
    logger.info("=" * 80)
    
    return df


# ============================================================================
# 4. TRAIN-TEST SPLIT (TIME-SERIES AWARE)
# ============================================================================


def train_test_split_timeseries(
    df: pd.DataFrame,
    test_size: float = 0.2,
    logger: logging.Logger = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into train and test sets respecting temporal order.
    
    Args:
        df: DataFrame with datetime index
        test_size: Fraction of data for test set
        logger: Logger instance
    
    Returns:
        Tuple of (train_df, test_df)
    """
    if logger:
        logger.info("\n" + "=" * 80)
        logger.info("TRAIN-TEST SPLIT (TIME-SERIES)")
        logger.info("=" * 80)
    
    # Sort by datetime to ensure temporal order
    df = df.sort_values('datetime').reset_index(drop=True)
    
    # Calculate split point
    split_idx = int(len(df) * (1 - test_size))
    
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    
    if logger:
        logger.info(f"\nSplit configuration:")
        logger.info(f"  Test size: {test_size * 100:.0f}%")
        logger.info(f"  Split index: {split_idx:,}")
        logger.info(f"\nTrain set:")
        logger.info(f"  Records: {len(train_df):,}")
        logger.info(f"  Date range: {train_df['datetime'].min()} to {train_df['datetime'].max()}")
        logger.info(f"\nTest set:")
        logger.info(f"  Records: {len(test_df):,}")
        logger.info(f"  Date range: {test_df['datetime'].min()} to {test_df['datetime'].max()}")
        logger.info("\n✓ No data leakage: test set is strictly after train set")
        logger.info("=" * 80)
    
    return train_df, test_df


# ============================================================================
# 5. FEATURE SELECTION AND PREPARATION
# ============================================================================


def prepare_features_and_target(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    logger: logging.Logger
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], StandardScaler, pd.Series, pd.Series, pd.DataFrame]:
    """
    Select features, handle missing values, and prepare data.
    
    Returns RAW features (no scaling) for tree-based models,
    and provides scaler for linear models.
    
    Args:
        train_df: Training DataFrame
        test_df: Test DataFrame
        logger: Logger instance
    
    Returns:
        Tuple of (X_train, X_test, y_train, y_test, feature_names, scaler, 
                  train_dates, test_dates, test_df_filtered)
    """
    logger.info("\n" + "=" * 80)
    logger.info("FEATURE PREPARATION")
    logger.info("=" * 80)
    
    # Define feature columns (exclude datetime and target)
    exclude_cols = ['datetime', 'target_pm10_t_plus_1', 'pm25', 'pm10', 'date']
    
    feature_cols = [col for col in train_df.columns if col not in exclude_cols]
    
    logger.info(f"\nFeature selection:")
    logger.info(f"  Total features: {len(feature_cols)}")
    logger.info(f"  Feature list:")
    for i, feat in enumerate(feature_cols, 1):
        logger.info(f"    {i:2d}. {feat}")
    
    # Extract features, target, and datetime
    X_train = train_df[feature_cols].copy()
    X_test = test_df[feature_cols].copy()
    y_train = train_df['target_pm10_t_plus_1'].values
    y_test = test_df['target_pm10_t_plus_1'].values
    train_dates = train_df['datetime'].copy()
    test_dates = test_df['datetime'].copy()
    train_df_filtered = train_df.copy()
    test_df_filtered = test_df.copy()
    
    # Handle missing values
    logger.info(f"\nHandling missing values...")
    
    # Check for NaN in features
    train_nan_count = X_train.isna().sum().sum()
    test_nan_count = X_test.isna().sum().sum()
    
    if train_nan_count > 0 or test_nan_count > 0:
        logger.info(f"  Train NaN count: {train_nan_count:,}")
        logger.info(f"  Test NaN count: {test_nan_count:,}")
        logger.info(f"  Strategy: Drop rows with any NaN in features")
        
        # Get valid indices
        train_valid_idx = X_train.notna().all(axis=1)
        test_valid_idx = X_test.notna().all(axis=1)
        
        # Filter data AND datetime AND full dataframe
        X_train = X_train[train_valid_idx]
        y_train = y_train[train_valid_idx]
        train_dates = train_dates[train_valid_idx]
        train_df_filtered = train_df_filtered[train_valid_idx]
        
        X_test = X_test[test_valid_idx]
        y_test = y_test[test_valid_idx]
        test_dates = test_dates[test_valid_idx]
        test_df_filtered = test_df_filtered[test_valid_idx]
        
        logger.info(f"  Train records after filtering: {len(X_train):,}")
        logger.info(f"  Test records after filtering: {len(X_test):,}")
    else:
        logger.info(f"  ✓ No missing values in features")
    
    # Check for NaN in target
    train_target_nan = np.isnan(y_train).sum()
    test_target_nan = np.isnan(y_test).sum()
    
    if train_target_nan > 0 or test_target_nan > 0:
        logger.info(f"\nHandling NaN in target...")
        logger.info(f"  Train target NaN: {train_target_nan}")
        logger.info(f"  Test target NaN: {test_target_nan}")
        
        train_target_valid = ~np.isnan(y_train)
        test_target_valid = ~np.isnan(y_test)
        
        X_train = X_train[train_target_valid]
        y_train = y_train[train_target_valid]
        train_dates = train_dates[train_target_valid]
        train_df_filtered = train_df_filtered[train_target_valid]
        
        X_test = X_test[test_target_valid]
        y_test = y_test[test_target_valid]
        test_dates = test_dates[test_target_valid]
        test_df_filtered = test_df_filtered[test_target_valid]
        
        logger.info(f"  Final train records: {len(X_train):,}")
        logger.info(f"  Final test records: {len(X_test):,}")
    
    # Reset index for all filtered data to ensure alignment
    train_dates = train_dates.reset_index(drop=True)
    test_dates = test_dates.reset_index(drop=True)
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    train_df_filtered = train_df_filtered.reset_index(drop=True)
    test_df_filtered = test_df_filtered.reset_index(drop=True)
    
    # Prepare scaler (will be used only for linear models)
    logger.info(f"\nPreparing StandardScaler (for linear models only)...")
    scaler = StandardScaler()
    scaler.fit(X_train)
    logger.info(f"  ✓ Scaler fitted on training data")
    
    # Summary statistics
    logger.info(f"\nFinal dataset statistics:")
    logger.info(f"  Train samples: {len(X_train):,}")
    logger.info(f"  Test samples: {len(X_test):,}")
    logger.info(f"  Features: {X_train.shape[1]}")
    logger.info(f"  Train target range: [{y_train.min():.2f}, {y_train.max():.2f}] µg/m³")
    logger.info(f"  Test target range: [{y_test.min():.2f}, {y_test.max():.2f}] µg/m³")
    logger.info(f"  Train target mean: {y_train.mean():.2f} µg/m³")
    logger.info(f"  Test target mean: {y_test.mean():.2f} µg/m³")
    logger.info(f"  Train dates aligned: {len(train_dates)}")
    logger.info(f"  Test dates aligned: {len(test_dates)}")
    logger.info(f"  Test DataFrame filtered aligned: {len(test_df_filtered)}")
    
    logger.info("\n✓ Returning RAW features (unscaled)")
    logger.info("  Tree models will use raw features")
    logger.info("  Linear models will apply scaling internally")
    
    logger.info("=" * 80)
    
    return X_train, X_test, y_train, y_test, feature_cols, scaler, train_dates, test_dates, test_df_filtered


# ============================================================================
# 6. NAIVE BASELINE EVALUATION
# ============================================================================


def evaluate_naive_baseline(
    test_df_filtered: pd.DataFrame,
    y_test: np.ndarray,
    logger: logging.Logger
) -> Dict[str, float]:
    """
    Evaluate naive persistence baseline: y(t+1) = y(t).
    
    The baseline predicts next hour PM10 as current hour PM10.
    
    Args:
        test_df_filtered: Filtered test DataFrame aligned with y_test
        y_test: True target values (t+1)
        logger: Logger instance
    
    Returns:
        Dictionary with baseline metrics
    """
    logger.info("\n" + "=" * 80)
    logger.info("NAIVE BASELINE EVALUATION")
    logger.info("=" * 80)
    logger.info("\nBaseline Model: Persistence (y[t+1] = y[t])")
    logger.info("Description: Current PM10 value predicts next hour")
    
    # Baseline prediction: current pm10 value
    y_pred_naive = test_df_filtered['pm10'].values
    
    # Ensure alignment
    min_len = min(len(y_test), len(y_pred_naive))
    y_test_aligned = y_test[:min_len]
    y_pred_naive = y_pred_naive[:min_len]
    
    logger.info(f"\nTest samples: {len(y_test_aligned):,}")
    
    # Calculate metrics
    baseline_rmse = np.sqrt(mean_squared_error(y_test_aligned, y_pred_naive))
    baseline_mae = mean_absolute_error(y_test_aligned, y_pred_naive)
    baseline_r2 = r2_score(y_test_aligned, y_pred_naive)
    baseline_mape = mean_absolute_percentage_error(y_test_aligned, y_pred_naive) * 100
    
    logger.info(f"\nNaive Baseline Performance:")
    logger.info(f"  RMSE: {baseline_rmse:.4f} µg/m³")
    logger.info(f"  MAE:  {baseline_mae:.4f} µg/m³")
    logger.info(f"  R²:   {baseline_r2:.4f}")
    logger.info(f"  MAPE: {baseline_mape:.2f}%")
    
    logger.info("\n✓ Baseline evaluation complete")
    logger.info("  ML models should outperform this baseline")
    logger.info("=" * 80)
    
    return {
        'rmse': baseline_rmse,
        'mae': baseline_mae,
        'r2': baseline_r2,
        'mape': baseline_mape
    }


# ============================================================================
# 7. TIME-SERIES CROSS-VALIDATION
# ============================================================================


def time_series_cv_evaluation(
    model,
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int,
    logger: logging.Logger,
    model_name: str,
    scaler: StandardScaler = None,
    use_scaling: bool = False
) -> Dict[str, float]:
    """
    Perform time-series cross-validation using TimeSeriesSplit.
    
    Args:
        model: Sklearn model instance
        X: Feature matrix
        y: Target vector
        n_splits: Number of CV splits
        logger: Logger instance
        model_name: Name of model for logging
        scaler: StandardScaler instance (only used if use_scaling=True)
        use_scaling: Whether to apply scaling (for linear models)
    
    Returns:
        Dictionary with mean and std of CV RMSE
    """
    logger.info(f"\n  Time-Series Cross-Validation ({n_splits} splits):")
    
    tscv = TimeSeriesSplit(n_splits=n_splits)
    cv_rmse_scores = []
    
    from sklearn.base import clone
    
    for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
        
        # Safe indexing (pandas / numpy compatible)
        if isinstance(X, pd.DataFrame):
            X_train_fold = X.iloc[train_idx].values
            X_val_fold = X.iloc[val_idx].values
        else:
            X_train_fold = X[train_idx]
            X_val_fold = X[val_idx]
        
        y_train_fold = y[train_idx]
        y_val_fold = y[val_idx]
        
        # Clone model to avoid state leakage
        m = clone(model)
        
        # Apply scaling if required (for linear models)
        if use_scaling and scaler is not None:
            fold_scaler = StandardScaler()
            X_train_fold = fold_scaler.fit_transform(X_train_fold)
            X_val_fold = fold_scaler.transform(X_val_fold)
        
        # Train and predict
        m.fit(X_train_fold, y_train_fold)
        y_val_pred = m.predict(X_val_fold)
        
        # Calculate RMSE
        fold_rmse = np.sqrt(mean_squared_error(y_val_fold, y_val_pred))
        cv_rmse_scores.append(fold_rmse)
        
        logger.info(f"    Fold {fold_idx}: RMSE = {fold_rmse:.4f} µg/m³")
    
    # Calculate statistics
    cv_mean = np.mean(cv_rmse_scores)
    cv_std = np.std(cv_rmse_scores)
    
    logger.info(f"  CV RMSE: {cv_mean:.4f} ± {cv_std:.4f} µg/m³")
    
    return {
        'cv_mean': cv_mean,
        'cv_std': cv_std,
        'cv_scores': cv_rmse_scores
    }


# ============================================================================
# 8. MODEL TRAINING AND EVALUATION
# ============================================================================


def train_and_evaluate_models(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    feature_names: List[str],
    scaler: StandardScaler,
    logger: logging.Logger
) -> Dict[str, Dict[str, Any]]:
    """
    Train multiple models and evaluate performance.
    
    Models:
    - Linear Regression (baseline, with scaling)
    - Ridge Regression (with scaling)
    - Lasso Regression (with scaling)
    - Random Forest (RAW features)
    - Gradient Boosting (RAW features)
    
    Includes time-series cross-validation for all models.
    
    Args:
        X_train: Training features (RAW, unscaled)
        X_test: Test features (RAW, unscaled)
        y_train: Training target
        y_test: Test target
        feature_names: List of feature names
        scaler: Fitted StandardScaler
        logger: Logger instance
    
    Returns:
        Dictionary of model results
    """
    logger.info("\n" + "=" * 80)
    logger.info("MODEL TRAINING AND EVALUATION")
    logger.info("=" * 80)
    
    results = {}
    
    # Define models with scaling indicator
    models_config = [
        (
            'Linear Regression',
            Pipeline([
                ('scaler', StandardScaler()),
                ('model', LinearRegression())
            ]),
            False
        ),
        (
            'Ridge Regression',
            Pipeline([
                ('scaler', StandardScaler()),
                ('model', Ridge(alpha=1.0))
            ]),
            False
        ),
        (
            'Lasso Regression',
            Pipeline([
                ('scaler', StandardScaler()),
                ('model', Lasso(alpha=0.1, max_iter=10000, random_state=RANDOM_STATE))
            ]),
            False
        ),
        (
            'Random Forest',
            RandomForestRegressor(
                n_estimators=100,
                max_depth=15,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=RANDOM_STATE,
                n_jobs=-1
            ),
            False
        ),
        (
            'Gradient Boosting',
            GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                min_samples_split=10,
                random_state=RANDOM_STATE
            ),
            False
        )
    ]
    
    # Train and evaluate each model
    for model_name, model, use_scaling in models_config:
        logger.info(f"\n{'=' * 80}")
        logger.info(f"Training: {model_name}")
        logger.info(f"{'=' * 80}")
        
        # Prepare data
        X_train_model = X_train
        X_test_model = X_test
        
        # Train
        logger.info(f"  Training model...")
        model.fit(X_train_model, y_train)
        logger.info(f"  ✓ Training complete")
        
        # Cross-validation on training set
        cv_results = time_series_cv_evaluation(
            model=model,
            X=X_train,
            y=y_train,
            n_splits=N_CV_SPLITS,
            logger=logger,
            model_name=model_name,
            scaler=scaler if use_scaling else None,
            use_scaling=use_scaling
        )
        
        # Predict
        logger.info(f"\n  Making predictions on train and test sets...")
        y_train_pred = model.predict(X_train_model)
        y_test_pred = model.predict(X_test_model)
        logger.info(f"  ✓ Predictions complete")
        
        # Evaluate
        logger.info(f"\n  Evaluating performance...")
        
        # Training metrics
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        train_mae = mean_absolute_error(y_train, y_train_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        train_mape = mean_absolute_percentage_error(y_train, y_train_pred) * 100
        
        # Test metrics
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        test_mae = mean_absolute_error(y_test, y_test_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        test_mape = mean_absolute_percentage_error(y_test, y_test_pred) * 100
        
        logger.info(f"\n  Train Performance:")
        logger.info(f"    RMSE: {train_rmse:.4f} µg/m³")
        logger.info(f"    MAE:  {train_mae:.4f} µg/m³")
        logger.info(f"    R²:   {train_r2:.4f}")
        logger.info(f"    MAPE: {train_mape:.2f}%")
        
        logger.info(f"\n  Test Performance:")
        logger.info(f"    RMSE: {test_rmse:.4f} µg/m³")
        logger.info(f"    MAE:  {test_mae:.4f} µg/m³")
        logger.info(f"    R²:   {test_r2:.4f}")
        logger.info(f"    MAPE: {test_mape:.2f}%")
        
        # Feature importance (for tree-based models)
        feature_importance = None
        if hasattr(model, 'feature_importances_'):
            feature_importance = model.feature_importances_
            logger.info(f"\n  Top 10 Important Features:")
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': feature_importance
            }).sort_values('importance', ascending=False)
            
            for idx, row in importance_df.head(10).iterrows():
                logger.info(f"    {row['feature']:30s}: {row['importance']:.4f}")
        
        # Store results
        results[model_name] = {
            'model': model,
            'train_rmse': train_rmse,
            'train_mae': train_mae,
            'train_r2': train_r2,
            'train_mape': train_mape,
            'test_rmse': test_rmse,
            'test_mae': test_mae,
            'test_r2': test_r2,
            'test_mape': test_mape,
            'cv_mean': cv_results['cv_mean'],
            'cv_std': cv_results['cv_std'],
            'y_train_pred': y_train_pred,
            'y_test_pred': y_test_pred,
            'feature_importance': feature_importance
        }
    
    # Summary comparison
    logger.info(f"\n{'=' * 80}")
    logger.info("MODEL COMPARISON SUMMARY")
    logger.info(f"{'=' * 80}")
    logger.info(f"\n{'Model':<25s} {'CV RMSE':>15s} {'Test RMSE':>12s} {'Test MAE':>12s} {'Test R²':>10s}")
    logger.info("-" * 80)
    
    for model_name, res in results.items():
        cv_str = f"{res['cv_mean']:.4f}±{res['cv_std']:.4f}"
        logger.info(f"{model_name:<25s} "
                   f"{cv_str:>15s} "
                   f"{res['test_rmse']:>12.4f} "
                   f"{res['test_mae']:>12.4f} "
                   f"{res['test_r2']:>10.4f}")
    
    # Best model
    best_model_name = min(results, key=lambda k: results[k]['test_rmse'])
    logger.info(f"\n✓ Best model (lowest Test RMSE): {best_model_name}")
    logger.info(f"  Test RMSE: {results[best_model_name]['test_rmse']:.4f} µg/m³")
    logger.info(f"  Test R²: {results[best_model_name]['test_r2']:.4f}")
    logger.info(f"  CV RMSE: {results[best_model_name]['cv_mean']:.4f} ± {results[best_model_name]['cv_std']:.4f} µg/m³")
    
    logger.info("=" * 80)
    
    return results


# ============================================================================
# 9. VISUALIZATION
# ============================================================================


def plot_model_comparison(results: Dict, baseline_metrics: Dict, output_path: Path, logger: logging.Logger) -> None:
    """Plot comparison of model performance metrics including baseline."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=FIG_DPI)
    
    all_model_names = ['Naive Baseline'] + list(results.keys())
    test_rmse = [baseline_metrics['rmse']] + [results[m]['test_rmse'] for m in results.keys()]
    test_mae = [baseline_metrics['mae']] + [results[m]['test_mae'] for m in results.keys()]
    test_r2 = [baseline_metrics['r2']] + [results[m]['test_r2'] for m in results.keys()]
    test_mape = [baseline_metrics['mape']] + [results[m]['test_mape'] for m in results.keys()]
    
    colors = ['#FF6B6B', '#E63946', '#457B9D', '#06A77D', '#F77F00', '#9D4EDD']
    
    ax1 = axes[0, 0]
    ax1.bar(range(len(all_model_names)), test_rmse, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_xticks(range(len(all_model_names)))
    ax1.set_xticklabels(all_model_names, rotation=45, ha='right', fontsize=9)
    ax1.set_ylabel('RMSE (µg/m³)', fontweight='bold')
    ax1.set_title('Root Mean Squared Error', fontweight='bold', pad=10)
    ax1.grid(True, alpha=0.3, axis='y')
    
    ax2 = axes[0, 1]
    ax2.bar(range(len(all_model_names)), test_mae, color=colors, alpha=0.7, edgecolor='black')
    ax2.set_xticks(range(len(all_model_names)))
    ax2.set_xticklabels(all_model_names, rotation=45, ha='right', fontsize=9)
    ax2.set_ylabel('MAE (µg/m³)', fontweight='bold')
    ax2.set_title('Mean Absolute Error', fontweight='bold', pad=10)
    ax2.grid(True, alpha=0.3, axis='y')
    
    ax3 = axes[1, 0]
    ax3.bar(range(len(all_model_names)), test_r2, color=colors, alpha=0.7, edgecolor='black')
    ax3.set_xticks(range(len(all_model_names)))
    ax3.set_xticklabels(all_model_names, rotation=45, ha='right', fontsize=9)
    ax3.set_ylabel('R² Score', fontweight='bold')
    ax3.set_title('R² Score', fontweight='bold', pad=10)
    ax3.set_ylim([0, 1])
    ax3.grid(True, alpha=0.3, axis='y')
    
    ax4 = axes[1, 1]
    ax4.bar(range(len(all_model_names)), test_mape, color=colors, alpha=0.7, edgecolor='black')
    ax4.set_xticks(range(len(all_model_names)))
    ax4.set_xticklabels(all_model_names, rotation=45, ha='right', fontsize=9)
    ax4.set_ylabel('MAPE (%)', fontweight='bold')
    ax4.set_title('Mean Absolute Percentage Error', fontweight='bold', pad=10)
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(f'Model Performance Comparison - PM10\n{CITY}, {COUNTRY}', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=FIG_DPI, bbox_inches='tight')
    plt.close()
    logger.info(f"  ✓ Saved: {output_path.name}")


def plot_predictions_vs_actual(y_test, results, test_dates, output_path, logger):
    """Plot predicted vs actual."""
    best_model_name = min(results, key=lambda k: results[k]['test_rmse'])
    y_pred = results[best_model_name]['y_test_pred']
    
    fig, axes = plt.subplots(2, 1, figsize=(16, 10), dpi=FIG_DPI)
    
    ax1 = axes[0]
    ax1.plot(test_dates, y_test, label='Actual', linewidth=1.5, alpha=0.8, color='#E63946')
    ax1.plot(test_dates, y_pred, label='Predicted', linewidth=1.5, alpha=0.8, color='#457B9D')
    ax1.set_xlabel('Date', fontweight='bold')
    ax1.set_ylabel('PM10 (µg/m³)', fontweight='bold')
    ax1.set_title(f'1-Hour-Ahead PM10 Forecast\nModel: {best_model_name}', fontweight='bold', pad=15)
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    fig.autofmt_xdate(rotation=45, ha='right')
    
    ax2 = axes[1]
    ax2.scatter(y_test, y_pred, alpha=0.5, s=20, color='#457B9D')
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect')
    ax2.set_xlabel('Actual PM10 (µg/m³)', fontweight='bold')
    ax2.set_ylabel('Predicted PM10 (µg/m³)', fontweight='bold')
    ax2.set_title('Prediction Accuracy', fontweight='bold', pad=15)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=FIG_DPI, bbox_inches='tight')
    plt.close()
    logger.info(f"  ✓ Saved: {output_path.name}")


def plot_residual_analysis(y_test, results, output_path, logger):
    """Plot residual analysis."""
    best_model_name = min(results, key=lambda k: results[k]['test_rmse'])
    y_pred = results[best_model_name]['y_test_pred']
    residuals = y_test - y_pred
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=FIG_DPI)
    
    axes[0, 0].scatter(y_pred, residuals, alpha=0.5, s=20, color='#457B9D')
    axes[0, 0].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[0, 0].set_xlabel('Predicted', fontweight='bold')
    axes[0, 0].set_ylabel('Residuals', fontweight='bold')
    axes[0, 0].set_title('Residuals vs Predicted', fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].hist(residuals, bins=50, color='#457B9D', alpha=0.7, edgecolor='black')
    axes[0, 1].axvline(x=0, color='r', linestyle='--', linewidth=2)
    axes[0, 1].set_xlabel('Residuals', fontweight='bold')
    axes[0, 1].set_ylabel('Frequency', fontweight='bold')
    axes[0, 1].set_title('Residuals Distribution', fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    stats.probplot(residuals, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title('Q-Q Plot', fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].plot(residuals, linewidth=0.8, alpha=0.7, color='#457B9D')
    axes[1, 1].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[1, 1].set_xlabel('Sample Index', fontweight='bold')
    axes[1, 1].set_ylabel('Residuals', fontweight='bold')
    axes[1, 1].set_title('Residuals Over Time', fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle(f'Residual Analysis - {best_model_name}', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=FIG_DPI, bbox_inches='tight')
    plt.close()
    logger.info(f"  ✓ Saved: {output_path.name}")


def plot_feature_importance(results, feature_names, output_path, logger):
    """Plot feature importance."""
    models_with_importance = {name: res for name, res in results.items() if res['feature_importance'] is not None}
    if not models_with_importance:
        return
    
    n_models = len(models_with_importance)
    fig, axes = plt.subplots(1, n_models, figsize=(7 * n_models, 8), dpi=FIG_DPI)
    if n_models == 1:
        axes = [axes]
    
    for ax, (model_name, res) in zip(axes, models_with_importance.items()):
        importance_df = pd.DataFrame({'feature': feature_names, 'importance': res['feature_importance']}).sort_values('importance', ascending=False).head(15)
        ax.barh(range(len(importance_df)), importance_df['importance'], color='#457B9D', alpha=0.7, edgecolor='black')
        ax.set_yticks(range(len(importance_df)))
        ax.set_yticklabels(importance_df['feature'])
        ax.set_xlabel('Importance', fontweight='bold')
        ax.set_title(f'Top 15 Features\n{model_name}', fontweight='bold')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis='x')
    
    plt.suptitle(f'Feature Importance - {CITY}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=FIG_DPI, bbox_inches='tight')
    plt.close()
    logger.info(f"  ✓ Saved: {output_path.name}")


def save_results_table(results, baseline_metrics, output_path, logger):
    """Save results table."""
    rows = [{'Model': 'Naive Baseline', 'CV_RMSE_Mean': None, 'CV_RMSE_Std': None, 'Train_RMSE': None, 
             'Train_MAE': None, 'Train_R2': None, 'Train_MAPE': None, 
             'Test_RMSE': baseline_metrics['rmse'], 'Test_MAE': baseline_metrics['mae'], 
             'Test_R2': baseline_metrics['r2'], 'Test_MAPE': baseline_metrics['mape']}]
    
    for model_name, res in results.items():
        rows.append({'Model': model_name, 'CV_RMSE_Mean': res['cv_mean'], 'CV_RMSE_Std': res['cv_std'],
                     'Train_RMSE': res['train_rmse'], 'Train_MAE': res['train_mae'], 
                     'Train_R2': res['train_r2'], 'Train_MAPE': res['train_mape'],
                     'Test_RMSE': res['test_rmse'], 'Test_MAE': res['test_mae'], 
                     'Test_R2': res['test_r2'], 'Test_MAPE': res['test_mape']})
    
    pd.DataFrame(rows).to_csv(output_path, index=False)
    logger.info(f"  ✓ Saved: {output_path.name}")


# ============================================================================
# MAIN
# ============================================================================


def main():
    """Main execution function."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logger(LOGS_DIR / "modeling_forecast_pm10_t1.log")
    
    logger.info("=" * 80)
    logger.info("PM10 1-HOUR-AHEAD FORECASTING PIPELINE")
    logger.info("=" * 80)
    logger.info(f"Location: {CITY}, {COUNTRY}")
    logger.info(f"Timezone: {TIMEZONE}")
    logger.info(f"Target: PM10 concentration 1 hour ahead")
    logger.info(f"Random seed: {RANDOM_STATE}")
    logger.info(f"CV splits: {N_CV_SPLITS}")
    logger.info("=" * 80)
    
    try:
        df = load_and_validate_data(logger)
        df_features = create_features(df, logger)
        train_df, test_df = train_test_split_timeseries(df_features, test_size=0.2, logger=logger)
        X_train, X_test, y_train, y_test, feature_names, scaler, train_dates, test_dates, test_df_filtered = prepare_features_and_target(train_df, test_df, logger)
        
        baseline_metrics = evaluate_naive_baseline(test_df_filtered, y_test, logger)
        results = train_and_evaluate_models(X_train, X_test, y_train, y_test, feature_names, scaler, logger)
        
        logger.info("\n" + "=" * 80)
        logger.info("GENERATING VISUALIZATIONS")
        logger.info("=" * 80)
        
        plot_model_comparison(results, baseline_metrics, FIGURES_DIR / "model_comparison_pm10_t1.png", logger)
        plot_predictions_vs_actual(y_test, results, test_dates, FIGURES_DIR / "predictions_vs_actual_pm10_t1.png", logger)
        plot_residual_analysis(y_test, results, FIGURES_DIR / "residual_analysis_pm10_t1.png", logger)
        plot_feature_importance(results, feature_names, FIGURES_DIR / "feature_importance_pm10_t1.png", logger)
        
        logger.info("\n" + "=" * 80)
        logger.info("SAVING RESULTS")
        logger.info("=" * 80)
        save_results_table(results, baseline_metrics, TABLES_DIR / "model_performance_pm10_t1.csv", logger)
        
        logger.info("\n" + "=" * 80)
        logger.info("PIPELINE COMPLETE")
        logger.info("=" * 80)
        logger.info(f"\nKey Improvements:")
        logger.info(f"  ✓ Naive baseline evaluated")
        logger.info(f"  ✓ Time-series CV ({N_CV_SPLITS} splits)")
        logger.info(f"  ✓ Proper scaling (linear models only)")
        logger.info("=" * 80)
        
        return 0
        
    except Exception as e:
        logger.error(f"\nFATAL ERROR: {str(e)}", exc_info=True)
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())