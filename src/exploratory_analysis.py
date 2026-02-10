"""
Exploratory Data Analysis for Almaty Air Quality Data
======================================================

This module performs comprehensive exploratory analysis of PM2.5 and PM10
pollution data for Almaty, Kazakhstan. Analysis is designed for inclusion
in a 30-35 page academic research report.

Author: Research Data Science Team
Date: 2026-02-06
"""

import logging
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import FIGURES_DIR, LOGS_DIR, RAW_DATA_DIR, CITY, COUNTRY, TIMEZONE


# Configure visualization style for scientific publications
plt.style.use('default')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9

# Global figure parameters for publication quality
FIG_DPI = 300
FIG_FORMAT = "png"


# ============================================================================
# LOGGING SETUP
# ============================================================================


def setup_logger(log_path: Path) -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        log_path: Path to log file
    
    Returns:
        Configured logger instance
    """
    # Ensure log directory exists
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger("exploratory_analysis")
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
# 1. DATA OVERVIEW AND DESCRIPTIVE STATISTICS
# ============================================================================


def print_data_overview(df: pd.DataFrame, logger: logging.Logger) -> None:
    """
    Print comprehensive data overview including shape, types, and statistics.
    
    Args:
        df: Pollution DataFrame with columns datetime, pm25, pm10
        logger: Logger instance for output
    """
    logger.info("=" * 80)
    logger.info("DATA OVERVIEW")
    logger.info("=" * 80)
    
    # Basic shape and structure
    logger.info(f"\nDataset Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    logger.info(f"Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Column types
    logger.info("\nColumn Data Types:")
    for col in df.columns:
        logger.info(f"  {col:15s}: {df[col].dtype}")
    
    # Temporal coverage
    logger.info("\nTemporal Coverage:")
    logger.info(f"  Start date:  {df['datetime'].min()}")
    logger.info(f"  End date:    {df['datetime'].max()}")
    time_span = (df['datetime'].max() - df['datetime'].min()).days
    logger.info(f"  Time span:   {time_span} days ({time_span/365.25:.2f} years)")
    
    # Temporal resolution
    time_diff = df['datetime'].diff().dropna()
    median_interval = time_diff.median()
    logger.info(f"  Median interval: {median_interval}")
    
    # Missing data analysis
    logger.info("\nMissing Data:")
    total_records = len(df)
    for col in ['pm25', 'pm10']:
        missing_count = df[col].isna().sum()
        missing_pct = (missing_count / total_records) * 100
        valid_count = df[col].notna().sum()
        logger.info(f"  {col.upper():5s}: {valid_count:,} valid, "
                   f"{missing_count:,} missing ({missing_pct:.2f}%)")
    
    # Summary statistics
    logger.info("\nSummary Statistics:")
    logger.info("\nPM2.5 (µg/m³):")
    if df['pm25'].notna().any():
        stats = df['pm25'].describe()
        logger.info(f"  Count:  {stats['count']:,.0f}")
        logger.info(f"  Mean:   {stats['mean']:.2f}")
        logger.info(f"  Std:    {stats['std']:.2f}")
        logger.info(f"  Min:    {stats['min']:.2f}")
        logger.info(f"  25%:    {stats['25%']:.2f}")
        logger.info(f"  Median: {stats['50%']:.2f}")
        logger.info(f"  75%:    {stats['75%']:.2f}")
        logger.info(f"  Max:    {stats['max']:.2f}")
    
    logger.info("\nPM10 (µg/m³):")
    if df['pm10'].notna().any():
        stats = df['pm10'].describe()
        logger.info(f"  Count:  {stats['count']:,.0f}")
        logger.info(f"  Mean:   {stats['mean']:.2f}")
        logger.info(f"  Std:    {stats['std']:.2f}")
        logger.info(f"  Min:    {stats['min']:.2f}")
        logger.info(f"  25%:    {stats['25%']:.2f}")
        logger.info(f"  Median: {stats['50%']:.2f}")
        logger.info(f"  75%:    {stats['75%']:.2f}")
        logger.info(f"  Max:    {stats['max']:.2f}")
    
    # === WHO Air Quality Guidelines (2021) — CORRECT IMPLEMENTATION ===
    logger.info("\nWHO Air Quality Guidelines (2021):")
    logger.info("  PM2.5: 15 µg/m³ (24-hour mean), 5 µg/m³ (annual mean)")
    logger.info("  PM10:  45 µg/m³ (24-hour mean), 15 µg/m³ (annual mean)")

    # --- 24-hour rolling means (WHO definition) ---
    df["pm25_24h_mean"] = df["pm25"].rolling(window=24, min_periods=24).mean()
    df["pm10_24h_mean"] = df["pm10"].rolling(window=24, min_periods=24).mean()

    pm25_24h_exceed = (df["pm25_24h_mean"] > 15).sum()
    pm10_24h_exceed = (df["pm10_24h_mean"] > 45).sum()

    logger.info("\nPM2.5 Analysis (WHO-compliant):")
    logger.info(f"  Number of 24-hour periods exceeding 15 µg/m³: {pm25_24h_exceed}")

    logger.info("\nPM10 Analysis (WHO-compliant):")
    logger.info(f"  Number of 24-hour periods exceeding 45 µg/m³: {pm10_24h_exceed}")

    # --- Annual mean by calendar year (NOT overall mean) ---
    annual_pm25 = df.groupby(df["datetime"].dt.year)["pm25"].mean()
    annual_pm10 = df.groupby(df["datetime"].dt.year)["pm10"].mean()

    logger.info("\nAnnual mean concentrations by year:")
    for year in annual_pm25.index:
        logger.info(
            f"  {year}: PM2.5 = {annual_pm25[year]:.2f} µg/m³, "
            f"PM10 = {annual_pm10[year]:.2f} µg/m³"
        )

    logger.info("\n" + "=" * 80)

# ============================================================================
# 2. TEMPORAL FEATURE ENGINEERING
# ============================================================================


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add temporal features for time-based analysis.
    
    Args:
        df: Input DataFrame with datetime column
    
    Returns:
        DataFrame with added temporal columns: hour, day_of_week, month, year, week
    """
    df = df.copy()
    
    # Extract temporal components
    df['hour'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.dayofweek  # Monday=0, Sunday=6
    df['month'] = df['datetime'].dt.month
    df['year'] = df['datetime'].dt.year
    df['date'] = df['datetime'].dt.date
    df['week'] = df['datetime'].dt.isocalendar().week
    
    return df


# ============================================================================
# 3. TIME-BASED AGGREGATION AND ANALYSIS
# ============================================================================


def analyze_diurnal_patterns(
    df: pd.DataFrame,
    logger: logging.Logger
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Analyze hourly (diurnal) patterns in PM2.5 and PM10.
    
    Args:
        df: DataFrame with hour, pm25, pm10 columns
        logger: Logger instance
    
    Returns:
        Tuple of (pm25_hourly, pm10_hourly) DataFrames
    """
    logger.info("\nAnalyzing diurnal patterns...")
    
    # Aggregate by hour
    hourly_pm25 = df.groupby('hour')['pm25'].agg(['mean', 'std', 'count']).reset_index()
    hourly_pm10 = df.groupby('hour')['pm10'].agg(['mean', 'std', 'count']).reset_index()
    
    logger.info(f"  PM2.5 peak hour: {hourly_pm25.loc[hourly_pm25['mean'].idxmax(), 'hour']:.0f}:00 "
               f"({hourly_pm25['mean'].max():.2f} µg/m³)")
    logger.info(f"  PM2.5 lowest hour: {hourly_pm25.loc[hourly_pm25['mean'].idxmin(), 'hour']:.0f}:00 "
               f"({hourly_pm25['mean'].min():.2f} µg/m³)")
    
    return hourly_pm25, hourly_pm10


def analyze_weekly_patterns(
    df: pd.DataFrame,
    logger: logging.Logger
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Analyze weekly patterns in PM2.5 and PM10.
    
    Args:
        df: DataFrame with day_of_week, pm25, pm10 columns
        logger: Logger instance
    
    Returns:
        Tuple of (pm25_weekly, pm10_weekly) DataFrames
    """
    logger.info("\nAnalyzing weekly patterns...")
    
    # Aggregate by day of week
    weekly_pm25 = df.groupby('day_of_week')['pm25'].agg(['mean', 'std', 'count']).reset_index()
    weekly_pm10 = df.groupby('day_of_week')['pm10'].agg(['mean', 'std', 'count']).reset_index()
    
    day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    peak_day = day_names[int(weekly_pm25.loc[weekly_pm25['mean'].idxmax(), 'day_of_week'])]
    logger.info(f"  PM2.5 peak day: {peak_day} ({weekly_pm25['mean'].max():.2f} µg/m³)")
    
    return weekly_pm25, weekly_pm10


def analyze_seasonal_patterns(
    df: pd.DataFrame,
    logger: logging.Logger
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Analyze seasonal (monthly) patterns in PM2.5 and PM10.
    
    Args:
        df: DataFrame with month, pm25, pm10 columns
        logger: Logger instance
    
    Returns:
        Tuple of (pm25_monthly, pm10_monthly) DataFrames
    """
    logger.info("\nAnalyzing seasonal patterns...")
    
    # Aggregate by month
    monthly_pm25 = df.groupby('month')['pm25'].agg(['mean', 'std', 'count']).reset_index()
    monthly_pm10 = df.groupby('month')['pm10'].agg(['mean', 'std', 'count']).reset_index()
    
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    peak_month = month_names[int(monthly_pm25.loc[monthly_pm25['mean'].idxmax(), 'month']) - 1]
    logger.info(f"  PM2.5 peak month: {peak_month} ({monthly_pm25['mean'].max():.2f} µg/m³)")
    logger.info(f"  PM2.5 lowest month: "
               f"{month_names[int(monthly_pm25.loc[monthly_pm25['mean'].idxmin(), 'month']) - 1]} "
               f"({monthly_pm25['mean'].min():.2f} µg/m³)")
    
    return monthly_pm25, monthly_pm10


# ============================================================================
# 4. VISUALIZATION FUNCTIONS
# ============================================================================


def plot_diurnal_patterns(
    hourly_pm25: pd.DataFrame,
    hourly_pm10: pd.DataFrame,
    output_path: Path
) -> None:
    """
    Plot mean PM2.5 and PM10 by hour of day.
    
    Args:
        hourly_pm25: Aggregated PM2.5 data by hour
        hourly_pm10: Aggregated PM10 data by hour
        output_path: Path to save figure
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), dpi=FIG_DPI)
    
    # PM2.5 diurnal pattern
    ax1 = axes[0]
    ax1.plot(hourly_pm25['hour'], hourly_pm25['mean'], 
             marker='o', linewidth=2, markersize=6, color='#E63946', label='Mean')
    ax1.fill_between(
        hourly_pm25['hour'],
        hourly_pm25['mean'] - hourly_pm25['std'],
        hourly_pm25['mean'] + hourly_pm25['std'],
        alpha=0.3, color='#E63946', label='±1 SD'
    )
    ax1.axhline(y=15, color='orange', linestyle='--', linewidth=2, 
                label='WHO 24h Guideline (15 µg/m³)')
    ax1.set_xlabel('Hour of Day', fontsize=12, fontweight='bold')
    ax1.set_ylabel('PM2.5 (µg/m³)', fontsize=12, fontweight='bold')
    ax1.set_title(f'Diurnal Pattern of PM2.5 - {CITY}, {COUNTRY}', 
                  fontsize=14, fontweight='bold', pad=15)
    ax1.legend(loc='best', framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xticks(range(0, 24, 2))
    
    # PM10 diurnal pattern
    ax2 = axes[1]
    ax2.plot(hourly_pm10['hour'], hourly_pm10['mean'], 
             marker='s', linewidth=2, markersize=6, color='#457B9D', label='Mean')
    ax2.fill_between(
        hourly_pm10['hour'],
        hourly_pm10['mean'] - hourly_pm10['std'],
        hourly_pm10['mean'] + hourly_pm10['std'],
        alpha=0.3, color='#457B9D', label='±1 SD'
    )
    ax2.axhline(y=45, color='orange', linestyle='--', linewidth=2, 
                label='WHO 24h Guideline (45 µg/m³)')
    ax2.set_xlabel('Hour of Day', fontsize=12, fontweight='bold')
    ax2.set_ylabel('PM10 (µg/m³)', fontsize=12, fontweight='bold')
    ax2.set_title(f'Diurnal Pattern of PM10 - {CITY}, {COUNTRY}', 
                  fontsize=14, fontweight='bold', pad=15)
    ax2.legend(loc='best', framealpha=0.9)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_xticks(range(0, 24, 2))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=FIG_DPI, bbox_inches='tight')
    plt.close()


def plot_weekly_patterns(
    weekly_pm25: pd.DataFrame,
    weekly_pm10: pd.DataFrame,
    output_path: Path
) -> None:
    """
    Plot mean PM2.5 and PM10 by day of week.
    
    Args:
        weekly_pm25: Aggregated PM2.5 data by day of week
        weekly_pm10: Aggregated PM10 data by day of week
        output_path: Path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=FIG_DPI)
    
    day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    
    # PM2.5 weekly pattern
    ax1 = axes[0]
    bars1 = ax1.bar(weekly_pm25['day_of_week'], weekly_pm25['mean'], 
                    color='#E63946', alpha=0.7, edgecolor='black', linewidth=1.2)
    ax1.errorbar(weekly_pm25['day_of_week'], weekly_pm25['mean'], 
                 yerr=weekly_pm25['std'], fmt='none', ecolor='black', 
                 capsize=5, capthick=2)
    ax1.axhline(y=15, color='orange', linestyle='--', linewidth=2, 
                label='WHO 24h Guideline')
    ax1.set_xlabel('Day of Week', fontsize=12, fontweight='bold')
    ax1.set_ylabel('PM2.5 (µg/m³)', fontsize=12, fontweight='bold')
    ax1.set_title(f'Weekly Pattern of PM2.5 - {CITY}', 
                  fontsize=13, fontweight='bold', pad=15)
    ax1.set_xticks(range(7))
    ax1.set_xticklabels(day_names, rotation=0)
    ax1.legend(loc='best', framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    # PM10 weekly pattern
    ax2 = axes[1]
    bars2 = ax2.bar(weekly_pm10['day_of_week'], weekly_pm10['mean'], 
                    color='#457B9D', alpha=0.7, edgecolor='black', linewidth=1.2)
    ax2.errorbar(weekly_pm10['day_of_week'], weekly_pm10['mean'], 
                 yerr=weekly_pm10['std'], fmt='none', ecolor='black', 
                 capsize=5, capthick=2)
    ax2.axhline(y=45, color='orange', linestyle='--', linewidth=2, 
                label='WHO 24h Guideline')
    ax2.set_xlabel('Day of Week', fontsize=12, fontweight='bold')
    ax2.set_ylabel('PM10 (µg/m³)', fontsize=12, fontweight='bold')
    ax2.set_title(f'Weekly Pattern of PM10 - {CITY}', 
                  fontsize=13, fontweight='bold', pad=15)
    ax2.set_xticks(range(7))
    ax2.set_xticklabels(day_names, rotation=0)
    ax2.legend(loc='best', framealpha=0.9)
    ax2.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=FIG_DPI, bbox_inches='tight')
    plt.close()


def plot_seasonal_patterns(
    monthly_pm25: pd.DataFrame,
    monthly_pm10: pd.DataFrame,
    output_path: Path
) -> None:
    """
    Plot mean PM2.5 and PM10 by month.
    
    Args:
        monthly_pm25: Aggregated PM2.5 data by month
        monthly_pm10: Aggregated PM10 data by month
        output_path: Path to save figure
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), dpi=FIG_DPI)
    
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    # PM2.5 seasonal pattern
    ax1 = axes[0]
    bars1 = ax1.bar(monthly_pm25['month'], monthly_pm25['mean'], 
                    color='#E63946', alpha=0.7, edgecolor='black', linewidth=1.2)
    ax1.errorbar(monthly_pm25['month'], monthly_pm25['mean'], 
                 yerr=monthly_pm25['std'], fmt='none', ecolor='black', 
                 capsize=5, capthick=2)
    ax1.axhline(y=15, color='orange', linestyle='--', linewidth=2, 
                label='WHO 24h Guideline (15 µg/m³)')
    ax1.axhline(y=5, color='red', linestyle='--', linewidth=2, 
                label='WHO Annual Guideline (5 µg/m³)')
    ax1.set_xlabel('Month', fontsize=12, fontweight='bold')
    ax1.set_ylabel('PM2.5 (µg/m³)', fontsize=12, fontweight='bold')
    ax1.set_title(f'Seasonal Pattern of PM2.5 - {CITY}, {COUNTRY}', 
                  fontsize=14, fontweight='bold', pad=15)
    ax1.set_xticks(range(1, 13))
    ax1.set_xticklabels(month_names, rotation=45, ha='right')
    ax1.legend(loc='best', framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    # PM10 seasonal pattern
    ax2 = axes[1]
    bars2 = ax2.bar(monthly_pm10['month'], monthly_pm10['mean'], 
                    color='#457B9D', alpha=0.7, edgecolor='black', linewidth=1.2)
    ax2.errorbar(monthly_pm10['month'], monthly_pm10['mean'], 
                 yerr=monthly_pm10['std'], fmt='none', ecolor='black', 
                 capsize=5, capthick=2)
    ax2.axhline(y=45, color='orange', linestyle='--', linewidth=2, 
                label='WHO 24h Guideline (45 µg/m³)')
    ax2.axhline(y=15, color='red', linestyle='--', linewidth=2, 
                label='WHO Annual Guideline (15 µg/m³)')
    ax2.set_xlabel('Month', fontsize=12, fontweight='bold')
    ax2.set_ylabel('PM10 (µg/m³)', fontsize=12, fontweight='bold')
    ax2.set_title(f'Seasonal Pattern of PM10 - {CITY}, {COUNTRY}', 
                  fontsize=14, fontweight='bold', pad=15)
    ax2.set_xticks(range(1, 13))
    ax2.set_xticklabels(month_names, rotation=45, ha='right')
    ax2.legend(loc='best', framealpha=0.9)
    ax2.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=FIG_DPI, bbox_inches='tight')
    plt.close()


def plot_time_series(df: pd.DataFrame, output_path: Path) -> None:
    """
    Plot full time series of PM2.5 and PM10.
    
    Args:
        df: DataFrame with datetime, pm25, pm10 columns
        output_path: Path to save figure
    """
    fig, axes = plt.subplots(2, 1, figsize=(16, 8), dpi=FIG_DPI, sharex=True)
    
    # PM2.5 time series
    ax1 = axes[0]
    ax1.plot(df['datetime'], df['pm25'], linewidth=0.8, color='#E63946', alpha=0.7)
    ax1.axhline(y=15, color='orange', linestyle='--', linewidth=2, alpha=0.8,
                label='WHO 24h Guideline (15 µg/m³)')
    ax1.axhline(y=5, color='red', linestyle='--', linewidth=2, alpha=0.8,
                label='WHO Annual Guideline (5 µg/m³)')
    ax1.set_ylabel('PM2.5 (µg/m³)', fontsize=12, fontweight='bold')
    ax1.set_title(f'PM2.5 Time Series - {CITY}, {COUNTRY}', 
                  fontsize=14, fontweight='bold', pad=15)
    ax1.legend(loc='upper right', framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # PM10 time series
    ax2 = axes[1]
    ax2.plot(df['datetime'], df['pm10'], linewidth=0.8, color='#457B9D', alpha=0.7)
    ax2.axhline(y=45, color='orange', linestyle='--', linewidth=2, alpha=0.8,
                label='WHO 24h Guideline (45 µg/m³)')
    ax2.axhline(y=15, color='red', linestyle='--', linewidth=2, alpha=0.8,
                label='WHO Annual Guideline (15 µg/m³)')
    ax2.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax2.set_ylabel('PM10 (µg/m³)', fontsize=12, fontweight='bold')
    ax2.set_title(f'PM10 Time Series - {CITY}, {COUNTRY}', 
                  fontsize=14, fontweight='bold', pad=15)
    ax2.legend(loc='upper right', framealpha=0.9)
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    # Format x-axis
    fig.autofmt_xdate(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=FIG_DPI, bbox_inches='tight')
    plt.close()


def plot_distribution_analysis(df: pd.DataFrame, output_path: Path) -> None:
    """
    Plot distribution analysis with histograms and box plots.
    
    Args:
        df: DataFrame with pm25, pm10 columns
        output_path: Path to save figure
    """
    fig = plt.figure(figsize=(14, 10), dpi=FIG_DPI)
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # PM2.5 histogram
    ax1 = fig.add_subplot(gs[0, 0])
    pm25_data = df['pm25'].dropna()
    ax1.hist(pm25_data, bins=50, color='#E63946', alpha=0.7, edgecolor='black')
    ax1.axvline(pm25_data.mean(), color='blue', linestyle='--', linewidth=2, 
                label=f'Mean: {pm25_data.mean():.1f}')
    ax1.axvline(pm25_data.median(), color='green', linestyle='--', linewidth=2, 
                label=f'Median: {pm25_data.median():.1f}')
    ax1.set_xlabel('PM2.5 (µg/m³)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax1.set_title('PM2.5 Distribution', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # PM10 histogram
    ax2 = fig.add_subplot(gs[0, 1])
    pm10_data = df['pm10'].dropna()
    ax2.hist(pm10_data, bins=50, color='#457B9D', alpha=0.7, edgecolor='black')
    ax2.axvline(pm10_data.mean(), color='blue', linestyle='--', linewidth=2, 
                label=f'Mean: {pm10_data.mean():.1f}')
    ax2.axvline(pm10_data.median(), color='green', linestyle='--', linewidth=2, 
                label=f'Median: {pm10_data.median():.1f}')
    ax2.set_xlabel('PM10 (µg/m³)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax2.set_title('PM10 Distribution', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Box plots
    ax3 = fig.add_subplot(gs[1, :])
    box_data = [pm25_data, pm10_data]
    bp = ax3.boxplot(
        box_data,
        tick_labels=["PM2.5", "PM10"],
        patch_artist=True,
        notch=True,
        showmeans=True
    )
    bp['boxes'][0].set_facecolor('#E63946')
    bp['boxes'][1].set_facecolor('#457B9D')
    for box in bp['boxes']:
        box.set_alpha(0.7)
    ax3.set_ylabel('Concentration (µg/m³)', fontsize=11, fontweight='bold')
    ax3.set_title('Distribution Comparison (Box Plots)', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Q-Q plots for normality assessment
    from scipy import stats
    
    ax4 = fig.add_subplot(gs[2, 0])
    stats.probplot(pm25_data, dist="norm", plot=ax4)
    ax4.set_title('PM2.5 Q-Q Plot (Normal Distribution)', fontsize=11, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    ax5 = fig.add_subplot(gs[2, 1])
    stats.probplot(pm10_data, dist="norm", plot=ax5)
    ax5.set_title('PM10 Q-Q Plot (Normal Distribution)', fontsize=11, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    plt.savefig(output_path, dpi=FIG_DPI, bbox_inches='tight')
    plt.close()


# ============================================================================
# 5. EXTREME POLLUTION ANALYSIS
# ============================================================================


def analyze_extreme_pollution(df: pd.DataFrame, logger: logging.Logger) -> None:
    """
    Analyze extreme pollution events (top 1% highest values).
    
    Identifies and reports extreme pollution episodes, which may be caused by:
    - Winter residential heating (coal/biomass burning)
    - Atmospheric inversions trapping pollutants
    - Stagnant meteorological conditions
    - Regional biomass burning or industrial emissions
    - Cross-boundary transport from neighboring regions
    
    Args:
        df: DataFrame with datetime, pm25, pm10 columns
        logger: Logger instance
    """
    logger.info("\n" + "=" * 80)
    logger.info("EXTREME POLLUTION ANALYSIS")
    logger.info("=" * 80)
    
    # PM2.5 extreme events
    if df['pm25'].notna().any():
        pm25_threshold = df['pm25'].quantile(0.99)
        extreme_pm25 = df[df['pm25'] >= pm25_threshold].copy()
        extreme_pm25 = extreme_pm25.sort_values('pm25', ascending=False)
        
        logger.info(f"\nPM2.5 Extreme Events (Top 1%, threshold: {pm25_threshold:.2f} µg/m³):")
        logger.info(f"  Number of extreme hours: {len(extreme_pm25)}")
        logger.info(f"  Maximum value: {extreme_pm25['pm25'].max():.2f} µg/m³")
        logger.info(f"  Date of maximum: {extreme_pm25.iloc[0]['datetime']}")
        
        # Temporal distribution of extremes
        if 'month' in extreme_pm25.columns:
            month_dist = extreme_pm25['month'].value_counts().sort_index()
            logger.info("\n  Monthly distribution of extreme PM2.5 events:")
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            for month, count in month_dist.items():
                logger.info(f"    {month_names[month-1]}: {count} events")
        
        # Top 10 extreme events
        logger.info("\n  Top 10 PM2.5 extreme events:")
        for idx, row in extreme_pm25.head(10).iterrows():
            logger.info(f"    {row['datetime']}: {row['pm25']:.2f} µg/m³")
        
        # Scientific interpretation
        logger.info("\n  Potential causes of extreme PM2.5 events:")
        logger.info("    - Winter heating emissions (residential coal/biomass)")
        logger.info("    - Atmospheric inversion layers trapping pollutants")
        logger.info("    - Stagnant meteorological conditions (low wind speed)")
        logger.info("    - Regional biomass burning or agricultural fires")
        logger.info("    - Industrial emissions during unfavorable dispersion")
    
    # PM10 extreme events
    if df['pm10'].notna().any():
        pm10_threshold = df['pm10'].quantile(0.99)
        extreme_pm10 = df[df['pm10'] >= pm10_threshold].copy()
        extreme_pm10 = extreme_pm10.sort_values('pm10', ascending=False)
        
        logger.info(f"\nPM10 Extreme Events (Top 1%, threshold: {pm10_threshold:.2f} µg/m³):")
        logger.info(f"  Number of extreme hours: {len(extreme_pm10)}")
        logger.info(f"  Maximum value: {extreme_pm10['pm10'].max():.2f} µg/m³")
        logger.info(f"  Date of maximum: {extreme_pm10.iloc[0]['datetime']}")
        
        # Top 10 extreme events
        logger.info("\n  Top 10 PM10 extreme events:")
        for idx, row in extreme_pm10.head(10).iterrows():
            logger.info(f"    {row['datetime']}: {row['pm10']:.2f} µg/m³")
        
        logger.info("\n  Potential causes of extreme PM10 events:")
        logger.info("    - Dust storms or wind-blown soil/sand")
        logger.info("    - Construction and road dust resuspension")
        logger.info("    - Same factors as PM2.5 (includes coarse particles)")
    
    logger.info("\n" + "=" * 80)


# ============================================================================
# 6. CORRELATION ANALYSIS
# ============================================================================


def plot_correlation_analysis(df: pd.DataFrame, output_path: Path) -> None:
    """
    Plot correlation between PM2.5 and PM10.
    
    Args:
        df: DataFrame with pm25, pm10 columns
        output_path: Path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=FIG_DPI)
    
    # Scatter plot with regression line
    ax1 = axes[0]
    valid_data = df[['pm25', 'pm10']].dropna()
    
    if len(valid_data) > 0:
        ax1.scatter(valid_data['pm25'], valid_data['pm10'], 
                   alpha=0.3, s=10, color='#457B9D')
        
        # Add regression line
        from scipy.stats import linregress
        slope, intercept, r_value, p_value, std_err = linregress(
            valid_data['pm25'], valid_data['pm10']
        )
        x_line = np.linspace(valid_data['pm25'].min(), valid_data['pm25'].max(), 100)
        y_line = slope * x_line + intercept
        ax1.plot(x_line, y_line, 'r-', linewidth=2, 
                label=f'R² = {r_value**2:.3f}\ny = {slope:.2f}x + {intercept:.2f}')
        
        ax1.set_xlabel('PM2.5 (µg/m³)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('PM10 (µg/m³)', fontsize=12, fontweight='bold')
        ax1.set_title(f'PM2.5 vs PM10 Correlation - {CITY}', 
                     fontsize=13, fontweight='bold', pad=15)
        ax1.legend(loc='upper left', framealpha=0.9, fontsize=11)
        ax1.grid(True, alpha=0.3)
    
    # Correlation matrix heatmap
    ax2 = axes[1]
    corr_matrix = df[['pm25', 'pm10']].corr()
    im = ax2.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    
    # Add text annotations
    for i in range(len(corr_matrix)):
        for j in range(len(corr_matrix)):
            text = ax2.text(j, i, f'{corr_matrix.iloc[i, j]:.3f}',
                          ha="center", va="center", color="black", 
                          fontsize=14, fontweight='bold')
    
    ax2.set_xticks([0, 1])
    ax2.set_yticks([0, 1])
    ax2.set_xticklabels(['PM2.5', 'PM10'])
    ax2.set_yticklabels(['PM2.5', 'PM10'])
    ax2.set_title('Correlation Matrix', fontsize=13, fontweight='bold', pad=15)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label('Correlation Coefficient', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=FIG_DPI, bbox_inches='tight')
    plt.close()


# ============================================================================
# 7. MAIN EXECUTION
# ============================================================================


def main():
    """
    Main function to execute full exploratory data analysis.
    
    Performs comprehensive analysis suitable for a 30-35 page academic report,
    including descriptive statistics, temporal patterns, visualizations, and
    extreme event analysis.
    """
    # Ensure directories exist
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = setup_logger(LOGS_DIR / "exploratory_analysis.log")
    
    logger.info("=" * 80)
    logger.info(f"EXPLORATORY DATA ANALYSIS: {CITY} AIR QUALITY")
    logger.info("=" * 80)
    logger.info(f"Analysis date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load pollution data
    pollution_path = RAW_DATA_DIR / "pollution.csv"
    logger.info(f"\nLoading data from: {pollution_path}")
    
    if not pollution_path.exists():
        logger.error(f"Data file not found: {pollution_path}")
        logger.error("Please run data download first: python src/data_download.py")
        return 1
    
    # Load data using pd.read_csv directly
    pollution_df = pd.read_csv(pollution_path)
    logger.info(f"✓ Loaded {len(pollution_df):,} records")
    
    # Safe timezone handling
    logger.info("\nProcessing datetime with safe timezone handling...")

    # 1) Parse everything as UTC (fixes mixed timezones)
    pollution_df["datetime"] = pd.to_datetime(
        pollution_df["datetime"],
        errors="coerce",
        utc=True,
)

    # 2) Convert UTC -> project local timezone
    pollution_df["datetime"] = pollution_df["datetime"].dt.tz_convert(TIMEZONE)

    # Count initial NaT values
    initial_nat = pollution_df["datetime"].isna().sum()
    if initial_nat > 0:
        logger.warning(f"  Found {initial_nat} invalid datetime values (will be dropped)")

    # Drop rows with invalid datetime
    pollution_df = pollution_df.dropna(subset=["datetime"])

    # Drop rows with NaT datetime
    nat_count = pollution_df['datetime'].isna().sum()
    if nat_count > 0:
        pollution_df = pollution_df.dropna(subset=['datetime'])
        logger.warning(f"  Dropped {nat_count} rows with invalid datetime")
    
    logger.info(f"✓ Final dataset: {len(pollution_df):,} records")
    
    # ========================================================================
    # 1. Data Overview
    # ========================================================================
    
    print_data_overview(pollution_df, logger)
    
    # ========================================================================
    # 2. Add Temporal Features
    # ========================================================================
    
    logger.info("\nAdding temporal features...")
    pollution_df = add_temporal_features(pollution_df)
    logger.info("✓ Temporal features added: hour, day_of_week, month, year, date, week")
    
    # ========================================================================
    # 3. Time-Based Analysis
    # ========================================================================
    
    logger.info("\n" + "=" * 80)
    logger.info("TEMPORAL PATTERN ANALYSIS")
    logger.info("=" * 80)
    
    # Diurnal patterns
    hourly_pm25, hourly_pm10 = analyze_diurnal_patterns(pollution_df, logger)
    plot_diurnal_patterns(
        hourly_pm25, hourly_pm10,
        FIGURES_DIR / "diurnal_patterns.png"
    )
    logger.info("✓ Saved: diurnal_patterns.png")
    
    # Weekly patterns
    weekly_pm25, weekly_pm10 = analyze_weekly_patterns(pollution_df, logger)
    plot_weekly_patterns(
        weekly_pm25, weekly_pm10,
        FIGURES_DIR / "weekly_patterns.png"
    )
    logger.info("✓ Saved: weekly_patterns.png")
    
    # Seasonal patterns
    monthly_pm25, monthly_pm10 = analyze_seasonal_patterns(pollution_df, logger)
    plot_seasonal_patterns(
        monthly_pm25, monthly_pm10,
        FIGURES_DIR / "seasonal_patterns.png"
    )
    logger.info("✓ Saved: seasonal_patterns.png")
    
    # ========================================================================
    # 4. Time Series Visualization
    # ========================================================================
    
    logger.info("\n" + "=" * 80)
    logger.info("TIME SERIES VISUALIZATION")
    logger.info("=" * 80)
    
    plot_time_series(pollution_df, FIGURES_DIR / "time_series_full.png")
    logger.info("✓ Saved: time_series_full.png")
    
    # ========================================================================
    # 5. Distribution Analysis
    # ========================================================================
    
    logger.info("\n" + "=" * 80)
    logger.info("DISTRIBUTION ANALYSIS")
    logger.info("=" * 80)
    
    plot_distribution_analysis(pollution_df, FIGURES_DIR / "distribution_analysis.png")
    logger.info("✓ Saved: distribution_analysis.png")
    
    # ========================================================================
    # 6. Correlation Analysis
    # ========================================================================
    
    logger.info("\n" + "=" * 80)
    logger.info("CORRELATION ANALYSIS")
    logger.info("=" * 80)
    
    plot_correlation_analysis(pollution_df, FIGURES_DIR / "correlation_analysis.png")
    logger.info("✓ Saved: correlation_analysis.png")
    
    # ========================================================================
    # 7. Extreme Pollution Analysis
    # ========================================================================
    
    analyze_extreme_pollution(pollution_df, logger)
    
    # ========================================================================
    # Summary
    # ========================================================================
    
    logger.info("\n" + "=" * 80)
    logger.info("ANALYSIS COMPLETE")
    logger.info("=" * 80)
    logger.info(f"\nAll figures saved to: {FIGURES_DIR}")
    logger.info("\nGenerated figures:")
    logger.info("  1. diurnal_patterns.png - Hourly patterns")
    logger.info("  2. weekly_patterns.png - Day-of-week patterns")
    logger.info("  3. seasonal_patterns.png - Monthly patterns")
    logger.info("  4. time_series_full.png - Complete time series")
    logger.info("  5. distribution_analysis.png - Statistical distributions")
    logger.info("  6. correlation_analysis.png - PM2.5 vs PM10 correlation")
    logger.info("\nThese figures are publication-ready and suitable for")
    logger.info("inclusion in a 30-35 page academic research report.")
    logger.info("=" * 80)
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())