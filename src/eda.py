"""
Exploratory Data Analysis module for Almaty Air Quality project.

This module provides functions for data auditing and visualization,
designed to work both in scripts and Jupyter notebooks.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator

from src.config import (
    FIGURES_DIR,
    TABLES_DIR,
    RAW_DATA_DIR,
    LOGS_DIR,
    TIMEZONE,
)
from src.utils import setup_logging, save_df, load_df, safe_mkdir


# ============================================================================
# Data Audit Functions
# ============================================================================


def audit_time_series(
    df: pd.DataFrame,
    name: str,
    datetime_col: str = "datetime",
    save_path: Optional[Path] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Perform comprehensive audit of time series data.
    
    Checks data quality metrics including dtypes, missing values, duplicates,
    temporal coverage, and generates monthly coverage statistics.
    
    Args:
        df: DataFrame to audit.
        name: Name of the dataset (for logging and output).
        datetime_col: Name of datetime column.
        save_path: Optional directory to save audit tables.
    
    Returns:
        Tuple of (audit_table_df, coverage_table_df):
        - audit_table_df: Overall data quality metrics
        - coverage_table_df: Monthly coverage statistics
    
    Examples:
        >>> df = pd.read_csv("data/raw/openaq.csv")
        >>> audit_df, coverage_df = audit_time_series(df, "OpenAQ")
        >>> print(audit_df)
    """
    print(f"\n{'=' * 70}")
    print(f"Time Series Audit: {name}")
    print(f"{'=' * 70}\n")
    
    # ========================================================================
    # Overall Audit Metrics
    # ========================================================================
    
    audit_data = []
    
    # Basic shape
    audit_data.append({
        "Metric": "Total Rows",
        "Value": f"{len(df):,}",
        "Status": "✓" if len(df) > 0 else "✗",
    })
    
    audit_data.append({
        "Metric": "Total Columns",
        "Value": f"{len(df.columns)}",
        "Status": "✓",
    })
    
    # Datetime checks
    if datetime_col in df.columns:
        dt_series = pd.to_datetime(df[datetime_col], errors="coerce")
        
        # Check if timezone-aware
        is_tz_aware = dt_series.dt.tz is not None
        audit_data.append({
            "Metric": "Timezone Aware",
            "Value": "Yes" if is_tz_aware else "No",
            "Status": "✓" if is_tz_aware else "⚠",
        })
        
        # Date range
        min_date = dt_series.min()
        max_date = dt_series.max()
        audit_data.append({
            "Metric": "Date Range Start",
            "Value": str(min_date),
            "Status": "✓",
        })
        audit_data.append({
            "Metric": "Date Range End",
            "Value": str(max_date),
            "Status": "✓",
        })
        
        # Time span
        time_span = (max_date - min_date).total_seconds() / 86400  # days
        audit_data.append({
            "Metric": "Time Span (days)",
            "Value": f"{time_span:.1f}",
            "Status": "✓",
        })
        
        # Median timestep
        if len(dt_series) > 1:
            time_diffs = dt_series.sort_values().diff().dropna()
            median_step = time_diffs.median()
            median_hours = median_step.total_seconds() / 3600
            audit_data.append({
                "Metric": "Median Timestep (hours)",
                "Value": f"{median_hours:.2f}",
                "Status": "✓" if median_hours > 0 else "✗",
            })
        
        # Duplicates in datetime
        n_duplicates = dt_series.duplicated().sum()
        audit_data.append({
            "Metric": "Duplicate Timestamps",
            "Value": f"{n_duplicates:,}",
            "Status": "✓" if n_duplicates == 0 else "⚠",
        })
    else:
        audit_data.append({
            "Metric": "Datetime Column",
            "Value": f"'{datetime_col}' not found",
            "Status": "✗",
        })
    
    # Missing values per column
    missing_summary = df.isnull().sum()
    missing_pct = (missing_summary / len(df) * 100).round(2)
    
    for col in df.columns:
        if missing_summary[col] > 0:
            status = "✓" if missing_pct[col] < 5 else "⚠" if missing_pct[col] < 20 else "✗"
            audit_data.append({
                "Metric": f"Missing: {col}",
                "Value": f"{missing_pct[col]:.2f}%",
                "Status": status,
            })
    
    # Data types
    for col in df.columns:
        audit_data.append({
            "Metric": f"Dtype: {col}",
            "Value": str(df[col].dtype),
            "Status": "ℹ",
        })
    
    audit_table = pd.DataFrame(audit_data)
    
    print("Overall Audit:")
    print(audit_table.to_string(index=False))
    print()
    
    # ========================================================================
    # Monthly Coverage Statistics
    # ========================================================================
    
    coverage_data = []
    
    if datetime_col in df.columns:
        df_temp = df.copy()
        df_temp[datetime_col] = pd.to_datetime(df_temp[datetime_col], errors="coerce")
        df_temp["year_month"] = df_temp[datetime_col].dt.to_period("M")
        
        # Group by month
        for period, group in df_temp.groupby("year_month"):
            month_stats = {
                "Year-Month": str(period),
                "Total Rows": len(group),
            }
            
            # Add missing counts for key variables
            key_vars = ["pm25", "pm10", "temperature_2m", "relative_humidity_2m",
                       "precipitation", "wind_speed_10m"]
            
            for var in key_vars:
                if var in group.columns:
                    missing_count = group[var].isnull().sum()
                    missing_pct = (missing_count / len(group) * 100)
                    month_stats[f"{var}_missing_%"] = f"{missing_pct:.1f}"
            
            coverage_data.append(month_stats)
        
        coverage_table = pd.DataFrame(coverage_data)
        
        print("Monthly Coverage:")
        print(coverage_table.to_string(index=False))
        print()
    else:
        coverage_table = pd.DataFrame()
    
    # ========================================================================
    # Save to files if path provided
    # ========================================================================
    
    if save_path is not None:
        safe_mkdir(save_path)
        
        audit_path = save_path / f"audit_{name.lower().replace(' ', '_')}.csv"
        audit_table.to_csv(audit_path, index=False)
        print(f"Saved audit table to: {audit_path}")
        
        if not coverage_table.empty:
            coverage_path = save_path / f"coverage_{name.lower().replace(' ', '_')}.csv"
            coverage_table.to_csv(coverage_path, index=False)
            print(f"Saved coverage table to: {coverage_path}")
    
    print(f"{'=' * 70}\n")
    
    return audit_table, coverage_table


# ============================================================================
# Visualization Functions
# ============================================================================


def plot_daily_means(
    df: pd.DataFrame,
    value_col: str,
    out_png_path: Path,
    title: str,
    y_label: str,
    datetime_col: str = "datetime",
    ref_lines: Optional[Dict[str, float]] = None,
    figsize: Tuple[int, int] = (12, 5),
    dpi: int = 300,
) -> None:
    """
    Create publication-quality daily mean time series plot.
    
    Args:
        df: DataFrame with datetime and value columns.
        value_col: Name of column containing values to plot.
        out_png_path: Output path for PNG file.
        title: Plot title.
        y_label: Y-axis label.
        datetime_col: Name of datetime column.
        ref_lines: Optional dict of reference line labels and values
                  (e.g., {"WHO Guideline": 15.0}).
        figsize: Figure size as (width, height).
        dpi: Resolution for output image.
    
    Examples:
        >>> plot_daily_means(
        ...     df, "pm25", Path("reports/figures/pm25_daily.png"),
        ...     "PM2.5 Daily Mean - Almaty", "PM2.5 (µg/m³)",
        ...     ref_lines={"WHO Guideline": 15.0}
        ... )
    """
    # Prepare data
    df_plot = df[[datetime_col, value_col]].copy()
    df_plot[datetime_col] = pd.to_datetime(df_plot[datetime_col])
    df_plot = df_plot.dropna(subset=[value_col])
    
    if len(df_plot) == 0:
        print(f"Warning: No data to plot for {value_col}")
        return
    
    # Calculate daily means
    df_plot["date"] = df_plot[datetime_col].dt.date
    daily_means = df_plot.groupby("date")[value_col].mean().reset_index()
    daily_means["date"] = pd.to_datetime(daily_means["date"])
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # Plot daily means
    ax.plot(
        daily_means["date"],
        daily_means[value_col],
        color="#2E86AB",
        linewidth=1.5,
        alpha=0.8,
        label="Daily Mean",
    )
    
    # Add reference lines if provided
    if ref_lines:
        colors = ["#E63946", "#F77F00", "#06A77D"]
        for idx, (label, value) in enumerate(ref_lines.items()):
            color = colors[idx % len(colors)]
            ax.axhline(
                y=value,
                color=color,
                linestyle="--",
                linewidth=2,
                alpha=0.7,
                label=label,
            )
    
    # Styling
    ax.set_xlabel("Date", fontsize=12, fontweight="bold")
    ax.set_ylabel(y_label, fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Legend
    if ref_lines:
        ax.legend(loc="best", framealpha=0.9, fontsize=10)
    
    # Format x-axis
    fig.autofmt_xdate(rotation=45, ha="right")
    
    # Tight layout
    plt.tight_layout()
    
    # Save
    safe_mkdir(out_png_path.parent)
    plt.savefig(out_png_path, dpi=dpi, bbox_inches="tight")
    print(f"Saved plot to: {out_png_path}")
    
    plt.close()


def plot_seasonality_heatmap(
    df: pd.DataFrame,
    value_col: str,
    out_png_path: Path,
    datetime_col: str = "datetime",
    title: Optional[str] = None,
    cmap: str = "RdYlBu_r",
    figsize: Tuple[int, int] = (12, 6),
    dpi: int = 300,
) -> None:
    """
    Create seasonality heatmap showing hour-of-day vs month-of-year patterns.
    
    Args:
        df: DataFrame with datetime and value columns.
        value_col: Name of column containing values.
        out_png_path: Output path for PNG file.
        datetime_col: Name of datetime column.
        title: Plot title. If None, auto-generated from value_col.
        cmap: Colormap name.
        figsize: Figure size as (width, height).
        dpi: Resolution for output image.
    
    Examples:
        >>> plot_seasonality_heatmap(
        ...     df, "pm25", Path("reports/figures/pm25_seasonality.png")
        ... )
    """
    # Prepare data
    df_plot = df[[datetime_col, value_col]].copy()
    df_plot[datetime_col] = pd.to_datetime(df_plot[datetime_col])
    df_plot = df_plot.dropna(subset=[value_col])
    
    if len(df_plot) == 0:
        print(f"Warning: No data to plot for {value_col}")
        return
    
    # Extract hour and month
    df_plot["hour"] = df_plot[datetime_col].dt.hour
    df_plot["month"] = df_plot[datetime_col].dt.month
    
    # Calculate mean values for each hour-month combination
    pivot_data = df_plot.groupby(["hour", "month"])[value_col].mean().reset_index()
    heatmap_data = pivot_data.pivot(index="hour", columns="month", values=value_col)
    
    # Ensure all hours and months are present
    heatmap_data = heatmap_data.reindex(range(24), axis=0)
    heatmap_data = heatmap_data.reindex(range(1, 13), axis=1)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # Create heatmap
    im = ax.imshow(
        heatmap_data.values,
        aspect="auto",
        cmap=cmap,
        interpolation="nearest",
    )
    
    # Set ticks
    ax.set_xticks(range(12))
    ax.set_yticks(range(24))
    
    # Set tick labels
    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    ax.set_xticklabels(month_labels, fontsize=10)
    ax.set_yticklabels(range(24), fontsize=10)
    
    # Labels
    ax.set_xlabel("Month", fontsize=12, fontweight="bold")
    ax.set_ylabel("Hour of Day", fontsize=12, fontweight="bold")
    
    # Title
    if title is None:
        title = f"{value_col.replace('_', ' ').title()} - Hourly Seasonality Pattern"
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label(value_col.replace("_", " ").title(), fontsize=11, fontweight="bold")
    cbar.ax.tick_params(labelsize=10)
    
    # Add values to cells (optional, only if not too many)
    if heatmap_data.shape[0] * heatmap_data.shape[1] <= 300:
        for i in range(heatmap_data.shape[0]):
            for j in range(heatmap_data.shape[1]):
                value = heatmap_data.iloc[i, j]
                if not np.isnan(value):
                    # Choose text color based on background
                    text_color = "white" if value > heatmap_data.values[~np.isnan(heatmap_data.values)].mean() else "black"
                    ax.text(
                        j, i, f"{value:.1f}",
                        ha="center", va="center",
                        color=text_color, fontsize=7, alpha=0.7,
                    )
    
    # Tight layout
    plt.tight_layout()
    
    # Save
    safe_mkdir(out_png_path.parent)
    plt.savefig(out_png_path, dpi=dpi, bbox_inches="tight")
    print(f"Saved plot to: {out_png_path}")
    
    plt.close()


def plot_correlation_heatmap(
    df: pd.DataFrame,
    out_png_path: Path,
    title: str = "Variable Correlation Matrix",
    figsize: Tuple[int, int] = (10, 8),
    dpi: int = 300,
) -> None:
    """
    Create correlation heatmap for numeric variables.
    
    Args:
        df: DataFrame with numeric columns.
        out_png_path: Output path for PNG file.
        title: Plot title.
        figsize: Figure size as (width, height).
        dpi: Resolution for output image.
    """
    # Select only numeric columns (exclude datetime)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) < 2:
        print("Warning: Need at least 2 numeric columns for correlation")
        return
    
    # Calculate correlation
    corr_matrix = df[numeric_cols].corr()
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # Create heatmap
    im = ax.imshow(
        corr_matrix.values,
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        aspect="auto",
    )
    
    # Set ticks
    ax.set_xticks(range(len(numeric_cols)))
    ax.set_yticks(range(len(numeric_cols)))
    
    # Set tick labels
    labels = [col.replace("_", " ").title() for col in numeric_cols]
    ax.set_xticklabels(labels, fontsize=9, rotation=45, ha="right")
    ax.set_yticklabels(labels, fontsize=9)
    
    # Title
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("Correlation Coefficient", fontsize=11, fontweight="bold")
    cbar.ax.tick_params(labelsize=10)
    
    # Add correlation values
    for i in range(len(numeric_cols)):
        for j in range(len(numeric_cols)):
            value = corr_matrix.iloc[i, j]
            text_color = "white" if abs(value) > 0.5 else "black"
            ax.text(
                j, i, f"{value:.2f}",
                ha="center", va="center",
                color=text_color, fontsize=8, fontweight="bold",
            )
    
    # Tight layout
    plt.tight_layout()
    
    # Save
    safe_mkdir(out_png_path.parent)
    plt.savefig(out_png_path, dpi=dpi, bbox_inches="tight")
    print(f"Saved plot to: {out_png_path}")
    
    plt.close()


def plot_distribution_comparison(
    df: pd.DataFrame,
    value_cols: List[str],
    out_png_path: Path,
    title: str = "Distribution Comparison",
    figsize: Tuple[int, int] = (14, 6),
    dpi: int = 300,
) -> None:
    """
    Create side-by-side box plots and histograms for multiple variables.
    
    Args:
        df: DataFrame with value columns.
        value_cols: List of column names to compare.
        out_png_path: Output path for PNG file.
        title: Plot title.
        figsize: Figure size as (width, height).
        dpi: Resolution for output image.
    """
    # Filter to available columns
    available_cols = [col for col in value_cols if col in df.columns]
    
    if len(available_cols) == 0:
        print("Warning: No specified columns found in DataFrame")
        return
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=dpi)
    
    # Box plot
    ax1 = axes[0]
    data_for_box = [df[col].dropna().values for col in available_cols]
    labels_for_box = [col.replace("_", " ").title() for col in available_cols]
    
    bp = ax1.boxplot(
        data_for_box,
        labels=labels_for_box,
        patch_artist=True,
        notch=True,
        showmeans=True,
    )
    
    # Color box plots
    colors = plt.cm.Set3(np.linspace(0, 1, len(available_cols)))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax1.set_ylabel("Value", fontsize=11, fontweight="bold")
    ax1.set_title("Box Plot Comparison", fontsize=12, fontweight="bold")
    ax1.grid(True, alpha=0.3, axis="y")
    ax1.tick_params(axis="x", rotation=45)
    
    # Histogram
    ax2 = axes[1]
    for col, color in zip(available_cols, colors):
        data = df[col].dropna()
        if len(data) > 0:
            ax2.hist(
                data,
                bins=30,
                alpha=0.6,
                label=col.replace("_", " ").title(),
                color=color,
                edgecolor="black",
                linewidth=0.5,
            )
    
    ax2.set_xlabel("Value", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Frequency", fontsize=11, fontweight="bold")
    ax2.set_title("Distribution Histograms", fontsize=12, fontweight="bold")
    ax2.legend(loc="best", fontsize=9)
    ax2.grid(True, alpha=0.3, axis="y")
    
    # Overall title
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    
    # Tight layout
    plt.tight_layout()
    
    # Save
    safe_mkdir(out_png_path.parent)
    plt.savefig(out_png_path, dpi=dpi, bbox_inches="tight")
    print(f"Saved plot to: {out_png_path}")
    
    plt.close()


# ============================================================================
# Main Script
# ============================================================================


def main():
    """
    Main function to run EDA on all raw datasets.
    
    Performs data audits and creates visualizations for:
    - OpenAQ air quality data (PM2.5, PM10)
    - Open-Meteo weather data (temperature, wind, etc.)
    """
    # Setup logging
    logger = setup_logging(LOGS_DIR / "step2_eda.log")
    
    logger.info("=" * 70)
    logger.info("Exploratory Data Analysis - Almaty Air Quality")
    logger.info("=" * 70)
    
    # ========================================================================
    # Load Data
    # ========================================================================
    
    logger.info("\nLoading datasets...")
    
    datasets = {}
    
    # Load OpenAQ data
    openaq_path = RAW_DATA_DIR / "openaq.csv"
    if openaq_path.exists():
        df_aq = load_df(openaq_path, logger=logger)
        datasets["openaq"] = df_aq
        logger.info(f"Loaded OpenAQ: {df_aq.shape}")
    else:
        logger.warning(f"OpenAQ file not found: {openaq_path}")
    
    # Load Open-Meteo data
    openmeteo_path = RAW_DATA_DIR / "openmeteo.csv"
    if openmeteo_path.exists():
        df_weather = load_df(openmeteo_path, logger=logger)
        datasets["openmeteo"] = df_weather
        logger.info(f"Loaded Open-Meteo: {df_weather.shape}")
    else:
        logger.warning(f"Open-Meteo file not found: {openmeteo_path}")
    
    if not datasets:
        logger.error("No datasets found. Run data_download.py first.")
        return 1
    
    # ========================================================================
    # Data Audits
    # ========================================================================
    
    logger.info("\n" + "=" * 70)
    logger.info("Running Data Audits")
    logger.info("=" * 70)
    
    for name, df in datasets.items():
        audit_table, coverage_table = audit_time_series(
            df,
            name=name.title(),
            datetime_col="datetime",
            save_path=TABLES_DIR,
        )
    
    # ========================================================================
    # Air Quality Visualizations
    # ========================================================================
    
    if "openaq" in datasets:
        logger.info("\n" + "=" * 70)
        logger.info("Creating Air Quality Visualizations")
        logger.info("=" * 70)
        
        df_aq = datasets["openaq"]
        
        # Pivot data for easier plotting (one row per datetime)
        # Aggregate by datetime and parameter
        if "parameter" in df_aq.columns:
            df_aq_pivot = df_aq.pivot_table(
                index="datetime",
                columns="parameter",
                values="value",
                aggfunc="mean",
            ).reset_index()
        else:
            df_aq_pivot = df_aq
        
        # PM2.5 plots
        if "pm25" in df_aq_pivot.columns:
            logger.info("\nCreating PM2.5 visualizations...")
            
            # Daily means
            plot_daily_means(
                df_aq_pivot,
                value_col="pm25",
                out_png_path=FIGURES_DIR / "pm25_daily_means.png",
                title="PM2.5 Daily Mean Concentrations - Almaty",
                y_label="PM2.5 (µg/m³)",
                ref_lines={
                    "WHO 24h Guideline": 15.0,
                    "WHO Annual Guideline": 5.0,
                },
            )
            
            # Seasonality heatmap
            plot_seasonality_heatmap(
                df_aq_pivot,
                value_col="pm25",
                out_png_path=FIGURES_DIR / "pm25_seasonality.png",
                title="PM2.5 Hourly Seasonality Pattern - Almaty",
            )
        
        # PM10 plots
        if "pm10" in df_aq_pivot.columns:
            logger.info("\nCreating PM10 visualizations...")
            
            # Daily means
            plot_daily_means(
                df_aq_pivot,
                value_col="pm10",
                out_png_path=FIGURES_DIR / "pm10_daily_means.png",
                title="PM10 Daily Mean Concentrations - Almaty",
                y_label="PM10 (µg/m³)",
                ref_lines={
                    "WHO 24h Guideline": 45.0,
                    "WHO Annual Guideline": 15.0,
                },
            )
            
            # Seasonality heatmap
            plot_seasonality_heatmap(
                df_aq_pivot,
                value_col="pm10",
                out_png_path=FIGURES_DIR / "pm10_seasonality.png",
                title="PM10 Hourly Seasonality Pattern - Almaty",
            )
        
        # Distribution comparison
        pm_cols = [col for col in df_aq_pivot.columns if col in ["pm25", "pm10"]]
        if pm_cols:
            plot_distribution_comparison(
                df_aq_pivot,
                value_cols=pm_cols,
                out_png_path=FIGURES_DIR / "pm_distributions.png",
                title="PM2.5 and PM10 Distribution Comparison - Almaty",
            )
    
    # ========================================================================
    # Weather Visualizations
    # ========================================================================
    
    if "openmeteo" in datasets:
        logger.info("\n" + "=" * 70)
        logger.info("Creating Weather Visualizations")
        logger.info("=" * 70)
        
        df_weather = datasets["openmeteo"]
        
        # Temperature plots
        if "temperature_2m" in df_weather.columns:
            logger.info("\nCreating temperature visualizations...")
            
            plot_daily_means(
                df_weather,
                value_col="temperature_2m",
                out_png_path=FIGURES_DIR / "temperature_daily_means.png",
                title="Daily Mean Temperature - Almaty",
                y_label="Temperature (°C)",
            )
            
            plot_seasonality_heatmap(
                df_weather,
                value_col="temperature_2m",
                out_png_path=FIGURES_DIR / "temperature_seasonality.png",
                title="Temperature Hourly Seasonality Pattern - Almaty",
                cmap="RdYlBu_r",
            )
        
        # Wind speed plots
        if "wind_speed_10m" in df_weather.columns:
            logger.info("\nCreating wind speed visualizations...")
            
            plot_daily_means(
                df_weather,
                value_col="wind_speed_10m",
                out_png_path=FIGURES_DIR / "wind_speed_daily_means.png",
                title="Daily Mean Wind Speed - Almaty",
                y_label="Wind Speed (m/s)",
            )
            
            plot_seasonality_heatmap(
                df_weather,
                value_col="wind_speed_10m",
                out_png_path=FIGURES_DIR / "wind_speed_seasonality.png",
                title="Wind Speed Hourly Seasonality Pattern - Almaty",
                cmap="YlGnBu",
            )
        
        # Correlation heatmap for weather variables
        weather_vars = [
            "temperature_2m", "relative_humidity_2m", "precipitation",
            "surface_pressure", "wind_speed_10m", "wind_direction_10m",
        ]
        available_weather_vars = [col for col in weather_vars if col in df_weather.columns]
        
        if len(available_weather_vars) >= 2:
            logger.info("\nCreating weather correlation heatmap...")
            plot_correlation_heatmap(
                df_weather[["datetime"] + available_weather_vars],
                out_png_path=FIGURES_DIR / "weather_correlation.png",
                title="Weather Variables Correlation Matrix - Almaty",
            )
    
    # ========================================================================
    # Summary
    # ========================================================================
    
    logger.info("\n" + "=" * 70)
    logger.info("EDA Complete!")
    logger.info("=" * 70)
    logger.info(f"\nOutputs saved to:")
    logger.info(f"  Figures: {FIGURES_DIR}")
    logger.info(f"  Tables: {TABLES_DIR}")
    logger.info("=" * 70)
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())