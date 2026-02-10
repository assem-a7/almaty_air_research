# Almaty Air Quality Forecasting & Analysis

A production-quality research project for analyzing and forecasting air quality in Almaty, Kazakhstan.

## 📋 Project Overview

This project provides a comprehensive framework for:
- Collecting air quality data for Almaty
- Exploratory data analysis and visualization
- Time series forecasting of air quality metrics
- Generating reports and insights

**Location:** Almaty, Kazakhstan (43.2220°N, 76.8512°E)  
**Time Period:** 2024-01-01 to 2025-12-31  
**Timezone:** Asia/Almaty

## 🚀 Quick Start

### 1. Setup

First, create the project structure:

```bash
python setup.py
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Notebooks in Order

Execute the notebooks sequentially:

1. **`01_data_collection.ipynb`** - Collect raw air quality data
   - Fetch data from APIs or other sources
   - Save to `data/raw/`

2. **`02_data_cleaning.ipynb`** - Clean and preprocess data
   - Handle missing values
   - Remove outliers
   - Save to `data/interim/`

3. **`03_exploratory_analysis.ipynb`** - Explore patterns and trends
   - Visualize temporal patterns
   - Analyze correlations
   - Generate figures in `reports/figures/`

4. **`04_feature_engineering.ipynb`** - Create features for modeling
   - Generate lag features
   - Create rolling statistics
   - Save to `data/processed/`

5. **`05_modeling.ipynb`** - Build and train forecasting models
   - Train time series models
   - Evaluate performance
   - Save model artifacts

6. **`06_reporting.ipynb`** - Generate final reports
   - Create summary tables
   - Generate final visualizations
   - Export reports

## 📁 Project Structure

```
almaty-air-quality/
├── data/
│   ├── raw/              # Original, immutable data
│   ├── interim/          # Intermediate cleaned data
│   └── processed/        # Final feature-engineered data
├── reports/
│   ├── figures/          # Generated plots and visualizations
│   └── tables/           # Summary tables and statistics
├── logs/                 # Application logs
├── src/
│   ├── config.py         # Configuration parameters
│   └── utils.py          # Utility functions
├── notebooks/            # Jupyter notebooks (run in order)
├── setup.py              # Project setup script
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## ⚙️ Configuration

All configuration parameters are centralized in `src/config.py`:

- **Location settings:** City, country, coordinates, timezone
- **Date ranges:** Start and end dates for analysis
- **Paths:** All directory paths as `pathlib.Path` objects
- **Model parameters:** Random seed, train/test split, CV folds
- **Retry settings:** API retry logic and timeouts

To modify settings, edit `src/config.py` directly.

## 🛠️ Utility Functions

The `src/utils.py` module provides:

- **`setup_logging()`** - Configure logging to file and console
- **`safe_mkdir()`** - Create directories safely
- **`save_df()`** - Save DataFrames with logging and auto-directory creation
- **`load_df()`** - Load DataFrames with logging
- **`validate_date_range()`** - Validate date strings
- **`get_data_info()`** - Get comprehensive DataFrame information

## 📊 Example Usage

```python
from pathlib import Path
from src.config import LOGS_DIR, RAW_DATA_DIR, START_DATE, END_DATE
from src.utils import setup_logging, save_df
import pandas as pd

# Setup logging
logger = setup_logging(LOGS_DIR / "data_collection.log")

# Your analysis code
logger.info(f"Starting data collection for {START_DATE} to {END_DATE}")

# ... fetch data ...

df = pd.DataFrame({"date": [...], "pm25": [...]})
save_df(df, RAW_DATA_DIR / "air_quality_almaty_raw.csv", logger=logger)
```

## 📝 Logging

Logs are automatically created in the `logs/` directory with:
- Timestamps for all events
- INFO level for console output
- DEBUG level for file output
- Automatic log rotation (configure as needed)

## 🔬 Research Workflow

1. **Data Collection** → Gather raw air quality measurements
2. **Data Cleaning** → Handle missing data and outliers
3. **EDA** → Understand patterns, seasonality, trends
4. **Feature Engineering** → Create predictive features
5. **Modeling** → Build and validate forecast models
6. **Reporting** → Generate insights and visualizations

## 📦 Dependencies

Key packages (add to `requirements.txt`):
- `pandas` - Data manipulation
- `numpy` - Numerical operations
- `matplotlib` / `seaborn` - Visualization
- `scikit-learn` - Machine learning
- `statsmodels` - Time series analysis
- `jupyter` - Notebook interface

## 🤝 Contributing

When adding new code:
1. Use type hints for all functions
2. Include comprehensive docstrings
3. Log important operations
4. Save outputs to appropriate directories
5. Update this README if adding new notebooks

## 📄 License

[Add your license here]

## 👤 Author

[Add your information here]

---

**Last Updated:** 2026-02-05