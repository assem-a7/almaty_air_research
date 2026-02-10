"""
Data download module for Almaty Air Quality Forecasting & Analysis.

This module implements a plugin-style architecture for downloading data from
multiple sources. It provides an abstract interface and concrete implementations
for OpenAQ air quality data and Open-Meteo weather data.
"""

import json
import logging
import time
import numpy as np
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.config import (
    CITY,
    COUNTRY,
    LATITUDE,
    LONGITUDE,
    TIMEZONE,
    START_DATE,
    END_DATE,
    RAW_DATA_DIR,
    LOGS_DIR,
    MAX_RETRIES,
    RETRY_DELAY,
    TIMEOUT,
)
from src.utils import setup_logging, save_df, safe_mkdir


# ============================================================================
# Abstract Base Class
# ============================================================================


class DataSource(ABC):
    """
    Abstract base class for data sources.
    
    All data sources must implement fetch() and metadata() methods.
    Provides common functionality for HTTP requests with retry logic.
    """
    
    def __init__(self, logger: Optional[Any] = None):
        """
        Initialize data source.
        
        Args:
            logger: Logger instance for output.
        """
        self.logger = logger
        self.session = self._create_session()
    
    def _create_session(self) -> requests.Session:
        """
        Create a requests session with retry logic.
        
        Returns:
            Configured requests.Session instance.
        """
        session = requests.Session()
        
        # Configure retry strategy
        retry_strategy = Retry(
            total=MAX_RETRIES,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"],
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        return session
    
    def _log(self, message: str, level: str = "info") -> None:
        """
        Log a message using the logger or print to console.
        
        Args:
            message: Message to log.
            level: Log level (info, warning, error, debug).
        """
        if self.logger:
            getattr(self.logger, level)(message)
        else:
            print(f"[{level.upper()}] {message}")
    
    @abstractmethod
    def fetch(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch data for the specified date range.
        
        Args:
            start_date: Start date in YYYY-MM-DD format.
            end_date: End date in YYYY-MM-DD format.
        
        Returns:
            DataFrame with datetime column (timezone-aware) and value columns.
        """
        pass
    
    @abstractmethod
    def metadata(self) -> Dict[str, Any]:
        """
        Get metadata about the data source.
        
        Returns:
            Dictionary containing source information.
        """
        pass


# ============================================================================
# OpenAQ Air Quality Data Source
# ============================================================================


class OpenAQMeasurementsSource(DataSource):
    """
    Data source for OpenAQ air quality measurements.
    
    Fetches PM2.5 and PM10 measurements with pagination, retries,
    and respectful rate limiting.
    """
    
    BASE_URL = "https://api.openaq.org/v2/measurements"
    PARAMETERS = ["pm25", "pm10"]
    PAGE_LIMIT = 1000  # Max results per page
    SLEEP_SECONDS = 1  # Respectful delay between requests
    
    def __init__(self, logger: Optional[Any] = None):
        """
        Initialize OpenAQ data source.
        
        Args:
            logger: Logger instance for output.
        """
        super().__init__(logger)
        self.coordinates = (LATITUDE, LONGITUDE)
        self.radius = 25000  # 25km radius around Almaty center
    
    def fetch(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch OpenAQ measurements for Almaty.
        
        Implements pagination to handle large datasets and includes
        rate limiting to respect API guidelines.
        
        Args:
            start_date: Start date in YYYY-MM-DD format.
            end_date: End date in YYYY-MM-DD format.
        
        Returns:
            DataFrame with columns: datetime, parameter, value, unit,
            location, city, country, latitude, longitude.
        """
        self._log(f"Fetching OpenAQ data from {start_date} to {end_date}")
        
        all_measurements: List[Dict] = []
        
        # Convert dates to datetime for chunking
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        # Chunk by month to avoid overwhelming the API
        current_start = start_dt
        while current_start < end_dt:
            current_end = min(current_start + timedelta(days=30), end_dt)
            
            self._log(
                f"Fetching chunk: {current_start.date()} to {current_end.date()}",
                level="debug",
            )
            
            chunk_data = self._fetch_chunk(
                current_start.strftime("%Y-%m-%d"),
                current_end.strftime("%Y-%m-%d"),
            )
            all_measurements.extend(chunk_data)
            
            current_start = current_end + timedelta(days=1)
        
        if not all_measurements:
            self._log("No data retrieved from OpenAQ", level="warning")
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(all_measurements)
        
        # Process datetime with timezone awareness
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        df["datetime"] = df["datetime"].dt.tz_convert(TIMEZONE)
        
        # Sort by datetime
        df = df.sort_values("datetime").reset_index(drop=True)
        
        self._log(f"Retrieved {len(df)} measurements from OpenAQ")
        self._log(f"Parameters: {df['parameter'].unique().tolist()}")
        self._log(f"Locations: {df['location'].nunique()}")
        
        return df
    
    def _fetch_chunk(self, start_date: str, end_date: str) -> List[Dict]:
        """
        Fetch a chunk of data with pagination.
        
        Args:
            start_date: Chunk start date.
            end_date: Chunk end date.
        
        Returns:
            List of measurement dictionaries.
        """
        measurements: List[Dict] = []
        page = 1
        
        while True:
            params = {
                "coordinates": f"{LATITUDE},{LONGITUDE}",
                "radius": self.radius,
                "date_from": f"{start_date}T00:00:00Z",
                "date_to": f"{end_date}T23:59:59Z",
                "parameter": ",".join(self.PARAMETERS),
                "limit": self.PAGE_LIMIT,
                "page": page,
                "order_by": "datetime",
            }
            
            try:
                response = self.session.get(
                    self.BASE_URL,
                    params=params,
                    timeout=TIMEOUT,
                )
                response.raise_for_status()
                data = response.json()
                
                results = data.get("results", [])
                if not results:
                    break
                
                # Extract relevant fields
                for result in results:
                    measurement = {
                        "datetime": result.get("date", {}).get("utc"),
                        "parameter": result.get("parameter"),
                        "value": result.get("value"),
                        "unit": result.get("unit"),
                        "location": result.get("location"),
                        "city": result.get("city"),
                        "country": result.get("country"),
                        "latitude": result.get("coordinates", {}).get("latitude"),
                        "longitude": result.get("coordinates", {}).get("longitude"),
                    }
                    measurements.append(measurement)
                
                self._log(
                    f"Page {page}: Retrieved {len(results)} measurements",
                    level="debug",
                )
                
                # Check if we've reached the end
                if len(results) < self.PAGE_LIMIT:
                    break
                
                page += 1
                
                # Respectful rate limiting
                time.sleep(self.SLEEP_SECONDS)
                
            except requests.exceptions.RequestException as e:
                self._log(f"Error fetching page {page}: {e}", level="error")
                time.sleep(RETRY_DELAY)
                continue
            except Exception as e:
                self._log(f"Unexpected error on page {page}: {e}", level="error")
                break
        
        return measurements
    
    def metadata(self) -> Dict[str, Any]:
        """
        Get metadata about OpenAQ data source.
        
        Returns:
            Dictionary with source information.
        """
        return {
            "source_name": "OpenAQ",
            "source_type": "Air Quality",
            "url": self.BASE_URL,
            "parameters": self.PARAMETERS,
            "location": {
                "city": CITY,
                "country": COUNTRY,
                "latitude": LATITUDE,
                "longitude": LONGITUDE,
                "radius_meters": self.radius,
            },
            "timezone": TIMEZONE,
            "update_frequency": "Real-time",
            "notes": "PM2.5 and PM10 measurements from nearby monitoring stations",
        }


# ============================================================================
# Open-Meteo Weather Data Source
# ============================================================================


class OpenMeteoArchiveSource(DataSource):
    """
    Data source for Open-Meteo historical weather data.
    
    Fetches hourly weather variables including temperature, humidity,
    precipitation, pressure, and wind.
    """
    
    BASE_URL = "https://archive-api.open-meteo.com/v1/archive"
    HOURLY_VARIABLES = [
        "temperature_2m",
        "relative_humidity_2m",
        "precipitation",
        "surface_pressure",
        "wind_speed_10m",
        "wind_direction_10m",
    ]
    MAX_DAYS_PER_REQUEST = 365  # Chunk large requests
    
    def __init__(self, logger: Optional[Any] = None):
        """
        Initialize Open-Meteo data source.
        
        Args:
            logger: Logger instance for output.
        """
        super().__init__(logger)
    
    def fetch(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch Open-Meteo weather data for Almaty.
        
        Args:
            start_date: Start date in YYYY-MM-DD format.
            end_date: End date in YYYY-MM-DD format.
        
        Returns:
            DataFrame with columns: datetime, temperature_2m,
            relative_humidity_2m, precipitation, surface_pressure,
            wind_speed_10m, wind_direction_10m.
        """
        self._log(f"Fetching Open-Meteo data from {start_date} to {end_date}")
        
        # Convert dates
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        all_data: List[pd.DataFrame] = []
        
        # Chunk by year to avoid API limits
        current_start = start_dt
        while current_start < end_dt:
            current_end = min(
                current_start + timedelta(days=self.MAX_DAYS_PER_REQUEST - 1),
                end_dt,
            )
            
            self._log(
                f"Fetching chunk: {current_start.date()} to {current_end.date()}",
                level="debug",
            )
            
            chunk_df = self._fetch_chunk(
                current_start.strftime("%Y-%m-%d"),
                current_end.strftime("%Y-%m-%d"),
            )
            
            if not chunk_df.empty:
                all_data.append(chunk_df)
            
            current_start = current_end + timedelta(days=1)
            
            # Respectful rate limiting
            time.sleep(0.5)
        
        if not all_data:
            self._log("No data retrieved from Open-Meteo", level="warning")
            return pd.DataFrame()
        
        # Concatenate all chunks
        df = pd.concat(all_data, ignore_index=True)
        
        # Sort by datetime
        df = df.sort_values("datetime").reset_index(drop=True)
        
        self._log(f"Retrieved {len(df)} hourly weather records from Open-Meteo")
        self._log(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")
        
        return df
    
    def _fetch_chunk(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch a chunk of weather data.
        
        Args:
            start_date: Chunk start date.
            end_date: Chunk end date.
        
        Returns:
            DataFrame with weather data.
        """
        params = {
            "latitude": LATITUDE,
            "longitude": LONGITUDE,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": ",".join(self.HOURLY_VARIABLES),
            "timezone": TIMEZONE,
        }
        
        try:
            response = self.session.get(
                self.BASE_URL,
                params=params,
                timeout=TIMEOUT,
            )
            response.raise_for_status()
            data = response.json()
            
            # Extract hourly data
            hourly_data = data.get("hourly", {})
            
            if not hourly_data or "time" not in hourly_data:
                self._log("No hourly data in response", level="warning")
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(hourly_data)
            
            # Rename time column to datetime
            df = df.rename(columns={"time": "datetime"})
            
            # Convert datetime to timezone-aware
            df["datetime"] = pd.to_datetime(df["datetime"])
            if df["datetime"].dt.tz is None:
                df["datetime"] = df["datetime"].dt.tz_localize(TIMEZONE)
            
            return df
            
        except requests.exceptions.RequestException as e:
            self._log(f"Error fetching weather data: {e}", level="error")
            return pd.DataFrame()
        except Exception as e:
            self._log(f"Unexpected error: {e}", level="error")
            return pd.DataFrame()
    
    def metadata(self) -> Dict[str, Any]:
        """
        Get metadata about Open-Meteo data source.
        
        Returns:
            Dictionary with source information.
        """
        return {
            "source_name": "Open-Meteo Archive",
            "source_type": "Weather",
            "url": self.BASE_URL,
            "parameters": self.HOURLY_VARIABLES,
            "location": {
                "city": CITY,
                "country": COUNTRY,
                "latitude": LATITUDE,
                "longitude": LONGITUDE,
            },
            "timezone": TIMEZONE,
            "temporal_resolution": "Hourly",
            "notes": "Historical weather data from weather reanalysis models",
        }


# ============================================================================
# Data Source Availability Check
# ============================================================================


def check_pollution_sources_almaty(radius_m: int = 25000) -> Dict[str, Any]:
    """
    Check availability of air quality data sources for Almaty.
    
    Performs lightweight queries to determine which data sources have
    PM2.5/PM10 data available for the configured location and date range.
    Does not download full historical data.
    
    Args:
        radius_m: Search radius in meters around Almaty coordinates.
    
    Returns:
        Dictionary with availability information:
        {
            "openaq_available": bool,
            "openaq_station_count": int,
            "openaq_station_names": list,
            "openmeteo_available": bool,
            "recommended_source": str,  # "openaq" or "openmeteo"
        }
    
    Examples:
        >>> result = check_pollution_sources_almaty()
        >>> if result["openaq_available"]:
        ...     print(f"Found {result['openaq_station_count']} OpenAQ stations")
        >>> print(f"Recommended source: {result['recommended_source']}")
    """
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 70)
    logger.info("Checking Air Quality Data Source Availability")
    logger.info("=" * 70)
    logger.info(f"Location: {CITY}, {COUNTRY} ({LATITUDE}°N, {LONGITUDE}°E)")
    logger.info(f"Search radius: {radius_m}m")
    logger.info(f"Date range: {START_DATE} to {END_DATE}")
    
    result = {
        "openaq_available": False,
        "openaq_station_count": 0,
        "openaq_station_names": [],
        "openmeteo_available": False,
        "recommended_source": None,
    }
    
    # Create session with retry logic
    session = requests.Session()
    retry_strategy = Retry(
        total=MAX_RETRIES,
        backoff_factor=2,  # Exponential backoff: 2, 4, 8 seconds
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS"],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    # ========================================================================
    # Check OpenAQ Availability
    # ========================================================================
    
    logger.info("\n--- Checking OpenAQ ---")
    
    try:
        # Query locations endpoint
        locations_url = "https://api.openaq.org/v2/locations"
        locations_params = {
            "coordinates": f"{LATITUDE},{LONGITUDE}",
            "radius": radius_m,
            "limit": 100,
        }
        
        logger.info(f"Querying OpenAQ locations: {locations_url}")
        logger.debug(f"Parameters: {locations_params}")
        
        response = session.get(
            locations_url,
            params=locations_params,
            timeout=TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()
        
        locations = data.get("results", [])
        logger.info(f"Found {len(locations)} monitoring locations")
        
        if not locations:
            logger.warning("No OpenAQ stations found in search area")
            result["openaq_available"] = False
        else:
            # Extract station information
            station_names = []
            pm25_available = False
            pm10_available = False
            
            for location in locations:
                name = location.get("name", "Unknown")
                station_names.append(name)
                
                # Check available parameters
                parameters = location.get("parameters", [])
                if parameters:
                    param_ids = [p.get("id") if isinstance(p, dict) else p for p in parameters]
                    if "pm25" in param_ids:
                        pm25_available = True
                    if "pm10" in param_ids:
                        pm10_available = True
                    
                    logger.debug(f"Station '{name}': {param_ids}")
            
            result["openaq_station_count"] = len(locations)
            result["openaq_station_names"] = station_names
            
            # If parameters info unclear, do a test measurement query
            if not pm25_available and not pm10_available:
                logger.info("Parameters unclear from locations, testing measurements endpoint...")
                
                # Test query for last 7 days
                test_end = pd.to_datetime(END_DATE)
                test_start = max(
                    pd.to_datetime(START_DATE),
                    test_end - timedelta(days=7)
                )
                
                measurements_url = "https://api.openaq.org/v2/measurements"
                
                # Test PM2.5
                test_params = {
                    "coordinates": f"{LATITUDE},{LONGITUDE}",
                    "radius": radius_m,
                    "parameter": "pm25",
                    "date_from": test_start.strftime("%Y-%m-%d"),
                    "date_to": test_end.strftime("%Y-%m-%d"),
                    "limit": 100,
                }
                
                time.sleep(1)  # Rate limiting
                response = session.get(
                    measurements_url,
                    params=test_params,
                    timeout=TIMEOUT,
                )
                response.raise_for_status()
                data = response.json()
                
                if data.get("results"):
                    pm25_available = True
                    logger.info(f"✓ PM2.5 data available ({len(data['results'])} test measurements)")
                
                # Test PM10
                test_params["parameter"] = "pm10"
                time.sleep(1)  # Rate limiting
                response = session.get(
                    measurements_url,
                    params=test_params,
                    timeout=TIMEOUT,
                )
                response.raise_for_status()
                data = response.json()
                
                if data.get("results"):
                    pm10_available = True
                    logger.info(f"✓ PM10 data available ({len(data['results'])} test measurements)")
            
            # Final determination
            if pm25_available or pm10_available:
                result["openaq_available"] = True
                logger.info(f"✓ OpenAQ available: PM2.5={pm25_available}, PM10={pm10_available}")
                logger.info(f"  Stations: {', '.join(station_names[:5])}" + 
                          (f" (+{len(station_names)-5} more)" if len(station_names) > 5 else ""))
            else:
                logger.warning("OpenAQ stations found but no PM2.5/PM10 data available")
                result["openaq_available"] = False
                
    except requests.exceptions.RequestException as e:
        logger.error(f"Error checking OpenAQ: {e}")
        result["openaq_available"] = False
    except Exception as e:
        logger.error(f"Unexpected error checking OpenAQ: {e}")
        result["openaq_available"] = False
    
    # ========================================================================
    # Check Open-Meteo Air Quality Availability
    # ========================================================================
    
    logger.info("\n--- Checking Open-Meteo Air Quality ---")
    
    try:
        # Test query for first 7 days
        test_start = pd.to_datetime(START_DATE)
        test_end = min(
            test_start + timedelta(days=7),
            pd.to_datetime(END_DATE)
        )
        
        air_quality_url = "https://air-quality-api.open-meteo.com/v1/air-quality"
        aq_params = {
            "latitude": LATITUDE,
            "longitude": LONGITUDE,
            "hourly": "pm2_5,pm10",
            "start_date": test_start.strftime("%Y-%m-%d"),
            "end_date": test_end.strftime("%Y-%m-%d"),
            "timezone": TIMEZONE,
        }
        
        logger.info(f"Querying Open-Meteo Air Quality: {air_quality_url}")
        logger.debug(f"Parameters: {aq_params}")
        
        time.sleep(1)  # Rate limiting
        response = session.get(
            air_quality_url,
            params=aq_params,
            timeout=TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()
        
        # Check if hourly data exists
        hourly_data = data.get("hourly", {})
        time_data = hourly_data.get("time", [])
        
        if time_data and len(time_data) > 0:
            result["openmeteo_available"] = True
            logger.info(f"✓ Open-Meteo Air Quality available ({len(time_data)} hourly records)")
            
            # Check data quality
            pm25_data = hourly_data.get("pm2_5", [])
            pm10_data = hourly_data.get("pm10", [])
            
            pm25_valid = sum(1 for v in pm25_data if v is not None)
            pm10_valid = sum(1 for v in pm10_data if v is not None)
            
            logger.info(f"  PM2.5: {pm25_valid}/{len(pm25_data)} non-null values")
            logger.info(f"  PM10: {pm10_valid}/{len(pm10_data)} non-null values")
        else:
            result["openmeteo_available"] = False
            logger.warning("No hourly data returned from Open-Meteo Air Quality")
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Error checking Open-Meteo: {e}")
        result["openmeteo_available"] = False
    except Exception as e:
        logger.error(f"Unexpected error checking Open-Meteo: {e}")
        result["openmeteo_available"] = False
    
    # ========================================================================
    # Determine Recommendation
    # ========================================================================
    
    logger.info("\n" + "=" * 70)
    logger.info("Recommendation")
    logger.info("=" * 70)
    
    if result["openaq_available"] and result["openmeteo_available"]:
        # Both available - prefer OpenAQ (ground-truth measurements)
        result["recommended_source"] = "openaq"
        logger.info("✓ Both sources available")
        logger.info("→ Recommended: OpenAQ (ground-truth station measurements)")
        logger.info("  Consider using Open-Meteo as supplementary/gap-filling")
    elif result["openaq_available"]:
        result["recommended_source"] = "openaq"
        logger.info("✓ OpenAQ available")
        logger.info("→ Recommended: OpenAQ")
    elif result["openmeteo_available"]:
        result["recommended_source"] = "openmeteo"
        logger.info("✓ Open-Meteo Air Quality available")
        logger.info("→ Recommended: Open-Meteo")
    else:
        result["recommended_source"] = None
        logger.warning("✗ No air quality sources available for this location")
        logger.warning("  Consider:")
        logger.warning("  - Expanding search radius")
        logger.warning("  - Using only weather data (Open-Meteo Archive)")
        logger.warning("  - Finding alternative data sources")
    
    logger.info("=" * 70)
    
    return result

# Add this function to src/data_download.py after check_pollution_sources_almaty()

def download_pollution_almaty(
    output_csv: Optional[Path] = None,
    output_meta: Optional[Path] = None,
    radius_m: int = 25000,
    force_source: Optional[str] = None,
) -> pd.DataFrame:
    """
    Download PM2.5/PM10 pollution data for Almaty.
    
    Downloads air quality data and returns a clean hourly time series
    in wide format with columns: datetime, pm25, pm10.
    
    Args:
        output_csv: Path to save CSV file. Defaults to data/raw/pollution.csv.
        output_meta: Path to save metadata JSON. Defaults to data/raw/metadata_pollution.json.
        radius_m: Search radius in meters for OpenAQ stations.
        force_source: Force specific source ("openaq" or "openmeteo"). 
                     If None, automatically selects best available source.
    
    Returns:
        DataFrame with columns: datetime (tz-aware), pm25, pm10
    
    Raises:
        RuntimeError: If no data sources are available or download fails.
        ValueError: If force_source is invalid.
    
    Examples:
        >>> df = download_pollution_almaty()
        >>> print(df.columns.tolist())
        ['datetime', 'pm25', 'pm10']
        
        >>> df = download_pollution_almaty(force_source="openaq")
        >>> print(df.shape)
    """
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 70)
    logger.info("Download Pollution Data for Almaty")
    logger.info("=" * 70)
    logger.info(f"Location: {CITY}, {COUNTRY} ({LATITUDE}°N, {LONGITUDE}°E)")
    logger.info(f"Date range: {START_DATE} to {END_DATE}")
    logger.info(f"Timezone: {TIMEZONE}")
    
    # Set default output paths
    if output_csv is None:
        output_csv = RAW_DATA_DIR / "pollution.csv"
    if output_meta is None:
        output_meta = RAW_DATA_DIR / "metadata_pollution.json"
    
    # Ensure output directories exist
    safe_mkdir(output_csv.parent)
    safe_mkdir(output_meta.parent)
    
    logger.info(f"Output CSV: {output_csv}")
    logger.info(f"Output metadata: {output_meta}")
    
    # ========================================================================
    # Step 1: Determine data source
    # ========================================================================
    
    logger.info("\n" + "=" * 70)
    logger.info("Step 1: Determine Data Source")
    logger.info("=" * 70)
    
    if force_source is not None:
        # Validate force_source
        if force_source not in ["openaq", "openmeteo"]:
            raise ValueError(f"Invalid force_source: {force_source}. Must be 'openaq' or 'openmeteo'.")
        
        source = force_source
        logger.info(f"Forced source: {source.upper()}")
        
    else:
        # Auto-select source
        logger.info("Auto-selecting source...")
        availability = check_pollution_sources_almaty(radius_m=radius_m)
        source = availability.get("recommended_source")
        
        if source is None:
            error_msg = (
                "No air quality data sources available for Almaty. "
                f"Location: {LATITUDE}°N, {LONGITUDE}°E, Radius: {radius_m}m. "
                "Try increasing radius_m or check data availability."
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        logger.info(f"✓ Recommended source: {source.upper()}")
    
    # Initialize metadata
    metadata = {
        "source_used": source,
        "parameters": ["pm25", "pm10"],
        "coordinates": {
            "city": CITY,
            "country": COUNTRY,
            "latitude": LATITUDE,
            "longitude": LONGITUDE,
        },
        "timezone": TIMEZONE,
        "date_range": {
            "start": START_DATE,
            "end": END_DATE,
        },
        "radius_m": radius_m if source == "openaq" else None,
    }
    
    df_pollution = None
    
    # ========================================================================
    # Step 2: Download data based on source
    # ========================================================================
    
    logger.info("\n" + "=" * 70)
    logger.info(f"Step 2: Download from {source.upper()}")
    logger.info("=" * 70)
    
    if source == "openaq":
        # ====================================================================
        # OpenAQ Source
        # ====================================================================
        
        logger.info("Using OpenAQ Measurements Source")
        
        # Fetch raw data
        logger.info(f"Fetching data from {START_DATE} to {END_DATE}...")
        openaq_source = OpenAQMeasurementsSource(logger=logger)
        df_raw = openaq_source.fetch(START_DATE, END_DATE)
        
        if df_raw.empty:
            error_msg = "OpenAQ returned no data"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        logger.info(f"✓ Retrieved {len(df_raw)} raw measurements")
        logger.info(f"  Parameters found: {df_raw['parameter'].unique().tolist()}")
        
        # Ensure datetime is timezone-aware
        if df_raw["datetime"].dt.tz is None:
            df_raw["datetime"] = df_raw["datetime"].dt.tz_localize(TIMEZONE)
        else:
            df_raw["datetime"] = df_raw["datetime"].dt.tz_convert(TIMEZONE)
        
        # Convert to wide format using pivot_table
        logger.info("Converting to wide format...")
        wide = df_raw.pivot_table(
            index="datetime",
            columns="parameter",
            values="value",
            aggfunc="mean"
        )
        
        logger.info(f"  Pivot result columns: {wide.columns.tolist()}")
        
        # Ensure pm25 and pm10 columns exist
        if "pm25" not in wide.columns:
            logger.warning("  pm25 column missing, creating with NaN")
            wide["pm25"] = np.nan
        
        if "pm10" not in wide.columns:
            logger.warning("  pm10 column missing, creating with NaN")
            wide["pm10"] = np.nan
        
        # Keep only pm25 and pm10 columns
        wide = wide[["pm25", "pm10"]]
        
        # Reset index to get datetime as a column
        wide = wide.reset_index()
        
        # Rename columns exactly to: datetime, pm25, pm10
        wide.columns = ["datetime", "pm25", "pm10"]
        
        # Sort by datetime
        wide = wide.sort_values("datetime")
        
        logger.info(f"  Wide format shape: {wide.shape}")
        logger.info(f"  Columns: {wide.columns.tolist()}")
        
        # Resample to hourly
        logger.info("Resampling to hourly frequency...")
        wide = wide.set_index("datetime")
        wide = wide.resample("H").mean()
        wide = wide.reset_index()
        
        logger.info(f"  Hourly shape: {wide.shape}")
        
        df_pollution = wide
        
        # Update metadata
        metadata["url"] = openaq_source.BASE_URL
        metadata["method"] = "pivot_table aggregation with hourly resampling"
        
    elif source == "openmeteo":
        # ====================================================================
        # Open-Meteo Air Quality Source
        # ====================================================================
        
        logger.info("Using Open-Meteo Air Quality API")
        
        api_url = "https://air-quality-api.open-meteo.com/v1/air-quality"
        
        # Create session with retry logic
        session = requests.Session()
        retry_strategy = Retry(
            total=MAX_RETRIES,
            backoff_factor=2,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        # Fetch data in chunks (to handle large date ranges)
        start_dt = pd.to_datetime(START_DATE)
        end_dt = pd.to_datetime(END_DATE)
        
        all_chunks = []
        current_start = start_dt
        chunk_days = 365
        
        logger.info(f"Fetching data from {START_DATE} to {END_DATE}...")
        
        while current_start <= end_dt:
            current_end = min(
                current_start + timedelta(days=chunk_days - 1),
                end_dt
            )
            
            logger.info(f"  Chunk: {current_start.date()} to {current_end.date()}")
            
            params = {
                "latitude": LATITUDE,
                "longitude": LONGITUDE,
                "hourly": "pm2_5,pm10",
                "start_date": current_start.strftime("%Y-%m-%d"),
                "end_date": current_end.strftime("%Y-%m-%d"),
                "timezone": TIMEZONE,
            }
            
            try:
                response = session.get(api_url, params=params, timeout=TIMEOUT)
                response.raise_for_status()
                data = response.json()
                
                hourly = data.get("hourly", {})
                
                if hourly and "time" in hourly:
                    df_chunk = pd.DataFrame({
                        "datetime": hourly.get("time", []),
                        "pm25": hourly.get("pm2_5", []),
                        "pm10": hourly.get("pm10", []),
                    })
                    all_chunks.append(df_chunk)
                    logger.info(f"    ✓ Retrieved {len(df_chunk)} hourly records")
                else:
                    logger.warning(f"    No data for this chunk")
                
            except requests.exceptions.RequestException as e:
                logger.error(f"    Error fetching chunk: {e}")
            
            current_start = current_end + timedelta(days=1)
            time.sleep(0.5)  # Rate limiting
        
        if not all_chunks:
            error_msg = "Open-Meteo Air Quality API returned no data"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        # Concatenate all chunks
        df_pollution = pd.concat(all_chunks, ignore_index=True)

        logger.info(f"✓ Retrieved {len(df_pollution)} total hourly records")

        # Ensure numeric pollution values
        df_pollution["pm25"] = pd.to_numeric(df_pollution["pm25"], errors="coerce")
        df_pollution["pm10"] = pd.to_numeric(df_pollution["pm10"], errors="coerce")

        # Robust timezone handling (DST-safe)
        dt = pd.to_datetime(df_pollution["datetime"], errors="coerce")
        df_pollution = df_pollution.dropna(subset=["datetime"])

        # Treat times as UTC first, then convert to Asia/Almaty
        df_pollution["datetime"] = dt.dt.tz_localize("UTC").dt.tz_convert(TIMEZONE)

        # Sort by datetime
        df_pollution = df_pollution.sort_values("datetime").reset_index(drop=True)

        
        # Update metadata
        metadata["url"] = api_url
        metadata["method"] = "Direct API fetch with hourly data"
        
    else:
        error_msg = f"Unknown source: {source}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)
    
    # ========================================================================
    # Step 3: Validate and log final DataFrame
    # ========================================================================
    
    logger.info("\n" + "=" * 70)
    logger.info("Step 3: Validate Final DataFrame")
    logger.info("=" * 70)
    
    # Validate columns
    expected_cols = ["datetime", "pm25", "pm10"]
    actual_cols = df_pollution.columns.tolist()
    
    if actual_cols != expected_cols:
        error_msg = f"Column mismatch. Expected {expected_cols}, got {actual_cols}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)
    
    logger.info(f"✓ Columns: {actual_cols}")
    logger.info(f"✓ Shape: {df_pollution.shape}")
    logger.info(f"✓ Retrieved rows: {len(df_pollution)}")
    
    # Date range
    datetime_min = df_pollution["datetime"].min()
    datetime_max = df_pollution["datetime"].max()
    logger.info(f"✓ Datetime range: {datetime_min} to {datetime_max}")
    
    # Data quality
    pm25_valid = df_pollution["pm25"].notna().sum()
    pm10_valid = df_pollution["pm10"].notna().sum()
    pm25_missing_pct = (df_pollution["pm25"].isna().sum() / len(df_pollution)) * 100
    pm10_missing_pct = (df_pollution["pm10"].isna().sum() / len(df_pollution)) * 100
    
    logger.info(f"  PM2.5: {pm25_valid:,} valid ({pm25_missing_pct:.1f}% missing)")
    logger.info(f"  PM10: {pm10_valid:,} valid ({pm10_missing_pct:.1f}% missing)")
    
    # Statistics
    if pm25_valid > 0:
        logger.info(f"  PM2.5 mean: {df_pollution['pm25'].mean():.2f} µg/m³")
    if pm10_valid > 0:
        logger.info(f"  PM10 mean: {df_pollution['pm10'].mean():.2f} µg/m³")
    
    # ========================================================================
    # Step 4: Save CSV
    # ========================================================================
    
    logger.info("\n" + "=" * 70)
    logger.info("Step 4: Save CSV")
    logger.info("=" * 70)
    
    save_df(df_pollution, output_csv, index=False, logger=logger)
    logger.info(f"✓ Saved to {output_csv}")
    
    # ========================================================================
    # Step 5: Save Metadata
    # ========================================================================
    
    logger.info("\n" + "=" * 70)
    logger.info("Step 5: Save Metadata")
    logger.info("=" * 70)
    
    # Update metadata with actual results
    metadata["retrieved_rows"] = len(df_pollution)
    metadata["datetime_min"] = str(datetime_min)
    metadata["datetime_max"] = str(datetime_max)
    metadata["created_at"] = datetime.now().isoformat()
    metadata["output_file"] = str(output_csv)
    metadata["missing_values"] = {
        "pm25": int(df_pollution["pm25"].isna().sum()),
        "pm10": int(df_pollution["pm10"].isna().sum()),
    }
    
    with open(output_meta, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    logger.info(f"✓ Saved metadata to {output_meta}")
    
    # ========================================================================
    # Summary
    # ========================================================================
    
    logger.info("\n" + "=" * 70)
    logger.info("DOWNLOAD COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Source: {source.upper()}")
    logger.info(f"Rows: {len(df_pollution):,}")
    logger.info(f"Columns: {df_pollution.columns.tolist()}")
    logger.info(f"Date range: {datetime_min} to {datetime_max}")
    logger.info(f"CSV: {output_csv}")
    logger.info(f"Metadata: {output_meta}")
    logger.info("=" * 70)
    
    return df_pollution

# ============================================================================
# Main Download Function
# ============================================================================


def download_all_sources(
    start_date: str,
    end_date: str,
    output_dir: Path,
    logger: Any,
) -> Dict[str, Dict[str, Any]]:
    """
    Download data from all sources and save to files.
    
    Args:
        start_date: Start date in YYYY-MM-DD format.
        end_date: End date in YYYY-MM-DD format.
        output_dir: Directory to save output files.
        logger: Logger instance.
    
    Returns:
        Dictionary mapping source names to metadata.
    """
    # Ensure output directory exists
    safe_mkdir(output_dir)
    
    # Define data sources
    sources = [
        ("openaq", OpenAQMeasurementsSource(logger)),
        ("openmeteo", OpenMeteoArchiveSource(logger)),
    ]
    
    results = {}
    
    for source_name, source in sources:
        logger.info("=" * 70)
        logger.info(f"Downloading from: {source_name}")
        logger.info("=" * 70)
        
        try:
            # Fetch data
            df = source.fetch(start_date, end_date)
            
            if df.empty:
                logger.warning(f"No data retrieved from {source_name}")
                continue
            
            # Save DataFrame
            output_path = output_dir / f"{source_name}.csv"
            save_df(df, output_path, index=False, logger=logger)
            
            # Get metadata
            meta = source.metadata()
            meta["date_range"] = {
                "start": start_date,
                "end": end_date,
            }
            meta["retrieved_rows"] = len(df)
            meta["retrieved_at"] = datetime.now().isoformat()
            meta["columns"] = list(df.columns)
            meta["output_file"] = str(output_path)
            
            # Save metadata
            meta_path = output_dir / f"metadata_{source_name}.json"
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved metadata to {meta_path}")
            
            results[source_name] = meta
            
        except Exception as e:
            logger.error(f"Failed to download from {source_name}: {e}", exc_info=True)
            continue
    
    return results


# ============================================================================
# CLI Entry Point
# ============================================================================


def main():
    """
    Main entry point for data download script.
    
    Downloads data from all sources using default configuration
    and saves to data/raw/ directory.
    """
    # Setup logging
    logger = setup_logging(
        LOGS_DIR / "step1_download.log",
        log_level="INFO",
    )
    
    logger.info("=" * 70)
    logger.info("Almaty Air Quality Data Download")
    logger.info("=" * 70)
    logger.info(f"City: {CITY}, {COUNTRY}")
    logger.info(f"Coordinates: {LATITUDE}°N, {LONGITUDE}°E")
    logger.info(f"Date range: {START_DATE} to {END_DATE}")
    logger.info(f"Output directory: {RAW_DATA_DIR}")
    logger.info("=" * 70)
    
    try:
        # First, check data source availability
        logger.info("\nChecking data source availability...")
        availability = check_pollution_sources_almaty(radius_m=25000)
        
        # Download all sources
        results = download_all_sources(
            start_date=START_DATE,
            end_date=END_DATE,
            output_dir=RAW_DATA_DIR,
            logger=logger,
        )
        
        # Summary
        logger.info("")
        logger.info("=" * 70)
        logger.info("Download Summary")
        logger.info("=" * 70)
        
        if results:
            for source_name, meta in results.items():
                logger.info(f"{source_name}:")
                logger.info(f"  Rows: {meta['retrieved_rows']:,}")
                logger.info(f"  Columns: {len(meta['columns'])}")
                logger.info(f"  File: {meta['output_file']}")
        else:
            logger.warning("No data was successfully downloaded")
        
        logger.info("=" * 70)
        logger.info("Download complete!")
        
    except Exception as e:
        logger.error(f"Fatal error during download: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())