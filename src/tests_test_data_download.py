"""
Unit tests for data_download module.

Run with: pytest tests/test_data_download.py -v
"""

import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import pandas as pd
import pytest

from src.data_download import (
    DataSource,
    OpenAQMeasurementsSource,
    OpenMeteoArchiveSource,
    download_all_sources,
)


class MockDataSource(DataSource):
    """Mock data source for testing abstract base class."""
    
    def fetch(self, start_date: str, end_date: str) -> pd.DataFrame:
        return pd.DataFrame({
            "datetime": pd.date_range("2024-01-01", periods=24, freq="h"),
            "value": range(24),
        })
    
    def metadata(self) -> dict:
        return {"source_name": "Mock", "url": "http://example.com"}


def test_data_source_abstract():
    """Test that DataSource cannot be instantiated directly."""
    with pytest.raises(TypeError):
        DataSource()


def test_mock_data_source():
    """Test mock data source implementation."""
    source = MockDataSource()
    
    # Test fetch
    df = source.fetch("2024-01-01", "2024-01-31")
    assert not df.empty
    assert "datetime" in df.columns
    assert "value" in df.columns
    
    # Test metadata
    meta = source.metadata()
    assert meta["source_name"] == "Mock"
    assert "url" in meta


@patch("src.data_download.requests.Session")
def test_openaq_session_creation(mock_session):
    """Test that OpenAQ source creates session with retry logic."""
    source = OpenAQMeasurementsSource()
    assert source.session is not None


def test_openaq_metadata():
    """Test OpenAQ metadata generation."""
    source = OpenAQMeasurementsSource()
    meta = source.metadata()
    
    assert meta["source_name"] == "OpenAQ"
    assert meta["source_type"] == "Air Quality"
    assert "pm25" in meta["parameters"]
    assert "pm10" in meta["parameters"]
    assert "location" in meta
    assert meta["location"]["city"] == "Almaty"


@patch("src.data_download.requests.Session.get")
def test_openaq_fetch_empty_response(mock_get):
    """Test OpenAQ fetch with empty API response."""
    # Mock empty response
    mock_response = Mock()
    mock_response.json.return_value = {"results": []}
    mock_response.raise_for_status = Mock()
    mock_get.return_value = mock_response
    
    source = OpenAQMeasurementsSource()
    df = source.fetch("2024-01-01", "2024-01-02")
    
    assert df.empty


@patch("src.data_download.requests.Session.get")
def test_openaq_fetch_with_data(mock_get):
    """Test OpenAQ fetch with sample data."""
    # Mock response with data
    mock_response = Mock()
    mock_response.json.return_value = {
        "results": [
            {
                "date": {"utc": "2024-01-01T00:00:00Z"},
                "parameter": "pm25",
                "value": 15.5,
                "unit": "µg/m³",
                "location": "Test Station",
                "city": "Almaty",
                "country": "Kazakhstan",
                "coordinates": {"latitude": 43.22, "longitude": 76.85},
            }
        ]
    }
    mock_response.raise_for_status = Mock()
    mock_get.return_value = mock_response
    
    source = OpenAQMeasurementsSource()
    df = source.fetch("2024-01-01", "2024-01-02")
    
    assert not df.empty
    assert "datetime" in df.columns
    assert "parameter" in df.columns
    assert "value" in df.columns
    assert df["parameter"].iloc[0] == "pm25"


def test_openmeteo_metadata():
    """Test Open-Meteo metadata generation."""
    source = OpenMeteoArchiveSource()
    meta = source.metadata()
    
    assert meta["source_name"] == "Open-Meteo Archive"
    assert meta["source_type"] == "Weather"
    assert "temperature_2m" in meta["parameters"]
    assert meta["temporal_resolution"] == "Hourly"


@patch("src.data_download.requests.Session.get")
def test_openmeteo_fetch_empty_response(mock_get):
    """Test Open-Meteo fetch with empty response."""
    mock_response = Mock()
    mock_response.json.return_value = {}
    mock_response.raise_for_status = Mock()
    mock_get.return_value = mock_response
    
    source = OpenMeteoArchiveSource()
    df = source.fetch("2024-01-01", "2024-01-02")
    
    assert df.empty


@patch("src.data_download.requests.Session.get")
def test_openmeteo_fetch_with_data(mock_get):
    """Test Open-Meteo fetch with sample data."""
    mock_response = Mock()
    mock_response.json.return_value = {
        "hourly": {
            "time": ["2024-01-01T00:00", "2024-01-01T01:00"],
            "temperature_2m": [5.2, 5.0],
            "relative_humidity_2m": [75, 76],
            "precipitation": [0.0, 0.1],
            "surface_pressure": [1013.2, 1013.0],
            "wind_speed_10m": [3.5, 3.7],
            "wind_direction_10m": [180, 185],
        }
    }
    mock_response.raise_for_status = Mock()
    mock_get.return_value = mock_response
    
    source = OpenMeteoArchiveSource()
    df = source.fetch("2024-01-01", "2024-01-02")
    
    assert not df.empty
    assert "datetime" in df.columns
    assert "temperature_2m" in df.columns
    assert len(df) == 2


@patch("src.data_download.OpenAQMeasurementsSource")
@patch("src.data_download.OpenMeteoArchiveSource")
def test_download_all_sources(mock_meteo, mock_aq, tmp_path):
    """Test downloading from all sources."""
    # Create mock logger
    logger = Mock()
    
    # Mock OpenAQ source
    mock_aq_instance = Mock()
    mock_aq_instance.fetch.return_value = pd.DataFrame({
        "datetime": pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC"),
        "parameter": ["pm25", "pm25", "pm25"],
        "value": [10, 11, 12],
    })
    mock_aq_instance.metadata.return_value = {
        "source_name": "OpenAQ",
        "parameters": ["pm25", "pm10"],
    }
    mock_aq.return_value = mock_aq_instance
    
    # Mock Open-Meteo source
    mock_meteo_instance = Mock()
    mock_meteo_instance.fetch.return_value = pd.DataFrame({
        "datetime": pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC"),
        "temperature_2m": [5.0, 5.5, 6.0],
    })
    mock_meteo_instance.metadata.return_value = {
        "source_name": "Open-Meteo",
        "parameters": ["temperature_2m"],
    }
    mock_meteo.return_value = mock_meteo_instance
    
    # Run download
    results = download_all_sources(
        start_date="2024-01-01",
        end_date="2024-01-02",
        output_dir=tmp_path,
        logger=logger,
    )
    
    # Check results
    assert "openaq" in results
    assert "openmeteo" in results
    
    # Check files were created
    assert (tmp_path / "openaq.csv").exists()
    assert (tmp_path / "openmeteo.csv").exists()
    assert (tmp_path / "metadata_openaq.json").exists()
    assert (tmp_path / "metadata_openmeteo.json").exists()
    
    # Verify metadata content
    with open(tmp_path / "metadata_openaq.json") as f:
        meta = json.load(f)
        assert meta["source_name"] == "OpenAQ"
        assert meta["retrieved_rows"] == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])