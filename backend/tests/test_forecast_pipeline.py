import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
import httpx

from backend.ml_model import (
    fetch_historical_air_quality,
    train_and_forecast_pm25,
    UpstreamProviderError,
    InsufficientDataError,
    ModelTrainingError,
    MODEL_CACHE
)
from backend.main import app


@pytest.fixture(autouse=True)
def clear_model_cache():
    MODEL_CACHE.clear()
    yield
    MODEL_CACHE.clear()


def make_valid_open_meteo_payload(hours=200):
    start = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=hours)
    times = [(start + pd.Timedelta(hours=i)).strftime("%Y-%m-%dT%H:00") for i in range(hours)]
    # Generate realistic PM2.5 values
    h_arr = np.array([(start + pd.Timedelta(hours=i)).hour for i in range(hours)])
    pm25 = (15.0 + 5.0 * np.sin(2 * np.pi * h_arr / 24.0)).tolist()

    return {
        "latitude": 28.61,
        "longitude": 77.23,
        "utc_offset_seconds": 0,
        "hourly": {
            "time": times,
            "pm2_5": pm25,
            "nitrogen_dioxide": [12.0] * hours,
            "ozone": [25.0] * hours,
            "wind_speed_10m": [3.5] * hours,
            "temperature_2m": [22.0] * hours,
            "relative_humidity_2m": [50.0] * hours,
        }
    }


# =====================================================================
# 1. Test Upstream API Failures
# =====================================================================

@pytest.mark.asyncio
async def test_upstream_http_status_error_raises_upstream_provider_error():
    mock_response = MagicMock()
    mock_response.status_code = 502
    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "502 Bad Gateway",
        request=MagicMock(),
        response=mock_response
    )

    with patch("httpx.AsyncClient.get", return_value=mock_response):
        with pytest.raises(UpstreamProviderError) as exc_info:
            await fetch_historical_air_quality(28.61, 77.23, allow_demo=False)
        assert exc_info.value.status_code == 502
        assert "Upstream air quality provider returned HTTP" in exc_info.value.message


@pytest.mark.asyncio
async def test_upstream_network_timeout_raises_upstream_provider_error():
    with patch("httpx.AsyncClient.get", side_effect=httpx.ConnectTimeout("Connection timed out")):
        with pytest.raises(UpstreamProviderError) as exc_info:
            await fetch_historical_air_quality(28.61, 77.23, allow_demo=False)
        assert exc_info.value.status_code == 502
        assert "unreachable" in exc_info.value.message


# =====================================================================
# 2. Test Malformed Data Payloads
# =====================================================================

@pytest.mark.asyncio
async def test_upstream_malformed_json_raises_upstream_provider_error():
    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {"error": "Invalid format, missing hourly"}

    with patch("httpx.AsyncClient.get", return_value=mock_response):
        with pytest.raises(UpstreamProviderError) as exc_info:
            await fetch_historical_air_quality(28.61, 77.23, allow_demo=False)
        assert exc_info.value.status_code == 502
        assert "invalid data payload" in exc_info.value.message.lower()


# =====================================================================
# 3. Test Empty Dataset Handling
# =====================================================================

@pytest.mark.asyncio
async def test_empty_observations_raises_insufficient_data_error():
    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {
        "utc_offset_seconds": 0,
        "hourly": {
            "time": [],
            "pm2_5": []
        }
    }

    with patch("httpx.AsyncClient.get", return_value=mock_response):
        with pytest.raises(InsufficientDataError) as exc_info:
            await fetch_historical_air_quality(28.61, 77.23, allow_demo=False)
        assert exc_info.value.status_code == 422
        assert "No historical PM2.5 observations" in exc_info.value.message


# =====================================================================
# 4. Test Insufficient History (< 50 Valid Readings)
# =====================================================================

@pytest.mark.asyncio
async def test_insufficient_readings_raises_insufficient_data_error():
    # Only 20 valid PM2.5 observations (below 50 minimum threshold)
    payload = make_valid_open_meteo_payload(hours=20)
    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = payload

    with patch("httpx.AsyncClient.get", return_value=mock_response):
        with pytest.raises(InsufficientDataError) as exc_info:
            await fetch_historical_air_quality(28.61, 77.23, allow_demo=False)
        assert exc_info.value.status_code == 422
        assert "minimum required is 50" in exc_info.value.message


@pytest.mark.asyncio
async def test_all_nans_pm25_raises_insufficient_data_error():
    payload = make_valid_open_meteo_payload(hours=100)
    # Set all PM2.5 readings to None / NaN
    payload["hourly"]["pm2_5"] = [None] * 100
    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = payload

    with patch("httpx.AsyncClient.get", return_value=mock_response):
        with pytest.raises(InsufficientDataError) as exc_info:
            await fetch_historical_air_quality(28.61, 77.23, allow_demo=False)
        assert exc_info.value.status_code == 422


# =====================================================================
# 5. Test Opt-In Demonstration Mode (allow_demo=True)
# =====================================================================

@pytest.mark.asyncio
async def test_opt_in_demo_mode_returns_explicitly_labeled_synthetic_data():
    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "500 Internal Server Error",
        request=MagicMock(),
        response=mock_response
    )

    with patch("httpx.AsyncClient.get", return_value=mock_response):
        # 1. fetch_historical_air_quality with allow_demo=True returns is_synthetic=True
        df, offset, is_synthetic = await fetch_historical_air_quality(28.61, 77.23, allow_demo=True)
        assert is_synthetic is True
        assert len(df) > 50

        # 2. train_and_forecast_pm25 with allow_demo=True produces explicitly flagged forecast
        result = await train_and_forecast_pm25(28.61, 77.23, allow_demo=True)
        assert result["status"] == "success"
        assert result["is_synthetic"] is True
        assert result["data_source"] == "synthetic_demo"
        assert "warning" in result
        assert "Demonstration data" in result["warning"]
        assert len(result["forecast"]) == 24


# =====================================================================
# 6. Test Successful Genuine Forecast Pipeline
# =====================================================================

@pytest.mark.asyncio
async def test_successful_genuine_forecasting():
    payload = make_valid_open_meteo_payload(hours=150)
    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = payload

    with patch("httpx.AsyncClient.get", return_value=mock_response):
        result = await train_and_forecast_pm25(28.61, 77.23, allow_demo=False)
        assert result["status"] == "success"
        assert result["is_synthetic"] is False
        assert result["data_source"] == "live_open_meteo"
        assert "warning" not in result
        assert len(result["forecast"]) == 24
        assert result["forecast_average"] > 0
        assert result["forecast_min"] > 0
        assert result["forecast_max"] >= result["forecast_min"]

        first_step = result["forecast"][0]
        assert first_step["step"] == 1
        assert "predicted_pm25" in first_step
        assert "lower_bound" in first_step
        assert "upper_bound" in first_step
        assert first_step["lower_bound"] <= first_step["predicted_pm25"] <= first_step["upper_bound"]


# =====================================================================
# 7. Test FastAPI HTTP Endpoints via TestClient
# =====================================================================

def test_api_predict_upstream_error_endpoint():
    client = TestClient(app)
    with patch("httpx.AsyncClient.get", side_effect=httpx.ConnectTimeout("Timeout")):
        response = client.get("/api/predict/28.61/77.23")
        assert response.status_code == 502
        body = response.json()
        assert body["status"] == "error"
        assert body["error_code"] == "UPSTREAM_PROVIDER_ERROR"
        assert body["is_synthetic"] is False


def test_api_predict_insufficient_data_endpoint():
    client = TestClient(app)
    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {"utc_offset_seconds": 0, "hourly": {"time": [], "pm2_5": []}}

    with patch("httpx.AsyncClient.get", return_value=mock_response):
        response = client.get("/api/predict/28.61/77.23")
        assert response.status_code == 422
        body = response.json()
        assert body["status"] == "error"
        assert body["error_code"] == "INSUFFICIENT_DATA"
        assert body["is_synthetic"] is False


def test_api_predict_opt_in_demo_endpoint():
    client = TestClient(app)
    with patch("httpx.AsyncClient.get", side_effect=httpx.ConnectTimeout("Timeout")):
        response = client.get("/api/predict/28.61/77.23?allow_demo=true")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "success"
        assert body["is_synthetic"] is True
        assert body["data_source"] == "synthetic_demo"
        assert "warning" in body
