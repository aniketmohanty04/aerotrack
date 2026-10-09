import os
import time
import math
import asyncio
import httpx
import numpy as np
import pandas as pd
import xgboost as xgb
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any, Optional, Tuple

# In-memory cache for models and predictions
# Key: (round(lat, 4), round(lon, 4), allow_demo), Value: { 'timestamp': float, 'model': model, 'data': dict }
MODEL_CACHE: Dict[tuple, Dict[str, Any]] = {}
CACHE_TTL_SECONDS = 1800  # 30 minutes

# In-memory cache for conformal calibration quantiles
# Key: (round(lat, 4), round(lon, 4), data_version) and (round(lat, 4), round(lon, 4))
# Value: { 'timestamp': float, 'quantiles': dict, 'calibration_samples': int, 'data_version': str }
CALIBRATION_CACHE: Dict[Any, Dict[str, Any]] = {}
CALIBRATION_TTL_SECONDS = 21600  # 6 hours
MODEL_VERSION = "v1.2"

# Coordinate-level asyncio locks to serialize duplicate concurrent forecast training
FORECAST_LOCKS: Dict[Tuple[float, float], asyncio.Lock] = {}
_LOCKS_GUARD = asyncio.Lock()


def normalize_coordinates(lat: float, lon: float) -> tuple[float, float]:
    """
    Ensure latitude is in [-90.0, 90.0] and longitude is in [-180.0, 180.0]
    to prevent Open-Meteo 400 Bad Request errors when clicking wrapped world maps.
    """
    clamped_lat = max(-89.9, min(89.9, float(lat)))
    wrapped_lon = (((float(lon) + 180.0) % 360.0 + 360.0) % 360.0) - 180.0
    return round(clamped_lat, 4), round(wrapped_lon, 4)


async def get_coordinate_lock(lat: float, lon: float) -> asyncio.Lock:
    """Retrieve or create an asyncio.Lock for the specified normalized coordinate pair."""
    norm_lat, norm_lon = normalize_coordinates(lat, lon)
    key = (norm_lat, norm_lon)
    async with _LOCKS_GUARD:
        if key not in FORECAST_LOCKS:
            FORECAST_LOCKS[key] = asyncio.Lock()
        return FORECAST_LOCKS[key]


import logging

logger = logging.getLogger("aerotrack.forecast")


class ForecastError(Exception):
    """Base exception for forecasting pipeline errors."""
    def __init__(self, message: str, error_code: str = "FORECAST_ERROR", status_code: int = 500, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.status_code = status_code
        self.details = details or {}


class UpstreamProviderError(ForecastError):
    """Raised when the upstream provider fails or returns unparseable/error responses."""
    def __init__(self, message: str, status_code: int = 502, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, error_code="UPSTREAM_PROVIDER_ERROR", status_code=status_code, details=details)


class InsufficientDataError(ForecastError):
    """Raised when historical observations are empty, missing, or have too few valid readings."""
    def __init__(self, message: str, status_code: int = 422, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, error_code="INSUFFICIENT_DATA", status_code=status_code, details=details)


class ModelTrainingError(ForecastError):
    """Raised when model training or inference fails."""
    def __init__(self, message: str, status_code: int = 500, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, error_code="MODEL_TRAINING_ERROR", status_code=status_code, details=details)


def _generate_synthetic_demo_data(lat: float, lon: float) -> tuple[pd.DataFrame, int]:
    """
    Explicit synthetic data generator used ONLY when allow_demo=True is requested.
    Generates simulated diurnal PM2.5 readings for demonstration purposes.
    """
    logger.warning(
        f"[DEMO_MODE] Generating synthetic demonstration data for coordinates ({lat}, {lon}). "
        "This data MUST be flagged as synthetic."
    )
    now_dt = datetime.now(timezone.utc).replace(tzinfo=None)
    start_dt = now_dt - timedelta(days=92)
    times_gen = [start_dt + timedelta(hours=i) for i in range(92 * 24)]
    n_gen = len(times_gen)
    hours_arr = np.array([t.hour for t in times_gen])
    diurnal = 16.0 + 6.0 * np.sin(2 * np.pi * (hours_arr - 6) / 24.0)
    synth_pm25 = np.maximum(2.0, np.round(diurnal, 1)).tolist()

    df = pd.DataFrame({
        "time": pd.to_datetime(times_gen),
        "pm25": synth_pm25,
        "pm2_5": synth_pm25,
        "no2": [12.0] * n_gen,
        "o3": [28.0] * n_gen,
        "wind_speed": [4.0] * n_gen,
        "temperature": [18.0] * n_gen,
        "humidity": [55.0] * n_gen,
    })
    return df, 0


async def fetch_historical_air_quality(
    lat: float,
    lon: float,
    allow_demo: bool = False
) -> tuple[pd.DataFrame, int, bool]:
    """
    Fetch up to 92 days of hourly PM2.5, NO2, O3, wind speed, temperature,
    and relative humidity from Open-Meteo Air Quality API.
    Returns (DataFrame, utc_offset_seconds, is_synthetic).
    
    If allow_demo=False (production default):
      Fails loudly with UpstreamProviderError or InsufficientDataError on failure.
    If allow_demo=True:
      Falls back to explicit synthetic simulation with is_synthetic=True.
    """
    lat, lon = normalize_coordinates(lat, lon)
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "pm2_5,nitrogen_dioxide,ozone,wind_speed_10m,temperature_2m,relative_humidity_2m",
        "past_days": 92,
        "forecast_days": 2,
        "timezone": "auto"
    }

    data = None
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.get(url, params=params)
            response.raise_for_status()
            data = response.json()
    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code if exc.response is not None else 502
        logger.error(
            f"Open-Meteo returned HTTP {status} for coordinates ({lat}, {lon})",
            extra={"latitude": lat, "longitude": lon, "status_code": status, "allow_demo": allow_demo}
        )
        if allow_demo:
            demo_df, offset = _generate_synthetic_demo_data(lat, lon)
            return demo_df, offset, True
        raise UpstreamProviderError(
            f"Upstream air quality provider returned HTTP {status}.",
            status_code=502
        ) from exc
    except (httpx.RequestError, httpx.TimeoutException) as exc:
        logger.error(
            f"Open-Meteo network/timeout failure for coordinates ({lat}, {lon}): {type(exc).__name__}",
            extra={"latitude": lat, "longitude": lon, "error_type": type(exc).__name__, "allow_demo": allow_demo}
        )
        if allow_demo:
            demo_df, offset = _generate_synthetic_demo_data(lat, lon)
            return demo_df, offset, True
        raise UpstreamProviderError(
            f"Upstream air quality provider is currently unreachable ({type(exc).__name__}).",
            status_code=502
        ) from exc
    except Exception as exc:
        logger.error(
            f"Unexpected error querying Open-Meteo for coordinates ({lat}, {lon}): {exc}",
            extra={"latitude": lat, "longitude": lon, "error_type": type(exc).__name__, "allow_demo": allow_demo}
        )
        if allow_demo:
            demo_df, offset = _generate_synthetic_demo_data(lat, lon)
            return demo_df, offset, True
        raise UpstreamProviderError(
            "An unexpected error occurred while contacting the upstream provider.",
            status_code=502
        ) from exc

    if not isinstance(data, dict) or "hourly" not in data:
        logger.error(f"Malformed payload from Open-Meteo for ({lat}, {lon})")
        if allow_demo:
            demo_df, offset = _generate_synthetic_demo_data(lat, lon)
            return demo_df, offset, True
        raise UpstreamProviderError("Upstream provider returned an invalid data payload.", status_code=502)

    hourly = data.get("hourly", {})
    if not isinstance(hourly, dict):
        if allow_demo:
            demo_df, offset = _generate_synthetic_demo_data(lat, lon)
            return demo_df, offset, True
        raise UpstreamProviderError("Malformed hourly object in upstream provider response.", status_code=502)

    times = hourly.get("time", [])
    pm25_vals = hourly.get("pm2_5", [])
    no2_vals = hourly.get("nitrogen_dioxide", [])
    o3_vals = hourly.get("ozone", [])
    wind_vals = hourly.get("wind_speed_10m", [])
    temp_vals = hourly.get("temperature_2m", [])
    humidity_vals = hourly.get("relative_humidity_2m", [])
    utc_offset_seconds = data.get("utc_offset_seconds", 0)

    if not times or not pm25_vals:
        logger.warning(f"Empty historical observations for coordinates ({lat}, {lon})")
        if allow_demo:
            demo_df, offset = _generate_synthetic_demo_data(lat, lon)
            return demo_df, offset, True
        raise InsufficientDataError(f"No historical PM2.5 observations returned for coordinates ({lat}, {lon}).", status_code=422)

    n = len(times)
    df = pd.DataFrame({
        "time": pd.to_datetime(times),
        "pm25": pm25_vals,
        "pm2_5": pm25_vals,
        "no2": no2_vals if no2_vals and len(no2_vals) == n else [np.nan] * n,
        "o3": o3_vals if o3_vals and len(o3_vals) == n else [np.nan] * n,
        "wind_speed": wind_vals if wind_vals and len(wind_vals) == n else [np.nan] * n,
        "temperature": temp_vals if temp_vals and len(temp_vals) == n else [np.nan] * n,
        "humidity": humidity_vals if humidity_vals and len(humidity_vals) == n else [np.nan] * n,
    })

    # Sort and clean data
    df = df.sort_values("time").reset_index(drop=True)

    # Convert columns to numeric
    numeric_cols = ["pm25", "pm2_5", "no2", "o3", "wind_speed", "temperature", "humidity"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Check for valid PM2.5 observations
    valid_pm25_count = int(df["pm25"].notna().sum())
    if valid_pm25_count < 50:
        logger.warning(
            f"Insufficient valid PM2.5 observations for ({lat}, {lon}): "
            f"{valid_pm25_count} valid rows found, minimum 50 required."
        )
        if allow_demo:
            demo_df, offset = _generate_synthetic_demo_data(lat, lon)
            return demo_df, offset, True
        raise InsufficientDataError(
            f"Insufficient historical PM2.5 observations for coordinates ({lat}, {lon}). "
            f"Found {valid_pm25_count} valid readings; minimum required is 50.",
            status_code=422
        )

    # Return raw sorted numeric DataFrame without cross-boundary interpolation.
    # Imputation is performed strictly on historical observations after cutoff partitioning!
    return df, utc_offset_seconds, False


def clean_and_impute_series(
    df: pd.DataFrame,
    min_valid_samples: int = 50,
    target_col: str = "pm25"
) -> pd.DataFrame:
    """
    Interpolate and impute missing observations strictly within the historical boundary.
    Must be called on historical data AFTER filtering to the historical cutoff
    so that future observations cannot leak backwards via bfill() or interpolation.
    """
    df = df.copy()
    col = target_col if target_col in df.columns else "pm2_5"
    valid_pm25 = int(df[col].notna().sum()) if col in df.columns else 0
    if valid_pm25 < min_valid_samples:
        raise InsufficientDataError(
            f"Insufficient valid PM2.5 observations in historical window ({valid_pm25} valid readings; minimum required is {min_valid_samples}).",
            status_code=422
        )

    # Linear interpolation strictly within historical observations,
    # followed by ffill() and bfill() strictly within this historical partition.
    df = df.interpolate(method="linear").ffill().bfill()

    # Fill auxiliary atmospheric defaults if completely missing
    if "no2" in df.columns and df["no2"].isna().all():
        df["no2"] = 15.0
    if "o3" in df.columns and df["o3"].isna().all():
        df["o3"] = 30.0
    if "wind_speed" in df.columns and df["wind_speed"].isna().all():
        df["wind_speed"] = 3.5
    if "temperature" in df.columns and df["temperature"].isna().all():
        df["temperature"] = 20.0
    if "humidity" in df.columns and df["humidity"].isna().all():
        df["humidity"] = 50.0

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Upgraded feature engineering module for PM2.5 forecasting:
    1. Cyclical sine/cosine encodings (hour_sin, hour_cos, dow_sin, dow_cos)
       replacing raw 'hour' and 'dow'
    2. Interaction features:
       - wind_pm25_ratio = pm25 / (wind_speed + 1)
       - humidity_temp_ratio = humidity / (temperature + 1)
       - no2_o3_ratio = no2 / (o3 + 1)
    3. Polynomial features:
       - pm25_lag1_squared = pm25_lag1 ** 2
       - pm25_lag24_squared = pm25_lag24 ** 2
    4. All existing lag and rolling features
    Ensures zero NaN values leak into the output training set.
    """
    df = df.copy()

    # 1. Cyclical sine/cosine encodings for hour and day of week
    hour = df["time"].dt.hour
    day_of_week = df["time"].dt.dayofweek

    df["hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
    df["hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)
    df["dow_sin"] = np.sin(2.0 * np.pi * day_of_week / 7.0)
    df["dow_cos"] = np.cos(2.0 * np.pi * day_of_week / 7.0)

    # 2. Existing lag features
    pm25_series = df["pm25"] if "pm25" in df.columns else df["pm2_5"]
    df["pm25_lag1"] = pm25_series.shift(1)
    df["pm25_lag2"] = pm25_series.shift(2)
    df["pm25_lag3"] = pm25_series.shift(3)
    df["pm25_lag6"] = pm25_series.shift(6)
    df["pm25_lag12"] = pm25_series.shift(12)
    df["pm25_lag24"] = pm25_series.shift(24)

    # Backwards compatibility aliases for lags
    df["lag_1"] = df["pm25_lag1"]
    df["lag_2"] = df["pm25_lag2"]
    df["lag_3"] = df["pm25_lag3"]
    df["lag_6"] = df["pm25_lag6"]
    df["lag_12"] = df["pm25_lag12"]
    df["lag_24"] = df["pm25_lag24"]

    # 3. Polynomial features for most important lags
    df["pm25_lag1_squared"] = df["pm25_lag1"] ** 2
    df["pm25_lag24_squared"] = df["pm25_lag24"] ** 2

    # 4. Interaction features with numerical stability protections
    wind = df["wind_speed"] if "wind_speed" in df.columns else 3.5
    temp = df["temperature"] if "temperature" in df.columns else 20.0
    humidity = df["humidity"] if "humidity" in df.columns else 50.0
    no2 = df["no2"] if "no2" in df.columns else 15.0
    o3 = df["o3"] if "o3" in df.columns else 30.0

    # Use pm25_lag1 (y_{t-1}) to prevent target leakage
    df["wind_pm25_ratio"] = df["pm25_lag1"] / (np.maximum(wind, 0.0) + 1.0)
    
    # Avoid zero-division if temperature == -1.0
    temp_denom = temp + 1.0
    temp_denom = np.where(np.abs(temp_denom) < 1e-4, 1e-4, temp_denom)
    df["humidity_temp_ratio"] = humidity / temp_denom

    df["no2_o3_ratio"] = no2 / (np.maximum(o3, 0.0) + 1.0)

    # 5. Existing rolling window features based on shifted PM2.5
    shifted_pm25 = pm25_series.shift(1)
    df["rolling_mean_6"] = shifted_pm25.rolling(window=6, min_periods=1).mean()
    df["rolling_mean_24"] = shifted_pm25.rolling(window=24, min_periods=1).mean()
    df["rolling_std_24"] = shifted_pm25.rolling(window=24, min_periods=1).std().fillna(0.0)

    # Replace any potential infinites with NaN
    df = df.replace([np.inf, -np.inf], np.nan)

    return df


# Backwards compatibility alias
build_features = engineer_features


def get_feature_columns() -> List[str]:
    """
    Returns the complete feature column list.
    Raw 'hour' and 'dow' are replaced by cyclical sine/cosine encodings.
    """
    return [
        # Cyclical temporal encodings (replacing hour & dow)
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
        # Interaction features
        "wind_pm25_ratio",
        "humidity_temp_ratio",
        "no2_o3_ratio",
        # Polynomial lag features
        "pm25_lag1_squared",
        "pm25_lag24_squared",
        # Existing lag features
        "pm25_lag1",
        "pm25_lag2",
        "pm25_lag3",
        "pm25_lag6",
        "pm25_lag12",
        "pm25_lag24",
        # Existing rolling window features
        "rolling_mean_6",
        "rolling_mean_24",
        "rolling_std_24"
    ]


def predict_autoregressive_rollout(
    model: xgb.XGBRegressor,
    history_df: pd.DataFrame,
    feature_cols: List[str],
    steps: int = 24,
    future_weather_df: Optional[pd.DataFrame] = None
) -> List[float]:
    """
    Autoregressively roll out predictions over `steps` hours.
    At each step h, lag and rolling features are derived strictly from history
    and previous model predictions (zero future data leakage).
    Optionally incorporates genuine future meteorological forecasts for prediction hours.
    """
    last_row = history_df.iloc[-1]
    last_time = last_row["time"]

    last_wind = float(last_row.get("wind_speed", 3.5))
    last_temp = float(last_row.get("temperature", 20.0))
    last_humidity = float(last_row.get("humidity", 50.0))
    last_no2 = float(last_row.get("no2", 15.0))
    last_o3 = float(last_row.get("o3", 30.0))

    target_series = history_df["pm25"] if "pm25" in history_df.columns else history_df["pm2_5"]
    pm_history = [float(x) for x in target_series.values]

    preds: List[float] = []

    for step in range(1, steps + 1):
        next_time = last_time + timedelta(hours=step)
        hour = next_time.hour
        day_of_week = next_time.weekday()

        # Dynamic future weather if available for the corresponding prediction hour;
        # otherwise gracefully defaults to the latest observed values.
        if future_weather_df is not None and not future_weather_df.empty:
            matching_rows = future_weather_df[future_weather_df["time"] == next_time]
            if not matching_rows.empty:
                w_row = matching_rows.iloc[0]
                step_wind = float(w_row.get("wind_speed", last_wind)) if pd.notna(w_row.get("wind_speed")) else last_wind
                step_temp = float(w_row.get("temperature", last_temp)) if pd.notna(w_row.get("temperature")) else last_temp
                step_humidity = float(w_row.get("humidity", last_humidity)) if pd.notna(w_row.get("humidity")) else last_humidity
                step_no2 = float(w_row.get("no2", last_no2)) if pd.notna(w_row.get("no2")) else last_no2
                step_o3 = float(w_row.get("o3", last_o3)) if pd.notna(w_row.get("o3")) else last_o3
            else:
                step_wind, step_temp, step_humidity, step_no2, step_o3 = last_wind, last_temp, last_humidity, last_no2, last_o3
        else:
            step_wind, step_temp, step_humidity, step_no2, step_o3 = last_wind, last_temp, last_humidity, last_no2, last_o3

        temp_denom = step_temp + 1.0 if abs(step_temp + 1.0) > 1e-4 else 1e-4
        humidity_temp_ratio = float(step_humidity / temp_denom)
        no2_o3_ratio = float(step_no2 / (max(0.0, step_o3) + 1.0))
        wind_denom = max(0.0, step_wind) + 1.0

        hour_sin = math.sin(2.0 * math.pi * hour / 24.0)
        hour_cos = math.cos(2.0 * math.pi * hour / 24.0)
        dow_sin = math.sin(2.0 * math.pi * day_of_week / 7.0)
        dow_cos = math.cos(2.0 * math.pi * day_of_week / 7.0)

        pm25_lag1 = pm_history[-1]
        pm25_lag2 = pm_history[-2] if len(pm_history) >= 2 else pm25_lag1
        pm25_lag3 = pm_history[-3] if len(pm_history) >= 3 else pm25_lag2
        pm25_lag6 = pm_history[-6] if len(pm_history) >= 6 else pm25_lag3
        pm25_lag12 = pm_history[-12] if len(pm_history) >= 12 else pm25_lag6
        pm25_lag24 = pm_history[-24] if len(pm_history) >= 24 else pm25_lag12

        pm25_lag1_squared = float(pm25_lag1 ** 2)
        pm25_lag24_squared = float(pm25_lag24 ** 2)

        wind_pm25_ratio = float(pm25_lag1 / wind_denom)

        recent_6 = pm_history[-6:]
        recent_24 = pm_history[-24:]
        rolling_mean_6 = float(sum(recent_6) / len(recent_6))
        rolling_mean_24 = float(sum(recent_24) / len(recent_24))
        if len(recent_24) > 1:
            var_24 = sum((x - rolling_mean_24) ** 2 for x in recent_24) / len(recent_24)
            rolling_std_24 = float(math.sqrt(var_24))
        else:
            rolling_std_24 = 0.0

        feature_map = {
            "hour_sin": hour_sin,
            "hour_cos": hour_cos,
            "dow_sin": dow_sin,
            "dow_cos": dow_cos,
            "wind_pm25_ratio": wind_pm25_ratio,
            "humidity_temp_ratio": humidity_temp_ratio,
            "no2_o3_ratio": no2_o3_ratio,
            "pm25_lag1_squared": pm25_lag1_squared,
            "pm25_lag24_squared": pm25_lag24_squared,
            "pm25_lag1": pm25_lag1,
            "pm25_lag2": pm25_lag2,
            "pm25_lag3": pm25_lag3,
            "pm25_lag6": pm25_lag6,
            "pm25_lag12": pm25_lag12,
            "pm25_lag24": pm25_lag24,
            "rolling_mean_6": rolling_mean_6,
            "rolling_mean_24": rolling_mean_24,
            "rolling_std_24": rolling_std_24
        }
        feat_vector = np.array([[feature_map[c] for c in feature_cols]], dtype=np.float32)

        pred_val = float(model.predict(feat_vector)[0])
        pred_val = max(1.0, round(pred_val, 1))
        preds.append(pred_val)
        pm_history.append(pred_val)

    return preds


def compute_prediction_interval(
    pred_val: float,
    step: int,
    conformal_quantiles: Optional[Dict[int, float]] = None,
    rolling_std_24: float = 0.0
) -> Tuple[float, float, str]:
    """
    Computes lower bound, upper bound, and interval method for a forecast step.
    
    Statistical Properties:
    - Split Conformal Prediction (when conformal_quantiles is provided):
      Provides finite-sample marginal coverage guarantees P(Y_{t+h} in [l, u]) >= 1 - alpha
      under exchangeable residuals. Quantiles q^{(h)} are calibrated on an independent
      chronological holdout partition.
    - Heuristic Uncertainty Band (fallback):
      Derived from step index, prediction magnitude, and recent rolling variance:
      margin = max(2.5, round((0.15 + (step * 0.015)) * pred_val + (0.2 * rolling_std_24), 1)).
      Clearly flagged as 'estimated_uncertainty_heuristic' without false confidence claims.
      
    Physical Validity:
    - Lower bound is clamped to max(0.0, ...) because PM2.5 concentrations are strictly non-negative.
    - Upper bound is guaranteed >= lower bound under all inputs.
    """
    if conformal_quantiles is not None and step in conformal_quantiles:
        margin = float(conformal_quantiles[step])
        method = "split_conformal_prediction"
    else:
        margin = max(2.5, round((0.15 + (step * 0.015)) * pred_val + (0.2 * rolling_std_24), 1))
        method = "estimated_uncertainty_heuristic"

    lower_bound = max(0.0, round(pred_val - margin, 1))
    upper_bound = max(lower_bound, round(pred_val + margin, 1))
    return lower_bound, upper_bound, method


def calibrate_conformal_quantiles(
    model: xgb.XGBRegressor,
    history_df: pd.DataFrame,
    calib_start_idx: int,
    calib_end_idx: int,
    feature_cols: List[str],
    target_col: str = "pm25",
    horizon: int = 24,
    alpha: float = 0.10,
    stride_hours: int = 12,
    max_origins: Optional[int] = 30
) -> Tuple[Dict[int, float], int]:
    """
    Calibrate horizon-specific split conformal quantiles on an independent holdout history partition.
    
    Parameters:
    - model: Fitted XGBoost regressor (trained strictly on proper training partition).
    - history_df: Continuous chronological time series.
    - calib_start_idx, calib_end_idx: Chronological slice boundaries for calibration.
    - horizon: Forecast steps (e.g. 24 hours).
    - alpha: Miscoverage level (e.g. 0.10 for 90% coverage).
    - stride_hours: Spacing between calibration evaluation origins (default: 12 hours).
    - max_origins: Optional upper bound on evaluation origins to ensure fast inference on cloud vCPU.
    
    Returns:
    - Dictionary mapping step h -> calibrated conformal radius q_{1-alpha}^{(h)}.
    - Number of valid calibration windows evaluated.
    """
    origins = list(range(calib_start_idx, calib_end_idx - horizon, stride_hours))
    if max_origins is not None and len(origins) > max_origins:
        step_sz = len(origins) / max_origins
        origins = [origins[int(i * step_sz)] for i in range(max_origins)]

    if len(origins) < 5:
        logger.warning(f"Insufficient calibration origins ({len(origins)}); falling back to heuristic.")
        return {}, 0

    residuals_by_h: Dict[int, List[float]] = {h: [] for h in range(1, horizon + 1)}

    for origin in origins:
        history_to_origin = history_df.iloc[:origin + 1]
        gt = history_df.iloc[origin + 1 : origin + 1 + horizon][target_col].values
        if len(gt) < horizon:
            continue

        preds = predict_autoregressive_rollout(model, history_to_origin, feature_cols, steps=horizon)
        for h in range(1, horizon + 1):
            residuals_by_h[h].append(abs(float(gt[h - 1]) - float(preds[h - 1])))

    quantiles: Dict[int, float] = {}
    prev_q = 0.0
    valid_origins_count = len(residuals_by_h[1]) if 1 in residuals_by_h else 0

    if valid_origins_count == 0:
        return {}, 0

    for h in range(1, horizon + 1):
        res = residuals_by_h[h]
        if not res:
            quantiles[h] = prev_q
            continue
        n = len(res)
        # Finite-sample split conformal quantile: ceil((n + 1) * (1 - alpha)) / n
        p = min(1.0, np.ceil((n + 1) * (1.0 - alpha)) / n)
        q = float(np.quantile(res, p))
        # Enforce non-decreasing quantiles with horizon: uncertainty grows with lead time
        q = max(q, prev_q)
        prev_q = q
        quantiles[h] = round(q, 2)

    return quantiles, valid_origins_count


async def train_and_forecast_pm25(lat: float, lon: float, allow_demo: bool = False) -> Dict[str, Any]:
    """
    Fetch 92 days of atmospheric and meteorological data, train an XGBoost model,
    and autoregressively forecast the next 24 hours of PM2.5.
    Uses in-memory caching with a 30-minute TTL and coordinate serialization locks.
    """
    norm_lat, norm_lon = normalize_coordinates(lat, lon)
    cache_key = (norm_lat, norm_lon, allow_demo)
    now = time.time()

    if cache_key in MODEL_CACHE:
        cached = MODEL_CACHE[cache_key]
        if now - cached["timestamp"] < CACHE_TTL_SECONDS:
            return cached["data"]

    coord_lock = await get_coordinate_lock(norm_lat, norm_lon)
    async with coord_lock:
        now = time.time()
        if cache_key in MODEL_CACHE:
            cached = MODEL_CACHE[cache_key]
            if now - cached["timestamp"] < CACHE_TTL_SECONDS:
                return cached["data"]

        t_pipeline_start = time.perf_counter()

        # 1. Fetch 92 days of hourly data (PM2.5, NO2, O3, Wind Speed, Temperature, Humidity)
        t_fetch_start = time.perf_counter()
        df, utc_offset_seconds, is_synthetic = await fetch_historical_air_quality(norm_lat, norm_lon, allow_demo=allow_demo)
        fetch_seconds = round(time.perf_counter() - t_fetch_start, 3)

        # Ensure df["time"] is timezone-naive for safe comparison against local naive timestamps
        if hasattr(df["time"].dtype, "tz") and df["time"].dt.tz is not None:
            df["time"] = df["time"].dt.tz_localize(None)

        # Determine current timestamp aligned to Open-Meteo's timezone
        if utc_offset_seconds != 0:
            loc_tz = timezone(timedelta(seconds=utc_offset_seconds))
            now_local = datetime.now(loc_tz).replace(tzinfo=None)
        else:
            # Fallback to UTC if no offset
            now_local = datetime.now(timezone.utc).replace(tzinfo=None)

        # 1. STRICT SEPARATION AT HISTORICAL CUTOFF BEFORE ANY IMPUTATION
        # Strictly isolate observations at or before current time
        df_past_raw = df[df["time"] <= now_local].copy().reset_index(drop=True)
        df_future_raw = df[df["time"] > now_local].copy().reset_index(drop=True)

        if df_past_raw.empty:
            # Fallback: use the most recent row with non-null pm25
            current_row = df.dropna(subset=["pm25"]).iloc[-1]
            last_idx = df.index[df["time"] == current_row["time"]].tolist()[0]
            df_past_raw = df.iloc[:last_idx + 1].copy().reset_index(drop=True)
            df_future_raw = df.iloc[last_idx + 1:].copy().reset_index(drop=True)

        # 2. Impute and clean strictly within the historical boundary
        # bfill() and interpolation cannot peek into future observations!
        current_time_series = clean_and_impute_series(df_past_raw, min_valid_samples=50)

        # 3. Clean future weather partition separately (without contaminating past)
        if not df_future_raw.empty:
            df_future_weather = df_future_raw.interpolate(method="linear").ffill().bfill()
        else:
            df_future_weather = None

        current_row = current_time_series.iloc[-1]
        last_timestamp = current_row["time"]
        last_wind = float(current_row["wind_speed"])
        last_temp = float(current_row["temperature"])
        last_humidity = float(current_row["humidity"])
        last_no2 = float(current_row["no2"])
        last_o3 = float(current_row["o3"])
        current_pm25_val = float(current_row["pm25"])

        # 2. Build feature matrix using upgraded engineer_features on historical past data
        t_feat_start = time.perf_counter()
        df_feat = engineer_features(current_time_series)
        feature_cols = get_feature_columns()

        # 3. Ensure no NaN values leak into the training set
        target_col = "pm25" if "pm25" in df_feat.columns else "pm2_5"
        train_df = df_feat.dropna(subset=feature_cols + [target_col]).reset_index(drop=True)
        train_df = train_df.replace([np.inf, -np.inf], np.nan).dropna(subset=feature_cols).reset_index(drop=True)
        feature_engineering_seconds = round(time.perf_counter() - t_feat_start, 3)

        if len(train_df) < 50:
            logger.warning(f"Training dataset too small ({len(train_df)} samples) for ({norm_lat}, {norm_lon})")
            raise InsufficientDataError(
                f"Insufficient historical data to train the forecasting model (minimum 50 samples required, got {len(train_df)}).",
                status_code=422
            )

        # 4. Train XGBoost Regressor and Calibrate Prediction Intervals
        conformal_quantiles: Optional[Dict[int, float]] = None
        calibration_samples_count: int = 0
        interval_method: str = "estimated_uncertainty_heuristic"
        nominal_coverage: Optional[float] = None
        interval_label: str = "Estimated Uncertainty Band (uncalibrated heuristic)"

        t_train_start = time.perf_counter()
        if not is_synthetic and len(train_df) >= 100:
            # Strict Chronological Partition: 80% Proper Train, 20% Conformal Calibration
            proper_train_len = int(len(train_df) * 0.80)
            proper_train_df = train_df.iloc[:proper_train_len]

            X_train = proper_train_df[feature_cols]
            y_train = proper_train_df[target_col]

            # Verify no NaN values in proper training set
            if X_train.isna().any().any() or y_train.isna().any():
                logger.error(f"NaN values detected in training set feature matrix for ({norm_lat}, {norm_lon})")
                raise ModelTrainingError("NaN values detected in training set feature matrix.", status_code=500)

            try:
                model = xgb.XGBRegressor(
                    n_estimators=120,
                    max_depth=4,
                    learning_rate=0.08,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    random_state=42,
                    n_jobs=-1
                )
                model.fit(X_train, y_train)
            except Exception as exc:
                logger.error(f"XGBoost training failure for ({norm_lat}, {norm_lon}): {exc}")
                raise ModelTrainingError("Model training failed on historical observations.", status_code=500) from exc
            train_seconds = round(time.perf_counter() - t_train_start, 3)

            # Check Conformal Calibration Cache safely using coordinates, data/version identity, and TTL
            t_calib_start = time.perf_counter()
            data_version = f"{last_timestamp.strftime('%Y%m%d%H')}_{MODEL_VERSION}"
            calib_version_key = (norm_lat, norm_lon, data_version)
            calib_coord_key = (norm_lat, norm_lon)
            calib_hit = False

            if calib_version_key in CALIBRATION_CACHE:
                cached_calib = CALIBRATION_CACHE[calib_version_key]
                if time.time() - cached_calib["timestamp"] < CALIBRATION_TTL_SECONDS:
                    conformal_quantiles = cached_calib["quantiles"]
                    calibration_samples_count = cached_calib["calibration_samples"]
                    calib_hit = True
                    logger.info(f"Reusing cached conformal calibration quantiles for ({norm_lat}, {norm_lon}) [version={data_version}]")
            elif calib_coord_key in CALIBRATION_CACHE:
                cached_calib = CALIBRATION_CACHE[calib_coord_key]
                if time.time() - cached_calib["timestamp"] < CALIBRATION_TTL_SECONDS:
                    conformal_quantiles = cached_calib["quantiles"]
                    calibration_samples_count = cached_calib["calibration_samples"]
                    calib_hit = True
                    logger.info(f"Reusing cached conformal calibration quantiles for ({norm_lat}, {norm_lon})")

            if not calib_hit:
                conformal_quantiles, calibration_samples_count = calibrate_conformal_quantiles(
                    model=model,
                    history_df=train_df,
                    calib_start_idx=proper_train_len,
                    calib_end_idx=len(train_df),
                    feature_cols=feature_cols,
                    target_col=target_col,
                    horizon=24,
                    alpha=0.10,
                    stride_hours=12,
                    max_origins=30
                )
                if conformal_quantiles:
                    calib_record = {
                        "timestamp": time.time(),
                        "quantiles": conformal_quantiles,
                        "calibration_samples": calibration_samples_count,
                        "data_version": data_version
                    }
                    CALIBRATION_CACHE[calib_version_key] = calib_record
                    CALIBRATION_CACHE[calib_coord_key] = calib_record

            if conformal_quantiles:
                interval_method = "split_conformal_prediction"
                nominal_coverage = 0.90
                interval_label = "90% Conformal Prediction Interval (calibrated on holdout history)"
                logger.info(f"Calibrated 90% conformal intervals across {calibration_samples_count} holdout windows for ({norm_lat}, {norm_lon})")
            calibration_seconds = round(time.perf_counter() - t_calib_start, 3)
        else:
            # For synthetic demo data or smaller datasets (<100 samples), train on full train_df and use heuristic
            X_train = train_df[feature_cols]
            y_train = train_df[target_col]

            if X_train.isna().any().any() or y_train.isna().any():
                logger.error(f"NaN values detected in training set feature matrix for ({norm_lat}, {norm_lon})")
                raise ModelTrainingError("NaN values detected in training set feature matrix.", status_code=500)

            try:
                model = xgb.XGBRegressor(
                    n_estimators=120,
                    max_depth=4,
                    learning_rate=0.08,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    random_state=42,
                    n_jobs=-1
                )
                model.fit(X_train, y_train)
            except Exception as exc:
                logger.error(f"XGBoost training failure for ({norm_lat}, {norm_lon}): {exc}")
                raise ModelTrainingError("Model training failed on historical observations.", status_code=500) from exc
            train_seconds = round(time.perf_counter() - t_train_start, 3)
            calibration_seconds = 0.0

        # 5. Multi-step Autoregressive Forecasting for next 24 hours
        t_forecast_start = time.perf_counter()
        predictions: List[Dict[str, Any]] = []

        pm_forecast_history = [float(v) for v in current_time_series["pm25"].values]

        for step in range(1, 25):
            next_time = last_timestamp + timedelta(hours=step)
            hour = next_time.hour
            day_of_week = next_time.weekday()

            # Dynamic future weather if available for the corresponding prediction hour;
            # otherwise gracefully defaults to the latest observed values.
            if df_future_weather is not None and not df_future_weather.empty:
                matching_rows = df_future_weather[df_future_weather["time"] == next_time]
                if not matching_rows.empty:
                    w_row = matching_rows.iloc[0]
                    step_wind = float(w_row.get("wind_speed", last_wind)) if pd.notna(w_row.get("wind_speed")) else last_wind
                    step_temp = float(w_row.get("temperature", last_temp)) if pd.notna(w_row.get("temperature")) else last_temp
                    step_humidity = float(w_row.get("humidity", last_humidity)) if pd.notna(w_row.get("humidity")) else last_humidity
                    step_no2 = float(w_row.get("no2", last_no2)) if pd.notna(w_row.get("no2")) else last_no2
                    step_o3 = float(w_row.get("o3", last_o3)) if pd.notna(w_row.get("o3")) else last_o3
                else:
                    step_wind, step_temp, step_humidity, step_no2, step_o3 = last_wind, last_temp, last_humidity, last_no2, last_o3
            else:
                step_wind, step_temp, step_humidity, step_no2, step_o3 = last_wind, last_temp, last_humidity, last_no2, last_o3

            temp_denom = step_temp + 1.0 if abs(step_temp + 1.0) > 1e-4 else 1e-4
            humidity_temp_ratio = float(step_humidity / temp_denom)
            no2_o3_ratio = float(step_no2 / (max(0.0, step_o3) + 1.0))
            wind_denom = max(0.0, step_wind) + 1.0

            # Cyclical temporal encodings
            hour_sin = math.sin(2.0 * math.pi * hour / 24.0)
            hour_cos = math.cos(2.0 * math.pi * hour / 24.0)
            dow_sin = math.sin(2.0 * math.pi * day_of_week / 7.0)
            dow_cos = math.cos(2.0 * math.pi * day_of_week / 7.0)

            # Lags from autoregressively updated history
            pm25_lag1 = pm_forecast_history[-1]
            pm25_lag2 = pm_forecast_history[-2] if len(pm_forecast_history) >= 2 else pm25_lag1
            pm25_lag3 = pm_forecast_history[-3] if len(pm_forecast_history) >= 3 else pm25_lag2
            pm25_lag6 = pm_forecast_history[-6] if len(pm_forecast_history) >= 6 else pm25_lag3
            pm25_lag12 = pm_forecast_history[-12] if len(pm_forecast_history) >= 12 else pm25_lag6
            pm25_lag24 = pm_forecast_history[-24] if len(pm_forecast_history) >= 24 else pm25_lag12

            # Polynomial features
            pm25_lag1_squared = float(pm25_lag1 ** 2)
            pm25_lag24_squared = float(pm25_lag24 ** 2)

            # Interaction features
            wind_pm25_ratio = float(pm25_lag1 / wind_denom)

            # Rolling statistics
            recent_6 = pm_forecast_history[-6:]
            recent_24 = pm_forecast_history[-24:]
            rolling_mean_6 = float(sum(recent_6) / len(recent_6))
            rolling_mean_24 = float(sum(recent_24) / len(recent_24))
            if len(recent_24) > 1:
                var_24 = sum((x - rolling_mean_24) ** 2 for x in recent_24) / len(recent_24)
                rolling_std_24 = float(math.sqrt(var_24))
            else:
                rolling_std_24 = 0.0

            feature_map = {
                "hour_sin": hour_sin,
                "hour_cos": hour_cos,
                "dow_sin": dow_sin,
                "dow_cos": dow_cos,
                "wind_pm25_ratio": wind_pm25_ratio,
                "humidity_temp_ratio": humidity_temp_ratio,
                "no2_o3_ratio": no2_o3_ratio,
                "pm25_lag1_squared": pm25_lag1_squared,
                "pm25_lag24_squared": pm25_lag24_squared,
                "pm25_lag1": pm25_lag1,
                "pm25_lag2": pm25_lag2,
                "pm25_lag3": pm25_lag3,
                "pm25_lag6": pm25_lag6,
                "pm25_lag12": pm25_lag12,
                "pm25_lag24": pm25_lag24,
                "rolling_mean_6": rolling_mean_6,
                "rolling_mean_24": rolling_mean_24,
                "rolling_std_24": rolling_std_24
            }
            feat_vector = np.array([[feature_map[col] for col in feature_cols]], dtype=np.float32)

            pred_val = float(model.predict(feat_vector)[0])
            pred_val = max(1.0, round(pred_val, 1))

            # Prediction interval calculation
            lower_bound, upper_bound, step_method = compute_prediction_interval(
                pred_val=pred_val,
                step=step,
                conformal_quantiles=conformal_quantiles,
                rolling_std_24=rolling_std_24
            )

            predictions.append({
                "step": step,
                "time": next_time.strftime("%Y-%m-%dT%H:%M"),
                "display_time": next_time.strftime("%I:%M %p"),
                "display_date": next_time.strftime("%b %d"),
                "predicted_pm25": pred_val,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "interval_method": step_method
            })

            # Append step prediction for subsequent lags
            pm_forecast_history.append(pred_val)

        forecast_seconds = round(time.perf_counter() - t_forecast_start, 3)
        total_seconds = round(time.perf_counter() - t_pipeline_start, 3)

        logger.info(
            f"Forecast pipeline timings for ({norm_lat}, {norm_lon}): "
            f"fetch={fetch_seconds}s, features={feature_engineering_seconds}s, "
            f"train={train_seconds}s, calib={calibration_seconds}s, "
            f"forecast={forecast_seconds}s, total={total_seconds}s"
        )

        # Calculate basic summary
        avg_pred = round(float(np.mean([p["predicted_pm25"] for p in predictions])), 1)
        min_pred = min(p["predicted_pm25"] for p in predictions)
        max_pred = max(p["predicted_pm25"] for p in predictions)
        latest_history = round(current_pm25_val, 1)

        result = {
            "status": "success",
            "latitude": lat,
            "longitude": lon,
            "training_samples": len(train_df),
            "days_trained": 92,
            "current_time": last_timestamp.strftime("%Y-%m-%dT%H:%M"),
            "current_display_time": last_timestamp.strftime("%I:%M %p"),
            "current_pm25": latest_history,
            "forecast_average": avg_pred,
            "forecast_min": min_pred,
            "forecast_max": max_pred,
            "forecast": predictions,
            "interval_method": interval_method,
            "nominal_coverage": nominal_coverage,
            "interval_label": interval_label,
            "calibration_samples": calibration_samples_count,
            "is_synthetic": is_synthetic,
            "data_source": "synthetic_demo" if is_synthetic else "live_open_meteo",
            "timings": {
                "fetch_seconds": fetch_seconds,
                "feature_engineering_seconds": feature_engineering_seconds,
                "train_seconds": train_seconds,
                "calibration_seconds": calibration_seconds,
                "forecast_seconds": forecast_seconds,
                "total_seconds": total_seconds,
            }
        }

        if is_synthetic:
            result["warning"] = "Demonstration data generated synthetically because upstream observations are unavailable."

        # Cache trained model AND prediction result
        MODEL_CACHE[cache_key] = {
            "timestamp": time.time(),
            "model": model,
            "data": result
        }

        return result
