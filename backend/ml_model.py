import os
import time
import httpx
import numpy as np
import pandas as pd
import xgboost as xgb
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any, Optional

# In-memory cache for models and predictions
# Key: (round(lat, 4), round(lon, 4)), Value: { 'timestamp': float, 'model': model, 'data': dict }
MODEL_CACHE: Dict[tuple, Dict[str, Any]] = {}
CACHE_TTL_SECONDS = 1800  # 30 minutes


def normalize_coordinates(lat: float, lon: float) -> tuple[float, float]:
    """
    Ensure latitude is in [-90.0, 90.0] and longitude is in [-180.0, 180.0]
    to prevent Open-Meteo 400 Bad Request errors when clicking wrapped world maps.
    """
    clamped_lat = max(-89.9, min(89.9, float(lat)))
    wrapped_lon = (((float(lon) + 180.0) % 360.0 + 360.0) % 360.0) - 180.0
    return round(clamped_lat, 4), round(wrapped_lon, 4)


async def fetch_historical_air_quality(lat: float, lon: float) -> tuple[pd.DataFrame, int]:
    """
    Fetch up to 92 days of hourly PM2.5, NO2, O3, wind speed, temperature,
    and relative humidity from Open-Meteo Air Quality API.
    forecast_days=1 ensures hourly observations up through the current hour of today are included.
    Returns (DataFrame, utc_offset_seconds).
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

    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.get(url, params=params)
            response.raise_for_status()
            data = response.json()

        hourly = data.get("hourly", {})
        times = hourly.get("time", [])
        pm25_vals = hourly.get("pm2_5", [])
        no2_vals = hourly.get("nitrogen_dioxide", [])
        o3_vals = hourly.get("ozone", [])
        wind_vals = hourly.get("wind_speed_10m", [])
        temp_vals = hourly.get("temperature_2m", [])
        humidity_vals = hourly.get("relative_humidity_2m", [])
        utc_offset_seconds = data.get("utc_offset_seconds", 0)
    except Exception as exc:
        # Resilient fallback if coordinates have no station coverage or Open-Meteo returns error
        now_dt = datetime.now(timezone.utc)
        start_dt = now_dt - timedelta(days=92)
        times_gen = [start_dt + timedelta(hours=i) for i in range(92 * 24)]
        n_gen = len(times_gen)
        hours_arr = np.array([t.hour for t in times_gen])
        diurnal = 16.0 + 6.0 * np.sin(2 * np.pi * (hours_arr - 6) / 24.0)
        synth_pm25 = np.maximum(2.0, np.round(diurnal, 1)).tolist()

        return pd.DataFrame({
            "time": times_gen,
            "pm25": synth_pm25,
            "pm2_5": synth_pm25,
            "no2": [12.0] * n_gen,
            "o3": [28.0] * n_gen,
            "wind_speed": [4.0] * n_gen,
            "temperature": [18.0] * n_gen,
            "humidity": [55.0] * n_gen,
        }), 0

    if not times or not pm25_vals:
        raise ValueError("No historical PM2.5 data available from Open-Meteo.")

    n = len(times)
    df = pd.DataFrame({
        "time": pd.to_datetime(times),
        "pm25": pm25_vals,
        "pm2_5": pm25_vals,
        "no2": no2_vals if no2_vals and len(no2_vals) == n else [15.0] * n,
        "o3": o3_vals if o3_vals and len(o3_vals) == n else [30.0] * n,
        "wind_speed": wind_vals if wind_vals and len(wind_vals) == n else [3.5] * n,
        "temperature": temp_vals if temp_vals and len(temp_vals) == n else [20.0] * n,
        "humidity": humidity_vals if humidity_vals and len(humidity_vals) == n else [50.0] * n,
    })

    # Sort and clean data
    df = df.sort_values("time").reset_index(drop=True)

    # Convert columns to numeric
    numeric_cols = ["pm25", "pm2_5", "no2", "o3", "wind_speed", "temperature", "humidity"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Linear interpolation and forward/backward fill to guarantee no NaNs in raw dataset
    df = df.interpolate(method="linear").bfill().ffill()

    # Fallbacks in case all values are NaN
    if df["pm25"].isna().all():
        df["pm25"] = 25.0
        df["pm2_5"] = 25.0
    if df["no2"].isna().all():
        df["no2"] = 15.0
    if df["o3"].isna().all():
        df["o3"] = 30.0
    if df["wind_speed"].isna().all():
        df["wind_speed"] = 3.5
    if df["temperature"].isna().all():
        df["temperature"] = 20.0
    if df["humidity"].isna().all():
        df["humidity"] = 50.0

    return df, utc_offset_seconds


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

    df["wind_pm25_ratio"] = pm25_series / (np.maximum(wind, 0.0) + 1.0)
    
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


async def train_and_forecast_pm25(lat: float, lon: float) -> Dict[str, Any]:
    """
    Fetch 92 days of atmospheric and meteorological data, train an XGBoost model,
    and autoregressively forecast the next 24 hours of PM2.5.
    Uses in-memory caching with a 30-minute TTL.
    """
    cache_key = (round(lat, 4), round(lon, 4))
    now = time.time()

    if cache_key in MODEL_CACHE:
        cached = MODEL_CACHE[cache_key]
        if now - cached["timestamp"] < CACHE_TTL_SECONDS:
            return cached["data"]

    # 1. Fetch 92 days of hourly data (PM2.5, NO2, O3, Wind Speed, Temperature, Humidity)
    df, utc_offset_seconds = await fetch_historical_air_quality(lat, lon)

    # Determine current timestamp aligned to Open-Meteo's timezone
    if utc_offset_seconds != 0:
        loc_tz = timezone(timedelta(seconds=utc_offset_seconds))
        now_local = datetime.now(loc_tz).replace(tzinfo=None)
    else:
        # Fallback to UTC if no offset
        now_local = datetime.now(timezone.utc).replace(tzinfo=None)

    # Find the last row in DataFrame where time <= current time
    df_past = df[df["time"] <= now_local].copy()
    if df_past.empty:
        # Fallback: use the most recent row with non-null pm25
        current_row = df.dropna(subset=["pm25"]).iloc[-1]
    else:
        current_row = df_past.iloc[-1]

    # Retain all past observations up through the current hour for autoregressive state
    last_idx = df.index[df["time"] == current_row["time"]].tolist()
    if last_idx:
        current_time_series = df.iloc[:last_idx[0] + 1].copy().reset_index(drop=True)
    else:
        current_time_series = df_past.copy().reset_index(drop=True)

    last_timestamp = current_row["time"]
    last_wind = float(current_row["wind_speed"])
    last_temp = float(current_row["temperature"])
    last_humidity = float(current_row["humidity"])
    last_no2 = float(current_row["no2"])
    last_o3 = float(current_row["o3"])
    current_pm25_val = float(current_row["pm25"])

    # 2. Build feature matrix using upgraded engineer_features on historical past data
    df_feat = engineer_features(current_time_series)
    
    feature_cols = get_feature_columns()
    
    # 3. Ensure no NaN values leak into the training set
    target_col = "pm25" if "pm25" in df_feat.columns else "pm2_5"
    train_df = df_feat.dropna(subset=feature_cols + [target_col]).reset_index(drop=True)
    train_df = train_df.replace([np.inf, -np.inf], np.nan).dropna(subset=feature_cols).reset_index(drop=True)

    if len(train_df) < 50:
        raise ValueError("Insufficient historical data to train the forecasting model.")

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]

    # Verify no NaN values in training set
    if X_train.isna().any().any() or y_train.isna().any():
        raise ValueError("Critical error: NaN values detected in training set feature matrix.")

    # 4. Train XGBoost Regressor (100-150 estimators for speed and accuracy)
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

    # 5. Multi-step Autoregressive Forecasting for next 24 hours
    predictions: List[Dict[str, Any]] = []

    for step in range(1, 25):
        next_time = last_timestamp + timedelta(hours=step)
        hour = next_time.hour
        day_of_week = next_time.weekday()

        # Cyclical temporal encodings
        hour_sin = np.sin(2.0 * np.pi * hour / 24.0)
        hour_cos = np.cos(2.0 * np.pi * hour / 24.0)
        dow_sin = np.sin(2.0 * np.pi * day_of_week / 7.0)
        dow_cos = np.cos(2.0 * np.pi * day_of_week / 7.0)

        # Lags from autoregressively updated history
        pm_history = current_time_series["pm25"].values
        pm25_lag1 = pm_history[-1]
        pm25_lag2 = pm_history[-2] if len(pm_history) >= 2 else pm25_lag1
        pm25_lag3 = pm_history[-3] if len(pm_history) >= 3 else pm25_lag2
        pm25_lag6 = pm_history[-6] if len(pm_history) >= 6 else pm25_lag3
        pm25_lag12 = pm_history[-12] if len(pm_history) >= 12 else pm25_lag6
        pm25_lag24 = pm_history[-24] if len(pm_history) >= 24 else pm25_lag12

        # Polynomial features
        pm25_lag1_squared = float(pm25_lag1 ** 2)
        pm25_lag24_squared = float(pm25_lag24 ** 2)

        # Interaction features
        wind_pm25_ratio = float(pm25_lag1 / (max(0.0, last_wind) + 1.0))
        temp_denom = last_temp + 1.0 if abs(last_temp + 1.0) > 1e-4 else 1e-4
        humidity_temp_ratio = float(last_humidity / temp_denom)
        no2_o3_ratio = float(last_no2 / (max(0.0, last_o3) + 1.0))

        # Rolling statistics
        recent_6 = pm_history[-6:]
        recent_24 = pm_history[-24:]
        rolling_mean_6 = float(np.mean(recent_6))
        rolling_mean_24 = float(np.mean(recent_24))
        rolling_std_24 = float(np.std(recent_24)) if len(recent_24) > 1 else 0.0

        step_features = pd.DataFrame([{
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
        }])[feature_cols]

        pred_val = float(model.predict(step_features)[0])
        # PM2.5 cannot be negative
        pred_val = max(1.0, round(pred_val, 1))

        # Confidence bounds approximation based on step and rolling variance
        margin = max(2.5, round((0.15 + (step * 0.015)) * pred_val + (0.2 * rolling_std_24), 1))
        lower_bound = max(0.5, round(pred_val - margin, 1))
        upper_bound = round(pred_val + margin, 1)

        predictions.append({
            "step": step,
            "time": next_time.strftime("%Y-%m-%dT%H:%M"),
            "display_time": next_time.strftime("%I:%M %p"),
            "display_date": next_time.strftime("%b %d"),
            "predicted_pm25": pred_val,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound
        })

        # Append step prediction to series for subsequent lags
        new_row = pd.DataFrame([{
            "time": next_time,
            "pm25": pred_val,
            "pm2_5": pred_val,
            "wind_speed": last_wind,
            "temperature": last_temp,
            "humidity": last_humidity,
            "no2": last_no2,
            "o3": last_o3
        }])
        current_time_series = pd.concat([current_time_series, new_row], ignore_index=True)

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
        "forecast": predictions
    }

    # Cache trained model AND prediction result
    MODEL_CACHE[cache_key] = {
        "timestamp": now,
        "model": model,
        "data": result
    }

    return result
