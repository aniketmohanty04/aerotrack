import asyncio
from datetime import datetime, timedelta, timezone
import logging
import re
import time
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import httpx
import numpy as np

try:
    from backend.ml_model import (
        train_and_forecast_pm25,
        normalize_coordinates,
        ForecastError,
        UpstreamProviderError,
        InsufficientDataError,
        ModelTrainingError
    )
except ImportError:
    from ml_model import (
        train_and_forecast_pm25,
        normalize_coordinates,
        ForecastError,
        UpstreamProviderError,
        InsufficientDataError,
        ModelTrainingError
    )
import os
from fastapi.responses import JSONResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("airquality-backend")

app = FastAPI(
    title="Air Quality Tracking & ML Forecast API",
    description="Real-time air quality metrics, 7-day trends, and 24-hour XGBoost PM2.5 forecasting.",
    version="1.0.0"
)

# Allowed CORS origins
allowed_origins_env = os.getenv("ALLOWED_ORIGINS", "")
if allowed_origins_env.strip():
    allowed_origins = [o.strip() for o in allowed_origins_env.split(",") if o.strip()]
else:
    allowed_origins = [
        "https://aerotrack-three.vercel.app",
        "http://localhost:5173",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
    ]

# Restrict CORS to configured production frontend and development hosts
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_origin_regex=r"^https:\/\/aerotrack-.*\.vercel\.app$",
    allow_credentials=False,
    allow_methods=["GET", "HEAD", "OPTIONS"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {
        "name": "AeroTrack API",
        "status": "online",
        "docs": "/docs",
        "health": "/api/health",
        "frontend": "https://aerotrack-three.vercel.app/"
    }


@app.get("/api/info")
@app.head("/api/info", include_in_schema=False)
def read_api_info():
    return {
        "name": "AeroTrack Air Quality & Forecasting API",
        "status": "online",
        "docs": "/docs",
        "frontend": "https://aerotrack-three.vercel.app/",
        "endpoints": {
            "health": "/api/health",
            "air_quality": "/api/air-quality/{lat}/{lon}",
            "trends": "/api/trends/{lat}/{lon}",
            "predict": "/api/predict/{lat}/{lon}",
            "search": "/api/search?q={query}",
            "reverse_geocode": "/api/reverse-geocode/{lat}/{lon}"
        }
    }


@app.get("/api/health")
@app.head("/api/health", include_in_schema=False)
def health_check():
    return {"status": "ok", "service": "aerotrack-backend"}


def validate_coords(lat: float, lon: float) -> tuple[float, float]:
    if not (-90 <= lat <= 90):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid latitude {lat}. Must be between -90 and 90."
        )
    if not (-180 <= lon <= 180):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid longitude {lon}. Must be between -180 and 180."
        )
    return lat, lon


def get_aqi_category(us_aqi: Optional[float]) -> Dict[str, str]:
    if us_aqi is None:
        return {"category": "Unknown", "level": "unknown", "color": "#6b7280"}
    
    val = round(us_aqi)
    if val <= 50:
        return {
            "category": "Good",
            "level": "good",
            "color": "#10b981",
            "description": "Air quality is considered satisfactory, and air pollution poses little or no risk."
        }
    elif val <= 100:
        return {
            "category": "Moderate",
            "level": "moderate",
            "color": "#f59e0b",
            "description": "Air quality is acceptable; however, some pollutants may pose moderate health concern for sensitive individuals."
        }
    elif val <= 150:
        return {
            "category": "Unhealthy for Sensitive Groups",
            "level": "unhealthy_sensitive",
            "color": "#f97316",
            "description": "Members of sensitive groups may experience health effects. General public is less likely to be affected."
        }
    elif val <= 200:
        return {
            "category": "Unhealthy",
            "level": "unhealthy",
            "color": "#ef4444",
            "description": "Everyone may begin to experience health effects; members of sensitive groups may experience more serious effects."
        }
    elif val <= 300:
        return {
            "category": "Very Unhealthy",
            "level": "very_unhealthy",
            "color": "#8b5cf6",
            "description": "Health alert: The risk of health effects is increased for everyone."
        }
    else:
        return {
            "category": "Hazardous",
            "level": "hazardous",
            "color": "#7f1d1d",
            "description": "Health warning of emergency conditions: The entire population is more likely to be affected."
        }


# US EPA AQI breakpoints — source: https://www.airnow.gov/sites/default/files/2020-05/aqi-technical-assistance-document-sept2018.pdf
# Averaging periods: PM2.5 (24h), PM10 (24h), O3 (8h), CO (8h), SO2 (1h), NO2 (1h)
# All concentrations in µg/m³

US_EPA_BREAKPOINTS = {
    "pm2_5": [9.0, 35.4, 55.4, 125.4, 225.4],
    "pm10":  [54, 154, 254, 354, 424],
    "o3":    [107.9, 137.2, 166.6, 205.8, 392.0],
    "co":    [5039, 10763, 14198, 17633, 34808],
    "so2":   [91.7, 196.5, 484.7, 796.5, 1582.5],
    "no2":   [99.6, 188.0, 676.8, 1220.1, 2348.1],
}

AQI_CATEGORIES = [
    "good",               # 0-50
    "moderate",           # 51-100
    "unhealthy_sensitive",# 101-150
    "unhealthy",          # 151-200
    "very_unhealthy",     # 201-300
    "hazardous",          # 301+
]

def get_rating(pollutant: str, value: Optional[float]) -> str:
    if value is None:
        return "unknown"
    normalized_key = {
        "pm2_5": "pm2_5",
        "pm10": "pm10",
        "o3": "o3",
        "ozone": "o3",
        "co": "co",
        "carbon_monoxide": "co",
        "so2": "so2",
        "sulphur_dioxide": "so2",
        "no2": "no2",
        "nitrogen_dioxide": "no2",
    }.get(pollutant.lower(), pollutant.lower())

    breakpoints = US_EPA_BREAKPOINTS.get(normalized_key)
    if not breakpoints:
        return "unknown"
    for i, bp in enumerate(breakpoints):
        if value <= bp:
            return AQI_CATEGORIES[i]
    return "hazardous"

get_pollutant_rating = get_rating


def calculate_pm25_to_us_aqi(pm25: Optional[float]) -> Optional[int]:
    """
    Converts PM2.5 concentration (µg/m³) to US EPA AQI using standard piecewise linear interpolation.
    Follows US EPA AQI breakpoints:
    0.0 - 9.0 -> 0 - 50 (Good)
    9.1 - 35.4 -> 51 - 100 (Moderate)
    35.5 - 55.4 -> 101 - 150 (Unhealthy for Sensitive Groups)
    55.5 - 125.4 -> 151 - 200 (Unhealthy)
    125.5 - 225.4 -> 201 - 300 (Very Unhealthy)
    225.5 - 325.4 -> 301 - 400 (Hazardous)
    325.5 - 500.4 -> 401 - 500 (Hazardous)
    """
    if pm25 is None:
        return None
    try:
        c = round(float(pm25), 1)
        if c < 0 or np.isnan(c) or np.isinf(c):
            return None
        if c <= 9.0:
            return round(((50.0 - 0.0) / (9.0 - 0.0)) * (c - 0.0) + 0.0)
        elif c <= 35.4:
            return round(((100.0 - 51.0) / (35.4 - 9.1)) * (c - 9.1) + 51.0)
        elif c <= 55.4:
            return round(((150.0 - 101.0) / (55.4 - 35.5)) * (c - 35.5) + 101.0)
        elif c <= 125.4:
            return round(((200.0 - 151.0) / (125.4 - 55.5)) * (c - 55.5) + 151.0)
        elif c <= 225.4:
            return round(((300.0 - 201.0) / (225.4 - 125.5)) * (c - 125.5) + 201.0)
        elif c <= 325.4:
            return round(((400.0 - 301.0) / (325.4 - 225.5)) * (c - 225.5) + 301.0)
        elif c <= 500.4:
            return round(((500.0 - 401.0) / (500.4 - 325.5)) * (c - 325.5) + 401.0)
        else:
            return 500
    except (ValueError, TypeError):
        return None



@app.get("/api/location/search")
async def search_location(query: str = Query(..., min_length=2)):
    """Search coordinates by city name using Open-Meteo Geocoding API."""
    url = "https://geocoding-api.open-meteo.com/v1/search"
    params = {"name": query, "count": 6, "language": "en", "format": "json"}
    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()
            results = []
            for item in data.get("results", []):
                results.append({
                    "name": item.get("name"),
                    "country": item.get("country", ""),
                    "admin1": item.get("admin1", ""),
                    "latitude": item.get("latitude"),
                    "longitude": item.get("longitude"),
                    "display_name": f"{item.get('name')}, {item.get('admin1') or ''} {item.get('country') or ''}".strip(", ")
                })
            return {"results": results}
    except Exception as e:
        logger.error(f"Geocoding error: {e}")
        return {"results": []}


# Shared in-memory cache for Open-Meteo current + trends data
# Key: (round(lat, 4), round(lon, 4)), Value: { 'timestamp': float, 'data': dict }
SHARED_AIR_CACHE: Dict[tuple, Dict[str, Any]] = {}
CACHE_TTL_AIR_SECONDS = 600  # 10 minutes
WAQI_TOKEN = os.getenv("WAQI_TOKEN", "b226d82981367a88a2e4747cea6c8c1b09fac5a1")
MAX_CACHE_SIZE = 500
IN_FLIGHT_AIR_REQUESTS: Dict[tuple, asyncio.Task] = {}



async def _fetch_waqi_raw(lat: float, lon: float) -> Dict[str, Any]:
    url = f"https://api.waqi.info/feed/geo:{lat};{lon}/?token={WAQI_TOKEN}"
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        data = resp.json()
        if data.get("status") != "ok":
            raise ValueError(f"WAQI API Error: {data.get('data')}")
        return data

async def _fetch_open_meteo_raw(lat: float, lon: float) -> Dict[str, Any]:
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": "european_aqi,us_aqi,pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone",
        "hourly": "pm2_5,pm10,us_aqi,european_aqi,ozone,nitrogen_dioxide",
        "past_days": 7,
        "forecast_days": 1,
        "timezone": "auto"
    }
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        return resp.json()


async def get_shared_open_meteo_data(lat: float, lon: float) -> Dict[str, Any]:
    """
    Fetch both current air quality and 7-day hourly trends from Open-Meteo in a single call,
    caching with a 10-minute TTL and deduplicating concurrent in-flight requests.
    """
    lat, lon = normalize_coordinates(lat, lon)
    cache_key = (lat, lon)
    now = time.time()

    if cache_key in SHARED_AIR_CACHE:
        cached = SHARED_AIR_CACHE[cache_key]
        age = now - cached["timestamp"]
        if age < CACHE_TTL_AIR_SECONDS:
            logger.info(f"Cache HIT for {cache_key}, age={age:.1f}s")
            return cached["data"]
        else:
            logger.info(f"Cache EXPIRED for {cache_key}, age={age:.1f}s - refetching")
            del SHARED_AIR_CACHE[cache_key]

    # Await existing in-flight task if air-quality and trends fired concurrently
    if cache_key in IN_FLIGHT_AIR_REQUESTS:
        logger.info(f"Awaiting in-flight Open-Meteo request for ({lat}, {lon})")
        return await IN_FLIGHT_AIR_REQUESTS[cache_key]

    logger.info(f"Cache MISS for {cache_key} - fetching from Open-Meteo")
    task = asyncio.create_task(_fetch_open_meteo_raw(lat, lon))
    IN_FLIGHT_AIR_REQUESTS[cache_key] = task
    try:
        data = await task
        SHARED_AIR_CACHE[cache_key] = {
            "timestamp": time.time(),
            "data": data
        }
        if len(SHARED_AIR_CACHE) > MAX_CACHE_SIZE:
            oldest_key = min(SHARED_AIR_CACHE, key=lambda k: SHARED_AIR_CACHE[k]["timestamp"])
            del SHARED_AIR_CACHE[oldest_key]
        return data
    finally:
        IN_FLIGHT_AIR_REQUESTS.pop(cache_key, None)


@app.get("/api/admin/clear-cache")
async def clear_cache():
    SHARED_AIR_CACHE.clear()
    # Also clear the model cache from ml_model
    try:
        from backend.ml_model import MODEL_CACHE
        MODEL_CACHE.clear()
    except ImportError:
        try:
            from ml_model import MODEL_CACHE
            MODEL_CACHE.clear()
        except ImportError:
            pass
    return {"status": "cleared", "shared_cache_size": 0}


@app.get("/api/air-quality/{lat}/{lon}")
async def get_air_quality(lat: float, lon: float):
    """
    Fetches live real-time air quality data for given latitude & longitude from WAQI (Ground Truth).
    """
    lat, lon = validate_coords(lat, lon)
    try:
        # We now fetch from WAQI for ground truth instead of Open-Meteo
        waqi_response = await _fetch_waqi_raw(lat, lon)
        waqi_data = waqi_response.get("data", {})
        iaqi = waqi_data.get("iaqi", {})

        original_us_aqi = waqi_data.get("aqi")
        if original_us_aqi == "-":
            original_us_aqi = None
            
        us_aqi = original_us_aqi
        if us_aqi is not None and isinstance(us_aqi, (int, float)) and us_aqi > 500:
            us_aqi = 500

        aqi_info = get_aqi_category(us_aqi) if us_aqi is not None else get_aqi_category(0)

        def _get_pollutant(waqi_key, label, desc):
            val = iaqi.get(waqi_key, {}).get("v")
            # WAQI returns US EPA AQI for each pollutant, not µg/m³.
            # We use the AQI category helper to get the rating level string.
            rating_level = get_aqi_category(val)["level"] if val is not None else "unknown"
            return {
                "label": label,
                "value": val,
                "unit": "AQI",
                "rating": rating_level,
                "description": desc
            }

        pollutants = {
            "pm2_5": _get_pollutant("pm25", "PM2.5", "Fine inhalable particles (AQI scale)"),
            "pm10": _get_pollutant("pm10", "PM10", "Inhalable particles (AQI scale)"),
            "nitrogen_dioxide": _get_pollutant("no2", "NO₂", "Nitrogen Dioxide (AQI scale)"),
            "sulphur_dioxide": _get_pollutant("so2", "SO₂", "Sulphur Dioxide (AQI scale)"),
            "ozone": _get_pollutant("o3", "O₃", "Ground-level ozone (AQI scale)"),
            "carbon_monoxide": _get_pollutant("co", "CO", "Carbon Monoxide (AQI scale)")
        }

        is_capped = bool(original_us_aqi is not None and isinstance(original_us_aqi, (int, float)) and original_us_aqi > 500)

        # WAQI ISO time string
        waqi_time = waqi_data.get("time", {}).get("iso", None)

        return {
            "latitude": lat,
            "longitude": lon,
            "time": waqi_time,
            "timezone": waqi_data.get("time", {}).get("tz", "UTC"),
            "us_aqi": us_aqi,
            "european_aqi": us_aqi,  # WAQI only provides US AQI, mirror it to avoid breaking frontend types
            "us_aqi_capped": is_capped,
            "is_capped": is_capped,
            "aqi_info": aqi_info,
            "pollutants": pollutants,
            "station_name": waqi_data.get("city", {}).get("name", "Unknown Station")
        }

    except httpx.HTTPError as e:
        logger.error(f"WAQI API request failed: {e}")
        raise HTTPException(status_code=502, detail="Upstream ground station network is currently unavailable.")
    except ValueError as ve:
        logger.error(f"WAQI API returned error: {ve}")
        raise HTTPException(status_code=500, detail=str(ve))
    except Exception as e:
        logger.error(f"Unexpected error in /api/air-quality: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An error occurred while processing air quality data.")


def clean_trend_numeric(val: Any, round_digits: int = 1) -> Optional[float]:
    """Helper to sanitize numerical values; drops negatives/NaNs/Infs."""
    if val is None:
        return None
    try:
        f = float(val)
        if np.isnan(f) or np.isinf(f) or f < 0:
            return None
        return round(f, round_digits) if round_digits > 0 else round(f)
    except (ValueError, TypeError):
        return None


@app.get("/api/trends/{lat}/{lon}")
async def get_trends(lat: float, lon: float):
    """
    Fetches 7-day historical and current hourly trends for PM2.5, PM10, and AQI.
    Explicitly separates historical observations from future forecast hours.
    Guarantees summary statistics and 24-hour averages include only historical observations.
    Shares a 10-minute cache with /api/air-quality to eliminate redundant API calls.
    """
    lat, lon = validate_coords(lat, lon)
    try:
        data = await get_shared_open_meteo_data(lat, lon)

        utc_offset_seconds = int(data.get("utc_offset_seconds", 0))
        timezone_str = data.get("timezone", "UTC")
        current_obj = data.get("current", {})
        current_time_str = current_obj.get("time")

        # Determine reference current observation cutoff in UTC
        if current_time_str:
            try:
                cdt = datetime.fromisoformat(current_time_str)
                if cdt.tzinfo is None:
                    current_utc_dt = (cdt - timedelta(seconds=utc_offset_seconds)).replace(tzinfo=timezone.utc)
                else:
                    current_utc_dt = cdt.astimezone(timezone.utc)
            except Exception:
                current_utc_dt = datetime.now(timezone.utc)
        else:
            current_utc_dt = datetime.now(timezone.utc)

        hourly = data.get("hourly", {})
        times = hourly.get("time", [])
        pm25 = hourly.get("pm2_5", [])
        pm10 = hourly.get("pm10", [])
        us_aqi = hourly.get("us_aqi", [])
        european_aqi = hourly.get("european_aqi", [])
        ozone = hourly.get("ozone", [])
        no2 = hourly.get("nitrogen_dioxide", [])

        # Process, normalize timezone to UTC, and deduplicate records by UTC timestamp
        points_by_utc: Dict[datetime, dict] = {}

        for i in range(len(times)):
            t_str = times[i]
            if not t_str or not isinstance(t_str, str):
                continue

            try:
                p_dt = datetime.fromisoformat(t_str)
                if p_dt.tzinfo is None:
                    p_utc_dt = (p_dt - timedelta(seconds=utc_offset_seconds)).replace(tzinfo=timezone.utc)
                else:
                    p_utc_dt = p_dt.astimezone(timezone.utc)
            except Exception:
                continue

            p25_val = clean_trend_numeric(pm25[i] if i < len(pm25) else None, 1)
            p10_val = clean_trend_numeric(pm10[i] if i < len(pm10) else None, 1)
            aqi_val = clean_trend_numeric(us_aqi[i] if i < len(us_aqi) else None, 0)
            if aqi_val is None and p25_val is not None:
                aqi_val = clean_trend_numeric(calculate_pm25_to_us_aqi(p25_val), 0)
            eaqi_val = clean_trend_numeric(european_aqi[i] if i < len(european_aqi) else None, 0)
            o3_val = clean_trend_numeric(ozone[i] if i < len(ozone) else None, 1)
            no2_val = clean_trend_numeric(no2[i] if i < len(no2) else None, 1)

            # Format human friendly label e.g. "Oct 08 02:00"
            display_label = t_str.replace("T", " ")
            if len(t_str) >= 16:
                month_day = t_str[5:10]
                hour_min = t_str[11:16]
                display_label = f"{month_day} {hour_min}"

            is_forecast = bool(p_utc_dt > current_utc_dt)
            record = {
                "utc_dt": p_utc_dt,
                "time": t_str,
                "utc_time": p_utc_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "label": display_label,
                "is_forecast": is_forecast,
                "point_type": "forecast" if is_forecast else "observation",
                "pm2_5": p25_val,
                "pm10": p10_val,
                "us_aqi": int(aqi_val) if aqi_val is not None else None,
                "european_aqi": int(eaqi_val) if eaqi_val is not None else None,
                "ozone": o3_val,
                "nitrogen_dioxide": no2_val,
            }

            # Handle duplicate timestamps (prefer records with valid PM2.5 readings)
            if p_utc_dt in points_by_utc:
                if points_by_utc[p_utc_dt]["pm2_5"] is None and p25_val is not None:
                    points_by_utc[p_utc_dt] = record
            else:
                points_by_utc[p_utc_dt] = record

        # Sort chronologically by UTC datetime to handle out-of-order upstream records
        sorted_records = sorted(points_by_utc.values(), key=lambda r: r["utc_dt"])

        # Separate historical observations from future forecast values
        historical_obs = [r for r in sorted_records if not r["is_forecast"]]
        forecast_obs = [r for r in sorted_records if r["is_forecast"]]

        valid_hist_pm25 = [r["pm2_5"] for r in historical_obs if r["pm2_5"] is not None]

        avg_pm25 = round(sum(valid_hist_pm25) / len(valid_hist_pm25), 1) if valid_hist_pm25 else 0.0
        min_pm25 = min(valid_hist_pm25) if valid_hist_pm25 else 0.0
        max_pm25 = max(valid_hist_pm25) if valid_hist_pm25 else 0.0

        # Exact 24-hour historical window ending at current observation time: [current_utc_dt - 24h, current_utc_dt]
        cutoff_24h_utc = current_utc_dt - timedelta(hours=24)
        last_24h_obs = [
            r for r in historical_obs
            if r["utc_dt"] >= cutoff_24h_utc and r["pm2_5"] is not None
        ]
        last_24h_pm25 = [r["pm2_5"] for r in last_24h_obs]
        last_24h_avg_pm25 = round(sum(last_24h_pm25) / len(last_24h_pm25), 1) if last_24h_pm25 else avg_pm25

        # Format output points (strip internal datetime objects)
        trend_points = []
        for r in sorted_records:
            trend_points.append({
                "time": r["time"],
                "utc_time": r["utc_time"],
                "label": r["label"],
                "pm2_5": r["pm2_5"],
                "pm10": r["pm10"],
                "us_aqi": r["us_aqi"],
                "european_aqi": r["european_aqi"],
                "ozone": r["ozone"],
                "nitrogen_dioxide": r["nitrogen_dioxide"],
                "is_forecast": r["is_forecast"],
                "point_type": r["point_type"],
            })

        return {
            "latitude": lat,
            "longitude": lon,
            "timezone": timezone_str,
            "utc_offset_seconds": utc_offset_seconds,
            "current_time": current_time_str,
            "current_time_utc": current_utc_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "total_points": len(trend_points),
            "stats": {
                "avg_pm25": avg_pm25,
                "min_pm25": min_pm25,
                "max_pm25": max_pm25,
                "last_24h_avg_pm25": last_24h_avg_pm25,
                "historical_points_count": len(valid_hist_pm25),
                "forecast_points_count": len(forecast_obs),
                "last_24h_points_count": len(last_24h_pm25)
            },
            "trends": trend_points
        }

    except httpx.HTTPError as e:
        logger.error(f"Open-Meteo trends error: {e}")
        raise HTTPException(status_code=502, detail="Upstream air quality trends service is currently unavailable.")
    except Exception as e:
        logger.error(f"Unexpected error in /api/trends: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An error occurred while processing air quality trends.")


@app.get("/api/predict/{lat}/{lon}")
async def get_prediction(
    lat: float,
    lon: float,
    allow_demo: bool = Query(
        False,
        description="Explicit opt-in to return synthetic demonstration data if live data is unavailable."
    )
):
    """
    Calls the XGBoost forecasting model trained on 92 days of PM2.5 data
    and returns a 24-hour prediction forecast.
    """
    lat, lon = validate_coords(lat, lon)
    try:
        forecast_result = await train_and_forecast_pm25(lat, lon, allow_demo=allow_demo)
        return forecast_result
    except ForecastError as fe:
        logger.error(
            f"Forecast pipeline error for ({lat}, {lon}) [{fe.error_code}]: {fe.message}",
            extra={"latitude": lat, "longitude": lon, "error_code": fe.error_code}
        )
        return JSONResponse(
            status_code=fe.status_code,
            content={
                "status": "error",
                "error_code": fe.error_code,
                "message": fe.message,
                "is_synthetic": False,
                "details": fe.details
            }
        )
    except Exception as e:
        logger.error(f"Unexpected forecast error for {lat}, {lon}: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "error_code": "INTERNAL_SERVER_ERROR",
                "message": "An unexpected error occurred while generating the forecast.",
                "is_synthetic": False
            }
        )


REVERSE_GEOCODE_CACHE: Dict[Tuple[float, float], Tuple[float, dict]] = {}
CACHE_TTL_GEOCODE_SECONDS = 86400  # 24 hours for successful geocode responses
SEARCH_CACHE: Dict[str, Tuple[float, List[Dict[str, Any]]]] = {}
CACHE_TTL_SEARCH_SECONDS = 600  # 10 minutes for place search queries

nominatim_lock: Optional[asyncio.Lock] = None
last_nominatim_call = 0.0

NOMINATIM_HEADERS = {
    "User-Agent": "AeroTrack/1.0 (https://github.com/aniketmohanty04/aerotrack; contact: aniketmohanty04@gmail.com)",
    "Accept-Language": "en"
}


def get_nominatim_lock() -> asyncio.Lock:
    global nominatim_lock
    if nominatim_lock is None:
        nominatim_lock = asyncio.Lock()
    return nominatim_lock


async def reverse_geocode(lat: float, lon: float) -> dict:
    """
    Reverse geocodes coordinates with multi-provider redundancy:
    1. Primary: OpenStreetMap Nominatim with English localization.
    2. Fallback: BigDataCloud Reverse Geocoding API if Nominatim times out, throttles, or returns empty.
    
    Caching & Failure Semantics:
    - Successful or partial location results (where at least one of city, region, or country is known)
      are cached for 24 hours.
    - Upstream failures or null fallback responses are NEVER cached, preventing transient outages
      from poisoning coordinates.
    """
    cache_key = (round(lat, 4), round(lon, 4))
    now = time.time()
    if cache_key in REVERSE_GEOCODE_CACHE:
        ts, cached_val = REVERSE_GEOCODE_CACHE[cache_key]
        if now - ts < CACHE_TTL_GEOCODE_SECONDS:
            logger.debug(f"Reverse geocode cache HIT for {cache_key}: {cached_val}")
            return cached_val

    global last_nominatim_call
    data: Optional[Dict[str, Any]] = None

    # --- 1. Primary: OpenStreetMap Nominatim ---
    url_nom = "https://nominatim.openstreetmap.org/reverse"
    params_nom = {
        "lat": lat,
        "lon": lon,
        "format": "json",
        "accept-language": "en",
        "zoom": 10,
        "addressdetails": 1,
        "namedetails": 1,
    }

    try:
        async with get_nominatim_lock():
            elapsed = time.time() - last_nominatim_call
            if elapsed < 1.0:
                await asyncio.sleep(1.0 - elapsed)
            try:
                async with httpx.AsyncClient(timeout=6.0) as client:
                    t0 = time.perf_counter()
                    r = await client.get(url_nom, params=params_nom, headers=NOMINATIM_HEADERS)
                    last_nominatim_call = time.time()
                    elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
                    if r.status_code == 200:
                        data = r.json()
                        logger.info(f"Nominatim reverse geocode OK ({lat}, {lon}) [{elapsed_ms}ms]")
                    else:
                        logger.warning(
                            f"Nominatim reverse geocode returned HTTP {r.status_code} for ({lat}, {lon}) [{elapsed_ms}ms]"
                        )
            except (httpx.TimeoutException, httpx.RequestError) as exc:
                last_nominatim_call = time.time()
                logger.warning(f"Nominatim reverse geocode network/timeout error for ({lat}, {lon}): {exc}")
    except Exception as exc:
        logger.warning(f"Nominatim lock or unexpected error: {exc}")

    city: Optional[str] = None
    region: Optional[str] = None
    country: Optional[str] = None

    def sanitize_en(val: Optional[str], fallback_en: Optional[str] = None) -> Optional[str]:
        candidate = fallback_en or val
        if not candidate:
            return None
        candidate = candidate.strip()
        # If contains CJK characters, prioritize English/Latin namedetails
        if re.search(r'[\u4e00-\u9fff\u3040-\u30ff]', candidate):
            eng = (
                (data.get("namedetails", {}).get("name:en") if data else None)
                or (data.get("namedetails", {}).get("_place_name:en") if data else None)
                or (data.get("namedetails", {}).get("int_name") if data else None)
                or (data.get("namedetails", {}).get("name:latin") if data else None)
                or (data.get("namedetails", {}).get("name:zh-Latn-pinyin") if data else None)
            )
            if eng and not re.search(r'[\u4e00-\u9fff]', eng):
                return eng
            cleaned = re.sub(r'[\u4e00-\u9fff\u3040-\u30ff\u0f00-\u0fff]', '', candidate).strip(' ,')
            return cleaned if cleaned else None
        return candidate if candidate else None

    if data:
        address = data.get("address", {})
        namedetails = data.get("namedetails", {})

        raw_city = (
            namedetails.get("name:en")
            or address.get("city")
            or address.get("town")
            or address.get("municipality")
            or address.get("state_district")
            or address.get("district")
            or address.get("county")
            or address.get("suburb")
            or address.get("village")
        )
        city = sanitize_en(raw_city, namedetails.get("name:en"))

        raw_region = (
            namedetails.get("state:en")
            or namedetails.get("region:en")
            or address.get("state")
            or address.get("province")
            or address.get("region")
        )
        region = sanitize_en(raw_region, namedetails.get("state:en") or namedetails.get("region:en"))

        raw_country = namedetails.get("country:en") or address.get("country")
        country = sanitize_en(raw_country, namedetails.get("country:en"))
        if not country and address.get("country_code") == "cn":
            country = "China"

    # --- 2. Fallback: BigDataCloud Reverse Geocoding if Nominatim failed or yielded all nulls ---
    if not (city or region or country):
        logger.info(f"Attempting BigDataCloud reverse geocode fallback for ({lat}, {lon})...")
        url_bdc = "https://api.bigdatacloud.net/data/reverse-geocode-client"
        params_bdc = {"latitude": lat, "longitude": lon, "localityLanguage": "en"}
        try:
            async with httpx.AsyncClient(timeout=4.0, follow_redirects=True) as client:
                r_bdc = await client.get(url_bdc, params=params_bdc)
                if r_bdc.status_code == 200:
                    bdc_data = r_bdc.json()
                    city = bdc_data.get("city") or bdc_data.get("locality") or city
                    region = bdc_data.get("principalSubdivision") or region
                    country = bdc_data.get("countryName") or country
                    logger.info(f"BigDataCloud resolved ({lat}, {lon}) -> city={city}, region={region}, country={country}")
        except Exception as exc:
            logger.warning(f"BigDataCloud reverse geocode fallback error for ({lat}, {lon}): {exc}")

    result = {
        "city": city if city else None,
        "region": region if region else None,
        "country": country if country else None,
    }

    # CRITICAL: NEVER cache all-null or failure fallbacks!
    # Cache only when at least one location attribute is successfully resolved.
    if city or region or country:
        REVERSE_GEOCODE_CACHE[cache_key] = (now, result)
        logger.info(f"Cached valid reverse geocode for ({lat}, {lon}): {result}")
    else:
        logger.warning(f"Reverse geocode returned all-null for ({lat}, {lon}); NOT caching failure.")

    return result


@app.get("/api/reverse-geocode/{lat}/{lon}")
async def get_reverse_geocode(lat: float, lon: float):
    lat, lon = validate_coords(lat, lon)
    return await reverse_geocode(lat, lon)


@app.get("/api/search")
async def search_places(q: str = Query(..., min_length=1)):
    """
    Search places by name using OpenStreetMap Nominatim.
    Preserves SearchResult schema expected by frontend/src/components/SearchBar.tsx.
    Distinguishes legitimate empty results from upstream timeouts, rate limits, and server errors.
    """
    query = q.strip()
    if not query:
        return []

    cache_key = query.lower()
    now = time.time()
    if cache_key in SEARCH_CACHE:
        ts, cached_results = SEARCH_CACHE[cache_key]
        if now - ts < CACHE_TTL_SEARCH_SECONDS:
            logger.debug(f"Search cache HIT for '{query}' ({len(cached_results)} results)")
            return cached_results

    global last_nominatim_call
    url = "https://nominatim.openstreetmap.org/search"
    params = {
        "q": query,
        "format": "json",
        "accept-language": "en",
        "limit": 5,
        "addressdetails": 1,
    }

    raw_results = None
    async with get_nominatim_lock():
        elapsed = time.time() - last_nominatim_call
        if elapsed < 1.0:
            await asyncio.sleep(1.0 - elapsed)
        try:
            async with httpx.AsyncClient(timeout=8.0) as client:
                res = await client.get(url, params=params, headers=NOMINATIM_HEADERS)
                last_nominatim_call = time.time()

                if res.status_code == 200:
                    raw_results = res.json()
                elif res.status_code == 429:
                    logger.warning(f"Nominatim search 429 rate limit exceeded for query '{query}'")
                    raise HTTPException(
                        status_code=429,
                        detail="Upstream place search rate limit exceeded. Please wait a moment before searching again."
                    )
                elif res.status_code >= 500:
                    logger.error(f"Nominatim search HTTP {res.status_code} server error for query '{query}': {res.text}")
                    raise HTTPException(
                        status_code=502,
                        detail="Upstream place search service is temporarily unavailable."
                    )
                else:
                    logger.error(f"Nominatim search unexpected HTTP {res.status_code} for query '{query}': {res.text}")
                    raise HTTPException(
                        status_code=502,
                        detail="Failed to query upstream place search service."
                    )
        except httpx.TimeoutException as exc:
            last_nominatim_call = time.time()
            logger.warning(f"Nominatim search timed out for query '{query}': {exc}")
            raise HTTPException(
                status_code=504,
                detail="Upstream place search service timed out. Please try again."
            ) from exc
        except httpx.RequestError as exc:
            last_nominatim_call = time.time()
            logger.error(f"Nominatim search connection error for query '{query}': {exc}")
            raise HTTPException(
                status_code=502,
                detail="Upstream place search service connection failed."
            ) from exc

    if raw_results is None:
        return []

    parsed_results = []
    for result in raw_results:
        display_name = result.get("display_name", "")
        parsed_results.append({
            "name": display_name.split(",")[0].strip() if display_name else "",
            "full_name": display_name,
            "latitude": float(result.get("lat", 0.0)),
            "longitude": float(result.get("lon", 0.0)),
            "type": result.get("type"),
        })

    # Cache successful results (even legitimate empty results) for 10 minutes
    SEARCH_CACHE[cache_key] = (now, parsed_results)
    return parsed_results


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
