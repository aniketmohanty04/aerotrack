import asyncio
import logging
import re
import time
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import httpx

try:
    from backend.ml_model import train_and_forecast_pm25, normalize_coordinates
except ImportError:
    from ml_model import train_and_forecast_pm25, normalize_coordinates

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("airquality-backend")

app = FastAPI(
    title="Air Quality Tracking & ML Forecast API",
    description="Real-time air quality metrics, 7-day trends, and 24-hour XGBoost PM2.5 forecasting.",
    version="1.0.0"
)

# Enable CORS for frontend development and production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.api_route("/", methods=["GET", "HEAD"])
def read_root():
    return {
        "name": "AeroTrack Air Quality & Forecasting API",
        "status": "online",
        "docs": "/docs",
        "frontend": "http://localhost:5173",
        "endpoints": {
            "health": "/api/health",
            "air_quality": "/api/air-quality/{lat}/{lon}",
            "trends": "/api/trends/{lat}/{lon}",
            "predict": "/api/predict/{lat}/{lon}",
            "search": "/api/search?q={query}",
            "reverse_geocode": "/api/reverse-geocode/{lat}/{lon}"
        }
    }


@app.api_route("/api/health", methods=["GET", "HEAD"])
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


@app.get("/api/health")
async def health_check():
    return {"status": "ok", "service": "air-quality-api"}


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


async def get_shared_open_meteo_data(lat: float, lon: float) -> Dict[str, Any]:
    """
    Fetch both current air quality and 7-day hourly trends from Open-Meteo in a single call,
    caching with a 10-minute TTL so /api/air-quality and /api/trends share the same data.
    """
    lat, lon = normalize_coordinates(lat, lon)
    cache_key = (lat, lon)
    now = time.time()

    if cache_key in SHARED_AIR_CACHE:
        cached = SHARED_AIR_CACHE[cache_key]
        if now - cached["timestamp"] < CACHE_TTL_AIR_SECONDS:
            logger.info(f"Shared Open-Meteo cache HIT for ({lat}, {lon})")
            return cached["data"]

    logger.info(f"Shared Open-Meteo cache MISS for ({lat}, {lon}) - fetching from Open-Meteo")
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

    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()

    SHARED_AIR_CACHE[cache_key] = {
        "timestamp": now,
        "data": data
    }
    return data


@app.get("/api/air-quality/{lat}/{lon}")
async def get_air_quality(lat: float, lon: float):
    """
    Fetches live real-time air quality data for given latitude & longitude from Open-Meteo.
    Shares a 10-minute cache with /api/trends to eliminate redundant API calls.
    """
    lat, lon = validate_coords(lat, lon)
    try:
        data = await get_shared_open_meteo_data(lat, lon)

        current = data.get("current", {})
        units = data.get("current_units", {})

        us_aqi = current.get("us_aqi")
        aqi_info = get_aqi_category(us_aqi)

        pollutants = {
            "pm2_5": {
                "label": "PM2.5",
                "value": current.get("pm2_5"),
                "unit": units.get("pm2_5", "µg/m³"),
                "rating": get_rating("pm2_5", current.get("pm2_5")),
                "description": "Fine inhalable particles with diameters 2.5 micrometers and smaller."
            },
            "pm10": {
                "label": "PM10",
                "value": current.get("pm10"),
                "unit": units.get("pm10", "µg/m³"),
                "rating": get_rating("pm10", current.get("pm10")),
                "description": "Inhalable particles with diameters 10 micrometers and smaller."
            },
            "nitrogen_dioxide": {
                "label": "NO₂",
                "value": current.get("nitrogen_dioxide"),
                "unit": units.get("nitrogen_dioxide", "µg/m³"),
                "rating": get_rating("no2", current.get("nitrogen_dioxide")),
                "description": "Nitrogen Dioxide, largely from emissions and fuel combustion."
            },
            "sulphur_dioxide": {
                "label": "SO₂",
                "value": current.get("sulphur_dioxide"),
                "unit": units.get("sulphur_dioxide", "µg/m³"),
                "rating": get_rating("so2", current.get("sulphur_dioxide")),
                "description": "Sulphur Dioxide, emitted by burning fossil fuels and industrial processes."
            },
            "ozone": {
                "label": "O₃",
                "value": current.get("ozone"),
                "unit": units.get("ozone", "µg/m³"),
                "rating": get_rating("o3", current.get("ozone")),
                "description": "Ground-level ozone formed by reactions between pollutants and sunlight."
            },
            "carbon_monoxide": {
                "label": "CO",
                "value": current.get("carbon_monoxide"),
                "unit": units.get("carbon_monoxide", "µg/m³"),
                "rating": get_rating("co", current.get("carbon_monoxide")),
                "description": "Carbon Monoxide, an odorless gas from incomplete combustion."
            }
        }

        return {
            "latitude": lat,
            "longitude": lon,
            "time": current.get("time"),
            "timezone": data.get("timezone", "UTC"),
            "us_aqi": us_aqi,
            "european_aqi": current.get("european_aqi"),
            "aqi_info": aqi_info,
            "pollutants": pollutants
        }

    except httpx.HTTPError as e:
        logger.error(f"Open-Meteo API request failed: {e}")
        raise HTTPException(status_code=502, detail=f"Failed to fetch air quality data from Open-Meteo: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error in /api/air-quality: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/trends/{lat}/{lon}")
async def get_trends(lat: float, lon: float):
    """
    Fetches 7-day historical and current hourly trends for PM2.5, PM10, and AQI.
    Shares a 10-minute cache with /api/air-quality to eliminate redundant API calls.
    """
    lat, lon = validate_coords(lat, lon)
    try:
        data = await get_shared_open_meteo_data(lat, lon)

        hourly = data.get("hourly", {})
        times = hourly.get("time", [])
        pm25 = hourly.get("pm2_5", [])
        pm10 = hourly.get("pm10", [])
        us_aqi = hourly.get("us_aqi", [])
        european_aqi = hourly.get("european_aqi", [])
        ozone = hourly.get("ozone", [])
        no2 = hourly.get("nitrogen_dioxide", [])

        trend_points = []
        valid_pm25 = []

        for i in range(len(times)):
            t_str = times[i]
            p25_val = pm25[i] if i < len(pm25) else None
            p10_val = pm10[i] if i < len(pm10) else None
            aqi_val = us_aqi[i] if i < len(us_aqi) else None
            eaqi_val = european_aqi[i] if i < len(european_aqi) else None
            o3_val = ozone[i] if i < len(ozone) else None
            no2_val = no2[i] if i < len(no2) else None

            if p25_val is not None:
                valid_pm25.append(p25_val)

            # Format human friendly time e.g. "Oct 08 02:00"
            display_label = t_str.replace("T", " ")
            if len(t_str) >= 16:
                month_day = t_str[5:10]  # MM-DD
                hour_min = t_str[11:16]  # HH:MM
                display_label = f"{month_day} {hour_min}"

            trend_points.append({
                "time": t_str,
                "label": display_label,
                "pm2_5": round(p25_val, 1) if p25_val is not None else None,
                "pm10": round(p10_val, 1) if p10_val is not None else None,
                "us_aqi": round(aqi_val) if aqi_val is not None else None,
                "european_aqi": round(eaqi_val) if eaqi_val is not None else None,
                "ozone": round(o3_val, 1) if o3_val is not None else None,
                "nitrogen_dioxide": round(no2_val, 1) if no2_val is not None else None,
            })

        avg_pm25 = round(sum(valid_pm25) / len(valid_pm25), 1) if valid_pm25 else 0.0
        min_pm25 = min(valid_pm25) if valid_pm25 else 0.0
        max_pm25 = max(valid_pm25) if valid_pm25 else 0.0

        return {
            "latitude": lat,
            "longitude": lon,
            "total_points": len(trend_points),
            "stats": {
                "avg_pm25": avg_pm25,
                "min_pm25": min_pm25,
                "max_pm25": max_pm25
            },
            "trends": trend_points
        }

    except httpx.HTTPError as e:
        logger.error(f"Open-Meteo trends error: {e}")
        raise HTTPException(status_code=502, detail=f"Failed to fetch trends from Open-Meteo: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error in /api/trends: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/predict/{lat}/{lon}")
async def get_prediction(lat: float, lon: float):
    """
    Calls the XGBoost forecasting model trained on 92 days of PM2.5 data
    and returns a 24-hour prediction forecast.
    """
    lat, lon = validate_coords(lat, lon)
    try:
        forecast_result = await train_and_forecast_pm25(lat, lon)
        return forecast_result
    except Exception as e:
        logger.error(f"Forecast error for {lat}, {lon}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Forecast training failed: {str(e)}")


nominatim_lock: Optional[asyncio.Lock] = None
last_nominatim_call = 0.0


def get_nominatim_lock() -> asyncio.Lock:
    global nominatim_lock
    if nominatim_lock is None:
        nominatim_lock = asyncio.Lock()
    return nominatim_lock


async def reverse_geocode(lat: float, lon: float) -> dict:
    url = "https://nominatim.openstreetmap.org/reverse"
    headers = {
        "User-Agent": "AeroTrack/1.0 (air quality demo project)",
        "Accept-Language": "en"
    }
    params = {
        "lat": lat,
        "lon": lon,
        "format": "json",
        "accept-language": "en",
        "zoom": 10,
        "addressdetails": 1,
        "namedetails": 1,
    }
    
    global last_nominatim_call
    async with get_nominatim_lock():
        now = time.time()
        elapsed = now - last_nominatim_call
        if elapsed < 1.0:
            await asyncio.sleep(1.0 - elapsed)
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                r = await client.get(url, params=params, headers=headers)
                last_nominatim_call = time.time()
                if r.status_code != 200:
                    return {"city": None, "region": None, "country": None}
                data = r.json()
        except Exception as e:
            last_nominatim_call = time.time()
            logger.error(f"Reverse geocode failed: {e}")
            return {"city": None, "region": None, "country": None}

    address = data.get("address", {})
    namedetails = data.get("namedetails", {})

    def sanitize_en(val: Optional[str], fallback_en: Optional[str] = None) -> Optional[str]:
        candidate = fallback_en or val
        if not candidate:
            return None
        # If contains CJK characters, prioritize English/Latin namedetails
        if re.search(r'[\u4e00-\u9fff\u3040-\u30ff]', candidate):
            eng = (
                namedetails.get("name:en")
                or namedetails.get("_place_name:en")
                or namedetails.get("int_name")
                or namedetails.get("name:latin")
                or namedetails.get("name:zh-Latn-pinyin")
            )
            if eng and not re.search(r'[\u4e00-\u9fff]', eng):
                return eng
            # Remove any trailing non-ASCII
            cleaned = re.sub(r'[\u4e00-\u9fff\u3040-\u30ff\u0f00-\u0fff]', '', candidate).strip(' ,')
            return cleaned if cleaned else None
        return candidate

    raw_city = (
        namedetails.get("name:en")
        or address.get("town")
        or address.get("city")
        or address.get("county")
        or address.get("village")
        or address.get("municipality")
    )
    city = sanitize_en(raw_city, namedetails.get("name:en"))

    raw_region = namedetails.get("state:en") or namedetails.get("region:en") or address.get("state") or address.get("province") or address.get("region")
    region = sanitize_en(raw_region, namedetails.get("state:en") or namedetails.get("region:en"))

    raw_country = namedetails.get("country:en") or address.get("country")
    country = sanitize_en(raw_country, namedetails.get("country:en"))
    if not country and address.get("country_code") == "cn":
        country = "China"

    return {"city": city, "region": region, "country": country}


@app.get("/api/reverse-geocode/{lat}/{lon}")
async def get_reverse_geocode(lat: float, lon: float):
    lat, lon = validate_coords(lat, lon)
    return await reverse_geocode(lat, lon)


@app.get("/api/search")
async def search_places(q: str = Query(..., min_length=1)):
    global last_nominatim_call
    url = "https://nominatim.openstreetmap.org/search"
    headers = {
        "User-Agent": "AeroTrack/1.0 (air quality demo project)",
        "Accept-Language": "en"
    }
    params = {
        "q": q,
        "format": "json",
        "accept-language": "en",
        "limit": 5,
        "addressdetails": 1,
    }

    async with get_nominatim_lock():
        now = time.time()
        elapsed = now - last_nominatim_call
        if elapsed < 1.0:
            await asyncio.sleep(1.0 - elapsed)
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                res = await client.get(url, params=params, headers=headers)
                last_nominatim_call = time.time()
                if res.status_code != 200:
                    logger.error(f"Nominatim error {res.status_code}: {res.text}")
                    return []
                raw_results = res.json()
        except Exception as e:
            last_nominatim_call = time.time()
            logger.error(f"Failed to query Nominatim: {e}")
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
    return parsed_results


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
