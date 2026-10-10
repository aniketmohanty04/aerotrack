import re
import os

with open('backend/main.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Add WAQI_TOKEN to the top
if 'WAQI_TOKEN =' not in content:
    content = content.replace(
        'CACHE_TTL_AIR_SECONDS = 600  # 10 minutes',
        'CACHE_TTL_AIR_SECONDS = 600  # 10 minutes\nWAQI_TOKEN = os.getenv("WAQI_TOKEN", "6ce5b38420030ebaabef8ef19478badd7861b2d6")'
    )

# 2. Add _fetch_waqi_raw
waqi_func = """
async def _fetch_waqi_raw(lat: float, lon: float) -> Dict[str, Any]:
    url = f"https://api.waqi.info/feed/geo:{lat};{lon}/?token={WAQI_TOKEN}"
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        data = resp.json()
        if data.get("status") != "ok":
            raise ValueError(f"WAQI API Error: {data.get('data')}")
        return data

"""
if '_fetch_waqi_raw' not in content:
    content = content.replace(
        'async def _fetch_open_meteo_raw',
        waqi_func + 'async def _fetch_open_meteo_raw'
    )

# 3. Rewrite get_air_quality
old_get_air_quality = """@app.get("/api/air-quality/{lat}/{lon}")
async def get_air_quality(lat: float, lon: float):
    \"\"\"
    Fetches live real-time air quality data for given latitude & longitude from Open-Meteo.
    Shares a 10-minute cache with /api/trends to eliminate redundant API calls.
    \"\"\"
    lat, lon = validate_coords(lat, lon)
    try:
        data = await get_shared_open_meteo_data(lat, lon)

        current = data.get("current", {})
        units = data.get("current_units", {})

        original_us_aqi = current.get("us_aqi")
        if original_us_aqi is None and current.get("pm2_5") is not None:
            original_us_aqi = calculate_pm25_to_us_aqi(current.get("pm2_5"))

        us_aqi = original_us_aqi
        if us_aqi is not None and us_aqi > 500:
            us_aqi = 500

        european_aqi = current.get("european_aqi")
        if european_aqi is not None and european_aqi > 500:
            european_aqi = 500

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

        is_capped = bool(original_us_aqi is not None and original_us_aqi > 500)

        return {
            "latitude": lat,
            "longitude": lon,
            "time": current.get("time"),
            "timezone": data.get("timezone", "UTC"),
            "us_aqi": us_aqi,
            "european_aqi": european_aqi,
            "us_aqi_capped": is_capped,
            "is_capped": is_capped,
            "aqi_info": aqi_info,
            "pollutants": pollutants
        }

    except httpx.HTTPError as e:
        logger.error(f"Open-Meteo API request failed: {e}")
        raise HTTPException(status_code=502, detail="Upstream air quality provider is currently unavailable.")
    except Exception as e:
        logger.error(f"Unexpected error in /api/air-quality: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An error occurred while processing air quality data.")"""

new_get_air_quality = """@app.get("/api/air-quality/{lat}/{lon}")
async def get_air_quality(lat: float, lon: float):
    \"\"\"
    Fetches live real-time air quality data for given latitude & longitude from WAQI (Ground Truth).
    \"\"\"
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
        raise HTTPException(status_code=500, detail="An error occurred while processing air quality data.")"""

if old_get_air_quality in content:
    content = content.replace(old_get_air_quality, new_get_air_quality)

with open('backend/main.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("Updated backend/main.py")
