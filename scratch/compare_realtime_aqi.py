import urllib.request
import json
import time

cities = [
    {"name": "New Delhi", "lat": 28.6139, "lon": 77.2090},
    {"name": "London", "lat": 51.5074, "lon": -0.1278},
    {"name": "New York", "lat": 40.7128, "lon": -74.0060},
    {"name": "Tokyo", "lat": 35.6762, "lon": 139.6503},
    {"name": "Sydney", "lat": -33.8688, "lon": 151.2093}
]

def fetch_json(url):
    req = urllib.request.Request(url, headers={'User-Agent': 'AeroTrack-SeniorDevAudit/1.0'})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode('utf-8'))
    except Exception as e:
        return {"error": str(e)}

print("=" * 80)
print("AEROTRACK REAL-TIME AQI AUDIT & GROUND TRUTH VALIDATION")
print("=" * 80)

for c in cities:
    name = c["name"]
    lat = c["lat"]
    lon = c["lon"]

    # 1. Open-Meteo direct
    om_url = f"https://air-quality-api.open-meteo.com/v1/air-quality?latitude={lat}&longitude={lon}&current=us_aqi,european_aqi,pm2_5,pm10,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone"
    om_data = fetch_json(om_url)
    om_current = om_data.get("current", {})

    # 2. Local AeroTrack Backend
    local_url = f"http://127.0.0.1:8000/api/air-quality/{lat}/{lon}"
    local_data = fetch_json(local_url)

    # 3. Production AeroTrack Backend
    prod_url = f"https://aerotrack-backend-1tlt.onrender.com/api/air-quality/{lat}/{lon}"
    prod_data = fetch_json(prod_url)

    # 4. WAQI / Public reference (feed via public search/geoloc tokenless API or OpenAQ)
    # Using WAQI free public token 'demo' or OpenAQ if possible, or print OM raw vs AeroTrack
    waqi_url = f"https://api.waqi.info/feed/geo:{lat};{lon}/?token=demo"
    waqi_data = fetch_json(waqi_url)
    waqi_aqi = "N/A"
    waqi_station = "N/A"
    if waqi_data.get("status") == "ok":
        d = waqi_data.get("data", {})
        waqi_aqi = d.get("aqi")
        waqi_station = d.get("city", {}).get("name", "Unknown")

    print(f"\nCity: {name} (Coords: {lat}, {lon})")
    print(f"  [Direct Open-Meteo] PM2.5: {om_current.get('pm2_5')} ug/m3 | US AQI: {om_current.get('us_aqi')} | EU AQI: {om_current.get('european_aqi')} | Time: {om_current.get('time')}")
    print(f"  [Local AeroTrack]  PM2.5: {local_data.get('pollutants', {}).get('pm2_5', {}).get('value')} ug/m3 | US AQI: {local_data.get('us_aqi')} (Capped: {local_data.get('is_capped')}) | Status: {local_data.get('aqi_info', {}).get('category')}")
    print(f"  [Prod AeroTrack]   PM2.5: {prod_data.get('pollutants', {}).get('pm2_5', {}).get('value')} ug/m3 | US AQI: {prod_data.get('us_aqi')} | Time: {prod_data.get('time')}")
    clean_station = str(waqi_station).encode('ascii', 'replace').decode('ascii')
    print(f"  [WAQI Ground Stn]  AQI: {waqi_aqi} | Nearest Station: {clean_station}")

    # Check alignment
    om_us = om_current.get("us_aqi")
    local_us = local_data.get("us_aqi")
    diff = abs(om_us - local_us) if (om_us is not None and local_us is not None) else None
    print(f"  => Telemetry match (Local vs Upstream Open-Meteo): {'PERFECT (0 delta)' if diff == 0 else f'DIFF: {diff}'}")
