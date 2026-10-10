"""Diagnose why AQI appears stuck — check Open-Meteo's own update frequency."""
import urllib.request, json, time
from datetime import datetime, timezone, timedelta

IST = timezone(timedelta(hours=5, minutes=30))
now_ist = datetime.now(IST)
print(f"Current IST time: {now_ist.strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 70)

# 1. Query Open-Meteo DIRECTLY with timezone=auto for Delhi
url_tz = "https://air-quality-api.open-meteo.com/v1/air-quality?latitude=28.6139&longitude=77.2090&current=us_aqi,european_aqi,pm2_5,pm10,ozone,nitrogen_dioxide&timezone=auto"
d = json.loads(urllib.request.urlopen(url_tz, timeout=10).read().decode())
cur = d["current"]
print(f"[Open-Meteo timezone=auto]")
print(f"  Timezone resolved: {d.get('timezone')}")
print(f"  current.time     : {cur['time']}")
print(f"  us_aqi           : {cur['us_aqi']}")
print(f"  pm2_5            : {cur['pm2_5']}")
print(f"  pm10             : {cur['pm10']}")
print()

# 2. Query with explicit hourly for the last 6 hours to see if values actually change
url_hourly = "https://air-quality-api.open-meteo.com/v1/air-quality?latitude=28.6139&longitude=77.2090&hourly=us_aqi,pm2_5,pm10&timezone=auto&past_hours=12&forecast_hours=0"
d2 = json.loads(urllib.request.urlopen(url_hourly, timeout=10).read().decode())
hourly = d2.get("hourly", {})
times = hourly.get("time", [])
aqis = hourly.get("us_aqi", [])
pm25s = hourly.get("pm2_5", [])
print(f"[Open-Meteo hourly last 12h for Delhi]")
print(f"  {'Time':<20} {'US_AQI':>8} {'PM2.5':>8}")
for t, a, p in zip(times[-12:], aqis[-12:], pm25s[-12:]):
    print(f"  {t:<20} {str(a):>8} {str(p):>8}")
print()

# 3. Check what AeroTrack's backend fetch URL looks like
print("[AeroTrack backend Open-Meteo fetch URL check]")
# Read main.py to find the _fetch_open_meteo_raw function
with open("backend/main.py", "r") as f:
    lines = f.readlines()
for i, line in enumerate(lines):
    if "_fetch_open_meteo_raw" in line or "air-quality-api.open-meteo.com" in line:
        print(f"  L{i+1}: {line.rstrip()}")
