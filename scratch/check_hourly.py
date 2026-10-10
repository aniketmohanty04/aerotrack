"""Check Open-Meteo's FULL hourly history to see if AQI actually changed."""
import urllib.request, json

url = ("https://air-quality-api.open-meteo.com/v1/air-quality"
       "?latitude=28.6139&longitude=77.2090"
       "&hourly=us_aqi,pm2_5,pm10,ozone,nitrogen_dioxide"
       "&timezone=auto&start_date=2026-10-09&end_date=2026-10-10")
d = json.loads(urllib.request.urlopen(url, timeout=15).read().decode())
hourly = d["hourly"]
times = hourly["time"]
aqis = hourly["us_aqi"]
pm25s = hourly["pm2_5"]
pm10s = hourly["pm10"]
ozones = hourly["ozone"]
no2s = hourly["nitrogen_dioxide"]

print(f"Open-Meteo hourly for Delhi (Oct 9-10, IST)")
print(f"{'Time':<20} {'US_AQI':>8} {'PM2.5':>8} {'PM10':>8} {'O3':>8} {'NO2':>8}")
print("-" * 72)
for i in range(len(times)):
    t = times[i]
    a = aqis[i] if aqis[i] is not None else "-"
    p = pm25s[i] if pm25s[i] is not None else "-"
    p10 = pm10s[i] if pm10s[i] is not None else "-"
    o3 = ozones[i] if ozones[i] is not None else "-"
    n = no2s[i] if no2s[i] is not None else "-"
    print(f"  {t:<20} {str(a):>8} {str(p):>8} {str(p10):>8} {str(o3):>8} {str(n):>8}")
