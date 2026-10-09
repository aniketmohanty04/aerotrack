import React, { useState, useEffect, useCallback } from 'react';
import axios from 'axios';
import {
  AirQualityData,
  TrendData,
  ForecastData,
  LocationPreset,
} from './types';
import { AirMap } from './components/AirMap';
import { PollutantCards } from './components/PollutantCards';
import { TrendChart } from './components/TrendChart';
import { ForecastChart } from './components/ForecastChart';
import { Recommendations } from './components/Recommendations';
import { SearchBar } from './components/SearchBar';
import {
  RefreshCw,
  Wind,
  MapPin,
  Clock,
  Compass,
} from 'lucide-react';

// Preset locations for quick navigation
const PRESET_LOCATIONS: LocationPreset[] = [
  { name: 'New Delhi', lat: 28.6139, lon: 77.2090, country: 'India' },
  { name: 'London', lat: 51.5074, lon: -0.1278, country: 'United Kingdom' },
  { name: 'New York', lat: 40.7128, lon: -74.0060, country: 'United States' },
  { name: 'Tokyo', lat: 35.6762, lon: 139.6503, country: 'Japan' },
  { name: 'Beijing', lat: 39.9042, lon: 116.4074, country: 'China' },
  { name: 'Paris', lat: 48.8566, lon: 2.3522, country: 'France' },
];

const API_BASE = import.meta.env.VITE_API_URL || '';

const formatCoords = (latitude: number, longitude: number): string => {
  const latDir = latitude >= 0 ? 'N' : 'S';
  const lonDir = longitude >= 0 ? 'E' : 'W';
  return `${Math.abs(latitude).toFixed(4)}° ${latDir}, ${Math.abs(longitude).toFixed(4)}° ${lonDir}`;
};

export const App: React.FC = () => {
  // Location state
  const [lat, setLat] = useState<number>(28.6139);
  const [lon, setLon] = useState<number>(77.2090);
  const [locationName, setLocationName] = useState<string>('New Delhi, India');

  // Data states
  const [airQuality, setAirQuality] = useState<AirQualityData | null>(null);
  const [trends, setTrends] = useState<TrendData | null>(null);
  const [forecast, setForecast] = useState<ForecastData | null>(null);
  const [forecastError, setForecastError] = useState<string | null>(null);

  // Loading states
  const [isAirQualityLoading, setIsAirQualityLoading] = useState<boolean>(true);
  const [isTrendsLoading, setIsTrendsLoading] = useState<boolean>(true);
  const [isForecastLoading, setIsForecastLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdated, setLastUpdated] = useState<string>('');

  // Fetch all 3 endpoints concurrently with independent state updates for instant rendering
  const fetchAllData = useCallback(async (targetLat: number, targetLon: number) => {
    // Immediately clear previous location data so old AQI never hovers or flashes on the new location
    setAirQuality(null);
    setTrends(null);
    setForecast(null);
    setIsAirQualityLoading(true);
    setIsTrendsLoading(true);
    setIsForecastLoading(true);
    setError(null);
    setForecastError(null);

    // 1. Live Air Quality telemetry (fastest: ~150-250ms)
    const pAirQuality = axios
      .get<AirQualityData>(`${API_BASE}/api/air-quality/${targetLat}/${targetLon}`)
      .then((res) => {
        setAirQuality(res.data);
        setLastUpdated(new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }));
      })
      .catch((err) => {
        setAirQuality(null);
        if (err.response?.status === 400) {
          setError("Invalid location. Please click somewhere within the valid map range.");
        } else if (err.response?.status && err.response.status >= 500) {
          setError("Server error. Please try again in a moment.");
        } else {
          setError("Failed to fetch air quality data. Check your connection.");
        }
      })
      .finally(() => setIsAirQualityLoading(false));

    // 2. 7-Day Historical Trends (~200-350ms)
    const pTrends = axios
      .get<TrendData>(`${API_BASE}/api/trends/${targetLat}/${targetLon}`)
      .then((res) => setTrends(res.data))
      .catch(() => setTrends(null))
      .finally(() => setIsTrendsLoading(false));

    // 3. 24-Hour ML PM2.5 Forecast with Split Conformal Intervals (~1-3s)
    const pForecast = axios
      .get<ForecastData>(`${API_BASE}/api/predict/${targetLat}/${targetLon}`)
      .then((res) => {
        const fData = res.data;
        if (fData.status === 'error') {
          setForecast(null);
          setForecastError((fData as any).message || 'Forecast unavailable.');
        } else {
          setForecast(fData);
          setForecastError(null);
        }
      })
      .catch((err) => {
        const errResp = err.response?.data;
        const errMsg =
          errResp?.message ||
          errResp?.detail?.message ||
          (typeof errResp?.detail === 'string' ? errResp.detail : null) ||
          'Historical air quality observations are unavailable or insufficient for this location.';
        setForecast(null);
        setForecastError(errMsg);
      })
      .finally(() => setIsForecastLoading(false));

    await Promise.allSettled([pAirQuality, pTrends, pForecast]);
  }, []);

  // Handle location update with optional displayName
  const handleSelect = (newLat: number, newLon: number, displayName?: string) => {
    const isNewCoords = Math.abs(newLat - lat) > 1e-4 || Math.abs(newLon - lon) > 1e-4;
    if (isNewCoords) {
      // Immediately reset previous readings so new click always calculates and shows freshly
      setAirQuality(null);
      setTrends(null);
      setForecast(null);
      setIsAirQualityLoading(true);
      setIsTrendsLoading(true);
      setIsForecastLoading(true);
      setLat(newLat);
      setLon(newLon);
    }
    if (displayName) {
      setLocationName(displayName);
    } else if (isNewCoords) {
      setLocationName(`${newLat.toFixed(3)}°, ${newLon.toFixed(3)}°`);
    }
  };

  const handleSelectLocation = handleSelect;

  // Opt-in explicit demo forecast loader for demonstrations or when upstream data is unavailable
  const handleEnableDemoForecast = async () => {
    setIsForecastLoading(true);
    setForecastError(null);
    try {
      const res = await axios.get<ForecastData>(`${API_BASE}/api/predict/${lat}/${lon}?allow_demo=true`);
      if (res.data.status === 'error') {
        setForecast(null);
        setForecastError((res.data as any).message || 'Failed to load demo forecast.');
      } else {
        setForecast(res.data);
      }
    } catch (err: any) {
      const errResp = err.response?.data;
      const errMsg =
        errResp?.message ||
        errResp?.detail?.message ||
        (typeof errResp?.detail === 'string' ? errResp.detail : null) ||
        'Failed to load demo forecast.';
      setForecastError(errMsg);
    } finally {
      setIsForecastLoading(false);
    }
  };

  // Trigger data fetches on coordinates change
  useEffect(() => {
    fetchAllData(lat, lon);
  }, [lat, lon, fetchAllData]);

  const isAnyLoading = isAirQualityLoading || isTrendsLoading || isForecastLoading;

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans">
      {/* Top Navbar */}
      <header className="sticky top-0 z-50 backdrop-blur-xl bg-slate-950/80 border-b border-slate-800">
        <div className="max-w-[1700px] w-full mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-xl bg-gradient-to-tr from-cyan-600 to-teal-400 shadow-lg shadow-cyan-900/30">
              <Wind className="w-5 h-5 text-white" />
            </div>
            <div>
              <h1 className="text-base sm:text-lg font-extrabold tracking-tight text-white flex items-center gap-2">
                AeroTrack
                <span className="text-[10px] font-bold uppercase tracking-wider bg-cyan-500/10 text-cyan-400 border border-cyan-500/30 px-2 py-0.5 rounded-full">
                  ML Forecasting
                </span>
              </h1>
              <p className="text-[11px] text-slate-400 hidden sm:block">
                Open-Meteo Environmental Intelligence & XGBoost 24-Hour Forecast
              </p>
            </div>
          </div>

          {/* Quick Refresh & Time */}
          <div className="flex items-center gap-3">
            {lastUpdated && (
              <span className="hidden md:flex items-center gap-1.5 text-xs text-slate-400">
                <Clock className="w-3.5 h-3.5 text-slate-500" />
                Updated at {lastUpdated}
              </span>
            )}
            <button
              onClick={() => fetchAllData(lat, lon)}
              disabled={isAnyLoading}
              className="p-2 rounded-xl bg-slate-900 border border-slate-800 hover:border-slate-700 text-slate-300 hover:text-white transition-all disabled:opacity-50"
              title="Refresh Data"
            >
              <RefreshCw className={`w-4 h-4 ${isAnyLoading ? 'animate-spin text-cyan-400' : ''}`} />
            </button>
          </div>
        </div>
      </header>

      {/* Main Content Area */}
      <main className="flex-1 max-w-[1700px] w-full mx-auto px-4 sm:px-6 lg:px-8 py-6 space-y-6">
        {/* Search & Location Bar */}
        <section className="bg-slate-900/70 border border-slate-800 backdrop-blur-md rounded-2xl p-4 shadow-xl space-y-3">
          <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
            {/* Autocomplete Search Bar */}
            <SearchBar onSelect={handleSelect} />

            {/* Current Selected Location Indicator */}
            <div className="flex items-center gap-2 text-slate-300">
              <MapPin className="w-4 h-4 text-cyan-400 flex-shrink-0" />
              <span className="text-sm font-bold text-white">{locationName}</span>
              <span className="text-xs text-slate-400 font-mono">
                ({lat.toFixed(3)}°, {lon.toFixed(3)}°)
              </span>
            </div>
          </div>

          {/* Preset City Quick Buttons */}
          <div className="flex items-center gap-2 overflow-x-auto pb-1 pt-1">
            <span className="text-xs font-medium text-slate-400 flex items-center gap-1 flex-shrink-0">
              <Compass className="w-3.5 h-3.5 text-slate-500" /> Presets:
            </span>
            {PRESET_LOCATIONS.map((preset) => (
              <button
                key={preset.name}
                onClick={() => handleSelectLocation(preset.lat, preset.lon, `${preset.name}, ${preset.country}`)}
                className={`px-3 py-1 rounded-xl text-xs font-semibold whitespace-nowrap transition-all border ${
                  Math.abs(lat - preset.lat) < 0.05 && Math.abs(lon - preset.lon) < 0.05
                    ? 'bg-cyan-500/20 text-cyan-300 border-cyan-500/50 shadow-sm'
                    : 'bg-slate-950/80 text-slate-400 border-slate-800 hover:border-slate-700 hover:text-slate-200'
                }`}
              >
                {preset.name}
              </button>
            ))}
          </div>
        </section>

        {/* Dismissible Red Error Banner above the Map */}
        {error && (
          <div className="error-banner bg-red-600 text-white p-4 rounded-xl mb-6 flex items-center justify-between shadow-lg">
            <span className="font-medium text-sm">{error}</span>
            <button
              onClick={() => setError(null)}
              className="text-white hover:text-red-200 text-2xl font-bold leading-none px-2 py-1 rounded transition-colors focus:outline-none"
              aria-label="Dismiss error"
            >
              ×
            </button>
          </div>
        )}

        {/* Map and Main Live Cards Row */}
        <section className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
          {/* Leaflet Map (Left Column) */}
          <div className="lg:col-span-5 space-y-3">
            <div className="flex flex-col gap-1.5">
              <h2 className="text-sm font-bold text-slate-300 flex items-center gap-2">
                <MapPin className="w-4 h-4 text-cyan-400" />
                Interactive Air Quality Map
              </h2>
              <div className="bg-slate-900/90 border border-slate-800 rounded-xl px-3.5 py-2.5 shadow-sm">
                <div className="text-xs text-slate-400 font-medium">
                  Selected: <span className="text-sm font-bold text-white">{locationName}</span>
                </div>
                <div className="text-[11px] text-slate-400 font-mono mt-0.5">
                  {formatCoords(lat, lon)}
                </div>
              </div>
            </div>
            <AirMap
              lat={lat}
              lon={lon}
              locationName={locationName}
              onSelect={handleSelectLocation}
              aqiInfo={airQuality?.aqi_info}
              aqiValue={airQuality?.us_aqi}
              isLoading={isAirQualityLoading}
            />
          </div>

          {/* Pollutant Cards & Highlights (Right Column) */}
          <div className="lg:col-span-7 min-w-0 w-full">
            <div className="mb-2 text-xs font-semibold text-slate-300">
              Showing data for: <span className="text-white font-bold">{locationName}</span>
            </div>
            <PollutantCards data={airQuality} isLoading={isAirQualityLoading} />
          </div>
        </section>

        {/* Rule-Based Recommendations */}
        {airQuality && !isAirQualityLoading && (
          <Recommendations
            aqiValue={airQuality.us_aqi}
            aqiInfo={airQuality.aqi_info}
          />
        )}

        {/* Charts Grid: 7-Day Trends and 24-Hour ML Forecast */}
        <section className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* 7-Day Historical Trend Chart */}
          <div>
            {isTrendsLoading && !trends ? (
              <div className="h-[470px] rounded-2xl border border-slate-800 bg-slate-900/40 animate-pulse flex items-center justify-center">
                <p className="text-sm text-slate-400">Loading 7-day trend history...</p>
              </div>
            ) : trends ? (
              <TrendChart data={trends} />
            ) : null}
          </div>

          {/* 24-Hour XGBoost Forecast Chart */}
          <div>
            <ForecastChart
              data={forecast}
              isLoading={isForecastLoading}
              error={forecastError}
              onEnableDemo={handleEnableDemoForecast}
            />
          </div>
        </section>
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-800/80 bg-slate-950 py-6 mt-12 text-center text-xs text-slate-400 space-y-1">
        <p>AeroTrack • Live Environmental Intelligence with Open-Meteo & XGBoost</p>
        <p className="text-slate-400">FastAPI • Python • React • TypeScript • Recharts • Leaflet</p>
      </footer>
    </div>
  );
};

export default App;
