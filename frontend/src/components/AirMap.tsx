import React, { useEffect, useRef } from 'react';
import { MapContainer, TileLayer, Marker, Popup, useMap, useMapEvents } from 'react-leaflet';
import L from 'leaflet';
import axios from 'axios';
import { MapPin, Navigation } from 'lucide-react';
import { AQIInfo } from '../types';

interface AirMapProps {
  lat: number;
  lon: number;
  locationName?: string;
  onSelect: (lat: number, lon: number, displayName?: string) => void;
  onLocationNameResolved?: (name: string) => void;
  aqiInfo?: AQIInfo;
  aqiValue?: number | null;
  isLoading?: boolean;
}

// Custom pulsing Leaflet marker icon using SVG to avoid missing asset paths
const createCustomMarker = (color: string = '#06b6d4') => {
  return L.divIcon({
    className: 'custom-pin-marker',
    html: `
      <div style="position: relative; width: 32px; height: 32px; display: flex; align-items: center; justify-content: center;">
        <div style="position: absolute; width: 32px; height: 32px; border-radius: 50%; background-color: ${color}; opacity: 0.35; animation: ping 1.5s cubic-bezier(0, 0, 0.2, 1) infinite;"></div>
        <div style="position: relative; width: 22px; height: 22px; border-radius: 50%; background-color: ${color}; border: 3px solid #ffffff; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.5); display: flex; align-items: center; justify-content: center;">
          <div style="width: 6px; height: 6px; border-radius: 50%; background-color: white;"></div>
        </div>
      </div>
    `,
    iconSize: [32, 32],
    iconAnchor: [16, 16],
    popupAnchor: [0, -16],
  });
};

const API_BASE = import.meta.env.VITE_API_URL || '';

// Map click listener hook
const MapClickHandler: React.FC<{
  onSelect: (lat: number, lon: number, displayName?: string) => void;
  onLocationNameResolved?: (name: string) => void;
}> = ({ onSelect, onLocationNameResolved }) => {
  const onSelectRef = useRef(onSelect);
  onSelectRef.current = onSelect;
  const onNameRef = useRef(onLocationNameResolved);
  onNameRef.current = onLocationNameResolved;
  const clickCounterRef = useRef<number>(0);

  useMapEvents({
    async click(e) {
      const wrapped = e.latlng.wrap();
      const lat = Math.max(-89.9, Math.min(89.9, wrapped.lat));
      const lon = ((wrapped.lng + 180) % 360 + 360) % 360 - 180;
      const currentClickId = ++clickCounterRef.current;

      // 1. Immediately trigger location selection and fresh data calculation
      const fallbackName = `${lat.toFixed(3)}°, ${lon.toFixed(3)}°`;
      onSelectRef.current(lat, lon, fallbackName);

      // 2. Fetch reverse-geocoded place name asynchronously in the background with timeout
      try {
        const res = await axios.get(`${API_BASE}/api/reverse-geocode/${lat}/${lon}`, {
          timeout: 6000,
        });
        if (clickCounterRef.current !== currentClickId) return;
        const { city, region, country } = res.data || {};
        const parts = [city, region, country].filter(Boolean);
        if (parts.length > 0 && onNameRef.current) {
          onNameRef.current(parts.join(", "));
        }
      } catch (err) {
        /* keep existing coordinate fallback */
      }
    },
  });
  return null;
};

// Auto pan controller when coordinates change
const MapViewController: React.FC<{ lat: number; lon: number }> = ({ lat, lon }) => {
  const map = useMap();
  useEffect(() => {
    map.flyTo([lat, lon], map.getZoom(), { duration: 1.2 });
  }, [lat, lon, map]);
  return null;
};

export const AirMap: React.FC<AirMapProps> = ({
  lat,
  lon,
  locationName,
  onSelect,
  onLocationNameResolved,
  aqiInfo,
  aqiValue,
  isLoading = false,
}) => {
  const isCalculating = isLoading;
  const markerColor = isCalculating ? '#06b6d4' : (aqiInfo?.color || '#06b6d4');
  const customIcon = createCustomMarker(markerColor);
  const markerRef = useRef<L.Marker>(null);

  useEffect(() => {
    if (markerRef.current) {
      markerRef.current.openPopup();
    }
  }, [lat, lon]);

  return (
    <div className="relative w-full h-[420px] rounded-2xl overflow-hidden border border-slate-800 shadow-2xl bg-slate-900">
      <div className="absolute top-3 left-3 z-[400] flex items-center gap-2 bg-slate-950/85 backdrop-blur-md px-3.5 py-1.5 rounded-xl border border-slate-800 text-xs text-slate-300 pointer-events-none shadow-lg">
        <MapPin className="w-4 h-4 text-cyan-400" />
        <span>Click anywhere on the map to inspect air quality</span>
      </div>

      <MapContainer
        center={[lat, lon]}
        zoom={6}
        scrollWheelZoom={true}
        worldCopyJump={true}
        className="w-full h-full"
      >
        <TileLayer
          attribution='&copy; OpenStreetMap contributors'
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />
        <MapClickHandler onSelect={onSelect} onLocationNameResolved={onLocationNameResolved} />
        <MapViewController lat={lat} lon={lon} />
        <Marker ref={markerRef} position={[lat, lon]} icon={customIcon}>
          <Popup>
            <div className="p-1 space-y-1.5 text-slate-200 min-w-[170px]">
              <div className="flex items-center gap-1.5 text-xs font-semibold text-slate-100">
                <MapPin className="w-3.5 h-3.5 text-cyan-400 flex-shrink-0" />
                <span className="truncate max-w-[200px]">
                  {locationName || `${lat.toFixed(4)}, ${lon.toFixed(4)}`}
                </span>
              </div>
              <div className="flex items-center gap-1.5 text-[10px] text-slate-400">
                <Navigation className="w-3 h-3 text-cyan-500 flex-shrink-0" />
                <span>{lat.toFixed(4)}, {lon.toFixed(4)}</span>
              </div>

              {isCalculating ? (
                <div className="space-y-1 pt-0.5">
                  <div className="text-sm font-semibold flex items-center gap-2">
                    <span className="text-slate-300">AQI:</span>
                    <span className="px-2 py-0.5 rounded text-xs font-bold text-cyan-300 bg-cyan-950/90 border border-cyan-500/40 animate-pulse">
                      Calculating...
                    </span>
                  </div>
                  <p className="text-[10px] text-cyan-400/90 leading-tight flex items-center gap-1">
                    <span className="inline-block w-1.5 h-1.5 rounded-full bg-cyan-400 animate-ping flex-shrink-0" />
                    Measuring real-time sensors...
                  </p>
                </div>
              ) : aqiValue != null ? (
                <div className="space-y-1 pt-0.5">
                  <div className="text-sm font-semibold flex items-center gap-2">
                    <span>AQI:</span>
                    <span
                      className="px-2 py-0.5 rounded text-xs font-bold text-white shadow-sm"
                      style={{ backgroundColor: aqiInfo?.color || '#06b6d4' }}
                    >
                      {aqiValue} - {aqiInfo?.category || 'Moderate'}
                    </span>
                  </div>
                  {aqiInfo?.description && (
                    <p className="text-[11px] text-slate-300 max-w-[200px] leading-tight">
                      {aqiInfo.description}
                    </p>
                  )}
                </div>
              ) : (
                <div className="space-y-1 pt-0.5">
                  <div className="text-sm font-semibold flex items-center gap-2">
                    <span className="text-slate-300">AQI:</span>
                    <span className="px-2 py-0.5 rounded text-xs font-bold text-slate-300 bg-slate-800 border border-slate-700">
                      Unavailable
                    </span>
                  </div>
                  <p className="text-[10px] text-slate-400 leading-tight">
                    Sensor readings unavailable for this location.
                  </p>
                </div>
              )}
            </div>
          </Popup>
        </Marker>
      </MapContainer>
    </div>
  );
};
