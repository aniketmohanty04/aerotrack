import React from 'react';
import { AirQualityData, PollutantDetail } from '../types';
import { Wind, ShieldAlert, Activity, Flame, Droplets, Sun, Gauge, Info, AlertCircle, RefreshCw } from 'lucide-react';

interface PollutantCardsProps {
  data?: AirQualityData | null;
  isLoading?: boolean;
  onRetry?: () => void;
}

const getBorderColorClass = (rating: PollutantDetail['rating']) => {
  switch (rating) {
    case 'good':
      return 'border-emerald-500/60 shadow-emerald-950/20 bg-emerald-950/10 text-emerald-400';
    case 'moderate':
      return 'border-amber-500/60 shadow-amber-950/20 bg-amber-950/10 text-amber-400';
    case 'unhealthy_sensitive':
      return 'border-orange-500/60 shadow-orange-950/20 bg-orange-950/10 text-orange-400';
    case 'unhealthy':
      return 'border-red-500/60 shadow-red-950/20 bg-red-950/10 text-red-400';
    case 'very_unhealthy':
      return 'border-purple-500/60 shadow-purple-950/20 bg-purple-950/10 text-purple-400';
    case 'hazardous':
      return 'border-rose-700/80 shadow-rose-950/30 bg-rose-950/20 text-rose-300';
    default:
      return 'border-slate-800 bg-slate-900/60 text-slate-400';
  }
};

const getRatingBadge = (rating: PollutantDetail['rating']) => {
  switch (rating) {
    case 'good':
      return { text: 'Good', bg: 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40' };
    case 'moderate':
      return { text: 'Moderate', bg: 'bg-amber-500/20 text-amber-300 border-amber-500/40' };
    case 'unhealthy_sensitive':
      return { text: 'Unhealthy (SG)', bg: 'bg-orange-500/20 text-orange-300 border-orange-500/40' };
    case 'unhealthy':
      return { text: 'Unhealthy', bg: 'bg-red-500/20 text-red-300 border-red-500/40' };
    case 'very_unhealthy':
      return { text: 'Very Unhealthy', bg: 'bg-purple-500/20 text-purple-300 border-purple-500/40' };
    case 'hazardous':
      return { text: 'Hazardous', bg: 'bg-rose-600/30 text-rose-200 border-rose-500/60' };
    default:
      return { text: 'Normal', bg: 'bg-slate-700/30 text-slate-300 border-slate-600/40' };
  }
};

const getPollutantIcon = (key: string) => {
  switch (key) {
    case 'pm2_5':
      return <Activity className="w-4 h-4" />;
    case 'pm10':
      return <Wind className="w-4 h-4" />;
    case 'nitrogen_dioxide':
      return <Flame className="w-4 h-4" />;
    case 'sulphur_dioxide':
      return <Droplets className="w-4 h-4" />;
    case 'ozone':
      return <Sun className="w-4 h-4" />;
    case 'carbon_monoxide':
      return <Gauge className="w-4 h-4" />;
    default:
      return <Wind className="w-4 h-4" />;
  }
};

const PollutantSkeleton: React.FC = () => {
  return (
    <div className="space-y-4 animate-pulse">
      {/* Primary AQI Highlight Header Skeleton */}
      <div className="rounded-2xl border-2 border-slate-800 bg-slate-900/60 p-6 shadow-xl backdrop-blur-xl">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div className="space-y-3">
            <div className="flex items-center gap-2">
              <div className="w-5 h-5 rounded-full bg-slate-800"></div>
              <div className="h-3 w-36 bg-slate-800 rounded"></div>
            </div>
            <div className="flex items-baseline gap-3">
              <div className="h-12 w-24 bg-slate-800 rounded-xl"></div>
              <div className="h-7 w-28 bg-slate-800 rounded-lg"></div>
            </div>
            <div className="h-4 w-72 bg-slate-800/80 rounded"></div>
          </div>
          <div className="flex gap-4 border-t md:border-t-0 md:border-l border-slate-800/80 pt-3 md:pt-0 md:pl-6">
            <div className="bg-slate-950/80 px-4 py-2.5 rounded-xl border border-slate-800/80 w-24 space-y-2">
              <div className="h-2.5 w-12 bg-slate-800 rounded mx-auto"></div>
              <div className="h-5 w-8 bg-slate-800 rounded mx-auto"></div>
            </div>
            <div className="bg-slate-950/80 px-4 py-2.5 rounded-xl border border-slate-800/80 w-24 space-y-2">
              <div className="h-2.5 w-14 bg-slate-800 rounded mx-auto"></div>
              <div className="h-5 w-8 bg-slate-800 rounded mx-auto"></div>
            </div>
          </div>
        </div>
      </div>

      {/* Grid of 6 Pollutant Skeleton Cards */}
      <div
        className="grid gap-3.5 w-full"
        style={{
          gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))',
          maxWidth: '100%',
          overflow: 'hidden',
        }}
      >
        {[1, 2, 3, 4, 5, 6].map((i) => (
          <div
            key={i}
            className="rounded-2xl border-2 border-slate-800/80 bg-slate-900/40 p-4 shadow-lg backdrop-blur-md flex flex-col justify-between"
          >
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <div className="w-7 h-7 rounded-lg bg-slate-800"></div>
                <div className="h-4 w-14 bg-slate-800 rounded"></div>
              </div>
              <div className="h-4 w-14 bg-slate-800 rounded-md"></div>
            </div>
            <div className="h-3 w-32 bg-slate-800/60 rounded mt-2.5"></div>
            <div className="flex items-baseline gap-2 mt-2">
              <div className="h-7 w-16 bg-slate-800 rounded-lg"></div>
              <div className="h-3 w-8 bg-slate-800/60 rounded"></div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export const PollutantCards: React.FC<PollutantCardsProps> = ({ data, isLoading, onRetry }) => {
  if (isLoading) {
    return <PollutantSkeleton />;
  }

  if (!data) {
    return (
      <div className="rounded-2xl border border-slate-800 bg-slate-900/60 p-8 shadow-xl backdrop-blur-xl text-center space-y-4">
        <div className="w-12 h-12 rounded-2xl bg-amber-500/10 border border-amber-500/30 flex items-center justify-center mx-auto text-amber-400">
          <AlertCircle className="w-6 h-6" />
        </div>
        <div className="space-y-1">
          <h3 className="text-base font-bold text-white">Air Quality Telemetry Unavailable</h3>
          <p className="text-xs text-slate-400 max-w-md mx-auto">
            Real-time sensor readings could not be retrieved from atmospheric monitoring stations for this location. Please check your connection or select another location on the map.
          </p>
        </div>
        {onRetry && (
          <button
            onClick={onRetry}
            className="px-4 py-2 rounded-xl text-xs font-semibold bg-cyan-600 hover:bg-cyan-500 text-white transition-all shadow-md inline-flex items-center gap-1.5"
          >
            <RefreshCw className="w-3.5 h-3.5" />
            Retry Telemetry
          </button>
        )}
      </div>
    );
  }

  const { us_aqi, european_aqi, aqi_info, pollutants } = data;

  const pollutantList: Array<{ key: string; item: PollutantDetail }> = [
    { key: 'pm2_5', item: pollutants.pm2_5 },
    { key: 'pm10', item: pollutants.pm10 },
    { key: 'nitrogen_dioxide', item: pollutants.nitrogen_dioxide },
    { key: 'sulphur_dioxide', item: pollutants.sulphur_dioxide },
    { key: 'ozone', item: pollutants.ozone },
    { key: 'carbon_monoxide', item: pollutants.carbon_monoxide },
  ];

  return (
    <div className="space-y-4">
      {/* Primary AQI Highlight Header Card */}
      <div
        className="relative overflow-hidden rounded-2xl border-2 p-6 shadow-xl backdrop-blur-xl transition-all"
        style={{
          borderColor: aqi_info.color,
          background: `linear-gradient(135deg, ${aqi_info.color}15 0%, #090d16 100%)`
        }}
      >
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div className="space-y-1.5">
            <div className="flex items-center gap-2">
              <ShieldAlert className="w-5 h-5" style={{ color: aqi_info.color }} />
              <span className="text-xs font-semibold tracking-wider uppercase text-slate-400">
                Air Quality Index (AQI)
              </span>
            </div>
            <div className="flex items-baseline gap-3">
              <span className="text-5xl font-extrabold tracking-tight text-white">
                {us_aqi ?? 'N/A'}
              </span>
              <span
                className="text-lg font-bold px-3 py-1 rounded-lg"
                style={{
                  backgroundColor: `${aqi_info.color}25`,
                  color: aqi_info.color,
                  border: `1px solid ${aqi_info.color}50`
                }}
              >
                {aqi_info.category}
              </span>
            </div>
            <p className="text-sm text-slate-300 max-w-xl">
              {aqi_info.description}
            </p>
          </div>

          <div className="flex gap-4 border-t md:border-t-0 md:border-l border-slate-800 pt-3 md:pt-0 md:pl-6">
            <div className="bg-slate-900/80 px-4 py-2.5 rounded-xl border border-slate-800 text-center">
              <span className="text-[11px] text-slate-400 block uppercase font-medium">US AQI</span>
              <span className="text-xl font-bold text-white">{us_aqi ?? '--'}</span>
            </div>
            <div className="bg-slate-900/80 px-4 py-2.5 rounded-xl border border-slate-800 text-center">
              <span className="text-[11px] text-slate-400 block uppercase font-medium">European AQI</span>
              <span className="text-xl font-bold text-white">{european_aqi ?? '--'}</span>
            </div>
          </div>
        </div>
      </div>

      {/* Grid of Pollutant Cards with Auto-Fit and Color-Coded Borders */}
      <div
        className="grid gap-3.5 w-full"
        style={{
          gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))',
          maxWidth: '100%',
          overflow: 'hidden',
        }}
      >
        {pollutantList.map(({ key, item }) => {
          const badge = getRatingBadge(item.rating);
          const borderClass = getBorderColorClass(item.rating);

          return (
            <div
              key={key}
              className={`rounded-2xl border-2 p-4 shadow-lg backdrop-blur-md transition-all duration-200 hover:scale-[1.01] flex flex-col justify-between ${borderClass}`}
            >
              {/* Header row: Icon + Label + Info on Left, Badge on Right */}
              <div className="flex items-center justify-between gap-2">
                <div className="flex items-center gap-2 min-w-0">
                  <div className="p-1.5 rounded-lg bg-slate-900/80 border border-slate-800 text-slate-300 flex-shrink-0">
                    {getPollutantIcon(key)}
                  </div>
                  <div className="flex items-center gap-1.5 min-w-0">
                    <h3 className="font-bold text-sm text-white whitespace-nowrap">{item.label}</h3>
                    <span
                      title={item.description}
                      className="cursor-help text-slate-400 hover:text-cyan-400 transition-colors inline-flex items-center flex-shrink-0"
                      aria-label={`Info about ${item.label}`}
                    >
                      <Info className="w-3.5 h-3.5" />
                    </span>
                  </div>
                </div>

                <span className={`text-[10px] uppercase font-bold px-2 py-0.5 rounded-md border flex-shrink-0 whitespace-nowrap ${badge.bg}`}>
                  {badge.text}
                </span>
              </div>

              {/* Sub-label: Atmospheric Concentration on its own row across full card width */}
              <div className="text-[11px] text-slate-400 font-medium mt-2 whitespace-nowrap">
                Atmospheric Concentration
              </div>

              {/* Value and Unit row */}
              <div className="mt-1 flex items-baseline gap-1.5">
                <span className="text-2xl sm:text-3xl font-extrabold text-white tracking-tight">
                  {item.value !== null ? item.value : '--'}
                </span>
                <span className="text-xs text-slate-400 font-medium">
                  {item.unit}
                </span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};
