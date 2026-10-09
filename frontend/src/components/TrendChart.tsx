import React, { useState } from 'react';
import {
  ResponsiveContainer,
  ComposedChart,
  Area,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  ReferenceLine,
} from 'recharts';
import { TrendData } from '../types';
import { TrendingUp, Calendar, Info, AlertTriangle, CheckCircle2 } from 'lucide-react';

interface TrendChartProps {
  data: TrendData;
}

// Reusable unit badge with hover info tooltip
const Ugm3Unit: React.FC<{ iconSize?: string }> = ({ iconSize = 'w-3 h-3' }) => (
  <span
    title="Micrograms per cubic meter — the standard unit for measuring air pollutant concentration."
    className="cursor-help inline-flex items-center gap-1 text-slate-400 hover:text-cyan-300 transition-colors"
  >
    <span>µg/m³</span>
    <Info className={`${iconSize} opacity-75`} />
  </span>
);

// Reusable two-line SVG label renderer for Recharts ReferenceLine
const twoLineLabel = (title: string, subtitle: string, color: string) => ({ viewBox }: any) => {
  const x = (viewBox?.width ?? 0) + (viewBox?.x ?? 0) + 8;
  const y = (viewBox?.y ?? 0) + ((viewBox?.height ?? 0) / 2);
  return (
    <g>
      <text
        x={x}
        y={y - 8}
        fill={color}
        fontSize={12}
        fontWeight={600}
        dominantBaseline="middle"
      >
        {title}
      </text>
      <text
        x={x}
        y={y + 6}
        fill={color}
        fontSize={10}
        opacity={0.7}
        dominantBaseline="middle"
      >
        {subtitle}
      </text>
    </g>
  );
};

// Pure helper function: compute 24h historical average PM2.5 strictly from observations
export function computeHistorical24hAvg(
  trends: TrendData['trends'],
  backendLast24h?: number,
  fallbackAvg: number = 0
): number {
  if (typeof backendLast24h === 'number') {
    return backendLast24h;
  }
  const historicalObs = trends.filter((p) => !p.is_forecast);
  const last24hHistoricalObs = historicalObs.slice(-24);
  const last24hPm25 = last24hHistoricalObs
    .map((p) => p.pm2_5)
    .filter((v): v is number => typeof v === 'number' && !isNaN(v));

  return last24hPm25.length
    ? last24hPm25.reduce((a, b) => a + b, 0) / last24hPm25.length
    : fallbackAvg;
}

// Pure helper function: prepare composed chart data separating observations and forecasts
export function prepareTrendChartData(
  trends: TrendData['trends'],
  metric: 'pm2_5' | 'us_aqi' = 'pm2_5'
) {
  const lastObsIndex = trends.reduce((lastIdx, pt, idx) => (!pt.is_forecast ? idx : lastIdx), -1);

  return trends.map((pt, idx) => {
    const val = metric === 'pm2_5' ? pt.pm2_5 : pt.us_aqi;
    const isForecast = !!pt.is_forecast;

    return {
      ...pt,
      // Observed value only on historical observation points
      observed_val: !isForecast ? val : null,
      // Forecast value on future points, seamlessly bridged from the final observation
      forecast_val: isForecast || idx === lastObsIndex ? val : null,
      metric_val: val,
    };
  });
}

export const TrendChart: React.FC<TrendChartProps> = ({ data }) => {
  const [metric, setMetric] = useState<'pm2_5' | 'us_aqi'>('pm2_5');

  const { trends, stats } = data;

  // Compute round Y-axis domain and ticks
  const metricValues = trends
    .map((t) => (metric === 'pm2_5' ? t.pm2_5 : t.us_aqi))
    .filter((v): v is number => typeof v === 'number' && !isNaN(v));

  const dataMax = metricValues.length ? Math.max(...metricValues) : 50;
  const effectiveMax = metric === 'pm2_5' ? Math.max(15, dataMax) : Math.max(50, dataMax);

  // Round top up to the nearest 50 (minimum 50)
  const roundedMax = Math.max(50, Math.ceil(effectiveMax / 50) * 50);

  // Generate 4-5 clean, even tick steps
  let step = 50;
  if (roundedMax === 50) {
    step = 10;
  } else if (roundedMax === 100) {
    step = 25;
  } else if (roundedMax <= 250) {
    step = 50;
  } else {
    step = Math.ceil(roundedMax / 5 / 50) * 50;
  }

  const yTicks: number[] = [];
  for (let val = 0; val <= roundedMax; val += step) {
    yTicks.push(val);
  }

  // Calculate Last 24-Hour Average PM2.5 strictly from historical observations
  const last24hAvg = computeHistorical24hAvg(trends, stats.last_24h_avg_pm25, stats.avg_pm25);

  let trendSummaryText = 'Air quality has been good this week.';
  let trendSummaryStyle = 'bg-emerald-500/10 border-emerald-500/30 text-emerald-300';
  let TrendSummaryIcon = CheckCircle2;

  if (last24hAvg > 55) {
    trendSummaryText = 'Air quality has been unhealthy this week — see recommendations below.';
    trendSummaryStyle = 'bg-rose-500/10 border-rose-500/30 text-rose-300';
    TrendSummaryIcon = AlertTriangle;
  } else if (last24hAvg > 35) {
    trendSummaryText = 'Air quality has been moderate — sensitive groups should be cautious.';
    trendSummaryStyle = 'bg-amber-500/10 border-amber-500/30 text-amber-300';
    TrendSummaryIcon = AlertTriangle;
  }

  // Format X-axis day-level labels e.g. "Oct 2", "Oct 3"
  const formatTrendDate = (timeStr: string) => {
    try {
      const d = new Date(timeStr);
      if (!isNaN(d.getTime())) {
        return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
      }
    } catch {}
    return timeStr.slice(5, 10);
  };

  // Fixed 24-hour interval for 7-day data
  const dayInterval = Math.max(1, Math.round(trends.length / 7) - 1);

  // Split into observed vs forecast lines for visual and semantic distinction
  const lastObsIndex = trends.reduce((lastIdx, pt, idx) => (!pt.is_forecast ? idx : lastIdx), -1);
  const lastObsPoint = lastObsIndex >= 0 ? trends[lastObsIndex] : null;

  const chartData = prepareTrendChartData(trends, metric);

  const CustomTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length) {
      const point = payload[0].payload;
      const isForecast = !!point.is_forecast;

      return (
        <div className="bg-slate-900/95 border border-slate-700 p-3 rounded-xl shadow-2xl backdrop-blur-md text-xs space-y-1.5 min-w-[170px]">
          <div className="flex items-center justify-between gap-2 border-b border-slate-800 pb-1">
            <span className="font-semibold text-slate-300">{point.label || point.time.replace('T', ' ')}</span>
            <span className={`text-[10px] font-bold px-2 py-0.5 rounded-full ${
              isForecast
                ? 'bg-purple-500/20 text-purple-300 border border-purple-500/40'
                : 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40'
            }`}>
              {isForecast ? 'Forecast' : 'Observed'}
            </span>
          </div>

          <div className="flex items-center gap-2">
            <span className={`w-2.5 h-2.5 rounded-full ${isForecast ? 'bg-purple-400' : 'bg-cyan-400'}`}></span>
            <span className="text-slate-400">PM2.5:</span>
            <span className="font-bold text-white flex items-center gap-1">
              {point.pm2_5 != null ? point.pm2_5 : 'No data'} <Ugm3Unit iconSize="w-2.5 h-2.5" />
            </span>
          </div>

          <div className="flex items-center gap-2">
            <span className={`w-2.5 h-2.5 rounded-full ${isForecast ? 'bg-purple-400' : 'bg-amber-400'}`}></span>
            <span className="text-slate-400">US AQI:</span>
            <span className="font-bold text-white">{point.us_aqi != null ? point.us_aqi : 'N/A'}</span>
          </div>
        </div>
      );
    }
    return null;
  };

  const dataLineColor = metric === 'pm2_5' ? '#22d3ee' : '#f59e0b';

  return (
    <div
      className="space-y-4"
      style={{
        background: 'rgba(255, 255, 255, 0.02)',
        border: '1px solid rgba(255, 255, 255, 0.06)',
        borderRadius: 12,
        padding: 20,
        boxShadow: '0 1px 3px rgba(0, 0, 0, 0.2)',
        backdropFilter: 'blur(8px)',
      }}
    >
      {/* Dynamic Plain-English Summary Line */}
      <div className={`px-3.5 py-2 rounded-xl border text-xs font-medium flex items-center justify-between gap-2 ${trendSummaryStyle}`}>
        <div className="flex items-center gap-2">
          <TrendSummaryIcon className="w-4 h-4 flex-shrink-0" />
          <span>{trendSummaryText}</span>
        </div>
        <span className="text-[11px] font-mono opacity-80 whitespace-nowrap">
          Historical 24h avg: {last24hAvg.toFixed(1)} µg/m³
        </span>
      </div>

      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
        <div className="space-y-1">
          <div className="flex items-center gap-2 text-cyan-400">
            <TrendingUp className="w-5 h-5" />
            <h2 className="text-lg font-bold text-white">7-Day Air Quality Trends</h2>
          </div>
          <p className="text-xs text-slate-400 flex items-center gap-1.5">
            <Calendar className="w-3.5 h-3.5" />
            Hourly observations & upcoming forecast from Open-Meteo
          </p>
        </div>

        {/* View Toggle & Technical Summary Pills */}
        <div className="flex items-center gap-2">
          <div className="inline-flex rounded-xl bg-slate-950 p-1 border border-slate-800 text-xs">
            {/* Fine Dust PM2.5 button */}
            <button
              onClick={() => setMetric('pm2_5')}
              className={`px-3 py-1.5 rounded-lg transition-all text-left ${
                metric === 'pm2_5'
                  ? 'bg-cyan-500 text-white shadow-sm'
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              <div className="font-bold text-xs flex items-center gap-1.5">
                <span>Fine Dust PM2.5</span>
                <span className={`text-[10px] ${metric === 'pm2_5' ? 'text-cyan-100' : 'text-slate-400'}`}>
                  <Ugm3Unit iconSize="w-2.5 h-2.5" />
                </span>
              </div>
            </button>

            {/* US AQI button */}
            <button
              onClick={() => setMetric('us_aqi')}
              className={`px-3 py-1.5 rounded-lg transition-all text-left ${
                metric === 'us_aqi'
                  ? 'bg-cyan-500 text-white shadow-sm'
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              <div className="font-bold text-xs">US AQI</div>
              <div className={`text-[10px] ${metric === 'us_aqi' ? 'text-cyan-100' : 'text-slate-400'}`}>
                Air Quality Index
              </div>
            </button>
          </div>
        </div>
      </div>

      {/* Mini Stats Bar - Strictly Historical Statistics */}
      <div className="grid grid-cols-3 gap-3 bg-slate-950/60 p-3 rounded-xl border border-slate-800/80 text-center">
        <div>
          <span className="text-[11px] text-slate-400 block font-medium">7-Day Historical Avg</span>
          <span className="text-sm sm:text-base font-bold text-cyan-400 flex items-center justify-center gap-1">
            {stats.avg_pm25} <Ugm3Unit iconSize="w-2.5 h-2.5" />
          </span>
          <span className="text-[10px] text-slate-400 block mt-0.5">Observed only</span>
        </div>
        <div>
          <span className="text-[11px] text-slate-400 block font-medium">Historical Minimum</span>
          <span className="text-sm sm:text-base font-bold text-emerald-400 flex items-center justify-center gap-1">
            {stats.min_pm25} <Ugm3Unit iconSize="w-2.5 h-2.5" />
          </span>
          <span className="text-[10px] text-slate-400 block mt-0.5">Observed low</span>
        </div>
        <div>
          <span className="text-[11px] text-slate-400 block font-medium">Historical Maximum</span>
          <span className="text-sm sm:text-base font-bold text-rose-400 flex items-center justify-center gap-1">
            {stats.max_pm25} <Ugm3Unit iconSize="w-2.5 h-2.5" />
          </span>
          <span className="text-[10px] text-slate-400 block mt-0.5">Observed spike</span>
        </div>
      </div>

      {/* Recharts Composed Chart with Gradient Fill & Round Ticks */}
      <div className="h-[280px] w-full pt-2">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={chartData} margin={{ top: 20, right: 130, left: 10, bottom: 10 }}>
            <defs>
              <linearGradient id="trendGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#22d3ee" stopOpacity={0.1} />
                <stop offset="95%" stopColor="#22d3ee" stopOpacity={0.0} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
            <XAxis
              dataKey="time"
              stroke="#64748b"
              tick={{ fontSize: 11, fill: '#64748b' }}
              interval={dayInterval}
              tickFormatter={formatTrendDate}
              padding={{ left: 10, right: 10 }}
            />
            <YAxis
              stroke="#64748b"
              tick={{ fontSize: 11, fill: '#64748b' }}
              domain={[0, roundedMax]}
              ticks={yTicks}
            />
            <Tooltip content={<CustomTooltip />} />

            {/* Gradient Area Fill under the Observed Line */}
            <Area
              type="monotone"
              dataKey="observed_val"
              fill="url(#trendGradient)"
              stroke="none"
            />

            {/* Safe and Caution Reference Lines - Bold solid 2.5px with two-line labels */}
            {metric === 'pm2_5' && (
              <>
                <ReferenceLine
                  y={15}
                  stroke="#10b981"
                  strokeWidth={2.5}
                  label={twoLineLabel('Safe Limit: 15', 'WHO Guideline', '#10b981')}
                />
                <ReferenceLine
                  y={35}
                  stroke="#f59e0b"
                  strokeWidth={2.5}
                  label={twoLineLabel('Caution Limit: 35', 'Moderate Threshold', '#f59e0b')}
                />
              </>
            )}

            {/* Observation cutoff reference line at "Now" */}
            {lastObsPoint && (
              <ReferenceLine
                x={lastObsPoint.time}
                stroke="#94a3b8"
                strokeWidth={1.5}
                strokeDasharray="2 2"
                label={twoLineLabel('Now', 'Observation Cutoff', '#94a3b8')}
              />
            )}

            {/* Observed historical line (Solid) */}
            <Line
              type="monotone"
              dataKey="observed_val"
              name={metric === 'pm2_5' ? 'Observed PM2.5' : 'Observed AQI'}
              stroke={dataLineColor}
              strokeWidth={2.5}
              dot={false}
              activeDot={{ r: 5, fill: dataLineColor, stroke: '#ffffff', strokeWidth: 2 }}
            />

            {/* Future forecast projection line (Dashed Purple) */}
            <Line
              type="monotone"
              dataKey="forecast_val"
              name={metric === 'pm2_5' ? 'Forecast PM2.5' : 'Forecast AQI'}
              stroke="#c084fc"
              strokeWidth={2}
              strokeDasharray="4 4"
              dot={false}
              activeDot={{ r: 5, fill: '#c084fc', stroke: '#ffffff', strokeWidth: 2 }}
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>

      {/* Legend below the chart distinguishing observed vs forecast */}
      <div style={{ display: 'flex', gap: 20, fontSize: 11, color: '#94a3b8', marginTop: 8, justifyContent: 'center', flexWrap: 'wrap' }}>
        <span><span style={{ color: dataLineColor, fontWeight: 700 }}>—</span> {metric === 'pm2_5' ? 'Observed PM2.5 (Historical)' : 'Observed AQI (Historical)'}</span>
        <span><span style={{ color: '#c084fc', fontWeight: 700 }}>┅</span> Forecast (Upcoming Projection)</span>
        <span><span style={{ color: '#10b981', fontWeight: 600 }}>—</span> Safe Limit: 15 µg/m³</span>
        <span><span style={{ color: '#f59e0b', fontWeight: 600 }}>—</span> Caution Limit: 35 µg/m³</span>
      </div>
    </div>
  );
};
