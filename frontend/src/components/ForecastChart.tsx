import React from 'react';
import {
  ResponsiveContainer,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  Area,
  ComposedChart,
  ReferenceLine,
} from 'recharts';
import { ForecastData } from '../types';
import { Sparkles, Brain, Cpu, Info, CheckCircle2, AlertTriangle } from 'lucide-react';

interface ForecastChartProps {
  data?: ForecastData | null;
  isLoading?: boolean;
  error?: string | null;
  onEnableDemo?: () => void;
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

export const ForecastChart: React.FC<ForecastChartProps> = ({ data, isLoading, error, onEnableDemo }) => {
  if (isLoading) {
    return (
      <div
        className="h-[470px] flex flex-col items-center justify-center text-center space-y-4"
        style={{
          background: 'rgba(255, 255, 255, 0.02)',
          border: '1px solid rgba(255, 255, 255, 0.06)',
          borderRadius: 12,
          padding: 20,
          boxShadow: '0 1px 3px rgba(0, 0, 0, 0.2)',
          backdropFilter: 'blur(8px)',
        }}
      >
        <div className="relative">
          <div className="w-16 h-16 rounded-2xl bg-purple-500/20 border border-purple-500/40 flex items-center justify-center animate-pulse">
            <Brain className="w-8 h-8 text-purple-400 animate-pulse" />
          </div>
          <div className="absolute -top-1 -right-1">
            <Sparkles className="w-5 h-5 text-amber-400 animate-spin" />
          </div>
        </div>
        <div className="space-y-1.5 max-w-sm">
          <h3 className="text-base font-bold text-white flex items-center justify-center gap-2">
            <span>Training ML model...</span>
            <span className="inline-block w-2 h-2 rounded-full bg-purple-400 animate-ping"></span>
          </h3>
          <p className="text-xs text-slate-400 leading-relaxed">
            Fetching 92 days of hourly data from Open-Meteo and training the XGBoost regressor for a 24-hour PM2.5 forecast.
          </p>
        </div>
        <div className="flex items-center gap-2 text-[11px] text-purple-300/80 bg-purple-950/50 border border-purple-800/60 px-3 py-1.5 rounded-full">
          <Cpu className="w-3.5 h-3.5 text-purple-400" />
          <span>XGBoost Autoregressive Model • 92 Days Training Set</span>
        </div>
      </div>
    );
  }

  if (!data) {
    return (
      <div
        className="h-[470px] flex flex-col items-center justify-center text-center space-y-4 p-6"
        style={{
          background: 'rgba(255, 255, 255, 0.02)',
          border: '1px solid rgba(255, 255, 255, 0.06)',
          borderRadius: 12,
          padding: 24,
          boxShadow: '0 1px 3px rgba(0, 0, 0, 0.2)',
          backdropFilter: 'blur(8px)',
        }}
      >
        <div className="w-14 h-14 rounded-2xl bg-amber-500/10 border border-amber-500/30 flex items-center justify-center">
          <AlertTriangle className="w-7 h-7 text-amber-400" />
        </div>
        <div className="space-y-2 max-w-md">
          <h3 className="text-base font-bold text-white">
            Forecast Unavailable
          </h3>
          <p className="text-xs text-slate-300 leading-relaxed">
            {error || "Historical air quality observations are unavailable or insufficient for this location."}
          </p>
          <p className="text-[11px] text-slate-400 leading-relaxed">
            AeroTrack never silently fabricates PM2.5 readings. Genuine forecasting requires at least 50 valid historical observations from Open-Meteo.
          </p>
        </div>
        {onEnableDemo && (
          <button
            onClick={onEnableDemo}
            className="mt-2 inline-flex items-center gap-2 px-4 py-2 rounded-xl text-xs font-semibold bg-purple-600/30 hover:bg-purple-600/50 text-purple-200 border border-purple-500/40 transition-colors shadow-sm"
          >
            <Brain className="w-4 h-4 text-purple-300" />
            <span>Load Opt-In Demo Forecast (Simulated)</span>
          </button>
        )}
      </div>
    );
  }

  const { forecast, forecast_average, forecast_min, forecast_max, current_pm25, days_trained, training_samples } = data;

  // Dynamic Y-axis scaling with clean round numbers including predicted values, upper bounds, and WHO target (15)
  const allYValues: number[] = [15];
  const lowerBounds: number[] = [];
  const upperBounds: number[] = [];

  forecast.forEach((pt) => {
    if (typeof pt.predicted_pm25 === 'number' && !isNaN(pt.predicted_pm25)) {
      allYValues.push(pt.predicted_pm25);
    }
    if (typeof pt.lower_bound === 'number' && !isNaN(pt.lower_bound)) {
      allYValues.push(pt.lower_bound);
      lowerBounds.push(pt.lower_bound);
    }
    if (typeof pt.upper_bound === 'number' && !isNaN(pt.upper_bound)) {
      allYValues.push(pt.upper_bound);
      upperBounds.push(pt.upper_bound);
    }
  });

  const p10Min = lowerBounds.length ? Math.min(...lowerBounds) : forecast_min;
  const p90Max = upperBounds.length ? Math.max(...upperBounds) : forecast_max;

  // Add vertical breathing room: round up to dataMax + 15
  const dataMax = Math.max(...allYValues);
  const roundedMax = Math.max(40, Math.ceil(dataMax + 15));

  // Determine clean tick intervals based on roundedMax
  let step = 20;
  if (roundedMax <= 50) {
    step = 10;
  } else if (roundedMax <= 100) {
    step = 20;
  } else if (roundedMax <= 160) {
    step = 25;
  } else {
    step = 50;
  }

  const yTicks: number[] = [];
  for (let val = 0; val <= roundedMax; val += step) {
    yTicks.push(val);
  }

  // Dynamic Trajectory Evaluation
  const predValues = forecast
    .map((f) => f.predicted_pm25)
    .filter((v): v is number => typeof v === 'number' && !isNaN(v));
  const minPred = predValues.length ? Math.min(...predValues) : current_pm25;
  const maxPred = predValues.length ? Math.max(...predValues) : current_pm25;
  const predRange = maxPred - minPred;
  const firstPred = predValues[0] ?? current_pm25;
  const lastPred = predValues[predValues.length - 1] ?? current_pm25;

  let trajectoryText = 'Relatively Stable';
  let trajectoryColor = 'text-cyan-400';
  if (predRange < 3) {
    trajectoryText = 'Relatively Stable';
    trajectoryColor = 'text-cyan-400';
  } else if (lastPred > firstPred) {
    trajectoryText = 'Rising Trajectory';
    trajectoryColor = 'text-rose-400';
  } else if (lastPred < firstPred) {
    trajectoryText = 'Falling Trajectory';
    trajectoryColor = 'text-emerald-400';
  } else {
    trajectoryText = 'Variable Trajectory';
    trajectoryColor = 'text-amber-400';
  }

  // Plain-English Summary Line above the Forecast Chart
  let forecastSummaryText = "Tomorrow's air is expected to be safe.";
  let forecastSummaryStyle = 'bg-emerald-500/10 border-emerald-500/30 text-emerald-300';
  let ForecastSummaryIcon = CheckCircle2;

  // Prepare chart data starting with "Now" (step 0, measured current PM2.5) followed by predictions
  const nowPoint = {
    step: 0,
    time: data.current_time || new Date().toISOString(),
    display_time: data.current_display_time || 'Now',
    display_date: 'Today',
    predicted_pm25: current_pm25,
    lower_bound: current_pm25,
    upper_bound: current_pm25,
    confidence_range: [current_pm25, current_pm25],
    is_now: true,
  };

  const chartData = [
    nowPoint,
    ...forecast.map((pt) => ({
      ...pt,
      confidence_range: [pt.lower_bound, pt.upper_bound],
      is_now: false,
    }))
  ];

  if (forecast_average > 55) {
    forecastSummaryText = "Tomorrow's air is expected to be unhealthy — limit outdoor activity.";
    forecastSummaryStyle = 'bg-rose-500/10 border-rose-500/30 text-rose-300';
    ForecastSummaryIcon = AlertTriangle;
  } else if (forecast_average > 35) {
    forecastSummaryText = "Tomorrow's air may be unhealthy for sensitive groups.";
    forecastSummaryStyle = 'bg-amber-500/10 border-amber-500/30 text-amber-300';
    ForecastSummaryIcon = AlertTriangle;
  }

  const nowTime = nowPoint.display_time;

  const CustomTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length) {
      const point = payload[0].payload;
      return (
        <div className="bg-slate-900/95 border border-cyan-500/40 p-3 rounded-xl shadow-2xl backdrop-blur-md text-xs space-y-1">
          <p className="text-slate-400 text-[11px] font-medium">
            {point.display_date} • {point.display_time}
          </p>
          {/* Line 1 (bold) */}
          <div className="text-sm font-bold text-cyan-300 flex items-center gap-1">
            <span>Expected: {point.predicted_pm25}</span>
            <Ugm3Unit iconSize="w-2.5 h-2.5" />
          </div>
          {/* Line 2 */}
          <div className="text-xs text-slate-200 flex items-center gap-1">
            <span>Estimate Range: {point.lower_bound} to {point.upper_bound}</span>
            <Ugm3Unit iconSize="w-2.5 h-2.5" />
          </div>
        </div>
      );
    }
    return null;
  };

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
      {/* Explicit Provenance Banner for Synthetic Demo Data */}
      {data.is_synthetic && (
        <div className="px-3.5 py-2.5 rounded-xl border border-amber-500/40 bg-amber-500/10 text-amber-200 text-xs flex items-start gap-2.5">
          <AlertTriangle className="w-4 h-4 text-amber-400 flex-shrink-0 mt-0.5" />
          <div className="space-y-0.5 text-left">
            <span className="font-bold tracking-wide uppercase text-[11px] text-amber-300">
              DEMO / SYNTHETIC DATA
            </span>
            <p className="text-[11px] text-amber-200/90 leading-relaxed">
              {data.warning || "Upstream historical observations are unavailable for these coordinates. This forecast was generated using simulated diurnal PM2.5 data for demonstration purposes."}
            </p>
          </div>
        </div>
      )}

      {/* Dynamic Plain-English Summary Line */}
      <div className={`px-3.5 py-2 rounded-xl border text-xs font-medium flex items-center justify-between gap-2 ${forecastSummaryStyle}`}>
        <div className="flex items-center gap-2">
          <ForecastSummaryIcon className="w-4 h-4 flex-shrink-0" />
          <span>{forecastSummaryText}</span>
        </div>
        <span className="text-[11px] font-mono opacity-80 whitespace-nowrap">
          24h avg: {forecast_average.toFixed(1)} µg/m³
        </span>
      </div>

      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
        <div className="space-y-1">
          <div className="flex items-center gap-2">
            <div className="p-1.5 rounded-lg bg-purple-500/20 text-purple-400 border border-purple-500/30">
              <Sparkles className="w-4 h-4" />
            </div>
            <h2 className="text-lg font-bold text-white">24-Hour ML PM2.5 Forecast</h2>
            {data.is_synthetic ? (
              <span className="inline-flex items-center gap-1 text-[11px] font-bold px-2.5 py-0.5 rounded-full bg-amber-500/20 text-amber-300 border border-amber-500/40 uppercase tracking-wider">
                <AlertTriangle className="w-3 h-3" />
                Demo / Synthetic
              </span>
            ) : (
              <span className="hidden sm:inline-flex items-center gap-1 text-[11px] font-semibold px-2.5 py-0.5 rounded-full bg-purple-500/10 text-purple-300 border border-purple-500/30">
                <Brain className="w-3 h-3" />
                XGBoost Model
              </span>
            )}
          </div>

          {/* Sub-header explaining the curve and band */}
          <p className="text-xs text-slate-400">
            The solid line is the most likely value. The shaded band shows the range of possible outcomes.
          </p>

          <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-slate-400 pt-0.5">
            <span>
              Trajectory: <span className={`font-semibold ${trajectoryColor}`}>{trajectoryText}</span> (Δ {(lastPred - firstPred).toFixed(1)} µg/m³)
            </span>
            <span className="text-slate-400">•</span>
            <span>
              {data.is_synthetic
                ? "Simulated Demo Observations"
                : `Trained on ${days_trained} days (${training_samples} hourly observations)`}
            </span>
          </div>
        </div>
      </div>

      {/* Forecast Metric Cards matching Confidence Band */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 bg-slate-950/60 p-3 rounded-xl border border-slate-800/80 text-center">
        <div>
          <span className="text-[11px] text-slate-400 block font-medium">Current PM2.5</span>
          <span className="text-sm sm:text-base font-bold text-slate-200 flex items-center justify-center gap-1">
            {current_pm25} <Ugm3Unit iconSize="w-2.5 h-2.5" />
          </span>
          <span className="text-[10px] text-slate-400 block mt-0.5">Measured</span>
        </div>
        <div>
          <span className="text-[11px] text-slate-400 block font-medium">24h Forecast Avg</span>
          <span className="text-sm sm:text-base font-bold text-purple-400 flex items-center justify-center gap-1">
            {forecast_average} <Ugm3Unit iconSize="w-2.5 h-2.5" />
          </span>
          <span className="text-[10px] text-slate-400 block mt-0.5">Expected mean</span>
        </div>
        <div>
          <span className="text-[11px] text-slate-400 block font-medium">Lower Estimate</span>
          <span className="text-sm sm:text-base font-bold text-emerald-400 flex items-center justify-center gap-1">
            {p10Min} <Ugm3Unit iconSize="w-2.5 h-2.5" />
          </span>
          <span className="text-[10px] text-slate-400 block mt-0.5">Heuristic range</span>
        </div>
        <div>
          <span className="text-[11px] text-slate-400 block font-medium">Upper Estimate</span>
          <span className="text-sm sm:text-base font-bold text-rose-400 flex items-center justify-center gap-1">
            {p90Max} <Ugm3Unit iconSize="w-2.5 h-2.5" />
          </span>
          <span className="text-[10px] text-slate-400 block mt-0.5">Heuristic range</span>
        </div>
      </div>

      {/* Recharts Composed Forecast Chart with Confidence Band */}
      <div className="h-[280px] w-full pt-2">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={chartData} margin={{ top: 20, right: 130, left: 10, bottom: 10 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
            <XAxis
              dataKey="display_time"
              stroke="#64748b"
              tick={{ fontSize: 11, fill: '#64748b' }}
              interval={3}
              padding={{ left: 10, right: 10 }}
            />
            <YAxis
              stroke="#64748b"
              tick={{ fontSize: 11, fill: '#64748b' }}
              domain={[0, roundedMax]}
              ticks={yTicks}
            />
            <Tooltip content={<CustomTooltip />} />

            {/* Safe Limit Reference Line (y=15): Bold, Solid green line (strokeWidth 2.5) */}
            <ReferenceLine
              y={15}
              stroke="#10b981"
              strokeWidth={2.5}
              label={twoLineLabel('Safe Limit: 15', 'WHO Guideline', '#10b981')}
            />

            {/* Caution Limit Reference Line (y=35): Bold, Solid orange line (strokeWidth 2.5) */}
            <ReferenceLine
              y={35}
              stroke="#f59e0b"
              strokeWidth={2.5}
              label={twoLineLabel('Caution Limit: 35', 'Moderate Threshold', '#f59e0b')}
            />

            {/* Vertical "Now" Reference Line: Subtle light gray dashed line with opacity 0.5 */}
            {nowTime && (
              <ReferenceLine
                x={nowTime}
                stroke="#94a3b8"
                strokeWidth={1}
                strokeOpacity={0.5}
                strokeDasharray="3 3"
                label={{ value: 'Now', fill: '#94a3b8', fontSize: 11, position: 'insideTopLeft' }}
              />
            )}

            {/* Confidence Band: ONLY shaded between lower_bound and upper_bound (0.15 opacity soft purple) */}
            <Area
              type="monotone"
              dataKey="confidence_range"
              fill="#a78bfa"
              fillOpacity={0.15}
              stroke="none"
            />

            {/* Subtle Dashed Bound Edges for Confidence Range */}
            <Line
              type="monotone"
              dataKey="upper_bound"
              stroke="#a78bfa"
              strokeWidth={1}
              strokeDasharray="3 3"
              dot={false}
              activeDot={false}
            />
            <Line
              type="monotone"
              dataKey="lower_bound"
              stroke="#a78bfa"
              strokeWidth={1}
              strokeDasharray="3 3"
              dot={false}
              activeDot={false}
            />

            {/* Forecast Line Glow Effect: Background Line underneath with strokeWidth=6 and opacity=0.2 */}
            <Line
              type="monotone"
              dataKey="predicted_pm25"
              stroke="#22d3ee"
              strokeWidth={6}
              strokeOpacity={0.25}
              dot={false}
              activeDot={false}
            />

            {/* Forecast Line (The Star): Solid, high-contrast cyan with white-ring dots */}
            <Line
              type="monotone"
              dataKey="predicted_pm25"
              stroke="#22d3ee"
              strokeWidth={3}
              dot={{ r: 4, fill: '#22d3ee', stroke: '#ffffff', strokeWidth: 1.5 }}
              activeDot={{ r: 6, fill: '#22d3ee', stroke: '#ffffff', strokeWidth: 2 }}
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>

      {/* Legend below the chart consistent with Trend Chart */}
      <div style={{ display: 'flex', gap: 24, fontSize: 11, color: '#94a3b8', marginTop: 8, justifyContent: 'center', flexWrap: 'wrap' }}>
        <span><span style={{ color: '#10b981', fontWeight: 600 }}>—</span> Safe Limit: 15 µg/m³ (WHO Guideline)</span>
        <span><span style={{ color: '#f59e0b', fontWeight: 600 }}>—</span> Caution Limit: 35 µg/m³ (Moderate Threshold)</span>
        <span><span style={{ color: '#22d3ee', fontWeight: 600 }}>━━</span> Predicted PM2.5 (ML Forecast)</span>
        <span><span style={{ color: '#a78bfa', fontWeight: 600 }}>░</span> Estimate Range</span>
      </div>

      <div className="text-[11px] text-slate-400 text-center pt-1">
        Predictions generated autoregressively using XGBoost with engineered lag and cyclical features.
      </div>
    </div>
  );
};
