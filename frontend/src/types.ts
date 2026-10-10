export interface PollutantDetail {
  label: string;
  value: number | null;
  unit: string;
  rating: 'good' | 'moderate' | 'unhealthy_sensitive' | 'unhealthy' | 'very_unhealthy' | 'hazardous' | 'unknown';
  description: string;
}

export interface AQIInfo {
  category: string;
  level: string;
  color: string;
  description: string;
}

export interface AirQualityData {
  latitude: number;
  longitude: number;
  time: string;
  timezone: string;
  us_aqi: number | null;
  european_aqi: number | null;
  us_aqi_capped?: boolean;
  is_capped?: boolean;
  station_name?: string;
  aqi_info: AQIInfo;
  pollutants: {
    pm2_5: PollutantDetail;
    pm10: PollutantDetail;
    nitrogen_dioxide: PollutantDetail;
    sulphur_dioxide: PollutantDetail;
    ozone: PollutantDetail;
    carbon_monoxide: PollutantDetail;
  };
}

export interface TrendPoint {
  time: string;
  utc_time?: string;
  label: string;
  pm2_5: number | null;
  pm10: number | null;
  us_aqi: number | null;
  european_aqi: number | null;
  ozone: number | null;
  nitrogen_dioxide: number | null;
  is_forecast?: boolean;
  point_type?: 'observation' | 'forecast';
}

export interface TrendData {
  latitude: number;
  longitude: number;
  timezone?: string;
  utc_offset_seconds?: number;
  current_time?: string;
  current_time_utc?: string;
  total_points: number;
  stats: {
    avg_pm25: number;
    min_pm25: number;
    max_pm25: number;
    last_24h_avg_pm25?: number;
    historical_points_count?: number;
    forecast_points_count?: number;
  };
  trends: TrendPoint[];
}

export interface ForecastPoint {
  step: number;
  time: string;
  display_time: string;
  display_date: string;
  predicted_pm25: number;
  lower_bound: number;
  upper_bound: number;
  interval_method?: string;
}

export interface ForecastData {
  status: string;
  latitude: number;
  longitude: number;
  training_samples: number;
  days_trained: number;
  current_time?: string;
  current_display_time?: string;
  current_pm25: number;
  forecast_average: number;
  forecast_min: number;
  forecast_max: number;
  forecast: ForecastPoint[];
  interval_method?: string;
  nominal_coverage?: number | null;
  interval_label?: string;
  calibration_samples?: number;
  is_synthetic?: boolean;
  data_source?: string;
  warning?: string;
}

export interface LocationPreset {
  name: string;
  lat: number;
  lon: number;
  country: string;
}
