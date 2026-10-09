import { describe, it, expect } from 'vitest';
import { computeHistorical24hAvg, prepareTrendChartData } from '../TrendChart';
import { TrendPoint } from '../../types';

describe('TrendChart historical trend pipeline', () => {
  const makeSamplePoint = (
    time: string,
    pm2_5: number | null,
    is_forecast: boolean
  ): TrendPoint => ({
    time,
    label: time.replace('T', ' '),
    pm2_5,
    pm10: 40.0,
    us_aqi: 70,
    european_aqi: 30,
    ozone: 25.0,
    nitrogen_dioxide: 15.0,
    is_forecast,
    point_type: is_forecast ? 'forecast' : 'observation',
  });

  describe('computeHistorical24hAvg', () => {
    it('calculates the 24h average strictly from historical observations, ignoring future forecasts', () => {
      // 24 historical points with PM2.5 = 20.0
      const historicalPoints: TrendPoint[] = Array.from({ length: 24 }, (_, i) =>
        makeSamplePoint(`2026-10-09T${String(i).padStart(2, '0')}:00`, 20.0, false)
      );

      // 12 future forecast points with extreme PM2.5 = 450.0
      const futurePoints: TrendPoint[] = Array.from({ length: 12 }, (_, i) =>
        makeSamplePoint(`2026-10-10T${String(i).padStart(2, '0')}:00`, 450.0, true)
      );

      const allPoints = [...historicalPoints, ...futurePoints];

      // If sliced simply with .slice(-24), it would take 12 historical and 12 future points = ~235.0!
      // But computeHistorical24hAvg MUST ignore future forecast points and evaluate strictly to 20.0!
      const avg = computeHistorical24hAvg(allPoints, undefined, 50.0);
      expect(avg).toBe(20.0);
    });

    it('uses the backend-provided true historical 24h window average when available', () => {
      const points = [
        makeSamplePoint('2026-10-09T10:00', 30.0, false),
        makeSamplePoint('2026-10-09T11:00', 40.0, false),
      ];

      const avg = computeHistorical24hAvg(points, 35.0, 50.0);
      expect(avg).toBe(35.0);
    });

    it('explicitly handles missing or null PM2.5 readings in historical data without crashing', () => {
      const points = [
        makeSamplePoint('2026-10-09T08:00', null, false),
        makeSamplePoint('2026-10-09T09:00', 10.0, false),
        makeSamplePoint('2026-10-09T10:00', 30.0, false),
        makeSamplePoint('2026-10-09T11:00', 100.0, true), // future point
      ];

      const avg = computeHistorical24hAvg(points, undefined, 0);
      // Valid historical points: 10.0 and 30.0. Average: (10 + 30) / 2 = 20.0
      expect(avg).toBe(20.0);
    });
  });

  describe('prepareTrendChartData', () => {
    it('distinguishes observed and forecast lines visually and semantically', () => {
      const points = [
        makeSamplePoint('2026-10-09T08:00', 15.0, false),
        makeSamplePoint('2026-10-09T09:00', 25.0, false),
        makeSamplePoint('2026-10-09T10:00', 35.0, true),
        makeSamplePoint('2026-10-09T11:00', 45.0, true),
      ];

      const chartData = prepareTrendChartData(points, 'pm2_5');

      // Point 0 (historical): observed_val = 15.0, forecast_val = null
      expect(chartData[0].observed_val).toBe(15.0);
      expect(chartData[0].forecast_val).toBeNull();
      expect(chartData[0].is_forecast).toBe(false);

      // Point 1 (boundary historical): observed_val = 25.0, forecast_val = 25.0 (for seamless line continuity)
      expect(chartData[1].observed_val).toBe(25.0);
      expect(chartData[1].forecast_val).toBe(25.0);
      expect(chartData[1].is_forecast).toBe(false);

      // Point 2 (future forecast): observed_val = null, forecast_val = 35.0
      expect(chartData[2].observed_val).toBeNull();
      expect(chartData[2].forecast_val).toBe(35.0);
      expect(chartData[2].is_forecast).toBe(true);

      // Point 3 (future forecast): observed_val = null, forecast_val = 45.0
      expect(chartData[3].observed_val).toBeNull();
      expect(chartData[3].forecast_val).toBe(45.0);
      expect(chartData[3].is_forecast).toBe(true);
    });

    it('handles AQI metric toggle correctly', () => {
      const points = [
        makeSamplePoint('2026-10-09T08:00', 15.0, false),
        makeSamplePoint('2026-10-09T09:00', 25.0, true),
      ];

      const chartData = prepareTrendChartData(points, 'us_aqi');
      expect(chartData[0].observed_val).toBe(70);
      expect(chartData[1].forecast_val).toBe(70);
    });
  });
});
