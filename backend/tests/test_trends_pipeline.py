import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from backend.main import app, SHARED_AIR_CACHE


@pytest.fixture(autouse=True)
def clear_shared_cache():
    SHARED_AIR_CACHE.clear()
    yield
    SHARED_AIR_CACHE.clear()


def make_trends_fixture(
    current_time_str="2026-10-09T12:00",
    utc_offset_seconds=0,
    past_hours=24,
    future_hours=12,
    past_pm25=40.0,
    future_pm25=180.0
):
    current_dt = datetime.fromisoformat(current_time_str)
    
    # Historical hours ending at current_time_str
    hist_times = [
        (current_dt - timedelta(hours=past_hours - i)).strftime("%Y-%m-%dT%H:00")
        for i in range(past_hours + 1)  # includes current_dt
    ]
    
    # Future hours starting immediately after current_dt
    future_times = [
        (current_dt + timedelta(hours=i + 1)).strftime("%Y-%m-%dT%H:00")
        for i in range(future_hours)
    ]
    
    all_times = hist_times + future_times
    all_pm25 = [past_pm25] * len(hist_times) + [future_pm25] * len(future_times)
    n = len(all_times)

    return {
        "latitude": 40.71,
        "longitude": -74.00,
        "timezone": "America/New_York",
        "utc_offset_seconds": utc_offset_seconds,
        "current": {
            "time": current_time_str,
            "pm2_5": past_pm25,
            "us_aqi": 80
        },
        "hourly": {
            "time": all_times,
            "pm2_5": all_pm25,
            "pm10": [50.0] * n,
            "us_aqi": [80] * len(hist_times) + [220] * len(future_times),
            "european_aqi": [40] * n,
            "ozone": [30.0] * n,
            "nitrogen_dioxide": [15.0] * n,
        }
    }


def test_trends_separates_history_and_forecast_no_future_in_averages():
    """
    Verify that future forecast points (future_pm25=180.0) do NOT pollute
    the historical stats (past_pm25=40.0).
    """
    client = TestClient(app)
    fixture = make_trends_fixture(
        current_time_str="2026-10-09T12:00",
        past_hours=24,
        future_hours=12,
        past_pm25=40.0,
        future_pm25=180.0
    )

    with patch("backend.main.get_shared_open_meteo_data", return_value=fixture):
        response = client.get("/api/trends/40.71/-74.00")
        assert response.status_code == 200
        data = response.json()

        stats = data["stats"]
        trends = data["trends"]

        # Historical stats MUST NOT include the future 180.0 points
        assert stats["avg_pm25"] == 40.0
        assert stats["max_pm25"] == 40.0
        assert stats["min_pm25"] == 40.0
        assert stats["last_24h_avg_pm25"] == 40.0
        assert stats["forecast_points_count"] == 12
        assert stats["historical_points_count"] == 25

        # Check point classifications
        for pt in trends:
            t_dt = datetime.fromisoformat(pt["time"])
            curr_dt = datetime.fromisoformat(data["current_time"])
            if t_dt <= curr_dt:
                assert pt["is_forecast"] is False
                assert pt["point_type"] == "observation"
                assert pt["pm2_5"] == 40.0
            else:
                assert pt["is_forecast"] is True
                assert pt["point_type"] == "forecast"
                assert pt["pm2_5"] == 180.0


def test_last_24h_historical_window_accuracy():
    """
    Verify that last_24h_avg_pm25 is computed strictly from the 24 hours
    ending at current_time, excluding older historical hours and all future hours.
    """
    client = TestClient(app)
    current_dt = datetime(2026, 10, 9, 12, 0)
    
    # 48 historical hours:
    # Hours -48 to -25 (older): PM2.5 = 100.0
    # Hours -24 to 0 (last 24h): PM2.5 = 20.0
    # Hours +1 to +12 (future): PM2.5 = 500.0
    hist_older = [(current_dt - timedelta(hours=48 - i)).strftime("%Y-%m-%dT%H:00") for i in range(24)]
    hist_recent = [(current_dt - timedelta(hours=24 - i)).strftime("%Y-%m-%dT%H:00") for i in range(25)]
    future_times = [(current_dt + timedelta(hours=i + 1)).strftime("%Y-%m-%dT%H:00") for i in range(12)]

    all_times = hist_older + hist_recent + future_times
    all_pm25 = [100.0] * len(hist_older) + [20.0] * len(hist_recent) + [500.0] * len(future_times)

    fixture = {
        "latitude": 51.5,
        "longitude": -0.1,
        "timezone": "Europe/London",
        "utc_offset_seconds": 0,
        "current": {"time": current_dt.strftime("%Y-%m-%dT%H:00")},
        "hourly": {
            "time": all_times,
            "pm2_5": all_pm25,
            "pm10": [30.0] * len(all_times),
            "us_aqi": [50] * len(all_times),
            "european_aqi": [20] * len(all_times),
            "ozone": [25.0] * len(all_times),
            "nitrogen_dioxide": [10.0] * len(all_times),
        }
    }

    with patch("backend.main.get_shared_open_meteo_data", return_value=fixture):
        response = client.get("/api/trends/51.5/-0.1")
        assert response.status_code == 200
        data = response.json()
        stats = data["stats"]

        # The last 24h average should strictly be 20.0! (Not contaminated by older 100.0 or future 500.0)
        assert stats["last_24h_avg_pm25"] == 20.0
        # The overall 48h historical average should be around (24*100 + 25*20) / 49 = 59.2
        assert stats["avg_pm25"] < 100.0
        assert stats["avg_pm25"] > 20.0
        assert stats["max_pm25"] == 100.0  # Max in history is 100, NEVER the future 500!


def test_deduplication_and_out_of_order_records():
    """
    Verify that out-of-order timestamps and duplicates are properly sorted
    and deduplicated chronologically.
    """
    client = TestClient(app)
    fixture = {
        "latitude": 28.61,
        "longitude": 77.23,
        "timezone": "Asia/Kolkata",
        "utc_offset_seconds": 19800,
        "current": {"time": "2026-10-09T12:00"},
        "hourly": {
            "time": [
                "2026-10-09T12:00",
                "2026-10-09T08:00",
                "2026-10-09T10:00",
                "2026-10-09T10:00",  # Duplicate timestamp
                "2026-10-09T14:00",  # Future
            ],
            "pm2_5": [25.0, 15.0, None, 18.0, 30.0],  # Second copy of 10:00 has valid 18.0
            "pm10": [40.0, 30.0, 35.0, 35.0, 50.0],
            "us_aqi": [50, 40, 45, 45, 60],
            "european_aqi": [20, 15, 18, 18, 25],
            "ozone": [20.0, 20.0, 20.0, 20.0, 20.0],
            "nitrogen_dioxide": [10.0, 10.0, 10.0, 10.0, 10.0],
        }
    }

    with patch("backend.main.get_shared_open_meteo_data", return_value=fixture):
        response = client.get("/api/trends/28.61/77.23")
        assert response.status_code == 200
        data = response.json()
        trends = data["trends"]

        # Deduplicated from 5 to 4 points
        assert len(trends) == 4

        # Strictly chronological order
        times = [t["time"] for t in trends]
        assert times == [
            "2026-10-09T08:00",
            "2026-10-09T10:00",
            "2026-10-09T12:00",
            "2026-10-09T14:00"
        ]

        # Duplicate merged with valid reading 18.0
        pt_10 = next(t for t in trends if t["time"] == "2026-10-09T10:00")
        assert pt_10["pm2_5"] == 18.0
        assert pt_10["is_forecast"] is False

        # Future point
        pt_14 = next(t for t in trends if t["time"] == "2026-10-09T14:00")
        assert pt_14["is_forecast"] is True


def test_invalid_and_negative_values_handling():
    """
    Verify that negative or NaN pollutant values are safely cleaned and do not corrupt statistics.
    """
    client = TestClient(app)
    fixture = {
        "latitude": 28.61,
        "longitude": 77.23,
        "timezone": "UTC",
        "utc_offset_seconds": 0,
        "current": {"time": "2026-10-09T03:00"},
        "hourly": {
            "time": ["2026-10-09T01:00", "2026-10-09T02:00", "2026-10-09T03:00"],
            "pm2_5": [-999.0, 20.0, 30.0],  # -999 is invalid
            "pm10": [25.0, 35.0, 45.0],
            "us_aqi": [40, 50, 60],
            "european_aqi": [10, 15, 20],
            "ozone": [10.0, 15.0, 20.0],
            "nitrogen_dioxide": [5.0, 10.0, 15.0],
        }
    }

    with patch("backend.main.get_shared_open_meteo_data", return_value=fixture):
        response = client.get("/api/trends/28.61/77.23")
        assert response.status_code == 200
        data = response.json()
        stats = data["stats"]
        trends = data["trends"]

        # -999 cleaned to None
        assert trends[0]["pm2_5"] is None
        # Average strictly computed on valid observations: (20 + 30) / 2 = 25.0
        assert stats["avg_pm25"] == 25.0
        assert stats["min_pm25"] == 20.0
        assert stats["max_pm25"] == 30.0


def test_calculate_highest_aqi_of_day():
    """
    Verify calculate_highest_aqi_of_day only selects recorded values for today
    up to current observation time and ignores future forecasts.
    """
    from backend.main import calculate_highest_aqi_of_day

    test_data = {
        "current": {"time": "2026-10-14T10:00", "us_aqi": 120},
        "hourly": {
            "time": [
                "2026-10-13T22:00",  # Yesterday
                "2026-10-14T02:00",  # Today recorded
                "2026-10-14T06:00",  # Today recorded (PEAK)
                "2026-10-14T10:00",  # Today current observation
                "2026-10-14T14:00",  # Today FUTURE forecast (should be ignored)
            ],
            "us_aqi": [250, 110, 185, 120, 300],
            "pm2_5": [120.0, 40.0, 85.0, 45.0, 150.0]
        }
    }

    peak_aqi, peak_time = calculate_highest_aqi_of_day(test_data)
    assert peak_aqi == 185
    assert peak_time == "2026-10-14T06:00"


def test_calculate_highest_aqi_capped_at_500():
    """
    Verify that extreme values exceeding EPA max are clamped to 500.
    """
    from backend.main import calculate_highest_aqi_of_day

    test_data = {
        "current": {"time": "2026-10-14T04:00", "us_aqi": 450},
        "hourly": {
            "time": ["2026-10-14T01:00", "2026-10-14T04:00"],
            "us_aqi": [850, 450],
            "pm2_5": [600.0, 350.0]
        }
    }

    peak_aqi, peak_time = calculate_highest_aqi_of_day(test_data)
    assert peak_aqi == 500
    assert peak_time == "2026-10-14T01:00"


def test_air_quality_and_trends_return_highest_aqi_today():
    """
    Verify both /api/air-quality and /api/trends include highest_aqi_today.
    """
    client = TestClient(app)
    fixture = {
        "latitude": 28.61,
        "longitude": 77.20,
        "timezone": "Asia/Kolkata",
        "utc_offset_seconds": 19800,
        "current": {
            "time": "2026-10-14T08:00",
            "us_aqi": 140,
            "european_aqi": 60,
            "pm2_5": 52.0,
            "pm10": 95.0,
            "nitrogen_dioxide": 25.0,
            "sulphur_dioxide": 12.0,
            "ozone": 30.0,
            "carbon_monoxide": 500.0
        },
        "hourly": {
            "time": [
                "2026-10-13T23:00",
                "2026-10-14T02:00",
                "2026-10-14T05:00",
                "2026-10-14T08:00",
                "2026-10-14T12:00",
            ],
            "pm2_5": [40.0, 60.0, 95.0, 52.0, 110.0],
            "pm10": [70.0, 100.0, 150.0, 95.0, 170.0],
            "us_aqi": [112, 153, 172, 140, 179],
            "european_aqi": [40, 60, 80, 60, 85],
            "ozone": [20.0, 25.0, 30.0, 30.0, 45.0],
            "nitrogen_dioxide": [15.0, 20.0, 35.0, 25.0, 40.0],
        }
    }

    with patch("backend.main.get_shared_open_meteo_data", return_value=fixture):
        # 1. Check /api/air-quality
        aq_resp = client.get("/api/air-quality/28.61/77.20")
        assert aq_resp.status_code == 200
        aq_data = aq_resp.json()
        assert aq_data["highest_aqi_today"] == 172
        assert aq_data["highest_aqi_time"] == "2026-10-14T05:00"
        assert aq_data["highest_aqi_info"]["category"] == "Unhealthy"

        # 2. Check /api/trends
        trends_resp = client.get("/api/trends/28.61/77.20")
        assert trends_resp.status_code == 200
        tr_data = trends_resp.json()
        assert tr_data["stats"]["highest_aqi_today"] == 172
        assert tr_data["stats"]["highest_aqi_today_time"] == "2026-10-14T05:00"

