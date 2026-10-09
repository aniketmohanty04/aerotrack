import pytest
import asyncio
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
import xgboost as xgb

from backend.ml_model import (
    predict_autoregressive_rollout,
    compute_prediction_interval,
    calibrate_conformal_quantiles,
    get_feature_columns,
    train_and_forecast_pm25,
    normalize_coordinates,
    get_coordinate_lock,
    MODEL_CACHE,
    CALIBRATION_CACHE,
    FORECAST_LOCKS
)


@pytest.fixture(autouse=True)
def clear_cache():
    MODEL_CACHE.clear()
    CALIBRATION_CACHE.clear()
    FORECAST_LOCKS.clear()
    yield
    MODEL_CACHE.clear()
    CALIBRATION_CACHE.clear()
    FORECAST_LOCKS.clear()


def make_synthetic_history(n_hours=150):
    start = pd.Timestamp("2026-01-01 00:00:00")
    times = [start + pd.Timedelta(hours=i) for i in range(n_hours)]
    h = np.array([t.hour for t in times])
    pm25 = 25.0 + 10.0 * np.sin(2 * np.pi * h / 24.0)

    return pd.DataFrame({
        "time": times,
        "pm25": pm25,
        "pm2_5": pm25,
        "wind_speed": 3.0,
        "temperature": 20.0,
        "humidity": 55.0,
        "no2": 15.0,
        "o3": 30.0
    })


def test_interval_ordering_heuristic():
    """Verify 0.0 <= lower_bound <= predicted <= upper_bound under heuristic bounds."""
    for step in range(1, 25):
        pred_val = 42.5
        rolling_std = 5.2
        lower, upper, method = compute_prediction_interval(
            pred_val=pred_val,
            step=step,
            conformal_quantiles=None,
            rolling_std_24=rolling_std
        )
        assert method == "estimated_uncertainty_heuristic"
        assert 0.0 <= lower <= pred_val <= upper
        assert upper > lower


def test_interval_ordering_conformal():
    """Verify 0.0 <= lower_bound <= predicted <= upper_bound under conformal quantiles."""
    quantiles = {step: 5.0 + (step * 0.5) for step in range(1, 25)}
    for step in range(1, 25):
        pred_val = 30.0
        lower, upper, method = compute_prediction_interval(
            pred_val=pred_val,
            step=step,
            conformal_quantiles=quantiles
        )
        assert method == "split_conformal_prediction"
        assert 0.0 <= lower <= pred_val <= upper
        assert upper > lower


def test_non_negative_bounds_clamping():
    """Verify lower bound is never negative even when margin exceeds prediction."""
    pred_val = 2.0
    # Large margin would yield negative lower bound if not clamped
    quantiles = {1: 15.0}
    lower, upper, method = compute_prediction_interval(
        pred_val=pred_val,
        step=1,
        conformal_quantiles=quantiles
    )
    assert lower == 0.0
    assert upper == 17.0
    assert lower <= upper


def test_extreme_predictions_handling():
    """Verify extreme inputs (very high or zero) maintain valid ordering without error."""
    # Extreme high pollution level
    high_pred = 2500.0
    lower_h, upper_h, _ = compute_prediction_interval(high_pred, step=12)
    assert 0.0 <= lower_h <= high_pred <= upper_h

    # Zero pollution
    zero_pred = 0.0
    lower_z, upper_z, _ = compute_prediction_interval(zero_pred, step=1)
    assert lower_z == 0.0
    assert upper_z >= 0.0
    assert lower_z <= upper_z

    # Negative raw prediction edge-case
    neg_pred = -10.0
    lower_n, upper_n, _ = compute_prediction_interval(neg_pred, step=1)
    assert lower_n == 0.0
    assert upper_n >= 0.0
    assert lower_n <= upper_n


def test_conformal_quantiles_monotonicity():
    """Verify calibrated conformal quantiles are non-decreasing with horizon."""
    history = make_synthetic_history(200)
    feature_cols = get_feature_columns()
    
    # Train dummy model
    X = np.random.randn(100, len(feature_cols))
    y = np.random.uniform(10, 50, size=100)
    model = xgb.XGBRegressor(n_estimators=10, max_depth=3, random_state=42)
    model.fit(X, y)

    quantiles, count = calibrate_conformal_quantiles(
        model=model,
        history_df=history,
        calib_start_idx=100,
        calib_end_idx=200,
        feature_cols=feature_cols,
        horizon=24,
        alpha=0.10,
        stride_hours=4
    )

    assert count > 0
    assert len(quantiles) == 24
    for h in range(2, 25):
        assert quantiles[h] >= quantiles[h - 1], f"Quantile at step {h} ({quantiles[h]}) is smaller than step {h-1} ({quantiles[h-1]})"


def test_conformal_calibration_reproducibility():
    """Verify identical data and model yield identical conformal quantiles."""
    history = make_synthetic_history(180)
    feature_cols = get_feature_columns()

    X = np.random.randn(80, len(feature_cols))
    y = np.random.uniform(10, 40, size=80)
    model = xgb.XGBRegressor(n_estimators=10, max_depth=3, random_state=42)
    model.fit(X, y)

    q1, c1 = calibrate_conformal_quantiles(model, history, 80, 180, feature_cols, horizon=24, alpha=0.10, stride_hours=4)
    q2, c2 = calibrate_conformal_quantiles(model, history, 80, 180, feature_cols, horizon=24, alpha=0.10, stride_hours=4)

    assert c1 == c2
    assert q1 == q2


def test_predict_autoregressive_rollout_missing_columns():
    """Verify rollout handles missing optional pollutant columns using robust defaults."""
    df_minimal = pd.DataFrame({
        "time": [pd.Timestamp("2026-01-01") + pd.Timedelta(hours=i) for i in range(30)],
        "pm25": [20.0 + i for i in range(30)],
    })
    feature_cols = get_feature_columns()

    X = np.random.randn(20, len(feature_cols))
    y = np.random.uniform(10, 30, size=20)
    model = xgb.XGBRegressor(n_estimators=5, max_depth=2, random_state=42)
    model.fit(X, y)

    preds = predict_autoregressive_rollout(model, df_minimal, feature_cols, steps=12)
    assert len(preds) == 12
    for p in preds:
        assert isinstance(p, float)
        assert p >= 0.0


@pytest.mark.asyncio
async def test_small_sample_and_synthetic_provenance():
    """Verify synthetic demo mode explicitly marks intervals as heuristic without false confidence claims."""
    result = await train_and_forecast_pm25(28.61, 77.23, allow_demo=True)
    assert result["interval_method"] in ["split_conformal_prediction", "estimated_uncertainty_heuristic"]
    if result["is_synthetic"]:
        assert result["interval_method"] == "estimated_uncertainty_heuristic"
        assert result["nominal_coverage"] is None
        assert "heuristic" in result["interval_label"].lower()


@pytest.mark.asyncio
async def test_conformal_calibration_caching():
    """Verify calibration quantiles are stored in CALIBRATION_CACHE and reused on subsequent model runs."""
    hours = 400
    start = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=hours)
    times = [(start + pd.Timedelta(hours=i)).replace(tzinfo=None) for i in range(hours)]
    h_arr = np.array([t.hour for t in times])
    pm25 = (20.0 + 8.0 * np.sin(2 * np.pi * h_arr / 24.0)).tolist()

    mock_df = pd.DataFrame({
        "time": times,
        "pm25": pm25,
        "pm2_5": pm25,
        "no2": [14.0] * hours,
        "o3": [28.0] * hours,
        "wind_speed": [3.2] * hours,
        "temperature": [21.5] * hours,
        "humidity": [52.0] * hours,
    })

    norm_lat, norm_lon = normalize_coordinates(28.6139, 77.2090)
    assert (norm_lat, norm_lon) not in CALIBRATION_CACHE

    with patch("backend.ml_model.fetch_historical_air_quality", return_value=(mock_df, 0, False)):
        res1 = await train_and_forecast_pm25(28.6139, 77.2090, allow_demo=False)
        assert res1["interval_method"] == "split_conformal_prediction"
        assert (norm_lat, norm_lon) in CALIBRATION_CACHE

        # Clear MODEL_CACHE to force re-training/forecast, but keep CALIBRATION_CACHE
        MODEL_CACHE.clear()

        # calibrate_conformal_quantiles should NOT be called on second run because cache is warm
        with patch("backend.ml_model.calibrate_conformal_quantiles") as mock_calib:
            res2 = await train_and_forecast_pm25(28.6139, 77.2090, allow_demo=False)
            mock_calib.assert_not_called()
            assert res2["interval_method"] == "split_conformal_prediction"
            assert res2["calibration_samples"] == res1["calibration_samples"]


@pytest.mark.asyncio
async def test_concurrent_forecast_requests_locking():
    """Verify concurrent requests for the exact same coordinates safely serialize via coordinate lock."""
    hours = 120
    start = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=hours)
    times = [(start + pd.Timedelta(hours=i)).replace(tzinfo=None) for i in range(hours)]
    mock_df = pd.DataFrame({
        "time": times,
        "pm25": [25.0] * hours,
        "pm2_5": [25.0] * hours,
        "no2": [12.0] * hours,
        "o3": [25.0] * hours,
        "wind_speed": [3.5] * hours,
        "temperature": [22.0] * hours,
        "humidity": [50.0] * hours,
    })

    call_count = 0

    async def delayed_fetch(lat, lon, allow_demo=False):
        nonlocal call_count
        call_count += 1
        await asyncio.sleep(0.05)
        return mock_df, 0, False

    with patch("backend.ml_model.fetch_historical_air_quality", side_effect=delayed_fetch):
        # Fire two concurrent coroutines simultaneously for the same coordinates
        res1, res2 = await asyncio.gather(
            train_and_forecast_pm25(19.076, 72.8777, allow_demo=False),
            train_and_forecast_pm25(19.076, 72.8777, allow_demo=False)
        )
        assert res1["status"] == "success"
        assert res2["status"] == "success"
        # Due to double-checked locking, only 1 fetch/train occurred!
        assert call_count == 1


@pytest.mark.asyncio
async def test_pipeline_timings_logged_and_present():
    """Verify timings dictionary is present in the response with non-negative timing values."""
    mock_df = make_synthetic_history(120)
    with patch("backend.ml_model.fetch_historical_air_quality", return_value=(mock_df, 0, False)):
        result = await train_and_forecast_pm25(28.61, 77.23, allow_demo=True)
    assert "timings" in result
    timings = result["timings"]
    expected_keys = [
        "fetch_seconds",
        "feature_engineering_seconds",
        "train_seconds",
        "calibration_seconds",
        "forecast_seconds",
        "total_seconds"
    ]
    for key in expected_keys:
        assert key in timings
        assert isinstance(timings[key], (int, float))
        assert timings[key] >= 0.0
    assert timings["total_seconds"] >= timings["forecast_seconds"]


def test_insufficient_calibration_samples_fallback():
    """Verify calibrate_conformal_quantiles falls back to heuristic when fewer than 5 origins exist."""
    history = make_synthetic_history(30)
    feature_cols = get_feature_columns()
    X = np.random.randn(20, len(feature_cols))
    y = np.random.uniform(10, 30, size=20)
    model = xgb.XGBRegressor(n_estimators=5, max_depth=2, random_state=42)
    model.fit(X, y)

    # With only 30 rows and horizon 24, calib slice 20..30 has < 5 origins
    quantiles, count = calibrate_conformal_quantiles(
        model=model,
        history_df=history,
        calib_start_idx=20,
        calib_end_idx=30,
        feature_cols=feature_cols,
        horizon=24,
        stride_hours=6
    )
    assert quantiles == {}
    assert count == 0


@pytest.mark.asyncio
async def test_model_cache_hit_returns_cached_prediction():
    """Verify second request for identical location returns immediately from MODEL_CACHE."""
    with patch("backend.ml_model.fetch_historical_air_quality") as mock_fetch:
        mock_fetch.return_value = (make_synthetic_history(150), 0, False)
        res1 = await train_and_forecast_pm25(28.6139, 77.2090, allow_demo=True)
        assert res1["status"] == "success"
        initial_call_count = mock_fetch.call_count

        # Second call should hit MODEL_CACHE without calling fetch_historical_air_quality again
        res2 = await train_and_forecast_pm25(28.6139, 77.2090, allow_demo=True)
        assert res2["status"] == "success"
        assert res2 == res1
        assert mock_fetch.call_count == initial_call_count
