import pytest
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
    MODEL_CACHE
)


@pytest.fixture(autouse=True)
def clear_cache():
    MODEL_CACHE.clear()
    yield
    MODEL_CACHE.clear()


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
    # In synthetic demo mode
    result = await train_and_forecast_pm25(28.61, 77.23, allow_demo=True)
    assert result["interval_method"] in ["split_conformal_prediction", "estimated_uncertainty_heuristic"]
    if result["is_synthetic"]:
        assert result["interval_method"] == "estimated_uncertainty_heuristic"
        assert result["nominal_coverage"] is None
        assert "heuristic" in result["interval_label"].lower()
