import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from backend.evaluate_model import (
    calculate_metrics,
    chronological_split,
    evaluate_persistence,
    run_autoregressive_rollout
)
from backend.ml_model import engineer_features, get_feature_columns


def make_dummy_timeseries(n_hours=100, start_time="2026-08-01T00:00"):
    start_dt = datetime.fromisoformat(start_time)
    times = [start_dt + timedelta(hours=i) for i in range(n_hours)]
    h_arr = np.array([t.hour for t in times])
    pm25 = (20.0 + 10.0 * np.sin(2 * np.pi * h_arr / 24.0)).tolist()

    return pd.DataFrame({
        "time": times,
        "pm25": pm25,
        "pm2_5": pm25,
        "wind_speed": [4.0] * n_hours,
        "temperature": [22.0] * n_hours,
        "humidity": [55.0] * n_hours,
        "no2": [12.0] * n_hours,
        "o3": [28.0] * n_hours
    })


def test_chronological_split_strict_boundaries_no_leakage():
    df = make_dummy_timeseries(n_hours=100)
    train_df, val_df, test_df = chronological_split(df, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15)

    assert len(train_df) == 70
    assert len(val_df) == 15
    assert len(test_df) == 15

    # Strict chronological assertion: train strictly precedes val, val strictly precedes test
    assert train_df["time"].max() < val_df["time"].min()
    assert val_df["time"].max() < test_df["time"].min()

    # Verify no records overlap
    train_times = set(train_df["time"])
    val_times = set(val_df["time"])
    test_times = set(test_df["time"])

    assert train_times.isdisjoint(val_times)
    assert val_times.isdisjoint(test_times)
    assert train_times.isdisjoint(test_times)


def test_chronological_split_insufficient_data():
    df_small = make_dummy_timeseries(n_hours=30)
    with pytest.raises(ValueError) as exc:
        chronological_split(df_small)
    assert "Insufficient data" in str(exc.value)


def test_feature_engineering_no_target_or_future_leakage():
    """
    Verify that altering future ground truth at t+1 has ZERO effect on
    the feature vector generated at time t.
    """
    df1 = make_dummy_timeseries(n_hours=60)
    df2 = df1.copy()

    # Modify observation at index 50 to an extreme value
    target_idx = 50
    df2.loc[target_idx, "pm25"] = 9999.0
    df2.loc[target_idx, "pm2_5"] = 9999.0

    feat1 = engineer_features(df1)
    feat2 = engineer_features(df2)

    feature_cols = get_feature_columns()

    # For all indices < target_idx, features in feat1 and feat2 MUST BE IDENTICAL
    past_idx = target_idx - 1
    row1 = feat1.loc[past_idx, feature_cols]
    row2 = feat2.loc[past_idx, feature_cols]

    pd.testing.assert_series_equal(row1, row2, check_names=False)

    # In particular, verify wind_pm25_ratio does NOT use future/current target y_t
    assert "wind_pm25_ratio" in feature_cols
    assert row1["wind_pm25_ratio"] == row2["wind_pm25_ratio"]


def test_metric_calculations_accuracy():
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([12.0, 18.0, 34.0])

    # Absolute errors: [2, 2, 4] -> MAE = 8/3 = 2.67
    # Squared errors: [4, 4, 16] -> MSE = 24/3 = 8 -> RMSE = sqrt(8) = 2.83
    # Total variance: mean = 20 -> (10-20)^2 + (20-20)^2 + (30-20)^2 = 100 + 0 + 100 = 200
    # SS_res = 24 -> R2 = 1 - 24/200 = 1 - 0.12 = 0.88
    metrics = calculate_metrics(y_true, y_pred)

    assert metrics["mae"] == 2.67
    assert metrics["rmse"] == 2.83
    assert metrics["r2"] == 0.88


def test_metric_calculations_empty_and_zero_variance():
    m_empty = calculate_metrics(np.array([]), np.array([]))
    assert m_empty["mae"] == 0.0
    assert m_empty["rmse"] == 0.0
    assert m_empty["r2"] == 0.0

    # Flat true values (zero variance in ground truth)
    m_flat = calculate_metrics(np.array([15.0, 15.0, 15.0]), np.array([15.0, 15.0, 15.0]))
    assert m_flat["mae"] == 0.0
    assert m_flat["rmse"] == 0.0
    assert m_flat["r2"] == 0.0


def test_persistence_baseline():
    last_obs = 37.5
    preds = evaluate_persistence(last_obs, horizon=24)

    assert len(preds) == 24
    assert np.all(preds == 37.5)


def test_evaluate_location_dataset_excludes_future_provider_forecasts():
    """Verify that evaluate_location_dataset excludes records past cutoff_time so test set contains only completed observations."""
    from backend.evaluate_model import evaluate_location_dataset
    # Generate 120 hours: first 96 hours are historical, next 24 hours are future provider forecast
    df_all = make_dummy_timeseries(n_hours=120, start_time="2026-08-01T00:00")
    cutoff = datetime.fromisoformat("2026-08-04T23:00")  # Exactly hour 95

    res = evaluate_location_dataset("TestCity", df_all, horizon=12, eval_stride_hours=12, cutoff_time=cutoff)
    summary = res["dataset_summary"]

    assert summary["future_provider_forecasts_excluded"] is True
    # Verify that total observations evaluated is 96 (0..95), and future 24 hours were excluded
    assert summary["total_observations"] == 96
    # Verify the test set end time does not exceed cutoff
    test_end = datetime.fromisoformat(summary["test_end"])
    assert test_end <= cutoff
