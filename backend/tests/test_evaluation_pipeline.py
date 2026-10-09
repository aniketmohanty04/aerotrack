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


def test_changing_future_observations_cannot_change_historical_features_or_targets():
    """
    Regression Test: Proves that changing future observations cannot change
    historical training features or targets when cutoff isolation is enforced.
    """
    from backend.ml_model import clean_and_impute_series

    cutoff = datetime.fromisoformat("2026-08-05T00:00")
    
    # 96 hours of past data ending at cutoff
    df_past = make_dummy_timeseries(n_hours=96, start_time="2026-08-01T01:00")
    # Introduce a missing value in the last historical observation
    df_past.loc[95, "pm25"] = np.nan
    df_past.loc[95, "pm2_5"] = np.nan
    
    # Future scenario A: Future PM2.5 spikes to 300 µg/m³
    times_future = [cutoff + timedelta(hours=i) for i in range(1, 25)]
    df_future_A = pd.DataFrame({
        "time": times_future,
        "pm25": [300.0] * 24,
        "pm2_5": [300.0] * 24,
        "wind_speed": [0.5] * 24,
        "temperature": [35.0] * 24,
        "humidity": [80.0] * 24,
        "no2": [45.0] * 24,
        "o3": [80.0] * 24
    })
    
    # Future scenario B: Future PM2.5 drops to 2 µg/m³
    df_future_B = pd.DataFrame({
        "time": times_future,
        "pm25": [2.0] * 24,
        "pm2_5": [2.0] * 24,
        "wind_speed": [12.0] * 24,
        "temperature": [5.0] * 24,
        "humidity": [20.0] * 24,
        "no2": [2.0] * 24,
        "o3": [10.0] * 24
    })
    
    df_combined_A = pd.concat([df_past, df_future_A], ignore_index=True)
    df_combined_B = pd.concat([df_past, df_future_B], ignore_index=True)
    
    # Isolate at historical cutoff and clean within partition
    past_clean_A = clean_and_impute_series(df_combined_A[df_combined_A["time"] <= cutoff].copy().reset_index(drop=True))
    past_clean_B = clean_and_impute_series(df_combined_B[df_combined_B["time"] <= cutoff].copy().reset_index(drop=True))
    
    # 1. Historical data frames must be 100% bit-for-bit identical regardless of future values
    pd.testing.assert_frame_equal(past_clean_A, past_clean_B)
    
    # 2. Historical engineered features must be 100% identical regardless of future values
    feat_A = engineer_features(past_clean_A)
    feat_B = engineer_features(past_clean_B)
    pd.testing.assert_frame_equal(feat_A, feat_B)
    
    # 3. Verify that the missing historical row at index 95 was NOT filled with future values (300 or 2)
    # It must be forward-filled from past row 94
    expected_filled_val = df_past.loc[94, "pm25"]
    assert past_clean_A.loc[95, "pm25"] == pytest.approx(expected_filled_val)
    assert past_clean_B.loc[95, "pm25"] == pytest.approx(expected_filled_val)
    assert past_clean_A.loc[95, "pm25"] != 300.0
    assert past_clean_B.loc[95, "pm25"] != 2.0


def test_causal_imputation_earlier_rows_cannot_depend_on_later_observations():
    """
    Direct causal independence test:
    Proves that missing values in earlier rows are causally imputed strictly from past data,
    and modifying ANY later row produces zero change in earlier imputed values or features.
    """
    from backend.ml_model import clean_and_impute_series

    # Base dataset of 80 hours
    df_base1 = make_dummy_timeseries(n_hours=80, start_time="2026-07-01T00:00")
    # Introduce missing values in interior rows 30 and 31
    df_base1.loc[30, "pm25"] = np.nan
    df_base1.loc[31, "pm25"] = np.nan

    # Create variant 2 where subsequent row 50 has a massive spike, and row 60 is dropped
    df_base2 = df_base1.copy()
    df_base2.loc[50, "pm25"] = 999.0
    df_base2.loc[50, "pm2_5"] = 999.0
    df_base2.loc[60, "pm25"] = 1.0
    df_base2.loc[60, "pm2_5"] = 1.0

    imputed1 = clean_and_impute_series(df_base1)
    imputed2 = clean_and_impute_series(df_base2)

    # Imputed rows 0 through 49 MUST be 100% bit-for-bit identical
    pd.testing.assert_frame_equal(imputed1.iloc[:50], imputed2.iloc[:50])

    # Check causal value: rows 30 and 31 must equal row 29, NOT interpolated towards row 50
    expected_val = df_base1.loc[29, "pm25"]
    assert imputed1.loc[30, "pm25"] == pytest.approx(expected_val)
    assert imputed1.loc[31, "pm25"] == pytest.approx(expected_val)
    assert imputed2.loc[30, "pm25"] == pytest.approx(expected_val)
    assert imputed2.loc[31, "pm25"] == pytest.approx(expected_val)

    # Features for rows 0 through 49 must also be 100% identical
    feat1 = engineer_features(imputed1)
    feat2 = engineer_features(imputed2)
    pd.testing.assert_frame_equal(feat1.iloc[:50], feat2.iloc[:50])

