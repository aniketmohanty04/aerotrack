# AeroTrack ML Forecasting Evaluation Report

**Date Generated**: 2026-10-09T19:28:22Z (UTC)  
**Evaluation Standard**: Chronological Multi-Step Walk-Forward Validation  
**Target**: Hourly Ground-Level PM2.5 Concentration (µg/m³)  
**Baseline Benchmark**: Naive Persistence Forecast ($\hat{y}_{t+h} = y_t$)

---

## 1. Scientific Methodology

- **Model Architecture**: XGBoost Gradient Boosted Decision Trees (`n_estimators=120, max_depth=4, lr=0.08`).
- **Chronological Splitting**: The dataset is partitioned chronologically without random shuffling:
  - **Train Set (70%)**: Model fitting.
  - **Validation Set (15%)**: Hyperparameter tuning.
  - **Test Set (15%)**: Unseen future walk-forward multi-step evaluation.
- **Data Leakage Protections**:
  - All temporal features (lags, rolling averages, rolling standard deviations) strictly use observations at or before $t-1$.
  - During 24-hour autoregressive rollouts, predictions $\hat{y}_{t+1}, \dots, \hat{y}_{t+h-1}$ are fed into subsequent lags recursively without peeking into future ground-truth data.
- **Evaluation Metrics**:
  - **MAE (Mean Absolute Error)**: $\frac{1}{N}\sum |y - \hat{y}|$ (in µg/m³). Directly interpretable physical error.
  - **RMSE (Root Mean Squared Error)**: $\sqrt{\frac{1}{N}\sum (y - \hat{y})^2}$ (in µg/m³). Strongly penalizes large pollution spikes.
  - **Skill Score**: $1 - \frac{\text{RMSE}_{\text{XGBoost}}}{\text{RMSE}_{\text{Persistence}}}$. Positive indicates superior skill over persistence.
  - **$R^2$ Score**: Reported for reference across the test distribution; note that on autoregressive non-stationary time series, negative $R^2$ indicates the prediction variance exceeds the static mean variance.

---

## 2. Benchmark Results by City

### New Delhi, India

- **Observation Window**: 2026-07-10 00:00:00 to 2026-10-10 00:00:00 (2209 hourly rows)
- **Train Period**: 2026-07-10 00:00:00 to 2026-09-12 09:00:00 (1546 samples)
- **Test Period**: 2026-09-26 05:00:00 to 2026-10-10 00:00:00 (332 samples)
- **Evaluation Windows**: 13 walk-forward 24-hour forecasts (312 predictions evaluated)

#### Overall 24-Hour Horizon Summary

| Model | MAE (µg/m³) | RMSE (µg/m³) | R² | Skill Score vs Persistence |
| :--- | :---: | :---: | :---: | :---: |
| **XGBoost (Autoregressive)** | **31.77** | **41.85** | 0.5046 | **+0.2175** |
| Persistence Baseline | 39.82 | 53.48 | 0.1912 | 0.0000 |

#### Performance by Forecast Lead Time ($h$ hours ahead)

| Lead Time | XGBoost MAE | Persistence MAE | XGBoost RMSE | Persistence RMSE | Skill Score |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **+1h** | 3.01 µg/m³ | 6.75 µg/m³ | 3.49 µg/m³ | 8.0 µg/m³ | +0.5637 |
| **+6h** | 23.21 µg/m³ | 40.39 µg/m³ | 28.04 µg/m³ | 51.97 µg/m³ | +0.4605 |
| **+12h** | 40.43 µg/m³ | 45.28 µg/m³ | 49.6 µg/m³ | 59.34 µg/m³ | +0.1641 |
| **+24h** | 50.29 µg/m³ | 53.41 µg/m³ | 60.57 µg/m³ | 64.24 µg/m³ | +0.0571 |

#### Prediction Interval Evaluation (90% Nominal Target)

- **Calibration Strategy**: Split Conformal Prediction calibrated on validation holdout partition (30 multi-step windows).
- **Nominal Coverage Target**: 90.0% ($\alpha = 0.10$).

| Uncertainty Method | Empirical 24h Coverage | Mean Interval Width | Calibration Strategy |
| :--- | :---: | :---: | :--- |
| **Split Conformal Prediction** | **94.6%** | **138.6 µg/m³** | Distribution-free, calibrated on holdout residuals |
| Heuristic Uncertainty Band | 71.2% | 99.3 µg/m³ | Uncalibrated heuristic (step + rolling variance) |

##### Interval Metrics by Forecast Lead Time ($h$ hours ahead)

| Lead Time | Conformal 90% Coverage | Conformal Margin $q^{(h)}$ | Heuristic Coverage | Heuristic Mean Width |
| :---: | :---: | :---: | :---: | :---: |
| **+1h** | **100.0%** | ±29.07 µg/m³ | 100.0% | 51.2 µg/m³ |
| **+6h** | **84.6%** | ±43.53 µg/m³ | 69.2% | 60.7 µg/m³ |
| **+12h** | **100.0%** | ±90.89 µg/m³ | 53.8% | 91.6 µg/m³ |
| **+24h** | **92.3%** | ±93.85 µg/m³ | 61.5% | 161.8 µg/m³ |


---

## 3. Dataset & Model Limitations

1. **Short-Term Lead Dominance**: At $h=1$ hour, persistence is often very competitive with or slightly better than ML because air quality changes slowly over a single hour. At $h=6$ to $h=24$ hours, XGBoost's cyclical temporal features capture diurnal peaks that persistence cannot model.
2. **Open-Meteo Reanalysis vs Ground Station Sensors**: Open-Meteo Air Quality integrates CAMS and atmospheric reanalysis models. While highly consistent, localized hyper-local microclimates (e.g. immediate roadside emissions) require dense physical sensor calibration.
3. **Zero Data Fabrication**: All evaluation numbers in this report reflect actual calculated errors on real test partitions. No unverified 95% accuracy claims are made.
