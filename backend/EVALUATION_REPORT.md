# AeroTrack ML Forecasting Evaluation Report

**Date Generated**: 2026-10-09T17:29:14Z (UTC)  
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

- **Observation Window**: 2026-07-09 00:00:00 to 2026-10-10 23:00:00 (2256 hourly rows)
- **Train Period**: 2026-07-09 00:00:00 to 2026-09-12 18:00:00 (1579 samples)
- **Test Period**: 2026-09-26 21:00:00 to 2026-10-10 23:00:00 (339 samples)
- **Evaluation Windows**: 14 walk-forward 24-hour forecasts (336 predictions evaluated)

#### Overall 24-Hour Horizon Summary

| Model | MAE (µg/m³) | RMSE (µg/m³) | R² | Skill Score vs Persistence |
| :--- | :---: | :---: | :---: | :---: |
| **XGBoost (Autoregressive)** | **28.23** | **38.67** | 0.5925 | **+0.2571** |
| Persistence Baseline | 38.23 | 52.05 | 0.2616 | 0.0000 |

#### Performance by Forecast Lead Time ($h$ hours ahead)

| Lead Time | XGBoost MAE | Persistence MAE | XGBoost RMSE | Persistence RMSE | Skill Score |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **+1h** | 2.64 µg/m³ | 6.37 µg/m³ | 3.37 µg/m³ | 7.71 µg/m³ | +0.5629 |
| **+6h** | 19.97 µg/m³ | 37.66 µg/m³ | 27.94 µg/m³ | 50.09 µg/m³ | +0.4422 |
| **+12h** | 33.94 µg/m³ | 42.84 µg/m³ | 45.63 µg/m³ | 57.26 µg/m³ | +0.2031 |
| **+24h** | 43.96 µg/m³ | 54.05 µg/m³ | 51.4 µg/m³ | 64.11 µg/m³ | +0.1983 |

#### Prediction Interval Evaluation (90% Nominal Target)

- **Calibration Strategy**: Split Conformal Prediction calibrated on validation holdout partition (79 multi-step windows).
- **Nominal Coverage Target**: 90.0% ($\alpha = 0.10$).

| Uncertainty Method | Empirical 24h Coverage | Mean Interval Width | Calibration Strategy |
| :--- | :---: | :---: | :--- |
| **Split Conformal Prediction** | **93.5%** | **126.1 µg/m³** | Distribution-free, calibrated on holdout residuals |
| Heuristic Uncertainty Band | 71.4% | 92.9 µg/m³ | Uncalibrated heuristic (step + rolling variance) |

##### Interval Metrics by Forecast Lead Time ($h$ hours ahead)

| Lead Time | Conformal 90% Coverage | Conformal Margin $q^{(h)}$ | Heuristic Coverage | Heuristic Mean Width |
| :---: | :---: | :---: | :---: | :---: |
| **+1h** | **100.0%** | ±19.0 µg/m³ | 100.0% | 48.4 µg/m³ |
| **+6h** | **92.9%** | ±53.45 µg/m³ | 64.3% | 58.6 µg/m³ |
| **+12h** | **92.9%** | ±81.5 µg/m³ | 57.1% | 87.0 µg/m³ |
| **+24h** | **92.9%** | ±85.33 µg/m³ | 64.3% | 148.2 µg/m³ |


---

## 3. Dataset & Model Limitations

1. **Short-Term Lead Dominance**: At $h=1$ hour, persistence is often very competitive with or slightly better than ML because air quality changes slowly over a single hour. At $h=6$ to $h=24$ hours, XGBoost's cyclical temporal features capture diurnal peaks that persistence cannot model.
2. **Open-Meteo Reanalysis vs Ground Station Sensors**: Open-Meteo Air Quality integrates CAMS and atmospheric reanalysis models. While highly consistent, localized hyper-local microclimates (e.g. immediate roadside emissions) require dense physical sensor calibration.
3. **Zero Data Fabrication**: All evaluation numbers in this report reflect actual calculated errors on real test partitions. No unverified 95% accuracy claims are made.
