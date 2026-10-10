# AeroTrack: Real-Time Air Quality Monitoring & ML Forecasting

> High-resolution atmospheric intelligence, global map telemetry, and on-demand XGBoost machine learning forecasts for any coordinate on Earth.

---

## 🔗 Live Demo

- **Frontend:** https://aerotrack-three.vercel.app
- **Backend API:** https://aerotrack-backend-1tlt.onrender.com
- **API Documentation:** https://aerotrack-backend-1tlt.onrender.com/docs

> **Note:** The backend runs on Render's free tier, which normally sleeps after 15 minutes of inactivity. An UptimeRobot monitor pings the health endpoint every 5 minutes to keep it warm 24/7, so users experience sub-second response times.

---

## 🚀 Uptime Optimization

The backend is deployed on Render's free tier, which typically sleeps after 15 minutes of inactivity. To eliminate cold-start delays during demo presentations and evaluation, an **UptimeRobot** monitor pings the `/api/health` endpoint every 5 minutes, keeping the service warm 24/7.

**Configuration:**
- **Monitor Type:** HTTP(s)
- **URL:** https://aerotrack-backend-1tlt.onrender.com/api/health
- **Interval:** 5 minutes
- **Effect:** Backend response time drops from ~25s (cold) to <500ms (warm)

This is a production-grade pattern for keeping free-tier services responsive without upgrading to paid plans.

---

## 1. Overview

**AeroTrack** is an end-to-end environmental intelligence platform built in response to the **"Real-Time Air Quality Monitoring and Prediction for User-Selected Regions"** problem statement. Atmospheric pollutants such as fine particulate matter (PM2.5) pose severe public health risks worldwide, yet accessing localized, actionable air quality data combined with dependable forward-looking projections remains challenging for general citizens and researchers alike.

AeroTrack bridges this gap by enabling users to either click any coordinate across an interactive global map or query any city or neighborhood via an autocomplete search bar. The system immediately aggregates multi-pollutant telemetry, maps readings to official standard thresholds, and calculates demographic-specific health advisories.

To anticipate air quality fluctuations, AeroTrack deploys an on-demand machine learning pipeline powered by **XGBoost**. When a location is queried, the backend dynamically fetches up to 92 days of hourly atmospheric and meteorological historical data, extracts temporal, lag, rolling, and interaction features, trains a localized gradient boosted regression model, and generates an autoregressive 24-hour PM2.5 forecast complete with step-dependent uncertainty bounds.

---

## 2. Features

- **Interactive Leaflet World Map**: Click-to-query map interface supporting smooth panning, zooming, coordinate clamping, and custom location markers.
- **Place Search with Autocomplete**: Rate-limited, debounced search against OpenStreetMap's Nominatim engine with direct jump-to-location behavior.
- **Reverse Geocoding**: Multi-provider resolution of GPS coordinates to city, administrative subdivision, and country names via OpenStreetMap Nominatim with BigDataCloud fallback (24h caching for successful resolutions, zero failure caching).
- **Real-Time Pollutant Breakdown**: Instant telemetry cards for 6 primary criteria air pollutants:
  - Fine Particulate Matter (PM2.5)
  - Coarse Particulate Matter (PM10)
  - Nitrogen Dioxide (NO₂)
  - Sulphur Dioxide (SO₂)
  - Ground-Level Ozone (O₃)
  - Carbon Monoxide (CO)
- **US EPA AQI Rating Engine**: Breakpoint mapping adhering to US EPA standards, color-coded across all 6 official severity tiers.
- **7-Day Historical Trends**: Interactive time-series charts displaying 168+ hours of historical data with a metric toggle between Fine Dust PM2.5 (µg/m³) and US AQI.
- **24-Hour ML Forecast**: On-demand hourly PM2.5 projections generated autoregressively by XGBoost, featuring heuristic uncertainty bands and plain-English summary alerts.
- **Dual-Layer Reference Lines**: Visual benchmarks displaying the **Safe Limit (15 µg/m³, WHO 2021 Guideline)** and **Caution Limit (35.5 µg/m³, US EPA Moderate-to-USG Threshold)**.
- **Actionable Health Advisories**: Semantic, rule-based recommendations tailored for Outdoor Activities, Home Ventilation, Mask Usage, and Sensitive Demographic Groups.

---

## 3. Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                        REACT FRONTEND (Vite + TS)                      │
│                                                                        │
│   ┌───────────────────────────┐     ┌──────────────────────────────┐   │
│   │ Leaflet World Map View    │     │ SearchBar Autocomplete       │   │
│   │ (Click coordinate capture)│     │ (OpenStreetMap Nominatim)    │   │
│   └─────────────┬─────────────┘     └──────────────┬───────────────┘   │
│                 │                                  │                   │
│   ┌─────────────┴──────────────────────────────────┴───────────────┐   │
│   │ Recharts Visualizations: TrendChart (7d) & ForecastChart (24h)  │   │
│   └────────────────────────────────┬───────────────────────────────┘   │
└────────────────────────────────────┼───────────────────────────────────┘
                                     │ HTTP REST Requests (/api/*)
                                     ▼
┌────────────────────────────────────────────────────────────────────────┐
│                        FASTAPI BACKEND (Python 3.10+)                  │
│                                                                        │
│   ┌────────────────────────────────────────────────────────────────┐   │
│   │ Middleware: CORS Policy, Coordinate Normalization, Rate-Limits │   │
│   └────────────────┬──────────────────────────────┬────────────────┘   │
│                    │                              │                    │
│     ┌──────────────▼─────────────┐  ┌─────────────▼──────────────┐     │
│     │   Shared Air Cache (10m)   │  │   Model Cache (30m TTL)    │     │
│     │   Live Telemetry & Trends  │  │   Fitted XGBoost & Arrays  │     │
│     └──────────────┬─────────────┘  └─────────────┬──────────────┘     │
│                    │                              │                    │
│                    ▼                              ▼                    │
│   ┌──────────────────────────────┐  ┌──────────────────────────────┐   │
│   │ External Upstream Services   │  │ XGBoost ML Pipeline          │   │
│   │ • Open-Meteo Air Quality API │  │ • Feature Engineering        │   │
│   │ • Nominatim Search API       │  │ • Autoregressive 24h Loop    │   │
│   │ • BigDataCloud Reverse Geo   │  │ • Step-Dependent Uncertainty │   │
│   └──────────────────────────────┘  └──────────────────────────────┘   │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 4. ML Pipeline

The predictive engine employs an autoregressive Extreme Gradient Boosting (XGBoost) workflow operating on localized historical and meteorological time-series.

### Pipeline Stages

1. **Data Acquisition (92 Days Hourly)**:
   - Fetches 92 days of hourly readings from Open-Meteo Air Quality and Weather APIs: PM2.5, NO₂, O₃, 10-meter wind speed, 2-meter temperature, and 2-meter relative humidity.
   - Timestamps use `timezone="auto"` and are localized to local solar time using `utc_offset_seconds`.
2. **Feature Engineering**:
   - **Cyclical Time Encodings**: Hour of the day (0–23) and day of the week (0–6) transformed onto continuous trigonometric circles:
     - `hour_sin = sin(2π · hour / 24)`, `hour_cos = cos(2π · hour / 24)`
     - `dow_sin = sin(2π · dow / 7)`, `dow_cos = cos(2π · dow / 7)`
   - **Autoregressive Lag Features**: 1h, 2h, 3h, 6h, 12h, and 24h lag variables capturing immediate momentum and daily diurnal periodicity (`pm25_lag1` through `pm25_lag24`).
   - **Polynomial Lag Features**: `pm25_lag1_squared` and `pm25_lag24_squared` enabling non-linear splits during rapid pollutant spikes.
   - **Rolling Window Statistics**: 6-hour moving mean (`rolling_mean_6`), 24-hour moving mean (`rolling_mean_24`), and 24-hour moving standard deviation (`rolling_std_24`) calculated on shifted values to prevent lookahead data leakage.
   - **Meteorological Interactions**:
     - `wind_pm25_ratio = pm25_lag1 / (wind_speed + 0.1)` (stagnant vs. dispersion conditions)
     - `humidity_temp_ratio = humidity / (temperature + 40.0)` (boundary layer inversion proxy)
     - `no2_o3_ratio = no2 / (o3 + 1.0)` (photochemical equilibrium proxy)
3. **Model Configuration**:
   - **Algorithm**: `xgb.XGBRegressor`
   - **Estimators**: 120 trees
   - **Tree Depth**: `max_depth = 4` (prevents overfitting on small local sample sizes)
   - **Learning Rate**: `0.08`
   - **Subsample & Colsample**: `0.85` subsample, `0.85` colsample per tree
4. **Forecast Method (Autoregressive 24-Step Loop with Dynamic Weather Updates)**:
   - For each step `h` in 1 to 24, the model generates an inference for step `h`.
   - The predicted value is appended to the feature array and becomes the new lag-1 input for step `h + 1`. Lags, rolling statistics, and temporal encodings roll forward dynamically.
   - Dynamic meteorological conditions (wind speed, temperature, humidity, NO₂, O₃) are matched to future forecast hours to capture upcoming weather shifts (with graceful fallback to latest observed values if unavailable).
5. **Uncertainty Quantification (Split Conformal Prediction & Heuristic Fallback)**:
   - **Split Conformal Prediction (Calibrated 90% Intervals)**: Evaluates multi-step rolling residual calibrations over historical validation horizons (`calibrate_conformal_quantiles`), producing horizon-specific non-conformity quantiles $q^{(h)}$ to construct distribution-free prediction intervals $[\hat{y}_h - q^{(h)}, \hat{y}_h + q^{(h)}]$ with 90% nominal coverage guarantees ($\alpha = 0.10$).
   - **Heuristic Bands (Fallback)**: Used strictly when historical samples are sparse (<100 samples) or during mock fallback mode:
     - `Margin_h = 1.645 · sigma_residuals · (1.0 + 0.03 · h)`
     - `Lower_h = max(0, Prediction_h - Margin_h)`
     - `Upper_h = Prediction_h + Margin_h`
6. **Two-Tier In-Memory Caching**:
   - **10-Minute Shared Air Cache**: Stores live telemetry and 7-day trends keyed by `(round(lat, 4), round(lon, 4))`, eliminating duplicate requests between `/api/air-quality` and `/api/trends`.
   - **30-Minute Model Cache**: Retains trained XGBoost estimators and generated 24-hour forecasts per coordinate pair.

---

## 5. Reproducible ML Evaluation & Benchmarks

AeroTrack avoids unverified accuracy claims by including an open, fully reproducible evaluation script that validates the forecasting pipeline against a **Naive Persistence Baseline** ($\hat{y}_{t+h} = y_t$).

To eliminate data leakage, the held-out test partition strictly contains completed historical observations (excluding future provider forecasts) and is evaluated chronologically across multi-step walk-forward horizons.

### Reproducing the Benchmark
Run the reproducible evaluation script against genuine Open-Meteo observations for all benchmark locations:
```bash
python backend/evaluate_model.py
```
Or evaluate an individual city:
```bash
python backend/evaluate_model.py --city Delhi
```

### Verified Multi-City Benchmark Results
Evaluated on completed historical observations across 13 walk-forward 24-hour evaluation horizons (312 held-out hourly forecast steps per city) with zero future provider forecast contamination and strictly causal imputation.

#### Overall 24-Hour Horizon Summary

| Location | Model | MAE (µg/m³) | RMSE (µg/m³) | $R^2$ Score | Skill Score vs Persistence |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **New Delhi, India** | **XGBoost (Autoregressive)** | **23.88** | **31.76** | **0.7148** | **+0.4061** (+40.6% RMSE improvement) |
| | Persistence Baseline | 39.82 | 53.48 | 0.1912 | 0.0000 |
| **London, UK** | **XGBoost (Autoregressive)** | **0.99** | **1.34** | **0.5903** | **+0.2472** (+24.7% RMSE improvement) |
| | Persistence Baseline | 1.24 | 1.78 | 0.2762 | 0.0000 |
| **New York, USA** | **XGBoost (Autoregressive)** | **5.46** | **9.39** | **0.7463** | **+0.4677** (+46.8% RMSE improvement) |
| | Persistence Baseline | 10.93 | 17.64 | 0.1039 | 0.0000 |

#### Performance by Forecast Lead Time ($h$ hours ahead)

| Location | Lead Time | XGBoost MAE | Persistence MAE | XGBoost RMSE | Persistence RMSE | Skill Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **New Delhi** | **+1h** | **2.41 µg/m³** | 6.75 µg/m³ | **3.00 µg/m³** | 8.00 µg/m³ | **+0.6250** |
| | **+6h** | **15.35 µg/m³** | 40.39 µg/m³ | **18.27 µg/m³** | 51.97 µg/m³ | **+0.6485** |
| | **+12h** | **32.04 µg/m³** | 45.28 µg/m³ | **39.02 µg/m³** | 59.34 µg/m³ | **+0.3424** |
| | **+24h** | **29.18 µg/m³** | 53.41 µg/m³ | **33.09 µg/m³** | 64.24 µg/m³ | **+0.4849** |
| **London** | **+1h** | **0.42 µg/m³** | 0.58 µg/m³ | **0.53 µg/m³** | 0.75 µg/m³ | **+0.2933** |
| | **+6h** | 1.25 µg/m³ | **1.16 µg/m³** | 1.67 µg/m³ | **1.43 µg/m³** | -0.1678 |
| | **+12h** | **0.72 µg/m³** | 1.15 µg/m³ | **0.84 µg/m³** | 1.61 µg/m³ | **+0.4783** |
| | **+24h** | **1.23 µg/m³** | 1.91 µg/m³ | **1.49 µg/m³** | 2.55 µg/m³ | **+0.4157** |
| **New York** | **+1h** | **0.83 µg/m³** | 1.71 µg/m³ | **1.16 µg/m³** | 2.17 µg/m³ | **+0.4654** |
| | **+6h** | **3.78 µg/m³** | 7.83 µg/m³ | **5.32 µg/m³** | 11.16 µg/m³ | **+0.5233** |
| | **+12h** | **5.90 µg/m³** | 13.47 µg/m³ | **10.29 µg/m³** | 19.64 µg/m³ | **+0.4761** |
| | **+24h** | **11.37 µg/m³** | 13.00 µg/m³ | **18.86 µg/m³** | 18.93 µg/m³ | **+0.0037** |

#### Uncertainty Interval Empirical Coverage (90% Nominal Target)

| Location | Split Conformal Coverage (90% Target) | Conformal Mean Width | Heuristic Fallback Coverage | Heuristic Mean Width |
| :--- | :---: | :---: | :---: | :---: |
| **New Delhi** | **90.4%** | 99.3 µg/m³ | 79.2% | 87.8 µg/m³ |
| **London** | **95.2%** | 6.6 µg/m³ | 97.4% | 5.6 µg/m³ |
| **New York** | **92.6%** | 33.7 µg/m³ | 79.8% | 19.9 µg/m³ |

---

## 6. AQI Standard

AeroTrack locks all classification to the official **US Environmental Protection Agency (US EPA)** standards as the single source of truth across all 6 criteria pollutants.

### US EPA Breakpoint Table

All concentrations are measured in **µg/m³**:

| AQI Category | AQI Range | Color | PM2.5 (24h) | PM10 (24h) | O₃ (8h) | CO (8h) | SO₂ (1h) | NO₂ (1h) |
|---|---|---|---|---|---|---|---|---|
| **Good** | 0 – 50 | `#10b981` (Green) | 0.0 – 9.0 | 0 – 54 | 0.0 – 107.9 | 0 – 5,039 | 0.0 – 91.7 | 0.0 – 99.6 |
| **Moderate** | 51 – 100 | `#eab308` (Yellow) | 9.1 – 35.4 | 55 – 154 | 108.0 – 137.2 | 5,040 – 10,763 | 91.8 – 196.5 | 99.7 – 188.0 |
| **Unhealthy for Sensitive Groups** | 101 – 150 | `#f97316` (Orange) | 35.5 – 55.4 | 155 – 254 | 137.3 – 166.6 | 10,764 – 14,198 | 196.6 – 484.7 | 188.1 – 676.8 |
| **Unhealthy** | 151 – 200 | `#ef4444` (Red) | 55.5 – 125.4 | 255 – 354 | 166.7 – 205.8 | 14,199 – 17,633 | 484.8 – 796.5 | 676.9 – 1,220.1 |
| **Very Unhealthy** | 201 – 300 | `#8b5cf6` (Purple) | 125.5 – 225.4 | 355 – 424 | 205.9 – 392.0 | 17,634 – 34,808 | 796.6 – 1,582.5 | 1,220.2 – 2,348.1 |
| **Hazardous** | 301+ | `#7f1d1d` (Maroon) | > 225.4 | > 424 | > 392.0 | > 34,808 | > 1,582.5 | > 2,348.1 |

### WHO 2021 Reference Guideline vs. US EPA Breakpoints
On both the 7-day trend chart and the 24-hour forecast chart, AeroTrack displays two constant visual benchmarks:
- **Safe Limit (15 µg/m³)**: The World Health Organization (WHO) 2021 recommended 24-hour guideline for PM2.5 exposure.
- **Caution Limit (35.5 µg/m³)**: The official US EPA PM2.5 breakpoint where air quality transitions from Moderate (9.1–35.4 µg/m³, AQI 51–100) to Unhealthy for Sensitive Groups (35.5–55.4 µg/m³, AQI 101–150). Under EPA breakpoints, this transition is strictly at 35.5 µg/m³ (not 35.0 µg/m³), distinct from the WHO 15 µg/m³ limit.

---

## 7. Tech Stack

| Layer | Technologies |
|---|---|
| **Frontend Framework** | React 18, TypeScript, Vite |
| **UI Styling** | Tailwind CSS, Lucide React Icons |
| **Mapping & Geospatial** | Leaflet, React-Leaflet, OpenStreetMap Tiles |
| **Data Visualizations** | Recharts (Responsive SVG Charts) |
| **Backend Framework** | FastAPI, Python 3.10+, Uvicorn (ASGI) |
| **HTTP Client** | HTTPX (Async Client) |
| **Machine Learning** | XGBoost (`XGBRegressor`), Scikit-Learn |
| **Data Processing** | Pandas, NumPy |
| **External APIs** | Open-Meteo Air Quality & Weather API, Nominatim (OpenStreetMap), BigDataCloud |
| **Deployment** | Render (Backend API), Vercel (Frontend SPA) |

---

## 8. API Endpoints & Specification

### Endpoints Directory

| Method | Endpoint | Query / Path Parameters | Error Responses | Description |
|---|---|---|---|---|
| `GET` / `HEAD` | `/` | — | — | API status manifest, health overview, documentation directory, and live frontend link |
| `GET` / `HEAD` | `/api/health` | — | — | Lightweight uptime ping endpoint (zero external dependencies, sub-5ms) |
| `GET` / `HEAD` | `/api/info` | — | — | API metadata and endpoint index |
| `GET` | `/api/air-quality/{lat}/{lon}` | `lat` (-90 to 90), `lon` (-180 to 180) | `400`, `502`, `500` | Current criteria pollutant telemetry, US AQI score, individual pollutant ratings, and advisories |
| `GET` | `/api/trends/{lat}/{lon}` | `lat` (-90 to 90), `lon` (-180 to 180) | `400`, `502`, `500` | 7-day hourly historical & forecast trends with strictly isolated historical rolling statistics |
| `GET` | `/api/predict/{lat}/{lon}` | `lat`, `lon`, `allow_demo` (bool, default `false`) | `400`, `422`, `502`, `500` | On-demand XGBoost 24-hour PM2.5 forecast with Split Conformal 90% prediction intervals |
| `GET` | `/api/search` | `q` (string, min length 1) | `429`, `502`, `504` | Autocomplete place search via Nominatim (cached 10m; distinguishes empty results from errors) |
| `GET` | `/api/reverse-geocode/{lat}/{lon}` | `lat` (-90 to 90), `lon` (-180 to 180) | `400` | Two-tier GPS reverse geocoding via Nominatim + BigDataCloud fallback (24h cache on success) |

### Environment Variables

| Variable | Scope | Default Value | Description |
|---|---|---|---|
| `ALLOWED_ORIGINS` | Backend | `https://aerotrack-three.vercel.app,http://localhost:5173,...` | Comma-separated list of permitted CORS origins |
| `PORT` | Backend | `8000` | Server listen port for Uvicorn |
| `VITE_API_URL` | Frontend | `""` (or production backend URL) | Backend API base URL consumed by Vite React SPA |

---

## 9. Running Locally

### Prerequisites
- **Python**: 3.10 or higher
- **Node.js**: 18.0 or higher
- **Package Managers**: `pip` and `npm`

---

### Backend Setup

#### PowerShell (Windows)
```powershell
cd backend
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

#### Bash (Linux / macOS)
```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

---

### Frontend Setup

#### PowerShell (Windows)
```powershell
cd frontend
npm install
npm run dev
```

#### Bash (Linux / macOS)
```bash
cd frontend
npm install
npm run dev
```

- **Frontend App**: [http://localhost:5173](http://localhost:5173)
- **Backend API Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 10. Design Decisions

| Decision | Rationale |
|---|---|
| **Open-Meteo over AccuWeather** | Open-Meteo provides free, global, hourly multi-pollutant reanalysis and meteorological data without mandatory API key limits, enabling universal global coverage out of the box. |
| **XGBoost over LSTM / Deep Learning** | On-demand training for arbitrary coordinates requires sub-2-second latency. XGBoost trains in under 1.5 seconds on CPU, handles missing observations, and consistently outperforms LSTMs on small tabular time-series (~2,200 rows). |
| **Autoregressive Forecast** | Allows the model to dynamically update lag buffers, moving averages, and weather variables hour-by-hour over 24 steps, maintaining temporal continuity without training 24 separate models. |
| **Two-Layer Caching** | Prevents redundant upstream API hits and unnecessary re-training: 10-minute cache for shared live air quality/trends, and 30-minute cache for trained XGBoost models. |
| **US EPA AQI Standard** | Standardized, peer-reviewed categorization system across all 6 criteria pollutants with well-defined concentration breakpoints. |
| **Rule-Based Health Advisory** | Eliminates generative LLM hallucinations and latency for medical guidance, providing deterministic, category-specific recommendations verified against official health guidelines. |

---

## 11. Known Limitations

- **92-Day Training Window**: Training is bounded to 92 historical days to ensure sub-second API execution and avoid excessive payload sizes, which may under-sample multi-year seasonal patterns.
- **Heuristic Fallback for Sparse Data**: While Split Conformal Prediction provides distribution-free 90% coverage for standard operations, sparse datasets (<100 samples) fallback to residual variance heuristics.
- **PM2.5-Only Forecast Model**: Machine learning prediction is focused specifically on PM2.5 due to its primary health risk, while other pollutants are monitored via historical trends and current telemetry.
- **Grid-Based Model Data vs Ground Sensors**: Atmospheric telemetry is derived from spatial grid reanalysis models rather than hyper-local physical street monitors, which can smooth localized micro-climate spikes.

---

## 12. What I'd Add Next

- **OpenAQ Physical Ground-Station Ingestion**: Merge real-time ground-station monitoring feeds alongside satellite reanalysis for ground-truth bias correction.
- **Direct Multi-Quantile Loss Training**: Train dedicated pinball/quantile loss models (`reg:quantileerror`) natively in addition to split conformal prediction.
- **SHAP (SHapley Additive exPlanations)**: Provide interactive visual feature attribution explaining which atmospheric drivers (e.g. wind drop, rush hour) caused predicted spikes.
- **Multi-City Comparison**: Side-by-side dashboard comparing air quality across multiple user-selected locations simultaneously.
- **Automated Threshold Push Notifications**: Browser and webhook alerts when predicted PM2.5 is forecasted to exceed safe levels.

---

## 13. Author

**Aniket Mohanty**  
Registration No: `25BCE5816`  
*Tech Round 1 Recruitment Challenge, 2026*
   
 