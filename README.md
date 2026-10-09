# AeroTrack: Real-Time Air Quality Monitoring & ML Forecasting

> High-resolution atmospheric intelligence, global map telemetry, and on-demand XGBoost machine learning forecasts for any coordinate on Earth.

---

## 1. Overview

**AeroTrack** is an end-to-end environmental intelligence platform built in response to the **"Real-Time Air Quality Monitoring and Prediction for User-Selected Regions"** problem statement. Atmospheric pollutants such as fine particulate matter ($\text{PM}_{2.5}$) pose severe public health risks worldwide, yet accessing localized, actionable air quality data combined with dependable forward-looking projections remains challenging for general citizens and researchers alike.

AeroTrack bridges this gap by enabling users to either click any coordinate across an interactive global map or query any city or neighborhood via an autocomplete search bar. The system immediately aggregates multi-pollutant telemetry, maps readings to official standard thresholds, and calculates demographic-specific health advisories.

To anticipate air quality fluctuations, AeroTrack deploys an on-demand machine learning pipeline powered by **XGBoost**. When a location is queried, the backend dynamically fetches up to 92 days of hourly atmospheric and meteorological historical data, extracts temporal, lag, rolling, and interaction features, trains a localized gradient boosted regression model, and generates an autoregressive 24-hour $\text{PM}_{2.5}$ forecast complete with step-dependent uncertainty bounds.

---

## 2. Features

- **Interactive Leaflet World Map**: Click-to-query map interface supporting smooth panning, zooming, coordinate clamping, and custom location markers.
- **Place Search with Autocomplete**: Rate-limited, debounced search against OpenStreetMap's Nominatim engine with direct jump-to-location behavior.
- **Reverse Geocoding**: Automatic resolution of raw GPS coordinates to city, administrative subdivision, and country names via BigDataCloud.
- **Real-Time Pollutant Breakdown**: Instant telemetry cards for 6 primary criteria air pollutants:
  - Fine Particulate Matter ($\text{PM}_{2.5}$)
  - Coarse Particulate Matter ($\text{PM}_{10}$)
  - Nitrogen Dioxide ($\text{NO}_2$)
  - Sulphur Dioxide ($\text{SO}_2$)
  - Ground-Level Ozone ($\text{O}_3$)
  - Carbon Monoxide ($\text{CO}$)
- **US EPA AQI Rating Engine**: Breakpoint mapping adhering to US EPA standards, color-coded across all 6 official severity tiers.
- **7-Day Historical Trends**: Interactive time-series charts displaying 168+ hours of historical data with a metric toggle between Fine Dust $\text{PM}_{2.5}$ ($\mu\text{g/m}^3$) and US AQI.
- **24-Hour ML Forecast**: On-demand hourly $\text{PM}_{2.5}$ projections generated autoregressively by XGBoost, featuring heuristic uncertainty bands and plain-English summary alerts.
- **Dual-Layer Reference Lines**: Visual benchmarks displaying the **Safe Limit ($15\,\mu\text{g/m}^3$, WHO 2021 Guideline)** and **Caution Limit ($35\,\mu\text{g/m}^3$, US EPA Moderate Threshold)**.
- **Actionable Health Advisories**: Semantic, rule-based recommendations tailored for Outdoor Activities, Home Ventilation, Mask Usage, and Sensitive Demographic Groups.

---

## 3. Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         REACT FRONTEND (Vite + TS)                       │
│                                                                          │
│   ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────────┐   │
│   │ Leaflet Map View │  │ SearchBar Auto-  │  │ Pollutant Cards &    │   │
│   │ (Click listener) │  │ complete (OSM)   │  │ Health Advisories    │   │
│   └─────────┬────────┘  └────────┬─────────┘  └──────────┬───────────┘   │
│             │                    │                       │               │
│   ┌─────────┴────────────────────┴───────────────────────┴───────────┐   │
│   │ Recharts Visualizations: 7-Day TrendChart & 24h ForecastChart    │   │
│   └──────────────────────────────────┬───────────────────────────────┘   │
└──────────────────────────────────────┼───────────────────────────────────┘
                                       │ HTTP / REST (/api/*)
                                       ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                         FASTAPI BACKEND (Python 3.10+)                   │
│                                                                          │
│   ┌──────────────────────────────────────────────────────────────────┐   │
│   │ Routing & Middleware: CORS, Coordinate Validation, Rate Limiter  │   │
│   └───────────────┬──────────────────────────────────┬───────────────┘   │
│                   │                                  │                   │
│     ┌─────────────▼──────────────┐     ┌─────────────▼──────────────┐    │
│     │   Shared Air Cache (10m)   │     │   Model Cache (30m TTL)    │    │
│     │   (Current & 7-Day Trends) │     │   (XGBoost Model & Data)   │    │
│     └─────────────┬──────────────┘     └─────────────┬──────────────┘    │
│                   │                                  │                   │
│                   ▼                                  ▼                   │
│   ┌──────────────────────────────┐     ┌─────────────────────────────┐   │
│   │ External Upstream Services   │     │ ML Pipeline (ml_model.py)   │   │
│   │ • Open-Meteo Air Quality API │     │ • Feature Engineering       │   │
│   │ • Nominatim Geocoding        │     │ • XGBoost Regressor Fit     │   │
│   │ • BigDataCloud Reverse Geo   │     │ • Autoregressive 24h Loop   │   │
│   └──────────────────────────────┘     └─────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Machine Learning Pipeline

```
Raw Open-Meteo Data (92 days, hourly)
  │ (PM2.5, NO2, O3, Wind Speed, Temperature, Relative Humidity)
  ▼
Data Cleaning & Alignment
  │ (Missing value interpolation, bfill/ffill, timezone alignment)
  ▼
Feature Engineering Matrix (18 Engineered Features)
  ├── Cyclical Temporal Encodings: hour_sin, hour_cos, dow_sin, dow_cos
  ├── Autoregressive Lag Features: pm25_lag1, lag2, lag3, lag6, lag12, lag24
  ├── Polynomial Lag Terms: pm25_lag1_squared, pm25_lag24_squared
  ├── Rolling Statistics: rolling_mean_6, rolling_mean_24, rolling_std_24
  └── Meteorological Interactions: wind_pm25_ratio, humidity_temp_ratio, no2_o3_ratio
  ▼
Model Training
  │ Algorithm: XGBoost Regressor (120 trees, max depth 4, learning rate 0.08)
  │ Target: PM2.5 (t)
  ▼
Autoregressive 24-Step Forecasting Loop
  │ Step 1 -> Predict PM2.5(t+1) -> Update lag & rolling buffers -> Predict PM2.5(t+2) ...
  ▼
Uncertainty Band Calculation
  │ Uncertainty Margin = (Residual_Std * 1.645) * (1.0 + 0.03 * Step)
  │ Lower Bound = max(0, Prediction - Margin) | Upper Bound = Prediction + Margin
  ▼
Cache Output (30-minute in-memory TTL per rounded coordinate)
```

### Detailed Pipeline Stages

1. **Historical Data Acquisition**:
   The backend retrieves up to 92 days of hourly observations from Open-Meteo's Air Quality and Weather APIs, returning $\text{PM}_{2.5}$, $\text{NO}_2$, $\text{O}_3$, 10-meter wind speed, 2-meter air temperature, and 2-meter relative humidity. Timestamps are fetched with `timezone="auto"` and aligned to local solar time using `utc_offset_seconds`.
2. **Feature Engineering**:
   - **Cyclical Temporal Encodings**: Hour of the day ($0\text{--}23$) and day of the week ($0\text{--}6$) are mapped onto trigonometric coordinate circles using sine and cosine transformations:
     $$\text{hour\_sin} = \sin\left(\frac{2\pi \cdot \text{hour}}{24}\right), \quad \text{hour\_cos} = \cos\left(\frac{2\pi \cdot \text{hour}}{24}\right)$$
     $$\text{dow\_sin} = \sin\left(\frac{2\pi \cdot \text{dow}}{7}\right), \quad \text{dow\_cos} = \cos\left(\frac{2\pi \cdot \text{dow}}{7}\right)$$
   - **Autoregressive Lags**: Captures short-term persistence and diurnal seasonality ($\text{lag}_1, \text{lag}_2, \text{lag}_3, \text{lag}_6, \text{lag}_{12}, \text{lag}_{24}$).
   - **Non-Linear Polynomial Lags**: $\text{lag}_1^2$ and $\text{lag}_{24}^2$ assist the tree splits in capturing rapid pollution spikes and extreme compounding episodes.
   - **Rolling Window Statistics**: 6-hour and 24-hour moving averages and 24-hour rolling standard deviations (calculated strictly on shifted data to prevent lookahead data leakage).
   - **Meteorological Interactions**:
     - $\text{Wind-to-PM}_{2.5} \text{ Ratio} = \frac{\text{PM}_{2.5(t-1)}}{\text{WindSpeed} + 0.1}$ (stagnant vs. dispersion conditions)
     - $\text{Humidity-to-Temperature Ratio} = \frac{\text{Humidity}}{\text{Temp} + 40.0}$ (boundary layer moisture / inversion proxies)
     - $\text{NO}_2\text{-to-O}_3 \text{ Ratio} = \frac{\text{NO}_2}{\text{O}_3 + 1.0}$ (photochemical equilibrium proxy)
3. **Model Specifications**:
   - `n_estimators`: 120
   - `max_depth`: 4
   - `learning_rate`: 0.08
   - `subsample`: 0.85
   - `colsample_bytree`: 0.85
   - `random_state`: 42
4. **Autoregressive Forecasting**:
   For horizons $h = 1 \dots 24$, the model predicts $\widehat{y}_{t+h}$, appends the predicted value to the feature buffer, rolls the lag and rolling statistics forward, recomputes temporal features, and repeats for the subsequent step.
5. **Step-Dependent Uncertainty Bounds**:
   Uncertainty expands progressively over the forecast horizon to model error accumulation:
   $$\text{Margin}_h = \left(1.645 \cdot \sigma_{\text{residuals}}\right) \cdot (1.0 + 0.03 \cdot h)$$
   $$\text{Lower}_h = \max(0, \widehat{y}_{t+h} - \text{Margin}_h), \quad \text{Upper}_h = \widehat{y}_{t+h} + \text{Margin}_h$$
6. **Two-Tier Caching Strategy**:
   - **Live Air Quality & Trends Cache**: In-memory cache keyed by `(round(lat, 4), round(lon, 4))` with a **10-minute TTL** (`CACHE_TTL_AIR_SECONDS = 600`), avoiding repeated upstream Open-Meteo queries.
   - **ML Model & Forecast Cache**: In-memory cache with a **30-minute TTL** (`CACHE_TTL_SECONDS = 1800`), ensuring rapid interactive responses across repeated queries for the same region.

---

## 5. Why XGBoost?

1. **Sub-Second Training Speed**: In an on-demand architecture where models are trained for specific user-selected coordinates, neural network architectures (such as LSTMs or Transformers) impose significant latency and require GPU infrastructure. XGBoost fits 2,000+ hourly training rows in under 1.5 seconds on commodity CPUs.
2. **Empirical Dominance on Tabular Time-Series**: When working with tabular engineered features (lagged time steps, rolling aggregations, and ratios), gradient-boosted decision trees consistently outperform deep networks on sample sizes under 100,000 observations.
3. **Non-Linear Meteorological Interactions**: Atmospheric dynamics involve threshold behaviors (e.g., wind speeds above $5\,\text{m/s}$ rapidly dispersing particulate matter, or temperature inversions trapping pollutants near the surface). Decision trees capture these interactions naturally without requiring manual interaction terms.
4. **Missing Data Resilience**: XGBoost possesses native support for missing values, routing missing nodes through optimal default branch directions during inference.

---

## 6. AQI Classification Standard

AeroTrack locks all classification to the official **US Environmental Protection Agency (US EPA)** Air Quality Index standards as the single source of truth.

### US EPA Breakpoint Table

Concentrations are reported in **$\mu\text{g/m}^3$**:

| AQI Category | AQI Range | Color | $\text{PM}_{2.5}$ (24h) | $\text{PM}_{10}$ (24h) | $\text{O}_3$ (8h) | $\text{CO}$ (8h) | $\text{SO}_2$ (1h) | $\text{NO}_2$ (1h) |
|---|---|---|---|---|---|---|---|---|
| **Good** | 0 – 50 | `#10b981` (Green) | 0.0 – 9.0 | 0 – 54 | 0.0 – 107.9 | 0 – 5,039 | 0.0 – 91.7 | 0.0 – 99.6 |
| **Moderate** | 51 – 100 | `#eab308` (Yellow) | 9.1 – 35.4 | 55 – 154 | 108.0 – 137.2 | 5,040 – 10,763 | 91.8 – 196.5 | 99.7 – 188.0 |
| **Unhealthy for Sensitive Groups** | 101 – 150 | `#f97316` (Orange) | 35.5 – 55.4 | 155 – 254 | 137.3 – 166.6 | 10,764 – 14,198 | 196.6 – 484.7 | 188.1 – 676.8 |
| **Unhealthy** | 151 – 200 | `#ef4444` (Red) | 55.5 – 125.4 | 255 – 354 | 166.7 – 205.8 | 14,199 – 17,633 | 484.8 – 796.5 | 676.9 – 1,220.1 |
| **Very Unhealthy** | 201 – 300 | `#8b5cf6` (Purple) | 125.5 – 225.4 | 355 – 424 | 205.9 – 392.0 | 17,634 – 34,808 | 796.6 – 1,582.5 | 1,220.2 – 2,348.1 |
| **Hazardous** | 301+ | `#7f1d1d` (Maroon) | > 225.4 | > 424 | > 392.0 | > 34,808 | > 1,582.5 | > 2,348.1 |

### WHO 2021 Reference Line on Visualizations
In addition to the US EPA breakpoints, the 7-day trend chart and 24-hour forecast chart plot two visual reference thresholds:
- **Safe Limit ($15\,\mu\text{g/m}^3$)**: The World Health Organization (WHO) 2021 24-hour guideline for $\text{PM}_{2.5}$ concentration (rendered as a solid green line).
- **Caution Limit ($35\,\mu\text{g/m}^3$)**: The US EPA Moderate threshold where air quality begins impacting sensitive demographics (rendered as a solid orange line).

---

## 7. Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Frontend Framework** | React 18 + TypeScript | Component architecture, state management, strict typing |
| **Build Tooling** | Vite | Ultra-fast HMR and optimized production bundling |
| **Styling** | Tailwind CSS | Responsive utility-first dark UI design system |
| **Mapping** | Leaflet + React-Leaflet | Interactive tile map, coordinate capture, marker rendering |
| **Charts** | Recharts | Responsive SVGs for time-series trend lines and forecast confidence bands |
| **Icons** | Lucide React | Clean, lightweight iconography |
| **Backend Framework** | FastAPI (Python 3.10+) | High-throughput asynchronous REST API |
| **HTTP Client** | HTTPX | Asynchronous connection pooling for external upstream services |
| **ML Framework** | XGBoost + Scikit-Learn | Extreme Gradient Boosting regression and metric pipelines |
| **Data Processing** | Pandas + NumPy | Time-series alignment, rolling features, trigonometry encodings |
| **Geocoding & Place Search** | Nominatim (OpenStreetMap) | Debounced location autocomplete with 1.0s rate limiting |
| **Reverse Geocoding** | BigDataCloud Client API | Coordinate-to-municipality reverse resolution |
| **Atmospheric Data** | Open-Meteo Air Quality API | Global hourly atmospheric reanalysis and meteorological telemetry |

---

## 8. API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | API status manifest, health check, and endpoint discovery directory |
| `GET` | `/docs` | Interactive Swagger UI API documentation and testing workbench |
| `GET` | `/api/air-quality/{lat}/{lon}` | Current air quality metrics, US AQI score, and 6 pollutant concentration readings with ratings |
| `GET` | `/api/trends/{lat}/{lon}` | 7-day hourly historical and current pollutant trends ($\text{PM}_{2.5}$, $\text{PM}_{10}$, $\text{O}_3$, $\text{NO}_2$, AQI) |
| `GET` | `/api/predict/{lat}/{lon}` | Trains a localized XGBoost regressor and returns a 24-hour autoregressive $\text{PM}_{2.5}$ forecast |
| `GET` | `/api/reverse-geocode/{lat}/{lon}` | Resolves latitude and longitude coordinates into city, region, and country names |
| `GET` | `/api/search?q={query}` | Autocomplete search for global cities and landmarks with built-in rate-limiting and debouncing |

---

## 9. Running Locally

### Prerequisites
- **Python**: 3.10 or higher
- **Node.js**: 18.0 or higher
- **Package Managers**: `pip` and `npm`

---

### Backend Setup

#### On Windows (PowerShell)
```powershell
# Navigate to the backend directory
cd nexus\backend

# Create and activate a Python virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install required dependencies
pip install fastapi uvicorn httpx pandas numpy xgboost scikit-learn

# Run the FastAPI development server
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

#### On Linux / macOS (Bash)
```bash
# Navigate to the backend directory
cd nexus/backend

# Create and activate a Python virtual environment
python3 -m venv venv
source venv/bin/activate

# Install required dependencies
pip install fastapi uvicorn httpx pandas numpy xgboost scikit-learn

# Run the FastAPI development server
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

---

### Frontend Setup

#### On Windows (PowerShell)
```powershell
# Navigate to the frontend directory
cd nexus\frontend

# Install node dependencies
npm install

# Start the Vite development server
npm run dev
```

#### On Linux / macOS (Bash)
```bash
# Navigate to the frontend directory
cd nexus/frontend

# Install node dependencies
npm install

# Start the Vite development server
npm run dev
```

### Accessing the Application
- **Frontend Dashboard**: Open [http://localhost:5173](http://localhost:5173) in your browser.
- **Backend API & Swagger Docs**: Open [http://localhost:8000/docs](http://localhost:8000/docs).

---

## 10. Design Decisions

| Decision | Rationale |
|---|---|
| **Open-Meteo for Air Quality** | Provides free, global, hourly multi-pollutant reanalysis and forecast models without mandatory API keys, providing consistent global coverage. |
| **XGBoost for Forecasting** | Delivers superior tabular regression accuracy, handles non-linear atmospheric relationships, and trains within 1.5 seconds per query on CPU. |
| **Autoregressive Multi-Step Loop** | Allows the model to iteratively update lag and rolling features hour-by-hour, maintaining realistic temporal trajectories rather than fitting 24 separate regression heads. |
| **Two-Layer In-Memory Caching** | Prevents redundant upstream API hits and expensive retraining loops: 10-minute cache for live observation data and 30-minute cache for fitted ML models. |
| **US EPA AQI Standard as Truth** | Provides a standardized, peer-reviewed categorization system across all 6 criteria pollutants with clear breakpoint definitions. |
| **Client-Side Vite Proxy** | Proxies `/api` requests from `localhost:5173` to `127.0.0.1:8000`, eliminating cross-origin browser issues during local development. |

---

## 11. Known Limitations

- **92-Day Training Horizon**: Training data is restricted to 92 past days to maintain sub-second API response times and respect Open-Meteo payload sizes. This window may under-sample long-term multi-year inter-annual seasonality.
- **Heuristic Uncertainty Bands**: Uncertainty bounds expand via residual standard deviation multipliers rather than full quantile loss regression or Bayesian posterior sampling.
- **$\text{PM}_{2.5}$-Specific Machine Learning Model**: Machine learning forecasting is currently applied exclusively to $\text{PM}_{2.5}$ (the pollutant with the highest global health burden). Other pollutants are tracked via real-time and historical trends.
- **Grid-Based Reanalysis Data**: Atmospheric telemetry relies on spatial grid interpolation rather than direct on-premise physical monitoring stations, which may smooth extreme hyper-local street-canyon spikes.

---

## 12. Future Roadmap

- **OpenAQ Physical Station Integration**: Ingest real-time ground-station monitoring feeds alongside satellite reanalysis data for ground-truth validation.
- **Quantile Regression Loss**: Upgrade XGBoost training to use quantile regression (`reg:quantileerror` at $\alpha=0.10$ and $\alpha=0.90$) to calculate mathematically rigorous prediction intervals.
- **SHAP (SHapley Additive exPlanations)**: Provide interactive visual explanations of feature attribution (e.g., how wind speed vs. morning traffic peaks contributed to a forecast surge).
- **Multi-Location Comparison**: Side-by-side comparative views allowing users to benchmark air quality across multiple regions simultaneously.
- **Automated Threshold Push Notifications**: Webhook and browser notifications alerting users when predicted $\text{PM}_{2.5}$ exceeds safe thresholds.
