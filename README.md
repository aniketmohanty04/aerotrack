# AeroTrack: Real-Time Air Quality Monitoring & ML Forecasting

> High-resolution atmospheric intelligence, global map telemetry, and on-demand XGBoost machine learning forecasts for any coordinate on Earth.

---

## 🔗 Live Demo

- **Frontend:** https://aerotrack-three.vercel.app
- **Backend API:** https://aerotrack-backend-1tlt.onrender.com
- **API Documentation:** https://aerotrack-backend-1tlt.onrender.com/docs

> **Note:** The backend runs on Render's free tier, which normally sleeps after 15 minutes of inactivity. An UptimeRobot monitor pings the health endpoint every 5 minutes to keep it warm 24/7, so users experience sub-second response times. See the Uptime Optimization section below for details.

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
   - Fetches 92 days of hourly readings from Open-Meteo Air Quality and Weather APIs: $\text{PM}_{2.5}$, $\text{NO}_2$, $\text{O}_3$, 10-meter wind speed, 2-meter temperature, and 2-meter relative humidity.
   - Timestamps use `timezone="auto"` and are localized to local solar time using `utc_offset_seconds`.
2. **Feature Engineering**:
   - **Cyclical Time Encodings**: Hour of the day ($0\text{--}23$) and day of the week ($0\text{--}6$) transformed onto continuous trigonometric circles:
     $$\text{hour\_sin} = \sin\left(\frac{2\pi \cdot \text{hour}}{24}\right), \quad \text{hour\_cos} = \cos\left(\frac{2\pi \cdot \text{hour}}{24}\right)$$
     $$\text{dow\_sin} = \sin\left(\frac{2\pi \cdot \text{dow}}{7}\right), \quad \text{dow\_cos} = \cos\left(\frac{2\pi \cdot \text{dow}}{7}\right)$$
   - **Autoregressive Lag Features**: 1h, 2h, 3h, 6h, 12h, and 24h lag variables capturing immediate momentum and daily diurnal periodicity.
   - **Polynomial Lag Features**: $\text{lag}_1^2$ and $\text{lag}_{24}^2$ enabling non-linear splits during rapid pollutant spikes.
   - **Rolling Window Statistics**: 6-hour moving mean, 24-hour moving mean, and 24-hour moving standard deviation calculated on shifted values to prevent lookahead data leakage.
   - **Meteorological Interactions**:
     - $\text{wind\_pm25\_ratio} = \frac{\text{PM}_{2.5(t-1)}}{\text{WindSpeed} + 0.1}$ (stagnant vs. dispersion conditions)
     - $\text{humidity\_temp\_ratio} = \frac{\text{Humidity}}{\text{Temperature} + 40.0}$ (boundary layer inversion proxy)
     - $\text{no2\_o3\_ratio} = \frac{\text{NO}_2}{\text{O}_3 + 1.0}$ (photochemical equilibrium proxy)
3. **Model Configuration**:
   - **Algorithm**: `xgb.XGBRegressor`
   - **Estimators**: 120 trees
   - **Tree Depth**: `max_depth = 4` (prevents overfitting on small local sample sizes)
   - **Learning Rate**: `0.08`
   - **Subsample & Colsample**: `0.85` subsample, `0.85` colsample per tree
4. **Forecast Method (Autoregressive 24-Step Loop)**:
   - For each step $h \in [1, 24]$, the model generates an inference $\widehat{y}_{t+h}$.
   - The predicted value is appended to the feature array and becomes the new $\text{lag}_1$ input for step $h+1$. Lags, rolling statistics, and temporal encodings roll forward dynamically.
5. **Uncertainty Bands**:
   - Derives step-dependent uncertainty margins from recent historical residual variance:
     $$\text{Margin}_h = \left(1.645 \cdot \sigma_{\text{residuals}}\right) \cdot (1.0 + 0.03 \cdot h)$$
     $$\text{Lower}_h = \max(0, \widehat{y}_{t+h} - \text{Margin}_h), \quad \text{Upper}_h = \widehat{y}_{t+h} + \text{Margin}_h$$
6. **Two-Tier In-Memory Caching**:
   - **10-Minute Shared Air Cache**: Stores live telemetry and 7-day trends (`(round(lat, 4), round(lon, 4))`), eliminating duplicate requests between `/api/air-quality` and `/api/trends`.
   - **30-Minute Model Cache**: Retains trained XGBoost estimators and generated 24-hour forecasts per coordinate pair.

---

## 5. AQI Standard

AeroTrack locks all classification to the official **US Environmental Protection Agency (US EPA)** standards as the single source of truth across all 6 criteria pollutants.

### US EPA Breakpoint Table

All concentrations are measured in **$\mu\text{g/m}^3$**:

| AQI Category | AQI Range | Color | $\text{PM}_{2.5}$ (24h) | $\text{PM}_{10}$ (24h) | $\text{O}_3$ (8h) | $\text{CO}$ (8h) | $\text{SO}_2$ (1h) | $\text{NO}_2$ (1h) |
|---|---|---|---|---|---|---|---|---|
| **Good** | 0 – 50 | `#10b981` (Green) | 0.0 – 9.0 | 0 – 54 | 0.0 – 107.9 | 0 – 5,039 | 0.0 – 91.7 | 0.0 – 99.6 |
| **Moderate** | 51 – 100 | `#eab308` (Yellow) | 9.1 – 35.4 | 55 – 154 | 108.0 – 137.2 | 5,040 – 10,763 | 91.8 – 196.5 | 99.7 – 188.0 |
| **Unhealthy for Sensitive Groups** | 101 – 150 | `#f97316` (Orange) | 35.5 – 55.4 | 155 – 254 | 137.3 – 166.6 | 10,764 – 14,198 | 196.6 – 484.7 | 188.1 – 676.8 |
| **Unhealthy** | 151 – 200 | `#ef4444` (Red) | 55.5 – 125.4 | 255 – 354 | 166.7 – 205.8 | 14,199 – 17,633 | 484.8 – 796.5 | 676.9 – 1,220.1 |
| **Very Unhealthy** | 201 – 300 | `#8b5cf6` (Purple) | 125.5 – 225.4 | 355 – 424 | 205.9 – 392.0 | 17,634 – 34,808 | 796.6 – 1,582.5 | 1,220.2 – 2,348.1 |
| **Hazardous** | 301+ | `#7f1d1d` (Maroon) | > 225.4 | > 424 | > 392.0 | > 34,808 | > 1,582.5 | > 2,348.1 |

### WHO 2021 Reference Guideline
On both the 7-day trend chart and the 24-hour forecast chart, AeroTrack displays two constant visual benchmarks:
- **Safe Limit ($15\,\mu\text{g/m}^3$)**: The World Health Organization (WHO) 2021 recommended 24-hour guideline for $\text{PM}_{2.5}$ exposure.
- **Caution Limit ($35\,\mu\text{g/m}^3$)**: The US EPA threshold boundary separating Good/Moderate from Unhealthy for Sensitive Groups.

---

## 6. Tech Stack

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

## 7. API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` / `HEAD` | `/` | API status manifest, health overview, and endpoint discovery directory |
| `GET` / `HEAD` | `/api/health` | Dedicated lightweight uptime and health-check endpoint for cloud platforms |
| `GET` | `/api/air-quality/{lat}/{lon}` | Current pollutant telemetry, US AQI score, and individual pollutant ratings |
| `GET` | `/api/trends/{lat}/{lon}` | 7-day hourly historical and current pollutant trends ($\text{PM}_{2.5}$, $\text{PM}_{10}$, $\text{O}_3$, $\text{NO}_2$, AQI) |
| `GET` | `/api/predict/{lat}/{lon}` | Trains a localized XGBoost regressor and returns a 24-hour autoregressive $\text{PM}_{2.5}$ forecast |
| `GET` | `/api/search?q={query}` | Autocomplete search for global places with rate-limiting and debouncing |
| `GET` | `/api/reverse-geocode/{lat}/{lon}` | Resolves latitude/longitude coordinates into city, administrative region, and country |

---

## 8. Running Locally

### Prerequisites
- **Python**: 3.10 or higher
- **Node.js**: 18.0 or higher
- **Package Managers**: `pip` and `npm`

---

### Backend Setup

#### PowerShell (Windows)
```powershell
cd nexus\backend
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

#### Bash (Linux / macOS)
```bash
cd nexus/backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

---

### Frontend Setup

#### PowerShell (Windows)
```powershell
cd nexus\frontend
npm install
npm run dev
```

#### Bash (Linux / macOS)
```bash
cd nexus/frontend
npm install
npm run dev
```

- **Frontend App**: [http://localhost:5173](http://localhost:5173)
- **Backend API Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 9. Design Decisions

| Decision | Rationale |
|---|---|
| **Open-Meteo over AccuWeather** | Open-Meteo provides free, global, hourly multi-pollutant reanalysis and meteorological data without mandatory API key limits, enabling universal global coverage out of the box. |
| **XGBoost over LSTM / Deep Learning** | On-demand training for arbitrary coordinates requires sub-2-second latency. XGBoost trains in under 1.5 seconds on CPU, handles missing observations, and consistently outperforms LSTMs on small tabular time-series (~2,200 rows). |
| **Autoregressive Forecast** | Allows the model to dynamically update lag buffers and moving averages hour-by-hour over 24 steps, maintaining temporal continuity without training 24 separate models. |
| **Two-Layer Caching** | Prevents redundant upstream API hits and unnecessary re-training: 10-minute cache for shared live air quality/trends, and 30-minute cache for trained XGBoost models. |
| **US EPA AQI Standard** | Standardized, peer-reviewed categorization system across all 6 criteria pollutants with well-defined concentration breakpoints. |
| **Rule-Based Health Advisory** | Eliminates generative LLM hallucinations and latency for medical guidance, providing deterministic, category-specific recommendations verified against official health guidelines. |

---

## 10. Known Limitations

- **92-Day Training Window**: Training is bounded to 92 historical days to ensure sub-second API execution and avoid excessive payload sizes, which may under-sample multi-year seasonal patterns.
- **Heuristic Confidence Bands**: Uncertainty intervals expand via residual variance multipliers rather than full quantile loss regression (`reg:quantileerror`) or Bayesian posterior sampling.
- **$\text{PM}_{2.5}$-Only Forecast Model**: Machine learning prediction is focused specifically on $\text{PM}_{2.5}$ due to its primary health risk, while other pollutants are monitored via historical trends and current telemetry.
- **Grid-Based Model Data vs Ground Sensors**: Atmospheric telemetry is derived from spatial grid reanalysis models rather than hyper-local physical street monitors, which can smooth localized micro-climate spikes.

---

## 11. What I'd Add Next

- **OpenAQ Physical Ground-Station Ingestion**: Merge real-time ground-station monitoring feeds alongside satellite reanalysis for ground-truth bias correction.
- **Quantile Regression Intervals**: Upgrade XGBoost training to use quantile regression loss to produce mathematically rigorous $10^{\text{th}}$ and $90^{\text{th}}$ percentile bounds.
- **SHAP (SHapley Additive exPlanations)**: Provide interactive visual feature attribution explaining which atmospheric drivers (e.g. wind drop, rush hour) caused predicted spikes.
- **Multi-City Comparison**: Side-by-side dashboard comparing air quality across multiple user-selected locations simultaneously.
- **Automated Threshold Push Notifications**: Browser and webhook alerts when predicted $\text{PM}_{2.5}$ is forecasted to exceed safe levels.

---

## 12. Author

**Aniket Mohanty**  
Registration No: `25BCE5816`  
*Tech Round 1 Recruitment Challenge, 2026*
