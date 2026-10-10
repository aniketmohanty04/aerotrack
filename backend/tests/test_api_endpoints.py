import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)


def test_root_returns_json_manifest():
    """Verify GET / returns a JSON API manifest pointing to the Vercel frontend and docs."""
    response = client.get("/")
    assert response.status_code == 200
    assert "application/json" in response.headers.get("content-type", "")
    
    data = response.json()
    assert isinstance(data, dict)
    assert data["name"] == "AeroTrack API"
    assert data["status"] == "online"
    assert data["docs"] == "/docs"
    assert data["health"] == "/api/health"
    assert data["frontend"] == "https://aerotrack-three.vercel.app/"


def test_api_health_canonical_endpoint():
    """Verify GET /api/health returns the canonical health status payload."""
    response = client.get("/api/health")
    assert response.status_code == 200
    assert "application/json" in response.headers.get("content-type", "")
    
    data = response.json()
    assert data == {"status": "ok", "service": "aerotrack-backend"}

    # HEAD request support
    head_response = client.head("/api/health")
    assert head_response.status_code == 200


def test_docs_and_openapi_remain_accessible():
    """Verify FastAPI automatic interactive documentation remains accessible."""
    docs_response = client.get("/docs")
    assert docs_response.status_code == 200
    assert "text/html" in docs_response.headers.get("content-type", "")

    openapi_response = client.get("/openapi.json")
    assert openapi_response.status_code == 200
    openapi_data = openapi_response.json()
    assert "paths" in openapi_data
    assert "/" in openapi_data["paths"]
    assert "/api/health" in openapi_data["paths"]
    assert "/api/predict/{lat}/{lon}" in openapi_data["paths"]


def test_api_info_endpoint():
    """Verify /api/info returns endpoint index and production frontend link."""
    response = client.get("/api/info")
    assert response.status_code == 200
    data = response.json()
    assert data["frontend"] == "https://aerotrack-three.vercel.app/"
    assert "endpoints" in data
    assert data["endpoints"]["health"] == "/api/health"


def test_prediction_endpoint_unaffected():
    """Verify GET /api/predict/{lat}/{lon} remains unaffected and returns complete forecast schema."""
    response = client.get("/api/predict/28.6139/77.2090?allow_demo=true")
    assert response.status_code == 200
    
    data = response.json()
    assert data["status"] == "success"
    assert "forecast" in data
    assert len(data["forecast"]) == 24
    
    first_step = data["forecast"][0]
    assert "step" in first_step
    assert "predicted_pm25" in first_step
    assert "lower_bound" in first_step
    assert "upper_bound" in first_step
    assert "interval_method" in first_step
    assert first_step["lower_bound"] <= first_step["upper_bound"]


def test_cors_headers_restrict_unauthorized_origins():
    """Verify CORS restricts allowed origins to deployed frontend and development origins."""
    # Allowed origin
    res_allowed = client.options(
        "/api/health",
        headers={"Origin": "https://aerotrack-three.vercel.app", "Access-Control-Request-Method": "GET"}
    )
    assert res_allowed.headers.get("access-control-allow-origin") == "https://aerotrack-three.vercel.app"

    # Unauthorized arbitrary origin
    res_disallowed = client.options(
        "/api/health",
        headers={"Origin": "https://malicious-site.example.com", "Access-Control-Request-Method": "GET"}
    )
    assert res_disallowed.headers.get("access-control-allow-origin") != "https://malicious-site.example.com"
    assert res_disallowed.headers.get("access-control-allow-origin") != "*"


def test_reverse_geocode_coordinate_validation():
    """Verify coordinate bounds validation returns 400 for out-of-range inputs."""
    res_lat = client.get("/api/reverse-geocode/95.0/80.0")
    assert res_lat.status_code == 400
    assert "Invalid latitude" in res_lat.json()["detail"]

    res_lon = client.get("/api/reverse-geocode/13.0/185.0")
    assert res_lon.status_code == 400
    assert "Invalid longitude" in res_lon.json()["detail"]


def test_reverse_geocode_success_and_caching(monkeypatch):
    """Verify reverse geocode returns city, region, country and caches successful responses."""
    from backend import main

    # Clear cache before test
    main.REVERSE_GEOCODE_CACHE.clear()

    # Mock reverse_geocode upstream response
    async def mock_reverse(lat: float, lon: float):
        res = {"city": "Chennai", "region": "Tamil Nadu", "country": "India"}
        main.REVERSE_GEOCODE_CACHE[(round(lat, 4), round(lon, 4))] = (1000.0, res)
        return res

    monkeypatch.setattr(main, "reverse_geocode", mock_reverse)

    res = client.get("/api/reverse-geocode/13.0827/80.2707")
    assert res.status_code == 200
    data = res.json()
    assert data["city"] == "Chennai"
    assert data["region"] == "Tamil Nadu"
    assert data["country"] == "India"
    assert (13.0827, 80.2707) in main.REVERSE_GEOCODE_CACHE


def test_reverse_geocode_never_caches_failures(monkeypatch):
    """Verify reverse geocoding never caches all-null failures or transient network errors."""
    from backend import main
    main.REVERSE_GEOCODE_CACHE.clear()

    # Call with non-cached coordinates where upstream returns all null
    import httpx

    class MockAsyncClient:
        def __init__(self, *args, **kwargs):
            pass
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            pass
        async def get(self, url, *args, **kwargs):
            # Simulate upstream timeout
            raise httpx.TimeoutException("Upstream timeout")

    monkeypatch.setattr(main.httpx, "AsyncClient", MockAsyncClient)

    res = client.get("/api/reverse-geocode/0.0/0.0")
    assert res.status_code == 200
    data = res.json()
    assert data == {"city": None, "region": None, "country": None}

    # CRITICAL: (0.0, 0.0) MUST NOT be cached in REVERSE_GEOCODE_CACHE!
    assert (0.0, 0.0) not in main.REVERSE_GEOCODE_CACHE


def test_search_places_success_and_schema(monkeypatch):
    """Verify place search returns valid SearchResult schema for Chennai, London, Tokyo."""
    from backend import main

    sample_results = [
        {
            "display_name": "Chennai, Tamil Nadu, India",
            "lat": "13.0827",
            "lon": "80.2707",
            "type": "administrative"
        }
    ]

    class MockResponse:
        status_code = 200
        def json(self):
            return sample_results

    class MockAsyncClient:
        def __init__(self, *args, **kwargs):
            pass
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            pass
        async def get(self, url, *args, **kwargs):
            return MockResponse()

    monkeypatch.setattr(main.httpx, "AsyncClient", MockAsyncClient)
    main.SEARCH_CACHE.clear()

    res = client.get("/api/search?q=Chennai")
    assert res.status_code == 200
    data = res.json()
    assert len(data) == 1
    item = data[0]
    assert item["name"] == "Chennai"
    assert item["full_name"] == "Chennai, Tamil Nadu, India"
    assert item["latitude"] == pytest.approx(13.0827)
    assert item["longitude"] == pytest.approx(80.2707)
    assert item["type"] == "administrative"


def test_search_places_legitimate_empty_result(monkeypatch):
    """Verify genuinely nonexistent place returns 200 OK with empty array []."""
    from backend import main

    class MockResponse:
        status_code = 200
        def json(self):
            return []

    class MockAsyncClient:
        def __init__(self, *args, **kwargs):
            pass
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            pass
        async def get(self, url, *args, **kwargs):
            return MockResponse()

    monkeypatch.setattr(main.httpx, "AsyncClient", MockAsyncClient)
    main.SEARCH_CACHE.clear()

    res = client.get("/api/search?q=xyznonexistentplace123456")
    assert res.status_code == 200
    assert res.json() == []


def test_search_places_distinguishes_upstream_errors(monkeypatch):
    """Verify upstream timeouts and rate limits raise distinct HTTP error codes instead of false empty arrays."""
    from backend import main
    import httpx

    # 1. Timeout -> 504 Gateway Timeout
    class MockTimeoutClient:
        def __init__(self, *args, **kwargs):
            pass
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            pass
        async def get(self, url, *args, **kwargs):
            raise httpx.TimeoutException("Upstream timed out")

    monkeypatch.setattr(main.httpx, "AsyncClient", MockTimeoutClient)
    main.SEARCH_CACHE.clear()

    res_timeout = client.get("/api/search?q=London")
    assert res_timeout.status_code == 504
    assert "timed out" in res_timeout.json()["detail"]

    # 2. Rate limit 429 -> 429 Too Many Requests
    class Mock429Response:
        status_code = 429
        text = "Too Many Requests"
        def json(self):
            return {}

    class Mock429Client:
        def __init__(self, *args, **kwargs):
            pass
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            pass
        async def get(self, url, *args, **kwargs):
            return Mock429Response()

    monkeypatch.setattr(main.httpx, "AsyncClient", Mock429Client)
    main.SEARCH_CACHE.clear()

    res_429 = client.get("/api/search?q=Tokyo")
    assert res_429.status_code == 429
    assert "rate limit" in res_429.json()["detail"]


def test_calculate_pm25_to_us_aqi_breakpoints():
    """Verify EPA PM2.5 to US AQI piecewise linear interpolation across category breakpoints."""
    from backend.main import calculate_pm25_to_us_aqi

    # None and negative values
    assert calculate_pm25_to_us_aqi(None) is None
    assert calculate_pm25_to_us_aqi(-5.0) is None

    # Good: 0 - 9.0 µg/m³ -> 0 - 50 AQI
    assert calculate_pm25_to_us_aqi(0.0) == 0
    assert calculate_pm25_to_us_aqi(4.5) == 25
    assert calculate_pm25_to_us_aqi(9.0) == 50

    # Moderate: 9.1 - 35.4 µg/m³ -> 51 - 100 AQI
    assert calculate_pm25_to_us_aqi(9.1) == 51
    assert calculate_pm25_to_us_aqi(35.4) == 100

    # Unhealthy for Sensitive Groups: 35.5 - 55.4 µg/m³ -> 101 - 150 AQI
    assert calculate_pm25_to_us_aqi(35.5) == 101
    assert calculate_pm25_to_us_aqi(55.4) == 150

    # Unhealthy: 55.5 - 125.4 µg/m³ -> 151 - 200 AQI
    assert calculate_pm25_to_us_aqi(55.5) == 151
    assert calculate_pm25_to_us_aqi(125.4) == 200

    # Very Unhealthy: 125.5 - 225.4 µg/m³ -> 201 - 300 AQI
    assert calculate_pm25_to_us_aqi(125.5) == 201
    assert calculate_pm25_to_us_aqi(225.4) == 300

    # Hazardous: 225.5+
    assert calculate_pm25_to_us_aqi(250.0) > 300


def test_air_quality_computes_aqi_from_pm25_when_upstream_us_aqi_null(monkeypatch):
    """Verify that when Open-Meteo returns null for us_aqi, backend computes it from pm2_5."""
    from backend import main

    main.SHARED_AIR_CACHE.clear()

    mock_raw_data = {
        "current": {
            "time": "2026-10-10T00:00",
            "us_aqi": None,  # Upstream station does not provide USAQI directly
            "european_aqi": 30,
            "pm2_5": 36.7,
            "pm10": 46.2,
            "carbon_monoxide": 200.0,
            "nitrogen_dioxide": 15.0,
            "sulphur_dioxide": 3.0,
            "ozone": 25.0
        },
        "current_units": {
            "pm2_5": "µg/m³",
            "pm10": "µg/m³"
        },
        "timezone": "Asia/Kolkata"
    }

    async def mock_fetch_raw(lat, lon):
        return mock_raw_data

    monkeypatch.setattr(main, "_fetch_open_meteo_raw", mock_fetch_raw)

    res = client.get("/api/air-quality/19.2839/84.5044")
    assert res.status_code == 200
    data = res.json()

    # Verify us_aqi is NOT None and is computed from PM2.5 (36.7 µg/m³ -> ~104 US AQI)
    assert data["us_aqi"] is not None
    assert data["us_aqi"] == 104
    assert data["aqi_info"]["category"] == "Unhealthy for Sensitive Groups"
    assert data["aqi_info"]["level"] == "unhealthy_sensitive"


def test_air_quality_caps_at_500(monkeypatch):
    """Verify that extreme pollution levels (>500) are clamped at 500 with is_capped flag set."""
    from backend import main

    main.SHARED_AIR_CACHE.clear()

    mock_extreme_data = {
        "current": {
            "time": "2026-10-10T12:00",
            "us_aqi": 850,  # Open-Meteo extreme extrapolation
            "european_aqi": 620,
            "pm2_5": 780.0,
            "pm10": 1100.0,
            "carbon_monoxide": 5000.0,
            "nitrogen_dioxide": 200.0,
            "sulphur_dioxide": 150.0,
            "ozone": 80.0
        },
        "current_units": {},
        "timezone": "Asia/Kolkata"
    }

    async def mock_fetch_raw(lat, lon):
        return mock_extreme_data

    monkeypatch.setattr(main, "_fetch_open_meteo_raw", mock_fetch_raw)

    res = client.get("/api/air-quality/28.6139/77.2090")
    assert res.status_code == 200
    data = res.json()

    assert data["us_aqi"] == 500
    assert data["european_aqi"] == 500
    assert data["is_capped"] is True
    assert data["us_aqi_capped"] is True
    assert data["aqi_info"]["category"] == "Hazardous"


def test_admin_clear_cache_endpoint():
    """Verify GET /api/admin/clear-cache clears cache and returns status."""
    from backend import main

    main.SHARED_AIR_CACHE[(28.6139, 77.2090)] = {"timestamp": 123.0, "data": {}}
    assert len(main.SHARED_AIR_CACHE) > 0

    res = client.get("/api/admin/clear-cache")
    assert res.status_code == 200
    assert res.json() == {"status": "cleared", "shared_cache_size": 0}
    assert len(main.SHARED_AIR_CACHE) == 0



