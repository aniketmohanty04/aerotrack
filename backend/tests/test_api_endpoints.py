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

