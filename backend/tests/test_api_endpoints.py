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
