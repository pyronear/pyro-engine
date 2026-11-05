# Copyright (C) 2022-2026, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

"""Exercise the HTTP contract across FastAPI and Starlette upgrades."""

import pytest
from fastapi.testclient import TestClient

from pyro_camera_api.main import app


@pytest.fixture
def client():
    # Without a context manager, TestClient leaves hardware/background workers off.
    client = TestClient(app)
    yield client
    client.close()


def test_health_and_cors(client):
    response = client.get("/health", headers={"Origin": "https://example.org"})
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert response.headers["access-control-allow-origin"] == "*"


def test_camera_query_validation(client):
    response = client.get("/cameras/capture")
    assert response.status_code == 422
    assert any(error["loc"] == ["query", "camera_ip"] for error in response.json()["detail"])


def test_unknown_camera_returns_json_error(client, monkeypatch):
    monkeypatch.setattr("pyro_camera_api.api.routes_cameras.CAMERA_REGISTRY", {})
    response = client.get("/cameras/capture", params={"camera_ip": "192.0.2.1"})
    assert response.status_code == 404
    assert isinstance(response.json()["detail"], str)


def test_openapi_preserves_camera_routes(client):
    response = client.get("/openapi.json")
    assert response.status_code == 200
    paths = response.json()["paths"]
    assert "get" in paths["/health"]
    assert "get" in paths["/cameras/capture"]
    assert "post" in paths["/control/click_to_move"]
