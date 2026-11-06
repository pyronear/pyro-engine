# Copyright (C) 2022-2026, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

"""Check FastAPI/Starlette middleware and request validation after upgrades."""

from fastapi.testclient import TestClient

from pyro_camera_api.main import app


def test_http_dependency_compatibility():
    # Keep lifespan off so this check never starts camera/background workers.
    client = TestClient(app)
    try:
        response = client.get("/health", headers={"Origin": "https://example.org"})
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}
        assert response.headers["access-control-allow-origin"] == "*"
        assert client.get("/cameras/capture").status_code == 422
    finally:
        client.close()
