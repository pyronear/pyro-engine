# Copyright (C) 2022-2026, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.


from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from pyro_camera_api.camera.patrol import stalled_patrols

router = APIRouter()


@router.get("/health")
def health():
    """
    Return a basic heartbeat response to confirm that the API is running.

    Kept free of camera checks: the Shelly watchdog reads it as Pi liveness and
    power-cycles the Pi and the router when it fails.
    """
    return {"status": "ok"}


@router.get("/health/patrols")
def health_patrols():
    """
    Return 200 while every running patrol makes progress, 503 with the hung cameras otherwise.

    Used by the compose healthcheck, which restarts the container on a 503.
    """
    stalled = stalled_patrols()
    if stalled:
        return JSONResponse(status_code=503, content={"status": "patrol_stalled", "cameras": stalled})
    return {"status": "ok"}
