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
    Return 200 while the API answers and every running patrol makes progress.

    Returns 503 with the hung cameras when a patrol thread is stuck, which the
    compose healthcheck turns into a container restart.
    """
    stalled = stalled_patrols()
    if stalled:
        return JSONResponse(status_code=503, content={"status": "patrol_stalled", "cameras": stalled})
    return {"status": "ok"}
