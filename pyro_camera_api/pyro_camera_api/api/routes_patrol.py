# Copyright (C) 2022-2026, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.


from __future__ import annotations

import logging
import threading

from fastapi import APIRouter, HTTPException, Request

from pyro_camera_api.camera.patrol import (
    FAILURE_COUNT,
    SKIP_UNTIL,
    is_stream_running_for,
    patrol_loop,
    static_loop,
)
from pyro_camera_api.camera.registry import CAMERA_REGISTRY, PATROL_FLAGS, PATROL_THREADS

router = APIRouter()
logger = logging.getLogger(__name__)

# How long stop_patrol waits for the patrol thread to finish its in-flight camera call
STOP_JOIN_TIMEOUT_S = 10.0


@router.post("/start_patrol")
def start_patrol(camera_ip: str, request: Request):
    """
    Start automated patrol mode for a given camera.

    For PTZ cameras, the patrol loop cycles across configured preset positions.
    For static cameras, the static_loop simply triggers periodic captures.

    Patrol will not start if:
    - the camera is unknown
    - a stream is currently active on the same camera (409 Conflict)
    - a patrol thread is already running for the camera (idempotent response)

    Returns:
        JSON object containing status, camera IP, and patrol loop type.
    """
    cam = CAMERA_REGISTRY.get(camera_ip)
    if not cam:
        raise HTTPException(status_code=404, detail="Camera not found")

    # block if a stream is active for this camera
    if is_stream_running_for(request.app, camera_ip):
        raise HTTPException(
            status_code=409,
            detail=f"Stream is running for {camera_ip}, patrol cannot start",
        )

    # already running
    if camera_ip in PATROL_THREADS and PATROL_THREADS[camera_ip].is_alive():
        return {
            "status": "already_running",
            "camera_ip": camera_ip,
            "loop_type": getattr(PATROL_THREADS[camera_ip], "_target").__name__,  # ruff: ignore[get-attr-with-constant]
        }

    stop_flag = threading.Event()

    if getattr(cam, "cam_type", "static") == "ptz":
        target_fn = patrol_loop
        loop_type = "patrol"
    else:
        target_fn = static_loop
        loop_type = "static"

    thread = threading.Thread(
        target=target_fn,
        args=(camera_ip, stop_flag),
        daemon=True,
    )
    PATROL_THREADS[camera_ip] = thread
    PATROL_FLAGS[camera_ip] = stop_flag
    thread.start()

    logger.info("[%s] Started %s loop", camera_ip, loop_type)
    return {"status": "started", "camera_ip": camera_ip, "loop_type": loop_type}


@router.post("/stop_patrol")
def stop_patrol(camera_ip: str):
    """
    Stop the patrol loop for a camera and wait until it no longer drives the camera.

    The thread is not killed: it sees the stop flag, finishes its in-flight camera call and
    exits. Callers move the camera right after this returns, so it waits for the thread
    (up to STOP_JOIN_TIMEOUT_S) instead of letting a late patrol move race theirs.
    If no patrol is running for the camera, an HTTP 404 error is raised.

    Returns:
        JSON object with status "stopped", or "stopping" if the thread is still busy
        after the timeout.
    """
    if camera_ip not in PATROL_FLAGS:
        raise HTTPException(status_code=404, detail="No patrol running for this camera")

    PATROL_FLAGS[camera_ip].set()
    thread = PATROL_THREADS.get(camera_ip)
    if thread is not None:
        thread.join(STOP_JOIN_TIMEOUT_S)
        if thread.is_alive():
            logger.warning("[%s] Patrol thread still busy %.0fs after stop", camera_ip, STOP_JOIN_TIMEOUT_S)
            return {"status": "stopping", "camera_ip": camera_ip}
    return {"status": "stopped", "camera_ip": camera_ip}


@router.get("/patrol_status")
def patrol_status(camera_ip: str):
    """
    Get the patrol state and diagnostic counters for a camera.

    Returns information including:
        - whether patrol is currently running
        - type of patrol loop (static or preset patrol)
        - number of recent patrol failures
        - timestamp at which retry should resume (skip-until logic)

    This endpoint is safe to call periodically from UI dashboards.
    """
    thread = PATROL_THREADS.get(camera_ip)
    flag = PATROL_FLAGS.get(camera_ip)
    is_running = bool(thread and thread.is_alive() and flag and not flag.is_set())

    loop_type = None
    if thread is not None and hasattr(thread, "_target"):
        target_fn = getattr(thread, "_target").__name__  # ruff: ignore[get-attr-with-constant]
        if target_fn == "patrol_loop":
            loop_type = "patrol"
        elif target_fn == "static_loop":
            loop_type = "static"

    return {
        "camera_ip": camera_ip,
        "patrol_running": is_running,
        "loop_type": loop_type,
        "failures": FAILURE_COUNT.get(camera_ip, 0),
        "skip_until": int(SKIP_UNTIL.get(camera_ip, 0.0)),
    }
