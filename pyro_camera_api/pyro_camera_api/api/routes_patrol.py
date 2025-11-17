# Copyright (C) 2022-2025, Pyronear.

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


@router.post("/start_patrol")
def start_patrol(camera_ip: str, request: Request):
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
        # keep same behavior as your old code, raw target name
        return {
            "status": "already_running",
            "camera_ip": camera_ip,
            "loop_type": PATROL_THREADS[camera_ip]._target.__name__,  # type: ignore[attr-defined]
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
    if camera_ip not in PATROL_FLAGS:
        raise HTTPException(status_code=404, detail="No patrol running for this camera")

    PATROL_FLAGS[camera_ip].set()
    return {"status": "stopping", "camera_ip": camera_ip}


@router.get("/patrol_status")
def patrol_status(camera_ip: str):
    thread = PATROL_THREADS.get(camera_ip)
    flag = PATROL_FLAGS.get(camera_ip)
    is_running = bool(thread and thread.is_alive() and flag and not flag.is_set())

    loop_type = None
    if thread is not None and hasattr(thread, "_target"):
        target_fn = thread._target.__name__
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
