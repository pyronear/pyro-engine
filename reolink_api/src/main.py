# Copyright (C) 2020-2025, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.


import logging
import threading
from contextlib import asynccontextmanager

from camera.capture import router as camera_capture_router
from camera.control import router as camera_control_router
from camera.focus import router as camera_focus_router
from camera.info import router as camera_info_router
from camera.patrol import patrol_loop, static_loop
from camera.patrol import router as camera_patrol_router
from camera.registry import CAMERA_REGISTRY, PATROL_FLAGS, PATROL_THREADS
from camera.stream import router as camera_stream_router
from camera.stream import stop_stream_if_idle
from fastapi import FastAPI


@asynccontextmanager
async def lifespan(app: FastAPI):
    for ip, cam in CAMERA_REGISTRY.items():
        if ip in PATROL_THREADS and PATROL_THREADS[ip].is_alive():
            continue

        stop_flag = threading.Event()

        if cam.cam_type == "ptz":
            thread = threading.Thread(
                target=patrol_loop,
                args=(ip, stop_flag),
                daemon=True,
            )
            logging.info(f"Starting patrol loop for PTZ camera {ip}")
        else:
            thread = threading.Thread(
                target=static_loop,
                args=(ip, stop_flag),
                daemon=True,
            )
            logging.info(f"Starting static loop for camera {ip}")

        PATROL_THREADS[ip] = thread
        PATROL_FLAGS[ip] = stop_flag
        thread.start()

    threading.Thread(target=stop_stream_if_idle, daemon=True).start()

    try:
        yield  # Startup complete
    finally:
        for ip, flag in PATROL_FLAGS.items():
            logging.info(f"Stopping loop for camera {ip}")
            flag.set()


# Initialize FastAPI app and camera registry
app = FastAPI(lifespan=lifespan)

app.include_router(camera_info_router, prefix="/info", tags=["Info"])
app.include_router(camera_capture_router, prefix="/capture", tags=["Capture"])
app.include_router(camera_control_router, prefix="/control", tags=["Control"])
app.include_router(camera_focus_router, prefix="/focus", tags=["Focus"])
app.include_router(camera_patrol_router, prefix="/patrol", tags=["Patrol"])
app.include_router(camera_stream_router, prefix="/stream", tags=["Stream"])
