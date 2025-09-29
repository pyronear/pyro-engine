# Copyright (C) 2020-2025, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.


import logging
import threading
from contextlib import asynccontextmanager

from anonymizer.rtsp_anonymize_srt import AnonymizerWorker, BoxStore, LastFrameStore
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
from fastapi.middleware.cors import CORSMiddleware


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Shared video state for the whole process
    if not hasattr(app.state, "frames"):
        app.state.frames = LastFrameStore()
    if not hasattr(app.state, "boxes"):
        app.state.boxes = BoxStore()
    if not hasattr(app.state, "anonymizer"):
        app.state.anonymizer = AnonymizerWorker(
            frame_store=app.state.frames,
            box_store=app.state.boxes,
            conf_thres=0.35,
        )
        app.state.anonymizer.start()

    # A registry for active stream pipelines
    if not hasattr(app.state, "stream_workers"):
        app.state.stream_workers = {}  # type: dict[str, object]

    # Your existing patrol threads bootstrap
    for ip, cam in CAMERA_REGISTRY.items():
        if ip in PATROL_THREADS and PATROL_THREADS[ip].is_alive():
            continue
        stop_flag = threading.Event()
        if cam.cam_type == "ptz":
            thread = threading.Thread(target=patrol_loop, args=(ip, stop_flag), daemon=True)
            logging.info(f"Starting patrol loop for PTZ camera {ip}")
        else:
            thread = threading.Thread(target=static_loop, args=(ip, stop_flag), daemon=True)
            logging.info(f"Starting static loop for camera {ip}")
        PATROL_THREADS[ip] = thread
        PATROL_FLAGS[ip] = stop_flag
        thread.start()

    # Idle auto stop thread, one per process
    threading.Thread(target=stop_stream_if_idle, daemon=True).start()

    try:
        yield
    finally:
        # Stop patrol loops
        for ip, flag in PATROL_FLAGS.items():
            logging.info(f"Stopping loop for camera {ip}")
            flag.set()

        # Stop any running pipelines
        try:
            workers = getattr(app.state, "stream_workers", {})
            for cam_id, p in list(workers.items()):
                try:
                    if hasattr(p, "encoder"):
                        p.encoder.stop()
                except Exception as e:
                    logging.warning(f"Failed to stop encoder for {cam_id}: {e}")
                try:
                    if hasattr(p, "decoder"):
                        p.decoder.stop()
                except Exception as e:
                    logging.warning(f"Failed to stop decoder for {cam_id}: {e}")
                workers.pop(cam_id, None)
        except Exception:
            pass

        # Stop anonymizer
        try:
            if hasattr(app.state, "anonymizer"):
                app.state.anonymizer.stop()
        except Exception:
            pass


# Initialize FastAPI app and camera registry
app = FastAPI(lifespan=lifespan)

# CORS: allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,  # must be False with "*"
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(camera_info_router, prefix="/info", tags=["Info"])
app.include_router(camera_capture_router, prefix="/capture", tags=["Capture"])
app.include_router(camera_control_router, prefix="/control", tags=["Control"])
app.include_router(camera_focus_router, prefix="/focus", tags=["Focus"])
app.include_router(camera_patrol_router, prefix="/patrol", tags=["Patrol"])
app.include_router(camera_stream_router, prefix="/stream", tags=["Stream"])
