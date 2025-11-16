# pyro_camera_api/main.py
# Copyright (C) 2020-2025, Pyronear.
# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

from __future__ import annotations

import logging
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pyro_camera_api.api.routes_cameras import router as cameras_router
from pyro_camera_api.api.routes_control import router as control_router
from pyro_camera_api.api.routes_focus import router as focus_router
from pyro_camera_api.api.routes_health import router as health_router
from pyro_camera_api.api.routes_patrol import router as patrol_router
from pyro_camera_api.api.routes_stream import router as stream_router
from pyro_camera_api.camera.patrol import patrol_loop, static_loop
from pyro_camera_api.camera.registry import CAMERA_REGISTRY, PATROL_FLAGS, PATROL_THREADS
from pyro_camera_api.core.logging import setup_logging
from pyro_camera_api.services.anonymizer import AnonymizerWorker, BoxStore, LastFrameStore
from pyro_camera_api.services.stream import set_app_for_stream, stop_stream_if_idle

# Configure logging once at import time
setup_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # shared video state
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

    # registries for stream pipelines and processes
    if not hasattr(app.state, "stream_workers"):
        app.state.stream_workers = {}  # type: ignore[assignment]
    if not hasattr(app.state, "stream_processes"):
        app.state.stream_processes = {}  # type: ignore[assignment]

    # allow stream service to access app.state
    set_app_for_stream(app)

    # start patrol or static loops for each camera
    for cam_id, cam in CAMERA_REGISTRY.items():
        if cam_id in PATROL_THREADS and PATROL_THREADS[cam_id].is_alive():
            continue

        stop_flag = threading.Event()
        if getattr(cam, "cam_type", "static") == "ptz":
            thread = threading.Thread(target=patrol_loop, args=(cam_id, stop_flag), daemon=True)
            logger.info("Starting patrol loop for PTZ camera %s", cam_id)
        else:
            thread = threading.Thread(target=static_loop, args=(cam_id, stop_flag), daemon=True)
            logger.info("Starting static loop for camera %s", cam_id)

        PATROL_THREADS[cam_id] = thread
        PATROL_FLAGS[cam_id] = stop_flag
        thread.start()

    # idle auto stop thread for streams
    threading.Thread(target=stop_stream_if_idle, daemon=True).start()

    try:
        yield
    finally:
        # stop patrol loops
        for cam_id, flag in PATROL_FLAGS.items():
            logger.info("Stopping loop for camera %s", cam_id)
            flag.set()

        # stop pipelines
        try:
            workers = getattr(app.state, "stream_workers", {})
            for cam_id, p in list(workers.items()):
                try:
                    if hasattr(p, "encoder"):
                        p.encoder.stop()
                except Exception as exc:
                    logger.warning("Failed to stop encoder for %s, %s", cam_id, exc)
                try:
                    if hasattr(p, "decoder"):
                        p.decoder.stop()
                except Exception as exc:
                    logger.warning("Failed to stop decoder for %s, %s", cam_id, exc)
                workers.pop(cam_id, None)
        except Exception:
            pass

        # stop ffmpeg processes
        try:
            procs = getattr(app.state, "stream_processes", {})
            for cam_id, proc in list(procs.items()):
                try:
                    proc.terminate()
                    try:
                        proc.wait(timeout=2)
                    except Exception:
                        proc.kill()
                except Exception as exc:
                    logger.warning("Failed to stop ffmpeg for %s, %s", cam_id, exc)
                procs.pop(cam_id, None)
        except Exception:
            pass

        # stop anonymizer worker
        try:
            if hasattr(app.state, "anonymizer"):
                app.state.anonymizer.stop()
        except Exception:
            pass


app = FastAPI(lifespan=lifespan)

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(health_router, prefix="/health", tags=["Health"])
app.include_router(cameras_router, prefix="/capture", tags=["Capture"])
app.include_router(control_router, prefix="/control", tags=["Control"])
app.include_router(focus_router, prefix="/focus", tags=["Focus"])
app.include_router(patrol_router, prefix="/patrol", tags=["Patrol"])
app.include_router(stream_router, prefix="/stream", tags=["Stream"])
