# minimal_static_api.py
# Copyright (C) 2025, Pyronear.
# Licensed under the Apache License 2.0

import logging
import threading
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from camera.registry import CAMERA_REGISTRY, PATROL_FLAGS, PATROL_THREADS

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s: %(message)s")

# ----------------------------------------------------------------------
# Static loop definition
# ----------------------------------------------------------------------

def static_loop(camera_ip: str, stop_flag: threading.Event):
    cam = CAMERA_REGISTRY[camera_ip]
    logging.info(f"[{camera_ip}] Starting static camera loop")
    settle_until = 0.0

    while not stop_flag.is_set():
        try:
            image = cam.capture()
            now = time.time()

            # After reconnect, wait a bit before saving
            if getattr(cam, "_opened_at", 0):
                settle_until = cam._opened_at + 1.0

            if image and now >= settle_until:
                print("image size", camera_ip, image.size)
                cam.last_images[-1] = image
                logging.info(f"[{camera_ip}] Updated static image (pose -1)")
        except Exception as e:
            logging.error(f"[{camera_ip}] Error capturing static image: {e}")

        # Wait 30s or exit if flag set
        if stop_flag.wait(30):
            break

    logging.info(f"[{camera_ip}] Static camera loop exited cleanly")


# ----------------------------------------------------------------------
# FastAPI app with lifespan
# ----------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.info("Starting minimal static loop service")

    # Start static loops
    for ip, cam in CAMERA_REGISTRY.items():
        stop_flag = threading.Event()
        thread = threading.Thread(target=static_loop, args=(ip, stop_flag), daemon=True)
        PATROL_THREADS[ip] = thread
        PATROL_FLAGS[ip] = stop_flag
        thread.start()
        logging.info(f"Started static loop for camera {ip}")

    yield

    # Stop all loops at shutdown
    for ip, flag in PATROL_FLAGS.items():
        logging.info(f"Stopping static loop for camera {ip}")
        flag.set()

    logging.info("All static loops stopped")


# ----------------------------------------------------------------------
# App initialization
# ----------------------------------------------------------------------

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------------------------------------------------
# Simple endpoints
# ----------------------------------------------------------------------

@app.get("/")
def root():
    return {"status": "running", "cameras": list(CAMERA_REGISTRY.keys())}


@app.get("/status/{camera_ip}")
def status(camera_ip: str):
    thread = PATROL_THREADS.get(camera_ip)
    flag = PATROL_FLAGS.get(camera_ip)
    running = thread and thread.is_alive() and flag and not flag.is_set()
    return {"camera_ip": camera_ip, "loop_running": bool(running)}


@app.post("/stop/{camera_ip}")
def stop(camera_ip: str):
    if camera_ip in PATROL_FLAGS:
        PATROL_FLAGS[camera_ip].set()
        return {"status": "stopping", "camera_ip": camera_ip}
    raise ValueError(f"No active loop for {camera_ip}")
