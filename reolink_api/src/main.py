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

CAPTURE_INTERVAL = 30.0  # target interval between two captures for each camera
PER_CAMERA_DELAY = 0.1   # small pause between cameras to avoid hammering

# ----------------------------------------------------------------------
# Static loop for all cameras, sequential capture
# ----------------------------------------------------------------------

def static_loop_all(stop_flag: threading.Event):
    logging.info("Starting static capture loop for all cameras")

    # one settle_until per camera
    settle_until = {ip: 0.0 for ip in CAMERA_REGISTRY}

    while not stop_flag.is_set():
        cycle_start = time.time()

        for ip, cam in CAMERA_REGISTRY.items():
            if stop_flag.is_set():
                break

            try:
                image = cam.capture()
                now = time.time()

                # After reconnect, wait a bit before saving
                if getattr(cam, "_opened_at", 0):
                    settle_until[ip] = cam._opened_at + 1.0

                if image and now >= settle_until[ip]:
                    print("image size", ip, image.size)
                    cam.last_images[-1] = image
                    logging.info(f"[{ip}] Updated static image (pose -1)")
            except Exception as e:
                logging.error(f"[{ip}] Error capturing static image: {e}")

            # short wait between cameras, and also early exit if stop_flag is set
            if stop_flag.wait(PER_CAMERA_DELAY):
                break

        # keep roughly CAPTURE_INTERVAL seconds between two passes
        elapsed = time.time() - cycle_start
        remaining = CAPTURE_INTERVAL - elapsed
        if remaining > 0 and stop_flag.wait(remaining):
            break

    logging.info("Static capture loop exited cleanly")


# single event controlling the whole loop
STATIC_STOP_FLAG = threading.Event()
STATIC_THREAD_KEY = "static_all"


# ----------------------------------------------------------------------
# FastAPI app with lifespan
# ----------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.info("Starting minimal static loop service")

    # For compatibility with existing code, register the same stop flag
    # under each camera ip in PATROL_FLAGS
    for ip in CAMERA_REGISTRY:
        PATROL_FLAGS[ip] = STATIC_STOP_FLAG

    # Start a single thread that loops over all cameras sequentially
    thread = threading.Thread(
        target=static_loop_all,
        args=(STATIC_STOP_FLAG,),
        daemon=True,
    )
    PATROL_THREADS[STATIC_THREAD_KEY] = thread
    thread.start()
    logging.info("Started single static loop for all cameras")

    try:
        yield
    finally:
        logging.info("Stopping static capture loop")
        STATIC_STOP_FLAG.set()
        thread.join(timeout=5.0)
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
    # status is now the same for every camera, controlled by the single worker thread
    thread = PATROL_THREADS.get(STATIC_THREAD_KEY)
    running = thread is not None and thread.is_alive() and not STATIC_STOP_FLAG.is_set()
    return {
        "camera_ip": camera_ip,
        "loop_running": bool(running),
    }


@app.post("/stop/{camera_ip}")
def stop(camera_ip: str):
    # stopping any camera stops the global static loop
    if camera_ip in PATROL_FLAGS:
        PATROL_FLAGS[camera_ip].set()
        return {
            "status": "stopping_all",
            "camera_ip": camera_ip,
        }
    raise ValueError(f"No active loop for {camera_ip}")
