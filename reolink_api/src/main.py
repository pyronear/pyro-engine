import io
import logging
import threading
import time
from contextlib import asynccontextmanager
from typing import Dict, Optional, Tuple

import cv2
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from camera.registry import CAMERA_REGISTRY  # builds cameras from RAW_CONFIG

log = logging.getLogger("rtsp_service")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

CAPTURE_INTERVAL = 30.0
PER_CAMERA_DELAY = 0.2
MAX_FAILS_BEFORE_SKIP = 1
SKIP_DURATION = 120.0

# runtime state
last_images: Dict[str, Image.Image] = {}
last_meta: Dict[str, Dict[str, object]] = {}

stop_flag = threading.Event()
worker_thread: Optional[threading.Thread] = None


def _img_size(img) -> Tuple[int, int]:
    try:
        w, h = img.size
        return w, h
    except Exception:
        try:
            h, w = img.shape[:2]
            return w, h
        except Exception:
            return -1, -1


def sequential_loop():
    names = list(CAMERA_REGISTRY.keys())
    failure_count = {name: 0 for name in names}
    skip_until = {name: 0.0 for name in names}

    log.info("Starting sequential capture loop with registry")
    log.info("Cameras: %s", ", ".join(names))

    while not stop_flag.is_set():
        cycle_start = time.time()
        log.info("New cycle")

        for name in names:
            if stop_flag.is_set():
                break

            cam = CAMERA_REGISTRY[name]
            now = time.time()
            if now < skip_until[name]:
                left = skip_until[name] - now
                log.warning("[%s] skipped for %.0fs due to previous failure", name, left)
                continue

            t0 = time.time()
            log.info("[%s] capture attempt", name)

            img = None
            err: Optional[str] = None
            try:
                img = cam.capture()  # RTSPCamera uses internal five second timeout
            except Exception as e:
                err = str(e)

            dt = time.time() - t0
            meta = {"ts": time.time(), "dt": round(dt, 3), "ok": img is not None}

            if img is not None:
                w, h = _img_size(img)
                meta["size"] = (w, h)
                last_images[name] = img
                failure_count[name] = 0
                log.info("[%s] capture ok in %.2fs, size %dx%d", name, dt, w, h)
            else:
                failure_count[name] += 1
                if err:
                    meta["error"] = err
                    log.error("[%s] capture raised in %.2fs: %s", name, dt, err)
                else:
                    meta["error"] = "timeout_or_no_image"
                    log.error("[%s] timed out or failed in %.2fs", name, dt)

                if failure_count[name] >= MAX_FAILS_BEFORE_SKIP:
                    skip_until[name] = time.time() + SKIP_DURATION
                    log.error("[%s] will be skipped for %.0fs", name, SKIP_DURATION)

            last_meta[name] = meta

            if stop_flag.wait(PER_CAMERA_DELAY):
                break

        elapsed = time.time() - cycle_start
        remaining = max(0.0, CAPTURE_INTERVAL - elapsed)
        log.info("Cycle finished in %.2fs, sleep %.2fs", elapsed, remaining)
        if stop_flag.wait(remaining):
            break

    log.info("Sequential loop exited cleanly")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global worker_thread
    log.info("Service starting")
    worker_thread = threading.Thread(target=sequential_loop, daemon=True)
    worker_thread.start()
    try:
        yield
    finally:
        log.info("Service stopping")
        stop_flag.set()
        if worker_thread is not None:
            worker_thread.join(timeout=5.0)
        log.info("All stopped")


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {
        "status": "running",
        "cameras": list(CAMERA_REGISTRY.keys()),
        "interval_s": CAPTURE_INTERVAL,
        "skip_duration_s": SKIP_DURATION,
    }


@app.get("/status")
def status_all():
    return {name: last_meta.get(name, {"ok": False, "note": "no capture yet"}) for name in CAMERA_REGISTRY}


@app.get("/status/{name}")
def status_one(name: str):
    if name not in CAMERA_REGISTRY:
        raise HTTPException(404, f"Unknown camera {name}")
    return last_meta.get(name, {"ok": False, "note": "no capture yet"})


@app.get("/image/{name}")
def image_one(name: str):
    if name not in CAMERA_REGISTRY:
        raise HTTPException(404, f"Unknown camera {name}")
    img = last_images.get(name)
    if img is None:
        raise HTTPException(404, f"No image for camera {name}")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return Response(content=buf.getvalue(), media_type="image/jpeg")
