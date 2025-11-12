# reolink_api/src/camera_api/app.py
import io
import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

# prefer TCP, ask for a 5 s socket timeout if the stack honors it
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|stimeout;5000000"

log = logging.getLogger("rtsp_service")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

# ---------------------------------------------------------
# Camera configuration
# ---------------------------------------------------------
CAMERAS: Dict[str, str] = {
    "Serre de Gruas": r"rtsp://DDSIS\pyronear:4kX<x64K+Pr4@srvcamera:554/live/E4CF7F9D-F85F-4ED6-AB56-E275181DD3EB",
    "Blandine": r"rtsp://DDSIS\pyronear:4kX<x64K+Pr4@srvcamera:554/live/1ECAC3E9-DB72-4CF3-8BD5-E55F4491356A",
    "Aubignas": r"rtsp://DDSIS\pyronear:4kX<x64K+Pr4@srvcamera:554/live/D2E6EC5F-5511-420B-A264-5B1447C6FF6F",
    "Pieds de Boeufs": r"rtsp://DDSIS\pyronear:4kX<x64K+Pr4@srvcamera:554/live/D4C8694C-964C-43BD-BD57-563E0E43C751",
    "Saint Jean Chambre": r"rtsp://DDSIS\pyronear:4kX<x64K+Pr4@srvcamera:554/live/6641704A-0873-40FE-82AE-22EC03AA4AA9",
    "Bidon": r"rtsp://DDSIS\pyronear:4kX<x64K+Pr4@srvcamera:554/live/14C4E0D6-E1D9-471D-802C-A903D91FE4C0",
    "La Forestiere": r"rtsp://DDSIS\pyronear:4kX<x64K+Pr4@srvcamera:554/live/3F8CD700-DFEE-401A-8445-CB9CB0DF3DFF",
    "Sampzon": r"rtsp://DDSIS\pyronear:4kX<x64K+Pr4@srvcamera:554/live/4E10857C-107B-465E-99B3-8E8F0DBCB3E7",
}

CAPTURE_INTERVAL = 30.0     # seconds between passes over all cameras
PER_CAMERA_DELAY = 0.2      # tiny delay between cameras
CAPTURE_TIMEOUT_S = 5.0     # hard timeout per camera read

# ---------------------------------------------------------
# State
# ---------------------------------------------------------
last_images: Dict[str, Image.Image] = {}
last_meta: Dict[str, Dict[str, object]] = {}  # per cam: {"ok": bool, "dt": float, "ts": float, "size": Tuple[int,int], "error": str}

stop_flag = threading.Event()
worker_thread: Optional[threading.Thread] = None


# ---------------------------------------------------------
# Capture helpers
# ---------------------------------------------------------
def _read_once(rtsp_url: str) -> Optional[np.ndarray]:
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        cap.release()
        return None
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return None
    return frame


def capture_with_timeout(rtsp_url: str, timeout_s: float) -> Tuple[Optional[np.ndarray], float, Optional[str]]:
    holder = {"frame": None, "err": None}

    def worker():
        try:
            holder["frame"] = _read_once(rtsp_url)
        except Exception as e:
            holder["err"] = str(e)

    t0 = time.time()
    t = threading.Thread(target=worker, daemon=True)
    t.start()
    t.join(timeout_s)
    dt = time.time() - t0

    if t.is_alive():
        return None, dt, "timeout"

    if holder["frame"] is None and holder["err"] is None:
        return None, dt, "read_failed"

    if holder["err"] is not None:
        return None, dt, holder["err"]

    return holder["frame"], dt, None


# ---------------------------------------------------------
# Background loop
# ---------------------------------------------------------
def sequential_loop():
    log.info("Starting RTSP sequential capture loop")
    log.info("Cameras: %s", ", ".join(CAMERAS.keys()))

    while not stop_flag.is_set():
        cycle_start = time.time()
        log.info("New cycle")

        for name, url in CAMERAS.items():
            if stop_flag.is_set():
                break

            log.info("[%s] capture attempt", name)
            frame, dt, err = capture_with_timeout(url, CAPTURE_TIMEOUT_S)

            meta = {"ts": time.time(), "dt": round(dt, 3), "ok": frame is not None}
            if frame is not None:
                h, w = frame.shape[:2]
                meta["size"] = (w, h)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                last_images[name] = Image.fromarray(rgb)
                log.info("[%s] capture ok in %.2fs, size %dx%d", name, dt, w, h)
            else:
                meta["error"] = err or "unknown"
                log.error("[%s] timed out or failed in %.2fs, reason=%s", name, dt, meta["error"])

            last_meta[name] = meta
            if stop_flag.wait(PER_CAMERA_DELAY):
                break

        elapsed = time.time() - cycle_start
        remaining = max(0.0, CAPTURE_INTERVAL - elapsed)
        log.info("Cycle finished in %.2fs, sleep %.2fs", elapsed, remaining)
        if stop_flag.wait(remaining):
            break

    log.info("Sequential loop exited cleanly")


# ---------------------------------------------------------
# FastAPI app with lifespan
# ---------------------------------------------------------
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


# ---------------------------------------------------------
# Endpoints
# ---------------------------------------------------------
@app.get("/")
def root():
    return {
        "status": "running",
        "cameras": list(CAMERAS.keys()),
        "interval_s": CAPTURE_INTERVAL,
        "timeout_s": CAPTURE_TIMEOUT_S,
    }


@app.get("/status")
def status_all():
    return {name: last_meta.get(name, {"ok": False, "note": "no capture yet"}) for name in CAMERAS}


@app.get("/status/{name}")
def status_one(name: str):
    if name not in CAMERAS:
        raise HTTPException(404, f"Unknown camera {name}")
    return last_meta.get(name, {"ok": False, "note": "no capture yet"})


@app.get("/image/{name}")
def image_one(name: str):
    if name not in CAMERAS:
        raise HTTPException(404, f"Unknown camera {name}")
    img = last_images.get(name)
    if img is None:
        raise HTTPException(404, f"No image for camera {name}")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return Response(content=buf.getvalue(), media_type="image/jpeg")
