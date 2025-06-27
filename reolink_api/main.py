import json
import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from io import BytesIO
from typing import Dict, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import Response
from reolink import ReolinkCamera  # your camera controller class

logging.basicConfig(level=logging.DEBUG)
# Load environment variables
CAM_USER = os.environ.get("CAM_USER")
CAM_PWD = os.environ.get("CAM_PWD")

if not CAM_USER or not CAM_PWD:
    raise RuntimeError("Environment variables CAM_USER and CAM_PWD must be set")

# Load camera configuration from JSON
CREDENTIALS_PATH = "/Users/mateo/pyronear/deploy/pyro-engine/data/credentials.json"
with open(CREDENTIALS_PATH) as f:
    raw_config = json.load(f)

# Define global registries
CAMERA_REGISTRY: Dict[str, ReolinkCamera] = {}
PATROL_THREADS = {}  # {camera_ip: threading.Thread}
PATROL_FLAGS = {}  # {camera_ip: threading.Event}

# Load cameras into registry
for ip, conf in raw_config.items():
    cam = ReolinkCamera(
        ip_address=ip,
        username=CAM_USER,
        password=CAM_PWD,
        cam_type=conf.get("type", "ptz"),
        cam_poses=conf.get("poses"),
        cam_azimuths=conf.get("azimuths"),
        focus_position=conf.get("focus_position", 720),
    )
    CAMERA_REGISTRY[ip] = cam


@asynccontextmanager
async def lifespan(app: FastAPI):
    monitored_ips = list(CAMERA_REGISTRY.keys())

    for ip in monitored_ips:
        if ip in PATROL_THREADS and PATROL_THREADS[ip].is_alive():
            continue

        stop_flag = threading.Event()
        thread = threading.Thread(
            target=patrol_loop,
            args=(ip, stop_flag),
            daemon=True,
        )
        PATROL_THREADS[ip] = thread
        PATROL_FLAGS[ip] = stop_flag
        thread.start()

    try:
        yield  # Startup done
    finally:
        for ip, flag in PATROL_FLAGS.items():
            flag.set()


# Initialize FastAPI app and camera registry
app = FastAPI(lifespan=lifespan)


# Helper to retrieve camera instance
def get_camera_by_ip(ip: str) -> ReolinkCamera:
    if ip not in CAMERA_REGISTRY:
        raise HTTPException(status_code=404, detail=f"Camera with IP '{ip}' not found")
    return CAMERA_REGISTRY[ip]


# Routes


@app.get("/cameras")
def list_cameras():
    return {"camera_ips": list(CAMERA_REGISTRY.keys())}


@app.get("/capture")
def capture(camera_ip: str, pos_id: Optional[int] = Query(default=None)):
    cam = get_camera_by_ip(camera_ip)
    img = cam.capture(pos_id=pos_id)
    if img is None:
        raise HTTPException(status_code=500, detail="Failed to capture image")
    buffer = BytesIO()
    img.save(buffer, format="JPEG")
    return Response(buffer.getvalue(), media_type="image/jpeg")


@app.post("/move")
def move_camera(camera_ip: str, operation: str, speed: int = 20, idx: int = 0):
    cam = get_camera_by_ip(camera_ip)
    cam.move_camera(operation=operation, speed=speed, idx=idx)
    return {"status": "ok", "camera_ip": camera_ip, "operation": operation}


@app.post("/move_in_seconds")
def move_in_seconds(camera_ip: str, seconds: float, operation: str = "Right", speed: int = 20):
    cam = get_camera_by_ip(camera_ip)
    cam.move_in_seconds(s=seconds, operation=operation, speed=speed)
    return {"status": "ok", "camera_ip": camera_ip}


@app.post("/focus/manual")
def manual_focus(camera_ip: str, position: int):
    cam = get_camera_by_ip(camera_ip)
    result = cam.set_manual_focus(position)
    return {"status": "manual_focus", "camera_ip": camera_ip, "position": position, "result": result}


@app.post("/focus/autofocus")
def toggle_autofocus(camera_ip: str, disable: bool = True):
    cam = get_camera_by_ip(camera_ip)
    result = cam.set_auto_focus(disable)
    return {"status": "autofocus", "camera_ip": camera_ip, "disabled": disable, "result": result}


@app.get("/focus/status")
def get_focus_status(camera_ip: str):
    cam = get_camera_by_ip(camera_ip)
    data = cam.get_focus_level()
    if not data:
        raise HTTPException(status_code=500, detail="Could not retrieve focus level")
    return {"camera_ip": camera_ip, "focus_data": data}


@app.get("/preset/list")
def list_presets(camera_ip: str):
    cam = get_camera_by_ip(camera_ip)
    presets = cam.get_ptz_preset()
    return {"camera_ip": camera_ip, "presets": presets}


@app.post("/preset/set")
def set_preset(camera_ip: str, idx: Optional[int] = None):
    cam = get_camera_by_ip(camera_ip)
    cam.set_ptz_preset(idx=idx)
    return {"status": "preset_set", "camera_ip": camera_ip, "id": idx}


@app.get("/latest_image")
def get_latest_image(camera_ip: str, pose: int):
    cam = get_camera_by_ip(camera_ip)

    if pose not in cam.last_images or cam.last_images[pose] is None:
        raise HTTPException(status_code=404, detail="No image available for this pose")

    buffer = BytesIO()
    cam.last_images[pose].save(buffer, format="JPEG")
    return Response(buffer.getvalue(), media_type="image/jpeg")


@app.post("/focus/focus_finder")
def run_focus_optimization(camera_ip: str, save_images: bool = False):
    cam = get_camera_by_ip(camera_ip)

    if cam.cam_type == "static":
        raise HTTPException(status_code=400, detail="Autofocus is not supported for static cameras")

    # Optional move to pose 1
    if cam.cam_poses and len(cam.cam_poses) > 1:
        pose1 = cam.cam_poses[1]
        cam.move_camera("ToPos", idx=pose1, speed=50)
        time.sleep(1)

    # Run autofocus
    best_position = cam.focus_finder(save_images=save_images)

    return {"camera_ip": camera_ip, "best_focus_position": best_position, "status": "focus_updated"}


@app.post("/start_patrol")
def start_patrol(camera_ip: str):
    if camera_ip in PATROL_THREADS and PATROL_THREADS[camera_ip].is_alive():
        return {"status": "already_running", "camera_ip": camera_ip}

    stop_flag = threading.Event()
    thread = threading.Thread(
        target=patrol_loop,
        args=(camera_ip, stop_flag),
        daemon=True,
    )
    PATROL_THREADS[camera_ip] = thread
    PATROL_FLAGS[camera_ip] = stop_flag
    thread.start()
    return {"status": "started", "camera_ip": camera_ip}


@app.post("/stop_patrol")
def stop_patrol(camera_ip: str):
    if camera_ip not in PATROL_FLAGS:
        raise HTTPException(status_code=404, detail="No patrol running for this camera")

    PATROL_FLAGS[camera_ip].set()
    return {"status": "stopping", "camera_ip": camera_ip}


def patrol_loop(camera_ip: str, stop_flag: threading.Event):
    cam = get_camera_by_ip(camera_ip)
    poses = cam.cam_poses or []

    if not poses:
        logging.warning(f"[{camera_ip}] No poses defined, exiting patrol loop")
        return

    print(f"[{camera_ip}] Starting patrol cycle with {len(poses)} poses")

    while not stop_flag.is_set():
        start_time = time.time()

        for pose in poses:
            if stop_flag.is_set():
                break

            try:
                cam.move_camera("ToPos", idx=pose, speed=50)
                logging.debug(f"[{camera_ip}] Moving to pose {pose}")
                time.sleep(1.5)  # Adjust based on real movement time

                image = cam.capture()
                if image:
                    cam.last_images[pose] = image
                    logging.debug(f"[{camera_ip}] Stored image for pose {pose}")

            except Exception as e:
                logging.error(f"[{camera_ip}] Error at pose {pose}: {e}")
                continue

        

        # To prevent big move get back to pose 0
        cam.move_camera("ToPos", idx=poses[0], speed=50)
        # Sleep to ensure the total loop is ~30s
        elapsed = time.time() - start_time
        sleep_time = max(0, 30 - elapsed)
        if stop_flag.wait(sleep_time):
            # To prevent big move get back to pose 0
            cam.move_camera("ToPos", idx=poses[0], speed=50)
            break

    print(f"[{camera_ip}] Patrol loop exited cleanly")

