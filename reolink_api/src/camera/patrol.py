import logging
import threading
import time

from fastapi import APIRouter, HTTPException

from camera.registry import CAMERA_REGISTRY

router = APIRouter()

PATROL_THREADS = {}  # {camera_ip: threading.Thread}
PATROL_FLAGS = {}  # {camera_ip: threading.Event}


def patrol_loop(camera_ip: str, stop_flag: threading.Event):
    cam = CAMERA_REGISTRY[camera_ip]
    poses = cam.cam_poses or []

    if not poses:
        logging.warning(f"[{camera_ip}] No poses defined, exiting patrol loop")
        return

    logging.info(f"[{camera_ip}] Starting patrol cycle with {len(poses)} poses")

    while not stop_flag.is_set():
        start_time = time.time()

        for pose in poses:
            if stop_flag.is_set():
                break

            try:
                cam.move_camera("ToPos", idx=pose, speed=50)
                logging.info(f"[{camera_ip}] Moving to pose {pose}")
                time.sleep(1.5)  # Adjust based on real movement time

                image = cam.capture()
                if image:
                    cam.last_images[pose] = image
                    logging.info(f"[{camera_ip}] Stored image for pose {pose}")

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

    logging.info(f"[{camera_ip}] Patrol loop exited cleanly")


def static_loop(camera_ip: str, stop_flag: threading.Event):
    cam = CAMERA_REGISTRY[camera_ip]

    logging.info(f"[{camera_ip}] Starting static camera loop")

    while not stop_flag.is_set():
        try:
            image = cam.capture()
            if image:
                cam.last_images[-1] = image  # Store with fake pose -1
                logging.info(f"[{camera_ip}] Updated static image (pose -1)")
        except Exception as e:
            logging.error(f"[{camera_ip}] Error capturing static image: {e}")

        if stop_flag.wait(30):  # every 30 seconds
            break

    logging.info(f"[{camera_ip}] Static camera loop exited cleanly")


@router.post("/start_patrol")
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


@router.post("/stop_patrol")
def stop_patrol(camera_ip: str):
    if camera_ip not in PATROL_FLAGS:
        raise HTTPException(status_code=404, detail="No patrol running for this camera")

    PATROL_FLAGS[camera_ip].set()
    return {"status": "stopping", "camera_ip": camera_ip}


@router.get("/patrol_status")
def patrol_status(camera_ip: str):
    is_running = (
        camera_ip in PATROL_THREADS and PATROL_THREADS[camera_ip].is_alive() and not PATROL_FLAGS[camera_ip].is_set()
    )
    return {"camera_ip": camera_ip, "patrol_running": is_running}
