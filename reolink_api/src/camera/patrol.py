# Copyright (C) 2020-2025, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.


import logging
import threading
import time

from fastapi import APIRouter, HTTPException

from camera.registry import CAMERA_REGISTRY, PATROL_FLAGS, PATROL_THREADS

router = APIRouter()


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

        # Return to pose 0 before sleep
        try:
            cam.move_camera("ToPos", idx=poses[0], speed=50)
            logging.info(f"[{camera_ip}] Returned to pose 0")
        except Exception as e:
            logging.warning(f"[{camera_ip}] Failed to return to pose 0: {e}")

        # Set focus if defined
        if getattr(cam, "focus_position", None) is not None:
            try:
                if cam.focus_position is not None:
                    cam.set_manual_focus(cam.focus_position)
                logging.info(f"[{camera_ip}] Restored manual focus to {cam.focus_position}")
            except Exception as e:
                logging.warning(f"[{camera_ip}] Failed to restore focus: {e}")

        # Sleep to ensure the total loop is ~30s
        elapsed = time.time() - start_time
        sleep_time = max(0, 30 - elapsed)
        stop_flag.wait(sleep_time)

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
    cam = CAMERA_REGISTRY.get(camera_ip)
    if not cam:
        raise HTTPException(status_code=404, detail="Camera not found")

    if camera_ip in PATROL_THREADS and PATROL_THREADS[camera_ip].is_alive():
        return {
            "status": "already_running",
            "camera_ip": camera_ip,
            "loop_type": PATROL_THREADS[camera_ip]._target.__name__,  # type: ignore[attr-defined]
        }

    stop_flag = threading.Event()

    if cam.cam_type == "ptz":
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

    logging.info(f"[{camera_ip}] 🚀 Started {loop_type} loop")
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
    is_running = thread and thread.is_alive() and flag and not flag.is_set()

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
    }
