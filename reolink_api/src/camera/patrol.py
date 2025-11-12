# Copyright (C) 2020-2025, Pyronear.

from __future__ import annotations

import logging
import threading
import time

from fastapi import APIRouter, HTTPException, Request

from camera.registry import CAMERA_REGISTRY, PATROL_FLAGS, PATROL_THREADS

router = APIRouter()

# backoff settings for static cameras
MAX_FAILS_BEFORE_SKIP = 2            # skip after 2 consecutive failures
SKIP_DURATION = 30 * 60.0            # skip for 30 minutes

# per camera state for the static loop
FAILURE_COUNT: dict[str, int] = {}
SKIP_UNTIL: dict[str, float] = {}

def _is_thread_alive(obj: object) -> bool:
    try:
        thr = getattr(obj, "_thread", None)
        return isinstance(thr, threading.Thread) and thr.is_alive()
    except Exception:
        return False


def _is_stream_running_for(app, camera_ip: str) -> bool:
    """
    True if a pipeline or an ffmpeg restream is active for this camera.
    Expects app.state.stream_workers and app.state.stream_processes to be set in lifespan.
    """
    try:
        workers = getattr(app.state, "stream_workers", {})
        procs = getattr(app.state, "stream_processes", {})
    except Exception:
        return False

    # three worker pipeline
    p = workers.get(camera_ip)
    if p is not None:
        try:
            if _is_thread_alive(p.decoder) and _is_thread_alive(p.encoder):
                return True
        except Exception:
            pass

    # plain ffmpeg restream
    proc = procs.get(camera_ip)
    if proc is not None:
        try:
            if proc.poll() is None:
                return True
        except Exception:
            pass

    return False


def patrol_loop(camera_ip: str, stop_flag: threading.Event):
    cam = CAMERA_REGISTRY[camera_ip]
    poses = getattr(cam, "cam_poses", []) or []

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
                time.sleep(1.5)

                image = cam.capture()
                if image:
                    cam.last_images[pose] = image
                    logging.info(f"[{camera_ip}] Stored image for pose {pose}")

            except Exception as e:
                logging.error(f"[{camera_ip}] Error at pose {pose}: {e}")
                continue

        try:
            cam.move_camera("ToPos", idx=poses[0], speed=50)
            logging.info(f"[{camera_ip}] Returned to pose 0")
        except Exception as e:
            logging.warning(f"[{camera_ip}] Failed to return to pose 0: {e}")

        if getattr(cam, "focus_position", None) is not None:
            try:
                if cam.focus_position is not None:
                    cam.set_manual_focus(cam.focus_position)
                logging.info(f"[{camera_ip}] Restored manual focus to {cam.focus_position}")
            except Exception as e:
                logging.warning(f"[{camera_ip}] Failed to restore focus: {e}")

        elapsed = time.time() - start_time
        sleep_time = max(0, 30 - elapsed)
        stop_flag.wait(sleep_time)

    logging.info(f"[{camera_ip}] Patrol loop exited cleanly")


def static_loop(camera_ip: str, stop_flag: threading.Event):
    cam = CAMERA_REGISTRY[camera_ip]
    logging.info(f"[{camera_ip}] Starting static camera loop")

    # init per camera state
    FAILURE_COUNT.setdefault(camera_ip, 0)
    SKIP_UNTIL.setdefault(camera_ip, 0.0)

    settle_until = 0.0

    while not stop_flag.is_set():
        now = time.time()

        # skip window
        if now < SKIP_UNTIL[camera_ip]:
            left = int(SKIP_UNTIL[camera_ip] - now)
            logging.warning(f"[{camera_ip}] Skipped for {left}s due to previous failures")
        else:
            try:
                # capture with internal timeout handled by RTSPCamera
                image = cam.capture()
                now = time.time()

                # after a reconnect, wait a bit before storing
                if getattr(cam, "_opened_at", 0):
                    settle_until = cam._opened_at + 1.0

                if image and now >= settle_until:
                    cam.last_images[-1] = image
                    logging.info(f"[{camera_ip}] Updated static image (pose -1)")
                    # success, reset failure counter and clear skip
                    FAILURE_COUNT[camera_ip] = 0
                    SKIP_UNTIL[camera_ip] = 0.0
                else:
                    # failure or filtered store
                    FAILURE_COUNT[camera_ip] += 1
                    logging.error(f"[{camera_ip}] Capture returned no image, failures={FAILURE_COUNT[camera_ip]}")
                    if FAILURE_COUNT[camera_ip] >= MAX_FAILS_BEFORE_SKIP:
                        SKIP_UNTIL[camera_ip] = time.time() + SKIP_DURATION
                        logging.error(f"[{camera_ip}] Entering skip window for {int(SKIP_DURATION)}s")

            except Exception as e:
                # capture raised
                FAILURE_COUNT[camera_ip] += 1
                logging.error(f"[{camera_ip}] Error capturing static image: {e}, failures={FAILURE_COUNT[camera_ip]}")
                if FAILURE_COUNT[camera_ip] >= MAX_FAILS_BEFORE_SKIP:
                    SKIP_UNTIL[camera_ip] = time.time() + SKIP_DURATION
                    logging.error(f"[{camera_ip}] Entering skip window for {int(SKIP_DURATION)}s")

        # sleep 30 seconds or exit early
        if stop_flag.wait(30):
            break

    logging.info(f"[{camera_ip}] Static camera loop exited cleanly")


@router.post("/start_patrol")
def start_patrol(camera_ip: str, request: Request):
    cam = CAMERA_REGISTRY.get(camera_ip)
    if not cam:
        raise HTTPException(status_code=404, detail="Camera not found")

    # Block if a stream is active for this camera
    if _is_stream_running_for(request.app, camera_ip):
        raise HTTPException(
            status_code=409,
            detail=f"Stream is running for {camera_ip}, patrol cannot start",
        )

    # Already running
    if camera_ip in PATROL_THREADS and PATROL_THREADS[camera_ip].is_alive():
        return {
            "status": "already_running",
            "camera_ip": camera_ip,
            "loop_type": PATROL_THREADS[camera_ip]._target.__name__,  # type: ignore[attr-defined]
        }

    stop_flag = threading.Event()

    if getattr(cam, "cam_type", "static") == "ptz":
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

    logging.info(f"[{camera_ip}] Started {loop_type} loop")
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
        "patrol_running": bool(is_running),
        "loop_type": loop_type,
        "failures": FAILURE_COUNT.get(camera_ip, 0),
        "skip_until": int(SKIP_UNTIL.get(camera_ip, 0.0)),
    }
