# Copyright (C) 2022-2025, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.


import logging
import threading
import time
from typing import Optional

from anonymizer.rtsp_anonymize_srt import RTSPAnonymizeSRTWorker
from fastapi import APIRouter, HTTPException

from camera.config import STREAMS
from camera.registry import CAMERA_REGISTRY
from camera.time_utils import seconds_since_last_command, update_command_time

router = APIRouter()
workers: dict[str, RTSPAnonymizeSRTWorker] = {}


def is_worker_running(w: Optional[RTSPAnonymizeSRTWorker]) -> bool:
    if w is None:
        return False
    t = getattr(w, "_thread", None)
    return isinstance(t, threading.Thread) and t.is_alive()


def stop_any_running_stream() -> Optional[str]:
    for cam_id, w in list(workers.items()):
        if is_worker_running(w):
            try:
                w.stop()
            except Exception as e:
                logging.warning(f"Failed to stop worker for {cam_id}: {e}")
            del workers[cam_id]
            return cam_id
    return None


def stop_stream_if_idle():
    while True:
        time.sleep(10)
        if seconds_since_last_command() > 120:
            stopped = stop_any_running_stream()
            if stopped:
                logging.info(f"Stream for {stopped} stopped due to inactivity")


@router.post("/start_stream/{camera_ip}")
def start_stream(camera_ip: str):
    update_command_time()

    if camera_ip not in STREAMS:
        raise HTTPException(status_code=404, detail=f"No stream config for camera {camera_ip}")

    if camera_ip in workers and is_worker_running(workers[camera_ip]):
        logging.info(f"Stream for {camera_ip} already running")
        return {"message": f"Stream for {camera_ip} already running"}

    stopped_cam = stop_any_running_stream()

    stream_info = STREAMS[camera_ip]
    input_url = stream_info["input_url"]
    srt_out = stream_info["output_url"]  # use the prebuilt SRT URL including streamid

    # start worker
    worker = RTSPAnonymizeSRTWorker(
        rtsp_url=input_url,
        srt_out=srt_out,
    )

    workers[camera_ip] = worker
    logging.info(f"[{camera_ip}] start worker, rtsp {input_url}, srt {srt_out}")
    worker.start()
    return {"message": f"Stream started for {camera_ip}", "previous_stream": stopped_cam or "None"}


@router.post("/stop_stream")
def stop_stream():
    update_command_time()
    stopped_cam = stop_any_running_stream()
    if stopped_cam:
        cam = CAMERA_REGISTRY.get(stopped_cam)
        if cam:
            try:
                cam.start_zoom_focus(position=0)
                logging.info(f"[{stopped_cam}] Zoom reset to position 0 after stream stop")
            except Exception as e:
                logging.warning(f"[{stopped_cam}] Failed to reset zoom: {e}")
        return {"message": f"Stream for {stopped_cam} stopped. Zoom reset if supported.", "camera_ip": stopped_cam}
    return {"message": "No active stream was running"}


@router.get("/status")
def stream_status():
    active = [cam_ip for cam_ip, w in workers.items() if is_worker_running(w)]
    if active:
        return {"active_streams": active}
    return {"message": "No stream is running"}


@router.get("/is_stream_running/{camera_ip}")
def is_stream_running(camera_ip: str):
    w = workers.get(camera_ip)
    return {"camera_ip": camera_ip, "running": bool(w and is_worker_running(w))}
