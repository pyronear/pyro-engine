# Copyright (C) 2022-2025, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import logging
import threading
import time
from dataclasses import dataclass
from typing import Optional

from anonymizer.rtsp_anonymize_srt import (
    AnonymizerWorker,
    BoxStore,
    EncoderWorker,
    LastFrameStore,
    RTSPDecoderWorker,
)
from fastapi import APIRouter, HTTPException, Request

from camera.config import STREAMS
from camera.registry import CAMERA_REGISTRY
from camera.time_utils import seconds_since_last_command, update_command_time

router = APIRouter()


@dataclass
class Pipeline:
    decoder: RTSPDecoderWorker
    encoder: EncoderWorker


def is_thread_alive(obj: object) -> bool:
    try:
        thr = getattr(obj, "_thread", None)
        return isinstance(thr, threading.Thread) and thr.is_alive()
    except Exception:
        return False


def is_pipeline_running(p: Optional[Pipeline]) -> bool:
    if p is None:
        return False
    return is_thread_alive(p.decoder) and is_thread_alive(p.encoder)


def _workers(request: Request) -> dict[str, Pipeline]:
    return request.app.state.stream_workers  # created in lifespan


def _stores(request: Request) -> tuple[LastFrameStore, BoxStore, AnonymizerWorker]:
    return request.app.state.frames, request.app.state.boxes, request.app.state.anonymizer


def stop_any_running_stream(request: Request) -> Optional[str]:
    workers = _workers(request)
    for cam_id, p in list(workers.items()):
        if is_pipeline_running(p):
            try:
                p.encoder.stop()
            except Exception as e:
                logging.warning(f"Failed to stop encoder for {cam_id}: {e}")
            try:
                p.decoder.stop()
            except Exception as e:
                logging.warning(f"Failed to stop decoder for {cam_id}: {e}")
            del workers[cam_id]
            return cam_id
    return None


def stop_stream_if_idle():
    while True:
        time.sleep(10)
        if seconds_since_last_command() > 120:
            # no Request here, import the global FastAPI app if you want to stop automatically,
            # or keep your existing background thread that calls the router stop endpoint
            pass


@router.post("/start_stream/{camera_ip}")
def start_stream(camera_ip: str, request: Request):
    update_command_time()

    if camera_ip not in STREAMS:
        raise HTTPException(status_code=404, detail=f"No stream config for camera {camera_ip}")

    workers = _workers(request)
    p = workers.get(camera_ip)
    if is_pipeline_running(p):
        logging.info(f"Stream for {camera_ip} already running")
        return {"message": f"Stream for {camera_ip} already running"}

    stopped_cam = stop_any_running_stream(request)

    frames, boxes, anonym = _stores(request)
    cfg = STREAMS[camera_ip]
    input_url: str = cfg["input_url"]
    srt_out: str = cfg["output_url"]
    width: int = int(cfg.get("width", 640))
    height: int = int(cfg.get("height", 360))
    fps: int = int(cfg.get("fps", 10))
    rtsp_transport: str = cfg.get("rtsp_transport", "tcp")

    # update anonymizer threshold without restart
    try:
        anonym._conf = float(cfg.get("conf", getattr(anonym, "_conf", 0.35)))  # type: ignore[attr-defined]
    except Exception:
        pass

    decoder = RTSPDecoderWorker(
        rtsp_url=input_url,
        width=width,
        height=height,
        fps=fps,
        rtsp_transport=rtsp_transport,
        store=frames,
    )
    encoder = EncoderWorker(
        frame_store=frames,
        box_store=boxes,
        width=width,
        height=height,
        srt_out=srt_out,
        target_fps=fps,
    )

    workers[camera_ip] = Pipeline(decoder=decoder, encoder=encoder)
    logging.info(f"[{camera_ip}] start pipeline, rtsp {input_url}, srt {srt_out}")
    decoder.start()
    encoder.start()
    return {"message": f"Stream started for {camera_ip}", "previous_stream": stopped_cam or "None"}


@router.post("/stop_stream")
def stop_stream(request: Request):
    update_command_time()
    stopped_cam = stop_any_running_stream(request)
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
def stream_status(request: Request):
    workers = _workers(request)
    active = [cam_ip for cam_ip, p in workers.items() if is_pipeline_running(p)]
    if active:
        return {"active_streams": active}
    return {"message": "No stream is running"}


@router.get("/is_stream_running/{camera_ip}")
def is_stream_running(camera_ip: str, request: Request):
    workers = _workers(request)
    p = workers.get(camera_ip)
    return {"camera_ip": camera_ip, "running": bool(is_pipeline_running(p))}
