# Copyright (C) 2022-2025, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

from __future__ import annotations

import logging
import subprocess
import threading

from fastapi import APIRouter, HTTPException, Request

from pyro_camera_api.camera.registry import CAMERA_REGISTRY
from pyro_camera_api.core.config import RAW_CONFIG, STREAMS
from pyro_camera_api.services.anonymizer_rtsp import EncoderWorker, RTSPDecoderWorker
from pyro_camera_api.services.stream import (
    Pipeline,
    build_ffmpeg_restream_cmd,
    get_processes,
    get_stores,
    get_workers,
    is_pipeline_running,
    is_process_running,
    log_ffmpeg_output,
    stop_any_running_stream,
)
from pyro_camera_api.utils.time_utils import update_command_time

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/start_stream/{camera_ip}")
def start_stream(camera_ip: str, request: Request):
    update_command_time()

    if camera_ip not in STREAMS:
        raise HTTPException(status_code=404, detail=f"No stream config for camera {camera_ip}")

    app = request.app
    workers = get_workers(app)
    procs = get_processes(app)

    # idempotent if already running
    if is_pipeline_running(workers.get(camera_ip)) or is_process_running(procs.get(camera_ip)):
        logger.info("Stream for %s already running", camera_ip)
        return {"message": f"Stream for {camera_ip} already running"}

    # stop any other active stream to keep a single stream active at a time
    stopped_cam = stop_any_running_stream(app)

    cfg_stream = STREAMS[camera_ip]
    input_url: str = cfg_stream["input_url"]
    output_url: str = cfg_stream["output_url"]

    # read anonymizer flag from RAW_CONFIG, default False
    anonym_cfg = RAW_CONFIG.get(camera_ip, {})
    anonym_enabled: bool = bool(anonym_cfg.get("anonymizer", False))

    if anonym_enabled:
        frames, boxes, anonym = get_stores(app)

        width: int = int(cfg_stream.get("width", 640))
        height: int = int(cfg_stream.get("height", 360))
        fps: int = int(cfg_stream.get("fps", 10))
        rtsp_transport: str = cfg_stream.get("rtsp_transport", "tcp")

        # update anonymizer threshold without restart
        try:
            anonym._conf = float(cfg_stream.get("conf", getattr(anonym, "_conf", 0.35)))
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
            srt_out=output_url,
            target_fps=fps,
        )

        workers[camera_ip] = Pipeline(decoder=decoder, encoder=encoder)
        logger.info("[%s] start pipeline, rtsp %s, srt %s", camera_ip, input_url, output_url)
        decoder.start()
        encoder.start()
        return {
            "message": f"Stream started for {camera_ip}",
            "previous_stream": stopped_cam or "None",
            "mode": "anonymizer",
        }

    # no anonymizer, plain restream
    cmd = build_ffmpeg_restream_cmd(input_url=input_url, output_url=output_url)
    logger.info("[%s] Running ffmpeg command, %s", camera_ip, " ".join(cmd))
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    procs[camera_ip] = proc
    threading.Thread(target=log_ffmpeg_output, args=(proc, camera_ip), daemon=True).start()
    return {
        "message": f"Stream started for {camera_ip}",
        "previous_stream": stopped_cam or "None",
        "mode": "restream",
    }


@router.post("/stop_stream")
def stop_stream(request: Request):
    update_command_time()
    app = request.app
    stopped_cam = stop_any_running_stream(app)
    if stopped_cam:
        cam = CAMERA_REGISTRY.get(stopped_cam)
        if cam:
            # only some backends implement start_zoom_focus, ignore errors for others
            try:
                cam.start_zoom_focus(position=0)  # type: ignore[attr-defined]
                logger.info("[%s] Zoom reset to position 0 after stream stop", stopped_cam)
            except Exception as exc:
                logger.warning("[%s] Failed to reset zoom, %s", stopped_cam, exc)
        return {"message": f"Stream for {stopped_cam} stopped. Zoom reset if supported.", "camera_ip": stopped_cam}
    return {"message": "No active stream was running"}


@router.get("/status")
def stream_status(request: Request):
    app = request.app
    workers = get_workers(app)
    procs = get_processes(app)
    active_workers = [cam_id for cam_id, p in workers.items() if is_pipeline_running(p)]
    active_procs = [cam_id for cam_id, pr in procs.items() if is_process_running(pr)]
    if active_workers or active_procs:
        return {"active_pipelines": active_workers, "active_ffmpeg": active_procs}
    return {"message": "No stream is running"}


@router.get("/is_stream_running/{camera_ip}")
def is_stream_running(camera_ip: str, request: Request):
    app = request.app
    workers = get_workers(app)
    procs = get_processes(app)
    running = bool(is_pipeline_running(workers.get(camera_ip)) or is_process_running(procs.get(camera_ip)))
    return {"camera_ip": camera_ip, "running": running}
