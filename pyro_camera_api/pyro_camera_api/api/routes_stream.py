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
    """
    Start live video streaming for a given camera.

    Two streaming modes are supported:
    - anonymizer mode: runs a decoder/encoder pipeline that overlays anonymized masks
    - restream mode: forwards video from RTSP to SRT via ffmpeg without anonymization

    Only one live stream is allowed across all cameras at a time.
    If a stream is already active for another camera, it is automatically stopped before starting the new stream.

    Returns:
        JSON response indicating the streaming mode used and whether another stream was stopped.
    """
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

    stopped_cam = stop_any_running_stream(app)

    cfg_stream = STREAMS[camera_ip]
    input_url: str = cfg_stream["input_url"]
    output_url: str = cfg_stream["output_url"]

    anonym_cfg = RAW_CONFIG.get(camera_ip, {})
    anonym_enabled: bool = bool(anonym_cfg.get("anonymizer", False))

    # anonymizer mode
    if anonym_enabled:
        frames, boxes, anonym = get_stores(app)

        width: int = int(cfg_stream.get("width", 640))
        height: int = int(cfg_stream.get("height", 360))
        fps: int = int(cfg_stream.get("fps", 10))
        rtsp_transport: str = cfg_stream.get("rtsp_transport", "tcp")

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

    # restream mode
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
    """
    Stop the currently running live stream, regardless of which camera is streaming.

    After stopping streaming, a best-effort zoom reset is applied to position 0
    for cameras that support zoom control. Failures during zoom reset are ignored.

    Returns:
        JSON confirmation and camera IP whose stream was stopped,
        or a message indicating that no stream was active.
    """
    update_command_time()
    app = request.app
    stopped_cam = stop_any_running_stream(app)
    if stopped_cam:
        cam = CAMERA_REGISTRY.get(stopped_cam)
        if cam:
            try:
                cam.start_zoom_focus(position=0)
                logger.info("[%s] Zoom reset to position 0 after stream stop", stopped_cam)
            except Exception as exc:
                logger.warning("[%s] Failed to reset zoom, %s", stopped_cam, exc)
        return {"message": f"Stream for {stopped_cam} stopped. Zoom reset if supported.", "camera_ip": stopped_cam}
    return {"message": "No active stream was running"}


@router.get("/status")
def stream_status(request: Request):
    """
    Return an overview of active streaming processes.

    A stream may be powered by:
        - a decoder/encoder pipeline (anonymizer mode)
        - an ffmpeg restream process (restream mode)

    Returns:
        Dict listing camera IDs with active anonymizer pipelines and/or FFmpeg processes,
        or a message if no stream is running.
    """
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
    """
    Check whether the given camera is currently streaming.

    Returns:
        JSON object containing the camera IP and a boolean flag indicating whether
        an anonymizer pipeline or an FFmpeg process is active for this camera.
    """
    app = request.app
    workers = get_workers(app)
    procs = get_processes(app)
    running = bool(is_pipeline_running(workers.get(camera_ip)) or is_process_running(procs.get(camera_ip)))
    return {"camera_ip": camera_ip, "running": running}
