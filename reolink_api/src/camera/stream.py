# Copyright (C) 2020-2025, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.


import logging
import threading
import time
from typing import Optional

# anonymizer streamer
# adjust the import path if you placed the file elsewhere
from anonymizer.anonymize_stream import (
    AnonymizingStreamer,
    DetectionSettings,
    EncoderSettings,
    StreamConfig,
)
from fastapi import APIRouter, HTTPException

from camera.config import FFMPEG_PARAMS, STREAMS
from camera.registry import CAMERA_REGISTRY
from camera.time_utils import seconds_since_last_command, update_command_time

router = APIRouter()

# We keep at most one active stream at a time
_streamers: dict[str, AnonymizingStreamer] = {}
_threads: dict[str, threading.Thread] = {}

# Optional defaults for output size and detection
DEFAULT_W = 640
DEFAULT_H = 360
DEFAULT_CONF = 0.30
DEFAULT_SCALE_DIV = 1


def _is_thread_alive(th: Optional[threading.Thread]) -> bool:
    return th is not None and th.is_alive()


def is_stream_running_for(camera_ip: str) -> bool:
    th = _threads.get(camera_ip)
    return _is_thread_alive(th)


def stop_any_running_stream() -> Optional[str]:
    """Stop whichever camera is currently streaming, return its id if any."""
    for cam_id, streamer in list(_streamers.items()):
        try:
            streamer.stop()
        except Exception as e:
            logging.warning("Error while stopping stream for %s: %s", cam_id, e)
        th = _threads.get(cam_id)
        try:
            if th and th.is_alive():
                th.join(timeout=2)
        except Exception:
            pass
        _streamers.pop(cam_id, None)
        _threads.pop(cam_id, None)
        return cam_id
    return None


def stop_stream_if_idle():
    """Background task that stops the stream if no command has arrived for 120 seconds."""
    while True:
        time.sleep(10)
        try:
            if seconds_since_last_command() > 120:
                stopped_cam = stop_any_running_stream()
                if stopped_cam:
                    logging.info("Stream for %s stopped due to inactivity", stopped_cam)
        except Exception as e:
            logging.warning("Idle stopper error: %s", e)


# Start the idle stopper once
_idle_guard = threading.Event()
if not _idle_guard.is_set():
    threading.Thread(target=stop_stream_if_idle, daemon=True, name="idle-stopper").start()
    _idle_guard.set()


def _build_streamer_for(camera_ip: str) -> AnonymizingStreamer:
    stream_info = STREAMS[camera_ip]
    input_url = stream_info["input_url"]
    output_url = stream_info["output_url"]

    w = stream_info.get("width", DEFAULT_W)
    h = stream_info.get("height", DEFAULT_H)

    transport = str(FFMPEG_PARAMS.get("rtsp_transport", "tcp"))
    low_delay = bool(FFMPEG_PARAMS.get("low_delay", True))
    discardcorrupt = bool(FFMPEG_PARAMS.get("discardcorrupt", True))
    analyzeduration = FFMPEG_PARAMS.get("analyzeduration", "1M")
    probesize = FFMPEG_PARAMS.get("probesize", "2M")

    # FFmpeg 4.3 uses stimeout
    stimeout_us = int(FFMPEG_PARAMS.get("stimeout_us", 5_000_000))

    # Use your configured encoder framerate as decoder CFR
    fps = int(FFMPEG_PARAMS.get("framerate", 10))

    stream_cfg = StreamConfig(
        rtsp_url=input_url,
        srt_out=output_url,
        width=w,
        height=h,
        rtsp_transport=transport,
        analyzeduration=analyzeduration,
        probesize=probesize,
        low_delay=low_delay,
        discardcorrupt=discardcorrupt,
        stimeout_us=stimeout_us,
        fps=fps,
    )

    keyint = int(FFMPEG_PARAMS.get("gop_size", 5))
    bitrate = str(FFMPEG_PARAMS.get("bitrate", "500k"))
    bufsize = str(FFMPEG_PARAMS.get("bufsize", "100k"))
    preset = str(FFMPEG_PARAMS.get("preset", "ultrafast"))
    tune = str(FFMPEG_PARAMS.get("tune", "zerolatency"))
    use_crf = bool(FFMPEG_PARAMS.get("use_crf", False))
    crf = int(FFMPEG_PARAMS.get("crf", 28))

    enc_cfg = EncoderSettings(
        keyint=keyint,
        use_crf=use_crf,
        crf=crf,
        bitrate=bitrate,
        bufsize=bufsize,
        preset=preset,
        tune=tune,
        threads=int(FFMPEG_PARAMS.get("threads", 1)),
    )

    det_cfg = DetectionSettings(
        conf_thres=float(stream_info.get("conf_thres", DEFAULT_CONF)),
        model_scale_div=int(stream_info.get("model_scale_div", DEFAULT_SCALE_DIV)),
    )

    return AnonymizingStreamer(stream_cfg, enc_cfg, det_cfg)


def _start_streamer_in_background(camera_ip: str, streamer: AnonymizingStreamer) -> None:
    """Run the streamer.start loop in a daemon thread."""

    def _runner():
        try:
            streamer.start()
        except Exception as e:
            logging.exception("Streamer for %s crashed: %s", camera_ip, e)
        finally:
            # clean up maps if the loop exits
            _streamers.pop(camera_ip, None)
            _threads.pop(camera_ip, None)

    th = threading.Thread(target=_runner, name=f"anonymizer-{camera_ip}", daemon=True)
    th.start()
    _threads[camera_ip] = th


@router.post("/start_stream/{camera_ip}")
def start_stream(camera_ip: str):
    """Start anonymized streaming for a given camera, unless it is already running."""
    update_command_time()

    if camera_ip not in STREAMS:
        raise HTTPException(status_code=404, detail=f"No stream config for camera {camera_ip}")

    if is_stream_running_for(camera_ip):
        logging.info("Stream for %s already running", camera_ip)
        return {"message": f"Stream for {camera_ip} already running"}

    stopped_cam = stop_any_running_stream()

    streamer = _build_streamer_for(camera_ip)
    _streamers[camera_ip] = streamer
    _start_streamer_in_background(camera_ip, streamer)

    logging.info("[%s] Anonymized stream started RTSP to SRT", camera_ip)
    return {"message": f"Stream started for {camera_ip}", "previous_stream": stopped_cam or "None"}


@router.post("/stop_stream")
def stop_stream():
    """Stop the active stream and reset zoom to position 0 if the camera supports it."""
    update_command_time()
    stopped_cam = stop_any_running_stream()
    if stopped_cam:
        cam = CAMERA_REGISTRY.get(stopped_cam)
        if cam:
            try:
                cam.start_zoom_focus(position=0)
                logging.info("[%s] Zoom reset to position 0 after stream stop", stopped_cam)
            except Exception as e:
                logging.warning("[%s] Failed to reset zoom: %s", stopped_cam, e)

        return {
            "message": f"Stream for {stopped_cam} stopped. Zoom reset if supported.",
            "camera_ip": stopped_cam,
        }

    return {"message": "No active stream was running"}


@router.get("/status")
def stream_status():
    """Return the list of camera ids that are currently streaming."""
    active_streams = [cam_ip for cam_ip, th in _threads.items() if _is_thread_alive(th)]
    if active_streams:
        return {"active_streams": active_streams}
    return {"message": "No stream is running"}


@router.get("/is_stream_running/{camera_ip}")
def is_stream_running(camera_ip: str):
    """Check if a specific camera is currently streaming."""
    return {"camera_ip": camera_ip, "running": is_stream_running_for(camera_ip)}
