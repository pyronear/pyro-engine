# Copyright (C) 2022-2025, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

from __future__ import annotations

import logging
import subprocess
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

from camera.config import FFMPEG_PARAMS, RAW_CONFIG, STREAMS  # FFMPEG_PARAMS may already exist in your config
from camera.registry import CAMERA_REGISTRY
from camera.time_utils import seconds_since_last_command, update_command_time

router = APIRouter()

# -----------------------------------------------------------------------------
# Shared getters wired from app.state by lifespan in main
# -----------------------------------------------------------------------------


@dataclass
class Pipeline:
    decoder: RTSPDecoderWorker
    encoder: EncoderWorker


# Optional global app ref for the idle stopper
_APP = None  # set by set_app_for_stream in main


def set_app_for_stream(app) -> None:
    global _APP
    _APP = app


def _workers(request: Request) -> dict[str, Pipeline]:
    return request.app.state.stream_workers


def _processes(request: Request) -> dict[str, subprocess.Popen]:
    return request.app.state.stream_processes


def _stores(request: Request) -> tuple[LastFrameStore, BoxStore, AnonymizerWorker]:
    return request.app.state.frames, request.app.state.boxes, request.app.state.anonymizer


# -----------------------------------------------------------------------------
# Helpers for pipelines and ffmpeg processes
# -----------------------------------------------------------------------------


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


def is_process_running(proc: Optional[subprocess.Popen]) -> bool:
    return bool(proc and proc.poll() is None)


def log_ffmpeg_output(proc: subprocess.Popen, camera_ip: str) -> None:
    if not proc.stderr:
        return
    for line in iter(proc.stderr.readline, b""):
        if not line:
            break
        try:
            logging.info("[ffmpeg %s] %s", camera_ip, line.decode(errors="ignore").rstrip())
        except Exception:
            pass


def build_ffmpeg_restream_cmd(input_url: str, output_url: str) -> list[str]:
    """
    Pass through restream, no decode in Python.
    Uses FFMPEG_PARAMS from config, with safe defaults if keys are missing.
    """
    params = FFMPEG_PARAMS if isinstance(FFMPEG_PARAMS, dict) else {}

    def get(k, d):
        return params.get(k, d)

    cmd: list[str] = ["ffmpeg", "-hide_banner", "-loglevel", "warning"]
    if get("discardcorrupt", True):
        cmd += ["-fflags", "discardcorrupt+nobuffer"]
    if get("low_delay", True):
        cmd += ["-flags", "low_delay"]

    cmd += [
        "-rtsp_transport",
        get("rtsp_transport", "tcp"),
        "-i",
        input_url,
        "-c:v",
        get("video_codec", "libx264"),
        "-bf",
        str(get("b_frames", 0)),
        "-g",
        str(get("gop_size", 14)),
        "-b:v",
        get("bitrate", "700k"),
        "-r",
        str(get("framerate", 10)),
        "-preset",
        get("preset", "veryfast"),
        "-tune",
        get("tune", "zerolatency"),
        "-flush_packets",
        "1",
    ]
    if get("audio_disabled", True):
        cmd.append("-an")

    cmd += ["-f", get("output_format", "mpegts"), output_url]
    return cmd


# -----------------------------------------------------------------------------
# Stop logic used by both the endpoint and the idle stopper
# -----------------------------------------------------------------------------


def stop_any_running_stream(request: Request | None) -> Optional[str]:
    """
    Stops one active stream, whether pipeline or ffmpeg process.
    Priority, stop pipelines first, then processes.
    Returns the camera_ip that was stopped or None.
    """
    if request is None and _APP is None:
        return None
    app = request.app if request is not None else _APP

    workers: dict[str, Pipeline] = app.state.stream_workers
    for cam_id, p in list(workers.items()):
        if is_pipeline_running(p):
            try:
                p.encoder.stop()
            except Exception as e:
                logging.warning("Failed to stop encoder for %s, %s", cam_id, e)
            try:
                p.decoder.stop()
            except Exception as e:
                logging.warning("Failed to stop decoder for %s, %s", cam_id, e)
            workers.pop(cam_id, None)
            return cam_id

    procs: dict[str, subprocess.Popen] = app.state.stream_processes
    for cam_id, proc in list(procs.items()):
        if is_process_running(proc):
            try:
                proc.terminate()
                try:
                    proc.wait(timeout=2)
                except Exception:
                    proc.kill()
            except Exception as e:
                logging.warning("Failed to stop ffmpeg for %s, %s", cam_id, e)
            procs.pop(cam_id, None)
            return cam_id

    return None


# -----------------------------------------------------------------------------
# Idle stopper thread entry
# -----------------------------------------------------------------------------


def stop_stream_if_idle():
    while True:
        time.sleep(10)
        try:
            if seconds_since_last_command() > 120:
                stopped = stop_any_running_stream(request=None)
                if stopped:
                    logging.info("Stream for %s stopped due to inactivity", stopped)
        except Exception as e:
            logging.warning("Idle stopper error, %s", e)


# -----------------------------------------------------------------------------
# API
# -----------------------------------------------------------------------------


@router.post("/start_stream/{camera_ip}")
def start_stream(camera_ip: str, request: Request):
    update_command_time()

    if camera_ip not in STREAMS:
        raise HTTPException(status_code=404, detail=f"No stream config for camera {camera_ip}")

    workers = _workers(request)
    procs = _processes(request)

    # idempotent if already running
    if is_pipeline_running(workers.get(camera_ip)) or is_process_running(procs.get(camera_ip)):
        logging.info("Stream for %s already running", camera_ip)
        return {"message": f"Stream for {camera_ip} already running"}

    # stop any other active stream to keep a single stream active at a time
    stopped_cam = stop_any_running_stream(request)

    cfg_stream = STREAMS[camera_ip]
    input_url: str = cfg_stream["input_url"]
    output_url: str = cfg_stream["output_url"]

    # read anonymizer flag from RAW_CONFIG, default False
    anonym_cfg = RAW_CONFIG.get(camera_ip, {})
    anonym_enabled: bool = bool(anonym_cfg.get("anonymizer", False))

    if anonym_enabled:
        frames, boxes, anonym = _stores(request)

        width: int = int(cfg_stream.get("width", 640))
        height: int = int(cfg_stream.get("height", 360))
        fps: int = int(cfg_stream.get("fps", 10))
        rtsp_transport: str = cfg_stream.get("rtsp_transport", "tcp")

        # update anonymizer threshold without restart
        try:
            anonym._conf = float(cfg_stream.get("conf", getattr(anonym, "_conf", 0.35)))  # type: ignore[attr-defined]
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
        logging.info("[%s] start pipeline, rtsp %s, srt %s", camera_ip, input_url, output_url)
        decoder.start()
        encoder.start()
        return {
            "message": f"Stream started for {camera_ip}",
            "previous_stream": stopped_cam or "None",
            "mode": "anonymizer",
        }
    cmd = build_ffmpeg_restream_cmd(input_url=input_url, output_url=output_url)
    logging.info("[%s] Running ffmpeg command, %s", camera_ip, " ".join(cmd))
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
    stopped_cam = stop_any_running_stream(request)
    if stopped_cam:
        cam = CAMERA_REGISTRY.get(stopped_cam)
        if cam:
            try:
                cam.start_zoom_focus(position=0)
                logging.info("[%s] Zoom reset to position 0 after stream stop", stopped_cam)
            except Exception as e:
                logging.warning("[%s] Failed to reset zoom, %s", stopped_cam, e)
        return {"message": f"Stream for {stopped_cam} stopped. Zoom reset if supported.", "camera_ip": stopped_cam}
    return {"message": "No active stream was running"}


@router.get("/status")
def stream_status(request: Request):
    workers = _workers(request)
    procs = _processes(request)
    active_workers = [cam_ip for cam_ip, p in workers.items() if is_pipeline_running(p)]
    active_procs = [cam_ip for cam_ip, pr in procs.items() if is_process_running(pr)]
    if active_workers or active_procs:
        return {"active_pipelines": active_workers, "active_ffmpeg": active_procs}
    return {"message": "No stream is running"}


@router.get("/is_stream_running/{camera_ip}")
def is_stream_running(camera_ip: str, request: Request):
    workers = _workers(request)
    procs = _processes(request)
    running = bool(is_pipeline_running(workers.get(camera_ip)) or is_process_running(procs.get(camera_ip)))
    return {"camera_ip": camera_ip, "running": running}
