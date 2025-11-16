# pyro_camera_api/services/stream.py

from __future__ import annotations

import logging
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Optional, cast

from anonymizer.rtsp_anonymize_srt import (
    AnonymizerWorker,
    BoxStore,
    EncoderWorker,
    LastFrameStore,
    RTSPDecoderWorker,
)
from fastapi import Request
from pyro_camera_api.core.config import FFMPEG_PARAMS
from pyro_camera_api.utils.time_utils import seconds_since_last_command

logger = logging.getLogger(__name__)


@dataclass
class Pipeline:
    decoder: RTSPDecoderWorker
    encoder: EncoderWorker


# Optional global app ref for the idle stopper
_APP = None  # set by set_app_for_stream in main


def set_app_for_stream(app) -> None:
    """
    Store the FastAPI app so the idle stopper can access
    stream workers and processes without a Request object.
    """
    global _APP
    _APP = app


def _workers(request: Request) -> dict[str, Pipeline]:
    return cast(dict[str, Pipeline], request.app.state.stream_workers)


def _processes(request: Request) -> dict[str, subprocess.Popen]:
    return cast(dict[str, subprocess.Popen], request.app.state.stream_processes)


def _stores(request: Request) -> tuple[LastFrameStore, BoxStore, AnonymizerWorker]:
    return (
        request.app.state.frames,
        request.app.state.boxes,
        request.app.state.anonymizer,
    )


def workers_from_app(app) -> dict[str, Pipeline]:
    return cast(dict[str, Pipeline], app.state.stream_workers)


def processes_from_app(app) -> dict[str, subprocess.Popen]:
    return cast(dict[str, subprocess.Popen], app.state.stream_processes)


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
            logger.info("[ffmpeg %s] %s", camera_ip, line.decode(errors="ignore").rstrip())
        except Exception:
            pass


def build_ffmpeg_restream_cmd(input_url: str, output_url: str) -> list[str]:
    """
    Build an ffmpeg command for pass through restreaming, no decode in Python.

    Uses FFMPEG_PARAMS from config, with safe defaults if keys are missing.
    """
    params = FFMPEG_PARAMS if isinstance(FFMPEG_PARAMS, dict) else {}

    def get(k, default):
        return params.get(k, default)

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


def stop_any_running_stream(request: Optional[Request]) -> Optional[str]:
    """
    Stops one active stream, whether pipeline or ffmpeg process.

    Priority is pipelines first then processes.

    Returns the camera id that was stopped or None.
    """
    app = None
    if request is not None:
        app = request.app
    else:
        app = _APP

    if app is None:
        return None

    workers = workers_from_app(app)
    for cam_id, p in list(workers.items()):
        if is_pipeline_running(p):
            try:
                p.encoder.stop()
            except Exception as exc:
                logger.warning("Failed to stop encoder for %s, %s", cam_id, exc)
            try:
                p.decoder.stop()
            except Exception as exc:
                logger.warning("Failed to stop decoder for %s, %s", cam_id, exc)
            workers.pop(cam_id, None)
            return cam_id

    procs = processes_from_app(app)
    for cam_id, proc in list(procs.items()):
        if is_process_running(proc):
            try:
                proc.terminate()
                try:
                    proc.wait(timeout=2)
                except Exception:
                    proc.kill()
            except Exception as exc:
                logger.warning("Failed to stop ffmpeg for %s, %s", cam_id, exc)
            procs.pop(cam_id, None)
            return cam_id

    return None


def stop_stream_if_idle() -> None:
    """
    Background loop that periodically checks the last command time
    and stops any running stream after a period of inactivity.
    """
    while True:
        time.sleep(10)
        try:
            if seconds_since_last_command() > 120:
                stopped = stop_any_running_stream(request=None)
                if stopped:
                    logger.info("Stream for %s stopped due to inactivity", stopped)
        except Exception as exc:
            logger.warning("Idle stopper error, %s", exc)


__all__ = [
    "Pipeline",
    "_processes",
    "_stores",
    "_workers",
    "build_ffmpeg_restream_cmd",
    "is_pipeline_running",
    "is_process_running",
    "is_thread_alive",
    "log_ffmpeg_output",
    "set_app_for_stream",
    "stop_any_running_stream",
    "stop_stream_if_idle",
]
