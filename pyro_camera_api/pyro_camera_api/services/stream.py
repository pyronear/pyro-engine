# pyro_camera_api/services/stream.py
# Copyright (C) 2022-2025, Pyronear.
# Licensed under the Apache License 2.0
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

from __future__ import annotations

import logging
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, cast

from pyro_camera_api.core.config import FFMPEG_PARAMS
from pyro_camera_api.services.anonymizer_rtsp import (
    AnonymizerWorker,
    BoxStore,
    EncoderWorker,
    LastFrameStore,
    RTSPDecoderWorker,
)
from pyro_camera_api.utils.time_utils import seconds_since_last_command

if TYPE_CHECKING:
    from fastapi import FastAPI

logger = logging.getLogger(__name__)


@dataclass
class Pipeline:
    """Simple container for a decoder plus encoder worker."""

    decoder: RTSPDecoderWorker
    encoder: EncoderWorker


# Optional global app reference for the idle stopper
_APP: Optional["FastAPI"] = None


def set_app_for_stream(app: "FastAPI") -> None:
    """
    Store the FastAPI app so that background helpers like stop_stream_if_idle
    can operate without a Request object.
    """
    global _APP
    _APP = app


def get_workers(app: "FastAPI") -> dict[str, Pipeline]:
    """Return the mapping camera_id -> Pipeline from app.state."""
    return cast(dict[str, Pipeline], getattr(app.state, "stream_workers", {}))


def get_processes(app: "FastAPI") -> dict[str, subprocess.Popen]:
    """Return the mapping camera_id -> ffmpeg process from app.state."""
    return cast(dict[str, subprocess.Popen], getattr(app.state, "stream_processes", {}))


def get_stores(app: "FastAPI") -> tuple[LastFrameStore, BoxStore, AnonymizerWorker]:
    """
    Return the shared frame and box stores plus the anonymizer worker
    from app.state.
    """
    frames = cast(LastFrameStore, app.state.frames)
    boxes = cast(BoxStore, app.state.boxes)
    anonym = cast(AnonymizerWorker, app.state.anonymizer)
    return frames, boxes, anonym


# ---------------------------------------------------------------------------
# Helpers for pipelines and ffmpeg processes
# ---------------------------------------------------------------------------


def is_thread_alive(obj: object) -> bool:
    """Return True if obj has a live _thread attribute."""
    try:
        thr = getattr(obj, "_thread", None)
        return isinstance(thr, threading.Thread) and thr.is_alive()
    except Exception:
        return False


def is_pipeline_running(p: Optional[Pipeline]) -> bool:
    """Return True if both decoder and encoder threads are alive."""
    if p is None:
        return False
    return is_thread_alive(p.decoder) and is_thread_alive(p.encoder)


def is_process_running(proc: Optional[subprocess.Popen]) -> bool:
    """Return True if an ffmpeg process is still running."""
    return bool(proc and proc.poll() is None)


def log_ffmpeg_output(proc: subprocess.Popen, camera_id: str) -> None:
    """Log ffmpeg stderr lines for a given camera."""
    if not proc.stderr:
        return
    for line in iter(proc.stderr.readline, b""):
        if not line:
            break
        try:
            logger.info("[ffmpeg %s] %s", camera_id, line.decode(errors="ignore").rstrip())
        except Exception:
            pass


def build_ffmpeg_restream_cmd(input_url: str, output_url: str) -> list[str]:
    """
    Build an ffmpeg command that restreams RTSP to SRT without decode in Python.

    Uses FFMPEG_PARAMS from config, with safe defaults if keys are missing.
    """
    params = FFMPEG_PARAMS if isinstance(FFMPEG_PARAMS, dict) else {}

    def get(key: str, default):
        return params.get(key, default)

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


# ---------------------------------------------------------------------------
# Stop logic used by both endpoints and idle stopper
# ---------------------------------------------------------------------------


def stop_any_running_stream(app: Optional["FastAPI"]) -> Optional[str]:
    """
    Stop one active stream, whether decoder plus encoder pipeline
    or a plain ffmpeg restream process.

    Priority is to stop pipelines first, then processes.
    Returns the camera_id that was stopped or None if nothing was running.
    """
    global _APP

    if app is None:
        app = _APP

    if app is None:
        return None

    workers = get_workers(app)
    for camera_id, pipeline in list(workers.items()):
        if is_pipeline_running(pipeline):
            try:
                pipeline.encoder.stop()
            except Exception as exc:
                logger.warning("Failed to stop encoder for %s, %s", camera_id, exc)
            try:
                pipeline.decoder.stop()
            except Exception as exc:
                logger.warning("Failed to stop decoder for %s, %s", camera_id, exc)
            workers.pop(camera_id, None)
            return camera_id

    procs = get_processes(app)
    for camera_id, proc in list(procs.items()):
        if is_process_running(proc):
            try:
                proc.terminate()
                try:
                    proc.wait(timeout=2)
                except Exception:
                    proc.kill()
            except Exception as exc:
                logger.warning("Failed to stop ffmpeg for %s, %s", camera_id, exc)
            procs.pop(camera_id, None)
            return camera_id

    return None


# ---------------------------------------------------------------------------
# Idle stopper thread entry point
# ---------------------------------------------------------------------------


def stop_stream_if_idle(
    check_interval_seconds: float = 10.0,
    idle_timeout_seconds: float = 120.0,
) -> None:
    """
    Background loop that checks last command time and stops the current stream
    if the system has been idle for more than idle_timeout_seconds.

    This expects set_app_for_stream(app) to have been called once at startup.
    """
    while True:
        time.sleep(check_interval_seconds)
        try:
            if seconds_since_last_command() > idle_timeout_seconds:
                stopped = stop_any_running_stream(app=None)
                if stopped:
                    logger.info("Stream for %s stopped due to inactivity", stopped)
        except Exception as exc:
            logger.warning("Idle stopper error, %s", exc)
