# Copyright (C) 2020-2025, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.


import logging
import subprocess
import threading
import time

from fastapi import APIRouter, HTTPException

from camera.config import FFMPEG_PARAMS, STREAMS
from camera.registry import CAMERA_REGISTRY
from camera.time_utils import seconds_since_last_command, update_command_time

router = APIRouter()
processes: dict[str, subprocess.Popen] = {}


def is_process_running(proc):
    """Check if a process is still running."""
    return proc and proc.poll() is None


def log_ffmpeg_output(proc, camera_id):
    """Reads and logs stderr from ffmpeg."""
    for line in proc.stderr:
        logging.error(f"[FFMPEG {camera_id}] {line.decode('utf-8').strip()}")


def stop_any_running_stream():
    """Stops any currently running stream."""
    for cam_id, proc in list(processes.items()):
        if is_process_running(proc):
            proc.terminate()
            proc.wait()
            del processes[cam_id]
            return cam_id
    return None


def stop_stream_if_idle():
    """Background task that stops the stream if no command is received for 120 seconds."""
    while True:
        time.sleep(10)
        if seconds_since_last_command() > 120:
            stopped_cam = stop_any_running_stream()
            if stopped_cam:
                logging.info(f"Stream for {stopped_cam} stopped due to inactivity")


@router.post("/start_stream/{camera_ip}")
def start_stream(camera_ip: str):
    """Start an FFmpeg stream for a given camera, unless it's already running."""
    update_command_time()
    if camera_ip not in STREAMS:
        raise HTTPException(status_code=404, detail=f"No stream config for camera {camera_ip}")

    if camera_ip in processes and is_process_running(processes[camera_ip]):
        logging.info(f"Stream for {camera_ip} already running")
        return {"message": f"Stream for {camera_ip} already running"}

    stopped_cam = stop_any_running_stream()

    stream_info = STREAMS[camera_ip]
    input_url = stream_info["input_url"]
    output_url = stream_info["output_url"]

    command = ["ffmpeg", "-hide_banner", "-loglevel", "warning"]
    if FFMPEG_PARAMS.get("discardcorrupt"):
        command += ["-fflags", "discardcorrupt+nobuffer"]
    if FFMPEG_PARAMS.get("low_delay"):
        command += ["-flags", "low_delay"]

    command += [
        "-rtsp_transport",
        FFMPEG_PARAMS["rtsp_transport"],
        "-i",
        input_url,
        "-c:v",
        FFMPEG_PARAMS["video_codec"],
        "-bf",
        str(FFMPEG_PARAMS["b_frames"]),
        "-g",
        str(FFMPEG_PARAMS["gop_size"]),
        "-b:v",
        FFMPEG_PARAMS["bitrate"],
        "-r",
        str(FFMPEG_PARAMS["framerate"]),
        "-preset",
        FFMPEG_PARAMS["preset"],
        "-tune",
        FFMPEG_PARAMS["tune"],
        "-flush_packets",
        "1",
    ]

    if FFMPEG_PARAMS.get("audio_disabled", False):
        command.append("-an")

    command += ["-f", FFMPEG_PARAMS["output_format"], output_url]

    logging.info(f"[{camera_ip}] Running ffmpeg command: {' '.join(command)}")
    proc = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    processes[camera_ip] = proc
    threading.Thread(target=log_ffmpeg_output, args=(proc, camera_ip), daemon=True).start()

    # Optional: reset zoom
    cam = CAMERA_REGISTRY.get(camera_ip)
    if cam:
        try:
            cam.start_zoom_focus(position=0)
            logging.info(f"[{camera_ip}] Zoom reset to 0")
        except Exception as e:
            logging.warning(f"[{camera_ip}] Failed to reset zoom: {e}")

    return {"message": f"Stream started for {camera_ip}", "previous_stream": stopped_cam or "None"}


@router.post("/stop_stream")
def stop_stream():
    """Stops any active stream and resets zoom to position 0 if camera exists."""
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
    """Returns which camera streams are currently running."""
    active_streams = [cam_ip for cam_ip, proc in processes.items() if is_process_running(proc)]
    if active_streams:
        return {"active_streams": active_streams}
    return {"message": "No stream is running"}


@router.get("/is_stream_running/{camera_ip}")
def is_stream_running(camera_ip: str):
    """Check if a specific camera is currently streaming."""
    proc = processes.get(camera_ip)
    if proc and is_process_running(proc):
        return {"camera_ip": camera_ip, "running": True}
    return {"camera_ip": camera_ip, "running": False}
