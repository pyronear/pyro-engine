# Copyright (C) 2020-2025, Pyronear.
#
# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import logging
import subprocess
import threading
import time
from urllib.parse import quote

import av
import cv2
import numpy as np
from fastapi import APIRouter, HTTPException

from anonymizer.anonymizer_loop import anonymizer_loop
from anonymizer.anonymizer_registry import ANON_FLAGS, ANON_THREADS, set_result
from anonymizer.vision import Anonymizer
from camera.config import CAM_USER, CAM_PWD, FFMPEG_PARAMS, STREAMS
from camera.registry import CAMERA_REGISTRY
from camera.time_utils import seconds_since_last_command, update_command_time

router = APIRouter()
processes: dict[str, subprocess.Popen] = {}

# Blur pipeline state
BLUR_THREADS: dict[str, threading.Thread] = {}
BLUR_FLAGS: dict[str, threading.Event] = {}
BLUR_PROCS: dict[str, subprocess.Popen] = {}

ANON_MODEL = Anonymizer()


def is_process_running(proc):
    """Check if a process is still running."""
    return proc and proc.poll() is None


def log_ffmpeg_output(proc, camera_id):
    """Reads and logs stderr from ffmpeg."""
    for line in proc.stderr:
        logging.error(f"[FFMPEG {camera_id}] {line.decode('utf-8').strip()}")


def _rtsp_input_url(camera_ip: str) -> str:
    """Build RTSP substream URL from config credentials."""
    u = quote(str(CAM_USER), safe="")
    p = quote(str(CAM_PWD), safe="")
    return f"rtsp://{u}:{p}@{camera_ip}:554/h264Preview_01_sub"


def _open_pyav(rtsp_url: str):
    """Open PyAV container with low buffering."""
    opts = {
        "rtsp_transport": "tcp",
        "fflags": "nobuffer",
        "reorder_queue_size": "0",
        "analyzeduration": "0",
        "probesize": "32",
    }
    container = av.open(rtsp_url, options=opts)
    stream = next(s for s in container.streams if s.type == "video")
    frames = container.decode(video=stream.index)
    return container, frames


def _blur_boxes_bgr(img_bgr: np.ndarray, preds) -> np.ndarray:
    """Apply gaussian blur to predicted boxes."""
    h, w = img_bgr.shape[:2]
    for p in preds or []:
        box = p.get("box") if isinstance(p, dict) else None
        if not box or len(box) != 4:
            continue
        x1, y1, x2, y2 = box

        # allow normalized boxes
        if x2 <= 1.0 and y2 <= 1.0:
            x1 = int(max(0, min(w - 1, x1 * w)))
            x2 = int(max(0, min(w,     x2 * w)))
            y1 = int(max(0, min(h - 1, y1 * h)))
            y2 = int(max(0, min(h,     y2 * h)))
        else:
            x1 = int(max(0, min(w - 1, x1)))
            x2 = int(max(0, min(w,     x2)))
            y1 = int(max(0, min(h - 1, y1)))
            y2 = int(max(0, min(h,     y2)))

        if x2 <= x1 or y2 <= y1:
            continue

        roi = img_bgr[y1:y2, x1:x2]
        if roi.size == 0:
            continue

        kx = max(9, int((x2 - x1) * 0.1) | 1)
        ky = max(9, int((y2 - y1) * 0.1) | 1)
        img_bgr[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (kx, ky), 0)

    return img_bgr


def _start_ffmpeg_sink(width: int, height: int, output_url: str) -> subprocess.Popen:
    """
    Spawn FFmpeg that reads raw BGR frames on stdin and pushes to output_url.
    Uses params from FFMPEG_PARAMS.
    """
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "warning",

        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}",
        "-r", str(FFMPEG_PARAMS["framerate"]),
        "-i", "pipe:0",

        "-c:v", FFMPEG_PARAMS["video_codec"],
        "-preset", FFMPEG_PARAMS.get("preset", "veryfast"),
        "-tune", FFMPEG_PARAMS.get("tune", "zerolatency"),
        "-b:v", FFMPEG_PARAMS["bitrate"],
        "-g", str(FFMPEG_PARAMS["gop_size"]),
        "-bf", str(FFMPEG_PARAMS["b_frames"]),
        "-threads", str(FFMPEG_PARAMS.get("threads", 1)),

        "-muxpreload", "0",
        "-muxdelay", "0",
        "-flush_packets", "1",
    ]

    if FFMPEG_PARAMS.get("audio_disabled", True):
        cmd.append("-an")

    if output_url.startswith("rtsp://"):
        cmd += ["-rtsp_transport", FFMPEG_PARAMS["rtsp_transport"]]

    cmd += ["-f", FFMPEG_PARAMS["output_format"], output_url]

    logging.info(f"[sink] {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    threading.Thread(target=log_ffmpeg_output, args=(proc, "blurred"), daemon=True).start()
    return proc


def _blur_worker(camera_ip: str, stop_flag: threading.Event):
    """Read RTSP, run anonymizer, blur, write to FFmpeg stdin."""
    rtsp_url = _rtsp_input_url(camera_ip)
    output_url = STREAMS[camera_ip]["output_url"]

    container = None
    frames = None
    sink = None
    backoff = 0.5

    try:
        container, frames = _open_pyav(rtsp_url)
        first = next(frames, None)
        if first is None:
            raise RuntimeError("No frame from RTSP")
        img = first.to_ndarray(format="bgr24")
        h, w = img.shape[:2]

        sink = _start_ffmpeg_sink(w, h, output_url)
        BLUR_PROCS[camera_ip] = sink
        backoff = 0.5

        preds = ANON_MODEL(img)
        set_result(camera_ip, preds)
        img = _blur_boxes_bgr(img, preds)
        sink.stdin.write(img.tobytes())

        while not stop_flag.is_set() and sink.poll() is None:
            frame = next(frames, None)
            if frame is None:
                raise RuntimeError("Decoder returned no frame")
            img = frame.to_ndarray(format="bgr24")
            preds = ANON_MODEL(img)
            set_result(camera_ip, preds)
            img = _blur_boxes_bgr(img, preds)
            try:
                sink.stdin.write(img.tobytes())
            except BrokenPipeError:
                raise RuntimeError("FFmpeg sink closed")

        logging.info(f"[{camera_ip}] Blur worker stopping")

    except Exception as e:
        logging.error(f"[{camera_ip}] Blur worker error: {e}")

    finally:
        try:
            if container is not None:
                container.close()
        except Exception:
            pass
        try:
            if sink is not None:
                try:
                    sink.stdin.close()
                except Exception:
                    pass
                sink.terminate()
        except Exception:
            pass
        BLUR_PROCS.pop(camera_ip, None)


def stop_blur_for(camera_ip: str):
    """Stop blur pipeline for a camera if running."""
    flag = BLUR_FLAGS.get(camera_ip)
    thr = BLUR_THREADS.get(camera_ip)
    if flag:
        flag.set()
    if thr:
        thr.join(timeout=2.0)
    proc = BLUR_PROCS.get(camera_ip)
    if proc and is_process_running(proc):
        try:
            proc.terminate()
        except Exception:
            pass
    BLUR_FLAGS.pop(camera_ip, None)
    BLUR_THREADS.pop(camera_ip, None)
    BLUR_PROCS.pop(camera_ip, None)


def stop_any_running_stream():
    """Stops any running pass through or blur stream, also stops anonymizer."""
    # stop plain ffmpeg pass through first
    for cam_id, proc in list(processes.items()):
        if is_process_running(proc):
            proc.terminate()
            proc.wait()
            del processes[cam_id]
            stop_anonymizer(cam_id)
            return cam_id

    # stop a blur stream if present
    for cam_id in list(BLUR_THREADS.keys()):
        stop_blur_for(cam_id)
        stop_anonymizer(cam_id)
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
    """Start a pass through FFmpeg stream and start anonymizer loop."""
    update_command_time()
    if camera_ip not in STREAMS:
        raise HTTPException(status_code=404, detail=f"No stream config for camera {camera_ip}")

    # if a blur stream is running for this camera, stop it
    if camera_ip in BLUR_THREADS and BLUR_THREADS[camera_ip].is_alive():
        stop_blur_for(camera_ip)

    if camera_ip in processes and is_process_running(processes[camera_ip]):
        logging.info(f"Stream for {camera_ip} already running")
        if camera_ip not in ANON_THREADS or not ANON_THREADS[camera_ip].is_alive():
            stop_flag = threading.Event()
            thr = threading.Thread(target=anonymizer_loop, args=(camera_ip, stop_flag), daemon=True)
            ANON_FLAGS[camera_ip] = stop_flag
            ANON_THREADS[camera_ip] = thr
            thr.start()
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
        "-threads",
        str(FFMPEG_PARAMS.get("threads", 1)),
        "-muxpreload",
        "0",
        "-muxdelay",
        "0",
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

    # Start anonymizer loop for metadata only
    stop_flag = threading.Event()
    thr = threading.Thread(target=anonymizer_loop, args=(camera_ip, stop_flag), daemon=True)
    ANON_FLAGS[camera_ip] = stop_flag
    ANON_THREADS[camera_ip] = thr
    thr.start()

    return {"message": f"Stream started for {camera_ip}", "previous_stream": stopped_cam or "None"}


@router.post("/start_blur_stream/{camera_ip}")
def start_blur_stream(camera_ip: str):
    """Start RTSP read, run anonymizer, blur, and restream to configured output."""
    update_command_time()
    if camera_ip not in STREAMS:
        raise HTTPException(status_code=404, detail=f"No stream config for camera {camera_ip}"})

    # stop any running pass through for this camera
    if camera_ip in processes and is_process_running(processes[camera_ip]):
        try:
            processes[camera_ip].terminate()
            processes[camera_ip].wait()
        except Exception:
            pass
        processes.pop(camera_ip, None)
        stop_anonymizer(camera_ip)

    if camera_ip in BLUR_THREADS and BLUR_THREADS[camera_ip].is_alive():
        return {"message": f"Blur stream for {camera_ip} already running"}

    # optional: stop any other camera to keep one active
    stopped_cam = stop_any_running_stream()

    stop_flag = threading.Event()
    thr = threading.Thread(target=_blur_worker, args=(camera_ip, stop_flag), daemon=True)
    BLUR_FLAGS[camera_ip] = stop_flag
    BLUR_THREADS[camera_ip] = thr
    thr.start()

    return {"message": f"Blurred stream started for {camera_ip}", "previous_stream": stopped_cam or "None"}


@router.post("/stop_blur_stream/{camera_ip}")
def stop_blur_stream(camera_ip: str):
    """Stop the blurred stream for a given camera."""
    update_command_time()
    stop_blur_for(camera_ip)
    return {"message": f"Blurred stream stopped for {camera_ip}"}


@router.post("/stop_stream")
def stop_stream():
    """Stop any active stream and reset zoom, also stop anonymizer."""
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
    """Return which camera streams are currently running."""
    active_plain = [cam_ip for cam_ip, proc in processes.items() if is_process_running(proc)]
    active_blur = [cam_ip for cam_ip, thr in BLUR_THREADS.items() if thr.is_alive()]
    if active_plain or active_blur:
        return {"active_streams": active_plain, "active_blur_streams": active_blur}
    return {"message": "No stream is running"}


@router.get("/is_stream_running/{camera_ip}")
def is_stream_running(camera_ip: str):
    """Check if a specific camera is currently streaming."""
    plain = processes.get(camera_ip)
    blur_thr = BLUR_THREADS.get(camera_ip)
    running = (plain and is_process_running(plain)) or (blur_thr and blur_thr.is_alive())
    return {"camera_ip": camera_ip, "running": bool(running)}


def stop_anonymizer(camera_ip: str):
    """Stop anonymizer thread for a camera if running."""
    flag = ANON_FLAGS.get(camera_ip)
    thr = ANON_THREADS.get(camera_ip)
    if flag:
        flag.set()
    if thr:
        thr.join(timeout=2.0)
    ANON_FLAGS.pop(camera_ip, None)
    ANON_THREADS.pop(camera_ip, None)
