# Copyright (C) 2020-2025, Pyronear.
#
# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import logging
import subprocess
import threading
import time

from fastapi import APIRouter, HTTPException
import numpy as np
from anonymizer.anonymizer_registry import ANON_FLAGS, ANON_THREADS, set_result
from anonymizer.vision import Anonymizer
from camera.config import FFMPEG_PARAMS, STREAMS
from camera.registry import CAMERA_REGISTRY
from camera.time_utils import seconds_since_last_command, update_command_time

router = APIRouter()
processes: dict[str, subprocess.Popen] = {}

ANON_MODEL = Anonymizer()


def is_process_running(proc):
    """Check if a process is still running."""
    return proc and proc.poll() is None


def log_ffmpeg_output(proc, camera_id):
    """Reads and logs stderr from ffmpeg."""
    for line in proc.stderr:
        logging.error(f"[FFMPEG {camera_id}] {line.decode('utf-8').strip()}")


def stop_any_running_stream():
    """Stops any running stream, also stops its worker thread."""
    for cam_id, proc in list(processes.items()):
        if is_process_running(proc):
            logging.info(f"[{cam_id}] stopping existing stream")
            try:
                proc.terminate()
                proc.wait(timeout=2.0)
            except Exception:
                pass
            finally:
                del processes[cam_id]
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


# --------- blur pipeline helpers ---------

def _open_pyav(input_url: str):
    """Open PyAV container with low buffering for low latency decode."""
    import av  # local import
    opts = {
        "rtsp_transport": "tcp",
        "fflags": "nobuffer",
        "reorder_queue_size": "0",
        "analyzeduration": "0",
        "probesize": "32",
    }
    logging.info(f"[worker] opening RTSP {input_url}")
    container = av.open(input_url, options=opts)
    stream = next(s for s in container.streams if s.type == "video")
    frames = container.decode(video=stream.index)
    return container, frames

def _blur_boxes_bgr(img_bgr, preds):
    """Redact boxes from raw preds by painting them black. Accepts ndarray, list, or dict."""
    if img_bgr is None or preds is None:
        return img_bgr

    h, w = img_bgr.shape[:2]
    boxes_px = []

    def to_pixels(x1, y1, x2, y2):
        # treat as normalized if values look like ratios
        if max(abs(x1), abs(y1), abs(x2), abs(y2)) <= 1.05:
            x1 *= w; x2 *= w; y1 *= h; y2 *= h
        x1 = int(round(x1)); x2 = int(round(x2))
        y1 = int(round(y1)); y2 = int(round(y2))
        # clip to image
        x1 = max(0, min(w - 1, x1)); x2 = max(0, min(w, x2))
        y1 = max(0, min(h - 1, y1)); y2 = max(0, min(h, y2))
        return x1, y1, x2, y2

    if isinstance(preds, np.ndarray):
        arr = preds.reshape(-1, preds.shape[-1]) if preds.ndim > 1 else preds.reshape(1, -1)
        for row in arr:
            if row.shape[0] >= 4:
                x1, y1, x2, y2 = map(float, row[:4])
                boxes_px.append(to_pixels(x1, y1, x2, y2))
    elif isinstance(preds, (list, tuple)):
        for p in preds:
            if isinstance(p, dict):
                box = p.get("box") or p.get("bbox")
                if box and len(box) >= 4:
                    x1, y1, x2, y2 = map(float, box[:4])
                    boxes_px.append(to_pixels(x1, y1, x2, y2))
            elif isinstance(p, (list, tuple, np.ndarray)) and len(p) >= 4:
                x1, y1, x2, y2 = map(float, p[:4])
                boxes_px.append(to_pixels(x1, y1, x2, y2))

    for x1, y1, x2, y2 in boxes_px:
        if x2 > x1 and y2 > y1:
            img_bgr[y1:y2, x1:x2] = 0

    return img_bgr


def _blur_boxes_px(img_bgr, boxes_px):
    """Redact a list of pixel boxes by painting them black."""
    if img_bgr is None or not boxes_px:
        return img_bgr
    h, w = img_bgr.shape[:2]
    for x1, y1, x2, y2 in boxes_px:
        if x2 <= x1 or y2 <= y1:
            continue
        x1 = max(0, min(w - 1, x1)); x2 = max(0, min(w, x2))
        y1 = max(0, min(h - 1, y1)); y2 = max(0, min(h, y2))
        img_bgr[y1:y2, x1:x2] = 0
    return img_bgr



def _start_ffmpeg_sink(width: int, height: int, output_url: str) -> subprocess.Popen:
    """
    Spawn FFmpeg that reads raw BGR frames on stdin and pushes to output_url.
    Uses params from FFMPEG_PARAMS. Adds scale to even dimensions for x264 safety.
    """
    scale_filter = "scale=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p"

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "warning",

        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}",
        "-r", str(FFMPEG_PARAMS["framerate"]),
        "-i", "pipe:0",

        "-vf", scale_filter,

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
def _parse_boxes_to_px(preds, w, h, score_thresh: float = 0.12, pad_ratio: float = 0.04):
    """Return pixel boxes from preds. Works with ndarray, list, or dict."""
    import numpy as np
    boxes = []
    if preds is None:
        return boxes

    def to_px(x1, y1, x2, y2):
        if max(abs(x1), abs(y1), abs(x2), abs(y2)) <= 1.05:
            x1 *= w; x2 *= w; y1 *= h; y2 *= h
        x1 = int(round(x1)); x2 = int(round(x2))
        y1 = int(round(y1)); y2 = int(round(y2))
        pad_x = int(round((x2 - x1) * pad_ratio))
        pad_y = int(round((y2 - y1) * pad_ratio))
        x1 -= pad_x; x2 += pad_x; y1 -= pad_y; y2 += pad_y
        x1 = max(0, min(w - 1, x1)); x2 = max(0, min(w, x2))
        y1 = max(0, min(h - 1, y1)); y2 = max(0, min(h, y2))
        return x1, y1, x2, y2

    if isinstance(preds, np.ndarray):
        arr = preds.reshape(-1, preds.shape[-1]) if preds.ndim > 1 else preds.reshape(1, -1)
        for row in arr:
            if row.shape[0] >= 4:
                score = float(row[4]) if row.shape[0] >= 5 else 1.0
                if score >= score_thresh:
                    x1, y1, x2, y2 = map(float, row[:4])
                    boxes.append(to_px(x1, y1, x2, y2))
    elif isinstance(preds, (list, tuple)):
        for p in preds:
            if isinstance(p, dict):
                box = p.get("box") or p.get("bbox")
                score = float(p.get("score", 1.0))
                if box and len(box) >= 4 and score >= score_thresh:
                    x1, y1, x2, y2 = map(float, box[:4])
                    boxes.append(to_px(x1, y1, x2, y2))
            elif isinstance(p, (list, tuple)) and len(p) >= 4:
                score = float(p[4]) if len(p) >= 5 else 1.0
                if score >= score_thresh:
                    x1, y1, x2, y2 = map(float, p[:4])
                    boxes.append(to_px(x1, y1, x2, y2))
    return boxes




def _blur_worker(camera_ip: str, stop_flag: threading.Event):
    stream_info = STREAMS[camera_ip]
    input_url = stream_info["input_url"]
    output_url = stream_info["output_url"]

    container = None
    frames = None
    sink = None

    # hold blur for this many milliseconds after the last detection
    STICKY_MS = 500
    last_boxes = []
    last_seen_ms = 0.0

    try:
        container, frames = _open_pyav(input_url)

        first = next(frames, None)
        if first is None:
            raise RuntimeError("No frame from RTSP input")
        img = first.to_ndarray(format="bgr24")
        h, w = img.shape[:2]

        sink = _start_ffmpeg_sink(w, h, output_url)
        processes[camera_ip] = sink
        logging.info(f"[{camera_ip}] blur worker started")

        while not stop_flag.is_set() and sink.poll() is None:
            frame = first if first is not None else next(frames, None)
            first = None
            if frame is None:
                raise RuntimeError("Decoder returned no frame")

            img = frame.to_ndarray(format="bgr24")

            preds = ANON_MODEL(img)
            set_result(camera_ip, preds)

            now_ms = time.perf_counter() * 1000.0
            boxes = _parse_boxes_to_px(preds, w, h)

            if boxes:
                last_boxes = boxes
                last_seen_ms = now_ms
            elif now_ms - last_seen_ms < STICKY_MS:
                boxes = last_boxes

            img = _blur_boxes_px(img, boxes)

            try:
                sink.stdin.write(img.tobytes())
            except BrokenPipeError:
                raise RuntimeError("FFmpeg sink closed")

        logging.info(f"[{camera_ip}] blur worker stopping")

    except Exception as e:
        logging.error(f"[{camera_ip}] blur worker error: {e}")

    finally:
        try:
            if sink is not None:
                try:
                    sink.stdin.close()
                except Exception:
                    pass
                sink.terminate()
        except Exception:
            pass
        processes.pop(camera_ip, None)
        try:
            if container is not None:
                container.close()
        except Exception:
            pass


# --------- routes ---------

@router.post("/start_stream/{camera_ip}")
def start_stream(camera_ip: str):
    """Start blurred stream for a camera by launching the blur worker thread."""
    update_command_time()
    if camera_ip not in STREAMS:
        raise HTTPException(status_code=404, detail=f"No stream config for camera {camera_ip}")

    # always stop any existing process so the blur path is used
    stopped_cam = stop_any_running_stream()
    if stopped_cam:
        logging.info(f"[{camera_ip}] previous stream {stopped_cam} stopped")

    stop_flag = threading.Event()
    thr = threading.Thread(target=_blur_worker, args=(camera_ip, stop_flag), daemon=True)
    ANON_FLAGS[camera_ip] = stop_flag
    ANON_THREADS[camera_ip] = thr
    thr.start()

    return {"message": f"Stream started for {camera_ip}", "previous_stream": stopped_cam or "None"}


@router.post("/stop_stream")
def stop_stream():
    """Stops any active stream and resets zoom, also stops worker thread."""
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


def stop_anonymizer(camera_ip: str):
    """Stop blur worker thread for a camera if running."""
    flag = ANON_FLAGS.get(camera_ip)
    thr = ANON_THREADS.get(camera_ip)
    if flag:
        flag.set()
    if thr:
        thr.join(timeout=2.0)
    ANON_FLAGS.pop(camera_ip, None)
    ANON_THREADS.pop(camera_ip, None)
