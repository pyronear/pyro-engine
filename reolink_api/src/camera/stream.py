# Copyright (C) 2020-2025, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.


import logging
import subprocess
import threading
import time
import sys
import signal
from collections import deque
from typing import Optional

import numpy as np
import cv2
from PIL import Image
from fastapi import APIRouter, HTTPException

from anonymizer.vision import Anonymizer
from camera.config import FFMPEG_PARAMS, STREAMS
from camera.registry import CAMERA_REGISTRY
from camera.time_utils import seconds_since_last_command, update_command_time

router = APIRouter()

# processes now holds StreamWorker objects keyed by camera_ip
processes: dict[str, "StreamWorker"] = {}


class StreamWorker:
    """Decode RTSP, run anonymizer continuously, draw black boxes, reencode to SRT."""
    def __init__(self, camera_ip: str, input_url: str, output_url: str, width: int, height: int):
        self.camera_ip = camera_ip
        self.input_url = input_url
        self.output_url = output_url
        self.W = int(width)
        self.H = int(height)

        self.dec: Optional[subprocess.Popen] = None
        self.enc: Optional[subprocess.Popen] = None
        self.reader_thread: Optional[threading.Thread] = None
        self.model_thread: Optional[threading.Thread] = None
        self.stop_evt = threading.Event()

        # shared state for boxes and latest RGB frame for the model
        self.boxes_px: list[tuple[int, int, int, int]] = []
        self.boxes_lock = threading.Lock()
        self.latest_rgb: deque[Image.Image] = deque(maxlen=1)
        self.new_frame_evt = threading.Event()

    def _start_decoder(self) -> subprocess.Popen:
        cmd = [
            "ffmpeg",
            "-rtsp_transport", FFMPEG_PARAMS.get("rtsp_transport", "tcp"),
            "-analyzeduration", "0",
            "-probesize", "32",
            "-fflags", "discardcorrupt+nobuffer" if FFMPEG_PARAMS.get("discardcorrupt") else "nobuffer",
            "-flags", "low_delay" if FFMPEG_PARAMS.get("low_delay") else "0",
            "-i", self.input_url,
            "-an",
            "-pix_fmt", "bgr24",
            "-f", "rawvideo",
            "-vsync", "passthrough",
            "-s", f"{self.W}x{self.H}",
            "pipe:1",
        ]
        logging.info(f"[{self.camera_ip}] decoder: {' '.join(cmd)}")
        return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)

    def _start_encoder(self) -> subprocess.Popen:
        # mirror your working flags
        keyint = str(FFMPEG_PARAMS.get("gop_size", 5))
        bitrate = FFMPEG_PARAMS.get("bitrate", "500k")
        preset = FFMPEG_PARAMS.get("preset", "ultrafast")
        tune = FFMPEG_PARAMS.get("tune", "zerolatency")
        b_frames = str(FFMPEG_PARAMS.get("b_frames", 0))
        threads = str(FFMPEG_PARAMS.get("threads", 1))

        cmd = [
            "ffmpeg",
            "-loglevel", "warning",
            "-nostats",
            "-fflags", "nobuffer",
            "-flags", "low_delay",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{self.W}x{self.H}",
            "-i", "pipe:0",
            "-an",
            "-c:v", FFMPEG_PARAMS.get("video_codec", "libx264"),
            "-preset", preset,
            "-tune", tune,
            "-x264-params", f"keyint={keyint}:min-keyint={keyint}:scenecut=0:rc-lookahead=0",
            "-bf", b_frames,
            "-g", keyint,
            "-threads", threads,
        ]
        if "bitrate" in FFMPEG_PARAMS:
            cmd += ["-b:v", bitrate, "-maxrate", bitrate, "-bufsize", FFMPEG_PARAMS.get("bufsize", "100k")]
        else:
            cmd += ["-crf", "28", "-bufsize", FFMPEG_PARAMS.get("bufsize", "100k")]

        cmd += [
            "-mpegts_flags", "resend_headers",
            "-muxdelay", "0",
            "-muxpreload", "0",
            "-f", FFMPEG_PARAMS.get("output_format", "mpegts"),
            self.output_url,
        ]
        logging.info(f"[{self.camera_ip}] encoder: {' '.join(cmd)}")
        return subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)

    def _normalize_boxes(self, preds, im_w: int, im_h: int):
        out = []
        if preds is None:
            return out
        if isinstance(preds, np.ndarray):
            preds_list = preds.tolist()
        else:
            preds_list = preds
        if not preds_list:
            return out
        first = preds_list[0]
        is_nested = isinstance(first, (list, tuple)) and first and isinstance(first[0], (list, tuple))
        if is_nested:
            iters = (d for dets in preds_list for d in dets)
        else:
            iters = preds_list
        for d in iters:
            if not isinstance(d, (list, tuple)) or len(d) < 4:
                continue
            x1, y1, x2, y2 = map(float, d[:4])
            conf = float(d[4]) if len(d) >= 5 else 1.0
            if x2 > 1.0001 or y2 > 1.0001:
                x1 /= im_w; y1 /= im_h; x2 /= im_w; y2 /= im_h
            out.append((x1, y1, x2, y2, conf))
        return out

    def _set_boxes_from_norm(self, norm_list, conf_th: float):
        new_boxes: list[tuple[int, int, int, int]] = []
        for it in norm_list:
            if it is None or len(it) < 4:
                continue
            x1, y1, x2, y2 = map(float, it[:4])
            conf = float(it[4]) if len(it) >= 5 else 1.0
            if conf < conf_th:
                continue
            x1p = max(0, min(self.W - 1, int(x1 * self.W)))
            y1p = max(0, min(self.H - 1, int(y1 * self.H)))
            x2p = max(0, min(self.W - 1, int(x2 * self.W)))
            y2p = max(0, min(self.H - 1, int(y2 * self.H)))
            if x2p > x1p and y2p > y1p:
                new_boxes.append((x1p, y1p, x2p, y2p))
        with self.boxes_lock:
            self.boxes_px = new_boxes

    def _model_loop(self, conf_th: float):
        logging.info(f"[{self.camera_ip}] Loading Anonymizer model")
        model = Anonymizer()
        logging.info(f"[{self.camera_ip}] Model ready")
        last_log = 0.0
        while not self.stop_evt.is_set():
            self.new_frame_evt.wait(timeout=0.1)
            self.new_frame_evt.clear()
            if not self.latest_rgb:
                continue
            im = self.latest_rgb[-1]
            try:
                preds = model(im)
                boxes_norm = self._normalize_boxes(preds, im.width, im.height)
                self._set_boxes_from_norm(boxes_norm, conf_th)
                now = time.time()
                if now - last_log > 1.0:
                    with self.boxes_lock:
                        n = len(self.boxes_px)
                    logging.info(f"[{self.camera_ip}] model updated boxes {n}")
                    last_log = now
            except Exception as e:
                logging.warning(f"[{self.camera_ip}] inference error: {e}")

    def _reader_loop(self, conf_th: float):
        frame_bytes = self.W * self.H * 3
        buffer = bytearray(frame_bytes)
        view = memoryview(buffer)
        last_stat = 0.0
        while not self.stop_evt.is_set():
            # read one raw BGR frame
            n = 0
            while n < frame_bytes and not self.stop_evt.is_set():
                chunk = self.dec.stdout.read(frame_bytes - n)
                if not chunk:
                    logging.error(f"[{self.camera_ip}] decoder ended")
                    self.stop_evt.set()
                    break
                view[n:n+len(chunk)] = chunk
                n += len(chunk)

            if self.stop_evt.is_set():
                break

            frame = np.frombuffer(buffer, dtype=np.uint8).reshape((self.H, self.W, 3))

            # give the model a copy every frame, overwrite previous
            small = cv2.resize(frame, (self.W, self.H), interpolation=cv2.INTER_AREA)
            rgb = Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
            if self.latest_rgb:
                self.latest_rgb[0] = rgb
            else:
                self.latest_rgb.append(rgb)
            self.new_frame_evt.set()

            # draw current boxes
            with self.boxes_lock:
                local_boxes = list(self.boxes_px)
            for x1, y1, x2, y2 in local_boxes:
                frame[y1:y2, x1:x2, :] = 0

            # write to encoder
            try:
                self.enc.stdin.write(buffer)
            except BrokenPipeError:
                logging.error(f"[{self.camera_ip}] encoder pipe closed")
                self.stop_evt.set()
                break

            now = time.time()
            if now - last_stat > 1.0:
                logging.info(f"[{self.camera_ip}] painted {len(local_boxes)} boxes")
                last_stat = now

        # cleanup
        try:
            if self.enc and self.enc.stdin:
                self.enc.stdin.close()
        except Exception:
            pass

    def start(self, conf_th: float):
        self.stop_evt.clear()
        self.dec = self._start_decoder()
        self.enc = self._start_encoder()

        # start threads
        self.reader_thread = threading.Thread(target=self._reader_loop, args=(conf_th,), daemon=True)
        self.model_thread = threading.Thread(target=self._model_loop, args=(conf_th,), daemon=True)
        self.reader_thread.start()
        self.model_thread.start()

        # stderr logging threads
        threading.Thread(target=self._pipe_logger, args=(self.dec.stderr, "DEC"), daemon=True).start()
        threading.Thread(target=self._pipe_logger, args=(self.enc.stderr, "ENC"), daemon=True).start()

    def _pipe_logger(self, pipe, tag: str):
        for line in iter(pipe.readline, b""):
            logging.error(f"[{self.camera_ip} {tag}] {line.decode(errors='ignore').rstrip()}")

    def is_running(self) -> bool:
        return (
            self.dec is not None
            and self.enc is not None
            and self.dec.poll() is None
            and self.enc.poll() is None
            and not self.stop_evt.is_set()
        )

    def stop(self):
        self.stop_evt.set()
        try:
            if self.enc and self.enc.stdin:
                self.enc.stdin.close()
        except Exception:
            pass
        for p in (self.enc, self.dec):
            try:
                if p and p.poll() is None:
                    p.terminate()
                    p.wait(timeout=2)
            except Exception:
                try:
                    p.kill()
                except Exception:
                    pass
        self.enc = None
        self.dec = None


def is_process_running(worker: Optional[StreamWorker]):
    return worker is not None and worker.is_running()


def stop_any_running_stream():
    for cam_id, worker in list(processes.items()):
        if is_process_running(worker):
            worker.stop()
            del processes[cam_id]
            return cam_id
    return None


def stop_stream_if_idle():
    while True:
        time.sleep(10)
        if seconds_since_last_command() > 120:
            stopped_cam = stop_any_running_stream()
            if stopped_cam:
                logging.info(f"Stream for {stopped_cam} stopped due to inactivity")


@router.post("/start_stream/{camera_ip}")
def start_stream(camera_ip: str, conf_thres: float = 0.30):
    update_command_time()
    if camera_ip not in STREAMS:
        raise HTTPException(status_code=404, detail=f"No stream config for camera {camera_ip}")

    if camera_ip in processes and is_process_running(processes[camera_ip]):
        logging.info(f"Stream for {camera_ip} already running")
        return {"message": f"Stream for {camera_ip} already running"}

    stopped_cam = stop_any_running_stream()

    info = STREAMS[camera_ip]
    input_url = info["input_url"]
    output_url = info["output_url"]
    width = info.get("width", 640)
    height = info.get("height", 360)

    worker = StreamWorker(camera_ip, input_url, output_url, width, height)
    processes[camera_ip] = worker
    worker.start(conf_thres)

    return {"message": f"Stream started for {camera_ip}", "previous_stream": stopped_cam or "None"}


@router.post("/stop_stream")
def stop_stream():
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
    active = [cam for cam, w in processes.items() if is_process_running(w)]
    if active:
        return {"active_streams": active}
    return {"message": "No stream is running"}


@router.get("/is_stream_running/{camera_ip}")
def is_stream_running_endpoint(camera_ip: str):
    w = processes.get(camera_ip)
    return {"camera_ip": camera_ip, "running": bool(w and w.is_running())}
