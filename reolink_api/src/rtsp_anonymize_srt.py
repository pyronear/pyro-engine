#!/usr/bin/env python3
# Copyright (C) 2020-2025, Pyronear.
# Licensed under the Apache License 2.0.

from __future__ import annotations

import logging
import subprocess
import threading
from typing import List, Optional, Tuple, Iterable, Sequence

import cv2
import numpy as np
from PIL import Image
from anonymizer.vision import Anonymizer


# ----------------------------- SRT helpers -----------------------------

def build_srt_url(
    srt: Optional[str] = None,
    host: Optional[str] = None,
    port: int = 8890,
    streamid: Optional[str] = None,
    pkt_size: int = 1316,
    latency: int = 50,
    mode: str = "caller",
    rcvlatency: Optional[int] = None,
    peerlatency: Optional[int] = None,
    tlpktdrop: int = 1,
) -> str:
    if srt:
        return srt
    if not host:
        raise ValueError("SRT host required when srt is not provided")
    params = {
        "pkt_size": str(pkt_size),
        "mode": mode,
        "latency": str(latency),
        "tlpktdrop": str(tlpktdrop),
    }
    if rcvlatency is not None:
        params["rcvlatency"] = str(rcvlatency)
    if peerlatency is not None:
        params["peerlatency"] = str(peerlatency)
    if streamid:
        params["streamid"] = streamid
    query = "&".join(f"{k}={v}" for k, v in params.items())
    return f"srt://{host}:{port}?{query}"


# ----------------------------- FFmpeg cmds -----------------------------

def build_decoder_cmd(
    rtsp_url: str,
    width: int,
    height: int,
    rtsp_transport: str = "tcp",
    fps: Optional[int] = 7,
    analyzeduration: str = "0",
    probesize: str = "32k",
    low_delay: bool = True,
    discardcorrupt: bool = True,
    dec_threads: int = 2,
) -> List[str]:
    cmd: List[str] = ["ffmpeg"]
    if analyzeduration is not None:
        cmd += ["-analyzeduration", str(analyzeduration)]
    if probesize is not None:
        cmd += ["-probesize", str(probesize)]
    if discardcorrupt:
        cmd += ["-fflags", "discardcorrupt+nobuffer"]
    if low_delay:
        cmd += ["-flags", "low_delay"]
    cmd += ["-use_wallclock_as_timestamps", "1", "-fflags", "+genpts"]
    cmd += ["-rtsp_transport", rtsp_transport, "-i", rtsp_url]
    if dec_threads and dec_threads > 1:
        cmd += ["-threads", str(dec_threads), "-thread_type", "slice"]

    if fps and fps > 0:
        cmd += ["-r", str(fps), "-vsync", "1"] 
    else:
        cmd += ["-fps_mode", "passthrough"]

    cmd += [
        "-an",
        "-pix_fmt", "bgr24",
        "-f", "rawvideo",
        "-s", f"{width}x{height}",
        "pipe:1",
    ]
    return cmd


def build_encoder_cmd(
    srt_out: str,
    width: int,
    height: int,
    keyint: int = 14,
    use_crf: bool = False,
    crf: int = 28,
    bitrate: str = "700k",
    bufsize: str = "800k",
    maxrate: str = "900k",
    threads: int = 1,
    preset: str = "veryfast",
    tune: str = "zerolatency",
    pix_fmt: str = "yuv420p",
    x264_params: str = "keyint=14:min-keyint=7:scenecut=40:rc-lookahead=0:ref=2:aq-mode=2",
    frame_threads: int = 1,           # kept for signature compatibility, not injected anymore
    sliced_threads: bool = True,
    enc_input_fps: int = 10,
) -> List[str]:
    # merge low latency params without frame-threads to avoid parser error
    params = x264_params.split(":") if x264_params else []
    have = {p.split("=")[0] for p in params if "=" in p}
    need = {
        "keyint": str(keyint),
        "min-keyint": str(max(1, keyint // 2)),
        "scenecut": "40",
        "rc-lookahead": "0",
        "ref": "3",
    }
    if sliced_threads:
        need["sliced-threads"] = "1"
    for k, v in need.items():
        if k not in have:
            params.insert(0, f"{k}={v}")

    # if user provided keyint in x264-params, sync -g to it
    g_val = keyint
    for p in params:
        if p.startswith("keyint="):
            try:
                g_val = int(p.split("=", 1)[1])
            except Exception:
                pass
            break

    x264_merged = ":".join(params)

    cmd: List[str] = [
        "ffmpeg",
        "-loglevel", "warning",
        "-nostats",
        "-fflags", "nobuffer",
        "-flags", "low_delay",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}",
        "-framerate", str(max(1, enc_input_fps)),
        "-fflags", "+genpts",
        "-i", "pipe:0",
        "-an",
        "-pix_fmt", pix_fmt,
        "-c:v", "libx264",
        "-preset", preset,
        "-tune", tune,
        "-x264-params", x264_merged,
        "-bf", "0",
        "-g", str(g_val),              # synced with x264 keyint if present
        "-threads", str(threads),
        "-mpegts_flags", "resend_headers",
        "-muxdelay", "0",
        "-muxpreload", "0",
    ]
    if use_crf:
        cmd += ["-crf", str(crf), "-maxrate", maxrate, "-bufsize", bufsize]
    else:
        cmd += ["-b:v", bitrate, "-maxrate", maxrate, "-bufsize", bufsize]
    cmd += ["-f", "mpegts", srt_out]
    return cmd



def log_ffmpeg_stderr(proc: subprocess.Popen, name: str) -> None:
    if not proc.stderr:
        return
    for line in iter(proc.stderr.readline, b""):
        if not line:
            break
        try:
            logging.info("[%s] %s", name, line.decode(errors="ignore").rstrip())
        except Exception:
            pass


# ----------------------------- Shared state -----------------------------

class LatestFrame:
    def __init__(self) -> None:
        self._im: Optional[Image.Image] = None
        self._lock = threading.Lock()
        self._event = threading.Event()

    def update(self, im: Image.Image) -> None:
        with self._lock:
            self._im = im
        self._event.set()

    def wait_and_get(self, timeout: float = 0.05) -> Optional[Image.Image]:
        if not self._event.wait(timeout=timeout):
            return None
        self._event.clear()
        with self._lock:
            return self._im


class BoxState:
    def __init__(self) -> None:
        self._boxes: List[Tuple[int, int, int, int]] = []
        self._lock = threading.Lock()

    def set(self, boxes: List[Tuple[int, int, int, int]]) -> None:
        with self._lock:
            self._boxes = list(boxes)

    def get(self) -> List[Tuple[int, int, int, int]]:
        with self._lock:
            return list(self._boxes)


# ----------------------------- Vision helpers -----------------------------

def boxes_px_from_norm(
    boxes_norm: Iterable[Sequence[float]],
    W: int,
    H: int,
    conf_th: float,
) -> List[Tuple[int, int, int, int]]:
    out_px: List[Tuple[int, int, int, int]] = []
    for it in boxes_norm:
        if it is None or len(it) < 4:
            continue
        x1, y1, x2, y2 = map(float, it[:4])
        conf = float(it[4]) if len(it) >= 5 else 1.0
        if conf < conf_th:
            continue
        x1p = max(0, min(W - 1, int(x1 * W)))
        y1p = max(0, min(H - 1, int(y1 * H)))
        x2p = max(0, min(W - 1, int(x2 * W)))
        y2p = max(0, min(H - 1, int(y2 * H)))
        if x2p > x1p and y2p > y1p:
            out_px.append((x1p, y1p, x2p, y2p))
    return out_px


def paint_black(arr: np.ndarray, boxes_px: List[Tuple[int, int, int, int]]) -> None:
    for x1, y1, x2, y2 in boxes_px:
        arr[y1:y2, x1:x2, :] = 0


# ----------------------------- Model thread -----------------------------

def anonymizer_thread_fn(
    latest: LatestFrame,
    boxes_state: BoxState,
    conf_thres: float,
    stop_event: threading.Event,
) -> None:
    model: Optional[Anonymizer] = None
    backoff_s = 1.0
    while not stop_event.is_set():
        try:
            if model is None:
                logging.info("Loading Anonymizer model")
                model = Anonymizer()
                logging.info("Model ready")

            im = latest.wait_and_get(timeout=0.05)
            if im is None:
                continue

            preds = model(im)
            boxes_px = boxes_px_from_norm(preds, im.width, im.height, conf_thres)
            boxes_state.set(boxes_px)

        except BaseException as e:
            logging.warning("Model thread error: %s", e)
            model = None
            stop_event.wait(backoff_s)
            backoff_s = min(backoff_s * 2.0, 10.0)


# ----------------------------- Worker -----------------------------

class RTSPAnonymizeSRTWorker:
    def __init__(
        self,
        rtsp_url: str,
        srt_out: Optional[str] = None,
        *,
        width: int = 640,
        height: int = 360,
        fps: int = 7,
        rtsp_transport: str = "tcp",
        srt_host: Optional[str] = None,
        srt_port: int = 8890,
        streamid: Optional[str] = None,
        conf_thres: float = 0.30,
        x264_preset: str = "veryfast",
        x264_tune: str = "zerolatency",
        bitrate: str = "700k",
        bufsize: str = "800k",
        maxrate: str = "900k",
        use_crf: bool = False,
        crf: int = 28,
        keyint: int = 14,
        pix_fmt: str = "yuv420p",
        enc_threads: int = 1,
        dec_threads: int = 2,
        # new SRT knobs
        srt_latency: int = 50,
        srt_pkt_size: int = 1316,
        srt_rcvlatency: Optional[int] = None,
        srt_peerlatency: Optional[int] = None,
        srt_tlpktdrop: int = 1,
    ) -> None:
        self.width = width
        self.height = height
        self.frame_bytes = width * height * 3
        self.conf_thres = conf_thres

        self.srt_out = srt_out or build_srt_url(
            host=srt_host,
            port=srt_port,
            streamid=streamid,
            pkt_size=srt_pkt_size,
            latency=srt_latency,
            mode="caller",
            rcvlatency=srt_rcvlatency,
            peerlatency=srt_peerlatency,
            tlpktdrop=srt_tlpktdrop,
        )
        self.dec_cmd = build_decoder_cmd(
            rtsp_url=rtsp_url,
            width=width,
            height=height,
            rtsp_transport=rtsp_transport,
            fps=fps,
            analyzeduration="0",
            probesize="32k",
            low_delay=True,
            discardcorrupt=True,
            dec_threads=dec_threads,
        )
        self.enc_cmd = build_encoder_cmd(
            srt_out=self.srt_out,
            width=width,
            height=height,
            keyint=keyint or 40,
            use_crf=use_crf,
            crf=crf,
            bitrate=bitrate,
            bufsize=bufsize,
            maxrate=maxrate,
            threads=enc_threads,
            preset=x264_preset,
            tune=x264_tune,
            pix_fmt=pix_fmt,
            frame_threads=1,
            sliced_threads=True,
            enc_input_fps=fps or 10,
        )

        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._dec: Optional[subprocess.Popen] = None
        self._enc: Optional[subprocess.Popen] = None

        self.latest = LatestFrame()
        self.boxes_state = BoxState()

        self._model_thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="rtsp-anon-srt", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)

    def _open_procs(self) -> None:
        logging.info("Starting decoder: %s", " ".join(self.dec_cmd))
        self._dec = subprocess.Popen(
            self.dec_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        logging.info("Starting encoder: %s", " ".join(self.enc_cmd))
        self._enc = subprocess.Popen(
            self.enc_cmd,
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        threading.Thread(target=log_ffmpeg_stderr, args=(self._dec, "decoder"), daemon=True).start()
        threading.Thread(target=log_ffmpeg_stderr, args=(self._enc, "encoder"), daemon=True).start()

    def _close_procs(self) -> None:
        for proc in (self._enc, self._dec):
            if not proc:
                continue
            try:
                if proc is self._enc and proc.stdin:
                    proc.stdin.close()
            except Exception:
                pass
            try:
                proc.terminate()
            except Exception:
                pass
            try:
                proc.wait(timeout=2)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        self._dec = None
        self._enc = None

    def _spawn_model_thread(self) -> None:
        if self._model_thread and self._model_thread.is_alive():
            return
        self._model_thread = threading.Thread(
            target=anonymizer_thread_fn,
            args=(self.latest, self.boxes_state, self.conf_thres, self._stop),
            daemon=True,
            name="model-thread",
        )
        self._model_thread.start()

    def _run(self) -> None:
        buffer = bytearray(self.frame_bytes)
        view = memoryview(buffer)

        try:
            self._open_procs()
            assert self._dec and self._dec.stdout
            assert self._enc and self._enc.stdin

            self._spawn_model_thread()

            while not self._stop.is_set():
                # read exactly one frame worth of bytes
                n = 0
                while n < self.frame_bytes and not self._stop.is_set():
                    chunk = self._dec.stdout.read(self.frame_bytes - n)
                    if not chunk:
                        logging.warning("Decoder ended")
                        self._stop.set()
                        break
                    view[n : n + len(chunk)] = chunk
                    n += len(chunk)
                if n < self.frame_bytes:
                    break

                frame = np.frombuffer(buffer, dtype=np.uint8).reshape((self.height, self.width, 3))

                # push a small RGB copy to the model thread
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.latest.update(Image.fromarray(rgb))

                # paint boxes from the last model result
                boxes = self.boxes_state.get()
                if boxes:
                    paint_black(frame, boxes)

                try:
                    self._enc.stdin.write(buffer)
                except BrokenPipeError:
                    logging.warning("Encoder pipe closed")
                    self._stop.set()
                    break

        except BaseException as e:
            logging.error("Worker error: %s", e)
        finally:
            self._close_procs()
            logging.info("Worker stopped")
