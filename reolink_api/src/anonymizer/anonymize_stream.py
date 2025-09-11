# Copyright (C) 2020-2025, Pyronear.
#
# Licensed under the Apache License 2.0.
# See LICENSE or <https://opensource.org/licenses/Apache-2.0> for details.

from __future__ import annotations

import logging
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from anonymizer.vision import Anonymizer
from PIL import Image

# ---------------------------------------------------------------------
# Data classes for configuration
# ---------------------------------------------------------------------


@dataclass
class StreamConfig:
    rtsp_url: str
    srt_out: str
    width: int = 640
    height: int = 360
    rtsp_transport: str = "tcp"
    analyzeduration: Optional[str] = None
    probesize: Optional[str] = None
    low_delay: bool = True
    discardcorrupt: bool = True
    stimeout_us: Optional[int] = None
    fps: Optional[int] = None
    genpts: bool = True
    wallclock_ts: bool = True
    nobuffer: bool = True


@dataclass
class EncoderSettings:
    keyint: int = 14
    use_crf: bool = False
    crf: int = 28
    bitrate: str = "700k"
    bufsize: str = "800k"
    maxrate: str = "900k"
    threads: int = 1
    preset: str = "veryfast"
    tune: str = "zerolatency"
    pix_fmt: str = "yuv420p"
    x264_params: str = "keyint=14:min-keyint=7:scenecut=40:rc-lookahead=0:ref=2:aq-mode=2"
    mpegts_flags: str = "resend_headers"
    muxdelay: str = "0"
    muxpreload: str = "0"


@dataclass
class DetectionSettings:
    conf_thres: float = 0.30
    model_scale_div: int = 1


# ---------------------------------------------------------------------
# Main streamer
# ---------------------------------------------------------------------


class AnonymizingStreamer:
    def __init__(
        self,
        stream_cfg: StreamConfig,
        enc_cfg: EncoderSettings | None = None,
        det_cfg: DetectionSettings | None = None,
    ) -> None:
        self.stream_cfg = stream_cfg
        self.enc_cfg = enc_cfg or EncoderSettings()
        self.det_cfg = det_cfg or DetectionSettings()

        self._dec_proc: subprocess.Popen | None = None
        self._enc_proc: subprocess.Popen | None = None

        self._boxes_px: List[Tuple[int, int, int, int]] = []
        self._boxes_lock = threading.Lock()

        self._latest_rgb: deque[Image.Image] = deque(maxlen=1)
        self._new_frame_event = threading.Event()

        self._model_thread: threading.Thread | None = None
        self._stderr_threads: List[threading.Thread] = []
        self._running = threading.Event()

        # model state
        self._model: Optional[Anonymizer] = None
        self._model_ready = threading.Event()
        self._model_error: Optional[BaseException] = None

    # ----------------------- Public control -----------------------

    def start(self) -> None:
        if self._running.is_set():
            logging.info("Streamer already running")
            return

        self._running.set()

        self._dec_proc = self._start_decoder()
        self._enc_proc = self._start_encoder()

        # background loader
        threading.Thread(target=self._load_model_bg, name="model-loader", daemon=True).start()

        self._model_thread = threading.Thread(target=self._model_loop, name="model-loop", daemon=True)
        self._model_thread.start()

        # consume ffmpeg stderr
        if self._dec_proc and self._dec_proc.stderr:
            t = threading.Thread(target=self._log_ffmpeg_stderr, args=(self._dec_proc, "decoder"), daemon=True)
            t.start()
            self._stderr_threads.append(t)
        if self._enc_proc and self._enc_proc.stderr:
            t = threading.Thread(target=self._log_ffmpeg_stderr, args=(self._enc_proc, "encoder"), daemon=True)
            t.start()
            self._stderr_threads.append(t)

        try:
            self._frame_loop()
        except KeyboardInterrupt:
            logging.info("Interrupted by user")
        finally:
            self.stop()

    def stop(self) -> None:
        if not self._running.is_set():
            return
        self._running.clear()

        try:
            if self._enc_proc and self._enc_proc.stdin:
                self._enc_proc.stdin.close()
        except Exception:
            pass

        for proc in (self._enc_proc, self._dec_proc):
            try:
                if proc:
                    proc.terminate()
            except Exception:
                pass
            try:
                if proc:
                    proc.wait(timeout=2)
            except Exception:
                try:
                    if proc:
                        proc.kill()
                except Exception:
                    pass

        self._enc_proc = None
        self._dec_proc = None

    # ----------------------- Internal loops -----------------------

    def _load_model_bg(self) -> None:
        backoff_s = 1.0
        while self._running.is_set() and not self._model_ready.is_set():
            try:
                logging.info("Loading Anonymizer model")
                self._model = Anonymizer()
                self._model_ready.set()
                self._model_error = None
                logging.info("Model ready")
                return
            except BaseException as e:
                self._model_error = e
                logging.warning("Model load failed, retry soon: %s", e)
                time.sleep(min(backoff_s, 10.0))
                backoff_s = min(backoff_s * 2.0, 10.0)

    def _model_loop(self) -> None:
        last_log = 0.0
        last_not_ready = 0.0

        while self._running.is_set():
            if not self._model_ready.is_set():
                now = time.time()
                if now - last_not_ready > 3.0:
                    if self._model_error:
                        logging.info("Model not ready. Last error: %s", self._model_error)
                    else:
                        logging.info("Model not ready yet")
                    last_not_ready = now
                time.sleep(0.05)
                continue

            model = self._model
            if model is None:
                time.sleep(0.05)
                continue

            self._new_frame_event.wait(timeout=0.1)
            self._new_frame_event.clear()
            if not self._latest_rgb:
                continue

            im = self._latest_rgb[-1]
            try:
                preds = model(im)
                boxes_norm = self._normalize_boxes(preds, im.width, im.height)
                self._set_boxes_from_norm(boxes_norm, self.det_cfg.conf_thres)
                now = time.time()
                if now - last_log > 1.0:
                    with self._boxes_lock:
                        n = len(self._boxes_px)
                    logging.info("model updated boxes %d", n)
                    last_log = now
            except Exception as e:
                logging.exception("inference error: %s", e)

    def _frame_loop(self) -> None:
        W, H = self.stream_cfg.width, self.stream_cfg.height
        frame_bytes = W * H * 3
        buffer = bytearray(frame_bytes)
        view = memoryview(buffer)
        last_stat = 0.0

        while self._running.is_set():
            if not self._dec_proc or not self._dec_proc.stdout:
                self._dec_proc = self._start_decoder()
                if self._dec_proc.stderr:
                    threading.Thread(
                        target=self._log_ffmpeg_stderr, args=(self._dec_proc, "decoder"), daemon=True
                    ).start()

            dec_out = self._dec_proc.stdout
            enc_in = self._enc_proc.stdin

            n = 0
            while n < frame_bytes and self._running.is_set():
                chunk = dec_out.read(frame_bytes - n)
                if not chunk:
                    logging.warning("Decoder ended, restarting")
                    try:
                        if self._dec_proc:
                            self._dec_proc.terminate()
                            self._dec_proc.wait(timeout=1)
                    except Exception:
                        pass
                    self._dec_proc = None
                    time.sleep(0.25)
                    break
                view[n : n + len(chunk)] = chunk
                n += len(chunk)

            if n < frame_bytes:
                continue

            frame = np.frombuffer(buffer, dtype=np.uint8).reshape((H, W, 3))

            div = max(1, int(self.det_cfg.model_scale_div))
            small = frame if div == 1 else cv2.resize(frame, (W // div, H // div), interpolation=cv2.INTER_AREA)

            rgb = Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
            if self._latest_rgb:
                self._latest_rgb[0] = rgb
            else:
                self._latest_rgb.append(rgb)
            self._new_frame_event.set()

            with self._boxes_lock:
                local_boxes = list(self._boxes_px)
            for x1, y1, x2, y2 in local_boxes:
                frame[y1:y2, x1:x2, :] = 0

            now = time.time()
            if now - last_stat > 1.0:
                logging.info("painted %d boxes", len(local_boxes))
                last_stat = now

            try:
                enc_in.write(buffer)
            except BrokenPipeError:
                logging.warning("Encoder pipe closed")
                return

    # ----------------------- FFmpeg helpers -----------------------

    def _start_decoder(self) -> subprocess.Popen:
        W, H = self.stream_cfg.width, self.stream_cfg.height
        sc = self.stream_cfg
        cmd = ["ffmpeg"]

        if sc.analyzeduration is not None:
            cmd += ["-analyzeduration", str(sc.analyzeduration)]
        if sc.probesize is not None:
            cmd += ["-probesize", str(sc.probesize)]
        if sc.discardcorrupt:
            cmd += ["-fflags", "discardcorrupt+nobuffer"]
        if sc.low_delay:
            cmd += ["-flags", "low_delay"]

        # timestamps on decode side
        if sc.wallclock_ts:
            cmd += ["-use_wallclock_as_timestamps", "1"]
        if sc.genpts:
            cmd += ["-fflags", "+genpts"]

        # stimeout_us is not supported on some builds, keep optional
        if sc.stimeout_us:
            cmd += ["-stimeout", str(sc.stimeout_us)]

        cmd += ["-rtsp_transport", sc.rtsp_transport, "-i", sc.rtsp_url]

        if sc.fps and sc.fps > 0:
            cmd += ["-r", str(sc.fps), "-vsync", "1"]
        else:
            cmd += ["-fps_mode", "passthrough"]

        cmd += [
            "-an",
            "-pix_fmt",
            "bgr24",
            "-f",
            "rawvideo",
            "-s",
            f"{W}x{H}",
            "pipe:1",
        ]

        logging.info("Starting decoder: %s", " ".join(cmd))
        return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)

    def _start_encoder(self) -> subprocess.Popen:
        W, H = self.stream_cfg.width, self.stream_cfg.height
        enc = self.enc_cfg

        # give timestamps to rawvideo pipe based on requested fps
        enc_input_fps = str(self.stream_cfg.fps if self.stream_cfg.fps and self.stream_cfg.fps > 0 else 10)

        cmd = [
            "ffmpeg",
            "-loglevel",
            "warning",
            "-nostats",
            "-fflags",
            "nobuffer" if self.stream_cfg.nobuffer else " ",
            "-flags",
            "low_delay",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{W}x{H}",
            "-framerate",
            enc_input_fps,  # important for monotonic PTS
            "-fflags",
            "+genpts",
            "-i",
            "pipe:0",
            "-an",
            "-pix_fmt",
            enc.pix_fmt,
            "-c:v",
            "libx264",
            "-preset",
            enc.preset,
            "-tune",
            enc.tune,
            "-x264-params",
            enc.x264_params,
            "-bf",
            "0",
            "-g",
            str(enc.keyint),
            "-threads",
            str(enc.threads),
            "-mpegts_flags",
            enc.mpegts_flags,
            "-muxdelay",
            enc.muxdelay,
            "-muxpreload",
            enc.muxpreload,
        ]
        if enc.use_crf:
            cmd += ["-crf", str(enc.crf), "-maxrate", enc.maxrate, "-bufsize", enc.bufsize]
        else:
            cmd += ["-b:v", enc.bitrate, "-maxrate", enc.maxrate, "-bufsize", enc.bufsize]

        cmd += ["-f", "mpegts", self.stream_cfg.srt_out]
        logging.info("Starting encoder: %s", " ".join(cmd))
        return subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)

    @staticmethod
    def _log_ffmpeg_stderr(proc: subprocess.Popen, name: str) -> None:
        if not proc.stderr:
            return
        for line in iter(proc.stderr.readline, b""):
            if not line:
                break
            try:
                logging.info("[%s] %s", name, line.decode(errors="ignore").rstrip())
            except Exception:
                pass

    # ----------------------- Box utilities -----------------------

    def _set_boxes_from_norm(self, norm_list: Iterable[Sequence[float]], conf_th: float) -> None:
        W, H = self.stream_cfg.width, self.stream_cfg.height
        new_boxes: List[Tuple[int, int, int, int]] = []
        for it in norm_list:
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
                new_boxes.append((x1p, y1p, x2p, y2p))
        with self._boxes_lock:
            self._boxes_px[:] = new_boxes

    @staticmethod
    def _normalize_boxes(preds, im_w: int, im_h: int) -> List[Tuple[float, float, float, float, float]]:
        out: List[Tuple[float, float, float, float, float]] = []
        if isinstance(preds, np.ndarray):
            preds_list = preds.tolist()
        else:
            preds_list = preds
        if not preds_list:
            return out

        first = preds_list[0]
        is_nested = isinstance(first, (list, tuple)) and len(first) > 0 and isinstance(first[0], (list, tuple))

        if is_nested:
            for dets in preds_list:
                for d in dets:
                    if len(d) < 4:
                        continue
                    x1, y1, x2, y2 = map(float, d[:4])
                    conf = float(d[4]) if len(d) >= 5 else 1.0
                    if x2 > 1.0001 or y2 > 1.0001:
                        x1 /= im_w
                        y1 /= im_h
                        x2 /= im_w
                        y2 /= im_h
                    out.append((x1, y1, x2, y2, conf))
        else:
            for d in preds_list:
                if not isinstance(d, (list, tuple)) or len(d) < 4:
                    continue
                x1, y1, x2, y2 = map(float, d[:4])
                conf = float(d[4]) if len(d) >= 5 else 1.0
                if x2 > 1.0001 or y2 > 1.0001:
                    x1 /= im_w
                    y1 /= im_h
                    x2 /= im_w
                    y2 /= im_h
                out.append((x1, y1, x2, y2, conf))
        return out
