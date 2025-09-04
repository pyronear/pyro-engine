# Copyright (C) 2020-2025, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.


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
    stimeout_us: Optional[int] = None  # microseconds, FFmpeg 4.3 RTSP option
    fps: Optional[int] = None


@dataclass
class EncoderSettings:
    keyint: int = 5
    use_crf: bool = True
    crf: int = 28
    bitrate: str = "500k"
    bufsize: str = "100k"
    threads: int = 1
    preset: str = "ultrafast"
    tune: str = "zerolatency"


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

    # ----------------------- Public control -----------------------

    def start(self) -> None:
        """Start decoder, encoder, and model thread, then enter the frame loop."""
        if self._running.is_set():
            logging.info("Streamer already running")
            return

        self._running.set()

        self._dec_proc = self._start_decoder()
        self._enc_proc = self._start_encoder()

        self._model_thread = threading.Thread(target=self._model_loop, name="model-loop", daemon=True)
        self._model_thread.start()

        # consume ffmpeg stderr to avoid blocking
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
        """Graceful shutdown for processes and threads."""
        if not self._running.is_set():
            return
        self._running.clear()

        # close encoder stdin to flush
        try:
            if self._enc_proc and self._enc_proc.stdin:
                self._enc_proc.stdin.close()
        except Exception:
            pass

        # terminate processes
        for proc_name, proc in (("encoder", self._enc_proc), ("decoder", self._dec_proc)):
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

    def _model_loop(self) -> None:
        """Run the anonymizer on the latest frame and publish boxes."""
        logging.info("Loading Anonymizer model")
        model = Anonymizer()
        logging.info("Model ready")

        last_log = 0.0
        while self._running.is_set():
            # wait for a fresh frame, always pick the latest
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
        """Read raw frames from decoder, apply boxes, write to encoder. Restart decoder on EOF."""
        W, H = self.stream_cfg.width, self.stream_cfg.height
        frame_bytes = W * H * 3

        buffer = bytearray(frame_bytes)
        view = memoryview(buffer)
        last_stat = 0.0

        while self._running.is_set():
            # ensure decoder exists
            if not self._dec_proc or not self._dec_proc.stdout:
                self._dec_proc = self._start_decoder()
                # attach stderr logger each time we spawn a decoder
                if self._dec_proc.stderr:
                    threading.Thread(
                        target=self._log_ffmpeg_stderr, args=(self._dec_proc, "decoder"), daemon=True
                    ).start()

            dec_out = self._dec_proc.stdout
            enc_in = self._enc_proc.stdin  # encoder is created once in start()

            # read one frame
            n = 0
            while n < frame_bytes and self._running.is_set():
                chunk = dec_out.read(frame_bytes - n)
                if not chunk:
                    logging.warning("Decoder ended, restarting")
                    # clean up and restart
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
                # we broke early, go back to top to respawn decoder
                continue

            frame = np.frombuffer(buffer, dtype=np.uint8).reshape((H, W, 3))

            # downscale for model if requested
            div = max(1, int(self.det_cfg.model_scale_div))
            small = frame if div == 1 else cv2.resize(frame, (W // div, H // div), interpolation=cv2.INTER_AREA)

            rgb = Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
            if self._latest_rgb:
                self._latest_rgb[0] = rgb
            else:
                self._latest_rgb.append(rgb)
            self._new_frame_event.set()

            # apply anonymization
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

        # FFmpeg 4.3 RTSP timeout
        if sc.stimeout_us:
            cmd += ["-stimeout", str(sc.stimeout_us)]

        # Use wallclock to create monotonic PTS, also ask FFmpeg to generate PTS if needed
        cmd += ["-use_wallclock_as_timestamps", "1", "-fflags", "+genpts"]

        cmd += ["-rtsp_transport", sc.rtsp_transport, "-i", sc.rtsp_url]

        # Constant frame rate on the rawvideo pipe
        # This removes the repeated timestamp warnings from the rawvideo muxer
        if sc.fps:
            cmd += ["-r", str(sc.fps), "-vsync", "1"]  # CFR
        else:
            cmd += ["-vsync", "2"]  # VFR as a safe default

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

        cmd = [
            "ffmpeg",
            "-loglevel",
            "warning",
            "-nostats",
            "-fflags",
            "nobuffer",
            "-flags",
            "low_delay",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{W}x{H}",
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            enc.preset,
            "-tune",
            enc.tune,
            "-x264-params",
            f"keyint={enc.keyint}:min-keyint={enc.keyint}:scenecut=0:rc-lookahead=0",
            "-bf",
            "0",
            "-g",
            str(enc.keyint),
            "-threads",
            str(enc.threads),
            "-mpegts_flags",
            "resend_headers",
            "-muxdelay",
            "0",
            "-muxpreload",
            "0",
        ]

        if enc.use_crf:
            cmd += ["-crf", str(enc.crf), "-bufsize", enc.bufsize]
        else:
            cmd += ["-b:v", enc.bitrate, "-maxrate", enc.bitrate, "-bufsize", enc.bufsize]

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

    def _set_boxes_from_norm(
        self,
        norm_list: Iterable[Sequence[float]],
        conf_th: float,
    ) -> None:
        """Convert normalized boxes into pixel boxes within frame bounds."""
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
        """
        Accept flat N by 5 or nested list,
        return list of (x1, y1, x2, y2, conf) in [0, 1].
        """
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
