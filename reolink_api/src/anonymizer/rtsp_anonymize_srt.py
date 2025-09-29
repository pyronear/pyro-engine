# Copyright (C) 2022-2025, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

from __future__ import annotations

import io
import logging
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Sequence, Tuple, cast

import cv2
import numpy as np
from PIL import Image

from anonymizer.vision import Anonymizer

# ----------------------------- Logging -----------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(threadName)s | %(message)s",
)

from urllib.parse import quote, urlencode

# ----------------------------- SRT defaults -----------------------------

SRT_PKT_SIZE = 1316
SRT_MODE = "caller"
SRT_LATENCY = 50
SRT_PORT_START = 8890
SRT_STREAMID_PREFIX = "publish"
MEDIAMTX_SERVER_IP = "91.134.47.14"


def normalize_stream_name(name: str) -> str:
    """Replace spaces and unsafe characters for SRT streamid"""
    return name.strip().replace(" ", "_")


def build_srt_output_url(name_or_id: str) -> str:
    """
    If value looks like a full SRT streamid already, pass as is.
    Otherwise prefix with publish and normalize.
    """
    if name_or_id.startswith("#!::") or name_or_id.startswith("publish:") or ":" in name_or_id:
        streamid = name_or_id
        safe_chars = ":,=/!"
    else:
        streamid = f"{SRT_STREAMID_PREFIX}:{normalize_stream_name(name_or_id)}"
        safe_chars = ":"

    query = urlencode(
        {
            "pkt_size": SRT_PKT_SIZE,
            "mode": SRT_MODE,
            "latency": SRT_LATENCY,
            "streamid": streamid,
        },
        safe=safe_chars,
    )
    return f"srt://{MEDIAMTX_SERVER_IP}:{SRT_PORT_START}?{query}"


# ----------------------------- Shared state -----------------------------


@dataclass
class FramePacket:
    array_bgr: np.ndarray
    ts: float


class LastFrameStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._packet: Optional[FramePacket] = None

    def update(self, frame_bgr: np.ndarray) -> None:
        with self._lock:
            self._packet = FramePacket(array_bgr=frame_bgr.copy(), ts=time.time())

    def get(self) -> Optional[FramePacket]:
        with self._lock:
            return self._packet


class BoxStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._boxes: List[Tuple[int, int, int, int]] = []
        self._ts_src: float = 0.0

    def set(self, boxes: List[Tuple[int, int, int, int]], ts_src: float) -> None:
        with self._lock:
            self._boxes = list(boxes)
            self._ts_src = ts_src

    def get(self) -> Tuple[List[Tuple[int, int, int, int]], float]:
        with self._lock:
            return list(self._boxes), self._ts_src


# ----------------------------- FFmpeg helpers -----------------------------


def build_decoder_cmd(
    rtsp_url: str,
    width: int,
    height: int,
    fps: Optional[int] = 10,
    rtsp_transport: str = "tcp",
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
        "-pix_fmt",
        "bgr24",
        "-f",
        "rawvideo",
        "-s",
        f"{width}x{height}",
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
    x264_params: str = "scenecut=40:rc-lookahead=0:ref=3",
    sliced_threads: bool = True,
    enc_input_fps: int = 10,
) -> List[str]:
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
        f"{width}x{height}",
        "-framerate",
        str(max(1, enc_input_fps)),
        "-fflags",
        "+genpts",
        "-i",
        "pipe:0",
        "-an",
        "-pix_fmt",
        pix_fmt,
        "-c:v",
        "libx264",
        "-preset",
        preset,
        "-tune",
        tune,
        "-x264-params",
        x264_merged,
        "-bf",
        "0",
        "-g",
        str(g_val),
        "-threads",
        str(threads),
        "-mpegts_flags",
        "resend_headers",
        "-muxdelay",
        "0",
        "-muxpreload",
        "0",
    ]
    if use_crf:
        cmd += ["-crf", str(crf), "-maxrate", maxrate, "-bufsize", bufsize]
    else:
        cmd += ["-b:v", bitrate, "-maxrate", maxrate, "-bufsize", bufsize]
    cmd += ["-f", "mpegts", "-flush_packets", "1", srt_out]
    return cmd


def log_ffmpeg_stderr(proc: subprocess.Popen[bytes], name: str) -> None:
    if not proc.stderr:
        return
    for line in iter(proc.stderr.readline, b""):
        if not line:
            break
        try:
            logging.info("[%s] %s", name, line.decode(errors="ignore").rstrip())
        except Exception:
            pass


class FPSMeter:
    def __init__(self, name: str, log_every_s: float = 5.0) -> None:
        self.name = name
        self._lock = threading.Lock()
        self._count = 0
        self._t0 = time.time()
        self._last_log = self._t0
        self._ema: Optional[float] = None
        self._log_every = log_every_s

    def tick(self, n: int = 1) -> None:
        now = time.time()
        with self._lock:
            self._count += n
            dt = now - self._t0
            if dt <= 0:
                return
            inst = self._count / dt
            self._ema = inst if self._ema is None else 0.9 * self._ema + 0.1 * inst
            if now - self._last_log >= self._log_every:
                logging.info("FPS %s, current %.2f, smoothed %.2f", self.name, inst, self._ema or inst)
                self._last_log = now
                self._t0 = now
                self._count = 0


# ----------------------------- Vision utils -----------------------------


def boxes_px_from_norm(
    boxes_norm: Iterable[Sequence[float]],
    W: int,
    H: int,
    conf_th: float,
) -> List[Tuple[int, int, int, int]]:
    out = []
    for it in boxes_norm:
        if not it or len(it) < 4:
            continue
        x1, y1, x2, y2 = map(float, it[:4])
        conf = float(it[4]) if len(it) >= 5 else 1.0
        if conf < conf_th:
            continue
        x1p = int(max(0, min(W, x1 * W)))
        y1p = int(max(0, min(H, y1 * H)))
        x2p = int(max(0, min(W, x2 * W)))
        y2p = int(max(0, min(H, y2 * H)))
        if x2p > x1p and y2p > y1p:
            # clip again to W minus one and H minus one to be safe for slicing
            x1p = min(x1p, W - 1)
            y1p = min(y1p, H - 1)
            x2p = min(x2p, W)
            y2p = min(y2p, H)
            out.append((x1p, y1p, x2p, y2p))
    return out


def paint_black(arr_bgr: np.ndarray, boxes_px: List[Tuple[int, int, int, int]]) -> None:
    for x1, y1, x2, y2 in boxes_px:
        arr_bgr[y1:y2, x1:x2, :] = 0


# ----------------------------- Workers -----------------------------


class RTSPDecoderWorker:
    def __init__(
        self,
        rtsp_url: str,
        width: int,
        height: int,
        fps: int = 10,
        rtsp_transport: str = "tcp",
        dec_threads: int = 2,
        store: Optional[LastFrameStore] = None,
    ) -> None:
        self.width = width
        self.height = height
        self.frame_bytes = width * height * 3
        self.dec_cmd = build_decoder_cmd(
            rtsp_url=rtsp_url,
            width=width,
            height=height,
            fps=fps,
            rtsp_transport=rtsp_transport,
            dec_threads=dec_threads,
        )
        self._store = store or LastFrameStore()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._proc: Optional[subprocess.Popen[bytes]] = None
        self._fps = FPSMeter("decoder")

    @property
    def store(self) -> LastFrameStore:
        return self._store

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="decoder", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)

    def _open(self) -> None:
        logging.info("Starting decoder: %s", " ".join(self.dec_cmd))
        self._proc = subprocess.Popen(
            self.dec_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        threading.Thread(target=log_ffmpeg_stderr, args=(self._proc, "decoder"), daemon=True).start()

    def _close(self) -> None:
        if not self._proc:
            return
        try:
            self._proc.terminate()
        except Exception:
            pass
        try:
            self._proc.wait(timeout=2)
        except Exception:
            try:
                self._proc.kill()
            except Exception:
                pass
        self._proc = None

    def _run(self) -> None:
        buffer = bytearray(self.frame_bytes)
        view = memoryview(buffer)
        try:
            self._open()
            assert self._proc and self._proc.stdout
            dec_out: io.BufferedReader = cast(io.BufferedReader, self._proc.stdout)

            while not self._stop.is_set():
                n = 0
                while n < self.frame_bytes and not self._stop.is_set():
                    m = dec_out.readinto(view[n:])
                    if not m:
                        logging.warning("Decoder ended")
                        self._stop.set()
                        break
                    n += m
                if n < self.frame_bytes:
                    break

                frame = np.frombuffer(buffer, dtype=np.uint8).reshape((self.height, self.width, 3))
                self._store.update(frame)
                self._fps.tick()

        except BaseException as e:
            logging.error("Decoder worker error: %s", e)
        finally:
            self._close()
            logging.info("Decoder stopped")


class AnonymizerWorker:
    def __init__(
        self,
        frame_store: LastFrameStore,
        box_store: Optional[BoxStore] = None,
        conf_thres: float = 0.3,
        poll_ms: int = 10,
    ) -> None:
        self._frames = frame_store
        self._boxes = box_store or BoxStore()
        self._conf = conf_thres
        self._poll = max(1, poll_ms) / 1000.0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._model: Optional[Callable[[Image.Image], Iterable[Sequence[float]]]] = None
        self._last_ts: float = 0.0
        self._fps = FPSMeter("anonymizer")

    @property
    def boxes(self) -> BoxStore:
        return self._boxes

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="anonymizer", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)

    def _ensure_model(self) -> None:
        if self._model is None:
            logging.info("Loading Anonymizer model")
            self._model = Anonymizer()
            logging.info("Model ready")

    def _run(self) -> None:
        backoff = 1.0
        while not self._stop.is_set():
            try:
                pkt = self._frames.get()
                if pkt is None or pkt.ts <= self._last_ts:
                    time.sleep(self._poll)
                    continue

                self._ensure_model()
                rgb = cv2.cvtColor(pkt.array_bgr, cv2.COLOR_BGR2RGB)
                im = Image.fromarray(rgb)
                preds = self._model(im)  # Iterable of [x1 y1 x2 y2 conf] normalized
                boxes_px = boxes_px_from_norm(preds, im.width, im.height, self._conf)
                self._boxes.set(boxes_px, pkt.ts)
                self._last_ts = pkt.ts
                backoff = 1.0
                self._fps.tick()

            except BaseException as e:
                logging.warning("Anonymizer error: %s", e)
                self._model = None
                time.sleep(backoff)
                backoff = min(10.0, backoff * 2.0)
        logging.info("Anonymizer stopped")


class EncoderWorker:
    def __init__(
        self,
        frame_store: LastFrameStore,
        box_store: BoxStore,
        width: int,
        height: int,
        srt_out: str,
        target_fps: int = 10,
        x264_preset: str = "veryfast",
        x264_tune: str = "zerolatency",
        x264_params: Optional[str] = "scenecut=40:rc-lookahead=0:ref=3",
        bitrate: str = "700k",
        bufsize: str = "800k",
        maxrate: str = "900k",
        use_crf: bool = False,
        crf: int = 28,
        keyint: int = 14,
        pix_fmt: str = "yuv420p",
        enc_threads: int = 1,
    ) -> None:
        self._frames = frame_store
        self._boxes = box_store
        self.width = width
        self.height = height
        self.frame_bytes = width * height * 3
        self._interval = 1.0 / max(1, target_fps)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._proc: Optional[subprocess.Popen[bytes]] = None
        self._fps = FPSMeter("encoder")

        self.enc_cmd = build_encoder_cmd(
            srt_out=srt_out,
            width=width,
            height=height,
            keyint=keyint,
            use_crf=use_crf,
            crf=crf,
            bitrate=bitrate,
            bufsize=bufsize,
            maxrate=maxrate,
            threads=enc_threads,
            preset=x264_preset,
            tune=x264_tune,
            pix_fmt=pix_fmt,
            x264_params=x264_params or "scenecut=40:rc-lookahead=0:ref=3",
            enc_input_fps=int(round(1.0 / self._interval)),
        )

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="encoder", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)

    def _open(self) -> None:
        logging.info("Starting encoder: %s", " ".join(self.enc_cmd))
        self._proc = subprocess.Popen(
            self.enc_cmd,
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        threading.Thread(target=log_ffmpeg_stderr, args=(self._proc, "encoder"), daemon=True).start()

    def _close(self) -> None:
        if not self._proc:
            return
        try:
            if self._proc.stdin:
                self._proc.stdin.close()
        except Exception:
            pass
        try:
            self._proc.terminate()
        except Exception:
            pass
        try:
            self._proc.wait(timeout=2)
        except Exception:
            try:
                self._proc.kill()
            except Exception:
                pass
        self._proc = None

    def _run(self) -> None:
        try:
            self._open()
            assert self._proc and self._proc.stdin
            enc_in: io.BufferedWriter = self._proc.stdin  # type: ignore[assignment]

            next_deadline = time.time()
            while not self._stop.is_set():
                now = time.time()
                if now < next_deadline:
                    time.sleep(next_deadline - now)
                next_deadline += self._interval

                pkt = self._frames.get()
                if pkt is None:
                    continue

                frame = pkt.array_bgr.copy()
                boxes, ts_src = self._boxes.get()
                if boxes and ts_src <= pkt.ts + 0.5:
                    paint_black(frame, boxes)

                try:
                    wrote = enc_in.write(frame.tobytes())
                    self._fps.tick()
                    if wrote is None or wrote == 0:
                        raise BrokenPipeError("encoder write returned zero")
                except BrokenPipeError:
                    logging.warning("Encoder pipe closed")
                    self._stop.set()
                    break

        except BaseException as e:
            logging.error("Encoder worker error: %s", e)
        finally:
            self._close()
            logging.info("Encoder stopped")


# ----------------------------- Runner -----------------------------


def run_pipeline(
    rtsp_url: str,
    srt_out: str,
    width: int = 640,
    height: int = 360,
    fps: int = 10,
    conf_thres: float = 0.35,
) -> Tuple[RTSPDecoderWorker, AnonymizerWorker, EncoderWorker]:
    last_frames = LastFrameStore()
    boxes = BoxStore()

    decoder = RTSPDecoderWorker(rtsp_url=rtsp_url, width=width, height=height, fps=fps, store=last_frames)

    anonym = AnonymizerWorker(
        frame_store=last_frames,
        box_store=boxes,
        conf_thres=conf_thres,
    )

    encoder = EncoderWorker(
        frame_store=last_frames,
        box_store=boxes,
        width=width,
        height=height,
        srt_out=srt_out,
        target_fps=fps,
    )

    decoder.start()
    anonym.start()
    encoder.start()

    return decoder, anonym, encoder
