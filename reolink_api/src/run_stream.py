#!/usr/bin/env python3
# Copyright (C) 2020-2025, Pyronear.
# Licensed under the Apache License 2.0.

from __future__ import annotations

import argparse
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image
from anonymizer.vision import Anonymizer


# ----------------------------- SRT helpers -----------------------------

def build_srt_url(
    srt: Optional[str],
    host: Optional[str],
    port: int,
    streamid: Optional[str],
    pkt_size: int,
    latency: int,
    mode: str,
    rcvlatency: Optional[int] = None,
    peerlatency: Optional[int] = None,
    tlpktdrop: Optional[int] = 1,
) -> str:
    if srt:
        return srt
    if not host:
        raise ValueError("SRT host is required when --srt is not provided")
    params = {
        "pkt_size": str(pkt_size),
        "mode": mode,
        "latency": str(latency),
    }
    if rcvlatency is not None:
        params["rcvlatency"] = str(rcvlatency)
    if peerlatency is not None:
        params["peerlatency"] = str(peerlatency)
    if tlpktdrop is not None:
        params["tlpktdrop"] = str(tlpktdrop)
    if streamid:
        params["streamid"] = streamid
    query = "&".join(f"{k}={v}" for k, v in params.items())
    return f"srt://{host}:{port}?{query}"


# ----------------------------- FFmpeg cmds -----------------------------

def build_decoder_cmd(
    rtsp_url: str,
    width: int,
    height: int,
    rtsp_transport: str,
    fps: Optional[int],
    analyzeduration: str,
    probesize: str,
    low_delay: bool,
    discardcorrupt: bool,
    dec_threads: int,
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
        # do not re-time, pass input cadence through
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
    keyint: int,
    use_crf: bool,
    crf: int,
    bitrate: str,
    bufsize: str,
    maxrate: str,
    threads: int,
    preset: str,
    tune: str,
    pix_fmt: str,
    x264_params: str,
    frame_threads: int = 1,
    sliced_threads: bool = True,
) -> List[str]:
    params = x264_params.split(":") if x264_params else []
    need = {
        "keyint": str(keyint),
        "min-keyint": str(max(1, keyint // 2)),
        "scenecut": "0",
        "rc-lookahead": "0",
        "frame-threads": str(frame_threads),
        "ref": "1",
    }
    if sliced_threads:
        need["sliced-threads"] = "1"
    have = {p.split("=")[0] for p in params if "=" in p}
    for k, v in need.items():
        if k not in have:
            params.insert(0, f"{k}={v}")
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
        "-fflags", "+genpts",
        "-i", "pipe:0",
        "-an",
        "-pix_fmt", pix_fmt,
        "-c:v", "libx264",
        "-preset", preset,
        "-tune", tune,
        "-x264-params", x264_merged,
        "-bf", "0",
        "-g", str(keyint),
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


# ----------------------------- Shared states -----------------------------

class LatestFrame:
    def __init__(self) -> None:
        self._im: Optional[Image.Image] = None
        self._ts: float = 0.0
        self._lock = threading.Lock()
        self._event = threading.Event()
    def update(self, im: Image.Image) -> None:
        with self._lock:
            self._im = im
            self._ts = time.perf_counter()
        self._event.set()
    def wait_and_get(self, timeout: float = 0.1) -> Tuple[Optional[Image.Image], float]:
        if not self._event.wait(timeout=timeout):
            return None, 0.0
        self._event.clear()
        with self._lock:
            return self._im, self._ts


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


class ModelStatsState:
    def __init__(self) -> None:
        self._last_infer_ms: float = 0.0
        self._last_stale_ms: float = 0.0
        self._last_boxes: int = 0
        self._lock = threading.Lock()
    def set(self, infer_ms: float, stale_ms: float, boxes: int) -> None:
        with self._lock:
            self._last_infer_ms = infer_ms
            self._last_stale_ms = stale_ms
            self._last_boxes = boxes
    def get(self) -> Tuple[float, float, int]:
        with self._lock:
            return self._last_infer_ms, self._last_stale_ms, self._last_boxes


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
        x1, y1, x2, y2 = map(float, it[:4])  # normalized
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


def paint_black(arr: np.ndarray, boxes_px: List[Tuple[int, int, int, int]]) -> float:
    t0 = time.perf_counter()
    for x1, y1, x2, y2 in boxes_px:
        arr[y1:y2, x1:x2, :] = 0
    return (time.perf_counter() - t0) * 1000.0


# ----------------------------- Model thread -----------------------------

def anonymizer_thread_fn(
    latest: LatestFrame,
    boxes_state: BoxState,
    model_stats: ModelStatsState,
    conf_thres: float,
    model_scale_div: int,
    stop_event: threading.Event,
) -> None:
    backoff_s = 1.0
    model: Optional[Anonymizer] = None
    while not stop_event.is_set():
        try:
            if model is None:
                logging.info("Loading Anonymizer model")
                model = Anonymizer()
                logging.info("Model ready")

            im, stamp = latest.wait_and_get(timeout=0.1)
            if im is None:
                continue

            stale_ms = (time.perf_counter() - stamp) * 1000.0

            small = im if model_scale_div <= 1 else im.resize(
                (max(1, im.width // model_scale_div), max(1, im.height // model_scale_div)),
                Image.BILINEAR,
            )

            t0 = time.perf_counter()
            preds = model(small)
            infer_ms = (time.perf_counter() - t0) * 1000.0

            boxes_px = boxes_px_from_norm(preds, im.width, im.height, conf_thres)
            boxes_state.set(boxes_px)
            model_stats.set(infer_ms, stale_ms, len(boxes_px))

        except BaseException as e:
            logging.warning("Model thread error: %s", e)
            model = None
            time.sleep(min(backoff_s, 10.0))
            backoff_s = min(backoff_s * 2.0, 10.0)


# ----------------------------- Main -----------------------------

def main() -> int:
    p = argparse.ArgumentParser(description="RTSP to SRT with background anonymization and timing")
    # input
    p.add_argument("--rtsp", required=True)
    p.add_argument("--rtsp-transport", default="tcp")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=360)
    p.add_argument("--fps", type=int, default=7)
    p.add_argument("--analyzeduration", default="0")
    p.add_argument("--probesize", default="32k")
    p.add_argument("--dec-threads", type=int, default=max(2, os.cpu_count() or 2))
    # output
    p.add_argument("--srt", default=None)
    p.add_argument("--srt-host", default=None)
    p.add_argument("--srt-port", type=int, default=8890)
    p.add_argument("--streamid", default=None)
    p.add_argument("--srt-mode", default="caller", choices=["caller", "listener", "rendezvous"])
    p.add_argument("--srt-latency", type=int, default=50)
    p.add_argument("--srt-pkt-size", type=int, default=1316)
    p.add_argument("--srt-rcvlatency", type=int, default=None)
    p.add_argument("--srt-peerlatency", type=int, default=None)
    p.add_argument("--srt-tlpktdrop", type=int, default=1)
    # encoder
    p.add_argument("--keyint", type=int, default=14)
    p.add_argument("--use-crf", action="store_true", default=False)
    p.add_argument("--crf", type=int, default=28)
    p.add_argument("--bitrate", default="700k")
    p.add_argument("--bufsize", default="800k")
    p.add_argument("--maxrate", default="900k")
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--preset", default="veryfast")
    p.add_argument("--tune", default="zerolatency")
    p.add_argument("--pix-fmt", default="yuv420p")
    p.add_argument("--x264-params", default="keyint=14:min-keyint=7:scenecut=40:rc-lookahead=0:ref=2:aq-mode=2")
    p.add_argument("--frame-threads", type=int, default=1)
    p.add_argument("--slice-threads", action="store_true", default=True)
    # detection
    p.add_argument("--conf", type=float, default=0.30)
    p.add_argument("--scale-div", type=int, default=1)
    # metrics
    p.add_argument("--metrics-every", type=float, default=1.0, help="seconds between reports")
    p.add_argument("--metrics-csv", default=None, help="optional path to write CSV rows")
    # misc
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s: %(message)s",
    )

    W, H = args.width, args.height
    frame_bytes = W * H * 3

    srt_out = build_srt_url(
        srt=args.srt,
        host=args.srt_host,
        port=args.srt_port,
        streamid=args.streamid,
        pkt_size=args.srt_pkt_size,
        latency=args.srt_latency,
        mode=args.srt_mode,
        rcvlatency=args.srt_rcvlatency,
        peerlatency=args.srt_peerlatency,
        tlpktdrop=args.srt_tlpktdrop,
    )

    dec_cmd = build_decoder_cmd(
        rtsp_url=args.rtsp,
        width=W,
        height=H,
        rtsp_transport=args.rtsp_transport,
        fps=args.fps,
        analyzeduration=args.analyzeduration,
        probesize=args.probesize,
        low_delay=True,
        discardcorrupt=True,
        dec_threads=args.dec_threads,
    )

    enc_threads = args.threads
    x264_params = args.x264_params
    if args.slice_threads and "sliced-threads" not in x264_params:
        x264_params = f"sliced-threads=1:{x264_params}"
    if "frame-threads" not in x264_params:
        x264_params = f"frame-threads={args.frame_threads}:{x264_params}"

    enc_cmd = build_encoder_cmd(
        srt_out=srt_out,
        width=W,
        height=H,
        keyint=args.keyint,
        use_crf=args.use_crf,
        crf=args.crf,
        bitrate=args.bitrate,
        bufsize=args.bufsize,
        maxrate=args.maxrate,
        threads=enc_threads,
        preset=args.preset,
        tune=args.tune,
        pix_fmt=args.pix_fmt,
        x264_params=x264_params,
        frame_threads=args.frame_threads,
        sliced_threads=args.slice_threads,
    )

    logging.info("Starting decoder: %s", " ".join(dec_cmd))
    dec = subprocess.Popen(dec_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
    logging.info("Starting encoder: %s", " ".join(enc_cmd))
    enc = subprocess.Popen(enc_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)

    threading.Thread(target=log_ffmpeg_stderr, args=(dec, "decoder"), daemon=True).start()
    threading.Thread(target=log_ffmpeg_stderr, args=(enc, "encoder"), daemon=True).start()

    latest = LatestFrame()
    boxes_state = BoxState()
    model_stats = ModelStatsState()
    stop_event = threading.Event()

    def handle_sig(signum, frame):
        logging.info("Signal received, stopping")
        stop_event.set()

    signal.signal(signal.SIGINT, handle_sig)
    signal.signal(signal.SIGTERM, handle_sig)

    threading.Thread(
        target=anonymizer_thread_fn,
        args=(latest, boxes_state, model_stats, args.conf, args.scale_div, stop_event),
        daemon=True,
        name="model-thread",
    ).start()

    buffer = bytearray(frame_bytes)
    view = memoryview(buffer)

    # metrics accumulators over a window
    win_start = time.perf_counter()
    acc_frames = 0
    acc_read_ms = 0.0
    acc_paint_ms = 0.0
    acc_write_ms = 0.0
    acc_frame_ms = 0.0

    csv_fp = None
    if args.metrics_csv:
        csv_fp = open(args.metrics_csv, "a", buffering=1)
        if csv_fp.tell() == 0:
            csv_fp.write("ts,read_ms,paint_ms,write_ms,frame_ms,fps,model_ms,model_stale_ms,boxes\n")

    try:
        assert dec.stdout is not None
        assert enc.stdin is not None

        while not stop_event.is_set():
            loop_t0 = time.perf_counter()

            # read one frame worth of bytes
            n = 0
            while n < frame_bytes and not stop_event.is_set():
                chunk = dec.stdout.read(frame_bytes - n)
                if not chunk:
                    logging.warning("Decoder ended")
                    stop_event.set()
                    break
                view[n:n + len(chunk)] = chunk
                n += len(chunk)
            if n < frame_bytes:
                break

            t_after_read = time.perf_counter()
            read_ms = (t_after_read - loop_t0) * 1000.0

            frame = np.frombuffer(buffer, dtype=np.uint8).reshape((H, W, 3))

            # give latest RGB to model
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if args.scale_div > 1:
                rgb = cv2.resize(rgb, (W // args.scale_div, H // args.scale_div), interpolation=cv2.INTER_AREA)
            latest.update(Image.fromarray(rgb))

            # paint current boxes
            boxes = boxes_state.get()
            paint_ms = paint_black(frame, boxes) if boxes else 0.0

            # write to encoder
            t_before_write = time.perf_counter()
            try:
                enc.stdin.write(buffer)
            except BrokenPipeError:
                logging.warning("Encoder pipe closed")
                stop_event.set()
                break
            write_ms = (time.perf_counter() - t_before_write) * 1000.0

            frame_ms = (time.perf_counter() - loop_t0) * 1000.0

            # accumulate
            acc_frames += 1
            acc_read_ms += read_ms
            acc_paint_ms += paint_ms
            acc_write_ms += write_ms
            acc_frame_ms += frame_ms

            # periodic metrics
            now = time.perf_counter()
            if now - win_start >= args.metrics_every:
                fps = acc_frames / (now - win_start) if acc_frames else 0.0
                model_ms, stale_ms, box_count = model_stats.get()
                logging.info(
                    "metrics read_ms=%.1f paint_ms=%.1f write_ms=%.1f frame_ms=%.1f fps=%.2f model_ms=%.1f model_stale_ms=%.1f boxes=%d",
                    acc_read_ms / max(1, acc_frames),
                    acc_paint_ms / max(1, acc_frames),
                    acc_write_ms / max(1, acc_frames),
                    acc_frame_ms / max(1, acc_frames),
                    fps,
                    model_ms,
                    stale_ms,
                    box_count,
                )
                if csv_fp:
                    csv_fp.write(
                        f"{time.time():.3f},{acc_read_ms/max(1,acc_frames):.3f},{acc_paint_ms/max(1,acc_frames):.3f},"
                        f"{acc_write_ms/max(1,acc_frames):.3f},{acc_frame_ms/max(1,acc_frames):.3f},{fps:.3f},"
                        f"{model_ms:.3f},{stale_ms:.3f},{box_count}\n"
                    )
                # reset window
                win_start = now
                acc_frames = 0
                acc_read_ms = acc_paint_ms = acc_write_ms = acc_frame_ms = 0.0

    finally:
        try:
            if enc.stdin:
                enc.stdin.close()
        except Exception:
            pass
        for proc in (enc, dec):
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
        if csv_fp:
            csv_fp.close()
        logging.info("Stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
