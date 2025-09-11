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

# ----------------------------- SRT helpers -----------------------------

def build_srt_url(
    srt: str | None,
    host: str | None,
    port: int,
    streamid: str | None,
    pkt_size: int,
    latency: int,
    mode: str,
    rcvlatency: int | None = None,
    peerlatency: int | None = None,
    tlpktdrop: int | None = 1,
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
    fps: int | None,
    analyzeduration: str,
    probesize: str,
    low_delay: bool,
    discardcorrupt: bool,
) -> list[str]:
    cmd = ["ffmpeg"]
    if analyzeduration is not None:
        cmd += ["-analyzeduration", str(analyzeduration)]
    if probesize is not None:
        cmd += ["-probesize", str(probesize)]

    # decoder flags
    if discardcorrupt:
        cmd += ["-fflags", "discardcorrupt"]
    cmd += ["-fflags", "nobuffer"]
    cmd += ["-fflags", "+genpts"]   # keep PTS generation on the demuxer side
    if low_delay:
        cmd += ["-flags", "low_delay"]

    cmd += ["-use_wallclock_as_timestamps", "1"]
    cmd += ["-rtsp_transport", rtsp_transport, "-i", rtsp_url]

    if fps and fps > 0:
        cmd += ["-r", str(fps), "-vsync", "1"]
    else:
        cmd += ["-fps_mode", "passthrough"]

    cmd += ["-an", "-pix_fmt", "bgr24", "-f", "rawvideo", "-s", f"{width}x{height}", "pipe:1"]
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
    enc_input_fps: int,
) -> list[str]:
    # merge a few safe low-latency defaults if missing

    cmd = [
        "ffmpeg",
        "-loglevel", "warning",
        "-nostats",
        # DO NOT set -fflags on the rawvideo input here
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}",
        "-framerate", str(max(1, enc_input_fps)),  # gives timestamps to rawvideo
        "-i", "pipe:0",
        "-an",
        "-pix_fmt", pix_fmt,
        "-c:v", "libx264",
        "-preset", preset,
        "-tune", tune,
        "-g", str(keyint),               # single source of truth for GOP
        "-x264-params", x264_params,     # no keyint here
        "-bf", "0",
        "-threads", str(threads),
        "-mpegts_flags", "resend_headers",
        "-muxdelay", "0",
        "-muxpreload", "0",
        "-flush_packets", "1",           # optional: flush immediately
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

# ----------------------------- Main -----------------------------

def main() -> int:
    p = argparse.ArgumentParser(description="Decode RTSP and immediately publish to SRT (no anonymization).")
    # input
    p.add_argument("--rtsp", required=True)
    p.add_argument("--rtsp-transport", default="tcp", choices=["tcp", "udp"])
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=360)
    p.add_argument("--fps", type=int, default=0, help="0 = passthrough; otherwise force output cadence")
    p.add_argument("--analyzeduration", default="0")
    p.add_argument("--probesize", default="32k")
    # output (SRT)
    p.add_argument("--srt", default=None, help="Full SRT URL; overrides host/port/streamid if set")
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
    p.add_argument("--use-crf", action="store_true", default=True)
    p.add_argument("--crf", type=int, default=22)
    p.add_argument("--bitrate", default="700k")
    p.add_argument("--bufsize", default="450k")
    p.add_argument("--maxrate", default="900k")
    p.add_argument("--threads", type=int, default=max(2, (os.cpu_count() or 2)))
    p.add_argument("--preset", default="veryfast")
    p.add_argument("--tune", default="zerolatency")
    p.add_argument("--pix-fmt", default="yuv420p")
    p.add_argument("--keyint", type=int, default=10, help="GOP length (frames)")
    p.add_argument("--x264-params",
        default="min-keyint=10:scenecut=0:rc-lookahead=0:ref=1:frame-threads=1:sliced-threads=1")

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
        fps=args.fps if args.fps > 0 else None,
        analyzeduration=args.analyzeduration,
        probesize=args.probesize,
        low_delay=True,
        discardcorrupt=True,
    )

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
        threads=args.threads,
        preset=args.preset,
        tune=args.tune,
        pix_fmt=args.pix_fmt,
        x264_params=args.x264_params,
        enc_input_fps=args.fps if args.fps > 0 else 10,  # give timestamps to rawvideo
    )

    logging.info("Starting decoder: %s", " ".join(dec_cmd))
    dec = subprocess.Popen(dec_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)

    logging.info("Starting encoder: %s", " ".join(enc_cmd))
    enc = subprocess.Popen(enc_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)

    threading.Thread(target=log_ffmpeg_stderr, args=(dec, "decoder"), daemon=True).start()
    threading.Thread(target=log_ffmpeg_stderr, args=(enc, "encoder"), daemon=True).start()

    stop = threading.Event()

    def on_sig(signum, frame):
        logging.info("Signal received, stopping…")
        stop.set()

    signal.signal(signal.SIGINT, on_sig)
    signal.signal(signal.SIGTERM, on_sig)

    try:
        assert dec.stdout is not None
        assert enc.stdin is not None

        view = memoryview(bytearray(frame_bytes))
        total = 0
        t0 = time.perf_counter()
        target_fps = args.fps if args.fps > 0 else 10

        while not stop.is_set():
            # read one raw frame from the decoder
            n = 0
            while n < frame_bytes and not stop.is_set():
                chunk = dec.stdout.read(frame_bytes - n)
                if not chunk:
                    logging.warning("Decoder ended")
                    stop.set()
                    break
                view[n : n + len(chunk)] = chunk
                n += len(chunk)
            if n < frame_bytes:
                break

            # write the frame to the encoder
            try:
                enc.stdin.write(view)
                total += 1
                if total % 50 == 0:
                    dt = max(1e-3, time.perf_counter() - t0)
                    logging.info("Relayed frames: %d (%.2f fps)", total, total / dt)

            except BrokenPipeError:
                logging.warning("Encoder pipe closed")
                stop.set()
                break

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
        logging.info("Stopped")
    return 0

if __name__ == "__main__":
    sys.exit(main())
