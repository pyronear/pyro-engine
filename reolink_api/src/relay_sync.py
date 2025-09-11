#!/usr/bin/env python3
# Minimal RTSP -> rawvideo pipe -> x264 -> SRT relay (low latency)

from __future__ import annotations
import argparse, logging, os, signal, subprocess, sys, threading, time

def build_srt_url(srt, host, port, streamid, pkt_size, latency, mode,
                  rcvlatency=None, peerlatency=None, tlpktdrop=1) -> str:
    if srt:
        return srt
    if not host:
        raise ValueError("--srt-host required when --srt is not provided")
    q = {
        "pkt_size": str(pkt_size),
        "mode": mode,
        "latency": str(latency),
    }
    if rcvlatency is not None: q["rcvlatency"] = str(rcvlatency)
    if peerlatency is not None: q["peerlatency"] = str(peerlatency)
    if tlpktdrop is not None: q["tlpktdrop"] = str(tlpktdrop)
    if streamid: q["streamid"] = streamid
    return f"srt://{host}:{port}?"+ "&".join(f"{k}={v}" for k,v in q.items())

def build_decoder_cmd(rtsp_url, width, height, rtsp_transport, fps,
                      analyzeduration, probesize, low_delay=True, discardcorrupt=True) -> list[str]:
    cmd = ["ffmpeg"]
    if analyzeduration is not None: cmd += ["-analyzeduration", str(analyzeduration)]
    if probesize is not None:       cmd += ["-probesize", str(probesize)]
    if discardcorrupt:              cmd += ["-fflags", "discardcorrupt"]
    cmd += ["-fflags", "nobuffer"]                      # keep tiny demux buffer
    if low_delay:                   cmd += ["-flags", "low_delay"]
    cmd += ["-use_wallclock_as_timestamps", "1"]
    cmd += ["-fflags", "+genpts"]                       # ensure PTS on decode side
    cmd += ["-rtsp_transport", rtsp_transport, "-i", rtsp_url]
    if fps and fps > 0:
        cmd += ["-r", str(fps), "-vsync", "1"]          # force cadence (tiny buffer but stable)
    else:
        cmd += ["-fps_mode", "passthrough"]             # keep camera cadence
    cmd += ["-an", "-pix_fmt", "bgr24", "-f", "rawvideo", "-s", f"{width}x{height}", "pipe:1"]
    return cmd

def build_encoder_cmd(srt_out, width, height, keyint, use_crf, crf, bitrate,
                      bufsize, maxrate, threads, preset, tune, pix_fmt,
                      x264_params, enc_input_fps) -> list[str]:
    cmd = [
        "ffmpeg",
        "-loglevel","warning","-nostats",
        "-f","rawvideo","-pix_fmt","bgr24","-s",f"{width}x{height}",
        "-framerate", str(max(1, enc_input_fps)),        # timestamp raw pipe
        "-i","pipe:0",
        "-an","-pix_fmt", pix_fmt,
        "-c:v","libx264","-preset", preset,"-tune", tune,
        "-g", str(keyint),                               # GOP here (single source of truth)
        "-x264-params", x264_params,                     # MUST NOT include keyint
        "-bf","0","-threads", str(threads),
        "-mpegts_flags","resend_headers",
        "-muxdelay","0","-muxpreload","0",
        "-flush_packets","1",                            # push ASAP
    ]
    if use_crf:
        cmd += ["-crf", str(crf), "-maxrate", maxrate, "-bufsize", bufsize]
    else:
        cmd += ["-b:v", bitrate, "-maxrate", maxrate, "-bufsize", bufsize]
    cmd += ["-f","mpegts", srt_out]
    return cmd

def log_ffmpeg(proc: subprocess.Popen, tag: str) -> None:
    if not proc.stderr: return
    for line in iter(proc.stderr.readline, b""):
        if not line: break
        try: logging.info("[%s] %s", tag, line.decode(errors="ignore").rstrip())
        except Exception: pass

def main() -> int:
    ap = argparse.ArgumentParser("Low-latency RTSP->SRT relay (no model)")
    # input
    ap.add_argument("--rtsp", required=True)
    ap.add_argument("--rtsp-transport", default="tcp", choices=["tcp","udp"])
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=360)
    ap.add_argument("--fps", type=int, default=10, help="Force output cadence; set 0 for passthrough")
    ap.add_argument("--analyzeduration", default="0")
    ap.add_argument("--probesize", default="32k")
    # output (SRT)
    ap.add_argument("--srt", default=None)
    ap.add_argument("--srt-host", default=None); ap.add_argument("--srt-port", type=int, default=8890)
    ap.add_argument("--streamid", default=None)
    ap.add_argument("--srt-mode", default="caller", choices=["caller","listener","rendezvous"])
    ap.add_argument("--srt-latency", type=int, default=30)
    ap.add_argument("--srt-rcvlatency", type=int, default=30)
    ap.add_argument("--srt-peerlatency", type=int, default=30)
    ap.add_argument("--srt-pkt-size", type=int, default=1316)
    ap.add_argument("--srt-tlpktdrop", type=int, default=1)
    # encoder
    ap.add_argument("--keyint", type=int, default=10)
    ap.add_argument("--use-crf", action="store_true", default=True)
    ap.add_argument("--crf", type=int, default=22)
    ap.add_argument("--bitrate", default="700k")
    ap.add_argument("--bufsize", default="450k")
    ap.add_argument("--maxrate", default="900k")
    ap.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 1)//2))
    ap.add_argument("--preset", default="veryfast")
    ap.add_argument("--tune", default="zerolatency")
    ap.add_argument("--pix-fmt", default="yuv420p")
    ap.add_argument("--x264-params",
        default="min-keyint=10:scenecut=0:rc-lookahead=0:ref=1:frame-threads=1:sliced-threads=1")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(asctime)s | %(levelname)s: %(message)s")

    W, H = args.width, args.height
    frame_bytes = W * H * 3

    srt_out = build_srt_url(
        args.srt, args.srt_host, args.srt_port, args.streamid,
        args.srt_pkt_size, args.srt_latency, args.srt_mode,
        args.srt_rcvlatency, args.srt_peerlatency, args.srt_tlpktdrop,
    )

    dec_cmd = build_decoder_cmd(
        args.rtsp, W, H, args.rtsp_transport,
        args.fps if args.fps > 0 else None,
        args.analyzeduration, args.probesize,
        low_delay=True, discardcorrupt=True,
    )
    enc_cmd = build_encoder_cmd(
        srt_out, W, H, args.keyint, args.use_crf, args.crf,
        args.bitrate, args.bufsize, args.maxrate, args.threads,
        args.preset, args.tune, args.pix_fmt, args.x264_params,
        args.fps if args.fps > 0 else 10,
    )

    logging.info("Starting decoder: %s", " ".join(dec_cmd))
    dec = subprocess.Popen(dec_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)

    logging.info("Starting encoder: %s", " ".join(enc_cmd))
    enc = subprocess.Popen(enc_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)

    threading.Thread(target=log_ffmpeg, args=(dec,"decoder"), daemon=True).start()
    threading.Thread(target=log_ffmpeg, args=(enc,"encoder"), daemon=True).start()

    stop = threading.Event()
    def on_sig(signum, frame):
        logging.info("Signal received, stopping…"); stop.set()
    signal.signal(signal.SIGINT, on_sig); signal.signal(signal.SIGTERM, on_sig)

    try:
        assert dec.stdout is not None and enc.stdin is not None
        buf = bytearray(frame_bytes); view = memoryview(buf)
        total = 0; t0 = time.perf_counter(); target_fps = args.fps if args.fps>0 else 10

        while not stop.is_set():
            # read exactly one frame of rawvideo from decoder
            n = 0
            while n < frame_bytes and not stop.is_set():
                chunk = dec.stdout.read(frame_bytes - n)
                if not chunk:
                    logging.warning("Decoder ended"); stop.set(); break
                view[n:n+len(chunk)] = chunk; n += len(chunk)
            if n < frame_bytes: break

            # write one frame to encoder
            try:
                enc.stdin.write(view)
            except BrokenPipeError:
                logging.warning("Encoder pipe closed"); stop.set(); break

            # simple pacing to avoid stdin buffering (keeps latency tiny)
            if target_fps > 0:
                if total == 0: t0 = time.perf_counter()
                total += 1
                next_deadline = t0 + total/float(target_fps)
                delay = next_deadline - time.perf_counter()
                if delay > 0: time.sleep(delay)
            else:
                total += 1

            if total % 50 == 0:
                dt = max(1e-3, time.perf_counter()-t0)
                logging.info("Relayed frames: %d (%.2f fps)", total, total/dt)

    finally:
        try:
            if enc.stdin: enc.stdin.close()
        except Exception: pass
        for p in (enc, dec):
            try: p.terminate()
            except Exception: pass
            try: p.wait(timeout=2)
            except Exception:
                try: p.kill()
                except Exception: pass
        logging.info("Stopped")
    return 0

if __name__ == "__main__":
    sys.exit(main())
