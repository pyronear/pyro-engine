#!/usr/bin/env python3

import argparse
import logging
import sys
import time
from rtsp_anonymize_srt import RTSPAnonymizeSRTWorker


def main() -> int:
    p = argparse.ArgumentParser(description="Run RTSP anonymizer → SRT streamer")
    p.add_argument("--rtsp", required=True)
    p.add_argument("--rtsp-transport", default="tcp")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=360)
    p.add_argument("--fps", type=int, default=7)

    p.add_argument("--srt", default=None)
    p.add_argument("--srt-host", default=None)
    p.add_argument("--srt-port", type=int, default=8890)
    p.add_argument("--streamid", default=None)

    p.add_argument("--conf", type=float, default=0.30)
    p.add_argument("--scale-div", type=int, default=1)

    # encoder params
    p.add_argument("--use-crf", action="store_true", default=False)
    p.add_argument("--crf", type=int, default=28)
    p.add_argument("--bitrate", default="700k")
    p.add_argument("--bufsize", default="800k")
    p.add_argument("--maxrate", default="900k")
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--preset", default="veryfast")
    p.add_argument("--tune", default="zerolatency")
    p.add_argument("--pix-fmt", default="yuv420p")
    p.add_argument("--x264-params", default="keyint=14:min-keyint=7:scenecut=40:rc-lookahead=0:ref=2")
    p.add_argument("--keyint", type=int, default=14)

    p.add_argument("--log-level", default="INFO")

    args = p.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s: %(message)s",
    )

    worker = RTSPAnonymizeSRTWorker(
        rtsp_url=args.rtsp,
        srt_out=args.srt,
        width=args.width,
        height=args.height,
        fps=args.fps,
        rtsp_transport=args.rtsp_transport,
        srt_host=args.srt_host,
        srt_port=args.srt_port,
        streamid=args.streamid,
        conf_thres=args.conf,
        model_scale_div=args.scale_div,
        x264_preset=args.preset,
        x264_tune=args.tune,
        bitrate=args.bitrate,
        bufsize=args.bufsize,
        maxrate=args.maxrate,
        use_crf=args.use_crf,
        crf=args.crf,
        keyint=args.keyint,
        pix_fmt=args.pix_fmt,
        enc_threads=args.threads,
    )

    worker.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        worker.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
