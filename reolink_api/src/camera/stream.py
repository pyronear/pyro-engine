# router.py

import logging
import threading
import time
from typing import Optional

from fastapi import APIRouter, HTTPException

from camera.config import FFMPEG_PARAMS, STREAMS, SRT_SETTINGS
from camera.registry import CAMERA_REGISTRY
from camera.time_utils import seconds_since_last_command, update_command_time

from rtsp_anonymize_srt import RTSPAnonymizeSRTWorker

router = APIRouter()
workers: dict[str, RTSPAnonymizeSRTWorker] = {}


def is_worker_running(w: Optional[RTSPAnonymizeSRTWorker]) -> bool:
    return bool(w and getattr(w, "_thread", None) and w._thread.is_alive())


def stop_any_running_stream() -> Optional[str]:
    for cam_id, w in list(workers.items()):
        if is_worker_running(w):
            try:
                w.stop()
            except Exception as e:
                logging.warning(f"Failed to stop worker for {cam_id}: {e}")
            del workers[cam_id]
            return cam_id
    return None


def stop_stream_if_idle():
    while True:
        time.sleep(10)
        if seconds_since_last_command() > 120:
            stopped = stop_any_running_stream()
            if stopped:
                logging.info(f"Stream for {stopped} stopped due to inactivity")


@router.post("/start_stream/{camera_ip}")
def start_stream(camera_ip: str):
    update_command_time()

    if camera_ip not in STREAMS:
        raise HTTPException(status_code=404, detail=f"No stream config for camera {camera_ip}")

    if camera_ip in workers and is_worker_running(workers[camera_ip]):
        logging.info(f"Stream for {camera_ip} already running")
        return {"message": f"Stream for {camera_ip} already running"}

    stopped_cam = stop_any_running_stream()

    stream_info = STREAMS[camera_ip]
    input_url = stream_info["input_url"]

    # SRT build from settings
    srt_host = stream_info.get("srt_host") or SRT_SETTINGS["host"]
    srt_port = int(stream_info.get("srt_port") or SRT_SETTINGS.get("port_start", 8890))
    streamid = f"{SRT_SETTINGS.get('streamid_prefix','publish')}:{camera_ip}"

    # Geometry and cadence
    width = int(FFMPEG_PARAMS.get("width", 640))
    height = int(FFMPEG_PARAMS.get("height", 360))
    fps = int(FFMPEG_PARAMS.get("framerate", 7))
    rtsp_transport = FFMPEG_PARAMS.get("rtsp_transport", "tcp")

    # Encoding knobs
    keyint = int(FFMPEG_PARAMS.get("gop_size", 14))
    threads = int(FFMPEG_PARAMS.get("threads", 1))
    preset = FFMPEG_PARAMS.get("preset", "veryfast")
    tune = FFMPEG_PARAMS.get("tune", "zerolatency")
    pix_fmt = FFMPEG_PARAMS.get("pix_fmt", "yuv420p")
    x264_params = FFMPEG_PARAMS.get("x264_params", "scenecut=40:rc-lookahead=0:ref=3")

    # Rate control
    use_crf = bool(FFMPEG_PARAMS.get("use_crf", True))
    crf = int(FFMPEG_PARAMS.get("crf", 22))
    bitrate = FFMPEG_PARAMS.get("bitrate", "500k")
    bufsize = FFMPEG_PARAMS.get("bufsize", "800k")
    maxrate = FFMPEG_PARAMS.get("maxrate", bitrate)

    # Anonymizer
    conf_thres = float(FFMPEG_PARAMS.get("anon_conf", 0.30))

    # SRT transport knobs
    pkt_size = int(SRT_SETTINGS.get("pkt_size", 1316))
    mode = SRT_SETTINGS.get("mode", "caller")
    latency = int(SRT_SETTINGS.get("latency", 50))

    worker = RTSPAnonymizeSRTWorker(
        rtsp_url=input_url,
        # geometry
        width=width,
        height=height,
        fps=fps,
        rtsp_transport=rtsp_transport,
        # anonymizer
        conf_thres=conf_thres,
        # encoder
        x264_preset=preset,
        x264_tune=tune,
        bitrate=bitrate,
        bufsize=bufsize,
        maxrate=maxrate,
        use_crf=use_crf,
        crf=crf,
        keyint=keyint,
        pix_fmt=pix_fmt,
        enc_threads=threads,
        # SRT
        srt_host=srt_host,
        srt_port=srt_port,
        streamid=streamid,
        srt_pkt_size=pkt_size,
        srt_latency=latency,
        # keep tlpktdrop at one for low delay
        srt_tlpktdrop=1,
    )

    workers[camera_ip] = worker
    logging.info(f"[{camera_ip}] Start worker, input {input_url}, srt {srt_host}:{srt_port} streamid {streamid}")
    worker.start()
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
    active = [cam_ip for cam_ip, w in workers.items() if is_worker_running(w)]
    if active:
        return {"active_streams": active}
    return {"message": "No stream is running"}


@router.get("/is_stream_running/{camera_ip}")
def is_stream_running(camera_ip: str):
    w = workers.get(camera_ip)
    return {"camera_ip": camera_ip, "running": bool(w and is_worker_running(w))}
