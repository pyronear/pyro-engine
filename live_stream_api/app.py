import json
import logging
import os
import subprocess
import threading
import time
from io import BytesIO
from typing import Optional

import urllib3
import yaml
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from reolink import ReolinkCamera

app = FastAPI()
processes = {}  # Store FFmpeg processes
last_command_time = time.time()  # Track the last command time
timer_thread = None  # Background thread for checking inactivity

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
logging.basicConfig(level=logging.DEBUG)


# Load ffmpeg config (YAML)
with open("ffmpeg_config.yaml", "r") as file:
    ffmpeg_config = yaml.safe_load(file)

SRT_SETTINGS = ffmpeg_config["srt_settings"]
FFMPEG_PARAMS = ffmpeg_config["ffmpeg_params"]


# Load environment variables
load_dotenv()

CAM_USER = os.getenv("CAM_USER")
CAM_PWD = os.getenv("CAM_PWD")
MEDIAMTX_SERVER_IP = os.getenv("MEDIAMTX_SERVER_IP")
STREAM_NAME = os.getenv("STREAM_NAME")

# Load camera credentials
CREDENTIALS_PATH = "/usr/src/app/data/credentials.json"

if not os.path.exists(CREDENTIALS_PATH):
    raise FileNotFoundError(f"Credentials file not found at {CREDENTIALS_PATH}")

with open(CREDENTIALS_PATH, "r") as file:
    credentials = json.load(file)


# Build cameras dictionary
CAMERAS = {
    ip: {"ip": ip, "username": CAM_USER, "password": CAM_PWD, "brand": credentials[ip].get("brand", "unknown")}
    for ip in credentials
}

CAMERA_OBJECTS = {
    ip: ReolinkCamera(
        ip_address=ip,
        username=cam_info["username"],
        password=cam_info["password"],
        cam_poses=credentials[ip].get("poses"),
        cam_azimuths=credentials[ip].get("azimuths"),
    )
    for ip, cam_info in CAMERAS.items()
}

# Build streams dictionary using config values
STREAMS = {
    cam_id: {
        "input_url": f"rtsp://{CAM_USER}:{CAM_PWD}@{cam_info['ip']}:554/h264Preview_01_sub",
        "output_url": (
            f"srt://{MEDIAMTX_SERVER_IP}:{SRT_SETTINGS['port_start']}?"
            f"pkt_size={SRT_SETTINGS['pkt_size']}&"
            f"mode={SRT_SETTINGS['mode']}&"
            f"latency={SRT_SETTINGS['latency']}&"
            f"streamid={SRT_SETTINGS['streamid_prefix']}:{STREAM_NAME}"
        ),
    }
    for cam_id, cam_info in CAMERAS.items()
}


def is_process_running(proc):
    """Check if a process is still running."""
    return proc and proc.poll() is None


def log_ffmpeg_output(proc, camera_id):
    """Reads and logs stderr from ffmpeg."""
    for line in proc.stderr:
        logging.error(f"[FFMPEG {camera_id}] {line.decode('utf-8').strip()}")


def stop_any_running_stream():
    """Stops any currently running stream."""
    for cam_id, proc in list(processes.items()):
        if is_process_running(proc):
            proc.terminate()
            proc.wait()
            del processes[cam_id]
            return cam_id
    return None


def stop_stream_if_idle():
    """Background task that stops the stream if no command is received for 120 seconds."""
    global last_command_time
    while True:
        time.sleep(120)
        if time.time() - last_command_time > 120:
            stopped_cam = stop_any_running_stream()
            if stopped_cam:
                logging.info(f"Stream for {stopped_cam} stopped due to inactivity")


# Start background thread
timer_thread = threading.Thread(target=stop_stream_if_idle, daemon=True)
timer_thread.start()


@app.post("/start_stream/{camera_id}")
async def start_stream(camera_id: str):
    """Starts an FFmpeg stream for a given camera (unless it's already running)."""
    global last_command_time
    last_command_time = time.time()

    if camera_id not in STREAMS:
        return {"error": "Invalid camera ID."}

    # ✅ Check if stream is already running
    existing_proc = processes.get(camera_id)
    if is_process_running(existing_proc):
        logging.info(f"Stream for {camera_id} is already running — skipping restart.")
        return {"message": f"Stream for {camera_id} is already running."}

    # Stop any existing stream (only if different one was running)
    stopped_cam = stop_any_running_stream()

    stream_info = STREAMS[camera_id]
    input_url = stream_info["input_url"]
    output_url = stream_info["output_url"]

    # Build ffmpeg command dynamically based on config
    command = ["ffmpeg", "-hide_banner", "-loglevel", "warning"]

    if FFMPEG_PARAMS["discardcorrupt"]:
        command += ["-fflags", "discardcorrupt+nobuffer"]
    if FFMPEG_PARAMS["low_delay"]:
        command += ["-flags", "low_delay"]

    command += [
        "-rtsp_transport",
        FFMPEG_PARAMS["rtsp_transport"],
        "-i",
        input_url,
        "-c:v",
        FFMPEG_PARAMS["video_codec"],
        "-bf",
        str(FFMPEG_PARAMS["b_frames"]),
        "-g",
        str(FFMPEG_PARAMS["gop_size"]),
        "-b:v",
        FFMPEG_PARAMS["bitrate"],
        "-r",
        str(FFMPEG_PARAMS["framerate"]),
        "-preset",
        FFMPEG_PARAMS["preset"],
        "-tune",
        FFMPEG_PARAMS["tune"],
        "-flush_packets",
        "1",
    ]

    if FFMPEG_PARAMS["audio_disabled"]:
        command.append("-an")

    command += ["-f", FFMPEG_PARAMS["output_format"], output_url]

    logging.info("Running ffmpeg command: %s", " ".join(command))

    # Start ffmpeg process
    proc = subprocess.Popen(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )

    # Store and monitor the process
    processes[camera_id] = proc
    threading.Thread(target=log_ffmpeg_output, args=(proc, camera_id), daemon=True).start()

    return {
        "message": f"Stream for {camera_id} started",
        "previous_stream": (stopped_cam or "No previous stream was running"),
    }


@app.post("/stop_stream")
async def stop_stream():
    """Stops any active stream."""
    global last_command_time
    last_command_time = time.time()

    stopped_cam = stop_any_running_stream()
    if stopped_cam:
        return {"message": f"Stream for {stopped_cam} stopped"}
    return {"message": "No active stream was running"}


@app.get("/status")
async def stream_status():
    """Returns which stream is currently running."""
    global last_command_time
    last_command_time = time.time()

    active_streams = [cam_id for cam_id, proc in processes.items() if is_process_running(proc)]
    if active_streams:
        return {"active_streams": active_streams}
    return {"message": "No stream is running"}


# Lookup dictionaries for pan speeds
PAN_SPEEDS = {
    "reolink-823S2": {1: 1.4723, 2: 2.7747, 3: 4.2481, 4: 5.6113, 5: 7.3217},
    "reolink-823A16": {1: 1.4403, 2: 2.714, 3: 4.1801, 4: 5.6259, 5: 7.2743},
}

# Lookup dictionaries for tilt speeds
TILT_SPEEDS = {"reolink-823S2": {1: 2.1392, 2: 3.9651, 3: 6.0554}, "reolink-823A16": {1: 1.7998, 2: 3.6733, 3: 5.5243}}


def get_pan_speed_per_sec(brand: str, level: int) -> Optional[float]:
    return PAN_SPEEDS.get(brand, {}).get(level)


def get_tilt_speed_per_sec(brand: str, level: int) -> Optional[float]:
    return TILT_SPEEDS.get(brand, {}).get(level)


@app.post("/move/{camera_id}")
async def move_camera(
    camera_id: str,
    direction: Optional[str] = None,
    speed: int = 10,
    pose_id: Optional[int] = None,
    degrees: Optional[float] = None,
):
    """
    Moves the camera:
    - If 'pose_id' is provided, move to the preset pose.
    - If 'degrees' is provided, move that many degrees in the given direction.
    - Otherwise, move in the specified 'direction' (Up, Down, Left, Right).
    """
    global last_command_time
    last_command_time = time.time()

    if camera_id not in CAMERAS or camera_id not in CAMERA_OBJECTS:
        return {"error": "Invalid camera ID."}

    cam_info = CAMERAS[camera_id]
    cam = CAMERA_OBJECTS[camera_id]
    brand = cam_info.get("brand", "unknown")

    try:
        if pose_id is not None:
            logging.info(f"Camera {camera_id}: moving to preset pose {pose_id} at speed {speed}")
            cam.move_camera("ToPos", speed=speed, idx=pose_id)
            return {"message": f"Camera {camera_id} moved to pose {pose_id} at speed {speed}"}

        if degrees is not None and direction:
            if direction in ["Left", "Right"]:
                deg_per_sec = get_pan_speed_per_sec(brand, speed)
            elif direction in ["Up", "Down"]:
                deg_per_sec = get_tilt_speed_per_sec(brand, speed)
            else:
                return {"error": f"Unsupported direction '{direction}'."}

            if deg_per_sec is None:
                return {"error": f"Unsupported brand '{brand}' or speed level {speed}."}

            duration_sec = abs(degrees) / deg_per_sec
            logging.info(
                f"Camera {camera_id}: moving {direction} for {duration_sec:.2f}s at speed {speed} (brand={brand})"
            )

            cam.move_camera(direction, speed=speed)
            time.sleep(duration_sec)
            cam.move_camera("Stop")

            logging.info(f"Camera {camera_id}: movement {direction} stopped after ~{duration_sec:.2f}s")

            return {
                "message": f"Camera {camera_id} moved {direction} to cover {degrees}° at speed {speed}",
                "duration": round(duration_sec, 2),
                "brand": brand,
            }

        if direction:
            logging.info(f"Camera {camera_id}: moving {direction} at speed {speed}")
            cam.move_camera(direction, speed=speed)
            return {"message": f"Camera {camera_id} moved {direction} at speed {speed}"}

        return {"error": "Either pose_id, degrees+direction, or direction must be specified."}

    except Exception as e:
        logging.error(f"Camera {camera_id}: movement error - {e}")
        return {"error": str(e)}


@app.post("/stop/{camera_id}")
async def stop_camera(camera_id: str):
    """Stops the camera movement."""
    global last_command_time
    last_command_time = time.time()

    if camera_id not in CAMERAS or camera_id not in CAMERA_OBJECTS:
        return {"error": "Invalid camera ID."}

    cam = CAMERA_OBJECTS[camera_id]

    try:
        cam.move_camera("Stop")
        logging.info(f"Camera {camera_id}: movement stopped")
        return {"message": f"Camera {camera_id} stopped moving"}
    except Exception as e:
        logging.error(f"Camera {camera_id}: failed to stop - {e}")
        return {"error": str(e)}


@app.post("/zoom/{camera_id}/{level}")
async def zoom_camera(camera_id: str, level: int):
    """Adjusts the camera zoom level (0 to 64)."""
    global last_command_time
    last_command_time = time.time()

    if camera_id not in CAMERAS or camera_id not in CAMERA_OBJECTS:
        return {"error": "Invalid camera ID."}

    if not (0 <= level <= 64):
        return {"error": "Zoom level must be between 0 and 64."}

    cam = CAMERA_OBJECTS[camera_id]

    try:
        cam.start_zoom_focus(level)
        logging.info(f"Camera {camera_id}: zoom set to {level}")
        return {"message": f"Camera {camera_id} zoom set to {level}"}
    except Exception as e:
        logging.error(f"Camera {camera_id}: failed to set zoom - {e}")
        return {"error": str(e)}


@app.get("/capture/{camera_id}")
async def capture_image(camera_id: str, pose_id: Optional[int] = None):
    """Captures an image from the camera (optionally after moving to a preset pose)."""
    global last_command_time
    last_command_time = time.time()

    if camera_id not in CAMERAS or camera_id not in CAMERA_OBJECTS:
        return {"error": "Invalid camera ID."}

    cam = CAMERA_OBJECTS[camera_id]

    try:
        image = cam.capture(pos_id=pose_id)
        if image is None:
            logging.error(f"Camera {camera_id}: failed to capture image")
            return {"error": "Failed to capture image from camera."}

        img_io = BytesIO()
        image.save(img_io, format="JPEG")
        img_io.seek(0)

        logging.info(f"Camera {camera_id}: image captured successfully (pose_id={pose_id})")
        return StreamingResponse(img_io, media_type="image/jpeg")

    except Exception as e:
        logging.error(f"Camera {camera_id}: capture failed - {e}")
        return {"error": str(e)}


@app.get("/is_stream_running/{camera_id}")
async def is_stream_running(camera_id: str):
    """Check if a specific camera is currently streaming."""
    proc = processes.get(camera_id)
    if proc and is_process_running(proc):
        return {"running": True}
    return {"running": False}


@app.get("/camera_infos")
async def get_camera_infos():
    """Returns list of cameras with their IP addresses, azimuths, and other metadata."""
    camera_infos = []

    for ip, cam_info in credentials.items():
        camera_infos.append({
            "ip": ip,
            "azimuths": cam_info.get("azimuths", []),
            "poses": cam_info.get("poses", []),
            "name": cam_info.get("name", "Unknown"),
            "id": cam_info.get("id"),
            "type": cam_info.get("type", "Unknown"),
            "brand": cam_info.get("brand", "unknown"),
        })

    return {"cameras": camera_infos}
