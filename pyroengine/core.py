# Copyright (C) 2022-2026, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.


import hashlib
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import requests
import urllib3
from PIL import Image, UnidentifiedImageError

# Add the parent folder of pyro_camera_api to the import path
sys.path.append(str(Path(__file__).resolve().parent.parent / "pyro_camera_api"))

from pyro_camera_api_client.client import PyroCameraAPIClient

from .engine import Engine
from .zoom_capture import RETURN_MARGIN_S, ZoomCapture

__all__ = ["SystemController", "is_day_time"]

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logger = logging.getLogger(__name__)

# Cameras flicker between colour and infrared at dusk, and one frame must not restart an ongoing
# event or put a site to sleep in daylight: both need this many night frames or passes in a row.
NIGHT_FRAMES_BEFORE_RESET = 3
NIGHT_PASSES_BEFORE_SLEEP = 3


def is_day_time(cache: Optional[Path], frame: Image.Image, strategy: str, delta: int = 0) -> bool:
    """
    Determine whether it is daytime based on the selected strategy.

    Args:
        cache (Path): Cache folder containing the `sunset_sunrise.txt` file (for time based strategy).
        frame (PIL.Image): Image frame to analyze (for IR based strategy).
        strategy (str): Strategy to determine daytime, one of "time", "ir", or "both".
        delta (int, optional): Tolerance in seconds around sunrise or sunset for day and night transition.

    Returns:
        bool: True if it is considered daytime, False otherwise.
    """
    is_day = True
    if cache and strategy in ["both", "time"]:
        with Path(cache.joinpath("sunset_sunrise.txt")).open() as f:
            lines = f.readlines()
        sunrise = datetime.strptime(lines[0].strip(), "%H:%M")
        sunset = datetime.strptime(lines[1].strip(), "%H:%M")
        now = datetime.strptime(datetime.now().isoformat().split("T")[1][:5], "%H:%M")
        if (now - sunrise).total_seconds() < -delta or (sunset - now).total_seconds() < -delta:
            is_day = False

    if strategy in ["both", "ir"]:
        frame_arr = np.array(frame)
        if np.max(frame_arr[:, :, 0] - frame_arr[:, :, 1]) == 0:
            is_day = False

    return is_day


class SystemController:
    """
    Controller to manage multiple cameras, capture images, and perform detection.
    """

    API_INITIAL_WAIT = 30
    API_RETRY_DELAY = 10
    POST_READY_WAIT = 10

    def __init__(
        self,
        engine: Engine,
        camera_data: Dict[str, Dict[str, Any]],
        pyro_camera_api_url: str,
        heartbeat_file: Optional[str] = None,
        zoom_capture: Optional[ZoomCapture] = None,
    ) -> None:
        self.engine = engine
        self.camera_data = camera_data
        # Temporary R&D: closer look at validated detections, uploaded to S3 (see zoom_capture.py)
        self.zoom_capture = zoom_capture
        self.is_day = True
        self.last_autofocus: Optional[datetime] = None
        self._last_frame_digest: Dict[str, bytes] = {}
        self._night_frames: Dict[str, int] = {}  # consecutive new night frames per cam_id
        self._night_passes = 0  # consecutive passes where every fresh frame was a night one
        # Touched on every loop so container liveness does not depend on log content.
        self.heartbeat_file = Path(heartbeat_file) if heartbeat_file else None

        # Wait for the camera API to be available
        logger.info("Waiting for Pyro Camera API at %s", pyro_camera_api_url)
        time.sleep(self.API_INITIAL_WAIT)
        while True:
            try:
                self.camera_api_client = PyroCameraAPIClient(pyro_camera_api_url)
                _ = self.camera_api_client.get_stream_status()
                logger.info("Pyro Camera API client ready")
                break
            except Exception as e:
                # Expected while the camera API container is still booting.
                logger.warning("Camera API not ready yet, retrying in %ds: %s", self.API_RETRY_DELAY, e)
                time.sleep(self.API_RETRY_DELAY)

        # Optional startup actions, do not fail hard
        for ip in self.camera_data:
            try:
                self.camera_api_client.start_patrol(ip)
            except Exception as e:
                logger.warning("[%s] Could not start patrol at startup, continuing: %s", ip, e)

        # Wait and then loop until inference passes once
        logger.info("Waiting for cameras (%d configured)", len(self.camera_data))
        time.sleep(self.POST_READY_WAIT)
        while True:
            try:
                self.inference_loop()
                break
            except Exception as e:
                # Cameras may not have produced a first frame yet, keep waiting.
                logger.warning("First inference pass failed, retrying in %ds: %s", self.API_RETRY_DELAY, e)
                time.sleep(self.API_RETRY_DELAY)

    def focus_finder(self) -> None:
        """
        Run hourly autofocus on non static cameras when it is daytime.

        This stops patrol, runs the server side focus optimization, then restarts patrol.
        """
        now = datetime.now()
        if self.is_day and (self.last_autofocus is None or (now - self.last_autofocus).total_seconds() > 3600):
            logger.info("Hourly autofocus triggered after idle period")

            for ip, cam in self.camera_data.items():
                if cam.get("type") != "static":
                    poses = cam.get("poses", [])
                    if not poses:
                        continue
                    pose = poses[-1]
                    if self._safe_get_latest_image(ip, pose) is not None:
                        # The camera answers, so the 2 min per-camera optimization timeout
                        # must not read as a dead engine.
                        self._write_heartbeat()
                        try:
                            self.camera_api_client.stop_patrol(ip)
                            time.sleep(0.5)
                            self.camera_api_client.run_focus_optimization(ip)
                            logger.info("[%s] Autofocus completed", ip)
                            self.camera_api_client.start_patrol(ip)
                            self.last_autofocus = now
                        except Exception:
                            logger.exception("[%s] Failed to run hourly focus finder", ip)

    def _any_stream_active(self) -> bool:
        """
        Return True if any stream is active.

        Supports both the new keys active_pipelines and active_ffmpeg
        and the legacy key active_streams used in older APIs and tests.
        """
        try:
            status = self.camera_api_client.get_stream_status()

            # New format with explicit lists
            active_pipelines = status.get("active_pipelines")
            active_ffmpeg = status.get("active_ffmpeg")
            if active_pipelines is not None or active_ffmpeg is not None:
                return bool(active_pipelines or active_ffmpeg)

            # Backward compatible support for legacy field
            active_streams = status.get("active_streams")
            if active_streams is not None:
                try:
                    return int(active_streams) > 0
                except (TypeError, ValueError):
                    return bool(active_streams)

            return False
        except Exception as e:
            logger.error("Could not fetch stream status: %s", e)
            return False

    def _safe_get_latest_image(self, ip: str, pose: int) -> Optional[Image.Image]:
        try:
            return self.camera_api_client.get_latest_image(ip, pose)
        except UnidentifiedImageError:
            return None
        except Exception as e:
            logger.error("[%s] Could not get image for pose %s: %s", ip, pose, e)
            return None

    def _maybe_zoom_capture(
        self, ip: str, pose: int, cam: Dict[str, Any], cam_id: str, frame: Image.Image, deadline: float
    ) -> None:
        """Start a zoom capture job when the engine validated detections on a Reolink PTZ pose."""
        if self.zoom_capture is None:
            return
        adapter = str(cam.get("adapter") or cam.get("brand") or "").lower()
        if not adapter.startswith("reolink"):
            return
        state = self.engine._states.get(cam_id)
        if not state or not state["ongoing"] or not state["last_predictions"]:
            return
        # Validated bboxes of the latest frame: normalized [x1, y1, x2, y2, conf]
        bboxes = state["last_predictions"][-1][2]
        if bboxes:
            # A stream starting mid-job aborts the PTZ steps: live viewing keeps priority.
            self.zoom_capture.trigger(
                self.camera_api_client, ip, pose, cam["name"], frame, bboxes, deadline, self._any_stream_active
            )

    def inference_loop(self, deadline: Optional[float] = None) -> None:
        """
        Run one inference pass on all cameras and poses.

        This skips processing entirely if a stream is currently active.

        Args:
            deadline (float, optional): time.monotonic() value before which zoom capture jobs must
                return the camera to its patrol; zoom capture is skipped when None.
        """
        # Liveness, not image flow: the healthcheck kills a stale engine, and restarting it
        # would not bring back an offline camera (the camera API reports those instead).
        self._write_heartbeat()

        if self._any_stream_active():
            logger.info("Stream detected, skipping inference on all cameras")
            return

        start_ts = time.time()
        analyzed = 0
        unchanged = 0
        night = 0
        failed = 0
        positive = 0
        max_conf = 0.0
        day_frames: List[bool] = []
        reset_cams: List[str] = []

        for ip, cam in self.camera_data.items():
            camera_name = cam["name"]
            is_ptz = cam.get("type") == "ptz"
            # Static cameras expose a single frame under the conventional pose -1.
            poses = cam.get("poses", []) if is_ptz else [-1]

            if self.zoom_capture is not None and self.zoom_capture.images_stale(ip):
                logger.info("[%s] Patrol just restarted after a zoom capture, skipping it this loop", ip)
                continue

            for pose in poses:
                if self._any_stream_active():
                    logger.info("Stream turned on during loop, stopping inference immediately")
                    return

                cam_id = f"{ip}_{pose}" if is_ptz else ip
                try:
                    frame = self._safe_get_latest_image(ip, pose)
                    if frame is None:
                        failed += 1
                        continue
                    logger.debug("[%s] Captured image from %s", cam_id, camera_name)
                    frame_is_day = is_day_time(None, frame, "ir")
                    day_frames.append(frame_is_day)
                    night_frames_before = self._night_frames.get(cam_id, 0)
                    conf = self._analyze(frame, cam_id, frame_is_day)
                    if not frame_is_day:
                        night += 1
                        if night_frames_before < NIGHT_FRAMES_BEFORE_RESET == self._night_frames.get(cam_id, 0):
                            reset_cams.append(cam_id)
                        continue
                    if conf is None:
                        unchanged += 1
                        continue
                    analyzed += 1
                    if deadline is not None and is_ptz:
                        self._maybe_zoom_capture(ip, pose, cam, cam_id, frame, deadline)
                    max_conf = max(max_conf, conf)
                    # The predictor applies hysteresis (0.8x threshold while an event is
                    # ongoing), so read its verdict instead of re-deriving the threshold:
                    # the summary must not say "quiet" while alerts are being staged.
                    if self.engine._states[cam_id]["ongoing"]:
                        positive += 1
                except requests.HTTPError as e:
                    failed += 1
                    body = e.response.text if e.response is not None else e
                    logger.error("[%s] HTTP error from %s: %s", cam_id, camera_name, body)
                except Exception:
                    failed += 1
                    logger.exception("[%s] Inference failed on %s", cam_id, camera_name)

        # One day frame keeps the site awake, so one camera in infrared (fog, a dark area) does
        # not blind the others. Only cameras with a fresh frame count: a camera down says nothing
        # about the light, and a frozen frame is already dropped by the API after max_age_s. The
        # site sleeps once every fresh frame has been a night one for several passes in a row.
        if any(day_frames):
            self.is_day = True
            self._night_passes = 0
            # A camera in infrared while the site sees daylight is blind (stuck IR cut, dark
            # area) and its frames are not analyzed. Warned once per episode.
            if reset_cams:
                logger.warning("Cameras in infrared while others see daylight: %s", reset_cams)
        elif day_frames:
            self._night_passes += 1
            if self._night_passes >= NIGHT_PASSES_BEFORE_SLEEP:
                self.is_day = False

        logger.info(
            "Inference round: analyzed=%d unchanged=%d night=%d failed=%d positive=%d max_confidence=%.2f "
            "duration=%.1fs",
            analyzed,
            unchanged,
            night,
            failed,
            positive,
            max_conf,
            time.time() - start_ts,
        )

    def _write_heartbeat(self) -> None:
        """Refresh the heartbeat file used by the container healthcheck.

        A failure here must not stop detection, so it is reported and execution continues.
        """
        if self.heartbeat_file is None:
            return
        try:
            self.heartbeat_file.write_text(datetime.now().isoformat())
        except OSError as e:
            logger.warning("Could not write heartbeat file %s: %s", self.heartbeat_file, e)

    def _sleep_with_heartbeat(self, duration: float, step: float = 60.0) -> None:
        """Sleep in steps, refreshing the heartbeat, so the long night sleep stays healthy."""
        remaining = duration
        while remaining > 0:
            self._write_heartbeat()
            chunk = min(step, remaining)
            time.sleep(chunk)
            remaining -= chunk

    def _analyze(self, frame: Image.Image, cam_id: str, frame_is_day: bool) -> Optional[float]:
        """Run inference on a new day frame and return its confidence, None when the frame is skipped."""
        # The engine reads the API cache, which the patrol refreshes slower than the loop runs:
        # the same frame read twice is normal, not a frozen camera (the API detects those). It
        # is skipped so a detection is not scored and uploaded again, and an isolated infrared
        # frame is not counted once per read.
        digest = hashlib.md5(frame.tobytes(), usedforsecurity=False).digest()
        if self._last_frame_digest.get(cam_id) == digest:
            logger.debug("[%s] Frame unchanged since the last pass, skipping inference", cam_id)
            # The camera still answers: a slow-refreshing URL camera must not look offline.
            self.engine.maybe_heartbeat(cam_id)
            return None
        if not frame_is_day:
            self._last_frame_digest[cam_id] = digest
            # The model is trained on day images, infrared frames only give false positives.
            logger.debug("[%s] Night frame, skipping inference", cam_id)
            self._night_frames[cam_id] = self._night_frames.get(cam_id, 0) + 1
            if self._night_frames[cam_id] == NIGHT_FRAMES_BEFORE_RESET:
                # Drop the camera's detection history so evening frames are not combined with
                # the next morning's ones. Not on the first frame: it would also drop an ongoing
                # event and its crop boxes on one isolated infrared frame.
                self.engine._states.pop(cam_id, None)
            return None
        self._night_frames[cam_id] = 0
        conf = float(self.engine.predict(frame, cam_id))
        # Stored after predict, so a failed prediction is retried on the next pass.
        self._last_frame_digest[cam_id] = digest
        return conf

    def _probe_day(self) -> bool:
        """Capture one frame per camera until one shows daylight.

        Only cameras that answer count: one down says nothing about the light and must not
        keep the site awake all night. When none answers, the site wakes up and lets the
        inference loop decide from the next fresh frames.
        """
        day = False
        night = 0
        for ip in self.camera_data:
            try:
                frame = self.camera_api_client.capture_image(ip)
            except Exception as e:
                logger.error("[%s] Failed to capture after sleep: %s", ip, e)
                continue
            if frame is None:
                continue
            if is_day_time(None, frame, "ir"):
                day = True
                break
            night += 1
        if day or night == 0:
            # Wake up with a clean count: the night streak must not put the site back to sleep
            # on the first pass, and a camera still in infrared at dawn must be warned again.
            self._night_frames.clear()
            self._night_passes = 0
            return True
        return False

    def check_and_restart_patrol(self) -> None:
        """
        Check stream activity and ensure patrol is running when no stream is active.
        """
        try:
            stream_status = self.camera_api_client.get_stream_status()
        except Exception as e:
            logger.error("Could not check if stream is running: %s", e)
            return

        active_pipelines = stream_status.get("active_pipelines") or []
        active_ffmpeg = stream_status.get("active_ffmpeg") or []
        if not active_pipelines and not active_ffmpeg:
            for ip in self.camera_data:
                try:
                    patrol_status = self.camera_api_client.get_patrol_status(ip)
                    if not patrol_status.get("patrol_running", False):
                        self.camera_api_client.start_patrol(ip)
                        logger.info("[%s] Patrol restarted", ip)
                except Exception as e:
                    logger.error("[%s] Could not check or restart patrol: %s", ip, e)

    def main_loop(self, period: int, send_alerts: bool = True) -> None:
        """
        Run the main control loop.

        This loop handles:
        detection alerts,
        autofocus,
        patrol management,
        image inference for all cameras.

        Args:
            period (int): Interval between analysis loops in seconds.
            send_alerts (bool, optional): Whether to process detection alerts.
        """
        while True:
            start_ts = time.time()
            start_mono = time.monotonic()

            if not self.is_day:
                for ip in self.camera_data:
                    try:
                        patrol_status = self.camera_api_client.get_patrol_status(ip)
                        if patrol_status.get("patrol_running", False):
                            self.camera_api_client.stop_patrol(ip)
                            logger.info("[%s] Patrol stopped for the night", ip)
                    except Exception as e:
                        logger.error("[%s] Failed to stop patrol: %s", ip, e)

                logger.info("Nighttime detected by every camera with a fresh image, sleeping for 1 hour")
                self._sleep_with_heartbeat(3600)

                try:
                    self.is_day = self._probe_day()
                    logger.info("Re checked day and night after sleep, is_day=%s", self.is_day)

                    if self.is_day:
                        logger.info("Day detected, restarting patrols")
                        self.check_and_restart_patrol()
                        time.sleep(30)
                except Exception as e:
                    logger.error("Failed to check day and night after sleep: %s", e)
                    self.is_day = False

            else:
                if len(self.engine._alerts) and send_alerts:
                    try:
                        self.engine._process_alerts()
                    except Exception:
                        logger.exception("Error processing alerts")
                else:
                    self.focus_finder()

                self.check_and_restart_patrol()
                if self.zoom_capture is None:
                    self.inference_loop()
                else:
                    # Zoom jobs run in parallel with inference and must hand the camera back to
                    # the patrol before the next loop: they abort at the deadline, then restore.
                    deadline = start_mono + period - RETURN_MARGIN_S
                    self.inference_loop(deadline)
                    self.zoom_capture.wait(max(0.0, deadline + RETURN_MARGIN_S + 5.0 - time.monotonic()))

                loop_time = time.time() - start_ts
                sleep_time = max(period - loop_time, 0)
                logger.debug("Loop ran in %.2fs, sleeping for %.2fs", loop_time, sleep_time)
                # Plain sleep: liveness during the day comes from successful captures, so a
                # blind engine must go stale here rather than keep the heartbeat fresh.
                time.sleep(sleep_time)
