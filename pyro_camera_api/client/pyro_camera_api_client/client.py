# Copyright (C) 2022-2026, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.


from __future__ import annotations

import io
import logging
from typing import Any, Dict, List, Optional

import requests
from PIL import Image

logger = logging.getLogger(__name__)


class PyroCameraAPIClient:
    def __init__(
        self,
        base_url: str,
        timeout: float = 5.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        json: Optional[Dict[str, Any]] = None,
        stream: bool = False,
        timeout: Optional[float] = None,
    ) -> requests.Response:
        """
        Internal helper for HTTP calls.

        If timeout is None, use the default timeout stored on the client.
        You can pass timeout=None to disable the requests timeout entirely.
        """
        url = f"{self.base_url}{path}"
        effective_timeout = self.timeout if timeout is None else timeout

        resp = requests.request(
            method=method,
            url=url,
            params=params,
            json=json,
            timeout=effective_timeout,
            stream=stream,
        )
        resp.raise_for_status()
        return resp

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    def health(self) -> Dict[str, Any]:
        resp = self._request("GET", "/health")
        return resp.json()

    # ------------------------------------------------------------------
    # Camera info
    # ------------------------------------------------------------------

    def list_cameras(self) -> List[str]:
        resp = self._request("GET", "/cameras/cameras_list")
        resp.raise_for_status()
        return resp.json()

    def get_camera_infos(self) -> List[Dict[str, Any]]:
        resp = self._request("GET", "/cameras/camera_infos")
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------
    # Capture
    # ------------------------------------------------------------------

    def capture_jpeg(
        self,
        camera_ip: str,
        patrol_id: Optional[int] = None,
        anonymize: bool = True,
        max_age_ms: Optional[int] = None,
        strict: bool = False,
        width: Optional[int] = None,
        quality: Optional[int] = None,
    ) -> bytes:
        """
        Capture a JPEG image from the camera.

        Parameters
        ----------
        camera_ip:
            Camera identifier
        patrol_id:
            Optional preset
        anonymize:
            Apply anonymization
        max_age_ms:
            Max age for boxes
        strict:
            Fail if no recent boxes when anonymize is true
        width:
            Target width in pixels
        quality:
            JPEG quality between 1 and 100. If None the server default is used.
        """
        params: Dict[str, Any] = {
            "camera_ip": camera_ip,
            "anonymize": anonymize,
            "strict": strict,
        }
        if patrol_id is not None:
            params["patrol_id"] = patrol_id
        if max_age_ms is not None:
            params["max_age_ms"] = max_age_ms
        if width is not None:
            params["width"] = width
        if quality is not None:
            params["quality"] = quality

        resp = self._request("GET", "/cameras/capture", params=params, stream=True)
        return resp.content

    def capture_image(
        self,
        camera_ip: str,
        patrol_id: Optional[int] = None,
        anonymize: bool = True,
        max_age_ms: Optional[int] = None,
        strict: bool = False,
        width: Optional[int] = None,
        quality: Optional[int] = None,
    ) -> Image.Image:
        data = self.capture_jpeg(
            camera_ip=camera_ip,
            patrol_id=patrol_id,
            anonymize=anonymize,
            max_age_ms=max_age_ms,
            strict=strict,
            width=width,
            quality=quality,
        )
        return Image.open(io.BytesIO(data)).convert("RGB")

    def get_latest_image(
        self,
        camera_ip: str,
        pose: int,
        quality: Optional[int] = None,
    ) -> Optional[Image.Image]:
        params = {"camera_ip": camera_ip, "pose": pose}
        if quality is not None:
            params["quality"] = quality

        resp = self._request("GET", "/cameras/latest_image", params=params, stream=True)

        if resp.status_code == 204 or not resp.content:
            return None

        return Image.open(io.BytesIO(resp.content)).convert("RGB")

    # ------------------------------------------------------------------
    # PTZ control
    # ------------------------------------------------------------------

    def move_camera(
        self,
        camera_ip: str,
        direction: Optional[str] = None,
        speed: int = 10,
        pose_id: Optional[int] = None,
        degrees: Optional[float] = None,
        duration: Optional[float] = None,
        zoom: int = 0,
    ) -> Dict[str, Any]:
        """Legacy overloaded move endpoint. Prefer goto_preset / start_move /
        stop_move / move_for_duration / move_by_degrees, which are the
        focused replacements."""
        if zoom > 0 and speed != 1:
            logger.warning("zoom=%s > 0: speed will be forced to 1 server-side (requested %s)", zoom, speed)
        params: Dict[str, Any] = {
            "camera_ip": camera_ip,
            "speed": speed,
            "zoom": zoom,
        }
        if direction is not None:
            params["direction"] = direction
        if pose_id is not None:
            params["pose_id"] = pose_id
        if degrees is not None:
            params["degrees"] = degrees
        if duration is not None:
            params["duration"] = duration

        resp = self._request("POST", "/control/move", params=params)
        return resp.json()

    # ------------------------------------------------------------------
    # Focused PTZ actions (preferred over move_camera)
    # ------------------------------------------------------------------

    def goto_preset(self, camera_ip: str, pose_id: int, speed: int = 50) -> Dict[str, Any]:
        """Move to a configured preset pose. Returns immediately."""
        params = {"camera_ip": camera_ip, "pose_id": pose_id, "speed": speed}
        resp = self._request("POST", "/control/goto_preset", params=params)
        return resp.json()

    def start_move(self, camera_ip: str, direction: str, speed: int = 10) -> Dict[str, Any]:
        """Start a continuous move; caller must call stop_move to halt."""
        params = {"camera_ip": camera_ip, "direction": direction, "speed": speed}
        resp = self._request("POST", "/control/start_move", params=params)
        return resp.json()

    def stop_move(self, camera_ip: str) -> Dict[str, Any]:
        """Halt any current movement."""
        resp = self._request("POST", f"/control/stop_move/{camera_ip}")
        return resp.json()

    def move_for_duration(
        self,
        camera_ip: str,
        direction: str,
        duration: float,
        speed: int = 10,
    ) -> Dict[str, Any]:
        """Move for a fixed wall-clock duration (seconds), then stop.
        Server holds a per-camera lock; raises on 409 if busy."""
        params = {
            "camera_ip": camera_ip,
            "direction": direction,
            "duration": duration,
            "speed": speed,
        }
        # Allow the server-side sleep + a small margin.
        req_timeout = max(self.timeout, duration + 5.0) if duration else self.timeout
        resp = self._request("POST", "/control/move_for_duration", params=params, timeout=req_timeout)
        return resp.json()

    def move_by_degrees(
        self,
        camera_ip: str,
        direction: str,
        degrees: float,
        speed: int = 10,
    ) -> Dict[str, Any]:
        """Move by an approximate angle using the server's calibrated speed
        table. Server reads current zoom and force-limits speed to 1 at
        zoom > 0. Raises on 409 if the camera is busy."""
        params = {
            "camera_ip": camera_ip,
            "direction": direction,
            "degrees": degrees,
            "speed": speed,
        }
        resp = self._request("POST", "/control/move_by_degrees", params=params, timeout=30.0)
        return resp.json()

    def click_to_move(
        self,
        camera_ip: str,
        click_x: float,
        click_y: float,
    ) -> Dict[str, Any]:
        """click_x and click_y are normalized coordinates in [0, 1]."""
        params: Dict[str, Any] = {
            "camera_ip": camera_ip,
            "click_x": click_x,
            "click_y": click_y,
        }
        resp = self._request("POST", "/control/click_to_move", params=params, timeout=30.0)
        return resp.json()

    def get_speed_tables(self, camera_ip: str) -> Dict[str, Any]:
        params = {"camera_ip": camera_ip}
        resp = self._request("GET", "/control/speed_tables", params=params)
        return resp.json()

    def stop_camera(self, camera_ip: str) -> Dict[str, Any]:
        resp = self._request("POST", f"/control/stop/{camera_ip}")
        return resp.json()

    def list_presets(self, camera_ip: str) -> Dict[str, Any]:
        params = {"camera_ip": camera_ip}
        resp = self._request("GET", "/control/preset/list", params=params)
        return resp.json()

    def set_preset(self, camera_ip: str, idx: Optional[int] = None) -> Dict[str, Any]:
        params: Dict[str, Any] = {"camera_ip": camera_ip}
        if idx is not None:
            params["idx"] = idx
        resp = self._request("POST", "/control/preset/set", params=params)
        return resp.json()

    def zoom(self, camera_ip: str, level: int) -> Dict[str, Any]:
        resp = self._request("POST", f"/control/zoom/{camera_ip}/{level}")
        return resp.json()

    # ------------------------------------------------------------------
    # Focus
    # ------------------------------------------------------------------

    def set_manual_focus(self, camera_ip: str, position: int) -> Dict[str, Any]:
        params = {"camera_ip": camera_ip, "position": position}
        resp = self._request("POST", "/focus/manual", params=params)
        return resp.json()

    def set_autofocus(self, camera_ip: str, disable: bool = True) -> Dict[str, Any]:
        params = {"camera_ip": camera_ip, "disable": disable}
        resp = self._request("POST", "/focus/set_autofocus", params=params)
        return resp.json()

    def get_focus_status(self, camera_ip: str) -> Dict[str, Any]:
        params = {"camera_ip": camera_ip}
        resp = self._request("GET", "/focus/status", params=params)
        return resp.json()

    def run_focus_optimization(
        self,
        camera_ip: str,
        save_images: bool = False,
        request_timeout: Optional[float] = 120.0,
    ) -> Dict[str, Any]:
        """
        Run the autofocus search on the camera.

        request_timeout lets you override the default client timeout.
        Use a larger value for long autofocus sequences or pass None to disable timeout.
        """
        params = {"camera_ip": camera_ip, "save_images": save_images}
        resp = self._request(
            "POST",
            "/focus/focus_finder",
            params=params,
            timeout=request_timeout,
        )
        return resp.json()

    # ------------------------------------------------------------------
    # Patrol
    # ------------------------------------------------------------------

    def start_patrol(self, camera_ip: str) -> Dict[str, Any]:
        params = {"camera_ip": camera_ip}
        resp = self._request("POST", "/patrol/start_patrol", params=params)
        return resp.json()

    def stop_patrol(self, camera_ip: str) -> Dict[str, Any]:
        params = {"camera_ip": camera_ip}
        resp = self._request("POST", "/patrol/stop_patrol", params=params)
        return resp.json()

    def get_patrol_status(self, camera_ip: str) -> Dict[str, Any]:
        params = {"camera_ip": camera_ip}
        resp = self._request("GET", "/patrol/patrol_status", params=params)
        return resp.json()

    # ------------------------------------------------------------------
    # Stream
    # ------------------------------------------------------------------

    def start_stream(self, camera_ip: str) -> Dict[str, Any]:
        resp = self._request("POST", f"/stream/start_stream/{camera_ip}")
        return resp.json()

    def stop_stream(self) -> Dict[str, Any]:
        resp = self._request("POST", "/stream/stop_stream")
        return resp.json()

    def get_stream_status(self) -> Dict[str, Any]:
        resp = self._request("GET", "/stream/status")
        return resp.json()

    def is_stream_running(self, camera_ip: str) -> Dict[str, Any]:
        resp = self._request("GET", f"/stream/is_stream_running/{camera_ip}")
        return resp.json()
