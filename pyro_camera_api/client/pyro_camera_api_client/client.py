# Copyright (C) 2022-2025, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.


from __future__ import annotations

import io
from typing import Any, Dict, List, Optional

import requests
from PIL import Image


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
        data = resp.json()
        return data.get("camera_ids", [])

    def get_camera_infos(self) -> List[Dict[str, Any]]:
        resp = self._request("GET", "/cameras/camera_infos")
        data = resp.json()
        return data.get("cameras", [])

    # ------------------------------------------------------------------
    # Capture
    # ------------------------------------------------------------------

    def capture_jpeg(
        self,
        camera_ip: str,
        pos_id: Optional[int] = None,
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
        pos_id:
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
        if pos_id is not None:
            params["pos_id"] = pos_id
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
        pos_id: Optional[int] = None,
        anonymize: bool = True,
        max_age_ms: Optional[int] = None,
        strict: bool = False,
        width: Optional[int] = None,
        quality: Optional[int] = None,
    ) -> Image.Image:
        data = self.capture_jpeg(
            camera_ip=camera_ip,
            pos_id=pos_id,
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
    ) -> Optional[Image.Image]:
        params = {"camera_ip": camera_ip, "pose": pose}
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
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {
            "camera_ip": camera_ip,
            "speed": speed,
        }
        if direction is not None:
            params["direction"] = direction
        if pose_id is not None:
            params["pose_id"] = pose_id
        if degrees is not None:
            params["degrees"] = degrees

        resp = self._request("POST", "/control/move", params=params)
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
