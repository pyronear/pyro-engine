# Copyright (C) 2022-2026, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.


from __future__ import annotations

import logging
import pathlib
import time
from io import BytesIO
from typing import Any, Callable, List, Optional
from urllib.parse import parse_qsl, urlencode, urljoin, urlparse, urlunparse

import cv2
import numpy as np
import requests
from PIL import Image
from requests.auth import HTTPBasicAuth, HTTPDigestAuth

from pyro_camera_api.camera.base import PAN_OPERATIONS, BaseCamera, FocusAbortedError, FocusMixin, PTZMixin

logger = logging.getLogger(__name__)


class CTronicsCamera(BaseCamera, PTZMixin, FocusMixin):
    """CTronics camera using HTTP snapshots and ONVIF PTZ/Imaging services."""

    def __init__(
        self,
        camera_id: str,
        ip_address: str,
        username: str,
        password: str,
        port: int = 80,
        protocol: str = "http",
        snapshot_path: str = "/tmpfs/snap.jpg",
        snapshot_command: Optional[str] = None,
        timeout: float = 5.0,
        model: Optional[str] = None,
        cam_type: str = "static",
        cam_poses: Optional[List[int]] = None,
        cam_azimuths: Optional[List[float]] = None,
        onvif_port: int = 8080,
        onvif_protocol: str = "http",
        onvif_wsdl_dir: Optional[str] = None,
        onvif_profile_token: Optional[str] = None,
        focus_path: str = "/web/cgi-bin/hi3510/ptzctrl.cgi",
        focus_step: int = 0,
        focus_speed: int = 45,
        focus_auth: str = "digest",
        focus_min: int = 0,
        focus_max: int = 1000,
    ) -> None:
        super().__init__(camera_id=camera_id, cam_type=cam_type)
        self.ip_address = ip_address
        self.username = username
        self.password = password
        self.port = port
        self.protocol = protocol
        self.snapshot_path = snapshot_path
        self.snapshot_command = snapshot_command
        self.timeout = timeout
        self.model = model
        self.cam_poses = cam_poses if cam_poses is not None else []
        self.cam_azimuths = cam_azimuths if cam_azimuths is not None else []
        self.onvif_port = onvif_port
        self.onvif_protocol = onvif_protocol
        self.onvif_wsdl_dir = onvif_wsdl_dir
        self.onvif_profile_token = onvif_profile_token
        self.focus_path = focus_path
        self.focus_step = focus_step
        self.focus_speed = focus_speed
        self.focus_min = focus_min
        self.focus_max = focus_max
        self.focus_auth = focus_auth.lower()
        self.focus_position: Optional[int] = None
        self.current_azimuth: Optional[float] = None
        self._onvif_camera: Any = None
        self._media_service: Any = None
        self._ptz_service: Any = None
        self._imaging_service: Any = None
        self._profile: Any = None

    @property
    def snapshot_url(self) -> str:
        """Build the authenticated snapshot URL without putting credentials in its authority."""
        base = f"{self.protocol}://{self.ip_address}:{self.port}/"
        path = self.snapshot_path.lstrip("/")
        query_params = {"usr": self.username, "pwd": self.password}
        if self.snapshot_command:
            query_params = {"cmd": self.snapshot_command, **query_params}
        query = urlencode(query_params)
        return urljoin(base, f"{path}?{query}")

    @staticmethod
    def _redact_url(url: str) -> str:
        parsed = urlparse(url)
        query = urlencode(
            [
                (key, "***" if key.lower() in {"usr", "user", "pwd", "password"} else value)
                for key, value in parse_qsl(parsed.query)
            ]
        )
        return urlunparse(parsed._replace(query=query))

    def capture(self, patrol_id: Optional[int] = None) -> Optional[Image.Image]:
        """Fetch one JPEG frame, returning ``None`` when capture fails."""
        _ = patrol_id
        url = self.snapshot_url
        redacted_url = self._redact_url(url)
        try:
            response = requests.get(url, timeout=self.timeout)
            response.raise_for_status()
            if not response.content:
                raise ValueError("empty response")
            image = Image.open(BytesIO(response.content)).convert("RGB")
        except Exception as exc:
            logger.error("CTronics capture failed for %s: %s", redacted_url, exc)
            return None

        logger.info("CTronics capture OK for %s, size=%s", redacted_url, image.size)
        return image

    def _ensure_onvif(self) -> None:
        if self._ptz_service is not None:
            return
        try:
            from onvif import ONVIFCamera
        except ImportError as exc:
            raise RuntimeError("Install onvif-zeep to use CTronics PTZ and focus controls") from exc

        args = [self.ip_address, self.onvif_port, self.username, self.password]
        if self.onvif_wsdl_dir:
            args.append(self.onvif_wsdl_dir)
        self._onvif_camera = ONVIFCamera(*args)
        self._media_service = self._onvif_camera.create_media_service()
        profiles = self._media_service.GetProfiles()
        if not profiles:
            raise RuntimeError(f"No ONVIF media profile found for {self.ip_address}:{self.onvif_port}")
        self._profile = next(
            (profile for profile in profiles if profile.token == self.onvif_profile_token), profiles[0]
        )
        self.onvif_profile_token = self._profile.token
        self._ptz_service = self._onvif_camera.create_ptz_service()
        try:
            self._imaging_service = self._onvif_camera.create_imaging_service()
        except Exception as exc:
            logger.warning("ONVIF Imaging unavailable for %s: %s", self.ip_address, exc)

    @staticmethod
    def _clamp(value: float, low: float, high: float) -> float:
        return max(low, min(high, value))

    def _continuous_move(self, pan: float, tilt: float, zoom: float, speed: int) -> None:
        self._ensure_onvif()

        move_speed = self._clamp(speed / 64.0, 0.1, 1.0)

        request = self._ptz_service.create_type("ContinuousMove")
        request.ProfileToken = self.onvif_profile_token
        request.Velocity = {
            "PanTilt": {"x": pan * move_speed, "y": tilt * move_speed},
            "Zoom": {"x": zoom * move_speed},
        }

        self._ptz_service.ContinuousMove(request)

    def _preset_token(self, preset_id: int) -> str:
        self._ensure_onvif()
        request = self._ptz_service.create_type("GetPresets")
        request.ProfileToken = self.onvif_profile_token
        presets = self._ptz_service.GetPresets(request) or []
        # logger.info(
        #     "CTronics ONVIF presets for %s, profile=%s, requested_id=%s: %s",
        #     self.ip_address,
        #     self.onvif_profile_token,
        #     preset_id,
        #     [
        #         {
        #             "token": getattr(preset, "token", None),
        #             "name": getattr(preset, "Name", getattr(preset, "name", None)),
        #         }
        #         for preset in presets
        #     ],
        # )
        for preset in presets:
            if str(getattr(preset, "token", "")) == str(preset_id):
                logger.info(
                    "CTronics preset id=%s matched ONVIF token=%s directly",
                    preset_id,
                    preset.token,
                )
                return str(preset.token)
        if 0 <= preset_id < len(presets):
            logger.info(
                "CTronics preset id=%s treated as list index, resolved ONVIF token=%s",
                preset_id,
                presets[preset_id].token,
            )
            return str(presets[preset_id].token)
        raise ValueError(f"ONVIF preset {preset_id} was not found on {self.ip_address}")

    def move_camera(self, operation: str, speed: int = 20, idx: int = 0) -> None:
        if self.cam_type == "static":
            return
        operation = operation.strip()
        if operation == "ToPos":
            self._ensure_onvif()
            logger.info(
                "CTronics GotoPreset requested: camera=%s profile=%s idx=%s",
                self.ip_address,
                self.onvif_profile_token,
                idx,
            )
            request = self._ptz_service.create_type("GotoPreset")
            request.ProfileToken = self.onvif_profile_token
            request.PresetToken = self._preset_token(int(idx))
            logger.info(
                "CTronics GotoPreset sending: camera=%s profile=%s token=%s",
                self.ip_address,
                request.ProfileToken,
                request.PresetToken,
            )
            response = self._ptz_service.GotoPreset(request)
            logger.info("CTronics GotoPreset response: camera=%s response=%r", self.ip_address, response)
            self._sync_azimuth_from_pose(int(idx))
            return
        if operation == "Stop":
            self._ensure_onvif()
            request = self._ptz_service.create_type("Stop")
            request.ProfileToken = self.onvif_profile_token
            request.PanTilt = True
            request.Zoom = True
            self._ptz_service.Stop(request)
            return

        vectors = {
            "Left": (-1, 0, 0),
            "Right": (1, 0, 0),
            "Up": (0, 1, 0),
            "Down": (0, -1, 0),
            "UpLeft": (-1, 1, 0),
            "UpRight": (1, 1, 0),
            "DownLeft": (-1, -1, 0),
            "DownRight": (1, -1, 0),
            "ZoomIn": (0, 0, 1),
            "ZoomOut": (0, 0, -1),
        }
        if operation not in vectors:
            raise ValueError(f"Unsupported PTZ operation: {operation}")
        if operation in PAN_OPERATIONS:
            self.current_azimuth = None
        self._continuous_move(*vectors[operation], speed=speed)

    def _sync_azimuth_from_pose(self, pose_id: int) -> None:
        if pose_id in self.cam_poses and len(self.cam_poses) == len(self.cam_azimuths):
            self.current_azimuth = float(self.cam_azimuths[self.cam_poses.index(pose_id)]) % 360.0
        else:
            self.current_azimuth = None

    def get_azimuth(self) -> Optional[float]:
        return self.current_azimuth

    def get_ptz_preset(self) -> Optional[list]:
        self._ensure_onvif()
        request = self._ptz_service.create_type("GetPresets")
        request.ProfileToken = self.onvif_profile_token
        return self._ptz_service.GetPresets(request)

    def set_ptz_preset(self, idx: Optional[int] = None, name: Optional[str] = None) -> Any:
        self._ensure_onvif()
        request = self._ptz_service.create_type("SetPreset")
        request.ProfileToken = self.onvif_profile_token
        request.PresetName = name or f"pos{idx if idx is not None else ''}"
        if idx is not None:
            request.PresetToken = str(idx)
        return self._ptz_service.SetPreset(request)

    def save_preset(self, idx: int, name: Optional[str] = None) -> Any:
        return self.set_ptz_preset(idx=idx, name=name)

    def reboot_camera(self) -> bool:
        self._ensure_onvif()
        self._onvif_camera.devicemgmt.SystemReboot()
        return True

    def _focus_request(self, position: int) -> Any:
        if self._imaging_service is None:
            raise RuntimeError("ONVIF Imaging service is unavailable")
        request = self._imaging_service.create_type("Move")
        request.VideoSourceToken = self._profile.VideoSourceConfiguration.SourceToken
        # FocusMove and AbsoluteFocus are nested ONVIF common-schema types,
        # not global elements in the Imaging WSDL. Zeep accepts nested dicts
        # here and creates the correct tt:FocusMove structure from Move's
        # schema definition.
        request.Focus = {
            "Absolute": {
                "Position": self._clamp(
                    (position - self.focus_min) / max(1, self.focus_max - self.focus_min), 0.0, 1.0
                )
            }
        }
        return request

    def set_manual_focus(self, position: int) -> None:
        self._ensure_onvif()
        self._imaging_service.Move(self._focus_request(position))
        self.focus_position = int(position)

    def move_focus(self, action: str, speed: Optional[int] = None) -> bool:
        """Move the focus one relative step using the camera's Hi3510 CGI endpoint."""
        actions = {
            "plus": "focusin",
            "in": "focusin",
            "focusin": "focusin",
            "minus": "focusout",
            "out": "focusout",
            "focusout": "focusout",
            "stop": "stop",
        }
        normalized_action = actions.get(action.strip().lower())
        if normalized_action is None:
            raise ValueError(f"Unsupported CTronics focus action: {action}")
        query = urlencode(
            {
                "-step": self.focus_step,
                "-act": normalized_action,
                "-speed": self.focus_speed if speed is None else speed,
            }
        )
        base = f"{self.protocol}://{self.ip_address}:{self.port}/"
        url = urljoin(base, f"{self.focus_path.lstrip('/')}?{query}")
        logger.info("CTronics focus %s request: %s", normalized_action, self._redact_url(url))
        auth = None
        if self.focus_auth == "digest":
            auth = HTTPDigestAuth(self.username, self.password)
        elif self.focus_auth == "basic":
            auth = HTTPBasicAuth(self.username, self.password)
        elif self.focus_auth != "none":
            raise ValueError(f"Unsupported CTronics focus authentication: {self.focus_auth}")
        try:
            response = requests.get(url, auth=auth, timeout=self.timeout)
            if response.status_code == 401 and self.focus_auth == "digest":
                challenge = response.headers.get("WWW-Authenticate", "")
                logger.warning(
                    "CTronics focus Digest authentication rejected (WWW-Authenticate=%r); "
                    "retrying with Basic authentication",
                    challenge,
                )
                response = requests.get(
                    url,
                    auth=HTTPBasicAuth(self.username, self.password),
                    timeout=self.timeout,
                )
            response.raise_for_status()
        except requests.RequestException as exc:
            logger.error("CTronics focus %s failed: %s", normalized_action, exc)
            raise RuntimeError(f"CTronics focus command {normalized_action!r} failed") from exc
        logger.info("CTronics focus %s response: status=%s body=%s", normalized_action, response.status_code, response.text[:200])
        return True

    def focus_plus(self, speed: Optional[int] = None) -> bool:
        """Move focus inward by one camera-defined step."""
        return self.move_focus("focusin", speed=speed)

    def focus_minus(self, speed: Optional[int] = None) -> bool:
        """Move focus outward by one camera-defined step."""
        return self.move_focus("focusout", speed=speed)

    def stop_focus(self, speed: Optional[int] = None) -> bool:
        """Stop the current relative focus movement."""
        return self.move_focus("stop", speed=speed)

    def get_focus_level(self) -> Optional[dict]:
        self._ensure_onvif()
        if self._imaging_service is None:
            return None
        request = self._imaging_service.create_type("GetStatus")
        request.VideoSourceToken = self._profile.VideoSourceConfiguration.SourceToken
        status = self._imaging_service.GetStatus(request)
        raw_focus = getattr(getattr(status, "FocusStatus20", None), "Position", None)
        focus = None
        if raw_focus is not None:
            focus = round(self.focus_min + float(raw_focus) * (self.focus_max - self.focus_min))
            self.focus_position = focus
        return {"focus": focus, "focus_raw": raw_focus, "zoom": None}

    def get_focus_options(self) -> Optional[dict]:
        """Return the focus range advertised by the camera's ONVIF Imaging service."""
        self._ensure_onvif()
        if self._imaging_service is None:
            return None
        request = self._imaging_service.create_type("GetOptions")
        request.VideoSourceToken = self._profile.VideoSourceConfiguration.SourceToken
        options = self._imaging_service.GetOptions(request)
        focus_options = getattr(options, "Focus", None)
        absolute = getattr(focus_options, "Absolute", None)
        focus_range = getattr(absolute, "Range", None)
        return {
            "raw": options,
            "absolute_min": getattr(focus_range, "Min", None),
            "absolute_max": getattr(focus_range, "Max", None),
            "focus_options": focus_options,
        }

    def get_auto_focus(self) -> Optional[dict]:
        self._ensure_onvif()
        if self._imaging_service is None:
            return None
        request = self._imaging_service.create_type("GetImagingSettings")
        request.VideoSourceToken = self._profile.VideoSourceConfiguration.SourceToken
        settings = self._imaging_service.GetImagingSettings(request)
        return {"mode": getattr(getattr(settings, "Focus", None), "AutoFocusMode", None)}

    def set_auto_focus(self, disable: bool) -> None:
        self._ensure_onvif()
        if self._imaging_service is None:
            raise RuntimeError("ONVIF Imaging service is unavailable")
        request = self._imaging_service.create_type("SetImagingSettings")
        request.VideoSourceToken = self._profile.VideoSourceConfiguration.SourceToken
        request.ImagingSettings = {"Focus": {"AutoFocusMode": "MANUAL" if disable else "AUTO"}}
        self._imaging_service.SetImagingSettings(request)

    def start_zoom_focus(self, position: int) -> None:
        self.set_manual_focus(position)

    @staticmethod
    def _measure_sharpness(image: Image.Image) -> float:
        return float(cv2.Laplacian(np.array(image.convert("L")), cv2.CV_64F).var())

    def focus_finder(
        self,
        save_images: bool = False,
        retry_depth: int = 0,
        should_abort: Optional[Callable[[], bool]] = None,
    ) -> int:
        _ = retry_depth
        if self.cam_type == "static":
            return self.focus_position or 0
        initial = self.focus_position if self.focus_position is not None else (self.focus_min + self.focus_max) // 2
        candidates = range(max(self.focus_min, initial - 50), min(self.focus_max, initial + 50) + 1, 10)
        scores = []
        for position in candidates:
            if should_abort is not None and should_abort():
                self.set_manual_focus(initial)
                raise FocusAbortedError
            self.set_manual_focus(position)
            time.sleep(1)
            image = self.capture()
            score = self._measure_sharpness(image) if image is not None else 0.0
            scores.append((position, score))
            if save_images and image is not None:
                folder = pathlib.Path("focus_debug") / self.ip_address.replace(".", "_")
                folder.mkdir(exist_ok=True, parents=True)
                image.save(folder / f"focus_{position}.jpg")
        best_focus = max(scores, key=lambda item: item[1])[0]
        self.set_manual_focus(best_focus)
        return best_focus