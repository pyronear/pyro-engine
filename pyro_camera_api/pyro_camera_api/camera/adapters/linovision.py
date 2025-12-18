# Copyright (C) 2022-2025, Pyronear.
# This program is licensed under the Apache License 2.0.

from __future__ import annotations

import logging
import time
import xml.etree.ElementTree as ET
from io import BytesIO
from typing import List, Optional

import requests
import urllib3
from PIL import Image
from requests.auth import HTTPDigestAuth

from pyro_camera_api.camera.base import BaseCamera, FocusMixin, PTZMixin

__all__ = ["LinovisionCamera"]

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logger = logging.getLogger(__name__)


class LinovisionCamera(BaseCamera, PTZMixin, FocusMixin):
    """
    A controller class for interacting with Linovision cameras using ISAPI.

    Key point
    cam_azimuths from config are REAL azimuths (kept on cam_azimuths for parity with Reolink).
    The camera expects its own azimuth reference, so on init we also compute camera-frame values
    using azimuth_offset_deg.
      camera_command_az = (real_az + azimuth_offset_deg) % 360
    """

    def __init__(
        self,
        camera_id: str,
        ip_address: str,
        username: str,
        password: str,
        cam_type: str = "ptz",
        cam_poses: Optional[List[int]] = None,
        cam_azimuths: Optional[List[int]] = None,
        protocol: str = "http",
        verify_tls: bool = False,
        snapshot_channel: str = "101",
        ptz_channel: str = "1",
        focus_position: Optional[int] = None,
        timeout: float = 3.0,
        azimuth_offset_deg: float = 0.0,
        default_elevation_deg: Optional[float] = 0,
    ):
        super().__init__(camera_id=camera_id, cam_type=cam_type)
        self.ip_address = ip_address
        self.username = username
        self.password = password
        self.cam_poses = cam_poses if cam_poses is not None else []
        # Real-world azimuths, kept with the same name as Reolink
        self.cam_azimuths = cam_azimuths if cam_azimuths is not None else []
        self.protocol = protocol
        self.verify_tls = verify_tls
        self.snapshot_channel = str(snapshot_channel)
        self.ptz_channel = str(ptz_channel)
        self.focus_position = focus_position
        self.timeout = float(timeout)

        self.azimuth_offset_deg = float(azimuth_offset_deg) % 360.0
        self.default_elevation_deg = default_elevation_deg
        # Convert real-world azimuths to camera-relative azimuths using the offset
        self.cam_azimuths_camera = [self._real_to_camera_azimuth(a) for a in self.cam_azimuths]

        self._auth = HTTPDigestAuth(self.username, self.password)
        self._base = f"{self.protocol}://{self.ip_address}"

    def _build_url(self, path: str) -> str:
        if not path.startswith("/"):
            path = "/" + path
        return self._base + path

    def _request(self, method: str, path: str, **kwargs) -> requests.Response:
        kwargs.setdefault("auth", self._auth)
        kwargs.setdefault("verify", self.verify_tls)
        kwargs.setdefault("timeout", self.timeout)
        return requests.request(method, self._build_url(path), **kwargs)

    def _handle_response(self, resp: requests.Response, success_message: str = "") -> Optional[requests.Response]:
        if resp.status_code in {200, 201, 204}:
            if success_message:
                logger.debug(success_message)
            return resp

        body = ""
        try:
            body = resp.text[:500]
        except Exception:
            body = "<unreadable body>"

        logger.error("ISAPI error, status %s, body %s", resp.status_code, body)
        return None

    @staticmethod
    def _clamp(v: float, vmin: float, vmax: float) -> float:
        return max(vmin, min(vmax, v))

    @staticmethod
    def _azimuth_deg_to_raw(azimuth_deg: float) -> int:
        raw = int(round((float(azimuth_deg) % 360.0) * 10.0))
        if raw == 3600:
            raw = 0
        return raw

    def _real_to_camera_azimuth(self, real_azimuth_deg: float) -> float:
        return (float(real_azimuth_deg) + self.azimuth_offset_deg) % 360.0

    def _camera_to_real_azimuth(self, camera_azimuth_deg: float) -> float:
        return (float(camera_azimuth_deg) - self.azimuth_offset_deg) % 360.0

    def _pose_to_target_camera_azimuth(self, pose_id: int) -> float:
        if not self.cam_poses or not self.cam_azimuths:
            raise RuntimeError("cam_poses and cam_azimuths must be provided to move by pose mapping")

        if len(self.cam_poses) != len(self.cam_azimuths):
            raise RuntimeError("cam_poses and cam_azimuths must have the same length")

        if pose_id not in self.cam_poses:
            raise RuntimeError(f"pose_id {pose_id} not found in cam_poses")

        i = self.cam_poses.index(pose_id)
        return float(self.cam_azimuths_camera[i]) % 360.0

    def capture(self, pos_id: Optional[int] = None, timeout: int = 2) -> Optional[Image.Image]:
        if pos_id is not None:
            self.move_camera("ToPos", idx=int(pos_id), speed=0)
            time.sleep(1)

        old_timeout = self.timeout
        self.timeout = float(timeout)
        try:
            path = f"/ISAPI/Streaming/channels/{self.snapshot_channel}/picture"
            resp = self._request("GET", path, headers={"Accept": "image/jpeg"})
            if resp.status_code == 200 and resp.content:
                return Image.open(BytesIO(resp.content)).convert("RGB")
            self._handle_response(resp)
            return None
        except requests.RequestException as exc:
            logger.error("Capture failed, %s", exc)
            return None
        finally:
            self.timeout = old_timeout

    def get_ptz_status(self) -> dict:
        path = f"/ISAPI/PTZCtrl/channels/{self.ptz_channel}/status"
        resp = self._request("GET", path, headers={"Accept": "application/xml"})
        if resp.status_code != 200:
            raise RuntimeError(f"PTZ status failed, status {resp.status_code}, body {resp.text[:200]}")

        root = ET.fromstring(resp.text)
        ns = {"ns": root.tag.split("}")[0].strip("{")} if "}" in root.tag else {}

        az_el = root.find(".//ns:azimuth", ns) if ns else root.find(".//azimuth")
        el_el = root.find(".//ns:elevation", ns) if ns else root.find(".//elevation")
        z_el = root.find(".//ns:absoluteZoom", ns) if ns else root.find(".//absoluteZoom")

        if az_el is None or el_el is None:
            raise RuntimeError(f"Unexpected PTZ status XML, body {resp.text[:400]}")

        az10 = int(az_el.text)
        el10 = int(el_el.text)
        zoom = int(z_el.text) if z_el is not None and z_el.text is not None else None

        return {
            "azimuth_deg": az10 / 10.0,
            "elevation_deg": el10 / 10.0,
            "azimuth_raw": az10,
            "elevation_raw": el10,
            "zoom_raw": zoom,
            "real_azimuth_deg": self._camera_to_real_azimuth(az10 / 10.0),
        }

    def wait_reached_azimuth_raw(
        self,
        target_azimuth_deg: float,
        timeout_s: float = 15.0,
        poll_s: float = 0.15,
    ) -> dict:
        target_raw = self._azimuth_deg_to_raw(target_azimuth_deg)

        t0 = time.time()
        last = None
        while time.time() - t0 < timeout_s:
            st = self.get_ptz_status()
            last = st
            if int(st["azimuth_raw"]) == target_raw:
                return st
            time.sleep(poll_s)

        raise RuntimeError(f"Timeout waiting for azimuth_raw={target_raw}, last={last}")

    def move_absolute(
        self,
        azimuth_deg: float,
        elevation_deg: Optional[float] = None,
        zoom: Optional[int] = None,
        horizontal_speed: float = 80.0,
        vertical_speed: float = 80.0,
        prefer_current_elevation: bool = False,
    ) -> None:
        if self.cam_type == "static":
            return

        az = float(azimuth_deg) % 360.0

        if elevation_deg is None:
            if prefer_current_elevation:
                st = self.get_ptz_status()
                el = float(st["elevation_deg"])
            elif self.default_elevation_deg is not None:
                el = self._clamp(float(self.default_elevation_deg), -10.0, 90.0)
            else:
                st = self.get_ptz_status()
                el = float(st["elevation_deg"])
        else:
            el = self._clamp(float(elevation_deg), -10.0, 90.0)

        hs = self._clamp(float(horizontal_speed), 0.1, 80.0)
        vs = self._clamp(float(vertical_speed), 0.1, 80.0)

        z = None
        if zoom is not None:
            z = int(self._clamp(float(zoom), 1.0, 25.0))

        path = f"/ISAPI/PTZCtrl/channels/{self.ptz_channel}/absoluteEx"

        if z is None:
            xml = (
                "<?xml version='1.0' encoding='UTF-8'?>"
                "<PTZAbsoluteEx version='2.0' xmlns='http://www.std-cgi.com/ver20/XMLSchema'>"
                f"<elevation>{el}</elevation>"
                f"<azimuth>{az}</azimuth>"
                f"<horizontalSpeed>{hs}</horizontalSpeed>"
                f"<verticalSpeed>{vs}</verticalSpeed>"
                "</PTZAbsoluteEx>"
            )
        else:
            xml = (
                "<?xml version='1.0' encoding='UTF-8'?>"
                "<PTZAbsoluteEx version='2.0' xmlns='http://www.std-cgi.com/ver20/XMLSchema'>"
                f"<elevation>{el}</elevation>"
                f"<azimuth>{az}</azimuth>"
                f"<absoluteZoom>{z}</absoluteZoom>"
                f"<horizontalSpeed>{hs}</horizontalSpeed>"
                f"<verticalSpeed>{vs}</verticalSpeed>"
                "</PTZAbsoluteEx>"
            )

        resp = self._request("PUT", path, data=xml, headers={"Content-Type": "application/xml"})
        if self._handle_response(resp, "Absolute move success") is None:
            raise RuntimeError(f"Absolute move failed, status {resp.status_code}, body {resp.text[:300]}")

    def move_absolute_perfect(
        self,
        azimuth_deg: float,
        elevation_deg: Optional[float] = None,
        zoom: Optional[int] = None,
        horizontal_speed: float = 80.0,
        vertical_speed: float = 80.0,
        timeout_s: float = 15.0,
        poll_s: float = 0.15,
        prefer_current_elevation: bool = False,
    ) -> dict:
        self.move_absolute(
            azimuth_deg=azimuth_deg,
            elevation_deg=elevation_deg,
            zoom=zoom,
            horizontal_speed=horizontal_speed,
            vertical_speed=vertical_speed,
            prefer_current_elevation=prefer_current_elevation,
        )
        return self.wait_reached_azimuth_raw(
            target_azimuth_deg=azimuth_deg,
            timeout_s=timeout_s,
            poll_s=poll_s,
        )

    def move_relative_deg(self, delta_azimuth_deg: float, delta_elevation_deg: float = 0.0) -> dict:
        st = self.get_ptz_status()
        new_az = (st["azimuth_deg"] + float(delta_azimuth_deg)) % 360.0
        new_el = st["elevation_deg"] + float(delta_elevation_deg)
        new_el = self._clamp(new_el, -10.0, 90.0)
        self.move_absolute(new_az, elevation_deg=new_el)
        return {"azimuth_deg": new_az, "elevation_deg": new_el}

    def move_to_pose(
        self,
        pose_id: int,
        elevation_deg: Optional[float] = None,
        zoom: Optional[int] = None,
        horizontal_speed: float = 80.0,
        vertical_speed: float = 80.0,
        timeout_s: float = 15.0,
        poll_s: float = 0.15,
        prefer_current_elevation: bool = False,
    ) -> dict:
        target_cam_az = self._pose_to_target_camera_azimuth(int(pose_id))

        if elevation_deg is None:
            if self.default_elevation_deg is not None:
                elevation_deg = float(self.default_elevation_deg)
            else:
                st = self.get_ptz_status()
                elevation_deg = float(st["elevation_deg"])

        st_done = self.move_absolute_perfect(
            azimuth_deg=target_cam_az,
            elevation_deg=elevation_deg,
            zoom=zoom,
            horizontal_speed=horizontal_speed,
            vertical_speed=vertical_speed,
            timeout_s=timeout_s,
            poll_s=poll_s,
            prefer_current_elevation=prefer_current_elevation,
        )

        return {
            "pose_id": int(pose_id),
            "target_real_azimuth_deg": self._camera_to_real_azimuth(target_cam_az),
            "target_camera_azimuth_deg": target_cam_az,
            "status": st_done,
        }

    def move_camera(self, operation: str, speed: int = 20, idx: int = 0):
        """
        Supported operations
        Left, Right, Up, Down
        UpLeft, UpRight, DownLeft, DownRight
        ZoomIn, ZoomOut
        Stop
        ToPos uses idx as preset id
          if cam_poses + cam_azimuths provided, it finds the mapped azimuth and reuses Absolute to move there
          else it falls back to ISAPI preset goto without speed control
        Absolute uses idx as azimuth degrees, waits perfectly by default (using default elevation if provided)
        """
        if self.cam_type == "static":
            return None

        op = operation.strip()

        if op == "ToPos":
            preset_id = int(idx)

            if self.cam_poses and self.cam_azimuths:
                if len(self.cam_poses) != len(self.cam_azimuths):
                    raise RuntimeError("cam_poses and cam_azimuths must have the same length")
                if preset_id in self.cam_poses:
                    target_cam_az = self._pose_to_target_camera_azimuth(preset_id)
                    return self.move_absolute_perfect(
                        azimuth_deg=target_cam_az,
                        elevation_deg=self.default_elevation_deg,
                        timeout_s=15.0,
                        poll_s=0.15,
                        prefer_current_elevation=False,
                    )

            path = f"/ISAPI/PTZCtrl/channels/{self.ptz_channel}/presets/{preset_id}/goto"
            resp = self._request("PUT", path)
            ok = self._handle_response(resp, "Preset goto success")
            if ok is None:
                resp2 = self._request("POST", path)
                ok2 = self._handle_response(resp2, "Preset goto success")
                if ok2 is None:
                    raise RuntimeError(f"Preset goto failed, status {resp2.status_code}, body {resp2.text[:300]}")
            time.sleep(0.2)
            return self.get_ptz_status()

        if op == "Stop":
            self._ptz_continuous(pan=0, tilt=0, zoom=0)
            time.sleep(0.15)
            return self.get_ptz_status()

        if op == "Absolute":
            return self.move_absolute_perfect(
                azimuth_deg=float(idx),
                timeout_s=15.0,
                poll_s=0.15,
            )

        pan, tilt, zoom = 0, 0, 0
        v = int(max(1, min(100, speed)))

        if op == "Left":
            pan = -v
        elif op == "Right":
            pan = v
        elif op == "Up":
            tilt = v
        elif op == "Down":
            tilt = -v
        elif op == "UpLeft":
            pan, tilt = -v, v
        elif op == "UpRight":
            pan, tilt = v, v
        elif op == "DownLeft":
            pan, tilt = -v, -v
        elif op == "DownRight":
            pan, tilt = v, -v
        elif op == "ZoomIn":
            zoom = v
        elif op == "ZoomOut":
            zoom = -v
        else:
            raise ValueError(f"Unsupported PTZ operation: {operation}")

        self._ptz_continuous(pan=pan, tilt=tilt, zoom=zoom)
        return None

    def _ptz_continuous(self, pan: int, tilt: int, zoom: int):
        path = f"/ISAPI/PTZCtrl/channels/{self.ptz_channel}/continuous"
        xml = (
            "<?xml version='1.0' encoding='UTF-8'?>"
            "<PTZData version='2.0' xmlns='http://www.std-cgi.com/ver20/XMLSchema'>"
            f"<pan>{pan}</pan>"
            f"<tilt>{tilt}</tilt>"
            f"<zoom>{zoom}</zoom>"
            "</PTZData>"
        )
        resp = self._request("PUT", path, data=xml, headers={"Content-Type": "application/xml"})
        if self._handle_response(resp, "Continuous PTZ success") is None:
            raise RuntimeError(f"Continuous PTZ failed, status {resp.status_code}, body {resp.text[:300]}")

    def move_in_seconds(self, s: float, operation: str = "Right", speed: int = 20, save_path: str = "im.jpg"):
        self.move_camera(operation, speed=speed)
        time.sleep(float(s))
        self.move_camera("Stop")
        time.sleep(0.5)
        im = self.capture()
        if im is not None and save_path:
            im.save(save_path)

    def get_ptz_preset(self) -> Optional[str]:
        path = f"/ISAPI/PTZCtrl/channels/{self.ptz_channel}/presets"
        resp = self._request("GET", path, headers={"Accept": "application/xml"})
        if resp.status_code == 200:
            return resp.text
        self._handle_response(resp)
        return None

    def set_ptz_preset(self, idx: Optional[int] = None, name: Optional[str] = None):
        if idx is None:
            raise ValueError("idx is required for ISAPI preset creation")

        preset_id = int(idx)
        preset_name = name or f"pos{preset_id}"

        path = f"/ISAPI/PTZCtrl/channels/{self.ptz_channel}/presets/{preset_id}"
        xml = (
            "<?xml version='1.0' encoding='UTF-8'?>"
            "<PTZPreset version='2.0' xmlns='http://www.std-cgi.com/ver20/XMLSchema'>"
            f"<id>{preset_id}</id>"
            f"<presetName>{preset_name}</presetName>"
            "</PTZPreset>"
        )
        resp = self._request("PUT", path, data=xml, headers={"Content-Type": "application/xml"})
        if self._handle_response(resp, "Preset saved") is None:
            raise RuntimeError(f"Save preset failed, status {resp.status_code}, body {resp.text[:300]}")

    def save_preset(self, idx: int, name: Optional[str] = None):
        self.set_ptz_preset(idx=idx, name=name)

    def reboot_camera(self) -> bool:
        path = "/ISAPI/System/reboot"
        resp = self._request("PUT", path)
        return self._handle_response(resp, "Reboot requested") is not None

    def get_auto_focus(self):
        raise NotImplementedError("Auto focus retrieval not implemented for Linovision")

    def set_auto_focus(self, disable: bool):
        raise NotImplementedError("Auto focus setting not implemented for Linovision")

    def start_zoom_focus(self, position: int):
        raise NotImplementedError("Zoom/focus control not implemented for Linovision")

    def set_manual_focus(self, position: int):
        self.focus_position = position
        raise NotImplementedError("Manual focus not implemented for Linovision yet")

    def get_focus_level(self):
        raise NotImplementedError("Focus level not implemented for Linovision yet")
