# Copyright (C) 2022-2026, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.


from __future__ import annotations

import logging
import re
import time
import xml.etree.ElementTree as ET
from io import BytesIO
from typing import Callable, List, Optional, Tuple

import requests
import urllib3
from PIL import Image
from requests.auth import HTTPDigestAuth

from pyro_camera_api.camera.base import BaseCamera, FocusMixin, PTZMixin

__all__ = ["HikvisionCamera"]

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logger = logging.getLogger(__name__)

# Hikvision ISAPI XML namespace, as returned by the DS-2DE7A432IWG1-E.
# (Linovision domes answer on the same paths but with the std-cgi namespace,
# which is one of several reasons the two adapters stay separate for now.)
HIKVISION_NS = "http://www.hikvision.com/ver20/XMLSchema"

# Physical tilt range of the DS-2DE7A432IWG1-E, in the dome elevation
# convention used by absoluteEx: 0 = horizon, positive = looking down,
# so -15 is 15 degrees above the horizon and 90 is straight down.
ELEVATION_MIN_DEG = -15.0
ELEVATION_MAX_DEG = 90.0

# Preset ids Hikvision reserves for device functions rather than positions.
# Calling one runs the function (33 = Auto-flip, 94 = Remote reboot,
# 99 = Start auto scan, ...) and writing to one can break the camera's own
# controls. Enumerated from a DS-2DE7A432IWG1-E; Hikvision documents the
# reserved space loosely as 33-64 and 90-105, but only these were present on
# the tested unit, so the block list stays evidence-based.
RESERVED_PRESET_IDS = frozenset({*range(33, 49), 50, 94, *range(96, 106)})

# Continuous PTZ speed range accepted by ISAPI /continuous.
CONTINUOUS_SPEED_MAX = 100.0

# Speed range accepted by absoluteEx horizontalSpeed / verticalSpeed.
ABSOLUTE_SPEED_MIN = 0.1
ABSOLUTE_SPEED_MAX = 80.0

# The rest of the API speaks Reolink-style speeds (1-64) and Reolink-style
# zoom levels (0-64); those are the source ranges we map from.
REOLINK_SPEED_MIN = 1.0
REOLINK_SPEED_MAX = 64.0
REOLINK_ZOOM_MIN = 0.0
REOLINK_ZOOM_MAX = 64.0


class HikvisionCamera(BaseCamera, PTZMixin, FocusMixin):
    """
    Controller for Hikvision PTZ domes over ISAPI, validated on a DS-2DE7A432IWG1-E.

    Verified behaviour of this model, which differs from the Linovision domes:

    * ``GET /ISAPI/PTZCtrl/channels/{ch}/absoluteEx`` reports the live position
      as ``PTZAbsoluteEx`` with ``azimuth`` and ``elevation`` in **decimal
      degrees** (e.g. ``239.58``), not in tenths of a degree.
    * ``absoluteZoom`` is the **optical ratio directly** (``1`` at 1x), not the
      ratio in tenths.
    * ``focus`` is exposed in the same document as a raw motor position.
    * the XML namespace is ``http://www.hikvision.com/ver20/XMLSchema``.

    Azimuth handling mirrors the rest of the codebase: ``cam_azimuths`` holds
    real-world azimuths, and the camera's own reference frame is reached with
    ``camera_az = (real_az + azimuth_offset_deg) % 360``.

    Verified on a DS-2DE7A432IWG1-E: snapshot, PTZ status, azimuth readback,
    absolute moves, relative moves with convergence (settles ~0.15 deg off the
    request, hence the tolerance rather than an equality check), pose moves
    through the azimuth mapping, preset listing, and zoom over the full 32x
    range.

    Not yet confirmed on hardware, and therefore the places to look first if
    something behaves oddly:

    * ``wide_fov_deg`` is taken from the datasheet, not measured. It only
      affects click_to_move.
    * the elevation sign convention (positive = looking down) is inherited from
      the existing click_to_move path rather than independently checked.
    * manual focus is not implemented at all, see ``set_manual_focus``.
    """

    azimuth_source = "hardware"

    # Preset and absolute moves block until the position is reached, so the
    # route layer does not need to hold the camera lock afterwards.
    preset_move_hold_s = 0.0

    def __init__(
        self,
        camera_id: str,
        ip_address: str,
        username: str,
        password: str,
        cam_type: str = "ptz",
        cam_poses: Optional[List[int]] = None,
        cam_azimuths: Optional[List[float]] = None,
        protocol: str = "http",
        verify_tls: bool = False,
        snapshot_channel: str = "101",
        ptz_channel: str = "1",
        focus_position: Optional[int] = None,
        timeout: float = 3.0,
        azimuth_offset_deg: float = 0.0,
        default_elevation_deg: Optional[float] = 0.0,
        zoom_max: float = 32.0,
        wide_fov_deg: Tuple[float, float] = (57.6, 34.5),
        azimuth_tolerance_deg: float = 0.5,
        disable_osd: bool = True,
    ) -> None:
        """
        Args:
            zoom_max: Maximum optical zoom ratio of the model. 32 on the
                DS-2DE7A432IWG1-E ("432" = 32x).
            wide_fov_deg: (horizontal, vertical) field of view at 1x, from the
                model datasheet. Used by click_to_move, which derives the FOV
                at ratio Z as ``2*atan(tan(fov0/2)/Z)``. Override per camera in
                credentials.json when deploying a different Hikvision model.
            azimuth_tolerance_deg: Convergence window when waiting for a move
                to complete. The camera reports two decimals, so an exact
                equality check would never settle.
            disable_osd: Best-effort removal of the burnt-in PTZ overlay, which
                would otherwise appear in the frames sent to the detector.
        """
        super().__init__(camera_id=camera_id, cam_type=cam_type)
        self.ip_address = ip_address
        self.username = username
        self.password = password
        self.cam_poses = cam_poses if cam_poses is not None else []
        # Real-world azimuths, same meaning as on the Reolink adapter.
        self.cam_azimuths = cam_azimuths if cam_azimuths is not None else []
        self.protocol = protocol
        self.verify_tls = verify_tls
        self.snapshot_channel = str(snapshot_channel)
        self.ptz_channel = str(ptz_channel)
        self.focus_position = focus_position
        self.timeout = float(timeout)

        self.azimuth_offset_deg = float(azimuth_offset_deg) % 360.0
        self.default_elevation_deg = default_elevation_deg
        self.zoom_max = float(zoom_max)
        self.wide_fov_deg = (float(wide_fov_deg[0]), float(wide_fov_deg[1]))
        self.azimuth_tolerance_deg = float(azimuth_tolerance_deg)

        self._auth = HTTPDigestAuth(self.username, self.password)
        self._base = f"{self.protocol}://{self.ip_address}"

        # Two valid ways to drive poses. With cam_azimuths, ToPos computes an
        # absolute move; without them it recalls a camera-side ISAPI preset,
        # which also restores zoom and focus. Presets must exist on the camera
        # and use ids outside RESERVED_PRESET_IDS. The platform azimuth sync
        # only serves "tracked" adapters, so nothing fills azimuths in later.
        if self.cam_type != "static" and self.cam_poses and not self.cam_azimuths:
            logger.info(
                "[%s] %d pose(s) configured without azimuths: ToPos will recall camera-side "
                "presets %s. Ensure they are registered on the camera, or set 'azimuths' in "
                "credentials.json to drive poses by absolute angle instead.",
                self.ip_address,
                len(self.cam_poses),
                self.cam_poses,
            )

        if disable_osd:
            self.disable_ptz_osd()

    # ------------------------------------------------------------------
    # HTTP plumbing
    # ------------------------------------------------------------------

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

        try:
            body = resp.text[:500]
        except Exception:
            body = "<unreadable body>"

        logger.error("[%s] ISAPI error, status %s, body %s", self.ip_address, resp.status_code, body)
        return None

    # ------------------------------------------------------------------
    # Small helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _clamp(v: float, vmin: float, vmax: float) -> float:
        return max(vmin, min(vmax, v))

    @staticmethod
    def _map_range(value: float, src_min: float, src_max: float, dst_min: float, dst_max: float) -> float:
        v = max(src_min, min(src_max, float(value)))
        if src_max == src_min:
            return dst_min
        return dst_min + (v - src_min) * (dst_max - dst_min) / (src_max - src_min)

    @staticmethod
    def _angular_diff(a: float, b: float) -> float:
        """Smallest absolute difference between two headings, in degrees."""
        return abs((a - b + 180.0) % 360.0 - 180.0)

    @staticmethod
    def _find_text(root: ET.Element, name: str) -> Optional[str]:
        """Find the first descendant with this local tag name, ignoring namespaces.

        Hikvision firmwares are inconsistent about whether they echo the
        namespace, so matching on the local name keeps parsing robust.
        """
        for el in root.iter():
            tag = el.tag.rsplit("}", 1)[-1]
            if tag == name:
                return el.text
        return None

    def _clamp_elevation(self, elevation_deg: float) -> float:
        return self._clamp(float(elevation_deg), ELEVATION_MIN_DEG, ELEVATION_MAX_DEG)

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
        return self._real_to_camera_azimuth(float(self.cam_azimuths[i]))

    # ------------------------------------------------------------------
    # Capture
    # ------------------------------------------------------------------

    def capture(self, patrol_id: Optional[int] = None, timeout: int = 2) -> Optional[Image.Image]:
        if patrol_id is not None:
            self.move_camera("ToPos", idx=int(patrol_id), speed=0)
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
            logger.error("[%s] Capture failed, %s", self.ip_address, exc)
            return None
        finally:
            self.timeout = old_timeout

    # ------------------------------------------------------------------
    # PTZ status
    # ------------------------------------------------------------------

    def get_ptz_status(self) -> dict:
        """Read the live PTZ position from absoluteEx.

        Unlike the Linovision status endpoint, every angle here is already in
        decimal degrees and absoluteZoom is already the optical ratio, so no
        tenths conversion is applied.
        """
        path = f"/ISAPI/PTZCtrl/channels/{self.ptz_channel}/absoluteEx"
        resp = self._request("GET", path, headers={"Accept": "application/xml"})
        if resp.status_code != 200:
            raise RuntimeError(f"PTZ status failed, status {resp.status_code}, body {resp.text[:200]}")

        root = ET.fromstring(resp.text)
        az_text = self._find_text(root, "azimuth")
        el_text = self._find_text(root, "elevation")
        zoom_text = self._find_text(root, "absoluteZoom")
        focus_text = self._find_text(root, "focus")

        if az_text is None or el_text is None:
            raise RuntimeError(f"Unexpected PTZ status XML, body {resp.text[:400]}")

        azimuth_deg = float(az_text) % 360.0
        elevation_deg = float(el_text)
        zoom_ratio = float(zoom_text) if zoom_text is not None else None
        focus = int(float(focus_text)) if focus_text is not None else None

        return {
            "azimuth_deg": azimuth_deg,
            "elevation_deg": elevation_deg,
            "zoom_ratio": zoom_ratio,
            # Hikvision absoluteZoom is the ratio itself, so raw == ratio here.
            "zoom_raw": zoom_ratio,
            "focus_raw": focus,
            "real_azimuth_deg": self._camera_to_real_azimuth(azimuth_deg),
        }

    def get_azimuth(self) -> Optional[float]:
        """Return the current real-world azimuth read back from the camera."""
        try:
            return float(self.get_ptz_status()["real_azimuth_deg"])
        except Exception as exc:
            logger.warning("[%s] Failed to read azimuth: %s", self.ip_address, exc)
            return None

    def wait_reached_azimuth(
        self,
        target_azimuth_deg: float,
        timeout_s: float = 15.0,
        poll_s: float = 0.15,
    ) -> dict:
        """Poll until the camera azimuth is within tolerance of the target.

        Raises RuntimeError on timeout. The azimuth is compared with a
        tolerance rather than for equality because this model reports two
        decimal places and settles a fraction of a degree off the request.
        """
        target = float(target_azimuth_deg) % 360.0

        deadline = time.time() + timeout_s
        last: Optional[dict] = None
        while time.time() < deadline:
            st = self.get_ptz_status()
            last = st
            if self._angular_diff(float(st["azimuth_deg"]), target) <= self.azimuth_tolerance_deg:
                return st
            time.sleep(poll_s)

        raise RuntimeError(f"Timeout waiting for azimuth {target:.2f}, last={last}")

    def wait_until_stationary(
        self,
        timeout_s: float = 20.0,
        poll_s: float = 0.25,
        still_epsilon_deg: float = 0.05,
        still_polls: int = 2,
    ) -> dict:
        """Poll until the camera stops moving, and return its final status.

        Used for moves where no target angle is known client-side, such as
        preset recall. Movement is considered finished once consecutive
        readings differ by less than ``still_epsilon_deg`` on both axes.
        Returns the last status on timeout rather than raising: the command was
        accepted, and a caller that gets a position is better off than one that
        gets an exception.
        """
        deadline = time.time() + timeout_s
        previous: Optional[dict] = None
        still = 0

        while time.time() < deadline:
            current = self.get_ptz_status()
            if previous is not None:
                az_delta = self._angular_diff(float(current["azimuth_deg"]), float(previous["azimuth_deg"]))
                el_delta = abs(float(current["elevation_deg"]) - float(previous["elevation_deg"]))
                still = still + 1 if (az_delta < still_epsilon_deg and el_delta < still_epsilon_deg) else 0
                if still >= still_polls:
                    return current
            previous = current
            time.sleep(poll_s)

        logger.warning("[%s] Camera still moving after %.1fs, returning last position", self.ip_address, timeout_s)
        return previous if previous is not None else self.get_ptz_status()

    # ------------------------------------------------------------------
    # Movement
    # ------------------------------------------------------------------

    def move_absolute(
        self,
        azimuth_deg: float,
        elevation_deg: Optional[float] = None,
        zoom: Optional[float] = None,
        horizontal_speed: float = 64.0,
        vertical_speed: float = 64.0,
        prefer_current_elevation: bool = False,
    ) -> None:
        """Issue a PTZAbsoluteEx move. Angles are sent in decimal degrees."""
        if self.cam_type == "static":
            return

        az = float(azimuth_deg) % 360.0

        if elevation_deg is not None:
            el = self._clamp_elevation(elevation_deg)
        elif prefer_current_elevation or self.default_elevation_deg is None:
            el = self._clamp_elevation(float(self.get_ptz_status()["elevation_deg"]))
        else:
            el = self._clamp_elevation(float(self.default_elevation_deg))

        # Callers speak Reolink-style speeds (1-64); absoluteEx wants 0.1-80.
        hs = self._map_range(
            horizontal_speed, REOLINK_SPEED_MIN, REOLINK_SPEED_MAX, ABSOLUTE_SPEED_MIN, ABSOLUTE_SPEED_MAX
        )
        vs = self._map_range(
            vertical_speed, REOLINK_SPEED_MIN, REOLINK_SPEED_MAX, ABSOLUTE_SPEED_MIN, ABSOLUTE_SPEED_MAX
        )

        zoom_field = ""
        if zoom is not None:
            # absoluteZoom is the optical ratio on this model (1 = 1x).
            z = self._clamp(float(zoom), 1.0, self.zoom_max)
            zoom_field = f"<absoluteZoom>{z:g}</absoluteZoom>"

        xml = (
            "<?xml version='1.0' encoding='UTF-8'?>"
            f"<PTZAbsoluteEx version='2.0' xmlns='{HIKVISION_NS}'>"
            f"<elevation>{el:g}</elevation>"
            f"<azimuth>{az:g}</azimuth>"
            f"{zoom_field}"
            f"<horizontalSpeed>{hs:g}</horizontalSpeed>"
            f"<verticalSpeed>{vs:g}</verticalSpeed>"
            "</PTZAbsoluteEx>"
        )

        path = f"/ISAPI/PTZCtrl/channels/{self.ptz_channel}/absoluteEx"
        resp = self._request("PUT", path, data=xml, headers={"Content-Type": "application/xml"})
        if self._handle_response(resp, "Absolute move success") is None:
            raise RuntimeError(f"Absolute move failed, status {resp.status_code}, body {resp.text[:300]}")

    def move_absolute_blocking(
        self,
        azimuth_deg: float,
        elevation_deg: Optional[float] = None,
        zoom: Optional[float] = None,
        horizontal_speed: float = 64.0,
        vertical_speed: float = 64.0,
        timeout_s: float = 15.0,
        poll_s: float = 0.15,
        prefer_current_elevation: bool = False,
    ) -> dict:
        """Absolute move that returns only once the azimuth has converged."""
        self.move_absolute(
            azimuth_deg=azimuth_deg,
            elevation_deg=elevation_deg,
            zoom=zoom,
            horizontal_speed=horizontal_speed,
            vertical_speed=vertical_speed,
            prefer_current_elevation=prefer_current_elevation,
        )
        return self.wait_reached_azimuth(azimuth_deg, timeout_s=timeout_s, poll_s=poll_s)

    def move_relative_deg(self, delta_azimuth_deg: float, delta_elevation_deg: float = 0.0) -> dict:
        """Move by a delta from the current position, then wait for both axes.

        Presence of this method is what makes the control routes use the
        hardware closed-loop click_to_move path instead of timed moves.
        """
        st = self.get_ptz_status()
        new_az = (float(st["azimuth_deg"]) + float(delta_azimuth_deg)) % 360.0
        new_el = self._clamp_elevation(float(st["elevation_deg"]) + float(delta_elevation_deg))

        self.move_absolute(new_az, elevation_deg=new_el)

        # Wait on both axes: a pure-tilt move would satisfy an azimuth-only
        # check immediately. A readback that never converges is logged, not
        # raised: the command was accepted and the camera did move.
        converged = False
        deadline = time.time() + 10.0
        while time.time() < deadline:
            st = self.get_ptz_status()
            az_err = self._angular_diff(float(st["azimuth_deg"]), new_az)
            el_err = abs(float(st["elevation_deg"]) - new_el)
            if az_err <= self.azimuth_tolerance_deg and el_err <= self.azimuth_tolerance_deg:
                converged = True
                break
            time.sleep(0.15)

        if not converged:
            logger.warning("[%s] move_relative_deg: position readback did not converge", self.ip_address)
        return {"azimuth_deg": new_az, "elevation_deg": new_el, "converged": converged}

    def move_to_pose(
        self,
        pose_id: int,
        elevation_deg: Optional[float] = None,
        zoom: Optional[float] = None,
        horizontal_speed: float = 64.0,
        vertical_speed: float = 64.0,
        timeout_s: float = 15.0,
        poll_s: float = 0.15,
    ) -> dict:
        """Move to a configured pose using its real-world azimuth mapping."""
        target_cam_az = self._pose_to_target_camera_azimuth(int(pose_id))

        status = self.move_absolute_blocking(
            azimuth_deg=target_cam_az,
            elevation_deg=elevation_deg,
            zoom=zoom,
            horizontal_speed=horizontal_speed,
            vertical_speed=vertical_speed,
            timeout_s=timeout_s,
            poll_s=poll_s,
        )
        return {
            "pose_id": int(pose_id),
            "target_real_azimuth_deg": self._camera_to_real_azimuth(target_cam_az),
            "target_camera_azimuth_deg": target_cam_az,
            "status": status,
        }

    def move_camera(self, operation: str, speed: int = 20, idx: int = 0):
        """
        Perform a PTZ operation.

        Supported operations:
          Left, Right, Up, Down, UpLeft, UpRight, DownLeft, DownRight,
          ZoomIn, ZoomOut  -> continuous motion until Stop
          Stop             -> halt continuous motion
          ToPos            -> idx is the pose id; uses the azimuth mapping when
                              cam_poses/cam_azimuths are configured, otherwise
                              falls back to an ISAPI preset goto
          Absolute         -> idx is a camera-frame azimuth in degrees
        """
        if self.cam_type == "static":
            return None

        op = operation.strip()

        if op == "ToPos":
            return self._goto_pose(int(idx))

        if op == "Stop":
            self._ptz_continuous(pan=0, tilt=0, zoom=0)
            time.sleep(0.15)
            return self.get_ptz_status()

        if op == "Absolute":
            return self.move_absolute_blocking(azimuth_deg=float(idx))

        # Continuous motion. Callers speak Reolink-style speeds (1-64), ISAPI
        # /continuous expects -100..100.
        v = round(self._map_range(speed, REOLINK_SPEED_MIN, REOLINK_SPEED_MAX, 1.0, CONTINUOUS_SPEED_MAX))
        pan, tilt, zoom = 0, 0, 0

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

    def _goto_pose(self, pose_id: int):
        """Absolute move to the pose azimuth, or ISAPI preset goto as fallback."""
        if self.cam_poses and self.cam_azimuths:
            if len(self.cam_poses) != len(self.cam_azimuths):
                raise RuntimeError("cam_poses and cam_azimuths must have the same length")
            if pose_id in self.cam_poses:
                return self.move_absolute_blocking(
                    azimuth_deg=self._pose_to_target_camera_azimuth(pose_id),
                    elevation_deg=self.default_elevation_deg,
                )

        self._reject_reserved_preset(pose_id, "move to")

        path = f"/ISAPI/PTZCtrl/channels/{self.ptz_channel}/presets/{pose_id}/goto"
        resp = self._request("PUT", path)
        if self._handle_response(resp, "Preset goto success") is None:
            raise RuntimeError(f"Preset goto failed, status {resp.status_code}, body {resp.text[:300]}")
        # Preset recall is fire-and-forget and the target angle lives in the
        # camera, so there is nothing to compare against: wait for the reported
        # position to stop changing instead. Without this the caller reads a
        # mid-travel position and the patrol loop can photograph the camera
        # between poses.
        return self.wait_until_stationary()

    def _ptz_continuous(self, pan: int, tilt: int, zoom: int) -> None:
        path = f"/ISAPI/PTZCtrl/channels/{self.ptz_channel}/continuous"
        xml = (
            "<?xml version='1.0' encoding='UTF-8'?>"
            f"<PTZData version='2.0' xmlns='{HIKVISION_NS}'>"
            f"<pan>{pan}</pan>"
            f"<tilt>{tilt}</tilt>"
            f"<zoom>{zoom}</zoom>"
            "</PTZData>"
        )
        resp = self._request("PUT", path, data=xml, headers={"Content-Type": "application/xml"})
        if self._handle_response(resp, "Continuous PTZ success") is None:
            raise RuntimeError(f"Continuous PTZ failed, status {resp.status_code}, body {resp.text[:300]}")

    # ------------------------------------------------------------------
    # Presets
    # ------------------------------------------------------------------

    @staticmethod
    def _reject_reserved_preset(preset_id: int, action: str) -> None:
        """Refuse to touch a preset id that the camera uses as a function key."""
        if preset_id in RESERVED_PRESET_IDS:
            raise ValueError(
                f"Refusing to {action} preset {preset_id}: Hikvision reserves it for a device "
                f"function (reboot, auto-scan, auto-flip, ...), not a position. "
                f"Use an id outside {min(RESERVED_PRESET_IDS)}-{max(RESERVED_PRESET_IDS)}, "
                f"typically 1-32."
            )

    def get_ptz_preset(self) -> Optional[str]:
        path = f"/ISAPI/PTZCtrl/channels/{self.ptz_channel}/presets"
        resp = self._request("GET", path, headers={"Accept": "application/xml"})
        if resp.status_code == 200:
            return resp.text
        self._handle_response(resp)
        return None

    def set_ptz_preset(self, idx: Optional[int] = None, name: Optional[str] = None) -> None:
        if idx is None:
            raise ValueError("idx is required for ISAPI preset creation")

        preset_id = int(idx)
        self._reject_reserved_preset(preset_id, "overwrite")
        preset_name = name or f"pos{preset_id}"
        path = f"/ISAPI/PTZCtrl/channels/{self.ptz_channel}/presets/{preset_id}"
        xml = (
            "<?xml version='1.0' encoding='UTF-8'?>"
            f"<PTZPreset version='2.0' xmlns='{HIKVISION_NS}'>"
            f"<id>{preset_id}</id>"
            f"<presetName>{preset_name}</presetName>"
            "</PTZPreset>"
        )
        resp = self._request("PUT", path, data=xml, headers={"Content-Type": "application/xml"})
        if self._handle_response(resp, "Preset saved") is None:
            raise RuntimeError(f"Save preset failed, status {resp.status_code}, body {resp.text[:300]}")

    def save_preset(self, idx: int, name: Optional[str] = None) -> None:
        self.set_ptz_preset(idx=idx, name=name)

    # ------------------------------------------------------------------
    # Zoom and focus
    # ------------------------------------------------------------------

    def start_zoom_focus(self, position: int) -> Optional[dict]:
        """Set optical zoom, keeping the current azimuth and elevation.

        ``position`` is the Reolink-style 0-64 level used across the API; it is
        mapped onto the camera's optical ratio range (1 to ``zoom_max``).
        """
        if self.cam_type == "static":
            return None

        st = self.get_ptz_status()
        z = self._map_range(float(position), REOLINK_ZOOM_MIN, REOLINK_ZOOM_MAX, 1.0, self.zoom_max)

        self.move_absolute(
            azimuth_deg=float(st["azimuth_deg"]),
            elevation_deg=float(st["elevation_deg"]),
            zoom=z,
        )
        # zoom_raw == zoom_ratio: absoluteZoom carries the ratio on this model.
        return {"zoom_raw": z, "zoom_ratio": z}

    def get_focus_level(self) -> Optional[dict]:
        """Return the raw focus motor position reported by absoluteEx.

        ``zoom`` is deliberately None: the shared contract for this key is the
        Reolink 0-64 zoom *level*, and this camera reports an optical *ratio*.
        Returning the ratio here would silently corrupt the FOV and speed
        limiting logic in the control routes. Callers that want the real zoom
        should read ``get_ptz_status()["zoom_ratio"]``.
        """
        try:
            st = self.get_ptz_status()
        except Exception as exc:
            logger.warning("[%s] Failed to read focus level: %s", self.ip_address, exc)
            return None
        return {"focus": st.get("focus_raw"), "zoom": None}

    def set_manual_focus(self, position: int) -> None:
        """Not implemented: the ISAPI focus write path is unverified on this model.

        absoluteEx reports focus but rejects it as an input, and the
        /Image FocusConfiguration route has not been validated on the
        DS-2DE7A432IWG1-E. The value is stored so the patrol loop's focus
        restore stays a no-op instead of failing.
        """
        self.focus_position = position
        logger.warning("[%s] Manual focus not implemented for Hikvision (position=%s)", self.ip_address, position)

    def focus_finder(
        self,
        save_images: bool = False,
        retry_depth: int = 0,
        should_abort: Optional[Callable[[], bool]] = None,
    ) -> int:
        """Not implemented: depends on set_manual_focus, which is unverified."""
        _ = (save_images, retry_depth, should_abort)
        logger.warning("[%s] Focus finder not implemented for Hikvision", self.ip_address)
        return self.focus_position if self.focus_position is not None else -1

    def get_auto_focus(self) -> None:
        logger.warning("[%s] Auto focus retrieval not implemented for Hikvision", self.ip_address)

    def set_auto_focus(self, disable: bool) -> None:
        logger.warning("[%s] Auto focus setting not implemented for Hikvision (disable=%s)", self.ip_address, disable)

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def reboot_camera(self) -> bool:
        resp = self._request("PUT", "/ISAPI/System/reboot")
        return self._handle_response(resp, "Reboot requested") is not None

    def disable_ptz_osd(self) -> bool:
        """Turn off burnt-in text overlays. Best effort, never raises.

        The overlays endpoint replaces the whole document, and a partial body
        is answered with 200 OK and silently ignored, so the current settings
        are read back and only the ``enabled`` flags are flipped. The edit is
        done on the raw XML rather than through ElementTree so the camera gets
        its own document back byte for byte apart from those flags.

        Note this cannot remove the pan/tilt readout ("P090|T00") that some
        firmwares burn in: it is not part of this document and is not exposed
        over ISAPI on the tested unit. Turn it off in the camera web UI under
        Configuration -> PTZ, where the PT/zoom status display duration lives.
        """
        path = "/ISAPI/System/Video/inputs/channels/1/overlays"
        try:
            current = self._request("GET", path, headers={"Accept": "application/xml"})
            if current.status_code != 200:
                logger.warning("[%s] Could not read overlay settings, status %s", self.ip_address, current.status_code)
                return False

            document = current.text
            for block in ("DateTimeOverlay", "channelNameOverlay", "PTZInfoOverlay"):
                document = re.sub(
                    rf"(<{block}>.*?<enabled>)true(</enabled>)",
                    r"\1false\2",
                    document,
                    flags=re.DOTALL,
                )

            resp = self._request(
                "PUT",
                path,
                data=document.encode("utf-8"),
                headers={"Content-Type": "application/xml"},
            )
        except requests.RequestException as exc:
            logger.warning("[%s] Could not disable OSD: %s", self.ip_address, exc)
            return False

        if self._handle_response(resp, "OSD overlays disabled") is None:
            logger.warning("[%s] Camera rejected the OSD disable request", self.ip_address)
            return False
        return True
