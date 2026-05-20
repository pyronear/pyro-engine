# Copyright (C) 2022-2026, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.


import argparse
import os
import sys
import time
from typing import Any, Dict, List, Optional

import requests
import urllib3
from dotenv import load_dotenv

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

PAN_STEP_DEG = 45.0
REFERENCE_PRESET = 10

# Reolink 823S2 calibrated values (see pyro_camera_api routes_control.py).
PAN_SPEED_LEVEL = 5
PAN_DEG_PER_SEC = 7.1806
PAN_BIAS_DEG = 2.4465

SETTLE_SECONDS = 1.5
GOTO_SETTLE_SECONDS = 6.0
HTTP_TIMEOUT = 5.0


class ReolinkClient:
    def __init__(self, ip: str, user: str, password: str, protocol: str = "https"):
        self.ip = ip
        self.user = user
        self.password = password
        self.protocol = protocol

    def _scrub(self, text: str) -> str:
        return text.replace(self.password, "***") if self.password else text

    def _post(self, command: str, payload: List[Dict[str, Any]]) -> Optional[List[Dict[str, Any]]]:
        url = f"{self.protocol}://{self.ip}/cgi-bin/api.cgi"
        params = {"cmd": command, "user": self.user, "password": self.password, "channel": 0}
        try:
            resp = requests.post(url, params=params, json=payload, verify=False, timeout=HTTP_TIMEOUT)  # nosec: B501
        except requests.RequestException as exc:
            print(f"❌ {self.ip}: request failed: {self._scrub(str(exc))}")
            return None
        if resp.status_code != 200:
            print(f"❌ {self.ip}: HTTP {resp.status_code}: {self._scrub(resp.text)}")
            return None
        try:
            data = resp.json()
        except ValueError:
            print(f"❌ {self.ip}: non-JSON response: {self._scrub(resp.text)[:200]}")
            return None
        if not data or data[0].get("code") != 0:
            print(f"❌ {self.ip}: camera returned error: {self._scrub(str(data))}")
            return None
        return data

    def has_preset(self, idx: int) -> bool:
        data = self._post("GetPtzPreset", [{"cmd": "GetPtzPreset", "action": 1, "param": {"channel": 0}}])
        if data is None:
            return False
        presets = data[0].get("value", {}).get("PtzPreset", [])
        return any(p.get("id") == idx and p.get("enable") == 1 for p in presets)

    def ptz(self, operation: str, speed: int = 20, idx: int = 0) -> bool:
        return (
            self._post(
                "PtzCtrl",
                [{"cmd": "PtzCtrl", "action": 0, "param": {"channel": 0, "op": operation, "id": idx, "speed": speed}}],
            )
            is not None
        )

    def pan_degrees(self, direction: str, degrees: float) -> bool:
        duration = max(0.0, (degrees - PAN_BIAS_DEG) / PAN_DEG_PER_SEC)
        if not self.ptz(direction, speed=PAN_SPEED_LEVEL):
            return False
        try:
            time.sleep(duration)
        finally:
            self.ptz("Stop")
        time.sleep(SETTLE_SECONDS)
        return True

    def set_preset(self, idx: int) -> bool:
        return (
            self._post(
                "SetPtzPreset",
                [
                    {
                        "cmd": "SetPtzPreset",
                        "action": 0,
                        "param": {"PtzPreset": {"channel": 0, "enable": 1, "id": idx, "name": f"pos{idx}"}},
                    }
                ],
            )
            is not None
        )


def process_camera(ip: str, user: str, password: str, protocol: str) -> None:
    print(f"\n🔧 Processing camera {ip}")
    cam = ReolinkClient(ip, user, password, protocol=protocol)

    if not cam.has_preset(REFERENCE_PRESET):
        print(f"⚠️  Camera {ip}: reference preset {REFERENCE_PRESET} not configured — skipping")
        return

    print(f"🧭 Going to reference preset {REFERENCE_PRESET}")
    if not cam.ptz("ToPos", speed=50, idx=REFERENCE_PRESET):
        print(f"❌ {ip}: failed to reach reference preset — aborting")
        return
    time.sleep(GOTO_SETTLE_SECONDS)

    print(f"⬅️  Panning {2 * PAN_STEP_DEG:.0f}° left")
    if not cam.pan_degrees("Left", 2 * PAN_STEP_DEG):
        print(f"❌ {ip}: left pan failed — aborting")
        return

    for preset_idx in (0, 1, 2, 3):
        if preset_idx > 0:
            print(f"➡️  Panning {PAN_STEP_DEG:.0f}° right")
            if not cam.pan_degrees("Right", PAN_STEP_DEG):
                print(f"❌ {ip}: right pan failed — aborting (preset {preset_idx} not saved)")
                return
        print(f"💾 Registering PTZ preset {preset_idx}")
        if not cam.set_preset(preset_idx):
            print(f"❌ {ip}: failed to save preset {preset_idx} — aborting")
            return


def main() -> None:
    load_dotenv()
    user = os.getenv("CAM_USER")
    password = os.getenv("CAM_PWD")
    if not user or not password:
        print("❌ Missing CAM_USER or CAM_PWD (set them in your .env)")
        sys.exit(1)

    parser = argparse.ArgumentParser(
        description="Register PTZ presets 0-3 around reference preset 10 on Reolink cameras."
    )
    parser.add_argument("ips", nargs="+", help="One or more camera IPs, e.g. 192.168.1.11 192.168.1.12")
    parser.add_argument("--protocol", default="https", choices=["http", "https"])
    args = parser.parse_args()

    for ip in args.ips:
        process_camera(ip, user, password, args.protocol)


if __name__ == "__main__":
    main()
