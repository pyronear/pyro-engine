# Copyright (C) 2022-2026, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.


import argparse
import concurrent.futures
import os
import sys
import time
from typing import Any, Dict, List, Optional

import requests
import urllib3
from dotenv import load_dotenv

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

PAN_STEP_DEG = 45.0

# Reolink 823S2 calibrated values (see pyro_camera_api routes_control.py).
PAN_SPEED_LEVEL = 5
PAN_DEG_PER_SEC = 7.1806
PAN_BIAS_DEG = 2.4465

SETTLE_SECONDS = 1.5
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
        # Reolink CGI does not URL-decode query values, so credentials are interpolated
        # raw (matches the production ReolinkCamera adapter). Passwords containing
        # '&', '#' or '=' will break this — keep them out of CAM_PWD.
        url = (
            f"{self.protocol}://{self.ip}/cgi-bin/api.cgi?"
            f"cmd={command}&user={self.user}&password={self.password}&channel=0"
        )
        try:
            resp = requests.post(url, json=payload, verify=False, timeout=HTTP_TIMEOUT)  # nosec: B501
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
    def log(msg: str) -> None:
        print(f"[{ip}] {msg}", flush=True)

    log("🔧 starting from current position")
    cam = ReolinkClient(ip, user, password, protocol=protocol)

    log(f"⬅️  Panning {2 * PAN_STEP_DEG:.0f}° left")
    if not cam.pan_degrees("Left", 2 * PAN_STEP_DEG):
        log("❌ left pan failed — aborting")
        return

    for preset_idx in (0, 1, 2, 3):
        if preset_idx > 0:
            log(f"➡️  Panning {PAN_STEP_DEG:.0f}° right")
            if not cam.pan_degrees("Right", PAN_STEP_DEG):
                log(f"❌ right pan failed — aborting (preset {preset_idx} not saved)")
                return
        log(f"💾 Registering PTZ preset {preset_idx}")
        if not cam.set_preset(preset_idx):
            log(f"❌ failed to save preset {preset_idx} — aborting")
            return
    log("✅ done")


def main() -> None:
    load_dotenv()
    user = os.getenv("CAM_USER")
    password = os.getenv("CAM_PWD")
    if not user or not password:
        print("❌ Missing CAM_USER or CAM_PWD (set them in your .env)")
        sys.exit(1)

    parser = argparse.ArgumentParser(
        description="Register PTZ presets 0-3 starting from each camera's current position."
    )
    parser.add_argument("ips", nargs="+", help="One or more camera IPs, e.g. 192.168.1.11 192.168.1.12")
    parser.add_argument("--protocol", default="https", choices=["http", "https"])
    args = parser.parse_args()

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(args.ips)) as executor:
        futures = [executor.submit(process_camera, ip, user, password, args.protocol) for ip in args.ips]
        for future in concurrent.futures.as_completed(futures):
            future.result()


if __name__ == "__main__":
    main()
