#!/usr/bin/env python3
"""
End-to-end relay test for the Pi Zero watchdog setup.

For each selected relay, this script:
  1) pings the controlled device to confirm it is reachable,
  2) drives the GPIO LOW for POWER_OFF_TIME seconds (cut power),
  3) verifies the device drops during the cut,
  4) drives the GPIO back HIGH (restore power) and waits for the device to come back.

Install dependencies (once per Pi):
  sudo apt install python3-rpi-lgpio

Run on the Pi Zero:
  python3 /home/pi/pyro-engine/watchdog/pi_zero/relay_check.py
  python3 /home/pi/pyro-engine/watchdog/pi_zero/relay_check.py --relay main
  python3 /home/pi/pyro-engine/watchdog/pi_zero/relay_check.py --relay cams

The cams test pings only the first camera in CAM_IPS as a proxy for the 12V rail.
Exits non-zero if any relay fails the verification.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

import RPi.GPIO as GPIO

# ================= CONFIG =================

RELAY_MAIN = 16
RELAY_CAMS = 26

_ENV_FILE = Path("/home/pi/watchdog.env")


def _load_env(path: Path) -> dict:
    env: dict = {}
    try:
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key, _, value = line.partition("=")
            env[key.strip()] = value.strip()
    except FileNotFoundError:
        pass
    return env


_env = _load_env(_ENV_FILE)

MAIN_PI_IP: str = _env.get("MAIN_PI_IP", "192.168.1.99")

_cam_ips_raw = _env.get("CAM_IPS", "")
CAM_IPS: list[str] = [ip.strip() for ip in _cam_ips_raw.split(",") if ip.strip()] or [
    "192.168.1.11",
    "192.168.1.12",
]

PING_COUNT = 1
PING_TIMEOUT = 2
POWER_OFF_TIME = 15

RETURN_TIMEOUT = 90
RETURN_POLL = 3


# ================ HELPERS =================


def ping(ip: str) -> bool:
    try:
        subprocess.check_output(
            ["ping", "-c", str(PING_COUNT), "-W", str(PING_TIMEOUT), ip],
            stderr=subprocess.DEVNULL,
        )
        return True
    except subprocess.CalledProcessError:
        return False


def wait_for_drop(ip: str, deadline_s: int) -> bool:
    # require two consecutive failed pings to debounce transient packet loss
    end = time.time() + deadline_s
    while time.time() < end:
        if not ping(ip) and not ping(ip):
            return True
        time.sleep(1)
    return False


def wait_for_return(ip: str, deadline_s: int, poll_s: int) -> bool:
    end = time.time() + deadline_s
    while time.time() < end:
        if ping(ip):
            return True
        time.sleep(poll_s)
    return False


def test_relay(name: str, gpio_pin: int, target_ip: str) -> bool:
    print(f"\n=== Testing relay '{name}' on GPIO {gpio_pin} (target {target_ip}) ===", flush=True)

    print(f"[BEFORE] ping {target_ip} ...", flush=True)
    if not ping(target_ip):
        print(f"[BEFORE] FAIL: {target_ip} unreachable before cut, aborting test", flush=True)
        return False
    print(f"[BEFORE] OK: {target_ip} reachable", flush=True)

    print(f"[CUT] GPIO {gpio_pin} -> LOW for {POWER_OFF_TIME}s", flush=True)
    GPIO.output(gpio_pin, GPIO.LOW)
    cut_start = time.time()
    try:
        dropped = wait_for_drop(target_ip, deadline_s=POWER_OFF_TIME)
        if not dropped:
            print(
                f"[CUT] FAIL: {target_ip} still reachable after {POWER_OFF_TIME}s -- relay or wiring issue",
                flush=True,
            )
            return False
        print(f"[CUT] OK: {target_ip} dropped after {time.time() - cut_start:.1f}s", flush=True)
        # hold the cut for the remainder of POWER_OFF_TIME to ensure a real power cycle
        remaining = POWER_OFF_TIME - (time.time() - cut_start)
        if remaining > 0:
            time.sleep(remaining)
    finally:
        print(f"[RESTORE] GPIO {gpio_pin} -> HIGH", flush=True)
        GPIO.output(gpio_pin, GPIO.HIGH)

    print(f"[AFTER] waiting up to {RETURN_TIMEOUT}s for {target_ip} to return ...", flush=True)
    if not wait_for_return(target_ip, deadline_s=RETURN_TIMEOUT, poll_s=RETURN_POLL):
        print(f"[AFTER] FAIL: {target_ip} did not come back within {RETURN_TIMEOUT}s", flush=True)
        return False
    print(f"[AFTER] OK: {target_ip} reachable again", flush=True)
    return True


# ================== MAIN ==================


RELAYS = {
    "main": (RELAY_MAIN, lambda: MAIN_PI_IP),
    "cams": (RELAY_CAMS, lambda: CAM_IPS[0]),
}


def main() -> int:
    parser = argparse.ArgumentParser(description="End-to-end relay test for Pi Zero")
    parser.add_argument(
        "--relay",
        choices=[*RELAYS.keys(), "all"],
        default="all",
        help="Relay to test (default: all)",
    )
    args = parser.parse_args()

    selected = list(RELAYS.keys()) if args.relay == "all" else [args.relay]

    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    for name in selected:
        pin, _ = RELAYS[name]
        GPIO.setup(pin, GPIO.OUT, initial=GPIO.HIGH)

    failures: list[str] = []
    try:
        for name in selected:
            pin, ip_getter = RELAYS[name]
            if not test_relay(name, pin, ip_getter()):
                failures.append(name)
    finally:
        for name in selected:
            pin, _ = RELAYS[name]
            GPIO.output(pin, GPIO.HIGH)
        GPIO.cleanup()

    print("", flush=True)
    if failures:
        print(f"FAILED: {', '.join(failures)}", flush=True)
        return 1
    print("ALL RELAYS OK", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
