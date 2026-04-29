#!/usr/bin/env python3
"""
End-to-end relay test for the main Pi watchdog setup.

For each relay, this script:
  1) pings the controlled device to confirm it is reachable,
  2) drives the GPIO LOW for POWER_OFF_TIME seconds (cut power),
  3) verifies the device drops during the cut,
  4) drives the GPIO back HIGH (restore power) and waits for the device to come back.

Run on the main Pi:
  python3 /home/pi/pyro-engine/watchdog/main_pi/test_relays.py

Exits non-zero if any relay fails the verification.
"""

import subprocess
import sys
import time
from pathlib import Path

import RPi.GPIO as GPIO

# ================= CONFIG =================

RELAY_PIZERO = 16

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

PIZERO_IP: str = _env.get("PIZERO_IP", "192.168.1.98")

PING_COUNT = 1
PING_TIMEOUT = 2
POWER_OFF_TIME = 15

DROP_GRACE = 5
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
    end = time.time() + deadline_s
    while time.time() < end:
        if not ping(ip):
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
    try:
        dropped = wait_for_drop(target_ip, deadline_s=DROP_GRACE)
        if not dropped:
            print(
                f"[CUT] FAIL: {target_ip} still reachable after {DROP_GRACE}s -- relay or wiring issue",
                flush=True,
            )
            time.sleep(max(0, POWER_OFF_TIME - DROP_GRACE))
            return False
        print(f"[CUT] OK: {target_ip} dropped", flush=True)
        time.sleep(max(0, POWER_OFF_TIME - DROP_GRACE))
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


def main() -> int:
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    GPIO.setup(RELAY_PIZERO, GPIO.OUT, initial=GPIO.HIGH)

    try:
        ok = test_relay("pizero", RELAY_PIZERO, PIZERO_IP)
    finally:
        GPIO.output(RELAY_PIZERO, GPIO.HIGH)
        GPIO.cleanup()

    print("", flush=True)
    if not ok:
        print("FAILED: pizero", flush=True)
        return 1
    print("ALL RELAYS OK", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
