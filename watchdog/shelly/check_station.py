#!/usr/bin/env python3
"""
Station commissioning test for the Shelly watchdog setup.

Runs from the main Pi (on-site commissioning) or from your PC (bench check
after setup, or remotely over the VPN once the station is installed). The
mode is auto-detected (overridable with --mode):

  both       Shelly reachable, pi_watchdog script enabled and running
  both       outputs 0 and 1 are ON and configured to power on at boot
  both       hardening applied (cloud / AP / BLE off, eco mode on)
  both       the Shelly itself can fetch the Pi /health URL configured in
             watchdog.js (the exact check watchdog.js performs)
  both       Pi /health answers 200, cameras answer ping
  pi only    cron entry installed, station internet check

Wiring tests (optional, cut power for real):

  --cycle-cameras   power-cycle output 0 (cameras / router 12V) and verify
                    the first camera actually drops then comes back.
  --cycle-pi        power-cycle output 1 (main Pi). From a PC: waits for
                    /health to come back (up to 5 min). From the Pi: cuts
                    its own power (the SSH session drops), the Shelly
                    restores the output after 15 s and the Pi reboots —
                    re-run check_station.py after reboot, it detects the
                    reboot and confirms the wiring.

Station IPs are fixed and used as defaults:

  Shelly    192.168.1.97   (--shelly-ip to override)
  main Pi   192.168.1.99   (--pi-url to override the /health URL)
  cameras   192.168.1.11 192.168.1.12   (--cam-ips to override)

The Pi health URL actually tested is the one extracted from the watchdog.js
code uploaded on the Shelly, so what is tested is what actually runs.

Usage:
  python3 check_station.py                     # all non-destructive checks
  python3 check_station.py --cycle-cameras    # + real power-cycle of output 0
  python3 check_station.py --cycle-pi         # + real power-cycle of the Pi (PC only)
"""

import argparse
import json
import re
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

DEFAULT_SHELLY_IP = "192.168.1.97"
DEFAULT_PI_URL = "http://192.168.1.99:8081/health"
DEFAULT_CAM_IPS = ["192.168.1.11", "192.168.1.12"]
SCRIPT_NAME = "pi_watchdog"
CYCLE_PI_MARKER = Path.home() / ".check_station_cycle_pi"
WATCHDOG_OUTPUTS = [0, 1]
CRON_PATTERN = "watchdog/shelly/main_pi/watchdog.py"

PASS = 0
FAIL = 0
WARN = 0


def ok(msg: str) -> None:
    global PASS
    PASS += 1
    print(f"\033[32m  PASS\033[0m  {msg}")


def ko(msg: str) -> None:
    global FAIL
    FAIL += 1
    print(f"\033[31m  FAIL\033[0m  {msg}")


def warn(msg: str) -> None:
    global WARN
    WARN += 1
    print(f"\033[33m  WARN\033[0m  {msg}")


def skip(msg: str) -> None:
    print(f"\033[90m  SKIP\033[0m  {msg}")


def section(title: str) -> None:
    print(f"\n=== {title} ===")


# ---------------------------------------------------------------- helpers


def rpc(shelly_ip: str, method: str, params: dict | None = None, timeout: int = 8):
    """Call a Shelly Gen2 RPC method, return the result dict or raise."""
    body = json.dumps({"id": 1, "method": method, "params": params or {}}).encode()
    req = Request(
        f"http://{shelly_ip}/rpc",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode())
    if "error" in data:
        raise RuntimeError(f"{method}: {data['error']}")
    return data.get("result", {})


def rpc_optional(shelly_ip: str, method: str, params: dict | None = None):
    """RPC that returns None instead of raising (beta firmwares miss methods)."""
    try:
        return rpc(shelly_ip, method, params)
    except Exception:
        return None


def ping_host(ip: str, count: int = 2, timeout: int = 2) -> bool:
    try:
        subprocess.check_output(
            ["ping", "-c", str(count), "-W", str(timeout), ip],
            stderr=subprocess.DEVNULL,
        )
        return True
    except Exception:
        return False


def http_status(url: str, timeout: int = 5) -> int:
    try:
        with urlopen(url, timeout=timeout) as resp:
            return resp.status
    except Exception:
        return 0


def extract_pi_url(script_code: str | None) -> str | None:
    if not script_code:
        return None
    m = re.search(r'PI_URL\s*=\s*"([^"]+)"', script_code)
    return m.group(1) if m else None


def linux_boot_time() -> int | None:
    """Epoch of the last boot, from /proc/stat (Linux only)."""
    try:
        for line in Path("/proc/stat").read_text().splitlines():
            if line.startswith("btime "):
                return int(line.split()[1])
    except Exception:
        pass
    return None


def detect_mode(pi_url: str, shelly_ip: str) -> str:
    """'pi' when this machine is the one watchdog.js monitors, else 'pc'."""
    host = urlsplit(pi_url).hostname
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect((shelly_ip, 80))
        local_ip = s.getsockname()[0]
        s.close()
        return "pi" if local_ip == host else "pc"
    except Exception:
        return "pc"


# ---------------------------------------------------------------- checks


def check_shelly(shelly_ip: str) -> str | None:
    """Device, script, outputs and hardening. Returns watchdog.js code or None."""
    section(f"Shelly at {shelly_ip}")

    try:
        info = rpc(shelly_ip, "Shelly.GetDeviceInfo")
    except Exception as exc:
        ko(f"Shelly unreachable: {exc}")
        return None
    ok(f"Shelly reachable ({info.get('app', '?')} fw {info.get('ver', '?')})")

    # pi_watchdog script installed, enabled, running
    code = None
    scripts = (rpc_optional(shelly_ip, "Script.List") or {}).get("scripts", [])
    match = [s for s in scripts if s.get("name") == SCRIPT_NAME]
    if not match:
        ko(f"no script named {SCRIPT_NAME} on the device (run setup_shelly_watchdog.sh)")
    else:
        script = match[0]
        if script.get("enable"):
            ok(f"{SCRIPT_NAME} enabled on boot")
        else:
            ko(f"{SCRIPT_NAME} not enabled on boot")
        if script.get("running"):
            ok(f"{SCRIPT_NAME} running")
        else:
            ko(f"{SCRIPT_NAME} not running")
        got = rpc_optional(shelly_ip, "Script.GetCode", {"id": script["id"]})
        code = (got or {}).get("data")

    # outputs ON now and ON at boot
    for out_id in WATCHDOG_OUTPUTS:
        status = rpc_optional(shelly_ip, "Switch.GetStatus", {"id": out_id})
        if status is None:
            ko(f"output {out_id}: Switch.GetStatus failed")
            continue
        if status.get("output") is True:
            ok(f"output {out_id} is ON")
        else:
            ko(f"output {out_id} is OFF — the station is not powered through it right now")
        config = rpc_optional(shelly_ip, "Switch.GetConfig", {"id": out_id}) or {}
        if config.get("initial_state") == "on":
            ok(f"output {out_id} initial_state is 'on' (survives a Shelly reboot)")
        else:
            ko(
                f"output {out_id} initial_state is {config.get('initial_state')!r}, "
                "expected 'on' (run harden_shelly.sh)"
            )

    # hardening (warn-level: the watchdog works without it)
    cloud = rpc_optional(shelly_ip, "Cloud.GetConfig") or {}
    (ok if cloud.get("enable") is False else warn)("cloud disabled" if cloud.get("enable") is False else "cloud still enabled")
    wifi = rpc_optional(shelly_ip, "WiFi.GetConfig") or {}
    ap_on = (wifi.get("ap") or {}).get("enable")
    (ok if ap_on is False else warn)("WiFi AP disabled" if ap_on is False else "WiFi AP still enabled")
    ble = rpc_optional(shelly_ip, "BLE.GetConfig") or {}
    (ok if ble.get("enable") is False else warn)("BLE disabled" if ble.get("enable") is False else "BLE still enabled")
    sysconf = rpc_optional(shelly_ip, "Sys.GetConfig") or {}
    eco = (sysconf.get("device") or {}).get("eco_mode")
    (ok if eco is True else warn)("eco mode on" if eco is True else "eco mode off")

    return code


def check_shelly_to_pi(shelly_ip: str, pi_url: str, extracted_url: str | None) -> None:
    """Ask the Shelly itself to fetch the Pi /health URL from watchdog.js."""
    section("Path Shelly -> Pi (the check watchdog.js performs)")

    if extracted_url is None:
        warn(f"PI_URL not readable from the uploaded watchdog.js, using {pi_url}")
    elif extracted_url != DEFAULT_PI_URL:
        warn(f"uploaded watchdog.js targets {extracted_url}, not the standard {DEFAULT_PI_URL}")
    else:
        ok(f"PI_URL configured in the uploaded script: {extracted_url}")

    try:
        result = rpc(shelly_ip, "HTTP.GET", {"url": pi_url, "timeout": 5}, timeout=15)
    except Exception as exc:
        ko(f"the Shelly cannot fetch {pi_url}: {exc}")
        return
    if result.get("code") == 200:
        ok("the Shelly fetches the Pi /health and gets 200 — watchdog.js will see the Pi as alive")
    else:
        ko(f"the Shelly fetched {pi_url} but got code {result.get('code')}")


def check_previous_pi_cycle() -> None:
    """Verdict of a --cycle-pi launched from the Pi before its power was cut."""
    if not CYCLE_PI_MARKER.exists():
        return
    try:
        cut_ts = int(CYCLE_PI_MARKER.read_text().strip())
    except Exception:
        cut_ts = None
    boot_ts = linux_boot_time()
    if cut_ts and boot_ts:
        if boot_ts > cut_ts:
            ok("Pi rebooted after the last --cycle-pi — output 1 wiring confirmed")
        else:
            ko("Pi has NOT rebooted since the last --cycle-pi — check the output 1 wiring")
    CYCLE_PI_MARKER.unlink(missing_ok=True)


def check_pi_side(cam_ips: list[str], pi_url: str, mode: str) -> None:
    section("Main Pi side" + (" (checked remotely)" if mode == "pc" else ""))

    if mode == "pi":
        check_previous_pi_cycle()

    status = http_status(pi_url)
    if status == 200:
        ok(f"{pi_url} answers 200 from this machine")
    else:
        ko(f"{pi_url} answers {status or 'nothing'} from this machine")

    # cron entry for the Pi watchdog (only meaningful on the Pi itself)
    if mode == "pi":
        try:
            crontab = subprocess.check_output(["crontab", "-l"], stderr=subprocess.DEVNULL).decode()
            if CRON_PATTERN in crontab:
                ok(f"cron entry found for {CRON_PATTERN}")
            else:
                ko(f"no cron entry mentioning {CRON_PATTERN} (crontab -e to add it)")
        except Exception:
            warn("could not read the crontab")
    else:
        skip("cron entry check: run this script on the Pi to verify it")

    # internet: locally on the Pi, through the Shelly when remote
    if mode == "pi":
        for url in ("https://clients3.google.com/generate_204", "http://cp.cloudflare.com"):
            if http_status(url) in (200, 204):
                ok(f"internet OK ({url})")
                break
        else:
            if any(ping_host(ip, count=1) for ip in ("1.1.1.1", "8.8.8.8")):
                ok("internet OK (ping fallback)")
            else:
                ko("no internet connectivity")
    else:
        skip("station internet check: run this script on the Pi to verify it")

    # cameras
    for ip in cam_ips:
        if ping_host(ip):
            ok(f"camera {ip} answers ping")
        else:
            ko(f"camera {ip} does not answer ping" + (" (is the VPN/route up?)" if mode == "pc" else ""))


def cycle_cameras_test(shelly_ip: str, cam_ips: list[str], off_seconds: int = 15) -> None:
    """Cut output 0 for real and verify the first camera drops then recovers."""
    section("Wiring test: power-cycle output 0 (cameras / router 12V)")

    cam = cam_ips[0]

    if not ping_host(cam):
        ko(f"camera {cam} already down, aborting the wiring test")
        return
    ok(f"camera {cam} up before the cut")

    print(f"  cutting output 0 for {off_seconds}s (Shelly-side toggle_after restore)...")
    try:
        rpc(shelly_ip, "Switch.Set", {"id": 0, "on": False, "toggle_after": off_seconds})
    except Exception as exc:
        # if the router rail is on output 0, losing the LAN before the HTTP
        # response is expected; toggle_after still restores the output
        warn(f"Switch.Set response not received ({exc}) — expected if the router is on output 0")

    time.sleep(4)
    if ping_host(cam, count=1, timeout=1):
        ko(f"camera {cam} still answers while output 0 is OFF — it is NOT wired on output 0")
    else:
        ok(f"camera {cam} dropped while output 0 is OFF — wiring confirmed")

    print("  waiting for the output to come back and the camera to boot (up to 180s)...")
    time.sleep(max(0, off_seconds - 4) + 2)

    deadline = time.time() + 180
    back = False
    while time.time() < deadline:
        if ping_host(cam, count=1, timeout=2):
            back = True
            break
        time.sleep(5)
    if back:
        ok(f"camera {cam} is back after the cycle")
    else:
        ko(f"camera {cam} did not come back within 180s — check it manually NOW")

    status = rpc_optional(shelly_ip, "Switch.GetStatus", {"id": 0}) or {}
    if status.get("output") is True:
        ok("output 0 restored ON by toggle_after")
    else:
        ko("output 0 still OFF — turn it back on manually: "
           f"curl 'http://{shelly_ip}/rpc/Switch.Set?id=0&on=true'")


def cycle_pi_test(shelly_ip: str, pi_url: str, mode: str, off_seconds: int = 15) -> None:
    """Cut output 1 for real and wait for the Pi /health to come back."""
    section("Wiring test: power-cycle output 1 (main Pi)")

    if mode == "pi":
        print("  !!! cutting power to THIS machine in 5s — the SSH session will drop")
        print(f"  !!! the Shelly restores output 1 after {off_seconds}s, then the Pi reboots")
        print("  !!! after reboot, re-run check_station.py: it will confirm the reboot")
        print("  !!! Ctrl+C now to abort")
        sys.stdout.flush()
        time.sleep(5)
        CYCLE_PI_MARKER.write_text(str(int(time.time())))
        try:
            rpc(shelly_ip, "Switch.Set", {"id": 1, "on": False, "toggle_after": off_seconds})
        except Exception as exc:
            CYCLE_PI_MARKER.unlink(missing_ok=True)
            ko(f"Switch.Set failed: {exc}")
            return
        # power should drop before this sleep ends; surviving it means the cut
        # never reached this machine
        time.sleep(off_seconds + 15)
        CYCLE_PI_MARKER.unlink(missing_ok=True)
        ko("still alive after the cut — the Pi is NOT wired on output 1")
        return

    if http_status(pi_url) != 200:
        ko(f"{pi_url} not answering before the cut, aborting")
        return
    ok("Pi /health up before the cut")

    print(f"  cutting output 1 for {off_seconds}s (Shelly-side toggle_after restore)...")
    try:
        rpc(shelly_ip, "Switch.Set", {"id": 1, "on": False, "toggle_after": off_seconds})
    except Exception as exc:
        ko(f"Switch.Set failed: {exc}")
        return

    time.sleep(4)
    if http_status(pi_url, timeout=2) == 200:
        ko("Pi still answers while output 1 is OFF — it is NOT wired on output 1")
    else:
        ok("Pi dropped while output 1 is OFF — wiring confirmed")

    print("  waiting for the Pi to reboot and /health to answer again (up to 300s)...")
    time.sleep(max(0, off_seconds - 4) + 2)

    deadline = time.time() + 300
    back = False
    while time.time() < deadline:
        if http_status(pi_url, timeout=3) == 200:
            back = True
            break
        time.sleep(5)
    if back:
        ok("Pi /health is back after the cycle")
    else:
        ko("Pi /health did not come back within 300s — check the station NOW")

    status = rpc_optional(shelly_ip, "Switch.GetStatus", {"id": 1}) or {}
    if status.get("output") is True:
        ok("output 1 restored ON by toggle_after")
    else:
        ko("output 1 still OFF — turn it back on manually: "
           f"curl 'http://{shelly_ip}/rpc/Switch.Set?id=1&on=true'")


# ----------------------------------------------------------------- main


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--shelly-ip",
        default=DEFAULT_SHELLY_IP,
        help=f"Shelly IP (default: {DEFAULT_SHELLY_IP})",
    )
    parser.add_argument(
        "--pi-url",
        help=f"Pi health URL (default: the one in the uploaded watchdog.js, else {DEFAULT_PI_URL})",
    )
    parser.add_argument(
        "--cam-ips",
        nargs="+",
        default=DEFAULT_CAM_IPS,
        metavar="IP",
        help=f"camera IPs to check (default: {' '.join(DEFAULT_CAM_IPS)})",
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "pi", "pc"],
        default="auto",
        help="where this script runs (default: auto-detect)",
    )
    parser.add_argument(
        "--cycle-cameras",
        action="store_true",
        help="REALLY power-cycle output 0 to prove the cameras are wired on it",
    )
    parser.add_argument(
        "--cycle-pi",
        action="store_true",
        help="REALLY power-cycle output 1; from the Pi itself this cuts your session, "
        "re-run the script after reboot to get the verdict",
    )
    args = parser.parse_args()

    shelly_ip = args.shelly_ip

    script_code = check_shelly(shelly_ip)
    extracted_url = extract_pi_url(script_code)
    pi_url = args.pi_url or extracted_url or DEFAULT_PI_URL

    mode = args.mode if args.mode != "auto" else detect_mode(pi_url, shelly_ip)
    print(f"\nMode: {mode}" + (" (auto-detected)" if args.mode == "auto" else ""))

    check_shelly_to_pi(shelly_ip, pi_url, extracted_url)
    check_pi_side(args.cam_ips, pi_url, mode)

    if args.cycle_cameras:
        cycle_cameras_test(shelly_ip, args.cam_ips)
    if args.cycle_pi:
        cycle_pi_test(shelly_ip, pi_url, mode)
    if not args.cycle_cameras and not args.cycle_pi:
        print("\n(wiring not proven: --cycle-cameras cuts output 0, --cycle-pi cuts output 1)")

    print("\n================================")
    print(f"  PASS: {PASS}   FAIL: {FAIL}   WARN: {WARN}")
    print("================================")
    if FAIL == 0:
        print("Station OK.")
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
