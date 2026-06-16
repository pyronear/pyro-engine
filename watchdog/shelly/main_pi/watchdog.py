#!/usr/bin/env python3
"""
Watchdog for the main Pi.

It checks internet connectivity and pings the camera IPs. After repeated
failures it asks the Shelly to power-cycle output 0 (cameras / router 12V),
with a cooldown and a daily reboot limit.

The main Pi health itself is monitored by the Shelly script (watchdog.js),
so this watchdog does not check or reboot the main Pi.

Cron setup (every 10 minutes at :05):
  1) Edit crontab:  crontab -e
  2) Add the line:
     5,15,25,35,45,55 * * * * /usr/bin/python3 /home/pi/pyro-engine/watchdog/shelly/main_pi/watchdog.py >> /home/pi/watchdog_main.log 2>&1

Adjust the paths to match where this repo lives on the Pi.
"""

import datetime as dt
import logging
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

# ================= CONFIG =================

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

SHELLY_IP: str = _env.get("SHELLY_IP", "192.168.1.97")
SHELLY_OUTPUT_ID = int(_env.get("SHELLY_OUTPUT_ID", "0"))

_cam_ips_raw = _env.get("CAM_IPS", "")
CAM_IPS: list[str] = [ip.strip() for ip in _cam_ips_raw.split(",") if ip.strip()] or [
    "192.168.1.11",
    "192.168.1.12",
]

_INTERNET_HTTP_URLS = [
    "https://clients3.google.com/generate_204",
    "https://connectivitycheck.gstatic.com/generate_204",
    "http://cp.cloudflare.com",
    "http://www.msftconnecttest.com/connecttest.txt",
]
_INTERNET_PING_IPS = ["1.1.1.1", "8.8.8.8", "9.9.9.9"]

PING_COUNT = 2
TIMEOUT = 2  # seconds, used for ping and HTTP timeout

MAX_FAILS = 3
POWER_OFF_TIME = 20

COOLDOWN_SECONDS = 30 * 60
MAX_REBOOTS_PER_DAY = 3

STATE_DIR = Path("/tmp")
LOG_FILE = Path("/home/pi/watchdog_main.log")

FAIL_INTERNET_FILE = STATE_DIR / "fail_internet"
FAIL_CAM_FILES = {ip: STATE_DIR / f"fail_cam_{ip.replace('.', '_')}" for ip in CAM_IPS}

LAST_REBOOT_FILE = STATE_DIR / "last_reboot_output0"
DAILY_REBOOT_FILE = STATE_DIR / "daily_reboots_output0"

# ================ LOGGING =================

logging.basicConfig(
    filename=str(LOG_FILE),
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ================ IO HELPERS ==============


def read_int(path: Path, default: int = 0) -> int:
    try:
        return int(path.read_text().strip())
    except Exception:
        return default


def write_int(path: Path, value: int) -> None:
    path.write_text(str(value))


def read_text(path: Path, default: str = "") -> str:
    try:
        return path.read_text().strip()
    except Exception:
        return default


def write_text(path: Path, value: str) -> None:
    path.write_text(value)


# ================ CHECKS ==================


def ping_host(ip: str, count: int = PING_COUNT, timeout: int = TIMEOUT) -> bool:
    try:
        subprocess.check_output(
            ["ping", "-c", str(count), "-W", str(timeout), ip],
            stderr=subprocess.DEVNULL,
        )
        return True
    except subprocess.CalledProcessError:
        return False


def internet_check_ok() -> bool:
    for url in _INTERNET_HTTP_URLS:
        try:
            with urlopen(url, timeout=TIMEOUT) as resp:
                if resp.status in (200, 204):
                    logging.info("Internet HTTP check OK: %s", url)
                    return True
        except Exception:  # noqa: S110
            pass

    for ip in _INTERNET_PING_IPS:
        if ping_host(ip, count=1, timeout=TIMEOUT):
            logging.info("Internet ping fallback OK: %s", ip)
            return True

    logging.warning("Internet connectivity FAILED")
    return False


# ================ SHELLY ==================


def shelly_power_cycle(output_id: int, off_seconds: int) -> bool:
    """Switch the output off and let the Shelly turn it back on after off_seconds.

    Using the Shelly-side ``toggle_after`` timer means the restore does not depend
    on the main Pi still being able to reach the Shelly while the output is off.
    """
    query = urlencode({"id": output_id, "on": "false", "toggle_after": off_seconds})
    url = f"http://{SHELLY_IP}/rpc/Switch.Set?{query}"
    req = Request(url, method="GET")
    try:
        with urlopen(req, timeout=TIMEOUT) as resp:
            return resp.status == 200
    except Exception as exc:
        logging.error("Shelly Switch.Set failed (id=%s): %s", output_id, exc)
        return False


# ================ FAIL COUNTERS ===========


def update_fail_counter(ok: bool, fail_file: Path, label: str, log_result: bool = True) -> int:
    if ok:
        if log_result:
            logging.info("%s check OK", label)
        write_int(fail_file, 0)
        return 0

    fails = read_int(fail_file, 0) + 1
    if log_result:
        logging.warning("%s check FAILED (%s/%s)", label, fails, MAX_FAILS)
    write_int(fail_file, fails)
    return fails


# ================ REBOOT GUARD ============


@dataclass(frozen=True)
class RebootGuard:
    cooldown_seconds: int
    max_reboots_per_day: int

    def _read_daily(self, daily_file: Path) -> tuple[str, int]:
        today = dt.date.today().isoformat()
        raw = read_text(daily_file, "")

        if not raw:
            return today, 0

        parts = raw.split()
        if len(parts) != 2:
            return today, 0

        day, count_s = parts[0], parts[1]
        try:
            count = int(count_s)
        except Exception:
            count = 0

        if day != today:
            return today, 0

        return day, count

    def can_reboot(self, now_ts: int, last_reboot_file: Path, daily_file: Path, label: str) -> bool:
        last_ts = read_int(last_reboot_file, 0)
        if last_ts and (now_ts - last_ts) < self.cooldown_seconds:
            remaining = self.cooldown_seconds - (now_ts - last_ts)
            logging.warning("%s: reboot blocked by cooldown, remaining_seconds=%s", label, remaining)
            return False

        day, count = self._read_daily(daily_file)
        if count >= self.max_reboots_per_day:
            logging.warning(
                "%s: reboot blocked by daily limit, count=%s, limit=%s",
                label,
                count,
                self.max_reboots_per_day,
            )
            write_text(daily_file, f"{day} {count}")
            return False

        return True

    def record_reboot(self, now_ts: int, last_reboot_file: Path, daily_file: Path) -> None:
        write_int(last_reboot_file, now_ts)

        today = dt.date.today().isoformat()
        day, count = self._read_daily(daily_file)
        if day != today:
            day, count = today, 0

        count += 1
        write_text(daily_file, f"{day} {count}")


def power_cycle(output_id: int, label: str, last_file: Path, daily_file: Path, guard: RebootGuard) -> bool:
    now_ts = int(time.time())

    if not guard.can_reboot(now_ts, last_file, daily_file, label):
        return False

    # Record the attempt before issuing it: output 0 may carry the router rail,
    # so the Shelly can apply the cut and drop the network before we read the
    # HTTP response. Counting up-front keeps the cooldown / daily limit enforced
    # and prevents runaway cycling. The Shelly-side toggle_after timer restores
    # the output regardless of whether the response reaches us.
    guard.record_reboot(now_ts, last_file, daily_file)

    logging.warning("%s: power cycle triggered (Shelly output %s, off for %ss)", label, output_id, POWER_OFF_TIME)
    if shelly_power_cycle(output_id, POWER_OFF_TIME):
        logging.info("%s: power cycle scheduled, output back on after %ss", label, POWER_OFF_TIME)
    else:
        logging.error("%s: Shelly power cycle request not confirmed (may still have applied)", label)
    return True


# ================= MAIN ===================


def main() -> None:
    guard = RebootGuard(
        cooldown_seconds=COOLDOWN_SECONDS,
        max_reboots_per_day=MAX_REBOOTS_PER_DAY,
    )

    reboot = False

    internet_ok = internet_check_ok()
    internet_fails = update_fail_counter(internet_ok, FAIL_INTERNET_FILE, "Internet")
    if internet_fails >= MAX_FAILS:
        reboot = True

    cam_results: list[tuple[str, bool, int]] = []
    for ip in CAM_IPS:
        ok = ping_host(ip)
        fails = update_fail_counter(ok, FAIL_CAM_FILES[ip], f"Camera {ip}", log_result=False)
        cam_results.append((ip, ok, fails))
        if fails >= MAX_FAILS:
            reboot = True

    if cam_results:
        parts = []
        any_failed = False
        for ip, ok, fails in cam_results:
            if ok:
                parts.append(f"{ip} OK")
            else:
                parts.append(f"{ip} FAILED ({fails}/{MAX_FAILS})")
                any_failed = True

        summary = "Cameras check: " + ", ".join(parts)
        if any_failed:
            logging.warning(summary)
        else:
            logging.info(summary)

    if reboot and power_cycle(SHELLY_OUTPUT_ID, "Cameras / Router 12V", LAST_REBOOT_FILE, DAILY_REBOOT_FILE, guard):
        for ip in CAM_IPS:
            write_int(FAIL_CAM_FILES[ip], 0)
        write_int(FAIL_INTERNET_FILE, 0)


if __name__ == "__main__":
    main()
