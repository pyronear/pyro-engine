#!/usr/bin/env python3
"""
Watchdog for the main Pi that pings the Pi Zero and power-cycles its relay
after repeated failures, with cooldown and daily limits.

Cron setup (every 10 minutes at :05):
  1) Edit crontab:  crontab -e
  2) Add the line:
     5,15,25,35,45,55 * * * * /usr/bin/python3 /home/pi/pyro-engine/watchdog/main_pi/watchdog.py >> /home/pi/watchdog_main.log 2>&1

Adjust the paths to match where this repo lives on the Pi.
"""

import datetime as dt
import logging
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import RPi.GPIO as GPIO

# ================= CONFIG =================

RELAY_PIZERO = 16
PIZERO_IP = "192.168.1.62"

PING_COUNT = 2
TIMEOUT = 2  # seconds, used for ping timeout

MAX_FAILS = 3
POWER_OFF_TIME = 15

COOLDOWN_SECONDS = 30 * 60
MAX_REBOOTS_PER_DAY = 3

STATE_DIR = Path("/tmp")
LOG_FILE = Path("/home/pi/watchdog_main.log")

FAIL_PIZERO_FILE = STATE_DIR / "fail_pizero"
LAST_REBOOT_FILE = STATE_DIR / "last_reboot_pizero"
DAILY_REBOOT_FILE = STATE_DIR / "daily_reboots_pizero"

# ================ LOGGING =================

logging.basicConfig(
    filename=str(LOG_FILE),
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ================ GPIO ====================

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

GPIO.setup(RELAY_PIZERO, GPIO.OUT, initial=GPIO.HIGH)

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

def ping_host(ip: str) -> bool:
    try:
        subprocess.check_output(
            ["ping", "-c", str(PING_COUNT), "-W", str(TIMEOUT), ip],
            stderr=subprocess.DEVNULL,
        )
        return True
    except subprocess.CalledProcessError:
        return False

# ================ FAIL COUNTERS ===========

def update_fail_counter(ok: bool, fail_file: Path, label: str) -> int:
    if ok:
        logging.info("%s check OK", label)
        write_int(fail_file, 0)
        return 0

    fails = read_int(fail_file, 0) + 1
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

def power_cycle(relay_gpio: int, label: str, last_file: Path, daily_file: Path, guard: RebootGuard) -> None:
    now_ts = int(time.time())

    if not guard.can_reboot(now_ts, last_file, daily_file, label):
        return

    logging.warning("%s: power cycle triggered", label)
    GPIO.output(relay_gpio, GPIO.LOW)
    time.sleep(POWER_OFF_TIME)
    GPIO.output(relay_gpio, GPIO.HIGH)
    logging.info("%s: power restored", label)

    guard.record_reboot(now_ts, last_file, daily_file)

# ================= MAIN ===================

def main() -> None:
    guard = RebootGuard(
        cooldown_seconds=COOLDOWN_SECONDS,
        max_reboots_per_day=MAX_REBOOTS_PER_DAY,
    )

    fails = update_fail_counter(ping_host(PIZERO_IP), FAIL_PIZERO_FILE, "Pi Zero")
    if fails >= MAX_FAILS:
        power_cycle(RELAY_PIZERO, "Pi Zero", LAST_REBOOT_FILE, DAILY_REBOOT_FILE, guard)
        write_int(FAIL_PIZERO_FILE, 0)

if __name__ == "__main__":
    try:
        main()
    finally:
        GPIO.cleanup()
