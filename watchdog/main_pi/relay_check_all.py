#!/usr/bin/env python3
"""
End-to-end relay test for the full watchdog setup (main Pi + Pi Zero).

Run from the main Pi. The script:
  1) runs watchdog/main_pi/relay_check.py locally, which power-cycles the
     Pi Zero via the main Pi's relay (RELAY_PIZERO),
  2) on success, SSHes to the Pi Zero and launches
     watchdog/pi_zero/relay_check.py detached. That second test cuts power
     to the main Pi via the Pi Zero's relay, which kills the SSH session
     and this orchestrator. The Pi Zero keeps running the test, restores
     power, and the main Pi reboots on its own.

After the main Pi comes back, reconnect and inspect the Pi Zero log:
  ssh pi@<PIZERO_IP> 'cat /tmp/relay_check.log'

Install dependencies (once per Pi):
  sudo apt install python3-rpi-lgpio

Run on the main Pi:
  python3 /home/pi/pyro-engine/watchdog/main_pi/relay_check_all.py

Exits non-zero if the main_pi step or the ssh launch fails. The Pi Zero
step is fire-and-forget; verify it via the log file on the Pi Zero.
"""

import subprocess
import sys
from pathlib import Path

# ================= CONFIG =================

_HERE = Path(__file__).resolve().parent
MAIN_PI_SCRIPT = _HERE / "relay_check.py"
PI_ZERO_SCRIPT = "/home/pi/pyro-engine/watchdog/pi_zero/relay_check.py"
PI_ZERO_LOG = "/tmp/relay_check.log"

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
PIZERO_USER: str = _env.get("PIZERO_USER", "pi")


# ================== MAIN ==================


def run_main_pi_test() -> int:
    print(f"=== Running main Pi relay test: {MAIN_PI_SCRIPT} ===", flush=True)
    return subprocess.call([sys.executable, str(MAIN_PI_SCRIPT)])


def launch_pi_zero_test() -> int:
    target = f"{PIZERO_USER}@{PIZERO_IP}"
    print(f"\n=== Launching detached Pi Zero relay test on {target} ===", flush=True)
    remote_cmd = f"nohup python3 {PI_ZERO_SCRIPT} > {PI_ZERO_LOG} 2>&1 < /dev/null & disown; exit"
    return subprocess.call(["ssh", target, remote_cmd])


def main() -> int:
    rc = run_main_pi_test()
    if rc != 0:
        print(f"\nmain_pi relay_check failed (exit {rc}); skipping Pi Zero test.", flush=True)
        return rc

    rc = launch_pi_zero_test()
    if rc != 0:
        print(f"\nFailed to launch Pi Zero relay_check (ssh exit {rc}).", flush=True)
        return rc

    print(
        "\nPi Zero relay test launched in background.\n"
        "The main Pi will lose power when the 'main' relay is tested and will reboot.\n"
        "Once it is back up, check the log on the Pi Zero with:\n"
        f"  ssh {PIZERO_USER}@{PIZERO_IP} 'cat {PI_ZERO_LOG}'",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
