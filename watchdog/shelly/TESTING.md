# Testing a station

Once the station is wired and both watchdogs are installed (see
`device/README.md`), run the commissioning script. It works **from the main
Pi** (on-site) or **from your PC** (bench check after setup, or remotely
over the VPN) — the mode is auto-detected, overridable with `--mode pi|pc`.

```bash
python3 watchdog/shelly/check_station.py
```

It checks the whole chain:

- Shelly reachable, `pi_watchdog` script enabled on boot and running
- outputs 0 and 1 ON, with `initial_state: "on"` (survives a Shelly reboot)
- hardening applied (cloud / AP / BLE off, eco mode) — warnings only
- the Shelly itself fetches the Pi `/health` URL configured in the uploaded
  `watchdog.js` and gets a 200 (the exact check the watchdog performs)
- Pi `/health` answers, cameras answer ping
- on the Pi only: cron entry installed, station internet check
  (skipped from a PC — your machine's internet says nothing about the station's)

Station IPs are fixed and baked in as defaults:

| Device | IP | Override |
| --- | --- | --- |
| Shelly | `192.168.1.97` | `--shelly-ip` |
| Main Pi | `192.168.1.99` (health on `:8081/health`) | `--pi-url` |
| Cameras | `192.168.1.11`, `192.168.1.12` | `--cam-ips` |

You normally never need the override flags — they exist for non-standard
stations. The Pi health URL actually tested is the one found in the
`watchdog.js` uploaded on the Shelly; the script warns if it differs from
the standard one.

Every run also appends its report to `~/check_station.log` (timestamped,
`--log-file` to override) on the machine running it — useful to review a
`--cycle-pi` run after reconnecting to the Pi.

## Wiring tests (cut power for real)

```bash
python3 watchdog/shelly/check_station.py --cycle-cameras   # Pi or PC
python3 watchdog/shelly/check_station.py --cycle-pi        # Pi or PC
```

- `--cycle-cameras` power-cycles output 0 (cameras / router 12V) with a
  Shelly-side `toggle_after` restore, verifies the first camera actually
  drops, then waits up to 180 s for it to come back. If the camera keeps
  answering during the cut, it is not wired on output 0.
- `--cycle-pi` power-cycles output 1 (main Pi).
  - From a PC: verifies `/health` drops, then waits up to 300 s for the Pi
    to reboot and answer again.
  - From the Pi: cuts its own power, so your SSH session drops — the Shelly
    restores the output after 15 s and the Pi reboots. Re-run
    `check_station.py` after reboot: it detects whether the reboot really
    happened and gives the wiring verdict. Handy during installation: run
    it remotely while the installer is on-site in case something goes wrong.

## Manual test (first station at least)

Trigger the Shelly watchdog for real: stop the `/health` service and wait
3 check intervals (~30 min) — the Shelly must cut outputs 0 and 1 for 2 min.
To speed it up, temporarily re-upload `watchdog.js` with a short
`CHECK_INTERVAL_MS`, then restore it with `update_shelly_watchdog.sh`.

Exit code is non-zero if any check fails.
