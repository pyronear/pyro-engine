#!/usr/bin/env python3
"""
collect.py - Periodic power data collection from a Shelly Gen2 device (local API).

Queries the Switch.GetStatus endpoint for each channel and appends the results
to a CSV file. Designed to be run periodically via cron on a Raspberry Pi.

Compatible devices (any Gen2 Shelly with Switch PM component):
  - Shelly 1 Mini Gen 3  (1 channel)
  - Shelly Pro 2PM       (2 channels)

Usage:
  python collect.py --host 192.168.1.42
  python collect.py --host 192.168.1.42 --channels 2 --output /data/shelly_data.csv

Cron example (every 10 minutes):
  */10 * * * * /usr/bin/python3 /path/to/collect.py --host 192.168.1.42 --channels 2 --output /data/shelly_data.csv >> /data/collect.log 2>&1
"""

import argparse
import csv
import json
import os
import sys
import urllib.request
from datetime import datetime

# - Constants ---------------------------------

# HTTP request timeout in seconds; keep short to avoid blocking cron jobs
TIMEOUT = 5

# Default output file path (can be overridden with --output)
DEFAULT_OUTPUT = "shelly_data.csv"

# Human-readable names for each channel, used as labels in the CSV.
# Edit these to match your physical setup (e.g. "rpi", "router_switch").
CHANNEL_NAMES = {
    0: "channel_0",
    1: "channel_1",
}

# CSV columns - order matters, must stay consistent across runs
CSV_FIELDNAMES = [
    "timestamp", "channel_id", "channel_name",
    "output", "power_w", "voltage_v", "current_a", "pf", "freq_hz",
    "energy_wh", "energy_by_minute_wh", "temperature_c",
]

# ---------------------------------------


def fetch_switch_status(host: str, channel_id: int) -> dict:
    """Query the Switch.GetStatus RPC endpoint for a single channel.

    Sends an HTTP GET request to the Shelly local API and returns the parsed
    JSON response. Returns an empty dict on any network or parsing error so
    that the caller can safely skip the channel without crashing.

    Args:
        host:       IP address or hostname of the Shelly device (e.g. "192.168.1.42").
        channel_id: Channel index to query (0 for the first channel, 1 for the second).

    Returns:
        Parsed JSON response as a dict, or {} on failure.
    """
    url = f"http://{host}/rpc/Switch.GetStatus?id={channel_id}"
    try:
        with urllib.request.urlopen(url, timeout=TIMEOUT) as resp:
            return json.loads(resp.read())
    except Exception as e:
        print(f"[ERROR] Channel {channel_id} - {e}", file=sys.stderr)
        return {}


def extract_metrics(status: dict, channel_id: int, channel_name: str) -> dict:
    """Build a flat metrics dict from a Switch.GetStatus response.

    Extracts the fields relevant for power monitoring and adds a timestamp.
    Missing fields (e.g. on devices without full PM support) are stored as None.

    Args:
        status:       Raw dict returned by fetch_switch_status().
        channel_id:   Channel index (int), stored as-is for reference.
        channel_name: Human-readable channel label (from CHANNEL_NAMES).

    Returns:
        A flat dict with keys matching CSV_FIELDNAMES.
    """
    return {
        "timestamp":   datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "channel_id":  channel_id,
        "channel_name": channel_name,
        # Relay output state: True = ON, False = OFF
        "output":      status.get("output"),
        # Instantaneous active power in Watts
        "power_w":     status.get("apower"),
        # RMS voltage in Volts
        "voltage_v":   status.get("voltage"),
        # RMS current in Amperes
        "current_a":   status.get("current"),
        # Power factor (dimensionless, 0.0-1.0)
        "pf":          status.get("pf"),
        # AC frequency in Hz
        "freq_hz":     status.get("freq"),
        # Cumulative energy counter in Wh (resets on Switch.ResetCounters)
        "energy_wh":   status.get("aenergy", {}).get("total"),
        # Energy consumed per minute over the last 3 minutes (JSON list of floats, Wh)
        "energy_by_minute_wh": json.dumps(
            status.get("aenergy", {}).get("by_minute", [])
        ),
        # Internal device temperature in Celsius (thermal protection monitoring)
        "temperature_c": status.get("temperature", {}).get("tC"),
    }


def append_to_csv(filepath: str, rows: list[dict]) -> None:
    """Append metric rows to a CSV file, creating it with a header if needed.

    Uses 'append' mode so it is safe to call on every cron run without
    overwriting existing data. The header is written only when the file
    does not yet exist.

    Args:
        filepath: Path to the CSV output file.
        rows:     List of metric dicts, each matching CSV_FIELDNAMES.
    """
    file_exists = os.path.isfile(filepath)
    with open(filepath, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    """Parse and validate command-line arguments.

    Returns:
        Parsed argument namespace. Exits with a help message on missing/invalid args.
    """
    parser = argparse.ArgumentParser(
        prog="collect.py",
        description=(
            "Collect power metrics from a Shelly Gen2 device (local API) and "
            "append them to a CSV file. Intended to run periodically via cron."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python collect.py --host 192.168.1.42\n"
            "  python collect.py --host 192.168.1.42 --channels 2 --output /data/shelly_data.csv\n\n"
            "cron (every 10 min):\n"
            "  */10 * * * * python3 /path/to/collect.py --host 192.168.1.42 "
            "--channels 2 --output /data/shelly_data.csv >> /data/collect.log 2>&1"
        ),
    )
    parser.add_argument(
        "--host",
        required=True,
        metavar="IP",
        help="IP address or hostname of the Shelly device (required). "
             "Find it in the Shelly app or your router's DHCP table.",
    )
    parser.add_argument(
        "--channels",
        type=int,
        default=2,
        choices=[1, 2],
        metavar="{1,2}",
        help="Number of channels to query. "
             "Use 1 for Shelly 1 Mini Gen 3, 2 for Shelly Pro 2PM (default: 2).",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        metavar="FILE",
        help=f"Path to the CSV output file (default: {DEFAULT_OUTPUT}).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    rows = []
    for ch_id in range(args.channels):
        name = CHANNEL_NAMES.get(ch_id, f"channel_{ch_id}")
        status = fetch_switch_status(args.host, ch_id)
        if status:
            row = extract_metrics(status, ch_id, name)
            rows.append(row)
            print(
                f"[OK] {row['timestamp']} | {name} | "
                f"{row['power_w']} W | {row['energy_wh']} Wh"
            )
        else:
            print(f"[SKIP] Channel {ch_id} - no data received.", file=sys.stderr)

    if rows:
        append_to_csv(args.output, rows)
        print(f"-> {len(rows)} row(s) appended to {args.output}")
    else:
        print("[WARN] No data collected.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
